"""Embed dispatch: route a source + spec into embedded conformers (the machinery behind `pipeline.embed`).

Turns SMILES / .xyz / Mol / metal `Isomer` + a constraint spec into an `Ensemble` (or an `EnsembleSet` of
candidates): parse -> (discover NCI -> constrain) -> the core seam -> `Ensemble`. Owns the source
normalisation, the metal carbon-surrogate swap, and the isomer / template / auto-NCI routing; every embed
here goes through the one core seam `rxembed.embed.seed_conformers` (encounter bounds, the Kabsch
graft, the substrate fold). The package root exports the user-facing objects constructed here.
"""

from __future__ import annotations

import logging
import os

from rdkit import Chem

import rxembed.metal_coordination as _cbuild
import rxembed.metal_core as _metal
import rxembed.metal_distance as _distance
import rxembed.metal_isomers as _kiso
import rxembed.metal_polyhedron as _poly
from rxembed.constraints import add_distance, compose, resolve_atom, resolve_core
from rxembed.constraints import template_to_fix as _core_template_to_fix
from rxembed.embed import (  # the module, not the `embed` function the package root re-exports
    fold_substrate,
    seed_conformers,
)
from rxembed.metal_smiles import parse_smiles
from rxembed.stereo import enumerate_unassigned

from . import nci as _nci
from . import stereo_check as _stereo
from .ensemble import Ensemble, EnsembleSet
from .perceive import read_xyz

logger = logging.getLogger("rxembed")

_TEMPLATE_LEN = 2  # template= is (reference, mapping)
_STEREO_MODES = {"racemic", "separate", "free", "preserve", "all", "invert"}
_STEREO_KINDS = {"point", "ez", "axial", "planar", "helical", "default"}
_STEREO_FILTERS = {"free", "preserve", "invert"}


def _validate_stereo(stereo):
    """Reject unknown global or per-kind stereo modes at every public door."""
    if isinstance(stereo, str) and stereo in _STEREO_MODES:
        return
    if (
        isinstance(stereo, dict)
        and all(kind in _STEREO_KINDS for kind in stereo)
        and all(mode in ("free", "preserve", "invert") for mode in stereo.values())
    ):
        return
    raise ValueError(
        f"unknown stereo mode {stereo!r}; use one of {sorted(_STEREO_MODES)} or "
        f"{{kind: mode}} with modes {sorted(_STEREO_FILTERS)}"
    )


def _normalize(source, charge=0):
    """Normalise SMILES / an .xyz path / an RDKit Mol to ``(mol with Hs, has_geometry)``.

    `charge` is used for xyz bond perception (a charged metal complex / ion).
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path, not just str
    if isinstance(source, Chem.Mol):
        mol = source
    elif isinstance(source, str) and source.lower().endswith(".xyz"):
        mol = read_xyz(source, charge)
        if mol is None:
            raise ValueError(f"could not read {source}")
    else:
        mol = parse_smiles(source)
    has_geom = mol.GetNumConformers() > 0
    return Chem.AddHs(mol, addCoords=has_geom), has_geom


def _nci_windows(contacts):
    """Fold NCI ``contacts`` into ``(distances, angles)`` windows (soft, releasable by ``mc(explore=)``).

    `contacts` may be a single Contact, a list of them, a ``{label: Contact}`` dict (uses all), or a raw
    ``{(i,j):(lo,hi)}`` distance dict (indices, already windowed).
    """
    distances, angles = {}, {}
    if contacts is None:
        return distances, angles
    items: list[_nci.Contact] = []
    if isinstance(contacts, _nci.Contact):
        items = [contacts]
    elif isinstance(contacts, dict):
        vals = [v for v in contacts.values() if isinstance(v, _nci.Contact)]
        if vals and len(vals) != len(contacts):  # {label: Contact} must be ALL Contacts
            raise TypeError("contacts: dict mixes Contact and non-Contact values")
        if vals:
            items = vals
        else:
            distances.update(contacts)  # a raw {(i,j):(lo,hi)} distance dict
    else:
        items = [c for c in contacts if isinstance(c, _nci.Contact)]
        if len(items) != len(list(contacts)):
            raise TypeError("contacts: list must contain Contact objects (from rx.nci_candidates)")
    for ct in items:
        distances.update(ct.distances)
        angles.update(ct.angles)
    return distances, angles


def _add_soft(cons, distances, angles):
    """Add soft (releasable) distance/angle windows to `cons`, recording them in the ``contacts`` provenance.

    Used for NCI ``contacts=`` grips folded on top of the resolved ``fix``/``constrain`` core. ``mc(explore=)``
    releases exactly these (and any ``constrain=`` windows) while structural holds stay put. Explicit wins:
    (DESIGN §5.2): a soft window that lands on a pair the user already pinned with a ``fix`` number (a
    a structural distance or angle not already soft is dropped, so the rigid fix is neither overwritten nor made
    releasable.
    """
    dk, ak = set(cons.contacts[0]), set(cons.contacts[1])
    struct_d = {k for k in cons.distances if k not in dk}  # fix numbers / frozen-core shape: non-releasable
    struct_a = {k for k in cons.angles if k not in ak}
    for (i, j), (lo, hi) in distances.items():
        key = (min(i, j), max(i, j))
        if key in struct_d:  # a user fix on this pair overrides a soft grip, so leave it rigid
            continue
        add_distance(cons.distances, i, j, lo, hi)
        dk.add(key)
    for akey, val in angles.items():
        key = tuple(akey)
        if key in struct_a:
            continue
        cons.angles[key] = val
        ak.add(key)
    cons.contacts = (frozenset(dk), frozenset(ak))


def _coordination_choices(iso, coordinate, nvac):
    """Resolve ``coordinate=`` to the donor set(s) to seat: one choice, or several for an ambiguous SMARTS."""
    if coordinate is None:
        return [None]
    if coordinate == "auto":
        atoms = _metal.lone_pair_donors(iso.mol, iso.metal, exclude=iso.donors)
        if len(atoms) > nvac:
            logger.info(
                "coordinate=auto: %d candidate donor(s) for %d vacant site(s); using %d "
                "(name them explicitly to choose)",
                len(atoms),
                nvac,
                nvac,
            )
        return [atoms[:nvac]]
    if isinstance(coordinate, (list, tuple)):  # explicit: one spec per vacancy
        return [[resolve_atom(iso.mol, s) for s in coordinate]]
    if isinstance(coordinate, str):  # a SMARTS, which may match several atoms
        ms = [m[0] for m in iso.mol.GetSubstructMatches(Chem.MolFromSmarts(coordinate))]
        if not ms:
            raise ValueError(f"coordinate={coordinate!r} matched no atoms")
        if len(ms) == 1:
            return [[ms[0]]]
        if nvac == 1:  # ambiguous + 1 pocket -> one mode per donor
            logger.info(
                "coordinate=%r matched %d atoms; embedding %d modes, one per donor",
                coordinate,
                len(ms),
                len(ms),
            )
            return [[a] for a in ms]
        raise ValueError(  # ambiguous + several pockets -> can't map
            f"coordinate={coordinate!r} matched {len(ms)} atoms for {nvac} vacant site(s); "
            f"name them explicitly as a list, e.g. ['[OX1]', '[OX2]']"
        )
    return [[resolve_atom(iso.mol, coordinate)]]  # an int index


def _embed_isomer(iso, *, coordinate, contacts, fix, constrain, n, seed, knowledge, keep_input=False):
    """Embed a metal `Isomer`, optionally binding a substrate; yield one `Ensemble` per binding candidate.

    Usually one, but several when ``coordinate=`` is a SMARTS matching several donor atoms (one candidate per
    donor). Each carries a ``.tag``; `keep_input` adds the Mol's input conformer as a seed (the retain-input path
    the embed seam then relaxes it into its windows, so it is not returned pristine).
    """
    base, graft_ref = iso.cons.copy(), {}
    if contacts is not None or fix or constrain:  # a substrate bound via fix / constrain / NCI contacts
        has_geom = iso.mol.GetNumConformers() > 0
        sub, graft_ref = resolve_core(iso.mol, fix=fix, constrain=constrain, has_geometry=has_geom)
        _add_soft(sub, *_nci_windows(contacts))  # only the NCI half is ours; the core's `fold_substrate`
        base = fold_substrate(base, sub, graft_ref)  # does the sphere-preserving compose (and refuses a graft
        #                                             that would overwrite the arrangement)
    choices = _coordination_choices(iso, coordinate, iso.vertices.count(_metal.VACANT))

    for atoms in choices:
        mol = Chem.Mol(iso.mol)  # own copy so candidates don't share conformers
        input_conf = Chem.Conformer(mol.GetConformer()) if (keep_input and mol.GetNumConformers()) else None
        cons = base.copy()
        # identity is geometric: arrangement (slot map) + metal chirality. `label` (cis/trans/mer/fac) is
        # kept only as a coarse, sometimes-wrong convenience tag, never the thing you select on.
        tag = {
            "geometry": iso.geometry,
            "arrangement": _kiso.arrangement(iso),
            "chirality": iso.chirality,
            "label": iso.label,
        }
        if iso.stereo_label:  # a ligand stereoisomer from the rx.metal coordination x stereo load-in
            tag["stereo"] = iso.stereo_label
        if atoms is not None:
            # `compose`, never a field-by-field pick: the seated donor's constraints are its window AND the DG
            # relief of the surrogate floor its neighbours inherit, and a cherry-pick silently drops the latter.
            cons = compose(cons, _cbuild.coordinate(iso, atoms))
            if len(choices) > 1:
                tag["coordinate"] = atoms[0]
        try:  # the core seam: free fragments tethered, donor hand held, embed, exact core grafted back
            mol, ids = seed_conformers(mol, cons, iso, n, seed=seed, knowledge=knowledge, graft_ref=graft_ref)
        except RuntimeError as e:  # triangle smoothing -> infeasible bounds
            raise ValueError(
                f"could not embed {iso.geometry} {_kiso.arrangement(iso)}"
                + (f" with atom(s) {atoms} coordinated" if atoms else "")
                + f": the coordination + the fix=/constrain=/template= spec are geometrically infeasible "
                f"(e.g. a substrate that can't chelate the requested vertices, or a grafted core that "
                f"doesn't fit the sphere). [{e}]"
            ) from e
        if input_conf is not None:  # ETKDG cleared confs; re-add the input as a seed
            ids = [mol.AddConformer(input_conf, assignId=True), *ids]
        logger.info(
            "embed[%s: %s%s%s]: %d seeds%s",  # name-agnostic identity: arrangement (+ chirality), not cis/trans
            iso.geometry,
            _kiso.arrangement(iso),
            f" {iso.chirality}" if iso.chirality else "",
            f" coord@{atoms}" if atoms else "",
            len(ids),
            " (incl. input geometry)" if input_conf else "",
        )
        ens = Ensemble(mol, ids, cons, iso, seed=int(seed))  # keep the seed so a stated metal hand can re-seed
        if atoms:
            ens.sphere[iso.metal] = [*iso.donors, *atoms]
        if ids:  # the labile-donor hand at the uniform initial embed; minimize() culls later inversions
            ens._donor_hand = {
                d: _metal.donor_chirality_sign(mol, ids[0], d) for d in _metal._labile_donors(mol, iso.donors)
            }
        ens.tag = tag
        yield ens


def _auto_contacts_embed(source, *, metal, fix, constrain, coordinate, charge, n, seed, knowledge):
    """Resolve ``contacts='auto'``: discover the inter-fragment NCI binding modes and conf-search each.

    One `Ensemble` if there is a single mode, else an `EnsembleSet` (`.tag['nci']` = the mode).
    A mode that cannot be embedded (geometrically infeasible clamp) is skipped with a warning, not fatal.
    """
    disc = source.mol if isinstance(source, _kiso.Isomer) else _normalize(source, charge)[0]
    try:
        modes = _nci.auto_binding_modes(disc, seed=seed)
    except Exception as err:
        raise ValueError(
            f"contacts='auto' could not discover binding modes ({type(err).__name__}: {err}); "
            "pass a geometry to discover from (a multi-fragment SMILES or an .xyz), or give "
            "contacts= explicitly"
        ) from err
    common = {
        "metal": metal,
        "fix": fix,
        "constrain": constrain,
        "coordinate": coordinate,
        "charge": charge,
        "n": n,
        "seed": seed,
        "knowledge": knowledge,
    }
    if not modes:
        logger.info("contacts='auto': no inter-fragment NCI binding mode detected -> plain embed")
        return _embed_dispatch(source, contacts=None, **common)
    logger.info("contacts='auto': %d binding modes", len(modes))
    out = EnsembleSet()
    for label, contact in modes.items():
        try:
            res = _embed_dispatch(source, contacts=contact, **common)
        except (ValueError, RuntimeError) as err:  # infeasible bounds (triangle smoothing) too
            logger.warning("contacts='auto': mode '%s' skipped (could not embed): %s", label, err)
            continue
        for ens in res if isinstance(res, EnsembleSet) else [res]:
            if not ens.ids:  # a 0-conformer embed (e.g. an orphaned anchor)
                logger.warning("contacts='auto': mode '%s' produced no conformers; skipped", label)
                continue
            ens.tag = {**(ens.tag or {}), "nci": label}
            out.append(ens)
    if not out:
        raise ValueError("contacts='auto': none of the discovered binding modes could be embedded")
    return out[0] if len(out) == 1 else out


def _attach_stereo(result, source, charge, stereo):
    """Tag the embedded ensemble(s) with the chirality filter ``(spec, reference signature)`` for `minimize`.

    The default modes engage ``'preserve'`` only when the input geometry carries chirality
    the embed can't keep (planar/axial/helical); a SMILES or a stereo-less input is left untouched.
    """
    if stereo == "free":
        return
    ref = source.stereo_ref if isinstance(source, _kiso.Isomer) else None
    if ref is None and not isinstance(source, _kiso.Isomer):
        try:
            mol, has_geom = _normalize(source, charge)
            ref = _stereo.signature(mol, charge=charge) if has_geom else None
        except Exception:
            ref = None
    if not ref:
        return
    if stereo in (
        "racemic",
        "separate",
    ):  # for a geometry input, preserve only a metallocene's PLANAR chirality, the part the embed cannot keep:
        # axial reads differently every rotamer on a labile bond, and point R/S is the embed's own job
        spec = {"planar": "preserve", "default": "free"} if "planar" in set(ref) else None
    else:
        spec = stereo
    if spec is None:
        return
    for ens in result if isinstance(result, EnsembleSet) else [result]:
        if isinstance(ens, Ensemble):
            ens._stereo = (spec, ref)


_STEREO_CAP = 32  # max stereoisomers embedded per source before truncating (a loud-logged safety valve)


def _stereo_expand(source, stereo, cap=_STEREO_CAP):
    """Return ``(variants, n_unassigned, total, unresolved)`` to enumerate, or ``None`` for the single-embed path.

    ``None`` when: `stereo` is not an enumerating mode (``'racemic'``/``'separate'``); the source carries a
    geometry or is a metal `Isomer` (its point stereo is 3D-perceived / the polyhedron path owns its handedness); or
    nothing is unspecified. Otherwise the source is coordinate-free (a SMILES / conformer-less Mol) with
    undefined stereocentres to expand; see `rxembed.stereo.enumerate_unassigned`.
    """
    if stereo not in ("racemic", "separate"):  # 'free' opts out; a dict/'preserve' filter-spec is a geometry input
        return None
    if isinstance(source, _kiso.Isomer):
        return None
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str) and source.lower().endswith(".xyz"):
        return None  # a geometry defines every stereocentre (AssignStereochemistryFrom3D)
    if isinstance(source, Chem.Mol):
        if source.GetNumConformers() > 0:
            return None  # ditto: a conformer defines the stereo
        mol = source
    elif isinstance(source, str):
        mol = Chem.MolFromSmiles(source)
        if mol is None:  # a genuine parse error; let _embed_dispatch raise the clear message
            return None
    else:
        return None
    if _metal.metal_index(mol) is not None:
        return None  # a metal complex: rx.metal/enumerate_isomers owns its coordination x ligand-stereo load-in
    expanded = enumerate_unassigned(mol, cap=cap)
    return None if expanded[1] == 0 else expanded


def _stereo_enumerated_embed(expanded, stereo, dispatch_kw):
    """Embed each stereoisomer variant and assemble per `stereo` mode: the racemate load-in stage.

    ``'racemic'`` folds every variant's candidate(s) into one flat `EnsembleSet`; ``'separate'`` keeps them apart
    as a ``list[EnsembleSet]`` (one per stereoisomer, uniform type across metal & organic). Each variant is
    embedded with the same effort and tagged ``stereo=<label>``; never pruned against each other (distinct species).
    """
    variants, n_unassigned, total, unresolved = expanded
    labels = ", ".join(lbl or "achiral" for _, lbl in variants)
    if total > _STEREO_CAP:  # more stereoisomers than the safety valve, so a truncated subset is embedded
        logger.warning(
            "stereo=%r: %d stereocentre(s) -> %d isomers, capped to %d (raise cap=): [%s]",
            stereo,
            n_unassigned,
            total,
            len(variants),
            labels,
        )
    else:
        logger.info(
            "stereo=%r: %d undefined stereocentre(s) -> embedding %d stereoisomer(s) [%s]",
            stereo,
            n_unassigned,
            len(variants),
            labels,
        )
    if unresolved:  # an allene/cumulene/atropisomer axis: EnumerateStereoisomers can't encode it from a flat SMILES
        logger.warning(
            "stereo=%r: %d stereo axis(es) not enumerable from a flat SMILES; one arbitrary hand each",
            stereo,
            unresolved,
        )
    groups = []
    for vmol, label in variants:
        try:
            res = _embed_dispatch(vmol, **dispatch_kw)
        except (ValueError, RuntimeError) as err:  # a single infeasible diastereomer must not abort the set
            logger.warning(
                "stereo=%r: stereoisomer [%s] could not be embedded (%s); skipped", stereo, label or "achiral", err
            )
            continue
        es = res if isinstance(res, EnsembleSet) else EnsembleSet([res])
        live = EnsembleSet()
        for ens in es:
            if not ens.ids:  # a 0-conformer embed (a strained trans-cyclooctene): drop, don't keep it
                logger.warning(
                    "stereo=%r: stereoisomer [%s] produced no conformers; skipped", stereo, label or "achiral"
                )
                continue
            ens.tag = {**(getattr(ens, "tag", None) or {}), "stereo": label}
            live.append(ens)
        if not live:
            continue
        logger.info("stereo=%r: stereoisomer [%s] -> %d candidate(s)", stereo, label or "achiral", len(live))
        groups.append(live)
    if stereo == "separate":
        return groups  # list[EnsembleSet], one per stereoisomer, uniform across metal and organic
    if not groups:
        raise ValueError("stereo enumeration: none of the stereoisomers could be embedded (infeasible bounds)")
    if len(groups) > 1:
        logger.info(
            "stereo=%r: %d stereoisomers in one EnsembleSet; do not energy-prune across them",
            stereo,
            len(groups),
        )
    flat = EnsembleSet(ens for es in groups for ens in es)  # 'racemic': fold into one candidate set
    return flat[0] if len(flat) == 1 else flat  # a lone (all-axial-collapsed) variant stays a bare Ensemble


def _template_to_fix(template, fix, own=None, target=None):
    """Fold ``template=`` into a coordinate ``fix``, extending the core resolver with the pipeline's `Ensemble`.

    The core takes a Mol, an ``.xyz`` path or an (N, 3) array; this adds an `Ensemble`, handing over the
    positions of its first tracked conformer. An optional SMARTS is matched once on the target and reference;
    an xyz or coordinate array uses an explicit index map. The template is read for coordinates and never
    perceived, which lets a hypervalent reacting core serve as a reference.

    ``own`` is the source's own coordinates, which a ``fix=[atoms]`` list beside the template is resolved
    against; the caller must read them from the *normalised* source.
    """
    if isinstance(template, (tuple, list)) and len(template) == _TEMPLATE_LEN:
        reference, mapping = template
        if isinstance(reference, Ensemble):
            if not reference.ids:
                raise ValueError(
                    "a template reference Ensemble needs a tracked conformer; this one is empty or every "
                    "conformer was discarded"
                )
            cid = reference.ids[0]
            reference = Chem.Mol(reference.mol)
            chosen = Chem.Conformer(reference.GetConformer(cid))
            reference.RemoveAllConformers()
            reference.AddConformer(chosen, assignId=False)
            template = (reference, mapping)
    return _core_template_to_fix(template, fix, own, target)


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo="racemic", lengths="auto"):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    The public `rx.metal` entry point: the two things the core enumerator does not do, namely reading
    a SMILES / ``.xyz`` path into a `Mol`, and the coordinate-derived chirality fingerprint (`stereo.signature`,
    xyzgraph) the ``stereo='preserve'`` gate compares against. See `rxembed.metal_isomers.enumerate_isomers` for
    `geometry` / `center` / `fix` / `stereo` / `lengths`. A geometry is nameable in full (``'octahedral'``) or by
    its 3-letter code (``'OCT'``), case-insensitively; the long name is what the `Isomer` stores and prints back.
    """
    _validate_stereo(stereo)
    if isinstance(mol, str):
        # A path goes through perception so `rx.metal('complex.xyz', center=...)` works and not just
        # `rx.embed`; it is needed anyway to retain a spectator metal. A SMILES goes through `parse_smiles`
        # for a clear error on a bad string rather than a cryptic `AddHs(None)`.
        if mol.lower().endswith(".xyz"):
            mol = read_xyz(mol, 0)
        else:
            mol = Chem.AddHs(parse_smiles(mol))
    ref_sig = None  # chirality fingerprint of the input geometry (real metals), letting stereo='preserve'
    if mol.GetNumConformers() > 0:  # hold a spectator's planar/axial/helical handedness
        try:
            ref_sig = _stereo.signature(mol)
        except Exception:
            ref_sig = None
    return _kiso.enumerate_isomers(mol, geometry, center, fix, stereo, stereo_ref=ref_sig, lengths=lengths)


def _dispatch_metal_source(source, *, metal, coordinate, contacts, fix, constrain, n, seed, knowledge, charge, stereo):
    """Route a ``metal=<geometry>`` / metal `Isomer` source: enumerate isomers, or embed the one chosen isomer."""
    _iso_kw = {
        "coordinate": coordinate,
        "contacts": contacts,
        "fix": fix,
        "constrain": constrain,
        "n": n,
        "seed": seed,
        "knowledge": knowledge,
    }
    if metal is None:
        results = list(_embed_isomer(source, **_iso_kw))  # a single chosen isomer
        return results[0] if len(results) == 1 else EnsembleSet(results)
    if isinstance(source, _kiso.Isomer):
        raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
    if isinstance(source, str) and source.lower().endswith(".xyz"):
        source = _normalize(source, charge)[0]  # xyz -> perceived Mol (enumerate wants a Mol/SMILES)
    stated = isinstance(source, Chem.Mol) and _kiso.stated_arrangement(source) is not None
    isos = enumerate_isomers(source, metal, stereo=stereo)
    out = EnsembleSet(e for iso in isos for e in _embed_isomer(iso, **_iso_kw))
    if not stated:
        logger.info(  # the resolved shapes, never the raw argument: a code request must name itself on screen
            "metal: %s -> %d distinct isomer(s); .summary() / .select() to pick one",
            ", ".join(_poly.describe(g) for g in (metal if isinstance(metal, (list, tuple)) else [metal])),
            len(out),
        )
    return out[0] if stated and len(out) == 1 else out


def _load_stated_arrangement(source, normalized, metal, charge):
    """Route a CXSMILES source through the metal geometry it states."""
    if metal is not None or isinstance(source, _kiso.Isomer):
        return source, normalized, metal
    normalized = normalized or _normalize(source, charge)
    stated = _kiso.stated_arrangement(normalized[0])
    return source, normalized, stated[0] if stated is not None else metal


def _embed_dispatch(
    source,
    *,
    metal=None,
    fix=None,
    constrain=None,
    template=None,
    contacts=None,
    coordinate=None,
    charge=0,
    n=None,
    seed=0xF00D,
    knowledge=True,
    stereo="racemic",
):
    """Dispatch the embed by input type and spec; see the public `embed` for documentation."""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path everywhere downstream
    normalized = None
    if (
        template is not None
    ):  # sugar: a reference core IS a coordinate fix, dissolved here before the routing so every route gets it
        # (the auto-NCI branch below took no `template` argument at all and dropped it silently). A `fix=[atoms]`
        # list beside it names the source's own coordinates, so the source is normalised first to read them out.
        if isinstance(source, _kiso.Isomer):
            own_mol = source.mol
        else:
            normalized = _normalize(source, charge)
            own_mol = normalized[0]
        own = own_mol.GetConformer().GetPositions() if own_mol.GetNumConformers() else None
        fix = _template_to_fix(template, fix, own, own_mol)
    source, normalized, metal = _load_stated_arrangement(source, normalized, metal, charge)
    routed_source = source if isinstance(source, _kiso.Isomer) else normalized[0] if normalized else source
    if coordinate is not None and metal is None and not isinstance(source, _kiso.Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__}. Use contacts=/constrain= otherwise"
        )
    if contacts == "auto":  # discover binding modes -> conf-search each
        return _auto_contacts_embed(
            routed_source,
            metal=metal,
            fix=fix,
            constrain=constrain,
            coordinate=coordinate,
            charge=charge,
            n=n,
            seed=seed,
            knowledge=knowledge,
        )
    if metal is not None or isinstance(source, _kiso.Isomer):  # the metal enumerate / isomer paths
        return _dispatch_metal_source(
            routed_source,
            metal=metal,
            coordinate=coordinate,
            contacts=contacts,
            fix=fix,
            constrain=constrain,
            n=n,
            seed=seed,
            knowledge=knowledge,
            charge=charge,
            stereo=stereo,
        )

    mol, has_geom = normalized if normalized is not None else _normalize(source, charge)
    if (
        has_geom
        and _metal.metal_index(mol) is not None
        and not (fix or constrain or contacts or coordinate or template)
    ):  # retain the input arrangement
        iso = _kiso.from_geometry(mol)
        logger.info(
            "metal: geometry input and no metal= -> retaining the input arrangement (%s: %s)",
            _poly.describe(iso.geometry),
            _kiso.arrangement(iso),
        )
        # the embed seam then relaxes this retained geometry (arrangement kept, M-donor sphere held <0.01 A), so
        # the input is not returned pristine; say so, or the retaining line reads as "returned as-is"
        logger.info("metal: the retained geometry is relaxed into its windows, not returned as-is")
        return next(
            _embed_isomer(
                iso,
                coordinate=None,
                contacts=None,
                fix=None,
                constrain=None,
                n=n,
                seed=seed,
                knowledge=knowledge,
                keep_input=True,
            )
        )
    iso = None
    metals_donors = {}
    hydrides = []
    # The metal surrogate assembly below is step for step `embed.prepare_relax`'s: read the M-donor bonds and
    # each metal's donors before the strip, surrogate every metal, record it as a polyhedron-less `Isomer`,
    # hold each sphere (`hold_shape`), add the metal FF terms. It is written out rather than calling that
    # helper because the embed path needs three arguments it has no parameter for. `has_geometry`: it
    # hardcodes True, relaxing a geometry it already has, while a SMILES here resolves at False and takes the
    # hydride window instead of a sphere hold. `ref`: it grafts the coordinate fix onto the input conformer
    # and hands nothing back, while here `ref` has to reach `seed_conformers` to graft the fresh seeds.
    # `from_surrogate(donors=)`: it names the first metal's donors, and this path deliberately does not (see
    # the seam comment below). The NCI `contacts` fold is the pipeline's alone. So unifying the two is an
    # `embed.py` change -- `prepare_relax` taking `has_geometry` and returning `ref` rather than consuming it
    # -- not a `dispatch.py` one.
    if _metal.metal_index(mol) is not None and (fix or constrain or contacts or template):
        # every M-donor bond, read before the strip, re-added as dative on the output (`connect_metal`)
        donor_bonds = [
            (n.GetIdx(), mi) for mi in _metal.metal_indices(mol) for n in mol.GetAtomWithIdx(mi).GetNeighbors()
        ]
        if has_geom:  # capture each metal's donors BEFORE the bonds
            metals_donors = {
                mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()] for mi in _metal.metal_indices(mol)
            }
        else:  # no geometry to hold the sphere from -> at least keep a modelled terminal M-H window
            hydrides = [
                (
                    mi,
                    nb.GetIdx(),
                    _distance.ml_distance(
                        mol,
                        mi,
                        nb.GetIdx(),
                        mol.GetAtomWithIdx(mi).GetAtomicNum(),
                        {n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()},
                        {},
                        hyb={},
                    ),
                )  # keep a terminal hydride bonded so an M-H NCI donor mode survives the surrogate strip
                for mi in _metal.metal_indices(mol)
                for nb in mol.GetAtomWithIdx(mi).GetNeighbors()
                if nb.GetAtomicNum() == 1 and nb.GetDegree() == 1
            ]
        mol, metals = _metal.surrogate_all_metals(mol)  # surrogate every metal (bimetallic-safe)
        iso = _kiso.from_surrogate(mol, metals, donor_bonds)  # no polyhedron here: hold_shape below holds the sphere
        logger.debug("metal complex: %d metal(s) swapped to carbon surrogate for the FF", len(metals))

    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=has_geom)
    _add_soft(
        cons, *_nci_windows(contacts)
    )  # NCI grips are soft and released by mc(explore=); fix numbers, the frozen-core shape and the sphere/M-H/
    # encounter holds are structural and never released
    user_graft = dict(ref)  # atoms to Kabsch-graft onto their exact coords (own / explicit / template)
    # a spectator metal's sphere is held intact-but-achiral: a relative all-pairs shape, deliberately not in
    # cons.frozen, because a graft would pin its handedness, which the stereo filter owns
    for mi, dons in metals_donors.items():
        _metal.hold_shape(mol, [mi, *dons], cons)
    if metals_donors and iso is not None:
        # A frozen metal has no DOF, but its bond-less carbon still fires fictitious FF terms, so it gets the
        # zero-vdW type and its floors but no pulls -- pulling a rigid shape's members tears the body. `hold_shape`
        # must run first: ff_terms reads the `cons.shapes` record it writes.
        real_z = {iso.metal: iso.real_z, **{mi: rz for mi, rz, _rq in iso.extra}}
        _distance.ff_terms(mol, cons, {mi: (real_z[mi], list(dons)) for mi, dons in metals_donors.items()})
    for mi, h, target in hydrides:
        half = _cbuild._ML_SEED_HALF_WIDTH
        add_distance(cons.distances, mi, h, target - half, target + half)

    # the core seam: free fragments tethered, donor hand held, embed, exact core grafted back. `iso` here has
    # no polyhedron (the sphere is held by `hold_shape` above) and no donor list, so the hand-hold is a no-op:
    # passed anyway, so it starts working the day this path learns its donors, rather than silently not.
    mol, ids = seed_conformers(mol, cons, iso, n, seed=seed, knowledge=knowledge, graft_ref=user_graft)
    n_frag = len(Chem.GetMolFrags(mol))
    logger.info(
        "embed: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles),
    )
    return Ensemble(mol, list(ids), cons, iso)
