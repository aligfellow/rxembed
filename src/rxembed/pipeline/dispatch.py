"""Expand pipeline inputs into candidates and run each through one embed executor.

This module adapts sources, templates, stereo, metal identities, vacant-site seating and NCI modes. Those axes
only produce candidate data. `_execute` then prepares constraints and calls the shared core seeding seam once per
candidate.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from rdkit import Chem

import rxembed.metal_core as _metal
import rxembed.metal_enumeration as _kiso
import rxembed.metal_isomer as _isomer
import rxembed.metal_polyhedron as _poly
from rxembed.constraints import Constraints, compose_soft, resolve_atom, resolve_core
from rxembed.constraints import template_to_fix as _core_template_to_fix
from rxembed.embed import (  # module functions, not the package facade
    _stereo_donor_bonds,
    prepare,
    require_seed_count,
    seed_conformers,
)
from rxembed.metal_core import VACANT
from rxembed.metal_smiles import parse_smiles
from rxembed.stereo import enumerate_unassigned

from . import nci as _nci
from . import stereo_check as _stereo
from .ensemble import Ensemble, EnsembleSet
from .perceive import read_xyz

logger = logging.getLogger("rxembed")

_TEMPLATE_LEN = 2  # template= is (reference, mapping)
_STEREO_MODES = {"unassigned", "racemic", "separate", "free", "preserve", "all", "invert"}
_STEREO_KINDS = {"point", "ez", "axial", "planar", "helical", "default"}
_STEREO_FILTERS = {"free", "preserve", "invert", "racemic"}


@dataclass
class _Candidate:
    """Carry one fully selected identity into the shared embed executor."""

    spec: object
    fix: object
    tag: dict
    keep_input: bool = False
    coordinated: tuple = ()


def _default_stereo(source, stereo):
    """Retain stereo from coordinates by default; enumerate unspecified stereo from a graph."""
    if stereo is not None:
        return stereo
    if isinstance(source, _isomer.Isomer):
        has_geometry = bool(source.mol.GetNumConformers())
    elif isinstance(source, Chem.Mol):
        has_geometry = bool(source.GetNumConformers())
    else:
        path = os.fspath(source) if isinstance(source, os.PathLike) else source
        has_geometry = isinstance(path, str) and path.lower().endswith(".xyz")
    return "all" if has_geometry else "unassigned"


def _validate_stereo(stereo):
    """Reject unknown global or per-kind stereo modes at every public door."""
    if isinstance(stereo, str) and stereo in _STEREO_MODES:
        return
    if (
        isinstance(stereo, dict)
        and all(kind in _STEREO_KINDS or re.fullmatch(r"[A-Z][a-z]?\d+", str(kind)) for kind in stereo)
        and all(mode in _STEREO_FILTERS for mode in stereo.values())
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


def _contact_constraints(mol, contacts):
    """Resolve contact inputs through the validated soft-constraint path."""
    distances, angles = {}, {}
    if contacts is None:
        return Constraints()
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
        values = list(contacts)
        items = [c for c in values if isinstance(c, _nci.Contact)]
        if len(items) != len(values):
            raise TypeError("contacts: list must contain Contact objects (from rx.nci_candidates)")
    for ct in items:
        distances.update(ct.distances)
        angles.update(ct.angles)
    if not distances and not angles:
        return Constraints()
    return resolve_core(mol, constrain={**distances, **angles}, has_geometry=bool(mol.GetNumConformers()))[0]


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


def _execute(spec, *, fix, constrain, contacts, n, seed, threads, knowledge, keep_input=False):
    """Compile and seed one candidate, returning its pipeline ensemble."""
    mol, cons, iso, graft_ref = prepare(spec, fix=fix, constrain=constrain)
    cons = compose_soft(cons, _contact_constraints(mol, contacts))
    input_conf = Chem.Conformer(mol.GetConformer()) if (keep_input and mol.GetNumConformers()) else None
    mol, ids, target = seed_conformers(
        mol,
        cons,
        iso,
        n,
        seed=seed,
        threads=threads,
        knowledge=knowledge,
        graft_ref=graft_ref,
    )
    if input_conf is not None:  # include the retained input within, not in addition to, the requested count
        input_id = mol.AddConformer(input_conf, assignId=True)
        if ids:
            mol.RemoveConformer(ids[-1])
            ids = [input_id, *ids[:-1]]
        else:
            ids = [input_id]
    require_seed_count(ids, target)
    ens = Ensemble(mol, list(ids), cons, iso, seed=int(seed))
    if iso is not None:
        for donor, metal_idx in iso.donor_bonds:
            ens.sphere.setdefault(metal_idx, []).append(donor)
        if ids:
            donors = sorted({donor for donor, _metal_idx in _stereo_donor_bonds(mol, iso)})
            ens._donor_hand = {
                d: _metal.donor_chirality_sign(
                    mol,
                    ids[0],
                    d,
                    [metal for donor, metal in iso.donor_bonds if donor == d],
                )
                for d in donors
            }
    return ens


def _expand_isomers(isomers, coordinate, fix, keep_input=False):
    """Expand selected metal identities by vacant-site seating without embedding them."""
    out = []
    for iso in isomers:
        choices = _coordination_choices(iso, coordinate, iso.vertices.count(VACANT))
        for atoms in choices:
            candidate = iso._seat_vacancies(atoms) if atoms is not None else iso
            tag = {
                "geometry": candidate.geometry,
                "arrangement": _isomer.arrangement(candidate),
                "chirality": candidate.chirality,
                "label": candidate.label,
            }
            if candidate.stereo_label:
                tag["stereo"] = candidate.stereo_label
            if atoms is not None and len(choices) > 1:
                tag["coordinate"] = atoms[0]
            out.append(_Candidate(candidate, fix, tag, keep_input=keep_input, coordinated=tuple(atoms or ())))
    return out


def _contact_modes(source, contacts, seed):
    """Return explicit contacts or discover independent automatic contact modes."""
    if contacts != "auto":
        return [(None, contacts)]
    disc = source.mol if isinstance(source, _isomer.Isomer) else source
    try:
        modes = _nci.auto_binding_modes(disc, seed=seed)
    except Exception as err:
        raise ValueError(
            f"contacts='auto' could not discover binding modes ({type(err).__name__}: {err}); "
            "pass a geometry to discover from (a multi-fragment SMILES or an .xyz), or give "
            "contacts= explicitly"
        ) from err
    if not modes:
        logger.info("contacts='auto': no inter-fragment NCI binding mode detected -> plain embed")
        return [(None, None)]
    logger.info("contacts='auto': %d binding modes", len(modes))
    return list(modes.items())


def _attach_stereo(result, source, charge, stereo):
    """Tag the embedded ensemble(s) with the chirality filter ``(spec, reference signature)`` for `minimize`.

    The default modes engage ``'preserve'`` only when the input geometry carries chirality
    the embed can't keep (planar/axial/helical); a SMILES or a stereo-less input is left untouched.
    """
    if stereo == "free":
        return
    ref = source.stereo_ref if isinstance(source, _isomer.Isomer) else None
    if ref:
        owned = {"point", "ez"}
        if source.cons.haptic:
            owned.add("planar")
        ref = {kind: labels for kind, labels in ref.items() if kind not in owned}
        stereo = "preserve"
    if ref is None and not isinstance(source, _isomer.Isomer) and source.GetNumConformers():
        try:
            ref = _stereo.signature(source, charge=charge)
        except Exception as err:
            logger.warning("stereo preservation unavailable for the input geometry: %s", err)
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
        ens._stereo = (spec, ref)


_STEREO_CAP = 32  # max stereoisomers embedded per source before truncating (a loud-logged safety valve)


def _stereo_variants(source, stereo, cap=_STEREO_CAP):
    """Return source variants and whether organic stereo was expanded.

    Geometry inputs and metal complexes remain one source. Coordinate-free organic graphs expand undefined
    stereochemistry before every other candidate axis.
    """
    if stereo not in ("unassigned", "racemic", "separate"):
        return [(source, "")], False
    if isinstance(source, _isomer.Isomer):
        return [(source, "")], False
    if source.GetNumConformers() > 0:
        return [(source, "")], False
    if _metal.metal_index(source) is not None:
        return [(source, "")], False
    expanded = enumerate_unassigned(source, cap=cap, include="all" if stereo == "racemic" else ())
    variants, n_unassigned, total, unresolved = expanded
    if n_unassigned == 0:
        return [(source, "")], False
    labels = ", ".join(lbl or "achiral" for _, lbl in variants)
    if total > cap:
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
    if unresolved:  # an allene/cumulene axis which RDKit cannot encode from a flat SMILES
        logger.warning(
            "stereo=%r: %d stereo axis(es) not enumerable from a flat SMILES; one arbitrary hand each",
            stereo,
            unresolved,
        )
    return variants, True


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


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo=None, lengths="auto"):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    The public `rx.metal` entry point: the two things the core enumerator does not do, namely reading
    a SMILES / ``.xyz`` path into a `Mol`, and the coordinate-derived chirality fingerprint (`stereo.signature`,
    xyzgraph) the ``stereo='preserve'`` gate compares against. See `rxembed.metal_enumeration.enumerate_isomers` for
    `geometry` / `center` / `fix` / `stereo` / `lengths`. A geometry is nameable in full (``'octahedral'``) or by
    its 3-letter code (``'OCT'``), case-insensitively. The `Isomer` stores the registry name and `.summary()`
    prints the compact code.
    """
    stereo = _default_stereo(mol, stereo)
    _validate_stereo(stereo)
    if isinstance(mol, str):
        # A path goes through perception so `rx.metal('complex.xyz', center=...)` works and not just
        # `rx.embed`; it is needed anyway to retain a spectator metal. A SMILES goes through `parse_smiles`
        # for a clear error on a bad string rather than a cryptic `AddHs(None)`.
        if mol.lower().endswith(".xyz"):
            mol = read_xyz(mol, 0)
        else:
            mol = parse_smiles(mol)
    _metal._reject_metal_bonds(mol)
    mol = Chem.AddHs(mol, addCoords=bool(mol.GetNumConformers()))
    ref_sig = None  # chirality fingerprint of the input geometry (real metals), letting stereo='preserve'
    if mol.GetNumConformers() > 0:  # hold a spectator's planar/axial/helical handedness
        try:
            ref_sig = _stereo.signature(mol)
        except Exception:
            ref_sig = None
    return _kiso.enumerate_isomers(mol, geometry, center, fix, stereo, stereo_ref=ref_sig, lengths=lengths)


def _source_candidates(source, *, metal, fix, coordinate, stereo):
    """Expand one normalized Mol or selected metal Isomer into molecular identities."""
    stated = False
    if metal is None and not isinstance(source, _isomer.Isomer):
        metals = _metal.metal_indices(source)
        arrangements = [_kiso.stated_arrangement(source, center=center) for center in metals]
        if len(metals) > 1 and any(arrangements):
            if not all(arrangements):
                raise ValueError("a coordinate-free multi-metal embed needs one geometry note per centre")
            isomers = _kiso.enumerate_isomers(source, center="all", stereo=stereo)
            if len(isomers) != 1:
                raise ValueError("stereo expansion produced several states; select one with rx.metal before embedding")
            source, stated = isomers[0], True
        elif len(metals) == 1 and arrangements[0] is not None:
            metal, stated = arrangements[0][0], True

    if coordinate is not None and metal is None and not isinstance(source, _isomer.Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__}. Use contacts=/constrain= otherwise"
        )

    if metal is not None or isinstance(source, _isomer.Isomer):
        if metal is None:
            isomers, pending_fix = [source], fix
        else:
            if isinstance(source, _isomer.Isomer):
                raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
            stated = stated or _kiso.stated_arrangement(source) is not None
            isomers = enumerate_isomers(source, metal, fix=fix, stereo=stereo)
            pending_fix = None  # identity enumeration compiled the fix once and stored it on every Isomer
        candidates = _expand_isomers(isomers, coordinate, pending_fix)
        if not candidates:
            raise ValueError(
                f"metal={metal!r} produced no feasible coordination identity; choose a compatible geometry, "
                "relax fix=, or call rx.metal(...) to inspect the enumeration"
            )
        if metal is not None and not stated:
            logger.info(
                "metal: %s -> %d distinct isomer(s); .summary() / .select() to pick one",
                ", ".join(_poly.describe(g) for g in (metal if isinstance(metal, (list, tuple)) else [metal])),
                len(isomers),
            )
        return source, candidates, metal is not None and not stated

    has_geometry = bool(source.GetNumConformers())
    if _metal.metal_index(source) is not None:
        if not has_geometry:
            raise ValueError("plain RDKit embedding does not model metals; pass metal=<geometry>")
        centre = "all" if len(_metal.metal_indices(source)) > 1 else None
        iso = _isomer.from_geometry(source, center=centre)
        logger.info(
            "metal: geometry input and no metal= -> retaining the input arrangement (%s: %s)",
            _poly.describe(iso.geometry),
            _isomer.arrangement(iso),
        )
        logger.info("metal: the retained geometry is relaxed into its windows, not returned as-is")
        return source, _expand_isomers([iso], None, fix, keep_input=True), False
    return source, [_Candidate(source, fix, {})], False


def _embed_mode(candidates, contact, nci_label, stereo_label, execute_kw):
    """Execute every molecular identity for one contact mode."""
    live = EnsembleSet()
    for candidate in candidates:
        tag = dict(candidate.tag)
        if stereo_label:
            tag["stereo"] = stereo_label
        if nci_label is not None:
            tag["nci"] = nci_label
        identity = tag.get("arrangement", "molecule")
        coordinated = f" with atom(s) {list(candidate.coordinated)} coordinated" if candidate.coordinated else ""
        try:
            ens = _execute(
                candidate.spec,
                fix=candidate.fix,
                contacts=contact,
                keep_input=candidate.keep_input,
                **execute_kw,
            )
        except RuntimeError as err:
            raise ValueError(
                f"could not embed {identity}{coordinated}: the coordination and constraint spec are "
                f"geometrically infeasible [{err}]"
            ) from err
        if not ens.ids:
            raise ValueError(
                f"could not embed {identity}{coordinated}: no conformer satisfied the geometry and constraints"
            )
        ens.tag = tag
        if "arrangement" in tag:
            logger.info(
                "embed[%s: %s%s%s]: %d seeds%s",
                tag["geometry"],
                tag["arrangement"],
                f" {tag['chirality']}" if tag.get("chirality") else "",
                f" coord@{list(candidate.coordinated)}" if candidate.coordinated else "",
                len(ens.ids),
                " (incl. input geometry)" if candidate.keep_input else "",
            )
        else:
            n_frag = len(Chem.GetMolFrags(ens.mol))
            count = len(ens.cons.distances) + len(ens.cons.angles) + len(ens.cons.dihedrals)
            logger.info(
                "embed: %d seeds (%d atoms, %d fragment%s, %d constraints)",
                len(ens.ids),
                ens.mol.GetNumAtoms(),
                n_frag,
                "s" if n_frag != 1 else "",
                count,
            )
        live.append(ens)
    return live


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
    threads=0,
    knowledge=True,
    stereo=None,
):
    """Expand every candidate axis, execute each candidate once, and assemble the public result."""
    stereo = _default_stereo(source, stereo)
    _validate_stereo(stereo)
    if not isinstance(source, _isomer.Isomer):
        source = _normalize(source, charge)[0]
    own_mol = source.mol if isinstance(source, _isomer.Isomer) else source
    if template is not None:
        own = own_mol.GetConformer().GetPositions() if own_mol.GetNumConformers() else None
        fix = _template_to_fix(template, fix, own, own_mol)
    variants, expanded = _stereo_variants(source, stereo)
    groups, force_set = [], False
    last_failure = None
    execute_kw = {
        "constrain": constrain,
        "n": n,
        "seed": seed,
        "threads": threads,
        "knowledge": knowledge,
    }
    for variant, stereo_label in variants:
        try:
            discovery, candidates, source_set = _source_candidates(
                variant,
                metal=metal,
                fix=fix,
                coordinate=coordinate,
                stereo=stereo,
            )
            modes = _contact_modes(discovery, contacts, seed)
            force_set = force_set or source_set
            live = EnsembleSet()
            for nci_label, contact in modes:
                try:
                    live.extend(_embed_mode(candidates, contact, nci_label, stereo_label, execute_kw))
                except ValueError as err:
                    last_failure = err
                    if nci_label is None:
                        raise
                    logger.warning("contacts='auto': mode %r skipped: %s", nci_label, err)
            if not live:
                raise ValueError("no candidate produced a conformer") from last_failure
        except (ValueError, RuntimeError) as err:
            last_failure = err
            if expanded:
                logger.warning(
                    "stereo=%r: stereoisomer [%s] could not be embedded (%s); skipped",
                    stereo,
                    stereo_label or "achiral",
                    err,
                )
                continue
            raise
        if expanded:
            logger.info(
                "stereo=%r: stereoisomer [%s] -> %d candidate(s)",
                stereo,
                stereo_label or "achiral",
                len(live),
            )
        groups.append(live)

    if not groups:
        raise ValueError(
            f"none of the expanded candidates could be embedded; last failure: {last_failure}"
        ) from last_failure
    if stereo == "separate" and expanded:
        result = groups
    else:
        if expanded and len(groups) > 1:
            logger.info(
                "stereo=%r: %d stereoisomers in one EnsembleSet; do not energy-prune across them",
                stereo,
                len(groups),
            )
        flattened = EnsembleSet(ens for group in groups for ens in group)
        result = flattened if force_set or len(flattened) != 1 else flattened[0]
    _attach_stereo(result, source, charge, stereo)
    return result
