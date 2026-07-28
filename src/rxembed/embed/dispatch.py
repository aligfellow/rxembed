"""Embed dispatch: route a source + spec into embedded conformers (the machinery behind `pipeline.embed`).

Turns SMILES / .xyz / Mol / metal `Isomer` + a constraint spec into an `Ensemble` (or an `EnsembleSet` of
candidates). Owns the source normalisation, the metal carbon-surrogate context, the vdW encounter bounds, the
Kabsch frozen-core graft, and the isomer / template / auto-NCI routing. The user-facing surface (`embed`,
`Ensemble`, `EnsembleSet`, `wrap`) lives in `rxembed.pipeline`; this module constructs those objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from rxembed import isomers as _isomers
from rxembed import stereo as _stereo
from rxembed.constraints import add_distance, resolve_atom, resolve_core
from rxembed.constraints import nci as _nci
from rxembed.inputs import _xyz_to_mol, parse_smiles
from rxembed.pipeline import Ensemble, EnsembleSet
from rxembed.rdkit_embed.constraints import coordination_builders as _cbuild
from rxembed.rdkit_embed.constraints import distance as _distance
from rxembed.rdkit_embed.constraints import metal as _metal
from rxembed.rdkit_embed.constraints.base import compose
from rxembed.rdkit_embed.embed import bounds as _embed
from rxembed.rdkit_embed.log import logger

_PT = Chem.GetPeriodicTable()
_EPS = 1e-6  # near-zero norm floor for the graft axis
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation
_XYZ_DIM = 3  # an (x, y, z) coordinate
_TEMPLATE_LEN = 2  # template= is (reference, mapping)
_MIN_FRAGS = 2  # below this there is no inter-fragment separation to enforce


def _normalize(source, charge=0):
    """Normalise SMILES / an .xyz path / an RDKit Mol to ``(mol with Hs, has_geometry)``.

    `charge` is used for xyz bond perception (a charged metal complex / ion).
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path, not just str
    if isinstance(source, Chem.Mol):
        mol = source
    elif isinstance(source, str) and source.lower().endswith(".xyz"):
        mol = _xyz_to_mol(source, charge)
        if mol is None:
            raise ValueError(f"could not read {source}")
    else:
        mol = parse_smiles(source)
    has_geom = mol.GetNumConformers() > 0
    return Chem.AddHs(mol, addCoords=has_geom), has_geom


def _encounter_bounds(mol, slack=1.5, seed=_embed.DEFAULT_SEED):
    """vdW-aware inter-fragment bounds so a multi-fragment SMILES does not embed on top of itself.

    For every pair of fragments, bound their closest heavy-atom pair to roughly van-der-Waals contact
    (sum of vdW radii .. + slack), giving the embedder a separated but touching encounter geometry. The
    probe conformer that decides *which* pair is closest comes from `bounds.probe_conformer` — see there
    for why its seed is fixed.
    """
    tmp = _embed.probe_conformer(mol, seed)
    if tmp is None:
        return {}
    pt = Chem.GetPeriodicTable()
    pos = tmp.GetConformer().GetPositions()
    frags = Chem.GetMolFrags(mol)

    def heavy(f):
        return [i for i in f if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]

    bounds = {}
    for a in range(len(frags)):
        for b in range(a + 1, len(frags)):
            fa, fb = heavy(frags[a]), heavy(frags[b])
            if not fa or not fb:
                continue
            i, j = min(((i, j) for i in fa for j in fb), key=lambda p: np.linalg.norm(pos[p[0]] - pos[p[1]]))
            vdw = pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) + pt.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            bounds[(i, j)] = (vdw, vdw + slack)
    return bounds


@dataclass
class _MetalCtx:
    mol: Chem.Mol
    metal: int
    real_z: int
    real_q: int = 0  # the metal's real formal charge; the surrogate is neutral, restored below
    donors: list | None = None  # the metal's real donor atoms; None = unknown
    geometry: str | None = None  # the coordination polyhedron name (a planar one is checked for pucker)
    extra: list = field(default_factory=list)  # other surrogated metals (idx, real_z, real_q) in a multi-metal complex
    donor_bonds: list = field(default_factory=list)  # the stripped M-donor bonds as (donor, metal) pairs, EVERY metal's
    # — `minimize` re-adds them (`metal.connect_metal`) once element+charge are settled, so the output mol is connected

    def restore(self):
        # element and charge: restoring only Z leaves an M(0) among anionic ligands, so `_calc_charge`
        # hands xtb a total charge wrong by the oxidation state.
        self.mol.GetAtomWithIdx(self.metal).SetAtomicNum(self.real_z)
        self.mol.GetAtomWithIdx(self.metal).SetFormalCharge(self.real_q)
        for mi, rz, rq in self.extra:  # restore the rest of a bimetallic complex
            self.mol.GetAtomWithIdx(mi).SetAtomicNum(rz)
            self.mol.GetAtomWithIdx(mi).SetFormalCharge(rq)


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
    releases exactly these (and any ``constrain=`` windows) while structural holds stay put. **Explicit wins**
    (DESIGN §5.2): a soft window that lands on a pair the user already pinned with a ``fix`` number (a
    structural distance/angle *not* already soft) is dropped — the rigid fix is neither overwritten nor made
    releasable.
    """
    dk, ak = set(cons.contacts[0]), set(cons.contacts[1])
    struct_d = {k for k in cons.distances if k not in dk}  # fix numbers / frozen-core shape — non-releasable
    struct_a = {k for k in cons.angles if k not in ak}
    for (i, j), (lo, hi) in distances.items():
        key = (min(i, j), max(i, j))
        if key in struct_d:  # user fix on this pair overrides a soft grip — leave it rigid
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


def _graft_frozen(mol, conf_ids, frozen, ref):
    """Restore the `frozen` atoms to their EXACT input geometry `ref`, Kabsch-fitted onto the embedded pose.

    Distance geometry only *approximates* a rigid core (~0.2-0.4 Å internal RMSD — fine for a normal
    molecule, wrong for a TS whose partial bonds must be preserved), so we restore the core EXACTLY and
    orient it to best match the embedded periphery. The frozen atoms are then held by `AddFixedPoint`.
    """
    frozen = list(frozen)
    core = np.asarray(ref, float)  # input core, input frame
    if len(frozen) <= 1:  # a point has no shape to restore
        return
    if len(frozen) == _BOND_ATOMS:  # a bond: no orientation, but fix the LENGTH exactly
        dt = float(np.linalg.norm(core[0] - core[1]))  # (keep one atom, slide the other to the template
        for c in conf_ids:  #  distance along the embedded axis)
            conf = mol.GetConformer(c)
            pa = np.array(conf.GetAtomPosition(frozen[0]))
            pb = np.array(conf.GetAtomPosition(frozen[1]))
            v = pb - pa
            n = np.linalg.norm(v)
            v = v / n if n > _EPS else np.array([1.0, 0.0, 0.0])
            conf.SetAtomPosition(frozen[1], (pa + dt * v).tolist())
        return
    ca = core.mean(0)
    core0 = core - ca
    for c in conf_ids:
        conf = mol.GetConformer(c)
        emb = np.array([list(conf.GetAtomPosition(i)) for i in frozen])  # embedded core
        cb = emb.mean(0)
        h = core0.T @ (emb - cb)
        u, _s, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T  # rotate input core onto the embedded one
        fitted = core0 @ r.T + cb
        for i, p in zip(frozen, fitted, strict=False):
            conf.SetAtomPosition(i, p.tolist())


def _bind_substrate(iso, base, contacts, fix, constrain):
    """Fold a substrate's fix/constrain/NCI grip onto the isomer coordination `base`; return ``(base, graft_ref)``.

    Provenance is subtractive, NOT the union `compose` would give: a user grip that lands on a sphere hold stays
    STRUCTURAL (never releasable), or mc(explore=) could dissociate the coordination sphere.
    """
    if not (contacts is not None or fix or constrain):  # a substrate bound via fix / constrain / NCI contacts
        return base, {}
    has_geom = iso.mol.GetNumConformers() > 0
    sub, graft_ref = resolve_core(iso.mol, fix=fix, constrain=constrain, has_geometry=has_geom)
    _add_soft(sub, *_nci_windows(contacts))
    sphere_d, sphere_a = set(base.distances), set(base.angles)  # the coordination sphere already in base —
    soft_d, soft_a = sub.contacts  # never release the sphere, so provenance records ONLY genuinely-new grips.
    # Field-driven via `compose` so no spec field is silently dropped (a hand-listed merge lost a `constrain=`
    # pi-stack in `sub.planes`); last-wins, so a spec landing on a sphere hold overrides it, not demoted to soft.
    base = compose(base, sub).copy(
        contacts=(  # subtractive: a user grip that lands on a sphere hold must not become releasable, or
            frozenset(k for k in soft_d if k not in sphere_d),  # mc(explore=) could dissociate the sphere
            frozenset(k for k in soft_a if k not in sphere_a),
        )
    )
    return base, graft_ref


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
    if isinstance(coordinate, str):  # a SMARTS — may match several atoms
        ms = [m[0] for m in iso.mol.GetSubstructMatches(Chem.MolFromSmarts(coordinate))]
        if not ms:
            raise ValueError(f"coordinate={coordinate!r} matched no atoms")
        if len(ms) == 1:
            return [[ms[0]]]
        if nvac == 1:  # ambiguous + 1 pocket -> one mode per donor
            logger.info(
                "coordinate=%r matched %d atoms; embedding %d coordination modes (one per donor) — "
                ".select(coordinate=<idx>) one or keep all",
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


def _frozen_core_ref(mol, frozen_atoms, graft_ref):
    """Return ``(frozen, ref_core)`` — atoms to Kabsch-graft and their exact target coords (``None`` if none)."""
    frozen = sorted(frozen_atoms)  # capture the input TS core BEFORE the embed
    if frozen and mol.GetNumConformers():  # graft each frozen atom to its explicit-fix coord, else own coord
        own = mol.GetConformer().GetPositions()
        return frozen, np.array([graft_ref.get(i, own[i]) for i in frozen])
    if graft_ref:  # SMILES metal + explicit-coords fix: no conformer, graft only the named atoms
        frozen = sorted(graft_ref)
        return frozen, np.array([graft_ref[i] for i in frozen])
    return frozen, None


def _embed_isomer(iso, *, coordinate, contacts, fix, constrain, n, seed, knowledge, keep_input=False):
    """Embed a metal `Isomer`, optionally binding a substrate; yield one `Ensemble` per binding candidate.

    Usually one, but several when ``coordinate=`` is a SMARTS matching several donor atoms (one candidate per
    donor). Each carries a ``.tag``; `keep_input` adds the Mol's input conformer as a seed (the retain-input path
    — the embed seam then relaxes it into its windows; not returned pristine).
    """
    base, graft_ref = _bind_substrate(iso, iso.cons.copy(), contacts, fix, constrain)
    choices = _coordination_choices(iso, coordinate, iso.vertices.count(_metal.VACANT))

    for atoms in choices:
        mol = Chem.Mol(iso.mol)  # own copy so candidates don't share conformers
        input_conf = Chem.Conformer(mol.GetConformer()) if (keep_input and mol.GetNumConformers()) else None
        cons = base.copy()
        # identity is geometric: arrangement (slot map) + metal chirality. `label` (cis/trans/mer/fac) is
        # kept only as a coarse, sometimes-wrong convenience tag — never the thing you select on.
        tag = {
            "geometry": iso.geometry,
            "arrangement": _metal.arrangement(iso),
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
        # tether any free fragment (a co-crystallised solvent / an un-bonded ligand the SMILES wrote as a
        # separate `.` fragment) at vdW contact, so it embeds as a vdW complex rather than drifting to
        # infinity — the same encounter-bounds the non-metal multi-fragment path applies.
        cons.distances.update(_float_encounter_bounds(mol, cons))
        frozen, ref_core = _frozen_core_ref(mol, cons.frozen, graft_ref)
        mol, held = _metal._hold_donor_chirality(mol, iso.metal, iso.donors, cons)  # hold a carbanion/amine donor's
        try:  # hand (a degree-3 centre with no M-C bond that ETKDG would otherwise let invert -> both hands identical)
            ids = _embed.embed(mol, cons, n or _embed.n_confs(mol, constrained=True), seed=seed, knowledge=knowledge)
        except RuntimeError as e:  # triangle smoothing -> infeasible bounds
            raise ValueError(
                f"could not embed {iso.geometry} {_metal.arrangement(iso)}"
                + (f" with atom(s) {atoms} coordinated" if atoms else "")
                + f": the coordination + substrate constraints are geometrically infeasible "
                f"(e.g. a substrate that can't chelate the requested vertices). [{e}]"
            ) from e
        ids = list(ids)
        mol = _metal._release_donor_chirality(mol, held, cons)  # drop the dummy D's + cons keys, restore charges
        if ref_core is not None:
            _graft_frozen(mol, ids, frozen, ref_core)  # restore the exact frozen TS core
        if input_conf is not None:  # ETKDG cleared confs; re-add the input as a seed
            ids = [mol.AddConformer(input_conf, assignId=True), *ids]
        logger.info(
            "embed[%s: %s%s%s]: %d seeds%s",  # name-agnostic identity: arrangement (+ chirality), not cis/trans
            iso.geometry,
            _metal.arrangement(iso),
            f" {iso.chirality}" if iso.chirality else "",
            f" coord@{atoms}" if atoms else "",
            len(ids),
            " (incl. input geometry)" if input_conf else "",
        )
        ctx = _MetalCtx(
            mol,
            iso.metal,
            iso.real_z,
            iso.real_q,
            iso.donors,
            iso.geometry,
            extra=iso.extra,
            donor_bonds=iso.donor_bonds,
        )
        ens = Ensemble(mol, ids, cons, ctx)
        if ids:  # the labile-donor hand at the uniform initial embed — minimize() culls any later inverted conformer
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
    disc = source.mol if isinstance(source, _metal.Isomer) else _normalize(source, charge)[0]
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
    logger.info("contacts='auto': %d binding mode(s) -> %s", len(modes), list(modes))
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

    ``'auto'`` engages ``'preserve'`` ONLY when the input geometry carries chirality
    the embed can't keep (planar/axial/helical); a SMILES or a stereo-less input is left untouched.
    """
    if stereo == "free":
        return
    ref = source.stereo_ref if isinstance(source, _metal.Isomer) else None
    if ref is None and not isinstance(source, _metal.Isomer):
        try:
            mol, has_geom = _normalize(source, charge)
            ref = _stereo.signature(mol, charge=charge) if has_geom else None
        except Exception:
            ref = None
    if not ref:
        return
    if stereo in ("racemic", "separate", "auto"):  # for a geometry input, auto-preserve ONLY a metallocene's
        #   PLANAR chirality (what the embed genuinely can't keep). xyzgraph's AXIAL/HELICAL perception over-fires
        #   on labile bonds (an aryl-N reads a different Rₐ/Sₐ every rotamer), rejecting valid conformers (BImP
        #   kept 0/4); point R/S is the embed's own job. A genuine atropisomer: opt in with stereo={'axial':'preserve'}.
        spec = {"planar": "preserve", "default": "free"} if "planar" in set(ref) else None
    else:
        spec = stereo
    if spec is None:
        return
    for ens in result if isinstance(result, EnsembleSet) else [result]:
        if isinstance(ens, Ensemble):
            ens._stereo = (spec, ref)


_STEREO_CAP = 32  # max stereoisomers embedded per source before truncating (a loud-logged safety valve)


def _stereo_expand(source, charge, stereo, cap=_STEREO_CAP):
    """Return ``(variants, n_unassigned, total, unresolved)`` to enumerate, or ``None`` for the single-embed path.

    ``None`` when: `stereo` is not an enumerating mode (``'auto'``/``'enumerate'``); the source carries a
    geometry or is a metal `Isomer` (its point stereo is 3D-perceived / the polyhedron path owns its Λ/Δ); or
    nothing is unspecified. Otherwise the source is coordinate-free (a SMILES / conformer-less Mol) with
    undefined stereocentres to expand — see `stereo.enumerate_unassigned`.
    """
    if stereo not in ("racemic", "separate"):  # 'free' opts out; a dict/'preserve' filter-spec is a geometry input
        return None
    if isinstance(source, _metal.Isomer):
        return None
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str) and source.lower().endswith(".xyz"):
        return None  # a geometry defines every stereocentre (AssignStereochemistryFrom3D) — nothing to enumerate
    if isinstance(source, Chem.Mol):
        if source.GetNumConformers() > 0:
            return None  # ditto: a conformer defines the stereo
        mol = source
    elif isinstance(source, str):
        mol = Chem.MolFromSmiles(source)
        if mol is None:  # a genuine parse error — let _embed_dispatch raise the clear message
            return None
    else:
        return None
    if _metal.metal_index(mol) is not None:
        return None  # a metal complex: rx.metal/enumerate_isomers owns its coordination x ligand-stereo load-in
    expanded = _stereo.enumerate_unassigned(mol, cap=cap)
    return None if expanded[1] == 0 else expanded


def _stereo_enumerated_embed(expanded, stereo, dispatch_kw):
    """Embed each stereoisomer variant and assemble per `stereo` mode — the racemate/diastereomer load-in stage.

    ``'racemic'`` folds every variant's candidate(s) into ONE flat `EnsembleSet`; ``'separate'`` keeps them apart
    as a ``list[EnsembleSet]`` (one per stereoisomer, uniform type across metal & organic). Each variant is
    embedded with the SAME effort and tagged ``stereo=<label>``; never pruned against each other (distinct species).
    """
    variants, n_unassigned, total, unresolved = expanded
    labels = ", ".join(lbl or "achiral" for _, lbl in variants)
    if total > _STEREO_CAP:  # more stereoisomers than the safety valve — embedded a truncated subset
        logger.warning(
            "stereo=%r: %d undefined stereocentre(s) -> %d stereoisomers, CAPPED to %d embedded (raise cap= "
            "to embed all): [%s]",
            stereo,
            n_unassigned,
            total,
            len(variants),
            labels,
        )
    else:
        logger.info(
            "stereo=%r: %d undefined stereocentre(s) detected -> deliberately embedding %d stereoisomer(s) "
            "as the racemate/diastereomer set [%s]",
            stereo,
            n_unassigned,
            len(variants),
            labels,
        )
    if unresolved:  # an allene/cumulene/atropisomer axis: EnumerateStereoisomers can't encode it from a flat SMILES
        logger.warning(
            "stereo=%r: %d stereo axis(es) (allene/cumulene/biaryl atropisomer) cannot be enumerated from a "
            "flat SMILES — that axis is embedded as a single ARBITRARY hand; pass a geometry (.xyz) to fix it",
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
            if not ens.ids:  # a 0-conformer embed (e.g. strained trans-cyclooctene) — drop, don't keep a dead candidate
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
        return groups  # list[EnsembleSet] — one per stereoisomer, uniform type across metal & organic
    if not groups:
        raise ValueError("stereo enumeration: none of the stereoisomers could be embedded (infeasible bounds)")
    if len(groups) > 1:
        logger.info(
            "stereo=%r: %d stereoisomers folded into one EnsembleSet — select/score each; do NOT energy-prune "
            "ACROSS them (distinct species, distinct constraint sets, energies not directly comparable)",
            stereo,
            len(groups),
        )
    flat = EnsembleSet(ens for es in groups for ens in es)  # 'racemic': fold into one candidate set
    return flat[0] if len(flat) == 1 else flat  # a lone (all-axial-collapsed) variant stays a bare Ensemble


def _reference_positions(reference, charge=0):
    """Resolve a template reference to an ``(N, 3)`` positions array (.xyz path / Mol / Ensemble / array)."""
    if isinstance(reference, Ensemble):
        cid = reference.ids[0] if reference.ids else reference.mol.GetConformers()[0].GetId()
        return reference.mol.GetConformer(cid).GetPositions()
    if isinstance(reference, Chem.Mol):
        if reference.GetNumConformers() == 0:
            raise ValueError("template reference Mol needs a conformer (a 3D geometry)")
        return reference.GetConformer().GetPositions()
    if isinstance(reference, os.PathLike):
        reference = os.fspath(reference)
    if isinstance(reference, str) and reference.lower().endswith(".xyz"):
        return _xyz_to_mol(reference, charge).GetConformer().GetPositions()
    arr = np.asarray(reference, float)
    if arr.ndim == _TEMPLATE_LEN and arr.shape[1] == _XYZ_DIM:
        return arr
    raise ValueError("template reference must be an .xyz path, a Mol with a conformer, an Ensemble, or an (N,3) array")


def _resolve_template_spec(template, charge):
    """Turn a ``template=(reference, {target_i: ref_i})`` kwarg into ``(positions, mapping)`` for `resolve_core`."""
    if not (isinstance(template, (tuple, list)) and len(template) == _TEMPLATE_LEN and isinstance(template[1], dict)):
        raise ValueError(
            "template= must be (reference, {target_index: reference_index}) — an explicit atom map "
            "(order-proof, no hidden SMARTS matching). E.g. template=('ts.xyz', {0: 5, 4: 1, 5: 6})."
        )
    reference, mapping = template
    return _reference_positions(reference, charge), mapping


def _float_encounter_bounds(mol, cons):
    """Bound fragments no constraint pins to vdW contact — so a free / stray fragment can't drift off.

    Generalises the old two cases into one: a fully unconstrained multi-fragment SMILES (every fragment
    floats -> all pairs bounded) and a fixed/templated core that leaves a spectator fragment (a counter-ion)
    untethered. A pair already linked by a fix / constrain / contact is left to that constraint.
    """
    frags = Chem.GetMolFrags(mol)
    if len(frags) < _MIN_FRAGS:
        return {}
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    touched = {frag_of[a] for a in cons.constrained_atoms()}
    if len(touched) >= len(frags):  # every fragment already pinned/linked
        return {}
    return {
        k: v for k, v in _encounter_bounds(mol).items() if frag_of[k[0]] not in touched or frag_of[k[1]] not in touched
    }


def _dispatch_metal_source(
    source, *, metal, template, coordinate, contacts, fix, constrain, n, seed, knowledge, charge, stereo
):
    """Route a ``metal=<geometry>`` / metal `Isomer` source: enumerate isomers, or embed the one chosen isomer."""
    if template is not None:
        raise ValueError("template= pins a core by graft; it does not compose with metal=/an Isomer source")
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
    if isinstance(source, _metal.Isomer):
        raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
    if isinstance(source, str) and source.lower().endswith(".xyz"):
        source = _normalize(source, charge)[0]  # xyz -> perceived Mol (enumerate wants a Mol/SMILES)
    isos = _isomers.enumerate_isomers(source, metal, stereo=stereo)
    out = EnsembleSet(e for iso in isos for e in _embed_isomer(iso, **_iso_kw))
    logger.info(
        "metal: ENUMERATING isomers of %s -> %d distinct candidate(s) (each a separate "
        "ensemble; .summary() / .select() one to conf-search)",
        metal,
        len(out),
    )
    return out


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
    """Dispatch the embed by input type / spec — see the public `embed` for documentation."""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path everywhere downstream
    if coordinate is not None and metal is None and not isinstance(source, _metal.Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__} — use contacts=/constrain= otherwise"
        )
    if contacts == "auto":  # discover binding modes -> conf-search each
        return _auto_contacts_embed(
            source,
            metal=metal,
            fix=fix,
            constrain=constrain,
            coordinate=coordinate,
            charge=charge,
            n=n,
            seed=seed,
            knowledge=knowledge,
        )
    if metal is not None or isinstance(source, _metal.Isomer):  # the metal enumerate / isomer paths
        return _dispatch_metal_source(
            source,
            metal=metal,
            template=template,
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

    mol, has_geom = _normalize(source, charge)
    if (
        has_geom
        and _metal.metal_index(mol) is not None
        and not (fix or constrain or contacts or coordinate or template)
    ):  # retain the input arrangement
        iso = _cbuild.from_geometry(mol)
        logger.info(
            "metal: source carries a geometry and no metal= -> RETAINING the input ligand "
            "arrangement (%s: %s); pass metal=<geometry> to enumerate isomers instead",
            iso.geometry,
            _metal.arrangement(iso),
        )
        # the embed seam then RELAXES this retained geometry (arrangement kept, M-donor sphere held <0.01 A), so
        # the input is NOT returned pristine — say so, or the RETAINING line reads as "returned as-is".
        logger.info(
            "metal: the retained input geometry is RELAXED into its constraint windows by the embed seam "
            "(arrangement kept, coordination sphere held) — it is not returned as-is/pristine"
        )
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
    metal_ctx = None
    metal_core = set()
    metals_donors = {}
    hydrides = []
    if _metal.metal_index(mol) is not None and (fix or constrain or contacts or template):
        # every M-donor bond, read BEFORE the strip — re-added as DATIVE on the pipeline output (`connect_metal`)
        donor_bonds = [
            (n.GetIdx(), mi) for mi in _metal.metal_indices(mol) for n in mol.GetAtomWithIdx(mi).GetNeighbors()
        ]
        if has_geom:  # capture each metal's donors BEFORE the bonds
            metals_donors = {
                mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()] for mi in _metal.metal_indices(mol)
            }
        else:  # no geometry to hold the sphere from -> at least
            hydrides = [
                (mi, mol.GetAtomWithIdx(mi).GetAtomicNum(), nb.GetIdx())  # keep a terminal HYDRIDE bonded
                for mi in _metal.metal_indices(mol)  # so an MH NCI donor mode (M-H
                for nb in mol.GetAtomWithIdx(mi).GetNeighbors()  # to an acceptor) survives the
                if nb.GetAtomicNum() == 1 and nb.GetDegree() == 1
            ]  # surrogate's bond strip
        mol, metals, core_donors = _metal.surrogate_all_metals(mol)  # surrogate EVERY metal (bimetallic-safe)
        (m, real_z, real_q), extra = metals[0], metals[1:]  # extra: [(idx, real_z, real_q), ...]
        metal_ctx = _MetalCtx(mol, m, real_z, real_q, extra=extra, donor_bonds=donor_bonds)
        _metal_core_donors = core_donors  # deferred: only pin the sphere if a core is actually grafted (below)
        logger.debug("metal complex: %d metal(s) swapped to carbon surrogate for the FF", len(metals))
    else:
        _metal_core_donors = []

    tmpl = _resolve_template_spec(template, charge) if template is not None else None
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, template=tmpl, has_geometry=has_geom)
    _add_soft(cons, *_nci_windows(contacts))  # NCI grips: soft, releasable by mc(explore=). fix numbers, the
    #   frozen-core shape and the sphere/M-H/encounter holds are structural and stay OUT of `contacts`, so
    #   mc(explore=) releases exactly the soft grips and never the structure.
    user_graft = dict(ref)  # atoms to Kabsch-graft onto their exact coords (own / explicit / template)
    if user_graft and _metal.metal_index(mol) is not None:  # a grafted core -> pin the metal coordination
        metal_core = set(_metal.metal_indices(mol)) | set(_metal_core_donors)  # cores so they can't drift
    cons.frozen |= metal_core
    for mi, dons in metals_donors.items():  # a spectator metal is held intact-but-achiral by
        _metal.hold_shape(mol, [mi, *dons], cons)  # hold_shape (NOT grafted — its handedness must
        #                                          stay free for the stereo filter), then pinned at the embed
    if metals_donors and metal_ctx is not None:
        # A frozen metal has no DOF, but its bond-less carbon still fires fictitious LJ at every ligand atom.
        # hold_shape above pinned each sphere as an all-pairs body, so ff_terms gives these metals the zero-vdW
        # type + floors but no pulls (pulling a rigid shape's M-donor members tears a spectator ferrocene).
        # hold_shape must run first: ff_terms reads the `cons.shapes` record it leaves.
        real_z = {metal_ctx.metal: metal_ctx.real_z, **{mi: rz for mi, rz, _rq in metal_ctx.extra}}
        _distance.ff_terms(mol, cons, {mi: (real_z[mi], list(dons)) for mi, dons in metals_donors.items()})
    for mi, rz, h in hydrides:  # covalent M-H window (no input geometry to read)
        d = _PT.GetRcovalent(rz) + _PT.GetRcovalent(1)
        add_distance(cons.distances, mi, h, d - 0.1, d + 0.15)
    cons.distances.update(_float_encounter_bounds(mol, cons))  # keep free/stray fragments from drifting off

    n = n or _embed.n_confs(mol, constrained=cons.is_constrained)
    graft_atoms = sorted(user_graft)
    ref_core = np.array([user_graft[i] for i in graft_atoms]) if graft_atoms else None
    ids = _embed.embed(mol, cons, n, seed=seed, knowledge=knowledge)
    if ref_core is not None:
        _graft_frozen(mol, ids, graft_atoms, ref_core)  # restore the exact fixed core (partial bonds preserved)
    n_frag = len(Chem.GetMolFrags(mol))
    logger.info(
        "embed: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles),
    )
    return Ensemble(mol, list(ids), cons, metal_ctx)
