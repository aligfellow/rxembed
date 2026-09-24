"""Embed an RDKit Mol or metal Isomer under explicit constraints.

`fix` is rigid and `constrain` is releasable. Edited RDKit bounds seed coordinates, an exact fixed core is
grafted back, and `Conformers.minimize()` applies restrained UFF. Parsing and NCI discovery belong upstream.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem
from rdkit.Numerics import rdAlignment

from . import metal_core as _metal
from . import metal_polyhedron as _poly
from . import metal_stereo as _metal_stereo
from . import stereo as _stereo
from .bounds import (
    _MAX_SEED_COUNT,
    DEFAULT_SEED,
    EmbedParams,
    probe_conformer,
    resolve_params,
    seed_coordinates,
    seed_count,
)
from .constraints import (
    _ANGLE_ATOMS,
    _DIHEDRAL_ATOMS,
    _DIST_ATOMS,
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    Constraints,
    _stated_dihedral,
    _structural_dihedral_owned,
    compose,
    constraint_value,
    resolve_atom,
    resolve_core,
    template_to_fix,
    within_window,
)
from .metal_core import VACANT, materialized_state
from .metal_isomer import Isomer, from_geometry, isomer_roles, winding_signature
from .metal_perceive import donor_orientation
from .relax import MAX_ITERS, UFFTypingError, _bonding_failure, _error_summary, restrained_uff

logger = logging.getLogger("rxembed")  # configured by rxembed.set_verbose


class EmbeddingError(ValueError):
    """Report that bounded conformer generation could not satisfy a requested identity."""


@dataclass(frozen=True)
class Failure:
    """Separate an acceptance policy key from its user-facing explanation."""

    kind: str
    detail: str

    def __str__(self):
        """Return the user-facing explanation."""
        return self.detail


def _high_force_candidate(reason):
    """Allow the expensive stiffness tail only for a correct shape missing an M-L wall by a small amount."""
    return isinstance(reason, Failure) and reason.kind == "ml_distance"


def _stable_coordination_failure(reason):
    """Return whether a converged endpoint settled on another labelled coordination shape."""
    return isinstance(reason, Failure) and reason.kind == "coordination_shape"


_EPS = 1e-6  # near-zero norm floor for the graft axis
_STRUCT_CAP_SLACK = 2.0  # deg: finite-force settling tolerance on coplanarity and umbrella caps
_SHAPE_TEAR_TOL = 0.10  # A beyond a rigid body's own 0.1 A all-pairs windows
# A model M-L window (about 0.05 A half-width) is a fitted target, not a user-stated number. FIX_DISTANCE_TOL's
# 0.001 A slack is force-field convergence noise, not measurement precision. On RITCIG's limiting M-L pair the
# restrained-UFF residual falls as C/fc with C = 0.56 A, so 0.001 A needs fc ~ 560, while 0.01 A only needs
# the 100x rung; the 300x/1000x rungs in _HIGH_FC_ESCALATION stay for the cases that rung misses.
_ML_WINDOW_TOL = 0.01  # A: physical slack for a model-derived (not user-fixed) M-L window


def _structural_failure(mol, cid, cons):
    """Name the first missed structural postcondition, including its measured value and accepted range."""
    conf = mol.GetConformer(int(cid))
    pos = conf.GetPositions()

    def failed(label, atoms, value, window, unit, kind="structural_constraint"):
        measured = "undefined"
        if value is not None and np.isfinite(value):
            excess = max(window[0] - value, value - window[1], 0.0)
            measured = f"{value:.3f} (excess {excess:.3g})"
        return Failure(
            kind,
            f"{label} {atoms}: {measured} {unit}; expected [{window[0]:g}, {window[1]:g}] {unit}",
        )

    for atoms, window in cons.distances.items():
        # A phantom is private scaffolding, not a postcondition on the flexible face's real centroid.
        if atoms in cons.contacts[0] or cons.phantoms.intersection(atoms) or not cons.metals.intersection(atoms):
            continue
        # `cons.fixed` is the one record of a user-stated exact distance (fix=/template=); anything else
        # reaching this loop is a model-derived M-L window (constrain= is already excluded via contacts[0]).
        tol = FIX_DISTANCE_TOL if atoms in cons.fixed else _ML_WINDOW_TOL
        value = constraint_value(pos, atoms, cons.haptic, window)
        if not within_window(value, window, tol):
            accepted = window[0] - tol, window[1] + tol
            return failed("M-L distance", atoms, value, accepted, "A", kind="ml_distance")
    for key in cons.coplanar:
        atoms, cap = key[:4], key[5]
        if _stated_dihedral(cons, *atoms):
            continue
        measured = constraint_value(pos, atoms, cons.haptic)
        accepted = 0.0, cap + _STRUCT_CAP_SLACK
        if measured is None or not np.isfinite(measured):
            return failed("coplanarity deviation", atoms, measured, accepted, "deg")
        value = abs(measured)
        deviation = min(value, 180.0 - value)
        if deviation > accepted[1]:
            return failed("coplanarity deviation", atoms, deviation, accepted, "deg")
    for atoms, ideal in cons.umbrellas.items():
        if ideal in (None, 0.0) or isinstance(ideal, tuple) or _stated_dihedral(cons, *atoms, improper=True):
            continue
        measured = constraint_value(pos, atoms, cons.haptic)
        accepted = ideal - _STRUCT_CAP_SLACK, 90.0 + _STRUCT_CAP_SLACK
        if measured is None or not np.isfinite(measured):
            return failed("absolute umbrella dihedral", atoms, measured, accepted, "deg")
        value = abs(measured)
        if not accepted[0] <= value <= accepted[1]:
            return failed("absolute umbrella dihedral", atoms, value, accepted, "deg")
    return None


SHAPE_PROP = "shape"  # RDKit conformer property an accepted conformer carries: see `_shape_clause`
# Machine-readable {metal atom: requested geometry} beside SHAPE_PROP, one accepted conformer at a time.
# metal_smiles.cxsmiles reads this to write CX in the requested frame instead of re-perceiving by argmin;
# a conformer without it (a raw geometry, never through the acceptance gate) keeps the argmin reading.
SHAPE_REQUEST_PROP = "shape_request"


def _encode_shape_request(requests):
    """Return `requests` (``{metal atom: requested geometry}``) as one `SHAPE_REQUEST_PROP` string."""
    return ";".join(f"{atom}:{geometry}" for atom, geometry in sorted(requests.items()))


def _decode_shape_request(value):
    """Return the `{metal atom: requested geometry}` a `SHAPE_REQUEST_PROP` string encodes."""
    pairs = (pair.split(":", 1) for pair in value.split(";") if pair)
    return {int(atom): geometry for atom, geometry in pairs}


def _shape_clause(geometry, residual, next_name, next_err):
    """Format one shape reading as its code, residual and runner-up: ``'TBP 0.053 (next SPY 0.198)'``."""
    clause = f"{_poly.record(geometry).code} {residual:.3f}"
    if next_name is not None:
        clause += f" (next {_poly.record(next_name).code} {next_err:.3f})"
    return clause


@dataclass(frozen=True)
class _CentreStatePrep:
    """Cache one metal centre's per-isomer coordination-state geometry, reused across every conformer."""

    state: _metal.MetalState
    poly: _poly.Polyhedron
    centre: str
    vertices: list
    haptic: dict
    donors: list
    occupied: list
    ideal: np.ndarray
    constitutionally_equivalent: bool
    exhaustive: bool
    orders: list | None  # every equivalent donor-slot seating; None when constitutionally_equivalent
    input_reading: tuple | None  # (residual, next name, next residual) on iso.mol's own conformer, or None


def _coordination_state_prep(iso):
    """Return `iso`'s per-centre coordination-state geometry, independent of any conformer's coordinates."""
    if iso is None:
        return None
    parts = _metal.materialized_states(iso.mol, iso.centres)
    prepped = []
    for state in iso.centres:
        poly = _poly.POLYHEDRA.get(state.geometry)
        if poly is None:
            continue
        centre = f"{Chem.GetPeriodicTable().GetElementSymbol(state.atomic_num)}{state.atom}"
        vertices, haptic, _winding, donors = parts[state.atom]
        occupied = [i for i, vertex in enumerate(vertices) if vertex != VACANT]
        if len(occupied) < 2:  # noqa: PLR2004 - one site has no relative direction and cannot define a shape
            continue
        ideal = np.array([poly.vertex_dirs[slot] for slot in occupied], float)
        ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
        classes = _metal_stereo.site_classes(iso._graph, vertices, haptic, isomer_roles(iso))
        keys = [None if site == VACANT else classes[site] for site in vertices]
        links = _metal_stereo.chelate_links(iso._graph, vertices, haptic)
        constitutionally_equivalent = len({keys[slot] for slot in occupied}) == 1 and not links
        orders = None
        if not constitutionally_equivalent:
            row = {slot: i for i, slot in enumerate(occupied)}
            orders = [
                [row[assignment[slot]] for slot in occupied]
                for assignment in _metal_stereo.equivalent_site_assignments(keys, links)
            ]
        # A graph input (no conformer) has no input part: `shape_gap` reads `iso.mol`, not the conformer under
        # validation, once per Isomer rather than once per conformer.
        residual, next_name, next_err, _accepted = _metal.shape_gap(
            iso.mol, state.atom, vertices, haptic, state.geometry
        )
        prepped.append(
            _CentreStatePrep(
                state=state,
                poly=poly,
                centre=centre,
                vertices=vertices,
                haptic=haptic,
                donors=donors,
                occupied=occupied,
                ideal=ideal,
                constitutionally_equivalent=constitutionally_equivalent,
                exhaustive=_poly.seating_is_exhaustive(ideal),
                orders=orders,
                input_reading=None if residual is None else (residual, next_name, next_err),
            )
        )
    return prepped


def _centre_observed(pos, prep):
    """Return `(points, observed)` for one centre's occupied sites, or the `Failure` explaining why not.

    `points` is `{slot: 3D position}`; `observed` stacks their unit directions from the metal in the same
    order as `prep.occupied`. Split out of `_centre_state_failure` to keep its own branching flat.
    """
    state, poly, centre = prep.state, prep.poly, prep.centre
    vertices, haptic, donors = prep.vertices, prep.haptic, prep.donors
    occupied = prep.occupied

    def failed(detail):
        return Failure("coordination_shape", detail)

    if not np.all(np.isfinite(pos[state.atom])):
        return failed(f"coordination state at {centre} has non-finite metal coordinates")
    points = {}
    for slot in occupied:
        vertex = vertices[slot]
        point = np.mean(pos[list(haptic[vertex])], axis=0) if vertex in haptic else pos[vertex]
        if not np.all(np.isfinite(point)):
            return failed(f"coordination state at {centre} has non-finite donor coordinates")
        points[slot] = point
    if poly.planar and donors and not _metal.coplanar(pos, state.atom, donors, haptic=haptic):
        return failed(f"coordination state at {centre} is nonplanar; expected {state.geometry}")
    if (
        not poly.planar
        and len(occupied) >= 3  # noqa: PLR2004 - fewer donors gives no plane to test against
        and np.linalg.matrix_rank(np.array([poly.vertex_dirs[i] for i in occupied], float)) == 3  # noqa: PLR2004
        and _metal.coplanar(pos, state.atom, donors, haptic=haptic)
    ):
        return failed(f"coordination state at {centre} collapsed into a plane; expected {state.geometry}")
    observed = []
    for slot in occupied:
        direction = points[slot] - pos[state.atom]
        if not np.all(np.isfinite(direction)):
            return failed(f"coordination state at {centre} has a non-finite donor direction")
        length = np.linalg.norm(direction)
        if length <= _EPS:
            return failed(f"coordination state at {centre} has a donor collapsed onto the metal")
        observed.append(direction / length)
    return points, np.array(observed)


def _shape_line(prep, requested_residual, next_name, next_err):
    """Format one accepted centre's `SHAPE_PROP` line: its reading, and the input's when there is one."""
    line = f"{prep.centre} {_shape_clause(prep.state.geometry, requested_residual, next_name, next_err)}"
    if prep.input_reading is not None:
        line += f"; input {_shape_clause(prep.state.geometry, *prep.input_reading)}"
    return line


def _centre_state_failure(pos, mol, cid, prep, lines=None, requests=None):
    """Return the first reason one metal centre's conformer geometry misses its prepped polyhedron.

    Appends this centre's accepted shape record to `lines` (see `SHAPE_PROP`) and its requested geometry to
    `requests` (see `SHAPE_REQUEST_PROP`) when it passes.
    """
    state, centre, occupied, ideal = prep.state, prep.centre, prep.occupied, prep.ideal

    def failed(detail):
        return Failure("coordination_shape", detail)

    result = _centre_observed(pos, prep)
    if isinstance(result, Failure):
        return result
    points, observed = result
    # Rule B: the requested shape is accepted within `_FIT_MARGIN` of the best reading (`_metal.shape_reading`);
    # a vacant site is read against the occupied coordination number. This residual doubles as `unrestricted_fit`
    # and, when constitutionally equivalent, `equivalent_fit` below: one ranking, not a repeat exhaustive search.
    pts = [points[slot] for slot in occupied]
    rms = _metal._plane_rms(pos[state.atom], pts)
    r = float(np.mean([np.linalg.norm(pt - pos[state.atom]) for pt in pts]))
    vacant = len(occupied) < len(prep.vertices)
    ranked = _metal.rank_shapes(observed, rms, r, vacancy=(state.geometry, ideal) if vacant else None)
    requested_residual, next_name, next_err, accepted = _metal.shape_reading(ranked, state.geometry)
    if not accepted:
        return failed(
            f"coordination state at {centre}: expected {state.geometry} {requested_residual:.3f}, "
            f"reads {next_name} {next_err:.3f}"
        )
    unrestricted_fit = requested_residual if prep.exhaustive else None
    if prep.constitutionally_equivalent:
        equivalent_fit = requested_residual
    else:
        equivalent_fit = min(
            (
                _poly.ordered_fit_residual(observed[order], ideal)
                for order in prep.orders
                if not state.hand or _poly.orientation_parity(observed[order], ideal) > 0
            ),
            default=None,
        )
        if equivalent_fit is None:
            return failed(f"coordination state at {centre} has no donor-slot seating compatible with {state.geometry}")
    if equivalent_fit > _metal._FIT_FLOOR:
        return failed(
            f"coordination state at {centre} does not fit {state.geometry}: "
            f"residual {equivalent_fit:.3f} > {_metal._FIT_FLOOR:.2f}"
        )
    # An exact unrestricted search is an identity witness: another constitutional seating fitting better means
    # this endpoint crossed its slot boundary. Above the resource cap it is only an approximation, so it
    # cannot reject an otherwise compatible labelled state.
    if unrestricted_fit is not None and equivalent_fit > unrestricted_fit + _EPS:
        return failed(f"coordination state at {centre} crossed its donor-slot seating")
    if lines is not None:
        lines.append(_shape_line(prep, requested_residual, next_name, next_err))
    if requests is not None:
        requests[state.atom] = state.geometry
    return None


def _coordination_state_failure(mol, cid, iso, states=None):
    """Return the first reason a conformer does not retain its selected polyhedron.

    Every centre's accepted shape record is written as `mol`'s `SHAPE_PROP` conformer property when all pass,
    beside its machine-readable `SHAPE_REQUEST_PROP` counterpart (`metal_smiles.cxsmiles` reads it back).
    """
    if iso is None:
        return None
    if states is None:
        states = _coordination_state_prep(iso)
    pos = mol.GetConformer(int(cid)).GetPositions()
    lines, requests = [], {}
    for prep in states:
        if reason := _centre_state_failure(pos, mol, cid, prep, lines, requests):
            return reason
    if lines:
        mol.GetConformer(int(cid)).SetProp(SHAPE_PROP, " | ".join(lines))
    if requests:
        mol.GetConformer(int(cid)).SetProp(SHAPE_REQUEST_PROP, _encode_shape_request(requests))
    return None


def _coordination_state_ok(mol, cid, iso, states=None):
    """Return whether one conformer retains each selected polyhedron."""
    return _coordination_state_failure(mol, cid, iso, states=states) is None


def _donor_facing_failure(mol, donors, frozen):
    """Warn on a donor angle below its census floor; never reject a conformer for it.

    A floor violation is not proof of folding: real crystal donors cross it too (see
    `metal_perceive.donor_orientation`), so refusing a conformer for this alone would refuse a legitimate
    geometry. Each warning names the donor, its substituent, and the measured angle.
    """
    for violation in donor_orientation(mol, mol.GetConformer().GetPositions(), donors, frozen):
        logger.warning("donor orientation: %s", violation.detail)


def _ligand_stereo_failure(mol, iso):
    """Return the first ligand point, E/Z or axial mismatch on the restored public graph."""

    def failed(detail, found):
        kind = "ligand_stereo_unassigned" if found == "unassigned" else "ligand_stereo"
        return Failure(kind, detail)

    wanted_points = _stereo.point_stereo(iso.stereo_label)
    wanted_ez = _stereo.bond_stereo(iso.stereo_label)
    wanted_axes = _stereo.axis_stereo(iso.stereo_label)
    if not (wanted_points or wanted_ez or wanted_axes):
        return None
    realised = _stereo.stereo_from_3d(mol, exclude=_metal.metal_indices(mol))
    realised_points = _stereo.point_stereo(realised)
    for atom, expected in wanted_points.items():
        if realised_points.get(atom) != expected:
            found = realised_points.get(atom, "unassigned")
            return failed(f"wrong ligand point stereo at atom {atom}: expected {expected}, found {found}", found)
    realised_ez = _stereo.bond_stereo(realised)
    for pair, expected in wanted_ez.items():
        if realised_ez.get(pair) != expected:
            found = realised_ez.get(pair, "unassigned")
            return failed(f"wrong ligand E/Z at bond {tuple(sorted(pair))}: expected {expected}, found {found}", found)
    realised_axes = _stereo.axis_stereo(realised)
    for pair, expected in wanted_axes.items():
        if realised_axes.get(pair) != expected:
            found = realised_axes.get(pair, "unassigned")
            return failed(
                f"wrong ligand axial stereo at bond {tuple(sorted(pair))}: expected {expected}, found {found}", found
            )
    return None


# Half-order steps find the lowest restraint stiffness that keeps the sphere intact.
BASE_STIFFNESS = 1.0
FC_ESCALATION = (1.0, 3.0, 10.0, 30.0, 100.0)
_HIGH_FC_ESCALATION = (300.0, 1000.0)
# Each bounded pool tolerates post-relax attrition; three deterministic batches cover DG basin variability.
_REPLACEMENT_FACTOR = 4
_REPLACEMENT_ROUNDS = 3


def encounter_bounds(mol, slack=1.5, seed=DEFAULT_SEED, *, fragments=None):
    """Keep each fragment pair near van der Waals contact through its closest heavy atoms."""
    tmp = probe_conformer(mol, seed)
    if tmp is None:
        return {}
    pt = Chem.GetPeriodicTable()
    pos = tmp.GetConformer().GetPositions()
    frags = Chem.GetMolFrags(mol) if fragments is None else fragments

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


def float_encounter_bounds(mol, cons):
    """Keep metric components near each other with one encounter bound per component pair."""
    frags = Chem.GetMolFrags(mol)
    if len(frags) < 2:  # noqa: PLR2004 - fewer than two fragments leaves no pair to bound
        return {}
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    groups = [set(fragment) for fragment in frags]
    for i, j in cons.distances:
        if i not in frag_of or j not in frag_of:
            continue  # a haptic centroid is materialised only after this step
        left, right = frag_of[i], frag_of[j]
        if groups[left] is groups[right]:
            continue
        joined = groups[left] | groups[right]
        for atom in joined:
            groups[frag_of[atom]] = joined
    components = list({id(group): tuple(sorted(group)) for group in groups}.values())
    if len(components) < 2:  # noqa: PLR2004 - fewer than two components leaves no pair to bound
        return {}
    component_of = {atom: ci for ci, component in enumerate(components) for atom in component}
    touched = {component_of[atom] for atom in cons.constrained_atoms() if atom in component_of}
    if len(touched) == len(components):
        return {}
    return {
        pair: window
        for pair, window in encounter_bounds(mol, fragments=components).items()
        if component_of[pair[0]] not in touched or component_of[pair[1]] not in touched
    }


def graft_frozen(mol, conf_ids, frozen, ref):
    """Kabsch-fit and restore a frozen core exactly after distance geometry."""
    frozen = list(frozen)
    core = np.asarray(ref, float)  # input core, input frame
    if len(frozen) <= 1:  # a point has no shape to restore
        return
    if len(frozen) == _DIST_ATOMS:  # a bond: keep one atom, slide the other to the template distance along the
        dt = float(np.linalg.norm(core[0] - core[1]))  # embedded axis: no orientation to restore, just length
        for c in conf_ids:
            conf = mol.GetConformer(c)
            pa = np.array(conf.GetAtomPosition(frozen[0]))
            pb = np.array(conf.GetAtomPosition(frozen[1]))
            v = pb - pa
            n = np.linalg.norm(v)
            v = v / n if n > _EPS else np.array([1.0, 0.0, 0.0])
            conf.SetAtomPosition(frozen[1], (pa + dt * v).tolist())
        return
    for c in conf_ids:
        conf = mol.GetConformer(c)
        emb = np.array([list(conf.GetAtomPosition(i)) for i in frozen])  # embedded core
        # reflect=False (the default): a proper rotation only, never a mirror image of the input core.
        _ssd, transform = rdAlignment.GetAlignmentTransform(emb, core)
        transform = np.asarray(transform)
        fitted = core @ transform[:3, :3].T + transform[:3, 3]  # rotate input core onto the embedded one
        for i, p in zip(frozen, fitted, strict=False):
            conf.SetAtomPosition(i, p.tolist())


def _frozen_core_ref(mol, frozen_atoms, graft_ref):
    """Return ``(frozen, ref_core)``: atoms to Kabsch-graft and their exact target coords, ``None`` if none."""
    frozen = sorted(frozen_atoms)  # capture the input core before the embed
    if frozen and mol.GetNumConformers():  # graft each frozen atom to its explicit-fix coord, else its own coord
        own = mol.GetConformer().GetPositions()
        return frozen, np.array([graft_ref.get(i, own[i]) for i in frozen])
    if graft_ref:  # SMILES metal + explicit-coords fix: no conformer, graft only the named atoms
        frozen = sorted(graft_ref)
        return frozen, np.array([graft_ref[i] for i in frozen])
    return frozen, None


def fold_substrate(base, sub, graft_ref, *, protect_arrangement=True):
    """Compose substrate constraints with a coordination sphere.

    Field-driven composition keeps every constraint kind, including `sub.planes`. A soft substrate grip may
    not replace a structural sphere hold. A numeric fix may override one explicitly. A coordinate graft may
    pin at most one sphere atom; two would silently replace the selected arrangement.
    """
    sphere = set(base.metals)
    sphere.update(atom for pair in base.distances if base.metals.intersection(pair) for atom in pair)
    pinned = sorted(sphere & set(graft_ref)) if protect_arrangement else []
    if len(pinned) > 1:
        raise ValueError(
            f"the coordinate graft (fix={{i: (x,y,z)}} / template=) pins coordination-sphere atoms {pinned}: "
            f"the graft restores them to their exact reference coordinates after the embed, so it, not the "
            f"coordination geometry, would decide how they sit, and you would get the reference's arrangement "
            f"under this isomer's label. Graft at most one sphere atom (a core over ligand backbone atoms "
            f"composes fine), or embed the reference geometry itself as the source."
        )
    sphere_d = set(base.distances)
    soft_d, soft_angular = sub.contacts
    canonical = lambda key: min(tuple(key), tuple(reversed(key)))  # noqa: E731
    sphere_angular = {canonical(k) for k in base.angles}
    overlap = sorted(
        (soft_d & sphere_d)
        | {
            k
            for k in soft_angular
            if canonical(k) in sphere_angular or (len(k) == _DIHEDRAL_ATOMS and _structural_dihedral_owned(base, k))
        }
    )
    if overlap:
        raise ValueError(
            f"constrain= overlaps structural coordination term(s) {overlap}: a soft bias cannot replace the "
            f"selected metal state. Keep the structural term, or use fix= for an explicit rigid override."
        )
    return compose(base, sub)


def _stereo_targets(iso):
    """Return metal states that require hand or haptic-winding selection after DG."""
    if iso is None:
        return []
    out = []
    for state in iso.centres:
        vertices, haptic, winding, _donors = materialized_state(iso, state)
        if state.hand or winding:
            out.append((state, vertices, haptic, winding))
    return out


def _winding_ranks(mol, targets):
    """Return donor and eta2 CIP ranks needed by the stated haptic windings."""
    rank_mol = mol
    if _metal.metal_indices(mol):
        rank_mol, _metals = _metal.surrogate_all_metals(Chem.Mol(mol))
    ranks = {}
    for state, vertices, haptic, winding in targets:
        donors = [d for d in vertices if d != VACANT and d not in haptic]
        donors += [atom for face in haptic.values() for atom in face]
        ranks[state.atom] = _metal_stereo.donor_classes(rank_mol, donors) if winding else {}
    eta2 = any(
        len(haptic[dummy]) == _metal._ETA2 for _state, _vertices, haptic, winding in targets for dummy in winding
    )
    return ranks, list(Chem.ComputeAtomCIPRanks(rank_mol)) if eta2 else None


def _restore_connected(mol, cid, iso):
    """Return conformer `cid` on the restored public graph, donor bonds reconnected for a stereo read.

    Shared by seed selection and the post-relax gate, which both need this real-metal, real-bond view.
    """
    restored = iso.restore(Chem.Mol(mol, False, int(cid)))
    return _metal.connect_metal(restored, iso.donor_bonds)


def _seed_ligand_stereo_matches(mol, cid, iso):
    """Return whether a raw seed retains all stated ligand stereo on the restored public graph."""
    return _ligand_stereo_failure(_restore_connected(mol, cid, iso), iso) is None


def _seed_stereo_matches(mol, cid, iso, targets, winding_ranks, eta2_ranks, reflectable, check_ligand=False):
    """Return whether one raw DG seed matches every requested discrete stereo state."""
    pos = mol.GetConformer(int(cid)).GetPositions()
    for state, vertices, haptic, winding in targets:
        hand = state.hand
        realised = (
            _metal_stereo.realised_chirality(
                mol,
                cid,
                state.geometry,
                vertices,
                state.atom,
                hand,
                haptic,
            )
            if hand
            else ""
        )
        if realised and realised != hand and reflectable:
            _reflect(mol, cid)  # exact isometry: no distance, angle or conformer diversity changes
            realised, pos = hand, mol.GetConformer(int(cid)).GetPositions()
        if hand and realised != hand:
            return False
        realised_winding = {
            dummy: sign
            for dummy in winding
            if (
                sign := _metal_stereo.face_winding(
                    mol, pos, state.atom, haptic[dummy], winding_ranks[state.atom], eta2_ranks
                )
            )
        }
        if len(realised_winding) != len(winding) or winding_signature(
            iso, state, realised_winding
        ) != winding_signature(iso, state, winding):
            return False
    return not check_ligand or _seed_ligand_stereo_matches(mol, cid, iso)


def _seed_geometry_matches(mol, cid, iso):
    """Return whether a raw seed has the selected coarse polyhedron before restrained cleanup.

    A vacant site is not read here (unlike the acceptance gate): a raw seed's occupied-site shell is too
    unsettled for a vacancy reading to be a useful sort key.
    """
    if iso is None:
        return True
    for state in iso.centres:
        vertices, haptic, _winding, _donors = materialized_state(iso, state)
        if VACANT in vertices:
            continue
        if not _metal.shape_gap(mol, state.atom, vertices, haptic, state.geometry, cid)[3]:
            return False
    return True


def _stereo_donor_bonds(mol, iso):
    """Return bonds that give labelled donors their full CIP reference during ETKDG."""
    if iso is None:
        return []
    points = set(_stereo.point_stereo(iso.stereo_label))
    return [
        (donor, metal)
        for donor, metal in iso.donor_bonds
        if (
            donor in points
            or mol.GetAtomWithIdx(donor).GetChiralTag()
            in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
        )
    ]


def seed_conformers(mol, cons, iso, n, params, graft_ref=None):
    """Seed `n` conformers, selecting metal and donor hands before cleanup and grafting a fixed core exactly."""
    if n is not None and int(n) <= 0:  # `n or seed_count(...)` below would treat 0 as "use the default"
        raise ValueError(f"n={n}: give a positive conformer count, or None for the flexibility-scaled default")
    cons.distances.update(float_encounter_bounds(mol, cons))  # keep free/stray fragments from drifting off
    frozen, ref_core = _frozen_core_ref(mol, cons.frozen, graft_ref or {})
    target = n or seed_count(mol, constrained=cons.is_constrained)
    targets = _stereo_targets(iso)
    point_stereo = bool(iso and _stereo.point_stereo(iso.stereo_label))
    ligand_stereo = bool(
        iso and (point_stereo or _stereo.bond_stereo(iso.stereo_label) or _stereo.axis_stereo(iso.stereo_label))
    )
    needs_selection = bool(targets or ligand_stereo)
    hands = sum(bool(state.hand) for state, _vertices, _haptic, _winding in targets)
    windings = sum(len(winding) for _state, _vertices, _haptic, winding in targets)
    winding_ranks, eta2_ranks = _winding_ranks(mol, targets)
    stereo_bonds = _stereo_donor_bonds(mol, iso)

    def generate(count, trial_params, *, hard_chirality=True):
        max_attempts = 30 if point_stereo else 0
        return list(
            seed_coordinates(
                mol, cons, count, trial_params, enforce_chirality=hard_chirality, max_attempts=max_attempts
            )
        )

    reflectable = _reflection_is_free(mol, iso, cons, targets, stereo_bonds)
    if stereo_bonds:
        missing = {
            donor
            for donor, _metal_idx in stereo_bonds
            if mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        }
        tags = {}
        if missing:
            full = Chem.Mol(mol)
            iso.restore(full)
            full = _metal.connect_metal(full, iso.donor_bonds)
            _stereo.apply_point_stereo(full, iso.stereo_label, missing)
            tags = {donor: full.GetAtomWithIdx(donor).GetChiralTag() for donor in missing}
        iso.restore(mol)  # use the actual metal, not an isotope surrogate, as the donor's fourth reference
        mol = _metal.connect_metal(mol, stereo_bonds)
        for donor, tag in tags.items():
            mol.GetAtomWithIdx(donor).SetChiralTag(tag)
    if not needs_selection:
        ids = generate(target, params)
        if ref_core is not None:
            graft_frozen(mol, ids, frozen, ref_core)  # restore the exact frozen core
    else:
        kept = Chem.Mol(mol)
        kept.RemoveAllConformers()
        pool = _REPLACEMENT_FACTOR * target
        # A spare-success margin for independent binary metal states; a free reflection needs no margin.
        bits = hands + windings
        trials = target if reflectable else 2**bits * (target + 2 * (math.isqrt(target - 1) + 1))
        budget = max(target, min(_MAX_SEED_COUNT, max(2 * trials, seed_count(mol, constrained=True))))
        used = attempt = generated = 0
        while kept.GetNumConformers() < target and used < budget:
            need = target - kept.GetNumConformers()
            batch_n = min(_REPLACEMENT_FACTOR * need, pool, budget - used)
            hard_chirality = not point_stereo or attempt == 0
            trial_prune = -1 if len(targets) > 1 else params.prune_rms
            trial = replace(params, seed=params.seed + attempt, prune_rms=trial_prune)
            ids = generate(batch_n, trial, hard_chirality=hard_chirality)
            used += batch_n
            attempt += 1
            generated += len(ids)
            if ref_core is not None:
                graft_frozen(mol, ids, frozen, ref_core)
            matches = [
                cid
                for cid in ids
                if _seed_stereo_matches(
                    mol,
                    cid,
                    iso,
                    targets,
                    winding_ranks,
                    eta2_ranks,
                    reflectable,
                    ligand_stereo,
                )
            ]
            matches.sort(key=lambda cid: not _seed_geometry_matches(mol, cid, iso))
            for cid in matches:
                kept.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
                if kept.GetNumConformers() == target:
                    break
        mol = kept
        ids = [conf.GetId() for conf in mol.GetConformers()]
        logger.log(
            logging.WARNING if len(ids) < target else logging.DEBUG,
            "DG search returned %d candidates; %d/%d satisfied requested stereo selection",
            generated,
            len(ids),
            target,
        )
    if stereo_bonds:
        mol, _metals = _metal.surrogate_all_metals(mol)  # UFF still receives the bondless surrogate graph
    return mol, ids, target if needs_selection else None


def require_seed_count(ids, target, iso=None):
    """Reject public results that underfill the requested discrete stereo identity."""
    if target is not None and len(ids) < target:
        identity = f" for {iso}" if iso is not None else ""
        raise RuntimeError(
            f"embed: found {len(ids)}/{target} seeds with the requested stereo state{identity} after "
            "the bounded DG seed budget; choose a compatible isomer or relax fix="
        )


def _mirror_is_free(mol):
    """Return True when reflection would invert only the metal centre.

    Atomic and axial stereo forbid reflection; E/Z does not. Explicit chiral tags are checked because removing
    M-L bonds can hide a donor stereocentre from perception.
    """
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return False
    return not any(e.type != Chem.StereoType.Bond_Double for e in Chem.FindPotentialStereo(mol))


def _reflection_is_free(mol, iso, cons, targets, stereo_bonds=None):
    """Return True when reflecting `mol` would invert only one metal centre's otherwise-free hand.

    Needs exactly one stereo target with a hand, no stated winding or haptic face, no other stereocentre
    or configured double bond (`_mirror_is_free`), no stereo-carrier donor bond, and no rigid coordinate
    hold. Shared by seed selection, before relax, and hand correction, after it.
    """
    if len(targets) != 1:
        return False
    state, _vertices, haptic, winding = targets[0]
    if stereo_bonds is None:
        stereo_bonds = _stereo_donor_bonds(mol, iso)
    return bool(
        state.hand
        and not winding
        and _mirror_is_free(mol)
        and not stereo_bonds
        and not cons.frozen
        and not cons.dihedrals
        and not haptic
    )


def _reflect(mol, cid):
    """Reflect conformer `cid` in place without changing its mirror-symmetric restraint energy."""
    conf = mol.GetConformer(int(cid))
    pos = conf.GetPositions()
    pos[:, 0] *= -1.0
    conf.SetPositions(pos)


def _realised_hand(mol, iso, cid):
    """Read a conformer's metal hand, or ``''`` when its stated geometry cannot be recovered.

    Spectator metals are masked so `from_geometry` reads the selected centre only.
    """
    one = Chem.Mol(mol, False, int(cid))
    for metal in _metal.metal_indices(one):
        if metal != iso.metal:
            one.GetAtomWithIdx(metal).SetAtomicNum(0)
    pos = one.GetConformer().GetPositions()
    if any(np.linalg.norm(pos[d] - pos[iso.metal]) < _EPS for d in iso.donors):
        return ""
    try:
        got = from_geometry(one)  # a single-conformer copy: `from_geometry` takes a Mol
    except (ValueError, IndexError):
        return ""  # a collapsed or otherwise unclassifiable sphere has no defensible hand
    return got.chirality if got.metal == iso.metal and got.geometry == iso.geometry else ""


@dataclass
class Conformers:
    """Carry one Mol, selected conformer ids and the constraints that shaped them.

    `.mol` returns the user-facing graph with any metal restored. `minimize()` mutates and chains; indexing
    returns another view over the same working Mol.
    """

    _mol: Chem.Mol
    ids: list
    cons: Constraints = field(default_factory=Constraints)
    iso: Isomer | None = None
    energies: dict = field(default_factory=dict)  # conformer id -> restrained-UFF or downstream score
    unrelaxed: list = field(default_factory=list, kw_only=True)
    # ids whose UFF endpoint did not converge, or which were restored because no endpoint passed the contract
    relax_failures: dict = field(default_factory=dict, kw_only=True)
    # last restrained-UFF endpoint reason by conformer id; retained when a later seed fails a different gate
    uff_surrogates: dict = field(default_factory=dict, kw_only=True)  # atom -> (real Z, private UFF Z)
    uff_retyped_bonds: set = field(default_factory=set, kw_only=True)  # private donor -> acceptor dative edges
    # what reproduces this batch's seeds; None means search-free (minimize(), wrap()), so replacement is off
    params: EmbedParams | None = field(default=None, kw_only=True)
    # seed plus snapshots from the accepted restrained-UFF attempt; recorded only when explicitly requested
    trajectory: Chem.Mol | None = field(default=None, kw_only=True)
    # memoized (isomer identity, its per-centre coordination-state prep); invalidated when `iso` is reassigned
    _coord_state_prep: tuple = field(default=(None, None), init=False, repr=False, compare=False)

    def _coordination_states(self, iso):
        """Return `iso`'s per-centre coordination-state prep, cached while `iso`'s identity is unchanged."""
        cached_iso, prep = self._coord_state_prep
        if cached_iso is not iso:
            prep = _coordination_state_prep(iso)
            self._coord_state_prep = (iso, prep)
        return prep

    def _restored_mol(self, cid=None):
        """Return tracked conformers, or one requested conformer, on the restored public graph."""
        if cid is None:
            mol = Chem.Mol(self._mol)  # never mutate the working surrogate
            tracked = set(self.ids)
            for conf in list(mol.GetConformers()):
                if conf.GetId() not in tracked:
                    mol.RemoveConformer(conf.GetId())
        else:
            mol = Chem.Mol(self._mol, False, int(cid))
        if self.iso is None:
            return mol
        self.iso.restore(mol)  # real element and formal charge, on our copy
        if self.iso.donor_bonds:
            mol = _metal.connect_metal(mol, self.iso.donor_bonds)
        missing = {
            donor
            for donor, _metal_idx in _stereo_donor_bonds(mol, self.iso)
            if mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        }
        if missing:
            _stereo.apply_point_stereo(mol, self.iso.stereo_label, missing)
        return mol

    @property
    def mol(self):
        """Return the tracked conformers on a copy with the real metal graph restored."""
        return self._restored_mol()

    def _store_trajectory(self, frames):
        """Store accepted cleanup frames on the restored public graph."""
        if frames is None:
            return
        if len(self.ids) != 1 or not frames:
            self.trajectory = None
            return
        final = self._mol.GetConformer(int(self.ids[0])).GetPositions()
        if not np.allclose(frames[-1], final):
            self.trajectory = None
            logger.warning("trajectory: final geometry was replaced after UFF; no continuous path was retained")
            return
        mol = self.mol
        mol.RemoveAllConformers()
        for positions in frames:
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetPositions(positions)
            mol.AddConformer(conf, assignId=True)
        self.trajectory = mol

    def _fixed_geometry_misses(self, cid):
        """Return numeric fixes missed by `cid`, including their excess over the public tolerance."""
        conf = self._mol.GetConformer(cid)
        pos = conf.GetPositions()
        misses = []
        for atoms, (lo, hi) in self.cons.fixed.items():
            actual = constraint_value(pos, atoms, self.cons.haptic, (lo, hi))
            tol, unit = (FIX_DISTANCE_TOL, "A") if len(atoms) == _DIST_ATOMS else (FIX_ANGLE_TOL, "deg")
            excess = abs(actual - lo) - tol if lo == hi else max(lo - actual, actual - hi)
            if not np.isfinite(actual):
                excess = float("inf")
            if excess > 0.0:
                misses.append((excess, atoms, actual, lo, hi, tol, unit))
        return misses

    def _remove(self, ids):
        """Stop tracking conformers and their attached status records."""
        removed = set(ids)
        self.ids = [cid for cid in self.ids if cid not in removed]
        self.unrelaxed = [cid for cid in self.unrelaxed if cid not in removed]
        for cid in removed:
            self.energies.pop(cid, None)

    def _report_missed_fixes(self, ids, operation):
        """Report conformers that could not retain a numeric fix and return their ids."""
        failures = {cid: miss for cid in ids if (miss := self._fixed_geometry_misses(cid))}
        bad = list(failures)
        if bad:
            _excess, atoms, actual, lo, hi, tol, unit = max(
                (miss for misses in failures.values() for miss in misses), key=lambda miss: miss[0]
            )
            requested = f"{lo:.6f} +/- {tol:g}" if lo == hi else f"[{lo:.6f}, {hi:.6f}]"
            kind = {2: "distance", 3: "angle", 4: "dihedral"}[len(atoms)]
            logger.warning(
                "%s: dropped %d conformer(s) missing numeric fix; worst %s %s requested %s %s, got %.6f %s",
                operation,
                len(bad),
                kind,
                atoms,
                requested,
                unit,
                actual,
                unit,
            )
        return bad

    def _required_failure(self, failures):
        """Return failed core requests that require acceptance to restore the starting count."""
        kinds = {failure.kind for failure in failures if isinstance(failure, Failure)}
        required = []
        if "numeric_fix" in kinds:
            required.append("fix=")
        if kinds & {"metal_state", "coordination_shape"}:
            required.append("the requested metal state")
        if kinds & {"ligand_stereo", "ligand_stereo_unassigned"}:
            required.append("requested ligand stereo")
        return tuple(required)

    def _geometry_failure(self, cid, iso=None):
        """Return the first failed core publication contract, or ``None``."""
        if self._fixed_geometry_misses(cid):
            return Failure("numeric_fix", "missed numeric fix")
        # Seed selection and publication share ligand integrity; metal and stated pairs are already exempt.
        if bonding := _bonding_failure(self._mol, cid, exclude=self.cons.frozen, constrained=self.cons.distances):
            return Failure("bonding", f"heavy-atom bonding/clash failure ({bonding})")
        pos = self._mol.GetConformer(int(cid)).GetPositions()
        for body in self.cons.shapes:
            for (i, j), (lo, hi) in self.cons.distances.items():
                if i in body and j in body:
                    value = float(np.linalg.norm(pos[i] - pos[j]))
                    if value < lo - _SHAPE_TEAR_TOL or value > hi + _SHAPE_TEAR_TOL:
                        return Failure("rigid_body", "torn rigid body")
        active = self.iso if iso is None else iso
        if reason := _coordination_state_failure(self._mol, cid, active, states=self._coordination_states(active)):
            return reason
        if active is not None:
            restored = _restore_connected(self._mol, cid, active)
            intended_donors = [donor for donor, _metal_idx in active.donor_bonds]
            if self.cons.donor_orientation:
                _donor_facing_failure(restored, intended_donors, self.cons.frozen)
            if reason := _ligand_stereo_failure(restored, active):
                return reason
        return _structural_failure(self._mol, cid, self.cons)

    def _relax_failures(self, donor_hands):
        """Return failed endpoint contracts without changing coordinates or optimizer status."""
        failures = {}
        # A wrong face must not stop the ladder before a later rung can retain it. Free metal mirrors have
        # already been corrected; no winding means no added check.
        targets = _stereo_targets(self.iso)
        states = self._metal_states() if any(winding for _state, _vertices, _haptic, winding in targets) else {}
        for cid in self.ids:
            reason = self._geometry_failure(cid)
            if reason is None and not states.get(cid, True):
                reason = Failure("metal_state", "requested metal hand or haptic winding was not retained")
            hand_ok = all(
                target is None or _metal.donor_chirality_sign(self._mol, cid, donor, references) == target
                for donor, (target, references) in donor_hands[cid].items()
            )
            if not hand_ok:
                if reason is None:
                    reason = Failure("donor_hand", "coordinated ligand hand changed")
            if reason is not None:
                failures[cid] = reason
        return failures

    def _seed_donor_hands(self, conf_ids):
        """Record labile coordinated-donor hands before UFF, which does not read chiral tags."""
        donors = [donor for donor, _metal_idx in _stereo_donor_bonds(self._mol, self.iso)]
        donor_bonds = self.iso.donor_bonds if self.iso is not None else []
        references = {donor: [metal for site, metal in donor_bonds if site == donor] for donor in donors}
        return {
            cid: {
                donor: (
                    _metal.donor_chirality_sign(self._mol, cid, donor, references[donor]),
                    references[donor],
                )
                for donor in donors
            }
            for cid in conf_ids
        }

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, operation="minimize", _frames=None):
        """Retry unresolved endpoints from their seeds, then restore any that never pass the contract."""
        relaxing = list(self.ids)
        if not relaxing:
            return np.array([])
        self.energies = {}
        self.relax_failures = {}
        seed_pos = {cid: self._mol.GetConformer(cid).GetPositions().copy() for cid in relaxing}
        donor_hands = self._seed_donor_hands(relaxing)
        self.unrelaxed = [cid for cid in self.unrelaxed if cid not in seed_pos]
        pending = relaxing
        restored, escalated = {}, 0
        highest_mult = FC_ESCALATION[0]
        force_field_error = None
        accepted_frames = []
        first = relaxing[0]

        rungs = (*FC_ESCALATION, *_HIGH_FC_ESCALATION)
        for mult in rungs:
            if not pending:
                break
            current = pending
            for cid in current:
                self._mol.GetConformer(cid).SetPositions(seed_pos[cid])
            statuses = {}
            snapshots = {} if _frames is not None else None
            try:
                restrained_uff(
                    self._mol,
                    self.cons,
                    stiffness=stiffness * mult,
                    max_iters=max_iters,
                    conf_ids=current,
                    _snapshots=snapshots,
                    _statuses=statuses,
                    _surrogates=self.uff_surrogates,
                    _retyped=self.uff_retyped_bonds,
                )
            except UFFTypingError as err:
                for cid in current:
                    self._mol.GetConformer(cid).SetPositions(seed_pos[cid])
                    restored[cid] = Failure("uff", "UFF unavailable")
                self.relax_failures = dict(restored)
                self.unrelaxed.extend(current)
                logger.warning(
                    "%s: UFF could not relax this system (%s); kept the starting geometries",
                    operation,
                    _error_summary(err),
                )
                force_field_error = err
                break

            active = replace(self, ids=current)
            active._correct_metal_hand(operation, _snapshots=snapshots)
            failed = active._relax_failures(donor_hands)
            accepted = set(current).difference(failed)
            # A converged force field that settled on another coordination shape or slot seating is in a
            # different basin. Increasing the same objective's stiffness cannot change that basin; sample a
            # fresh DG seed instead. Keep handedness and structural failures on the normal escalation ladder.
            terminal_shape = {
                cid
                for cid, reason in failed.items()
                if statuses.get(cid, 1) == 0 and _stable_coordination_failure(reason)
            }
            # A donor-hand change is a geometric endpoint failure and may recover; an opposite ligand
            # stereoisomer is never chased by a stronger wall.
            pending = [
                cid
                for cid, reason in failed.items()
                if cid not in terminal_shape
                if mult != rungs[-1]
                and (mult < FC_ESCALATION[-1] or _high_force_candidate(reason))
                # A metal restraint that already tears an internal ligand bond cannot be repaired by
                # increasing that restraint.  Keep organic constrained retries and fresh metal seeds;
                # only the competing metal objective is terminal here.
                and not (
                    self.iso is not None and isinstance(reason, Failure) and reason.kind in {"bonding", "rigid_body"}
                )
            ]
            rejected = {cid: reason for cid, reason in failed.items() if cid not in pending}
            restored.update(rejected)
            for cid in rejected:
                self._mol.GetConformer(cid).SetPositions(seed_pos[cid])
            self.unrelaxed.extend(
                cid for cid in current if cid not in pending and (cid in failed or statuses.get(cid, 1))
            )
            if first in accepted and snapshots is not None:
                accepted_frames = snapshots.get(first, [])
            if mult != FC_ESCALATION[0] and accepted:
                escalated += len(accepted)
                highest_mult = mult

        if _frames is not None:
            final = self._mol.GetConformer(first).GetPositions().copy()
            path = (
                []
                if accepted_frames is None
                else [seed_pos[first], *(frame[: self._mol.GetNumAtoms()] for frame in accepted_frames)]
            )
            if path and not np.allclose(path[-1], final):
                path.append(final)
            _frames[:] = path

        self.relax_failures = dict(restored)

        if escalated:
            logger.info(
                "%s: %d/%d conformer(s) needed restraint stiffness up to %gx",
                operation,
                escalated,
                len(relaxing),
                highest_mult,
            )
        if restored and force_field_error is None:
            counts = Counter(restored.values())
            detail = ", ".join(f"{count}x {reason}" for reason, count in counts.items())
            logger.warning(
                "%s: rejected %d/%d UFF endpoints (%s); restored seeds before final acceptance",
                operation,
                len(restored),
                len(relaxing),
                detail,
            )
        if force_field_error is not None:
            return None
        return restrained_uff(
            self._mol,
            self.cons,
            stiffness=stiffness,
            max_iters=0,
            conf_ids=relaxing,
            _surrogates=self.uff_surrogates,
            _retyped=self.uff_retyped_bonds,
        )

    def _rescore_restrained(self, stiffness):
        """Record comparable restrained-UFF single points for settled conformers."""
        settled = [cid for cid in self.ids if cid not in self.unrelaxed]
        if not settled:
            self.energies = {}
            return
        e = restrained_uff(
            self._mol,
            self.cons,
            stiffness=stiffness,
            max_iters=0,
            conf_ids=settled,
            _surrogates=self.uff_surrogates,
            _retyped=self.uff_retyped_bonds,
        )
        self.energies = {i: float(v) for i, v in zip(settled, e, strict=True)}

    def _metal_hands(self):
        """Return ``{conformer id: realised metal-centre hand}`` over the tracked conformers."""
        mol = self.mol  # bind once: the finalize copies the whole molecule
        return {c: _realised_hand(mol, self.iso, c) for c in self.ids}

    def _metal_states(self):
        """Return whether each conformer realises every stated metal stereo target."""
        mol = self.mol
        targets = _stereo_targets(self.iso)
        ranks, eta2 = _winding_ranks(mol, targets)
        return {c: _seed_stereo_matches(mol, c, self.iso, targets, ranks, eta2, False) for c in self.ids}

    def _replace_failed(
        self, failed, stiffness, max_iters, *, operation="minimize", template=None, validator=None, params=None
    ):
        """Replace failed conformers from bounded fresh seed batches; return ids not replaced.

        Replacements keep the original ids. Each batch receives a constraint copy because embedding may add
        encounter bounds or move phantom indices. ``validator`` may add pipeline publication checks after the
        core contract; it never controls how a replacement is generated or relaxed.
        Rank settled survivors by their endpoint scores; keep unscored fallbacks in generation order.
        """
        params = self.params if params is None else params
        if params is None:
            return list(failed)
        left = list(failed)
        source = self._mol if template is None else template
        for batch_index in range(_REPLACEMENT_ROUNDS):
            if not left:
                break
            count = _REPLACEMENT_FACTOR * len(left)
            trial_params = replace(params, seed=params.seed + 1 + batch_index)
            cons = self.cons.copy()
            mol, ids, _target = seed_conformers(Chem.Mol(source), cons, self.iso, count, trial_params)
            if not ids:
                continue
            batch = Conformers(mol, ids, cons, self.iso, params=trial_params)
            # A constrained replacement must use the same stiffness ladder as the original batch.  A single
            # base-force pass can reject a seed that a higher rung would satisfy, while silently making fresh
            # seeds weaker than the batch they replace.
            if batch.cons.is_constrained:
                batch._relax_constrained(stiffness, max_iters, operation)
            else:
                batch._relax_once(stiffness, max_iters, operation)
            self.relax_failures.update(
                {("replacement", batch_index, cid): reason for cid, reason in batch.relax_failures.items()}
            )
            failures = batch._acceptance_failures(operation, validator=validator)
            batch_rejected = {cid for rejected in failures.values() for cid in rejected}
            if batch_rejected == set(batch.ids) and batch_rejected:
                reasons = [reason for reason, rejected in failures.items() if rejected]
                if (
                    len(reasons) == 1
                    and _stable_coordination_failure(reasons[0])
                    and any(_stable_coordination_failure(reason) for reason in self.relax_failures.values())
                ):
                    return left
            batch._remove({cid for rejected in failures.values() for cid in rejected})
            self.uff_surrogates.update(batch.uff_surrogates)
            self.uff_retyped_bonds.update(batch.uff_retyped_bonds)
            scores = {
                cid: energy
                for cid, energy in batch.energies.items()
                if cid not in batch.unrelaxed and np.isfinite(energy)
            }
            order = sorted(batch.ids, key=lambda cid: scores.get(cid, math.inf))
            for cid, src in zip(list(left), order, strict=False):
                pos = batch._mol.GetConformer(int(src)).GetPositions()
                self._mol.GetConformer(int(cid)).SetPositions(pos)
                if cid not in self.ids:
                    self.ids.append(cid)
                self.energies.pop(cid, None)
                if src in batch.energies:
                    self.energies[cid] = batch.energies[src]
                if src in batch.unrelaxed:
                    if cid not in self.unrelaxed:
                        self.unrelaxed.append(cid)
                elif cid in self.unrelaxed:
                    self.unrelaxed.remove(cid)
                left.remove(cid)
        return left

    def _correct_metal_hand(self, operation="minimize", *, _snapshots=None):
        """Reflect free metal inversions and return conformers that remain in the wrong state.

        Distance and angle constraints cannot choose a mirror; a signed dihedral can. The hand is read after
        relaxation, then corrected by reflection only when no stated geometry distinguishes the mirror.
        """
        iso = self.iso
        targets = _stereo_targets(iso)
        if iso is None or not self.ids or not targets:
            return []
        states = self._metal_states()
        wrong = [c for c, matches in states.items() if not matches]
        if not wrong:
            return []
        hands = self._metal_hands()
        reflected = 0
        # The mirror is free only where the metal centre is the one thing it inverts: no other stereocentre
        # (`_mirror_is_free`), no grafted core (its contract is the caller's exact geometry, and a chiral
        # core's mirror is a different core, so re-seeding re-grafts it instead), and no haptic face, which
        # can be planar-chiral in a way this tier cannot perceive (`pipeline.select_stereo` owns that).
        if _reflection_is_free(self._mol, iso, self.cons, targets):
            target_hand = targets[0][0].hand
            reflectable = [cid for cid in wrong if hands[cid] and hands[cid] != target_hand]
            for cid in reflectable:
                _reflect(self._mol, cid)
                if _snapshots is not None:
                    _snapshots[cid] = None  # Reflection is not a continuous UFF step.
            reflected = len(reflectable)
            if reflected and _snapshots is not None:
                logger.warning("trajectory: metal-hand reflection discarded the discontinuous UFF path")
            states = self._metal_states()
            wrong = [cid for cid in wrong if not states[cid]]
        logger.info(
            "%s: metal-state correction: %d reflected, %d unresolved",
            operation,
            reflected,
            len(wrong),
        )
        return wrong

    def _acceptance_failures(self, operation="minimize", validator=None, ids=None):
        """Return publication failures grouped by reason after free hand correction.

        ``ids`` narrows the per-conformer gate to a subset; metal-hand correction still scans every id.
        """
        failures = {}
        wrong = set(self._correct_metal_hand(operation))
        for cid in self.ids if ids is None else ids:
            if reason := self._geometry_failure(cid):
                failures.setdefault(reason, []).append(cid)
            elif cid in wrong:
                failure = Failure("metal_state", "requested metal hand or haptic winding was not retained")
                failures.setdefault(failure, []).append(cid)
            elif validator is not None and (reason := validator(self, cid)):
                failures.setdefault(reason, []).append(cid)
        return failures

    def _accept_relaxed(
        self,
        stiffness,
        max_iters,
        operation="minimize",
        *,
        validator=None,
        template=None,
        params=None,
        allow_replacement=True,
    ):
        """Search bounded replacements for publication failures, then reject unresolved ids.

        Optimizer status is metadata: an unrelaxed conformer may survive when its restored seed still satisfies
        the structural contract. Structurally invalid seeds and wrong metal states share the same fresh-seed
        replacement loop. Internal replacement batches call `_acceptance_failures` directly and cannot recurse.
        Embedding that rejects every relaxed candidate raises, even without a failed stereo or fix request.
        """
        target = len(self.ids)
        failures = self._acceptance_failures(operation, validator)
        requirements = list(self._required_failure(failures))
        failed = {cid for rejected in failures.values() for cid in rejected}
        replacement_params = self.params if params is None else params
        replaced = bool(failed and allow_replacement and replacement_params is not None)
        if replaced:
            self._replace_failed(
                failed,
                stiffness,
                max_iters,
                operation=operation,
                template=template,
                validator=validator,
                params=replacement_params,
            )
        if failed:
            # A replacement only touches `failed` ids; every other one's status cannot have changed since
            # the first pass, so keep its result rather than re-running the full gate over it again.
            failures = {
                reason: kept
                for reason, rejected in failures.items()
                if (kept := [c for c in rejected if c not in failed])
            }
            for reason, rejected in self._acceptance_failures(operation, validator, ids=failed).items():
                failures.setdefault(reason, []).extend(rejected)
            requirements.extend(
                requirement for requirement in self._required_failure(failures) if requirement not in requirements
            )
            failed = {cid for rejected in failures.values() for cid in rejected}
            missed = [
                cid
                for failure, rejected in failures.items()
                if isinstance(failure, Failure) and failure.kind == "numeric_fix"
                for cid in rejected
            ]
            self._report_missed_fixes(missed, operation)
            self._remove(failed)
        if not failures:
            return {}

        reason = ", ".join(f"{len(ids)}x {name}" for name, ids in failures.items())
        attempt = " after bounded fresh-seed replacement" if replaced else ""
        message = (
            f"{operation}: could not produce {target} conformer(s); {reason} remained{attempt}; "
            "the generated geometries did not satisfy the publication gate"
        )
        if self.relax_failures:
            previous = Counter(str(failure) for failure in self.relax_failures.values())
            rows = [f"{count}x {failure}" for failure, count in previous.items()]
            detail = ", ".join(rows[:4])
            if extra := rows[4:]:
                detail += f", {len(extra)} more"
            message += f"; restrained UFF endpoint failures: {detail}"
        if len(self.ids) < target and (requirements or (operation == "embed" and not self.ids)):
            requirement = " and ".join(requirements) or "the publication gate"
            raise EmbeddingError(f"{message}; could not satisfy {requirement}")
        logger.warning(message)
        return failures

    def _relax_once(self, stiffness, max_iters, operation="minimize", frames=None):
        """Run one same-seed relaxation policy without generating replacement seeds."""
        if not self.ids:
            return None
        self.energies = {}
        if not self.cons.is_constrained:
            statuses = {}
            try:
                e = restrained_uff(
                    self._mol,
                    self.cons,
                    stiffness=stiffness,
                    max_iters=max_iters,
                    conf_ids=self.ids,
                    _statuses=statuses,
                    _surrogates=self.uff_surrogates,
                    _retyped=self.uff_retyped_bonds,
                )
            except UFFTypingError as err:
                self.unrelaxed = list(self.ids)
                logger.warning(
                    "%s: UFF could not relax this system (%s); kept the starting geometries",
                    operation,
                    _error_summary(err),
                )
                return None
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
            self.unrelaxed = [i for i in self.ids if statuses.get(i, 1) != 0]
            return e
        self.unrelaxed = []
        e = self._relax_constrained(stiffness, max_iters, operation, frames)
        if e is not None:
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
        return e

    def minimize(self, stiffness=BASE_STIFFNESS, max_iters=MAX_ITERS):
        """Relax every conformer in place with restrained UFF; return ``self``.

        `stiffness` scales restraint penalties, not native UFF (see `relax.restrained_uff`). A constrained run
        retries missed geometry, a stated haptic face, or a wrong metal state with up to three bounded
        fresh-seed batches, then restores whatever still fails into `.unrelaxed`. An untypeable graph keeps
        its embedded geometry with no energy; a finite max-iterations endpoint that still satisfies the
        structural contract is kept too, just marked unrelaxed.

        `.energies` ranks settled results within one species only; a restored or unconverged conformer has
        none. `pipeline.Ensemble.minimize()` adds further acceptance gates on top.
        """
        self.trajectory = None
        if not self.ids:
            return self
        e = self._relax_once(stiffness, max_iters)
        self._accept_relaxed(stiffness, max_iters, allow_replacement=e is not None)
        if self.cons.is_constrained and (e is not None or self.energies):
            # Rescue and hand re-seeding may relax different conformers at different stiffnesses. Rank them
            # only after one single-point pass on the caller's stated objective.
            self._rescore_restrained(stiffness)
        return self

    def measure(self, atoms):
        """Return mean, minimum and maximum distance, angle or dihedral over tracked conformers.

        `atoms` contains two to four atom indices or SMARTS selectors. The result also includes the count as `n`.
        """
        if not self.ids:
            raise ValueError(
                "measure() on an empty result: embed/minimize may have produced no conformers (check the log)"
            )
        idx = tuple(resolve_atom(self._mol, a) for a in atoms)
        if len(idx) not in (_DIST_ATOMS, _ANGLE_ATOMS, _DIHEDRAL_ATOMS):
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        window = None
        if len(idx) == _DIHEDRAL_ATOMS:
            window = self.cons.dihedrals.get(min(idx, idx[::-1]))
        v = [constraint_value(self._mol.GetConformer(c).GetPositions(), idx, window=window) for c in self.ids]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}

    def xyz(self, conf_id=None):
        """Return conformer `conf_id` as an xyz block, or every tracked conformer as one multi-frame block.

        `conf_id` is an RDKit conformer id (what `.ids` holds), not a position, and must be one this result
        still tracks; otherwise a slice would emit a conformer it no longer owns.
        """
        if conf_id is not None and int(conf_id) not in [int(c) for c in self.ids]:
            raise ValueError(f"conformer id {conf_id} is not one of this result's ids {list(self.ids)}")
        mol = self.mol
        return "".join(Chem.MolToXYZBlock(mol, confId=int(c)) for c in (self.ids if conf_id is None else [conf_id]))

    def dump(self, path):
        """Write the tracked conformers to `path` as a multi-frame .xyz; return the path."""
        if not self.ids:  # a 0-byte file that reads as a successful write is the worst possible outcome
            raise ValueError("nothing to dump: this result has no conformers (the embed produced none)")
        with open(path, "w") as f:
            f.write(self.xyz())
        return path

    def dump_trajectory(self, path):
        """Write the captured restrained-UFF trajectory to `path`; return the path."""
        if self.trajectory is None or not self.trajectory.GetNumConformers():
            raise ValueError(
                "no trajectory retained: use rx.embed(..., n=1, trajectory=True); "
                "if requested already, check trajectory warnings"
            )
        with open(path, "w") as f:
            for conf in self.trajectory.GetConformers():
                f.write(Chem.MolToXYZBlock(self.trajectory, confId=conf.GetId()))
        return path

    def __getitem__(self, key):
        """Pick conformer(s) by position as a new `Conformers`: ``confs[0]``, ``confs[:3]``."""
        sel = self.ids[key]
        sel = sel if isinstance(sel, list) else [sel]
        return replace(
            self,
            ids=sel,
            energies={i: self.energies[i] for i in sel if i in self.energies},
            unrelaxed=[i for i in self.unrelaxed if i in sel],  # a slice must not silently lose these flags
            uff_surrogates=dict(self.uff_surrogates),
            uff_retyped_bonds=set(self.uff_retyped_bonds),
            trajectory=Chem.Mol(self.trajectory) if self.trajectory is not None and sel == self.ids else None,
        )

    def __len__(self):
        """Return the number of conformers in play."""
        return len(self.ids)

    def __repr__(self):
        """Summarise the result: conformer count and the isomer identity when there is one."""
        who = f", {self.iso}" if self.iso is not None else ""
        return f"<Conformers: {len(self.ids)} conformer{'s' if len(self.ids) != 1 else ''}{who}>"


def _check_bare_mol(mol):
    """Reject implicit hydrogens or a metal Mol that bypassed the Isomer surrogate."""
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        raise ValueError("embed() needs a Mol with explicit hydrogens; pass Chem.AddHs(mol)")
    metals = _metal.metal_indices(mol)
    if metals:
        raise ValueError(
            f"atom(s) {metals} are metal centres: a metal is embedded through a SURROGATE, because neither "
            f"the bounds matrix nor UFF describes one directly. Route the complex through "
            f"Isomer(mol, geometry, sites) or enumerate_isomers(mol, geometry), which own that surrogate and "
            f"the coordination model with it"
        )


def embed(spec, *, fix=None, constrain=None, template=None, n=None, seed=None, threads=None, params=None):
    """Embed conformers of `spec` under ``fix``/``constrain``; return a `Conformers`.

    `fix`/`constrain` keys are 0-based atom indices in `spec`'s own order. The full vocabulary and how the two
    verbs compose is the `constraints` module docstring. For an `Isomer`, these are raw DG seeds: the
    coordination-state gate that names and records a conformer's shape (`SHAPE_PROP`) runs inside
    `minimize()`, not here.

    Parameters
    ----------
    spec : Mol | Isomer
        An RDKit `Mol` with explicit Hs (parsing and Hs are the caller's job), or a metal `Isomer`, whose
        polyhedron is composed with the spec so a substrate binds a named coordination sphere.
    fix, constrain : list | dict, optional
        Rigid vs soft geometric holds; the `constraints` module docstring has the full vocabulary::

            embed(ts_mol, fix=[3, 7, 11, 12])   # graft a reacting core at its own coords, 0.000 Å
            embed(mol, fix={(3, 11): 2.05})     # state the forming bond; the rest is free
    template : tuple, optional
        ``(reference, SMARTS_or_map)``. A SMARTS must have one ordered match on both molecular graphs; use an
        explicit ``{target_index: reference_index}`` map for a symmetric core or coordinate array.
    n : int, optional
        Conformer count. ``None`` means a flexibility-scaled count (`bounds.seed_count`), not RDKit's 10.
    seed, threads : optional
        Convenience for `params.seed` / `params.threads` when every other `EmbedParams` field can stay at its
        default; give at most one of ``seed``/``threads`` or ``params``, not both.
    params : EmbedParams, optional
        The full sampling, native-model and ablation switches; `bounds.resolve_params` folds `seed`/`threads`
        into one when `params` is omitted.
    """
    if not isinstance(spec, (Chem.Mol, Isomer)):
        raise TypeError(
            f"embed() takes an RDKit Mol or an Isomer, got {type(spec).__name__}; parse a SMILES / .xyz "
            f"yourself (perception is the caller's job) and add Hs"
        )
    params = resolve_params(params, seed, threads)
    if template is not None:  # sugar: a reference core is a coordinate fix. Dissolved before anything routes.
        src = spec.mol if isinstance(spec, Isomer) else spec
        own = src.GetConformer().GetPositions() if src.GetNumConformers() else None
        fix = template_to_fix(template, fix, own, src)
    if not isinstance(spec, Isomer):
        _check_bare_mol(spec)
    mol, cons, iso, graft_ref = prepare(spec, fix=fix, constrain=constrain, params=params)
    mol, ids, target = seed_conformers(mol, cons, iso, n, params, graft_ref=graft_ref)
    require_seed_count(ids, target, iso)
    n_frag = len(Chem.GetMolFrags(mol))
    if not ids:  # ETKDG met no constraint set it could realise; silence reads as "embedded, then pruned"
        logger.warning(
            "embed: no conformer under %d constraint(s); the spec may not be realisable",
            len(cons.distances) + len(cons.angles) + len(cons.dihedrals),
        )
    logger.info(
        "embed[%s]: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        str(iso) if iso is not None else "molecule",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles) + len(cons.dihedrals),
    )
    return Conformers(mol, ids, cons, iso, params=params)


def prepare(spec, *, fix=None, constrain=None, external=None, params=None):
    """Prepare a copied Mol and its constraints for embedding or relaxation.

    Return ``(mol, constraints, isomer, graft_reference)``. A metal geometry becomes one retained `Isomer`;
    a coordinate-free metal must arrive as an explicitly selected `Isomer`. `external` exposes downstream
    constraint ownership to model compilation without changing how those constraints are applied. `params`
    supplies the `donor_orientation`/`conjugation` switches; both default True when `params` is omitted.
    """
    iso = spec if isinstance(spec, Isomer) else None
    mol = Chem.Mol(iso.mol if iso is not None else spec)  # our own copy: never the caller's conformers
    has_geometry = mol.GetNumConformers() > 0
    if iso is None and _metal.metal_index(mol) is not None and has_geometry:
        centre = "all" if len(_metal.metal_indices(mol)) > 1 else None
        iso = from_geometry(mol, center=centre)
        mol = Chem.Mol(iso.mol)
    elif iso is None and _metal.metal_index(mol) is not None:
        raise ValueError(
            "a coordinate-free metal needs a selected coordination state; pass an Isomer from "
            "enumerate_isomers()/rx.metal(), or use the pipeline metal=<geometry> sugar"
        )
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=has_geometry)
    donor_orientation = True if params is None else params.donor_orientation
    conjugation = True if params is None else params.conjugation
    if iso is not None:
        graft_ref = {**iso._graft_ref, **ref}
        cons = fold_substrate(
            iso._constraints(cons, external, donor_orientation=donor_orientation, conjugation=conjugation),
            cons,
            graft_ref,
            protect_arrangement=bool(iso._constrained_metals) and iso._protect_arrangement,
        )
        ref = graft_ref
    else:
        cons = cons.copy(donor_orientation=donor_orientation, conjugation=conjugation)
    return mol, cons, iso, ref


def prepare_relax(spec, *, fix=None, constrain=None):
    """Prepare a copied geometry for search-free relaxation.

    Returns ``(mol, ids, constraints, isomer)``. Coordinate-fixed atoms are grafted exactly.
    """
    mol, cons, iso, ref = prepare(spec, fix=fix, constrain=constrain)
    ids = [c.GetId() for c in mol.GetConformers()]
    if ref:
        graft = sorted(ref)
        graft_frozen(mol, ids, graft, np.array([ref[i] for i in graft]))
    return mol, ids, cons, iso


def minimize(spec, *, fix=None, constrain=None, template=None, stiffness=None):
    """Relax an existing geometry toward ``fix`` or ``constrain`` targets without searching.

    The input must be a Mol or Isomer with a conformer. An Isomer retains its coordination constraints.
    """
    if not isinstance(spec, (Chem.Mol, Isomer)):
        raise TypeError(f"minimize() takes an RDKit Mol or an Isomer, got {type(spec).__name__}")
    if template is not None:
        src = spec.mol if isinstance(spec, Isomer) else spec
        own = src.GetConformer().GetPositions() if src.GetNumConformers() else None
        fix = template_to_fix(template, fix, own, src)
    if not (spec.mol if isinstance(spec, Isomer) else spec).GetNumConformers():
        raise ValueError(
            "minimize() relaxes an existing geometry; give a Mol with a conformer (embed() first if you have "
            "only a graph)"
        )
    mol, ids, cons, iso = prepare_relax(spec, fix=fix, constrain=constrain)
    confs = Conformers(mol, ids, cons, iso)
    return confs.minimize() if stiffness is None else confs.minimize(stiffness=stiffness)
