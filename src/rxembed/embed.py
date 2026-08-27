"""Embed an RDKit Mol or metal Isomer under explicit constraints.

`fix` is rigid and `constrain` is releasable. Edited ETKDG bounds seed coordinates, an exact fixed core is
grafted back, and `Conformers.minimize()` applies restrained UFF. Parsing and NCI discovery belong upstream.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from . import metal_core as _metal
from . import metal_polyhedron as _poly
from . import metal_stereo as _metal_stereo
from .bounds import DEFAULT_SEED, probe_conformer, seed_coordinates, seed_count
from .constraints import (
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    Constraints,
    _central_bond,
    _structural_torsion_bonds,
    compose,
    constraint_value,
    resolve_atom,
    resolve_core,
    template_to_fix,
    within_window,
)
from .metal_core import VACANT, materialized_state
from .metal_isomer import Isomer, from_geometry
from .relax import MAX_ITERS, _error_summary, bonding_ok, restrained_uff

logger = logging.getLogger("rxembed")  # configured by rxembed.set_verbose


_EPS = 1e-6  # near-zero norm floor for the graft axis
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation
_ANGLE_ATOMS = 3
_DIHEDRAL_ATOMS = 4
_STRUCT_ANGLE_SLACK = 0.05  # deg: numerical settling tolerance at a finite-force orientation wall
_STRUCT_CAP_SLACK = 2.0  # deg: finite-force settling tolerance on coplanarity and umbrella caps
_SHAPE_TEAR_TOL = 0.10  # A beyond a rigid body's own 0.1 A all-pairs windows
_SPATIAL_DIMENSION = 3


def _structural_dihedrals_ok(conf, cons):
    """Return whether graph-derived coplanarity and umbrella terms pass their structural caps."""
    pos = conf.GetPositions()
    stated = {_central_bond(atoms) for atoms in cons.dihedrals}
    for key in cons.coplanar:
        atoms, cap = key[:4], key[5]
        if _central_bond(atoms) in stated:
            continue
        measured = constraint_value(pos, atoms, cons.haptic)
        if measured is None or not np.isfinite(measured):
            return False
        value = abs(measured)
        if min(value, 180.0 - value) > cap + _STRUCT_CAP_SLACK:
            return False
    for atoms, ideal in cons.umbrellas.items():
        if ideal is None or _central_bond(atoms) in stated:
            continue
        measured = constraint_value(pos, atoms, cons.haptic)
        if measured is None or not np.isfinite(measured):
            return False
        value = abs(measured)
        if not ideal - _STRUCT_CAP_SLACK <= value <= 90.0 + _STRUCT_CAP_SLACK:
            return False
    return True


def _structural_constraints_ok(mol, cid, cons):
    """Return whether one conformer satisfies the coordination terms that are structural postconditions."""
    conf = mol.GetConformer(int(cid))
    pos = conf.GetPositions()
    soft_d, soft_angular = cons.contacts

    for key, window in cons.distances.items():
        if key in soft_d or not cons.metals.intersection(key):
            continue
        value = constraint_value(pos, key, cons.haptic, window)
        if not within_window(value, window, FIX_DISTANCE_TOL):
            return False
    for key, window in cons.angles.items():
        # D-M-D walls bias ideal seating, but a chelate or haptic face can be a valid distorted polyhedron.
        # `_coordination_state_ok` judges those angles together with donor-to-slot correspondence.
        if key in soft_angular or key[1] in cons.metals or not cons.metals.intersection(key):
            continue
        value = constraint_value(pos, key, cons.haptic, window)
        if not within_window(value, window, _STRUCT_ANGLE_SLACK):
            return False
    return _structural_dihedrals_ok(conf, cons)


def _coordination_state_ok(mol, cid, iso):
    """Return whether one conformer retains each selected polyhedron."""
    if iso is None:
        return True
    pos = mol.GetConformer(int(cid)).GetPositions()
    parts = _metal.materialized_states(iso.mol, iso.centres)
    for state in iso.centres:
        poly = _poly.POLYHEDRA.get(state.geometry)
        if poly is None:
            continue
        vertices, haptic, _winding, donors = parts[state.atom]
        occupied = [i for i, vertex in enumerate(vertices) if vertex != VACANT]
        if len(occupied) < 2:  # noqa: PLR2004 - one site has no relative direction and cannot define a shape
            continue
        if not np.all(np.isfinite(pos[state.atom])):
            return False
        points = {}
        for slot in occupied:
            vertex = vertices[slot]
            point = np.mean(pos[list(haptic[vertex])], axis=0) if vertex in haptic else pos[vertex]
            if not np.all(np.isfinite(point)):
                return False
            points[slot] = point

        if poly.planar and donors and not _metal.coplanar(pos, state.atom, donors, haptic=haptic):
            return False
        if (
            not poly.planar
            and len(occupied) >= _SPATIAL_DIMENSION
            and np.linalg.matrix_rank(np.array([poly.vertex_dirs[i] for i in occupied], float)) == _SPATIAL_DIMENSION
            and _metal.coplanar(pos, state.atom, donors, haptic=haptic)
        ):
            return False
        observed = []
        for slot in occupied:
            direction = points[slot] - pos[state.atom]
            if not np.all(np.isfinite(direction)):
                return False
            length = np.linalg.norm(direction)
            if length <= _EPS:
                return False
            observed.append(direction / length)
        ideal = np.array([poly.vertex_dirs[slot] for slot in occupied], float)
        ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
        if _poly.ordered_fit_residual(np.array(observed), ideal) > _metal._FIT_FLOOR:
            return False
    return True


_MIN_FRAGS = 2  # below this there is no inter-fragment separation to enforce
# Half-order steps find the lowest restraint stiffness that keeps the sphere intact.
BASE_STIFFNESS = 1.0
FC_ESCALATION = (1.0, 3.0, 10.0, 30.0, 100.0)
# Metal relaxations permit a 1.5x surrogate/tight-bite stretch; coplanarity is gated separately.
METAL_BOND_TOL = 1.5
BOND_TOL = 1.3
# A fresh seed re-rolls the metal hand, so retries are bounded and over-embed by two.
_MAX_HAND_ROUNDS = 5
_HAND_BUFFER = 2


def encounter_bounds(mol, slack=1.5, seed=DEFAULT_SEED):
    """Keep each fragment pair near van der Waals contact through its closest heavy atoms."""
    tmp = probe_conformer(mol, seed)
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


def float_encounter_bounds(mol, cons):
    """Apply encounter bounds only to fragments not already pinned by a constraint."""
    frags = Chem.GetMolFrags(mol)
    if len(frags) < _MIN_FRAGS:
        return {}
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    touched = {frag_of[a] for a in cons.constrained_atoms()}
    if len(touched) >= len(frags):  # every fragment already pinned/linked
        return {}
    return {
        k: v for k, v in encounter_bounds(mol).items() if frag_of[k[0]] not in touched or frag_of[k[1]] not in touched
    }


def graft_frozen(mol, conf_ids, frozen, ref):
    """Kabsch-fit and restore a frozen core exactly after distance geometry."""
    frozen = list(frozen)
    core = np.asarray(ref, float)  # input core, input frame
    if len(frozen) <= 1:  # a point has no shape to restore
        return
    if len(frozen) == _BOND_ATOMS:  # a bond: keep one atom, slide the other to the template distance along the
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


def _frozen_core_ref(mol, frozen_atoms, graft_ref):
    """Return ``(frozen, ref_core)``: atoms to Kabsch-graft and their exact target coords, ``None`` if none."""
    frozen = sorted(frozen_atoms)  # capture the input core BEFORE the embed
    if frozen and mol.GetNumConformers():  # graft each frozen atom to its explicit-fix coord, else its own coord
        own = mol.GetConformer().GetPositions()
        return frozen, np.array([graft_ref.get(i, own[i]) for i in frozen])
    if graft_ref:  # SMILES metal + explicit-coords fix: no conformer, graft only the named atoms
        frozen = sorted(graft_ref)
        return frozen, np.array([graft_ref[i] for i in frozen])
    return frozen, None


def fold_substrate(base, sub, graft_ref, *, protect_arrangement=True):
    """Compose substrate constraints with a coordination sphere.

    Field-driven composition keeps every constraint kind; the previous hand-written merge dropped `sub.planes`.
    A soft substrate grip may not replace a structural sphere hold. A numeric fix may override one explicitly.
    A coordinate graft may pin at most one sphere atom; two would silently replace the selected arrangement.
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
    sphere_angular = {canonical(k) for k in (*base.angles, *base.dihedrals)}
    structural_torsions = _structural_torsion_bonds(base)
    overlap = sorted(
        (soft_d & sphere_d)
        | {
            k
            for k in soft_angular
            if canonical(k) in sphere_angular or (len(k) == _DIHEDRAL_ATOMS and _central_bond(k) in structural_torsions)
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
    ranks = {}
    for state, vertices, haptic, winding in targets:
        donors = [d for d in vertices if d != VACANT and d not in haptic]
        donors += [atom for face in haptic.values() for atom in face]
        ranks[state.atom] = _metal_stereo.donor_classes(mol, donors) if winding else {}
    eta2 = any(
        len(haptic[dummy]) == _metal._ETA2 for _state, _vertices, haptic, winding in targets for dummy in winding
    )
    return ranks, list(Chem.ComputeAtomCIPRanks(mol)) if eta2 else None


def _seed_stereo_matches(mol, cid, targets, winding_ranks, eta2_ranks, reflectable):
    """Return whether one raw DG seed matches every stated metal hand and haptic winding."""
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
        if any(
            _metal_stereo.face_winding(mol, pos, state.atom, haptic[dummy], winding_ranks[state.atom], eta2_ranks)
            != sign
            for dummy, sign in winding.items()
        ):
            return False
    return True


def _stereo_batch_size(need, bits, reflectable):
    """Return a DG batch large enough to retain ``need`` selected stereoisomers."""
    if reflectable:
        return need
    # A 2 sqrt(N) spare-success margin avoids a coin-flip second batch at the exact 2^k mean.
    return 2**bits * (need + 2 * (math.isqrt(need - 1) + 1))


def _hold_donors(mol, iso, cons):
    """Cap every labile donor carried by an isomer and return the added atoms."""
    held = []
    if iso is None:
        return mol, held
    for metal in sorted({m for _d, m in iso.donor_bonds}):
        donors = [d for d, m in iso.donor_bonds if m == metal]
        mol, added = _metal._hold_donor_chirality(mol, metal, donors, cons)
        held.extend(added)
    return mol, held


def seed_conformers(mol, cons, iso, n, *, seed=DEFAULT_SEED, knowledge=True, prune_rms=0.1, threads=0, graft_ref=None):
    """Seed `n` conformers, selecting metal and donor hands before cleanup and grafting a fixed core exactly."""
    if int(seed) < 0:  # every embed route validates here, not just the front door: RDKit's -1 draws from the
        raise ValueError(  # global RNG, so the result silently depends on prior consumption (measured: two
            f"seed={seed}: a negative seed draws from RDKit's global RNG and is not reproducible"
        )  # identical rx.embed('CCO', seed=-1) calls gave different coordinates)
    if n is not None and int(n) <= 0:  # `n or seed_count(...)` below would treat 0 as "use the default"
        raise ValueError(f"n={n}: give a positive conformer count, or None for the flexibility-scaled default")
    cons.distances.update(float_encounter_bounds(mol, cons))  # keep free/stray fragments from drifting off
    frozen, ref_core = _frozen_core_ref(mol, cons.frozen, graft_ref or {})
    target = n or seed_count(mol, constrained=cons.is_constrained)
    targets = _stereo_targets(iso)
    hands = sum(bool(state.hand) for state, _vertices, _haptic, _winding in targets)
    windings = sum(len(winding) for _state, _vertices, _haptic, winding in targets)
    winding_ranks, eta2_ranks = _winding_ranks(mol, targets)
    reflectable = bool(
        len(targets) == 1
        and hands
        and not windings
        and _mirror_is_free(mol)
        and not cons.frozen
        and not cons.dihedrals
        and not targets[0][2]
    )
    mol, held = _hold_donors(mol, iso, cons)
    if not targets:
        ids = list(
            seed_coordinates(
                mol,
                cons,
                target,
                seed=seed,
                prune_rms=prune_rms,
                knowledge=knowledge,
                threads=threads,
            )
        )
        if ref_core is not None:
            graft_frozen(mol, ids, frozen, ref_core)  # restore the exact frozen core
    else:
        kept = Chem.Mol(mol)
        kept.RemoveAllConformers()
        for attempt in range(_MAX_HAND_ROUNDS):
            need = target - kept.GetNumConformers()
            if need <= 0:
                break
            stereo_bits = hands + windings
            batch_n = _stereo_batch_size(need, stereo_bits, reflectable)
            ids = list(
                seed_coordinates(
                    mol,
                    cons,
                    batch_n,
                    seed=seed + attempt,
                    prune_rms=-1 if len(targets) > 1 else prune_rms,
                    knowledge=knowledge,
                    threads=threads,
                )
            )
            if ref_core is not None:
                graft_frozen(mol, ids, frozen, ref_core)
            for cid in ids:
                if _seed_stereo_matches(mol, cid, targets, winding_ranks, eta2_ranks, reflectable):
                    kept.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
                    if kept.GetNumConformers() == target:
                        break
        mol = kept
        ids = [conf.GetId() for conf in mol.GetConformers()]
    if iso is not None:
        mol = _metal._release_donor_chirality(mol, held, cons)  # drop the dummy D's + cons keys, restore charges
    return mol, ids, target if targets else None


def require_seed_count(ids, target):
    """Reject public results that underfill the requested coordination identity."""
    if target is not None and len(ids) < target:
        raise RuntimeError(
            f"embed: found {len(ids)}/{target} seeds with the requested metal state after "
            f"{_MAX_HAND_ROUNDS} DG batches; choose a compatible metal identity or relax fix="
        )


def _mirror_is_free(mol):
    """Return True when reflection would invert only the metal centre.

    Atomic and axial stereo forbid reflection; E/Z does not. Explicit chiral tags are checked because removing
    M-L bonds can hide a donor stereocentre from perception.
    """
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return False
    return not any(e.type != Chem.StereoType.Bond_Double for e in Chem.FindPotentialStereo(mol))


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
    # ids restored to their embed seed because no stiffness produced a converged result that passed the contract
    seed: int | None = field(default=None, kw_only=True)
    # seed plus snapshots from the accepted restrained-UFF attempt; recorded only when explicitly requested
    trajectory: Chem.Mol | None = field(default=None, kw_only=True)

    @property
    def mol(self):
        """Return the tracked conformers on a copy with the real metal graph restored."""
        mol = Chem.Mol(self._mol)  # never mutate the working surrogate
        tracked = set(self.ids)
        for conf in list(mol.GetConformers()):
            if conf.GetId() not in tracked:
                mol.RemoveConformer(conf.GetId())
        if self.iso is None:
            return mol
        self.iso.restore(mol)  # real element and formal charge, on our copy
        return _metal.connect_metal(mol, self.iso.donor_bonds) if self.iso.donor_bonds else mol

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
            tol, unit = (FIX_DISTANCE_TOL, "A") if len(atoms) == _BOND_ATOMS else (FIX_ANGLE_TOL, "deg")
            excess = abs(actual - lo) - tol if lo == hi else max(lo - actual, actual - hi)
            if not np.isfinite(actual):
                excess = float("inf")
            if excess > 0.0:
                misses.append((excess, atoms, actual, lo, hi, tol, unit))
        return misses

    def _fixed_geometry_ok(self, cid):
        """Return whether every scalar target or explicit fixed window meets its contract."""
        return not self._fixed_geometry_misses(cid)

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
        """Name the failed core request that requires acceptance to restore the starting count."""
        required = []
        if "missed numeric fix" in failures:
            required.append("fix=")
        if "the requested metal state" in failures:
            required.append("the requested metal state")
        return " and ".join(required)

    def _coordination_ok(self, cid, iso=None):
        """Return whether every selected coordination state retains its shape."""
        return _coordination_state_ok(self._mol, cid, self.iso if iso is None else iso)

    def _geometry_failure(self, cid, iso=None):
        """Return the first failed core publication contract, or ``None``."""
        if not self._fixed_geometry_ok(cid):
            return "missed numeric fix"
        bond_tol = METAL_BOND_TOL if self.iso is not None or iso is not None else BOND_TOL
        if not bonding_ok(self._mol, cid, bond_tol=bond_tol, exclude=self.cons.frozen, constrained=self.cons.distances):
            return "broken bond"
        pos = self._mol.GetConformer(int(cid)).GetPositions()
        for body in self.cons.shapes:
            for (i, j), (lo, hi) in self.cons.distances.items():
                if i in body and j in body:
                    value = float(np.linalg.norm(pos[i] - pos[j]))
                    if value < lo - _SHAPE_TEAR_TOL or value > hi + _SHAPE_TEAR_TOL:
                        return "torn rigid body"
        if not self._coordination_ok(cid, iso):
            return "wrong coordination state"
        if not _structural_constraints_ok(self._mol, cid, self.cons):
            return "missed structural constraint"
        return None

    def _relax_ok(self, cid):
        """Return whether one relaxed conformer satisfies every structural acceptance gate."""
        return self._geometry_failure(cid) is None

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, operation="minimize", _frames=None):
        """Run the shared restrained-relax and per-conformer retry policy."""
        relaxing = list(self.ids)
        seed_pos = {cid: self._mol.GetConformer(cid).GetPositions().copy() for cid in relaxing}
        energies = self._relax_batch(stiffness, max_iters, relaxing, operation, _frames)
        retried = self._retry_relaxation(seed_pos, stiffness, max_iters, operation, _frames)
        if energies is not None and retried:
            energies = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=0, conf_ids=relaxing)
        return energies

    def _relax_batch(self, stiffness, max_iters, relaxing, operation, frames):
        """Relax with restrained UFF, escalating only when no conformer passes the accept gate.

        Return the accepted energies, or ``None`` when UFF cannot type the graph.
        """
        # Hold a labile (carbanion/amine) donor's hand through the relax: the surrogate's bare degree-3 centre
        # inverts under UFF. Cap it with a dummy D, release when done.
        n_atoms = self._mol.GetNumAtoms()
        initial = self._mol.GetConformer(relaxing[0]).GetPositions().copy() if frames is not None else None
        recorded, trajectory_done = [], False
        self._mol, hold = _hold_donors(self._mol, self.iso, self.cons)
        e, fc = None, stiffness
        try:
            retrying = set(relaxing)
            self.unrelaxed = [i for i in self.unrelaxed if i not in retrying]
            embed_pos = {c: self._mol.GetConformer(c).GetPositions().copy() for c in relaxing}
            donor_hands = {
                c: {donor: _metal.donor_chirality_sign(self._mol, c, donor) for _dummy, donor, _charge in hold}
                for c in relaxing
            }

            def restore(ids=relaxing):
                for cid in ids:
                    self._mol.GetConformer(cid).SetPositions(embed_pos[cid])

            statuses = {}
            for step, mult in enumerate(FC_ESCALATION):  # half-order steps: find the minimum sufficient stiffness
                if step:  # restore the embed geometry before a stiffer retry (a big jump over-stiffens it)
                    restore()
                fc = stiffness * mult
                snapshots = {} if frames is not None else None
                try:
                    e = restrained_uff(
                        self._mol,
                        self.cons,
                        stiffness=fc,
                        max_iters=max_iters,
                        conf_ids=relaxing,
                        _snapshots=snapshots,
                        _statuses=statuses,
                    )
                except RuntimeError as err:  # UFF can't build a force field for this graph, so keep the embed
                    restore()
                    trajectory_done = True
                    self.unrelaxed.extend(i for i in relaxing if i not in self.unrelaxed)
                    logger.warning(
                        "%s: UFF could not relax this system (%s); keeping the embedded geometry",
                        operation,
                        _error_summary(err),
                    )
                    return None
                # Accept only converged conformers satisfying the same structural contract every retry reads.
                kept = [i for i in relaxing if statuses.get(i, 1) == 0 and self._relax_ok(i)]
                if kept or step == len(FC_ESCALATION) - 1:  # some survived (accept), or out of steps (caller drops)
                    if snapshots is not None:
                        recorded = snapshots.get(relaxing[0], [])
                    if step and kept:
                        logger.info(
                            "%s: relax missed the accept gate; escalated restraint stiffness to %gx", operation, fc
                        )
                    elif step:  # exhausted: what comes back is the %gx relax, not the stiffness that was asked
                        logger.warning(  # for, and nothing else says so. Fail loud.
                            "%s: no stiffness up to %gx satisfied the accept gate; returning the ungated relax",
                            operation,
                            fc,
                        )
                    break
            self.unrelaxed.extend(i for i in relaxing if statuses.get(i, 1) != 0 and i not in self.unrelaxed)
            inverted = [
                c
                for c in relaxing
                if any(
                    target is not None and _metal.donor_chirality_sign(self._mol, c, donor) != target
                    for donor, target in donor_hands[c].items()
                )
            ]
            if inverted:
                restore(inverted)
                recorded = []  # the UFF path inverted the centre; the accepted result is the restored seed
                self.unrelaxed.extend(c for c in inverted if c not in self.unrelaxed)
                e = restrained_uff(self._mol, self.cons, stiffness=fc, max_iters=0, conf_ids=relaxing)
                logger.warning(
                    "%s: UFF inverted a coordinated ligand stereocentre in %d conformer(s); kept the DG seed",
                    operation,
                    len(inverted),
                )
            trajectory_done = True
        finally:  # release the hold on every exit path, including the early UFF-failure return; the dummy D
            if hold:  # is scaffolding for this relax alone
                self._mol = _metal._release_donor_chirality(self._mol, hold, self.cons)
            if frames is not None and trajectory_done:
                assert initial is not None
                final = self._mol.GetConformer(relaxing[0]).GetPositions().copy()
                path = [initial, *(frame[:n_atoms] for frame in recorded)]
                if not np.allclose(path[-1], final):
                    path.append(final)
                frames[:] = path
        return e

    def _retry_relaxation(self, seed_pos, stiffness, max_iters, operation, frames):
        """Retry each failed conformer separately; restore its seed if every stiffness fails.

        Returns the number tried. The pipeline separately rejects a donor hand that changes during retry.
        """

        def place(cid, pos):
            self._mol.GetConformer(cid).SetPositions(pos)

        failed = [c for c in self.ids if c in self.unrelaxed or not self._relax_ok(c)]
        rescued = 0
        for cid in failed:
            accepted = False
            for mult in FC_ESCALATION[1:]:  # rung 0 is the pass that already tore it
                place(cid, seed_pos[cid])
                snapshots = {} if frames is not None else None
                statuses = {}
                try:
                    restrained_uff(
                        self._mol,
                        self.cons,
                        stiffness=stiffness * mult,
                        max_iters=max_iters,
                        conf_ids=[int(cid)],
                        _snapshots=snapshots,
                        _statuses=statuses,
                    )
                except RuntimeError:  # UFF cannot build for this graph, so the seed is the best available
                    break
                if statuses.get(cid, 1) == 0 and self._relax_ok(cid):
                    accepted = True
                    if frames is not None:
                        assert snapshots is not None
                        final = self._mol.GetConformer(cid).GetPositions().copy()
                        path = [seed_pos[cid], *snapshots.get(cid, [])]
                        if not np.allclose(path[-1], final):
                            path.append(final)
                        frames[:] = path
                    break
            if accepted:
                rescued += 1
                self.unrelaxed = [i for i in self.unrelaxed if i != cid]
                continue
            place(cid, seed_pos[cid])
            if frames is not None:
                frames[:] = [seed_pos[cid]]
            if cid not in self.unrelaxed:
                self.unrelaxed.append(cid)  # every retry failed, so publish the seed only with an explicit flag
        if failed:
            logger.warning(  # a geometry that never completed a relax is not what `minimize` promises: say so
                "%s: %d/%d failed the relax contract; %d rescued, %d restored to their seeds",
                operation,
                len(failed),
                len(self.ids),
                rescued,
                len(failed) - rescued,
            )
        return len(failed)

    def _rescore_restrained(self, stiffness):
        """Record one comparable restrained-UFF single point for every tracked conformer."""
        try:
            e = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=0, conf_ids=self.ids)
        except RuntimeError as err:
            logger.warning("minimize: UFF could not score the relaxed conformers (%s)", _error_summary(err))
            self.energies = {}
            return
        self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=True)}

    def _metal_hands(self):
        """Return ``{conformer id: realised metal-centre hand}`` over the tracked conformers."""
        mol = self.mol  # bind once: the finalize copies the whole molecule
        return {c: _realised_hand(mol, self.iso, c) for c in self.ids}

    def _metal_states(self):
        """Return whether each conformer realises every stated metal stereo target."""
        mol = self.mol
        targets = _stereo_targets(self.iso)
        ranks, eta2 = _winding_ranks(mol, targets)
        return {c: _seed_stereo_matches(mol, c, targets, ranks, eta2, False) for c in self.ids}

    def _replace_failed(self, failed, stiffness, max_iters, *, template=None, validator=None, seed=None):
        """Replace failed conformers from one fresh-seed loop; return ids not replaced.

        Replacements keep the original ids. Each batch receives a constraint copy because embedding may add
        encounter bounds or move phantom indices. ``validator`` may add pipeline publication checks after the
        core contract; it never controls how a replacement is generated or relaxed.
        """
        seed = self.seed if seed is None else seed
        if seed is None:
            return list(failed)
        left = list(failed)
        source = self._mol if template is None else template
        for attempt in range(1, _MAX_HAND_ROUNDS + 1):
            if not left:
                break
            cons = self.cons.copy()
            mol, ids, _target = seed_conformers(
                Chem.Mol(source), cons, self.iso, len(left) + _HAND_BUFFER, seed=seed + attempt
            )
            if not ids:
                continue
            batch = Conformers(mol, ids, cons, self.iso, seed=seed + attempt)
            batch._relax_once(stiffness, max_iters)
            failures = batch._acceptance_failures(validator=validator)
            batch._remove({cid for rejected in failures.values() for cid in rejected})
            for cid, src in zip(list(left), batch.ids, strict=False):
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

    def _correct_metal_hand(self, operation="minimize"):
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
        # core's mirror is a different core -- re-seeding re-grafts it instead), and no haptic face, which can
        # be planar-chiral in a way this tier cannot perceive (`pipeline.select_stereo` owns that).
        if (
            iso.chirality
            and len(targets) == 1
            and _mirror_is_free(self._mol)
            and not self.cons.frozen
            and not self.cons.dihedrals
            and not iso.haptic
        ):
            reflectable = [cid for cid in wrong if hands[cid] and hands[cid] != iso.chirality]
            for cid in reflectable:
                _reflect(self._mol, cid)
            reflected = len(reflectable)
            states = self._metal_states()
            wrong = [cid for cid in wrong if not states[cid]]
        logger.info(
            "%s: metal-state correction: %d reflected, %d unresolved",
            operation,
            reflected,
            len(wrong),
        )
        return wrong

    def _acceptance_failures(self, operation="minimize", validator=None):
        """Return publication failures grouped by reason after free hand correction."""
        failures = {}
        wrong = self._correct_metal_hand(operation)
        for cid in self.ids:
            if reason := self._geometry_failure(cid):
                failures.setdefault(reason, []).append(cid)
            elif validator is not None and (reason := validator(self, cid)):
                failures.setdefault(reason, []).append(cid)
        if wrong:
            failures["the requested metal state"] = wrong
        return failures

    def _accept_relaxed(self, stiffness, max_iters, operation="minimize", *, validator=None, template=None, seed=None):
        """Replace core publication failures once, reject unresolved ids and return their reasons.

        Optimizer status is metadata: an unrelaxed conformer may survive when its restored seed still satisfies
        the structural contract. Structurally invalid seeds and wrong metal states share the same fresh-seed
        replacement loop. Internal replacement batches call `_acceptance_failures` directly and cannot recurse.
        """
        target = len(self.ids)
        failures = self._acceptance_failures(operation, validator)
        requirement = self._required_failure(failures)
        failed = {cid for rejected in failures.values() for cid in rejected}
        if failed:
            self._replace_failed(
                failed,
                stiffness,
                max_iters,
                template=template,
                validator=validator,
                seed=seed,
            )
            failures = self._acceptance_failures(operation, validator)
            failed = {cid for rejected in failures.values() for cid in rejected}
            self._report_missed_fixes(failures.get("missed numeric fix", ()), operation)
            self._remove(failed)
        if not failures:
            return {}

        reason = ", ".join(f"{len(ids)}x {name}" for name, ids in failures.items())
        message = (
            f"{operation}: could not produce {target} conformer(s); {reason} remained after re-seeding; "
            "the arrangement may be infeasible"
        )
        if len(self.ids) < target and requirement:
            raise ValueError(f"{message}; could not satisfy {requirement}")
        logger.warning(message)
        return failures

    def _relax_once(self, stiffness, max_iters, operation="minimize", frames=None):
        """Run one same-seed relaxation policy without generating replacement seeds."""
        if not self.ids:
            return None
        if not self.cons.is_constrained:
            statuses = {}
            e = restrained_uff(
                self._mol,
                self.cons,
                stiffness=stiffness,
                max_iters=max_iters,
                conf_ids=self.ids,
                _statuses=statuses,
            )
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

        `stiffness` scales flat-bottomed walls and strict scalar fixes; approximate target pulls stay fixed.
        A constrained run escalates restraints only when the relaxed geometries miss the accept gate, then retries
        each failed seed separately. A seed no rung can relax is
        restored and listed in `.unrelaxed`; an untypeable graph keeps its embedded geometry without an energy.
        A max-iteration seed remains in `.unrelaxed` only when it satisfies the structural contract. A wrong
        metal state is re-seeded, then rejected; a public call fails if that spends the requested count.

        `.energies` holds restrained-UFF values for ranking this result, not comparison across species.
        `pipeline.Ensemble.minimize()` adds acceptance gates and may drop failures.
        """
        self.trajectory = None
        if not self.ids:
            return self
        e = self._relax_once(stiffness, max_iters)
        self._accept_relaxed(stiffness, max_iters)
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
        if len(idx) not in (_BOND_ATOMS, _ANGLE_ATOMS, _DIHEDRAL_ATOMS):
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

    def __getitem__(self, key):
        """Pick conformer(s) by position as a new `Conformers`: ``confs[0]``, ``confs[:3]``."""
        sel = self.ids[key]
        sel = sel if isinstance(sel, list) else [sel]
        return Conformers(
            self._mol,
            sel,
            self.cons,
            self.iso,
            {i: self.energies[i] for i in sel if i in self.energies},
            unrelaxed=[i for i in self.unrelaxed if i in sel],  # a slice must not silently lose these flags
            seed=self.seed,
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


def embed(
    spec,
    *,
    fix=None,
    constrain=None,
    template=None,
    n=None,
    seed=DEFAULT_SEED,
    prune_rms=0.1,
    threads=0,
    knowledge=True,
):
    """Embed conformers of `spec` under ``fix``/``constrain``; return a `Conformers`.

    `fix`/`constrain` keys are 0-based atom indices in `spec`'s own order. The full vocabulary and how the two
    verbs compose is the `constraints` module docstring.

    Parameters
    ----------
    spec : Mol | Isomer
        An RDKit `Mol` with explicit Hs (parsing and Hs are the caller's job), or a metal `Isomer`, whose
        polyhedron is composed with the spec so a substrate binds a named coordination sphere.
    fix : list | dict, optional
        Rigid, and the kinds may mix: a list of indices grafts them at `spec`'s own coordinates (needs a
        conformer), ``{i: (x, y, z)}`` at coordinates you supply, or tuple keys of 2 (distance), 3 (angle) or
        4 (dihedral) atoms at scalar numbers reproduced within 0.001 A or 0.005 degrees. Explicit numeric
        windows stay inside their stated range. These are restraints, not coordinate grafts::

            embed(ts_mol, fix=[3, 7, 11, 12])   # graft a reacting core at its own coords, 0.000 Å
            embed(mol, fix={(3, 11): 2.05})     # state the forming bond; the rest is free
    constrain : dict, optional
        Soft: a window on the seed a real energy may overrule. ``{(i, j): (lo, hi)}`` distance,
        ``{(i, j, k): (lo, hi)}`` angle, ``{(i, j, k, l): (lo, hi)}`` dihedral, or
        ``{(ring_a, ring_b): separation}`` π-stack.
    template : tuple, optional
        ``(reference, SMARTS_or_map)``. A SMARTS must have one ordered match on both molecular graphs; use an
        explicit ``{target_index: reference_index}`` map for a symmetric core or coordinate array.
    n : int, optional
        Conformer count. ``None`` means a flexibility-scaled count (`bounds.seed_count`), not RDKit's 10.
    seed, prune_rms, threads, knowledge
        Passed to ETKDG. Two defaults are not RDKit's: `seed` is always set (RDKit's -1 draws from the global
        RNG, so every result depends on prior consumption) and `prune_rms` is 0.1, not off.
    """
    if not isinstance(spec, (Chem.Mol, Isomer)):
        raise TypeError(
            f"embed() takes an RDKit Mol or an Isomer, got {type(spec).__name__}; parse a SMILES / .xyz "
            f"yourself (perception is the caller's job) and add Hs"
        )
    if template is not None:  # sugar: a reference core is a coordinate fix. Dissolved before anything routes.
        src = spec.mol if isinstance(spec, Isomer) else spec
        own = src.GetConformer().GetPositions() if src.GetNumConformers() else None
        fix = template_to_fix(template, fix, own, src)
    if not isinstance(spec, Isomer):
        _check_bare_mol(spec)
    mol, cons, iso, graft_ref = prepare(spec, fix=fix, constrain=constrain)
    mol, ids, target = seed_conformers(
        mol, cons, iso, n, seed=seed, knowledge=knowledge, prune_rms=prune_rms, threads=threads, graft_ref=graft_ref
    )
    require_seed_count(ids, target)
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
    return Conformers(mol, ids, cons, iso, seed=int(seed))  # `minimize`'s handedness gate re-seeds off this


def prepare(spec, *, fix=None, constrain=None):
    """Prepare a copied Mol and its constraints for embedding or relaxation.

    Return ``(mol, constraints, isomer, graft_reference)``. A metal geometry becomes one retained `Isomer`;
    a coordinate-free metal must arrive as an explicitly selected `Isomer`.
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
    if iso is not None:
        graft_ref = {**iso._graft_ref, **ref}
        cons = fold_substrate(
            iso.cons,
            cons,
            graft_ref,
            protect_arrangement=bool(iso._constrained_metals),
        )
        ref = graft_ref
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
