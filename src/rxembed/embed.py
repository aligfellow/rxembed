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
from rdkit.Chem import rdMolTransforms

from . import metal_core as _metal
from . import metal_polyhedron as _poly
from .bounds import DEFAULT_SEED, probe_conformer, seed_coordinates, seed_count
from .constraints import (
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    Constraints,
    compose,
    resolve_atom,
    resolve_core,
    template_to_fix,
)
from .metal_isomers import Isomer, from_geometry
from .relax import MAX_ITERS, _error_summary, bonding_ok, restrained_uff

logger = logging.getLogger("rxembed")  # configured by rxembed.set_verbose


_EPS = 1e-6  # near-zero norm floor for the graft axis
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation
_ANGLE_ATOMS = 3
_DIHEDRAL_ATOMS = 4


def _periodic_near(value, lo, hi):
    """Return the periodic image of an angle nearest a stated interval's midpoint."""
    middle = 0.5 * (lo + hi)
    return middle + (value - middle + 180.0) % 360.0 - 180.0


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


def _determined_by_graft(base, graft_ref):
    """Return grafted sphere atoms when two or more would determine the arrangement."""
    if not graft_ref:
        return []
    sphere = {a for k in base.distances for a in k}
    pinned = sorted(sphere & set(graft_ref))
    return pinned if len(pinned) > 1 else []


def fold_substrate(base, sub, graft_ref):
    """Compose substrate constraints with a coordination sphere.

    Field-driven composition keeps every constraint kind; the previous hand-written merge dropped `sub.planes`.
    A substrate grip overlapping a sphere hold remains structural. A coordinate graft may pin at most one sphere
    atom; two would silently replace the selected arrangement with the reference arrangement.
    """
    pinned = _determined_by_graft(base, graft_ref)
    if pinned:
        raise ValueError(
            f"the coordinate graft (fix={{i: (x,y,z)}} / template=) pins coordination-sphere atoms {pinned}: "
            f"the graft restores them to their exact reference coordinates after the embed, so it, not the "
            f"coordination geometry, would decide how they sit, and you would get the reference's arrangement "
            f"under this isomer's label. Graft at most one sphere atom (a core over ligand backbone atoms "
            f"composes fine), or embed the reference geometry itself as the source."
        )
    sphere_d = set(base.distances)
    sphere_angular = set(base.angles) | set(base.dihedrals)
    soft_d, soft_angular = sub.contacts
    merged = compose(base, sub)
    return merged.copy(
        contacts=(
            frozenset(k for k in soft_d if k not in sphere_d),
            frozenset(k for k in soft_angular if k not in sphere_angular),
        ),
    )


def _seed_stereo_matches(mol, cid, iso, winding_ranks, reflectable):
    """Return whether one raw DG seed has the selected metal hand and haptic winding."""
    expected_hand = iso.chirality
    realised = (
        _metal.realised_chirality(mol, cid, iso.geometry, iso.vertices, iso.metal, expected_hand, iso.haptic)
        if expected_hand
        else ""
    )
    if realised and realised != expected_hand and reflectable:
        _reflect(mol, cid)  # an exact isometry: no distance, angle or conformer diversity changes
        realised = expected_hand
    pos = mol.GetConformer(int(cid)).GetPositions()
    winding_ok = all(
        _metal._face_winding(mol, pos, iso.metal, iso.haptic[dummy], winding_ranks) == winding
        for dummy, winding in iso.haptic_winding.items()
    )
    return (not expected_hand or realised == expected_hand) and winding_ok


def _stereo_batch_size(need, bits, reflectable, has_winding):
    """Return a DG batch large enough to retain ``need`` selected stereoisomers."""
    if reflectable:
        return need
    if has_winding:
        # A 2 sqrt(N) spare-success margin avoids a coin-flip second batch at the exact 2^k mean.
        return 2**bits * (need + 2 * (math.isqrt(need - 1) + 1))
    return 2 * need + _HAND_BUFFER


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
    expected_hand = iso.chirality if iso is not None else ""
    expected_winding = iso.haptic_winding if iso is not None else {}
    winding_ranks = _metal._donor_classes(mol, iso.donors) if expected_winding else {}
    reflectable = bool(
        expected_hand
        and not expected_winding
        and _mirror_is_free(mol)
        and not cons.frozen
        and not cons.dihedrals
        and not iso.haptic
    )
    held = []
    if iso is not None:  # hold a carbanion/amine donor's hand: a degree-3 centre with no M-C bond would
        mol, held = _metal._hold_donor_chirality(mol, iso.metal, iso.donors, cons)  # else invert freely
    if not expected_hand and not expected_winding:
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
            stereo_bits = int(bool(expected_hand)) + len(expected_winding)
            batch_n = _stereo_batch_size(need, stereo_bits, reflectable, bool(expected_winding))
            ids = list(
                seed_coordinates(
                    mol,
                    cons,
                    batch_n,
                    seed=seed + attempt,
                    prune_rms=prune_rms,
                    knowledge=knowledge,
                    threads=threads,
                )
            )
            if ref_core is not None:
                graft_frozen(mol, ids, frozen, ref_core)
            for cid in ids:
                if _seed_stereo_matches(mol, cid, iso, winding_ranks, reflectable):
                    kept.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
                    if kept.GetNumConformers() == target:
                        break
        mol = kept
        ids = [conf.GetId() for conf in mol.GetConformers()]
        if len(ids) < target:
            logger.warning(
                "embed: kept %d/%d seeds with the requested %s after %d DG batch(es)",
                len(ids),
                target,
                " and ".join(
                    part
                    for part in (
                        f"{expected_hand} metal hand" if expected_hand else "",
                        f"{len(expected_winding)} haptic winding(s)" if expected_winding else "",
                    )
                    if part
                ),
                _MAX_HAND_ROUNDS,
            )
    if iso is not None:
        mol = _metal._release_donor_chirality(mol, held, cons)  # drop the dummy D's + cons keys, restore charges
    return mol, ids


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
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, xyz.tolist())


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
    # ids restored to their embed seed because no stiffness relaxed them without tearing or missing a fix
    seed: int | None = field(default=None, kw_only=True)
    # ids whose metal centre does not realise the selected hand
    wrong_hand: list = field(default_factory=list, kw_only=True)
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

    @property
    def _bond_tol(self):
        """The break threshold the accept gate uses; looser for a metal (see `METAL_BOND_TOL`)."""
        return METAL_BOND_TOL if self.iso is not None else BOND_TOL

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
            for atom, xyz in enumerate(positions):
                conf.SetAtomPosition(atom, xyz.tolist())
            mol.AddConformer(conf, assignId=True)
        self.trajectory = mol

    def _intact(self, cid):
        """Return True if `cid` retains its graph and every numeric fix."""
        return self._fixed_geometry_ok(cid) and bonding_ok(
            self._mol, cid, bond_tol=self._bond_tol, exclude=self.cons.frozen, constrained=self.cons.distances
        )

    def _fixed_geometry_misses(self, cid):
        """Return numeric fixes missed by `cid`, including their excess over the public tolerance."""
        conf = self._mol.GetConformer(cid)
        pos = conf.GetPositions()
        misses = []
        for atoms, (lo, hi) in self.cons.fixed.items():
            if len(atoms) == _BOND_ATOMS:
                i, j = atoms
                actual, tol, unit = float(np.linalg.norm(pos[i] - pos[j])), FIX_DISTANCE_TOL, "A"
            elif len(atoms) == _ANGLE_ATOMS:
                actual, tol, unit = rdMolTransforms.GetAngleDeg(conf, *atoms), FIX_ANGLE_TOL, "deg"
            else:
                actual = _periodic_near(rdMolTransforms.GetDihedralDeg(conf, *atoms), lo, hi)
                tol, unit = FIX_ANGLE_TOL, "deg"
            excess = abs(actual - lo) - tol if lo == hi else max(lo - actual, actual - hi)
            if not np.isfinite(actual):
                excess = float("inf")
            if excess > 0.0:
                misses.append((excess, atoms, actual, lo, hi, tol, unit))
        return misses

    def _fixed_geometry_ok(self, cid):
        """Return whether every scalar target or explicit fixed window meets its contract."""
        return not self._fixed_geometry_misses(cid)

    def _reject_missed_fixes(self, operation):
        """Reject and report conformers that could not retain a numeric fix."""
        failures = {cid: miss for cid in self.ids if (miss := self._fixed_geometry_misses(cid))}
        bad = list(failures)
        if bad:
            rejected = set(bad)
            self.ids = [cid for cid in self.ids if cid not in rejected]
            self.unrelaxed = [cid for cid in self.unrelaxed if cid not in rejected]
            for cid in bad:
                self.energies.pop(cid, None)
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

    def _coordination_ok(self, cid, iso=None):
        """Reject a puckered result for a declared planar polyhedron."""
        iso = self.iso if iso is None else iso
        if iso is None or not _poly.is_planar(iso.geometry) or not iso.donors:
            return True
        pos = self._mol.GetConformer(cid).GetPositions()
        # a haptic face is one vertex
        return _metal.coplanar(pos, iso.metal, iso.donors, haptic=self.cons.haptic)

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, conf_ids=None, operation="minimize", _frames=None):
        """Relax with restrained UFF, escalating only when no conformer passes the accept gate.

        Returns the accepted energies, or ``None`` when UFF cannot type the graph. `conf_ids` limits the batch.
        """
        # Hold a labile (carbanion/amine) donor's hand through the relax: the surrogate's bare degree-3 centre
        # inverts under UFF. Cap it with a dummy D, release when done.
        n_atoms = self._mol.GetNumAtoms()
        all_ids = [c.GetId() for c in self._mol.GetConformers()]
        relaxing = all_ids if conf_ids is None else [int(c) for c in conf_ids]
        initial = self._mol.GetConformer(relaxing[0]).GetPositions().copy() if _frames is not None else None
        recorded, trajectory_done = [], False
        iso = self.iso
        held = _metal._hold_donor_chirality(self._mol, iso.metal, iso.donors, self.cons) if iso else (self._mol, [])
        self._mol, hold = held
        e, fc = None, stiffness
        try:
            retrying = set(relaxing)
            self.unrelaxed = [i for i in self.unrelaxed if i not in retrying]
            embed_pos = {
                c: [list(self._mol.GetConformer(c).GetAtomPosition(a)) for a in range(self._mol.GetNumAtoms())]
                for c in relaxing
            }
            donor_hands = {
                c: {donor: _metal.donor_chirality_sign(self._mol, c, donor) for _dummy, donor, _charge in hold}
                for c in relaxing
            }

            def restore(ids=relaxing):
                for cid in ids:
                    conf = self._mol.GetConformer(cid)
                    for a, xyz in enumerate(embed_pos[cid]):
                        conf.SetAtomPosition(a, xyz)

            for step, mult in enumerate(FC_ESCALATION):  # half-order steps: find the minimum sufficient stiffness
                if step:  # restore the embed geometry before a stiffer retry (a big jump over-stiffens it)
                    restore()
                fc = stiffness * mult
                snapshots = {} if _frames is not None else None
                try:
                    e = restrained_uff(
                        self._mol,
                        self.cons,
                        stiffness=fc,
                        max_iters=max_iters,
                        conf_ids=conf_ids,
                        _snapshots=snapshots,
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
                # accept a stiffness only if a conformer is both bonded and, for a planar polyhedron, coplanar,
                # so escalation never forces a phantom through as an out-of-plane pucker.
                kept = [i for i in self.ids if self._intact(i) and self._coordination_ok(i)]
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
                e = restrained_uff(self._mol, self.cons, stiffness=fc, max_iters=0, conf_ids=conf_ids)
                logger.warning(
                    "%s: UFF inverted a coordinated ligand stereocentre in %d conformer(s); kept the DG seed",
                    operation,
                    len(inverted),
                )
            trajectory_done = True
        finally:  # release the hold on every exit path, including the early UFF-failure return; the dummy D
            if hold:  # is scaffolding for this relax alone
                self._mol = _metal._release_donor_chirality(self._mol, hold, self.cons)
            if _frames is not None and trajectory_done:
                assert initial is not None
                final = self._mol.GetConformer(relaxing[0]).GetPositions().copy()
                path = [initial, *(frame[:n_atoms] for frame in recorded)]
                if not np.allclose(path[-1], final):
                    path.append(final)
                _frames[:] = path
        return e

    def _rescue_torn(self, seed_pos, stiffness, operation="minimize", _frames=None):
        """Retry each torn or off-fix conformer separately; restore its seed if every stiffness fails.

        Returns the number tried. The pipeline separately rejects a donor hand that changes during rescue.
        """

        def place(cid, pos):
            conf = self._mol.GetConformer(cid)
            for a, xyz in enumerate(pos):
                conf.SetAtomPosition(a, xyz.tolist())

        torn = [c for c in self.ids if not self._intact(c)]
        rescued = 0
        for cid in torn:
            for mult in FC_ESCALATION[1:]:  # rung 0 is the pass that already tore it
                place(cid, seed_pos[cid])
                snapshots = {} if _frames is not None else None
                try:
                    restrained_uff(
                        self._mol,
                        self.cons,
                        stiffness=stiffness * mult,
                        conf_ids=[int(cid)],
                        _snapshots=snapshots,
                    )
                except RuntimeError:  # UFF cannot build for this graph, so the seed is the best available
                    break
                if self._intact(cid):
                    if _frames is not None:
                        assert snapshots is not None
                        final = self._mol.GetConformer(cid).GetPositions().copy()
                        path = [seed_pos[cid], *snapshots.get(cid, [])]
                        if not np.allclose(path[-1], final):
                            path.append(final)
                        _frames[:] = path
                    rescued += 1
                    self.unrelaxed = [i for i in self.unrelaxed if i != cid]
                    break
            else:
                place(cid, seed_pos[cid])
                if _frames is not None:
                    _frames[:] = [seed_pos[cid]]
                if cid not in self.unrelaxed:
                    self.unrelaxed.append(cid)  # never relaxed: say so, or nothing downstream can tell
                continue
            if not self._intact(cid):
                place(cid, seed_pos[cid])
                if _frames is not None:
                    _frames[:] = [seed_pos[cid]]
                if cid not in self.unrelaxed:
                    self.unrelaxed.append(cid)
        if torn:
            logger.warning(  # a geometry that never completed a relax is not what `minimize` promises: say so
                "%s: %d/%d torn or off-fix; %d rescued, %d restored to their seeds",
                operation,
                len(torn),
                len(self.ids),
                rescued,
                len(torn) - rescued,
            )
        return len(torn)

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

    def _reseed_hand(self, wrong, stiffness, max_iters):
        """Replace wrong-hand conformers with fresh right-hand seeds; return ids not replaced.

        Replacements keep the original ids. Each batch receives a constraint copy because embedding may add
        encounter bounds or move phantom indices.
        """
        seed, iso = self.seed, self.iso
        if seed is None or iso is None:  # `_hold_metal_hand` gates both; a direct caller gets a no-op, not a
            return list(wrong)  # crash, and "fixed none of them" is the honest answer without a seed to re-roll
        left = list(wrong)
        for attempt in range(1, _MAX_HAND_ROUNDS + 1):
            if not left:
                break
            cons = self.cons.copy()
            mol, ids = seed_conformers(Chem.Mol(self._mol), cons, iso, len(left) + _HAND_BUFFER, seed=seed + attempt)
            if not ids:
                continue
            batch = Conformers(mol, ids, cons, iso).minimize(stiffness, max_iters, _retry=False)
            hands = batch._metal_hands()  # `_retry=False` above: this batch is read here, never re-seeded again
            spare = [c for c in batch.ids if hands[c] == iso.chirality and c not in batch.unrelaxed]
            for cid, src in zip(list(left), spare, strict=False):
                pos = batch._mol.GetConformer(int(src)).GetPositions()
                conf = self._mol.GetConformer(int(cid))
                for a, xyz in enumerate(pos):
                    conf.SetAtomPosition(a, xyz.tolist())
                self.energies.pop(cid, None)
                if src in batch.energies:
                    self.energies[cid] = batch.energies[src]
                if cid in self.unrelaxed:  # the geometry that flag described is gone: the replacement relaxed
                    self.unrelaxed.remove(cid)
                left.remove(cid)
        return left

    def _hold_metal_hand(self, stiffness, max_iters, operation="minimize"):
        """Correct the selected metal hand where possible and record any failures.

        Distance and angle constraints cannot choose a mirror; a signed dihedral can. The hand is read after
        relaxation, then corrected by reflection only when no stated geometry distinguishes the mirror.
        """
        iso = self.iso
        self.wrong_hand = []
        if iso is None or not iso.chirality or not self.ids:
            return
        hands = self._metal_hands()
        wrong = [c for c, hand in hands.items() if hand != iso.chirality]
        if not wrong:
            return
        reflected, reseeded = 0, 0
        # The mirror is free only where the metal centre is the one thing it inverts: no other stereocentre
        # (`_mirror_is_free`), no grafted core (its contract is the caller's exact geometry, and a chiral
        # core's mirror is a different core -- re-seeding re-grafts it instead), and no haptic face, which can
        # be planar-chiral in a way this tier cannot perceive (`pipeline.select_stereo` owns that).
        if _mirror_is_free(self._mol) and not self.cons.frozen and not self.cons.dihedrals and not iso.haptic:
            reflectable = [cid for cid in wrong if hands[cid]]
            for cid in reflectable:
                _reflect(self._mol, cid)
            reflected = len(reflectable)
            wrong = [cid for cid in wrong if not hands[cid]]
        if wrong and self.seed is not None:
            reseeded = len(wrong)
            wrong = self._reseed_hand(wrong, stiffness, max_iters)
            reseeded -= len(wrong)
        self.wrong_hand = wrong
        logger.info(
            "%s: corrected metal hand in %d conformer(s): %d reflected, %d re-seeded",
            operation,
            reflected + reseeded + len(wrong),
            reflected,
            reseeded,
        )
        if wrong:  # the caller asked for one hand and is getting the other: only this list says so
            logger.warning(
                "%s: %d of %d conformer(s) are not the %s centre that was asked for (see .wrong_hand)",
                operation,
                len(wrong),
                len(self.ids),
                iso.chirality,
            )

    def minimize(self, stiffness=BASE_STIFFNESS, max_iters=MAX_ITERS, _retry=True):
        """Relax every conformer in place with restrained UFF; return ``self``.

        `stiffness` scales flat-bottomed walls and strict scalar fixes; approximate target pulls stay fixed.
        A constrained run escalates restraints only when the relaxed geometries miss the accept gate, then retries
        each failed seed separately. A seed no rung can relax is
        restored and listed in `.unrelaxed`; an untypeable graph keeps its embedded geometry without an energy.
        A conformer that cannot retain a numeric fix is rejected; other failures remain flagged in
        `.unrelaxed` or `.wrong_hand`.

        `.energies` holds restrained-UFF values for ranking this result, not comparison across species.
        `pipeline.Ensemble.minimize()` adds acceptance gates and may drop failures.
        """
        self.trajectory = None
        if not self.ids:
            return self
        if not self.cons.is_constrained:  # no window to tear against, so no ladder to climb
            e = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=max_iters, conf_ids=self.ids)
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
            return self
        self.unrelaxed = []  # a re-minimize re-decides it; a stale list would outlive the geometry it described
        self.wrong_hand = []
        seed_pos = {c: self._mol.GetConformer(c).GetPositions() for c in self.ids}
        e = self._relax_constrained(stiffness, max_iters, conf_ids=self.ids)
        self._rescue_torn(seed_pos, stiffness)
        if e is not None:
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
        self._reject_missed_fixes("minimize")
        if _retry:
            self._hold_metal_hand(stiffness, max_iters)
        if e is not None or self.energies:
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
        idx = [resolve_atom(self._mol, a) for a in atoms]
        fns = {2: rdMolTransforms.GetBondLength, 3: rdMolTransforms.GetAngleDeg, 4: rdMolTransforms.GetDihedralDeg}
        if len(idx) not in fns:
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        f = fns[len(idx)]
        v = [f(self._mol.GetConformer(c), *idx) for c in self.ids]
        if len(idx) == _DIHEDRAL_ATOMS:
            key = min(tuple(idx), tuple(reversed(idx)))
            if window := self.cons.dihedrals.get(key):
                v = [_periodic_near(value, *window) for value in v]
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
            wrong_hand=[i for i in self.wrong_hand if i in sel],
            trajectory=Chem.Mol(self.trajectory) if self.trajectory is not None and sel == self.ids else None,
        )

    def __len__(self):
        """Return the number of conformers in play."""
        return len(self.ids)

    def __repr__(self):
        """Summarise the result: conformer count and the isomer identity when there is one."""
        who = f", {self.iso.summary()}" if self.iso is not None else ""
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
    iso = spec if isinstance(spec, Isomer) else None  # `seed`/`n` are validated at the seam (`seed_conformers`)
    if iso is None:
        _check_bare_mol(spec)
    mol = Chem.Mol(iso.mol if iso is not None else spec)  # our own copy: conformers are never shared with the caller
    cons, graft_ref = Constraints(), {}
    if iso is None or fix or constrain:
        cons, graft_ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=mol.GetNumConformers() > 0)
    if iso is not None:
        cons = (
            fold_substrate(iso.coordination().copy(), cons, graft_ref)
            if (fix or constrain)
            else iso.coordination().copy()
        )
    mol, ids = seed_conformers(
        mol, cons, iso, n, seed=seed, knowledge=knowledge, prune_rms=prune_rms, threads=threads, graft_ref=graft_ref
    )
    n_frag = len(Chem.GetMolFrags(mol))
    if not ids:  # ETKDG met no constraint set it could realise; silence reads as "embedded, then pruned"
        logger.warning(
            "embed: no conformer under %d constraint(s); the spec may not be realisable",
            len(cons.distances) + len(cons.angles) + len(cons.dihedrals),
        )
    logger.info(
        "embed[%s]: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        iso.summary() if iso is not None else "molecule",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles) + len(cons.dihedrals),
    )
    return Conformers(mol, ids, cons, iso, seed=int(seed))  # `minimize`'s handedness gate re-seeds off this


def prepare_relax(spec, *, fix=None, constrain=None):
    """Prepare a copied geometry for search-free relaxation.

    Returns ``(mol, ids, constraints, isomer)``. A plain metal Mol keeps its input sphere; an Isomer keeps its
    declared coordination constraints. Coordinate-fixed atoms are grafted exactly.
    """
    from .metal_distance import ff_terms
    from .metal_isomers import from_surrogate

    iso = spec if isinstance(spec, Isomer) else None
    mol = Chem.Mol(iso.mol if iso is not None else spec)  # our own copy: never the caller's conformers
    spheres, metals = {}, []
    if iso is None and _metal.metal_index(mol) is not None:  # as `embed`: a bond-less surrogate on a zero-vdW FF,
        spheres = {  # because handing UFF a real metal makes the force field depend on which metal you have
            mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()] for mi in _metal.metal_indices(mol)
        }
        donor_bonds = [(d, mi) for mi, dons in spheres.items() for d in dons]  # re-added DATIVE on the output
        mol, metals = _metal.surrogate_all_metals(mol)
        iso = from_surrogate(mol, metals, donor_bonds, donors=spheres.get(metals[0][0], ()))
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=True)
    if spheres:  # perceived from a plain Mol: hold what the input realises
        rz = {mi: z for mi, z, _q in metals}
        sphere_cons = Constraints()
        for mi, dons in spheres.items():
            _metal.hold_shape(mol, [mi, *dons], sphere_cons)
        ff_terms(mol, sphere_cons, {mi: (rz[mi], dons) for mi, dons in spheres.items()})
        cons = compose(sphere_cons, cons)
    elif iso is not None:  # an Isomer arrives with its polyhedron built: the same fold `embed` does
        base = iso.coordination().copy()  # copy: the relax edits cons in place, the Isomer keeps its record
        cons = fold_substrate(base, cons, ref) if (fix or constrain) else base
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
