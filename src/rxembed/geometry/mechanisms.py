"""Write constraints into edited RDKit distance-geometry bounds and restrained UFF.

Each mechanism keeps one kind's DG and FF terms together. `bounds._write` runs them in registry order, and restrained
UFF runs their FF hooks in the same order. Frozen, Pull, Dihedral, Umbrella and the organic UFF repairs are FF-only.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import override

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

from rxembed.geometry.constraints import (
    DIST_ATOMS,
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    canonical_key,
    donor_directions,
    graft_owns,
    periodic_window,
    plane_torsion_cap,
    requested_dihedral,
)
from rxembed.graph import disconnect_metal, no_aromatic_axis
from rxembed.utils import CARBON_Z, DISCONNECTED, SP2_DEGREE, STRAIGHT, conjugated_quartets

_PHANTOM_FLOOR = 0.30  # Å: a haptic centroid dummy may sit this close to any atom, living inside its own ring
_RIGHT_ANGLE = 90.0  # deg: syn/anti split and the open end of a trigonal angle
_PLANE_PAD = 0.3  # Å half-width on a pi-stack cross-ring distance
_MIN_HALF_WIDTH = 0.01  # Å: avoids division by zero for a near-zero contact window
_FLOOR_INF = 1e3  # Å: practical infinity for a one-sided floor

# RDKit distance terms use 1/2 k dev², angle terms k dev² in degrees; angle walls span legs of any length
PIN_FC = 1e4  # kcal/mol/Å²: requested distances sit on the metal-fidelity plateau
RELEASABLE_FC_SCALE = 0.3  # NCI wall scale: soft, so a releasable contact yields to the ligand instead of straining it
CONTACT_PUSH = 320.0  # kcal/mol/Å at either wall; the knee of the contact-centring response
ANGLE_FC = 30.0  # kcal/mol/deg²; fidelity is flat above this, and a higher constant breaks some windows
PI_STACK_FC = 2e3  # kcal/mol/Å²: softer than a requested distance because the seed already formed the stack
TARGET_FC = 1e4  # kcal/mol/Å²: holds an M-L target against the competing UFF pull
# kcal/mol/deg²: much softer than the angle wall, so it nudges without overriding it
_ANGLE_TARGET_FC = 0.001 * ANGLE_FC
FIX_DISTANCE_FC = 3e7  # kcal/mol/Å²: stiff enough to hold a fixed pair against the reactive-pair LJ repulsion
FIX_ANGLE_FC = 1e3  # kcal/mol/deg²: holds a fixed angle to within the acceptance gate
_COPLANAR_FC = 10.0  # kcal/mol/deg²: soft, but stiff enough to hold the plane against a rigid conjugated backbone
_UMBRELLA_FC = 3.0  # kcal/mol/deg²: stiff enough to keep the declared side of the plane at short M-L bonds
_SP2_HOLD_FC = 10.0 / 3.0  # kcal/mol/deg²: three ordered terms share the total restraint strength
_SP2_HOLD_WIN = 5.0  # deg around the seed improper: preserve existing curvature, never create it
_CONJ_CAP = 20.0  # deg: inside the geometry check's 30° conjugation window
_UFF_SMALL_SP2_RINGS = (3, 4)  # RDKit UFF uses nonperiodic angle potentials for these ring sizes


def _inside(window, margin):
    """Inset a flat wall so its finite-force equilibrium remains inside the user's fixed range."""
    lo, hi = window
    mid = 0.5 * (lo + hi)
    return min(lo + margin, mid), max(hi - margin, mid)


def _angular_wall(atoms, window, cons, stiffness, releasable):
    """Select the shared soft or strict angular wall and force constant."""
    fixed = cons.fixed.get(atoms)
    if fixed is not None:
        bounds = fixed if fixed[0] == fixed[1] else _inside(fixed, FIX_ANGLE_TOL)
        return *bounds, stiffness * FIX_ANGLE_FC
    # Never escalate a soft angular wall: a stiffer one displaces atoms the window should leave alone.
    scale = RELEASABLE_FC_SCALE if atoms in releasable else 1.0
    return *window, min(stiffness, 1.0) * ANGLE_FC * scale


@dataclass
class DGContext:
    """Hold what a DG writer needs beyond the Constraints: the molecule, the matrix and the pending windows."""

    mol: Chem.Mol
    bm: np.ndarray
    pairs: dict = field(default_factory=dict)  # (i, j) -> (lo, hi), committed into `bm` after RELIEVE
    _topo: np.ndarray | None = None
    basis: Chem.Mol | None = None

    @property
    def topo(self):
        """Return bond-path lengths on the metal-stripped graph; a real backbone gives a finite distance."""
        if self._topo is None:
            # a donor's temporary M-L carrier bond and any stale distance cache give no angle-prior ownership
            self._topo = Chem.GetDistanceMatrix(disconnect_metal(self.mol), force=True)
        return self._topo

    def mid(self, i, j):
        """Return the midpoint of the matrix's current window for a pair."""
        a, b = (i, j) if i < j else (j, i)
        return 0.5 * (self.bm[a][b] + self.bm[b][a])

    def leg(self, i, j):
        """Return an angle leg's length: an explicit window's midpoint, else the matrix default.

        It reads `pairs` while angles are still being written, so an earlier angle supplies a later one's leg: insertion
        order matters.
        """
        v = self.pairs.get(canonical_key((i, j)))
        return 0.5 * (v[0] + v[1]) if v else self.mid(i, j)


class Mechanism:
    """Write one constraint kind's DG and FF terms; a subclass implements only the hooks its kind needs."""

    def dg_windows(self, cons, ctx):
        """WINDOW: contribute candidate ``(lo, hi)`` windows to ``ctx.pairs``."""

    def dg_relief(self, cons, ctx):
        """RELIEVE: lower a matrix bound. Runs before COMMIT, so an explicit window always wins."""

    def dg_post(self, cons, ctx):
        """POST: read the committed matrix and tighten it."""

    def uff_terms(self, ff, cons, conf, stiffness):
        """FF: add terms for this field at the requested stiffness rung."""


class Frozen(Mechanism):
    """Pin atoms at their embedded coordinates with zero degrees of freedom; FF-only."""

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for idx in cons.frozen:
            ff.AddFixedPoint(idx)


class Distance(Mechanism):
    """Write distance windows to DG and FF, and hold every distance fix.

    A ranged fix is one strict inset wall, a scalar fix its padded window's wall plus a point hold.
    """

    @override
    def dg_windows(self, cons, ctx):
        ctx.pairs.update(cons.distances)  # override only the constrained pairs; RDKit keeps the rest

    @override
    def dg_relief(self, cons, ctx):
        """Relieve adjacent native floors for short held contacts."""
        # a contact deficit is shared only with direct neighbours (triangle inequality); it proves nothing feasible
        native = ctx.bm.copy()
        basis = ctx.basis if ctx.basis is not None else ctx.mol
        fragments = Chem.GetMolFrags(basis)
        component = {atom: frag for frag, atoms in enumerate(fragments) for atom in atoms}
        for atoms, window in cons.distances.items():
            if len(atoms) != DIST_ATOMS or not (atoms in cons.fixed or cons.frozen.issuperset(atoms)):
                continue
            i, j = atoms
            if component[i] == component[j]:
                continue
            hi = cons.fixed.get(atoms, window)[1]
            a, b = sorted((i, j))
            overlap = native[b, a] - hi
            if overlap <= 0.0:
                continue
            for origin, opposite in ((i, j), (j, i)):
                for neighbor in basis.GetAtomWithIdx(origin).GetNeighbors():
                    k = neighbor.GetIdx()
                    c, d = sorted((k, opposite))
                    pair = (c, d)
                    if (
                        pair in ctx.pairs
                        or pair in cons.floors
                        or pair in cons.dg_floors
                        or k in cons.phantoms
                        or opposite in cons.phantoms
                    ):
                        continue
                    ctx.bm[d, c] = min(ctx.bm[d, c], max(0.0, native[d, c] - overlap))

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        releasable = cons.contacts.distances
        seed_radii = {tuple(sorted((dummy, atom))) for dummy, face in cons.haptic.items() for atom in face}
        for (i, j), (lo, hi) in cons.distances.items():
            fixed = cons.fixed.get((i, j))
            if (i, j) in seed_radii and fixed is None:
                continue
            if fixed is not None and fixed[0] != fixed[1]:
                used_lo, used_hi = _inside(fixed, FIX_DISTANCE_TOL)
                fc = FIX_DISTANCE_FC
            else:
                used_lo, used_hi = lo, hi
                fc = PIN_FC * (RELEASABLE_FC_SCALE if (i, j) in releasable else 1.0)
            ff.AddDistanceConstraint(i, j, used_lo, used_hi, stiffness * fc)
        for atoms, (lo, hi) in cons.fixed.items():
            if len(atoms) == DIST_ATOMS and lo == hi:
                ff.AddDistanceConstraint(*atoms, lo, hi, stiffness * FIX_DISTANCE_FC)


class Pull(Mechanism):
    """Bias an unfixed distance or angle toward its preferred window, and a soft contact to its midpoint; FF-only."""

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for atoms, (lo, hi) in cons.pulls.items():
            if len(atoms) == DIST_ATOMS:
                ff.AddDistanceConstraint(*atoms, lo, hi, TARGET_FC)
            else:  # an angle window pushes only from outside it, at the same soft constant
                ff.UFFAddAngleConstraint(*atoms, False, lo, hi, _ANGLE_TARGET_FC)
        # a contact has no UFF bond in its window, so bias it to the midpoint without storing that target twice
        for key in cons.contacts.distances:
            window = cons.distances.get(key)
            if window is None or key in cons.pulls or key[::-1] in cons.pulls:
                continue
            half = max(0.5 * (window[1] - window[0]), _MIN_HALF_WIDTH)
            mid = 0.5 * (window[0] + window[1])
            ff.AddDistanceConstraint(key[0], key[1], mid, mid, CONTACT_PUSH / half)


class Floor(Mechanism):
    """Apply one-sided FF distance walls and relieve false metal-surrogate DG floors."""

    @override
    def dg_relief(self, cons, ctx):
        for (a, b), floor in cons.dg_floors.items():  # keys are (metal, X), sorted; bm[b][a] is the lower bound
            if (a, b) in ctx.pairs:  # an explicit window is the truth; never override it
                continue
            if floor < ctx.bm[b][a] <= ctx.bm[a][b]:  # only ever relax, and only if genuinely too high
                ctx.bm[b][a] = floor

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for (i, j), floor in cons.floors.items():
            # Explicit holds override the automatic wall; seed-only encounter windows do not.
            if (i, j) in cons.fixed or (i, j) in cons.contacts.distances:
                continue
            ff.AddDistanceConstraint(i, j, floor, _FLOOR_INF, stiffness * PIN_FC)


class Angle(Mechanism):
    """Write an FF angle wall and its law-of-cosines 1-3 DG prior, and hold every angle fix as its wall.

    The prior is intersected with a connected backbone and written outright only across the stripped metal; a disjoint
    intersection keeps RDKit's backbone bounds.
    """

    @override
    def dg_windows(self, cons, ctx):
        for (i, j, k), (lo, hi) in cons.angles.items():
            a, b = min(i, k), max(i, k)  # sorted, as `add_distance` stores them
            if (a, b) in ctx.pairs:  # an explicit distance window already owns this pair
                continue
            dij, djk = ctx.leg(i, j), ctx.leg(j, k)
            # Scalar fixed-angle seed padding can extend beyond the physical domain used by UFF.
            ang_lo = law_of_cosines(dij, djk, max(0.0, lo))
            ang_hi = law_of_cosines(dij, djk, min(STRAIGHT, hi))
            if ctx.topo[a][b] < DISCONNECTED:
                lo_hi = (max(ang_lo, ctx.bm[b][a]), min(ang_hi, ctx.bm[a][b]))
                ctx.pairs[(a, b)] = lo_hi if lo_hi[0] <= lo_hi[1] else (ctx.bm[b][a], ctx.bm[a][b])
            else:
                ctx.pairs[(a, b)] = (ang_lo, ang_hi)

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        releasable = cons.contacts.angular
        for atoms, window in cons.angles.items():
            i, j, k = atoms
            used_lo, used_hi, fc = _angular_wall(atoms, window, cons, stiffness, releasable)
            ff.UFFAddAngleConstraint(
                i,
                j,
                k,
                False,
                max(0.0, used_lo),
                min(180.0, used_hi),
                fc,
            )


class Dihedral(Mechanism):
    """Write periodic dihedral windows to UFF, every dihedral fix as its wall; DG cannot encode their signed hand."""

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        releasable = cons.contacts.angular
        for atoms, window in cons.dihedrals.items():
            used_lo, used_hi, fc = _angular_wall(atoms, window, cons, stiffness, releasable)
            ff.UFFAddTorsionConstraint(*atoms, False, used_lo, used_hi, fc)


class Coplanar(Mechanism):
    """Restore the donor-plane term the stripped M-donor bond took: a POST 1,4 bound and a soft FF torsion.

    The bound projects the full distance and M-D-X windows, since midpoint legs can exclude a valid chelate. Without a
    requested M-D-X angle only the FF torsion applies.
    """

    @override
    def dg_post(self, cons, ctx):
        for row in cons.coplanar:
            # a requested dihedral owns the seed as it owns the FF; a seeded row's graph gives a plane but not its well
            if requested_dihedral(cons, *row.atoms) or row.seeded:
                continue
            i, j, _k, _w = row.atoms
            for k, w in row.axes:
                key = canonical_key((i, w))
                if (
                    key in cons.distances
                ):  # a requested distance window is truth; an angle-derived prior only intersects
                    continue
                window = cons.angles.get((i, j, k)) or cons.angles.get((k, j, i))
                if window is None:  # nothing pins M-D-X, so say nothing
                    continue
                torsion_cap = row.cap
                if row.out_of_plane:  # the loosest dihedral the M-D-X window allows, so still a bound
                    torsion_cap = plane_torsion_cap(
                        row.cap, min(window, key=lambda angle: math.sin(math.radians(angle)))
                    )
                edge = _coplanar_bound(ctx.bm, (i, j, k, w), window, row.anchor, torsion_cap)
                if edge is None:
                    continue
                anti = row.anchor >= _RIGHT_ANGLE  # the metal is anti (far) to w in-plane vs syn (near)
                a, b = key  # a < b; bm[b][a] is the lower bound, bm[a][b] the upper
                if anti and ctx.bm[b][a] < edge <= ctx.bm[a][b]:  # anti: floor at the cap edge
                    ctx.bm[b][a] = edge
                elif not anti and ctx.bm[b][a] <= edge < ctx.bm[a][b]:  # syn: ceiling at the cap edge
                    ctx.bm[a][b] = edge

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for row in cons.coplanar:
            if graft_owns(row.atoms, cons.frozen, cons.haptic) or requested_dihedral(cons, *row.atoms):
                continue
            i, j, _k, _w = row.atoms
            for k, w in row.axes:
                torsion_cap = row.cap
                if row.out_of_plane:
                    torsion_cap = plane_torsion_cap(row.cap, rdMolTransforms.GetAngleDeg(conf, i, j, k))
                phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
                lo, hi = _coplanar_window(phi, torsion_cap, row.anchor)
                ff.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC / len(row.axes))


class Plane(Mechanism):
    """Hold a parallel pi-stack by cross-ring distances; explicit and angle-derived windows win."""

    @override
    def dg_windows(self, cons, ctx):
        for ring_a, ring_b, sep in cons.planes:  # cross-ring d = sqrt(sep^2 + in-plane^2)
            for u, au in enumerate(ring_a):
                for v, bv in enumerate(ring_b):
                    offset = 0.0 if u == v else ctx.mid(au, ring_a[v])
                    d = math.hypot(sep, offset)
                    ctx.pairs.setdefault(canonical_key((au, bv)), (d - _PLANE_PAD, d + _PLANE_PAD))

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        pos = conf.GetPositions()  # hold the stack as embedded; the seed already reached the separation
        for ring_a, ring_b, _sep in cons.planes:
            for a, b in itertools.product(ring_a, ring_b):
                d = float(np.linalg.norm(pos[a] - pos[b]))
                ff.AddDistanceConstraint(a, b, d - _PLANE_PAD, d + _PLANE_PAD, stiffness * PI_STACK_FC)


class Haptic(Mechanism):
    """Seat the DG helper inside its haptic face, then hold it at the moving centroid during UFF.

    A zero-length spring on every dummy-to-face-atom pair plus a negative spring on every face-atom pair gives E = K/2
    |p - mean(r)|², which leaves the face's own shape free. It is a finite penalty, not an exact dependent coordinate,
    so acceptance measures the real centroid separately.
    """

    @override
    def dg_relief(self, cons, ctx):
        n = len(ctx.bm)
        for p in cons.phantoms:
            for x in range(n):
                a, b = (p, x) if p < x else (x, p)
                if x == p or (a, b) in ctx.pairs:  # itself, or an explicit centroid window -> leave it
                    continue
                ctx.bm[b][a] = min(ctx.bm[b][a], _PHANTOM_FLOOR)  # bm[b][a] is the lower bound; only ever lower
        # A face atom sits near its site's angle to a donor; cap its cross-fragment floor at that angle's distance.
        for (i, metal, k), (lo, _hi) in cons.angles.items():
            for dummy, donor in ((i, k), (k, i)):
                if dummy not in cons.haptic or donor in cons.phantoms:
                    continue
                for atom in cons.haptic[dummy]:
                    a, b = sorted((atom, donor))
                    if ctx.topo[a][b] >= DISCONNECTED:  # a backbone floor is real
                        floor = law_of_cosines(ctx.leg(metal, atom), ctx.leg(metal, donor), lo)
                        ctx.bm[b][a] = min(ctx.bm[b][a], floor)

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for dummy, face in cons.haptic.items():
            n = len(face)
            force = stiffness * PIN_FC
            for atom in face:
                ff.UFFAddDistanceConstraint(dummy, atom, False, 0.0, 0.0, force / n)
            for left, right in itertools.combinations(face, 2):
                ff.UFFAddDistanceConstraint(left, right, False, 0.0, 0.0, -force / n**2)


class TrigonalAngle(Mechanism):
    """Remove UFF's collapsed trigonal angle basin at its native stiffness.

    RDKit uses k(1-cos(3 theta))/9 for every SP2 atom, and its correction (RDKit issue 7901) starts at 30 degrees, where
    a constrained optimization can stall. A native-curvature half-harmonic below 90 degrees removes that basin without
    changing ordinary trigonal angles, and reaches past the 60-degree saddle. Small-ring angles, requested geometry and
    pair floors are untouched.
    """

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        mol = conf.GetOwningMol()
        for atom in mol.GetAtoms():
            if atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            if any(atom.IsInRingSize(size) for size in _UFF_SMALL_SP2_RINGS):
                continue
            centre = atom.GetIdx()
            for left, right in itertools.combinations(atom.GetNeighbors(), 2):
                a, b = sorted((left.GetIdx(), right.GetIdx()))
                if (
                    (a, centre, b) in cons.angles
                    or (b, centre, a) in cons.angles
                    or (a, b) in cons.distances
                    or (a, b) in cons.floors
                    or graft_owns((a, centre, b), cons.frozen)
                ):
                    continue
                params = rdForceFieldHelpers.GetUFFAngleBendParams(mol, a, centre, b)
                if params is None:
                    raise RuntimeError(f"UFF lacks angle parameters for {a}-{centre}-{b}")
                force, _ = params
                # RDKit's constraint energy is K*delta_degrees**2, without the harmonic one-half factor.
                ff.UFFAddAngleConstraint(
                    a, centre, b, False, _RIGHT_ANGLE, STRAIGHT, 0.5 * force * math.radians(1) ** 2
                )


class Sp2Planar(Mechanism):
    """Hold each organic sp2 carbon at its seed improper; never pull it flat.

    UFF puckers a planar conjugated carbon; a window centred on the seed stops that without inventing curvature. A
    carbon a coordination sphere owns is left out, and so is one the graft touches: grafting places part of it after
    seeding, so its seed improper is no reference and UFF's own inversion term holds it. A TS carbon that must stay
    pyramidal needs its substituents in the core. Three torsions are needed, since RDKit has no permutation-invariant
    improper.
    """

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        if not cons.conjugation:
            return
        mol = conf.GetOwningMol()
        coordinated = _coordination_owned_carbons(mol, cons)
        requested = {frozenset(atoms) for atoms in cons.dihedrals}
        for atom in mol.GetAtoms():
            centre = atom.GetIdx()
            if atom.GetAtomicNum() != CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) != SP2_DEGREE:
                continue
            if not cons.frozen.isdisjoint((centre, *nbrs)):
                continue
            if centre in coordinated or frozenset((centre, *nbrs)) in requested:
                continue
            a, b, c = nbrs
            for key in ((a, b, c, centre), (b, a, c, centre), (c, a, b, centre)):
                phi = rdMolTransforms.GetDihedralDeg(conf, *key)
                lo, hi = periodic_window((phi - _SP2_HOLD_WIN, phi + _SP2_HOLD_WIN))
                ff.UFFAddTorsionConstraint(*key, False, lo, hi, _SP2_HOLD_FC)


def _coordination_owned_carbons(mol, cons):
    """Return the carbon sites a coordination sphere owns the geometry of.

    A filled aromatic framework takes ligand cleanup unless a haptic face or a compiled donor direction owns it.
    """
    contacts = {tuple(sorted(pair)) for pair in cons.contacts.distances}
    haptic = {atom for face in cons.haptic.values() for atom in face}
    oriented = donor_directions(
        cons.angles, cons.coplanar, cons.metals, releasable=cons.contacts.angular, fixed=cons.fixed
    )
    sites = set()
    for left, right in cons.distances:
        if tuple(sorted((left, right))) in contacts or (left in cons.metals) == (right in cons.metals):
            continue
        site = right if left in cons.metals else left
        sites.update(cons.haptic.get(site, (site,)))
    return {
        atom
        for atom in sites
        if atom < mol.GetNumAtoms()
        and mol.GetAtomWithIdx(atom).GetAtomicNum() == CARBON_Z
        and (atom in haptic or atom in oriented or not no_aromatic_axis(mol.GetAtomWithIdx(atom)))
    }


class ConjugationCap(Mechanism):
    """Pull organic conjugated C=X-N/O torsions to the nearest in-plane well.

    It targets the C-X torsion, unlike `Sp2Planar`'s improper, and reads the same quartets as the geometry check. A
    coordinated pi carbon gets no cap, while a ligand amide or enamine is treated as a metal-free one.
    """

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        if not cons.conjugation:
            return
        mol = conf.GetOwningMol()
        flex = _coordination_owned_carbons(mol, cons)
        for a, c, x, s in conjugated_quartets(mol):
            key = (a, c, x, s)
            if {a, c, x} & flex or requested_dihedral(cons, *key) or graft_owns(key, cons.frozen):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, *key)
            lo, hi = _coplanar_window(phi, _CONJ_CAP)
            ff.UFFAddTorsionConstraint(*key, False, lo, hi, _COPLANAR_FC)


class Umbrella(Mechanism):
    """Hold a metal pyramid or tetrahedral point on its seeded side with soft FF impropers.

    D-M-D angle walls alone let a pyramid flatten or a plane pucker; this restores the out-of-plane force, with no force
    inside each row's window. It runs only in restrained UFF and guarantees no stereo, so acceptance still checks it.
    """

    @override
    def uff_terms(self, ff, cons, conf, stiffness):
        for key, row in cons.umbrellas.items():
            if graft_owns(key, cons.frozen, cons.haptic) or requested_dihedral(cons, *key, improper=True):
                continue
            orders = row.torsions(key)
            for atoms in orders:
                lo, hi = row.window(rdMolTransforms.GetDihedralDeg(conf, *atoms))
                ff.UFFAddTorsionConstraint(*atoms, False, lo, hi, _UMBRELLA_FC * row.weight / len(orders))


# one order for both drivers: DG needs distance before angle and coplanar after COMMIT
MECHANISM_ORDER = (
    Frozen(),
    Distance(),
    Pull(),
    Floor(),
    Angle(),
    Dihedral(),
    Coplanar(),
    Plane(),
    Haptic(),
    TrigonalAngle(),
    Sp2Planar(),
    ConjugationCap(),
    Umbrella(),
)


def _cosine_rule_sq(a, b, cosine):
    """Return the squared side opposite an angle of the given cosine between sides `a` and `b`."""
    return a**2 + b**2 - 2 * a * b * cosine


def law_of_cosines(dij, djk, theta_deg):
    """Return the side opposite angle `theta_deg` (degrees) between sides `dij` and `djk`."""
    return math.sqrt(_cosine_rule_sq(dij, djk, math.cos(math.radians(theta_deg))))


def triangle_distances(left, right, angles):
    """Enclose the opposite distance for nonnegative side intervals and angles in degrees."""
    cosine = math.cos(math.radians(angles[0]))
    # This convex quadratic attains its minimum on an edge of the side-length rectangle.
    lower = [max(0.0, _cosine_rule_sq(a, min(max(a * cosine, right[0]), right[1]), cosine)) for a in left]
    lower += [max(0.0, _cosine_rule_sq(min(max(b * cosine, left[0]), left[1]), b, cosine)) for b in right]
    cosine = math.cos(math.radians(angles[1]))
    upper = [max(0.0, _cosine_rule_sq(a, b, cosine)) for a, b in itertools.product(left, right)]
    return math.sqrt(min(lower)), math.sqrt(max(upper))


def opposite_angle(a, b, d):
    """Return the angle in radians between sides `a` and `b` of a triangle whose third side is `d`."""
    return math.acos(max(-1.0, min(1.0, (a * a + b * b - d * d) / (2 * a * b))))


def triangle_angles(left, right, span):
    """Enclose the angle between two positive side intervals, in radians, or return None without support."""
    if span[0] > left[1] + right[1] or span[1] < max(0.0, left[0] - right[1], right[0] - left[1]):
        return None
    lower = min(opposite_angle(a, b, span[0]) for a, b in itertools.product(left, right))
    upper = [opposite_angle(a, b, span[1]) for a, b in itertools.product(left, right)]
    # cosine has no interior stationary point; the edge minimum has a²=b²-d², and corners cover the rest
    for variable, fixed in ((left, right), (right, left)):
        for b in fixed:
            a = math.sqrt(max(0.0, b * b - span[1] * span[1]))
            if variable[0] <= a <= variable[1]:
                upper.append(opposite_angle(a, b, span[1]))
    return lower, max(upper)


def _coplanar_bound(bm, atoms, window, anchor, cap):
    """Enclose a syn ceiling or anti floor using the committed four-point distance intervals.

    At j, with theta=angle(i,j,k), beta=angle(k,j,w) and gamma=angle(i,j,w),
    cos(gamma)=cos(theta)cos(beta)+sin(theta)sin(beta)cos(phi); the cap edge and rectangle corners bound the required
    cosine extremum. Dropping beta-length correlations is conservative, not always tight.
    """
    if anchor not in (0.0, STRAIGHT) or not 0 <= cap <= _RIGHT_ANGLE or not 0 <= window[0] <= window[1] <= STRAIGHT:
        return None
    i, j, k, w = atoms
    legs = []
    for left, right in ((i, j), (j, k), (k, w), (j, w)):
        a, b = sorted((left, right))
        legs.append((float(bm[b, a]), float(bm[a, b])))
    if any(not 0 < lo <= hi or not math.isfinite(hi) for lo, hi in legs):
        return None
    ij, jk, kw, jw = legs
    beta = triangle_angles(jk, jw, kw)
    if beta is None:
        return None
    anti = anchor == STRAIGHT
    edge = (-1.0 if anti else 1.0) * math.cos(math.radians(cap))
    cosines = [
        math.cos(t) * math.cos(u) + edge * math.sin(t) * math.sin(u)
        for t, u in itertools.product(map(math.radians, window), beta)
    ]
    # anti maximizes cosine and syn minimizes it; an edge stationary point is the opposite extremum
    cosine = max(-1.0, min(1.0, max(cosines) if anti else min(cosines)))
    if anti:
        # The convex distance quadratic attains its minimum on an edge, possibly between its corners.
        values = [_cosine_rule_sq(a, min(max(a * cosine, jw[0]), jw[1]), cosine) for a in ij]
        values += [_cosine_rule_sq(min(max(b * cosine, ij[0]), ij[1]), b, cosine) for b in jw]
        return math.sqrt(max(0.0, min(values)))
    return math.sqrt(max(0.0, *(_cosine_rule_sq(a, b, cosine) for a, b in itertools.product(ij, jw))))


def _coplanar_window(phi, cap, anchor=None):
    """Return a one-sided torsion window from ``phi`` to a requested or nearest in-plane well.

    One-sided means riding the wall reaches the target; RDKit handles bounds outside ±180° periodically.
    """
    well = (
        0.0 if anchor == 0.0 or (anchor is None and abs(phi) < _RIGHT_ANGLE) else (STRAIGHT if phi >= 0 else -STRAIGHT)
    )
    return (well, well + cap) if phi >= well else (well - cap, well)
