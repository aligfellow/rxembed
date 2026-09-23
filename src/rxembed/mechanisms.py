"""Constraint writers for edited RDKit distance-geometry bounds and restrained UFF.

Field mechanisms keep the DG and FF representations together when both exist. `bounds.py` runs WINDOW and
RELIEVE in registry order, commits pending pairs, runs POST, then smooths. Restrained UFF runs the additive FF
hooks in the same registry order. Frozen, Pull and Umbrella are FF-only; the organic UFF repairs read the
molecule.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

from .constraints import (
    _DIST_ATOMS,
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    _graft_owns,
    _merge_pulls,
    _periodic_window,
    _stated_dihedral,
)
from .metal_core import disconnect_metal
from .utils import _CARBON_Z, _DISCONNECTED, _SP2_DEGREE, conjugated_quartets

_PHANTOM_FLOOR = 0.30  # Å: a haptic centroid dummy may sit this close to any atom, living inside its own ring
_RIGHT_ANGLE = 90.0  # deg: syn/anti split and the open end of a pyramidal improper
_PLANE_PAD = 0.3  # Å half-width on a pi-stack cross-ring distance
_MIN_HALF_WIDTH = 0.01  # Å: avoids division by zero for a near-zero contact window
_FLOOR_INF = 1e3  # Å: practical infinity for a one-sided floor

# RDKit DistanceConstraint uses E = 1/2 k dev²; AngleConstraint and TorsionConstraint use k dev² in degrees.
# Separate distance/angular constants are necessary: measured angle legs span 1.01-3.10 Å.
PIN_FC = 1e4  # kcal/mol/Å²: stated distances sit on the measured 500-1e5 metal-fidelity plateau
RELEASABLE_FC_SCALE = 0.3  # NCI wall scale; measured p90 overshoot 0.014 Å
CONTACT_PUSH = 320.0  # kcal/mol/Å at either wall; measured knee for contact centring
ANGLE_FC = 30.0  # kcal/mol/deg²; flat measured fidelity, while 1000 broke 12% of windows
PI_STACK_FC = 2e3  # kcal/mol/Å²: softer than a stated distance because the seed already formed the stack
TARGET_FC = 1e4  # kcal/mol/Å²: holds an M-L target to 0.004 Å against a measured 43 kcal/mol/Å pull
_ANGLE_TARGET_FC = 0.001 * ANGLE_FC  # kcal/mol/deg²: common-shell benchmark, 19/25 fresh vs 15/25 with walls alone
FIX_DISTANCE_FC = 3e7  # kcal/mol/Å²: measured 1.557 -> 1.55760 Å against the reactive-pair LJ repulsion
FIX_ANGLE_FC = 1e3  # kcal/mol/deg²: measured 123.456 -> 123.45543 degrees before the acceptance gate
_COPLANAR_FC = 10.0  # kcal/mol/deg²: below 5 tears a diphosphine on a rigid diene
_UMBRELLA_FC = 3.0  # kcal/mol/deg²: shortest M-L systems need 2-3 to retain the declared side of the plane
# Planar crystals have p95 9.83°; 15° gives RMS 0.063r and is 43% of the 35.264° pyramid improper.
_PLANAR_CAP = 15.0  # deg: keeps a held plane outside the pyramid basin
_STRAIGHT = 180.0
_SP2_HOLD_FC = 10.0 / 3.0  # kcal/mol/deg²: three ordered terms share the measured total restraint strength
_SP2_HOLD_WIN = 5.0  # deg around the seed improper: preserve existing curvature, never create it
_CONJ_CAP = 20.0  # deg: inside the 30° conjugation gate, measured on BIMP, Takemoto and Schreiner
_UFF_SMALL_SP2_RINGS = (3, 4)  # RDKit UFF uses nonperiodic angle potentials for these ring sizes


def _inside(window, margin):
    """Inset a flat wall so its finite-force equilibrium remains inside the public fixed range."""
    lo, hi = window
    mid = 0.5 * (lo + hi)
    return min(lo + margin, mid), max(hi - margin, mid)


def _angular_wall(atoms, window, cons, stiffness, releasable):
    """Select the shared soft or strict angular wall and force constant."""
    fixed = cons.fixed.get(atoms)
    if fixed is not None:
        bounds = fixed if fixed[0] == fixed[1] else _inside(fixed, FIX_ANGLE_TOL)
        return *bounds, stiffness * FIX_ANGLE_FC
    # Never escalate a soft angular wall: a stiffer one displaced one measured structure by 0.67 Å.
    scale = RELEASABLE_FC_SCALE if atoms in releasable else 1.0
    return *window, min(stiffness, 1.0) * ANGLE_FC * scale


@dataclass
class DGContext:
    """Everything a DG writer needs beyond the Constraints: the molecule, the matrix, and the pending windows."""

    mol: Chem.Mol
    bm: np.ndarray
    pairs: dict = field(default_factory=dict)  # (i, j) -> (lo, hi), committed into `bm` after RELIEVE
    _topo: np.ndarray | None = None

    @property
    def topo(self):
        """Bond-path lengths on the metal-stripped graph; a real backbone gives a finite distance."""
        if self._topo is None:
            # A donor's temporary M-L stereo carrier is not a ligand-backbone path. RDKit may also retain
            # a distance cache from before that bond was added; neither may change angle-prior ownership.
            self._topo = Chem.GetDistanceMatrix(disconnect_metal(self.mol), force=True)
        return self._topo

    def mid(self, i, j):
        """Midpoint of the matrix's current window for a pair."""
        a, b = (i, j) if i < j else (j, i)
        return 0.5 * (self.bm[a][b] + self.bm[b][a])

    def leg(self, i, j):
        """Angle-leg length: an explicit window's midpoint, else the (possibly bond-less) matrix default.

        Reads `pairs` while the angle writer is still filling it, so an angle written earlier supplies the leg
        for one written later. Insertion order is therefore load-bearing.
        """
        v = self.pairs.get((min(i, j), max(i, j)))
        return 0.5 * (v[0] + v[1]) if v else self.mid(i, j)


class Mechanism:
    """A constraint field's two writers. Subclasses implement only the hooks their field needs."""

    def _dg_windows(self, cons, ctx):
        """WINDOW: contribute candidate ``(lo, hi)`` windows to ``ctx.pairs``."""

    def _dg_relief(self, cons, ctx):
        """RELIEVE: lower a matrix bound. Runs before COMMIT, so an explicit window always wins."""

    def _dg_post(self, cons, ctx):
        """POST: read the committed matrix and tighten it."""

    def _ff_terms(self, ff, cons, conf, stiffness):
        """FF: add terms for this field at the requested stiffness rung."""


class Frozen(Mechanism):
    """Pin atoms at their embedded coordinates with zero degrees of freedom; FF-only."""

    def _ff_terms(self, ff, cons, conf, stiffness):
        for idx in cons.frozen:
            ff.AddFixedPoint(idx)


class Distance(Mechanism):
    """Write distance windows to DG and FF; Haptic replaces derived centroid radii during relaxation."""

    def _dg_windows(self, cons, ctx):
        ctx.pairs.update(cons.distances)  # override only the constrained pairs; RDKit keeps the rest

    def _ff_terms(self, ff, cons, conf, stiffness):
        releasable, _ = cons.contacts
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


class Pull(Mechanism):
    """Bias a modelled distance or angle toward its preferred value; FF-only."""

    def _ff_terms(self, ff, cons, conf, stiffness):
        for atoms, target in _merge_pulls({}, cons.pulls).items():
            if atoms in cons.fixed or atoms[::-1] in cons.fixed:
                continue
            if len(atoms) == _DIST_ATOMS:
                ff.AddDistanceConstraint(*atoms, target, target, TARGET_FC)
            else:
                ff.UFFAddAngleConstraint(*atoms, False, target, target, _ANGLE_TARGET_FC)
        for atoms, (lo, hi) in cons.fixed.items():
            if len(atoms) == _DIST_ATOMS and lo == hi:  # scalar distance fix; ranges use the strict wall above
                ff.AddDistanceConstraint(*atoms, lo, hi, stiffness * FIX_DISTANCE_FC)
        # A contact has no UFF bond keeping it inside its window, so bias it to the midpoint without storing
        # that derived target a second time.
        releasable, _ = cons.contacts
        for key in releasable:
            window = cons.distances.get(key)
            if window is None or key in cons.pulls or key[::-1] in cons.pulls:
                continue
            half = max(0.5 * (window[1] - window[0]), _MIN_HALF_WIDTH)
            mid = 0.5 * (window[0] + window[1])
            ff.AddDistanceConstraint(key[0], key[1], mid, mid, CONTACT_PUSH / half)


class Floor(Mechanism):
    """Apply one-sided FF distance walls and relieve false metal-surrogate DG floors."""

    def _dg_relief(self, cons, ctx):
        # Never relieve onto an atom inside a rigid body: it has 6 dof not 3n, every internal distance is
        # already stated, and lowering a floor there only lets the DG fold the body inward.
        rigid = set().union(*cons.shapes) if cons.shapes else set()
        for (a, b), floor in cons.dg_floors.items():  # keys are (metal, X), sorted; bm[b][a] is the lower bound
            if (a, b) in ctx.pairs:  # an explicit window is the truth; never override it
                continue
            far = b if a in cons.metals else a  # the non-metal end of the pair
            if far in rigid:  # a metal inside a rigid body still gets reliefs to atoms outside it
                continue
            if floor < ctx.bm[b][a] <= ctx.bm[a][b]:  # only ever relax, and only if genuinely too high
                ctx.bm[b][a] = floor

    def _ff_terms(self, ff, cons, conf, stiffness):
        for (i, j), floor in cons.floors.items():
            # Explicit holds override the automatic wall; seed-only encounter windows do not.
            if (i, j) in cons.fixed or (i, j) in cons.contacts[0]:
                continue
            ff.AddDistanceConstraint(i, j, floor, _FLOOR_INF, stiffness * PIN_FC)


class Angle(Mechanism):
    """Write an FF angle wall and its law-of-cosines 1-3 DG prior.

    Intersect the prior with a connected backbone; write it outright only across the stripped metal. A
    disjoint intersection keeps RDKit's backbone bounds.
    """

    def _dg_windows(self, cons, ctx):
        for (i, j, k), (lo, hi) in cons.angles.items():
            a, b = min(i, k), max(i, k)  # sorted, as `add_distance` stores them
            if (a, b) in ctx.pairs:  # an explicit distance window already owns this pair
                continue
            dij, djk = ctx.leg(i, j), ctx.leg(j, k)
            # Scalar fixed-angle seed padding can extend beyond the physical domain used by UFF.
            ang_lo = _law_of_cosines(dij, djk, max(0.0, lo))
            ang_hi = _law_of_cosines(dij, djk, min(_STRAIGHT, hi))
            if ctx.topo[a][b] < _DISCONNECTED:
                lo_hi = (max(ang_lo, ctx.bm[b][a]), min(ang_hi, ctx.bm[a][b]))
                ctx.pairs[(a, b)] = lo_hi if lo_hi[0] <= lo_hi[1] else (ctx.bm[b][a], ctx.bm[a][b])
            else:
                ctx.pairs[(a, b)] = (ang_lo, ang_hi)

    def _ff_terms(self, ff, cons, conf, stiffness):
        _, releasable = cons.contacts
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
    """Write periodic dihedral windows to UFF; distance geometry cannot encode their signed hand."""

    def _ff_terms(self, ff, cons, conf, stiffness):
        _, releasable = cons.contacts
        for atoms, window in cons.dihedrals.items():
            used_lo, used_hi, fc = _angular_wall(atoms, window, cons, stiffness, releasable)
            ff.UFFAddTorsionConstraint(*atoms, False, used_lo, used_hi, fc)


class Coplanar(Mechanism):
    """Restore the donor-plane term lost with the M-donor bond: a POST 1,4 bound and soft FF torsion.

    Project the complete distance and M-D-X angle windows; midpoint legs can exclude a valid chelate.
    Without a stated M-D-X angle, only the Cartesian FF torsion applies.
    """

    def _dg_post(self, cons, ctx):
        for i, j, k, w, anchor, cap in cons.coplanar:
            if _stated_dihedral(cons, i, j, k, w):  # the same override owns the seed and the FF
                continue
            if anchor is None:  # the graph proves a plane but not which periodic well
                continue
            key = (min(i, w), max(i, w))
            if key in cons.distances:  # a stated distance window is truth; an angle-derived prior only intersects
                continue
            window = cons.angles.get((i, j, k)) or cons.angles.get((k, j, i))
            if window is None:  # nothing pins M-D-X, so say nothing
                continue
            edge = _coplanar_bound(ctx.bm, (i, j, k, w), window, anchor, cap)
            if edge is None:
                continue
            anti = anchor >= _RIGHT_ANGLE  # the metal is anti (far) to w in-plane vs syn (near)
            a, b = key  # a < b; bm[b][a] is the lower bound, bm[a][b] the upper
            if anti and ctx.bm[b][a] < edge <= ctx.bm[a][b]:  # anti: floor at the cap edge
                ctx.bm[b][a] = edge
            elif not anti and ctx.bm[b][a] <= edge < ctx.bm[a][b]:  # syn: ceiling at the cap edge
                ctx.bm[a][b] = edge

    def _ff_terms(self, ff, cons, conf, stiffness):
        if not cons.coplanar:
            return
        for i, j, k, w, anchor, cap in cons.coplanar:
            if _graft_owns((i, j, k, w), cons.frozen, cons.haptic):
                continue
            if _stated_dihedral(cons, i, j, k, w):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
            lo, hi = _coplanar_window(phi, cap, anchor)
            ff.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC)


class Plane(Mechanism):
    """Hold a parallel pi-stack by cross-ring distances; explicit and angle-derived windows win."""

    def _dg_windows(self, cons, ctx):
        for ring_a, ring_b, sep in cons.planes:  # cross-ring d = sqrt(sep^2 + in-plane^2)
            for u, au in enumerate(ring_a):
                for v, bv in enumerate(ring_b):
                    offset = 0.0 if u == v else ctx.mid(au, ring_a[v])
                    d = math.hypot(sep, offset)
                    ctx.pairs.setdefault((min(au, bv), max(au, bv)), (d - _PLANE_PAD, d + _PLANE_PAD))

    def _ff_terms(self, ff, cons, conf, stiffness):
        pos = conf.GetPositions()  # hold the stack as embedded; the seed already realised the separation
        for ring_a, ring_b, _sep in cons.planes:
            for a, b in itertools.product(ring_a, ring_b):
                d = float(np.linalg.norm(pos[a] - pos[b]))
                ff.AddDistanceConstraint(a, b, d - _PLANE_PAD, d + _PLANE_PAD, stiffness * PI_STACK_FC)


class Haptic(Mechanism):
    """Seat the DG helper inside its face and restrain it to the moving centroid during UFF.

    Native zero-distance springs implement E = K/2 |p - mean(r)|² through
    sum_i |p-r_i|²/n - sum_i<j |r_i-r_j|²/n². The complete sum is nonnegative: the negative terms cancel
    internal face strain, leaving no radius or planarity target. This finite penalty is not an exact
    dependent coordinate; publication measures the real centroid independently.
    """

    def _dg_relief(self, cons, ctx):
        n = len(ctx.bm)
        for p in cons.phantoms:
            for x in range(n):
                a, b = (p, x) if p < x else (x, p)
                if x == p or (a, b) in ctx.pairs:  # itself, or an explicit centroid window -> leave it
                    continue
                ctx.bm[b][a] = min(ctx.bm[b][a], _PHANTOM_FLOOR)  # bm[b][a] is the lower bound; only ever lower

    def _ff_terms(self, ff, cons, conf, stiffness):
        for dummy, face in cons.haptic.items():
            n = len(face)
            force = stiffness * PIN_FC
            for atom in face:
                ff.UFFAddDistanceConstraint(dummy, atom, False, 0.0, 0.0, force / n)
            for left, right in itertools.combinations(face, 2):
                ff.UFFAddDistanceConstraint(left, right, False, 0.0, 0.0, -force / n**2)


class TrigonalAngle(Mechanism):
    """Remove UFF's collapsed trigonal angle basin using its native stiffness.

    RDKit uses k(1-cos(3 theta))/9 for every SP2 atom, not only carbon. Its #7901 correction starts at 30
    degrees, where constrained optimization can stall. A native-curvature half-harmonic below 90 degrees removes
    that basin without changing energies or forces at ordinary trigonal angles. Its wall extends past the
    60-degree saddle: ending there still allows coupled angles to trap cleanup. Native small-ring angles use
    a different potential and are untouched. Stated geometry and existing pair floors remain authoritative.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
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
                    or _graft_owns((a, centre, b), cons.frozen)
                ):
                    continue
                params = rdForceFieldHelpers.GetUFFAngleBendParams(mol, a, centre, b)
                if params is None:
                    raise RuntimeError(f"UFF lacks angle parameters for {a}-{centre}-{b}")
                force, _ = params
                # RDKit's constraint energy is K*delta_degrees**2, without the harmonic one-half factor.
                ff.UFFAddAngleConstraint(
                    a, centre, b, False, _RIGHT_ANGLE, _STRAIGHT, 0.5 * force * math.radians(1) ** 2
                )


class Sp2Planar(Mechanism):
    """Preserve each organic sp2 carbon's seed improper; never target flat.

    UFF puckers a planar conjugated carbon from 0.003 to 0.20 Å. A seed-centred window stops added pucker
    without inventing curvature. A coordinated carbon's improper is excluded because its coordination state
    owns that geometry. Three ordered torsions are needed because RDKit exposes no permutation-invariant
    improper restraint.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        if not cons.conjugation:
            return
        mol = conf.GetOwningMol()
        coordinated = _coordination_owned_carbons(mol, cons)
        stated = {frozenset(atoms) for atoms in cons.dihedrals}
        for atom in mol.GetAtoms():
            centre = atom.GetIdx()
            if atom.GetAtomicNum() != _CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) != _SP2_DEGREE:
                continue
            if _graft_owns((centre, *nbrs), cons.frozen, cons.haptic):
                continue
            if centre in coordinated or frozenset((centre, *nbrs)) in stated:
                continue
            a, b, c = nbrs
            for key in ((a, b, c, centre), (b, a, c, centre), (c, a, b, centre)):
                phi = rdMolTransforms.GetDihedralDeg(conf, *key)
                lo, hi = _periodic_window((phi - _SP2_HOLD_WIN, phi + _SP2_HOLD_WIN))
                ff.UFFAddTorsionConstraint(*key, False, lo, hi, _SP2_HOLD_FC)


def _coordination_owned_carbons(mol, cons):
    """Return carbon sites whose geometry is owned by a coordination sphere."""
    contacts = {tuple(sorted(pair)) for pair in cons.contacts[0]}
    sites = set()
    for left, right in cons.distances:
        if tuple(sorted((left, right))) in contacts or (left in cons.metals) == (right in cons.metals):
            continue
        site = right if left in cons.metals else left
        sites.update(cons.haptic.get(site, (site,)))
    return {atom for atom in sites if atom < mol.GetNumAtoms() and mol.GetAtomWithIdx(atom).GetAtomicNum() == _CARBON_Z}


class ConjugationCap(Mechanism):
    """Pull organic conjugated C=X-N/O torsions to the nearest in-plane well.

    This targets the C-X torsion, unlike `Sp2Planar`'s seed-centred improper, and shares
    `conjugated_quartets` with the geometry gate. A coordinated pi carbon keeps the gate's wider flexible
    window, while an ordinary ligand amide or enamine still receives the same cleanup as a metal-free one.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        if not cons.conjugation:
            return
        mol = conf.GetOwningMol()
        flex = _coordination_owned_carbons(mol, cons)
        for a, c, x, s in conjugated_quartets(mol):
            key = (a, c, x, s)
            if {a, c, x} & flex or _stated_dihedral(cons, *key) or _graft_owns(key, cons.frozen):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, *key)
            lo, hi = _coplanar_window(phi, _CONJ_CAP)
            ff.UFFAddTorsionConstraint(*key, False, lo, hi, _COPLANAR_FC)


class Umbrella(Mechanism):
    """Keep a selected metal pyramid or tetrahedral point on its seeded side with native FF impropers.

    The D-M-D walls alone flatten a pyramid to 117.5° or pucker a plane to 112°. A pyramid uses its record's
    ideal improper and seed hand; a plane uses ±`_PLANAR_CAP`. Zero holds only point handedness, with no force
    on its seeded half-circle. Average the six torsion axes so carrier order cannot change the penalty.
    These soft UFF walls do not guarantee stereo; publication still checks the final geometry. They are absent
    from DG and downstream real-energy optimization. D-M-D angle targets crushed a side-on imine to 1.42 Å.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        for key, ideal in cons.umbrellas.items():
            if _graft_owns(key, cons.frozen, cons.haptic) or _stated_dihedral(cons, *key, improper=True):
                continue
            if isinstance(ideal, tuple):
                anchor, weight = ideal
                ff.UFFAddTorsionConstraint(
                    *key, False, anchor - _PLANAR_CAP, anchor + _PLANAR_CAP, _UMBRELLA_FC * weight
                )
                continue
            orders = (
                {frozenset(order[1:3]): order for order in itertools.permutations(key)}.values()
                if ideal == 0.0
                else (key,)
            )
            for atoms in orders:
                phi = rdMolTransforms.GetDihedralDeg(conf, *atoms)
                if ideal is None:
                    anchor = 0.0 if abs(phi) <= _RIGHT_ANGLE else math.copysign(_STRAIGHT, phi)
                    # RDKit wraps periodic bounds; clipping at 180 would remove half the planar window.
                    lo, hi = anchor - _PLANAR_CAP, anchor + _PLANAR_CAP
                elif ideal == 0.0:
                    lo, hi = (0.0, _STRAIGHT) if phi >= 0 else (-_STRAIGHT, 0.0)
                else:
                    lo, hi = (ideal, _RIGHT_ANGLE) if phi >= 0 else (-_RIGHT_ANGLE, -ideal)
                ff.UFFAddTorsionConstraint(*atoms, False, lo, hi, _UMBRELLA_FC / len(orders))


# One order for both drivers. DG needs distance before angle and coplanar after COMMIT; later FF-only repairs
# have no ordering dependency.
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


def _law_of_cosines(dij, djk, theta_deg):
    return math.sqrt(dij**2 + djk**2 - 2 * dij * djk * math.cos(math.radians(theta_deg)))


def _triangle_distances(left, right, angles):
    """Enclose the opposite distance for nonnegative side intervals and angles in degrees."""

    def square(a, b, cosine):
        return max(0.0, a * a + b * b - 2 * a * b * cosine)

    cosine = math.cos(math.radians(angles[0]))
    # This convex quadratic attains its minimum on an edge of the side-length rectangle.
    lower = [square(a, min(max(a * cosine, right[0]), right[1]), cosine) for a in left]
    lower += [square(min(max(b * cosine, left[0]), left[1]), b, cosine) for b in right]
    cosine = math.cos(math.radians(angles[1]))
    upper = [square(a, b, cosine) for a, b in itertools.product(left, right)]
    return math.sqrt(min(lower)), math.sqrt(max(upper))


def _triangle_angles(left, right, span):
    """Enclose the angle between two positive side intervals, in radians, or return None without support."""
    if span[0] > left[1] + right[1] or span[1] < max(0.0, left[0] - right[1], right[0] - left[1]):
        return None

    def angle(a, b, d):
        return math.acos(max(-1.0, min(1.0, (a * a + b * b - d * d) / (2 * a * b))))

    lower = min(angle(a, b, span[0]) for a, b in itertools.product(left, right))
    upper = [angle(a, b, span[1]) for a, b in itertools.product(left, right)]
    # Cosine has no nondegenerate interior stationary point. Its edge minimum has a²=b²-d²;
    # corner clipping also covers every supported collinear boundary.
    for variable, fixed in ((left, right), (right, left)):
        for b in fixed:
            a = math.sqrt(max(0.0, b * b - span[1] * span[1]))
            if variable[0] <= a <= variable[1]:
                upper.append(angle(a, b, span[1]))
    return lower, max(upper)


def _coplanar_bound(bm, atoms, window, anchor, cap):
    """Enclose a syn ceiling or anti floor using the committed four-point distance intervals.

    At j, let theta=angle(i,j,k), beta=angle(k,j,w), gamma=angle(i,j,w). Then
    cos(gamma)=cos(theta)cos(beta)+sin(theta)sin(beta)cos(phi). The cap edge and rectangle corners bound
    the required cosine extremum. Relaxing beta/length correlations is conservative, not always tight.
    """
    if anchor not in (0.0, _STRAIGHT) or not 0 <= cap <= _RIGHT_ANGLE or not 0 <= window[0] <= window[1] <= _STRAIGHT:
        return None
    i, j, k, w = atoms
    legs = []
    for left, right in ((i, j), (j, k), (k, w), (j, w)):
        a, b = sorted((left, right))
        legs.append((float(bm[b, a]), float(bm[a, b])))
    if any(not 0 < lo <= hi or not math.isfinite(hi) for lo, hi in legs):
        return None
    ij, jk, kw, jw = legs
    beta = _triangle_angles(jk, jw, kw)
    if beta is None:
        return None
    anti = anchor == _STRAIGHT
    edge = (-1.0 if anti else 1.0) * math.cos(math.radians(cap))
    cosines = [
        math.cos(t) * math.cos(u) + edge * math.sin(t) * math.sin(u)
        for t, u in itertools.product(map(math.radians, window), beta)
    ]
    # Anti maximizes cosine, syn minimizes it. Any stationary point along either edge is the opposite
    # extremum: its sine coefficient is nonpositive for anti and nonnegative for syn.
    cosine = max(-1.0, min(1.0, max(cosines) if anti else min(cosines)))

    def square(a, b):
        return a * a + b * b - 2 * a * b * cosine

    if anti:
        # The convex distance quadratic attains its minimum on an edge, possibly between its corners.
        values = [square(a, min(max(a * cosine, jw[0]), jw[1])) for a in ij]
        values += [square(min(max(b * cosine, ij[0]), ij[1]), b) for b in jw]
        return math.sqrt(max(0.0, min(values)))
    return math.sqrt(max(0.0, *(square(a, b) for a, b in itertools.product(ij, jw))))


def _coplanar_window(phi, cap, anchor=None):
    """Return a one-sided torsion window from ``phi`` to a stated or nearest in-plane well.

    One-sided means riding the wall reaches the target; RDKit handles bounds outside ±180° periodically.
    """
    well = (
        0.0
        if anchor == 0.0 or (anchor is None and abs(phi) < _RIGHT_ANGLE)
        else (_STRAIGHT if phi >= 0 else -_STRAIGHT)
    )
    return (well, well + cap) if phi >= well else (well - cap, well)
