"""Constraint writers for edited ETKDG bounds and restrained UFF.

Field mechanisms keep the DG and FF representations together when both exist. `bounds.py` runs WINDOW and
RELIEVE in registry order, commits pending pairs, runs POST, then smooths. Restrained UFF runs the additive FF
hooks in the same registry order. Frozen, Pull and Umbrella are FF-only; Haptic is DG-only; the organic UFF
repairs read the molecule.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from . import metal_donor_orient as _donor  # module import keeps the gate and caps on the same functions
from .constraints import _DIST_ATOMS, FIX_ANGLE_TOL, FIX_DISTANCE_TOL
from .utils import _CARBON_Z, _DISCONNECTED, _SP2_DEGREE, conjugated_quartets

_PHANTOM_FLOOR = 0.30  # Å: a haptic centroid dummy may sit this close to any atom, living inside its own ring
_RIGHT_ANGLE = 90.0  # deg: syn/anti split and the open end of a pyramidal improper
_COPLANAR_SCAN = 12  # samples across the stated M-D-X angle window when sizing a coplanarity bound
_PLANE_PAD = 0.3  # Å half-width on a pi-stack cross-ring distance
_MIN_HALF_WIDTH = 0.01  # Å: avoids division by zero for a near-zero contact window
_FLOOR_INF = 1e3  # Å: practical infinity for a one-sided floor

# UFF walls are E = 1/2 k dev^2 outside their window. Bond stretches are about 700 kcal/mol/Å² and angle bends
# 0.106 kcal/mol/deg². They need separate constants because measured angle legs span 1.01-3.10 Å.
PIN_FC = 1e4  # kcal/mol/Å²: stated distances sit on the measured 500-1e5 metal-fidelity plateau
RELEASABLE_FC_SCALE = 0.3  # NCI wall scale; measured p90 overshoot 0.014 Å
CONTACT_PUSH = 320.0  # kcal/mol/Å at either wall; measured knee for contact centring
ANGLE_FC = 30.0  # kcal/mol/deg²; flat measured fidelity, while 1000 broke 12% of windows
PI_STACK_FC = 2e3  # kcal/mol/Å²: softer than a stated distance because the seed already formed the stack
TARGET_FC = 1e4  # kcal/mol/Å²: holds an M-L target to 0.004 Å against a measured 43 kcal/mol/Å pull
FIX_DISTANCE_FC = 3e7  # kcal/mol/Å²: measured 1.557 -> 1.55760 Å against the reactive-pair LJ repulsion
FIX_ANGLE_FC = 1e3  # kcal/mol/deg²: measured 123.456 -> 123.45543 degrees before the acceptance gate
_COPLANAR_FC = 10.0  # kcal/mol/deg²: below 5 tears a diphosphine on a rigid diene
_UMBRELLA_FC = 3.0  # kcal/mol/deg²: shortest M-L systems need 2-3 to retain the declared side of the plane
# Planar crystals have p95 9.83°; 15° gives RMS 0.063r and is 43% of the 35.264° pyramid improper.
_PLANAR_CAP = 15.0  # deg: keeps a held plane outside the pyramid basin
_STRAIGHT = 180.0
_SP2_HOLD_FC = 10.0  # kcal/mol/deg²: thiourea stays clean down to 3; the window, not k, preserves bowls
_SP2_HOLD_WIN = 5.0  # deg around the seed improper: preserve existing curvature, never create it
_CONJ_CAP = 20.0  # deg: inside the 30° conjugation gate, measured on BIMP, Takemoto and Schreiner


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


def _stated_dihedral(cons, *atoms):
    """Return whether the user states any torsion around this central bond."""
    bond = frozenset(atoms[1:3])
    return any(frozenset(key[1:3]) == bond for key in cons.dihedrals)


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
            self._topo = Chem.GetDistanceMatrix(self.mol)
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
    """Write distance windows to both DG and FF, scaling only contacts marked releasable."""

    def _dg_windows(self, cons, ctx):
        ctx.pairs.update(cons.distances)  # override only the constrained pairs; RDKit keeps the rest

    def _ff_terms(self, ff, cons, conf, stiffness):
        releasable, _ = cons.contacts
        for (i, j), (lo, hi) in cons.distances.items():
            fixed = cons.fixed.get((i, j))
            if fixed is not None and fixed[0] != fixed[1]:
                used_lo, used_hi = _inside(fixed, FIX_DISTANCE_TOL)
                fc = FIX_DISTANCE_FC
            else:
                used_lo, used_hi = lo, hi
                fc = PIN_FC * (RELEASABLE_FC_SCALE if (i, j) in releasable else 1.0)
            ff.AddDistanceConstraint(i, j, used_lo, used_hi, stiffness * fc)


class Pull(Mechanism):
    """Bias a pair to a target inside its flat-bottomed distance wall; FF-only."""

    def _ff_terms(self, ff, cons, conf, stiffness):
        for (i, j), target in cons.pulls.items():
            if (i, j) in cons.fixed:
                continue
            ff.AddDistanceConstraint(i, j, target, target, TARGET_FC)
        for atoms, (lo, hi) in cons.fixed.items():
            if len(atoms) == _DIST_ATOMS and lo == hi:  # scalar distance fix; ranges use the strict wall above
                ff.AddDistanceConstraint(*atoms, lo, hi, stiffness * FIX_DISTANCE_FC)
        # A contact has no UFF bond keeping it inside its window, so bias it to the midpoint without storing
        # that derived target a second time.
        releasable, _ = cons.contacts
        for key in releasable:
            window = cons.distances.get(key)
            if window is None or key in cons.pulls:  # a metal pull already states where this pair sits
                continue
            half = max(0.5 * (window[1] - window[0]), _MIN_HALF_WIDTH)
            mid = 0.5 * (window[0] + window[1])
            ff.AddDistanceConstraint(key[0], key[1], mid, mid, CONTACT_PUSH / half)


class Floor(Mechanism):
    """Keep non-donors off the zero-vdW metal while relieving RDKit's false carbon-vdW DG floor."""

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
            ang_lo, ang_hi = _law_of_cosines(dij, djk, lo), _law_of_cosines(dij, djk, hi)
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

    Size the bound over the full M-D-X window because proper and improper 1,4 distances vary in opposite
    directions. Without a stated M-D-X angle, only the Cartesian FF torsion applies.
    """

    def _dg_post(self, cons, ctx):
        for i, j, k, w, anchor, cap in cons.coplanar:
            key = (min(i, w), max(i, w))
            if key in cons.distances:  # a stated distance window is truth; an angle-derived prior only intersects
                continue
            window = cons.angles.get((i, j, k)) or cons.angles.get((k, j, i))
            if window is None:  # nothing pins M-D-X, so say nothing
                continue
            d_ij, d_jk, d_kw, d_jw = ctx.leg(i, j), ctx.leg(j, k), ctx.leg(k, w), ctx.leg(j, w)
            cj = (d_jk**2 + d_kw**2 - d_jw**2) / (2 * d_jk * d_kw)  # j-k-w from the matrix legs (all real)
            th_jkw = math.degrees(math.acos(max(-1.0, min(1.0, cj))))
            anti = anchor >= _RIGHT_ANGLE  # the metal is anti (far) to w in-plane vs syn (near)
            phi = anchor - cap if anti else anchor + cap
            lo, hi = window
            edges = [
                _chain_distance(d_ij, d_jk, d_kw, lo + (hi - lo) * s / _COPLANAR_SCAN, th_jkw, phi)
                for s in range(_COPLANAR_SCAN + 1)
            ]
            a, b = key  # a < b; bm[b][a] is the lower bound, bm[a][b] the upper
            if anti and ctx.bm[b][a] < min(edges) <= ctx.bm[a][b]:  # anti: floor at the cap edge
                ctx.bm[b][a] = min(edges)
            elif not anti and ctx.bm[b][a] <= max(edges) < ctx.bm[a][b]:  # syn: ceiling at the cap edge
                ctx.bm[a][b] = max(edges)

    def _ff_terms(self, ff, cons, conf, stiffness):
        if not cons.coplanar:  # organic / no metal: nothing to hold, and skip the hybridisation perception cost
            return
        # A conjugated bidentate's plane is already pinned by the polyhedron's two M-D distances plus the
        # L-M-L bite, two coplanar contacts being enough. Per-donor FF torsions add rather than intersect, so
        # redundant ones fight and twist a crowded chelate's backbone: skip the FF torsion for any donor a
        # co-donor shares the plane with. FF-only: the DG bound composes via the matrix intersect and stays.
        mol = conf.GetOwningMol()
        hyb = _donor._stripped_hybridisation(mol)
        donors = {  # every donor of the metal, from the M-donor distance windows (a co-donor need not be capped)
            b if a in cons.metals else a for a, b in cons.distances if a in cons.metals or b in cons.metals
        } or {e[1] for e in cons.coplanar}
        for i, j, k, w, _anchor, cap in cons.coplanar:
            if _stated_dihedral(cons, i, j, k, w):
                continue
            if _donor.codonor_in_plane(mol, j, donors, hyb):
                continue  # a redundant restatement of the bite-pinned plane; see above
            phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
            lo, hi = _coplanar_window(phi, cap)
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
    """Relieve RDKit's ~3.4 Å floor so a haptic centroid can sit at its sub-vdW face radius; DG-only."""

    def _dg_relief(self, cons, ctx):
        n = len(ctx.bm)
        for p in cons.phantoms:
            for x in range(n):
                a, b = (p, x) if p < x else (x, p)
                if x == p or (a, b) in ctx.pairs:  # itself, or an explicit centroid window -> leave it
                    continue
                ctx.bm[b][a] = min(ctx.bm[b][a], _PHANTOM_FLOOR)  # bm[b][a] is the lower bound; only ever lower


class Sp2Planar(Mechanism):
    """Preserve each organic sp2 carbon's seed improper; never target flat.

    UFF puckers a planar conjugated carbon from 0.003 to 0.20 Å. A seed-centred window stops added pucker
    without inventing curvature. Metal systems are excluded because the stripped surrogate hides coordinated
    carbons from the matching geometry exemption and tore five measured cases.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        if cons.metals:  # see class docstring: the Li surrogate defeats gate-matching on a metal system
            return
        mol = conf.GetOwningMol()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != _CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            if atom.GetIdx() in cons.frozen:  # a frozen-core atom is held with zero DOF; never double-restrain it
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) != _SP2_DEGREE:
                continue
            key = (nbrs[0], nbrs[1], nbrs[2], atom.GetIdx())
            if _stated_dihedral(cons, *key):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, *key)
            ff.UFFAddTorsionConstraint(*key, False, phi - _SP2_HOLD_WIN, phi + _SP2_HOLD_WIN, _SP2_HOLD_FC)


class ConjugationCap(Mechanism):
    """Pull organic conjugated C=X-N/O torsions to the nearest in-plane well.

    This targets the C-X torsion, unlike `Sp2Planar`'s seed-centred improper, and shares
    `conjugated_quartets` with the geometry gate. Metal systems and quartets touching frozen atoms are excluded.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        if cons.metals:  # see class docstring: the Li surrogate defeats gate-matching on a metal system
            return
        for a, c, x, s in conjugated_quartets(conf.GetOwningMol()):
            key = (a, c, x, s)
            if _stated_dihedral(cons, *key) or cons.frozen.intersection(key):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, *key)
            lo, hi = _coplanar_window(phi, _CONJ_CAP)
            ff.UFFAddTorsionConstraint(*key, False, lo, hi, _COPLANAR_FC)


class Umbrella(Mechanism):
    """Hold a metal on the declared side of a donor plane with one FF improper.

    The D-M-D walls alone flatten a pyramid to 117.5° or pucker a plane to 112°. A pyramid uses its record's
    ideal improper and seed hand; a plane uses ±`_PLANAR_CAP`. This biases only the UFF seed and is absent from
    DG and downstream real-energy optimization. Replacing it with D-M-D angle targets crushed a side-on imine's
    Ni-N distance to 1.42 Å.
    """

    def _ff_terms(self, ff, cons, conf, stiffness):
        for key, ideal in cons.umbrellas.items():
            if cons.frozen.intersection(key):  # a fix= core already pins this geometry exactly
                continue
            if _stated_dihedral(cons, *key):
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, *key)
            lo, hi = (
                (-_PLANAR_CAP, _PLANAR_CAP)  # a plane has one ideal (zero) and no hand, so the cap is symmetric
                if ideal is None
                else (ideal, _RIGHT_ANGLE)
                if phi >= 0
                else (-_RIGHT_ANGLE, -ideal)  # keep the seed's own hand
            )
            ff.UFFAddTorsionConstraint(*key, False, lo, hi, _UMBRELLA_FC)


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
    Sp2Planar(),
    ConjugationCap(),
    Umbrella(),
)


def _law_of_cosines(dij, djk, theta_deg):
    return math.sqrt(dij**2 + djk**2 - 2 * dij * djk * math.cos(math.radians(theta_deg)))


def _chain_distance(d_ij, d_jk, d_kl, th_ijk, th_jkl, phi):
    """1,4 distance d(i,l) of a chain i-j-k-l from its bonds, two angles and dihedral (deg).

    Purely geometric: the k-l "bond" need not be one, which is why the same formula serves a proper dihedral
    and an improper.
    """
    ti, tj, ph = math.radians(th_ijk), math.radians(th_jkl), math.radians(phi)
    p_i = np.array([d_ij * math.cos(ti), d_ij * math.sin(ti), 0.0])
    p_w = np.array([d_jk - d_kl * math.cos(tj), d_kl * math.sin(tj) * math.cos(ph), d_kl * math.sin(tj) * math.sin(ph)])
    return float(np.linalg.norm(p_i - p_w))


def _coplanar_window(phi, cap):
    """Return a one-sided torsion window from ``phi`` to its nearest 0 or ±180° in-plane well.

    One-sided means riding the wall reaches the target; RDKit handles bounds outside ±180° periodically.
    """
    well = 0.0 if abs(phi) < _RIGHT_ANGLE else (_STRAIGHT if phi >= 0 else -_STRAIGHT)
    return (well, well + cap) if phi >= well else (well - cap, well)
