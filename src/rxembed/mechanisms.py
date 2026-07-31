"""One class per constraint field, holding its distance-geometry writer and its force-field writer together.

Every field is written twice, as an edit to the ETKDG bounds matrix and as a restrained-UFF term.
Co-locating the two halves is what stops them drifting into a field with a DG bound and no matching FF
term. `MECHANISM_ORDER` at the bottom is the one place their order is stated.

The split is by field, not by source: splitting by source would reorder `cons.angles` iteration, and the
bounds matrix depends on it, `leg()` reading the `pairs` dict the angle loop is still mutating.

Phases, whose order the driver owns in ~6 lines of `bounds.py`:

    WINDOW   dg_windows   distances -> angles -> planes, all into `ctx.pairs` (not yet the matrix)
    RELIEVE  dg_relief    lower RDKit's phantom floors, before the windows are committed
    COMMIT   (driver)     write `ctx.pairs` into the matrix
    POST     dg_post      read the committed matrix (the coplanar 1,4 bound needs the legs)
    SMOOTH   (driver)     triangle smoothing

The FF side is a flat additive loop, so a mechanism there implements only `ff_terms`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# The perception rulers these FF caps read, imported as MODULES (not their names): the gate and the caps must
# see the same function object, so a test that swaps one out swaps it for both.
from . import metal_donor_orient as _donor

# one definition each, in the metal constants hub: RDKit's inter-fragment topological distance, the
# empty-vertex sentinel, and the atom -> ligand-fragment map.
from .metal_core import VACANT, _frag_map
from .metal_polyhedron import POLYHEDRA, resolve_geometry
from .utils import _CARBON_Z, _DISCONNECTED, _SP2_DEGREE, conjugated_quartets

_PHANTOM_FLOOR = 0.30  # Å: a haptic centroid dummy may sit this close to any atom, living inside its own ring
_RIGHT_ANGLE = 90.0  # deg, two landmarks at one number: the syn/anti split of an in-plane anchor
#   (`Coplanar`), and the open end of a pyramid's improper window (`Umbrella`).
_COPLANAR_SCAN = 12  # samples across the stated M-D-X angle window when sizing a coplanarity bound
_PLANE_PAD = 0.3  # Å half-width on a pi-stack cross-ring distance
_FLOOR_HI = 1e3  # Å: a floor is one-sided, [floor, +inf), and 1e3 stands in for infinity
_SOFT_PULL_FC = 1e4  # soft harmonic onto the M-donor target, inside the flat-bottomed wall: the wall forbids
#   leaving the window, this says where in it to sit. Full strength would fight the field and wreck convergence.
_COPLANAR_FC = 10.0  # kcal/deg² (UFF torsion penalty is k*dev², dev in degrees): the two organic-plane
#   caps. Far softer than the distance walls, since real out-of-plane scatter runs to ~40° (tmQM census p95):
#   remove the gross deviation, never pin flat. Below 5 a diphosphine-on-rigid-diene chelate tears.
_UMBRELLA_FC = 3.0  # kcal/deg²: a wall, so what matters is the knee, and it moves with M-L because the wall
#   is a scale-free improper while survival is an absolute out-of-plane RMS. Long M-L saturates by 0.1, short
#   (anionic amide/alkoxide) not until 2-3: sweep an anionic amide before lowering this.
_PLANE_FC_SCALE = 0.2  # a stack hold is softer than a stated distance wall
_STRAIGHT = 180.0
_SP2_HOLD_FC = (
    10.0  # kcal/deg², equal to `_COPLANAR_FC` and kept there: a sweep to 0.01 showed the two demands do not overlap
)
# (a thiourea stays clean to fc 3, a bowled PAH needs <=0.01). The tension is in the window, not the fc.
_SP2_HOLD_WIN = 5.0  # deg half-window around the seed's own improper: hold-at-seed. Preserves a curve the
#   seed has, never re-forms one it lacks (ETKDG seeds corannulene's bowl flat); widening it lets the
#   thiourea over-pucker.
_CONJ_CAP = 20.0  # deg half-window of the organic conjugation torsion cap (`ConjugationCap`). Must sit inside
#   `geometry.conjugation`'s 30° gate: at cap==30 the window rides out to the gate line and still flags, the
#   angular-spring failure. 20° lands flat with margin, measured on bimp/takemoto/schreiner.


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

    def dg_windows(self, cons, ctx):
        """WINDOW: contribute candidate ``(lo, hi)`` windows to ``ctx.pairs``."""

    def dg_relief(self, cons, ctx):
        """RELIEVE: lower a matrix bound. Runs before COMMIT, so an explicit window always wins."""

    def dg_post(self, cons, ctx):
        """POST: read the committed matrix and tighten it."""

    def ff_terms(self, ff, cons, conf, fc):
        """FF: add restraint terms for this field. ``fc`` is the distance force constant."""


class Frozen(Mechanism):
    """Atoms pinned to their embedded coordinates: zero degrees of freedom, not a stiff spring.

    A soft pin lets UFF crush a TS core's sub-Angstrom partial bonds. The DG half is the pairwise shape the
    builder writes into `distances`.
    """

    def ff_terms(self, ff, cons, conf, fc):
        for idx in cons.frozen:
            ff.AddFixedPoint(idx)


class Distance(Mechanism):
    """A distance window: the bounds-matrix cell, and a flat-bottomed wall that forbids leaving it.

    Seeds `ctx.pairs` first, so every later writer's `leg()` reads stated windows rather than matrix defaults.
    """

    def dg_windows(self, cons, ctx):
        ctx.pairs.update(cons.distances)  # override only the constrained pairs; RDKit keeps the rest

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), (lo, hi) in cons.distances.items():
            ff.AddDistanceConstraint(i, j, lo, hi, fc)


class Pull(Mechanism):
    """A soft harmonic at the M-donor target, inside the flat-bottomed wall. No DG half.

    `AddDistanceConstraint` exerts zero force between lo and hi, so without a target the donor rides whichever
    wall it was last pushed to; this says where in the window to sit. Skipped for a rigid-body member, where
    pulling a subset of its windows tears the rest.
    """

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), target in cons.pulls.items():
            ff.AddDistanceConstraint(i, j, target, target, _SOFT_PULL_FC)


class Floor(Mechanism):
    """The anti-overbond guard: a one-sided minimum M...X, and the matching relief in the matrix.

    Two halves of one physical distance that must ship together. The FF `floors` raises a bound, since zeroing
    the surrogate vdW removed the only thing keeping a non-donor off the metal; the DG `dg_floors` lowers one,
    since the bond-less carbon makes RDKit floor every M...X at a ~3.4 Å carbon-vdW contact while real
    second-sphere atoms sit at 2.8-3.0 Å. A `floors` without its twin leaves the matrix forbidding the target.
    """

    def dg_relief(self, cons, ctx):
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

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), floor in cons.floors.items():
            ff.AddDistanceConstraint(i, j, floor, _FLOOR_HI, fc)


class Angle(Mechanism):
    """An angle window: a law-of-cosines 1-3 distance in the matrix, and a UFF angle wall.

    The intersect rule: one predicate, no per-feature special case. An angle-derived distance is a prior, not
    truth. Where a real backbone bond-path connects the end atoms, the matrix already holds their diagonal, so
    the two intersect and the tighter real bound wins; where they meet only through the stripped metal (an
    L-M-L or M-D-X wall) the matrix holds nothing but a phantom carbon-vdW floor, so the angle is written
    outright. This is what lets the chelate bite, polyhedron angles, coplanarity cap and haptic rings compose.
    A disjoint intersection keeps the backbone and discards the angle.
    """

    def dg_windows(self, cons, ctx):
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

    def ff_terms(self, ff, cons, conf, fc):
        afc = min(fc, 1e3)  # deg^-2, not rad^-2: an over-stiff angle distorts a rigid/bidentate framework
        for (i, j, k), (lo, hi) in cons.angles.items():
            ff.UFFAddAngleConstraint(i, j, k, False, max(0.0, lo), min(180.0, hi), afc)


class Coplanar(Mechanism):
    """Keep the metal in a conjugated sp2 donor's plane: a 1,4 distance bound, and a soft UFF torsion.

    An artefact field rather than physics: it exists only because the surrogate strips the M-donor bond and
    its UFF improper. Runs in POST, where the 1,4 bound reads the committed legs and can only tighten.

    The 1,4 distance carries dihedral information only once M-D-X is pinned, so it is read from `cons.angles`
    and sized at the extremum over the whole window. A single fixed angle cannot do this: the dependence on
    M-D-X reverses between a proper dihedral (d rises with the angle) and an improper (d falls), so one guess
    forbids planar geometry on the other, while the extremum is safe on both. A donor with no stated M-D-X
    angle gets no bound, and the FF torsion (a real Cartesian dihedral) still holds it.
    """

    def dg_post(self, cons, ctx):
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

    def ff_terms(self, ff, cons, conf, fc):
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
            if _donor.codonor_in_plane(mol, j, donors, hyb):
                continue  # a redundant restatement of the bite-pinned plane; see above
            phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
            lo, hi = _coplanar_window(phi, cap)
            ff.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC)


class Plane(Mechanism):
    """A parallel pi-stack: cross-ring distances from the stated separation, and pairwise holds in the FF.

    Uses `setdefault`, so a user `constrain=` on a cross-ring pair wins, as does an angle-derived window
    written earlier in the same phase: the stack is a heuristic and the angle is not.
    """

    def dg_windows(self, cons, ctx):
        for ring_a, ring_b, sep in cons.planes:  # cross-ring d = sqrt(sep^2 + in-plane^2)
            for u, au in enumerate(ring_a):
                for v, bv in enumerate(ring_b):
                    offset = 0.0 if u == v else ctx.mid(au, ring_a[v])
                    d = math.hypot(sep, offset)
                    ctx.pairs.setdefault((min(au, bv), max(au, bv)), (d - _PLANE_PAD, d + _PLANE_PAD))

    def ff_terms(self, ff, cons, conf, fc):
        pos = conf.GetPositions()  # hold the stack as embedded; the seed already realised the separation
        for ring_a, ring_b, _sep in cons.planes:
            for a, b in itertools.product(ring_a, ring_b):
                d = float(np.linalg.norm(pos[a] - pos[b]))
                ff.AddDistanceConstraint(a, b, d - _PLANE_PAD, d + _PLANE_PAD, _PLANE_FC_SCALE * fc)


class Haptic(Mechanism):
    """An eta>=3 face's transient centroid dummy: seat it inside its own ring.

    RDKit floors a bond-less atom at a ~3.4 Å vdW contact, forbidding the ~1.2 Å the ring radius needs. The
    dummy's own explicit windows (M->centroid, centroid->ring) are already in `pairs` and are the truth. The FF
    half is elsewhere (the dummy is retyped to an untypeable element before any term is added).
    """

    def dg_relief(self, cons, ctx):
        n = len(ctx.bm)
        for p in cons.phantoms:
            for x in range(n):
                a, b = (p, x) if p < x else (x, p)
                if x == p or (a, b) in ctx.pairs:  # itself, or an explicit centroid window -> leave it
                    continue
                ctx.bm[b][a] = min(ctx.bm[b][a], _PHANTOM_FLOOR)  # bm[b][a] is the lower bound; only ever lower


class Sp2Planar(Mechanism):
    """Hold each sp2 carbon at the improper its ETKDG seed already has: an FF-only preserve, never target-flat.

    UFF's inversion term is too weak to keep a conjugated thiourea/amidine sp2 carbon flat, so the relax
    puckers a seed that was planar (chb C3 0.003 -> 0.20 Å), tripping `geometry.planarity`. A soft improper
    window centred on the seed's own value carries that planarity into the relax: it forbids adding pucker
    beyond ±`_SP2_HOLD_WIN` but does not re-form a curve the seed lacks, so corannulene's real bowl, seeded
    flat, stays flat. Additive, supplying an FF term nothing else does.

    Organic only (`not cons.metals`). On a metal system the Li FF-surrogate has its bonds stripped and is not
    in `_METAL_Z`, so it defeats `geometry.planarity`'s metal/coordinated-carbon exemption:
    `_coordinating_carbons` perceives no shell, and the hold lands on coordinated carbons and fights the
    coordination (measured: tears 5 metal cases). On an organic system, excluding `cons.frozen` reproduces the
    gate's population exactly.
    """

    def ff_terms(self, ff, cons, conf, fc):
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
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], atom.GetIdx())
            ff.UFFAddTorsionConstraint(
                nbrs[0], nbrs[1], nbrs[2], atom.GetIdx(), False, phi - _SP2_HOLD_WIN, phi + _SP2_HOLD_WIN, _SP2_HOLD_FC
            )


class ConjugationCap(Mechanism):
    """Pull each conjugated C=X-N/O torsion flat: the organic twin of `Coplanar`, target-flat.

    UFF has no term holding a thiourea/amidine/enamine's conjugated plane, so the relax twists the C-X torsion
    out of the π system (a thiourea seed already at 89.8°), tripping `geometry.conjugation`. This is the same
    UFF lever as the metal `Coplanar` cap, a soft `UFFAddTorsionConstraint` toward the nearest in-plane well
    via `_coplanar_window`, and it reads the shared `conjugated_quartets` so enforcement and gate agree.

    Distinct from `Sp2Planar`, which holds the sp2 improper: this holds the C-X torsion, and targets flat
    rather than the seed, because the seed is often already twisted. Additive.

    Organic only (`not cons.metals`), and skips a quartet touching `cons.frozen`; same scoping as `Sp2Planar`.
    """

    def ff_terms(self, ff, cons, conf, fc):
        if cons.metals:  # see class docstring: the Li surrogate defeats gate-matching on a metal system
            return
        for a, c, x, s in conjugated_quartets(conf.GetOwningMol()):
            if cons.frozen.intersection((a, c, x, s)):  # frozen core held with zero DOF; never double-restrain it
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, a, c, x, s)
            lo, hi = _coplanar_window(phi, _CONJ_CAP)
            ff.UFFAddTorsionConstraint(a, c, x, s, False, lo, hi, _COPLANAR_FC)


class Umbrella(Mechanism):
    """Keep a requested pyramid's metal off its donor plane: a one-sided minimum pyramidalisation, FF-only.

    The angular twin of `Floor` rather than `Pull`: a bound to clear, not a target to spike to. The
    polyhedron's ±8° window is flat-bottomed and its wall falls on the planar side of `classify_geometry`'s
    ruler, so trigonal_pyramidal survived embed+relax in 13 of 49 conformers where the other 11 shapes
    survived 100%.

    Refuted: narrowing that window instead (the ruler is an absolute RMS while an angle is scale-free, so the
    wall would be bond-length dependent), and three D-M-D point targets (which pin the in-plane Y-distortion
    one weakly-coupled improper leaves free).

    The wall is the record's ideal (h = r/3), not perception's threshold, which would be teaching to the test;
    the price is that a shallow real pyramid seeds ~55% too deep. There is no DG half, and `optimize` hands
    xtb only `cons.frozen`, so a real energy stays free to flatten it. This biases the seed; it is not a claim
    the pyramid is right. The hand is not declared: `phi >= 0` reads the seed's own sign.

    Fires only on a `cons.spheres` recipe.
    """

    def ff_terms(self, ff, cons, conf, fc):
        frag = _frag_map(conf.GetOwningMol())  # same ligand = same fragment
        for recipe in cons.spheres:
            ideal = POLYHEDRA[resolve_geometry(recipe.geometry)].umbrella_improper
            if ideal is None:  # not a flat-based pyramid: nothing one improper can say
                continue
            base = [recipe.donors[k] for k in recipe.order]
            haptic = {d for d, _ring in recipe.haptic}
            if any(d == VACANT or d in haptic for d in base):  # an empty vertex or centroid face: no case
                continue
            if cons.frozen.intersection((*base, recipe.metal)):  # a fix= core already pins this geometry exactly
                continue
            if len({frag[d] for d in base}) == 1:  # a kappa3 of one ligand: its backbone owns the whole base,
                # so the improper could only be met by twisting it. Measured on mer-terpy/Cu(I): N-Cu-N pinched
                # 154.4 -> 126.1 deg, inter-ring torsions twisted 13 deg, the metal pushed 0.96 A off terpy's
                # own plane. A bipy+Cl base has two fragments, a real inter-ligand pair, and stays held.
                continue
            d0, d1, d2 = base[0], base[1], base[2]  # the same three vertices `umbrella_improper` measures
            phi = rdMolTransforms.GetDihedralDeg(conf, d0, d1, d2, recipe.metal)
            lo, hi = (ideal, _RIGHT_ANGLE) if phi >= 0 else (-_RIGHT_ANGLE, -ideal)  # keep the seed's own hand
            ff.UFFAddTorsionConstraint(d0, d1, d2, recipe.metal, False, lo, hi, _UMBRELLA_FC)


# The one place mechanism order is stated, satisfying both drivers at once:
#   DG WINDOW  distances -> angles -> planes   (Distance seeds `pairs`; Angle's `leg()` reads it)
#   DG RELIEVE phantom floors -> centroid floors;  DG POST  coplanar alone, after COMMIT
#   FF         frozen -> distances -> pulls -> floors -> angles -> coplanar -> planes -> sp2_planar
#              -> conjugation -> umbrella   (the last three have no DG hook; their FF order is immaterial)
MECHANISM_ORDER = (
    Frozen(),
    Distance(),
    Pull(),
    Floor(),
    Angle(),
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
    """One-sided torsion window ``cap`` deg wide, pulling a seed dihedral toward its nearest in-plane well.

    A conjugated donor is coplanar at 0 (syn) or ±180 (anti). The window opens on the seed's own side of the
    well, so riding the wall lands on the target, where a straddling window would ride out to ``cap`` instead.
    Bounds may fall outside [-180, 180]; RDKit's torsion penalty is periodic.
    """
    well = 0.0 if abs(phi) < _RIGHT_ANGLE else (_STRAIGHT if phi >= 0 else -_STRAIGHT)
    return (well, well + cap) if phi >= well else (well - cap, well)
