"""One class per constraint field, holding its distance-geometry writer and its force-field writer together.

**Why co-located.** Every field is written twice — as an edit to the ETKDG bounds matrix and as a restrained-UFF
term. Co-locating the two halves (they used to live modules apart) is what stops them drifting: a field gaining a
DG bound but not its matching FF term. `MECHANISM_ORDER` at the bottom is the one place their order is stated.

**Split by FIELD, not by source.** Splitting by source would reorder `cons.angles` iteration, and the bounds
matrix depends on that order — `leg()` reads the `pairs` dict the angle loop is still mutating, so an angle
processed earlier supplies a leg to one processed later.

**Phases** (the driver owns the order, ~6 lines of `embed/bounds.py`; a mechanism only says what it does):

    WINDOW   dg_windows   distances -> angles -> planes, all into `ctx.pairs` (not yet the matrix)
    RELIEVE  dg_relief    lower RDKit's phantom floors — BEFORE the windows are committed
    COMMIT   (driver)     write `ctx.pairs` into the matrix
    POST     dg_post      read the COMMITTED matrix (the coplanar 1,4 bound needs the legs)
    SMOOTH   (driver)     triangle smoothing

The FF side is a flat additive loop, so a mechanism there only implements `ff_terms`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from .metal import _DISCONNECTED  # one definition, in the metal constants hub (RDKit's inter-fragment topo distance)

_PHANTOM_FLOOR = 0.30  # Å: a haptic centroid dummy may sit this close to any atom — it lives INSIDE its own ring
_RIGHT_ANGLE = 90.0  # deg: an in-plane anchor at/above this is anti (metal far from the reference), below it syn
_COPLANAR_SCAN = 12  # samples across the stated M-D-X angle window when sizing a coplanarity bound
_PLANE_PAD = 0.3  # Å half-width on a pi-stack cross-ring distance
_FLOOR_HI = 1e3  # Å — a floor is one-sided: [floor, +inf). 1e3 stands in for infinity.
_SOFT_PULL_FC = 1e4  # soft harmonic onto the M-donor target, inside the flat-bottomed wall. Soft deliberately:
#   a full-strength point restraint fights the field and wrecks convergence. The wall forbids leaving the window;
#   this says where in it to sit.
_COPLANAR_FC = 10.0  # kcal/rad²: sp2-donor coplanarity torsion. Far softer than the distance walls because real
#   out-of-plane scatter runs to ~40° (donor_orient.CENSUS_OOP_P95) — remove the gross deviation, never pin flat.
#   Pinned by `test_reembed_retry_delivers_clean_geometry`: at 5 a diphosphine-on-rigid-diene chelate tears.
_PLANE_FC_SCALE = 0.2  # a stack hold is softer than a stated distance wall
_STRAIGHT = 180.0
_SP2_HOLD_FC = 10.0  # kcal/rad²: == _COPLANAR_FC. Kept at 10 (not lowered): review-batch.md §R4 swept 10→0.01 and
#   the two demands do NOT overlap — the chb thiourea stays clean down to fc 3, but a genuinely-bowled PAH
#   (corannulene) needs fc≤0.01 before its bowl re-forms, where the thiourea puckers back. The tension runs
#   through the WINDOW below, not the fc; corannulene is out of the organocatalysis domain.
_SP2_HOLD_WIN = 5.0  # deg half-window around the seed's OWN improper — hold-at-seed. Preserves a curve the seed
#   HAS, never re-forms one it LACKS (ETKDG seeds corannulene's bowl FLAT); widening it lets the thiourea over-pucker.
_CONJ_CAP = 20.0  # deg half-window of the organic conjugation torsion cap (`ConjugationCap`). Must sit INSIDE
#   `geometry.conjugation`'s 30° gate (at cap==30 the window rides out to the gate line and still flags — the
#   `angular-spring` failure); 20° lands flat with margin. Measured on bimp/takemoto/schreiner (ff-handling.md §4).


@dataclass
class DGContext:
    """Everything a DG writer needs beyond the Constraints: the molecule, the matrix, and the pending windows."""

    mol: Chem.Mol
    bm: np.ndarray
    pairs: dict = field(default_factory=dict)  # (i, j) -> (lo, hi), committed into `bm` after RELIEVE
    _topo: np.ndarray | None = None

    @property
    def topo(self):
        """Bond-path lengths on the (metal-stripped) graph — a real backbone gives a finite distance."""
        if self._topo is None:
            self._topo = Chem.GetDistanceMatrix(self.mol)
        return self._topo

    def mid(self, i, j):
        """Midpoint of the matrix's current window for a pair."""
        a, b = (i, j) if i < j else (j, i)
        return 0.5 * (self.bm[a][b] + self.bm[b][a])

    def leg(self, i, j):
        """Angle-leg length: prefer an explicit window's midpoint over the (maybe bond-less) matrix default.

        Reads `pairs`, which the angle writer is still filling — so an angle processed earlier supplies the leg
        for one processed later. That is why mechanisms are split by field: it keeps `cons.angles` one loop in
        its builder's insertion order.
        """
        v = self.pairs.get((min(i, j), max(i, j)))
        return 0.5 * (v[0] + v[1]) if v else self.mid(i, j)


class Mechanism:
    """A constraint field's two writers. Subclasses implement only the hooks their field needs."""

    def dg_windows(self, cons, ctx):
        """WINDOW: contribute candidate ``(lo, hi)`` windows to ``ctx.pairs``."""

    def dg_relief(self, cons, ctx):
        """RELIEVE: LOWER a matrix bound. Runs before COMMIT, so an explicit window always wins."""

    def dg_post(self, cons, ctx):
        """POST: read the COMMITTED matrix and tighten it."""

    def ff_terms(self, ff, cons, conf, fc):
        """FF: add restraint terms for this field. ``fc`` is the distance force constant."""


class Frozen(Mechanism):
    """Atoms pinned to their embedded coordinates — held with ZERO degrees of freedom, not a stiff spring.

    A TS reacting core must not be relaxed at all (a soft pin lets UFF crush a partial bond's sub-Å contacts).
    The DG half is the pairwise shape the builder writes into `distances` (`add_pairwise_shape`).
    """

    def ff_terms(self, ff, cons, conf, fc):
        for idx in cons.frozen:
            ff.AddFixedPoint(idx)


class Distance(Mechanism):
    """A distance window: the bounds-matrix cell, and a flat-bottomed wall that forbids leaving it.

    Seeds `ctx.pairs` FIRST so every later writer's `leg()` reads stated windows rather than matrix defaults.
    """

    def dg_windows(self, cons, ctx):
        ctx.pairs.update(cons.distances)  # override only the constrained pairs; RDKit keeps the rest

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), (lo, hi) in cons.distances.items():
            ff.AddDistanceConstraint(i, j, lo, hi, fc)


class Pull(Mechanism):
    """A soft harmonic at the M-donor target, inside the flat-bottomed wall.

    DG half: none. `AddDistanceConstraint(lo, hi, fc)` exerts zero force between lo and hi, so without a target
    the donor rides whichever wall it was last pushed to — this says where inside the window to sit. Suppressed
    for a rigid-body member: singling out a subset of its windows makes the relax tear the rest.
    """

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), target in cons.pulls.items():
            ff.AddDistanceConstraint(i, j, target, target, _SOFT_PULL_FC)


class Floor(Mechanism):
    """The anti-overbond guard: a one-sided minimum M...X, and the matching relief in the matrix.

    Two halves of ONE physical distance that must ship together. The FF `floors` raises a bound (zeroing the
    surrogate vdW removed the only thing keeping a non-donor off the metal); the DG `dg_floors` LOWERS one (the
    bond-less carbon makes RDKit floor every M...X at a ~3.4 Å carbon-vdW contact while real second-sphere atoms
    sit at 2.8-3.0 Å). A `floors` without its `dg_floors` twin leaves the matrix forbidding the FF's target.
    """

    def dg_relief(self, cons, ctx):
        # Never relieve onto an atom inside a rigid body: it has 6 dof not 3n, every internal distance is
        # already stated, and lowering a floor there only lets the DG fold the body inward.
        rigid = set().union(*cons.shapes) if cons.shapes else set()
        for (a, b), floor in cons.dg_floors.items():  # keys are (metal, X), sorted; bm[b][a] is the lower bound
            if (a, b) in ctx.pairs:  # an explicit window is the truth; never override it
                continue
            far = b if a in cons.metals else a  # the non-metal end of the pair
            if far in rigid:  # (a metal inside a rigid body still gets reliefs to atoms OUTSIDE it)
                continue
            if floor < ctx.bm[b][a] <= ctx.bm[a][b]:  # only ever relax, and only if genuinely too high
                ctx.bm[b][a] = floor

    def ff_terms(self, ff, cons, conf, fc):
        for (i, j), floor in cons.floors.items():
            ff.AddDistanceConstraint(i, j, floor, _FLOOR_HI, fc)


class Angle(Mechanism):
    """An angle window: a law-of-cosines 1-3 distance in the matrix, and a UFF angle wall.

    THE INTERSECT RULE, one predicate with no per-feature special case. An angle-derived distance is a PRIOR,
    not truth: where a real backbone bond-path connects the end atoms the matrix already holds their diagonal,
    so INTERSECT (the tighter real bound wins); where they meet only through the stripped metal (L-M-L, an
    M-D-X wall) the matrix holds only a phantom carbon-vdW floor, so the angle is written outright. This is what
    lets the chelate bite, polyhedron angles, coplanarity cap and haptic rings compose. A disjoint intersection
    keeps the BACKBONE and discards the angle, by design.
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
    """Keep the metal in a conjugated sp2 donor's plane — a 1,4 distance bound, and a soft UFF torsion.

    An ARTEFACT field, not physics: it exists only because the surrogate strips the M-donor bond and its UFF
    improper. Runs in POST (the 1,4 bound reads the COMMITTED legs and can only tighten).

    The 1,4 distance carries dihedral information only once M-D-X is pinned, so it is read from `cons.angles`
    and sized at the EXTREMUM over the whole window. A single fixed angle cannot do this: the dependence on
    M-D-X REVERSES between a proper dihedral (d rises with the angle) and an improper (d falls), so one guess
    forbids planar geometry on the other — the extremum is safe on both. A donor with no stated M-D-X angle
    gets no bound; the FF torsion (a real Cartesian dihedral) still holds it.
    """

    def dg_post(self, cons, ctx):
        for i, j, k, w, anchor, cap in cons.coplanar:
            key = (min(i, w), max(i, w))
            if key in cons.distances:  # a stated distance window is truth; an angle-derived prior only intersects
                continue
            window = cons.angles.get((i, j, k)) or cons.angles.get((k, j, i))
            if window is None:  # nothing pins M-D-X — say nothing
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
        from rdkit.Chem import rdMolTransforms

        if not cons.coplanar:  # organic / no metal: nothing to hold, and skip the hybridisation perception cost
            return
        # lazy: it owns the perception ruler these read
        from rxembed.rdkit_embed.constraints import donor_orient as _donor

        # A conjugated bidentate's plane is already pinned by the polyhedron's two M-D distances + L-M-L bite
        # (two coplanar contacts pin a plane). Per-donor FF torsions ADD rather than intersect, so redundant ones
        # fight and twist a crowded chelate's backbone — skip the FF torsion for any donor a co-donor shares the
        # plane with. FF-ONLY: the DG bound (`dg_post`) composes via the matrix INTERSECT and stays for every donor.
        mol = conf.GetOwningMol()
        hyb = _donor._stripped_hybridisation(mol)
        donors = {  # every donor of the metal, from the M-donor distance windows (a co-donor need not be capped)
            b if a in cons.metals else a for a, b in cons.distances if a in cons.metals or b in cons.metals
        } or {e[1] for e in cons.coplanar}
        for i, j, k, w, _anchor, cap in cons.coplanar:
            if _donor.codonor_in_plane(mol, j, donors, hyb):
                continue  # redundant restatement of the bite-pinned plane — see above
            phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
            lo, hi = _coplanar_window(phi, cap)
            ff.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC)


class Plane(Mechanism):
    """A parallel pi-stack: cross-ring distances from the stated separation, and pairwise holds in the FF.

    Uses `setdefault`, so a user `constrain=` on a cross-ring pair wins — and so does an angle-derived window
    written earlier in the same phase, since the stack is a heuristic and the angle is not.
    """

    def dg_windows(self, cons, ctx):
        for ring_a, ring_b, sep in cons.planes:  # cross-ring d = sqrt(sep^2 + in-plane^2)
            for u, au in enumerate(ring_a):
                for v, bv in enumerate(ring_b):
                    offset = 0.0 if u == v else ctx.mid(au, ring_a[v])
                    d = math.hypot(sep, offset)
                    ctx.pairs.setdefault((min(au, bv), max(au, bv)), (d - _PLANE_PAD, d + _PLANE_PAD))

    def ff_terms(self, ff, cons, conf, fc):
        pos = conf.GetPositions()  # hold the stack AS EMBEDDED — the seed already realised the separation
        for ring_a, ring_b, _sep in cons.planes:
            for a, b in itertools.product(ring_a, ring_b):
                d = float(np.linalg.norm(pos[a] - pos[b]))
                ff.AddDistanceConstraint(a, b, d - _PLANE_PAD, d + _PLANE_PAD, _PLANE_FC_SCALE * fc)


class Haptic(Mechanism):
    """An eta>=3 face's transient centroid dummy: seat it INSIDE its own ring.

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
    """Hold each sp2 carbon at the improper its ETKDG seed already has — an FF-only preserve, never target-flat.

    UFF's inversion term is too weak to keep a conjugated thiourea/amidine sp2 carbon flat, so the relax puckers
    a seed that WAS planar (chb C3 0.003 → 0.20 Å), tripping `geometry.planarity`. A soft improper window centred
    on the SEED's own value carries that planarity into the relax: it forbids ADDING pucker beyond ±`_SP2_HOLD_WIN`
    but does not re-form a curve the seed lacks (corannulene's real bowl, seeded flat, stays ~flat; ff-refuted in
    review-batch.md §R4, out of domain). Additive — an FF term nothing else supplies.

    **Organic only** (`not cons.metals`): on a metal system the Li FF-surrogate (bonds stripped, not in
    `_METAL_Z`) defeats `geometry.planarity`'s metal/coordinated-carbon exemption, so `_coordinating_carbons`
    perceives no shell and the hold lands on coordinated carbons and fights the coordination (measured: tears 5
    metal cases). On an organic system, excluding `cons.frozen` reproduces the gate's population exactly.
    """

    def ff_terms(self, ff, cons, conf, fc):
        from rdkit.Chem import rdMolTransforms

        # lazy: the kernel perception the planarity gate also reads
        from rxembed.rdkit_embed.coordination import _SP2_DEGREE
        from rxembed.rdkit_embed.vecmath import _CARBON_Z

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
    """Pull each conjugated C=X-N/O torsion flat — the organic twin of `Coplanar`, target-flat not hold-at-seed.

    UFF has no term holding a thiourea/amidine/enamine's conjugated plane, so the relax twists the C-X torsion
    out of the π system (a thiourea seed already at 89.8°), tripping `geometry.conjugation`. The SAME UFF lever
    as the metal `Coplanar` cap (soft `UFFAddTorsionConstraint` toward the nearest in-plane well via
    `_coplanar_window`), reading the shared `coordination.conjugated_quartets`, so enforcement and gate agree.

    Distinct from `Sp2Planar` (which holds the sp2 IMPROPER); this holds the C-X TORSION, and is TARGET-FLAT
    (the seed is often already twisted, so preserving it cannot help). Additive.

    **Organic only** (`not cons.metals`) and skip a quartet touching `cons.frozen` — same scoping/reason as
    `Sp2Planar`.
    """

    def ff_terms(self, ff, cons, conf, fc):
        from rdkit.Chem import rdMolTransforms

        # lazy: the kernel perception the gate also reads
        from rxembed.rdkit_embed.coordination import conjugated_quartets

        if cons.metals:  # see class docstring: the Li surrogate defeats gate-matching on a metal system
            return
        for a, c, x, s in conjugated_quartets(conf.GetOwningMol()):
            if cons.frozen.intersection((a, c, x, s)):  # frozen core held with zero DOF; never double-restrain it
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, a, c, x, s)
            lo, hi = _coplanar_window(phi, _CONJ_CAP)
            ff.UFFAddTorsionConstraint(a, c, x, s, False, lo, hi, _COPLANAR_FC)


# The one place mechanism order is stated. It satisfies BOTH drivers at once:
#   DG WINDOW  distances -> angles -> planes   (Distance seeds `pairs`; Angle's `leg()` reads it; Plane yields)
#   DG RELIEVE phantom floors -> centroid floors
#   DG POST    coplanar, alone, after COMMIT
#   FF         frozen -> distances -> pulls -> floors -> angles -> coplanar -> planes -> sp2_planar -> conjugation
# Sp2Planar and ConjugationCap are FF-only (no DG hook), so they sit out of the DG phases; their order among FF
# terms is immaterial (every ff_terms reads the same seed conf). Orthogonal caps: the sp2 improper + the C-X torsion.
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
)


def _law_of_cosines(dij, djk, theta_deg):
    return math.sqrt(dij**2 + djk**2 - 2 * dij * djk * math.cos(math.radians(theta_deg)))


def _chain_distance(d_ij, d_jk, d_kl, th_ijk, th_jkl, phi):
    """1,4 distance d(i,l) of a chain i-j-k-l from its two flanking bonds, middle, two angles, dihedral (deg).

    Purely geometric — the "k-l bond" need not be a real bond (for the 2-substituent improper M-D-X-Y it is the
    X...Y 1,3 distance), which is why one formula serves both the proper dihedral and the improper.
    """
    ti, tj, ph = math.radians(th_ijk), math.radians(th_jkl), math.radians(phi)
    p_i = np.array([d_ij * math.cos(ti), d_ij * math.sin(ti), 0.0])
    p_w = np.array([d_jk - d_kl * math.cos(tj), d_kl * math.sin(tj) * math.cos(ph), d_kl * math.sin(tj) * math.sin(ph)])
    return float(np.linalg.norm(p_i - p_w))


def _coplanar_window(phi, cap):
    """One-sided torsion window pulling a seed dihedral toward its nearest in-plane well, ``cap`` deg wide.

    A conjugated donor is coplanar at 0 (syn) or ±180 (anti). Open the window on the SEED's own side of the
    nearest well: a straddling window would let the metal ride the wall out at ``cap``, a point target would
    kill the spread. Bounds may fall outside [-180, 180] — RDKit's torsion penalty is periodic (intended).
    """
    well = 0.0 if abs(phi) < _RIGHT_ANGLE else (_STRAIGHT if phi >= 0 else -_STRAIGHT)
    return (well, well + cap) if phi >= well else (well - cap, well)
