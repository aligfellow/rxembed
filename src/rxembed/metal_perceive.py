"""Metal coordination-sphere gates: has an atom collapsed onto the metal, and does a ligand still point right.

Read-only: these gates judge a finished geometry without changing it. `pipeline.geom_check` consumes them
directly; `embed._donor_facing_failure` also reads `donor_orientation` inside the embed acceptance path, but
only logs it at DEBUG. They look into the coordination sphere, which every `geometry` check excludes because a
dative distance is not covalent: `metal_overbond` asks whether an atom reached bonding distance,
`donor_orientation` whether a ligand still donates along its axis, and `donor_fold` reports the metric behind
the same walk. The rest is the sphere perception both resolve on. `pipeline.geom_check` also reads `spheres`
for its side-on, coordinated-carbon and cis-donor exemptions, and `pipeline.select` reads `coordinating_atoms`
for a wrapped metal with no bonds.

The shape reading (`classify_geometry`, `shape_gap`) names the polyhedron a sphere's coordinates fit best; the
acceptance gate, enumeration and CX writing all read a sphere through it.

Enforcement (``orient_donor`` / ``coplanar_donor``) lives in ``metal_donor_orient``; both paths share its
donation-axis rules. Shared coordinate math and ``Violation`` live in ``utils``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from .constraints import graft_owns
from .metal_core import COORDINATION_METALS, EPS_LEN, VACANT, ligand_graph
from .metal_distance import (
    APEX,
    DONOR_COLLAPSE_RATIO,
    NEAR,
    NEAR_REPORT_RATIO,
    OUTER_REPORT_MARGIN,
    overbond_tier,
)
from .metal_donor_orient import (
    FOLD_MEDIAN,
    FOLD_WINDOW,
    donation_axis,
    stripped_hybridisation,
)
from .metal_polyhedron import POLYHEDRA, best_fit_residual, describe, fit_residual, geometries_for_cn, record
from .utils import Violation, as_positions, bond_angle, dihedral_angle

logger = logging.getLogger("rxembed.metal")  # the name metal_core's logger explains
_PT = Chem.GetPeriodicTable()

_COORD_FACTOR = 1.3  # a heavy atom within this x covalent-sum of a metal is a coordinating donor


def metal_overbond(mol, pos, donors=None) -> list[Violation]:
    """Return atoms that have collapsed onto a metal; over-short is the direction other gates miss.

    Every other gate excludes metals, so a crushed atom is otherwise invisible. A donor is floored only
    against `DONOR_COLLAPSE_RATIO` (reported as `metal_collapse`), the one ruler that still works for a
    monatomic hydride or halide; a non-donor is floored by `overbond_tier`'s tiers, below the FF's own floor
    so a floored relax never trips this. An undeclared H is skipped.

    `donors` is the intended set, and the caller must pass it: perceiving it here is circular, since a
    collapsed atom would enter the radius, re-classify as a donor, and get asked the donor's looser question
    instead of its own. Pass ``{metal: donors}`` to declare only some metals (see `spheres`); `donors=None`
    falls back to perception only for a genuinely unknown sphere.
    """
    out: list[Violation] = []
    for m, sphere in spheres(mol, pos, donors).items():
        rm = _PT.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i == m or a.GetAtomicNum() in COORDINATION_METALS:
                continue
            r_sum = rm + _PT.GetRcovalent(a.GetAtomicNum())
            if i in sphere:  # a donor gets a floor too (see docstring): its licence is a bonding window, not a
                # half-line to zero, so only collapse is left to judge: the one ruler for a monatomic donor
                floor, what = DONOR_COLLAPSE_RATIO * r_sum, "donor"
            elif a.GetAtomicNum() == 1:
                continue  # an undeclared H: a beta-agostic contact reaches this distance and is not an over-bond
            else:
                tier = overbond_tier(mol, sphere, i)
                if tier == APEX:  # a bite apex or haptic backbone is fixed by its donors' own windows
                    continue
                floor, what = (NEAR_REPORT_RATIO * r_sum if tier == NEAR else r_sum + OUTER_REPORT_MARGIN), "non-donor"
            d = float(np.linalg.norm(pos[i] - pos[m]))
            if d < floor:
                out.append(
                    Violation(
                        kind="metal_collapse" if what == "donor" else "metal_overbond",
                        atoms=(m, i),
                        value=d,
                        limit=floor,
                        detail=f"{what} {a.GetSymbol()}{i} is {d:.2f} A from the metal (floor {floor:.2f} A): it has "
                        + ("collapsed into it" if what == "donor" else "effectively bonded to it"),
                    )
                )
    return out


@dataclass(frozen=True)
class DonorAngle:
    """One judged M-D-X donation angle: the donor's class, the angle, and how far it folds."""

    metal: int
    donor: int
    sub: int  # X, a heavy substituent of the donor
    element: str = field(kw_only=True)  # the donor's element; thiolate at 103°, carboxylate at
    hyb: Chem.HybridizationType = field(kw_only=True)  # 126°: pooling by hyb alone voids the gate
    angle: float = field(kw_only=True)
    deviation: float = field(kw_only=True)  # |angle - class median|: the reference-free `fold`
    planarity: float | None = field(default=None, kw_only=True)  # metal out-of-plane in ligand plane (None
    # if not conjugated); report-only: crystals reach 87.6° and it can never gate

    @property
    def cls(self) -> tuple[str, Chem.HybridizationType]:
        """Return the census class this donation is judged against: ``(donor element, hybridisation)``."""
        return (self.element, self.hyb)


@dataclass
class FoldReport:
    """Result of ``donor_fold()``: the report-only donation metric, which never gates."""

    angles: list[DonorAngle] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)  # donors the ruler refused to judge: the two estimators
    # disagreed on the hybridisation, or the (element, hyb) class has n < 6 in the census (`FOLD_WINDOW`). A
    # rising count is a free perception-bug detector.

    @property
    def fold(self) -> float:
        """Return the max |M-D-X - class median| over every judged donation (0.0 when nothing was judged).

        Needs no reference structure: it asks the geometry about itself, so unlike an RMSD-to-crystal metric
        it works in generation mode.
        """
        return max((a.deviation for a in self.angles), default=0.0)

    @property
    def planarity(self) -> float:
        """Return the max deg the metal lies out of a conjugated donor's ligand plane (0.0 if none); report only."""
        return max((a.planarity for a in self.angles if a.planarity is not None), default=0.0)

    @property
    def outside_window(self) -> list[DonorAngle]:
        """Donations outside the census 99% window, including the overshoot the gate declines to judge.

        The gate fires on the fold floor only, a ceiling being a false positive on a healthy phosphine, so a
        real over-splay would otherwise be invisible rather than merely unjudged.
        """
        return [a for a in self.angles if not (FOLD_WINDOW[a.cls][0] <= a.angle <= FOLD_WINDOW[a.cls][1])]

    def summary(self) -> str:
        """Format the fold / planarity maxima and the unjudged donors."""
        return (
            f"fold {self.fold:.1f}° (max |M-D-X - class median|, n={len(self.angles)}), "
            f"planarity {self.planarity:.1f}° out of plane (report-only), "
            f"{len(self.outside_window)} outside the census window, {len(self.unknown)} donor(s) unclassified"
        )


def donor_fold(mol, conf_id: int = -1, *, donors=None, frozen=frozenset()) -> FoldReport:
    """Measure how far every ligand folds off its donation axis. Reports; never rejects.

    The same walk `donor_orientation` gates on, reported in full, so a merely unusual geometry stays visible
    where the gate would pass it. Read ``.fold``, ``.planarity``, ``.unknown``.
    """
    angles, unknown = _donor_walk(mol, as_positions(mol, conf_id), donors, frozen)
    return FoldReport(angles, unknown)


def donor_orientation(mol, pos, donors=None, frozen=frozenset()) -> list[Violation]:
    """Report donor angles below their class-specific empirical floors.

    `metal_overbond` is blind to the commonest fold: a short D-X bond can swing fully side-on without ever
    reaching the over-bond floor (a carbonyl bent to a right angle holds its O at 2.13 A over a 2.04 A floor).
    This reads the M-D-X angle instead, which folding collapses. The floor per class is the census 0.5th
    percentile minus 5 deg, keyed on (element, hybridisation) since a thiolate donates at 103 deg where a
    carboxylate donates at 126 deg. A floor violation flags an unusual angle, not proven folding or inversion:
    a tetrahedral donor can sit inside its carrier hull below this floor, and real crystals do (a Zn-bound
    C(SiMe3)3 donor reads 101 deg against its 104 deg floor). `embed._donor_facing_failure` logs it at DEBUG
    and never rejects. `donor_fold` reports the rest.

    Exemptions follow `donation_axis` (no axis to judge), plus one more here: an M-D-X unit wholly inside the
    frozen core takes its orientation from the reference TS, not this gate. A donor the two estimators
    disagree on, or an uncalibrated class (n < 6), is never gated.

    `donors` is the intended sphere and the caller must pass it: perceiving it is circular, since a folded
    atom would enter the radius, read as a co-donor, and write its own donor off as haptic.
    """
    out: list[Violation] = []
    for a in _donor_walk(mol, pos, donors, frozen)[0]:
        lo = FOLD_WINDOW[a.cls][0]
        if a.angle >= lo:  # floor only: gating the ceiling false-positives on a healthy phosphine at 153.9°
            continue  # (see `FOLD_WINDOW`); the overshoot is reported by `FoldReport.outside_window`
        xsym = mol.GetAtomWithIdx(a.sub).GetSymbol()
        out.append(
            Violation(
                kind="donor_orientation",
                atoms=(a.metal, a.donor, a.sub),
                value=a.angle,
                limit=lo,
                detail=f"{a.element}{a.donor} ({str(a.hyb).lower()}) M-D-X to {xsym}{a.sub}: "
                f"{a.angle:.1f}° < census floor {lo:.0f}°; inspect donor geometry and restraints",
            )
        )
    return out


def _donor_walk(mol, pos, donors=None, frozen=frozenset()) -> tuple[list[DonorAngle], list[int]]:
    """Walk M -> D -> X: one judged angle per heavy donor substituent, plus the donors it refused to judge.

    The single source of truth for both the gate (``donor_orientation``) and the metric (``donor_fold``), so the
    two can never disagree about what was measured. See ``donor_orientation`` for the exemptions.
    """
    by_metal = spheres(mol, pos, donors)
    bound = Counter(d for s in by_metal.values() for d in s)  # donor -> metal count; its keys are the co-donors
    hyb = stripped_hybridisation(mol, bound)
    stripped = ligand_graph(mol)
    out: list[DonorAngle] = []
    unknown: list[int] = []
    for m, sphere in by_metal.items():
        for d in sorted(sphere):
            subs = donation_axis(mol, d, bound, sphere=sphere, hyb=hyb, metals=bound[d], stripped=stripped)
            if subs is None:  # hydride, bridging or haptic: the donation question does not apply
                continue
            cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None  # None when estimators disagree
            if cls not in FOLD_WINDOW:  # estimators disagreed, or n < 6 for this (element, hyb) class (an sp3
                if subs:  # ether O, an sp2 P): a threshold read off < 6 donations is noise, so abstain. Record
                    unknown.append(d)  # only if it had a judgeable substituent: an honest unknown, not a bare
                continue  # skip.
            for x in subs:
                if graft_owns((m, d, x), frozen):
                    continue
                ang = bond_angle(pos[m], pos[d], pos[x])
                dev = abs(ang - FOLD_MEDIAN[cls])
                out.append(
                    DonorAngle(
                        m,
                        d,
                        x,
                        element=cls[0],
                        hyb=cls[1],
                        angle=ang,
                        deviation=dev,
                        planarity=_planarity_dev(mol, pos, hyb=hyb, m=m, d=d, x=x),
                    )
                )
    return out, unknown


def _planarity_dev(mol, pos, *, hyb, m, d, x) -> float | None:
    """Return the deg the metal lies out of a conjugated donor's ligand plane, or None if D-X is not conjugated.

    A carboxylate or pyridine plane rotated into the coordination sphere with an acceptable M-D-X angle.
    Report only: it can never gate. The census p95 is 54.8 deg and real crystals reach 87.6 deg out of plane.
    """
    bond = mol.GetBondBetweenAtoms(d, x)
    sp2 = Chem.HybridizationType.SP2
    if bond is None or not bond.GetIsConjugated() or hyb.get(d) != sp2 or hyb.get(x) != sp2:
        return None
    best = None
    for y in mol.GetAtomWithIdx(x).GetNeighbors():
        j = y.GetIdx()
        if j in (d, m) or y.GetAtomicNum() == 1 or y.GetAtomicNum() in COORDINATION_METALS:
            continue
        t = abs(dihedral_angle(pos[m], pos[d], pos[x], pos[j]))
        dev = min(t, abs(180.0 - t))  # 0 = metal lies in the ligand plane
        best = dev if best is None else min(best, dev)
    return best


def coordinating_atoms(mol, pos, m, declared=frozenset()) -> set[int]:
    """Return non-metal atoms inside metal ``m``'s coordination shell (``_COORD_FACTOR`` x covalent-sum).

    A heavy atom at coordination distance counts as a donor. A hydrogen does not, since a hydride and an
    agostic C-H sit at the same M-H distance, so an H counts only when in ``declared``: the sole way an H
    enters a sphere; the element screen never overrides a declaration.
    """
    rm = _PT.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
    out = set()
    for a in mol.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        if i == m or z in COORDINATION_METALS or (z == 1 and i not in declared):
            continue
        if float(np.linalg.norm(pos[m] - pos[i])) <= _COORD_FACTOR * (rm + _PT.GetRcovalent(z)):
            out.add(i)
    return out


def declared_donors(donors, m) -> set[int]:
    """Return the donors declared for metal `m`: from a ``{metal: donors}`` mapping, or one flat collection."""
    if isinstance(donors, Mapping):
        return {int(d) for d in donors.get(m, ())}
    return set() if donors is None else {int(d) for d in donors}


def spheres(mol, pos, donors=None) -> dict[int, set[int]]:
    """Return ``{metal: its coordination sphere}``: declared donors where declared, perceived where not.

    `donors` is ``{metal: donor atoms}``, or one flat collection declared for every metal. Resolved per metal,
    because in a bimetallic complex the caller may know only the reacting centre's sphere; an undeclared metal
    falls back to perception rather than flag its own ligands. A flat collection cannot say which metal a
    bridging donor was declared for, so a partial declaration needs the mapping. A declared donor that has left
    the shell is dropped; `coordination_changed` is what gates that.
    """
    out: dict[int, set[int]] = {}
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS):
        known = declared_donors(donors, m)
        shell = coordinating_atoms(mol, pos, m, known)  # `known` also admits a declared hydride the element screen
        out[m] = (known & shell) or shell  # would drop; else `known & shell` silently loses it
    return out


APICAL_MIN = 3  # a face of this many atoms caps a face (a piano stool); an eta2 alkene fills one ordinary
#   in-plane site. Site occupancy, not geometry: it steers the default-polyhedron guess, never the embed.


# FIT_FLOOR is where no shape fits: above this residual, `classify_geometry` still returns the argmin, but
# it is a name, not a reading. The acceptance gate compares a requested shape's own residual with this
# floor, not with the argmin.
FIT_FLOOR = 0.45  # Procrustes residual; 45 corpus centres read median 0.069, 90th percentile 0.30

# _FIT_MARGIN is the resolution of a shape reading: two shapes within it of each other tie, for the input,
# for the acceptance gate (`shape_reading`, rule B), and for `observed_only`. A near-tie input then reads
# the same way at every seed instead of a seed lottery (measured on VALRAE and TILFAW).
_FIT_MARGIN = 0.01

# Å RMS out-of-plane of {metal + vertices} above which a sphere is not planar: the accept gate on a
# declared-`planar` record and its non-planar counterpart, `coplanar`. An absolute RMS, so its tolerance is
# proportionally looser at a short M-L bond and tighter at a long one.
COPLANAR_TOL = 0.25


def _plane_rms(metal_pos, verts):
    """Return the RMS distance of {metal + coordination vertices} from their best-fit plane (Å).

    Fewer than 4 points define a plane exactly, so they score 0: trivially coplanar.
    """
    pts = np.array([metal_pos, *verts])
    if len(pts) < 4:  # noqa: PLR2004  a plane needs >=3 points; <4 total is trivially coplanar
        return 0.0
    dev = (pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0))[2][2]  # signed distance from best-fit plane
    return float(np.sqrt(np.mean(dev**2)))


def _read_sphere(mol, metal, sites, cid=-1):
    """Return `(obs, rms, r)`: unit rays, coplanarity RMS and mean bond length for `sites` at `metal`.

    ``sites`` are coordination sites, not atoms: a haptic face is one vertex via its centroid. Returns
    ``None`` when there are no sites, or a site sits on the metal with no defined direction.
    """
    pos = mol.GetConformer(cid).GetPositions()
    points = []
    for site in sites:
        atoms = [site] if isinstance(site, (int, np.integer)) else list(site)
        points.append(np.mean([pos[a] for a in atoms], axis=0))
    if not points:
        return None
    obs = np.array([p - pos[metal] for p in points], float)
    if not np.all(np.linalg.norm(obs, axis=1) > EPS_LEN):
        return None
    obs = obs / np.linalg.norm(obs, axis=1, keepdims=True)
    rms = _plane_rms(pos[metal], points)
    r = float(np.mean([np.linalg.norm(p - pos[metal]) for p in points]))
    return obs, rms, r


def rank_shapes(obs, *, vacancy=None):
    """Return every same-CN polyhedron's fit residual to `obs`, best first: ``[(residual, name), ...]``.

    `vacancy` is ``(name, ideal_dirs)``: a requested polyhedron's own occupied-vertex-subset directions,
    ranked under `name` alongside the genuine records at this coordination number, needed only when a vertex
    is vacant and so `name` is absent from `POLYHEDRA` at this CN. Ties break by `POLYHEDRA` insertion order
    (a stable sort).
    """
    same_cn = [(n, p) for n, p in POLYHEDRA.items() if p.cn == len(obs)]
    ranked = [(fit_residual(obs, p), name) for name, p in same_cn]
    if vacancy is not None:
        name, ideal = vacancy
        ideal = np.asarray(ideal, float)
        ideal = ideal / np.linalg.norm(ideal, axis=1, keepdims=True)
        ranked.append((best_fit_residual(obs, ideal), name))
    return sorted(ranked, key=lambda t: t[0])


def shape_reading(ranked, requested):
    """Return ``(requested residual, next name, next residual, accepted)`` for `requested` in a ranking.

    `ranked` is `rank_shapes`' sorted output. `next` is the best-reading polyhedron other than `requested`
    (``None`` when there is none to compare against). `accepted` is rule B, the tie predicate the acceptance
    gate, `observed_only` and the input reading all share: `requested` is accepted when it reads within
    `_FIT_MARGIN` of the best reading, `ranked[0]`. Returns ``(None, None, None, False)`` if `requested` has
    no entry in `ranked` at all (a different coordination number).
    """
    requested_err = next((err for err, name in ranked if name == requested), None)
    if requested_err is None:
        return None, None, None, False
    best_err, best_name = ranked[0]
    if best_name == requested:
        next_err, next_name = ranked[1] if len(ranked) > 1 else (None, None)
    else:
        next_err, next_name = best_err, best_name
    accepted = next_err is None or requested_err - best_err < _FIT_MARGIN
    return requested_err, next_name, next_err, accepted


SHAPE_PROP = "shape"  # RDKit conformer property an accepted conformer carries: see `shape_clause`
# Machine-readable {metal atom: requested geometry} beside SHAPE_PROP, one accepted conformer at a time.
# metal_smiles.cxsmiles reads this to write CX in the requested frame instead of re-perceiving by argmin;
# a conformer without it (a raw geometry, never through the acceptance gate) keeps the argmin reading.
SHAPE_REQUEST_PROP = "shape_request"


def encode_shape_request(requests):
    """Return `requests` (``{metal atom: requested geometry}``) as one `SHAPE_REQUEST_PROP` string."""
    return ";".join(f"{atom}:{geometry}" for atom, geometry in sorted(requests.items()))


def decode_shape_request(value):
    """Return the `{metal atom: requested geometry}` a `SHAPE_REQUEST_PROP` string encodes."""
    pairs = (pair.split(":", 1) for pair in value.split(";") if pair)
    return {int(atom): geometry for atom, geometry in pairs}


def shape_clause(geometry, residual, next_name, next_err):
    """Format one shape reading as its code, residual and runner-up: ``'TBP 0.053 (next SPY 0.198)'``."""
    clause = f"{record(geometry).code} {residual:.3f}"
    if next_name is not None:
        clause += f" (next {record(next_name).code} {next_err:.3f})"
    return clause


def shape_gap(mol, metal, vertices, haptic, requested, cid=-1):
    """Return `shape_reading` for `requested` at `metal`, reading `mol`'s conformer `cid`.

    `vertices`/`haptic` are one centre's per-slot donor map, as `materialized_state` returns them; a vacant
    slot (`VACANT`) is read against the occupied coordination number, ranked alongside `requested`'s own
    occupied-vertex subset. Returns ``(None, None, None, True)``, a neutral read, when there is no conformer
    or the sphere cannot be read at all (see `_read_sphere`).
    """
    if mol.GetNumConformers() == 0:
        return None, None, None, True
    occupied = [i for i, v in enumerate(vertices) if v != VACANT]
    sites = [haptic.get(vertices[i], vertices[i]) for i in occupied]
    sphere = _read_sphere(mol, metal, sites, cid)
    if sphere is None:
        return None, None, None, True
    obs, _rms, _r = sphere
    vacancy = None
    if len(occupied) < len(vertices):
        poly = POLYHEDRA.get(requested)
        if poly is None:
            return None, None, None, True
        vacancy = (requested, [poly.vertex_dirs[i] for i in occupied])
    return shape_reading(rank_shapes(obs, vacancy=vacancy), requested)


def classify_geometry(mol, metal, sites, cid=-1, *, warn=True):
    """Name the coordination polytope by best orthogonal vertex fit.

    Returns ``None`` only when no record has that vertex count, otherwise the argmin: the continuous
    residual from `rank_shapes` decides, with no flatness pre-filter run first. Warns above `FIT_FLOOR` (no
    record really fits) and on a near-tie within `_FIT_MARGIN` of the runner-up; either way the returned
    shape stays the argmin, only the log line changes. Set ``warn=False`` for repeated internal validation.

    The fit preserves vertex correspondence while allowing rotation, reflection and donor reordering.
    `metal_polyhedron.fit_residual` owns the bounded-exact seating search and its high-CN approximation.
    """
    sphere = _read_sphere(mol, metal, sites, cid)
    if sphere is None:
        return None
    obs, rms, r = sphere
    ranked = rank_shapes(obs)
    if not ranked:
        return None
    best_err, best = ranked[0]
    poor = best_err > FIT_FLOOR  # no record fits
    runner_err, runner_name = ranked[1] if len(ranked) > 1 else (None, None)
    near_tie = runner_err is not None and runner_err - best_err < _FIT_MARGIN
    runner = f"; next {describe(runner_name)} {runner_err:.3f}" if runner_name is not None else ""
    if poor and warn:
        logger.warning(
            "geometry: no shape fits; nearest %s (residual %.3f > %.2f); pass geometry= to state it",
            describe(best),
            best_err,
            FIT_FLOOR,
        )
    elif near_tie and warn:
        best_code, runner_code = record(best).code, record(runner_name).code
        logger.warning(
            "geometry: metal %d is a near-tie, %s %.3f vs %s %.3f; using %s, pass geometry='%s' for the other",
            metal,
            best_code,
            best_err,
            runner_code,
            runner_err,
            best_code,
            runner_code,
        )
    else:
        logger.debug("geometry: %s residual %.3f%s", describe(best), best_err, runner)
    logger.debug(
        "sphere: %s; coplanarity RMS %.3f A vs %.2f tol; M-L %.2f A",
        "in-plane" if rms <= COPLANAR_TOL else "out-of-plane",
        rms,
        COPLANAR_TOL,
        r,
    )
    return best


def geometry_for(n_donors, has_apical=False):
    """Default coordination polyhedron name for `n_donors`, or None.

    An apical (eta>=3) face is an axial cone, so a CN4 carrying one is a piano stool, never the flat
    `square_planar` the vertex count would pick, which would seat a ligand trans through the ring. Only CN4
    flips. An eta2 face is not apical: it is an ordinary single-site vertex and keeps the default.
    """
    gs = geometries_for_cn(n_donors)  # best-default first
    g = gs[0].name if gs else None
    if has_apical and g == "square_planar":
        return "tetrahedral"
    return g


def coplanar(pos, metal, donors, haptic=None):
    """Return True if the metal and its coordination vertices lie in one plane.

    The feasibility test for a declared planar record: a bite squeezing the in-plane angles is still planar,
    but an arrangement that can only satisfy its ligands by twisting out of plane (an impossible trans-chelate)
    is not. An eta>=3 face is one vertex, so `haptic` collapses each face to its centroid first; that
    bookkeeping is all this adds over `_plane_rms`.
    """
    ring_atoms = {a for ring in (haptic or {}).values() for a in ring}
    verts = [pos[d] for d in donors if d not in ring_atoms]  # each sigma/eta2 donor is its own vertex
    verts += [np.mean([pos[a] for a in ring], axis=0) for ring in (haptic or {}).values()]  # each face -> centroid
    return _plane_rms(pos[metal], verts) <= COPLANAR_TOL
