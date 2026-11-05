"""Read a metal's coordination sphere: the shape it takes, and whether an atom collapsed or a donor folded.

`classify_geometry` names the polyhedron a sphere fits best, and `shape_gap` judges whether a centre takes its requested
one; the gates, readers, seeding and enumeration all use it. `metal_overbond` and `donor_orientation` look inside the
sphere every covalent check excludes. Nothing here changes a geometry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rdkit import Chem

from rxembed.geometry.constraints import graft_owns
from rxembed.graph import COORDINATION_METALS, ligand_graph
from rxembed.metal.centres import EPS_LEN
from rxembed.metal.distances import (
    APEX,
    DONOR_COLLAPSE_RATIO,
    NEAR,
    NEAR_REPORT_RATIO,
    OUTER_REPORT_MARGIN,
    overbond_tier,
)
from rxembed.metal.orientation import (
    FOLD_WINDOW,
    donation_axis,
    stripped_hybridisation,
)
from rxembed.metal.polyhedra import POLYHEDRA, VACANT, fit_residual, geometries_for_cn, record
from rxembed.utils import Violation, bond_angle

_PT = Chem.GetPeriodicTable()

_COORD_FACTOR = 1.3  # a heavy atom within this x covalent-sum of a metal is a coordinating donor


def metal_overbond(mol, pos, donors) -> list[Violation]:
    """Return atoms that have collapsed onto a metal, which every other check misses by excluding metals.

    A donor is floored only at `DONOR_COLLAPSE_RATIO`, the one ruler that works for a lone hydride or halide; a
    non-donor at `overbond_tier`'s tiers, below the FF's floor so a floored relax never trips it. An undeclared H is
    skipped.

    The caller must pass the intended `donors`: perceiving them would read a collapsed atom as a donor and ask it the
    donor's looser question. ``{metal: donors}`` declares only some metals.
    """
    out: list[Violation] = []
    for m, sphere in spheres(mol, pos, donors).items():
        rm = _PT.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i == m or a.GetAtomicNum() in COORDINATION_METALS:
                continue
            r_sum = rm + _PT.GetRcovalent(a.GetAtomicNum())
            if i in sphere:
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
    """Hold one judged M-D-X donation angle: the donor's class and the angle."""

    metal: int
    donor: int
    sub: int  # X, a heavy substituent of the donor
    element: str = field(kw_only=True)  # with `hyb` the class: hyb alone pools thiolate (103°) and carboxylate (126°)
    hyb: Chem.HybridizationType = field(kw_only=True)
    angle: float = field(kw_only=True)

    @property
    def cls(self) -> tuple[str, Chem.HybridizationType]:
        """Return the class this donation is judged against: ``(donor element, hybridisation)``."""
        return (self.element, self.hyb)


def donor_orientation(mol, pos, donors, frozen=frozenset(), *, hyb=None) -> list[Violation]:
    """Return a `Violation` for each M-D-X angle below its class's crystal-structure floor (`FOLD_WINDOW`).

    `metal_overbond` misses the commonest fold: a short D-X bond can swing side-on without reaching the over-bond floor
    while the M-D-X angle collapses. A violation flags an unusual angle, not proof of folding: real crystals sit below
    this floor too, so acceptance never rejects on it. Donors with no donation axis, an M-D-X unit inside the frozen
    core, an untyped donor or a class with no row are not judged.

    The caller must pass the intended `donors`: perceiving them would read a folded atom as a co-donor. `hyb` reuses
    hybridisations already read with this sphere's bound-donor counts.
    """
    out: list[Violation] = []
    for a in _donor_walk(mol, pos, donors, frozen, hyb=hyb):
        lo = FOLD_WINDOW[a.cls][0]
        if a.angle >= lo:  # floor only: gating the ceiling false-positives on healthy donors
            continue  # (see `FOLD_WINDOW`)
        xsym = mol.GetAtomWithIdx(a.sub).GetSymbol()
        out.append(
            Violation(
                kind="donor_orientation",
                atoms=(a.metal, a.donor, a.sub),
                value=a.angle,
                limit=lo,
                detail=f"{a.element}{a.donor} ({str(a.hyb).lower()}) M-D-X to {xsym}{a.sub}: "
                f"{a.angle:.1f}° < crystal-structure floor {lo:.0f}°; inspect donor geometry and restraints",
            )
        )
    return out


def _donor_walk(mol, pos, donors, frozen, *, hyb) -> list[DonorAngle]:
    """Walk M -> D -> X: one judged angle per heavy donor substituent. See ``donor_orientation`` for the exemptions."""
    by_metal = spheres(mol, pos, donors)
    bound = Counter(d for s in by_metal.values() for d in s)  # donor -> metal count; its keys are the co-donors
    if hyb is None:
        hyb = stripped_hybridisation(mol, bound)
    stripped = ligand_graph(mol)
    out: list[DonorAngle] = []
    for m, sphere in by_metal.items():
        for d in sorted(sphere):
            subs = donation_axis(mol, d, bound, sphere=sphere, hyb=hyb, metals=bound[d], stripped=stripped)
            if subs is None:  # hydride, bridging or haptic: the donation question does not apply
                continue
            cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None
            if cls not in FOLD_WINDOW:  # untyped, or an uncalibrated class (an sp3 ether O, an sp2 P)
                continue
            for x in subs:
                if graft_owns((m, d, x), frozen):
                    continue
                ang = bond_angle(pos[m], pos[d], pos[x])
                out.append(DonorAngle(m, d, x, element=cls[0], hyb=cls[1], angle=ang))
    return out


def coordinating_atoms(mol, pos, m, declared=frozenset()) -> set[int]:
    """Return the non-metal atoms inside metal ``m``'s shell (``_COORD_FACTOR`` x covalent sum).

    A heavy atom at that distance counts as a donor. A hydrogen counts only when ``declared``, since a hydride and an
    agostic C-H sit at the same M-H distance.
    """
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    rc = np.array([_PT.GetRcovalent(int(x)) for x in z])
    cand = ~np.isin(z, list(COORDINATION_METALS)) & ((z != 1) | np.isin(np.arange(len(z)), list(declared)))
    cand[m] = False
    dist = np.linalg.norm(pos[m] - np.asarray(pos, float), axis=-1)
    # A matrix norm may differ from the scalar norm in the last ulp: screen a hair wide, re-check with the scalar.
    near = cand & (dist <= _COORD_FACTOR * (rc[m] + rc) * (1 + 1e-9))
    return {
        i
        for i in np.flatnonzero(near).tolist()
        if float(np.linalg.norm(pos[m] - pos[i])) <= _COORD_FACTOR * (rc[m] + rc[i])
    }


def declared_donors(donors, m) -> set[int]:
    """Return the donors declared for metal `m`: from a ``{metal: donors}`` mapping, or one flat collection."""
    if isinstance(donors, Mapping):
        return {int(d) for d in donors.get(m, ())}
    return {int(d) for d in donors}


def spheres(mol, pos, donors) -> dict[int, set[int]]:
    """Return ``{metal: its coordination sphere}``: declared donors where declared, perceived where not.

    `donors` is ``{metal: donor atoms}``, or one collection for every metal; a partial declaration needs the mapping,
    since a flat collection cannot say which metal a bridge belongs to. A declared donor that left the shell is dropped,
    so a metal whose donors all left has an empty sphere rather than reading a collapsed atom as its donor.
    """
    out: dict[int, set[int]] = {}
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS):
        known = declared_donors(donors, m)
        shell = coordinating_atoms(mol, pos, m, known)  # `known` also admits a declared hydride the element screen
        out[m] = known & shell if known else shell  # would drop; else `known & shell` silently loses it
    return out


APICAL_MIN = 3  # atoms in a haptic face that caps a polyhedron face (a piano stool); eta2 fills one site
# Site occupancy, not geometry: it steers the default-polyhedron guess, never the embed.


# FIT_FLOOR: above it a sphere still reads as its nearest shape, but a conformer rxembed made is refused
FIT_FLOOR = 0.45  # Procrustes residual

# _FIT_MARGIN: two shapes within it tie, so a near-tie input reads the same at every seed
_FIT_MARGIN = 0.01


def site_rays(pos, metal, vertices, haptic):
    """Return the unit rays from `metal` to each occupied site of `vertices`, or None when one has no direction.

    A haptic face is one site at its centroid and a vacancy is skipped; a site within `EPS_LEN` of the metal, or not
    finite, points nowhere.
    """
    sites = [pos[list(haptic[v])].mean(axis=0) if v in haptic else pos[v] for v in vertices if v is not VACANT]
    rays = np.array(sites, float) - pos[metal]
    lengths = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / lengths if np.all(np.isfinite(lengths) & (lengths > EPS_LEN)) else None


def rank_shapes(obs, *, vacancy=None):
    """Return every same-CN polyhedron's `Fit` to `obs` by name, best first.

    `vacancy` is ``(name, ideal_dirs)``, a requested polyhedron's occupied-vertex directions ranked with the records of
    this coordination number when a vertex is vacant. Ties keep `POLYHEDRA` order. An approximate fit is an upper bound,
    so a ranking holding one is approximate.
    """
    fits = {name: fit_residual(obs, p.vertex_dirs) for name, p in POLYHEDRA.items() if p.cn == len(obs)}
    if vacancy is not None:
        name, ideal = vacancy
        fits[name] = fit_residual(obs, ideal)
    return dict(sorted(fits.items(), key=lambda item: item[1].residual))


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeReading:
    """Hold how a centre's sphere reads against its requested polyhedron, and which rule decided it.

    `verdict` is 'fits' or the first rule missed: 'unreadable' (no rays, so no residual), 'other_shape' (another
    polyhedron reads better by `_FIT_MARGIN` or more) or 'distorted' (residual above `FIT_FLOOR`). `next_name` is the
    best other polyhedron, if any. `exact` is False when a ranked residual is approximate, so the verdict proves
    nothing.
    """

    verdict: Literal["fits", "unreadable", "other_shape", "distorted"]
    residual: float | None = None
    next_name: str | None = None
    next_residual: float | None = None
    exact: bool = True

    @property
    def matches(self):
        """Return whether the sphere reads as the requested polyhedron: every site points somewhere, none reads better.

        The identity a given geometry has, so the readers ask this of any conformer, distorted or not.
        """
        return self.verdict in ("fits", "distorted")

    @property
    def accepted(self):
        """Return whether a conformer rxembed made takes the requested polyhedron: it matches within `FIT_FLOOR`."""
        return self.verdict == "fits"

    def clause(self, geometry):
        """Format the reading of `geometry` as its code, residual and runner-up: ``'TBP 0.053 (next SPY 0.198)'``.

        An approximate reading says so: ``'TCT 0.051 (next CTP 0.140), approximate'``.
        """
        if self.residual is None:
            return f"{record(geometry).code} unreadable"
        clause = f"{record(geometry).code} {self.residual:.3f}"
        if self.next_name is not None:
            clause += f" (next {record(self.next_name).code} {self.next_residual:.3f})"
        return clause if self.exact else f"{clause}, approximate"


def shape_reading(rays, requested, *, vacancy=None):
    """Return the `ShapeReading` of `requested` for the unit `rays`, ranked as `rank_shapes` ranks them with `vacancy`.

    ``None`` rays read as ``'unreadable'``.
    """
    if rays is None:
        return ShapeReading(verdict="unreadable")
    return _ranked_reading(rank_shapes(rays, vacancy=vacancy), requested)


def _ranked_reading(ranked, requested):
    """Return the `ShapeReading` of `requested` within `ranked`, `rank_shapes`' sorted output.

    `requested` must read within `_FIT_MARGIN` of the best, and its residual be at most `FIT_FLOOR`.
    """
    fit, best = ranked[requested], next(iter(ranked.values()))
    next_name = next((name for name in ranked if name != requested), None)
    if fit.residual - best.residual >= _FIT_MARGIN:
        verdict = "other_shape"
    else:
        verdict = "distorted" if fit.residual > FIT_FLOOR else "fits"
    return ShapeReading(
        verdict=verdict,
        residual=fit.residual,
        next_name=next_name,
        next_residual=None if next_name is None else ranked[next_name].residual,
        exact=all(other.exact for other in ranked.values()),
    )


def shape_gap(mol, metal, vertices, haptic, requested, cid=-1):
    """Return the `ShapeReading` of `requested` at `metal` on conformer `cid`: whether the centre takes it.

    The one shape judge: a reader asks whether the sphere `matches`, and what rxembed makes must be `accepted`. A vacant
    slot is read against the occupied coordination number, ranked with `requested`'s occupied-vertex subset. A planar
    shell's flatness is the umbrella gate's to judge.
    """
    occupied = [slot for slot, vertex in enumerate(vertices) if vertex is not VACANT]
    vacancy = None
    if len(occupied) < len(vertices):
        vacancy = (requested, [POLYHEDRA[requested].vertex_dirs[slot] for slot in occupied])
    rays = site_rays(mol.GetConformer(cid).GetPositions(), metal, vertices, haptic)
    return shape_reading(rays, requested, vacancy=vacancy)


@dataclass(frozen=True, slots=True, kw_only=True)
class BestShape:
    """Hold a sphere's best polyhedron `name`, its `reading`, and whether the next reads within `_FIT_MARGIN`."""

    name: str
    reading: ShapeReading
    tied: bool


def classify_geometry(mol, metal, sites, haptic, cid=-1):
    """Return the `BestShape` of a sphere, the polyhedron its site rays fit best, or None when none can be read.

    The least `rank_shapes` residual decides, with rotation, reflection and donor reordering allowed and no flatness
    filter; the caller says when nothing really fits or the reading is tied.
    """
    obs = site_rays(mol.GetConformer(cid).GetPositions(), metal, sites, haptic)
    ranked = {} if obs is None else rank_shapes(obs)
    if not ranked:
        return None
    name = next(iter(ranked))
    reading = _ranked_reading(ranked, name)
    tied = reading.next_name is not None and _ranked_reading(ranked, reading.next_name).matches
    return BestShape(name=name, reading=reading, tied=tied)


def geometry_for(n_donors, has_apical=False):
    """Return the default coordination polyhedron for `n_donors`, or None.

    An apical (eta>=3) face is an axial cone that cannot hold a neighbour at 90 deg or closer, so it takes the best
    default with no such pair: a CN4 piano stool is tetrahedral. An eta2 face is an ordinary vertex.
    """
    shells = geometries_for_cn(n_donors)  # best-default first
    if has_apical:
        shells = [p for p in shells if not p.planar or all(a > 90 for a in p.pair_angles.values())]  # noqa: PLR2004 - right angle
    return shells[0].name if shells else None
