"""The ``POLYHEDRA`` templates, plus the symmetry and chirality read off them.

The isomer identity rxembed selects on is geometric, not a chemistry name: cis/trans/mer/fac are fragile, and
a tris-chelate is lambda/delta rather than mer/fac. One flat ``Polyhedron`` record per geometry, and two
pure-numpy pieces over a record's ``vertex_dirs``:

- the point-group split (`point_group`): every vertex permutation is a proper (rotation) or improper
  (reflection) isometry of the template, and that group is what an arrangement canonicalises over.
- handedness (`handedness`): the centre's chirality as the parity of the canonicalising frame, achiral iff
  some reflection fixes the donor-class plus chelate-bite labelling.

Parity rather than RDKit's native metal stereo because that permutation is not order-invariant for equivalent
ligands. Attribution for the templates and the algorithm: README, "References".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np


def _vertex_angle(u, v):
    u, v = np.array(u), np.array(v)
    return round(float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))))


_FLAT_EPS = 1e-6  # smallest singular value below which a point set is one plane
_IMPROPER_VERTICES = 3  # an improper states exactly three vertices; a CN4 record would be held 3-of-4


def _improper(p1, p2, p3, p4):
    """Signed improper dihedral 1-2-3-4 in degrees, the convention `rdMolTransforms.GetDihedralDeg` uses."""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(m @ n2, n1 @ n2)))


@dataclass(frozen=True)
class Polyhedron:
    """One coordination geometry: its vertex template + the hand-authored constraint data keyed off it.

    `vertex_dirs` (unit directions) defines slot numbering; `angles` and `permutations` index those slots. At
    CN>=5 `angles` is a minimal spanning subset (octahedral states 6 of 15 pairs): the unconstrained pairs
    drift on the bond-less surrogate, but stating all of them is measured-refuted, with no fidelity gain. At
    CN<=4 all pairs IS the minimal subset, no trans row meaning no pair is implied by the others.
    ``angles=None`` derives every pair from the vectors, at CN7 and CN8 only. `permutations` is ``None`` when
    the geometry has no canned coordination-isomer list, which is load-bearing.

    `planar` is not decoration but the perception filter: a record declares whether its metal lies in its
    vertex plane, and `classify_geometry` only considers records whose declaration matches the measured
    coplanarity. That is what separates trigonal_planar from the CN3 pyramid without a fitted angle boundary,
    so every record must round-trip -- construct it, re-perceive it, get it back.
    """

    name: str
    default_rank: int  # preference within the CN (0 == the default polyhedron)
    vertex_dirs: tuple
    angles: tuple | None  # hand-authored subset; None -> derive from vertex_dirs
    permutations: tuple | None = None  # None -> no canned isomer list
    planar: bool = False  # metal and vertices coplanar by definition, and the `classify_geometry` filter
    geometric_isomerism: bool = True  # False -> one arrangement only, no cis/trans label
    code: str = ""  # 3-letter spelling, accepted wherever a geometry name is (`resolve_geometry`).
    # Not the IUPAC polyhedral symbol, which differs for most rows: TPY here is the CN3 pyramid, and other
    # conventions spell a CN4 one the same way, so anything translating must key on the coordination number.

    @property
    def cn(self):
        """Return the coordination number: the vertex count (== len(vertex_dirs)), derived not stored."""
        return len(self.vertex_dirs)

    @property
    def umbrella_improper(self):
        """Ideal |improper| of vertices 0-1-2 vs the metal (deg); ``None`` unless the record is a flat-based pyramid.

        The pyramidalisation coordinate a D-M-D angle basis cannot state. An improper is scale-free, so one
        number per record serves every bond length, where a D-M-D window would have to move with it.

        Selected by geometry, not by shape name: a non-`planar` record whose own vertices are coplanar, so a
        single improper describes the whole umbrella. Today that is `trigonal_pyramidal` alone, at 35.264°.

        Exact only AT the record. For a distorted sphere the improper hinges on one base edge, so clearing the
        wall on the seated edge does not settle the shape; what holds in practice is measured, not
        constructional.
        """
        if self.planar or self.cn != _IMPROPER_VERTICES:
            return None
        v = np.array([np.asarray(d, float) / np.linalg.norm(d) for d in self.vertex_dirs])
        if np.linalg.svd(v - v.mean(axis=0))[1][2] > _FLAT_EPS:  # a 3-D vertex set: not one umbrella
            return None
        return abs(_improper(v[0], v[1], v[2], np.zeros(3)))

    @property
    def resolved_angles(self):
        """Return the hand-authored `angles`, or every vertex pair derived from the template (CN7/CN8)."""
        if self.angles is not None:
            return self.angles
        v = self.vertex_dirs
        return tuple((i, j, _vertex_angle(v[i], v[j])) for i in range(len(v)) for j in range(i + 1, len(v)))


_TBP_ISOMERS = (
    (0, 1, 2, 3, 4),
    (0, 2, 1, 3, 4),
    (0, 3, 1, 2, 4),
    (0, 4, 1, 2, 3),
    (1, 2, 0, 3, 4),
    (1, 3, 0, 2, 4),
    (1, 4, 0, 2, 3),
    (2, 3, 0, 1, 4),
    (2, 4, 0, 1, 3),
    (3, 4, 0, 1, 2),
)
_SPY_ISOMERS = (
    (0, 1, 2, 3, 4),
    (0, 1, 3, 2, 4),
    (0, 1, 4, 2, 3),
    (1, 0, 2, 3, 4),
    (1, 0, 3, 2, 4),
    (1, 0, 4, 2, 3),
    (2, 1, 0, 3, 4),
    (2, 1, 3, 0, 4),
    (2, 1, 4, 0, 3),
    (3, 1, 2, 0, 4),
    (3, 1, 0, 2, 4),
    (3, 1, 4, 2, 0),
    (4, 1, 2, 3, 0),
    (4, 1, 3, 2, 0),
    (4, 1, 0, 2, 3),
)
_OCT_ISOMERS = (
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 4, 3, 5),
    (0, 1, 2, 5, 3, 4),
    (0, 2, 1, 3, 4, 5),
    (0, 2, 1, 4, 3, 5),
    (0, 2, 1, 5, 3, 4),
    (0, 3, 2, 1, 4, 5),
    (0, 3, 2, 4, 1, 5),
    (0, 3, 2, 5, 1, 4),
    (0, 4, 2, 3, 1, 5),
    (0, 4, 2, 1, 3, 5),
    (0, 4, 2, 5, 3, 1),
    (0, 5, 2, 3, 4, 1),
    (0, 5, 2, 4, 3, 1),
    (0, 5, 2, 1, 3, 4),
)


# The supported polyhedra, one record each. Insertion order is the classify_geometry tie-break;
# `geometries_for_cn` groups by CN for the default and the "did you mean".
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        Polyhedron(
            name="linear",
            code="LIN",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1)),
            angles=((0, 1, 180),),
            planar=True,
            geometric_isomerism=False,
        ),
        Polyhedron(
            name="trigonal_planar",
            code="TPL",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)),
            angles=((0, 1, 120), (1, 2, 120), (0, 2, 120)),
            planar=True,
            geometric_isomerism=False,
        ),
        Polyhedron(
            name="t_shape",
            code="TSH",
            default_rank=1,
            vertex_dirs=((0, 0, 1), (1, 0, 0), (0, 0, -1)),  # 0,2 trans; 1 perpendicular
            angles=((0, 1, 90), (1, 2, 90), (0, 2, 180)),
            permutations=((0, 1, 2), (1, 0, 2), (0, 2, 1)),
            planar=True,
        ),
        # the IUPAC TPY-3 pyramid: three donors as a base, the metal at the apex, at the vacant tetrahedron's
        # 109.47 deg. Ammonia's 107 is refused, being a lone-pair number this does not model. Angle cannot
        # tell it from `trigonal_planar` either, so the discriminator is `planar`, a rule with a natural zero.
        Polyhedron(
            name="trigonal_pyramidal",
            code="TPY",
            default_rank=2,
            # the base, wound counterclockwise from +x like trigonal_planar / tbp's equator, tipped so the
            # metal sits on the +z apex (z = -1/3, the tetrahedral cap)
            vertex_dirs=(
                (0.9428090, 0, -0.3333333),
                (-0.4714045, 0.8164966, -0.3333333),
                (-0.4714045, -0.8164966, -0.3333333),
            ),
            angles=((0, 1, 109.5), (1, 2, 109.5), (0, 2, 109.5)),  # 109.5 as `tetrahedral` spells it
            # No canned list, as `tetrahedral` has none: the three vertices are one orbit under C3v, so there is no
            # geometric isomerism to enumerate. Metal-centred enantiomers are not enumerated for either shape.
            permutations=None,
            geometric_isomerism=False,
            # planar=False separates it from trigonal_planar, and deliberately not by a nearest-template
            # angle contest: on the 5 corpus CN3 centres the TPL-over-TPY margin is only +1.6°, because
            # in-plane Y-distortion moves the spectrum. Coplanarity is blind to that, which is the point.
        ),
        Polyhedron(
            name="square_planar",
            code="SPL",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),  # 0,2 and 1,3 trans; 0,1 cis
            angles=((0, 2, 180), (1, 3, 180), (0, 1, 90)),
            permutations=((0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1)),
            planar=True,
        ),
        Polyhedron(
            name="tetrahedral",
            code="TET",
            default_rank=1,
            vertex_dirs=((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
            angles=((0, 1, 109.5), (0, 2, 109.5), (0, 3, 109.5), (1, 2, 109.5), (1, 3, 109.5), (2, 3, 109.5)),
            geometric_isomerism=False,
        ),
        # its trans pair is 0,1 as octahedral/tbp, not square_planar's 0,2. Idealised 180°/120°, knowingly
        # wider than SF4 (173/102, a lone pair this does not model) or the d8 sawhorse Fe(CO)4 (~150/120):
        # no census pins a d-block value, so a real sawhorse rides the axial wall. Distort off a census only.
        Polyhedron(
            name="seesaw",
            code="SEE",
            default_rank=2,
            vertex_dirs=(
                (0, 0, 1),
                (0, 0, -1),
                (1, 0, 0),
                (-0.5, math.sqrt(3) / 2, 0),
            ),  # 0,1 axial; 2,3 equatorial 120°
            angles=((0, 1, 180), (2, 3, 120), (0, 3, 90), (1, 2, 90)),
            permutations=((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)),
        ),
        Polyhedron(
            name="trigonal_bipyramidal",
            code="TBP",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)),
            angles=((0, 1, 180), (1, 2, 90), (0, 3, 90), (2, 3, 120), (3, 4, 120), (2, 4, 120)),
            permutations=_TBP_ISOMERS,
        ),
        # basal donors sit below the metal's equatorial plane (apex-basal ~105°, trans-basal ~150°): a real
        # pyramid, not an octahedron minus a vertex, and a flat 90/180 template misclassifies VOacac2 as tbp.
        # subset kept: all 10 pairs makes it rigid but gains no crystal fidelity (same verdict as all-pairs).
        Polyhedron(
            name="square_pyramidal",
            code="SPY",
            default_rank=1,
            vertex_dirs=(
                (0, 0, 1),  # 0 apex; 1-4 basal at ±x/±y, = [sin105·cosφ, sin105·sinφ, cos105]
                (0.965926, 0.0, -0.258819),
                (0.0, 0.965926, -0.258819),
                (-0.965926, 0.0, -0.258819),
                (0.0, -0.965926, -0.258819),
            ),
            angles=((0, 1, 105), (0, 3, 105), (1, 3, 150), (2, 4, 150), (1, 4, 86), (2, 3, 86)),
            permutations=_SPY_ISOMERS,
        ),
        Polyhedron(
            name="octahedral",
            code="OCT",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),  # trans PAIRS:
            #   0,1 / 2,3 / 4,5, unlike square_planar, where the trans partner of 0 is 2
            angles=((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
            permutations=_OCT_ISOMERS,
        ),
        Polyhedron(
            name="trigonal_prismatic",
            code="TPR",
            default_rank=1,
            vertex_dirs=(  # two ECLIPSED equilateral triangles (D3h). Regular => every edge equal, so the
                (0.7559289, 0.0, 0.6546537),  # half-height is sqrt(3)/2 of the circumradius: (cos t, sin t,
                (-0.3779645, 0.6546537, 0.6546537),  # +-sqrt(3)/2), normalised. Spectrum 81.8 x9 / 135.6 x6.
                (-0.3779645, -0.6546537, 0.6546537),
                (0.7559289, 0.0, -0.6546537),
                (-0.3779645, 0.6546537, -0.6546537),
                (-0.3779645, -0.6546537, -0.6546537),
            ),
            angles=None,  # DERIVED, unlike its CN6 neighbour: octahedral's minimal subset was measured to be
            #   sufficient AND better than all-pairs, and nobody has run that measurement on this record.
            #   Deriving is the honest default; a subset here would be a guess wearing octahedral's evidence.
        ),
        Polyhedron(
            name="pentagonal_bipyramidal",
            code="PBP",
            default_rank=0,
            vertex_dirs=(  # 0,1 axial (±z); 2-6 an equatorial pentagon (72° spacing)
                (0, 0, 1),
                (0, 0, -1),
                (1, 0, 0),
                (0.309017, 0.951057, 0),
                (-0.809017, 0.587785, 0),
                (-0.809017, -0.587785, 0),
                (0.309017, -0.951057, 0),
            ),
            angles=None,  # angles derived; no canned permutation list
        ),
        Polyhedron(
            name="capped_octahedral",
            code="COC",
            default_rank=1,
            vertex_dirs=(  # the octahedron stood on a C3 axis (polar angle acos(1/sqrt3) = 54.74 deg, azimuths
                (0.0, 0.0, 1.0),  # 0/120/240 up and 60/180/300 down) plus a cap on the exposed face (C3v).
                (0.8164966, 0.0, 0.5773503),  # Spectrum 54.7 x3 / 90 x12 / 125.3 x3 / 180 x3.
                (-0.4082483, 0.7071068, 0.5773503),
                (-0.4082483, -0.7071068, 0.5773503),
                (0.4082483, 0.7071068, -0.5773503),
                (-0.8164966, 0.0, -0.5773503),
                (0.4082483, -0.7071068, -0.5773503),
            ),
            angles=None,
        ),
        Polyhedron(
            name="capped_trigonal_prismatic",
            code="CTP",
            default_rank=2,
            vertex_dirs=(  # the trigonal prism above, capped on the outward normal of one square face, which
                (0.7559289, 0.0, 0.6546537),  # face is spanned by the azimuth-0 and azimuth-120 edges, so its
                (-0.3779645, 0.6546537, 0.6546537),  # normal is at azimuth 60 in the equator (C2v).
                (-0.3779645, -0.6546537, 0.6546537),
                (0.7559289, 0.0, -0.6546537),
                (-0.3779645, 0.6546537, -0.6546537),
                (-0.3779645, -0.6546537, -0.6546537),
                (0.5, 0.8660254, 0.0),
            ),
            angles=None,
        ),
        Polyhedron(
            name="square_antiprism",
            code="SQA",
            default_rank=0,
            vertex_dirs=(  # two staggered squares: top (+z) 0-3, bottom (-z) 4-7, rotated 45°
                (-0.5773503, 0.5773503, 0.5773503),
                (0.5773503, 0.5773503, 0.5773503),
                (-0.5773503, -0.5773503, 0.5773503),
                (0.5773503, -0.5773503, 0.5773503),
                (-0.8164966, 0.0, -0.5773503),
                (0.0, -0.8164966, -0.5773503),
                (0.8164966, 0.0, -0.5773503),
                (0.0, 0.8164966, -0.5773503),
            ),
            angles=None,
        ),
        Polyhedron(
            name="dodecahedral",
            code="DOD",
            default_rank=1,
            vertex_dirs=(  # the triangular dodecahedron / bisdisphenoid (D2d): two orthogonal disphenoids,
                (0.5792812, 0.0, 0.8151278),  # A at polar 35.40 deg alternating +-z, B at 74.30 deg. Those two
                (-0.5792812, 0.0, 0.8151278),  # angles are solved, not tabulated: they minimise the spread of
                (0.0, 0.5792812, -0.8151278),  # the 18 edges (8 vertices, 18 edges, 12 faces; Euler checks).
                (0.0, -0.5792812, -0.8151278),  # An exactly equal-edge form does not exist on a sphere, which
                (0.0, 0.9626917, 0.2706004),  # is why the literature calls this the MOST SPHERICAL compromise;
                (0.0, -0.9626917, 0.2706004),  # residual spread 0.091. It sits 9.1 deg from square_antiprism
                (0.9626917, 0.0, -0.2706004),  # in the angle spectrum, so the two are separable.
                (-0.9626917, 0.0, -0.2706004),
            ),
            angles=None,
        ),
    ]
}


_BY_CODE = {p.code: p.name for p in POLYHEDRA.values() if p.code}
# the one alias table: long name and 3-letter code, both case-insensitive. No name is 3 characters and no
# uppercased name collides with a code (pinned by `test_no_alias_collides`), so one flat dict is unambiguous.
_ALIASES = {**{n.lower(): n for n in POLYHEDRA}, **{c.lower(): n for c, n in _BY_CODE.items()}}


def resolve_geometry(geometry):
    """Return the canonical name for a geometry name or 3-letter code, case- and whitespace-insensitively.

    An unrecognised string passes through unchanged, so the caller raises its own "unknown geometry" error.
    The long name is the stored identity; a code is input only.
    """
    return _ALIASES.get(geometry.strip().lower(), geometry) if isinstance(geometry, str) else geometry


def record(geometry):
    """Return the `Polyhedron` for a name or 3-letter code, or ``None`` when no record has that name."""
    return POLYHEDRA.get(resolve_geometry(geometry)) if isinstance(geometry, str) else None


def describe(geometry):
    """Return a log-ready identity for a geometry: ``'square_planar (SPL, CN 4)'``.

    The one formatter every log line names a shape through, so no message is ambiguous about which polytope it
    means. An unknown string (the ``'N-coordinate'`` pseudo-name) is returned as-is, never invented into a record.
    """
    p = record(geometry)
    return f"{p.name} ({p.code}, CN {p.cn})" if p else str(geometry)


def vertex_dirs(geometry):
    """Return `geometry`'s vertex unit-direction template, or ``None`` if there is no such record."""
    return getattr(record(geometry), "vertex_dirs", None)


def isomer_permutations(geometry):
    """Return `geometry`'s canned coordination-isomer vertex orderings, or ``None`` if it has none."""
    return getattr(record(geometry), "permutations", None)


def is_planar(geometry):
    """Return True if `geometry`'s metal and vertices are coplanar by definition (unknown -> False)."""
    return bool(getattr(record(geometry), "planar", False))


def geometries_for_cn(n):
    """Return the supported polyhedra for coordination number `n`, best-default first (by `default_rank`)."""
    return sorted((p for p in POLYHEDRA.values() if p.cn == n), key=lambda p: p.default_rank)


def _fit_trace(h):
    """Return the best alignment achievable from the cross-covariance `h`, reflections allowed.

    Naming a shape is not asking which enantiomer: an octahedron and its mirror are both octahedra, and most
    templates here are achiral anyway (their point group already contains the improper operation, so the
    "mirror" seating is the same seating). Handedness is a separate question, answered over the point group
    by `handedness`, and it must stay separate: forbidding the reflection here was measured to seat three donors
    of a real octahedron (DUGVUX) into TRANS slots that subtend 93 degrees.
    """
    return float(np.linalg.svd(h, compute_uv=False).sum())


def _seat_by_alignment(dd, v_ideal, rounds=3):
    """Seat each donor at its nearest ideal vertex; return ``order`` (vertex -> index into the donor list).

    Used two ways: to seat a real sphere on a record (`metal_isomers`), and to score how well it fits
    one at all (`fit_residual`). Enumerating orderings is n!, or 40 320 at CN8, so
    the rotation is seeded instead: three correspondences fix an orthogonal map, so every ordered triple of
    donors is tried against vertices 0-1-2 (P(8,3) = 336), each is completed by assigning the remaining
    vertices to their best-pointing donor (best pair first, so one bad vertex cannot cascade), and the best is
    refined. Measured against a full n! search on the corpus's CN7 and CN8 structures: identical seating.

    A single greedy pass from the identity is not enough: on a square antiprism it converges to a local
    optimum scoring 5.63 where the true seating scores 7.90, because the starting alignment is meaningless.
    """
    n = len(v_ideal)

    def complete(rot):
        """Assign every vertex to the donor pointing most nearly at it, under rotation `rot`."""
        fit = (dd @ rot) @ v_ideal.T  # fit[donor, vertex]
        order, taken_d, taken_v = [0] * n, set(), set()
        for d, v in zip(*np.unravel_index(np.argsort(fit, axis=None)[::-1], fit.shape), strict=True):
            if d not in taken_d and v not in taken_v:
                order[int(v)] = int(d)
                taken_d.add(d)
                taken_v.add(v)
        return order

    def align(order):
        u, _s, vt = np.linalg.svd(dd[list(order)].T @ v_ideal)
        return u @ vt

    def score(order):
        return _fit_trace(dd[list(order)].T @ v_ideal)

    seeds = {
        tuple(complete(align([*seed, *[k for k in range(n) if k not in seed]])))
        for seed in itertools.permutations(range(n), min(3, n))
    }
    best = max(seeds, key=score)
    for _ in range(rounds):  # refine: re-align on the winner, re-assign, until it stops moving
        nxt = complete(align(list(best)))
        if score(nxt) <= score(best):
            break
        best = tuple(nxt)
    return list(best)


_SYM_TOL = 1e-6  # residual below which a vertex permutation is an exact template isometry
LAMBDA, DELTA = "lambda", "delta"  # the two metal-centre handedness tags: typeable, not glyphs
_TAGS = {LAMBDA: LAMBDA, DELTA: DELTA, "λ": LAMBDA, "δ": DELTA}  # what `chirality_tag` accepts as input


def chirality_tag(chirality):
    """Return the canonical handedness tag for `chirality`, accepting the Δ/Λ glyphs as input.

    The stored form is the word (``'delta'`` / ``'lambda'``) so nothing downstream has to carry a glyph;
    the glyphs are accepted here because they are what the literature and older callers use.
    """
    if not isinstance(chirality, str):
        return chirality
    return _TAGS.get(chirality.strip().lower(), chirality)  # "Δ".lower() is "δ", hence both spellings


@lru_cache(maxsize=None)
def _perms(n):
    return tuple(permutations(range(n)))


@lru_cache(maxsize=None)
def point_group(dirs):
    """Return ``(rotations, reflections)``: the vertex perms realisable by a proper / improper isometry.

    `dirs` is a hashable tuple of vertex unit directions (a geometry's ``vertex_dirs`` template). A perm is a
    template symmetry iff the template maps onto its permuted self with ~0 residual under the best
    orthogonal map of that parity; a planar/degenerate template (linear, square-planar) realises the same
    perm both ways, so those are always achiral. Cached per template.
    """
    t = np.array(dirs, float)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    n = len(t)
    perms = np.array(_perms(n))
    permuted = t[perms]  # (P, n, 3)
    u, _, wt = np.linalg.svd(np.einsum("pni,nj->pij", permuted, t))  # per-perm permuted.T @ t
    sgn = np.sign(np.linalg.det(u @ wt))
    sgn[sgn == 0] = 1.0

    def realises(det_sign):  # perms whose best orthogonal fit (of this parity) is exact
        d = np.broadcast_to(np.eye(3), (len(perms), 3, 3)).copy()
        d[:, 2, 2] = det_sign
        resid = ((np.einsum("ni,pji->pnj", t, u @ d @ wt) - permuted) ** 2).sum(axis=(1, 2))
        return frozenset(tuple(int(x) for x in perms[p]) for p in np.flatnonzero(resid < _SYM_TOL))

    return realises(sgn), realises(-sgn)  # (rotations det +1, reflections det -1)


def handedness(dirs, order, donor_class, chelate_edges=frozenset()):
    """Return a metal centre's handedness over template `dirs`: ``'delta'``, ``'lambda'``, or ``''`` (achiral).

    `order[vertex]` is the donor atom seated at that vertex; `donor_class[donor]` its symmetry class (so
    equivalent donors share a label); `chelate_edges` the set of ``frozenset({vertex_i, vertex_j})`` whose
    two donors belong to one chelating ligand (a tris/bis-chelate's handedness lives in this bite graph, not the
    per-vertex donor class). Achiral iff a reflection of the template maps the decorated labelling to
    itself; else the sign is the parity of the frame that canonicalises it.

    A vacant vertex (donor ``metal.VACANT``, i.e. < 0) leaves the centre unfixed -> achiral (``''``): a
    parity needs every vertex occupied.
    """
    n = len(dirs)
    if len(order) != n or any(d < 0 for d in order):  # a vacancy (or padding) can't fix a parity
        return ""
    rot, refl = point_group(tuple(map(tuple, dirs)))
    label = {v: donor_class[order[v]] for v in range(n)}
    edges = [tuple(e) for e in chelate_edges]

    def form(q):  # the decorated arrangement in the frame `q`: (per-vertex labels, chelate bite edges)
        verts = tuple(label[q.index(v)] for v in range(n))
        bites = tuple(sorted(tuple(sorted((q[a], q[b]))) for a, b in edges))
        return verts, bites

    base = form(tuple(range(n)))
    if any(form(q) == base for q in refl):  # a mirror fixes labels AND bites -> no handedness
        return ""
    q_star = min(rot | refl, key=form)  # the canonicalising frame; its parity is the sign
    return DELTA if q_star in rot else LAMBDA


def fit_residual(dirs_obs, record):
    """Return how badly `dirs_obs` fails to BE `record`: RMS per vertex after the best rotation.

    Orthogonal Procrustes: seat the observed donor directions on the record's vertices, find the orthogonal map
    best carrying one onto the other (SVD of the cross-covariance), and report what is left over. Reflections
    are allowed (see `_fit_trace`): a shape and its mirror are the same shape.

    Chosen over the sorted angle spectrum it replaced for two measured reasons. The spectrum's length is
    C(n,2), so an RMS over it changes scale with coordination number: 0.08 deg median at CN2 against 9.10 at
    CN5 on the corpus, which is why one global fit floor could not serve both ends; this residual is per
    vertex and sits at 0.066-0.135 across CN 3-8. And the spectrum discards which vertex is which, so it
    cannot tell a distorted octahedron from a trigonal prism by anything but a scalar; measured, it handed
    the cis-dioxo Mo BESJUE to `trigonal_prismatic` the moment that record was added.
    """
    ideal = np.array(record.vertex_dirs, float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    order = _seat_by_alignment(dirs_obs, ideal)
    u, _s, vt = np.linalg.svd(dirs_obs[order].T @ ideal)
    return float(np.sqrt(np.mean(np.sum((dirs_obs[order] @ (u @ vt) - ideal) ** 2, axis=1))))
