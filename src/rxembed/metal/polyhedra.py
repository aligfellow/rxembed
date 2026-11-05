"""Define the coordination polyhedra, every fact derived from their vertices, and how a seating rotates or mirrors.

Arrangements are folded over proper template rotations, and an improper frame identifies the opposite hand of a chiral
sphere, so canonical slots and metal hands do not depend on atom order, unlike RDKit's metal stereo permutation.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import NamedTuple

import numpy as np

from rxembed.geometry.constraints import FIX_ANGLE_TOL

VACANT = None  # an empty vertex: a slot no donor fills, wherever a vertex is listed
_FLAT_EPS = 1e-6  # smallest singular value below which a point set is one plane
IMPROPER_VERTICES = 3  # an improper states exactly three vertices; a CN4 record would be held 3-of-4
_TRANS_ANGLE = 135  # deg: a pair this wide is trans; a same-ligand pair this wide needs a trans-spanning backbone
LABELLED_CN = 6  # cis/trans and mer/fac name isomers up to six vertices; past six that meaning is lost
# A resource limit on the proper-rotation orbits one seating search scores, not chemistry.
_MAX_SEATING_ORBITS = 1_000
_AXIS_EPS = 1e-12  # a bite pair this close to antipodal has no well-defined rotation axis
_SHELL_DAMPING = 0.5  # relaxed_shell's per-round step; undamped, a closed bite cycle oscillates, never converging
FLOAT_CACHE = 16_384  # entries per float-keyed memo: bite targets vary continuously, so bound it for batch runs
_SYM_TOL = 1e-6  # residual below which a vertex permutation is an exact template isometry
_BSA_POLAR = 1.1343877116  # rad: the square rings' polar angle in the ten-point Thomson minimum


def improper_dihedral(p1, p2, p3, p4):
    """Return the signed improper dihedral 1-2-3-4 in degrees, in `rdMolTransforms.GetDihedralDeg`'s convention."""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(m @ n2, n1 @ n2)))


def _unit(rays):
    """Return `rays` as an array of unit vectors."""
    v = np.asarray(rays, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def trans_pairs(rays):
    """Return the index pairs ``(i, j)``, ``i < j``, of `rays` at least `_TRANS_ANGLE` apart: the one trans rule.

    A polyhedron's trans pairs answer every trans question: which same-ligand pairs get no bite window, where no short
    chelate sits, which triples an oriented type ignores, and what cis/trans and mer/fac call trans.
    """
    v = _unit(rays)
    limit = math.cos(math.radians(_TRANS_ANGLE))
    return frozenset(
        pair for pair in itertools.combinations(range(len(v)), 2) if float(v[pair[0]] @ v[pair[1]]) <= limit
    )


@dataclass(frozen=True)
class Polyhedron:
    """Describe one coordination geometry by its vertex directions; every other fact is derived from them.

    `vertex_dirs` defines the slots.
    """

    name: str
    default_rank: int  # preference within the CN (0 == the default polyhedron)
    vertex_dirs: tuple
    # 3-letter code, not the IUPAC symbol: TPY here is the CN3 pyramid, which others use for CN4
    code: str = ""
    site_groups: tuple = ()  # optional conventional display groups: ((name, vertex indices), ...)

    @property
    def cn(self):
        """Return the coordination number: the vertex count (== len(vertex_dirs)), derived not stored."""
        return len(self.vertex_dirs)

    @cached_property
    def pair_angles(self):
        """Return every vertex pair's angle in degrees, unrounded, keyed by ``frozenset((i, j))``.

        Every pair, since a subset under-determines the polyhedron and loses references and isomers.
        """
        v = _unit(self.vertex_dirs)
        return {
            frozenset(pair): float(np.degrees(np.arccos(np.clip(v[pair[0]] @ v[pair[1]], -1.0, 1.0))))
            for pair in itertools.combinations(range(self.cn), 2)
        }

    @cached_property
    def trans_pairs(self):
        """Return the vertex pairs ``(i, j)``, ``i < j``, the trans rule (`trans_pairs`) calls trans."""
        return trans_pairs(self.vertex_dirs)

    @cached_property
    def planar(self):
        """Return whether the metal and every vertex lie in one plane."""
        return bool(np.linalg.matrix_rank(_unit(self.vertex_dirs), tol=_SYM_TOL) < 3)  # noqa: PLR2004 - space

    @cached_property
    def cis_trans(self):
        """Return whether cis and trans name this polyhedron's donor pairs: a pair can sit on a trans pair or off one.

        Never above `LABELLED_CN` vertices.
        """
        return self.cn <= LABELLED_CN and 0 < len(self.trans_pairs) < len(self.pair_angles)

    @cached_property
    def mer_fac(self):
        """Return whether mer and fac name its donor triads: they do where trans pairs pair off every vertex in 3D."""
        paired = sorted(vertex for pair in self.trans_pairs for vertex in pair)
        return self.cis_trans and not self.planar and paired == list(range(self.cn))

    @property
    def umbrella_improper(self):
        """Return the ideal flat-based-pyramid improper, or None.

        The scale-free improper gives pyramidalisation without a length-dependent D-M-D window; it is exact only for the
        ideal record.
        """
        if self.planar or self.cn != IMPROPER_VERTICES:
            return None
        v = _unit(self.vertex_dirs)
        if np.linalg.svd(v - v.mean(axis=0))[1][2] > _FLAT_EPS:  # a 3-D vertex set: not one umbrella
            return None
        return abs(improper_dihedral(v[0], v[1], v[2], np.zeros(3)))


# the polyhedra; insertion order breaks classify_geometry ties, and `geometries_for_cn` groups by CN
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        # one donor has a distance but no angle; kept apart from linear so no open site is invented
        Polyhedron(
            name="monocoordinate",
            code="MCO",
            default_rank=0,
            vertex_dirs=((0, 0, 1),),
        ),
        Polyhedron(
            name="linear",
            code="LIN",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1)),
        ),
        Polyhedron(
            name="trigonal_planar",
            code="TPL",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)),
        ),
        Polyhedron(
            name="t_shape",
            code="TSH",
            default_rank=1,
            vertex_dirs=((0, 0, 1), (1, 0, 0), (0, 0, -1)),  # 0,2 trans; 1 perpendicular
            site_groups=(("axial", (0, 2)), ("equatorial", (1,))),
        ),
        # the TPY-3 pyramid: three donors as a base, the metal at the apex, at the tetrahedral 109.47 deg
        Polyhedron(
            name="trigonal_pyramidal",
            code="TPY",
            default_rank=2,
            # the base wound counterclockwise from +x, tipped so the metal sits on the +z apex (z = -1/3)
            vertex_dirs=(
                (0.9428090, 0, -0.3333333),
                (-0.4714045, 0.8164966, -0.3333333),
                (-0.4714045, -0.8164966, -0.3333333),
            ),
        ),
        Polyhedron(
            name="square_planar",
            code="SPL",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),  # 0,2 and 1,3 trans; 0,1 cis
        ),
        Polyhedron(
            name="tetrahedral",
            code="TET",
            default_rank=1,
            vertex_dirs=((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
        ),
        # trans pair 0,1 as octahedral; idealised 180/120 deg, since no crystal-structure fit pins a d-block sawhorse
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
            site_groups=(("axial", (0, 1)), ("equatorial", (2, 3))),
        ),
        Polyhedron(
            name="trigonal_bipyramidal",
            code="TBP",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)),
            site_groups=(("axial", (0, 1)), ("equatorial", (2, 3, 4))),
        ),
        # basal donors sit below the metal (apex-basal ~105 deg): a flat 90/180 template misreads it as tbp
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
            site_groups=(("apical", (0,)), ("basal", (1, 2, 3, 4))),
        ),
        Polyhedron(
            name="octahedral",
            code="OCT",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),  # trans pairs:
            #   0,1 / 2,3 / 4,5, unlike square_planar, where the trans partner of 0 is 2
        ),
        Polyhedron(
            name="trigonal_prismatic",
            code="TPR",
            default_rank=1,
            vertex_dirs=(  # two eclipsed equilateral triangles (D3h). Regular => every edge equal, so the
                (0.7559289, 0.0, 0.6546537),  # half-height is sqrt(3)/2 of the circumradius: (cos t, sin t,
                (-0.3779645, 0.6546537, 0.6546537),  # +-sqrt(3)/2), normalised. Spectrum 81.8 x9 / 135.6 x6.
                (-0.3779645, -0.6546537, 0.6546537),
                (0.7559289, 0.0, -0.6546537),
                (-0.3779645, 0.6546537, -0.6546537),
                (-0.3779645, -0.6546537, -0.6546537),
            ),
        ),
        Polyhedron(
            name="hexagonal_planar",
            code="HPL",
            default_rank=2,
            vertex_dirs=tuple(
                (math.cos(angle), math.sin(angle), 0.0)
                for angle in (0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3)
            ),
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
            site_groups=(("axial", (0, 1)), ("equatorial", (2, 3, 4, 5, 6))),
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
        ),
        Polyhedron(
            name="dodecahedral",
            code="DOD",
            default_rank=1,
            vertex_dirs=(  # the triangular dodecahedron / bisdisphenoid (D2d): two orthogonal disphenoids,
                (0.5792812, 0.0, 0.8151278),  # A at polar 35.40 deg alternating +-z, B at 74.30 deg: the angles
                (-0.5792812, 0.0, 0.8151278),  # that minimise the spread of the 18 edges, since no equal-edge
                (0.0, 0.5792812, -0.8151278),  # form exists on a sphere
                (0.0, -0.5792812, -0.8151278),
                (0.0, 0.9626917, 0.2706004),
                (0.0, -0.9626917, 0.2706004),
                (0.9626917, 0.0, -0.2706004),
                (-0.9626917, 0.0, -0.2706004),
            ),
        ),
        Polyhedron(
            name="tricapped_trigonal_prismatic",
            code="TCT",
            default_rank=0,
            # a trigonal prism plus its three rectangle normals, built exactly so it keeps D3h symmetry
            vertex_dirs=(
                *(
                    (math.sqrt(4 / 7) * math.cos(phi), math.sqrt(4 / 7) * math.sin(phi), z)
                    for z in (math.sqrt(3 / 7), -math.sqrt(3 / 7))
                    for phi in (0, 2 * math.pi / 3, 4 * math.pi / 3)
                ),
                (-1, 0, 0),
                (0.5, -math.sqrt(3) / 2, 0),
                (0.5, math.sqrt(3) / 2, 0),
            ),
        ),
        Polyhedron(
            name="bicapped_square_antiprismatic",
            code="BSA",
            default_rank=0,
            vertex_dirs=(  # two poles and two staggered squares: the ten-point Thomson minimum (D4d)
                (0, 0, 1),
                *(
                    (math.sin(polar) * math.cos(phi), math.sin(polar) * math.sin(phi), math.cos(polar))
                    for polar, offset in ((_BSA_POLAR, 0), (math.pi - _BSA_POLAR, math.pi / 4))
                    for phi in (offset + k * math.pi / 2 for k in range(4))
                ),
                (0, 0, -1),
            ),
        ),
        Polyhedron(
            name="edge_contracted_icosahedral",
            code="ECI",
            default_rank=0,
            vertex_dirs=(  # the eleven-point Thomson minimum (C2v), solved numerically, its C2 axis on z
                (0.0, 0.591976946100, 0.805954896558),
                (0.0, -0.591976946100, 0.805954896558),
                (0.833392584869, 0.0, 0.552681462947),
                (-0.833392584869, 0.0, 0.552681462947),
                (0.488922403640, 0.855926337070, -0.168359700430),
                (-0.488922403640, 0.855926337070, -0.168359700430),
                (-0.488922403640, -0.855926337070, -0.168359700430),
                (0.488922403640, -0.855926337070, -0.168359700430),
                (0.857005571875, 0.0, -0.515307141204),
                (-0.857005571875, 0.0, -0.515307141204),
                (0.0, 0.0, -1.0),
            ),
        ),
    ]
}


_BY_CODE = {p.code: p.name for p in POLYHEDRA.values() if p.code}
# the one alias table, case-insensitive; no name is 3 characters, so names and codes never collide
_ALIASES = {**{n.lower(): n for n in POLYHEDRA}, **{c.lower(): n for c, n in _BY_CODE.items()}}


def resolve_geometry(geometry):
    """Return the `POLYHEDRA` name a name or code spells (any case), else `geometry` for the caller to reject."""
    return _ALIASES.get(geometry.strip().lower(), geometry) if isinstance(geometry, str) else geometry


def record(geometry):
    """Return the `Polyhedron` for a name or 3-letter code, or ``None`` when no record has that name."""
    return POLYHEDRA.get(resolve_geometry(geometry)) if isinstance(geometry, str) else None


def describe(geometry):
    """Return a log-ready identity such as ``'square_planar (SPL, CN 4)'``."""
    p = record(geometry)
    return f"{p.name} ({p.code}, CN {p.cn})" if p else str(geometry)


def vertex_dirs(geometry):
    """Return `geometry`'s vertex unit-direction template, or ``None`` if there is no such record."""
    return getattr(record(geometry), "vertex_dirs", None)


def geometries_for_cn(n):
    """Return the supported polyhedra for coordination number `n`, best-default first (by `default_rank`)."""
    return sorted((p for p in POLYHEDRA.values() if p.cn == n), key=lambda p: p.default_rank)


def _greedy_orders(dd, v_ideal, rotations):
    """Assign every vertex to the donor pointing most nearly at it, under each of `rotations`, one order each.

    Each round takes, per rotation, the best pair whose donor and vertex are both still free.
    """
    n = len(v_ideal)
    fits = ((dd @ rotations) @ v_ideal.T).reshape(len(rotations), -1)  # [rotation, donor * n + vertex]
    rank = np.empty_like(fits, dtype=np.intp)
    np.put_along_axis(rank, np.argsort(fits, axis=1), np.arange(fits.shape[1]), axis=1)  # ties as argsort orders them
    orders = np.zeros((len(fits), n), dtype=np.intp)
    rows = np.arange(len(fits))
    for _ in range(n):
        flat = rank.argmax(axis=1)
        donor, vertex = np.divmod(flat, n)
        orders[rows, vertex] = donor
        free = rank.reshape(len(fits), n, n)
        free[rows, donor, :] = -1
        free[rows, :, vertex] = -1
    return [tuple(int(d) for d in order) for order in orders]


def _best_rotations(dd, v_ideal, donors, vertices):
    """Return the orthogonal maps, reflections allowed, that best align each row of `donors` onto the vertices."""
    u, _s, vt = np.linalg.svd(np.transpose(dd[np.asarray(donors)], (0, 2, 1)) @ v_ideal[list(vertices)])
    return u @ vt


def _scores(dd, v_ideal, orders):
    """Return each order's alignment score, the singular-value sum of its donor-vertex correlation.

    Reflections are allowed: shape and hand are separate, and forbidding them can seat donors of a real complex in trans
    slots only ~90 deg apart.
    """
    correlations = np.transpose(dd[np.asarray(orders)], (0, 2, 1)) @ v_ideal
    return np.linalg.svd(correlations, compute_uv=False).sum(axis=1)


def seat_by_alignment(dd, v_ideal):
    """Seat donors on ideal vertices; return ``order`` from vertex to donor-list index.

    Every proper-rotation orbit is searched while that pool is small (`_seating_is_exhaustive`); above it one
    well-conditioned basis is aligned and refined, an approximate seat rather than a proof. Ties keep the first order.
    """
    n = len(v_ideal)
    dirs = tuple(map(tuple, v_ideal))
    if _seating_is_exhaustive(dirs):
        orders = _seating_permutations(dirs)
        return list(orders[int(np.argmax(_scores(dd, v_ideal, orders)))])

    width = min(3, n)
    rank = np.linalg.matrix_rank(v_ideal)
    anchor = max(
        itertools.combinations(range(n), width),
        key=lambda vertices: float(np.prod(np.linalg.svd(v_ideal[list(vertices)], compute_uv=False)[:rank])),
    )
    rotations = _best_rotations(dd, v_ideal, list(itertools.permutations(range(n), width)), anchor)
    seeds = list(set(_greedy_orders(dd, v_ideal, rotations)))  # a tie goes to the first seed in the set's order
    best = seeds[int(np.argmax(_scores(dd, v_ideal, seeds)))]
    for _ in range(3):  # refine: re-align on the winner, re-assign, until it stops moving
        (nxt,) = _greedy_orders(dd, v_ideal, _best_rotations(dd, v_ideal, [best], range(n)))
        moved, kept = _scores(dd, v_ideal, [nxt, best])
        if moved <= kept:
            break
        best = nxt
    return list(best)


LAMBDA, DELTA = "lambda", "delta"  # the two metal-centre hands: typeable words, not glyphs
HAND_GLYPHS = {DELTA: "Δ", LAMBDA: "Λ"}  # each hand as a species row writes it
_TAGS = {**{h: h for h in HAND_GLYPHS}, **{g.lower(): h for h, g in HAND_GLYPHS.items()}, "achiral": "", "-": ""}


def chirality_tag(chirality):
    """Return the word-form hand, accepting Δ/Λ glyphs and ``achiral``."""
    if not isinstance(chirality, str):
        return chirality
    return _TAGS.get(chirality.strip().lower(), chirality)


@lru_cache(maxsize=None)
def point_group(dirs):
    """Return the ``(proper, improper)`` vertex permutations the template's isometries give, cached per template."""
    t = np.array(dirs, float)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    gram = t @ t.T
    signatures = [np.sort(row) for row in gram]
    candidates = [
        [j for j, signature in enumerate(signatures) if np.allclose(signature, signatures[i], rtol=0, atol=_SYM_TOL)]
        for i in range(len(t))
    ]
    automorphisms = []
    order = [-1] * len(t)
    used = set()

    def extend(i):
        if i == len(t):
            automorphisms.append(tuple(order))
            return
        for j in candidates[i]:
            if j in used or any(abs(gram[i, k] - gram[j, order[k]]) > _SYM_TOL for k in range(i)):
                continue
            order[i] = j
            used.add(j)
            extend(i + 1)
            used.remove(j)

    extend(0)
    proper, improper = set(), set()
    rank = np.linalg.matrix_rank(t, tol=_SYM_TOL)
    for permutation in automorphisms:
        permuted = t[list(permutation)]
        u, _s, vt = np.linalg.svd(permuted.T @ t)
        transform = u @ vt
        if np.sum((permuted @ transform - t) ** 2) >= _SYM_TOL:
            continue
        if rank < t.shape[1]:
            proper.add(permutation)
            improper.add(permutation)
        elif np.linalg.det(transform) > 0:
            proper.add(permutation)
        else:
            improper.add(permutation)
    return frozenset(proper), frozenset(improper)


@lru_cache(maxsize=None)
def seating_frames(dirs, occupied):
    """Return the vertex permutations that relabel a seating of `occupied` vertices without moving an atom.

    The proper rotations always do. A vacancy has no atom, so once the occupied vertices are coplanar with the metal the
    reflection through that plane moves no atom and the improper permutations join them: such a seating has no hand. A
    planar template's improper permutations are already rotations, so it returns `rot`, whose order breaks ties.
    """
    rot, refl = point_group(dirs)
    rays = np.array(dirs, float)[list(occupied)]
    if refl <= rot or (len(rays) >= 3 and np.linalg.matrix_rank(rays, tol=_SYM_TOL) == 3):  # noqa: PLR2004
        return rot
    return rot | refl


def relabels(dirs, occupied, moved, handed):
    """Return whether the vertex map `moved` relabels a seating of `occupied` vertices without changing it.

    A `handed` seating relabels by its `seating_frames`, an unhanded one by every isometry of the template.
    """
    frames = seating_frames(dirs, occupied) if handed else frozenset().union(*point_group(dirs))
    return any(all(q[vertex] == image for vertex, image in moved.items()) for q in frames)


@lru_cache(maxsize=None)
def _polygon_faces(poly):
    """Return the polyhedron's convex-hull faces of four or more vertices, as vertex-index arrays.

    A face is every vertex on a plane that leaves the other vertices and the metal on one side. Triangles are always
    planar and are left out, and a planar polyhedron has no hull faces.
    """
    if poly.planar:
        return ()
    t = _unit(poly.vertex_dirs)
    pts = np.vstack([t, np.zeros(3)])
    faces = set()
    for i, j, k in itertools.combinations(range(len(t)), 3):
        normal = np.cross(t[j] - t[i], t[k] - t[i])
        if np.linalg.norm(normal) < _SYM_TOL:
            continue
        side = (pts - t[i]) @ (normal / np.linalg.norm(normal))
        if np.all(side <= _SYM_TOL) or np.all(side >= -_SYM_TOL):
            face = tuple(int(v) for v in np.flatnonzero(np.abs(side[: len(t)]) <= _SYM_TOL))
            if len(face) > 3:  # noqa: PLR2004 - a triangle is always planar
                faces.add(face)
    return tuple(np.array(face) for face in sorted(faces))


def _proper_orbit_permutations(dirs):
    """Yield one vertex assignment per proper-rotation orbit."""
    rotations = point_group(dirs)[0]
    for order in itertools.permutations(range(len(dirs))):
        if order == min(tuple(order[q[v]] for v in range(len(dirs))) for q in rotations):
            yield order


@lru_cache(maxsize=len(POLYHEDRA))
def _seating_permutations(dirs):
    """Return one vertex assignment per proper-rotation orbit of `dirs`, cached; only for an exhaustive seating.

    Isomer enumeration, which is unbounded, streams through `_proper_orbit_permutations` uncached instead.
    """
    return tuple(_proper_orbit_permutations(dirs))


def isomer_permutations(geometry):
    """Yield one candidate per proper-rotation orbit from the polyhedron vertices.

    Proper rotations identify the same arrangement; reflections remain separate so enantiomers survive.
    """
    polyhedron = record(geometry)
    if polyhedron is None:
        return
    yield from _proper_orbit_permutations(tuple(map(tuple, polyhedron.vertex_dirs)))


def mirror_seating(dirs, order):
    """Return the vertex-ordered `order` seated as its mirror image, by any one improper symmetry of `dirs`."""
    q = min(point_group(tuple(map(tuple, dirs)))[1])
    return [order[q[v]] for v in range(len(order))]


def seat_properly(dirs_obs, dirs, order):
    """Re-seat `order` so the observed sphere is reached by rotation, not reflection; the score is unchanged."""
    u, _s, vt = np.linalg.svd(np.asarray(dirs_obs)[list(order)].T @ np.asarray(dirs))  # the best fit's orthogonal map
    return list(order) if np.linalg.det(u @ vt) >= 0 else mirror_seating(dirs, order)


def relaxed_shell(poly, bites):
    """Return unit rays for `poly` with each bite pair moved to its target angle, or None.

    Only the bite rays move on purpose; every other vertex shifts just enough to keep the hull faces planar (folding a
    face is a named path: a square pyramid's base gives the Berry pseudorotation, a trigonal prism's rectangles the
    Bailar twist). Each round's step is damped, since a vertex shared by two bite pairs on a closed cycle otherwise
    oscillates. None when a bite pair is antipodal, 500 rounds do not converge, or the result's oriented type no longer
    matches the template, which is another geometry rather than a distortion. Cached; the array is read-only.
    """
    key = tuple(sorted((tuple(sorted(pair)), float(target)) for pair, target in bites.items()))
    return _relaxed_shell_cached(poly, key)


def _cross3(a, b):
    """Return the cross product of two (N, 3) arrays, bit-identical to `np.cross(a, b)`.

    It is the exact formula `np.cross` reduces to at this fixed shape, without its generic axis handling.
    """
    out = np.empty_like(a)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


@lru_cache(maxsize=FLOAT_CACHE)
def _relaxed_shell_cached(poly, bites_key):
    template = _unit(poly.vertex_dirs)
    r = template.copy()
    if not bites_key:  # no bite pair to relax: the ideal template is already the (trivially converged) shell
        r.setflags(write=False)
        return r
    i_idx = np.array([pair[0] for pair, _target in bites_key])
    j_idx = np.array([pair[1] for pair, _target in bites_key])
    targets = np.array([target for _pair, target in bites_key], float)
    faces = _polygon_faces(poly)
    for _ in range(500):
        for face in faces:  # a polygon face folding along a diagonal is the Berry or Bailar path
            pts = r[face]
            centre = pts.mean(axis=0)
            normal = np.linalg.svd(pts - centre)[2][2]
            pts = pts - np.outer((pts - centre) @ normal, normal)
            r[face] = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        ri, rj = r[i_idx], r[j_idx]
        theta = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", ri, rj), -1.0, 1.0)))
        axis = _cross3(ri, rj)
        norm = np.linalg.norm(axis, axis=1)
        if np.any(norm < _AXIS_EPS):
            return None
        if float(np.max(np.abs(theta - targets))) < FIX_ANGLE_TOL:
            r.setflags(write=False)
            before, after = _oriented_signs(template, poly), _oriented_signs(r, poly)
            kept = all(abs(v) < _SYM_TOL or (after[k] * v > 0 and abs(after[k]) > _SYM_TOL) for k, v in before.items())
            return r if kept else None  # an oriented type the template states, kept with its sign, or another shape
        half = (_SHELL_DAMPING * np.radians(theta - targets) / 2.0)[:, None] * (axis / norm[:, None])
        move = np.zeros_like(r)
        np.add.at(move, i_idx, half)
        np.add.at(move, j_idx, -half)
        phi = np.linalg.norm(move, axis=1)
        moving = phi > 0.0
        n, v, p = move[moving] / phi[moving, None], r[moving], phi[moving, None]
        dot = np.einsum("ij,ij->i", n, v)[:, None]
        r[moving] = v * np.cos(p) + _cross3(n, v) * np.sin(p) + n * dot * (1.0 - np.cos(p))
    return None  # did not converge within 500 rounds


def _oriented_signs(r, poly):
    """Return each triple's signed orientation, or each pair's in-plane sense for a planar `poly`.

    Triples holding a trans pair are dropped: a trans pair bends either way about its axis without changing the
    arrangement.
    """
    if poly.planar:
        normal = np.cross(r[0], r[1])
        if np.linalg.norm(normal) < _SYM_TOL:
            normal = np.cross(r[0], r[2])
        normal = normal / np.linalg.norm(normal)
        return {
            pair: float(np.cross(r[pair[0]], r[pair[1]]) @ normal) for pair in itertools.combinations(range(len(r)), 2)
        }
    return {
        triple: float(np.linalg.det(r[list(triple)]))
        for triple in itertools.combinations(range(len(r)), 3)
        if not any(pair in poly.trans_pairs for pair in itertools.combinations(triple, 2))
    }


class SlotForm(NamedTuple):
    """Hold one arrangement in one frame as comparable data: its seated site keys and slot-labelled links."""

    seats: tuple  # (slot, site key) per occupied vertex
    links: tuple  # ((slot, slot), graph distance) per same-ligand pair


def slot_form(q, keys, links=None):
    """Return the `SlotForm` of the arrangement with site `keys` and labelled `links` in the frame `q`."""
    return SlotForm(
        seats=tuple(sorted((q[v], keys[v]) for v in range(len(q)) if keys[v] is not None)),
        links=tuple(sorted((tuple(sorted(q[v] for v in edge)), label) for edge, label in (links or {}).items())),
    )


def least_frames(dirs, keys, links=None):
    """Return each ``slots[vertex]`` of the seating's frames whose `SlotForm` is least.

    `keys` identifies each site (None when vacant) and `links` labels same-ligand pairs by graph distance; neither may
    depend on atom order. Every frame seats the same arrangement; the first is the canonical one, and a writer folding
    tied sites must keep its link-preserving donor-to-slot pairing.
    """
    frames = list(seating_frames(tuple(map(tuple, dirs)), tuple(v for v, key in enumerate(keys) if key is not None)))
    forms = [slot_form(q, keys, links) for q in frames]
    least = min(forms)
    return [list(q) for q, form in zip(frames, forms, strict=True) if form == least]


_SLOT_NOTE = re.compile(r"^s(\d+)([+\-cPM]?)$")  # a donor's canonical slot and optional haptic face token
SLOT_BOND_PROP = "_rxSlot"  # parsed bridge assignment; bond-local so atom renumbering cannot swap two centres


def slot_note(slot, winding=""):
    """Render a canonical donor slot as ``s<n>`` with an optional haptic face token."""
    return f"s{slot}{winding}"


def read_slot_notes(note):
    """Parse one ``(slot, winding)`` per adjacent metal from a ``;``-separated donor note; None for another note."""
    slots = []
    for part in note.split(";"):
        if (m := _SLOT_NOTE.match(part)) is None:
            return None
        slots.append((int(m.group(1)), m.group(2)))
    return slots


def handedness(dirs, order, donor_class, chelate_links):
    """Return 'delta', 'lambda', or '' for an achiral or incomplete centre.

    Donor symmetry classes and chelate links decorate the seated template; a reflection that keeps both makes it
    achiral, otherwise the canonical frame's parity gives the hand. A vacancy decorates its vertex as a class of its
    own.
    """
    n = len(dirs)
    if len(order) != n:
        return ""
    key = tuple(map(tuple, dirs))
    rot, refl = point_group(key)
    if seating_frames(key, tuple(v for v in range(n) if order[v] is not VACANT)) & refl:
        return ""
    label = [() if order[v] is VACANT else donor_class[order[v]] for v in range(n)]
    forms = {q: slot_form(q, label, chelate_links) for q in rot | refl}
    base = forms[tuple(range(n))]
    if any(forms[q] == base for q in refl):  # a mirror fixes labels and bites -> no handedness
        return ""
    q_star = min(forms, key=lambda q: forms[q])  # the canonicalising frame; its parity is the sign
    return DELTA if q_star in rot else LAMBDA


def _seating_is_exhaustive(dirs):
    """Return whether seating donors on the template `dirs` searches every proper-rotation orbit: the one judge."""
    return math.factorial(len(dirs)) // len(point_group(dirs)[0]) <= _MAX_SEATING_ORBITS


class Fit(NamedTuple):
    """Hold one sphere's fit to one template: the per-vertex RMS after seating, and whether that seating is exact."""

    residual: float
    exact: bool


def fit_residual(dirs_obs, dirs):
    """Return the `Fit` of the observed unit rays `dirs_obs` to the template directions `dirs`.

    The residual is the per-vertex RMS after seating: unlike an angle-spectrum RMS it keeps one scale across
    coordination numbers and keeps vertex correspondence. It is exact where every proper-rotation orbit is searched, and
    an upper bound above that, which ranks shapes but never proves identity.
    """
    ideal = _unit(dirs)
    seated = dirs_obs[seat_by_alignment(dirs_obs, ideal)]
    u, _s, vt = np.linalg.svd(seated.T @ ideal)
    residual = float(np.sqrt(np.mean(np.sum((seated @ (u @ vt) - ideal) ** 2, axis=1))))
    return Fit(residual=residual, exact=_seating_is_exhaustive(tuple(map(tuple, ideal))))
