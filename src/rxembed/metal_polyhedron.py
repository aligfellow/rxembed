"""Coordination templates and their symmetry operations.

Arrangements are folded over proper template rotations; an improper frame identifies the opposite hand when
the decorated sphere is chiral. This makes canonical slots and metal handedness independent of input atom
order, unlike RDKit's metal stereo permutation. Template and algorithm references are listed in the README.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _vertex_angle(u, v):
    u, v = np.array(u), np.array(v)
    return round(float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))))


_FLAT_EPS = 1e-6  # smallest singular value below which a point set is one plane
_IMPROPER_VERTICES = 3  # an improper states exactly three vertices; a CN4 record would be held 3-of-4
CHELATE_SPAN_ANGLE = 135  # a same-ligand donor pair this wide needs a trans-spanning backbone


def _improper(p1, p2, p3, p4):
    """Signed improper dihedral 1-2-3-4 in degrees, the convention `rdMolTransforms.GetDihedralDeg` uses."""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(m @ n2, n1 @ n2)))


@dataclass(frozen=True)
class Polyhedron:
    """Describe one coordination geometry and its hand-authored constraints.

    `vertex_dirs` defines slots. `angles` may be the measured minimal subset; ``None`` derives every pair.
    `planar` is a perception constraint that separates the CN3 plane from the pyramid without a fitted angle
    boundary.
    """

    name: str
    default_rank: int  # preference within the CN (0 == the default polyhedron)
    vertex_dirs: tuple
    angles: tuple | None  # hand-authored subset; None -> derive from vertex_dirs
    planar: bool = False  # metal and vertices coplanar by definition, and the `classify_geometry` filter
    geometric_isomerism: bool = True  # False -> no cis/trans label; distinct metal hands may still exist
    code: str = ""  # 3-letter spelling, accepted wherever a geometry name is (`resolve_geometry`).
    site_groups: tuple = ()  # optional conventional display groups: ((name, vertex indices), ...)
    # Not the IUPAC polyhedral symbol, which differs for most rows: TPY here is the CN3 pyramid, and other
    # conventions spell a CN4 one the same way, so anything translating must key on the coordination number.

    @property
    def cn(self):
        """Return the coordination number: the vertex count (== len(vertex_dirs)), derived not stored."""
        return len(self.vertex_dirs)

    @property
    def umbrella_improper(self):
        """Return the ideal flat-based-pyramid improper, or ``None``.

        The scale-free improper states pyramidalisation without a bond-length-dependent D-M-D window. It is
        exact only for the ideal record; a distorted sphere still needs measurement.
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


# The supported polyhedra, one record each. Insertion order is the classify_geometry tie-break;
# `geometries_for_cn` groups by CN for the default and the "did you mean".
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        # One donor defines a distance but no relative angle. Keep it distinct from CN2 linear so inference
        # does not invent an open coordination site; requesting `linear` still states that pocket explicitly.
        Polyhedron(
            name="monocoordinate",
            code="MCO",
            default_rank=0,
            vertex_dirs=((0, 0, 1),),
            angles=(),
            planar=True,
            geometric_isomerism=False,
        ),
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
            planar=True,
            site_groups=(("axial", (0, 2)), ("equatorial", (1,))),
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
            site_groups=(("axial", (0, 1)), ("equatorial", (2, 3))),
        ),
        Polyhedron(
            name="trigonal_bipyramidal",
            code="TBP",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)),
            angles=((0, 1, 180), (1, 2, 90), (0, 3, 90), (2, 3, 120), (3, 4, 120), (2, 4, 120)),
            site_groups=(("axial", (0, 1)), ("equatorial", (2, 3, 4))),
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
            site_groups=(("apical", (0,)), ("basal", (1, 2, 3, 4))),
        ),
        Polyhedron(
            name="octahedral",
            code="OCT",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),  # trans PAIRS:
            #   0,1 / 2,3 / 4,5, unlike square_planar, where the trans partner of 0 is 2
            angles=((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
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
            angles=None,
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
    """Resolve a name or code; pass unknown values through for the caller to reject."""
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


def _fit_trace(h):
    """Return the best alignment from cross-covariance `h`, allowing reflections.

    Shape and handedness are separate. Forbidding reflection seated three DUGVUX donors in trans slots only
    93° apart.
    """
    return float(np.linalg.svd(h, compute_uv=False).sum())


def _seat_by_alignment(dd, v_ideal, rounds=3):
    """Seat donors on ideal vertices; return ``order`` from vertex to donor-list index.

    Ordered triples seed the rotation, avoiding 40,320 CN8 permutations. Completing and refining those seeds
    matched full search on measured CN7/CN8 structures; one greedy pass scored 5.63 instead of 7.90 on a
    square antiprism.
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
_TAGS = {LAMBDA: LAMBDA, DELTA: DELTA, "λ": LAMBDA, "δ": DELTA, "achiral": "", "-": ""}


def chirality_tag(chirality):
    """Return the word-form hand, accepting Δ/Λ glyphs and ``achiral``."""
    if not isinstance(chirality, str):
        return chirality
    return _TAGS.get(chirality.strip().lower(), chirality)  # "Δ".lower() is "δ", hence both spellings


@lru_cache(maxsize=None)
def point_group(dirs):
    """Return ``(proper, improper)`` vertex permutations realised by template isometries.

    Planar or degenerate templates may realise the same permutation with both parities. Results are cached by
    the hashable direction template.
    """
    t = np.array(dirs, float)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    n = len(t)
    perms = np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(n))), dtype=int).reshape(-1, n)
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


def rotation_group(geometry):
    """Return proper vertex rotations, or ``None`` when the geometry has no template.

    Folding over reflections too would erase lambda/delta.
    """
    dirs = vertex_dirs(geometry)
    return None if dirs is None else point_group(tuple(map(tuple, dirs)))[0]


@lru_cache(maxsize=None)
def isomer_permutations(geometry):
    """Derive one candidate per proper-rotation orbit from the polyhedron vertices.

    Proper rotations identify the same arrangement; reflections remain separate so enantiomers survive.
    """
    polyhedron = record(geometry)
    if polyhedron is None:
        return None
    rotations = rotation_group(polyhedron.name)
    seen, out = set(), []
    for order in itertools.permutations(range(polyhedron.cn)):
        if order in seen:
            continue
        out.append(order)
        seen.update(tuple(order[q[v]] for v in range(polyhedron.cn)) for q in rotations)
    return tuple(out)


def seat_properly(dirs_obs, dirs, order):
    """Re-seat `order` so the observed sphere is reached by rotation, not reflection.

    Reflection leaves the alignment score unchanged, so one improper template symmetry flips the fitted
    parity without another search. This distinguished all 7 chiral centres in the 45-structure corpus.
    Re-seating changed the minimal angle subset on 3 structures; `metal_constraints` handles the chelate case.
    """
    refl = point_group(tuple(map(tuple, dirs)))[1]
    if not refl or orientation_parity(np.asarray(dirs_obs)[list(order)], dirs) >= 0:
        return list(order)
    q = min(refl)  # any one improper element; which one only moves the result within its proper orbit
    return [order[q[v]] for v in range(len(order))]


def orientation_parity(dirs_obs, dirs):
    """Return +1 for a proper best fit of observed directions to an ideal shape, otherwise -1."""
    u, _s, vt = np.linalg.svd(np.asarray(dirs_obs).T @ np.asarray(dirs))
    return 1 if np.linalg.det(u @ vt) >= 0 else -1


def canonical_slots(dirs, keys, bites=frozenset()):
    """Return ``slots[vertex]`` minimised over proper template rotations.

    `keys` identifies each site and `bites` identifies chelate edges. Neither may depend on atom order. Tied
    keys are interchangeable, so renderers must pair sorted sites with sorted slots inside each tied class.
    """
    rot = point_group(tuple(map(tuple, dirs)))[0]
    n = len(dirs)
    if not rot:
        return None
    edges = [tuple(e) for e in bites]

    def form(q):
        seats = tuple(sorted((q[v], keys[v]) for v in range(n) if keys[v] is not None))
        return seats, tuple(sorted(tuple(sorted((q[a], q[b]))) for a, b in edges))

    return list(min(rot, key=form))


_SLOT_NOTE = re.compile(r"^s(\d+)([+-]?)$")  # a donor's canonical slot, with an optional haptic winding sign
SLOT_BOND_PROP = "_rxSlot"  # parsed bridge assignment; bond-local so atom renumbering cannot swap two centres


def slot_note(slot, winding=""):
    """Render a canonical donor slot as ``s<n>`` with an optional winding sign."""
    return f"s{slot}{winding}"


def read_slot_note(note):
    """Parse a donor slot note; return ``None`` for any other note."""
    m = _SLOT_NOTE.match(note)
    return None if m is None else (int(m.group(1)), m.group(2))


def read_slot_notes(note):
    """Parse one slot per adjacent metal from a semicolon-separated donor note."""
    values = [read_slot_note(part) for part in note.split(";")]
    return None if any(value is None for value in values) else values


def handedness(dirs, order, donor_class, chelate_edges=frozenset()):
    """Return ``'delta'``, ``'lambda'``, or ``''`` for an achiral or incomplete centre.

    Donor symmetry classes and chelate edges decorate the seated template. A reflection that preserves both
    makes it achiral; otherwise the canonical frame's parity gives the hand. A vacant vertex cannot fix parity.
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


def ordered_fit_residual(dirs_obs, dirs_ideal):
    """Return the RMS after best orthogonal alignment with correspondence fixed."""
    u, _s, vt = np.linalg.svd(dirs_obs.T @ dirs_ideal)
    return float(np.sqrt(np.mean(np.sum((dirs_obs @ (u @ vt) - dirs_ideal) ** 2, axis=1))))


def fit_residual(dirs_obs, record):
    """Return the per-vertex RMS after the best orthogonal seating on `record`.

    Unlike an angle-spectrum RMS, this stays on one scale across coordination numbers and retains vertex
    correspondence. Measured residuals span 0.066-0.135 for CN3-CN8; the spectrum misclassified cis-dioxo Mo
    BESJUE as trigonal prismatic.
    """
    ideal = np.array(record.vertex_dirs, float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    order = _seat_by_alignment(dirs_obs, ideal)
    return ordered_fit_residual(dirs_obs[order], ideal)
