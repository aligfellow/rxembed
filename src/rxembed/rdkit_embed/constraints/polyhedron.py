"""Coordination-polyhedron symmetry + metal-centre chirality, name-agnostically.

The isomer *identity* rxembed selects on is geometric, not a chemistry name (cis/trans/mer/fac are
fragile — a tris-chelate is Λ/Δ, not mer/fac). The polyhedron templates themselves live here (``POLYHEDRA``,
one ``Polyhedron`` record per geometry); the two symmetry pieces below are pure numpy over a template's
vertex-direction list (``POLYHEDRA[g].vertex_dirs``):

- **point-group split** (`point_group`): every vertex permutation is a proper (rotation, det +1) or improper
  (reflection, det -1) isometry of the template — the group the arrangement canonicalises over.
- **handedness** (`handedness`): a metal centre's Λ/Δ chirality as the *parity of the canonicalising frame*
  over that group — achiral iff some improper symmetry fixes the (donor-class + chelate-bite) labelling.

Provenance (all under ``/home/ali/Documents/Codes/``): the point-group-parity **algorithm** is adapted from
the ``tmc_round`` project (`tmc_round/src/tmc_round/polyhedron.py::_point_group` and
`notation.py::_handedness`), which settled on it after finding RDKit's native metal stereo permutation is
*not* order-invariant for equivalent ligands. That in turn traces to **OIN-SMILES** (Open Isomer Notation,
`OIN-SMILES/src/oinsmiles/oin/inline.py` — the ``@``/``@@`` parity coset; `utils/oin_aligner.py::TEMPLATE_SPECS`
— the polyhedron templates + 3-letter codes) and **trex** (Kevlishvili, MIT; `trex/chirality.py` — the
tris-/bis-chelate Δ/Λ helical descriptor). rxembed applies the *algorithm* to its own ``metal.POLYHEDRA``
templates (not OIN's vectors), so the geometry conventions stay rxembed's; only the parity machinery is lifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

_s = math.sqrt(3) / 2  # sin 60° = cos 30°, the y of a 120°-spaced vertex


def _vertex_angle(u, v):
    u, v = np.array(u), np.array(v)
    return round(float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))))


@dataclass(frozen=True)
class Polyhedron:
    """One coordination geometry: its vertex template + the hand-authored constraint data keyed off it.

    `vertex_dirs` (unit directions) defines slot numbering; `angles` and `permutations` index those same
    slots. `angles` is the deliberate MINIMAL spanning subset — octahedral states 6 of 15 pairs, tbp 6 of 10;
    the unconstrained pairs are held by nothing on the bond-less surrogate and drift, but stating all pairs is
    measured-refuted (no crystal-fidelity gain — see plan.md "all-pairs polyhedron angles"). ``angles=None``
    derives every pair from the vectors (CN7/CN8 only). `permutations` is ``None`` when the geometry has no
    canned coordination-isomer list — a load-bearing distinction (linear/trigonal_planar/tetrahedral/CN7/CN8).
    """

    name: str
    default_rank: int  # preference within the CN (0 == the default polyhedron)
    vertex_dirs: tuple
    angles: tuple | None  # hand-authored subset; None -> derive from vertex_dirs
    permutations: tuple | None = None  # None -> no canned isomer list
    planar: bool = False  # metal + vertices coplanar by definition
    geometric_isomerism: bool = True  # False -> one arrangement only, no cis/trans label

    @property
    def cn(self):
        """Return the coordination number — the vertex count (== len(vertex_dirs)), derived not stored."""
        return len(self.vertex_dirs)

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


# The supported polyhedra, one record each (angles/permutations adapted from TMC_embed's angle_dict /
# coord_permutations; CN7/CN8 vectors from OIN oin_aligner.TEMPLATE_SPECS). Insertion order is the
# classify_geometry tie-break; `geometries_for_cn` groups by CN for the default / "did you mean".
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        Polyhedron(
            name="linear",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1)),
            angles=((0, 1, 180),),
            planar=True,
            geometric_isomerism=False,
        ),
        Polyhedron(
            name="trigonal_planar",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-0.5, _s, 0), (-0.5, -_s, 0)),
            angles=((0, 1, 120), (1, 2, 120), (0, 2, 120)),
            planar=True,
            geometric_isomerism=False,
        ),
        Polyhedron(
            name="t_shape",
            default_rank=1,
            vertex_dirs=((0, 0, 1), (1, 0, 0), (0, 0, -1)),  # 0,2 trans; 1 perpendicular
            angles=((0, 1, 90), (1, 2, 90), (0, 2, 180)),
            permutations=((0, 1, 2), (1, 0, 2), (0, 2, 1)),
            planar=True,
        ),
        Polyhedron(
            name="square_planar",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),
            angles=((0, 2, 180), (1, 3, 180), (0, 1, 90)),
            permutations=((0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1)),
            planar=True,
        ),
        Polyhedron(
            name="tetrahedral",
            default_rank=1,
            vertex_dirs=((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
            angles=((0, 1, 109.5), (0, 2, 109.5), (0, 3, 109.5), (1, 2, 109.5), (1, 3, 109.5), (2, 3, 109.5)),
            geometric_isomerism=False,
        ),
        Polyhedron(
            name="seesaw",
            default_rank=2,
            vertex_dirs=((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, _s, 0)),  # 0,1 axial; 2,3 equatorial 120°
            angles=((0, 1, 180), (2, 3, 120), (0, 3, 90), (1, 2, 90)),
            permutations=((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)),
        ),
        Polyhedron(
            name="trigonal_bipyramidal",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, _s, 0), (-0.5, -_s, 0)),
            angles=((0, 1, 180), (1, 2, 90), (0, 3, 90), (2, 3, 120), (3, 4, 120), (2, 4, 120)),
            permutations=_TBP_ISOMERS,
        ),
        # basal donors sit BELOW the metal's equatorial plane (apex-basal ~105°, trans-basal ~150°) — a real
        # pyramid, NOT an octahedron minus a vertex; a flat 90/180 template misclassifies VOacac2 as tbp. Minimal
        # subset kept: all 10 pairs makes it rigid but gains no crystal fidelity (same verdict as all-pairs).
        Polyhedron(
            name="square_pyramidal",
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
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
            angles=((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
            permutations=_OCT_ISOMERS,
        ),
        Polyhedron(
            name="pentagonal_bipyramidal",
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
            name="square_antiprism",
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
    ]
}


def geometries_for_cn(n):
    """Return the supported polyhedra for coordination number `n`, best-default first (by `default_rank`)."""
    return sorted((p for p in POLYHEDRA.values() if p.cn == n), key=lambda p: p.default_rank)


def vertex_dirs(geometry):
    """Return `geometry`'s vertex unit-direction template, or None if unknown."""
    p = POLYHEDRA.get(geometry)
    return p.vertex_dirs if p else None


def isomer_permutations(geometry):
    """Return `geometry`'s canned coordination-isomer vertex orderings, or None (no canned list / unknown)."""
    p = POLYHEDRA.get(geometry)
    return p.permutations if p else None


def is_planar(geometry):
    """Return True if `geometry`'s metal + vertices are coplanar by definition (unknown/None -> False)."""
    p = POLYHEDRA.get(geometry)
    return bool(p and p.planar)


_SYM_TOL = 1e-6  # residual below which a vertex permutation is an exact template isometry
_PROPER, _IMPROPER = "Δ", "Λ"  # proper-frame / improper-frame parity tags (Delta / Lambda)


@lru_cache(maxsize=None)
def _perms(n):
    return tuple(permutations(range(n)))


@lru_cache(maxsize=None)
def point_group(dirs):
    """Return ``(rotations, reflections)`` — the vertex perms realisable by a proper / improper isometry.

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
    """Return a metal centre's chirality tag over template `dirs`: ``'Δ'``, ``'Λ'``, or ``''`` (achiral).

    `order[vertex]` is the donor atom seated at that vertex; `donor_class[donor]` its symmetry class (so
    equivalent donors share a label); `chelate_edges` the set of ``frozenset({vertex_i, vertex_j})`` whose
    two donors belong to one chelating ligand (a tris/bis-chelate's Λ/Δ lives in this bite graph, not the
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
    return _PROPER if q_star in rot else _IMPROPER
