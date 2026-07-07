"""Coordination-polyhedron symmetry + metal-centre chirality, name-agnostically.

The isomer *identity* rxembed selects on is geometric, not a chemistry name (cis/trans/mer/fac are
fragile — a tris-chelate is Λ/Δ, not mer/fac). Two pieces live here, both pure numpy over a geometry's
vertex-direction template (``metal.VERTEX_DIRS``):

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
tris-/bis-chelate Δ/Λ helical descriptor). rxembed applies the *algorithm* to its own ``metal.VERTEX_DIRS``
templates (not OIN's vectors), so the geometry conventions stay rxembed's; only the parity machinery is lifted.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import numpy as np

_SYM_TOL = 1e-6  # residual below which a vertex permutation is an exact template isometry
_PROPER, _IMPROPER = "Δ", "Λ"  # proper-frame / improper-frame parity tags (Delta / Lambda)


@lru_cache(maxsize=None)
def _perms(n):
    return tuple(permutations(range(n)))


@lru_cache(maxsize=None)
def point_group(dirs):
    """Return ``(rotations, reflections)`` — the vertex perms realisable by a proper / improper isometry.

    `dirs` is a hashable tuple of vertex unit directions (a geometry's ``VERTEX_DIRS`` entry). A perm is a
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
