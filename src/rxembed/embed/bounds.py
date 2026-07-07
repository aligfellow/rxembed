"""Distance-geometry embedding, biased by a constraints bounds matrix.

Upstream: RDKit ETKDGv3 + the bounds-matrix edit kernel shared by racerts and
nci_embed. Pure RDKit — this is the piece we'll most likely diverge from racerts on.
"""

from __future__ import annotations

import math

from rdkit import DistanceGeometry
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from rxembed.log import logger

_SMOOTH_LOOSE = 0.1  # Å: smoothing beyond this means the constraints are genuinely over-tight — worth reporting


def _smooth(bm, max_tol=0.4):
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed")
        bm[:] = back
    if tol > _SMOOTH_LOOSE:  # the bounds were metric-inconsistent — loosened to embed at all (the relax may
        logger.info(  # still land a good geometry, but the arrangement is tight enough to be worth flagging)
            "embed: bounds smoothing loosened to %.2f A — these distance/angle constraints are over-tight for "
            "this arrangement (near-infeasible); check the embedded geometry",
            tol,
        )


def _mid(bm, i, j):  # current mean of (upper, lower) bound for a pair
    a, b = (i, j) if i < j else (j, i)
    return 0.5 * (bm[a][b] + bm[b][a])


def _law_of_cosines(dij, djk, theta_deg):
    return math.sqrt(dij**2 + djk**2 - 2 * dij * djk * math.cos(math.radians(theta_deg)))


def _bounds(mol, cons):
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)  # RDKit's knowledge-derived geometric bounds; we
    pairs = dict(cons.distances)  # OVERRIDE only the constrained pairs, keeping the rest

    def leg(i, j):  # angle-leg length: prefer the explicit constraint over the (maybe bondless) default
        v = pairs.get((min(i, j), max(i, j)))
        return 0.5 * (v[0] + v[1]) if v else _mid(bm, i, j)

    for (i, j, k), (lo, hi) in cons.angles.items():  # angle -> i..k distance (law of cosines)
        dij, djk = leg(i, j), leg(j, k)
        pairs.setdefault((i, k), (_law_of_cosines(dij, djk, lo), _law_of_cosines(dij, djk, hi)))
    for ring_a, ring_b, sep in cons.planes:  # parallel stack: cross-ring d = sqrt(sep^2 + in-plane^2)
        for u, au in enumerate(ring_a):
            for v, bv in enumerate(ring_b):
                offset = 0.0 if u == v else _mid(bm, au, ring_a[v])
                d = math.hypot(sep, offset)
                pairs.setdefault((min(au, bv), max(au, bv)), (d - 0.3, d + 0.3))  # a user constrain= on this
                # cross-ring pair wins over the stack heuristic (setdefault, like the angle path above)
    for (i, j), (lo, hi) in pairs.items():
        a, b = (i, j) if i < j else (j, i)
        bm[a][b], bm[b][a] = hi, lo
    _smooth(bm)
    return bm


def embed(mol, cons, n, seed=0xF00D, prune_rms=0.1, knowledge=True):
    """Embed ``n`` conformers via ETKDGv3 on the (edited) bounds matrix; return the conformer ids."""
    p = rdDistGeom.ETKDGv3()
    if not knowledge:  # plain distance geometry — no experimental-torsion/basic-knowledge seeding
        p.useExpTorsionAnglePrefs = False
        p.useBasicKnowledge = False
    p.randomSeed = seed
    p.numThreads = 0
    p.pruneRmsThresh = prune_rms or -1.0
    if cons.distances or cons.angles or cons.planes:
        p.embedFragmentsSeparately = False
        p.SetBoundsMat(_bounds(mol, cons))  # a custom (edited) bounds matrix
    # Use ETKDG's KNOWLEDGE-based initial coordinates (an eigenvalue start from the bounds matrix, plus the
    # experimental-torsion terms) — they survive a custom bounds matrix, so the periphery is seeded with
    # chemistry. Fall back to random coordinates only if that yields nothing: a tightly constrained core (an
    # organic TS freeze) can make the knowledge-seeded start metric-infeasible, where random coords still embed.
    p.useRandomCoords = False
    ids = list(rdDistGeom.EmbedMultipleConfs(mol, n, p))
    if not ids:
        p.useRandomCoords = True
        ids = list(rdDistGeom.EmbedMultipleConfs(mol, n, p))
    return ids


def n_confs(mol, constrained=False):
    """Bounds-biased ETKDG seed count, scaled by flexibility (openconf-style, not a flat 50).

    For an unconstrained ``.mc()`` run openconf re-seeds and these are replaced. For a **constrained** run
    they are the biased pool openconf searches *around* with the held atoms pose-frozen — and that pose-mode
    is **rotor-only** (no low-mode/ring/global moves), so it under-samples unless given more distinct
    starting points: seed ~1.6x more, with a higher floor/cap.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(250, max(40, 10 * r))
    return min(150, max(24, 6 * r))  # cf. openconf max(20, 3*r); a touch more for biased seeds
