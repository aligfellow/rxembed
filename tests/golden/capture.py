"""Capture the embed kernel's intermediate state, phase by phase, without changing its behaviour.

The DG build is a driver walking `mechanisms.MECHANISM_ORDER` through WINDOW -> RELIEVE -> COMMIT -> POST -> SMOOTH,
so the phase boundaries are the mechanism methods themselves. Wrapping them snapshots the matrix at each
boundary with no kernel edit, and each phase's contribution is a DIFFERENCE of two arrays — which is what
makes an angle regression distinguishable from a coplanar or a smoothing one.

    GetMoleculeBoundsMatrix        -> `rdkit`      RDKit's knowledge-derived baseline
    WINDOW  (Distance/Angle/Plane) -> `pairs`      the assembled windows, per-key and addressable
    RELIEVE (Floor.dg_relief)      -> `relief`     lowers M...X floors (delta)
    RELIEVE (Haptic.dg_relief)     -> `centroid`   seats a haptic dummy in its ring (delta)
    COMMIT                         -> `commit`     (delta)
    POST    (Coplanar.dg_post)     -> `coplanar`   the 1,4 dihedral bound (delta)
    SMOOTH  (_smooth)              -> `presmooth` / `final`, plus the settled tolerance

A conditional phase is recorded only when its FIELD is non-empty. The driver now calls every mechanism
unconditionally (each is a no-op on an empty field), but which phases *contribute* is part of the signature:
a refactor that stops firing the coplanar phase must drop `coplanar` from the record, not silently zero it.
"""

from __future__ import annotations

import contextlib
import itertools

import numpy as np

from rxembed.rdkit_embed.constraints import mechanisms as _mech
from rxembed.rdkit_embed.embed import bounds as _bounds


@contextlib.contextmanager
def capture():
    """Record every `_bounds` phase for each call made inside the block; yields the list of records."""
    records: list[dict] = []
    floor, haptic, coplanar = _mech.Floor, _mech.Haptic, _mech.Coplanar
    real = {
        "bounds": _bounds._bounds,
        "smooth": _bounds._smooth,
        "relief": floor.dg_relief,
        "centroid": haptic.dg_relief,
        "post": coplanar.dg_post,
    }
    cur: dict = {}

    def _bounds_wrapper(mol, cons):
        nonlocal cur
        cur = {}
        bm, tol = real["bounds"](mol, cons)
        cur["final"] = np.array(bm, dtype=np.float64)
        cur["tol"] = tol  # straight from the engine — no second copy of the escalator
        cur["cons"] = _cons_signature(cons)
        records.append(cur)
        return bm, tol

    def _relief(self, cons, ctx):
        # The FIRST hook to fire: the WINDOW phase writes only `ctx.pairs`, so `bm` is still RDKit's baseline.
        cur["rdkit"] = np.array(ctx.bm, dtype=np.float64)
        cur["pairs"] = {f"{min(k)},{max(k)}": [float(v[0]), float(v[1])] for k, v in ctx.pairs.items()}
        real["relief"](self, cons, ctx)
        cur["relief"] = np.array(ctx.bm, dtype=np.float64)

    def _centroid(self, cons, ctx):
        real["centroid"](self, cons, ctx)
        if cons.phantoms:  # only a haptic face contributes a centroid relief
            cur["centroid"] = np.array(ctx.bm, dtype=np.float64)

    def _post(self, cons, ctx):
        if cons.coplanar:
            cur["commit"] = np.array(ctx.bm, dtype=np.float64)  # post-COMMIT, pre-coplanar
        real["post"](self, cons, ctx)
        if cons.coplanar:
            cur["coplanar"] = np.array(ctx.bm, dtype=np.float64)

    def _smooth(bm, max_tol=0.4):
        cur["presmooth"] = np.array(bm, dtype=np.float64)
        return real["smooth"](bm, max_tol)

    _bounds._bounds, _bounds._smooth = _bounds_wrapper, _smooth
    floor.dg_relief, haptic.dg_relief, coplanar.dg_post = _relief, _centroid, _post
    try:
        yield records
    finally:
        _bounds._bounds, _bounds._smooth = real["bounds"], real["smooth"]
        floor.dg_relief, haptic.dg_relief, coplanar.dg_post = real["relief"], real["centroid"], real["post"]


def _cons_signature(cons):
    """Canonical, order-independent, JSON-safe view of every Constraints field."""
    from dataclasses import fields

    def norm(v):
        if isinstance(v, dict):
            return sorted((repr(k), repr(val)) for k, val in v.items())
        if isinstance(v, (set, frozenset)):
            return sorted(repr(x) for x in v)
        if isinstance(v, (list, tuple)):
            return [norm(x) if isinstance(x, (dict, set, frozenset)) else repr(x) for x in v]
        return repr(v)

    return {f.name: norm(getattr(cons, f.name)) for f in fields(type(cons))}


# Snapshots in the order the driver produces them. `centroid`, `commit` and `coplanar` are conditional, so
# deltas are taken between consecutive PRESENT snapshots.
PHASES = ("rdkit", "relief", "centroid", "commit", "coplanar", "presmooth", "final")


def deltas(rec):
    """Per-phase contributions to the bounds matrix, as {"a->b": sparse {(i,j): [before, after]}}.

    The final `presmooth->final` step is the exception: it is `DoTriangleSmoothing`, C++ float arithmetic that
    is bit-stable on one build but not guaranteed across RDKit versions or BLAS, and it touches thousands of
    cells on a large system. Storing its values would both contradict the tolerance policy and dwarf every
    diagnostic phase. Only WHICH cells it moved is kept; the values are covered by the tolerance check on the
    final matrix itself.
    """
    present = [p for p in PHASES if p in rec]
    out = {}
    for earlier, later in itertools.pairwise(present):
        a, b = rec[earlier], rec[later]
        key = f"{earlier}->{later}"
        if a.shape != b.shape:
            out[key] = "SHAPE-CHANGED"
            continue
        idx = np.argwhere(a != b)
        if later == "final":
            out[key] = {"cells_moved": [[int(i), int(j)] for i, j in idx]}
        else:
            out[key] = {f"{i},{j}": [float(a[i, j]), float(b[i, j])] for i, j in idx}
    return out
