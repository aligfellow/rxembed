"""T2 diagnostic: WHY does the raw seed miss its angle windows?

Two candidate causes, and they imply opposite fixes:

  (a) UNDER-TIGHT BOUNDS — the angle window was never fully written into the bounds matrix, so the
      1-3 distance d_ik the seed realises is INSIDE the matrix bound while the angle is outside its
      window. Fix: tighten the bounds.
  (b) ETKDG WALKS OFF THE MATRIX — the bound is tight but the embed's coordinate refinement
      (experimental torsions, basic knowledge, the final error-function minimise) leaves it, so
      d_ik lands OUTSIDE the matrix bound too. Fix: nothing in the bounds can help; it is a
      downstream-enforcement problem.

For every violated angle in every seed we classify which happened. Also compares knowledge=True vs
knowledge=False seeds — if the ETKDG knowledge terms are what pushes the seed off, dropping them
should cut the violation.

Run: uv run python t2_why_seed_misses.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import metal as _metal
from rxembed.embed import bounds as _bounds
from t2_seed_vs_relax import NCONF, SEED, TMQM, crystal

logging.disable(logging.WARNING)
HERE = os.path.dirname(__file__)
sel = [r["name"] for r in json.load(open(os.path.join(HERE, "t2_selection.json")))]

inside_bm, outside_bm, total_viol = 0, 0, 0
rows = []
for nm in sel:
    path = os.path.join(TMQM, f"{nm}.xyz")
    sym, cry, q = crystal(path)
    try:
        ens = rx.embed(path, charge=q, n=NCONF, seed=SEED)
    except Exception:
        continue
    if ens._metal is None or not ens.cons.angles:
        continue
    mol, cons = ens.mol, ens.cons
    # rebuild the SAME matrix the embed used (same function, same cons) on the same phantom-materialised work mol
    work = _metal.materialise_phantoms(mol, cons.haptic)
    try:
        bm, tol = _bounds._feasible_bounds(work, cons)
    except Exception:
        continue
    ids = list(ens.ids)
    seeds = ids[1:] if np.array_equal(mol.GetConformer(ids[0]).GetPositions(), cry) else ids
    n_in = n_out = n_v = 0
    for cid in seeds:
        p = mol.GetConformer(cid).GetPositions()
        for (i, j, k), (lo, hi) in cons.angles.items():
            a = geo._angle(p[i], p[j], p[k])
            if lo - 1e-6 <= a <= hi + 1e-6:
                continue
            n_v += 1
            d = float(np.linalg.norm(p[i] - p[k]))
            a_, b_ = (i, k) if i < k else (k, i)
            bm_lo, bm_hi = bm[b_][a_], bm[a_][b_]  # bm[lower][upper] convention: bm[a][b]=hi, bm[b][a]=lo
            if bm_lo - 1e-6 <= d <= bm_hi + 1e-6:
                n_in += 1  # (a) matrix satisfied, angle not -> the bound is under-tight
            else:
                n_out += 1  # (b) matrix itself violated -> ETKDG walked off it
    inside_bm += n_in
    outside_bm += n_out
    total_viol += n_v
    rows.append((nm, tol, n_v, n_in, n_out, len(cons.angles) * len(seeds)))

print("WHY THE SEED MISSES ITS ANGLE WINDOWS")
print(
    f"{'name':8s} {'smooth_tol':>10s} {'viol':>6s} {'/checked':>9s} | {'(a) bound under-tight':>22s} {'(b) ETKDG off-matrix':>21s}"
)
for nm, tol, nv, ni, no, nc in rows:
    print(f"{nm:8s} {tol:10.3f} {nv:6d} {nc:9d} | {ni:22d} {no:21d}")
print(f"\nTOTAL violated angle instances: {total_viol}")
if total_viol:
    print(
        f"  (a) 1-3 distance INSIDE the bounds matrix but angle outside its window: {inside_bm} ({100 * inside_bm / total_viol:.0f}%)"
    )
    print(
        f"  (b) 1-3 distance OUTSIDE the bounds matrix too (ETKDG left the matrix):  {outside_bm} ({100 * outside_bm / total_viol:.0f}%)"
    )

print()
print("KNOWLEDGE TERMS: does dropping ETKDG's experimental-torsion / basic-knowledge seeding help?")
print(f"{'name':8s} {'awin_max K=True':>16s} {'K=False':>10s} {'dwin_max K=True':>16s} {'K=False':>10s}")
kt, kf = [], []
for nm in sel:
    path = os.path.join(TMQM, f"{nm}.xyz")
    sym, cry, q = crystal(path)
    out = []
    for know in (True, False):
        try:
            e = rx.embed(path, charge=q, n=NCONF, seed=SEED, knowledge=know)
        except Exception:
            out.append(None)
            continue
        ids = list(e.ids)
        sd = ids[1:] if np.array_equal(e.mol.GetConformer(ids[0]).GetPositions(), cry) else ids
        if not sd:
            out.append(None)
            continue
        av, dv = [], []
        for cid in sd:
            p = e.mol.GetConformer(cid).GetPositions()
            av.append(
                max(
                    [
                        max(0.0, lo - geo._angle(p[i], p[j], p[k]), geo._angle(p[i], p[j], p[k]) - hi)
                        for (i, j, k), (lo, hi) in e.cons.angles.items()
                    ]
                    or [0.0]
                )
            )
            dv.append(
                max(
                    [
                        max(0.0, lo - float(np.linalg.norm(p[i] - p[j])), float(np.linalg.norm(p[i] - p[j])) - hi)
                        for (i, j), (lo, hi) in e.cons.distances.items()
                    ]
                    or [0.0]
                )
            )
        out.append((float(np.median(av)), float(np.median(dv))))
    if out[0] and out[1]:
        kt.append(out[0])
        kf.append(out[1])
        print(f"{nm:8s} {out[0][0]:16.2f} {out[1][0]:10.2f} {out[0][1]:16.3f} {out[1][1]:10.3f}")
if kt:
    print(
        f"\nMEAN over {len(kt)} structures: angle-window max  K=True {np.mean([x[0] for x in kt]):.2f} deg -> K=False {np.mean([x[0] for x in kf]):.2f} deg"
    )
    print(
        f"                              dist-window  max  K=True {np.mean([x[1] for x in kt]):.3f} A -> K=False {np.mean([x[1] for x in kf]):.3f} A"
    )
