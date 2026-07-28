"""Calibrate the fusion threshold on the REAL case-4 pipeline collapse: ratio d(O,O)/Σrcov per collapsed seed."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import GetPeriodicTable

import rxembed as rx
from rxembed import geometry as geo

import kdiag_harness as H

rx.set_verbose("CRITICAL")
geo.over_compression = lambda *a, **k: []  # DISABLE the new fusion check so the reembed loop keeps collapsed seeds
_PT = GetPeriodicTable()
RCOV_OO = _PT.GetRcovalent(8) * 2  # Σrcov for an O-O pair


def oco(pos):
    v1, v2 = pos[2] - pos[3], pos[4] - pos[3]
    a = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    return a, float(np.linalg.norm(pos[2] - pos[4]))


print(f"Σrcov(O,O) = {RCOV_OO:.3f} A  -> floor at ratio 1.0 = {RCOV_OO:.3f}, at 1.2 = {1.2 * RCOV_OO:.3f}")
iso_set = rx.metal(H.CASES["case4"], "square_planar")
rows = []
caught_10 = caught_12 = fusion_flagged = total_collapsed = 0
for k, iso in enumerate(iso_set):
    for s in range(40):
        seed = 0xF00D + s
        ens = rx.embed(iso, n=1, seed=seed).minimize()
        if not ens.ids:
            continue
        cid = ens.ids[0]
        pos = ens.mol.GetConformer(cid).GetPositions()
        ang, d = oco(pos)
        if ang < 90.0:  # a collapsed ester
            total_collapsed += 1
            ratio = d / RCOV_OO
            sphere = sorted({dd for ds in ens.sphere.values() for dd in ds}) or list(iso.donors)
            rep = geo.check(ens.mol, cid, donors=sphere)
            has_fusion = any(v.kind == "fusion" for v in rep.violations)
            fusion_flagged += has_fusion
            caught_10 += d < 1.0 * RCOV_OO
            caught_12 += d < 1.2 * RCOV_OO
            rows.append((k, seed, ang, d, ratio, has_fusion))

print(f"\ncollapsed (O-C-O<90) conformers: {total_collapsed}")
if rows:
    ds = np.array([r[3] for r in rows])
    ratios = np.array([r[4] for r in rows])
    print(f"  d(O,O):  min {ds.min():.3f}  median {np.median(ds):.3f}  max {ds.max():.3f}")
    print(f"  ratio :  min {ratios.min():.3f}  median {np.median(ratios):.3f}  max {ratios.max():.3f}")
    print(f"  caught by ratio 1.0: {caught_10}/{total_collapsed}")
    print(f"  caught by ratio 1.2: {caught_12}/{total_collapsed}")
    print(f"  geo.check 'fusion' flagged (current ratio {geo._FUSE_RATIO}): {fusion_flagged}/{total_collapsed}")
    print("\n  detail (iso, seed, O-C-O deg, d(O,O), ratio, flagged):")
    for r in sorted(rows, key=lambda x: x[3]):
        print(f"    iso{r[0]} seed{r[1]}: {r[2]:5.1f}deg  d={r[3]:.3f}  ratio={r[4]:.3f}  flagged={r[5]}")
