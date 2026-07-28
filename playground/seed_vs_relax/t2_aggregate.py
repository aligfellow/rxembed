"""T2 aggregation: per-axis seed-vs-relax table + outliers, from t2_main_out.json."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "t2_main_out.json")))
ok = [r for r in d if "skip" not in r]

pairs = [(r, c) for r in ok for c in r["confs"]]
print(f"structures: {len(ok)};  paired conformers: {len(pairs)}")
n_seeds = sum(r["n_seeds"] for r in ok)
n_kept = sum(r["n_survived_minimize"] for r in ok)
print(
    f"seeds embedded: {n_seeds};  survived minimize(): {n_kept}  ({n_seeds - n_kept} dropped, {100 * (n_seeds - n_kept) / n_seeds:.0f}%)"
)
print(
    f"non-zero seed->relax drift on every pair: {all(c['drift'] > 1e-9 for _, c in pairs)} "
    f"(min {min(c['drift'] for _, c in pairs):.3f} A, median {np.median([c['drift'] for _, c in pairs]):.2f} A)"
)
print(f"reproducible embed on every structure: {all(r['reproducible_embed'] for r in ok)}")
print(
    f"metal surrogate C pre / real Z + oxidation state post on every structure: "
    f"{all(r['surrogate_z'] == 6 and r['restored_z'] == r['real_z'] and r['restored_q'] == r['real_q'] for r in ok)}"
)
print()

AXES = [
    ("ml_mae", "M-donor dist MAE vs crystal (A)", "lower"),
    ("ml_max", "M-donor dist MAX vs crystal (A)", "lower"),
    ("bond_mae", "bond-length MAE vs crystal (A)", "lower"),
    ("bond_max", "bond-length MAX vs crystal (A)", "lower"),
    ("fold", "donor fold, deg off class median", "lower"),
    ("gate_fold_viol", "donor_orientation gate violations", "lower"),
    ("dwin_max", "distance-window violation, max (A)", "lower"),
    ("awin_max", "angle-window violation, max (deg)", "lower"),
    ("rmsd", "heavy-atom RMSD to crystal (A)", "lower"),
]

print(f"{'axis':40s} {'seed':>10s} {'relax':>10s} {'crystal':>9s}  {'delta':>9s}  {'help/hurt/tie (per conf)':>26s}")
print("-" * 116)
rows = []
for key, label, _ in AXES:
    s = np.array([c["seed"][key] for _, c in pairs if c["seed"][key] is not None], float)
    r = np.array([c["relax"][key] for _, c in pairs if c["relax"][key] is not None], float)
    cy = np.array([rr["crystal"][key] for rr in ok if "crystal" in rr and rr["crystal"][key] is not None], float)
    helps = int(np.sum(r < s - 1e-9))
    hurts = int(np.sum(r > s + 1e-9))
    ties = len(s) - helps - hurts
    cstr = f"{np.mean(cy):9.3f}" if len(cy) else "       --"
    print(
        f"{label:40s} {np.mean(s):10.3f} {np.mean(r):10.3f} {cstr}  {np.mean(r) - np.mean(s):+9.3f}  {helps:6d} /{hurts:5d} /{ties:5d}"
    )
    rows.append(
        (key, label, float(np.mean(s)), float(np.mean(r)), float(np.mean(cy)) if len(cy) else None, helps, hurts, ties)
    )

print()
print("PER-STRUCTURE MEANS (median over that structure's conformers)")
print(
    f"{'name':8s} {'kept':>5s} | {'ml_mae s->r':>18s} | {'bond_mae s->r':>18s} | {'fold s->r':>17s} | {'awin_max s->r':>16s} | {'rmsd s->r':>15s}"
)
per = []
for r in ok:
    if not r["confs"]:
        print(f"{r['name']:8s} {r['n_survived_minimize']:5d} |  (no surviving conformer - minimize dropped all)")
        continue

    def med(side, k):
        return float(np.median([c[side][k] for c in r["confs"]]))

    row = dict(
        name=r["name"],
        kept=r["n_survived_minimize"],
        nseed=r["n_seeds"],
        ml_s=med("seed", "ml_mae"),
        ml_r=med("relax", "ml_mae"),
        bd_s=med("seed", "bond_mae"),
        bd_r=med("relax", "bond_mae"),
        fo_s=med("seed", "fold"),
        fo_r=med("relax", "fold"),
        aw_s=med("seed", "awin_max"),
        aw_r=med("relax", "awin_max"),
        rm_s=med("seed", "rmsd"),
        rm_r=med("relax", "rmsd"),
        fo_c=r["crystal"]["fold"],
        aw_c=r["crystal"]["awin_max"],
        ml_c=r["crystal"]["ml_mae"],
    )
    per.append(row)
    print(
        f"{row['name']:8s} {row['kept']:2d}/{row['nseed']:<2d} | {row['ml_s']:7.3f} -> {row['ml_r']:7.3f} | "
        f"{row['bd_s']:7.3f} -> {row['bd_r']:7.3f} | {row['fo_s']:6.1f} -> {row['fo_r']:6.1f} | "
        f"{row['aw_s']:6.1f} -> {row['aw_r']:5.1f} | {row['rm_s']:5.2f} -> {row['rm_r']:5.2f}"
    )

print()
print("OUTLIERS")


def top(field_s, field_r, label, n=6, worse=True):
    key = (lambda x: x[field_r] - x[field_s]) if worse else (lambda x: x[field_s] - x[field_r])
    xs = sorted(per, key=key, reverse=True)[:n]
    print(f"  {label}:")
    for x in xs:
        print(f"    {x['name']:8s} {x[field_s]:8.3f} -> {x[field_r]:8.3f}  (delta {x[field_r] - x[field_s]:+.3f})")


top("bd_s", "bd_r", "bond-length MAE MOST DEGRADED by relax")
top("fo_s", "fo_r", "donor fold MOST DEGRADED by relax")
top("rm_s", "rm_r", "heavy-atom RMSD to crystal MOST DEGRADED by relax")
top("ml_s", "ml_r", "M-donor distance MOST IMPROVED by relax", worse=False)
top("aw_s", "aw_r", "angle-window satisfaction MOST IMPROVED by relax", worse=False)

print()
print("CRYSTAL BASELINE — does the crystal itself satisfy the windows the embed applied?")
cw = [
    (r["name"], r["crystal"]["awin_max"], r["crystal"]["dwin_max"], r["crystal"]["fold"], r["crystal"]["ml_mae"])
    for r in ok
    if "crystal" in r
]
print(
    f"  crystal angle-window violation: mean {np.mean([x[1] for x in cw]):.2f} deg, max {max(x[1] for x in cw):.2f} deg"
)
print(f"  crystal dist-window  violation: mean {np.mean([x[2] for x in cw]):.3f} A,  max {max(x[2] for x in cw):.3f} A")
print(
    f"  crystal donor fold:             mean {np.mean([x[3] for x in cw]):.1f} deg, max {max(x[3] for x in cw):.1f} deg"
)
print("  structures whose CRYSTAL violates its own angle window by > 5 deg:")
for nm, a, dd, f, _ in sorted(cw, key=lambda x: -x[1])[:10]:
    if a > 5:
        print(f"    {nm:8s} angle {a:6.2f} deg, dist {dd:.3f} A")
json.dump(per, open(os.path.join(HERE, "t2_per_structure.json"), "w"), indent=1)
