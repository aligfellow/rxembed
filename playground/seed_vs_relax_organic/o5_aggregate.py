"""Aggregate `o3_seed_vs_relax_organic.py`'s output into the per-axis table, outliers and seed sensitivity.

Two weightings are reported because they disagree and the disagreement matters:
  * PER-CONFORMER -- every conformer counts once. `bimp-smiles-auto` (32 NCI candidates x 4 conformers)
    then supplies most of the sample, so this is dominated by one input.
  * PER-CASE -- each corpus entry contributes the mean over its own conformers, once. This is the
    headline weighting; the per-conformer numbers are given alongside so the weighting can be checked.

Usage:  uv run python o5_aggregate.py <main.json>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np

AXES = [
    ("conj_max", "conjugation twist, max (deg)", "lower"),
    ("conj_mean", "conjugation twist, mean (deg)", "lower"),
    ("conj_nviol", "conjugation quartets > 30 deg (count)", "lower"),
    ("oop_max", "sp2 out-of-plane, max (A)", "lower"),
    ("dwin_max", "distance-window violation, max (A)", "lower"),
    ("awin_max", "angle-window violation, max (deg)", "lower"),
    ("bond_mae", "bond-length MAE vs reference (A)", "lower"),
    ("bond_max", "bond-length max vs reference (A)", "lower"),
    ("frozen_drift", "frozen-core drift vs reference (A)", "lower"),
]


def pairs(rows):
    """Yield (row, cand, conf) for every scored, constrained conformer."""
    for r in rows:
        if "skip" in r:
            continue
        for c in r.get("cands", []):
            if not c.get("constrained", False):
                continue
            for cf in c.get("confs", []):
                yield r, c, cf


def main(path):
    rows = json.load(open(path))
    allp = list(pairs(rows))

    print(
        f"\n=== corpus ===\n{len({r['id'] for r in rows if 'skip' not in r})} cases x "
        f"{len({r['seed'] for r in rows if 'skip' not in r})} seeds; {len(allp)} scored conformer pairs"
    )
    skips = [r for r in rows if "skip" in r]
    if skips:
        print("SKIPPED:", Counter(f"{r['id']}: {r['skip'][:60]}" for r in skips))

    # --- guards
    bad_repro = {r["id"] for r in rows if "skip" not in r and r["reproducible"] is not True}
    unmoved = sum(1 for _, _, cf in allp if not cf["moved"])
    print(f"G3 reproducible embed: {'ALL PASS' if not bad_repro else 'FAIL ' + str(bad_repro)}")
    print(f"G4 seed != relax coords: {len(allp) - unmoved}/{len(allp)} moved ({unmoved} unmoved)")
    noop = [
        (r["id"], c.get("noop_verified"))
        for r in rows
        if "skip" not in r
        for c in r["cands"]
        if not c.get("constrained", True)
    ]
    if noop:
        ok = sum(1 for _, v in noop if v)
        print(f"G1 unconstrained embeds: {ok}/{len(noop)} verified byte-identical no-op (the relax must not fire)")

    # --- per-axis
    for weighting in ("PER-CONFORMER", "PER-CASE"):
        print(f"\n=== per-axis, {weighting} ===")
        print(f"{'axis':40s} {'seed':>9} {'relax':>9} {'ref':>9} {'delta':>9}   help/hurt/tie")
        for key, label, _ in AXES:
            s_all = [cf["seed"][key] for _, _, cf in allp if cf["seed"][key] is not None]
            r_all = [cf["relax"][key] for _, _, cf in allp if cf["relax"][key] is not None]
            if not s_all:
                continue
            help_ = sum(
                1
                for _, _, cf in allp
                if cf["seed"][key] is not None
                and cf["relax"][key] is not None
                and cf["relax"][key] < cf["seed"][key] - 1e-9
            )
            hurt = sum(
                1
                for _, _, cf in allp
                if cf["seed"][key] is not None
                and cf["relax"][key] is not None
                and cf["relax"][key] > cf["seed"][key] + 1e-9
            )
            tie = len(s_all) - help_ - hurt
            if weighting == "PER-CASE":
                per = defaultdict(lambda: ([], []))
                for r, _, cf in allp:
                    if cf["seed"][key] is None:
                        continue
                    per[r["id"]][0].append(cf["seed"][key])
                    per[r["id"]][1].append(cf["relax"][key])
                s_all = [float(np.mean(v[0])) for v in per.values()]
                r_all = [float(np.mean(v[1])) for v in per.values()]
                help_ = sum(1 for v in per.values() if np.mean(v[1]) < np.mean(v[0]) - 1e-9)
                hurt = sum(1 for v in per.values() if np.mean(v[1]) > np.mean(v[0]) + 1e-9)
                tie = len(per) - help_ - hurt
            refs = [
                c["ref"][key]
                for r in rows
                if "skip" not in r
                for c in r.get("cands", [])
                if "ref" in c and c["ref"].get(key) is not None
            ]
            rf = f"{np.mean(refs):9.4f}" if refs else f"{'-':>9}"
            print(
                f"{label:40s} {np.mean(s_all):9.4f} {np.mean(r_all):9.4f} {rf} "
                f"{np.mean(r_all) - np.mean(s_all):+9.4f}   {help_}/{hurt}/{tie}"
            )

    # --- the gate
    gs = sum(cf["seed"]["gate_ok"] for _, _, cf in allp)
    gr = sum(cf["relax"]["gate_ok"] for _, _, cf in allp)
    print(
        f"\n=== geometry.check ===\npass rate: seed {gs}/{len(allp)} ({100 * gs / len(allp):.1f}%)  ->  "
        f"relax {gr}/{len(allp)} ({100 * gr / len(allp):.1f}%)"
    )
    for side in ("seed", "relax"):
        ct = Counter(k for _, _, cf in allp for k in cf[side]["gate_kinds"])
        print(f"  {side:5s} violations by kind: {dict(ct.most_common())}")
    flips = Counter()
    for _, _, cf in allp:
        a, b = cf["seed"]["gate_ok"], cf["relax"]["gate_ok"]
        flips["clean->clean" if a and b else "clean->FAIL" if a else "FAIL->clean" if b else "FAIL->FAIL"] += 1
    print(f"  transitions: {dict(flips)}")

    # --- per-case gate table
    print("\n=== per case: geometry.check clean, pooled over seeds ===")
    per = defaultdict(lambda: [0, 0, 0])
    fam = {}
    for r, _, cf in allp:
        p = per[r["id"]]
        p[0] += cf["seed"]["gate_ok"]
        p[1] += cf["relax"]["gate_ok"]
        p[2] += 1
        fam[r["id"]] = r["family"]
    print(f"{'case':24s} {'family':10s} {'seed':>9} {'relax':>9}   conj_max seed->relax")
    cj = defaultdict(lambda: ([], []))
    for r, _, cf in allp:
        cj[r["id"]][0].append(cf["seed"]["conj_max"])
        cj[r["id"]][1].append(cf["relax"]["conj_max"])
    for k in sorted(per, key=lambda k: (per[k][1] - per[k][0]) / max(per[k][2], 1)):
        s, rr, n = per[k]
        a, b = cj[k]
        print(
            f"{k:24s} {fam[k]:10s} {s:4d}/{n:<4d} {rr:4d}/{n:<4d}   "
            f"{np.mean(a):5.1f} -> {np.mean(b):5.1f}  (max {max(b):.1f})"
        )

    # --- seed sensitivity
    print("\n=== seed sensitivity: geometry.check clean fraction by embed seed ===")
    bysd = defaultdict(lambda: [0, 0, 0])
    for r, _, cf in allp:
        p = bysd[r["seed"]]
        p[0] += cf["seed"]["gate_ok"]
        p[1] += cf["relax"]["gate_ok"]
        p[2] += 1
    for sd in sorted(bysd):
        s, rr, n = bysd[sd]
        print(
            f"  seed {sd:<8} seed {s:4d}/{n:<4d} ({100 * s / n:5.1f}%)  ->  relax {rr:4d}/{n:<4d} ({100 * rr / n:5.1f}%)"
        )

    # --- outliers
    print("\n=== outliers: conjugation most degraded (per case, mean over all conformers/seeds) ===")
    deg = sorted(((np.mean(b) - np.mean(a), k) for k, (a, b) in cj.items()), reverse=True)
    for d, k in deg[:10]:
        print(f"  {k:24s} {np.mean(cj[k][0]):5.1f} -> {np.mean(cj[k][1]):5.1f} deg  ({d:+.1f})")
    print("  --- most improved ---")
    for d, k in deg[-6:]:
        print(f"  {k:24s} {np.mean(cj[k][0]):5.1f} -> {np.mean(cj[k][1]):5.1f} deg  ({d:+.1f})")

    print("\n=== outliers: window violation most improved (per case) ===")
    win = defaultdict(lambda: ([], []))
    for r, _, cf in allp:
        win[r["id"]][0].append(max(cf["seed"]["dwin_max"], cf["seed"]["awin_max"] / 30.0))
        win[r["id"]][1].append(max(cf["relax"]["dwin_max"], cf["relax"]["awin_max"] / 30.0))
    wd = defaultdict(lambda: ([], [], [], []))
    for r, _, cf in allp:
        wd[r["id"]][0].append(cf["seed"]["dwin_max"])
        wd[r["id"]][1].append(cf["relax"]["dwin_max"])
        wd[r["id"]][2].append(cf["seed"]["awin_max"])
        wd[r["id"]][3].append(cf["relax"]["awin_max"])
    print(f"{'case':24s} {'dist win (A)':>22}  {'angle win (deg)':>22}")
    for k in sorted(wd):
        ds, dr, as_, ar = (np.mean(v) for v in wd[k])
        print(f"{k:24s}   {ds:8.3f} -> {dr:8.3f}    {as_:8.2f} -> {ar:8.2f}")


def detail(path):
    """Per-case violation kinds and, where a real geometry exists, the reference's own scores.

    `main()` shows THAT the gate pass rate falls; this shows WHICH violation fires, which is the
    difference between one harm mode and several.

    Usage:  uv run python o5_aggregate.py <main.json> detail
    """
    rows = json.load(open(path))
    allp = list(pairs(rows))

    print("\n=== per case: which violation fires (counts over all conformers x seeds) ===")
    kinds = defaultdict(lambda: (Counter(), Counter()))
    for r, _, cf in allp:
        for k in cf["seed"]["gate_kinds"]:
            kinds[r["id"]][0][k] += 1
        for k in cf["relax"]["gate_kinds"]:
            kinds[r["id"]][1][k] += 1
    for k in sorted(kinds):
        s, rr = kinds[k]
        print(f"{k:24s} seed {str(dict(s.most_common())):46s} relax {dict(rr.most_common())}")

    print("\n=== cases with a REAL reference geometry: seed / relax / reference ===")
    per = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    for r, c, cf in allp:
        if "ref" not in c:
            continue
        for key, _, _ in AXES:
            if cf["seed"][key] is None:
                continue
            per[r["id"]][key][0].append(cf["seed"][key])
            per[r["id"]][key][1].append(cf["relax"][key])
            per[r["id"]][key][2].append(c["ref"][key])
    for case in sorted(per):
        print(f"\n  {case}")
        for key, label, _ in AXES:
            v = per[case][key]
            if not v[0]:
                continue
            print(f"    {label:38s} seed {np.mean(v[0]):8.4f}  relax {np.mean(v[1]):8.4f}  REF {np.mean(v[2]):8.4f}")
        print(
            f"    reference passes geometry.check: "
            f"{[c['ref']['gate_ok'] for r, c, _ in allp if r['id'] == case][0]}  "
            f"(reference gate kinds {[c['ref']['gate_kinds'] for r, c, _ in allp if r['id'] == case][0]})"
        )

    print("\n=== drift: how far the relax moved each conformer (heavy-atom RMSD, A) ===")
    dr = defaultdict(list)
    for r, _, cf in allp:
        dr[r["id"]].append(cf["drift"])
    for k in sorted(dr, key=lambda k: -np.mean(dr[k])):
        print(f"  {k:24s} mean {np.mean(dr[k]):6.3f}  max {max(dr[k]):6.3f}")


if __name__ == "__main__":
    (detail if "detail" in sys.argv[2:] else main)(sys.argv[1])
