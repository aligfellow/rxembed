"""Diff the per-tree JSONs from cr_measure.py — per half, pooled, and per named structure.

Comparison unit is one (structure, seed) pair, aggregated over that pair's conformers, so trees are
compared on identical inputs and never across different geometries. A structure is called improved /
regressed only when it moves in the SAME direction on a majority of seeds (>= 2 of 3), which is what
keeps one lucky seed from naming a regression.

Usage: uv run python cr_compare.py <tree1.json> <tree2.json> <tree3.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

# axes where LOWER is better, and the tolerance below which a change is called noise
AXES = {
    "ml_mae": 0.005,  # A, M-donor distance error vs crystal
    "fold": 1.0,  # deg, donor fold
    "bond_mae": 0.005,  # A, bond length error vs crystal
    "rmsd": 0.02,  # A, heavy-atom RMSD to crystal
    "dwin_max": 0.005,  # A, worst distance-window violation
    "awin_max": 0.5,  # deg, worst angle-window violation
}


def load(p):
    return json.load(open(p))


def key(r):
    return (r["name"], r["seed"])


def conf_agg(rec, stage="embed"):
    """Mean over a record's conformers for each axis, plus the gate pass rate and kind census."""
    confs = [c[stage] for c in rec.get("confs", []) if c.get(stage) is not None]
    if not confs:
        return None
    out = {}
    for ax in AXES:
        vals = [c[ax] for c in confs if c.get(ax) is not None]
        out[ax] = float(np.mean(vals)) if vals else None
    # Derived from the KIND CENSUS, not the recorded `gate_ok`: an early cr_measure.py wrote
    # `bool(rep.ok)` where `ok` is a *method*, so the stored flag is True for every conformer ever
    # measured. The census comes straight off `rep.violations` and is unaffected, and "no violation
    # of any kind" is exactly `GeometryReport.ok()`. cr_measure.py is fixed; this stays so the
    # already-collected runs remain readable.
    out["gate_pass"] = float(np.mean([0.0 if c["gate_kinds"] else 1.0 for c in confs]))
    kinds = defaultdict(int)
    for c in confs:
        for k, v in c["gate_kinds"].items():
            kinds[k] += v
    out["gate_kinds"] = dict(kinds)
    out["n"] = len(confs)
    return out


def index(data, stage="embed"):
    """(name, seed) -> {aggregated axes, plus the record-level counters}."""
    ix = {}
    for r in data:
        k = key(r)
        rec = {
            "half": r.get("half"),
            "skip": r.get("skip"),
            "error": r.get("error"),
            "zero_conf": bool(r.get("zero_conf")),
            "n_embed_conf": r.get("n_embed_conf", 0),
            "n_survived_minimize": r.get("n_survived_minimize", 0),
            "tol_nonzero": r.get("tol_nonzero", 0),
            "tol_max": r.get("tol_max", 0.0),
            "charge_ok": r.get("charge_ok"),
            "agg": conf_agg(r, stage),
            "crystal": r.get("crystal"),
        }
        ix[k] = rec
    return ix


def summarise(ix, label, half=None):
    rows = [v for k, v in ix.items() if half is None or v["half"] == half]
    scored = [v for v in rows if v["agg"]]
    out = {
        "label": label,
        "half": half or "pooled",
        "n_runs": len(rows),
        "n_scored": len(scored),
        "n_zero_conf": sum(1 for v in rows if v["zero_conf"]),
        "n_error": sum(1 for v in rows if v["error"]),
        "n_skip": sum(1 for v in rows if v["skip"] and not v["zero_conf"]),
        "conf_total": sum(v["n_embed_conf"] for v in rows),
        "kept_total": sum(v["n_survived_minimize"] for v in rows),
        "runs_with_crossover": sum(1 for v in rows if v["tol_nonzero"] > 0),
        "charge_bad": sum(1 for v in rows if v["charge_ok"] is False),
    }
    for ax in AXES:
        vals = [v["agg"][ax] for v in scored if v["agg"].get(ax) is not None]
        out[ax] = float(np.mean(vals)) if vals else None
    out["gate_pass"] = float(np.mean([v["agg"]["gate_pass"] for v in scored])) if scored else None
    kinds = defaultdict(int)
    for v in scored:
        for k, n in v["agg"]["gate_kinds"].items():
            kinds[k] += n
    out["gate_kinds"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))
    return out


def movers(a, b, ax, tol):
    """Per-structure verdict on one axis: improved / regressed on a MAJORITY of seeds."""
    per = defaultdict(list)
    for k in set(a) & set(b):
        name, _seed = k
        va, vb = a[k]["agg"], b[k]["agg"]
        if not va or not vb or va.get(ax) is None or vb.get(ax) is None:
            continue
        per[name].append(vb[ax] - va[ax])  # negative = b better
    imp, reg, flat = [], [], []
    for name, ds in per.items():
        nb = sum(1 for d in ds if d < -tol)
        nw = sum(1 for d in ds if d > tol)
        m = float(np.mean(ds))
        if nb > len(ds) / 2:
            imp.append((name, m))
        elif nw > len(ds) / 2:
            reg.append((name, m))
        else:
            flat.append((name, m))
    imp.sort(key=lambda t: t[1])
    reg.sort(key=lambda t: -t[1])
    return imp, reg, flat


def seed_sensitivity(ix, ax):
    """Spread across seeds within one structure — how much of any delta is seed noise."""
    per = defaultdict(list)
    for (name, _seed), v in ix.items():
        if v["agg"] and v["agg"].get(ax) is not None:
            per[name].append(v["agg"][ax])
    spreads = [max(v) - min(v) for v in per.values() if len(v) > 1]
    return {
        "n_structs": len(spreads),
        "mean_spread": float(np.mean(spreads)) if spreads else None,
        "median_spread": float(np.median(spreads)) if spreads else None,
        "p90_spread": float(np.percentile(spreads, 90)) if spreads else None,
        "max_spread": float(np.max(spreads)) if spreads else None,
    }


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main(paths, stage="embed"):
    labels = ["1:HEAD", "2:+T3", "3:+T3+T4"][: len(paths)]
    ixs = [index(load(p), stage) for p in paths]
    # Restrict every tree to the (structure, seed) pairs ALL trees produced. Pooling unmatched run
    # sets would compare different corpora — and while a run is in flight the trees are at different
    # points in the list, which is exactly how that happens.
    common = set.intersection(*(set(ix) for ix in ixs))
    dropped = [sorted({k for ix in ixs for k in ix} - common)]
    ixs = [{k: v for k, v in ix.items() if k in common} for ix in ixs]
    print(f"[matched keys: {len(common)}; unmatched dropped: {len(dropped[0])}]")

    print(f"\n{'=' * 100}\nSTAGE: {stage}\n{'=' * 100}")
    for half in ("tmQM", "fixtures", None):
        hn = half or "POOLED"
        print(f"\n----- {hn} -----")
        sums = [summarise(ix, lb, half) for ix, lb in zip(ixs, labels)]
        cols = [
            "n_runs",
            "n_scored",
            "n_zero_conf",
            "n_error",
            "conf_total",
            "kept_total",
            "runs_with_crossover",
            "charge_bad",
            "gate_pass",
            *AXES,
        ]
        print(f"{'metric':22s}" + "".join(f"{lb:>14s}" for lb in labels))
        for c in cols:
            vals = []
            for s in sums:
                v = s[c]
                vals.append(f"{v:>14d}" if isinstance(v, int) else f"{fmt(v):>14s}")
            print(f"{c:22s}" + "".join(vals))
        for s in sums:
            print(f"  gate_kinds[{s['label']}]: {s['gate_kinds']}")

    print(f"\n----- seed sensitivity ({stage}) -----")
    for ax in ("ml_mae", "rmsd", "awin_max"):
        for ix, lb in zip(ixs, labels):
            print(f"  {ax:10s} {lb:10s} {seed_sensitivity(ix, ax)}")

    for i, j, tag in (
        (0, 1, "T3 alone (tree1 -> tree2)"),
        (1, 2, "T4 alone (tree2 -> tree3)"),
        (0, 2, "both (tree1 -> tree3)"),
    ):
        if j >= len(ixs):
            continue
        print(f"\n{'=' * 100}\nMOVERS: {tag}  [{stage}]\n{'=' * 100}")
        for ax, tol in AXES.items():
            imp, reg, flat = movers(ixs[i], ixs[j], ax, tol)
            print(f"\n  {ax} (tol {tol}): improved {len(imp)}, regressed {len(reg)}, flat {len(flat)}")
            if imp:
                print("    best :", ", ".join(f"{n} {d:+.3f}" for n, d in imp[:6]))
            if reg:
                print("    WORST:", ", ".join(f"{n} {d:+.3f}" for n, d in reg[:10]))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for st in ("embed", "post"):
        main(args, st)
