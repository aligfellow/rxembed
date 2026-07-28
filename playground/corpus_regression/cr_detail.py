"""Named-structure detail behind the aggregate: zero-conf cases, crossovers, clash jump, key regressions.

Usage: uv run python cr_detail.py <tree1.json> <tree2.json> <tree3.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def load(p):
    return {(r["name"], r["seed"]): r for r in json.load(open(p))}


A, B, C = (load(p) for p in sys.argv[1:4])
names = sorted({n for n, _s in A})
seeds = sorted({s for _n, s in A})


def half_of(n):
    for k in A:
        if k[0] == n:
            return A[k].get("half")
    return "?"


print("=" * 90)
print("ZERO-CONFORMER RETURNS (silent embed failures) — by tree, any seed")
print("=" * 90)
for lbl, ix in (("HEAD", A), ("+T3", B), ("+T3+T4", C)):
    zc = defaultdict(list)
    for (n, s), r in ix.items():
        if r.get("zero_conf"):
            zc[n].append(hex(s))
    print(f"  {lbl:8s}: {len(zc)} structures — " + ", ".join(f"{n}[{half_of(n)}]{v}" for n, v in sorted(zc.items())))

print("\n" + "=" * 90)
print("BOUND CROSSOVER (smoothing tol > 0 at embed) — by tree")
print("=" * 90)
for lbl, ix in (("HEAD", A), ("+T3", B), ("+T3+T4", C)):
    cx = defaultdict(list)
    for (n, s), r in ix.items():
        if r.get("tol_nonzero", 0) > 0:
            cx[n].append((hex(s), r.get("tol_max")))
    print(
        f"  {lbl:8s}: {len(cx)} structures — " + ", ".join(f"{n}[{half_of(n)}]{v}" for n, v in sorted(cx.items()))
        if cx
        else f"  {lbl:8s}: none"
    )

print("\n" + "=" * 90)
print("dg_floors emitted per structure (the T4 substrate — non-zero means dg_floors were WRITTEN)")
print("=" * 90)
nz = [(n, A[(n, seeds[0])].get("n_dg_floors", 0)) for n in names if A.get((n, seeds[0]))]
haz = [(n, v) for n, v in nz if v]
print(f"  structures with any dg_floors: {len(haz)} / {len(nz)}")
print("  (these are all from nondonor_floors, the single-source producer — collisions need TWO sources,")
print("   which only the coordinate= path creates; see cr_t4_fires.json: 0 collisions corpus-wide)")


def perconf_clashes(ix):
    """conformers with a clash violation at post, and the structures they belong to."""
    n_clash, structs = 0, defaultdict(int)
    for (n, s), r in ix.items():
        for c in r.get("confs", []):
            post = c.get("post")
            if post and post["gate_kinds"].get("clash"):
                n_clash += 1
                structs[n] += 1
    return n_clash, structs


print("\n" + "=" * 90)
print("POST-MINIMIZE CLASH conformers — where the census jump 28->126 comes from")
print("=" * 90)
for lbl, ix in (("HEAD", A), ("+T3", B)):
    nc, st = perconf_clashes(ix)
    print(
        f"  {lbl:8s}: {nc} clash-conformers across {len(st)} structures: "
        + ", ".join(f"{n}[{half_of(n)}]x{c}" for n, c in sorted(st.items(), key=lambda t: -t[1])[:15])
    )


def post_axis(ix, n, ax):
    vals = []
    for s in seeds:
        r = ix.get((n, s))
        if not r:
            continue
        cv = [c["post"][ax] for c in r.get("confs", []) if c.get("post") and c["post"].get(ax) is not None]
        if cv:
            vals.append(float(np.mean(cv)))
    return float(np.mean(vals)) if vals else None


print("\n" + "=" * 90)
print("KEY POST REGRESSIONS — HEAD vs +T3, mean over seeds")
print("=" * 90)
for n in ["WUVJAB", "SOHMEJ", "YIXSIJ", "PdCl2-RR-BDNN", "WELROW", "HOXLAK", "QUZKAZ", "KEGFOU"]:
    row = []
    for ax in ("ml_mae", "fold", "rmsd", "bond_mae"):
        a, b = post_axis(A, n, ax), post_axis(B, n, ax)
        if a is None or b is None:
            row.append(f"{ax}: n/a")
        else:
            row.append(f"{ax}: {a:.3f}->{b:.3f} ({b - a:+.3f})")
    print(f"  {n:16s}[{half_of(n)}]  " + " | ".join(row))


print("\n" + "=" * 90)
print("CONFORMER YIELD — embedded vs kept-through-minimize, pooled by tree")
print("=" * 90)
for lbl, ix in (("HEAD", A), ("+T3", B), ("+T3+T4", C)):
    emb = sum(r.get("n_embed_conf", 0) for r in ix.values())
    kept = sum(r.get("n_survived_minimize", 0) for r in ix.values())
    scored = sum(1 for r in ix.values() if r.get("confs"))
    print(f"  {lbl:8s}: embedded={emb} kept={kept} ({100 * kept / max(emb, 1):.1f}%) scored_runs={scored}")

# structures that go fully empty AFTER minimize (kept==0) though they embedded
print("\n  runs that embedded but minimize emptied (kept==0):")
for lbl, ix in (("HEAD", A), ("+T3", B)):
    empt = sorted(
        {n for (n, s), r in ix.items() if r.get("n_embed_conf", 0) > 0 and r.get("n_survived_minimize", 0) == 0}
    )
    print(f"    {lbl:8s}: {len(empt)} — {empt}")
