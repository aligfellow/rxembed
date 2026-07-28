"""Aggregate the sweep JSONL into readable tables."""

from __future__ import annotations

import glob
import json

recs = []
for fn in sorted(
    glob.glob(
        "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad/res_*.jsonl"
    )
):
    for line in open(fn):
        recs.append(json.loads(line))

# preserve intended order
ORDER = [
    "baseline cap45 fc10",
    "global fc5",
    "global fc3",
    "global fc1",
    "global fc0",
    "global cap60",
    "global cap75",
    "global cap90",
    "crowd>=3 fc3",
    "crowd>=3 fc1",
    "crowd>=3 fc0",
    "crowd>=3 cap75",
    "crowd>=3 cap90",
    "crowd>=4 fc1",
]
by = {r["label"]: r for r in recs}


def ctrl_str(d):
    # d: {donor_idx: (median, max, sym, n)}
    return " ".join(f"{v[2]}:{v[0]:.0f}/{v[1]:.0f}" for v in d.values())


print(
    f"{'setting':22} | case2 (kinds)           | case3 (kinds)           | epox    | thione med/p95/max | HENRY med/max  | KETONE        | PICO"
)
print("-" * 170)
for lbl in ORDER:
    if lbl not in by:
        continue
    r = by[lbl]
    c2r, _c2n, c2k = r["case2"]
    c3r, _c3n, c3k = r["case3"]
    ep_c, ep_t, ep_m = r["epoxide"]
    th = r["thione"]

    def kk(k):
        return ",".join(f"{a}{b}" for a, b in k.items()) or "-"

    print(
        f"{lbl:22} | {c2r:4.0%} {kk(c2k):18} | {c3r:4.0%} {kk(c3k):18} | {ep_c:2}/{ep_t} m{ep_m:3.0f} | "
        f"{th[0]:4.0f}/{th[1]:4.0f}/{th[2]:4.0f}      | {ctrl_str(r['HENRY']):14} | {ctrl_str(r['KETONE']):13} | {ctrl_str(r['PICO'])}"
    )

print()
print("Legend: case columns show flag-rate + geom.check kind histogram (planarity/conjugation/clash...).")
print("        epox = # collapsed (O-C-O<90)/30 seeds, m=median O-C-O deg. thione = Ni-S-C-N |dev| folded to [0,90].")
print("        controls = per capped donor  sym:median/max out-of-plane deg (LOW = metal held in plane = cap working).")
