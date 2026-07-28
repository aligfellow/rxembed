"""Does also disabling the DG seed bound on crowded metals beat FF-only softening?

Combinations on crowded (>=3 capped) metals, n=10 x 3 seeds, case2/case3 + epoxide + control check.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad")

import capsweep_lib as L

OUT = "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad/capsweep_dgexp.jsonl"
SEEDS = (0xF00D, 0xBEEF, 0x1234)

COMBOS = [
    ("A: crowd fc1, DG on", dict(crowd_fc=1.0, crowd_n=3, crowd_dg_off=False)),
    ("B: crowd fc1, DG off", dict(crowd_fc=1.0, crowd_n=3, crowd_dg_off=True)),
    ("C: crowd fc10, DG off", dict(crowd_fc=10.0, crowd_n=3, crowd_dg_off=True)),  # DG-only softening
    ("D: crowd fc0, DG off", dict(crowd_fc=0.0, crowd_n=3, crowd_dg_off=True)),  # full cap off on crowded
]


def multi(case, n, seeds):
    rates, kinds = [], {}
    for sd in seeds:
        r, _n, k = L.flag_rate(case, n=n, seed=sd)
        rates.append(round(r, 3))
        for a, b in k.items():
            kinds[a] = kinds.get(a, 0) + b
    return rates, kinds


f = open(OUT, "w")
for lbl, kw in COMBOS:
    L.set_levers(cap=45.0, fc=10.0, **kw)
    c2, c2k = multi("case2", 10, SEEDS)
    c3, c3k = multi("case3", 10, SEEDS)
    ep = L.epoxide(nseeds=40)
    # control (must be baseline-identical: 2-capped, predicate never fires)
    hen = L.control_oop(L.HENRY, n=8, seeds=(1, 7, 13))
    L.reset_levers()
    rec = {
        "label": lbl,
        "case2": c2,
        "case2_kinds": c2k,
        "case3": c3,
        "case3_kinds": c3k,
        "epoxide": list(ep),
        "HENRY": {str(k): (round(v[0], 1), round(v[1], 1), v[2]) for k, v in hen.items()},
    }
    f.write(json.dumps(rec) + "\n")
    f.flush()
    print(f"{lbl}: case2={c2} case3={c3} epox={ep[0]}/{ep[1]} HENRY={rec['HENRY']}", flush=True)
f.close()
