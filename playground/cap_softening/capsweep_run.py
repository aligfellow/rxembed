"""Full coplanar-cap softening sweep. Writes one JSON line per setting, incrementally."""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad")

import capsweep_lib as L

OUT = "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad/capsweep_results.jsonl"

# (label, kwargs to set_levers)
SETTINGS = [
    ("baseline cap45 fc10", dict(cap=45.0, fc=10.0)),
    # --- FC lever (global), CAP=45 ---
    ("global fc5", dict(cap=45.0, fc=5.0)),
    ("global fc3", dict(cap=45.0, fc=3.0)),
    ("global fc1", dict(cap=45.0, fc=1.0)),
    ("global fc0", dict(cap=45.0, fc=0.0)),
    # --- CAP lever (global), FC=10 ---
    ("global cap60", dict(cap=60.0, fc=10.0)),
    ("global cap75", dict(cap=75.0, fc=10.0)),
    ("global cap90", dict(cap=90.0, fc=10.0)),
    # --- crowding-conditional: soften FC only on >=3 capped donors ---
    ("crowd>=3 fc3", dict(cap=45.0, fc=10.0, crowd_fc=3.0, crowd_n=3)),
    ("crowd>=3 fc1", dict(cap=45.0, fc=10.0, crowd_fc=1.0, crowd_n=3)),
    ("crowd>=3 fc0", dict(cap=45.0, fc=10.0, crowd_fc=0.0, crowd_n=3)),
    # --- crowding-conditional: widen CAP only on >=3 capped donors ---
    ("crowd>=3 cap75", dict(cap=45.0, fc=10.0, crowd_cap=75.0, crowd_n=3)),
    ("crowd>=3 cap90", dict(cap=45.0, fc=10.0, crowd_cap=90.0, crowd_n=3)),
    # --- crowding-conditional: only >=4 (isolate case2) ---
    ("crowd>=4 fc1", dict(cap=45.0, fc=10.0, crowd_fc=1.0, crowd_n=4)),
]


def measure_all(label, kwargs):
    L.set_levers(**kwargs)
    rec = {"label": label, "kwargs": {k: v for k, v in kwargs.items()}}
    t0 = time.time()
    rec["case2"] = L.flag_rate("case2", n=6)
    rec["case3"] = L.flag_rate("case3", n=6)
    rec["epoxide"] = L.epoxide(nseeds=30)
    rec["thione"] = L.thione(n=6)
    rec["HENRY"] = {str(k): v for k, v in L.control_oop(L.HENRY, n=6, seeds=(1, 7, 13)).items()}
    rec["KETONE"] = {str(k): v for k, v in L.control_oop(L.KETONE, n=6, seeds=(1, 7, 13)).items()}
    rec["PICO"] = {str(k): v for k, v in L.control_oop(L.PICO, n=6, seeds=(1, 7, 13)).items()}
    rec["secs"] = round(time.time() - t0, 1)
    L.reset_levers()
    return rec


if __name__ == "__main__":
    # optional: run a subset by index range, e.g. `capsweep_run.py 0 5`
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(SETTINGS)
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    with open(out, "w") as f:
        for label, kwargs in SETTINGS[lo:hi]:
            rec = measure_all(label, kwargs)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"done: {label}  ({rec['secs']}s)  case2={rec['case2'][0]:.0%} case3={rec['case3'][0]:.0%} "
                f"epox={rec['epoxide'][0]}/{rec['epoxide'][1]}",
                flush=True,
            )
