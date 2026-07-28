"""Confirmation: crowd>=3 with FC in {1,2,3} at higher n + multiple initial seeds (seed sensitivity).

Also: control oop at the chosen setting (must equal baseline), epoxide-40, and the RDP pin test.
Writes to capsweep_confirm.jsonl.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad")

import capsweep_lib as L
import rxembed as rx
from rxembed import geometry as geo

OUT = "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad/capsweep_confirm.jsonl"
SEEDS = (0xF00D, 0xBEEF, 0x1234)

RDP = (
    "CCC1=C2CCCCC2=C(CC)[P](c2ccccc2)(c2ccccc2)->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[P]1(c1ccccc1)c1ccccc1"
)


def flag_multi(case, n, seeds):
    rates = []
    kinds_tot = {}
    for sd in seeds:
        r, _nn, k = L.flag_rate(case, n=n, seed=sd)
        rates.append(round(r, 3))
        for a, b in k.items():
            kinds_tot[a] = kinds_tot.get(a, 0) + b
    return rates, kinds_tot


def rdp_pin(**kw):
    L.set_levers(**kw)
    iso = rx.metal(RDP, "square_planar")[0]
    ens = rx.embed(iso, n=6, seed=0xF00D).minimize()
    clean = sum(1 for c in ens.ids if geo.check(ens.mol, c).ok())
    L.reset_levers()
    return len(ens.ids), clean


def run():
    out = open(OUT, "w")
    for fc in (1.0, 2.0, 3.0):
        kw = dict(cap=45.0, fc=10.0, crowd_fc=fc, crowd_n=3)
        L.set_levers(**kw)
        c2, c2k = flag_multi("case2", 10, SEEDS)
        c3, c3k = flag_multi("case3", 10, SEEDS)
        L.reset_levers()
        pin_n, pin_clean = rdp_pin(**kw)
        rec = {
            "label": f"crowd>=3 fc{fc}",
            "fc": fc,
            "case2_rates": c2,
            "case2_kinds": c2k,
            "case3_rates": c3,
            "case3_kinds": c3k,
            "rdp_pin": [pin_n, pin_clean],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(
            f"crowd>=3 fc{fc}: case2={c2} k={c2k} | case3={c3} k={c3k} | RDP pin {pin_clean}/{pin_n} clean", flush=True
        )

    # baseline for reference at same higher n/seeds
    L.set_levers(cap=45.0, fc=10.0)
    b2, b2k = flag_multi("case2", 10, SEEDS)
    b3, b3k = flag_multi("case3", 10, SEEDS)
    L.reset_levers()
    bn, bc = rdp_pin(cap=45.0, fc=10.0)
    rec = {
        "label": "baseline",
        "case2_rates": b2,
        "case2_kinds": b2k,
        "case3_rates": b3,
        "case3_kinds": b3k,
        "rdp_pin": [bn, bc],
    }
    out.write(json.dumps(rec) + "\n")
    out.flush()
    print(f"baseline: case2={b2} k={b2k} | case3={b3} k={b3k} | RDP pin {bc}/{bn} clean", flush=True)

    # control oop at chosen fc=2 (must be identical to baseline) + baseline, higher n
    for lbl, kw in (
        ("baseline", dict(cap=45.0, fc=10.0)),
        ("crowd>=3 fc2", dict(cap=45.0, fc=10.0, crowd_fc=2.0, crowd_n=3)),
    ):
        L.set_levers(**kw)
        hen = L.control_oop(L.HENRY, n=10, seeds=(1, 7, 13, 21, 42))
        ket = L.control_oop(L.KETONE, n=10, seeds=(1, 7, 13, 21, 42))
        pic = L.control_oop(L.PICO, n=10, seeds=(1, 7, 13, 21, 42))
        L.reset_levers()

        def fmt(d):
            return {str(k): (round(v[0], 1), round(v[1], 1), v[2]) for k, v in d.items()}

        rec = {"label": f"CONTROL {lbl}", "HENRY": fmt(hen), "KETONE": fmt(ket), "PICO": fmt(pic)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"CONTROL {lbl}: HENRY={fmt(hen)} KETONE={fmt(ket)} PICO={fmt(pic)}", flush=True)

    # epoxide at fc2, 40 seeds
    L.set_levers(cap=45.0, fc=10.0, crowd_fc=2.0, crowd_n=3)
    ep = L.epoxide(nseeds=40)
    L.reset_levers()
    ep_b = None
    L.set_levers(cap=45.0, fc=10.0)
    ep_b = L.epoxide(nseeds=40)
    L.reset_levers()
    rec = {"label": "EPOXIDE", "crowd_fc2": ep, "baseline": ep_b}
    out.write(json.dumps(rec) + "\n")
    out.flush()
    print(f"EPOXIDE crowd>=3 fc2={ep}  baseline={ep_b}", flush=True)
    out.close()


if __name__ == "__main__":
    run()
