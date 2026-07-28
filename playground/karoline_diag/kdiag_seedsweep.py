"""Seed sweep: does .minimize() ever FORM a bond (epoxide) that geom.check does not catch?

Runs the notebook invocation rx.embed(iso, n=1, seed=S).minimize() across many seeds and all isomers,
detecting connectivity changes and whether geom.check flagged them.

Usage: .venv/bin/python kdiag_seedsweep.py <case> [nseeds]
"""

from __future__ import annotations

import argparse
from collections import Counter

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import metal as _metal

import kdiag_harness as H

rx.set_verbose("CRITICAL")


def run(case, nseeds):
    smi = H.CASES[case]
    iso_set = rx.metal(smi, "square_planar")
    print(f"\n### {case}: {len(iso_set)} isomers, {nseeds} seeds each ###")
    formed_hist = Counter()
    formed_and_geomclean = 0
    total = 0
    examples = []
    for k, iso in enumerate(iso_set):
        for s in range(nseeds):
            seed = 0xF00D + s
            ens = rx.embed(iso, n=1, seed=seed).minimize()
            if not ens.ids:
                continue
            total += 1
            metals = set(_metal.metal_indices(ens.mol))
            sphere_donors = sorted({d for ds in ens.sphere.values() for d in ds}) or list(iso.donors)
            for cid in ens.ids:
                formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
                if formed or broken:
                    rep = geo.check(ens.mol, cid, donors=sphere_donors)
                    key = (tuple(formed), tuple(broken))
                    formed_hist[key] += 1
                    if rep.ok():
                        formed_and_geomclean += 1
                    if len(examples) < 6:
                        examples.append((k, seed, cid, formed, broken, rep.ok(), ens))
    print(f"  runs with a connectivity change: {sum(formed_hist.values())}/{total}")
    print(f"  of those, geom.check-CLEAN (silently passed): {formed_and_geomclean}")
    for key, cnt in formed_hist.most_common():
        print(f"    formed={key[0]} broken={key[1]}  x{cnt}")
    return examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("nseeds", nargs="?", type=int, default=25)
    args = ap.parse_args()
    ex = run(args.case, args.nseeds)
    # dump the first epoxide example for inspection
    if ex:
        k, seed, cid, formed, broken, ok, ens = ex[0]
        p = f"/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/epoxide_{args.case}_iso{k}_seed{seed}.xyz"
        try:
            ens.dump(p)
            print(f"\n  dumped first example -> {p}  (iso{k} seed{seed} formed={formed} geom_ok={ok})")
        except Exception as e:
            print("  dump failed:", e)
