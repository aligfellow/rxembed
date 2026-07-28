"""Post-minimize geom.check + connectivity for each case, matching the notebook `rx.embed(iso,n).minimize()`.

The metal is restored inside minimize(), so geom.check/connectivity run on the real metal directly.
Reports per conformer: geom kinds, and the connectivity diff (the epoxide = a FORMED bond).

Usage: .venv/bin/python playground/karoline_diag/kdiag_minimize.py <case> [n] [--ablate what]
"""

from __future__ import annotations

import argparse
from collections import Counter

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import metal as _metal

import kdiag_harness as H  # noqa: E402  (same dir; run from playground/karoline_diag or add path)

rx.set_verbose("ERROR")


def run(case, n, ablate):
    smi = H.CASES[case]
    print(f"\n########## {case} POST-MINIMIZE (ablate={ablate}) ##########")
    iso_set = rx.metal(smi, "square_planar")
    total = Counter()
    flagged = 0
    nconf = 0
    epoxide_confs = 0
    for k, iso in enumerate(iso_set):
        donors = list(iso.donors)
        ens = rx.embed(iso, n=n, seed=0xF00D).minimize()
        # after minimize the metal is restored; sphere holds donors
        sphere_donors = sorted({d for ds in ens.sphere.values() for d in ds}) or donors
        metals = set(_metal.metal_indices(ens.mol))
        print(f"\n[iso {k}] {iso.summary()}  kept={len(ens.ids)}  sphere={ens.sphere}")
        for cid in ens.ids:
            rep = geo.check(ens.mol, cid, donors=sphere_donors)
            kinds = Counter(v.kind for v in rep.violations)
            formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
            nconf += 1
            bad = (not rep.ok()) or formed or broken
            if bad:
                flagged += 1
            if formed:
                epoxide_confs += 1
            total.update(kinds)
            cc = (f" FORMED={formed}" if formed else "") + (f" BROKEN={broken}" if broken else "")
            print(f"   conf {cid}: geom_ok={rep.ok()} kinds={dict(kinds)}{cc}")
            if formed or broken:
                for v in rep.violations:
                    print(f"        {v}")
    print(f"\n== {case} POST-MIN: {flagged}/{nconf} flagged; {epoxide_confs} with a FORMED bond; kinds {dict(total)}")
    return flagged, nconf, epoxide_confs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--ablate", default="none")
    args = ap.parse_args()
    if args.ablate != "none":
        H.apply_ablation(args.ablate)
    cases = list(H.CASES) if args.case == "all" else [args.case]
    for c in cases:
        run(c, args.n, args.ablate)
