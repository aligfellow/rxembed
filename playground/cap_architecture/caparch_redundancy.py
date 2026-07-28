"""Decisive per-donor redundancy test for the metal coplanarity cap.

For each complex and each capped donor D, measure the metal-out-of-D's-plane deviation
(deg, via the donor's OWN cap plane k,w) post-minimize under three constraint regimes:
  - BASE   : all caps on (baseline)
  - SELF   : remove ONLY D's cap (keep every other donor's cap + polyhedron + backbone)
  - ALL    : remove every cap

If SELF stays as low as BASE, D's plane is held by the backbone+polyhedron WITHOUT its cap
-> the cap is REDUNDANT on D.  If SELF drifts up toward ALL, the cap is LOAD-BEARING on D.

Removal patches BOTH halves of the cap (Coplanar.dg_post + Coplanar.ff_terms) by filtering
cons.coplanar on a module-level SKIP set.  Null-measurement guard: prints the FF torsion count.

Usage: uv run --no-sync python playground/cap_architecture/caparch_redundancy.py [complex ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")

import rxembed as rx
from rxembed.constraints import mechanisms as mech
from rxembed.constraints import metal as M  # noqa: N812

import capsweep_lib as L  # noqa: E402
import kdiag_harness as H  # noqa: E402

rx.set_verbose("CRITICAL")

COMPLEXES = {
    "case1": H.CASES["case1"],
    "case2": H.CASES["case2"],
    "case3": H.CASES["case3"],
    "case4": H.CASES["case4"],
    "HENRY": L.HENRY,
    "KETONE": L.KETONE,
    "PICO": L.PICO,
}

SKIP: set[int] = set()  # donor indices whose cap is removed (both DG + FF halves)
_FF_COUNT = [0]

_orig_ff = mech.Coplanar.ff_terms
_orig_dg = mech.Coplanar.dg_post


def _patched_ff(self, ff_obj, cons, conf, fc_dist):
    from rdkit.Chem import rdMolTransforms as T

    from rxembed.constraints.mechanisms import _COPLANAR_FC, _coplanar_window

    for i, j, k, w, _anchor, cap in cons.coplanar:
        if j in SKIP:
            continue
        _FF_COUNT[0] += 1
        phi = T.GetDihedralDeg(conf, i, j, k, w)
        lo, hi = _coplanar_window(phi, cap)
        ff_obj.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC)


def _patched_dg(self, cons, ctx):
    kept = [e for e in cons.coplanar if e[1] not in SKIP]
    saved = cons.coplanar
    cons.coplanar = kept
    try:
        _orig_dg(self, cons, ctx)
    finally:
        cons.coplanar = saved


mech.Coplanar.ff_terms = _patched_ff
mech.Coplanar.dg_post = _patched_dg


def oop(mol, cid, i, j, k, w):
    """Angle (deg) of metal i out of the plane through donor-substituents j,k,w (0 = coplanar)."""
    p = mol.GetConformer(cid).GetPositions()
    nrm = np.cross(p[k] - p[j], p[w] - p[j])
    nn = np.linalg.norm(nrm)
    if nn < 1e-6:
        return 0.0
    nrm /= nn
    v = p[i] - p[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(nrm, v))))))


def measure(smi, n=8, seeds=(1, 7, 13, 21, 0xF00D), max_iso=1):
    """{donor: {regime: [oop...]}} across isomers/seeds/conformers."""
    iso_set = rx.metal(smi, "square_planar")
    caps = iso_set[0].cons.coplanar
    donor_planes = {e[1]: (e[0], e[2], e[3]) for e in caps}  # donor -> (metal, k, w)
    donors = list(donor_planes)
    out = defaultdict(lambda: defaultdict(list))
    regimes = {"BASE": set(), "ALL": set(donors)}
    for d in donors:
        regimes[f"SELF:{d}"] = {d}
    for regime, skip in regimes.items():
        SKIP.clear()
        SKIP.update(skip)
        for iso in list(iso_set)[:max_iso]:
            # only measure planes on the isomer that actually has these donors capped
            iso_caps = {e[1]: (e[0], e[2], e[3]) for e in iso.cons.coplanar}
            for seed in seeds:
                ens = rx.embed(iso, n=n, seed=seed).minimize()
                for cid in ens.ids:
                    for d, (mi, k, w) in iso_caps.items():
                        out[d][regime].append(oop(ens.mol, cid, mi, d, k, w))
    SKIP.clear()
    return out, donor_planes


def stat(vals):
    if not vals:
        return "  n/a "
    a = np.array(vals)
    return f"{np.median(a):5.1f}/{a.max():5.1f}"


if __name__ == "__main__":
    which = sys.argv[1:] or list(COMPLEXES)
    for name in which:
        smi = COMPLEXES[name]
        out, planes = measure(smi)
        print(f"\n===== {name} =====  (median/max metal-out-of-donor-plane, deg)")
        print(f"   {'donor':>16} | {'BASE':>11} | {'SELF-off':>11} | {'ALL-off':>11} | verdict")
        for d in planes:
            sym = rx.metal(smi, "square_planar")[0].mol.GetAtomWithIdx(d).GetSymbol()
            base = out[d]["BASE"]
            self_ = out[d][f"SELF:{d}"]
            allc = out[d]["ALL"]
            bm = np.median(base) if base else float("nan")
            sm = np.median(self_) if self_ else float("nan")
            am = np.median(allc) if allc else float("nan")
            # redundant if removing D's own cap barely changes its plane vs baseline
            drift = sm - bm
            verdict = "REDUNDANT" if drift < 5.0 else ("load-bearing" if drift > 12.0 else "partial")
            print(
                f"   D={d:>3}({sym})        | {stat(base):>11} | {stat(self_):>11} | "
                f"{stat(allc):>11} | Δself={drift:+5.1f}  {verdict}"
            )
    print(f"\n[guard] total FF torsion terms written across all runs: {_FF_COUNT[0]} (>0 confirms path live)")
