"""T2 claim-checks: the four named metal refcodes, the organic soft-window claim, square-planar sphere.

Run:  uv run python t2_claims.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.dirname(__file__))
import rxembed as rx
from rxembed import geometry as geo
from t2_seed_vs_relax import NCONF, SEED, TMQM, crystal, run

logging.disable(logging.WARNING)

print("=" * 78)
print("A. NAMED METAL CLAIMS: seed window violation, per the same harness as the 30")
print("   claimed: XAWQUH 48.2 deg, NEWVOB 30.3 deg, ZOPNOH 26.8 deg -> all 0.00 after relax")
print("   (the task text lists ZOPNOH at 30.3 and NEWVOB at 26.8 in one place; both are checked)")
print("=" * 78)
for nm in ["XAWQUH", "NEWVOB", "ZOPNOH", "TUXRUZ", "XAQDUS"]:
    try:
        r = run(nm)
    except Exception as e:
        print(f"{nm}: EXCEPTION {type(e).__name__}: {e}")
        continue
    if "skip" in r:
        print(f"{nm}: {r['skip']}")
        continue
    sa = [c["seed"]["awin_max"] for c in r["confs"]]
    ra = [c["relax"]["awin_max"] for c in r["confs"]]
    sd = [c["seed"]["dwin_max"] for c in r["confs"]]
    rd = [c["relax"]["dwin_max"] for c in r["confs"]]
    cry_a = r.get("crystal", {}).get("awin_max")
    cry_d = r.get("crystal", {}).get("dwin_max")
    print(
        f"{nm}: n={len(sa)} angle-window max  seed {max(sa):6.2f} (mean {np.mean(sa):5.2f})"
        f" -> relax {max(ra):5.2f} | dist-window max seed {max(sd):.3f} -> relax {max(rd):.3f}"
        f" | CRYSTAL itself: angle {cry_a:.2f} deg, dist {cry_d:.3f} A"
    )

print()
print("=" * 78)
print("B. ORGANIC SOFT-WINDOW CLAIM: constrain={(i,j):(2.5,3.0)} violated on the raw seed?")
print("   claimed: 0.318 A. The exact molecule is not recorded anywhere in the repo (grepped),")
print("   so the CLASS is tested on a fixed, documented SMILES list, pair (0,8), seed 0xF00D.")
print("=" * 78)
ORGANIC = [
    "CCCCCCCCCC",  # decane
    "OCCCCCCCCO",  # 1,9-nonanediol
    "c1ccccc1CCCCO",  # 4-phenylbutanol
    "CC(=O)OCCCCCCN",  # aminohexyl acetate
    "CCOC(=O)CCCCC(=O)O",  # monoethyl pimelate
    "NCCCCCCCCN",  # 1,8-diaminooctane
    "c1ccccc1Oc1ccccc1",  # diphenyl ether
    # "CC(C)CC(C)CC(C)CO" dropped: 2 undefined stereocentres -> rx.embed returns an EnsembleSet, not an Ensemble
]
for smi in ORGANIC:
    ens = rx.embed(smi, constrain={(0, 8): (2.5, 3.0)}, n=NCONF, seed=SEED)
    mol = ens.mol
    key = (0, 8) if (0, 8) in ens.cons.distances else None
    lo, hi = ens.cons.distances[key] if key else (2.5, 3.0)

    def viol(m, cid):
        p = m.GetConformer(cid).GetPositions()
        d = float(np.linalg.norm(p[0] - p[8]))
        return max(0.0, lo - d, d - hi), d

    seed_v = [viol(mol, c) for c in ens.ids]
    pre = {c: mol.GetConformer(c).GetPositions().copy() for c in ens.ids}
    ens.minimize()
    relax_v = [viol(ens.mol, c) for c in ens.ids]
    drift = float(
        np.mean(
            [np.sqrt(np.mean(np.sum((ens.mol.GetConformer(c).GetPositions() - pre[c]) ** 2, axis=1))) for c in ens.ids]
        )
    )
    print(
        f"  {smi:22s} window=({lo:.2f},{hi:.2f}) n={len(seed_v)} "
        f"seed viol max {max(v for v, _ in seed_v):.3f} mean {np.mean([v for v, _ in seed_v]):.3f} "
        f"-> relax max {max(v for v, _ in relax_v):.3f} | mean drift {drift:.2f} A"
    )

print()
print("=" * 78)
print("C. SQUARE-PLANAR CLAIM: '15/20 square-planar crystals look tetrahedral until .minimize()'")
print("   Sphere shape measured as the RMS out-of-plane distance of the 4 donors from their")
print("   best-fit plane through the metal. Square planar ~0.0 A; tetrahedral ~0.5-0.9 A.")
print("   Corpus: every CN=4 tmQM structure whose CRYSTAL sphere is planar (oop < 0.35 A).")
print("=" * 78)
import glob
import json


def sphere_oop(pos, m, donors):
    """RMS out-of-plane distance of the donors from the best-fit plane through the metal."""
    v = pos[list(donors)] - pos[m]
    u, s, vt = np.linalg.svd(v - v.mean(axis=0) * 0)
    n = vt[-1]
    return float(np.sqrt(np.mean((v @ n) ** 2)))


rows = json.load(open(os.path.join(os.path.dirname(__file__), "t2_survey.json")))
cn4 = [r["name"] for r in rows if r["cn"] == 4]
planar, results = [], []
for nm in cn4:
    path = os.path.join(TMQM, f"{nm}.xyz")
    sym, cry, q = crystal(path)
    try:
        ens = rx.embed(path, charge=q, n=NCONF, seed=SEED)
    except Exception as e:
        print(f"  {nm}: embed failed ({type(e).__name__})")
        continue
    if getattr(ens, "_metal", None) is None or not ens._metal.donors:
        continue
    m, don = ens._metal.metal, sorted(ens._metal.donors)
    if len(don) != 4:
        continue
    c_oop = sphere_oop(cry, m, don)
    if c_oop > 0.35:  # the crystal is not square planar — out of scope for this claim
        continue
    planar.append(nm)
    ids = list(ens.ids)
    seeds = ids[1:] if np.array_equal(ens.mol.GetConformer(ids[0]).GetPositions(), cry) else ids
    if not seeds:
        results.append((nm, c_oop, None, None, "FAILED EMBED (input only)"))
        continue
    s_oop = [sphere_oop(ens.mol.GetConformer(c).GetPositions(), m, don) for c in seeds]
    ens.minimize(_retry=False)
    keep = [c for c in ens.ids if c in seeds]
    r_oop = [sphere_oop(ens.mol.GetConformer(c).GetPositions(), m, don) for c in keep] or [float("nan")]
    results.append((nm, c_oop, float(np.median(s_oop)), float(np.median(r_oop)), ""))

print(f"  {len(planar)} of {len(cn4)} CN=4 tmQM structures have a planar crystal sphere")
print(f"  {'name':8s} {'crystal':>8s} {'seed':>8s} {'relax':>8s}   verdict")
tetra_seed = 0
for nm, c, s, r, note in results:
    if s is None:
        print(f"  {nm:8s} {c:8.3f}      --       --   {note}")
        continue
    v = "seed TETRAHEDRAL-ish" if s > 0.35 else "seed already planar"
    tetra_seed += s > 0.35
    print(f"  {nm:8s} {c:8.3f} {s:8.3f} {r:8.3f}   {v}")
print(f"  => {tetra_seed}/{len(planar)} planar-crystal spheres are non-planar (oop>0.35 A) on the RAW SEED")
