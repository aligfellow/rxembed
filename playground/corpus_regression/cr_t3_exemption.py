"""How much does T3's `bonding_ok(constrained=)` exemption WEAKEN the tear gate on this corpus?

T3 is described as two pieces (the embed relax seam + `_rescue_torn`), but it also carries a third,
quieter one: `metrics.bonding_ok` now skips any pair listed in `Constraints.distances`. The rationale
is sound for a user-requested dissociation (`fix={(1,2): 2.4}`). The risk is that a pair which is
*also a real bond* becomes invisible to the tear check — a bond could then break with no gate firing.

Measured here, per structure:
  exempt_pairs        pairs the exemption newly skips (metal pairs excluded — already skipped at HEAD)
  exempt_bonded       of those, pairs that are ACTUAL BONDS in the graph — the ones that can now tear
                      undetected
  tear_masked         conformers where HEAD's bonding_ok says BROKEN but the new one says fine

The third number is the one that matters: it is the count of geometries the new gate lets through
that the old gate caught.

Usage: PYTHONPATH=<tree3>/src uv run --no-sync python cr_t3_exemption.py <out.json>
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np

import rxembed as rx
from rxembed import metrics as _metrics
from rxembed.constraints import metal as _cmetal  # noqa: F401  (kept: import order matters for lazy load)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_corpus import CORPUS, crystal_full  # noqa: E402

_METAL_Z = _metrics._METAL_Z
SEED = 0xF00D


def run(name, path, half):
    _sym, _cry, q, _k = crystal_full(path)
    ens = rx.embed(path, charge=q, n=4, seed=SEED)
    if isinstance(ens, list) or hasattr(ens, "candidates") or ens._metal is None or not ens.ids:
        return {"name": name, "half": half, "skip": "not the single metal retain-input path"}
    mol, cons = ens.mol, ens.cons
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z}

    exempt = [p for p in cons.distances if not (set(p) & metals)]
    exempt_bonded = [p for p in exempt if frozenset(p) in bonded]

    # the decisive comparison: same geometry, gate WITH vs WITHOUT the exemption
    masked = 0
    for cid in ens.ids:
        strict = _metrics.bonding_ok(mol, cid, exclude=cons.frozen)
        lax = _metrics.bonding_ok(mol, cid, exclude=cons.frozen, constrained=cons.distances)
        if lax and not strict:
            masked += 1
    return {
        "name": name,
        "half": half,
        "n_dist_windows": len(cons.distances),
        "exempt_pairs": len(exempt),
        "exempt_bonded": len(exempt_bonded),
        "exempt_bonded_examples": [list(map(int, p)) for p in exempt_bonded[:5]],
        "n_conf": len(ens.ids),
        "tear_masked": masked,
    }


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    out = []
    for name, path, half in CORPUS:
        try:
            r = run(name, path, half)
        except Exception as e:
            r = {"name": name, "half": half, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(
            f"{name:24s} {half:8s} "
            + (
                r.get("skip")
                or r.get("error")
                or f"exempt={r['exempt_pairs']} bonded={r['exempt_bonded']} masked={r['tear_masked']}/{r['n_conf']}"
            ),
            flush=True,
        )
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    ok = [r for r in out if "exempt_pairs" in r]
    print("\n=== TOTALS ===")
    print("structures scored     :", len(ok))
    print("exempt pairs (total)  :", sum(r["exempt_pairs"] for r in ok))
    print("of which REAL BONDS   :", sum(r["exempt_bonded"] for r in ok))
    print(
        "structures w/ bonded  :",
        sum(1 for r in ok if r["exempt_bonded"]),
        [r["name"] for r in ok if r["exempt_bonded"]][:20],
    )
    print(
        "conformers TEAR-MASKED:",
        sum(r["tear_masked"] for r in ok),
        "in",
        sum(1 for r in ok if r["tear_masked"]),
        "structures",
        [r["name"] for r in ok if r["tear_masked"]][:20],
    )
