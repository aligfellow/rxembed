"""Where CAN T4 fire? Reachability of the dg_floors relief, and the maintainer's own evidence.

The corpus protocol (`rx.embed(xyz, charge=q)`) never passes `coordinate=`, so `metal.coordinate()`
is never called and T4's three parts are all unreachable — measured, 0/142. This script asks the
follow-up question the corpus cannot: when the seating path IS taken, does the relief fire, and does
`min` vs `max` on dg_floors change the geometry?

Part A  re-run the maintainer's own case (`OCCCN->[Pd](Cl)Cl`, alkoxide O seated) on this tree and
        report the pre-relax M...C distance and M-O-C angle the docstring quotes.
Part B  sweep the corpus with `coordinate="auto"`, counting structures with a VACANCY (the only ones
        where seating does anything) and the dg_floors collisions that result.

Usage: PYTHONPATH=<tree>/src uv run --no-sync python cr_t4_domain.py <out.json>
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys

import numpy as np

import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import base as cbase

importlib.import_module("rxembed.embed.dispatch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_corpus import CORPUS, crystal_full  # noqa: E402

STATS = {"compose_calls": 0, "dgf_collisions": 0, "dgf_min_ne_max": 0, "coordinate_calls": 0, "max_gap": 0.0}
_orig_compose = cbase.compose


def _spy_compose(*parts):
    STATS["compose_calls"] += 1
    seen = {}
    for p in parts:
        for k, v in (getattr(p, "dg_floors", None) or {}).items():
            if k in seen:
                STATS["dgf_collisions"] += 1
                if abs(seen[k] - v) > 1e-9:
                    STATS["dgf_min_ne_max"] += 1
                    STATS["max_gap"] = max(STATS["max_gap"], abs(seen[k] - v))
            seen[k] = v
    return _orig_compose(*parts)


_patched = [
    n for n, m in list(sys.modules.items()) if n.startswith("rxembed") and getattr(m, "compose", None) is _orig_compose
]
for n in _patched:
    sys.modules[n].compose = _spy_compose
assert "rxembed.embed.dispatch" in _patched, f"spy not in dispatch: {_patched}"

from rxembed.constraints import metal as _metal  # noqa: E402

_orig_coord = _metal.coordinate


def _spy_coord(iso, atoms, **k):
    STATS["coordinate_calls"] += 1
    c = _orig_coord(iso, atoms, **k)
    STATS.setdefault("coord_dgf_emitted", 0)
    STATS["coord_dgf_emitted"] += len(getattr(c, "dg_floors", None) or {})
    return c


_metal.coordinate = _spy_coord


def part_a():
    """The docstring's own case: seat an alkoxide O on OCCCN->[Pd](Cl)Cl, report Pd...C and Pd-O-C."""
    out = {}
    for smi, label in (("OCCCN->[Pd](Cl)Cl", "alkoxide_Pd"),):
        for coord in (0,):  # atom 0 is the alkoxide O
            before = dict(STATS)
            try:
                # coordinate= needs the metal geometry declared on the same call (rx.metal returns an
                # IsomerSet, which embed rejects here)
                ens = rx.embed(smi, metal="square_planar", coordinate=coord, n=6, seed=0xF00D)
                if isinstance(ens, list) or hasattr(ens, "candidates"):
                    ens = ens[0]
                mol, m = ens.mol, ens._metal.metal
                # Pd...C over the alkoxide's carbon neighbour, and the Pd-O-C angle
                o = coord
                nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(o).GetNeighbors() if a.GetAtomicNum() == 6]
                rows = []
                for cid in ens.ids:
                    p = mol.GetConformer(cid).GetPositions()
                    for c_ in nbrs:
                        rows.append(
                            {
                                "cid": int(cid),
                                "M_C": float(np.linalg.norm(p[m] - p[c_])),
                                "M_O_C": float(geo._angle(p[m], p[o], p[c_])),
                            }
                        )
                out[label] = {
                    "n": len(ens.ids),
                    "M_C_mean": float(np.mean([r["M_C"] for r in rows])) if rows else None,
                    "M_C_min": float(np.min([r["M_C"] for r in rows])) if rows else None,
                    "M_O_C_mean": float(np.mean([r["M_O_C"] for r in rows])) if rows else None,
                    "rows": rows,
                    "coord_calls": STATS["coordinate_calls"] - before["coordinate_calls"],
                    "collisions": STATS["dgf_collisions"] - before["dgf_collisions"],
                    "min_ne_max": STATS["dgf_min_ne_max"] - before["dgf_min_ne_max"],
                    "dgf_emitted": STATS.get("coord_dgf_emitted", 0) - before.get("coord_dgf_emitted", 0),
                }
            except Exception as e:
                out[label] = {"error": f"{type(e).__name__}: {e}"}
    return out


def part_b():
    """Sweep the corpus with coordinate='auto' — the only corpus-shaped way to reach the seating path."""
    rows = []
    for name, path, half in CORPUS:
        before = dict(STATS)
        _sym, _cry, q, _k = crystal_full(path)
        try:
            ens = rx.embed(path, charge=q, n=2, seed=0xF00D, coordinate="auto")
            n = len(getattr(ens, "ids", []) or []) if not isinstance(ens, list) else -1
            err = None
        except Exception as e:
            n, err = 0, f"{type(e).__name__}: {e}"
        rows.append(
            {
                "name": name,
                "half": half,
                "n": n,
                "err": err,
                "coord_calls": STATS["coordinate_calls"] - before["coordinate_calls"],
                "dgf_emitted": STATS.get("coord_dgf_emitted", 0) - before.get("coord_dgf_emitted", 0),
                "collisions": STATS["dgf_collisions"] - before["dgf_collisions"],
                "min_ne_max": STATS["dgf_min_ne_max"] - before["dgf_min_ne_max"],
            }
        )
        print(
            f"{name:24s} {half:8s} n={n} coord={rows[-1]['coord_calls']} "
            f"dgf={rows[-1]['dgf_emitted']} collide={rows[-1]['collisions']} "
            f"min!=max={rows[-1]['min_ne_max']}" + (f" ERR {err[:50]}" if err else ""),
            flush=True,
        )
    return rows


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    res = {"part_a": part_a()}
    print("PART A:", json.dumps(res["part_a"], indent=1)[:1500], flush=True)
    res["part_b"] = part_b()
    res["totals"] = STATS
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("\n=== TOTALS ===", json.dumps(STATS, indent=1))
