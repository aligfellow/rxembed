"""Does the DG bounds matrix already subsume the coplanarity cap for a conjugated bidentate?

Instruments Coplanar.dg_post: for each capped donor, reports the pre-existing matrix window on the cap's
M...w 1,4 pair BEFORE the cap tightens it.  A TIGHT window == the rigid backbone already pinned that
distance (cap subsumed / composes); an UNBOUNDED window == the matrix says nothing, so the cap is the sole
information (load-bearing).  Substantiates the 'full embedding in the DG, FF after' architecture.

Usage: uv run --no-sync python playground/cap_architecture/caparch_dgwindow.py [case]
"""

from __future__ import annotations

import sys

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")
sys.path.insert(0, "playground/cap_architecture")

import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import mechanisms as mech

import caparch_spike as S  # noqa: E402
import kdiag_harness as H  # noqa: E402

rx.set_verbose("CRITICAL")

case = sys.argv[1] if len(sys.argv) > 1 else "case2"
orig = mech.Coplanar.dg_post
rows = []


def spy(self, cons, ctx):
    hyb = geo._stripped_hybridisation(ctx.mol)
    donors = set()
    for a, b in cons.distances:
        if a in cons.metals:
            donors.add(b)
        elif b in cons.metals:
            donors.add(a)
    for i, _j, _k, w, _anchor, _cap in cons.coplanar:
        a, b = min(i, w), max(i, w)
        rows.append((_j, S.plane_locked(ctx.mol, _j, donors, hyb), round(ctx.bm[b][a], 2), round(ctx.bm[a][b], 2)))
    return orig(self, cons, ctx)


mech.Coplanar.dg_post = spy
iso = rx.metal(H.CASES[case], "square_planar")[0]
rx.embed(iso, n=1, seed=7)
print(f"{case} dg_post: pre-existing matrix window on the cap's M...w pair (Angstrom)")
for j, locked, lo, hi in sorted(set(rows)):
    tag = "conjugated-bidentate -> backbone pins it, cap SUBSUMED" if locked else "one-contact -> cap is the SOLE info"
    print(f"   D={j:>3}  window=[{lo}, {hi}]  width={round(hi - lo, 2)}   {tag}")
mech.Coplanar.dg_post = orig
