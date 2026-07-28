"""Direct epoxide probe for case4: measure the C1-O2-C3 angle / C1..C3 distance and raw-perceive triangles.

The suspected epoxide is a C1-C3 fusion across ether O2 (a 1-3 pair). metrics.connectivity CANNOT see it
(_MIN_TOPO=3). So we measure geometry directly and re-perceive with raw xyzgraph (which the k-file scan uses).
"""

from __future__ import annotations

import tempfile

import networkx as nx
import numpy as np
import xyzgraph

import rxembed as rx
from rxembed.constraints import metal as _metal

import kdiag_harness as H

rx.set_verbose("CRITICAL")

smi = H.CASES["case4"]
iso_set = rx.metal(smi, "square_planar")

print("case4: probing C1(idx1)-O2(idx2)-C3(idx3) for oxirane fusion")
print("seed  iso  stage         d(C1,C3)  angle(C1-O2-C3)  raw_triangles  3ring_has_{1,3}?")
worst = []
for k, iso in enumerate(iso_set):
    for s in range(30):
        seed = 0xF00D + s
        ens_e = rx.embed(iso, n=1, seed=seed)  # post embed-relax (surrogate still live)
        ens = rx.embed(iso, n=1, seed=seed).minimize()  # notebook stage
        for label, e in (("post-min", ens),):
            if not e.ids:
                continue
            cid = e.ids[0]
            pos = e.mol.GetConformer(cid).GetPositions()
            d13 = float(np.linalg.norm(pos[1] - pos[3]))
            v1 = pos[1] - pos[2]
            v3 = pos[3] - pos[2]
            ang = np.degrees(np.arccos(np.clip(v1.dot(v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)), -1, 1)))
            # raw perception via a dumped xyz (real metal), triangle detection
            p = tempfile.mktemp(suffix=".xyz")
            e.dump(p)
            g = xyzgraph.build_graph(p, quick=True)
            tris = [c for c in nx.cycle_basis(g) if len(c) == 3]
            has13 = any({1, 3} <= set(c) for c in tris)
            flag = "  <== OXIRANE" if has13 or d13 < 1.9 else ""
            if has13 or d13 < 2.1:
                print(f"{seed:>5} {k:>3}  {label:12} {d13:8.3f}  {ang:14.1f}   {len(tris):>3}          {has13}{flag}")
                worst.append((d13, seed, k, has13))

if worst:
    worst.sort()
    print(f"\nMIN d(C1,C3) seen: {worst[0][0]:.3f} A at seed {worst[0][1]} iso {worst[0][2]} (oxirane={worst[0][3]})")
else:
    print("\nno conformer with d(C1,C3) < 2.1 A across 120 runs — no oxirane fusion reproduced")
