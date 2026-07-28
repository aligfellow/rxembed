"""Pin down the ester O2-C3-O4 collapse in case4: which seed, which stage, caught by any gate?

O2 and O4 are both bonded to C3 (a 1-3 pair). A collapse of the O-C-O angle fuses them (the "epoxide").
metrics.connectivity is blind (topo(O2,O4)=2 < _MIN_TOPO=3); clash is blind (1-3 pair excluded).
"""

from __future__ import annotations

import numpy as np

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import metal as _metal

import kdiag_harness as H

rx.set_verbose("CRITICAL")


def oco(pos):
    v1, v2 = pos[2] - pos[3], pos[4] - pos[3]
    a = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    return a, float(np.linalg.norm(pos[2] - pos[4]))


iso_set = rx.metal(H.CASES["case4"], "square_planar")
COLLAPSE = 90.0  # deg: below this the ester is folding toward the O-O fusion

print("Searching for the ester collapse (O2-C3-O4 < 90 deg) ...")
for k, iso in enumerate(iso_set):
    donors = list(iso.donors)
    for s in range(40):
        seed = 0xF00D + s
        # STAGE A: post embed-relax (surrogate live) -- restore for a clean read
        ens_e = rx.embed(iso, n=1, seed=seed)
        if not ens_e.ids:
            continue
        mc = ens_e._metal
        mol_e = H.restored_mol(ens_e.mol, mc)
        posE = mol_e.GetConformer(ens_e.ids[0]).GetPositions()
        angE, dE = oco(posE)
        # also the RAW seed (before the embed seam relax) via ablation copy
        # STAGE B: post minimize
        ens_m = rx.embed(iso, n=1, seed=seed).minimize()
        if not ens_m.ids:
            continue
        posM = ens_m.mol.GetConformer(ens_m.ids[0]).GetPositions()
        angM, dM = oco(posM)
        if angE < COLLAPSE or angM < COLLAPSE:
            sphere = sorted({d for ds in ens_m.sphere.values() for d in ds}) or donors
            repM = geo.check(ens_m.mol, ens_m.ids[0], donors=sphere)
            formed, broken = met.connectivity(ens_m.mol, ens_m.ids[0], metals=set(_metal.metal_indices(ens_m.mol)))
            print(f"\niso{k} seed{seed}: O-C-O  embed={angE:.1f}deg/{dE:.2f}A  min={angM:.1f}deg/{dM:.2f}A")
            print(f"   post-min geom.check ok={repM.ok()} kinds={[v.kind for v in repM.violations]}")
            print(f"   metrics.connectivity formed={formed} broken={broken}  (blind to O2-O4: topo=2)")
