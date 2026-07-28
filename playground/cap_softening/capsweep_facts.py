"""Structural facts: capped-donor count (crowding signal) + isomer count per case/control."""

from __future__ import annotations

import sys

sys.path.insert(0, "playground/karoline_diag")

import rxembed as rx
from rxembed.constraints import metal as M  # noqa: N812

import kdiag_harness as H

rx.set_verbose("CRITICAL")

HENRY = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
KETONE = "CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1"
# picolinate (pyridine-2-carboxylate) N,O chelate on Ni + ethylenediamine — a simple pyridine/carboxylate control
PICO = "O=C1[O-]->[Ni+2]2(<-[NH2]CC[NH2]->2)<-n2ccccc21"

ALL = {**H.CASES, "HENRY": HENRY, "KETONE": KETONE, "PICO": PICO}

for name, smi in ALL.items():
    geom = "square_planar"
    try:
        iso_set = rx.metal(smi, geom)
    except Exception as e:  # noqa: BLE001
        print(f"{name:8}: PARSE/METAL FAIL: {type(e).__name__}: {e}")
        continue
    iso0 = iso_set[0]
    n_capped = len({e[1] for e in iso0.cons.coplanar})
    donors = list(iso0.donors)
    syms = [iso0.mol.GetAtomWithIdx(d).GetSymbol() for d in donors]
    capped_donors = sorted({e[1] for e in iso0.cons.coplanar})
    print(
        f"{name:8}: isomers={len(iso_set):2}  donors={dict(zip(donors, syms))}  "
        f"capped_sp2_donors={n_capped} at {capped_donors}"
    )
