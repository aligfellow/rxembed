"""Probe: does _CONJ_O_MDC_ANGLE (105,140) actually reach cons.angles, or is it a setdefault no-op?

_orient_donor (fold wall, runs first) sets (metal, O, C) for a calibrated ('O',SP2) donor; _coplanar_donor
then does setdefault on the SAME key. Check the realised value for henry (k1 carboxylate) and acac (k2)."""

from __future__ import annotations
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import rxembed as rx
from rxembed.constraints import metal as M

CASES = {
    "henry_k1_carboxylate": (
        "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
        "square_planar",
    ),
    "acac_k2": ("CC(=O)/C=C(\\C)[O-]", None),  # placeholder; use the golden acac below instead
}


def probe(label, smi, geom):
    isos = rx.metal(smi, geom) if geom else None
    if isos is None:
        return
    iso = isos[0]
    mol = iso.mol
    print(f"\n== {label} ==  metal={iso.metal}")
    print(f"  _CONJ_O_MDC_ANGLE constant = {M._CONJ_O_MDC_ANGLE}")
    for d in iso.donors:
        if mol.GetAtomWithIdx(d).GetSymbol() != "O":
            continue
        cs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(d).GetNeighbors() if nb.GetAtomicNum() > 1]
        for c in cs:
            key = (iso.metal, d, c)
            val = iso.cons.angles.get(key)
            is_mdc = val == M._CONJ_O_MDC_ANGLE
            print(
                f"  O{d}: angle{key} = {val}   ({'== _CONJ_O_MDC_ANGLE' if is_mdc else 'from fold-wall/other' if val else 'ABSENT'})"
            )


if __name__ == "__main__":
    probe("henry k1 carboxylate", CASES["henry_k1_carboxylate"][0], "square_planar")
    # acac from the golden fixture SMILES
    probe("acac_ni k2", "CC(=O)C=C(C)[O-].CC(=O)C=C(C)[O-].[Ni+2]", "square_planar")
