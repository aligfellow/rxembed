"""T5 reproduction: measure henry thiosemicarbazone S6 metal-out-of-plane on the CURRENT tree."""

from __future__ import annotations
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import rxembed as rx
from rxembed import geometry as geo

SMI = "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
SEEDS = (1, 7, 13, 21, 42, 99, 123, 7777)


def planarity_of(mol, cid, m, d):
    """Metal-out-of-plane for donor d via geometry._planarity_dev, over its heavy substituents."""
    pos = mol.GetConformer(cid).GetPositions()
    hyb = geo._stripped_hybridisation(mol)
    best = None
    for x in mol.GetAtomWithIdx(d).GetNeighbors():
        if x.GetAtomicNum() <= 1 or x.GetAtomicNum() in geo._METAL_Z:
            continue
        pv = geo._planarity_dev(mol, pos, hyb, m, d, x.GetIdx())
        if pv is not None:
            best = pv if best is None else min(best, pv)
    return best


def main():
    es = rx.metal(SMI, "square_planar")
    iso0 = es[3]
    mol0 = iso0.mol
    s = next(d for d in iso0.donors if mol0.GetAtomWithIdx(d).GetSymbol() == "S")
    o = next(d for d in iso0.donors if mol0.GetAtomWithIdx(d).GetSymbol() == "O")
    print(f"S donor idx = {s}, O donor idx = {o}, metal = {iso0.metal}")
    print(f"iso[3] coplanar entries: {iso0.cons.coplanar}")
    hyb = geo._stripped_hybridisation(mol0)
    print(f"stripped hyb: S={hyb.get(s)}, O={hyb.get(o)}")

    s_oop, o_oop = [], []
    for seed in SEEDS:
        iso = rx.metal(SMI, "square_planar")[3]
        ens = rx.embed(iso, n=8, seed=seed).minimize()
        for cid in ens.ids:
            sv = planarity_of(ens.mol, cid, iso.metal, s)
            ov = planarity_of(ens.mol, cid, iso.metal, o)
            if sv is not None:
                s_oop.append(sv)
            if ov is not None:
                o_oop.append(ov)
    s_oop, o_oop = np.array(s_oop), np.array(o_oop)
    print(
        f"\nS6  metal-out-of-plane  n={len(s_oop)}: median={np.median(s_oop):.1f}  mean={s_oop.mean():.1f}  "
        f"max={s_oop.max():.1f}  p95={np.percentile(s_oop, 95):.1f}"
    )
    print(
        f"O   metal-out-of-plane  n={len(o_oop)}: median={np.median(o_oop):.1f}  mean={o_oop.mean():.1f}  "
        f"max={o_oop.max():.1f}  p95={np.percentile(o_oop, 95):.1f}"
    )


if __name__ == "__main__":
    main()
