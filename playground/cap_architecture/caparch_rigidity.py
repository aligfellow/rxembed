"""Structural rigidity of each capped donor's chelate path — the candidate cap predicate.

For every capped donor D, on the metal-CUT graph (donors reach each other only via backbone):
  - nearest co-donor D' and the shortest backbone path D..D'
  - whether that path is RIGID: every bond on it is a ring bond OR non-single (double/aromatic)
    i.e. contains NO freely-rotatable single bond -> the donor's plane is locked by the backbone.

Prediction: RIGID chelate path  -> cap REDUNDANT (plane already fixed).
            no co-donor / FLEXIBLE path -> cap LOAD-BEARING (roll about M-D is free).

Usage: uv run --no-sync python playground/cap_architecture/caparch_rigidity.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")

from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import metal as M  # noqa: N812

import capsweep_lib as L  # noqa: E402
import kdiag_harness as H  # noqa: E402

rx.set_verbose("CRITICAL")

COMPLEXES = {
    "case1": H.CASES["case1"], "case2": H.CASES["case2"], "case3": H.CASES["case3"],
    "case4": H.CASES["case4"], "HENRY": L.HENRY, "KETONE": L.KETONE, "PICO": L.PICO,
}  # fmt: skip


def cut_mol(mol, metal):
    rw = Chem.RWMol(Chem.Mol(mol))
    for nb in [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()]:
        if rw.GetBondBetweenAtoms(metal, nb) is not None:
            rw.RemoveBond(metal, nb)
    m2 = rw.GetMol()
    m2.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m2)
    return m2


def shortest_path(m2, a, b):
    try:
        return list(Chem.GetShortestPath(m2, int(a), int(b)))
    except Exception:
        return []


def path_is_rigid(m2, path, hyb):
    """True if every bond along `path` is a RIGID link: aromatic, double, or a single bond between two sp2 atoms.

    A single bond between two sp2 atoms is a conjugated (partial-double) link that does NOT freely rotate the
    donor's plane (aryl-carboxyl, C(sp2)-C(sp2)); a single bond touching an sp3 atom is a free hinge.
    """
    SP2 = Chem.HybridizationType.SP2  # noqa: N806
    if len(path) < 2:
        return True
    for u, v in zip(path, path[1:]):
        b = m2.GetBondBetweenAtoms(u, v)
        if b is None:
            return False
        if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.SINGLE:
            continue  # aromatic / double: rigid
        if hyb.get(u) == SP2 and hyb.get(v) == SP2:
            continue  # conjugated sp2-sp2 single bond: planar, rigid
        return False  # a single bond with an sp3 end: a free hinge
    return True


for name, smi in COMPLEXES.items():
    iso = rx.metal(smi, "square_planar")[0]
    mol, metal, donors = iso.mol, iso.metal, list(iso.donors)
    m2 = cut_mol(mol, metal)
    hyb = geo._stripped_hybridisation(mol)
    print(f"\n===== {name} =====  donors={donors}")
    for i, d, k, w, _anchor, _cap in iso.cons.coplanar:
        sym = mol.GetAtomWithIdx(d).GetSymbol()
        # nearest co-donor by backbone
        best = None
        for dd in donors:
            if dd == d:
                continue
            p = shortest_path(m2, d, dd)
            if p and (best is None or len(p) < len(best[1])):
                best = (dd, p)
        if best is None:
            print(f"   D={d}({sym}): NO backbone co-donor (monodentate) -> LOAD-BEARING predicted")
            continue
        dd, p = best
        rigid = path_is_rigid(m2, p, hyb)
        pred = "REDUNDANT" if rigid else "LOAD-BEARING"
        print(f"   D={d}({sym}) chelates D'={dd} path={p} rigid={rigid} -> {pred} predicted")
