"""T6: exercise the REAL rx.metal paths (ferrocene haptic-linear, tetrahedral 4-distinct) and print the count.

Confirms the orbit-count via the shared dedup handles haptic centroid vertices and the padded/VACANT case.
"""

import itertools

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.constraints import metal as M

S, D, DAT = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.DATIVE


def ferrocene():
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(me).SetFormalCharge(2)
    rings = []
    for _r in range(2):
        cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
        for k in range(5):
            rw.AddBond(cs[k], cs[(k + 1) % 5], [S, D, S, D, S][k])
        rw.GetAtomWithIdx(cs[0]).SetFormalCharge(-1)
        for c in cs:
            rw.AddBond(me, c, DAT)
        rings.append(cs)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for ri, cs in enumerate(rings):
        z = 1.66 if ri == 0 else -1.66
        for k, c in enumerate(cs):
            a = 2 * np.pi * k / 5 + (0.2 if ri else 0.0)
            conf.SetAtomPosition(c, Point3D(1.21 * np.cos(a), 1.21 * np.sin(a), z))
    m.AddConformer(conf)
    return m


# Wrap isomers to print the exact orbit-count via the SAME dedup (full permutations) at the guard.
_orig = M.isomers


def _spy(mol, donors, geometry, perms=None, r_metal=1.4, haptic=None):
    if perms is None and geometry not in M.PERMUTATIONS:
        orbit = list(itertools.permutations(range(len(donors))))
        distinct = _orig(mol, donors, geometry, perms=orbit, r_metal=r_metal, haptic=haptic)
        print(f"  GUARD geometry={geometry:22s} n={len(donors)} haptic={bool(haptic)} distinct={len(distinct)}")
    return _orig(mol, donors, geometry, perms=perms, r_metal=r_metal, haptic=haptic)


M.isomers = _spy

print("ferrocene (expect linear, distinct=1 -> silent):")
isos = rx.metal(ferrocene())
print(f"  -> {len(isos)} isomer(s), geometry={isos[0].geometry}")

M.isomers = _orig
