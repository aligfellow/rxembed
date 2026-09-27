"""Build chemistry fixtures shared by metal-module tests."""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx

# Each amine N carries one acid and one carboxylate arm, and only one carboxylate binds Pt(II). Cutting the metal
# leaves a C2-symmetric ligand; the complex has no such symmetry, so its four bound-N hand pairs are distinct.
ONE_ARM_BOUND_PT = "OC(=O)CN1(CC(=O)[O-]->[Pt+2]12<-[Cl-])CCN->2(CC(=O)O)CC(=O)[O-]"
# (Buta-1,3-diene)Fe(CO)3 and its isoprene analogue: an eta4 diene whose central bond rotates when free, so each
# bound s-cis or s-trans form is an isomer. Isoprene's methyl makes its two s-cis faces distinct.
BUTADIENE_FE_CO3 = "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[CH]->1[CH]->2=[CH2]->3"
ISOPRENE_FE_CO3 = "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[C](C)->1[CH]->2=[CH2]->3"


def ferrocene():
    """Build two Cp anions datively bound to Fe(II) with an eta5 sandwich geometry."""
    rw = Chem.RWMol()
    iron = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(iron).SetFormalCharge(2)
    rings = []
    for _ in range(2):
        ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
        for k, bond in enumerate([Chem.BondType.SINGLE, Chem.BondType.DOUBLE] * 2 + [Chem.BondType.SINGLE]):
            rw.AddBond(ring[k], ring[(k + 1) % 5], bond)
        rw.GetAtomWithIdx(ring[0]).SetFormalCharge(-1)
        for carbon in ring:
            rw.AddBond(iron, carbon, Chem.BondType.DATIVE)
        rings.append(ring)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(iron, Point3D(0, 0, 0))
    for face, ring in enumerate(rings):
        z = 1.66 if face == 0 else -1.66
        for k, carbon in enumerate(ring):
            angle = 2 * np.pi * k / 5 + (0.2 if face else 0.0)
            conf.SetAtomPosition(carbon, Point3D(1.21 * np.cos(angle), 1.21 * np.sin(angle), z))
    mol.AddConformer(conf)
    return mol


def one_arm_bound_pt():
    """Embed `ONE_ARM_BOUND_PT` once: a geometry input whose two bound amine N are coordination-locked."""
    return rx.embed(rx.metal(ONE_ARM_BOUND_PT, "SPL")[0], n=1, seed=7).mol
