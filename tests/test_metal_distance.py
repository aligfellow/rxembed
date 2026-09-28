"""Test metal-ligand seed distances and anti-overbond floors."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

import rxembed as rx
from rxembed import metal_distance
from rxembed.metal_donor_orient import stripped_hybridisation

# the N-bound Ni(II) linkage isomer, depe backbone: a P donor (capped), an anionic O and an amidate N
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
_OXYGEN, _MOLYBDENUM = 8, 42
# Pt(II) with a phosphine-tethered agostic C-H: Pt is inside the bridge census (`BRIDGE_H`)


def test_pnictogen_uses_dative_cap_not_halide():
    pt = GetPeriodicTable()
    iso = rx.metal(_NI_N, "square_planar")[0]
    q = metal_distance.delocalised_charges(iso.mol)
    p = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 15)
    r_ni, r_p = pt.GetRcovalent(28), pt.GetRcovalent(15)
    got = metal_distance.ml_distance(iso.mol, iso.metal, p, 28, set(iso.donors), q, hyb=stripped_hybridisation(iso.mol))
    assert got < r_ni + r_p, "the cap must CONTRACT the pnictogen, not lengthen it"
    # A neutral, non-haptic pnictogen contracts to 0.82x its covalent radius past the metal's own radius
    # (metal_distance.ml_distance's docstring names this fraction; ml_distance is the one place it is applied).
    assert got == pytest.approx(r_ni + 0.82 * r_p, abs=1e-9)

    pdcl = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]
    cl = next(d for d in pdcl.donors if pdcl.mol.GetAtomWithIdx(d).GetAtomicNum() == 17)
    got_cl = metal_distance.ml_distance(
        pdcl.mol,
        pdcl.metal,
        cl,
        46,
        set(pdcl.donors),
        metal_distance.delocalised_charges(pdcl.mol),
        hyb=stripped_hybridisation(pdcl.mol),
    )
    assert got_cl > got, "a halide was wrongly given the pnictogen's soft-donor contraction"


def test_coordinate_backed_chelate_windows_use_the_same_radial_policy():
    rw = Chem.RWMol()
    metal, n_left, carbon, n_right, chloride, bromide = (rw.AddAtom(Chem.Atom(z)) for z in (46, 7, 6, 7, 17, 35))
    rw.AddBond(n_left, carbon, Chem.BondType.SINGLE)
    rw.AddBond(carbon, n_right, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(((0, 0, 0), (2.1, 1, 0), (2.1, 0, 0), (2.1, -1, 0), (2.1, 0, 1), (2.1, 0, -1))):
        conf.SetAtomPosition(index, point)
    mol.AddConformer(conf)
    distances = {
        (metal, n_left): (2.1, 2.3),
        (metal, n_right): (2.1, 2.3),
        (metal, chloride): (2.1, 2.3),
        (metal, bromide): (2.1, 2.3),
    }
    cons = rx.Constraints(distances=distances)

    metal_distance.ff_terms(mol, cons, metal, 46, [n_left, n_right, chloride, bromide])

    assert (metal, n_left) not in cons.pulls
    assert (metal, n_right) not in cons.pulls
    assert cons.pulls[(metal, chloride)] == pytest.approx(2.2)
    assert cons.pulls[(metal, bromide)] == pytest.approx(2.2)
    mol.RemoveAllConformers()
    without_coordinates = rx.Constraints(distances=distances)
    metal_distance.ff_terms(mol, without_coordinates, metal, 46, [n_left, n_right, chloride, bromide])
    assert cons == without_coordinates
