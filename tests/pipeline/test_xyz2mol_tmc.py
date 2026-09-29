"""Test canonical ligand ordering and bond-order ranking in the TMC perceiver."""

from __future__ import annotations

import math
import random

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.pipeline import xyz2mol_tmc as tmc
from rxembed.pipeline.xyz2mol_tmc import get_lig_mol, get_tmc_mol


@pytest.mark.parametrize("element", ["Pb"])
def test_trimethylstannyl_platinum_trichloride_reads_a_closed_shell_stannyl_anion(element):
    """Every metal-ligand bond is cut before ligand perception, whatever the donor element.

    An element list had left the Pt-Sn bond uncut, so the stannyl never got ligand perception and read as a
    neutral Sn on Pt(I).
    """
    source = Chem.MolFromSmiles(f"C[{element}](C)(C)[Pt](Cl)(Cl)Cl", sanitize=False)
    source.UpdatePropertyCache(strict=False)
    source = Chem.AddHs(source)
    assert rdDistGeom.EmbedMolecule(source, randomSeed=7, useRandomCoords=True) == 0
    for atom in source.GetAtoms():
        atom.SetNoImplicit(True)

    out, _coords = get_tmc_mol(None, -2, graph=(source, source.GetConformer().GetPositions()))

    assert [(a.GetSymbol(), a.GetFormalCharge()) for a in out.GetAtoms() if a.GetAtomicNum() > 9] == [
        (element, -1),
        ("Pt", 2),
        *[("Cl", -1)] * 3,
    ]
    assert not any(a.GetNumRadicalElectrons() for a in out.GetAtoms())


def test_bis_silylamido_cyclopentadienyl_zirconium_chloride_reads_as_zirconium_four():
    """USUQAB's constrained-geometry ligand: a Cp ring with two Me2Si-NMe arms, all seven atoms on Zr.

    Each amido nitrogen is bonded only to saturated carbons and silicons with no lone pair to offer as
    a pi partner, so `_saturated_donor_charges` charges it before the native search runs. Charged up
    front, the native ladder reads the trianion (Cp- and two amides) that NBO gives, at Zr+4.
    """
    ligand = Chem.AddHs(Chem.MolFromSmiles("C[N-][Si](C)(C)C1=CC([Si](C)(C)[N-]C)=C[CH-]1"))
    assert rdDistGeom.EmbedMolecule(ligand, randomSeed=7) == 0
    rw = Chem.RWMol(ligand)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    donors = [atom.GetIdx() for atom in ligand.GetAtoms() if atom.IsInRing() or atom.GetAtomicNum() == 7]
    zirconium, chlorine = rw.AddAtom(Chem.Atom(40)), rw.AddAtom(Chem.Atom(17))
    for donor in [*donors, chlorine]:
        rw.AddBond(zirconium, donor, Chem.BondType.SINGLE)
    source = rw.GetMol()
    source.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(source.GetNumAtoms())  # Zr at the origin: a graph= read never uses metal positions
    for index, point in enumerate(ligand.GetConformer().GetPositions()):
        conf.SetAtomPosition(index, Point3D(*point))
    conf.SetAtomPosition(chlorine, Point3D(0, 0, 2.4))
    source.AddConformer(conf, assignId=True)

    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        perceived, _coords = get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))

        charges = {atom.GetSymbol(): 0 for atom in perceived.GetAtoms()}
        for atom in perceived.GetAtoms():
            charges[atom.GetSymbol()] += atom.GetFormalCharge()
        assert charges == {"Zr": 4, "Cl": -1, "N": -2, "C": -1, "Si": 0, "H": 0}


def test_rescued_ring_on_borole_fluoroborole_titanium_chloride_does_not_follow_atom_order():
    """Either borole dianion alone neutralised brings Ti+5 to Ti+3; which ring is chosen must not follow file order."""
    source, _coords = _bis_borole_titanium_trichloride(chlorides=1, lower_boron_substituent=9)
    readings = set()
    for seed in range(4):
        order = list(range(source.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        mol = Chem.RenumberAtoms(source, order)
        out, _coords = get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))
        assert out.GetProp("_rxembedChargeRescue").endswith("read Ti+3 with 1 ligand charge(s) changed")
        readings.add(Chem.MolToSmiles(out))
    assert len(readings) == 1


def test_cyclotriphosphine_ring_donor_stays_single_bonded(tmp_path):
    """A saturated P3H3 ring (cyclotriphosphine) donating to a metal through all three phosphines must keep
    single P-P bonds, not the P#P one RDKit's compiled search promotes without this fix.
    """
    pp, ph = 2.21, 1.42  # A: a typical P-P single bond and P-H bond
    radius = pp / math.sqrt(3)
    angles = (0, 120, 240)
    p_pos = [(radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), 0.0) for a in angles]
    scale = (radius + ph) / radius
    h_pos = [(x * scale, y * scale, z * scale) for x, y, z in p_pos]

    lines = ["7", "charge=0"]
    for symbol, pos in [
        *zip(("P", "P", "P"), p_pos, strict=True),
        *zip(("H", "H", "H"), h_pos, strict=True),
        ("Ni", (0.0, 0.0, 2.2)),
    ]:
        lines.append(f"{symbol} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    path = tmp_path / "cyclotriphosphine_nickel.xyz"
    path.write_text("\n".join(lines) + "\n")

    # Pin the metal's charge: three plain phosphine donors on Ni(0) is unambiguous, and skipping the
    # oxidation-state search keeps this test about the ring's bond orders, not that separate guess.
    mol = rx.read_xyz(str(path), charge=0, bond_orders="xyz2mol", metal_charges={6: 0})

    ring_bonds = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() < 3 and b.GetEndAtomIdx() < 3]
    assert len(ring_bonds) == 3
    assert not any(b.GetBondType() == Chem.BondType.TRIPLE for b in ring_bonds)


def _bis_borole_titanium_trichloride(chlorides=3, lower_boron_substituent=1):
    """Ti between two eta5-borole rings (B + 4 CH each, above and below) with `chlorides` in the waist.

    `lower_boron_substituent` is the atomic number on the lower ring's boron, in place of its hydrogen.
    """
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    points = {titanium: (0.0, 0.0, 0.0)}
    for side, substituent in ((1, 1), (-1, lower_boron_substituent)):
        ring = [rw.AddAtom(Chem.Atom(5))] + [rw.AddAtom(Chem.Atom(6)) for _ in range(4)]
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            rw.AddBond(a, b, Chem.BondType.SINGLE)
        for i, atom in enumerate(ring):
            angle = 2 * math.pi * i / len(ring)
            points[atom] = (1.2 * math.cos(angle), 1.2 * math.sin(angle), 2.0 * side)
            hydrogen = rw.AddAtom(Chem.Atom(substituent if i == 0 else 1))
            rw.AddBond(atom, hydrogen, Chem.BondType.SINGLE)
            points[hydrogen] = (2.2 * math.cos(angle), 2.2 * math.sin(angle), 2.3 * side)
            rw.AddBond(titanium, atom, Chem.BondType.SINGLE)
    for i in range(chlorides):
        chlorine = rw.AddAtom(Chem.Atom(17))
        rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
        points[chlorine] = (2.4 * math.cos(2 * math.pi * i / 3), 2.4 * math.sin(2 * math.pi * i / 3), 0.0)
    rw.UpdatePropertyCache(strict=False)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in points.items():
        conf.SetAtomPosition(index, Point3D(*point))
    mol.AddConformer(conf)
    return mol, mol.GetConformer().GetPositions()


def _flat_explicit_ligand(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def test_graph_equivalent_nitrogen_oxide_uses_the_electronic_charge_hint():
    mol = _flat_explicit_ligand("[O-][N+](=O)[O-]")
    donors = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]

    ligand, charge = tmc._fast_bond_orders(mol, -3, donors)
    nitrogen = next(atom for atom in ligand.GetAtoms() if atom.GetAtomicNum() == 7)

    assert charge == Chem.GetFormalCharge(ligand) == -3
    assert nitrogen.GetFormalCharge() == 0


def _c7h7_ring():
    """A bare eta7-cycloheptatrienyl ring: 7 CH, single bonds only, no charge or bond order proposed yet."""
    rw = Chem.RWMol()
    ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(7)]
    for atom in ring:
        rw.GetAtomWithIdx(atom).SetNoImplicit(True)
    for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    for carbon in ring:
        hydrogen = rw.AddAtom(Chem.Atom(1))
        rw.GetAtomWithIdx(hydrogen).SetNoImplicit(True)
        rw.AddBond(carbon, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, ring


@pytest.mark.parametrize("hint", [-1])
def test_ligand_charge_ladder_stays_bounded_regardless_of_hint_sign(hint):
    """Nothing should let the ladder settle for the fully-carbanion heptaanion over the textbook
    trianion just because a Hueckel hint landed on the wrong side of zero: the accepted ligand's own
    reported charge must match its atoms' formal-charge sum, and no atom should carry a charge no real
    Lewis structure would give a plain hydrocarbon donor.
    """
    mol, ring = _c7h7_ring()

    ligand, charge = get_lig_mol(mol, hint, ring)

    assert charge == Chem.GetFormalCharge(ligand)
    assert abs(charge) <= 3
    assert max(abs(atom.GetFormalCharge()) for atom in ligand.GetAtoms()) <= 2


@pytest.mark.parametrize(("distance", "expected_charge"), [(1.43, -2)])
def test_charge_hint_needs_bond_length_agreement(distance, expected_charge):
    mol = _flat_explicit_ligand("O=C(F)F")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((distance, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.66, 1.143, 0.0), (-0.66, -1.143, 0.0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    ligand, charge = tmc._fast_bond_orders(mol, -2, [0, 1])

    assert charge == expected_charge
    assert ligand.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == (2.0 if charge == 0 else 1.0)


def test_multimetal_reassembly_preserves_supplied_oxidation_states(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number, charge in ((26, 2), (25, 1), (17, -1), (17, -1)):
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    rw.AddBond(2, 0, Chem.BondType.DATIVE)
    rw.AddBond(3, 1, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    mol.AddConformer(Chem.Conformer(4))
    monkeypatch.setattr(tmc, "get_proposed_ligand_charge", Chem.GetFormalCharge)
    monkeypatch.setattr(tmc, "get_lig_mol", lambda ligand, charge, _donors: (ligand, charge))

    perceived, _coords = get_tmc_mol(None, 1, graph=(mol, mol.GetConformer().GetPositions()))

    assert {atom.GetAtomicNum(): atom.GetFormalCharge() for atom in perceived.GetAtoms() if atom.GetIdx() < 2} == {
        26: 2,
        25: 1,
    }
    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals"):
        get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))


def test_three_centre_borohydride_completes_a_missing_boron_hydrogen_edge():
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 1, 1, 1, 1):
        rw.AddAtom(Chem.Atom(atomic_number))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(0, 2, Chem.BondType.SINGLE)
    for hydrogen in (3, 4, 5):
        rw.AddBond(1, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()

    completed = tmc._reconnect_metal_hydride_bridges(
        mol,
        ((0, 0, 0), (2.076, 0, 0), (1.307, 1.087, 0), (3.1, 0, 0), (2.1, 1.1, 0), (2.1, -1.1, 0)),
    )

    assert completed.GetBondBetweenAtoms(1, 2) is not None
    assert completed.GetAtomWithIdx(1).GetFormalCharge() == -1


def test_ligand_checks_stream_resonance_candidates():
    """`lig_checks(resonate=True)` yields benzene's own aromatic form with no formal charges."""
    candidates = list(tmc.lig_checks(Chem.MolFromSmiles("c1ccccc1"), ()))

    assert len(candidates) == 1
    mol, positive, negative, n_aromatic, invented, pairless = candidates[0]
    assert (positive, negative, pairless) == (0, 0, 0)
    assert (n_aromatic, invented) == (6, 6)  # 6 aromatic carbons, each with one implicit ring H
    assert Chem.MolToSmiles(mol) == "c1ccccc1"
