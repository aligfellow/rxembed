"""`utils.py`: the RDKit facts every layer shares, a ranking and a bond edit that keep what a graph means."""

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed.identity.smiles_io import parse_smiles
from rxembed.utils import bond_removal_mirrors, conjugated_quartets, flat_ranks, remove_bond


def test_flat_ranks_ignore_where_charge_or_bond_order_is_drawn():
    """Acac's -1 written on one oxygen ranks neither apart; a map ranks atoms only when asked, ties break when asked."""
    for name, smiles, options, atoms, alike in (
        ("acac oxygens", r"CC(=O)/C=C(\C)[O-]", {}, (2, 6), True),
        ("mapped ethane", "[CH3:1][CH3:2]", {}, (0, 1), True),
        ("ranked by map", "[CH3:1][CH3:2]", {"maps": True}, (0, 1), False),
        ("ties broken", "CC", {"break_ties": True}, (0, 1), False),
    ):
        ranks = flat_ranks(Chem.AddHs(Chem.MolFromSmiles(smiles)), **options)
        assert (ranks[atoms[0]] == ranks[atoms[1]]) is alike, name


def test_bond_removal_mirrors_by_slot_parity_at_given_degrees():
    """Removing slot p of n bonds costs n - 1 - p swaps; a degree-five P has no tetrahedral parity to keep."""
    for smiles, partner, degrees, mirrors in (
        ("F[C@](Cl)(Br)I", 0, (4,), True),
        ("F[C@](Cl)(Br)I", 2, (4,), False),
        ("F[P@](Cl)Br", 2, (4,), False),
        ("F[P@](Cl)Br", 2, (3, 4), True),
        ("F[P@](Cl)(Br)(I)F", 2, (4,), False),
    ):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        assert bond_removal_mirrors(mol.GetAtomWithIdx(1), partner, degrees) is mirrors, (smiles, partner, degrees)


def test_remove_bond_keeps_what_a_tetrahedral_or_ez_tag_means():
    """The ligand left is the free one with a lone pair where Pt was, in any bond order; an E/Z drawn against Pt moves
    to the other substituent of its end, or is dropped where there is none."""
    for name, smiles, bond, free in (
        ("P drawn second", "C[P@@](->[Pt])(CC)c1ccccc1", (1, 2), "C[P@@](CC)c1ccccc1.[Pt]"),
        ("P drawn third", "C[P@](CC)(->[Pt])c1ccccc1", (1, 4), "C[P@@](CC)c1ccccc1.[Pt]"),
        ("imine against Pt", r"C/C=N(/[Pt])C", (2, 3), r"C/C=N\C.[Pt]"),
        ("imine N bare", r"C/C=N/[Pt]", (2, 3), "CC=N.[Pt]"),
    ):
        rw = Chem.RWMol(Chem.MolFromSmiles(smiles))

        remove_bond(rw, *bond)

        rw.UpdatePropertyCache(strict=False)
        assert Chem.MolToSmiles(rw) == Chem.MolToSmiles(Chem.MolFromSmiles(free)), name


def test_without_zero_bonds_drops_only_zero_order_contacts():
    """An O...N contact goes; the stereocentre, the silver dative, the maps and the conformer stay, the input too."""
    molecule = Chem.AddHs(Chem.MolFromSmiles("F[C@H](O)C=O.[NH3]->[Ag+]"))
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetPositions(np.arange(molecule.GetNumAtoms() * 3, dtype=float).reshape(-1, 3))
    molecule.AddConformer(conformer)
    contacted = Chem.RWMol(molecule)
    contacted.AddBond(2, 5, Chem.BondType.ZERO)
    before = Chem.MolsToJSON([contacted])

    chemical = rx.without_zero_bonds(contacted)

    assert Chem.MolToSmiles(chemical, canonical=False) == Chem.MolToSmiles(molecule, canonical=False)
    np.testing.assert_array_equal(chemical.GetConformer().GetPositions(), molecule.GetConformer().GetPositions())
    assert Chem.MolsToJSON([contacted]) == before
    assert rx.without_zero_bonds(molecule) is molecule


def test_conjugated_quartets_are_open_amide_or_ester_planes():
    """A lactam's ring bond, a thioester's S, a dative N-Pd substituent and an excluded atom give none."""
    quartets = {  # (SMILES, excluded atoms) -> its quartets
        ("CC(=O)NC", ()): [(2, 1, 3, 4)],
        ("CC(=O)OC", ()): [(2, 1, 3, 4)],
        ("CC(=O)NC", (3,)): [],
        ("O=C1NCCC1", ()): [],
        ("CC(=O)SC", ()): [],
        ("CC(=O)[NH]->[Pd+2]", ()): [],
    }
    assert {key: list(conjugated_quartets(parse_smiles(key[0]), key[1])) for key in quartets} == quartets
