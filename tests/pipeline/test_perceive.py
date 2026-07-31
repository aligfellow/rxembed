"""`pipeline/perceive.py`: a user source (SMILES / .xyz path) becomes a Mol, or fails loudly."""

from importlib.util import find_spec

import pytest
from rdkit import Chem

from rxembed.pipeline.perceive import _xyz_to_mol, parse_smiles

_BIMP = "examples/structures/bimp.xyz"  # a metal-free TS with a stretched reacting core
_MN_H2 = "examples/structures/mn-h2.xyz"  # bimetallic: Mn centre + a spectator ferrocene


def test_a_bad_smiles_raises_instead_of_returning_the_none_rdkit_gives():
    """RDKit's None crashes three calls later with no mention of the input that caused it."""
    assert parse_smiles("CCO").GetNumAtoms() == 3
    with pytest.raises(ValueError, match="could not parse SMILES"):
        parse_smiles("C1CC")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_an_xyz_becomes_a_molecule_with_bond_orders_not_a_bag_of_atoms():
    """`build_graph(quick=True)` would return all-single bonds, and a TS core would lose every double bond."""
    mol = _xyz_to_mol(_BIMP, 0)
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().GetPositions().shape == (mol.GetNumAtoms(), 3)
    assert any(b.GetBondTypeAsDouble() > 1.0 for b in mol.GetBonds())


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_transition_metal_is_perceived_where_rdkits_own_perceiver_refuses():
    """Why perception is xyzgraph's job and not ours: `rdDetermineBonds` raises on the same file."""
    from rdkit.Chem import rdDetermineBonds

    mol = _xyz_to_mol(_MN_H2, 0)
    mn = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Mn")
    assert mol.GetAtomWithIdx(mn).GetDegree() > 0, "the metal came back with no ligands"

    with pytest.raises((ValueError, RuntimeError)):  # red-first: RDKit alone, same file
        rdDetermineBonds.DetermineBonds(Chem.MolFromXYZFile(_MN_H2), charge=0)
