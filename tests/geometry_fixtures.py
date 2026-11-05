"""Build the small molecules the geometry tests embed or place."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


def embedded(smiles, seed=7):
    """Return `smiles` with explicit H and one RDKit-embedded conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def placed(mol, points):
    """Return `mol`, or a SMILES as drawn, with one conformer at `points`, one row per atom."""
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetPositions(np.array(points, dtype=float))
    mol.AddConformer(conformer)
    return mol


def mirrored(mol):
    """Return a copy of `mol` with its conformer reflected through the yz plane."""
    mirror = Chem.Mol(mol)
    mirror.GetConformer().SetPositions(mirror.GetConformer().GetPositions() * [-1, 1, 1])
    return mirror
