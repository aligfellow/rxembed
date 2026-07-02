"""Shared fixtures: RDKit embeds and availability skips for xtb / openconf."""

import os
import shutil

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


def rdkit_embed(smiles, seed=1, optimize=True):
    """A clean reference conformer via plain ETKDG (+ optional MMFF) — no rxembed needed."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def embed():
    """Callable fixture: ``embed(smiles)`` -> a clean RDKit conformer."""
    return rdkit_embed


def _xtb_available():
    exe = os.environ.get("XTB_EXE", "xtb")
    return bool(shutil.which(exe) or os.path.exists(exe))


def _openconf_available():
    try:
        import openconf  # noqa: F401
    except ImportError:
        return False
    return True


needs_xtb = pytest.mark.skipif(not _xtb_available(), reason="xtb binary not on PATH / $XTB_EXE")
needs_openconf = pytest.mark.skipif(not _openconf_available(), reason="openconf not installed")
