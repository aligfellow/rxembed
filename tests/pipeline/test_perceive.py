"""Test XYZ graph perception and backend fallback behavior."""

import logging
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
from rdkit import Chem

from rxembed.pipeline import perceive
from rxembed.pipeline.perceive import read_xyz

_BIMP = "examples/structures/bimp.xyz"  # a metal-free TS with a stretched reacting core

_CORPUS = Path(__file__).resolve().parents[2] / "benchmark/corpus"
# `benchmark/` is gitignored and local-only (AGENTS.md), so a clean clone must skip, not error.
needs_corpus = pytest.mark.skipif(not _CORPUS.is_dir(), reason="needs the local-only benchmark/corpus")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz_returns_bonded_molecule():
    mol = read_xyz(_BIMP, 0)
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().GetPositions().shape == (mol.GetNumAtoms(), 3)
    assert any(b.GetBondTypeAsDouble() > 1.0 for b in mol.GetBonds())


def test_missing_xyzgraph_warns_and_names_each_fallback(monkeypatch, caplog, tmp_path):
    monkeypatch.setitem(sys.modules, "xyzgraph", None)
    caplog.set_level(logging.WARNING, logger="rxembed")
    mol = read_xyz("examples/structures/ru-co.xyz", 0)
    assert any(a.GetSymbol() == "Ru" for a in mol.GetAtoms())
    assert "xyzgraph unavailable; using xyz2mol" in caplog.text

    caplog.clear()
    water = tmp_path / "water.xyz"
    water.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    mol = read_xyz(str(water), 0)
    assert mol.GetNumBonds() == 2
    assert "xyzgraph unavailable; using RDKit" in caplog.text

    with pytest.raises(ValueError, match="single-metal"):
        read_xyz("examples/structures/mn-h2.xyz", 0)

    monkeypatch.setattr(perceive, "_rank_orders", lambda *_args: pytest.fail("xyz2mol fallback was ranked twice"))
    read_xyz("examples/structures/ru-co.xyz", 0, bond_orders="xyz2mol")

    for argument in ("connectivity", "bond_orders"):
        with pytest.raises(ValueError, match=argument):
            read_xyz(_BIMP, 0, **{argument: "typo"})


def test_runtime_perceiver_failure_warns_and_uses_the_other(monkeypatch, caplog):
    fallback = Chem.MolFromXYZFile(_BIMP)
    assert fallback is not None

    def fail(*_args):
        raise ValueError("no assignment")

    monkeypatch.setattr(perceive, "_from_xyz2mol", fail)
    monkeypatch.setattr(perceive, "_from_xyzgraph", lambda *_args: fallback)
    caplog.set_level(logging.WARNING, logger="rxembed")
    assert read_xyz(_BIMP, 0, connectivity="xyz2mol", bond_orders="xyz2mol") is fallback
    assert "xyz2mol failed (no assignment); using xyzgraph" in caplog.text

    rw = Chem.RWMol()
    for atomic_number in (44, 7, 7):
        rw.AddAtom(Chem.Atom(atomic_number))
    rw.AddBond(1, 0, Chem.BondType.DATIVE)
    rw.AddBond(2, 0, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    conf = Chem.Conformer(3)
    for i, xyz in enumerate(((0, 0, 0), (2, 0, 0), (-2, 0, 0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)

    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", fail)
    assert perceive._rank_orders(mol, 0) is mol
    assert "keeping input bond orders" in caplog.text

    def drop_a_bond(*_args, **kwargs):
        graph = kwargs["graph"][0]
        changed = Chem.RWMol(graph)
        bond = changed.GetBondWithIdx(0)
        changed.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return (changed.GetMol(),)

    caplog.clear()
    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", drop_a_bond)
    assert perceive._rank_orders(mol, 0) is mol
    assert "changed connectivity" in caplog.text


# --- choosing a perceiver -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("connectivity", "bond_orders"),
    [("xyzgraph", "xyzgraph"), ("xyzgraph", "xyz2mol"), ("xyz2mol", "xyz2mol")],
)
@needs_corpus
def test_perceiver_pairs_read_complex(connectivity, bond_orders):
    path = _CORPUS / "CisPlatin.xyz"
    mol = read_xyz(str(path), 0, connectivity=connectivity, bond_orders=bond_orders)
    assert mol.GetNumAtoms() == 11
    assert mol.GetNumConformers() == 1


@needs_corpus
def test_xyzgraph_bond_orders_need_xyzgraph_connectivity():
    path = _CORPUS / "CisPlatin.xyz"
    with pytest.raises(ValueError, match="connectivity='xyzgraph'"):
        read_xyz(str(path), 0, connectivity="xyz2mol", bond_orders="xyzgraph")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz2mol_bond_orders_preserve_xyzgraph_connectivity():
    path = "examples/structures/ru-co.xyz"
    before = read_xyz(path, 0)
    after = read_xyz(path, 0, bond_orders="xyz2mol")

    def bonds(mol):
        return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}

    assert bonds(after) == bonds(before)
