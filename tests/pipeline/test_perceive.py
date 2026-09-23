"""Test XYZ graph perception and backend fallback behavior."""

import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.metal_core import COORDINATION_METALS
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
    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        read_xyz(_BIMP, 0, bond_orders="xyz2mol", fallback=False)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyzgraph_metal_bonds_round_trip_as_dative(tmp_path):
    source = read_xyz("examples/structures/mnh.xyz", 0)
    metals = {atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    coordination = [
        bond for bond in source.GetBonds() if (bond.GetBeginAtomIdx() in metals) != (bond.GetEndAtomIdx() in metals)
    ]
    assert coordination
    assert all(bond.GetBondType() == Chem.BondType.DATIVE and bond.GetEndAtomIdx() in metals for bond in coordination)
    assert not any(atom.GetNumRadicalElectrons() for atom in source.GetAtoms())

    embedded = rx.embed(rx.cxsmiles(source), n=1, seed=0xF00D)
    output = tmp_path / "mnh.xyz"
    Chem.MolToXYZFile(embedded.mol, str(output), confId=embedded.ids[0])
    recovered = read_xyz(str(output), 0)

    assert rx.dative_smiles(recovered) == rx.dative_smiles(source)
    assert rx.cxsmiles(recovered) == rx.cxsmiles(source)


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

    with pytest.raises(ValueError, match="multiple metals"):
        read_xyz("examples/structures/mn-h2.xyz", 0)

    read_xyz("examples/structures/ru-co.xyz", 0, bond_orders="xyz2mol")

    for argument in ("connectivity", "bond_orders"):
        with pytest.raises(ValueError, match=argument):
            read_xyz(_BIMP, 0, **{argument: "typo"})


def test_runtime_perceiver_failure_warns_and_uses_the_other(monkeypatch, caplog):
    fallback = Chem.MolFromXYZFile(_BIMP)
    assert fallback is not None

    def fail(*_args, **_kwargs):
        raise ValueError("no assignment")

    monkeypatch.setattr(perceive, "_from_xyz2mol", fail)
    monkeypatch.setattr(perceive, "_from_xyzgraph", lambda *_args: fallback)
    caplog.set_level(logging.WARNING, logger="rxembed")
    with pytest.raises(ValueError, match="no assignment"):
        read_xyz(_BIMP, 0, connectivity="xyz2mol", bond_orders="xyz2mol", fallback=False)
    assert read_xyz(_BIMP, 0, connectivity="xyz2mol", bond_orders="xyz2mol") is fallback
    assert "xyz2mol failed (no assignment); using xyzgraph" in caplog.text
    assert fallback.GetProp("_rxembedConnectivity") == "xyzgraph"
    assert fallback.GetBoolProp("_rxembedPerceptionFallback")

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
    with pytest.raises(ValueError, match="could not assign bond orders"):
        perceive._rank_orders(mol, 0)

    def drop_a_bond(*_args, **kwargs):
        graph = kwargs["graph"][0]
        changed = Chem.RWMol(graph)
        bond = changed.GetBondWithIdx(0)
        changed.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return (changed.GetMol(),)

    caplog.clear()
    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", drop_a_bond)
    with pytest.raises(ValueError, match="changed connectivity"):
        perceive._rank_orders(mol, 0)

    mol.GetAtomWithIdx(0).SetFormalCharge(1)
    with pytest.raises(ValueError, match="could not assign bond orders"):
        perceive._rank_orders(mol, 0)


def test_failed_bond_order_and_connectivity_assignment_raise_together(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")

    def fail(*_args):
        raise ValueError("no assignment")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", fail)

    with pytest.raises(ValueError, match=r"xyzgraph connectivity.*xyz2mol connectivity also failed"):
        read_xyz("unused.xyz", bond_orders="xyz2mol")


def test_failed_assignments_do_not_keep_a_selected_graph_with_the_wrong_charge(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+3]<-[Cl-]")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", lambda *_args: selected)

    with pytest.raises(ValueError, match="perceived total charge 1 does not match charge=0"):
        read_xyz("unused.xyz", charge=0, bond_orders="xyz2mol")


def test_boron_cage_is_rejected_before_valence_search(monkeypatch):
    rw = Chem.RWMol()
    for _ in range(5):
        rw.AddAtom(Chem.Atom(5))
    for index in range(4):
        rw.AddBond(index, index + 1, Chem.BondType.SINGLE)
    selected = rw.GetMol()
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="boron cage ligand"):
        read_xyz("unused.xyz", bond_orders="xyz2mol")


def test_unbonded_bridging_hydrogen_is_rejected_before_valence_search(monkeypatch):
    """SIJQIL analogue: xyzgraph leaves a B-H-B bridge hydrogen with zero bonded neighbours."""
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(5))
    rw.AddAtom(Chem.Atom(5))
    rw.AddAtom(Chem.Atom(1))  # bridging H, deliberately left unbonded by the connectivity backend
    selected = rw.GetMol()
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(1.8, 0.0, 0.0))
    conf.SetAtomPosition(2, Point3D(0.9, 1.006, 0.0))  # ~1.35 A from each boron
    selected.AddConformer(conf, assignId=True)
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="3-centre"):
        read_xyz("unused.xyz", bond_orders="xyz2mol")


def test_invalid_selected_connectivity_falls_back_with_provenance(monkeypatch, caplog):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")
    fallback = Chem.MolFromSmiles("[F-]->[Fe+2]<-[F-]")

    def fail(*_args):
        raise ValueError("bad graph")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: fallback)

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = read_xyz("unused.xyz", bond_orders="xyz2mol")

    assert result is fallback
    assert "using xyz2mol connectivity" in caplog.text
    assert result.GetProp("_rxembedConnectivity") == "xyz2mol"
    assert result.GetProp("_rxembedBondOrders") == "xyz2mol"


def test_default_xyzgraph_keeps_a_native_omitted_metal_donor_contact(monkeypatch, caplog):
    """SOSHOY analogue: a real xyzgraph Zn-O 2.43 A contact must survive a native RDKit omission.

    xyzgraph and xyz2mol are the metal-aware readers; native RDKit connectivity never arbitrates a
    metal-donor contact between them (user direction), so the selected reader's donor set is untouched.
    """

    def graph(long_contact):
        rw = Chem.RWMol()
        for atomic_number in (30, 7, 7, 8, 8):  # Zn, 2x NH3 N, 2x carboxylate O (SOSHOY analogue)
            atom = Chem.Atom(atomic_number)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for donor in (1, 2, 3, 4):
            rw.AddBond(donor, 0, Chem.BondType.DATIVE)
        if not long_contact:
            rw.RemoveBond(0, 4)
        mol = rw.GetMol()
        conf = Chem.Conformer(5)
        for index, xyz in enumerate(((0, 0, 0), (2.05, 0, 0), (-2.05, 0, 0), (0, 2.05, 0), (0, -2.43, 0))):
            conf.SetAtomPosition(index, xyz)
        mol.AddConformer(conf)
        mol.UpdatePropertyCache(strict=False)
        return mol

    def donor_edges(mol):
        return {
            frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() == 0 or b.GetEndAtomIdx() == 0
        }

    raw = donor_edges(graph(True))  # r_cov(Zn)+r_cov(O) = 1.88; 2.43 A is 0.10 over the 2.33 outer limit
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (graph(True), "xyzgraph"))
    monkeypatch.setattr(perceive, "_from_rdkit_connectivity", lambda *_args: graph(False))

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = read_xyz("unused.xyz", bond_orders="xyzgraph")

    assert donor_edges(result) == raw
    assert result.GetBondBetweenAtoms(0, 4) is not None
    assert not result.HasProp("_rxembedConnectivityRemoved")
    assert result.GetProp("_rxembedConnectivityAdded") == ""

    strict = read_xyz("unused.xyz", bond_orders="xyzgraph", fallback=False)
    assert donor_edges(strict) == raw


def test_default_xyzgraph_restores_consensus_internal_ligand_bonds_only(monkeypatch, caplog):
    def graph(include_internal, include_extra_metal=False):
        rw = Chem.RWMol()
        for atomic_number in (26, 16, 6, 8):
            atom = Chem.Atom(atomic_number)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        rw.AddBond(1, 0, Chem.BondType.DATIVE)
        if include_internal:
            rw.AddBond(1, 2, Chem.BondType.SINGLE)
        if include_extra_metal:
            rw.AddBond(3, 0, Chem.BondType.DATIVE)
        out = rw.GetMol()
        conf = Chem.Conformer(out.GetNumAtoms())
        for index, point in enumerate(((0, 0, 0), (2, 0, 0), (3, 0, 0), (0, 3, 0))):
            conf.SetAtomPosition(index, point)
        out.AddConformer(conf)
        out.UpdatePropertyCache(strict=False)
        return out

    selected = graph(False)
    native = graph(True)
    joint = graph(True, include_extra_metal=True)
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_from_rdkit_connectivity", lambda *_args: native)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: joint)

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = read_xyz("unused.xyz", bond_orders="xyzgraph")

    assert result.GetBondBetweenAtoms(1, 2) is not None
    assert result.GetBondBetweenAtoms(0, 3) is None
    assert result.GetProp("_rxembedConnectivityAdded") == "1-2"
    assert "internal ligand bonds confirmed" in caplog.text


def test_strict_bond_order_choice_does_not_change_connectivity(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")

    def fail(*_args):
        raise ValueError("bad selected graph")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: pytest.fail("changed connectivity"))

    with pytest.raises(ValueError, match="bad selected graph"):
        read_xyz("unused.xyz", bond_orders="xyz2mol", fallback=False)


def test_connectivity_fallback_preserves_explicit_metal_charge_allocation(monkeypatch):
    selected = Chem.MolFromSmiles("[Fe+2].[Mn+].[Cl-].[Cl-].[Cl-]")
    fallback = Chem.MolFromSmiles("[Fe+].[Mn+2].[Cl-].[Cl-].[Cl-]")

    def fail(*_args):
        raise ValueError("bad graph")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: fallback)

    result = read_xyz("unused.xyz", charge=0, metal_charges={0: 2, 1: 1})

    assert [result.GetAtomWithIdx(index).GetFormalCharge() for index in (0, 1)] == [2, 1]


# --- choosing a perceiver -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("connectivity", "bond_orders"),
    [
        ("xyzgraph", "xyzgraph"),
        ("rdkit", "xyz2mol"),
        ("xyzgraph", "xyz2mol"),
        ("xyz2mol", "xyz2mol"),
    ],
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


def test_bond_order_perception_does_not_accept_auto_mode():
    with pytest.raises(ValueError, match="bond_orders must be 'xyzgraph' or 'xyz2mol'"):
        read_xyz("unused.xyz", bond_orders="auto")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz2mol_bond_orders_preserve_xyzgraph_connectivity():
    path = "examples/structures/ru-co.xyz"
    before = perceive._from_xyzgraph(path, 0)
    after = read_xyz(path, 0, connectivity="xyzgraph", bond_orders="xyz2mol", fallback=False)

    def bonds(mol):
        return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}

    assert bonds(after) == bonds(before)


def test_rdkit_connectivity_uses_native_bond_orders_for_an_organic_graph(caplog):
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol")

    assert any(bond.GetBondTypeAsDouble() > 1 for bond in mol.GetBonds())
    assert "xyz2mol bond-order assignment is metal-only; using RDKit" in caplog.text
    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol", fallback=False)


def test_strict_xyz2mol_orders_do_not_substitute_rdkit_for_an_unsupported_metal(monkeypatch):
    selected = Chem.MolFromSmiles("[Ce]")
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "rdkit"))

    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        read_xyz("unused.xyz", connectivity="rdkit", bond_orders="xyz2mol", fallback=False)


def test_xyz2mol_orders_are_ranked_on_the_selected_connectivity(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")
    ranked = Chem.MolFromSmiles("[F-]->[Fe+2]<-[F-]")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(
        perceive, "_rank_orders", lambda mol, charge: ranked if (mol, charge) == (selected, 0) else None
    )

    assert read_xyz("unused.xyz", connectivity="xyzgraph", bond_orders="xyz2mol") is ranked
    monkeypatch.setattr(perceive, "_rank_orders", lambda *_args: pytest.fail("default changed bond-order backend"))
    assert read_xyz("unused.xyz", connectivity="xyzgraph") is selected


def test_read_xyz_rejects_a_final_charge_different_from_the_request(monkeypatch):
    selected = Chem.MolFromSmiles("C")
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="does not match charge=1; use bond_orders='xyz2mol'"):
        read_xyz("unused.xyz", charge=1, bond_orders="xyzgraph")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_multimetal_xyz_requires_explicit_charge_allocation():
    path = "examples/structures/mn-h2.xyz"
    options = {"bond_orders": "xyz2mol"}

    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals"):
        read_xyz(path, charge=0, **options)
    with pytest.raises(ValueError, match="metal_charges must name every metal atom index"):
        read_xyz(path, charge=0, metal_charges={0: 2}, **options)

    mol = read_xyz(path, charge=0, metal_charges={0: 2, 1: 1}, **options)
    assert Chem.GetFormalCharge(mol) == 0
    assert [(mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(i).GetFormalCharge()) for i in (0, 1)] == [
        ("Fe", 2),
        ("Mn", 1),
    ]


def test_bond_order_ranking_keeps_metal_bonds_out_of_ligand_valence(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 1, 1, 1, 1):
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    for hydrogen in range(2, 6):
        rw.AddBond(1, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    def check_private_graph(*_args, **kwargs):
        graph = kwargs["graph"][0]
        bond = graph.GetBondBetweenAtoms(0, 1)
        assert bond.GetBondType() == Chem.BondType.DATIVE
        assert bond.GetBeginAtomIdx() == 1
        assert graph.GetAtomWithIdx(1).GetValence(Chem.ValenceType.EXPLICIT) == 4
        return (graph,)

    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", check_private_graph)
    ranked = perceive._rank_orders(mol, 0)

    assert ranked.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DATIVE


def test_bond_order_ranking_does_not_promote_a_nonmetal_hydrogen_contact(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number in (26, 8, 1, 8):
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for pair in ((0, 1), (1, 2), (2, 3)):
        rw.AddBond(*pair, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1.4, 0, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    def check_private_graph(*_args, **kwargs):
        graph = kwargs["graph"][0]
        assert graph.GetAtomWithIdx(2).GetDegree() == 2
        return (graph,)

    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", check_private_graph)

    perceive._rank_orders(mol, 0)


def test_bond_order_ranking_keeps_the_ligand_leg_of_a_hydride_bridge(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 1):
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    rw.AddBond(0, 2, Chem.BondType.SINGLE)
    rw.AddBond(1, 2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(3)
    for atom, point in enumerate(((0, 0, 0), (2.0, 0, 0), (0.8, 0, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    def check_private_graph(*_args, **kwargs):
        graph = kwargs["graph"][0]
        assert graph.GetBondBetweenAtoms(1, 2) is not None
        assert graph.GetBondBetweenAtoms(0, 2) is None
        return (graph,)

    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", check_private_graph)

    ranked = perceive._rank_orders(mol, 0)

    contact = ranked.GetBondBetweenAtoms(0, 2)
    assert contact.GetBondType() == Chem.BondType.DATIVE
    assert contact.GetBeginAtomIdx() == 2


def test_symmetric_nonmetal_shared_hydrogen_uses_a_zero_order_contact(monkeypatch):
    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", lambda *_args, **kwargs: (kwargs["graph"][0],))
    outcomes = []
    for edges in (((1, 2), (3, 2)), ((3, 2), (1, 2))):
        rw = Chem.RWMol()
        for atomic_number in (26, 7, 1, 8):
            atom = Chem.Atom(atomic_number)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for edge in edges:
            rw.AddBond(*edge, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        conf = Chem.Conformer(4)
        for atom, point in enumerate(((5, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0))):
            conf.SetAtomPosition(atom, Point3D(*point))
        mol.AddConformer(conf)

        ranked = perceive._rank_orders(mol, 0)
        contacts = [bond for bond in ranked.GetAtomWithIdx(2).GetBonds() if bond.GetBondType() == Chem.BondType.ZERO]
        ordinary = [bond for bond in ranked.GetAtomWithIdx(2).GetBonds() if bond.GetBondType() != Chem.BondType.ZERO]
        outcomes.append((contacts[0].GetOtherAtomIdx(2), ordinary[0].GetOtherAtomIdx(2)))

    assert outcomes[0] == outcomes[1]


def test_connectivity_sources_preserve_an_open_eta3_face(monkeypatch):
    rw = Chem.RWMol()
    for symbol in ("Pd", "C", "C", "C"):
        rw.AddAtom(Chem.Atom(symbol))
    for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3)):
        rw.AddBond(*pair, Chem.BondType.SINGLE)
    source = rw.GetMol()
    conf = Chem.Conformer(4)
    for atom, point in enumerate(((0, 0, 0), (1.8, 0.7, 0), (2.5, 0, 0), (1.8, -0.7, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    source.AddConformer(conf)
    source.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(source)

    def edges(mol):
        return {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in mol.GetBonds()}

    expected = edges(source)

    graph = nx.Graph()
    positions = source.GetConformer().GetPositions()
    for atom in source.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            formal_charge=0,
            position=positions[atom.GetIdx()],
        )
    for left, right in expected:
        graph.add_edge(left, right, bond_order=1)
    monkeypatch.setitem(sys.modules, "xyzgraph", SimpleNamespace(build_graph=lambda *_args, **_kwargs: graph))
    monkeypatch.setattr(perceive, "_coordinates", lambda _path: Chem.Mol(source))
    monkeypatch.setattr("rxembed.pipeline.xyz2mol_tmc.get_tmc_mol", lambda *_args, **_kwargs: (Chem.Mol(source),))
    assert edges(perceive._from_xyzgraph("unused.xyz", 0)) == expected
    assert edges(perceive._from_xyz2mol("unused.xyz", 0)) == expected
