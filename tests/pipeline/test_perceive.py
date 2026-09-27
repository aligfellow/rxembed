"""Test XYZ graph perception and backend fallback behavior."""

import itertools
import logging
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.metal_core import COORDINATION_METALS, canonical_metal_graph, reject_boron_cages
from rxembed.pipeline import perceive
from tests.conftest import EXAMPLES_DIR

_BIMP = str(EXAMPLES_DIR / "bimp.xyz")  # a metal-free TS with a stretched reacting core


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz_returns_bonded_molecule():
    mol = perceive.read_xyz(_BIMP, 0)
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().GetPositions().shape == (mol.GetNumAtoms(), 3)
    assert any(b.GetBondTypeAsDouble() > 1.0 for b in mol.GetBonds())
    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        perceive.read_xyz(_BIMP, 0, bond_orders="xyz2mol", fallback=False)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyzgraph_metal_bonds_round_trip_as_dative(tmp_path):
    source = perceive.read_xyz(str(EXAMPLES_DIR / "mnh.xyz"), 0)
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
    recovered = perceive.read_xyz(str(output), 0)

    assert rx.dative_smiles(recovered) == rx.dative_smiles(source)
    assert rx.cxsmiles(recovered) == rx.cxsmiles(source)


def test_missing_xyzgraph_warns_and_names_each_fallback(monkeypatch, caplog, tmp_path):
    monkeypatch.setitem(sys.modules, "xyzgraph", None)
    caplog.set_level(logging.WARNING, logger="rxembed")
    mol = perceive.read_xyz(str(EXAMPLES_DIR / "ru-co.xyz"), 0)
    assert any(a.GetSymbol() == "Ru" for a in mol.GetAtoms())
    assert "xyzgraph unavailable; using xyz2mol" in caplog.text

    caplog.clear()
    water = tmp_path / "water.xyz"
    water.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    mol = perceive.read_xyz(str(water), 0)
    assert mol.GetNumBonds() == 2
    assert "xyzgraph unavailable; using RDKit" in caplog.text

    with pytest.raises(ValueError, match="multiple metals"):
        perceive.read_xyz(str(EXAMPLES_DIR / "mn-h2.xyz"), 0)

    perceive.read_xyz(str(EXAMPLES_DIR / "ru-co.xyz"), 0, bond_orders="xyz2mol")

    for argument in ("connectivity", "bond_orders"):
        with pytest.raises(ValueError, match=argument):
            perceive.read_xyz(_BIMP, 0, **{argument: "typo"})


def test_runtime_perceiver_failure_warns_and_uses_the_other(monkeypatch, caplog):
    fallback = Chem.MolFromXYZFile(_BIMP)
    assert fallback is not None

    def fail(*_args, **_kwargs):
        raise ValueError("no assignment")

    monkeypatch.setattr(perceive, "_from_xyz2mol", fail)
    monkeypatch.setattr(perceive, "_from_xyzgraph", lambda *_args: fallback)
    caplog.set_level(logging.WARNING, logger="rxembed")
    with pytest.raises(ValueError, match="no assignment"):
        perceive.read_xyz(_BIMP, 0, connectivity="xyz2mol", bond_orders="xyz2mol", fallback=False)
    assert perceive.read_xyz(_BIMP, 0, connectivity="xyz2mol", bond_orders="xyz2mol") is fallback
    assert "xyz2mol failed (no assignment); using xyzgraph" in caplog.text
    assert fallback.GetProp("_rxembedConnectivity") == "xyzgraph"
    assert fallback.GetBoolProp("_rxembedPerceptionFallback")


def test_rank_orders_refuses_a_changed_or_broken_graph(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("no assignment")

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

    monkeypatch.setattr(perceive, "get_tmc_mol", fail)
    with pytest.raises(ValueError, match="could not assign bond orders"):
        perceive._rank_orders(mol, 0)

    def drop_a_bond(*_args, **kwargs):
        graph = kwargs["graph"][0]
        changed = Chem.RWMol(graph)
        bond = changed.GetBondWithIdx(0)
        changed.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return (changed.GetMol(),)

    monkeypatch.setattr(perceive, "get_tmc_mol", drop_a_bond)
    with pytest.raises(ValueError, match="changed connectivity"):
        perceive._rank_orders(mol, 0)

    mol.GetAtomWithIdx(0).SetFormalCharge(1)
    with pytest.raises(ValueError, match="could not assign bond orders"):
        perceive._rank_orders(mol, 0)


@pytest.mark.parametrize(("smiles", "kept"), [("[Cl-]->[Fe+2]<-[Cl-]", True), ("[Cl-]->[Fe+3]<-[Cl-]", False)])
def test_iron_dichloride_keeps_xyzgraph_charges_only_when_both_xyz2mol_readings_fail_and_they_fit(
    monkeypatch, caplog, smiles, kept
):
    selected = Chem.MolFromSmiles(smiles)

    def fail(*_args, **_kwargs):
        raise ValueError("no assignment")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", fail)

    if not kept:  # Fe+3 and two Cl- total +1, not the requested 0
        with pytest.raises(ValueError, match=r"xyz2mol found no bond orders on the xyzgraph graph.*also failed"):
            perceive.read_xyz("unused.xyz", bond_orders="xyz2mol")
        return
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz("unused.xyz", bond_orders="xyz2mol")
    assert [a.GetFormalCharge() for a in mol.GetAtoms()] == [-1, 2, -1]
    assert mol.GetProp("_rxembedBondOrders") == "xyzgraph"
    assert mol.GetBoolProp("_rxembedPerceptionFallback")
    assert mol.GetProp("_rxembedChargeRescue").endswith("kept xyzgraph's Fe+2")
    assert "kept xyzgraph's Fe+2" in caplog.text


def _write_complex(tmp_path, name, atoms, order):
    """Write `atoms` as an .xyz with its lines in `order`."""
    rows = [atoms[i] for i in order]
    path = tmp_path / f"{name}-{''.join(map(str, order))}.xyz"
    path.write_text(f"{len(rows)}\n{name}\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, x, y, z in rows))
    return str(path)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("atoms", "metal", "radical"),
    [
        # HgI3: a neutral doublet, so the unpaired electron sits on an iodine next to d10 Hg(II).
        ([("Hg", 0, 0, 0), ("I", 2.7, 0, 0), ("I", -1.35, 2.338, 0), ("I", -1.35, -2.338, 0)], "Hg+2", "I"),
        # [VO2Cl2]- read without its charge, as PIVPOB's dioxo V(V) was: V+6 at charge=0 gives an oxo O radical.
        (
            [
                ("V", 0, 0, 0),
                ("O", 0.924, 0.924, 0.924),
                ("O", 0.924, -0.924, -0.924),
                ("Cl", -1.270, 1.270, -1.270),
                ("Cl", -1.270, -1.270, 1.270),
            ],
            "V+5",
            "O",
        ),
    ],
    ids=["mercury_triiodide", "dioxovanadium_dichloride"],
)
def test_metal_past_its_valence_at_charge_zero_is_read_at_its_cap_with_a_donor_radical(
    tmp_path, caplog, atoms, metal, radical
):
    smiles = set()
    for order in (list(range(len(atoms))), list(reversed(range(len(atoms))))):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rxembed"):
            mol = perceive.read_xyz(_write_complex(tmp_path, radical, atoms, order), 0, bond_orders="xyz2mol")

        centre = next(a for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
        assert f"{centre.GetSymbol()}{centre.GetFormalCharge():+d}" == metal
        assert [(a.GetSymbol(), a.GetNumRadicalElectrons()) for a in mol.GetAtoms() if a.GetNumRadicalElectrons()] == [
            (radical, 1)
        ]
        assert Chem.GetFormalCharge(mol) == 0
        assert mol.GetProp("_rxembedBondOrders") == "xyz2mol"
        assert not mol.GetBoolProp("_rxembedPerceptionFallback")
        note = mol.GetProp("_rxembedChargeRescue")
        assert f"read {metal} with a radical on {radical}" in note
        assert note.endswith("check charge=")
        assert f"read_xyz: {note}" in caplog.text
        smiles.add(Chem.MolToSmiles(mol))
    assert len(smiles) == 1


def test_failed_assignments_do_not_keep_a_selected_graph_with_the_wrong_charge(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+3]<-[Cl-]")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", lambda *_args: selected)

    with pytest.raises(ValueError, match="perceived total charge 1 does not match charge=0"):
        perceive.read_xyz("unused.xyz", charge=0, bond_orders="xyz2mol")


def _closo_b6h6():
    """Build a closo-B6H6 octahedron: each boron bonds four borons and one hydrogen."""
    rw = Chem.RWMol()
    borons = [rw.AddAtom(Chem.Atom(5)) for _ in range(6)]
    hydrogens = [rw.AddAtom(Chem.Atom(1)) for _ in range(6)]
    antipode = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    for i, j in itertools.combinations(range(6), 2):
        if antipode[i] != j:
            rw.AddBond(borons[i], borons[j], Chem.BondType.SINGLE)
    for boron, hydrogen in zip(borons, hydrogens, strict=True):
        rw.AddBond(boron, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def test_boron_cage_is_rejected_before_valence_search(monkeypatch):
    """RDKit will not sanitise a five-bonded boron from SMILES, so this is built as a raw graph."""
    selected = _closo_b6h6()
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="two-centre donor model"):
        perceive.read_xyz("unused.xyz", bond_orders="xyz2mol")


def test_five_boryl_groups_in_one_ligand_are_not_a_cage():
    """Five three-coordinate borons on one chain are not a cage: no vertex exceeds its octet."""
    rw = Chem.RWMol()
    chain = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    for left, right in itertools.pairwise(chain):
        rw.AddBond(left, right, Chem.BondType.SINGLE)
    for carbon in chain:
        boron = rw.AddAtom(Chem.Atom(5))
        rw.AddBond(carbon, boron, Chem.BondType.SINGLE)
        for _ in range(2):
            chlorine = rw.AddAtom(Chem.Atom(17))
            rw.AddBond(boron, chlorine, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    reject_boron_cages(mol)  # must not raise: every boron is three-coordinate


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
        perceive.read_xyz("unused.xyz", bond_orders="xyz2mol")


def test_invalid_selected_connectivity_falls_back_with_provenance(monkeypatch, caplog):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")
    fallback = Chem.MolFromSmiles("[F-]->[Fe+2]<-[F-]")

    def fail(*_args):
        raise ValueError("bad graph")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: fallback)

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = perceive.read_xyz("unused.xyz", bond_orders="xyz2mol")

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
        result = perceive.read_xyz("unused.xyz", bond_orders="xyzgraph")

    assert donor_edges(result) == raw
    assert result.GetBondBetweenAtoms(0, 4) is not None
    assert not result.HasProp("_rxembedConnectivityRemoved")
    assert result.GetProp("_rxembedConnectivityAdded") == ""

    strict = perceive.read_xyz("unused.xyz", bond_orders="xyzgraph", fallback=False)
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
        result = perceive.read_xyz("unused.xyz", bond_orders="xyzgraph")

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
        perceive.read_xyz("unused.xyz", bond_orders="xyz2mol", fallback=False)


def test_connectivity_fallback_preserves_explicit_metal_charge_allocation(monkeypatch):
    selected = Chem.MolFromSmiles("[Fe+2].[Mn+].[Cl-].[Cl-].[Cl-]")
    fallback = Chem.MolFromSmiles("[Fe+].[Mn+2].[Cl-].[Cl-].[Cl-]")

    def fail(*_args):
        raise ValueError("bad graph")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(perceive, "_rank_orders", fail)
    monkeypatch.setattr(perceive, "_from_xyz2mol", lambda *_args: fallback)

    result = perceive.read_xyz("unused.xyz", charge=0, metal_charges={0: 2, 1: 1})

    assert [result.GetAtomWithIdx(index).GetFormalCharge() for index in (0, 1)] == [2, 1]


@pytest.mark.parametrize(
    ("smiles", "charge", "warning"),
    [
        ("N->[Cu](<-N)(<-N)<-N", 0, "Cu+0 has an odd electron count"),  # [Cu(NH3)4]2+ without its charge
        ("[I-]->[Hg+3](<-[I-])<-[I-]", 0, "Hg+3 is over its 2 valence electrons"),  # xyzgraph's HgI3
        ("C[N+](C)(C)C.[O+]#[C-]->[Co-](<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", 0, "Co-1 is negative"),
        ("[Cl-]->[Zn+2]<-[Cl-]", 0, None),  # d10 Zn(II): nothing suggests a missing charge
        ("N->[Cu+2](<-N)(<-N)<-N", 2, None),  # the charge was given
    ],
    ids=["tetraammine_copper", "mercury_triiodide", "tetracarbonylcobaltate", "zinc_dichloride", "copper_two_plus"],
)
def test_metal_read_at_charge_zero_warns_when_its_reading_suggests_a_missing_charge(
    monkeypatch, caplog, smiles, charge, warning
):
    selected = Chem.MolFromSmiles(smiles)
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz("unused.xyz", charge=charge)

    assert Chem.GetFormalCharge(mol) == charge
    hints = [record.getMessage() for record in caplog.records if "pass charge=" in record.getMessage()]
    assert hints == (
        [] if warning is None else [f"read_xyz: at charge=0, {warning}; pass charge= for a charged complex"]
    )


# --- choosing a perceiver -------------------------------------------------------------------------


def _cisplatin_xyz(tmp_path):
    """Write a cisplatin (Pt(NH3)2Cl2) geometry to `tmp_path`, exercising every metal-aware perceiver pair."""
    iso = rx.metal("[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]", "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1)
    path = tmp_path / "cisplatin.xyz"
    Chem.MolToXYZFile(ens.mol, str(path), confId=ens.ids[0])
    return str(path)


@pytest.mark.parametrize(
    ("connectivity", "bond_orders"),
    [
        ("xyzgraph", "xyzgraph"),
        ("rdkit", "xyz2mol"),
        ("xyzgraph", "xyz2mol"),
        ("xyz2mol", "xyz2mol"),
    ],
)
def test_perceiver_pairs_read_complex(connectivity, bond_orders, tmp_path):
    mol = perceive.read_xyz(_cisplatin_xyz(tmp_path), 0, connectivity=connectivity, bond_orders=bond_orders)
    assert mol.GetNumAtoms() == 11
    assert mol.GetNumConformers() == 1


def test_xyzgraph_bond_orders_need_xyzgraph_connectivity(tmp_path):
    with pytest.raises(ValueError, match="connectivity='xyzgraph'"):
        perceive.read_xyz(_cisplatin_xyz(tmp_path), 0, connectivity="xyz2mol", bond_orders="xyzgraph")


def test_bond_order_perception_does_not_accept_auto_mode():
    with pytest.raises(ValueError, match="bond_orders must be 'xyzgraph' or 'xyz2mol'"):
        perceive.read_xyz("unused.xyz", bond_orders="auto")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz2mol_bond_orders_preserve_xyzgraph_connectivity():
    path = str(EXAMPLES_DIR / "ru-co.xyz")
    before = perceive._from_xyzgraph(path, 0)
    after = perceive.read_xyz(path, 0, connectivity="xyzgraph", bond_orders="xyz2mol", fallback=False)

    def bonds(mol):
        return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}

    assert bonds(after) == bonds(before)


def test_rdkit_connectivity_uses_native_bond_orders_for_an_organic_graph(caplog):
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol")

    assert any(bond.GetBondTypeAsDouble() > 1 for bond in mol.GetBonds())
    assert "xyz2mol bond-order assignment is metal-only; using RDKit" in caplog.text
    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        perceive.read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol", fallback=False)


def test_strict_xyz2mol_orders_do_not_substitute_rdkit_for_an_unsupported_metal(monkeypatch):
    selected = Chem.MolFromSmiles("[Ce]")
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "rdkit"))

    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        perceive.read_xyz("unused.xyz", connectivity="rdkit", bond_orders="xyz2mol", fallback=False)


def test_xyz2mol_orders_are_ranked_on_the_selected_connectivity(monkeypatch):
    selected = Chem.MolFromSmiles("[Cl-]->[Fe+2]<-[Cl-]")
    ranked = Chem.MolFromSmiles("[F-]->[Fe+2]<-[F-]")

    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    monkeypatch.setattr(
        perceive, "_rank_orders", lambda mol, charge: ranked if (mol, charge) == (selected, 0) else None
    )

    assert perceive.read_xyz("unused.xyz", connectivity="xyzgraph", bond_orders="xyz2mol") is ranked
    monkeypatch.setattr(perceive, "_rank_orders", lambda *_args: pytest.fail("default changed bond-order backend"))
    assert perceive.read_xyz("unused.xyz", connectivity="xyzgraph") is selected


def test_read_xyz_rejects_a_final_charge_different_from_the_request(monkeypatch):
    selected = Chem.MolFromSmiles("C")
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="does not match charge=1; use bond_orders='xyz2mol'"):
        perceive.read_xyz("unused.xyz", charge=1, bond_orders="xyzgraph")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_multimetal_xyz_requires_explicit_charge_allocation():
    path = str(EXAMPLES_DIR / "mn-h2.xyz")
    options = {"bond_orders": "xyz2mol"}

    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals"):
        perceive.read_xyz(path, charge=0, **options)
    with pytest.raises(ValueError, match="metal_charges must name every metal atom index"):
        perceive.read_xyz(path, charge=0, metal_charges={0: 2}, **options)

    mol = perceive.read_xyz(path, charge=0, metal_charges={0: 2, 1: 1}, **options)
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

    monkeypatch.setattr(perceive, "get_tmc_mol", check_private_graph)
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

    monkeypatch.setattr(perceive, "get_tmc_mol", check_private_graph)

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

    monkeypatch.setattr(perceive, "get_tmc_mol", check_private_graph)

    ranked = perceive._rank_orders(mol, 0)

    contact = ranked.GetBondBetweenAtoms(0, 2)
    assert contact.GetBondType() == Chem.BondType.DATIVE
    assert contact.GetBeginAtomIdx() == 2


@pytest.mark.parametrize(("left", "right", "bridge"), [(7, 8, False), (5, 5, True)])
def test_shared_hydrogen_keeps_a_zero_order_leg_only_to_an_atom_without_a_lone_pair(monkeypatch, left, right, bridge):
    """N-H...O is a hydrogen bond and leaves the graph; B-H-B is a three-centre bond and keeps its second leg."""
    monkeypatch.setattr(perceive, "get_tmc_mol", lambda *_args, **kwargs: (kwargs["graph"][0],))
    outcomes = []
    for edges in (((1, 2), (3, 2)), ((3, 2), (1, 2))):
        rw = Chem.RWMol()
        for atomic_number in (26, left, 1, right):
            atom = Chem.Atom(atomic_number)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(2 if atomic_number == 5 else 0)  # terminal B-H of a borane
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
        legs = {bond.GetOtherAtomIdx(2): bond.GetBondType() for bond in ranked.GetAtomWithIdx(2).GetBonds()}
        outcomes.append(legs)

    assert outcomes[0] == outcomes[1]
    assert sorted(outcomes[0].values()) == (
        [Chem.BondType.SINGLE, Chem.BondType.ZERO] if bridge else [Chem.BondType.SINGLE]
    )


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
    monkeypatch.setattr(perceive, "get_tmc_mol", lambda *_args, **_kwargs: (Chem.Mol(source),))
    assert edges(perceive._from_xyzgraph("unused.xyz", 0)) == expected
    assert edges(perceive._from_xyz2mol("unused.xyz", 0)) == expected


# --- _drop_bridgehead_bonds: a reader-only fix for a chelate bridgehead with no donor orbital ----------
#
# `rx.parse_smiles` and hand-built RWMols are both used only as convenient ways to construct a graph
# shaped like what a coordinate-derived reader can mis-bond; `_drop_bridgehead_bonds` is graph-only and
# does not care how its input Mol was built. A real user SMILES is never routed through this guard: only
# `read_xyz` calls it.

# Class A: a bridgehead with 4+ non-metal sigma bonds and no lone pair of its own.
_DTP_NI = "C[P]12(C)=[S]->[Ni+2]<-1<-[S-]2"  # dimethyldithiophosphinate kappa2, plus a wrong explicit Ni-P bond
_BH4_NI = "[H]1[BH2-]2[H]->[Ni+2]<-1<-2"  # kappa2-BH4 bridging two H, plus a wrong explicit Ni-B bond
_SIH_NI = "C[Si]1(C)(C)[H]->[Ni+2]<-1"  # sigma-silane eta2-Si-H: only the H neighbour of Si is metal-bound
_PHOSPHINE_NI = "C[PH](C)->[Ni+2]"  # an ordinary phosphine: the lone pair donates straight to the metal
_CARBOXYLATE_NI = "C[C]1(=O)[O-]->[Ni+2]<-1"  # kappa1 carboxylate with an (uncorrected) M-C contact


def _metal_neighbours(mol):
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
    return metal, sorted(n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors())


def test_read_xyz_drops_a_bridgehead_bond_the_connectivity_backend_mis_bonded(monkeypatch):
    """`read_xyz` applies the bridgehead guard itself, whichever backend perceived the graph."""
    selected = rx.parse_smiles(_DTP_NI, remove_hs=False)
    metal, before = _metal_neighbours(selected)
    assert len(before) == 3  # the two real S donors plus the wrong P bond
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    result = perceive.read_xyz("unused.xyz", charge=Chem.GetFormalCharge(selected))

    after = sorted(n.GetSymbol() for n in result.GetAtomWithIdx(metal).GetNeighbors())
    assert after == ["S", "S"]


# Class B: a TRIGONAL bridgehead (exactly 3 non-metal sigma bonds) with no lone pair of its own.

# A kappa2 carboxylate/dithiocarbamate written with the wrong M-C bond still present, using ring-closure
# digits the way `_DTP_NI`/`_BH4_NI` do above: one real donor bonds Ni inline, the other two (one real,
# one the erroneous C) close back to that same Ni.
_CARBOXYLATE_KAPPA2_NI = "C[C]1(=[O]2)[O-]->[Ni+2]<-1<-2"
_DITHIOCARBAMATE_KAPPA2_NI = "CN(C)[C]1(=[S]2)[S-]->[Ni+2]<-1<-2"


def _chain_bound_to_metal(symbols, bond_orders, charges, metal="Ni", metal_charge=2):
    """Build a chain of `symbols`, each atom also sigma-bonded to one metal (a misperceived hapticity)."""
    rw = Chem.RWMol()
    idx = [rw.AddAtom(Chem.Atom(s)) for s in symbols]
    for i, order in enumerate(bond_orders):
        rw.AddBond(idx[i], idx[i + 1], Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    for i, charge in zip(idx, charges, strict=True):
        rw.GetAtomWithIdx(i).SetFormalCharge(charge)
    m = rw.AddAtom(Chem.Atom(metal))
    rw.GetAtomWithIdx(m).SetFormalCharge(metal_charge)
    for i in idx:
        rw.AddBond(i, m, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol, idx, m


def _ring_bound_to_metal(symbols, bond_orders, charges, metal="Ni", metal_charge=2):
    """As `_chain_bound_to_metal`, but closed into a ring (a misperceived haptic face)."""
    rw = Chem.RWMol()
    idx = [rw.AddAtom(Chem.Atom(s)) for s in symbols]
    n = len(symbols)
    for i, order in enumerate(bond_orders):
        rw.AddBond(idx[i], idx[(i + 1) % n], Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    for i, charge in zip(idx, charges, strict=True):
        rw.GetAtomWithIdx(i).SetFormalCharge(charge)
    m = rw.AddAtom(Chem.Atom(metal))
    rw.GetAtomWithIdx(m).SetFormalCharge(metal_charge)
    for i in idx:
        rw.AddBond(i, m, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol, idx, m


def _eta2_formaldehyde():
    rw = Chem.RWMol()
    c, o = rw.AddAtom(Chem.Atom("C")), rw.AddAtom(Chem.Atom("O"))
    rw.AddBond(c, o, Chem.BondType.DOUBLE)
    metal = rw.AddAtom(Chem.Atom("Ni"))
    rw.GetAtomWithIdx(metal).SetFormalCharge(2)
    rw.AddBond(c, metal, Chem.BondType.DATIVE)
    rw.AddBond(o, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _hydride_transfer_ru_h_carbon():
    # A carbon bonded to a bridging H (the hydride-transfer contact) and a real O donor, plus the
    # erroneous Ru-C bond this rule could otherwise prune; the H flanking donor keeps it.
    rw = Chem.RWMol()
    c, h, o, me = (rw.AddAtom(Chem.Atom(sym)) for sym in ("C", "H", "O", "C"))
    rw.AddBond(c, h, Chem.BondType.SINGLE)
    rw.AddBond(c, o, Chem.BondType.DOUBLE)
    rw.AddBond(c, me, Chem.BondType.SINGLE)
    metal = rw.AddAtom(Chem.Atom("Ru"))
    rw.GetAtomWithIdx(metal).SetFormalCharge(2)
    rw.AddBond(c, metal, Chem.BondType.DATIVE)
    rw.AddBond(h, metal, Chem.BondType.DATIVE)
    rw.AddBond(o, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol


@pytest.mark.parametrize(
    ("smiles", "kept_symbols", "detail"),
    [
        pytest.param(_DTP_NI, ["S", "S"], "P1-", id="dithiophosphinate-P"),
        pytest.param(_BH4_NI, ["H", "H"], None, id="kappa2-BH4-B"),
        pytest.param(_CARBOXYLATE_KAPPA2_NI, ["O", "O"], None, id="kappa2-carboxylate-C"),
        pytest.param(_DITHIOCARBAMATE_KAPPA2_NI, ["S", "S"], None, id="dithiocarbamate-C"),
    ],
)
def test_bridgehead_donor_loses_its_wrong_metal_bond(caplog, smiles, kept_symbols, detail):
    """A bridgehead donor (4+ sigma bonds, or 3 with no ring of its own) loses its spurious metal bond."""
    mol = rx.parse_smiles(smiles, remove_hs=False)
    metal, _before = _metal_neighbours(mol)

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        out = perceive._drop_bridgehead_bonds(mol)

    after = [out.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in out.GetAtomWithIdx(metal).GetNeighbors()]
    assert sorted(after) == kept_symbols
    assert "bridgehead" in caplog.text
    if detail:
        assert detail in caplog.text


@pytest.mark.parametrize(
    "build",
    [
        lambda: rx.parse_smiles(_SIH_NI, remove_hs=False),
        lambda: rx.parse_smiles(_PHOSPHINE_NI, remove_hs=False),
        lambda: rx.parse_smiles(_CARBOXYLATE_NI, remove_hs=False),
        lambda: _chain_bound_to_metal(["C", "C", "C"], [2, 1], [0, 0, -1])[0],
        lambda: _ring_bound_to_metal(["C"] * 5, [2, 1, 2, 1, 1], [0, 0, 0, 0, -1])[0],
        # N1, C2, N3, C4, C5: C2 sits between the two ring nitrogens, exactly the bridgehead's flanking-donor
        # pattern, but N1/C2/N3 share a real (metal-free) ring, so condition 3 keeps the Ni-C2 bond.
        lambda: _ring_bound_to_metal(["N", "C", "N", "C", "C"], [1, 2, 1, 2, 1], [0, 0, 0, 0, 0])[0],
        _eta2_formaldehyde,
        _hydride_transfer_ru_h_carbon,
    ],
    ids=[
        "sigma-silane",
        "phosphine",
        "kappa1-carboxylate-MC",
        "allyl",
        "cyclopentadienide",
        "imidazolyl",
        "eta2-formaldehyde",
        "hydride-transfer-C",
    ],
)
def test_non_bridgehead_donor_keeps_its_metal_bond(build):
    """A donor with its own lone pair, or one member of a rigid haptic face, keeps its metal bond."""
    mol = build()
    metal, before = _metal_neighbours(mol)

    out = perceive._drop_bridgehead_bonds(mol)

    assert sorted(n.GetIdx() for n in out.GetAtomWithIdx(metal).GetNeighbors()) == before


def test_smiles_input_keeps_a_bridgehead_bond_canonical_metal_graph_no_longer_drops():
    """A user SMILES is not routed through `_drop_bridgehead_bonds`; `canonical_metal_graph` keeps it."""
    mol = rx.parse_smiles(_DTP_NI, remove_hs=False)
    metal, before = _metal_neighbours(mol)

    out = canonical_metal_graph(mol)

    assert sorted(n.GetIdx() for n in out.GetAtomWithIdx(metal).GetNeighbors()) == before


# AREPUK's failure mode in miniature: a quinolin-8-yl kappa2-N,C chelate, its ring-junction C8a bonded
# to both real donors (N1, the anionic ipso C8) directly, plus the wrong Pd-C8a contact a coordinate
# reader can add. N1 shares the pyridine ring with C8a, C8 shares the benzo ring with C8a: two different
# rings fused at C8a, not one ring holding every donor.
_QUINOLINYL_PD = "c1c[c-]3c24n(->[Pd+2]<-3<-4)cccc2c1"


def test_read_xyz_drops_the_ring_junction_of_a_four_membered_quinolinyl_chelate(monkeypatch):
    """A ring-junction bridgehead prunes even through a carbon donor (regression for the AREPUK read)."""
    selected = rx.parse_smiles(_QUINOLINYL_PD, remove_hs=False)
    metal, before = _metal_neighbours(selected)
    assert len(before) == 3  # N1, the real ipso C8, and the wrong C8a contact
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    result = perceive.read_xyz("unused.xyz", charge=Chem.GetFormalCharge(selected))

    after = sorted(n.GetSymbol() for n in result.GetAtomWithIdx(metal).GetNeighbors())
    assert after == ["C", "N"]


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_read_drops_a_triazolate_cross_ring_contact(tmp_path):
    """A 1,2,4-triazole's two ring carbons sit 2.095 A apart, just inside xyzgraph's C-C cutoff.

    xyzgraph 1.6.14 bonds that contact anyway, since its strict three-ring check only rejects an obtuse
    apex above 110 deg (mean-Z adjusted) and this one is 101.8 deg; a real three-ring cannot have an obtuse
    apex at all, so `_right_angle_ring_chords` drops the bond and xyzgraph rebuilds without it.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1ncnc1"))  # ring order N1-N2-C3-N4-C5, N4 flanked by both carbons
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    c3, c5 = 2, 4
    path = tmp_path / "triazole.xyz"
    Chem.MolToXYZFile(mol, str(path))

    out = rx.read_xyz(str(path), charge=0, bond_orders="xyz2mol")

    assert out.GetBondBetweenAtoms(c3, c5) is None
    assert Chem.MolToSmiles(out) == "[H]c1nc([H])n([H])n1"
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)  # must not raise or invent a chiral ring carbon
    assert out.GetAtomWithIdx(c3).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    assert out.GetAtomWithIdx(c5).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


# Coordinates excerpted verbatim from tmQMg's SORGAK.xyz: La, its thiolate S2, and the dioxazole ring
# (O9, O10, N12, C46, its own H47) that folds C50 close enough for xyzgraph to bond La-C50 (3.02 A).
# C46's second ring substituent (C43) is capped with H in place of extending the ligand further; this
# is the smallest slice that keeps the fold and the ring intact.
_SORGAK_BRIDGEHEAD_FRAGMENT = """9

La    -0.1979   0.1255  -0.1040
S      0.9839   1.6458  -2.2685
O      1.8636  -0.8016  -1.5928
O      4.0509  -0.4435  -1.0874
N      3.4846   0.7917  -1.5000
C      3.0189  -1.4340  -1.0183
C      2.2489   0.5720  -1.8130
H      3.3107  -2.3162  -1.6344
H      2.7946  -1.7327   0.0057
"""


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_read_drops_a_bridgehead_ring_carbon_flanked_by_only_one_ring_donor(caplog, tmp_path):
    """A folded kappa2-S,O chelate: the bridgehead's ring donor (O) keeps a real ring face, but its
    exocyclic thiolate (S) does not, so one shared ring is not a face and the La-C bond still drops.
    """
    path = tmp_path / "sorgak_fragment.xyz"
    path.write_text(_SORGAK_BRIDGEHEAD_FRAGMENT)

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz(str(path), charge=1, connectivity="xyzgraph", bond_orders="xyz2mol")

    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
    kept = sorted(n.GetSymbol() for n in mol.GetAtomWithIdx(metal).GetNeighbors())
    assert kept == ["O", "S"]
    assert "bridgehead" in caplog.text
