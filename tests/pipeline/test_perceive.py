"""Test XYZ graph perception and backend fallback behavior."""

import logging
import math
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.metal_core import COORDINATION_METALS
from rxembed.pipeline import perceive
from tests.conftest import EXAMPLES_DIR

_BIMP = str(EXAMPLES_DIR / "bimp.xyz")  # a metal-free TS with a stretched reacting core


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
    ids=["dioxovanadium_dichloride"],
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


def test_side_on_disulfide_bond_restored_by_consensus_reads_closed_shell(tmp_path, caplog):
    """WICHIZ analogue: xyzgraph drops an eta2-S2 S-S bond that closes a three-ring through the metal.

    RDKit and xyz2mol both find it, so xyzgraph re-reads with it and assigns its Lewis form on that graph,
    not the two sulfur radicals it gave without the bond.
    """
    atoms = [
        ("Nb", 0.0, 0.0, 0.0),
        ("S", 2.35, 1.06, 0.0),
        ("S", 2.35, -1.06, 0.0),
        ("Cl", -2.4, 0.0, 0.0),
        ("Cl", 0.0, 0.0, 2.4),
        ("Cl", 0.0, 0.0, -2.4),
    ]
    path = _write_complex(tmp_path, "NbS2Cl3", atoms, range(len(atoms)))

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz(path, 0)

    assert mol.GetBondBetweenAtoms(1, 2) is not None
    assert mol.GetProp("_rxembedConnectivityAdded") == "1-2"
    assert "added 1-2 internal ligand bonds confirmed" in caplog.text
    assert not any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())


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


@pytest.mark.parametrize(("fold", "kept"), [pytest.param(0.0, False, id="kappa2-flat")])
def test_read_xyz_keeps_a_dithiocarboxylate_carbon_only_once_the_fold_brings_it_inside_the_sulfur_legs(
    monkeypatch, fold, kept
):
    """The S-C-S carbon keeps its Ni bond only when it sits closer to Ni than the Ni-S bonds do.

    Ni-S 2.22 A, S-C 1.72 A, S-C-S 114 deg: Ni...C is 2.62 A flat, 2.33 A at QAHFOV's 57 deg fold (a kappa2
    dithiocarbamate the reader bonds anyway) and 2.07 A at 80 deg, the carbon of an eta3-S,C,S ligand.
    """
    selected = _four_ring("[S]1=[CH]2[S-]->[Ni+2]<-1<-2", leg=2.22, arm=1.72, apex=114.0, fold=fold)

    result = _read_graph(monkeypatch, selected)

    assert (result.GetBondBetweenAtoms(1, 3) is not None) == kept


def test_read_xyz_keeps_the_linear_centre_of_a_triphosphaallyl_in_the_chelate_plane(monkeypatch):
    """POBSUU in miniature: a linear P keeps a pi orbital in the plane, so 1.12x its legs does not drop it."""
    selected = _four_ring("[P]1#[P]2=[P-]->[Zr+2]<-1<-2", leg=2.54, arm=2.12, apex=118.0, fold=0.0)

    result = _read_graph(monkeypatch, selected)

    assert result.GetBondBetweenAtoms(1, 3) is not None


def test_read_xyz_drops_the_ring_junction_of_a_four_membered_quinolinyl_chelate(monkeypatch):
    """A ring junction is no face: its donors lie in two rings, so the wrong Pd-C8a contact drops."""
    selected = rx.parse_smiles(_QUINOLINYL_PD, remove_hs=False)
    pd, c8a = 5, 3
    rw = Chem.RWMol(selected)
    rw.RemoveBond(pd, c8a)
    ring = rw.GetMol()
    ring.UpdatePropertyCache(strict=False)
    assert rdDistGeom.EmbedMolecule(ring, randomSeed=7) == 0  # the flat kappa2 ring, without the diagonal
    selected.AddConformer(ring.GetConformer(), assignId=True)

    result = _read_graph(monkeypatch, selected)

    assert sorted(n.GetSymbol() for n in result.GetAtomWithIdx(pd).GetNeighbors()) == ["C", "N"]


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


@pytest.mark.parametrize(
    ("smiles", "charge", "warning"),
    [
        ("[I-]->[Hg+3](<-[I-])<-[I-]", 0, "Hg+3 is over its 2 valence electrons"),  # xyzgraph's HgI3
        ("[Cl-]->[Zn+2]<-[Cl-]", 0, None),  # d10 Zn(II): nothing suggests a missing charge
    ],
    ids=["mercury_triiodide", "zinc_dichloride"],
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


def _square_planar_xyz(tmp_path, smiles):
    """Write an embedded square-planar geometry of `smiles` to `tmp_path` and return its path."""
    ens = rx.embed(rx.metal(smiles, "square_planar")[0], n=1, seed=1)
    path = tmp_path / "complex.xyz"
    Chem.MolToXYZFile(ens.mol, str(path), confId=ens.ids[0])
    return str(path)


def _cisplatin_xyz(tmp_path):
    """Write a cisplatin (Pt(NH3)2Cl2) geometry to `tmp_path`, exercising every metal-aware perceiver pair."""
    return _square_planar_xyz(tmp_path, "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_tetraammineplatinum_default_read_takes_xyz2mol_orders_when_xyzgraph_misses_the_total(tmp_path, caplog):
    """xyzgraph reads [Pt(NH3)4]2+ as Pt(0); the default read keeps its graph and takes xyz2mol's charges."""
    path = _square_planar_xyz(tmp_path, "[NH3]->[Pt+2](<-[NH3])(<-[NH3])<-[NH3]")

    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz(path, charge=2)

    assert [(a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge()] == [("Pt", 2)]
    assert (mol.GetProp("_rxembedConnectivity"), mol.GetProp("_rxembedBondOrders")) == ("xyzgraph", "xyz2mol")
    assert mol.GetBoolProp("_rxembedPerceptionFallback")
    assert "xyzgraph's charges total 0, not charge=2" in caplog.text
    with pytest.raises(ValueError, match="does not match charge=2"):
        perceive.read_xyz(path, charge=2, fallback=False)


@pytest.mark.parametrize(
    ("connectivity", "bond_orders"),
    [
        ("xyzgraph", "xyzgraph"),
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


def test_rdkit_connectivity_uses_native_bond_orders_for_an_organic_graph(caplog):
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        mol = perceive.read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol")

    assert any(bond.GetBondTypeAsDouble() > 1 for bond in mol.GetBonds())
    assert "xyz2mol bond-order assignment is metal-only; using RDKit" in caplog.text
    with pytest.raises(ValueError, match=r"xyz2mol.*does not support"):
        perceive.read_xyz(_BIMP, 0, connectivity="rdkit", bond_orders="xyz2mol", fallback=False)


def test_read_xyz_rejects_a_final_charge_different_from_the_request(monkeypatch):
    selected = Chem.MolFromSmiles("C")
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))

    with pytest.raises(ValueError, match="does not match charge=1; use bond_orders='xyz2mol'"):
        perceive.read_xyz("unused.xyz", charge=1, bond_orders="xyzgraph")


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

        monkeypatch.setattr(perceive, "_with_fallback", lambda *_args, mol=mol: (mol, "rdkit"))
        ranked = perceive.read_xyz("unused.xyz", connectivity="rdkit", bond_orders="xyz2mol")
        legs = {bond.GetOtherAtomIdx(2): bond.GetBondType() for bond in ranked.GetAtomWithIdx(2).GetBonds()}
        outcomes.append(legs)

    assert outcomes[0] == outcomes[1]
    assert sorted(outcomes[0].values()) == (
        [Chem.BondType.SINGLE, Chem.BondType.ZERO] if bridge else [Chem.BondType.SINGLE]
    )


# --- _drop_bridgehead_bonds: a reader-only fix for a bond across a chelate ring's diagonal ---------------
#
# `rx.parse_smiles` and hand-built RWMols are both used only as convenient ways to construct a graph
# shaped like what a coordinate-derived reader can mis-bond. A saturated bridgehead and a non-bridgehead are
# settled by the graph alone; any other bridgehead is judged on coordinates, which those tests set by hand.
# A real user SMILES is never routed through this guard: only `read_xyz` calls it.

# A saturated bridgehead: 4+ non-metal sigma bonds and no lone pair of its own.


def _four_ring(smiles, leg, arm, apex, fold):
    """Return `smiles` (atoms D1, X, D2, M in that order) with a ring M-D1-X-D2 folded `fold` deg about D1...D2.

    `leg` is M-D, `arm` X-D, `apex` the D1-X-D2 angle; at fold 0 the ring is flat with M opposite X.
    """
    mol = rx.parse_smiles(smiles, remove_hs=False)
    half = math.radians(apex) / 2
    span, h_x = arm * math.sin(half), arm * math.cos(half)
    h_m, phi = math.sqrt(leg**2 - span**2), math.radians(fold)
    metal = (-h_m * math.cos(phi), 0, h_m * math.sin(phi))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate([(0, span, 0), (h_x, 0, 0), (0, -span, 0), metal]):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf, assignId=True)
    return mol


def _read_graph(monkeypatch, selected):
    monkeypatch.setattr(perceive, "_with_fallback", lambda *_args: (selected, "xyzgraph"))
    return perceive.read_xyz("unused.xyz", charge=Chem.GetFormalCharge(selected))


# AREPUK's failure mode in miniature: a quinolin-8-yl kappa2-N,C chelate, its ring-junction C8a bonded
# to both real donors (N1, the anionic ipso C8) directly, plus the wrong Pd-C8a contact a coordinate
# reader can add. N1 shares the pyridine ring with C8a, C8 shares the benzo ring with C8a: two different
# rings fused at C8a, not one ring holding every donor.
_QUINOLINYL_PD = "c1c[c-]3c24n(->[Pd+2]<-3<-4)cccc2c1"


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
