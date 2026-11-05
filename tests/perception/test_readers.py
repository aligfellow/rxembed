"""`perception/readers.py`: XYZ geometries and bonded graphs read into Mols, their backends, fallbacks and guards."""

import itertools
import logging
import math
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.perception import readers
from rxembed.perception.xyz2mol_tmc import get_tmc_mol
from tests.geometry_fixtures import embedded


def _atoms(block):
    """Return the ``(symbol, (x, y, z))`` atoms of the coordinate lines of an .xyz block."""
    return [(row.split()[0], tuple(map(float, row.split()[1:4]))) for row in block.strip().splitlines()]


CISPLATIN = _atoms("""
N 1.03 -1.14 -0.19
Pt 0.45 0.84 0.05
N -1.52 0.21 0.14
Cl -0.19 2.94 0.62
Cl 2.58 1.51 -0.37
H 0.46 -1.86 0.30
H 0.80 -1.21 -1.21
H 2.02 -1.44 -0.09
H -2.32 0.88 0.06
H -1.50 -0.11 1.13
H -1.78 -0.62 -0.44
""")
ETHANOL = readers.xyz_atoms(embedded("CCO"))
# [VO2Cl2]- read without its charge, as PIVPOB's dioxo V(V) was: V+6 at charge=0
VANADYL_DICHLORIDE = _atoms("""
V 0 0 0
O 0.924 0.924 0.924
O 0.924 -0.924 -0.924
Cl -1.270 1.270 -1.270
Cl -1.270 -1.270 1.270
""")
# Pt(PH3)2(eta2-O2): Pt(II) with a peroxide, or Pt(0) with neutral O2, as xyzgraph's own orders read it
PLATINUM_DIOXYGEN = _atoms("""
Pt 0.000 0.000 0.000
P -1.600 1.610 0.000
P 1.600 1.610 0.000
H -0.984 2.883 0.000
H -2.404 1.472 -1.156
H -2.404 1.472 1.156
H 2.877 1.001 0.000
H 1.457 2.413 -1.156
H 1.457 2.413 1.156
O -0.720 -1.870 0.000
O 0.720 -1.870 0.000
""")
# rxflow's repro: a carbocation zwitterion xyzgraph orders read at -2 against charge 0
CARBOCATION_ZWITTERION = _atoms("""
C -1.78367991 -0.94484834 1.62723587
C -1.48786183 -0.09599687 0.38534858
C -1.76641282 1.35270574 0.66624355
C -0.76900289 2.31277703 0.77096139
O -1.02397006 3.48581580 1.09011472
N 0.54914200 1.92524261 0.49554493
C 0.90716403 0.67027112 0.00933098
C 2.20598792 0.41672488 -0.38842787
C 2.58432749 -0.83672330 -0.93718689
O 3.75323875 -1.06363090 -1.29364373
C 1.56260269 -1.82343165 -1.06482359
C 0.28191803 -1.56565409 -0.65889001
C -0.07334262 -0.33862089 -0.11067325
H -2.81087011 -0.81047613 1.96768304
H -1.62835513 -2.00700458 1.43724313
H -1.12430030 -0.65679541 2.44566418
H -2.16567193 -0.41111791 -0.42712050
H -2.77461522 1.66406787 0.86724850
H 1.24887505 2.65229799 0.55889270
H 2.95595030 1.17857105 -0.29813421
H 1.83078512 -2.77636106 -1.48415299
H -0.47190857 -2.32781297 -0.76012566
""")
# O-H...O at 1.40 A: xyzgraph's quick read bonds the hydrogen to both oxygens
SHORT_WATER_DIMER = _atoms("""
O 0.000 0.000 0.000
H 1.000 0.000 0.000
H -0.240 0.927 0.000
O 2.400 0.000 0.000
H 2.740 0.000 0.900
H 2.740 0.000 -0.900
""")
# PtCl(eta2-ethylene)(Et(Me)P-CH2CH2-P(Me)Et)+ as rx.embed placed it, both P hands specified
PLATINUM_ETHYLENE_DIPHOSPHINE = "[Pt+2]12(<-[Cl-])(<-[CH2]=[CH2]->1)<-[P@](C)(CC)CC[P@@](C)(CC)->2"
PLATINUM_ETHYLENE_DIPHOSPHINE_ATOMS = _atoms("""
Pt 0.16 1.50 0.70
Cl 1.72 2.88 1.62
C -1.25 3.17 0.58
C -1.39 2.62 1.79
P 1.79 0.06 -0.02
C 2.47 -0.81 1.54
C 3.39 0.34 -1.05
C 4.21 -0.93 -1.26
C 0.96 -1.39 -0.96
C -0.39 -1.67 -0.33
P -1.34 -0.03 -0.10
C -1.93 0.35 -1.88
C -2.97 -0.63 0.73
C -3.62 -1.81 0.03
H -1.95 2.93 -0.21
H -0.66 4.07 0.45
H -0.91 3.07 2.66
H -2.20 1.93 1.99
H 3.25 -0.17 2.01
H 2.89 -1.80 1.29
H 1.64 -0.94 2.26
H 4.01 1.09 -0.53
H 3.10 0.75 -2.04
H 3.63 -1.69 -1.83
H 4.54 -1.35 -0.29
H 5.12 -0.68 -1.85
H 1.59 -2.30 -0.91
H 0.83 -1.10 -2.02
H -0.96 -2.36 -1.00
H -0.25 -2.15 0.66
H -2.78 1.06 -1.85
H -1.09 0.81 -2.45
H -2.24 -0.58 -2.40
H -3.68 0.22 0.75
H -2.74 -0.91 1.78
H -3.84 -1.57 -1.03
H -2.98 -2.71 0.09
H -4.58 -2.05 0.54
""")


_QUINOLINYL_PD = "c1c[c-]3c24n(->[Pd+2]<-3<-4)cccc2c1"  # C8 (2) and N1 (4) on Pd5, C8a (3) bonded to both


def _no_reading(*_args, **_kwargs):
    raise ValueError("no reading")


def _xyzgraph_reads(monkeypatch, build_graph):
    """Stand in for xyzgraph with `build_graph`, so `read_xyz` judges the graph a coordinate reader can draw."""
    monkeypatch.setitem(sys.modules, "xyzgraph", SimpleNamespace(build_graph=build_graph, __version__="9"))


def _xyz2mol_reads(monkeypatch, read):
    """Stand in for the vendored xyz2mol perceiver with ``read(path, charge, graph)``, a Mol or an error."""
    monkeypatch.setattr(readers, "get_tmc_mol", lambda path, charge, graph=None, **_kw: (read(path, charge, graph),))


def _drawn(selected):
    """Return a `build_graph` drawing `selected`: its atoms with hydrogens added, its conformer and its bonds."""
    mol = Chem.Mol(selected)
    mol.UpdatePropertyCache(strict=False)
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    mol = Chem.AddHs(mol, addCoords=bool(mol.GetNumConformers()), onlyOnAtoms=heavy)
    if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
        Chem.Kekulize(mol, clearAromaticFlags=True)
    positions = mol.GetConformer().GetPositions() if mol.GetNumConformers() else np.zeros((mol.GetNumAtoms(), 3))
    graph = nx.Graph(total_charge=Chem.GetFormalCharge(mol))
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_number=atom.GetAtomicNum(), formal_charge=atom.GetFormalCharge())
        graph.nodes[atom.GetIdx()]["position"] = positions[atom.GetIdx()]
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_order=max(1, int(bond.GetBondTypeAsDouble())))
    return lambda *_args, **_kwargs: graph


def _four_ring(smiles, leg, arm, apex, fold):
    """Return `smiles` (atoms D1, X, D2, M first) with a ring M-D1-X-D2 folded `fold` deg about D1...D2.

    `leg` is M-D, `arm` X-D, `apex` the D1-X-D2 angle; at fold 0 the ring is flat with M opposite X.
    """
    mol = rx.parse_smiles(smiles, remove_hs=False)
    half = math.radians(apex) / 2
    span, h_x = arm * math.sin(half), arm * math.cos(half)
    h_m, phi = math.sqrt(leg**2 - span**2), math.radians(fold)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(
        [(0, span, 0), (h_x, 0, 0), (0, -span, 0), (-h_m * math.cos(phi), 0, h_m * math.sin(phi))]
    ):
        conformer.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conformer, assignId=True)
    return mol


def _quinolinyl_palladium():
    """Return the kappa2-N,C quinolin-8-yl Pd with the wrong Pd-C8a contact, on the flat ring without it.

    C8, C8a, N1 and Pd come first, as `_four_ring` numbers D1, X, D2 and M.
    """
    selected = rx.parse_smiles(_QUINOLINYL_PD, remove_hs=False)
    ring = Chem.RWMol(selected)
    ring.RemoveBond(5, 3)
    ring = ring.GetMol()
    ring.UpdatePropertyCache(strict=False)
    assert rdDistGeom.EmbedMolecule(ring, randomSeed=7) == 0
    selected.AddConformer(ring.GetConformer(), assignId=True)
    return Chem.RenumberAtoms(selected, [2, 3, 4, 5, 0, 1, *range(6, selected.GetNumAtoms())])


def _bond_graph(mol):
    """Return the canonical SMILES of `mol`'s bonds alone: every bond single, no charges, radicals or stereo."""
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
    rw.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(rw, isomericSmiles=False)


def _assembled(reactants, formed=(), cut=()):
    """Return the reactants' joint graph as a reaction assembler edits it: explicit H, single bonds, no charges.

    `formed` and `cut` name bonds by the atom maps of `reactants`, which the graph does not keep.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(reactants))
    at = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
    graph = Chem.RWMol()
    for atom in mol.GetAtoms():
        bare = Chem.Atom(atom.GetAtomicNum())
        bare.SetNoImplicit(True)
        graph.AddAtom(bare)
    for i, j in [
        *((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()),
        *((at[i], at[j]) for i, j in formed),
    ]:
        graph.AddBond(i, j, Chem.BondType.SINGLE)
    for i, j in cut:
        graph.RemoveBond(at[i], at[j])
    return graph.GetMol()


def _read_by(atoms, **options):
    """Return the connectivity and bond-order backends `read_xyz` records for `atoms`, and whether one fell back."""
    mol = rx.read_xyz(atoms, **options)
    connectivity, orders = mol.GetProp("_rxembedConnectivity"), mol.GetProp("_rxembedBondOrders")
    return connectivity, orders, mol.GetBoolProp("_rxembedPerceptionFallback")


def test_read_xyz_refuses_an_unreadable_option():
    for options, message in (
        ({"connectivity": "typo"}, "connectivity must be 'rdkit', 'xyzgraph', or 'xyz2mol'"),
        ({"bond_orders": "auto"}, "bond_orders must be 'xyzgraph', 'xyz2mol', or None"),
        ({"connectivity": "rdkit", "bond_orders": "xyz2mol"}, "connectivity='rdkit' needs charge="),
        ({"charge": 0, "connectivity": "xyz2mol"}, "needs connectivity='xyzgraph'"),
        ({"charge": 0, "metal_charges": {0: 2}}, r"metal_charges must name every metal atom index \[1\]"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.read_xyz(CISPLATIN, **options)


def test_read_xyz_refuses_atoms_that_are_no_geometry():
    for atoms, error, message in (
        ([("Xx", (0.0, 0.0, 0.0))], ValueError, "not an element symbol"),
        ([("C", (0.0, 0.0))], ValueError, "three finite coordinates"),
        ([("C", (0.0, math.nan, 0.0))], ValueError, "three finite coordinates"),
        ([], ValueError, "no atoms"),
        ([("C", 0.0)], TypeError, r"an \.xyz path or atoms as \(symbol, \(x, y, z\)\) pairs"),
    ):
        with pytest.raises(error, match=message):
            rx.read_xyz(atoms, charge=0)


def test_read_xyz_reads_atoms_in_memory_as_their_file(tmp_path):
    path = tmp_path / "cisplatin.xyz"
    path.write_text(f"{len(CISPLATIN)}\n\n" + "".join(f"{s} {x} {y} {z}\n" for s, (x, y, z) in CISPLATIN))
    for connectivity in ("xyzgraph", "xyz2mol", "rdkit"):
        options = {"charge": 0, "connectivity": connectivity, "bond_orders": "xyz2mol", "fallback": False}
        from_file, in_memory = (rx.read_xyz(source, **options) for source in (path, CISPLATIN))
        assert Chem.MolToSmiles(in_memory) == Chem.MolToSmiles(from_file), connectivity
        np.testing.assert_allclose(in_memory.GetConformer().GetPositions(), [xyz for _symbol, xyz in CISPLATIN])


def test_read_xyz_bonds_each_donor_to_its_metal_by_a_dative_bond():
    mol = rx.read_xyz(CISPLATIN, charge=0)

    on_metal = [bond for bond in mol.GetBonds() if 1 in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]
    assert [(bond.GetBondType(), bond.GetEndAtomIdx()) for bond in on_metal] == [(Chem.BondType.DATIVE, 1)] * 4


def test_read_xyz_reads_each_geometry_as_its_species():
    """An ion reads at the total xyzgraph reads, a radical at the charge given, and a hydrogen bond is no bond."""
    for name, atoms, options, smiles in (
        ("acetate", readers.xyz_atoms(embedded("CC(=O)[O-]")), {}, "CC(=O)[O-]"),
        (
            "tempo-stated-neutral",
            readers.xyz_atoms(embedded("CC1(C)CCCC(C)(C)N1[O]")),
            {"charge": 0},
            "CC1(C)CCCC(C)(C)N1[O]",
        ),
        ("short-water-dimer", SHORT_WATER_DIMER, {}, "O.O"),
    ):
        assert rx.smiles(rx.read_xyz(atoms, **options)) == rx.smiles(rx.parse_smiles(smiles)), name


def test_read_xyz_without_bond_orders_reads_the_bonds_alone():
    """A zwitterion reads with no charge placed, and a hydrogen the quick read bonds twice keeps one bond."""
    for atoms, smiles in ((CARBOCATION_ZWITTERION, "CC1[CH+]C(=O)Nc2cc([O-])ccc21"), (SHORT_WATER_DIMER, "O.O")):
        mol = rx.read_xyz(atoms, 0, bond_orders=None)
        assert _bond_graph(mol) == _bond_graph(Chem.AddHs(Chem.MolFromSmiles(smiles))), smiles
        assert not any(atom.GetFormalCharge() for atom in mol.GetAtoms()), smiles


def test_read_xyz_records_its_backends_and_their_fallback(monkeypatch):
    """xyz2mol connectivity needs a d-block metal, and xyz2mol orders one it has a charge model for."""
    magnesium_dichloride = [("Mg", (0, 0, 0)), ("Cl", (2.2, 0, 0)), ("Cl", (-2.2, 0, 0))]
    orders = {"charge": 0, "bond_orders": "xyz2mol"}
    for name, atoms, options, expected in (
        ("cisplatin", CISPLATIN, {"charge": 0}, ("xyzgraph", "xyzgraph", False)),
        ("ethanol-xyz2mol", ETHANOL, orders | {"connectivity": "xyz2mol"}, ("xyzgraph", "xyzgraph", True)),
        ("ethanol-rdkit-connectivity", ETHANOL, orders | {"connectivity": "rdkit"}, ("rdkit", "rdkit", True)),
        ("MgCl2 xyz2mol orders", magnesium_dichloride, orders, ("xyzgraph", "xyzgraph", True)),
    ):
        assert _read_by(atoms, **options) == expected, name
    _xyz2mol_reads(monkeypatch, lambda path, charge, graph: _no_reading() if graph else get_tmc_mol(path, charge)[0])
    assert _read_by(CISPLATIN, **orders) == ("xyz2mol", "xyz2mol", True), "xyz2mol orders no xyzgraph graph"
    _xyzgraph_reads(monkeypatch, _no_reading)
    for name, atoms, expected in (("cisplatin", CISPLATIN, "xyz2mol"), ("ethanol", ETHANOL, "rdkit")):
        assert _read_by(atoms, charge=0) == (expected, expected, True), f"xyzgraph failed on {name}"


def test_read_xyz_refuses_where_no_backend_may_read(monkeypatch):
    rdkit, xyz2mol = ({"charge": 0, "connectivity": name, "bond_orders": "xyz2mol"} for name in ("rdkit", "xyz2mol"))
    for atoms, options, message in (
        (ETHANOL, xyz2mol | {"fallback": False}, "no TM"),
        (ETHANOL, rdkit | {"fallback": False}, r"bond_orders='xyz2mol' cannot rank this graph"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.read_xyz(atoms, **options)
    _xyz2mol_reads(monkeypatch, _no_reading)
    with pytest.raises(ValueError, match=r"no bond orders on the rdkit graph .*xyz2mol connectivity also failed"):
        rx.read_xyz(CISPLATIN, **rdkit)  # an RDKit graph has no charges of its own to keep
    _xyzgraph_reads(monkeypatch, _no_reading)
    for options, message in (
        ({}, "xyzgraph failed .*; pass charge= to read with RDKit"),
        ({"charge": 0, "fallback": False}, "no reading"),
        ({"charge": 1}, "RDKit fallback also failed .*use connectivity='xyzgraph'"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.read_xyz(ETHANOL, **options)


def test_read_xyz_keeps_fewest_radicals_without_a_closed_shell(monkeypatch):
    """V+6 at charge=0 reads V+5 with an oxo radical in any atom order; with no xyz2mol reading, xyzgraph's own."""
    readings = [
        rx.read_xyz(atoms, 0, bond_orders="xyz2mol") for atoms in (VANADYL_DICHLORIDE, VANADYL_DICHLORIDE[::-1])
    ]
    _xyz2mol_reads(monkeypatch, _no_reading)
    readings.append(rx.read_xyz(CISPLATIN, 0, bond_orders="xyz2mol"))

    found = [
        (
            Chem.MolToSmiles(mol),
            [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons()],
            mol.GetProp("_rxembedBondOrders"),
            bool(mol.GetProp("_rxembedChargeRescue")),
        )
        for mol in readings
    ]
    vanadium = ("[O-2]->[V+5](<-[O-])(<-[Cl-])<-[Cl-]", ["O"], "xyz2mol", True)
    assert found == [vanadium, vanadium, (Chem.MolToSmiles(rx.read_xyz(CISPLATIN, 0)), [], "xyzgraph", True)]


def test_read_xyz_refuses_a_reading_that_misses_the_charge(monkeypatch):
    """xyzgraph's orders keep a neutral O2 under a named Pt(II), so the total misses; xyz2mol would rank a peroxide."""
    with pytest.raises(ValueError, match="does not match charge=0; use bond_orders='xyz2mol' for the named"):
        rx.read_xyz(PLATINUM_DIOXYGEN, 0, metal_charges={0: 2})
    _xyzgraph_reads(monkeypatch, _drawn(Chem.MolFromSmiles("C")))
    with pytest.raises(ValueError, match="does not match charge=1; check charge="):
        rx.read_xyz("unused.xyz", charge=1)


def test_read_xyz_ranks_the_ligands_on_the_metal_charges_named():
    mol = rx.read_xyz(PLATINUM_DIOXYGEN, 0, bond_orders="xyz2mol", metal_charges={0: 2})

    assert [atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetSymbol() in {"Pt", "O"}] == [2, -1, -1]


def test_read_xyz_at_charge_zero_warns_of_a_missing_metal_charge(monkeypatch, caplog):
    """A d10 Zn(II) suggests nothing; a metal past its valence electrons, negative or with an odd count does."""
    for smiles, warned in (
        ("[Cl-]->[Zn+2]<-[Cl-]", False),
        ("[I-]->[Hg+3](<-[I-])<-[I-]", True),
        ("[O+]#[C-]->[Co-](<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+].[Na+]", True),
        ("[Cl-]->[Cu+2]<-[Cl-]", True),
    ):
        _xyzgraph_reads(monkeypatch, _drawn(Chem.MolFromSmiles(smiles)))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rxembed"):
            rx.read_xyz("unused.xyz", charge=0)
        assert ("pass charge= for a charged complex" in caplog.text) == warned, smiles


def test_read_xyz_drops_only_chelate_ring_diagonal_metal_bonds(monkeypatch):
    """X bonded to two donors of M is a ring corner unless it is linear, binds a donor without a lone pair, lies in a
    ring face, beside an open carbon, or folded inside the M-D legs; a saturated X always is one."""
    for name, selected, kept in (
        ("flat-dithioformate-nickel", _four_ring("[S]1=[CH]2[S-]->[Ni+2]<-1<-2", 2.22, 1.72, 114.0, 0.0), False),
        ("folded-dithioformate-nickel", _four_ring("[S]1=[CH]2[S-]->[Ni+2]<-1<-2", 2.22, 1.72, 114.0, 80.0), True),
        ("folded-silanedithiolate-nickel", _four_ring("[S-]1[SiH2]2[S-]->[Ni+2]<-1<-2", 2.22, 2.1, 100.0, 80.0), False),
        ("triphosphaallyl-zirconium", _four_ring("[P]1#[P]2=[P-]->[Zr+2]<-1<-2", 2.54, 2.12, 118.0, 0.0), True),
        ("flat-azaallyl-palladium", _four_ring("[CH2-]1[CH]2=[NH]->[Pd+2]<-1<-2", 2.1, 1.4, 120.0, 0.0), True),
        ("flat-allyl-palladium", _four_ring("[CH2-]1[CH]2=[CH2]->[Pd+2]<-1<-2", 2.1, 1.4, 120.0, 0.0), True),
        ("kappa2 pyrimidine Pd face", _four_ring("n12[cH]3[n](->[Pd+2]<-1<-3)ccc2", 2.1, 1.34, 115.0, 0.0), True),
        ("quinolinyl Pd junction", _quinolinyl_palladium(), False),
    ):
        _xyzgraph_reads(monkeypatch, _drawn(selected))
        mol = rx.read_xyz("unused.xyz", charge=Chem.GetFormalCharge(selected))
        assert (mol.GetBondBetweenAtoms(1, 3) is not None) == kept, name


def test_read_xyz_keeps_a_second_h_leg_only_off_an_h_bond(monkeypatch):
    """N-H...O is a hydrogen bond and leaves the graph; B-H-B is a three-centre bond and keeps its second leg, and a
    B-H-Fe bridge its dative one."""
    _xyz2mol_reads(monkeypatch, lambda _path, _charge, graph: graph[0])  # xyz2mol orders nothing, so the legs decide
    single, dative, zero = Chem.BondType.SINGLE, Chem.BondType.DATIVE, Chem.BondType.ZERO
    for left, right, legs in ((7, 8, [single]), (5, 5, sorted([single, zero])), (5, 26, sorted([single, dative]))):
        found = []
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
            conformer = Chem.Conformer(4)
            conformer.SetPositions(np.array([(5, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0)], float))
            mol.AddConformer(conformer)
            _xyzgraph_reads(monkeypatch, _drawn(mol))
            ranked = rx.read_xyz("unused.xyz", bond_orders="xyz2mol")
            found.append({bond.GetOtherAtomIdx(2): bond.GetBondType() for bond in ranked.GetAtomWithIdx(2).GetBonds()})
        assert found[0] == found[1], (left, right)
        assert sorted(found[0].values()) == legs, (left, right)


def test_read_xyz_refuses_an_unbonded_h_between_heavy_atoms(monkeypatch):
    """SIJQIL analogue: H2, about 1.35 A from each boron, is a 3c-2e bridge; H3, dropped beside one boron, is not."""
    bridge = Chem.RWMol()
    for atomic_number in (5, 5, 1, 1):
        bridge.AddAtom(Chem.Atom(atomic_number))
    bridge = bridge.GetMol()
    conformer = Chem.Conformer(4)
    conformer.SetPositions(np.array([(0, 0, 0), (1.8, 0, 0), (0.9, 1.006, 0), (-1.2, 0, 0)], float))
    bridge.AddConformer(conformer)
    _xyzgraph_reads(monkeypatch, _drawn(bridge))

    with pytest.raises(ValueError, match=r"3-centre bridge\(s\) H2 to 0 \(1\.350 A\) and 1 \(1\.350 A\) lie outside"):
        rx.read_xyz("unused.xyz", charge=0)


def test_read_xyz_refuses_a_graph_outside_the_kekule_contract(monkeypatch):
    """An order-0 edge without its NCI flag, as a renamed flag would give a hydrogen bond, or charges off the total."""
    unflagged, off_total = nx.Graph(total_charge=0), nx.Graph(total_charge=1)
    for graph in (unflagged, off_total):
        for node, (atomic_number, x) in enumerate(((8, 0.0), (1, 1.0))):
            graph.add_node(node, atomic_number=atomic_number, formal_charge=0, position=(x, 0.0, 0.0))
    unflagged.add_edge(0, 1, bond_order=0.0, NCI=False)
    off_total.add_edge(0, 1, bond_order=1)
    for graph in (unflagged, off_total):
        _xyzgraph_reads(monkeypatch, lambda *_args, graph=graph, **_kwargs: graph)
        with pytest.raises(ValueError, match="install the version rxembed requires"):
            rx.read_xyz("unused.xyz", fallback=False)


def test_read_xyz_reads_back_a_pt_diphosphines_p_hands():
    (species,) = rx.isomers(PLATINUM_ETHYLENE_DIPHOSPHINE, geometry="SPL")

    fresh = rx.read_xyz(PLATINUM_ETHYLENE_DIPHOSPHINE_ATOMS, 1, bond_orders="xyz2mol")

    assert rx.cxsmiles(fresh) == rx.cxsmiles(species)


def test_read_graph_gives_the_lewis_structure_of_a_bonded_graph():
    """A product graph and its total read as the species, in one of its resonance forms, keeping every given bond."""
    for reactants, formed, cut, charge, product in (
        ("C[NH2:1].[CH3:2][Br:3]", [(1, 2)], [(2, 3)], 0, "C[NH2+]C.[Br-]"),
        (
            "[PH3]->[Pd:1]<-[PH3].[CH3:2][Br:3]",
            [(1, 2), (1, 3)],
            [(2, 3)],
            0,
            "[CH3-]->[Pd+2](<-[PH3])(<-[PH3])<-[Br-]",
        ),
        # the Ni-C diagonal a geometry would drop is kept; the fewest charges read Ni(I), a geometry Ni(II)
        ("[S]1=[CH]2[S-]->[Ni+2]<-1<-2", [], [], 0, "[S]1=[CH]2[S-]->[Ni+]<-1<-2"),
    ):
        forms = {
            rx.smiles(rx.parse_smiles(Chem.MolToSmiles(form)))
            for form in Chem.ResonanceMolSupplier(rx.parse_smiles(product), Chem.KEKULE_ALL)
        }
        mol = rx.read_graph(_assembled(reactants, formed, cut), charge)
        assert rx.smiles(mol) in forms, product
        assert (Chem.GetFormalCharge(mol), mol.GetNumConformers()) == (charge, 0), product


def test_read_graph_refuses_a_graph_it_cannot_read():
    cage = Chem.RWMol(_assembled(".".join(["[BH]"] * 6)))  # closo B6H6 2-: each B bonds four B and one H
    for i, j in itertools.combinations(range(6), 2):
        if (i, j) not in {(0, 1), (2, 3), (4, 5)}:  # the three opposite vertex pairs
            cage.AddBond(i, j, Chem.BondType.SINGLE)
    for graph, charge, metal_charges, message in (
        (Chem.MolFromSmiles("CC"), 0, None, r"Chem\.AddHs \(counts on C0, C1\)"),
        (_assembled("O"), None, None, "rx.read_graph needs charge="),
        (_assembled("O"), 1, None, "RDKit could not assign bond orders.*check charge="),
        (_assembled("Cl[Pd]Cl"), 0, {1: 3}, "perceived total charge 1 does not match charge=0; check charge="),
        (_assembled("Cl[Pd]1(Cl)Cl[Pd](Cl)(Cl)Cl1"), -2, None, "name each metal's charge with metal_charges="),
        (cage, -2, None, "multi-centre cage vertex"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.read_graph(graph, charge, metal_charges=metal_charges)


def test_read_graph_splits_a_cl_bridged_pd_dimer_as_named():
    mol = rx.read_graph(_assembled("Cl[Pd]1(Cl)Cl[Pd](Cl)(Cl)Cl1"), -2, metal_charges={1: 2, 4: 2})

    assert [mol.GetAtomWithIdx(pd).GetFormalCharge() for pd in (1, 4)] == [2, 2]


def test_read_graph_reads_an_h_bond_to_a_lone_pair_as_a_contact():
    """A water H also bonded to the other water's O, which keeps a lone pair, is a contact, not a bond."""
    graph = Chem.RWMol(_assembled("O.O"))
    graph.AddBond(4, 0, Chem.BondType.SINGLE)

    assert rx.smiles(rx.read_graph(graph, 0)) == rx.smiles(rx.parse_smiles("O.O"))
