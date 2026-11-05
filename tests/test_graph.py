"""`graph.py`: the graph facts every layer shares about metals and the ligands around them."""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom

from rxembed import graph
from rxembed.identity.smiles_io import parse_smiles
from rxembed.utils import EZ_RELATION, TETRAHEDRAL_TAGS
from tests.geometry_fixtures import placed

# PtCl(eta2-ethylene)(diphosphine) as a seed draws it: only the stereo donors bound to Pt
PT_DIPHOSPHINE_ETHYLENE = "[Pt+2]1<-[P@](C)(CC)CC[P@@](C)(CC)->1.[Cl-].C=C"


def _unsanitized(smiles):
    """Return `smiles` as a connectivity read writes it, every bond single, its valences computed."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _with_zero_bond(mol, i, j):
    """Return a copy of `mol` whose atoms `i` and `j` are joined by a zero-order contact, as a reader writes one."""
    rw = Chem.RWMol(mol)
    rw.RemoveBond(i, j)
    rw.AddBond(i, j, Chem.BondType.ZERO)
    return rw.GetMol()


def _real_bounds(mol, count):
    """Return the native DG bounds among the first `count` atoms of `mol`."""
    with rdBase.BlockLogs():
        return rdDistGeom.GetMoleculeBoundsMatrix(mol, doTriangleSmoothing=False)[:count, :count]


def test_haptic_sites_group_donors_into_sigma_sites_and_pi_faces():
    """A multiple or aromatic bond, a chain of open valences and a charged end spread a face; with ``pairs`` a bonded
    pair is one site whatever its order, without it only an isolated diatomic is; cyclo-As6 stays six sites."""
    for name, mol, donors, pairs, sites in (
        ("hydrazine", Chem.MolFromSmiles("NN"), [0, 1], True, [(0, 1)]),
        ("hydrazine unpaired", Chem.MolFromSmiles("NN"), [0, 1], False, [(0,), (1,)]),
        ("peroxide unpaired", Chem.MolFromSmiles("[O-][O-]"), [0, 1], False, [(0, 1)]),
        ("ethylene unpaired", Chem.MolFromSmiles("C=C"), [0, 1], False, [(0, 1)]),
        ("benzene", Chem.MolFromSmiles("c1ccccc1"), range(6), True, [(0, 1, 2, 3, 4, 5)]),
        ("allyl anion", Chem.MolFromSmiles("[CH2-]C=C"), range(3), True, [(0, 1, 2)]),
        ("single-bond Cp", _unsanitized("[CH]1[CH][CH][CH][CH]1"), range(5), True, [(0, 1, 2, 3, 4)]),
        ("cyclo-As6", _unsanitized("[AsH]1[AsH][AsH][AsH][AsH][AsH]1"), range(6), True, [(i,) for i in range(6)]),
    ):
        assert graph.haptic_sites(mol, donors, pairs=pairs) == sites, name


def test_torsion_path_orders_an_open_face_whose_centre_rotates():
    """A cyclopentadiene's ring holds its central bond; a cyclooctadiene's eight-ring does not."""
    paths = {  # (SMILES, face) -> its torsion path
        ("C=CC=C", (0, 1, 2, 3)): [0, 1, 2, 3],
        ("C=CC=C", (3, 2, 1, 0)): [0, 1, 2, 3],
        ("C1=CCCCCC=C1", (0, 1, 7, 6)): [1, 0, 7, 6],
        ("C1=CC=CC1", (0, 1, 2, 3)): None,
        ("C1=CC=C1", (0, 1, 2, 3)): None,
        ("C=CC=C", (0, 1, 2)): None,
    }
    assert {(smiles, face): graph.torsion_path(Chem.MolFromSmiles(smiles), face) for smiles, face in paths} == paths


def test_hydrogen_legs_list_a_bridging_hydrogen_ligand_leg_first():
    """A nonmetal comes before a metal, then the nearer atom, then the lower rank, so a true tie reads alike."""
    diborane = "[BH2]1[H][BH2][H]1"
    nearer_b2 = placed(_unsanitized(diborane), [(0, 0, 0), (1.5, 0.8, 0), (2, 0, 0), (1, -0.8, 0)])
    for name, mol, legs in (
        ("B-H-Pt", _unsanitized("C[BH2][H][Pt]"), {2: [1, 3]}),
        ("diborane", _unsanitized(diborane), {1: [0, 2], 3: [0, 2]}),
        ("diborane B2 nearer", nearer_b2, {1: [2, 0], 3: [0, 2]}),
        ("ethane", Chem.MolFromSmiles("CC"), {}),
    ):
        assert graph.hydrogen_legs(mol, graph.COORDINATION_METALS) == legs, name


def test_ligand_graph_cuts_metal_and_h_bonds_in_atom_order():
    """Every metal by default, else those given; a zero-order H...O contact goes, a B-H...B three-centre bond stays."""
    ammines = parse_smiles("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N")
    bridged = parse_smiles("[NH3]->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH2-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[NH3]")
    for name, mol, metals, smiles in (
        ("Pd ammines", ammines, None, "N.N.[Cl-].[Cl-].[Pd+2]"),
        ("no metal cut", ammines, (), "[NH3]->[Pd+2](<-[NH3])(<-[Cl-])<-[Cl-]"),
        ("bridge cut at Pd", bridged, [1], "N.[Cl-].[Cl-].[NH2-]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-].[Pd+2]"),
        ("water dimer", _with_zero_bond(Chem.AddHs(Chem.MolFromSmiles("O.O")), 2, 1), (), "[H]O[H].[H]O[H]"),
        ("diborane", _with_zero_bond(_unsanitized("[BH2]1[H][BH2][H]1"), 1, 2), (), "[H]1[BH2][H]~B1"),
    ):
        assert Chem.MolToSmiles(graph.ligand_graph(mol, metals)) == smiles, name


def test_no_aromatic_axis_where_no_in_plane_direction_is_left():
    """Three ligand neighbours fill a ring carbon's plane; a pyridine N or a cyclopentadienide carbon keeps a pair."""
    for name, mol, atom, held in (
        ("pyridine N on Pd", parse_smiles("c1ccncc1->[Pd+2]"), 3, False),
        ("Cp- carbon", Chem.MolFromSmiles("[cH-]1cccc1"), 0, False),
        ("ethylene carbon", Chem.MolFromSmiles("C=C"), 0, False),
        ("benzene carbon", Chem.MolFromSmiles("c1ccccc1"), 0, True),
        ("N-methylpyrrole N", Chem.MolFromSmiles("Cn1cccc1"), 1, True),
    ):
        assert graph.no_aromatic_axis(mol.GetAtomWithIdx(atom)) is held, name


def test_materialise_phantoms_appends_a_face_centroid_dummy():
    """The P-Pt-P donor-stereo ring and every real-atom bound stay; the copy's distance matrix is its own."""
    mol = Chem.AddHs(Chem.MolFromSmiles(PT_DIPHOSPHINE_ETHYLENE))
    count = mol.GetNumAtoms()
    face = tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetDegree() == 3)
    mol = placed(mol, np.arange(3 * count, dtype=float).reshape(-1, 3))
    cached = Chem.GetDistanceMatrix(mol).copy()
    oracle = Chem.Mol(mol)
    Chem.GetSymmSSSR(oracle, includeDativeBonds=True)
    donors = {0, *(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)}

    out = graph.materialise_phantoms(mol, {count: face})

    positions = out.GetConformer().GetPositions()
    assert (out.GetNumAtoms(), out.GetAtomWithIdx(count).GetDegree()) == (count + 1, 0)
    np.testing.assert_allclose(positions[count], positions[list(face)].mean(axis=0))
    assert any(donors <= set(ring) for ring in out.GetRingInfo().AtomRings())
    np.testing.assert_allclose(_real_bounds(out, count), _real_bounds(oracle, count), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(Chem.GetDistanceMatrix(out), Chem.GetDistanceMatrix(out, force=True))
    np.testing.assert_array_equal(Chem.GetDistanceMatrix(mol), cached)
    assert graph.materialise_phantoms(mol, {}) is mol


def test_strip_phantoms_keeps_real_indices_and_bounds():
    mol = Chem.AddHs(Chem.MolFromSmiles(PT_DIPHOSPHINE_ETHYLENE))
    count = mol.GetNumAtoms()
    oracle = Chem.Mol(mol)
    Chem.GetSymmSSSR(oracle, includeDativeBonds=True)
    with_dummies = Chem.RWMol(mol)
    for _ in range(2):
        with_dummies.AddAtom(Chem.Atom(6))

    out = graph.strip_phantoms(with_dummies.GetMol(), {count, count + 1})

    assert Chem.MolToSmiles(out, canonical=False) == Chem.MolToSmiles(mol, canonical=False)
    np.testing.assert_allclose(_real_bounds(out, count), _real_bounds(oracle, count), atol=1e-12, rtol=0)
    assert graph.strip_phantoms(mol, set()) is mol


def test_surrogate_metals_keep_only_holdable_donor_hands():
    """A phosphine keeps its hand; an amine left with two carriers, or an E/Z referenced only to its metal, loses it."""
    thione = parse_smiles("CC(O)=[S]->[Zn]")
    bond = thione.GetBondBetweenAtoms(1, 3)
    bond.SetStereoAtoms(0, 4)  # Zn is the sulfur-side E/Z reference before the strip
    bond.SetStereo(Chem.BondStereo.STEREOZ)
    for name, mol, hands in (
        ("Pt phosphine", parse_smiles("C[P@H](->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-])CC"), {1}),
        ("amine bridging Pd and Pt", parse_smiles("C[N@H](->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br"), set()),
        ("Zn thione", thione, set()),
    ):
        engine = graph.surrogate_all_metals(mol)

        surrogates = [engine.GetAtomWithIdx(m) for m in graph.metal_indices(mol)]
        assert {(a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge()) for a in surrogates} == {(6, 0, 0)}, name
        assert {a.GetIdx() for a in engine.GetAtoms() if a.GetChiralTag() in TETRAHEDRAL_TAGS} == hands, name
        assert all(len(b.GetStereoAtoms()) == 2 for b in engine.GetBonds() if b.GetStereo() in EZ_RELATION), name
