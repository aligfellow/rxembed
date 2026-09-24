"""Test metal identification, shape perception and surrogate restoration."""

from __future__ import annotations

import importlib
import logging
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import core, stereo
from rxembed import metal_core as _metal
from rxembed.metal_core import classify_geometry, geometry_for
from rxembed.metal_polyhedron import POLYHEDRA, describe
from rxembed.pipeline.perceive import read_xyz
from rxembed.relax import bonding_ok
from tests.conftest import EXAMPLES_DIR

emb = importlib.import_module("rxembed.embed")

_MN_H2 = str(EXAMPLES_DIR / "mn-h2.xyz")  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MN_H2_RC = [1, 5, 63, 64, 65, 66]  # its reacting core
_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # covalent notation for the reliably embedding chelate fixture
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"  # net 0, Ni(II)


@pytest.mark.parametrize(
    ("smiles", "degree"),
    [("[C-:1](#[O+])->[Pt+2]", 1), ("N#[C:1][Pt]", 1), ("[CH:1](->[Pt])#C", 2)],
)
def test_ligand_degree_ignores_metal_but_counts_all_hydrogens(smiles, degree):
    source = Chem.MolFromSmiles(smiles)
    for mol in (source, Chem.AddHs(source)):
        for work in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
            atom = next(a for a in work.GetAtoms() if a.GetAtomMapNum() == 1)
            assert _metal.ligand_degree(atom) == degree


@pytest.mark.parametrize("face", ["centred-arene", "slipped-arene", "open-allyl"])
def test_haptic_centroid_target_matches_measured_geometry(face):
    if face == "open-allyl":
        mol = Chem.MolFromSmiles("C=CC")
        positions = np.array([[-1.2, 0.0, 0.0], [0.0, 0.7, 0.0], [1.2, 0.0, 0.0]])
    else:
        mol = Chem.MolFromSmiles("c1ccccc1")
        angles = np.arange(6) * np.pi / 3
        positions = 1.4 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    metal = np.array([0.0 if face == "centred-arene" else 1.2, 0.0, 1.5])
    conf = Chem.Conformer(len(positions))
    conf.SetPositions(positions)
    mol.AddConformer(conf)
    radius = _metal._site_radius(mol, tuple(range(len(positions))), positions=positions)
    lengths = np.linalg.norm(positions - metal, axis=1)
    actual = np.linalg.norm(metal - positions.mean(axis=0))
    assert _metal._site_height(radius, lengths) == pytest.approx(actual)


@pytest.mark.parametrize("operation", ["collapse", "materialise", "strip"])
def test_haptic_centroids_rebuild_cached_topology(operation):
    mol = Chem.AddHs(Chem.MolFromSmiles("C=C.C=C"))
    count = mol.GetNumAtoms()
    haptic = {count: (0, 1), count + 1: (2, 3)}
    if operation == "strip":
        mol = _metal.materialise_phantoms(mol, haptic)
    fresh = Chem.Mol(mol)
    cached = Chem.GetDistanceMatrix(mol).copy()

    def apply(candidate):
        if operation == "collapse":
            return _metal._collapse_haptic(candidate, [0, 1, 2, 3])[0]
        if operation == "strip":
            return _metal.strip_phantoms(candidate, set(haptic))
        return _metal.materialise_phantoms(candidate, haptic)

    out, expected = apply(mol), apply(fresh)
    actual = Chem.GetDistanceMatrix(out).copy()
    np.testing.assert_array_equal(actual, Chem.GetDistanceMatrix(out, force=True))
    np.testing.assert_array_equal(_metal._bounds_matrix(out), _metal._bounds_matrix(expected))
    np.testing.assert_array_equal(Chem.GetDistanceMatrix(mol), cached)


@pytest.mark.parametrize("operation", ["materialise", "strip"])
@pytest.mark.parametrize("reverse", [False, True])
def test_haptic_helpers_preserve_native_fused_ring_bounds(operation, reverse):
    mol = Chem.AddHs(Chem.MolFromSmiles("[cH-]1ccc2ccccc21"))
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    face = next(tuple(ring) for ring in Chem.GetSymmSSSR(mol) if len(ring) == 5)
    count = mol.GetNumAtoms()
    positions = np.arange(3 * count, dtype=float).reshape(count, 3)
    conf = Chem.Conformer(count)
    conf.SetPositions(positions)
    mol.AddConformer(conf)
    expected = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetChiralTag()) for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()]
    if operation == "strip":
        # Build the removable helper independently; the addition owner must not supply the oracle.
        builder = Chem.RWMol(mol)
        helper = Chem.Atom(6)
        helper.SetNoImplicit(True)
        helper.SetHybridization(Chem.HybridizationType.SP3)
        assert builder.AddAtom(helper) == count
        mol = builder.GetMol()
        mol.UpdatePropertyCache(strict=False)
    before = mol.ToBinary()

    out = (
        _metal.materialise_phantoms(mol, {count: face})
        if operation == "materialise"
        else _metal.strip_phantoms(mol, {count})
    )

    np.testing.assert_allclose(rdDistGeom.GetMoleculeBoundsMatrix(out)[:count, :count], expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(out.GetConformer().GetPositions()[:count], positions)
    assert [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetChiralTag()) for a in list(out.GetAtoms())[:count]] == atoms
    assert [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in out.GetBonds()] == bonds
    assert mol.ToBinary() == before


def test_haptic_collapse_positions_centroids_in_every_conformer():
    mol = Chem.MolFromSmiles("C=C.C=C")
    for cid in (7, 11):
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(cid)
        conf.SetPositions(np.arange(12, dtype=float).reshape(4, 3) + cid)
        mol.AddConformer(conf, assignId=False)
    out, vertices, haptic = _metal._collapse_haptic(mol, [0, 1, 2, 3])
    assert vertices == list(haptic)
    assert {conf.GetId() for conf in out.GetConformers()} == {7, 11}
    for conf in out.GetConformers():
        original = mol.GetConformer(conf.GetId()).GetPositions()
        np.testing.assert_array_equal(conf.GetPositions()[:4], original)
        for dummy, face in haptic.items():
            np.testing.assert_array_equal(conf.GetPositions()[dummy], original[list(face)].mean(axis=0))


def _ideal_sphere(dirs, r):
    """A bare Mol whose atom 0 is a metal and 1..N its vertices at radius `r` along `dirs`."""
    rw = Chem.RWMol()
    for _ in range(len(dirs) + 1):
        rw.AddAtom(Chem.Atom(6))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for i, d in enumerate(dirs):
        u = np.array(d, float)
        conf.SetAtomPosition(i + 1, Point3D(*(u / np.linalg.norm(u) * r)))
    mol.AddConformer(conf, assignId=True)
    return mol


def test_delocalised_charge_canonicalization_requires_rdkit_resonance_proof():
    ligand = Chem.MolFromSmiles("[c-]1cc[nH]c1")
    rw = Chem.RWMol(Chem.CombineMols(ligand, Chem.MolFromSmiles("[Fe+]")))
    metal = rw.GetNumAtoms() - 1
    for atom in list(rw.GetAtoms())[:metal]:
        rw.AddBond(atom.GetIdx(), metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    before = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

    out = _metal._canonicalise_delocalised_charge(mol)

    assert [atom.GetFormalCharge() for atom in out.GetAtoms()] == before


def _metal_neighbours(mol):
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _metal.COORDINATION_METALS)
    return metal, sorted(n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors())


_METALLAOXIRANE_HF = [
    ("[Cl-]->[Hf+4]1(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[O-][C-]->1(C)C", -1),  # X2 dianion: already ionic
    ("[Cl-]->[Hf+2]1(<-[Cl-])(<-[Cl-])(<-[Cl-])<-O=C->1(C)C", 0),  # L ketone: a genuine neutral donor
    ("Cl[Hf]1(Cl)(Cl)(Cl)OC1(C)C", -1),  # covalent: the reader's own ionic correction, unaffected by the merge
]


@pytest.mark.parametrize(("smiles", "charge"), _METALLAOXIRANE_HF, ids=["dianion", "neutral_ketone", "covalent"])
def test_bonded_donor_pair_is_one_site_whatever_its_lewis_form(smiles, charge):
    """RITCIG's C-O metallaoxirane is one haptic site whatever its drawn Lewis form.

    Five vertices (four Cl plus the O-C pair) whether the pair is an X2 dianion, an L ketone or plain
    covalent SMILES; `_canonical_metal_graph` keeps `pairs=False`, so each form's own O/C charge is
    unaffected by the site merge (the covalent form's own ionic correction still leaves O and C at -1).
    """
    mol = rx.parse_smiles(smiles, remove_hs=False)
    _metal_idx, donors = _metal_neighbours(mol)

    sites = _metal._haptic_sites(mol, donors)

    faces = [site for site in sites if len(site) > 1]
    assert len(sites) == len(donors) - 1
    assert len(faces) == 1
    pair = faces[0]
    assert {mol.GetAtomWithIdx(a).GetSymbol() for a in pair} == {"O", "C"}

    canon = _metal._canonical_metal_graph(mol)
    assert all(canon.GetAtomWithIdx(a).GetFormalCharge() == charge for a in pair)


def _classify(dirs, r):
    return classify_geometry(_ideal_sphere(dirs, r), 0, list(range(1, len(dirs) + 1)))


def _bailar(degrees):
    """The octahedron's own vertices with one C3 face rotated by `degrees` about the body diagonal.

    0° leaves the octahedron, 60° reaches the trigonal prism. Built from the record's own `vertex_dirs` so the
    probe cannot drift away from the shape it is distorting.
    """
    dirs = np.array(POLYHEDRA["octahedral"].vertex_dirs, float)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    face = [i for i, d in enumerate(dirs) if d @ axis > 0]
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)  # Rodrigues
    dirs[face] = dirs[face] @ rot.T
    return dirs


def _tilted_square(tilt_deg):
    """A square plane with an alternating `tilt_deg` out-of-plane bow: the tetrahedral distortion of Ni/Pd(II)."""
    t = np.radians(tilt_deg)
    return [
        (np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t) * (1 if k % 2 == 0 else -1))
        for k, phi in enumerate((0.0, np.pi / 2, np.pi, 3 * np.pi / 2))
    ]


# --- perception: the shape invariant ------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(POLYHEDRA))
def test_all_records_round_trip(name):
    got = _classify(POLYHEDRA[name].vertex_dirs, 2.1)
    assert got == name, f"{describe(name)} re-perceives as {got}"


def test_unbound_metal_has_no_coordination_geometry():
    mol = Chem.MolFromSmiles("[Hg]")
    mol.AddConformer(Chem.Conformer(1))

    assert classify_geometry(mol, 0, []) is None


def test_short_bonded_pyramid_is_not_flatness_excluded():
    dirs = POLYHEDRA["trigonal_pyramidal"].vertex_dirs
    assert _metal._ideal_plane_rms(POLYHEDRA["trigonal_pyramidal"], 1.4) < _metal.COPLANAR_TOL, "fixture premise"
    assert _classify(dirs, 1.4) == "trigonal_pyramidal"


def test_bowed_square_plane_reads_square_planar():
    assert _classify(_tilted_square(8), 2.3) == "square_planar"


def test_bis_chelate_zinc_tetrahedron_reads_tetrahedral():
    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral").select(index=0)
    mol = iso.restore(rx.embed(iso, n=2, seed=7).minimize().mol)
    assert [i.geometry for i in rx.metal(mol)] == ["tetrahedral"]


@pytest.mark.parametrize(
    ("twist", "expected"),
    [(0.0, "octahedral"), (60.0, "trigonal_prismatic")],
    ids=["octahedral", "trigonal-prismatic"],
)
def test_bailar_twist_endpoints(twist, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(_bailar(twist), 2.1), 0, list(range(1, 7)))
    assert got == expected
    assert not [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_hexagonal_plane_is_not_forced_into_a_three_dimensional_cn6_shape():
    directions = [(np.cos(angle), np.sin(angle), 0.0) for angle in np.arange(6) * np.pi / 3]

    assert _classify(directions, 2.1) == "hexagonal_planar"


def test_poor_shape_returns_record_and_warns(caplog):
    squashed = np.array([(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])])
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(squashed, 2.1), 0, list(range(1, 7)))
    assert got is not None, "a poor fit is still the nearest record, reported loudly"
    assert [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_poor_shape_can_be_checked_silently(caplog):
    squashed = np.array([(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])])
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        classify_geometry(_ideal_sphere(squashed, 2.1), 0, list(range(1, 7)), warn=False)
    assert not [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_near_tie_keeps_the_argmin_and_names_the_runner_up(caplog):
    """A CN5 witness almost equidistant between trigonal_bipyramidal and square_pyramidal (residual gap
    ~3e-5, far inside `_FIT_MARGIN`) still returns one name, the argmin, and logs the runner-up rather than
    silently picking either. The acceptance gate is what accepts a requested shape this close to the argmin
    (`shape_reading`, rule B); `classify_geometry` itself always names the one nearest reading.
    """
    sp = np.array(POLYHEDRA["square_pyramidal"].vertex_dirs, float)
    tbp = np.array(POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, float)
    dirs = 0.662 * sp + 0.338 * tbp
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(dirs, 2.1), 0, list(range(1, 6)))
    assert got == "trigonal_bipyramidal"
    assert [r for r in caplog.records if "near-tie" in r.message and "square_pyramidal" in r.message], caplog.text


def test_cn_defaults_are_the_common_shapes():
    assert geometry_for(3) == "trigonal_planar"
    assert geometry_for(3, has_apical=True) == "trigonal_planar"
    assert geometry_for(4) == "square_planar"
    assert geometry_for(4, has_apical=True) == "tetrahedral"
    assert geometry_for(5, has_apical=True) == "trigonal_bipyramidal"


def _sigma_pair_on_pi_face():
    rw = Chem.RWMol()
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    sigma = [rw.AddAtom(Chem.Atom(8)) for _ in range(2)]
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    for carbon, oxygen in zip(face, sigma, strict=True):
        rw.AddBond(carbon, oxygen, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, [*face, *sigma]


def _diatomic_codonors(bond_type):
    """Build two same-element atoms joined by `bond_type`, no implicit Hs."""
    rw = Chem.RWMol()
    atoms = [Chem.Atom(16) for _ in range(2)]
    for atom in atoms:
        atom.SetNoImplicit(True)
    pair = [rw.AddAtom(atom) for atom in atoms]
    rw.AddBond(pair[0], pair[1], bond_type)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, pair


def _ring_donors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, list(mol.GetRingInfo().AtomRings()[0])


@pytest.mark.parametrize(
    ("build", "check"),
    [
        (_sigma_pair_on_pi_face, lambda s, d: s == [tuple(d[:2]), (d[2],), (d[3],)]),
        (lambda: _diatomic_codonors(Chem.BondType.SINGLE), lambda s, d: s == [tuple(d)]),
        (lambda: _diatomic_codonors(Chem.BondType.DOUBLE), lambda s, d: s == [tuple(d)]),
        (lambda: _diatomic_codonors(Chem.BondType.TRIPLE), lambda s, d: s == [tuple(d)]),
        (lambda: (Chem.MolFromSmiles("NN"), [0, 1]), lambda s, d: s == [(0, 1)]),
        (lambda: _ring_donors("C1CCCC1"), lambda s, d: s == [(i,) for i in sorted(d)]),
        (lambda: _ring_donors("[CH-]1C=CC=C1"), lambda s, d: s == [tuple(sorted(d))]),
        (lambda: _ring_donors("C=C1C=CC=C1"), lambda s, d: s == [tuple(sorted(d))]),
        (lambda: _ring_donors("C1=CCCC1"), lambda s, d: sorted(map(len, s)) == [1, 1, 1, 2]),
        (lambda: (Chem.MolFromSmiles("C[S](=O)(=[CH2])[CH2-]"), [1, 3, 4]), lambda s, d: s == [(1,), (3,), (4,)]),
        (lambda: (Chem.MolFromSmiles("[CH2-][S+]=[CH2]"), [0, 1, 2]), lambda s, d: s == [(0, 1, 2)]),
    ],
    ids=[
        "sigma-donors-not-merged-into-pi-face",
        "diatomic-codonors-single-bond",
        "diatomic-codonors-double-bond",
        "diatomic-codonors-triple-bond",
        "bonded-donor-pair-is-one-site-whatever-its-hydrogens-or-bond-order",
        "saturated-sigma-ring-of-three-or-more-stays-separate-sites",
        "kekule-cyclopentadienyl-one-face",
        "fulvene-like-sp2-ring-one-face",
        "cyclopentene-does-not-promote-sp3-into-a-face",
        "hypervalent-sp3-multiple-bond-not-a-pi-face",
        "sp2-thiaallyl-one-pi-face",
    ],
)
def test_haptic_sites_group_donors_into_pi_faces_and_isolated_sigma_sites(build, check):
    mol, donors = build()

    sites = _metal._haptic_sites(mol, donors)

    assert check(sites, donors)


# --- which atoms are metal centres: one predicate behind every gate ---------------------------------------


# CN4 rather than CN3: `COPLANAR_TOL` is absolute, so a trigonal-planar sphere at the ~3.1 A the covalent-sum
# fallback gives an f-block centre pyramidalises past the accept gate (CeCl3 relaxes to 0.383 A out-of-plane
# against the 0.25 tol, and `minimize` says so). That is the tolerance's known scaling, not this predicate's.
def _crushed_pair(z, d=0.5):
    """Element `z` single-bonded to a Cl at `d` A, with a conformer: a separation no radius rule can accept.

    0.5 A is well under 0.7 x 1.59 A, and 1.59 A (F-Cl) is the smallest covalent-radius sum this fixture can
    make, so whether `bonding_ok` passes the pair is its metal exemption and nothing else.
    """
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(int(z)))
    rw.AddAtom(Chem.Atom(17))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(d, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    return mol


def test_embed_and_clash_share_metal_predicate():
    zs = range(3, 113)  # Li upwards: paired with H the fixture has one heavy atom, so there is no pair to judge
    exempt = {z for z in zs if bonding_ok(_crushed_pair(z), 0)}
    surrogated = {z for z in zs if _metal.metal_indices(_crushed_pair(z))}
    assert exempt == surrogated, (
        f"clash-gate-only {sorted(exempt - surrogated)}, surrogate-only {sorted(surrogated - exempt)}"
    )


def test_f_block_requires_isomer():
    block, smiles = "lanthanide", "[Ce](Cl)(Cl)(Cl)Cl"  # actinides take the same predicate and branch
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert _metal.metal_indices(mol) == [0], f"the {block} centre is not read as a metal at all"

    with pytest.raises(ValueError, match="are metal centres"):
        core.embed(mol, n=1, seed=1)

    iso = core.enumerate_isomers(mol)[0]
    confs = core.embed(iso, n=2, seed=1).minimize()
    assert len(confs) >= 1, f"the {block} is refused at both doors, so it is supported by neither"
    assert confs.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == mol.GetAtomWithIdx(0).GetAtomicNum()
    assert all(bonding_ok(confs.mol, cid) for cid in confs.ids), "the relax tore the sphere"


# --- the surrogate round-trip: oxidation state ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "smiles"),
    [
        ("[PdCl4]2-", "[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"),
        # a genuinely neutral metal: a phantom charge on it would cancel against the ligands in any net total
        ("Fe(CO)5", "[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"),
    ],
    ids=["palladate", "iron-carbonyl"],
)
def test_surrogate_preserves_metal_charge(name, smiles):
    m0 = Chem.MolFromSmiles(smiles)
    metal_idx = _metal.metal_index(m0)
    q0 = m0.GetAtomWithIdx(metal_idx).GetFormalCharge()

    surrogate, m, _donors, real_z, real_q = _metal.surrogate_metal(m0)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == _metal.SURROGATE
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == 0, f"{name}: the DG surrogate must be neutral"
    assert real_q == q0, f"{name}: the oxidation state was thrown away, not captured"

    _metal.restore_metal(surrogate, m, real_z, real_q)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == real_z
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == q0


def _metal_charges(mol):
    """`{atom index: formal charge}` for every metal: a neutral metal must come back 0, not gain a phantom."""
    return {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() in _metal.COORDINATION_METALS}


def _assert_charge_roundtrip(name, mol_charge, ens):
    """After minimize: the mol charge and the number sent to xtb both equal the input's net charge."""
    ens = (ens.candidates[0] if hasattr(ens, "candidates") else ens).minimize()
    assert Chem.GetFormalCharge(ens.mol) == mol_charge, f"{name}: the metal's oxidation state was lost"
    # the number that reaches xtb; pinned explicitly so a _calc_charge refactor cannot silently re-break it
    assert ens._calc_charge(None) == mol_charge, f"{name}: _calc_charge sends {ens._calc_charge(None):+d}"
    return ens


@pytest.mark.parametrize("route", ["isomer", "isomer-constrain"])
def test_public_embed_restores_oxidation_state(route):
    want = _metal_charges(Chem.MolFromSmiles(_NI_N))
    spec = rx.metal(_NI_N)[0]
    kw = {} if route == "isomer" else {"constrain": {(0, 1): (1.4, 1.6)}}
    ens = _assert_charge_roundtrip(route, 0, rx.embed(spec, n=2, seed=1, **kw))
    assert _metal_charges(ens.mol) == want, f"{route}: the metal came back at the wrong oxidation state"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_isomer_restore_restores_all_metal_states():
    source = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    iso = rx.metal(source, "octahedral", center="Mn", fix=_MN_H2_RC)[0]
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == _metal.SURROGATE  # still the neutral carbon
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == 0
    iso.restore()
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == iso.real_q
    for mi, _rz, rq in iso.spectator_metals:  # the spectator ferrocene Fe
        assert iso.mol.GetAtomWithIdx(mi).GetFormalCharge() == rq


# --- the surrogate round-trip: connectivity -------------------------------------------------------------


def test_connect_disconnect_metal_are_inverses():
    mol = Chem.MolFromSmiles(_EN_PDBRCL)  # the en chelate closes its ring through the Pd
    stripped, m, donors, real_z, real_q = _metal.surrogate_metal(mol)
    assert len(Chem.GetMolFrags(stripped)) > 1, "the surrogate must leave the metal a separate fragment"

    _metal.restore_metal(stripped, m, real_z, real_q)
    connected = _metal.connect_metal(stripped, [(d, m) for d in donors])
    assert len(Chem.GetMolFrags(connected)) == 1
    for d in donors:
        b = connected.GetBondBetweenAtoms(int(d), int(m))
        assert b is not None, f"donor {d} has no bond to the metal"
        assert b.GetBondType() == Chem.BondType.DATIVE, f"donor {d}'s bond to the metal is not DATIVE"
        assert b.GetBeginAtomIdx() == int(d), f"donor {d}'s dative bond runs the wrong way"
    assert connected.GetRingInfo().NumRings() >= 1, "the chelate ring closed through the metal was not re-perceived"
    assert connected.GetSubstructMatches(Chem.MolFromSmarts("[R]"))  # a ring-aware query must not raise

    assert len(Chem.GetMolFrags(_metal.disconnect_metal(connected))) > 1, "disconnect is not connect's inverse"


def test_metal_bond_edits_invalidate_cached_paths():
    mol = Chem.MolFromSmiles("[Cu+].NCCO")
    mol.SetProp("source", "retained")
    before = Chem.GetDistanceMatrix(mol).copy()
    connected = _metal.connect_metal(mol, [(1, 0)])

    assert connected.GetProp("source") == "retained"
    assert np.array_equal(Chem.GetDistanceMatrix(mol), before), "connecting must not mutate the input"
    distances = Chem.GetDistanceMatrix(connected)
    assert distances[0, 1] == 1
    assert distances[0, 4] == 4
    assert np.array_equal(distances, Chem.GetDistanceMatrix(connected, force=True))
    disconnected = _metal.disconnect_metal(connected)
    assert np.array_equal(Chem.GetDistanceMatrix(disconnected), before)


def test_core_embed_returns_connected_copy():
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1)
    assert ens._stage != "minimized", "the fixture must be a bare embed, or this measures nothing"

    out = ens.mol
    assert len(Chem.GetMolFrags(out)) == 1, "bare .mol is not connected"
    assert out.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z
    assert "Pd" in Chem.MolToSmiles(out), "the connected graph must round-trip to a real complex SMILES"

    assert ens._mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == _metal.SURROGATE, "the finalize mutated `_mol`"
    assert len(Chem.GetMolFrags(ens._mol)) > 1, "the working mol was re-bonded"
    for cid in ens.ids:
        assert np.allclose(ens._mol.GetConformer(cid).GetPositions(), out.GetConformer(cid).GetPositions())


# --- metal-bound donor stereochemistry ------------------------------------------------------------------

_CARBANION_NI = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
_CARBANION_C = 23  # the metal-bound sp3 carbanion donor: a stereocentre only WHILE bound


def _chirality_volume(mol, cid, centre):
    conf = mol.GetConformer(cid)
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()][:3]
    p = np.array([list(conf.GetAtomPosition(i)) for i in [centre, *nbrs]])
    return float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))


def test_metal_bound_carbanion_embeds_both_hands():
    en = rx.metal(_CARBANION_NI, "square_planar").filter(label="cis")
    assert {i.stereo_label for i in en} == {"C23:R", "C23:S"}  # metal-priority CIP labels
    hands = []
    for iso in en:
        e = rx.embed(iso, n=1).minimize()
        assert e.n >= 1
        assert e.mol.GetNumAtoms() == 65  # embedding does not append a transient atom
        assert not any(a.GetIsotope() == 2 for a in e.mol.GetAtoms())
        assert any(a.GetSymbol() == "Ni" for a in e.mol.GetAtoms())  # the surrogate is switched back
        signs = {np.sign(_chirality_volume(e.mol, i, _CARBANION_C)) for i in e.ids}
        assert len(signs) == 1  # every conformer of this isomer has the same donor hand
        hands.append(signs.pop())
    assert len(set(hands)) == 2  # and the two isomers are opposite


# --- the donor's hand across the strip: a tag is a parity over the DONOR's bond order, not a symbol to copy -

# The M-L bond written FIRST at the P is the odd slot the surrogate strip mirrors; written LAST is the
# even control. `<-` is the dative arrow the README's complexes use; a bare bond is the covalent alternative.
_CHIRAL_P = {
    "dative_first": "Cl[Pt](Cl)(Cl)<-[P@](C)(CC)c1ccccc1",
    "dative_last": "[P@](C)(CC)(c1ccccc1)->[Pt](Cl)(Cl)Cl",
    "covalent_first": "Cl[Pt](Cl)(Cl)[P@](C)(CC)c1ccccc1",
}


_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _hand(mol, centre, order, cid=-1):
    """The tag naming this conformer's hand at `centre`, read in the fixed bond `order` given.

    RDKit's convention, pinned against RDKit itself by the first test below: negative volume is CW.
    """
    p = mol.GetConformer(cid).GetPositions()
    v = float(np.dot(np.cross(p[order[0]] - p[centre], p[order[1]] - p[centre]), p[order[2]] - p[centre]))
    return _TETRAHEDRAL[0] if v < 0 else _TETRAHEDRAL[1]


def _phosphorus(mol):
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P")


def test_geometric_hand_convention_is_rdkits_own():
    for smiles in ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "F[C@](Cl)(Br)I", "O[C@H]1CC[C@@H](N)CC1"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
        Chem.AssignStereochemistryFrom3D(mol)
        tagged = [a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() in _TETRAHEDRAL]
        assert tagged, f"{smiles} carries no tag, so this cell asserts nothing"
        for idx in tagged:
            cw = mol.GetAtomWithIdx(idx).GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
            assert (_metal.donor_chirality_sign(mol, -1, idx) < 0) is cw


@pytest.mark.parametrize("case", sorted(_CHIRAL_P))
def test_stripped_donor_hand_matches_conformer(case):
    mol = Chem.AddHs(Chem.MolFromSmiles(_CHIRAL_P[case]))
    donor = _phosphorus(mol)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D) == 0  # ETKDG builds the declared hand
    stripped, _m, _donors, _z, _q = _metal.surrogate_metal(mol)
    order = [b.GetOtherAtomIdx(donor) for b in stripped.GetAtomWithIdx(donor).GetBonds()]
    assert stripped.GetAtomWithIdx(donor).GetChiralTag() in _TETRAHEDRAL, "the donor lost its tag"
    assert stripped.GetAtomWithIdx(donor).GetChiralTag() == _hand(stripped, donor, order)


# the odd slot and the even control; `covalent_first` is the odd slot again, and is kept at the strip above
@pytest.mark.parametrize("case", ["dative_first", "dative_last"])
@pytest.mark.parametrize("tag", ["[P@]", "[P@@]"])
def test_chiral_phosphorus_hand_roundtrips(case, tag):
    mol = Chem.AddHs(Chem.MolFromSmiles(_CHIRAL_P[case].replace("[P@]", tag)))
    donor = _phosphorus(mol)
    order = [b.GetOtherAtomIdx(donor) for b in mol.GetAtomWithIdx(donor).GetBonds()]  # the INPUT's own basis
    declared = mol.GetAtomWithIdx(donor).GetChiralTag()
    confs = rx.embed(rx.enumerate_isomers(mol)[0], n=2, seed=0xF00D).minimize()
    assert len(confs)
    for cid in confs.ids:
        assert _hand(confs.mol, donor, order, int(cid)) == declared, f"{case}/{tag}: came back as the mirror"


@pytest.mark.parametrize("tag", ["[S@]", "[S@@]"])
def test_external_dative_donor_stereo_survives_strip(tag):
    mol = Chem.AddHs(Chem.MolFromSmiles(f"Cl[Pd](Cl)(Cl)<-{tag}(=O)(C)CC"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    # sanitize=False because a strict sanitize rejects the Pd complex; the reader writes the tag either way
    back = Chem.MolFromMolBlock(Chem.MolToV3KMolBlock(mol), sanitize=False, removeHs=False)
    back.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(back, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    donor = next(a.GetIdx() for a in back.GetAtoms() if a.GetSymbol() == "S")
    bond = next(
        b
        for b in back.GetAtomWithIdx(donor).GetBonds()
        if back.GetAtomWithIdx(b.GetOtherAtomIdx(donor)).GetSymbol() == "Pd"
    )
    assert bond.GetBondType() == Chem.BondType.DATIVE, "the round trip did not keep the coordination bond"
    assert bond.GetBeginAtomIdx() == donor, "the dative bond does not leave the donor, so this asserts nothing"

    stripped, _m, _donors, _z, _q = _metal.surrogate_metal(back)
    order = [b.GetOtherAtomIdx(donor) for b in stripped.GetAtomWithIdx(donor).GetBonds()]
    assert stripped.GetAtomWithIdx(donor).GetChiralTag() == _hand(stripped, donor, order)


def test_multimetal_surrogate_preserves_hand():
    mol = Chem.AddHs(Chem.MolFromSmiles(_CHIRAL_P["dative_first"]))
    donor = _phosphorus(mol)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    one, _m, _donors, _z, _q = _metal.surrogate_metal(mol)
    every, _metals = _metal.surrogate_all_metals(mol)
    assert every.GetAtomWithIdx(donor).GetChiralTag() == one.GetAtomWithIdx(donor).GetChiralTag()


def test_single_and_all_surrogates_preserve_the_same_donor_hydrogens():
    mol = Chem.MolFromSmiles(_EN_PDBRCL)
    one, _m, donors, _z, _q = _metal.surrogate_metal(mol)
    every, _metals = _metal.surrogate_all_metals(mol)

    assert [one.GetAtomWithIdx(d).GetTotalNumHs() for d in donors] == [
        every.GetAtomWithIdx(d).GetTotalNumHs() for d in donors
    ]


def test_multimetal_surrogate_repairs_stereo_orphaned_by_the_strip():
    mol = rx.parse_smiles("CC(O)=[S]->[Zn]")
    bond = mol.GetBondBetweenAtoms(1, 3)
    bond.SetStereoAtoms(0, 4)  # Zn is the sulfur-side E/Z reference before the coordination bond is stripped
    bond.SetStereo(Chem.BondStereo.STEREOZ)

    stripped, _metals = _metal.surrogate_all_metals(mol)

    assert not _orphaned(stripped)


def test_surrogate_preserves_a_perceived_quinoid_aromatic_form():
    mol = rx.parse_smiles("[S]=C1C=CC=CC1=[P+]->[Ni]<-[S-]")
    ring = set(range(1, 7))
    for index in ring:
        mol.GetAtomWithIdx(index).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= ring:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)

    surrogate, *_rest = _metal.surrogate_metal(mol)

    before = [(bond.GetBondType(), bond.GetIsAromatic()) for bond in mol.GetBonds() if bond.GetBeginAtomIdx() in ring]
    after = [
        (bond.GetBondType(), bond.GetIsAromatic()) for bond in surrogate.GetBonds() if bond.GetBeginAtomIdx() in ring
    ]
    assert after == before


def test_surrogate_does_not_invent_a_radical_on_aromatic_sulfur():
    mol = rx.parse_smiles("c1sccc1.N->[Ni]")
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    sulfur.SetNoImplicit(True)

    surrogate, *_rest = _metal.surrogate_metal(mol)

    assert surrogate.GetAtomWithIdx(sulfur.GetIdx()).GetNumRadicalElectrons() == 0


def test_surrogate_clears_a_forged_aromatic_donor_point_tag():
    mol = Chem.MolFromSmiles("Cc1cc[cH-](c1)->[Ru+]")
    donor = next(atom for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetDegree() == 3)
    donor.SetHybridization(Chem.HybridizationType.SP3)
    donor.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    stripped, *_ = _metal.surrogate_metal(mol)

    assert stripped.GetAtomWithIdx(donor.GetIdx()).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_metal_referenced_donor_path_is_not_double_corrected():
    for iso in rx.metal(_CARBANION_NI, "square_planar").filter(label="cis"):
        order = [b.GetOtherAtomIdx(_CARBANION_C) for b in iso.mol.GetAtomWithIdx(_CARBANION_C).GetBonds()]
        assert len(order) == _metal._MIN_STEREO_NEIGHBOURS, "the donor is not the stripped degree-3 case"
        tag = iso.mol.GetAtomWithIdx(_CARBANION_C).GetChiralTag()
        assert tag in _TETRAHEDRAL, "the carbanion lost its tag, so this asserts nothing"
        e = rx.embed(iso, n=2, seed=0xF00D).minimize()
        for cid in e.ids:
            assert _hand(e.mol, _CARBANION_C, order, int(cid)) == tag


def test_donor_charge_is_unchanged_by_embedding():
    # The temporary dative bond must not change the charge sent to a downstream calculator.
    for iso in rx.metal(_CARBANION_NI, "square_planar").filter(label="cis"):
        assert rx.embed(iso, n=1).mol.GetAtomWithIdx(_CARBANION_C).GetFormalCharge() == -1


def test_direct_isomer_constructor_protects_a_tagged_amine_from_cleanup(monkeypatch):
    source = Chem.AddHs(rx.parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)"))
    expected = stereo.defined_stereo_label(source, {0})
    iso = rx.Isomer(source, "SPL", [1, 2, 3, 4])
    conformers = core.embed(iso, n=1, seed=2)

    assert iso.stereo_label == ""
    assert iso.mol.GetAtomWithIdx(4).GetChiralTag() in _TETRAHEDRAL
    assert emb._stereo_donor_bonds(iso.mol, iso) == [(4, 0)]

    def reflect(mol, _cons, *, conf_ids, max_iters, _statuses=None, **_kwargs):
        for cid in conf_ids:
            if max_iters:
                positions = mol.GetConformer(cid).GetPositions()
                positions[:, 0] *= -1.0
                mol.GetConformer(cid).SetPositions(positions)
            if _statuses is not None:
                _statuses[cid] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    conformers._relax_constrained(emb.BASE_STIFFNESS, max_iters=10)

    assert conformers.unrelaxed == conformers.ids
    assert {
        stereo.stereo_from_3d(Chem.Mol(conformers.mol, False, int(cid)), exclude={iso.metal}) for cid in conformers.ids
    } == {expected}


def test_pipeline_tracks_a_tagged_amine_from_the_direct_constructor():
    source = Chem.AddHs(rx.parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)"))
    iso = rx.Isomer(source, "SPL", [1, 2, 3, 4])

    ensemble = rx.embed(iso, n=1, seed=2)

    assert set(ensemble._donor_hand) == {4}


def test_pipeline_tracks_a_tagged_phosphorus_donor():
    iso = rx.metal("F[P@](Cl)(Br)->[Pd](Cl)(Cl)Cl", "square_planar")[0]
    donor = next(atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "P")

    ensemble = rx.embed(iso, n=1, seed=2)
    realised = stereo.stereo_from_3d(ensemble.mol, exclude=_metal.metal_indices(ensemble.mol))

    assert set(ensemble._donor_hand) == {donor}
    assert stereo.point_stereo(realised) == stereo.point_stereo(iso.stereo_label)


@pytest.mark.parametrize("linker", ["[N@H](C)CC[N@@H]->2C", "[P@](C)(CC)CC[P@@](C)(CC)->2"])
def test_haptic_helpers_preserve_the_transient_donor_stereo_ring(linker, monkeypatch, tmp_path):
    isomers = rx.metal(f"[Pt+2]12(<-[Cl-])(<-[CH2]=[CH2]->1)<-{linker}", "SPL")
    captured = []
    original = _metal.materialise_phantoms

    def observe(mol, haptic):
        out = original(mol, haptic)
        bonds = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE
        ]
        if haptic and bonds:
            assert len(bonds) == 2
            assert any({a for pair in bonds for a in pair} <= set(ring) for ring in out.GetRingInfo().AtomRings())
            oracle = Chem.Mol(mol)
            oracle.ClearComputedProps()
            oracle.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(oracle, includeDativeBonds=True)
            builder = Chem.RWMol(mol)
            for index in sorted(haptic):
                assert builder.AddAtom(Chem.Atom(6)) == index
            removed = _metal.strip_phantoms(builder.GetMol(), set(haptic))
            with rdBase.BlockLogs():
                expected = rdDistGeom.GetMoleculeBoundsMatrix(oracle, doTriangleSmoothing=False)
                for candidate in (out, removed):
                    actual = rdDistGeom.GetMoleculeBoundsMatrix(candidate, doTriangleSmoothing=False)
                    np.testing.assert_allclose(actual[: len(expected), : len(expected)], expected, atol=1e-12, rtol=0)
            captured.append(bonds)
        return out

    monkeypatch.setattr(_metal, "materialise_phantoms", observe)
    assert isomers
    for iso in isomers:
        assert len(iso.haptic) == 1
        assert len(next(iter(iso.haptic.values()))) == 2
        assert len(emb._stereo_donor_bonds(iso.mol, iso)) == 2
        ensemble = rx.embed(iso, n=1, seed=2, threads=1)
        assert not ensemble.unrelaxed
        assert ensemble.check()[ensemble.ids[0]].ok()
        realised = stereo.stereo_from_3d(ensemble.mol, exclude=_metal.metal_indices(ensemble.mol))
        assert stereo.point_stereo(realised) == stereo.point_stereo(iso.stereo_label)
        assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
        if find_spec("xyzgraph") is not None:
            path = tmp_path / "donor-stereo.xyz"
            Chem.MolToXYZFile(ensemble.mol, str(path))
            fresh = rx.read_xyz(str(path), charge=Chem.GetFormalCharge(ensemble.mol), bond_orders="xyz2mol")
            assert rx.cxsmiles(fresh) == rx.cxsmiles(iso)
    assert captured


def test_geometry_does_not_make_an_unspecified_monodentate_amine_chiral():
    from rxembed.metal_isomer import from_geometry

    mol = Chem.AddHs(rx.parse_smiles("[Pd+2](<-[Cl-])(<-[Cl-])(<-[Cl-])<-[N@H](C)O"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    donor = 4
    mol.GetAtomWithIdx(donor).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    iso = from_geometry(mol)
    assert iso.stereo_label == ""
    assert iso.mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    embedded = core.embed(iso, n=2, params=rx.EmbedParams(seed=2, prune_rms=-1)).minimize()
    metals = set(_metal.metal_indices(embedded.mol))

    assert embedded.unrelaxed == []
    assert embedded.mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    assert all(
        not stereo.point_stereo(stereo.stereo_from_3d(Chem.Mol(embedded.mol, False, int(cid)), exclude=metals))
        for cid in embedded.ids
    )


@pytest.mark.parametrize("tag", ["@", "@@"])
@pytest.mark.parametrize("second_metal", ["Pt", "Pd"])
def test_two_metal_bridge_uses_its_absolute_label_after_the_surrogate_strip(tag, second_metal):
    mol = Chem.AddHs(rx.parse_smiles(f"C[N{tag}H](->[Pd](Cl)(Cl)Cl)->[{second_metal}](Br)(Br)Br"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    iso = rx.metal(mol, center="all")[0]
    assert iso.mol.GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    embedded = core.embed(iso, n=2, params=rx.EmbedParams(seed=2, prune_rms=-1)).minimize()
    metals = set(_metal.metal_indices(embedded.mol))
    donor = embedded.mol.GetAtomWithIdx(1)

    assert embedded.unrelaxed == []
    assert donor.GetChiralTag() in _TETRAHEDRAL
    assert stereo.defined_stereo_label(embedded.mol, metals) == iso.stereo_label
    assert {stereo.stereo_from_3d(Chem.Mol(embedded.mol, False, int(cid)), exclude=metals) for cid in embedded.ids} == {
        iso.stereo_label
    }


def test_bridge_stereo_forbids_global_metal_hand_reflection(monkeypatch):
    mol = Chem.AddHs(rx.parse_smiles("C[N@H](->[Co](F)(Cl)Br)->[Pt](Br)(Br)Br"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    iso = rx.metal(mol, center="all")[0]
    assert iso.chirality
    assert iso.stereo_label
    assert iso.mol.GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    monkeypatch.setattr(emb, "_reflect", lambda *_args: pytest.fail("reflection inverted hidden bridge stereo"))

    assert core.embed(iso, n=1, params=rx.EmbedParams(seed=0, prune_rms=-1)).ids


# ---------------------------------------------------------------------------------------------------------
# RDKit STATE across the metal-bond surgery (was test_mol_state.py)
#
# RDKit state must stay valid across the metal-bond surgery.
#
# Stripping the M-donor bonds can invalidate RDKit state that was derived while the metal was still bonded, and
# RDKit does not always notice. Two ways it bit us, both found on the tmQM corpus:
#
#   * a double bond left FLAGGED stereo with its two reference atoms dropped; because the metal was one of them.
#     RDKit's own ETKDG then indexes the empty vector and SEGFAULTS (rc 139), which no try/except can catch.
#   * `metal.surrogate_metal` re-imposing a STRICT sanitize on a structure the reader deliberately admitted leniently,
#     rejecting chemistry (an unkekulisable quinoid ring, a BPh4- boron) that perception had already accepted.
#
# These are asserted as INVARIANTS rather than as "structure X does not crash": the invariant is what the next
# surgery site has to honour, and it is what makes the guarantee checkable on any input.
# ---------------------------------------------------------------------------------------------------------


# rxembed's own fixtures. The invariants below must hold on any metal complex, so the gate runs on structures
# this repo ships rather than reaching into a sibling checkout: a unit suite that depends on an absolute path
# outside the project is not portable and is not a gate. The 144-structure corpus SWEEP that originally found
# these defects is a measurement, not a gate, and lives in `benchmark/` where the corpus is in scope.
_CORPUS = sorted(str(p) for p in EXAMPLES_DIR.glob("*.xyz"))
corpus_only = pytest.mark.skipif(not _CORPUS, reason="no structure fixtures found")


def _orphaned(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != 2]


@corpus_only
@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_prepared_mol_has_no_orphaned_stereo_flags():
    violations = []
    for path in _CORPUS:
        charges = {0: 2, 1: 1} if path.endswith("/mn-h2.xyz") else None
        mol = read_xyz(path, 0, metal_charges=charges, bond_orders="xyz2mol")
        if not _metal.metal_indices(mol):
            continue
        prepared, *_ = _metal.surrogate_metal(mol)
        if _orphaned(prepared):
            violations.append(path.rsplit("/", 1)[-1])
    assert violations == [], f"orphaned stereo flags survive surrogate_metal(): {violations}"


@corpus_only
def test_surrogate_accepts_all_readable_metals():
    rejected = []
    for path in _CORPUS:
        try:
            mol = read_xyz(path, 0)
        except Exception:
            continue  # perception itself declined it: not surrogate_metal's business
        if not _metal.metal_indices(mol):
            continue
        try:
            _metal.surrogate_metal(mol)
        except Exception as exc:
            rejected.append((path.rsplit("/", 1)[-1], type(exc).__name__))
    assert rejected == [], f"surrogate_metal() rejects structures the reader accepted: {rejected}"


def test_ligands_reports_denticity_per_metal():
    mol = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    ligs = _metal.ligands(mol)
    assert ligs, "the fixture must have ligands"

    for lig in ligs:
        assert lig.mol.GetNumConformers(), "a ligand must carry its own geometry, or it cannot be re-placed"
        assert lig.mol.GetNumAtoms() == len(lig.atoms), "`atoms` must index the original positionally"
        for donors in lig.donors.values():
            assert donors, "a metal with no donors must not appear as a key"
            assert all(0 <= d < lig.mol.GetNumAtoms() for d in donors), "donors index the LIGAND, not the complex"

    bridging = [lig for lig in ligs if len(lig.donors) > 1]
    assert bridging, "this fixture's backbone bridges both metals; without one the per-metal split is untested"
    assert sorted(len(d) for d in bridging[0].donors.values()) == [3, 5]

    # every donor of every metal is accounted for, exactly once
    seen = sorted(lig.atoms[d] for lig in ligs for ds in lig.donors.values() for d in ds)
    want = sorted(n.GetIdx() for m in _metal.metal_indices(mol) for n in mol.GetAtomWithIdx(m).GetNeighbors())
    assert seen == want, "the ligands must partition the coordination sphere"


def test_ligands_refuses_a_molecule_with_no_metal():
    with pytest.raises(ValueError, match="no metal centre"):
        _metal.ligands(Chem.AddHs(Chem.MolFromSmiles("CCO")))
