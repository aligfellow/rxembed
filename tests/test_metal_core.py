"""Test metal identification, shape perception and surrogate restoration."""

from __future__ import annotations

import importlib
import logging
import pathlib
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

emb = importlib.import_module("rxembed.embed")

_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MN_H2_RC = [1, 5, 63, 64, 65, 66]  # its reacting core
_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # neutral en-PdBrCl, which reliably embeds: the connectivity fixture
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"  # net 0, Ni(II)


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


def test_short_bonded_pyramid_is_not_flatness_excluded():
    dirs = POLYHEDRA["trigonal_pyramidal"].vertex_dirs
    assert _metal._ideal_plane_rms(POLYHEDRA["trigonal_pyramidal"], 1.4) < _metal.COPLANAR_TOL, "fixture premise"
    assert _classify(dirs, 1.4) == "trigonal_pyramidal"


def test_bowed_square_plane_is_not_reclassified():
    assert _classify(_tilted_square(8), 2.3) == "square_planar"


def test_bis_chelate_tetrahedron_is_not_reclassified():
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


def test_poor_shape_returns_record_and_warns(caplog):
    squashed = np.array([(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])])
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(squashed, 2.1), 0, list(range(1, 7)))
    assert got is not None, "a poor fit is still the nearest record, reported loudly"
    assert [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_cn_defaults_are_the_common_shapes():
    assert geometry_for(3) == "trigonal_planar"
    assert geometry_for(4) == "square_planar"


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
    iso = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)[0]
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
    en = rx.metal(_CARBANION_NI, "square_planar")
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
    confs = rx.embed(rx.enumerate_isomers(mol, stereo="free")[0], n=2, seed=0xF00D).minimize()
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


def test_metal_referenced_donor_path_is_not_double_corrected():
    for iso in rx.metal(_CARBANION_NI, "square_planar"):
        order = [b.GetOtherAtomIdx(_CARBANION_C) for b in iso.mol.GetAtomWithIdx(_CARBANION_C).GetBonds()]
        assert len(order) == _metal._MIN_STEREO_NEIGHBOURS, "the donor is not the stripped degree-3 case"
        tag = iso.mol.GetAtomWithIdx(_CARBANION_C).GetChiralTag()
        assert tag in _TETRAHEDRAL, "the carbanion lost its tag, so this asserts nothing"
        e = rx.embed(iso, n=2, seed=0xF00D).minimize()
        for cid in e.ids:
            assert _hand(e.mol, _CARBANION_C, order, int(cid)) == tag


def test_donor_charge_is_unchanged_by_embedding():
    # The temporary dative bond must not change the charge sent to a downstream calculator.
    for iso in rx.metal(_CARBANION_NI, "square_planar"):
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


def test_geometry_label_restores_an_unspecified_amine_tag():
    from rxembed.metal_isomer import from_geometry

    mol = Chem.AddHs(rx.parse_smiles("[Pd+2](<-[Cl-])(<-[Cl-])(<-[Cl-])<-[N@H](C)O"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    donor = 4
    mol.GetAtomWithIdx(donor).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    iso = from_geometry(mol)
    assert iso.stereo_label == "N4:R"
    assert iso.mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    embedded = core.embed(iso, n=2, seed=2, prune_rms=-1).minimize()
    metals = set(_metal.metal_indices(embedded.mol))

    assert embedded.unrelaxed == []
    assert embedded.mol.GetAtomWithIdx(donor).GetChiralTag() in _TETRAHEDRAL
    assert stereo.defined_stereo_label(embedded.mol, metals) == iso.stereo_label
    assert {stereo.stereo_from_3d(Chem.Mol(embedded.mol, False, int(cid)), exclude=metals) for cid in embedded.ids} == {
        iso.stereo_label
    }


@pytest.mark.parametrize("tag", ["@", "@@"])
@pytest.mark.parametrize("second_metal", ["Pt", "Pd"])
def test_two_metal_bridge_uses_its_absolute_label_after_the_surrogate_strip(tag, second_metal):
    mol = Chem.AddHs(rx.parse_smiles(f"C[N{tag}H](->[Pd](Cl)(Cl)Cl)->[{second_metal}](Br)(Br)Br"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    iso = rx.metal(mol, center="all")[0]
    assert iso.mol.GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    embedded = core.embed(iso, n=2, seed=2, prune_rms=-1).minimize()
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

    assert core.embed(iso, n=1, seed=0, prune_rms=-1).ids


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
_CORPUS = sorted(str(p) for p in pathlib.Path("examples/structures").glob("*.xyz"))
corpus_only = pytest.mark.skipif(not _CORPUS, reason="no structure fixtures found")


def _orphaned(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != 2]


@corpus_only
@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_prepared_mol_has_no_orphaned_stereo_flags():
    violations = []
    for path in _CORPUS:
        mol = read_xyz(path, 0)
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
    mol = read_xyz(_MN_H2)
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
