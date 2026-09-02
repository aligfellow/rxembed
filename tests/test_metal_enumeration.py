"""Test metal-isomer enumeration and embedding integration."""

from __future__ import annotations

import itertools
from collections import Counter
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_enumeration as K  # noqa: N812
from rxembed import metal_isomer as I  # noqa: N812
from rxembed import metal_slots as slots
from rxembed import metal_stereo as MS  # noqa: N812
from rxembed.embed import embed as core_embed
from rxembed.metal_core import HapticSite
from rxembed.pipeline import geom_check as geom
from tests.metal_fixtures import ferrocene

_MA2B2 = "CCCN[Pd](Cl)(Cl)NCCC"  # square-planar MA2B2 -> the cis / trans pair
_ASYMMETRIC_NN_NI = "O=C1[O-]->[Ni+2]2(<-[CH-](c3ccccc3)N1c1ccccc1)<-[N](O)=C(c1ccccn1)c1cccc[n]->21"
S, D, DAT = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.DATIVE


def _assert_clean(ens):
    """Every conformer of a minimized ensemble passes the metal-aware geometry gate."""
    ens = ens.minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert rep.ok(), rep.summary()


def _angle(pos, i, j, k):
    a, b = pos[i] - pos[j], pos[k] - pos[j]
    return float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))


def cp_ticl3():
    """CpTiCl3: a Cp- ring + three chlorides on Ti(IV): a C3v piano stool (one face + 3 sigma donors, CN4)."""
    rw = Chem.RWMol()
    ti = rw.AddAtom(Chem.Atom(22))
    rw.GetAtomWithIdx(ti).SetFormalCharge(4)
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    for k, b in enumerate([S, D, S, D, S]):
        rw.AddBond(cs[k], cs[(k + 1) % 5], b)
    rw.GetAtomWithIdx(cs[0]).SetFormalCharge(-1)
    for c in cs:
        rw.AddBond(ti, c, DAT)
    cls = [rw.AddAtom(Chem.Atom(17)) for _ in range(3)]
    for cl in cls:
        rw.AddBond(ti, cl, S)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(ti, Point3D(0, 0, 0))
    for k, c in enumerate(cs):
        a = 2 * np.pi * k / 5
        conf.SetAtomPosition(c, Point3D(1.2 * np.cos(a), 1.2 * np.sin(a), 1.9))  # ring up
    for k, cl in enumerate(cls):
        a = 2 * np.pi * k / 3
        conf.SetAtomPosition(cl, Point3D(1.9 * np.cos(a), 1.9 * np.sin(a), -1.3))  # tripod down
    m.AddConformer(conf)
    return m


def tetrahedral_four_distinct():
    """Zn(II) with four distinct monodentate dative donors (N, O, S, P) -> tetrahedral, two enantiomers.

    All six vertex pairs share 109.47°, so the pairwise element/angle signature is identical for both hands;
    only the centre's handedness separates them, which is what makes this the genuine un-enumerated limitation.
    """
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(30))
    rw.GetAtomWithIdx(me).SetFormalCharge(2)
    donors = []
    for z, n_h in [(7, 3), (8, 2), (16, 2), (15, 3)]:  # NH3, OH2, SH2, PH3 (dative M<-donor)
        d = rw.AddAtom(Chem.Atom(z))
        for _ in range(n_h):
            rw.AddBond(d, rw.AddAtom(Chem.Atom(1)), S)
        rw.AddBond(me, d, DAT)
        donors.append(d)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for d, v in zip(donors, [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], strict=True):
        conf.SetAtomPosition(d, Point3D(*(np.array(v, float) / np.sqrt(3) * 2.1)))
    for a in m.GetAtoms():  # splay each donor's H's off it deterministically
        if a.GetAtomicNum() == 1:
            base = np.array(conf.GetAtomPosition(a.GetNeighbors()[0].GetIdx()))
            conf.SetAtomPosition(a.GetIdx(), Point3D(*(base * 1.4 + np.random.RandomState(a.GetIdx()).randn(3) * 0.3)))
    m.AddConformer(conf)
    return m


# --- enumeration: the textbook isomers ------------------------------------------------------------------


def test_single_donor_defaults_to_monocoordinate_without_a_vacancy():
    isomers = rx.metal("N->[Pd+2]")

    assert len(isomers) == 1
    assert isomers[0].geometry == "monocoordinate"
    assert isomers[0].vertices == [isomers[0].donors[0]]
    assert "MCO" in str(isomers[0])
    assert rx.metal(rx.cxsmiles(isomers[0]))[0].geometry == "monocoordinate"


@pytest.mark.parametrize(
    ("smiles", "geometry", "labels"),
    [
        (_MA2B2, "square_planar", {"cis", "trans"}),  # MA2B2
        ("[NH3][Co]([NH3])([NH3])(Cl)(Cl)Cl", "octahedral", {"mer", "fac"}),  # MA3B3
    ],
    ids=["square-planar-ma2b2", "octahedral-ma3b3"],
)
def test_metal_isomers_embed_clean(smiles, geometry, labels):
    cands = rx.embed(smiles, metal=geometry, n=4)
    assert {e.tag["label"] for e in cands} == labels
    for e in cands:
        _assert_clean(e)


def test_bis_en_octahedral_has_three_stereoisomers():
    isos = rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")
    embeddable = [iso for iso in isos if rx.embed(iso, n=2).minimize().n]
    assert {i.chirality for i in embeddable} == {"", "delta", "lambda"}, [i.chirality for i in embeddable]


@pytest.mark.parametrize(
    ("geometry", "smiles", "per_hand"),
    [
        ("seesaw", "[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])<-N", 6),
        ("trigonal_bipyramidal", "[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O", 10),
        ("square_pyramidal", "[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O", 15),
        ("octahedral", "[O+]#[C-]->[Co+3](<-[F-])(<-[Cl-])(<-[Br-])(<-N)<-O", 15),
        ("trigonal_prismatic", "O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", 60),
    ],
    ids=["seesaw", "TBP", "square-pyramidal", "octahedral", "trigonal-prismatic"],
)
def test_chiral_polyhedra_enumerate_both_hands(geometry, smiles, per_hand):
    isos = rx.metal(smiles, geometry, stereo="free")
    assert Counter(i.chirality for i in isos) == {"delta": per_hand, "lambda": per_hand}
    assert {i.label for i in isos} == {""}  # all donors differ, so cis/trans has no meaning


def test_distinct_eta2_faces_define_both_octahedral_hands():
    smiles = (
        r"C/[CH]1=[CH](/F)->[Co+3]2(<-[CH](Cl)=[CH](Br)->2)(<-[NH3])"
        r"(<-[Cl-])(<-[Br-])(<-[F-])<-1"
    )
    isomers = rx.metal(smiles, "OCT", stereo="free")
    assert Counter(iso.chirality for iso in isomers) == {"delta": 15, "lambda": 15}
    assert len({I.arrangement(iso) for iso in isomers}) == len(isomers) == 30


def test_haptic_face_classes_follow_set_orbits_not_atom_orbits():
    rw = Chem.RWMol()
    prisms = []
    for _ in range(2):
        ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(10)]
        for offset in (0, 5):
            for i in range(5):
                rw.AddBond(ring[offset + i], ring[offset + (i + 1) % 5], S)
        for i in range(5):
            rw.AddBond(ring[i], ring[i + 5], S)
        prisms.append(ring)
    metal = rw.AddAtom(Chem.Atom(27))
    rw.GetAtomWithIdx(metal).SetFormalCharge(3)
    for donor in (prisms[0][0], prisms[0][1], prisms[1][3], prisms[1][8]):
        rw.AddBond(donor, metal, DAT)
    for z in (7, 9, 17, 35):
        donor = rw.AddAtom(Chem.Atom(z))
        rw.AddBond(donor, metal, DAT)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    isomers = rx.enumerate_isomers(mol, "OCT", stereo="free")
    assert Counter(iso.chirality for iso in isomers) == {"delta": 15, "lambda": 15}
    assert len({I.arrangement(iso) for iso in isomers}) == 30
    assert len({rx.cxsmiles(iso) for iso in isomers}) == 30


def test_bridging_donor_role_distinguishes_otherwise_identical_sites():
    rw = Chem.RWMol()
    co, pt = rw.AddAtom(Chem.Atom(27)), rw.AddAtom(Chem.Atom(78))
    donors = [rw.AddAtom(Chem.Atom(z)) for z in (7, 7, 9, 17, 35, 8, 53, 16, 15)]
    for donor in donors[:2]:
        rw.GetAtomWithIdx(donor).SetNumExplicitHs(3)
        rw.GetAtomWithIdx(donor).SetNoImplicit(True)
    for donor in donors[:6]:
        rw.AddBond(donor, co, DAT)
    for donor in (donors[0], *donors[6:]):
        rw.AddBond(donor, pt, DAT)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = [
        (0, 0, 0),
        (4, 0, 0),
        (2, 0, 0),
        (-2, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, 0, 2),
        (0, 0, -2),
        (6, 0, 0),
        (4, 2, 0),
        (4, -2, 0),
    ]
    for atom, position in enumerate(positions):
        conf.SetAtomPosition(atom, Point3D(*position))
    mol.AddConformer(conf)

    isomers = rx.enumerate_isomers(mol, "OCT", center="Co", stereo="free")
    assert Counter(iso.chirality for iso in isomers) == {"delta": 15, "lambda": 15}
    strings = {rx.cxsmiles(iso) for iso in isomers}
    assert len(strings) == 30
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    reversed_isomers = rx.enumerate_isomers(reversed_mol, "OCT", center="Co", stereo="free")
    assert strings == {rx.cxsmiles(iso) for iso in reversed_isomers}


def test_defined_and_enumerated_ligand_stereo_share_one_label():
    isomers = rx.metal("[Pd](Cl)(Cl)(Cl)([N@H](C)C(O)C)", "SPL")
    assert {iso.stereo_label for iso in isomers} == {"N4:R,C6:R", "N4:R,C6:S"}


def test_asymmetric_nn_complex_keeps_both_substrate_orientations_per_stereoisomer():
    mol = Chem.AddHs(Chem.MolFromSmiles(_ASYMMETRIC_NN_NI))
    by_stereo = {}
    for iso in rx.enumerate_isomers(mol, "square_planar", stereo="racemic"):
        by_stereo.setdefault(iso.stereo_label, set()).add(tuple(iso.vertices))
    assert len(by_stereo) == 2
    assert {len(arrangements) for arrangements in by_stereo.values()} == {2}


def test_eta2_pair_passes_orientation_screen():
    smi = (
        "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)[c-]1->[Rh+]123(<-[C-]#[O+])"
        "<-[CH]4=[CH]->1CC[CH]->2=[CH]->3CC4"
    )
    assert len(rx.metal(smi)) > 0


# --- vertex-derived permutation pools -------------------------------------------------------------------


def test_derived_single_ordering_is_retained():
    isos = rx.metal("Cl[Mo](Cl)(Cl)(Cl)(Cl)(Cl)Cl")  # homoleptic MoCl7: one ordering, no haptic face
    assert isos[0].geometry == "pentagonal_bipyramidal"


def test_four_distinct_tetrahedral_donors_define_both_hands():
    isos = rx.metal(tetrahedral_four_distinct(), "tetrahedral")
    assert Counter(iso.chirality for iso in isos) == {"delta": 1, "lambda": 1}


def test_trigonal_prismatic_pair_keeps_triangle_and_vertical_edges_distinct():
    assert len(rx.metal("N->[Co+2](<-N)(<-N)(<-N)(<-O)<-O", "TPR")) == 4


def test_high_coordination_homoleptic_model_is_one_arrangement():
    isos = rx.metal("O->[La+3](<-O)(<-O)(<-O)(<-O)(<-O)(<-O)<-O", "DOD")
    assert len(isos) == 1
    assert isos[0].geometry == "dodecahedral"


def test_high_coordination_candidates_defer_their_public_molecule_copy():
    isos = rx.metal("O->[La+3](<-N)(<-P)(<-S)(<-[F-])(<-[Cl-])(<-[Br-])<-[I-]", "DOD")
    assert len(isos) == 10_080
    assert all(iso._mol is None for iso in isos)
    assert len({id(iso._graph) for iso in isos}) == 1
    assert "DOD" in str(isos[2])
    assert "DOD" in rx.cxsmiles(isos[2])
    assert isos[2]._mol is None


# --- the template graft over a coordination sphere -------------------------------------------------------


def _cis_reference(n=2):
    """Embed the MA2B2 pair and return `(cis ensemble, its positions, {symbol: [idx]})`."""
    pair = rx.embed(_MA2B2, metal="square_planar", n=n)
    cis = next(e for e in pair if e.tag["label"] == "cis")
    where: dict = {}
    for a in cis.mol.GetAtoms():
        where.setdefault(a.GetSymbol(), []).append(a.GetIdx())
    return cis, cis.mol.GetConformer(cis.ids[0]).GetPositions(), where


def test_offsphere_template_leaves_arrangement_free():
    ref, pos, where = _cis_reference()
    pd, cl = where["Pd"][0], where["Cl"]
    core = [0, 1, 2]  # the propyl backbone of one ligand: no metal, no donor pair

    out = rx.embed(_MA2B2, metal="square_planar", n=2, template=(ref.mol, {i: i for i in core}))
    assert {e.tag["label"] for e in out} == {"cis", "trans"}
    for ens in out:
        assert sorted(ens.cons.frozen) == core  # the templated atoms, and not the metal sphere
        for cid in ens.ids:
            p = ens.mol.GetConformer(cid).GetPositions()
            for a, b in [(0, 1), (0, 2), (1, 2)]:  # the whole grafted core, not just one distance
                assert np.linalg.norm(p[a] - p[b]) == pytest.approx(np.linalg.norm(pos[a] - pos[b]), abs=1e-6)
            got = _angle(p, cl[0], pd, cl[1])  # the arrangement the label promises SURVIVED the graft
            assert got > 150 if ens.tag["label"] == "trans" else got < 120, (ens.tag["label"], got)


def test_graft_over_the_coordination_sphere_is_refused():
    ref, pos, where = _cis_reference()
    sphere = [where["Pd"][0], *where["Cl"], *where["N"]]
    for core in ([*where["Cl"]], sphere):
        with pytest.raises(ValueError, match="coordination-sphere atoms"):
            rx.embed(_MA2B2, metal="square_planar", n=2, template=(ref.mol, {i: i for i in core}))
    iso = next(iter(rx.metal(_MA2B2, "square_planar")))
    with pytest.raises(ValueError, match="coordination-sphere atoms"):
        rx.embed(iso, n=2, fix={i: tuple(pos[i]) for i in where["Cl"]})


# --- haptic faces: one vertex, a transient centroid -------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_sandwich_uses_two_centroids(door, tmp_path):
    if door == "enumerate":
        isos = rx.metal(ferrocene())
        assert len(isos) == 1  # two identical faces on one metal: a single achiral identity
        iso = isos[0]
        ens = rx.embed(iso, n=1, seed=1)
        arr = I.arrangement(iso)
        assert arr.count("η5") == 2, "the vertex must render as a hapticity tag, from the ring the mol really has"
        assert isos.select(arrangement=arr) is not None
    else:
        ens = rx.embed(ferrocene(), n=1, seed=1)  # the public geometry-retention door, not its internal builder
        iso = I.from_geometry(ferrocene())
        assert iso.cons.haptic == dict(iso.haptic)
        assert len(iso.cons.angles) == 1  # one centroid-M-centroid angle (was 45 across raw ring atoms)

    assert len(iso.haptic) == 2  # one centroid per face
    assert len(iso.vertices) == 2  # TWO coordination sites, not ten raw ring atoms
    assert all(v in iso.haptic for v in iso.vertices)
    assert iso.mol.GetNumAtoms() == 11  # the stored mol is REAL (Fe + 2 Cp)
    assert set(iso.donors) == set(range(1, 11))  # every ring atom IS a donor, not a centroid dummy
    assert len(iso.cons.phantoms) == 2, "the constraints must name the centroid dummies (the scaffolding)"
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms), "a dummy sits inside the real atoms"
    assert all((min(iso.metal, p), max(iso.metal, p)) in iso.cons.pulls for p in iso.cons.phantoms)

    ens.minimize()
    assert ens.mol.GetNumAtoms() == 11  # still real after the full relax
    assert ens.ids
    cid = next(iter(ens.ids))
    assert geom.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    mc = sorted(float(np.linalg.norm(pos[iso.metal] - pos[c])) for c in iso.donors)
    assert mc[0] >= 1.9, "the closest ring atom collapsed onto the metal"
    assert mc[-1] <= 2.3, "the farthest ring atom left the η5 shell"
    for ring in ens.cons.haptic.values():
        d = float(np.linalg.norm(pos[iso.metal] - np.mean([pos[a] for a in ring], axis=0)))
        assert 1.5 < d < 1.85, "Fe->Cp-centroid must sit at the ~1.66 Å crystal distance"
    ens.filter("connectivity")  # re-perceives the graph; must not choke on (or find) a phantom
    assert ens.ids

    with open(ens.dump(str(tmp_path / "ferrocene.xyz"))) as f:
        assert int(f.readline()) == 11, "a centroid dummy reached the dumped xyz"


def test_haptic_complex_survives_the_mc_search():
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc().ids
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc(explore=True).ids


def test_shape_held_haptic_spectator_needs_no_centroid_constraints():
    combined = Chem.CombineMols(ferrocene(), ferrocene(), Point3D(6, 0, 0))
    iso = rx.metal(combined, center=0, stereo="free")[0]
    expected = list(range(iso.mol.GetNumAtoms(), iso.mol.GetNumAtoms() + 2))
    assert sorted(iso.cons.phantoms) == sorted(iso.cons.haptic) == expected
    assert [sum(isinstance(site, HapticSite) for site in state.vertices) for state in iso.centres] == [2, 2]
    assert core_embed(iso, n=1, seed=1).ids


@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_half_sandwich_uses_piano_stool_shape(door):
    iso = rx.metal(cp_ticl3())[0] if door == "enumerate" else I.from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1


def test_haptic_centroid_participates_in_post_dg_metal_hand_selection():
    mol = cp_ticl3()
    sigma = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17]
    sigma[0].SetAtomicNum(9)
    sigma[2].SetAtomicNum(35)
    iso = rx.metal(mol, "tetrahedral")[0]
    assert iso.chirality
    assert len(iso.haptic) == 1
    conformers = core_embed(iso, n=8, seed=7, prune_rms=-1)
    assert len(conformers) == 8
    assert {
        MS.realised_chirality(conformers._mol, cid, iso.geometry, iso.vertices, iso.metal, iso.chirality, iso.haptic)
        for cid in conformers.ids
    } == {iso.chirality}


def test_piano_stool_chlorides_avoid_trans_ring():
    iso = rx.metal(cp_ticl3())[0]
    ens = rx.embed(iso, n=6).minimize()
    assert ens.ids
    pos = ens.mol.GetConformer(next(iter(ens.ids))).GetPositions()
    ring = next(iter(iso.cons.haptic.values()))
    centroid = np.mean([pos[a] for a in ring], axis=0)
    cls = [a.GetIdx() for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 17]
    for cl in cls:
        u, v = centroid - pos[iso.metal], pos[cl] - pos[iso.metal]
        assert np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))) < 150.0
    for a, b in itertools.combinations(cls, 2):
        assert _angle(pos, a, iso.metal, b) < 150.0


def test_eta2_face_collapses_a_cn7_miscount_to_octahedral():
    smi = "CN(C)c1ncc[cH]2->[W+2]34(<-[N-]=O)(<-[cH]12)(<-[n]1cccn1[BH-](n1ccc[n]->31)n1ccc[n]->41)<-[P](C)(C)C"
    isos = rx.metal(smi)
    assert isos
    assert all(iso.geometry == "octahedral" for iso in isos)
    assert all(len(iso.haptic) == 1 for iso in isos)


def test_empty_isomer_enumeration_warns(caplog, monkeypatch):
    import logging

    monkeypatch.setattr(slots, "distinct_vertex_orderings", lambda *a, **kw: [])  # every candidate rejected
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = K.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("Br[Pd]1(Cl)NCCN1")), "square_planar")
    assert not out, "the monkeypatch must leave the enumeration empty"
    assert any("no arrangement survived" in r.getMessage() for r in caplog.records), caplog.text
