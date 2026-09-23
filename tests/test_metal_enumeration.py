"""Test metal-isomer enumeration and embedding integration."""

from __future__ import annotations

import csv
import itertools
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_enumeration as K  # noqa: N812
from rxembed import metal_isomer as I  # noqa: N812
from rxembed import metal_slots as slots
from rxembed import metal_stereo as MS  # noqa: N812
from rxembed import stereo as ligand_stereo
from rxembed.constraints import Constraints
from rxembed.embed import embed as core_embed
from rxembed.metal_core import HapticSite, metal_indices
from rxembed.metal_polyhedron import hull_edges, vertex_dirs
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


def _assert_embeds_as(iso, *, extra=None, seed=42, threads=1):
    """Embed one screened isomer and assert it relaxes clean and keeps its identity; return the ensemble."""
    ensemble = rx.embed(iso, n=1, seed=seed, threads=threads)
    assert not ensemble.unrelaxed
    for report in ensemble.check().values():
        assert report.ok(), report.summary()
    assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
    if extra is not None:
        extra(ensemble, iso)
    return ensemble


def _agostic_hand_angle_below_110(ensemble, iso):
    mol = ensemble.mol
    phosphorus = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetAtomicNum() == 15)
    hydrogen = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetAtomicNum() == 1)
    assert rdMolTransforms.GetAngleDeg(mol.GetConformer(), phosphorus, iso.metal, hydrogen) < 110


def test_resonance_dependent_ez_is_not_enumerated_from_geometry():
    mol = Chem.MolFromSmiles(r"CN(C)/C(C)=C1/C=CC=C[CH-]1")
    rdDepictor.Compute2DCoords(mol)
    mol.GetConformer().Set3D(True)
    form = Chem.Mol(Chem.ResonanceMolSupplier(mol, maxStructs=3)[0])
    Chem.RemoveStereochemistry(form)

    variants, _mode, n_unassigned, unresolved = K._ligand_stereo_variants(form, "all")

    assert [label for _variant, label in variants] == [""]
    assert n_unassigned == unresolved == 0


def test_coordination_locked_imine_is_not_duplicated_as_ez():
    iso = rx.metal("C1=[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-[NH2]CC1", "square_planar")[0]
    realised = rx.embed(iso, n=1, seed=7).mol
    locked = ligand_stereo._coordination_locked_double_bonds(realised, metal_indices(realised))
    back = rx.metal(realised)
    text = rx.cxsmiles(realised)
    matches = [candidate for candidate in back if rx.cxsmiles(candidate) == text]

    assert locked
    assert len(matches) == 1
    assert all(not ligand_stereo.bond_stereo(candidate.stereo_label) for candidate in back)
    assert rx.cxsmiles(rx.metal(text)[0]) == text


def test_encoded_locked_ez_keeps_equivalent_chelate_sites_equivalent():
    text = "O[N]1=Cc2cccc[n]2->[Cd+2]<-12(<-[I-])(<-[I-])<-[N](O)=Cc1cccc[n]->21 |t:1,atomProp:12._rxEZ0.E:14._rxEZ0.E|"
    mol = rx.parse_smiles(text)
    candidates = rx.metal(mol, screen=False)

    assert len(candidates) == 11
    assert len({rx.cxsmiles(iso) for iso in candidates}) == len(candidates)


def test_unbound_metal_has_no_coordination_isomer():
    source = rx.parse_smiles("[Hg].[C-]#[O+]")

    with pytest.raises(ValueError, match="Hg0 has no donor bonds"):
        rx.metal(source)


def test_boron_cage_fails_at_enumeration_boundary():
    source = rx.parse_smiles("[BH-]1[BH-][BH-][BH-][BH-]1->[Fe+2]")

    with pytest.raises(ValueError, match=r"boron cage ligand.*two-centre donor model"):
        rx.metal(source)
    assert rx.metal("[BH3-][H]->[Fe+]")

    conformer = Chem.Mol(Chem.AddHs(source))
    coordinates = Chem.Conformer(conformer.GetNumAtoms())
    for index in range(conformer.GetNumAtoms()):
        coordinates.SetAtomPosition(index, (float(index), 0.0, 0.0))
    coordinates.Set3D(True)
    conformer.AddConformer(coordinates)
    with pytest.raises(ValueError, match=r"boron cage ligand.*two-centre donor model"):
        rx.embed(conformer)


def test_selective_ez_clear_keeps_a_shared_direction_for_the_other_double_bond():
    mol = Chem.MolFromSmiles(r"F/C=C/C=C/C=C/F")
    pairs = ligand_stereo.bond_stereo(ligand_stereo.defined_stereo_label(mol))
    target = sorted(pairs, key=min)[1]
    shared = [mol.GetBondBetweenAtoms(2, 3), mol.GetBondBetweenAtoms(4, 5)]

    assert all(bond.GetBondDir() != Chem.BondDir.NONE for bond in shared)
    K._clear_ez(mol, {target})

    remaining = ligand_stereo.bond_stereo(ligand_stereo.defined_stereo_label(mol))
    assert target not in remaining
    assert remaining == {pair: label for pair, label in pairs.items() if pair != target}
    assert all(bond.GetBondDir() != Chem.BondDir.NONE for bond in shared)


def test_public_haptic_chelate_bounds_hide_internal_uff_typer_diagnostics(capfd):
    complex_smi = r"C/[CH]1=[CH](\F)->[Pt+2]2(<-[S-]CC[NH2]->2)(<-[Cl-])<-1"
    iso = rx.metal(complex_smi, "square_planar").select(index=0)
    _cons = iso.cons
    rx.embed(iso, n=1, seed=7)

    assert "UFFTYPER" not in capfd.readouterr().err


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


def test_covalent_and_dative_notation_share_enumeration():
    covalent = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")
    dative = rx.metal("[Br-]->[Pd+4]1(<-[Cl-])<-[NH-]CC[NH-]->1", "square_planar")

    assert [rx.cxsmiles(iso) for iso in covalent] == [rx.cxsmiles(iso) for iso in dative]
    assert [iso.cons for iso in covalent] == [iso.cons for iso in dative]


def test_input_geometry_does_not_invent_an_unmeasurable_donor_hand():
    mol = rx.parse_smiles("C[N](CC)(CCC)->[Pt+2](<-[Cl-])(<-[Br-])<-[I-]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(index, Point3D(float(index), 0, 0))
    mol.AddConformer(conf)

    isomers = rx.metal(mol, "square_planar", lengths="model")

    assert isomers
    assert all(not iso.stereo_label for iso in isomers)


def test_input_geometry_preserves_the_indexed_hand_of_equivalent_donors():
    mol = rx.parse_smiles("[C-](F)(Cl)(Br)->[Pt+2](<-[C-](F)(Cl)Br)(<-[I-])<-P")
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = [
        (2, 0, 0),
        (2.4, 1, 1),
        (2.4, -1, 1),
        (2.4, 0, -1),
        (0, 0, 0),
        (-2, 0, 0),
        (-2.4, 1, 1),
        (-2.4, -1, 1),
        (-2.4, 0, -1),
        (0, 2, 0),
        (0, -2, 0),
    ]
    for atom, position in enumerate(positions):
        conf.SetAtomPosition(atom, Point3D(*position))
    mol.AddConformer(conf)

    isomers = rx.metal(mol, "square_planar")

    assert isomers
    assert {iso.stereo_label for iso in isomers} == {"C0:S,C5:R"}
    assert all(sum(value == 0.0 for value in iso.cons.umbrellas.values()) == 2 for iso in isomers)


def test_input_geometry_does_not_rewrite_an_already_correct_phosphorus_hand():
    mol = rx.parse_smiles("CO[P@]1(=O)[O-]->[Y+3](<-[Cl-])(<-[Cl-])<-[N]=C1C")
    assert ligand_stereo.defined_stereo_label(mol, metal_indices(mol)) == "P2:S"
    realised = rx.embed(rx.metal(mol, "tetrahedral")[0], n=1, seed=42).mol

    assert ligand_stereo.defined_stereo_label(realised, metal_indices(realised)) == "P2:S"
    assert ligand_stereo.stereo_from_3d(realised, metal_indices(realised)) == "P2:S"
    assert len(rx.metal(realised)) == 1


def test_frozen_site_permutations_are_lazy_and_keep_the_input_vertices(monkeypatch):
    monkeypatch.setattr(slots, "input_ordering", lambda *_args: [2, 0, 3, 1])

    pool = K._frozen_permutations(Chem.Mol(), 0, [10, 11, 12, 13], "square_planar", {10, 12}, 4)

    assert iter(pool) is pool
    assert list(pool) == [[2, 0, 1, 3], [2, 0, 3, 1]]


def test_frozen_site_permutations_preserve_a_missing_input_order(monkeypatch):
    monkeypatch.setattr(slots, "input_ordering", lambda *_args: None)

    assert K._frozen_permutations(Chem.Mol(), 0, [10, 11], "linear", {10}, 2) is None


def test_frozen_high_coordination_refuses_a_factorial_free_site_pool(monkeypatch):
    monkeypatch.setattr(slots, "input_ordering", lambda *_args: list(range(10)))

    with pytest.raises(ValueError, match=r"fix= leaves exactly 362,880 free-site arrangements.*retain more"):
        K._frozen_permutations(Chem.Mol(), 0, list(range(10)), "BSA", {0}, 10)


def test_monodentate_imine_ez_is_retained_without_mutating_the_variant_graph():
    mol = rx.parse_smiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Br-])<-[I-]")
    bond = mol.GetBondBetweenAtoms(1, 2)
    bond.SetStereoAtoms(0, 3)
    bond.SetStereo(Chem.BondStereo.STEREOE)

    variants, *_rest = K._ligand_stereo_variants(mol, "unassigned")

    assert variants[0][1] == "C1=N2:E"
    assert variants[0][0].GetBondBetweenAtoms(1, 2).GetStereo() == Chem.BondStereo.STEREOE


def test_coordinated_carbonyl_oxygen_does_not_invent_ez():
    mol = rx.parse_smiles("CC=[O]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")

    variants, *_rest = K._ligand_stereo_variants(mol, "unassigned")

    assert len(variants) == 1
    assert not ligand_stereo.bond_stereo(variants[0][1])


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
    embeddable = []
    for iso in isos:
        try:
            if rx.embed(iso, n=2).minimize().n:
                embeddable.append(iso)
        except rx.EmbeddingError:
            pass
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
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    horizontal = (prisms[0][0], prisms[0][1])
    vertical = (prisms[1][3], prisms[1][8])

    assert len(set(MS.donor_classes(mol, [*horizontal, *vertical]).values())) == 1
    classes = MS.site_classes(mol, [100, 101], {100: horizontal, 101: vertical})
    assert classes[100] != classes[101]


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


@pytest.mark.parametrize(("linker", "count"), [("-", 1), ("CC", 1), ("CCCCCCC", 2)])
def test_bipyridyl_reach_respects_donor_direction_and_linker_length(linker, count):
    smiles = "[Cl-]->[Ni+2]1(<-[Br-])<-[n]2ccccc2-c2cccc[n]->12".replace("-c2", f"{linker}c2")
    mol = rx.parse_smiles(smiles)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    identities = {rx.cxsmiles(iso) for iso in rx.metal(mol, "SPL")}
    assert len(identities) == count
    assert {rx.cxsmiles(iso) for iso in rx.metal(reversed_mol, "SPL")} == identities


def test_bipyridyl_reach_preserves_bond_change_authority():
    mol = rx.parse_smiles("[Cl-]->[Ni+2]1(<-[Br-])<-[n]2ccccc2-c2cccc[n]->12")
    link = next(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == S and bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()
    )

    assert len(rx.metal(mol, "SPL", fix={(link.GetBeginAtomIdx(), link.GetEndAtomIdx()): 2.0})) == 2


@pytest.mark.parametrize("metal", ["Ni", "Co", "Rh", "Ir"])
@pytest.mark.parametrize("linker", ["-", "CC"])
def test_two_bipyridyl_chelates_share_the_compiled_coordination_network(metal, linker):
    smiles = (f"[Cl-]->[{metal}+2]12(<-[Cl-])(<-[n]3ccccc3-c3cccc[n]->13)<-[n]3ccccc3-c3cccc[n]->23").replace(
        "-c3", f"{linker}c3"
    )
    mol = rx.parse_smiles(smiles)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    identities = {rx.cxsmiles(iso) for iso in rx.metal(mol, "OCT")}
    assert len(identities) == 3
    assert {rx.cxsmiles(iso) for iso in rx.metal(reversed_mol, "OCT")} == identities
    if linker == "-":
        link = next(
            bond
            for bond in mol.GetBonds()
            if bond.GetBondType() == S and bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()
        )
        unrestricted = {
            rx.cxsmiles(iso): iso
            for iso in rx.metal(mol, "OCT", fix={(link.GetBeginAtomIdx(), link.GetEndAtomIdx()): 2.0})
        }
        assert len(unrestricted) == 5
        assert {rx.cxsmiles(iso) for iso in rx.metal(mol, "OCT", screen=False)} == unrestricted.keys()
        assert identities <= unrestricted.keys()
        ligand_graph = Chem.FragmentOnBonds(
            mol, [bond.GetIdx() for bond in mol.GetBonds() if bond.GetBondType() == DAT], addDummies=False
        )
        pairs = [
            [i for i in fragment if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
            for fragment in Chem.GetMolFrags(ligand_graph)
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in fragment)
        ]
        assert len(pairs) == 2
        directions = np.asarray(K.POLYHEDRA["octahedral"].vertex_dirs)
        for missing in unrestricted.keys() - identities:
            excluded = unrestricted[missing]
            opposed = [
                np.allclose(directions[excluded.vertices.index(left)], -directions[excluded.vertices.index(right)])
                for left, right in pairs
            ]
            assert any(opposed)


def test_dodecahedral_bipyridyl_screen_rejects_unspanned_whole_ligand_assignments():
    smiles = "[F-]->[Nb+4]12(<-[F-])(<-[F-])(<-[F-])(<-[n]3ccccc3-c3cccc[n]->13)<-[n]1ccccc1-c1cccc[n]->21"
    mol = rx.parse_smiles(smiles)
    coordinate_free = Chem.Mol(mol)
    assert len(rx.metal(coordinate_free, "dodecahedral", screen=False)) == 66
    assert len(rx.metal(coordinate_free, "dodecahedral")) == 29
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 41)
    donors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetAtomPosition(metal, (0.0, 0.0, 0.0))
    directions = np.asarray(K.POLYHEDRA["dodecahedral"].vertex_dirs, float)
    for slot, donor in enumerate(donors):
        # Deliberately exceed native bipyridyl reach only when these radii are requested.
        radius = 3.0 if mol.GetAtomWithIdx(donor).GetAtomicNum() == 7 else 1.9
        conformer.SetAtomPosition(donor, tuple(radius * directions[slot]))
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in {metal, *donors}:
            conformer.SetAtomPosition(atom.GetIdx(), (10.0 + atom.GetIdx(), 0.0, 0.0))
    mol.AddConformer(conformer)

    unrestricted = rx.metal(mol, "dodecahedral", screen=False)
    screened = rx.metal(mol, "dodecahedral")

    assert len(unrestricted) == 66
    assert {rx.cxsmiles(iso) for iso in screened} == {
        rx.cxsmiles(iso) for iso in rx.metal(coordinate_free, "dodecahedral")
    }
    assert len(rx.metal(mol, "dodecahedral", lengths="input", screen=False)) == 66
    assert len(rx.metal(mol, "dodecahedral", lengths="input")) == 0


@pytest.mark.parametrize(
    ("smiles", "geometry", "count", "extra"),
    [
        ("[Cl-]->[Ni+2]1(<-[Br-])<-[n]2ccccc2-c2cccc[n]->12", "SPL", 1, None),
        (
            "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1",
            "SPL",
            2,
            _agostic_hand_angle_below_110,
        ),
    ],
    ids=["bipyridyl", "tethered_phosphine_agostic_hydrogen"],
)
def test_screened_isomers_embed_and_keep_identity(smiles, geometry, count, extra):
    isomers = rx.metal(smiles, geometry)
    assert len(isomers) == count
    identities = {rx.cxsmiles(iso) for iso in isomers}
    assert len(identities) == count
    for iso in isomers:
        _assert_embeds_as(iso, extra=extra)


def test_refined_chelate_screen_keeps_explicit_authority_and_embeddable_states(monkeypatch):
    smiles = "[Cl-]->[Ni+2]12(<-[Cl-])(<-[n]3ccccc3CCc3cccc[n]->13)<-[n]3ccccc3CCc3cccc[n]->23"
    mol = rx.parse_smiles(smiles)
    retained = rx.metal(mol, "OCT")
    assert len(retained) == 3
    for iso in retained:
        _assert_embeds_as(iso)
    unrestricted = rx.metal(mol, "OCT", screen=False)
    assert len(unrestricted) == 5
    for iso in unrestricted:
        stated = rx.metal(rx.cxsmiles(iso))
        assert len(stated) == 1
        assert rx.cxsmiles(stated[0]) == rx.cxsmiles(iso)
    bridge = next(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == S
        and all(
            atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
    )
    assert len(rx.metal(mol, "OCT", fix={(bridge.GetBeginAtomIdx(), bridge.GetEndAtomIdx()): 1.5})) == 5
    certificate = K._bounds._euclidean_conflict
    monkeypatch.setattr(K._bounds, "_euclidean_conflict", lambda matrix, **_kwargs: certificate(matrix))
    assert len(rx.metal(mol, "OCT")) == 5


@pytest.mark.parametrize(("carbons", "count"), [(2, 2), (3, 2), (5, 3), (7, 3)])
def test_compiled_network_screen_preserves_cis_and_long_trans_chelates(carbons, count):
    mol = rx.parse_smiles(f"Cl[Pt]1(F)N(C){'C' * carbons}N1")
    isomers = rx.metal(mol)
    identities = {rx.cxsmiles(iso) for iso in isomers}
    assert len(isomers) == len(identities) == count
    order = list(reversed(range(mol.GetNumAtoms())))
    assert {rx.cxsmiles(iso) for iso in rx.metal(Chem.RenumberAtoms(mol, order))} == identities
    for iso in isomers:
        _assert_embeds_as(iso)


def test_short_trans_requires_euclidean_consistency_not_only_triangle_smoothing(monkeypatch):
    """Disabling either check alone still excludes the trans chelate; the edge rule now also proves it.

    The dropped isomer puts the N,N chelate on square_planar's one non-edge (trans) vertex pair, so
    `metal_slots._chelate_edge_links` excludes it independently of `_euclidean_conflict`; both must be
    disabled together to show triangle smoothing alone would have let it through.
    """
    smiles = "Cl[Pt]1(F)N(C)CCCN1"
    assert len(rx.metal(smiles)) == 2
    assert len(rx.metal(smiles, screen=False)) == 3
    monkeypatch.setattr(K._bounds, "_euclidean_conflict", lambda _matrix, **_kwargs: None)
    assert len(rx.metal(smiles)) == 2
    monkeypatch.setattr(slots, "_chelate_edge_links", lambda *args, **kwargs: frozenset())
    assert len(rx.metal(smiles)) == 3


def test_trans_reach_screen_uses_the_shared_150_degree_slot_boundary(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.C.C.C.[He]")
    iso = SimpleNamespace(
        centres=(0, 1),
        _base_cons=Constraints(),
        _graph=mol,
        metal=4,
        vertices=(0, 1, 2, 3),
        haptic={},
        geometry="square_pyramidal",
        _length_mol=mol,
        _lengths="model",
        real_z=46,
        donors=(0, 1, 2, 3),
    )
    reach = np.full((5, 5), 10.0)
    reach[1, 3] = reach[3, 1] = 3.0
    monkeypatch.setattr(K._constraints, "_donor_distance_window", lambda *_args, **_kwargs: (2.0, 2.1))
    monkeypatch.setattr(K, "_donor_facing_failure", lambda *_args: None)

    failure = K._unreachable_span(iso, reach, {}, ())

    needed = K._mech._law_of_cosines(2.0, 2.0, slots.TRANS_ANGLE)
    assert failure == f"donors 1/3 need >= {needed:.3f} A; ligand reach <= 3.000 A"


def test_bonded_donors_keep_native_triangle_for_reach(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("CC.C.C.[He]")
    iso = SimpleNamespace(
        centres=(0, 1),
        _base_cons=Constraints(),
        _graph=mol,
        metal=4,
        vertices=(0, 2, 1, 3),
        haptic={},
        geometry="square_planar",
        _length_mol=mol,
        _lengths="input",
        real_z=46,
        donors=(0, 1, 2, 3),
    )
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    reach = np.full((5, 5), 10.0)
    reach[0, 1] = reach[1, 0] = 3.0
    monkeypatch.setattr(K._constraints, "_donor_distance_window", lambda *_args, **_kwargs: (2.0, 2.1))
    monkeypatch.setattr(K, "_donor_facing_failure", lambda *_args: None)

    assert K._unreachable_span(iso, reach, {}, ()) is None
    iso.vertices = (0, 2, 3, 1)
    assert K._unreachable_span(iso, reach, {}, ()) is None


def test_embedded_chelate_survives_input_length_screening():
    smiles = "[Zn+2]12(<-[NH2]CC[NH2]->1)<-[NH2]CC[NH2]->2"
    iso = rx.metal(smiles, "TET")[0]
    embedded = rx.embed(iso, n=1, seed=42, threads=1).mol

    assert len(rx.metal(embedded, lengths="model")) == 1
    assert len(rx.metal(embedded, lengths="input")) == 1
    assert len(rx.metal(embedded, lengths="input", screen=False)) == 1


def test_direct_donor_bond_keeps_its_native_triangle_for_input_screening():
    mol = rx.parse_smiles("N1N->[Zn+2](<-[Cl-])(<-[Cl-])<-1")
    span = 1.398
    radius = 2.0
    theta = np.arcsin(span / (2 * radius))
    chloride_radius = 2.2
    chloride_z = -chloride_radius / np.sqrt(3)
    chloride_y = chloride_radius * np.sqrt(2 / 3)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, Point3D(radius * np.sin(theta), 0.0, radius * np.cos(theta)))
    conformer.SetAtomPosition(1, Point3D(-radius * np.sin(theta), 0.0, radius * np.cos(theta)))
    conformer.SetAtomPosition(2, Point3D(0.0, 0.0, 0.0))
    conformer.SetAtomPosition(3, Point3D(0.0, chloride_y, chloride_z))
    conformer.SetAtomPosition(4, Point3D(0.0, -chloride_y, chloride_z))
    mol.AddConformer(conformer)

    reference = rx.metal(mol, lengths="input", screen=False)
    screened = rx.metal(mol, lengths="input")

    assert np.linalg.norm(conformer.GetPositions()[0] - conformer.GetPositions()[1]) == pytest.approx(span)
    assert len(reference) == 1
    assert reference[0].geometry == "tetrahedral"
    assert {rx.cxsmiles(iso) for iso in screened} == {rx.cxsmiles(iso) for iso in reference}


@pytest.mark.parametrize(("carbons", "count"), [(3, 1), (7, 2)])
def test_cyclic_chelate_checks_equal_length_routes_without_losing_long_trans(carbons, count):
    mol = rx.parse_smiles(f"Cl[Pt]1(F)N2{'C' * carbons}N1{'C' * carbons}2")
    states = rx.metal(mol, "SPL")
    identities = {rx.cxsmiles(iso) for iso in states}
    assert len(states) == len(identities) == count
    assert len(rx.metal(mol, "SPL", screen=False)) == 2
    rng = np.random.default_rng(42)
    for _ in range(4):
        permuted = Chem.RenumberAtoms(mol, rng.permutation(mol.GetNumAtoms()).tolist())
        assert {rx.cxsmiles(iso) for iso in rx.metal(permuted, "SPL")} == identities
    for iso in states:
        _assert_embeds_as(iso)


def test_equal_length_routes_are_checked_together(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C1CCC1.[He]")
    iso = SimpleNamespace(_graph=mol, cons=None, metal=4, donors=(0, 2))
    points = np.array(((-1.0, 0.0, 0.0), (0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.8, -0.6, 0.0), (0.0, 0.0, 0.0)))
    matrix = np.linalg.norm(points[:, None] - points, axis=2)
    monkeypatch.setattr(K._bounds, "coordination_reach", lambda *_: matrix.copy())
    assert K._compiled_span_failure(iso, None) is None
    # Both donor routes remain realizable alone, but their off-axis points cannot be this far apart.
    matrix[1, 3] = matrix[3, 1] = 1.24
    assert K.DistanceGeometry.DoTriangleSmoothing(matrix.copy())
    assert K._compiled_span_failure(iso, None) is not None


def test_compiled_span_uses_local_triangle_before_global_certificate(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.[He].[He].[He].[He]")
    constraints = Constraints(
        metals={4},
        distances={(0, 4): (2.0, 2.0), (2, 4): (2.0, 2.0)},
        angles={(0, 4, 2): (136.0, 152.0)},
    )
    iso = SimpleNamespace(_graph=mol, cons=constraints, metal=4, donors=(0, 2))
    native = np.full((5, 5), np.inf)
    native[2, 0], native[0, 2] = 0.0, 2.5
    monkeypatch.setattr(
        K._bounds,
        "coordination_reach",
        lambda *_args, **_kwargs: pytest.fail("the local contradiction should short-circuit the global certificate"),
    )

    assert K._compiled_span_failure(iso, native, native=native) == (
        "compiled coordination distances conflict with native ligand reach"
    )


def test_repeated_route_unions_do_not_repeat_the_same_search(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C1CCC1.[He]")
    iso = SimpleNamespace(_graph=mol, cons=None, metal=4, donors=(0, 1, 2, 3))
    matrix = np.ones((5, 5)) - np.eye(5)
    calls = Counter()

    def checked(block, **_kwargs):
        calls[len(block)] += 1

    monkeypatch.setattr(K._bounds, "coordination_reach", lambda *_: matrix.copy())
    monkeypatch.setattr(K._bounds, "_euclidean_conflict", checked)
    monkeypatch.setattr(K, "_route_has_donor_bond", lambda *_args: False)
    assert K._compiled_span_failure(iso, None) is None
    assert calls == {3: 4, 4: 4, 5: 1}


def test_compiled_span_skips_a_bridged_donor_route(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("NN.[He]")
    iso = SimpleNamespace(_graph=mol, cons=None, metal=2, donors=(0, 1))
    matrix = np.ones((3, 3)) - np.eye(3)
    monkeypatch.setattr(K._bounds, "coordination_reach", lambda *_args, **_kwargs: matrix.copy())
    monkeypatch.setattr(K._bounds, "_euclidean_conflict", lambda *_args, **_kwargs: pytest.fail("bridged route"))

    assert K._route_has_donor_bond(mol, (0, 1), {0, 1})
    assert K._compiled_span_failure(iso, matrix, native=matrix) is None


def test_long_route_keeps_only_the_complete_euclidean_witness():
    path = tuple(range(9))

    assert list(K._route_certificate_subsets(path)) == [path]


@pytest.mark.parametrize(("carbons", "count"), [(3, 1), (7, 2)])
def test_separate_pi_ligand_does_not_disable_chelate_network_screen(carbons, count):
    smiles = f"[Cl-]->[Pt+2]12(<-[NH2]{'C' * carbons}[NH2]->1)<-[CH2]=[CH2]->2"
    mol = rx.parse_smiles(smiles)
    isomers = rx.metal(mol, "SPL")
    identities = {rx.cxsmiles(iso) for iso in isomers}
    assert len(isomers) == len(identities) == count
    assert all(len(iso.haptic) == 1 for iso in isomers)
    unrestricted = rx.metal(mol, "SPL", screen=False)
    assert len(unrestricted) == 2
    for iso in unrestricted:
        stated = rx.metal(rx.cxsmiles(iso))
        assert len(stated) == 1
        assert rx.cxsmiles(stated[0]) == rx.cxsmiles(iso)
    backbone = next(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == S
        and all(atom.GetAtomicNum() == 6 for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
    )
    assert len(rx.metal(mol, "SPL", fix={(backbone.GetBeginAtomIdx(), backbone.GetEndAtomIdx()): 2.0})) == 2
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert {rx.cxsmiles(iso) for iso in rx.metal(reversed_mol, "SPL")} == identities
    for iso in isomers:
        _assert_embeds_as(iso)


def test_inconclusive_haptic_subset_keeps_the_prior_donor_facing_screen(monkeypatch):
    smiles = "[Cl-]->[Pt+2]12(<-[NH2]CCC[NH2]->1)<-[CH2]=[CH2]->2"
    monkeypatch.setattr(K, "_compiled_span_failure", lambda *_args: None)
    monkeypatch.setattr(K, "_donor_facing_failure", lambda *_args: "prior donor-facing conflict")

    assert len(rx.metal(smiles, "SPL")) == 0
    assert len(rx.metal(smiles, "SPL", screen=False)) == 2


def test_haptic_borane_cage_keeps_realised_mixed_point_stereo():
    smiles = "C[N]1(C)CC[N](C)(C)->[Ni+2]<-123<-[BH]1C4([Si](C)(C)C)BC1([Si](C)(C)C)[BH-]->2=[BH-]->34"
    candidates = rx.metal(smiles, "SPL")
    mixed = [iso for iso in candidates if set(ligand_stereo.point_stereo(iso.stereo_label).values()) == {"R", "S"}]
    assert mixed, "A seed-model contradiction must not discard this realised cage seating"
    _assert_embeds_as(mixed[0])


def test_compiled_span_search_streams_subsets_after_the_whole_path(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("CCCC.[He]")
    iso = SimpleNamespace(_graph=mol, cons=None, metal=4, donors=(0, 3))
    matrix = np.ones((5, 5)) - np.eye(5)
    monkeypatch.setattr(K._bounds, "coordination_reach", lambda *_args: matrix)
    monkeypatch.setattr(K._bounds, "_euclidean_conflict", lambda _matrix, **_kwargs: 1.0)
    combinations = itertools.combinations

    def guarded_subsets(values, size):
        assert size == 2, "smaller subsets were consumed before testing the whole path"
        yield from combinations(values, size)

    monkeypatch.setattr(K.itertools, "combinations", guarded_subsets)
    assert K._compiled_span_failure(iso, matrix) is not None


@pytest.mark.parametrize("enumerate_isomers", [rx.metal, rx.enumerate_isomers], ids=["pipeline", "core"])
@pytest.mark.parametrize(("screen", "count"), [(True, 2), (False, 3)])
def test_agostic_tether_reach_is_atom_order_invariant_and_allows_a_longer_arm(enumerate_isomers, screen, count):
    short = "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1"
    mol = rx.parse_smiles(short)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    isomers = enumerate_isomers(mol, "SPL", screen=screen)
    identities = {rx.cxsmiles(iso) for iso in isomers}
    assert len(isomers) == len(identities) == count
    assert {rx.cxsmiles(iso) for iso in enumerate_isomers(reversed_mol, "SPL", screen=screen)} == identities
    baseline = {rx.cxsmiles(iso): iso.cons for iso in enumerate_isomers(mol, "SPL")}
    assert baseline.keys() <= identities
    for iso in isomers:
        if (identity := rx.cxsmiles(iso)) in baseline:
            assert iso.cons == baseline[identity]
        # Explicit CX slots select the same single state even if the default model would screen it out.
        stated = enumerate_isomers(rx.parse_smiles(rx.cxsmiles(iso)), screen=not screen)
        assert len(stated) == 1
        assert rx.cxsmiles(stated[0]) == rx.cxsmiles(iso)
        assert stated[0].cons == enumerate_isomers(rx.parse_smiles(rx.cxsmiles(iso)), screen=screen)[0].cons
    long = short.replace("C(C)(C)C[H]", "C(C)(C)CCCC[H]")
    assert len(enumerate_isomers(rx.parse_smiles(long), "SPL", screen=screen)) == 3
    assert len(enumerate_isomers(rx.parse_smiles(_MA2B2), "SPL", screen=screen)) == 2


def test_tethered_donor_network_rejects_the_unrealistic_fihtoh_cis_state():
    smiles = "COC(=O)c1cc2O[P](C(C)C)(C(C)C)->[Ni+2]3(<-[Cl-])<-[c-]2c(O[P]->3(C(C)C)C(C)C)c1"

    screened = rx.metal(smiles, "square_planar")
    unrestricted = rx.metal(smiles, "square_planar", screen=False)

    assert len(screened) == 1
    assert screened[0].label == "trans"
    assert len(unrestricted) == 2
    _assert_embeds_as(screened[0])


def test_observed_tethered_screen_keeps_the_matching_assignment(monkeypatch):
    smiles = "COC(=O)c1cc2O[P](C(C)C)(C(C)C)->[Ni+2]3(<-[Cl-])<-[c-]2c(O[P]->3(C(C)C)C(C)C)c1"
    source = rx.embed(rx.metal(smiles, "square_planar")[0], n=1, seed=42, threads=1).mol
    monkeypatch.setattr(slots, "_MAX_EXHAUSTIVE_ORBITS", 1)

    isomers = rx.metal(source, "square_planar")

    assert len(isomers) == 1
    assert isomers[0].label == "trans"


def test_narrow_span_pruning_clears_the_tethered_orbit_cap():
    """A tris-dien La(III) sphere (all 9 donors one fragment) needs `_narrow_span_pairs` under the cap.

    Without the up-front prune, `distinct_vertex_orderings` streams all 10,098 raw orbits of `tricapped_trigonal_
    prismatic` and trips `_MAX_EXHAUSTIVE_ORBITS` (1,000) before the per-candidate reach screen ever runs.
    128 without the chelate edge rule; each of its 6 dien arms (2 per ligand) must additionally land on a
    `tricapped_trigonal_prismatic` hull edge, which drops 66 more (all 66 place an arm on a non-edge pair).
    """
    smiles = "C1C[NH]2CC[NH2]->[La+3]<-23456(<-[NH2]1)(<-[NH2]CC[NH]->3CC[NH2]->4)<-[NH2]CC[NH]->5CC[NH2]->6"
    mol = rx.parse_smiles(smiles)

    isomers = rx.metal(mol)

    assert len(isomers) == 62
    with pytest.raises(ValueError, match="more than 1,000 distinct constitutional"):
        rx.metal(mol, screen=False)


@pytest.mark.parametrize(
    ("smiles", "count"),
    [
        ("[Cl-]->[La+3]12(<-[Cl-])<-[NH2]CC[NH]->1CC[NH2]->2", 5),
        ("[Cl-]->[La+3]123(<-[Cl-])(<-[NH2]CC[NH2]->1)<-[NH2]CC[NH]->2CC[NH2]->3", 24),
    ],
    ids=["tbp", "pbp"],
)
def test_narrow_span_pruning_loses_no_reachable_arrangement(monkeypatch, smiles, count):
    """The pruned and unpruned tethered pools agree exactly: pruning removes only already-doomed orbits."""
    mol = rx.parse_smiles(smiles)

    pruned = {rx.cxsmiles(iso) for iso in rx.metal(mol)}
    assert len(pruned) == count

    monkeypatch.setattr(K, "_narrow_span_pairs", lambda *args, **kwargs: frozenset())
    unpruned = {rx.cxsmiles(iso) for iso in rx.metal(mol)}

    assert unpruned == pruned


_EN_LA_HEXACHLORO_SQA = "[Cl-]->[La+3]1(<-[Cl-])(<-[Cl-])(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[NH2]CC[NH2]->1"


def test_chelate_edge_rule_keeps_every_isomer_on_a_hull_edge():
    """En's two N donors sit on a square-antiprism hull edge in every screened isomer, never a diagonal.

    `screen=False` reaches more: the edge rule (like the reach screen) is a model claim, not a proof that a
    wider placement is unreachable in principle.
    """
    mol = rx.parse_smiles(_EN_LA_HEXACHLORO_SQA)
    edges = hull_edges(tuple(map(tuple, vertex_dirs("square_antiprism"))))
    en_donors = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "N"}

    screened = rx.metal(mol, "SQA")
    for iso in screened:
        pair = frozenset(vertex for vertex, donor in enumerate(iso.vertices) if donor in en_donors)
        assert pair in edges, f"{iso.label}: en placed on a non-edge vertex pair {sorted(pair)}"

    unrestricted = rx.metal(mol, "SQA", screen=False)
    assert len(unrestricted) > len(screened)


def test_chelate_edge_rule_mutation_lets_a_diagonal_placement_survive(monkeypatch):
    """Disabling the edge rule's contribution must break the previous test's guarantee."""
    mol = rx.parse_smiles(_EN_LA_HEXACHLORO_SQA)
    edges = hull_edges(tuple(map(tuple, vertex_dirs("square_antiprism"))))
    en_donors = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "N"}

    monkeypatch.setattr(slots, "_chelate_edge_links", lambda *args, **kwargs: frozenset())
    isomers = rx.metal(mol, "SQA")

    pairs = [frozenset(vertex for vertex, donor in enumerate(iso.vertices) if donor in en_donors) for iso in isomers]
    assert any(pair not in edges for pair in pairs), "the mutation should let a diagonal placement through"


def test_chelate_edge_rule_bonded_pair_forces_the_zudwuq_perimeter():
    """ZUDWUQ: a cyclo-As6 ring on Ni: every ring bond is a 3-membered chelate, so only the perimeter isomer
    survives (a ring-adjacent placement on every hexagonal-planar hull edge); the other two raw isomers put a
    bonded pair on a hexagon diagonal and are dropped.
    """
    smiles = (
        "CC(C)(C)[As]12->[Ni]3456<-[As]1(C(C)(C)C)[As]->3(C(C)(C)C)[As]->4(C(C)(C)C)[As]->5(C(C)(C)C)[As]->62C(C)(C)C"
    )
    mol = rx.parse_smiles(smiles)

    screened = rx.metal(mol, "hexagonal_planar")
    unrestricted = rx.metal(mol, "hexagonal_planar", screen=False)

    assert len(screened) == 1
    assert len(unrestricted) > 1


_TMQMG_DIR = Path("/home/ali/Documents/Codes/tmQMg/data")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.skipif(not _TMQMG_DIR.is_dir(), reason="needs a local tmQMg clone")
@pytest.mark.parametrize("tmqmg_id", ["ROGWIW", "IKOYOX", "KUVQOK"])
def test_observed_only_survives_a_forbidden_measured_arrangement(tmqmg_id):
    """Regression: 3 real tmQMg structures went from 1 (their measured isomer) to 0 isomers once the
    forbidden-pair filter (narrow/linked) incorrectly ran on `observed_only`'s explicit retained order too.
    """
    charges = {
        row["id"]: int(row["charge"])
        for row in csv.DictReader((_TMQMG_DIR / "tmQMg_properties_and_targets.csv").open())
    }
    mol = rx.read_xyz(
        str(_TMQMG_DIR / "xyz" / f"{tmqmg_id}.xyz"),
        charge=charges[tmqmg_id],
        connectivity="xyzgraph",
        bond_orders="xyz2mol",
    )
    assert len(rx.metal(mol, observed_only=True)) == 1


def test_tethered_haptic_faces_reject_an_unreachable_trans_state():
    smiles = "CC#[N]->[Ru+2]123(<-[Cl-])(<-[Cl-])(<-[N]#CC)<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31"

    isomers = rx.metal(smiles, "octahedral", screen=False)
    trans, cis = isomers[0], isomers[1]
    seed = rx.embed(cis, n=1, seed=42, threads=1).mol
    trans._length_mol.AddConformer(Chem.Conformer(seed.GetConformer(0)))
    reach = K._bounds.ligand_reach(trans._length_mol)
    links = K._coord_stereo.chelate_links(trans._graph, trans.vertices, trans.haptic)
    classes = K._coord_stereo.site_classes(trans._graph, trans.vertices, trans.haptic, ())

    assert len(isomers) == 6
    assert "haptic faces" in K._unreachable_span(trans, reach, classes, links)

    model = rx.metal(smiles, "octahedral", screen=False)[0]
    model_reach = K._bounds.ligand_reach(model._length_mol)
    model_links = K._coord_stereo.chelate_links(model._graph, model.vertices, model.haptic)
    model_classes = K._coord_stereo.site_classes(model._graph, model.vertices, model.haptic, ())
    assert "haptic faces" in K._unreachable_span(model, model_reach, model_classes, model_links)


def test_chelate_links_ignore_coordination_shortcuts():
    mol = Chem.MolFromSmiles("CCCCC")
    rw = Chem.RWMol(mol)
    metal = rw.AddAtom(Chem.Atom(78))
    rw.AddBond(0, metal, DAT)
    rw.AddBond(4, metal, DAT)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    assert Chem.GetDistanceMatrix(mol)[0, 4] == 2
    assert MS.chelate_links(mol, (0, 4)) == {frozenset((0, 1)): 4}


def test_haptic_centroid_reach_uses_a_valid_minimum_matching():
    reach = np.full((4, 4), np.inf)
    reach[0, 2], reach[2, 0] = 5.0, 5.0
    reach[0, 3], reach[3, 0] = 1.0, 1.0
    reach[1, 2], reach[2, 1] = 1.0, 1.0
    reach[1, 3], reach[3, 1] = 5.0, 5.0

    assert K._centroid_reach(reach, (0, 1), (2, 3)) == pytest.approx(1.0)


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
def test_haptic_span_keeps_realizable_member_angles(measured, reordered):
    from types import SimpleNamespace

    positions = np.array([[2, 0.7, 0], [2, -0.7, 0], [-2, 0.7, 0], [-2, -0.7, 0], [0, 0, 0]])
    mol = Chem.MolFromSmiles("C.C.C.C.[Mo]")
    metal, left, right = 4, (0, 1), (2, 3)
    if reordered:
        mol = Chem.RenumberAtoms(mol, [4, 3, 2, 1, 0])
        positions = positions[::-1, [1, 2, 0]] + np.array([1.0, 2.0, 3.0])
        metal, left, right = 0, (4, 3), (2, 1)
    conformer = Chem.Conformer(5)
    conformer.SetPositions(positions)
    mol.AddConformer(conformer)
    iso = SimpleNamespace(haptic={5: left, 6: right}, _length_mol=mol, metal=metal)
    reach = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    cons = Constraints(distances={(metal, 5): (2.0, 2.0), (metal, 6): (2.0, 2.0)})
    for atom in (*left, *right):
        radius = float(np.linalg.norm(positions[atom] - positions[metal]))
        cons.distances[tuple(sorted((metal, atom)))] = (radius, radius)

    # The centroids are trans, but their individual member rays are not.
    assert K._haptic_span_failure(iso, reach, metal, 5, 6, 180.0, measured, cons) is None
    for a, b in itertools.product(left, right):
        reach[a, b] = reach[b, a] = 3.5
    assert "centroid reach" in K._haptic_span_failure(iso, reach, metal, 5, 6, 180.0, measured, cons)


@pytest.mark.parametrize("enumerate_isomers", [rx.metal, rx.enumerate_isomers], ids=["pipeline", "core"])
@pytest.mark.parametrize("screen", [None, "False", 0])
def test_enumeration_screen_requires_a_boolean(enumerate_isomers, screen):
    with pytest.raises(TypeError, match="screen must be a bool"):
        enumerate_isomers(rx.parse_smiles(_MA2B2), screen=screen)


def test_explicit_bond_change_bypasses_ground_state_reach_screen():
    smiles = "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1"
    mol = rx.parse_smiles(smiles)
    hydrogen = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    carbon = next(a for a in hydrogen.GetNeighbors() if a.GetAtomicNum() == 6)

    assert len(rx.metal(mol, "SPL", fix={(carbon.GetIdx(), hydrogen.GetIdx()): 2.0})) == 3


@pytest.mark.parametrize("screen", [True, False])
def test_multimetal_numeric_fix_bypasses_ground_state_reach_screen(screen):
    smiles = "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1"
    first = rx.embed(rx.metal(smiles, "SPL")[0], n=1, seed=42, threads=1).mol
    second = rx.embed(rx.metal("N->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL")[0], n=1, seed=42, threads=1).mol
    combined = Chem.CombineMols(first, second, Point3D(8, 0, 0))
    hydrogen = next(
        a for a in first.GetAtoms() if a.GetAtomicNum() == 1 and any(n.GetAtomicNum() == 46 for n in a.GetNeighbors())
    )
    carbon = next(a for a in hydrogen.GetNeighbors() if a.GetAtomicNum() == 6)

    ordinary = rx.metal(combined, center="all", stereo="free", screen=screen)
    assert len(ordinary) == (2 if screen else 3)
    assert len({rx.cxsmiles(iso) for iso in ordinary}) == len(ordinary)
    isomers = rx.metal(
        combined, center="all", stereo="free", fix={(carbon.GetIdx(), hydrogen.GetIdx()): 2.0}, screen=screen
    )
    assert len(isomers) == 3


@pytest.mark.parametrize("geometry", [None, "square_planar"])
def test_only_explicit_observed_selection_bypasses_reach_screen(monkeypatch, geometry):
    smiles = "[NH2]1CC[NH2]->[Pd+2](<-[Cl-])(<-[Br-])<-1"
    mol = rx.embed(rx.metal(smiles, "SPL")[0], n=1, seed=42, threads=1).mol
    reference = rx.cxsmiles(mol)
    monkeypatch.setattr(K, "_unreachable_span", lambda *_args: "incompatible native reach")
    assert not rx.metal(mol, geometry)
    isomers = rx.metal(mol, geometry, observed_only=True)

    assert len(isomers) == 1
    assert rx.cxsmiles(isomers[0]) == reference


def test_observed_only_is_an_explicit_resource_bounded_choice(monkeypatch):
    source = rx.embed(rx.metal(_MA2B2, "SPL")[0], n=1, seed=42, threads=1).mol
    reference = rx.cxsmiles(source)
    monkeypatch.setattr(slots, "_MAX_EXHAUSTIVE_ORBITS", 1)

    with pytest.raises(ValueError, match="more than 1 distinct constitutional"):
        rx.metal(source)

    observed = rx.metal(source, observed_only=True)
    assert len(observed) == 1
    assert rx.cxsmiles(observed[0]) == reference
    assert len(rx.metal(source, "SPL", observed_only=True)) == 1


def test_observed_only_requires_coordinates():
    with pytest.raises(ValueError, match="observed_only=True requires an input conformer"):
        rx.metal(_MA2B2, observed_only=True)


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
    isos = rx.metal("O->[La+3](<-O)(<-N)(<-N)(<-[F-])(<-[F-])(<-[Cl-])<-[Cl-]", "DOD")
    assert len(isos) == 648
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


def test_retained_source_graft_authority_is_independent_of_constraint_compilation():
    from rxembed.embed import prepare

    ref, pos, where = _cis_reference(n=1)
    source = ref.mol
    fix = {i: tuple(pos[i]) for i in [*where["Cl"], where["N"][0]]}
    for spec in (source, I.from_geometry(source)):
        _, cons, _, graft = prepare(spec, fix=fix)
        assert set(fix) <= cons.frozen
        assert set(graft) == set(fix)
    for selected in rx.metal(source):
        with pytest.raises(ValueError, match="coordination-sphere atoms"):
            prepare(selected, fix=fix)
    ens = rx.embed(source, fix=fix, contacts={(0, 2): (1.0, 5.0)}, n=1, seed=42, threads=1)
    assert ens.cons.contacts[0] == {(0, 2)}
    actual = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    for left, right in itertools.combinations(fix, 2):
        assert np.linalg.norm(actual[left] - actual[right]) == pytest.approx(
            np.linalg.norm(pos[left] - pos[right]), abs=1e-6
        )
    assert rx.cxsmiles(ens.mol) == rx.cxsmiles(source)


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


def test_haptic_spectator_uses_the_same_centroid_constraints():
    combined = Chem.CombineMols(ferrocene(), ferrocene(), Point3D(6, 0, 0))
    iso = rx.metal(combined, center=0, stereo="free")[0]
    expected = list(range(iso.mol.GetNumAtoms(), iso.mol.GetNumAtoms() + 4))
    assert sorted(iso.cons.phantoms) == sorted(iso.cons.haptic) == expected
    assert [sum(isinstance(site, HapticSite) for site in state.vertices) for state in iso.centres] == [2, 2]
    assert core_embed(iso, n=1, seed=1).ids
    positions = combined.GetConformer().GetPositions()
    second = combined.GetNumAtoms() // 2
    positions[second + 1 :] = positions[second] + 1.1 * (positions[second + 1 :] - positions[second])
    combined.GetConformer().SetPositions(positions)
    assert rx.metal(combined, center=0, stereo="free")[0].cons == iso.cons


@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_half_sandwich_uses_piano_stool_shape(door):
    iso = rx.metal(cp_ticl3())[0] if door == "enumerate" else I.from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1


@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_open_eta3_allyl_is_not_an_apical_face(door):
    mol = rx.parse_smiles("[C-]->[Pd+2]12(<-[Cl-])<-[CH2]=[CH]->1[CH2-]->2")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(
        ((-1, 1.732, 0), (0, 0, 0), (2, 0, 0), (0.212, -2.432, 0), (-1, -1.732, 0), (-2.212, -1.032, 0))
    ):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    isomers = rx.metal(mol) if door == "enumerate" else [I.from_geometry(mol)]

    assert len(isomers) == 1
    assert isomers[0].geometry == "trigonal_planar"
    assert len(isomers[0].haptic) == 1


def test_coordinate_free_open_haptic_face_is_apical():
    mol = rx.parse_smiles("[C-]->[Pd+2]12(<-[Cl-])(<-N)<-[CH2]=[CH]->1[CH2-]->2")

    isomers = rx.metal(mol, stereo="free")

    assert len(isomers) == 2  # tetrahedral with 4 distinct sites is chiral-only: Lambda/Delta, no cis/trans split
    assert {iso.geometry for iso in isomers} == {"tetrahedral"}
    assert all(len(iso.haptic) == 1 for iso in isomers)


def test_coordinate_free_open_eta4_diene_is_apical():
    mol = "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[CH]->1[CH]->2=[CH2]->3"
    assert {iso.geometry for iso in rx.metal(mol)} == {"tetrahedral"}


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

    # No chelate here (4 distinct monodentates), so narrow/linked stay empty and the generic message applies.
    monkeypatch.setattr(slots, "distinct_vertex_orderings", lambda *a, **kw: [])  # every candidate rejected
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = K.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("Br[Pd](Cl)(N)O")), "square_planar")
    assert not out, "the monkeypatch must leave the enumeration empty"
    assert any("exact symmetry enumeration produced no arrangement" in r.getMessage() for r in caplog.records), (
        caplog.text
    )


def test_empty_isomer_enumeration_from_a_pruned_pool_names_the_screen_remedy(caplog):
    """`linear` has no hull edge at all, so `_chelate_edge_links` prunes every streamed order before the
    per-candidate reach screen ever runs; the warning must still name the `screen=False` remedy, not the
    generic message that `unreachable` alone used to gate.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = rx.metal("[Cu+]1<-[NH2]CC[NH2]->1", "linear")
    assert not out
    assert any("no model-compatible arrangement; use screen=False" in r.getMessage() for r in caplog.records), (
        caplog.text
    )
    # screen=False is the wrong remedy for a model-length contradiction (it returns non-embeddable states);
    # lengths='input' must be named too.
    assert any("lengths='input'" in r.getMessage() for r in caplog.records), caplog.text
