"""Test metal-centre handedness perception and selection."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_stereo as metal
from rxembed.metal_isomer import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, orientation_parity
from tests.metal_fixtures import BUTADIENE_FE_CO3

emb = importlib.import_module("rxembed.embed")


@pytest.mark.parametrize(
    ("smiles", "want"),
    [
        ("[O-]S1(=O)=[O]->[Pd+2](<-[Cl-])(<-[P](C)(C)C)<-[O-]1", 1),
    ],
    ids=[
        "kappa2-sulfate-expanded-octet",
    ],
)
def test_resonance_identity_case_table_isomer_counts(smiles, want):
    # The kappa1 acetates sit in two separate conjugated systems drawn in opposite Lewis forms and must
    # still merge into one ligand class. The crotononitriles carry E/Z stereo on a bond inside the merged
    # conjugated system and must stay diastereomeric, so the rule must not erase stated bond stereo. The
    # phosphinate and sulfate oxygens sit on a p-block centre RDKit does not perceive as conjugated and must
    # still merge, the same drawing choice as a carboxylate's.
    assert len(rx.metal(smiles, "SPL")) == want


def test_aqua_chloride_hydrogen_bond_is_not_a_chelate_bite():
    mol = rx.parse_smiles("[H]O([H])->[Pt+2](<-[NH3])(<-[Br-])<-[Cl-]", remove_hs=False)
    contact = Chem.RWMol(mol)
    contact.AddBond(0, 6, Chem.BondType.ZERO)  # H0...Cl6

    assert len(rx.metal(contact.GetMol(), "square_planar")) == len(rx.metal(mol, "square_planar")) == 3


def test_face_winding_abstains_at_the_plane_and_is_scale_invariant():
    face_mol = Chem.MolFromSmiles("[c-]1(F)c(Br)ccc1")
    rw = Chem.RWMol(Chem.CombineMols(face_mol, Chem.MolFromSmiles("[Fe+2]")))
    metal_idx = rw.GetNumAtoms() - 1
    face = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetIsAromatic()]
    for atom in face:
        rw.AddBond(atom, metal_idx, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    pos = np.zeros((mol.GetNumAtoms(), 3))
    theta = 2 * np.pi * np.arange(len(face)) / len(face)
    pos[face, 0], pos[face, 1] = np.cos(theta), np.sin(theta)
    pos[metal_idx] = (2.0, 0.0, 0.0)

    assert metal.face_winding(mol, pos, metal_idx, face, metal.donor_classes(mol, face)) == ""

    eta2 = Chem.AddHs(rx.parse_smiles(r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"))
    metal_idx, face = 4, (1, 2)
    donors = [
        bond.GetBeginAtomIdx()
        for bond in eta2.GetBonds()
        if bond.GetBondType() == Chem.BondType.DATIVE and bond.GetEndAtomIdx() == metal_idx
    ]
    pos = np.zeros((eta2.GetNumAtoms(), 3))
    for atom, point in {
        0: (-1.2, 0.7, 0.0),
        1: (-0.5, 0.0, 0.0),
        2: (0.5, 0.0, 0.0),
        3: (1.2, 0.7, 0.0),
        4: (0.0, -1.0, 1e-6),
        11: (-1.2, -0.7, 0.0),
        12: (1.2, -0.7, 0.0),
    }.items():
        pos[atom] = point
    ranks = metal.donor_classes(eta2, donors)
    cip = list(Chem.ComputeAtomCIPRanks(eta2))

    assert {metal.face_winding(eta2, pos * scale, metal_idx, face, ranks, cip) for scale in (1e-3, 1.0, 1e3)} == {"-"}


def test_diene_class_turns_at_ninety_degrees_and_abstains_on_a_boundary():
    mol = Chem.MolFromSmiles(BUTADIENE_FE_CO3)
    iron, path = 2, [7, 8, 9, 10]
    ranks = metal.donor_classes(mol, path)

    def token(degrees):
        phi = np.radians(degrees)
        pos = np.zeros((mol.GetNumAtoms(), 3))
        pos[path] = [(-0.7, 1.2, 0.0), (0.0, 0.0, 0.0), (1.45, 0.0, 0.0), (2.15, 1.2 * np.cos(phi), 1.2 * np.sin(phi))]
        pos[iron] = (0.7, 0.6, -1.7)
        return metal.face_winding(mol, pos, iron, path, ranks)

    assert [token(degrees) for degrees in (0, 89, 91, 179, -91, 90, 180)] == ["c", "c", "P", "P", "M", "", ""]


def _orientation_parity(mol, cid, iso):
    """Fit every observed vertex to the ideal shape and return proper (+1) or mirrored (-1)."""
    pos = mol.GetConformer(int(cid)).GetPositions()

    def point(atom):
        ring = iso.haptic.get(atom)
        return np.mean(pos[list(ring)], axis=0) if ring else pos[atom]

    observed = np.asarray([point(atom) - pos[iso.metal] for atom in iso.vertices])
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    ideal = np.asarray(POLYHEDRA[iso.geometry].vertex_dirs, float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    return orientation_parity(observed, ideal)


def test_orientation_reader_skips_achiral_coplanar_and_collapsed_centres():
    iso = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", "square_planar")[0]
    assert metal.realised_chirality(iso.mol, 0, iso.geometry, iso.vertices, iso.metal, iso.chirality, iso.haptic) == ""

    # A vacant vertex is a direction with no atom: five distinct donors place it, two in one plane cannot.
    assert rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]", "octahedral")[0].chirality
    assert not rx.metal("[Cl-]->[Zn+2]<-[Br-]", "trigonal_pyramidal")[0].chirality

    chiral = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "octahedral")[0]
    mol = Chem.Mol(chiral.mol)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    assert (
        metal.realised_chirality(
            mol, 0, chiral.geometry, chiral.vertices, chiral.metal, chiral.chirality, chiral.haptic
        )
        == ""
    )


def test_chiral_ligand_filters_wrong_metal_hands_before_uff(monkeypatch):
    smiles = "O->[Co+3](<-[Cl-])(<-[CH3-])(<-[NH2][C@H](C)CC)(<-[F-])<-P"
    expected = [x for x in Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles), includeUnassigned=True) if x[1] != "?"]
    iso = rx.metal(smiles, "octahedral")[0]
    calls = []
    native = emb.seed_coordinates

    def counted(*args, **kwargs):
        calls.append(args[2])
        ids = list(native(*args, **kwargs))
        if len(calls) == 1:  # make the first batch entirely wrong-handed to exercise the retry
            mol = args[0]
            for cid in ids:
                if _orientation_parity(mol, cid, iso) == 1:
                    emb._reflect(mol, cid)
        return ids

    monkeypatch.setattr(emb, "seed_coordinates", counted)
    conformers = emb.embed(iso, n=8, params=rx.EmbedParams(seed=7, prune_rms=-1))
    assert len(conformers) == 8
    assert calls[0] > 8, "a mirror-unsafe ligand must sample both DG hands before selecting one"
    assert len(calls) > 1, "an all-wrong first batch must retry before cleanup"
    assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}
    for cid in conformers.ids:
        one = Chem.Mol(conformers.mol, False, int(cid))
        one.GetAtomWithIdx(expected[0][0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistryFrom3D(one, confId=one.GetConformer().GetId(), replaceExistingTags=True)
        assert [x for x in Chem.FindMolChiralCenters(one, includeUnassigned=True) if x[1] != "?"] == expected


def _distinct_donor_isomer(geometry):
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(92))
    donors = []
    for isotope in range(14, 14 + POLYHEDRA[geometry].cn):
        donor = Chem.Atom(7)
        donor.SetIsotope(isotope)
        index = rw.AddAtom(donor)
        for _ in range(3):
            rw.AddBond(index, rw.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
        rw.AddBond(index, metal, Chem.BondType.DATIVE)
        donors.append(index)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Isomer(mol, geometry, donors)


def test_post_dg_gate_is_geometry_derived_for_every_nonplanar_polyhedron():
    for geometry, polyhedron in POLYHEDRA.items():
        if polyhedron.planar:
            continue
        iso = _distinct_donor_isomer(geometry)
        assert iso.chirality in {"delta", "lambda"}, geometry
        conformers = emb.embed(iso, n=2, params=rx.EmbedParams(seed=7, prune_rms=-1))
        assert len(conformers) == 2, geometry
        assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}, geometry
