"""Test graph stereoisomer enumeration before embedding."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import stereo
from rxembed.metal_core import metal_indices
from rxembed.metal_smiles import parse_smiles
from tests.metal_fixtures import ONE_ARM_BOUND_PT

_CHIRAL_P_PD = "C[P](CC)(c1ccccc1)[Pd](Cl)(Cl)Cl"
# an alpha-diimine-style chelate: both imine C=N sit in the 5-membered ring the metal closes
_ALPHA_DIIMINE_NI = (
    "O=C1[O-]->[Ni+2]2(<-[N](=C3C(=[N]->2c2cccc4ccccc24)c2cccc4cccc3c24)c2cccc3ccccc23)<-[N-](c2ccccc2)C1c1ccccc1"
)
_ETA2_PT = "C[CH]1=[CH](F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
_NATIVE_ATROP = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"
_AZA_BIARYL_CR = "[O+]#[C-]->[Cr]1(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[n]2cccnc2-c2nccc[n]->12"


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"fixture SMILES did not parse: {smiles}"
    return mol


def _with_metals(smiles):
    """Return ``(mol, metal indices)``: what a coordination caller passes as ``exclude``."""
    mol = _mol(smiles)
    return mol, set(metal_indices(mol))


def _labels(mol, **kw):
    return sorted(label for _variant, label in stereo.enumerate_unassigned(mol, **kw)[0])


# ---------------------------------------------------------------------------------------------------------
# nothing to enumerate: the pass-through, which must be exact
# ---------------------------------------------------------------------------------------------------------


def test_no_stereo_returns_input_object():
    mol = _mol("CCO")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol)
    assert (n_unassigned, total, unresolved) == (0, 1, 0)
    assert variants == [(mol, "")]

    defined = _mol("C[C@H](N)C(=O)O")
    assert stereo.unassigned_centres(defined) == []
    assert stereo.enumerate_unassigned(defined)[1] == 0


def test_coordinated_phosphonate_uses_full_graph_cip_without_radical_warnings(capfd):
    mol, metals = _with_metals("O=[P@](O)([O-]->[Zn+4])C")

    assert stereo.defined_stereo_label(mol, metals) == "P1:R"
    assert "Unusual charge" not in capfd.readouterr().err


def test_metal_point_tag_is_not_ligand_stereo():
    metal, metals = _with_metals("F[Cu@](Cl)(Br)I")
    ligand, ligand_metals = _with_metals("C[P@](F)(Cl)Br.[Cu]")

    assert stereo.point_centres(metal, metals) == set()
    assert stereo.point_centres(ligand, ligand_metals) == {1}


def test_point_cip_memo_tracks_content_not_object_identity():
    """Flipping a chiral tag in place must not get served the previous, now-stale, cached answer."""
    mol, metals = _with_metals(
        "CCO[C@H](c1ccccc1C)[CH]1=[CH]2[CH]3=[CH2]->[Fe]<-3<-2<-1(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"
    )
    idx = 3
    assert stereo.defined_stereo_label(mol, metals) == "C3:R"

    atom = mol.GetAtomWithIdx(idx)
    mirrored = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    }[atom.GetChiralTag()]
    atom.SetChiralTag(mirrored)

    assert stereo.defined_stereo_label(mol, metals) == "C3:S"


# ---------------------------------------------------------------------------------------------------------
# the expansion
# ---------------------------------------------------------------------------------------------------------


def test_undefined_point_centre_expands_with_cip_labels():
    assert _labels(_mol("CC(N)C(=O)O")) == ["C1:R", "C1:S"]


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C[C@H](F)Cl", {1: "R"}),
        ("OC(=O)[C@H]1CC[C@@H](CC1)O[C@@H](F)Cl", {3: "S", 6: "r", 10: "S"}),
    ],
)
def test_point_labels_use_accurate_cip_independent_of_legacy_perception(legacy, smiles, expected):
    previous = Chem.GetUseLegacyStereoPerception()
    try:
        Chem.SetUseLegacyStereoPerception(legacy)
        mol = _mol(smiles)
        assert stereo.point_stereo(stereo.defined_stereo_label(mol)) == expected
        assert Chem.GetUseLegacyStereoPerception() is legacy
    finally:
        Chem.SetUseLegacyStereoPerception(previous)


def test_stereo_selectors_are_concise_only_when_unambiguous():
    assert stereo.matches_stereo("C1:R", "C1:R")
    assert stereo.matches_stereo("C1:R", "1R")
    assert stereo.matches_stereo("C1:R", "C:R")
    assert stereo.matches_stereo("C1:R", "R")
    assert stereo.matches_stereo("C1:R,N3:S", "C:R")
    assert stereo.matches_stereo("C1:R,N3:S", "N:S")
    assert stereo.matches_stereo("C1:R,N3:S", "C:R,N:S")
    assert stereo.matches_stereo("C1:R,N3:S", "1R")
    assert stereo.matches_stereo("C3=C4:E", "C=C:E")
    assert stereo.matches_stereo("C3=C4:E", "E")
    assert stereo.matches_stereo("C5-C6:M", "M")
    with pytest.raises(ValueError, match=r"use one of.*C1:R.*C3:S"):
        stereo.matches_stereo("C1:R,C3:S", "C:R")
    with pytest.raises(ValueError, match=r"use one of.*C1:R.*N3:S"):
        stereo.matches_stereo("C1:R,N3:S", "R")
    with pytest.raises(ValueError, match=r"use one of.*C1=C2:E.*C3=C4:Z"):
        stereo.matches_stereo("C1=C2:E,C3=C4:Z", "E")


def test_undefined_alkene_expands_with_point_centres():
    labels = _labels(_mol("CC=CC(N)O"))
    assert len(labels) == 4
    assert sum(":E" in x for x in labels) == 2
    assert sum(":Z" in x for x in labels) == 2


def test_skipped_stereo_is_not_expanded_behind_the_filter():
    mol = _mol("CC=CC(N)O")
    point = next(iter(stereo.point_centres(mol)))

    points, n_points, total_points, _ = stereo.enumerate_unassigned(mol, skip_bonds=True)
    bonds, n_bonds, total_bonds, _ = stereo.enumerate_unassigned(mol, skip_points={point})

    assert (len(points), n_points, total_points) == (2, 1, 2)
    assert (len(bonds), n_bonds, total_bonds) == (2, 1, 2)
    assert mol.GetAtomWithIdx(point).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    assert all(variant.GetAtomWithIdx(point).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for variant, _ in bonds)


def test_meso_duplicate_is_dropped():
    variants, n_unassigned, total, _unresolved = stereo.enumerate_unassigned(_mol("CC(O)C(O)C"))
    assert (n_unassigned, total) == (2, 4)
    assert len(variants) == 3


def test_enumeration_preserves_atom_order():
    mol = _mol("CC=CC(N)O")
    original = [a.GetAtomicNum() for a in mol.GetAtoms()]
    for variant, _label in stereo.enumerate_unassigned(mol)[0]:
        assert [a.GetAtomicNum() for a in variant.GetAtoms()] == original


def test_stereo_cap_reports_uncapped_total():
    variants, n_unassigned, total, _unresolved = stereo.enumerate_unassigned(_mol("CC(N)C(=O)O"), cap=1)
    assert (n_unassigned, total) == (1, 2)
    assert len(variants) == 1


# ---------------------------------------------------------------------------------------------------------
# axial chirality
# ---------------------------------------------------------------------------------------------------------


def test_allene_axis_is_unresolved_without_question_label():
    variants, n_unassigned, _total, unresolved = stereo.enumerate_unassigned(_mol("CC(F)=C=C(F)C"))
    assert n_unassigned == 1
    assert unresolved == 1
    assert [label for _v, label in variants] == [""], "an unresolved centre must be dropped from the label"


def test_terminal_isothiocyanate_donor_is_not_a_stereo_axis():
    mol, metals = _with_metals("S=C=[N-]->[Fe+2]<-[N-]=C=S")

    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, exclude=metals)

    assert variants == [(mol, "")]
    assert (n_unassigned, total, unresolved) == (0, 1, 0)


def test_stated_native_atrop_axis_is_racemized_and_measured_from_3d():
    mol = _mol(_NATIVE_ATROP)
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, include="all")

    assert (len(variants), n_unassigned, total, unresolved) == (2, 1, 2, 0)
    assert {label.rsplit(":", 1)[-1] for _variant, label in variants} == {"M", "P"}

    embedded = Chem.AddHs(_mol(_NATIVE_ATROP))
    assert rdDistGeom.EmbedMolecule(embedded, randomSeed=7) == 0
    before = stereo.axis_stereo(stereo.stereo_from_3d(embedded))
    positions = embedded.GetConformer().GetPositions()
    positions[:, 0] *= -1
    embedded.GetConformer().SetPositions(positions)
    after = stereo.axis_stereo(stereo.stereo_from_3d(embedded))

    assert len(before) == 1
    pair, code = next(iter(before.items()))
    assert after[pair] == {"M": "P", "P": "M"}[code]


def test_atrop_axis_must_be_stated():
    assert _labels(_mol(_NATIVE_ATROP.split(" |", 1)[0])) == [""]


def test_explicit_hydrogen_leaves_an_aryl_imine_axis_free():
    """An explicit-H aryl-imine single bond is not enumerated as an atrop axis."""
    ligand = Chem.AddHs(_mol("Cc1cccc(C)c1C=NC"))
    variants, *_ = stereo.enumerate_unassigned(ligand, include="all")
    assert all(not stereo.axis_stereo(label) for _variant, label in variants)


def test_equivalent_ring_paths_do_not_define_an_atrop_axis():
    """A biaryl bond with one ring symmetric under either ortho path is not an atrop axis."""
    mol = _mol("Cc1cccc(C)c1-c1c(Br)cccc1I")
    variants, *_ = stereo.enumerate_unassigned(mol, include="all")
    assert all(not stereo.axis_stereo(label) for _variant, label in variants)


def test_unsubstituted_aza_biaryl_is_not_an_atrop_axis():
    """A chelate-locked, unsubstituted aza-biaryl bond is not enumerated as an atrop axis."""
    mol, metals = _with_metals(_AZA_BIARYL_CR)
    variants, *_ = stereo.enumerate_unassigned(mol, exclude=metals, include="all")
    assert all(not stereo.axis_stereo(label) for _variant, label in variants)


# ---------------------------------------------------------------------------------------------------------
# metal safety
# ---------------------------------------------------------------------------------------------------------


def test_metal_bound_chiral_phosphorus_survives_strip():
    mol, metals = _with_metals(_CHIRAL_P_PD)
    labels = _labels(mol, exclude=metals)
    assert len(labels) == 2
    assert len(set(labels)) == 2, "the two P-epimers must get distinct labels"


def test_dative_cap_participates_in_point_stereo_perception_from_3d():
    mol, metals = _with_metals("F[P](Cl)(Br)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(
        ((1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1), (-2, -1, -1), (-1, -2, -1), (-1, -1, -2))
    ):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    before = stereo.point_stereo(stereo.stereo_from_3d(mol, metals))
    embedded = mol.GetConformer()
    positions = embedded.GetPositions()
    positions[:, 0] *= -1
    embedded.SetPositions(positions)
    after = stereo.point_stereo(stereo.stereo_from_3d(mol, metals))

    assert before == {1: "S"}
    assert after == {1: "R"}


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C[N@H](->[Pd](Cl)(Cl)Cl)O", "N1:S"),
        ("C[P@H](->[Pt](Cl)(Cl)Cl)CC", "P1:S"),
        ("C[S@](->[Pt](Cl)(Cl)Cl)CC", "S1:S"),
        ("F[C@](Cl)(Br)[Pt](Cl)(Cl)Cl", "C1:S"),
    ],
)
def test_defined_donor_stereo_keeps_replacement_parity_and_bond_type(smiles, expected):
    mol = parse_smiles(smiles)
    assert stereo.defined_stereo_label(mol, metal_indices(mol)) == expected


def test_absolute_donor_label_reapplies_across_a_metal_priority_crossover():
    expected = {
        "Ni": Chem.ChiralType.CHI_TETRAHEDRAL_CW,  # Br outranks Ni
        "Pd": Chem.ChiralType.CHI_TETRAHEDRAL_CCW,  # Pd outranks Br
    }
    for metal, tag in expected.items():
        mol = parse_smiles(f"C[CH-](Br)->[{metal}+2](<-[Cl-])(<-[Cl-])<-[Cl-]")

        stereo.apply_point_stereo(mol, "C1:R", {1})

        assert mol.GetAtomWithIdx(1).GetChiralTag() == tag
        assert stereo.defined_stereo_label(mol, metal_indices(mol)) == "C1:R"


def test_ph3_donor_is_not_made_stereogenic_by_the_metal_cap():
    mol, metals = _with_metals("[PH3]->[Pt](Cl)(Cl)Cl")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
    assert (len(variants), n_unassigned, total, unresolved) == (1, 0, 1, 0)


def test_untagged_amine_donor_does_not_gain_point_stereo_from_its_metal_cap():
    mol = parse_smiles("C[NH](O)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 1), (0, 2, 1), (0, 0, 2))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    assert stereo.point_stereo(stereo.stereo_from_3d(mol, metal_indices(mol))) == {}


def test_chelated_amine_donor_enumerates_both_configurations():
    mol, metals = _with_metals("C[NH]1CC[O-]->[Pd+2](<-[Cl-])(<-[Cl-])<-1")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, exclude=metals)

    assert {label for _variant, label in variants} == {"N1:R", "N1:S"}
    assert (n_unassigned, total, unresolved) == (1, 2, 0)


def test_bound_amine_and_alkyl_are_coordination_locked_but_phosphine_is_not():
    """Each donor is a stereocentre of its complex; only the phosphine P is one without the metal as well."""
    for smiles, locked in (
        ("C[NH]1CC[O-]->[Pd+2](<-[Cl-])(<-[Cl-])<-1", ["N"]),
        ("[CH3-]->[Pd+2](<-[Cl-])<-[CH-](C)CC", ["C"]),
        ("C[P](CC)(c1ccccc1)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]", []),
    ):
        mol, metals = _with_metals(smiles)
        assert stereo.point_centres(mol, metals), smiles
        symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in stereo.coordination_locked_centres(mol, metals))
        assert symbols == locked, smiles


def test_bound_amine_hands_merge_only_where_the_complex_is_symmetric():
    """Two bound-N hand pairs are one variant only by the complex's own symmetry, not by the metal-cut graph's:
    all four pairs stay when one carboxylate binds, and the two mixed pairs of a symmetric macrocycle merge.
    """
    one_arm, metals = _with_metals(ONE_ARM_BOUND_PT)
    macrocycle, macro_metals = _with_metals("C1C[NH]2->[Zn+2]34(<-[Cl-])<-[NH](C1)CC[NH]->3CC[NH]->4CCC2")

    assert len(_labels(one_arm, exclude=metals)) == 4
    assert len(_labels(macrocycle, exclude=macro_metals)) == 3


def test_equivalent_chelate_arms_do_not_create_donor_point_stereo():
    mol, metals = _with_metals("CN12->[Rh+](<-[I-])(<-[C-]#[O+])<-P3(C)CN(CN(C1)C3)C2")
    donors = {
        atom.GetIdx() for atom in mol.GetAtoms() if any(neighbor.GetIdx() in metals for neighbor in atom.GetNeighbors())
    }

    assert donors.isdisjoint(stereo.point_centres(mol, metals))
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
    assert variants == [(mol, "")]
    assert (n_unassigned, total, unresolved) == (0, 1, 0)


@pytest.mark.parametrize(
    ("smiles", "unresolved"),
    [
        ("Cc1ccc2c(c1)CN1Cc3cc(C)ccc3N(C1)C2", 1),
        ("ClN1CC1C", 1),
        ("COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", 0),
    ],
    ids=[
        "troger-base-chiral-cage",
        "n-chloro-2-methylaziridine-three-ring",
        "quinine-carbon-bridgehead-holds-the-cage",
    ],
)
def test_stable_nitrogen_is_reported_only_when_its_hand_is_free(smiles, unresolved):
    # RDKit holds a bridgehead or three-ring N but cannot measure a three-carrier N from 3D, so a free hand is
    # reported. A bridgehead N's hand is the cage's, which the quinuclidine's carbon bridgehead already carries.
    assert stereo.enumerate_unassigned(_mol(smiles))[3] == unresolved


@pytest.mark.parametrize("symbol", ["P", "As"])
def test_three_coordinate_pnictogen_is_reported_unresolved_instead_of_enumerated(symbol):
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(_mol(f"F[{symbol}](Cl)C"))

    assert [label for _variant, label in variants] == [""]
    assert (n_unassigned, total, unresolved) == (0, 1, 1)


def test_three_coordinate_sulfur_remains_measurable_from_3d():
    mol = _mol("C[S](=O)Cl")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    assert set(stereo.point_stereo(stereo.stereo_from_3d(mol))) == {1}


@pytest.mark.parametrize("smiles", ["[C@](F)(Cl)(Br)I", "[P@](F)(Cl)(Br)->[Pd+2]"])
@pytest.mark.parametrize("shape", ["inside", "outside", "on_face", "flat_carriers"])
def test_measured_point_requires_centre_inside_carriers_independent_of_bond_order(smiles, shape):
    source = _mol(smiles)
    positions = np.array([(0, 0, 0), (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float)
    if shape == "outside":
        positions[0] = (2, 2, 2)
    elif shape == "on_face":
        positions *= 3
        positions[0] = (1, 1, -1)
    elif shape == "flat_carriers":
        positions[1:, 2] = 0
        positions[0, 2] = 1
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in source.GetBonds()]
    labels = []
    for order in itertools.permutations(bonds):
        work = Chem.RWMol(source)
        for i, j, _kind in bonds:
            work.RemoveBond(i, j)
        for i, j, kind in order:
            work.AddBond(i, j, kind)
        mol = work.GetMol()
        mol.UpdatePropertyCache(strict=False)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetPositions(positions)
        mol.AddConformer(conf)
        before = mol.GetAtomWithIdx(0).GetChiralTag()
        labels.append(stereo.point_stereo(stereo.stereo_from_3d(mol, metal_indices(mol))))
        assert mol.GetAtomWithIdx(0).GetChiralTag() == before
        assert stereo.point_stereo(stereo.stereo_from_3d(mol, metal_indices(mol), apply=True)) == labels[-1]
        if shape == "inside":
            assert set(labels[-1]) == {0}
        else:
            assert labels[-1] == {}
            assert mol.GetAtomWithIdx(0).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    assert all(label == labels[0] for label in labels)


def test_aromatic_eta1_donor_is_not_made_stereogenic_by_stale_hybridization():
    """An aromatic eta1 donor keeps a stale SP3 hybridization tag from becoming a point stereocentre."""
    mol, metals = _with_metals("Cc1cc[cH-](c1)->[Ru+]")
    donor = next(
        atom
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and any(n.GetIdx() in metals for n in atom.GetNeighbors())
    )
    donor.SetHybridization(Chem.HybridizationType.SP3)

    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol, exclude=metals)

    assert variants == [(mol, "")]
    assert (n_unassigned, total, unresolved) == (0, 1, 0)


def test_coordination_locked_alkene_is_not_enumerated():
    mol, metals = _with_metals(_ALPHA_DIIMINE_NI)
    assert stereo.coordination_locked_double_bonds(mol, metals), "the metal-closed imine was not detected"
    variants, _n, _total, _unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
    assert len(variants) == 2, "only the real point stereocentre should expand, not the locked imines"


def test_coordination_locked_explicit_ez_is_not_kept_on_the_variant():
    mol, metals = _with_metals(r"C/C1=N/[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1")
    (variant, _label), *_ = stereo.enumerate_unassigned(mol, exclude=metals)[0]
    pair = next(iter(stereo.coordination_locked_double_bonds(mol, metals)))

    assert variant.GetBondBetweenAtoms(*pair).GetStereo() == Chem.BondStereo.STEREONONE


def test_eta2_alkene_is_not_coordination_locked():
    mol, metals = _with_metals(_ETA2_PT)
    assert stereo.coordination_locked_double_bonds(mol, metals) == set()
    assert set(_labels(mol, exclude=metals)) == {"C1=C2:E", "C1=C2:Z"}


def test_metallacyclopentene_central_double_bond_is_the_diene_class_not_ez():
    """Drawn as a sigma2,pi metallacyclopentene, a diene's central C=C is its s-cis/s-trans class, not E/Z."""
    mol, metals = _with_metals("C[C]1->2=[C]->3(C)[CH2-]->[Zr+2]23<-[CH2-]1")
    assert frozenset((1, 2)) in stereo.coordination_locked_double_bonds(mol, metals)
    assert stereo.unassigned_centres(mol, exclude=metals) == []


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "C1=C2:Z"),
        (r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "C1=C2:E"),
    ],
)
def test_eta2_alkene_keeps_explicit_ez_after_metal_strip(smiles, expected):
    mol, metals = _with_metals(smiles)
    assert stereo.defined_stereo_label(mol, metals) == expected


def test_pendant_alkene_remains_unlocked_by_metal():
    mol, metals = _with_metals("CC=CC[NH2]->[Ni+2](<-[O-]C(=O)C)<-[NH2]CC=CC")
    assert stereo.coordination_locked_double_bonds(mol, metals) == set()


def test_flexible_metal_closed_alkene_keeps_ligand_side_ez():
    isomer = rx.metal(r"N1CC/C=C/CC[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-1", "square_planar")[0]
    mol = rx.embed(isomer, n=1, seed=7).mol
    metals = metal_indices(mol)
    locked = stereo.coordination_locked_double_bonds(mol, metals)

    assert locked == set()
    pair = next(iter(stereo.bond_stereo(stereo.defined_stereo_label(mol, metals))))
    bond = mol.GetBondBetweenAtoms(*pair)
    bond.SetStereo(Chem.BondStereo.STEREOE if bond.GetStereo() == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ)
    assert stereo.defined_stereo_label(mol, metals) != stereo.stereo_from_3d(mol, metals)


def test_inferred_ez_drops_a_resonance_dependent_cip_path():
    stated = _mol(r"CN(C)/C(C)=C1/C=CC=C[CH-]1")
    rdDepictor.Compute2DCoords(stated)
    stated.GetConformer().Set3D(True)

    assert stereo.bond_stereo(stereo.stereo_from_3d(stated))

    forms = Chem.ResonanceMolSupplier(stated, maxStructs=3)
    for index in (0, 2):
        inferred = Chem.Mol(forms[index])
        Chem.RemoveStereochemistry(inferred)
        label = stereo.stereo_from_3d(inferred, apply=True)

        assert stereo.bond_stereo(label) == {}
        assert all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in inferred.GetBonds())


def test_apply_inferred_ez_uses_an_independent_measurement_graph():
    mol = Chem.AddHs(Chem.MolFromSmiles("F/C=C/F"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
    Chem.RemoveStereochemistry(mol)

    label = stereo.stereo_from_3d(mol, apply=True)

    assert stereo.bond_stereo(label)
    assert stereo.defined_stereo_label(mol) == label


def test_inferred_ez_drops_when_its_bond_order_moves_in_resonance():
    mol = _mol(r"C/C=C/[CH2-]")
    rdDepictor.Compute2DCoords(mol)
    mol.GetConformer().Set3D(True)
    Chem.RemoveStereochemistry(mol)

    assert stereo.stereo_from_3d(mol) == ""


def test_coordinate_free_ez_is_enumerated_only_when_resonance_stable(capfd):
    unstable = stereo.enumerate_unassigned(_mol("CC=C[CH2-]"))
    stable = stereo.enumerate_unassigned(_mol("CC=CC"))
    priority = _mol("CN(C)C(C)=C1C=CC=C[CH-]1")
    reversed_priority = Chem.RenumberAtoms(priority, list(reversed(range(priority.GetNumAtoms()))))

    assert ([label for _variant, label in unstable[0]], unstable[1:]) == ([""], (0, 1, 0))
    assert ({label for _variant, label in stable[0]}, stable[1:]) == ({"C1=C2:E", "C1=C2:Z"}, (1, 2, 0))
    for candidate in (priority, reversed_priority):
        variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(candidate)
        assert ([label for _variant, label in variants], n_unassigned, total, unresolved) == ([""], 0, 1, 0)
    assert "Pre-condition Violation" not in capfd.readouterr().err


def test_inferred_ez_abstains_when_resonance_search_hits_its_cap(monkeypatch, caplog):
    monkeypatch.setattr(stereo, "_RESONANCE_EZ_CAP", 0)
    for smiles, kept in ((r"C/C=C/C", True), (r"C/C=C/C=O", False)):  # an isolated alkene; one in a conjugated group
        mol = _mol(smiles)
        bond = mol.GetBondBetweenAtoms(1, 2).GetIdx()
        for structural in (False, True):
            caplog.clear()
            assert stereo._resonance_stable_ez(mol, [bond], set(), structural=structural) == ([bond] if kept else [])
            assert ("1=2 left unassigned" in caplog.text) is not kept
            assert stereo._resonance_stable_ez(mol, [bond], {bond}, structural=structural) == [bond]


def test_propenyl_ez_is_enumerated_beside_six_carboxylates():
    # Six independent carboxylates give 64 resonance forms, past the search cap; the isolated C=C is in none.
    carboxylates = "C(C(=O)[O-])C(C(=O)[O-])C(C(=O)[O-])C(C(=O)[O-])C(C(=O)[O-])C(=O)[O-]"
    mol = Chem.AddHs(_mol("CC=CCC" + carboxylates))

    assert (1, 2) in stereo.unassigned_centres(mol)


def test_coordinate_free_donor_imine_in_a_large_chelate_keeps_ez():
    mol = parse_smiles(r"C/N1=C(/C)CCCCCC[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-1")
    labels = [iso.stereo_label for iso in rx.metal(mol, "square_planar")]

    assert labels
    assert all(label == "N1=C2:Z" for label in labels)


def test_monodentate_donor_imine_keeps_ez():
    mol, metals = _with_metals("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")

    assert stereo.coordination_locked_double_bonds(mol, metals) == set()
    assert _labels(mol, exclude=metals) == ["C1=N2:E", "C1=N2:Z"]


def test_stereo_references_survive_metal_bond_removal_and_renumbering():
    mol = parse_smiles(r"[H]/[N](=C(\C))->[Pt+4](<-[Cl-])(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[N](/[H])=C(/C)")
    for candidate in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
        label = stereo.defined_stereo_label(candidate, set(metal_indices(candidate)))
        assert sorted(stereo.bond_stereo(label).values()) == ["E", "E"]


def test_donor_cap_is_a_stereo_proxy_and_does_not_overcap_a_double_bond(capfd):
    """A metal donor cap must not add a point stereocentre on top of the donor's own double-bond E/Z."""
    imine, metals = _with_metals("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(imine, exclude=metals)
    assert {label for _variant, label in variants} == {"C1=N2:E", "C1=N2:Z"}
    assert (n_unassigned, total, unresolved) == (1, 2, 0)

    referenced, metals = _with_metals("[H][C](=O)(P)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    referenced.GetAtomWithIdx(1).SetHybridization(Chem.HybridizationType.SP3)  # stale input perception
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(referenced, exclude=metals)
    assert variants == [(referenced, "")]
    assert (n_unassigned, total, unresolved) == (0, 1, 0)

    # two identical H already rule out a point centre; capping this donor anyway overvalences it
    bis_amine, metals = _with_metals("CC=CC[NH2]->[Ni+2](<-[O-]C(=O)C)<-[NH2]CC=CC")
    capfd.readouterr()  # discard warnings from the cases above
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(bis_amine, exclude=metals)
    assert (n_unassigned, total, unresolved) == (2, 4, 0)
    assert "valence" not in capfd.readouterr().err


def test_metal_strip_and_graft_are_inverse():
    for smiles in ("Cl[Pd](Cl)(Cl)<-[P@](C)(CC)C(C)(N)O", "[P@](C)(CC)(C(C)(N)O)->[Pd](Cl)(Cl)Cl"):
        mol, metals = _with_metals(smiles)
        donor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P")
        declared = mol.GetAtomWithIdx(donor).GetChiralTag()
        variants, n_unassigned, _total, _unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
        assert n_unassigned == 1, f"{smiles}: the carbon centre was not enumerated, so this asserts nothing"
        for vmol, _label in variants:
            assert vmol.GetAtomWithIdx(donor).GetChiralTag() == declared, smiles


def test_multimetal_donor_enumeration_replays_sequential_parity():
    mol = parse_smiles("C[N](->[Pd])(->[Pt])O")
    metals = set(metal_indices(mol))
    variants, *_ = stereo.enumerate_unassigned(mol, exclude=metals)

    assert {label for _, label in variants} == {"N1:R", "N1:S"}
    assert all(stereo.defined_stereo_label(variant, metals) == label for variant, label in variants)


def test_same_element_metal_bridge_uses_the_full_ligand_spheres():
    asymmetric = parse_smiles("C[N](->[Pd](Cl)(Cl)Cl)(->[Pd](Br)(Br)Br)O")
    metals = set(metal_indices(asymmetric))
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(asymmetric, exclude=metals)

    assert {label for _, label in variants} == {"N1:R", "N1:S"}
    assert (n_unassigned, total, unresolved) == (1, 2, 0)
    assert all(stereo.defined_stereo_label(variant, metals) == label for variant, label in variants)

    symmetric = parse_smiles("C[N](->[Pd](Cl)(Cl)Cl)(->[Pd](Cl)(Cl)Cl)O")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(
        symmetric, exclude=set(metal_indices(symmetric))
    )
    assert [label for _, label in variants] == [""]
    assert (n_unassigned, total, unresolved) == (0, 1, 0)


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [3, 4, 2, 1, 0]])
def test_cx_encoded_e_applies_as_e_in_any_atom_order(order):
    mol = Chem.RenumberAtoms(_mol("CC=C(Cl)Br |atomProp:1._rxEZ0.E:2._rxEZ0.E|"), order)

    stereo.apply_encoded_bond_stereo(mol)

    bond = mol.GetBondBetweenAtoms(order.index(1), order.index(2))
    assert stereo.bond_stereo_code(mol, bond.GetIdx()) == "E"
