"""Test graph stereoisomer enumeration before embedding."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import stereo
from rxembed.metal_core import metal_indices
from rxembed.metal_smiles import parse_smiles

_CHIRAL_P_PD = "C[P](CC)(c1ccccc1)[Pd](Cl)(Cl)Cl"
# an alpha-diimine-style chelate: both imine C=N sit in the 5-membered ring the metal closes
_ALPHA_DIIMINE_NI = (
    "O=C1[O-]->[Ni+2]2(<-[N](=C3C(=[N]->2c2cccc4ccccc24)c2cccc4cccc3c24)c2cccc3ccccc23)<-[N-](c2ccccc2)C1c1ccccc1"
)
_ETA2_PT = "C[CH]1=[CH](F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
_NATIVE_ATROP = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"


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


# ---------------------------------------------------------------------------------------------------------
# the expansion
# ---------------------------------------------------------------------------------------------------------


def test_undefined_point_centre_expands_with_cip_labels():
    assert _labels(_mol("CC(N)C(=O)O")) == ["C1:R", "C1:S"]


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
    assert n_unassigned == 2
    assert unresolved == 2
    assert [label for _v, label in variants] == [""], "an unresolved centre must be dropped from the label"


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


# ---------------------------------------------------------------------------------------------------------
# metal safety
# ---------------------------------------------------------------------------------------------------------


def test_metal_bound_chiral_phosphorus_survives_strip():
    mol, metals = _with_metals(_CHIRAL_P_PD)
    labels = _labels(mol, exclude=metals)
    assert len(labels) == 2
    assert len(set(labels)) == 2, "the two P-epimers must get distinct labels"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C[N@H](->[Pd](Cl)(Cl)Cl)O", "N1:S"),
        ("C[P@H](->[Pt](Cl)(Cl)Cl)CC", "P1:S"),
        ("C[S@](->[Pt](Cl)(Cl)Cl)CC", "S1:R"),
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


def test_coordination_locked_alkene_is_not_enumerated():
    mol, metals = _with_metals(_ALPHA_DIIMINE_NI)
    assert stereo._coordination_locked_double_bonds(mol, metals), "the metal-closed imine was not detected"
    variants, _n, _total, _unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
    assert len(variants) == 2, "only the real point stereocentre should expand, not the locked imines"


def test_eta2_alkene_is_not_coordination_locked():
    mol, metals = _with_metals(_ETA2_PT)
    assert stereo._coordination_locked_double_bonds(mol, metals) == set()
    assert set(_labels(mol, exclude=metals)) == {"C1=C2:E", "C1=C2:Z"}


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
    assert stereo._coordination_locked_double_bonds(mol, metals) == set()


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
