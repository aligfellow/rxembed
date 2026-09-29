"""Test graph stereoisomer enumeration before embedding."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom

from rxembed import stereo
from rxembed.metal_core import metal_indices
from rxembed.metal_smiles import parse_smiles

# an alpha-diimine-style chelate: both imine C=N sit in the 5-membered ring the metal closes
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


# ---------------------------------------------------------------------------------------------------------
# the expansion
# ---------------------------------------------------------------------------------------------------------


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


def test_skipped_stereo_is_not_expanded_behind_the_filter():
    mol = _mol("CC=CC(N)O")
    point = next(iter(stereo.point_centres(mol)))

    points, n_points, total_points, _ = stereo.enumerate_unassigned(mol, skip_bonds=True)
    bonds, n_bonds, total_bonds, _ = stereo.enumerate_unassigned(mol, skip_points={point})

    assert (len(points), n_points, total_points) == (2, 1, 2)
    assert (len(bonds), n_bonds, total_bonds) == (2, 1, 2)
    assert mol.GetAtomWithIdx(point).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    assert all(variant.GetAtomWithIdx(point).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for variant, _ in bonds)


# ---------------------------------------------------------------------------------------------------------
# axial chirality
# ---------------------------------------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------------------------------------
# metal safety
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C[P@H](->[Pt](Cl)(Cl)Cl)CC", "P1:S"),
    ],
)
def test_defined_donor_stereo_keeps_replacement_parity_and_bond_type(smiles, expected):
    mol = parse_smiles(smiles)
    assert stereo.defined_stereo_label(mol, metal_indices(mol)) == expected


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
        ("COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", 0),
    ],
    ids=[
        "quinine-carbon-bridgehead-holds-the-cage",
    ],
)
def test_stable_nitrogen_is_reported_only_when_its_hand_is_free(smiles, unresolved):
    # RDKit holds a bridgehead or three-ring N but cannot measure a three-carrier N from 3D, so a free hand is
    # reported. A bridgehead N's hand is the cage's, which the quinuclidine's carbon bridgehead already carries.
    assert stereo.enumerate_unassigned(_mol(smiles))[3] == unresolved


@pytest.mark.parametrize("smiles", ["[C@](F)(Cl)(Br)I"])
@pytest.mark.parametrize("shape", ["flat_carriers"])
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


def test_coordination_locked_explicit_ez_is_not_kept_on_the_variant():
    mol, metals = _with_metals(r"C/C1=N/[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1")
    (variant, _label), *_ = stereo.enumerate_unassigned(mol, exclude=metals)[0]
    pair = next(iter(stereo.coordination_locked_double_bonds(mol, metals)))

    assert variant.GetBondBetweenAtoms(*pair).GetStereo() == Chem.BondStereo.STEREONONE


def test_metallacyclopentene_central_double_bond_is_the_diene_class_not_ez():
    """Drawn as a sigma2,pi metallacyclopentene, a diene's central C=C is its s-cis/s-trans class, not E/Z."""
    mol, metals = _with_metals("C[C]1->2=[C]->3(C)[CH2-]->[Zr+2]23<-[CH2-]1")
    assert frozenset((1, 2)) in stereo.coordination_locked_double_bonds(mol, metals)
    assert stereo.unassigned_centres(mol, exclude=metals) == []


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


def test_monodentate_donor_imine_keeps_ez():
    mol, metals = _with_metals("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")

    assert stereo.coordination_locked_double_bonds(mol, metals) == set()
    assert _labels(mol, exclude=metals) == ["C1=N2:E", "C1=N2:Z"]


def test_stereo_references_survive_metal_bond_removal_and_renumbering():
    mol = parse_smiles(r"[H]/[N](=C(\C))->[Pt+4](<-[Cl-])(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[N](/[H])=C(/C)")
    for candidate in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
        label = stereo.defined_stereo_label(candidate, set(metal_indices(candidate)))
        assert sorted(stereo.bond_stereo(label).values()) == ["E", "E"]


@pytest.mark.parametrize("order", [[4, 3, 2, 1, 0]])
def test_cx_encoded_e_applies_as_e_in_any_atom_order(order):
    mol = Chem.RenumberAtoms(_mol("CC=C(Cl)Br |atomProp:1._rxEZ0.E:2._rxEZ0.E|"), order)

    stereo.apply_encoded_bond_stereo(mol)

    bond = mol.GetBondBetweenAtoms(order.index(1), order.index(2))
    assert stereo.bond_stereo_code(mol, bond.GetIdx()) == "E"
