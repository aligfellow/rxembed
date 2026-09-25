"""`pipeline/stereo_check.py`: the chirality fingerprint and the preserve/invert gate over it."""

from collections import Counter
from importlib.util import find_spec

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

import rxembed as rx
from rxembed import core, metal_core
from rxembed.pipeline.stereo_check import (
    _independent_summary,
    _native_signature,
    mismatch,
    signature,
)
from tests.metal_fixtures import ferrocene

_PLANAR = {"planar": Counter({"Rₚ": 1})}  # a metallocene's planar chirality: what the filter is *for*
_FLIPPED_PLANAR = {"planar": Counter({"Sₚ": 1})}


def _reference_conformer(smiles, seed=1, optimize=True):
    """Return a Mol with one conformer from plain ETKDG (+ MMFF): a geometry rxembed had no hand in making."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


# --- mismatch: the four modes, on one inverted element ----------------------------------------------


@pytest.mark.parametrize(
    ("mode", "kept", "flipped"),
    [("preserve", True, False), ("free", True, True), ("invert", False, True), ("all", True, False)],
    ids=["preserve", "free", "invert", "all"],
)
def test_stereo_modes_accept_expected_hands(mode, kept, flipped):
    assert (mismatch(_PLANAR, _PLANAR, mode) is None) is kept
    assert (mismatch(_FLIPPED_PLANAR, _PLANAR, mode) is None) is flipped


def test_preserve_holds_only_nongraph_stereo():
    left = _native_signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False))
    right = _native_signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False))

    assert mismatch(right, left, "preserve") is None
    assert mismatch(right, left, "all") is not None


def test_native_stereo_normalizes_metal_notation_before_comparing():
    raw = ferrocene()
    canonical = metal_core.canonical_metal_graph(raw)

    assert _native_signature(raw) == _native_signature(canonical)


def test_graph_mismatch_is_not_reported_as_axial_stereo():
    left = _native_signature(_reference_conformer("CCO"))
    right = _native_signature(_reference_conformer("COC"))

    assert mismatch(right, left) == "molecular graphs differ after stereo is removed"


def test_preserve_ignores_fingerprint_multiplicity():
    for sig in ({}, {"planar": Counter({"Rₚ": 1})}, {"planar": Counter({"Rₚ": 2})}):
        assert mismatch(sig, _PLANAR, "preserve") is None


def test_preserve_accepts_a_reference_with_both_planar_hands():
    both = {"planar": Counter({"Rₚ": 1, "Sₚ": 1})}

    assert mismatch(both, both, "preserve") is None
    assert mismatch(both, _PLANAR, "preserve") is not None


def test_symmetric_native_hands_remain_attached_to_the_whole_graph():
    left = _reference_conformer("C[C@H]1[C@H](C)[C@H](C)[C@H](C)[C@H](C)[C@@H]1C", optimize=False)
    right = _reference_conformer("C[C@H]1[C@H](C)[C@H](C)[C@H](C)[C@@H](C)[C@@H]1C", optimize=False)
    reference, perceived = _native_signature(left), _native_signature(right)

    assert reference["point"] == perceived["point"]  # the old orbit-count representation collided here
    assert mismatch(perceived, reference, "all") is not None
    assert mismatch(perceived, reference, {"point": "free", "default": "free"}) is None
    reordered = Chem.RenumberAtoms(left, list(reversed(range(left.GetNumAtoms()))))
    assert _native_signature(reordered) == reference


def test_mismatch_names_the_stereo_kind_and_hands():
    left = _native_signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False))
    right = _native_signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False))

    detail = mismatch(right, left, "all")
    assert detail is not None
    assert detail.startswith("point ")


def test_all_reference_kinds_must_pass():
    left = _native_signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False))
    right = _native_signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False))
    ref = {**_PLANAR, **left}
    assert mismatch({**_FLIPPED_PLANAR, **left}, ref, "all") is not None
    assert mismatch({**_PLANAR, **right}, ref, "all") is not None

    sig = {**_FLIPPED_PLANAR, **right}
    assert mismatch(sig, ref, {"planar": "free", "point": "preserve"}) is not None
    assert mismatch(sig, ref, {"planar": "free", "default": "free"}) is None


def test_native_ez_can_be_inverted_without_holding_other_kinds():
    trans = _native_signature(_reference_conformer("F/C=C/F", optimize=False))
    cis = _native_signature(_reference_conformer("F/C=C\\F", optimize=False))

    assert mismatch(cis, trans, {"ez": "preserve", "default": "free"}) is not None
    assert mismatch(cis, trans, {"ez": "invert", "default": "free"}) is None


def test_native_point_can_be_inverted_without_holding_other_kinds():
    right = _native_signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False))
    left = _native_signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False))

    assert mismatch(left, right, {"point": "invert", "default": "free"}) is None
    assert mismatch(left, right, {"point": "preserve", "default": "free"}) is not None


# --- signature: read the handedness back out of the coordinates -------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_two_enantiomers_get_opposite_point_labels():
    left = signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False), 0)
    right = signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False), 0)
    assert set(left.get("point", ())), f"no point label read from the R conformer: {left}"
    assert set(right.get("point", ())), f"no point label read from the S conformer: {right}"
    assert set(left["point"]) != set(right["point"])
    assert mismatch(right, left, "all") is not None  # ...and the gate sees the inversion


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_native_atrop_identity_survives_atom_order_and_detects_a_mirror():
    mol = _reference_conformer("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|", optimize=False)
    reference = signature(mol)
    assert reference.get("axial")

    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert signature(reordered) == reference

    mirrored = Chem.Mol(mol)
    conf = mirrored.GetConformer()
    for atom in range(mirrored.GetNumAtoms()):
        point = conf.GetAtomPosition(atom)
        conf.SetAtomPosition(atom, (-point.x, point.y, point.z))
    observed = signature(mirrored)
    assert observed.get("axial")
    assert mismatch(observed, reference) is not None
    assert mismatch(observed, reference, {"axial": "invert", "default": "free"}) is None


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_equivalent_multiple_atrop_axes_have_one_joint_atom_order_invariant_key():
    smiles = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"
    left = Chem.MolFromSmiles(smiles)
    right = Chem.MolFromSmiles(smiles.replace("wU", "wD"))
    assert rdDistGeom.EmbedMolecule(left, randomSeed=1) == 0
    assert rdDistGeom.EmbedMolecule(right, randomSeed=2) == 0
    mol = Chem.CombineMols(left, right)
    order = [
        11,
        15,
        10,
        0,
        5,
        28,
        21,
        17,
        29,
        20,
        18,
        6,
        26,
        3,
        8,
        23,
        19,
        14,
        4,
        25,
        27,
        13,
        12,
        1,
        7,
        9,
        16,
        22,
        2,
        24,
    ]

    assert _native_signature(Chem.RenumberAtoms(mol, order)) == _native_signature(mol)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_atrop_markers_do_not_overflow_at_the_rdkit_isotope_limit():
    mol = _reference_conformer("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|", optimize=False)
    mol.GetAtomWithIdx(0).SetIsotope(65535)
    reference = _native_signature(mol)
    mirrored = Chem.Mol(mol)
    positions = mirrored.GetConformer().GetPositions()
    positions[:, 0] *= -1
    mirrored.GetConformer().SetPositions(positions)
    observed = _native_signature(mirrored)

    assert mismatch(observed, reference, {"axial": "free", "default": "free"}) is None
    assert mismatch(observed, reference, {"axial": "invert", "default": "free"}) is None


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_bis_oxime_pyridyl_axes_disappear_without_the_coordinating_metal():
    import xyzgraph
    from xyzgraph.stereo import annotate_stereo

    complex_smi = r"O/[N]1=C/c2cccc[n]2->[Cd+2]<-12(<-[I-])(<-[I-])<-[N](/O)=C\c1cccc[n]->21"
    mol = core.embed(rx.metal(complex_smi, "octahedral")[0], n=1, seed=7).mol
    conf = mol.GetConformer()
    atoms = [(atom.GetSymbol(), tuple(conf.GetAtomPosition(atom.GetIdx()))) for atom in mol.GetAtoms()]
    raw = annotate_stereo(xyzgraph.build_graph(atoms, kekule=True))

    assert raw["axial"]
    assert "axial" not in signature(mol)


def test_mirror_symmetric_haptic_face_does_not_gain_planar_chirality_from_the_metal():
    mol = ferrocene()
    metal = metal_core.metal_indices(mol)[0]
    donors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
    face = next(site for site in metal_core.haptic_sites(mol, donors) if len(site) > 1)
    summary = {"planar": [{"label": "Sₚ", "ring": list(face)}]}

    assert _independent_summary(mol, summary)["planar"] == []
