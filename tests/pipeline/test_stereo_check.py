"""`pipeline/stereo_check.py`: the chirality fingerprint and the preserve/invert gate over it."""

from collections import Counter
from importlib.util import find_spec

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed import metal_core
from rxembed.pipeline.stereo_check import (
    _independent_summary,
    _native_signature,
    mismatch,
    signature,
)
from tests.metal_fixtures import ferrocene

_PLANAR = {"planar": Counter({"Rₚ": 1})}  # a metallocene's planar chirality: what the filter is *for*
_FLIPPED_PLANAR = {"planar": Counter({"Sₚ": 1})}


def test_mirror_symmetric_haptic_face_does_not_gain_planar_chirality_from_the_metal():
    mol = ferrocene()
    metal = metal_core.metal_indices(mol)[0]
    donors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
    face = next(site for site in metal_core.haptic_sites(mol, donors) if len(site) > 1)
    summary = {"planar": [{"label": "Sₚ", "ring": list(face)}]}

    assert _independent_summary(mol, summary)["planar"] == []


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
    [("preserve", True, False)],
    ids=["preserve"],
)
def test_stereo_modes_accept_expected_hands(mode, kept, flipped):
    assert (mismatch(_PLANAR, _PLANAR, mode) is None) is kept
    assert (mismatch(_FLIPPED_PLANAR, _PLANAR, mode) is None) is flipped


def test_graph_mismatch_is_not_reported_as_axial_stereo():
    left = _native_signature(_reference_conformer("CCO"))
    right = _native_signature(_reference_conformer("COC"))

    detail = mismatch(right, left)
    assert detail is not None
    assert "graph" in detail
    assert "axial" not in detail


def test_symmetric_native_hands_remain_attached_to_the_whole_graph():
    left = _reference_conformer("C[C@H]1[C@H](C)[C@H](C)[C@H](C)[C@H](C)[C@@H]1C", optimize=False)
    right = _reference_conformer("C[C@H]1[C@H](C)[C@H](C)[C@H](C)[C@@H](C)[C@@H]1C", optimize=False)
    reference, perceived = _native_signature(left), _native_signature(right)

    assert reference["point"] == perceived["point"]  # the old orbit-count representation collided here
    assert mismatch(perceived, reference, "all") is not None
    assert mismatch(perceived, reference, {"point": "free", "default": "free"}) is None
    reordered = Chem.RenumberAtoms(left, list(reversed(range(left.GetNumAtoms()))))
    assert _native_signature(reordered) == reference


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


@pytest.mark.parametrize(("charge", "expected"), [(None, -1), (0, 0), (-2, -2)])
def test_signature_uses_declared_charge(monkeypatch, charge, expected):
    xyzgraph = pytest.importorskip("xyzgraph")
    from xyzgraph import stereo

    mol = Chem.MolFromSmiles("[Cl-]")
    mol.AddConformer(Chem.Conformer(1))
    seen = []
    monkeypatch.setattr(xyzgraph, "build_graph", lambda *args, **kwargs: seen.append(kwargs["charge"]))
    monkeypatch.setattr(stereo, "annotate_stereo", lambda graph: {})
    if charge is None:
        signature(mol)
    else:
        signature(mol, charge=charge)
    assert seen == [expected]
