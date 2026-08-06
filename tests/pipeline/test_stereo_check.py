"""`pipeline/stereo_check.py`: the chirality fingerprint and the preserve/invert gate over it."""

from collections import Counter
from importlib.util import find_spec

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed.pipeline.stereo_check import satisfies_spec, signature

_PLANAR = {"planar": Counter({"Rₚ": 1})}  # a metallocene's planar chirality: what the filter is *for*
_POINT = {"point": Counter({"R": 1})}  # a graph stereocentre: the embed's own job, free by default
_FLIPPED_PLANAR = {"planar": Counter({"Sₚ": 1})}


def _reference_conformer(smiles, seed=1, optimize=True):
    """Return a Mol with one conformer from plain ETKDG (+ MMFF): a geometry rxembed had no hand in making."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


# --- satisfies_spec: the four modes, on one inverted element ----------------------------------------------


@pytest.mark.parametrize(
    ("mode", "kept", "flipped"),
    [("preserve", True, False), ("free", True, True), ("invert", False, True), ("all", True, False)],
    ids=["preserve", "free", "invert", "all"],
)
def test_stereo_modes_accept_expected_hands(mode, kept, flipped):
    assert satisfies_spec(_PLANAR, _PLANAR, mode) is kept
    assert satisfies_spec(_FLIPPED_PLANAR, _PLANAR, mode) is flipped


def test_preserve_holds_only_nongraph_stereo():
    flipped = {"point": Counter({"S": 1})}
    assert satisfies_spec(flipped, _POINT, "preserve")
    assert not satisfies_spec(flipped, _POINT, "all")


def test_preserve_ignores_fingerprint_multiplicity():
    for sig in ({}, {"planar": Counter({"Rₚ": 1})}, {"planar": Counter({"Rₚ": 2})}):
        assert satisfies_spec(sig, _PLANAR, "preserve")


def test_all_reference_kinds_must_pass():
    ref = {"planar": Counter({"Rₚ": 1}), "axial": Counter({"Rₐ": 1})}
    assert not satisfies_spec({"planar": Counter({"Rₚ": 1}), "axial": Counter({"Sₐ": 1})}, ref, "preserve")

    ref = {**_PLANAR, **_POINT}
    sig = {"planar": Counter({"Sₚ": 1}), "point": Counter({"S": 1})}
    assert satisfies_spec(sig, ref, {"planar": "free", "point": "preserve"}) is False  # point held, and flipped
    assert satisfies_spec(sig, ref, {"planar": "free", "default": "free"})


# --- signature: read the handedness back out of the coordinates -------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_two_enantiomers_get_opposite_point_labels():
    left = signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False), 0)
    right = signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False), 0)
    assert set(left.get("point", ())), f"no point label read from the R conformer: {left}"
    assert set(right.get("point", ())), f"no point label read from the S conformer: {right}"
    assert set(left["point"]) != set(right["point"])
    assert not satisfies_spec(right, left, "all")  # ...and the gate sees the inversion
