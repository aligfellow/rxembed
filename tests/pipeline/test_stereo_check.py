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
)
def test_each_mode_accepts_the_handedness_it_names(mode, kept, flipped):
    assert satisfies_spec(_PLANAR, _PLANAR, mode) is kept
    assert satisfies_spec(_FLIPPED_PLANAR, _PLANAR, mode) is flipped


def test_preserve_holds_only_the_nongraph_kinds_where_all_holds_every_kind():
    """A labile protic-amine centre is the embed's own business; locking it by default would drop real poses."""
    flipped = {"point": Counter({"S": 1})}
    assert satisfies_spec(flipped, _POINT, "preserve")
    assert not satisfies_spec(flipped, _POINT, "all")


def test_handedness_is_the_judgement_and_the_count_is_not():
    """xyzgraph's per-conformer perception is count-unstable ({}, {Sₚ:1}, {Sₚ:2}) for one configuration."""
    for sig in ({}, {"planar": Counter({"Rₚ": 1})}, {"planar": Counter({"Rₚ": 2})}):
        assert satisfies_spec(sig, _PLANAR, "preserve")


def test_every_reference_kind_must_pass_and_a_dict_spec_sets_them_one_at_a_time():
    """Passing on the planar element must not excuse an inverted axial one, and per-kind modes compose."""
    ref = {"planar": Counter({"Rₚ": 1}), "axial": Counter({"Rₐ": 1})}
    assert not satisfies_spec({"planar": Counter({"Rₚ": 1}), "axial": Counter({"Sₐ": 1})}, ref, "preserve")

    ref = {**_PLANAR, **_POINT}
    sig = {"planar": Counter({"Sₚ": 1}), "point": Counter({"S": 1})}
    assert satisfies_spec(sig, ref, {"planar": "free", "point": "preserve"}) is False  # point held, and flipped
    assert satisfies_spec(sig, ref, {"planar": "free", "default": "free"})


# --- signature: read the handedness back out of the coordinates -------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_the_two_enantiomers_get_opposite_point_labels():
    """The fingerprint is read from the GEOMETRY, so mirror-image conformers must not share a label."""
    left = signature(_reference_conformer("C[C@H](N)C(=O)O", optimize=False), 0)
    right = signature(_reference_conformer("C[C@@H](N)C(=O)O", optimize=False), 0)
    assert set(left.get("point", ())), f"no point label read from the R conformer: {left}"
    assert set(right.get("point", ())), f"no point label read from the S conformer: {right}"
    assert set(left["point"]) != set(right["point"])
    assert not satisfies_spec(right, left, "all")  # ...and the gate sees the inversion
