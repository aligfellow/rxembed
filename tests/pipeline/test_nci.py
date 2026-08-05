"""`pipeline/nci.py`: the `KINDS` registry and the binding modes enumerated from it."""

from importlib.util import find_spec

import pytest
from rdkit import Chem

from rxembed.pipeline import nci

_ACID_PYRIDINE = "OC(=O)c1ccccc1.n1ccccc1"  # one unambiguous grip: the acid O-H onto the pyridine N
_ACID_DIMER = "CC(=O)O.CC(=O)O"  # each fragment can donate to the other: the reciprocal case


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _hydroxyl_hydrogens(mol):
    """[(H index, its O)] for every O-H: the donors of the dimer's two competing grips."""
    return [
        (a.GetIdx(), a.GetNeighbors()[0].GetIdx())
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() == 8
    ]


# --- the registry: adding a contact type is a row, so a bad row must be caught here ------------------------


@pytest.mark.parametrize("kind", nci.KINDS.values(), ids=list(nci.KINDS))
def test_every_registry_row_is_one_the_generic_enumerator_can_handle(kind):
    anchors = {"atom": ("donor_h", "heavy"), "ring": ("ring_h", "ring_ion"), "hydride": ("hydride",)}
    assert kind.family in anchors
    assert kind.anchor in anchors[kind.family], f"anchor {kind.anchor!r} is not a {kind.family} anchor"

    if kind.family == "ring":
        assert isinstance(kind.window, float), "a ring contact's window is one atom-centroid distance"
    else:
        lo, hi = kind.window
        assert 0 < lo < hi

    # an orientation window with no apex to measure from is unenforceable; an apex with no window is unused
    assert (kind.orient is None) == (kind.apex is None)
    if kind.orient:
        lo, hi = kind.orient
        assert 90.0 < lo < hi <= 180.0, "a contact orientation is an obtuse-to-linear window"


def test_the_sigma_hole_sets_hold_only_polarisable_heavies():
    assert 9 not in nci._SIGMA_HOLE_Z["XB"], "organic C-F has no sigma hole"
    assert 8 not in nci._SIGMA_HOLE_Z["ChB"], "an ether O is not a donor"
    assert 7 not in nci._SIGMA_HOLE_Z["PnB"], "an amine N is a lone-pair DONOR"


def test_acceptor_quality_ranks_a_localised_lone_pair_over_a_delocalised_one():

    def quality(smiles, symbol):
        mol = _mol(smiles)
        idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol)
        return nci._acceptor_quality(mol, ("atom", idx))

    amine, ketone = quality("CN", "N"), quality("CC(=O)C", "O")
    aromatic, amide = quality("n1ccccc1", "N"), quality("CC(=O)N", "N")
    assert amine > ketone > aromatic, "a strong localised base must outrank a carbonyl O, and both a ring N"
    assert amide == aromatic, "an amide N is delocalised into the C=O: a good DONOR, a poor acceptor"


# --- enumeration ------------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_the_acid_pyridine_grip_is_discovered_with_its_orientation_angle():
    mol = _mol(_ACID_PYRIDINE)
    assert mol.GetNumConformers() == 0, "discovery must not depend on a probe conformer's luck"
    cands = nci.candidate_contacts(mol, kinds=("HB",))
    assert len(cands) == 1, f"exactly one inter-fragment H-bond is available here: {list(cands)}"
    contact = next(iter(cands.values()))
    assert next(iter(contact.distances.values())) == nci.KINDS["HB"].window
    assert contact.angles, "a directional grip with no orientation angle binds bent and is weakly detected"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_ring_contacts_are_off_by_default_but_available_on_request():
    mol = _mol(_ACID_PYRIDINE)
    assert not any(k.startswith("HBPI") for k in nci.auto_binding_modes(mol))
    assert any(k.startswith("HBPI") for k in nci.candidate_contacts(mol))


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_a_reciprocal_two_cycle_is_split_into_two_one_way_modes():
    mol = _mol(_ACID_DIMER)
    (h1, o1), (h2, o2) = _hydroxyl_hydrogens(mol)
    a, b = tuple(sorted((h1, o2))), tuple(sorted((h2, o1)))
    modes = nci.auto_binding_modes(mol)
    assert any(a in m.distances for m in modes.values()), "the A-H...B direction was lost with the 2-cycle"
    assert any(b in m.distances for m in modes.values()), "the B-H...A direction was lost with the 2-cycle"
    assert not any({a, b} <= set(m.distances) for m in modes.values()), "a reciprocal 2-cycle survived in one mode"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_a_multipoint_mode_ranks_above_the_single_contacts_it_contains():
    modes = list(nci.auto_binding_modes(_mol(_ACID_DIMER)).values())
    assert len(modes[0].distances) >= len(modes[-1].distances)
    assert len(modes[0].distances) > 1, "the acid dimer's two-point grip was not enumerated"
