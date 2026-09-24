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
def test_registry_rows_are_enumerable(kind):
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


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("kind", "silent", "audible"),
    [
        # Mutation: put F (9) back into _SIGMA_HOLE_Z["XB"] and the fluorobenzene case fires.
        pytest.param("XB", "Fc1ccccc1.n1ccccc1", "Ic1ccccc1.n1ccccc1", id="XB-F-vs-I"),
        # Mutation: put O (8) back into _SIGMA_HOLE_Z["ChB"] and the ether case fires.
        pytest.param("ChB", "COC.n1ccccc1", "CSC.n1ccccc1", id="ChB-O-vs-S"),
        # Mutation: put N (7) back into _SIGMA_HOLE_Z["PnB"] and the amine case fires.
        pytest.param("PnB", "CNC.n1ccccc1", "C[AsH]C.n1ccccc1", id="PnB-N-vs-As"),
    ],
)
def test_sigma_hole_kinds_require_a_polarisable_heavy_donor(kind, silent, audible):
    """Organic C-F/O/N have no usable sigma hole; a heavier congener of the same family does."""
    assert not nci.candidate_contacts(_mol(silent), kinds=(kind,)), f"{kind} must not fire on {silent!r}"
    assert nci.candidate_contacts(_mol(audible), kinds=(kind,)), f"{kind} must fire on {audible!r}"


def test_acceptor_quality_prefers_localised_lone_pair():

    def quality(smiles, symbol):
        mol = _mol(smiles)
        idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol)
        return nci._acceptor_quality(mol, ("atom", idx))

    amine, ketone = quality("CN", "N"), quality("CC(=O)C", "O")
    aromatic, amide = quality("n1ccccc1", "N"), quality("CC(=O)N", "N")
    assert amine > ketone > aromatic, "a strong localised base must outrank a carbonyl O, and both a ring N"
    assert amide == aromatic, "an amide N is delocalised into the C=O: a good DONOR, a poor acceptor"


# --- enumeration ------------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_acid_pyridine_grip_has_orientation():
    mol = _mol(_ACID_PYRIDINE)
    assert mol.GetNumConformers() == 0, "discovery must not depend on a probe conformer's luck"
    cands = nci.candidate_contacts(mol, kinds=("HB",))
    assert len(cands) == 1, f"exactly one inter-fragment H-bond is available here: {list(cands)}"
    contact = next(iter(cands.values()))
    assert next(iter(contact.distances.values())) == nci.KINDS["HB"].window
    assert contact.angles, "a directional grip with no orientation angle binds bent and is weakly detected"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_ring_contacts_are_opt_in():
    mol = _mol(_ACID_PYRIDINE)
    assert not any(k.startswith("HBPI") for k in nci.auto_binding_modes(mol))
    assert any(k.startswith("HBPI") for k in nci.candidate_contacts(mol))


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_reciprocal_cycle_splits_into_one_way_modes():
    mol = _mol(_ACID_DIMER)
    (h1, o1), (h2, o2) = _hydroxyl_hydrogens(mol)
    a, b = tuple(sorted((h1, o2))), tuple(sorted((h2, o1)))
    modes = nci.auto_binding_modes(mol)
    assert any(a in m.distances for m in modes.values()), "the A-H...B direction was lost with the 2-cycle"
    assert any(b in m.distances for m in modes.values()), "the B-H...A direction was lost with the 2-cycle"
    assert not any({a, b} <= set(m.distances) for m in modes.values()), "a reciprocal 2-cycle survived in one mode"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_multipoint_mode_outranks_single_contacts():
    modes = list(nci.auto_binding_modes(_mol(_ACID_DIMER)).values())
    assert len(modes[0].distances) >= len(modes[-1].distances)
    assert len(modes[0].distances) > 1, "the acid dimer's two-point grip was not enumerated"
