"""`pipeline/nci.py`: the `KINDS` registry and the binding modes enumerated from it."""

from importlib.util import find_spec

import pytest
from rdkit import Chem

from rxembed.pipeline import nci

_ACID_DIMER = "CC(=O)O.CC(=O)O"  # each fragment can donate to the other: the reciprocal case


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("kind", "silent", "audible"),
    [
        # Mutation: put O (8) back into _SIGMA_HOLE_Z["ChB"] and the ether case fires.
        pytest.param("ChB", "COC.n1ccccc1", "CSC.n1ccccc1", id="ChB-O-vs-S"),
    ],
)
def test_sigma_hole_kinds_require_a_polarisable_heavy_donor(kind, silent, audible):
    """Organic C-F/O/N have no usable sigma hole; a heavier congener of the same family does."""
    assert not nci.candidate_contacts(_mol(silent), kinds=(kind,)), f"{kind} must not fire on {silent!r}"
    assert nci.candidate_contacts(_mol(audible), kinds=(kind,)), f"{kind} must fire on {audible!r}"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_cation_pi_needs_an_aromatic_ring():
    """A saturated ring has no pi face; the aromatic ring of the same size does."""
    assert not nci.candidate_contacts(_mol("C1CCCCC1.[NH4+]"), kinds=("CATPI",)), "cyclohexane is not a pi ring"
    assert nci.candidate_contacts(_mol("c1ccccc1.[NH4+]"), kinds=("CATPI",)), "benzene is a pi ring"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("smiles", "stronger_first"),
    [
        pytest.param("CO.CC(=NC)CCC(C)=O", [(4, 10)], id="imine-over-ketone"),
    ],
)
def test_acceptor_quality_prefers_localised_lone_pair(smiles, stronger_first):
    """A methanol donor ranks an N lone pair outside any pi system above a carbonyl O, and one inside below it."""
    modes = list(nci.auto_binding_modes(_mol(smiles)).values())
    accepted = [next(iter(m.distances))[0] for m in modes]
    for strong, weak in stronger_first:
        assert accepted.index(strong) < accepted.index(weak), f"acceptor {strong} must outrank {weak}: {accepted}"


# --- enumeration ------------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_truncated_mode_enumeration_warns(monkeypatch, caplog):
    monkeypatch.setattr(nci, "_MAX_ASSIGNMENTS", 1)
    with caplog.at_level("WARNING", logger="rxembed"):
        nci.auto_binding_modes(_mol(_ACID_DIMER))

    assert "may miss the best grip" in caplog.text
