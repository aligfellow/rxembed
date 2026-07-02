"""Unit tests for the constraint resolver — ``resolve_core`` (the fix / constrain / template grammar).

Pure and fast: no embedding, no force field, no optional deps. Each test pins one cell of the DESIGN
grammar (§3) or one invariant (§5), asserting the exact `Constraints` produced — windows, frozen set,
graft ``ref``, and the ``contacts`` provenance that decides what ``mc(explore=)`` may release. These are
deliberately strict (exact dict contents, not "at least") so a loosened resolver is caught.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.constraints import resolve_core


def _mol(smiles="CCO", seed=1):
    """A small molecule WITH a conformer (ethanol: heavy atoms 0=C, 1=C, 2=O)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


# --- fix: own coordinates (a list) -------------------------------------------


def test_fix_list_holds_own_coords():
    m = _mol()
    cons, ref = resolve_core(m, fix=[0, 1, 2], has_geometry=True)
    assert cons.frozen == {0, 1, 2}
    assert set(ref) == {0, 1, 2}
    pos = m.GetConformer().GetPositions()
    for i in (0, 1, 2):
        assert np.allclose(ref[i], pos[i])
    # 3 pairwise shape windows (C(3,2)), each bracketing the true distance
    assert set(cons.distances) == {(0, 1), (0, 2), (1, 2)}
    for (i, j), (lo, hi) in cons.distances.items():
        assert lo <= np.linalg.norm(pos[i] - pos[j]) <= hi
    # a grafted core is structural — nothing releasable
    assert cons.contacts == (frozenset(), frozenset())
    assert cons.relaxed().distances == cons.distances


def test_fix_list_needs_geometry():
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # no conformer
    with pytest.raises(ValueError, match="own coordinates"):
        resolve_core(m, fix=[0, 1, 2], has_geometry=False)


def test_fix_list_rejects_smarts():
    m = _mol()
    with pytest.raises(ValueError, match="index"):
        resolve_core(m, fix=["[OX2]"], has_geometry=True)


# --- fix: explicit coordinates / template ------------------------------------


def test_fix_explicit_coords():
    m = _mol()
    coords = {0: (0.0, 0.0, 0.0), 1: (1.5, 0.0, 0.0), 2: (1.5, 1.4, 0.0)}
    cons, ref = resolve_core(m, fix=coords, has_geometry=True)
    assert cons.frozen == {0, 1, 2}
    for i, c in coords.items():
        assert np.allclose(ref[i], c)
    lo, hi = cons.distances[(0, 1)]  # shape from the GIVEN coords, not the molecule's own
    assert lo <= 1.5 <= hi


def test_template_grafts_reference_by_map():
    m = _mol()
    ref_pos = _mol("CCCC", seed=3).GetConformer().GetPositions()  # a different geometry as reference
    cons, ref = resolve_core(m, template=(ref_pos, {0: 1, 1: 2, 2: 3}), has_geometry=True)
    assert cons.frozen == {0, 1, 2}
    assert np.allclose(ref[0], ref_pos[1])
    assert np.allclose(ref[1], ref_pos[2])
    assert np.allclose(ref[2], ref_pos[3])


# --- fix: numbers (tight windows, UFF-pulled, structural) --------------------


def test_fix_number_distance_is_tight_and_structural():
    m = _mol()
    cons, ref = resolve_core(m, fix={(0, 2): 2.0}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((1.98, 2.02))
    assert set(cons.distances) == {(0, 2)}
    assert cons.frozen == set()  # not grafted...
    assert ref == {}  # ...and no graft coords
    assert cons.contacts == (frozenset(), frozenset())  # fix is never releasable


def test_fix_number_angle():
    m = _mol()
    cons, _ = resolve_core(m, fix={(0, 1, 2): 109.5}, has_geometry=True)
    assert cons.angles[(0, 1, 2)] == pytest.approx((107.5, 111.5))
    assert cons.contacts == (frozenset(), frozenset())


def test_fix_number_window_taken_verbatim():
    m = _mol()
    cons, _ = resolve_core(m, fix={(0, 2): (1.9, 2.1)}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((1.9, 2.1))


def test_fix_dict_mixes_coords_and_numbers():
    m = _mol()
    cons, ref = resolve_core(
        m, fix={0: (0.0, 0.0, 0.0), 1: (1.5, 0.0, 0.0), 2: (1.5, 1.4, 0.0), (3, 4): 1.1}, has_geometry=True
    )
    assert cons.frozen == {0, 1, 2}
    assert set(ref) == {0, 1, 2}
    assert cons.distances[(3, 4)] == pytest.approx((1.08, 1.12))  # the number, alongside the graft shape
    assert (3, 4) not in {(0, 1), (0, 2), (1, 2)}


# --- constrain: soft windows + planes (releasable) ---------------------------


def test_constrain_distance_soft_and_releasable():
    m = _mol()
    cons, ref = resolve_core(m, constrain={(0, 2): (2.6, 3.0)}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((2.6, 3.0))
    assert ref == {}
    assert cons.contacts == (frozenset({(0, 2)}), frozenset())
    assert cons.relaxed().distances == {}  # released for the exploratory pass


def test_constrain_scalar_widens_to_soft_window():
    m = _mol()
    cons, _ = resolve_core(m, constrain={(0, 2): 2.8}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((2.7, 2.9))  # ± the soft pad, wider than fix


def test_constrain_angle_soft():
    m = _mol()
    cons, _ = resolve_core(m, constrain={(0, 1, 2): 120.0}, has_geometry=True)
    assert cons.angles[(0, 1, 2)] == pytest.approx((115.0, 125.0))
    assert cons.contacts[1] == frozenset({(0, 1, 2)})


def test_constrain_plane_stack():
    m = _mol("c1ccccc1.c1ccccc1")
    ra, rb = (tuple(r) for r in m.GetRingInfo().AtomRings()[:2])
    cons, _ = resolve_core(m, constrain={(ra, rb): 3.7}, has_geometry=True)
    assert cons.planes == [(ra, rb, 3.7)]


# --- fix + constrain compose: provenance keeps them distinct -----------------


def test_fix_structural_constrain_releasable_compose():
    m = _mol()
    cons, _ = resolve_core(m, fix={(0, 1): 1.5}, constrain={(0, 2): (2.6, 3.0)}, has_geometry=True)
    relaxed = cons.relaxed()
    assert (0, 1) in relaxed.distances  # fix held through explore
    assert (0, 2) not in relaxed.distances  # constrain released


# --- errors & advisories (invariants) ----------------------------------------


def test_smarts_key_raises_with_guidance():
    m = _mol()
    with pytest.raises(ValueError, match="index-driven"):
        resolve_core(m, constrain={("[OX2]", "[CH3]"): (2.6, 3.0)}, has_geometry=True)


def test_out_of_range_index_raises():
    m = _mol()
    with pytest.raises(ValueError, match="out of range"):
        resolve_core(m, fix={(0, 999): 2.0}, has_geometry=True)


def test_fewer_than_three_graft_atoms_warns(caplog):
    m = _mol()
    with caplog.at_level("WARNING"):
        resolve_core(m, fix=[0, 1], has_geometry=True)
    assert any("orientable" in r.message or ">=3" in r.message for r in caplog.records)


def test_underdetermined_angle_warns(caplog):
    m = _mol()
    with caplog.at_level("WARNING"):
        resolve_core(m, fix={(0, 1): 1.5, (1, 2): 1.4}, has_geometry=True)  # shared atom 1, no angle
    assert any("angle" in r.message.lower() for r in caplog.records)


def test_no_underdetermined_warning_when_angle_given(caplog):
    m = _mol()
    with caplog.at_level("WARNING"):
        resolve_core(m, fix={(0, 1): 1.5, (1, 2): 1.4, (0, 1, 2): 109.0}, has_geometry=True)
    assert not any("bent" in r.message for r in caplog.records)


def test_echo_names_atoms_with_element_symbols(caplog):
    # invariant 1: the INFO echo must name atoms by element+index so a 1-based / wrong-atom pick is obvious
    m = _mol()  # ethanol: 0=C, 1=C, 2=O
    with caplog.at_level("INFO", logger="rxembed"):
        resolve_core(m, fix={(0, 2): 2.0}, has_geometry=True)
    echo = " ".join(r.message for r in caplog.records if "resolve:" in r.message)
    assert "C0" in echo  # element symbols, not bare indices...
    assert "O2" in echo
    assert "2.0" in echo  # and the target it pinned


def test_no_underdetermined_warning_for_rich_network(caplog):
    m = _mol("CCCC")  # 4 heavy atoms: a deliberate distance web, not the ambiguous 3-atom case
    with caplog.at_level("WARNING"):
        resolve_core(m, fix={(0, 1): 1.5, (1, 2): 1.5, (2, 3): 1.5}, has_geometry=True)
    assert not any("bent" in r.message for r in caplog.records)
