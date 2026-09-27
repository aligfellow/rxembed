"""Test the native-reach screen that prunes enumerated coordination candidates."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem, DistanceGeometry

import rxembed as rx
from rxembed import metal_screen
from rxembed.bounds import ligand_reach
from rxembed.constraints import Constraints
from rxembed.mechanisms import law_of_cosines
from rxembed.metal_slots import SPAN_TOL, TRANS_ANGLE
from rxembed.metal_stereo import chelate_links, site_classes


@pytest.mark.parametrize("refine", [False, True])
def test_interval_euclidean_certificate_is_not_a_midpoint_or_rank_test(refine):
    matrix = np.full((4, 4), 2.0)
    matrix[3, :3] = matrix[:3, 3] = 1.1
    np.fill_diagonal(matrix, 0.0)
    assert DistanceGeometry.DoTriangleSmoothing(matrix.copy())
    assert metal_screen._euclidean_conflict(matrix, refine=refine) is not None
    matrix[3, :3], matrix[:3, 3] = 0.1, 1.16  # Contains the valid equilateral-base centre.
    assert metal_screen._euclidean_conflict(matrix, refine=refine) is None
    # Valid in dimension four.
    assert metal_screen._euclidean_conflict(np.ones((5, 5)) - np.eye(5), refine=refine) is None
    rng = np.random.default_rng(42)
    for _ in range(100):
        points = rng.normal(size=(5, 3))
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        slack = rng.uniform(0.0, 0.5, distances.shape)
        intervals = np.triu(distances + slack, 1) + np.tril(np.maximum(0.0, distances - slack), -1)
        assert metal_screen._euclidean_conflict(intervals, refine=refine) is None


def test_interval_euclidean_certificate_preserves_degenerate_eigenspaces():
    groups = np.arange(9) // 3
    squared = np.where(groups[:, None] == groups[None, :], 4.0, 1.21)
    np.fill_diagonal(squared, 0.0)
    lower, upper = np.sqrt(0.65 * squared), np.sqrt(1.35 * squared)
    rng = np.random.default_rng(42)
    for _ in range(20):
        order = rng.permutation(9)
        intervals = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        assert DistanceGeometry.DoTriangleSmoothing(intervals.copy())
        assert metal_screen._euclidean_conflict(intervals) == pytest.approx(0.2995, abs=1e-12)
    for invalid in (-0.1, np.nan, np.inf):
        intervals[1, 0] = invalid
        assert metal_screen._euclidean_conflict(intervals) is None


def test_refinement_certifies_coupled_modes_without_atom_order_dependence():
    n = 8
    u = np.array((1, 1, 1, 1, -1, -1, -1, -1)) / np.sqrt(n)
    v = np.array((1, 1, -1, -1, 1, 1, -1, -1)) / np.sqrt(n)
    gram = np.eye(n) - np.ones((n, n)) / n - 1.2 * np.outer(u, u) - 1.1 * np.outer(v, v)
    squared = np.diag(gram)[:, None] + np.diag(gram) - 2 * gram
    slack = np.where(np.outer(u, u) * np.outer(v, v) < 0, 0.12, 0.0)
    lower, upper = np.sqrt(np.maximum(squared - slack, 0.0)), np.sqrt(squared + slack)
    rng = np.random.default_rng(42)
    margin = None
    for _ in range(10):
        order = rng.permutation(n)
        matrix = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        before = matrix.copy()
        assert metal_screen._euclidean_conflict(matrix) is None
        found = metal_screen._euclidean_conflict(matrix, refine=True)
        assert found is not None
        if margin is not None:
            assert found == pytest.approx(margin, abs=1e-10)
        margin = found
        np.testing.assert_array_equal(matrix, before)


def test_short_trans_requires_euclidean_consistency_not_only_triangle_smoothing():
    """A tethered N,N chelate too short for square_planar's trans slot pair is screened out."""
    smiles = "Cl[Pt]1(F)N(C)CCCN1"
    assert len(rx.metal(smiles)) == 2
    assert len(rx.metal(smiles, screen=False)) == 3


def test_trans_reach_screen_uses_the_shared_150_degree_slot_boundary(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.C.C.C.[He]")
    iso = SimpleNamespace(
        centres=(0, 1),
        base_cons=Constraints(),
        graph=mol,
        metal=4,
        vertices=(0, 1, 2, 3),
        haptic={},
        geometry="square_pyramidal",
        length_mol=mol,
        lengths="model",
        real_z=46,
        donors=(0, 1, 2, 3),
    )
    monkeypatch.setattr(metal_screen, "donor_distance_window", lambda *_args, **_kwargs: (2.0, 2.1))
    monkeypatch.setattr(metal_screen, "_opposed_donor_span_failure", lambda *_args: None)
    # the trans slot boundary at each donor's lower M-L bound (2.0, not the upper 2.1), widened by SPAN_TOL
    boundary = law_of_cosines(2.0, 2.0, TRANS_ANGLE) - SPAN_TOL

    reach = np.full((5, 5), 10.0)
    reach[1, 3] = reach[3, 1] = boundary - 0.02
    assert metal_screen.unreachable_span(iso, reach, {}, ()) is not None

    reach[1, 3] = reach[3, 1] = boundary + 0.02
    assert metal_screen.unreachable_span(iso, reach, {}, ()) is None


def test_compiled_span_uses_local_triangle_before_global_certificate(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.[He].[He].[He].[He]")
    constraints = Constraints(
        metals={4},
        distances={(0, 4): (2.0, 2.0), (2, 4): (2.0, 2.0)},
        angles={(0, 4, 2): (136.0, 152.0)},
    )
    iso = SimpleNamespace(graph=mol, cons=constraints, metal=4, donors=(0, 2))
    native = np.full((5, 5), np.inf)
    native[2, 0], native[0, 2] = 0.0, 2.5
    monkeypatch.setattr(
        metal_screen,
        "coordination_reach",
        lambda *_args, **_kwargs: pytest.fail("the local contradiction should short-circuit the global certificate"),
    )

    assert "native ligand reach" in metal_screen._compiled_span_failure(iso, native, native=native)


def test_repeated_route_unions_do_not_repeat_the_same_search(monkeypatch):
    """Every route union is checked down to its 3-donor diagnostic subsets, not just the whole path."""
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C1CCC1.[He]")
    iso = SimpleNamespace(graph=mol, cons=None, metal=4, donors=(0, 1, 2, 3))
    matrix = np.ones((5, 5)) - np.eye(5)
    calls = Counter()

    def checked(block, **_kwargs):
        calls[len(block)] += 1

    monkeypatch.setattr(metal_screen, "coordination_reach", lambda *_: matrix.copy())
    monkeypatch.setattr(metal_screen, "_euclidean_conflict", checked)
    monkeypatch.setattr(metal_screen, "_route_has_donor_bond", lambda *_args: False)
    assert metal_screen._compiled_span_failure(iso, None) is None
    assert calls[3] > 0, "no 3-donor diagnostic subset was ever checked"
    assert set(calls) == {3, 4, 5}, "every route union size down to a 3-donor subset must be checked"


def test_tris_dien_lanthanum_stays_under_the_orbit_cap():
    """A tris-dien La(III) sphere (CN9, all one fragment) enumerates without tripping the exact-orbit cap."""
    smiles = "C1C[NH]2CC[NH2]->[La+3]<-23456(<-[NH2]1)(<-[NH2]CC[NH]->3CC[NH2]->4)<-[NH2]CC[NH]->5CC[NH2]->6"
    mol = rx.parse_smiles(smiles)

    isomers = rx.metal(mol)

    assert len(isomers) == 62
    with pytest.raises(ValueError, match="more than 1,000 distinct constitutional"):
        rx.metal(mol, screen=False)


def test_tethered_haptic_faces_reject_an_unreachable_trans_state():
    smiles = "CC#[N]->[Ru+2]123(<-[Cl-])(<-[Cl-])(<-[N]#CC)<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31"

    isomers = rx.metal(smiles, "octahedral", screen=False)
    trans, cis = isomers[0], isomers[1]
    seed = rx.embed(cis, n=1, seed=42, threads=1).mol
    trans.length_mol.AddConformer(Chem.Conformer(seed.GetConformer(0)))
    reach = ligand_reach(trans.length_mol)
    links = chelate_links(trans.graph, trans.vertices, trans.haptic)
    classes = site_classes(trans.graph, trans.vertices, trans.haptic, ())

    assert len(isomers) == 6
    assert "haptic faces" in metal_screen.unreachable_span(trans, reach, classes, links)

    model = rx.metal(smiles, "octahedral", screen=False)[0]
    model_reach = ligand_reach(model.length_mol)
    model_links = chelate_links(model.graph, model.vertices, model.haptic)
    model_classes = site_classes(model.graph, model.vertices, model.haptic, ())
    assert "haptic faces" in metal_screen.unreachable_span(model, model_reach, model_classes, model_links)


def test_haptic_centroid_reach_uses_a_valid_minimum_matching():
    reach = np.full((4, 4), np.inf)
    reach[0, 2], reach[2, 0] = 5.0, 5.0
    reach[0, 3], reach[3, 0] = 1.0, 1.0
    reach[1, 2], reach[2, 1] = 1.0, 1.0
    reach[1, 3], reach[3, 1] = 5.0, 5.0

    assert metal_screen._centroid_reach(reach, (0, 1), (2, 3)) == pytest.approx(1.0)


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
def test_haptic_span_keeps_realizable_member_angles(measured, reordered):
    from types import SimpleNamespace

    positions = np.array([[2, 0.7, 0], [2, -0.7, 0], [-2, 0.7, 0], [-2, -0.7, 0], [0, 0, 0]])
    mol = Chem.MolFromSmiles("C.C.C.C.[Mo]")
    metal, left, right = 4, (0, 1), (2, 3)
    if reordered:
        mol = Chem.RenumberAtoms(mol, [4, 3, 2, 1, 0])
        positions = positions[::-1, [1, 2, 0]] + np.array([1.0, 2.0, 3.0])
        metal, left, right = 0, (4, 3), (2, 1)
    conformer = Chem.Conformer(5)
    conformer.SetPositions(positions)
    mol.AddConformer(conformer)
    iso = SimpleNamespace(
        haptic={5: left, 6: right}, length_mol=mol, metal=metal, lengths="input" if measured else "model"
    )
    reach = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    cons = Constraints(distances={(metal, 5): (2.0, 2.0), (metal, 6): (2.0, 2.0)})
    for atom in (*left, *right):
        radius = float(np.linalg.norm(positions[atom] - positions[metal]))
        cons.distances[tuple(sorted((metal, atom)))] = (radius, radius)

    # The centroids are trans, but their individual member rays are not.
    assert metal_screen._haptic_span_failure(iso, reach, 5, 6, 180.0, cons) is None
    for a, b in itertools.product(left, right):
        reach[a, b] = reach[b, a] = 3.5
    assert "centroid reach" in metal_screen._haptic_span_failure(iso, reach, 5, 6, 180.0, cons)
