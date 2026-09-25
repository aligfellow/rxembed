"""Test the native-reach screen that prunes enumerated coordination candidates."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem, DistanceGeometry

import rxembed as rx
from rxembed import metal_enumeration, metal_screen
from rxembed.bounds import ligand_reach
from rxembed.constraints import Constraints
from rxembed.mechanisms import law_of_cosines
from rxembed.metal_slots import TRANS_ANGLE
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


def test_short_trans_requires_euclidean_consistency_not_only_triangle_smoothing(monkeypatch):
    """Disabling either check alone still excludes the trans chelate; the edge rule now also proves it.

    The dropped isomer puts the N,N chelate on square_planar's one non-edge (trans) vertex pair, so
    `metal_slots.chelate_edge_links` excludes it independently of `_euclidean_conflict`; both must be
    disabled together to show triangle smoothing alone would have let it through.
    """
    smiles = "Cl[Pt]1(F)N(C)CCCN1"
    assert len(rx.metal(smiles)) == 2
    assert len(rx.metal(smiles, screen=False)) == 3
    monkeypatch.setattr(metal_screen, "_euclidean_conflict", lambda _matrix, **_kwargs: None)
    assert len(rx.metal(smiles)) == 2
    monkeypatch.setattr(metal_enumeration, "chelate_edge_links", lambda *args, **kwargs: frozenset())
    assert len(rx.metal(smiles)) == 3


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
    reach = np.full((5, 5), 10.0)
    reach[1, 3] = reach[3, 1] = 3.0
    monkeypatch.setattr(metal_screen, "donor_distance_window", lambda *_args, **_kwargs: (2.0, 2.1))
    monkeypatch.setattr(metal_screen, "_opposed_donor_span_failure", lambda *_args: None)

    failure = metal_screen.unreachable_span(iso, reach, {}, ())

    needed = law_of_cosines(2.0, 2.0, TRANS_ANGLE)
    assert failure == f"donors 1/3 need >= {needed:.3f} A; ligand reach <= 3.000 A"


def test_bonded_donors_keep_native_triangle_for_reach(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("CC.C.C.[He]")
    iso = SimpleNamespace(
        centres=(0, 1),
        base_cons=Constraints(),
        graph=mol,
        metal=4,
        vertices=(0, 2, 1, 3),
        haptic={},
        geometry="square_planar",
        length_mol=mol,
        lengths="input",
        real_z=46,
        donors=(0, 1, 2, 3),
    )
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    reach = np.full((5, 5), 10.0)
    reach[0, 1] = reach[1, 0] = 3.0
    monkeypatch.setattr(metal_screen, "donor_distance_window", lambda *_args, **_kwargs: (2.0, 2.1))
    monkeypatch.setattr(metal_screen, "_opposed_donor_span_failure", lambda *_args: None)

    assert metal_screen.unreachable_span(iso, reach, {}, ()) is None
    iso.vertices = (0, 2, 3, 1)
    assert metal_screen.unreachable_span(iso, reach, {}, ()) is None


def test_equal_length_routes_are_checked_together(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C1CCC1.[He]")
    iso = SimpleNamespace(graph=mol, cons=None, metal=4, donors=(0, 2))
    points = np.array(((-1.0, 0.0, 0.0), (0.8, 0.6, 0.0), (1.0, 0.0, 0.0), (0.8, -0.6, 0.0), (0.0, 0.0, 0.0)))
    matrix = np.linalg.norm(points[:, None] - points, axis=2)
    monkeypatch.setattr(metal_screen, "coordination_reach", lambda *_: matrix.copy())
    assert metal_screen._compiled_span_failure(iso, None) is None
    # Both donor routes remain realizable alone, but their off-axis points cannot be this far apart.
    matrix[1, 3] = matrix[3, 1] = 1.24
    assert DistanceGeometry.DoTriangleSmoothing(matrix.copy())
    assert metal_screen._compiled_span_failure(iso, None) is not None


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

    assert metal_screen._compiled_span_failure(iso, native, native=native) == (
        "compiled coordination distances conflict with native ligand reach"
    )


def test_repeated_route_unions_do_not_repeat_the_same_search(monkeypatch):
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
    assert calls == {3: 4, 4: 4, 5: 1}


def test_compiled_span_skips_a_bridged_donor_route(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("NN.[He]")
    iso = SimpleNamespace(graph=mol, cons=None, metal=2, donors=(0, 1))
    matrix = np.ones((3, 3)) - np.eye(3)
    monkeypatch.setattr(metal_screen, "coordination_reach", lambda *_args, **_kwargs: matrix.copy())
    monkeypatch.setattr(metal_screen, "_euclidean_conflict", lambda *_args, **_kwargs: pytest.fail("bridged route"))

    assert metal_screen._route_has_donor_bond(mol, (0, 1), {0, 1})
    assert metal_screen._compiled_span_failure(iso, matrix, native=matrix) is None


def test_long_route_keeps_only_the_complete_euclidean_witness():
    path = tuple(range(9))

    assert list(metal_screen._route_certificate_subsets(path)) == [path]


def test_inconclusive_haptic_subset_keeps_the_prior_donor_facing_screen(monkeypatch):
    smiles = "[Cl-]->[Pt+2]12(<-[NH2]CCC[NH2]->1)<-[CH2]=[CH2]->2"
    monkeypatch.setattr(metal_screen, "_compiled_span_failure", lambda *_args: None)
    monkeypatch.setattr(metal_screen, "_opposed_donor_span_failure", lambda *_args: "prior donor-facing conflict")

    assert len(rx.metal(smiles, "SPL")) == 0
    assert len(rx.metal(smiles, "SPL", screen=False)) == 2


def test_compiled_span_search_streams_subsets_after_the_whole_path(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("CCCC.[He]")
    iso = SimpleNamespace(graph=mol, cons=None, metal=4, donors=(0, 3))
    matrix = np.ones((5, 5)) - np.eye(5)
    monkeypatch.setattr(metal_screen, "coordination_reach", lambda *_args: matrix)
    monkeypatch.setattr(metal_screen, "_euclidean_conflict", lambda _matrix, **_kwargs: 1.0)
    combinations = itertools.combinations

    def guarded_subsets(values, size):
        assert size == 2, "smaller subsets were consumed before testing the whole path"
        yield from combinations(values, size)

    monkeypatch.setattr(itertools, "combinations", guarded_subsets)
    assert metal_screen._compiled_span_failure(iso, matrix) is not None


def test_narrow_span_pruning_clears_the_tethered_orbit_cap():
    """A tris-dien La(III) sphere (all 9 donors one fragment) needs `narrow_span_pairs` under the cap.

    Without the up-front prune, `distinct_vertex_orderings` streams all 10,098 raw orbits of `tricapped_trigonal_
    prismatic` and trips `MAX_EXHAUSTIVE_ORBITS` (1,000) before the per-candidate reach screen ever runs.
    128 without the chelate edge rule; each of its 6 dien arms (2 per ligand) must additionally land on a
    `tricapped_trigonal_prismatic` hull edge, which drops 66 more (all 66 place an arm on a non-edge pair).
    """
    smiles = "C1C[NH]2CC[NH2]->[La+3]<-23456(<-[NH2]1)(<-[NH2]CC[NH]->3CC[NH2]->4)<-[NH2]CC[NH]->5CC[NH2]->6"
    mol = rx.parse_smiles(smiles)

    isomers = rx.metal(mol)

    assert len(isomers) == 62
    with pytest.raises(ValueError, match="more than 1,000 distinct constitutional"):
        rx.metal(mol, screen=False)


@pytest.mark.parametrize(
    ("smiles", "count"),
    [
        # 4, not 5: the all-equatorial seating (both dien bites on adjacent equatorial slots) forces the
        # third equatorial pair to 180 deg -- a square-pyramidal reading, not trigonal_bipyramidal, at every
        # point in its bite windows including the anchor -- so `metal_constraints.bounded_bites` refuses it
        # ("chelate bites leave trigonal_bipyramidal") independently of this test's own pruning screen.
        ("[Cl-]->[La+3]12(<-[Cl-])<-[NH2]CC[NH]->1CC[NH2]->2", 4),
        ("[Cl-]->[La+3]123(<-[Cl-])(<-[NH2]CC[NH2]->1)<-[NH2]CC[NH]->2CC[NH2]->3", 24),
    ],
    ids=["tbp", "pbp"],
)
def test_narrow_span_pruning_loses_no_reachable_arrangement(monkeypatch, smiles, count):
    """The pruned and unpruned tethered pools agree exactly: pruning removes only already-doomed orbits."""
    mol = rx.parse_smiles(smiles)

    pruned = {rx.cxsmiles(iso) for iso in rx.metal(mol)}
    assert len(pruned) == count

    monkeypatch.setattr(metal_enumeration, "narrow_span_pairs", lambda *args, **kwargs: frozenset())
    unpruned = {rx.cxsmiles(iso) for iso in rx.metal(mol)}

    assert unpruned == pruned


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
