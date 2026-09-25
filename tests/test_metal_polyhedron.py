"""Test polyhedron records, aliases and derived predicates."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

from rxembed.constraints import FIX_ANGLE_TOL
from rxembed.metal_perceive import _plane_rms, rank_shapes
from rxembed.metal_polyhedron import (
    _ALIASES,
    POLYHEDRA,
    _fit_trace,
    canonical_slots,
    describe,
    geometries_for_cn,
    hull_edges,
    isomer_permutations,
    point_group,
    relaxed_shell,
    resolve_geometry,
    seat_by_alignment,
    seat_properly,
    vertex_angle,
    vertex_dirs,
)

# --- codes and aliases --------------------------------------------------------------------------------


def test_records_resolve_and_match_vertex_count():
    for name, p in POLYHEDRA.items():
        assert resolve_geometry(p.code) == name, f"{p.code} does not resolve to {name}"
        assert name in [q.name for q in geometries_for_cn(p.cn)]

        v = np.array(p.vertex_dirs, float)
        assert v.shape == (p.cn, 3), f"{name}: vertex_dirs is {v.shape}, expected ({p.cn}, 3)"
        norms = np.linalg.norm(v, axis=1)
        assert np.all(norms > 1e-6), f"{name}: a zero-length vertex direction"
        cos = (v / norms[:, None]) @ (v / norms[:, None]).T
        same = [(i, j) for i in range(p.cn) for j in range(i + 1, p.cn) if cos[i, j] > 0.999]
        assert not same, f"{name}: vertices {same} point the same way"


def test_aliases_are_unique_and_case_insensitive():
    names, codes = [n.lower() for n in POLYHEDRA], [p.code.lower() for p in POLYHEDRA.values() if p.code]
    assert len(codes) == len(set(codes)), "two polyhedra share a 3-letter code"
    assert not set(names) & set(codes), "a polyhedron name is spelled like another's code"
    assert len(_ALIASES) == len(names) + len(codes)

    for spelling in ("OCT", "oct", " Oct ", "octahedral", "OCTAHEDRAL"):
        assert resolve_geometry(spelling) == "octahedral"

    assert describe("SPL") == "square_planar (SPL, CN 4)"
    assert describe("3-coordinate") == "3-coordinate"  # the from_geometry pseudo-name is not invented into a record


# --- the records themselves ---------------------------------------------------------------------------


def test_angle_rows_match_vertex_geometry():
    for name, p in POLYHEDRA.items():
        if p.angles is None:  # CN7 / CN8: no hand-authored rows, `resolved_angles` derives them
            continue
        for i, j, a in p.angles:
            assert abs(a - vertex_angle(p.vertex_dirs[i], p.vertex_dirs[j])) <= 0.5, f"{name} row {(i, j, a)}"


def test_only_flat_based_pyramid_gets_umbrella():
    held = {n: p.umbrella_improper for n, p in POLYHEDRA.items() if p.umbrella_improper is not None}
    assert list(held) == ["trigonal_pyramidal"]

    dirs = POLYHEDRA["trigonal_pyramidal"].vertex_dirs
    for r in (1.4, 3.0):
        rw = Chem.RWMol()
        for _ in range(len(dirs) + 1):
            rw.AddAtom(Chem.Atom(6))
        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(len(dirs), Point3D(0.0, 0.0, 0.0))  # the metal, last
        for i, d in enumerate(dirs):
            u = np.array(d, float) / np.linalg.norm(d)
            conf.SetAtomPosition(i, Point3D(*(u * r)))
        mol.AddConformer(conf)
        got = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), 0, 1, 2, len(dirs))
        assert got == pytest.approx(held["trigonal_pyramidal"], abs=1e-6), f"r={r} Å: the record states {got}°"


def test_angle_tables_cover_claimed_pairs():
    for name, rec in POLYHEDRA.items():
        pairs = {(i, j) for i, j, _a in rec.resolved_angles}
        assert len(pairs) == len(rec.resolved_angles), f"{name}: a vertex pair is stated twice"
        assert all(0 <= i < rec.cn and 0 <= j < rec.cn and i != j for i, j in pairs), f"{name}: a pair is off-record"
        if rec.angles is None:  # derived -> it must be every pair, or the derivation is silently partial
            assert len(pairs) == rec.cn * (rec.cn - 1) // 2, f"{name}: derived table is not all pairs"


def test_new_records_separate_same_cn_neighbors():
    for cn in {p.cn for p in POLYHEDRA.values()}:
        recs = [p for p in POLYHEDRA.values() if p.cn == cn]
        for a, b in itertools.combinations(recs, 2):
            sa, sb = (np.array(sorted(a for _i, _j, a in r.resolved_angles), float) for r in (a, b))
            if len(sa) != len(sb):
                continue
            rms = float(np.sqrt(np.mean((sa - sb) ** 2)))
            assert rms > 5.0, f"{a.name} and {b.name} differ by only {rms:.1f} deg of angle spectrum"


# --- the fold group, the seating parity and the canonical slot labelling -------------------------------


def test_full_point_group_is_rotations_times_reflection():
    expected_rotations = {
        "monocoordinate": 1,
        "linear": 2,
        "trigonal_planar": 6,
        "t_shape": 2,
        "trigonal_pyramidal": 3,
        "square_planar": 8,
        "tetrahedral": 12,
        "seesaw": 2,
        "trigonal_bipyramidal": 6,
        "square_pyramidal": 4,
        "octahedral": 24,
        "trigonal_prismatic": 6,
        "hexagonal_planar": 12,
        "pentagonal_bipyramidal": 10,
        "capped_octahedral": 3,
        "capped_trigonal_prismatic": 2,
        "square_antiprism": 8,
        "dodecahedral": 4,
        "tricapped_trigonal_prismatic": 6,
        "bicapped_square_antiprismatic": 8,
        "edge_contracted_icosahedral": 2,
    }
    assert set(expected_rotations) == set(POLYHEDRA)
    for name, rec in POLYHEDRA.items():
        rot, refl = point_group(tuple(map(tuple, rec.vertex_dirs)))
        dirs = vertex_dirs(name)
        proper = None if dirs is None else point_group(tuple(map(tuple, dirs)))[0]
        assert rot == proper, f"{name}: point_group's proper rotations disagree between vertex_dirs sources"
        assert len(rot) == len(refl) == expected_rotations[name], (
            f"{name}: {len(rot)} rotations against {len(refl)} reflections, expected {expected_rotations[name]}"
        )
        g = min(refl)
        assert {tuple(g[q[v]] for v in range(rec.cn)) for q in rot} == refl, f"{name}: refl is not g o rot"
        assert (min(refl) == tuple(range(rec.cn))) == rec.planar, f"{name}: planarity disagrees with rot & refl"


def test_isomer_permutations_are_complete_proper_orbit_representatives():
    for name, rec in POLYHEDRA.items():
        if rec.cn > 9:
            continue  # the exhaustive public path refuses these pools before iterating them
        dirs = vertex_dirs(name)
        rotations = None if dirs is None else point_group(tuple(map(tuple, dirs)))[0]
        assert rotations is not None, f"{name}: no vertex-direction template"
        count = 0
        for order in isomer_permutations(name):
            orbit = {tuple(order[q[v]] for v in range(rec.cn)) for q in rotations}
            assert order == min(orbit), f"{name}: representative is not its orbit minimum"
            count += 1
        assert count * len(rotations) == math.factorial(rec.cn), f"{name}: incomplete pool"


def test_seat_properly_excludes_reflection():
    dirs = POLYHEDRA["octahedral"].vertex_dirs
    obs = np.array(dirs, float) @ np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])  # a rotation
    order = list(range(6))
    assert seat_properly(obs, dirs, order) == order, "a rotated template is already seated properly"
    mirrored = obs * np.array([1.0, 1.0, -1.0])
    assert seat_properly(mirrored, dirs, order) != order, "the mirrored sphere kept the same seating"
    both = np.linalg.svd(np.array(mirrored)[seat_properly(mirrored, dirs, order)].T @ np.array(dirs, float))
    assert np.linalg.det(both[0] @ both[2]) > 0, "the re-seating is still a reflection"


def test_canonical_slots_fold_rotations_not_reflection():
    dirs = POLYHEDRA["octahedral"].vertex_dirs
    rot, refl = point_group(tuple(map(tuple, dirs)))
    keys = [("a",), ("b",), ("c",), ("d",), ("e",), ("f",)]  # six distinguishable donors: a chiral labelling

    def slots_for(q):
        return canonical_slots(dirs, [keys[q[v]] for v in range(6)])

    assert all(sorted(slots_for(q)) == list(range(6)) for q in rot), "a fold returned a non-permutation"
    seats = {tuple(sorted(zip(slots_for(q), (keys[q[v]] for v in range(6)), strict=True))) for q in rot}
    assert len(seats) == 1, f"the fold is not constant on a proper orbit: {seats}"
    mirror = {tuple(sorted(zip(slots_for(g), (keys[g[v]] for v in range(6)), strict=True))) for g in refl}
    assert not (seats & mirror), "the fold gave an enantiomeric labelling the same canonical seating"


def test_seating_finds_distorted_antiprism_optimum():
    directions = np.array(vertex_dirs("square_antiprism"), float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rng = np.random.RandomState(0)
    optima = (7.9744270801213695, 7.988318361026678, 7.989758899386766)
    for trial, optimum in enumerate(optima):
        observed = directions[rng.permutation(len(directions))] + rng.randn(len(directions), 3) * 0.05
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        order = seat_by_alignment(observed, directions)
        score = float(np.linalg.svd(observed[list(order)].T @ directions, compute_uv=False).sum())
        assert score == pytest.approx(optimum, abs=1e-12), f"trial {trial}: {score:.6f} vs {optimum:.6f}"


def test_seating_searches_the_exact_bounded_orbit_pool():
    observed = np.array(
        [
            [-0.270833112875, -0.663552188301, 0.697386491388],
            [-0.432464667431, -0.593676326291, 0.678618251321],
            [0.060819489794, 0.995413034256, -0.073850395361],
            [0.355595341221, -0.922396266561, 0.150788198263],
            [0.593947166999, 0.297089208410, 0.747639461947],
            [0.201852608751, 0.101956005758, -0.974094706499],
        ]
    )
    ideal = np.array(vertex_dirs("octahedral"), float)

    order = seat_by_alignment(observed, ideal)

    assert tuple(order) == (0, 5, 1, 4, 2, 3)
    assert _fit_trace(observed[order].T @ ideal) == pytest.approx(4.956143442072902, abs=1e-12)


def test_seating_reuses_template_assignments_but_refits_each_geometry(monkeypatch):
    from rxembed import metal_polyhedron as poly

    poly._seating_permutations.cache_clear()
    generate = poly._proper_orbit_permutations
    calls = []

    def counted(dirs):
        calls.append(dirs)
        yield from generate(dirs)

    monkeypatch.setattr(poly, "_proper_orbit_permutations", counted)
    ideal = np.array(vertex_dirs("octahedral"), float)
    for observed in (ideal, ideal[[0, 2, 1, 3, 5, 4]]):
        order = seat_by_alignment(observed, ideal)
        assert _fit_trace(observed[order].T @ ideal) == pytest.approx(6.0)
    assert len(calls) == 1


# --- the convex-hull edge test, for the metal_slots chelate edge rule --------------------------------


_HULL_EDGE_COUNTS = {  # counted by hand against a supporting-plane definition; see metal_slots.chelate_edge_links
    "OCT": 12,
    "TPR": 9,
    "CTP": 13,
    "COC": 15,
    "SQA": 16,
    "DOD": 18,
    "TCT": 21,
    "BSA": 24,
    "ECI": 27,
    "TET": 6,  # every pair: a tetrahedron has no diagonal or trans pair to exclude
}


@pytest.mark.parametrize(("code", "expected"), sorted(_HULL_EDGE_COUNTS.items()))
def test_hull_edges_match_expected_counts(code, expected):
    rec = POLYHEDRA[resolve_geometry(code)]
    assert len(hull_edges(tuple(map(tuple, rec.vertex_dirs)))) == expected


def test_hull_edges_exclude_every_trans_pair():
    for name, rec in POLYHEDRA.items():
        dirs = rec.vertex_dirs
        edges = hull_edges(tuple(map(tuple, dirs)))
        trans = {
            frozenset((i, j))
            for i, j in itertools.combinations(range(len(dirs)), 2)
            if vertex_angle(dirs[i], dirs[j]) == 180
        }
        assert not (trans & edges), f"{name}: a 180 deg (trans) pair counted as a hull edge"


def test_hull_edges_exclude_a_square_face_diagonal():
    dirs = POLYHEDRA["square_antiprism"].vertex_dirs  # 0,1,3,2 is the top face's cycle; 0-3 and 1-2 are diagonals
    edges = hull_edges(tuple(map(tuple, dirs)))
    assert frozenset((0, 1)) in edges
    assert frozenset((0, 3)) not in edges
    assert frozenset((1, 2)) not in edges


def test_hull_edges_are_invariant_under_point_group_rotations():
    """A genuine geometric edge set is a union of proper-rotation orbits: no rotation can turn an edge into
    a non-edge or vice versa. This catches an edge set that swaps one record's edge/non-edge pair, or adds
    an extra (non-orbit) chord to a planar record, even though a total-count check alone would miss it.
    """
    for name, rec in POLYHEDRA.items():
        dirs = tuple(map(tuple, rec.vertex_dirs))
        edges = hull_edges(dirs)
        vdirs = vertex_dirs(name)
        rotations = None if vdirs is None else point_group(tuple(map(tuple, vdirs)))[0]
        assert rotations is not None, f"{name}: no vertex-direction template"
        for q in rotations:
            mapped = frozenset(frozenset((q[i], q[j])) for i, j in (tuple(pair) for pair in edges))
            assert mapped == edges, f"{name}: hull_edges is not invariant under a proper rotation"


def test_seating_allows_reflection_for_achiral_template():
    ideal = np.array(vertex_dirs("octahedral"), float)
    rng = np.random.RandomState(0)
    for magnitude in (0.10, 0.18):
        observed = ideal[rng.permutation(6)] * np.array([1.0, 1.0, -1.0]) + rng.randn(6, 3) * magnitude
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    order = max(
        isomer_permutations("octahedral"),
        key=lambda candidate: _fit_trace(observed[list(candidate)].T @ ideal),
    )
    trans = [
        float(np.degrees(np.arccos(np.clip(observed[order[i]] @ observed[order[j]], -1, 1))))
        for i, j in ((0, 1), (2, 3), (4, 5))
    ]
    assert min(trans) > 140.0


# --- relaxed_shell ------------------------------------------------------------------------------------


def test_relaxed_shell_returns_none_when_it_does_not_converge():
    """A square-pyramidal basal 4-cycle pulled from its 86 deg ideal to 104 deg oscillates: the residual
    stalls at 14 deg regardless of iteration count, rather than shrinking toward `constraints.FIX_ANGLE_TOL`.
    """
    dirs = POLYHEDRA["square_pyramidal"].vertex_dirs
    bites = {frozenset((1, 2)): 104.0, frozenset((2, 3)): 104.0, frozenset((3, 4)): 104.0, frozenset((4, 1)): 104.0}
    assert relaxed_shell(dirs, bites) is None


def test_relaxed_shell_ignores_a_bent_trans_pair_for_the_oriented_type():
    """A square-pyramidal apex fan bitten to both members of a trans basal pair (150 deg ideal) keeps its
    type down to a crystal-realistic 83 deg: the trans pair itself bends without the shape changing, so its
    triple must not decide the oriented-type test. A tetrahedral 4-cycle, which has no trans pair to exclude,
    is unaffected and still reads as a different shape.
    """
    sp = POLYHEDRA["square_pyramidal"].vertex_dirs
    assert relaxed_shell(sp, {frozenset((0, 1)): 83.0, frozenset((0, 3)): 83.0}) is not None

    tet = POLYHEDRA["tetrahedral"].vertex_dirs
    cycle = {frozenset((0, 1)): 80.5, frozenset((1, 2)): 89.0, frozenset((2, 3)): 80.5, frozenset((3, 0)): 89.0}
    assert relaxed_shell(tet, cycle) is None


def test_relaxed_shell_converges_on_a_closed_bite_cycle():
    """A porphyrin-like closed ring of four bites, each donor shared by its two ring neighbours, used to
    oscillate in a +-1.5 deg limit cycle instead of converging: the undamped simultaneous update overshoots
    every round on a closed cycle (see `metal_polyhedron._SHELL_DAMPING`). Every bite must land within
    `constraints.FIX_ANGLE_TOL` of its own target, not just settle at some smaller but nonzero residual.
    """
    dirs = POLYHEDRA["octahedral"].vertex_dirs  # equatorial 4-cycle: 0-2-1-3-0 (each adjacent pair cis, 90 deg ideal)
    bites = {frozenset((0, 2)): 86.5, frozenset((2, 1)): 90.5, frozenset((1, 3)): 96.5, frozenset((3, 0)): 86.5}

    rays = relaxed_shell(dirs, bites)

    assert rays is not None
    for pair, target in bites.items():
        i, j = tuple(pair)
        angle = math.degrees(math.acos(np.clip(rays[i] @ rays[j], -1.0, 1.0)))
        assert angle == pytest.approx(target, abs=FIX_ANGLE_TOL)


def test_square_pyramid_apex_fan_keeps_its_base_planar():
    """An apex bitten to both ends of a basal diagonal (83 deg) is the Berry mode toward trigonal bipyramidal:
    the basal face must fold about that diagonal, not tip as a rigid rectangle, so the four basal rays stay
    coplanar and the shell still reads as the requested square_pyramidal, not the argmin trigonal_bipyramidal.
    """
    dirs = POLYHEDRA["square_pyramidal"].vertex_dirs
    rays = relaxed_shell(dirs, {frozenset((0, 1)): 83.0, frozenset((0, 3)): 83.0})

    assert rays is not None
    base = rays[1:]
    plane_rms = _plane_rms(base.mean(axis=0), base)
    assert plane_rms < 1e-6, "the basal face folded instead of staying planar"

    ranked = rank_shapes(rays)
    assert ranked[0][1] == "square_pyramidal", ranked[:2]
