"""Test polyhedron records, aliases and derived predicates."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

from rxembed.metal_core import n_sites
from rxembed.metal_polyhedron import (
    _ALIASES,
    POLYHEDRA,
    _vertex_angle,
    canonical_slots,
    describe,
    geometries_for_cn,
    point_group,
    resolve_geometry,
    rotation_group,
    seat_properly,
)

# --- codes and aliases --------------------------------------------------------------------------------


def test_records_resolve_and_match_vertex_count():
    for name, p in POLYHEDRA.items():
        assert resolve_geometry(p.code) == name, f"{p.code} does not resolve to {name}"
        assert n_sites(name) == p.cn == len(p.vertex_dirs), f"{name}: cn {p.cn} is not its vertex count"
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
        assert n_sites(spelling) == 6

    assert describe("SPL") == "square_planar (SPL, CN 4)"
    assert describe("3-coordinate") == "3-coordinate"  # the from_geometry pseudo-name is not invented into a record


# --- the records themselves ---------------------------------------------------------------------------


def test_angle_rows_match_vertex_geometry():
    for name, p in POLYHEDRA.items():
        if p.angles is None:  # CN7 / CN8: no hand-authored rows, `resolved_angles` derives them
            continue
        for i, j, a in p.angles:
            assert abs(a - _vertex_angle(p.vertex_dirs[i], p.vertex_dirs[j])) <= 0.5, f"{name} row {(i, j, a)}"


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
    for name, rec in POLYHEDRA.items():
        rot, refl = point_group(tuple(map(tuple, rec.vertex_dirs)))
        assert rot == rotation_group(name), f"{name}: rotation_group is not point_group's proper half"
        assert len(rot) == len(refl) > 0, f"{name}: {len(rot)} rotations against {len(refl)} reflections"
        g = min(refl)
        assert {tuple(g[q[v]] for v in range(rec.cn)) for q in rot} == refl, f"{name}: refl is not g o rot"
        assert (min(refl) == tuple(range(rec.cn))) == rec.planar, f"{name}: planarity disagrees with rot & refl"


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
