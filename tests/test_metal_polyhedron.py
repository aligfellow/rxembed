"""`metal_polyhedron`: the POLYHEDRA records, their codes/aliases, and the table-derived predicates.

Table facts only: no embed, no force field. What a record can get wrong on its own: an angle row that
disagrees with the vertex directions it indexes, a code that collides, a shape that is not distinct from its
CN neighbours.
"""

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
    describe,
    geometries_for_cn,
    resolve_geometry,
)


def _spectrum(name):
    """The sorted vertex-metal-vertex angle spectrum of a record: what `classify_geometry` matches on."""
    return sorted(_vertex_angle(u, v) for u, v in itertools.combinations(POLYHEDRA[name].vertex_dirs, 2))


# --- codes and aliases --------------------------------------------------------------------------------


def test_every_record_resolves_from_its_code_and_agrees_on_its_vertex_count():
    """Each record's 3-letter code resolves to its name, and `cn` is the number of vertices it actually carries.

    One test over the whole table rather than a case per shape: a new row is covered the moment it is written.
    """
    for name, p in POLYHEDRA.items():
        assert resolve_geometry(p.code) == name, f"{p.code} does not resolve to {name}"
        assert n_sites(name) == p.cn == len(p.vertex_dirs), f"{name}: cn {p.cn} is not its vertex count"
        assert name in [q.name for q in geometries_for_cn(p.cn)]


def test_the_alias_table_has_no_collision():
    """Names and codes share one case-insensitive table, so a collision would silently rename a shape."""
    names, codes = [n.lower() for n in POLYHEDRA], [p.code.lower() for p in POLYHEDRA.values() if p.code]
    assert len(codes) == len(set(codes)), "two polyhedra share a 3-letter code"
    assert not set(names) & set(codes), "a polyhedron name is spelled like another's code"
    assert len(_ALIASES) == len(names) + len(codes)


def test_a_code_and_a_name_alias_case_insensitively():
    """A code and a full name are equally acceptable input, in any case: only the long name comes back."""
    for spelling in ("OCT", "oct", " Oct ", "octahedral", "OCTAHEDRAL"):
        assert resolve_geometry(spelling) == "octahedral"
        assert n_sites(spelling) == 6


def test_describe_names_the_shape_unambiguously():
    """`describe`, the one formatter every log line goes through, carries the code and the CN."""
    assert describe("SPL") == "square_planar (SPL, CN 4)"
    assert describe("trigonal_pyramidal") == "trigonal_pyramidal (TPY, CN 3)"
    assert describe("3-coordinate") == "3-coordinate"  # the from_geometry pseudo-name is not invented into a record


# --- the records themselves ---------------------------------------------------------------------------


def test_every_hand_authored_angle_row_is_the_angle_its_own_vertices_subtend():
    """Rows and directions are written independently, so a typo in either desynchronises them silently.

    The DG target would then disagree with the template the isomer machinery (handedness, `classify_geometry`,
    the sphere solver) reads off `vertex_dirs`.
    """
    for name, p in POLYHEDRA.items():
        if p.angles is None:  # CN7 / CN8: no hand-authored rows, `resolved_angles` derives them
            continue
        for i, j, a in p.angles:
            assert abs(a - _vertex_angle(p.vertex_dirs[i], p.vertex_dirs[j])) <= 0.5, f"{name} row {(i, j, a)}"


def test_the_low_cn_shapes_are_not_a_flattened_parent():
    """T-shape / seesaw / trigonal-pyramidal are each angularly distinct from their CN neighbours."""
    assert _spectrum("t_shape") != _spectrum("trigonal_planar")
    for other in ("trigonal_planar", "t_shape"):
        assert _spectrum("trigonal_pyramidal") != _spectrum(other)
    assert _spectrum("seesaw") != _spectrum("tetrahedral")


def test_the_cn3_pyramid_is_a_tetrahedron_minus_a_vertex():
    """Its one angle is `tetrahedral`'s own 109.47°: the record adds no second constant to the table.

    Ammonia's 107° would, and that is a main-group bond-pair/lone-pair number; no census can arbitrate (the
    corpus holds 5 CN3 metal centres, every one planar). `planar=False` is the whole discriminator against
    trigonal_planar, whose spectrum it otherwise shares a continuum with.
    """
    v = np.array(POLYHEDRA["trigonal_pyramidal"].vertex_dirs, float)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    exact = [np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in itertools.combinations(v, 2)]
    assert np.allclose(exact, 109.4712, atol=1e-3), exact  # not `_spectrum`: `_vertex_angle` rounds to a degree
    assert POLYHEDRA["trigonal_pyramidal"].permutations is None  # C3v: the 3 vertices are one orbit
    assert not POLYHEDRA["trigonal_pyramidal"].planar


def test_only_a_flat_based_pyramid_gets_an_umbrella_improper():
    """`umbrella_improper` selects `trigonal_pyramidal` alone, and states that record's own ideal improper.

    The scope is a predicate over the table (metal off its vertex plane + a COPLANAR vertex set), not a shape
    name, so a new record could silently opt itself in. Only a record whose vertices share one plane has a
    single umbrella coordinate; TET/SEE/TBP/SPY/OCT/PBP/SQA are genuinely 3-D and must stay out.

    The angle is read back off the record's own vertices at two bond lengths rather than compared to a copied
    constant: `Umbrella` measures a D-D-D-M dihedral, which is scale-free, and a table value drifting from the
    geometry it names would hold the FF at an angle the record does not describe.
    """
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


def test_every_record_is_structurally_well_formed():
    """Vertex count == CN, every direction non-degenerate, no two vertices in the same place.

    Cheap, and it is the class of defect a template table actually acquires: auditing a sibling project's
    equivalent table turned up a CN7 record with `[0,0,2]` listed twice (the -z vertex lost to a copy-paste)
    and a CN8 record carrying nine vertices. Neither is visible by reading, and both silently poison every
    classification at that coordination number.
    """
    for name, rec in POLYHEDRA.items():
        v = np.array(rec.vertex_dirs, float)
        assert v.shape == (rec.cn, 3), f"{name}: vertex_dirs is {v.shape}, expected ({rec.cn}, 3)"
        norms = np.linalg.norm(v, axis=1)
        assert np.all(norms > 1e-6), f"{name}: a zero-length vertex direction"
        u = v / norms[:, None]
        cos = u @ u.T
        same = [(i, j) for i in range(rec.cn) for j in range(i + 1, rec.cn) if cos[i, j] > 0.999]
        assert not same, f"{name}: vertices {same} point the same way"


def test_every_records_angle_table_covers_the_pairs_it_claims():
    """A hand-authored subset must name real vertices; a derived one must be the complete C(n,2) set."""
    for name, rec in POLYHEDRA.items():
        pairs = {(i, j) for i, j, _a in rec.resolved_angles}
        assert len(pairs) == len(rec.resolved_angles), f"{name}: a vertex pair is stated twice"
        assert all(0 <= i < rec.cn and 0 <= j < rec.cn and i != j for i, j in pairs), f"{name}: a pair is off-record"
        if rec.angles is None:  # derived -> it must be every pair, or the derivation is silently partial
            assert len(pairs) == rec.cn * (rec.cn - 1) // 2, f"{name}: derived table is not all pairs"


def test_the_new_records_are_separable_from_their_neighbours_at_the_same_cn():
    """A record that duplicates one already there makes classification WORSE, not better.

    The whole reason to add a competitor is that four records were alone at their CN and won by default. That
    only helps if the newcomer is actually distinguishable, so every same-CN pair must differ by more than
    the fit floor, or the argmin between them is noise.
    """
    for cn in {p.cn for p in POLYHEDRA.values()}:
        recs = [p for p in POLYHEDRA.values() if p.cn == cn]
        for a, b in itertools.combinations(recs, 2):
            sa, sb = (np.array(sorted(a for _i, _j, a in r.resolved_angles), float) for r in (a, b))
            if len(sa) != len(sb):
                continue
            rms = float(np.sqrt(np.mean((sa - sb) ** 2)))
            assert rms > 5.0, f"{a.name} and {b.name} differ by only {rms:.1f} deg of angle spectrum"
