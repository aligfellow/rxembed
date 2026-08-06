"""Test the Constraints struct and fix/constrain resolution."""

from dataclasses import fields

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.constraints import Constraints, SphereRecipe, add_distance, compose, match, resolve_core, template_to_fix


def _mol(smiles="CCO", seed=1):
    """Ethanol (heavy atoms 0=C, 1=C, 2=O) with a conformer, or any SMILES."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def _populated():
    """A `Constraints` with every field non-empty, so a dropped field is detectable."""
    c = Constraints(
        planes=[((0, 1, 2), (3, 4, 5), 3.6)],
        coplanar=[(0, 1, 2, 3, 180.0, 45.0)],
        frozen={7, 8},
        contacts=(frozenset({(0, 1)}), frozenset({(0, 1, 2)})),
        metals={9},
        pulls={(9, 0): 2.1},
        floors={(9, 4): 2.8},
        dg_floors={(9, 4): 2.8},
        shapes=[{1, 2, 3}],
        phantoms=frozenset({12}),
        spheres=(SphereRecipe(9, (0, 4), "SPL", (0, 1), 28, ()),),
        haptic={12: [3, 4, 5, 6, 7]},
    )
    add_distance(c.distances, 0, 1, 1.9, 2.1)
    c.angles[(0, 1, 2)] = (85.0, 95.0)
    assert all(getattr(c, f.name) for f in fields(Constraints)), "a field was left empty: the guard goes blind"
    return c


# ---------------------------------------------------------------------------------------------------------
# the struct: copy / relaxed / compose
# ---------------------------------------------------------------------------------------------------------


def test_copy_carries_every_field():
    c = _populated()
    d = c.copy()
    for f in fields(Constraints):
        assert getattr(d, f.name) == getattr(c, f.name), f"copy() dropped {f.name}"


def test_copy_does_not_alias_mutable_state():
    c = _populated()
    d = c.copy()
    d.distances[(4, 5)] = (1.0, 2.0)
    d.frozen.add(99)
    d.shapes[0].add(42)
    d.planes.append(((9,), (9,), 1.0))
    assert (4, 5) not in c.distances
    assert 99 not in c.frozen
    assert 42 not in c.shapes[0]
    assert len(c.planes) == 1


def test_relaxed_releases_only_the_seeded_contacts():
    c = _populated()
    r = c.relaxed()
    assert (0, 1) not in r.distances
    assert (0, 1, 2) not in r.angles
    assert r.contacts == (frozenset(), frozenset())
    for f in ("coplanar", "metals", "pulls", "floors", "dg_floors", "phantoms", "haptic", "planes", "frozen", "shapes"):
        assert getattr(r, f) == getattr(c, f), f"relaxed() dropped the structural hold {f}"


def test_compose_merges_every_field():
    a = _populated()
    b = Constraints(frozen={20}, metals={21}, phantoms=frozenset({22}), planes=[((9,), (9,), 1.0)])
    add_distance(b.distances, 4, 5, 1.0, 2.0)
    m = compose(a, b)
    assert m.frozen == {7, 8, 20}
    assert m.metals == {9, 21}
    assert m.phantoms == frozenset({12, 22})
    assert len(m.planes) == 2  # concatenated, never de-duplicated
    assert m.distances[(0, 1)] == (1.9, 2.1)
    assert m.distances[(4, 5)] == (1.0, 2.0)


def test_compose_does_not_mutate_its_inputs():
    a, b = _populated(), Constraints(frozen={20})
    before = {f.name: getattr(a, f.name) for f in fields(Constraints)}
    compose(a, b)
    for name, v in before.items():
        assert getattr(a, name) == v, f"compose mutated input field {name}"


def test_compose_distance_is_last_wins():
    a, b = Constraints(), Constraints()
    add_distance(a.distances, 0, 1, 1.9, 2.1)
    add_distance(b.distances, 0, 1, 2.5, 2.7)
    assert compose(a, b).distances[(0, 1)] == (2.5, 2.7)


def test_compose_keeps_strict_wall_and_full_relief():
    a = Constraints(floors={(9, 4): 2.8}, dg_floors={(9, 4): 2.8})
    b = Constraints(floors={(9, 4): 3.1}, dg_floors={(9, 4): 3.1})
    for one, two in ((a, b), (b, a)):
        assert compose(one, two).floors[(9, 4)] == 3.1
        assert compose(one, two).dg_floors[(9, 4)] == 2.8


@pytest.mark.parametrize(
    ("field_name", "one", "two"),
    [("pulls", {(9, 0): 2.1}, {(9, 0): 2.4}), ("haptic", {12: [1, 2]}, {12: [3, 4]})],
    ids=["pulls", "haptic"],
)
def test_compose_rejects_constraint_collision(field_name, one, two):
    a, b = Constraints(**{field_name: one}), Constraints(**{field_name: two})
    with pytest.raises(ValueError, match=field_name):
        compose(a, b)


# ---------------------------------------------------------------------------------------------------------
# fix: a rigid hold; own coordinates, explicit coordinates, or exact numbers
# ---------------------------------------------------------------------------------------------------------


def test_fix_list_holds_own_coords():
    m = _mol()
    cons, ref = resolve_core(m, fix=[0, 1, 2], has_geometry=True)
    assert cons.frozen == {0, 1, 2}
    assert set(ref) == {0, 1, 2}
    pos = m.GetConformer().GetPositions()
    for i in (0, 1, 2):
        assert np.allclose(ref[i], pos[i])
    assert set(cons.distances) == {(0, 1), (0, 2), (1, 2)}  # C(3,2) shape windows
    for (i, j), (lo, hi) in cons.distances.items():
        assert lo <= np.linalg.norm(pos[i] - pos[j]) <= hi
    assert cons.contacts == (frozenset(), frozenset())  # a graft is structural; nothing releasable
    assert cons.relaxed().distances == cons.distances


def test_fix_list_needs_geometry():
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    with pytest.raises(ValueError, match="own coordinates"):
        resolve_core(m, fix=[0, 1, 2], has_geometry=False)


def test_explicit_fix_uses_given_coordinates():
    m = _mol()
    coords = {0: (0.0, 0.0, 0.0), 1: (1.5, 0.0, 0.0), 2: (1.5, 1.4, 0.0)}
    cons, ref = resolve_core(m, fix=coords, has_geometry=True)
    assert cons.frozen == {0, 1, 2}
    for i, c in coords.items():
        assert np.allclose(ref[i], c)
    lo, hi = cons.distances[(0, 1)]
    assert lo <= 1.5 <= hi


def test_numeric_fix_is_tight_and_nonreleasable():
    m = _mol()
    cons, ref = resolve_core(m, fix={(0, 2): 2.0, (0, 1, 2): 109.5}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((1.98, 2.02))
    assert cons.angles[(0, 1, 2)] == pytest.approx((107.5, 111.5))
    assert set(cons.distances) == {(0, 2)}
    assert cons.frozen == set()
    assert ref == {}
    assert cons.contacts == (frozenset(), frozenset())


# ---------------------------------------------------------------------------------------------------------
# constrain: a soft, releasable window
# ---------------------------------------------------------------------------------------------------------


def test_constrain_number_creates_releasable_window():
    m = _mol()
    cons, ref = resolve_core(m, constrain={(0, 2): 2.8, (0, 1, 2): 120.0}, has_geometry=True)
    assert cons.distances[(0, 2)] == pytest.approx((2.7, 2.9))  # +/-0.1 A, five times the fix pad
    assert cons.angles[(0, 1, 2)] == pytest.approx((115.0, 125.0))  # +/-5 deg
    assert ref == {}
    assert cons.contacts == (frozenset({(0, 2)}), frozenset({(0, 1, 2)}))
    assert cons.relaxed().distances == {}
    assert cons.relaxed().angles == {}


def test_explicit_window_is_taken_verbatim_by_either_verb():
    m = _mol()
    assert resolve_core(m, fix={(0, 2): (1.9, 2.1)}, has_geometry=True)[0].distances[(0, 2)] == pytest.approx(
        (1.9, 2.1)
    )
    assert resolve_core(m, constrain={(0, 2): (2.6, 3.0)}, has_geometry=True)[0].distances[(0, 2)] == pytest.approx(
        (2.6, 3.0)
    )


def test_constrain_ring_pair_becomes_a_pi_stack_plane():
    m = _mol("c1ccccc1.c1ccccc1")
    ra, rb = (tuple(r) for r in m.GetRingInfo().AtomRings()[:2])
    cons, _ = resolve_core(m, constrain={(ra, rb): 3.7}, has_geometry=True)
    assert cons.planes == [(ra, rb, 3.7)]


# ---------------------------------------------------------------------------------------------------------
# refusals: each is a spec that used to be accepted and then produce a wrong answer
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"constrain": {("[OX2]", "[CH3]"): (2.6, 3.0)}}, "index-driven"),
        ({"fix": ["[OX2]"]}, "index"),
        ({"fix": {(0, 999): 2.0}}, "out of range"),
        ({"fix": {(0, 1, 2): 400.0}}, "angle in 0-180"),
        ({"fix": {(0, 1, 2): -30.0}}, "angle in 0-180"),
        ({"fix": {(0, 0): 2.0}}, "same atom twice"),
    ],
    ids=["smarts-key", "smarts-in-list", "out-of-range", "angle-too-big", "angle-negative", "same-atom"],
)
def test_resolve_rejects_invalid_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_core(_mol(), has_geometry=True, **spec)


def test_one_key_under_both_verbs_is_refused():
    with pytest.raises(ValueError, match="two contradictory intents"):
        resolve_core(_mol("CCCl"), fix={(1, 2): 2.5}, constrain={(2, 1): (1.7, 1.8)}, has_geometry=True)


def test_window_inside_the_fixed_core_is_dropped_loudly(caplog):
    coords = {i: (float(i), 0.0, 0.0) for i in (0, 1, 2)}  # collinear, 1.0 A apart
    with caplog.at_level("WARNING", logger="rxembed"):
        cons, _ref = resolve_core(_mol("CCCl"), fix=coords, constrain={(1, 2): 3.4}, has_geometry=True)
    assert "inside the fixed core" in caplog.text
    assert cons.distances[(1, 2)] == pytest.approx((0.95, 1.05)), "the graft's own shape window is back"
    assert cons.contacts == (frozenset(), frozenset()), "a dropped window must not stay releasable"


# ---------------------------------------------------------------------------------------------------------
# advisories: a warning nobody can act on is noise, so the quiet cases are asserted too
# ---------------------------------------------------------------------------------------------------------


def test_fewer_than_three_graft_atoms_warn(caplog):
    m = _mol()
    with caplog.at_level("WARNING", logger="rxembed"):
        resolve_core(m, fix=[0, 1], has_geometry=True)
    assert "orientation needs at least 3" in caplog.text


def test_two_distances_warn_free_angle(caplog):
    m = _mol()
    with caplog.at_level("WARNING", logger="rxembed"):
        resolve_core(m, fix={(0, 1): 1.5, (1, 2): 1.4}, has_geometry=True)
    assert any("no angle is fixed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("fix", "why"),
    [
        ({(0, 1): 1.5, (1, 2): 1.4, (0, 1, 2): 109.0}, "the angle IS fixed"),
        ({(0, 1): 1.5, (1, 2): 1.5, (2, 3): 1.5}, "a >3-atom web is deliberate, not the ambiguous case"),
    ],
    ids=["angle-given", "rich-network"],
)
def test_determined_network_does_not_warn(caplog, fix, why):
    m = _mol("CCCC")
    with caplog.at_level("WARNING", logger="rxembed"):
        resolve_core(m, fix=fix, has_geometry=True)
    assert not any("no angle is fixed" in r.getMessage() for r in caplog.records), why


def test_info_echo_names_atoms_by_element_and_index(caplog):
    m = _mol()  # ethanol: 0=C, 1=C, 2=O
    with caplog.at_level("INFO", logger="rxembed"):
        resolve_core(m, fix={(0, 2): 2.0}, has_geometry=True)
    echo = " ".join(r.getMessage() for r in caplog.records if r.getMessage().startswith("resolve:"))
    assert "C0" in echo
    assert "O2" in echo
    assert "1.98-2.02" in echo


def test_match_rejects_ambiguous_pattern():
    mol = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1CCl"))
    assert len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl]"))) == 2, "the fixture must be ambiguous"
    with pytest.raises(ValueError, match="matched 2 times"):
        match(mol, "[Cl]")

    assert match(mol, "[CH2]Cl"), "an unambiguous pattern still resolves"
    with pytest.raises(ValueError, match="matched nothing"):
        match(mol, "[Br]")
    with pytest.raises(ValueError, match="did not parse"):
        match(mol, "[not a smarts")


def test_symmetric_template_requires_atom_map():
    reference = _mol("Cc1ccccc1")
    target = _mol("CCc1ccccc1")

    with pytest.raises(ValueError, match=r"symmetry-equivalent.*explicit"):
        template_to_fix((reference, "c1ccccc1"), target=target)
