"""The `Constraints` IR contract: a field can never be silently dropped by a copy or a merge."""

from dataclasses import fields

import pytest

from rxembed.rdkit_embed.constraints.base import _CLONE, _MERGE, Constraints, add_distance, compose


def _populated():
    """A Constraints with every field non-empty, so a dropped field is detectable."""
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
        haptic={12: [3, 4, 5, 6, 7]},
    )
    add_distance(c.distances, 0, 1, 1.9, 2.1)
    c.angles[(0, 1, 2)] = (85.0, 95.0)
    return c


def test_registries_cover_every_field():
    """A field added to the dataclass without a clone/merge policy must fail loudly, not default silently."""
    names = {f.name for f in fields(Constraints)}
    assert set(_CLONE) == names
    assert set(_MERGE) == names


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
    d.shapes[0].add(42)  # the one field whose ELEMENTS are mutable
    d.planes.append(((9,), (9,), 1.0))
    assert (4, 5) not in c.distances
    assert 99 not in c.frozen
    assert 42 not in c.shapes[0]
    assert len(c.planes) == 1


def test_copy_overrides_replace_outright():
    c = _populated()
    d = c.copy(distances={}, frozen={1})
    assert d.distances == {}
    assert d.frozen == {1}
    assert d.coplanar == c.coplanar  # untouched fields still ride along


def test_relaxed_releases_only_seeded_contacts_and_keeps_structure():
    c = _populated()
    r = c.relaxed()
    assert (0, 1) not in r.distances  # the seeded grip is released
    assert (0, 1, 2) not in r.angles
    assert r.contacts == (frozenset(), frozenset())  # provenance cleared
    for f in ("coplanar", "metals", "pulls", "floors", "dg_floors", "phantoms", "haptic", "planes", "frozen"):
        assert getattr(r, f) == getattr(c, f), f"relaxed() dropped the structural hold {f}"


def test_compose_merges_every_field():
    a = _populated()
    b = Constraints(frozen={20}, metals={21}, phantoms=frozenset({22}), planes=[((9,), (9,), 1.0)])
    add_distance(b.distances, 4, 5, 1.0, 2.0)
    m = compose(a, b)
    assert m.frozen == {7, 8, 20}
    assert m.metals == {9, 21}
    assert m.phantoms == frozenset({12, 22})
    assert len(m.planes) == 2  # concat, never de-duplicated
    assert m.distances[(0, 1)] == (1.9, 2.1)
    assert m.distances[(4, 5)] == (1.0, 2.0)


def test_compose_does_not_mutate_its_inputs():
    a, b = _populated(), Constraints(frozen={20})
    before = {f.name: getattr(a, f.name) for f in fields(Constraints)}
    compose(a, b)
    for name, v in before.items():
        assert getattr(a, name) == v, f"compose mutated input field {name}"


def test_compose_distance_is_last_wins():
    """A spec landing on a structural hold is a deliberate user override, not a conflict."""
    a, b = Constraints(), Constraints()
    add_distance(a.distances, 0, 1, 1.9, 2.1)
    add_distance(b.distances, 0, 1, 2.5, 2.7)
    assert compose(a, b).distances[(0, 1)] == (2.5, 2.7)


def test_compose_wall_takes_the_stricter_claim_and_relief_the_fullest():
    """`floors` is a WALL (max, the strictest guard); `dg_floors` is a RELIEF of a phantom (min, the fullest).

    Opposite directions on purpose — see `compose`. A max on `dg_floors` keeps the larger *phantom* carbon-vdW
    floor rather than the better-informed physical one, which makes the relief a silent no-op on exactly the
    pairs two sources both claim (a substrate tethered to a real donor).
    """
    a = Constraints(floors={(9, 4): 2.8}, dg_floors={(9, 4): 2.8})
    b = Constraints(floors={(9, 4): 3.1}, dg_floors={(9, 4): 3.1})
    for one, two in ((a, b), (b, a)):  # both order-independent, unlike a last-wins update
        assert compose(one, two).floors[(9, 4)] == 3.1
        assert compose(one, two).dg_floors[(9, 4)] == 2.8


@pytest.mark.parametrize(
    ("field_name", "one", "two"),
    [("pulls", {(9, 0): 2.1}, {(9, 0): 2.4}), ("haptic", {12: [1, 2]}, {12: [3, 4]})],
)
def test_compose_refuses_an_unresolvable_collision(field_name, one, two):
    """Two harmonic targets on one pair, or two faces claiming one reserved phantom index, cannot be merged."""
    a, b = Constraints(**{field_name: one}), Constraints(**{field_name: two})
    with pytest.raises(ValueError, match=field_name):
        compose(a, b)
