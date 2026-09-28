"""Test the Constraints struct and fix/constrain resolution."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.constraints import (
    Constraints,
    compose,
    compose_soft,
    constraint_value,
    resolve_atom,
    resolve_core,
    template_to_fix,
)


def _mol(smiles="CCO", seed=1):
    """Ethanol (heavy atoms 0=C, 1=C, 2=O) with a conformer, or any SMILES."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


# ---------------------------------------------------------------------------------------------------------
# the struct: copy / relaxed / compose
# ---------------------------------------------------------------------------------------------------------


def test_angle_preferences_merge_once_and_release_with_their_contact():
    key = (0, 1, 2)
    base = Constraints(angles={key: (80.0, 140.0)}, pulls={key: 120.0, (9, 0): 2.1})
    merged = compose(base, Constraints(pulls={key[::-1]: 120.0}))
    assert merged.pulls == base.pulls
    with pytest.raises(ValueError, match="conflicting pulls"):
        compose(base, Constraints(pulls={key[::-1]: 110.0}))
    merged.contacts = (frozenset(), frozenset({key}))
    assert merged.relaxed().pulls == {(9, 0): 2.1}
    assert merged.pulls == base.pulls


@pytest.mark.parametrize("value", [181.0])
def test_angle_preferences_reject_invalid_degrees(value):
    with pytest.raises(ValueError, match="finite angle"):
        compose(Constraints(pulls={(0, 1, 2): value}))


def test_compose_soft_owns_each_dihedral_by_its_central_bond():
    base_key, incoming_key, other_axis_key = (0, 1, 2, 3), (4, 1, 2, 5), (1, 2, 3, 6)
    fixed = Constraints(dihedrals={base_key: (-62.0, -58.0)}, fixed={base_key: (-60.0, -60.0)})
    incoming = Constraints(
        dihedrals={incoming_key: (80.0, 100.0), other_axis_key: (80.0, 100.0)},
        contacts=(frozenset(), frozenset({incoming_key, other_axis_key})),
    )

    merged = compose_soft(fixed, incoming)
    assert set(merged.dihedrals) == {base_key, other_axis_key}, (
        "a fixed torsion must protect only its own bond axis, not a neighbouring one"
    )

    soft = fixed.copy(fixed={}, contacts=(frozenset(), frozenset({base_key})))
    with pytest.raises(ValueError, match="state each degree of freedom once"):
        compose_soft(soft, incoming)


def test_constraint_value_rejects_invalid_real_and_haptic_indices():
    positions = np.zeros((3, 3))

    assert constraint_value(positions, (0, -1)) is None
    assert constraint_value(positions, (0, 9), {9: [0, 3]}) is None


def test_numeric_fix_wins_over_approximate_builder_in_either_order():
    key, angle = (0, 1), (0, 1, 2)
    fixed = Constraints(fixed={key: (2.0, 2.0), angle: (110.0, 110.0)})
    approximate = Constraints(distances={key: (2.3, 2.5)}, angles={angle[::-1]: (90.0, 100.0)}, pulls={key: 2.4})

    for parts in ((fixed, approximate), (approximate, fixed)):
        merged = compose(*parts)
        assert merged.distances[key] == (1.98, 2.02)
        assert merged.angles[angle] == (108.0, 112.0)
        assert angle[::-1] not in merged.angles
        assert key not in merged.pulls

    window = Constraints(distances={key: (1.9, 2.1)}, fixed={key: (1.9, 2.1)})
    for parts in ((window, approximate), (approximate, window)):
        assert key not in compose(*parts).pulls, "an approximate builder added a target to a fixed range"


# ---------------------------------------------------------------------------------------------------------
# fix: a rigid hold; own coordinates, explicit coordinates, or exact numbers
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# constrain: a soft, releasable window
# ---------------------------------------------------------------------------------------------------------


def test_constrain_number_creates_releasable_window():
    m = _mol("CCCC")
    cons, ref = resolve_core(
        m,
        constrain={(0, 2): 2.8, (0, 1, 2): 120.0, (0, 1, 2, 3): -60.0},
        has_geometry=True,
    )
    assert cons.distances[(0, 2)] == pytest.approx((2.7, 2.9))  # +/-0.1 A, five times the fix pad
    assert cons.angles[(0, 1, 2)] == pytest.approx((115.0, 125.0))  # +/-5 deg
    assert cons.dihedrals[(0, 1, 2, 3)] == pytest.approx((-65.0, -55.0))
    assert ref == {}
    assert cons.contacts == (
        frozenset({(0, 2)}),
        frozenset({(0, 1, 2), (0, 1, 2, 3)}),
    )
    assert cons.relaxed().distances == {}
    assert cons.relaxed().angles == {}
    assert cons.relaxed().dihedrals == {}


def test_constrain_ring_pair_becomes_a_pi_stack_plane():
    m = _mol("c1ccccc1.c1ccccc1")
    ra, rb = (tuple(r) for r in m.GetRingInfo().AtomRings()[:2])
    cons, _ = resolve_core(m, constrain={(ra, rb): 3.7}, has_geometry=True)
    assert cons.planes == [(ra, rb, 3.7)]
    assert cons.relaxed().planes == []


# ---------------------------------------------------------------------------------------------------------
# refusals: each is a spec that used to be accepted and then produce a wrong answer
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"fix": {(0, 1, 2, 3): (0.0, 361.0)}}, "no wider than 360"),
    ],
    ids=[
        "dihedral-too-wide",
    ],
)
def test_resolve_rejects_invalid_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_core(_mol(), has_geometry=True, **spec)


@pytest.mark.parametrize(
    "fix",
    [
        {(0, 1, 2): 100.0, (2, 1, 0): 110.0},
    ],
    ids=["angle"],
)
def test_reversed_numeric_fix_conflict_is_refused(fix):
    with pytest.raises(ValueError, match="conflicting values"):
        resolve_core(_mol("CCCC"), fix=fix, has_geometry=True)


@pytest.mark.parametrize(
    ("one", "two", "expected"),
    [
        (180.0, -180.0, (180.0, 180.0)),
    ],
    ids=["half-turn"],
)
def test_periodic_dihedral_aliases_compose(one, two, expected):
    mol = _mol("CCCC")
    a = resolve_core(mol, fix={(0, 1, 2, 3): one}, has_geometry=True)[0]
    b = resolve_core(mol, fix={(3, 2, 1, 0): two}, has_geometry=True)[0]
    assert compose(a, b).fixed[(0, 1, 2, 3)] == expected


@pytest.mark.parametrize("door", ["fix"])
def test_window_inside_the_fixed_core_is_dropped_loudly(caplog, door):
    """A distance window inside a coordinate graft yields to the graft, from either door."""
    coords = {i: (float(i), 0.0, 0.0) for i in (0, 1, 2)}  # collinear, 1.0 A apart
    fix = {**coords, (1, 2): 3.4} if door == "fix" else coords
    constrain = None if door == "fix" else {(1, 2): 3.4}
    with caplog.at_level("WARNING", logger="rxembed"):
        cons, _ref = resolve_core(_mol("CCCl"), fix=fix, constrain=constrain, has_geometry=True)
    assert "inside the fixed core" in caplog.text
    assert cons.distances[(1, 2)] == pytest.approx((0.95, 1.05)), "the graft's own shape window is back"
    assert cons.contacts == (frozenset(), frozenset()), "a dropped window must not stay releasable"
    assert (1, 2) not in cons.fixed, "a numeric fix window dropped by the graft must not linger in cons.fixed"


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


def test_resolve_atom_rejects_an_ambiguous_or_unparsable_smarts():
    """2-chlorobenzyl chloride: `[Cl]` names both chlorines, so no single atom may be picked silently."""
    mol = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1CCl"))
    with pytest.raises(ValueError, match="matched 2 times"):
        resolve_atom(mol, "[Cl]")
    with pytest.raises(ValueError, match="did not parse"):
        resolve_atom(mol, "[not a smarts")
    assert resolve_atom(mol, "[Cl]C[c]") == 8


@pytest.mark.parametrize(
    ("reference_smiles", "target_smiles"),
    [("CCO", "Cc1ccccc1")],
    ids=["reference-symmetric"],
)
def test_symmetric_template_requires_atom_map(reference_smiles, target_smiles):
    """A SMARTS symmetric on either the reference or the target alone still needs an explicit map."""
    reference, target = _mol(reference_smiles), _mol(target_smiles)

    with pytest.raises(ValueError, match=r"symmetry-equivalent.*explicit"):
        template_to_fix((reference, "[CX4]-[#6]"), target=target)
