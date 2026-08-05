"""`mechanisms.py`: each mechanism writes the terms its own `Constraints` field claims, and nothing else.

`MECHANISM_ORDER` is the whole of "how a constraint becomes DG bounds and FF terms". These drive each member
directly, against a hand-built `Constraints` and a force field that records what it is asked for, so a
mechanism that starts writing an unasked-for term, or stops writing one it was asked for, fails here rather
than as a geometry difference three stages later.

The load-bearing assertion is the NEGATIVE one: with its field empty, a field-driven mechanism must write
nothing. That is what makes an organic embed bit-identical under the coordination machinery, and it is why
every metal concern can live in the same struct as an organic one.

The metal-specific mechanisms' positive contracts live with the chemistry that motivated them; `Coplanar` in
`test_metal_donor_orient.py`, `Umbrella` in `test_metal_coordination.py`, `Haptic` in
`test_metal_isomers.py`, `Sp2Planar` / `ConjugationCap` in `tests/pipeline/test_geom_check.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolTransforms

from rxembed import mechanisms as mech_mod
from rxembed.constraints import Constraints


class SpyFF:
    """A force field that records the restraint calls made on it instead of building one."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def kinds(self):
        return sorted({name for name, _a, _k in self.calls})


def _mol(smiles="CC(=O)NC", seed=1):
    """An amide by default: one sp2 carbon and one conjugated O=C-N-C quartet."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _ctx(mol):
    return mech_mod.DGContext(mol=mol, bm=rdDistGeom.GetMoleculeBoundsMatrix(mol))


def _ff_terms(mech, cons, mol, fc=1.0):
    ff = SpyFF()
    mech.ff_terms(ff, cons, mol.GetConformer(0), fc)
    return ff


# `Sp2Planar` and `ConjugationCap` are not constraints at all but repairs to UFF (a 3-neighbour sp2 carbon that
# pyramidalises, a conjugated plane that twists), so they perceive from the molecule and fire unconditionally.
# Giving them a field would mean filling it on every path that can reach the relax.
_FF_REPAIRS = {"Sp2Planar", "ConjugationCap"}
_FIELD_DRIVEN = [m for m in mech_mod.MECHANISM_ORDER if type(m).__name__ not in _FF_REPAIRS]


# ---------------------------------------------------------------------------------------------------------
# the negative contract: a field-driven mechanism is silent off an empty struct
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mech", _FIELD_DRIVEN, ids=lambda m: type(m).__name__)
def test_a_field_driven_mechanism_is_silent_off_an_empty_struct(mech):
    mol = _mol()
    assert _ff_terms(mech, Constraints(), mol).calls == [], f"{type(mech).__name__} wrote an FF term unasked"

    ctx = _ctx(mol)
    before = ctx.bm.copy()
    for hook in (mech.dg_windows, mech.dg_relief, mech.dg_post):
        hook(Constraints(), ctx)
    assert ctx.pairs == {}, f"{type(mech).__name__} proposed a window off an empty struct"
    assert np.array_equal(ctx.bm, before), f"{type(mech).__name__} edited the matrix off an empty struct"


@pytest.mark.parametrize("name", sorted(_FF_REPAIRS))
def test_an_ff_repair_fires_on_the_molecule_not_on_a_field(name):
    mech = next(m for m in mech_mod.MECHANISM_ORDER if type(m).__name__ == name)
    mol = _mol()
    assert _ff_terms(mech, Constraints(), mol).calls, f"{name} must fire on a bare struct"
    assert _ff_terms(mech, Constraints(metals={0}), mol).calls == [], (
        f"{name} must stand down on a metal system: the Li surrogate defeats the sp2 perception it relies on"
    )


# ---------------------------------------------------------------------------------------------------------
# the positive contract: a populated field writes the term it claims
# ---------------------------------------------------------------------------------------------------------


def test_every_constant_here_is_calibrated_against_one_penalty_law():
    mol = _mol("CCCC")
    conf = mol.GetConformer()
    d0 = rdMolTransforms.GetBondLength(conf, 0, 1)

    def penalty(k, dev):
        bare = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
        held = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
        held.AddDistanceConstraint(0, 1, d0 - dev, d0 - dev, k)
        held.Initialize()
        return held.CalcEnergy() - bare.CalcEnergy()

    for k in (100.0, 400.0):
        for dev in (0.1, 0.2):
            assert penalty(k, dev) == pytest.approx(0.5 * k * dev**2, rel=1e-6), (
                f"the penalty law moved at k={k}, dev={dev}: every constant in mechanics.py is calibrated on it"
            )


def test_distance_seeds_the_pending_pairs_and_walls_the_pair_in_the_field():
    mol = _mol()
    cons = Constraints(distances={(0, 3): (2.0, 2.4)})

    ctx = _ctx(mol)
    mech_mod.Distance().dg_windows(cons, ctx)
    assert ctx.pairs[(0, 3)] == (2.0, 2.4)

    ff = _ff_terms(mech_mod.Distance(), cons, mol)
    assert ff.kinds() == ["AddDistanceConstraint"]
    _name, args, _kw = ff.calls[0]
    assert args[:4] == (0, 3, 2.0, 2.4)


def test_pull_collapses_its_window_to_a_point_which_is_what_makes_it_a_spring():
    mol = _mol()
    ff = _ff_terms(mech_mod.Pull(), Constraints(pulls={(0, 3): 2.1}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == args[3] == 2.1


def test_floor_is_one_sided():
    mol = _mol()
    ff = _ff_terms(mech_mod.Floor(), Constraints(floors={(0, 3): 2.6}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == 2.6
    assert args[3] >= 1e3, f"a floor's upper bound must stand in for infinity, got {args[3]}"


def test_frozen_pins_points_rather_than_restraining_them():
    mol = _mol()
    ff = _ff_terms(mech_mod.Frozen(), Constraints(frozen={0, 1, 2}), mol)
    assert ff.kinds() == ["AddFixedPoint"]
    assert sorted(a[0] for _n, a, _k in ff.calls) == [0, 1, 2]


def test_angle_writes_a_uff_angle_wall_and_a_matrix_diagonal():
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})

    ctx = _ctx(mol)
    mech_mod.Angle().dg_windows(cons, ctx)
    assert (0, 3) in ctx.pairs, "an angle must state its end-atom diagonal in the matrix"

    ff = _ff_terms(mech_mod.Angle(), cons, mol)
    assert ff.kinds() == ["UFFAddAngleConstraint"]
    _name, args, _kw = ff.calls[0]
    assert args[:3] == (0, 1, 3)


def test_the_angle_wall_is_a_stated_number_the_ladder_cannot_raise():
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})
    assert _ff_terms(mech_mod.Angle(), cons, mol, fc=1.0).calls[0][1][-1] == mech_mod.ANGLE_FC
    assert _ff_terms(mech_mod.Angle(), cons, mol, fc=0.1).calls[0][1][-1] == pytest.approx(0.1 * mech_mod.ANGLE_FC)
    assert _ff_terms(mech_mod.Angle(), cons, mol, fc=100.0).calls[0][1][-1] == mech_mod.ANGLE_FC, (
        "the ladder must not be able to escalate the angle wall"
    )


def test_a_releasable_contact_is_walled_more_softly_than_a_stated_one():
    mol = _mol("c1ccccc1.c1ccccc1")
    cons = Constraints(
        distances={(0, 3): (2.0, 2.4), (1, 4): (2.0, 2.4)},
        contacts=(frozenset({(0, 3)}), frozenset()),
    )
    emitted = {tuple(a[:2]): a[-1] for _n, a, _kw in _ff_terms(mech_mod.Distance(), cons, mol).calls}
    assert emitted[(1, 4)] == pytest.approx(mech_mod.PIN_FC), "a structural window is held at PIN"
    softened = mech_mod.PIN_FC * mech_mod.RELEASABLE_FC_SCALE
    assert emitted[(0, 3)] == pytest.approx(softened), "a releasable one is softened"


def test_plane_holds_a_pi_stack_by_cross_ring_distances():
    mol = _mol("c1ccccc1.c1ccccc1")
    ra, rb = (tuple(r) for r in mol.GetRingInfo().AtomRings()[:2])
    cons = Constraints(planes=[(ra, rb, 3.6)])

    ctx = _ctx(mol)
    mech_mod.Plane().dg_windows(cons, ctx)
    assert ctx.pairs, "a stack must state cross-ring windows in the matrix"
    assert all(i in ra and j in rb for i, j in ctx.pairs), "a stack window must span the two rings"

    ff = _ff_terms(mech_mod.Plane(), cons, mol)
    assert ff.kinds() == ["AddDistanceConstraint"]


# ---------------------------------------------------------------------------------------------------------
# MECHANISM_ORDER itself: the one place order is stated, and the order is load-bearing
# ---------------------------------------------------------------------------------------------------------


def test_distance_precedes_angle_because_angle_reads_the_windows_distance_seeds():
    order = [type(m).__name__ for m in mech_mod.MECHANISM_ORDER]
    assert order.index("Distance") < order.index("Angle")


# ---------------------------------------------------------------------------------------------------------
# the force constants: what a number in this module actually costs
# ---------------------------------------------------------------------------------------------------------


def _populated(mol):
    """A `Constraints` carrying every field the distance-side writers read, on `mol`'s own indices."""
    ra, rb = (tuple(r) for r in mol.GetRingInfo().AtomRings()[:2])
    return Constraints(
        distances={(0, 3): (2.0, 2.4)},
        angles={(0, 1, 3): (100.0, 120.0)},
        planes=[(ra, rb, 3.6)],
        frozen={5},
        pulls={(1, 2): 2.1},
        floors={(2, 4): 2.6},
    )


def test_only_the_spring_ignores_the_stiffness_ladder():
    mol = _mol("c1ccccc1.c1ccccc1")
    cons = _populated(mol)

    emitted: dict[str, dict[float, float]] = {}
    for fc in (1.0, 100.0):
        for mech in mech_mod.MECHANISM_ORDER:
            for name, args, _kw in _ff_terms(mech, cons, mol, fc=fc).calls:
                if name.endswith("Constraint"):
                    emitted.setdefault(type(mech).__name__, {})[fc] = args[-1]

    rode = {m for m, by_fc in emitted.items() if len(set(by_fc.values())) > 1}
    assert "Pull" in emitted, "the fixture must exercise Pull for this to mean anything"
    assert "Pull" not in rode, "Pull rode the stiffness ladder; it is the spring and must stay put"
    assert {"Distance", "Floor", "Plane"} <= rode, f"a wall must scale with the caller's stiffness, got {rode}"
