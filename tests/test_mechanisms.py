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
from rdkit.Chem import rdDistGeom

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
    """No force-field term, no pending pair, no matrix edit; on either driver."""
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
    """The two UFF repairs fire off a bare struct, and stand down on a metal system."""
    mech = next(m for m in mech_mod.MECHANISM_ORDER if type(m).__name__ == name)
    mol = _mol()
    assert _ff_terms(mech, Constraints(), mol).calls, f"{name} must fire on a bare struct"
    assert _ff_terms(mech, Constraints(metals={0}), mol).calls == [], (
        f"{name} must stand down on a metal system: the Li surrogate defeats the sp2 perception it relies on"
    )


# ---------------------------------------------------------------------------------------------------------
# the positive contract: a populated field writes the term it claims
# ---------------------------------------------------------------------------------------------------------


def test_distance_seeds_the_pending_pairs_and_walls_the_pair_in_the_field():
    """`Distance` states its window in `ctx.pairs` for later writers and as a UFF distance wall."""
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
    """`Pull` is the only spring: lo == hi, where every sibling passes a flat-bottomed window."""
    mol = _mol()
    ff = _ff_terms(mech_mod.Pull(), Constraints(pulls={(0, 3): 2.1}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == args[3] == 2.1


def test_floor_is_one_sided():
    """`Floor` states a minimum and an effectively infinite ceiling: a wall, not a window."""
    mol = _mol()
    ff = _ff_terms(mech_mod.Floor(), Constraints(floors={(0, 3): 2.6}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == 2.6
    assert args[3] >= 1e3, f"a floor's upper bound must stand in for infinity, got {args[3]}"


def test_frozen_pins_points_rather_than_restraining_them():
    """A frozen core is held with zero degrees of freedom; `AddFixedPoint`, never a stiff spring."""
    mol = _mol()
    ff = _ff_terms(mech_mod.Frozen(), Constraints(frozen={0, 1, 2}), mol)
    assert ff.kinds() == ["AddFixedPoint"]
    assert sorted(a[0] for _n, a, _k in ff.calls) == [0, 1, 2]


def test_angle_writes_a_uff_angle_wall_and_a_matrix_diagonal():
    """`Angle` states the 1-3 distance in the matrix and the angle itself in the force field."""
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})

    ctx = _ctx(mol)
    mech_mod.Angle().dg_windows(cons, ctx)
    assert (0, 3) in ctx.pairs, "an angle must state its end-atom diagonal in the matrix"

    ff = _ff_terms(mech_mod.Angle(), cons, mol)
    assert ff.kinds() == ["UFFAddAngleConstraint"]
    _name, args, _kw = ff.calls[0]
    assert args[:3] == (0, 1, 3)


def test_an_ordinary_angle_force_constant_passes_through_but_an_over_stiff_one_is_capped():
    """An over-stiff angle distorts a rigid or bidentate framework, so the caller's fc is a request, not a value."""
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})
    assert _ff_terms(mech_mod.Angle(), cons, mol, fc=1.0).calls[0][1][-1] == 1.0
    assert _ff_terms(mech_mod.Angle(), cons, mol, fc=1e9).calls[0][1][-1] < 1e9


def test_plane_holds_a_pi_stack_by_cross_ring_distances():
    """`Plane` turns a (ring_a, ring_b, separation) record into cross-ring distance walls."""
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


def test_every_mechanism_appears_exactly_once():
    """A duplicate would double every term it writes; a missing one would silently drop a whole field."""
    names = [type(m).__name__ for m in mech_mod.MECHANISM_ORDER]
    assert len(names) == len(set(names)), f"duplicated mechanism: {names}"


def test_distance_precedes_angle_because_angle_reads_the_windows_distance_seeds():
    """`Angle.dg_windows` sizes its diagonal from `ctx.pairs`, which `Distance` fills; order is not free."""
    order = [type(m).__name__ for m in mech_mod.MECHANISM_ORDER]
    assert order.index("Distance") < order.index("Angle")


# ---------------------------------------------------------------------------------------------------------
# the force constants: what a number in this module actually costs
# ---------------------------------------------------------------------------------------------------------


def _uff(mol):
    from rdkit.Chem import rdForceFieldHelpers  # as `relax.py` does; AllChem has no stubs for it

    return rdForceFieldHelpers.UFFGetMoleculeForceField(mol)


_PENALTY_LAW = {  # kind -> (is the penalty halved, probes as (fc, deviation))
    "distance": (True, ((1.0, 0.1), (100.0, 0.2))),  # deviation in Angstrom
    "torsion": (False, ((1.0, 5.0), (10.0, 30.0))),  # deviation in DEGREES
}


@pytest.mark.parametrize("kind", sorted(_PENALTY_LAW), ids=lambda k: f"{k}-penalty-law")
def test_a_force_constant_means_what_this_module_assumes_it_means(kind):
    """RDKit's two constraint families do not share a penalty law, and every constant here rests on which.

    Measured: a distance restraint costs ``0.5 * fc * dev**2`` with dev in Angstrom, a torsion one costs
    ``fc * dev**2`` with dev in degrees and no half. That asymmetry is why 1e4 (a distance) and 3 (a torsion)
    are not the mismatch they look like, and it is why `_COPLANAR_FC` = 10 is 9000 kcal/mol at 30 degrees.
    If RDKit changes either law, every number in this module silently changes meaning and nothing else in the
    suite would notice.
    """
    from rdkit.Chem import rdMolTransforms

    halved, probes = _PENALTY_LAW[kind]
    mol = _mol("CCCC")
    conf = mol.GetConformer(0)
    base = _uff(mol).CalcEnergy()

    for fc, dev in probes:
        ff = _uff(mol)
        if kind == "distance":
            target = rdMolTransforms.GetBondLength(conf, 0, 3) + dev
            ff.AddDistanceConstraint(0, 3, target, target, fc)
        else:
            target = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3) + dev
            ff.UFFAddTorsionConstraint(0, 1, 2, 3, False, target, target, fc)
        want = (0.5 if halved else 1.0) * fc * dev**2
        assert ff.CalcEnergy() - base == pytest.approx(want, rel=1e-3), (
            f"{kind} fc={fc} at dev={dev}: penalty {ff.CalcEnergy() - base:.3f}, expected {want:.3f}"
        )


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
    """`Pull` is the one term whose force constant does not ride `distance_fc`, and that is what "soft" means.

    At base stiffness `_SOFT_PULL_FC` equals `DISTANCE_FC` exactly, so the pull is soft only relative to an
    ESCALATED wall: `_relax_constrained` climbs the wall to 100x while the pull stays put. A pull that rode
    the ladder would reach 1e6 kcal/A^2 as a point restraint on every M-donor pair, which is the shape of
    restraint that tears a sphere rather than seating it.
    """
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


_ANGULAR_FC_CEILING = 100.0  # kcal/deg^2: 2500 kcal/mol at 5 deg, already past any cap this module wants


def _writes_angular_terms():
    """Mechanisms whose `ff_terms` calls a ``UFFAdd...Constraint``, read off the source.

    Derived rather than listed, so a mechanism that grows an angular term is covered the day it is added.
    """
    import inspect

    return {m for m in mech_mod.MECHANISM_ORDER if "UFFAdd" in inspect.getsource(type(m).ff_terms)}


def _angular_fcs():
    """Every ``(mechanism, fc)`` an angular writer emits, over fixtures that between them reach all of them.

    Three, because the population is disjoint: `Sp2Planar` and `ConjugationCap` stand down the moment
    `cons.metals` is set, `Umbrella` needs a `cons.spheres` recipe on a flat-based pyramid, and `Coplanar`
    needs a conjugated sp2 donor, which a phosphine is not.
    """
    import rxembed.pipeline as rx

    def sweep(cons, mol):
        return [
            (type(mech).__name__, args[-1])
            for mech in mech_mod.MECHANISM_ORDER
            for name, args, _kw in _ff_terms(mech, cons, mol).calls
            if name.startswith("UFFAdd")
        ]

    def seated(smiles, geometry):
        """One embedded conformer of a real complex, plus the constraints that shaped it.

        A real embed rather than a bare ETKDG one: the bond-less metal surrogate is free to land on top of a
        ligand atom, and `Umbrella` reads a dihedral through it.
        """
        iso = rx.metal(smiles, geometry).select(index=0)
        ens = rx.embed(iso, n=1, seed=1)
        work = Chem.Mol(ens._mol)
        work.RemoveAllConformers()
        work.AddConformer(ens._mol.GetConformer(ens.ids[0]), assignId=True)
        return iso.cons, work

    out = sweep(Constraints(angles={(0, 1, 3): (100.0, 120.0)}), _mol("CC(=O)NC"))  # Angle + the two repairs
    out += sweep(*seated("CP(C)(C)->[Fe](<-P(C)(C)C)<-P(C)(C)C", "trigonal_pyramidal"))  # Umbrella
    out += sweep(*seated("CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1", "square_planar"))  # Coplanar
    return out


def test_no_angular_force_constant_carries_a_distance_scale_number():
    """A kcal/A^2 number pasted into a kcal/deg^2 slot is silent, and 1e4 there is 250 000 kcal at 5 degrees.

    The coverage assertion is the load-bearing one. Read off emitted calls, a ceiling passes trivially over an
    empty set: the first version of this asserted nothing at all, because its fixture set `cons.metals`
    (silencing two writers) and carried no `cons.spheres` (silencing a third), leaving only the
    separately-capped angle wall. Every mechanism that can emit an angular term must be reached, or the
    ceiling below means nothing.
    """
    seen = _angular_fcs()
    missing = {type(m).__name__ for m in _writes_angular_terms()} - {m for m, _fc in seen}
    assert not missing, f"these emit angular terms but no fixture exercised them: {sorted(missing)}"

    over = [(m, fc) for m, fc in seen if fc > _ANGULAR_FC_CEILING]
    assert not over, f"angular force constant(s) on a distance scale: {over}"
