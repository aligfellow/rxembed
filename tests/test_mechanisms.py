"""Test that each mechanism consumes only its own Constraints fields."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolTransforms

from rxembed import mechanisms as mech_mod
from rxembed.constraints import Constraints, compose


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


@pytest.mark.parametrize("warm_cache", [False, True])
def test_dg_backbone_excludes_temporary_metal_stereo_carriers(warm_cache):
    mol = Chem.MolFromSmiles("[Cu+]<-NCCO")
    if warm_cache:
        assert Chem.GetDistanceMatrix(mol)[0, 4] == 4
    ctx = mech_mod.DGContext(mol, np.zeros((5, 5)))

    # DG's ligand-backbone test must ignore the metal leg without deleting its native stereo carrier.
    assert ctx.topo[0, 1] > mol.GetNumAtoms()
    assert ctx.topo[1, 4] == 3
    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DATIVE
    assert Chem.GetDistanceMatrix(mol, force=True)[0, 4] == 4


def test_dg_backbone_recomputes_cached_paths_after_a_ligand_edit():
    mol = Chem.MolFromSmiles("CC.CC")
    Chem.GetDistanceMatrix(mol)
    edited = Chem.RWMol(mol)
    edited.AddBond(1, 2, Chem.BondType.SINGLE)
    ctx = mech_mod.DGContext(edited.GetMol(), np.zeros((4, 4)))

    assert ctx.topo[0, 3] == 3


@pytest.mark.parametrize("anchor", [0.0, 180.0])
def test_angle_projection_keeps_fixed_collinear_geometry(anchor):
    mol = Chem.MolFromSmiles("C.C.C")
    cons = compose(Constraints(distances={(0, 1): (1.0, 1.0), (1, 2): (2.0, 2.0)}, fixed={(0, 1, 2): (anchor, anchor)}))
    ctx = mech_mod.DGContext(mol, np.triu(np.full((3, 3), 10.0), 1))
    mech_mod.Distance().dg_windows(cons, ctx)
    mech_mod.Angle().dg_windows(cons, ctx)

    lo, hi = ctx.pairs[(0, 2)]
    assert lo <= (1.0 if anchor == 0.0 else 3.0) <= hi


@pytest.mark.parametrize("anchor", [0.0, 180.0])
@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 1, 0, 2)])
def test_coplanar_projection_preserves_full_distance_windows(anchor, order):
    mol = Chem.RenumberAtoms(Chem.MolFromSmiles("CCCC"), order)
    atoms = tuple(order.index(i) for i in range(4))
    matrix = np.triu(np.full((4, 4), 10.0), 1)
    intervals = {(0, 1): (1.9, 2.1), (1, 2): (1.39, 1.41), (2, 3): (1.39, 1.41), (1, 3): (2.4, 2.45)}
    pairs = {}
    for (left, right), (lo, hi) in intervals.items():
        a, b = sorted((atoms[left], atoms[right]))
        matrix[b, a], matrix[a, b] = lo, hi
        pairs[(a, b)] = (lo, hi)
    row = (*atoms, anchor, 15.0)
    cons = Constraints(angles={atoms[:3]: (110.0, 120.0)}, coplanar=[row])
    ctx = mech_mod.DGContext(mol, matrix, pairs=pairs)
    anti = anchor == 180.0
    radius, theta = (1.9, 110.0) if anti else (2.1, 120.0)
    theta, psi, phi = map(math.radians, (theta, 120.0, anchor - 15.0 if anti else 15.0))
    points = np.array(
        [
            [radius * math.cos(theta), radius * math.sin(theta), 0.0],
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [1.4 - 1.4 * math.cos(psi), 1.4 * math.sin(psi) * math.cos(phi), 1.4 * math.sin(psi) * math.sin(phi)],
        ]
    )
    for (a, b), (lo, hi) in intervals.items():
        assert lo - 1e-12 <= np.linalg.norm(points[a] - points[b]) <= hi + 1e-12
    a, b = sorted((atoms[0], atoms[3]))
    before = (ctx.bm[b, a], ctx.bm[a, b])
    mech_mod.Coplanar().dg_post(cons, ctx)
    assert (ctx.bm[b, a], ctx.bm[a, b]) != before, "the coplanar cap must actually tighten this pair"
    distance = np.linalg.norm(points[0] - points[3])
    assert ctx.bm[b, a] - 1e-12 <= distance <= ctx.bm[a, b] + 1e-12


def test_coplanar_post_uses_committed_intervals_not_pending_midpoints():
    mol = Chem.MolFromSmiles("CCCC")
    matrix = np.triu(np.full((4, 4), 10.0), 1)
    for a, b, length in [(0, 1, 2.0), (1, 2, math.sqrt(2)), (2, 3, math.sqrt(2)), (1, 3, 2.0)]:
        matrix[b, a] = matrix[a, b] = length
    cons = Constraints(angles={(0, 1, 2): (90.0, 90.0)}, coplanar=[(0, 1, 2, 3, 180.0, 0.0)])
    ctx = mech_mod.DGContext(mol, matrix, pairs={(1, 3): (1.0, 3.0)})
    mech_mod.Coplanar().dg_post(cons, ctx)
    assert ctx.bm[3, 0] == pytest.approx(math.sqrt(8 + 4 * math.sqrt(2)))


def test_coplanar_interval_extrema_include_interior_edge_points():
    angles = mech_mod.triangle_angles((1.0, 3.0), (2.0, 2.0), (1.0, 1.0))
    assert angles == pytest.approx((0.0, math.pi / 6))
    matrix = np.triu(np.full((4, 4), 10.0), 1)
    for a, b, length in [(1, 2, 2.0), (1, 3, 2.0), (2, 3, 4 * math.sin(math.radians(10)))]:
        matrix[b, a] = matrix[a, b] = length
    matrix[1, 0], matrix[0, 1] = 1.0, 3.0
    # beta=20 degrees and theta=10 degrees give gamma=30. The closest i lies inside its radial interval.
    assert mech_mod._coplanar_bound(matrix, (0, 1, 2, 3), (10.0, 10.0), 180.0, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("window", "anchor", "cap"),
    [
        ((np.nan, 120.0), 180.0, 15.0),
        ((120.0, np.inf), 180.0, 15.0),
        ((120.0, 110.0), 180.0, 15.0),
        ((-1.0, 90.0), 180.0, 15.0),
        ((0.0, 181.0), 180.0, 15.0),
        ((90.0, 90.0), 180.0, np.nan),
        ((90.0, 90.0), 90.0, 15.0),
        ((90.0, 90.0), 180.0, 91.0),
    ],
)
def test_coplanar_projection_abstains_outside_its_geometric_domain(window, anchor, cap):
    matrix = np.ones((4, 4)) - np.eye(4)
    assert mech_mod._coplanar_bound(matrix, (0, 1, 2, 3), window, anchor, cap) is None


def uff_terms(mech, cons, mol, fc=1.0):
    ff = SpyFF()
    mech.uff_terms(ff, cons, mol.GetConformer(0), fc)
    return ff


# `Sp2Planar` and `ConjugationCap` are not constraints at all but repairs to UFF (a 3-neighbour sp2 carbon that
# pyramidalises, a conjugated plane that twists), so they perceive from the molecule. Giving them a field would
# mean filling it on every path that can reach the relax.
_FF_REPAIRS = {"Sp2Planar", "TrigonalAngle", "ConjugationCap"}
_FIELD_DRIVEN = [m for m in mech_mod.MECHANISM_ORDER if type(m).__name__ not in _FF_REPAIRS]


# ---------------------------------------------------------------------------------------------------------
# the negative contract: a field-driven mechanism is silent off an empty struct
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mech", _FIELD_DRIVEN, ids=lambda m: type(m).__name__)
def test_mechanism_is_silent_on_empty_constraints(mech):
    mol = _mol()
    assert uff_terms(mech, Constraints(), mol).calls == [], f"{type(mech).__name__} wrote an FF term unasked"

    ctx = _ctx(mol)
    before = ctx.bm.copy()
    for hook in (mech.dg_windows, mech.dg_relief, mech.dg_post):
        hook(Constraints(), ctx)
    assert ctx.pairs == {}, f"{type(mech).__name__} proposed a window off an empty struct"
    assert np.array_equal(ctx.bm, before), f"{type(mech).__name__} edited the matrix off an empty struct"


@pytest.mark.parametrize("name", sorted(_FF_REPAIRS))
def test_ff_repair_uses_molecule_state(name):
    mech = next(m for m in mech_mod.MECHANISM_ORDER if type(m).__name__ == name)
    mol = _mol()
    assert uff_terms(mech, Constraints(), mol).calls, f"{name} must fire on a bare struct"


def test_sp2_planar_excludes_only_coordination_owned_carbon():
    mol = _mol("CC(=C)C")
    centre = next(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2 and atom.GetDegree() == 3
    )
    neighbors = {atom.GetIdx() for atom in mol.GetAtomWithIdx(centre).GetNeighbors()}
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in neighbors | {centre})
    mechanism = mech_mod.Sp2Planar()

    def calls(cons):
        return uff_terms(mechanism, cons, mol).calls

    bare = calls(Constraints())
    torsions = [args[:4] for name, args, _kwargs in bare if name == "UFFAddTorsionConstraint" and args[3] == centre]
    assert len(torsions) == 3
    assert {name for name, _args, _kwargs in bare} == {"UFFAddTorsionConstraint"}

    pair = tuple(sorted((metal, centre)))
    coordinated = Constraints(metals={metal}, distances={pair: (1.9, 2.1)})
    assert not any(args[3] == centre for name, args, _kwargs in calls(coordinated) if name == "UFFAddTorsionConstraint")

    contact = coordinated.copy(contacts=(frozenset({pair}), frozenset()))
    assert any(args[3] == centre for name, args, _kwargs in calls(contact) if name == "UFFAddTorsionConstraint")

    dummy = mol.GetNumAtoms()
    haptic = Constraints(metals={metal}, distances={(metal, dummy): (1.9, 2.1)}, haptic={dummy: (centre,)})
    haptic_calls = calls(haptic)
    assert not any(args[3] == centre for name, args, _kwargs in haptic_calls if name == "UFFAddTorsionConstraint")


def test_stated_improper_owns_its_carbon_but_a_proper_torsion_does_not():
    mol = _mol("CCC(=O)N")
    centre = 2
    neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(centre).GetNeighbors()]
    mechanism = mech_mod.Sp2Planar()
    a, b, c = neighbors
    for key in ((a, b, c, centre), (centre, b, a, c), (c, a, centre, b)):
        assert not uff_terms(mechanism, Constraints(dihedrals={key: (-10.0, 10.0)}), mol).calls

    calls = uff_terms(mechanism, Constraints(dihedrals={(0, 1, 2, 3): (-10.0, 10.0)}), mol).calls
    assert len(calls) == 3


@pytest.mark.parametrize("owner", ["angle", "distance", "floor", "graft"])
def test_trigonal_angle_repair_defers_to_stated_geometry(owner):
    mol = _mol("C=C")
    centre = 0
    a, b = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(centre).GetNeighbors() if neighbor.GetAtomicNum() == 1]
    cons = {
        "angle": Constraints(angles={(b, centre, a): (80.0, 100.0)}),
        "distance": Constraints(distances={(a, b): (1.0, 2.0)}),
        "floor": Constraints(floors={(a, b): 1.0}),
        "graft": Constraints(frozen={a, centre, b}),
    }[owner]

    calls = uff_terms(mech_mod.TrigonalAngle(), cons, mol).calls
    assert calls
    assert not any(args[:3] == (a, centre, b) for _name, args, _kwargs in calls)


def test_trigonal_angle_repair_leaves_native_small_ring_angles_alone():
    assert not uff_terms(mech_mod.TrigonalAngle(), Constraints(), _mol("O=C1CC1")).calls


@pytest.mark.parametrize("smiles", ["C=C", "NC=O"])
def test_trigonal_angle_repair_penalizes_collapsed_geometry(smiles):
    mol = _mol(smiles)
    conf = mol.GetConformer()
    centre = mol.GetAtomWithIdx(0)
    hydrogens = [atom.GetIdx() for atom in centre.GetNeighbors() if atom.GetAtomicNum() == 1]
    origin = np.array(conf.GetAtomPosition(0))
    conf.SetAtomPosition(hydrogens[0], origin + np.array((-0.5, 0.866, 0)))
    conf.SetAtomPosition(hydrogens[1], origin + np.array((-0.866, 0.5, 0)))
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)

    mech_mod.TrigonalAngle().uff_terms(ff, Constraints(), mol.GetConformer(), 1.0)
    ff.Initialize()

    assert ff.CalcEnergy() > 0
    assert np.linalg.norm(ff.CalcGrad()) > 0


@pytest.mark.parametrize("smiles", ["C1=CCCC1", "C1=CC=CC1", "C=C", "NC=O"])
def test_trigonal_angle_repair_preserves_unstrained_native_energy_and_gradient(smiles):
    mol = _mol(smiles)
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
    ff.Initialize()
    energy, gradient = ff.CalcEnergy(), ff.CalcGrad()

    mech_mod.TrigonalAngle().uff_terms(ff, Constraints(), mol.GetConformer(), 1.0)
    ff.Initialize()

    assert ff.CalcEnergy() == pytest.approx(energy, abs=1e-12)
    assert np.allclose(ff.CalcGrad(), gradient, atol=1e-12, rtol=0)


def test_conjugation_cap_excludes_only_a_coordinated_pi_carbon():
    mol = _mol()
    a, c, x, s = next(mech_mod.conjugated_quartets(mol))
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in {a, c, x, s})
    mechanism = mech_mod.ConjugationCap()

    assert uff_terms(mechanism, Constraints(metals={metal}, distances={(metal, s): (1.4, 1.6)}), mol).calls
    assert not uff_terms(mechanism, Constraints(metals={metal}, distances={(metal, c): (1.4, 1.6)}), mol).calls
    contact = (min(metal, c), max(metal, c))
    assert uff_terms(
        mechanism,
        Constraints(metals={metal}, distances={contact: (1.0, 2.0)}, contacts=(frozenset({contact}), frozenset())),
        mol,
    ).calls


def test_conjugation_cleanup_switch_is_uff_only():
    mol = _mol()
    mechanism = mech_mod.ConjugationCap()

    assert uff_terms(mechanism, Constraints(), mol).calls
    assert not uff_terms(mechanism, Constraints(conjugation=False), mol).calls


# ---------------------------------------------------------------------------------------------------------
# the positive contract: a populated field writes the term it claims
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["distance", "angle", "torsion"])
def test_native_restraint_penalty_law(kind):
    # One (k, dev) case per kind: this pins RDKit's own force-field energy formula, not rxembed logic,
    # so sweeping k and dev re-derives the same tautology and adds nothing.
    mol = _mol("CCCC")
    conf = mol.GetConformer()
    k, dev = 100.0, 0.1
    held = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
    if kind == "distance":
        target = rdMolTransforms.GetBondLength(conf, 0, 1) - dev
        held.AddDistanceConstraint(0, 1, target, target, k)
    elif kind == "angle":
        target = rdMolTransforms.GetAngleDeg(conf, 0, 1, 2) - dev
        held.UFFAddAngleConstraint(0, 1, 2, False, target, target, k)
    else:
        target = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3) - dev
        held.UFFAddTorsionConstraint(0, 1, 2, 3, False, target, target, k)
    held.Initialize()
    factor = 0.5 if kind == "distance" else 1.0
    assert held.CalcEnergy() == pytest.approx(factor * k * dev**2, rel=1e-6)


def test_distance_writes_dg_and_ff():
    mol = _mol()
    cons = Constraints(distances={(0, 3): (2.0, 2.4)})

    ctx = _ctx(mol)
    mech_mod.Distance().dg_windows(cons, ctx)
    assert ctx.pairs[(0, 3)] == (2.0, 2.4)

    ff = uff_terms(mech_mod.Distance(), cons, mol)
    assert ff.kinds() == ["AddDistanceConstraint"]
    _name, args, _kw = ff.calls[0]
    assert args[:4] == (0, 3, 2.0, 2.4)


def test_haptic_radius_is_a_seed_prior_unless_explicitly_fixed():
    mol = _mol()
    cons = Constraints(distances={(0, 3): (0.6, 0.8), (1, 3): (0.6, 0.8), (2, 3): (1.9, 2.1)}, haptic={3: (0, 1)})
    ctx = _ctx(mol)
    mech_mod.Distance().dg_windows(cons, ctx)
    assert ctx.pairs == cons.distances

    for fixed in ({}, {(0, 3): (0.6, 0.8)}):
        cons.fixed = fixed
        calls = uff_terms(mech_mod.Distance(), cons, mol).calls
        assert {args[:2] for _name, args, _kw in calls} == {(2, 3)} | set(fixed)


@pytest.mark.parametrize("size", [2, 3, 6])
@pytest.mark.parametrize("stiffness", [1.0, 100.0])
def test_haptic_penalty_moves_the_centroid_without_straining_its_face(size, stiffness):
    mol = Chem.MolFromSmiles(".".join(["[He]"] * (size + 1)))
    conf = Chem.Conformer(size + 1)
    face = np.random.default_rng(42).normal(size=(size, 3)) * (1.0, 2.0, 0.4)
    conf.SetPositions(np.vstack((face, face.mean(axis=0))))
    mol.AddConformer(conf)
    force = stiffness * mech_mod.PIN_FC

    for members in (tuple(range(size)), tuple(reversed(range(size)))):
        ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
        mech_mod.Haptic().uff_terms(ff, Constraints(haptic={size: members}), mol.GetConformer(), stiffness)
        ff.Initialize()
        for offset in (np.zeros(3), np.array((0.3, -0.2, 0.4))):
            positions = np.vstack((face, face.mean(axis=0) + offset)).ravel().tolist()
            assert ff.CalcEnergy(positions) == pytest.approx(0.5 * force * np.dot(offset, offset), abs=1e-8)
            expected = np.vstack((np.tile(-force * offset / size, (size, 1)), force * offset))
            np.testing.assert_allclose(np.array(ff.CalcGrad(positions)).reshape(-1, 3), expected, atol=1e-8)


@pytest.mark.parametrize("frozen", [{0, 1}, {0, 1, 2, 3}])
def test_overlapping_haptic_centroids_respect_frozen_members(frozen):
    mol = Chem.MolFromSmiles("[He].[He].[He].[He].[He].[He]")
    conf = Chem.Conformer(6)
    positions = np.random.default_rng(42).normal(size=(6, 3))
    conf.SetPositions(positions)
    mol.AddConformer(conf)
    cons = Constraints(haptic={4: (0, 1, 2), 5: (1, 2, 3)}, frozen=frozen)
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
    mech_mod.Frozen().uff_terms(ff, cons, mol.GetConformer(), 100.0)
    mech_mod.Haptic().uff_terms(ff, cons, mol.GetConformer(), 100.0)
    ff.Initialize()

    assert ff.Minimize(maxIts=200) == 0

    relaxed = mol.GetConformer().GetPositions()
    np.testing.assert_array_equal(relaxed[sorted(frozen)], positions[sorted(frozen)])
    for dummy, face in cons.haptic.items():
        np.testing.assert_allclose(relaxed[dummy], relaxed[list(face)].mean(axis=0), atol=1e-6)


def test_pull_collapses_window_to_spring():
    mol = _mol()
    ff = uff_terms(mech_mod.Pull(), Constraints(pulls={(0, 3): 2.1}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == args[3] == 2.1

    fixed = uff_terms(
        mech_mod.Pull(),
        Constraints(fixed={(0, 3): (2.1, 2.1)}),
        mol,
    )
    assert fixed.calls[0][1][2:4] == (2.1, 2.1)
    assert fixed.calls[0][1][-1] > args[-1]


def test_reversed_distance_preference_suppresses_contact_midpoint_force():
    cons = Constraints(distances={(0, 3): (2.0, 4.0)}, pulls={(3, 0): 2.5}, contacts=(frozenset({(0, 3)}), frozenset()))
    calls = uff_terms(mech_mod.Pull(), cons, _mol()).calls
    assert len(calls) == 1
    assert calls[0][1][2:4] == (2.5, 2.5)


def test_native_angle_preference_pulls_inside_the_window_without_escalation():
    mol = Chem.MolFromSmiles("[He].[He].[He]")
    conf = Chem.Conformer(3)
    conf.SetPositions(np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0]]))
    mol.AddConformer(conf)
    key = (0, 1, 2)
    cons = Constraints(angles={key: (80.0, 140.0)}, pulls={key: 120.0, key[::-1]: 120.0})
    energies = []
    for stiffness in (1.0, 100.0):
        ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
        mech_mod.Angle().uff_terms(ff, cons, mol.GetConformer(), stiffness)
        mech_mod.Pull().uff_terms(ff, cons, mol.GetConformer(), stiffness)
        ff.Initialize()
        energy = ff.CalcEnergy()
        assert energy > 0
        energies.append(energy)
        positions = mol.GetConformer().GetPositions().ravel()
        grad = np.asarray(ff.CalcGrad())
        assert ff.CalcEnergy((positions - 1e-6 * grad).tolist()) < energy
        assert ff.Minimize(maxIts=200) == 0
        assert rdMolTransforms.GetAngleDeg(mol.GetConformer(), *key) == pytest.approx(120.0, abs=1e-3)
        mol.GetConformer().SetPositions(conf.GetPositions())
    assert energies[0] == pytest.approx(energies[1])
    calls = uff_terms(mech_mod.Pull(), cons, mol).calls
    assert len(calls) == 1
    for fixed in (key, key[::-1]):
        cons.fixed = {fixed: (100.0, 110.0)}
        assert not uff_terms(mech_mod.Pull(), cons, mol).calls


def test_floor_is_one_sided():
    mol = _mol()
    ff = uff_terms(mech_mod.Floor(), Constraints(floors={(0, 3): 2.6}), mol)
    _name, args, _kw = ff.calls[0]
    assert args[2] == 2.6
    assert args[3] >= 1e3, f"a floor's upper bound must stand in for infinity, got {args[3]}"


@pytest.mark.parametrize("authority", ["fixed", "contact", "automatic"])
def test_later_distance_overrides_only_an_explicitly_owned_floor(authority):
    mol = _mol()
    pair, other = (0, 3), (1, 4)
    window = (2.0, 2.1)
    base = Constraints(floors={pair: 2.6, other: 2.7}, dg_floors={pair: 2.6, other: 2.7})
    stated = Constraints(
        distances={pair: window},
        fixed={pair: window} if authority == "fixed" else {},
        contacts=(frozenset({pair}) if authority == "contact" else frozenset(), frozenset()),
    )
    cons = compose(base, stated)
    ctx = _ctx(mol)
    mech_mod.Distance().dg_windows(cons, ctx)
    mech_mod.Floor().dg_relief(cons, ctx)

    assert ctx.pairs[pair] == window
    ff = uff_terms(mech_mod.Floor(), cons, mol)
    expected = [(*pair, 2.6), (*other, 2.7)] if authority == "automatic" else [(*other, 2.7)]
    assert [args[:3] for _name, args, _kw in ff.calls] == expected
    assert base.floors[pair] == 2.6, "overriding a pair must not mutate the reusable structural model"
    if authority == "contact":
        released = uff_terms(mech_mod.Floor(), cons.relaxed(), mol)
        assert [args[:3] for _name, args, _kw in released.calls] == [(*pair, 2.6), (*other, 2.7)]


def test_frozen_pins_points_rather_than_restraining_them():
    mol = _mol()
    ff = uff_terms(mech_mod.Frozen(), Constraints(frozen={0, 1, 2}), mol)
    assert ff.kinds() == ["AddFixedPoint"]
    assert sorted(a[0] for _n, a, _k in ff.calls) == [0, 1, 2]


def test_angle_writes_dg_and_ff_terms():
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})

    ctx = _ctx(mol)
    mech_mod.Angle().dg_windows(cons, ctx)
    assert (0, 3) in ctx.pairs, "an angle must state its end-atom diagonal in the matrix"

    ff = uff_terms(mech_mod.Angle(), cons, mol)
    assert ff.kinds() == ["UFFAddAngleConstraint"]
    _name, args, _kw = ff.calls[0]
    assert args[:3] == (0, 1, 3)

    fixed = uff_terms(
        mech_mod.Angle(),
        Constraints(angles={(0, 1, 3): (108.0, 112.0)}, fixed={(0, 1, 3): (110.0, 110.0)}),
        mol,
    )
    assert fixed.calls[0][1][4:6] == (110.0, 110.0)
    assert fixed.calls[0][1][-1] > args[-1]


def test_angle_force_is_not_scaled_by_ladder():
    mol = _mol()
    cons = Constraints(angles={(0, 1, 3): (100.0, 120.0)})
    assert uff_terms(mech_mod.Angle(), cons, mol, fc=1.0).calls[0][1][-1] == mech_mod.ANGLE_FC
    assert uff_terms(mech_mod.Angle(), cons, mol, fc=0.1).calls[0][1][-1] == pytest.approx(0.1 * mech_mod.ANGLE_FC)
    assert uff_terms(mech_mod.Angle(), cons, mol, fc=100.0).calls[0][1][-1] == mech_mod.ANGLE_FC, (
        "the ladder must not be able to escalate the angle wall"
    )


def test_dihedral_writes_periodic_uff_term():
    mol = _mol("CCCC")
    cons = Constraints(dihedrals={(0, 1, 2, 3): (170.0, 190.0)})
    ff = uff_terms(mech_mod.Dihedral(), cons, mol)
    assert ff.kinds() == ["UFFAddTorsionConstraint"]
    assert ff.calls[0][1][:7] == (0, 1, 2, 3, False, 170.0, 190.0)


@pytest.mark.parametrize("smiles", ["C(F)(Cl)(Br)I", "C(F)(Cl)Br"], ids=["four-carriers", "implicit-h"])
@pytest.mark.parametrize("height", [0.05, 0.6, 1.0, 4.0])
def test_point_umbrella_leaves_every_same_hand_geometry_unbiased(smiles, height):
    mol = Chem.MolFromSmiles(smiles)
    points = np.array([(0, 0, 0), (1, 1, height), (1, -1, -height), (-1, 1, -height), (-1, -1, height)])
    points = points[: mol.GetNumAtoms()]
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetPositions(points)
    mol.AddConformer(conf)
    mirror = points * (-1, 1, 1)
    mirror_energies = []
    for carriers in itertools.permutations(range(1, mol.GetNumAtoms())):
        key = carriers if len(carriers) == 4 else (*carriers, 0)
        ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
        mech_mod.Umbrella().uff_terms(ff, Constraints(umbrellas={key: 0.0}), mol.GetConformer(), 1.0)
        ff.Initialize()

        assert ff.CalcEnergy() == pytest.approx(0.0, abs=1e-10), key
        np.testing.assert_allclose(ff.CalcGrad(), 0.0, atol=1e-10)
        mirror_energies.append(ff.CalcEnergy(mirror.ravel().tolist()))
        assert mirror_energies[-1] > 0.0, key
    assert mirror_energies == pytest.approx([mirror_energies[0]] * len(mirror_energies))


@pytest.mark.parametrize("seed_phi", [1.0, 179.0, -179.0])
def test_planar_umbrella_preserves_both_sides_of_its_periodic_cap(seed_phi):
    mol = _mol("CCCC")
    key = (0, 1, 2, 3)
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, *key, seed_phi)
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
    mech_mod.Umbrella().uff_terms(ff, Constraints(umbrellas={key: None}), conf, 1.0)
    ff.Initialize()
    centre = 0.0 if abs(seed_phi) < 90.0 else 180.0
    for offset in (-20.0, -14.0, 0.0, 14.0, 20.0):
        rdMolTransforms.SetDihedralDeg(conf, *key, centre + offset)
        positions = conf.GetPositions()
        energy = ff.CalcEnergy(positions.ravel().tolist())
        gradient = np.array(ff.CalcGrad(positions.ravel().tolist())).reshape(-1, 3)
        if abs(offset) <= mech_mod._PLANAR_CAP:
            assert energy == pytest.approx(0.0, abs=1e-8), "inside the cap must be flat"
            np.testing.assert_allclose(gradient, 0.0, atol=1e-8)
        else:
            assert energy > 0.0, "outside the cap must be penalized"
        reflected = positions * (1.0, 1.0, -1.0)
        assert ff.CalcEnergy(reflected.ravel().tolist()) == pytest.approx(energy, abs=1e-8)
        reflected_gradient = np.array(ff.CalcGrad(reflected.ravel().tolist())).reshape(-1, 3)
        np.testing.assert_allclose(reflected_gradient, gradient * (1.0, 1.0, -1.0), atol=1e-8)


def test_point_umbrella_defers_to_stated_dihedral_independent_of_carrier_order():
    mol = _mol("C(F)(Cl)(Br)I")
    for key in itertools.permutations((1, 2, 3, 4)):
        cons = Constraints(umbrellas={key: 0.0}, dihedrals={(1, 2, 3, 4): (10.0, 30.0)})
        assert not uff_terms(mech_mod.Umbrella(), cons, mol).calls, key


def test_weighted_planar_umbrella_scales_energy_and_gradient():
    mol = _mol("CCCC")
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, 120.0)
    results = []
    for weight in (0.25, 0.5):
        ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
        cons = Constraints(umbrellas={(0, 1, 2, 3): (180.0, weight)})
        mech_mod.Umbrella().uff_terms(ff, cons, conf, 1.0)
        ff.Initialize()
        results.append((ff.CalcEnergy(), np.array(ff.CalcGrad())))
    assert results[0][0] > 0.0
    assert results[1][0] == pytest.approx(2.0 * results[0][0])
    np.testing.assert_allclose(results[1][1], 2.0 * results[0][1], atol=1e-9)


@pytest.mark.parametrize("ideal", [None, 30.0, 0.0, (180.0, 0.25)])
def test_umbrella_force_matches_support_overrides_and_not_shared_axes(ideal):
    mol = _mol("CCCCCC")
    for key in itertools.permutations((0, 1, 2, 3)):
        independent = (4, key[1], key[2], 5)
        cons = Constraints(umbrellas={key: ideal}, dihedrals={independent: (10.0, 30.0)})
        assert uff_terms(mech_mod.Umbrella(), cons, mol).calls
        assert uff_terms(mech_mod.Umbrella(), cons.copy(frozen=set(key[:3])), mol).calls
        assert not uff_terms(mech_mod.Umbrella(), cons.copy(frozen=set(key)), mol).calls
        cons.dihedrals = {(1, 0, 3, 2): (10.0, 30.0)}
        assert not uff_terms(mech_mod.Umbrella(), cons, mol).calls


@pytest.mark.parametrize("ideal", [None, 30.0, 0.0, (180.0, 0.25)])
def test_composed_numeric_fix_overrides_the_same_umbrella_support(ideal):
    mol = _mol("CCCCCC")
    base = Constraints(umbrellas={(0, 1, 2, 3): ideal})
    for key, suppressed in (((1, 0, 3, 2), True), ((4, 1, 2, 5), False)):
        cons = compose(base, Constraints(fixed={key: (20.0, 20.0)}))
        for active in (cons, cons.relaxed()):
            assert active.fixed[key] == (20.0, 20.0)
            assert active.umbrellas == base.umbrellas
            assert bool(uff_terms(mech_mod.Umbrella(), active, mol).calls) != suppressed
    key = (1, 0, 3, 2)
    soft = Constraints(dihedrals={key: (10.0, 30.0)}, contacts=(frozenset(), frozenset({key})))
    point = compose(Constraints(umbrellas={(0, 1, 2, 3): 0.0}), soft)
    assert not uff_terms(mech_mod.Umbrella(), point, mol).calls
    assert uff_terms(mech_mod.Umbrella(), point.relaxed(), mol).calls


def test_releasable_contact_uses_softer_wall():
    mol = _mol("c1ccccc1.c1ccccc1")
    cons = Constraints(
        distances={(0, 3): (2.0, 2.4), (1, 4): (2.0, 2.4)},
        contacts=(frozenset({(0, 3)}), frozenset()),
    )
    emitted = {tuple(a[:2]): a[-1] for _n, a, _kw in uff_terms(mech_mod.Distance(), cons, mol).calls}
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

    ff = uff_terms(mech_mod.Plane(), cons, mol)
    assert ff.kinds() == ["AddDistanceConstraint"]


# ---------------------------------------------------------------------------------------------------------
# MECHANISM_ORDER itself: the one place order is stated, and the order is load-bearing
# ---------------------------------------------------------------------------------------------------------


def test_distance_precedes_angle_window_read():
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


def test_distance_walls_scale_without_strengthening_target_pulls():
    mol = _mol("c1ccccc1.c1ccccc1")
    cons = _populated(mol)

    emitted: dict[str, dict[float, float]] = {}
    for fc in (1.0, 100.0):
        for mech in mech_mod.MECHANISM_ORDER:
            for name, args, _kw in uff_terms(mech, cons, mol, fc=fc).calls:
                if name.endswith("Constraint"):
                    emitted.setdefault(type(mech).__name__, {})[fc] = args[-1]

    rode = {m for m, by_fc in emitted.items() if len(set(by_fc.values())) > 1}
    assert "Pull" in emitted, "the fixture must exercise Pull for this to mean anything"
    assert "Pull" not in rode, "Pull rode the stiffness ladder; it is the spring and must stay put"
    assert {"Distance", "Floor", "Plane"} <= rode, f"a wall must scale with the caller's stiffness, got {rode}"
