"""Test that each mechanism consumes only its own Constraints fields."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed import mechanisms as mech_mod
from rxembed.constraints import Constraints, resolve_core


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


def test_distance_relief_uses_bounds_basis_and_preserves_owned_geometry():
    mol = Chem.AddHs(Chem.MolFromSmiles("CBr.[Cl-]"))
    pins = {0: (0.0, 0.0, 0.0), 1: (2.316, 0.0, 0.0), 2: (-2.122, 0.0, 0.0)}
    cons, _ = resolve_core(mol, fix=pins)
    cons = cons.copy(floors={(2, 3): 3.1})
    native = Chem.Mol(mol)
    connected = Chem.RWMol(mol)
    connected.AddBond(0, 2, Chem.BondType.DATIVE)
    graph = connected.GetMol()
    ctx = mech_mod.DGContext(graph, rdDistGeom.GetMoleculeBoundsMatrix(native), basis=native)
    mech_mod.Distance().dg_windows(cons, ctx)
    before = ctx.bm.copy()

    mech_mod.Distance().dg_relief(cons, ctx)

    assert ctx.bm[4, 2] < before[4, 2], "a held contact can relieve its external neighbour floor"
    assert ctx.bm[3, 2] == pytest.approx(before[3, 2]), "an explicit floor remains authoritative"
    for pair in ((0, 1), (0, 2), (1, 2)):
        a, b = sorted(pair)
        assert ctx.bm[b, a] == pytest.approx(before[b, a])
        assert ctx.bm[a, b] == pytest.approx(before[a, b])


@pytest.mark.parametrize("anchor", [0.0])
@pytest.mark.parametrize("order", [(0, 1, 2, 3)])
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


def test_aromatic_carbanion_bridge_keeps_coordination_ownership():
    mol = _mol("[cH-]1cccc1.C.C")  # two bondless metal surrogates bind the same carbon
    cons = Constraints(metals={5, 6}, distances={(0, 5): (2.0, 2.2), (0, 6): (2.0, 2.2)})
    calls = uff_terms(mech_mod.Sp2Planar(), cons, mol).calls
    assert calls, "uncoordinated ring carbons still receive ligand cleanup"
    assert not any(args[3] == 0 for _name, args, _kwargs in calls), "the bridge owns its carbon geometry"


# ---------------------------------------------------------------------------------------------------------
# the negative contract: a field-driven mechanism is silent off an empty struct
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# the positive contract: a populated field writes the term it claims
# ---------------------------------------------------------------------------------------------------------


def test_reversed_distance_preference_suppresses_contact_midpoint_force():
    cons = Constraints(distances={(0, 3): (2.0, 4.0)}, pulls={(3, 0): 2.5}, contacts=(frozenset({(0, 3)}), frozenset()))
    calls = uff_terms(mech_mod.Pull(), cons, _mol()).calls
    assert len(calls) == 1
    assert calls[0][1][2:4] == (2.5, 2.5)


@pytest.mark.parametrize("smiles", ["C(F)(Cl)Br"], ids=["implicit-h"])
@pytest.mark.parametrize("height", [4.0])
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


@pytest.mark.parametrize("ideal", [(180.0, 0.25)])
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
# the force constants: what a number in this module actually costs
# ---------------------------------------------------------------------------------------------------------
