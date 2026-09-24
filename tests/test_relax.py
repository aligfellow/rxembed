"""Test restrained UFF and geometry-based bond acceptance."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, GetDihedralDeg, SetDihedralDeg

from rxembed import relax as relax_module
from rxembed.constraints import Constraints
from rxembed.relax import (
    FF_SURROGATE,
    UFF_GHOST,
    _bonding_failure,
    _ff_surrogate,
    bonding_ok,
    ff_energies,
    restrained_uff,
)


def _mol(smiles="CCCl", seed=7):
    """Chloroethane by default: C0-C1-Cl2, so a tear can be made in the middle or at the end."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _stretch(mol, i, j, length):
    """Pull atom j along the i-j axis so the bond reads `length`: a torn bond, made deliberately."""
    conf = mol.GetConformer(0)
    pi, pj = np.array(conf.GetAtomPosition(i)), np.array(conf.GetAtomPosition(j))
    axis = (pj - pi) / np.linalg.norm(pj - pi)
    conf.SetAtomPosition(j, Chem.rdGeometry.Point3D(*(pi + axis * length)))
    return mol


def _sn2():
    """Return a linear F...C-Cl core with both partial bonds drawn as single bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles("[F-].CCl"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=7)
    conf = mol.GetConformer()
    c, cl = np.array(conf.GetAtomPosition(1)), np.array(conf.GetAtomPosition(2))
    axis = (cl - c) / np.linalg.norm(cl - c)
    conf.SetAtomPosition(0, (c - 2.0 * axis).tolist())
    rw = Chem.RWMol(mol)
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(1).SetHybridization(Chem.HybridizationType.SP3D)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


# ---------------------------------------------------------------------------------------------------------
# bonding_ok: the arbiter
# ---------------------------------------------------------------------------------------------------------


def test_stated_distance_is_not_judged_as_a_bond():
    mol = _stretch(_mol(), 1, 2, 2.4)
    assert not bonding_ok(mol, 0), "unconstrained, a 2.4 A C-Cl should read as torn"
    assert bonding_ok(mol, 0, constrained={(1, 2): (2.35, 2.45)}), "a stated pair must be exempt"
    assert _bonding_failure(mol, 0).startswith("bond 1-2 2.400 A above")


def test_exemption_is_per_pair_not_per_atom():
    mol = _stretch(_stretch(_mol(), 1, 2, 2.4), 0, 1, 4.0)
    assert not bonding_ok(mol, 0, constrained={(1, 2): (2.35, 2.45)}), (
        "stating C-Cl must not excuse the torn C-C that shares atom 1"
    )


def test_bonding_gate_rejects_nonfinite_coordinates():
    mol = _mol()
    mol.GetConformer().SetAtomPosition(0, (float("nan"), 0.0, 0.0))

    assert not bonding_ok(mol, 0)


# ---------------------------------------------------------------------------------------------------------
# restrained_uff; relax, and score-without-moving
# ---------------------------------------------------------------------------------------------------------


def test_restrained_uff_pulls_pair_into_window():
    mol = _mol()
    before = GetBondLength(mol.GetConformer(0), 1, 2)
    restrained_uff(mol, Constraints(distances={(1, 2): (2.35, 2.45)}))
    after = GetBondLength(mol.GetConformer(0), 1, 2)
    assert abs(after - 2.4) < abs(before - 2.4), f"the pair moved away from its window: {before} -> {after}"


def test_max_iters_zero_scores_without_moving_an_atom():
    mol = _mol()
    cons = Constraints(distances={(1, 2): (2.35, 2.45)})
    before = mol.GetConformer(0).GetPositions().copy()
    restrained_uff(mol, cons, max_iters=0)
    assert np.allclose(mol.GetConformer(0).GetPositions(), before), "max_iters=0 moved atoms"


def _worst_sp2_improper(mol):
    """The largest |improper dihedral| over every degree-3 aromatic-carbon sp2 centre."""
    conf = mol.GetConformer()
    sp2 = [
        (a.GetIdx(), [n.GetIdx() for n in a.GetNeighbors()])
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP2 and a.GetDegree() == 3
    ]
    return max(abs(GetDihedralDeg(conf, nb[0], nb[1], nb[2], c)) for c, nb in sp2)


def test_sp2_hold_preserves_a_seeded_pucker_instead_of_flattening_it():
    """The sp2-carbon hold must PRESERVE a seed's existing pucker, never flatten it back to planar.

    Bare UFF has no reason to keep an aromatic ring bent, so it relaxes a manually puckered benzene ring
    straight back to ~0 deg; `restrained_uff` (which adds the sp2-hold window) must not do that.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=1) == 0
    assert _worst_sp2_improper(mol) < 1.0, "ETKDG did not seed benzene flat: the premise this bends away from is void"

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos[0, 2] += 0.5  # bend one ring carbon out of plane: a synthetic, deterministic pucker
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, p.tolist())
    seeded = _worst_sp2_improper(mol)
    assert seeded > 30.0, "the manual bend did not survive onto the measured improper"

    restrained_uff(mol, Constraints())
    held = _worst_sp2_improper(mol)
    assert held > 0.5 * seeded, f"worst sp2 improper fell to {held:.1f} deg from a {seeded:.1f} deg seed: flattened"


@pytest.mark.parametrize("trajectory", [False, True])
def test_restrained_energy_scores_endpoint_on_the_same_field(monkeypatch, trajectory):
    mol = _mol()
    cons = Constraints(distances={(1, 2): (2.35, 2.45)})
    native_builder = rdForceFieldHelpers.UFFGetMoleculeForceField
    built = []
    expected = []
    endpoints = []

    def force_field(*args, **kwargs):
        ff = native_builder(*args, **kwargs)
        built.append(ff)
        method = "MinimizeTrajectory" if trajectory else "Minimize"
        native_minimize = getattr(ff, method)

        def minimize(*args, **kwargs):
            result = native_minimize(*args, **kwargs)
            coordinates = np.asarray(ff.Positions())
            endpoints.append(coordinates.copy())
            expected.append(ff.CalcEnergy(tuple(coordinates)))
            # A rejected trial can leave cached distances without changing the returned coordinates.
            coordinates[0] += 0.5
            trial_energy = ff.CalcEnergy(tuple(coordinates))
            assert trial_energy != pytest.approx(expected[-1])
            return result

        setattr(ff, method, minimize)
        return ff

    monkeypatch.setattr(rdForceFieldHelpers, "UFFGetMoleculeForceField", force_field)
    snapshots = {} if trajectory else None
    energy = restrained_uff(mol, cons, max_iters=1, _statuses={}, _snapshots=snapshots)

    assert len(built) == len(expected) == 1, "rescoring must not rebuild the anchored force field"
    np.testing.assert_allclose(energy, expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(mol.GetConformer().GetPositions().ravel(), endpoints[0])
    if trajectory:
        assert snapshots is not None
        assert snapshots[0]


def test_restrained_uff_default_does_not_capture_trajectory(monkeypatch):
    mol = _mol()
    cons = Constraints(distances={(1, 2): (2.35, 2.45)})
    native_builder = rdForceFieldHelpers.UFFGetMoleculeForceField

    def force_field(*args, **kwargs):
        ff = native_builder(*args, **kwargs)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("default restrained UFF must not capture snapshots")

        ff.MinimizeTrajectory = forbidden
        return ff

    monkeypatch.setattr(rdForceFieldHelpers, "UFFGetMoleculeForceField", force_field)
    restrained_uff(mol, cons, max_iters=1)


def test_restrained_uff_defers_internal_optimizer_status_to_its_acceptance_gate(monkeypatch, caplog):
    mol = _mol()
    seen = []

    def force_field(_target, **_kwargs):
        def minimize(**kwargs):
            seen.append(kwargs["maxIts"])
            return 1

        return SimpleNamespace(
            Initialize=lambda: None, Minimize=minimize, Positions=lambda: (), CalcEnergy=lambda _positions: 12.5
        )

    monkeypatch.setattr(relax_module._mech, "MECHANISM_ORDER", ())
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFHasAllMoleculeParams", lambda _mol: True)
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFGetMoleculeForceField", force_field)
    statuses = {}
    with caplog.at_level("WARNING", logger="rxembed.relax"):
        restrained_uff(mol, Constraints(), _statuses=statuses)

    assert seen == [2000]
    assert statuses == {0: 1}
    assert "did not converge" not in caplog.text

    restrained_uff(mol, Constraints())
    assert "did not converge" in caplog.text


def test_untyped_selenium_uses_a_radius_corrected_sulfur_ff_graph(caplog):
    mol = Chem.MolFromSmiles("NC(=[Se])N")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((-1.2, 0.8, 0.0), (0.0, 0.0, 0.0), (2.2, 0.0, 0.0), (-1.2, -0.8, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before = Chem.MolToSmiles(mol)
    selenium = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Se")
    carbon = mol.GetAtomWithIdx(selenium).GetNeighbors()[0].GetIdx()

    sulfur = Chem.RWMol(mol)
    sulfur.GetAtomWithIdx(selenium).SetAtomicNum(16)
    sulfur = sulfur.GetMol()
    sulfur.UpdatePropertyCache(strict=False)
    sulfur_r0 = rdForceFieldHelpers.GetUFFBondStretchParams(sulfur, carbon, selenium)[1]
    radius_delta = Chem.GetPeriodicTable().GetRcovalent(34) - Chem.GetPeriodicTable().GetRcovalent(16)

    surrogates = {}
    with caplog.at_level("WARNING", logger="rxembed.relax"):
        energies = restrained_uff(mol, Constraints(), max_iters=200, _surrogates=surrogates)

    assert np.isfinite(energies).all()
    assert Chem.MolToSmiles(mol) == before
    assert mol.GetAtomWithIdx(selenium).GetAtomicNum() == 34
    assert surrogates == {selenium: (34, 16)}
    assert GetBondLength(mol.GetConformer(), carbon, selenium) == pytest.approx(sulfur_r0 + radius_delta, abs=0.03)
    assert f"Se{selenium}->S" in caplog.text


def test_untyped_arsenic_uses_a_private_phosphorus_type():
    mol = Chem.MolFromSmiles("C=[As-]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (1.9, 0.0, 0.0))
    mol.AddConformer(conf)
    before = Chem.MolToSmiles(mol)
    surrogates = {}

    energies = restrained_uff(mol, Constraints(), max_iters=5, _surrogates=surrogates)

    assert np.isfinite(energies).all()
    assert Chem.MolToSmiles(mol) == before
    assert surrogates == {1: (33, 15)}


def test_isolated_untyped_boron_uses_a_private_carbon_type():
    mol = Chem.MolFromSmiles("C=[B]C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before = Chem.MolToSmiles(mol)
    surrogates = {}

    energies = restrained_uff(mol, Constraints(), max_iters=5, _surrogates=surrogates)

    assert np.isfinite(energies).all()
    assert Chem.MolToSmiles(mol) == before
    assert surrogates == {1: (5, 6)}


def test_dithiocarbene_donor_gets_a_recognised_sulfur_charge_state():
    """A ZTDXCO-shaped `[C-2]=[S+]` donor: UFF ignores S's charge for a divalent double-bonded S and

    keeps the native (too-short) double-bond radius, crushing the C-S bond to ~1.43 A (measured on
    ZTDXCO's real crystal geometry, no metal, no constraints). Re-deriving S's hybridisation from its
    sigma degree gives the recognised ~1.59 A type instead.
    """
    mol = Chem.MolFromSmiles("C[S+]=[CH0-2]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((-1.8, 0.0, 0.0), (0.0, 0.0, 0.0), (1.6, 0.0, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    before_smiles = Chem.MolToSmiles(mol)
    surrogates = {}

    energies = restrained_uff(mol, Constraints(), max_iters=200, _surrogates=surrogates)

    length = GetBondLength(mol.GetConformer(0), 1, 2)
    assert np.isfinite(energies).all()
    assert length >= 1.55, f"the C-S bond crushed to {length:.3f} A"
    assert [a.GetFormalCharge() for a in mol.GetAtoms()] == before_charges, "public formal charges moved"
    assert Chem.MolToSmiles(mol) == before_smiles, "the public molecule was retyped, not just its private FF graph"
    assert surrogates == {1: (16, 16)}


def test_boron_network_is_not_retyped_as_carbon():
    mol = Chem.MolFromSmiles("C=[B]B")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    assert relax_module._uff_surrogate_graph(mol, Constraints()) is None


def test_native_uff_typing_is_not_replaced_for_a_poor_angle_objective():
    mol = Chem.MolFromSmiles("[O-][Cl+3]([O-])([O-])[O-]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(
        ((1.5, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 1.4, 0.0), (-0.5, -0.7, 1.2), (-0.5, -0.7, -1.2))
    ):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    surrogates = {}

    restrained_uff(mol, Constraints(), max_iters=0, _surrogates=surrogates)

    assert surrogates == {}


def test_multiple_uff_surrogates_are_groupwise_and_atom_order_invariant():
    mol = Chem.MolFromSmiles("NC(=[Se])N.C=[As-]")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    expected = {(34, 16), (33, 15)}

    for candidate in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
        selected = relax_module._uff_surrogate_graph(candidate, Constraints())
        assert selected is not None
        _graph, _constraints, replacements = selected
        assert set(replacements.values()) == expected


def test_plain_ff_energy_rejects_an_incomplete_uff_objective():
    mol = Chem.MolFromSmiles("NC(=[Se])N")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    with pytest.raises(relax_module.UFFTypingError, match="Se2"):
        ff_energies(mol, minimize=False)


def test_uff_typing_error_is_one_message_from_one_place():
    """`ff_energies` and the internal surrogate-graph search must not diverge on the remedy they name."""
    mol = Chem.MolFromSmiles("[He].[He]")
    with pytest.raises(relax_module.UFFTypingError, match=r"He0, He1.*use a different relaxation backend"):
        relax_module._raise_uff_typing_error(mol, [0, 1])


def test_surrogate_single_point_reports_its_private_objective_once(caplog):
    mol = _mol("NC(=[Se])N")
    surrogates = {}

    with caplog.at_level(logging.WARNING, logger="rxembed.relax"):
        restrained_uff(mol, Constraints(), max_iters=0, _surrogates=surrogates)
        restrained_uff(mol, Constraints(), max_iters=0, _surrogates=surrogates)

    assert caplog.text.count("private surrogate typing") == 1


def test_invalid_constraint_is_reported_as_setup_failure():
    mol = _mol()

    with pytest.raises(relax_module.UFFOptimizationError, match="constraint setup failed"):
        restrained_uff(mol, Constraints(distances={(0, 99): (1.0, 1.0)}), max_iters=0)


def test_force_field_construction_failure_is_a_typing_error(monkeypatch):
    mol = _mol()

    def fail(*_args, **_kwargs):
        raise RuntimeError("bad params pointer")

    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFGetMoleculeForceField", fail)
    with pytest.raises(relax_module.UFFTypingError, match="construction failed: bad params pointer"):
        restrained_uff(mol, Constraints(), max_iters=0)


def test_restrained_uff_seats_an_antipodal_fixed_dihedral():
    mol = _mol("CCCC")
    atoms = (0, 1, 2, 3)
    SetDihedralDeg(mol.GetConformer(), *atoms, 180.0)
    cons = Constraints(dihedrals={atoms: (-0.02, 0.02)}, fixed={atoms: (0.0, 0.0)})

    restrained_uff(mol, cons)

    assert GetDihedralDeg(mol.GetConformer(), *atoms) == pytest.approx(0.0, abs=0.005)


def test_fixed_dihedral_seating_does_not_move_a_frozen_atom():
    mol = _mol("CCCC")
    atoms = (0, 1, 2, 3)
    SetDihedralDeg(mol.GetConformer(), *atoms, 180.0)
    before = mol.GetConformer().GetAtomPosition(3)
    cons = Constraints(dihedrals={atoms: (-0.02, 0.02)}, fixed={atoms: (0.0, 0.0)}, frozen={3})

    restrained_uff(mol, cons)

    assert mol.GetConformer().GetAtomPosition(3).Distance(before) < 1e-12


def test_ff_energies_excludes_constraint_penalties():
    mol = _mol()
    cons = Constraints(distances={(1, 2): (2.35, 2.45)})  # unsatisfied by the seed -> a real penalty
    restrained = float(restrained_uff(Chem.Mol(mol), cons, max_iters=0)[0])
    plain = float(ff_energies(Chem.Mol(mol), minimize=False)[0])
    assert restrained != pytest.approx(plain), "the restrained energy carried no constraint penalty"


@pytest.mark.parametrize(("smiles", "mmff"), [("CCCl", True), ("CB(C)C", False)])
@pytest.mark.parametrize("threads", [1, 2])
def test_ff_energies_scores_batch_on_its_original_field(monkeypatch, smiles, mmff, threads):
    mol = _mol(smiles)
    mol.GetConformer().SetId(4)
    second = Chem.Conformer(mol.GetConformer())
    second.SetId(9)
    second.SetAtomPosition(0, second.GetAtomPosition(0) + Chem.rdGeometry.Point3D(0.2, 0.0, 0.0))
    mol.AddConformer(second, assignId=False)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol) == mmff
    native_optimize = rdForceFieldHelpers.OptimizeMoleculeConfs
    builder_name = "MMFFGetMoleculeForceField" if mmff else "UFFGetMoleculeForceField"
    native_builder = getattr(rdForceFieldHelpers, builder_name)
    built = []
    expected, statuses = [], {}

    def build(*args, **kwargs):
        built.append(native_builder(*args, **kwargs))
        return built[-1]

    def optimize(target, objective, **kwargs):
        assert kwargs["numThreads"] == 0
        result = native_optimize(target, objective, **(kwargs | {"numThreads": threads}))
        for conf, (status, _) in zip(target.GetConformers(), result, strict=True):
            expected.append(objective.CalcEnergy(tuple(conf.GetPositions().ravel())))
            statuses[conf.GetId()] = status
        return [(status, float("nan")) for status, _ in result]

    monkeypatch.setattr(rdForceFieldHelpers, "OptimizeMoleculeConfs", optimize)
    monkeypatch.setattr(rdForceFieldHelpers, builder_name, build)
    recorded = {}
    energies = ff_energies(mol, max_iters=1, _statuses=recorded)

    assert len(expected) == 2
    assert len(built) == 1, "each endpoint must retain the initial nonbonded contribution list"
    np.testing.assert_allclose(energies, expected, atol=1e-10, rtol=0)
    assert recorded == statuses
    assert set(recorded) == {4, 9}


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_force_field_minimizer_failure_rolls_back_the_batch(monkeypatch, error_type):
    mol = _mol()
    failed = mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    before = {c.GetId(): c.GetPositions().copy() for c in mol.GetConformers()}

    def force_field(target, **kwargs):
        conf_id = kwargs["confId"]

        def diverge(**kwargs):
            target.GetConformer(conf_id).SetAtomPosition(0, (99.0, 99.0, 99.0))
            if conf_id == failed:
                raise error_type("BFGS diverged")
            return 0

        return SimpleNamespace(
            Initialize=lambda: None, Minimize=diverge, Positions=lambda: (), CalcEnergy=lambda _positions: 12.5
        )

    monkeypatch.setattr(relax_module._mech, "MECHANISM_ORDER", ())
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFHasAllMoleculeParams", lambda mol: True)
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFGetMoleculeForceField", force_field)

    expected = relax_module.UFFOptimizationError if error_type is RuntimeError else error_type
    with pytest.raises(expected, match="BFGS diverged"):
        restrained_uff(mol, Constraints())
    for cid, positions in before.items():
        assert np.array_equal(mol.GetConformer(cid).GetPositions(), positions), (
            "a failed batch left partial coordinates"
        )


def test_untypable_fixed_core_still_relaxes_periphery(caplog):
    mol = _sn2()
    before = mol.GetConformer().GetPositions().copy()
    retyped = set()

    with caplog.at_level("WARNING", logger="rxembed.relax"):
        energies = restrained_uff(mol, Constraints(frozen={0, 1, 2}), _retyped=retyped)

    after = mol.GetConformer().GetPositions()
    assert np.isfinite(energies).all()
    assert np.array_equal(after[[0, 1, 2]], before[[0, 1, 2]]), "the fixed TS core moved"
    ch = [GetBondLength(mol.GetConformer(), 1, h) for h in (3, 4, 5)]
    assert ch == pytest.approx([1.1094] * 3, abs=0.01), "the private FF graph damaged the free C-H bonds"
    assert min(GetAngleDeg(mol.GetConformer(), h, 1, k) for h, k in ((3, 4), (3, 5), (4, 5))) > 115.0, (
        "the private FF graph lost the trigonal-bipyramidal TS angle terms"
    )
    assert mol.GetNumBonds() == 5, "the public graph was replaced by the force-field copy"
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds())
    assert retyped == {(1, 2)}
    assert "private fixed-core dative typing for C1->Cl2" in caplog.text


def test_fallback_skips_untypable_candidate_bonds():
    normal = _mol("CCCC")
    reactive = _sn2()
    offset = normal.GetNumAtoms()
    mol = Chem.CombineMols(normal, reactive, Chem.rdGeometry.Point3D(8, 0, 0))
    frozen = {0, 1, 2, 3} | {offset, offset + 1, offset + 2}

    assert np.isfinite(restrained_uff(mol, Constraints(frozen=frozen))).all()


def test_fallback_retypes_multiple_reactive_centres(caplog):
    first = _sn2()
    second = _sn2()
    offset = first.GetNumAtoms()
    mol = Chem.CombineMols(first, second, Chem.rdGeometry.Point3D(8, 0, 0))
    frozen = {0, 1, 2, offset, offset + 1, offset + 2}

    with caplog.at_level("WARNING", logger="rxembed.relax"):
        assert np.isfinite(restrained_uff(mol, Constraints(frozen=frozen))).all()
    assert caplog.text.count("->Cl") == 2


def test_haptic_phantom_composes_with_core_fallback():
    mol = _sn2()
    cons = Constraints(frozen={0, 1, 2}, haptic={6: (3, 4, 5)}, phantoms=frozenset({6}))

    energy = restrained_uff(mol, cons)

    assert np.isfinite(energy).all()
    assert [GetBondLength(mol.GetConformer(), 1, h) for h in (3, 4, 5)] == pytest.approx([1.1094] * 3, abs=0.01)


# ---------------------------------------------------------------------------------------------------------
# the force-field surrogate
# ---------------------------------------------------------------------------------------------------------


def test_ff_surrogate_preserves_atom_indices():
    organic = _mol()
    assert _ff_surrogate(organic, set(), ()) is organic, "no metal, no phantom, no copy"

    mol = _mol("CCO")
    out = _ff_surrogate(mol, {0}, (1,))
    assert out is not mol
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out.GetAtomWithIdx(0).GetAtomicNum() == FF_SURROGATE
    assert out.GetAtomWithIdx(1).GetAtomicNum() == UFF_GHOST
    assert out.GetAtomWithIdx(0).GetDegree() == mol.GetAtomWithIdx(0).GetDegree()


@pytest.mark.parametrize("bond_type", [Chem.BondType.ZERO, Chem.BondType.UNSPECIFIED])
def test_ff_surrogate_removes_only_private_zero_order_contacts(bond_type):
    mol = _mol("C.O")
    rw = Chem.RWMol(mol)
    rw.AddBond(0, 1, bond_type)
    mol = rw.GetMol()
    before = mol.GetConformer().GetPositions().copy()

    out = _ff_surrogate(mol, set())

    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out.GetBondBetweenAtoms(0, 1) is None
    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == bond_type
    assert np.array_equal(out.GetConformer().GetPositions(), before)
    assert np.isfinite(restrained_uff(mol, Constraints(), max_iters=0)).all()
