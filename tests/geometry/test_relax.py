"""`geometry/relax.py`: restrained UFF on a private typeable graph, the coupled endpoint relax, and the bond judge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.rdMolTransforms import GetBondLength
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.geometry.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL, Constraints, constraint_value
from rxembed.geometry.relax import (
    CoEmbedParams,
    Coupling,
    Endpoint,
    UFFOptimizationError,
    UFFRecord,
    UFFTypingError,
    bonding_failure,
    coupled_uff,
    restrained_uff,
)
from tests.geometry_fixtures import embedded, placed


def _sn2_core():
    """Return F0...C1-Cl2 drawn as two single bonds in a linear backside attack: a carbon UFF cannot type."""
    mol = embedded("[F-].CCl")
    carbon, chlorine = mol.GetConformer().GetPositions()[1:3]
    mol.GetConformer().SetAtomPosition(0, Point3D(*(2 * carbon - chlorine)))
    rw = Chem.RWMol(mol)
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(1).SetHybridization(Chem.HybridizationType.SP3D)
    rw.UpdatePropertyCache(strict=False)
    return rw.GetMol()


def test_restrained_uff_holds_each_fix_within_its_gate():
    fixed = {(0, 1): (1.6, 1.6), (1, 2): (1.4, 1.45), (0, 1, 2): (100.0, 100.0), (1, 2, 3): (120.0, 125.0)}
    fixed[(0, 1, 2, 3)] = (60.0, 60.0)
    butane = embedded("CCCC")
    assert np.isfinite(restrained_uff(butane, Constraints(fixed=fixed))).all()
    tol = {2: FIX_DISTANCE_TOL, 3: FIX_ANGLE_TOL, 4: FIX_ANGLE_TOL}
    gates = {key: pytest.approx((lo + hi) / 2, abs=(hi - lo) / 2 + tol[len(key)]) for key, (lo, hi) in fixed.items()}
    assert {key: constraint_value(butane.GetConformer().GetPositions(), key) for key in fixed} == gates


def test_restrained_uff_reports_each_minimisation_to_its_record(caplog):
    """Butane collapsed onto one axis cannot converge in one iteration; a single point moves nothing."""
    snapshots, butane = {}, embedded("CCCC")
    record = UFFRecord(snapshots=snapshots)
    restrained_uff(butane, Constraints(), record=record)
    assert record.statuses == {0: 0}
    np.testing.assert_allclose(snapshots[0][-1], butane.GetConformer().GetPositions())
    collapsed = [(0.01 * i, 0.0, 0.0) for i in range(14)]
    record, mol = UFFRecord(), placed(Chem.AddHs(Chem.MolFromSmiles("CCCC")), collapsed)
    with caplog.at_level("WARNING", logger="rxembed.relax"):
        restrained_uff(mol, Constraints(), max_iters=1, record=record)
        statuses, positions = dict(record.statuses), mol.GetConformer().GetPositions()
        restrained_uff(mol, Constraints(), max_iters=0, record=record)
        assert "did not converge" not in caplog.text
        restrained_uff(placed(Chem.AddHs(Chem.MolFromSmiles("CCCC")), collapsed), Constraints(), max_iters=1)
    assert record.statuses == statuses != {0: 0}
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), positions)
    assert "1 conformer(s) did not converge" in caplog.text


def test_restrained_uff_relaxes_a_private_typeable_graph():
    cores = Chem.CombineMols(_sn2_core(), _sn2_core(), Point3D(8, 0, 0))  # each carbon donates to its chloride
    sulfur = placed("C[S+]=[CH0-2]", [(-1.8, 0, 0), (0, 0, 0), (1.6, 0, 0)])  # UFF alone crushes C-S to 1.43 A
    for mol, frozen, surrogates, retyped in (
        (placed("C=[B]C", [(0, 0, 0), (1.5, 0, 0), (3, 0, 0)]), set(), {1: (5, 6)}, set()),
        (sulfur, set(), {1: (16, 16)}, set()),
        (embedded("NC(=[Se])N"), set(), {2: (34, 16)}, set()),
        (cores, {0, 1, 2, 6, 7, 8}, {}, {(1, 2), (7, 8)}),
    ):
        drawn, record = (Chem.MolToSmiles(mol), [atom.GetFormalCharge() for atom in mol.GetAtoms()]), UFFRecord()
        assert np.isfinite(restrained_uff(mol, Constraints(frozen=frozen), max_iters=200, record=record)).all()
        assert (Chem.MolToSmiles(mol), [atom.GetFormalCharge() for atom in mol.GetAtoms()]) == drawn
        assert (record.surrogates, record.retyped) == (surrogates, retyped), drawn[0]
    assert GetBondLength(sulfur.GetConformer(), 1, 2) >= 1.55
    contact = Chem.RWMol(embedded("C.O"))
    contact.AddBond(0, 1, Chem.BondType.ZERO)
    assert np.isfinite(restrained_uff(contact, Constraints(), max_iters=0)).all()
    assert contact.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.ZERO
    zinc = Chem.AddHs(rx.parse_smiles("[NH3]->[Zn+2](<-[Cl-])<-[Cl-]"))  # N0 Zn1 Cl2 Cl3, relaxed as a bondless sphere
    assert rdDistGeom.EmbedMolecule(zinc, randomSeed=7) == 0
    windows, drawn = {(0, 1): (2.0, 2.1), (1, 2): (2.2, 2.3), (1, 3): (2.2, 2.3)}, Chem.MolToSmiles(zinc)
    restrained_uff(zinc, Constraints(metals={1}, distances=windows))
    lengths = {key: constraint_value(zinc.GetConformer().GetPositions(), key) for key in windows}
    assert lengths == {key: pytest.approx((lo + hi) / 2, abs=(hi - lo) / 2 + 0.01) for key, (lo, hi) in windows.items()}
    assert Chem.MolToSmiles(zinc) == drawn


def test_restrained_uff_scores_each_conformer_on_its_own_field():
    borane = embedded("CB(C)C")
    borane.GetConformer().SetId(4)
    shifted = Chem.Conformer(borane.GetConformer())
    shifted.SetId(9)
    shifted.SetAtomPosition(0, shifted.GetAtomPosition(0) + Point3D(0.2, 0.0, 0.0))
    borane.AddConformer(shifted, assignId=False)
    native = [rdForceFieldHelpers.UFFGetMoleculeForceField(borane, confId=cid).CalcEnergy() for cid in (4, 9)]
    assert restrained_uff(borane, Constraints(), max_iters=0) == pytest.approx(native)
    assert restrained_uff(borane, Constraints(), max_iters=0, conf_ids=[9]) == pytest.approx(native[1:])


def test_unbuildable_uff_objective_is_refused_and_restored():
    with pytest.raises(UFFTypingError, match=r"unsupported atom types \(B1\); use a different relaxation backend"):
        restrained_uff(placed("C=[B]B", np.zeros((3, 3))), Constraints(), max_iters=0)
    propane = embedded("CCC")
    before = propane.GetConformer().GetPositions()
    with pytest.raises(UFFOptimizationError, match="constraint setup failed"):
        restrained_uff(propane, Constraints(distances={(0, 99): (1.0, 1.0)}))
    np.testing.assert_array_equal(propane.GetConformer().GetPositions(), before)


def test_bonding_failure_names_a_stretched_bond_or_a_heavy_clash():
    """Ethanol C0 C1 O2 drawn without hydrogens; 1.87 A, inside 1.3 times the covalent sum but past the native bound,
    is how a butanediamine spanning trans on palladium closed its 7-ring."""
    ethanol, clash = [(0, 0, 0), (1.52, 0, 0), (2.0, 1.35, 0)], [(0, 0, 0), (1.43, 0, 0), (0, 0.9, 0)]
    stretched = [(-0.35, 0, 0), *ethanol[1:]]
    contact = Chem.RWMol(Chem.MolFromSmiles("C.O"))
    contact.AddBond(0, 1, Chem.BondType.ZERO)
    contact.UpdatePropertyCache(strict=False)
    rows = (
        ("CCO", ethanol, {}, None),
        ("CCO", stretched, {}, ("bond_length", (0, 1))),
        ("CCO", stretched, {"constrained": [(1, 0)]}, None),
        ("CCO", stretched, {"exclude": {0, 1}}, None),
        ("CCO", [*ethanol[:2], (2.42, 0, 0)], {}, ("bond_length", (1, 2))),
        ("CO.O", clash, {}, ("clash", (0, 2))),
        ("CO.O", clash, {"clash_tol": 0.0}, None),
        ("CO.O", clash, {"constrained": [(2, 0)]}, None),
        ("C[Zn]C", [(-5, 0, 0), (0, 0, 0), (1.95, 0, 0)], {}, None),
        (contact.GetMol(), [(0, 0, 0), (5, 0, 0)], {}, None),
        ("CCO", [*ethanol[:2], (np.nan, 0, 0)], {}, ("nonfinite", ())),
    )
    found = [bonding_failure(placed(mol, points), 0, **options) for mol, points, options, _expected in rows]
    assert [failure and (failure.kind, failure.atoms) for failure in found] == [row[3] for row in rows]


def _methanols(shift, atoms=slice(None)):
    """Return two methanols in one conformation, `atoms` of the second translated by `shift`."""
    left, right = embedded("CO"), embedded("CO")
    positions = right.GetConformer().GetPositions()
    positions[atoms] += shift
    right.GetConformer().SetPositions(positions)
    return left, right


def test_coupling_costs_an_rdkit_restraint_at_twice_its_weight():
    left, right = _methanols((2.0, 0.2, -0.1))
    coupling = Coupling(atoms=np.array([0, 1]), weights=np.array([1.7, 0.4]), spectators=1)
    positions = [mol.GetConformer().GetPositions() for mol in (left, right)]
    penalty, costs = coupling.penalty(*positions), []
    for side, mol in enumerate((left, right)):
        base, held = (rdForceFieldHelpers.UFFGetMoleculeForceField(mol) for _ in range(2))
        for atom, weight in zip(coupling.atoms, coupling.weights, strict=True):
            mol.GetConformer().SetAtomPosition(int(atom), positions[1 - side][atom].tolist())
            held.UFFAddPositionConstraint(int(atom), 0.0, float(2 * weight))
        mol.GetConformer().SetPositions(positions[side])
        base.Initialize()
        held.Initialize()
        gradient = (np.asarray(held.CalcGrad()) - base.CalcGrad()).reshape(-1, 3)
        costs.append((held.CalcEnergy() - base.CalcEnergy(), gradient))
    assert costs == [(pytest.approx(penalty.value), pytest.approx(g, abs=1e-10)) for g in (penalty.left, penalty.right)]


def test_alignment_rmsd_measures_spectators_without_fitting():
    left, right = _methanols((2.0, 0.2, -0.1))
    coupling = Coupling(atoms=np.array([0, 1]), weights=np.array([1.7, 0.4]), spectators=1)
    positions = [mol.GetConformer().GetPositions() for mol in (left, right)]
    assert coupling.alignment_rmsd(*positions) == pytest.approx(np.linalg.norm((2.0, 0.2, -0.1)))


def test_coupled_uff_relaxes_both_ends_on_one_budgeted_objective():
    """Two methanols coupled at C0 and O1, with a haptic centroid on their C-O bond the score rebuilds when moved."""
    coupling = Coupling(atoms=np.array([0, 1]), weights=np.array([5.0, 5.0]), spectators=1)
    face = Constraints(haptic={6: (0, 1)})
    for params, fixed, reason in (
        (CoEmbedParams(), None, "force"),
        (CoEmbedParams(), 0, "force"),
        (CoEmbedParams(joint_energy_tol=0.0, joint_max_iters=1, max_cycles=1), None, "cycle_limit"),
        (CoEmbedParams(joint_gtol=1e-9, joint_energy_tol=10.0, joint_max_iters=5), None, "energy"),
    ):
        left, right = _methanols((2.0, 0.2, 0.1))
        start = [mol.GetConformer().GetPositions() for mol in (left, right)]
        start_score = sum(restrained_uff(Chem.Mol(mol), face, max_iters=0)[0] for mol in (left, right))
        endpoints = [Endpoint(mol=mol, cons=face, record=UFFRecord()) for mol in (left, right)]
        joint = coupled_uff(endpoints, coupling, [0], 1.0, params, fixed=fixed)[0]
        positions, record = [mol.GetConformer().GetPositions() for mol in (left, right)], joint.record
        assert (record["stopping_reason"], record["converged"]) == (reason, reason in ("force", "energy"))
        assert record["iterations"] <= 2 * params.joint_max_iters * record["cycles"]
        assert record["cycles"] <= params.max_cycles
        assert record["score"] < start_score + coupling.penalty(*start).value
        assert record["score"] == pytest.approx(sum(joint.energies(positions)) + coupling.penalty(*positions).value)
        assert np.linalg.norm(positions[0][0] - positions[1][0]) < np.linalg.norm(start[0][0] - start[1][0])
        assert np.array_equal(positions[0], start[0]) == (fixed == 0)
        moved = [Chem.Mol(mol) for mol in (left, right)]
        for mol in moved:
            mol.GetConformer().SetPositions(mol.GetConformer().GetPositions() + 0.05)
        fresh = [restrained_uff(mol, face, max_iters=0)[0] for mol in moved]
        assert joint.energies([mol.GetConformer().GetPositions() for mol in moved]) == pytest.approx(fresh)


def test_unbuildable_coupling_registers_its_start_if_it_helps(caplog):
    """Registered on its spectator C0, a translated methanol moves onto the other; one with only C0 displaced stays."""
    broken = Constraints(distances={(0, 99): (1.0, 1.0)})
    for weights, atoms, registered in (((5.0, 5.0), slice(None), True), ((1.0, 5.0), 0, False)):
        left, right = _methanols((2.0, 0.2, 0.1), atoms)
        start = right.GetConformer().GetPositions()
        caplog.clear()
        with caplog.at_level("WARNING", logger="rxembed.relax"):
            endpoints = [Endpoint(mol=mol, cons=broken, record=UFFRecord()) for mol in (left, right)]
            coupling = Coupling(atoms=np.array([0, 1]), weights=np.array(weights), spectators=1)
            joint = coupled_uff(endpoints, coupling, [0], 1.0, CoEmbedParams())[0]
        assert (joint.record["stopping_reason"], joint.record["score"]) == ("failure", np.inf)
        assert joint.energies([mol.GetConformer().GetPositions() for mol in (left, right)]) == [np.inf, np.inf]
        assert "joint optimization failed for pair 0; restored its seed" in caplog.text
        on_left = np.allclose(right.GetConformer().GetPositions()[0], left.GetConformer().GetPositions()[0])
        assert (on_left, np.array_equal(right.GetConformer().GetPositions(), start)) == (registered, not registered)


def test_coembed_params_refuse_out_of_range_weights_and_budgets():
    refused: list[tuple[str, Any]] = [("alignment_weight", 0.0), ("react_weight", -1.0), ("joint_gtol", np.nan)]
    refused += [("joint_energy_tol", -1e-3), ("alignment_weight", True), ("seed_relax_iters", -1)]
    refused += [("joint_max_iters", 0), ("max_cycles", 2.0)]
    for name, value in refused:
        with pytest.raises(ValueError, match=f"{name} must be"):
            rx.CoEmbedParams(**{name: value})
    assert rx.CoEmbedParams(joint_energy_tol=0.0, seed_relax_iters=0).seed_relax_iters == 0
