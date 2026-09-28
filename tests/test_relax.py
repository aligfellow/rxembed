"""Test restrained UFF and geometry-based bond acceptance."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.rdMolTransforms import GetBondLength

from rxembed import relax as relax_module
from rxembed.constraints import Constraints
from rxembed.relax import (
    UFFRecord,
    ff_energies,
    restrained_uff,
)


def _mol(smiles="CCCl", seed=7):
    """Chloroethane by default: C0-C1-Cl2, so a tear can be made in the middle or at the end."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
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
# bonding_failure: the arbiter
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# restrained_uff; relax, and score-without-moving
# ---------------------------------------------------------------------------------------------------------


def _collapsed(smiles="CCCC"):
    """A molecule with every atom pushed onto one axis: a seed too strained to converge in one iteration."""
    mol = _mol(smiles)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (0.01 * i, 0.0, 0.0))
    return mol


def test_restrained_uff_defers_internal_optimizer_status_to_its_acceptance_gate(caplog):
    """A caller that hands its own record reads the raw optimizer status instead of an unsolicited warning."""
    record = UFFRecord()
    with caplog.at_level("WARNING", logger="rxembed.relax"):
        restrained_uff(_collapsed(), Constraints(), max_iters=1, record=record)

    assert record.statuses[0] != 0, "the fixture must fail to converge for this test to mean anything"
    assert "did not converge" not in caplog.text

    restrained_uff(_collapsed(), Constraints(), max_iters=1)
    assert "did not converge" in caplog.text


def test_isolated_untyped_boron_uses_a_private_carbon_type():
    mol = Chem.MolFromSmiles("C=[B]C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before = Chem.MolToSmiles(mol)
    record = UFFRecord()

    energies = restrained_uff(mol, Constraints(), max_iters=5, record=record)

    assert np.isfinite(energies).all()
    assert Chem.MolToSmiles(mol) == before
    assert record.surrogates == {1: (5, 6)}


def test_dithiocarbene_donor_gets_a_recognised_sulfur_charge_state():
    """Re-derive a divalent double-bonded sulfur's hybridisation from its sigma degree.

    UFF's native charge-blind double-bond radius crushes the C-S bond to ~1.43 A; the recognised type holds
    it near 1.59 A.
    """
    mol = Chem.MolFromSmiles("C[S+]=[CH0-2]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((-1.8, 0.0, 0.0), (0.0, 0.0, 0.0), (1.6, 0.0, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    before_smiles = Chem.MolToSmiles(mol)
    record = UFFRecord()

    energies = restrained_uff(mol, Constraints(), max_iters=200, record=record)

    length = GetBondLength(mol.GetConformer(0), 1, 2)
    assert np.isfinite(energies).all()
    assert length >= 1.55, f"the C-S bond crushed to {length:.3f} A"
    assert [a.GetFormalCharge() for a in mol.GetAtoms()] == before_charges, "public formal charges moved"
    assert Chem.MolToSmiles(mol) == before_smiles, "the public molecule was retyped, not just its private FF graph"
    assert record.surrogates == {1: (16, 16)}


def test_boron_network_is_not_retyped_as_carbon():
    """A connected boron network has no per-atom carbon surrogate: UFF is left to reject it outright."""
    mol = Chem.MolFromSmiles("C=[B]B")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    with pytest.raises(relax_module.UFFTypingError, match="B1"):
        restrained_uff(mol, Constraints(), max_iters=0)


def test_plain_ff_energy_rejects_an_incomplete_uff_objective():
    mol = Chem.MolFromSmiles("NC(=[Se])N")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    with pytest.raises(relax_module.UFFTypingError, match="Se2"):
        ff_energies(mol, minimize=False)


def test_invalid_constraint_is_reported_as_setup_failure():
    mol = _mol()

    with pytest.raises(relax_module.UFFOptimizationError, match="constraint setup failed"):
        restrained_uff(mol, Constraints(distances={(0, 99): (1.0, 1.0)}), max_iters=0)


def test_ff_energies_scores_batch_on_its_original_field():
    """Each conformer's returned energy matches its own rebuilt UFF/MMFF field, not another conformer's."""
    smiles, mmff = "CB(C)C", False
    mol = _mol(smiles)
    mol.GetConformer().SetId(4)
    second = Chem.Conformer(mol.GetConformer())
    second.SetId(9)
    second.SetAtomPosition(0, second.GetAtomPosition(0) + Chem.rdGeometry.Point3D(0.2, 0.0, 0.0))
    mol.AddConformer(second, assignId=False)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol) == mmff

    recorded = {}
    energies = ff_energies(mol, max_iters=1, statuses=recorded)

    expected = []
    for conf in mol.GetConformers():
        if mmff:
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            field = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        else:
            field = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        expected.append(field.CalcEnergy())

    assert set(recorded) == {4, 9}
    np.testing.assert_allclose(sorted(energies), sorted(expected), atol=1e-6, rtol=0)


def test_fallback_retypes_multiple_reactive_centres(caplog):
    first = _sn2()
    second = _sn2()
    offset = first.GetNumAtoms()
    mol = Chem.CombineMols(first, second, Chem.rdGeometry.Point3D(8, 0, 0))
    frozen = {0, 1, 2, offset, offset + 1, offset + 2}

    with caplog.at_level("WARNING", logger="rxembed.relax"):
        assert np.isfinite(restrained_uff(mol, Constraints(frozen=frozen))).all()
    assert caplog.text.count("->Cl") == 2


# ---------------------------------------------------------------------------------------------------------
# the force-field surrogate
# ---------------------------------------------------------------------------------------------------------


def test_ff_surrogate_removes_only_private_zero_order_contacts():
    """A zero-order contact does not become a real UFF bond, and the public graph never carries the surrogate."""
    bond_type = Chem.BondType.ZERO
    mol = _mol("C.O")
    rw = Chem.RWMol(mol)
    rw.AddBond(0, 1, bond_type)
    mol = rw.GetMol()

    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == bond_type
    assert np.isfinite(restrained_uff(mol, Constraints(), max_iters=0)).all()
