"""Test restrained UFF and geometry-based bond acceptance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, GetDihedralDeg, SetDihedralDeg

from rxembed import relax as relax_module
from rxembed.constraints import Constraints
from rxembed.relax import (
    FF_SURROGATE,
    UFF_GHOST,
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


def test_exemption_is_per_pair_not_per_atom():
    mol = _stretch(_stretch(_mol(), 1, 2, 2.4), 0, 1, 4.0)
    assert not bonding_ok(mol, 0, constrained={(1, 2): (2.35, 2.45)}), (
        "stating C-Cl must not excuse the torn C-C that shares atom 1"
    )


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


@pytest.mark.parametrize("smiles", ["P=[Se]->[Li]", "P=[Se][Pd]"], ids=["dative-metal", "covalent-metal"])
def test_selenium_uses_available_uff_type_on_private_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.5, 2.0, 0.0))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    before = Chem.MolToSmiles(mol)

    energies = restrained_uff(mol, Constraints(metals={2}), max_iters=5)

    assert np.isfinite(energies).all()
    assert Chem.MolToSmiles(mol) == before
    assert mol.GetAtomWithIdx(1).GetHybridization() == Chem.HybridizationType.SP2


def test_selenium_fallback_does_not_retype_nonterminal_pse():
    mol = Chem.MolFromSmiles("P=[Se](C)->[Li]")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    with pytest.raises(RuntimeError, match="UFF has unsupported atom types outside the fixed core"):
        restrained_uff(mol, Constraints(metals={3}), max_iters=0)


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


def test_force_field_minimizer_failure_rolls_back_the_batch(monkeypatch):
    mol = _mol()
    failed = mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    before = {c.GetId(): c.GetPositions().copy() for c in mol.GetConformers()}

    def force_field(target, **kwargs):
        conf_id = kwargs["confId"]

        def diverge(**kwargs):
            target.GetConformer(conf_id).SetAtomPosition(0, (99.0, 99.0, 99.0))
            if conf_id == failed:
                raise RuntimeError("BFGS diverged")

        return SimpleNamespace(Initialize=lambda: None, Minimize=diverge, CalcEnergy=lambda: 12.5)

    monkeypatch.setattr(relax_module._mech, "MECHANISM_ORDER", ())
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFHasAllMoleculeParams", lambda mol: True)
    monkeypatch.setattr(relax_module.rdForceFieldHelpers, "UFFGetMoleculeForceField", force_field)

    with pytest.raises(RuntimeError, match="UFF minimization failed: BFGS diverged"):
        restrained_uff(mol, Constraints())
    for cid, positions in before.items():
        assert np.array_equal(mol.GetConformer(cid).GetPositions(), positions), (
            "a failed batch left partial coordinates"
        )


def test_untypable_fixed_core_still_relaxes_periphery(caplog):
    mol = _sn2()
    before = mol.GetConformer().GetPositions().copy()

    with caplog.at_level("WARNING", logger="rxembed.relax"):
        energies = restrained_uff(mol, Constraints(frozen={0, 1, 2}))

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
    assert "retyped 1 fixed-core bond(s) as outward dative edges" in caplog.text


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
    assert "retyped 2 fixed-core bond(s)" in caplog.text


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
