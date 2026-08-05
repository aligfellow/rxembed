"""`relax.py`: the constraint-enforcing restrained UFF and `bonding_ok`, the arbiter every stage accepts on.

A restrained UFF can satisfy a window by pulling a bond apart, and no energy says so. That is why acceptance
is `bonding_ok` rather than convergence, and why the exemption it grants a *stated* pair is per PAIR and never
per atom. The stiffness ladder built on top of this lives in `embed.py`; see `test_embed.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetBondLength

from rxembed.constraints import Constraints
from rxembed.relax import FF_SURROGATE, UFF_GHOST, _bond_pruned, _ff_surrogate, bonding_ok, ff_energies, restrained_uff


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


# ---------------------------------------------------------------------------------------------------------
# bonding_ok: the arbiter
# ---------------------------------------------------------------------------------------------------------


def test_a_stated_distance_is_not_judged_as_a_bond():
    mol = _stretch(_mol(), 1, 2, 2.4)
    assert not bonding_ok(mol, 0), "unconstrained, a 2.4 A C-Cl should read as torn"
    assert bonding_ok(mol, 0, constrained={(1, 2): (2.35, 2.45)}), "a stated pair must be exempt"


def test_the_exemption_is_per_pair_not_per_atom():
    mol = _stretch(_stretch(_mol(), 1, 2, 2.4), 0, 1, 4.0)
    assert not bonding_ok(mol, 0, constrained={(1, 2): (2.35, 2.45)}), (
        "stating C-Cl must not excuse the torn C-C that shares atom 1"
    )


# ---------------------------------------------------------------------------------------------------------
# restrained_uff; relax, and score-without-moving
# ---------------------------------------------------------------------------------------------------------


def test_restrained_uff_pulls_a_stated_pair_toward_its_window():
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


def test_ff_energies_is_unconstrained_and_so_is_not_interchangeable():
    mol = _mol()
    cons = Constraints(distances={(1, 2): (2.35, 2.45)})  # unsatisfied by the seed -> a real penalty
    restrained = float(restrained_uff(Chem.Mol(mol), cons, max_iters=0)[0])
    plain = float(ff_energies(Chem.Mol(mol), minimize=False)[0])
    assert restrained != pytest.approx(plain), "the restrained energy carried no constraint penalty"


# ---------------------------------------------------------------------------------------------------------
# the two force-field preparations restrained_uff runs first
# ---------------------------------------------------------------------------------------------------------


def test_the_ff_surrogate_retypes_in_place_without_changing_an_index():
    organic = _mol()
    assert _ff_surrogate(organic, set(), ()) is organic, "no metal, no phantom, no copy"

    mol = _mol("CCO")
    out = _ff_surrogate(mol, {0}, (1,))
    assert out is not mol
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out.GetAtomWithIdx(0).GetAtomicNum() == FF_SURROGATE
    assert out.GetAtomWithIdx(1).GetAtomicNum() == UFF_GHOST
    assert out.GetAtomWithIdx(0).GetDegree() == mol.GetAtomWithIdx(0).GetDegree()


def test_the_bond_prune_drops_only_frozen_frozen_bonds_and_is_skipped_when_there_are_none():
    mol = _mol()
    assert _bond_pruned(mol, frozen=set()) is None
    assert _bond_pruned(mol, frozen={0, 2}) is None, "C0 and Cl2 are two bonds apart, so nothing joins two frozen"

    out = _bond_pruned(mol, frozen={0, 1})  # the C-C bond joins two frozen atoms
    assert out is not None
    assert out.GetBondBetweenAtoms(0, 1) is None
    assert out.GetNumBonds() == mol.GetNumBonds() - 1
    assert out.GetBondBetweenAtoms(1, 2) is not None, "the frozen-free C-Cl bond was dropped too"
