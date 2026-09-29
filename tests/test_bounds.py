"""Test RDKit bounds editing, smoothing and coordinate seeding."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import bounds as bnd
from rxembed import metal_core
from rxembed.constraints import Constraints, add_distance


def _graph(smiles):
    """Return a Mol with explicit Hs and no conformer for seed-count tests."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _mol(smiles="CCO", seed=1):
    mol = _graph(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


# ---------------------------------------------------------------------------------------------------------
# embed_parameters: native model and reproducible sampling defaults
# ---------------------------------------------------------------------------------------------------------


def test_native_parameters_reach_rdkit_without_model_fallback_or_stale_bounds():
    """The caller's own EmbedParams(native=...) object is restored to its prior state after use."""
    native = rdDistGeom.srETKDGv3()
    native.useRandomCoords = True
    params = bnd.EmbedParams(seed=42, threads=1, prune_rms=-1, native=native)
    # Set the caller's own pre-existing native fields only after wrapping: EmbedParams requires them at
    # RDKit's defaults, but seed_coordinates must still save and restore whatever the caller had before it.
    native.randomSeed, native.numThreads, native.pruneRmsThresh = 7, 2, 0.4
    before = json.loads(rdDistGeom.EmbedParametersToJSON(native))

    assert bnd.seed_coordinates(_graph("CCCC"), Constraints(), 1, params)

    after = json.loads(rdDistGeom.EmbedParametersToJSON(native))
    assert after.pop("boundsMatrix")
    assert after == before


# ---------------------------------------------------------------------------------------------------------
# EmbedParams: one owner per setting
# ---------------------------------------------------------------------------------------------------------


def test_seed_or_threads_together_with_params_is_a_loud_conflict():
    with pytest.raises(ValueError, match=r"dataclasses\.replace"):
        bnd.resolve_params(bnd.EmbedParams(), 5, None)
    with pytest.raises(ValueError, match=r"dataclasses\.replace"):
        bnd.resolve_params(bnd.EmbedParams(), None, 2)
    assert bnd.resolve_params(None, 5, 2) == bnd.EmbedParams(seed=5, threads=2)
    assert bnd.resolve_params(None, None, None) == bnd.EmbedParams()


@pytest.mark.parametrize(("rdkit_name", "field_name"), [("pruneRmsThresh", "prune_rms")])
def test_native_sampling_fields_must_stay_at_rdkits_own_defaults(rdkit_name, field_name):
    native = rdDistGeom.KDG()
    setattr(native, rdkit_name, 42 if rdkit_name != "pruneRmsThresh" else 0.4)
    with pytest.raises(ValueError, match=f"native.{rdkit_name}.*EmbedParams\\({field_name}="):
        bnd.EmbedParams(native=native)


def test_knowledge_together_with_native_must_agree_or_stay_unset():
    with pytest.raises(ValueError, match="useBasicKnowledge"):
        bnd.EmbedParams(native=rdDistGeom.KDG(), knowledge=False)
    with pytest.raises(ValueError, match="useBasicKnowledge"):
        bnd.EmbedParams(native=rdDistGeom.ETDG(), knowledge=True)
    assert bnd.EmbedParams(native=rdDistGeom.KDG(), knowledge=True).knowledge
    assert bnd.EmbedParams(native=rdDistGeom.KDG()).native.useBasicKnowledge  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------------------------------------
# probe_conformer: a throwaway geometry that decides a discrete question
#
# That its seed decides the answer, rather than process history, is asserted end to end on the one caller
# that consumes it: `tests/test_embed.py`'s encounter-bounds pair.
# ---------------------------------------------------------------------------------------------------------


def test_ligand_reach_keeps_native_bounds_for_ring_closure():
    ring = Chem.MolFromSmiles("Pc1ccccc1P")
    native = rdDistGeom.GetMoleculeBoundsMatrix(ring, set14bounds=True, set15bounds=False)
    assert bnd.ligand_reach(ring)[0, 7] == pytest.approx(native[0, 7])

    # A saturated ring's central bond is not aromatic, so only the native 1-4 lower bound carries over;
    # the upper bound is free-torsion, which is looser than RDKit's sp2-sp2 cis template (4.456 -> 4.552).
    saturated = Chem.MolFromSmiles("PC1CCCCC1P")
    native = rdDistGeom.GetMoleculeBoundsMatrix(saturated, set14bounds=True, set15bounds=False)
    assert bnd.ligand_reach(saturated)[0, 7] >= native[0, 7] - 1e-9

    chain = Chem.MolFromSmiles("PCCCCP")
    free = bnd.ligand_reach(chain)
    native = rdDistGeom.GetMoleculeBoundsMatrix(chain, set14bounds=True, set15bounds=False)
    assert free[0, 5] > native[0, 5]


@pytest.mark.parametrize("window", [(1.0, math.inf), (3.0, 3.1)])
def test_three_bond_reach_abstains_without_supported_local_triangles(window):
    matrix = np.ones((4, 4))
    matrix[1, 0], matrix[0, 1] = window
    assert math.isinf(bnd._chain_upper(matrix, (0, 1, 2, 3)))


def test_inconsistent_ligand_reach_is_not_repaired(monkeypatch):
    monkeypatch.setattr(bnd.DistanceGeometry, "DoTriangleSmoothing", lambda *_args: False)
    with pytest.raises(ValueError, match="native ligand reach bounds are inconsistent"):
        bnd.ligand_reach(_graph("CCCN"))


# ---------------------------------------------------------------------------------------------------------
# _smooth: the tolerance is the signal
# ---------------------------------------------------------------------------------------------------------


def test_carried_nitrile_bond_and_real_iron_seed_from_the_surrogate_basis():
    """A carried sp-donor bond and its restored Fe(II) leave the bounds of the bondless-carbon surrogate graph."""
    real = Chem.AddHs(Chem.MolFromSmiles("CC#N.[Fe+2]"))  # C0 C1 N2 Fe3, methyl H4-H6
    surrogate = Chem.RWMol(real)
    surrogate.GetAtomWithIdx(3).SetAtomicNum(metal_core.SURROGATE)
    surrogate.GetAtomWithIdx(3).SetFormalCharge(0)
    surrogate = surrogate.GetMol()
    surrogate.UpdatePropertyCache(strict=False)
    cons = Constraints(metals={3}, distances={(2, 3): (1.9, 1.95)})

    carried = bnd._write(metal_core.connect_metal(real, [(2, 3)]), cons).bm

    np.testing.assert_array_equal(carried, bnd._write(surrogate, cons).bm)


# ---------------------------------------------------------------------------------------------------------
# _feasible_bounds: the edit, and what happens when it cannot be satisfied
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# _cap_fragment_contacts: every free component repels every other at its van der Waals floor and stays
# within a shared, formula-derived ceiling; no probe conformer, no chosen contact pair
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# embed: the native distance-geometry call itself
# ---------------------------------------------------------------------------------------------------------


def test_native_failure_counts_are_reported_once_with_a_random_start_remedy(monkeypatch, caplog):
    counts = [0] * (max(map(int, rdDistGeom.EmbedFailureCauses.names.values())) + 1)
    counts[int(rdDistGeom.EmbedFailureCauses.INITIAL_COORDS)] = 2

    def reject(_mol, _n, params):
        assert params.trackFailures
        return []

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", reject)
    monkeypatch.setattr(rdDistGeom.EmbedParameters, "GetFailureCounts", lambda _self: tuple(counts))
    with caplog.at_level("DEBUG", logger="rxembed.bounds"):
        assert not bnd.seed_coordinates(_graph("CC"), Constraints(), 1, bnd.EmbedParams(seed=42))

    messages = [record.message for record in caplog.records if record.name == "rxembed.bounds"]
    assert len(messages) == 1
    assert "random=False" in messages[0]
    assert "{'INITIAL_COORDS': 2}" in messages[0]
    assert "EmbedParams(native=...), useRandomCoords=True" in messages[0]


# ---------------------------------------------------------------------------------------------------------
# seed_count: scale the seed count by flexibility rather than holding it flat
# ---------------------------------------------------------------------------------------------------------


def test_seed_count_scales_and_clamps():
    rigid, mid, floppy = _graph("CCO"), _graph("C" * 25), _graph("C" * 60)
    assert bnd.seed_count(rigid) > 10  # RDKit's own flat default, which this exists to replace
    assert bnd.seed_count(rigid) < bnd.seed_count(mid) < bnd.seed_count(floppy)
    assert bnd.seed_count(rigid) == bnd.seed_count(_graph("CC")), "the floor must clamp a rigid molecule"
    assert bnd.seed_count(floppy) == bnd.seed_count(_graph("C" * 120)), "the ceiling must clamp a long chain"
    for mol in (rigid, mid):
        assert bnd.seed_count(mol, constrained=True) > bnd.seed_count(mol)


# ---------------------------------------------------------------------------------------------------------
# The angle rules; how an angle-derived window meets the matrix
#
#   R1  a stated `cons.distances` window on the angle's 1-3 pair wins outright; the angle is discarded
#   R2  a real bond path between the end atoms -> intersect the angle-derived window with the matrix
#   R2' ...and if that intersection is disjoint, the backbone wins and the angle contributes nothing
#   R3  no bond path (the atoms meet only through the stripped metal) -> write the angle outright
#   C1  a coplanar entry whose M-D-X angle is unstated contributes nothing (a 1,4 distance carries no
#       dihedral information until that angle is pinned)
#   C2  the 1,4 edge is an EXTREMUM over the stated angle window, never a single pinned angle
#
# Each is built on a hand-made topology, because the branch it selects is chosen by the topology and no real
# molecule offers all six. C1 in particular is silent when wrong: a refactor that drops the skip, or that
# "helpfully" derives the missing angle from the matrix, changes the bounds and raises nothing.
# ---------------------------------------------------------------------------------------------------------


def _rule_mol(smi):
    return Chem.AddHs(Chem.MolFromSmiles(smi))


def _edited(mol, cons):
    """The edited bounds matrix alone; `_feasible_bounds` also returns the tolerance it settled at."""
    bm, _tol = bnd._feasible_bounds(mol, cons)
    return bm


def _coplanar_case(angle_window):
    """A 4-atom chain with an explicit coplanar cap; `angle_window` optionally pins its M-D-X angle."""
    mol = _rule_mol("C.C.C.C")  # disconnected, so nothing but our own windows reaches the matrix
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)  # pins th_jkw, which is derived from the matrix legs
    if angle_window is not None:
        cons.angles[(0, 1, 2)] = angle_window
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    return mol, cons


@pytest.mark.parametrize(("atoms", "overrides"), [((0, 1, 2, 3), True)])
def test_stated_torsion_supersedes_coplanar_seed_bound(atoms, overrides):
    mol, cons = _coplanar_case((118.0, 122.0))
    without_cap = _edited(mol, cons.copy(coplanar=[]))
    with_cap = _edited(mol, cons)
    assert not np.array_equal(with_cap, without_cap), "the unoverridden cap must be active"

    stated = cons.copy(dihedrals={atoms: (89.0, 91.0)})
    np.testing.assert_array_equal(_edited(mol, stated), without_cap if overrides else with_cap)


def test_repeated_inactive_cap_does_not_replace_stronger_owner():
    mol, cons = _coplanar_case((118.0, 122.0))
    row = cons.coplanar[0]
    weak, strong = (*row[:5], 45.0), (*row[:5], 20.0)
    cons.coplanar = [weak, strong, weak]
    ctx = bnd._write(mol, cons)
    np.testing.assert_array_equal(ctx.bm, bnd._write(mol, cons.copy(coplanar=[strong])).bm)
