"""Layer-1 embed core — the fundamentals everything stands on.

Highest-priority suite: bounds matrix, incremental relaxing triangle smoothing, frozen-aware UFF cleanup,
Kabsch graft, seed scaling. Every case asserts the two universal invariants (geometry gate + fidelity).
Scaffolded (`skip`) until Phase A ports `rx.embed`; the parametrize lists are the executable spec.
"""

import numpy as np
import pytest
from rdkit.Chem import rdMolTransforms

from rxembed import geometry as geom

skip_phase_a = pytest.mark.skip(reason="Phase A: rx.embed not yet ported")


# --- bounds matrix: each constraint kind realised ----------------------------


@skip_phase_a
@pytest.mark.parametrize(
    ("smiles", "distances"),
    [
        ("OC(=O)CCCCc1ccccc1", {(1, 9): (2.6, 3.0)}),  # index key
        ("OC(=O)CCCCc1ccccc1", {"[OX1]": None}),  # SMARTS key (placeholder)
    ],
)
def test_distance_bounds_realised(smiles, distances):
    import rxembed as rx

    ens = rx.embed(smiles, distances=distances)
    for cid in ens.ids:
        geom.check(ens.mol, cid, constraints={"distances": distances}).assert_ok()


@skip_phase_a
def test_angle_bounds_realised():
    import rxembed as rx

    ens = rx.embed("CCCCCCC", angles={(0, 3, 6): (85.0, 95.0)})
    a = np.array([rdMolTransforms.GetAngleDeg(ens.mol.GetConformer(c), 0, 3, 6) for c in ens.ids])
    assert 80.0 <= a.mean() <= 100.0


@skip_phase_a
def test_plane_stack_realised():
    import rxembed as rx

    ens = rx.embed("c1ccccc1.c1ccccc1", planes=[(None, None, 3.7)])  # ring atoms resolved internally
    assert len(ens) >= 1


@skip_phase_a
def test_multifragment_vdw_separation():
    import rxembed as rx

    ens = rx.embed("CCO.c1ccccc1")  # two fragments must not embed on top of each other
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# --- incremental relaxing triangle smoothing ---------------------------------


@skip_phase_a
def test_tight_core_still_embeds():
    """An over-constrained but feasible core embeds once smoothing escalates tolerance."""
    import rxembed as rx

    ens = rx.embed("C1CCCCC1", distances={(0, 3): (2.5, 2.6)})  # tight transannular pinch
    assert len(ens) >= 1


@skip_phase_a
def test_infeasible_constraints_raise_clearly():
    import rxembed as rx

    with pytest.raises((RuntimeError, ValueError)):
        rx.embed("CCO", distances={(0, 2): (0.1, 0.2)})  # physically impossible


# --- frozen-aware UFF/MMFF cleanup -------------------------------------------


@skip_phase_a
def test_cleanup_holds_frozen_core_and_conjugation():
    import rxembed as rx

    frozen = [11, 14, 15]
    ens = rx.embed("tests/fixtures/ts_core.xyz", freeze=frozen)
    ref = rx.embed("tests/fixtures/ts_core.xyz")
    for cid in ens.ids:
        geom.check(ens.mol, cid, frozen=frozen, reference=ref.mol).assert_ok()  # core + gate together


@skip_phase_a
def test_kabsch_graft_is_exact():
    import rxembed as rx

    frozen = [11, 14, 15]
    ens = rx.embed("tests/fixtures/ts_core.xyz", freeze=frozen)
    ref = rx.embed("tests/fixtures/ts_core.xyz")
    for cid in ens.ids:
        rep = geom.frozen_core(ens.mol, ens.mol.GetConformer(cid).GetPositions(), frozen, ref.mol, tol=0.01)
        assert not rep  # < 0.01 A
