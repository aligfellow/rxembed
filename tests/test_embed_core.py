"""Integration tests for the embed core — ``rx.embed(...).minimize()`` on fast, xtb-free systems.

Layer-1 fundamentals through the real public API: the bounds-matrix edit, multi-fragment vdW separation,
the restrained-UFF enforcement of ``constrain`` windows and ``fix`` numbers, and the universal geometry
gate. Small SMILES only (no metals, no xtb, no openconf) so the suite runs anywhere in well under a second.
The frozen-core graft, template transfer, and constraint stacking live in ``test_frozen.py``.

Tolerances are deliberately tight: a ``constrain`` window must be realised inside itself (+ the same small
slack the pipeline's own validator allows), and a ``fix`` number must land within 0.1 Å / 8° of target —
loose enough for the flat-bottom UFF compromise, strict enough to catch a constraint that is silently
ignored.
"""

import pytest

from rxembed import geometry as geom

_SLACK_A = 0.15  # the pipeline's own _validate distance slack — a window realised within this is "held"
_FIX_TOL_A = 0.1  # a numbers-fix distance must land this close to target (flat-bottom UFF compromise)
_FIX_TOL_DEG = 8.0


# --- baseline: clean peripheries ---------------------------------------------


@pytest.mark.parametrize("smiles", ["CCO", "OC(=O)CCCCc1ccccc1", "C1CC1C(=O)O", "c1ccc2ccccc2c1"])
def test_free_embed_is_clean(smiles):
    import rxembed as rx

    ens = rx.embed(smiles, n=6).minimize()  # the gate is the acceptance test on settled geometries
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


def test_multifragment_vdw_separation():
    import rxembed as rx

    ens = rx.embed("CCO.c1ccccc1", n=6).minimize()  # two fragments must not embed on top of each other
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# --- constrain: soft windows realised ----------------------------------------


def test_constrain_distance_realised_in_window():
    import rxembed as rx

    lo, hi = 2.6, 3.0
    smi = "OC(=O)CCCCc1ccccc1"
    ens = rx.embed(smi, constrain={(1, 9): (lo, hi)}, n=10).minimize()
    assert ens.n >= 1
    stats = ens.measure((1, 9))
    assert stats["min"] >= lo - _SLACK_A  # every conformer inside the window...
    assert stats["max"] <= hi + _SLACK_A
    free = rx.embed(smi, n=10).minimize().measure((1, 9))  # and the window actually bit: free sits outside it
    assert free["mean"] > hi + _SLACK_A


def test_constrain_angle_realised():
    import rxembed as rx

    ens = rx.embed("CCCCCCC", constrain={(0, 3, 6): (85.0, 95.0)}, n=10).minimize()
    stats = ens.measure((0, 3, 6))
    assert stats["min"] >= 85.0 - 5.0  # every conformer inside the window (+ validator slack)...
    assert stats["max"] <= 95.0 + 5.0
    free = rx.embed("CCCCCCC", n=10).minimize().measure((0, 3, 6))  # ...and the window bit (free ranges wide)
    assert free["max"] > 95.0 + 5.0


def test_constrain_plane_stacks_two_rings():
    import numpy as np
    from rdkit import Chem

    import rxembed as rx

    # 1,4-diphenylbutane: the tethered phenyls flop apart when free; the plane must pull them into a stack
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CCCCc1ccccc1"))
    ra, rb = (tuple(r) for r in mol.GetRingInfo().AtomRings()[:2])

    def sep_align(ens):
        seps, aligns = [], []
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            a, b = pos[list(ra)], pos[list(rb)]
            na, nb = np.linalg.svd(a - a.mean(0))[2][2], np.linalg.svd(b - b.mean(0))[2][2]
            seps.append(float(np.linalg.norm(a.mean(0) - b.mean(0))))
            aligns.append(abs(float(np.dot(na, nb))))
        return float(np.mean(seps)), float(np.mean(aligns))

    stacked_sep, stacked_align = sep_align(rx.embed(mol, constrain={(ra, rb): 3.7}, n=10).minimize())
    free_sep, _ = sep_align(rx.embed(mol, n=10).minimize())
    assert abs(stacked_sep - 3.7) < 0.5  # centroid separation near the requested 3.7 A
    assert stacked_align > 0.9  # rings brought parallel (|n_a . n_b| ~ 1), i.e. an actual stack
    assert free_sep > stacked_sep + 2.0  # and the constraint genuinely bit (free phenyls sit far apart)


# --- fix numbers: a TS core from SMILES (DESIGN W3) --------------------------


def test_fix_numbers_deliver_sn2_core():
    import rxembed as rx

    # [F-].CCl -> F(0), C(1), Cl(2): a linear 3-centre SN2 core specified purely by numbers. NB the near-linear
    # 178° triangle is numerically degenerate for RDKit's bounds-smoothing eigenvalue start (the geometry is
    # valid; smoothing "feasibility" is not geometric validity), so ~2% of process configs it embeds distorted
    # and the relax tears every seed -> 0. More seeds give the valid-but-marginal core enough shots to survive.
    ens = rx.embed("[F-].CCl", fix={(0, 1): 2.0, (1, 2): 2.2, (0, 1, 2): 178.0}, n=24).minimize()
    assert ens.n >= 1
    assert abs(ens.measure((0, 1))["mean"] - 2.0) < _FIX_TOL_A
    assert abs(ens.measure((1, 2))["mean"] - 2.2) < _FIX_TOL_A
    assert abs(ens.measure((0, 1, 2))["mean"] - 178.0) < _FIX_TOL_DEG


# --- feasibility ------------------------------------------------------------


def test_tight_but_feasible_core_embeds():
    import rxembed as rx

    ens = rx.embed("C1CCCCC1", constrain={(0, 3): (2.5, 2.6)}, n=6).minimize()  # tight transannular pinch
    assert ens.n >= 1
    assert ens.measure((0, 3))["max"] <= 2.6 + _SLACK_A  # the pinch actually held (free d(0,3) ~ 2.9)


def test_infeasible_fix_raises():
    import rxembed as rx

    with pytest.raises((RuntimeError, ValueError)):
        rx.embed("CCO", fix={(0, 2): 0.15}, n=4)  # a physically impossible C..O distance


def test_unknown_kwarg_is_rejected():
    import rxembed as rx

    with pytest.raises(TypeError, match="fix / constrain / template"):
        rx.embed("CCO", freeze=[0, 1, 2])  # the old kwarg is gone — fail loudly, don't silently ignore
