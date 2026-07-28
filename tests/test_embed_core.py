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


# --- minimize records its drops, exactly as prune does -----------------------


def test_minimize_records_its_energy_window_drops(monkeypatch):
    """A conformer minimize() drops on its relax-energy window must land in `discarded`, not vanish silently.

    prune promises "Nothing is lost" — every conformer it merges away is recorded in `discarded`. minimize
    drops conformers too (energy window, torn bond, out-of-plane sphere, inverted donor hand, wrong stereo) and
    once did so *silently*, so that promise was false for any ensemble that saw a minimize first. Spike ONE
    conformer's FF energy far past the window: the real relax still runs (the geometry gates upstream see a
    normal structure and pass it through to the window), so the drop is the window's, and it must be recorded.
    """
    import numpy as np

    import rxembed as rx
    from rxembed import refine as _refine

    ens = rx.embed("CCCCO", n=8)  # unconstrained -> not yet minimized; `n` is a request, so read the real ids
    assert len(ens.ids) >= 2, "need >=2 conformers so dropping one still leaves a non-empty ensemble"
    victim = ens.ids[0]

    real = _refine.ff_energies  # what minimize() calls for an UNCONSTRAINED ensemble (pipeline.minimize, ~line 838)

    def spiked(mol, *args, **kwargs):
        e = np.asarray(real(mol, *args, **kwargs), dtype=float)  # the relax runs for real; only the number is faked
        for k, conf in enumerate(mol.GetConformers()):  # e is in conformer-enumeration order, mapped by GetId()
            if conf.GetId() == victim:
                e[k] = 1e4  # a non-physical energy for the victim alone -> its ΔE >> the 250 kcal/mol window
        return e

    monkeypatch.setattr(_refine, "ff_energies", spiked)  # the binding pipeline.minimize resolves at call time
    ens.minimize()

    assert victim not in ens.ids, "the energy window never fired — the test would be a null measurement"
    assert victim in ens.discarded, "minimize dropped the conformer but did not record it in `discarded`"


# --- minimize degrades on an untypable graph in BOTH relax entry points ------


def test_minimize_single_point_branch_degrades_on_untypable_graph(monkeypatch, caplog):
    """minimize()'s single-point relax must degrade like its sibling `_relax_constrained`, not crash.

    `_relax_constrained` wraps `restrained_uff` in try/except RuntimeError so an untypable / hypervalent
    reacting core keeps its embedded geometry. The single-point branch taken once `embed` has relaxed the
    seeds (`_seeds_relaxed=True`) called `restrained_uff` WITHOUT that guard, so it propagated the error
    where the sibling degraded — and `_relax_into_windows` sets `_seeds_relaxed` even when its own relax
    build failed, so a later `.minimize()` re-hits the same graph. RDKit's UFF builds even for actinides, so
    the natural trigger is rare; fault-inject the RuntimeError to prove both entry points now degrade alike.
    """
    import rxembed as rx
    from rxembed import refine as _refine

    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert ens.ids, "embed produced no conformers"
    assert ens._seeds_relaxed, "embed did not mark the seeds relaxed -> minimize won't take the single-point branch"
    assert not ens._minimized, "minimize must still run"
    kept = list(ens.ids)

    def raiser(*a, **kw):  # UFFGetMoleculeForceField raises at build time on an untypable graph
        raise RuntimeError("UFF: could not type atom")

    monkeypatch.setattr(_refine, "restrained_uff", raiser)  # pipeline.minimize resolves the binding at call time
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.minimize()  # single-point branch -> must NOT propagate the RuntimeError

    assert any("UFF could not relax" in r.message for r in caplog.records), "the guard never fired (null test)"
    assert ens.ids == kept, "the embedded geometry must be kept when the single point cannot be typed"
