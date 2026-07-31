"""`pipeline/api.py`: the public verbs, driven end to end through the real surface.

`rxembed.pipeline.embed(...).minimize()` on fast, xtb-free systems. These drive the PIPELINE verb
(`str -> Ensemble`), not the core one; they need no extra, so a base install runs them rather than skipping.
The equivalent contracts on the core verb are `tests/test_embed.py`, and what the returned object guarantees
is `tests/pipeline/test_ensemble.py`.

The bounds-matrix edit, multi-fragment vdW separation, restrained-UFF enforcement of `constrain` windows and
`fix` numbers, and the geometry gate. Small SMILES only, so the file runs in well under a second. The
frozen-core graft, template transfer and constraint stacking are in `tests/pipeline/test_dispatch.py`.

Tolerances are tight on purpose: a `constrain` window must be realised inside itself plus the same slack the
pipeline's own validator allows, and a `fix` number must land within 0.1 A / 8 deg. Loose enough for the
flat-bottomed UFF compromise, strict enough to catch a constraint that is silently ignored.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed.pipeline as rx
from rxembed import embed as core_embed
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.dispatch import _embed_dispatch

_SLACK_A = 0.15  # the pipeline's own _validate distance slack: a window realised within this is "held"
_FIX_TOL_A = 0.1  # a numbers-fix distance must land this close to target (flat-bottom UFF compromise)
_FIX_TOL_DEG = 8.0


# --- baseline: clean peripheries ---------------------------------------------


@pytest.mark.parametrize("smiles", ["CCO", "OC(=O)CCCCc1ccccc1", "C1CC1C(=O)O", "c1ccc2ccccc2c1"])
def test_free_embed_is_clean(smiles):
    import rxembed.pipeline as rx

    ens = rx.embed(smiles, n=6).minimize()  # the gate is the acceptance test on settled geometries
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


def test_multifragment_vdw_separation():
    import rxembed.pipeline as rx

    ens = rx.embed("CCO.c1ccccc1", n=6).minimize()  # two fragments must not embed on top of each other
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# --- constrain: soft windows realised ----------------------------------------


def test_constrain_distance_realised_in_window():
    import rxembed.pipeline as rx

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
    import rxembed.pipeline as rx

    ens = rx.embed("CCCCCCC", constrain={(0, 3, 6): (85.0, 95.0)}, n=10).minimize()
    stats = ens.measure((0, 3, 6))
    assert stats["min"] >= 85.0 - 5.0  # every conformer inside the window (+ validator slack)...
    assert stats["max"] <= 95.0 + 5.0
    free = rx.embed("CCCCCCC", n=10).minimize().measure((0, 3, 6))  # ...and the window bit (free ranges wide)
    assert free["max"] > 95.0 + 5.0


def test_constrain_plane_stacks_two_rings():
    import numpy as np
    from rdkit import Chem

    import rxembed.pipeline as rx

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
    """A linear 3-centre SN2 core stated purely as numbers is delivered; on every conformer, not on average.

    `[F-].CCl` -> F(0), C(1), Cl(2). Two distances and the angle between them, no reference geometry.

    Asserted per conformer with the RMSD prune OFF. With it on, this near-rigid six-atom system collapses 24
    seeds to one (correctly; they are the same shape), and the test then rests on that single conformer
    clearing the geometry gate, which it does not always do. That made it fail about one run in thirty. Judging
    every conformer instead is both deterministic and a stronger claim than the mean of the survivors.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("[F-].CCl"))
    fix = {(0, 1): 2.0, (1, 2): 2.2, (0, 1, 2): 178.0}
    confs = core_embed(mol, fix=fix, n=8, seed=0xF00D, prune_rms=-1).minimize()
    assert len(confs) == 8, "the prune is off, so every seed must come back"

    out = confs.mol
    for cid in confs.ids:
        conf = out.GetConformer(int(cid))
        assert abs(rdMolTransforms.GetBondLength(conf, 0, 1) - 2.0) < _FIX_TOL_A
        assert abs(rdMolTransforms.GetBondLength(conf, 1, 2) - 2.2) < _FIX_TOL_A
        assert abs(rdMolTransforms.GetAngleDeg(conf, 0, 1, 2) - 178.0) < _FIX_TOL_DEG


# --- feasibility ------------------------------------------------------------


def test_tight_but_feasible_core_embeds():
    import rxembed.pipeline as rx

    ens = rx.embed("C1CCCCC1", constrain={(0, 3): (2.5, 2.6)}, n=6).minimize()  # tight transannular pinch
    assert ens.n >= 1
    assert ens.measure((0, 3))["max"] <= 2.6 + _SLACK_A  # the pinch actually held (free d(0,3) ~ 2.9)


def test_infeasible_fix_raises():
    import rxembed.pipeline as rx

    with pytest.raises((RuntimeError, ValueError)):
        rx.embed("CCO", fix={(0, 2): 0.15}, n=4)  # a physically impossible C..O distance


def test_unknown_kwarg_is_rejected():
    import rxembed.pipeline as rx

    with pytest.raises(TypeError, match="fix / constrain / template"):
        rx.embed("CCO", freeze=[0, 1, 2])  # the old kwarg is gone; fail loudly, don't silently ignore


# --- minimize records its drops, exactly as prune does -----------------------


def test_minimize_records_its_energy_window_drops(monkeypatch):
    """A conformer minimize() drops on its relax-energy window must land in `discarded`, not vanish silently.

    prune promises "Nothing is lost": every conformer it merges away is recorded in `discarded`. minimize
    drops conformers too (energy window, torn bond, out-of-plane sphere, inverted donor hand, wrong stereo) and
    once did so *silently*, so that promise was false for any ensemble that saw a minimize first. Spike one
    conformer's FF energy far past the window: the real relax still runs (the geometry gates upstream see a
    normal structure and pass it through to the window), so the drop is the window's, and it must be recorded.
    """
    import numpy as np

    import rxembed.pipeline as rx
    from rxembed.pipeline import calculators as _refine

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

    assert victim not in ens.ids, "the energy window never fired: the test would be a null measurement"
    assert victim in ens.discarded, "minimize dropped the conformer but did not record it in `discarded`"


# --- minimize degrades on an untypable graph in both relax entry points ------


def test_minimize_single_point_branch_degrades_on_untypable_graph(monkeypatch, caplog):
    """minimize()'s single-point relax must degrade like its sibling `_relax_constrained`, not crash.

    `_relax_constrained` wraps `restrained_uff` in try/except RuntimeError so an untypable / hypervalent
    reacting core keeps its embedded geometry. The single-point branch taken once `embed` has relaxed the
    seeds (`_seeds_relaxed=True`) called `restrained_uff` without that guard, so it propagated the error
    where the sibling degraded, and `_relax_into_windows` sets `_seeds_relaxed` even when its own relax
    build failed, so a later `.minimize()` re-hits the same graph. RDKit's UFF builds even for actinides, so
    the natural trigger is rare; fault-inject the RuntimeError to prove both entry points now degrade alike.
    """
    import rxembed.pipeline as rx
    from rxembed.pipeline import calculators as _refine

    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert ens.ids, "embed produced no conformers"
    assert ens._seeds_relaxed, "embed did not mark the seeds relaxed -> minimize won't take the single-point branch"
    assert not ens._minimized, "minimize must still run"
    kept = list(ens.ids)

    def raiser(*a, **kw):  # UFFGetMoleculeForceField raises at build time on an untypable graph
        raise RuntimeError("UFF: could not type atom")

    monkeypatch.setattr(_refine, "restrained_uff", raiser)  # pipeline.minimize resolves the binding at call time
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.minimize()  # single-point branch -> must not propagate the RuntimeError

    assert any("UFF could not relax" in r.message for r in caplog.records), "the guard never fired (null test)"
    assert ens.ids == kept, "the embedded geometry must be kept when the single point cannot be typed"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": -1}, "not reproducible"),
        ({"n": 0}, "positive conformer count"),
    ],
)
def test_the_pipeline_embed_refuses_a_silently_wrong_argument(kwargs, match):
    """The pipeline reaches the core at `seed_conformers`, so the seam, not the front door, owns these.

    `rx.embed('CCO', seed=-1)` used to be accepted and returned different coordinates on identical calls
    (RDKit's -1 draws from the global RNG); `n=0` fell through `n or n_confs(...)` and silently meant "auto".
    """
    import rxembed.pipeline as rx

    with pytest.raises(ValueError, match=match):
        rx.embed("CCO", **kwargs)


def test_dump_refuses_an_empty_ensemble(tmp_path):
    """A 0-byte file that reads as a successful write is the worst possible outcome (the `Conformers` rule).

    Reachable from the pipeline: `minimize()` can drop every conformer, and `EnsembleSet.dump` fans out here.
    """
    import rxembed.pipeline as rx

    ens = rx.embed("CCO", n=2)
    ens.ids = []
    with pytest.raises(ValueError, match="nothing to dump"):
        ens.dump(str(tmp_path / "empty.xyz"))


# ---------------------------------------------------------------------------------------------------------
# Seed vs relax: which stage puts the geometry in the window (was test_seed_windows.py)
#
# `rxembed.pipeline.embed()` output must satisfy the constraint windows it was embedded under.
#
# Filed here because it drives the PIPELINE verb, which relaxes its seeds into their windows; the core verb
# does not. It needs no extra, so on a base install it runs rather than skips.
#
# The suite had no assertion of this at all, which is how a defect this size survived 285 green tests: the
# raw ETKDG seed misses its own angle windows by 9.5 deg on average and up to 42.8, and every behavioural
# test downstream calls a stage (`minimize`/`prune`/`score`) that relaxes first, so none of them could see
# it. These tests read `ens.ids` straight off `rx.embed` and nothing else.
#
# ---------------------------------------------------------------------------------------------------------


# Slack, not zero. The relax honours a window to within numerical noise, but `fix={(i, j): d}` writes a
# window narrower than UFF's own equilibrium, so a stiff pull settles a hair outside it. These bars are
# far below the RAW-SEED violations they exist to catch (17-63 deg, 0.55 A); see the module docstring.
_ANGLE_SLACK = 2.0  # deg
_DIST_SLACK = 0.05  # A


def _per_conformer(ens):
    """[(max angle-window violation deg, max distance-window violation A)], one entry per conformer."""
    out = []
    for cid in ens.ids:
        pos = ens.mol.GetConformer(cid).GetPositions()
        ang = dist = 0.0
        for (i, j), (lo, hi) in ens.cons.distances.items():
            d = float(np.linalg.norm(pos[i] - pos[j]))
            dist = max(dist, lo - d, d - hi)
        for (i, j, k), (lo, hi) in ens.cons.angles.items():
            u, v = pos[i] - pos[j], pos[k] - pos[j]
            a = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
            ang = max(ang, lo - float(a), float(a) - hi)
        out.append((max(ang, 0.0), max(dist, 0.0)))
    return out


def _worst(ens):
    """(max angle-window violation deg, max distance-window violation A) over every conformer."""
    per = _per_conformer(ens)
    return max((a for a, _ in per), default=0.0), max((d for _, d in per), default=0.0)


def test_metal_embed_satisfies_its_coordination_windows():
    """A bis-en Co(III) octahedron: `rx.embed` alone, no `.minimize()`.

    The in-repo stand-in for XAWQUH (an external tmQM refcode, and `tests/` deliberately depends on no
    corpus outside the repo). Same phenomenon, larger: this arrangement's raw seed misses a coordination
    angle window by 63 deg, against XAWQUH's 41 -- both go to 0.00 once the relax runs. It is also the
    molecule that exercises the angle intersect branch, so the two nets cover one structure.
    """
    ens = rx.embed(rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")[0], n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    ang, dist = _worst(ens)
    assert ang < _ANGLE_SLACK, f"embed() left a coordination angle {ang:.1f} deg outside its window"
    assert dist < _DIST_SLACK, f"embed() left a distance {dist:.3f} A outside its window"


def test_a_seed_the_relax_tears_keeps_its_seed_geometry_not_a_wrecked_one():
    """`embed` must never hand back a conformer the relax wrecked, and must not silently spend the caller's `n`.

    the N-bound Ni is the case that forces this: the window relax tears 3 of 8 seeds at the base stiffness, one with
    the kappa1 carboxylate wrenched to a 119 deg anti-offset (an sp2 carbon is rigidly 180). `_rescue_torn`
    re-relaxes each at its own minimal sufficient stiffness and falls back to the seed for any that survives no
    rung, so the count is preserved and no conformer is worse than the seed it came from. The residual is
    stated, not hidden: a fallback conformer keeps its seed's window violation, which is why this asserts a
    MAJORITY satisfy the windows rather than all of them.
    """
    ni_n = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(ni_n, "square_planar")[0]
    seeds = _embed_dispatch(iso, n=8, seed=1)
    ens = rx.embed(iso, n=8, seed=1)
    assert len(ens.ids) == len(seeds.ids), "embed dropped conformers: the caller's n must survive the relax"

    # an sp2 carboxyl carbon holds its two substituents rigidly anti; the seed is 180 on every conformer
    o = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "O")
    c = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    subs = [
        nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1
    ]
    for cid in ens.ids:
        conf = ens.mol.GetConformer(cid)
        phis = [rdMolTransforms.GetDihedralDeg(conf, iso.metal, o, c, x) for x in subs]
        anti = abs((phis[0] - phis[1] + 540) % 360 - 180)
        assert anti > 120.0, f"conformer {cid}: the relax wrecked the carboxylate (anti offset {anti:.1f} deg)"

    ok = sum(1 for a, d in _per_conformer(ens) if a < _ANGLE_SLACK and d < _DIST_SLACK)
    seed_ok = sum(1 for a, d in _per_conformer(seeds) if a < _ANGLE_SLACK and d < _DIST_SLACK)
    assert seed_ok == 0, "the raw seed is supposed to satisfy NOTHING here: the premise moved"
    assert ok >= len(ens.ids) - 2, f"only {ok}/{len(ens.ids)} conformers satisfy their windows"


@pytest.mark.parametrize(
    ("smiles", "constrain"),
    [
        ("OC(=O)CCCCc1ccccc1", {(1, 9): (2.6, 3.0)}),  # an arene-acid: the seed misses by 0.55 A
        ("NCCCCCCC(=O)O", {(0, 8): (2.5, 3.0)}),  # the class the finding reproduced at 0.208-0.432 A
    ],
)
def test_organic_constrain_window_is_satisfied_by_embed(smiles, constrain):
    """A soft `constrain=` window on a plain organic: no metal, no frozen core, no `.minimize()`."""
    ens = rx.embed(smiles, constrain=constrain, n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    _ang, dist = _worst(ens)
    assert dist < _DIST_SLACK, f"embed() left the constrain= window violated by {dist:.3f} A"
