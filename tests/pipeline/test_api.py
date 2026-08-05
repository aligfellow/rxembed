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
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.dispatch import _embed_dispatch

_SLACK_A = 0.15  # the pipeline's own _validate distance slack: a window realised within this is "held"


# --- baseline: clean peripheries ---------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "OC(=O)CCCCc1ccccc1",  # close polar contacts that must not read as clashes
        "CCO.c1ccccc1",  # two fragments must not embed on top of each other
    ],
)
def test_free_embed_is_clean(smiles):
    import rxembed.pipeline as rx

    ens = rx.embed(smiles, n=6).minimize()  # the gate is the acceptance test on settled geometries
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


# --- feasibility ------------------------------------------------------------


def test_infeasible_fix_raises():
    import rxembed.pipeline as rx

    with pytest.raises((RuntimeError, ValueError)):
        rx.embed("CCO", fix={(0, 2): 0.15}, n=4)  # a physically impossible C..O distance


@pytest.mark.parametrize("kw", [{"freeze": [0, 1, 2]}, {"n_confs": 2}, {"num_confs": 2}])
def test_removed_embed_keywords_are_rejected(kw):
    import rxembed.pipeline as rx

    with pytest.raises(TypeError, match=next(iter(kw))):
        rx.embed("CCO", **kw)


@pytest.mark.parametrize("door", ["embed", "metal"])
@pytest.mark.parametrize("stereo", ["auto", "enumerate"])
def test_removed_stereo_modes_are_rejected(door, stereo):
    import rxembed.pipeline as rx

    with pytest.raises(ValueError, match="unknown stereo mode"):
        getattr(rx, door)("CCO", stereo=stereo)


@pytest.mark.parametrize("stereo", [{"point": "bogus"}, {"bogus": "free"}])
def test_unknown_per_kind_stereo_specs_are_rejected(stereo):
    import rxembed.pipeline as rx

    with pytest.raises(ValueError, match="unknown stereo mode"):
        rx.embed("CCO", stereo=stereo)


# --- minimize records its drops, exactly as prune does -----------------------


def test_minimize_records_its_energy_window_drops(monkeypatch):
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


def test_pipeline_minimize_takes_template_like_embed():
    ref = rx.embed("CCO", n=1, seed=1)
    core = [0, 1, 2]  # C, C, O: three atoms, so the graft restores a shape rather than sliding a bond
    want = ref.mol.GetConformer(ref.ids[0]).GetPositions()

    moved = Chem.Mol(ref.mol)  # same graph, the core pulled apart so a graft has something to undo
    conf = moved.GetConformer()
    for a in core:
        p = conf.GetAtomPosition(a)
        conf.SetAtomPosition(a, [p.x + 0.6 * a, p.y - 0.4 * a, p.z])
    torn = moved.GetConformer().GetPositions()
    off = abs(np.linalg.norm(torn[0] - torn[2]) - np.linalg.norm(want[0] - want[2]))
    assert off > 0.5, f"the core was not distorted, so a graft would be invisible (d off by {off:.2f} A)"

    out = rx.minimize(moved, template=(ref, {i: i for i in core}))

    got = out.mol.GetConformer(out.ids[0]).GetPositions()
    for i, j in ((0, 1), (1, 2), (0, 2)):
        d_ref = float(np.linalg.norm(want[i] - want[j]))
        d_out = float(np.linalg.norm(got[i] - got[j]))
        assert abs(d_out - d_ref) < 0.01, f"template= did not graft d({i},{j}): {d_out:.3f} vs {d_ref:.3f} A"


def test_pipeline_minimize_keeps_an_isomers_coordination_while_composing_a_template_and_list_fix():
    seed = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=1, seed=1)
    iso = rx.metal(seed.mol, "square_planar")[0]
    sphere = {iso.metal, *iso.donors}
    core = [a.GetIdx() for a in iso.mol.GetAtoms() if a.GetIdx() not in sphere][:3]

    out = rx.minimize(iso, template=(iso.mol, {i: i for i in core[:2]}), fix=core[2:])

    assert out.n == 1
    assert sorted(out.cons.frozen) == core
    assert out.sphere == {iso.metal: iso.donors}, "the Isomer's coordination context was lost"


def test_the_two_enumerate_isomers_on_the_pipeline_namespace_disagree_loudly():
    smiles = "CCCN[Pd](Cl)Cl"
    assert len(rx.metal(smiles, "square_planar")) > 0, "the pipeline verb must read this string"
    with pytest.raises(TypeError, match="takes an RDKit Mol"):
        rx.enumerate_isomers(smiles, "square_planar")


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
    ens = rx.embed(rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")[0], n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    ang, dist = _worst(ens)
    assert ang < _ANGLE_SLACK, f"embed() left a coordination angle {ang:.1f} deg outside its window"
    assert dist < _DIST_SLACK, f"embed() left a distance {dist:.3f} A outside its window"


def test_a_seed_the_relax_tears_keeps_its_seed_geometry_not_a_wrecked_one():
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
    # At most two irreparable conformers may keep the seed's residual rather than a torn relaxed geometry.
    assert ok >= len(ens.ids) - 2, f"only {ok}/{len(ens.ids)} conformers satisfy their windows"


def test_organic_embed_satisfies_its_constrain_window():
    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=4, seed=1)
    assert ens.ids
    _ang, dist = _worst(ens)
    assert dist < _DIST_SLACK, f"embed() left constrain= violated by {dist:.3f} A"
