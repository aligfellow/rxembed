"""Test the public pipeline verbs end to end."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL
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
    ids=["polar-chain", "separate-fragments"],
)
def test_free_embed_is_clean(smiles):
    import rxembed as rx

    ens = rx.embed(smiles, n=6).minimize()  # the gate is the acceptance test on settled geometries
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# --- constrain: soft windows realised ----------------------------------------


@pytest.mark.parametrize(
    "fix",
    [
        {(1, 2): 2.026, (2, 0): 1.557},
        {(1, 2): (2.006, 2.046), (2, 0): (1.537, 1.577)},
    ],
    ids=["scalar", "window"],
)
def test_numeric_pair_fix_survives_cleanup_without_fixing_angle(fix):
    ens = rx.embed(
        "[O-].ClCCCCBr",
        fix=fix,
        n=6,
        seed=1,
        stereo="free",
    ).minimize()

    assert ens.ids
    for pair, (lo, hi) in ens.cons.fixed.items():
        measured = ens.measure(pair)
        if lo == hi:
            assert measured["min"] == pytest.approx(lo, abs=FIX_DISTANCE_TOL)
            assert measured["max"] == pytest.approx(lo, abs=FIX_DISTANCE_TOL)
        else:
            assert lo <= measured["min"] <= measured["max"] <= hi
    angle = ens.measure((1, 2, 0))
    assert angle["max"] - angle["min"] > 30.0, "pair fixing became a rigid three-atom graft"


@pytest.mark.parametrize(
    "target",
    [60.0, -180.0, (170.0, 190.0), (-190.0, -170.0)],
    ids=["scalar", "half-turn-alias", "periodic-window", "periodic-window-alias"],
)
def test_numeric_dihedral_fix_survives_cleanup(target):
    ens = rx.embed("CCCC", fix={(0, 1, 2, 3): target}, n=4, seed=1, stereo="free").minimize()
    assert ens.ids
    lo, hi = ens.cons.fixed[(0, 1, 2, 3)]
    middle = 0.5 * (lo + hi)
    for cid in ens.ids:
        actual = rdMolTransforms.GetDihedralDeg(ens.mol.GetConformer(cid), 0, 1, 2, 3)
        actual = middle + (actual - middle + 180.0) % 360.0 - 180.0
        if lo == hi:
            assert actual == pytest.approx(lo, abs=FIX_ANGLE_TOL)
        else:
            assert lo <= actual <= hi
    measured = ens.measure((0, 1, 2, 3))
    assert lo - FIX_ANGLE_TOL <= measured["min"] <= measured["max"] <= hi + FIX_ANGLE_TOL


@pytest.mark.parametrize("target", [0.0, 60.0, (-5.0, 5.0)], ids=["antipodal", "twisted", "narrow-window"])
def test_numeric_dihedral_overrides_internal_torsion_repair(target):
    atoms = (0, 1, 3, 4)
    ens = rx.embed("CC(=O)NC", fix={atoms: target}, n=2, seed=2, stereo="free").minimize()
    assert ens.ids
    lo, hi = ens.cons.fixed[atoms]
    for cid in ens.ids:
        actual = rdMolTransforms.GetDihedralDeg(ens.mol.GetConformer(cid), *atoms)
        if lo == hi:
            assert actual == pytest.approx(lo, abs=FIX_ANGLE_TOL)
        else:
            assert lo <= actual <= hi


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


# --- feasibility ------------------------------------------------------------


def test_infeasible_fix_raises():
    import rxembed as rx

    with pytest.raises((RuntimeError, ValueError)):
        rx.embed("CCO", fix={(0, 2): 0.15}, n=4)  # a physically impossible C..O distance


@pytest.mark.parametrize(
    "kw", [{"freeze": [0, 1, 2]}, {"n_confs": 2}, {"num_confs": 2}], ids=["freeze", "n-confs", "num-confs"]
)
def test_removed_embed_keywords_are_rejected(kw):
    import rxembed as rx

    with pytest.raises(TypeError, match=next(iter(kw))):
        rx.embed("CCO", **kw)


@pytest.mark.parametrize("door", ["embed", "metal"])
@pytest.mark.parametrize("stereo", ["auto", "enumerate"])
def test_removed_stereo_modes_are_rejected(door, stereo):
    import rxembed as rx

    with pytest.raises(ValueError, match="unknown stereo mode"):
        getattr(rx, door)("CCO", stereo=stereo)


@pytest.mark.parametrize("stereo", [{"point": "bogus"}, {"bogus": "free"}], ids=["point-mode", "stereo-kind"])
def test_unknown_stereo_specs_are_rejected(stereo):
    import rxembed as rx

    with pytest.raises(ValueError, match="unknown stereo mode"):
        rx.embed("CCO", stereo=stereo)


# --- minimize records its drops, exactly as prune does -----------------------


def test_minimize_records_its_energy_window_drops(monkeypatch):
    import numpy as np

    import rxembed as rx
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


def test_untypable_minimize_keeps_embedded_geometry(monkeypatch, caplog):
    import rxembed as rx
    from rxembed.pipeline import calculators as _refine

    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert ens.ids, "embed produced no conformers"
    assert ens._seeds_relaxed, "embed did not mark the seeds relaxed -> minimize won't take the single-point branch"
    assert not ens._minimized, "minimize must still run"
    kept = list(ens.ids)

    def raiser(*a, **kw):  # UFFGetMoleculeForceField raises at build time on an untypable graph
        raise RuntimeError("Pre-condition Violation\nbad params pointer\nRDKIT: 2026.03.3\nBOOST: 1_85")

    monkeypatch.setattr(_refine, "restrained_uff", raiser)  # pipeline.minimize resolves the binding at call time
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.minimize()  # single-point branch -> must not propagate the RuntimeError

    warnings = [r.message for r in caplog.records if "UFF could not relax" in r.message]
    assert warnings, "the guard never fired (null test)"
    assert "bad params pointer" in warnings[0]
    assert "\n" not in warnings[0], "a multiline RDKit exception leaked into one log record"
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


def test_minimize_composes_isomer_template_and_fix():
    seed = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=1, seed=1)
    iso = rx.metal(seed.mol, "square_planar")[0]
    sphere = {iso.metal, *iso.donors}
    core = [a.GetIdx() for a in iso.mol.GetAtoms() if a.GetIdx() not in sphere][:3]

    out = rx.minimize(iso, template=(iso.mol, {i: i for i in core[:2]}), fix=core[2:])

    assert out.n == 1
    assert sorted(out.cons.frozen) == core
    assert out.sphere == {iso.metal: iso.donors}, "the Isomer's coordination context was lost"


def test_pipeline_enumerate_isomer_names_conflict():
    smiles = "CCCN[Pd](Cl)Cl"
    assert len(rx.metal(smiles, "square_planar")) > 0, "the pipeline verb must read this string"
    with pytest.raises(TypeError, match="takes an RDKit Mol"):
        rx.enumerate_isomers(smiles, "square_planar")


def test_pipeline_embeds_stated_cxsmiles(monkeypatch):
    import rxembed.pipeline.ensemble as ensemble_module

    isomers = rx.metal("Cl[Co]12(Cl)(N[C@@H](C)CN1)NCCN2", "octahedral", stereo="free")
    selected = next(i for i in isomers if i.chirality == "delta")
    assert selected is not isomers[0], "the fixture must detect a reader that silently takes the first isomer"
    text = rx.cxsmiles(selected)

    reference = rx.embed(text, n=1, seed=2)
    sphere = {reference.iso.metal, *reference.iso.donors}
    centre = next(
        a
        for a in reference.mol.GetAtoms()
        if a.GetIdx() not in sphere
        and len([n for n in a.GetNeighbors() if n.GetIdx() not in sphere and n.GetAtomicNum() > 1]) >= 2
    )
    core = [centre.GetIdx(), *[n.GetIdx() for n in centre.GetNeighbors() if n.GetIdx() not in sphere][:2]]
    ref_pos = reference.mol.GetConformer(reference.ids[0]).GetPositions()
    fix = {i: tuple(ref_pos[i]) for i in core}

    ens = rx.embed(text, fix=fix, n=3, seed=1)

    assert ens.iso is not None
    assert rx.cxsmiles(ens.iso) == text
    assert [rx.cxsmiles(ens[k].mol) for k in range(len(ens))] == [text] * len(ens)
    target = ens.n
    initial_ids = set(ens.ids)
    seed_calls = 0

    seed_conformers = ensemble_module.seed_conformers
    geometry_check = ensemble_module._geometry.check
    checked_frozen = []
    checked_spheres = []

    def tracked_seed(*args, **kwargs):
        nonlocal seed_calls
        seed_calls += 1
        return seed_conformers(*args, **kwargs)

    def tracked_check(*args, **kwargs):
        checked_frozen.append(frozenset(kwargs.get("frozen", ())))
        return geometry_check(*args, **kwargs)

    connectivity_scan = ensemble_module.Ensemble._scan_connectivity

    def tracked_scan(self, *args, **kwargs):
        checked_spheres.append(dict(self.sphere))
        return connectivity_scan(self, *args, **kwargs)

    drop = ensemble_module.Ensemble._drop_bad_geometries
    forced = False

    def reject_one(self, iso):
        nonlocal forced
        drops = drop(self, iso)
        if not forced and self.ids:
            self.ids.pop()
            drops["forced rejection"] = 1
            forced = True
        return drops

    from rxembed.pipeline import calculators as _refine

    restrained_uff = _refine.restrained_uff
    failed_single_point = False

    def fail_first_single_point(*args, **kwargs):
        nonlocal failed_single_point
        if kwargs.get("max_iters") == 0 and not failed_single_point:
            failed_single_point = True
            raise RuntimeError("forced single-point failure")
        return restrained_uff(*args, **kwargs)

    monkeypatch.setattr(ensemble_module, "seed_conformers", tracked_seed)
    monkeypatch.setattr(ensemble_module._geometry, "check", tracked_check)
    monkeypatch.setattr(ensemble_module.Ensemble, "_scan_connectivity", tracked_scan)
    monkeypatch.setattr(ensemble_module.Ensemble, "_drop_bad_geometries", reject_one)
    monkeypatch.setattr(_refine, "restrained_uff", fail_first_single_point)
    ens.minimize()

    assert forced, "the retry path was not exercised"
    assert failed_single_point, "the initial single-point failure was not exercised"
    assert seed_calls, "the retry did not call the shared fresh-seed embed seam"
    assert checked_frozen
    assert set(checked_frozen) == {frozenset(ens.cons.frozen)}
    assert checked_spheres
    assert all(ens.sphere == sphere for sphere in checked_spheres)
    assert ens.n == target, "the retry path did not restore the starting count"
    assert set(ens.ids) - initial_ids, "the rejected conformer was re-used instead of freshly embedded"
    assert ens.wrong_hand == []
    assert ens.energy_kind == "ff"
    assert set(ens.energies) == set(ens.ids)
    assert {c.GetId() for c in ens.mol.GetConformers()} == set(ens.ids)
    assert {c.GetId() for c in ens._mol.GetConformers()} == set(ens.ids) | set(ens.discarded)
    assert [rx.cxsmiles(ens[i].mol) for i in range(ens.n)] == [text] * ens.n
    pairs = ((core[0], core[1]), (core[0], core[2]), (core[1], core[2]))
    for cid in ens.ids:
        pos = ens.mol.GetConformer(cid).GetPositions()
        drift = max(abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j])) for i, j in pairs)
        assert drift < 0.01, "the fresh retry lost the coordinate-fixed core"


def test_pipeline_ignores_unrelated_cxsmiles_atom_notes():
    ens = rx.embed("CCO |atomProp:0.atomNote.foo|", n=1)

    assert ens.iso is None
    assert ens.ids


def test_failed_embed_relax_is_not_marked_settled(monkeypatch):
    import importlib

    core_embed = importlib.import_module("rxembed.embed")

    seed = {}

    def fail(mol, *args, **kwargs):
        seed.update({c.GetId(): c.GetPositions().copy() for c in mol.GetConformers()})
        for conf in mol.GetConformers():
            conf.SetAtomPosition(0, (99.0, 99.0, 99.0))
        raise RuntimeError("unsupported atom type")

    monkeypatch.setattr(core_embed, "restrained_uff", fail)
    ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=2, seed=1)

    assert not ens._seeds_relaxed
    assert ens.unrelaxed == ens.ids
    for cid in ens.ids:
        assert np.array_equal(ens.mol.GetConformer(cid).GetPositions(), seed[cid])


def test_embed_records_the_real_restrained_uff_cleanup():
    smiles = "C[P]1(C)CC[P](C)(C)->[Ni+2]<-12<-[O-]C(=O)C[N-]->2C"
    iso = rx.metal(smiles, "SPL")[0]
    ens = rx.embed(iso, n=1, seed=19, trajectory=True)
    trail = ens.trajectory

    assert trail.GetNumConformers() > 2
    assert trail.GetAtomWithIdx(iso.metal).GetSymbol() == "Ni"
    assert not np.allclose(trail.GetConformer(0).GetPositions(), trail.GetConformer(1).GetPositions())
    assert np.allclose(
        trail.GetConformer(trail.GetNumConformers() - 1).GetPositions(), ens.mol.GetConformer().GetPositions()
    )
    assert all(report.ok() for report in ens.check().values())

    ens.minimize()
    assert ens.trajectory.GetNumConformers() == trail.GetNumConformers()
    assert ens[:0].trajectory is None


def test_trajectory_is_an_explicit_single_path_request():
    spec = {"constrain": {(0, 4): (2.5, 3.0)}}
    with pytest.raises(TypeError, match="True or False"):
        rx.embed("OCCCO", n=1, trajectory=10, **spec)
    with pytest.raises(ValueError, match="requires n=1"):
        rx.embed("OCCCO", n=2, trajectory=True, **spec)


# ---------------------------------------------------------------------------------------------------------
# Seed vs relax: which stage puts the geometry in the window (was test_seed_windows.py)
#
# `rxembed.embed()` output must satisfy the constraint windows it was embedded under.
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
# far below the raw-seed violations they catch (17-63 deg, 0.55 A); see the module docstring.
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


def test_torn_relax_restores_seed_geometry():
    ni_n = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(ni_n, "square_planar")[0]
    seeds = _embed_dispatch(iso, n=4, seed=1)
    ens = rx.embed(iso, n=4, seed=1)
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
    # At most one irreparable conformer may keep the seed's residual rather than a torn relaxed geometry.
    assert ok >= len(ens.ids) - 1, f"only {ok}/{len(ens.ids)} conformers satisfy their windows"


def test_organic_embed_satisfies_its_constrain_window():
    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=4, seed=1)
    assert ens.ids
    _ang, dist = _worst(ens)
    assert dist < _DIST_SLACK, f"embed() left constrain= violated by {dist:.3f} A"
