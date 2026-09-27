"""Test the public pipeline verbs end to end, source normalization and embedding dispatch."""

from __future__ import annotations

import importlib
import itertools
import logging
from collections import Counter
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolTransforms

import rxembed as rx
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.stereo_check import signature
from rxembed.relax import UFFTypingError
from tests.conftest import EXAMPLES_DIR

_SLACK_A = 0.15  # the pipeline's own _validate distance slack: a window realised within this is "held"


# --- baseline: clean peripheries ---------------------------------------------


@pytest.mark.parametrize("engine", [rx.embed, rx.core.embed])
def test_params_round_trips_through_both_facades(engine):
    ethanol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    result = engine(ethanol, n=2, seed=5)
    assert result.params == rx.EmbedParams(seed=5)

    again = engine(ethanol, n=2, params=result.params)
    for cid in result.ids:
        np.testing.assert_array_equal(
            result.mol.GetConformer(cid).GetPositions(), again.mol.GetConformer(cid).GetPositions()
        )

    by_seed = engine(ethanol, n=2, seed=7)
    by_params = engine(ethanol, n=2, params=rx.EmbedParams(seed=7))
    for cid in by_seed.ids:
        np.testing.assert_array_equal(
            by_seed.mol.GetConformer(cid).GetPositions(), by_params.mol.GetConformer(cid).GetPositions()
        )

    assert result[:1].params == result.params


@pytest.mark.parametrize(("coplanar_14", "metal_floor_relief"), [(True, False), (False, True)])
def test_matrix_edit_controls_leave_uff_constraints_intact(monkeypatch, coplanar_14, metal_floor_relief):
    from rxembed import bounds

    isomer = next(
        iso for iso in rx.metal("[Pt+2](<-[Cl-])(<-[Cl-])(<-n1ccccc1)<-n1ccccc1", "SPL") if iso.label == "cis"
    )
    native = rdDistGeom.KDG()
    params = rx.EmbedParams(
        seed=42, threads=1, native=native, coplanar_14=coplanar_14, metal_floor_relief=metal_floor_relief
    )
    core = importlib.import_module("rxembed.embed")
    native_bounds, native_uff = bounds._feasible_bounds, core.restrained_uff
    seen_dg, seen_uff = [], []

    def matrix(mol, cons, used):
        assert used is native
        seen_dg.append((bool(cons.coplanar), bool(cons.dg_floors)))
        return native_bounds(mol, cons, used)

    def cleanup(mol, cons, **kwargs):
        seen_uff.append((bool(cons.coplanar), bool(cons.dg_floors), bool(cons.floors)))
        return native_uff(mol, cons, **kwargs)

    monkeypatch.setattr(bounds, "_feasible_bounds", matrix)
    monkeypatch.setattr(core, "restrained_uff", cleanup)
    result = rx.embed(isomer, n=1, params=params)
    result.minimize()
    assert seen_dg
    assert all(flags == (coplanar_14, metal_floor_relief) for flags in seen_dg)
    assert seen_uff
    assert all(flags == (True, True, True) for flags in seen_uff)
    assert result.cons.coplanar
    assert result.cons.dg_floors
    assert (result.params.coplanar_14, result.params.metal_floor_relief) == (coplanar_14, metal_floor_relief)
    assert (result[0].params.coplanar_14, result[0].params.metal_floor_relief) == (coplanar_14, metal_floor_relief)
    geom.check(result.mol, donors=isomer.donors).assert_ok()


@pytest.mark.parametrize("engine", [rx.embed, rx.core.embed])
def test_custom_cleanup_controls_are_public_ablations(engine):
    isomer = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", "SPL")[0]
    native = engine(isomer, n=1, params=rx.EmbedParams(seed=42, donor_orientation=False, conjugation=False))

    assert not native.cons.donor_orientation
    assert not native.cons.conjugation
    # D-M-D shell angles remain: only rxembed's donor-axis additions are disabled.
    assert len(native.cons.angles) < len(isomer.cons.angles)


_RELAXATION_CALLS = {
    "embed": lambda value: rx.embed("CCO", n=1, max_iters=value),
    "minimize": lambda value: rx.minimize("CCO", max_iters=value),
}


@pytest.mark.parametrize(("door", "value"), [("embed", 0), ("embed", True), ("embed", "2000"), ("minimize", 0)])
def test_relaxation_cap_rejects_non_positive_ints(door, value):
    with pytest.raises(ValueError, match="max_iters must be a positive integer"):
        _RELAXATION_CALLS[door](value)


def test_free_periphery_clash_remains_a_diagnostic():
    ens = rx.embed("C", n=1, seed=1)
    hydrogens = [atom.GetIdx() for atom in ens._mol.GetAtoms() if atom.GetAtomicNum() == 1]
    conf = ens._mol.GetConformer(ens.ids[0])
    conf.SetAtomPosition(hydrogens[1], conf.GetAtomPosition(hydrogens[0]))

    assert ens._workflow_failure(ens, ens.ids[0]) is None
    assert any(violation.detail == "H...H clash" for violation in ens.check()[ens.ids[0]].violations)


def test_repeated_haptic_alkene_cleanup_keeps_geminal_hydrogens_separate():
    smiles = "FC(F)(F)C1=CC(=[O]->[Rh+]23(<-[CH2]=[CH2]->2)(<-[CH2]=[CH2]->3)<-[O-]1)C(F)(F)F"
    isomer = rx.metal(smiles, "square_planar")[0]
    for _ in range(2):
        ensemble = rx.embed(isomer, n=1, seed=9, threads=1)

        assert ensemble.n == 1
        assert not ensemble.unrelaxed
        assert not any(v.detail == "H...H clash" for v in ensemble.check()[ensemble.ids[0]].violations)


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "aio"])
def test_chiral_diene_keeps_its_face_through_native_refinement(monkeypatch, legacy):
    from rxembed import bounds

    default_parameters = bounds.embed_parameters

    def parameters(*args, **kwargs):
        params = default_parameters(*args, **kwargs)
        params.useLegacyImplementation = legacy
        return params

    monkeypatch.setattr(bounds, "embed_parameters", parameters)
    smiles = "CCO[C@H](c1ccccc1C)[CH]1=[CH]2[CH]3=[CH2]->[Fe]<-3<-2<-1(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"
    isomer = rx.metal(smiles, "tetrahedral")[0]
    expected = rx.cxsmiles(isomer)

    ensemble = rx.embed(isomer, n=1, seed=42, threads=1)

    assert ensemble.n == 1
    assert not ensemble.unrelaxed
    assert rx.cxsmiles(ensemble.mol) == expected
    geom.check(ensemble.mol).assert_ok()


@pytest.mark.parametrize("inspect_first", [False, True])
def test_macrocyclic_donor_stereo_embeds_without_read_order_dependencies(inspect_first):
    # This tetradentate macrocycle bites all four consecutive donor pairs at once, so both diagonal rows widen
    # to the free consequence of the relaxed shell (metal_polyhedron.relaxed_shell); the bounded bite box
    # (metal_constraints.bounded_bites) keeps that widening inside the requested tetrahedron rather than
    # opening a seesaw-like basin next to it, so the embed itself -- not only the compiled contract -- must
    # survive an early read.
    smiles = "C1C[N@@H]2->[Cu+]34<-[S](CC2)CC/[C-]->3=[NH+]/CC[S]->4C1"
    isomer = rx.metal(smiles, "tetrahedral")[0]
    if inspect_first:
        assert isomer.cons.distances
        rx.cxsmiles(isomer)

    ensemble = rx.embed(isomer, n=1, seed=42, threads=1)

    assert ensemble.n == 1
    assert not ensemble.unrelaxed
    assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(isomer)


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


@pytest.mark.parametrize(("door", "stereo"), [("embed", "auto"), ("metal", "enumerate")])
def test_unknown_stereo_mode_is_rejected_by_both_doors(door, stereo):
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

    core_embed = importlib.import_module("rxembed.embed")

    ens = rx.embed("CCCCO", n=8)  # unconstrained -> not yet minimized; `n` is a request, so read the real ids
    assert len(ens.ids) >= 2, "need >=2 conformers so dropping one still leaves a non-empty ensemble"
    victim = ens.ids[0]

    real = core_embed.restrained_uff

    def spiked(mol, cons, **kwargs):
        e = np.asarray(real(mol, cons, **kwargs), dtype=float)  # relax for real; fake only one score
        for k, conf in enumerate(mol.GetConformers()):  # e is in conformer-enumeration order, mapped by GetId()
            if conf.GetId() == victim:
                e[k] = 1e4  # a non-physical energy for the victim alone -> its ΔE >> the 250 kcal/mol window
        return e

    monkeypatch.setattr(core_embed, "restrained_uff", spiked)
    ens.minimize()

    assert victim not in ens.ids, "the energy window never fired: the test would be a null measurement"
    assert victim in ens.discarded, "minimize dropped the conformer but did not record it in `discarded`"


# --- force-field capability and optimizer failures stay distinct -------------


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
    assert {metal: set(donors) for metal, donors in out.sphere.items()} == {iso.metal: set(iso.donors)}
    assert rx.cxsmiles(out.iso) == rx.cxsmiles(iso), "the Isomer's coordination arrangement was lost"
    assert rx.cxsmiles(out.mol) == rx.cxsmiles(iso)


def test_metal_geometry_without_the_stereo_extra_says_preservation_is_off(monkeypatch, caplog):
    geometry = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=1, seed=1).mol

    def missing(*_args, **_kwargs):
        raise ImportError("signature needs xyzgraph; pip install 'rxembed[workflow]'")

    monkeypatch.setattr(importlib.import_module("rxembed.pipeline.dispatch"), "signature", missing)
    with caplog.at_level("WARNING", logger="rxembed"):
        isomers = rx.metal(geometry, "square_planar")

    assert all(iso.stereo_ref is None for iso in isomers)
    assert "stereo preservation unavailable" in caplog.text
    assert "rxembed[workflow]" in caplog.text


def test_explicit_invert_on_a_helical_isomer_is_not_replaced_by_preserve(monkeypatch):
    dispatch = importlib.import_module("rxembed.pipeline.dispatch")
    ensemble = importlib.import_module("rxembed.pipeline.ensemble")
    geometry = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=1, seed=1).mol
    monkeypatch.setattr(dispatch, "signature", lambda *_args, **_kwargs: {"helical": Counter({"M": 1})})
    iso = rx.metal(geometry, "square_planar")[0]
    monkeypatch.setattr(ensemble, "signature", lambda *_args, **_kwargs: {"helical": Counter({"P": 1})})

    ens = rx.embed(iso, n=1, seed=1, stereo="invert")

    assert ens.n == 1
    assert ens.stereo_filter[0] == "invert"


def test_stereoisomer_cap_names_a_remedy_the_caller_has(monkeypatch, caplog):
    monkeypatch.setattr(importlib.import_module("rxembed.pipeline.dispatch"), "_STEREO_CAP", 1)
    with caplog.at_level("WARNING", logger="rxembed"):
        rx.embed("CC(O)C(C)O", n=1, seed=1)

    assert "stereo='free'" in caplog.text


def test_metal_reads_an_ionic_xyz_at_its_total_charge(tmp_path):
    salt = "[Cl-]->[Pt+2](<-[Cl-])(<-N)<-N.C[N+](C)(C)C"  # cisplatin beside a tetramethylammonium cation
    path = tmp_path / "cisplatin_nme4.xyz"
    Chem.MolToXYZFile(rx.embed(rx.metal(salt, "square_planar")[0], n=1, seed=1).mol, str(path))

    assert rx.metal(str(path), "square_planar", charge=1)


def test_auto_contacts_without_the_extra_name_the_extra(monkeypatch):
    def missing(*_args, **_kwargs):
        raise ImportError("analyzer needs xyzgraph; pip install 'rxembed[workflow]'")

    monkeypatch.setattr(importlib.import_module("rxembed.pipeline.dispatch"), "auto_binding_modes", missing)
    with pytest.raises(ImportError, match=r"rxembed\[workflow\]"):
        rx.embed("CC(=O)O.CC(=O)O", contacts="auto", n=1, seed=1)


def test_pipeline_enumerate_isomer_names_conflict():
    smiles = "CCCN[Pd](Cl)Cl"
    assert len(rx.metal(smiles, "square_planar")) > 0, "the pipeline verb must read this string"
    with pytest.raises(TypeError, match="takes an RDKit Mol"):
        rx.enumerate_isomers(smiles, "square_planar")


def test_pipeline_embeds_stated_cxsmiles(monkeypatch):
    core_embed = importlib.import_module("rxembed.embed")
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
    seed_calls = []

    seed_conformers = core_embed.seed_conformers
    checked_spheres = []

    def tracked_seed(*args, **kwargs):
        params = kwargs.get("params", args[4] if len(args) > 4 else None)
        seed_calls.append(params.seed)
        return seed_conformers(*args, **kwargs)

    connectivity_scan = ensemble_module.Ensemble._scan_connectivity

    def tracked_scan(self, *args, **kwargs):
        checked_spheres.append(dict(self.sphere))
        return connectivity_scan(self, *args, **kwargs)

    workflow_failure = ensemble_module.Ensemble._workflow_failure
    forced = False

    def reject_one(self, owner, cid):
        nonlocal forced
        reason = workflow_failure(self, owner, cid)
        if not forced and owner is self:
            forced = True
            return core_embed.Failure("physical_geometry", "forced rejection")
        return reason

    monkeypatch.setattr(core_embed, "seed_conformers", tracked_seed)
    monkeypatch.setattr(ensemble_module.Ensemble, "_scan_connectivity", tracked_scan)
    monkeypatch.setattr(ensemble_module.Ensemble, "_workflow_failure", reject_one)
    ens._stage = "seeded"
    ens.minimize()

    assert forced, "the retry path was not exercised"
    assert seed_calls, "the retry did not call the shared fresh-seed embed seam"
    assert seed_calls[0] == ens.params.seed + 1
    assert checked_spheres
    assert all(ens.sphere == sphere for sphere in checked_spheres)
    assert ens.n == target, "the retry path did not restore the starting count"
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
    core_embed = importlib.import_module("rxembed.embed")

    seed = {}

    def fail(mol, *args, **kwargs):
        if not seed:
            seed.update({c.GetId(): c.GetPositions().copy() for c in mol.GetConformers()})
        ids = kwargs.get("conf_ids") or [c.GetId() for c in mol.GetConformers()]
        for cid in ids:
            mol.GetConformer(int(cid)).SetAtomPosition(0, (99.0, 99.0, 99.0))
        raise UFFTypingError("unsupported atom type")

    monkeypatch.setattr(core_embed, "restrained_uff", fail)
    ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=2, seed=1)

    assert ens._stage == "seeded"
    assert ens.unrelaxed == ens.ids
    for cid in ens.ids:
        assert np.array_equal(ens.mol.GetConformer(cid).GetPositions(), seed[cid])


def test_max_iteration_embed_keeps_valid_seed_marked_unrelaxed(monkeypatch, caplog):
    core_embed = importlib.import_module("rxembed.embed")
    restrained_uff = core_embed.restrained_uff
    calls = []

    def fail_first_conformer(mol, *args, **kwargs):
        calls.append(mol)
        energies = restrained_uff(mol, *args, **kwargs)
        record = kwargs.get("record")
        if mol is calls[0] and record is not None and 0 in record.statuses:
            record.statuses[0] = 1
        return energies

    monkeypatch.setattr(core_embed, "restrained_uff", fail_first_conformer)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=2, seed=1)

    assert ens.n == 2
    assert ens.unrelaxed == [ens.ids[0]]
    assert ens._geometry_failure(ens.unrelaxed[0]) is None
    assert [record.getMessage() for record in caplog.records] == [
        "embed: 1/2 conformer(s) have no converged UFF geometry; see .unrelaxed"
    ]
    ens.minimize()
    assert ens.n == 2
    assert ens.unrelaxed == [ens.ids[0]]


def test_constrained_embed_runs_no_single_point(monkeypatch):
    """embed() publishes geometry, not an energy: its relax-into-windows pass must take no trailing UFF call."""
    core_embed = importlib.import_module("rxembed.embed")
    calls = []
    real_uff = core_embed.restrained_uff

    def spy_uff(mol, cons, *, max_iters, conf_ids=None, **kw):
        calls.append(max_iters)
        return real_uff(mol, cons, max_iters=max_iters, conf_ids=conf_ids, **kw)

    monkeypatch.setattr(core_embed, "restrained_uff", spy_uff)
    ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=2, seed=1)

    assert ens.ids
    assert 0 not in calls


@pytest.mark.parametrize("n", [1, 2], ids=("empty", "partial"))
def test_embed_reports_rejection_and_keeps_partial_results(monkeypatch, caplog, n):
    core_embed = importlib.import_module("rxembed.embed")

    def stall(mol, _cons, **kwargs):
        ids = kwargs.get("conf_ids") or [conf.GetId() for conf in mol.GetConformers()]
        record = kwargs.get("record")
        if record is not None:
            record.statuses.update(dict.fromkeys(ids, 1))
        return [0.0] * len(ids)

    monkeypatch.setattr(core_embed, "restrained_uff", stall)
    monkeypatch.setattr(
        core_embed.Conformers,
        "_geometry_failure",
        lambda _self, cid, _iso=None: (
            core_embed.Failure("structural_constraint", "missed structural constraint") if cid == 0 else None
        ),
    )
    monkeypatch.setattr(core_embed.Conformers, "_replace_failed", lambda _self, failed, *_args, **_kw: list(failed))

    with caplog.at_level("WARNING", logger="rxembed"):
        if n == 1:
            with pytest.raises(rx.EmbeddingError, match="missed structural constraint in 1/1 rejected seeds"):
                rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=n, seed=1)
        else:
            ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=n, seed=1)
            assert ens.n == n - 1
            assert "embed: kept 1/2 conformers; missed structural constraint" in caplog.text


def test_embed_never_returns_an_unresolved_metal_state(monkeypatch):
    core_embed = importlib.import_module("rxembed.embed")
    iso = next(i for i in rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral") if i.chirality)
    monkeypatch.setattr(core_embed, "_mirror_is_free", lambda _mol: False)
    monkeypatch.setattr(
        core_embed.Conformers,
        "_metal_states",
        lambda self: dict.fromkeys(self.ids, False),
    )
    monkeypatch.setattr(core_embed.Conformers, "_replace_failed", lambda _self, failed, *_args, **_kw: list(failed))

    with pytest.raises(rx.EmbeddingError, match="loses its requested hand"):
        rx.embed(iso, n=1, seed=1)


def test_haptic_winding_rank_is_unchanged_when_the_real_metal_is_restored(monkeypatch):
    macpfe = "COC(=O)[CH]1=[CH2]->[Fe]<-1(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[P](c1ccccc1)(c1ccccc1)c1ccccc1"
    iso = next(
        candidate
        for candidate in rx.metal(macpfe, "TBP")
        if set(candidate.haptic_winding.values()) == {"-"}
        and candidate.mol.GetAtomWithIdx(candidate.vertices[1]).GetSymbol() == "P"
    )
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", lambda self, **_kw: self)  # keep the raw DG seed
    ensemble = rx.embed(iso, n=1, seed=42)

    assert ensemble._metal_states() == {ensemble.ids[0]: True}


def test_selected_metal_identity_survives_workflow_finalize():
    iso = next(i for i in rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral") if i.chirality)
    ens = rx.embed(iso, n=1, seed=1).minimize()

    assert ens.iso is iso
    assert all(ens._metal_states().values())


def test_embed_records_the_real_restrained_uff_cleanup(tmp_path):
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

    before = [conf.GetPositions() for conf in trail.GetConformers()]
    path = ens.dump_trajectory(tmp_path / "cleanup.xyz")
    assert path.read_text() == "".join(Chem.MolToXYZBlock(trail, confId=conf.GetId()) for conf in trail.GetConformers())
    for conf, positions in zip(trail.GetConformers(), before, strict=True):
        np.testing.assert_array_equal(conf.GetPositions(), positions)

    ens.minimize()
    assert ens.trajectory.GetNumConformers() == trail.GetNumConformers()
    assert ens[:0].trajectory is None


def test_trajectory_is_an_explicit_single_path_request():
    spec = {"constrain": {(0, 4): (2.5, 3.0)}}
    with pytest.raises(TypeError, match="True or False"):
        rx.embed("OCCCO", n=1, trajectory=10, **spec)
    with pytest.raises(ValueError, match="requires n=1"):
        rx.embed("OCCCO", n=2, trajectory=True, **spec)


# --- seed vs relax: which stage puts the geometry in the window --------------------------------------------
#
# `rxembed.embed()` output must satisfy the constraint windows it was embedded under. Filed here because it
# drives the pipeline verb, which relaxes its seeds into their windows; the core verb does not. These tests
# read `ens.ids` straight off `rx.embed` and nothing else, rather than a later stage that would relax first.
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
    isos = rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")
    ens = rx.embed(next(iso for iso in isos if iso.chirality == "delta"), n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    ang, dist = _worst(ens)
    assert ang < _ANGLE_SLACK, f"embed() left a coordination angle {ang:.1f} deg outside its window"
    assert dist < _DIST_SLACK, f"embed() left a distance {dist:.3f} A outside its window"


def test_torn_relax_restores_seed_geometry(monkeypatch):
    ni_n = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(ni_n, "square_planar")[0]
    with monkeypatch.context() as raw:
        raw.setattr(rx.Ensemble, "relax_into_windows", lambda self, **_kw: self)
        seeds = rx.embed(iso, n=4, seed=1)
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
    assert seed_ok == 0, "the raw seed is supposed to satisfy none of its windows here: the premise moved"
    # At most one irreparable conformer may keep the seed's residual rather than a torn relaxed geometry.
    assert ok >= len(ens.ids) - 1, f"only {ok}/{len(ens.ids)} conformers satisfy their windows"


def test_organic_embed_satisfies_its_constrain_window():
    ens = rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=4, seed=1)
    assert ens.ids
    _ang, dist = _worst(ens)
    assert dist < _DIST_SLACK, f"embed() left constrain= violated by {dist:.3f} A"


_GRAFT_TOL = 0.01  # a fixed core is held exactly: the frozen-core distance assertion the project guarantees
_AMIDE_CORE = [0, 1, 2, 3]  # the conserved C-C(=O)-N motif: the same leading indices in every analogue below
_SN2 = str(EXAMPLES_DIR / "sn2.xyz")
_SN2_CORE = [4, 0, 5]  # F...C...Cl reacting core, from the templated-TS notebook


def _embedded(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


@pytest.mark.parametrize(
    "smiles",
    [
        r"C1C[N@@H]2->[Cu+]34<-[C-](=[NH+]\CC[S]->3C1)/CC[S]->4CC2",
        "C1C[N@@H]2->[Cu+]34<-[S](CC2)CC/[C-]->3=[NH+]/CC[S]->4C1",
    ],
    ids=["carbon-first", "sulfur-first"],
)
def test_reading_an_isomer_does_not_change_its_dg_seed(monkeypatch, smiles):
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", lambda self, **_kw: self)  # keep the raw DG seed
    positions = []
    for inspect_first in (False, True):
        isomer = rx.metal(smiles, "tetrahedral")[0]
        if inspect_first:
            assert isomer.cons.distances
            rx.cxsmiles(isomer)
        ensemble = rx.embed(isomer, n=1, seed=42, threads=1)
        positions.append(ensemble.mol.GetConformer().GetPositions())

    # Reading is invariant within one traversal; RDKit does not promise the same random draw after renumbering.
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-10, rtol=0)


@pytest.mark.parametrize(("engine", "source_kind"), [("pipeline", "mol"), ("pipeline", "isomer"), ("core", "isomer")])
@pytest.mark.parametrize("n", [1, 2])
def test_coordinate_sources_always_generate_new_conformers(monkeypatch, source_kind, n, engine):
    from rxembed import core

    seed_iso = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Br-]", "SPL")[0]
    source = rx.embed(seed_iso, n=1, seed=1).mol
    positions = source.GetConformer().GetPositions() + 100.0
    source.GetConformer().SetPositions(positions)
    spec = source if source_kind == "mol" else rx.metal(source, observed_only=True)[0]
    calls, native = [], rdDistGeom.EmbedMultipleConfs

    def generate(*args):
        calls.append(args[1])
        return native(*args)

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", generate)
    ensemble = (core.embed if engine == "core" else rx.embed)(spec, n=n, seed=42, threads=1)

    assert calls, "an input conformer must not bypass RDKit"
    assert len(ensemble.ids) == n
    for cid in ensemble.ids:
        assert not np.allclose(ensemble.mol.GetConformer(cid).GetPositions(), positions)
    np.testing.assert_array_equal(source.GetConformer().GetPositions(), positions)


@pytest.mark.parametrize(("engine", "source_kind"), [("pipeline", "mol"), ("pipeline", "isomer"), ("core", "isomer")])
def test_failed_native_embedding_does_not_return_source_coordinates(monkeypatch, source_kind, engine):
    from rxembed import core

    iso = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Br-]", "SPL")[0]
    source = rx.embed(iso, n=1, seed=1).mol
    positions = source.GetConformer().GetPositions()
    spec = source if source_kind == "mol" else rx.metal(source, observed_only=True)[0]
    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", lambda *_args: [])

    if engine == "core":
        result = core.embed(spec, n=1, seed=42, threads=1)
        assert not result.ids
        assert not result.mol.GetNumConformers()
    else:
        with pytest.raises(ValueError, match="no conformer"):
            rx.embed(spec, n=1, seed=42, threads=1)
    np.testing.assert_array_equal(source.GetConformer().GetPositions(), positions)


def test_seed_budget_exhaustion_does_not_claim_geometric_infeasibility(monkeypatch):
    from rxembed.pipeline import dispatch

    def no_seeds(mol, _cons, _iso, n, _params, **_kwargs):
        return mol, [], n

    iso = rx.metal("N->[Pt+2](<-[Cl-])(<-[Br-])<-P", "SPL")[0]
    monkeypatch.setattr(dispatch, "seed_conformers", no_seeds)
    with pytest.raises(rx.EmbeddingError, match="found 0/1 DG seeds") as caught:
        rx.embed(iso, n=1, seed=42)
    assert str(caught.value).count(str(iso)) == 1, "the isomer is named once"
    assert "infeasible" not in str(caught.value)


def _max_core_drift(mol, ids, core, ref_pos):
    """Largest deviation of any core pair's distance from the reference, over all conformers (frame-free)."""
    return max(
        abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j]))
        for pos in (mol.GetConformer(c).GetPositions() for c in ids)
        for i, j in itertools.combinations(core, 2)
    )


# --- fix: the rigid graft, in each of the three forms the resolver accepts ---------------------------------


def test_fix_grafts_indices_and_coordinate_dict():
    mol = _embedded("CC(=O)Nc1ccccc1", seed=1)
    own = mol.GetConformer().GetPositions()
    ens = rx.embed(mol, fix=_AMIDE_CORE, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, own) < _GRAFT_TOL

    ref = _embedded("CC(=O)Nc1ccccc1", seed=7).GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccccc1", fix={i: tuple(ref[i]) for i in _AMIDE_CORE}, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL


def test_numeric_fix_reaches_explicit_hydrogen():
    mol = Chem.AddHs(Chem.MolFromSmiles("CN"))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    h = next(a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors() if a.GetAtomicNum() == 1)
    ens = rx.embed(mol, fix={(n, h): 1.20}, n=6).minimize()  # held past its ~1.01 A equilibrium
    assert ens.n >= 1
    assert ens.measure((n, h))["mean"] == pytest.approx(1.20, abs=0.1)


def test_unqualified_coordinate_free_metal_fails_loudly():
    with pytest.raises(ValueError, match="plain RDKit embedding does not model metals"):
        rx.embed("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", n=1)


def test_unknown_coordinate_string_names_the_accepted_forms():
    pocket = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "SPL").select(index=0)
    with pytest.raises(ValueError, match="SMARTS pattern, an atom index, or a list"):
        rx.embed(pocket, coordinate="auto", n=1)


def test_geometry_metal_constraint_uses_shared_preparation():
    selected = rx.metal("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "square_planar")[0]
    source = rx.embed(selected, n=1, seed=1).mol
    metal = next(atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() == 46)
    donors = [atom.GetIdx() for atom in source.GetAtomWithIdx(metal).GetNeighbors()]
    carbons = [atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() == 6]
    pair = (carbons[0], carbons[-1])
    pos = source.GetConformer().GetPositions()
    target = float(np.linalg.norm(pos[pair[0]] - pos[pair[1]]))

    ens = rx.embed(source, constrain={pair: (target - 0.2, target + 0.2)}, n=1, seed=2)

    assert ens.iso is not None
    assert set(ens.iso.donors) == set(donors)
    for donor in donors:
        key = (min(metal, donor), max(metal, donor))
        assert ens.cons.distances[key] == selected.cons.distances[key]
    restored = ens.mol
    assert restored.GetAtomWithIdx(metal).GetAtomicNum() == 46
    assert all(restored.GetBondBetweenAtoms(donor, metal).GetBondType() == Chem.BondType.DATIVE for donor in donors)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_xyz_ts_core_holds_and_passes_gate(monkeypatch):
    from rxembed.pipeline.perceive import read_xyz

    reference = read_xyz(_SN2, 0)
    ref = reference.GetConformer().GetPositions()
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", lambda self, **_kw: self)  # keep the raw DG seed
    seeds = rx.embed(_SN2, fix=_SN2_CORE, n=1)
    assert _max_core_drift(seeds.mol, seeds.ids, _SN2_CORE, ref) < _GRAFT_TOL
    for cid in seeds.ids:
        geom.check(seeds.mol, cid, frozen=_SN2_CORE, reference=reference).assert_ok()


def test_rigid_core_and_a_soft_window_compose():
    mol = _embedded("OC(=O)CCCCc1ccccc1", seed=3)
    core = [0, 1, 2]  # the carboxyl O, C, =O
    ref = mol.GetConformer().GetPositions()
    soft, lo, hi = (1, 9), 3.5, 4.2  # carbonyl C to a ring carbon; free d ~ 7.5 A, so the window must pull

    ens = rx.embed(mol, fix=core, constrain={soft: (lo, hi)}, n=10).minimize()
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, core, ref) < _GRAFT_TOL
    stats = ens.measure(soft)
    assert stats["min"] >= lo - 0.15
    assert stats["max"] <= hi + 0.15
    assert rx.embed(mol, n=8).minimize().measure(soft)["mean"] > hi + 1.0, "the window did not bite vs a free embed"
    assert any(geom.check(ens.mol, cid, frozen=core).ok() for cid in ens.ids)


# --- template: the same graft, expressed as a reference plus a map -----------------------------------------


def test_reference_core_transfers_onto_a_different_backbone():
    ref_mol = _embedded("CC(=O)Nc1ccccc1", seed=5)
    ref = ref_mol.GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccc(C(C)(C)C)cc1", template=(ref_mol, "CC(=O)N"), n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()  # ...and the backbone around it is clean


def test_smarts_template_needs_two_molecular_graphs():
    with pytest.raises(ValueError, match="explicit index map"):
        rx.embed("CCO", template=(np.zeros((3, 3)), "CCO"), n=1)


def test_template_composes_with_source_geometry_fix():
    ref = _embedded("CC(=O)Nc1ccccc1", seed=5)
    mol = Chem.Mol(ref)
    conf = mol.GetConformer()
    for atom in (2, 3):
        p = conf.GetAtomPosition(atom)
        conf.SetAtomPosition(atom, (p.x + 0.4, p.y - 0.3, p.z + 0.2))
    own = mol.GetConformer().GetPositions()
    mixed = ref.GetConformer().GetPositions().copy()
    mixed[[2, 3]] = own[[2, 3]]

    ens = rx.embed(mol, template=(ref, {0: 0, 1: 1}), fix=[2, 3], n=2, seed=1)

    assert sorted(ens.cons.frozen) == [0, 1, 2, 3], "the pipeline dropped the list fix beside template="
    assert _max_core_drift(ens.mol, ens.ids, [0, 1, 2, 3], mixed) < _GRAFT_TOL, (
        "the list fix did not take atoms 2/3 from the source's own geometry"
    )


def test_empty_ensemble_cannot_supply_template():
    ref = rx.embed("CCO", n=1, seed=1)
    ref.ids.clear()

    with pytest.raises(ValueError, match="tracked conformer"):
        rx.embed("CCO", template=(ref, {0: 0}), n=1, seed=1)


def test_ensemble_template_keeps_smarts_ambiguity_guard():
    ref = rx.embed("Cc1ccccc1", n=1, seed=1)

    with pytest.raises(ValueError, match="symmetry-equivalent"):
        rx.embed("CCc1ccccc1", template=(ref, "c1ccccc1"), n=1, seed=1)


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_auto_contacts_preserve_template_core():
    smi = "CC(=O)O.n1ccccc1"
    ref = rx.embed(smi, n=2)
    ref_pos = ref.mol.GetConformer(ref.ids[0]).GetPositions()
    core = [0, 1, 2, 3]  # the acetic-acid heavy core

    out = rx.embed(smi, contacts="auto", n=4, template=(ref.mol, {i: i for i in core}))
    for ens in out if isinstance(out, rx.EnsembleSet) else [out]:
        assert sorted(ens.cons.frozen) == core, "the template never reached the auto-contacts route"
        assert _max_core_drift(ens.mol, ens.ids[:1], core, ref_pos) < 1e-6


def test_raw_contacts_use_the_shared_constraint_validator():
    with pytest.raises(ValueError, match="out of range"):
        rx.embed("CCCC", contacts={(0, 99): (2.0, 3.0)}, n=1)


def test_contacts_and_constrain_give_model_compilation_the_same_ownership():
    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    window = {(0, 2): (1.0, 5.0)}
    constrained = rx.embed(iso, constrain=window, n=1, seed=42, threads=1)
    contacted = rx.embed(iso, contacts=window, n=1, seed=42, threads=1)
    assert contacted.cons.angles == constrained.cons.angles != iso.cons.angles
    assert contacted.cons.contacts == constrained.cons.contacts
    assert contacted.cons.distances == constrained.cons.distances


def test_discarded_contacts_do_not_change_independent_bite_targets():
    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    ignored = {tuple(sorted((iso.metal, iso.donors[0]))): (1.0, 5.0)}
    contacted = rx.embed(iso, contacts=ignored, n=1, seed=42, threads=1)
    assert not any(contacted.cons.contacts)
    assert contacted.cons.angles == iso.cons.angles


def test_plane_contact_ownership_reaches_the_model_before_seeding(monkeypatch):
    from rxembed.pipeline import dispatch

    class ReachedSeedError(Exception):
        pass

    captured = []

    def capture(_mol, cons, *_args, **_kwargs):
        captured.append(cons)
        raise ReachedSeedError

    monkeypatch.setattr(dispatch, "seed_conformers", capture)
    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    rings = [tuple(ring) for ring in Chem.GetSymmSSSR(iso.mol)]
    for keyword in ("contacts", "constrain"):
        with pytest.raises(ReachedSeedError):
            rx.embed(iso, n=1, seed=42, **{keyword: {(rings[0], rings[1]): 3.7}})
    assert captured[0].planes == captured[1].planes
    assert captured[0].angles == captured[1].angles != iso.cons.angles


def test_cxsmiles_contacts_use_restored_metal_graph(monkeypatch):
    from rxembed.pipeline import dispatch

    text = rx.cxsmiles(rx.metal("[NH3]->[Pt](<-[NH3])(Cl)Cl.O", "square_planar")[0])
    seen = {}

    def capture(mol, seed):
        seen["mol"] = Chem.Mol(mol)
        return {}

    monkeypatch.setattr(dispatch, "auto_binding_modes", capture)

    rx.embed(text, contacts="auto", n=1)

    discovered = seen["mol"]
    assert any(a.GetAtomicNum() == 78 for a in discovered.GetAtoms()), "contact discovery saw the carbon surrogate"
    assert len(Chem.GetMolFrags(discovered)) == 2, "the coordinated ligands were presented as separate fragments"


def test_stated_metal_return_shape_does_not_depend_on_source_representation():
    from rxembed.pipeline import dispatch

    text = rx.cxsmiles(rx.metal("[NH3]->[Pt](<-[NH3])(Cl)Cl", "square_planar")[0])
    mol = dispatch._normalize(text)[0]

    from_text = rx.embed(text, metal="square_planar", n=1, seed=1)
    from_mol = rx.embed(mol, metal="square_planar", n=1, seed=1)

    assert isinstance(from_text, rx.Ensemble)
    assert type(from_text) is type(from_mol)


def test_metal_candidate_failure_is_not_hidden_by_siblings(monkeypatch):
    from rxembed.pipeline import dispatch

    calls = 0

    def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second candidate failed")
        mol = _embedded("CC")
        return dispatch.Ensemble(mol, [0])

    monkeypatch.setattr(dispatch, "_execute", execute)

    with pytest.raises(ValueError, match="second candidate failed"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1)
    assert calls == 2


def _fail_relax_for(label):
    """Return a `relax_into_windows` that raises `EmbeddingError` for the isomer with `label` (all when None)."""
    real = rx.Ensemble.relax_into_windows

    def relax(self, **kwargs):
        if label is None or self.iso.label == label:
            raise rx.EmbeddingError(f"{self.iso}: synthetic relax failure", isomer=self.iso)
        return real(self, **kwargs)

    return relax


def test_multi_isomer_embed_returns_the_isomers_that_embed(monkeypatch, caplog):
    """An isomer that cannot be built is one WARNING line and one `errors` entry; the others come back."""
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", _fail_relax_for("cis"))
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1, seed=1)

    assert [ens.tag["label"] for ens in result] == ["trans"]
    assert [err.isomer.label for err in result.errors] == ["cis"]
    assert [record.getMessage() for record in caplog.records] == [str(result.errors[0])]
    assert result.minimize().errors == result.errors, "a mapped verb keeps the record"


def test_multi_isomer_embed_raises_only_when_no_isomer_embeds(monkeypatch, caplog):
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", _fail_relax_for(None))
    with pytest.raises(rx.EmbeddingError, match="none of 2 candidates could be built; first: Pd"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1, seed=1)

    caplog.clear()
    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    with caplog.at_level(logging.WARNING, logger="rxembed"), pytest.raises(rx.EmbeddingError) as caught:
        rx.embed(iso, n=1, seed=1)
    assert caught.value.isomer is iso
    assert not caplog.records, "a lone isomer raises its own error instead of warning first"


def test_multi_isomer_embed_skips_an_isomer_that_fails_at_seeding(monkeypatch, caplog):
    from rxembed.pipeline import dispatch

    real = dispatch.require_seed_count

    def underfill_cis(ids, target, iso=None):
        if iso is not None and iso.label == "cis":
            raise rx.EmbeddingError(f"{iso}: synthetic seed shortfall", isomer=iso)
        return real(ids, target, iso)

    monkeypatch.setattr(dispatch, "require_seed_count", underfill_cis)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        result = rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1, seed=1)

    assert [ens.tag["label"] for ens in result] == ["trans"]
    assert [err.isomer.label for err in result.errors] == ["cis"]
    assert [record.getMessage() for record in caplog.records] == [str(result.errors[0])]


def _fail_relax_for_stereo(label):
    """Return a `relax_into_windows` that raises `EmbeddingError` for the stereoisomer tagged `label`."""
    real = rx.Ensemble.relax_into_windows

    def relax(self, **kwargs):
        if self.tag.get("stereo") == label:
            raise rx.EmbeddingError(f"{label}: synthetic relax failure")
        return real(self, **kwargs)

    return relax


def test_racemate_with_one_failed_enantiomer_keeps_the_failure_in_errors(monkeypatch):
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", _fail_relax_for_stereo("C1:R"))
    result = rx.embed("CC(O)CC", n=1, seed=1)

    assert isinstance(result, rx.EnsembleSet), "a failure must not collapse the survivor to a bare Ensemble"
    assert [ens.tag["stereo"] for ens in result] == ["C1:S"]
    assert [str(err) for err in result.errors] == ["C1:R: synthetic relax failure"]


def test_separate_stereo_keeps_a_failed_configuration_with_its_errors(monkeypatch):
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", _fail_relax_for_stereo("C1:R"))
    result = rx.embed("CC(O)CC", n=1, seed=1, stereo="separate")

    assert [[ens.tag["stereo"] for ens in group] for group in result] == [["C1:S"], []]
    assert [[str(err) for err in group.errors] for group in result] == [[], ["C1:R: synthetic relax failure"]]


def test_empty_metal_candidate_is_not_published(monkeypatch):
    from rxembed.pipeline import dispatch

    monkeypatch.setattr(dispatch, "_execute", lambda *args, **kwargs: dispatch.Ensemble(_embedded("CC"), []))

    with pytest.raises(rx.EmbeddingError, match=r"no conformer satisfied the constraints; try another seed=$"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1)


def test_empty_identity_expansion_names_the_candidate_axis(monkeypatch):
    from rxembed.pipeline import dispatch

    monkeypatch.setattr(dispatch, "enumerate_isomers", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="no feasible coordination identity"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_geometry_source_is_normalized_once_for_stereo(tmp_path):
    """An xyz input's helical twist keeps its hand through a constrained re-embed.

    A [5]helicene's P/M twist is not a bonds-matrix property (unlike a tetrahedral centre), so a plain
    re-embed can flip it; only reading the input's own signature back (dispatch._stereo_filter) catches that.
    The check only runs while relaxing into a window, so the re-embed needs one trivial fix= to engage it.
    """
    helicene = Chem.AddHs(Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12"))
    assert rdDistGeom.EmbedMolecule(helicene, randomSeed=1, useRandomCoords=True) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(helicene, maxIters=5000)
    hand = signature(helicene, charge=0)["helical"]

    conf = helicene.GetConformer()
    fix = {(0, 1): round(conf.GetAtomPosition(0).Distance(conf.GetAtomPosition(1)), 3)}
    xyz = tmp_path / "helicene.xyz"
    xyz.write_text(Chem.MolToXYZBlock(helicene))

    ens = rx.embed(str(xyz), n=1, seed=3, fix=fix)
    assert signature(ens.mol, conf_id=ens.ids[0], charge=0)["helical"] == hand


# --- contacts: a discovered binding mode is one the embed can actually realise ------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_auto_contacts_form_hydrogen_bonds():
    es = rx.embed("OC(=O)c1ccccc1.n1ccccc1", contacts="auto", n=6)  # acid + pyridine
    for ens in es if isinstance(es, rx.EnsembleSet) else [es]:
        assert ens.n >= 1
        grip = ens.cons.contacts[0]
        assert grip, "a discovered binding mode must seed a releasable contact"
        settled = ens.minimize()
        for cid in settled.ids:
            geom.check(settled.mol, cid, constraints=settled.cons).assert_ok()
        for pair in grip:
            lo, hi = settled.cons.distances[pair]
            measured = settled.measure(pair)
            positions = [(measured[key] - lo) / (hi - lo) for key in ("min", "max")]
            assert positions[0] >= 0.0, f"the seeded grip fell below its window at position {positions[0]:.3f}"
            assert positions[1] < 0.8, f"the seeded grip rode its upper wall at position {positions[1]:.3f}"


def test_stereo_and_contact_candidates_compose_before_embedding(monkeypatch):
    from rxembed.pipeline import dispatch

    modes = {"near": dispatch.Contact(), "far": dispatch.Contact()}
    monkeypatch.setattr(dispatch, "auto_binding_modes", lambda _mol, seed: modes)

    result = rx.embed("CC(N)O.N", contacts="auto", n=1, seed=1)

    assert isinstance(result, rx.EnsembleSet)
    assert {(ens.tag["stereo"], ens.tag["nci"]) for ens in result} == {
        (hand, mode) for hand in ("C1:R", "C1:S") for mode in modes
    }


# --- rx.metal: what spec the isomer enumerator hands down ---------------------------------------------------


def test_numeric_metal_fix_needs_no_input_geometry():
    iso = rx.metal("P->[Pd](Cl)Cl", "square_planar", fix={(0, 1): 2.1})[0]
    assert iso.cons.fixed[(0, 1)] == (2.1, 2.1)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_spectator_sphere_uses_polyhedron_constraints_without_implicit_shape_holds():
    source = rx.read_xyz(str(EXAMPLES_DIR / "mn-h2.xyz"), metal_charges={0: 2, 1: 1})
    iso = rx.metal(source, "octahedral", center="Mn", fix=[1, 5, 63, 64, 65, 66])[0]
    spectators = {m for m in iso.cons.metals if m != iso.metal}
    assert spectators, "mn-h2 is bimetallic: the ferrocene Fe must be surrogated as a spectator"
    assert not iso.cons.shapes
    states = {state.atom: state for state in iso.centres}
    assert spectators <= states.keys()
    assert all(any(getattr(site, "winding", "") for site in states[metal].vertices) for metal in spectators)
    assert all(any(metal in pair for pair in iso.cons.distances) for metal in spectators)
    for donor in iso.donors:
        pair = (min(iso.metal, donor), max(iso.metal, donor))
        assert (pair in iso.cons.pulls) != ({iso.metal, donor} <= iso.cons.frozen)

    from_smiles = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]  # no input geometry -> nothing shape-held
    assert not from_smiles.cons.shapes
    assert len(from_smiles.cons.pulls) == len(from_smiles.donors)


# --- rx.minimize: the search-free companion ----------------------------------------------------------------


def test_minimize_relaxes_existing_geometry():
    mol = _embedded("CCCCCCC", seed=1)  # heptane; pull the two ends together
    mean = rx.minimize(mol, constrain={(0, 6): (3.0, 3.4)}).measure((0, 6))["mean"]
    assert 2.85 <= mean <= 3.55


def test_minimize_accepts_xyz_and_rejects_smiles(tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(Chem.MolToXYZBlock(_embedded("CCCCCCC", seed=1)))
    assert rx.minimize(str(xyz), fix={(0, 6): 3.0}).measure((0, 6))["mean"] == pytest.approx(3.0, abs=0.15)
    with pytest.raises(ValueError, match="existing geometry"):
        rx.minimize("CCO", fix={(0, 2): 2.0})


# --- the `stereo=` route: an undefined centre is a set of distinct species ----------------------------------
#
# The dispatch drives the core `stereo.enumerate_unassigned` and folds the variants into one EnsembleSet.
# Its graph-level contract lives in tests/test_stereo.py; these are the pipeline checks.


def _configs(es):
    return sorted(e.tag.get("stereo") for e in es)


def test_hands_return_as_tagged_ensemble_set():
    es = rx.embed("CC(N)C(=O)O", n=3)  # undefined alpha-carbon
    assert isinstance(es, rx.EnsembleSet)
    assert _configs(es) == ["C1:R", "C1:S"]  # atom-qualified, index-keyed CIP tags
    for e in es:
        em = e.minimize()  # the raw ETKDG seed can carry a conjugation twist; minimise, then read the hand
        Chem.AssignStereochemistryFrom3D(em.mol, confId=em.ids[0])
        ((idx, code),) = Chem.FindMolChiralCenters(em.mol, useLegacyImplementation=False)
        symbol = em.mol.GetAtomWithIdx(idx).GetSymbol()
        assert e.tag["stereo"] == f"{symbol}{idx}:{code}"


@pytest.mark.parametrize(
    ("smi", "kw"),
    [
        pytest.param("C[C@H](N)C(=O)O", {}, id="defined-centre-kept"),
        pytest.param("CCO", {}, id="no-stereocentre"),
        pytest.param("CC(N)C(=O)O", {"stereo": "free"}, id="stereo-free-opts-out"),
    ],
)
def test_no_stereo_returns_ensemble(smi, kw):
    assert isinstance(rx.embed(smi, n=2, **kw), rx.Ensemble)


@pytest.mark.parametrize(
    ("smi", "n_candidates", "why"),
    [
        pytest.param(
            "CC=CC(N)O", 4, "one undefined C x one undefined C=C: the E/Z axis reaches the route too", id="alkene"
        ),
        pytest.param("CC(O)C(O)C", 3, "two centres, but the meso pair collapses; 3 candidates, not 4", id="meso"),
    ],
)
def test_stereo_expansion_returns_candidate_count(smi, n_candidates, why):
    es = rx.embed(smi, n=2)
    assert isinstance(es, rx.EnsembleSet)
    assert len(es) == n_candidates, why
    if n_candidates == 4:
        assert any(":E" in c for c in _configs(es))
        assert any(":Z" in c for c in _configs(es))


def test_stereo_axis_composes_with_metal_axis():
    # an aminoacidate on Pd: 2 ligand enantiomers x the square-planar coordination isomers
    r = rx.embed("CC(N)C(=O)[O-]->[Pd]([Cl])[Cl]", metal="square_planar", n=2)
    assert isinstance(r, rx.EnsembleSet)
    assert {"C1:R", "C1:S"} == {e.tag["stereo"] for e in r}
    assert {"cis", "trans"} <= {e.tag["label"] for e in r}
    for e in r:  # every candidate carries both axes
        assert e.tag.get("stereo")
        assert e.tag.get("label")


def test_allene_axis_stays_a_bare_chainable_ensemble():
    # RDKit can't enumerate allene/cumulene axial chirality from a flat SMILES -> one arbitrary hand, not an
    # EnsembleSet-of-1 (that would break the documented rx.embed(smi).mc().prune() chain), and no '?' tag.
    r = rx.embed("CC(F)=C=C(F)C", n=2)
    assert isinstance(r, rx.Ensemble)
    assert hasattr(r, "mc")
    assert "?" not in (r.tag.get("stereo") or "")


def test_unembeddable_stereoisomer_is_skipped_and_recorded(monkeypatch):
    from rxembed.pipeline import dispatch

    def execute(spec, **_kwargs):
        if any(bond.GetStereo() == Chem.BondStereo.STEREOE for bond in spec.GetBonds()):
            raise RuntimeError("synthetic E failure")
        mol = _embedded("CC")
        return dispatch.Ensemble(mol, [0])

    monkeypatch.setattr(dispatch, "_execute", execute)
    r = rx.embed("C1CCC=CCCC1", n=1)

    assert [ens.tag.get("stereo") for ens in r] == ["C3=C4:Z"]
    assert [str(err) for err in r.errors] == ["stereoisomer [C3=C4:E]: could not embed molecule: synthetic E failure"]
