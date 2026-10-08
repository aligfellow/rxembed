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
from rxembed import bounds
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL
from rxembed.pipeline import dispatch
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.stereo_check import signature

# --- baseline: clean peripheries ---------------------------------------------


@pytest.mark.parametrize("engine", [rx.core.embed])
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


@pytest.mark.parametrize(("coplanar_14", "metal_floor_relief"), [(False, True)])
def test_matrix_edit_controls_leave_uff_constraints_intact(monkeypatch, coplanar_14, metal_floor_relief):
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


_RELAXATION_CALLS = {
    "embed": lambda value: rx.embed("CCO", n=1, max_iters=value),
    "minimize": lambda value: rx.minimize("CCO", max_iters=value),
}


@pytest.mark.parametrize(("door", "value"), [("minimize", 0)])
def test_relaxation_cap_rejects_non_positive_ints(door, value):
    with pytest.raises(ValueError, match="max_iters must be a positive integer"):
        _RELAXATION_CALLS[door](value)


@pytest.mark.parametrize("inspect_first", [False])
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


# --- constrain: soft windows realised ----------------------------------------


@pytest.mark.parametrize(
    "fix",
    [
        {(1, 2): (2.006, 2.046), (2, 0): (1.537, 1.577)},
    ],
    ids=["window"],
)
def test_numeric_pair_fix_survives_cleanup_without_fixing_angle(fix):
    ens = rx.embed(
        "[O-].ClCCCCBr",
        fix=fix,
        n=25,
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


@pytest.mark.parametrize("target", [(-5.0, 5.0)], ids=["narrow-window"])
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


# --- feasibility ------------------------------------------------------------


# --- minimize records its drops, exactly as prune does -----------------------


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


def test_max_iteration_embed_keeps_valid_seed_marked_unrelaxed(caplog):
    """A real max_iters=1 cap leaves an unconverged but still valid seed, not a rejected one."""
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=2, seed=1, max_iters=1)

    assert ens.n == 2
    assert ens.unrelaxed
    assert all(ens._geometry_failure(cid) is None for cid in ens.unrelaxed)
    assert any("no converged UFF geometry; see .unrelaxed" in record.getMessage() for record in caplog.records)
    ens.minimize()
    assert ens.n == 2


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


# --- seed vs relax: which stage puts the geometry in the window --------------------------------------------
#
# `rxembed.embed()` output must satisfy the constraint windows it was embedded under. Filed here because it
# drives the pipeline verb, which relaxes its seeds into their windows; the core verb does not. These tests
# read `ens.ids` straight off `rx.embed` and nothing else, rather than a later stage that would relax first.
# ---------------------------------------------------------------------------------------------------------


# Slack, not zero. The relax honours a window to within numerical noise, but `fix={(i, j): d}` writes a
# window narrower than UFF's own equilibrium, so a stiff pull settles a hair outside it. These bars are
# far below the raw-seed violations they catch (17-63 deg, 0.55 A); see the module docstring.


_GRAFT_TOL = 0.01  # a fixed core is held exactly: the frozen-core distance assertion the project guarantees
_AMIDE_CORE = [0, 1, 2, 3]  # the conserved C-C(=O)-N motif: the same leading indices in every analogue below


def _embedded(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def test_seed_budget_exhaustion_does_not_claim_geometric_infeasibility(monkeypatch):
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


def test_cxsmiles_contacts_use_restored_metal_graph(monkeypatch):
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
    text = rx.cxsmiles(rx.metal("[NH3]->[Pt](<-[NH3])(Cl)Cl", "square_planar")[0])
    mol = rx.parse_smiles(text)

    from_text = rx.embed(text, metal="square_planar", n=1, seed=1)
    from_mol = rx.embed(mol, metal="square_planar", n=1, seed=1)

    assert isinstance(from_text, rx.Ensemble)
    assert type(from_text) is type(from_mol)


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


def _fail_relax_for_stereo(label):
    """Return a `relax_into_windows` that raises `EmbeddingError` for the stereoisomer tagged `label`."""
    real = rx.Ensemble.relax_into_windows

    def relax(self, **kwargs):
        if self.tag.get("stereo") == label:
            raise rx.EmbeddingError(f"{label}: synthetic relax failure")
        return real(self, **kwargs)

    return relax


def test_separate_stereo_keeps_a_failed_configuration_with_its_errors(monkeypatch):
    monkeypatch.setattr(rx.Ensemble, "relax_into_windows", _fail_relax_for_stereo("C1:R"))
    result = rx.embed("CC(O)CC", n=1, seed=1, stereo="separate")

    assert [[ens.tag["stereo"] for ens in group] for group in result] == [["C1:S"], []]
    assert [[str(err) for err in group.errors] for group in result] == [[], ["C1:R: synthetic relax failure"]]


def test_empty_metal_candidate_is_not_published(monkeypatch):
    monkeypatch.setattr(dispatch, "_execute", lambda *args, **kwargs: dispatch.Ensemble(_embedded("CC"), []))

    with pytest.raises(rx.EmbeddingError, match=r"no conformer satisfied the constraints; try another seed=$"):
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


# --- rx.metal: what spec the isomer enumerator hands down ---------------------------------------------------


# --- rx.minimize: the search-free companion ----------------------------------------------------------------


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


def test_unembeddable_stereoisomer_is_skipped_and_recorded(monkeypatch):
    def execute(spec, **_kwargs):
        if any(bond.GetStereo() == Chem.BondStereo.STEREOE for bond in spec.GetBonds()):
            raise RuntimeError("synthetic E failure")
        mol = _embedded("CC")
        return dispatch.Ensemble(mol, [0])

    monkeypatch.setattr(dispatch, "_execute", execute)
    r = rx.embed("C1CCC=CCCC1", n=1)

    assert [ens.tag.get("stereo") for ens in r] == ["C3=C4:Z"]
    assert [str(err) for err in r.errors] == ["stereoisomer [C3=C4:E]: could not embed molecule: synthetic E failure"]
