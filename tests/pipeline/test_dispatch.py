"""Test source normalization and embedding dispatch."""

import itertools
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed.pipeline import geom_check as geom

_GRAFT_TOL = 0.01  # a fixed core is held exactly: the frozen-core distance assertion the project guarantees
_AMIDE_CORE = [0, 1, 2, 3]  # the conserved C-C(=O)-N motif: the same leading indices in every analogue below
_SN2 = "examples/structures/sn2.xyz"
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
def test_reading_an_isomer_does_not_change_its_dg_seed(smiles):
    from rxembed.pipeline.dispatch import _embed_dispatch

    positions = []
    for inspect_first in (False, True):
        isomer = rx.metal(smiles, "tetrahedral")[0]
        if inspect_first:
            assert isomer.cons.distances
            rx.cxsmiles(isomer)
        ensemble = _embed_dispatch(isomer, n=1, seed=42, threads=1)
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

    def no_seeds(mol, _cons, _iso, n, **_kwargs):
        return mol, [], n

    iso = rx.metal("N->[Pt+2](<-[Cl-])(<-[Br-])<-P", "SPL")[0]
    monkeypatch.setattr(dispatch, "seed_conformers", no_seeds)
    with pytest.raises(ValueError, match=r"found 0/1 seeds.*bounded DG seed budget") as caught:
        rx.embed(iso, n=1, seed=42)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) in str(caught.value)
    assert "infeasible" not in str(caught.value)


def _max_core_drift(mol, ids, core, ref_pos):
    """Largest deviation of any core pair's distance from the reference, over all conformers (frame-free)."""
    return max(
        abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j]))
        for pos in (mol.GetConformer(c).GetPositions() for c in ids)
        for i, j in itertools.combinations(core, 2)
    )


def test_threads_reach_both_seed_dispatches(monkeypatch):
    from rxembed.pipeline import dispatch

    seen = []
    real = dispatch.seed_conformers

    def capture(mol, _cons, _iso, _n, **kwargs):
        seen.append(kwargs["threads"])
        return real(mol, _cons, _iso, _n, **kwargs)

    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    monkeypatch.setattr(dispatch, "seed_conformers", capture)
    monkeypatch.setattr(dispatch._nci, "auto_binding_modes", lambda _mol, seed: {})

    rx.embed(_embedded("CCO"), fix=[0, 1], n=1)
    rx.embed(_embedded("CCO"), fix=[0, 1], n=1, threads=3)
    rx.embed("CCO", contacts="auto", n=1, threads=3)
    rx.embed(iso, n=1, threads=3)

    assert seen == [0, 3, 3, 3]


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


def test_coordinate_free_hydride_uses_the_ml_target_through_a_metal_state():
    from rxembed import metal_distance as distance
    from rxembed.metal_constraints import _ML_SEED_HALF_WIDTH

    params = Chem.SmilesParserParams()
    params.removeHs = False
    source = Chem.MolFromSmiles("[H][Ru](Cl)(Cl)Cl", params)
    mol = Chem.AddHs(source)
    metal, hydride = 1, 0
    donors = {n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()}
    target = distance.ml_distance(
        mol,
        metal,
        hydride,
        44,
        donors,
        {},
        hyb={},
    )
    iso = rx.metal(source, "tetrahedral")[0]
    lo, hi = iso.cons.distances[(hydride, metal)]

    assert (lo + hi) / 2 == pytest.approx(target)
    assert hi - lo == pytest.approx(2 * _ML_SEED_HALF_WIDTH)


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
def test_xyz_ts_core_holds_and_passes_gate():
    from rxembed.pipeline.dispatch import _embed_dispatch
    from rxembed.pipeline.perceive import read_xyz

    reference = read_xyz(_SN2, 0)
    ref = reference.GetConformer().GetPositions()
    seeds = _embed_dispatch(_SN2, fix=_SN2_CORE, n=1)
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

    monkeypatch.setattr(dispatch._nci, "auto_binding_modes", capture)

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


def test_empty_metal_candidate_is_not_published(monkeypatch):
    from rxembed.pipeline import dispatch

    monkeypatch.setattr(dispatch, "_execute", lambda *args, **kwargs: dispatch.Ensemble(_embedded("CC"), []))

    with pytest.raises(ValueError, match="no conformer satisfied"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1)


def test_empty_identity_expansion_names_the_candidate_axis(monkeypatch):
    from rxembed.pipeline import dispatch

    monkeypatch.setattr(dispatch, "enumerate_isomers", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="no feasible coordination identity"):
        rx.embed("Cl[Pd](Cl)(N)N", metal="square_planar", n=1)


def test_geometry_source_is_normalized_once_for_stereo(monkeypatch):
    from rxembed.pipeline import dispatch

    normalized = _embedded("CC")
    normalized_calls = []
    signature_calls = []
    monkeypatch.setattr(
        dispatch, "_normalize", lambda source, charge=0: (normalized_calls.append(source) or normalized, True)
    )
    monkeypatch.setattr(
        dispatch._stereo,
        "signature",
        lambda mol, charge=0: signature_calls.append(mol) or {},
    )
    monkeypatch.setattr(
        dispatch,
        "_execute",
        lambda *args, **kwargs: dispatch.Ensemble(normalized, [normalized.GetConformer().GetId()]),
    )

    dispatch._embed_dispatch("input.xyz", n=1, stereo="all")

    assert normalized_calls == ["input.xyz"]
    assert signature_calls == [normalized]


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

    modes = {"near": dispatch._nci.Contact(), "far": dispatch._nci.Contact()}
    monkeypatch.setattr(dispatch._nci, "auto_binding_modes", lambda _mol, seed: modes)

    result = rx.embed("CC(N)O.N", contacts="auto", n=1, seed=1)

    assert isinstance(result, rx.EnsembleSet)
    assert {(ens.tag["stereo"], ens.tag["nci"]) for ens in result} == {
        (hand, mode) for hand in ("C1:R", "C1:S") for mode in modes
    }


# --- the organic path pays nothing for the metal path ------------------------------------------------------


# --- rx.metal: what spec the isomer enumerator hands down ---------------------------------------------------


def test_numeric_metal_fix_needs_no_input_geometry():
    iso = rx.metal("P->[Pd](Cl)Cl", "square_planar", fix={(0, 1): 2.1})[0]
    assert iso.cons.fixed[(0, 1)] == (2.1, 2.1)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_spectator_sphere_uses_polyhedron_constraints_without_implicit_shape_holds():
    source = rx.read_xyz("examples/structures/mn-h2.xyz", metal_charges={0: 2, 1: 1})
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


# The subject below is the core's `bounds._bounds`; it is pinned here because the spec that reaches it is
# the dispatch's, and this file is where that spec is otherwise exercised.
def test_reversed_angle_preserves_distance_window():
    from rxembed import bounds
    from rxembed.constraints import Constraints, add_distance

    mol = _embedded("CCCC", seed=1)

    def window(angle_key):
        c = Constraints()
        add_distance(c.distances, 0, 3, 1.50, 1.56)
        c.angles[angle_key] = (95.0, 105.0)
        bm, _tol = bounds._bounds(mol, c)
        return bm[3][0], bm[0][3]  # (lo, hi) for the pair (0, 3)

    assert window((0, 1, 3)) == pytest.approx(window((3, 1, 0))), "angle index ORDER changed the bounds"
    assert window((3, 1, 0)) == pytest.approx((1.50, 1.56), abs=1e-6), "the explicit window was clobbered"


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


def test_unembeddable_stereoisomer_is_skipped(monkeypatch):
    from rxembed.pipeline import dispatch

    def execute(spec, **_kwargs):
        if any(bond.GetStereo() == Chem.BondStereo.STEREOE for bond in spec.GetBonds()):
            raise RuntimeError("synthetic E failure")
        mol = _embedded("CC")
        return dispatch.Ensemble(mol, [0])

    monkeypatch.setattr(dispatch, "_execute", execute)
    r = rx.embed("C1CCC=CCCC1", n=1)

    assert isinstance(r, rx.Ensemble), "one surviving candidate must collapse to a bare Ensemble, not a set"
    assert r.n == 1
    assert r.tag.get("stereo") == "C3=C4:Z"
