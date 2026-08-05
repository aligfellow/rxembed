"""`pipeline/dispatch.py`: a source plus a spec becomes an Ensemble, and every argument reaches every route.

The frozen-core Kabsch graft is the load-bearing claim: a TS's partial bonds must survive a random-frame
embed, so a fixed core's internal geometry is preserved to < 0.01 Å on every conformer.
"""

import itertools
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import rxembed.pipeline as rx
from rxembed.pipeline import geom_check as geom

_GRAFT_TOL = 0.01  # a fixed core is held exactly: the frozen-core distance assertion the project guarantees
_AMIDE_CORE = [0, 1, 2, 3]  # the conserved C-C(=O)-N motif: the same leading indices in every analogue below
_SN2 = "examples/structures/sn2.xyz"
_SN2_CORE = [4, 0, 5]  # F...C...Cl reacting core, from the templated-TS notebook


def _embedded(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def _max_core_drift(mol, ids, core, ref_pos):
    """Largest deviation of any core pair's distance from the reference, over all conformers (frame-free)."""
    return max(
        abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j]))
        for pos in (mol.GetConformer(c).GetPositions() for c in ids)
        for i, j in itertools.combinations(core, 2)
    )


# --- fix: the rigid graft, in each of the three forms the resolver accepts ---------------------------------


def test_fix_grafts_a_core_from_an_index_list_and_from_a_coordinate_dict():
    mol = _embedded("CC(=O)Nc1ccccc1", seed=1)
    own = mol.GetConformer().GetPositions()
    ens = rx.embed(mol, fix=_AMIDE_CORE, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, own) < _GRAFT_TOL

    ref = _embedded("CC(=O)Nc1ccccc1", seed=7).GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccccc1", fix={i: tuple(ref[i]) for i in _AMIDE_CORE}, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL


def test_fix_as_an_exact_number_reaches_an_explicit_hydrogen():
    mol = Chem.AddHs(Chem.MolFromSmiles("CN"))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    h = next(a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors() if a.GetAtomicNum() == 1)
    ens = rx.embed(mol, fix={(n, h): 1.20}, n=6).minimize()  # held past its ~1.01 A equilibrium
    assert ens.n >= 1
    assert ens.measure((n, h))["mean"] == pytest.approx(1.20, abs=0.1)


def test_a_coordinate_free_hydride_uses_the_ml_target(monkeypatch):
    from rxembed import metal_distance as distance
    from rxembed.metal_coordination import _ML_SEED_HALF_WIDTH
    from rxembed.pipeline import dispatch

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
    monkeypatch.setattr(dispatch, "seed_conformers", lambda mol, *args, **kwargs: (mol, []))

    ens = rx.embed(source, constrain={(2, 3): (2.5, 4.0)}, n=1)
    lo, hi = ens.cons.distances[(hydride, metal)]

    assert (lo + hi) / 2 == pytest.approx(target)
    assert hi - lo == pytest.approx(2 * _ML_SEED_HALF_WIDTH)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_real_ts_core_from_an_xyz_holds_and_its_seed_passes_the_gate():
    from rxembed.pipeline.dispatch import _embed_dispatch
    from rxembed.pipeline.perceive import _xyz_to_mol

    reference = _xyz_to_mol(_SN2, 0)
    ref = reference.GetConformer().GetPositions()
    seeds = _embed_dispatch(_SN2, fix=_SN2_CORE, n=1)
    assert _max_core_drift(seeds.mol, seeds.ids, _SN2_CORE, ref) < _GRAFT_TOL
    for cid in seeds.ids:
        geom.check(seeds.mol, cid, frozen=_SN2_CORE, reference=reference).assert_ok()


def test_a_rigid_core_and_a_soft_window_compose():
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


def test_a_reference_core_transfers_onto_a_different_backbone():
    ref = _embedded("CC(=O)Nc1ccccc1", seed=5).GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccc(C(C)(C)C)cc1", template=(ref, {i: i for i in _AMIDE_CORE}), n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()  # ...and the backbone around it is clean


def test_a_template_composes_with_a_list_fix_from_the_sources_own_geometry():
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


def test_an_empty_ensemble_cannot_silently_supply_a_discarded_template_conformer():
    ref = rx.embed("CCO", n=1, seed=1)
    ref.ids.clear()

    with pytest.raises(ValueError, match="tracked conformer"):
        rx.embed("CCO", template=(ref, {0: 0}), n=1, seed=1)


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_template_reaches_the_auto_contacts_route_too():
    smi = "CC(=O)O.n1ccccc1"
    ref = rx.embed(smi, n=2)
    ref_pos = ref.mol.GetConformer(ref.ids[0]).GetPositions()
    core = [0, 1, 2, 3]  # the acetic-acid heavy core

    out = rx.embed(smi, contacts="auto", n=4, template=(ref.mol, {i: i for i in core}))
    for ens in out if isinstance(out, rx.EnsembleSet) else [out]:
        assert sorted(ens.cons.frozen) == core, "the template never reached the auto-contacts route"
        assert _max_core_drift(ens.mol, ens.ids[:1], core, ref_pos) < 1e-6


# --- contacts: a discovered binding mode is one the embed can actually realise ------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
@pytest.mark.parametrize("complex_smiles", ["OC(=O)c1ccccc1.n1ccccc1"])  # acid + pyridine -> O-H...N
def test_every_auto_discovered_grip_forms_at_an_h_bond_distance(complex_smiles):
    es = rx.embed(complex_smiles, contacts="auto", n=6)
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


# --- the organic path pays nothing for the metal path ------------------------------------------------------


# --- rx.metal: what spec the isomer enumerator hands down ---------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_shape_held_sphere_gets_no_pull_but_a_modelled_window_does():
    iso = rx.metal("examples/structures/mn-h2.xyz", "octahedral", center="Mn", fix=[1, 5, 63, 64, 65, 66])[0]
    spectators = {m for m in iso.cons.metals if m != iso.metal}
    assert spectators, "mn-h2 is bimetallic: the ferrocene Fe must be surrogated as a spectator"
    assert spectators <= set().union(*iso.cons.shapes), "the spectator's sphere is the rigid body hold_shape pinned"
    assert not [k for k in iso.cons.pulls if spectators & set(k)]
    for d in iso.donors:  # ...while the enumerated centre's modelled window still gets its pull
        assert (min(iso.metal, d), max(iso.metal, d)) in iso.cons.pulls
    assert iso.cons.relaxed().shapes == iso.cons.shapes, "mc(explore=) must not release a structural shape hold"

    from_smiles = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]  # no input geometry -> nothing shape-held
    assert not from_smiles.cons.shapes
    assert len(from_smiles.cons.pulls) == len(from_smiles.donors)


# The subject below is the core's `bounds._bounds`; it is pinned here because the spec that reaches it is
# the dispatch's, and this file is where that spec is otherwise exercised.
def test_an_angle_written_backwards_does_not_clobber_an_explicit_distance_window():
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


@pytest.mark.parametrize(("spec", "expected"), [({"constrain": {(0, 6): (3.0, 3.4)}}, (3.0, 3.4))])
def test_minimize_relaxes_an_existing_geometry_toward_its_targets(spec, expected):
    mol = _embedded("CCCCCCC", seed=1)  # heptane; pull the two ends together
    lo, hi = expected
    assert lo - 0.15 <= rx.minimize(mol, **spec).measure((0, 6))["mean"] <= hi + 0.15


def test_minimize_accepts_an_xyz_path_and_refuses_a_smiles_that_has_no_geometry(tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(Chem.MolToXYZBlock(_embedded("CCCCCCC", seed=1)))
    assert rx.minimize(str(xyz), fix={(0, 6): 3.0}).measure((0, 6))["mean"] == pytest.approx(3.0, abs=0.15)
    with pytest.raises(ValueError, match="existing geometry"):
        rx.minimize("CCO", fix={(0, 2): 2.0})


# --- the `stereo=` route: an undefined centre is a set of distinct species ----------------------------------
#
# `_stereo_expand` drives the core `stereo.enumerate_unassigned` (its own graph-level contract is
# tests/test_stereo.py) and folds the variants into one EnsembleSet. These are the pipeline end of it.


def _configs(es):
    return sorted(e.tag.get("stereo") for e in es)


def test_both_hands_come_back_as_a_set_and_each_tag_is_the_geometrys_own_configuration():
    es = rx.embed("CC(N)C(=O)O", n=3)  # undefined alpha-carbon
    assert isinstance(es, rx.EnsembleSet)
    assert _configs(es) == ["1R", "1S"]  # index-keyed CIP tags
    for e in es:
        em = e.minimize()  # the raw ETKDG seed can carry a conjugation twist; minimise, then read the hand
        Chem.AssignStereochemistryFrom3D(em.mol, confId=em.ids[0])
        ((idx, code),) = Chem.FindMolChiralCenters(em.mol, useLegacyImplementation=False)
        assert e.tag["stereo"] == f"{idx}{code}"


@pytest.mark.parametrize(
    ("smi", "kw"),
    [
        pytest.param("C[C@H](N)C(=O)O", {}, id="defined-centre-kept"),
        pytest.param("CCO", {}, id="no-stereocentre"),
        pytest.param("CC(N)C(=O)O", {"stereo": "free"}, id="stereo-free-opts-out"),
    ],
)
def test_one_plain_ensemble_when_there_is_nothing_to_enumerate(smi, kw):
    assert isinstance(rx.embed(smi, n=2, **kw), rx.Ensemble)


@pytest.mark.parametrize(
    ("smi", "n_candidates", "why"),
    [
        ("CC=CC(N)O", 4, "one undefined C x one undefined C=C: the E/Z axis reaches the route too"),
        ("CC(O)C(O)C", 3, "two centres, but the meso pair collapses; 3 candidates, not 4"),
    ],
)
def test_the_cores_species_list_reaches_the_set_neither_padded_nor_collapsed(smi, n_candidates, why):
    es = rx.embed(smi, n=2)
    assert isinstance(es, rx.EnsembleSet)
    assert len(es) == n_candidates, why
    if n_candidates == 4:
        assert any(":E" in c for c in _configs(es))
        assert any(":Z" in c for c in _configs(es))


def test_the_stereo_axis_composes_with_the_metal_coordination_axis():
    # an aminoacidate on Pd: 2 ligand enantiomers x the square-planar coordination isomers
    r = rx.embed("CC(N)C(=O)[O-]->[Pd]([Cl])[Cl]", metal="square_planar", n=2)
    assert isinstance(r, rx.EnsembleSet)
    assert {"1R", "1S"} == {e.tag["stereo"] for e in r}
    assert {"cis", "trans"} <= {e.tag["label"] for e in r}
    for e in r:  # every candidate carries both axes
        assert e.tag.get("stereo")
        assert e.tag.get("label")


def test_an_allene_axis_stays_a_bare_chainable_ensemble():
    # RDKit can't enumerate allene/cumulene axial chirality from a flat SMILES -> one arbitrary hand, not an
    # EnsembleSet-of-1 (that would break the documented rx.embed(smi).mc().prune() chain), and no '?' tag.
    r = rx.embed("CC(F)=C=C(F)C", n=2)
    assert isinstance(r, rx.Ensemble)
    assert hasattr(r, "mc")
    assert "?" not in (r.tag.get("stereo") or "")


def test_a_stereoisomer_that_will_not_embed_is_skipped_not_kept_empty():
    # trans-cyclooctene is too strained for ETKDG (0 conformers); only the embeddable Z survives, and no
    # dead 0-conformer candidate is kept in the result. Asserting the E is GONE, not just that whatever came
    # back is non-empty: a kept 0-conformer E candidate satisfied the latter and left the claim untested.
    r = rx.embed("C1CCC=CCCC1", n=1)
    assert isinstance(r, rx.Ensemble), "one surviving candidate must collapse to a bare Ensemble, not a set"
    assert r.n >= 1, "the embeddable Z was dropped too"
    stereo = {e.tag.get("stereo") for e in r}
    assert stereo == {"3=4:Z"}, f"the unembeddable E was kept as a dead candidate: {stereo}"
