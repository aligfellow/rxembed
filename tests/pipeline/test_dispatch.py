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
_BIMP = "examples/structures/bimp.xyz"
_BIMP_CORE = [10, 11, 12, 14]  # the reacting core, from the 04_organic_ts notebook


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


def test_fix_as_an_index_list_grafts_the_sources_own_coordinates():
    mol = _embedded("CC(=O)Nc1ccccc1", seed=1)
    ref = mol.GetConformer().GetPositions()
    ens = rx.embed(mol, fix=_AMIDE_CORE, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL


def test_fix_as_a_coordinate_dict_grafts_an_external_reference():
    """The coordinate dict is the primary reference mechanism; `template=` is only sugar over it."""
    ref = _embedded("CC(=O)Nc1ccccc1", seed=7).GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccccc1", fix={i: tuple(ref[i]) for i in _AMIDE_CORE}, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL


def test_fix_as_an_exact_number_reaches_an_explicit_hydrogen():
    """The FLP H-transfer idiom: an index into the AddHs Mol must survive the embed and the H must move."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CN"))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    h = next(a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors() if a.GetAtomicNum() == 1)
    ens = rx.embed(mol, fix={(n, h): 1.20}, n=6).minimize()  # held past its ~1.01 A equilibrium
    assert ens.n >= 1
    assert ens.measure((n, h))["mean"] == pytest.approx(1.20, abs=0.1)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_real_ts_core_from_an_xyz_holds_and_its_seed_passes_the_gate():
    """A real 172-atom TS: the graft is asserted on the public output, the whole-molecule gate on the raw seed.

    `rx.embed` returns through `_relax_into_windows`, and that relax costs bimp's thiourea conjugation; UFF
    under-restrains the H-N-C=S torsion and twists it out of plane. So the gate goes on `_embed_dispatch`'s
    seed, where a clean/not-clean answer means something; on the relaxed output an `any(ok)` would pass on
    luck. Recorded, not accommodated: a stiffer sp2 C=S/N torsion is the fix.
    """
    from rxembed.pipeline.dispatch import _embed_dispatch
    from rxembed.pipeline.perceive import _xyz_to_mol

    reference = _xyz_to_mol(_BIMP, 0)  # hoisted: re-perceiving a 172-atom TS per conformer is pure cost
    ref = reference.GetConformer().GetPositions()

    ens = rx.embed(_BIMP, fix=_BIMP_CORE, n=3)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _BIMP_CORE, ref) < _GRAFT_TOL

    seeds = _embed_dispatch(_BIMP, fix=_BIMP_CORE, n=3)
    assert _max_core_drift(seeds.mol, seeds.ids, _BIMP_CORE, ref) < _GRAFT_TOL
    for cid in seeds.ids:
        geom.check(seeds.mol, cid, frozen=_BIMP_CORE, reference=reference).assert_ok()


def test_a_rigid_core_and_a_soft_window_compose():
    """ "Hold a core, softly bias the periphery": the graft stays exact while the window genuinely bites."""
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
    """The organocatalysis headline: hold one known core geometry and place it on another scaffold."""
    ref = _embedded("CC(=O)Nc1ccccc1", seed=5).GetConformer().GetPositions()
    ens = rx.embed("CC(=O)Nc1ccc(C(C)(C)C)cc1", template=(ref, {i: i for i in _AMIDE_CORE}), n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, _AMIDE_CORE, ref) < _GRAFT_TOL
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()  # ...and the backbone around it is clean


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_template_reaches_the_auto_contacts_route_too():
    """`template=` must reach every route: the auto-NCI branch took no template argument and silently dropped it."""
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
@pytest.mark.parametrize(
    "complex_smiles",
    [
        "OC(=O)c1ccccc1.n1ccccc1",  # carboxylic acid + pyridine -> O-H...N
        "CC(=O)[O-].C[NH3+]",  # acetate + methylammonium -> a salt-bridge H-bond
    ],
)
def test_every_auto_discovered_grip_forms_at_an_h_bond_distance(complex_smiles):
    """Discovery is worthless if the seed does not realise it, so the contact distance is measured, not assumed."""
    es = rx.embed(complex_smiles, contacts="auto", n=6)
    for ens in es if isinstance(es, rx.EnsembleSet) else [es]:
        assert ens.n >= 1
        grip = ens.cons.contacts[0]
        assert grip, "a discovered binding mode must seed a releasable contact"
        settled = ens.minimize()
        for cid in settled.ids:
            geom.check(settled.mol, cid).assert_ok()
        for pair in grip:
            assert settled.measure(pair)["mean"] < 2.6, "the seeded grip did not form"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_a_named_mode_can_be_requested_instead_of_auto():
    """`rx.nci_modes(mol)['HB:...']` is the documented way to pick one grip: it must embed like 'auto' does."""
    from rxembed.pipeline.dispatch import _normalize

    smi = "OC(=O)c1ccccc1.n1ccccc1"
    modes = rx.nci_modes(_normalize(smi)[0])
    assert modes
    ens = rx.embed(smi, contacts=modes[next(iter(modes))], n=6).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# --- the organic path pays nothing for the metal path ------------------------------------------------------


def test_an_organic_spec_carries_no_metal_fields():
    """The zero-blast-radius guarantee: the three metal fields stay empty when there is no metal."""
    cons = rx.embed("CCO", n=1, seed=1).cons
    assert not cons.metals
    assert not cons.pulls
    assert not cons.floors


# --- rx.metal: what spec the isomer enumerator hands down ---------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_shape_held_sphere_gets_no_pull_but_a_modelled_window_does():
    """A pull on a rigid-body member is bought by tearing the pairs it did not pull, so the two never overlap."""
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
    """`_bounds` must sort its keys: an angle stated (k, j, i) once overwrote the distance window on (i, k)."""
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


@pytest.mark.parametrize(
    ("spec", "expected"),
    [({"fix": {(0, 6): 3.0}}, (3.0, 3.0)), ({"constrain": {(0, 6): (3.0, 3.4)}}, (3.0, 3.4))],
)
def test_minimize_relaxes_an_existing_geometry_toward_its_targets(spec, expected):
    mol = _embedded("CCCCCCC", seed=1)  # heptane; pull the two ends together
    lo, hi = expected
    assert lo - 0.15 <= rx.minimize(mol, **spec).measure((0, 6))["mean"] <= hi + 0.15


def test_minimize_accepts_an_xyz_path_and_refuses_a_smiles_that_has_no_geometry(tmp_path):
    """`minimize` relaxes what you already have: a SMILES has nothing to relax and must say so, not embed one."""
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
    """The tag is a claim about the coordinates, so it is read back off them: a mislabelled hand is silent."""
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
    """A defined centre, no centre, or stereo='free' each yields one Ensemble, not an EnsembleSet."""
    assert isinstance(rx.embed(smi, n=2, **kw), rx.Ensemble)


def test_stereo_enumerate_keeps_the_species_separate_and_uniformly_typed():
    r = rx.embed("CC(N)C(=O)O", n=2, stereo="enumerate")
    assert isinstance(r, list)
    assert len(r) == 2
    assert all(isinstance(g, rx.EnsembleSet) for g in r)  # uniform type, even for a lone organic variant
    assert sorted(g[0].tag["stereo"] for g in r) == ["1R", "1S"]


@pytest.mark.parametrize(
    ("smi", "n_candidates", "why"),
    [
        ("CC=CC(N)O", 4, "one undefined C x one undefined C=C: the E/Z axis reaches the route too"),
        ("CC(O)C(O)C", 3, "two centres, but the meso pair collapses; 3 candidates, not 4"),
    ],
)
def test_the_cores_species_list_reaches_the_set_neither_padded_nor_collapsed(smi, n_candidates, why):
    """The fold is a plumbing step: whatever `stereo.enumerate_unassigned` decided must arrive candidate for
    candidate. (What it decides is tests/test_stereo.py's; that it survives the fold is this.)"""
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
    # dead 0-conformer candidate is kept in the result.
    r = rx.embed("C1CCC=CCCC1", n=4)
    for e in [r] if isinstance(r, rx.Ensemble) else r:
        assert e.n >= 1
