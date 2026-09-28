"""Test Ensemble and EnsembleSet behavior."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_core as metal
from rxembed.pipeline import ensemble as ensemble_module
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.calculators import Calculator
from rxembed.utils import Violation

_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # covalent notation for the standard square-planar chelate fixture


def _pd_ensemble(n=3, smiles=_EN_PDBRCL):
    iso = rx.metal(smiles, "square_planar")[0]
    return iso, rx.embed(iso, n=n, seed=1).minimize()


def test_optional_connectivity_gate_does_not_break_base_minimize(monkeypatch, caplog):
    ens = rx.embed("CCCC", n=1, seed=1)
    monkeypatch.setattr(
        ensemble_module.Ensemble,
        "_scan_connectivity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("xyzgraph unavailable")),
    )

    with caplog.at_level("INFO", logger="rxembed"):
        assert ens.minimize().ids
    assert "connectivity is not re-checked" in caplog.text, "the skipped gate must say so"


def test_changed_connectivity_is_not_published(monkeypatch):
    ens = rx.embed("CCCC", n=1, seed=1)
    failed = ens.ids[0]
    monkeypatch.setattr(ensemble_module, "connectivity", lambda *args, **kwargs: ([(0, 3)], []))

    ens.minimize()

    assert not ens.ids
    assert failed in ens.discarded


def test_puckered_aromatic_ring_remains_a_geometry_diagnostic():
    ens = rx.embed("c1ccccc1", n=1, seed=1)
    cid = ens.ids[0]
    conf = ens._mol.GetConformer(cid)
    point = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, (point.x, point.y, point.z + 0.5))

    violations = ens.check()[cid].violations

    assert any(violation.detail == "aromatic ring puckered" for violation in violations)


def test_check_flags_an_undeclared_atom_that_collapsed_onto_the_metal():
    """A ring carbon pushed onto the metal must not be judged by a donor's looser floor.

    check() supplies the stated donors so metal_overbond can tell a real collapsed donor from a plain atom
    that only looks close: without them, the atom is perceived as a donor by distance alone and passes.
    """
    iso = rx.metal("[Cl-]->[Pt+2](<-[Cl-])(<-n1ccccc1)<-n1ccccc1", "SPL")[0]
    ens = rx.embed(iso, n=1, seed=42, threads=1)
    cid = ens.ids[0]
    mol = ens.mol  # `mol` is a property (a fresh restored copy); read it once and reuse that copy
    metal_idx = metal.metal_indices(mol)[0]
    ring = mol.GetRingInfo().AtomRings()[0]  # one bound pyridine
    nitrogen = next(i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
    distances = Chem.GetDistanceMatrix(mol)
    para = max(ring, key=lambda i: distances[nitrogen, i])  # not itself a donor

    conf = ens._mol.GetConformer(cid)
    pos = conf.GetPositions()
    pos[para] = pos[metal_idx] + 2.0 * (pos[para] - pos[metal_idx]) / np.linalg.norm(pos[para] - pos[metal_idx])
    conf.SetPositions(pos)

    assert "metal_overbond" in {v.kind for v in ens.check()[cid].violations}


def test_cleanup_ablations_filter_only_their_own_workflow_diagnostics(monkeypatch):
    iso = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", "SPL")[0]
    ens = rx.embed(iso, n=1, params=rx.EmbedParams(seed=42, donor_orientation=False, conjugation=False))
    report = geom.GeometryReport(
        [
            Violation("donor_orientation", (0, 1, 2), value=0.0, limit=90.0),
            Violation("conjugation", (0, 1, 2, 3), value=90.0, limit=30.0),
        ]
    )
    monkeypatch.setattr(ensemble_module.geom_check, "check", lambda *_args, **_kwargs: report)

    assert ens._workflow_failure(ens, ens.ids[0]) is None


def test_constraint_warning_reports_the_worst_conformer(caplog):
    ens = rx.embed("CC", n=1, seed=1)
    ens.ids.append(ens._mol.AddConformer(Chem.Conformer(ens._mol.GetConformer(ens.ids[0])), assignId=True))
    ens.cons.distances[(0, 1)] = (4.9, 5.1)
    for cid, distance in zip(ens.ids, (1.0, 9.0), strict=True):
        conf = ens._mol.GetConformer(cid)
        origin = np.array(conf.GetAtomPosition(0))
        conf.SetAtomPosition(1, (origin + np.array([distance, 0.0, 0.0])).tolist())

    with caplog.at_level("WARNING", logger="rxembed"):
        ens._validate()

    assert "distance(0, 1) = 1.00" in caplog.text


# --- the connectivity finalize: the output is a molecule, not a bag of fragments ---------------------------


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_derived_ensemble_preserves_connectivity():
    _iso, ens = _pd_ensemble(n=4)
    if ens.n < 2:
        pytest.skip("need >=2 conformers to derive a representative set")
    assert len(Chem.GetMolFrags(ens.representatives().mol)) == 1


# --- minimize and measure: what a caller may read back ------------------------------------------------------


# --- filter(): drop what is no longer the molecule you asked for --------------------------------------------


def test_geometry_filter_drops_only_failed_conformer():
    ens = rx.embed("CCO", n=1, seed=1).minimize()
    good = ens.ids[0]
    bad = ens._mol.AddConformer(Chem.Conformer(ens._mol.GetConformer(good)), assignId=True)
    ens.ids.insert(0, bad)
    ens.energies[bad] = ens.energies[good]
    conf = ens._mol.GetConformer(bad)
    conf.SetAtomPosition(0, conf.GetAtomPosition(1))
    reports = ens.check()
    assert not reports[bad].ok()
    assert reports[good].ok()

    energies = dict(ens.energies)
    assert ens.filter("geometry").ids == [good]
    assert ens.energies == {good: energies[good]}
    assert bad in ens.discarded


def test_geometry_filter_inherits_only_frozen_atoms():
    ens = rx.embed("CCO", n=1, seed=1).minimize()
    conf = ens._mol.GetConformer(ens.ids[0])
    conf.SetAtomPosition(0, conf.GetAtomPosition(1))
    ens.cons.frozen.update((0, 1))
    ens.cons.distances[(0, 2)] = (100.0, 101.0)
    assert ens.filter("geometry").ids
    with pytest.raises(RuntimeError, match="all 1 conformer"):
        ens.filter("geometry", constraints=ens.cons)

    ens.ids = []
    with pytest.raises(TypeError, match="frozne"):
        ens.filter("geometry", frozne=(0, 1))


# --- EnsembleSet: distinct species, never pooled -------------------------------------------------------------
#
# The racemate is the cheapest source of one.


# `mol` is a real Ensemble attribute; `ids` exists only as a bare class-level annotation, and the two reach
# the message down different halves of the `known` predicate.
@pytest.mark.parametrize("verb", ["ids"])
def test_set_rejects_scalar_ensemble_verbs(verb):
    r = rx.embed("CC(N)C(=O)O", n=2)
    assert isinstance(r, rx.EnsembleSet)
    with pytest.raises(AttributeError, match=r"pick one with \[0\] or \.select"):
        getattr(r, verb)
    assert not hasattr(r, "not_a_verb_at_all")  # an unrelated miss stays a plain AttributeError


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_set_keeps_both_meanings_of_filter():
    es = rx.embed("CC(N)C(=O)O", n=2)  # a racemate -> EnsembleSet
    assert es.filter(stereo="R") == es.filter(stereo="C:R") == es.filter(stereo="C1:R") == es.filter(stereo="1R")
    assert len(es.filter("connectivity")) == len(es)


def test_set_empty_stereo_matches_an_untagged_candidate():
    es = rx.EnsembleSet([rx.embed("CCO", n=1)])
    assert es.filter(stereo="") == es


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_minimize_and_prune_map_over_ensemble_set():
    r = rx.embed("CC(N)C(=O)O", n=3).minimize().prune()
    assert isinstance(r, rx.EnsembleSet)
    assert {e.tag["stereo"] for e in r} == {"C1:R", "C1:S"}  # both enantiomers survive the mapped chain
    for e in r:
        assert e.n >= 1
    assert isinstance(rx.embed("CCO", n=2).minimize().prune(), rx.Ensemble)  # a stereo-free SMILES stays single


def test_dump_paths_distinguish_unlabelled_metal_arrangements(tmp_path):
    isos = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "OCT", stereo="free")
    same_hand = rx.IsomerSet(isos.filter(hand="delta")[:2])
    ensembles = rx.EnsembleSet(rx.embed(iso, n=1, seed=1) for iso in same_hand)
    assert all(e.tag["arrangement"] in repr(ensembles) for e in ensembles)
    paths = ensembles.dump(str(tmp_path / "oct.xyz"))
    assert len(paths) == len(set(paths)) == 2


def test_best_refuses_ff_energies_across_species():
    # ranking distinct species needs real energies; FF (minimize / score('ff')) is not cross-comparable
    s = rx.embed("CC(N)C(=O)O", n=2).minimize()
    assert {e.energy_kind for e in s} == {"ff"}  # minimize tags FF energies
    with pytest.raises(ValueError, match="real energies"):
        s.best()
    assert {e.energy_kind for e in s.score("ff")} == {"ff"}  # an FF single point is still not "real"
    with pytest.raises(ValueError, match="real energies"):
        s.score("ff").best()


def test_uff_surrogate_cleanup_is_reported_as_an_approximate_objective():
    ens = rx.embed("NC(=[Se])N", n=1, seed=1).minimize()
    selenium = next(atom.GetIdx() for atom in ens.mol.GetAtoms() if atom.GetSymbol() == "Se")

    assert ens.energies
    assert ens.uff.surrogates == {selenium: (34, 16)}
    assert ens.energy_kind == "uff-surrogate"
    assert ens[0].energy_kind == "uff-surrogate"


def test_slice_preserves_ensemble_state():
    ens = rx.embed("CCCCO", n=4, params=rx.EmbedParams(seed=1, knowledge=False)).minimize()
    flagged = ens.ids[0]
    ens.unrelaxed = [flagged]
    ens.uff.surrogates = {3: (34, 16)}
    ens.uff.retyped = {(1, 2)}
    assert ens.energy_kind == "ff"
    child = ens[0]
    assert child.energy_kind == "ff"
    assert child.params == rx.EmbedParams(seed=1, knowledge=False)
    assert child.unrelaxed is not ens.unrelaxed
    assert child.uff is not ens.uff
    assert child.unrelaxed == [flagged]
    assert child.uff == ens.uff
    assert ens.lowest(2).energy_kind == "ff"
    assert ens.lowest(2).uff.surrogates == ens.uff.surrogates
    aligned = ens.align()
    assert aligned.energy_kind == "ff"
    assert (aligned.params.knowledge, aligned.params.prune_rms) == (False, None)

    ens.trajectory = Chem.Mol(ens._mol)
    assert ens[:1].trajectory is None


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_refinement_preserves_ensemble_records():
    class FixedCalculator(Calculator):
        def energy(self, mol, conf_id=-1):
            return -1.0

        def optimize(self, mol, conf_id=-1, level="normal", fix=()):
            return mol.GetConformer(conf_id).GetPositions(), -1.0

    _iso, ens = _pd_ensemble(n=3)
    dropped = ens.ids.pop()
    ens.discarded.append(dropped)

    for out in (ens.score(FixedCalculator()), ens.optimize(FixedCalculator())):
        assert out.sphere == ens.sphere
        assert out.discarded == ens.discarded
        assert dropped in {c.GetId() for c in out._mol.GetConformers()}
        assert dropped not in {c.GetId() for c in out.mol.GetConformers()}


def test_wrap_labels_its_computed_force_field_single_point():
    mol = rx.embed("CCO", n=1, seed=1).mol
    ens = rx.wrap(mol, minimized=True)

    assert ens.energies
    assert ens.energy_kind == "ff"


# --- fragment contacts: mc's explore pass must re-bound every free ion pair, not one chosen contact ---
