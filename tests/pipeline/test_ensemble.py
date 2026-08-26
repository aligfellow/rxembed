"""Test Ensemble and EnsembleSet behavior."""

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed import metal_core as metal
from rxembed.pipeline.calculators import Calculator

_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # a neutral square-planar chelate: the standard metal fixture
_MN_H2 = "examples/structures/mn-h2.xyz"  # bimetallic: an Mn centre and a spectator ferrocene
_MN_H2_RC = [1, 5, 63, 64, 65, 66]


def _pd_ensemble(n=3, smiles=_EN_PDBRCL):
    iso = rx.metal(smiles, "square_planar")[0]
    return iso, rx.embed(iso, n=n, seed=1).minimize()


def _dative(mol, donor, metal_idx):
    b = mol.GetBondBetweenAtoms(int(donor), int(metal_idx))
    return b is not None and b.GetBondType() == Chem.BondType.DATIVE and b.GetBeginAtomIdx() == int(donor)


def _dissociate(ens, cid, atom, centre, distance=4.0):
    conf = ens._mol.GetConformer(cid)
    p, pm = np.array(conf.GetAtomPosition(int(atom))), np.array(conf.GetAtomPosition(int(centre)))
    conf.SetAtomPosition(int(atom), (pm + distance * (p - pm) / np.linalg.norm(p - pm)).tolist())


# --- the connectivity finalize: the output is a molecule, not a bag of fragments ---------------------------


def test_all_isomers_return_connected():
    seen = 0
    for iso in rx.metal("Cl[Pd](Cl)(N)N", "square_planar"):
        ens = rx.embed(iso, n=2, seed=1).minimize()
        if not ens.n:
            continue
        seen += 1
        assert len(Chem.GetMolFrags(ens.mol)) == 1, f"isomer {ens.tag.get('arrangement')} came out disconnected"
        for d in iso.donors:
            assert _dative(ens.mol, d, iso.metal), f"donor {d} is not dative-bonded to the metal"
        assert "Pd" in Chem.MolToSmiles(ens.mol), "the connected graph must round-trip to a real complex SMILES"
        assert Chem.GetFormalCharge(ens.mol) == 0, "Cl2Pd(N)2 is neutral; the finalize must not perturb it"
        dative = [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
        assert dative, "the finalize added no dative bond at all: there is nothing to judge"
        for b in dative:  # connectivity only: no ligand atom gained a neighbour
            assert iso.metal in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    assert seen, "every isomer minimised to empty: the finalize was never exercised"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_spectator_ferrocene_stays_rigid():

    from rxembed.pipeline.perceive import read_xyz

    ref = read_xyz(_MN_H2, 0)  # find the spectator from the MOLECULE, so a missing record fails loudly
    fe = next(
        a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() in metal.TRANSITION_METALS and a.GetSymbol() != "Mn"
    )
    shape = {fe, *(n.GetIdx() for n in ref.GetAtomWithIdx(fe).GetNeighbors())}

    isomers = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    iso = isomers.select(arrangement="C62 N6 C61 N5 P2 H63")
    windows = {k: v for k, v in iso.cons.distances.items() if set(k) <= shape}
    assert len(windows) > 50, "the rigid body is all pairs of {Fe, *10 Cp carbons}"

    ens = rx.embed(iso, n=2, seed=1).minimize(_retry=False)  # seed 0 tears; seed 1 is the smallest live witness
    assert ens.n, "no conformer survived: the per-conformer assertion below never ran"
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    assert all(ens.mol.GetAtomWithIdx(m).GetDegree() for m in metal.metal_indices(ens.mol))
    for cid in ens.ids:
        worst = max(  # how far outside its own window the worst held pair has been pushed
            max(lo - (d := rdMolTransforms.GetBondLength(ens.mol.GetConformer(cid), i, j)), d - hi, 0.0)
            for (i, j), (lo, hi) in windows.items()
        )
        assert worst < 0.15, f"conf {cid}: the spectator's shape tore by {worst:.3f} A"


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_derived_ensemble_preserves_connectivity():
    _iso, ens = _pd_ensemble(n=4)
    if ens.n < 2:
        pytest.skip("need >=2 conformers to derive a representative set")
    assert len(Chem.GetMolFrags(ens.representatives().mol)) == 1


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_search_disconnects_then_minimize_reconnects_metal():
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    searched = rx.embed(iso, n=6, seed=1)
    floors = dict(searched.cons.floors)
    searched.minimize().mc(preset="ensemble", seed=1).minimize()
    assert searched.n >= 1, "the mc search collapsed: a bonded metal was handed to the relax"
    assert searched.cons.floors == floors, "the search dropped the structural non-donor floor"
    assert searched.metal_bonds, "the M-L bond record must outlive the surrogate teardown"
    assert len(Chem.GetMolFrags(searched.mol)) == 1


def test_organic_minimize_adds_no_dative_bonds():
    ens = rx.embed("CCO").minimize()
    assert ens.n
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    assert not [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]


# --- minimize and measure: what a caller may read back ------------------------------------------------------


def test_stretched_bond_survives_relax_and_measure():
    ens = rx.embed("CCCl", fix={(1, 2): 2.4}, n=4, seed=42).minimize()
    assert ens.ids, "the requested dissociating C-Cl was thrown away for being what was asked for"
    stats = ens.measure((1, 2))
    assert stats["mean"] == pytest.approx(2.4, abs=0.1)
    assert 90.0 < ens.measure((0, 1, 2))["mean"] < 130.0, "a C-C-Cl angle outside any sp3 range"


# --- filter(): drop what is no longer the molecule you asked for --------------------------------------------


def test_filter_rejects_rmsd_dedup():
    with pytest.raises(ValueError, match="prune"):
        rx.embed("CCO", n=1, seed=1).filter("rmsd")


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
    assert ens.energies == energies
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


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_filter_drops_only_dissociated_ligand():
    iso, ens = _pd_ensemble(n=6)
    assert ens.sphere, "the coordination sphere was forgotten by minimize()"
    assert not ens._scan_connectivity(), "a healthy metal ensemble must not be flagged"

    donor = iso.donors[0]
    _dissociate(ens, ens.ids[0], donor, iso.metal)
    before = ens.n
    assert ens._scan_connectivity(), "a dissociated ligand was not seen through the pipeline"
    ens.cons.frozen.update((iso.metal, donor))
    assert not ens._scan_connectivity(), "a reacting-core metal-donor pair was judged as ground-state coordination"
    ens.cons.frozen.difference_update((iso.metal, donor))
    ens.filter("connectivity")
    assert ens.n == before - 1, "filter must drop the dissociated conformer and ONLY that one"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    "smi",
    [
        # a 1.71 A C=P phosphaalkene: xyzgraph refuses to perceive it and calls the bond broken
        "Cc1cc(C)c([CH]2=[PH]->[Ni+2]<-23<-[O-]C(=O)C(c2ccccc2)[N-]->3c2ccccc2)c(C)c1",
        # a short 1,3 carbon pair at 2.02 A: xyzgraph calls that separation a newly formed bond
        "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    ],
    ids=["phosphaalkene", "short-1-3-pair"],
)
def test_healthy_catalyst_keeps_connectivity(smi):
    judged = False
    for iso in rx.metal(smi, "square_planar", stereo="free"):
        ens = rx.embed(iso, n=1, seed=1).minimize()
        if not ens.n:
            continue
        assert not ens._scan_connectivity()
        assert ens.filter("connectivity").n == ens.n
        judged = True
        break
    assert judged, "every isomer minimised to empty: nothing was ever judged"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_reacted_conformer_is_flagged_before_filtering():
    ens = rx.embed("[NH3+]CC(=O)[O-]", n=1, seed=1).minimize()
    cid = ens.ids[0]
    conf = ens._mol.GetConformer(cid)
    n = next(a.GetIdx() for a in ens._mol.GetAtoms() if a.GetSymbol() == "N")
    o = next(a.GetIdx() for a in ens._mol.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    h = next(x.GetIdx() for x in ens._mol.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 1)
    cc = next(x.GetIdx() for x in ens._mol.GetAtomWithIdx(o).GetNeighbors() if x.GetAtomicNum() == 6)
    po, pc = np.array(conf.GetAtomPosition(o)), np.array(conf.GetAtomPosition(cc))
    conf.SetAtomPosition(h, (po + 0.98 * (po - pc) / np.linalg.norm(po - pc)).tolist())  # a real transfer

    changed = ens._scan_connectivity()  # exactly what optimize() runs on its output geometry
    assert cid in changed
    assert all(changed[cid]), "a transfer both forms and breaks a bond"
    assert ens.ids == [cid], "flagging must NOT shrink the ensemble"
    with pytest.raises(RuntimeError, match="different species"):
        ens.filter("connectivity")


# --- EnsembleSet: distinct species, never pooled -------------------------------------------------------------
#
# The racemate is the cheapest source of one.


# `mol` is a real Ensemble attribute; `ids` exists only as a bare class-level annotation, and the two reach
# the message down different halves of the `known` predicate.
@pytest.mark.parametrize("verb", ["mol", "ids"])
def test_set_rejects_scalar_ensemble_verbs(verb):
    r = rx.embed("CC(N)C(=O)O", n=2)
    assert isinstance(r, rx.EnsembleSet)
    with pytest.raises(AttributeError, match="stereo='free'"):
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


def test_dump_writes_one_tagged_xyz_per_candidate(tmp_path):
    paths = rx.embed("CC(N)C(=O)O", n=2).minimize().dump(str(tmp_path / "amac.xyz"))
    assert sorted(Path(p).name for p in paths) == ["amac_C1_R.xyz", "amac_C1_S.xyz"]
    for p in paths:
        assert int(open(p).readline().strip()) == 13  # a valid .xyz (atom count header)


def test_dump_paths_distinguish_unlabelled_metal_arrangements(tmp_path):
    isos = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "OCT", stereo="free")
    same_hand = rx.IsomerSet(isos.filter(hand="delta")[:2])
    ensembles = rx.EnsembleSet(rx.embed(iso, n=1, seed=1) for iso in same_hand)
    assert all(e.tag["arrangement"] in repr(ensembles) for e in ensembles)
    paths = ensembles.dump(str(tmp_path / "oct.xyz"))
    assert len(paths) == len(set(paths)) == 2


def test_dump_refuses_an_empty_ensemble(tmp_path):
    ens = rx.embed("CCO", n=2)
    ens.ids = []
    with pytest.raises(ValueError, match="nothing to dump"):
        ens.dump(str(tmp_path / "empty.xyz"))


def test_best_refuses_ff_energies_across_species():
    # ranking distinct species needs real energies; FF (minimize / score('ff')) is not cross-comparable
    s = rx.embed("CC(N)C(=O)O", n=2).minimize()
    assert {e.energy_kind for e in s} == {"ff"}  # minimize tags FF energies
    with pytest.raises(ValueError, match="real energies"):
        s.best()
    assert {e.energy_kind for e in s.score("ff")} == {"ff"}  # an FF single point is still not "real"
    with pytest.raises(ValueError, match="real energies"):
        s.score("ff").best()


def test_slice_preserves_ensemble_state():
    ens = rx.embed("CCCCO", n=4, seed=1).minimize()
    flagged = ens.ids[0]
    ens.seed = 1
    ens._seeds_relaxed = True
    ens.unrelaxed = [flagged]
    ens.wrong_hand = [flagged]
    assert ens.energy_kind == "ff"
    child = ens[0]
    assert child.energy_kind == "ff"
    assert child.seed == 1
    assert child._seeds_relaxed
    assert child.unrelaxed == [flagged]
    assert child.wrong_hand == [flagged]
    assert ens.lowest(2).energy_kind == "ff"
    assert ens.align().energy_kind == "ff"


def test_reembedded_conformer_drops_stale_unrelaxed_id():
    iso = rx.metal("[Pd](Cl)(Cl)(Cl)([N@H](C)O)", "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=2)
    assert ens.unrelaxed == [0]
    ens.minimize()
    assert ens.ids == [1]
    assert not ens.unrelaxed


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


def test_optimize_rejects_numeric_fix_drift():
    class MovingCalculator(Calculator):
        def optimize(self, mol, conf_id=-1, level="normal", fix=()):
            pos = mol.GetConformer(conf_id).GetPositions()
            pos[0] += [1.0, 0.0, 0.0]
            return pos, -1.0

    ens = rx.embed("[O-].ClCCCCBr", fix={(2, 0): 1.557}, n=1, seed=1, stereo="free")
    with pytest.raises(RuntimeError, match="moved every numeric fix"):
        ens.optimize(MovingCalculator())
