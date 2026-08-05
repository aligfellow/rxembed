"""`pipeline/ensemble.py`: what `Ensemble`/`EnsembleSet` guarantee about the mol they hand back.

The surrogate strips the M-donor bonds so the DG/FF can embed a bond-less metal; ``minimize()`` is where that
is torn down again; element, oxidation state, geometry, then connectivity. These pin what a caller gets.
"""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed.pipeline as rx
from rxembed import metal_core as metal

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
    conf = ens.mol.GetConformer(cid)
    p, pm = np.array(conf.GetAtomPosition(int(atom))), np.array(conf.GetAtomPosition(int(centre)))
    conf.SetAtomPosition(int(atom), (pm + distance * (p - pm) / np.linalg.norm(p - pm)).tolist())


# --- the connectivity finalize: the output is a molecule, not a bag of fragments ---------------------------


def test_every_isomer_comes_back_connected_through_its_metal():
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


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_bimetallic_output_connects_every_metal():
    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    ens = rx.embed(isos[0], n=4, seed=1).minimize(_retry=False)
    if not ens.n:
        pytest.skip("no conformer survived the relax at this deterministic seed; connectivity is unexercised")
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    for mi in metal.metal_indices(ens.mol):
        assert ens.mol.GetAtomWithIdx(mi).GetDegree() > 0, f"metal {mi} was left disconnected"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_the_spectator_ferrocenes_rigid_body_is_never_traded_for_a_pull():

    from rxembed.pipeline.perceive import _xyz_to_mol

    ref = _xyz_to_mol(_MN_H2, 0)  # find the spectator from the MOLECULE, so a missing record fails loudly
    fe = next(
        a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() in metal.TRANSITION_METALS and a.GetSymbol() != "Mn"
    )
    shape = {fe, *(n.GetIdx() for n in ref.GetAtomWithIdx(fe).GetNeighbors())}

    iso = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)[0]
    windows = {k: v for k, v in iso.cons.distances.items() if set(k) <= shape}
    assert len(windows) > 50, "the rigid body is all pairs of {Fe, *10 Cp carbons}"

    ens = rx.embed(iso, n=2, seed=1).minimize(_retry=False)  # seed 0 tears; seed 1 is the smallest live witness
    assert ens.n, "no conformer survived: the per-conformer assertion below never ran"
    for cid in ens.ids:
        worst = max(  # how far outside its own window the worst held pair has been pushed
            max(lo - (d := rdMolTransforms.GetBondLength(ens.mol.GetConformer(cid), i, j)), d - hi, 0.0)
            for (i, j), (lo, hi) in windows.items()
        )
        assert worst < 0.15, f"conf {cid}: the spectator's shape tore by {worst:.3f} A"


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_a_derived_ensemble_inherits_the_finalized_connectivity():
    _iso, ens = _pd_ensemble(n=4)
    if ens.n < 2:
        pytest.skip("need >=2 conformers to derive a representative set")
    assert len(Chem.GetMolFrags(ens.representatives().mol)) == 1


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_the_search_runs_on_the_bare_mol_and_the_closing_minimize_re_connects():
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    searched = rx.embed(iso, n=6, seed=1)
    floors = dict(searched.cons.floors)
    searched.minimize().mc(preset="ensemble", seed=1).minimize()
    assert searched.n >= 1, "the mc search collapsed: a bonded metal was handed to the relax"
    assert searched.cons.floors == floors, "the search dropped the structural non-donor floor"
    assert searched.metal_bonds, "the M-L bond record must outlive the surrogate teardown"
    assert len(Chem.GetMolFrags(searched.mol)) == 1


def test_an_organic_output_is_untouched_by_the_finalize():
    ens = rx.embed("CCO").minimize()
    assert ens.n
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    assert not [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]


# --- minimize and measure: what a caller may read back ------------------------------------------------------


def test_a_requested_stretched_bond_survives_the_relax_and_measure_reads_it_back():
    ens = rx.embed("CCCl", fix={(1, 2): 2.4}, n=4, seed=42).minimize()
    assert ens.ids, "the requested dissociating C-Cl was thrown away for being what was asked for"
    stats = ens.measure((1, 2))
    assert stats["mean"] == pytest.approx(2.4, abs=0.1)
    assert 90.0 < ens.measure((0, 1, 2))["mean"] < 130.0, "a C-C-Cl angle outside any sp3 range"


# --- filter(): drop what is no longer the molecule you asked for --------------------------------------------


def test_filter_refuses_a_method_that_is_really_a_dedup():
    with pytest.raises(ValueError, match="prune"):
        rx.embed("CCO", n=1, seed=1).filter("rmsd")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_dissociated_ligand_is_seen_through_the_pipeline_and_only_that_conformer_drops():
    iso, ens = _pd_ensemble(n=6)
    assert ens.sphere, "the coordination sphere was forgotten by minimize()"
    assert not ens._scan_connectivity(), "a healthy metal ensemble must not be flagged"

    _dissociate(ens, ens.ids[0], iso.donors[0], iso.metal)
    before = ens.n
    assert ens._scan_connectivity(), "a dissociated ligand was not seen through the pipeline"
    ens.filter("connectivity")
    assert ens.n == before - 1, "filter must drop the dissociated conformer and ONLY that one"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
@pytest.mark.parametrize(
    "smi",
    [
        # a 1.71 A C=P phosphaalkene: xyzgraph refuses to perceive it and calls the bond broken
        "Cc1cc(C)c([CH]2=[PH]->[Ni+2]<-23<-[O-]C(=O)C(c2ccccc2)[N-]->3c2ccccc2)c(C)c1",
        # a 1,3 geminal pair at 2.02 A: xyzgraph calls that separation a newly formed bond
        "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    ],
)
def test_a_healthy_catalyst_is_never_flagged_as_having_reacted(smi):
    judged = False
    for iso in rx.metal(smi, "square_planar", stereo="free"):
        ens = rx.embed(iso, n=3, seed=1).minimize()
        if not ens.n:
            continue
        assert not ens._scan_connectivity()
        assert ens.filter("connectivity").n == ens.n
        judged = True
        break
    assert judged, "every isomer minimised to empty: nothing was ever judged"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_reacted_conformer_is_flagged_first_and_only_dropped_when_asked():
    ens = rx.embed("[NH3+]CC(=O)[O-]", n=1, seed=1).minimize()
    cid = ens.ids[0]
    conf = ens.mol.GetConformer(cid)
    n = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "N")
    o = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    h = next(x.GetIdx() for x in ens.mol.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 1)
    cc = next(x.GetIdx() for x in ens.mol.GetAtomWithIdx(o).GetNeighbors() if x.GetAtomicNum() == 6)
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
def test_a_scalar_ensemble_verb_on_a_set_raises_and_names_the_way_out(verb):
    r = rx.embed("CC(N)C(=O)O", n=2)
    assert isinstance(r, rx.EnsembleSet)
    with pytest.raises(AttributeError, match="stereo='free'"):
        getattr(r, verb)
    assert not hasattr(r, "not_a_verb_at_all")  # an unrelated miss stays a plain AttributeError


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_the_set_keeps_both_meanings_of_filter():
    es = rx.embed("CC(N)C(=O)O", n=2)  # a racemate -> EnsembleSet
    assert len(es.filter(stereo="1R")) == 1, "the tag selector was broken"
    assert len(es.filter("connectivity")) == len(es)


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_the_headline_chain_maps_over_a_set_and_carries_the_tags():
    r = rx.embed("CC(N)C(=O)O", n=3).minimize().prune()
    assert isinstance(r, rx.EnsembleSet)
    assert {e.tag["stereo"] for e in r} == {"1R", "1S"}  # both enantiomers survive the mapped chain
    for e in r:
        assert e.n >= 1
    assert isinstance(rx.embed("CCO", n=2).minimize().prune(), rx.Ensemble)  # a stereo-free SMILES stays single


def test_dump_writes_one_tagged_xyz_per_candidate(tmp_path):
    paths = rx.embed("CC(N)C(=O)O", n=2).minimize().dump(str(tmp_path / "amac.xyz"))
    assert sorted(p.rsplit("_", 1)[1] for p in paths) == ["1R.xyz", "1S.xyz"]  # tag folded into each filename
    for p in paths:
        assert int(open(p).readline().strip()) == 13  # a valid .xyz (atom count header)


def test_dump_refuses_an_empty_ensemble(tmp_path):
    ens = rx.embed("CCO", n=2)
    ens.ids = []
    with pytest.raises(ValueError, match="nothing to dump"):
        ens.dump(str(tmp_path / "empty.xyz"))


def test_best_refuses_ff_energies_across_species():
    # ranking distinct species needs real energies; FF (minimize / score('ff')) is not cross-comparable
    s = rx.embed("CC(N)C(=O)O", n=2).minimize()
    assert {e.energy_kind for e in s} == {"ff"}  # minimize tags FF energies
    with pytest.raises(ValueError, match="real energy"):
        s.best()
    assert {e.energy_kind for e in s.score("ff")} == {"ff"}  # an FF single point is still not "real"
    with pytest.raises(ValueError, match="real energy"):
        s.score("ff").best()


def test_a_derived_ensemble_keeps_the_kind_of_the_energies_it_carries():
    ens = rx.embed("CCCCO", n=4, seed=1).minimize()
    assert ens.energy_kind == "ff"
    assert ens.lowest(2).energy_kind == "ff"
    assert ens.align().energy_kind == "ff"
