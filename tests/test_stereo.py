"""Undefined-stereocentre enumeration — the deliberate racemate/diastereomer load-in on the embed front.

A coordinate-free input (SMILES) with *unlabeled* stereocentres embeds every stereoisomer instead of one
arbitrary hand: ``stereo='auto'`` folds them into one `EnsembleSet`, ``stereo='enumerate'`` keeps them
separate, ``stereo='free'`` opts out. Point R/S + double-bond E/Z, defined centres held fixed, meso dropped,
chiral-at-P included, and it composes with the metal coordination-isomer axis. Pure RDKit — no xtb.
"""

from rdkit import Chem

import rxembed as rx


def _configs(es):
    return sorted(e.tag.get("stereo") for e in es)


def test_undefined_centre_auto_embeds_racemate_as_one_set():
    r = rx.embed("CC(N)C(=O)O", n=2)  # undefined alpha-carbon
    assert isinstance(r, rx.EnsembleSet)
    assert _configs(r) == ["1R", "1S"]  # both enantiomers, index-keyed CIP tags


def test_defined_centre_is_untouched():
    r = rx.embed("C[C@H](N)C(=O)O", n=2)  # a labelled centre -> the single, kept configuration
    assert isinstance(r, rx.Ensemble)


def test_no_stereocentre_is_a_single_ensemble():
    assert isinstance(rx.embed("CCO", n=2), rx.Ensemble)


def test_enumerate_keeps_stereoisomers_separate_and_uniform():
    r = rx.embed("CC(N)C(=O)O", n=2, stereo="enumerate")
    assert isinstance(r, list)
    assert len(r) == 2
    assert all(isinstance(g, rx.EnsembleSet) for g in r)  # uniform type, even for a lone organic variant
    assert sorted(g[0].tag["stereo"] for g in r) == ["1R", "1S"]


def test_racemate_ensembleset_is_chainable():
    # the headline chain must work on a racemate EnsembleSet, mapping over each stereoisomer (tags carried)
    r = rx.embed("CC(N)C(=O)O", n=3).minimize().prune()
    assert isinstance(r, rx.EnsembleSet)
    assert {e.tag["stereo"] for e in r} == {"1R", "1S"}  # both enantiomers survive the mapped chain
    for e in r:
        assert e.n >= 1
    assert isinstance(rx.embed("CCO", n=2).minimize().prune(), rx.Ensemble)  # a stereo-free SMILES stays single


def test_racemate_ensembleset_dumps_one_xyz_per_stereoisomer(tmp_path):
    paths = rx.embed("CC(N)C(=O)O", n=2).minimize().dump(str(tmp_path / "amac.xyz"))
    assert sorted(p.rsplit("_", 1)[1] for p in paths) == ["1R.xyz", "1S.xyz"]  # tag folded into each filename
    for p in paths:
        assert int(open(p).readline().strip()) == 13  # a valid .xyz (atom count header)


def test_best_refuses_ff_energies_across_species():
    # ranking distinct species needs REAL energies; FF (minimize / score('ff')) is not cross-comparable -> refused
    import pytest

    s = rx.embed("CC(N)C(=O)O", n=2).minimize()
    assert {e.energy_kind for e in s} == {"ff"}  # minimize tags FF energies
    with pytest.raises(ValueError, match="REAL energy"):
        s.best()
    assert {e.energy_kind for e in s.score("ff")} == {"ff"}  # an FF single point is still not "real"
    with pytest.raises(ValueError, match="REAL energy"):
        s.score("ff").best()


def test_free_opts_out_of_enumeration():
    assert isinstance(rx.embed("CC(N)C(=O)O", n=2, stereo="free"), rx.Ensemble)


def test_double_bond_ez_is_enumerated():
    r = rx.embed("CC=CC(N)O", n=2)  # one undefined C + one undefined C=C -> 4
    cfgs = _configs(r)
    assert len(cfgs) == 4
    assert any(":E" in c for c in cfgs)
    assert any(":Z" in c for c in cfgs)


def test_meso_duplicate_is_dropped():
    r = rx.embed("CC(O)C(O)C", n=2)  # 2 centres, but the meso pair collapses -> 3, not 4
    assert isinstance(r, rx.EnsembleSet)
    assert len(r) == 3


def test_embedded_3d_handedness_matches_the_tag():
    r = rx.embed("CC(N)C(=O)O", n=3)
    for e in r:
        em = e.minimize()  # the raw ETKDG seed can carry a conjugation twist; minimise, then read the hand
        Chem.AssignStereochemistryFrom3D(em.mol, confId=em.ids[0])
        ((idx, code),) = Chem.FindMolChiralCenters(em.mol, useLegacyImplementation=False)
        assert e.tag["stereo"] == f"{idx}{code}"  # the label is the geometry's actual configuration


def test_composes_with_metal_coordination_isomers():
    # an aminoacidate on Pd: 2 ligand enantiomers x the square-planar coordination isomers
    r = rx.embed("CC(N)C(=O)[O-]->[Pd]([Cl])[Cl]", metal="square_planar", n=2)
    assert isinstance(r, rx.EnsembleSet)
    assert {"1R", "1S"} == {e.tag["stereo"] for e in r}  # both enantiomers present
    assert {"cis", "trans"} <= {e.tag["label"] for e in r}  # each coordination isomer too
    for e in r:  # every candidate carries BOTH axes
        assert e.tag.get("stereo")
        assert e.tag.get("label")


def test_chiral_at_metal_bound_phosphorus_is_enumerated():
    r = rx.embed("C[P](CC)(c1ccccc1)[Pd](Cl)(Cl)Cl", metal="square_planar", n=2)
    assert isinstance(r, rx.EnsembleSet)
    assert len({e.tag["stereo"] for e in r}) == 2  # the two P-epimers, distinct after the sphere strip


def _chirality_volume(mol, cid, centre):
    import numpy as np

    conf = mol.GetConformer(cid)
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()][:3]
    p = np.array([list(conf.GetAtomPosition(i)) for i in [centre, *nbrs]])
    return float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))


def test_metal_bound_carbanion_donor_embeds_distinct_hands():
    # a carbanion-C donor is a stereocentre only WHILE metal-bound; the embed holds it (neutralise + dummy-D)
    # so the two enumerated hands relax to OPPOSITE geometries, not the same one.
    import numpy as np

    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    en = rx.metal(smi, "square_planar")
    assert {i.stereo_label for i in en} == {"23R", "23S"}  # metal-priority CIP labels
    hands = []
    for iso in en:
        e = rx.embed(iso, n=4).minimize()
        assert e.n >= 1
        assert e.mol.GetNumAtoms() == 65  # the hold's dummy deuterium is removed after embed + relax
        assert not any(a.GetIsotope() == 2 for a in e.mol.GetAtoms())  # no leaked D
        assert any(a.GetSymbol() == "Ni" for a in e.mol.GetAtoms())  # the surrogate is switched back to the metal
        signs = {np.sign(_chirality_volume(e.mol, i, 23)) for i in e.ids}
        assert len(signs) == 1  # EVERY conformer of this isomer has the SAME donor hand (held through the relax)
        hands.append(signs.pop())
    assert len(set(hands)) == 2  # and the two isomers are opposite enantiomers of the carbanion donor


def test_donor_charge_is_restored_after_the_hold():
    # the hold NEUTRALISES the carbanion during the embed (a -1 C can't take a 4th bond) then RESTORES it —
    # a leaked neutralisation would corrupt the donor charge (and the xtb charge downstream).
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    for iso in rx.metal(smi, "square_planar"):
        e = rx.embed(iso, n=1)
        assert e.mol.GetAtomWithIdx(23).GetFormalCharge() == -1  # the carbanion is back to -1 after the hold


def test_allene_axis_stays_a_bare_chainable_ensemble():
    # RDKit can't enumerate allene/cumulene axial chirality from a flat SMILES -> one arbitrary hand, NOT an
    # EnsembleSet-of-1 (that would break the documented rx.embed(smi).mc().prune() chain), and no '?' tag.
    r = rx.embed("CC(F)=C=C(F)C", n=2)
    assert isinstance(r, rx.Ensemble)
    assert hasattr(r, "mc")
    assert "?" not in (r.tag.get("stereo") or "")


def test_infeasible_or_dead_stereoisomer_is_skipped_not_kept():
    # trans-cyclooctene is too strained for ETKDG (0 conformers); only the embeddable Z survives, and no
    # dead 0-conformer candidate is kept in the result.
    r = rx.embed("C1CCC=CCCC1", n=6)
    for e in [r] if isinstance(r, rx.Ensemble) else r:
        assert e.n >= 1
