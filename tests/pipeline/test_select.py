"""`pipeline/select.py`: the per-conformer latent, the clustering over it, and the dedup prunes."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms

from rxembed.pipeline import select


def _ens(smiles, n=6, **kw):
    import rxembed.pipeline as rx

    return rx.embed(smiles, n=n, seed=1, **kw).minimize()


def _two_blobs(n=12, gap=20.0):
    """Two well-separated point clouds: a latent whose clustering has one correct answer."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(0, 0.1, (n, 3)), rng.normal(gap, 0.1, (n, 3))])


# --- the latent: frame-free, and it says which blocks are live --------------------------------------------


def test_the_dihedral_features_are_invariant_to_rotation_and_translation():
    """The latent is compared across conformers in arbitrary frames, so a frame-dependent feature is a bug."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=1) == 0
    quads = select.rotatable_quads(mol)
    before = select.dihedrals(mol, 0, quads)

    conf = mol.GetConformer(0)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    conf.SetPositions(conf.GetPositions() @ rot.T + np.array([3.0, -1.0, 2.0]))
    assert select.dihedrals(mol, 0, quads) == pytest.approx(before, abs=1e-9)


def test_a_quad_is_only_ever_built_around_a_bond_that_actually_turns():
    """A ring bond turns nothing, and counting one invents conformational freedom the molecule does not have.

    Cyclohexane, not benzene: an aromatic bond never matches the single-bond pattern anyway, so only a
    SATURATED ring puts the not-in-ring qualifier under test.
    """
    assert select.rotatable_quads(Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))) == []

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1CCO"))  # a saturated ring welded to a real rotor chain
    quads = select.rotatable_quads(mol)
    assert quads
    for _a, b, c, _d in quads:
        bond = mol.GetBondBetweenAtoms(int(b), int(c))
        assert bond is not None, f"({b},{c}) is not even a bond"
        assert not bond.IsInRing(), f"({b},{c}) is a ring bond, not a rotor"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
@pytest.mark.parametrize(
    ("smiles", "kw", "kinds", "mode"),
    [
        ("CCCCO", {}, ["dihedral"], "conformer family"),
        ("CCCC.CCCC", {}, ["dihedral", "relpose"], "relative arrangement"),
        ("OC(=O)c1ccccc1.n1ccccc1", {"contacts": "auto"}, ["dihedral", "relpose", "nci"], "contact pattern"),
    ],
)
def test_the_live_latent_blocks_name_what_a_mode_means_here(smiles, kw, kinds, mode):
    """'mode' means something different per system, and the block that decides it is the most specific one live."""
    import rxembed.pipeline as rx

    ens = _ens(smiles, **kw)
    ens = ens[0] if isinstance(ens, rx.EnsembleSet) else ens
    assert select.active_feature_kinds(ens.mol, ens.ids) == kinds
    assert select.mode_kind(ens.mol, ens.ids) == mode


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_a_metal_suppresses_the_relpose_and_nci_blocks():
    """The surrogate strips M-L bonds, so a complex looks multi-fragment: its mode is the L-M-L block alone."""
    import rxembed.pipeline as rx

    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=3, seed=1).minimize()
    assert select.active_feature_kinds(ens.mol, ens.ids) == ["dihedral", "metal"]
    assert select.mode_kind(ens.mol, ens.ids) == "ligand arrangement"


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_the_metal_block_carries_the_real_l_m_l_angles():
    """The block's *presence* was pinned but never its values, so a latent of constants stayed green.

    A square-planar Pd sees six donor pairs: four cis near 90 deg and two trans near 180.
    """
    import rxembed.pipeline as rx

    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=2, seed=1).minimize()
    angles = sorted(select._metal_features(ens.mol, ens.ids)[0])
    assert len(angles) == 6, "four donors give six L-M-L pairs"
    assert angles[:4] == pytest.approx([90.0] * 4, abs=25.0)
    assert angles[4:] == pytest.approx([180.0] * 2, abs=25.0)


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_the_feature_matrix_has_one_finite_row_per_conformer():
    """A NaN from a degenerate dihedral does not raise: it silently poisons every distance in the clustering."""
    ens = _ens("CCCCO", n=4)
    feats = select.feature_matrix(ens.mol, ens.ids)
    assert feats.shape[0] == len(ens.ids)
    assert np.isfinite(feats).all()


# --- clustering -------------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_two_separated_blobs_never_share_a_cluster_label():
    """`leaf` selection splits a blob into sub-modes freely; what it may never do is merge two distant ones."""
    n = 12
    labels = select.cluster_on(_two_blobs(n), min_cluster=3)
    assert set(labels[:n]) & set(labels[n:]) <= {-1}


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_too_few_conformers_to_cluster_are_one_mode_not_all_noise():
    """Labelling a 2-conformer ensemble -1/-1 would make `representatives()` call every conformer rare."""
    assert list(select.cluster_on(np.zeros((2, 4)), min_cluster=3)) == [0, 0]


# --- the prunes -------------------------------------------------------------------------------------------


def test_energy_prune_keeps_one_frame_per_degenerate_band_and_compares_only_within_a_mode():
    """Two poses can be isoenergetic and still be different binding modes; labels gate the comparison."""
    assert list(select.energy_prune([0.0, 0.01, 0.5, 0.52, 1.0], energy_tol=0.05)) == [True, False, True, False, True]
    assert list(select.energy_prune([0.0, 0.0], labels=["A", "B"], energy_tol=0.05)) == [True, True]
    assert list(select.energy_prune([0.0, 0.0], energy_tol=0.05)) == [True, False]


def test_an_unknown_dedup_method_is_refused_and_names_the_alternatives():
    ens = _ens("CCCCO", n=2)
    with pytest.raises(ValueError, match="representatives"):
        select.apply(ens.mol, ens.ids, [ens.energies[i] for i in ens.ids], method="cluster")


def test_none_still_sorts_by_energy_because_every_caller_assumes_that_order():
    """The masks the other methods build are aligned to the energy sort, so 'none' must not be an exception."""
    ens = _ens("CCCCO", n=4)
    energies = [3.0, 1.0, 2.0, 0.0]
    ids, _ = select.apply(ens.mol, ens.ids, energies[: len(ens.ids)], method="none")
    assert ids == sorted(ens.ids, key=lambda i: energies[i])[: len(ids)]


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_rmsd_dedup_collapses_a_duplicated_conformer():
    """Two copies of one geometry are one conformer, and the prune must say which absorbed which."""
    ens = _ens("CCCCO", n=4)
    cid = ens.ids[0]
    dup = ens.mol.AddConformer(Chem.Conformer(ens.mol.GetConformer(cid)), assignId=True)
    ids, _ = select.apply(ens.mol, [*ens.ids, dup], [*[ens.energies[i] for i in ens.ids], ens.energies[cid]])
    assert dup not in ids or cid not in ids, "an exact duplicate survived the RMSD prune"
    assert select.nearest_kept(ens.mol, [cid], [dup])[dup] == (cid, 0.0)


# --- the prune verb on the Ensemble -----------------------------------------------------------------------


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_the_prune_verb_dedups_and_explains_what_it_merged():
    ens = _ens("OC(=O)CCCCc1ccccc1", n=10)
    before = len(ens.ids)
    ens.prune(by="rmsd", max_rmsd=2.5)  # deliberately coarse, so something is certain to merge
    assert len(ens.ids) < before, "a 2.5 A RMSD threshold merged nothing: the prune never ran"
    assert ens.discarded
    assert set(ens.duplicates()) <= set(ens.ids), "duplicates() must group the dropped under a KEPT conformer"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_the_cascade_drops_a_reacted_conformer_before_anything_can_be_merged_into_it():
    """`prune(by=['connectivity','rmsd'])` composes a filter and a dedup, and the order is the point: dedup
    first would let the reacted conformer absorb a good one, which the filter then throws away with it."""
    ens = _ens("OC(=O)CCCCc1ccccc1", n=10)
    broken = ens.ids[0]
    rdMolTransforms.SetBondLength(ens.mol.GetConformer(broken), 3, 4, 2.60)  # ~1.7x the C-C covalent sum
    before = len(ens.ids)

    ens.prune(by=["connectivity", "rmsd"], max_rmsd=2.5)
    assert broken in ens.reacted, "the validity filter never ran"
    assert broken not in ens.ids
    assert len(ens.ids) < before - 1, "the dedup never ran"
    assert set(ens.duplicates()) <= set(ens.ids), "a conformer was absorbed by one the filter then dropped"


def test_prune_refuses_a_misspelled_tuning_knob_instead_of_ignoring_it():
    """A silently-ignored `rmsd=0.1` would leave the caller believing they had tuned the threshold."""
    ens = _ens("CCCCO", n=2)
    with pytest.raises(TypeError, match="max_rmsd"):
        ens.prune(by="rmsd", rmsd=0.1)


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_moi_on_a_multifragment_system_warns_that_it_over_merges(caplog):
    ens = _ens("CCCC.CCCC", n=4)
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.prune(by="moi")
    assert any("over-merge" in r.getMessage() for r in caplog.records)
