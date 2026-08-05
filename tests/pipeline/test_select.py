"""`pipeline/select.py`: the per-conformer latent, the clustering over it, and the dedup prunes."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rxembed.pipeline import select


def _ens(smiles, n=6, **kw):
    import rxembed.pipeline as rx

    return rx.embed(smiles, n=n, seed=1, **kw).minimize()


# --- the latent: which blocks are live, and what they carry -----------------------------------------------


def test_a_quad_is_only_ever_built_around_a_bond_that_actually_turns():
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
    import rxembed.pipeline as rx

    ens = _ens(smiles, **kw)
    ens = ens[0] if isinstance(ens, rx.EnsembleSet) else ens
    assert select.active_feature_kinds(ens.mol, ens.ids) == kinds
    assert select.mode_kind(ens.mol, ens.ids) == mode


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_a_metal_suppresses_the_relpose_and_nci_blocks_and_carries_real_l_m_l_angles():
    import rxembed.pipeline as rx

    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=3, seed=1).minimize()
    assert select.active_feature_kinds(ens.mol, ens.ids) == ["dihedral", "metal"]
    assert select.mode_kind(ens.mol, ens.ids) == "ligand arrangement"

    angles = sorted(select._metal_features(ens.mol, ens.ids)[0])
    assert len(angles) == 6, "four donors give six L-M-L pairs"
    assert angles[:4] == pytest.approx([90.0] * 4, abs=25.0)
    assert angles[4:] == pytest.approx([180.0] * 2, abs=25.0)


def test_the_metal_latent_reads_the_declared_sphere_not_an_absolute_cutoff():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(57))
    donors = [rw.AddAtom(Chem.Atom(34)) for _ in range(2)]
    near_non_donor = rw.AddAtom(Chem.Atom(8))
    for donor in donors:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (-3.0, 0.0, 0.0), (0.0, 2.6, 0.0))):
        conf.SetAtomPosition(atom, xyz)
    cid = mol.AddConformer(conf, assignId=True)
    stretched = Chem.Conformer(conf)
    stretched.SetAtomPosition(donors[0], (5.0, 0.0, 0.0))
    stretched_cid = mol.AddConformer(stretched, assignId=True)

    assert select._metal_donors(mol, [cid]) == (metal, donors)
    assert select._metal_donors(mol, [stretched_cid, cid]) == (metal, donors), (
        "reordering conformers changed the descriptor's declared donor columns"
    )
    assert select._metal_features(mol, [cid]).shape == (1, 1), (
        f"long La-Se donors were missed or nearby O{near_non_donor} was mistaken for one"
    )


def test_only_a_fully_bondless_metal_uses_geometric_sphere_perception():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(57))
    donors = [rw.AddAtom(Chem.Atom(34)) for _ in range(2)]
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (-3.0, 0.0, 0.0))):
        conf.SetAtomPosition(atom, xyz)
    cid = mol.AddConformer(conf, assignId=True)
    assert select._metal_donors(mol, [cid]) == (metal, donors)

    rw = Chem.RWMol(mol)
    other_metal = rw.AddAtom(Chem.Atom(26))
    rw.AddBond(metal, other_metal, Chem.BondType.SINGLE)
    bonded = rw.GetMol()
    bonded_conf = Chem.Conformer(conf)
    bonded_conf.SetAtomPosition(other_metal, (2.5, 2.5, 0.0))
    bonded.RemoveAllConformers()
    bonded_cid = bonded.AddConformer(bonded_conf, assignId=True)
    assert select._metal_donors(bonded, [bonded_cid]) == (metal, []), (
        "an M-M bond is a declared graph, not permission to guess nearby ligand donors"
    )


# --- clustering -------------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_too_few_conformers_to_cluster_are_one_mode_not_all_noise():
    assert list(select.cluster_on(np.zeros((2, 4)), min_cluster=3)) == [0, 0]


# --- the prunes -------------------------------------------------------------------------------------------


def test_energy_prune_keeps_one_frame_per_degenerate_band_and_compares_only_within_a_mode():
    assert list(select.energy_prune([0.0, 0.01, 0.5, 0.52, 1.0], energy_tol=0.05)) == [True, False, True, False, True]
    assert list(select.energy_prune([0.0, 0.0], labels=["A", "B"], energy_tol=0.05)) == [True, True]
    assert list(select.energy_prune([0.0, 0.0], energy_tol=0.05)) == [True, False]


def test_an_unknown_dedup_method_is_refused_and_names_the_alternatives():
    ens = _ens("CCCCO", n=2)
    with pytest.raises(ValueError, match="representatives"):
        select.apply(ens.mol, ens.ids, [ens.energies[i] for i in ens.ids], method="cluster")


def test_none_still_sorts_by_energy_because_every_caller_assumes_that_order():
    ens = _ens("CCCCO", n=4)
    energies = [3.0, 1.0, 2.0, 0.0]
    ids, _ = select.apply(ens.mol, ens.ids, energies[: len(ens.ids)], method="none")
    assert ids == sorted(ens.ids, key=lambda i: energies[i])[: len(ids)]


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_rmsd_dedup_collapses_a_duplicated_conformer():
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
    ens = _ens("CCCCO", n=2)
    with pytest.raises(TypeError, match="rmsd"):
        ens.prune(by="rmsd", rmsd=0.1)


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[select]")
def test_moi_on_a_multifragment_system_warns_that_it_over_merges(caplog):
    ens = _ens("CCCC.CCCC", n=4)
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.prune(by="moi")
    assert any("over-merge" in r.getMessage() for r in caplog.records)
