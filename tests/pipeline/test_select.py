"""`pipeline/select.py`: the per-conformer latent, the clustering over it, and the dedup prunes."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rxembed.pipeline import select


def _ens(smiles, n=6, **kw):
    import rxembed as rx

    return rx.embed(smiles, n=n, seed=1, **kw).minimize()


# --- the latent: which blocks are live, and what they carry -----------------------------------------------


def test_quad_requires_rotatable_bond():
    assert select.rotatable_quads(Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))) == []

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1CCO"))  # a saturated ring welded to a real rotor chain
    quads = select.rotatable_quads(mol)
    assert quads
    for _a, b, c, _d in quads:
        bond = mol.GetBondBetweenAtoms(int(b), int(c))
        assert bond is not None, f"({b},{c}) is not even a bond"
        assert not bond.IsInRing(), f"({b},{c}) is a ring bond, not a rotor"


def test_butanol_dihedral_latent_excludes_the_methyl_and_hydroxyl_rotors():
    """1-butanol (explicit H) has 2 backbone torsions, C0-C1-C2-C3 and C1-C2-C3-O4.

    The terminal methyl (C0) and hydroxyl (O4) each have one heavy neighbour, so rotating around their bond
    to the chain is not a distinct heavy-atom conformer; a heavy-atom-blind bond count would add both.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    quads = select.rotatable_quads(mol)
    bonds = {tuple(sorted((b, c))) for _a, b, c, _d in quads}
    assert bonds == {(1, 2), (2, 3)}


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("smiles", "kw", "kinds", "mode"),
    [
        ("CCCCO", {}, ["dihedral"], "conformer family"),
        ("CCCC.CCCC", {}, ["dihedral", "relpose"], "relative arrangement"),
        ("OC(=O)c1ccccc1.n1ccccc1", {"contacts": "auto"}, ["dihedral", "relpose", "nci"], "contact pattern"),
    ],
    ids=["conformer", "relative", "contact"],
)
def test_active_features_define_mode_kind(smiles, kw, kinds, mode):
    import rxembed as rx

    ens = _ens(smiles, **kw)
    ens = ens[0] if isinstance(ens, rx.EnsembleSet) else ens
    assert select.active_feature_kinds(ens.mol, ens.ids) == kinds
    assert select.mode_kind(ens.mol, ens.ids) == mode


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_metal_latent_suppresses_other_blocks():
    import rxembed as rx

    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=3, seed=1).minimize()
    assert select.active_feature_kinds(ens.mol, ens.ids) == ["dihedral", "metal"]
    assert select.mode_kind(ens.mol, ens.ids) == "ligand arrangement"

    angles = sorted(select._metal_features(ens.mol, ens.ids)[0])
    assert len(angles) == 6, "four donors give six L-M-L pairs"
    assert angles[:4] == pytest.approx([90.0] * 4, abs=25.0)
    assert angles[4:] == pytest.approx([180.0] * 2, abs=25.0)


def test_metal_latent_uses_declared_sphere():
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


def test_only_bondless_metal_uses_geometric_sphere():
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


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_small_samples_form_one_cluster():
    assert list(select.cluster_on(np.zeros((2, 4)), min_cluster=3)) == [0, 0]


# --- the prunes -------------------------------------------------------------------------------------------


def test_energy_prune_keeps_one_per_band_and_mode():
    assert list(select.energy_prune([0.0, 0.01, 0.5, 0.52, 1.0], energy_tol=0.05)) == [True, False, True, False, True]
    assert list(select.energy_prune([0.0, 0.0], labels=["A", "B"], energy_tol=0.05)) == [True, True]
    assert list(select.energy_prune([0.0, 0.0], energy_tol=0.05)) == [True, False]


def test_unknown_dedup_method_names_choices():
    ens = _ens("CCCCO", n=2)
    with pytest.raises(ValueError, match="representatives"):
        select.apply(ens.mol, ens.ids, [ens.energies[i] for i in ens.ids], method="cluster")


def test_none_mode_still_sorts_by_energy():
    ens = _ens("CCCCO", n=4)
    energies = [3.0, 1.0, 2.0, 0.0]
    ids, _ = select.apply(ens.mol, ens.ids, energies[: len(ens.ids)], method="none")
    assert ids == sorted(ens.ids, key=lambda i: energies[i])[: len(ids)]


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_rmsd_dedup_collapses_a_duplicated_conformer():
    ens = _ens("CCCCO", n=4)
    cid = ens.ids[0]
    dup = ens._mol.AddConformer(Chem.Conformer(ens._mol.GetConformer(cid)), assignId=True)
    ids, _ = select.apply(ens._mol, [*ens.ids, dup], [*[ens.energies[i] for i in ens.ids], ens.energies[cid]])
    assert dup not in ids or cid not in ids, "an exact duplicate survived the RMSD prune"
    assert select.nearest_kept(ens._mol, [cid], [dup])[dup] == (cid, 0.0)


# --- the prune verb on the Ensemble -----------------------------------------------------------------------


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_prune_verb_dedups_and_explains_what_it_merged():
    ens = _ens("OC(=O)CCCCc1ccccc1", n=10)
    before = len(ens.ids)
    ens.prune(by="rmsd", max_rmsd=2.5)  # deliberately coarse, so something is certain to merge
    assert len(ens.ids) < before, "a 2.5 A RMSD threshold merged nothing: the prune never ran"
    assert ens.discarded
    assert set(ens.duplicates()) <= set(ens.ids), "duplicates() must group the dropped under a KEPT conformer"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_cascade_drops_reacted_before_dedup():
    ens = _ens("OC(=O)CCCCc1ccccc1", n=10)
    broken = ens.ids[0]
    rdMolTransforms.SetBondLength(ens._mol.GetConformer(broken), 3, 4, 2.60)  # ~1.7x the C-C covalent sum
    before = len(ens.ids)

    ens.prune(by=["connectivity", "rmsd"], max_rmsd=2.5)
    assert broken in ens.reacted, "the validity filter never ran"
    assert broken not in ens.ids
    assert len(ens.ids) < before - 1, "the dedup never ran"
    assert set(ens.duplicates()) <= set(ens.ids), "a conformer was absorbed by one the filter then dropped"


def test_prune_rejects_unknown_tuning_kwarg():
    ens = _ens("CCCCO", n=2)
    with pytest.raises(TypeError, match="rmsd"):
        ens.prune(by="rmsd", rmsd=0.1)


@pytest.mark.skipif(find_spec("prism_pruner") is None or find_spec("sklearn") is None, reason="needs rxembed[workflow]")
def test_multifragment_moi_warns_overmerge(caplog):
    ens = _ens("CCCC.CCCC", n=4)
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.prune(by="moi")
    assert any("over-merge" in r.getMessage() for r in caplog.records)
