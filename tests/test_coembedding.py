"""Test paired reaction endpoints through the public coembedding API."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.stereo import defined_stereo_label, stereo_from_3d
from tests.conftest import EXAMPLES_DIR


def _bromobutane_and_chlorobutane():
    """Return 1-bromobutane + Cl- and 1-chlorobutane + Br- on one atom order: halides at 4 and 5, then the Hs."""
    reactant = Chem.AddHs(Chem.MolFromSmiles("CCCCBr.[Cl-]"))
    product = Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]"))
    order = [0, 1, 2, 3, 5, 4, *range(6, product.GetNumAtoms())]
    return reactant, Chem.RenumberAtoms(product, order)


def test_sn2_copies_keep_their_own_bonds_and_overlay_the_spectator_chain():
    reactant, product = _bromobutane_and_chlorobutane()

    result = rx.coembed(reactant, product, n=4, seed=3)
    left, right = result.reactant, result.product

    assert left.ids == right.ids
    assert 0 < len(left.ids) <= 4
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[3] - r[4]) < 2.1  # C-Br bonded
        assert np.linalg.norm(r[3] - r[5]) > 2.6  # Cl- free
        assert np.linalg.norm(p[3] - p[5]) < 1.9  # C-Cl bonded
        assert np.linalg.norm(p[3] - p[4]) > 2.6  # Br- free
        assert np.linalg.norm(r[:2] - p[:2], axis=1).max() < 0.5  # the far end of the chain: one shared pose


def test_constrain_holds_the_chloride_beside_the_chain_end_in_both_copies():
    reactant, product = _bromobutane_and_chlorobutane()

    result = rx.coembed(reactant, product, constrain={(0, 5): (3.0, 3.3)}, n=3, seed=3)
    left, right = result.reactant, result.product

    assert result.ids
    for conformers in (left, right):  # free Cl- in the reactant, bonded Cl in the product
        for cid in conformers.ids:
            pos = conformers.mol.GetConformer(cid).GetPositions()
            assert 2.95 < np.linalg.norm(pos[0] - pos[5]) < 3.35


def test_metal_copies_keep_their_own_coordination_sphere():
    reactant = Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[NH3:3].[Cl-:4]"))  # N0 Cu1 N2 Cl3
    product = Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[Cl-:4].[NH3:3]"))  # N0 Cu1 Cl2 N3
    product = Chem.RenumberAtoms(product, [0, 1, 3, 2, *range(4, product.GetNumAtoms())])

    result = rx.coembed(rx.metal(reactant, "linear")[0], rx.metal(product, "linear")[0], n=2, seed=1)
    left, right = result.reactant, result.product

    assert left.ids == right.ids
    assert 0 < len(left.ids) <= 2
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[1] - r[2]) < 2.2  # Cu-N bonded
        assert np.linalg.norm(r[1] - r[3]) > 2.6  # Cl- free
        assert np.linalg.norm(p[1] - p[3]) < 2.4  # Cu-Cl bonded
        assert np.linalg.norm(p[1] - p[2]) > 2.6  # NH3 free


def test_hexammine_potassium_co_embeds_where_every_eigenvector_start_is_rejected():
    hexammine = Chem.AddHs(rx.parse_smiles("N->[K+](<-N)(<-N)(<-N)(<-N)<-N"))  # K sits at its donors' centroid

    left = rx.coembed(hexammine, Chem.Mol(hexammine), n=2, seed=7).reactant

    assert len(left.ids) == 2


@pytest.mark.parametrize("smiles", ["C[C@H](O)CC", "Cc1cccc(O)c1-c1c(C)cccc1O |wU:7.6|"], ids=["centre", "axis"])
def test_copies_that_differ_only_in_configuration_each_keep_their_own(smiles):
    left = Chem.AddHs(Chem.MolFromSmiles(smiles))
    right = Chem.AddHs(Chem.MolFromSmiles(smiles.replace("@", "@@").replace("wU", "wD")))

    result = rx.coembed(left, right, n=4, seed=1)
    copies = result.reactant, result.product

    assert 0 < len(copies[0].ids) <= 4
    for conformers, mol in zip(copies, (left, right), strict=True):
        for cid in conformers.ids:
            assert stereo_from_3d(Chem.Mol(conformers.mol, confId=cid)) == defined_stereo_label(mol)


def test_metal_copies_that_differ_only_in_a_bound_amine_hand_co_embed():
    left, right = (
        rx.metal(Chem.AddHs(rx.parse_smiles(f"[Pd](Cl)(Cl)(Cl)([N{tag}H](C)O)")), "SPL")[0] for tag in ("@", "@@")
    )

    assert 0 < len(rx.coembed(left, right, n=4, seed=1)) <= 4


def test_inverting_sn2_copies_overlay_the_spectator_carbons():
    reactant = Chem.AddHs(Chem.MolFromSmiles("C[C@H](Br)CC.[Cl-]"))
    product = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](Cl)CC.[Br-]"))
    product = Chem.RenumberAtoms(product, [0, 1, 5, 3, 4, 2, *range(6, product.GetNumAtoms())])

    result = rx.coembed(reactant, product, n=4, seed=1)
    left, right = result.reactant, result.product

    assert left.ids
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[[0, 3, 4]] - p[[0, 3, 4]], axis=1).max() < 0.5


def test_sn2_endpoints_can_use_different_atom_orders():
    reactant, _product = _bromobutane_and_chlorobutane()
    product = Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]"))

    result = rx.coembed(reactant, product, n=1, seed=3)

    assert result.ids
    assert len(result.correspondence) == reactant.GetNumAtoms()
    assert result.reactant.mol.GetAtomWithIdx(4).GetSymbol() == "Br"
    assert result.product.mol.GetAtomWithIdx(4).GetSymbol() == "Cl"
    assert dict(result.correspondence)[4] == 5


def test_nazarov_xyz_endpoints_keep_their_input_atom_order():
    paths = [str(EXAMPLES_DIR / f"nazarov_{side}.xyz") for side in ("start", "end")]
    originals = [Chem.MolFromXYZFile(path) for path in paths]

    result = rx.coembed(*paths, n=1, seed=1, stereo="free")

    assert len(result) == 1
    assert result.correspondence == tuple((i, i) for i in range(originals[0].GetNumAtoms()))
    for endpoint, original in zip((result.reactant, result.product), originals, strict=True):
        assert [atom.GetSymbol() for atom in endpoint.mol.GetAtoms()] == [
            atom.GetSymbol() for atom in original.GetAtoms()
        ]
        assert np.all(np.isfinite(endpoint.mol.GetConformer(result.ids[0]).GetPositions()))
    assert np.isfinite(result.alignment_scores[result.ids[0]])
    assert np.isfinite(result.ts_rmsds[result.ids[0]])
