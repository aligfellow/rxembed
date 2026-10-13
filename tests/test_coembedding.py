"""Test co-embedding: two graphs on one atom order embedded as tied copies."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.coembedding import coembed


def _bromobutane_and_chlorobutane():
    """Return 1-bromobutane + Cl- and 1-chlorobutane + Br- on one atom order: halides at 4 and 5, then the Hs."""
    reactant = Chem.AddHs(Chem.MolFromSmiles("CCCCBr.[Cl-]"))
    product = Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]"))
    order = [0, 1, 2, 3, 5, 4, *range(6, product.GetNumAtoms())]
    return reactant, Chem.RenumberAtoms(product, order)


def test_sn2_copies_keep_their_own_bonds_and_overlay_the_spectator_chain():
    reactant, product = _bromobutane_and_chlorobutane()

    left, right = coembed(reactant, product, n=4, seed=3)

    assert left.ids == right.ids
    assert len(left.ids) == 4
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[3] - r[4]) < 2.1  # C-Br bonded
        assert np.linalg.norm(r[3] - r[5]) > 2.6  # Cl- free
        assert np.linalg.norm(p[3] - p[5]) < 1.9  # C-Cl bonded
        assert np.linalg.norm(p[3] - p[4]) > 2.6  # Br- free
        assert np.linalg.norm(r[:2] - p[:2], axis=1).max() < 0.5  # the far end of the chain: one shared pose


def test_constrain_holds_the_chloride_beside_the_chain_end_in_both_copies():
    reactant, product = _bromobutane_and_chlorobutane()

    left, right = coembed(reactant, product, constrain={(0, 5): (3.0, 3.3)}, n=3, seed=3)

    for conformers in (left, right):  # free Cl- in the reactant, bonded Cl in the product
        for cid in conformers.ids:
            pos = conformers.mol.GetConformer(cid).GetPositions()
            assert 2.95 < np.linalg.norm(pos[0] - pos[5]) < 3.35


def test_metal_copies_keep_their_own_coordination_sphere():
    reactant = Chem.AddHs(rx.parse_smiles("[NH3]->[Cu+]<-[NH3].[Cl-]"))  # N0 Cu1 N2 Cl3
    product = Chem.AddHs(rx.parse_smiles("[NH3]->[Cu+]<-[Cl-].[NH3]"))  # N0 Cu1 Cl2 N3
    product = Chem.RenumberAtoms(product, [0, 1, 3, 2, *range(4, product.GetNumAtoms())])

    left, right = coembed(rx.metal(reactant, "linear")[0], rx.metal(product, "linear")[0], n=2, seed=1)

    assert left.ids == right.ids
    assert len(left.ids) == 2
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[1] - r[2]) < 2.2  # Cu-N bonded
        assert np.linalg.norm(r[1] - r[3]) > 2.6  # Cl- free
        assert np.linalg.norm(p[1] - p[3]) < 2.4  # Cu-Cl bonded
        assert np.linalg.norm(p[1] - p[2]) > 2.6  # NH3 free


def test_hexammine_potassium_co_embeds_where_every_eigenvector_start_is_rejected():
    hexammine = Chem.AddHs(rx.parse_smiles("N->[K+](<-N)(<-N)(<-N)(<-N)<-N"))  # K sits at its donors' centroid

    left, _right = coembed(hexammine, Chem.Mol(hexammine), n=2, seed=7)

    assert len(left.ids) == 2


def test_copies_must_share_one_atom_order():
    reactant, _product = _bromobutane_and_chlorobutane()

    with pytest.raises(ValueError, match="element order"):
        coembed(reactant, Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]")))
