"""Test shared coordinate, RDKit and QA helpers."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.utils import (
    assign_stereo_from_3d,
    bond_removal_mirrors,
    flat_ranks,
    repair_bond_stereo,
)


def _mol(smiles, seed=None):
    """A Mol with explicit Hs, carrying a conformer only when a seed is given."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if seed is not None:
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _without(mol, atom):
    """Delete `atom`, keeping any conformer: the bond surgery `repair_bond_stereo` cleans up after."""
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


# ---------------------------------------------------------------------------------------------------------
# coordinate math
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# conjugated_quartets: the one perception the QA gate and the FF cap must agree on
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# flat_ranks: charge and bond order removed before canonical ranking, so a delocalised -1 written on one
# arbitrary donor cannot rank it apart from its chemically equivalent partner
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)/C=C(\\C)[O-]",  # acac: same delocalised-oxygen artefact over a longer conjugated backbone
    ],
)
def test_delocalised_oxygens_rank_equal_despite_the_written_charge(smiles):
    mol = _mol(smiles)
    oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "O"]
    assert len(oxygens) == 2

    charged_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    assert charged_ranks[oxygens[0]] != charged_ranks[oxygens[1]], (
        "fixture no longer carries the artefact: the two oxygens already rank equal with charge intact"
    )

    ranks = flat_ranks(mol)
    assert ranks[oxygens[0]] == ranks[oxygens[1]]


# ---------------------------------------------------------------------------------------------------------
# repair_bond_stereo: the cleanup bond surgery needs
# ---------------------------------------------------------------------------------------------------------


def test_bond_removal_drops_orphaned_stereo_flag():
    work = _without(Chem.MolFromSmiles(r"C/C=C/Cl"), 3)  # no conformer: nothing to re-perceive from
    assert repair_bond_stereo(work) == 1
    for b in work.GetBonds():
        assert b.GetStereo() == Chem.BondStereo.STEREONONE, "a flagged bond kept fewer than two reference atoms"

    clean = _mol(r"C/C=C/C", seed=1)
    Chem.AssignStereochemistryFrom3D(clean)
    before = [(b.GetIdx(), b.GetStereo()) for b in clean.GetBonds()]
    assert repair_bond_stereo(clean) == 0, "nothing was orphaned, so nothing may be re-perceived"
    assert [(b.GetIdx(), b.GetStereo()) for b in clean.GetBonds()] == before


def test_geometry_rebases_orphaned_ez_flag():
    work = _without(_mol(r"C/C=C/Cl", seed=1), 3)  # the Cl was one of the two reference atoms
    assert repair_bond_stereo(work) == 1
    bond = work.GetBondBetweenAtoms(1, 2)
    assert bond.GetStereo() != Chem.BondStereo.STEREONONE, "the E/Z was discarded, not re-referenced"
    assert len(bond.GetStereoAtoms()) == 2
    assert 3 not in list(bond.GetStereoAtoms())


# ---------------------------------------------------------------------------------------------------------
# the chiral tag is a parity over the atom's own bond order (`bond_removal_mirrors` / `remove_bond`)
# ---------------------------------------------------------------------------------------------------------


def _names_the_hand(mol, centre):
    """Whether the tag at `centre` names its conformer's hand, in whatever bond order it has now.

    RDKit's own definition: the signed volume of the FIRST THREE bonds about the centre, negative for CW. It
    holds at degree 3 as at degree 4, the fourth reference (implicit H, lone pair, the centre itself) sitting
    last either way. Calibrated against `AssignStereochemistryFrom3D` by the first test below, not assumed.
    """
    atom = mol.GetAtomWithIdx(centre)
    nbrs = [b.GetOtherAtomIdx(centre) for b in atom.GetBonds()][:3]
    p = mol.GetConformer().GetPositions()
    vol = float(np.dot(np.cross(p[nbrs[0]] - p[centre], p[nbrs[1]] - p[centre]), p[nbrs[2]] - p[centre]))
    cw = Chem.ChiralType.CHI_TETRAHEDRAL_CW
    return atom.GetChiralTag() == (cw if vol < 0 else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


@pytest.mark.parametrize(
    ("smiles", "slot", "mirrors"),
    [
        ("F[P@](Cl)(Br)(I)F", 1, False),  # degree 5: the arithmetic would say odd, and is refuted there
    ],
    ids=["hypervalent"],
)
def test_parity_rule_is_bounded_to_degree_four(smiles, slot, mirrors):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    centre = mol.GetAtomWithIdx(1)
    partner = [b.GetOtherAtomIdx(1) for b in centre.GetBonds()][slot]
    assert bond_removal_mirrors(centre, partner) is mirrors


# ---------------------------------------------------------------------------------------------------------
# assign_stereo_from_3d: the writer half of the same rule
# ---------------------------------------------------------------------------------------------------------

# A sulfoxide S donating through a dative arrow: the one shape where RDKit's 3D writer both assigns a tag and
# reads a different bond order from every one of its readers. `->` puts the same bond LAST, the even control.
_DATIVE_S = "Cl[Pd](Cl)(Cl)<-[S@](=O)(C)CC"


def _sulfur(mol):
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")


def test_rdkit_3d_writer_emits_unreadable_bond_order():
    mol = _mol(_DATIVE_S, seed=0xF00D)
    centre = _sulfur(mol)
    Chem.AssignStereochemistryFrom3D(mol)  # deliberately the RAW call: this test is about what it does
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert not _names_the_hand(mol, centre), "the writer already agrees with its readers; the premise is gone"


@pytest.mark.parametrize("smiles", [_DATIVE_S], ids=["dative-first"])
def test_3d_stereo_matches_reader_bond_order(smiles):
    mol = _mol(smiles, seed=0xF00D)
    centre = _sulfur(mol)
    assign_stereo_from_3d(mol)
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert _names_the_hand(mol, centre), "the tag names the mirror of the geometry it was written from"
