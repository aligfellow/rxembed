"""Test shared coordinate, RDKit and QA helpers."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.utils import (
    assign_stereo_from_3d,
    bond_angle,
    bond_removal_mirrors,
    conjugated_quartets,
    dihedral_angle,
    flat_ranks,
    remove_bond,
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


def test_angle_and_signed_dihedral_conventions():
    o, x, y = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert bond_angle(x, o, y) == pytest.approx(90.0)
    assert bond_angle(x, o, -x) == pytest.approx(180.0)
    with np.errstate(all="raise"):
        assert np.isnan(bond_angle(o, o, x))

    p0, p1, p2 = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    plus, minus = np.array([0, 1.0, 1.0]), np.array([0, -1.0, 1.0])
    assert dihedral_angle(p0, p1, p2, plus) == pytest.approx(-dihedral_angle(p0, p1, p2, minus))
    assert dihedral_angle(p0, p1, p2, plus) != 0.0


# ---------------------------------------------------------------------------------------------------------
# conjugated_quartets: the one perception the QA gate and the FF cap must agree on
# ---------------------------------------------------------------------------------------------------------


def test_conjugation_detects_amide_not_saturated_chain():
    quartets = list(conjugated_quartets(_mol("CC(=O)NC")))
    assert quartets, "the amide plane was not perceived"
    for a, c, x, s in quartets:
        assert len({a, c, x, s}) == 4, "a quartet must name four distinct atoms"
    assert list(conjugated_quartets(_mol("CCCC"))) == []


def test_exclude_drops_a_quartet_naming_an_excluded_atom():
    mol = _mol("CC(=O)NC")
    hit = next(iter(conjugated_quartets(mol)))
    assert hit not in list(conjugated_quartets(mol, exclude={hit[2]}))


def test_dative_metal_is_not_an_organic_conjugation_substituent():
    assert list(conjugated_quartets(_mol("CC(=O)[O-]->[Zn+]"))) == []
    assert list(conjugated_quartets(_mol("CC(=O)OC"))), "an ordinary ester substituent must still be judged"


# ---------------------------------------------------------------------------------------------------------
# flat_ranks: charge and bond order removed before canonical ranking, so a delocalised -1 written on one
# arbitrary donor cannot rank it apart from its chemically equivalent partner
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)[O-]",  # acetate: the -1 sits on one written oxygen, the other carries the C=O
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


def test_bond_repair_leaves_native_atropisomer_stereo_alone():
    mol = Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|")
    axis = next(bond for bond in mol.GetBonds() if bond.GetStereo() == Chem.BondStereo.STEREOATROPCCW)

    assert repair_bond_stereo(mol) == 0
    assert axis.GetStereo() == Chem.BondStereo.STEREOATROPCCW


# ---------------------------------------------------------------------------------------------------------
# the chiral tag is a parity over the atom's own bond order (`bond_removal_mirrors` / `remove_bond`)
# ---------------------------------------------------------------------------------------------------------

_HALIDE_C = "F[C@](Cl)(Br)I"  # one tagged degree-4 centre whose four bonds are all distinguishable


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


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_bond_removal_preserves_geometry_tag(slot):
    mol = _mol(_HALIDE_C, seed=11)
    Chem.AssignStereochemistryFrom3D(mol)  # calibrate: RDKit's own writer, on this very conformer
    assert _names_the_hand(mol, 1), "the sign convention this test refereeds by is wrong"
    rw = Chem.RWMol(mol)
    remove_bond(rw, 1, [b.GetOtherAtomIdx(1) for b in mol.GetAtomWithIdx(1).GetBonds()][slot])
    assert _names_the_hand(rw.GetMol(), 1), f"slot {slot}: the tag now names the mirror of its own geometry"


def test_adding_a_bond_back_needs_no_counterpart():
    mol = _mol(_HALIDE_C, seed=11)
    Chem.AssignStereochemistryFrom3D(mol)
    partner = next(b.GetOtherAtomIdx(1) for b in mol.GetAtomWithIdx(1).GetBonds())  # slot 0: an odd one
    rw = Chem.RWMol(mol)
    remove_bond(rw, 1, partner)
    rw.AddBond(1, partner, Chem.BondType.SINGLE)
    assert _names_the_hand(rw.GetMol(), 1), "the re-added bond needed a second correction, so it is not last"


@pytest.mark.parametrize(
    ("smiles", "slot", "mirrors"),
    [
        ("F[P@](Cl)(Br)(I)F", 1, False),  # degree 5: the arithmetic would say odd, and is refuted there
        ("F[C@](Cl)Br", 1, False),  # degree 3: no representable tag survives, so the caller clears it
    ],
    ids=["hypervalent", "trigonal"],
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
_DATIVE_S_LAST = "[S@](=O)(C)(CC)->[Pd](Cl)(Cl)Cl"
_COVALENT_S = "Cl[Pd](Cl)(Cl)[S@](=O)(C)CC"


def _sulfur(mol):
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")


def test_rdkit_3d_writer_emits_unreadable_bond_order():
    mol = _mol(_DATIVE_S, seed=0xF00D)
    centre = _sulfur(mol)
    Chem.AssignStereochemistryFrom3D(mol)  # deliberately the RAW call: this test is about what it does
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert not _names_the_hand(mol, centre), "the writer already agrees with its readers; the premise is gone"


@pytest.mark.parametrize(
    "smiles", [_DATIVE_S, _DATIVE_S_LAST, _COVALENT_S], ids=["dative-first", "dative-last", "covalent"]
)
def test_3d_stereo_matches_reader_bond_order(smiles):
    mol = _mol(smiles, seed=0xF00D)
    centre = _sulfur(mol)
    assign_stereo_from_3d(mol)
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert _names_the_hand(mol, centre), "the tag names the mirror of the geometry it was written from"


def test_stereo_assignment_preserves_hypervalent_tag():
    mol = Chem.MolFromSmiles("F[C@](Cl)(Br)(I)->[Pd]", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    centre = mol.GetAtomWithIdx(1)
    assert centre.GetDegree() == 5, "the fixture is not the hypervalent case, so this asserts nothing"
    assert bond_removal_mirrors(centre, 5) is False, "the predicate itself lost the degree bound"

    raw = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(raw)
    assign_stereo_from_3d(mol)
    written = raw.GetAtomWithIdx(1).GetChiralTag()
    assert written != Chem.ChiralType.CHI_UNSPECIFIED, "the writer tagged nothing here, so this asserts nothing"
    assert mol.GetAtomWithIdx(1).GetChiralTag() == written, "the door re-based a hypervalent tag"
