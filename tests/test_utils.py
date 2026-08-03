"""`utils.py`; coordinate math, small RDKit facts, and the QA result type: the core's shared leaf.

Its admission rule is that nothing here may know what a metal, a donor or a constraint is; that is what lets
the coordination perception and the pipeline's QA gate measure an angle the same way, with no second
definition and no import cycle. The first test is what stops the rule being only a comment.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.utils import (
    Violation,
    _angle,
    _dihedral,
    assign_stereo_from_3d,
    bond_removal_mirrors,
    conjugated_quartets,
    remove_bond,
    repair_bond_stereo,
)

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "src/rxembed/utils.py"


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
# the admission rule
# ---------------------------------------------------------------------------------------------------------


def test_utils_imports_nothing_from_the_package():
    """A leaf with no in-package import cannot acquire a domain, and cannot be in an import cycle."""
    tree = ast.parse(_SOURCE.read_text())
    reached = [
        node.module or f"(relative level {node.level})"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").startswith("rxembed"))
    ]
    reached += [n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names]
    assert not [m for m in reached if m.startswith(("rxembed", "(relative"))], (
        f"utils reached back into the package: {reached}"
    )


# ---------------------------------------------------------------------------------------------------------
# coordinate math
# ---------------------------------------------------------------------------------------------------------


def test_angle_is_degrees_and_measured_at_the_middle_atom():
    """`_angle(a, b, c)` measures at b: a right angle reads 90, not pi/2."""
    o, x, y = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert _angle(x, o, y) == pytest.approx(90.0)
    assert _angle(x, o, -x) == pytest.approx(180.0)


def test_dihedral_is_signed():
    """The sign is what distinguishes the two hands, so mirrored inputs give opposite, non-zero values."""
    p0, p1, p2 = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    plus, minus = np.array([0, 1.0, 1.0]), np.array([0, -1.0, 1.0])
    assert _dihedral(p0, p1, p2, plus) == pytest.approx(-_dihedral(p0, p1, p2, minus))
    assert _dihedral(p0, p1, p2, plus) != 0.0


# ---------------------------------------------------------------------------------------------------------
# conjugated_quartets: the one perception the QA gate and the FF cap must agree on
# ---------------------------------------------------------------------------------------------------------


def test_a_conjugation_plane_is_perceived_on_an_amide_and_not_on_a_saturated_chain():
    """O=C-N-C is the canonical quartet; no double bond means no plane, so the cap cannot fire on an alkane."""
    quartets = list(conjugated_quartets(_mol("CC(=O)NC")))
    assert quartets, "the amide plane was not perceived"
    for a, c, x, s in quartets:
        assert len({a, c, x, s}) == 4, "a quartet must name four distinct atoms"
    assert list(conjugated_quartets(_mol("CCCC"))) == []


def test_exclude_drops_a_quartet_naming_an_excluded_atom():
    """`exclude` is how a frozen core or a metal keeps its own geometry: the QA gate passes the same set."""
    mol = _mol("CC(=O)NC")
    hit = next(iter(conjugated_quartets(mol)))
    assert hit not in list(conjugated_quartets(mol, exclude={hit[2]}))


# ---------------------------------------------------------------------------------------------------------
# repair_bond_stereo: the cleanup bond surgery needs
# ---------------------------------------------------------------------------------------------------------


def test_a_stereo_flag_orphaned_by_bond_removal_is_dropped_and_a_clean_molecule_is_left_alone():
    """A flagged double bond with no reference atoms left segfaults ETKDG: no try/except catches it."""
    work = _without(Chem.MolFromSmiles(r"C/C=C/Cl"), 3)  # no conformer: nothing to re-perceive from
    assert repair_bond_stereo(work) == 1
    for b in work.GetBonds():
        assert b.GetStereo() == Chem.BondStereo.STEREONONE, "a flagged bond kept fewer than two reference atoms"

    clean = _mol(r"C/C=C/C", seed=1)
    Chem.AssignStereochemistryFrom3D(clean)
    before = [(b.GetIdx(), b.GetStereo()) for b in clean.GetBonds()]
    assert repair_bond_stereo(clean) == 0, "nothing was orphaned, so nothing may be re-perceived"
    assert [(b.GetIdx(), b.GetStereo()) for b in clean.GetBonds()] == before


def test_a_geometry_re_references_an_orphaned_flag_rather_than_discarding_the_e_z():
    """With a conformer the E/Z is re-derived against the substituents that survived, not dropped.

    Stripping the M-donor bonds orphans a coordinated imine routinely (RDKit picks the metal as one of the C=N
    reference atoms), and the 3D arrangement is still there to read, so only a flag no reference survives is lost.
    """
    work = _without(_mol(r"C/C=C/Cl", seed=1), 3)  # the Cl was one of the two reference atoms
    assert repair_bond_stereo(work) == 1
    bond = work.GetBondBetweenAtoms(1, 2)
    assert bond.GetStereo() != Chem.BondStereo.STEREONONE, "the E/Z was discarded, not re-referenced"
    assert len(bond.GetStereoAtoms()) == 2
    assert 3 not in list(bond.GetStereoAtoms())


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
def test_a_bond_removal_leaves_the_tag_naming_the_same_geometry(slot):
    """THE RULE, measured: taking the bond at slot `p` of four out of the order mirrors the tag when 3-p is odd.

    Nothing in RDKit does this: `RWMol.RemoveBond` leaves the tag verbatim in a basis that has changed. A
    chiral-at-P donor whose M-L bond sat at an odd slot therefore came back as its own mirror image, in 8 of 8
    conformers, silently. Refereed by the geometry rather than by CIP, so no label convention enters.
    """
    mol = _mol(_HALIDE_C, seed=11)
    Chem.AssignStereochemistryFrom3D(mol)  # calibrate: RDKit's own writer, on this very conformer
    assert _names_the_hand(mol, 1), "the sign convention this test refereeds by is wrong"
    rw = Chem.RWMol(mol)
    remove_bond(rw, 1, [b.GetOtherAtomIdx(1) for b in mol.GetAtomWithIdx(1).GetBonds()][slot])
    assert _names_the_hand(rw.GetMol(), 1), f"slot {slot}: the tag now names the mirror of its own geometry"


def test_adding_a_bond_back_needs_no_counterpart():
    """`RWMol.AddBond` appends LAST, which is the slot the missing reference already occupied.

    This is why `connect_metal`, the D-cap and every `AddHs` are free of the rule, and why one correction at
    the removal is the whole of it rather than half of a pair.
    """
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
        ("F[C@](Cl)(Br)I", 0, True),  # degree 4: 3 - 0 is odd
        ("F[C@](Cl)(Br)I", 1, False),  # degree 4: 3 - 1 is even
        ("F[P@](Cl)(Br)(I)F", 1, False),  # degree 5: the arithmetic would say odd, and is refuted there
        ("F[C@](Cl)Br", 1, False),  # degree 3: no representable tag survives, so the caller clears it
    ],
)
def test_the_parity_rule_is_bounded_to_degree_four(smiles, slot, mirrors):
    """Off degree four the symbol is kept: four directions summing to zero is what makes ``n - 1 - p`` work.

    Measured refuted at degree 5 (13/35 and anti-correlated on the eight hypervalent tags in the corpora), so
    the bound is part of the rule rather than a caution around it. The two `False` rows below are the ones
    that bite: the unbounded arithmetic calls both of them odd.
    """
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


def test_rdkits_3d_writer_reads_a_bond_order_none_of_its_readers_read():
    """THE PREMISE, pinned against RDKit itself, because the whole re-base is worthless if it ever stops holding.

    `assignChiralTypesFrom3D` skips a DATIVE bond whose BEGIN atom is the centre; the DG embedder, both CIP
    labellers and the SMILES writer count it. So the raw writer hands back the symbol for the mirror of the
    geometry it was just shown, and nothing complains. If RDKit reconciles the two, this cell fails and the
    re-base should be deleted rather than kept working around a bug that is gone.
    """
    mol = _mol(_DATIVE_S, seed=0xF00D)
    centre = _sulfur(mol)
    Chem.AssignStereochemistryFrom3D(mol)  # deliberately the RAW call: this test is about what it does
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert not _names_the_hand(mol, centre), "the writer already agrees with its readers; the premise is gone"


@pytest.mark.parametrize("smiles", [_DATIVE_S, _DATIVE_S_LAST, _COVALENT_S])
def test_a_tag_written_from_3d_names_its_own_geometry_in_the_readers_bond_order(smiles):
    """THE CONTRACT. Whatever the bond types, the tag left behind names the hand of the conformer it was read from.

    This is what lets one parity rule hold for every tag in the graph without asking where a tag came from:
    provenance is not recoverable (the SMILES parser leaves no positive marker), and a rule betting either way
    on a dative bond loses somewhere. Fixing it at the writer means nothing downstream has to bet.
    """
    mol = _mol(smiles, seed=0xF00D)
    centre = _sulfur(mol)
    assign_stereo_from_3d(mol)
    assert mol.GetAtomWithIdx(centre).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED, "nothing was written"
    assert _names_the_hand(mol, centre), "the tag names the mirror of the geometry it was written from"


def test_the_re_base_leaves_a_hypervalent_perception_alone():
    """It carries the degree-four bound, so a carborane cage vertex keeps whatever symbol the writer gave it.

    The writer reaches those (it drops the dative and tags the remaining four), but the parity arithmetic is
    refuted above degree four and the embedder truncates such an atom to its first four bonds anyway, so
    re-basing there would be guessing. Asserted against the door itself, on a mol with a conformer, because
    the bound only means something if the door is what honours it.
    """
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


# ---------------------------------------------------------------------------------------------------------
# Violation: the QA result type both the core perception and the pipeline gate return
# ---------------------------------------------------------------------------------------------------------


def test_a_violation_formats_the_atoms_the_value_and_the_limit():
    """It is read in a log line, so the one-liner must name all three, with or without a detail."""
    assert str(Violation("clash", (3, 7), 1.234, 2.5, "H...H")) == "[clash] atoms 3-7: 1.234 vs 2.500 H...H"
    assert str(Violation("clash", (3, 7), 1.234, 2.5)) == "[clash] atoms 3-7: 1.234 vs 2.500"
