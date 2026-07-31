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

from rxembed.utils import Violation, _angle, _dihedral, conjugated_quartets, repair_bond_stereo

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
# Violation: the QA result type both the core perception and the pipeline gate return
# ---------------------------------------------------------------------------------------------------------


def test_a_violation_formats_the_atoms_the_value_and_the_limit():
    """It is read in a log line, so the one-liner must name all three, with or without a detail."""
    assert str(Violation("clash", (3, 7), 1.234, 2.5, "H...H")) == "[clash] atoms 3-7: 1.234 vs 2.500 H...H"
    assert str(Violation("clash", (3, 7), 1.234, 2.5)) == "[clash] atoms 3-7: 1.234 vs 2.500"
