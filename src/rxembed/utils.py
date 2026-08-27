"""Coordinate math, small RDKit facts, and the QA result type: the core's shared leaf.

Admission rule: if it needs to know what a metal, a donor or a constraint is, it does not belong here.
Everything in this module reads a molecule or an array and answers a question that has no domain in it, which
is what lets the coordination perception and the pipeline QA gate measure an angle the same way without a
second definition or an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()
_CARBON_Z = 6
_SP2_DEGREE = 3  # a planar sp2 centre has exactly three neighbours
_STEREO_REFS = 2  # a double bond's stereo needs exactly two reference atoms; fewer means surgery orphaned it
_DISCONNECTED = 1e6  # RDKit's topological distance for atoms in different fragments (it returns ~1e8)
_TETRAHEDRAL_DEGREE = 4  # the only degree at which a CW/CCW tag has a bond-order parity: see `bond_removal_mirrors`
_MIRRORED = {  # the two tetrahedral tags; no other ChiralType is a parity over the bond order
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


@dataclass(frozen=True)
class Violation:
    """One failed check. ``value`` breached ``limit`` for the atoms named."""

    kind: str
    atoms: tuple[int, ...]
    value: float = field(kw_only=True)
    limit: float = field(kw_only=True)
    detail: str = field(default="", kw_only=True)

    def __str__(self) -> str:
        """Format the violation as a readable one-liner."""
        a = "-".join(map(str, self.atoms))
        return f"[{self.kind}] atoms {a}: {self.value:.3f} vs {self.limit:.3f} {self.detail}".rstrip()


def conjugated_quartets(mol, exclude=frozenset()):
    """Yield each ``(a, c, x, s)`` quartet whose A=C-X-S dihedral measures a conjugation plane.

    A single, non-ring bond X-C (X in N/O, C bearing a double bond to A) where X carries a substituent S. The
    sole perception both ``geometry.conjugation`` and ``ConjugationCap`` read, so they cannot name different
    atoms. ``exclude`` drops a quartet whose X or C is a frozen / metal atom, as the gate does.
    """
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != _CARBON_Z:
                continue
            if x_atom.GetIdx() in exclude or c_atom.GetIdx() in exclude:
                continue
            dbl = [
                n
                for n in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [n for n in x_atom.GetNeighbors() if n.GetIdx() != c_atom.GetIdx()]
            if not dbl or not subs:
                continue
            yield dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()


def _positions(mol_or_pos, conf_id: int = -1) -> np.ndarray:
    """(N, 3) coordinates from a Mol conformer or a pass-through array."""
    if isinstance(mol_or_pos, np.ndarray):
        return mol_or_pos
    return mol_or_pos.GetConformer(conf_id).GetPositions()


def _rcov(z: int) -> float:
    return _PT.GetRcovalent(z)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, w = a - b, c - b
    cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral p0-p1-p2-p3 in degrees."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    axis = np.linalg.norm(b1)
    if axis == 0.0:
        return float("nan")
    b1 /= axis
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    eps = np.finfo(float).eps
    if np.linalg.norm(v) <= eps * np.linalg.norm(b0) or np.linalg.norm(w) <= eps * np.linalg.norm(b2):
        return float("nan")
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def mirror_tag(tag):
    """Return the tetrahedral tag naming the mirror image of `tag`; anything else passes through unchanged."""
    return _MIRRORED.get(tag, tag)


def bond_removal_mirrors(atom, partner) -> bool:
    """Return whether removing the bond to `partner` changes `atom`'s tetrahedral-tag parity.

    RDKit CW/CCW tags are relative to ``atom.GetBonds()`` order. Removing slot ``p`` changes that basis by
    ``n - 1 - p`` swaps, so an odd count requires mirroring the tag. Call this on the graph that still has
    the bond, even if no tag exists yet; the same correction applies inversely when grafting a tag. A plain
    ``AddBond`` needs no correction because RDKit appends the bond last.

    This parity rule is defined only for degree four. Higher-degree centres are left unchanged; at lower
    degrees, removing a bond leaves no representable tetrahedral chirality.
    """
    partners = [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]
    if len(partners) != _TETRAHEDRAL_DEGREE or partner not in partners:
        return False
    return (len(partners) - 1 - partners.index(partner)) % 2 == 1


def bond_replacement_mirrors(atom, partner) -> bool:
    """Return whether replacing a tetrahedral neighbour by an appended one changes tag parity."""
    partners = [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]
    if len(partners) not in (3, 4) or partner not in partners:
        return False
    return (len(partners) - 1 - partners.index(partner)) % 2 == 1


def remove_bond(rw, i, j) -> None:
    """Remove a bond while preserving the geometry named by degree-four tetrahedral tags.

    RDKit does not update chiral tags on bond removal, so persistent stereochemistry-sensitive edits use this
    wrapper. Direct removal inverted four chiral-P centres across three structures; RMSD missed
    equivalent-donor swaps.
    """
    for a, other in ((int(i), int(j)), (int(j), int(i))):
        atom = rw.GetAtomWithIdx(a)
        if bond_removal_mirrors(atom, other):
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
    rw.RemoveBond(int(i), int(j))


def assign_stereo_from_3d(mol, conf_id: int = -1) -> None:
    """Assign 3D stereochemistry in the bond-order basis used by rxembed readers.

    RDKit's 3D writer omits a donor-originating dative bond from the centre's neighbour basis, while its CIP
    and SMILES readers include it. On the sulfoxide fixture, the same geometry is CIP R after parsing and CIP
    S after raw 3D assignment. Mirror degree-four tags when that difference changes parity, then refresh their
    CIP labels. Higher-degree tags remain unchanged because the parity rule does not hold there. All production
    3D assignments use this wrapper.
    """
    Chem.AssignStereochemistryFrom3D(mol, confId=conf_id)
    rebased = False
    for atom in mol.GetAtoms():
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() == atom.GetIdx():
                if bond_removal_mirrors(atom, bond.GetOtherAtomIdx(atom.GetIdx())):
                    atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
                    rebased = True
    if rebased:
        # AssignStereochemistryFrom3D also writes `_CIPCode`; refresh it after mirroring a tag.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def repair_bond_stereo(mol) -> int:
    """Re-derive (or drop) any bond stereo whose reference atoms were lost to bond surgery; return how many.

    Removing a bond can leave a double bond still FLAGGED ``STEREOZ``/``STEREOE`` while RDKit silently drops its
    two stereo reference atoms, because one of them was the removed partner. Stripping the M-donor bonds does
    this routinely: for a coordinated imine, RDKit picks the metal itself as one of the C=N reference atoms.
    The flag then carries no information, and RDKit's own ETKDG indexes the empty vector and segfaults:
    a crash no ``try``/``except`` can catch.

    Where a conformer survives the surgery the geometry is re-perceived from it, so the E/Z is PRESERVED rather
    than discarded, re-expressed against the substituents that remain (a bond that was "Z relative to the
    metal" becomes "E relative to the other ring atom": same 3D arrangement, new reference). Only where no
    reference survives is the flag dropped, and there the information genuinely did not survive the surgery.

    This is the bond-stereo counterpart of the atom-level cleanups `surrogate_metal` already does (it clears the
    surrogate's chiral tag for exactly this reason, and `_clear_labile_donor_stereo` clears a donor's).
    """
    # Metal-ring SMILES can make RDKit choose the metal as both E/Z references. Once bond surgery removes
    # the metal, the original slash bonds are still the least ambiguous source of the ligand's E/Z. Preserve
    # already-valid E/Z verbatim: SetBondStereoFromDirections otherwise rewrites E/Z as TRANS/CIS.
    stated = {
        b.GetIdx(): (b.GetStereo(), tuple(b.GetStereoAtoms()))
        for b in mol.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
        and len(b.GetStereoAtoms()) == _STEREO_REFS
        and len(set(b.GetStereoAtoms())) == _STEREO_REFS
    }
    Chem.SetBondStereoFromDirections(mol)
    for idx, (tag, refs) in stated.items():
        bond = mol.GetBondWithIdx(idx)
        bond.SetStereoAtoms(*refs)
        bond.SetStereo(tag)
    orphaned = [
        b
        for b in mol.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != _STEREO_REFS
    ]
    if not orphaned:
        return 0
    if mol.GetNumConformers():  # re-perceive from the geometry: keeps the E/Z, re-referenced
        assign_stereo_from_3d(mol)
    for b in orphaned:  # whatever re-perception could not re-reference carries no information, so drop it
        if len(b.GetStereoAtoms()) != _STEREO_REFS:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return len(orphaned)
