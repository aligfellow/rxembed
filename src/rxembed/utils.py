"""Coordinate math, small RDKit facts, and the QA result type: the core's shared leaf.

Admission rule: if it needs to know what a metal, a donor or a constraint is, it does not belong here.
Everything in this module reads a molecule or an array and answers a question that has no domain in it, which
is what lets the coordination perception and the pipeline QA gate measure an angle the same way without a
second definition or an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    value: float
    limit: float
    detail: str = ""

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
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def mirror_tag(tag):
    """Return the tetrahedral tag naming the mirror image of `tag`; anything else passes through unchanged."""
    return _MIRRORED.get(tag, tag)


def bond_removal_mirrors(atom, partner) -> bool:
    """Whether taking `atom`'s bond to `partner` out of its bond list makes a tag there name the MIRROR.

    A CW/CCW tag is a parity over the atom's OWN bond order (insertion order, ``atom.GetBonds()``), with the
    fourth reference it does not bond to -- an implicit H, a lone pair, the centre itself -- pinned LAST.
    Detaching `partner` from slot ``p`` permutes that basis by ``n - 1 - p`` transpositions, so the same symbol
    names the opposite hand whenever that count is odd. Nothing in RDKit does this for you: neither
    `RWMol.RemoveBond` nor `RemoveAtom` touches a tag, and `SanitizeMol` only ever drops one.

    Read it on whichever graph still HAS the bond, and it answers both directions: the correction is its own
    inverse, so it serves a removal (`remove_bond`) and a tag written back across a bond the source graph did
    not have (`stereo.graft`). An ADDITION needs no counterpart, since `RWMol.AddBond` always appends last,
    which is the slot the missing reference already occupied.

    Deliberately independent of whether a tag is present, so a centre tagged only later still gets the right
    basis. Degree four only: four directions from one centre sum to zero, which is what makes the signed
    volumes alternate and ``n - 1 - p`` predict the hand. A perceived hypervalent centre has no such closure
    and the arithmetic is refuted there (13/35, anti-correlated, on the eight degree-5 tags in the corpora),
    so those keep their symbol. Below four the removal leaves no representable tag and the caller clears it.
    """
    partners = [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]
    if len(partners) != _TETRAHEDRAL_DEGREE or partner not in partners:
        return False
    return (len(partners) - 1 - partners.index(partner)) % 2 == 1


def remove_bond(rw, i, j) -> None:
    """`RWMol.RemoveBond`, with each end's chiral tag left naming the geometry it already named.

    The single door for bond removal in the core, so the `bond_removal_mirrors` rule cannot be forgotten at a
    new surgery site (`tests/test_init.py` walks the AST to keep it that way). The metal-donor strip is the
    case that bit: a chiral-at-P donor whose M-L bond sat at an odd slot came back as its own mirror image,
    silently, with no CIP or valence complaint anywhere. Four centres over three structures, read as a
    perceived R/S descriptor rather than an RMSD, which cannot tell an inversion from two equivalent donors
    swapping.
    """
    for a, other in ((int(i), int(j)), (int(j), int(i))):
        atom = rw.GetAtomWithIdx(a)
        if bond_removal_mirrors(atom, other):
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
    rw.RemoveBond(int(i), int(j))


def assign_stereo_from_3d(mol, conf_id: int = -1) -> None:
    """`Chem.AssignStereochemistryFrom3D`, leaving every tag it writes in the basis its readers use.

    The one door for writing stereo from a geometry, because RDKit's 3D writer is the only thing in the
    toolkit that reads an atom's neighbours differently from everything else: it drops a DATIVE bond whose
    BEGIN atom is the centre (`Chirality.cpp` `bondAffectsAtomChirality`). The DG embedder, both CIP
    labellers and the SMILES writer all count that bond. So at a dative-bonded donor the writer produces a
    parity over a basis nothing downstream reads, and one symbol names opposite hands depending on who is
    asking: on `Cl[Pd](Cl)(Cl)<-[S@](=O)(C)CC` the SAME geometry gets CIP R from the parsed tag and CIP S
    from the written one.

    Re-basing here, at the one place the writer is known, is what lets `bond_removal_mirrors` hold for every
    tag in the graph without asking where a tag came from. Inferring it later cannot work: the parser leaves
    no positive marker, and a rule betting either way loses (betting "the dative is never in the basis"
    mirrors the README's dative-SMILES idiom; betting "always" mirrors `LISVIW`'s sulfoxide S).

    It carries `bond_removal_mirrors`' degree-four bound, deliberately: the writer reaches a hypervalent
    perception too (a carborane cage vertex, where it drops the dative and tags the remaining four), but the
    parity arithmetic is refuted there and the embedder truncates such an atom to its first four bonds
    anyway, so those tags are left exactly as they were.
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
        # `AssignStereochemistryFrom3D` stamps `_CIPCode` inside itself, from the tag it wrote, so a
        # re-based atom is left carrying the label of its own mirror until the CIP pass is re-run. Leaving
        # the mol self-contradictory is worse than the mis-based tag was: `geom_check._cip` reads the
        # property, not the tag, and reported the mirrored R/S at every dative-bonded donor either way.
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
