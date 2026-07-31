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
        Chem.AssignStereochemistryFrom3D(mol)
    for b in orphaned:  # whatever re-perception could not re-reference carries no information, so drop it
        if len(b.GetStereoAtoms()) != _STEREO_REFS:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return len(orphaned)
