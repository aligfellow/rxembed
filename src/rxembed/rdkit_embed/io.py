"""Repair RDKit bond stereo left dangling by the metal-bond surgery (Mol -> Mol engine plumbing).

Not user-input parsing — that moved to the shell leaf `rxembed.inputs`. What remains is the Mol->Mol
cleanup the kernel metal surrogate (`constraints/metal.surrogate_metal`) calls after stripping M-donor
bonds. Depends on rdkit alone and imports no kernel module, so it stays the bottom of the stack.
"""

from __future__ import annotations

from rdkit import Chem

_STEREO_REFS = 2  # a bond stereo descriptor is meaningful only with BOTH its reference atoms


def repair_bond_stereo(mol) -> int:
    """Re-derive (or drop) any bond stereo whose reference atoms were lost to bond surgery; return how many.

    Removing a bond can leave a double bond still FLAGGED ``STEREOZ``/``STEREOE`` while RDKit silently drops its
    two stereo reference atoms — because one of them WAS the removed partner. Stripping the M-donor bonds does
    this routinely: for a coordinated imine, RDKit picks the metal itself as one of the C=N reference atoms.
    The flag then carries no information, and RDKit's own ETKDG indexes the empty vector and **segfaults** —
    a crash no ``try``/``except`` can catch.

    Where a conformer survives the surgery the geometry is re-perceived from it, so the E/Z is PRESERVED rather
    than discarded — re-expressed against the substituents that remain (a bond that was "Z relative to the
    metal" becomes "E relative to the other ring atom": same 3D arrangement, new reference). Only where no
    reference survives is the flag dropped, and there the information genuinely did not survive the surgery.

    This is the bond-stereo counterpart of the atom-level cleanups `metal.surrogate_metal` already does (it clears the
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
    for b in orphaned:  # whatever re-perception could not re-reference carries no information — drop it
        if len(b.GetStereoAtoms()) != _STEREO_REFS:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return len(orphaned)
