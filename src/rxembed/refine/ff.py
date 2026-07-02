"""Force-field energies and constraint-enforcing restrained UFF.

The embed only *biases* constraints — ETKDG torsion terms override soft bounds —
so a restrained minimisation is what actually enforces them: distance/angle
windows, and frozen atoms pinned exactly (zero DOF).
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers


def _bond_pruned(mol, frozen):
    """Copy `mol` with bonds *between two frozen atoms* removed; return None if there are none.

    Frozen atoms are held exactly by ``AddFixedPoint`` (zero DOF), so a bond purely among them carries no
    force on any relaxing atom — but a TS reacting core's partial/hypervalent bonds (a sub-Å H...H being
    cleaved, a bridging hydride, an over-coordinated centre) make UFF's line search diverge ("bad direction
    in linearSearch"). Dropping only frozen-frozen bonds removes those pathological terms; every
    frozen-FREE bond (which anchors a relaxing atom to the held core) is kept, and no atom moves.
    """
    ff_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in frozen and b.GetEndAtomIdx() in frozen
    ]
    if not ff_bonds:
        return None
    rw = Chem.RWMol(mol)  # copies the conformers (atom indices unchanged)
    for a in frozen:
        rw.GetAtomWithIdx(a).SetNoImplicit(True)  # dropping a bond must not sprout an implicit H
    for i, j in ff_bonds:
        rw.RemoveBond(i, j)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out


def ff_energies(mol, minimize=True):
    """FF energies (MMFF94s where typeable, else UFF); optimise in place first when ``minimize``."""
    use_mmff = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
    if minimize:
        res = (
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s")
            if use_mmff
            else rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        )
        return np.array([e for _conv, e in res])
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s") if use_mmff else None

    def ff(c):
        return (
            rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=c)
            if use_mmff
            else rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=c)
        )

    return np.array([ff(c.GetId()).CalcEnergy() for c in mol.GetConformers()])


def restrained_uff(mol, cons, distance_fc=500.0, angle_fc=None, max_iters=500, extra_frozen=(), conf_ids=None):
    """Minimise each conformer enforcing distance/angle windows + frozen atoms (pinned exactly, zero DOF).

    `angle_fc` is the force constant for angle (orientation) restraints; it defaults to a gentler value
    than `distance_fc` (which can be very stiff, 1e4) because distance (Å⁻²) and angle (rad⁻²) constants
    are not the same units, and an over-stiff angle distorts the framework of a rigid/bidentate contact.

    `extra_frozen` pins additional atoms (beyond ``cons.frozen``) for this call only — used by the metal
    donor-proton relax to hold every atom except the donor hydrogens. `conf_ids` restricts the settle to a
    subset of conformers (default: all) — used to settle different seeds to different in-window targets.
    """
    afc = angle_fc if angle_fc is not None else min(distance_fc, 1e3)
    frozen = (*cons.frozen, *extra_frozen)

    def build(target, conf_id):  # a restrained UFF for one conformer of `target`
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target, confId=conf_id, ignoreInterfragInteractions=False)
        for idx in frozen:
            ff.AddFixedPoint(idx)  # remove the DOF entirely (exact hold) rather than a stiff harmonic — a TS
            #                        core with partial bonds must NOT be relaxed; a soft pin would let UFF
            #                        crush the sub-Å H...H contacts into garbage
        for (i, j), (lo, hi) in cons.distances.items():
            ff.AddDistanceConstraint(i, j, lo, hi, distance_fc)
        for (i, j, k), (lo, hi) in cons.angles.items():
            ff.UFFAddAngleConstraint(i, j, k, False, max(0.0, lo), min(180.0, hi), afc)
        pos = target.GetConformer(conf_id).GetPositions()
        for ring_a, ring_b, _sep in cons.planes:  # hold the embedded parallel stack
            for a, b in itertools.product(ring_a, ring_b):
                d = float(np.linalg.norm(pos[a] - pos[b]))
                ff.AddDistanceConstraint(a, b, d - 0.3, d + 0.3, 0.2 * distance_fc)
        ff.Initialize()
        return ff

    pruned = None  # a bond-pruned copy, built lazily only if a conformer's minimise diverges
    energies = []
    confs = mol.GetConformers() if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    for conf in confs:
        cid = conf.GetId()
        try:
            ff = build(mol, cid)
            ff.Minimize(maxIts=max_iters)
            energies.append(ff.CalcEnergy())
        except RuntimeError:  # UFF line search diverged — a pathological frozen-core partial/hypervalent bond
            if pruned is None:
                pruned = _bond_pruned(mol, frozen)
            if pruned is None:
                raise  # nothing frozen-frozen to drop -> genuinely can't relax; caller keeps the embed
            ff = build(pruned, cid)
            ff.Minimize(maxIts=max_iters)
            src = pruned.GetConformer(cid)  # copy the relaxed FREE-atom positions back (frozen ones unmoved)
            for a in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(a, src.GetAtomPosition(a))
            energies.append(ff.CalcEnergy())
    return np.array(energies)
