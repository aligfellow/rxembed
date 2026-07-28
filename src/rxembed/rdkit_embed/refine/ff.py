"""Force-field energies and constraint-enforcing restrained UFF.

The embed only *biases* constraints — ETKDG torsion terms override soft bounds —
so a restrained minimisation is what actually enforces them: distance/angle
windows, and frozen atoms pinned exactly (zero DOF).
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdForceFieldHelpers

logger = logging.getLogger("rxembed.refine.ff")  # pinned name: kept under the "rxembed" logger tree
#   (set_verbose configures) and the exact child caplog filters on — NOT getLogger(__name__), which would go dark
#   under set_verbose once the kernel graduates to a standalone `rdkit_embed` package (no longer under "rxembed").


@contextlib.contextmanager
def _quiet_uff(active):
    """Silence RDKit's C++ UFF-typer log iff `active` — the haptic centroid Xe ghost is deliberately untypeable.

    UFF emits one 'Unrecognized atom type: Xe3+4' per ghost per build; that is the intended behaviour (the ghost
    contributes no energy terms), so the warning is pure noise. Scoped to the FF construction and gated on there
    being a phantom, so a real UFF-typing problem in any normal system is still reported.
    """
    if active:
        with rdBase.BlockLogs():
            yield
    else:
        yield


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


def _ff_surrogate(mol, metals, phantoms=()):
    """Return a copy of ``mol`` with metals re-typed to the zero-vdW FF surrogate and phantoms Xe-ghosted.

    Nothing is added or removed, so every atom index is unchanged on both sides and no restraint can be silently
    dropped or mis-keyed. Each metal -> ``FF_SURROGATE`` (Li, a soft zero-vdW sphere); each ``phantom`` (a haptic
    centroid dummy) -> ``UFF_GHOST`` (Xe, untypeable, so UFF omits ALL its terms — it sits inside its own ring
    where a real vdW would explode). ``mol`` itself when there is nothing to re-type.
    """
    if not metals and not phantoms:
        return mol  # an organic system: the identical object, so this whole path is a strict no-op
    from rxembed.rdkit_embed.constraints.metal import FF_SURROGATE, UFF_GHOST

    rw = Chem.RWMol(mol)  # copies the conformers
    for m in metals:
        a = rw.GetAtomWithIdx(int(m))
        a.SetAtomicNum(FF_SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
    for p in phantoms:  # a haptic centroid dummy: kept for its restraints, but zero energy terms (see UFF_GHOST)
        a = rw.GetAtomWithIdx(int(p))
        a.SetAtomicNum(UFF_GHOST)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out


def restrained_uff(mol, cons, distance_fc=500.0, max_iters=500, conf_ids=None):
    """Minimise each conformer enforcing distance/angle windows + frozen atoms (pinned exactly, zero DOF).

    A metal brings more fields, all empty for an organic system (so this is bit-identical there): ``cons.metals``
    re-types the metal to the zero-vdW surrogate, ``cons.pulls`` adds the soft harmonic inside the flat-bottomed
    distance walls, ``cons.floors`` the anti-overbond guard, ``cons.coplanar`` the coplanarity cap. They ship
    together — each dropped alone regresses (see `metal.FF_SURROGATE` / `distance.ff_terms`).

    `conf_ids` restricts the settle to a subset of conformers (default: all) — used to settle different seeds to
    different in-window targets.
    """
    from rxembed.rdkit_embed.constraints import mechanisms as _mech
    from rxembed.rdkit_embed.constraints.metal import materialise_phantoms  # lazy: metal imports ff-adjacent modules

    frozen = set(cons.frozen)
    work = materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    work = _ff_surrogate(work, cons.metals, cons.phantoms)  # `mol` itself when there is no metal/phantom

    def build(target, conf_id):
        """Build a restrained UFF for one conformer by walking the mechanism registry.

        A flat additive loop — unlike the DG build there is no phase order here, because FF terms are
        independent. `mechanisms.MECHANISM_ORDER` still fixes the sequence so the two writers stay in step and a
        field's two halves are read together.
        """
        with _quiet_uff(bool(cons.phantoms)):  # the Xe ghost is untypeable by design — hush that one warning
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target, confId=conf_id, ignoreInterfragInteractions=False)
        conf = target.GetConformer(conf_id)
        for m in _mech.MECHANISM_ORDER:
            m.ff_terms(ff, cons, conf, distance_fc)
        ff.Initialize()
        return ff

    def bring_home(src_mol, conf):  # copy the relaxed coords back off a working copy (same indices, always)
        if src_mol is mol:
            return
        src = src_mol.GetConformer(conf.GetId())
        for a in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(a, src.GetAtomPosition(a))

    pruned = None  # a bond-pruned copy, built lazily only if a conformer's minimise diverges
    energies = []
    confs = mol.GetConformers() if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    for conf in confs:
        cid = conf.GetId()
        ff = build(work, cid)  # a BUILD failure (UFF can't type the graph) propagates -> caller keeps the embed
        try:
            ff.Minimize(maxIts=max_iters)
            energies.append(ff.CalcEnergy())
            bring_home(work, conf)
        except RuntimeError:  # the UFF line search DIVERGED on this conformer — a pathological frozen-core
            if pruned is None:  # partial/hypervalent bond, or an RDKit BFGS "bad direction" on a hard metal core.
                pruned = _bond_pruned(work, frozen)  # NEVER fatal: retry pruned, else keep this conformer unrelaxed.
            relaxed = False
            if pruned is not None:  # retry on a copy with the frozen-frozen partial bonds dropped
                try:
                    pff = build(pruned, cid)
                    pff.Minimize(maxIts=max_iters)
                    src = pruned.GetConformer(cid)  # copy the relaxed FREE-atom positions back (frozen unmoved)
                    for a in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(a, src.GetAtomPosition(a))
                    energies.append(pff.CalcEnergy())
                    relaxed = True
                except RuntimeError:
                    pass
            if not relaxed:  # keep this conformer's geometry (frozen core still pinned); bonding_ok is the arbiter
                logger.debug("restrained_uff: conformer %d could not relax (BFGS diverged); kept unrelaxed", cid)
                try:
                    energies.append(float(ff.CalcEnergy()))
                except RuntimeError:
                    energies.append(float("nan"))  # can't even score it — placeholder, kept aligned to confs
    return np.array(energies)
