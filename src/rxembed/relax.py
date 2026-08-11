"""Relax conformers under constraints and reject geometrically broken results."""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import GetPeriodicTable, rdForceFieldHelpers, rdMolTransforms

from . import mechanisms as _mech
from .metal_core import COORDINATION_METALS, materialise_phantoms, strip_phantoms

FF_SURROGATE = 3  # Bondless Li avoids the singular CN>=3 UFF angle term while retaining a soft vdW sphere.
UFF_GHOST = 54  # Untypeable Xe gives a haptic centroid no UFF terms inside its ring.

logger = logging.getLogger("rxembed.relax")

MAX_ITERS = 500  # the one restrained-UFF iteration cap: every relax entry point defaults from this name
_DIHEDRAL_ATOMS = 4
_PT = GetPeriodicTable()


def _error_summary(error):
    """Return the useful line from a multiline RDKit force-field error."""
    noise = ("Pre-condition Violation", "Violation occurred", "Failed Expression", "RDKIT:", "BOOST:")
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return next((line for line in lines if not line.startswith(noise)), lines[-1] if lines else "unknown error")


def bonding_ok(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset(), constrained=()):
    """Return whether heavy-atom bonds and clashes match the stated graph.

    Frozen-core, metal and explicitly constrained pairs are exempt because covalent radii do not define their
    intended distances. The exemption is per pair, so broken free-periphery bonds still fail.
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    metals = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in COORDINATION_METALS}
    exclude = set(exclude)
    stated = {frozenset(p) for p in constrained}
    rcov = {i: _PT.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy}
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    for n, i in enumerate(heavy):
        for j in heavy[n + 1 :]:
            if (i in exclude and j in exclude) or i in metals or j in metals:
                continue
            if frozenset((i, j)) in stated:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cut = rcov[i] + rcov[j]
            if frozenset((i, j)) in bonded:
                if d > bond_tol * cut or d < clash_tol * cut:  # bonded pair stretched/broken OR crushed
                    return False
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                return False
    return True


def _uff_core_graphs(mol, frozen):
    """Yield private FF graphs with implicated frozen-core bonds made dative.

    RDKit excludes a dative bond from its start atom's valence, keeping a hypervalent electrophile typeable
    without disconnecting the force-field graph. The public molecular graph is untouched.
    """
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in frozen and b.GetEndAtomIdx() in frozen and b.GetBondType() != Chem.BondType.DATIVE
    ]

    def pressure(i):
        atom = mol.GetAtomWithIdx(i)
        default = max(_PT.GetDefaultValence(atom.GetAtomicNum()), 0)
        excess = atom.GetValence(Chem.ValenceType.EXPLICIT) - default
        return atom.HasValenceViolation(), excess, atom.GetDegree()

    directed = []
    for i, j in bonds:
        start, end = sorted((i, j), key=pressure, reverse=True)
        if mol.GetAtomWithIdx(start).HasValenceViolation():
            directed.append((start, end))

    # Try one edge first. If several atoms violate valence, retype only enough low-pressure neighbours at each
    # centre to remove its excess valence; this stays linear instead of enumerating 2**N subsets.
    choices = [[edge] for edge in directed]
    by_start = {}
    for edge in directed:
        by_start.setdefault(edge[0], []).append(edge)
    combined = []
    for start, edges in by_start.items():
        needed = max(1, int(pressure(start)[1]))
        combined.extend(sorted(edges, key=lambda edge: pressure(edge[1]))[:needed])
    if len(combined) > 1:
        choices.append(combined)
    for chosen in choices:
        rw = Chem.RWMol(mol)
        for start, end in chosen:
            rw.RemoveBond(start, end)
        for a in {a for edge in chosen for a in edge}:
            atom = rw.GetAtomWithIdx(a)
            atom.SetNoImplicit(True)
            atom.SetHybridization(Chem.HybridizationType.UNSPECIFIED)
        mid = rw.GetMol()
        mid.UpdatePropertyCache(strict=False)
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mid, Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION, catchErrors=True)
        rw = Chem.RWMol(mid)
        for start, end in chosen:
            rw.AddBond(start, end, Chem.BondType.DATIVE)
        out = rw.GetMol()
        out.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(out)
        yield out, len(chosen)


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
    """Retype metals and haptic centroids on a private UFF graph without changing atom indices."""
    if not metals and not phantoms:
        return mol  # an organic system: the identical object, so this whole path is a strict no-op
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


def _restore_positions(conf, positions):
    """Restore one conformer's Cartesian coordinates."""
    for atom, xyz in enumerate(positions):
        conf.SetAtomPosition(atom, xyz)


def _seat_fixed_dihedrals(confs, fixed, frozen):
    """Rotate connected fixed dihedrals near target without disturbing a rigid graft."""
    frozen = sorted(frozen)
    for atoms, (lo, hi) in fixed.items():
        if len(atoms) != _DIHEDRAL_ATOMS:
            continue
        i, j, k, w = atoms
        for conf in confs:
            before = conf.GetPositions().copy() if frozen else None
            try:
                rdMolTransforms.SetDihedralDeg(conf, i, j, k, w, 0.5 * (lo + hi))
            except (RuntimeError, ValueError):
                pass  # rings may not rotate; the strict post-UFF gate remains authoritative
            if before is not None and not np.array_equal(conf.GetPositions()[frozen], before[frozen]):
                _restore_positions(conf, before)  # a coordinate graft is stricter than a numeric torsion


def _prepare_uff_work(mol, cons, confs, frozen, max_iters):
    """Seat torsions and build private FF graphs atomically, returning the original coordinates."""
    original = {conf.GetId(): conf.GetPositions().copy() for conf in confs}
    try:
        _seat_fixed_dihedrals(confs, cons.fixed if max_iters else {}, frozen)
        work = materialise_phantoms(mol, cons.haptic)  # private FF graphs inherit the seated coordinates
        work = _ff_surrogate(work, cons.metals, cons.phantoms)  # `mol` itself for an organic system
    except Exception:
        for cid, positions in original.items():
            _restore_positions(mol.GetConformer(cid), positions)
        raise
    return work, original


def restrained_uff(mol, cons, *, stiffness=1.0, max_iters=MAX_ITERS, conf_ids=None):
    """Minimise conformers with frozen atoms and flat-bottomed constraint terms.

    ``stiffness`` scales the restraint walls. ``conf_ids`` restricts the operation to selected conformers.
    Metal and haptic typing changes only a private graph; relaxed real-atom coordinates return to ``mol``.
    """
    frozen = set(cons.frozen)
    confs = list(mol.GetConformers()) if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    work, original = _prepare_uff_work(mol, cons, confs, frozen, max_iters)

    typed = {}

    def build(target, conf_id):
        if target not in typed:
            probe = strip_phantoms(target, cons.phantoms)
            with rdBase.BlockLogs():
                typed[target] = rdForceFieldHelpers.UFFHasAllMoleculeParams(probe)
        if not typed[target]:
            raise RuntimeError("UFF has unsupported atom types")
        with rdBase.BlockLogs():
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target, confId=conf_id, ignoreInterfragInteractions=False)
        conf = target.GetConformer(conf_id)
        for mechanism in _mech.MECHANISM_ORDER:
            mechanism._ff_terms(ff, cons, conf, stiffness)
        ff.Initialize()
        return ff

    energies = []
    fallback = None
    retyped = 0
    try:
        for conf in confs:
            cid = conf.GetId()
            target = fallback or work
            try:
                ff = build(target, cid)
            except RuntimeError as error:
                for candidate, count in _uff_core_graphs(work, frozen):
                    try:
                        ff = build(candidate, cid)
                    except RuntimeError:
                        continue
                    fallback = target = candidate
                    retyped = count
                    break
                else:
                    raise RuntimeError("UFF has unsupported atom types outside the fixed core") from error
                log = logger.warning if max_iters else logger.debug
                log("UFF: retyped %d fixed-core bond(s) as outward dative edges", retyped)
            try:
                ff.Minimize(maxIts=max_iters)
                energy = ff.CalcEnergy()
            except RuntimeError as error:
                raise RuntimeError(f"UFF minimization failed: {_error_summary(error)}") from error
            energies.append(energy)
            if target is not mol:  # copy only real atoms off the metal/phantom/fixed-core FF graph
                src = target.GetConformer(cid)
                for atom in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(atom, src.GetAtomPosition(atom))
    except RuntimeError:
        for cid, positions in original.items():
            _restore_positions(mol.GetConformer(cid), positions)
        raise
    return np.array(energies)
