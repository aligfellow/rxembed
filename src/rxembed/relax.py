"""Relax conformers under constraints and reject geometrically broken results."""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import GetPeriodicTable, rdForceFieldHelpers, rdMolTransforms
from rdkit.Chem import rdtrajectory as _rdtrajectory

from . import mechanisms as _mech
from .metal_core import COORDINATION_METALS, materialise_phantoms, strip_phantoms

FF_SURROGATE = 3  # Bondless Li avoids the singular CN>=3 UFF angle term while retaining a soft vdW sphere.
UFF_GHOST = 54  # Untypeable Xe gives a haptic centroid no UFF terms inside its ring.

logger = logging.getLogger("rxembed.relax")

MAX_ITERS = 2000  # the one restrained-UFF iteration cap: every relax entry point defaults from this name
_DIHEDRAL_ATOMS = 4
_PT = GetPeriodicTable()
_BORON_Z, _CARBON_Z = 5, 6
_MAIN_GROUPS = (
    (3, 11, 19, 37, 55, 87),
    (4, 12, 20, 38, 56, 88),
    (5, 13, 31, 49, 81),
    (6, 14, 32, 50, 82),
    (7, 15, 33, 51, 83),
    (8, 16, 34, 52, 84),
    (9, 17, 35, 53, 85),
    (10, 18, 36, 54, 86),
)
_LIGHTER_CONGENER = {z: group[i - 1] for group in _MAIN_GROUPS for i, z in enumerate(group) if i}


class UFFTypingError(RuntimeError):
    """Report that RDKit cannot construct the requested UFF objective."""


class UFFOptimizationError(RuntimeError):
    """Report that RDKit failed while evaluating a constructed UFF objective."""


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
    return _bonding_failure(mol, conf_id, bond_tol, clash_tol, exclude, constrained) is None


def _bonding_failure(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset(), constrained=()):
    """Return the first heavy-atom bond or clash violation, or ``None``."""
    pos = mol.GetConformer(conf_id).GetPositions()
    if not np.all(np.isfinite(pos)):
        return "non-finite coordinates"
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
                    relation = "above" if d > bond_tol * cut else "below"
                    limit = bond_tol * cut if relation == "above" else clash_tol * cut
                    return f"bond {i}-{j} {d:.3f} A {relation} {limit:.3f} A"
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                return f"clash {i}-{j} {d:.3f} A below {clash_tol * cut:.3f} A"
    return None


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
        yield out, tuple(chosen)


def ff_energies(mol, minimize=True, max_iters=MAX_ITERS, _statuses=None):
    """FF energies (MMFF94s where typeable, else UFF); optimise in place first when ``minimize``."""
    use_mmff = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
    if not use_mmff and not _uff_typeable(mol, ()):
        missing = _uff_missing_atoms(mol, ())
        atoms = ", ".join(f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in missing) or "unknown"
        raise UFFTypingError(
            f"UFF has unsupported atom types ({atoms}); use restrained metal relaxation or another backend"
        )
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s") if use_mmff else None

    def ff(c):
        return (
            rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=c)
            if use_mmff
            else rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=c)
        )

    if minimize:
        statuses = {} if _statuses is None else _statuses
        objective = ff(-1)
        res = rdForceFieldHelpers.OptimizeMoleculeConfs(mol, objective, numThreads=0, maxIters=max_iters)
        statuses.update(
            {conf.GetId(): int(status) for conf, (status, _energy) in zip(mol.GetConformers(), res, strict=True)}
        )
        _warn_unconverged(statuses, max_iters, (conf.GetId() for conf in mol.GetConformers()))
        # Refresh endpoint distances on the original field, retaining its initial nonbonded contribution list.
        return np.array([objective.CalcEnergy(tuple(c.GetPositions().ravel())) for c in mol.GetConformers()])
    return np.array([ff(c.GetId()).CalcEnergy() for c in mol.GetConformers()])


def _ff_surrogate(mol, metals, phantoms=()):
    """Retype unsupported centres and remove non-valence contacts on a private UFF graph."""
    metals = {int(m) for m in metals}
    partial = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if not bond.GetBondTypeAsDouble()
    ]
    if not metals and not phantoms and not partial:
        return mol  # an organic system: the identical object, so this whole path is a strict no-op
    rw = Chem.RWMol(mol)  # copies the conformers
    for begin, end in partial:
        rw.RemoveBond(begin, end)  # UFF requires positive bond order; a partial contact carries no bond energy
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


def _uff_typeable(mol, phantoms):
    """Return whether RDKit has a UFF atom type for every real atom."""
    with rdBase.BlockLogs():
        return rdForceFieldHelpers.UFFHasAllMoleculeParams(strip_phantoms(mol, phantoms))


def _uff_missing_atoms(mol, phantoms):
    """Return atom indices lacking a UFF type through RDKit's per-atom parameter lookup."""
    probe = strip_phantoms(mol, phantoms)
    with rdBase.BlockLogs():
        return tuple(
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if rdForceFieldHelpers.GetUFFVdWParams(probe, atom.GetIdx(), atom.GetIdx()) is None
        )


def _uff_surrogate_graph(mol, cons):
    """Build a valid private UFF graph and original-radius bond holds, if possible."""
    missing = _uff_missing_atoms(mol, cons.phantoms)
    if not missing:
        return None
    rw = Chem.RWMol(mol)
    replacements = {}
    for idx in missing:
        atom = rw.GetAtomWithIdx(idx)
        real_z = atom.GetAtomicNum()
        surrogate_z = _LIGHTER_CONGENER.get(real_z)
        if surrogate_z is None and real_z == _BORON_Z:
            # RDKit has no type for some isolated, hypervalent boron forms.  Carbon is only
            # a private fallback here; boron-hydrogen and boron-boron networks remain unsupported.
            if atom.GetTotalNumHs() or any(neighbor.GetAtomicNum() == _BORON_Z for neighbor in atom.GetNeighbors()):
                return None
            surrogate_z = _CARBON_Z
        if surrogate_z is None:
            return None
        atom.SetAtomicNum(surrogate_z)
        replacements[idx] = (real_z, surrogate_z)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    if not _uff_typeable(out, cons.phantoms):
        return None

    distances = dict(cons.distances)
    affected = {
        bond.GetIdx()
        for idx in replacements
        for bond in out.GetAtomWithIdx(idx).GetBonds()
        if bond.GetBondType() != Chem.BondType.DATIVE
    }
    with rdBase.BlockLogs():
        for bond_idx in affected:
            bond = out.GetBondWithIdx(bond_idx)
            i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if (i, j) in distances or (j, i) in distances:
                continue
            params = rdForceFieldHelpers.GetUFFBondStretchParams(out, i, j)
            if params is None:
                return None
            radius_delta = sum(
                _PT.GetRcovalent(real_z) - _PT.GetRcovalent(surrogate_z)
                for atom in (i, j)
                for real_z, surrogate_z in (replacements.get(atom, (0, 0)),)
                if real_z
            )
            target = params[1] + radius_delta
            distances[(i, j)] = (target, target)
    return out, cons.copy(distances=distances), replacements


def _select_uff_graph(work, cons, frozen):
    """Select the least invasive private graph that RDKit can type."""
    if _uff_typeable(work, cons.phantoms):
        return work, cons, (), {}
    for candidate, retyped in _uff_core_graphs(work, frozen):
        if _uff_typeable(candidate, cons.phantoms):
            return candidate, cons, retyped, {}
    fallback = _uff_surrogate_graph(work, cons)
    if fallback is not None:
        target, effective_cons, replacements = fallback
        return target, effective_cons, (), replacements
    for candidate, retyped in _uff_core_graphs(work, frozen):
        fallback = _uff_surrogate_graph(candidate, cons)
        if fallback is not None:
            target, effective_cons, replacements = fallback
            return target, effective_cons, retyped, replacements
    missing = _uff_missing_atoms(work, cons.phantoms)
    atoms = ", ".join(f"{work.GetAtomWithIdx(i).GetSymbol()}{i}" for i in missing) or "unknown"
    raise UFFTypingError(f"UFF has unsupported atom types ({atoms}); use a different relaxation backend")


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
                conf.SetPositions(before)  # a coordinate graft is stricter than a numeric torsion


def _report_uff_typing(mol, retyped, replacements, previous_retyped, previous_surrogates):
    """Report private typing changes once per relaxation workflow."""
    if retyped:
        first_report = previous_retyped is None or not set(retyped) <= previous_retyped
        names = ", ".join(
            f"{mol.GetAtomWithIdx(start).GetSymbol()}{start}->{mol.GetAtomWithIdx(end).GetSymbol()}{end}"
            for start, end in retyped
        )
        (logger.warning if first_report else logger.debug)("UFF: private fixed-core dative typing for %s", names)
    if replacements:
        first_report = previous_surrogates is None or any(
            previous_surrogates.get(idx) != pair for idx, pair in replacements.items()
        )
        names = ", ".join(
            f"{_PT.GetElementSymbol(real_z)}{idx}->{_PT.GetElementSymbol(surrogate_z)}"
            for idx, (real_z, surrogate_z) in sorted(replacements.items())
        )
        (logger.warning if first_report else logger.debug)(
            "UFF: private surrogate typing for %s; radius-corrected bonded terms", names
        )


def restrained_uff(
    mol,
    cons,
    *,
    stiffness=1.0,
    max_iters=MAX_ITERS,
    conf_ids=None,
    _snapshots=None,
    _statuses=None,
    _surrogates=None,
    _retyped=None,
):
    """Minimise conformers with frozen atoms and flat-bottomed constraint terms.

    ``stiffness`` scales distance, floor, stack, centroid and explicit-fix penalties. Ordinary angle and
    dihedral walls stop strengthening at 1. Native UFF, target pulls and structural repairs are unchanged;
    this is not a multiplier for the entire objective. ``conf_ids`` selects the conformers to operate on.
    Metal and haptic typing changes only a private graph; relaxed real-atom coordinates return to ``mol``.
    Internal callers may collect convergence codes and private graph substitutions. Any exception restores
    the selected conformers to their coordinates before torsion seating or minimization.
    """
    frozen = set(cons.frozen)
    confs = list(mol.GetConformers()) if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    statuses = {} if _statuses is None else _statuses
    original = {conf.GetId(): conf.GetPositions().copy() for conf in confs}
    energies = []
    try:
        _seat_fixed_dihedrals(confs, cons.fixed if max_iters else {}, frozen)
        work = materialise_phantoms(mol, cons.haptic)
        work = _ff_surrogate(work, cons.metals, cons.phantoms)
        target, effective_cons, retyped, replacements = _select_uff_graph(work, cons, frozen)
        _report_uff_typing(mol, retyped, replacements, _retyped, _surrogates)
        for conf in confs:
            cid = conf.GetId()
            stage = "force-field construction"
            try:
                with rdBase.BlockLogs():
                    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
                        target, confId=cid, ignoreInterfragInteractions=False
                    )
                stage = "constraint setup"
                with rdBase.BlockLogs():
                    for mechanism in _mech.MECHANISM_ORDER:
                        mechanism._ff_terms(ff, effective_cons, target.GetConformer(cid), stiffness)
                ff.Initialize()
                stage = "minimization"
                if _snapshots is None:
                    status = ff.Minimize(maxIts=max_iters)
                else:
                    status, snapshots = ff.MinimizeTrajectory(1, maxIts=max_iters)
                    trajectory = _rdtrajectory.Trajectory(3, target.GetNumAtoms(), snapshots)
                    _snapshots[cid] = [
                        np.array(
                            [
                                [snapshot.GetPoint3D(i).x, snapshot.GetPoint3D(i).y, snapshot.GetPoint3D(i).z]
                                for i in range(mol.GetNumAtoms())
                            ]
                        )
                        for snapshot in (trajectory.GetSnapshot(i) for i in range(len(trajectory)))
                    ]
                if max_iters:  # A single point has no optimizer convergence status.
                    statuses[cid] = int(status)
                # Explicit coordinates clear native distance caches left by a rejected line-search trial.
                energy = ff.CalcEnergy(ff.Positions())
            except RuntimeError as error:
                error_type = UFFTypingError if stage == "force-field construction" else UFFOptimizationError
                raise error_type(f"UFF {stage} failed: {_error_summary(error)}") from error
            energies.append(energy)
            if target is not mol:  # copy only real atoms off the metal/phantom/fixed-core FF graph
                conf.SetPositions(target.GetConformer(cid).GetPositions()[: mol.GetNumAtoms()])
    except Exception:
        for cid, positions in original.items():
            mol.GetConformer(cid).SetPositions(positions)
        raise
    if _surrogates is not None:
        _surrogates.update(replacements)
    if _retyped is not None:
        _retyped.update(retyped)
    if _statuses is None:
        _warn_unconverged(statuses, max_iters, (conf.GetId() for conf in confs))
    return np.array(energies)


def _warn_unconverged(statuses, max_iters, conf_ids):
    """Warn for recorded non-converged force-field results."""
    failed = [cid for cid in conf_ids if statuses.get(cid)]
    if failed:
        logger.warning(
            "force field: %d conformer(s) did not converge in %d iterations: %s", len(failed), max_iters, failed
        )
