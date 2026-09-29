"""Relax conformers under constraints and reject geometrically broken results."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import GetPeriodicTable, rdForceFieldHelpers, rdtrajectory

from .mechanisms import MECHANISM_ORDER
from .metal_core import COORDINATION_METALS, materialise_phantoms, strip_phantoms
from .utils import CARBON_Z, Violation, atom_label

FF_SURROGATE = 3  # Bondless Li avoids the singular CN>=3 UFF angle term while retaining a soft vdW sphere.
UFF_GHOST = 54  # Untypeable Xe gives a haptic centroid no UFF terms inside its ring.

logger = logging.getLogger("rxembed.relax")

MAX_ITERS = 2000  # the one restrained-UFF iteration cap: every relax entry point defaults from this name
_PT = GetPeriodicTable()
_MAIN_GROUP_Z = frozenset(range(3, 89)) - COORDINATION_METALS  # every main-group element, d/f-block excluded
# The main-group element one row up with the same valence electron count; period 2 has none.
_LIGHTER_CONGENER = {
    z: y
    for z in _MAIN_GROUP_Z
    for y in _MAIN_GROUP_Z
    if _PT.GetRow(y) == _PT.GetRow(z) - 1 and _PT.GetNOuterElecs(y) == _PT.GetNOuterElecs(z)
}


class UFFTypingError(RuntimeError):
    """Report that RDKit cannot construct the requested UFF objective."""


class UFFOptimizationError(RuntimeError):
    """Report that RDKit failed while evaluating a constructed UFF objective."""


def _raise_uff_typing_error(mol, atoms):
    """Raise UFFTypingError naming the untypeable atoms and the one remedy: a different relaxation backend."""
    names = ", ".join(atom_label(mol, i) for i in atoms) or "unknown"
    raise UFFTypingError(f"UFF has unsupported atom types ({names}); use a different relaxation backend")


def error_summary(error):
    """Return the useful line from a multiline RDKit force-field error."""
    noise = ("Pre-condition Violation", "Violation occurred", "Failed Expression", "RDKIT:", "BOOST:")
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return next((line for line in lines if not line.startswith(noise)), lines[-1] if lines else "unknown error")


def bonding_failure(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset(), constrained=()):
    """Return the first heavy-atom bond or clash violation as a `Violation`, or ``None``.

    Frozen-core, metal and explicitly constrained pairs are exempt because covalent radii do not define their
    intended distances. The exemption is per pair, so broken free-periphery bonds still fail. `atoms` names the
    pair, `()` for the whole-conformer non-finite case.
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    if not np.all(np.isfinite(pos)):
        return Violation("nonfinite", (), value=float("nan"), limit=0.0, detail="non-finite coordinates")
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
                    stretched = d > bond_tol * cut
                    limit = bond_tol * cut if stretched else clash_tol * cut
                    change = "stretched" if stretched else "squeezed"
                    detail = f"bond {atom_label(mol, i)}-{atom_label(mol, j)} {change} to {d:.2f} A"
                    return Violation("bond", (i, j), value=d, limit=limit, detail=detail)
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                detail = f"{atom_label(mol, i)}...{atom_label(mol, j)} clash at {d:.2f} A"
                return Violation("clash", (i, j), value=d, limit=clash_tol * cut, detail=detail)
    return None


def _valence_pressure(mol, idx):
    """Rank an atom for a dative retype by valence violation, excess valence, then degree."""
    atom = mol.GetAtomWithIdx(idx)
    default = max(_PT.GetDefaultValence(atom.GetAtomicNum()), 0)
    excess = atom.GetValence(Chem.ValenceType.EXPLICIT) - default
    return atom.HasValenceViolation(), excess, atom.GetDegree()


def _rehybridise(rw, atoms):
    """Return a copy of `rw` with `atoms`' hybridisation re-derived from their current bonds alone."""
    for idx in atoms:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNoImplicit(True)
        atom.SetHybridization(Chem.HybridizationType.UNSPECIFIED)
    mid = rw.GetMol()
    mid.UpdatePropertyCache(strict=False)
    with rdBase.BlockLogs():
        Chem.SanitizeMol(mid, Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION, catchErrors=True)
    return Chem.RWMol(mid)


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

    directed = []
    for i, j in bonds:
        start, end = sorted((i, j), key=lambda idx: _valence_pressure(mol, idx), reverse=True)
        if mol.GetAtomWithIdx(start).HasValenceViolation():
            directed.append((start, end))

    # Try one edge first. If several atoms violate valence, retype only enough low-pressure neighbours at each
    # centre to remove its excess valence; this stays linear instead of enumerating 2**N subsets.
    choices = [[edge] for edge in directed]
    combined = []
    for start in dict.fromkeys(start for start, _end in directed):
        edges = sorted((edge for edge in directed if edge[0] == start), key=lambda e: _valence_pressure(mol, e[1]))
        combined += edges[: max(1, int(_valence_pressure(mol, start)[1]))]
    if len(combined) > 1:
        choices.append(combined)
    for chosen in choices:
        rw = Chem.RWMol(mol)
        for start, end in chosen:
            rw.RemoveBond(start, end)
        rw = _rehybridise(rw, {a for edge in chosen for a in edge})
        for start, end in chosen:
            rw.AddBond(start, end, Chem.BondType.DATIVE)
        out = rw.GetMol()
        out.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(out)
        yield out, tuple(chosen)


def _force_field(mol, props, conf_id):
    """Build MMFF94s from its molecule properties, or UFF when `props` is ``None``."""
    if props is not None:
        return rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf_id)


def ff_energies(mol, minimize=True, max_iters=MAX_ITERS, statuses=None):
    """FF energies (MMFF94s where typeable, else UFF); optimise in place first when ``minimize``.

    ``statuses``, when given, receives each conformer's optimizer status by id.
    """
    use_mmff = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
    if not use_mmff and not _uff_typeable(mol, ()):
        _raise_uff_typing_error(mol, _uff_missing_atoms(mol, ()))
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s") if use_mmff else None
    if minimize:
        recorded = {} if statuses is None else statuses
        objective = _force_field(mol, props, -1)
        res = rdForceFieldHelpers.OptimizeMoleculeConfs(mol, objective, numThreads=0, maxIters=max_iters)
        recorded.update(
            {conf.GetId(): int(status) for conf, (status, _energy) in zip(mol.GetConformers(), res, strict=True)}
        )
        _warn_unconverged(recorded, max_iters, (conf.GetId() for conf in mol.GetConformers()))
        # Refresh endpoint distances on the original field, retaining its initial nonbonded contribution list.
        return np.array([objective.CalcEnergy(tuple(c.GetPositions().ravel())) for c in mol.GetConformers()])
    return np.array([_force_field(mol, props, c.GetId()).CalcEnergy() for c in mol.GetConformers()])


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
    # A haptic centroid dummy keeps its restraints but gets no energy terms (see UFF_GHOST).
    for idx, z in [*((m, FF_SURROGATE) for m in metals), *((p, UFF_GHOST) for p in phantoms)]:
        a = rw.GetAtomWithIdx(int(idx))
        a.SetAtomicNum(z)
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


def _rejected_charge_states(mol):
    """Return atoms whose formal charge is a bookkeeping artefact UFF's typer ignores.

    An ionic-dative donor SMILES can draw a lone-pair donation as a charge-separated double or triple bond,
    such as a dithiocarbene ``[C-2]=[S+]``. Of an unequal pair, the smaller-magnitude partner carries the
    leftover charge and is re-derived (re-deriving the other fits worse); an equal pair, such as an amidinium,
    is ordinary delocalised charge and keeps its UFF type.
    """
    out = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            continue
        i, j = bond.GetBeginAtom(), bond.GetEndAtom()
        if i.GetAtomicNum() not in _MAIN_GROUP_Z or j.GetAtomicNum() not in _MAIN_GROUP_Z:
            continue
        ci, cj = i.GetFormalCharge(), j.GetFormalCharge()
        if ci == 0 or cj == 0 or (ci > 0) == (cj > 0) or abs(ci) == abs(cj):
            continue
        out.add(i.GetIdx() if abs(ci) < abs(cj) else j.GetIdx())
    return out


def _neutralise_charge_states(rw, indices):
    """Return a copy of `rw` with rejected formal charges neutralised and hybridisation re-derived.

    A double or triple bond into an `indices` atom reads as single while RDKit derives hybridisation, so the
    pi character the charge was drawn to balance does not skew the UFF type; the bond order is then restored.
    """
    restore = []
    for idx in indices:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetFormalCharge(0)
        for bond in atom.GetBonds():
            if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                restore.append((bond.GetIdx(), bond.GetBondType()))
                bond.SetBondType(Chem.BondType.SINGLE)
    rw = _rehybridise(rw, indices)
    for bond_idx, bond_type in restore:
        rw.GetBondWithIdx(bond_idx).SetBondType(bond_type)
    return rw


def _uff_surrogate_graph(mol, cons):
    """Build a valid private UFF graph and original-radius bond holds, if possible."""
    missing = _uff_missing_atoms(mol, cons.phantoms)
    rejected = _rejected_charge_states(mol)
    if not missing and not rejected:
        return None
    rw = Chem.RWMol(mol)
    if rejected:
        rw = _neutralise_charge_states(rw, rejected)
    replacements = {idx: (mol.GetAtomWithIdx(idx).GetAtomicNum(),) * 2 for idx in rejected}
    for idx in missing:
        atom = rw.GetAtomWithIdx(idx)
        real_z = atom.GetAtomicNum()
        surrogate_z = _LIGHTER_CONGENER.get(real_z)
        if surrogate_z is None and real_z == 5:  # noqa: PLR2004  boron
            # UFF has no linear boron type. One measured structure reaches this (fixtures, tmQMg sample and
            # issues): VALRAE's neutral sp boryl, whose anionic charge xyz2mol places on a ring carbon, so an
            # isoelectronic rule keyed on formal charge (B- -> C) loses it. B-H and B-B networks stay unsupported.
            if atom.GetTotalNumHs() or any(
                neighbor.GetAtomicNum() == 5  # noqa: PLR2004  boron
                for neighbor in atom.GetNeighbors()
            ):
                return None
            surrogate_z = CARBON_Z
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
                for real_z, surrogate_z in (replacements[atom] for atom in (i, j) if atom in replacements)
            )
            target = params[1] + radius_delta
            distances[(i, j)] = (target, target)
    return out, cons.copy(distances=distances), replacements


def _select_uff_graph(work, cons, frozen):
    """Select the least invasive private graph RDKit can type: native, then dative-core, then element surrogates."""
    for candidate, retyped in itertools.chain([(work, ())], _uff_core_graphs(work, frozen)):
        if _uff_typeable(candidate, cons.phantoms) and not _rejected_charge_states(candidate):
            return candidate, cons, retyped, {}
    for candidate, retyped in itertools.chain([(work, ())], _uff_core_graphs(work, frozen)):
        if (surrogate := _uff_surrogate_graph(candidate, cons)) is not None:
            target, effective_cons, replacements = surrogate
            return target, effective_cons, retyped, replacements
    missing = set(_uff_missing_atoms(work, cons.phantoms)) | _rejected_charge_states(work)
    _raise_uff_typing_error(work, sorted(missing))


@dataclass
class UFFRecord:
    """Collect what `restrained_uff` reports besides energies, across every call given the same record.

    A caller that passes a record owns the non-convergence warning; without one, `restrained_uff` warns.
    """

    statuses: dict = field(default_factory=dict)  # conformer id -> status of its last minimisation, 0 converged
    snapshots: dict | None = None  # conformer id -> frames of its last minimisation; recorded only in a dict
    surrogates: dict = field(default_factory=dict)  # atom -> (real Z, private UFF Z) element substitution
    retyped: set = field(default_factory=set)  # private donor -> acceptor dative bond edges


def _report_uff_typing(mol, retyped, replacements, record):
    """Report private typing changes at WARNING the first time `record` sees them, then at DEBUG."""
    if retyped:
        first_report = not set(retyped) <= record.retyped
        names = ", ".join(f"{atom_label(mol, start)}->{atom_label(mol, end)}" for start, end in retyped)
        (logger.warning if first_report else logger.debug)("UFF: private fixed-core dative typing for %s", names)
    if replacements:
        first_report = any(record.surrogates.get(idx) != pair for idx, pair in replacements.items())
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
    record=None,
):
    """Minimise conformers with frozen atoms and flat-bottomed constraint terms.

    ``stiffness`` scales distance, floor, stack, centroid and explicit-fix penalties; angle and dihedral
    walls stop strengthening at 1, and native UFF, target pulls and structural repairs are unchanged, so it
    is not a multiplier over the whole objective. Metal and haptic typing changes only a private graph;
    relaxed real-atom coordinates return to ``mol``. Any exception restores every selected conformer to its
    coordinates before minimisation.

    ``record``, a `UFFRecord`, receives each minimised conformer's optimizer status and frames and the private
    typing changes; a single point records no status.
    """
    frozen = set(cons.frozen)
    confs = list(mol.GetConformers()) if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    out = UFFRecord() if record is None else record
    original = {conf.GetId(): conf.GetPositions().copy() for conf in confs}
    energies = []
    try:
        work = materialise_phantoms(mol, cons.haptic)
        work = _ff_surrogate(work, cons.metals, cons.phantoms)
        target, effective_cons, dative_edges, replacements = _select_uff_graph(work, cons, frozen)
        _report_uff_typing(mol, dative_edges, replacements, out)
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
                    for mechanism in MECHANISM_ORDER:
                        mechanism.uff_terms(ff, effective_cons, target.GetConformer(cid), stiffness)
                ff.Initialize()
                stage = "minimization"
                if out.snapshots is None:
                    status = ff.Minimize(maxIts=max_iters)
                else:
                    status, frames = ff.MinimizeTrajectory(1, maxIts=max_iters)
                    trajectory = rdtrajectory.Trajectory(3, target.GetNumAtoms(), frames)
                    out.snapshots[cid] = [
                        np.array([list(snapshot.GetPoint3D(i)) for i in range(mol.GetNumAtoms())])
                        for snapshot in map(trajectory.GetSnapshot, range(len(trajectory)))
                    ]
                if max_iters:  # A single point has no optimizer convergence status.
                    out.statuses[cid] = int(status)
                # Explicit coordinates clear native distance caches left by a rejected line-search trial.
                energy = ff.CalcEnergy(ff.Positions())
            except RuntimeError as error:
                typing = stage == "force-field construction"
                error_type, remedy = (
                    (UFFTypingError, "use a different relaxation backend")
                    if typing
                    else (UFFOptimizationError, "try another seed=")
                )
                raise error_type(f"UFF {stage} failed: {error_summary(error)}; {remedy}") from error
            energies.append(energy)
            if target is not mol:  # copy only real atoms off the metal/phantom/fixed-core FF graph
                conf.SetPositions(target.GetConformer(cid).GetPositions()[: mol.GetNumAtoms()])
    except Exception:
        for cid, positions in original.items():
            mol.GetConformer(cid).SetPositions(positions)
        raise
    out.surrogates.update(replacements)
    out.retyped.update(dative_edges)
    if record is None:
        _warn_unconverged(out.statuses, max_iters, (conf.GetId() for conf in confs))
    return np.array(energies)


def _warn_unconverged(statuses, max_iters, conf_ids):
    """Warn for recorded non-converged force-field results."""
    failed = [cid for cid in conf_ids if statuses.get(cid)]
    if failed:
        logger.warning(
            "force field: %d conformer(s) did not converge in %d iterations: %s", len(failed), max_iters, failed
        )
