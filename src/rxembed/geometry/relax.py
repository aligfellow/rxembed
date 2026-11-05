"""Relax conformers by restrained UFF, alone or as a coupled reaction pair, and judge a broken bond or clash."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import GetPeriodicTable, rdDistGeom, rdForceFieldHelpers, rdMolAlign, rdtrajectory
from rdkit.Geometry import Point3D

from rxembed.geometry.mechanisms import MECHANISM_ORDER
from rxembed.graph import (
    COORDINATION_METALS,
    materialise_phantoms,
    metal_indices,
    strip_phantoms,
    surrogate_all_metals,
)
from rxembed.utils import CARBON_Z, Violation, atom_label

if TYPE_CHECKING:
    from rxembed.geometry.constraints import Constraints

FF_SURROGATE = 3  # Bondless Li avoids the singular CN>=3 UFF angle term while retaining a soft vdW sphere.
UFF_GHOST = 54  # Untypeable Xe gives a haptic centroid no UFF terms inside its ring.

logger = logging.getLogger("rxembed.relax")

MAX_ITERS = 2000  # the one restrained-UFF iteration cap: every relax entry point defaults from this name
BOND_SLACK = 0.16  # Å: clean builds settle a ligand bond within this of its native 1-2 bound; stretched ones leave it
CLASH_TOL = 0.7  # a pair inside this multiple of its covalent-radius sum has fused
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


def error_summary(error):
    """Return the useful line from a multiline RDKit force-field error."""
    noise = ("Pre-condition Violation", "Violation occurred", "Failed Expression", "RDKIT:", "BOOST:")
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return next((line for line in lines if not line.startswith(noise)), lines[-1] if lines else "unknown error")


def bond_violations(mol, pos, exclude, constrained):
    """Return each ligand bond stretched past its native 1-2 bound by more than `BOND_SLACK`, or squeezed.

    The bound is RDKit's 1-2 interval on the engine graph the seed was drawn in, so a species Mol and its engine graph
    are judged alike. Its lower end is the Lewis rest length, which charge-separated bonds and X-ray hydrogens sit far
    inside, so a squeezed bond is one under `CLASH_TOL` times its covalent sum. Metal ends, zero-order contacts, pairs
    in `exclude` and requested pairs are exempt. Pairs are named ``(i, j)`` with ``i < j``.
    """
    metals = set(metal_indices(mol))
    engine = Chem.RWMol(mol)
    for bond in [b for b in mol.GetBonds() if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= metals]:
        engine.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    engine = surrogate_all_metals(engine) if metals else mol
    with rdBase.BlockLogs():
        bm = rdDistGeom.GetMoleculeBoundsMatrix(engine, set14bounds=False, set15bounds=False, doTriangleSmoothing=False)
    requested = {frozenset(pair) for pair in constrained}
    out = []
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        elements = [bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()]
        if bond.GetBondType() == Chem.BondType.ZERO or COORDINATION_METALS.intersection(elements) or {i, j} <= exclude:
            continue
        lo = CLASH_TOL * sum(_PT.GetRcovalent(z) for z in elements)
        hi, d = bm[i, j] + BOND_SLACK, float(np.linalg.norm(pos[i] - pos[j]))
        if lo <= d <= hi or frozenset((i, j)) in requested:
            continue
        change, limit = ("stretched", hi) if d > hi else ("squeezed", lo)
        detail = f"bond {atom_label(mol, i)}-{atom_label(mol, j)} {change} to {d:.2f} A"
        out.append(Violation("bond_length", (i, j), value=d, limit=float(limit), detail=detail))
    return out


def bonding_failure(mol, conf_id, clash_tol=CLASH_TOL, exclude=frozenset(), constrained=()):
    """Return the first ligand bond off its native bound or heavy-atom clash, else None.

    A clash is a non-bonded heavy pair under `clash_tol` times its covalent sum. Exemptions are per pair, so a broken
    free-periphery bond still fails.
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    if not np.all(np.isfinite(pos)):
        return Violation("nonfinite", (), value=float("nan"), limit=0.0, detail="non-finite coordinates")
    found = bond_violations(mol, pos, exclude, constrained)
    heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=int)
    z = np.array([mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in heavy], dtype=int)
    is_metal = np.isin(z, list(COORDINATION_METALS))
    in_exclude = np.isin(heavy, list(exclude))
    rcov = np.array([_PT.GetRcovalent(int(x)) for x in z])
    cut = rcov[:, None] + rcov[None, :]
    skip = Chem.GetAdjacencyMatrix(mol, force=True).astype(bool)[np.ix_(heavy, heavy)]
    skip |= (in_exclude[:, None] & in_exclude[None, :]) | is_metal[:, None] | is_metal[None, :]
    slot = {int(a): k for k, a in enumerate(heavy)}
    for i, j in (map(slot.get, p) for p in constrained):
        if i is not None and j is not None:
            skip[i, j] = skip[j, i] = True
    dist = np.linalg.norm(pos[heavy][:, None] - pos[heavy][None], axis=-1)
    # A matrix norm may differ from the scalar norm in the last ulp: screen a hair wide, re-check with the scalar.
    for n, k in np.argwhere(np.triu((dist < clash_tol * cut * (1 + 1e-9)) & ~skip, 1)):
        i, j = int(heavy[n]), int(heavy[k])
        d, limit = float(np.linalg.norm(pos[i] - pos[j])), clash_tol * float(cut[n, k])
        if d < limit:
            detail = f"{atom_label(mol, i)}...{atom_label(mol, j)} clash at {d:.2f} A"
            found.append(Violation("clash", (i, j), value=d, limit=limit, detail=detail))
            break
    return min(found, key=lambda v: v.atoms, default=None)


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
    """Yield private FF graphs with frozen-core bonds made dative, so a hypervalent electrophile stays typeable."""
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

    # retype only enough low-pressure neighbours at each centre, linear rather than 2**N subsets
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


def _ff_surrogate(mol, metals, phantoms=()):
    """Retype unsupported centres and remove non-valence contacts on a private UFF graph."""
    metals = {int(m) for m in metals}
    partial = [  # zero-order contacts, and dative bonds to a restrained metal, which its windows hold
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if not bond.GetBondTypeAsDouble()
        or (bond.GetBondType() == Chem.BondType.DATIVE and {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & metals)
    ]
    if not metals and not phantoms and not partial:
        return mol  # `mol` itself, so the organic path is a strict no-op
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

    A dative donor SMILES can draw a lone-pair donation as a charge-separated multiple bond, such as ``[C-2]=[S+]``; the
    smaller-magnitude partner of an unequal pair is re-derived, while an equal pair (an amidinium) is ordinary charge.
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
    """Return a copy of `rw` with rejected charges neutralised and hybridisation re-derived, bond orders kept."""
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
    """Return ``(graph, cons, replacements)``: untypeable atoms swapped for typeable ones, bonds held at real radii.

    A missing type takes its lighter congener, or carbon for a lone boron; a rejected charge state is neutralised. None
    when nothing needs a change or no swap types the graph.
    """
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
        # no linear boron UFF type; a boryl charge may sit on a ring C, so the rule cannot key on B-
        if (
            surrogate_z is None
            and real_z == 5  # noqa: PLR2004  boron
            and not atom.GetTotalNumHs()
            and not any(neighbor.GetAtomicNum() == 5 for neighbor in atom.GetNeighbors())  # noqa: PLR2004  boron
        ):
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
    return out, replace(cons, distances=distances), replacements


def _select_uff_graph(work, cons, frozen):
    """Return the least invasive private graph RDKit types, its constraints and typing changes, else raise.

    The native graph comes first, then a dative frozen core, then element surrogates; none raises `UFFTypingError`.
    """
    for candidate, retyped in itertools.chain([(work, ())], _uff_core_graphs(work, frozen)):
        if _uff_typeable(candidate, cons.phantoms) and not _rejected_charge_states(candidate):
            return candidate, cons, retyped, {}
    for candidate, retyped in itertools.chain([(work, ())], _uff_core_graphs(work, frozen)):
        if (surrogate := _uff_surrogate_graph(candidate, cons)) is not None:
            target, effective_cons, replacements = surrogate
            return target, effective_cons, retyped, replacements
    missing = set(_uff_missing_atoms(work, cons.phantoms)) | _rejected_charge_states(work)
    names = ", ".join(atom_label(work, i) for i in sorted(missing)) or "unknown"
    raise UFFTypingError(f"UFF has unsupported atom types ({names}); use a different relaxation backend")


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


class UFFTarget(NamedTuple):
    """Hold the private graph restrained UFF relaxes a Mol on, the constraints read there and its typing changes."""

    mol: Chem.Mol
    cons: Constraints
    retyped: tuple
    replacements: dict


def _uff_target(mol, cons, record):
    """Return the `UFFTarget` of `mol` under `cons`, its typing changes reported to `record` (`_report_uff_typing`).

    Centroids are materialised, metals retyped and the least invasive graph RDKit types is selected.
    """
    work = _ff_surrogate(materialise_phantoms(mol, cons.haptic), cons.metals, cons.phantoms)
    target = UFFTarget(*_select_uff_graph(work, cons, set(cons.frozen)))
    _report_uff_typing(mol, target.retyped, target.replacements, record)
    return target


def _restrained_field(target, cid, stiffness):
    """Return the initialised UFF force field of conformer `cid` of a `UFFTarget`, every mechanism's terms added.

    A field RDKit cannot build raises `UFFTypingError`; one whose restraint terms fail, `UFFOptimizationError`.
    """
    stage = "force-field construction"
    try:
        with rdBase.BlockLogs():
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target.mol, confId=cid, ignoreInterfragInteractions=False)
            stage = "constraint setup"
            for mechanism in MECHANISM_ORDER:
                mechanism.uff_terms(ff, target.cons, target.mol.GetConformer(cid), stiffness)
            ff.Initialize()
    except RuntimeError as error:
        typing = stage == "force-field construction"
        remedy = "use a different relaxation backend" if typing else "try another seed="
        failure = UFFTypingError if typing else UFFOptimizationError
        raise failure(f"UFF {stage} failed: {error_summary(error)}; {remedy}") from error
    return ff


def restrained_uff(
    mol,
    cons,
    *,
    stiffness=1.0,
    max_iters=MAX_ITERS,
    conf_ids=None,
    record=None,
):
    """Minimise conformers `conf_ids` (default all) under `cons` and return their final energies.

    ``stiffness`` scales the distance, floor, stack, centroid and fix penalties; angle and dihedral walls stop at 1, and
    native UFF, target pulls and repairs are unchanged, so it does not scale the whole objective. Typing changes affect
    only a private graph. Any exception restores every selected conformer. ``record`` receives each conformer's
    optimizer status and frames and the typing changes.
    """
    confs = list(mol.GetConformers()) if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    out = UFFRecord() if record is None else record
    original = {conf.GetId(): conf.GetPositions().copy() for conf in confs}
    energies = []
    try:
        target = _uff_target(mol, cons, out)
        for conf in confs:
            cid = conf.GetId()
            ff = _restrained_field(target, cid, stiffness)
            try:
                if out.snapshots is None:
                    status = ff.Minimize(maxIts=max_iters)
                else:
                    status, frames = ff.MinimizeTrajectory(1, maxIts=max_iters)
                    trajectory = rdtrajectory.Trajectory(3, target.mol.GetNumAtoms(), frames)
                    out.snapshots[cid] = [
                        np.array([list(snapshot.GetPoint3D(i)) for i in range(mol.GetNumAtoms())])
                        for snapshot in map(trajectory.GetSnapshot, range(len(trajectory)))
                    ]
                if max_iters:  # A single point has no optimizer convergence status.
                    out.statuses[cid] = int(status)
                # Explicit coordinates clear native distance caches left by a rejected line-search trial.
                energies.append(ff.CalcEnergy(ff.Positions()))
            except RuntimeError as error:
                raise UFFOptimizationError(
                    f"UFF minimization failed: {error_summary(error)}; try another seed="
                ) from error
            if target.mol is not mol:  # copy only real atoms off the metal/phantom/fixed-core FF graph
                conf.SetPositions(target.mol.GetConformer(cid).GetPositions()[: mol.GetNumAtoms()])
    except Exception:
        for cid, positions in original.items():
            mol.GetConformer(cid).SetPositions(positions)
        raise
    out.surrogates.update(target.replacements)
    out.retyped.update(target.retyped)
    if record is None and (failed := [conf.GetId() for conf in confs if out.statuses.get(conf.GetId())]):
        logger.warning("UFF: %d conformer(s) did not converge in %d iterations: %s", len(failed), max_iters, failed)
    return np.array(energies)


class Energy(NamedTuple):
    """Hold an energy over two endpoints and its gradient on each endpoint's coordinates."""

    value: float
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class CoEmbedParams:
    """Hold the coupled relaxation's weights, stopping tolerances and step budgets for `rx.coembed`, validated.

    `alignment_weight` and `react_weight` pull spectator and reacting atoms together, each normalized over its group.
    `joint_gtol` and `joint_energy_tol` stop the joint relaxation; a score plateau gives no force guarantee.
    `seed_relax_iters` caps each endpoint's preparation, zero skipping it, and each endpoint takes at most
    `joint_max_iters` native steps per cycle for `max_cycles` cycles.
    """

    alignment_weight: float = 25.0
    react_weight: float = 1.25
    joint_gtol: float = 0.1
    joint_energy_tol: float = 1e-3
    seed_relax_iters: int = 100
    joint_max_iters: int = 100
    max_cycles: int = 100

    def __post_init__(self):
        """Refuse invalid weights, tolerances or iteration budgets."""
        for name in ("alignment_weight", "react_weight", "joint_gtol", "joint_energy_tol"):
            value = getattr(self, name)
            least = "nonnegative" if name == "joint_energy_tol" else "positive"
            number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
            if not (number and np.isfinite(value) and (value > 0 or (value == 0 and least == "nonnegative"))):
                raise ValueError(f"{name} must be finite and {least}")
        for name in ("seed_relax_iters", "joint_max_iters", "max_cycles"):
            value = getattr(self, name)
            minimum = 0 if name == "seed_relax_iters" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                least = "nonnegative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {least} integer")


class Coupling(NamedTuple):
    """Hold which atoms couple two reaction endpoints that share one atom order, the first `spectators` the spectators.

    An atom costs its weight (kcal/mol/Å²) times its squared displacement between endpoints.
    """

    atoms: np.ndarray
    weights: np.ndarray
    spectators: int

    def alignment_rmsd(self, left, right):
        """Measure spectator RMSD in the shared frame without fitting."""
        atoms = self.atoms[: self.spectators]
        delta = left[atoms] - right[atoms]
        return float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))) if len(atoms) else 0.0

    def penalty(self, left, right):
        """Return the coupling's `Energy` at reactant and product coordinates `left` and `right`."""
        delta = left[self.atoms] - right[self.atoms]
        force = 2 * self.weights[:, None] * delta
        on_left, on_right = np.zeros_like(left), np.zeros_like(right)
        np.add.at(on_left, self.atoms, force)
        np.add.at(on_right, self.atoms, -force)
        return Energy(float(np.sum(self.weights[:, None] * delta**2)), on_left, on_right)


class Endpoint(NamedTuple):
    """Hold one endpoint of a coupled relax: its engine Mol, its `Constraints` and the `UFFRecord` it reports to."""

    mol: Chem.Mol
    cons: Constraints
    record: UFFRecord


class Joint(NamedTuple):
    """Retain a pair's seed-derived native objective and its optimizer diagnostic for consistent final scoring."""

    targets: list
    fields: list | None  # None when the objective could not be built
    record: dict

    def energies(self, positions):
        """Evaluate retained fields, rebuilding centroids when validation changes their real coordinates.

        Without fields, as after a failure to build them, every endpoint scores inf.
        """
        if self.fields is None:
            return [math.inf] * len(self.targets)
        values = []
        for target, ff, pos in zip(self.targets, self.fields, positions, strict=True):
            numerical = target.mol.GetConformer().GetPositions()
            if not np.array_equal(numerical[: len(pos)], pos):
                for dummy, ring in target.cons.haptic.items():
                    numerical[dummy] = pos[list(ring)].mean(axis=0)
            numerical[: len(pos)] = pos
            values.append(ff.CalcEnergy(tuple(numerical.ravel())))
        return values


def coupled_uff(endpoints, coupling, ids, stiffness, params, *, fixed=None):
    """Relax paired ids on one endpoint-UFF plus correspondence objective, kept for scoring.

    Spectators are registered only when that lowers the coupling, and `fixed` is kept exactly. A pair whose objective
    cannot be built or run keeps its start geometry with an infinite score; endpoint acceptance is the caller's.
    """
    targets = [_uff_target(endpoint.mol, endpoint.cons, endpoint.record) for endpoint in endpoints]
    results = {}
    for cid in ids:
        local = [target._replace(mol=Chem.Mol(target.mol, confId=cid)) for target in targets]
        pos = [target.mol.GetConformer(cid).GetPositions() for target in local]
        pos = _aligned_pair(local, coupling, pos, cid, fixed)
        fields = None
        try:
            fields = [_restrained_field(target, cid, stiffness) for target in local]
            relaxed, record = _coupled(Joint(local, fields, {}), coupling, cid, stiffness, params, fixed)
            if not all(np.isfinite(positions).all() for positions in relaxed):
                raise FloatingPointError("non-finite joint coordinates")
        except (RuntimeError, FloatingPointError) as error:
            relaxed = pos
            record = {
                "converged": False,
                "stopping_reason": "failure",
                "message": error_summary(error),
                "cycles": 0,
                "iterations": None,
                "iteration_budget_used": None,
                "max_force": math.inf,
                "gtol": params.joint_gtol,
                "energy_tol": params.joint_energy_tol,
                "score_change": math.inf,
                "score": math.inf,
                "score_history": (),
                "seconds": 0.0,
            }
            logger.warning(
                "coembed: joint optimization failed for pair %d; restored its seed (%s)", cid, error_summary(error)
            )
        for endpoint, target, positions in zip(endpoints, local, relaxed, strict=True):
            endpoint.mol.GetConformer(cid).SetPositions(positions[: endpoint.mol.GetNumAtoms()])
            target.mol.GetConformer(cid).SetPositions(positions)
        results[cid] = Joint(local, fields, {**record, "stiffness": stiffness})
    for endpoint, target in zip(endpoints, targets, strict=True):
        endpoint.record.surrogates.update(target.replacements)
        endpoint.record.retyped.update(target.retyped)
    return results


def _coupled(objective, coupling, cid, stiffness, params, fixed):
    """Alternate native endpoint updates on the retained objective within separate step and cycle budgets."""
    targets, fields = objective.targets, objective.fields
    started = perf_counter()
    pos = [target.mol.GetConformer(cid).GetPositions() for target in targets]
    moving = tuple(side for side in (0, 1) if side != fixed)
    joint = _joint(fields, coupling, pos)
    force = _joint_force(joint, targets, moving)
    cycles, iterations, change = 0, 0, math.inf
    reason = "force" if force <= params.joint_gtol else "cycle_limit"
    history = [joint.value]
    while force > params.joint_gtol and cycles < params.max_cycles:
        previous, previous_iterations = joint.value, iterations
        completed = True
        for side in moving:
            held = targets[side]._replace(mol=Chem.Mol(targets[side].mol))
            ff = _held_field(held, coupling, side, pos, cid, stiffness)
            _status, frames = ff.MinimizeTrajectory(1, maxIts=params.joint_max_iters)
            iterations += len(frames)
            trial = list(pos)
            trial[side] = np.reshape(ff.Positions(), (-1, 3))
            updated = _joint(fields, coupling, trial)
            if updated.value > joint.value + 1e-8 * max(1.0, abs(joint.value)):
                reason, completed = "stalled", False
                break
            pos, joint = trial, updated
            force = _joint_force(joint, targets, moving)
        if not completed:
            break
        cycles += 1
        change = previous - joint.value
        history.append(joint.value)
        if force > params.joint_gtol and iterations == previous_iterations:
            reason = "stalled"
        elif force > params.joint_gtol and params.joint_energy_tol > 0 and 0 <= change <= params.joint_energy_tol:
            reason = "energy"
        else:
            continue
        break
    if force <= params.joint_gtol:  # met after a full cycle or after a stalled second update
        reason = "force"
    return pos, {
        "converged": reason in ("force", "energy"),
        "stopping_reason": reason,
        "message": {
            "force": "joint force tolerance reached",
            "energy": "joint score plateau reached",
            "cycle_limit": "joint cycle limit reached",
            "stalled": "native updates could not advance the joint objective",
        }[reason],
        "cycles": cycles,
        "iterations": iterations,
        "iteration_budget_used": iterations,
        "max_force": force,
        "gtol": params.joint_gtol,
        "energy_tol": params.joint_energy_tol,
        "score_change": change,
        "score": joint.value,
        "score_history": tuple(history),
        "seconds": perf_counter() - started,
    }


def _joint_force(joint, targets, moving):
    """Measure the largest joint-gradient component on free endpoint coordinates."""
    largest = 0.0
    for side in moving:
        gradient = (joint.left, joint.right)[side].copy()
        gradient[list(targets[side].cons.frozen)] = 0.0
        largest = max(largest, float(np.max(np.abs(gradient))))
    return largest


def _aligned_pair(targets, coupling, positions, cid, fixed):
    """Return the pair fitted by its spectators when that lowers its penalty, preserving the fixed frame."""
    side = 0 if fixed == 1 else 1
    fit = Chem.Mol(targets[side].mol)
    held = [(int(atom), int(atom)) for atom in coupling.atoms[: coupling.spectators]]
    rdMolAlign.AlignMol(fit, targets[1 - side].mol, prbCid=cid, refCid=cid, atomMap=held)
    trial = list(positions)
    trial[side] = fit.GetConformer(cid).GetPositions()
    if coupling.penalty(*trial).value <= coupling.penalty(*positions).value:
        targets[side].mol.GetConformer(cid).SetPositions(trial[side])
        return trial
    return positions


def _held_field(target, coupling, side, pos, cid, stiffness):
    """Return endpoint `side`'s field at `pos`, each coupled atom held toward its place in the other.

    Each hold is an RDKit position restraint at the other endpoint's coordinates with twice its weight, RDKit's half-k
    convention.
    """
    ff = _restrained_field(target, cid, stiffness)
    conf = target.mol.GetConformer(cid)
    for atom, weight in zip(coupling.atoms, coupling.weights, strict=True):
        conf.SetAtomPosition(int(atom), Point3D(*pos[1 - side][atom]))
        ff.UFFAddPositionConstraint(int(atom), 0.0, float(2 * weight))
    conf.SetPositions(pos[side])
    ff.Initialize()
    return ff


def _joint(fields, coupling, pos):
    """Return the coupled objective's `Energy` at `pos`: both endpoint fields' energies plus the coupling's."""
    coupled = coupling.penalty(*pos)
    flat = [tuple(p.ravel()) for p in pos]
    value = coupled.value + sum(ff.CalcEnergy(x) for ff, x in zip(fields, flat, strict=True))
    left, right = (
        g + np.reshape(ff.CalcGrad(x), (-1, 3))
        for g, ff, x in zip((coupled.left, coupled.right), fields, flat, strict=True)
    )
    if not (np.isfinite(value) and np.isfinite(left).all() and np.isfinite(right).all()):
        raise FloatingPointError("non-finite joint energy or gradient")
    return Energy(value, left, right)
