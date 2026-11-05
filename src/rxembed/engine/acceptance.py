"""Judge conformers through the core acceptance gates (`CORE_GATES`), name why one failed and offer a remedy."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from rxembed.geometry.constraints import (
    DIST_ATOMS,
    FIX_ANGLE_TOL,
    FIX_DISTANCE_TOL,
    constraint_value,
    metal_distance_tolerance,
    out_of_plane,
    requested_dihedral,
    within_window,
)
from rxembed.geometry.relax import bonding_failure
from rxembed.identity.reading import compared
from rxembed.identity.species import shape_readings
from rxembed.metal.polyhedra import record
from rxembed.utils import atom_label

if TYPE_CHECKING:
    from collections.abc import Callable


class EmbeddingError(ValueError):
    """Report that bounded conformer generation could not build a requested species.

    The message is one sentence naming the species; `failures` counts rejected seeds by `Failure.key`.
    """

    def __init__(self, message, failures=None):
        """Keep the one-sentence message with the rejected-seed counts."""
        super().__init__(message)
        self.failures = Counter() if failures is None else failures


@dataclass(frozen=True)
class Failure:
    """Hold one rejection: its `kind`, its user-facing `detail` and the `atoms` it was read at.

    `atoms` is empty for a whole-molecule failure; replacement's two-batch stop rule reads it.
    """

    kind: str
    detail: str
    atoms: tuple = ()

    @property
    def key(self):
        """Return ``(kind, atoms)``, which groups the same failure at the same site across seeds."""
        return self.kind, self.atoms

    def __str__(self):
        """Return the user-facing explanation."""
        return self.detail


# the remedies each `Failure.kind` offers after "try"; an unlisted kind offers another isomer
_ISOMER, _LOOSER_FIX, _LOOSER_CONSTRAIN = "another isomer", "a looser fix=", "a looser constrain="
_REMEDY = {
    "coordination_shape": ("geometry=", _ISOMER),
    "ml_distance": (_ISOMER,),
    "metal_hand": ("the other hand from rx.isomers",),
    "bonding": (_LOOSER_FIX, _ISOMER),
    "numeric_fix": (_LOOSER_FIX, "constrain="),
    "connectivity": (_LOOSER_FIX, _ISOMER),
    "requested_stereo": ("stereo='free'", _ISOMER),
    "locked_stereo": ("rx.isomers(..., stereo={'locked': 'racemic'})", _ISOMER),
    "seeding": (_LOOSER_FIX, _LOOSER_CONSTRAIN, _ISOMER),
    "uff": ("a different relaxation backend",),
}


def remedy(kind, species, cons):
    """Return one ``try ...`` clause from the remedies for `kind` that this request can use, else a new seed."""
    usable = {
        _ISOMER: bool(species.centres),  # another isomer exists only for a species with centres
        _LOOSER_FIX: bool(cons.frozen or cons.fixed),
        _LOOSER_CONSTRAIN: bool(cons.contacts),
    }
    options = [option for option in _REMEDY.get(kind, (_ISOMER,)) if usable.get(option, True)]
    return f"try {' or '.join(options or ['another seed='])}"


def group_failures(failures):
    """Count failures by `Failure.key`, each group keyed by its first `Failure`."""
    first, counts = {}, Counter()
    for failure in failures:
        counts[first.setdefault(failure.key, failure)] += 1
    return counts


def describe_groups(groups):
    """Return grouped failures as ``'3x detail, 1x detail'``, most common first."""
    return ", ".join(f"{count}x {failure}" for failure, count in groups.most_common())


def headline(groups, species, cons):
    """Return the most common failure, its share of the rejected seeds and its remedy as one clause."""
    failure, count = groups.most_common(1)[0]
    return f"{failure} in {count}/{sum(groups.values())} rejected seeds; {remedy(failure.kind, species, cons)}"


EPS = 1e-6  # near-zero norm floor for the graft axis


# Each refused `ShapeReading.verdict` as the gate reports it.
_SHAPE_MISSES = {
    "unreadable": "{centre} has an undefined donor direction",
    "other_shape": "{centre} relaxes from {code} ({residual:.3f}) to {next_code} ({next_residual:.3f}{approximate})",
    "distorted": "{centre} distorts far from {code} ({residual:.2f}{approximate})",
}


def coordination_state_failure(mol, cid, specification, *, require_fit=True):
    """Return the first reason conformer `cid` misses the polyhedron and seating `specification` gives.

    Each centre must take its polyhedron and seat its donors as specified or as the mirror image. `require_fit` also
    requires the model-fit limit, for geometry rxembed builds; a calculator's geometry only needs the species.
    """
    readings = shape_readings(mol, specification.species, int(cid))
    for state in specification.species.centres:
        reading = readings[state.atom]
        centre = atom_label(specification.mol, state.atom)
        if not (reading.accepted if require_fit else reading.matches):
            detail = _SHAPE_MISSES[reading.verdict].format(
                centre=centre,
                code=record(state.geometry).code,
                residual=reading.residual,
                next_code=None if reading.next_name is None else record(reading.next_name).code,
                next_residual=reading.next_residual,
                approximate="" if reading.exact else ", approximate",
            )
            return Failure("coordination_shape", detail, atoms=(state.atom,))
    if crossed := next((miss for miss in conformer_misses(specification, mol, cid) if miss.kind == "seating"), None):
        centre = atom_label(specification.mol, crossed.atoms[0])
        return Failure("seating_crossed", f"{centre} donors swap sites (another isomer)", atoms=crossed.atoms)
    return None


def shape_report(mol, cid, specification):
    """Return how conformer `cid` reads each centre's polyhedron, with the input's own reading, or None for an organic.

    One clause per centre, ``'Pd4 SPL 0.070 (next SEE 0.322)'``, joined by `` | ``.
    """
    readings = shape_readings(mol, specification.species, int(cid))
    lines = []
    for state in specification.species.centres:
        line = f"{atom_label(specification.mol, state.atom)} {readings[state.atom].clause(state.geometry)}"
        if (on_input := specification.inputs.get(state.atom)) is not None:
            line += f"; input {on_input.clause(state.geometry)}"
        lines.append(line)
    return " | ".join(lines) or None


def conformer_misses(specification, mol, cid):
    """Return each element `specification` gives that conformer `cid` shows otherwise, cached per geometry."""
    coords = mol.GetConformer(int(cid)).GetPositions()
    if (cached := specification.misses.get(cid)) is not None and np.array_equal(cached[0], coords):
        return cached[1]
    misses = compared(specification, mol.GetConformer(int(cid))).misses
    specification.misses[cid] = (coords, misses)
    return misses


_LIGAND_SITES = {"point": "{}", "ez": "double bond {}={}", "axis": "axis {}-{}"}


def _ligand_failure(conformers, cid):
    """Return the first specified or coordination-locked ligand stereo element conformer `cid` shows otherwise."""
    specification = conformers.specification
    miss = next(
        (miss for miss in conformer_misses(specification, conformers._mol, cid) if miss.kind in _LIGAND_SITES), None
    )
    if miss is None:
        return None
    site = _LIGAND_SITES[miss.kind].format(*(atom_label(specification.mol, atom) for atom in miss.atoms))
    if miss.observed:
        kind, detail = "ligand_stereo", f"{site} reads {miss.observed}, not {miss.specified}"
    else:
        kind, detail = "ligand_stereo_unassigned", f"{site} no longer reads {miss.specified}"
    if miss.kind == "point" and miss.atoms[0] in specification.locked:  # the input arrangement's hand, not this one's
        kind = "locked_stereo"
    return Failure(kind, detail, atoms=miss.atoms)


class Gate(NamedTuple):
    """Name one acceptance check; `CORE_GATES` lists them in the order a conformer's first failure is read.

    `check(conformers, cid)` returns a `Failure` or None. A relaxed conformer whose first failure is a `ladder` gate's
    is retried on a stiffer rung, then restored to its seed. `required` lists failure kinds the caller asked for: an
    underfilled request failing on one raises instead of returning fewer conformers. `calculator` says whether an
    external optimizer's geometry is dropped, flagged or not checked, and `calculator_check` replaces `check` for it.
    """

    name: str
    check: Callable[..., Failure | None]
    ladder: bool = True
    required: tuple[str, ...] = ()
    calculator: str = "ignore"
    calculator_check: Callable[..., Failure | None] | None = None


_STRUCT_CAP_SLACK = 2.0  # deg: finite-force settling tolerance on coplanarity and umbrella caps


@dataclass(frozen=True, slots=True, kw_only=True)
class FixMiss:
    """Name one numeric fix a conformer misses: its `atoms`, the `actual` value and the `wanted` text, in `unit`.

    `excess` is how far `actual` lies past the fix's tolerance, infinite when it is undefined.
    """

    atoms: tuple[int, ...]
    actual: float
    wanted: str
    unit: str
    excess: float


def fixed_geometry_misses(conformers, cid):
    """Return the `FixMiss` of each numeric fix conformer `cid` misses."""
    conf = conformers._mol.GetConformer(cid)
    pos = conf.GetPositions()
    misses = []
    for atoms, (lo, hi) in conformers.cons.fixed.items():
        actual = constraint_value(pos, atoms, conformers.cons.haptic, (lo, hi))
        tol, unit = (FIX_DISTANCE_TOL, "A") if len(atoms) == DIST_ATOMS else (FIX_ANGLE_TOL, "deg")
        excess = abs(actual - lo) - tol if lo == hi else max(lo - actual, actual - hi)
        if not np.isfinite(actual):
            excess = float("inf")
        if excess > 0.0:
            wanted = f"{lo:.3f}" if lo == hi else f"{lo:.3f}-{hi:.3f}"
            misses.append(FixMiss(atoms=atoms, actual=actual, wanted=wanted, unit=unit, excess=excess))
    return misses


def fix_failure(conformers, *ids):
    """Return a `Failure` naming the largest numeric fix miss over conformers `ids`, or None."""
    if not (misses := [miss for cid in ids for miss in fixed_geometry_misses(conformers, cid)]):
        return None
    miss = max(misses, key=lambda miss: miss.excess)
    names = "-".join(atom_label(conformers.specification.mol, atom) for atom in miss.atoms)
    return Failure("numeric_fix", f"fix= {names} not held ({miss.actual:.3f} {miss.unit}, requested {miss.wanted})")


def _bonding_failure(conformers, cid):
    """Return a torn ligand bond or clash, metal and constrained pairs exempt, as seed selection checks."""
    cons = conformers.cons
    bonding = bonding_failure(conformers._mol, cid, exclude=cons.frozen, constrained=cons.distances)
    return Failure("bonding", bonding.detail, atoms=bonding.atoms) if bonding else None


def _coordination_failure(conformers, cid, *, require_fit=True):
    """Return the first centre that leaves its polyhedron."""
    return coordination_state_failure(conformers._mol, cid, conformers.specification, require_fit=require_fit)


def _window_failure(conformers, cid, label, atoms, value, window):
    """Return a `Failure` naming a measured value outside its accepted window.

    A two-atom window is an M-L distance; anything longer is a structural angle. A haptic centroid is named by its face.
    """
    graph, cons = conformers.specification.mol, conformers.cons
    names = "-".join(
        f"({','.join(atom_label(graph, a) for a in cons.haptic[atom])})"
        if atom in cons.haptic
        else atom_label(graph, atom)
        for atom in atoms
    )
    distance = len(atoms) == DIST_ATOMS
    kind, unit, digits = ("ml_distance", "A", 2) if distance else ("structural_constraint", "deg", 1)
    if value is None or not np.isfinite(value):
        return Failure(kind, f"{label} {names} is undefined", atoms=tuple(atoms))
    lo, hi = window
    excess = max(lo - value, value - hi, 0.0)
    span = f"{lo:.{digits}f}-{hi:.{digits}f}"
    detail = f"{label} {names} {value:.{digits}f} {unit}, {excess:.2g} {unit} outside {span}"
    return Failure(kind, detail, atoms=tuple(atoms))


def _ml_window_failure(conformers, cid):
    """Return the first model M-L distance outside its window and tolerance."""
    cons = conformers.cons
    pos = conformers._mol.GetConformer(int(cid)).GetPositions()
    for atoms, window in cons.distances.items():
        # A phantom is private scaffolding, not a postcondition on the flexible face's real centroid.
        if atoms in cons.contacts.distances or cons.phantoms.intersection(atoms) or not cons.metals.intersection(atoms):
            continue
        tol = metal_distance_tolerance(atoms, cons)
        value = constraint_value(pos, atoms, cons.haptic, window)
        if not within_window(value, window, tol):
            accepted = window[0] - tol, window[1] + tol
            return _window_failure(conformers, cid, "M-L distance", atoms, value, accepted)
    return None


def _coplanarity_failure(conformers, cid):
    """Return the first coplanarity cap missed by more than `_STRUCT_CAP_SLACK`."""
    cons = conformers.cons
    pos = conformers._mol.GetConformer(int(cid)).GetPositions()
    for row in cons.coplanar:
        if requested_dihedral(cons, *row.atoms):
            continue
        if row.out_of_plane:  # crystal structures measured this angle, not a dihedral about one donor bond
            deviation = out_of_plane(pos, *row.atoms)
        else:
            measured = constraint_value(pos, row.atoms, cons.haptic)
            deviation = None if measured is None else min(abs(measured), 180.0 - abs(measured))
        accepted = 0.0, row.cap + _STRUCT_CAP_SLACK
        if deviation is None or not np.isfinite(deviation) or deviation > accepted[1]:
            return _window_failure(conformers, cid, "coplanarity", row.atoms, deviation, accepted)
    return None


def _umbrella_failure(conformers, cid):
    """Return the first umbrella improper outside its hold's window by more than `_STRUCT_CAP_SLACK`.

    This is the one judge of a planar shell's flatness. A miss is a ``coordination_shape`` failure, as the shell leaves
    its polyhedron, so the conformer waits for a fresh seed. A window on the negative side is reported as a magnitude.
    """
    cons = conformers.cons
    pos = conformers._mol.GetConformer(int(cid)).GetPositions()
    for atoms, row in cons.umbrellas.items():
        if requested_dihedral(cons, *atoms, improper=True):
            continue
        measured = constraint_value(pos, atoms, cons.haptic)
        window = row.window(np.nan if measured is None else measured)
        value = constraint_value(pos, atoms, cons.haptic, window)  # in the window's own half-turn
        if not within_window(value, window, _STRUCT_CAP_SLACK):
            if window[1] <= 0.0:
                window, value = (-window[1], -window[0]), None if value is None else -value
            accepted = window[0] - _STRUCT_CAP_SLACK, window[1] + _STRUCT_CAP_SLACK
            failure = _window_failure(conformers, cid, "umbrella", atoms, value, accepted)
            return replace(failure, kind="coordination_shape")
    return None


_CENTRE_KINDS = {"hand": "hand", "face": "haptic face"}  # a centre's `Miss.kind` and what a message calls it


def _centre_failure(conformers, cid, *, kind):
    """Return the first specified metal `kind`, 'hand' or 'face', that conformer `cid` shows otherwise."""
    specification = conformers.specification
    for miss in conformer_misses(specification, conformers._mol, cid):
        if miss.kind == kind:
            centre = atom_label(specification.mol, miss.atoms[0])
            return Failure(f"metal_{kind}", f"{centre} loses its requested {_CENTRE_KINDS[kind]}", atoms=miss.atoms)
    return None


CORE_GATES = (
    Gate("fixed", fix_failure, required=("numeric_fix",), calculator="drop"),
    Gate("bonding", _bonding_failure),
    Gate(
        "coordination_state",
        _coordination_failure,
        required=("coordination_shape", "seating_crossed"),
        calculator="drop",
        calculator_check=partial(_coordination_failure, require_fit=False),
    ),
    # Off the ladder: a mirror the relax cannot reflect back is reported, not restored to its seed.
    Gate(
        "metal_hand", partial(_centre_failure, kind="hand"), ladder=False, required=("metal_hand",), calculator="drop"
    ),
    Gate(
        "ligand_stereo",
        _ligand_failure,
        required=("ligand_stereo", "ligand_stereo_unassigned", "locked_stereo"),
        calculator="drop",
    ),
    Gate("ml_windows", _ml_window_failure),
    Gate("coplanarity", _coplanarity_failure),
    Gate("umbrella", _umbrella_failure, required=("coordination_shape",)),
    Gate("metal_face", partial(_centre_failure, kind="face"), required=("metal_face",), calculator="drop"),
)
