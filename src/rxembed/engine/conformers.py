"""Relax embedded conformers, accept them through their gates and replace failures from fresh seeds."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rdkit import Chem

from rxembed.engine.acceptance import (
    CORE_GATES,
    EmbeddingError,
    Failure,
    describe_groups,
    group_failures,
    headline,
)
from rxembed.engine.seeding import (
    REPLACEMENT_FACTOR,
    reflect_to_specification,
    reflection_is_free,
    seed_conformers,
)
from rxembed.geometry.constraints import (
    ANGLE_ATOMS,
    DIHEDRAL_ATOMS,
    DIST_ATOMS,
    Constraints,
    constraint_value,
    resolve_atom,
)
from rxembed.geometry.relax import MAX_ITERS, UFFRecord, UFFTypingError, error_summary, restrained_uff
from rxembed.identity.reading import Specification
from rxembed.utils import with_conformers

if TYPE_CHECKING:
    from rxembed.engine.acceptance import Gate
    from rxembed.geometry.bounds import Sampling
    from rxembed.identity.species import Species

logger = logging.getLogger(__name__)  # under the "rxembed" tree `set_verbose` configures


class Verdict(NamedTuple):
    """Hold one conformer's gate results, valid only for the coordinates and request they were read on.

    `rows` maps each gate row to its first `Failure`, or None; `inputs` holds the constraints and species, compared by
    identity.
    """

    coords: bytes
    rows: dict[Gate, Failure | None]
    inputs: tuple


def _log_replacement_round(operation, index, kept, asked):
    """Report one fresh-seed replacement round's budget and cost."""
    logger.debug(
        "%s: replace round %d/%d, %d/%d kept",
        operation,
        index + 1,
        _REPLACEMENT_ROUNDS,
        kept,
        asked,
    )


# Half-order steps find the lowest restraint stiffness that keeps the sphere intact.
BASE_STIFFNESS = 1.0
FC_ESCALATION = (1.0, 3.0, 10.0, 30.0, 100.0)
_HIGH_FC_ESCALATION = (300.0, 1000.0)
# Three deterministic batches cover DG basin variability.
_REPLACEMENT_ROUNDS = 3
# another shape or seating waits for a fresh seed, not a stiffer rung: angular walls stop strengthening at 1x
STABLE_SHAPE_KINDS = {"coordination_shape", "seating_crossed"}


def repeated_rejection(signature, previous):
    """Stop bounded replacement when the same specific non-shape failures repeat."""
    return signature == previous and all(kind not in STABLE_SHAPE_KINDS and atoms for kind, atoms in signature)


def _stop_replacing(signature, previous, failures, ladder):
    """Return whether endpoint replacement should stop: every seed crossed seating, or every reason repeats.

    A repeating reason must be a known non-shape site: a wrong shape never stops the search, since a later seed can
    still cross its near tie. Crossed seating is read on a restored seed, not on a ladder endpoint.
    """
    gate_keys = {reason.key for reason, cids in failures.items() if cids}
    crossed_every_seed = (
        len(gate_keys) == 1
        and next(iter(gate_keys))[0] == "seating_crossed"
        and any(reason.kind in STABLE_SHAPE_KINDS for reason in ladder.values())
    )
    return crossed_every_seed or repeated_rejection(signature, previous)


@dataclass
class Conformers:
    """Carry one Mol, selected conformer ids and the constraints that shaped them.

    Construction copies the supplied Mol, ids and energies. `.mol` returns the chemical graph with the selected
    coordinates. `minimize()` mutates and chains; indexing copies.
    rxembed's own modules may use its underscore members; users see only the public surface.
    """

    _mol: Chem.Mol
    ids: list
    cons: Constraints = field(default_factory=Constraints)
    species: Species = field(kw_only=True)  # the record the conformers embed; an organic has no centres
    energies: dict = field(default_factory=dict)  # conformer id -> restrained-UFF or downstream score
    unrelaxed: list = field(default_factory=list, kw_only=True)
    # ids whose UFF endpoint did not converge, or which were restored because no endpoint passed the contract
    relax_failures: dict = field(default_factory=dict, kw_only=True)
    # last restrained-UFF endpoint reason by conformer id; retained when a later seed fails a different gate
    _uff: UFFRecord = field(default_factory=UFFRecord, kw_only=True)  # every relax's statuses and private typing
    _stiffness: float = field(default=BASE_STIFFNESS, kw_only=True)  # requested objective, never a recovery rung
    # what reproduces this batch's seeds; None means search-free (minimize(), wrap()), so replacement is off
    sampling: Sampling | None = field(default=None, kw_only=True)
    # seed plus snapshots from the accepted restrained-UFF attempt; recorded only when explicitly requested
    trajectory: Chem.Mol | None = field(default=None, kw_only=True)
    # {species: Specification}; replace() shares it, so every rung, batch and view of one species prepares it once.
    _specifications: dict = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    # Gate results carry their coordinate bytes; copied results can reuse them until their own coordinates change.
    _verdicts: dict = field(default_factory=dict, kw_only=True, repr=False, compare=False)

    def __post_init__(self):
        """Copy the constructor's Mol, ids and records, so this result owns them."""
        self._mol = Chem.Mol(self._mol)
        self.ids = list(self.ids)
        self.energies = dict(self.energies)
        self._verdicts = dict(self._verdicts)

    @property
    def specification(self):
        """Return what `species` specifies, prepared once for every conformer it is compared with."""
        if (cached := self._specifications.get(self.species)) is None:
            cached = self._specifications[self.species] = Specification(species=self.species)
        return cached

    def _restored_mol(self, cid=None):
        """Return tracked conformers, or one requested conformer, on a copy of the species Mol with its notes."""
        return with_conformers(self.specification.mol, self._mol, self.ids if cid is None else [cid])

    @property
    def mol(self):
        """Return the tracked conformers on a copy of the species Mol: real metals, donor bonds dative, notes."""
        return self._restored_mol()

    def _store_trajectory(self, frames):
        """Store accepted cleanup frames on the restored species Mol."""
        if frames is None:
            return
        if len(self.ids) != 1 or not frames:
            self.trajectory = None
            return
        final = self._mol.GetConformer(int(self.ids[0])).GetPositions()
        if not np.allclose(frames[-1], final):
            self.trajectory = None
            logger.warning("trajectory: final geometry was replaced after UFF; no continuous path was retained")
            return
        mol = self.mol
        mol.RemoveAllConformers()
        for positions in frames:
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetPositions(positions)
            mol.AddConformer(conf, assignId=True)
        self.trajectory = mol

    def _remove(self, ids):
        """Stop tracking conformers and their attached status records."""
        removed = set(ids)
        self.ids = [cid for cid in self.ids if cid not in removed]
        self.unrelaxed = [cid for cid in self.unrelaxed if cid not in removed]
        for cid in removed:
            self.energies.pop(cid, None)

    def _coordinates_changed(self):
        """Forget what was read on the old coordinates: energies, relax records and gate verdicts."""
        self.energies, self.unrelaxed, self.relax_failures, self._uff = {}, [], {}, UFFRecord()
        self.trajectory = None
        self._verdicts = {}

    gates = CORE_GATES

    def _first_failure(self, cid, gates):
        """Return the first of `gates` conformer `cid` fails, or None; a gate runs only on coordinates not yet read."""
        coords = self._mol.GetConformer(int(cid)).GetPositions().tobytes()
        verdict = self._verdicts.get(cid)
        fresh = verdict is not None and verdict.coords == coords
        rows = (
            dict(verdict.rows) if fresh and verdict.inputs[0] is self.cons and verdict.inputs[1] is self.species else {}
        )
        for gate in gates:
            if gate not in rows:
                rows[gate] = gate.check(self, cid)
                self._verdicts[cid] = Verdict(coords, rows, (self.cons, self.species))
            if reason := rows[gate]:
                return reason
        return None

    def _relax(self, stiffness, max_iters, operation="minimize", frames=None, *, ids=None, mirror=True):
        """Relax conformers `ids` (default all) with restrained UFF; return whether UFF could type the system.

        Without constraints this is one UFF pass with its energies kept. With them, each conformer climbs a ladder of
        restraint stiffness until it passes the gates the ladder repairs; a wrong basin or a torn ligand bond waits for
        a fresh seed instead. A conformer that never passes returns to its seed with its reason in `relax_failures`. The
        ladder keeps no energies, so a rung's stiffness never reaches a reported energy: `_rescore_restrained` scores
        once on the caller's objective. A `frames` list receives the first conformer's accepted path. Every conformer it
        returns, a restored seed included, has a free mirrored metal hand reflected back, so acceptance only reads,
        unless `mirror` is False: a coembed endpoint shares its spectators' coordinates with its partner, so it reaches
        its own configuration from the shared seed and is never reflected.
        """
        self._stiffness = stiffness
        relaxing = list(self.ids if ids is None else ids)
        if not relaxing:
            return True
        if not self.cons.is_constrained:
            self.unrelaxed = list(dict.fromkeys([*self.unrelaxed, *relaxing]))
            typed = True
            try:
                e = restrained_uff(
                    self._mol,
                    self.cons,
                    stiffness=stiffness,
                    max_iters=max_iters,
                    conf_ids=relaxing,
                    record=self._uff,
                )
                self.energies.update({i: float(v) for i, v in zip(relaxing, e, strict=True)})
                self.unrelaxed = [i for i in self.unrelaxed if i not in relaxing] + [
                    i for i in relaxing if not max_iters or self._uff.statuses.get(i, 1) != 0
                ]
            except UFFTypingError as err:
                logger.warning(
                    "%s: UFF could not relax this system (%s); kept the starting geometries",
                    operation,
                    error_summary(err),
                )
                typed = False
            self._correct_metal_hand(operation, ids=relaxing, mirror=mirror)
            return typed
        seed_pos = {cid: self._mol.GetConformer(cid).GetPositions().copy() for cid in relaxing}
        pending = relaxing
        restored, rungs = {}, {}  # rungs: each accepted endpoint's stiffness multiplier
        typed = True
        first = relaxing[0]

        # The ladder's view of `self._uff`: a shallow copy shares its statuses and typing, and keeps frames apart.
        record = replace(self._uff, snapshots={} if frames is not None else None)
        for mult in (*FC_ESCALATION, *_HIGH_FC_ESCALATION):
            if not pending:
                break
            current = pending  # on their seeds: the first rung starts there, and a failed endpoint returns there
            self.unrelaxed = list(dict.fromkeys([*self.unrelaxed, *current]))
            try:
                restrained_uff(
                    self._mol,
                    self.cons,
                    stiffness=stiffness * mult,
                    max_iters=max_iters,
                    conf_ids=current,
                    record=record,
                )
            except UFFTypingError as err:  # restrained_uff put `current` back on its seeds
                restored.update({cid: Failure("uff", "UFF unavailable") for cid in current})
                logger.warning(
                    "%s: UFF could not relax this system (%s); kept the starting geometries",
                    operation,
                    error_summary(err),
                )
                typed = False
                break

            self.unrelaxed = [cid for cid in self.unrelaxed if cid not in current]
            self.energies = {}
            self._correct_metal_hand(operation, ids=current, snapshots=record.snapshots, mirror=mirror)
            failed = {}  # each conformer's first failure, where the gate that caught it is one the ladder repairs
            for cid in current:
                reason = self._first_failure(cid, self.gates)
                if reason and next(gate for gate in self.gates if self._verdicts[cid].rows[gate]).ladder:
                    failed[cid] = reason
            accepted = set(current).difference(failed)
            pending = [  # climb to the next rung, unless a fresh seed is the only repair
                cid
                for cid, reason in failed.items()
                if reason.kind not in STABLE_SHAPE_KINDS
                and mult != _HIGH_FC_ESCALATION[-1]
                and (mult < FC_ESCALATION[-1] or reason.kind == "ml_distance")
                and not (reason.kind == "bonding" and self.species.centres)
            ]
            restored.update({cid: reason for cid, reason in failed.items() if cid not in pending})
            for cid in failed:
                self._mol.GetConformer(cid).SetPositions(seed_pos[cid])
            self.unrelaxed.extend(
                cid
                for cid in current
                if cid not in pending and (cid in failed or not max_iters or record.statuses.get(cid, 1))
            )
            rungs.update(dict.fromkeys(accepted, mult))

        if frames is not None:  # an accepted endpoint is not minimised again, so its snapshot is the accepted path
            snapshots = (record.snapshots or {}).get(first, []) if first in rungs else []
            seed, final = seed_pos[first], self._mol.GetConformer(first).GetPositions()
            # the seed, its UFF snapshots, then `final` if they miss it; no path past a reflection
            path = [] if snapshots is None else [seed, *(frame[: len(seed)] for frame in snapshots)]
            frames[:] = path if not path or np.allclose(path[-1], final) else [*path, final]

        self.relax_failures = {
            cid: reason for cid, reason in self.relax_failures.items() if cid not in seed_pos
        } | restored

        if escalated := [mult for mult in rungs.values() if mult != FC_ESCALATION[0]]:
            logger.debug(
                "%s: %d/%d conformer(s) needed restraint stiffness up to %gx",
                operation,
                len(escalated),
                len(relaxing),
                max(escalated),
            )
        if restored and typed:
            logger.debug(
                "%s: rejected %d/%d UFF results (%s); back to the seed",
                operation,
                len(restored),
                len(relaxing),
                describe_groups(group_failures(restored.values())),
            )
        self._correct_metal_hand(operation, ids=list(restored), mirror=mirror)  # a restored seed was not corrected
        return typed

    def _rescore_restrained(self, stiffness, *, ids=None):
        """Score settled conformers, or explicit `ids`, on the requested restrained-UFF objective.

        An explicit single point does not change convergence or relaxation failures.
        """
        ids = [cid for cid in self.ids if cid not in self.unrelaxed] if ids is None else list(ids)
        self._stiffness = stiffness
        if not ids:
            self.energies = {}
            return
        e = restrained_uff(
            self._mol,
            self.cons,
            stiffness=stiffness,
            max_iters=0,
            conf_ids=ids,
            record=self._uff,
        )
        self.energies = {i: float(v) for i, v in zip(ids, e, strict=True)}

    def _replace_failed(self, failed, stiffness, max_iters, *, operation="minimize"):
        """Replace failed conformers from bounded fresh seed batches; return the batches' relaxation failures.

        Replacements keep the original ids, best endpoint score first. The bounded search stops early on crossed
        seating or repeated specific non-shape failures (`_stop_replacing`).
        """
        if self.sampling is None:
            return []
        left, rejected = list(failed), []
        previous_signature = None
        for batch_index in range(_REPLACEMENT_ROUNDS):
            if not left:
                break
            count = REPLACEMENT_FACTOR * len(left)
            sampling = self.sampling.retry("replacement", 1 + batch_index)
            seeds = seed_conformers(Chem.Mol(self._mol), self.cons, self.species, count, sampling)
            if not seeds.ids:
                previous_signature = None
                _log_replacement_round(operation, batch_index, 0, count)
                continue
            # A batch is its owner's type, so it reads the owner's gates and a constrained batch climbs the ladder.
            batch = replace(self, _mol=seeds.mol, ids=seeds.ids, sampling=sampling)
            batch._coordinates_changed()
            batch._relax(stiffness, max_iters, operation)
            rejected.extend(batch.relax_failures.values())
            failures = batch._acceptance_failures()
            # Each rejected seed's ladder failure, else its acceptance one.
            reasons = {cid: batch.relax_failures.get(cid) or r for r, cids in failures.items() for cid in cids}
            if reasons:
                logger.debug(
                    "%s: reseed batch %d/%d, %d/%d rejected: %s",
                    operation,
                    batch_index + 1,
                    _REPLACEMENT_ROUNDS,
                    len(reasons),
                    len(batch.ids),
                    describe_groups(group_failures(reasons.values())),
                )
            signature = {reason.key for reason in reasons.values()} if set(reasons) == set(batch.ids) else None
            if signature is not None and _stop_replacing(signature, previous_signature, failures, batch.relax_failures):
                _log_replacement_round(operation, batch_index, 0, count)
                return rejected
            previous_signature = signature
            batch._remove(reasons)
            _log_replacement_round(operation, batch_index, len(batch.ids), count)
            scores = {
                cid: energy
                for cid, energy in batch.energies.items()
                if cid not in batch.unrelaxed and np.isfinite(energy)
            }
            order = sorted(batch.ids, key=lambda cid: scores.get(cid, math.inf))
            for cid, src in zip(list(left), order, strict=False):
                self._adopt(cid, batch, src)
                left.remove(cid)
        return rejected

    def _adopt(self, cid, batch, src):
        """Take `batch`'s conformer `src` as conformer `cid`, added when absent, with its records and status."""
        source = batch._mol.GetConformer(int(src))
        if int(cid) not in {conf.GetId() for conf in self._mol.GetConformers()}:
            added = Chem.Conformer(source)
            added.SetId(int(cid))
            self._mol.AddConformer(added, assignId=False)
        self._mol.GetConformer(int(cid)).SetPositions(source.GetPositions())
        self._verdicts.pop(cid, None)
        if src in batch._verdicts:
            self._verdicts[cid] = batch._verdicts[src]
        for mine, theirs in ((self._uff.statuses, batch._uff.statuses), (self.relax_failures, batch.relax_failures)):
            mine.pop(cid, None)
            if src in theirs:
                mine[cid] = theirs[src]
        self._uff.surrogates.update(batch._uff.surrogates)
        self._uff.retyped.update(batch._uff.retyped)
        if cid not in self.ids:
            self.ids.append(cid)
        self.energies.pop(cid, None)
        if src in batch.energies:
            self.energies[cid] = batch.energies[src]
        self.unrelaxed = [i for i in self.unrelaxed if i != cid]
        if src in batch.unrelaxed:
            self.unrelaxed.append(cid)

    def _correct_metal_hand(self, operation="minimize", *, ids=None, snapshots=None, mirror=True):
        """Reflect free metal inversions in place; the `metal_hand` gate rejects what stays wrong.

        Distance and angle constraints cannot choose a mirror; a signed dihedral can. The hand is read after
        relaxation, then corrected by reflection only when no specified geometry distinguishes the mirror.
        """
        ids = self.ids if ids is None else ids
        if not (mirror and ids and reflection_is_free(self.specification, self.cons)):
            return
        reflected = reflect_to_specification(self.specification, self._mol, ids)
        for cid in reflected:
            self.energies.pop(cid, None)
        if reflected and snapshots is not None:
            snapshots.update(dict.fromkeys(reflected))  # Reflection is not a continuous UFF step.
            logger.warning("trajectory: metal-hand reflection discarded the discontinuous UFF path")
        if reflected:
            logger.debug("%s: reflected %d conformer(s) to their specified metal hand", operation, len(reflected))

    def _acceptance_failures(self, ids=None):
        """Return each conformer's first failure of `gates`, grouped by reason; ``ids`` narrows it to a subset.

        Reading changes no coordinate: the step that writes coordinates corrects a free metal hand.
        """
        failures = {}
        for cid in self.ids if ids is None else ids:
            if reason := self._first_failure(cid, self.gates):
                failures.setdefault(reason, []).append(cid)
        return failures

    def _accept(self, stiffness, max_iters, operation="minimize", *, typed=True, raise_if_empty=False):
        """Replace conformers that fail `gates` from fresh seeds, then drop what still fails.

        A requested identity left underfilled raises `EmbeddingError`, as does rejecting every conformer under
        `raise_if_empty`; fewer conformers otherwise warn. An unrelaxed conformer whose restored seed passes is
        kept and reported once. `typed` is False when UFF could not type the system, and then nothing is replaced.
        """
        target = len(self.ids)
        requested = {kind for gate in self.gates for kind in gate.required}  # failure kinds a request makes fatal
        failures = self._acceptance_failures()
        required = {failure.kind for failure in failures} & requested
        failed = {cid for rejected in failures.values() for cid in rejected}
        replacement_failures = []
        if failed and typed:
            replacement_failures = self._replace_failed(failed, stiffness, max_iters, operation=operation)
        if failed:
            # A replacement only touches `failed` ids, and every other id passed the first gate.
            failures = self._acceptance_failures(ids=failed)
            required |= {failure.kind for failure in failures} & requested
            failed = {cid for rejected in failures.values() for cid in rejected}
            self._remove(failed)
        who = str(self.species) or operation
        if failures:
            # One entry per rejected seed: its rejected UFF result, else its final gate reason.
            groups = group_failures(
                [
                    *self.relax_failures.values(),
                    *replacement_failures,
                    *(
                        reason
                        for reason, rejected in failures.items()
                        for cid in rejected
                        if cid not in self.relax_failures
                    ),
                ]
            )
            if len(self.ids) < target and (required or (raise_if_empty and not self.ids)):
                raise EmbeddingError(f"{who}: {headline(groups, self.species, self.cons)}", failures=groups)
            logger.warning(
                "%s: kept %d/%d conformers; %s", who, len(self.ids), target, headline(groups, self.species, self.cons)
            )
        unrelaxed = [cid for cid in self.ids if cid in self.unrelaxed]
        if unrelaxed and typed:
            logger.warning(
                "%s: %d/%d conformer(s) have no converged UFF geometry; see .unrelaxed",
                who,
                len(unrelaxed),
                len(self.ids),
            )
        return failures

    def minimize(self, stiffness=BASE_STIFFNESS, max_iters=MAX_ITERS):
        """Relax every conformer in place with restrained UFF and return ``self``.

        `stiffness` scales the restraint penalties, not native UFF. A constrained run climbs the stiffness ladder,
        replaces conformers that miss the gates from fresh-seed batches when it has seed parameters (an `embed()` result
        does), and drops what still fails. A typing refusal keeps prior energies on unchanged coordinates, and an
        unconverged conformer that passes the gates is listed in `unrelaxed`. Energies rank results within one species
        only; `Ensemble.minimize()` adds further gates.
        """
        self.trajectory = None
        if self.ids:
            typed = self._relax(stiffness, max_iters)
            self._accept(stiffness, max_iters, typed=typed)
            if typed and self.cons.is_constrained:  # retried conformers relaxed at different stiffnesses: score once
                self._rescore_restrained(stiffness)
        return self

    def measure(self, atoms):
        """Return mean, minimum and maximum distance, angle or dihedral over tracked conformers.

        `atoms` contains two to four atom indices or SMARTS selectors. The result also includes the count as `n`.
        """
        if not self.ids:
            raise ValueError("nothing to measure: this result has no conformers; embed again with another seed=")
        idx = tuple(resolve_atom(self.specification.mol, a) for a in atoms)
        if len(idx) not in (DIST_ATOMS, ANGLE_ATOMS, DIHEDRAL_ATOMS):
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        window = None
        if len(idx) == DIHEDRAL_ATOMS:
            window = self.cons.dihedrals.get(min(idx, idx[::-1]))
        v = [constraint_value(self._mol.GetConformer(c).GetPositions(), idx, window=window) for c in self.ids]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}

    def xyz(self, conf_id=None):
        """Return conformer `conf_id` as an xyz block, or every tracked conformer as one multi-frame block.

        `conf_id` is an RDKit conformer id from `ids`, not a position, and must still be tracked.
        """
        if conf_id is not None and int(conf_id) not in [int(c) for c in self.ids]:
            raise ValueError(f"conformer id {conf_id} is not one of this result's ids {list(self.ids)}")
        mol = self.mol
        return "".join(Chem.MolToXYZBlock(mol, confId=int(c)) for c in (self.ids if conf_id is None else [conf_id]))

    def dump(self, path):
        """Write the tracked conformers to `path` as a multi-frame .xyz; return the path."""
        if not self.ids:  # an empty file would read as a successful write
            raise ValueError("nothing to dump: this result has no conformers; embed again with another seed=")
        with open(path, "w") as f:
            f.write(self.xyz())
        return path

    def dump_trajectory(self, path):
        """Write the captured restrained-UFF trajectory to `path`; return the path."""
        if self.trajectory is None or not self.trajectory.GetNumConformers():
            raise ValueError(
                "no trajectory kept; embed with rx.embed(..., n=1, trajectory=True) and check its warnings"
            )
        with open(path, "w") as f:
            for conf in self.trajectory.GetConformers():
                f.write(Chem.MolToXYZBlock(self.trajectory, confId=conf.GetId()))
        return path

    def _derive(self, ids, mol=None):
        """Return a new result of the selected conformers and their records, on working Mol `mol` when given.

        Untracked frames stay for discarded-geometry views. A trajectory belongs only to an unchanged full selection.
        """
        ids = list(ids)
        selected = set(ids)
        working = self._mol if mol is None else mol
        excluded = set(self.ids) - selected if mol is None else set()
        retained = {c.GetId() for c in working.GetConformers()} - excluded
        out = replace(
            self,
            _mol=working,
            ids=ids,
            energies={i: self.energies[i] for i in ids if i in self.energies},
            unrelaxed=[i for i in self.unrelaxed if i in selected],
            relax_failures={i: reason for i, reason in self.relax_failures.items() if i in retained},
            _verdicts={i: verdict for i, verdict in self._verdicts.items() if i in retained},
            _uff=replace(
                self._uff,
                statuses={i: status for i, status in self._uff.statuses.items() if i in retained},
                snapshots=(
                    None
                    if self._uff.snapshots is None
                    else deepcopy({i: frames for i, frames in self._uff.snapshots.items() if i in retained})
                ),
                surrogates=dict(self._uff.surrogates),
                retyped=set(self._uff.retyped),
            ),
            trajectory=(
                Chem.Mol(self.trajectory) if mol is None and self.trajectory is not None and ids == self.ids else None
            ),
        )
        for cid in excluded:
            out._mol.RemoveConformer(int(cid))
        return out

    def __getitem__(self, key):
        """Copy conformer(s) by position as a new result: ``confs[0]``, ``confs[:3]``."""
        sel = self.ids[key]
        return self._derive(sel if isinstance(sel, list) else [sel])

    def __len__(self):
        """Return the number of conformers in play."""
        return len(self.ids)

    def __repr__(self):
        """Summarise the result: conformer count and the species' centres when it has any."""
        who = f", {self.species}" if str(self.species) else ""
        return f"<Conformers: {len(self.ids)} conformer{'s' if len(self.ids) != 1 else ''}{who}>"
