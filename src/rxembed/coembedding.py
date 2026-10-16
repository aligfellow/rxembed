"""Couple native endpoint workflows around one mapped starting geometry."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .bounds import seed_bounds
from .embed import (
    _REPLACEMENT_ROUNDS,
    BASE_STIFFNESS,
    FC_ESCALATION,
    Conformers,
    EmbeddingError,
    _high_force_candidate,
    _stable_coordination_failure,
    seed_conformers,
)
from .metal_core import materialise_phantoms
from .relax import MAX_ITERS, UFFRecord, _alignment_terms, _coupled_uff, _prepare_uff, error_summary
from .stereo import axis_stereo, rotate_axial_stereo

logger = logging.getLogger("rxembed")

# Sampling defaults for the joint spectator and reacting-atom penalties.
ALIGNMENT_WEIGHT = 25.0
TS_WEIGHT = 1.25


def correspondence_budgets(molecules, terms, correspondence, params):
    """Estimate each matched atom's movement allowance from incompatible endpoint distance windows.

    Assign the largest distance-window gap involving an atom to its budget. This is a conservative heuristic:
    a gap needs the sum of two atoms' displacements, and distance windows do not establish stereo feasibility.
    """
    left, right = np.asarray(correspondence, dtype=int).T
    matrices = [seed_bounds(mol, cons, params) for mol, cons in zip(molecules, terms, strict=True)]
    # Normalize each triangular matrix before selecting arbitrary endpoint-local index orders.
    lower, upper = [], []
    for matrix, indices in zip(matrices, (left, right), strict=True):
        selection = np.ix_(indices, indices)
        lower.append(np.minimum(matrix, matrix.T)[selection])
        upper.append(np.maximum(matrix, matrix.T)[selection])
    gap = np.maximum(np.maximum(*lower) - np.minimum(*upper), 0.0)
    budgets = gap.max(axis=1)
    return dict(zip(correspondence, map(float, budgets), strict=True))


def seed_paired_conformers(mol, cons, endpoints, n, params, *, max_iters, reseed=True):
    """Seed one endpoint and transfer its mapped geometry before native endpoint relaxation."""
    # Break bond-count ties with fewer DG centroid helpers; an intact side-on ligand can be harder to seed.
    side = max(range(2), key=lambda index: (endpoints[index]._mol.GetNumBonds(), -len(endpoints[index].cons.haptic)))
    owner, other = endpoints[side], endpoints[1 - side]
    logger.info(
        "Coembedding shared seed: %s, %d atoms, %d pairs", ("reactant", "product")[side], owner._mol.GetNumAtoms(), n
    )
    mol.RemoveAllConformers()
    ids = []
    last_failure = None
    for attempt in range(_REPLACEMENT_ROUNDS + 1 if reseed else 1):
        trial = replace(params, seed=params.seed + attempt)
        try:
            source, seeded_ids, _target = seed_conformers(
                Chem.Mol(owner._mol), owner.cons, owner.iso, n - len(ids), trial, reseed=reseed
            )
            if not seeded_ids:
                logger.info("coembed: seed %d produced no shared seeds", trial.seed)
                continue
            source = Conformers(source, seeded_ids, owner.cons, owner.iso, params=trial, _reseed=reseed)
            source.minimize(max_iters=max_iters)
            partner = Chem.Mol(other._mol)
            partner.RemoveAllConformers()
            for cid in source.ids:
                positions = source._mol.GetConformer(cid).GetPositions()
                copied = np.zeros((partner.GetNumAtoms(), 3))
                for pair in cons.seed_correspondence:
                    copied[pair[1 - side]] = positions[pair[side]]
                conf = Chem.Conformer(partner.GetNumAtoms())
                conf.SetId(cid)
                conf.SetPositions(copied)
                partner.AddConformer(conf, assignId=False)
            # Repair changed hybridisation before seed-centred conjugation holds capture the copied coordinates.
            partner = Conformers(
                partner, list(source.ids), other.cons.copy(conjugation=False), other.iso, params=trial, _reseed=reseed
            )
            if axis_stereo(partner._stated_stereo):
                split = endpoints[0]._mol.GetNumAtoms() + len(endpoints[0].cons.phantoms)
                stationary = {pair[1 - side] - (split if side == 0 else 0) for pair in cons.alignment_pairs or ()}
                for cid in partner.ids:
                    public = partner.restored_mol(cid)
                    rotate_axial_stereo(public, partner._stated_stereo, stationary)
                    partner._mol.GetConformer(cid).SetPositions(public.GetConformer().GetPositions())
            typed = partner._relax_once(BASE_STIFFNESS, max_iters)
            failures = partner._acceptance_failures()
            discarded = {
                cid for reason, cids in failures.items() if _stable_coordination_failure(reason) for cid in cids
            }
            # A wrong coordination basin needs a different shared seed, not independent endpoint embedding.
            partner._remove(discarded)
            partner._accept_relaxed(BASE_STIFFNESS, max_iters, typed=typed)
            for cid in (cid for cid in source.ids if cid in partner.ids):
                blocks = [
                    np.zeros((endpoint._mol.GetNumAtoms() + len(endpoint.cons.phantoms), 3)) for endpoint in endpoints
                ]
                for block_index, seeded in ((side, source), (1 - side, partner)):
                    positions = seeded._mol.GetConformer(cid).GetPositions()
                    blocks[block_index][: len(positions)] = positions
                for block, endpoint in zip(blocks, endpoints, strict=True):
                    for dummy, atoms in endpoint.cons.haptic.items():
                        block[dummy] = block[list(atoms)].mean(axis=0)
                conf = Chem.Conformer(mol.GetNumAtoms())
                conf.SetPositions(np.concatenate(blocks))
                conf.SetIntProp("_rxembed_seed", trial.seed)
                conf.SetIntProp("_rxembed_source_id", cid)
                ids.append(mol.AddConformer(conf, assignId=True))
        except EmbeddingError as error:
            last_failure = error
            logger.info("coembed: seed %d rejected: %s", trial.seed, error)
            continue
        if len(ids) == n:
            break
        logger.info("coembed: retained %d/%d shared seeds; retrying only missing pairs", len(ids), n)
    if not ids and last_failure is not None:
        raise last_failure
    if len(ids) < n:
        logger.warning("coembed: kept %d/%d shared seeds after search; try another seed=", len(ids), n)
    return mol, ids


@dataclass
class PairedConformers(Conformers):
    """Couple two endpoint objectives while retaining the shared relaxation and recovery policy."""

    endpoints: tuple = field(kw_only=True)
    validators: tuple = field(default=(None, None), kw_only=True, repr=False)
    _seed_max_iters: int = field(default=MAX_ITERS, kw_only=True, repr=False)
    alignment_weight: float = field(default=ALIGNMENT_WEIGHT, kw_only=True)
    joint_gtol: float = field(default=0.1, kw_only=True)
    joint_energy_tol: float = field(default=1e-3, kw_only=True)
    joint_results: dict = field(default_factory=dict, kw_only=True)
    optimizer_attempts: list = field(default_factory=list, kw_only=True)
    recovery_batches: list = field(default_factory=list, kw_only=True, repr=False)

    def _sync_endpoints(self):
        """Copy packed coordinates into endpoint graphs without changing either chemical identity."""
        offset = 0
        for owner in self.endpoints:
            count = owner._mol.GetNumAtoms()
            block_count = count + len(owner.cons.phantoms)
            owner._mol.RemoveAllConformers()
            owner.ids = list(self.ids)
            for cid in self.ids:
                conf = Chem.Conformer(count)
                conf.SetId(cid)
                conf.SetPositions(self._mol.GetConformer(cid).GetPositions()[offset : offset + count])
                owner._mol.AddConformer(conf, assignId=False)
            offset += block_count

    def _pack_endpoints(self):
        """Rebuild transient haptic points while packing endpoint coordinates into their shared frame."""
        blocks = [materialise_phantoms(owner._mol, owner.cons.haptic) for owner in self.endpoints]
        for cid in self.ids:
            positions = np.concatenate([block.GetConformer(cid).GetPositions() for block in blocks])
            self._mol.GetConformer(cid).SetPositions(positions)

    def _refresh_scores(self, stiffness, *, excluded=()):
        """Rank settled pairs by uniformly rescored endpoint energies plus the alignment penalty."""
        for owner in self.endpoints:
            owner._rescore_restrained(stiffness, ids=[cid for cid in self.ids if cid not in excluded])
        self.energies = {}
        self.unrelaxed = sorted({cid for owner in self.endpoints for cid in owner.unrelaxed if cid in self.ids})
        for cid in self.ids:
            if all(cid in owner.energies for owner in self.endpoints):
                pos = self._mol.GetConformer(cid).GetPositions()
                penalty = _alignment_terms(pos, self.cons, self.alignment_weight)[0]
                self.energies[cid] = sum(owner.energies[cid] for owner in self.endpoints) + penalty

    def _replacement_batch(self, count, params):
        """Generate a whole paired batch through the same seed engine as the initial search."""
        mol, ids = seed_paired_conformers(
            Chem.Mol(self._mol),
            self.cons,
            self.endpoints,
            count,
            params,
            max_iters=self._seed_max_iters,
            reseed=self._reseed,
        )
        endpoints = tuple(
            replace(
                owner,
                _mol=Chem.Mol(owner._mol),
                ids=[],
                energies={},
                unrelaxed=[],
                relax_failures={},
                uff=UFFRecord(),
                params=params,
                _reseed=self._reseed,
            )
            for owner in self.endpoints
        )
        batch = PairedConformers(
            mol,
            ids,
            self.cons,
            params=params,
            _reseed=self._reseed,
            endpoints=endpoints,
            validators=tuple(
                check.__func__.__get__(fresh, type(fresh)) if getattr(check, "__self__", None) is owner else check
                for check, owner, fresh in zip(self.validators, self.endpoints, endpoints, strict=True)
            ),
            _seed_max_iters=self._seed_max_iters,
            alignment_weight=self.alignment_weight,
            joint_gtol=self.joint_gtol,
            joint_energy_tol=self.joint_energy_tol,
            _coord_state_prep=self._coord_state_prep,
        )
        self.recovery_batches.append(batch)
        return batch

    def _adopt_replacement(self, cid, batch, src):
        """Adopt both endpoint coordinates and their stage diagnostics into the original pair id."""
        super()._adopt_replacement(cid, batch, src)
        self.joint_results[cid] = dict(batch.joint_results[src])
        for owner, source in zip(self.endpoints, batch.endpoints, strict=True):
            if src in source.uff.statuses:
                owner.uff.statuses[cid] = source.uff.statuses[src]
            owner.unrelaxed = [index for index in owner.unrelaxed if index != cid]
            if src in source.unrelaxed:
                owner.unrelaxed.append(cid)
            owner.uff.surrogates.update(source.uff.surrogates)
            owner.uff.retyped.update(source.uff.retyped)

    def _required_failure(self, failures):
        """Preserve both endpoints' required identity and workflow failures when recovery underfills."""
        return set().union(*(owner._required_failure(failures) for owner in self.endpoints))

    def _joint_pass(self, stiffness, max_iters, stage, *, ids=None):
        """Rigidly register spectators when it lowers coupling, then optimize the complete paired objective."""
        assert self.params is not None
        split = self.endpoints[0]._mol.GetNumAtoms() + len(self.endpoints[0].cons.phantoms)
        alignment = self.cons.alignment_pairs if self.cons.alignment_pairs is not None else tuple(self.cons.distances)
        left, right = np.asarray(alignment, dtype=int).T
        for cid in self.ids if ids is None else ids:
            seed = self._mol.GetConformer(cid).GetPositions().copy()
            graphs, field_terms = [], []
            packed_offset = 0
            for owner in self.endpoints:
                target, terms, retyped, replacements = _prepare_uff(owner._mol, owner.cons, owner.uff)
                for dummy in owner.cons.phantoms:
                    target.GetConformer(cid).SetAtomPosition(
                        int(dummy), self._mol.GetConformer(cid).GetPositions()[packed_offset + int(dummy)]
                    )
                graphs.append(target)
                field_terms.append(terms)
                owner.uff.retyped.update(retyped)
                owner.uff.surrogates.update(replacements)
                packed_offset += target.GetNumAtoms()
            rdMolAlign.AlignMol(
                graphs[1],
                graphs[0],
                prbCid=cid,
                refCid=cid,
                atomMap=[(int(j - split), int(i)) for i, j in alignment],
            )
            registered = np.concatenate([graph.GetConformer(cid).GetPositions() for graph in graphs])
            if (
                _alignment_terms(registered, self.cons, self.alignment_weight)[0]
                <= _alignment_terms(seed, self.cons, self.alignment_weight)[0]
            ):
                seed = registered
            else:
                graphs[1].GetConformer(cid).SetPositions(seed[split:])
            try:
                positions, record = _coupled_uff(
                    graphs,
                    field_terms,
                    self.cons,
                    cid,
                    weight=self.alignment_weight,
                    stiffness=stiffness,
                    gtol=self.joint_gtol,
                    energy_tol=self.joint_energy_tol,
                    max_iters=max_iters,
                )
                if not np.all(np.isfinite(positions)):
                    raise FloatingPointError("non-finite joint coordinates")
            except (RuntimeError, FloatingPointError) as error:
                positions = seed
                record = {
                    "converged": False,
                    "message": error_summary(error),
                    "stopping_reason": "failure",
                    "cycles": 0,
                    "iterations": None,
                    "iteration_budget_used": None,
                    "energy_tol": self.joint_energy_tol,
                    "score_change": math.inf,
                    "score_history": (),
                    "seconds": 0.0,
                    "max_force": math.inf,
                    "gtol": self.joint_gtol,
                    "score": math.inf,
                }
                logger.warning(
                    "coembed: joint optimization failed for pair %d; restored its starting geometry (%s)",
                    cid,
                    error_summary(error),
                )
            distances = np.linalg.norm(positions[left] - positions[right], axis=1)
            source = self._mol.GetConformer(cid).GetPropsAsDict(includePrivate=True)
            record = {
                **record,
                "alignment_score": float(np.sqrt(np.mean(distances**2))),
                "seed": source.get("_rxembed_seed", self.params.seed),
                "source_id": source.get("_rxembed_source_id", cid),
                "stage": stage,
                "seed_method": "shared",
                "stiffness": stiffness,
            }
            self.joint_results[cid] = record
            self.optimizer_attempts.append(dict(record))
            self._mol.GetConformer(cid).SetPositions(positions)

        self._sync_endpoints()

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, operation="coembed", _frames=None):
        """Couple endpoint fields, run native cleanup, then polish the coupled geometry before validation."""
        if self.params is None:
            raise ValueError("paired relaxation needs EmbedParams; construct the pair with coembed(...)")
        self._seed_max_iters = max_iters
        self._sync_endpoints()
        self.joint_results, self.optimizer_attempts, self.recovery_batches = {}, [], []
        failed = {cid for cids in self._acceptance_failures(operation).values() for cid in cids}
        # Initial coupling can cross a declared stereo barrier too; retain valid inputs as well as cleanup.
        checkpoint = {cid: self._mol.GetConformer(cid).GetPositions().copy() for cid in self.ids if cid not in failed}
        self._joint_pass(stiffness, max_iters, "initial")
        typed = True
        self.relax_failures = {}
        offset = 0
        for owner in self.endpoints:
            typed = owner._relax_once(stiffness, max_iters, operation=operation) and typed
            for cid, reason in owner.relax_failures.items():
                self.relax_failures.setdefault(
                    cid, replace(reason, atoms=tuple(index + offset for index in reason.atoms))
                )
            offset += owner._mol.GetNumAtoms() + len(owner.cons.phantoms)
        self._pack_endpoints()
        if typed:
            failed = {cid for cids in self._acceptance_failures(operation).values() for cid in cids}
            checkpoint.update(
                (cid, self._mol.GetConformer(cid).GetPositions().copy()) for cid in self.ids if cid not in failed
            )
            pending = list(self.ids)
            for multiplier in FC_ESCALATION:
                self._joint_pass(stiffness * multiplier, max_iters, "polish", ids=pending)
                failures = self._acceptance_failures(operation)
                failed = {cid for cids in failures.values() for cid in cids}
                checkpoint.update(
                    (cid, self._mol.GetConformer(cid).GetPositions().copy()) for cid in self.ids if cid not in failed
                )
                pending = [cid for reason, cids in failures.items() if _high_force_candidate(reason) for cid in cids]
                if not pending:
                    break
            restored = failed & checkpoint.keys()
            # Optional polishing must not turn a validated cleanup into a costly fresh-seed search.
            for cid in restored:
                self._mol.GetConformer(cid).SetPositions(checkpoint[cid])
            self._sync_endpoints()
            self._refresh_scores(stiffness)
            self._mark_validation_adjusted(restored, "coupled relaxation rejected; retained last validated geometry")
        return typed

    def _acceptance_failures(self, operation="coembed", validator=None, ids=None):
        """Validate endpoints separately and reject either failure as a failure of the whole pair."""
        self._sync_endpoints()
        failures, rejected = {}, set()
        adjusted = set()
        offset = 0
        for endpoint, (owner, check) in enumerate(zip(self.endpoints, self.validators, strict=True)):
            before = {cid: owner._mol.GetConformer(cid).GetPositions().copy() for cid in self.ids}
            reasons = owner._acceptance_failures(operation, validator=check, ids=ids)
            adjusted.update(
                cid
                for cid in self.ids
                if not np.allclose(before[cid], owner._mol.GetConformer(cid).GetPositions(), atol=1e-10, rtol=0)
            )
            for reason, cids in reasons.items():
                failure = replace(
                    reason,
                    detail=f"{('reactant', 'product')[endpoint]}: {reason.detail}",
                    atoms=tuple(index + offset for index in reason.atoms),
                )
                selected = [cid for cid in cids if cid not in rejected]
                if selected:
                    failures.setdefault(failure, []).extend(selected)
                    rejected.update(selected)
            offset += owner._mol.GetNumAtoms() + len(owner.cons.phantoms)
        self._pack_endpoints()
        self._refresh_scores(BASE_STIFFNESS, excluded=rejected)
        self._mark_validation_adjusted(adjusted, "endpoint validation changed coordinates after joint polishing")
        return failures

    def _mark_validation_adjusted(self, ids, message):
        """Invalidate optimizer convergence when publication changes the joint coordinates."""
        left, right = np.asarray(self.cons.alignment_pairs or tuple(self.cons.distances), dtype=int).T
        for cid in ids:
            if cid not in self.joint_results:
                continue
            positions = self._mol.GetConformer(cid).GetPositions()
            self.joint_results[cid] = {
                **self.joint_results[cid],
                "stage": "validation",
                "stopping_reason": "validation_adjusted",
                "converged": False,
                "message": message,
                "alignment_score": float(np.sqrt(np.mean(np.sum((positions[left] - positions[right]) ** 2, axis=1)))),
                "score": self.energies.get(cid, math.inf),
                "max_force": math.inf,
                "score_history": (),
                "score_change": math.inf,
            }
