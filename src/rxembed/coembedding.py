"""Couple native endpoint workflows around one mapped starting geometry."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .bounds import seed_bounds, seed_coordinates
from .constraints import Constraints
from .embed import (
    _REPLACEMENT_FACTOR,
    _REPLACEMENT_ROUNDS,
    BASE_STIFFNESS,
    Conformers,
    EmbeddingError,
    _ligand_stereo_failure,
    _seed_geometry_matches,
    seed_conformers,
)
from .metal_core import materialise_phantoms
from .relax import MAX_ITERS, UFFRecord, _alignment_terms, _coupled_uff, _prepare_uff, error_summary
from .stereo import reflect_point_stereo, rotate_axial_stereo, stereo_restraints

logger = logging.getLogger("rxembed")

# Sampling defaults for the joint spectator and reacting-atom penalties.
ALIGNMENT_WEIGHT = 25.0
REACT_WEIGHT = 1.25


def _bond_graphs(endpoints, correspondence):
    """Return the reaction-mapped endpoint bonds and their changes."""
    originals = [owner.restored_mol() for owner in endpoints]
    back = {j: i for i, j in correspondence}
    graphs = []
    for side, mol in enumerate(originals):
        edges = {}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if side:
                a, b = back[a], back[b]
            key = tuple(sorted((a, b)))
            direction = (a, b) if str(bond.GetBondType()).startswith("DATIVE") else key
            edges[key] = bond.GetBondType(), direction
        graphs.append(edges)
    left, right = graphs
    changed = {edge for edge in left.keys() | right.keys() if left.get(edge) != right.get(edge)}
    return back, left, right, changed


def _seed_cpci(left, right, changed, *, back=None):
    """Build reaction-graph CPCI preferences, optionally reindexed onto the product graph."""
    cpci = dict.fromkeys(changed, -1.0)
    repelled = {
        tuple(sorted(set(first) ^ set(second)))
        for first in left.keys() - right.keys()
        for second in right.keys() - left.keys()
        if len(set(first) & set(second)) == 1
    }
    cpci.update(dict.fromkeys(repelled - left.keys() - right.keys(), 1.0))
    if back is not None:
        forward = {left_idx: right_idx for right_idx, left_idx in back.items()}
        cpci = {tuple(sorted((forward[i], forward[j]))): value for (i, j), value in cpci.items()}
    return cpci


def seed_union_conformers(endpoints, correspondence, n, params, *, use_cpci=True):
    """Embed one numerical union using averaged overlapping windows and hulls for disjoint windows.

    Keep all original endpoint graphs authoritative. A changing bond includes a bond-order or dative-direction
    change. Changed atoms have no hybridization in this numerical graph; endpoint hybridization remains intact.
    Optional CPCI preferences attract changing edges and repel unbonded outer atoms of one forming and one
    breaking edge. Their unit coefficients are numerical seed preferences, not physical charges. A supplied
    native parameter object owns its CPCI settings, which RDKit cannot expose for merging.
    """
    back, left, right, changed = _bond_graphs(endpoints, correspondence)
    union = Chem.RWMol(endpoints[0]._mol)
    union.RemoveAllConformers()
    # A broken aromatic cycle needs a numerical Kekule representation; endpoint aromaticity stays intact.
    aromatic = {edge for edge, (kind, _direction) in left.items() if kind == Chem.BondType.AROMATIC}
    affected = set(itertools.chain.from_iterable(changed & aromatic))
    if affected:
        pending = set(affected)
        while pending:
            found = {atom for edge in aromatic if pending.intersection(edge) for atom in edge} - affected
            affected |= found
            pending = found
        kekule = Chem.Mol(endpoints[0]._mol)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        for edge in aromatic:
            if affected.intersection(edge):
                bond = union.GetBondBetweenAtoms(*edge)
                bond.SetBondType(kekule.GetBondBetweenAtoms(*edge).GetBondType())
                bond.SetIsAromatic(False)
        for atom in affected:
            union.GetAtomWithIdx(atom).SetIsAromatic(False)
    for edge, (kind, direction) in left.items():
        if union.GetBondBetweenAtoms(*edge) is None:
            union.AddBond(int(direction[0]), int(direction[1]), kind)
    for edge in right.keys() - left.keys():
        union.AddBond(int(edge[0]), int(edge[1]), Chem.BondType.ZERO)
    for edge in changed:
        bond = union.GetBondBetweenAtoms(*edge)
        bond.SetBondType(Chem.BondType.ZERO)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in set(itertools.chain.from_iterable(changed)):
        union.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        union.GetAtomWithIdx(atom).SetHybridization(Chem.HybridizationType.UNSPECIFIED)
    union = union.GetMol()
    union.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(union)
    # Expand packed lower/upper triangles before reordering the product atom indices.
    lowers, uppers = [], []
    order = [j for _i, j in sorted(correspondence)]
    for side, owner in enumerate(endpoints):
        matrix = seed_bounds(owner._mol, owner.cons, params)
        lower, upper = np.tril(matrix, -1), np.triu(matrix, 1)
        indices = order if side else list(range(len(correspondence)))
        lowers.append((lower + lower.T)[np.ix_(indices, indices)])
        uppers.append((upper + upper.T)[np.ix_(indices, indices)])
    lower, upper = np.minimum(*lowers), np.maximum(*uppers)
    # Average compatible windows rather than collapsing touching intervals to a fixed distance.
    # Disjoint windows retain their hull to accommodate incompatible endpoint distances.
    compatible = np.maximum(*lowers) <= np.minimum(*uppers)
    lower = np.where(compatible, 0.5 * (lowers[0] + lowers[1]), lower)
    upper = np.where(compatible, 0.5 * (uppers[0] + uppers[1]), upper)
    cpci = _seed_cpci(left, right, changed) if use_cpci else {}
    cons = Constraints(
        distances={
            (i, j): (float(lower[i, j]), float(upper[i, j]))
            for i in range(len(correspondence))
            for j in range(i + 1, len(correspondence))
        },
        seed_cpci=cpci,
        metals=endpoints[0].cons.metals | {back[i] for i in endpoints[1].cons.metals},
        conjugation=False,
    )
    ids = seed_coordinates(union, cons, n, params)
    return union, ids


def seed_endpoint_conformers(endpoints, correspondence, n, params, *, use_cpci=True):
    """Seed from the endpoint with more bonds, preferring fewer haptic sites on ties."""
    back, left, right, changed = _bond_graphs(endpoints, correspondence)
    source_side = max(
        range(2),
        key=lambda side: (
            endpoints[side]._mol.GetNumBonds(),
            -len(endpoints[side].cons.haptic),
            -side,
        ),
    )
    owner = endpoints[source_side]
    mol = Chem.Mol(owner._mol)
    mol.RemoveAllConformers()
    preferences = {}
    if use_cpci:
        preferences = _seed_cpci(left, right, changed, back=back if source_side else None)
    cons = owner.cons.copy(conjugation=False, seed_cpci=preferences)
    mol, ids, _target = seed_conformers(mol, cons, owner.iso, n, params)
    return mol, ids, source_side


def _seed_conformers(endpoints, correspondence, n, params, *, seed_union, seed_cpci):
    """Run the selected shared or endpoint-topology seed strategy."""
    if seed_union:
        mol, ids = seed_union_conformers(endpoints, correspondence, n, params, use_cpci=seed_cpci)
        return mol, ids, None
    return seed_endpoint_conformers(endpoints, correspondence, n, params, use_cpci=seed_cpci)


def _prepare_seed_endpoints(endpoints, cons, shared, seeded_ids, source_side, params, need, max_iters):
    """Copy mapped seeds, select compatible metal hands, and guard independent endpoint relaxation."""
    prepared = []
    split = endpoints[0]._mol.GetNumAtoms() + len(endpoints[0].cons.phantoms)
    for side, owner in enumerate(endpoints):
        graph = Chem.Mol(owner._mol)
        graph.RemoveAllConformers()
        for cid in seeded_ids:
            positions = shared.GetConformer(cid).GetPositions()
            copied = np.zeros((graph.GetNumAtoms(), 3))
            for i, j in cons.seed_correspondence:
                target = i if side == 0 else j
                source = i if source_side is None or source_side == 0 else j
                copied[target] = positions[source]
            conf = Chem.Conformer(graph.GetNumAtoms())
            conf.SetId(cid)
            conf.SetPositions(copied)
            graph.AddConformer(conf, assignId=False)
        endpoint = Conformers(graph, list(seeded_ids), owner.cons.copy(conjugation=False), owner.iso, params=params)
        stationary = {pair[side] - (split if side else 0) for pair in cons.alignment_pairs or ()}
        for cid in endpoint.ids:
            public = endpoint.restored_mol(cid)
            reflect_point_stereo(public, endpoint._stated_stereo, stationary)
            rotate_axial_stereo(public, endpoint._stated_stereo, stationary)
            endpoint._mol.GetConformer(cid).SetPositions(public.GetConformer().GetPositions())
        last_stereo_error = None
        for cid in endpoint.ids:
            try:
                owner.cons = stereo_restraints(endpoint.restored_mol(cid), endpoint._stated_stereo, owner.cons)
                break
            except ValueError as error:
                last_stereo_error = error
        else:
            raise EmbeddingError(str(last_stereo_error)) from last_stereo_error
        endpoint.cons = owner.cons.copy(conjugation=False)
        prepared.append(endpoint)
    common = set(seeded_ids)
    for endpoint in prepared:
        endpoint._correct_metal_hand("coembed preparation")
        common.intersection_update(cid for cid, matches in endpoint._metal_states().items() if matches)
    selected = sorted(
        (cid for cid in seeded_ids if cid in common),
        key=lambda cid: (
            sum(
                _ligand_stereo_failure(endpoint.restored_mol(cid), endpoint._stated_stereo) is not None
                for endpoint in prepared
            ),
            sum(not _seed_geometry_matches(endpoint._mol, cid, endpoint.iso) for endpoint in prepared),
        ),
    )[:need]
    for endpoint in prepared:
        endpoint.ids = list(selected)
        # A short UFF preparation can tear an intact ligand while resolving metal restraints.
        # The core guard restores rejected endpoints to the shared seed without independent reseeding.
        endpoint._relax_once(BASE_STIFFNESS, max_iters, operation="coembed preparation")
    return prepared, selected


def seed_paired_conformers(
    mol,
    cons,
    endpoints,
    n,
    params,
    *,
    seed_relax_iters=100,
    seed_union=True,
    seed_cpci=True,
    oversample=True,
):
    """Select shared stereo-compatible seeds and prepare both endpoint copies without losing intact seeds."""
    mol.RemoveAllConformers()
    ids = []
    last_failure = None
    for attempt in range(_REPLACEMENT_ROUNDS + 1):
        trial = replace(params, seed=params.seed + attempt)
        active_union = seed_union and attempt == 0
        active_cpci = seed_cpci if active_union or not seed_union else False
        mol.SetIntProp("_rxembed_last_seed", trial.seed)
        if attempt:
            logger.info(
                "coembed: retry seed %d with %s",
                trial.seed,
                "endpoint seeding",
            )
        need = n - len(ids)
        count = _REPLACEMENT_FACTOR * need if oversample and any(owner.iso for owner in endpoints) else need
        try:
            shared, seeded_ids, source_side = _seed_conformers(
                endpoints,
                cons.seed_correspondence,
                count,
                trial,
                seed_union=active_union,
                seed_cpci=active_cpci,
            )
        except (EmbeddingError, RuntimeError) as error:
            last_failure = error if isinstance(error, EmbeddingError) else EmbeddingError(error_summary(error))
            logger.info("coembed: seed %d rejected: %s", trial.seed, error)
            continue
        if not seeded_ids:
            continue
        try:
            prepared, selected = _prepare_seed_endpoints(
                endpoints, cons, shared, seeded_ids, source_side, trial, need, seed_relax_iters
            )
        except EmbeddingError as error:
            last_failure = error
            logger.info("coembed: seed %d rejected: %s", trial.seed, error)
            continue
        if not selected:
            continue
        packed = [
            materialise_phantoms(endpoint._mol, owner.cons.haptic)
            for endpoint, owner in zip(prepared, endpoints, strict=True)
        ]
        for cid in selected:
            blocks = [graph.GetConformer(cid).GetPositions() for graph in packed]
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetPositions(np.concatenate(blocks))
            conf.SetIntProp("_rxembed_seed", trial.seed)
            conf.SetIntProp("_rxembed_source_id", cid)
            conf.SetProp("_rxembed_seed_method", "union" if active_union else "endpoint")
            conf.SetProp("_rxembed_seed_bounds", "average" if active_union else "endpoint")
            conf.SetBoolProp("_rxembed_seed_cpci", active_cpci and params.native is None)
            ids.append(mol.AddConformer(conf, assignId=True))
        if len(ids) == n:
            break
    if not ids and last_failure is not None:
        raise last_failure
    if len(ids) < n:
        logger.warning("coembed: kept %d/%d seeds; try another seed=", len(ids), n)
    return mol, ids


@dataclass
class PairedConformers(Conformers):
    """Couple two endpoint objectives while retaining the shared relaxation and recovery policy."""

    endpoints: tuple = field(kw_only=True)
    validators: tuple = field(default=(None, None), kw_only=True, repr=False)
    alignment_weight: float = field(default=ALIGNMENT_WEIGHT, kw_only=True)
    joint_gtol: float = field(default=0.1, kw_only=True)
    joint_energy_tol: float = field(default=1e-3, kw_only=True)
    max_cycles: int = field(default=100, kw_only=True)
    seed_relax_iters: int = field(default=100, kw_only=True)
    seed_union: bool = field(default=True, kw_only=True)
    seed_cpci: bool = field(default=True, kw_only=True)
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
        """Replace rejected pairs with fresh endpoint seeds without generated CPCI."""
        params = replace(params, seed=max(params.seed, self._mol.GetIntProp("_rxembed_last_seed") + 1))
        self._mol.SetIntProp("_rxembed_last_seed", params.seed + _REPLACEMENT_ROUNDS)
        logger.info("coembed: replacing rejected pairs with endpoint seeds without CPCI")
        cons = self.cons.copy()
        mol, ids = seed_paired_conformers(
            Chem.Mol(self._mol),
            cons,
            self.endpoints,
            count,
            params,
            seed_relax_iters=self.seed_relax_iters,
            seed_union=False,
            seed_cpci=False,
            oversample=False,  # The core replacement count already includes the selection margin.
        )
        self._mol.SetIntProp("_rxembed_last_seed", mol.GetIntProp("_rxembed_last_seed"))
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
            )
            for owner in self.endpoints
        )
        batch = PairedConformers(
            mol,
            ids,
            cons,
            params=params,
            endpoints=endpoints,
            validators=tuple(
                check.__func__.__get__(fresh, type(fresh)) if getattr(check, "__self__", None) is owner else check
                for check, owner, fresh in zip(self.validators, self.endpoints, endpoints, strict=True)
            ),
            alignment_weight=self.alignment_weight,
            joint_gtol=self.joint_gtol,
            joint_energy_tol=self.joint_energy_tol,
            max_cycles=self.max_cycles,
            seed_relax_iters=self.seed_relax_iters,
            seed_union=False,
            seed_cpci=False,
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

    def _joint_pass(self, stiffness, max_iters):
        """Rigidly register spectators when it lowers coupling, then optimize the complete paired objective."""
        assert self.params is not None
        split = self.endpoints[0]._mol.GetNumAtoms() + len(self.endpoints[0].cons.phantoms)
        alignment = self.cons.alignment_pairs if self.cons.alignment_pairs is not None else tuple(self.cons.distances)
        left, right = np.asarray(alignment, dtype=int).T
        for cid in self.ids:
            seed = self._mol.GetConformer(cid).GetPositions().copy()
            graphs, field_terms = [], []
            packed_offset = 0
            for owner in self.endpoints:
                # Copied coordinates can have the other endpoint's hybridisation. Let native UFF repair
                # them without preserving seed-centred conjugation torsions from that geometry.
                cons = owner.cons.copy(conjugation=False)
                target, terms, retyped, replacements = _prepare_uff(owner._mol, cons, owner.uff)
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
                    max_cycles=self.max_cycles,
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
                "stage": "joint",
                "seed_method": source.get("_rxembed_seed_method", "endpoint"),
                "seed_bounds": source.get("_rxembed_seed_bounds", "endpoint"),
                "seed_cpci": source.get("_rxembed_seed_cpci", False),
                "stiffness": stiffness,
            }
            self.joint_results[cid] = record
            self.optimizer_attempts.append(dict(record))
            self._mol.GetConformer(cid).SetPositions(positions)

        self._sync_endpoints()

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, operation="coembed", _frames=None):
        """Run one free joint optimization pass, then validate the shared pair."""
        if self.params is None:
            raise ValueError("paired relaxation needs EmbedParams; construct the pair with coembed(...)")
        self._sync_endpoints()
        self.joint_results, self.optimizer_attempts, self.recovery_batches = {}, [], []
        self.relax_failures = {}
        failed = {cid for cids in self._acceptance_failures(operation).values() for cid in cids}
        checkpoint = {cid: self._mol.GetConformer(cid).GetPositions().copy() for cid in self.ids if cid not in failed}
        self._joint_pass(stiffness, max_iters)
        failed = {cid for cids in self._acceptance_failures(operation).values() for cid in cids}
        restored = failed & checkpoint.keys()
        for cid in restored:
            self._mol.GetConformer(cid).SetPositions(checkpoint[cid])
        self._sync_endpoints()
        self._refresh_scores(stiffness)
        self._mark_validation_adjusted(restored, "joint optimization rejected; retained valid prepared pair")
        return True

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
        self._mark_validation_adjusted(adjusted, "endpoint validation changed coordinates after joint optimization")
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
