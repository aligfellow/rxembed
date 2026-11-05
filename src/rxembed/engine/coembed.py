"""Co-embed a reaction's two endpoints: one shared seed, guarded endpoint preparation and coupled relaxation.

It also matches the endpoints' metal sites by least motion, weights their atoms for the coupled objective and
carries stereo from one endpoint to the other. The restrained UFF objective itself belongs to `relax`.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolTransforms

from rxembed.engine.acceptance import EmbeddingError, Failure, conformer_misses, describe_groups, group_failures
from rxembed.engine.atom_mapping import bond_sides, point_inversion_branches, site_motions
from rxembed.engine.conformers import BASE_STIFFNESS, Conformers, repeated_rejection
from rxembed.engine.seeding import REPLACEMENT_FACTOR, seed_conformers, seed_takes_shape
from rxembed.geometry.bounds import seed_bounds, seed_coordinates, seed_count
from rxembed.geometry.constraints import Constraints
from rxembed.geometry.relax import Coupling, Endpoint, coupled_uff
from rxembed.graph import frag_map, metal_indices
from rxembed.identity.smiles_io import distinct
from rxembed.identity.stereoisomers import unassigned_centres
from rxembed.metal.centres import donor_bonds
from rxembed.metal.polyhedra import VACANT, vertex_dirs
from rxembed.metal.sphere import LigandFacts
from rxembed.metal.stereo import chelate_links, equivalent_site_assignments, site_classes
from rxembed.stereo import apply_point_stereo, stereo_from_3d
from rxembed.utils import TETRAHEDRAL_TAGS, atom_label, dihedral_angle, with_conformers

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


_SAME = 1e-6  # a slot-motion change below this is rounding, not motion


_PAIR_ROUNDS = 4  # co-embed rounds: the first shared seed, then refills of the pairs still missing


_ENDPOINTS = ("reactant", "product")


class Paired(NamedTuple):
    """Hold accepted endpoints and joint diagnostics carrying each retained seed's provenance."""

    reactant: Conformers
    product: Conformers
    joint_results: dict
    optimizer_attempts: list


def coembed_pair(endpoints, coupling, n, params, *, fixed=None):
    """Prepare shared seeds, jointly relax both endpoint objectives, and replace rejected pairs in bounded rounds.

    Seed generation has four attempts and joint replacement three rounds, each drawing new union seeds.
    Keep accepted pairs. `fixed` builds one partner around a validated supplied conformer without random batches.
    """
    if fixed is None:
        operation, remedy = "coembed", "try another seed or endpoint"
        n = (
            max(seed_count(owner._mol, constrained=owner.cons.is_constrained) for owner in endpoints)
            if n is None
            else n
        )
        batch, provenance = seed_paired_conformers(endpoints, coupling, n, params, endpoints[0].sampling)
    else:
        operation = f"coembed around fixed {_ENDPOINTS[fixed]}"
        remedy = "try another supplied conformer, change the partner requirements or omit fixed="
        n = 1
        batch, provenance = _fixed_pair(endpoints, coupling, params, endpoints[0].sampling, fixed)
    slots = len(batch[0].ids)
    kept, records, attempts, rejected, last, previous = None, {}, [], [], None, None
    unspecified = _unspecified_stereo(endpoints)
    for attempt in range(1 if fixed is not None else _PAIR_ROUNDS):
        need = slots - len(records)
        if attempt:
            sampling = endpoints[0].sampling.retry("coembed replacement", attempt)
            try:
                batch, provenance = seed_paired_conformers(
                    endpoints, coupling, REPLACEMENT_FACTOR * need, params, sampling, oversample=False
                )
            except EmbeddingError as error:
                last = error
                logger.debug("coembed: replacement round %d rejected (%s)", attempt, error)
                continue
        joint, failures, raw = _relaxed_pairs(batch, coupling, params, unspecified, fixed=fixed)
        rejected.extend(failures.values())
        attempts.extend({**record, **provenance[cid], "stage": "joint"} for cid, record in raw.items())
        if batch[0].ids:
            kept, adopted = _merged(kept, batch, coupling, need)
            for cid, source in adopted.items():
                records[cid] = {**joint[source], **provenance[source], "stage": joint[source].get("stage", "joint")}
        signature = {reason.key for reason in failures.values()} if not batch[0].ids else None
        if len(records) == slots or (signature and repeated_rejection(signature, previous)):
            break
        previous = signature if attempt else None
    groups = group_failures(rejected)
    if kept is None:
        raise EmbeddingError(
            f"{operation}: kept no pairs ({describe_groups(groups)}); {remedy}", failures=groups
        ) from last
    if len(records) < n:
        logger.warning("%s: kept %d/%d pairs; %s", operation, len(records), n, remedy)
    ranked = sorted(kept[0].ids, key=lambda cid: pair_score(kept, coupling, cid))
    kept = [_renumbered(endpoint, ranked) for endpoint in kept]
    return Paired(kept[0], kept[1], {new: records[old] for new, old in enumerate(ranked)}, attempts)


def _renumbered(endpoint, ranked):
    """Return `endpoint` with pair `ranked[k]` as conformer k, so pair 0 scores lowest; `_adopt` carries its records.

    Capped preparation is not a failure to relax: the joint pass is the result, and its convergence is in the joint
    records, so no pair is listed as unrelaxed.
    """
    mol = Chem.Mol(endpoint._mol)
    mol.RemoveAllConformers()
    out = replace(endpoint, _mol=mol, ids=[], energies={}, unrelaxed=[], relax_failures={}, _verdicts={})
    out._uff = replace(endpoint._uff, statuses={}, snapshots=None)
    for new, old in enumerate(ranked):
        out._adopt(new, endpoint, old)
    out.unrelaxed = []
    return out


def _fixed_pair(endpoints, coupling, params, sampling, fixed):
    """Return the endpoints, the supplied one frozen and its partner prepared in its frame, and each pair's source."""
    owner = endpoints[fixed]
    owner.cons = replace(owner.cons, frozen=frozenset(range(owner._mol.GetNumAtoms())))
    batch = _shared_pair(endpoints, owner._mol, owner.ids, coupling, sampling, fixed=fixed)
    _prepare_pairs(batch, params, 1, fixed=fixed)
    provenance = {
        cid: {"seed": None, "source_id": cid, "seed_method": "fixed", "seed_bounds": "endpoint", "seed_cpci": False}
        for cid in batch[0].ids
    }
    return batch, provenance


def seed_paired_conformers(endpoints, coupling, n, params, sampling, *, oversample=True):
    """Return the endpoints with prepared shared seeds, and each pair's provenance, over four attempts at most.

    Each attempt draws new seeds from the endpoints' union graph.

    When no union seed has passed by the last attempt, that attempt seeds one endpoint instead, without CPCI: some
    unions, such as a catalyst's contact grips with ZERO-order changes, have bounds native seeding cannot embed.
    `oversample` draws four times the missing pairs for a metal species, whose seeds miss its arrangement more often.
    """
    kept, provenance, last = None, {}, None
    for attempt in range(_PAIR_ROUNDS):
        need = n - len(provenance)
        if need <= 0:
            break
        trial = sampling.retry("coembed seed", attempt)
        count = REPLACEMENT_FACTOR * need if oversample and any(owner.species.centres for owner in endpoints) else need
        union = kept is not None or attempt < _PAIR_ROUNDS - 1
        try:
            shared, ids = (seed_union_conformers if union else seed_endpoint_conformers)(endpoints, count, trial)
            batch = _shared_pair(endpoints, shared, ids, coupling, trial)
            _prepare_pairs(batch, params, need)
        except (EmbeddingError, RuntimeError) as error:
            last = error
            logger.debug("coembed: seed %d rejected (%s)", trial.seed, error)
            continue
        if batch[0].ids:
            kept, adopted = _merged(kept, batch, coupling, need)
            for cid, source in adopted.items():
                provenance[cid] = {
                    "seed": trial.seed,
                    "source_id": source,
                    "seed_method": "union" if union else "endpoint",
                    "seed_bounds": "average" if union else "endpoint",
                    "seed_cpci": union and trial.params.native is None,
                }
    if kept is None:
        raise EmbeddingError("coembed: no shared seed passed preparation; try another seed=") from last
    if len(provenance) < n:
        logger.warning("coembed: kept %d/%d shared seeds; try another seed=", len(provenance), n)
    return kept, provenance


def _merged(kept, batch, coupling, need):
    """Return `kept` with the `need` lowest-scoring pairs of `batch` in the lowest ids it does not hold, else `batch`.

    A refilled pair takes a dropped pair's id, as `_replace_failed` does, so the ids stay in order without gaps.
    """
    if len(batch[0].ids) > need:
        ranked = sorted(batch[0].ids, key=lambda cid: pair_score(batch, coupling, cid))
        for endpoint in batch:
            endpoint._remove(ranked[need:])
            endpoint.ids = ranked[:need]
    if kept is None:
        return batch, {cid: cid for cid in batch[0].ids}
    free = (cid for cid in itertools.count() if cid not in kept[0].ids)
    adopted = {}
    for src, cid in zip(batch[0].ids, free, strict=False):
        adopted[cid] = src
        for mine, theirs in zip(kept, batch, strict=True):
            mine._adopt(cid, theirs, src)
            mine.ids.sort()
    return kept, adopted


def pair_score(endpoints, coupling, cid):
    """Return pair `cid`'s score: both endpoints' restrained-UFF energies plus `coupling`'s penalty, else inf."""
    if any(cid not in endpoint.energies for endpoint in endpoints):
        return math.inf
    positions = [endpoint._mol.GetConformer(cid).GetPositions() for endpoint in endpoints]
    value = sum(endpoint.energies[cid] for endpoint in endpoints) + coupling.penalty(*positions).value
    return value if np.isfinite(value) else math.inf


def _bond_graphs(endpoints):
    """Read endpoint bonds in their shared atom order, including dative direction and bond-order changes."""
    graphs = []
    for endpoint in endpoints:
        edges = {}
        for bond in endpoint.specification.mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            key = tuple(sorted((a, b)))
            direction = (a, b) if str(bond.GetBondType()).startswith("DATIVE") else key
            edges[key] = bond.GetBondType(), direction
        graphs.append(edges)
    left, right = graphs
    changed = {edge for edge in left.keys() | right.keys() if left.get(edge) != right.get(edge)}
    return left, right, changed


def _union_graph(endpoints, left, right, changed):
    """Build the union graph, changing bonds at ZERO order, clearing only changed aromatic and local stereo state."""
    union = Chem.RWMol(endpoints[0]._mol)
    union.RemoveAllConformers()
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
        for edge in (edge for edge in aromatic if affected.intersection(edge)):
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
    return union


def seed_union_conformers(endpoints, n, sampling):
    """Embed averaged compatible endpoint windows and disjoint-window hulls on one ZERO-order union graph.

    CPCI attracts changing edges with -1 and repels unbonded outer atoms of forming/breaking edge pairs with +1.
    Endpoint chemical graphs remain authoritative; caller-supplied native parameters retain their own CPCI.
    """
    left, right, changed = _bond_graphs(endpoints)
    union = _union_graph(endpoints, left, right, changed)
    lowers, uppers = [], []
    real = endpoints[0]._mol.GetNumAtoms()
    for owner in endpoints:
        matrix = seed_bounds(owner._mol, owner.cons, sampling)[:real, :real]
        lower, upper = np.tril(matrix, -1), np.triu(matrix, 1)
        lowers.append(lower + lower.T)
        uppers.append(upper + upper.T)
    compatible = np.maximum(*lowers) <= np.minimum(*uppers)
    lower = np.where(compatible, 0.5 * (lowers[0] + lowers[1]), np.minimum(*lowers))
    upper = np.where(compatible, 0.5 * (uppers[0] + uppers[1]), np.maximum(*uppers))
    cpci = dict.fromkeys(changed, -1.0)
    repelled = {
        tuple(sorted(set(first) ^ set(second)))
        for first in left.keys() - right.keys()
        for second in right.keys() - left.keys()
        if len(set(first) & set(second)) == 1
    }
    cpci.update(dict.fromkeys(repelled - left.keys() - right.keys(), 1.0))
    cons = Constraints(
        distances={(i, j): (float(lower[i, j]), float(upper[i, j])) for i in range(real) for j in range(i + 1, real)},
        seed_cpci=cpci,
        metals=endpoints[0].cons.metals | endpoints[1].cons.metals,
        conjugation=False,
    )
    with rdBase.BlockLogs():  # UFF cannot type the union's changing atoms and says so; every pair is set above
        return union, seed_coordinates(union, cons, n, sampling)


def seed_endpoint_conformers(endpoints, n, sampling):
    """Seed the endpoint with more bonds and fewer haptic sites, without generated CPCI, when no union seed passes."""
    side = max((0, 1), key=lambda k: (endpoints[k]._mol.GetNumBonds(), -len(endpoints[k].cons.haptic), -k))
    owner = endpoints[side]
    cons = replace(owner.cons, conjugation=False, seed_cpci={})
    seeds = seed_conformers(Chem.Mol(owner._mol), cons, owner.species, n, sampling)
    seeds.require(owner.species, cons)
    return seeds.mol, seeds.ids


def _shared_pair(endpoints, shared, ids, coupling, sampling, *, fixed=None):
    """Copy the shared coordinates to each endpoint and prepare its own specified stereo restraints."""
    if not ids:
        raise EmbeddingError("coembed: native seeding returned no conformers; try another seed=")
    batch = []
    stationary = set(coupling.atoms[: coupling.spectators].tolist())
    for side, owner in enumerate(endpoints):
        cons = owner.cons if side == fixed else replace(owner.cons, conjugation=False)
        endpoint = replace(
            owner, _mol=with_conformers(owner._mol, shared, ids), ids=list(ids), cons=cons, sampling=sampling
        )
        endpoint._coordinates_changed()
        if side != fixed:
            for cid in endpoint.ids:
                restored = endpoint._restored_mol(cid)
                prepare_copied_stereo(restored, endpoint.species.stereo, stationary)
                endpoint._mol.GetConformer(cid).SetPositions(restored.GetConformer().GetPositions())
            last = None
            for cid in endpoint.ids:
                try:
                    endpoint.cons = stereo_restraints(endpoint._restored_mol(cid), endpoint.species.stereo, cons)
                    break
                except ValueError as error:
                    last = error
            else:
                raise EmbeddingError(str(last)) from last
        batch.append(endpoint)
    return batch


def _prepare_pairs(batch, params, need, *, fixed=None):
    """Select shared metal hands and rank ligand stereo and shape before guarded short endpoint preparation."""
    selected = [
        cid
        for cid in batch[0].ids
        if not any(
            miss.kind in {"hand", "face"}
            for endpoint in batch
            for miss in conformer_misses(endpoint.specification, endpoint._mol, cid)
        )
    ]
    selected.sort(
        key=lambda cid: (
            sum(
                any(
                    m.kind in {"point", "ez", "axis"}
                    for m in conformer_misses(endpoint.specification, endpoint._mol, cid)
                )
                for endpoint in batch
            ),
            sum(not seed_takes_shape(endpoint._mol, cid, endpoint.species) for endpoint in batch),
        )
    )
    selected = selected[:need]
    for side, endpoint in enumerate(batch):
        endpoint._remove(set(endpoint.ids) - set(selected))
        endpoint.ids = list(selected)
        if side != fixed and selected and params.seed_relax_iters:
            endpoint._relax(BASE_STIFFNESS, params.seed_relax_iters, "coembed preparation", mirror=False)


def _relaxed_pairs(batch, coupling, params, unspecified, *, fixed=None):
    """Run one joint pass, restore valid prepared pairs if rejected, and score the retained native objective."""
    failed = _pair_failures(batch, unspecified)
    checkpoint = {
        cid: [endpoint._mol.GetConformer(cid).GetPositions().copy() for endpoint in batch]
        for cid in batch[0].ids
        if cid not in failed
    }
    targets = [Endpoint(mol=endpoint._mol, cons=endpoint.cons, record=endpoint._uff) for endpoint in batch]
    objectives = coupled_uff(targets, coupling, batch[0].ids, BASE_STIFFNESS, params, fixed=fixed)
    optimized = {
        cid: [endpoint._mol.GetConformer(cid).GetPositions().copy() for endpoint in batch] for cid in batch[0].ids
    }
    failures = _pair_failures(batch, unspecified)
    restored = failures.keys() & checkpoint.keys()
    for cid in restored:
        for endpoint, pos in zip(batch, checkpoint[cid], strict=True):
            endpoint._mol.GetConformer(cid).SetPositions(pos)
    if restored:
        failures = _pair_failures(batch, unspecified)
    records = {}
    for cid, objective in objectives.items():
        positions = [endpoint._mol.GetConformer(cid).GetPositions() for endpoint in batch]
        failed_run = objective.record["stopping_reason"] == "failure"
        energies = [math.inf] * len(batch) if failed_run else objective.energies(positions)
        for endpoint, energy in zip(batch, energies, strict=True):
            endpoint.energies[cid] = energy
        record: dict = dict(
            objective.record,
            alignment_score=coupling.alignment_rmsd(*positions),
        )
        if cid in restored:  # validation only reads; a rejected pair goes back to its prepared coordinates
            record.update(
                stage="validation",
                converged=False,
                stopping_reason="validation_adjusted",
                message="endpoint validation restored the prepared coordinates",
                max_force=math.inf,
                score_history=(),
                score_change=math.inf,
            )
        if cid in restored and not failed_run:
            record["score"] = pair_score(batch, coupling, cid)
        records[cid] = record
    for endpoint in batch:
        endpoint._remove(failures)
    raw = {
        cid: dict(objective.record, alignment_score=coupling.alignment_rmsd(*optimized[cid]))
        for cid, objective in objectives.items()
    }
    return records, failures, raw


def _pair_failures(batch, unspecified):
    """Return each pair id's first failure, named by endpoint: either endpoint's gates, the reactant's first.

    A pair both endpoints accept fails as ``'unspecified_stereo'`` when an `Unspecified` element reads one configuration
    in its reactant and the other in its product, since its two frames are then two stereoisomers.
    """
    failures = {}
    for name, endpoint in zip(_ENDPOINTS, batch, strict=True):
        for reason, cids in endpoint._acceptance_failures().items():
            named = replace(reason, detail=f"{name}: {reason.detail}")
            for cid in cids:
                failures.setdefault(cid, named)
    graph = batch[0].specification.mol
    for cid in [cid for cid in batch[0].ids if cid not in failures] if unspecified else ():
        reactant, product = (endpoint._mol.GetConformer(int(cid)).GetPositions() for endpoint in batch)
        if element := next((e for e in unspecified if e.configuration(reactant) != e.configuration(product)), None):
            names = "=".join(atom_label(graph, atom) for atom in element.atoms)
            detail = f"unspecified {names} reads opposite configurations in the reactant and the product"
            failures[cid] = Failure("unspecified_stereo", detail, atoms=element.atoms)
    return failures


@dataclass(frozen=True, slots=True, kw_only=True)
class Unspecified:
    """Hold one stereo element both endpoints leave unspecified and the reaction leaves bonded alike.

    `frame` is four atoms in the endpoints' one atom order, so no CIP rank, which a nearby substitution can turn,
    enters: a centre's neighbours by index, closed by the centre when it has three, or a double bond's lowest-indexed
    substituent at each end. `read` turns the frame's dihedral (radians) into the configuration: `math.sin` reads a
    centre's hand, `math.cos` whether a double bond is cis.
    """

    atoms: tuple
    frame: tuple
    read: Callable[[float], float]

    def configuration(self, positions):
        """Return the configuration `positions` give the element, as the sign `read` gives its frame's dihedral."""
        return self.read(math.radians(dihedral_angle(*positions[list(self.frame)]))) > 0.0


def _unspecified_stereo(endpoints):
    """Return an `Unspecified` for each centre and double bond neither endpoint specifies, bonded alike in both.

    Either endpoint's graph may name the element as unassigned: a nearby substitution can make or unmake a stereocentre
    whose own bonds stay. An element either endpoint specifies stays with that endpoint's gates, and one whose atoms the
    reaction rebonds is the reaction's; one inside a group a specified configuration change turns is still read. An
    atropisomer axis is not read, nor a bond that is not an uncumulated double bond in both endpoints.
    """
    graphs = [endpoint.specification.mol for endpoint in endpoints]
    specified = {tuple(sorted(e.atoms)) for endpoint in endpoints for e in endpoint.species.stereo.elements}
    found = {element for graph in graphs for element in unassigned_centres(graph, metal_indices(graph))}
    out = []
    for element in sorted(found, key=lambda element: element.atoms):
        atoms = element.atoms
        here, there = (
            [sorted(n.GetIdx() for n in side.GetAtomWithIdx(a).GetNeighbors()) for a in atoms] for side in graphs
        )
        if atoms in specified or here != there:
            continue
        ends = [[n for n in neighbours if n not in atoms] for neighbours in here]
        match element.kind:
            case "point":
                out.append(Unspecified(atoms=atoms, frame=(*ends[0], *atoms)[:4], read=math.sin))
            case "ez" if all(
                side.GetBondBetweenAtoms(*atoms).GetBondType() == Chem.BondType.DOUBLE
                and Chem.HybridizationType.SP not in {side.GetAtomWithIdx(a).GetHybridization() for a in atoms}
                for side in graphs
            ):
                out.append(Unspecified(atoms=atoms, frame=(ends[0][0], *atoms, ends[1][0]), read=math.cos))
    return out


def least_motion(left, right):
    """Return the reactant and product `Species`, on one atom order, relabelled so the fewest sites move.

    A Species names a stereoisomer; which of its equivalent sites fills which slot is a labelling that the shared atom
    order makes physical. From the given labellings, each endpoint in turn takes the labelling that moves the fewest
    sites against the other's (`site_motions`), then the least RMS shift of their slot directions, until neither
    improves; a tie keeps the given labelling.
    """
    options = [_labellings(species) for species in (left, right)]
    best = [left.centres, right.centres]
    score = _cost(*best)
    improved = True
    while improved:
        improved = False
        for side in (1, 0):
            for centres in options[side]:
                trial = [centres, best[1]] if side == 0 else [best[0], centres]
                motion = _cost(*trial)
                if motion < (score[0], score[1] - _SAME):  # fewer moving sites, else a smaller shift
                    best, score, improved = trial, motion, True
    return tuple(
        species if centres == species.centres else replace(species, centres=centres)
        for species, centres in zip((left, right), best, strict=True)
    )


def _labellings(species):
    """Return the centres of `species` and of each relabelling that keeps its stereoisomer; coordinates bind it.

    At each centre a relabelling moves the atoms of sites between slots of equal class and chelate links
    (`equivalent_site_assignments`), each slot keeping its face winding, as the canonical notes do
    (`smiles_io.arrangement_notes`). Where a ligand binds two centres, a relabelling that changes how they stand names
    another species (`distinct`).
    """
    if species.mol.GetNumConformers():
        return [species.centres]
    facts = LigandFacts(species.mol)
    graph = facts.mol
    per_state = []
    for state in species.centres:
        vertices, haptic, _winding, _donors = species.sites[state.atom]
        found = site_classes(graph, vertices, haptic, facts.roles)
        classes = [None if site is VACANT else found[site] for site in vertices]
        per_state.append(
            [
                state._replace(
                    vertices=tuple(
                        site if site is VACANT else replace(site, atoms=state.vertices[assignment[slot]].atoms)
                        for slot, site in enumerate(state.vertices)
                    )
                )
                for assignment in equivalent_site_assignments(classes, chelate_links(graph, vertices, haptic))
            ]
        )
    labellings = list(itertools.product(*per_state))
    owner = frag_map(graph)
    bound = {(owner[donor], metal) for donor, metal in donor_bonds(species.mol)}
    if len(bound) == len({ligand for ligand, _metal in bound}):  # no ligand binds two centres
        return labellings
    return [centres for centres in labellings if len(distinct([species, replace(species, centres=centres)])) == 1]


def _cost(left, right):
    """Return the sort key of how far centres `right` move against `left`: moving sites, then RMS slot shift."""
    return sum(map(len, site_motions(left, right).values())), _slot_motion(left, right)


def _slot_motion(left, right):
    """Return the summed RMS shift of the slot directions of the sites both endpoints bind, one rotation per metal."""
    total = 0.0
    for state in left:
        other = next((s for s in right if s.atom == state.atom), None)
        if other is None:
            continue
        a, b = _slot_dirs(state), _slot_dirs(other)
        shared = sorted(a.keys() & b.keys(), key=sorted)
        if len(shared) < 2:  # noqa: PLR2004 - a rotation needs two directions
            continue
        x, y = np.array([a[key] for key in shared]), np.array([b[key] for key in shared])
        u, _s, vt = np.linalg.svd(x.T @ y)
        rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        total += float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))
    return total


def _slot_dirs(state):
    """Return each bound site's polyhedron direction, keyed by its atoms."""
    dirs = vertex_dirs(state.geometry)
    return {frozenset(site.atoms): np.asarray(dirs[k]) for k, site in enumerate(state.vertices) if site is not VACANT}


def coupling(spectators, reacting, params):
    """Couple corresponding atoms at zero displacement with separately normalized spectator and reacting weights."""
    held = len(spectators)
    weights = [params.alignment_weight / held for _atom in spectators]
    weights += [params.react_weight / len(reacting) for _atom in reacting]
    return Coupling(
        atoms=np.asarray([*spectators, *reacting], dtype=int),
        weights=np.array(weights),
        spectators=held,
    )


def prepare_copied_stereo(mol, stereo, stationary):
    """Reflect mismatched points and rotate axial hands of `stereo`, keeping stationary groups and other stereo.

    Torsion half-spaces handle E/Z. Cyclic axes and inseparable inversions are left to the identity gate and recovery.
    """
    for centre, wanted in stereo.points.items():
        _reflect_point(mol, centre, wanted, stationary)
    conf = mol.GetConformer()
    for pair, code in stereo.axes.items():
        found = stereo_from_3d(mol, exclude=metal_indices(mol)).axes.get(pair)
        if found in (None, code):
            continue
        if (sides := bond_sides(mol, tuple(pair))) is None:  # a ring bond has no side free to turn
            continue
        # the side that moves fewest stationary atoms turns, then fewest heavy atoms, then fewest atoms
        moving = min(sides, key=lambda side: _moving_cost(mol, side, stationary))
        left, right = sorted(pair, key=lambda atom: atom in moving)
        first = next(n.GetIdx() for n in mol.GetAtomWithIdx(left).GetNeighbors() if n.GetIdx() != right)
        last = next(n.GetIdx() for n in mol.GetAtomWithIdx(right).GetNeighbors() if n.GetIdx() != left)
        phi = rdMolTransforms.GetDihedralDeg(conf, first, left, right, last)
        rdMolTransforms.SetDihedralDeg(conf, first, left, right, last, -phi)


def _reflect_point(mol, centre, wanted, stationary):
    """Try separable inversions of `centre` that keep every other hand and every metal-containing branch."""
    excluded = set(metal_indices(mol))
    before = stereo_from_3d(mol, exclude=excluded)
    if before.points.get(centre) == wanted:
        return
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    branches = sorted(
        point_inversion_branches(mol, centre), key=lambda item: _moving_cost(mol, item.moving, stationary)
    )
    for moving, anchors in branches:
        a, b, c = positions[anchors]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length <= np.finfo(float).eps:
            continue
        normal /= length
        indices = sorted(moving)
        trial = positions.copy()
        trial[indices] -= 2 * ((trial[indices] - a) @ normal)[:, None] * normal
        conf.SetPositions(trial)
        after = stereo_from_3d(mol, exclude=excluded)
        if (
            after.points == {**before.points, centre: wanted}
            and after.bonds == before.bonds
            and after.axes == before.axes
        ):
            break
        conf.SetPositions(positions)


def stereo_restraints(mol, stereo, cons):
    """Translate endpoint stereo into native torsion half-spaces without choosing reaction approach angles."""
    terms = dict(cons.dihedrals)
    for centre in stereo.points:
        atom = mol.GetAtomWithIdx(centre)
        neighbours = [neighbour.GetIdx() for neighbour in atom.GetNeighbors()]
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            apply_point_stereo(mol, stereo, {centre})
        tag = atom.GetChiralTag()
        if tag not in TETRAHEDRAL_TAGS or len(neighbours) not in (3, 4):
            raise ValueError(f"could not assign a native point-stereo force at atom {centre}; inspect endpoint stereo")
        quartet = (
            tuple(neighbours)
            if len(neighbours) == 4  # noqa: PLR2004 - explicit tetrahedron
            else (neighbours[0], centre, neighbours[1], neighbours[2])
        )
        terms.setdefault(quartet, (0.0, 180.0) if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else (-180.0, 0.0))
    for bond in mol.GetBonds():
        atoms = list(bond.GetStereoAtoms())
        if len(atoms) != 2:  # noqa: PLR2004 - one stereo reference at each end
            continue
        tag = bond.GetStereo()
        if tag in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            window = (90.0, 270.0)
        elif tag in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            window = (-90.0, 90.0)
        else:
            continue
        terms.setdefault((atoms[0], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), atoms[1]), window)
    axes = stereo_from_3d(mol, exclude=metal_indices(mol)).axes if stereo.axes else {}
    for axis, wanted in stereo.axes.items():
        if axes.get(axis) != wanted:
            raise ValueError(
                f"could not prepare requested axial stereo {wanted} on {sorted(axis)}; try another starting geometry"
            )
        first, last = sorted(axis)
        left = next(a.GetIdx() for a in mol.GetAtomWithIdx(first).GetNeighbors() if a.GetIdx() != last)
        right = next(a.GetIdx() for a in mol.GetAtomWithIdx(last).GetNeighbors() if a.GetIdx() != first)
        quartet = left, first, last, right
        phi = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *quartet)
        terms.setdefault(quartet, (0.0, 180.0) if phi >= 0 else (-180.0, 0.0))
    return replace(cons, dihedrals=terms)


def _moving_cost(mol, atoms, stationary):
    """Rank a group that could move: fewest stationary atoms, then fewest heavy atoms, then fewest atoms."""
    heavy = sum(mol.GetAtomWithIdx(atom).GetAtomicNum() != 1 for atom in atoms)
    return len(atoms & stationary), heavy, len(atoms), min(atoms)
