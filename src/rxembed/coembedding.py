"""Embed two graphs on one atom order as tied copies of one system: aligned reaction endpoints.

A path method (NEB, a string, a TS search) needs two endpoints that each read as their own graph and differ only
where the graphs do. Both copies are seeded, relaxed and validated as one system through the ordinary embed: each
keeps its own compiled constraints, and `Constraints.ties` ties every atom to its copy past a budget
(`mechanisms.Tie`). An atom's budget is its largest gap between the two graphs' seed windows to any other atom, so
a reacting atom moves as far as its graphs require and a spectator's tie starts at `TIE_FLOOR`.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from rdkit import Chem

from .bounds import resolve_params, seed_bounds, seed_count
from .embed import Conformers, coordination_state_failure, prepare, require_seed_count, seed_conformers
from .metal_core import metal_indices
from .metal_isomer import Isomer, paired_isomer
from .utils import combined

TIE_FLOOR = 0.3  # Å: a spectator's budget, nonzero so DG is not degenerate; unmeasured
logger = logging.getLogger(__name__)


def coembed(left, right, *, constrain=None, contacts=None, n=None, seed=None, threads=None, params=None):
    """Embed `left` and `right` as tied copies; return their `Conformers`, paired by id and in one frame.

    Both are explicit-H Mols, or both selected `Isomer`s, with the same element at every index: the atom map is the
    caller's (`rxembed.coembed` overlays the two graphs). `constrain` takes distance, angle and dihedral keys by
    atom index in that shared order and holds them in both copies. `contacts` takes the left copy's soft
    non-covalent terms in the same form, such as an assembly's H-bonds; the right copy holds each term whose atoms
    its graph bonds alike, so an H-bond that becomes the bond holds only where it is a contact. ``n=None`` takes
    the count one copy would embed with, since the tied copies share one conformer space.
    A pair is kept only when both copies pass their own acceptance. The copies carry no energies: a pair's
    restrained energy ranks neither endpoint.
    """
    _check_pair(left, right)
    constrain, contacts = _index_keys(constrain), _index_keys(contacts)
    params = resolve_params(params, seed, threads)
    held = _alike(contacts, *(spec.graph if isinstance(spec, Isomer) else spec for spec in (left, right)))
    copies = [
        prepare(spec, constrain=constrain, contacts=terms, params=params)
        for spec, terms in ((left, contacts), (right, held))
    ]
    mol, cons = copies[0][0], copies[0][1]
    count = mol.GetNumAtoms()
    budgets = _tie_budgets(copies, count, params)
    union = paired_isomer(left, right) if isinstance(left, Isomer) else combined(left, right)
    pair_mol, pair_cons, iso, graft_ref = prepare(
        union,
        constrain=_union_terms(constrain, constrain, count),
        contacts=_union_terms(contacts, held, count),
        params=params,
    )
    pair_cons = pair_cons.copy(ties={(i, count + i): float(budget) for i, budget in enumerate(budgets)})
    target = n or seed_count(mol, constrained=cons.is_constrained)
    pair_mol, ids, selected = seed_conformers(pair_mol, pair_cons, iso, target, params, graft_ref=graft_ref)
    require_seed_count(ids, selected, iso)
    if not ids:  # as `embed`: an empty pair would read as pruned
        logger.warning("coembed: DG built no seed pair; loosen constrain= or contacts=")
    pair = Conformers(pair_mol, ids, pair_cons, iso, params=params).minimize()
    return _split(pair, copies, count)


def _alike(terms, left, right):
    """Return the `terms` whose atoms `right` bonds as `left` does."""

    def bonded(graph, key):
        return {(i, j) for i, j in itertools.combinations(key, 2) if graph.GetBondBetweenAtoms(int(i), int(j))}

    return {key: value for key, value in (terms or {}).items() if bonded(left, key) == bonded(right, key)}


def _index_keys(terms):
    """Return `terms`, rejecting a key that is not atom indices, since the right copy's keys are shifted ones."""
    keys = [key for key in terms or {} if not all(isinstance(atom, (int, np.integer)) for atom in key)]
    if keys:
        raise ValueError(f"coembed() constrain= and contacts= take atom-index keys (2-4 atoms); got {keys[0]!r}")
    return terms


def _union_terms(left, right, count):
    """Return the `left` copy's terms with the `right` copy's terms `count` atoms on, or None for neither."""
    if not left and not right:
        return None
    return {**(left or {}), **{tuple(int(a) + count for a in key): value for key, value in (right or {}).items()}}


def _check_pair(left, right):
    """Reject a pair that is not two Mols or two Isomers with the same element at every index."""
    if not any(all(isinstance(spec, kind) for spec in (left, right)) for kind in (Chem.Mol, Isomer)):
        raise TypeError("coembed() takes two explicit-H Mols or two selected Isomers")
    graphs = [spec.graph if isinstance(spec, Isomer) else spec for spec in (left, right)]
    if not isinstance(left, Isomer) and any(metal_indices(graph) for graph in graphs):
        raise TypeError("a metal copy embeds through its coordination state; pass two Isomers from rx.metal(...)")
    if any(atom.GetTotalNumHs() for graph in graphs for atom in graph.GetAtoms()):
        raise ValueError("coembed() needs explicit hydrogens on both copies; pass Chem.AddHs(mol)")
    elements = [[atom.GetAtomicNum() for atom in graph.GetAtoms()] for graph in graphs]
    if elements[0] != elements[1]:
        raise ValueError(
            "the copies differ in atom count or element order; put one on the other's order (rx.coembed overlays "
            "two graphs)"
        )


def _tie_budgets(copies, count, params):
    """Return each atom's budget: its largest gap between the two copies' seed windows, at least `TIE_FLOOR`."""
    left, right = (seed_bounds(mol, cons, params)[:count, :count] for mol, cons, _iso, _ref in copies)
    lower = np.maximum(np.tril(left, -1), np.tril(right, -1))  # lower bounds sit below the diagonal
    upper = np.minimum(np.triu(left, 1), np.triu(right, 1)).T
    gap = np.clip(lower - upper, 0.0, None)
    return np.maximum(TIE_FLOOR, (gap + gap.T).max(axis=1))


def _split(pair, copies, count):
    """Return each copy's `Conformers` over the pair's ids, both in the pair's one frame."""
    out = []
    for offset, (prepared, cons, iso, _ref) in zip((0, count), copies, strict=True):
        mol = Chem.Mol(prepared)
        mol.RemoveAllConformers()
        for cid in pair.ids:
            conf = Chem.Conformer(count)
            conf.SetId(int(cid))
            conf.SetPositions(pair.mol.GetConformer(int(cid)).GetPositions()[offset : offset + count])
            mol.AddConformer(conf, assignId=False)
            coordination_state_failure(mol, cid, iso)  # records the copy's shape; the pair passed the same check
        out.append(Conformers(mol, list(pair.ids), cons, iso, unrelaxed=list(pair.unrelaxed)))
    return tuple(out)
