"""Enumerate donor-to-slot assignments, chelate bites and labels on the supplied surrogate graph."""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from rxembed.graph import frag_map, ligand_graph
from rxembed.metal.centres import vertex_atom
from rxembed.metal.polyhedra import (
    POLYHEDRA,
    VACANT,
    isomer_permutations,
    least_frames,
    point_group,
    seat_by_alignment,
    seat_properly,
    seating_frames,
    slot_form,
    vertex_dirs,
)
from rxembed.metal.stereo import chelate_links

MAX_ARRANGEMENTS = 1_000  # a resource limit on the arrangements one enumeration lists or scans, not chemistry
SPAN_TOL = 0.1  # A numerical slack when a graph-derived bite window is conditioned on RDKit bounds
# Model bite ranges by chelate ring size, shared by DG/UFF and enumeration compatibility checks.
_CHELATE_BITE = {4: (58.0, 81.0), 5: (70.0, 91.0), 6: (74.0, 104.0)}


def pinned_vertices(mol, core, metal, sites, geometry):
    """Return ``{vertex: k}`` seating each ``sites[k]`` that `core` pins, by the best fit of those donors alone.

    `core` maps pinned atoms to coordinates; a metal it leaves free stands where `mol`'s conformer puts it. One
    donor orients no polyhedron, so fewer than two give ``{}``.
    """
    pinned = [k for k, site in enumerate(sites) if site in core]
    if len(pinned) < 2:  # noqa: PLR2004
        return {}
    if metal not in core and not mol.GetNumConformers():
        raise ValueError(
            f"fix= pins coordination-sphere atoms {sorted(sites[k] for k in pinned)} but not their metal {metal}, "
            "so they state no vertex; add the metal to fix="
        )
    centre = core[metal] if metal in core else mol.GetConformer().GetPositions()[metal]
    rays = np.zeros((len(sites), 3))
    for k in pinned:
        rays[k] = np.subtract(core[sites[k]], centre)
    rays /= np.where((norm := np.linalg.norm(rays, axis=1, keepdims=True)) > 0, norm, 1.0)
    dirs = POLYHEDRA[geometry].vertex_dirs
    order = seat_properly(rays, dirs, seat_by_alignment(rays, np.array(dirs, float)))
    return {vertex: k for vertex, k in enumerate(order) if k in pinned}


def assignment_cap_error(geometry, limit=None):
    """Return the resource-limit error shared by raw and screened slot pools."""
    limit = MAX_ARRANGEMENTS if limit is None else int(limit)
    return ValueError(
        f"metal[{geometry}]: more than {limit:,} distinct constitutional slot "
        "assignments; exact enumeration is required. Use rx.isomers(mol) to keep a coordinate input's "
        "measured arrangement, or rx.isomers(rx.cxsmiles(mol)) to request its specified arrangement"
    )


def _check_assignment_cap(geometry, counts, rotations, limit):
    """Reject a provably oversized constitutional orbit pool before generating its permutations."""
    possible = math.factorial(sum(counts))
    for count in counts:
        possible //= math.factorial(count)
    if (possible + len(rotations) - 1) // len(rotations) > limit:
        raise assignment_cap_error(geometry, limit)


def chelate_bite_window(mol, a, b, donors=()):
    """Return the calibrated ring-size bite window, or ``None`` outside a 4-6 membered donor-free backbone."""
    path = Chem.GetShortestPath(mol, a, b)
    if not path:
        return None
    blocked = set(donors) - {a, b}
    queue, distances = deque([a]), {a: 0}
    while queue and b not in distances:
        atom = queue.popleft()
        for neighbour in mol.GetAtomWithIdx(atom).GetNeighbors():
            index = neighbour.GetIdx()
            if index not in blocked and index not in distances:
                distances[index] = distances[atom] + 1
                queue.append(index)
    # a strictly longer donor-free route is not the pair's backbone and gives it no bite prior
    if distances.get(b) != len(path) - 1:
        return None
    return _CHELATE_BITE.get(distances[b] + 2)


def short_chelate_pairs(mol, padded, haptic=None, distances=None):
    """Return the same-ligand donor pairs no seating may put on a trans pair.

    A model prior, not a proof: a bonded pair or a pair on a 4-6 membered chelate backbone does not reach that wide. A
    cis pair's bite is judged against the ligand, not the polyhedron. Haptic face atoms are never endpoints and block
    backbone paths. A pair with a third donor on a shortest path between them is dropped, since the shorter links
    through that donor still hold.
    """
    haptic = haptic or {}
    frag = frag_map(mol)
    real_slots = [i for i, donor in enumerate(padded) if donor is not VACANT and donor not in haptic]
    face_atoms = {atom for face in haptic.values() for atom in face}
    blockers = {padded[i] for i in real_slots} | face_atoms
    distances = Chem.GetDistanceMatrix(ligand_graph(mol)) if distances is None else distances
    linked = set()
    for i, j in itertools.combinations(real_slots, 2):
        a, b = padded[i], padded[j]
        if frag[a] != frag[b]:
            continue
        bonded = mol.GetBondBetweenAtoms(a, b) is not None
        if not bonded and chelate_bite_window(mol, a, b, blockers) is None:
            continue
        linked.add(frozenset((i, j)))
    for pair in list(linked):
        i, j = tuple(pair)
        a, b = padded[i], padded[j]
        if any(
            distances[a][padded[k]] + distances[padded[k]][b] == distances[a][b] for k in real_slots if k not in pair
        ):
            linked.discard(pair)
    return frozenset(linked)


def has_tether(donors, frag, haptic=None):
    """Return whether two coordination vertices belong to the same ligand fragment."""
    atoms = [vertex_atom(haptic, donor) for donor in donors if donor is not VACANT]
    return len(atoms) != len({frag[atom] for atom in atoms})


def _donor_triad(mol, od, haptic=None):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else None.

    That is a tridentate chelate, or exactly three monodentate donors of one element (MA3B3). MA4B2 is cis/trans, and
    en2 or en3 has no mer/fac, so neither is forced into a triad.
    """
    real = [(p, vertex_atom(haptic, od[p])) for p in range(len(od)) if od[p] is not VACANT]
    if len(real) < 3:  # noqa: PLR2004
        return None
    frag = frag_map(mol)
    by_frag = {}
    for p, d in real:
        by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():  # a tridentate chelate (one ligand, exactly 3 donors)
        if len(ps) == 3:  # noqa: PLR2004
            return tuple(ps)
    by_elem = {}
    for p, d in real:
        if len(by_frag[frag[d]]) == 1:  # else exactly three monodentate same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == 3:  # noqa: PLR2004
            return tuple(ps)
    return None


def order_label(mol, od, geometry, haptic=None):
    """Return the isomer label of vertex-ordered donors `od` on the ideal polyhedron; no conformer needed.

    Vacancies are ignored, and so is a polyhedron cis and trans do not name. Where mer and fac name triads, a triad
    holding a trans pair is mer and one holding none fac; otherwise cis or trans is judged on the minority same-element
    pair, which defines the isomerism: the 2 Cl of an MA4B2, not the 4 A.
    """
    poly = POLYHEDRA.get(geometry)
    if poly is None or not poly.cis_trans:
        return ""
    if poly.mer_fac and (tri := _donor_triad(mol, od, haptic)) is not None:
        return "mer" if any(pair in poly.trans_pairs for pair in itertools.combinations(tri, 2)) else "fac"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] is not VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(vertex_atom(haptic, od[p])).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) == 2}  # noqa: PLR2004  a cis/trans pair
    if not pairs:
        return ""  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    return "trans" if tuple(pairs[e]) in poly.trans_pairs else "cis"


def input_ordering(mol, metal, donors, geometry, haptic=None, *, classes):
    """Return the order ``donors[order[slot]]`` that seats the conformer's donors on `geometry`, else None.

    None without a conformer, a geometry record or a matching vertex count. The fit may reflect, then the hand is
    restored, and site classes and chelate links pick one proper frame, so renumbering atoms changes nothing.
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([np.zeros(3) if d is VACANT else pos[d] - pos[metal] for d in donors], float)
    dd /= np.where((norm := np.linalg.norm(dd, axis=1, keepdims=True)) > 0, norm, 1.0)
    v_ideal = np.array(dirs_ref, float)
    order = seat_properly(dd, dirs_ref, seat_by_alignment(dd, v_ideal))
    seated = [donors[index] for index in order]
    keys = [None if donor is VACANT else classes[donor] for donor in seated]
    frame = least_frames(dirs_ref, keys, chelate_links(mol, seated, haptic))[0]
    canonical = [0] * len(order)
    for source, target in enumerate(frame):
        canonical[target] = order[source]
    return canonical


def _seat_pruned_orderings(poly, rotations, linked):
    """Backtrack donor placements, dropping a branch once a linked pair sits on a trans pair of `poly`.

    Each pair is tested once, when its later vertex is placed, which is sound because a violated pair stays violated.
    Placing vertices in order keeps `itertools.permutations`' order with the dead subtrees skipped.
    """
    n = poly.cn
    trans_before = [
        [i for i in range(j) if (i, j) in poly.trans_pairs] for j in range(n)
    ]  # earlier vertices trans to j
    order = [-1] * n
    used = [False] * n

    def backtrack(vertex):
        if vertex == n:
            seated = tuple(order)
            if seated == min(tuple(order[q[k]] for k in range(n)) for q in rotations):
                yield seated
            return
        for donor in range(n):
            if used[donor] or any(frozenset((order[i], donor)) in linked for i in trans_before[vertex]):
                continue
            order[vertex], used[donor] = donor, True
            yield from backtrack(vertex + 1)
            used[donor] = False
        order[vertex] = -1

    yield from backtrack(0)


def _class_orders(donors, classes):
    """Yield one donor order per distinct string of site classes, in first-seen class order.

    At CN10, A9B and A8B2 give 10 and 45 candidates before rotations instead of 10!.
    """
    groups = {}
    for index, donor in enumerate(donors):
        groups.setdefault(classes[donor], []).append(index)
    counts = {label: len(indices) for label, indices in groups.items()}

    def strings(prefix):
        if len(prefix) == len(donors):
            yield prefix
            return
        for label in groups:
            if counts[label]:
                counts[label] -= 1
                yield from strings((*prefix, label))
                counts[label] += 1

    for string in strings(()):
        picks = {label: iter(indices) for label, indices in groups.items()}
        yield tuple(next(picks[label]) for label in string)


@dataclass(frozen=True)
class SeatingProblem:
    """Describe one sphere's donors to seat on a polyhedron and the graph facts that tell seatings apart.

    `donors` is padded with VACANT to the polyhedron size, each haptic face as its centroid key in `haptic`. `linked`
    holds the same-ligand donor pairs no seating puts on a trans pair.
    """

    mol: Chem.Mol
    donors: tuple | list
    geometry: str
    haptic: dict
    classes: dict
    distances: np.ndarray | None  # the ligand distance matrix of a tethered sphere, else None
    linked: frozenset = frozenset()


def distinct_vertex_orderings(problem, perms=None, *, retained=None, max_arrangements=None):
    """Return the vertex orderings of `problem`'s distinct coordination isomers, under exact symmetries.

    `perms` overrides the candidates: a ``fix=`` enumeration passes those that keep each frozen donor on its vertex, and
    nothing prunes them. Otherwise `retained`, the conformer's ordering, comes first, and the trans-pair rule prunes the
    tethered pool before the cap can raise. More than `max_arrangements` orderings raise. Feasibility stays the
    embedder's to judge: no graph rule proves a seating can be reached.
    """
    mol, donors, geometry, haptic = problem.mol, problem.donors, problem.geometry, problem.haptic
    dirs = vertex_dirs(geometry)
    frag = frag_map(mol)  # same ligand = same fragment
    limit = MAX_ARRANGEMENTS if max_arrangements is None else int(max_arrangements)
    tethered = has_tether(donors, frag, haptic)
    classes = problem.classes
    # Separate equivalent monodentates have one arrangement in every geometry, so skip the factorial pool.
    if (
        perms is None
        and VACANT not in donors
        and not haptic
        and len(set(classes.values())) == 1
        and len({frag[donor] for donor in donors}) == len(donors)
    ):
        return [tuple(range(len(donors)))]
    template = tuple(map(tuple, dirs))
    rotations = point_group(template)[0]
    retained = tuple(retained) if retained is not None and perms is None else None
    if perms is None and not tethered and VACANT not in donors and len(set(classes.values())) == len(donors):
        _check_assignment_cap(geometry, [1] * len(donors), rotations, limit)
        perms = isomer_permutations(geometry)
    elif perms is None and not tethered and VACANT not in donors and not haptic:
        _check_assignment_cap(geometry, list(Counter(classes[d] for d in donors).values()), rotations, limit)
        perms = itertools.chain((tuple(range(len(donors))),), _class_orders(donors, classes))
    elif perms is None:
        perms = (
            _seat_pruned_orderings(POLYHEDRA[geometry], rotations, problem.linked)
            if problem.linked
            else isomer_permutations(geometry)
        )
    if retained is not None:
        perms = itertools.chain((retained,), (order for order in perms if order != retained))
    site = {VACANT: ("vacant",)} | {d: ("donor", classes[d]) for d in donors if d is not VACANT}
    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        links = chelate_links(mol, od, haptic, problem.distances) if tethered else {}
        frames = seating_frames(template, tuple(v for v, donor in enumerate(od) if donor is not VACANT))
        form = min(slot_form(q, [site[donor] for donor in od], links) for q in frames)  # canonical over the frame group
        if form not in seen:
            seen.add(form)
            out.append(order)
            if len(out) > limit:
                raise assignment_cap_error(geometry, limit)
    return out
