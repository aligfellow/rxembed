"""Enumerate distinct donor assignments to metal-polyhedron slots."""

from __future__ import annotations

import itertools
import math
from collections import deque

import numpy as np
from rdkit import Chem

from .metal_core import VACANT, _frag_map, _ligand_distance_matrix, _vertex_atom
from .metal_polyhedron import (
    _MAX_EXHAUSTIVE_ORBITS,
    CHELATE_SPAN_ANGLE,
    POLYHEDRA,
    _seat_by_alignment,
    _vertex_angle,
    canonical_slots,
    hull_edges,
    isomer_permutations,
    point_group,
    seat_properly,
    vertex_dirs,
)
from .metal_stereo import chelate_links, site_classes

TRANS_ANGLE = 150  # same-element donor pairs beyond this are trans
_PAIR = 2
_TRIAD = 3
_SPAN_TOL = 0.1  # A numerical slack when a graph-derived bite window is conditioned on RDKit bounds
# Model bite ranges by chelate ring size, shared by DG/UFF and enumeration compatibility checks.
_CHELATE_BITE = {4: (58.0, 81.0), 5: (70.0, 91.0), 6: (74.0, 104.0)}


def _assignment_cap_error(geometry, limit=None):
    """Return the resource-limit error shared by raw and screened slot pools."""
    limit = _MAX_EXHAUSTIVE_ORBITS if limit is None else int(limit)
    return ValueError(
        f"metal[{geometry}]: more than {limit:,} distinct constitutional slot "
        "assignments; exact enumeration is required. Use rx.embed(mol) without metal= to retain "
        "coordinate input, rx.metal(mol, observed_only=True) for only its measured arrangement, "
        "or rx.metal(rx.cxsmiles(mol)) to request its stated arrangement"
    )


def _check_assignment_cap(geometry, counts, rotations, limit=None):
    """Reject a provably oversized constitutional orbit pool before generating its permutations."""
    limit = _MAX_EXHAUSTIVE_ORBITS if limit is None else int(limit)
    possible = math.factorial(sum(counts))
    for count in counts:
        possible //= math.factorial(count)
    if (possible + len(rotations) - 1) // len(rotations) > limit:
        raise _assignment_cap_error(geometry, limit)


def _chelate_bite_window(mol, a, b, donors=()):
    """Return the ring-size census bite window, or ``None`` outside a 4-6 membered donor-free backbone.

    Existence and census test only: `metal_constraints._seated_bites` is the sole caller that also needs the
    native ligand reach triangle, and it builds that itself (`mechanisms._triangle_angles`) around this census.
    `_chelate_edge_links` and the long-arc skip in `metal_enumeration._chelate_span_failure` read this
    existence test alone, same as before.
    """
    if not 0 <= a < mol.GetNumAtoms() or not 0 <= b < mol.GetNumAtoms():
        return None  # a haptic centroid is a coordination site, not an atom with a ligand-backbone path
    path = Chem.GetShortestPath(mol, a, b)
    if not path:
        return None
    blocked = {d for d in donors if 0 <= d < mol.GetNumAtoms()} - {a, b}
    queue, distances = deque([a]), {a: 0}
    while queue and b not in distances:
        atom = queue.popleft()
        for neighbour in mol.GetAtomWithIdx(atom).GetNeighbors():
            index = neighbour.GetIdx()
            if index not in blocked and index not in distances:
                distances[index] = distances[atom] + 1
                queue.append(index)
    # Equal shortest routes through and around a third donor are insertion-order equivalent. A strictly longer
    # donor-free route is not the pair's chelate backbone and must not supply its bite prior.
    if distances.get(b) != len(path) - 1:
        return None
    return _CHELATE_BITE.get(distances[b] + 2)


def _chelate_edge_links(mol, padded, haptic=None, distances=None):
    """Return same-ligand donor-position pairs the edge rule holds to a polyhedron hull edge.

    A chemistry claim, not a no-loss rule: 0 violations across 3,373 crystal reference centres for a
    `_chelate_bite_window`-linked pair (a 2-4 bond, 4-6 membered chelate backbone), after the through-donor
    exclusion below. A directly-bonded pair (a 3-membered M-a-b ring, held apart at only ~50-70 deg) is
    added on geometric grounds, not that measurement: its bite is tighter still, so it cannot span a
    non-edge vertex pair either (`ZUDWUQ`'s cyclo-As6 ring: 3 raw isomers drop to the 1 perimeter one).
    Haptic face atoms never count as a link endpoint and always block a backbone path between two others.
    Escapes: `screen=False`, `observed_only=True`, a stated CX arrangement (`rx.metal(rx.cxsmiles(mol))`),
    and `fix=` all bypass this rule, same as the pair rule it is folded into.

    Through-donor exclusion: when a third donor `c` sits astride the a-b backbone (`d(a,c) + d(c,b) ==
    d(a,b)` on the plain graph distance), the a-b link is dropped as redundant; the shorter a-c and c-b links
    still hold. Without it, a tridentate whose outer amides reach the metal only via a coordinated central N
    (NONVIV) would wrongly force those two outer donors onto a hull edge as well as the centre.
    """
    haptic = haptic or {}
    frag = _frag_map(mol)
    real_slots = [i for i, donor in enumerate(padded) if donor != VACANT and donor not in haptic]
    face_atoms = {atom for face in haptic.values() for atom in face}
    blockers = {padded[i] for i in real_slots} | face_atoms
    distances = _ligand_distance_matrix(mol) if distances is None else distances
    linked = set()
    for i, j in itertools.combinations(real_slots, 2):
        a, b = padded[i], padded[j]
        if frag[a] != frag[b]:
            continue
        bonded = mol.GetBondBetweenAtoms(a, b) is not None
        if not bonded and _chelate_bite_window(mol, a, b, blockers) is None:
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


def _has_tether(donors, frag, haptic=None):
    """Return whether two coordination vertices belong to the same ligand fragment."""
    atoms = [_vertex_atom(haptic, donor) for donor in donors if donor != VACANT]
    return len(atoms) != len({frag[atom] for atom in atoms})


def _octahedral_triad(mol, od, haptic=None):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else ``None``.

    Either a tridentate chelate (exactly 3 donors of one ligand fragment) or exactly 3 monodentate donors of
    one element (an MA3B3 set); ``None`` otherwise, and then cis/trans is used. The exactly-3 and monodentate
    conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate (en2, en3)
    have no mer/fac, so neither must be forced into a triad.
    """
    real = [(p, _vertex_atom(haptic, od[p])) for p in range(len(od)) if od[p] != VACANT]
    if len(real) < _TRIAD:
        return None
    frag = _frag_map(mol)
    by_frag = {}
    for p, d in real:
        by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():  # a tridentate chelate (one ligand, exactly 3 donors)
        if len(ps) == _TRIAD:
            return tuple(ps)
    by_elem = {}
    for p, d in real:
        if len(by_frag[frag[d]]) == 1:  # else exactly three monodentate same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == _TRIAD:
            return tuple(ps)
    return None


def order_label(mol, donors, geometry, order, haptic=None):
    """Build the isomer label from the ideal polyhedron; no conformer needed.

    Vacant vertices are ignored. Octahedral with a donor triad is mer/fac (one trans pair in the triad means
    mer, none means fac); otherwise cis/trans, judged on the minority same-element donor pair, which is the
    set whose placement defines the isomerism: the 2 Cl of an MA4B2, not the 4 A that always have a trans
    pair.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return f"isomer{order}"
    if not POLYHEDRA[geometry].geometric_isomerism:
        return ""  # no cis/trans distinction for this geometry
    od = [donors[k] for k in order]
    if geometry == "octahedral":
        tri = _octahedral_triad(mol, od, haptic)
        if tri is not None:
            trans = sum(
                1 for i in range(3) for j in range(i + 1, 3) if _vertex_angle(dirs[tri[i]], dirs[tri[j]]) > TRANS_ANGLE
            )
            return "fac" if trans == 0 else "mer"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] != VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(_vertex_atom(haptic, od[p])).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) == _PAIR}
    if not pairs:
        return ""  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    ps = pairs[e]
    trans = any(
        _vertex_angle(dirs[ps[i]], dirs[ps[j]]) >= TRANS_ANGLE for i in range(len(ps)) for j in range(i + 1, len(ps))
    )
    return "trans" if trans else "cis"


def realised_label(mol, metal, donors, cid, geometry=None):
    """Label a realised coordination arrangement as cis or trans."""
    polyhedron = POLYHEDRA.get(geometry)
    if polyhedron is not None and not polyhedron.geometric_isomerism:
        return ""
    positions = mol.GetConformer(cid).GetPositions()
    for a in range(len(donors)):
        for b in range(a + 1, len(donors)):
            same = mol.GetAtomWithIdx(donors[a]).GetSymbol() == mol.GetAtomWithIdx(donors[b]).GetSymbol()
            angle = _vertex_angle(positions[donors[a]] - positions[metal], positions[donors[b]] - positions[metal])
            if same and angle >= TRANS_ANGLE:
                return "trans"
    return "cis"


def input_ordering(mol, metal, donors, geometry, haptic=None, coordination=(), *, classes=None):
    """Fit input donors to ideal polyhedron slots.

    Orthogonal Procrustes finds ``donors[order[slot]]``. Reflection is allowed for the fit, then
    `seat_properly` restores the correct hand. The graph-canonical site classes and chelate links then choose
    one proper-rotation frame, so renumbering atoms cannot change the retained constraints or slot notes.
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([np.zeros(3) if d == VACANT else pos[d] - pos[metal] for d in donors], float)
    dd /= np.where((norm := np.linalg.norm(dd, axis=1, keepdims=True)) > 0, norm, 1.0)
    v_ideal = np.array(dirs_ref, float)
    order = seat_properly(dd, dirs_ref, _seat_by_alignment(dd, v_ideal))
    seated = [donors[index] for index in order]
    classes = site_classes(mol, seated, haptic, coordination) if classes is None else classes
    keys = [None if donor == VACANT else classes[donor] for donor in seated]
    frame = canonical_slots(dirs_ref, keys, chelate_links(mol, seated, haptic))
    canonical = [0] * len(order)
    for source, target in enumerate(frame):
        canonical[target] = order[source]
    return canonical


def _forbidden_vertex_pairs(dirs, narrow, linked):
    """Return the wide and non-edge vertex-pair sets that `narrow` and `linked` each forbid a donor pair on.

    Used by `_seat_pruned_orderings` (early-reject generator); also backs the sweep-then-filter oracle
    `test_metal_slots.py`'s equivalence test keeps as `_drop_forbidden_orderings`.
    """
    pairs = [(i, j) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
    wide = [(i, j) for i, j in pairs if _vertex_angle(dirs[i], dirs[j]) >= CHELATE_SPAN_ANGLE] if narrow else ()
    edges = hull_edges(tuple(map(tuple, dirs))) if linked else frozenset()
    off_edge = [(i, j) for i, j in pairs if frozenset((i, j)) not in edges] if linked else ()
    return wide, off_edge


def _seat_pruned_orderings(dirs, rotations, narrow, linked):
    """Backtrack donor-to-vertex placements, rejecting a branch once a forbidden pair is fully seated.

    Replaces generating the full `isomer_permutations` sweep and filtering it after the fact
    (`_drop_forbidden_orderings`) for a tethered or haptic enumeration with non-empty `narrow` or `linked`.
    Both rules are a conjunction of independent per-vertex-pair predicates (`_forbidden_vertex_pairs`), so
    testing a pair the moment its later vertex is placed is sound (a violated pair can never be un-violated by
    completing the rest) and complete (every pair is tested exactly once, at its later vertex). Placing vertex
    0, 1, ..., n-1 with donors tried in increasing index order reproduces `itertools.permutations`'
    lexicographic order with the dead subtrees skipped, so the surviving sequence, filtered by the same
    proper-rotation-orbit representative check `isomer_permutations` applies at the leaf, is identical to
    `_drop_forbidden_orderings(isomer_permutations(geometry), dirs, narrow, linked)`, element for element
    (see test_metal_slots.py's equivalence test). Lives beside the prune rules it enforces (metal_slots), not
    metal_polyhedron: the vertex geometry only supplies `dirs`, `_forbidden_vertex_pairs` already lives here.
    """
    n = len(dirs)
    wide, off_edge = _forbidden_vertex_pairs(dirs, narrow, linked)
    checks = [[] for _ in range(n)]  # checks[vertex] = [(earlier_vertex, forbidden_donor_pairs), ...]
    for i, j in wide:
        checks[j].append((i, narrow))
    for i, j in off_edge:
        checks[j].append((i, linked))

    order = [-1] * n
    used = [False] * n

    def backtrack(vertex):
        if vertex == n:
            seated = tuple(order)
            if seated == min(tuple(order[q[k]] for k in range(n)) for q in rotations):
                yield seated
            return
        for donor in range(n):
            if used[donor] or any(frozenset((order[i], donor)) in forbidden for i, forbidden in checks[vertex]):
                continue
            order[vertex], used[donor] = donor, True
            yield from backtrack(vertex + 1)
            used[donor] = False
        order[vertex] = -1

    yield from backtrack(0)


def distinct_vertex_orderings(  # noqa: C901 - build then dedupe the candidate pool in one linear pass
    mol,
    donors,
    geometry,
    perms=None,
    haptic=None,
    *,
    classes=None,
    distances=None,
    retained=None,
    max_orbits=None,
    narrow=frozenset(),
    linked=frozenset(),
):
    """Enumerate distinct coordination isomers under exact graph and polyhedron symmetries.

    `perms` overrides the candidate vertex orderings (default ``isomer_permutations(geometry)``); a ``fix=``
    enumeration passes the subset that keeps each frozen donor pinned to its input vertex. A coordinate-derived
    ordering is retained first; explicit ``fix=`` permutations remain authoritative. `narrow` prunes same-ligand
    donor-position pairs that cannot span a wide bite (`metal_enumeration._narrow_span_pairs`); `linked` prunes
    same-ligand donor-position pairs held to a polyhedron hull edge (`_chelate_edge_links`). Both are only
    consulted for the streamed tethered pool, where the cap would otherwise raise on the unpruned count.

    Canonical vertex classes and same-ligand path lengths distinguish candidates under proper rotations.
    No graph heuristic is a proof of conformational reachability, so feasibility remains the embedder's job.
    `narrow` and `linked` each drop a streamed tethered order placing a constrained same-ligand donor pair on
    a vertex pair its rule forbids (see `_forbidden_vertex_pairs`), before the cap below can raise on an
    orbit no completion of it could ever satisfy anyway.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return perms
    frag = _frag_map(mol)  # same ligand = same fragment
    limit = _MAX_EXHAUSTIVE_ORBITS if max_orbits is None else int(max_orbits)
    tethered = _has_tether(donors, frag, haptic)
    classes = site_classes(mol, donors, haptic) if classes is None else classes
    # Separate equivalent monodentates have one arrangement in every geometry, so skip the factorial pool.
    if (
        perms is None
        and VACANT not in donors
        and not haptic
        and len(set(classes.values())) == 1
        and len({frag[donor] for donor in donors}) == len(donors)
    ):
        return [tuple(range(len(donors)))]
    rotations = point_group(tuple(map(tuple, dirs)))[0]
    retained = tuple(retained) if retained is not None and perms is None else None
    if perms is None and not tethered and VACANT not in donors and len(set(classes.values())) == len(donors):
        _check_assignment_cap(geometry, [1] * len(donors), rotations, limit)
        perms = isomer_permutations(geometry)
    elif perms is None and not tethered and VACANT not in donors and not haptic:
        # Enumerate constitutional class strings, not labelled permutations. This turns A9B and A8B2 at CN10
        # into 10 and 45 candidates before rotations instead of 10!, while preserving one representative atom
        # assignment for each string.
        groups = {}
        for index, donor in enumerate(donors):
            groups.setdefault(classes[donor], []).append(index)
        labels, counts = list(groups), {label: len(indices) for label, indices in groups.items()}
        _check_assignment_cap(geometry, list(counts.values()), rotations, limit)

        def class_orders(prefix=()):
            if len(prefix) == len(donors):
                used = dict.fromkeys(labels, 0)
                order = []
                for label in prefix:
                    order.append(groups[label][used[label]])
                    used[label] += 1
                yield tuple(order)
                return
            for label in labels:
                if counts[label]:
                    counts[label] -= 1
                    yield from class_orders((*prefix, label))
                    counts[label] += 1

        perms = itertools.chain((tuple(range(len(donors))),), class_orders())
    # An explicit `perms` (a `fix=` pool, or `observed_only`'s single retained order) is authoritative:
    # the caller already chose it, so neither rule's forbidden-vertex-pair screen applies to it.
    elif perms is None:
        perms = (
            isomer_permutations(geometry)
            if not narrow and not linked
            else _seat_pruned_orderings(dirs, rotations, narrow, linked)
        )
    if retained is not None:
        perms = itertools.chain((retained,), (order for order in perms if order != retained))
    dmat = (_ligand_distance_matrix(mol) if distances is None else distances) if tethered else None
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))] if tethered else ()

    def donor_class(d):
        return ("vacant",) if d == VACANT else ("donor", classes[d])

    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        links = chelate_links(mol, od, haptic, dmat) if tethered else {}
        sig = min(
            (
                tuple(donor_class(od[q[v]]) for v in range(len(dirs))),
                tuple(links.get(frozenset((q[p], q[r])), -1) for p, r in pairs),
            )
            for q in rotations
        )
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if len(out) > limit:
                raise _assignment_cap_error(geometry, limit)
    return out
