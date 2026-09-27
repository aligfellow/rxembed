"""Enumerate distinct donor assignments to metal-polyhedron slots."""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .metal_core import VACANT, frag_map, ligand_distance_matrix, vertex_atom
from .metal_polyhedron import (
    MAX_EXHAUSTIVE_ORBITS,
    POLYHEDRA,
    canonical_slots,
    hull_edges,
    isomer_permutations,
    point_group,
    seat_by_alignment,
    seat_properly,
    vertex_angle,
    vertex_dirs,
)
from .metal_stereo import chelate_links, site_classes

TRANS_ANGLE = 150  # same-element donor pairs beyond this are trans
SPAN_TOL = 0.1  # A numerical slack when a graph-derived bite window is conditioned on RDKit bounds
# Model bite ranges by chelate ring size, shared by DG/UFF and enumeration compatibility checks.
_CHELATE_BITE = {4: (58.0, 81.0), 5: (70.0, 91.0), 6: (74.0, 104.0)}


def assignment_cap_error(geometry, limit=None):
    """Return the resource-limit error shared by raw and screened slot pools."""
    limit = MAX_EXHAUSTIVE_ORBITS if limit is None else int(limit)
    return ValueError(
        f"metal[{geometry}]: more than {limit:,} distinct constitutional slot "
        "assignments; exact enumeration is required. Use rx.embed(mol) without metal= to retain "
        "coordinate input, rx.metal(mol, observed_only=True) for only its measured arrangement, "
        "or rx.metal(rx.cxsmiles(mol)) to request its stated arrangement"
    )


def _check_assignment_cap(geometry, counts, rotations, limit=None):
    """Reject a provably oversized constitutional orbit pool before generating its permutations."""
    limit = MAX_EXHAUSTIVE_ORBITS if limit is None else int(limit)
    possible = math.factorial(sum(counts))
    for count in counts:
        possible //= math.factorial(count)
    if (possible + len(rotations) - 1) // len(rotations) > limit:
        raise assignment_cap_error(geometry, limit)


def chelate_bite_window(mol, a, b, donors=()):
    """Return the ring-size census bite window, or ``None`` outside a 4-6 membered donor-free backbone."""
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


def chelate_edge_links(mol, padded, haptic=None, distances=None):
    """Return same-ligand donor-position pairs the edge rule holds to a polyhedron hull edge.

    A measured claim, not a proof: a pair on a 4-6 membered chelate backbone (`chelate_bite_window`) or a
    directly bonded pair (a tighter 3-membered ring) sits on a hull edge in all but 3 of 3,373 tmQMg
    reference centres. Haptic face atoms are never endpoints and block backbone paths. A pair with a third
    donor on a shortest graph path between them is dropped; the shorter links through that donor still hold.
    `chelate_bite_window` already rejects a pair whose only short route runs through a donor, so this
    exclusion acts only when an equally short donor-free route also exists.
    """
    haptic = haptic or {}
    frag = frag_map(mol)
    real_slots = [i for i, donor in enumerate(padded) if donor != VACANT and donor not in haptic]
    face_atoms = {atom for face in haptic.values() for atom in face}
    blockers = {padded[i] for i in real_slots} | face_atoms
    distances = ligand_distance_matrix(mol) if distances is None else distances
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
    atoms = [vertex_atom(haptic, donor) for donor in donors if donor != VACANT]
    return len(atoms) != len({frag[atom] for atom in atoms})


def _octahedral_triad(mol, od, haptic=None):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else ``None``.

    Either a tridentate chelate (exactly 3 donors of one ligand fragment) or exactly 3 monodentate donors of
    one element (an MA3B3 set); ``None`` otherwise, and then cis/trans is used. The exactly-3 and monodentate
    conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate (en2, en3)
    have no mer/fac, so neither must be forced into a triad.
    """
    real = [(p, vertex_atom(haptic, od[p])) for p in range(len(od)) if od[p] != VACANT]
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
    """Build the isomer label of vertex-ordered donors `od` from the ideal polyhedron; no conformer needed.

    Vacant vertices are ignored. Octahedral with a donor triad is mer/fac (one trans pair in the triad means
    mer, none means fac); otherwise cis/trans, judged on the minority same-element donor pair, which is the
    set whose placement defines the isomerism: the 2 Cl of an MA4B2, not the 4 A that always have a trans
    pair.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None or not POLYHEDRA[geometry].geometric_isomerism:
        return ""  # no polyhedron, or no cis/trans distinction for this geometry
    if geometry == "octahedral":
        tri = _octahedral_triad(mol, od, haptic)
        if tri is not None:
            trans = sum(
                1 for i in range(3) for j in range(i + 1, 3) if vertex_angle(dirs[tri[i]], dirs[tri[j]]) > TRANS_ANGLE
            )
            return "fac" if trans == 0 else "mer"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] != VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(vertex_atom(haptic, od[p])).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) == 2}  # noqa: PLR2004  a cis/trans pair
    if not pairs:
        return ""  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    ps = pairs[e]
    trans = any(
        vertex_angle(dirs[ps[i]], dirs[ps[j]]) >= TRANS_ANGLE for i in range(len(ps)) for j in range(i + 1, len(ps))
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
            angle = vertex_angle(positions[donors[a]] - positions[metal], positions[donors[b]] - positions[metal])
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
    order = seat_properly(dd, dirs_ref, seat_by_alignment(dd, v_ideal))
    seated = [donors[index] for index in order]
    classes = site_classes(mol, seated, haptic, coordination) if classes is None else classes
    keys = [None if donor == VACANT else classes[donor] for donor in seated]
    frame = canonical_slots(dirs_ref, keys, chelate_links(mol, seated, haptic))
    canonical = [0] * len(order)
    for source, target in enumerate(frame):
        canonical[target] = order[source]
    return canonical


def _seat_pruned_orderings(dirs, rotations, linked):
    """Backtrack donor-to-vertex placements, rejecting a branch once a linked pair sits off a hull edge.

    The edge rule is a conjunction of independent vertex-pair predicates, so testing a pair as soon as its
    later vertex is placed is sound (a violated pair stays violated) and complete (every pair is tested once).
    Placing vertices in order 0, 1, ..., n-1 reproduces `itertools.permutations`' lexicographic order with the
    dead subtrees skipped, and the leaf applies the same proper-rotation-orbit check as the full sweep.
    """
    n = len(dirs)
    edges = hull_edges(tuple(map(tuple, dirs)))
    off_edge = [[i for i in range(j) if frozenset((i, j)) not in edges] for j in range(n)]  # earlier vertices
    order = [-1] * n
    used = [False] * n

    def backtrack(vertex):
        if vertex == n:
            seated = tuple(order)
            if seated == min(tuple(order[q[k]] for k in range(n)) for q in rotations):
                yield seated
            return
        for donor in range(n):
            if used[donor] or any(frozenset((order[i], donor)) in linked for i in off_edge[vertex]):
                continue
            order[vertex], used[donor] = donor, True
            yield from backtrack(vertex + 1)
            used[donor] = False
        order[vertex] = -1

    yield from backtrack(0)


def _class_orders(donors, classes):
    """Yield one donor order per distinct string of site classes, in first-seen class order.

    Enumerating class strings instead of labelled permutations turns A9B and A8B2 at CN10 into 10 and 45
    candidates before rotations instead of 10!, keeping one representative atom assignment per string.
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

    `donors` is VACANT-padded to the polyhedron size, with each haptic face as its centroid key in `haptic`.
    `classes` and `distances` default to `site_classes` and `ligand_distance_matrix` of `mol`. `linked` holds
    same-ligand donor-position pairs to a polyhedron hull edge (`chelate_edge_links`).
    """

    mol: Chem.Mol
    donors: tuple | list
    geometry: str
    haptic: dict | None = None
    classes: dict | None = None
    distances: np.ndarray | None = None
    linked: frozenset = frozenset()


def distinct_vertex_orderings(problem, perms=None, *, retained=None, max_orbits=None):
    """Enumerate distinct coordination isomers under exact graph and polyhedron symmetries.

    `perms` overrides the candidate vertex orderings (default ``isomer_permutations(geometry)``); a ``fix=``
    enumeration passes the subset that keeps each frozen donor pinned to its input vertex, and a
    coordinate-derived ordering is retained first. The edge rule (`problem.linked`) prunes the streamed
    tethered pool only, dropping an order it forbids before the cap below can raise on an orbit no completion
    of it could ever satisfy.

    Canonical vertex classes and same-ligand path lengths distinguish candidates under proper rotations. No
    graph heuristic proves conformational reachability, so feasibility remains the embedder's job.
    """
    mol, donors, geometry, haptic = problem.mol, problem.donors, problem.geometry, problem.haptic
    dirs = vertex_dirs(geometry)
    frag = frag_map(mol)  # same ligand = same fragment
    limit = MAX_EXHAUSTIVE_ORBITS if max_orbits is None else int(max_orbits)
    tethered = has_tether(donors, frag, haptic)
    classes = site_classes(mol, donors, haptic) if problem.classes is None else problem.classes
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
        _check_assignment_cap(geometry, list(Counter(classes[d] for d in donors).values()), rotations, limit)
        perms = itertools.chain((tuple(range(len(donors))),), _class_orders(donors, classes))
    # An explicit `perms` (a `fix=` pool, or `observed_only`'s single retained order) is authoritative:
    # the caller already chose it, so the edge rule does not apply to it.
    elif perms is None:
        perms = (
            _seat_pruned_orderings(dirs, rotations, problem.linked) if problem.linked else isomer_permutations(geometry)
        )
    if retained is not None:
        perms = itertools.chain((retained,), (order for order in perms if order != retained))
    dmat = (ligand_distance_matrix(mol) if problem.distances is None else problem.distances) if tethered else None
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))] if tethered else ()

    site = {VACANT: ("vacant",)} | {d: ("donor", classes[d]) for d in donors if d != VACANT}
    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        links = chelate_links(mol, od, haptic, dmat) if tethered else {}
        sig = min(
            (
                tuple(site[od[q[v]]] for v in range(len(dirs))),
                tuple(links.get(frozenset((q[p], q[r])), -1) for p, r in pairs),
            )
            for q in rotations
        )
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if len(out) > limit:
                raise assignment_cap_error(geometry, limit)
    return out
