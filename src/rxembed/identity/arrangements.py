"""Enumerate the coordination and ligand stereoisomers of a metal graph as `Species`, screened against ligand reach."""

from __future__ import annotations

import functools
import itertools
import math
from collections import defaultdict
from dataclasses import replace
from typing import NamedTuple

import numpy as np
from rdkit import Chem, DistanceGeometry

from rxembed.geometry.bounds import coordination_reach
from rxembed.geometry.mechanisms import law_of_cosines
from rxembed.graph import frag_map, ligand_graph, metal_indices, reject_metal_bonds, torsion_path
from rxembed.identity.smiles_io import (
    SpecifiedArrangement,
    distinct,
    refuse_unordered_metals,
    specified_arrangement,
    specified_centres,
)
from rxembed.identity.species import (
    Species,
    canonical_metals,
    centre_name,
    centre_state,
    in_canonical_order,
    observed_centre,
    observed_haptic_windings,
    retained_geometry,
)
from rxembed.identity.stereoisomers import ligand_variants, unassigned_centres
from rxembed.metal.centres import (
    chemical_graph,
    collapse_haptic,
    logger,
    materialized_states,
    seated_centre,
    state_with_winding,
    symmetric_metals,
)
from rxembed.metal.polyhedra import (
    POLYHEDRA,
    VACANT,
    describe,
    geometries_for_cn,
    mirror_seating,
)
from rxembed.metal.shape import shape_gap
from rxembed.metal.slots import (
    MAX_ARRANGEMENTS,
    SPAN_TOL,
    SeatingProblem,
    assignment_cap_error,
    distinct_vertex_orderings,
    has_tether,
    input_ordering,
    pinned_vertices,
    short_chelate_pairs,
)
from rxembed.metal.sphere import LigandFacts, compile_constraints
from rxembed.metal.stereo import MIRROR, chelate_links, chirality_of, donor_classes, face_orientations, site_classes

_SCREEN_SITES = 8  # Keep the finite raw pool bounded before applying whole-network reach screening.


def _select_geometries(base, m, donors, haptic, geometry):
    """Return the polyhedra to enumerate: the requested names, else the default for the donor count."""
    n = len(donors)
    if geometry is not None:
        logger.debug("geometry: requested %s", ", ".join(map(describe, geometry)))
        return list(geometry)
    selected = retained_geometry(base, m, donors, haptic)
    if selected is None:
        raise ValueError(
            f"no default geometry for {n} donors; pass geometry= a name or list "
            f"(options for {n} donors: {[p.name for p in geometries_for_cn(n)]})"
        )
    return [selected]


def _pinned_orders(frozen_v, free_v, free_di, padded):
    """Yield every vertex order that keeps each frozen donor at its vertex and seats the free donors.

    Vacancies are interchangeable, so only real free donors are permuted.
    """
    real = [di for di in free_di if padded[di] is not VACANT]
    vacant = [di for di in free_di if padded[di] is VACANT]
    for seats in itertools.permutations(free_v, len(real)):
        order = [None] * len(padded)
        for vertex, donor_index in (*frozen_v.items(), *zip(seats, real, strict=True)):
            order[vertex] = donor_index
        empty = iter(vacant)
        yield [next(empty) if donor_index is None else donor_index for donor_index in order]


def _frozen_permutations(mol, core, metal, padded, geom):
    """Return every free-donor vertex order with each donor `core` pins at its observed vertex, else None.

    It bypasses the symmetry-reduced orderings, whose representatives may seat an equivalent free donor where the frame
    pins another.
    """
    if not (frozen_v := pinned_vertices(mol, core, metal, padded, geom)):
        return None
    sites = len(padded)
    free_v = [v for v in range(sites) if v not in frozen_v]
    free_di = [di for di in range(sites) if di not in frozen_v.values()]
    count = math.perm(len(free_v), sum(padded[di] is not VACANT for di in free_di))
    if count > MAX_ARRANGEMENTS:
        raise ValueError(
            f"metal[{geom}]: fix= leaves {count:,} free-site arrangements (limit {MAX_ARRANGEMENTS:,}); pin more donors"
        )
    logger.info(
        "metal[%s]: %d frozen donors; deduplicating %d free-site arrangements",
        geom,
        len(frozen_v),
        count,
    )
    return _pinned_orders(frozen_v, free_v, free_di, padded)


def _retained_seating(problem, metal, facts):
    """Return the vertex order the conformer seats its donors in, or None where it proves nothing.

    The seating must take `problem.geometry`. Under a screen (`facts`), every same-ligand donor span must also fit the
    native reach the screen reads: a hand-placed conformer can put donors where the ligand cannot reach, and only a
    reachable seating may skip the reach screen.
    """
    base, padded, geom, haptic = problem.mol, problem.donors, problem.geometry, problem.haptic
    if not base.GetNumConformers():
        return None
    retained = input_ordering(base, metal, padded, geom, haptic, classes=problem.classes)
    if retained is None or not shape_gap(base, metal, [padded[k] for k in retained], haptic, geom).matches:
        return None
    if facts is None:
        return retained
    at, native_reach = facts.mol.GetConformer().GetPositions(), facts.native_reach
    real = sorted(d for d in padded if d is not VACANT and d not in haptic)
    frag = frag_map(base)
    witnessed = not any(
        frag[a] == frag[b] and np.linalg.norm(at[a] - at[b]) > float(native_reach[a, b]) + SPAN_TOL
        for a, b in itertools.combinations(real, 2)
    )
    return retained if witnessed else None


def _screened(request, facts, geom):
    """Return the `LigandFacts` a tethered sphere is screened with, or None when it is not screened.

    Pins on a sphere atom skip the screen with an INFO log; a native reach RDKit cannot give skips it with a warning.
    """
    if not request.screen:
        return None
    if request.pins.graft.keys() & {atom for bond in facts.donor_bonds for atom in bond}:
        logger.info(
            "metal[%s]: fix= pins the sphere, so neither the reach screen nor the trans-pair rule applies", geom
        )
        return None
    if facts.native_reach is None:
        logger.warning("metal[%s]: native ligand reach failed; no arrangement is screened", geom)
        return None
    return facts


_REACH_CONFLICT = "compiled coordination distances conflict with native ligand reach"
_CROSS_EPS = 1e-9  # Å, floating-point slack shared by every closed-bound comparison against a reach matrix
_HAPTIC_MATCHING_CAP = 7  # factorial matching remains tiny for ordinary pi faces; larger faces use the safe mean bound.
_EUCLIDEAN_SUBSET_BUDGET = 128  # Keep local route certificates bounded; the full route remains authoritative.
_PROJECTOR_EPS = 1e-10  # Ignore the numerically null centred constant eigenspace.
_CERTIFICATE_STEPS = 32  # accelerated-gradient step budget for the refinement search, not a feasibility threshold


def _reach_conflict(species, centre, links, facts, pins):
    """Return why `centre` of `species` compiles to constraints beyond native ligand reach, else None.

    This checks the compiled model, not chemistry, and passing proves nothing. Centroid rows are not compiled, so a
    tethered face on a trans pair is checked against its centroid reach (`haptic_span_conflict`).
    """
    compiled = compile_constraints(species, pins, facts=facts)
    if failure := compiled_reach_conflict(species, centre, compiled, facts):
        return failure
    poly = POLYHEDRA[centre.geometry]
    vertices, haptic, _winding, _donors = species.sites[centre.atom]
    for pair in links:
        i, j = sorted(pair)
        if {vertices[i], vertices[j]} & haptic.keys() and (i, j) in poly.trans_pairs:
            angle, faces = poly.pair_angles[pair], (vertices[i], vertices[j])
            if failure := haptic_span_conflict(species, centre, facts.native_reach, faces, angle, compiled):
                return failure
    return None


def haptic_span_conflict(species, centre, reach, pair, angle, compiled):
    """Return why two tethered sites, one a face, cannot span their distance at `angle`, else None.

    A model prior: it reads the ideal vertex angle and the centroid reach, since member rays need not share the angle
    between centroids.
    """
    (left, right), metal, haptic = pair, centre.atom, species.sites[centre.atom].haptic
    left_face, right_face = haptic.get(left, (left,)), haptic.get(right, (right,))
    radii = tuple(compiled.distances[tuple(sorted((metal, vertex)))][0] for vertex in (left, right))
    available = _centroid_reach(reach, left_face, right_face)
    needed = law_of_cosines(radii[0], radii[1], angle)
    if math.isfinite(available) and needed > available + SPAN_TOL:
        return f"haptic faces {left}/{right} need >= {needed:.3f} A; centroid reach <= {available:.3f} A"
    return None


def _centroid_reach(reach, left, right):
    """Return an upper bound on the distance between two donor centroids.

    For equal faces the centroid difference is the mean of matched displacements, so by the triangle inequality the
    smallest matching bounds it. Unequal or large faces use the all-pairs mean.
    """
    values = [[float(reach[min(a, b), max(a, b)]) for b in right] for a in left]
    if not all(math.isfinite(value) for row in values for value in row):
        return math.inf
    if len(left) == len(right) and len(left) <= _HAPTIC_MATCHING_CAP:
        return min(
            sum(row[index] for row, index in zip(values, order, strict=True))
            for order in itertools.permutations(range(len(right)))
        ) / len(left)
    return sum(map(sum, values)) / (len(left) * len(right))


def compiled_reach_conflict(species, centre, constraints, facts):
    """Return why `centre`'s compiled donor network cannot meet native reach, else None.

    The compiled rows meet native reach in `coordination_reach`; the network is then triangle-smoothed and each shortest
    donor-to-donor route gets the Euclidean certificate (`_certified_conflict`).
    """
    mol, metal, memo = facts.mol, centre.atom, facts.memo
    closed = coordination_reach(constraints, facts.native_reach)
    if not DistanceGeometry.DoTriangleSmoothing(closed, _CROSS_EPS):
        return _REACH_CONFLICT
    donors = frozenset(species.sites[metal].donors)
    if (route_key := ("route_atoms", donors)) not in memo:
        # Routes depend only on topology and the donor set, which every candidate of one screen shares.
        memo[route_key] = _route_certificate_paths(mol, donors, Chem.GetDistanceMatrix(mol, force=True))
    certificate_memo = memo.setdefault("certificate_memo", {})
    for path in memo[route_key]:
        # stream subsets per route, never stored: the route, then its 3- and 4-atom subsets while they fit the budget
        sizes = [n for n in (3, 4) if n < len(path)]
        subsets = [path]
        if sum(math.comb(len(path), n) for n in sizes) <= _EUCLIDEAN_SUBSET_BUDGET:
            subsets = itertools.chain(subsets, *(itertools.combinations(path, n) for n in sizes))
        for subset in subsets:
            atoms = sorted((metal, *subset))
            # Refine the whole route once; its smaller diagnostic subsets retain the cheap midpoint test.
            if (
                euclidean_conflict(closed[np.ix_(atoms, atoms)], refine=subset == path, memo=certificate_memo)
                is not None
            ):
                return f"compiled coordination distances have no Euclidean realization at atoms {atoms}"
    return None


def _route_certificate_paths(mol, donors, topology):
    """Return the atoms of each donor-pair route, deduplicated, without routes through a donor bond."""
    tested_paths = set()
    paths = []
    for left, right in itertools.combinations(sorted(donors), 2):
        separation = topology[left, right]
        if not 0 < separation < mol.GetNumAtoms():
            continue
        # use every equally short route: picking one depends on atom order, skipping ties loses the check
        path = tuple(map(int, np.flatnonzero(topology[left] + topology[right] == separation)))
        if path in tested_paths:
            continue
        tested_paths.add(path)
        # a bond between two donors (a bridge or multicentre bond) does not fit the two-centre model; the gates judge it
        if any(
            left in donors and right in donors and mol.GetBondBetweenAtoms(int(left), int(right)) is not None
            for left, right in itertools.combinations(path, 2)
        ):
            continue
        paths.append(path)
    return paths


@functools.lru_cache(maxsize=32)
def _centering_matrix(n):
    """Return the size-`n` double-centering projector."""
    return np.eye(n) - np.ones((n, n)) / n


def euclidean_conflict(matrix, *, refine=False, memo=None):
    """Return `_certified_conflict` of these bounds, cached in `memo` since every candidate asks again."""
    array = np.ascontiguousarray(matrix, dtype=float)
    key = (array.shape[0], array.tobytes(), bool(refine))
    if memo is None:
        return _certified_conflict(*key)
    if key not in memo:
        memo[key] = _certified_conflict(*key)
    return memo[key]


def _projected_witnesses(gram, lower, upper, n):
    """Yield the witness an accelerated projected-gradient search finds for a centred Gram matrix in the box."""
    centre = _centering_matrix(n)
    current, extrapolated, momentum = gram, gram.copy(), 1.0
    for _ in range(_CERTIFICATE_STEPS):
        squared = np.diag(extrapolated)[:, None] + np.diag(extrapolated) - 2 * extrapolated
        errors = squared - np.clip(squared, lower, upper)
        # half the squared violations have Lipschitz bound 2*n on centred Gram matrices; joint steps keep symmetry
        proposal = extrapolated - (np.diag(errors.sum(axis=1)) - errors) / (2 * n)
        proposal = centre @ ((proposal + proposal.T) / 2) @ centre
        eigenvalues, eigenvectors = np.linalg.eigh(proposal)
        updated = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - current)
        current, momentum = updated, next_momentum
    basis = (centre @ eigenvectors) * np.sqrt(np.maximum(-eigenvalues, 0.0))
    yield basis @ basis.T


def _certified_conflict(n, payload, refine):
    """Return a certified positive squared-distance margin, or None without a Euclidean contradiction.

    For PSD W with W*1 = 0, trace(W D²) = -2 trace(X.T W X) <= 0 for every Euclidean point set X, so a positive minimum
    of that expression over the squared-distance intervals rules out every dimension, not only 3D. Midpoint eigenspaces
    propose W, as whole projectors so repeated eigenvalues need no choice; the optional refinement proposes a stronger W
    when the midpoint misses coupled distances. Only the certified margin rejects a box, never a convergence failure,
    and no certificate does not prove feasibility. The matrix comes as `n` and raw bytes so the result can be cached.
    """
    matrix = np.frombuffer(payload, dtype=float).reshape(n, n)
    lower, upper = np.tril(matrix, -1), np.triu(matrix, 1)
    lower, upper = lower + lower.T, upper + upper.T
    if (
        n <= 1
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower < 0)
        or np.any(lower > upper)
    ):
        return None
    lower, upper = lower**2, upper**2
    centre = _centering_matrix(n)
    gram = -0.25 * centre @ (lower + upper) @ centre
    values, vectors = np.linalg.eigh(gram)
    groups = np.split(np.arange(n), np.flatnonzero(np.diff(values) > 1e-9 * max(1.0, *abs(values))) + 1)
    # the projector onto each negative eigenspace of the midpoint Gram matrix
    witnesses = (
        basis @ basis.T for basis in (centre @ vectors[:, group] for group in groups if min(values[group]) < 0)
    )
    if refine:
        witnesses = itertools.chain(witnesses, _projected_witnesses(gram, lower, upper, n))
    for candidate in witnesses:
        if np.trace(candidate) < _PROJECTOR_EPS:
            continue
        weights = candidate / np.trace(candidate)
        terms = weights * np.where(weights >= 0, lower, upper)
        margin = float(terms.sum())
        if margin > 1e-9 * max(1.0, float(np.abs(terms).sum())):
            return margin
    return None


def _told_apart(facts, context, metal, classes, haptic):
    """Return the site `classes` of `metal`, each site split off whose ligand binds another open centre.

    Donors equivalent at this centre stop being so once the other is seated: a bridge trans to the chloride at both
    metals (syn) and at one (anti) are two species.
    """
    frag = facts.fragments
    others = {centre.atom for centre in context if not centre.geometry} - {metal}
    joined = {frag[donor] for donor, other in facts.donor_bonds if other in others}
    return {
        site: (*key, site) if any(frag[atom] in joined for atom in haptic.get(site, (site,))) else key
        for site, key in classes.items()
    }


def _centres_for_geometry(request, facts, context, centre, geom, collapsed):
    """Return every distinct arrangement of `centre` in polyhedron `geom`, pinned donors held.

    `context` holds the other centres the reach screen compiles beside each candidate.
    """
    base, m, donors, haptic = collapsed.mol, centre.atom, collapsed.sites, collapsed.haptic
    vacant = POLYHEDRA[geom].cn - len(donors)
    if vacant < 0:
        raise ValueError(f"{geom} has {POLYHEDRA[geom].cn} coordination sites but the metal has {len(donors)} donors")
    padded = [*donors, *[VACANT] * vacant]  # leave empty vertices as a pocket
    if vacant:
        logger.info(
            "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)",
            geom,
            len(padded),
            len(donors),
            vacant,
        )
    perms = _frozen_permutations(base, request.pins.graft, m, padded, geom)
    tethered = has_tether(padded, frag_map(base), haptic)
    distances = Chem.GetDistanceMatrix(ligand_graph(base)) if tethered else None
    screen = _screened(request, facts, geom) if tethered else None
    # a short chelate pair stays off trans pairs (`slots.short_chelate_pairs`), gated as the reach screen is
    linked = short_chelate_pairs(base, padded, haptic, distances) if screen is not None else frozenset()
    if linked:
        logger.info(
            "metal[%s]: held %d short chelate pair(s) off trans pairs (screen=False lifts it)", geom, len(linked)
        )
    classes = site_classes(base, padded, haptic, facts.roles)
    problem = SeatingProblem(
        base, tuple(padded), geom, haptic, _told_apart(facts, context, m, classes, haptic), distances, linked
    )
    retained = _retained_seating(problem, m, screen)
    out, unreachable = [], 0
    raw_limit = math.factorial(len(padded)) if screen is not None and len(padded) <= _SCREEN_SITES else None
    # read once: a face's winding depends on the input geometry, not on the vertex it lands at
    winding = observed_haptic_windings(
        facts.mol, m, [donor for donor, metal in facts.donor_bonds if metal == m], haptic
    )
    for order in distinct_vertex_orderings(problem, perms, retained=retained, max_arrangements=raw_limit):
        od = [padded[k] for k in order]  # vertex -> donor atom, haptic centroid, or VACANT
        links = chelate_links(base, od, haptic, distances) if tethered else ()
        hand = chirality_of(base, geom, od, haptic, classes=classes, links=links)
        active = seated_centre(centre, geom, od, haptic, winding, hand)
        others = (other for other in context if other.atom != m)
        candidate = Species(mol=facts.graph, centres=(active, *others))
        if screen is not None and (retained is None or tuple(order) != tuple(retained)):
            failure = _reach_conflict(candidate, active, links, screen, request.pins.cons)
            if failure:
                logger.debug("metal[%s]: omit arrangement: %s", geom, failure)
                unreachable += 1
                continue
        out.append(active)
        if len(out) > MAX_ARRANGEMENTS:
            raise assignment_cap_error(geom)
    if unreachable:
        logger.info(
            "metal[%s]: omitted %d arrangements the ligands cannot reach (screen=False keeps them)", geom, unreachable
        )
    if not out and (unreachable or linked):
        # `linked` can prune every order before the screen counts one (a linear pocket is one trans pair)
        logger.warning("metal[%s]: no model-compatible arrangement; screen=False keeps ones that may not embed", geom)
    return out


def _winding_variants(facts, centre, faces):
    """Return the symmetry-distinct windings of one centre's unwound faces among `faces`."""
    vertices, haptic, stored, donors = materialized_states(facts.graph, (centre,))[centre.atom]
    faces = [dummy for dummy in haptic if dummy in faces and dummy not in stored]
    if not faces:
        return [centre]
    ranks = donor_classes(facts.mol, donors)
    choices = {dummy: tokens for dummy in faces if (tokens := face_orientations(facts.mol, haptic[dummy], ranks))}
    if not choices:
        return [centre]
    variants, seen = [], set()
    for tokens in itertools.product(*choices.values()):
        variant = state_with_winding(centre, vertices, stored | dict(zip(choices, tokens, strict=True)))
        if (state := centre_state(facts, variant)) not in seen:
            seen.add(state)
            variants.append(variant)
    return variants


class HapticModes(NamedTuple):
    """Hold the haptic modes: `side` for a face, `diene` for a bound diene's s-cis/s-trans class."""

    side: str
    diene: str


def _haptic_choices(facts, centres, modes):
    """Return `centres` with their face windings expanded under `modes`.

    A diene face takes the diene mode, any other the side mode. 'racemic' drops the observed winding and 'invert'
    mirrors it; both then take every symmetry-distinct winding.
    """
    if set(modes) <= {"free", "preserve"}:
        return centres
    out = []
    for centre in centres:
        vertices, haptic, stored, _donors = materialized_states(facts.graph, (centre,))[centre.atom]
        mode = {face: modes.diene if torsion_path(facts.mol, atoms) else modes.side for face, atoms in haptic.items()}
        winding = {  # racemic: discard the observed configuration first
            face: MIRROR[token] if mode[face] == "invert" else token
            for face, token in stored.items()
            if mode[face] != "racemic"
        }
        open_faces = {face for face in haptic if mode[face] not in ("free", "preserve")}
        out.extend(_winding_variants(facts, state_with_winding(centre, vertices, winding), open_faces))
    return out


def _reject_unenumerable(mol, request, selected):
    """Refuse a request on several metals that no enumeration answers, before any variant is drawn.

    Under ``center='all'``, `geometry=` cannot name one polyhedron for every metal, and three or more metals joined by
    bridges refuse, since notes cannot record how bridges join them. One `center` among three or more equivalent metals
    refuses, since its other arrangements have no canonical name; any other single `center` needs a conformer for the
    remaining metals. Notes on any metal, or ``observed=True``, skip these checks.
    """
    metals, center = metal_indices(mol), request.center
    if center == "all" and request.geometry is not None:
        raise ValueError("geometry= names one polyhedron, but every metal is enumerated here, each in its own; omit it")
    if (
        len(metals) <= 1
        or request.observed
        or selected
        or any(specified_arrangement(mol, m) is not None for m in metals)
    ):
        return
    if center != "all":
        if any(center in group for group in symmetric_metals(mol) if len(group) > 2):  # noqa: PLR2004  a pair is named
            raise ValueError(
                f"center={center} is one of three or more symmetry-equivalent metals with no canonical names; "
                "enumerate one alone"
            )
        if not mol.GetNumConformers():
            raise ValueError(
                f"enumerating one centre of a {len(metals)}-metal complex needs a geometry for the others; read "
                "one with rx.read_xyz"
            )
        return
    for fragment in Chem.GetMolFrags(mol):
        if len(joined := sorted(set(fragment) & set(metals))) > 2:  # noqa: PLR2004  a pair is seated jointly
            raise ValueError(
                f"metals {joined} are joined by bridges, which no name records for three or more; specify each "
                "centre's arrangement"
            )


def enumerate_isomers(mol, request, *, selected=()):
    """Return the `Species` of each distinct coordination and ligand stereoisomer of metal graph `mol`.

    `request` is read once on `mol`. With ``center='all'`` every centre is enumerated and each combination listed once;
    with one `center` the others keep what the conformer shows. A geometry input keeps its observed ligand and face
    stereo by default; a graph enumerates what is undefined.

    ``screen`` applies the chelate trans-pair rule and the native ligand-reach screen at model lengths; False keeps
    assignments beyond those limits, which may not embed, and pins on a sphere atom turn the screens off. Only
    arrangements that seat each pinned donor where it stands remain. ``observed=True`` returns only the observed
    arrangement, a resource limit for high-coordinate inputs, not a claim that others are impossible. `selected` carries
    known centres through a ligand-stereo request.
    """
    reject_metal_bonds(mol)
    _reject_unenumerable(mol, request, selected)
    out = []
    for variant in ligand_variants(mol, request.stereo):
        stereo = variant.stereo
        out.extend(replace(species, stereo=stereo) for species in _arrangements(variant.mol, request, selected))
    return out


def _arrangements(mol, request, selected):
    """Return every `Species` of one ligand variant: the product over its metals of each centre's choices.

    A variant with notes takes the centre each note gives; otherwise each metal's choices come from `_open_choices`, and
    a product over every metal is screened and deduplicated (`_combined`).
    """
    metals = metal_indices(mol)
    facts = LigandFacts(chemical_graph(mol))
    if selected:  # the selected centres, checked as parsed notes are
        notes = {
            centre.atom: SpecifiedArrangement(
                geometry=centre.geometry,
                slots={slot: site.atoms[0] for slot, site in enumerate(centre.vertices) if site is not VACANT},
                hand=centre.hand,
                windings={
                    slot: site.winding
                    for slot, site in enumerate(centre.vertices)
                    if site is not VACANT and site.winding
                },
            )
            for centre in selected
        }
    else:
        notes = {m: specified_arrangement(mol, m) for m in metals}
    from_notes = not request.observed and any(notes.values())
    order = metals if request.center == "all" else [request.center, *(m for m in metals if m != request.center)]
    if from_notes:
        choices = _specified_choices(mol, facts, order, request.geometry, notes)
        if request.stereo.mirror:  # each centre read from the input is seated as its mirror image
            choices = [[_mirrored(centre) for centre in centres] for centres in choices]
    else:
        choices = _open_choices(mol, facts, order, notes, request)
    modes = _haptic_modes(mol, request, notes)
    choices = [_haptic_choices(facts, centres, modes) for centres in choices]
    if not from_notes and request.center == "all":
        return _combined(facts, mol, choices, request)
    built = [Species(mol=facts.graph, centres=centres) for centres in itertools.product(*choices)]
    return [in_canonical_order(species, mol) for species in built] if from_notes and request.center == "all" else built


def _haptic_modes(mol, request, notes):
    """Return the `HapticModes` the faces of variant `mol` take under `request`.

    An observed diene class is kept where the request would re-enumerate it, and so is a noted one with a conformer.
    Without a conformer and without notes, a preserved face is enumerated.
    """
    stereo, has_conformer = request.stereo, bool(mol.GetNumConformers())
    table = stereo.modes(mol)
    modes = HapticModes(side=table["planar"], diene=table["diene"])
    all_noted = all(note is not None for note in notes.values())
    if modes.diene == "racemic" and (request.observed or (all_noted and has_conformer and stereo.locked is None)):
        modes = modes._replace(diene="preserve")
    if not has_conformer and (request.observed or not any(notes.values())):
        modes = modes._replace(**{kind: "racemic" for kind, mode in modes._asdict().items() if mode == "preserve"})
    return modes


def _open_choices(mol, facts, order, notes, request):
    """Return the centres each metal in `order` can take when `mol` has no notes.

    The requested metal takes every arrangement found, the others the one their conformer shows; ``center='all'``
    enumerates every metal in canonical order. Under `mirror`, centres read from the conformer are mirrored.
    """
    if request.center == "all":
        order = canonical_metals(mol, order)
    if request.stereo.mirror and request.observed is None and request.geometry is None and mol.GetNumConformers():
        enumerated = ()
    else:
        enumerated = order if request.center == "all" else order[:1]
    # notes survive here only under observed=True, which reads each noted metal in its noted polyhedron
    noted = {m: (notes[m].geometry,) for m in order if notes[m]}  # as `IsomerRequest.geometry` names polyhedra
    fixed = {m: observed_centre(facts, m, notes[m] and notes[m].geometry) for m in order if m not in enumerated}
    context = tuple(fixed.get(metal, facts.centre(metal)) for metal in metal_indices(mol))
    choices = [
        [fixed[m]] if m in fixed else _enumerated(request, facts, m, context, request.geometry or noted.get(m))
        for m in order
    ]
    if not request.stereo.mirror:
        return choices
    return [
        [_mirrored(centre) for centre in centres] if request.observed or m in fixed else centres
        for m, centres in zip(order, choices, strict=True)
    ]


def _combined(facts, mol, choices, request):
    """Return once each species the product of `choices` gives, every metal of `mol` enumerated.

    When two or more centres vary, more than `MAX_ARRANGEMENTS` combinations refuse before any is built, and three or
    more equivalent metals given different arrangements refuse as the writer does (`refuse_unordered_metals`).
    """
    varied = {centres[0].atom for centres in choices if len(centres) > 1}
    if len(varied) > 1 and (count := math.prod(map(len, choices))) > MAX_ARRANGEMENTS:
        raise ValueError(
            f"the centres combine into {count:,} arrangements ({' x '.join(str(len(c)) for c in choices)}; "
            f"limit {MAX_ARRANGEMENTS:,}); specify each centre's arrangement"
        )
    built = [Species(mol=facts.graph, centres=centres) for centres in itertools.product(*choices)]
    symmetric = symmetric_metals(mol)
    for species in built:
        refuse_unordered_metals(species, symmetric, facts)
    return distinct(_ring_screened(built, request, facts))


def _ring_screened(built, request, facts):
    """Return the species of `built` whose rings through shared donors the compiled model can close.

    A donor two centres share closes a ring neither centre's screen sees: two chloride bridges cis at both square-planar
    metals close one, trans at either does not. The ring's bounds are smoothed and certified (`euclidean_conflict`), a
    proof against the model, not chemistry. ``screen=False``, pins on a sphere atom and ``observed=True`` keep all.
    """
    sharing = defaultdict(set)
    for donor, metal in facts.donor_bonds:
        sharing[donor].add(metal)
    rings = defaultdict(list)
    for donor, metals in sharing.items():
        if len(metals) == 2:  # noqa: PLR2004  a donor that two metals share
            rings[tuple(sorted(metals))].append(donor)
    rings = {pair: donors for pair, donors in rings.items() if len(donors) > 1}
    pinned = request.pins.graft.keys() & {atom for bond in facts.donor_bonds for atom in bond}
    if not rings or not request.screen or request.observed or pinned:
        return built
    if facts.native_reach is None:
        logger.warning("metal: native ligand reach failed; no bridge is screened")
        return built
    kept = [species for species in built if not _ring_conflict(species, rings, facts, request.pins.cons)]
    if len(kept) < len(built):
        log = logger.info if kept else logger.warning
        log(
            "metal: omitted %d arrangements whose bridges cannot close (screen=False keeps them)",
            len(built) - len(kept),
        )
    return kept


def _ring_conflict(species, rings, facts, pins):
    """Return the first metal pair whose ring of shared donors `species` cannot close under `pins`, else None."""
    compiled = compile_constraints(species, pins, facts=facts)
    closed = coordination_reach(compiled, facts.native_reach)
    memo = facts.memo.setdefault("certificate_memo", {})
    for pair, donors in rings.items():
        atoms = sorted((*pair, *donors))
        box = closed[np.ix_(atoms, atoms)]
        if not DistanceGeometry.DoTriangleSmoothing(box, _CROSS_EPS) or euclidean_conflict(box, refine=True, memo=memo):
            return pair
    return None


def _mirrored(centre):
    """Return `centre` seated as its mirror image, with the other hand."""
    if not centre.hand:
        return centre
    vertices = mirror_seating(POLYHEDRA[centre.geometry].vertex_dirs, centre.vertices)
    return centre._replace(vertices=tuple(vertices), hand=MIRROR[centre.hand])


def _specified_choices(mol, facts, order, geometry, notes):
    """Return the one centre the notes give each metal in `order`; `geometry=` may only repeat the first's.

    Undefined ligand stereo is warned about, since seeds would mix both hands.
    """
    if undefined := unassigned_centres(mol, exclude=metal_indices(mol)):
        logger.warning(
            "Species: ligand stereo element(s) %s undefined; seeds mix both hands (use rx.isomers)",
            [e.atoms[0] if e.kind == "point" else "=".join(map(str, e.atoms)) for e in undefined],
        )
    read = {centre.atom: centre for centre in specified_centres(notes, facts)}
    name = read[order[0]].geometry
    if geometry not in (None, (name,)):
        raise ValueError(
            f"this input already specifies a {name} arrangement, so geometry={list(geometry)} has nothing to do; "
            "drop it"
        )
    logger.info("using the noted %s arrangement", describe(name))
    return [[read[m]] for m in order]


def _enumerated(request, facts, metal, context, geometry):
    """Return every centre `metal` can take in `geometry`, else in its default polyhedron.

    `context` holds one centre per metal for the reach screen. With ``observed=True`` it is the one centre the conformer
    shows in each polyhedron.
    """
    collapsed = collapse_haptic(facts.mol, [d for d, m in facts.donor_bonds if m == metal])
    base, donors, haptic = collapsed
    centre = next(c for c in context if c.atom == metal)
    if not donors and geometry is None:
        remedy = "use the Mol directly or pass geometry= to define a vacant coordination pocket"
        raise ValueError(
            f"{centre_name(centre)} has no donor bonds, so there is no coordination isomer to enumerate; "
            + (remedy if len(context) == 1 else "enumerate the other metals without it")
        )
    geoms = _select_geometries(base, metal, donors, haptic, geometry)
    if request.observed:
        return [observed_centre(facts, metal, geom) for geom in geoms]
    return [
        arranged
        for geom in geoms
        for arranged in _centres_for_geometry(request, facts, context, centre, geom, collapsed)
    ]
