"""Screen enumerated coordination candidates against the ligands' native reach before embedding.

`unreachable_span` checks each generated candidate for a native-reach conflict.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
from rdkit import Chem, DistanceGeometry

from .bounds import coordination_reach, coordination_reach_base
from .constraints import ML_WINDOW_TOL, Constraints
from .mechanisms import law_of_cosines, triangle_distances
from .metal_constraints import (
    CoordinationSphere,
    bounded_bites,
    compile_constraints,
    donor_distance_window,
    resolve_lengths,
    seated_bites,
)
from .metal_core import VACANT
from .metal_distance import delocalised_charges
from .metal_donor_orient import FOLD_WINDOW, donation_axis, stripped_hybridisation
from .metal_perceive import FIT_FLOOR
from .metal_polyhedron import CHELATE_SPAN_ANGLE, POLYHEDRA, vertex_angle
from .metal_slots import SPAN_TOL, TRANS_ANGLE, chelate_bite_window
from .metal_stereo import equivalent_site_assignments

_CHELATE_PATH_MIN = 3
_CHELATE_SPAN_MIN = 120.0
_HAPTIC_MATCHING_CAP = 7  # factorial matching remains tiny for ordinary pi faces; larger faces use the safe mean bound.
_FIT_ROUNDOFF = 1e-10  # Numerical comparison on unit rays, not an angular or chemical tolerance.
_EUCLIDEAN_SUBSET_BUDGET = 128  # Keep local route certificates bounded; the full route remains authoritative.
_ROUTE_ATOMS_KEY = "route_atoms"  # context cache: _route_certificate_paths is candidate-independent, see there.
_PROJECTOR_EPS = 1e-10  # Ignore the numerically null centered constant eigenspace.
_CERTIFICATE_STEPS = 32  # accelerated-gradient step budget for the refinement search, not a feasibility threshold
_CROSS_EPS = 1e-9  # Å, floating-point slack shared by every closed-bound comparison against a reach matrix
_COMPILED_CONSTRAINTS_KEY = "_span_compiled"  # context scratch slot: this candidate's compiled constraints,
# written by _compiled_span_failure and consumed once by unreachable_span; never read stale (see both).
_CERTIFICATE_MEMO_KEY = "_certificate_memo"  # context scratch slot: one screen's _certified_conflict results


def _radial_distance_windows(iso, atoms, positions, base_distances=None, context=None):
    """Return each donor atom's model M-donor distance window and hybridisation.

    `positions` is resolved by the caller: `unreachable_span` measures from `iso.length_mol`, which need
    not be `iso.graph` itself. `context` reuses a caller's already-compiled hybridisation/charges
    (`compile_context`) instead of recomputing them for every candidate.
    """
    mol, metal, real_z, donors = iso.graph, iso.metal, iso.real_z, set(iso.donors)
    if context is None:
        hyb = stripped_hybridisation(mol)
        charges = delocalised_charges(mol) if positions is None else None
    else:
        hyb = context["hybridisation"]
        charges = context["charges"] if positions is None else None
    radial = Constraints()
    base_distances = base_distances or {}
    for donor in atoms:
        key = tuple(sorted((metal, donor)))
        radial.distances[key] = base_distances.get(key) or donor_distance_window(
            mol, metal, donor, real_z, donors, positions=positions, charges=charges, hyb=hyb
        )
    return radial, hyb


def _chelate_span_failure(iso, reach, radial, links, compiled=None, context=None):
    """Reject an independent long-arc chelate whose slot separation exceeds its native upper reach.

    "Long-arc" excludes a pair compile holds at its own backbone bite (`metal_slots.chelate_bite_window`);
    that pair is skipped here and left to `_compiled_span_failure`, which checks the bite row instead.

    Also rejects when this candidate's own seated bites (`metal_constraints.seated_bites`, the same windows
    `coordination` compiles) cannot jointly support the requested polyhedron at all: not a distortion `compile`
    could widen into, but a fold `metal_constraints.bounded_bites` finds even at the ideal-clamped anchor.
    A haptic bite's outright triangle is compile-only here: `_compiled_span_failure` and
    `bounds.coordination_reach` already skip every centroid row, so this screen's own seated bites only need
    to settle the fold check above, not the exact haptic window `coordination` will compile.
    """
    mol, metal, vertices, haptic = iso.graph, iso.metal, iso.vertices, iso.haptic
    poly = POLYHEDRA[iso.geometry]
    directions = poly.vertex_dirs
    measured = iso.lengths == "input"
    sphere = CoordinationSphere(
        mol, metal, iso.real_z, tuple(vertices), haptic, poly, context={} if context is None else context
    )
    ideal_angles = {
        frozenset((i, j)): vertex_angle(directions[i], directions[j])
        for i, j in itertools.combinations(range(len(vertices)), 2)
    }
    seated = seated_bites(sphere, ideal_angles, radial.distances)
    if seated and bounded_bites(directions, iso.geometry, ideal_angles, seated) is None:
        return f"chelate bites leave {iso.geometry}"
    # Acute or coupled bites are soft DG priors; only independent long arcs are an outer bound.
    independent = len({vertex for pair in links for vertex in pair}) == 2 * len(links)
    for pair in links or ():
        i, j = sorted(pair)
        left, right = vertices[i], vertices[j]
        if VACANT in (left, right):
            continue
        angle = vertex_angle(directions[i], directions[j])
        haptic_pair = bool({left, right} & haptic.keys())
        if haptic_pair:
            if (not measured and compiled is None) or angle < TRANS_ANGLE:
                continue
            failure = _haptic_span_failure(iso, reach, left, right, angle, compiled)
            if failure is not None:
                return failure
            continue
        if mol.GetBondBetweenAtoms(left, right) is not None:
            continue
        if angle < CHELATE_SPAN_ANGLE and chelate_bite_window(mol, left, right, sphere.real) is not None:
            continue  # compile holds this pair at its backbone bite, not `angle`; _compiled_span_failure checks it
        if not independent or links[pair] < _CHELATE_PATH_MIN or angle < _CHELATE_SPAN_MIN:
            continue
        radii = tuple(radial.distances[tuple(sorted((metal, donor)))][0] for donor in (left, right))
        needed = law_of_cosines(radii[0], radii[1], angle)
        available = float(reach[min(left, right), max(left, right)])
        if math.isfinite(available) and needed > available + SPAN_TOL:
            return f"chelate donors {left}/{right} need >= {needed:.3f} A; ligand reach <= {available:.3f} A"
    return None


def _haptic_span_failure(iso, reach, left, right, angle, compiled):
    """Check centroid reach; member rays need not share the angle between centroid sites."""
    metal, measured = iso.metal, iso.lengths == "input"
    left_face, right_face = iso.haptic.get(left, (left,)), iso.haptic.get(right, (right,))
    try:
        if measured:
            positions = iso.length_mol.GetConformer().GetPositions()
            radii = tuple(
                float(np.linalg.norm(np.mean(positions[list(face)], axis=0) - positions[metal]))
                for face in (left_face, right_face)
            )
        else:
            assert compiled is not None
            radii = tuple(compiled.distances[tuple(sorted((metal, vertex)))][0] for vertex in (left, right))
        available = _centroid_reach(reach, left_face, right_face)
    except (KeyError, IndexError, ZeroDivisionError):
        return None
    needed = law_of_cosines(radii[0], radii[1], angle)
    if math.isfinite(available) and needed > available + SPAN_TOL:
        return f"haptic faces {left}/{right} need >= {needed:.3f} A; centroid reach <= {available:.3f} A"
    return None


def _centroid_reach(reach, left, right):
    """Return a safe upper bound for the distance between two donor centroids.

    For any matching of equal-size faces, the centroid difference is the mean of the matched displacement
    vectors. The triangle inequality makes that matching mean an upper bound; taking the smallest matching
    keeps the certificate while dropping the looser all-pairs mean. Unequal or unusually large faces keep the
    uniform coupling bound.
    """
    values = [[float(reach[min(a, b), max(a, b)]) for b in right] for a in left]
    if not values or not values[0] or not all(math.isfinite(value) for row in values for value in row):
        return math.inf
    if len(left) == len(right) and len(left) <= _HAPTIC_MATCHING_CAP:
        return min(
            sum(row[index] for row, index in zip(values, order, strict=True))
            for order in itertools.permutations(range(len(right)))
        ) / len(left)
    return sum(map(sum, values)) / (len(left) * len(right))


def unreachable_span(iso, reach, classes, links, native=None, context=None):
    """Return why a candidate's compiled targets or span priors exceed native ligand reach, else ``None``.

    A single centre gets the compiled certificate and, with chelate links or haptic faces, the span priors
    and the opposed-donor fit budget. A multi-centre candidate, whose spectator geometry no certificate owns,
    gets the span priors only: the chelate-bite fold, the long-arc and haptic spans, and trans pairs allowed
    to close to TRANS_ANGLE (above 90 degrees a span grows with both M-L lengths). None of this proves
    chemical impossibility.
    """
    mol, metal, vertices, haptic = iso.graph, iso.metal, iso.vertices, iso.haptic
    directions = POLYHEDRA[iso.geometry].vertex_dirs
    compiled = None
    if len(iso.centres) == 1:
        failure = _compiled_span_failure(iso, reach, None, native, context)
        if failure is not None:
            return failure
        if links or iso.haptic:
            # `_compiled_span_failure` already compiled this exact candidate's full constraints (it must
            # succeed to reach here); reuse them instead of compiling the same candidate a second time.
            compiled = context.pop(_COMPILED_CONSTRAINTS_KEY, None) if context is not None else None
            if compiled is None:
                compiled = compile_constraints(iso, context=context)
        if not iso.haptic and not links:
            return None
    pairs = [
        (left, right)
        for (i, left), (j, right) in itertools.combinations(enumerate(vertices), 2)
        if VACANT not in (left, right)
        and not {left, right} & haptic.keys()
        and mol.GetBondBetweenAtoms(left, right) is None
        and vertex_angle(directions[i], directions[j]) >= TRANS_ANGLE
    ]
    if not pairs and not links:
        return None
    atoms = set(vertices) - haptic.keys() - {VACANT}
    positions, _ = resolve_lengths(iso.length_mol, iso.lengths)
    radial, hyb = _radial_distance_windows(iso, atoms, positions, iso.base_cons.distances, context=context)
    if failure := _chelate_span_failure(iso, reach, radial, links, compiled, context):
        return failure
    for left, right in pairs:
        a, b = (radial.distances[tuple(sorted((metal, donor)))][0] for donor in (left, right))
        needed = law_of_cosines(a, b, TRANS_ANGLE)
        available = float(reach[min(left, right), max(left, right)])
        if needed > available + SPAN_TOL:
            return f"donors {left}/{right} need >= {needed:.3f} A; ligand reach <= {available:.3f} A"
    return _opposed_donor_span_failure(iso, reach, radial, hyb, classes, links)


def _route_certificate_subsets(path):
    """Return a bounded set of local route witnesses after the complete route."""
    subset_count = sum(math.comb(len(path), n) for n in (3, 4) if n < len(path))
    if subset_count > _EUCLIDEAN_SUBSET_BUDGET:
        return (path,)
    return itertools.chain((path,), *(itertools.combinations(path, n) for n in (3, 4) if n < len(path)))


def _route_has_donor_bond(mol, path, donors):
    """Return whether a shortest donor route contains a bond between two metal donors.

    The route certificate assumes ordinary two-centre ligand connectivity. A donor-donor edge is a bridge or
    multicentre donor network, where its native bond bounds are not a valid proxy for the independent metal-ray
    geometry. Leave that route to the normal embedding and acceptance gates.
    """
    return any(
        left in donors and right in donors and mol.GetBondBetweenAtoms(int(left), int(right)) is not None
        for left, right in itertools.combinations(path, 2)
    )


def _route_certificate_paths(mol, donors, topology):
    """Return each donor-pair route's deduped, bond-filtered atom path, independent of vertex seating.

    Routes depend only on topology and the donor set, so `_compiled_span_failure` caches them per screen.
    """
    tested_paths = set()
    paths = []
    for left, right in itertools.combinations(sorted(donors), 2):
        separation = topology[left, right]
        if not 0 < separation < mol.GetNumAtoms():
            continue
        # The union includes every equally short route through a cyclic ligand. Selecting one route
        # depends on atom numbering; skipping ties instead loses the reach check for the entire chelate.
        path = tuple(map(int, np.flatnonzero(topology[left] + topology[right] == separation)))
        if path in tested_paths:
            continue
        tested_paths.add(path)
        if _route_has_donor_bond(mol, path, donors):
            continue
        paths.append(path)
    return paths


@functools.lru_cache(maxsize=32)
def _centering_matrix(n):
    """Return the size-`n` double-centering projector shared by every squared-distance Gram build.

    Depends only on the atom count, never on chemistry, so caching it across calls is safe; an evicted
    size is just recomputed.
    """
    return np.eye(n) - np.ones((n, n)) / n


def _euclidean_conflict(matrix, *, refine=False, context=None):
    """Say whether these distance bounds are impossible, memoised since every candidate reasks the same ones."""
    array = np.ascontiguousarray(matrix, dtype=float)
    key = (array.shape[0], array.tobytes(), bool(refine))
    if context is None:
        return _certified_conflict(*key)
    memo = context.setdefault(_CERTIFICATE_MEMO_KEY, {})
    if key not in memo:
        memo[key] = _certified_conflict(*key)
    return memo[key]


def _eigen_witnesses(groups, values, vectors, centre):
    """Yield the projector onto each negative eigenspace of the midpoint Gram matrix."""
    for group in groups:
        if min(values[group]) < 0:
            basis = centre @ vectors[:, group]
            yield basis @ basis.T


def _projected_witnesses(gram, lower, upper, n):
    """Yield the witness an accelerated projected-gradient search finds for a centred Gram matrix in the box."""
    centre = _centering_matrix(n)
    current, extrapolated, momentum = gram, gram.copy(), 1.0
    for _ in range(_CERTIFICATE_STEPS):
        squared = np.diag(extrapolated)[:, None] + np.diag(extrapolated) - 2 * extrapolated
        errors = squared - np.clip(squared, lower, upper)
        # Half the squared pair violations have gradient diag(errors.sum(1))-errors and
        # Lipschitz bound 2*n on centred Gram matrices. Simultaneous steps preserve atom symmetry.
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

    For PSD W with W*1=0, trace(W D²)=-2*trace(X.T W X)<=0 for every Euclidean point set X.
    Minimize that linear expression over the squared-distance intervals. A positive lower bound
    excludes all dimensions, not just 3D. Midpoint eigenspaces only propose W; whole projectors avoid
    arbitrary eigenvector choices at repeated eigenvalues. Optional centred-Gram refinement proposes a
    stronger witness when the midpoint misses coupled distances. Only its certified interval margin
    rejects a box, never convergence failure. No certificate does not prove feasibility.

    Takes the matrix as `n` plus its raw bytes so the result can be memoised; see `_euclidean_conflict`.
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
    witnesses = _eigen_witnesses(groups, values, vectors, centre)
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


def _upper_closure(upper):
    """Shortest-path closure of the upper bounds: a chain of bounds can hold a pair tighter than its own."""
    closed = upper.copy()
    for k in range(len(upper)):
        cand = closed[:, k, None] + closed[None, k, :]
        closed = np.where(cand < closed - _CROSS_EPS, cand, closed)
    return closed


def _lower_reach(lower, closed):
    """How far each pair is driven apart through a third atom: ``max_k (lower[i][k] - closed[k][j])``.

    Asymmetric by construction (it is atom ``i`` that is held away from ``k``), so read both orientations.
    """
    best = np.full(lower.shape, -np.inf)
    for k in range(len(lower)):
        cand = lower[:, k, None] - closed[None, k, :]
        best = np.where(cand > best, cand, best)
    return best


def _compiled_span_failure(iso, reach, compiled=None, native=None, context=None):
    """Reject a jointly inconsistent compiled donor network, not an unsuccessful embedding attempt."""
    mol = iso.graph
    if compiled is None:
        constraints = compile_constraints(iso, force_field=False, context=context) if context is not None else iso.cons
    else:
        constraints = compiled
    if constraints is not None:
        native = coordination_reach_base(mol, reach, {iso.metal}) if native is None else native
        for (left, centre, right), angles in constraints.angles.items():
            if centre != iso.metal or constraints.haptic.keys() & {left, right}:
                continue
            legs = (
                constraints.distances.get(tuple(sorted((left, centre)))),
                constraints.distances.get(tuple(sorted((centre, right)))),
            )
            if legs[0] is None or legs[1] is None:
                continue
            a, b = sorted((left, right))
            lower, upper = triangle_distances(*legs, angles)
            native_lower, native_upper = native[b, a], native[a, b]
            if lower > native_upper + _CROSS_EPS or upper < native_lower - _CROSS_EPS:
                return "compiled coordination distances conflict with native ligand reach"
        if context is not None and compiled is None:
            constraints = compile_constraints(iso, context=context)
            # Stash this candidate's full compile for `unreachable_span`'s chelate/haptic follow-up check,
            # which otherwise recompiles the identical (mol, candidate) constraints a second time.
            context[_COMPILED_CONSTRAINTS_KEY] = constraints
    matrix = (
        coordination_reach(mol, constraints, reach)
        if native is None
        else coordination_reach(mol, constraints, reach, native=native)
    )
    closed = matrix.copy()
    if not DistanceGeometry.DoTriangleSmoothing(closed):
        upper, lower = np.triu(matrix, 1), np.tril(matrix, -1)
        upper = _upper_closure(upper + upper.T)
        lower = lower + lower.T
        raised = _lower_reach(lower, upper)
        if np.any(np.maximum(np.maximum(raised, raised.T), lower) - upper > _CROSS_EPS):
            return "compiled coordination distances conflict with native ligand reach"
        return None  # An unexplained native failure is not evidence against this arrangement.
    donors = set(iso.donors)
    if context is None:
        route_paths = _route_certificate_paths(mol, donors, Chem.GetDistanceMatrix(mol, force=True))
    else:
        # Candidates that share one `context` differ only in which vertex each donor takes, so this key
        # never collides within one screen.
        route_key = (_ROUTE_ATOMS_KEY, frozenset(donors))
        route_paths = context.get(route_key)
        if route_paths is None:
            topology = Chem.GetDistanceMatrix(mol, force=True)
            route_paths = context[route_key] = _route_certificate_paths(mol, donors, topology)
    for path in route_paths:
        # Store only route unions, not their fourth-order number of subsets; stream those once per union.
        for subset in _route_certificate_subsets(path):
            atoms = sorted((iso.metal, *subset))
            # Refine the whole route once; its smaller diagnostic subsets retain the cheap midpoint test.
            if _euclidean_conflict(closed[np.ix_(atoms, atoms)], refine=subset == path, context=context) is not None:
                return f"compiled coordination distances have no Euclidean realization at atoms {atoms}"
    return None


def _fold_gap(reach, radii, axes, floors, donors, theta):
    """Return how far the pair's donor-axis atoms overshoot their native reach at angle `theta`, or 0."""
    a, b = radii[0][0], radii[1][0]
    span = law_of_cosines(a, b, math.degrees(theta))
    gaps = [0.0]
    for k, other in enumerate(reversed(donors)):
        beta = math.atan2(radii[1 - k][1] * math.sin(theta), radii[k][0] - radii[1 - k][1] * math.cos(theta))
        away = max(0.0, floors[k] - beta)
        for atom in axes[k]:
            available = float(reach[min(atom, other), max(atom, other)])
            if math.isfinite(available) and available > 0:
                gaps.append(span * math.sin(min(away, math.pi / 2)) - available - SPAN_TOL)
    return max(gaps)


def _donor_pair_fit_cost(reach, radii, axes, floors, donors):
    """Lower-bound opposed-pair fit cost using length-free donor cones."""
    if _fold_gap(reach, radii, axes, floors, donors, math.pi) <= 0:
        return 0.0
    lo, hi = math.pi / 2, math.pi
    if _fold_gap(reach, radii, axes, floors, donors, lo) > 0:
        hi = lo  # Any unexamined acute angle spends at least this much opposed-pair error.
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            if _fold_gap(reach, radii, axes, floors, donors, mid) > 0:
                hi = mid
            else:
                lo = mid
    return 4 - 4 * math.sin(hi / 2)  # Use the upper bracket: a conservative lower fit cost.


def _opposed_donor_span_failure(iso, reach, radial, hyb, classes, links):
    """Reject only when donor-facing reach exhausts the shared fit budget in every equivalent seating.

    An ideal opposed pair capped at angle theta costs at least 4-4*sin(theta/2), a squared unit-ray error on
    the same raw (unaveraged, pre-sqrt) scale `metal_polyhedron.fit_residual` averages and roots: occupied *
    FIT_FLOOR**2 converts the per-vertex RMS floor back into that raw-sum budget for `occupied` vertices, so
    disjoint pairs sum against one shared budget, not a fresh one per chelate. Caps follow from the existing
    single-endpoint cones without assuming a donor-substituent bond length. Native ligand reach intervals
    remain model priors, not proof of chemical impossibility.
    """
    vertices, mol, metal = iso.vertices, iso.graph, iso.metal
    occupied = sum(vertex != VACANT for vertex in vertices)
    if len(iso.centres) != 1:
        return None
    donors = set(iso.donors)
    # Coordination can expand radial caps for bonded co-donors; the uncompiled windows cannot bound them.
    bonded_donors = {d for d in donors if any(n.GetIdx() in donors for n in mol.GetAtomWithIdx(d).GetNeighbors())}
    directions = [
        tuple(value / math.hypot(*direction) for value in direction)
        for direction in POLYHEDRA[iso.geometry].vertex_dirs
    ]
    opposed = [
        (i, j)
        for i, j in itertools.combinations(range(len(vertices)), 2)
        if VACANT not in (vertices[i], vertices[j])
        and math.dist(directions[i], tuple(-value for value in directions[j])) <= _FIT_ROUNDOFF
    ]
    used = [slot for pair in opposed for slot in pair]
    if not opposed or len(used) != len(set(used)):
        return None  # A nonstandard template cannot spend one donor's fit error twice.

    @functools.cache
    def pair_cost(left, right):
        if {left, right} & (iso.haptic.keys() | bonded_donors):
            return 0.0
        if not Chem.GetShortestPath(mol, left, right):
            return 0.0
        radii = [radial.distances[tuple(sorted((metal, atom)))] for atom in (left, right)]
        radii = [(lo - ML_WINDOW_TOL, hi + ML_WINDOW_TOL) for lo, hi in radii]
        if any(not (0 < lo <= hi) or not all(map(math.isfinite, (lo, hi))) for lo, hi in radii):
            return 0.0
        axes, floors = [], []
        for atom in (left, right):
            floor = FOLD_WINDOW.get((mol.GetAtomWithIdx(atom).GetSymbol(), hyb.get(atom)))
            floors.append(math.radians(floor[0]) if floor else 0.0)
            axes.append(tuple(donation_axis(mol, atom, donors, hyb=hyb, network=False) or ()) if floor else ())
        if not any(axes):
            return 0.0

        return _donor_pair_fit_cost(reach, radii, axes, floors, (left, right))

    keys = [None if vertex == VACANT else classes[vertex] for vertex in vertices]
    budget = occupied * FIT_FLOOR**2
    minimum = math.inf
    for assignment in equivalent_site_assignments(keys, links):
        cost = sum(pair_cost(*sorted((vertices[assignment[i]], vertices[assignment[j]]))) for i, j in opposed)
        if cost <= budget + _FIT_ROUNDOFF:
            return None
        minimum = min(minimum, cost)
    if not math.isfinite(minimum):
        return None
    return f"donor-facing network needs squared fit error >= {minimum:.3f}; shape budget is {budget:.3f}"
