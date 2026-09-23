"""Edit native RDKit bounds and seed coordinates with KDG + AIO.

A supplied EmbedParameters selects another model. SetBoundsMat supplies distances, not the internal
coordinate assumptions that produced them. Triangle smoothing detects metric contradictions, not every
3D infeasibility.
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
from contextlib import nullcontext

import numpy as np
from rdkit import Chem, DistanceGeometry, rdBase
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from . import mechanisms as _mech
from . import metal_core as _metal
from .constraints import FIX_DISTANCE_TOL

logger = logging.getLogger("rxembed.bounds")  # under the "rxembed" tree `set_verbose` configures

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
_SMOOTH_LOOSE = 0.1  # smoothing beyond this means the constraints are genuinely contradictory, not merely tight
_MAX_SEED_COUNT = 250  # bound one RDKit DG search; stereo selection reuses this ceiling in small serial batches
_PROJECTOR_EPS = 1e-10  # Ignore the numerically null centered constant eigenspace.
_CERTIFICATE_STEPS = 32  # Bounded proposal search, not a feasibility threshold; see benchmark/.
_CROSS_EPS = 1e-9  # Å, floating-point slack shared by every closed-bound comparison against this matrix


def _smooth(bm, max_tol=0.4):
    """Triangle-smooth ``bm`` in place and return the crossover tolerance used.

    Positive tolerance permits RDKit to repair crossed bounds, including ligand internals, by raising the
    upper bound to the lower. It is a fraction of the upper bound, not a distance. Each retry starts from
    the original matrix; `_feasible_bounds` names the repaired pair by diffing it against that original.
    """
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed")
        bm[:] = back
    if tol > _SMOOTH_LOOSE:  # `_feasible_bounds` owns the user-facing narrative and names the pair there
        logger.debug("smoothing repaired a %.0f%% bound crossover", tol * 100.0)
    return tol


def embed_parameters(seed, *, knowledge=True, threads=0, prune_rms=None):
    """Build native KDG + AIO parameters with reproducible sampling defaults.

    Require a seed rather than inheriting RDKit's process-global RNG default. ``knowledge=False`` disables
    basic geometry for plain distance geometry, retaining AIO refinement.
    """
    p = rdDistGeom.KDG()
    p.useLegacyImplementation = False
    p.randomSeed = int(seed)
    p.numThreads = threads
    p.useBasicKnowledge = knowledge
    if prune_rms is not None:
        p.pruneRmsThresh = prune_rms  # passed through with RDKit's meaning: 0.0 prunes only identical, -1 is off
    return p


def embedding_options(
    seed,
    threads,
    knowledge,
    prune_rms=None,
    embed_params=None,
    *,
    coplanar_14=True,
    metal_floor_relief=True,
    donor_orientation=True,
    conjugation=True,
):
    """Resolve convenience keywords once, leaving native model choices on their parameter object."""
    for name, value in (
        ("coplanar_14", coplanar_14),
        ("metal_floor_relief", metal_floor_relief),
        ("donor_orientation", donor_orientation),
        ("conjugation", conjugation),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False")
    if embed_params is not None:
        if not isinstance(embed_params, rdDistGeom.EmbedParameters):
            raise TypeError("embed_params must be an RDKit EmbedParameters object")
        if knowledge is not None:
            raise ValueError("use embed_params.useBasicKnowledge/useExpTorsionAnglePrefs, not knowledge together")
        if seed is None:
            seed = DEFAULT_SEED if embed_params.randomSeed == -1 else embed_params.randomSeed
        threads = embed_params.numThreads if threads is None else threads
        prune_rms = embed_params.pruneRmsThresh if prune_rms is None else prune_rms
    return (
        DEFAULT_SEED if seed is None else int(seed),
        0 if threads is None else int(threads),
        True if knowledge is None else knowledge,
        0.1 if prune_rms is None else float(prune_rms),
    )


def probe_conformer(mol, seed):
    """Embed one throwaway conformer on a copy of ``mol``, or return None on failure.

    Probes choose contact atoms, not conformer diversity, so require an explicit reproducible seed.
    """
    work = Chem.Mol(mol)
    if rdDistGeom.EmbedMolecule(work, embed_parameters(seed)) != 0:
        return None
    return work


def _chain_upper(bm, path):
    """Enclose three-bond reach at every torsion using the native bond/angle intervals."""
    a, b, c, d = path
    intervals = [
        (bm[max(i, j), min(i, j)], bm[min(i, j), max(i, j)]) for i, j in ((a, b), (b, c), (c, d), (a, c), (b, d))
    ]
    if any(not 0 < lo <= hi or not math.isfinite(hi) for lo, hi in intervals):
        return math.inf
    ab, bc, cd, ac, bd = intervals
    alpha, beta = _mech._triangle_angles(ab, bc, ac), _mech._triangle_angles(bc, cd, bd)
    if alpha is None or beta is None:
        return math.inf
    # Relax the terminal-bond dot product to a*c, then use each angle's maximum. The resulting expression
    # is convex in each bond length, so interval corners enclose every torsion, including cis amides and S-S.
    square = max(
        (a + c - b) ** 2 + 4 * b * (a * math.sin(alpha[1] / 2) ** 2 + c * math.sin(beta[1] / 2) ** 2)
        for a, b, c in itertools.product(ab, bc, cd)
    )
    return math.sqrt(square + 1e-12 * max(1.0, square))  # Outward roundoff, not a chemical tolerance.


def ligand_reach(mol):
    """Close native ligand bounds with torsion-independent three-bond upper limits for enumeration.

    Native 1-4/1-5 bounds include conformational preferences. Removing them also loses useful free-torsion
    reach, which triangle smoothing alone cannot recover. Project that consequence from one unmodified
    1-2/1-3 basis before smoothing. A ring-closed path keeps RDKit's native 1-4 lower bound: unlike an
    acyclic chain, its closure is a graph constraint rather than a free torsion (NITWIY). The native 1-4
    upper bound only carries over when the central bond is aromatic; RDKit derives any other ring bond's
    1-4 upper bound from an sp2-sp2 cis template regardless of ring size (`_getShareRingBond14Type`), which
    is a hybridisation preference, not a ring constraint, and otherwise underestimates a saturated-ring
    torsion (JOYDIK: a thiourea S...S reach twisted by a diazepane ring). This is a model upper bound,
    not proof of whole-ligand realizability; embedding still retains RDKit's torsion knowledge.
    Unsupported local projections abstain.
    """
    with rdBase.BlockLogs():
        basis = rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=False, set15bounds=False, doTriangleSmoothing=False)
        ring_bounds = rdDistGeom.GetMoleculeBoundsMatrix(
            mol, set14bounds=True, set15bounds=False, doTriangleSmoothing=False
        )
    reach = basis.copy()
    for path in Chem.FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=True, onlyShortestPaths=True):
        a, b = sorted((path[0], path[-1]))
        bond = mol.GetBondBetweenAtoms(path[1], path[2])
        if bond.IsInRing():
            reach[b, a] = ring_bounds[b, a]
        if bond.GetIsAromatic():
            reach[a, b] = ring_bounds[a, b]
        else:
            reach[a, b] = min(reach[a, b], _chain_upper(basis, path))
    if not DistanceGeometry.DoTriangleSmoothing(reach):
        raise ValueError("native ligand reach bounds are inconsistent")
    return reach


def _put(matrix, atoms, window):
    """Narrow one matrix cell to admit ``window``, keyed the way the matrix sorts a pair."""
    a, b = sorted(atoms)
    matrix[b, a] = max(matrix[b, a], window[0])
    matrix[a, b] = min(matrix[a, b], window[1])


def _coordination_reach_base(mol, reach, metals):
    """Build the native ligand interval matrix shared by one coordination screen."""
    matrix = np.triu(np.full((mol.GetNumAtoms(), mol.GetNumAtoms()), math.inf), 1)
    topology = Chem.GetDistanceMatrix(mol, force=True)
    with rdBase.BlockLogs():
        raw = rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=False, set15bounds=False, doTriangleSmoothing=False)
    for fragment in Chem.GetMolFrags(mol):
        for a, b in itertools.combinations(sorted(set(fragment) - set(metals)), 2):
            _put(matrix, (a, b), (0.0, reach[a, b]))
            if topology[a, b] in (1, 2):
                _put(matrix, (a, b), (raw[b, a], raw[a, b]))
    return matrix


def coordination_reach(mol, cons, reach, *, native=None):
    """Intersect compiled targets with native ligand reach, without repairing incompatible model priors.

    An enumeration outer bound, not the embedding matrix: native ligand 1-2/1-3 intervals and free-torsion
    upper reach are kept, nonbonded floors come only from `cons`, and the publication tolerance is added to
    real metal-distance windows so the screen cannot reject their accepted boundary. Every angle projects
    from the same closed side intervals, so dictionary order cannot change a leg. Virtual-centroid rows are
    omitted: a native interval is only a seed prior, and a hard centroid equality there could turn an
    accepted ligand distortion into a false exclusion; the caller already excludes externally constrained
    graphs, so the omission cannot strengthen this bound.
    """
    matrix = _coordination_reach_base(mol, reach, cons.metals) if native is None else native.copy()
    for atoms, (lo, hi) in cons.distances.items():
        if cons.haptic.keys() & set(atoms):
            continue
        tolerance = FIX_DISTANCE_TOL if cons.metals.intersection(atoms) else 0.0
        _put(matrix, atoms, (max(0.0, lo - tolerance), hi + tolerance))
    for atoms, floor in cons.floors.items():
        if cons.haptic.keys() & set(atoms):
            continue
        _put(matrix, atoms, (floor, math.inf))
    basis = matrix.copy()
    if not DistanceGeometry.DoTriangleSmoothing(basis):
        return matrix
    for (i, j, k), angles in cons.angles.items():
        if cons.haptic.keys() & {i, j, k}:
            continue
        left = basis[max(i, j), min(i, j)], basis[min(i, j), max(i, j)]
        right = basis[max(j, k), min(j, k)], basis[min(j, k), max(j, k)]
        if np.isfinite((*left, *right)).all():
            _put(matrix, (i, k), _mech._triangle_distances(left, right, angles))
    return matrix


@functools.lru_cache(maxsize=32)
def _centering_matrix(n):
    """Return the size-`n` double-centering projector shared by every squared-distance Gram build.

    A pure function of the certificate's atom count, never of chemistry, so it is safe to share across
    calls. `_euclidean_conflict` only certifies small metal-anchored subsets (route unions bounded by
    `metal_enumeration._EUCLIDEAN_SUBSET_BUDGET`, mostly 4-6 atoms), so distinct sizes stay well under
    this cache's bound; an evicted size is just recomputed, never a correctness risk.
    """
    return np.eye(n) - np.ones((n, n)) / n


def _euclidean_conflict(matrix, *, refine=False):
    """Say whether these distance bounds are impossible, cached since every candidate reasks the same ones."""
    array = np.ascontiguousarray(matrix, dtype=float)
    return _certified_conflict(array.shape[0], array.tobytes(), bool(refine))


@functools.lru_cache(maxsize=4096)
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

    def witnesses():
        for group in groups:
            if min(values[group]) < 0:
                basis = centre @ vectors[:, group]
                yield basis @ basis.T
        if not refine:
            return
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

    for candidate in witnesses():
        if np.trace(candidate) < _PROJECTOR_EPS:
            continue
        weights = candidate / np.trace(candidate)
        terms = weights * np.where(weights >= 0, lower, upper)
        margin = float(terms.sum())
        if margin > 1e-9 * max(1.0, float(np.abs(terms).sum())):
            return margin
    return None


def _write(mol, cons, params=None):
    """Edit native smoothed bounds without re-smoothing the edits; return the filled context.

    The phase order is the algorithm, and this is the only place it is stated; each mechanism
    (`mechanisms.py`) says what it writes, never when. RELIEVE must precede COMMIT so an explicit
    window always beats a floor relief; POST must follow it so the coplanar bound can read committed legs.

    Separate from `_bounds` because smoothing repairs in place: `_feasible_bounds` needs the matrix as
    written, before repair, to diff against the smoothed result.
    """
    # Bounds generation tries UFF typing even though no force field is requested. The shared wrapper hides
    # those diagnostics only for charged/metal graphs; exceptions still escape.
    # Use the same native priors for the edited matrix and its refinement.
    # Labelled ligand stereo temporarily adds dative M-L edges so RDKit has the donor's full CIP basis. Remove
    # only edges owned by a selected metal's explicit M-L distance on a private copy; retain every other edge.
    owned = {tuple(sorted(atoms)) for atoms in cons.distances}
    removable = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DATIVE:
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        metals = [idx for idx in (begin, end) if mol.GetAtomWithIdx(idx).GetAtomicNum() in _metal.COORDINATION_METALS]
        if len(metals) == 1 and metals[0] in cons.metals and tuple(sorted((begin, end))) in owned:
            removable.append((begin, end))
    native = mol
    if removable:
        rw = Chem.RWMol(mol)
        for begin, end in removable:
            rw.RemoveBond(begin, end)
        native = rw.GetMol()
        native.ClearComputedProps()
        native.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(native, includeDativeBonds=True)
    bm = _metal._bounds_matrix(native, params)
    ctx = _mech.DGContext(mol, bm)
    for m in _mech.MECHANISM_ORDER:
        m._dg_windows(cons, ctx)  # WINDOW   distances, angles, planes -> candidate windows
    for m in _mech.MECHANISM_ORDER:
        m._dg_relief(cons, ctx)  # RELIEVE  lower RDKit's phantom floors, before anything is committed
    for (i, j), (lo, hi) in ctx.pairs.items():
        a, b = (i, j) if i < j else (j, i)
        ctx.bm[a][b], ctx.bm[b][a] = hi, lo  # COMMIT
    for m in _mech.MECHANISM_ORDER:
        m._dg_post(cons, ctx)  # POST     read the committed matrix (the coplanar 1,4 bound)
    return ctx


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


def _bounds(mol, cons, params=None):
    """Edit RDKit's bounds matrix with every constraint; return ``(matrix, settled tolerance)``."""
    bm = _write(mol, cons, params).bm
    return bm, _smooth(bm)  # SMOOTH; the settled tolerance travels with the matrix


def _feasible_bounds(mol, cons, params=None):
    """Build smoothed seed bounds and report the pair implicated in any repair.

    Smoothing may repair the seed matrix, but must not rewrite the Constraints used for relaxation and
    acceptance. A stated window moved by the repair is named first, even where smoothing moved an unstated
    pair further, because only a stated one is actionable; a full attribution needs the closure this no
    longer keeps (see git history).
    """
    bm, tol = _bounds(mol, cons, params)
    if tol <= 0.0:
        return bm, tol  # no triangle contradiction detected; full embedding and validation are still required
    ctx = _write(mol, cons, params)  # rebuild the pre-smoothing matrix to diff the repair against
    diff = np.abs(np.asarray(bm, float) - ctx.bm)
    changed = np.triu(np.maximum(diff, diff.T), 1)  # either bound moving counts, so fold lower onto upper
    stated = [(min(i, j), max(i, j)) for i, j in ctx.pairs]
    named = max(stated, key=lambda p: changed[p], default=None)
    if named is None or changed[named] == 0.0:
        named = tuple(int(x) for x in np.unravel_index(np.argmax(changed), changed.shape))
    i, j = named
    kind = "distance" if named in cons.distances else "atoms"
    logger.warning(
        "RDKit DG bounds needed smoothing for %s %d-%d (gap %.2f A); continuing with the repaired seed bounds "
        "before constrained relaxation",
        kind,
        i,
        j,
        float(changed[named]),
    )
    return bm, tol


def _bring_real_confs(mol, work, ids):
    """Copy each embedded conformer's real-atom positions from `work` (with phantom) onto the real `mol`, same ids.

    The haptic centroid dummy lives only in the transient `work`; the real molecule keeps exactly its own atoms.
    `EmbedMultipleConfs` replaced `work`'s conformers, so replace `mol`'s conformers too.
    """
    mol.RemoveAllConformers()
    n = mol.GetNumAtoms()
    for cid in ids:
        wc = work.GetConformer(int(cid))
        conf = Chem.Conformer(n)
        conf.SetId(int(cid))
        conf.SetPositions(wc.GetPositions()[:n])
        mol.AddConformer(conf, assignId=False)


def seed_coordinates(
    mol,
    cons,
    n,
    seed=DEFAULT_SEED,
    prune_rms=0.1,
    knowledge=True,
    threads=0,
    enforce_chirality=True,
    max_attempts=0,
    embed_params=None,
    random_coords=False,
    coplanar_14=True,
    metal_floor_relief=True,
):
    """Generate ``n`` new conformers with native RDKit parameters and edited bounds."""
    work = _metal.materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    constrained = bool(cons.distances or cons.angles or cons.planes or cons.coplanar)
    search_count = int(n)
    p = (
        embed_params
        if embed_params is not None
        else embed_parameters(seed, knowledge=knowledge, threads=threads, prune_rms=prune_rms)
    )
    if embed_params is None:
        p.enforceChirality = enforce_chirality
        p.maxIterations = int(max_attempts)
        p.trackFailures = True
    if embed_params is not None or constrained:
        # Always replace a supplied object's matrix: it may belong to a previous candidate or helper graph.
        # These switches affect only our matrix edits. UFF and publication retain the complete Constraints.
        dg = (
            cons
            if coplanar_14 and metal_floor_relief
            else cons.copy(
                coplanar=cons.coplanar if coplanar_14 else [],
                dg_floors=cons.dg_floors if metal_floor_relief else {},
            )
        )
        bm, _tol = _feasible_bounds(work, dg, p)
        p.SetBoundsMat(bm)  # a custom (edited) bounds matrix
    overrides = {"randomSeed": seed, "numThreads": threads, "pruneRmsThresh": prune_rms, "clearConfs": True}
    if constrained or embed_params is not None:
        overrides["embedFragmentsSeparately"] = False
    previous = {}
    if embed_params is not None:
        previous = {name: getattr(p, name) for name in overrides}
        # Native embedding materialises these defaults in place; do not leak them across molecules.
        previous.update(maxIterations=p.maxIterations, basinThresh=p.basinThresh)
    # Disconnected hydrides are intentional. Suppress their native implicit-H warning only during embedding.
    isolated_h = any(a.GetAtomicNum() == 1 and not a.GetDegree() for a in work.GetAtoms())
    try:
        for name, value in overrides.items():
            setattr(p, name, value)
        for attempt in range(2 if embed_params is None else 1):
            if embed_params is None:
                p.useRandomCoords = bool(
                    random_coords or attempt
                )  # Retry an empty search with the same model and bounds.
            with rdBase.BlockLogs() if isolated_h else nullcontext():
                ids = list(rdDistGeom.EmbedMultipleConfs(work, search_count, p))
            # Counts describe rejected native attempts, even in successful searches, not failed conformers.
            counts = p.GetFailureCounts() if p.trackFailures else ()
            failures = {
                name: counts[int(cause)]
                for name, cause in rdDistGeom.EmbedFailureCauses.names.items()
                if int(cause) < len(counts) and counts[int(cause)]
            }
            logger.debug(
                "DG random=%s knowledge=%s chirality=%s legacy=%s: %d/%d conformers; rejected attempts %s",
                p.useRandomCoords,
                p.useBasicKnowledge,
                p.enforceChirality,
                p.useLegacyImplementation,
                len(ids),
                search_count,
                failures if p.trackFailures else "not tracked",
            )
            if any(cid < 0 for cid in ids):
                raise TimeoutError(
                    f"native RDKit embedding exceeded timeout={p.timeout}s; "
                    "increase embed_params.timeout or inspect bounds/stereo with rx.set_verbose('DEBUG')"
                )
            if ids:
                break
    finally:
        for name, value in previous.items():
            setattr(p, name, value)
    from .relax import bonding_ok

    # Stereo selection may stop after its first match. Try intact seeds before repairable candidates.
    ids.sort(key=lambda cid: not bonding_ok(work, cid, clash_tol=0.0, exclude=cons.frozen, constrained=cons.distances))
    if work is not mol:  # discard the phantom: bring only the real-atom coords back onto the real molecule
        _bring_real_confs(mol, work, ids)
    return ids


def seed_count(mol, constrained=False):
    """Return the RDKit DG seed count, scaled by rotatable-bond count rather than a flat default.

    A constrained run gets ~1.6x more, because openconf's pose-frozen search is rotor-only and under-samples
    unless handed more distinct starting points. An unconstrained ``.mc()`` re-seeds and replaces these.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(_MAX_SEED_COUNT, max(40, 10 * r))
    return min(150, max(24, 6 * r))  # cf. openconf max(20, 3*r); a touch more for biased seeds
