"""Edit RDKit's native bounds and draw bounded KDG seeds, unless `EmbedParams.native` selects RDKit's own search.

SetBoundsMat supplies distances, not the internal-coordinate assumptions behind them, and triangle smoothing finds
metric contradictions, not every 3D infeasibility.
"""

from __future__ import annotations

import itertools
import logging
import math
import zlib
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem, DistanceGeometry, rdBase
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from rxembed.geometry.constraints import DIST_ATOMS, canonical_key, is_index, metal_distance_tolerance
from rxembed.geometry.mechanisms import MECHANISM_ORDER, DGContext, triangle_angles, triangle_distances
from rxembed.geometry.relax import bonding_failure
from rxembed.graph import COORDINATION_METALS, SURROGATE, materialise_phantoms
from rxembed.utils import atom_label

logger = logging.getLogger("rxembed.bounds")  # under the "rxembed" tree `set_verbose` configures

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
_MAX_SMOOTH_TOL = 0.4  # crossover ratio (not Angstrom) beyond which smoothing gives up
MAX_SEED_COUNT = 250  # bound one RDKit DG search; stereo selection reuses this ceiling in small serial batches


def bounds_matrix(mol, params=None):
    """Return RDKit's bounds matrix of `mol`, its UFF-typing log silenced on a charged or metal graph."""
    noisy = any(atom.GetFormalCharge() or atom.GetAtomicNum() in COORDINATION_METALS for atom in mol.GetAtoms())
    options = {} if params is None else {"embedParams": params}
    if noisy:
        with rdBase.BlockLogs():
            return rdDistGeom.GetMoleculeBoundsMatrix(mol, **options)
    return rdDistGeom.GetMoleculeBoundsMatrix(mol, **options)


def embed_parameters(seed, *, knowledge=True, threads=0, prune_rms=None):
    """Build KDG + AIO parameters with five attempts per candidate and a required seed.

    ``knowledge=False`` turns off basic geometry for plain distance geometry, keeping AIO refinement.
    """
    p = rdDistGeom.KDG()
    p.useLegacyImplementation = False
    p.maxIterations = 5
    p.trackFailures = True
    p.randomSeed = int(seed)
    p.numThreads = threads
    p.useBasicKnowledge = knowledge
    if prune_rms is not None:
        p.pruneRmsThresh = prune_rms  # passed through with RDKit's meaning: 0.0 prunes only identical, -1 is off
    return p


LENGTHS = ("model", "input")


@dataclass(frozen=True, kw_only=True)
class EmbedParams:
    """Hold the DG model and rxembed's switches; seed and threads are the verbs' `seed=` and `threads=`.

    `native` selects RDKit's own `EmbedParameters` in place of rxembed's bounded KDG search and is kept by reference:
    RDKit's object cannot be copied, so do not share it between concurrent embeds. `prune_rms` and `knowledge` of None
    let the model decide: rxembed's prunes at 0.1 with basic knowledge on, a native object keeps its own.
    """

    prune_rms: float | None = None
    knowledge: bool | None = None
    native: rdDistGeom.EmbedParameters | None = None
    coplanar_14: bool = True
    metal_floor_relief: bool = True
    donor_orientation: bool = True
    conjugation: bool = True
    lengths: str = "model"  # metal M-donor windows: the distance model, or "input" measured from the conformer

    def __post_init__(self):
        """Check that every setting has exactly one owner: rxembed, or a supplied native object."""
        for name in ("coplanar_14", "metal_floor_relief", "donor_orientation", "conjugation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be True or False")
        if self.lengths not in LENGTHS:
            raise ValueError(f"lengths={self.lengths!r}; expected one of {LENGTHS}")
        if self.knowledge is not None and not isinstance(self.knowledge, bool):
            raise TypeError("knowledge must be True, False or None")
        if self.native is not None:
            self._check_native()

    def _check_native(self):
        """Refuse a native object setting the sampling rxembed owns, or a useBasicKnowledge `knowledge` contradicts."""
        if not isinstance(self.native, rdDistGeom.EmbedParameters):
            raise TypeError("native must be an RDKit EmbedParameters object")
        # rxembed owns sampling; a native object must arrive at RDKit's own defaults for these three fields.
        for rdkit_name, setting, default in (
            ("randomSeed", "the verb's seed=", -1),
            ("numThreads", "the verb's threads=", 1),
            ("pruneRmsThresh", "EmbedParams(prune_rms=...)", -1.0),
        ):
            current = getattr(self.native, rdkit_name)
            if current != default:
                raise ValueError(
                    f"native.{rdkit_name}={current} is set; pass {setting} and leave native.{rdkit_name} at {default}"
                )
        # The DG model belongs to RDKit; knowledge= only sets rxembed's own KDG model.
        if self.knowledge is not None and self.knowledge != self.native.useBasicKnowledge:
            raise ValueError(
                f"native.useBasicKnowledge={self.native.useBasicKnowledge} is set; pass "
                f"knowledge={self.native.useBasicKnowledge} or leave knowledge=None"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampling:
    """Hold what reproduces one embed's seed batches: its `EmbedParams`, seed and thread count.

    A seed below 1 is refused: RDKit seeds conformer i with seed * (i + 1), so seed 0 makes every conformer the same.
    """

    params: EmbedParams = field(default_factory=EmbedParams)
    seed: int = DEFAULT_SEED
    threads: int = 0

    def __post_init__(self):
        """Refuse params that are not `EmbedParams`, or a seed RDKit cannot draw distinct conformers from."""
        if not isinstance(self.params, EmbedParams):
            raise TypeError(
                f"params= takes EmbedParams, got {type(self.params).__name__}; wrap an RDKit object as "
                "EmbedParams(native=...)"
            )
        if not is_index(self.seed) or self.seed <= 0:
            raise ValueError(
                f"seed={self.seed!r}: RDKit multiplies the seed by the conformer index; give a seed >= 1 as an integer"
            )
        object.__setattr__(self, "seed", int(self.seed))

    def retry(self, stage, index):
        """Return this sampling for retry `index` of `stage`, with a seed no other stage or neighbour shares."""
        if not index:
            return self
        key = np.random.SeedSequence([self.seed, zlib.crc32(stage.encode()), index])
        # RDKit's generator is minstd, modulo 2**31 - 1, so a seed of 0 or 2**31 - 1 seeds every conformer alike.
        return replace(self, seed=int(key.generate_state(1)[0]) % (2**31 - 2) + 1)


def resolve_sampling(params, seed, threads):
    """Return a verb's `Sampling`; each of `params`, `seed` and `threads` left as None takes the record's default."""
    given = {"params": params, "seed": seed, "threads": threads}
    return Sampling(**{name: value for name, value in given.items() if value is not None})


def probe_conformer(mol, seed):
    """Embed one throwaway conformer on a copy of ``mol``, or return None on failure.

    Probes choose contact atoms, not conformer diversity, so require an explicit reproducible seed.
    """
    work = Chem.Mol(mol)
    if not _embed_batch(work, 1, embed_parameters(seed), fallback=True):
        return None
    return work


def chain_upper(bm, path):
    """Enclose three-bond reach at every torsion using the native bond/angle intervals."""
    a, b, c, d = path
    intervals = [
        (bm[max(i, j), min(i, j)], bm[min(i, j), max(i, j)]) for i, j in ((a, b), (b, c), (c, d), (a, c), (b, d))
    ]
    if any(not 0 < lo <= hi or not math.isfinite(hi) for lo, hi in intervals):
        return math.inf
    ab, bc, cd, ac, bd = intervals
    alpha, beta = triangle_angles(ab, bc, ac), triangle_angles(bc, cd, bd)
    if alpha is None or beta is None:
        return math.inf
    # relaxing the terminal-bond dot product to a*c keeps the reach convex, so interval corners enclose every torsion
    square = max(
        (a + c - b) ** 2 + 4 * b * (a * math.sin(alpha[1] / 2) ** 2 + c * math.sin(beta[1] / 2) ** 2)
        for a, b, c in itertools.product(ab, bc, cd)
    )
    return math.sqrt(square + 1e-12 * max(1.0, square))  # Outward roundoff, not a chemical tolerance.


def ligand_reach(mol):
    """Return native ligand intervals with a torsion-independent upper reach, for enumeration.

    Native 1-4 and 1-5 bounds encode conformational preferences, so each three-bond upper limit is projected from the
    unmodified 1-2 and 1-3 basis instead, and a path it cannot project abstains. A ring-closed path keeps RDKit's 1-4
    lower bound, and its 1-4 upper bound only across an aromatic bond: RDKit takes any other ring bond's from an sp2 cis
    template, which underestimates a saturated ring. Pairs in one fragment take the closed reach as upper bound, a 1-2
    or 1-3 pair the native interval too; other lower bounds are zero and pairs between fragments unbounded. This is a
    model bound, not proof that the ligand can be built.
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
            reach[a, b] = min(reach[a, b], chain_upper(basis, path))
    if not DistanceGeometry.DoTriangleSmoothing(reach):
        raise ValueError("native ligand reach bounds are inconsistent; check the ligand graph's bonds and charges")
    matrix = np.triu(np.full((mol.GetNumAtoms(), mol.GetNumAtoms()), math.inf), 1)
    topology = Chem.GetDistanceMatrix(mol, force=True)
    for fragment in Chem.GetMolFrags(mol):
        for a, b in itertools.combinations(sorted(fragment), 2):
            _put(matrix, (a, b), (0.0, reach[a, b]))
            if topology[a, b] in (1, 2):
                _put(matrix, (a, b), (basis[b, a], basis[a, b]))
    return matrix


def _put(matrix, atoms, window):
    """Narrow one matrix cell to admit ``window``, keyed the way the matrix sorts a pair."""
    a, b = sorted(atoms)
    matrix[b, a] = max(matrix[b, a], window[0])
    matrix[a, b] = min(matrix[a, b], window[1])


_FRAGMENT_CONTACT_SLACK = 1.5  # A: van der Waals contact slack, the same approach an H-bond donor makes to its acceptor
_NATIVE_UNSET = 900.0  # A: below RDKit's raw "no computed bound" default of 1000 A, above any real finite reach


def _fragment_components(mol, cons):
    """Group atoms into chemical components: RDKit fragments joined by any requested distance.

    A dative M-L bond lives only in `cons.distances`, so RDKit's graph puts a complex's donor arms in separate
    fragments; a real spectator (counterion, free ligand, solvent) stays disconnected. None when there is nothing to
    group.
    """
    frags = Chem.GetMolFrags(mol)
    if len(frags) < 2:  # noqa: PLR2004 - fewer than two fragments leaves nothing to group
        return None
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    groups = [set(fragment) for fragment in frags]
    for i, j in cons.distances:
        if i not in frag_of or j not in frag_of:
            continue  # a haptic centroid is materialised only after this step
        left, right = frag_of[i], frag_of[j]
        if groups[left] is groups[right]:
            continue
        joined = groups[left] | groups[right]
        for atom in joined:
            groups[frag_of[atom]] = joined
    components = list({id(group): tuple(sorted(group)) for group in groups}.values())
    return components if len(components) > 1 else None


def fragment_contacts(mol, cons, bm):
    """Return ``{(i, j): (floor, ceiling)}`` keeping every free cross-component heavy pair within contact range.

    RDKit gives a pair with no bonded path an unbounded upper limit, so a free component could be placed anywhere. Every
    cross-component pair gets one ceiling: RDKit's vdW floor plus each side's largest span plus slack, so a fragment
    settles wherever distance geometry puts it, a vacant metal site if one fits.
    """
    components = _fragment_components(mol, cons)
    if components is None:
        return {}
    component_of = {a: ci for ci, c in enumerate(components) for a in c}
    # two pinned components get no ceiling, which could fight an intended separation such as two frozen TS fragments
    touched = {component_of[atom] for atom in cons.constrained_atoms() if atom in component_of}
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    reach = [0.0] * len(components)
    for ci, comp in enumerate(components):
        members = [a for a in comp if mol.GetAtomWithIdx(a).GetAtomicNum() > 1]
        for x, y in itertools.combinations(members, 2):
            span = bm[min(x, y)][max(x, y)]
            if span < _NATIVE_UNSET and span > reach[ci]:
                reach[ci] = span
    windows = {}
    for i, j in itertools.combinations(heavy, 2):
        ci, cj = component_of.get(i), component_of.get(j)
        if ci is None or cj is None or ci == cj or (ci in touched and cj in touched):
            continue
        a, b = min(i, j), max(i, j)
        vdw = bm[b][a]  # RDKit's own van der Waals floor for this pair
        windows[a, b] = (vdw, vdw + reach[ci] + reach[cj] + _FRAGMENT_CONTACT_SLACK)
    return windows


def coordination_reach(cons, native):
    """Write compiled targets over the native ligand reach `native` into an enumeration outer bound.

    `native` is copied, never repaired. Each requested distance owns its cell and no angle row reaches it, as in the
    seed bounds, so a ``fix=`` on two donors makes no cis or trans choice. Each metal-distance window is widened by its
    acceptance tolerance; angle rows and native reach are not, so the screen refuses some species acceptance would take.
    Haptic-centroid rows are left out, since a hard centroid equality could exclude a distortion acceptance takes.
    """
    matrix = native.copy()
    for atoms, (lo, hi) in cons.distances.items():
        if cons.haptic.keys() & set(atoms):
            continue
        tolerance = metal_distance_tolerance(atoms, cons) if cons.metals.intersection(atoms) else 0.0
        a, b = sorted(atoms)
        matrix[b, a], matrix[a, b] = max(0.0, lo - tolerance), hi + tolerance
    for atoms, floor in cons.floors.items():
        if cons.haptic.keys() & set(atoms):
            continue
        _put(matrix, atoms, (floor, math.inf))
    basis = matrix.copy()
    if not DistanceGeometry.DoTriangleSmoothing(basis):
        return matrix
    for (i, j, k), angles in cons.angles.items():
        if cons.haptic.keys() & {i, j, k} or canonical_key((i, k)) in cons.distances:
            continue
        left = basis[max(i, j), min(i, j)], basis[min(i, j), max(i, j)]
        right = basis[max(j, k), min(j, k)], basis[min(j, k), max(j, k)]
        if np.isfinite((*left, *right)).all():
            _put(matrix, (i, k), triangle_distances(left, right, angles))
    return matrix


def _bounds_basis(mol, cons):
    """Return the graph native bounds are read from: the surrogate graph every other seed uses.

    Each dative edge a restrained metal's window owns is removed, and each coordination metal becomes its surrogate
    carbon.
    """
    owned = {tuple(sorted(atoms)) for atoms in cons.distances}
    removable = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        one_metal = (begin in cons.metals) != (end in cons.metals)
        if bond.GetBondType() == Chem.BondType.DATIVE and one_metal and tuple(sorted((begin, end))) in owned:
            removable.append((begin, end))
    if not removable and not any(atom.GetAtomicNum() in COORDINATION_METALS for atom in mol.GetAtoms()):
        return mol
    rw = Chem.RWMol(mol)
    for begin, end in removable:
        rw.RemoveBond(begin, end)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() in COORDINATION_METALS and not atom.GetDegree():
            atom.SetAtomicNum(SURROGATE)
            atom.SetFormalCharge(0)
    basis = rw.GetMol()
    basis.ClearComputedProps()
    basis.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(basis, includeDativeBonds=True)
    return basis


def _write(mol, cons, params=None):
    """Edit native smoothed bounds without re-smoothing the edits; return the filled context.

    The phase order is the algorithm, written only here; each mechanism says what it writes, never when. RELIEVE
    precedes COMMIT so an explicit window always beats a floor relief, and POST follows it so the coplanar bound reads
    committed legs.
    """
    basis = _bounds_basis(mol, cons)
    bm = bounds_matrix(basis, params)
    for (a, b), (_lo, hi) in fragment_contacts(basis, cons, bm).items():  # fragment contacts cap the upper bounds
        bm[a][b] = min(bm[a][b], hi)
    ctx = DGContext(mol, bm, basis=basis)
    for m in MECHANISM_ORDER:
        m.dg_windows(cons, ctx)  # WINDOW   distances, angles, planes -> candidate windows
    for m in MECHANISM_ORDER:
        m.dg_relief(cons, ctx)  # RELIEVE  lower RDKit's phantom floors, before anything is committed
    for (i, j), (lo, hi) in ctx.pairs.items():
        a, b = (i, j) if i < j else (j, i)
        ctx.bm[a][b], ctx.bm[b][a] = hi, lo  # COMMIT
    for m in MECHANISM_ORDER:
        m.dg_post(cons, ctx)  # POST     read the committed matrix (the coplanar 1,4 bound)
    return ctx


def _feasible_bounds(mol, cons, params=None):
    """Build smoothed seed bounds, naming in a log message the pair a repair touched.

    A positive tolerance, a fraction of the upper bound, lets RDKit repair crossed bounds by raising the upper bound to
    the lower; each retry starts from the written matrix. A repair shapes only the seeds, never the constraints the
    relax and gates use. A requested window it moved is named first, since only that is actionable, and only a caller's
    own ``fix`` or ``constrain`` pair warns.
    """
    ctx = _write(mol, cons, params)
    bm, written = ctx.bm, ctx.bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > _MAX_SMOOTH_TOL:
            raise ValueError("triangle smoothing failed: the bounds contradict each other; loosen fix=/constrain=")
        bm[:] = written
    if tol <= 0.0:
        return bm  # no triangle contradiction detected; full embedding and validation are still required
    diff = np.abs(np.asarray(bm, float) - written)
    changed = np.triu(np.maximum(diff, diff.T), 1)  # either bound moving counts, so fold lower onto upper
    requested = [canonical_key((i, j)) for i, j in ctx.pairs]
    named = max(requested, key=lambda p: changed[p], default=None)
    if named is None or changed[named] == 0.0:
        named = tuple(int(x) for x in np.unravel_index(np.argmax(changed), changed.shape))
    requested = {frozenset(pair) for pair in (*cons.fixed, *cons.contacts.distances) if len(pair) == DIST_ATOMS}
    (logger.warning if frozenset(named) in requested else logger.debug)(
        "DG bounds for %s-%s needed smoothing (gap %.2f A); seeding from the repaired bounds",
        *(f"M{atom}" if atom in cons.metals else atom_label(mol, atom) for atom in named),
        float(changed[named]),
    )
    return bm


def _bring_real_confs(mol, work, ids):
    """Copy each embedded conformer's real-atom positions from `work` onto `mol`, ids kept, replacing `mol`'s."""
    mol.RemoveAllConformers()
    n = mol.GetNumAtoms()
    for cid in ids:
        wc = work.GetConformer(int(cid))
        conf = Chem.Conformer(n)
        conf.SetId(int(cid))
        conf.SetPositions(wc.GetPositions()[:n])
        mol.AddConformer(conf, assignId=False)


def _native_params(sampling):
    """Return the RDKit embedding parameters `sampling` selects and the RMS prune threshold to seed with."""
    params = sampling.params
    prune_rms = params.prune_rms
    if prune_rms is None:  # rxembed's own KDG model prunes by default; an unset native object keeps RDKit's off
        prune_rms = -1.0 if params.native is not None else 0.1
    if params.native is not None:
        return params.native, prune_rms
    knowledge = True if params.knowledge is None else params.knowledge
    kdg = embed_parameters(sampling.seed, knowledge=knowledge, threads=sampling.threads, prune_rms=prune_rms)
    return kdg, prune_rms


def _bounds_constraints(cons, params):
    """Return `cons` with the matrix-only ablation switches applied; UFF and publication keep the full payload."""
    if params.coplanar_14 and params.metal_floor_relief:
        return cons
    return replace(
        cons,
        coplanar=cons.coplanar if params.coplanar_14 else (),
        dg_floors=cons.dg_floors if params.metal_floor_relief else {},
    )


def seed_bounds(mol, cons, sampling):
    """Return the smoothed bounds matrix `seed_coordinates` draws `mol` from, haptic centroid rows last."""
    native, _prune_rms = _native_params(sampling)
    return _feasible_bounds(materialise_phantoms(mol, cons.haptic), _bounds_constraints(cons, sampling.params), native)


def seed_coordinates(mol, cons, n, sampling, *, enforce_chirality=True):
    """Return the ids of up to `n` DG seeds drawn from edited bounds onto `mol`, replacing its conformers.

    Seeds with an intact ligand graph come first. An explicit native model keeps its own search. An RDKit timeout raises
    `TimeoutError`.
    """
    work = materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    edits_bounds = bool(cons.distances or cons.angles or cons.planes or cons.coplanar)
    search_count = int(n)
    params = sampling.params
    embed_params = params.native
    seed, threads = sampling.seed, sampling.threads
    p, prune_rms = _native_params(sampling)
    if embed_params is None:
        p.enforceChirality = enforce_chirality
        if cons.seed_cpci:
            p.SetCPCI(dict(cons.seed_cpci))
    multi_fragment = len(Chem.GetMolFrags(work)) > 1
    if embed_params is not None or edits_bounds or multi_fragment:
        # Always replace a supplied object's matrix: it may belong to a previous candidate or helper graph.
        bm = _feasible_bounds(work, _bounds_constraints(cons, params), p)
        p.SetBoundsMat(bm)  # a custom (edited) bounds matrix
    overrides = {
        "randomSeed": seed,
        "numThreads": threads,
        "pruneRmsThresh": prune_rms,
        "clearConfs": True,
        # fragments share one frame: the contact cap keeps every free component in range
        "embedFragmentsSeparately": False,
    }
    previous = {}
    if embed_params is not None:
        previous = {name: getattr(p, name) for name in overrides}
        # native embedding writes these defaults in place, so none may leak across molecules
        previous.update(
            {n: getattr(p, n) for n in ("maxIterations", "basinThresh", "initialEmbeddingMode") if hasattr(p, n)}
        )
    try:
        for name, value in overrides.items():
            setattr(p, name, value)
        ids = _embed_batch(work, search_count, p, fallback=embed_params is None)
    finally:
        for name, value in previous.items():
            setattr(p, name, value)
    # Stereo selection may stop after its first match. Try intact seeds before repairable candidates.
    ids.sort(
        key=lambda cid: (
            bonding_failure(work, cid, clash_tol=0.0, exclude=cons.frozen, constrained=cons.distances) is not None
        )
    )
    if work is not mol:  # discard the phantom: bring only the real-atom coords back onto the real molecule
        _bring_real_confs(mol, work, ids)
    return ids


# (useLegacyImplementation, useRandomCoords) per phase: rxembed's AIO search, then legacy refinement from random
# starts, fast where AIO's starts fail (a Mn H2 complex), then AIO from random starts, which holds where legacy
# refinement cannot (an eta2 alkyne on Os)
_SEED_PHASES = ((False, False), (True, True), (False, True))


def _embed_batch(mol, n, params, *, fallback):
    """Draw a native batch through `_SEED_PHASES`, each later phase drawing only the shortfall.

    Successes stay on one Mol so pruning spans every phase, and each samples with the same seed. An explicit native
    model has no fallback, and a timeout raises without retrying.
    """
    ids = []
    isolated_h = any(a.GetAtomicNum() == 1 and not a.GetDegree() for a in mol.GetAtoms())
    with rdBase.BlockLogs() if isolated_h else nullcontext():
        for phase, (legacy, random) in enumerate(_SEED_PHASES if fallback else _SEED_PHASES[:1]):
            if phase:
                refinement = "legacy" if legacy else "AIO"
                logger.info("DG: kept %d/%d seeds; trying %s refinement from random starts", len(ids), n, refinement)
                params.useLegacyImplementation, params.useRandomCoords, params.clearConfs = legacy, random, False
            count = n - len(ids)
            drawn = list(rdDistGeom.EmbedMultipleConfs(mol, count, params))
            counts = params.GetFailureCounts() if params.trackFailures else ()
            failures = {
                name: counts[int(cause)]
                for name, cause in rdDistGeom.EmbedFailureCauses.names.items()
                if int(cause) < len(counts) and counts[int(cause)]
            }
            logger.debug(
                "DG legacy=%s random=%s knowledge=%s chirality=%s seed=%d attempts=%d: native=%d/%d; failures=%s",
                params.useLegacyImplementation,
                params.useRandomCoords,
                params.useBasicKnowledge,
                params.enforceChirality,
                params.randomSeed,
                params.maxIterations,
                sum(cid >= 0 for cid in drawn),
                count,
                failures if params.trackFailures else "not tracked",
            )
            if any(cid < 0 for cid in drawn):
                raise TimeoutError(
                    f"native RDKit embedding exceeded timeout={params.timeout}s; raise native.timeout or see "
                    "rx.set_verbose('DEBUG')"
                )
            # Shared-frame atoms cannot all sit at the origin; separate monatomic fragments legitimately can.
            lost = {
                cid
                for cid in drawn
                if not params.embedFragmentsSeparately
                and mol.GetNumAtoms() > 1
                and not mol.GetConformer(cid).GetPositions().any()
            }
            if lost:
                first = min(drawn)
                for cid in lost:
                    mol.RemoveConformer(cid)
                drawn = [cid for cid in drawn if cid not in lost]
                for new, cid in enumerate(drawn, start=first):
                    mol.GetConformer(cid).SetId(new)
                drawn = list(range(first, first + len(drawn)))
            ids.extend(drawn)
            if len(ids) == n:
                break
    return ids


def seed_count(mol, constrained=False):
    """Return the DG seed count, scaled by rotatable bonds rather than a flat default.

    A constrained run gets more, since its seeds are frozen in pose and a later rotor search only spreads from them.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(MAX_SEED_COUNT, max(40, 10 * r))
    return min(150, max(24, 6 * r))
