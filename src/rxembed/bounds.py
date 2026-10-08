"""Edit native RDKit bounds and seed coordinates with KDG + AIO.

A supplied EmbedParameters selects another model. SetBoundsMat supplies distances, not the internal
coordinate assumptions that produced them. Triangle smoothing detects metric contradictions, not every
3D infeasibility.
"""

from __future__ import annotations

import itertools
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DistanceGeometry, rdBase
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from .constraints import DIST_ATOMS, metal_distance_tolerance
from .mechanisms import MECHANISM_ORDER, DGContext, triangle_angles, triangle_distances
from .metal_core import COORDINATION_METALS, SURROGATE, materialise_phantoms
from .relax import bonding_failure
from .utils import atom_label

logger = logging.getLogger("rxembed.bounds")  # under the "rxembed" tree `set_verbose` configures

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
MAX_SEED_COUNT = 250  # bound one RDKit DG search; stereo selection reuses this ceiling in small serial batches


def bounds_matrix(mol, params=None, *, set14bounds=True):
    """Build RDKit bounds without leaking its internal UFF-typing diagnostics for likely noisy graphs."""
    noisy = any(atom.GetFormalCharge() or atom.GetAtomicNum() in COORDINATION_METALS for atom in mol.GetAtoms())
    options = {} if params is None else {"embedParams": params}
    if noisy:
        with rdBase.BlockLogs():
            return rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=set14bounds, **options)
    return rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=set14bounds, **options)


def _smooth(bm, max_tol=0.4):
    """Triangle-smooth ``bm`` in place and return the crossover tolerance used.

    Positive tolerance permits RDKit to repair crossed bounds, including ligand internals, by raising the
    upper bound to the lower. It is a fraction of the upper bound, not a distance. Each retry starts from
    the original matrix.
    """
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed: the bounds contradict each other; loosen fix=/constrain=")
        bm[:] = back
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


@dataclass(frozen=True, kw_only=True)
class EmbedParams:
    """Hold what reproduces one embed's seed batches: sampling, the DG model and rxembed's switches.

    `native` selects RDKit's own `EmbedParameters` model in place of rxembed's default KDG + AIO, and is kept
    by reference: RDKit's object cannot be copied or pickled, so do not share it between concurrent embeds.
    `prune_rms` and `knowledge` of ``None`` mean "let the model decide": rxembed's KDG model prunes at 0.1 and
    keeps basic knowledge on; a native object keeps its own setting.
    """

    seed: int = DEFAULT_SEED
    threads: int = 0
    prune_rms: float | None = None
    knowledge: bool | None = None
    native: rdDistGeom.EmbedParameters | None = None
    coplanar_14: bool = True
    metal_floor_relief: bool = True
    donor_orientation: bool = True
    conjugation: bool = True

    def __post_init__(self):
        """Check that every setting has exactly one owner: rxembed, or a supplied native object."""
        for name in ("coplanar_14", "metal_floor_relief", "donor_orientation", "conjugation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be True or False")
        if self.knowledge is not None and not isinstance(self.knowledge, bool):
            raise TypeError("knowledge must be True, False or None")
        if int(self.seed) < 0:
            raise ValueError(f"seed={self.seed}: a negative seed is not reproducible; give a seed >= 0")
        if self.native is None:
            return
        if not isinstance(self.native, rdDistGeom.EmbedParameters):
            raise TypeError("native must be an RDKit EmbedParameters object")
        # rxembed owns sampling; a native object must arrive at RDKit's own defaults for these three fields.
        for rdkit_name, field_name, default in (
            ("randomSeed", "seed", -1),
            ("numThreads", "threads", 1),
            ("pruneRmsThresh", "prune_rms", -1.0),
        ):
            current = getattr(self.native, rdkit_name)
            if current != default:
                raise ValueError(
                    f"native.{rdkit_name}={current} is set; pass EmbedParams({field_name}=...) and leave "
                    f"native.{rdkit_name} at {default}"
                )
        # The DG model belongs to RDKit; knowledge= only sets rxembed's own KDG model.
        if self.knowledge is not None and self.knowledge != self.native.useBasicKnowledge:
            raise ValueError(
                f"native.useBasicKnowledge={self.native.useBasicKnowledge} is set; pass "
                f"knowledge={self.native.useBasicKnowledge} or leave knowledge=None"
            )


def resolve_params(params, seed, threads):
    """Fold a facade's plain ``seed=``/``threads=`` into one `EmbedParams`, or pass one through unchanged."""
    if params is None:
        kwargs = {k: v for k, v in (("seed", seed), ("threads", threads)) if v is not None}
        return EmbedParams(**kwargs)
    if not isinstance(params, EmbedParams):
        raise TypeError(
            f"params= takes EmbedParams, got {type(params).__name__}; wrap an RDKit object as EmbedParams(native=...)"
        )
    if seed is not None or threads is not None:
        raise ValueError("seed=/threads= and params= both set sampling; use dataclasses.replace(params, seed=...)")
    return params


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
    alpha, beta = triangle_angles(ab, bc, ac), triangle_angles(bc, cd, bd)
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

    Native 1-4/1-5 bounds encode conformational preferences, so each three-bond upper limit is projected from
    the unmodified 1-2/1-3 basis instead; a path it cannot project abstains. A ring-closed path keeps RDKit's
    1-4 lower bound, a graph constraint rather than a free torsion (NITWIY), and its 1-4 upper bound only
    across an aromatic bond: RDKit takes any other ring bond's from an sp2-sp2 cis template
    (`_getShareRingBond14Type`), which underestimates a saturated-ring torsion (JOYDIK). The result is a
    model upper bound, not proof that the whole ligand is realisable.
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
        raise ValueError("native ligand reach bounds are inconsistent; check the ligand graph's bonds and charges")
    return reach


def _put(matrix, atoms, window):
    """Narrow one matrix cell to admit ``window``, keyed the way the matrix sorts a pair."""
    a, b = sorted(atoms)
    matrix[b, a] = max(matrix[b, a], window[0])
    matrix[a, b] = min(matrix[a, b], window[1])


_FRAGMENT_CONTACT_SLACK = 1.5  # A: van der Waals contact slack, the same approach an H-bond donor makes to its acceptor
_NATIVE_UNSET = 900.0  # A: below RDKit's raw "no computed bound" default of 1000 A, above any real finite reach


def _fragment_components(mol, cons):
    """Group atoms into real chemical components: RDKit fragments merged by any held distance.

    RDKit's own bond graph fragments every donor arm on a coordinate-free metal (a dative M-L bond lives only
    in `cons.distances`, never as a real bond), so two atoms of one coordination complex can sit in different
    RDKit fragments. Union fragments through any stated distance pair before asking what is genuinely free;
    a real spectator (counterion, free ligand, solvent molecule) is whatever is left disconnected. Return
    `None` when there is nothing to group (one component, or none held together).
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
    """Return ``{(i, j): (floor, ceiling)}`` keeping every free cross-component heavy pair in contact range.

    RDKit leaves a pair with no bonded path at its raw "no information" upper bound, so nothing stops distance
    geometry placing a free component arbitrarily far away. Every cross-component pair gets one ceiling:
    RDKit's own van der Waals floor, plus each side's largest intra-component span, plus contact slack. No
    pair is singled out, so a fragment settles wherever distance geometry puts it, a vacant metal site if
    one fits. This is the one rule for keeping free components together.
    """
    components = _fragment_components(mol, cons)
    if components is None:
        return {}
    component_of = {a: ci for ci, c in enumerate(components) for a in c}
    # A component already pinned some other way (fix=/constrain=) needs no contact ceiling of its own; only
    # skip a pair where BOTH sides are pinned, since forcing one together could fight an intentional separation
    # (e.g. two frozen TS fragments). A pair with one free side still gets the ceiling.
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


def _cap_fragment_contacts(mol, cons, bm):
    """Write `fragment_contacts` as an upper-bound ceiling into the embedding matrix, in place."""
    for (a, b), (_lo, hi) in fragment_contacts(mol, cons, bm).items():
        bm[a][b] = min(bm[a][b], hi)


def coordination_reach_base(mol, reach, metals):
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
    """Intersect compiled targets with native ligand reach into an enumeration outer bound, repairing nothing.

    Nonbonded floors come only from `cons`, and each real metal-distance window is widened by its publication
    tolerance, so the screen cannot reject a boundary publication accepts. Every angle projects from the same
    smoothed side intervals, so dictionary order cannot change a leg. Haptic-centroid rows are omitted: a hard
    centroid equality could turn an accepted ligand distortion into a false exclusion.
    """
    matrix = coordination_reach_base(mol, reach, cons.metals) if native is None else native.copy()
    for atoms, (lo, hi) in cons.distances.items():
        if cons.haptic.keys() & set(atoms):
            continue
        tolerance = metal_distance_tolerance(atoms, cons) if cons.metals.intersection(atoms) else 0.0
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
            _put(matrix, (i, k), triangle_distances(left, right, angles))
    return matrix


def _write(mol, cons, params=None):
    """Edit native smoothed bounds without re-smoothing the edits; return the filled context.

    The phase order is the algorithm, and this is the only place it is stated; each mechanism
    (`mechanisms.py`) says what it writes, never when. RELIEVE must precede COMMIT so an explicit
    window always beats a floor relief; POST must follow it so the coplanar bound can read committed legs.
    """
    # `seed_conformers` temporarily restores the real metal and adds dative M-L edges so RDKit sees a labelled
    # donor's full CIP basis and a terminal sp donor's linear axis. The bounds basis stays the surrogate graph
    # every other seed uses: on a private copy, remove the edges owned by a selected metal's explicit M-L
    # distance and give each metal left bondless its surrogate carbon, whose floors the metal model assumes.
    owned = {tuple(sorted(atoms)) for atoms in cons.distances}
    removable = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DATIVE:
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        metals = [idx for idx in (begin, end) if mol.GetAtomWithIdx(idx).GetAtomicNum() in COORDINATION_METALS]
        if len(metals) == 1 and metals[0] in cons.metals and tuple(sorted((begin, end))) in owned:
            removable.append((begin, end))
    native = mol
    if removable or any(atom.GetAtomicNum() in COORDINATION_METALS for atom in mol.GetAtoms()):
        rw = Chem.RWMol(mol)
        for begin, end in removable:
            rw.RemoveBond(begin, end)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() in COORDINATION_METALS and not atom.GetDegree():  # undo `restore_metal`
                atom.SetAtomicNum(SURROGATE)
                atom.SetFormalCharge(0)
        native = rw.GetMol()
        native.ClearComputedProps()
        native.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(native, includeDativeBonds=True)
    bm = bounds_matrix(native, params)
    _cap_fragment_contacts(native, cons, bm)
    ctx = DGContext(mol, bm, basis=native)
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
    """Build smoothed seed bounds and report the pair implicated in any repair.

    Smoothing may repair the seed matrix, but must not rewrite the Constraints used for relaxation and
    acceptance. A stated window moved by the repair is named first, even where smoothing moved an unstated
    pair further, because only a stated one is actionable; naming every touched pair would need a full
    shortest-path closure, which this function does not compute. The repair only shapes seeds, so it warns
    only when the named pair is the caller's own `fix`/`constrain` distance.
    """
    ctx = _write(mol, cons, params)
    bm, written = ctx.bm, ctx.bm.copy()
    tol = _smooth(bm)
    if tol <= 0.0:
        return bm, tol  # no triangle contradiction detected; full embedding and validation are still required
    diff = np.abs(np.asarray(bm, float) - written)
    changed = np.triu(np.maximum(diff, diff.T), 1)  # either bound moving counts, so fold lower onto upper
    stated = [(min(i, j), max(i, j)) for i, j in ctx.pairs]
    named = max(stated, key=lambda p: changed[p], default=None)
    if named is None or changed[named] == 0.0:
        named = tuple(int(x) for x in np.unravel_index(np.argmax(changed), changed.shape))
    requested = {frozenset(pair) for pair in (*cons.fixed, *cons.contacts[0]) if len(pair) == DIST_ATOMS}
    (logger.warning if frozenset(named) in requested else logger.debug)(
        "DG bounds for %s-%s needed smoothing (gap %.2f A); seeding from the repaired bounds",
        *(f"M{atom}" if atom in cons.metals else atom_label(mol, atom) for atom in named),
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


def seed_coordinates(mol, cons, n, params, *, enforce_chirality=True, max_attempts=0):
    """Generate ``n`` new conformers with native RDKit parameters and edited bounds."""
    work = materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    edits_bounds = bool(cons.distances or cons.angles or cons.planes or cons.coplanar)
    search_count = int(n)
    embed_params = params.native
    seed, threads = params.seed, params.threads
    knowledge = True if params.knowledge is None else params.knowledge
    prune_rms = params.prune_rms
    if prune_rms is None:  # rxembed's own KDG model prunes by default; an unset native object keeps RDKit's off
        prune_rms = -1.0 if embed_params is not None else 0.1
    p = (
        embed_params
        if embed_params is not None
        else embed_parameters(seed, knowledge=knowledge, threads=threads, prune_rms=prune_rms)
    )
    if embed_params is None:
        p.enforceChirality = enforce_chirality
        p.maxIterations = int(max_attempts)
        p.trackFailures = True
    multi_fragment = len(Chem.GetMolFrags(work)) > 1
    if embed_params is not None or edits_bounds or multi_fragment:
        # Always replace a supplied object's matrix: it may belong to a previous candidate or helper graph.
        # These switches affect only our matrix edits. UFF and publication retain the complete Constraints.
        dg = (
            cons
            if params.coplanar_14 and params.metal_floor_relief
            else cons.copy(
                coplanar=cons.coplanar if params.coplanar_14 else [],
                dg_floors=cons.dg_floors if params.metal_floor_relief else {},
            )
        )
        bm, _tol = _feasible_bounds(work, dg, p)
        p.SetBoundsMat(bm)  # a custom (edited) bounds matrix
    overrides = {
        "randomSeed": seed,
        "numThreads": threads,
        "pruneRmsThresh": prune_rms,
        "clearConfs": True,
        # Fragments always share one frame: `_cap_fragment_contacts` (in `_write`) then keeps every free
        # component within contact range, so a separate frame per fragment (RDKit's default) is never needed.
        "embedFragmentsSeparately": False,
    }
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
        with rdBase.BlockLogs() if isolated_h else nullcontext():
            ids = list(rdDistGeom.EmbedMultipleConfs(work, search_count, p))
        # Counts describe rejected native attempts, even in a successful search, not failed conformers.
        counts = p.GetFailureCounts() if p.trackFailures else ()
        failures = {
            name: counts[int(cause)]
            for name, cause in rdDistGeom.EmbedFailureCauses.names.items()
            if int(cause) < len(counts) and counts[int(cause)]
        }
        logger.debug(
            "DG random=%s knowledge=%s chirality=%s legacy=%s: %d/%d conformers; rejected attempts %s%s",
            p.useRandomCoords,
            p.useBasicKnowledge,
            p.enforceChirality,
            p.useLegacyImplementation,
            len(ids),
            search_count,
            failures if p.trackFailures else "not tracked",
            "" if ids else "; random starts: EmbedParams(native=...), useRandomCoords=True",
        )
        if any(cid < 0 for cid in ids):
            raise TimeoutError(
                f"native RDKit embedding exceeded timeout={p.timeout}s; "
                "increase native.timeout or inspect bounds/stereo with rx.set_verbose('DEBUG')"
            )
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


def seed_count(mol, constrained=False):
    """Return the RDKit DG seed count, scaled by rotatable-bond count rather than a flat default.

    A constrained run gets about 1.6x more, because its seeds are pose-frozen and any later rotor search can
    only spread from the distinct starting points it is handed. The multipliers are unmeasured.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(MAX_SEED_COUNT, max(40, 10 * r))
    return min(150, max(24, 6 * r))
