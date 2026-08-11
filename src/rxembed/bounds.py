"""Distance-geometry embedding, biased by an EDITED bounds matrix.

RDKit's knowledge-derived bounds are edited in place, not replaced, so the experimental-torsion and
basic-knowledge terms survive under a custom core; the matrix is then triangle-smoothed and handed to ETKDGv3.
numpy and RDKit only.

Smoothing can report that the edited bounds admit no point set at all, and it reports it as one number with
no address. `crossings` supplies the address: it names the stated windows on the chain that contradicts.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from . import mechanisms as _mech
from . import metal_core as _metal

logger = logging.getLogger("rxembed.bounds")  # under the "rxembed" tree `set_verbose` configures

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
_SMOOTH_LOOSE = 0.1  # smoothing beyond this means the constraints are genuinely contradictory, not merely tight


def _smooth(bm, max_tol=0.4):
    """Triangle-smooth ``bm`` in place, escalating the crossover budget until metric; return the tolerance used.

    The returned tolerance is the signal, not a detail: 0.0 means the constraints are mutually realisable,
    and anything above it means some triple contradicts, so RDKit repaired a crossed pair by rewriting
    ``upper := lower``. The repaired pair need not be the over-specified one: it can land on a ligand internal
    RDKit derived correctly, which is why a non-zero tolerance is reported rather than absorbed. `crossings`
    is the other half: the tolerance says how far, and it says on which pair and from which windows.

    The tolerance is a RATIO of the crossed pair's upper bound, not an Angstrom (measured: a 0.2635 Å crossing
    on a 1.02 Å upper settles at 0.2583). So a rung means the same thing on a bond and on a coordination span.
    """
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed")
        bm[:] = back
    if tol > _SMOOTH_LOOSE:  # `_feasible_bounds` owns the user-facing narrative and names the windows there
        logger.debug("smoothing repaired a %.0f%% bound crossover", tol * 100.0)
    return tol


def etkdg(seed, *, knowledge=True, threads=0, prune_rms=None):
    """Build ETKDGv3 parameters: the one place RDKit's embed defaults are overridden.

    ``seed`` is required and positional on purpose. RDKit's own ``randomSeed`` default is -1, meaning *draw
    from the global RNG*, which silently makes every result downstream depend on how much randomness the
    process happened to consume earlier. Requiring it here makes that defect unrepresentable rather than
    something each call site has to remember (one of three call sites did not).

    ``knowledge=False`` drops the experimental-torsion / basic-knowledge terms for plain distance geometry.
    """
    p = rdDistGeom.ETKDGv3()
    p.randomSeed = int(seed)
    p.numThreads = threads
    if not knowledge:
        p.useExpTorsionAnglePrefs = False
        p.useBasicKnowledge = False
    if prune_rms is not None:
        p.pruneRmsThresh = prune_rms  # passed through with RDKit's meaning: 0.0 prunes only identical, -1 is off
    return p


def probe_conformer(mol, seed):
    """Embed one throwaway conformer on a copy of ``mol``; return it, or None if the embed failed.

    A probe's only job is to decide a discrete question: which inter-fragment pair is closest, which
    sigma-hole apex a contact uses. The geometry is discarded but the decision is kept, so the seed is FIXED:
    reproducibility is wanted here, not sampling diversity.
    """
    work = Chem.Mol(mol)
    if rdDistGeom.EmbedMolecule(work, etkdg(seed)) != 0:
        return None
    return work


def _write(mol, cons):
    """Edit RDKit's bounds matrix with every constraint, unsmoothed; return the filled context.

    The phase order is the algorithm, and this is the only place it is stated; each mechanism
    (`mechanisms.py`) says what it writes, never when. RELIEVE must precede COMMIT so an explicit
    window always beats a floor relief; POST must follow it so the coplanar bound can read committed legs.

    Separate from `_bounds` because smoothing repairs in place: `crossings` needs the matrix as written, and
    by the time a tolerance is reported the crossed bound has already been rewritten away.
    """
    ctx = _mech.DGContext(mol, rdDistGeom.GetMoleculeBoundsMatrix(mol))
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


def _bounds(mol, cons):
    """Edit RDKit's bounds matrix with every constraint; return ``(matrix, settled tolerance)``."""
    bm = _write(mol, cons).bm
    return bm, _smooth(bm)  # SMOOTH; the settled tolerance travels with the matrix


# --- Naming the windows that over-determine the matrix ----------------------------------------------------
#
# A crossed bound is never one window's doing. Smoothing reports that some CHAIN of bounds admits no point
# set, and the pair it repairs need not be one any constraint stated, so naming the culprit is a shortest-path
# question rather than a lookup. `crossings` closes the upper bounds (Floyd-Warshall, keeping the path),
# propagates the lower bounds through that closure once, and reports each pair whose lower now exceeds its
# upper together with the windows on the path that forced it. One propagation pass is enough: the closure
# already satisfies the triangle inequality, so a lower bound raised through it is dominated by the one-step
# statement it came from.

_RDKIT = "rdkit"  # a cell nothing stated: RDKit's own bond / backbone / vdW knowledge
_CROSS_EPS = 1e-9  # Å, floating-point slack: smoothing's own repair leaves exact equalities behind
_NAMED_IN_LINE = 2  # windows quoted in the one-line symptom; the record carries the rest. Two is the line
#   budget: a real crossing reads "0...12 by 0.16 A (11%): coplanar cap 0-59-58-55, M-L distance 0-46" and the
#   caller's own prefix has to fit beside it.


@dataclass(frozen=True)
class Window:
    """One matrix cell and the constraint that wrote it, `kind` naming the family it belongs to.

    ``key`` is that constraint's own key, so a caller can go straight back to `cons.angles[key]`, and
    ``lo``/``hi`` are the window as STATED: degrees for an angle or a dihedral cap, Å for a distance.
    """

    kind: str
    key: tuple
    lo: float
    hi: float

    def __str__(self):
        """Name the window in a log line: its family and the atoms its own constraint key names."""
        return f"{self.kind} {'-'.join(str(a) for a in self.key)}"


@dataclass(frozen=True)
class Crossing:
    """A pair no point set can satisfy: its propagated lower bound exceeds its shortest-path upper bound.

    ``pushing`` are the windows whose LOWER bounds drive the two atoms apart, ``capping`` those whose UPPER
    bounds hold them together. Widening any single one of them, in its own direction, by ``gap`` clears this
    crossing, which is what makes the pair actionable rather than merely reported.
    """

    pair: tuple
    gap: float
    lower: float
    upper: float
    pushing: tuple
    capping: tuple

    @property
    def ratio(self):
        """The gap as a fraction of the upper bound, which is the unit `_smooth`'s own tolerance is stated in.

        The same quantity smoothing reports, from the other end, but not always the same number: RDKit repairs
        in place as it sweeps, so it never sees the whole closure. Measured, the ratio runs from just above
        the settled tolerance (COJKAO 0.108 against 0.107) to well above it (QELZOZ 0.368 against 0.258).
        """
        return self.gap / self.upper if self.upper else float("inf")

    def windows(self):
        """Return the stated windows implicated, pushing first; RDKit's own are excluded (never widen one).

        Deduplicated across the two sides: a window stated with its own lower above its own upper caps and
        pushes the same pair, and naming it twice reads as two culprits.
        """
        return tuple(dict.fromkeys(w for w in (*self.pushing, *self.capping) if w.kind != _RDKIT))

    def __str__(self):
        """State the symptom in one line: the pair, how far it crosses, and the windows that crossed it."""
        named = ", ".join(str(w) for w in self.windows()[:_NAMED_IN_LINE]) or "RDKit's own bounds alone"
        return f"{self.pair[0]}...{self.pair[1]} by {self.gap:.2f} A ({self.ratio * 100:.0f}%): {named}"


def crossings(mol, cons):
    """Name the windows whose combination leaves the bounds matrix with no point set; worst crossing first.

    Empty when the matrix is metric, so this is the detector as well as the diagnosis. Not on the ordinary
    path: it costs a second matrix build plus an O(n³) closure, and only a ``tol > 0`` embed needs it.
    """
    ctx = _write(mol, cons)
    return _crossings(np.asarray(ctx.bm, float), _window_map(cons))


def _crossings(bm, sources):
    """Find the crossed pairs of an unsmoothed matrix, attributing each through `sources` (``{pair: [Window]}``)."""
    upper = np.triu(bm, 1)
    upper = upper + upper.T
    lower = np.tril(bm, -1)
    lower = lower + lower.T
    closed, via = _upper_closure(upper)
    reach, witness = _lower_reach(lower, closed)
    gaps = np.triu(np.maximum(np.maximum(reach, reach.T), lower) - closed, 1)
    out = []
    for a, b in zip(*np.nonzero(gaps > _CROSS_EPS), strict=True):
        i, j = int(a), int(b)
        if lower[i][j] > closed[i][j]:  # the pair's own window is already wider than any path between them
            push, cap = [(i, j)], []
        else:  # ...else a third atom k is held far from one end and close to the other
            near, far = (i, j) if reach[i][j] >= reach[j][i] else (j, i)
            k = int(witness[near][far])
            push, cap = [(near, k)], _path_edges(via, k, far)
        cap += _path_edges(via, i, j)
        out.append(
            Crossing(
                pair=(i, j),
                gap=float(gaps[i][j]),
                lower=float(max(lower[i][j], reach[i][j], reach[j][i])),
                upper=float(closed[i][j]),
                pushing=_attribute(push, sources),
                capping=_attribute(cap, sources),
            )
        )
    return sorted(out, key=lambda c: -c.ratio)  # by the tolerance's own unit, so the head is the pair it reports


def _upper_closure(upper):
    """Shortest-path closure of the upper bounds; return ``(closed, via)``.

    ``via[i][j]`` is the atom the shortest i..j chain of bounds passes through, or -1 where the direct bound
    is already the shortest. The path is what names the windows, so it is reconstructed, not just its length.
    """
    n = len(upper)
    closed = upper.copy()
    via = np.full((n, n), -1, dtype=int)
    for k in range(n):
        cand = closed[:, k, None] + closed[None, k, :]
        better = cand < closed - _CROSS_EPS
        closed = np.where(better, cand, closed)
        via[better] = k
    return closed, via


def _lower_reach(lower, closed):
    """How far each pair is driven apart through a third atom: ``max_k (lower[i][k] - closed[k][j])``.

    Returns that maximum and the atom ``k`` achieving it. Asymmetric by construction (it is atom ``i`` that is
    held away from ``k``), so a caller must read both orientations of a pair.
    """
    n = len(lower)
    best = np.full((n, n), -np.inf)
    witness = np.full((n, n), -1, dtype=int)
    for k in range(n):
        cand = lower[:, k, None] - closed[None, k, :]
        better = cand > best
        best = np.where(better, cand, best)
        witness[better] = k
    return best, witness


def _path_edges(via, i, j):
    """Walk out the direct-bound edges of the shortest upper-bound chain from ``i`` to ``j``."""
    k = int(via[i][j])
    if k < 0:
        return [(i, j)]
    return _path_edges(via, i, k) + _path_edges(via, k, j)


def _attribute(pairs, sources):
    """Name the windows on each pair, once apiece and in order; an unstated cell is RDKit's own."""
    out = []
    for i, j in pairs:
        for w in sources.get((min(i, j), max(i, j))) or [Window(_RDKIT, (min(i, j), max(i, j)), 0.0, 0.0)]:
            if w not in out:
                out.append(w)
    return tuple(out)


def _window_map(cons):
    """``{pair: [Window]}`` for every matrix cell a constraint writes, in the writers' own precedence.

    Mirrors `_write`'s phases rather than re-deriving them: a stated distance owns its pair outright, an angle
    claims only a 1-3 pair no earlier window took, a plane only what is left, and the coplanar cap is a SECOND
    window on a cell the M-D-X fold usually already owns, since it runs in POST and only tightens.
    """
    out = {}
    for (i, j), (lo, hi) in cons.distances.items():
        out[_key(i, j)] = [Window(_distance_kind(cons, i, j), (i, j), lo, hi)]
    for (i, j, k), (lo, hi) in cons.angles.items():
        out.setdefault(_key(i, k), [Window(_angle_kind(cons, i, j, k), (i, j, k), lo, hi)])
    for ring_a, ring_b, sep in cons.planes:
        for a, b in itertools.product(ring_a, ring_b):
            out.setdefault(_key(a, b), [Window("pi-stack", (a, b), sep, sep)])
    # `dg_floors` is deliberately absent: a relief only ever LOWERS a lower bound, so it cannot be what
    # over-constrains, and POST may have overwritten the cell it relieved anyway (COJKAO's Pd...C is relieved
    # to 2.26 Å and then pushed back up by the coplanar cap). Naming it would point a repair at the one
    # statement already giving as much ground as it can.
    for i, j, k, w, anchor, cap in cons.coplanar:
        if _key(i, w) not in cons.distances:  # a stated distance window is truth; the cap never reaches the cell
            out.setdefault(_key(i, w), []).append(Window("coplanar cap", (i, j, k, w), anchor - cap, anchor + cap))
    return out


def _key(i, j):
    """Sort a pair the way the matrix does, which is how every window field keys itself."""
    return (min(i, j), max(i, j))


def _distance_kind(cons, i, j):
    """Which distance family a window belongs to, read off the atoms it names rather than off its source."""
    if i in cons.metals or j in cons.metals:
        return "M-L distance"
    if i in cons.phantoms or j in cons.phantoms:
        return "face radius"
    return "distance"


def _angle_kind(cons, i, j, k):
    """Which angle family a window belongs to: the metal's position in the triple is what distinguishes them."""
    if j in cons.metals:
        return "D-M-D angle"
    if i in cons.metals or k in cons.metals:
        return "M-D-X fold"
    return "angle"


def _feasible_bounds(mol, cons):
    """Build the bounds matrix, and name the windows that contradict when no point set satisfies them all.

    ``_smooth`` returning a non-zero tolerance is the detector: RDKit repaired a crossed bound to embed at
    all, and the pair it rewrote need not be the over-specified one, so the tolerance alone points nowhere.
    `crossings` closes that gap by naming the windows on the chain that crossed.

    Reported, not repaired. A repair was measured over the four structures in 331 that cross
    (`benchmark/crossover.py`): widening the implicated window until the matrix is metric leaves every sphere
    RMSD, M-L MAE and embed outcome where it was, and the one structure whose median moved (UDITUW 0.73 ->
    0.47) scatters 0.30-0.74 across seeds inside every arm alike. PHSNFE, the only one that embeds nothing,
    embeds nothing from a provably metric matrix too. A crossed matrix is a symptom of the windows, not a
    cause of a bad geometry, so this says which window to go and look at and hands the matrix on.
    """
    bm, tol = _bounds(mol, cons)
    if tol <= 0.0:
        return bm, tol  # mutually realisable, the overwhelmingly common case; say nothing
    crossed = crossings(mol, cons)
    if crossed:
        worst = crossed[0]
        named = str(worst.windows()[0]) if worst.windows() else f"atoms {worst.pair[0]}-{worst.pair[1]}"
        logger.warning(
            "ETKDG bounds needed smoothing for %s (gap %.2f A); continuing with the repaired seed bounds "
            "before constrained relaxation; see DEBUG",
            named,
            worst.gap,
        )
        logger.debug("embed: constraint crossing: %s", worst)
    else:  # the closure agrees with smoothing on 350 of 350 calls measured; if it ever does not, say so
        logger.warning("embed: constraints not mutually realisable (%.0f%% repaired), cause unattributed", tol * 100.0)
    return bm, tol


def _bring_real_confs(mol, work, ids):
    """Copy each embedded conformer's real-atom positions from `work` (with phantom) onto the real `mol`, same ids.

    The haptic centroid dummy lives only in the transient `work`; the real molecule keeps exactly its own atoms.
    `EmbedMultipleConfs` replaced `work`'s conformers (clearConfs default), so clear `mol`'s stale ones to match:
    the caller re-adds any retained input geometry afterward, exactly as the no-phantom path relies on.
    """
    mol.RemoveAllConformers()
    n = mol.GetNumAtoms()
    for cid in ids:
        wc = work.GetConformer(int(cid))
        conf = Chem.Conformer(n)
        conf.SetId(int(cid))
        for a in range(n):
            conf.SetAtomPosition(a, wc.GetAtomPosition(a))
        mol.AddConformer(conf, assignId=False)


def seed_coordinates(mol, cons, n, seed=DEFAULT_SEED, prune_rms=0.1, knowledge=True, threads=0):
    """Seed ``n`` conformers with ETKDGv3 using the edited bounds matrix; return their ids."""
    work = _metal.materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    p = etkdg(seed, knowledge=knowledge, threads=threads, prune_rms=prune_rms)
    if cons.distances or cons.angles or cons.planes or cons.coplanar:
        p.embedFragmentsSeparately = False
        bm, _tol = _feasible_bounds(work, cons)
        p.SetBoundsMat(bm)  # a custom (edited) bounds matrix
    # Use ETKDG's knowledge-based start (eigenvalue from the bounds matrix + experimental torsions), which
    # survives a custom matrix. Random coords only as a fallback: a tightly constrained core can make the
    # knowledge-seeded start metric-infeasible where random still embeds.
    p.useRandomCoords = False
    ids = list(rdDistGeom.EmbedMultipleConfs(work, n, p))
    if not ids:
        p.useRandomCoords = True
        ids = list(rdDistGeom.EmbedMultipleConfs(work, n, p))
    if work is not mol:  # discard the phantom: bring only the real-atom coords back onto the real molecule
        _bring_real_confs(mol, work, ids)
    return ids


def seed_count(mol, constrained=False):
    """ETKDG seed count, scaled by rotatable-bond count rather than a flat default.

    A constrained run gets ~1.6x more, because openconf's pose-frozen search is rotor-only and under-samples
    unless handed more distinct starting points. An unconstrained ``.mc()`` re-seeds and replaces these.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(250, max(40, 10 * r))
    return min(150, max(24, 6 * r))  # cf. openconf max(20, 3*r); a touch more for biased seeds
