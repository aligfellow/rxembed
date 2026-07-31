"""Distance-geometry embedding, biased by an EDITED bounds matrix.

RDKit's knowledge-derived bounds are edited in place, not replaced, so the experimental-torsion and
basic-knowledge terms survive under a custom core; the matrix is then triangle-smoothed and handed to ETKDGv3.
Pure RDKit, with no third-party dependency.
"""

from __future__ import annotations

import logging

from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from . import mechanisms as _mech
from . import metal_core as _metal
from . import metal_sphere as _sphere

logger = logging.getLogger("rxembed.bounds")  # under the "rxembed" tree `set_verbose` configures

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
_SMOOTH_LOOSE = 0.1  # smoothing beyond this means the constraints are genuinely contradictory, not merely tight


def _smooth(bm, max_tol=0.4):
    """Triangle-smooth ``bm`` in place, escalating the crossover budget until metric; return the tolerance used.

    The returned tolerance is the signal, not a detail: 0.0 means the constraints are mutually realisable,
    and anything above it means some triple contradicts, so RDKit repaired a crossed pair by rewriting
    ``upper := lower``. The repaired pair need not be the over-specified one: it can land on a ligand internal
    RDKit derived correctly, which is why a non-zero tolerance is reported rather than absorbed.
    """
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed")
        bm[:] = back
    if tol > _SMOOTH_LOOSE:  # `_feasible_bounds` owns the user-facing narrative; it calls this more than once
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


def _bounds(mol, cons):
    """Edit RDKit's bounds matrix with every constraint; return ``(matrix, settled tolerance)``.

    The phase order is the algorithm, and this is the only place it is stated; each mechanism
    (`mechanisms.py`) says what it writes, never when. RELIEVE must precede COMMIT so an explicit
    window always beats a floor relief; POST must follow it so the coplanar bound can read committed legs.
    """
    ctx = _mech.DGContext(mol, rdDistGeom.GetMoleculeBoundsMatrix(mol))
    for m in _mech.MECHANISM_ORDER:
        m.dg_windows(cons, ctx)  # WINDOW   distances, angles, planes -> candidate windows
    for m in _mech.MECHANISM_ORDER:
        m.dg_relief(cons, ctx)  # RELIEVE  lower RDKit's phantom floors, before anything is committed
    for (i, j), (lo, hi) in ctx.pairs.items():
        a, b = (i, j) if i < j else (j, i)
        ctx.bm[a][b], ctx.bm[b][a] = hi, lo  # COMMIT
    for m in _mech.MECHANISM_ORDER:
        m.dg_post(cons, ctx)  # POST     read the committed matrix (the coplanar 1,4 bound)
    return ctx.bm, _smooth(ctx.bm)  # SMOOTH; the settled tolerance travels with the matrix


def _feasible_bounds(mol, cons):
    """Build the bounds matrix, falling back to a solved coordination sphere if the raw targets contradict.

    ``_smooth`` returning a non-zero tolerance is the detector: it means no point set satisfies the stated
    constraints, so RDKit repaired a crossed bound to embed at all, and the pair it rewrote need not be the
    over-specified one. When a coordination sphere is present that contradiction is usually its own (the
    distance model, the polytope and the ligand's reach are derived independently), so the sphere is re-solved
    onto a realisable point set and the matrix rebuilt. Every outcome is logged; nothing degrades silently.

    The fallback is one attempt, never a loop, and it is kept only if it strictly improves. Failing to reach
    0.0 is itself the useful signal: the contradiction is then provably not in the coordination sphere.
    """
    bm, tol = _bounds(mol, cons)
    if tol <= 0.0:
        return bm, tol  # mutually realisable, the overwhelmingly common case; say nothing
    if not cons.spheres:
        logger.warning(
            "embed: constraints not mutually realisable (%.0f%% crossover repaired); no sphere to re-centre",
            tol * 100.0,
        )
        return bm, tol
    logger.info(
        "embed: coordination targets not mutually realisable (%.0f%% crossover); re-centring the sphere",
        tol * 100.0,
    )
    solved = _sphere.solve_targets(mol, cons)
    if solved is None:
        logger.warning("embed: the sphere could not be solved; keeping the raw targets at %.0f%%", tol * 100.0)
        return bm, tol
    try:
        bm2, tol2 = _bounds(mol, solved)
    except RuntimeError:  # the solved targets are worse than contradictory: they are infeasible outright
        logger.warning(  # A fallback must never turn a survivable embed into a failure.
            "embed: the solved sphere is infeasible; keeping the raw targets at %.0f%%", tol * 100.0
        )
        return bm, tol
    if tol2 >= tol:
        logger.warning(
            "embed: solving the sphere did not help (%.0f%% -> %.0f%%); keeping the raw targets",
            tol * 100.0,
            tol2 * 100.0,
        )
        return bm, tol
    logger.info("embed: solved sphere is realisable, crossover %.0f%% -> %.0f%%", tol * 100.0, tol2 * 100.0)
    return bm2, tol2


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


def embed(mol, cons, n, seed=DEFAULT_SEED, prune_rms=0.1, knowledge=True, threads=0):
    """Embed ``n`` conformers via ETKDGv3 on the (edited) bounds matrix; return the conformer ids."""
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


def n_confs(mol, constrained=False):
    """ETKDG seed count, scaled by rotatable-bond count rather than a flat default.

    A constrained run gets ~1.6x more, because openconf's pose-frozen search is rotor-only and under-samples
    unless handed more distinct starting points. An unconstrained ``.mc()`` re-seeds and replaces these.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(250, max(40, 10 * r))
    return min(150, max(24, 6 * r))  # cf. openconf max(20, 3*r); a touch more for biased seeds
