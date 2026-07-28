"""Distance-geometry embedding, biased by a constraints bounds matrix.

Upstream: RDKit ETKDGv3 + the bounds-matrix edit kernel shared by racerts and
nci_embed. Pure RDKit — this is the piece we'll most likely diverge from racerts on.
"""

from __future__ import annotations

import logging

from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from rxembed.rdkit_embed.constraints import mechanisms as _mech
from rxembed.rdkit_embed.constraints import metal as _metal
from rxembed.rdkit_embed.constraints import solver as _solver

logger = logging.getLogger("rxembed.embed.bounds")  # pinned name: kept under the "rxembed" logger tree
#   (set_verbose configures + the sphere-solver caplog filters on it), unchanged by the carve to rdkit_embed.

DEFAULT_SEED = 0xF00D  # the one embed seed default; a probe may state its own, but never *no* seed
_SMOOTH_LOOSE = 0.1  # smoothing beyond this means the constraints are genuinely over-tight — worth reporting.
#   NB a RELATIVE crossover fraction, not Å: `DoTriangleSmoothing(bm, tol)` repairs a pair whose bounds have
#   crossed by rewriting `upper := lower` iff `(lower - upper) / lower < tol`. Verified by construction — the
#   tol a given crossover needs is invariant under scaling the whole matrix. So `max_tol=0.4` licenses a 40%
#   crossover repair on ANY pair, including ligand internals RDKit derived correctly.


def _smooth(bm, max_tol=0.4):
    """Triangle-smooth ``bm`` in place, escalating the crossover budget until metric; return the tolerance used.

    The returned tolerance is the signal, not a detail: **0.0 means the constraints are mutually realisable**,
    and anything above it means some triple contradicts, so RDKit repaired a crossed pair by rewriting
    ``upper := lower``. The repaired pair need not be the over-specified one — it can land on a ligand internal
    RDKit derived correctly — which is why a non-zero tolerance is reported rather than absorbed.
    """
    back = bm.copy()
    tol = 0.0
    while not DistanceGeometry.DoTriangleSmoothing(bm, tol):
        tol = 1.2 * tol + 0.02
        if tol > max_tol:
            raise RuntimeError("triangle smoothing failed")
        bm[:] = back
    if tol > _SMOOTH_LOOSE:  # `_feasible_bounds` owns the user-facing narrative — it calls this more than once
        logger.debug("smoothing repaired a %.0f%% bound crossover", tol * 100.0)
    return tol


def etkdg(seed, *, knowledge=True, threads=0, prune_rms=None):
    """Build ETKDGv3 parameters — the one place RDKit's embed defaults are overridden.

    ``seed`` is required and positional on purpose. RDKit's own ``randomSeed`` default is -1, meaning *draw
    from the global RNG* — which silently makes every result downstream depend on how much randomness the
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
        p.pruneRmsThresh = prune_rms or -1.0
    return p


def probe_conformer(mol, seed):
    """Embed ONE throwaway conformer on a copy of ``mol``; return it, or None if the embed failed.

    A *probe* is a rough geometry whose only job is to decide a discrete question — which inter-fragment
    heavy-atom pair is closest (`dispatch._encounter_bounds`), which sigma-hole apex or ring radius a contact
    uses (`nci.candidate_contacts`). The conformer is discarded, but the decision it drives is kept, so the
    seed is FIXED rather than varied: what is wanted is reproducibility, not sampling diversity.
    """
    work = Chem.Mol(mol)
    if rdDistGeom.EmbedMolecule(work, etkdg(seed)) != 0:
        return None
    return work


def _bounds(mol, cons):
    """Edit RDKit's bounds matrix with every constraint; return ``(matrix, settled tolerance)``.

    The phase order IS the algorithm, and this is the only place it is stated — each mechanism
    (`constraints/mechanisms.py`) says what it writes, never when. RELIEVE must precede COMMIT so an explicit
    window always beats a floor relief; POST must follow it so the coplanar bound can read committed legs.
    """
    ctx = _mech.DGContext(mol, rdDistGeom.GetMoleculeBoundsMatrix(mol))
    for m in _mech.MECHANISM_ORDER:
        m.dg_windows(cons, ctx)  # WINDOW  — distances, angles, planes -> candidate windows
    for m in _mech.MECHANISM_ORDER:
        m.dg_relief(cons, ctx)  # RELIEVE — lower RDKit's phantom floors, before anything is committed
    for (i, j), (lo, hi) in ctx.pairs.items():
        a, b = (i, j) if i < j else (j, i)
        ctx.bm[a][b], ctx.bm[b][a] = hi, lo  # COMMIT
    for m in _mech.MECHANISM_ORDER:
        m.dg_post(cons, ctx)  # POST    — read the committed matrix (the coplanar 1,4 bound)
    return ctx.bm, _smooth(ctx.bm)  # SMOOTH; the settled tolerance travels with the matrix


def _feasible_bounds(mol, cons):
    """Build the bounds matrix, falling back to a solved coordination sphere if the raw targets contradict.

    ``_smooth`` returning a non-zero tolerance is the detector: it means no point set satisfies the stated
    constraints, so RDKit repaired a crossed bound to embed at all — and the pair it rewrote need not be the
    over-specified one. When a coordination sphere is present that contradiction is usually its own (the
    distance model, the polytope and the ligand's reach are derived independently), so the sphere is re-solved
    onto a realisable point set and the matrix rebuilt. Every outcome is logged; nothing degrades silently.

    The fallback is one attempt, never a loop, and it is kept only if it strictly improves. Failing to reach
    0.0 is itself the useful signal: the contradiction is then provably NOT in the coordination sphere.
    """
    bm, tol = _bounds(mol, cons)
    if tol <= 0.0:
        return bm, tol  # mutually realisable — the overwhelmingly common case; say nothing
    if not cons.spheres:
        logger.warning(
            "embed: constraints are not mutually realisable (%.0f%% bound crossover repaired) and there is no "
            "coordination sphere to re-centre — the contradiction is in the stated spec (a fix / frozen core / "
            "plane), not in a polyhedron",
            tol * 100.0,
        )
        return bm, tol
    logger.info(
        "embed: coordination targets are not mutually realisable (%.0f%% crossover) — re-centring the sphere on "
        "a solved point set (the M-L model, the polytope and the ligand's reach are derived independently, so "
        "they can disagree)",
        tol * 100.0,
    )
    solved = _solver.solve_targets(mol, cons)
    if solved is None:
        logger.warning("embed: the sphere could not be solved — keeping the raw targets at %.0f%%", tol * 100.0)
        return bm, tol
    try:
        bm2, tol2 = _bounds(mol, solved)
    except RuntimeError:  # the solved targets are worse than contradictory — they are infeasible outright.
        logger.warning(  # A fallback must never turn a survivable embed into a failure.
            "embed: the solved sphere is infeasible — keeping the raw targets at %.0f%%", tol * 100.0
        )
        return bm, tol
    if tol2 >= tol:
        logger.warning(
            "embed: solving the sphere did not help (%.0f%% -> %.0f%%) — the contradiction is NOT in the "
            "coordination sphere; check the fix / frozen core / floors. Keeping the raw targets",
            tol * 100.0,
            tol2 * 100.0,
        )
        return bm, tol
    logger.info("embed: solved sphere is realisable — crossover %.0f%% -> %.0f%%", tol * 100.0, tol2 * 100.0)
    return bm2, tol2


def _bring_real_confs(mol, work, ids):
    """Copy each embedded conformer's real-atom positions from `work` (with phantom) onto the real `mol`, same ids.

    The haptic centroid dummy lives only in the transient `work`; the real molecule keeps exactly its own atoms.
    `EmbedMultipleConfs` replaced `work`'s conformers (clearConfs default), so clear `mol`'s stale ones to match —
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
    # Use ETKDG's KNOWLEDGE-based initial coordinates (an eigenvalue start from the bounds matrix, plus the
    # experimental-torsion terms) — they survive a custom bounds matrix, so the periphery is seeded with
    # chemistry. Fall back to random coordinates only if that yields nothing: a tightly constrained core (an
    # organic TS freeze) can make the knowledge-seeded start metric-infeasible, where random coords still embed.
    p.useRandomCoords = False
    ids = list(rdDistGeom.EmbedMultipleConfs(work, n, p))
    if not ids:
        p.useRandomCoords = True
        ids = list(rdDistGeom.EmbedMultipleConfs(work, n, p))
    if work is not mol:  # discard the phantom: bring only the real-atom coords back onto the real molecule
        _bring_real_confs(mol, work, ids)
    return ids


def n_confs(mol, constrained=False):
    """Bounds-biased ETKDG seed count, scaled by flexibility (openconf-style, not a flat 50).

    For an unconstrained ``.mc()`` run openconf re-seeds and these are replaced. For a **constrained** run
    they are the biased pool openconf searches *around* with the held atoms pose-frozen — and that pose-mode
    is **rotor-only** (no low-mode/ring/global moves), so it under-samples unless given more distinct
    starting points: seed ~1.6x more, with a higher floor/cap.
    """
    r = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if constrained:
        return min(250, max(40, 10 * r))
    return min(150, max(24, 6 * r))  # cf. openconf max(20, 3*r); a touch more for biased seeds
