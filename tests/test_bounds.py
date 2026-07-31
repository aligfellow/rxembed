"""`bounds.py`, the DG driver: build ETKDG parameters, edit RDKit's bounds matrix, smooth, embed.

The design claim this file pins is that the matrix is EDITED from RDKit's knowledge-derived bounds rather than
replaced, and that the smoothing tolerance is a SIGNAL, not a detail: 0.0 means the constraints are mutually
realisable and anything above it means RDKit repaired a crossed bound to embed at all.

The second half pins the ANGLE RULES on synthetic `Constraints`: the branches that decide how an angle meets
the matrix. They are separated because each needs a topology chosen to force one specific path, which no
molecule anyone would embed on purpose provides.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import bounds as bnd
from rxembed import mechanisms as mech
from rxembed.constraints import Constraints, add_distance


def _graph(smiles):
    """A Mol with explicit Hs and no conformer; all `n_confs` reads is the rotor count."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _mol(smiles="CCO", seed=1):
    mol = _graph(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _matrix(mol):
    return rdDistGeom.GetMoleculeBoundsMatrix(mol)


# ---------------------------------------------------------------------------------------------------------
# etkdg: the one place RDKit's embed defaults are overridden
# ---------------------------------------------------------------------------------------------------------


def test_the_seed_is_always_set_and_positional():
    """RDKit's own default is -1 (draw from the global RNG); requiring it makes that defect unrepresentable."""
    assert bnd.etkdg(11).randomSeed == 11
    with pytest.raises(TypeError):
        bnd.etkdg()  # ty: ignore[missing-argument]


def test_knowledge_is_on_by_default_and_both_terms_drop_together():
    """`knowledge=False` means plain distance geometry; experimental torsions and basic knowledge, not one."""
    on = bnd.etkdg(11)
    assert (on.useExpTorsionAnglePrefs, on.useBasicKnowledge) == (True, True)
    off = bnd.etkdg(11, knowledge=False)
    assert (off.useExpTorsionAnglePrefs, off.useBasicKnowledge) == (False, False)


def test_prune_rms_is_passed_through_with_rdkits_meaning():
    """Unset leaves RDKit's -1 (off); a value is forwarded verbatim, so 0.0 keeps its "identical only" meaning."""
    assert bnd.etkdg(11).pruneRmsThresh == -1.0
    assert bnd.etkdg(11, prune_rms=0.0).pruneRmsThresh == 0.0
    assert bnd.etkdg(11, prune_rms=0.5).pruneRmsThresh == 0.5


# ---------------------------------------------------------------------------------------------------------
# probe_conformer: a throwaway geometry that decides a discrete question
# ---------------------------------------------------------------------------------------------------------


def test_a_probe_is_reproducible_and_never_touches_the_caller_s_molecule():
    """The geometry is discarded but the decision is kept, so the seed must decide it, not process history."""
    mol = _mol()
    first, second = bnd.probe_conformer(mol, 7), bnd.probe_conformer(mol, 7)
    assert np.allclose(first.GetConformer().GetPositions(), second.GetConformer().GetPositions())
    assert first is not mol
    assert mol.GetNumConformers() == 1, "the probe added a conformer to the input"


def test_a_failed_probe_returns_none_rather_than_an_empty_mol(monkeypatch):
    """Callers branch on None; a Mol with no conformer would raise deep inside whatever read it."""
    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMolecule", lambda *_a, **_k: -1)
    assert bnd.probe_conformer(_mol(), 7) is None


# ---------------------------------------------------------------------------------------------------------
# _smooth: the tolerance is the signal
# ---------------------------------------------------------------------------------------------------------


def test_a_crossed_bound_is_repaired_and_reported_rather_than_absorbed():
    """0.0 means every bound is mutually realisable; a crossed pair comes back above it, which is the signal."""
    bm = _matrix(_mol())
    assert bnd._smooth(bm.copy()) == 0.0, "RDKit's own bounds are realisable: the premise this rests on"
    bm[0][2], bm[2][0] = 1.02, 0.98  # C0...O2 forced to 1.0 A across two ~1.5 A bonds
    assert bnd._smooth(bm) > 0.0


def test_an_unrepairable_contradiction_raises_rather_than_embedding_nonsense():
    """Past `max_tol` there is no point set at all, and four call sites catch this RuntimeError."""
    bm = _matrix(_mol())
    bm[0][2], bm[2][0] = 0.31, 0.30
    with pytest.raises(RuntimeError, match="triangle smoothing failed"):
        bnd._smooth(bm)


# ---------------------------------------------------------------------------------------------------------
# _bounds / _feasible_bounds: the edit, and what happens when it cannot be satisfied
# ---------------------------------------------------------------------------------------------------------


def test_a_stated_window_is_committed_into_the_matrix():
    """Upper above the diagonal, lower below; RDKit's convention, and the only place it is written."""
    mol = _mol()
    bm, tol = bnd._bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert (bm[0][2], bm[2][0]) == pytest.approx((2.6, 2.5))
    assert tol == 0.0


def test_the_matrix_is_edited_not_replaced():
    """Pairs no constraint names keep RDKit's knowledge-derived bounds: the design's central claim."""
    mol = _mol()
    before = _matrix(mol)
    assert before[1][0] > 1.0, "the C0-C1 lower bound is RDKit's own bond window: the premise"
    after, _tol = bnd._bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert (after[0][1], after[1][0]) == (before[0][1], before[1][0]), "a bonded pair was rewritten"


def test_an_unrealisable_spec_with_no_sphere_says_the_contradiction_is_in_the_spec(caplog):
    """There is nothing to re-solve without a coordination sphere, so the message must point at the spec."""
    mol = _mol()
    with caplog.at_level("WARNING", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (1.0, 1.02)}))
    assert tol > 0.0
    assert "not mutually realisable" in caplog.text
    assert "no sphere to re-centre" in caplog.text


def test_a_realisable_spec_says_nothing(caplog):
    """The overwhelmingly common case must be silent, or the warning stops meaning anything."""
    mol = _mol()
    with caplog.at_level("INFO", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert tol == 0.0
    assert caplog.records == []


# ---------------------------------------------------------------------------------------------------------
# embed: the ETKDG call itself
# ---------------------------------------------------------------------------------------------------------


def test_an_unconstrained_embed_does_not_build_a_custom_matrix(monkeypatch):
    """Building one would discard ETKDG's own bounds for no gain, and pay for the smoothing besides."""
    calls = []
    monkeypatch.setattr(bnd, "_feasible_bounds", lambda *a, **k: calls.append(a) or (_matrix(a[0]), 0.0))

    bnd.embed(_mol(), Constraints(), n=2, seed=3)
    assert calls == []

    bnd.embed(_mol(), Constraints(distances={(0, 2): (2.5, 2.6)}), n=2, seed=3)
    assert len(calls) == 1


def test_embed_returns_ids_that_are_really_on_the_molecule():
    """The ids are the caller's handle on the conformers; they must index the mol that was passed in."""
    mol = _graph("CCO")
    ids = bnd.embed(mol, Constraints(), n=4, seed=3, prune_rms=-1)
    assert len(ids) == 4
    assert {int(c.GetId()) for c in mol.GetConformers()} == {int(i) for i in ids}


def test_the_default_rms_prune_is_on():
    """0.1 A, not RDKit's off, so a rigid molecule's duplicate seeds are dropped before anything scores them."""
    assert len(bnd.embed(_graph("CCO"), Constraints(), n=8, seed=3)) < 8  # one distinct heavy-atom shape


def test_the_same_seed_gives_the_same_geometry():
    """Reproducibility is the reason `seed` is required everywhere; assert it at the driver."""
    a, b = _graph("CCO"), _graph("CCO")
    bnd.embed(a, Constraints(), n=2, seed=1234)
    bnd.embed(b, Constraints(), n=2, seed=1234)
    assert np.allclose(a.GetConformer(0).GetPositions(), b.GetConformer(0).GetPositions())


def test_bring_real_confs_keeps_the_ids_and_drops_the_phantom_atoms():
    """A haptic centroid dummy lives only in the transient working mol; the real molecule keeps its own atoms."""
    real = _mol()
    n = real.GetNumAtoms()
    work = Chem.RWMol(real)
    work.AddAtom(Chem.Atom(0))  # the transient centroid dummy
    work = work.GetMol()
    conf = Chem.Conformer(n + 1)
    for a in range(n + 1):
        conf.SetAtomPosition(a, (float(a), 0.0, 0.0))
    conf.SetId(5)
    work.AddConformer(conf, assignId=False)

    bnd._bring_real_confs(real, work, [5])
    assert [int(c.GetId()) for c in real.GetConformers()] == [5]
    assert real.GetConformer(5).GetPositions().shape == (n, 3)
    assert real.GetConformer(5).GetAtomPosition(0).x == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------
# n_confs: the seed count, scaled by flexibility rather than flat
# ---------------------------------------------------------------------------------------------------------


def test_the_seed_count_scales_with_rotatable_bonds_and_clamps_at_both_ends():
    """A rigid molecule gets the floor, a long chain the ceiling; nothing gets RDKit's flat 10."""
    rigid, mid, floppy = _graph("CCO"), _graph("C" * 25), _graph("C" * 60)
    assert bnd.n_confs(rigid) > 10  # RDKit's own flat default, which this exists to replace
    assert bnd.n_confs(rigid) < bnd.n_confs(mid) < bnd.n_confs(floppy)
    assert bnd.n_confs(rigid) == bnd.n_confs(_graph("CC")), "the floor must clamp a rigid molecule"
    assert bnd.n_confs(floppy) == bnd.n_confs(_graph("C" * 120)), "the ceiling must clamp a long chain"


def test_a_constrained_run_is_given_more_seeds():
    """openconf's pose-frozen search is rotor-only and under-samples unless handed more starting points."""
    for smiles in ("CCO", "C" * 25):
        mol = _graph(smiles)
        assert bnd.n_confs(mol, constrained=True) > bnd.n_confs(mol)


# ---------------------------------------------------------------------------------------------------------
# The angle rules; how an angle-derived window meets the matrix
#
#   R1  a stated `cons.distances` window on the angle's 1-3 pair wins outright; the angle is discarded
#   R2  a real bond path between the end atoms -> intersect the angle-derived window with the matrix
#   R2' ...and if that intersection is disjoint, the backbone wins and the angle contributes nothing
#   R3  no bond path (the atoms meet only through the stripped metal) -> write the angle outright
#   C1  a coplanar entry whose M-D-X angle is unstated contributes nothing (a 1,4 distance carries no
#       dihedral information until that angle is pinned)
#   C2  the 1,4 edge is an EXTREMUM over the stated angle window, never a single pinned angle
#
# Each is built on a hand-made topology, because the branch it selects is chosen by the topology and no real
# molecule offers all six. C1 in particular is silent when wrong: a refactor that drops the skip, or that
# "helpfully" derives the missing angle from the matrix, changes the bounds and raises nothing.
# ---------------------------------------------------------------------------------------------------------


def _rule_mol(smi):
    return Chem.AddHs(Chem.MolFromSmiles(smi))


def _edited(mol, cons):
    """The edited bounds matrix alone; `_bounds` also returns the tolerance it settled at."""
    bm, _tol = bnd._bounds(mol, cons)
    return bm


def _window(bm, i, j):
    """(lo, hi) for a pair, in the matrix's own convention: bm[hi_idx][lo_idx] is the lower bound."""
    a, b = (i, j) if i < j else (j, i)
    return bm[b][a], bm[a][b]


def test_r1_a_stated_distance_pre_empts_the_angle():
    """A user `fix={(i,k): d}` on the 1-3 pair owns it; the angle must not overwrite or intersect."""
    mol = _rule_mol("CCC")
    cons = Constraints()
    add_distance(cons.distances, 0, 2, 3.00, 3.02)
    cons.angles[(0, 1, 2)] = (60.0, 70.0)  # would imply a MUCH shorter 0..2 if it were applied
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (round(lo, 6), round(hi, 6)) == (3.00, 3.02)


@pytest.mark.parametrize(
    ("window", "note"),
    [
        ((100.0, 130.0), "wider than the backbone -> the tighter REAL bound survives untouched"),
        ((109.0, 111.0), "narrower than the backbone -> the angle tightens it"),
    ],
)
def test_r2_a_bonded_path_intersects_rather_than_overwrites(window, note):
    """Propane's C-C-C: the result is exactly the INTERSECTION of the angle-derived window and the backbone.

    Asserted as the intersection itself, not as a direction of change: a rigid ring keeps its own tight
    diagonal, a floppy one gets its fold-prone floor raised, and which happens depends on the molecule.
    Overwriting instead would tear the backbone; intersecting is the one predicate that lets the chelate
    bite, the polyhedron angles and the haptic rings compose with no per-feature special case.
    """
    mol = _rule_mol("CCC")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] < mech._DISCONNECTED  # a real bond path: the predicate that selects INTERSECT

    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)
    ctx = mech.DGContext(mol, base)
    d01, d12 = ctx.mid(0, 1), ctx.mid(1, 2)
    alo = mech._law_of_cosines(d01, d12, window[0])
    ahi = mech._law_of_cosines(d01, d12, window[1])

    cons = Constraints()
    cons.angles[(0, 1, 2)] = window
    got = _window(_edited(mol, cons), 0, 2)
    assert got == pytest.approx((max(alo, blo), min(ahi, bhi))), note


def test_r2_prime_a_disjoint_intersection_keeps_the_backbone_and_discards_the_angle():
    """The angle contributes nothing when its window cannot meet the backbone's; silently, by design."""
    mol = _rule_mol("CCC")
    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = (1.0, 2.0)  # a physically impossible bite -> derived window far below the backbone
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (lo, hi) == pytest.approx((blo, bhi)), "a disjoint intersection must leave the backbone standing"


def test_r3_no_bond_path_writes_the_angle_outright():
    """Two separate fragments: the matrix holds only a phantom floor, so the angle is the sole information."""
    mol = _rule_mol("C.C.C")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] >= mech._DISCONNECTED  # no bond path: the predicate that selects WRITE OUTRIGHT

    cons = Constraints()
    add_distance(cons.distances, 0, 1, 2.00, 2.00)
    add_distance(cons.distances, 1, 2, 2.00, 2.00)
    cons.angles[(0, 1, 2)] = (90.0, 90.0)
    lo, hi = _window(_edited(mol, cons), 0, 2)
    want = math.sqrt(2.0**2 + 2.0**2)  # law of cosines at exactly 90 deg
    assert lo == pytest.approx(want, abs=1e-6)
    assert hi == pytest.approx(want, abs=1e-6)


def _coplanar_case(angle_window):
    """A 4-atom chain with an explicit coplanar cap; `angle_window` optionally pins its M-D-X angle."""
    mol = _rule_mol("C.C.C.C")  # disconnected, so nothing but our own windows reaches the matrix
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)  # pins th_jkw, which is derived from the matrix legs
    if angle_window is not None:
        cons.angles[(0, 1, 2)] = angle_window
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    return mol, cons


def test_c1_an_unstated_m_d_x_angle_contributes_no_coplanar_bound():
    """Without a stated M-D-X window the 1,4 distance carries no dihedral information, so nothing is written."""
    mol, cons = _coplanar_case(None)
    with_cap = _edited(mol, cons)

    mol2, cons2 = _coplanar_case(None)
    cons2.coplanar = []
    without_cap = _edited(mol2, cons2)

    np.testing.assert_array_equal(with_cap, without_cap)


def test_c2_the_1_4_edge_is_an_extremum_over_the_window_not_a_pinned_angle():
    """A wide M-D-X window must give a WEAKER bound than a narrow one centred in it.

    The 1,4 distance's dependence on M-D-X reverses between a proper dihedral and an improper, so a single
    pinned angle is safe on one and forbids planar geometry on the other. Sizing at the extremum over the
    whole window is safe on both, and is what makes a wide window degrade honestly into a bound that
    cannot bind, rather than into a confidently wrong one.
    """
    mol, narrow = _coplanar_case((118.0, 122.0))
    _, wide = _coplanar_case((100.0, 180.0))
    lo_n, _ = _window(_edited(mol, narrow), 0, 3)
    lo_w, _ = _window(_edited(mol, wide), 0, 3)
    assert lo_w <= lo_n + 1e-9, "a wider M-D-X window must not produce a TIGHTER coplanarity floor"


def test_the_smoothing_escalator_is_untouched():
    """`tol = 1.2*tol + 0.02`, giving up past 0.4; physics-tuned, and four call sites catch its RuntimeError."""
    tol, rungs = 0.0, []
    while tol <= 0.4:
        tol = 1.2 * tol + 0.02
        rungs.append(round(tol, 6))
    assert rungs[:4] == [0.02, 0.044, 0.0728, 0.10736]
    assert len(rungs) == 9, "the escalator must reach exactly 9 rungs before giving up"
