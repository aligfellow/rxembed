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


def test_etkdg_states_every_default_it_overrides():
    on = bnd.etkdg(11)
    assert on.randomSeed == 11
    with pytest.raises(TypeError):
        bnd.etkdg()  # ty: ignore[missing-argument]

    assert (on.useExpTorsionAnglePrefs, on.useBasicKnowledge) == (True, True)
    off = bnd.etkdg(11, knowledge=False)
    assert (off.useExpTorsionAnglePrefs, off.useBasicKnowledge) == (False, False)

    assert on.pruneRmsThresh == -1.0
    assert bnd.etkdg(11, prune_rms=0.0).pruneRmsThresh == 0.0
    assert bnd.etkdg(11, prune_rms=0.5).pruneRmsThresh == 0.5


# ---------------------------------------------------------------------------------------------------------
# probe_conformer: a throwaway geometry that decides a discrete question
#
# That its seed decides the answer, rather than process history, is asserted end to end on the one caller
# that consumes it: `tests/test_embed.py`'s encounter-bounds pair.
# ---------------------------------------------------------------------------------------------------------


def test_a_failed_probe_returns_none_rather_than_an_empty_mol(monkeypatch):
    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMolecule", lambda *_a, **_k: -1)
    assert bnd.probe_conformer(_mol(), 7) is None


# ---------------------------------------------------------------------------------------------------------
# _smooth: the tolerance is the signal
# ---------------------------------------------------------------------------------------------------------


def test_an_unrepairable_contradiction_raises_rather_than_embedding_nonsense():
    bm = _matrix(_mol())
    bm[0][2], bm[2][0] = 0.31, 0.30
    with pytest.raises(RuntimeError, match="triangle smoothing failed"):
        bnd._smooth(bm)


# ---------------------------------------------------------------------------------------------------------
# crossings: WHICH window over-determined the matrix
#
# The tolerance says how far smoothing had to give; these say where. Every case below is built so the answer
# is known in advance, because a diagnosis that names a plausible window is indistinguishable from one that
# names the right one.
# ---------------------------------------------------------------------------------------------------------


def test_a_realisable_matrix_names_nothing_and_a_crossed_one_names_its_own_window():
    mol = _mol()
    assert bnd._smooth(_matrix(mol)) == 0.0, "RDKit's own bounds are realisable: the premise this rests on"
    assert bnd.crossings(mol, Constraints(distances={(0, 2): (2.5, 2.6)})) == []

    cons = Constraints(distances={(0, 2): (1.0, 1.02)})  # C0...O2 forced to 1.0 A across two ~1.5 A bonds
    bm = _matrix(mol)
    bm[0][2], bm[2][0] = 1.02, 0.98
    assert bnd._smooth(bm) > 0.0
    crossed = bnd.crossings(mol, cons)
    assert crossed, "smoothing has to widen for this spec; the closure must see the same thing"
    assert [str(w) for w in crossed[0].windows()] == ["distance 0-2"]


def test_widening_the_named_window_by_the_reported_gap_is_what_clears_the_crossing():
    mol = _mol()
    worst = bnd.crossings(mol, Constraints(distances={(0, 2): (1.0, 1.02)}))[0]
    assert bnd._bounds(mol, Constraints(distances={(0, 2): (1.0, 1.02 + worst.gap)}))[1] == 0.0
    assert bnd._bounds(mol, Constraints(distances={(0, 2): (1.0, 1.02 + 0.9 * worst.gap)}))[1] > 0.0


@pytest.mark.parametrize(
    ("metal", "kind"),
    [(1, "D-M-D angle"), (0, "M-D-X fold")],
)
def test_an_angle_window_is_named_by_where_the_metal_sits_in_its_triple(metal, kind):
    mol = _graph("C.C.C.C")
    cons = Constraints(metals={metal})
    for (i, j), d in {(0, 1): 4.0, (1, 2): 4.0, (2, 3): 3.9, (0, 3): 3.9}.items():
        add_distance(cons.distances, i, j, d, d)
    cons.angles[(0, 1, 2)] = (180.0, 180.0)

    worst = bnd.crossings(mol, cons)[0]
    assert worst.pair == (0, 3)
    assert worst.gap == pytest.approx(0.2)
    assert [(w.kind, w.key) for w in worst.pushing] == [(kind, (0, 1, 2))]
    assert {w.key for w in worst.capping} == {(2, 3), (0, 3)}


# ---------------------------------------------------------------------------------------------------------
# _bounds / _feasible_bounds: the edit, and what happens when it cannot be satisfied
# ---------------------------------------------------------------------------------------------------------


def test_the_matrix_is_edited_not_replaced():
    mol = _mol()
    before = _matrix(mol)
    assert before[1][0] > 1.0, "the C0-C1 lower bound is RDKit's own bond window: the premise"

    after, tol = bnd._bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert (after[0][2], after[2][0]) == pytest.approx((2.6, 2.5))
    assert tol == 0.0
    assert (after[0][1], after[1][0]) == (before[0][1], before[1][0]), "a bonded pair was rewritten"


def test_an_unrealisable_spec_says_which_window_it_could_not_satisfy(caplog):
    mol = _mol()
    with caplog.at_level("WARNING", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (1.0, 1.02)}))
    assert tol > 0.0
    assert "not mutually realisable" in caplog.text
    assert "distance 0-2" in caplog.text, "the tolerance alone points nowhere; the window is the point"


def test_a_realisable_spec_says_nothing(caplog):
    mol = _mol()
    with caplog.at_level("INFO", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert tol == 0.0
    assert caplog.records == []


# ---------------------------------------------------------------------------------------------------------
# embed: the ETKDG call itself
# ---------------------------------------------------------------------------------------------------------


def test_an_unconstrained_embed_does_not_build_a_custom_matrix(monkeypatch):
    calls = []
    monkeypatch.setattr(bnd, "_feasible_bounds", lambda *a, **k: calls.append(a) or (_matrix(a[0]), 0.0))

    bnd.embed(_mol(), Constraints(), n=2, seed=3)
    assert calls == []

    bnd.embed(_mol(), Constraints(distances={(0, 2): (2.5, 2.6)}), n=2, seed=3)
    assert len(calls) == 1


def test_embed_returns_reproducible_ids_that_are_really_on_the_molecule():
    mol = _graph("CCO")
    ids = bnd.embed(mol, Constraints(), n=4, seed=3, prune_rms=-1)
    assert len(ids) == 4
    assert {int(c.GetId()) for c in mol.GetConformers()} == {int(i) for i in ids}

    assert len(bnd.embed(_graph("CCO"), Constraints(), n=8, seed=3)) < 8  # one distinct heavy-atom shape

    a, b = _graph("CCO"), _graph("CCO")
    bnd.embed(a, Constraints(), n=2, seed=1234)
    bnd.embed(b, Constraints(), n=2, seed=1234)
    assert np.allclose(a.GetConformer(0).GetPositions(), b.GetConformer(0).GetPositions())


def test_bring_real_confs_keeps_the_ids_and_drops_the_phantom_atoms():
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
    rigid, mid, floppy = _graph("CCO"), _graph("C" * 25), _graph("C" * 60)
    assert bnd.n_confs(rigid) > 10  # RDKit's own flat default, which this exists to replace
    assert bnd.n_confs(rigid) < bnd.n_confs(mid) < bnd.n_confs(floppy)
    assert bnd.n_confs(rigid) == bnd.n_confs(_graph("CC")), "the floor must clamp a rigid molecule"
    assert bnd.n_confs(floppy) == bnd.n_confs(_graph("C" * 120)), "the ceiling must clamp a long chain"
    for mol in (rigid, mid):
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
    mol = _rule_mol("CCC")
    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = (1.0, 2.0)  # a physically impossible bite -> derived window far below the backbone
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (lo, hi) == pytest.approx((blo, bhi)), "a disjoint intersection must leave the backbone standing"


def test_r3_no_bond_path_writes_the_angle_outright():
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
    mol, cons = _coplanar_case(None)
    with_cap = _edited(mol, cons)

    mol2, cons2 = _coplanar_case(None)
    cons2.coplanar = []
    without_cap = _edited(mol2, cons2)

    np.testing.assert_array_equal(with_cap, without_cap)


def test_c2_the_1_4_edge_is_an_extremum_over_the_window_not_a_pinned_angle():
    mol, narrow = _coplanar_case((118.0, 122.0))
    _, wide = _coplanar_case((100.0, 180.0))
    lo_n, _ = _window(_edited(mol, narrow), 0, 3)
    lo_w, _ = _window(_edited(mol, wide), 0, 3)
    assert lo_w <= lo_n + 1e-9, "a wider M-D-X window must not produce a TIGHTER coplanarity floor"
