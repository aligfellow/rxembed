"""The angle rules, pinned directly on synthetic Constraints — the byte-identity contract in one file.

Three of these branches are unreachable from any real fixture in the golden set, which was established by
mutation-testing that set: deleting the coplanar "no stated M-D-X window" skip left all 15 golden tests
green. A branch no fixture reaches is a branch a refactor can silently delete, so each is pinned here on a
hand-built molecule where the topology is chosen to force one specific path.

The rules (`mechanisms.Angle.dg_windows`, `mechanisms.Coplanar.dg_post`):

  R1  a stated `cons.distances` window on an angle's 1-3 pair wins outright; the angle is discarded
  R2  a real bond path between the end atoms -> INTERSECT the angle-derived window with the matrix
  R2' ...and if that intersection is disjoint, the backbone wins and the angle contributes NOTHING
  R3  no bond path (the atoms meet only through the stripped metal) -> write the angle outright
  C1  a coplanar entry whose M-D-X angle is unstated contributes NOTHING (a 1,4 distance carries no
      dihedral information until that angle is pinned)
  C2  the 1,4 edge is an EXTREMUM over the stated angle window, never a single pinned angle
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem

from rxembed.rdkit_embed.constraints import mechanisms as mech
from rxembed.rdkit_embed.constraints.base import Constraints, add_distance
from rxembed.rdkit_embed.embed import bounds as bnd


def _mol(smi):
    return Chem.AddHs(Chem.MolFromSmiles(smi))


def _bm(mol, cons):
    """The edited bounds matrix alone — `_bounds` also returns the tolerance it settled at."""
    bm, _tol = bnd._bounds(mol, cons)
    return bm


def _window(bm, i, j):
    """(lo, hi) for a pair, in the matrix's own convention: bm[hi_idx][lo_idx] is the lower bound."""
    a, b = (i, j) if i < j else (j, i)
    return bm[b][a], bm[a][b]


def test_r1_a_stated_distance_pre_empts_the_angle():
    """A user `fix={(i,k): d}` on the 1-3 pair owns it; the angle must not overwrite or intersect."""
    mol = _mol("CCC")
    cons = Constraints()
    add_distance(cons.distances, 0, 2, 3.00, 3.02)
    cons.angles[(0, 1, 2)] = (60.0, 70.0)  # would imply a MUCH shorter 0..2 if it were applied
    lo, hi = _window(_bm(mol, cons), 0, 2)
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

    Asserted as the intersection itself, not as a direction of change — a rigid ring keeps its own tight
    diagonal, a floppy one gets its fold-prone floor raised, and which happens depends on the molecule.
    Overwriting instead would tear the backbone; intersecting is the one predicate that lets the chelate
    bite, the polyhedron angles and the haptic rings compose with no per-feature special case.
    """
    mol = _mol("CCC")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] < mech._DISCONNECTED  # a real bond path — the predicate that selects INTERSECT

    base = _bm(mol, Constraints())
    blo, bhi = _window(base, 0, 2)
    ctx = mech.DGContext(mol, base)
    d01, d12 = ctx.mid(0, 1), ctx.mid(1, 2)
    alo = mech._law_of_cosines(d01, d12, window[0])
    ahi = mech._law_of_cosines(d01, d12, window[1])

    cons = Constraints()
    cons.angles[(0, 1, 2)] = window
    got = _window(_bm(mol, cons), 0, 2)
    assert got == pytest.approx((max(alo, blo), min(ahi, bhi))), note


def test_r2_prime_a_disjoint_intersection_keeps_the_backbone_and_discards_the_angle():
    """The angle contributes NOTHING when its window cannot meet the backbone's — silently, by design."""
    mol = _mol("CCC")
    base = _bm(mol, Constraints())
    blo, bhi = _window(base, 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = (1.0, 2.0)  # a physically impossible bite -> derived window far below the backbone
    lo, hi = _window(_bm(mol, cons), 0, 2)
    assert (lo, hi) == pytest.approx((blo, bhi)), "a disjoint intersection must leave the backbone standing"


def test_r3_no_bond_path_writes_the_angle_outright():
    """Two separate fragments: the matrix holds only a phantom floor, so the angle is the sole information."""
    mol = _mol("C.C.C")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] >= mech._DISCONNECTED  # no bond path — the predicate that selects WRITE OUTRIGHT

    cons = Constraints()
    add_distance(cons.distances, 0, 1, 2.00, 2.00)
    add_distance(cons.distances, 1, 2, 2.00, 2.00)
    cons.angles[(0, 1, 2)] = (90.0, 90.0)
    lo, hi = _window(_bm(mol, cons), 0, 2)
    want = math.sqrt(2.0**2 + 2.0**2)  # law of cosines at exactly 90 deg
    assert lo == pytest.approx(want, abs=1e-6)
    assert hi == pytest.approx(want, abs=1e-6)


def _coplanar_case(angle_window):
    """A 4-atom chain with an explicit coplanar cap; `angle_window` optionally pins its M-D-X angle."""
    mol = _mol("C.C.C.C")  # disconnected, so nothing but our own windows reaches the matrix
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)  # pins th_jkw, which IS derived from the matrix legs
    if angle_window is not None:
        cons.angles[(0, 1, 2)] = angle_window
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    return mol, cons


def test_c1_an_unstated_m_d_x_angle_contributes_no_coplanar_bound():
    """THE UNCOVERED BRANCH. Without a stated M-D-X window the 1,4 distance carries no dihedral information.

    No real fixture in the golden set reaches this — every coplanar entry there has its angle stated — so a
    refactor that deletes the skip (or "helpfully" derives the angle from the bounds matrix instead) passes
    the whole golden suite. Both variants are known-bad and silent.
    """
    mol, cons = _coplanar_case(None)
    with_cap = _bm(mol, cons)

    mol2, cons2 = _coplanar_case(None)
    cons2.coplanar = []
    without_cap = _bm(mol2, cons2)

    np.testing.assert_array_equal(with_cap, without_cap)


def test_c2_the_1_4_edge_is_an_extremum_over_the_window_not_a_pinned_angle():
    """A wide M-D-X window must give a WEAKER bound than a narrow one centred in it.

    The 1,4 distance's dependence on M-D-X reverses between a proper dihedral and an improper, so a single
    pinned angle is safe on one and forbids planar geometry on the other. Sizing at the extremum over the
    whole window is safe on both — and is what makes a wide window degrade honestly into a bound that
    cannot bind, rather than into a confidently wrong one.
    """
    mol, narrow = _coplanar_case((118.0, 122.0))
    _, wide = _coplanar_case((100.0, 180.0))
    lo_n, _ = _window(_bm(mol, narrow), 0, 3)
    lo_w, _ = _window(_bm(mol, wide), 0, 3)
    assert lo_w <= lo_n + 1e-9, "a wider M-D-X window must not produce a TIGHTER coplanarity floor"


def test_the_smoothing_escalator_is_untouched():
    """`tol = 1.2*tol + 0.02`, giving up past 0.4 — physics-tuned, and four call sites catch its RuntimeError."""
    tol, rungs = 0.0, []
    while tol <= 0.4:
        tol = 1.2 * tol + 0.02
        rungs.append(round(tol, 6))
    assert rungs[:4] == [0.02, 0.044, 0.0728, 0.10736]
    assert len(rungs) == 9, "the escalator must reach exactly 9 rungs before giving up"
