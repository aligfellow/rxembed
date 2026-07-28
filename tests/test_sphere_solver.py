"""The coordination-sphere solver and the fallback that reaches it.

The solver is a FALLBACK, not a mechanism: `bounds._feasible_bounds` reaches it only when triangle smoothing
reports the raw targets are non-metric (`tol > 0`). Measured across this suite, that never happens on real
chemistry — 246 of 251 smoothing calls settle at exactly 0.0 — so the trigger has to be constructed here, or
the whole path would ship untested.

That rarity is the point. The solver costs fidelity (it abandons the fitted `ml_distance` numbers for a
jointly realisable compromise), which is why it must not be routine; and an ablation over 144 structures
already found it does not justify itself as one. Gating it on a measured contradiction gives both: it cannot
touch the 98% that are already realisable, and it is available where the alternative is a silent bound rewrite.
"""

import logging

import numpy as np
import pytest

from rxembed.rdkit_embed.constraints import sphere
from rxembed.rdkit_embed.constraints.base import Constraints, add_distance
from rxembed.rdkit_embed.embed import bounds as bnd

scipy_only = pytest.mark.skipif(not sphere.available(), reason="the sphere solver needs scipy")


def test_the_detector_fires_only_on_a_real_contradiction():
    """tol == 0.0 for a realisable set; > 0 for a contradictory one. This is the whole trigger."""
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("C.C.C"))

    ok = Constraints()  # a 3-4-5 triangle: perfectly realisable
    add_distance(ok.distances, 0, 1, 3.0, 3.0)
    add_distance(ok.distances, 1, 2, 4.0, 4.0)
    add_distance(ok.distances, 0, 2, 5.0, 5.0)
    assert bnd._bounds(mol, ok)[1] == 0.0

    bad = Constraints()  # violates the triangle inequality outright: 1 + 1 < 5
    add_distance(bad.distances, 0, 1, 1.0, 1.0)
    add_distance(bad.distances, 1, 2, 1.0, 1.0)
    add_distance(bad.distances, 0, 2, 5.0, 5.0)
    with pytest.raises(RuntimeError):  # not even the escalator can repair this one
        bnd._bounds(mol, bad)


def test_no_sphere_to_solve_is_reported_not_silently_widened(caplog):
    """A contradiction with no coordination sphere must say so — nothing can re-centre it."""
    from rdkit import Chem

    # A MILD contradiction — measured at a 26% crossover. It has to be mild: a gross one exhausts the escalator
    # and raises instead, which is a different (and correct) outcome, covered above.
    mol = Chem.AddHs(Chem.MolFromSmiles("C.C.C.C"))
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)
    cons.angles[(0, 1, 2)] = (118.0, 122.0)
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    with caplog.at_level(logging.WARNING, logger="rxembed.embed.bounds"):
        bm, tol = bnd._feasible_bounds(mol, cons)
    assert tol > 0.0, "this fixture is supposed to be contradictory"
    assert bm is not None
    assert any("no coordination sphere to re-centre" in r.message for r in caplog.records), caplog.text


def test_a_realisable_set_logs_nothing_and_does_not_reach_the_solver(caplog):
    """The 98% path must stay silent and must never construct a solver."""
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cons = Constraints()
    add_distance(cons.distances, 0, 2, 2.3, 2.5)
    with caplog.at_level(logging.INFO, logger="rxembed.embed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, cons)
    assert tol == 0.0
    assert caplog.records == []


@scipy_only
def test_the_solver_returns_a_realisable_point_set():
    """Given contradictory targets the solver must return points whose OWN distances are metric.

    That is the feasibility theorem the fallback rests on: a real point set's distance matrix satisfies every
    triangle inequality, so bounds drawn around it smooth at tol 0 by construction.
    """
    n = 4
    targets = np.full(n, 2.0)
    site_rows = [[i] for i in range(n)]
    u_star = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], float)
    # demand every donor pair sit 1.2 A apart — impossible at radius 2.0 in a square plane
    ligand_pairs = [(i, j, 1.15, 1.25) for i in range(n) for j in range(i + 1, n)]
    solver = sphere.SphereSolver(n, targets, list(range(n)), [], site_rows, u_star, ligand_pairs)
    z0 = np.concatenate([u_star[i] * targets[i] for i in range(n)])
    x, _ok = solver.solve(z0)

    assert x.shape == (n, 3)
    assert np.all(np.isfinite(x))
    # NOT the triangle inequality: that is a metric axiom, true of ANY finite point set in R^3, so asserting it
    # cannot fail and would pass while the solver collapsed the sphere to 0.75 A. The falsifiable statement is
    # that the solved points stay near their targets while absorbing the impossible reach.
    r = np.linalg.norm(x, axis=1)
    assert np.all(r > 1.0), f"the sphere collapsed: radii {r}"
    assert abs(float(r.mean()) - 2.0) < 0.5, f"radii ran away from the 2.0 A target: {r}"


@scipy_only
def test_the_solver_keeps_a_satisfiable_sphere_on_its_targets():
    """With targets that ARE realisable the solver must not move them — it is a repair, not a re-optimisation."""
    n = 4
    targets = np.full(n, 2.0)
    u_star = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], float)
    ideal = np.array([u * 2.0 for u in u_star])
    wide = [(i, j, 0.5, 9.0) for i in range(n) for j in range(i + 1, n)]  # reach that forbids nothing
    solver = sphere.SphereSolver(n, targets, list(range(n)), [], [[i] for i in range(n)], u_star, wide)
    x, _ok = solver.solve(ideal.ravel())
    assert np.allclose(np.linalg.norm(x, axis=1), 2.0, atol=1e-3), "radii drifted off an achievable target"


def test_solve_targets_is_a_noop_without_a_recipe():
    """No `spheres` recipe means nothing to re-derive — the caller keeps the raw targets."""
    from rxembed.rdkit_embed.constraints import solver as m

    assert m.solve_targets(None, Constraints()) is None


def test_missing_scipy_degrades_to_the_raw_targets(monkeypatch):
    """The kernel's floor is rdkit + numpy: without scipy the fallback is skipped, not fatal."""
    from rxembed.rdkit_embed.constraints import solver as m

    monkeypatch.setattr(sphere, "available", lambda: False)
    cons = Constraints(spheres=((0, (1, 2), "square_planar", (0, 1), 46, ()),))
    assert m.solve_targets(None, cons) is None


@scipy_only
def test_a_haptic_face_keeps_its_symmetry_and_its_centroid_in_step():
    """A sandwich must solve to ONE uniform M-C distance per face, with the M-centroid window agreeing.

    Regression for two bugs that shipped together and hid each other. The solver's row index was keyed on the
    SITE ordinal rather than the atom ROW, so every ring atom of a face collapsed onto one row: the circumradius
    came out 0 and eight of ten carbons leaked out of the rigid-ring parametrisation into free points — exactly
    the symmetry break the rigid n-gon exists to prevent. Separately the M-C targets were derived with the
    VERTEX list as the donor set, so a Cp carbon read as eta-0 (a sigma carbanion) instead of eta-5. Ferrocene
    emerged with Fe-C from 0.86 to 2.15 A, and an M-centroid window still at its raw value beside them.

    Singleton sigma sites make row and site index coincide, which is why the whole 272-test suite passed.
    """
    import sys

    sys.path.insert(0, "tests")
    import rxembed as rx
    import test_haptic as th
    from rxembed.rdkit_embed.constraints import distance as d
    from rxembed.rdkit_embed.constraints import metal as m
    from rxembed.rdkit_embed.constraints import solver as s

    iso = next(iter(rx.metal(th.ferrocene())))
    work = m.materialise_phantoms(iso.mol, iso.cons.haptic)
    solved = s.solve_targets(work, iso.cons)
    assert solved is not None

    fe = next(iter(iso.cons.metals))
    ring = {a for atoms in iso.cons.haptic.values() for a in atoms}
    mc = [0.5 * (lo + hi) for (i, j), (lo, hi) in solved.distances.items() if fe in (i, j) and ({i, j} - {fe}) <= ring]
    assert len(mc) == len(ring), "every ring atom must get a solved M-C window"
    assert max(mc) - min(mc) < 0.01, f"eta-5 equivalents split: {sorted(round(d, 3) for d in mc)}"
    # Pin the CONVENTION, not a plausible range: the solver must derive its target from the real coordinating
    # atoms and the delocalised charges, exactly as the raw path does. Reading the vertex list instead makes a
    # Cp carbon look like an eta-0 sigma carbanion at 1.956 A — well inside any "is this a real bond" window,
    # which is why a range assertion cannot catch it.
    want = d.ml_distance(work, fe, next(iter(ring)), 26, set(ring), d.delocalised_charges(work))
    assert abs(mc[0] - want) < 0.02, f"solved Fe-C {mc[0]:.3f} disagrees with the eta-5 target {want:.3f}"

    radius = _ring_radius(work, iso.cons.haptic)
    for dummy in iso.cons.haptic:
        lo, hi = solved.distances[(min(fe, dummy), max(fe, dummy))]
        want = (mc[0] ** 2 - radius**2) ** 0.5  # the centroid sits on the face's own axis
        assert abs(0.5 * (lo + hi) - want) < 0.05, "the M-centroid window contradicts the M-C windows"


def _ring_radius(mol, haptic):
    import numpy as np

    ring = next(iter(haptic.values()))
    pos = mol.GetConformer().GetPositions()
    return float(np.linalg.norm(pos[list(ring)] - pos[list(ring)].mean(0), axis=1).mean())


@scipy_only
def test_every_polytope_states_a_reachable_chord_target():
    """A chord target is compared against UNIT site centroids, so it can never exceed 2.0.

    `POLYHEDRA['tetrahedral'].vertex_dirs` is written (1,1,1) — norm sqrt(3) — so its raw chord target was 2.83
    against a maximum achievable 2.0, and the term could not be satisfied for any geometry. The template is normalised
    inside the solver rather than in the table, which keeps the readable form for its other consumers.
    """
    import numpy as np

    from rxembed.rdkit_embed.constraints import metal as m

    for geometry, poly in m.POLYHEDRA.items():
        u = np.array(poly.vertex_dirs, float)
        solver = sphere.SphereSolver(
            len(u), np.full(len(u), 2.0), list(range(len(u))), [], [[i] for i in range(len(u))], u, []
        )
        assert len(solver.chord_t) == 0 or solver.chord_t.max() <= 2.0 + 1e-9, (
            f"{geometry}: unreachable chord target {solver.chord_t.max():.4f}"
        )


def test_a_worse_solve_is_discarded_not_accepted(caplog, monkeypatch):
    """The fallback keeps a solved sphere only if it strictly improves — otherwise the raw targets stand."""
    from rdkit import Chem

    from rxembed.rdkit_embed.constraints import solver as m

    mol = Chem.AddHs(Chem.MolFromSmiles("C.C.C.C"))
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)
    cons.angles[(0, 1, 2)] = (118.0, 122.0)
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    cons.spheres = ((0, (1, 2), "square_planar", (0, 1), 46, ()),)  # a recipe, so the fallback is entered

    raw_bm, raw_tol = bnd._bounds(mol, cons)
    assert raw_tol > 0.0

    worse = cons.copy()  # a "solution" that is no better — a wider window on the same contradictory pair
    add_distance(worse.distances, 1, 3, 2.30, 2.50)
    monkeypatch.setattr(m, "solve_targets", lambda _mol, _cons: worse)

    with caplog.at_level(logging.WARNING, logger="rxembed.embed.bounds"):
        bm, tol = bnd._feasible_bounds(mol, cons)
    assert tol == raw_tol, "a solve that does not improve must not be adopted"
    np.testing.assert_array_equal(bm, raw_bm)
    assert any("did not help" in r.message for r in caplog.records), caplog.text


def test_an_infeasible_solve_falls_back_instead_of_killing_the_embed(caplog, monkeypatch):
    """A fallback must never turn a survivable embed into a failure.

    `_bounds` RAISES when smoothing cannot repair a matrix at all, so a solved sphere that lands there would
    propagate out of `_feasible_bounds` and abort an embed that was about to succeed on the raw targets.
    """
    from rdkit import Chem

    from rxembed.rdkit_embed.constraints import solver as m

    mol = Chem.AddHs(Chem.MolFromSmiles("C.C.C.C"))
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)
    cons.angles[(0, 1, 2)] = (118.0, 122.0)
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    cons.spheres = ((0, (1, 2), "square_planar", (0, 1), 46, ()),)
    raw_bm, raw_tol = bnd._bounds(mol, cons)

    doomed = cons.copy()  # far beyond what the escalator can repair
    add_distance(doomed.distances, 0, 3, 9.0, 9.1)
    monkeypatch.setattr(m, "solve_targets", lambda _mol, _cons: doomed)

    with caplog.at_level(logging.WARNING, logger="rxembed.embed.bounds"):
        bm, tol = bnd._feasible_bounds(mol, cons)  # must NOT raise
    assert tol == raw_tol
    np.testing.assert_array_equal(bm, raw_bm)
    assert any("infeasible" in r.message for r in caplog.records), caplog.text
