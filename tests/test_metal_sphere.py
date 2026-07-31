"""`metal_sphere`: the geometric coordination-sphere solver, and the fallback that reaches it.

The solver is a FALLBACK, not a mechanism: `bounds._feasible_bounds` reaches it only when triangle smoothing
reports the raw targets are non-metric (`tol > 0`), which across this suite is 5 calls in 251, so the trigger
has to be CONSTRUCTED here or the path would ship untested. That rarity is the point: the solver costs fidelity
(it abandons the fitted `ml_distance` numbers for a jointly realisable compromise), and an ablation over 144
structures found it does not justify itself as a routine step.

`sphere` is the one core extra, so the solver tests skip without it while the trigger and degrade tests, the
part of this module a base install really runs; do not. The marker asks the module's own `available()` rather
than importing scipy, so it can never disagree with the guard under test.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed import bounds as bnd
from rxembed import metal_sphere as sphere
from rxembed.constraints import Constraints, add_distance

needs_sphere = pytest.mark.skipif(not sphere.available(), reason="needs rxembed[sphere] (scipy)")


def _contradictory_chain():
    """Four carbons with a MILD (26% crossover) distance contradiction; enough to trip smoothing, not to raise.

    A gross contradiction exhausts the escalator and raises instead, which is a different (correct) outcome.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("C.C.C.C"))
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)
    cons.angles[(0, 1, 2)] = (118.0, 122.0)
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    return mol, cons


def _square_solver(n=4, radius=2.0, ligand_pairs=()):
    """A `SphereSolver` over `n` unit square-planar sites at `radius`, plus its ideal starting point set."""
    u_star = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], float)[:n]
    targets = np.full(n, radius)
    solver = sphere.SphereSolver(n, targets, list(range(n)), [], [[i] for i in range(n)], u_star, list(ligand_pairs))
    return solver, np.array([u * radius for u in u_star])


# --- the trigger: only a real contradiction reaches the solver -------------------------------------------


def test_the_detector_fires_only_on_a_real_contradiction():
    """`_bounds` returns tol 0.0 for a realisable set and RAISES on one the escalator cannot repair."""
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
    with pytest.raises(RuntimeError):
        bnd._bounds(mol, bad)


def test_a_realisable_set_logs_nothing_and_never_constructs_a_solver(caplog):
    """The 98% path stays silent: a fallback that runs routinely is a fidelity cost with no trigger."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cons = Constraints()
    add_distance(cons.distances, 0, 2, 2.3, 2.5)
    with caplog.at_level(logging.INFO, logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, cons)
    assert tol == 0.0
    assert caplog.records == []


def test_a_contradiction_with_no_sphere_is_reported_not_silently_widened(caplog):
    """With no coordination sphere there is nothing to re-centre, so the bounds stand and the log says so."""
    mol, cons = _contradictory_chain()
    with caplog.at_level(logging.WARNING, logger="rxembed.bounds"):
        bm, tol = bnd._feasible_bounds(mol, cons)
    assert tol > 0.0, "this fixture is supposed to be contradictory"
    assert bm is not None
    assert any("no sphere to re-centre" in r.message for r in caplog.records), caplog.text


def test_solve_targets_abstains_without_a_recipe_and_without_scipy(monkeypatch):
    """Nothing to re-derive, or nothing to re-derive it with: the caller keeps the raw targets either way.

    The scipy half is the core's floor, rdkit + numpy, so the fallback is skipped, never fatal.
    """
    assert sphere.solve_targets(None, Constraints()) is None

    monkeypatch.setattr(sphere, "available", lambda: False)
    cons = Constraints(spheres=((0, (1, 2), "square_planar", (0, 1), 46, ()),))
    assert sphere.solve_targets(None, cons) is None


# --- the solver itself --------------------------------------------------------------------------------


@needs_sphere
def test_the_solver_absorbs_an_impossible_reach_without_collapsing_the_sphere():
    """Demanding every donor pair 1.2 Å apart at radius 2.0 is impossible; the radii stay near their target.

    not the triangle inequality: that is a metric axiom, true of any point set in R³, so asserting it cannot
    fail: it would pass while the solver collapsed the sphere to 0.75 Å. The falsifiable statement is that the
    solved points stay near their targets while absorbing the impossible reach.
    """
    n = 4
    pairs = [(i, j, 1.15, 1.25) for i in range(n) for j in range(i + 1, n)]
    solver, ideal = _square_solver(n, 2.0, pairs)
    x, _ok = solver.solve(ideal.ravel())
    assert x.shape == (n, 3)
    assert np.all(np.isfinite(x))
    r = np.linalg.norm(x, axis=1)
    assert np.all(r > 1.0), f"the sphere collapsed: radii {r}"
    assert abs(float(r.mean()) - 2.0) < 0.5, f"radii ran away from the 2.0 Å target: {r}"


@needs_sphere
def test_the_solver_leaves_a_satisfiable_sphere_on_its_targets():
    """With reach that forbids nothing the radii do not move: it is a repair, not a re-optimisation."""
    wide = [(i, j, 0.5, 9.0) for i in range(4) for j in range(i + 1, 4)]
    solver, ideal = _square_solver(4, 2.0, wide)
    x, _ok = solver.solve(ideal.ravel())
    assert np.allclose(np.linalg.norm(x, axis=1), 2.0, atol=1e-3), "radii drifted off an achievable target"


@needs_sphere
def test_every_polytope_states_a_reachable_chord_target():
    """A chord target is compared against UNIT site centroids, so it can never exceed 2.0.

    `POLYHEDRA['tetrahedral'].vertex_dirs` is written (1,1,1), norm √3, so its raw chord target was 2.83
    against a maximum achievable 2.0 and could not be satisfied for any geometry. The template is normalised
    inside the solver rather than in the table, which keeps the readable form for its other consumers.
    """
    from rxembed import metal_polyhedron as p

    for geometry, poly in p.POLYHEDRA.items():
        u = np.array(poly.vertex_dirs, float)
        solver = sphere.SphereSolver(
            len(u), np.full(len(u), 2.0), list(range(len(u))), [], [[i] for i in range(len(u))], u, []
        )
        assert len(solver.chord_t) == 0 or solver.chord_t.max() <= 2.0 + 1e-9, (
            f"{geometry}: unreachable chord target {solver.chord_t.max():.4f}"
        )


@needs_sphere
def test_a_haptic_face_solves_to_one_uniform_distance_with_its_centroid_in_step():
    """A sandwich must solve to one M-C per face, at the η5 target, with the M-centroid window agreeing.

    Two bugs shipped together and hid each other: the solver's row index was keyed on the SITE ordinal rather
    than the atom ROW (so a face's circumradius came out 0 and eight of ten carbons leaked out of the rigid
    n-gon parametrisation), and the M-C targets were derived with the VERTEX list as the donor set (so a Cp
    carbon read as eta-0: a sigma carbanion at 1.956 Å, well inside any "is this a real bond" window, which is
    why a range assertion cannot catch it). Singleton sigma sites make row and site index coincide, which is why
    the whole suite passed; the assertion is therefore the CONVENTION, `ml_distance` over the real ring atoms.
    """
    import rxembed.pipeline as rx
    from rxembed import metal_core as m
    from rxembed import metal_distance as d
    from tests.test_metal_isomers import ferrocene

    iso = next(iter(rx.metal(ferrocene())))
    work = m.materialise_phantoms(iso.mol, iso.cons.haptic)
    solved = sphere.solve_targets(work, iso.cons)
    assert solved is not None

    fe = next(iter(iso.cons.metals))
    ring = {a for atoms in iso.cons.haptic.values() for a in atoms}
    mc = [0.5 * (lo + hi) for (i, j), (lo, hi) in solved.distances.items() if fe in (i, j) and ({i, j} - {fe}) <= ring]
    assert len(mc) == len(ring), "every ring atom must get a solved M-C window"
    assert max(mc) - min(mc) < 0.01, f"η5 equivalents split: {sorted(round(x, 3) for x in mc)}"
    want = d.ml_distance(work, fe, next(iter(ring)), 26, set(ring), d.delocalised_charges(work))
    assert abs(mc[0] - want) < 0.02, f"solved Fe-C {mc[0]:.3f} disagrees with the η5 target {want:.3f}"

    pos = work.GetConformer().GetPositions()
    one_ring = list(next(iter(iso.cons.haptic.values())))
    radius = float(np.linalg.norm(pos[one_ring] - pos[one_ring].mean(0), axis=1).mean())
    for dummy in iso.cons.haptic:
        lo, hi = solved.distances[(min(fe, dummy), max(fe, dummy))]
        # the centroid sits on the face's own axis, so its distance is the leg of that right triangle
        assert abs(0.5 * (lo + hi) - (mc[0] ** 2 - radius**2) ** 0.5) < 0.05, "M-centroid contradicts the M-C windows"


# --- the fallback never makes things worse ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra_pair", "log"),
    [
        ((1, 3, 2.30, 2.50), "did not help"),  # a "solution" no better: a wider window on the same bad pair
        ((0, 3, 9.0, 9.1), "infeasible"),  # far beyond what the escalator can repair
    ],
    ids=["no-improvement", "infeasible"],
)
def test_a_solve_that_does_not_strictly_improve_is_discarded(extra_pair, log, caplog, monkeypatch):
    """The raw targets stand, byte-identically, and the reason is logged: a fallback must never abort an embed.

    `_bounds` RAISES when smoothing cannot repair a matrix at all, so an infeasible solved sphere would
    propagate out of `_feasible_bounds` and kill an embed that was about to succeed on the raw targets.
    """
    mol, cons = _contradictory_chain()
    cons.spheres = ((0, (1, 2), "square_planar", (0, 1), 46, ()),)  # a recipe, so the fallback is entered
    raw_bm, raw_tol = bnd._bounds(mol, cons)
    assert raw_tol > 0.0

    candidate = cons.copy()
    add_distance(candidate.distances, *extra_pair)
    monkeypatch.setattr(sphere, "solve_targets", lambda _mol, _cons: candidate)

    with caplog.at_level(logging.WARNING, logger="rxembed.bounds"):
        bm, tol = bnd._feasible_bounds(mol, cons)  # must not raise
    assert tol == raw_tol
    np.testing.assert_array_equal(bm, raw_bm)
    assert any(log in r.message for r in caplog.records), caplog.text


# --- which faces are solved rigidly ------------------------------------------------------------------------


def _allyl_pdcl():
    """(η³-allyl)PdCl with a seed geometry: an open 3-atom face, so degree-irregular.

    Its centroid sits 0.467 Å from the central carbon and 1.234 Å from each terminal one; the metal is 2.20 Å
    from the terminals against 1.88 Å from the centre. A rigid n-gon would put all three at the mean radius.
    """
    rw = Chem.RWMol()
    pd = rw.AddAtom(Chem.Atom(46))
    rw.GetAtomWithIdx(pd).SetFormalCharge(2)
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(3)]
    rw.AddBond(cs[0], cs[1], Chem.BondType.DOUBLE)
    rw.AddBond(cs[1], cs[2], Chem.BondType.SINGLE)  # a CHAIN, not a cycle: C1 and C3 are not bonded
    rw.GetAtomWithIdx(cs[2]).SetFormalCharge(-1)
    for c in cs:
        rw.AddBond(pd, c, Chem.BondType.DATIVE)
    cl = rw.AddAtom(Chem.Atom(17))
    rw.GetAtomWithIdx(cl).SetFormalCharge(-1)
    rw.AddBond(pd, cl, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in (
        (cs[1], (0.0, 0.0, 0.0)),
        (cs[0], (-1.212, 0.70, 0.0)),
        (cs[2], (1.212, 0.70, 0.0)),
        (pd, (0.0, 0.467, 1.821)),
        (cl, (0.0, -2.30, 2.60)),
    ):
        conf.SetAtomPosition(idx, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


def _solver_partition(mol, monkeypatch):
    """Run `solve_targets` and report how it partitioned the rows: ``(free_rows, ring_row_groups)``.

    Captured at construction, so it asserts the partition rather than a downstream number. Needs scipy:
    `solve_targets` checks `available()` and returns before it builds a solver at all.
    """
    import rxembed.pipeline as rx
    from rxembed import metal_core as m

    seen = {}
    real = sphere.SphereSolver

    class Spy(real):
        def __init__(self, n, targets, free_rows, rings, *args, **kw):
            seen["free"] = list(free_rows)
            seen["rings"] = [list(rows) for rows, _radius in rings]
            super().__init__(n, targets, free_rows, rings, *args, **kw)

    monkeypatch.setattr(sphere, "SphereSolver", Spy)
    iso = next(iter(rx.metal(mol)))
    sphere.solve_targets(m.materialise_phantoms(iso.mol, iso.cons.haptic), iso.cons)
    return seen["free"], seen["rings"]


@needs_sphere
def test_an_irregular_haptic_face_is_solved_as_free_atoms_not_a_rigid_n_gon(monkeypatch):
    """An open allyl gets 3 dof per atom; only a face with one shared radius is fitted as a rigid n-gon.

    The gate was the atom count alone (`len(rows) >= 3`), a proxy for regularity that disagrees with
    `_regular_face` on 9 of the 21 haptic faces in `benchmark/corpus`; on the two η³ allyls (JIWHOQ, NUKHEG)
    the count is the wrong one. The solver is a fallback and neither allyl reaches it, so the defect is latent
    and a test is the only thing that can hold it.
    """
    free, rings = _solver_partition(_allyl_pdcl(), monkeypatch)
    assert rings == [], f"the irregular face was fitted as a rigid n-gon: {rings}"
    assert len(free) == 4, f"3 allyl carbons + 1 chloride should each be free; got {free}"


@needs_sphere
def test_a_regular_haptic_face_is_still_solved_as_one_rigid_n_gon(monkeypatch):
    """The complement: a Cp is regular and keeps the ring form, so the fix cannot be "never fit a ring"."""
    from tests.test_metal_isomers import ferrocene

    free, rings = _solver_partition(ferrocene(), monkeypatch)
    assert free == [], f"a sandwich has no sigma donor, so no free rows; got {free}"
    assert sorted(len(r) for r in rings) == [5, 5], f"both Cp faces must stay rigid n-gons; got {rings}"
