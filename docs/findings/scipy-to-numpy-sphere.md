# scipy → numpy for the coordination-sphere solver?

**Question (from the maintainer).** The kernel's dependency floor is meant to be `numpy + rdkit`.
The sphere solver (`rdkit_embed/constraints/sphere.py`, `solver.py`) uses scipy. Can scipy be
replaced by numpy-only without losing the "rescue" — scipy converging on cases a naive approach
does not?

**Answer in one line.** Keep scipy as a *properly declared* optional extra — **recommendation (b)**.
A hand-rolled numpy LM matches scipy to machine precision on every case that occurs in real
chemistry, but it does **not** faithfully reproduce scipy's convergence on the contradictory
"rescue" cases — and reimplementing it well is *more* code and *less* robustness than a one-line
library call, i.e. negative-KISS. If the true goal is to purge scipy from the closure entirely,
**delete the solver (c)** is more KISS than reimplementing it; it is unreachable on real chemistry
and an ablation already refused it. **(a) replace-with-numpy is rejected** with evidence below.

---

## 1. What exactly does scipy do here?

scipy appears in **exactly one place** — `constraints/sphere.py` — and nowhere else in
`rdkit_embed/` (grep-verified: no `scipy.linalg`, no `scipy.minimize`, no `scipy.spatial`).

| Site | Call | Role |
|---|---|---|
| `sphere.py:64` | `import scipy.optimize` inside `available()` | soft probe: is the extra importable? |
| `sphere.py:189` | `from scipy.optimize import least_squares` inside `solve()` | the guarded import |
| `sphere.py:194` | `least_squares(self.residual, z0, method="trf", max_nfev=800, xtol=1e-10, ftol=1e-10)` | **the only real use** |

`solver.py` touches scipy only indirectly, via `sphere.available()` (`solver.py:42`).

**What it solves.** An *unconstrained* nonlinear least-squares (no `bounds=` is passed, so `trf`
degenerates to a trust-region NLS). Variables `z` = free-donor coordinates (3 dof each) + per rigid
haptic ring a centroid(3) + normal(3). The residual (`SphereSolver.residual`, pure vectorised
numpy) stacks four scaled terms: radial (hit the M–L target), chord (hit the polytope vertex
separation), a softplus ligand-reach hinge, and `|normal| = 1`. Inputs: `z0` from `_solve_start`
(the ideal sphere). Output: `(unpack(sol.x), sol.success)` — the re-centred donor point set.

**Note vs the OIN origin** (`.../wip/sphere_solver.py`): OIN passed an *analytic* Jacobian
(`jac=self.jac`, ~50× faster). rxembed's port dropped it and uses scipy's **numerical** Jacobian —
an explicit decision (`sphere.py:180-187` docstring) because "this path is a rare fallback and a
hand-derived Jacobian is a correctness liability that only pays off in a hot loop." So scipy is
already doing the finite-difference Jacobian; only the trust-region step logic is scipy's.

**How hot is the path?** Ice cold — see §2.

## 2. Is it reachable at all? (nearly dead — confirmed)

`solve_targets` (the sole caller path) runs only when **all** hold:
1. `cons.spheres` is non-empty (a metal coordination sphere exists), and
2. `_smooth` returns `tol > 0` — the raw M–L/polytope/reach targets are non-metric
   (`bounds._feasible_bounds:116-133`), and
3. scipy is importable (`sphere.available()`), else it logs and keeps raw targets.

Evidence it essentially never fires on real chemistry:
- **Project memory** (`rxembed-sphere-solver-port`, verified against current code): *246 of 251*
  smoothing calls across the whole suite settle at exactly `tol = 0.0`; the nonzero ones are
  synthetic test fixtures; no real complex (en-Pd, bis-en Co, tris-en Co, 4-ring chelate, bipy-Pd)
  reaches `tol > 0`. The **angle-intersect rule** absorbs the contradiction *before* it reaches the
  matrix (`rxembed-angle-intersect-composition`).
- **The test file says so in its own docstring** (`tests/test_sphere_solver.py:1-12`): "the solver
  is a FALLBACK... across this suite that never happens on real chemistry — the trigger has to be
  constructed here, or the whole path would ship untested."
- **`tests/test_sphere_solver.py` passes** (11/11 locally, 0.18 s). The scipy-exercising tests
  (`@scipy_only`) construct `SphereSolver` **directly** on synthetic square-plane/tetrahedral
  fixtures. The *only* real-molecule path into scipy is `test_a_haptic_face...` (ferrocene) — and
  that solves a **feasible** problem (rigid rings, consistent targets), not a rescue.
- **The one real molecule that reaches `solve()` (ferrocene) does so on a feasible sphere** — its
  solved Fe–C windows all land at 2.1357 Å (uniform η⁵), which is a repair-to-self, not a rescue.

**Correctness is guarded independently of the solver.** `_feasible_bounds` adopts a solved sphere
**only if it strictly improves** (`tol2 < tol`, `bounds.py:144`), rebuilds the matrix, and on
`RuntimeError`/failure keeps the raw targets (`bounds.py:134-153`). `solve_targets` returns `None`
on non-convergence or missing scipy. So *whatever* `solve()` returns — good basin, bad basin, or
nothing — it can never corrupt an embed; the worst case is "keep raw targets," which is already the
**shipped default** (scipy is optional; see §4). Two tests pin this
(`test_a_worse_solve_is_discarded_not_accepted`, `test_an_infeasible_solve_falls_back...`) and they
are solver-agnostic (they monkeypatch `solve_targets`), so they protect a numpy LM too.

**Conclusion:** replacing or removing scipy here is low-risk *by construction* — the path is
unreachable on real chemistry and its output is firewalled by the `tol2 < tol` guard.

## 3. Can numpy replace it without losing the rescue?

I prototyped two numpy-only Levenberg–Marquardt solvers (LM's adaptive damping *is* the
Gauss-Newton↔gradient-descent trust-region blend that gives the "rescue" on ill-conditioned
Jacobians — a *naive* Gauss-Newton would not, and is the correct thing to reject). Both drive the
**real** `SphereSolver.residual`/`unpack` with the **same `z0`** and are compared head-to-head with
scipy's `least_squares(trf)`. Scripts in the scratchpad (`lm_numpy.py`, `lm_scaled.py`,
`compare.py`, `spectrum.py`, `spectrum2.py`).

**On the regime that actually occurs — feasible / near-feasible targets — numpy is exact:**

| Case | scipy | numpy LM | agreement |
|---|---|---|---|
| satisfiable square-plane (no-op) | radii 2.0 | radii 2.0 | `|Δcoord| = 0` |
| **ferrocene, full `solve_targets` end-to-end** (rings+centroids, real path) | Fe–C 2.1357 ×10 | Fe–C 2.1357 ×10 | **max |Δ| = 4.4e-12** |
| ferrocene, scaled-LM variant | — | — | max |Δ| = 6.8e-11 |

**On the regime that fires the solver — contradictory targets — numpy does NOT match scipy:**
the residual is **non-convex with rotational symmetry and multiple local minima**. Sweeping the
ligand-reach ceiling from feasible into contradiction (square-plane and octahedral):

- The plain LM lands in **worse** basins even on *mild* contradictions (e.g. square-plane
  reach 3.0: scipy cost 12.3 vs LM 44.9) and collapses the sphere (radii ~0.84) on several
  polytopes (trigonal-planar, square-planar, tetrahedral, octahedral).
- The scaled LM (scipy-`x_scale='jac'`-style column scaling) sometimes beats scipy
  (reach 2.83: LM 15.3 vs scipy 21.0) and sometimes loses (reach 3.0 / 2.0), and still fails to
  converge within budget on the largest polytopes (tbp, spy, pbp, square-antiprism → `ok=False`).

Interpretation: on the contradictory cases "matching scipy" is **not even well-defined** — scipy
itself is only a local minimiser there, and its result is a function of its (tuned) trust-region
step logic. A ~40-line hand-rolled LM cannot reproduce `trf`'s basin selection; scipy's `trf` is
~500 lines of variable scaling + trust-region-subproblem machinery. **This is the "scipy rescued
some really slow fails" caveat, made concrete: scipy reliably converges (`ok=True`) on the
ill-conditioned polytope contradictions where the hand-rolled LM stalls (`ok=False`) or collapses.**

Two mitigating facts keep this from mattering in practice: (i) those contradictory inputs do not
occur in real chemistry (§2); (ii) any worse/collapsed numpy solution is caught by the `tol2 < tol`
adoption guard and discarded in favour of the raw targets. So numpy "loses the rescue" *strictly*,
but the loss is confined to cases that never happen and is firewalled where it could.

## 4. Recommendation — (b), with (c) as the alternative; (a) rejected

**Primary: (b) keep scipy as the optional extra — but declare it properly.**

- The solver exists **for** the contradictory regime, and that is exactly where a numpy LM is
  strictly less robust (§3). Swapping a one-line, battle-tested `least_squares` call for 40+ lines
  of hand-tuned LM (damping schedule, gain-ratio, collapse-basin risk) is **negative KISS**: more
  surface to own, less reliability, on a path that is nearly dead anyway.
- scipy being optional **already preserves the numpy+rdkit floor**: without it, `available()` →
  `False`, `solve_targets` → `None`, and the embed keeps raw targets — the shipped default and a
  guarded, tested outcome. Nothing is broken today when scipy is absent.
- **Latent packaging bug to fix regardless:** there is **no `sphere` extra** in
  `pyproject.toml` (only `ase`, `viz`, `all`), and **scipy is not a declared dependency** — it is
  currently importable only because `scikit-learn` drags it in transitively. The error message in
  `sphere.py:191-192` promises `install the 'sphere' extra`, which does not exist. When the kernel
  graduates to standalone `rdkit_embed` (numpy+rdkit, no scikit-learn), scipy vanishes and the
  solver silently no-ops. If the intent is that the solver *works* when asked for, add
  `[project.optional-dependencies] sphere = ["scipy"]` and depend on it explicitly there.

**Alternative: (c) delete the solver — only if the goal is to purge scipy from the closure.**
The evidence supports it: unreachable on real chemistry (§2), an ablation over 144 structures
already found it "does not justify itself" (`test_sphere_solver.py:9-11`), and CLAUDE.md says "dead
machinery gets deleted." Deleting `sphere.py` + `solver.py` + the `@scipy_only` tests + the
`_feasible_bounds` fallback branch removes scipy from the kernel with *less* code than porting it.
Cost: you lose graceful re-centring for the rare synthetic/edge contradiction (today exercised only
by tests, never by real input), and `_feasible_bounds` reverts to "log the crossover, embed on the
smoothed matrix." Given the guard already tolerates a missing solver, this is a clean subtraction.
Between (a) and (c), **(c) is the more KISS way to be scipy-free**: delete beats reimplement.

**Rejected: (a) replace with numpy LM.** It keeps the feature always-on and matches scipy on all
real cases, but it is strictly less robust on the only cases the solver exists for, and it *adds*
maintained numerical-optimizer code rather than removing a dependency line — the worst of both
worlds for a KISS kernel. The collapse-to-0.84-Å basins are currently caught by the `tol2 < tol`
guard, but they are a foot-gun that scipy does not have.

### The KISS/maintainability tradeoff, plainly
- **scipy today** = 1 library call, provably robust, guarded-optional, already no-ops cleanly when
  absent. Its only real cost is being an *undeclared, transitive* dependency — a packaging fix, not
  a code problem.
- **numpy LM** = 40+ lines you own forever, less robust on the rescue cases, buys only "the feature
  works without scipy" for a feature that essentially never fires.
- **delete** = least code of all, backed by the reachability + ablation evidence, at the cost of a
  fallback that only tests use.

The lightest kernel that *keeps* the feature is (b) + declare the extra. The lightest kernel that
*drops* scipy is (c) delete. Reimplementing in numpy (a) is the one option that is heavier than
both.
