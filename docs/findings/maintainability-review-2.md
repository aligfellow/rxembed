# Maintainability review v2 — post-carve (2026-07-23)

A 4-lens read-only review (over-abstraction/KISS · onboarding/readability · recent-changes clarity ·
dead-code/cruft), extending the prior `overnight/maintainability.md` (F1–F17). Per-lens detail:
`docs/findings/review-{kiss,onboarding,recent-changes,cruft}.md`.

## Verdict

**The codebase is NOT over-abstracted, and the kernel carve *improved* the KISS story.** All four lenses
independently agree: the `Mechanism` phase-hook pattern earns its keep (it co-locates the DG+FF writers that
used to drift), the old monolith (prior F2, 2042-line metal.py) is split, and the copy-footgun (prior F1) is
now compiler-guarded (`base.py` registry-driven `copy()`/`compose()` + import-time field-sync assert). There
is **no high-severity over-abstraction** and **no function-level dead code** (ruff F401/F811/F841 clean; the
`.mol`/`_mol` split is correct and locked by `test_ml_connected.py`).

**The real debt is doc-drift + a handful of tiny cleanups.** The code got cleaner and *moved* (the carve), but
the map didn't follow: `CLAUDE.md` "Where things live" is stale in the same way per **all four lenses**. That
is the single highest-leverage fix.

## SAFE-FIX batch (behaviour-preserving, golden-bit-identical; applied this session)

Docs / signage:
- **CLAUDE.md "Where things live" + stale prose** (`:26,:65,:124,:130,:146-168`) — add the `rxembed.rdkit_embed`
  kernel subpackage; fix rows crediting `bounds.py`/`base.py`/`metal.py`/`builders.py` to old paths; list
  `inputs.py`, `isomers.py`; attribute donor orientation to `donor_orient.py`; fix the gitignored `plan.md`
  pointer. *(all 4 lenses)*
- **kernel `README.md`** — drop "Still to do: always carry the M-L bonds" (landed `521505d`); reword the
  "drop scipy → numpy" item to reflect the measured decision (numpy-replace rejected).
- Docstrings: `prune`/`representatives`/`cluster` document the implicit `minimize()` (moves atoms); `embed()`
  documents its return types incl. the bare `list` for `stereo='separate'`; `.mol` documents the
  finalize-on-access cost; the Mutation-contract docstring adds `filter`/`score`/`optimize`.

Code (each re-verified behaviour-identical by the golden + full suite gate):
- Delete redundant `GEOM` dict (`metal.py:80`, a copy of `GEOM_OPTIONS[n][0]`; inline at the one read site).
- Drop the dead `donor_set` param of `_coplanar_donor` (`donor_orient.py:301`) + its call site.
- De-duplicate `_DISCONNECTED = 1e6` (`mechanisms.py` + `metal.py`) to one definition.
- Remove the dead `n_sites` ANGLES-fallback branch (`metal.py`, never runs) — *only if re-verified dead*.
- Pin `rdkit_embed/refine/ff.py` logger to the `"rxembed.*"` convention its 4 siblings use + comment the WHY
  (graduation-safety), so it doesn't go dark under `set_verbose` once the kernel graduates.
- Rename the `mc` metal-context local in `pipeline.py` → `metal_ctx` (collides with `.mc()` / the `_mc` module;
  `dispatch.py` already uses `metal_ctx`).
- **scipy message** (`sphere.py:192`, high): declare `sphere = ["scipy"]` in the ROOT `pyproject.toml` so the
  "install the 'sphere' extra" message is true (today the extra exists only in the inert placeholder; scipy is
  only transitive via scikit-learn). Decision-independent; correct whether the solver is later kept or deleted.
- Add the missing test locking `inputs.py` imports nothing from `rxembed` (the cycle guard its kernel twin has).

## JUDGMENT-CALLS (surfaced for the maintainer — NOT applied)

1. **Move the coordination *builders* out of `metal.py`** (`coordination`/`coordinate`/`coordination_from_geometry`
   /`_centroid_constraints`) into a consumer-layer module — they consume distance/donor_orient and force 5 lazy
   in-function imports to dodge the cycle. Cleaner deps, but a structural move (you rejected over-splitting before).
2. **Collapse the 5 index-aligned polyhedron tables** (`GEOM_OPTIONS`/`ANGLES`/`PERMUTATIONS`/`VERTEX_DIRS`/…)
   into one keyed `Polyhedron` record — adding a geometry is up to 5 hand-synced edits today. Keep ANGLES
   hand-authored (all-pairs refuted).
3. **Sphere solver keep-vs-delete** (`sphere.py`+`solver.py`, ~340 lines, scipy-gated ~2% fallback, effectively
   unreachable on the base install) — the clearest "generic machinery, almost-never caller". Ties to the scipy
   keep/delete decision (numpy-replace already refuted). See `docs/findings/scipy-to-numpy-sphere.md`.

## Not re-chased (already resolved / measured-refuted)
Prior F1 (copy footgun), F2 (metal.py monolith), F13 (`_CONJ_O_MDC_ANGLE`) — fixed. The §4 DO-NOT-TOUCH list
(merge floors/dg_floors; all-pairs ANGLES; charge-keyed contraction; delete donor caps) — untouched, honored.
