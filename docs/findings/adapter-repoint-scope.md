# OIN adapter repoint — scope (READ-ONLY findings)

Scoping pass for the `oin_adapter/` repoint after the kernel carve moved
`rxembed.constraints.metal` → `rxembed.rdkit_embed.constraints.metal`. **Nothing was edited or
committed** — this doc only specifies the diff and summarizes prior overnight state.

---

## Job A — the exact OIN adapter repoint

### A.1 The two stale import lines (the only `rxembed.constraints` refs in the package)

```
oin_adapter/reconstruct.py:15   from rxembed.constraints import metal as _metal
oin_adapter/validate.py:15      from rxembed.constraints import metal as M   # noqa: N812
```
Repoint both to the kernel path:

```
oin_adapter/reconstruct.py:15   from rxembed.rdkit_embed.constraints import metal as _metal
oin_adapter/validate.py:15      from rxembed.rdkit_embed.constraints import metal as M   # noqa: N812
```

The old shell `src/rxembed/constraints/` still exists (`__init__.py` re-exports `base`/`builders`,
plus `nci.py`) but **has no `metal` submodule** — `metal.py` moved wholesale. So the current
`from rxembed.constraints import metal` raises `ImportError` (no such submodule); the repoint fixes
that.

### A.2 The shell imports at validate.py:13–14 are STILL VALID — confirmed

`src/rxembed/geometry.py` (21k) and `src/rxembed/metrics.py` (11k) both exist as SHELL modules.
`import rxembed.geometry as geo` and `from rxembed import metrics` resolve, and the attributes the
adapter reads are present: `geo._METAL_Z` ✓, `geo.check` ✓, `metrics.connectivity` ✓,
`metrics.coordination_changed` ✓, `metrics.describe` ✓. No repoint needed for these two lines.

### A.3 The repoint is NECESSARY BUT NOT SUFFICIENT — four symbols the adapter reads have moved/renamed OUT of `metal`

The two import LINES resolve after the repoint (the `metal` module imports cleanly), but four
runtime **attribute** references the adapter makes are no longer on `metal` (or on `geometry`).
These are call-time `AttributeError`s, not import-time — verified by direct `hasattr` probing of the
kernel modules:

| adapter call | site | status on new `metal` | where it lives now / fix |
|---|---|---|---|
| `_metal.prepare(dative_ref)` | reconstruct.py:36 | **GONE** | renamed to **`_metal.surrogate_metal(dative_ref)`** — same 5-tuple return `(base, m, donors, real_z, real_q)`. Rename happened in commit `8337574` ("T14: rename for clarity"), *before* the carve — so the adapter's `prepare` was already stale pre-carve. |
| `_metal._order_label(...)` | reconstruct.py:68 | **GONE** | moved to **`rxembed.isomers._order_label`** (shell module), same signature `(mol, donors, geometry, order)`. |
| `M.ml_distance(...)` | validate.py:110 | **GONE** | moved to **`rxembed.rdkit_embed.constraints.distance.ml_distance`**, same positional signature `(mol, metal, d, real_z, donor_set, ...)`. Only *lazily* imported inside `metal.py` functions, so it is NOT a `metal` module attribute. |
| `geo._coordinating(mol, pos, m)` | validate.py:89 | **GONE from geometry** | moved to **`rxembed.rdkit_embed.coordination._coordinating`**, same signature. Fallback-only path (hit when the mol has no DATIVE M-bonds), so it fails less visibly. |

Everything else the adapter reads off `metal` IS still present with compatible names/signatures:
`Isomer` (constructor still binds the adapter's 9 positional args + `chirality=`/`haptic=` kwargs),
`coordination`, `n_sites`, `VACANT`, `_collapse_haptic`, `strip_phantoms`, `chirality_of`,
`VERTEX_DIRS`, `_APICAL_MIN`, `geometry_for`, `_frag_map`, and the `Isomer.restore()` method
(`iso.restore()`). So a *complete* repoint is 2 import lines + 4 symbol fixes (or add compat
re-exports in `metal.py`/`geometry.py`).

### A.4 Other stale references — none beyond the above

`grep -rn` over the whole package: exactly two `rxembed.constraints` refs (the two above), **zero**
`rxembed.embed` refs. All other `rxembed` references are plain module imports (`import rxembed as rx`,
`import rxembed.geometry as geo`, `from rxembed import metrics`) that resolve.

### A.5 Import-smoke result — OIN-side blocker hits FIRST

`uv run --no-sync python -c "import oin_adapter"` → **fails on the OIN side, not rxembed:**

```
File ".../oin_adapter/parse.py", line 14, in <module>
    from oinsmiles.core.constants import TRANSITION_METALS_NUM
ModuleNotFoundError: No module named 'oinsmiles'
```

`oinsmiles` is not installed here, and `parse.py` (imported first from `__init__.py`) pulls it before
execution ever reaches the stale `rxembed.constraints` import in `reconstruct.py`. So the smoke test
**cannot exercise the rxembed side at all**.

Isolating the rxembed side directly (imported each rxembed piece the adapter needs, then `hasattr`):
`rxembed` / `rxembed.geometry` / `rxembed.metrics` all import; the repointed
`rxembed.rdkit_embed.constraints.metal` imports; and the attribute census produced exactly the A.3
table (`prepare`/`_order_label`/`ml_distance`/`geo._coordinating` = absent, all others present).

**Bottom line for the orchestrator:** the repoint makes every rxembed-side *import statement*
resolve, but the adapter will not *function* until the four moved/renamed symbols in A.3 are
re-pointed too. Running `import oin_adapter` to completion additionally requires `oinsmiles` on the
path (an OIN-side dependency, orthogonal to the rxembed repoint).

---

## Job B — prior overnight state (so later phases don't redo it)

Files in `overnight/`: `embed-module-plan.md`, `maintainability.md`, `constraint-flow.md`,
`baseline_summary.md` + `baseline.tsv`/`baseline.json`, `baseline_head.md` +
`baseline_head.json`, `geometries/`, `geometries_head/`, `run.log`.

### B.1 `embed-module-plan.md` — the restructure plan (PARTIALLY LANDED)

Plan for a shared `rdkit_embed/` embed engine (pure rdkit+numpy, scipy confined to one file) that
rxembed AND OIN both import as the *same code*. Core ideas:
1. **One ordered mechanism registry** both `embed()` (DG) and `relax()` (FF) walk, so a constraint's
   DG half and FF half can't drift (fixes the coplanar-cap split).
2. **Sphere solver** returns as an **off-by-default retry fallback** (bounded `budget_s`/`tries`
   ladder → sphere recenter → clean give-up) so a starved tail never 60-s hangs.
3. Dissolve the `metal.py` 2000-line monolith into `metal/{perception,coordination,tables,isomers,
   polyhedron}` + shared mechanisms; break the `metal ⇄ geometry` cycle via a leaf `fold.py`.

**Route options it names:** the **OIN adapter repoint** (§6 step 6, "Facade + the OIN adapter
re-pointed at the module" — i.e. exactly Job A) and **scipy containment** (scipy in `sphere.py`
only — a "confine", not a scipy→numpy rewrite). It does **not** name an "M-L bonds" or an
`xyz_to_mol` route — those aren't in this plan.

**Already done / superseded:** a large chunk has landed. The kernel carve to `rxembed.rdkit_embed`
happened; `rdkit_embed/constraints/` now holds `distance.py` (ml_distance split out — the plan's
`metal_distance`/`tables` idea), `donor_orient.py` (orientation split — `metal_orient`),
`mechanisms.py` (recent commit `c437d29` "P2: co-locate each field's DG and FF writer" = the plan's
registry §2), and **`sphere.py` + `solver.py`** exist (the fallback §3). Isomer enumeration +
`_order_label` moved to shell `src/rxembed/isomers.py` (plan's `metal/isomers.py`, though it landed
in the shell). `metal.py` is still ~69k/large but distance/orient/mechanisms are carved out. So the
plan is **in progress**, not fresh — treat §4/§6 as a partial checklist, not a to-do from zero.

### B.2 `maintainability.md` — a prior KISS/maintainability pass (EXTEND, don't duplicate; PRE-CARVE, line numbers stale)

Synthesis of six independent reviews at HEAD `ce59f34` (**pre-carve**, so its file/line references —
`metal.py:1720`, `bounds.py:151`, `ff.py:189`, etc. — no longer resolve; the *findings* still hold).
Verdict: **not over-engineered; clean design with a handful of clustered rough edges.** Ranked
findings F1–F17 (the upcoming code-review phase should EXTEND these):
- **F1 (top)** — `Constraints` copied field-by-field in 3 sites with different subsets; the settle
  target has **already silently dropped `coplanar`** (a live bug). Fix: add `Constraints.copy/merge`.
- **F2/F3** — `metal.py` 2042-line monolith of 4 concerns + a `metal ⇄ geometry` import cycle behind
  ~5 lazy imports. Fix: split + hoist the fold census to a leaf. *(Partly addressed by the carve.)*
- **F4–F17** — proven-safe cleanup bundle: `_bounds` has no docstring / undocumented stage ordering
  (F4); `GEOM` is a redundant copy of `GEOM_OPTIONS[n][0]` (F5); `_PHANTOM_FLOOR` name collides with
  the function it doesn't belong to (F8); `_coplanar_donor` has a dead `donor_set` param (F9);
  `contacts` field is a 3-way name collision (F10); duplicated sentinels across modules (F11);
  `_frag_map` written 4× (F12); fold-wall vs conjugated-O M-O-C wall silently contend for one
  `cons.angles` key (F13, medium-risk).
- **§4 DO-NOT-TOUCH (measured load-bearing):** do **not** merge `floors`/`dg_floors` (merge measured
  to degrade 223→220 — they differ in membership, FF skips H+APEX); keep `ANGLES` hand-authored (no
  all-pairs); never reintroduce the charge-keyed M-L contraction (reverted twice); keep the
  `_coplanar_bounds` 12-sample torsion scan; keep the metal↔geometry split itself.

### B.3 `constraint-flow.md` — the constraint-pipeline reference (STALE)

A flow-chart + reference for the whole constraint system: it is **one struct `Constraints` (13
fields)** that every source path fills via a builder (`resolve_core`, nci `auto_binding_modes`, metal
`coordination`/`from_geometry` + orient/coplanar/haptic helpers, `_encounter_bounds`) and every
downstream stage reads (bounds/DG matrix, `restrained_uff`/FF, `geometry.check` gate, mc/minimize/
prune) — "one struct in, one struct out." It's read from the **pre-carve** paths (`constraints/
base.py`, `constraints/metal.py`, `embed/bounds.py`, `refine/ff.py`), so it is **stale**; per
maintainability §S3 its "Tweak 1" (merge `floors`/`dg_floors`) actively points at a **measured
landmine** and should be inverted before anyone follows it.

### B.4 Baselines — two corpus snapshots (for the corpus-validation phase)

A **coordination-permutation census**: for each corpus complex, OIN perceives it → `rx.metal`
permutes isomers → each `rx.embed(iso, stereo='free', n=6).minimize(distance_fc=1e5)`, subprocess-
isolated with a 40 s cap; a `verdict` gates on distance band / unmatched vertex angle / chelate bite
/ overlap / coordination+connectivity change / `geometry.check` (NOT on poly_mae). Two snapshots:

| snapshot | commit | corpus | perms | SANE (verdict ok) | all-sane complexes |
|---|---|---|---|---|---|
| `baseline_summary.md` (+ `.tsv`/`.json`) | "current main" | 154 struct (OIN fixtures + tmQM + rxembed examples) | 292 over 142 complexes | **195/292 = 66.8%** | 89/142 = 62.7% |
| `baseline_head.md` (+ `baseline_head.json`) | clean **`e959f64`** | 144 complexes | 285 | **223/285 = 78%** (the "NO-DEGRADATION REFERENCE") | 106/144 |

`baseline_head` (later, 17 Jul 21:04) is the **no-degradation reference** to hold future work
against; it also reports a_mae-vs-DFT (per-complex best isomer: all 7.16° mean / 4.45° median; oct
4.27°, sq-planar 8.84°). `baseline_summary` traced its misses to causes: (1) a phantom *trans*-chelate
survives the reach-only span gate; (2) a hydride donor reads as a broken molecule after M-bond strip
(all 32 `coordination_changed` rows); (3) many "bite" misses are sub-degree edge grazes; (4) organic
TS examples with no metal are correctly refused by OIN (scope, not defects). Per-permutation geometries
are dumped under `overnight/geometries*/<set>/<code>/<label>.xyz`.
