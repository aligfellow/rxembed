# Separability assessment: is the embed kernel droppable into OIN-SMILES?

> **CORRECTION (verified post-assessment):** the claim below that "T7's decomposition is not merged here"
> is WRONG — T7 **is** merged (commit `c45e557`; `_number_shared_labels`/`_distinct_orderings` present,
> no CC≥15 in `metal.py`). The *module-concentration* point still holds: T7 decomposed functions (per-function
> CC) but was +169 lines net, so `metal.py`/`pipeline.py`/`geometry.py` remain large — that's what the T8–T13
> splits address, not T7.

Read-only architecture assessment, current branch `rdkit-embed-kernel`, measured 2026-07-22 against the
working tree (not the plan's numbers — those predate this session and are stale: `metal.py` is 2397 lines,
not the plan's "~1798 after T8/T9"; `pipeline.py` 1594 not 1436).

Method: full intra-package import grep per file (absolute + relative + lazy/function-body, AST-equivalent to
`tests/test_import_hygiene.py`), field/attribute read-site counts, function line-span ranking. Every claim
below cites `file:line`. Where I could not verify, I say so.

---

## Bottom line

**The kernel is already import-clean: ZERO edges reach from any kernel module up into the shell.** The
plan's "5 lazy imports in `metal.py` block the carve" is stale and, read strictly, wrong — those 5–6 lazy
imports all target `geometry` and `stereo`, which are *themselves* shell-free, so they are kernel-internal,
not carve-blockers. What actually stands between today and `from rdkit_embed import embed, Constraints` is
**not tangled imports** but three packaging/organisation facts (single `__init__` fan-out, 4 shell-only base
deps, and two kernel modules that are internally half-shell). The carve is small; it is a *packaging* job,
not a *decoupling* job.

---

## A. The kernel ↔ shell boundary (actual import graph)

**Kernel** (the pure embedding engine): `io.py`, `log.py`, `constraints/{base,builders,mechanisms,metal,
polyhedron,sphere}.py`, `embed/bounds.py`, `refine/ff.py`, plus — established below — `geometry.py` and
`stereo.py`.
**Shell**: `pipeline.py`, `dedup/*`, `refine/{xtb,calculator}.py`, `constraints/nci.py`, `viz.py`,
`embed/{dispatch,mc}.py`.

### Kernel → shell import edges: **NONE**

Grepping every kernel module for `pipeline|dedup|viz|nci|dispatch|refine.xtb|refine.calculator|embed.mc`
imports returns nothing except one docstring line (`io.py:5`, prose: "…without importing the embed
dispatch…"). No kernel module imports any shell module, at module level or in a function body.

Corrections to expectations:

- **`log.py` has no real rxembed import.** The `import rxembed` at `log.py:3` is *inside the module
  docstring* (usage example, lines 1–6); the only real imports are stdlib `logging`. Clean leaf.
- **`io.py` is the bottom of the stack** — zero rxembed imports (`test_io_is_the_bottom_of_the_stack`
  encodes this). Its `import xyzgraph` (`io.py:28`) is a `try/except ImportError` fallback to
  `rdDetermineBonds` — a *soft* dep, not a hard one.
- **`metal.py` reaches nothing in the shell.** Its imports are `io` (`:23`), `polyhedron` (`:25`),
  `base` (`:26`), and lazy `sphere` (`:1311`), `builders` (`:2049`), `geometry` (`:840,895,2244,2332`),
  `stereo` (`:1801,2031`). Every one is kernel. `test_metal_does_not_reach_the_embed_dispatch` passes.

### The "5 lazy imports" the plan names are kernel-internal, not shell edges

They resolve entirely inside the kernel, so they do not block a carve; they only need to travel together:

| from | line | to | to's own deps |
|---|---|---|---|
| `metal.py` | 840, 895, 2244, 2332 | `geometry` | `geometry` imports only `metal` (`:483,893`) + numpy/rdkit |
| `metal.py` | 1801, 2031 | `stereo` | `stereo` imports only `io` (`:17`) + rdkit/xyzgraph(soft) |
| `mechanisms.py` | 265, 357, 401 | `geometry` | (same as above) |

`metal ↔ geometry` is a mutual lazy-import pair (metal→geometry and geometry→metal at `:483,893`), broken by
deferral. Both sides are kernel. Same for `mechanisms → geometry → metal`. No cycle passes through the shell,
and none passes through `mechanisms` (nothing in the kernel imports `mechanisms` except `bounds.py:14` and
`ff.py:124`, both kernel).

### This session added NO new kernel→shell edge

Verified for each addition named in the brief:

- `geometry.codonor_in_plane` — called at `mechanisms.py:280` **through the pre-existing** lazy
  `from rxembed import geometry` at `:265`. No new edge.
- `geometry.over_compression` — called at `geometry.py:694` (inside `check`), intra-module. No edge.
- `mechanisms.Sp2Planar` / `ConjugationCap` — added **two new kernel-internal** lazy `geometry` imports
  (`mechanisms.py:357, 401`), taking mechanisms→geometry from 1 site to 3. Still kernel→kernel; the knot is
  slightly denser but no worse for the carve.
- The FF-only cap skip (`Coplanar.ff_terms` → `codonor_in_plane`) — no new edge.

### Where `geometry.py` sits: kernel perception, but split-personality

`geometry.py` imports only `metal` (`:483,893`), never the shell — so structurally it is **kernel**, and it
*must* be (mechanisms' three caps and metal's fold/coplanar builders all consume its perception:
`_stripped_hybridisation`, `donation_axis`, `conjugated_quartets`, `inplane_sp2_donor`, `codonor_in_plane`,
`over_compression`). But the same 1048-line file also houses the **QA gate** — `check` (`:659`), `Violation`
(`:102`), `GeometryReport` (`:118`), and the ~15 `check_*`/violation functions — which is shell-facing (called
by `pipeline`/notebooks, not by the embedder). So `geometry.py` is half engine-perception, half QA gate. That
is a *readability/organisation* smell (T8 moves the census into `metal`; T12 splits the file), **not** an
import blocker.

---

## B. What an OIN-SMILES drop-in would import, and what it drags in

### The clean seam exists, and it is at `bounds.embed`

The pure "constrained embedding" surface, none of which touches pipeline/mc/score/dedup:

- `embed.bounds.embed(mol, cons, n, seed, prune_rms, knowledge, threads)` (`bounds.py:172`) — bounds-matrix
  ETKDG, returns conformer ids. Pulls `mechanisms` + `metal` at module level; `metal` pulls `io`, `polyhedron`,
  `base`. Lazily pulls `geometry`, `sphere`, `builders`, `stereo`.
- `refine.ff.restrained_uff(mol, cons, …)` (`ff.py:113`) — the FF relax that actually *enforces* the windows
  (embed only biases). Pulls `mechanisms` + `metal` (`ff.py:124,94,125`), both kernel.
- `constraints.base.Constraints` + `compose` (`base.py:24,198`) — the one struct in / one struct out.
- `constraints.builders.resolve_core` (`builders.py`) — the fix/constrain/template resolver (organic path).
- For a metal: `constraints.metal.enumerate_isomers` / `coordinate` / `from_geometry` — build metal `cons`.
- `io._xyz_to_mol` / `io.parse_smiles` (`io.py`) — load a source.

So `from rdkit_embed import embed, Constraints` **can** deliver the bounds-matrix embedder + FF relax with
**no** pipeline/mc/score machinery. The seam is real: `bounds.embed` (+ `restrained_uff`) is the embedding;
`pipeline.Ensemble` / `embed/dispatch.py` is the wrapping/search/score/dedup that sits *above* it. `dispatch`
is shell purely because it imports `pipeline` (`dispatch.py:24`, `Ensemble`/`EnsembleSet`) — its job is to
turn a source+spec into those objects.

### Dependency closure of that surface: RDKit + numpy only (hard)

External imports across the whole kernel are **numpy and rdkit** only, with two *soft* extras:
- `scipy` — only in `sphere.py` behind `available()` (`sphere.py:59–66`, `try: import scipy.optimize`).
- `xyzgraph` — only in `io.py:28` and `stereo.py:71`, both `try/except ImportError` perception fallbacks.

No kernel module imports `openconf`, `scikit-learn`, `networkx`, `xtb`, `ase`, `matplotlib`, `seaborn`, or
`pandas`. Those live in the shell (`mc.py`, `dedup/`, `nci.py`, `refine/xtb.py`, `viz.py`).

### What stands in the way of an actual drop-in (3 things, none is an edge)

1. **The single-package `__init__` fan-out.** `rxembed/__init__.py:5–11` imports `pipeline`, `nci`,
   `geometry` eagerly, so `import rxembed.embed.bounds` first executes the package `__init__` and pulls the
   *entire* shell (and transitively openconf/sklearn/networkx). `test_import_hygiene.py`'s own docstring
   (lines 6–9) calls this out. The kernel must become a **separate top-level package** (`rdkit_embed`) whose
   `__init__` imports nothing shell-ward. This is the whole of the carve.

2. **Four shell-only deps are declared *base*.** `pyproject.toml` `dependencies = [numpy, rdkit, openconf,
   xyzgraph, scikit-learn, networkx]`. The kernel needs only numpy + rdkit (+ soft scipy/xyzgraph);
   `openconf`, `scikit-learn`, `networkx` are shell-only and must move to extras before `rdkit_embed` can be
   installed light. **Gotcha:** `scipy` is *not declared anywhere* — the sphere solver gets it transitively
   via scikit-learn (confirmed: no scipy in `pyproject`). Drop scikit-learn in the carve and the sphere
   solver silently disables (`sphere.available()` returns False). Declare `scipy` explicitly for the kernel.

3. **Two kernel modules are internally half-shell (cosmetic for imports, real for a clean engine).**
   - `metal.py` (2397 lines) is ~60% coordinate-engine (kernel: `prepare`/`coordinate`/`ml_distance`/
     `solve_targets`/`hold_shape`/…) and ~40% **isomer enumeration** (`Isomer` `:1634`, `IsomerSet` `:1698`,
     `enumerate_isomers` `:1998`, `_isomers_for_geometry`, `arrangement`, `_load_in_ligand_stereo`, …). The
     enumeration is what `_SHELL = {…,"isomers"}` (`test_import_hygiene.py:20`) names, and it is the only
     reason `metal` reaches `stereo` (`:1801,2031`). Not an import blocker — it is all within `metal`, which
     has no shell edges — but the carved kernel would drag `enumerate_isomers` + `stereo` along.
   - `geometry.py` — the QA gate vs. perception split described in A.

---

## C. Cleanliness audit

The embed mechanics are **inherently intricate**, and most of the density is essential (the DG phase order,
the surrogate double-write, the intersect rule, the coplanarity extremum). The struct-integrity discipline is
genuinely good: `base.py:191–195` fails at import if any `Constraints` field lacks a `_CLONE`/`_MERGE`
policy, so a field can never be silently dropped by a copy/compose site. All 14 fields are read outside
`base.py` (distances 50, angles 36, frozen 25, haptic 13, coplanar 12, shapes 10, metals 9, planes 7,
phantoms 7, contacts 6, spheres 6, pulls 3, floors 3, dg_floors 3) — **no dead field**. Every session
addition is wired in (`codonor_in_plane`, `over_compression`, `conjugated_quartets`, `Sp2Planar`,
`ConjugationCap` all have live callers) — **this session added no dead code**.

Concrete issues, ranked:

### 1. `Mechanism.field` is dead metadata (write-only on all 11 classes)

`mechanisms.py` sets `field = "…"` on every `Mechanism` subclass (`:93,116,129,147,164,197,234,294,321,352,
396`), but **nothing reads it** — grep for `.field` finds zero read sites; the two drivers (`bounds.py:90–98`,
`ff.py:141`) dispatch by *method* (`m.dg_windows`/`m.ff_terms`), never by `m.field`. The tell that it is
decorative, not machinery: `Sp2Planar.field = "sp2_planar"` (`:352`) and `ConjugationCap.field = "conjugation"`
(`:396`) name Constraints fields **that do not exist** (their own docstrings say "reads no Constraints
field"). Minor, but it is exactly the "indirection a reader chases for no payoff" the house rules forbid.
Either delete it or make the base-class `field` load-bearing (e.g. assert each equals a real
`Constraints` field, which would also have caught the two fictional names).

### 2. The coplanarity family is dense but essential (3 constructs, 1 UFF lever)

`Coplanar` (metal, DG+FF, `:218`), `Sp2Planar` (organic sp2-carbon *improper*, hold-at-seed, FF-only,
`:333`), `ConjugationCap` (organic C–X *torsion*, target-flat, FF-only, `:376`) all attack "keep a conjugated
system planar" and share `_coplanar_window` (`:451`) + `_COPLANAR_FC` (`:44`). This reads as duplicative at
first glance, but the docstrings establish genuine distinctions (improper vs torsion DOF; metal vs organic
population; hold-at-seed vs target-flat), and the field-count/read-site check shows none is redundant. This is
**essential complexity, not cruft** — but it is at the edge: three ~25-line caps whose *why they differ* lives
in paragraph docstrings, which per AGENTS.md ("rename first, then delete the compensating docstring") is a
mild smell. Do not merge them (the plan already recorded "one cap clears both" as REFUTED by measurement,
HANDOFF T3c/T3d); leave as-is but treat as the ceiling of acceptable density here.

### 3. Concentration at the module level (the real readability cost)

Not monster functions — the longest are `_embed_dispatch` (150 lines, shell, `dispatch.py:620`), `mc`
(115, `pipeline.py:477`), `embed` (102, doc-heavy public API, `pipeline.py:247`), `_coplanar_donor` (94,
`metal.py:866`), `enumerate_isomers` (90, `metal.py:1998`). All hold. The cost is **module size**:
`metal.py` 2397, `pipeline.py` 1594, `geometry.py` 1048, `dispatch.py` 770. A newcomer cannot hold `metal.py`
(60+ functions spanning surrogate + distance model + polytope + solver + isomers + labelling). This is what
Phase-2 T11/T12/T13 target. **Note:** T7 (decompose to CC≤15) is committed on branch `t7-decompose`, **not
merged** to this branch — so `enumerate_isomers` etc. are still at their pre-T7 sizes here.

### Not cruft (deliberately retained, do not "clean")

- The **sphere solver** (`sphere.py`, `solve_targets` `metal.py:1300`) is reached only from
  `bounds._feasible_bounds:131` when raw coordination targets are non-metric — HANDOFF measures 246/251
  smoothing calls at tol 0.0, i.e. effectively unreachable on the default path, gated by monkeypatch tests.
  The maintainer's recorded decision is **keep it** (Scope decisions). It is a documented fallback, not dead
  code to delete.
- `spheres` field "no constraint WRITER reads it" (`base.py:73–75`) — read by the *engine*
  (`bounds._feasible_bounds` → `metal.solve_targets`) for re-derivation. Not dead.

---

## D. Verdict + shortest path to droppable

**How far today:** very close. The decoupling work is *already done* — 0 kernel→shell import edges, RDKit+
numpy hard-dep closure, a real seam at `bounds.embed`/`restrained_uff`/`Constraints`. What remains is
packaging and two internal tidy-ups. This is **not** "Phase 2 must land first" wholesale.

**Shortest clean path (load-bearing steps only):**

1. **T16 packaging is the actual carve and it is 90% of the value.** Make `rdkit_embed` a separate top-level
   package (kernel modules move; `metal.py` stays a re-export shim if not yet split); give it an `__init__`
   that imports nothing shell-ward; move `openconf`/`scikit-learn`/`networkx` to extras and **declare `scipy`
   explicitly** (else the sphere solver silently dies). After this, `from rdkit_embed import embed,
   Constraints` works with numpy+rdkit. Fix `test_import_hygiene.py` by inversion at the same time (**T15** —
   its current `_SHELL = {pipeline,dedup,viz,isomers}` is a denylist that, per the plan's own measurement,
   catches only 2 of 10 injected edges; an allowlist of the ~10 kernel names is the correct gate for a
   separate package).

2. **T8 + T9 are the two load-bearing tidy-ups, and both are genuinely needed for a *clean* engine** (not
   just cosmetic): T8 folds the fold-census perception from `geometry.py` into `metal.py`, collapsing the
   `metal↔geometry` mutual lazy-import and shrinking geometry's engine footprint; T9 lifts `enumerate_isomers`
   /`IsomerSet` out of `metal.py` into an `isomers` module — which is the *only* thing pulling `stereo` into
   the metal kernel, and `isomers` is shell (`_SHELL` names it). After T9 the carved metal kernel no longer
   drags the isomer enumerator or `stereo`.

3. **T11/T12/T13 are readability, not carve-blockers.** Splitting `metal.py` seven ways, `geometry.py` four
   ways, and de-duplicating `Ensemble`/`EnsembleSet` improve the module-concentration problem (issue C.3) but
   are **not** required for `rdkit_embed` to import cleanly or for OIN to consume it. Do them for the
   maintainers' sake, after the carve, or not at all before it. Likewise **T14** (renames) is cosmetic.

**What the plan gets wrong / this session changed:**

- The plan's premise "boundary = 5 lazy imports in `metal.py`" **overstates the blocker**: those imports are
  kernel-internal (→ geometry/stereo, both shell-free), so a carve does not need to eliminate them first —
  T8/T9 eliminate them for *tidiness*, not to unblock the package split. The genuine prerequisites are the
  `__init__` fan-out (2) and the base-dep set (3) — both packaging, both quick.
- This session made the carve **neither easier nor harder for imports** (0 new shell edges) but slightly
  **denser internally** (mechanisms→geometry went 1→3 lazy sites via the two new caps) — which raises the
  stakes on T8 (geometry perception should live with the kernel it serves).
- **Watch item the plan does not flag:** the undeclared-`scipy`-via-scikit-learn dependency (HANDOFF lists it
  under "Also outstanding" but not as a carve blocker). It *is* a carve blocker: the moment scikit-learn
  leaves the base deps, the sphere solver disables unless scipy is declared for the kernel.

**One-line verdict:** the embedding engine is cleanly separable today (0 kernel→shell edges, RDKit+numpy
closure, seam at `bounds.embed`); dropping it into OIN-SMILES is a packaging job — T16 + the base-dep/scipy
fix are load-bearing, T8/T9 make the carved kernel clean rather than merely functional, and T11–T14 are
optional readability that need not precede the carve.
