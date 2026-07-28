# Phase 2 simplification — progress record

## T8 — Fold census down: `geometry.py` → `constraints/metal.py` (DONE)

**What moved (perception + census, from `geometry.py` into `constraints/metal.py`):**
functions `_pi_hybridisation`, `_stripped_hybridisation`, `donation_axis`, `inplane_sp2_donor`,
`codonor_in_plane`; constants `_FOLD_WINDOW`, `_FOLD_MEDIAN`, `_FOLD_WALL_FLOOR`, `_MAX_SIGMA`,
`_CONJUGATING_LP`, the `_SP/_SP2/_SP3` hybridisation aliases (needed by the census dicts), and `_METAL_Z`.
These sit beside the `_orient_donor` / `_coplanar_donor` enforcement that must name the same donor class and
axis — enforcement and its perception are now co-located, so a seed the wall biases cannot be flagged the
other way by a drifting gate.

**What stayed on `geometry.py`'s gate side (correcting the stale plan):** the plan predates four functions
this project added. Decided cleanly per the task: `over_compression` (a whole-molecule QA gate keyed on the
graph, not metal-specific) and `conjugated_quartets` (organic amide/ester conjugation, read by the
`conjugation` gate and `mechanisms.ConjugationCap`) both **stay** in `geometry.py`. `inplane_sp2_donor` and
`codonor_in_plane` are metal-donor perception and **moved** with the census. `_PLANARITY_P95` (report-only
planarity threshold, geometry-gate side) stayed.

**The cycle is collapsed and one-directional now.** Before: `geometry ⇄ metal`, both via lazy in-function
imports (5 on each side). After: `metal.py` imports **nothing** from `geometry` (all 4 lazy `_geo` / deferred
imports deleted — `_orient_donor`, `_coplanar_donor`, `_donor_faces_metal`, `_distinct_orderings` now use
same-module names); `geometry.py` imports the ruler + census from `metal` in **one module-level** `from
rxembed.constraints.metal import (...)` (safe because `metal`'s module-level deps are only `io` / `polyhedron`
/ `base`, none of which touch `geometry`, and `__init__` loads `geometry` first — verified by import at
startup). `metal` owns the ruler; the gate imports it; nothing imports back.

**Call-site follow-through:** `mechanisms.Coplanar.ff_terms` now reads `_stripped_hybridisation` /
`codonor_in_plane` from `metal` (was `geometry`); its lazy import repointed. `tests/test_coplanar.py`
references updated `geo.* → M.*` for the moved symbols — **load-bearing** for the monkeypatch test
(`test_the_flag_rate_drops_on_a_crowded_conjugated_chelate` patches `codonor_in_plane` on the same module
object the code reads, so it had to move from `geo` to `M` or the patch would silently no-op).
`geometry._FOLD_WINDOW` / `_FOLD_MEDIAN` / `_stripped_hybridisation` / `donation_axis` / `_METAL_Z` remain
accessible as `geometry.*` because the gate genuinely uses them (re-exported by the import, not a shim) — so
`test_donor_fold.py`'s `geom.*` references needed no change.

**Surprise / note:** `metrics.py` keeps its **own** private `_METAL_Z` copy (identical definition) — left
untouched, out of scope. `metal.TRANSITION_METALS` (d-block subset) and the moved `metal._METAL_Z` (full
d/f-block, dative-exclusion set) are genuinely different sets serving different questions; both kept, no
consolidation attempted.

**Behaviour preserved.** No logic changed — a pure relocation. Golden bit-identical
(`tests/golden/test_golden_bounds.py` green), frozen-core graft assertions hold, full suite **329 passed / 2
skipped** (unchanged; no tests added, only `geo→M` reference repoints in `test_coplanar.py`). `ruff check` +
`ruff format` clean.

## T9 — Isomer cluster up: `constraints/metal.py` → new `isomers.py` (DONE)

**What moved (the coordination-isomer enumeration cluster, from `constraints/metal.py` into the new top-level
shell module `rxembed/isomers.py`):** the 6 named symbols `enumerate_isomers`, `IsomerSet`, `isomers`,
`_order_label`, `_resolve_center`, `_input_ordering` **plus the 17 helpers reachable only from them** —
`_octahedral_triad`, `_load_in_ligand_stereo`, `_prepare_spectators`, `_select_geometries`,
`_frozen_permutations`, `_isomers_for_geometry`, `_number_shared_labels`, and the trans-span pre-filter chain
`_central_trans` / `_span_bounds` / `_reach` / `_donor_faces_metal` / `_chelate_span_ok` / `_distinct_orderings`
(23 defs, 751 lines). `isomers.py` imports the kernel names it composes over from `constraints.metal`
(polytope tables, surrogate, `chirality_of`, `coordination`, the fold ruler, `Isomer`, `arrange`, `logger`).

**What stayed on the kernel side (`constraints/metal.py`):** `Isomer` and `arrangement`/`arrange` (required by
the task — kernel `from_geometry` returns `Isomer`, `Isomer.summary` calls `arrangement`), plus `from_geometry`,
`coordination_from_geometry`, `coordination`, `chirality_of`, `_donor_classes`, `_chelate_edges`, `label`,
`lone_pair_donors`, `coordinate` — none of which call any moved function.

**The last 2 kernel→shell edges are deleted — confirmed.** They were the two lazy `from rxembed import stereo`
imports (in `_load_in_ligand_stereo` and `enumerate_isomers`); both moved with the cluster. `metal.py` now has
**zero** rxembed-shell imports at any depth — its only remaining lazy imports are `from . import sphere`
(kernel) and `from rdkit.Chem import rdDistGeom` (external). `test_metal_does_not_reach_the_embed_dispatch`
(whose `_SHELL` already lists `isomers`) is green. Call sites repointed: `__init__.py` (`rx.metal`) and
`dispatch.py:_dispatch_metal_source` now import `enumerate_isomers` from `rxembed.isomers`.

**Plan staleness.** The plan named 6 symbols / "≈436 lines" and cited `metal.py:1490`/`1670`; both are stale.
T8 grew `metal.py` to 2615 lines (those line numbers now land elsewhere), and T7's decomposition of
`enumerate_isomers` into 7 named stages plus the span-gate helpers means the cohesive movable cluster is 23
functions / 751 lines, not 6 / 436. A move of *only* the 6 named symbols is impossible without either leaving
`_load_in_ligand_stereo`'s stereo edge behind in the kernel or introducing a new `metal → isomers` back-edge
(it recursively calls `enumerate_isomers`), so the whole cluster had to travel.

**Surprises / notes.** (1) `_SPAN_ANGLE` is also read by the **staying** `coordination` (`metal.py:1489`, the
cis-vs-trans chelate-bite branch), so it and its documented pair `_SPAN_TOL` **stayed** in `metal.py` and are
imported by `isomers.py` — an initial move of them tripped `ruff` F821, caught before any test ran. (2)
`from collections import Counter` and `compose` (from `.base`) became unused in `metal.py` after the move and
were deleted (F401). (3) `isomers.py` **reuses `metal`'s `logger`** (imported by name), so the enumeration's
log records still carry the name `rxembed.constraints.metal` — this keeps `test_permutation_warning.py` (which
pins `caplog ... logger="rxembed.constraints.metal"`) green with no test edit, a behaviour-preserving choice.
(4) `enumerate_isomers`'s `from .builders import resolve_core` was rewritten absolute
(`from rxembed.constraints.builders import ...`) since the code now lives in a different package.

**Behaviour preserved.** A pure relocation. Golden bit-identical, frozen-core graft assertions hold, full suite
**329 passed / 2 skipped** (unchanged; no tests added). `ruff check` + `ruff format` clean. Zero new `ty`
diagnostics — the one `ty` finding (`metal.py:1080`, `_FOLD_WINDOW[cls]` typing in the untouched
`_coplanar_donor`) is a pre-existing T8 artifact, verified present identically at HEAD.

## T11 — Split `metal.py` (three coherent sub-models carved out) (DONE)

The HANDOFF "split seven ways + re-export shim" plan was overridden by the maintainer as over-abstraction.
Carved `metal.py` (1923 lines) into **three** genuinely-coherent standalone modules, no shim — every caller
updated to import from the new module directly. Metal core drops to ~1150 lines.

**What moved:**
- `constraints/distance.py` (~320 lines) — the M–L bond-length model + surrogate anti-overbond floors:
  `ml_distance`, `delocalised_charges`, `_hapticity`, `ff_terms`, `overbond_tier`, `_tier_floor`,
  `nondonor_floors`, and their constants (`_PHYS_COEF`/`_METAL_GROUP`/`_PAULING_EN`, the floor ratios,
  `APEX`/`NEAR`/`OUTER`, `_SOFT_DATIVE_DONORS`, …). Imports only `TRANSITION_METALS`, `VACANT` from the core.
- `constraints/donor_orient.py` (~365 lines) — donor perception + the orientation holds the surrogate loses:
  `_stripped_hybridisation`, `_pi_hybridisation`, `inplane_sp2_donor`, `codonor_in_plane`, `donation_axis`,
  `_orient_donor`, `_coplanar_donor`, and the fold-census tables (`_FOLD_WINDOW`/`_FOLD_MEDIAN`/`_FOLD_WALL_FLOOR`,
  the coplanar-cap constants, `CENSUS_OOP_P95`, …). Imports `overbond_tier`/`APEX`/`_APEX_DONORS` from
  `distance`, `_METAL_Z` from the core.
- `constraints/solver.py` (~145 lines) — the sphere-solver fallback: `solve_targets`, `_dummies`,
  `_ligand_pairs`, `_solve_start`. Imports `ml_distance`/`delocalised_charges` from `distance`,
  `VERTEX_DIRS`/`_site_radius`/`_frag_map`/`VACANT`/`logger` from the core, `add_distance` from `base`.

**Why only these three (and why the polytope stayed in core).** The task named "distance/surrogate",
"polytope + solver", and "donor orientation" as candidate seams. Distance and donor-orientation are clean
sub-models. For seam 2, the polytope **tables** (`GEOM`/`ANGLES`/`PERMUTATIONS`/`VERTEX_DIRS` + `classify_geometry`
/`geometry_for`/`coplanar`) are the shared vocabulary the staying `coordination`/`chirality_of`/`classify_geometry`
glue speaks, so isolating them from their consumers would *reduce* coherence — a core importing its own
vocabulary. The **solver** is the genuinely off-to-the-side "~2% feasibility fallback"; extracting it (and only
it) declutters the core's `coordination`→`Isomer` main flow. Combining polytope+solver into one module was
rejected: the solver reaches back into core haptic helpers (`_site_radius`, `_frag_map`), which would make a
`core ↔ polytope` cycle.

**Import direction (no cycle, no shim, ruff-clean).** Mirrors the existing `metal↔geometry` pattern: the three
new modules import `metal`'s foundational constants at their top (E402-clean), and `metal` imports *them* lazily
inside the four coordination-building functions (`coordination`, `_centroid_constraints`,
`coordination_from_geometry`, `coordinate` — `PLC0415` is ignored project-wide). `metal` has **no** eager edge to
any of the three, so `metal` loads standalone and the sub-modules load after it (verified by cold-importing each
in a fresh process). `rdDistGeom` dropped from `metal`'s top import (only the moved `_ligand_pairs` used it there).

**Callers updated directly (no shim):** `geometry.py`, `metrics.py`, `mechanisms.py` (lazy), `isomers.py`,
`pipeline.py`, `embed/dispatch.py` (→ `distance.ff_terms`), `embed/bounds.py` (→ `solver.solve_targets`); tests
`test_connectivity`, `test_coplanar`, `test_donor_fold`, `test_metal_chelates`, `test_sphere_solver`. Load-bearing
prose pointers (`base.py`, `ff.py`, geometry/metrics docstrings) repointed to the new module names. The
`test_sphere_solver` monkeypatches now target `solver.solve_targets` (the module `bounds` actually calls) and the
census-provenance test now inspects `donor_orient`'s source.

**Behaviour preserved.** Pure relocation, byte-identical moved code (`ruff format` left all files unchanged).
Golden **bit-identical**, frozen-core graft holds (test_frozen 10/10, 0.00000 Å). Full suite **329 passed /
2 skipped** — unchanged, no tests added. The one intermittent `test_fix_numbers_deliver_sn2_core` "failure" seen
in a full run is the flake its own docstring documents (~2% on the numerically-marginal near-linear SN2 core, an
**organic** path touching none of the moved code); it passes 5/5 in isolation. `ruff check` + `ruff format` clean.

## T12 — Split `geometry.py` (types + vecmath + coordination carved out) (DONE)

The HANDOFF "four ways: types/vecmath/donor/checks" plan was taken with the maintainer's T11 override —
clean readable flow, no re-export shim, callers updated directly. `geometry.py` (845 lines) was one coherent
"geometry gate", but it held a genuinely separable ~290-line metal-coordination concern; carved into **three**
new standalone modules + the checks remainder that keeps the public name `geometry`.

**What moved (verbatim bodies, byte-identical):**
- `report.py` (55 lines) — `Violation` + `GeometryReport`. The load-bearing split: both gate modules build
  `Violation`, so making it a leaf both import is what breaks the report↔checks cycle (else `coordination`
  imports `Violation` from `geometry` while `geometry` imports the gates from `coordination`).
- `vecmath.py` (64 lines) — the coordinate/periodic-table/vector-math leaf: `_PT`, `_CARBON_Z`, `_EPS`,
  `_positions`, `_rcov`, `_rvdw`, `_kabsch_rmsd`, `_plane_offset`, `_angle`, `_dihedral`. Needed as its own
  leaf so `coordination` can reach `_angle`/`_dihedral`/`_positions`/`_rcov` without importing `geometry`
  (which would re-form the cycle) — not fragmenting for count.
- `coordination.py` (373 lines) — the metal coordination-sphere gates + sphere perception: `metal_overbond`,
  `donor_orientation`, `donor_fold`, `DonorAngle`, `FoldReport`, `_donor_walk`, `_planarity_dev`, `_spheres`,
  `_coordinating`, `_coordinating_carbons`, `_coordination_pairs`, `_eta2_pi_atoms`, + `_COORD_FACTOR`/
  `_SIDEON_*`/`_PLANARITY_P95`.
- `geometry.py` remainder (410 lines) — the general ground-state checks (`bond_lengths`, `hydrogens`,
  `clashes`, `over_compression`, `planarity`, `conjugated_quartets`, `conjugation`, `frozen_core`,
  `check_constraints`, `stereo`) + the `check()` orchestration + `_cip`/`_attr`.

**Import direction (acyclic, no shim):** `report` and `vecmath` are leaves; `coordination` → {report, vecmath,
constraints.{distance,donor_orient,metal}}; `geometry` → {report, vecmath, coordination, constraints.metal}.
Verified by cold-import in a fresh process. `check()` calls `metal_overbond`/`donor_orientation` by the name
bound in `geometry`'s namespace, so `test_donor_fold`'s `monkeypatch.setattr(geom, "donor_orientation", spy)`
still intercepts (behaviour preserved).

**Callers updated directly:** tests `test_donor_fold` / `test_connectivity` / `test_metal_chelates` now import
`coordination` for the gate functions (`_stripped_hybridisation` / `_FOLD_WINDOW` pulled from their real home
`constraints.donor_orient`); stale prose pointers (`geometry.metal_overbond` / `.donor_orientation` /
`._donor_walk`) in `distance.py`, `donor_orient.py`, `metrics.py` repointed to `coordination.*`. **No src code
called the moved gates** — the only external code call to this file is `pipeline._geometry.check` (`check` stays)
and `mechanisms._geo.{conjugated_quartets,_CARBON_Z,_SP2_DEGREE}` (all still resolve via `geometry`), so
`mechanisms.py` needed no change.

**Surprise / honest ledger:** the four files total ~902 lines vs 845 — **+57** from three module docstrings +
import blocks; this buys readability, not fewer lines (per AGENTS.md on "moving is not simplifying"). The plan's
`types.py`/`donor.py` names were changed to `report.py`/`coordination.py`: `types` shadows stdlib and `donor`
collides confusingly with `constraints/donor_orient`. **Behaviour preserved** — golden **bit-identical**,
frozen-core graft 0.00000 Å, full suite **329 passed / 2 skipped** unchanged, `ruff check`/`format` + `ty` clean.

## T13 — De-duplicate `Ensemble` / `EnsembleSet` (NO CHANGE — plan premise already stale)

**The plan is stale: the duplication T13 targets does not exist.** T13's text ("`pipeline.py` re-implements
the same stage logic twice … extract the shared surface so `EnsembleSet` maps over candidates without copying
each method body") describes a pre-`_map` state that was already gone when the plan was written. Verified: the
HANDOFF-creating commit `28d38de` — pipeline.py **exactly 1436 lines**, the number the plan cites — already
carries `EnsembleSet._map` and the thin delegators, structurally identical to HEAD; `_map` itself landed even
earlier, in `d83ae4d` ("make EnsembleSet chainable"). So "maps over candidates without copying each method
body" is **already implemented**. Every stage's logic lives exactly once, in `Ensemble`: the seven verb methods
on `EnsembleSet` (`mc`/`minimize`/`prune`/`score`/`optimize`/`lowest`/`representatives`) are one-line
`return self._map("<verb>", *args, **kw)` delegators; `EnsembleSet.dump` calls `Ensemble.dump` per candidate;
`EnsembleSet.filter`'s `by=` branch routes through `_map`. No method body is copied between the two classes —
the overlapping-name methods are either delegators or genuinely different concerns (`__repr__` = set-summary vs
ensemble-summary; `filter` = tag-subset vs connectivity; `dump` = one-file-per-candidate vs multi-frame).

**No behaviour-preserving change reads more clearly than what is there, so per the task's own escape hatch
("prefer a smaller, clearer de-dup or leave it") this is LEFT UNCHANGED.** The only residual repetition is the
seven one-line delegators. Collapsing them the sole remaining way — a metaprogrammed `setattr` loop or
`__getattr__` — would delete their per-verb docstrings (each documents the WITHIN-candidate, never-pooled
semantics), hide the supported-verb surface from readers/IDEs, and mask real `AttributeError`s: exactly the
"confusing abstraction/wrapper layer" the maintainer ruled out (clarity over cleverness, no over-abstraction).
That is a strictly worse read, not a de-dup. No src edit made; golden trivially **bit-identical**, full suite
**329 passed / 2 skipped**, `ruff check`/`format` clean (all unchanged — nothing touched but this record).

## T14 — Rename, then shrink the compensating docstrings (DONE)

**Renames (clarity-first, every call site grep-verified across `src/` + `tests/`):**
- `mechanisms.REGISTRY` → `MECHANISM_ORDER` (the tuple whose order IS the DG/FF phase algorithm). Sites:
  `mechanisms.py` def + module docstring, `bounds.py` (×3), `ff.py` (loop + docstring), `tests/golden/capture.py`
  + `tests/test_conjugation_cap.py` prose.
- `dedup.active_blocks` → `active_feature_kinds`. Sites: `dedup/features.py` def + 2 calls + docstring,
  `dedup/__init__.py` import + `__all__`, `pipeline.py` docstring.
- `isomers.isomers` → `distinct_vertex_orderings` (resolves the module-name/function-name collision — a function
  `isomers` living in `isomers.py`). Sites: the def + its one intra-module call, plus 3 backtick prose refs in
  `metal.py`. **Plan location was stale** — the plan named `metal.isomers`, but T9/T11 moved this function to
  `isomers.py`; verified there before renaming.
- `metal.prepare` → `surrogate_metal`, `metal.prepare_all` → `surrogate_all_metals`, `metal.restore` →
  `restore_metal`. Sites: the 3 defs + their cross-references in `metal.py`, `isomers.py` (import + 2 calls),
  `pipeline.py`, `dispatch.py`, prose in `io.py`/`distance.py`, and `tests/test_metal_charge.py` +
  `tests/test_mol_state.py` (calls, one test-fn name, assertion messages).
- `geometry.stereo` → `stereo_violations` (def + its one caller inside `check`).
- `stereo.passes` → `satisfies_spec` (def + 3 calls in `pipeline.py`).

**Dead code deleted:**
- `Mechanism.field` metadata — the `field = "…"` class attribute on the base + all 10 subclasses. Confirmed
  **write-only** (zero `.field` reads anywhere; both drivers dispatch by method, `m.dg_windows`/`m.ff_terms`),
  matching `separability-assessment.md §1`. Two of them (`Sp2Planar`/`ConjugationCap`) even named nonexistent
  `Constraints` fields. The per-class trailing comment ("reads no Constraints field…") went with the attribute;
  each class docstring already says so.
- `stereo.kinds` and `stereo.relevant` — **plan named these for rename, but both are dead** (zero callers in
  `src/`, `tests/`, or `playground/`; not exported, no `__all__`). Renaming code no one reads has no clarity
  payoff, so per the house rules (Delete-before-you-add / YAGNI) they were **deleted, not renamed** — the one
  deviation from the literal plan, reported as staleness. `_NONGRAPH` (relevant's only unique dependency) is
  still read by `_mode`, so it stayed. `stereo.passes` is live and was the only one of the trio to rename.

**Naming judgment (recorded so it is not re-litigated):** the plan gave explicit targets for the load-bearing
renames but none for the `stereo` trio. `passes` → `satisfies_spec` reads correctly at every call site
(`if _stereo.satisfies_spec(sig, ref, spec):`). `prepare_all` was renamed alongside `prepare` (the plan named
only `prepare`/`restore`) because leaving `prepare_all` beside `surrogate_metal` breaks the pair a reader relies
on; `surrogate_all_metals` keeps them recognisably one family. The `Isomer.restore()` / `_MetalCtx.restore()`
**methods** were left as `.restore()` (the plan named the module-level functions; `iso.restore()` reads fine on
an object that carries the metal context, and they now call `restore_metal` internally).

**Docstring trim: deliberately minimal, not a mass trim.** Per the plan's own note, the ~6 doc-heavy places
(`donor_orientation`, `pipeline.embed`) are where prose *legitimately* goes — left untouched, and AGENTS.md
forbids mass-deleting chemistry prose. The renames I made left no compensating hedging to cut (the surviving
docstrings are load-bearing chemistry/design prose); the only prose removed was the now-redundant `field`
attribute comments (deleted with the attribute). Four `metal.py`/`isomers.py` lines were re-wrapped only because
the longer `distinct_vertex_orderings` name pushed them past 120 cols — content byte-preserved, just reflowed.

**Behaviour preserved — pure renames + dead-code deletion.** No logic changed. Golden **bit-identical**
(`tests/golden/` 23/23 green), frozen-core graft holds (0.00000 Å), full suite **329 passed / 2 skipped**
(unchanged — the one renamed test still runs; no tests added or removed). `ruff check` + `ruff format` clean.

## T15 — Invert `tests/test_import_hygiene.py` to a kernel allowlist (DONE)

**What changed (test-only; no `src/` touched):** replaced the shell *blocklist* (`_SHELL = {pipeline, dedup,
viz, isomers}` + `_layer()` returning `parts[1]`) with a kernel **allowlist** `_KERNEL` — the 18 dotted names
the embed kernel is built from (`io`, `log`, `report`, `vecmath`, `coordination`, `geometry`, the
`rxembed.constraints` package + its 10 kernel leaves `base`/`builders`/`polyhedron`/`sphere`/`mechanisms`/
`metal`/`distance`/`donor_orient`/`solver`, `embed.bounds`, `refine.ff`). A kernel module may import these and
nothing else under `rxembed`; everything absent is shell. `_layer()` and `_SHELL` are **deleted**.

**Why the blocklist was broken (measured).** `_layer` collapsed a dotted name to `parts[1]`, so a shell module
sharing a package with the kernel was invisible: `constraints.nci` → `"constraints"`, `embed.mc` → `"embed"`,
`refine.xtb`/`refine.calculator` → `"refine"` — none in `_SHELL`, all passed silently. Reproduced side-by-side
in this session: the old logic caught **0/4** of the named edges (nci/networkx, mc/openconf, xtb, calculator/ase);
the allowlist catches **4/4**. The new `test_an_injected_kernel_to_shell_edge_is_caught` (10 shell targets ×
2 import shapes) is the standing guard, and a live mutation — injecting `from rxembed.refine import xtb` into
`constraints/base.py` — flipped `test_every_kernel_module_imports_only_the_kernel[…base]` red as required, then
was reverted.

**How the allowlist judges the leaf, not the layer.** `_module_of()` resolves each imported dotted name to the
longest prefix that is a real module file (`constraints.metal.TRANSITION_METALS` → `constraints.metal`), so a
symbol import is judged by its module and a `from <kernel-package> import <shell-leaf>` is judged by the leaf —
`from rxembed.constraints import nci` flags `constraints.nci` even though the `constraints` anchor is allowed.
The one deliberate excusal is the top-level `rxembed` package itself: `from rxembed import geometry` (used by
`mechanisms`) puts a bare `rxembed` in the import set, so `_shell_reached` subtracts `{"rxembed"}` and judges the
carried submodule — matching the old test, which also never flagged bare `rxembed` (`_layer` → `""`).

**The test now covers the whole kernel, not a 5-module sample.** Parametrizing over `sorted(_KERNEL)` (18 cases)
makes the invariant "the kernel is import-closed" — every kernel module imports only kernel modules — which the
inversion makes cheap to state once. `geometry`, `coordination`, `report`, `vecmath` are classified kernel
because the code forces it: `mechanisms` (kernel, imported by `bounds`/`ff`) imports `geometry`, `geometry`
imports `coordination`/`report`/`vecmath`, and each of those imports only other kernel modules — there is no
partition of the *current committed* tree where they are shell and the tree is clean, so the classification is
read off the code, not chosen. `test_the_allowlist_names_only_real_modules` guards `_KERNEL` against a stale
entry after a future rename.

**Plan staleness (reported, not silently corrected).** The HANDOFF says "may import only the **10** kernel dotted
names" and cites the 2-of-10-caught figure. The "10" predates T11's split of `metal.py` into `distance`/
`donor_orient`/`solver` and T12's split of `geometry.py` into `report`/`vecmath`/`coordination` — the final
layout has **18** kernel dotted names, so the allowlist is 18, not 10. The 4 named misses (nci/mc/xtb/calculator)
reproduce exactly.

**Behaviour preserved (src) / net new tests.** No `src/` file touched → golden **bit-identical** and frozen-core
graft holds by construction (both harnesses green inside the run). The three original assertions survive
(`test_io_is_the_bottom_of_the_stack`, `test_metal_does_not_reach_the_embed_dispatch`, and the leaf-import check
now generalised to all kernel modules). File went 6 → 31 tests (+18 kernel-module cases, +10 injected-edge cases,
+1 allowlist-integrity, −4 collapsed into the parametrized whole-kernel check); full suite **354 passed / 2
skipped** — the T8–T14 baseline of 329 plus exactly the 25 tests added here. `ruff check` + `ruff format` clean.
