# Generality cleanup — ordered implementation plan

Synthesis of four read-only investigations: `gen-fold-tables.md`, `gen-element-sets.md`,
`gen-polyhedra-home.md`, `scope-notebooks-and-verbosity.md`. This plan orders the work, separates
mechanical go-aheads from trade-off decisions, and maps the file collisions that force a sequence.

The three items that all edit `metal.py` / `donor_orient.py` **collide** — the whole point of the
ordering below is to touch each file once, in one direction.

---

## Recommended master order (at a glance)

| Phase | Work | Gated on | Touches |
|---|---|---|---|
| 0 | Notebook fixes; `pipeline.py` + `isomers.py` de-verbose | nothing — start now | notebooks, pipeline.py, isomers.py |
| 1 | Get the three decisions' nods (2.1, 2.2, 2.3) | maintainer | — |
| 2 | `metal.py` chain: **[move] → de-clunk → de-verbose** (serial) | decision 2.3 | polyhedron.py, metal.py |
| 3 | `donor_orient.py`: apply 2.1 provenance **with** the de-verbose trim (serial) | decision 2.1 | donor_orient.py |
| 4 | Golden validation + notebook re-run | phases 2-3 | tests |

Phase 0 is fully independent of everything else and can land immediately. Phases 2 and 3 are
independent of each other (different files) and can run in parallel once their decisions are made.

---

## PART 1 — CLEAR GO-AHEADS (mechanical, low-risk)

### 1.1 Notebook import/attr fixes — DO FIRST, independent

Four broken cells (`scope-notebooks-and-verbosity.md` Job A), all surgical single-line edits:

- `07_metal.ipynb` cell 1, `henry.ipynb` cell 1, `karoline.ipynb` cell 2:
  `from rxembed.constraints.metal import metal_indices` →
  `from rxembed.rdkit_embed.constraints.metal import metal_indices`
- `01_basics.ipynb` cell 11: `dedup.active_blocks(ens.mol, ens.ids)` →
  `dedup.active_feature_kinds(ens.mol, ens.ids)` (pre-existing T14-rename break, not a maintainer edit).

**Recommendation:** fix all four. Optional, not required: repoint the `from rxembed.embed.dispatch
import _xyz_to_mol` lines (06/07/henry/karoline) to the canonical `rxembed.inputs` — works today via
re-export, so leave unless doing a house-style pass.

**Risk:** low. Two live traps:
1. The `rxembed.constraints.metal | ...` strings in **output** cells are **logger names, not imports**
   (loggers are deliberately pinned to `rxembed.*`). Do NOT touch them.
2. All 8 notebooks are git-dirty (maintainer re-runs + edits). Edit only the one broken **code** cell
   per notebook via `NotebookEdit` — never checkout/revert, or you lose the maintainer's work.

**Golden-validated?** No golden. Proof is the notebook's own `geom.check(...).assert_ok()` gates on
re-run (phase 4).

**Order:** fully independent — no source overlap with any other item. Start here.

### 1.2 POLYHEDRA literal de-clunk — SEQUENCE AFTER the move decision (2.3)

Per `gen-polyhedra-home.md` §2: keyword args + derive `cn` via `__post_init__` (delete the redundant
hand-typed field, `cn == len(vertex_dirs)` for all 13 records) + hoist the three long isomer blocks
(`_OCT_ISOMERS`, `_TBP_ISOMERS`, `_SPY_ISOMERS`) to named constants. All values **bit-identical**.

**Recommendation:** the full form (keyword + derived `cn` + hoist). Minimal floor if you want a
zero-dataclass-change diff: keyword-args-only.

**Risk:** touches **MEASURED / golden geometric templates** (the `vertex_dirs` and `angles` are the
canonical polyhedra). Values are verbatim, so risk is mechanical-only — but it must be golden-validated
because the templates seed every metal embed. Keep the `{p.name: p ...}` dict-comprehension: insertion
order is the `classify_geometry` tie-break (load-bearing).

**Golden-validated?** **Yes** — `test_sphere_solver.py`, `test_metal_chelates.py`, `test_haptic.py`,
plus the metal-isomer suite.

**Order:** must NOT run before the move decision. If 2.3 = **yes**, do the move first and de-clunk in
the new home (`polyhedron.py`) — never edit the literal in `metal.py` then move it. If 2.3 = **no**,
de-clunk in place in `metal.py`. Either way, de-clunk **before** the `metal.py` de-verbose sweep (1.3)
so the file is edited once in that region.

### 1.3 Docstring / comment de-verbose sweep — SEQUENCE LAST per file

Per `scope-notebooks-and-verbosity.md` Job B, priority order by prose weight:
1. `pipeline.py` (biggest sink, ~42k prose chars) — `embed` (54L), `mc` (23L), `_relax_into_windows`
   (20L), `prune`, `Ensemble` class. **Independent — do in phase 0.**
2. `metal.py` (268 doc lines / 1047 LOC) — broad function-docstring trim. **Sequence after 1.2/2.3.**
3. `donor_orient.py` (highest comment density) — `_coplanar_donor` (28L), `_orient_donor`,
   `donation_axis`, `_stripped_hybridisation` essays + census `#` blocks. **Sequence with 2.1.**
4. `isomers.py` — `enumerate_isomers`, `_donor_faces_metal`. **Independent — do in phase 0.**
5. `mechanisms.py` module essay + subclass docstrings — **with carve-outs below.**
6. Runners-up: `coordination.py`, `distance.py`, `nci.py`, `geometry.py`, `embed/dispatch.py`.

**Recommendation:** trim essays to one-line imperative summaries + only non-inferable specifics
(the `ruff D` house rule). Proceed module-by-module.

**DO NOT CUT (load-bearing):**
- `mechanisms.py` `Mechanism` base hooks (`dg_windows`/`dg_relief`/`dg_post`/`ff_terms`) — already
  one-liners, the single source of the contract that `D102` is exempted for. Do NOT add subclass
  docstrings either (that reintroduces the duplication the exemption removes).
- `mechanisms.py` measured-refutation constant comments (`_COPLANAR_FC`, `_SP2_HOLD_FC`,
  `_SP2_HOLD_WIN`, `_CONJ_CAP`, `_SOFT_PULL_FC`) — keep the number, the pinning test name
  (`test_reembed_retry_delivers_clean_geometry`), and the one-clause "can't move because…". Trim only
  surrounding prose.
- `mechanisms.py:262/407` ordering comments (why *fields* drive DG order) — a real correctness note.
- The `donor_orient.py` fold-table provenance — this sweep must **preserve/sharpen** it, not cut it
  (see 2.1); it is the `AGENTS.md` "say where the data came from" requirement.

**Risk:** low (prose only) — but the sweep touches `metal.py` and `donor_orient.py`, which collide with
1.2/2.3 and 2.1 respectively. That is the whole reason for the phase ordering.

**Golden-validated?** No golden needed; tests must stay green (imports/symbols unchanged). Bundle into
phase 4.

**Order:** `pipeline.py` + `isomers.py` in phase 0 (independent). `metal.py` **after** the
move+de-clunk. `donor_orient.py` **with** decision 2.1.

---

## PART 2 — DECISIONS NEEDING YOUR NOD (fidelity / behaviour trade-offs)

### 2.1 Generalise `_FOLD_WINDOW` / `_FOLD_MEDIAN` — RECOMMEND: KEEP + DOCUMENT (reject scaling)

Per `gen-fold-tables.md`. The 10 `(element, hyb)` rows are a **measured crystallographic census** (721
M-D-X angles over 121 tmQM/Kulik + fixture crystals; generated by `OIN-SMILES
tools/donor_angle_census.py`, commit `a05f2bc6`). Two quantities behave oppositely:

- The **MEDIAN** is smooth — ≈ VSEPR ideal(hyb) ± ~5°; a group/period fit reproduces it (RMSE 4.2°).
- The **FLOOR** (the safety-critical **gate**, and it seeds the wall) is a per-class distribution tail
  (p0.5) and is **NOT smooth**: a group/period fit moves `(N, sp)` **+12.6°** (would false-flag real
  nitrile/isocyanide crystals — the domain-fatal direction) and `(C, sp3)` **−18.4°** (silently weakens
  the gate). This is exactly the "global-fit win hides a per-row regression" failure the project has
  reverted **twice** (`metal-ff-cancellation`, `covalent-radius-tmqm-grounding`).

**Recommendation — option (b):** keep the measured table; add a compact provenance + why-per-class
block to its docstring (generator path/commit, corpus, `floor = min(p0.5, corpus_min) − 5°` at 0/144
false flags, `ceiling = p99.5 + 5°`, `median = p50`, `n < 6 abstains`, and the one-line "per-class not
scaled because the floor is a scatter tail — scaling false-flags nitriles"). This satisfies
`AGENTS.md:34-35` (census fold windows are the named canonical example of *data* that must say where it
came from).

**Acceptable partial — option (c) hybrid** (only if row-count reduction is itself the goal): derive
`_FOLD_MEDIAN` from `ideal(hyb) + small offset`, keep `(S, sp3)` as an explicit override, and **leave
`_FOLD_WINDOW` measured**. Removes ~half the rows at no gate-fidelity cost, but adds a second code path
for modest value. Not preferred.

**Reject — option (a):** fully group/period-scaled. Faithful for the median, a **regression for the
gate**. Do not replace the measured floor with a model.

**Risk:** highest of the set — directly on **MEASURED gate values**. The recommendation (b) is doc-only
(add prose), so it is low-risk itself; the risk is in the alternatives.

**Golden-validated?** (b) doc-only: no golden, but re-run the fold/donor-orientation census regression
to confirm zero text/value change. (c) hybrid: **yes**, must re-run the 144-crystal 0-false-flag
census.

**Order:** decide before touching `donor_orient.py`. Apply (b)'s provenance **together with** the
`donor_orient.py` de-verbose (1.3 #3) — the sweep trims the `_coplanar_donor`/`_orient_donor` essays
while the fold table *gains* a tight provenance block (net can still be shorter). One coordinated edit
to that file.

### 2.2 General predicates for `_CONJUGATING_LP` / `_LONE_PAIR_Z` — RECOMMEND: KEEP BOTH

Per `gen-element-sets.md`.

- **`_CONJUGATING_LP = {7, 8}`** (donor_orient.py:94, N/O lone-pair-conjugation typing). The maintainer's
  candidate rule "p-block period-2 with a lone pair" = `{7,8,9,10}` — it **wrongly admits F, Ne**. No
  clean general predicate yields exactly `{7,8}` and reads more truthfully than the two-element
  allow-list. The load-bearing axis is **period 2 vs 3+** (PPh₃ must stay pyramidal), already in the
  comment. **KEEP the allow-list.**
- **`_LONE_PAIR_Z = {7, 8, 15, 16}`** (metal.py:1032, `coordinate="auto"` candidate-donor **screen**,
  not a hard gate). An *exact* equivalent exists — `GetNOuterElecs ∈ {5,6} ∧ period ≤ 3` (N/O/P/S,
  excludes Se/As) — and is self-documenting, but only **re-spells** the list. The `period ≤ 3` cap is
  the load-bearing, validated part: dropping it silently admits V/Cr/Nb/Mo (RDKit reports NOE 5/6) plus
  untested Se/As/Sb/Te. **KEEP** (adopt the predicate only as a consistent house-style pass, cap
  retained verbatim; no correctness gain either way).

**Recommendation:** keep both as explicit allow-lists. The only action, if any, is a one-line comment
sharpening on set 1 ("period-2 only; P/S must not conjugate") — which the existing comment already
covers.

**Risk:** low, and the risky directions are refuted (see Part 3). Set 1 is a per-atom hybridisation
typing; set 2 is a soft screen that logs+truncates.

**Golden-validated?** Only if a predicate is actually adopted (set 2 house-style) — then run the
`coordinate="auto"` donor-selection cases. The recommended KEEP needs none.

**Order:** independent (a no-op / comment-only). Fold any comment tweak into the `donor_orient.py`
(set 1) and `metal.py` (set 2) de-verbose passes so those files are touched once.

### 2.3 Move `POLYHEDRA` + `Polyhedron` to `polyhedron.py` — RECOMMEND: YES

Per `gen-polyhedra-home.md` §1. The "risk a cycle" concern is **backwards**: `polyhedron.py` imports
nothing from rxembed (stdlib + numpy only) and the only edge is `metal → polyhedron`. Moving data into
the imported-from module cannot close a cycle unless that data imports back — and `POLYHEDRA` is
tuples/ints/bools. **No cycle.**

**Recommendation:** move `_s`, `_vertex_angle`, `Polyhedron`, `POLYHEDRA`, `geometries_for_cn`,
`vertex_dirs`, `isomer_permutations`, `is_planar` into `polyhedron.py` (add `import math`, `from
dataclasses import dataclass, field`). **Re-export all of them from `metal.py`** so no consumer changes
(`isomers.py`, `coordination_builders.py`, `solver.py`, `pipeline.py`, the three tests keep importing
from `.metal`). `_vertex_angle` must move with the data (else `polyhedron → metal` *is* the cycle);
`metal.label()` still needs it, so re-import it too. Leave `classify_geometry`/`geometry_for`/`n_sites`/
`coplanar` in `metal.py` (they take an rdkit `Mol` — moving them would give the rdkit-free
`polyhedron.py` an rdkit dependency; cycle-safe but changes the module's character).

**Risk:** moving **golden geometric data** across a module boundary. Import graph is verified safe;
consumers are shielded by the re-export. The real risk is a mechanical slip during the move.

**Golden-validated?** **Yes** — `test_sphere_solver.py`, `test_metal_chelates.py`, `test_haptic.py`,
metal-isomer suite. Same gate as 1.2.

**Order:** this is the **head** of the `metal.py` chain. If yes: **move → de-clunk (1.2) → de-verbose
metal.py (1.3 #2)**, serial, so the POLYHEDRA block is relocated verbatim first, cleaned in its new
home second, and the shrunken `metal.py` de-versed last. If no: skip to de-clunk-in-place → de-verbose.

---

## PART 3 — DO NOT ATTEMPT (measured-refuted / logically refuted)

1. **Fully group/period-scaling the fold FLOOR / WINDOW** (2.1 option a). Domain-fatal: `(N, sp)` floor
   +12.6° false-flags real nitrile crystals; `(C, sp3)` −18.4° weakens the gate. Repeats the
   twice-reverted "global fit hides a per-row regression" pattern. The median may be scaled *only* under
   the hybrid (c), and only if row-count reduction is the explicit goal.
2. **Generalising `_CONJUGATING_LP` to "period-2 p-block with a lone pair."** Admits F/Ne — an untested,
   physically meaningless widening (F/Ne have no substituents to pyramidalise) the census never
   validated.
3. **Dropping the `period ≤ 3` cap on `_LONE_PAIR_Z`.** Silently admits transition metals V/Cr/Nb/Mo
   and untested heavy donors Se/As/Sb/Te. The cap is the validated boundary, not the group membership.
4. **"Completing" the POLYHEDRA angle lists during the move/de-clunk.** `metal.ANGLES` states a
   *minimal* vertex-pair subset by design (oct 6 of 15); all-pairs is **refuted**
   (`under-determined-polyhedron-angles` memo — the "17°→7°" was a pooled-wrong-isomer artefact,
   all-pairs made fidelity *worse*). Move/hoist the angle tuples **verbatim**; do not add pairs.

---

## PART 4 — Validation gate (after phases 2-3)

- Metal / isomer / haptic / sphere-solver / chelate suites green (guards 1.2, 2.3, and any 2.1(c) or
  2.2-adoption).
- Fold / donor-orientation census regression unchanged (guards 2.1).
- `just check` (lint + type + test) — the de-verbose sweep must keep `ruff D` happy and all symbols
  importable.
- Re-run the four fixed notebooks; each `geom.check(...).assert_ok()` passes (guards 1.1). Clear
  outputs before a lightweight commit.

Independent tracks that can proceed without waiting: **1.1 notebooks**, **1.3 pipeline.py +
isomers.py**. Everything else funnels through the two single-file serial chains (`metal.py`,
`donor_orient.py`) whose heads are decisions 2.3 and 2.1.
