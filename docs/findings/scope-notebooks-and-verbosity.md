# Scope: notebook breakage audit + docstring/comment verbosity map

READ-ONLY scoping pass. Nothing was edited. All findings verified against the current working tree
(`branch rdkit-embed-kernel`) by live import in the `uv` env and AST analysis.

---

## JOB A — notebook breakage audit

### Where the moved symbols now live (verified)

| Symbol | Old path (stale) | Current path (verified live-import) |
|---|---|---|
| `metal_indices` | `rxembed.constraints.metal` (module deleted) | `rxembed.rdkit_embed.constraints.metal` |
| `_xyz_to_mol` | `rxembed.embed.dispatch` | canonical `rxembed.inputs`; **still re-exported** from `rxembed.embed.dispatch` (line 21 `from rxembed.inputs import _xyz_to_mol`), so the old import path STILL RESOLVES |
| coordination builders | (n/a) | `rxembed.rdkit_embed.constraints.coordination_builders` |
| `POLYHEDRA` table | (n/a) | `rxembed.rdkit_embed.constraints.polyhedron` (re-used in metal.py / coordination_builders.py / isomers.py) |
| `active_blocks` (dedup) | `rxembed.dedup.active_blocks` (renamed away) | `rxembed.dedup.active_feature_kinds(mol, ids, nci=True)` — renamed by commit `8337574` "T14: rename for clarity" |
| `rx.metal` | — | public: `rxembed.isomers.enumerate_isomers` (aliased in `__init__`) |

`metal_indices` itself is alive and unchanged at `src/rxembed/rdkit_embed/constraints/metal.py:575`.

### Per-notebook result (8 in scope)

**BROKEN — 4 of 8:**

- **`07_metal.ipynb`** (cell 1) — `from rxembed.constraints.metal import metal_indices`
  → `from rxembed.rdkit_embed.constraints.metal import metal_indices`.
  Also imports `_xyz_to_mol` from `rxembed.embed.dispatch` (works, see note below).
- **`henry.ipynb`** (cell 1) — same broken `metal_indices` import → same fix.
- **`karoline.ipynb`** (cell 2) — same broken `metal_indices` import → same fix.
  The maintainer's own dirty diff has **captured the live failure in the output cell**:
  `ModuleNotFoundError: No module named 'rxembed.constraints.metal'` at `In[2], line 7`. This is
  exactly the error the task describes; it is real, not stale output.
- **`01_basics.ipynb`** (cell 11) — NOT an import; an attribute call:
  `dedup.active_blocks(ens.mol, ens.ids)` → `dedup.active_feature_kinds(ens.mol, ens.ids)`.
  `active_blocks` exists nowhere in `src/` anymore; the current function has the matching
  `(mol, ids, nci=True)` signature. This break is **pre-existing** (from the T14 rename commit), NOT a
  maintainer edit — `git diff` for that line is empty.

**FINE — public `rx.*` API only (no internal reach):**

- **`02_constraints.ipynb`** — only `rx.embed(..., constrain=/fix=).mc().prune()/.minimize()`. All verified present.
- **`03_nci.ipynb`** — only `rx.nci_candidates`, `rx.nci_modes`, `rx.embed(..., contacts=)`. All present.
- **`09_assemble_ts.ipynb`** — `from rxembed import geometry as geom` (public shell module, exported in
  `__init__`) + `rx.embed(fix=)`. Fine.

**FINE, BUT REACHES INTO AN INTERNAL MODULE (works via re-export; not broken):**

- **`06_organocatalysis.ipynb`** (cell 1) — `from rxembed.embed.dispatch import _xyz_to_mol`. Resolves today
  because dispatch re-imports it from `rxembed.inputs`. Canonical home is now `rxembed.inputs._xyz_to_mol`;
  optional repoint, not a break. (07_metal / henry / karoline import it the same way.)

### Public-API calls used in the notebooks — all verified INTACT

`rx.embed(metal/fix/constrain/template/contacts/coordinate/charge/n/seed/knowledge/stereo=)`,
`rx.metal(smi, geometry, center=, fix=, stereo=)` (→ `enumerate_isomers`), `EnsembleSet.select(index=/chirality=/arrangement=)`,
`.summary()`, `.mc()`, `.prune()`, `.minimize()`, `.score()`, `.representatives()`, `.landscape()`, `.align()`,
`.lowest()`, `.dump()`, `dedup.apply`, `geom.check(...).assert_ok()`. None require changes.

### Two traps for the fixer

1. **The `rxembed.constraints.metal | ...` strings in the output cells are LOGGER NAMES, not import paths.**
   The logger is deliberately pinned to `rxembed.constraints.metal` (per project memory: loggers pinned to
   `rxembed.*`) even though the module moved to `rxembed.rdkit_embed.constraints.metal`. Do **not** "fix" those
   log strings — only the one `from rxembed.constraints.metal import metal_indices` code line is broken.
2. **All 8 notebooks are git-dirty (maintainer re-runs + edits).** Diff sizes:
   `01_basics 237/14`, `02_constraints 501/22`, `03_nci 822/18`, `06_organocatalysis 1/1`,
   `07_metal 809/784`, `09_assemble_ts 123/131`, `henry 20161/22282`, `karoline 8272/616`.
   A fix must be **surgical** — edit only the single broken code line per notebook (via `NotebookEdit` on the
   exact cell) so the maintainer's re-run outputs and code edits are preserved, never a checkout/revert.
   The maintainer's diffs do NOT already fix the stale import (only capture its failure), so the fix is still needed.

### Out-of-scope notebooks (not in the 8, not git-dirty) — for completeness

`04_organic_ts`, `05_templated_ts`, `10_retarget_ts` each also use `from rxembed.embed.dispatch import _xyz_to_mol`
(works via re-export). None carry the broken `constraints.metal` import. `08_energies` is pure public API.
The `metal_indices` break is confined to the three metal notebooks, all in scope.

---

## JOB B — docstring / comment verbosity map

Measured by AST over `src/rxembed` (kernel + shell): docstring chars/lines + consecutive `#`-comment blocks.

### Top modules by combined prose weight (doc chars + comment chars)

| # | Module | LOC | doc chars / lines | comment chars / lines | TOTAL |
|---|---|---:|---:|---:|---:|
| 1 | `pipeline.py` | 1689 | 32791 / 458 | 9092 / 99 | **41883** |
| 2 | `rdkit_embed/constraints/metal.py` | 1047 | 20031 / 268 | 3721 / 36 | **23752** |
| 3 | `rdkit_embed/constraints/donor_orient.py` | 374 | 11931 / 142 | 5949 / 54 | **17880** |
| 4 | `isomers.py` | 769 | 12621 / 173 | 2901 / 29 | **15522** |
| 5 | `rdkit_embed/constraints/mechanisms.py` | 454 | 11156 / 150 | 3291 / 35 | **14447** |
| 6 | `rdkit_embed/coordination.py` | 411 | 11864 / 157 | 724 / 8 | 12588 |
| 7 | `rdkit_embed/constraints/distance.py` | 319 | 6640 / 89 | 5586 / 53 | 12226 |
| 8 | `constraints/nci.py` | 549 | 8739 / 131 | 2314 / 25 | 11053 |
| 9 | `geometry.py` | 384 | 6795 / 100 | 2639 / 26 | 9434 |
| 10 | `embed/dispatch.py` | 787 | 5980 / 86 | 3199 / 34 | 9179 |

`donor_orient.py` (#3) and `distance.py` (#7) stand out for **comment-heavy** ratios: ~16 chars of `#`
comment per line of code — the highest prose-comment density in the tree.

### Heaviest individual docstrings (prime cut targets)

| Lines | Module::symbol | Note |
|---:|---|---|
| 54 | `pipeline.py::embed` | multi-paragraph return-shape narrative + relax-seam essay; the single biggest docstring |
| 35 | `rdkit_embed/coordination.py::donor_orientation` | |
| 28 | `rdkit_embed/constraints/donor_orient.py::_coplanar_donor` | private fn, essay-length |
| 28 | `rdkit_embed/constraints/sphere.py::<module>` | |
| 25 | `rdkit_embed/constraints/builders.py::<module>` | |
| 24 | `rdkit_embed/coordination.py::metal_overbond` | |
| 24 | `rdkit_embed/constraints/mechanisms.py::<module>` | Why-co-located / Split-by-FIELD / Phases essay (see load-bearing note) |
| 23 | `pipeline.py::mc` | |
| 21 | `rdkit_embed/constraints/donor_orient.py::_orient_donor` / `donation_axis` | two more essay-length privates |
| 21 | `pipeline.py::Ensemble` (class) | |
| 20 | `rdkit_embed/constraints/mechanisms.py::Sp2Planar`, `pipeline.py::_relax_into_windows`, `polyhedron.py::<module>` | |
| 19 | `donor_orient::_stripped_hybridisation`, `mechanisms::ConjugationCap`, `isomers::enumerate_isomers`, `geometry::<module>` | |

### Heaviest multiline comment blocks (>=3 consecutive `#` lines)

| Lines | Location | Opening |
|---:|---|---|
| 11 | `geometry.py:53` | "when its realised separation drops below this x their covalent-radii…" |
| 9 | `donor_orient.py:34` | "--- the donor-fold census: what `_orient_donor` enforces…" |
| 9 | `distance.py:57` | "--- the M-donor bond length ---" |
| 9 | `distance.py:305` | "…and the SAME per-tier distance (`_tier_floor`) handed to the bounds…" |
| 9 | `sphere.py:40` | "Residual weights. Each sigma is a DIVISOR…" |
| 8 | `stereo.py:240` | "Build the enumeration graph by DISCONNECTING each metal…" |
| 8 | `mechanisms.py:407` | "The one place mechanism order is stated…" (the `MECHANISM_ORDER` rationale) |
| 7 | `donor_orient.py:75 / :18`, `base.py:66`, `mechanisms.py:262` | census tables + cap narratives |

Also note the **module-constant inline-comment paragraphs** in `mechanisms.py:47-79`
(`_SOFT_PULL_FC`, `_COPLANAR_FC`, `_SP2_HOLD_FC`, `_SP2_HOLD_WIN`, `_CONJ_CAP`): each constant carries a
3–5 line wrapped `#` comment. Some of these are load-bearing (see below).

### Prioritised de-verbose list

1. **`pipeline.py`** — by far the biggest sink (42k prose chars). Cut `embed` (54L → 1-line summary +
   the return-shape table belongs in prose docs, not the docstring), `mc` (23L), `_relax_into_windows` (20L),
   `prune` (18L), `Ensemble` class (21L). Highest ROI.
2. **`rdkit_embed/constraints/metal.py`** — 268 doc lines over 1047 LOC; broad trim of function docstrings.
3. **`rdkit_embed/constraints/donor_orient.py`** — highest *comment* density; the `_coplanar_donor` (28L),
   `_orient_donor` (21L), `donation_axis` (21L), `_stripped_hybridisation` (19L) private-fn essays + the
   census `#` blocks at :18/:34/:75.
4. **`isomers.py`** — 173 doc lines; `enumerate_isomers` (19L), `_donor_faces_metal` (16L) + siblings.
5. **`rdkit_embed/constraints/mechanisms.py`** — the 24L module essay and subclass docstrings
   (`Sp2Planar` 20L, `ConjugationCap` 19L, `Frozen`) are prose-heavy — **but see load-bearing carve-outs.**
6. (runners-up) `coordination.py`, `distance.py`, `nci.py`, `geometry.py`, `embed/dispatch.py`.

### Load-bearing — do NOT cut

- **No doctests exist anywhere** in `src/rxembed` (AST-verified: zero docstrings contain `>>>`). So there is
  no doctest to accidentally break — but also nothing to specially preserve on that count.
- **The `Mechanism` base-class hook contracts** in `mechanisms.py` — `dg_windows` / `dg_relief` / `dg_post` /
  `ff_terms` — are the **single source of the phase-hook contract** that `ruff` `D102` is deliberately
  disabled for (`per-file-ignores: "src/rxembed/rdkit_embed/**/mechanisms.py" = ["D102"]`, with an explicit
  rationale comment in `pyproject.toml`: subclass overrides intentionally omit docstrings *because* the base
  documents the contract once). These four are already one-liners — **keep them**. Do not let a de-verbose
  pass "add missing docstrings" to the subclass overrides either; that would reintroduce the duplication the
  exemption exists to prevent.
- **The measured-refutation constant comments** in `mechanisms.py` (`_COPLANAR_FC`, `_SP2_HOLD_FC`,
  `_SP2_HOLD_WIN`, `_CONJ_CAP`, `_SOFT_PULL_FC`) cite the exact pinning test
  (`test_reembed_retry_delivers_clean_geometry`), the swept-and-refuted ranges, and the findings docs. These
  are the load-bearing "why" the standard says to keep — the value encodes a settled experiment (project
  memory repeatedly warns "measured-refuted — do not re-explore"). Trim the *prose*, but keep the pin: the
  number, the test name, and the one-clause reason it can't move. Same caution for the `mechanisms.py:262/407`
  ordering comments (they explain why fields, not sources, drive the DG order — a real correctness constraint).
