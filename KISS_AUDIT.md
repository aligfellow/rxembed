# KISS audit — rxembed

*A "keep it simple" pass over the whole `src/rxembed/` package (~5,000 LOC), against the project's own
stated principle (`CLAUDE.md`): "Adding a capability should be a registry row or one argument, not a branch;
dead machinery gets deleted; one struct in, one struct out" — while respecting "physical honesty over
cleverness" (genuine chemistry/geometry logic must not be simplified away).*

**Method.** The core / redesign files (`pipeline`, `embed/dispatch`, `constraints/builders`, `bounds`,
`base`, `mc`, `metrics`) were audited directly; the domain modules (`metal`, `nci`, `dedup`, `refine`,
`geometry`, `stereo`, `viz`) by two focused reviewers. Every "dead" claim in Tier 1 was verified by grepping
`src/`, `tests/`, and `examples/`. Findings are ranked by *simplification value ÷ risk*.

**Headline.** The codebase is disciplined. No dead *functions*; the `KINDS` registry genuinely drives a
generic NCI enumerator (a new kind is one row, no driver branching); the dense comments are almost all
load-bearing chemistry "why". The wins are a handful of **confirmed dead parameters / branches** plus some
**DRY tidying** — nothing structural is wrong.

---

## Tier 1 — Confirmed dead code, low-risk deletions

| # | Location | Finding | Fix |
|---|---|---|---|
| 1 | `refine/xtb.py` | **`grad=` plumbing is unreachable.** `grad=True` is passed nowhere in `src`/`tests`/`examples`; `_read_gradient` is self-referenced only; the `composite()` gradient arithmetic never runs. `singlepoint` returns `(e, None)` solely to keep a tuple shape. | Drop `grad` from `singlepoint` / `composite` / `energy`; return a scalar Eh; delete `_read_gradient`; `XTB.energy` drops the `[0]`. **Verify** it isn't deliberate forward-infra for an ASE force/optimizer path before deleting. |
| 2 | `dedup/select.py` | **`labels=` param + branches dead.** Callers (`apply`) never pass it, so `energy_prune`'s per-label dict, `descriptor_prune`'s forward, and `_DescriptorConfig.labels` / the `evaluate_sim` guard are all no-ops — a "same-binding-mode-only" gate that was never wired up. | Delete `labels` from both functions and the dataclass field; collapse `energy_prune` to one running list and `evaluate_sim` to the distance test. If mode-gated dedup is wanted, it's a `PLAN.md` item, not carried-but-unused code. |
| 3 | `constraints/metal.py:301` | **`real_z` param unused** in `coordination_from_geometry` — appears only in the signature; the body uses realised distances/angles from the conformer. | Drop the param and the one call-site argument (`metal.py:328`). |
| 4 | `constraints/metal.py:241` | **`n_sites` fallback dead / misleading.** `VERTEX_DIRS` and `ANGLES` have identical key sets, so `VERTEX_DIRS.get(geometry) or ()` is never empty for a valid geometry; the `or max(...ANGLES...)+1` arm never runs. | `return len(VERTEX_DIRS[geometry])`. |

**Net:** removes a full return-channel + file parser (`grad`), a dead gating feature (`labels`), and two
misleading params — real dead-surface, zero behavior change.

---

## Tier 2 — DRY / small refactors (low risk, real win)

- **`pipeline.py:470 / 493 / 557`** — three identical `Ensemble(mol, ids, self.cons, None, energies, True,
  tag=dict(self.tag))` constructions in `score()` (both branches) and `optimize()`. Extract one
  `self._scored(mol, ids, energies)` helper (the 8-line inline FF branch collapses to one line).
- **`dedup/features.py:46 / 258`** — the donor-list comprehension (heavy atom within `_M_DONOR_CUT` of the
  metal) is copied verbatim in `_metal_features` and `_metal_donors`. Have `_metal_features` call
  `_metal_donors` for `(m, donors)`, then build angles. Pure refactor.
- **`_frag_map` comprehension** (`{a: fi for fi, f in enumerate(GetMolFrags(mol)) for a in f}`) is duplicated
  at `metal.py:273/362/703/815` and `nci.py:192`, while `dedup/features.py:63` already has the named helper.
  Lift it to a neutral util (a `constraints → dedup` import is the wrong direction — likely why it was
  copied) and reuse.
- **Dead `cid=-1` defaults** on `metal.hold_shape` and `coordination_from_geometry` — `cid` is never passed
  non-default anywhere. Inline `-1` and drop the params (or keep if multi-conformer callers are imminent).

---

## Tier 3 — Verify with the maintainer (structure, or borderline-chemistry)

- **`dedup/features.py` — `_blocks` vs `active_blocks` encode block-presence twice.** `_blocks` decides which
  latent blocks exist by *materializing* each; `active_blocks` re-derives the *same* presence rules cheaply.
  Their docstrings admit they "must agree" — the metal-donor threshold, the fragment-count rule, and the
  metal-suppresses-nci/relpose rule are each encoded in two places and can drift. Unify onto one predicate
  (keep the cheap short-circuit path). Near-chemistry — worth a conversation, not a silent merge.
- **`constraints/metal.py:534` — `enumerate_isomers` is ~155 lines.** Extract `_load_input_mol(mol)`,
  `_resolve_donors(mol, metals, center, fix)`, `_resolve_geometries(geometry, n)`, leaving the orchestration
  loop. Structure-only, no logic change intended.
- **`constraints/metal.py` — duplicated triad-discovery** `_octahedral_triad` (351) / `_central_trans` (766)
  both rebuild a by-fragment grouping and hunt a length-3 chelate triad. Shared helper possible, but this is
  mer/fac chemistry — err toward leaving it.
- **`nci.py` — `inter_fragment` flag is always `True` internally** (threaded through three handlers), but it
  is a documented public knob on `nci_candidates` / `nci_modes`. Keep unless intramolecular NCI is off the
  roadmap; then drop the param and the four `inter_fragment and …` guards.

---

## Tier 4 — Keep (flagged, but justified)

- **`refine/calculator.py::ASE`** — constructed nowhere and lacks an `optimize()` (so it can't drive
  `Ensemble.optimize`), but it is a legitimate *public extension seam* (`resolve` accepts any `Calculator`;
  MACE / AIMNet2 / ORCA). **Keep, but add a smoke test** (a trivial fake ASE calc) so the seam is real; the
  missing `optimize()` is a latent sharp edge worth a docstring note.
- **Resolver `_warn_underdetermined` (~18 lines) + `_echo` (~30 lines)** in `constraints/builders.py` — ~48
  of the file's ~290 lines are non-fatal advisory / logging. Earned by DESIGN invariant §1 (the element-symbol
  echo that catches 1-based / wrong-atom indices) and the dumb-user review. Keep; a trim candidate only if the
  file ever needs slimming.
- **Minor, in the redesign files (leave):** the graft "capture ref-core → embed → `_graft_frozen`" pattern is
  near-duplicated between `_embed_isomer` and `_embed_dispatch` (the isomer variant carries an extra
  SMILES-metal-no-conformer branch, so a shared helper has small payoff); `_add_soft(cons, *_nci_windows(…))`
  appears in both dispatch paths (separable concerns — parse vs merge). `_embed_dispatch` at 147 lines is
  inherent to a dispatcher (the linear flow *is* the routing; the project deliberately ignores `PLR0915`).
- **Constant proliferation from `ruff PLR2004`** (`_DIST_ATOMS == 2`, `_ANGLE_ATOMS == 3`, …) — `len(idx) ==
  _DIST_ATOMS` reads worse than `== 2`, but that is a lint-rule tension, not a defect. Not worth relaxing the
  rule.

---

## Audited clean — do not simplify

`geometry.py` (kind-specific clash thresholds, N-excluded planarity, conjugation dihedral, frozen-core
Kabsch RMSD, metal-Z exclusion — all reached from `check()`, all exercised by tests) · `stereo.py`
(handedness-not-count logic is justified by xyzgraph instability; distinct in purpose from
`geometry.stereo()`) · `metrics.py` (`bonding_ok` — the single fast in-pipeline gate, distinct consumer/output
from `geometry`) · `viz.py` · `dedup/descriptors.py` · `refine/ff.py` (`_bond_pruned` recovery and
`restrained_uff`'s `AddFixedPoint` / angle-fc rationale are real chemistry; `extra_frozen` / `conf_ids` /
`angle_fc` all used by the metal donor-proton relax) · `embed/bounds.py` · `constraints/base.py` ·
`embed/mc.py`.

---

## Suggested order of attack

1. **Tier 1** — the confirmed deletions (most dead-surface removed, zero behavior change). Do `#3`/`#4`
   unconditionally; confirm the `grad` (`#1`) and ASE-force intent first.
2. **Tier 2** — safe DRY, tests as the gate.
3. **Tier 3** — one maintainer conversation each; `_blocks`/`active_blocks` unification is the highest-value.
4. **Tier 4** — no code change beyond adding the ASE smoke test.
