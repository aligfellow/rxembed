# rxembed: correctness, then simplification, then carve

`CLAUDE.md` is what rxembed is. `AGENTS.md` is how we change it — read it first.
This is the ordered work.

---

# → CURRENT WORK: the `rdkit_embed` extraction — ORCHESTRATOR, START HERE

You are the orchestrator for the `rdkit_embed` extraction. The full design + steps are settled in
**`docs/rdkit_embed-migration-plan.md`** (with the package README draft `docs/rdkit_embed-README.draft.md`).
Your job is to **sequence the work and fold short agent reports — never to read source, run tests, or review
code yourself.**

## The one rule
**Delegate every check.** You never open a source file, run pytest, grep, or audit code in your own context —
you dispatch a **fresh-context agent** with a narrow brief and receive a short verdict. If you catch yourself
about to Read/Grep/run something to "verify," stop and dispatch an agent instead. Your context holds only the
plan and a ledger of what's done — keep it clean so no stage biases the next.

## What to run
Execute the Moves in order — **Move 1 → 1.5 → 2 → 3 → add-geometries → README (last)** — each through the
per-step loop in the plan doc, every stage a fresh-context agent:

  implement → break-it ∥ four review lenses (KISS/YAGNI · arch/naming/location · user+dev clarity · rdkit-native)
  → test/regress → fold → next Move.

Read the plan doc for each Move's exact contents; do not re-derive it. The design is **settled — do not
re-litigate** (the surface, the `Conformers` result, `Isomer(mol, code, sites)`, 3-letter geometry codes,
template→fix, contacts→constrain, the flat 13-module layout).

## Gates — a test-agent reports these; you never run them
- golden **23/23 BIT-IDENTICAL** (this migration moves & renames; it must not change physics — a golden move = a bug)
- full suite **383/2**, import-hygiene green
- Move 3 also: the **144-structure OIN corpus**, 72.9% / median 0.211 Å unchanged

## Conventions (put in every agent brief)
- **No AI co-author trailer** on commits; terse, factual messages; **no AI slop** anywhere (plan principle 8).
- Commit **only the agent's own files**; **never** stage the maintainer's dirty files (`examples/*.ipynb`,
  `playground/*`, `*.xyz`). The pre-commit hook stashes/restores those; that's expected.
- Move 1 lands behind **re-export shims** so old paths keep working; Move 2 deletes them. Each Move is one
  reversible, golden-gated commit.

## Operating gotchas (learned this project — hand to each agent)
- Agents **stall on a buffered `pytest -q`** (they background it then yield). Brief each implement/test agent to
  run the suite and **wait for it, reading the result before committing**. If one stalls mid-suite, wait for the
  run to finish (watch the process / result file), then let it — or a fresh agent — commit.
- **Never run two pytest suites against the tree at once** — concurrent runs corrupt results. Serialize
  test-running agents; read-only review agents may run in parallel.
- Respect the machine: keep a sensible concurrency cap (confirm with the maintainer — ~4 has been the norm; heavy
  pytest one at a time).

## State
Branch `rdkit-embed-kernel`. The plan + README draft are committed; **no extraction code exists yet** — Move 1 is
the first code. The three consumers to land on the surface: `rxembed` (this repo), `../OIN-SMILES/oin_adapter`,
`../rxnts` (confirm its call sites first).

---

Everything below was measured 2026-07-19/20 by an adversarially-verified assessment, against
**xyzgraph 1.6.14** (pinned in `pyproject.toml`; the corpus numbers do not reproduce on 1.6.12).
Claims refuted during that assessment are marked; do not resurrect them.

---

## Status — 2026-07-23 (overnight session)

All three phases (T1–T16) are DONE. The embed kernel is carved **in-place** as the `rxembed.rdkit_embed`
subpackage (`src/rxembed/rdkit_embed/`, numpy+rdkit only); the uv-workspace/`packages/` split was tried then
**unwound** (`cf338d5`). Beyond T16 this session also landed (full suite **381 passed / 2 skipped**, golden
**bit-identical** throughout):

- **M–L always-connected** (`521505d`) — the user-facing `Ensemble.mol` always carries the real metal element
  + oxidation state + M–donor DATIVE bonds; the internal `_mol` stays the bond-less surrogate the engine needs.
- **xyz_to_mol → shell leaf** (`fb2bd40`) — `_xyz_to_mol`/`parse_smiles` moved from the kernel `io.py` to the
  new shell leaf `rxembed.inputs`; **xyzgraph left the kernel closure**.
- **OIN adapter repointed + corpus-validated** (`b6379e4` / `003084d`) — `oin_adapter/` is the OIN-cxSMILES→embed
  drop-in; **105/144 pass@1.0 Å, median RMSD 0.211 Å, no adapter bug**; σ-aryl fixtures both pass (0.13 Å).
  `docs/findings/oin-corpus-results.md`, `oin-integration-recon.md`.
- **Maintainability review v2** (`1f0031a` / `431d839` / `7341225`) — 4 lenses; verdict **NOT over-abstracted**,
  the carve improved KISS, old F1/F2/F13 already fixed. Safe fixes applied (doc-drift + tiny cleanups).
  `docs/findings/maintainability-review-2.md`.
- **scipy→numpy** — numpy-replace **measured-REFUTED** (negative-KISS). `docs/findings/scipy-to-numpy-sphere.md`.
- **plan_rxnts_integration.md** — effectively **redundant** for OIN (rxnts wishlist; 2/3 asks withdrawn/committed).

**Open judgment-calls for the maintainer** (none applied — structural or a real decision):
1. Sphere solver **keep-vs-delete** (~340 lines, scipy-gated ~2% fallback, effectively unreachable). If deleted,
   drop the `sphere` extra too.
2. Move the coordination **builders** out of foundational `metal.py` (old F3; removes 5 lazy imports).
3. Collapse the 5 index-aligned **polyhedron tables** into one keyed record (old F6; keep ANGLES hand-authored).
4. plan_rxnts **Ask 2** (η¹-through-an-aromatic-C–H → auto-promote to η² or refuse) — niche, off the OIN path.
5. **Upstream OIN** `rxembed-seam`: OIN must lift `_accept` + cxSMILES perception off the doomed vendored
   `generation/rdkit_embed.py` (perception is on `ali-dev`, not `ag-dev`). Documented, not pushed into OIN.

Corpus for the metal numbers — **144 structures**, both halves:
`OIN-SMILES/tests/integration/tmQM/*.xyz` (103) and `OIN-SMILES/tests/fixtures/*.xyz` (41).
The perception fix changes 5, all in the tmQM half; it fires on **none** of the 41 fixtures, which
are clean on 1.6.14 (0 fail, 0 error). Numbers quoted as "of 103" below are tmQM-only — check which
half a claim covers before extending it.

Reproduction geometries: `playground/perception_diag/` and `playground/falsify/`.

## How to work this document

**One task, one agent, fresh context.** Spawn a subagent per numbered task, carrying only that
task's text plus the files it names. Do not load this whole document into a working agent, and do
not carry one task's exploration into the next — that is how context rots and how settled questions
get re-litigated.

**Exploration is its own task.** If a task needs a question answered before it can proceed, stop and
hand *that question* to a fresh agent. Its answer comes back as a short written finding, not a
transcript.

**Findings go in files, not in chat.** Write to `docs/findings/<topic>.md` — the measurement, the
command that produced it, the verdict. Then link it from the task here. If a finding changes the
plan, edit the task. If it does not, the file is still the record that stops the next agent
re-measuring it.

**The numbers here are the contract.** If your change moves one, that is the finding — report it
rather than updating the number silently.

---

## The shape of it

Three phases, in this order, because each unblocks the next:

1. **Correctness** (T1–T6). `rx.embed()` returns unrelaxed seeds. Until that is settled every
   quality measurement is unreliable — including ones taken to justify later steps.
2. **Simplification** (T7–T15). Eight functions at CC≥30 hold ~700 lines; two modules are past what
   a reader holds.
3. **Carve** (T16). The kernel boundary is **5 lazy imports from clean, all 5 in `metal.py`**, so
   T8/T9/T11 are the prerequisite, not a detour.

~4,600 code lines total. The problem is concentration, not volume.

---

## Phase 1 — correctness

### T1. `minimize` must report its drops — **DONE** (285 passed / 2 skipped)
`pipeline.py`. `minimize` discarded conformers without appending them to `self.discarded`, so
`prune`'s docstring promise "Nothing is lost" was false. Fixed by mirroring `prune`'s
capture-ids/set-difference idiom, placed to cover **all six** minimize gates, not just the energy
window. Regression test `test_minimize_records_its_energy_window_drops` (verified red without the
fix).

**First, because the missing report caused a misdiagnosis.** Over-pruning was investigated and found
*not real*: the dropped conformers are exploded UFF relaxes at +10,000 to +20,800 kcal/mol that the
energy window rejects independently. `_shape_intact` is not the culprit — disabling it entirely
gives an identical drop sequence — and is a deletion candidate pending spectator-only and non-metal
fixtures.

### T2. Seed quality vs cleanup — **DONE**, see `docs/findings/seed-vs-relax.md`
30 tmQM structures spanning CN 2–14 / 20 metals / 9 donor elements, 186 paired conformers,
`seed=0xF00D`, `n=8`. Scripts in `playground/seed_vs_relax/`.

| axis | seed | relax | crystal | help/hurt |
|---|---|---|---|---|
| M–donor MAE (Å) | 0.063 | **0.006** | 0 | 185/1 |
| bond-length MAE (Å) | 0.025 | **0.021** | 0 | 163/23 |
| donor fold (°) | 20.2 | **17.0** | 9.9 | 95/55 |
| angle-window violation (°) | 9.54 | **0.00** | 0 | 171/6 |
| heavy-atom RMSD (Å) | 2.365 | 2.377 | 0 | 96/90 (neutral) |

**The premise is confirmed, the feared consequence is refuted.** The raw DG seed genuinely does not
satisfy its own constraints and the relax is what enforces them — but the relax moves geometry
*toward* the crystal on every local axis and is neutral on the global one. **There is no measured
axis on which the relax degrades the answer**, so T3 proceeds as written.

**REFUTED — do not resurrect: "the donor fold is UFF-born, the DG seed is clean."** The seed's fold
(20.2°) is *worse* than the relaxed (17.0°); relax helps 95 conformers and hurts 55. The old census
holds only on a minority (COJKAO, AKILAJ, CSBRHB, NUDXOC, SOHMEJ, SIFJUO). Both are far from the
crystal's 9.9°. (This does not disturb the separate, still-standing `tol > 0` finding.)

**The answer is neither tighter bounds nor a lighter relax.** Of 1399 violated angle instances,
**85% have the realised 1–3 distance outside a bounds-matrix bound that already exists**, and
triangle-smoothing tolerance is 0.000 on all 30 — the constraints are already mutually realisable.
Tightening a bound the solver is not landing on cannot help. `knowledge=False` barely moves it
(14.65° → 14.01°), so ETKDG's torsion terms are not the cause either. **The gap is between the
bounds matrix and the coordinates ETKDG returns from it** — a new thread, recorded under "Also
outstanding".

Claim reconciliation: 15/20 square-planar is really **17/20**. NEWVOB 32.65° (claimed 30.3),
ZOPNOH 23.64° (26.8), XAWQUH 42.55° (48.2 — same phenomenon, exact figure not reproducible; the
original `n` was never recorded). The organic window violation reproduces as a class (0.208–0.432 Å);
the exact 0.318 Å molecule is not in the repo.

### T3. Relax at the Ensemble seam
`embed/dispatch.py::_embed_isomer` never calls `restrained_uff`. Seeds violate their applied
constraint windows — mean 9.54°, up to 42.8°; all 0.00 after relax. **17/20 square-planar crystals
look tetrahedral until `.minimize()`.** Not metal-specific: an organic `constrain=` window is
violated by 0.208–0.432 Å. Numbers and per-structure detail in `docs/findings/seed-vs-relax.md`
(T2), which also establishes that the relax degrades **no** measured axis — so routing through it
is safe.

**DONE — 288 passed / 2 skipped, golden snapshots untouched.** Per-conformer window satisfaction
**6/60 → 56/60** across 9 cases. Frozen-core graft still 0.00000 Å on all 8 fixtures. Conformer
counts unchanged everywhere: embed drops nothing, so **total attrition at embed time is impossible**
— the gates stay at `minimize()` where they were. Runtime 138s → 280s.

**The briefed seam was REFUTED — do not re-propose it.** Routing through `_relax`/`minimize()` was
built and rejected: `minimize()` restores the metal and drops the surrogate context, so consuming it
at embed time strips the rest of the chain — `embed().mc().minimize()` lost `_reembed_until_clean`,
the coplanarity gate, the donor-hand hold and the metal bond tolerance (henry Ni 11/11 geom-clean →
4/5, zero re-embed calls, openconf then searching a bond-less real Co UFF cannot type). It also
emptied ensembles at embed time: `rx.embed("CCCl", fix={(1,2): 2.4})` returned **zero** conformers,
because `bonding_ok` reads a deliberately-stretched TS bond as broken — killing the headline
TS-from-SMILES capability.

The seam is **`_relax_constrained` at `pipeline.embed()`'s single return**, via a new
`Ensemble._relax_into_windows()`. It carries the three things a bare `restrained_uff` skips
(stiffness ladder, `_hold_donor_chirality`, bonding/coordination acceptance) **without** the restore,
so `_metal` stays live and `_minimized` stays unset. Plus `_rescue_torn` (re-relaxes each torn
conformer at its own minimal sufficient stiffness, falls back to its seed — the global escalation
stops at the first rung leaving *any* conformer intact, which left henry 3/8 torn) and a
`_seeds_relaxed` flag so `minimize()` takes a single point instead of relaxing twice (the double
relax rode flat-bottomed windows onto their walls — the no-restoring-force pattern again).

**SCALE CHECK — corpus-validated, SHIP** (`docs/findings/corpus-regression-t3-t4.md`, 142 in-scope
structures × 3 seeds). Big near-uniform embed-stage win (window-gate 0.28→0.68, M–L MAE 0.064→0.013 Å,
angle-window max 16.5°→3.7°) that **converges to HEAD's answer after `minimize()`** (post M–L MAE
0.0055 vs 0.0056) — HEAD's `minimize` already did this relax; T3 just moves it earlier. Durable post
wins: conformer survival 72.6%→78.3%; **rescues 1 silent zero-conformer failure (TUXRUZ 0→1)** and 4
empty ensembles (HgI3, RERHEB, WELROW, ZAHZUD); **introduces zero new failures** (failure set is a
strict subset of HEAD's). The `bonding_ok(constrained=)` exemption (see T3b) masks **no** bond tear:
of 834 exempt pairs, 0 are real bonds. One delivered-geometry regression, contained not a class:
**WUVJAB** (M–L MAE +0.025 Å, fold +7.7°); plus SOHMEJ rmsd +0.39 and a clash tradeoff on
ZAHZUD/WUVJAB/WELROW (mostly from the rescued-from-empty structures). Per-conformer gate rate still
rises. Scripts in `playground/corpus_regression/`.

### T3a. Organic seed-vs-relax — **DONE**, see `docs/findings/seed-vs-relax-organic.md`
Scripts in `playground/seed_vs_relax_organic/`. **Verdict: the relax stands as-is. No gate needed.**

**The thiourea claim reproduces exactly** (bimp seeds 1–3: gate clean 4/4, 3/4, 4/4 → 1/4, 0/4, 2/4;
conjugation 0.9–4.3° → 16.7–68.2°). Nothing in the concern was refuted.

**But its framing is REFUTED — do not treat T3 as a regression.** Run the documented
`embed → minimize` chain from the seed (the old pipeline) and from the relaxed seed (today) and they
land on the same geometry: **235/235 identical gate verdicts, conjugation difference 0.00°**.
`_relax_into_windows` calls `_relax_constrained(_DISTANCE_FC)` — the *same function* `minimize`
calls. **The relax was moved earlier, not added.** Where the paths differ at all (`salt-bridge`) the
new one keeps **two more** conformers. No measured degradation is damage T3 did to the pipeline's
product; it surfaces pre-existing damage one stage sooner, at the public surface every notebook gates.

That surfacing is real and worth knowing: `geometry.check` on `embed()`'s own output **44.6% → 30.3%**
over 1054 pairs (211 clean→FAIL vs 60 FAIL→clean), *not* seed-sensitive in aggregate (29.7–31.4%
over five seeds).

**What the relax buys:** angle-window violation 6.74° → **0.000°**, distance-window 0.0735 → 0.0032 Å,
bond-length MAE 0.0270 → **0.0179 Å improving on 23/23 cases** against real DFT geometries, frozen
core 0.0000 Å both sides, and a verified **byte-identical no-op on unconstrained embeds** (30/30).

**Also refuted:** "the seed was the cleaner geometry." It is cleaner *only* on conjugation and *only*
where a C=S is present — on `thia-ma` the seed (63.5°) is worse than the relax (56.4°) against a
reference at 16.6°. And a test comment claiming "embed's output is never worse than the seed on any
conformer" is false as written (true only as scoped to bond tearing).

The two defects this exposed are T3c and T3d below. **Neither is caused by T3.**

### T3c + T3d. The UFF thio-carbon defect — **BOTH DONE** (307 passed / 2 skipped)
Two orthogonal FF-only caps in `mechanisms.py`, both organic-only, both golden bit-identical:
- **T3d `Sp2Planar`** (`ddab708`) — holds each 3-neighbour sp2 carbon at its **seed** improper ±5°
  (hold-at-seed, never target-0). chb 0/8 → 8/8 planarity-clean.
- **T3c `ConjugationCap`** (`ccef4f7`) — **target-flat** `UFFAddTorsionConstraint` on the conjugated
  C=X–N quartet, cap=20° (inside the 30° gate), `fc=_COPLANAR_FC`. Schreiner thiourea twist 34–40° →
  <15°. Reuses `mechanisms.Coplanar`'s pattern; the gate and cap share
  `geometry.conjugated_quartets()` (agree-by-construction).

FF direction settled by `docs/findings/ff-handling.md` (`6ddf92f`): **MMFF is out** (can't type the
Li surrogate → `MMFFHasAllMoleculeParams` False; an organic-only fork breaks the one-chain design);
UFF's torsion-improper is the correct construct for the missing oop term; **openconf's
`transition_metal` preset** is a move-set bundle (hard-pin k=1e5), not an FF change — not adopted (it
would lose rxembed's construct-from-model coordination). The rest of this section is the historical
diagnosis, kept for the record:

**T3d collapsed these two into one defect.** `docs/findings/planarity-mode.md` +
`docs/findings/seed-vs-relax-organic.md`.

The conjugated thiourea/isothiourea/amidine **sp2 carbon** is mistreated by bare UFF, and it shows up
two ways on the same atom:
- **T3c (torsion):** 534/864 thiourea `S=C-N` quartets cross the 30° conjugation gate after the
  relax (vs 10 before) — the C-N torsion twists. Every C=O analogue *improves* (carboxyl 24.6°→2.4°).
- **T3d (improper):** `planarity` violations 15 → 281 — the sp2 carbon itself pyramidalises
  (chb-tetramisole C3 0.000 → 0.20 Å). ~256 of the 281 are this one carbon class.

**Diagnosed as a REAL UFF force-field defect, not a gate false-positive** (ablation: MMFF94s, GFN-FF,
GFN2 and DFT all keep the centre flat; only UFF puckers it; the 0.15 Å / 30° gates are well-calibrated
and every reference clears them). Seed-independent (100% of conformers, 5 seeds). It hits **the
organocatalysis thiourea motif this package targets**, so `rx.embed()` on those catalysts returns a
gate-failing geometry.

**Mitigating:** the final real-energy answer (`score`/`optimize` with g-xTB) is unaffected — g-xTB
keeps the centre flat. The defect is in the FF-relaxed seed geometry and the geometry-gate verdict,
not the deliverable energy.

**The "one cap clears both" recommendation is REFUTED by measurement**
(`docs/findings/preserve-planarity-fix.md`). The improper (T3d) and the conjugation torsion (T3c) are
different DOF on usually-different molecules (chb is pure T3d; bimp/takemoto/schreiner pure T3c), and
holding the carbon planar even makes bimp's conjugation *worse* (47°→65°). They are two fixes:

**T3d — DONE/IN PROGRESS (the cheap one).** The user's ETKDG insight holds here: the seed is already
flat (chb C3 = 0.003), only the relax breaks it. Fix is a **~22-line FF-only `Sp2Planar` mechanism**
in `constraints/mechanisms.py` + one `REGISTRY` entry: hold each 3-neighbour sp2 carbon at its own
**seed** improper ±5° (hold-at-seed, **never target-0** — a corannulene 9° bowl proves target-0
wrongly flattens a real pucker; hold-at-seed mathematically cannot). No `Constraints` field, no
`builders.py`, no bounds/DG change → **golden untouched** (FF-only, additive). Spiked through the real
pipeline: chb 0/8 → 8/8 gate-clean. Scoped via `geometry._coordinating_carbons` (excl. frozen /
metals / metal-coordinated C).

**T3c — DONE** (see the summary at the top of this section). It needed **target-flat**, not preserve,
because the conjugation seed is often already twisted (bimp 89.8°; 36% of quartets >10° at seed).
Landed as `ConjugationCap`. Note it clears **46 of 72** measured cases, not all — a real, partial win
that harms nothing; the residual tail is the FF-seed geometry only (g-xTB fixes the deliverable).

### T3b. `bonding_ok` contradicts the resolver — **DONE** (290 passed / 2 skipped)
An exact-number `fix=` populates `cons.distances` but **not** `cons.frozen`, so the gate called the
user's requested TS bond broken.

**The rule:** `bonding_ok` asks *"is this bond length chemistry-plausible?"* — a question that only
has meaning where chemistry, not an explicit instruction, sets the length. Where the constraint
system **states** a separation for a bonded pair, that length is the request and the radius rule has
nothing left to judge; whether the stated window was *met* is `check_constraints`/`.measure()`'s
question, not this gate's. Implemented as `constrained=` on `bonding_ok` taking `Constraints.distances`,
skipping pairs that are keys there. Per **pair**, not per atom.

**Rejected alternative — do not re-propose:** recording an exact-number `fix=` into `cons.frozen`.
`frozen` means *pin to embedded coordinates* — it drives the Kabsch graft, `mc` pose-freezing and
`_echo`, and `builders.py` documents a numbers-`fix` as explicitly "Not grafted, not pose-frozen."
That would change embed and search behaviour package-wide to fix a gate. It is also a set of
*atoms*, not pairs, so it reintroduces exactly the over-widening described below. The gate's
predicate was wrong; the resolver's record was right.

**The defect was live on the committed path, not merely under T3's abandoned seam** — `minimize()`
passed `exclude=self.cons.frozen` only, so `rx.embed("CCCl", fix={(1,2): 2.4})` went 1 conformer
after embed → **0 after minimize**.

**Deleting T3's `_rescue_torn` workaround was a strengthening, not a regression.** Its
`cons.frozen | constrained_atoms()` exemption was per-*atom* and far wider than its purpose: on
`rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")` it blanket-exempted the whole ethylenediamine
backbone (N3-C4, C4-C5, C5-N6) from the tear test — precisely the tear `_rescue_torn` exists to
catch. On `CCCl` and octahedral tris-acac the two predicates are identical.

### T4. Sphere crossover — **DONE** (291 passed / 2 skipped, golden did not move)
All three edits landed. `dispatch.py`'s 3-line cherry-pick **deleted** for
`cons = compose(cons, _metal.coordinate(iso, atoms))`; `metal.coordinate()` now relieves the phantom
floor by re-reading `overbond_tier` against the *augmented* donor set and reusing `_tier_floor` (DG
half only — the Li FF surrogate keeps its own vdW, unlike the DG's full-vdW carbon);
`constraints/base.py` splits `_merge_floor` (wall, `max`) from `_merge_relief` (relief, `min`).
Net −7/+40 (the excess is recorded derivation, per AGENTS.md on load-bearing why).

**Edit (3) verdict: `min()` is correct — the note stands, P0 was wrong.** The derivation, so this
is not re-litigated: `dg_floors`' only consumer, `mechanisms.Floor.dg_relief`, **only ever lowers** a
bound. So `floors` and `dg_floors` share a *number* but are opposite *mechanisms* — `floors` is a
**wall** the FF raises, `dg_floors` is a **relief** cutting RDKit's phantom ~3.4 Å bond-less-carbon
floor down to the real distance. "Take the stricter" is a category error on the relief side: the max
of two reliefs keeps more of the **phantom**, which is not physics at all. P0's commit message
asserted "a floor is a physical minimum" — true of `floors`, false of `dg_floors`.
The positive rule: two sources differ on a pair only because one saw a coordination the other did
not, and `_tier_floor` is monotone in exactly that (`APEX < NEAR < OUTER`) — **the lower claim is
always the better-informed one**, and it is bounded below by the bare covalent sum, so it can never
license a bond-length collapse. Now stated in the `compose` docstring.
Measured collision (`OCCCN->[Pd](Cl)Cl`): base 3.375 Å (stale OUTER) vs `coordinate()` 2.257 Å
(NEAR) — `max()` keeps 3.375 and the relief is a **no-op**, as predicted.

**The true crossover is 0.0443%, not 0.016%** (bisected). Both derive from the same 0.000545 Å; the
old figure divided by the 3.4 Å floor, but RDKit's `tol` is a ratio against the **repaired pair's own
lower bound** — here the 1.230 Å C=O. Same defect, different denominator. "2%" overstated it ~45×.
**After the fix the crossover is 0.0%** — the log line and the actionless
*"solving the sphere did not help (2% -> 2%)"* both gone. `_SMOOTH_LOOSE` and `max_tol` untouched.

**SCALE CHECK — corpus-validated, and it re-frames what T4 is** (`docs/findings/corpus-regression-t3-t4.md`,
142 in-scope structures × 3 seeds, three-tree worktree isolation). **T4 is INERT on the default
`rx.embed(xyz)` path — its corpus-wide crossover-relief claim is REFUTED, but it is a correct fix for
the `coordinate=` seating path and harms nothing.** A plain `rx.embed(xyz)` never passes
`coordinate=`, so `metal.coordinate()` is called **0 times** corpus-wide, `compose()` 0 times,
`dg_floors` collisions 0 — the +T3 and +T3+T4 result JSONs are **bit-for-bit identical** (0 differing
leaves). Bound crossover fires on exactly one corpus structure (TiCat2, 7.28%) and T4 leaves it
**unchanged** — that is a *genuine* contradiction, which is what the warning is for; the phantom T4
relieves is a different, seating-only floor. The relief DOES fire through the seating path
(`OCCCN->[Pd](Cl)Cl`: 22 `dg_floors`, seed Pt···C 3.100→2.880 Å) — reproduces directionally but
**costs one conformer there**. **Verdict: keep it.** It is gated by
`test_coordinate_relieves_the_phantom_floor_it_creates`, which correctly exercises the seating path
(`coordinate=o`) — no corpus fixture will ever catch a regression in it, so that test is the only net.
The seed-level tables below are therefore **seating-path measurements, not default-`embed` behaviour.**

`minimize()`d output barely moves (2.894 → 2.908 Å). At seed level (n=40, seating path):

| | seed Pt···C | seed Pt–O=C |
|---|---|---|
| before | 3.399 ± **0.001** Å | 170.0 ± 2.1° |
| after | 3.199 ± 0.133 Å | 142.8 ± 15.6° |

Every seed was pinned against the phantom with near-zero variance, forcing an sp2 carbonyl to ~170°
— a 50° distortion. After, the DG samples a real range and 31 → **40** seeds survive. The residual
142.8° (not 120°) is polyhedron/surrogate bias, **not** this defect.

**Golden did not move, and that is justified rather than lucky:** no golden fixture uses
`coordinate=` (all 13 go through `rx.embed(iso, n=2, seed=1)` with `atoms is None`), and
instrumenting the merge across all 13 found **0 cases where the two claims differ** — the only
situation where `min` and `max` can disagree. `min()` is provably inert on the baseline.

### T5. Derive conjugated sp2 donation — **DONE** (295 passed / 2 skipped, golden bit-identical)
`docs/findings/conjugated-sp2-donation.md`, scripts in `playground/t5_conjugation/`.

**The derived predicate: `geometry.conjugated_sp2_donor(mol, d, hyb=None)`** — a donor is conjugated
when it is sp2 (the two-estimator ruler the fold gate already uses) AND its donation axis is in a π
system (aromatic, or a conjugated bond to an sp2 heavy neighbour). Element-agnostic, no table. Lives
in `geometry.py` (perception); `metal._coplanar_donor` now defers to it, so enforcement and
perception can no longer drift. Deleted `metal._CONJ_DONORS`.

**`_CONJUGATING_LP` was KEPT — the "same list twice" premise is partly REFUTED.** It is genuine
period-2 lone-pair physics (only an N/O lone pair planarises into an adjacent π system; adding S/P
would type every PPh₃ sp2 and void the gate), and it sits on the lone-pair branch of
`_pi_hybridisation` that the aryl-carbanion and thione cases never reach. **One `{7,8}` derived away,
one correctly kept as data.** Do not try to delete it.

**`_CONJ_O_MDC_ANGLE` deleted — the note was right, and now confirmed WHY:** it was dead code.
`_orient_donor`'s fold wall already pins M-O-C for every calibrated `('O', SP2)` donor *before*
`_coplanar_donor` runs, so the `setdefault` was always a no-op — bit-identical on all fixtures. (The
"apparent gain came from the deleted wall" framing was imprecise; the real story is the wall was
redundant with the fold wall.)

**`('S', SP2)` census: UNCALIBRATED, gate abstains — as designed.** Across the 144 corpus the
thione-like subset S6 belongs to is **n=0** (all sp2-S = 14 records / 7 structures; gate-relevant
non-exempt = 8 / 4, a 47°-wide grab-bag). No `_FOLD_WINDOW` row added — fabricating one from too
little data is the thing not to do. Graceful degradation verified: the fold *gate/wall* abstain on
S6, but the coplanarity *cap* reads `cons.coplanar` only (decoupled from `_FOLD_WINDOW`) so it still
fires.

**Post-T3 re-measure (henry S6, 8 seeds × 8 conformers):** median fold 26.0 → 25.0°, **max 62.2 →
39.3°** (gross tail capped at ~census p95; the cap can't pull the median fully planar without a
calibrated wall floor, which n=0 denies it). The carboxylate O it ripples through *improved* 20.4 →
6.6° with its own constraints byte-identical — unregressed. Aryl-carbanion ipso C and thione S both
uncapped before (red-first), capped after.

### T6. The permutation warning — **DONE** (297 passed / 2 skipped, golden bit-identical)
The warning is now gated on **the count of distinct arrangements these donors admit on the polytope**,
not on a geometry-name table. Extracted the enumerator's own connectivity-aware dedup into
`_distinct_orderings(...)` and ran it over the full permutation orbit — its signature already
collapses symmetry-equivalent orderings, so distinct signatures == distinct arrangements. Warn only
when count > 1: linear → 1 (silent, any donors — **ferrocene no longer warns**); tetrahedral 4-distinct
→ 2 enantiomers (warns); tetrahedral homoleptic → 1 (silent); CN7/CN8 distinct → many (warns). The
**return is unchanged** (single identity ordering; `rx.metal(...)[0]` never raises) — only the log
line's condition changed, plus the shared helper. Short-circuit at `limit=2` makes every warn case
instant; precomputing the fixed vertex angle once dropped the homoleptic MoCl7 CN7 path 1.6s → 193ms.

**Separable improvement, reported not implemented:** `_distinct_orderings` now returns the actual
distinct orderings, so `rx.metal` *could* enumerate tetrahedral/CN7/CN8 isomers instead of warning —
a behaviour change needing its own feasibility filter and tests. Recorded under "Also outstanding".

---

## Phase 2 — simplification — **DONE** (354 passed / 2 skipped, golden bit-identical throughout)

Run as a golden-gated Workflow (`phase2_workflow.js`); detail in `docs/findings/phase2-progress.md`.
The maintainer **overrode the "split seven ways + re-export shim" plan as over-abstraction** — the
actual result is concern-based modules, **no shims** (callers import directly), and net-negative
source lines where dead code was found. New layout:

| module | lines | concern |
|---|---|---|
| `constraints/metal.py` | 2397→**1160** | coordination polytope + glue core |
| `constraints/distance.py` | **319** | M–L bond-length model + surrogate floors |
| `constraints/donor_orient.py` | **363** | donor perception + orientation holds |
| `constraints/solver.py` | **144** | the sphere-solver fallback |
| `isomers.py` | **751** | coordination-isomer enumeration (shell) |
| `geometry.py` | 1048→**410** | ground-state checks + `check()` orchestration |
| `report.py` / `vecmath.py` / `coordination.py` | 55 / 64 / 373 | Violation+GeometryReport / vector math / coordination gates |

- **T8/T9** broke the `metal↔geometry` mutual cycle and lifted the isomer cluster (metal.py now has
  **zero shell imports**). **T11** split metal.py into 3 coherent modules, not 7, no shim. **T12**
  split geometry.py into report/vecmath/coordination + kept `check()` under the public `geometry`
  name. **T13** was a **no-op** — the Ensemble/EnsembleSet dedup was already done by `_map`, and the
  worker correctly refused to collapse the thin verb delegators (that would hide the surface). **T14**
  renamed for clarity and **deleted** the dead `Mechanism.field` + two dead stereo helpers. **T15**
  inverted the import-hygiene test to a kernel allowlist (now catches nci/mc/xtb/calculator edges the
  old blocklist missed).
- The plan's line numbers / symbol lists were stale in every task (recorded per-task in
  `plan_stale`); the workers verified against current code before moving.

## Phase 2 — original task specs (superseded by the DONE summary above)

Each step gated by `just test` (297 passed / 2 skipped current baseline), in a git worktree.

> **Worktree caveat (from T7):** `isolation: "worktree"` created the worktree from a **stale base**
> (`20f2de2`, ~3,500 lines behind, no `tests/golden/`). The agent caught it (its suite was 115 tests,
> not 297) and fast-forwarded its branch onto current HEAD. **Any future worktree task must verify its
> base is current** (`git merge-base <branch> HEAD` == HEAD, suite count == 297) before trusting a
> result. Do not assume the worktree starts from HEAD.

### T7. Decompose the high-complexity functions — **DONE** on branch `t7-decompose`
The plan's CC numbers were from a different state. Re-measured with `ruff --select C901` on current
code and decomposed everything still ≥ 15 (each into meaningfully-named units, not metric-gaming —
`minimize` now reads as its five named gates; `enumerate_isomers` as its seven load-in stages):

| function | file | before | after |
|---|---|---|---|
| `enumerate_isomers` | metal.py | 35 | **12** |
| `resolve_core` | builders.py | 21 | **1** |
| `minimize` | pipeline.py | 21 | **8** |
| `_embed_isomer` | dispatch.py | 20 | **9** |
| `enumerate_unassigned` | stereo.py | 17 | **13** |
| `_embed_dispatch` | dispatch.py | 17 | **13** |
| `auto_binding_modes` | nci.py | 16 | **12** |

`solve_targets` and `restrained_uff` were **already < 15** on current code (T6's `_distinct_orderings`
extraction already lowered `solve_targets`), left untouched. Nothing had to be left un-split for
behaviour. **Honest ledger: net ≈ +169 source lines** — this buys per-function readability and a
regression ceiling (`ruff C901 max-complexity = 15`, now enforced in `pyproject.toml`), **not** fewer
lines. One genuine deletion: `resolve_core`'s write-only `n_fix_a`. Golden bit-identical, 297/2 in the
worktree, frozen-core holds, zero new `ty` diagnostics.

**MERGE STATUS:** `t7-decompose` = `cb9123c` + 7 commits, verified merge-base = `cb9123c`, touches 6
src files + `pyproject.toml`, does **not** touch `mechanisms.py` (so it composes cleanly with the T3d
`Sp2Planar` fix landing on main). Merge into main + full-suite reconcile pending T3d completion.

### T8. Fold census down: `geometry.py` → `constraints/metal.py`
Move `_stripped_hybridisation` (44), `donation_axis` (41), `_pi_hybridisation` (26) plus
`_FOLD_WINDOW`, `_FOLD_MEDIAN`, `_MAX_SIGMA`, `_CONJUGATING_LP`, `_METAL_Z`, **and
`_FOLD_WALL_FLOOR`** (`geometry.py:77` — leave it behind and the kernel→shell edge at line 841
survives).

Deletes 3 lazy imports and ~12% of `geometry.py`. Verified acyclic.

### T9. Isomer cluster up: new `rxembed/isomers.py`
`enumerate_isomers`, `IsomerSet`, `isomers`, `_order_label`, `_resolve_center`, `_input_ordering`
≈ **436 lines**.

`Isomer` and `arrangement` must **stay** — kernel `from_geometry` returns `Isomer`
(`metal.py:1490`) and `Isomer.summary()` calls `arrangement` (`1670`). Deletes the last 2
kernel→shell edges.

### T10. `from_geometry` should collapse haptic faces
`metal.py:1471` never calls `_collapse_haptic`, so raw ring atoms flow into `cons`, `vertices`,
`label` and `chirality_of`, and `Isomer.haptic` stays `{}` — **the transient-centroid mechanism
never fires on the `rx.embed(xyz)` path at all.** The haptic work in `26df6ca` applies only to the
`rx.metal(smiles)` → `enumerate_isomers` path.

Measured: `from_geometry(ferrocene)` → `geom=linear` but `donors=10, vertices=10, haptic={}`, `cons`
holding 10 distances and 45 angles across raw ring atoms.

Fix: call `_collapse_haptic` after `prepare`, mirroring `enumerate_isomers`; thread `haptic=` into
`chirality_of` and `Isomer`. Then the site derivation added by the sites/donors fix and its 6-line
comment both become unnecessary. Also folds the apical divergence — `from_geometry` passes
`geometry_for(len(sites))` where its sibling passes `has_apical=apical`; verified a CN4 piano stool
gets `square_planar` instead of `tetrahedral` whenever `classify_geometry` returns `None`.

### T11. Split `metal.py` seven ways
~1,798 lines after T8/T9 → constants 233 · chirality 129 · surrogate 200 · distance 230 ·
polytope 228 · solver 378 · isomers remainder.

Requires 4 relocations or you get two cycles: `from_geometry` + `coordination_from_geometry` →
isomers; `_chelate_bite_window` + `_ligand_pairs` + `_solve_start` → solver. Independently verified
acyclic including the 45 unassigned definitions. Keep `metal.py` as a re-export shim.

**This adds ~120–170 lines. Be honest about that** — it buys readability, not less code.

Fold in the four `/simplify` findings on the xyzgraph diff while the file is open.

### T12. Split `geometry.py` four ways
`types.py` (86 — `Violation` + `GeometryReport`; this is what breaks the report↔checks cycle) ·
`vecmath.py` (68) · `donor.py` (274) · `checks.py` (328).

### T13. De-duplicate `Ensemble` / `EnsembleSet`
`pipeline.py` (1,436 lines) re-implements the same stage logic twice. Extract the shared surface so
`EnsembleSet` maps over candidates without copying each method body.

### T14. Rename, then shrink the compensating docstrings
`REGISTRY` → `MECHANISM_ORDER` (the order is load-bearing; the name hides it) ·
`active_blocks` → `active_feature_kinds` · `metal.isomers` → `distinct_vertex_orderings` ·
`prepare` / `restore` → `surrogate_metal` / `restore_metal` · `geometry.stereo` →
`stereo_violations` · `stereo.relevant` / `passes` / `kinds`.

Only ~6 places are genuinely doc-heavier-than-code (`geometry.donor_orientation` 35/19,
`pipeline.embed` 47/48), ~120 lines. That is where the prose legitimately goes — not a mass trim.

### T15. Fix `tests/test_import_hygiene.py` by inversion
A kernel module may import only the 10 kernel dotted names. Deletes `_layer()` and the shell
enumeration. Measured: the current widened-`_SHELL` approach catches **2 of 10** injected edges —
`nci` (networkx), `mc` (openconf), `xtb`, `calculator` (ase) all pass silently because `_layer()`
returns `parts[1]`.

---

## Phase 3 — carve — **DONE** (365 passed / 2 skipped, golden bit-identical throughout)

### T16. `rdkit_embed` package — **DONE**
The embed kernel is now a **same-repo uv-workspace package** `packages/rdkit_embed/` (numpy + rdkit
core; scipy = `sphere` extra, xyzgraph = `perception` extra — both soft-guarded). `rxembed` is the
shell that depends on it (re-exports the kernel names its pipeline/shell use). **`from rdkit_embed
import embed, Constraints` closes over numpy + rdkit ONLY** — verified by a fresh-process `sys.modules`
check + static grep: zero `rxembed`/openconf/sklearn/networkx/xtb/ase/matplotlib. `import rxembed as
rx; rx.embed(...)` still works.

Kernel (moved to `rdkit_embed`): `io`, `log`, `report`, `vecmath`, `constraints/{base,builders,
mechanisms,metal,distance,donor_orient,solver,sphere,polyhedron,coordination}`, `refine/ff`,
`embed/bounds`. Shell (stays `rxembed`): `pipeline`, `dedup`, `refine/xtb`+`calculator`, `nci`, `viz`,
`embed/dispatch`, `isomers`, `stereo`, **`geometry` (the QA gate)**, `metrics`.

The plan's "5 lazy imports in metal.py block it" was stale (separability-assessment.md: **zero**
kernel→shell edges). The one real edge found was `mechanisms → geometry` for two **perception**
symbols (`conjugated_quartets`, `_SP2_DEGREE`) — a Phase-2 T12 residual left co-resident with the QA
gate. Resolved (maintainer's Option A, commit `6f723e0`): those two moved into `coordination.py`
(kernel), so `geometry.py` is now purely the shell QA gate importing FROM the kernel. Commits: STEP 0
`6f723e0`, layers `61cf638`/`c3b17de`/`d1f50d2`, workspace skeleton `f5db8b3`.
**Gotcha handled:** module loggers were pinned to `"rxembed.*"` names (not `__name__`) so `set_verbose`
+ the `caplog(logger="rxembed…")` filters survive the move (a real empty-echo regression surfaced and
was fixed mid-carve).

**Remaining, optional:** move `rxembed`'s own base deps (openconf/sklearn/networkx) to `rxembed` extras
to lighten the *shell* install — does NOT affect the `rdkit_embed` drop-in (already numpy+rdkit).

---

## Adversarial review + live-usage findings (2026-07-21)

The Phase-1 batch was adversarially reviewed and exercised live on real Ni organocatalyst complexes.
Findings and their status (detail: `docs/findings/review-batch.md`, `karoline-metal-embed.md`,
`coplanar-cap-softening.md`):

- **R3 — isolated sp2 metal donors lost the coplanarity cap** (T5 over-narrowed the predicate). **DONE**
  `ce2ce71`: `conjugated_sp2_donor` → `inplane_sp2_donor` (element-agnostic `sp2` test).
- **The epoxide (silent wrong-molecule) — DONE** (`8338de1`). The coordinated ester O-C-O collapse fused
  a 1-3 O···O to ~1.27 Å (perceived 3-ring), passing every gate silently. Closed with a new
  `geometry.over_compression` gate: a non-bonded 1-3 pair crushed below `_FUSE_RATIO=1.0 × Σr_cov` (the
  covalent sum — "closer than a bond can be") is flagged and reseeded away. Keys on the **graph**, not
  the angle (a genuine epoxide's terminals are bonded → never enters), so real 3-rings stay clean; the
  144-corpus gains no false flags. The cap fix below independently drops the collapse to 0/40 at source.
  (The floor-relief narrowing (b) is now moot — the cap fix removed the collapse; the min-merge's latent
  over-relief on coordinated esters remains a low-priority watch item, not chased.)
- **Cap over-flagging on crowded conjugated-donor chelates — DONE** (`3c57708`), from principle not patch.
  The maintainer rejected the crowding-conditional as a heuristic branch. The architecture investigation
  (`docs/findings/coplanar-cap-architecture.md`) found: **a plane is pinned to the metal by TWO coplanar
  contacts** (two M-D distances + the L-M-L bite angle — both already imposed by the polyhedron), so the
  coplanarity cap's improper is real information only at a **single** contact. When a co-donor of the same
  metal lies in the same conjugated sp2 plane (a conjugated bidentate), the pair already pins the metal
  and the cap is a redundant restatement. **FF-only** (the DG bound composes via INTERSECT and stays; only
  the FF torsion ADDs and fights): `geometry.codonor_in_plane` — a co-donor reachable through an all-sp2
  backbone path (reads only `_stripped_hybridisation`, no element list / π-flag / count) — and
  `Coplanar.ff_terms` skips its torsion. Case 2 59→22%, case 3 65→40%; henry/ketone controls
  bit-identical; skipped donors held ~2° by the backbone; epoxide 0/40; **golden bit-identical**. A
  deletion of a redundant runtime term.
- **Case 3's residual ~40% flags — OPEN, deeper.** After the cap fix, case 3's remaining flags are a
  *different* mechanism (floor-relief + a clash the diagnosis named), not the cap. Not chased.
- **R1 — guard-gap** (robustness): `minimize()`'s single-point branch lacks the `try/except RuntimeError`
  its sibling `_relax_constrained` has → crashes on an untypable core where the old path degraded. Small
  fix. Low reachability. **OPEN.**
- **R2 — retain-input `ids[0]` is now relaxed** (0.68 Å): maintainer says **keep relaxing (fair scoring)
  but make it CLEAR** (log/docstring). **OPEN.**
- **R4 — `Sp2Planar` flattens genuine bowls** (hold-at-seed pins a flat ETKDG seed flat; corannulene): the
  safety claim is refuted. Maintainer: **lower the force constant a little and TEST** the balance (chb
  thiourea vs corannulene bowl), and fix `pp_safety.py` to start from the flat seed. **OPEN.**
- **The thione ±45° coplanar window** (case 1): the metal scatters in/out of the S=C plane — the
  deliberately-wide `_COPLANAR_CAP`. Cosmetic; tighten or accept as part of the cap-softening work.

## Also outstanding

- **Warn on zero-conformer returns.** TUXRUZ and XAQDUS return the input geometry bit-identically
  with no warning and score #1 and #2 on every quality metric. A failed embed looks perfect.
  T2 re-confirmed both. **Worse than recorded:** on the default retain-input path (`keep_input=True`)
  `ids[0]` is *always* a bit-identical copy of the input, so any harness scoring `ens.ids` naively
  scores the crystal against itself. This is a live null-measurement trap for every future study.
- **The bounds matrix → ETKDG coordinate gap** (T2). 85% of violated angle instances have their
  realised 1–3 distance outside a bound that already exists in the matrix, at smoothing tol 0.000.
  The solver is not landing on bounds it has been given. `Ensemble._settle_seeds` (`pipeline.py:569`)
  is already a targeted window-only relax for exactly this, but runs only inside `mc()` — never on
  `embed().minimize()`. Promoting it is the obvious next experiment; **inferred, not tested.**
- **The relax's real cost is attrition, not accuracy** (T2): 50/236 seeds (21%) destroyed at
  `minimize()`. T3 confirmed embed itself drops nothing, so this cost did not move to embed time.
- **The empty-ensemble warning is still chelate-flavoured on organics.** T3b saw
  `0 of 1 conformer(s) survived the relax ... check the isomer/coordination requested` on plain
  `CCCl`. T3 replaced the warning's *guessed cause* with the gates that actually fired, so this may
  be a second call site or a residual tail — one grep, not an investigation.
- **`rx.embed("mn-h2.xyz")` retain-input rests on a mis-perception** (T3) — 10 ferrocene Cp carbons
  typed as a *linear* sphere, whose coplanarity gate can never be satisfied. Accounts for 2 of T3's
  4 residual conformers.
- **H-bond grips land on the window's upper wall** (T3): 40/41 within 0.02 Å of 2.20 Å. The
  no-restoring-force pattern (a flat-bottomed window has no spring, so the relax rides its wall)
  seen a third time now. Net motion is toward physical, so not a regression — but the pattern itself
  is recurring and may deserve one fix rather than three sightings.
- **`La` (Z=57) and `Sb` missing from `_METAL_GROUP`** — La–P falls back to bare covalent at 3.140 Å
  vs 2.947 (0.193 Å); Sb 0.250 Å on Pd–Sb.
- **`sphere.py` names a `'sphere'` extra that does not exist.** scipy is nowhere in `pyproject`; it
  arrives transitively via scikit-learn. Drop scikit-learn and the solver silently disables.
- **WIMCAA** — the one genuine tear in 103 tmQM embeds (the 41 fixtures were not scored for embed
  quality — that is an open gap, not a clean result); a nido-borane cage on Cd, a UFF/DG
  parameterisation failure, not perception (all 24 geometric B–B pairs are perceived).
  **Out of scope** — see Scope decisions.
- **WUVJAB** — T3's one delivered-geometry regression at corpus scale (M–L MAE +0.025 Å after
  minimize, fold +7.7°). Contained, not a class — the only structure that lost accuracy T3 didn't
  otherwise rescue. Worth a look after Phase 1 correctness; not a blocker for T3.
- **The 41 fixtures now have first-ever embed-quality baseline numbers** in
  `docs/findings/corpus-regression-t3-t4.md` (they were never scored before — the standing open gap).
  These are baseline, not a before/after comparison. Someone should sanity-read them.
- **Enumerate tetrahedral / CN7 / CN8 coordination isomers** (surfaced by T6). `_distinct_orderings`
  now yields the actual distinct orderings for these untabulated geometries, so `rx.metal` could
  expand them into candidates instead of warning. Needs a feasibility filter (CN8-distinct is 5040
  orderings to embed) and its own tests. A real capability, deferred deliberately.

## Scope decisions — recorded so they are not rediscovered

- **`dispatch.py:24` importing `pipeline` is not a violation.** The kernel is bounds, mechanisms,
  ff, sphere, base, builders, polyhedron, io, log. `dispatch` returns `Ensemble`/`EnsembleSet`, so
  it is *shell* — the kernel entry point is `bounds.embed` plus the `Constraints` struct. The old
  "P0-B dispatch↔pipeline cycle" task is therefore closed by definition, not by work. If T16
  disagrees when the carve is attempted, that is the finding.
- **The borane/carborane cage is out of scope** (maintainer, 2026-07-20). WIMCAA's tear and its
  total relax attrition are one UFF parameterisation failure — missing `B_5`/`B_6` atom types — not
  a perception or embed defect. It is a legitimate worst case to *cite* when a change's blast radius
  needs bounding; it is not a target. Do not special-case it, and do not let it shape a design.
- **The `embed_until` retry ladder is a non-goal.** It does not exist in the code and is not
  planned. Deferred deliberately: its rungs only improve the tail, and the failures they would
  catch are better fixed at their cause (T3 is one such cause).
- **The sphere solver is ported but effectively unreachable** — 246/251 smoothing calls land at
  `tol 0.0`, because the angle-intersect rule already absorbs the contradiction it repairs. Its own
  tests reach it by monkeypatch. Leave it; do not raise `max_tol` to 0.8 (that makes
  `rx.embed("CCO", fix={(0,2): 0.15})` silently embed a 0.195 Å C···O). T4 is about the *warning*,
  not the solver.

## Settled — do not reopen

- **SPY/TBP fidelity is fine.** 0/11 confusion, 0/46 on the 4-coordinate control, after relax. An
  ideal TBP misses the SPY rows by 30° against a ±8° window, so the template is *not*
  under-determined. "Add angle rows" is dead. (An earlier 22° figure was an index bug.)
- **Over-pruning is not real.** See T1.
- **The alleged WELROW P–Se tear does not reproduce.** Cloned to `ce59f34` with xyzgraph 1.6.12 —
  the artifact's own stated configuration — and ran 80 conformers: zero crushed bonds. The stored
  `.xyz` came from a code path that no longer exists.
- **The multiple-bond guard for xyzgraph's remaining ~5-per-100k cases is refuted** — it would
  revert GODNOD and ZOPJOG, whose bridges carry genuine C=S and C=O double bonds to their flanking
  donors.
- **The ring-id-map optimisation in `_prune_crosslinks` is not worth it** — 10× on synthetic worst
  cases, unreachable on real data, and a trial implementation silently diverged on NUDXOC and
  ZOPJOG.
- **All-pairs polyhedron angles** (`metal.py:160-180`) — measured 4.64° → 4.66°, no gain. Keep the
  numbers in the comment.
- **AGUFEN's κ²-N,P centroid is not a defect.** OIN's ±0.4 Å asymmetry cut deletes three real P
  donors; rxembed's donor set is right. A residual 0.243 Å scatter across three chemically identical
  bonds is real and **unexplained** — but it is not caused by `_centroid_constraints`, which is
  never called on this path (verified by spy wrapper; mean→RMS monkeypatch gives bit-identical
  output).
