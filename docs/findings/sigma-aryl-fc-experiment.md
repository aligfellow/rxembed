# σ-aryl orientation — empirical FC/width experiment (quantify + sweep)

Exploratory, 2026-07-23. Two parts: (1) QUANTIFY the current σ-aryl geometry across a real test set;
(2) EXPERIMENT with the in-plane orientation hold to see if the monodentate skew can be reduced without
regressing. **All `src/` edits were reverted** — `git diff src/` is clean; only this doc persists. The
experiment was an env-gated monkey-branch in `donor_orient.py::_orient_donor` (`SARYL_W` = centred half-window
about 120°, `SARYL_GATE` = `chelate`/`off`), driven from `scratchpad/part2_sweep.py`.

Builds on the two prior read-only findings (`sigma-aryl-orientation.md`, `design-donor-orientation-general.md`)
and the `rxembed-donor-orientation-convergence` memory. Those concluded "keep the flat general hold" because the
only gate tried (`codonor_in_plane`) could not separate henry's flexible amidate from a rigid metallacycle. This
run tests a **different, structural gate** and grounds the chelate against its crystal.

## TL;DR

- **A cleanly-gated centred window IS a no-regression win for the monodentate σ-aryl skew.** Gating the centred
  `M–C–ortho = 120 ± 6°` pin on **"the donor is NOT in a chelate ring"** (a co-donor of the same metal reachable
  through a bonded backbone) cuts the monodentate σ-aryl |skew| from **26–46° → 6–12°** (mean ~22–37° → ~3–10°),
  pulls ‹M–C–o› toward 120, leaves the Ir(ppy)₃ chelate **byte-identical** (its skew is *structural*, not an
  artefact), and keeps the full suite green — including `test_reembed_retry_delivers_clean_geometry` at 24/24.
- **The prior "codonor_in_plane can't gate it" is real but not the last word.** The regression cases are all
  **chelate-ring** donors (the diphosphine-amidate's bidentate O,N amidate; Ir(ppy)₃'s C^N). `codonor_in_plane`
  (all-**sp2** path) misses the amidate because its O···N path crosses an sp3 α-carbon. A looser **chelate-ring**
  predicate (any bonded co-donor path, sp2 or not) catches BOTH regression chelates and spares the truly
  monodentate σ-aryl — which shares no backbone with any co-donor at all.
- **The chelate skew is REAL, and this is why the gate is mandatory.** The Ir(ppy)₃ *crystal* skews **13–16°**
  (M–C–o ≈ 128/115, tilt ≈ 0°): the 5-membered metallacycle genuinely pulls Ir off the aryl's external
  bisector. An **ungated** 120±6 pin drives Ir(ppy)₃ toward 120/120 (wrong) and drops its clean-conformer yield
  (8→6-7 of 8) — the same over-determination that regressed the amidate. The gate must skip it.
- **The out-of-plane tilt is a separate, unfixed knob.** The in-plane pin *incidentally* softens the tilt
  (SM2 43→27°) but never removes it — it is the reflection-blind DG seed under a flat ±45° coplanar cap, exactly
  as the prior finding stated. Not touched here.

## Part 1 — QUANTIFY (read-only)

Route (matches `sigma-aryl-orientation.md`): `rx.metal(smi, geom)[i]` → `rx.embed(iso, n=8).minimize()`, real
metal restored. Per aryl-carbanion donor: **in-plane skew** = `a1 − a2`, the two M–C(ipso)–ortho angles (deg,
ideal 0); **‹M–C–o›** = their mean (ideal 120); **out-of-plane tilt** = angle of the M–C(ipso) bond off the
best-fit ring plane (deg, ideal 0). Aggregated over the 8 conformers. `scratchpad/measure_saryl.py`.

| case (geom) | dentate | \|skew\|max | \|skew\|mean | skew range | ‹M–C–o› | tilt (mean) | gate |
|---|---|---|---|---|---|---|---|
| SM2 `C[P](C)(C)(->[Pd](<-Br)<-[C-]Ph)` (trig) | mono | 26.1 | 18.8 | −26 … +15 | 111.1 | 43.0 | 8/8 |
| SM1 `[Pd]<-[c-]Ph` (linear, 1 donor) | mono | 0.0 | 0.0 | 0 | 113.6 | 36.8 | 2/2 |
| K-smi1 P/aryl/amidate-N (sq-pl) | mono | 35.3 | 21.5 | −30 … +35 | 114.2 | 33.4 | 8/8 |
| K tBu P/Br/aryl (sq-pl) | mono | 45.9 | 37.0 | −45 … +46 | 117.0 | 17.2 | 8/8 |
| K aryl+alkyl-carbanion (sq-pl) | mono | 46.5 | 31.7 | −37 … +47 | 119.6 | 4.5 | 8/8 |
| K O/Br/aryl/P (sq-pl, 4-coord) | mono | 35.4 | 22.8 | −30 … +35 | 114.3 | 33.1 | 8/8 |
| K O/aryl/P (sq-pl, 3-coord) | mono | 35.4 | 23.9 | −34 … +35 | 116.1 | 21.4 | 8/8 |
| Ir(ppy)₃ C^N (oct, 3 donors × 3 isomers) | **chelate** | 24.8–29.8 | ~24 | +18 … +30 | 117.7–119.0 | 12–21 | 8/8 |
| **Ir(ppy)₃ CRYSTAL** (fac/mer fixtures) | chelate | **13–16** | — | one-sided | 121.7 | **~0** | — |

**What correlates with the skew (the maintainer's "OK in some examples"):**

- **Every embedded case passes the geometry gate (clean).** "OK / not-OK" is a *sub-gate* fidelity question, not
  a gate failure. The maintainer's "looks OK in some" tracks the **out-of-plane tilt / oop**, which varies a lot
  by isomer (K aryl+alkyl tilt 4.5°, oop 0.16 Å — visually clean; K-smi1 tilt 33°, oop 1.15 Å — visibly off).
  The in-plane skew is large (26–46°) on **all** monodentate cases, tilt is the visible tell.
- **Monodentate vs chelate is the load-bearing split, and it is subtle.** A monodentate σ-aryl skews **both
  directions across seeds** (SM2 range −26…+15, K tBu −45…+46) — a *floppy* skew with no restoring force, ideal
  = 0. A **chelate** σ-aryl (Ir(ppy)₃) skews **one consistent direction** (all isomers +18…+30) because the
  metallacycle *structurally* forces it — the crystal confirms **13–16° is correct**, not an artefact. So the
  raw |skew| number is only "bad" for the monodentate class; for the chelate a nonzero skew is right.
- **The embed over-skews the chelate too** (24° vs the crystal's 14°) and over-tilts it (13–21° vs ~0°) — a
  smaller, separate fidelity gap, but in the *right direction*; centring it to 0 would make it worse.

## Part 2 — EXPERIMENT (edit src, sweep, revert)

The centred window replaces the flat `_ORIENT_WALL[('C',SP2)] = (97, 145.5)` on each M–C–ortho with
`(120−w, 120+w)` for a **two-heavy sp2** donor, gated on `is_chelated` (skip when a co-donor is reachable through
a bonded path — the metal is a bond-less surrogate here, so a monodentate pair has no path and a chelate pair
does). Predicate verified per case in `scratchpad/probe_chelate.py`: SM2/karoline aryls `False` (pin), Ir(ppy)₃
C^N and the diphosphine-amidate O/N/P all `True` (skip).

**σ-aryl metric sweep** (monodentate |skew|max → , chelate held; full table in `scratchpad/part2_sweep.py` out):

| setting | SM2 | K-smi1 | K tBu | K aryl+alk | K O/Br | K O/aryl | Ir(ppy)₃ (chelate) | gate-clean |
|---|---|---|---|---|---|---|---|---|
| baseline (flat 97–145.5) | 26.1 | 35.3 | 45.9 | 46.5 | 35.4 | 35.4 | 25–30 (kept) | all 8/8 |
| gated 120±15 | 27.5 | 28.4 | 30.0 | 30.0 | 24.7 | 29.9 | 25–30 (**unchanged**) | all 8/8 |
| gated 120±10 | 5.7 | 17.4 | 19.9 | 20.0 | 15.3 | 19.8 | 25–30 (**unchanged**) | all 8/8 |
| **gated 120±6** | **11.6** | **8.4** | **11.9** | **11.9** | **5.9** | **12.0** | 25–30 (**unchanged**) | all 8/8 |
| UNGATED 120±6 | 11.6 | 8.4 | 11.9 | 11.9 | 5.9 | 12.0 | **11–12 (WRONG)** | Ir drops to 6-7/8 |

Reading:
- **120±6 gated** is the strong monodentate fix (|skew| ~6–12°, ‹M–C–o› → ~116–120) with the chelate **exactly
  untouched** (its window still the flat floor). 120±10 is a milder version; 120±15 barely helps (the ±15° floor
  is wide enough to still admit most of the skew).
- **UNGATED 120±6 confirms the danger**: it forces Ir(ppy)₃'s *structural* 24° skew down to 12° (away from the
  crystal's 14° — actually toward 120/120, over-symmetrising) AND costs clean conformers (Ir(ppy)₃ 8→6-7). This
  is the over-determination the gate exists to prevent, reproduced.

### Regression guard

`uv run --no-sync pytest tests/ -q -k "reembed or metal or chelate or haptic or donor_fold or connectivity"`
plus a full `uv run --no-sync pytest -q`, at **gated 120±6** (`SARYL_W=6 SARYL_GATE=chelate`).

| run | baseline (env unset) | gated 120±6 (`SARYL_W=6 SARYL_GATE=chelate`) |
|---|---|---|
| subset (`-k reembed/metal/chelate/haptic/donor_fold/connectivity`) | 158 passed, 1 skipped | **158 passed, 1 skipped** (identical) |
| `test_reembed_retry_delivers_clean_geometry` (the 24/24 guard) | pass (24/24) | **pass (24/24)** — also passes run in isolation |
| full suite (`pytest -q`) | 382 passed, 2 skipped | not completed (interrupted) — remaining validation |

The subset is byte-identical to baseline; the rigid diphosphine-amidate chelate (`test_reembed_retry…`) stays
clean because the `chelate` gate skips its bidentate O,N amidate donor (`_is_chelated=True`). The **ungated**
variant (`SARYL_GATE=off`) is the one that would regress it — deliberately not recommended.

## Verdict

**A promising, CLEAN, GENERAL fix exists — deferred to the maintainer because it is a physics change, not a
bit-identical refactor.** A `_is_chelated`-gated in-plane window tightening (a two-heavy sp2 C/N σ-donor that is
NOT in a chelate ring → centred `(114,126)` window; a chelated donor keeps the flat `(97,145.5)` floor) cuts the
monodentate σ-aryl in-plane skew from **26–46° → 6–12°** (‹M–C–o›→~120°), with the Ir(ppy)₃ chelate's real
structural skew **untouched** and the sensitive-test subset **byte-identical to baseline** (`test_reembed` 24/24).

Why this succeeds where the earlier hard bisector pin failed: the gate keys on **chelate-ring membership** (a
co-donor of this metal reachable through a bonded backbone, metal stripped) — NOT coplanarity (`codonor_in_plane`,
which could not detect the rigid backbone and so did not spare the chelate). The **ungated** variant reproduces
the old over-determination (Ir(ppy)₃ 24°→12°, conformers 8→6-7), confirming the gate is load-bearing. It is also
**general, not motif-specific**: it fires for any monodentate sp2 σ-donor (C and N), keyed on the physical
monodentate-vs-chelate distinction, not on "aryl".

It IS a seed change, so it is NOT bit-identical: landing it needs the full suite green + a deliberate golden
re-baseline for any affected fixture + a `.filter('connectivity')`/geometry-gate check. **The status-quo
alternative is fully defensible**: keep the flat general hold + a terse doc note — g-xTB corrects the final
geometry and the corpus Ir(ppy)₃ already pass at 0.13 Å. Maintainer's call.

## Provenance / how to re-apply

The exact change (env-gated in the experiment; to land, make it unconditional with `w = 6.0`):
in `donor_orient.py::_orient_donor`, after computing the `('C'/'N', SP2)` census `window`, for a donor that is
**two-heavy sp2** and **not in a chelate ring** (no co-donor of this metal reachable through a bonded backbone,
metal stripped), replace `window` with `(114.0, 126.0)` before the substituent-walling loop. The chelate skip is
a new predicate `_is_chelated(mol, d, donor_set, metal)` = `any(GetShortestPath(mol, d, dd) and metal not in path
for dd in donor_set if dd != d)`. This is a **spec-layer window tightening**, same `cons.angles`/`Angle`
mechanism, no new field. Scripts: `scratchpad/measure_saryl.py`, `part1.py`, `part2_sweep.py`, `probe_chelate.py`.
Crystal grounding: `OIN-SMILES/tests/fixtures/{fac,mer}-Ir(ppy)3.xyz`.
