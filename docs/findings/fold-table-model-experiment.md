# Can the donor-fold FLOOR be replaced by a period/group/hyb MODEL? — an empirical fit + gate validation

**Verdict: NO for the floor (the gate); PARTIAL overall.** A richer, properly-fit period/group/hybridisation
model reproduces the **median** to ~4° but **cannot** reproduce the **floor** — the load-bearing safety gate —
without re-introducing the exact failure the project reverted twice. Every model tested (naive, rich with
group², period×group interaction, lone-pair term; OLS and ridge) has a max floor residual of **14–16°**, which
exceeds the table's own **5° calibration margin**, so each **false-flags the `(C, sp2)` calibration tail** (a
real ~90° carbon donation the measured floor of 85° was set to admit). Suppressing that with a global safety
margin (~9°) collapses the gate: **6 of 10 classes stop catching a 90° right-angle fold**. This confirms
`gen-fold-tables.md` with a richer model and quantifies it against the corpus.

This is a follow-up to `gen-fold-tables.md` (which refuted a naive scaling analytically). Here the census was
**rebuilt from the corpus with rxembed's own production perception**, richer models were fit and
LOO-cross-validated, and the FLOOR gate was validated for false flags against the tail the table actually
encodes. Read-only experiment; no src touched.

---

## 1. Rebuilt census (sanity check)

The M→D→X donation angles were recomputed over the 142-crystal corpus (`../OIN-SMILES/tests/integration/tmQM`
101 + `.../tests/fixtures` 41, the 2 borane cages excluded) using **production perception end-to-end**:
`rxembed.inputs._xyz_to_mol` (xyzgraph bond perception) → `coordination._spheres` (geometric donor perception,
`donors=None`) → `donor_orient._stripped_hybridisation` + `donation_axis` → `vecmath._angle`. This is the same
walk as `coordination._donor_walk`, minus its `_FOLD_WINDOW` membership filter so **every** `(element, hyb)`
class is captured. Result: **142/142 structures parsed, 607 donation angles**.

| class      | n   | rebuilt p0.5 | rebuilt p50 | rebuilt min | table floor / median / ceiling |
|------------|----:|-------------:|------------:|------------:|--------------------------------|
| (P, sp3)   | 191 | 99.9  | **115.8** | 94.2  | 89.0 / **115.8** / 138.5 |
| (N, sp2)   | 173 | 101.3 | 121.2 | 101.3 | 96.0 / 120.7 / 180.0 |
| (C, sp)    |  50 | 166.6 | **176.3** | 164.7 | 155.0 / **176.2** / 180.0 |
| (C, sp2)   |  50 | 111.7 | 120.1 | 111.5 | 85.0 / 124.4 / 145.5 |
| (N, sp3)   |  38 | 103.2 | 109.8 | 103.1 | 82.0 / 110.3 / 158.4 |
| (C, sp3)   |  34 | 110.9 | 126.9 | 110.8 | 104.0 / 113.3 / 133.8 |
| (S, sp3)   |  25 | 96.1  | **103.4** | 96.0  | 91.0 / **103.4** / 137.1 |
| (O, sp2)   |  15 | 109.5 | 128.6 | 109.0 | 90.0 / 125.7 / 156.8 |
| (N, sp)    |  10 | 145.1 | 177.0 | 145.0 | 140.0 / 179.3 / 180.0 |
| (As, sp3)  |   6 | 100.3 | **119.4** | 100.3 | 95.0 / **119.4** / 131.3 |

**MEDIAN reproduces well** — P/S/As/C-sp exact, the rest within ~1–4°: production perception agrees with the
original census on central tendency. **FLOOR (p0.5) does NOT** — the rebuilt p0.5/min sit **5–27° above** the
table floor for `(C, sp2)`, `(N, sp3)`, `(O, sp2)`, `(C, sp)`. Cause: my perception yields **607 angles vs the
original 721** (`gen-fold-tables`: 721 over 121 structures) — it **did not reproduce the low-angle tail
donations** the original census saw. The rebuilt census is therefore *cleaner* than the original.

> **Consequence for the gate test (critical):** the table floor is **at or below the rebuilt corpus min for
> every class**, so the *current table gives 0/607 false flags on the rebuild* — but so does any reasonable
> model, because the rebuilt minima carry a 5–27° buffer. Validating only against the rebuilt census is **too
> lenient** to discriminate. The honest test (Section 3) is against the tail the table actually encodes.

---

## 2. Fitted models (OLS + ridge, leave-one-out validated)

Features per class, mirroring `ml_distance`'s chemical-only style: `ideal(hyb)` (sp 180 / sp2 120 / sp3 109.5),
`period−2`, `group−14`, and for "rich": `+ (group−14)²` (a VSEPR lone-pair-crowding parabola, the analogue of
`ml_distance`'s d-electron `c3·g + c4·g²`) `+ (period−2)(group−14)`. Fit to the 10 **table** values (the gate
ground truth), 10 data points.

### MEDIAN — smooth, modelable (with one override)

- naive `median = 12.07 + 0.928·ideal + 2.13·(p−2) − 1.94·(g−14)`: in-sample RMSE **3.8°**, max 8.6°;
  **LOO RMSE 7.4°** (S,sp3 LOO residual −13.6°).
- rich (6 param): in-sample RMSE **1.2°** but **LOO RMSE 8.3°** — it *overfits* (fits S,sp3 to 0.0 in-sample,
  but LOO residual +16/−18 on O,sp2/S,sp3). Extra features do not help out-of-sample.
- The one stubborn point is `(S, sp3)` (103.4°, ~8–13° below its smooth prediction — the group-16 sp3 lone-pair
  crowding), exactly as `gen-fold-tables` found. The median is modelable **only with an `(S, sp3)` override**.

### FLOOR — the gate — NOT smooth at any richness

| floor model     | in-sample RMSE | in-sample max\|res\| | LOO RMSE | LOO max\|res\| |
|-----------------|---------------:|---------------------:|---------:|---------------:|
| naive           | 6.95 | 14.4 | 11.6 | 22.6 |
| naive, ridge 1  | 7.01 | 13.9 |  9.2 | 17.4 |
| rich            | 6.57 | 15.9 | 11.7 | 25.6 |
| rich, ridge 1   | 6.83 | 14.0 |  9.1 | 17.6 |
| lone-pair, ridge 1 | 6.98 | 13.9 | 9.4 | 17.5 |

Richness does **not** lower the max residual — it stays **14–16°** in-sample and **17–26°** under LOO across
every parameterisation. Two residuals of opposite sign persist in every fit and are the killers:

- **`(C, sp2)`: model over-predicts by ~+14°** (measured floor 85°, model ~99°). `(C, sp2)`'s scatter
  (median 124.4 − floor 85 = **39.4°**, the widest in the table) is invisible to a smooth function that sees
  its sp2 neighbours N/O floor at 90–96°; the model averages `(C, sp2)` toward them and lands 14° too high.
- **`(C, sp3)`: model under-predicts by ~+13°** (measured 104°, model ~91°). The opposite error, same fit.

These two coexist in one model, so **no single global margin can fix both** — the reason the tail is irreducible.

---

## 3. GATE VALIDATION (the make-or-break) — false flags, model vs table

The table rule is `floor = min(p0.5, corpus_min) − 5°`, and `corpus_min ≤ p0.5` always, so the lowest real
donation each class was calibrated **not** to flag is **`orig_min = table_floor + 5`**. A model false-flags
class *c* iff `model_floor(c) > orig_min(c)`, i.e. iff its floor residual exceeds the built-in **5° margin**.

| reference set for "must pass"            | TABLE floor false-flags | MODEL floor false-flags |
|------------------------------------------|:-----------------------:|:-----------------------:|
| rebuilt census (607 angles)              | **0** | **0** (too lenient to discriminate — Section 1) |
| calibration tail encoded by the table    | **0** (by construction) | **`(C, sp2)`** — every model, every ridge |

Every model (naive/rich × ridge 0/1) puts the `(C, sp2)` floor at **99–101°** while the original census holds a
real `(C, sp2)` donation at **90°** (table floor 85 + 5). That crystal would be wrongly flagged "folded" — the
domain-fatal direction, and **the same failure `gen-fold-tables` found on `(N, sp)`, merely relocated to
`(C, sp2)` by the richer fit**. Max floor residual 14–16° > the 5° margin ⇒ a false flag is unavoidable.

### Model + a conservative global safety margin?

To force 0 false flags a global margin `M ≈ 9–11°` must be subtracted from every model floor. It works (0
flags) but **guts the gate** (numbers for rich, ridge 1, `M = 9.0°`):

| class     | table floor | model | model−M | median | catches 90° fold? (table → model−M) |
|-----------|------------:|------:|--------:|-------:|:-----------------------------------:|
| (N, sp2)  | 96.0  | 94.7 | 85.7 | 120.7 | YES → **no** |
| (S, sp3)  | 91.0  | 88.9 | 79.9 | 103.4 | YES → **no** |
| (C, sp3)  | 104.0 | 90.3 | 81.3 | 113.3 | YES → **no** |
| (As, sp3) | 95.0  | 94.8 | 85.7 | 119.4 | YES → **no** |
| (O, sp2)  | 90.0  | 90.1 | 81.0 | 125.7 | YES → **no** |
| (C, sp2)  | 85.0  | 99.0 | 90.0 | 124.4 | no → no |
| (P, sp3)  | 89.0  | 90.4 | 81.4 | 115.8 | no → no |
| (N, sp3)  | 82.0  | 86.0 | 77.0 | 110.3 | no → no |
| (C, sp)   | 155.0 | 148.6 | 139.6 | 176.2 | YES → YES |
| (N, sp)   | 140.0 | 144.3 | 135.3 | 179.3 | YES → YES |

After the margin the gate's reach (`median − floor`) balloons from the table's tight, per-class 9–39° to a
uniform **23–48°**: a donor could fold **23–48° off its axis before being caught**. **6 of 10 classes stop
catching a right-angle (90°) fold** the table catches. The whole value of the table is a *tight per-class*
floor; the model + margin trades the false flag for a blind gate. This is the precise "global-fit win hides a
domain-fatal per-row regression" pattern of `metal-ff-cancellation` / `covalent-radius-tmqm-grounding`.

---

## 4. Period/group reach — is there a generalisation payoff?

A model's promise is covering donor classes the 10-row table lacks. **The corpus does not contain any exotic
donors** to cover: `{Se, Sb, Te, Br, I}` donor set is **empty**. The only non-table classes present are all
sub-threshold or borderline, and extending the gate to them **actively introduces a false flag**:

| class     | n | rebuilt p50 | model floor (naive, ridge 1) | outcome |
|-----------|--:|------------:|-----------------------------:|---------|
| (S, sp2)  | 8 | 101.0 | 95.4 | **would FALSE-FLAG** (rebuilt min 89.1) |
| (Si, sp3) | 4 | 108.7 | 94.2 | ok on rebuilt min 105.6 — but n<4, noise |
| (P, sp2)  | 2 | 118.1 | 99.1 | ok on rebuilt min — but n=2, noise |
| (O, sp3)  | 1 | 105.1 | 82.8 | n=1 |

So the reach benefit is **negative** on this corpus: no exotic donor to generalise to, and gating `(S, sp2)`
from the model would flag a real 89° donation. Abstention on `n < 6` (the table's policy) is a deliberate safety
choice the model would override for no gain.

---

## Verdict

**NO — the floor (the gate) is an irreducible measured scatter tail.** A period/group/hybridisation model,
however rich or regularised, has a **14–16° max floor residual** that exceeds the table's **5° calibration
margin**, so it **false-flags the `(C, sp2)` crystal tail** (model floor 99° vs a real 90° donation) — the twice
-reverted failure mode, relocated not removed. The `(C, sp2)` +14° over-prediction and `(C, sp3)` −13° under-
prediction are opposite-signed and simultaneous, so no global margin repairs the gate; the ~9° margin that
suppresses the false flag makes **6 of 10 classes miss a 90° fold**. There are no exotic donors in the corpus,
so there is no generalisation payoff, and gating the one extra populated class `(S, sp2)` from the model would
itself false-flag.

**PARTIAL — the median is modelable** (VSEPR-smooth: naive `12.07 + 0.928·ideal(hyb) + 2.13·(p−2) −
1.94·(g−14)`, RMSE 3.8°) **with an explicit `(S, sp3)` override** — but the median is the *report-only* half (`fold =
|angle − median|`), not the safety-critical half, and its LOO RMSE (7.4°) already exceeds its own use tolerance
on 10 points, so the row-count saving is small and buys a second code path. Exactly the `gen-fold-tables`
recommendation (c), no stronger.

**Recommendation: do not model the floor. Keep `_FOLD_WINDOW` measured** (it is *data* per `AGENTS.md`); the
faithful improvement is to record its provenance in the docstring, not to derive it. Optionally derive
`_FOLD_MEDIAN` from `ideal(hyb)` with an `(S, sp3)` override if row-count reduction is itself the goal — modest,
and it leaves the gate untouched.

### False-flag count, model vs table (for the orchestrator)
- **Table floor:** 0 false flags — on the 607 rebuilt donations *and* on the calibration tail (by construction).
- **Model floor:** 0 on the 607 rebuilt donations (that census is too clean to discriminate), but **re-introduces
  a false flag on the `(C, sp2)` calibration tail** (floor 99° vs the real 90° the table admits) under **every**
  parameterisation; the ~9° global margin needed to remove it blinds 6 of 10 classes to a 90° fold.
- **Implement?** No — keep the measured table; the floor is an irreducible per-class tail. The median-only
  derivation is optional and low-value.

*(Reproduce: `scratchpad/rebuild_census.py` → `census.pkl`; `scratchpad/fit_and_validate.py`;
`scratchpad/validate_honest.py`.)*
