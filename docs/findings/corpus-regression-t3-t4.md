# Corpus regression: the embed-relax seam (T3) and the phantom-floor fix (T4)

**Task.** Measurement only — no `src/` was changed by this study. Date 2026-07-20, branch
`rdkit-embed-kernel`. Baseline tree is `HEAD` at `acd359c`; the two changes under test are
uncommitted in the working tree. The whole question: **do T3 and T4 hold at 144-structure scale, or
should either be reverted?**

**Verdict in one line.**
- **T3 (embed relax seam): SHIP.** A uniform, large embed-stage win (window satisfaction, M-donor
  MAE, folded seeds) that converges to HEAD's answer after `minimize()`, plus four rescued empty
  ensembles and one rescued silent zero-conformer failure (TUXRUZ), with **zero new failures** and no
  masked bond tear anywhere in the corpus. Named post-`minimize` regressions are small and few
  (WUVJAB, SOHMEJ, and a handful of fold moves).
- **T4 (phantom-floor fix): SHIP-AS-INERT / do not credit at corpus scale.** T4 is **provably inert
  on all 142 in-scope structures at 3 seeds** — the tree-2 and tree-3 result JSONs are **bit-for-bit
  identical**. Its relief path (`metal.coordinate()`) is never entered by a plain `rx.embed(xyz)`, and
  its `min`-merge is unobservable because `dg_floors` has a single producer on that path (no collision
  → `min == max`). It is **not a regression** (it cannot make anything worse here), but its one
  documented win — relieving a bound crossover — **fires on zero corpus structures**. The single
  structure that hits a crossover (TiCat2, 7.28%) is **unchanged** by T4. Keep it for the
  `coordinate=`-seating path it was written for; do not claim a corpus-scale benefit.

---

## 1. Method — three trees, isolated by worktree

The two edits split cleanly by file, so they were separated and each measured on its own tree:

| Tree | Contents | `src/` files patched |
|---|---|---|
| **1 `HEAD`** | before both changes | none |
| **2 `+T3`** | HEAD + embed relax seam only | `pipeline.py`, `metrics.py` |
| **3 `+T3+T4`** | working tree (both) | + `constraints/base.py`, `constraints/metal.py`, `embed/dispatch.py` |

A fourth tree (`HEAD + T4 only`) was built for the T4 attribution probe in §4.

```bash
# working tree preserved first (AGENTS.md: do not lose it)
git diff > $S/t34_both_FULL.patch            # 8.5 MB, md5 5fc5423…
git diff -- src/rxembed/pipeline.py src/rxembed/metrics.py > $S/cr_T3.patch
git diff -- src/rxembed/constraints/base.py src/rxembed/constraints/metal.py \
            src/rxembed/embed/dispatch.py > $S/cr_T4.patch
git worktree add $W/tree1_head  HEAD --detach
git worktree add $W/tree2_t3    HEAD --detach && (cd $W/tree2_t3  && git apply $S/cr_T3.patch)
git worktree add $W/tree3_both  HEAD --detach && (cd $W/tree3_both && git apply $S/cr_T3.patch && git apply $S/cr_T4.patch)
```

Tree-3 `src/` was verified **bit-identical** to the working tree, and the working-tree `src/` diff was
re-verified intact at the end of the study (`md5 2dd4f0e5…`, matches the saved patch). Each tree is
imported via `PYTHONPATH=<tree>/src`; import isolation asserted (`_relax_embedded` present only in
2/3, `_merge_relief` only in 3).

**Corpus (144, both halves, no sampling):** 103 tmQM + 41 fixtures. **Two structures excluded**
structurally (≥5 boron, UFF has no `B_5`/`B_6`): **WIMCAA** (the named borane) **and XAQDUS** — XAQDUS
is *also* the named zero-conformer trap, so its exclusion removes a false-perfect at the same time.
In-scope corpus = **142** (101 tmQM + 41 fixtures).

**Parameters (matched to prior art `docs/findings/seed-vs-relax.md`):** `n=8`, `charge` from the file
header, `minimize(_retry=False)` to keep the seed↔relax pairing. **Three seeds** `0xF00D 0xBEEF 0x1234`
— 426 runs per tree. Every geometry is scored twice: at the **embed** return (`rx.embed(...)`) and at
**post** (`.minimize(_retry=False)`).

**Null-measurement guards (each asserted, per AGENTS.md):**
- `keep_input` prepends the crystal → `ids[0]` bit-identical to input → **excluded**; only ETKDG seeds
  scored (the `keep_input` trap).
- A run whose only conformer is that input geometry is a **zero-conformer silent failure**, recorded
  as a failure, never scored (TUXRUZ, ticat3).
- Metal restored to the **real element and oxidation state** post-`restore` — verified every run
  (`charge_bad = 0` on all three trees).
- `GeometryReport.ok` is a **method**; an early harness wrote `bool(rep.ok)` (always truthy) and
  reported a 100% gate pass beside 9,203 recorded violations. Fixed to `rep.ok()`; the pass rate is
  derived from the per-kind census, which is unaffected.
- The T4 relief probe's `compose` spy is **asserted installed in `embed.dispatch`** — an earlier
  version swallowed an `ImportError` and reported 0 for the whole corpus (the exact null this repo
  warns about). See §4.

```bash
S=playground/corpus_regression
PYTHONPATH=<tree>/src uv run python $S/cr_measure.py <out.json> 0xF00D 0xBEEF 0x1234
uv run python $S/cr_compare.py cr_out_HEAD.json cr_out_T3.json cr_out_T3T4.json
uv run python $S/cr_detail.py  cr_out_HEAD.json cr_out_T3.json cr_out_T3T4.json
```

Result JSONs and the two console captures are checked in under `playground/corpus_regression/`.

---

## 2. Pooled and per-half tables

Matched keys = 426/426 across all three trees (identical inputs; nothing pooled across different run
sets). `gate_pass` = fraction of scored conformers with **no** `geometry.check` violation of any kind.
Lower is better on every quantitative axis.

### Embed-stage (`rx.embed(...)` return — what a caller sees before any `minimize`)

| metric | | tmQM (101) | | | fixtures (41) | | | **POOLED (142)** | |
|---|---|---|---|---|---|---|---|---|---|
| | HEAD | +T3 | +T4 | HEAD | +T3 | +T4 | **HEAD** | **+T3** | **+T4** |
| zero-conf runs | 3 | **0** | 0 | 3 | 3 | 3 | **6** | **3** | 3 |
| conformers kept | 1731 | 2138 | 2138 | 437 | 529 | 529 | **2168** | **2667** | 2667 |
| runs w/ crossover | 0 | 0 | 0 | 3 | 3 | 3 | **3** | **3** | 3 |
| charge wrong | 0 | 0 | 0 | 0 | 0 | 0 | **0** | **0** | 0 |
| gate_pass | .251 | .640 | .640 | .338 | .792 | .792 | **.276** | **.683** | .683 |
| ml_mae (Å) | .068 | .013 | .013 | .055 | .013 | .013 | **.064** | **.013** | .013 |
| fold (°) | 26.6 | 23.5 | 23.5 | 15.3 | 11.8 | 11.8 | **23.4** | **20.2** | 20.2 |
| bond_mae (Å) | .024 | .021 | .021 | .035 | .034 | .034 | **.027** | **.025** | .025 |
| rmsd (Å) | 2.46 | 2.22 | 2.22 | 1.68 | 1.42 | 1.42 | **2.23** | **1.99** | 1.99 |
| dwin_max (Å) | .057 | .009 | .009 | .039 | .013 | .013 | **.051** | **.010** | .010 |
| awin_max (°) | 17.6 | 3.60 | 3.60 | 13.8 | 3.95 | 3.95 | **16.5** | **3.70** | 3.70 |

### Post-`minimize` (the delivered geometry of the standard `embed().minimize()` path)

| metric | | tmQM | | | fixtures | | | **POOLED** | |
|---|---|---|---|---|---|---|---|---|---|
| | HEAD | +T3 | +T4 | HEAD | +T3 | +T4 | **HEAD** | **+T3** | **+T4** |
| gate_pass | .677 | .689 | .689 | .879 | .889 | .889 | **.730** | **.742** | .742 |
| ml_mae (Å) | .0056 | .0058 | .0058 | .0054 | .0052 | .0052 | **.0055** | **.0056** | .0056 |
| fold (°) | 22.6 | 22.5 | 22.5 | 13.5 | 12.9 | 12.9 | **20.2** | **19.9** | 19.9 |
| bond_mae (Å) | .019 | .019 | .019 | .029 | .029 | .029 | **.022** | **.022** | .022 |
| rmsd (Å) | 2.40 | 2.14 | 2.14 | 1.59 | 1.34 | 1.34 | **2.19** | **1.93** | 1.93 |
| dwin_max (Å) | .0001 | .0001 | .0001 | 0 | 0 | 0 | **.0001** | **.0001** | .0001 |
| awin_max (°) | .0002 | .0006 | .0006 | .0001 | .0001 | .0001 | **.0002** | **.0005** | .0005 |

**+T4 is identical to +T3 in every cell, both halves, both stages.** This is not a rounding artifact:
a full-precision field-by-field diff of the two JSONs (`cr_t4_identity.py`) finds **0 differing leaves
across 0 structures** — exact bit identity.

The **fixtures half has never been scored for embed quality before** (per the task). Its numbers above
are therefore **new baseline**, not a comparison against a prior study.

### The two facts that drive T3's shape

1. **Embed-stage wins are large and near-uniform; post-`minimize` they mostly wash out.** ml_mae
   .064→.013 and awin_max 16.5°→3.7° at embed collapse to ml_mae .0055 vs .0056 and awin ≈0 at post.
   This is expected and *correct*: HEAD's `minimize()` already relaxed seeds into their windows; T3
   moves that relax **earlier**, into `embed()`'s return. The value of that is (a) `rx.embed(...)`
   without a follow-up `minimize` now returns window-satisfying geometry (matters for `mc()`, which
   searches *around* the seeds), and (b) the endpoints below.

2. **The durable post wins are yield and rescues, not local-axis accuracy.** Conformer survival
   72.6%→78.3%; **four ensembles rescued from empty** (`HgI3, RERHEB, WELROW, ZAHZUD`) and **one
   silent zero-conformer failure rescued** (`TUXRUZ`: 0→1 conf at ml_mae 0.003 Å, rmsd 0.90 Å). RMSD
   improves on 110 structures post-`minimize` largely because more conformers survive to be scored.
   **No structure was newly emptied and none newly went zero-conf** — T3's failure set is a strict
   subset of HEAD's.

---

## 3. Improved / regressed distribution, by name

Per-structure verdict = same direction on a **majority of the 3 seeds** (guards against one lucky
seed). Deltas are +T3 − HEAD; negative = better.

### Post-`minimize` (the deliverable)

| axis | improved | regressed | named regressions (Δ) |
|---|---|---|---|
| ml_mae | 0 | **1** | WUVJAB +0.025 |
| rmsd | **110** | 3 | SOHMEJ +0.388, YIXSIJ +0.062, PdCl2-RR-BDNN +0.047 |
| fold | 14 | 9 | **WUVJAB +7.66, WELROW +7.06, HOXLAK +4.29**, QUZKAZ +2.80, FeH2(CO)4 +2.15, OWAHEC +2.03, PESHUT +1.51, REVWAU +1.04, IROXET +0.99 |
| bond_mae | 1 | 0 | — |
| dwin_max | 0 | 0 | (both ≈0 at post) |
| awin_max | 0 | 0 | (both ≈0 at post) |

**The one delivered-geometry regression that matters is WUVJAB** (tmQM): its M-donor MAE goes
0.001→0.026 Å and its donor fold 14.5°→22.2°. WUVJAB is also one of three structures carrying a
post-`minimize` clash (see below). It is a single named structure, not a class.

**WELROW/HOXLAK fold regressions come bundled with RMSD wins** (WELROW rmsd −1.25, rescued from empty;
HOXLAK −0.14): the relax trades donor-fold — the census-calibrated weak axis — for global position on
these. SOHMEJ is the clearest pure rmsd regression (+0.39, no offsetting gain).

### Embed-stage (for completeness — the near-uniform win)

ml_mae improved 139 / regressed 0; awin_max 132 / 0; dwin_max 93 / 2 (TiCat1, NOBKIA); rmsd 128 / 10
(worst KEGFOU +1.78 — but KEGFOU *improves* by post-`minimize`, rmsd −0.45); fold 59 / 26.

### The clash tradeoff (named)

Post-`minimize` conformers carrying a `clash` violation rise **8 → 28** (summed over 3 seeds),
concentrated in exactly three structures: **ZAHZUD 4→19, WUVJAB 3→5, WELROW 1→4**. ZAHZUD and WELROW
were **empty at HEAD** (`minimize` dropped everything) and are **rescued** by T3 — so the extra clashes
are the cost of returning *something* where HEAD returned nothing. The per-conformer gate pass **rate**
still rises (0.730→0.742), i.e. T3 is not worse per conformer; it keeps more, and a minority of the
newly kept ones clash. This is the one place to watch, and it is contained to named structures.

### Seed sensitivity

The embed-stage T3 signal dwarfs seed noise: ml_mae improvement (~0.05 Å) vs median across-seed spread
0.0004–0.008 Å; awin improvement (~13°) vs median spread 0.0002–4.8°. RMSD is the seed-sensitive axis
(pooled improvement ~0.24 Å ≈ median across-seed spread ~0.27 Å), which is exactly why the per-structure
verdicts use the majority-of-seeds rule rather than a single seed. All T4 seed-sensitivity rows are
identical between +T3 and +T4 (consistent with exact identity).

---

## 4. T4 attribution — is the phantom-floor relief reachable at all?

**T4 has three parts, and the corpus protocol reaches none of them:**

| part | what it does | reachable by `rx.embed(xyz, charge=q)`? |
|---|---|---|
| P1 `metal.coordinate()` emits `dg_floors` | new relief on a **seated** donor's neighbours | only when `coordinate=` is passed |
| P2 `dispatch` **composes** instead of cherry-picking | carries P1's relief into the bounds matrix | only when a donor is seated (`atoms is not None`) |
| P3 `base._merge_relief` (`min`) replaces `_merge_floor` (`max`) on `dg_floors` | picks the fuller relief on a key collision | only when **two** parts both write the same `dg_floors` key |

Instrumenting `compose()` and `metal.coordinate()` across all 142 in-scope structures × the embed path
(`cr_t4_fires.py`, spy **asserted** installed in `dispatch`):

```
compose_calls = 0    coordinate_calls = 0    dgf_key_collisions = 0    min≠max collisions = 0
```

`coordinate=` is never passed by the corpus protocol, so `metal.coordinate()` is **never called** and
P1/P2 never run. `dg_floors` *is* populated on 136/142 structures — but entirely by the **single**
producer `nondonor_floors`, so on the default path there is never a second source to collide with, and
`min(x) == max(x) == x`: **P3 is unobservable**. The bounds writer reads exactly the same `dg_floors`
under T4 as under HEAD. Hence the exact JSON identity in §2.

**Where T4 *does* fire** (`cr_t4_domain.py`, the maintainer's own case, `OCCCN->[Pd](Cl)Cl` with the
alkoxide O seated via `metal="square_planar", coordinate=0`): `coordinate()` runs (22 `dg_floors`
emitted, 18 key collisions, **2 with min≠max**). Attributed across the four trees:

| tree | seed-level Pt···C mean / min (Å) | conformers |
|---|---|---|
| HEAD + T4 only (raw seeds) | 3.147 / **2.880** | 5 |
| HEAD (raw seeds) | 3.247 / 3.100 | 6 |
| +T3 (relaxed) | 3.617 / 3.374 | 6 |
| +T3+T4 (relaxed) | 3.660 / 3.374 | 5 |

The maintainer's **seed-level** claim reproduces directionally (T4 pulls the phantom-floored Pt···C
in: mean 3.247→3.147, min 3.100→2.880). But it is a **seeding** effect that (a) requires the
`coordinate=` path, and (b) costs a conformer here (6→5) and, once T3's relax runs on top, nudges
Pt···C slightly *longer* (3.617→3.660). None of this reaches the corpus.

**T4's stated target — bound-crossover relief — fires on zero corpus structures.** Exactly one
structure hits a non-zero smoothing tolerance at all: **TiCat2** (fixtures, 7.28%), and its crossover
is **identical on all three trees** — T4 does not reduce it (T4's relief is unreachable for it,
because it comes through `nondonor_floors`, not the seating path).

---

## 5. T3 safety — does the quieter third change weaken the tear gate?

T3 is described as two pieces (the `_relax_into_windows` seam + `_rescue_torn`), but it carries a
third: `metrics.bonding_ok` now skips any pair listed in `Constraints.distances` (the `constrained=`
argument). Sound for a user-requested dissociation; the risk is a pair that is *also a real bond*
becoming invisible to the tear check. Measured on all 142 (`cr_t3_exemption.py`):

```
exempt pairs (non-metal, corpus-wide) = 834
of which are REAL BONDS in the graph  = 0
conformers where lax gate passes but strict gate fails (tear-masked) = 0
```

**The exemption cannot hide a broken bond anywhere in the corpus.** Every exempt non-metal pair is an
angle-/inter-donor separation the constraint system states, not a bond a radius rule would judge. Zero
tears are masked. This closes the one plausible way T3 could silently degrade a result.

---

## 6. Verdict

**T3 — SHIP.** At 144-structure scale (142 in-scope, 3 seeds, 426 runs/tree) the embed relax seam is a
uniform embed-stage win that converges to HEAD's answer after `minimize()`, so it never degrades the
delivered local geometry except on **WUVJAB** (ml_mae +0.025 Å, fold +7.7°). It rescues one silent
zero-conformer failure (**TUXRUZ**) and four empty ensembles (**HgI3, RERHEB, WELROW, ZAHZUD**),
introduces **no new failures**, keeps 5.7 pts more conformers, and **masks no bond tear** (0/142). The
costs are contained and named: a clash tradeoff on **ZAHZUD/WUVJAB/WELROW** (the rescued-from-empty
ones carry some clashing/folded poses), SOHMEJ rmsd +0.39, and a short list of small fold regressions.
Ship; watch WUVJAB and the ZAHZUD/WELROW rescue quality. Add a regression test asserting `rx.embed` of
a constrained metal returns window-satisfying seeds (the gap the green suite missed — HEAD returned
seeds violating angle windows by up to 42.8°, and 284 tests stayed green).

**T4 — SHIP but do not credit at corpus scale (candidate for deferral, not revert).** T4 is **exactly
inert on all 142 in-scope structures at 3 seeds** — bit-identical result JSONs — so it **cannot be a
regression** on this corpus. But its relief path is entered only via `coordinate=`-seating, which the
corpus protocol never uses, and its `min`-merge is unobservable on the default path (single-source
`dg_floors` ⇒ no collision). Its one measurable benefit, crossover relief, **fires on zero corpus
structures**; the only crossover present (TiCat2) is untouched. Its seed-level claim reproduces **only**
in the seating case it was written for, where it also costs a conformer. **Recommendation:** keep T4
for the `coordinate=` path it targets, but do not attribute any corpus-scale improvement to it, and
gate it behind a test that exercises the *seating* path (e.g. the `OCCCN->[Pd](Cl)Cl` case), since no
corpus fixture will ever catch a regression in it.

### Refuted / could not confirm

- **REFUTED at corpus scale: "T4 relieves the phantom bond-less-carbon floor."** True only through the
  `coordinate=` seating path; on a plain `rx.embed(xyz)` of any of the 142 structures the path is never
  entered (0 `coordinate()` calls) and the change is bit-for-bit inert.
- **REFUTED: T4 reduces bound-crossover incidence.** One structure crosses over (TiCat2, 7.28%);
  identical on all three trees.
- **Confirmed at scale (maintainer's T3 evidence): raw seeds miss their windows** — pooled embed-stage
  awin_max 16.5° and dwin_max 0.051 Å at HEAD, and the relax enforces them (3.7° / 0.010 Å). The 6/60→
  56/60 window-satisfaction claim generalises.
- **Not established: any T3 net regression.** Post-`minimize` T3 improves rmsd on 110 structures and
  regresses M-donor accuracy on exactly one (WUVJAB); it is a net win, not a net regression.
```
