# Seed quality vs cleanup

**Task T2.** Measurement only — no code was changed by this study. Date 2026-07-20, branch
`rdkit-embed-kernel` at `4dee01b`.

> **Tree provenance.** A concurrent agent modified `src/rxembed/pipeline.py` during this session
> (17:06), adding `discarded` bookkeeping to `minimize()`. That edit landed **before** the main run
> (17:16), so the whole corpus was measured against one consistent tree, and the change is inert for
> every metric here (it appends to `discarded`; it does not alter `self.ids` or any coordinate).
> Verified by re-running six structures — including both zero/near-zero-survivor cases — against the
> current tree and diffing every metric: **bit-for-bit identical**
> (`t2_seed_vs_relax.py $S/t2_recheck.json`). Recorded because AGENTS.md warns that a concurrent
> repo write has produced a spurious result here before.

**The concern being tested.** That `rxembed` embeds noise and lets UFF force it into shape, so UFF's
biases — folds, preferred torsions — decide the output rather than the constraints.

**Verdict in one line.** The *premise* is confirmed and the *feared consequence* is refuted: the raw
DG seed genuinely does not satisfy its own constraints (angle windows missed by 9.5° on average, up
to 42.8°), and the relax is what enforces them — but the relax moves geometry **toward** the crystal
on every local axis measured and is neutral on the global one. There is no measured axis on which
the relax degrades the answer. **The answer is neither "tighter bounds" nor "a lighter relax"** —
see [Verdict](#verdict).

---

## 1. Scripts and commands

All scripts are preserved in `playground/seed_vs_relax/` (they were written to the session
scratchpad and copied into the repo so the numbers stay reproducible).

| Script | Produces |
|---|---|
| `t2_survey.py` | `t2_survey.json` — metal, CN (`MND`), charge (`q`), atom count, donor elements for all 103 tmQM structures |
| `t2_select.py` | `t2_selection.json` — the 30 structures (selection rule in §2) |
| `t2_seed_vs_relax.py` | `t2_main_out.json` — the paired per-conformer seed/relax/crystal scores |
| `t2_aggregate.py` | the per-axis table and outlier lists (§4, §5) |
| `t2_why_seed_misses.py` | the bounds-matrix diagnostic and the `knowledge=` comparison (§6) |
| `t2_bm_convention_check.py` | verifies the bounds-matrix index convention §6 depends on |
| `t2_claims.py` | the named-claim reproductions (§7) |

```bash
S=playground/seed_vs_relax
uv run python $S/t2_survey.py
uv run python $S/t2_select.py
uv run python $S/t2_seed_vs_relax.py $S/t2_selection.json $S/t2_main_out.json
uv run python $S/t2_aggregate.py
uv run python $S/t2_why_seed_misses.py
uv run python $S/t2_bm_convention_check.py
uv run python $S/t2_claims.py
```

Fixed parameters everywhere: `seed=0xF00D` (`bounds.DEFAULT_SEED`), `n=8` conformers per structure,
`charge=` taken from the tmQM header's `q =` field. `minimize(_retry=False)` — see §3.

---

## 2. The 30 structures and how they were chosen

Corpus: `/home/ali/Documents/Codes/OIN-SMILES/tests/integration/tmQM/*.xyz` (103 structures, all
single-metal; CN 2–14; charges −1/0/+1; 14–203 atoms).

**Selection rule** (`t2_select.py`, deterministic, no RNG anywhere):

> Bucket the 103 structures by coordination number (the header's `MND`). Round-robin over the CN
> buckets in ascending CN; within a bucket, round-robin over metal elements in alphabetical order,
> each turn taking the alphabetically-first not-yet-taken refcode of that metal. Emit until 30.

This spans coordination number first and metal element second, and is a pure function of the corpus
listing — re-running it reproduces the same 30.

| Refcode | M | CN | q | atoms | donors | Refcode | M | CN | q | atoms | donors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ABEZAJ | Ti | 5 | 0 | 74 | C,Cl,N | LOSCEE | V | 5 | 0 | 42 | C,N,O |
| ADUDAG | Cr | 9 | 0 | 70 | C | NUDXOC | Zn | 5 | 0 | 69 | C,N |
| AKILAJ | Ni | 4 | 0 | 61 | P,S | NUKHEG | Fe | 7 | 0 | 28 | C,O |
| ASIPAW | W | 9 | 0 | 49 | C,P | QISROZ | Ir | 8 | 0 | 45 | C,Cl,P |
| CBZYTA | Ta | 12 | 0 | 48 | C | QIXJOW | Cd | 2 | 0 | 165 | Si |
| COJKAO | Pd | 4 | 0 | 64 | C,O,P,S | RERHEB | Hf | 14 | 0 | 67 | C |
| CSBRHB | Rh | 7 | 0 | 74 | C,Cl,P | SIFJUO | Fe | 3 | 0 | 92 | C |
| DASSUL | Cd | 6 | −1 | 55 | S | SOHMEJ | Hg | 4 | 0 | 93 | Br,N |
| DEYMIE | V | 7 | 0 | 47 | N,O | TARJON | Fe | 6 | 0 | 57 | C,Cl,N,O |
| DUNSEN | Au | 2 | 0 | 65 | Cl,P | UTANUA | Ru | 9 | 0 | 67 | C,Cl,N |
| ENCYSM | Co | 6 | +1 | 37 | C,N,S | WIMCAA | Cd | 8 | 0 | 103 | B,N |
| ESOVIU | Ag | 2 | 0 | 41 | C,Cl | ZOPNOH | Mn | 5 | 0 | 36 | C,N,O |
| FIHTIZ | Fe | 8 | 0 | 71 | C,O,P | HIBPAK | Ag | 2 | 0 | 33 | C,Cl |
| FIJTOH | Ag | 6 | +1 | 43 | S | HURVOI | Au | 4 | +1 | 54 | Br,S |
| ILONON | Zr | 12 | 0 | 92 | C,Cl | KASFIU | W | 7 | 0 | 61 | C,N,P |

**Span achieved.** CN 2 (4), 3 (1), 4 (4), 5 (4), 6 (4), 7 (4), 8 (3), 9 (3), 12 (2), 14 (1);
20 distinct metals (Ag, Au, Cd, Co, Cr, Fe, Hf, Hg, Ir, Mn, Ni, Pd, Rh, Ru, Ta, Ti, V, W, Zn, Zr);
donor elements B, Br, C, Cl, N, O, P, S, Si.

---

## 3. Methodology and the null-measurement guards

Each structure is embedded once and the *same conformer* is scored before and after the relax, so
nothing is compared across different geometries:

```python
ens = rx.embed(path, charge=q, n=8, seed=0xF00D)   # metal xyz, no metal= -> retain-input path
# snapshot every seed's coordinates HERE, before minimize() is called at all
ens.minimize(_retry=False)
# pair on conformer id
```

`_retry=False` disables `Ensemble._reembed_until_clean`, which otherwise injects **fresh seeds with
a new randomSeed** part-way through `minimize()`. Left on, it would both destroy the pairing and
confound "the relax improved this seed" with "a different seed was substituted". This is the one
non-default argument in the harness.

Both sides are scored on **one** molecule — the post-`minimize` `Ensemble.mol`, whose metal has been
restored — with its conformer coordinates driven to each coordinate set in turn. So the graph,
perception, hybridisation and donor list are byte-identical across the comparison; only the
coordinates differ.

### Guards (each asserted in code, not assumed)

| # | Guard | Result over the 30 |
|---|---|---|
| G1 | The retain-input path prepends the crystal conformer, so `ids[0]` is bit-identical to the input **by construction**. It is excluded; the scored seeds are `ids[1:]`. | `input_is_first_conf = True` for 30/30 |
| G2 | A structure whose only conformer is that input (no ETKDG seed at all) is a **failed embed** and is excluded entirely. Any seed bit-identical to the crystal is likewise dropped. | 0 of the 30 failed this way; **TUXRUZ and XAQDUS do** — see §7 |
| G3 | A second, independent `rx.embed()` with the same seed must reproduce the snapshot bit-for-bit. | `True` for 30/30 |
| G4 | Seed and relaxed coordinates must actually differ. | `True` for 186/186 pairs; min drift **0.290 Å**, median **1.62 Å** |
| G5 | Metal must be the carbon surrogate pre-`minimize` and the real element *with its oxidation state* post-restore. | `surrogate_z == 6` and `restored_z == real_z` and `restored_q == real_q` for 30/30 |
| G6 | Atom count and element ordering (metal aside) identical pre/post, so one scoring graph is valid for both sides. | `True` for 30/30 |

G1 deserves emphasis: on this code path the "embed" output *always* contains a bit-identical copy of
the input as conformer 0, deliberately (`_embed_isomer(..., keep_input=True)`). Any harness that
scores `ens.ids` naively will score the crystal against itself and report a perfect result. This is
the exact null-measurement the task warned about, and it is live on the default path.

### Sample size

236 seeds embedded across the 30 structures; **186** survived `minimize()` and form the paired set.
**50 seeds (21%) were dropped by the relax** — see §5, this is the relax's main measured cost.

---

## 4. Per-axis results

186 paired conformers, 30 structures. "crystal" is the crystal geometry scored on the same metrics
through the same code — the floor. "help/hurt/tie" counts individual conformers.

| Axis | seed | relax | crystal | Δ (relax−seed) | help / hurt / tie | Reading |
|---|---|---|---|---|---|---|
| M–donor distance MAE vs crystal (Å) | 0.063 | **0.006** | 0.000 | **−0.057** | 185 / 1 / 0 | relax helps, near-universally |
| M–donor distance max vs crystal (Å) | 0.096 | **0.013** | 0.000 | −0.083 | 182 / 4 / 0 | relax helps |
| bond-length MAE vs crystal (Å) | 0.025 | **0.021** | 0.000 | −0.005 | 163 / 23 / 0 | relax helps, modestly |
| bond-length max vs crystal (Å) | 0.142 | **0.124** | 0.000 | −0.019 | 119 / 67 / 0 | relax helps, weakly |
| donor fold (° off class median) | 20.20 | **17.04** | 9.87 | −3.16 | 95 / 55 / 36 | relax helps **on net, but mixed** |
| `donor_orientation` gate violations | 0.387 | **0.253** | 0.000 | −0.134 | 29 / 16 / 141 | relax helps |
| distance-window violation, max (Å) | 0.047 | **0.000** | 0.000 | −0.047 | 131 / 3 / 52 | relax enforces |
| angle-window violation, max (°) | 9.54 | **0.00** | 0.00 | −9.54 | 171 / 6 / 9 | relax enforces |
| heavy-atom RMSD to crystal (Å) | 2.365 | 2.377 | 0.000 | **+0.012** | 96 / 90 / 0 | **neutral** |

### What each axis says

**Window satisfaction — the seed does not satisfy its own constraints.** Angle windows are missed by
9.54° on average (per-conformer max), with per-structure medians up to 42.8° (DEYMIE) and 28.6°
(TARJON). Distance windows are missed far less (0.047 Å mean max). After the relax both are 0.00
essentially everywhere. **This is the premise of the concern, and it is confirmed.**

**The relax is not merely satisfying the constraint at the expense of the structure.** M–donor
distances go from 0.063 Å to 0.006 Å against the crystal (185 of 186 conformers improve), and general
bond lengths improve too. If UFF were "forcing noise into shape", the bond lengths and the M–L
distances would be where that showed, and they move the right way.

**Donor fold is the weakest axis and the one genuine caveat.** Net improvement (20.2° → 17.0°), but
55 of 186 conformers get *worse*, and both sides sit well above the crystal's own 9.9°. So neither
the seed nor the relax reproduces crystal donor orientation; the relax is merely the better of two
mediocre numbers.

**Global conformation is untouched.** Heavy-atom RMSD to the crystal is statistically flat
(+0.012 Å; 96 improve, 90 worsen). This is expected — a conformer search is not trying to reproduce
the crystal's packing-determined conformer — and it is the axis on which "UFF's preferred torsions
decide the output" would show up as a systematic degradation. **It does not.**

### Crystal baseline caveat (important)

The crystal satisfies **every** applied window exactly (angle violation 0.00°, distance 0.000 Å, over
all 30). That is **not** independent evidence that the windows are right: on the retain-input path
the windows are *derived from the input geometry* (`metal.hold_shape` / `from_geometry`), so the
crystal satisfies them by construction. What it does establish is that "relax reaches 0.00 window
violation" means the relax moves toward the region the crystal occupies, not away from it — the
windows and the crystal are not in conflict.

---

## 5. Outliers by name

**Donor fold most degraded by the relax** (the UFF-born folds — the one axis where the concern has
real instances):

| Structure | seed → relax | Δ |
|---|---|---|
| COJKAO | 20.0° → 28.1° | +8.1 |
| AKILAJ | 28.7° → 35.7° | +7.0 |
| CSBRHB | 17.6° → 23.8° | +6.2 |
| NUDXOC | 40.2° → 45.1° | +4.8 |
| SOHMEJ | 23.7° → 26.1° | +2.4 |
| SIFJUO | 6.6° → 8.2° | +1.6 |

**Heavy-atom RMSD to crystal most degraded:** DASSUL 1.81 → 2.26 Å (+0.44), FIHTIZ 2.89 → 3.20
(+0.31), DEYMIE 2.58 → 2.77 (+0.18), ESOVIU 1.65 → 1.83 (+0.18). These are the largest single
regressions anywhere in the study and they are small.

**Bond-length MAE most degraded:** AKILAJ +0.001 Å. Nothing else exceeds +0.0005 Å. **No structure's
bond lengths are meaningfully worsened by the relax.**

**M–donor distance most improved:** QIXJOW 0.196 → 0.002 Å, CSBRHB 0.131 → 0.005, KASFIU 0.093 →
0.006, ILONON 0.087 → 0.001, ABEZAJ 0.089 → 0.010, ASIPAW 0.078 → 0.007.

**Angle-window satisfaction most improved:** DEYMIE 42.8° → 0.00, TARJON 28.6° → 0.00, ZOPNOH 17.9°
→ 0.00, DUNSEN 14.5° → 0.00, HIBPAK 11.4° → 0.00, AKILAJ 11.1° → 0.00.

**Seed attrition to the relax** (the relax's real cost — dropped by `bonding_ok` /
`_coordination_ok` / `_shape_intact` / the relax-energy window):

| Structure | kept / seeds |
|---|---|
| **WIMCAA** | **0 / 8** — every conformer destroyed |
| AKILAJ | 1 / 8 |
| UTANUA | 2 / 8 |
| ASIPAW | 3 / 7 |
| QISROZ | 3 / 8 |
| COJKAO, RERHEB, TARJON | 4 / 8, 4 / 8, 4 / 7 |

**WIMCAA** is the single worst case in the study: a Cd/B/N complex whose borane cage UFF cannot type
(`UFFTYPER: Unrecognized atom type: B_5 / B_6` on every build), and the relax leaves **zero**
conformers. Its seeds were fine. Overall 50/236 = **21% of seeds are consumed by the relax**.

---

## 6. Why the seed misses its windows — the decisive diagnostic

If the seed misses its angle windows, there are two possible causes and they imply **opposite**
fixes:

- **(a) under-tight bounds** — the angle window was never fully written into the bounds matrix, so
  the 1–3 distance the seed realises is *inside* the matrix bound while the angle is outside its
  window. Fix: tighten the bounds.
- **(b) ETKDG leaves the matrix** — the bound is tight but the embed's coordinate refinement does not
  land on it, so the 1–3 distance is *outside* the matrix bound too. Fix: nothing in the bounds can
  help.

`t2_why_seed_misses.py` classifies every violated angle instance across all 30 structures by
rebuilding the same matrix the embed used (`bounds._feasible_bounds`, same `cons`, same
phantom-materialised mol) and asking where the realised 1–3 distance sits.

```
TOTAL violated angle instances: 1399
  (a) 1-3 distance INSIDE the bounds matrix but angle outside its window:  215  (15%)
  (b) 1-3 distance OUTSIDE the bounds matrix too (ETKDG left the matrix): 1184  (85%)
```

**Triangle-smoothing tolerance was 0.000 for all 30 structures** — the constraints are mutually
realisable everywhere in this corpus; nothing was repaired, nothing was widened. The matrix is not
over-constrained and it is not under-specified.

**So 85% of the angle miss cannot be fixed by tightening the bounds matrix — the seed is already
outside the bound that exists.** The remaining 15% is the genuine expressiveness limit: a bounds
matrix constrains a 1–3 *distance*, and if the two legs stretch, the same 1–3 distance corresponds
to a different angle. That part is not fixable by tightening either; it is inherent to encoding an
angle as a distance.

### Convention check

The result above depends on reading `bm[a][b]` as the upper and `bm[b][a]` as the lower bound for
`a < b`. `t2_bm_convention_check.py` verifies it: `lower <= upper` holds for every pair, and the
fraction of *all* atom pairs whose realised distance lies inside the matrix is high, so ETKDG is not
globally ignoring the matrix — it leaves it specifically on the constrained pairs:

| Structure | smooth tol | seed pairs inside bm | crystal pairs inside bm |
|---|---|---|---|
| ZOPNOH | 0.000 | 95.6% | 87.9% |
| NUKHEG | 0.000 | 80.7% | 86.8% |
| DEYMIE | 0.000 | 93.3% | 90.7% |
| HIBPAK | 0.000 | 97.2% | 90.9% |

### Are ETKDG's knowledge terms the culprit?

Tested directly by re-embedding all 30 with `knowledge=False` (drops the experimental-torsion and
basic-knowledge terms, leaving plain distance geometry), same seed:

```
MEAN over 30 structures: angle-window max  K=True 14.65 deg -> K=False 14.01 deg
                         dist-window  max  K=True 0.049 A   -> K=False 0.029 A
```

**The knowledge terms are not the cause of the angle miss** (14.65° → 14.01°, a 4% change). They do
account for roughly 40% of the — much smaller — distance-window miss. Turning them off is not a fix,
and would cost the chemistry-seeded periphery that `bounds.embed` deliberately keeps.

---

## 7. Named-claim reproduction

Run by `t2_claims.py` through the identical harness (`n=8`, `seed=0xF00D`).

| Claim | Claimed | Measured | Status |
|---|---|---|---|
| XAWQUH seed violates its window by 48.2° | 48.2° | **42.55°** max (mean 31.72°), → 0.00 after relax | **Substantially reproduced** — same phenomenon, exact figure not reproduced |
| NEWVOB 30.3° | 30.3° | **32.65°** max (mean 27.45°), → 0.00 | **Reproduced** |
| ZOPNOH 26.8° | 26.8° | **23.64°** max (mean 18.12°), → 0.00 | **Substantially reproduced** |
| TUXRUZ returns the input bit-identically, no warning | — | **CONFIRMED**: `FAILED EMBED: output is the input geometry only (no ETKDG seed)` | **Confirmed** |
| XAQDUS ditto | — | **CONFIRMED**, same | **Confirmed** |
| organic `constrain={(0,8):(2.5,3.0)}` violated by 0.318 Å on the raw seed | 0.318 Å | seed violation **0.208–0.432 Å**, mean 0.24 Å across 7 molecules | **Reproduced as a class** |
| "15/20 square-planar crystals look tetrahedral until `.minimize()`" | 15/20 | **17/20** (§7.1) | **Confirmed**, understated |

The three metal figures are all within a few degrees of the claims and all show the same
seed→0.00-after-relax behaviour; the exact values were not reproducible because the conformer count
used to produce the originals is not recorded (this study fixes `n=8`). I treat the *phenomenon* as
confirmed and the *precise numbers* as not reproducible.

The organic 0.318 Å figure could not be traced to any molecule in the repo (grepped `tests/`,
`playground/`, `docs/` for `0.318`, the constraint literal, and the pair). The class was therefore
tested on a fixed, documented SMILES list; 0.318 Å sits squarely inside the measured range, so the
claim is credible but its exact provenance is unestablished.

### 7.1 The square-planar claim

Measured by taking every CN=4 tmQM structure whose **crystal** coordination sphere is planar (RMS
out-of-plane distance of the four donors from their best-fit plane through the metal < 0.35 Å;
square planar ≈ 0.0 Å, tetrahedral ≈ 0.5–0.9 Å) and comparing the raw seed's sphere to the relaxed
one.

20 of the 29 CN=4 tmQM structures have a planar crystal sphere. Median over 8 seeds each:

| Structure | crystal | seed | relax | | Structure | crystal | seed | relax |
|---|---|---|---|---|---|---|---|---|
| AKILAJ | 0.249 | **0.514** | 0.066 | | POFFOG | 0.000 | 0.316 | 0.046 |
| BUWXIC | 0.106 | **0.375** | 0.251 | | ROLWAR | 0.000 | **0.394** | 0.166 |
| GODNOD | 0.005 | **0.407** | 0.112 | | SACJEO | 0.270 | **0.691** | 0.236 |
| GUVVID | 0.188 | 0.316 | 0.233 | | TUXRUZ | 0.050 | — | — |
| HURVOI | 0.013 | **0.362** | 0.274 | | WUSXIT | 0.098 | **0.525** | 0.245 |
| IHUBUJ | 0.010 | **0.376** | 0.167 | | XAWQUH | 0.097 | **0.719** | 0.173 |
| LISVIW | 0.092 | **0.600** | 0.237 | | XEGGEY | 0.016 | **0.614** | 0.250 |
| NEWVOB | 0.000 | **0.687** | 0.124 | | XEYNIA | 0.002 | **0.463** | 0.254 |
| OFULEJ | 0.052 | **0.826** | 0.197 | | YAPZAS | 0.184 | **0.550** | 0.199 |
| PAPMEZ | 0.065 | **0.440** | 0.181 | | YEYSOK | 0.089 | **0.357** | 0.287 |

**17 of 20** planar-crystal spheres are non-planar (oop > 0.35 Å) on the raw seed and are pulled
back to 0.05–0.29 Å by the relax. **The 15/20 claim is CONFIRMED, and is if anything understated.**

Two notes:

- **TUXRUZ is in this set and is a failed embed** — its crystal sphere is planar (0.050 Å), so a
  harness that did not detect the bit-identical return would have scored it as a perfect seed *and*
  a perfect relax. This is guard G2 earning its place on live data.
- The relax does not fully reach the crystal either (0.05–0.29 Å vs the crystal's 0.00–0.27 Å), but
  the improvement is large and in the right direction on every one of the 19 embeddable structures.
  This is the sharpest single demonstration in the study that the raw seed is *not* the cleaner
  geometry.

---

## 8. Verdict

**Is the answer tighter bounds, a lighter relax, or both?**

**Neither, as stated. The measured answer is a third thing: the gap is between the bounds matrix and
the coordinates ETKDG returns from it.**

### Not tighter bounds — measured

- Triangle-smoothing tolerance is **0.000 on all 30 structures**: the constraints are already
  mutually realisable and nothing is being repaired or widened.
- **85% of angle-window violations have the realised 1–3 distance outside the matrix bound that
  already exists.** A tighter bound cannot be honoured by a solver that is not landing on the loose
  one.
- The remaining 15% is the inherent limit of encoding an angle as a 1–3 distance (legs stretch), not
  a bound that was written too loosely.

### Not a lighter relax — measured

- The relax improves M–donor distance vs crystal (0.063 → 0.006 Å, 185/186 conformers), bond lengths
  (0.025 → 0.021 Å), donor fold on net (20.2° → 17.0°) and the `donor_orientation` gate, and is
  **neutral** on heavy-atom RMSD to crystal (+0.012 Å, 96 improve / 90 worsen).
- There is **no axis measured on which the relax systematically degrades the geometry.** Lightening
  it would restore the 9.5° angle-window violation and the 0.057 Å M–donor error and buy nothing
  measured.
- **17 of 20 square-planar crystal spheres come out of the DG seed looking tetrahedral** (§7.1,
  oop 0.32–0.83 Å) and the relax pulls every one back toward planar. On this axis the raw seed is
  not merely imprecise, it has the wrong coordination geometry, and only the relax fixes it.
- The specific fear — "UFF's preferred torsions decide the output" — should appear as a systematic
  RMSD degradation. It does not.

### What the relax genuinely costs — measured

1. **21% seed attrition** (50 of 236). WIMCAA loses all 8 (UFF cannot type its borane cage); AKILAJ
   keeps 1 of 8, UTANUA 2 of 8. This is the largest real cost found and it is a *throughput and
   diversity* cost, not an accuracy one. Note this is attrition measured with `_retry=False`; on the
   default path `_reembed_until_clean` refills from fresh seeds, so a user sees fewer missing
   conformers but pays the re-embed — the underlying tear rate is the 21%.
2. **A residual fold gap.** Relaxed donor fold is 17.0° against the crystal's 9.9°, and 55 of 186
   conformers fold *worse* after the relax. Six structures degrade by 1.6–8.1° (COJKAO, AKILAJ,
   CSBRHB, NUDXOC, SOHMEJ, SIFJUO). This is the one place the original concern has real instances.

### The claim this refutes

> "the fold-wall census found the donor fold is UFF-born — the DG seed is clean"

**REFUTED as a general statement on this corpus.** The DG seed's donor fold is 20.2°, *worse* than
the relaxed 17.0°, and the relax improves fold more often than it hurts (95 vs 55). It holds only on
a minority subset — the six structures in §5 — which is presumably where the census looked. Both
figures are far from the crystal's 9.9°, so "clean" describes neither side.

(Methodological note: this measurement restores the metal before scoring, which the memory note
`rxembed-foldwall-and-donor-orientation` says is required — `.dump()` false-cleans. G5 confirms the
restore, including oxidation state.)

### What to do instead — inferred, not measured

The measurements point at the **enforcement step between the matrix and the coordinates**, not at
the matrix and not at the strength of the full relax. What is missing is something that drives the
ETKDG output onto the bounds it was already given.

The pipeline already contains a mechanism of exactly this shape. `Ensemble._settle_seeds`
(`pipeline.py:569`) is a stiff but *window-only* `restrained_uff` pass whose docstring says in as
many words: "Fixes the raw ETKDG seed sitting *outside* a tight window." It is a targeted relax
rather than a bounds edit — which matches the diagnostic, since §6 shows the bounds are already
correct and already tight. **It currently runs only inside `mc()`**, so the plain
`embed(...).minimize()` path measured here never sees it.

So the shape of the answer is closer to *a lighter, earlier, more targeted relax* than to either
option in the question — with the full UFF minimisation retained, since §4 shows it is what buys the
M–donor and bond-length accuracy. Whether promoting `_settle_seeds` ahead of `minimize()` actually
helps is **not measured here** and should be its own experiment; it would be cheap to run with this
harness.

The second, independent target is the 21% attrition, particularly the UFF-typing failures (WIMCAA:
`B_5`/`B_6` untypeable, 0 of 8 conformers survive). That is a force-field coverage problem, not a
bounds or relax-strength problem.

---

## 9. What could not be established

- **The historical claim.** "This reportedly used to be better on the direct embed" was **not
  tested** — no prior revision was measured. Nothing here speaks to whether the direct embed
  regressed.
- **Exact provenance of three cited numbers.** The 48.2° / 30.3° / 26.8° figures reproduce to within
  a few degrees but not exactly; the conformer count and revision that produced them are not
  recorded. The organic 0.318 Å could not be traced to any molecule in the repo.
- **Whether the constraint windows are physically right.** On the retain-input path they are derived
  from the input geometry, so the crystal satisfies them by construction. This study measures
  *satisfaction* of the windows, never their correctness. A SMILES-sourced metal complex, where the
  windows come from the fitted distance model instead, was not measured and could behave differently.
- **`cons` drift across `_hold_donor_chirality`.** §6 rebuilds the bounds matrix from the
  *post-embed* `ens.cons`. For a structure with a labile donor hold, the embed-time `cons` carried
  extra dummy-D constraints since released. Those would only *tighten* the matrix, so the 85% figure
  is conservative — but it is not exact for those structures.
- **Organic systems generally.** Only the single soft-window claim (§7) was tested on organics. The
  whole per-axis table is metal-complex data; a metal complex's relax carries FF machinery
  (surrogate, pulls, floors, coplanarity cap) an organic relax does not, so the balance may differ.
- **`mc()` and anything downstream of it.** Out of scope; this compares `embed` to `minimize` only.
