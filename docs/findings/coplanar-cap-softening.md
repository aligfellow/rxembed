# Coplanar-cap softening — the measured setting

> **RECOMMENDATION REJECTED 2026-07-21 (maintainer).** The crowding-conditional FC below *works
> numerically* but is a heuristic **branch** (`if ≥3 donors`), which the house rules resist ("a
> capability is one more argument, not a branch; if a change needs a new code path the abstraction is
> wrong"). The interference is a symptom that the **constraint setup is over-determined** — multiple
> per-donor coplanarity caps each pulling the metal into a different plane on a crowded chelate. The
> fix is being taken at the constraint-architecture level, NOT this conditional. The measurements here
> stand as evidence of the over-determination and of what FF-force does; the recommended *action* does
> not. The silent-fusion gate hole this study surfaced was fixed independently (`8338de1`).

Follow-up to `karoline-metal-embed.md` (§ cases 2/3/4). The metal **coplanarity cap**
(`metal._coplanar_donor` → `cons.coplanar`, enforced in the DG bound `mechanisms.Coplanar.dg_post`
and the FF torsion `mechanisms.Coplanar.ff_terms`) over-constrains a **crowded, multi-conjugated-donor**
chelate: it twists the bulky backbone and the relax pays for it with `planarity`/`conjugation` gate
flags elsewhere. It is correct and load-bearing on ordinary κ1/single-chelate cases (it is what keeps
henry's amidate/carboxylate metal in plane). This measures the right softening instead of guessing it.

**Bottom line.** A *global* softening (of either lever) fixes the crowded cases but **silently pulls the
control metals out of plane** and breaks the FC-pin test — the exact failure to avoid. A **crowding-
conditional** softening — reduce the FF torsion force constant `_COPLANAR_FC` from 10 → **1 kcal/rad²
only on a metal carrying ≥3 conjugated sp2 donors** — fixes the crowded cases and the silent epoxide
collapse while leaving every 2-donor control **bit-identical to baseline**. Recommended.

---

## The two levers, and the null-measurement guard

| lever | constant | enters | how it is patched in the spike |
|---|---|---|---|
| **window** | `metal._COPLANAR_CAP` = 45° | baked into each `cons.coplanar` tuple at `rx.metal()` time → read by **both** `Coplanar.dg_post` (DG 1,4 bound) and `Coplanar.ff_terms` (FF torsion window) | set `metal._COPLANAR_CAP` *before* `rx.metal()` |
| **force** | `mechanisms._COPLANAR_FC` = 10 kcal/rad² | **FF only** — the torsion force constant in `Coplanar.ff_terms` (for a metal system `Sp2Planar`/`ConjugationCap` early-return, so this touches nothing else) | set `mechanisms._COPLANAR_FC` before `.minimize()` |

**Guard (proven, not assumed).** Patching `metal._COPLANAR_CAP` from 45→70 changes the cap in every
`cons.coplanar` tuple (`{45.0}`→`{70.0}`, case2, 4 entries). Patching `mechanisms._COPLANAR_FC` from
10→3 changes the force written to `UFFAddTorsionConstraint` (spied: `{10.0}`→`{3.0}`). Both take effect
on the real pipeline path. `src/` was never edited; all measurements are monkeypatch (see
`playground/cap_softening/`).

## The crowding signal is clean and structural

Number of **capped sp2 donors** on the metal = `len({e[1] for e in cons.coplanar})` (each in-plane sp2
donor gets exactly one entry). It separates the controls from the problem cases with no overlap:

| complex | capped sp2 donors | role |
|---|---|---|
| HENRY (diphosphine·amidate) | **2** (carboxylate O, amidate N) | control — cap needed |
| KETONE (`CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1`) | **2** (ketone O, pyridine N) | control — cap needed |
| PICO (picolinate·en) | **2** (carboxylate O, pyridine N) | control — cap needed |
| RDP (rigid-diene diphosphine, the FC-pin test) | **2** (carboxylate O, amidate N) | control — cap needed |
| case 1 (thione) | 3 | crowded |
| case 3 / case 4 | 3 | crowded |
| case 2 | 4 | crowded |

The predicate is `n_capped ≥ 3`. It is derived (any element the derived `inplane_sp2_donor` accepts
counts — no element list), single, and available exactly where the FC is applied. The epoxide (case 4,
3 capped) requires `≥3`, not `≥4`: at `≥4` the collapse stays at baseline (11/30).

---

## Sweep A — global levers (exploratory, n=6, seed 0xF00D)

flag-rate = fraction of post-`minimize` conformers `geom.check` flags, across all isomers, with the
kind histogram; epox = # of iso3 seeds collapsing the ester O2–C3–O4 < 90° (the silent "epoxide")
out of 30; thione = case-1 Ni–S–C–N dihedral folded to [0,90] (median/p95/max); controls = per capped
donor **out-of-plane median/max** (low = metal held in plane = cap working).

| setting | case2 | case3 | epox | thione | HENRY O / N | KETONE O / N | PICO O / N |
|---|---|---|---|---|---|---|---|
| **baseline** cap45 fc10 | 75% pl13,cj6 | 75% cj5,pl8 | **11/30** | 24/42/45 | 0/39 · 8/42 | 28/35 · 0/42 | 1/9 · 1/13 |
| global fc5 | 58% | 88% | 1/30 | 5/36/39 | 8/42 · 3/42 | 8/35 · 0/42 | 2/19 · 1/17 |
| global fc3 | 25% | 64% | 0/30 | 14/40/41 | **21**/43 · 9/42 | 27/35 · **18**/42 | 0/7 · 1/4 |
| global fc1 | 25% | 58% | 0/30 | 20/35/37 | **14**/44 · **13**/42 | 26/35 · 0/42 | 0/27 · 0/15 |
| global fc0 | 50% | 50% | 0/30 | 14/34/48 | **27**/41 · **11**/49 | **35**/42 · **39**/51 | 3/22 · 1/17 |
| global cap60 | 50% | 70% | 0/30 | 7/38/42 | 1/27 · **31**/51 | **41**/49 · 0/50 | 0/6 · 0/5 |
| global cap75 | 42% | 50% | 0/30 | 5/55/**73** | **25**/44 · 4/45 | **45**/55 · 0/51 | 1/28 · 0/7 |
| global cap90 | 33% | 89% | 0/30 | 17/42/57 | 10/**77** · 3/51 | **37**/54 · 0/51 | 2/25 · 1/5 |
| crowd≥3 fc3 | 25% | 64% | 0/30 | 14/40/41 | **0/39 · 8/42** | **28/35 · 0/42** | **1/9 · 1/13** |
| crowd≥3 fc1 | 25% | 58% | 0/30 | 20/35/37 | **0/39 · 8/42** | **28/35 · 0/42** | **1/9 · 1/13** |
| crowd≥3 fc0 | 50% | 50% | 0/30 | 14/34/48 | **0/39 · 8/42** | **28/35 · 0/42** | **1/9 · 1/13** |
| crowd≥3 cap75 | 33% | 43% | 0/30 | 21/37/49 | **0/39 · 8/42** | **28/35 · 0/42** | **1/9 · 1/13** |
| crowd≥3 cap90 | 42% | 55% | 0/30 | 17/45/49 | **0/39 · 8/42** | **28/35 · 0/42** | **1/9 · 1/13** |
| crowd≥4 fc1 | 25% | 75% | **11/30** | 24/42/45 | =baseline | =baseline | =baseline |

Reading the bold:
- **Global softening silently breaks the controls.** Every global row raises a control metal's *median*
  out-of-plane far above baseline: HENRY carboxylate-O 0°→14–27°, HENRY amidate-N 8°→up to 31° (cap60),
  KETONE pyridine-N 0°→**39°** (fc0), KETONE ketone-O 28°→45° (cap75), HENRY O max→**77°** (cap90). This
  is the cap failing on henry — the silent failure the task names.
- **The crowd≥3 rows leave all three controls literally unchanged** (identical digits to baseline),
  because the predicate never fires on a 2-donor metal.
- **Window-widening is the worse lever.** It gives up more of the fix (crowd cap75 case2 33% vs crowd fc1
  25%) and rides the thione max out to 49–73° — a flat-bottomed window removes the restoring force out to
  its edge (the recurring "angular-spring" pattern), whereas the FF force keeps the metal biased toward
  the plane, only gently.
- **The epoxide is killed by any softening** (baseline 11/30 → 0/30) and the predicate must be `≥3` to
  cover case 4 (crowd≥4 leaves it at 11/30).
- **FC=0 is worse than FC=1** on case2 (50% vs 25%): a weak residual cap beats none.

## Sweep B — confirmation (reliable, n=10, seeds 0xF00D / 0xBEEF / 0x1234)

The n=6 snapshot is coarse (≈8–12 conformers/case). At n=10 × 3 seeds the flag rate is stable and
essentially seed-insensitive:

| setting | case2 (3 seeds) | case3 (3 seeds) | RDP pin test (n=6 → clean) |
|---|---|---|---|
| baseline | 55 / 55 / 55 % | 67 / 67 / 64 % | 6/6 |
| **crowd≥3 fc1** | **40 / 40 / 40 %** | **58 / 50 / 47 %** | **6/6** |
| crowd≥3 fc2 | 50 / 45 / 50 % | 82 / 75 / 81 % | 6/6 |
| crowd≥3 fc3 | 50 / 60 / 55 % | 67 / 58 / 60 % | 6/6 |

FC=1 is the sweet spot among {1,2,3}: lowest and most stable case2 (40%, zero seed spread) and lowest
case3 (~52%). All crowd settings keep the **RDP FC-pin test** (`test_reembed_retry_delivers_clean_
geometry`, a 2-capped rigid-diene diphosphine) at 6/6 clean — because the predicate leaves it at
FC=10. (Its comment "at 5 a diphosphine-on-rigid-diene chelate tears" is *why a global* FC drop is
unsafe; the conditional never touches it. Measured directly: global cap75 drops that test to 4/6 clean.)

## Control in-plane deviation — baseline vs crowd≥3 (n=10, 5 seeds)

Bit-identical, proving the softening does not reach the controls:

| control | donor | baseline med/max | crowd≥3 med/max |
|---|---|---|---|
| HENRY | carboxylate O | 1.0° / 39.5° | 1.0° / 39.5° |
| HENRY | amidate N | 8.6° / 42.3° | 8.6° / 42.3° |
| KETONE | ketone O | 25.0° / 35.3° | 25.0° / 35.3° |
| KETONE | pyridine N | 0.3° / 42.3° | 0.3° / 42.3° |
| PICO | carboxylate O | 2.5° / 19.6° | 2.5° / 19.6° |
| PICO | pyridine N | 0.6° / 15.4° | 0.6° / 15.4° |

All control medians stay low (metal in plane) and maxes at the census p95 band (~40°) — the cap keeps
doing its job everywhere it is needed.

## Sweep C — is the DG seed bound worth softening too? (n=10, 3 seeds)

The crowd rows above soften only the **FF force** (DG ±45° 1,4 seed bound intact). Testing the DG bound
on crowded metals as well:

| crowded setting | case2 | case3 | epox | controls |
|---|---|---|---|---|
| baseline (fc10, DG on) | 55% | 65% | 12/40 | in-plane |
| **A: fc1, DG on** (recommended) | 40% | **52%** | 0/40 | untouched |
| B: fc1, DG off | **25%** | 63% | 0/40 | untouched |
| C: fc10, DG off | 43% | 72% | 0/40 | untouched |
| D: fc0, DG off (cap fully off on crowded) | 30% | 61% | 0/40 | untouched |

There is a **case2/case3 tension in the DG lever**: dropping the seed bound helps case2 (40→25%) but
hurts case3 (52→63%). No combination clears both. FF-only (A) gives the best case3 (the more stubborn,
more severe case — diagnosis had it at 100%) with a solid case2, and is a **one-number change in one
place**. (Removing the DG bound also kills the epoxide by itself — combo C, fc10+DGoff — so the ester
collapse has *both* a seed and a relax contributor; either softening removes it.)

---

## Recommendation

**Crowding-conditional, FF-force only.** In `mechanisms.Coplanar.ff_terms`, when the metal carries
≥3 capped sp2 donors, use a reduced force constant:

```python
def ff_terms(self, ff, cons, conf, fc):
    from rdkit.Chem import rdMolTransforms
    crowded = len({e[1] for e in cons.coplanar}) >= _CROWDED_DONORS   # _CROWDED_DONORS = 3
    kfc = _COPLANAR_CROWDED_FC if crowded else _COPLANAR_FC           # 1.0 vs 10.0 kcal/rad²
    for i, j, k, w, _anchor, cap in cons.coplanar:
        phi = rdMolTransforms.GetDihedralDeg(conf, i, j, k, w)
        lo, hi = _coplanar_window(phi, cap)
        ff.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, kfc)
```

Leave `_COPLANAR_CAP` (±45° window) and `Coplanar.dg_post` (the DG seed bound) **unchanged**.

- **Global vs conditional:** conditional. A global FC≤3 or any window-widening measurably pulls the
  control metals out of plane (HENRY O median 0°→14–27°, KETONE N 0°→39°) and a global cap-widen breaks
  the FC-pin test (4/6 clean). Only a conditional keeps the controls untouched.
- **Which conditional shape:** #2 (crowding-conditional), on a clean structural predicate, not a case
  list. The signal — number of conjugated sp2 donors on the metal — is what the physics says
  over-determines the plane: two in-plane donors can share the metal, but ≥3 mutually tilted donor
  planes generically cannot all contain it, so at full stiffness the relax satisfies them by twisting
  the conjugated backbone. Softening the per-donor force where ≥3 compete lets the backbone settle.
- **Predicate (exact):** `len({e[1] for e in cons.coplanar}) >= 3`, evaluated inside `Coplanar.ff_terms`
  (`cons.coplanar` is fully built by then; each capped donor contributes exactly one entry).
- **Value:** `_COPLANAR_CROWDED_FC = 1.0` kcal/rad² (a 10× reduction). FC=1 is the measured sweet spot:
  FC≥2 under-softens (case2 45–55%), FC=0 over-softens (case2 back to 50% — a weak residual cap still
  biases usefully; "bias the seed, let energy decide").

### Measured effect on all five cases (crowd≥3 fc1, n=10 × 3 seeds unless noted)

| case | capped | baseline | recommended | verdict |
|---|---|---|---|---|
| **case 2** (over-flagged) | 4 | 55% flagged (pl/cj) | **40%** flagged | improved; still not clean |
| **case 3** (over-flagged) | 3 | 65% flagged (pl/cj) | **52%** flagged | improved; substantial residual (below) |
| **case 4** (epoxide) | 3 | 12/40 O–C–O collapse | **0/40** | fixed |
| **case 1** (thione, cosmetic) | 3 | 24/42/45° spread | 20/35/37° (tightens) | slightly tighter, no regression |
| **CONTROL** HENRY/KETONE/PICO/RDP | 2 | in-plane, pin 6/6 | **bit-identical**, pin 6/6 | preserved |

### Honest residual — case 3 is only partly a cap problem

No cap setting clears case 3 below ~50%. Its flags are only *partly* cap-driven: the diagnosis names
the **phantom-floor relief** and a **clash** as further contributors, and full cap removal here still
leaves ~60%. So the cap softening is **necessary but not sufficient** for case 3 — the recommended
setting takes it 65%→52% without harming the control, and the remaining case-3 flags need a *different*
mechanism (floor relief / the +1 clash), which is out of scope for this cap-softening task. This is the
partial "shape 3" outcome for case 3: a constant fixes what the cap owns, and no cap constant fixes the
rest.

### Provenance

Spikes and raw results in `playground/cap_softening/` (`capsweep_lib.py` + drivers; `res_*.jsonl` =
Sweep A, `capsweep_confirm.jsonl` = Sweep B, `capsweep_dgexp.jsonl` = Sweep C). All numbers are
monkeypatch measurements on the installed env; `src/` untouched. Absolute flag rates differ from the
diagnosis (baseline 55%/65% here vs its 71%/100% at n=8 single-seed) — trust the **relative** effect and
the n=10 × 3-seed confirmation.
