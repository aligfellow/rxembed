> **DECISION 2026-07-22: NOT landing the in-plane bisector - documented as a known limit.**
> The gated bisector pin fixes the sigma-aryl skew (21.5->0.1 deg) and improves henry, but it regresses a
> RIGID diphosphine-amidate chelate (test_reembed_retry_delivers_clean_geometry: 24/24->22/24 clean --
> pinning the amidate N pulls the metal, the rigid metallacycle can't absorb it, the carboxylate twists
> out of conjugation). codonor_in_plane cannot separate henry's FLEXIBLE sp3-hinged amidate (should pin)
> from this RIGID one (must not) -- the signal is global backbone rigidity, which no clean predicate
> captures. Root: fixing a *skew* needs a TIGHT symmetric pin, which is exactly the over-pinning the
> fold-wall was deliberately kept a one-sided FLOOR to avoid (a tighter wall 'propagates strain'). So the
> in-plane skew cannot be fixed without adding an over-determining constraint. Left as a known seed-level
> limit on sigma-aryl carbanion donors (a niche donor type; g-xTB corrects the final geometry). See
> [[rxembed-donor-orientation-convergence]].

# σ-aryl (phenyl carbanion) donor orientation — check + prototype

Maintainer-queued artefact (spec: `scratchpad/sigma_aryl_artefact.md`). **Prototype only — no fix committed.**
Every `src/` spike was a runtime monkeypatch, reverted; `git diff src/` is clean. Numbers below are from
`uv run --no-sync python` scripts in the shared scratchpad (`*_saryl.py`, `diag_tilt.py`, `gate_saryl.py`,
`regress_saryl.py`).

## TL;DR

- The artefact is **two** distortions, not one. A σ-aryl (phenyl carbanion) Pd sits (a) **skewed off the
  in-plane external bisector** toward an ortho (SM2: |skew| up to **21.5°**, seed-dependent) and (b) **tilted
  ~37–42° out of the ring plane** (present even when the in-plane part is symmetric).
- The **in-plane skew** is the constraint gap the maintainer named: nothing pins the donation direction. The
  fix — feed `donation_axis` into `_orient_donor` as a tight `M–D–ortho ≈ 120°` bisector pin — **fixes it**
  (SM2 21.5° → **0.1°**) and pulls ‹M–C–o› toward 120, with **no regression** and a small **improvement** on
  every other two-heavy sp2 donor (amidate N, pyridine N).
- The **out-of-plane tilt** is a *separate* distance-geometry seed artefact (the bounds matrix is
  reflection-blind about the sp2 plane, and the geometry is hypersensitive near 120°). The bisector pin only
  softens it (42° → ~35°); it is **not** eliminated and is **not** subsumed by the pin. Fully fixing it is a
  distinct, riskier change to the coplanar cap (make it a restoring pull, not a flat ±45° window). Flagged,
  not bundled.
- Unification verdict: a **genuine but partial** consolidation. In-plane directionality is now single-sourced
  from `donation_axis` (and the pin dedups the fold-wall's manual neighbour loop via `donation_axis`'s existing
  co-donor/APEX/haptic/frozen exclusions). It does **not** reduce the field count — the coplanar cap stays.

## 1. Reproduction

Route: `rx.metal(smi, geom)[0]` → `rx.embed(iso, n=1, seed=s).minimize()`, real Pd restored (the output is
connected). `rx.embed(smi)` on a bare metal SMILES does **not** surrogate (0 constraints, raw Pd, UFFTYPER
error) — the artefact lives only in the metal coordination path.

- **SM1** `[Pd]<-[c-]1ccccc1` → `linear` (one aryl donor + one VACANT vertex); ipso donor = atom 1.
- **SM2** `C[P](C)(C)(->[Pd+2](<-[Br-])<-[C-]1=CC=CC=C1)` → `trigonal_planar` (P / Br / aryl-C); ipso = atom 6.

Both ipso donors type **sp2** (`_stripped_hybridisation`), `inplane_sp2_donor=True`, and receive the coplanar
cap `(M, ipso, o1, o2, 180, 45)` + the loose fold walls `(M, ipso, o_k) = (97, 145.5)` to each ortho.

`M–C–o1/o2` = the two Pd–C(ipso)–ortho angles; `tilt` = angle of the Pd–C(ipso) bond off the best-fit ring
plane (`oop` = perpendicular distance, Å). Ideal σ-aryl: **120° / 120°, tilt 0** (Pd on the in-plane external
bisector, "the normal H direction").

| case | seeds | in-plane skew (a1−a2) | ‹M–C–o› | out-of-plane tilt | oop (Å) |
|---|---|---|---|---|---|
| SM1 | 1–8 | **0.0°** (single donor ⇒ symmetric) | 113.6 | **36.9°** (36.7–37.2) | 1.26 |
| SM2 | 1–8 | **−21.5 … +6.4°** (mean −5.8, \|skew\|max 21.5) | 111.7 | **42.1°** (39.6–44.5) | 1.41 |

So SM1's in-plane part is symmetric (there is nothing to skew against) but Pd is still **37° / 1.26 Å out of
plane**; SM2 is skewed in-plane **and** tilted out. Both ‹M–C–o› sit below 120 (Pd pulled out of plane
foreshortens both arms).

**The gate ships these as clean.** `geom.check(...).ok() == True` for every conformer: the `donor_orientation`
floor for `('C', SP2)` is 85°, so 97.9° passes, and `donor_fold.planarity` is report-only. Yet SM2's measured
planarity (52–56°) **exceeds the census p95 of 40°** (`donor_orient.CENSUS_OOP_P95`) — genuinely anomalous, not
tolerated scatter.

## 2. Located cause

**The out-of-plane is a DG-seed artefact — the FF relax never touches it.** Raw ETKDG seed tilt == relaxed tilt
to 0.1° (`diag_tilt.py`: SM1 36.8→36.8, SM2 44.5→44.5, …). So the ±45° coplanar cap's *FF torsion* exerts no
restoring force (flat-bottomed; the seed already sits inside ±45°), and the tilt is whatever the bounds matrix
seeded.

**Ablations** (`ablate_saryl.py`, seeds 1–6, monkeypatch `_orient_donor` / `_coplanar_donor` to no-ops):

| condition | SM1 tilt | SM2 \|skew\|max | SM2 tilt |
|---|---|---|---|
| baseline (cap + fold wall) | 36.9 | 21.5 | 42.5 |
| coplanar cap OFF (fold wall only) | 36.8 | 15.5 | **71.4** |
| fold wall OFF (cap only) | 36.9 | **25.7** | 44.0 |
| both OFF (surrogate excluded volume only) | 36.8 | 7.2 | 72.0 |

Reading:
- **Coplanar cap** = out-of-plane bound only. Removing it lets SM2 blow out to 58–76°; keeping it clamps to
  the **±45° flat edge (~42°)** but never pulls toward 0 (no restoring force).
- **Fold wall** `(97, 145.5)` = weak in-plane bound only. Removing it worsens SM2 skew to ±26°. Being flat and
  symmetric, it does **not** center the 120° bisector.
- **Surrogate excluded volume alone does not orient it** (SM1 still 37°).

**The missing constraint:** nothing pins the **in-plane donation direction** (the external bisector, `M–D–ortho
≈ 120°` to *both* ring neighbours). `donation_axis` already computes exactly this axis for the QA gate but is
never fed into the embed. The out-of-plane DOF *is* held in principle (coplanar cap) but only to a loose flat
±45°.

## 3. Prototype

In `_orient_donor`, for an sp2 donor with ≥2 heavy substituents `X = donation_axis(mol, d, donor_set)`, replace
the loose symmetric fold walls with a **tight bisector pin** on each arm (coplanar cap left intact):

```python
subs = donation_axis(mol, d, set(donor_set))          # already excludes co-donor / APEX / haptic / frozen
if hyb == SP2 and subs is not None and len(subs) >= 2:
    for x in subs:
        cons.angles[(metal, d, x)] = (114.0, 126.0)   # 120 ± 6: the sp2 external-bisector direction
    return                                            # this tightens/centres rule (3)'s fold walls
# ... otherwise the existing end-on / pnictogen-splay / fold-wall path, unchanged
```

This lives in rule (3) of `_orient_donor` (the fold wall): for the symmetric two-heavy sp2 case it *centres and
tightens* the same `cons.angles` M–D–X windows on the bisector and sources the substituent set from
`donation_axis`. Same field, same `Angle` mechanism (DG 1-3 bound + UFF angle wall) — no new field.

**Measured effect on the two σ-aryl cases** (`final_saryl.py`, seeds 1–8, coplanar cap kept):

| case | \|skew\|max | ‹M–C–o› | tilt range |
|---|---|---|---|
| SM1 baseline → proto | 0.1 → **0.8** | 113.6 → 114.1 | 36.7–37.2 → 34.3–35.5 |
| SM2 baseline → proto | **21.5 → 0.1** | 111.7 → **114.0** | 39.6–44.5 → 35.3–36.1 |

In-plane skew **fixed**; ‹M–C–o› pulled toward 120; out-of-plane **modestly reduced but not eliminated**
(~35°).

**Why the pin does not also fix the out-of-plane** (this bounds the maintainer's hypothesis): three fixed
distances from D, o1, o2 leave M determined only **up to reflection through the ring plane** — the distance
bounds matrix is blind to the out-of-plane sign — and the tilt is hypersensitive near 120° (algebra: `M–C–o` =
119° ⇒ 14° tilt; 117° ⇒ 25° tilt, matching a ±3° flat window's measured 24°). So a flat `[114,126]` window
permits below-120 arms and the seed keeps ~35° of tilt. The out-of-plane genuinely needs the coplanar cap to
become a **restoring pull** (soft harmonic toward the plane, not a flat ±45° window) — a separate change that
touches the census-tail cases (real conjugated donors reach 40° out of plane), so it is flagged here, not
bundled.

**Sub-sumption test (negative):** removing the two-heavy coplanar improper and relying on the pin alone
*regresses* the out-of-plane (SM2 tilt 42 → 52 at a ±12° window; `proto_saryl.py`). So the pin does **not**
subsume the coplanar cap — the out-of-plane hold must stay.

## 4. Regression controls

`regress_saryl.py`, seeds 1–4, `rx.embed(iso, n=4).minimize()`, coplanar cap kept. `clean` = conformers passing
`geom.check(..., donors=...)`.

| fixture | baseline clean | proto clean | touched sp2 two-heavy donor: fold angles baseline → proto |
|---|---|---|---|
| nitrile (sp, end-on) | 3/3 | 3/3 | — (untouched) |
| en-chelate (sp3 N) | 4/4 | 4/4 | — |
| amine (sp3 N) | 4/4 | 4/4 | — |
| acac (one-heavy sp2 O) | 2/2 | 2/2 | — |
| henry-depe (amidate N) | 4/4 | 4/4 | N23 [108.9, 122.5] → **[114.0, 122.0]** |
| henry-dppe (amidate N) | 4/4 | 4/4 | N4 [125.6, 115.4] → **[118.4, 114.1]** |
| pyridine/σ-aryl-C | 2/2 | 3/3 | C5 [123.3, 104.3] → **[114.0, 114.0]** |

- sp / sp3 / one-heavy-O / pnictogen donors are **not** two-heavy sp2 → untouched, all stay clean.
- The two-heavy sp2 donors the pin *does* touch (amidate N, pyridine/aryl C) stay clean and **improve** — skew
  reduced, arms pulled toward a symmetric ~120°. Henry's amidate/carboxylate embed is undisturbed (still 4/4).
- Baseline suite `tests/test_donor_fold.py tests/test_coplanar.py`: **59 passed** (green tree confirmed before
  the analysis).

Care point for a real implementation: the early `return` in the two-heavy branch skips the pnictogen-proton
splay / sp end-on holds — harmless here (a two-heavy sp2 donor is neither), but a full fold-in should order the
splay/end-on holds *before* the bisector `return`, or gate the pin on non-pnictogen sp2.

## 5. Does it unify / simplify the donor-orientation machinery?

**Partly — honestly, it is a tightening of an existing term plus real single-sourcing, not a field deletion.**

- **Yes, single-sourced.** In-plane directionality now comes from one mechanism reading `donation_axis` (joining
  sp end-on + pnictogen splay under the same "read the donation axis" umbrella). The pin reuses
  `donation_axis`'s co-donor / APEX / haptic / frozen exclusions, dedupping the fold wall's hand-rolled
  neighbour loop — a genuine consolidation.
- **No new field.** It is `cons.angles` + the existing `Angle` mechanism; for the symmetric two-heavy case it
  *replaces* the loose asymmetric fold walls with a tight centred one.
- **But it does not reduce the term count.** The coplanar cap (out-of-plane) is orthogonal and stays — the pin
  handles only the in-plane DOF, and the out-of-plane is a distinct reflection-blind DG-seed problem the cap
  owns. Calling the whole thing "one mechanism from `donation_axis`" would overstate it: two DOFs, two holds.

**Recommendation.** Land the bisector pin as the clean, low-risk fix for the **in-plane skew** (the nameable,
directional artefact `donation_axis` directly answers). Treat the residual **out-of-plane tilt** as a separate
thread: it needs the coplanar cap to gain a restoring pull (or a σ-donor-specific tighter cap), which touches
the census-tail calibration and should be its own measured change.
