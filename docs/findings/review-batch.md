# Adversarial review of the Phase-1 correctness batch

Two independent worktree-isolated reviewers (robustness + chemistry) attacked the batch at `ccef4f7`
(307 passed / 2 skipped). Most of the batch was **refuted clean**: `bonding_ok` per-pair keying, the
`dg_floors` min-merge collapse (bounded below by the covalent sum), conjugation perception on
allene/cumulene/ketenimine/CO₂, the permutation count on CN8, `_rescue_torn`'s never-worse guarantee,
`_seeds_relaxed` bookkeeping, the phantom-floor physics, `ConjugationCap`'s kekulization/well-choice,
and T7's decomposition. Four issues survived. Reproductions were in the reviewers' scratchpads.

## Resolutions (maintainer decisions folded in)

### R1 — guard-gap in `minimize()`'s single-point branch (robustness, MEDIUM-LOW) → FIX
`_relax_constrained` wraps `restrained_uff` in `try/except RuntimeError` to degrade gracefully on an
untypable/hypervalent core; the single-point branch taken when `_seeds_relaxed=True` calls
`restrained_uff` **without** that guard, so it crashes where the old path degraded. Low natural
reachability (RDKit's UFF builds even for actinides; only fault-injection triggered it), but a real
asymmetry the sibling guard's own comment says is genuine. **Fix:** mirror the guard.

### R2 — the retain-input `ids[0]` is now relaxed (robustness, LOW-MEDIUM) → KEEP, MAKE CLEAR
`rx.embed(crystal_xyz)` with no user constraints retains the input arrangement at `ids[0]`; the seam
now relaxes it (0.68 Å Kabsch-aligned internal distortion; M-donor sphere held <0.01 Å so the
*arrangement* survives). **Maintainer: keep relaxing — that makes scoring valid and fair (an
unrelaxed `ids[0]` would score perfectly against itself) — but make it CLEAR** (a log line / docstring
so a user knows their input was relaxed, not returned pristine). No behaviour change; a
clarity/logging change.

### R3 — isolated sp2 carbonyl/imine metal donors lost the coplanarity cap (chemistry, MEDIUM) → **DONE** (`ce2ce71`)
Fixed: `conjugated_sp2_donor` → `inplane_sp2_donor` (body `hyb.get(d)==SP2`, element-agnostic). Isolated
ketone/aldehyde/imine donors regain the cap (uncapped fold acetone O 48.5°/imine N 70.7° → capped
35.3°/42.3°). No over-broadening (pure π-donors go through the centroid path). 310 passed, golden
bit-identical, red-first verified. The thione S (user's related case) is unchanged — cap fires, median
6.7° in-plane; its tail is the ±45° soft-window spread, a separate window-tuning question.

Original diagnosis (kept for the record):
T5's `geometry.conjugated_sp2_donor` gates the metal coplanarity cap on the donation axis being in a
π system. But a metal binds **any** sp2 O/N donor from its **in-plane σ lone pair** — conjugation is
not required. RDKit marks an *isolated* ketone/aldehyde/ketimine C=X as `GetIsConjugated()==False`, so
these now lose the cap (verified: `rx.embed("CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1", ...)` caps the pyridine
N only, drops the acetone O; `C/C=N->[Zn]...` drops both imine donors). Lewis-acid activation of a
simple carbonyl and simple imine ligands are in-scope motifs. The old `_CONJ_DONORS={7,8}` capped these
correctly. **The predicate over-narrowed** — the physically correct test is "sp2 donor with an in-plane
lone pair" (≈ sp2 donor), element-agnostic, NOT "sp2 *and* π-conjugated." Masked because no fixture has
an isolated-carbonyl/imine metal donor. **Fix:** re-derive the predicate; add the missing-fixture test.

### R4 — `Sp2Planar` flattens a genuinely-bowled sp2 system — **DONE** (fc kept at 10, honestly documented)
Hold-at-seed's safety claim ("can't flatten a genuine pucker") was **refuted**: ETKDG seeds curved sp2
systems FLAT, so holding-at-seed pins them flat (a flat corannulene ships silently). Swept
`_SP2_HOLD_FC` 10 → 0.01: **the two demands do not overlap** — the chb thiourea stays planarity-clean
down to fc 3, but corannulene's ~19° bowl needs fc ≤ 0.01 to re-form past the window edge, and at fc
0.01 the thiourea pucker is back (30/40). The tension runs through the **window** (±5°), not the fc —
ETKDG seeds the bowl flat, so hold-at-seed holds it flat regardless of fc; lowering fc only sheds
thiourea margin for ~0° of bowl. **Conclusion: keep fc = 10** and document corannulene as an
out-of-organocatalysis-domain limitation. The `Sp2Planar` docstring + fc/window comments now state the
honest behaviour ("preserves a curve the seed HAS, does not re-form one it LACKS"). The safety test was
rewritten (`test_hold_at_seed_preserves_a_genuine_pucker` → `test_flat_seed_bowl_is_held_near_flat`) to
start from the **flat ETKDG seed** (the real failure the old pre-bowled test never exercised) and assert
the measured band (rides the ±5° window edge, neither ~0° nor the ~19° bowl) — red both if over-stiffened
to ~0° or softened enough to leak the bowl (the same lever that returns the thiourea pucker). A
segfault in the test's dead `bare.Minimize` block (a force field on an un-held temporary Mol) was
removed. Golden bit-identical (fc unchanged).
