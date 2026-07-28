# Derive conjugated-sp2 donation — do not extend the element list

> **SUPERSEDED IN PART 2026-07-21 by review R3** (`docs/findings/review-batch.md` §R3, commit `ce2ce71`).
> The predicate below was **too narrow**: requiring the donation axis to be π-conjugated dropped the cap
> for *isolated* sp2 O/N donors (a simple ketone/aldehyde/ketimine, `GetIsConjugated()==False`), which a
> metal still binds in-plane via its σ lone pair. The correct predicate is just **sp2** (element-agnostic,
> no π test), so `geometry.conjugated_sp2_donor` was renamed **`inplane_sp2_donor`** with body
> `hyb.get(d) == SP2`. T5's real gain — extending the cap to the aryl-carbanion C and thione S — stands
> (both are sp2). `_CONJUGATING_LP` and the `_CONJ_O_MDC_ANGLE` deletion below are unaffected. Read this
> doc for the T5 reasoning, but the live predicate is `inplane_sp2_donor`.

Task T5. Date 2026-07-20, branch `rdkit-embed-kernel`. Scripts preserved in
`playground/t5_conjugation/` (`t5_repro_henry_s6.py`, `t5_census_s_sp2_v2.py`, `t5_mdc_angle_probe.py`).
All numbers measured against the **current** tree (the pipeline now relaxes seeds into their windows at
the return point), with explicit RDKit seeds.

## The defect (reproduced)

`_coplanar_donor` (the conjugated-sp2-donor coplanarity cap) gated on `_CONJ_DONORS = frozenset({7, 8})`
(N, O). Two conjugated sp2 donors are not element 7 or 8 and were dropped:

- an **aryl carbanion** ipso C (`[c-]1ccccc1`, aromatic, sp2) — carbon (6);
- a **thione / thioamide C=S** sulfur — sulfur (16).

Reproduction, henry Ni(II) thiosemicarbazone `C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1`,
S6 metal-out-of-conjugated-plane over 8 seeds × 8 conformers (embed → minimize):

| donor | before (uncapped S) | after (capped) |
|---|---|---|
| **S6** median / mean / **max** | 26.0 / 23.8 / **62.2°** | 25.0 / 20.0 / **39.3°** |
| carboxylate O median / mean / max | 20.4 / 16.3 / 39.9° | **6.6** / 12.8 / 41.6° |

The task cited "~32.6°" for S6; the median/mean here are ~24–26° with a gross **62°** tail — same defect,
different aggregation. The fix caps the gross tail at ~the census p95 (40°). The carboxylate O **improves**
(median 20.4 → 6.6°): its own constraints are byte-identical (see below), so the gain is the *ripple* of
now holding the S in plane — the square-planar sphere settles more symmetrically.

## The derived predicate, and where it lives

`geometry.conjugated_sp2_donor(mol, d, hyb=None)` — a donor is conjugated when it is **sp2** (the
two-estimator, metal-stripped `_stripped_hybridisation` the fold gate already uses) **and its donation
axis is in a π system** — it is aromatic, or has a conjugated bond to an sp2 heavy neighbour. Element-
agnostic: the graph answers it for carbon, sulfur, nitrogen, oxygen at once. No element table.

It lives in **`geometry.py`** (the perception layer). `metal.py` already imports geometry lazily as `_geo`
(geometry imports `metal.APEX`, so a module-level import the other way would cycle); `_coplanar_donor` now
calls `_geo.conjugated_sp2_donor` the same way `_orient_donor` already calls `_geo._stripped_hybridisation`.
This is the unification the task asked for: enforcement now **defers to** the same perception ruler that
feeds the fold gate, so the two cannot drift.

## The two `{7, 8}` sets — one derived away, one is genuine data (premise partly REFUTED)

The brief's premise was that **both** copies of `{7, 8}` are the same rule and both disappear. Measurement
says only one is a removable duplicate:

- **`metal.py` `_CONJ_DONORS = {7, 8}`** — pattern matching. You would extend it by adding 6 and 16 (the
  exact AGENTS.md symptom). **Deleted**, replaced by `conjugated_sp2_donor`.
- **`geometry.py` `_CONJUGATING_LP = {7, 8}`** (in `_pi_hybridisation`) — **genuine period-2 physics,
  kept.** It fires only for an atom with *no π bond of its own* whose *lone pair* planarises into an
  adjacent π system (amide N, carboxylate anti-O). A period-2 2p lone pair conjugates; a period-3 one does
  not — PPh₃ is pyramidal, and *adding* S/P here would type every triarylphosphine sp2 and void the gate on
  exactly the phosphine donors these catalysts are made of. This set is correctly bounded (its heavier
  congeners are deliberately excluded), which is data, not a case list.

Crucially, `_CONJUGATING_LP` is **irrelevant to the two dropped donors**: the aryl carbanion is sp2 via the
aromatic branch and the thione via the one-π-bond branch — neither reaches the lone-pair branch. So there
was never a reason to touch it. It stays as the single canonical home of the period-2 rule, which the
derived predicate consumes through `_pi_hybridisation`. **Net: one `{7, 8}` deleted, one kept as data; the
"drift" the brief worried about is closed by the shared perception, not by a second deletion.**

## The `('S', SP2)` census — uncalibrated, gate abstains

Measured M-S-X fold and metal-out-of-plane for sp2 sulfur donors across the 144-structure corpus
(`OIN-SMILES/tests/integration/tmQM/*.xyz` + `tests/fixtures/*.xyz`; 2 unreadable → 142 scanned), using the
same `_stripped_hybridisation` class and the same `donation_axis` exemptions the fold gate applies
(`t5_census_s_sp2_v2.py`):

- all sp2 S donors: **14** M-S-X records from 7 structures;
- **non-exempt** (what the gate would judge): **8** records from 4 structures (HURVOI, LISVIW, TUDGUU,
  TUXRUZ), M-S-X **89–136°**, a **47°-wide grab-bag** (dithiolene / polythioether-type);
- the **conjugated subset** — the thione-like population henry S6 belongs to, where the D-X bond is a
  conjugated sp2–sp2 bond (`bond.GetIsConjugated()`, the predicate `_planarity_dev` uses): **n = 0**.

**Verdict: leave `('S', SP2)` out of `_FOLD_WINDOW` (uncalibrated).** The population the change actually
targets has **zero** corpus representatives; the 8 records that exist are a chemically distinct, non-
conjugated class whose 47°-wide window would mis-judge a thione. Fabricating a row from them is the "null
measurement" AGENTS.md warns against. The fold **gate** (`donor_orientation`) and the fold **wall**
(`_orient_donor` rule 3) therefore abstain on sp2 S — verified: S6 is reported `unknown` by `donor_fold`,
and no M-S-X window is written to `cons.angles`.

**Graceful degradation, verified.** The coplanarity **cap** reads only `cons.coplanar` (an FF torsion that
re-detects the in-plane well per conformer) — **not** `_FOLD_WINDOW`. So the uncalibrated thione still gets
capped (S6 receives a `cons.coplanar` entry and its 62° tail collapses to 39°) even though the gate/wall
abstain. Census abstention and cap enforcement are decoupled by construction.

## `_CONJ_O_MDC_ANGLE` — dead code, deleted (the note is right; here's why the carboxylate is unaffected)

The tension: one note says delete `_CONJ_O_MDC_ANGLE`; a code comment says a carboxylate's coplanarity cap
*needs* an M-O-C angle wall or the DG plane bound "can't get a fix". Resolved by measurement
(`t5_mdc_angle_probe.py`):

`_orient_donor` (the fold wall) runs **before** `_coplanar_donor` in the same donor loop and, for every
**calibrated** `('O', SP2)` donor, sets `(metal, O, C)` = the fold window **(102.0, 156.8)**. The old
`_coplanar_donor` line then did `cons.angles.setdefault((metal, O, C), _CONJ_O_MDC_ANGLE)` — a **no-op**,
because the key already exists. Measured realised value of `(metal, O, C)` on henry (κ1) **and** acac_ni
(κ2): **(102.0, 156.8)** in both; `_CONJ_O_MDC_ANGLE` **(105, 140)** never reaches `cons.angles` in any
fixture.

So **both notes are right about different things**: the plane bound *does* need M-O-C pinned (true), but
the fold wall already pins it — `_CONJ_O_MDC_ANGLE` only ever restated that key. **Deleted.** The deletion
is bit-identical on every current fixture (the golden harness stays green, below), confirming it was dead.
It is **not** generalised to S: an uncalibrated `('S', SP2)` gets no fold wall, so the thione's coplanar DG
`dg_post` finds no M-S-C angle and writes no DG bound — the FF torsion holds it alone. This is exactly why
the earlier REFUTED two-edit fix "got its gain from deleting the M-S-C wall": for S there is no wall to
begin with, and the seed bound is simply declined rather than guessed.

## Golden — did NOT move

`tests/golden/` : **23 passed**, bit-identical. No golden fixture contains a thione S or aryl-carbanion C,
so no fixture's cap membership changed; and `_CONJ_O_MDC_ANGLE` was dead, so its deletion changes no bounds
matrix. The change is therefore invisible to the golden set — the correct outcome, and evidence the
deletion was truly redundant.

## What changed

- **added** `geometry.conjugated_sp2_donor` (the one derived predicate);
- **deleted** `metal._CONJ_DONORS = frozenset({7, 8})` (pattern matching → predicate);
- **deleted** `metal._CONJ_O_MDC_ANGLE` + its `setdefault` (dead code: the fold wall pre-sets the key);
- **kept** `geometry._CONJUGATING_LP = frozenset({7, 8})` (genuine period-2 lone-pair data);
- **tests**: `test_a_thione_sulfur_donor_gets_the_cap`, `test_an_aryl_carbanion_carbon_donor_gets_the_cap`,
  `test_the_uncalibrated_thione_cap_is_decoupled_from_the_fold_window` (all red-first verified); the
  `{7,8}`-encoding assertion in `test_a_non_conjugated_or_haptic_or_sp_donor_gets_no_cap` now asserts the
  derived predicate.
