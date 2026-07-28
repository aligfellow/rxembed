# Donor-orientation unification — does one `donation_axis` mechanism subsume the pile?

Maintainer's concern: donor orientation has fragmented into per-hybridisation special cases, and a sixth gap
(an sp3 amine folding its protons onto the metal) has appeared. Hypothesis to test: all of these are one
statement — *put M on the donor's `donation_axis`, hold its substituents at the ideal angle away* — and one
`donation_axis`-driven mechanism can **subsume** the special cases rather than adding a seventh.

**Prototype only — no fix committed.** Every `src/` spike is a runtime monkeypatch of `_orient_donor`
(`scratchpad/proto_orient.py`), reverted; `git diff src/` is empty. Numbers are from `uv run --no-sync python`
over seeds 1–8 (`scratchpad/measure_all.py`, `scratchpad/measure_karoline.py`, `scratchpad/sptight_test.py`).

## TL;DR

- **The amine bug is real and silent.** An sp3 amine `[NH2]` donor's two protons are held off M by **nothing**:
  the fold wall skips protons (`donor_orient.py:305`) and the proton-splay hold is scoped to pnictogen P/As/Sb
  only (`:295`). Walling only the one heavy neighbour leaves the H/H/lone-pair tripod free to roll about the
  M–N–Nheavy axis. Measured (Ni amine, seeds 1–10, minimized): median M–N–H **100 / 128** (ideal 109/109),
  min M–N–H **65°** (a proton folded onto M), **4/33 fully inverted** (both H's on M, lone pair pointing away),
  and **0/33 gate-flagged** — `donor_orientation` judges only the heavy substituent (N4 at ~108°, fine) and
  skips protons, so the fold ships as clean.
- **The unification of `_orient_donor`'s three holds is a GENUINE reduction: 3 branches → 1 loop + a data
  table, deleting 3 constants, and it fixes the amine for free.** Measured, zero-regression, with two caveats:
  the pnictogen hold (rule 2) collapses cleanly; the sp end-on hold (rule 1) collapses only if sp keeps a tight
  wall floor **as data** (else the nitrile regresses 165°→152°).
- **The BROADER hypothesis — "one `donation_axis` mechanism subsumes everything" — is REFUTED.** The out-of-plane
  coplanar cap (`_coplanar_donor`) and the in-plane σ-aryl bisector are **different DOFs** that do not fold into
  the substituent-walling loop. The proton-inclusion unification leaves the σ-aryl skew **unchanged** (18.6°;
  σ-aryl has no protons); the σ-aryl finding already proved the bisector does not subsume the coplanar cap.
- **Recommendation: unify `_orient_donor` (Option A below)** — it is exactly the "pile" the maintainer named,
  and it collapses cleanly. Land the amine fix inside that collapse. Keep the coplanar cap and (separately) the
  σ-aryl bisector as the two irreducible, distinct-DOF pieces.

---

## 1. Machinery inventory (`constraints/donor_orient.py`)

Two enforcement functions, called per donor from `metal.py:898` (`_orient_donor`) and `:903` (`_coplanar_donor`).
Both write **flat-bottomed walls** (no restoring force — bias the seed, let energy decide).

| # | rule (`donor_orient.py`) | DOF it controls | window | fires on |
|---|---|---|---|---|
| 1 | `_orient_donor` **sp end-on** (`:291-294`) | M–D–X angle → 180 | `_SP_DONATION` (165,180) | `hyb==SP`, each **heavy** nbr |
| 2 | `_orient_donor` **pnictogen splay** (`:295-298`) | M–D–H angle → ~109 | `_PROTON_SPLAY` (95,130) | atom ∈ `_HIGH_BARRIER_DONORS` {P,As,Sb}, each **proton** |
| 3 | `_orient_donor` **fold wall** (`:299-309`) | M–D–X angle → off M | `_FOLD_WALL_FLOOR[cls]`..`_FOLD_WINDOW[cls][1]` | calibrated (elem,hyb), each **heavy** nbr (protons skipped `:305`, apex skipped `:307`) |
| — | `_coplanar_donor` **out-of-plane cap** (`:312-363`) | dihedral M-out-of-sp2-plane | `_COPLANAR_ANCHOR`±`_COPLANAR_CAP` (180±45) | `inplane_sp2_donor` |

Supporting pieces (all read the one metal-stripped hybridisation ruler `_stripped_hybridisation`, `:130`):

- **`donation_axis`** (`:224`) — already computes a donor's judgeable heavy substituents X (excludes co-donor /
  APEX bite / haptic / bridging / frozen / protons), returning `None` for a donor with no axis. Used today only
  by the **gate** (`coordination._donor_walk:277`), never fed into the embed.
- **`codonor_in_plane`** (`:197`) — the free-DOF skip: True when a co-donor of the same metal lies in this
  donor's sp2 plane (the bite already pins it). Used by the coplanar FF skip (`mechanisms.py:273`). **Orthogonal.**
- **`inplane_sp2_donor`** (`:176`) — the coplanar-cap predicate (`== SP2`).
- **`_FOLD_WINDOW` / `_FOLD_MEDIAN` / `_FOLD_WALL_FLOOR`** (`:54-84`) — the census DATA (721 angles, 121 crystals).
  `_FOLD_WALL_FLOOR = min(gate_floor + 12, median − 5)` is the seed-bias floor, distinct from the gate floor.

**Where they overlap / leave gaps.** Rules 1 and 3 both wall the *heavy* substituent of a donor (rule 1 wins for
sp via `setdefault` running first) — near-duplicate. Rule 2 walls *protons* but only for pnictogens. Rule 3
walls *heavy only*. The result: **no rule walls the protons of an N/O/C sp3 donor** — the gap the amine falls
through. The gate side is blind to it by construction (`donor_orientation` judges heavy substituents only).

## 2. The amine bug — reproduced and cause located

`CCNC1N[NH2]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[S]=1` (square_planar; donors N5 S24 O7 C17).
`rx.metal(...)[0]` → `rx.embed(iso, n, seed).minimize()`, real Ni restored (atom 6, output connected).

The amine N5 (`[NH2]`) has neighbours N4 (heavy), H32, H33. Its constraints:

```
cons.angles touching donor 5:  (6,5,4): (94.0, 158.4)   <- the ('N',SP3) fold wall on the ONE heavy nbr N4
                               (5,6,7), (5,6,24)          <- L-M-L polyhedron angles (donor placement, not orient)
```

Nothing touches H32/H33. **Cause, confirmed:** the fold wall skips protons (`:305`); rule 2 is scoped to P/As/Sb
so N never enters (`:295`, `_HIGH_BARRIER_DONORS = {15,33,51}`); and no rule places M on the sp3 lone-pair axis.
Walling only N4 at ≥94° leaves the (2H + lone pair) tripod free to rotate about the M–N5–N4 axis, so the metal
lands on the lone pair, on an H, or in between — unselected.

Measured severity (seeds 1–10, minimized, `scratchpad/amine_repro2.py`):

| | M–N–H med lo/hi | min M–N–H | inverted (LP>90°) | gate-flagged |
|---|---|---|---|---|
| raw ETKDG seed | 100 / 129 | 64 | 3/39 | 39/39 (other seed defects) |
| **minimized** | 100 / 129 | **65** | **4/33** | **0/33** |

The comment at `:305` ("N/O invert freely and g-xTB re-splays a folded seed itself") is the design rationale for
skipping N/O protons — but the embed→minimize path is **UFF, no g-xTB**, and the M–N bond is stripped so UFF has
no M–N–H bend to re-splay it. A user who embeds→minimizes→dumps (no calculator — the supported degrade path)
ships an inverted amine, gate-clean.

## 3. The unified design

The three `_orient_donor` holds are one statement: **wall every donor substituent (heavy OR proton) at the
census window for the donor's (element, hyb) class.** The per-class ideal angle *is* the census data (sp→~180,
sp2→~120, sp3→~109), which the codebase already tabulates. Collapsing the three:

```python
def _orient_donor(mol, metal, d, donor_set, cons, core_frozen=()):
    a = mol.GetAtomWithIdx(d)
    if metal in core_frozen or d in core_frozen:
        return
    cls = (a.GetSymbol(), _stripped_hybridisation(mol).get(d))
    window = _ORIENT_WALL.get(cls)          # census fold-wall floors + sp tightened (see below); or None
    if window is None:                       # uncalibrated (n<6) -> abstain, as the gate does
        return
    for nb in a.GetNeighbors():
        z = nb.GetAtomicNum()
        if z in _METAL_Z:
            continue
        if z > 1 and _is_apex(mol, nb.GetIdx(), donor_set):   # a >=2-donor bite apex is geometrically forced
            continue
        cons.angles.setdefault((metal, d, nb.GetIdx()), window)
```

`_ORIENT_WALL = { (elem,hyb): (floor, _FOLD_WINDOW[cls][1]) }` derived from `_FOLD_WALL_FLOOR`, **with the sp
classes overridden tight** — a linear donor's wall floor belongs near its median, not 25° below it:
`_ORIENT_WALL[('N',SP)] = (165,180)`, `('C',SP) = (165,180)`. That override is DATA (one comment, measured
provenance), not a branch.

This subsumes:
- **rule 3** (fold wall) — unchanged for heavy substituents; now also walls protons (fixes the amine — the amine
  gets its 3rd wall, forcing M onto the lone pair);
- **rule 2** (pnictogen splay) — a P/As/Sb proton now falls out of the `('P'/'As',SP3)` fold window; `_PROTON_SPLAY`
  and `_HIGH_BARRIER_DONORS` deleted;
- **rule 1** (sp end-on) — the sp donor's heavy substituent falls out of the (tightened) `_ORIENT_WALL[('*',SP)]`
  window; `_SP_DONATION` deleted.

It does **not** subsume `_coplanar_donor` (a dihedral, not an M–D–X angle) or the σ-aryl in-plane bisector
(a directional *centring*, needs a tight symmetric window + the `codonor_in_plane` gate).

## 4. Measured prototype coverage (seeds 1–8)

Variants: **min_amine** = rules 1+2+3 with rule 3's proton-skip removed (the ultra-minimal fix); **unified** =
the one-loop collapse above **without** the sp data override; **unified+saryl** = unified + the σ-aryl bisector
(tight symmetric 120±6 for two-heavy sp2, gated by `not codonor_in_plane`). "sp-data-override" = unified with
`_ORIENT_WALL` sp rows tightened.

| case | metric (ideal) | baseline | min_amine | unified | unified+saryl | +sp-data-override |
|---|---|---|---|---|---|---|
| **amine** N5 | min M–N–H (>90) | **65** | **94** ✓ | **94** ✓ | 94 ✓ | 94 ✓ |
| **amine** N5 | inverted LP>90° | **3/26** | **0/32** ✓ | 0/32 ✓ | 0/32 ✓ | 0/32 ✓ |
| **nitrile** (sp) | M–N–C (180) | 165 | 165 ✓ | **152 ✗** | 152 ✗ | **165** ✓ |
| **phosphine** (P–H) | min M–P–H (~105) | 104 | 104 ✓ | 101 ✓ | 101 ✓ | 101 ✓ |
| **acac** (1-heavy sp2 O) | M–O–C (~125) | 130 | 130 ✓ | 130 ✓ | 130 ✓ | 130 ✓ |
| **σ-aryl SM2** | \|skew\| med/max (0) | 18.6/32.4 | 18.6 | 18.6 | **0.0/11.6** ✓ | — |
| **henry** amidate | geo-clean | 32/32 | 32/32 | 32/32 | 32/32 | — |
| **karoline case2** | flag rate | 21% | 21% | 21% | 17% | — |
| **karoline case3** | flag rate | 44% | 44% | 44% | 44% | — |
| **karoline ester** | O–C–O collapse | 0/12 | 0/12 | 0/12 | 0/12 | — |

Reading it:
- **Amine fixed by every fix variant** — protons walled, no fold below 94°, no inversions. (Residual 100/128
  asymmetry is the flat-bottomed wall, not a spring — cosmetic; the lone pair still points at M, ~18°.)
- **Pnictogen (rule 2) collapses cleanly** — walling P–H at the fold window is if anything slightly *more*
  splayed (min 104→101). No regression.
- **sp (rule 1) does NOT collapse for free** — the fold-wall floor formula `min(lo+12, median−5)` gives
  `('N',SP)=152`, 13° below `_SP_DONATION`'s 165, so the nitrile bends to 152°. The **sp-data-override recovers
  165°** while keeping the amine + phosphine wins → sp survives the collapse only as a data row.
- **σ-aryl is untouched by the unification** (no protons) — only `unified+saryl` fixes the skew (18.6→0.0), and
  that is the σ-aryl finding's separate bisector, not a consequence of unifying.
- **Karoline no-regression holds** — case2/case3/ester are bit-identical across baseline/min_amine/unified
  (their donors are heavy-substituent sp2; nothing new is walled), and `unified+saryl` is *slightly better* on
  case2 (17%) because its bisector is gated off the plane-shared donors by `codonor_in_plane`, exactly as the
  σ-aryl finding established. No variant regresses any karoline case.

## 5. What genuinely cannot unify (the irreducible pieces)

- **`_coplanar_donor`** — the out-of-plane DOF (a dihedral). The σ-aryl finding measured that the in-plane pin
  does not subsume it (removing the improper regresses the tilt 42→52°). Different DOF; **stays**.
- **the σ-aryl in-plane bisector** — a directional *centring* of a two-heavy sp2 donor on 120/120. It is a
  window **tightening** sourced from `donation_axis` and **gated** by `codonor_in_plane` (or it fights the
  karoline crowded chelates). It does not fall out of the census-window loop; if landed it is +1 gated branch.
  The σ-aryl finding already recommends it as a separate thread.
- **`codonor_in_plane`** — the free-DOF skip; orthogonal, shared by the coplanar cap FF term. **Stays.**
- **`_FOLD_WINDOW` / `_FOLD_MEDIAN`** — physical census DATA. **Stays** (tabulated, per AGENTS.md).

## 6. Complexity verdict — the maintainer's question

**Branch count in `_orient_donor` (the "pile"):**

| | orientation branches | dedicated constants |
|---|---|---|
| **before** | 3 (`if hyb==SP` · `if in _HIGH_BARRIER` · fold-wall proton/apex skips) | 3 (`_SP_DONATION`, `_PROTON_SPLAY`, `_HIGH_BARRIER_DONORS`) |
| **after (Option A, unify)** | **1** (one loop; apex skip) | **0** (deleted) + `_ORIENT_WALL` data table w/ 2 sp rows |
| **after + σ-aryl** | 2 (loop + gated bisector) | 1 (`_BISECTOR`) |

**Verdict: the unification of `_orient_donor` is a genuine reduction, not a relabel.** It collapses the three
per-hybridisation holds — exactly the pile the maintainer named — into one substituent-walling loop over the
census data, deletes three constants, and fixes the amine for free with zero regression. The one judgement call
is the sp data override (a linear donor's wall floor near its median), which is data, not a branch.

**But the broader "one `donation_axis` mechanism for all of donor orientation" is REFUTED.** The out-of-plane
cap and the in-plane bisector are distinct DOFs that stay separate; the proton-inclusion unification does not
touch σ-aryl. Claiming one mechanism for the whole surface would overstate it. The clean win is bounded to
`_orient_donor`.

## 7. Recommendation

**Option A (recommended) — unify `_orient_donor`.** Replace rules 1/2/3 with the single loop in §3 over an
`_ORIENT_WALL` table (census `_FOLD_WALL_FLOOR` + sp tightened to (165,180)). Deletes `_SP_DONATION`,
`_PROTON_SPLAY`, `_HIGH_BARRIER_DONORS` and two branches; fixes the amine inside the collapse. Zero regression
measured on nitrile (165, via the data row), phosphine (splayed), acac, henry, and all three karoline cases.
This is the answer to the maintainer's worry: the fix *removes* machinery.

*Care point* carried from the σ-aryl finding: order the abstain/skip so an uncalibrated class still returns
early, and keep the apex skip — both preserved in the §3 sketch.

**Option B (ultra-minimal fallback) — if the sp/pnictogen holds must not be touched.** Delete only the
proton-skip at `donor_orient.py:305-306`. One-line change; fixes the amine; **bit-identical** on nitrile,
phosphine, acac, henry, karoline (rules 1 & 2 still own sp/pnictogen via `setdefault`, the new proton wall fires
only on the previously-unwalled N/O/C sp3 donor protons). Use this only if Option A's sp data override is
unwanted — but it leaves the three-branch pile intact, which is the thing the maintainer wants gone.

**Do NOT bundle σ-aryl.** Keep the bisector pin and the coplanar cap as the two separate, distinct-DOF pieces
the earlier finding already scoped.

## Provenance

Monkeypatch measurements on the installed env; `src/` untouched (`git diff src/` empty; baseline 362 passed /
2 skipped at `8f1214f`). Scripts in the shared scratchpad: `proto_orient.py` (the three variants + sp-data
override), `amine_atoms.py` / `amine_repro2.py` (§2), `measure_all.py` (§4 amine/nitrile/phosphine/acac/
saryl/henry), `measure_karoline.py` (§4 karoline case2/3 + ester), `sptight_test.py` (the sp data override).
