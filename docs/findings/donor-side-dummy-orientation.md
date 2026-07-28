# A real BONDED donor-side dummy — does letting RDKit orient the donor natively DELETE the caps?

Maintainer's idea (distinct from the two prior findings): stop stripping the M-donor bond and re-imposing
orientation with explicit walls. Instead **append a REAL BONDED light atom in the donor's M-bond direction and
let RDKit's own ETKDG + UFF place the donor's substituents natively** — an sp3 N with R + 2H + a bonded dummy-D
just embeds tetrahedral (H's splayed off the D) for free; an sp2 donor embeds trigonal-planar with the D on the
bisector for free. "Take advantage of RDKit." If it holds, it **deletes** the `_orient_donor` walling loop + the
σ-aryl bisector + the `_coplanar_donor` cap instead of adding to them. This is the one variant the
`donation-axis-dcap.md` finding did **not** test: that D-cap was a *transient point + explicit angle
constraints*; THIS is a *real bonded atom* whose angles RDKit derives from the bond graph itself.

The natural vehicle is the existing `_hold_donor_chirality`, which already appends a charge-neutralised bonded
dummy-D (a deuterium) to *labile* donors and pins it near the metal (`_DUMMY_M_LO/_HI = 0.8/1.8 Å`). The test:
extend that dummy to **every** σ-donor, turn the caps **off**, and measure.

**Prototype only — no fix committed.** Every `src/` spike is a runtime monkeypatch (`bdsp_proto.py`), reverted;
`git diff src/` is empty. Numbers are `uv run --no-sync python`, seeds 1–8, `rx.metal(smi, geom)[0] →
rx.embed(iso, n, seed).minimize()`, real metal restored. Baseline 362 passed / 2 skipped at `ba48e72`.

## TL;DR — decisive NO, it hits BOTH walls

- **Wall 2 (does RDKit orient it right through the DG?) — HIT.** RDKit builds the correct *local* donor geometry
  for free (sp3 tetrahedral, sp2 trigonal, sp linear — confirmed at the raw seed), but a **distance-only pin
  does not orient the donor**: nothing ties the dummy to the metal *direction*, so the rigid donor group is free
  to **rotate about the M-donor axis**. Caps-off, minimized: the amine's worst fold is **unfixed** (min M-N-H
  **64°**, baseline 65°; the gross inversions *are* fixed, 3→0); the phosphine's protons **fold** (41/93 below
  90°, baseline 0); the sp nitrile **bends** (M-N-C **165→121°**). The orientation recovers **only** when an
  explicit `M–donor–dummy` colinear angle is re-added (phosphine folds 41→0, nitrile 121→165) — i.e. exactly the
  orientation wall the idea set out to delete. The bond gives RDKit the *intra-donor* angles; the *metal-relative*
  orientation — the thing the caps actually enforce — still needs an explicit angle.
- **Wall 1 (does the dummy perturb the sphere?) — HIT.** The maintainer correctly dodged the classic
  "tetrahedralise-the-sphere" failure (the dummy is bonded to the **donor**, so UFF never types the metal — it
  keeps its bond-less surrogate + polyhedron). But the donor-side dummies **crowd the metal by a new route**:
  3–6 extra atoms bonded to donors and pinned 0.8–1.8 Å from M push the donors off their polytope angles.
  Attributed cleanly (caps-off *alone* is bit-identical to baseline; only the dummy moves the number): an
  octahedral L-M-L RMS-from-ideal **1.2 → 6.3°**, henry geo-clean **32 → 25**, karoline flag rate **21 → 58%**
  and **44 → 67%**.
- **Out-of-plane still needs the coplanar cap** (as both prior findings found): caps-off, the σ-aryl OOP tilt
  **43.9 → 57.9°** and acac OOP **18 → 38°** get *worse*. A colinear dummy cannot fix the sign (reflection-blind;
  a 1-heavy-O donor has no OOP information at all).
- **Verdict: reject the bonded donor-side dummy; adopt Option A** (the `_orient_donor` walling-loop unification
  the prior two findings already recommend). It **deletes nothing** the walling loop doesn't already delete
  (you must re-add a colinear angle to orient, and keep the coplanar cap for OOP), while **adding** per-donor
  valence surgery + a dummy + a strip-before-score + a third transient-dummy plumbing system + an sp
  re-perception breakage — and it *perturbs the sphere*. This closes the "real bonded atom" door the D-cap and
  unification findings left open.

---

## 1. The prototype (`bdsp_proto.py`)

`install(caps_off=True)` does two things via runtime monkeypatch (both picked up because `coordination()`
imports the caps lazily and `dispatch`/`pipeline` call `_metal._hold_donor_chirality` by module attribute):

1. **Caps off** — `_orient_donor` and `_coplanar_donor` become no-ops, so the isomer's `cons` carries only
   M-donor distances + the L-M-L polyhedron angles. No orientation walls, no coplanar cap.
2. **Bonded dummy on every σ-donor** — `_hold_donor_chirality` is replaced with a version that appends a bonded
   deuterium to every donor for which `donation_axis(...)` is not `None` **and** which has ≥1 substituent
   (so haptic / bridging / hydride and bare halides are skipped), neutralises the donor's charge, sets
   `NoImplicit`, and pins `add_distance(metal, dummy, 0.8, 1.8)` — the chirality-cap pin, reused verbatim. The
   dummy lives in the DG embed **and** the UFF relax (both go through the same seam) and is stripped by the
   matching release (`git diff src/` clean). This is faithful to "let RDKit do the angles": the donor is now a
   full-valence centre, so the ETKDG bounds matrix and the full UFF force field both carry its D–donor–X angle
   terms with no explicit `cons.angle`.

**Where M sits vs the dummy** (`bdsp_diag.py`, raw seed): the dummy sits *between* M and the donor on the M
side (M-dummy 1.1–2.0 Å < M-donor 2.0–2.4 Å; M beyond it). For an *isolated* donor it lands near the axis
(M–donor–dummy 13–26°) and the substituents splay correctly at the seed (nitrile M-N-C 165, phosphine
94/116/119, octahedral N 112/125/113). For a *crowded* donor it drifts off-axis at the seed (amine
M–donor–dummy 61°, one H folded to 58°) — the first sign the distance pin is too loose. The dummy is a fake
atom and **must be removed before scoring / the connectivity finalize**, exactly like the metal surrogate and
the haptic centroid.

## 2. Wall 2 — RDKit gets the local geometry, not the orientation (seeds 1–8, minimized)

| case | metric (ideal) | **baseline** (caps on) | **bonded dummy, caps off** | **+ explicit colinear angle** |
|---|---|---|---|---|
| **amine** N (sp3) | min M–N–H (>90) | 65 | **64** ✗ (fold persists) | 85 (still < 94) |
| **amine** N | inverted LP>90° | 3/26 | **0/32** ✓ | 0/32 ✓ |
| **phosphine** PH₃ | #folded M–P–H<90 | 0/72 | **41/93** ✗ | 0/93 ✓ |
| **nitrile** (sp) | M–N–C (180) | 165 | **121** ✗ (bent) | 165 ✓ |
| **acac** O (1-heavy sp2) | M–O–C (~125) | 130 | 127 ✓ | — |
| **σ-aryl** SM2 | \|skew\| med (0, in-plane) | 18.6 | 14.0 ~ | — |

**Reading it:** the bonded dummy makes RDKit build the right *local* class (the seed diagnostics show
tetrahedral / trigonal / linear), but caps-off it does **not** orient the donor to the metal. The mechanism is
a free rotation of the rigid donor group about the M-donor axis: if the dummy pointed *exactly* at M, all three
sp3 substituents would sit at 109.5° from M for any rotation — but the distance pin (M-dummy 0.8–1.8, with
M-donor ≈2.1 and dummy-donor ≈1.0) does **not** force M–donor–dummy ≈ 0, so the tripod rotates and a proton
swings onto the metal. The **only** thing that fixes it is re-adding an explicit `M–donor–dummy` colinear angle
(0–15°): phosphine folds 41→0, nitrile 121→165, amine min 64→85. That colinear angle **is** an explicit
orientation wall — the dummy has bought nothing over walling the substituents directly, and even *with* it the
amine (min 85) is worse than the walling / D-cap approaches (min 94–102).

**The sp nitrile breaks structurally, not just rotationally.** Adding a bonded D gives the sp N a *second*
substituent, and `_stripped_hybridisation` re-reads a 2-substituent "sp" centre as **sp2** (its own bent-acyl
correction) — so the ruler and RDKit's own bent bounds pull the end-on 180° to 121°. Extending the dummy to sp
donors needs a special-case to undo this, the opposite of a simplification.

## 3. Wall 1 — the donor-side dummy crowds and perturbs the sphere (attributed, `bdsp_attribute.py`)

The metal keeps its bond-less surrogate (the dummy is on the donor), so UFF never types the metal and there is
no D–M–D′ collapse. But the dummies crowd the metal:

| octahedral `[NH3][Co]([NH3])([NH3])(Cl)(Cl)Cl` | L-M-L RMS-from-ideal | geo-clean |
|---|---|---|
| baseline (caps on, no dummy) | **1.2°** | 31/31 |
| caps-off ONLY (no dummy) | **1.2°** (identical) | 31/31 |
| caps-off + donor-side dummy | **6.3°** (5×) | 42/42 |

| henry square-planar | L-M-L RMS | geo-clean |
|---|---|---|
| baseline | 4.6° | 32/32 |
| caps-off ONLY | 4.8° | **32/32** |
| caps-off + donor-side dummy | 5.1° | **25/32** |

Caps-off *alone* is bit-identical to baseline on the octahedron and keeps henry fully clean — so the 1.2→6.3°
distortion and the henry 32→25 loss are the **dummy's** doing, not the missing caps. The 3 N-donor dummies
pinned 0.8–1.8 Å from Co (and their UFF vdW + excluded volume) push the donors off 90/180. **Karoline** (the
crowded conjugated chelates, `bdsp_measure.py karoline`) regresses hardest: case2 flag **21 → 58%**, case3
**44 → 67%** — the dummies pile into an already-tight sphere.

## 4. Out-of-plane — unchanged conclusion: it belongs to the coplanar cap

Caps-off, the σ-aryl out-of-plane tilt is **43.9 → 57.9°** (worse) and acac OOP **18 → 38°** (worse). A colinear
dummy does not fix the sign: the dummy on an sp2 donor is itself in the ring plane and reflection-ambiguous
about it (the same blindness `donation-axis-dcap.md §4` measured), and a 1-heavy-O donor's single dummy carries
zero out-of-plane information. The out-of-plane hold is `_coplanar_donor`'s job and stays — so even in the best
case the bonded dummy cannot delete the cap.

## 5. Delete vs add — the maintainer's question, concretely

**DELETES:** nothing the incumbent Option A doesn't already delete. To orient at all you must re-add a colinear
`M–donor–dummy` angle (§2), which is an explicit orientation wall; and you must keep `_coplanar_donor` (§4). So
the walling loop is **not** removed — it is relocated into a per-donor colinear angle plus an atom.

**ADDS:**
- a bonded dummy atom **per σ-donor**, with valence surgery (neutralise charge, `NoImplicit`, lenient
  sanitize) — for a nitrile / pyridine this also silently changes the donor's formal charge/valence during embed;
- a **strip before scoring / finalize** (the dummy is fake, like the surrogate);
- a **third transient-dummy plumbing system** on top of the haptic centroid and the chirality D-cap:
  `materialise_phantoms` currently **refuses** a second appended-first transient
  (`metal.py:591`), so composing a σ-donor dummy with a haptic face forces extending `_shift_phantoms` and the
  reservation check in both `bounds.embed` and `restrained_uff` (the prototype already had to call
  `_shift_phantoms` in hold and un-shift in release);
- an **sp re-perception special-case** (§2) to stop the nitrile bending;
- and it **perturbs the sphere** (§3) with nothing to fix it.

| | orientation mechanism | dummy atoms | sphere fidelity | out-of-plane owner |
|---|---|---|---|---|
| **Option A (recommended)** | 1 walling loop (centred window ⇒ bisector free) | 0 | intact | coplanar cap |
| **bonded donor-side dummy** | 1 colinear angle **+ a dummy** per donor | +1 per σ-donor | **degraded** | coplanar cap (unchanged) |

## 6. Recommendation

**Reject the bonded donor-side dummy. Adopt Option A** — the `_orient_donor` walling-loop unification from
`donor-orientation-unification.md` (one loop over the census `_FOLD_WALL_FLOOR` table with a *centred* window so
the σ-aryl bisector falls out for free), keeping `_coplanar_donor` as the out-of-plane hold. That is the real
simplification: it *removes* the three per-hybridisation branches and three constants, fixes the amine inside
the collapse, keeps the sphere intact, and adds no dummy, no strip, no third plumbing system.

The idea's appeal was real — "let RDKit do the angles" — but RDKit only does the *intra-donor* angles; the
*metal-relative* orientation that the caps exist for is precisely what a bonded distance-pinned dummy leaves
free (Wall 2), and the extra atoms crowd the sphere (Wall 1). This is the clean, decisive close of the "real
bonded atom" variant so it is not revisited.

## Provenance

Monkeypatch on the installed env; `src/` untouched (`git diff src/` empty; baseline 362 passed / 2 skipped at
`ba48e72`). Scripts in the shared scratchpad: `bdsp_proto.py` (the caps-off + bonded-dummy patch, with the
`COLINEAR_ANGLE` salvage toggle), `bdsp_diag.py` (§1 seed-level dummy geometry), `bdsp_measure.py` (§2 amine /
nitrile / phosphine / acac / saryl + §3 oct / henry / karoline), `bdsp_salvage.py` (§2 colinear-angle recovery +
the nitrile sp→sp2 re-perception probe), `bdsp_attribute.py` (§3 caps-off-only vs +dummy attribution).
