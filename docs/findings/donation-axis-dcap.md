# The single "D-cap" — does a transient donation-axis dummy replace the three orientation pieces?

Maintainer's push-back on `donor-orientation-unification.md`: the three-piece answer (a substituent-walling
loop, an in-plane bisector, an out-of-plane coplanar cap) may be overcomplicating. Their idea — **one "D-cap":
materialise a transient dummy `Xd` at the donor's lone-pair / bisector position (`donation_axis`), and pin the
metal along `D→Xd`. One mechanism for EVERY donor.** The insight: a single "M lies along the donation-axis
vector" constraint pins BOTH the in-plane bisector AND the out-of-plane at once, so the σ-aryl's two pieces and
the sp3 amine and the sp end-on all collapse into one.

**Prototype only — no fix committed.** Every `src/` spike is a runtime monkeypatch (`dcap_proto.py`), reverted;
`git diff src/` is empty. Numbers are `uv run --no-sync python`, seeds 1–8, `rx.metal(smi, geom)[0] →
rx.embed(iso, n=1, seed=s).minimize()`, real metal restored. Baseline 362 passed / 2 skipped at `108952c`;
`tests/test_coplanar.py tests/test_donor_fold.py` = 59 passed with the tree clean.

## TL;DR

- **The D-cap dummy is arithmetically IDENTICAL to walling every substituent at the ideal M–D–X angle.**
  Placing `Xd = −Σ unit(D→Xᵢ)` (the lone-pair direction) and pinning M colinear along `D→Xd` gives, measured on
  a real amine, `M–D–Xᵢ = 111.3 / 114.3 / 112.8°` — i.e. every substituent lands at ~109.5°. The dummy adds
  **zero geometric information** over "wall each Xᵢ." It is pure indirection.
- **The single axis-pin fixes the amine and the σ-aryl IN-plane, but does NOT fix the σ-aryl OUT-of-plane** —
  refuting the specific hope. Measured: amine min M–N–H **64.9 → 102.5°, inversions 1/8 → 0/8** (fixed);
  σ-aryl |skew| max **21.5 → 15.9°, median 0.5°** (in-plane fixed); σ-aryl out-of-plane tilt **41.9 → ~40°,
  with OR without the cap** (NOT fixed). The out-of-plane is reflection-blind in the distance geometry: a dummy
  placed from the donor's own (coplanar) atoms is itself reflection-ambiguous about the ring plane, so pinning M
  to it inherits the ambiguity.
- **The out-of-plane fix lives in the coplanar cap, not the D-cap.** A stiff/narrow coplanar cap (a restoring
  pull, ±5° instead of the flat ±45° window) drives the σ-aryl tilt **41.9 → 0.0°**. The dummy is not involved.
  A 1-heavy-neighbour sp2 O (carboxylate/carbonyl) has ONE substituent, so the D-cap gives it ONE wall (in-plane
  only) and NO out-of-plane hold at all — the cap is the sole information and is irreducible.
- **Verdict: REFUTED. The D-cap is "one cap in principle, more moving parts in practice."** It deletes nothing
  that the walling loop doesn't already delete, keeps the coplanar cap, and ADDS a dummy atom + N placement
  constraints (the same N as the walls) + a third transient-dummy plumbing system alongside the haptic centroid
  and the chirality D-cap. Recommend the three-piece Option A. **Bonus:** a *centred* window on the walling loop
  merges pieces 1 and 2 (fold wall + σ-aryl bisector) into one loop with **no dummy** — the real simplification.

---

## 1. What the D-cap actually is — the identity (`dcap_identity.py`)

`donation_axis(mol, d, …)` returns a **list of substituent atom indices**, not a direction vector (verified:
`return [nb.GetIdx() for nb in a.GetNeighbors() if …]`). The direction is `−Σ unit(D→Xᵢ)` over those
substituents — which needs *coordinates*, and the distance-geometry embed (where the artefacts are seeded) has
none. So the maintainer's premise "`donation_axis` already computes the direction" is off by one step: it names
the substituents whose geometry defines the axis.

Taking a real relaxed amine and doing exactly what the D-cap proposes — place `Xd` at the lone-pair endpoint,
pin M colinear along `D→Xd` at the M–D distance — yields:

```
(a) IDENTITY  amine: M pinned along donation axis -> M-D-Xi = ['111.3', '114.3', '112.8']
    (all ~109.5  =>  the colinear pin == walling every substituent at the ideal angle)
```

**A single colinear pin to a lone-pair dummy is the same constraint set as `angle(M, D, Xᵢ) = ideal` for every
substituent.** This is not an approximation; it is what "M on the axis symmetric to the substituents" means. The
prototype below therefore emits those angle windows directly (the dummy's exact geometric content) — faithfully
measuring the D-cap without the dummy plumbing, and letting the plumbing be assessed separately (§4).

## 2. Prototype (`dcap_proto.py`)

Replace `_orient_donor` with one loop: for every donor substituent (heavy **and** proton), a **centred** window
on the donor's census median M–D–X (sp3 ~109.5, sp2 ~120/125, sp 180), ±8°. Gated by `codonor_in_plane` (skip
where a co-donor already lies in the donor's plane — the polyhedron pins the direction). `cap=True` keeps
`_coplanar_donor`; `cap=False` removes it (the maintainer's "one mechanism, cap deleted" ideal).

The centred window is the load-bearing choice: a *centred* wall on a 2-heavy sp2 donor's two substituents **is**
the σ-aryl bisector (120° ± 8 to each ortho = the external bisector), and a *centred* wall on an sp3 donor's
three substituents is the fold hold. One loop, one window per class, covers the fold wall AND the bisector.

## 3. Measured coverage (seeds 1–8)

| case | metric (ideal) | **baseline** | **D-cap + cap** | **D-cap, cap OFF** |
|---|---|---|---|---|
| **amine** N5 | min M–N–H (>90) | **64.9** | **102.5** ✓ | 101.6 ✓ |
| **amine** N5 | inverted (LP>90°) | **1/8** | **0/8** ✓ | 0/8 ✓ |
| **σ-aryl SM2** | \|skew\| med / max (0) | 7.6 / **21.5** | **0.5 / 15.9** ✓ | 0.9 / 14.6 ✓ |
| **σ-aryl SM2** | ‹M–C–o› (120) | 111.7 | **114.2** | 113.8 |
| **σ-aryl SM2** | out-of-plane tilt (0) | 41.9 / 44.2 | **40.7 / 41.0** ✗ | 40.0 / 41.5 ✗ |
| **nitrile** (sp) | M–N–C (180) | 165.0 | 172.0 | 172.0 |
| **phosphine** (P–H) | min M–P–H (splay) | 111.3 | 123.1 | 123.1 |
| **acac** (1-heavy sp2 O) | M–O–C | 114.2 | 120.0 | 117.7 |
| **henry** amidate | geo-clean | 15/16 | 15/16 | 15/16 |
| **karoline case2** | flag rate | 21% | **12%** | 8% |
| **karoline case3** | flag rate | 44% | **43%** | 26% |
| **karoline ester** | O–C–O collapse | 0/31 | 0/31 | 0/37 |

**Reading it:**

- **Amine — fixed.** Protons are now walled (the census fold wall skips protons; that is the gap the amine fell
  through). No fold below ~102°, no inversions. Exactly Option A's proton-inclusion win.
- **σ-aryl in-plane — fixed.** The centred window centres both M–C–o arms on 120°; the skew collapses (median
  0.5°). This IS the bisector, falling out of the same loop for free.
- **σ-aryl out-of-plane — NOT fixed, and the key negative result.** The tilt stays ~40° whether the coplanar
  cap is present (40.7°) or removed (40.0°). Two tight 120° walls on the two orthos still leave M free up to
  reflection through the ring plane (both walls are satisfied on either side); the flat window pins the *edge*
  at ~40°, never 0°. The single axis-pin does **not** pin the out-of-plane — the maintainer's central hope.
- **Controls — unregressed.** Nitrile more linear (165→172, census median 179), phosphine still splayed, acac
  in-plane, henry clean. Nitrile's window sits at the (180−8) floor; if the flat-window edge matters, sp keeps a
  tighter data row exactly as the prior unification found.
- **Karoline — improved, never regressed.** The `codonor_in_plane` gate skips the conjugated-bidentate donors:
  for case2/case3 the D-cap walls only **O17 (carboxylate) and N27/C27 (anilide/carbanion)** — the one-contact
  donors — and skips **N15/N34 (pyridylimine)**. The crowded chelate is untouched, so its flag rate drops (the
  tighter one-contact walls help), and the ester never collapses.

## 4. The out-of-plane belongs to the coplanar cap (`dcap_identity.py` part b)

The one regime where a dummy could beat flat walls: a **restoring** in-plane pull at relax time. Tested by
stiffening the *existing* coplanar cap to a narrow restoring window (±5° vs the flat ±45°) — the same physics a
stiff dummy point-pin would supply:

```
(b) SARYL tilt [baseline flat +-45]:              med=41.9  max=44.2
(b) SARYL tilt [stiff narrow +-5 (restoring)]:    med=0.0   max=0.4
```

**The out-of-plane tilt IS fixable — by making the coplanar cap a restoring pull, entirely inside piece 3.** The
dummy plays no part. This confirms the σ-aryl finding's proposed fix and locates it: the out-of-plane hold is
the coplanar cap's job, not the D-cap's. (A blanket ±5° would over-constrain real conjugated donors that reach
~40° out of plane — census tail — so it stays a separate calibration thread, not bundled here.)

Two independent reasons the D-cap cannot subsume the cap:
1. **A 1-heavy-neighbour sp2 O donor has ONE substituent.** The D-cap gives it ONE wall (the in-plane
   direction). A single M–O–C wall is one cone — zero out-of-plane constraint. The cap is the sole information;
   §`coplanar-cap-architecture` measured this cap load-bearing (HENRY O Δself +16.5°, case2 O17 +13.9°).
2. **A 2-heavy sp2 donor's substituents are coplanar with D**, so any dummy placed from them is reflection-blind
   about the ring plane — the same blindness the σ-aryl finding measured (three fixed distances from D,o1,o2
   determine M only up to reflection). Pinning M to that dummy cannot fix the sign.

## 5. The machinery count — the maintainer's real question

**Three-piece Option A** (the incumbent target):

| piece | mechanism | code |
|---|---|---|
| 1 | substituent-walling loop (fold wall; +proton-inclusion fixes the amine) | one loop over `cons.angles`, census data table |
| 2 | σ-aryl in-plane bisector | falls out of piece 1 as a **centred** window (measured §3) — **no separate code** |
| 3 | coplanar out-of-plane cap `_coplanar_donor` | `cons.coplanar`, DG `Coplanar.dg_post` + FF torsion; **owns the out-of-plane** (§4) |

So Option A is really **two** pieces: one walling loop (subsuming the bisector via a centred window) + the
coplanar cap. Both are plain, index-driven, read directly.

**D-cap** — to express the *same* geometry as piece 1, it must:

- **ADD** a transient dummy atom per σ-donor;
- **ADD** N placement constraints — `angle(Xd, D, Xᵢ)` for each substituent — the **same N** as the walls it
  replaces (the DG has no coordinates, so the axis endpoint must be pinned by the substituent angles);
- **ADD** 1 M-pin constraint (`M–D–Xd` / a colinear distance);
- **ADD** a **third transient-dummy plumbing system**, on top of the two that already exist and already collide:
  - the haptic centroid dummy (`_collapse_haptic` / `materialise_phantoms`) and the chirality D-cap
    (`_hold_donor_chirality`) already need `_shift_phantoms` to reconcile their index reservations;
  - `materialise_phantoms` currently **refuses** a second appended-first transient outright (`bounds.py:591`:
    *"another transient atom (a donor-chirality D-cap?) was appended first; … cannot compose"*). A third block
    forces extending `_shift_phantoms`, the reservation check, and the materialise/strip in **both**
    `bounds.embed` and `restrained_uff`;
  - `Xd` must be placed two ways — a geometric formula at relax, DG bounds at the fresh embed.
- **DELETES** nothing the walling loop doesn't already delete, and **keeps the coplanar cap** (§4).

| | orientation mechanisms | dummy atoms | transient-plumbing systems | out-of-plane owner |
|---|---|---|---|---|
| **Option A (recommended)** | 1 walling loop (centred window ⇒ bisector free) | 0 | 2 (haptic, chirality) | coplanar cap |
| **D-cap** | 1 loop **placing a dummy** + 1 M-pin | +1 per donor | **3** (+ reconcile) | coplanar cap (unchanged) |

**Newcomer readability.** Option A is two sentences a reader executes directly: *"wall every donor substituent
at its census angle; cap an sp2 donor's out-of-plane."* The D-cap is a paragraph — *"append a transient dummy at
the lone-pair endpoint computed from the substituent geometry, reserve its index against the haptic and
chirality dummy blocks, materialise it in the embed and again in the relax, pin the metal colinear to it, strip
it after"* — and the reader must then learn it is equivalent to the walls anyway. The indirection is the cost,
not a saving.

## 6. Recommendation

**Reject the D-cap dummy; adopt Option A.** The dummy is "one cap in principle, more moving parts in practice":
it relocates the walls' N angle terms into an atom + placement + pin + a third plumbing system, expresses
exactly the same geometry (proven identity, §1), fixes the same amine + in-plane σ-aryl the walls do, and leaves
the out-of-plane exactly where it was — on the coplanar cap (§4).

Concretely land:

1. **The unified substituent-walling loop with a *centred* window** (Option A of the prior finding, +the centred
   window). One loop over the census data, fixes the amine (proton inclusion) and the σ-aryl in-plane skew
   (centred window = bisector) in a single mechanism. Deletes the three per-hybridisation holds and — via the
   centred window — the separate bisector pin. No dummy.
2. **Keep `_coplanar_donor`** as the out-of-plane hold. It is irreducible (1-heavy-O donors; 2-heavy reflection
   blindness) and is where the real out-of-plane fix lives — a restoring pull drives the σ-aryl tilt to 0 (§4),
   a separate calibration thread (census-tail-safe) if the residual ~40° tilt must be driven out.

## Provenance

Monkeypatch measurements on the installed env; `src/` untouched (`git diff src/` empty; baseline 362 passed / 2
skipped at `108952c`; `test_coplanar.py`+`test_donor_fold.py` 59 passed, tree clean). Scripts in the shared
scratchpad: `dcap_proto.py` (the D-cap monkeypatch + `cap` toggle), `dcap_measure.py` (§3 amine / σ-aryl /
nitrile), `dcap_controls.py` (§3 phosphine / acac / henry / karoline + the `codonor_in_plane` gate readout),
`dcap_identity.py` (§1 the dummy≡walls identity + §4 the restoring-cap out-of-plane test).
