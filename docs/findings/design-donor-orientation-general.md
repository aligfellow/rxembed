# Donor orientation as ONE general principle — is σ-aryl "just the sp2 stuff", and can the caps be a
# single clean construction rather than a table of motif rows?

READ-ONLY design + chemistry analysis. No `src/`/tests edits, no commit. Reads the LANDED tree
(`constraints/donor_orient.py`, `metal.py::coordination`, `mechanisms.py::Coplanar`) and the four measured
findings it rests on (`sigma-aryl-orientation.md`, `donor-orientation-unification.md`, `donation-axis-dcap.md`,
`donor-side-dummy-orientation.md`) + the convergence memory.

## TL;DR

- **σ-aryl `M<-[c-]1ccccc1` IS the sp2 stuff.** An aryl carbanion ipso C is an sp2 in-plane σ-donor —
  perceptually and mechanically identical to an amidate N / pyridine N / carboxylate O. It is held by the SAME
  two orthogonal one-DOF holds every sp2 donor gets: `_coplanar_donor` (out-of-plane: M in the ring plane) and
  `_orient_donor` (in-plane/axial: M on the donation axis = the external bisector). It is **not** a new motif and
  wants no new code path. The "M in the aryl plane" half **==** the coplanar cap; the "M on the in-plane
  bisector" half **==** the general donation-axis in-plane hold. Both already exist and already fire on it.
- **The donor-orientation model is ALREADY one general construction, not a motif table.** The three old
  per-hybridisation rows (sp end-on, pnictogen splay, heavy fold) were collapsed at commit `c2ae975` into ONE
  `_orient_donor` walling loop keyed on `_stripped_hybridisation` + a census DATA table; `_coplanar_donor` is one
  loop keyed on the element-agnostic `inplane_sp2_donor`. The only remaining "rows" are the physical census
  windows (`_FOLD_WINDOW`, data per AGENTS.md) + two sp override rows (one judgement call). There is no motif
  branch left to delete — four attempts to fold the two DOFs into ONE mechanism are measured-refuted.
- **Why σ-aryl skews today is NOT a perception gap and NOT a missing motif** — it is that the general in-plane
  hold is a **WIDE FLAT one-sided window** `(97.0, 145.5)` on each M–C–ortho, deliberately flat (a floor, not a
  restoring pin) to avoid over-pinning. For a chelated sp2 donor the backbone/polyhedron co-pins the plane, so
  flat is enough. A **monodentate** σ-aryl has no co-donor in its plane, so nothing centres it and the flat
  window lets the seed skew toward one ortho (up to 21.5°). The ~40° out-of-plane tilt is the flat ±45° coplanar
  cap (no restoring force) + reflection-blind DG seed.
- **The make-or-break answer: a general formulation does NOT dodge the rigid-chelate regression — it BROADENS
  it.** The regression is a property of **tightening a flat floor into a tight centred symmetric pin**, not of
  being motif-specific. Generalising the *expression* (already done) is regression-free and green. Generalising
  so as to *also fix the skew* requires the tight centred window, and a general centred window applies that same
  over-determining pin to EVERY sp2 σ-donor — including the rigid diphosphine-amidate metallacycle that the
  specific gated pin already broke (24/24→22/24). It hits the regression on more donors, not fewer. Fixing the
  skew and staying general are in genuine tension, gated only by a global-backbone-rigidity predicate that does
  not exist as clean perception. **Verdict: keep the flat general hold; σ-aryl skew stays a documented
  known-limit (g-xTB corrects the final geometry).**

---

## 1. PHYSICS — what σ-aryl `M<-[c-]1ccccc1` wants

The ipso carbon is **sp2** (aromatic ring, trigonal). Its two ring bonds (to the two ortho carbons) and its
in-plane σ lone pair — the carbanion lone pair left when the C–H was removed — all lie in the aryl plane; the
ring π system is perpendicular. The metal binds that in-plane σ lone pair, so **crisply**:

> **M sits in the aryl plane, on the external bisector of the two ipso→ortho bonds, pointed where the C–H used
> to be.** Equivalently: `M–C(ipso)–ortho ≈ 120°` to BOTH orthos (a symmetric in-plane placement), and
> out-of-plane tilt = 0 (M coplanar with the ring). Ideal = 120°/120°, tilt 0.

This is exactly the geometry of any in-plane sp2 σ-donor. There is nothing carbanion-specific about it beyond
"element C" — the lone-pair axis is the sp2 external bisector, same as a pyridine N's or a carboxylate O's.

## 2. IS IT THE SAME AS sp2? — decompose into the two existing mechanisms

σ-aryl's target = "M in the plane" + "M on the in-plane bisector". Both are already-live, already-general
mechanisms; neither is σ-aryl-specific.

**(a) "M in the aryl sp2 plane" == the conjugated-sp2 coplanar cap — YES, the SAME mechanism.**
`inplane_sp2_donor(mol, d)` is `_stripped_hybridisation.get(d) == SP2`, element-agnostic (the old `{7,8}` N/O
list was deleted precisely so an aryl carbanion C and a thione S qualify — `conjugated-sp2-donation.md`). An
aromatic `[c-]` types sp2 (aromatic branch of `_pi_hybridisation`; RDKit agrees), so `inplane_sp2_donor=True`
and `_coplanar_donor` fires. Its two heavy neighbours (the two orthos) take the `len(heavy)==2` branch → the
improper `cons.coplanar = (M, ipso, o1, o2, 180, 45)`. This is byte-for-byte the same construction an amidate N
or pyridine N gets. **σ-aryl's out-of-plane hold is literally the sp2 coplanar cap, no special case.**

**(b) "M on the in-plane external bisector" == the general donation-axis in-plane hold — YES, the SAME
statement.** For a 2-heavy sp2 donor the donation axis (the in-plane lone-pair direction) IS the external
bisector of its two substituents. `_orient_donor` walls each M–D–X at the donor's `(element, hyb)` census
window; for `('C', SP2)` that is a window on each M–C–ortho angle. Walling both M–C–ortho toward ~120° = placing
M on the external bisector = "M on the donation axis". This is the exact same statement as the sp end-on hold
("M–D–X → 180, M on the linear axis") and the sp3 fold hold ("M–D–X → ~109, M on the lone-pair tripod axis"):
one rule — *hold each substituent at the donor class's census angle away from M, so M lands on the lone-pair
axis*. `donation_axis()` computes exactly this substituent set (with the co-donor/APEX/haptic/frozen
exclusions), and it is the axis the QA gate already measures — but note it is **gate-only today**; the landed
`_orient_donor` re-derives the substituent set by iterating neighbours directly, not by calling `donation_axis`.

**Confirm/correct against the code.** σ-aryl is conceptually — and mechanically — **coplanar + in-plane
donation-axis, both already-existing mechanisms**. The finding's own reproduction confirms it: "Both ipso donors
type sp2, `inplane_sp2_donor=True`, and receive the coplanar cap `(M, ipso, o1, o2, 180, 45)` + the loose fold
walls `(M, ipso, o_k) = (97, 145.5)`" (`sigma-aryl-orientation.md §1`). So the answer to the maintainer's "is it
just the same as the sp2 stuff?" is an unqualified **yes** — same donor class, same two DOFs, same two holds.
What differs is not the mechanism but the *tightness* of one of them (see §3).

## 3. WHY DOES IT SKEW TODAY? — trace it

**Not a perception fall-through.** An aromatic `[c-]` ipso carbon is perceived sp2 (both estimators agree via
the aromatic branch), enters `inplane_sp2_donor`, gets the coplanar cap, AND gets the `('C', SP2)` in-plane
wall. It falls through nothing. (The one perception holdout named in `_stripped_hybridisation` is a *non*-aromatic
metal-bound `[CH-]` — RDKit sp2 vs π-count sp3 disagree → unknown; the aromatic phenyl carbanion is not that
case.) So σ-aryl is fully constrained by the general machinery; the skew is a **tightness**, not a **coverage**,
problem.

**The in-plane skew — the flat window has no centring force.** `_ORIENT_WALL[('C', SP2)] = (97.0, 145.5)` (floor
= `min(85+12, 124.4−5) = 97`; ceiling = census 145.5). This is a **wide, asymmetric, flat-bottomed** window
written independently on each M–C–ortho via `setdefault`. Being flat it exerts **no restoring force** toward
120°, and being independent per arm it permits one arm at 100° and the other at 140° — a skew. For a *chelated*
sp2 donor the L–M–L polyhedron angle + the two M–D distances co-pin the metal in the plane (two contacts pin a
plane — `codonor_in_plane` / the `Coplanar` FF skip), so the flat window's lack of centring never shows. A
**monodentate** σ-aryl (SM1: one aryl donor + a vacant vertex; SM2: aryl + P + Br, no co-donor in the aryl
plane) has **no co-donor to centre it**, so nothing but the flat floor acts and the seed skews up to 21.5°
(`sigma-aryl-orientation.md §1`, `donor-orientation-unification.md §4`: σ-aryl |skew| 18.6/32.4).

**The out-of-plane tilt — the flat ±45° cap over a reflection-blind seed.** The coplanar cap is a flat ±45°
window with no restoring pull; the raw ETKDG seed tilt == the relaxed tilt to 0.1° (`sigma-aryl §2`), i.e. the
FF torsion never acts because the seed already sits inside ±45°. And the distance-geometry seed is **blind to
the sign of the out-of-plane** — three fixed distances from D, o1, o2 place M only up to reflection through the
ring plane — so the seed keeps ~37–42° of tilt that no in-plane constraint can remove.

**Why the SPECIFIC bisector pin regressed (the recorded decision).** The fix for the skew is a **tight symmetric
centred** pin — `M–C–ortho = 120 ± 6` on both arms (a restoring window, not a floor). Measured, it works
(21.5°→0.1°) and helps henry. But it **regressed a RIGID diphosphine-amidate chelate**
(`test_reembed_retry_delivers_clean_geometry` 24/24→22/24): pinning the amidate N tightly *pulls the metal*, and
a globally rigid metallacycle **cannot absorb** the pull, so the carboxylate twists out of conjugation. The gate
`codonor_in_plane` cannot separate henry's **flexible** sp3-hinged amidate (should pin) from this **rigid** one
(must not) — both are sp3-hinged locally; the discriminating signal is **global backbone rigidity**, which no
clean local predicate captures. The root cause, stated exactly: *fixing a skew needs a tight symmetric pin,
which is exactly the over-determination the fold-wall was deliberately kept a one-sided FLOOR to avoid ("a
tighter wall propagates strain").* A tight wall on a rigid metallacycle over-determines the metal.

## 4. THE DESIGN QUESTION — one general principle, and does general dodge the regression?

**The general principle, stated.** *M sits along the donor's donation axis (the in-plane/axial lone-pair
direction), and — for an sp2 donor — is additionally held in the donor's local π-plane.* Derived entirely from
perception:

- **the donation axis** is fixed by `_stripped_hybridisation(d)` → `(element, hyb)` → the census angle each
  substituent is walled away from M (sp→180 linear; sp2→~120 external bisector; sp3→~109 tripod). One loop,
  `_orient_donor`, over `donation_axis`'s substituent set. This is the whole in-plane/axial DOF for **every**
  donor — sp end-on, pnictogen splay, sp3 amine, and sp2 σ-aryl are all instances, not rows.
- **the local plane** is the `inplane_sp2_donor` (== sp2) branch: `_coplanar_donor` writes the out-of-plane
  dihedral cap. One loop, element-agnostic. This is the out-of-plane DOF, and only sp2 donors have it.

**This is already the landed architecture** (commit `c2ae975`). The "table of motif-specific rows" the
maintainer fears is gone: three per-hybridisation branches → one perception-keyed walling loop; the `{7,8}`
element list → the derived `inplane_sp2_donor`. What remains that looks like a table — `_FOLD_WINDOW` /
`_FOLD_MEDIAN` and the two sp override rows in `_ORIENT_WALL` — is **physical census DATA** (721 angles, 121
crystals) plus one judgement call (a linear donor's floor sits at its median), which is data, not branching, and
per AGENTS.md is meant to be a table. So the answer to "can it be ONE general principle?" is: **it already is —
two orthogonal one-DOF holds derived from one perception ruler, not a motif table.**

**Can the TWO holds collapse into ONE mechanism?** No — measured-refuted four ways, do not re-attempt:
- one `donation_axis` mechanism for both DOFs (`donor-orientation-unification.md`) — the in-plane axis and the
  out-of-plane dihedral are **distinct DOFs**; the in-plane pin does not subsume the cap (removing the improper
  regresses the tilt 42→52°).
- the transient "D-cap" dummy at the axis endpoint (`donation-axis-dcap.md`) — arithmetically *identical* to
  walling substituents (zero geometric content added), and reflection-blind about the ring plane so it does not
  fix the out-of-plane either; adds a 3rd dummy-plumbing system.
- a bonded donor-side dummy letting RDKit orient natively (`donor-side-dummy-orientation.md`) — a distance pin
  does not orient (donor rotates free about the M–D axis); recovers only on re-adding an explicit angle; and the
  dummies crowd the sphere (oct L–M–L 1.2→6.3°).
- deleting the caps — refuted; the orientation angle is metal-RELATIVE and RDKit cannot infer it (no metal
  coordinates until the polyhedron places it), so it can be unified, never deleted.

The irreducible end state is genuinely **two clearly-separated one-DOF holds**, both general, both
perception-derived. That is the clean formulation, and it is what ships.

**MAKE-OR-BREAK: would the GENERAL formulation avoid the rigid-chelate regression, or hit it too?**

**It hits it — and BROADER — if "general" means "also fix the skew"; it avoids it entirely if "general" means
only "clean the expression".** The distinction is decisive:

1. **Generalising the EXPRESSION** (the landed Option A: one flat perception-keyed walling loop + one coplanar
   loop) **does NOT hit the regression.** The flat one-sided window on the amidate N does not pull the metal, so
   the rigid metallacycle is undisturbed — `test_reembed_retry_delivers_clean_geometry` stays 24/24 and the tree
   is green. Generality of *form* is free of the regression. This is the maintainer's cleanliness goal, and it is
   already achieved.

2. **Generalising to also FIX THE SKEW** requires the flat window to become a **tight centred restoring pin**
   (the skew has no restoring force otherwise). A *general* centred window applies that tight pin to **every sp2
   σ-donor uniformly** — including the amidate N of the rigid diphosphine chelate. So a general centred
   formulation reproduces the regression **on more donors, not fewer**: it is the specific pin's regression
   generalised, with the same root cause (a tight pin over-determines a rigid metallacycle) now firing wherever
   an sp2 σ-donor sits in a rigid backbone. Generality does **not** buy an escape — because the regression is a
   property of **tightness**, orthogonal to whether the rule is motif-specific or general.

The escape hatch either formulation would need is identical and identically missing: a predicate that says
"this backbone is flexible enough to absorb a tight pin" vs "this metallacycle is globally rigid — leave the
floor loose". `codonor_in_plane` measures *local* plane-sharing, not *global* rigidity, and no clean local
perception captures global rigidity. Until such a predicate exists, **a general formulation must keep the flat
floor** (regression-free, but leaves the monodentate σ-aryl skew), or accept the regression (fixes the skew,
breaks the rigid chelate). The two goals are in real tension; generality does not resolve it.

**The out-of-plane residual is a separate, also-tension'd knob.** A restoring narrow coplanar cap (±5° vs the
flat ±45°) drives the σ-aryl tilt 41.9→0.0° (`donation-axis-dcap.md §4`) — the fix lives entirely in
`_coplanar_donor`, no dummy — but a blanket ±5° over-constrains real conjugated donors that legitimately reach
~40° out of plane (the census p95). So it too trades against the census tail and is a separate calibration
thread, not a general win.

## 5. Verdict

- **σ-aryl is the same as the sp2 stuff** — an sp2 in-plane σ-donor, held by the same coplanar cap +
  donation-axis hold as any amidate/pyridine/carboxylate donor. No new motif, no new code path.
- **The model is already one general principle** — two orthogonal one-DOF holds (donation axis + local plane)
  derived from one perception ruler (`_stripped_hybridisation`), not a table of motif rows. The remaining table
  is census DATA. The two DOFs do not collapse into one mechanism (four refutations).
- **The skew is a tightness gap, not a generality gap.** It shows only on a *monodentate* sp2 σ-donor where no
  co-donor centres the deliberately-flat in-plane floor.
- **A general formulation does NOT dodge the regression if it fixes the skew** — it generalises the
  over-pinning to every sp2 σ-donor, hitting the rigid metallacycle case more broadly. The regression is
  intrinsic to tightening a flat floor into a restoring pin on a rigid backbone; only a (missing) global-rigidity
  predicate would gate it. Keep the flat general hold; σ-aryl skew stays a documented known-limit.

## Provenance

Reads only. Landed state confirmed at `donor_orient.py` (`_orient_donor` = one `setdefault` loop over the flat
`_ORIENT_WALL`, no `donation_axis` call, no centring; `_ORIENT_WALL[('C',SP2)] = (97.0, 145.5)`) and
`mechanisms.py::Coplanar` (flat ±45° dihedral, `codonor_in_plane` FF skip). Measured history:
`sigma-aryl-orientation.md` (the skew reproduction + the rigid-chelate regression decision),
`donor-orientation-unification.md` (Option A, the landed collapse), `donation-axis-dcap.md` +
`donor-side-dummy-orientation.md` (the two refuted dummy variants), `conjugated-sp2-donation.md` (the
element-agnostic `inplane_sp2_donor`), and the `rxembed-donor-orientation-convergence` memory.
