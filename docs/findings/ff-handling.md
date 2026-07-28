# FF handling — the UFF constraint API is the only lever (MMFF ruled out), and it already suffices

**Investigation only** — no `src/` was changed (spikes were monkeypatches; `git diff src/` clean). Date
2026-07-21, branch `rdkit-embed-kernel` at `4dee01b`. RDKit **2026.03.3**. Spikes and their console output
are preserved in `playground/ff_handling/` (`ffh_t3c_conjcap.py`, `ffh_q1_improper_shapes.py`). Companion to
`planarity-mode.md` (T3d), `seed-vs-relax-organic.md` §6/§8 (T3c), `preserve-planarity-fix.md` (the shipped
`Sp2Planar`).

## Verdict in one line

**MMFF is out** (it cannot type the metal path, and an organic-only fork violates the one-chain rule), so
UFF's four-verb constraint API is the only lever — and it is enough. `Sp2Planar`'s torsion-improper hold is
**the correct, best-available UFF shape for T3d** (validated, with one optional tightening). **T3c is "reuse
the coplanarity cap with an organic perception"**: a target-flat `UFFAddTorsionConstraint` on the conjugated
`C=X–N` quartet, cap 20°, `_COPLANAR_FC` — measured through the real pipeline it drives the thiourea twist
**to 0.0°** on bimp/takemoto/schreiner (gate-clean 32→46 of 72), reusing `mechanisms._coplanar_window`
verbatim. It is ~20 lines, additive, golden-invisible. No other UFF-API change is load-bearing.

---

## 1. Why MMFF is out (one measured line, then done)

MMFF has the native out-of-plane term UFF lacks (`SetMMFFOopTerm`, `GetMMFFOopBendParams`, all present in
2026.03.3), and it keeps every T3c/T3d centre flat natively — but **it cannot type the metal path**. The Li
FF-surrogate (`refine/ff.py::FF_SURROGATE`, bonds stripped) makes `MMFFHasAllMoleculeParams → False` and
`MMFFGetMoleculeProperties → None` (measured: a bond-less Li among organics returns `None`; a Pd-ammine
returns `False`). So any MMFF adoption would be **organic-only, forked by presence-of-metal** — exactly the
branch AGENTS.md forbids ("a capability is one more argument, not a branch"), on a metal-first package. Ruled
out by the maintainer; recorded here only as the reason.

> Aside, not a contradiction: `refine/ff.py::ff_energies` *already* does MMFF94s-where-typeable-else-UFF for
> the surrogate **energy** (`score('ff')`). That is a per-mol capability probe, not a fork in the relax; the
> constraint-enforcing relax (`restrained_uff`) is the one place that hardcodes UFF, and it stays UFF.

## 2. Current UFF handling — the map

`restrained_uff` (`refine/ff.py:113`) builds one `UFFGetMoleculeForceField` per conformer and walks
`mechanisms.REGISTRY`, each mechanism's `ff_terms` adding its restraint. The UFF-specific surface is small:

| Mechanism | FF term written | API used |
|---|---|---|
| `Frozen` | pin core, zero DOF | `ff.AddFixedPoint` |
| `Distance` / `Pull` / `Floor` / `Plane` | flat-bottom walls, soft pulls, floors, stack holds | `ff.AddDistanceConstraint` |
| `Angle` | angle wall | `ff.UFFAddAngleConstraint` |
| `Coplanar` (metal) | metal-in-donor-plane torsion cap | `ff.UFFAddTorsionConstraint` |
| `Sp2Planar` (organic) | sp2-carbon improper hold | `ff.UFFAddTorsionConstraint` |

Only `UFFAddAngleConstraint` / `UFFAddTorsionConstraint` are UFF-named; `AddFixedPoint` /
`AddDistanceConstraint` are FF-agnostic base-class methods. (Measured aside, since it de-risked the whole
MMFF question before it was ruled out: `UFFAddAngleConstraint` / `UFFAddTorsionConstraint` / `AddFixedPoint`
**work unchanged on an MMFF force-field object** — they are FF-agnostic constraint contribs, not UFF energy
terms. So the mechanisms layer is not the obstacle to any FF choice; the metal *typeability* is.)

**Golden blast radius: none.** `tests/golden/test_golden_bounds.py` captures only the DG **bounds** pipeline
— `cons`, the WINDOW `pairs`, the phase deltas, `tol`, and the final matrix. It never touches FF output. Any
FF-writer change that adds no new `Constraints` field (both fixes below are FF-only, perceived from the mol)
is **invisible to the golden harness**. The frozen-core graft is likewise untouched: it is `AddFixedPoint`
(zero DOF), unrelated to any planarity term.

## 3. Q1 — is `Sp2Planar`'s torsion improper the best UFF shape for T3d? YES

An sp2 carbon's out-of-plane displacement is a **4-body** quantity. UFF's constraint API is
{Distance, Angle, Torsion, Position}, and only two of the four can even encode it:

- **Position** pins atoms in place — but the substituents must move during the relax, so it cannot express
  "stay planar while relaxing". Not applicable.
- **Distance** — no single atom-pair distance is the centre-to-plane offset. Not applicable.
- **Angle** — three `Xᵢ–C–Xⱼ` angles at 120° force planarity (they sum to 360° only when flat), but that is
  **3 terms**, over-determined, fighting UFF's own angle terms.
- **Torsion** — one improper dihedral `n0–n1–n2–C` directly penalises the out-of-plane coordinate. This is
  the standard FF improper convention (CHARMM/AMBER impropers, and MMFF's own oop term, are all Wilson-angle
  impropers). **One term, the canonical shape.**

Measured on `chb-tetramisole` (`ffh_q1_improper_shapes.py`, seeds 1/2/3, worst sp2-C off-plane, gate 0.15 Å):

| seed | seed | none | **torsion_win** (shipped, ±5°) | torsion_pt (point) | angle_triple (3×120°) |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.202 **P!** | **0.062** | 0.000 | 0.048 |
| 2 | 0.001 | 0.198 **P!** | **0.061** | 0.001 | 0.050 |
| 3 | 0.003 | 0.197 **P!** | **0.060** | 0.002 | 0.050 |

Reading: bare UFF fails (~0.20 Å); the shipped torsion hold clears it (0.06 Å); the 3-term angle alternative
also clears it but is *worse* than the 1-term torsion and over-determined. **So the torsion improper is the
correct and best UFF lever — `Sp2Planar`'s shape is validated.**

**One optional tightening.** The shipped hold is a ±5° flat-bottomed window centred on the seed, which by
design has no restoring force inside the band and drifts to ~0.06 Å (~2.4°) — comfortably under the 0.15 Å
gate, so it is *fine*. A **point target** (`lo == hi == seed`, the `Pull` pattern) pins it dead at the seed
(**0.000 Å**) at the same `fc=10`, and is equally safe for a genuine bowl because it still targets the seed's
*own* improper (corannulene 9° → target 9°, preserved). `preserve-planarity-fix.md` §4 anticipated exactly
this. It is a marginal, zero-risk tightening — worth doing only if the ~0.06 Å drift ever matters; the shipped
window is already gate-clean. **Not a defect in the shipped fix.**

## 4. Q2 — the T3c fix, UFF-only: reuse the coplanarity cap with an organic perception

T3c (the conjugated `C=S`/`C=O` twist, `seed-vs-relax-organic.md` §6.1) rides the **C–N torsion**, a different
DOF from the carbon improper (`preserve-planarity-fix.md` §2: holding impropers does not pin the C–N
rotation, and the seed is not reliably planar — up to 89.8°). So T3c needs a **target-flat torsion cap**, and
that is *exactly* what `mechanisms.Coplanar` / `metal._coplanar_donor` already write for the metal: a soft
`UFFAddTorsionConstraint` toward the nearest in-plane well via `_coplanar_window(phi, cap)` at `_COPLANAR_FC`.
The only new part is the **perception**.

**Concrete recipe (measured to work):**

- **Which atoms.** The quartet `geometry.conjugation` already scores: for a single bond `X–C` (X ∈ {N,O},
  not in a ring) where C bears a double bond, the dihedral `(a, c, x, s)` — `a` = C's double-bond partner
  (=S/=O/=C), `c` = C, `x` = N/O, `s` = the substituent on X. Perceiving from the **same rule the gate uses**
  makes enforcement and gate agree by construction (the `Sp2Planar ↔ planarity` discipline).
- **Target window.** `mechanisms._coplanar_window(phi, cap)` — one-sided toward the nearest well (0° or
  ±180°), respecting both the syn and anti conjugation wells. Reused verbatim.
- **cap = 20°**, not 30°. Measured: cap 30° (== the gate) lets the flat-bottom **ride out to the gate line**
  (conjugation lands at 30.0–30.2°, still flagged) — the recurring `angular-spring` failure. cap 20° sits
  comfortably inside the 30° gate and lands the twist at 0.0°.
- **Force constant** `_COPLANAR_FC` (10 kcal/rad²) — the same soft constant metal coplanarity and `Sp2Planar`
  use.
- **Where it plugs in.** A new FF-only mechanism in `constraints/mechanisms.py`, sibling to `Sp2Planar`,
  appended to `REGISTRY`. It reads **no new `Constraints` field** (perceives from `conf.GetOwningMol()`), so
  **no `builders.py` change and no golden movement**. Organic-only (`if cons.metals: return`), and skip a
  quartet touching `cons.frozen` (already held with zero DOF) — same guards as `Sp2Planar`.
- **Rough size.** ~20 lines (a perception walk over conjugated quartets + one `UFFAddTorsionConstraint` per
  quartet reusing `_coplanar_window`), directly analogous to `Sp2Planar`. Additive — it supplies a term UFF
  lacks (the conjugated C–N torsion barrier); it deletes nothing.

**Measured through the real pipeline** (`ffh_t3c_conjcap.py`, `rx.embed` then `geometry.check`, seeds 1/2/3;
worst conjugation twist, gate 30°):

| case | BASELINE (UFF+Sp2Planar) conj | **+conjcap20** conj | baseline gate | +conjcap20 gate |
|---|---|---|---|---|
| bimp | 52.9–62.1° **C!** | **0.0°** | 0/12 | 0/12 † |
| takemoto-acetone | 40.4–45.2° **C!** | **0.0°** | 1/12 | **10/12** |
| schreiner-acetone | 36.3–40.4° **C!** | **0.0°** | 7/12 | **12/12** |
| chb-tetramisole (T3d) | 0.0° | 0.0° | 24/24 | 24/24 (unharmed) |
| cpa (T3d Mode-B) | 0.0° | 0.0° | 0/12 | 0/12 (planarity, unrelated) |
| **TOTAL** | | | **32/72** | **46/72** |

† bimp's conjugation is fully fixed (62°→0°); its residual fail becomes a *marginal planarity* (0.151 Å, just
over the 0.15 gate) once the C–N torsion flattens — a T3d/Mode-B tail, **not** T3c. cpa stays failed on its
frozen-core Mode-B planarity (`planarity-mode.md` §4), also not T3c.

So the cap **clears the T3c conjugation mode outright** on every named case, reusing existing machinery, and
does not touch T3d. This is the fix `seed-vs-relax-organic.md` §8 and `preserve-planarity-fix.md` §6 proposed;
it is now **measured**, not merely proposed.

## 5. Q3 — other UFF-API improvements

None load-bearing. `UFFAddPositionConstraint` (a soft position spring) is the one API verb unused, and
deliberately: the frozen core wants `AddFixedPoint` (zero DOF, a TS core must not relax), and no soft-position
use case exists. Every current `UFFAdd*` call is well-shaped (`Angle` caps `afc` at 1e3 deg⁻² to avoid
distorting a rigid framework; distance walls are flat-bottomed with the soft `Pull` inside). The only two real
improvements are the two above: keep `Sp2Planar` (optionally point-target it), add the conjugation cap. Two
orthogonal caps, each supplying one term UFF lacks — the improper (T3d) and the conjugated C–N torsion (T3c) —
both via `UFFAddTorsionConstraint`, which is the whole of what the API can offer here.

## 6. openconf's `preset="transition_metal"` — does it improve rxembed's metal FF handling? No, but it is a documented alternative

Studied at `/home/ali/Documents/Codes/openconf` (`master`, commit `1fa1441`; transition-metal support is
live, not upstream-only). What the preset and its metal handling actually do:

**The preset is a parameter bundle** (`config.py:502-529`) — it changes the **move set**, not the force
field: it adds `tm_ligand_rotate` (0.25) and `tm_haptic_rotate` (0.10) — rigid rotations of a whole ligand
fragment about the M–donor (or M–centroid) axis by 30–120° (`propose/moves.py:449-504`) — turns on low-mode
following, widens `energy_window_kcal` to 100, and forces sequential minimisation. The "auto metal-move
budget" (`api.py:28-59`) injects the same 35% TM-move mass into *any* config when a metal is detected, so you
get metal moves without picking the preset.

**How openconf holds the metal — genuinely different from rxembed.** It **keeps the M–ligand bonds** (no
strip, no Li surrogate, no dummy) and pins the shell with **hard** constraints (`constraints.py:104-147`):
metal position frozen (`MMFFAddPositionConstraint`, k=1e4), each first-shell M–L distance locked to ±0.05 Å
(k=1e5), then the constrained atoms **snapped back to reference** after every minimise
(`constraints.py:175-204`). The FF is **MMFF94s where typeable, else UFF for the whole molecule**
(`relax.py:93-102`) — the same MMFF-else-UFF pattern; for a real organometallic MMFF typing fails and the
metal is held purely by those position/distance constraints during a **UFF** minimise.

**Why it does not improve rxembed's `_ff_surrogate` / metal `ff_terms`.** openconf **preserves an input
coordination shell** — it locks to the *measured* M–L distances of a geometry it was handed. rxembed
**constructs** coordination that may have no input geometry (SMILES → isomer/haptic enumeration): its Li
surrogate supplies DG excluded volume and a zero-vdW FF sphere, and its `ff_terms` build the polytope
**angles**, donor **orientation** (sp end-on, pnictogen splay), the **coplanarity** cap, and haptic
centroids — none of which openconf's position+distance lock expresses, because a preserved shell already has
them. Swapping in openconf's hard lock would lose the construct semantics and re-litigate the long-settled,
measured surrogate physics (memory: `metal-ff-cancellation`, `angular-spring`, `foldwall-and-donor-orientation`
— "do not re-litigate Xe/Li"). It also fights rxembed's "bias the seed, let energy decide" (soft windows) with
k=1e5 hard locks. **Not adopted.**

**No harmful duplication, but a seam worth naming.** The two metal-holds run at *different stages*: openconf's
lock during `mc()` search (on the real metal in `self.mol` — the Li surrogate is transient, built only inside
rxembed's own `restrained_uff`), rxembed's surrogate during its own embed/minimize/score relax. They do not
collide. The one shared fact — both fall back to the same RDKit **UFF** for an untyped metal — is why this
whole finding matters: **UFF is the common FF for metals in both halves of the pipeline, and improving its
constraint handling (the two caps above, and rxembed's existing soft shape/orientation terms) is the only
lever either side has.** If rxembed's soft approach ever proves fragile, openconf's hard-pin-plus-snapback is
a proven alternative already inside the dependency — recorded, not recommended.

## 7. Recommendation

1. **T3d — keep `Sp2Planar` as shipped** (torsion improper, hold-at-seed, `fc=10`); it is the best UFF shape
   (§3). Optionally tighten the ±5° window to a point target (0.06 → 0.00 Å, zero risk) — cosmetic, not
   required.
2. **T3c — add the conjugation torsion cap** (§4): a ~20-line FF-only mechanism reusing `_coplanar_window` +
   `UFFAddTorsionConstraint` @ `_COPLANAR_FC`, cap 20°, perceived from `geometry.conjugation`'s quartet rule,
   organic-only. Measured to drive the thiourea twist to 0.0° (gate-clean 32→46/72), golden-invisible,
   additive. This is the open T3c defect closed.
3. **Metal — no change.** MMFF cannot type it; openconf's lock preserves rather than constructs. UFF + the
   existing soft surrogate/shape/orientation terms stays.

Net code: **two additive FF-only mechanisms** (`Sp2Planar` already shipped; the conjugation cap ~20 new
lines). No fork, no golden movement, no `builders.py`/DG change, no metal-path change. It does not delete
`Sp2Planar` — the two caps are orthogonal (improper vs C–N torsion), each closing one mode.

## 8. What could not be established / caveats

- The exact UFF force constant that under-restrains the improper / the C=S torsion was not read from RDKit
  source; the defects and fixes are established by ablation + through-pipeline gate measurement.
- The conjugation cap was measured on the five named T3c/T3d cases × seeds 1/2/3, not the full organic corpus;
  a full sweep and a regression test belong with the implementation (AGENTS.md: add the test that would have
  caught it).
- `mc()` and anything downstream are out of scope, as in the parent studies (`mc()` clears `_seeds_relaxed`
  and re-relaxes through openconf, whose metal handling §6 describes).
- bimp's residual marginal planarity (0.151 Å) after the conjugation cap is a T3d/Mode-B tail; whether the
  point-target `Sp2Planar` tightening (§3) also clears it was not measured.
