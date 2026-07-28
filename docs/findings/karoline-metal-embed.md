# Karoline Ni square-planar embed — diagnosis

Four Ni(II) square-planar organocatalyst complexes the maintainer ran live. Repro scripts in
`playground/karoline_diag/` (`kdiag_*.py`). Invocation matches the notebook:
`rx.metal(smi, "square_planar")` → for each isomer `rx.embed(iso, n, seed).minimize()`.

**Method note (a null-measurement trap avoided).** Pre-`minimize`, the metal is still the **carbon
surrogate** (Z=6, M–donor bonds stripped), so a naive `geom.check` on the embedded mol reports four
phantom "heavy-atom steric overlap" clashes of every donor with atom-5 (the surrogate). Every measurement
here **restores the real Ni element+charge first** (`kdiag_harness.restored_mol`), or reads the
post-`minimize` mol (already restored). All seeds are fixed (`0xF00D+k`); several seeds per isomer.

**The committed `examples/k*.xyz` are NOT these cases** — they are all `C22 P1 Pd1 N1 O1 H40` (the morpholine
/ phosphine Pd complexes from the current notebook). The four Ni cases come from a newer notebook the
maintainer ran; they were reproduced here directly from the SMILES in the report.

---

## Summary table

| Case | Post-min flag rate | Dominant defect | Root cause | Session regression? | Severity |
|---|---|---|---|---|---|
| 1 thiourea | 0% flagged | thione Ni–S=C rides ±45° out of plane (median 17.6°, max 45.0°) | soft `_COPLANAR_CAP=45°` on the sp2 thione S donor | **pre-existing** | low/cosmetic |
| 2 pyridylimine·amidate(N) | **71%** | planarity + conjugation (backbone twisted) | metal **coplanarity cap** on the conjugated donors | **pre-existing** | high, but LOUD |
| 3 pyridylimine·amidate(C) | **100%** | planarity + conjugation (+1 clash) | metal **coplanarity cap** on the conjugated donors | **pre-existing** | high, but LOUD |
| 4 imidazole ester | 0% flagged (but…) | **ester O–C–O collapses to 56°** → O2–O4 fuse to 1.27 Å (the "epoxide") | pre-existing cap × **session floor-relief** interaction | **partly — session floor-relief amplified 1/40 → 12/40** | **HIGH — silent** |

**"Almost all embedded geometries flagged"** is TRUE for cases 2 & 3 (68–100%), FALSE for 1 & 4 (0%).
It is **pre-existing**, driven by the metal coplanarity cap on conjugated sp2 donors — **not** this
session's new mechanisms (seam / `Sp2Planar` / `ConjugationCap` are all innocent by ablation).

---

## The ablation (all monkeypatch, `src/` untouched)

Each new mechanism disabled in turn, measured on the exact defect. `coplanar_mech` / `coplanar_donor` /
`orient_donor` disable the **metal** coplanarity/orientation caps (pre-existing, but touched this session
by `ce2ce71`). `pre-ce2ce71` re-imposes the old "conjugated sp2 only" predicate to isolate that commit.

### Case 4 ester collapse — iso3 (`O4 N23 C16 O6`), n=1, 40 seeds

| Ablation | collapsed (O–C–O < 90°) |
|---|---|
| none (baseline) | **12/40** |
| seam off | 12/40 (unchanged — innocent) |
| caps off (`Sp2Planar`+`ConjugationCap`) | 12/40 (unchanged — innocent; they are `if cons.metals: return`) |
| **floor relief off** (`Floor.dg_relief`) | **1/40** |
| metal `Coplanar` mech off | **0/40** |
| `_coplanar_donor` off | 0/40 |
| `_orient_donor` off | 0/40 |
| pre-ce2ce71 predicate (conjugated only) | 12/40 (unchanged — ce2ce71 broadening innocent here) |

### Cases 2 / 3 flag rate

| Ablation | case2 | case3 |
|---|---|---|
| none (baseline) | 71% | 100% |
| seam off | 63% | — |
| caps off | 67% | — |
| floor relief off | 25% | — |
| **metal `Coplanar` mech off** | **0%** | **27%** |
| pre-ce2ce71 predicate | 68% | 100% |

---

## Case 1 — thiourea thione (`C[N]1(C)NC(N)=[S]->[Ni+2]...`)

- metal = C7(Ni), donors = S6, O8, N1, N18. Thione is C4(=S6)(–N3H)(–N5H2).
- Post-`minimize`: **0/95 conformers geom-flagged**. Geometry is chemically valid.
- Thione in-plane deviation `|dihedral(Ni7–S6–C4–N3)|`, folded to [0,90]:
  **median 17.6°, p95 37.6°, max 45.0°** — it rides the full `_COPLANAR_CAP = 45°` window edge.
- **Confirmed**: the thione S *does* get the coplanarity cap (`_coplanar_donor`, S is sp2 → `inplane_sp2_donor`).
  The prior fix's "median 6.7°" is a different metric (likely a plane-fit); the raw Ni–S–C–N dihedral shows
  a real ~18° median with a tail to the 45° cap edge.
- **This is what the user reacts to.** The `±45°` window (`CENSUS_OOP_P95 = 40°`, `_COPLANAR_CAP = 45°`,
  soft `_COPLANAR_FC = 10 kcal/rad²`) is deliberately wide, so conformers scatter across it — "in plane on
  some, not on others." No gate flags it (metal-plane planarity is report-only). **Pre-existing**, not a
  session regression.
- **Fix direction**: the window is too loose *for this donor class*. Either tighten `_COPLANAR_CAP` (or the
  FC) for the uncalibrated thione `('S', SP2)` class, or accept it as real census scatter and stop rendering
  it as a defect. Do not point-pin (kills the spread deliberately kept).

## Case 4 — the "epoxide" (`CCOC1=[O]->[Ni+2]...imidazole`) — HIGHEST PRIORITY

- metal = C5(Ni), donors = O4(ester C=O), O6(amidate), N23(imidazole), C16(carbanion). Ester is
  C1–**O2**–C3(=**O4**), with O4 coordinating Ni.
- **The "epoxide" is the ester carbon C3 being crushed**: the O2–C3–O4 angle collapses from ~122° to
  **55.6°**, fusing O2···O4 to **1.27 Å** (shorter than a peroxide O–O bond). A bond perceiver (xyzgraph)
  reads O2–O4 as bonded → a 3-membered **O2–C3–O4** ring. (The C1–O2–C3 ether never fuses; and the
  `N23–C24–C28` triangle the raw scan also shows is just xyzgraph over-perceiving the normal 2.1 Å imidazole
  2,5-distance — not a defect.)
- **Which stage.** The **embed seed is clean** (O–C–O ~122°). The collapse is introduced by **`minimize`**,
  specifically the full `_relax_constrained` inside `_reembed_until_clean`'s batch (the outer embed seam
  does *not* collapse it — `_rescue_torn` reverts a torn seed there). On iso3, **12/40 seeds (n=1, the
  notebook setting)** produce it. It is **iso3-specific** (the `O4 N23 C16 O6` arrangement).
- **The exact bond & atoms**: a *formed* O2–O4 contact (idx 2–4) at 1.27 Å across the ester carbon C3(idx 3).
- **Caught by any gate? NO — silent.** All three blind for the same structural reason (O2, O4 are a **1-3
  pair** across C3):
  - `geometry.check` → `clashes()` excludes every 1-3 pair (shared neighbour C3);
  - `metrics.bonding_ok` (minimize's accept gate) floors a non-bonded pair at `clash_tol=0.7`×Σr = **0.92 Å**,
    so 1.27 Å passes as "not fused";
  - `metrics.connectivity` / `.filter('connectivity')` requires `topo ≥ _MIN_TOPO = 3`; topo(O2,O4)=2, so a
    formed O2–O4 bond is **never reported**. Verified live: `.filter('connectivity')` keeps the collapsed
    conformer, `reacted = {}`.
- **Cause (ablation).** Two contributors, either of which resolves it:
  1. the metal coplanarity/orientation cap on the **conjugated ester O4 donor** (`_coplanar_donor` writes a
     proper dihedral M–O4–C3–X and `_orient_donor` the M–O–C fold wall for a one-neighbour O) — cap off → 0/40.
     This is **pre-existing** (O4's C=O is conjugated, so the pre-ce2ce71 predicate still capped it → 12/40).
  2. the **phantom-floor relief** (`cons.dg_floors` + `base._merge_relief` min-merge, applied by
     `Floor.dg_relief`), broadened this session (commits `5c8f9e5`, `4dee01b`) — relief off → 1/40.
- **Verdict**: a **pre-existing** cap-induced distortion of the coordinated ester, **amplified from ~rare
  (1/40) to frequent (12/40) by this session's floor-relief broadening** — i.e. a real session regression on
  frequency, on top of a latent pre-existing defect, that **no gate catches**.
- **Fix direction** (do not implement):
  - **Close the silent hole** — a 1-3 pair *can* fuse when its bridging angle collapses; the "1-3 is fixed by
    an angle" assumption behind `_MIN_TOPO=3` / the clash-exclusion is false under a hard relax. Add an
    over-compression check on 1-3 pairs (e.g. flag a 1-3 heavy pair below ~1.7×min or below a real-bond
    distance), or gate the O–C–O / X–D–Y donor angle explicitly. This is the load-bearing fix — a wrong
    molecule passing every gate is the worst outcome.
  - **Reduce the frequency** — investigate the floor relief for the coordinated-ester pairs (why min-merge /
    the broadened `nondonor_floors` lets the re-embed seed the collapse), and/or soften the one-neighbour-O
    `_orient_donor` fold wall + `_coplanar_donor` proper-dihedral on an ester whose carbon also carries a
    second O.

## Cases 2 & 3 — "awful geometry" (bulky N-aryl amidate / pyridylimine)

- Both: metal = C16(Ni), donors = N15(pyridyl), O17(amidate), N34(imine, case2) / C16-amidate-carbanion
  (case3), N27(anilide N⁻). Bulky 2,6-diisopropylphenyl imine + N-benzyl + N-phenyl.
- Post-`minimize`: **case2 71% flagged, case3 100% flagged**, dominated by **`planarity` + `conjugation`**
  (the conjugated ligand backbones — amide O=C–N, pyridyl, imine C=N — twisted out of plane; +1 clash in
  case3).
- **Cause (ablation).** The **metal coplanarity cap** (`Coplanar` mech / `_coplanar_donor`): forcing the metal
  into each *conjugated* sp2 donor's plane rotates the donor's substituents and twists the bulky conjugated
  backbone. Disable it → **case2 71%→0%, case3 100%→27%**. The phantom-floor relief is a secondary contributor
  (case2 71%→25%).
- **Regression? PRE-EXISTING.** The cap on *conjugated* donors predates this session; the donors here
  (pyridyl N, amidate O⁻/N⁻, imine N) are conjugated, so `ce2ce71`'s broadening ("cap all sp2, not just
  conjugated") does **not** newly cap them — pre-ce2ce71 predicate leaves case2 at 68%, case3 at 100%
  (unchanged). The session's own new mechanisms are innocent: **seam off 63%, caps off 67%** (both unchanged).
- **Severity**: high (visibly distorted, geom-flagged) but **LOUD** — the maintainer sees the flags. Not a
  silent-defect risk like case 4.
- **Fix direction**: the coplanarity cap is too aggressive on a **crowded, multi-donor conjugated chelate** —
  four conjugated donors each pulling the metal into its own plane over-determines the square plane and the
  ligand pays by twisting. Options: down-weight `_COPLANAR_FC` when ≥3 conjugated donors share one metal; or
  restrict the cap's *ligand-side* reference so it holds the metal plane without dragging the backbone (it
  already uses direct substituents — the residual twist is the DG 1,4 bound + FF torsion competing with the
  four-donor plane). Needs a targeted look, not a blanket removal (the cap earns its keep on κ1 carboxylates).

---

## Refuting "this session caused it"

- **Over-flagging (cases 2/3)**: this session is **largely innocent**. Seam, `Sp2Planar`, `ConjugationCap`
  ablate to no change; ce2ce71's coplanar broadening ablates to no change. The driver is the **pre-existing**
  coplanarity cap on conjugated donors. Say this loudly.
- **Epoxide (case 4)**: this session **is partly culpable** — the **floor-relief broadening** amplified a rare
  latent collapse (1/40) to frequent (12/40). But the underlying distortion (the coplanar/orient cap on the
  coordinated ester) is pre-existing, and the **silent-gate hole is entirely pre-existing** and is the real
  problem.
- **Case 1**: pre-existing soft-window width; not a regression.

## Prioritized causes

1. **[HIGH, silent] Case 4 — the 1-3 fusion gate hole.** No gate (`geom.check` clash, `bonding_ok`,
   `connectivity`/`filter`) catches an O–C–O collapse fusing two 1-3 atoms. A wrong molecule ships. Fix the
   gate first.
2. **[HIGH, frequency regression] Case 4 — floor relief amplifies the collapse** (1/40 → 12/40). Session
   change (`5c8f9e5`/`4dee01b`). Investigate the relieved floors on the coordinated ester.
3. **[HIGH, loud] Cases 2/3 — coplanarity cap over-constrains a crowded conjugated chelate** (71–100% flagged).
   Pre-existing; needs a multi-donor-aware softening.
4. **[LOW] Case 1 — thione ±45° coplanarity window is loose**, producing visible in/out-of-plane scatter.
   Pre-existing; tighten the class window or accept as census scatter.
