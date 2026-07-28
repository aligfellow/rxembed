# Preserve the seed's planarity through the relax — the T3d fix, and why it is NOT the T3c fix

**Companion to `planarity-mode.md` (T3d) and `seed-vs-relax-organic.md` §6/§8 (T3c).** Investigation only —
`src/` was spiked once to confirm the mechanism through the real pipeline and **reverted** (`git diff src/`
clean). Date 2026-07-21, branch `rdkit-embed-kernel` at `4dee01b`. Scripts in
`playground/preserve_planarity/` (`pp_seed_quality.py`, `pp_mechanism.py`, `pp_regression.py`,
`pp_safety.py`). Seeds fixed and reported everywhere (1, 2, 3, and 1/2/3/7/0xF00D for the corpus sweep).

## Verdict in one line

The reframing is **RIGHT for T3d and WRONG for T3c**, and they are **not the same carbon**. Preserving the
seed's planarity — a soft sp2-improper hold seeded from the conformer's own geometry — **cleanly clears the
T3d planarity mode** (`chb-tetramisole` 0/8 → 8/8 gate-clean, planarity violations 8 → 0, through the real
pipeline) as **~22 lines in `constraints/mechanisms.py`, no new `Constraints` field, no `builders.py` change,
no bounds/DG change (golden untouched)**. It does **not** clear the T3c C=S conjugation mode: measured, the
carbon's improper and the C–N conjugation torsion are **different degrees of freedom**, the seed is **not
reliably planar** on the conjugation axis (bimp-smiles-auto seed twist to **89.8°**), and holding the sp2
carbon can make conjugation *worse* (bimp 47° → 65°). T3c needs a separate **toward-flat torsion cap** — the
organic twin of `metal._coplanar_donor` — not a preserve.

---

## 1. Q1 — is the seed already good? (the make-or-break)

`pp_seed_quality.py` reuses the seed-vs-relax monkeypatch (snapshot each conformer immediately before
`Ensemble._relax_into_windows`), then for **every centre the RELAXED geometry flags** — planarity (sp2-carbon
off-plane) or conjugation (twisted quartet) — reads the **SEED's** deviation on the same driven mol. Organic
corpus, seeds 1/2/3.

```
uv run python playground/preserve_planarity/pp_seed_quality.py 1 2 3
```

**PLANARITY (T3d) — seed is reliably good.**

| flagged sp2-carbon centres | seed off-plane |
|---|---|
| all 197 instances | min 0.000, **median 0.008**, max 0.158 Å |
| already flat (≤ 0.05 Å) | **137/197 (70 %)** |
| per-case worst seed off-plane | chb **0.003**, bimp-smiles-auto 0.118, cpa 0.158, acid-dist 0.078 |

Every thiourea/isothiourea planarity centre the relax breaks has a **seed under the 0.15 gate** — chb C3 is
0.003, bimp-smiles-auto C9 tops out at 0.118 (< 0.15). The **one** seed at the gate line is `cpa` C73 (0.158)
— the frozen-core-adjacency **Mode B** `planarity-mode.md` already flagged as a separate, marginal mechanism,
not the thiourea defect. **So for T3d, preserve-only is SUFFICIENT.**

**CONJUGATION (T3c) — seed is NOT reliably good.**

| flagged conjugation quartets | seed twist |
|---|---|
| all 682 instances | min 0.0, median 4.6, **max 89.8°** |
| already planar (≤ 10°) | 435/682 (64 %) |
| seed ALSO twisted (> 10°) | **247/682 (36 %)** |
| per-case worst seed twist | bimp-smiles-auto **89.8°**, thia-ma **52.6°**, takemoto 3.0°, schreiner 0.6° |

The conjugation mode is *dominantly* a relax defect (median seed 4.6°), but **36 % of the flagged quartets
have a seed that is already twisted**, to 89.8° on bimp-smiles-auto and 52.6° on thia-ma. **Preserve-only
cannot fix T3c** — holding a seed that is already 89° twisted keeps it 89° twisted. T3c needs a real
restraint that drives *toward* the plane, not one that preserves the seed. **This is the single most
important finding: the "preserve" frame holds for T3d and breaks for T3c.**

## 2. Q2 — the mechanism, measured on both modes

`pp_mechanism.py` takes the planar ETKDG seed and relaxes it several ways, replicating `restrained_uff`'s
organic build faithfully (a `UFFGetMoleculeForceField` walked by `mechanisms.REGISTRY`), then reports the
whole molecule's **worst sp2-carbon off-plane** (planarity gate 0.15 Å) *and* **worst conjugation twist**
(gate 30°) — exactly what `geometry.check` decides on. The improper hold is
`UFFAddTorsionConstraint(n0,n1,n2,centre, ±5°, fc=10)` (fc = `mechanisms._COPLANAR_FC`).

```
uv run python playground/preserve_planarity/pp_mechanism.py
```

| case / seed | metric | seed | restrained_uff | **+holdC_seed** | +holdCNO_seed | +holdCNO_flat | mmff94s | gfnff |
|---|---|---|---|---|---|---|---|---|
| **chb** s1 (T3d) | planar (Å) | 0.000 | **0.202 P!** | **0.062** | 0.063 | 0.062 | 0.016 | 0.031 |
| | conj (°) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **bimp** s1 (T3c) | planar (Å) | 0.089 | 0.112 | 0.092 | 0.113 | 0.069 | 0.060 | 0.055 |
| | conj (°) | 0.0 | **47.4 C!** | **65.2 C!** | 43.5 C! | 0.0 | 0.0 | 0.0 |
| **takemoto** s1 (T3c) | conj (°) | 0.0 | 46.7 C! | **45.2 C!** | 44.2 C! | 30.0 C! | 0.0 | 0.0 |
| **schreiner** s2 (T3c) | conj (°) | 0.0 | 0.0 | **36.3 C!** | 38.3 C! | 39.1 C! | 0.0 | 0.0 |

Reading (consistent across seeds 1/2/3 for all four cases):

- **+holdC_seed (hold sp2 CARBON at seed) CLEARS T3d.** chb 0.202 → 0.062 on every seed; gate PASS. This is
  the fix.
- **+holdC_seed does NOT clear T3c** — and on `bimp` it makes conjugation **worse** (47.4 → 65.2°). Holding
  the carbon's own improper does not constrain the C–N torsion the conjugation twist rides.
- **Holding all sp2 (C/N/O) does not fix T3c either** (`+holdCNO_seed` bimp 43.5°, takemoto 44.2°): two
  coplanar impropers still leave the C–N bond free to *rotate* about itself — neither improper pins that
  relative rotation. `+holdCNO_flat` only *sometimes* removes it (bimp 0.0° but schreiner 39.1°, takemoto
  30.0°) — coincidental coupling, not a mechanism.
- **MMFF94s clears BOTH, every case** (conj 0.0°, planar ≤ 0.06 Å), matching GFN-FF/GFN2/DFT.

**Through the real `rx.embed` pipeline** (spiked `Sp2Planar` mechanism, then reverted — `pp_pipeline_spike`,
`geometry.check` gate on the shipped output):

| case | baseline gate-clean · planarity_viol | **+Sp2Planar** gate-clean · planarity_viol · conj_viol |
|---|---|---|
| chb-tetramisole s1/2/3 | **0/8 · 8** each | **8/8 · 0 · 0** each |
| cpa s1/2/3 | 0/4 · 4 each | 2/4·2·0, 2/4·2·0, **4/4·0·0** |
| bimp s1/2/3 | 1/4·0, 0/4·0, 2/4·0 | 1/4·0·3, 0/4·0·4, 2/4·0·3 (conj unchanged) |
| takemoto s1/2/3 | 0/4·0 each | 0/4·0·3, 1/4·0·3, 0/4·0·5 (conj unchanged) |

chb T3d: **0/8 → 8/8, planarity 8 → 0.** cpa Mode-B: planarity 4 → 2 (its C73 seed is already at 0.158, so
hold-at-seed keeps it marginal — the pre-identified separate mode). bimp/takemoto T3c: planarity was never
their axis; **conjugation is unchanged** — as Q2 predicts.

### Candidate (b) — an existing UFF/RDKit lever — is not available

Read, not merely inferred: UFF **already emits** the sp2 inversion term and it is exactly what is too weak
(`planarity-mode.md` §3: bare UFF alone puckers chb C3 to 0.174 Å). There is nothing to "keep". ETKDG's
basic-knowledge planarity is a **distance-geometry bound** in the bounds matrix (it is why the seed is flat),
with **no** representation in the UFF relax and no RDKit API to inject it as an FF term. Candidate (a) *is*
the way to carry that planarity into the relax — as a soft improper term seeded from the geometry it produced.

## 3. Recommended minimal fix (for T3d)

A new FF-only mechanism, `Sp2Planar`, in **`src/rxembed/constraints/mechanisms.py`**, added to `REGISTRY`.
Its `ff_terms` perceives the sp2 carbons **from the mol itself** (`conf.GetOwningMol()`), so it needs **no
new `Constraints` field and no `builders.py` change** (which matters: a concurrent refactor owns
`builders.py`). Scope = the planarity **gate's own** scope (3-neighbour sp2 carbon, excluding `cons.frozen`,
metals, and metal-coordinated carbons via the existing `geometry._coordinating_carbons`), so enforcement and
gate agree by construction. This is the exact spike that produced §2's pipeline numbers:

```python
_SP2_HOLD_FC = 10.0   # == _COPLANAR_FC; soft — remove the gross pucker, do not pin dead flat
_SP2_HOLD_WIN = 5.0   # deg half-window around the seed's own improper

class Sp2Planar(Mechanism):
    """Hold each perceived-planar sp2 carbon at the improper the ETKDG seed already has (preserve, not target-0)."""
    field = "sp2_planar"                       # reads no cons field; scope perceived from the mol + conformer
    def ff_terms(self, ff, cons, conf, fc):
        from rdkit.Chem import rdMolTransforms
        from rxembed import geometry as _geo   # lazy: as metal.py already imports geometry
        mol, pos = conf.GetOwningMol(), conf.GetPositions()
        skip = set(cons.frozen) | _geo._coordinating_carbons(mol, pos)
        skip |= {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _geo._METAL_Z}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6 or atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            if atom.GetIdx() in skip:
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) != 3:
                continue
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], atom.GetIdx())
            ff.UFFAddTorsionConstraint(nbrs[0], nbrs[1], nbrs[2], atom.GetIdx(), False,
                                       phi - _SP2_HOLD_WIN, phi + _SP2_HOLD_WIN, _SP2_HOLD_FC)
```

- **~22 lines** (2 constants + ~18-line class + 1 `REGISTRY` entry). Deletes nothing — an additive term, so
  say so plainly (AGENTS.md); it earns its lines by carrying the seed's planarity, which nothing else does.
- **No `builders.py` change, no new `Constraints` field, no DG/bounds change** (the mechanism has only
  `ff_terms`; `dg_windows`/`dg_relief`/`dg_post` are the base no-ops), so **the golden bit-identity harness —
  which gates the *bounds* pipeline — is untouched.**
- The improper target is read from the conformer **at build time**, which is the seed (the relax has not run
  yet); the escalation path restores the seed before each stiffer rebuild, so it stays a genuine "hold at
  seed" up the ladder.
- **Clears T3d, not T3c.** T3c (§5) is a separate torsion cap.

**Why (a) over (c) MMFF94s.** MMFF clears *both* modes (§2) and would be the one-shot fix, but its blast
radius is large and lands on hot code: `restrained_uff`/`ff_energies` hardcode UFF, the **metal surrogate is
Li (not MMFF-typeable)** so it would need a per-system UFF fallback, and switching the organic relax FF is
exactly the kind of change the golden harness and the metal path are there to catch. (a) is purely additive,
invisible to bounds, and touches only the term set — the smaller, safer change for the mode Q1 says is
preserve-fixable.

## 4. Q3 — does the hold wrongly freeze a should-pucker centre?

The danger the brief names: an sp2-perceived carbon whose true minimum is pyramidal (a bowl PAH, a strained
ring; a TS centre mid-inversion is `frozen`, hence already excluded). The decisive property is that
**hold-at-SEED is a ±5° window centred on the seed's own improper**, so it can only *prevent the relax adding*
pucker — never flatten a pucker the seed has. Contrast hold-at-FLAT (target 0), which would.

`pp_safety.py` isolates this: relax **re-started from a genuinely puckered geometry** (UFF's own 9° corannulene
bowl), not the flat ETKDG seed:

```
corannulene UFF bowl, worst sp2-C pyramidalisation = 9.0°  (a real pucker)
  bowl (start)        9.0°
  holdC_seed          9.0°   <- preserved: the window is around the start
  holdC_flat          2.3°   <- collapsed toward flat: would wrongly freeze a should-pucker centre
```

`pp_regression.py` corroborates on the flat-seed side (corannulene, acenaphthylene, cyclopropene-vinyl): when
the seed *is* flat, `holdC_seed` and `holdC_flat` **coincide** and both track GFN2 — no harm. **So hold-at-SEED
is the safe choice; hold-at-FLAT (target-0) is not.** It also answers Q1's T3c gap correctly: a
genuinely-twisted conjugation seed must be driven toward the plane (target-flat), which is unsafe as a blanket
improper rule but *correct* as a chemistry-scoped conjugation cap (§5).

**"Flat window has no restoring force, rides the wall"** (the recurring `angular-spring` failure): real but
**bounded**. The wall sits ±5° from the *seed target*, so riding it costs ≤ 5°: chb rides from a 0.000 Å seed
out to ~0.06 Å (≈ 2.4°) — well under the 0.15 gate. If ever wanted tighter, a soft harmonic pull to the seed
value (the `pulls` pattern) pins it nearer 0 at no risk. It is not the pathological case, because there the
wall was far from the target; here the target *is* the seed and the wall is 5° away.

## 5. Q4 — metal generality: orthogonal to `_coplanar_donor`, no subsumption

`metal._coplanar_donor` / `cons.coplanar` is a **torsion cap holding the METAL in a conjugated donor's plane**
(the M–D–X–Y dihedral, `geometry._planarity_dev`). The sp2-improper hold constrains a **CARBON's own wag**
(its 3-neighbour improper). Different atoms, different DOF — the hold on henry's thione carbon fixes the
ligand's own planarity, not where the Ni sits relative to the S plane. So the general sp2-planarity hold
**does not subsume and does not conflict with** `_coplanar_donor`; it is an orthogonal sibling (and it is
correctly *excluded* from metal-coordinated carbons by scope, so it never fights the coordination geometry).

The genuine kinship is the other way: the **organic T3c conjugation cap** (below) is the direct analogue of
`metal._coplanar_donor` — both are `cons.coplanar` toward-plane torsion caps re-detecting the well per
conformer. That is the shared mechanism worth unifying, not the improper hold.

## 6. T3c is a separate fix — a toward-flat conjugation cap (not this, not preserve)

Because (Q1) the conjugation seed is not reliably planar and (Q2) no improper hold touches the C–N torsion,
T3c wants the mechanism `seed-vs-relax-organic.md` §8 already proposed: a soft `UFFAddTorsionConstraint` on
the conjugated `X=C–N–S` dihedral toward its nearest in-plane well — element-agnostically perceived, exactly
`metal._coplanar_donor`/`mechanisms.Coplanar` applied to an organic conjugated bond. It must target the plane
(not preserve the seed), and — unlike the blanket improper — that is safe *because it is scoped to conjugated
bonds* (a lactam/biaryl is already excluded by `geometry.conjugation`'s own rules). MMFF94s (§3) remains the
alternative that dispatches T3c and T3d together at a larger blast radius. **Not built or measured here beyond
establishing that the improper hold is not it.**

## 7. What could not be established / caveats

- **The exact UFF term** (which inversion force constant) is not inspected; the defect and the fix are
  established by ablation and by the spike's pipeline gate result, not by reading UFF source.
- **`mc()` and downstream** are out of scope, as in the parent studies: `mc()` clears `_seeds_relaxed` and
  re-relaxes from openconf geometries, so the hold there becomes "hold at the openconf pose", untested.
- **Metal relaxes are touched** by the recommended universal scope (a metal complex's non-coordinated ligand
  sp2 carbons get the hold). Chemically correct (aryl carbons should stay planar) but it changes metal-path
  relax output, so the metal tests would need re-blessing; gating it `if not cons.metals: return` is the
  conservative alternative if that blast radius is unwanted, at the cost of leaving metal-ligand sp2 carbons
  unprotected.
- **cpa Mode-B is not fixed** by preserve (its C73 seed is already 0.158 Å ≥ gate); it needs the
  frozen-core-adjacency mitigation `planarity-mode.md` §5 describes, or is left as the marginal 2/4 it is.
- **T3c is diagnosed, not solved** here (§6).
