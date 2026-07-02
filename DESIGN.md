# DESIGN — the embedding constraint API (`fix` / `constrain` / `template`)

**Status: planned, not yet implemented.** This is the spec for a focused redesign of how a user *specifies*
what geometry to hold or bias during `embed`. The embed *mechanics* (graft, restrained-UFF, pose-freeze,
bounds-matrix editing) already exist and already work — **this change is confined to the specification /
resolver layer.** A fresh session should implement this, then update `README.md` + `CLAUDE.md` to the new
grammar.

---

## 1. Why

Today a core/TS is specified through **six** kwargs with an invisible two-tier model:

| today | tier | problem |
|---|---|---|
| `freeze=[atoms]` | rigid (graft to own coords) | only works when the source *is* the geometry |
| `distances=` / `angles=` | **soft** (bias seed, pose-freeze at achieved value) | approximate; users expect exact and don't get it |
| `template=` + `match=` + `anchor=` | rigid (graft to a reference) | 3 kwargs; hidden SMARTS auto-match can flip a symmetric core |

The `flp_ts_core` 15-liner and the SMARTS-triple `angles={("[F-]","[CH2]Cl","Cl"):178}` exist *only* because
there is no clean "here is the core geometry, hold it" entry. Users can't tell soft from rigid; the numbers
drift; symmetric cores silently mis-map.

## 2. The one idea

**The target is always a Mol** (from SMILES, `.xyz`, or an RDKit Mol). Only the **source of the core geometry**
varies. Every case feeds **one resolver** producing the existing `Constraints` struct:

```
target Mol ─┬─ core atoms (indices) ─┬─ geometry from: own coords | explicit coords | a reference | numbers
            │                        └─ strictness:    rigid (fix)  |  soft (constrain)
            └────────────────────────── one resolve_core() → Constraints (shape + hold) → embed / minimize
```

Three user-facing verbs replace the six kwargs:

- **`fix=`** — *rigid*. The named atoms **will** have this geometry.
  - coordinates exist ⇒ **Kabsch graft** to 0.000 Å (handedness-correct).
  - numbers only ⇒ **tight window in the bounds matrix + constrained-UFF pull** toward the exact values
    (`restrained_uff`, `distance_fc=1e4`); verify with `.measure()`.
- **`constrain=`** — *soft*. Bias the seed; a real energy may win. Wider windows, lighter force constant.
- **`template=`** — *sugar* for `fix` where the coordinates come from a reference structure. Mechanically
  identical to a coords-`fix`.

> **Naming note:** in molecular simulation "constrain" conventionally = *exact* and "restrain" = *soft*; here
> we deliberately use `fix` = rigid and `constrain` = soft (maintainer's call). Keep the code comments honest
> about this so nobody re-reads it as the MD convention.

## 3. Grammar (index-driven — the user resolves atoms; the API never SMARTS-matches internally)

```python
embed(source,
      # ── rigid ────────────────────────────────────────────────────────────────
      fix=[i, j, k],                     # hold atoms at the source's OWN coords     → graft (source has geometry)
      fix={i: (x, y, z), …},             # hold atoms at EXPLICIT coords              → graft
      fix={(i, j): d, (i, j, k): θ},     # NUMBERS: distance(s) + angle(s)            → tight-window + UFF pull
      template=(reference, {i: ref_i}),  # sugar: pull ref coords for mapped atoms    → coords-fix graft
      # ── soft ─────────────────────────────────────────────────────────────────
      constrain={(i, j): (lo, hi), (i, j, k): (lo, hi)},   # soft windows            → bias, energy decides
)
```

- **Keys are integer atom indices** (0-based, xyz-order). SMARTS matching is the *user's* job (two RDKit lines,
  shown once in the notebooks). This is deliberate: it is transparent and kills the symmetric-core
  automorphism flip.
- **Type-dispatch per entry:** `list` → own-coords; `int` key → a coordinate; `tuple` key → a distance/angle.
  A `fix` dict may mix `int`-keyed (coords) and `tuple`-keyed (numbers) entries.
- **`reference`** accepts an `.xyz` path, a Mol, an `Ensemble`, or an `N×3` array. The correspondence is an
  explicit `{target_idx: ref_idx}` dict — order-proof, no hidden matching.
- **The coordinate-dict form is the primary reference mechanism.** Getting a reference coordinate by index is
  trivial (`ref_pos = _xyz_to_mol("ts.xyz").GetConformer().GetPositions(); ref_pos[k]`), so
  `fix={target_i: ref_pos[ref_i]}` covers the reference case in one line; `template=` is kept only as sugar.
- **`fix` vs `constrain` is the strictness axis**; both accept scalars (a target) or `(lo, hi)` windows. Under
  `fix` a scalar becomes a tight window and is UFF-pulled/grafted; under `constrain` a window is a soft bias.

## 4. Closure — target × geometry-source (every cell must resolve or error cleanly)

| target ↓ \ core geometry from → | its **own** coords | **numbers** (d+θ) | a **reference** | **explicit coords** |
|---|---|---|---|---|
| **SMILES** (no coords) | ✗ error: *"no geometry — give numbers or a reference"* | ✓ tight-window+UFF | ✓ graft | ✓ graft |
| **xyz / Mol / Ensemble** | ✓ graft-in-place | ✓ impose numbers on the existing geom | ✓ graft reference | ✓ graft |

The only error is SMILES-wants-its-own-coords — a true user error with a clear message. An xyz user and a Mol
user hit identical syntax; only the input differs.

## 5. Invariants (make these true and assert them)

1. **0-based, xyz-order indexing, everywhere.** `_xyz_to_mol` preserves xyz line order. State it at every entry
   point. `fix`/`template` should **echo what they pinned** (`"fixed C0–F5 at 2.02 Å, ∠F5–C0–Cl6 = 178°"`) so a
   wrong (e.g. 1-based) index is obvious immediately.
2. **Explicit wins.** A `fix`/`constrain` distance on a pair overrides any vdW **encounter bound** for that
   pair. Encounter bounds apply *only* to inter-fragment pairs the user did not name. (This makes a sub-vdW
   forming bond in a multi-fragment SN2 survive the fragment-separation bounds.)
3. **≥3 atoms to orient.** 1 atom = a position pin; 2 = a distance (no angle/orientation); ≥3 = full pairwise
   shape (angles determined) and a graftable orientation. `add_pairwise_shape` already encodes this; warn `<3`.
4. **Chirality needs a graft.** A distance matrix is reflection-invariant, so soft `constrain` can embed the
   mirror image. A stereo-defined core ⇒ `fix` with coordinates (the Kabsch graft picks the correct hand),
   never `constrain`.
5. **`fix` numbers past one bond require the angle.** Two bonds among three named atoms leave the angle free
   (the 1–3 diagonal is not implied) — raise a clear error rather than silently under-constraining.
6. **`fix` delivers the values.** Numbers-`fix` runs the constrained-UFF pull (and snaps a lone 2-atom distance
   exactly, as `_graft_frozen` already does); the workflow ends with `.measure()` confirming. "fix" earns its
   name only if the realized geometry matches.
7. **Hydrogens: pass an explicit-H Mol.** `AddHs` is idempotent, so a user who does
   `mol = Chem.AddHs(Chem.MolFromSmiles(smi))`, indexes the H on *that* Mol, and passes the Mol keeps stable
   indices. No internal SMARTS-with-H resolution.

## 6. Reuse across operations — same spec, two verbs

The resolved `Constraints` already lives on the `Ensemble` (`self.cons`) and is already honored downstream by
`.tighten()`, `mc` (pose-freeze), and `.minimize()` (`restrained_uff`). So the same `fix`/`constrain` spec is
consumed by both entry points:

- **`embed(source, fix=…, constrain=…)`** — *search*: bias the bounds matrix, graft/hold, re-sample the
  periphery → an `Ensemble`.
- **`minimize(source, fix=…, constrain=…)`** — *relax an existing structure*: wrap the input geometry, apply
  the same spec, run the constrained UFF pull toward the targets → the manipulated structure(s). This is
  "manipulate an xyz **toward** a TS" without a full search. Same vocabulary, same resolver, same
  `restrained_uff` — no new mechanics.

Both are thin over one resolver. Advanced users can build a `Constraints` object once and pass it to either —
that object *is* the reusable, racerts-style config; the `fix`/`constrain` kwargs are just the ergonomic way to
produce it. Keep the kwargs as the default surface (transparent, KISS); the object is the power path.

`fix` vs `constrain` maps onto existing knobs: `fix` → tight windows + `distance_fc=1e4`; `constrain` → wider
windows + a lighter force constant. Nothing new in `restrained_uff`.

## 7. Blast radius

**Change (spec / resolver layer):**
- `constraints/builders.py` — one `resolve_core(mol, spec, *, rigid, has_geometry) → Constraints`, replacing
  `from_spec` + `from_template`. Owns: type-dispatch of the grammar, index validation, pairwise-shape from
  coords, tight-window construction from numbers, the reference-coords extraction.
- `pipeline.embed()` signature + `embed/dispatch.py::_embed_dispatch` — `freeze`/`distances`/`angles`/
  `planes`/`template`/`match`/`anchor` → **`fix` / `constrain` / `template`**, routed through the resolver.
  Drop the `template`-vs-rest mutual-exclusion special-case (it becomes one path).
- Add **`pipeline.minimize(source, *, fix, constrain, …)`** module-level entry (wrap input → resolve → the
  existing `Ensemble.minimize()`).

**Do NOT change (mechanics — already consume `self.cons`):** `bounds.py` (bounds-matrix override), the Kabsch
graft (`_graft_frozen`), `restrained_uff`, `_MetalCtx`, `mc` pose-freeze, `.tighten()`, the geometry gate.

**Out of scope for this redesign** (separate roadmap items): NCI directionality, metal binding-mode UX, the
GFN-FF pool tier, seed-count consistency. `metal=` / `coordinate=` / `contacts=` keep their current kwargs and
reuse `fix`/`constrain` for their held cores.

## 8. Migration from the current API (clean break — no back-compat)

| old | new |
|---|---|
| `freeze=[a,b,c]` (xyz source) | `fix=[a,b,c]` |
| `distances={(i,j):d}` + `angles={(i,j,k):θ}` (exact TS) | `fix={(i,j):d, (i,j,k):θ}` |
| `distances={(i,j):(lo,hi)}` (gentle target) | `constrain={(i,j):(lo,hi)}` |
| `template="ts.xyz", match=SMARTS, anchor={n:t}` | `fix={i: ref_pos[t], …}` or `template=("ts.xyz", {i:t})` |
| `planes=[(ringA,ringB,sep)]` | `constrain={…plane form…}` (fold π-stacks into `constrain`; minor) |

## 9. Worked examples (the acceptance test — none needs a special path)

```python
# W1 — DFT TS, search its conformers (graft in place)
embed("ts.xyz", fix=[14, 15]).mc().prune()

# W2 — put that TS core on an analogue (coords-fix; primary reference form)
ref = _xyz_to_mol("ts.xyz").GetConformer().GetPositions()
embed(analogue_smi, fix={c: ref[1], f: ref[5], cl: ref[6]}).mc().prune()

# W3 — SMILES-only TS from numbers (tight-window + UFF; verify with .measure())
f, c, cl = (mol.GetSubstructMatch(Chem.MolFromSmarts(s))[0] for s in ("[F-]", "[CH2][Cl]", "[Cl-,Cl]"))
embed("[F-].CCCCl", fix={(f, c): 2.02, (c, cl): 2.28, (f, c, cl): 178}).mc().prune()

# FLP — H-transfer core: index on an explicit-H Mol, pass the Mol
mol = Chem.AddHs(Chem.MolFromSmiles("CB(...)...CN1C=CC=C1"))
B, H, C, N = (…SMARTS on mol…)
embed(mol, fix={(B, H): 2.08, (N, H): 1.51, (C, H): 1.29, (B, C): 1.69, (B, N): 3.06}).mc().prune()

# W5 — manipulate an existing xyz toward a TS (relax, not search)
minimize("mol.xyz", fix={(i, j): 2.0, (i, j, k): 178})

# soft — a gentle approach, energy decides
embed(smi, constrain={(i, j): (2.6, 3.0)}).mc().prune()

# compose — hard TS core held + soft substrate approach guided, one call
embed("ts.xyz", fix=[14, 15], constrain={(donor, acceptor): (2.6, 3.2)}).mc().prune()
```

## 10. Decisions (locked)

- Verbs: **`fix`** (rigid) · **`constrain`** (soft) · **`template`** (reference sugar). Clean break, no
  back-compat.
- **Index-driven**; user does SMARTS. **Coordinate-dict** is the primary reference form; `template=` is sugar.
- Hydrogens via **explicit-H Mol**.
- Existing-structure relax = **`rx.minimize(source, fix=…, constrain=…)`** (same vocabulary as `embed`); the
  `Constraints` object is the reusable config underneath.

## 11. Open (decide during implementation)

- Numbers-`fix` window tightness (recommend a small ±0.02 Å window, not hard equality — ETKDG needs slack).
- Where `planes` (π-stacks) live: a form inside `constrain`, or a thin dedicated kwarg.
- Whether `embed` on an xyz should *seed from* the input geometry (preserve periphery) or always re-sample —
  default re-sample; `minimize` is the preserve path.
