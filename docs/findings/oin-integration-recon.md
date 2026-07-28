# OIN ↔ rxembed integration recon

*Read-only recon, 2026-07-22 (post kernel-carve `cf338d5`). Grounds the overnight OIN-integration
and corpus-validation work. Author: recon subagent; placed here by the orchestrator.*

## Headline: the adapter already exists — Phase 2 is a re-wiring, not a build

`/home/ali/Documents/Codes/rxembed/oin_adapter/` (4 files: `__init__ parse reconstruct validate`) is a
working cxSMILES→embed adapter. It imports the **old** kernel path `from rxembed.constraints import metal`
(`reconstruct.py:15`, `validate.py:15`) which the carve moved to `rxembed.rdkit_embed.constraints.metal`.
Its `rxembed.geometry` / `rxembed.metrics` imports (`validate.py:13-14`) are still valid (those stayed shell).
**Repoint = one import line in each of `reconstruct.py` and `validate.py`.**

## 1. `PLAN_rxnts_integration.md` — effectively redundant for OIN

It is an **rxnts** (Pd(II) reaction-TS) consumer wishlist, not an OIN doc and not an rxembed roadmap;
measured GFN-FF-only on 6 Pd pairs. Its three asks:
- **Ask 1** (return M–L bonds + input formal charges on the Mol) — half superseded by the committed
  `rxembed-metal-charge-lost` fix (oxidation state now restored, `3087a4b`); atom-index preservation
  confirmed by the doc itself. Residual = the returned `Ensemble.mol` exposing DATIVE M–donor bonds to an
  external caller → **this is exactly the Phase-1 "M–L always-connected" task.**
- **Ask 2** (η¹ through an aromatic C–H should auto-promote to η² or refuse with a named reason) — the ONE
  genuinely-remaining item. A `<-[CH]1=...` σ-dative through an H-bearing arene C puts the metal ~1.86 Å
  from that H. **Niche, rxnts-specific, OFF the OIN critical path.** Carry as a small open item only.
- **Ask 3** — withdrawn, no code change.

**Verdict: redundant/orthogonal to OIN.** Do not build against it; only Ask 2 survives as niche backlog.

## 2. OIN string format + the adapter surface

Two OIN formats exist in `../OIN-SMILES`:
1. **Sidecar/inline `{n}` form** (`generation/oin_parser.py`) — drives the default **MetalloGen** generator.
2. **cxSMILES `|atomProp:` form** (`utils/cxsmiles.py`, **`ali-dev` branch only**) — a canonical dative-SMILES
   core + RDKit CXSMILES `atomProp`: metal `atomNote` = 3-letter geometry (`OCT`/`SPL`/…), each donor
   `atomNote` = vertex slot `s<n>` (optional CIP `s<n>R`/`s<n>S`), M–donor bonds = RDKit `DATIVE`.

**The adapter consumes format #2.** Flow (`parse.py`→`reconstruct.py`): `cxsmiles_to_mol` → `Perceived`
(mol, metal_idx, geometry, slot_of) → `_build_isomer` pins the exact isomer via `_metal.prepare` /
`coordination` / `Isomer` + an OIN-slot→`VERTEX_DIRS` map (`slot_to_vertex`, with square-pyramidal
`_SPY_REMAP`) — **no `rx.metal` enumeration, OIN already knows the isomer** → `rx.embed(iso,
stereo="free", n=n).minimize(distance_fc=1e5)` → keep the first conformer OIN's `_accept` re-encodes.
Public surface: `embed_cxsmiles(cx, n=24)`, `enumerate_cxsmiles(cx, n=8)`, `validate_geometry(mol)`.
It pins OIN's stiffer wall FC (`OIN_DISTANCE_FC=1e5`) — rxembed's softer 1e4 default drops otherwise-sane
geometries (XAMPUY 0/1→1/1). Haptic / multi-metal / TPY / UNK fall back to `enumerate`.

## 2b. Upstream OIN changes — the real blocker for the corpus run

The adapter's OIN deps live only on **`ali-dev`**, not the active `ag-dev`:
- `oinsmiles.utils.cxsmiles` (perception: `cxsmiles_to_mol`/`strip_to_dative_smiles`/`geo_of`)
- `oinsmiles.generation.rdkit_embed._accept` (the lossless re-encode gate) — and OIN plans to **delete**
  the vendored `generation/rdkit_embed.py` (2326-line copy of rxembed's engine) as YAGNI. OIN's own backlog
  names the fix: **`rxembed-seam`** — lift `_accept` + cxSMILES perception out of the doomed file into
  OIN's stable shell.

**Decision for the overnight:** repoint + smoke-test the adapter *here*; run the corpus validation against
OIN `ali-dev` (which carries the perception) READ-ONLY; **document the `rxembed-seam` upstream change as a
recommendation — do NOT push edits into the OIN repo autonomously** (separate project, separate branches).

## 3. Corpus validation recipe

- **Corpus:** 103 `../OIN-SMILES/tests/integration/tmQM/*.xyz` + 41 `tests/fixtures/*.xyz` = **144** plain XYZ.
- **Metric:** coordination-sphere **RMSD < 1.0 Å** (`tests/integration/rmsd_utils.py::calculate_tmc_rmsd`):
  metal + directly-bonded donors only, composition-matched (`CEILING_TOL=1.0`), permutation-Kabsch (≤5
  atoms/elem) or anchor+ICP (>5). A missing donor = mapping failure, not a small RMSD.
- **Harness:** `tests/integration/verify_roundtrip.py` (XYZ→OIN→XYZ→OIN). `tmqm_expected.py` is **empty** →
  all 103 tmQM are RMSD-only (no pinned string); string identity applies only to the ~30 named fixtures.
- **To score the adapter:** XYZ → `XYZToSMILES().convert` (OIN encoder) → cxSMILES → `embed_cxsmiles` →
  `calculate_tmc_rmsd(mol_input, mol_generated)`; pass-rate over 144 at 1.0 Å, vs MetalloGen `get_embedding`
  baseline. Per OIN `plan.md:9-24` the swap is fidelity-preserving **by construction** (rxembed's
  `ml_distance`/`_PHYS_COEF` is bit-identical to OIN's vendored copy) — the sweep confirms, not discovers.

## 4. σ-aryl `[c-]1ccccc1` corpus relevance — barely a needle-mover

- Corpus σ-aryl carbanion donors: only **fac-/mer-Ir(ppy)3** (2 of 41 fixtures, 6 aryl-C donors total). The
  TiCat `c{n}` carbons are **haptic** (whole ring → one slot), a different case (haptic-centroid, not the
  σ-aryl limit). tmQM σ-aryl count is unmeasurable read-only but cyclometalated aryls are rare in tmQM.
- The known-limit (`docs/findings/sigma-aryl-orientation.md`, decision: bisector pin NOT landing) is a
  **seed-level donation-axis skew** (~21° in-plane, ~37° out-of-plane of the M–C *direction*). The corpus
  metric scores atom **positions** and the ipso-C position / M–C distance (~2.11 Å) stay correct → the skew
  barely moves scored atoms; and any optimizer (xtb/MACE, harness default) relaxes it away. It would only
  bite an FF-only full-molecule-RMSD scoring, which the harness does not do. **Not material to the 144 score.**

## Actionable summary for the overnight

1. **Phase 1 · M–L always-connected** — makes `Ensemble.mol` always carry DATIVE M–donor bonds (also
   satisfies plan_rxnts Ask 1 residual).
2. **Phase 2 · repoint the adapter** — `from rxembed.rdkit_embed.constraints import metal` in
   `reconstruct.py` + `validate.py`; smoke-test `embed_cxsmiles` on cisplatin-class cxSMILES.
3. **Phase 2 · corpus run** — against OIN `ali-dev` read-only; record 144-structure pass-rate + RMSD
   distribution vs the MetalloGen baseline; document `rxembed-seam` as the upstream recommendation.
4. **Backlog (not overnight-critical):** plan_rxnts Ask 2 (η¹-through-aromatic-C–H auto-promote/refuse).
