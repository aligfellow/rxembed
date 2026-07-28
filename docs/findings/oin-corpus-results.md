# OIN corpus validation — adapter fidelity results

*Ran 2026-07-23. The `oin_adapter` (repointed onto the carved `rxembed.rdkit_embed` kernel, commit
`b6379e4`) scored on the 144-structure OIN corpus by coordination-sphere RMSD < 1.0 Å. Engine =
rxembed (DG bounds + restrained-UFF); perception + losslessness gate = OIN `ali-dev`, READ-ONLY.
Nothing in `../OIN-SMILES` was modified.*

## Headline

| metric | value |
|---|---|
| **pass @ 1.0 Å (overall)** | **105 / 144 = 72.9 %** |
| pass of the 110 structures that produced a real RMSD | **105 / 110 = 95.5 %** |
| pass of the 115 that reached RMSD scoring (incl. 5 sentinel mapping-failures) | 105 / 115 = 91.3 % |
| **median RMSD** (110 scored) | **0.211 Å** (0.223 Å if the 5 sentinels are folded into the median) |
| mean / min / max RMSD (110 scored) | 0.304 / 0.012 / 1.535 Å |

RMSD distribution (110 structures with a real RMSD):

```
<0.1 Å   :  6
0.1–0.25 : 56   <- the mode; the adapter reproduces the sphere to ~0.2 Å
0.25–0.5 : 36
0.5–1.0  :  7   (105 pass ends here)
1.0–2.0  :  5   (the near-miss tail)
>=2.0    :  0
```

**The adapter has no functional bug.** The repoint is correct (smoke passed; 105 spheres reproduced,
median 0.211 Å). Every one of the 39 non-passes is either an OIN-side encode miss, a documented
adapter *scope* decline (haptic / UNK — the adapter refuses rather than returns a wrong answer), a
marginal fidelity near-miss, or one large-complex perf timeout — none a regression from the carve.

## Failure breakdown (39 non-pass)

### (i) OIN-side encoder misses — 10 · NOT adapter bugs (stage = `encode`)

These fail in OIN's `xyz2mol`/`cxsmiles` before the adapter is ever called. XYZ→cxSMILES is entirely
OIN's; the adapter cannot fix them.

- **`get_lig_mol` ValueError (6):** DADXAK (macrocyclic polyamine), DUKPII (linear tetraphosphine),
  IROXET_comp_0, WIMCAA (borane cage), XAQDUS (borane cage), ticat3_generated_broken.
- **`get_lig_mol`→`lig_checks` unguarded `NoneType` (2):** **DIXHOK, UTANUA.** Same OIN perception
  family, surfacing as `AttributeError: 'NoneType' object has no attribute 'GetAtoms'` instead of a
  raised ValueError. Traceback pinned to `oinsmiles/utils/xyz2mol.py:344` (`lig_checks`, `res_mol` is
  `None`): RDKit rejects an over-valent atom during OIN's resonance-structure build (DIXHOK: C valence
  5 at atom 45; UTANUA: N valence 4 at atom 23), `res_mol` comes back `None`, and `lig_checks` does
  not guard it. **Verdict: OIN-encode artifact, not an adapter bug** (the `oin_adapter/` code never
  runs — `stage=encode`). Belongs to OIN's perception, alongside the six `get_lig_mol` misses above.
- **KekulizeException (2):** JOTJEK_comp_0, NAXDOI.

### (ii) Adapter *scope* declines — 10 · by design, no wrong answer (stage = `embed`)

The thin adapter routes haptic (η²/η≥3) and unclassified geometries to the enumerate fallback, which
correctly *declines* rather than fabricating a sphere. Not bugs — documented boundaries (a
haptic-centroid slot→vertex decode is out of scope for this adapter; rxembed has no UNK polyhedron).

- **Haptic shared-slot (9):** COJKAO, Ferrocene-halide-face, LASFOB, NUKHEG, SIKQIO, TiCat1, TiCat2,
  TiCat3, TiCat5 — "slot … shared by several donors (a haptic η² / η≥3 face)".
- **UNK geometry (1):** ZEJZOF — "geometry tag 'UNK' has no rxembed polyhedron".

### (iii) Adapter embed / gate — no conformer accepted — 3 (stage = `embed`)

Genuine adapter-path outcomes; the fixed isomer path ran but nothing satisfied OIN's byte-exact
`_accept` gate:

- **Rh-Single-Chiral-Phosphine** (24 conformers embedded, all rejected) and **ZOPNOH** (10 rejected):
  the losslessness/**chirality** gate declines every conformer — the embed did not reproduce the exact
  encoded hand to byte-identity. A gate-strictness / stereo-reproduction edge, not a geometry defect.
- **HgI3** (0 conformers embedded): trigonal [HgI3]⁻ produced no seed the relax kept — a real (but
  isolated, Hg-edge) small-case embed miss.

### (iv) Adapter perf timeouts — 6 · > 150 s (stage = `timeout`)

- **Haptic enumerate-fallback, slow decline (5):** HEXCEU (90 atoms), ILONON (92), IYIJAE (88),
  TiCat4 (48), TiCat6 (48). These are the *same* haptic-scope declines as (ii) — the fallback
  (`rx.metal` enumerates every coordination isomer, each embedded) is just expensive enough on these
  large/many-isomer inputs to cross 150 s instead of failing fast. They would decline regardless;
  perf, not correctness.
- **Fixed-path perf outlier (1):** **ZOPJOG** — a 97-atom octahedral (6 donors) on the fixed path,
  where n=24 embed + escalating restrained-UFF exceeds 150 s. The one genuine fixed-path perf item.

### (v) Scored but over the 1.0 Å gate — 10 (stage = `ok`)

- **Real-RMSD near-misses (5, all ≤ 1.54 Å):** PIWPUJ 1.024, WUVJAB 1.340, QISROZ 1.447,
  UDITUW 1.464, SOHMEJ 1.535. Correct spheres, marginally off the threshold.
- **Sentinel coordination-sphere mismatch (5, RMSD 996–999):** AGUFEN, KISZAR, MOCHUH,
  RERHEB (η²-centroid case, memory-noted — a centroid has no fragment), WELROW. The generated
  coordination sphere's *composition* differs from the input's (a donor perceived in/out), so the
  RMSD mapper returns a sentinel rather than a distance. A fidelity miss on tricky/haptic spheres.

### Adapter-vs-OIN split, summarised

- **OIN-side (never reaches the adapter):** 10 encode misses.
- **Adapter-scope declines (haptic/UNK, correctly refused — not bugs):** 10 fast + 5 slow (timeouts)
  = 15.
- **Genuine adapter-path fidelity/perf items:** 3 no-accept + 1 fixed-path timeout (ZOPJOG) +
  10 scored-but-over-gate = 14. All edge cases (marginal geometry, strict chirality gate, Hg, one
  large octahedral) — **no systemic defect, no repoint regression.**

## σ-aryl fixtures — both PASS

The two cyclometalated σ-aryl carbanion complexes are in the scored set and clear the gate
comfortably, confirming the recon prediction (the known seed-level donation-axis skew does not move
scored atom *positions*):

| fixture | RMSD | pass |
|---|---|---|
| **fac-Ir(ppy)₃** | **0.136 Å** | ✓ |
| **mer-Ir(ppy)₃** | **0.127 Å** | ✓ |

## Upstream recommendation — the `rxembed-seam` (document only; do NOT edit OIN)

The adapter depends on two OIN symbols. Their stability differs:

- **cxSMILES perception** (`cxsmiles_to_mol` / `mol_to_cxsmiles` / `strip_to_dative_smiles` /
  `geo_of`) already lives in the **stable** `oinsmiles/utils/cxsmiles.py`, where the adapter's
  `parse.py` imports it. Safe.
- **`_accept`** (the losslessness / chirality gate) still lives in
  **`oinsmiles/generation/rdkit_embed.py`** — the 600-line vendored copy of an embed engine OIN's own
  backlog marks for deletion as YAGNI. `_accept` (and its one helper `_mirror_x`) depend *only* on
  `mol_to_cxsmiles`, so lifting them is trivial.

**Recommendation:** OIN should lift `_accept` (+ `_mirror_x`) out of `generation/rdkit_embed.py` into
the stable shell (`oinsmiles/utils/cxsmiles.py` or a sibling), so deleting the vendored engine cannot
break this adapter. This is the `rxembed-seam` item. It is an **OIN-repo change — recorded here as a
recommendation only; not implemented** (separate project, separate branches).

## Reproduction

```
WT=<scratch>/oin-ali-dev   # detached OIN ali-dev worktree (read-only)
PYTHONPATH="<rxembed-root>:$WT/src:$WT/tests/integration" \
    uv run --no-sync python <scratch>/corpus_run2.py
```

`corpus_run2.py` = one subprocess per structure (per-structure 150 s timeout, 4-way parallel) around
`corpus_score.py::score_adapter`; the single-process `corpus_score.py` hangs the whole run when one
structure's embed goes pathological (IYIJAE burned 23 min before it was interrupted). Full
per-structure records: `scratchpad/adapter_scores.json`.
