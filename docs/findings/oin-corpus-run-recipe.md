# OIN corpus validation — runnable recipe

*Read-only environment-prep, 2026-07-22. Stages the 144-structure corpus sweep of the `oin_adapter`
(coordination-sphere RMSD < 1.0 Å). The sweep is **not run here** — it waits on the `oin_adapter` repoint
(Phase 2, another agent). This doc + `scratchpad/corpus_score.py` are the ready-to-fire recipe. Nothing in
`../OIN-SMILES` or in `src/` was modified; only a **detached** OIN worktree was added under the scratchpad.*

## TL;DR

- The **OIN half imports and runs standalone in the rxembed venv** — proven (`scratchpad/dry_smoke.py`):
  `get_tmc_mol → mol_to_cxsmiles` encodes CisPlatin/FeCO5/Ferrocene and `calculate_tmc_rmsd` returns
  0.0000 Å self-RMSD. No pulp / MACE / torch needed for the encoder + metric — pure **rdkit+numpy+scipy+networkx**, all already in the venv.
- **Least-invasive env-setup = a detached OIN worktree + PYTHONPATH** (one command, disturbs nothing):
  ```
  git -C /home/ali/Documents/Codes/OIN-SMILES worktree add --detach \
      "$SCRATCH/oin-ali-dev" origin/ali-dev
  export WT="$SCRATCH/oin-ali-dev"
  # then prefix every run with:  PYTHONPATH=$WT/src:$WT/tests/integration
  ```
- **Blocker for the sweep (not for env-setup):** the adapter's `oin_adapter/reconstruct.py:109` calls
  `_accept(cand, cx)` (2 args) but ali-dev's `_accept(cand, text, total_charge)` requires **3** (no default).
  The repoint agent must also pass `total_charge` (e.g. `Chem.GetFormalCharge(cand)`), or `embed_cxsmiles`
  `TypeError`s on the fixed-isomer path.

## 1. Confirmed OIN pieces on `origin/ali-dev` (import paths + signatures)

All present on `origin/ali-dev` (worktree HEAD `7cba136c`); **none exist on the active `ag-dev`** (its
`rdkit_embed.py` has no `_accept`, and `ali-dev` has only 34 fixtures vs ag-dev's 41 — see §3).

| Symbol | Module (ali-dev) | Signature / returns |
|---|---|---|
| `get_tmc_mol` | `oinsmiles/utils/xyz2mol.py:509` | `get_tmc_mol(xyz_file, overall_charge, with_stereo=False) -> (tmc_mol, xyz_coords)` |
| `mol_to_cxsmiles` | `oinsmiles/utils/cxsmiles.py:405` | `mol_to_cxsmiles(mol: Chem.Mol) -> str` (dative core + `\|atomProp:...\|`) |
| `cxsmiles_to_mol` | `oinsmiles/utils/cxsmiles.py:421` | `cxsmiles_to_mol(text: str) -> Chem.Mol` (notes+DATIVE bonds, no 3D) |
| `strip_to_dative_smiles` | `oinsmiles/utils/cxsmiles.py:437` | `strip_to_dative_smiles(text: str) -> str` (core before `\|`) |
| `geo_of` | `oinsmiles/utils/cxsmiles.py:199` | `geo_of(note: str) -> str` (3-letter geo tag → name) |
| `XYZToSMILES` | `oinsmiles/core/translator.py:8` | `.convert(xyz_file_path) -> str` — **OIN inline `{n}` string** (the MetalloGen/baseline encoder, NOT cxSMILES) |
| `_accept` | `oinsmiles/generation/rdkit_embed.py:180` | `_accept(cand: Chem.Mol, text: str, total_charge: int) -> bool` — **3 args, no default** |
| `calculate_tmc_rmsd` | `tests/integration/rmsd_utils.py:28` | `calculate_tmc_rmsd(mol1, mol2, mol2_bonded=None) -> float` (999.0/998.0 on mapping failure) |
| `OIN3DGeneratorMetallogen` | `oinsmiles/generation/metallogen_adapter.py` | baseline; `.generate(oin_string) -> gen_result` with `.mol` (bonded) + `.xyz_string` |

### The XYZ→cxSMILES path is `mol_to_cxsmiles(get_tmc_mol(xyz))`, not `XYZToSMILES().convert`

The recon's "`XYZToSMILES().convert` (OIN encoder → cxSMILES)" is imprecise: `.convert` emits OIN's **inline
`{n}`** string (what the MetalloGen baseline consumes). The **adapter consumes the cxSMILES `|atomProp:`
form**, produced by `mol_to_cxsmiles(get_tmc_mol(path, 0)[0])` — exactly `tests/unit/test_cxsmiles_writer.py::_cx`.
`corpus_score.py` uses this path.

### The metric, concretely

`calculate_tmc_rmsd(mol_input, mol_generated, mol2_bonded=mol_generated)`: `mol_input =
Chem.MolFromXYZFile(xyz)` (no bonds — sphere1 is distance-based); `mol_generated` = the adapter's returned
real-metal **dative Mol with a 3D conformer** (sphere2 uses its bonds). Pass = `rmsd < 1.0`.

## 2. Environment setup (least-invasive — recommended)

The rxembed venv already satisfies every OIN-half dependency; no throwaway venv, no `uv pip install` into it,
no editable install of `oinsmiles`. Measured in-venv: **rdkit 2026.03.3, numpy 2.5.0, scipy 1.18.0,
networkx 3.6.1** — encoder path (`xyz2mol`→networkx, `cxsmiles`/`oin_aligner`→scipy) and `rmsd_utils`→scipy
all resolve. `oinsmiles` is made importable purely by **PYTHONPATH into a detached worktree** — the source is
never installed, the OIN checkout stays on `ag-dev`, and the local `ali-dev` branch is never moved.

```bash
SCRATCH=/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad
# 1. detached worktree of OIN ali-dev (does NOT switch OIN's branch, does NOT touch local ali-dev):
git -C /home/ali/Documents/Codes/OIN-SMILES worktree add --detach "$SCRATCH/oin-ali-dev" origin/ali-dev
export WT="$SCRATCH/oin-ali-dev"
# 2. run anything against it from the rxembed dir (--no-sync avoids a rebuild):
cd /home/ali/Documents/Codes/rxembed
PYTHONPATH="$WT/src:$WT/tests/integration" uv run --no-sync python "$SCRATCH/dry_smoke.py" \
    /home/ali/Documents/Codes/OIN-SMILES
# cleanup when the sweep is done:
git -C /home/ali/Documents/Codes/OIN-SMILES worktree remove "$SCRATCH/oin-ali-dev"
```

Why not a throwaway venv: it would re-download rdkit/scipy/networkx and still need `PYTHONPATH` for the
`ali-dev` source (OIN's installed `ag-dev` copy lacks the perception). The venv-reuse + worktree is strictly
less work and leaves both repos' checkouts untouched.

`PYTHONPATH` needs **both** `"$WT/src"` (the `oinsmiles` package) **and** `"$WT/tests/integration"`
(`rmsd_utils.py` is a loose module there, not in the package — the OIN harness imports it the same way).

## 3. Dry-smoke result (does the OIN half import + run standalone? — YES)

`scratchpad/dry_smoke.py`, run under the setup above, output:

```
=== imports OK: oinsmiles.utils.xyz2mol / .cxsmiles, oinsmiles.core.translator, rmsd_utils ===
[CisPlatin]  geo SPL  core [NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]        self-RMSD 0.0000 A
[FeCO5]      geo TBP  core [O+]#[C-]->[Fe](<-[C-]#[O+])(...)             self-RMSD 0.0000 A
[Ferrocene]  geo LIN  core [cH]12->[Fe+2]...(haptic)                     self-RMSD 0.0000 A
```

`scratchpad/corpus_score.py --encode-only --limit 10` encoded **9/10** tmQM to cxSMILES in ~1.5 s; the one
miss (`DADXAK`, a macrocyclic polyamine) fails inside OIN's `get_lig_mol` — an **OIN-side perception limit,
independent of rxembed** — and the sweep records it (`stage=encode`) and continues. Expect a handful of such
encode-stage misses; they cap the adapter's achievable pass-rate and belong to OIN, not the embed engine.

**Corpus counts.** `corpus_score.py::corpus_files` globs **144** from the on-disk OIN root
(`/home/ali/Documents/Codes/OIN-SMILES`, currently `ag-dev`) = 103 `tests/integration/tmQM` + 41
`tests/fixtures`. The `ali-dev` **worktree** carries only 137 (103 + 34 fixtures). Recommended split: **glob
the corpus from the on-disk root (144)** for the full set, but import `oinsmiles` from the `ali-dev` worktree
(perception) — the XYZ files are branch-independent coordinate data, so this is safe and hits the target 144.
(If exact-branch purity is wanted instead, point `--corpus "$WT"` for the 137 that ship on `ali-dev`.)

## 4. Running the sweep (AFTER the repoint lands)

`oin_adapter` repoint (Phase 2, another agent) — kernel confirmed at `src/rxembed/rdkit_embed/constraints/metal.py`:
- `oin_adapter/reconstruct.py:15`: `from rxembed.constraints import metal as _metal` → `from rxembed.rdkit_embed.constraints import metal as _metal`
- `oin_adapter/validate.py:15`:   `from rxembed.constraints import metal as M`      → `from rxembed.rdkit_embed.constraints import metal as M`
- **Also reconcile `_accept` arity** (see TL;DR blocker): `reconstruct.py:109` `_accept(cand, cx)` →
  `_accept(cand, cx, Chem.GetFormalCharge(cand))` (or give OIN's `total_charge` a default upstream).

Then:
```bash
cd /home/ali/Documents/Codes/rxembed
PYTHONPATH="$WT/src:$WT/tests/integration" uv run --no-sync python "$SCRATCH/corpus_score.py" \
    --corpus /home/ali/Documents/Codes/OIN-SMILES --n 24 --json "$SCRATCH/adapter_scores.json"
```
`corpus_score.py` (staged in scratchpad) does per structure: `get_tmc_mol → mol_to_cxsmiles → embed_cxsmiles
→ calculate_tmc_rmsd`, each structure wrapped in try/except (records `{name, cxsmiles, rmsd, pass, stage,
error}`), then prints **pass-rate over 144 at 1.0 Å**, RMSD median/mean/min/max + a 5-bucket distribution,
and a non-pass list with per-structure reasons. Flags: `--encode-only` (OIN half only, no adapter),
`--limit N` (smoke a prefix), `--n` (conformers), `--baseline`, `--json PATH`.

Smoke it incrementally: `--encode-only` first (proves the OIN half over all 144), then `--limit 5` (5
structures through the adapter) before the full run.

## 5. MetalloGen baseline (optional — heavier deps)

`--baseline` scores OIN's own generator on the same 144 (`XYZToSMILES().convert → OIN3DGeneratorMetallogen()
.generate → calculate_tmc_rmsd`), guarded so missing deps just print `[baseline SKIPPED]` and the adapter
result still stands. It is **heavier than the adapter path**: `metallogen_adapter` import already fails in the
rxembed venv with `ModuleNotFoundError: No module named 'pulp'` (a MetalloGen core dep). A *fair* side-by-side
also wants the same post-FF optimizer the OIN harness defaults to (g-xTB via `xtb` on `$XTB_EXE`; MACE is the
opt-in `torch==2.3.1` cu118 extra). Recommendation: run the adapter sweep first and report it standalone;
enable `--baseline` only in an env with `pulp` (+ `xtb`) installed — treat it as a separate, optional pass.

## Artifacts (scratchpad)

- `scratchpad/dry_smoke.py` — the OIN-half standalone proof (encoder + rmsd, no adapter).
- `scratchpad/corpus_score.py` — the staged sweep (adapter + optional baseline).
- `scratchpad/oin-ali-dev/` — detached OIN `ali-dev` worktree (remove with `git worktree remove` when done).
