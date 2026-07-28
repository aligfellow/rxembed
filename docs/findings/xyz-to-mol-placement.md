# Where does xyz→mol belong: embed kernel or shell?

**Question (maintainer):** "I don't know that we should have the xyz→mol function? like that's user
not the embed side is it?" — i.e. does XYZ→RDKit-Mol conversion belong in the embed **kernel**
(`src/rxembed/rdkit_embed/`, closure = numpy + rdkit) or on the **shell** (`src/rxembed/`,
user-input prep that drives the kernel)?

## TL;DR verdict: MOVE (small, graduation-prep — not a correctness/closure fix)

The maintainer's instinct is right. Two functions in the kernel — `_xyz_to_mol` and `parse_smiles`
— are **user-input adaptation** (path/string → Mol) with **zero kernel-engine callers**; every real
call site is on the shell. The kernel's actual engine entry `embed(mol, cons, …)` already takes a
**Mol**. The kernel's own README already lists this as a to-do ("the kernel should take a mol, not a
path or string").

The move is **not** forced by the closure (they don't break numpy+rdkit — `xyzgraph` is a declared,
guarded soft-extra), and it is **not** urgent. It's a layering cleanup that pays off when the kernel
graduates to a standalone `rdkit_embed`. A third io function, `repair_bond_stereo`, is genuine engine
plumbing (has a real kernel caller) and **must stay**. So the move *splits* one small module, it
doesn't empty it.

---

## 1. Inventory — every xyz→mol / source-parsing path

All source-parsing lives in **one kernel module**, `src/rxembed/rdkit_embed/io.py` (3 functions),
plus one stray raw-RDKit read in the shell geometry gate. The kernel `embed()` engine touches none of
it.

| Symbol | File (kernel/shell) | Input → output | Callers | Kernel-engine caller? |
|---|---|---|---|---|
| `_xyz_to_mol(path, charge=0)` | `rdkit_embed/io.py` (**kernel**) | **path/str** → Mol (xyzgraph perception, `rdDetermineBonds` fallback, + conformer) | `embed/dispatch.py:47` (`_normalize`), `embed/dispatch.py:562` (template ref), `isomers.py:466` (`rx.metal`) — **all shell** | **No** (metal.py:418 is only a comment) |
| `parse_smiles(smi)` | `rdkit_embed/io.py` (**kernel**) | **str** → Mol (clear error, not `None`) | `embed/dispatch.py:51` (`_normalize`), `isomers.py:468` — **all shell** | **No** |
| `repair_bond_stereo(mol)` | `rdkit_embed/io.py` (**kernel**) | **Mol** → Mol (re-derive/drop orphaned bond stereo after bond surgery) | `stereo.py:265` (shell) **and `constraints/metal.py:427` (kernel `surrogate_metal`)** | **Yes** |
| `Chem.MolFromXYZFile(reference)` | `geometry.py:333` (**shell**) | path → coords-only Mol (QA-gate reference) | inline in `geometry.check` | n/a — already shell, inline RDKit |

Notes:
- The kernel engine entry is `embed(mol, cons, n, seed, …)` in `rdkit_embed/embed/bounds.py:174` — it
  takes a **Mol**. `bounds.py` has **zero** xyz/path/`MolFrom*` references. The engine already
  "takes a mol, not a path."
- `io.py`'s only non-rdkit import is `xyzgraph`, guarded by `try/except ImportError` → `rdDetermineBonds`.
  It imports **no** kernel module (it is the bottom of the stack). So `_xyz_to_mol` does **not**
  violate the numpy+rdkit closure — `xyzgraph` is already the kernel's declared soft-extra for
  perception.
- Constant ownership (for a clean split): `_AROMATIC_BO_TOL` is used **only** by `_xyz_to_mol` (moves
  with it); `_STEREO_REFS` is used **only** by `repair_bond_stereo` (stays).

## 2. Is there an xyz→mol in the kernel at all? — Yes, and it's user-input adaptation

`rdkit_embed/io.py` *is* an xyz-parser: `_xyz_to_mol` reads a **path** and builds a fresh RWMol from
raw coordinates (perceived bonds + a conformer). `parse_smiles` is the SMILES twin. Both take a
path/string and produce the Mol that the shell then hands to the engine.

This is squarely **"user input adaptation," not "the embed engine needs this."** The distinguishing
test — *does the engine call it?* — comes back **no** for both: they have zero kernel-engine callers,
and `embed()` already consumes a Mol. Contrast `repair_bond_stereo`, which the kernel's own metal
surrogate surgery (`constraints/metal.py:427`, inside `surrogate_metal`) calls to clean orphaned C=N
stereo after stripping M–donor bonds — that is engine plumbing on a Mol graph and belongs in the
kernel.

The kernel README says so explicitly (`rdkit_embed/README.md`, "Still to do"):

> **Reconsider `io._xyz_to_mol`.** xyz→mol reads as a user / IO concern, not an embed-engine one —
> the kernel should take a mol, not a path or string. Decide whether it belongs in the shell instead.

## 3. Why it currently sits in the kernel — a deliberate, *tested* cycle-avoidance choice

This is not an accident, and any move must respect the reason. Two golden import-hygiene tests
(`tests/test_import_hygiene.py`) pin the current layout:

- `test_io_is_the_bottom_of_the_stack` (l.135): `rdkit_embed/io.py` must import **no** kernel module.
- `test_metal_does_not_reach_the_embed_dispatch` (l.140): kernel `constraints/metal.py` must not reach
  `rxembed.embed.*`. Its docstring records the history: `enumerate_isomers` (shell `isomers.py`, the
  `rx.metal` path) used to import `_xyz_to_mol`/`parse_smiles` **from `embed.dispatch`**, and
  `dispatch` imports `constraints.metal` at module level — so `rx.metal(...)` dragged the whole shell
  and formed a `metal → dispatch → metal` cycle. The fix recorded there: *"`rdkit_embed.io` now owns
  both readers."*

So the readers were pushed **down** into a leaf that both the shell embed dispatch **and** the
metal/isomers layer can reach without a cycle. The constraint the move must satisfy is therefore:
**the readers' new home must be a leaf that both `embed/dispatch.py` and `isomers.py` can import
without re-forming the dispatch↔isomers cycle** (dispatch already imports isomers at `dispatch.py:17`).

That rules out the naive "move it into `embed/dispatch.py`" (which is, confusingly, where CLAUDE.md's
"Where things live" still *says* it lives — that line is already stale). Putting the readers back in
`dispatch` re-creates the exact cycle these tests guard against. The correct target is a **new shell
leaf** that imports nothing from `rxembed`.

## 4. The exact minimal move

Split `io.py` along the "engine vs input" line:

- **Stays** in kernel `src/rxembed/rdkit_embed/io.py`: `repair_bond_stereo` + `_STEREO_REFS` (has a
  kernel caller; pure Mol→Mol, rdkit-only). Rewrite the module docstring — its current claim that
  `_xyz_to_mol`'s xyzgraph-optionality "lets the metal layer read a geometry without … closing a
  cycle" describes the *readers*, which are leaving; the surviving `repair_bond_stereo` is what the
  kernel metal layer actually calls.
- **Moves** to a **new shell leaf** `src/rxembed/inputs.py` (or, to mirror the kernel name,
  `src/rxembed/io.py`): `_xyz_to_mol` + `parse_smiles` + `_AROMATIC_BO_TOL`. This module imports only
  `rdkit` (+ guarded `xyzgraph`) — nothing from `rxembed` — so it is a shell leaf with no cycle.

### Import edits (blast radius — mechanical, no behaviour change)

Source (3 files):
1. **New** `src/rxembed/inputs.py` — the two readers + their constant.
2. `embed/dispatch.py:26` — `from rxembed.rdkit_embed.io import _xyz_to_mol, parse_smiles` →
   `from rxembed.inputs import _xyz_to_mol, parse_smiles`. (Keeps binding both names in dispatch's
   namespace — see re-export note below.)
3. `isomers.py:19` — its `_io` is used **only** for `_xyz_to_mol`/`parse_smiles` (lines 466, 468);
   `repair_bond_stereo` is not used here. Repoint those two calls to `rxembed.inputs`. isomers.py can
   then drop the `from rxembed.rdkit_embed import io as _io` import entirely.

Unchanged (still read `repair_bond_stereo` from kernel io):
- `constraints/metal.py:22,427` — unchanged; kernel closure and `test_io_is_the_bottom_of_the_stack`
  still hold (kernel io.py still imports no kernel module).
- `stereo.py:17,265` — unchanged.
- `rdkit_embed/__init__.py` — `io` stays in `__all__`; neither reader was ever a kernel public export,
  so the drop-in surface (`embed`, `Constraints`, `resolve_core`, …) is unaffected.

Tests (targeted — do not run the full suite; another agent owns that):
- `tests/test_determinism.py:14` — `from rxembed.rdkit_embed.io import _xyz_to_mol` → `from rxembed.inputs …`.
- `tests/test_mol_state.py:21` — split: `_xyz_to_mol` from `rxembed.inputs`, `repair_bond_stereo` from
  `rxembed.rdkit_embed.io`.
- `tests/test_frozen.py:67`, `tests/test_connectivity.py:16/547/643`, `tests/test_metal_charge.py:11`
  import `_xyz_to_mol` **from `rxembed.embed.dispatch`** — these keep working **only if** dispatch.py
  keeps the name bound in its namespace (step 2 does: `from rxembed.inputs import _xyz_to_mol` re-binds
  it). No edit needed, but verify the re-export stays.
- Doc-only staleness to sweep: `tests/test_import_hygiene.py:145` ("`rdkit_embed.io` now owns both
  readers"), `tests/test_mol_state.py:83`/`:145` docstrings, and CLAUDE.md "Where things live"
  (l.150 lists `_xyz_to_mol` under `embed/dispatch.py`).

### Does it break the closure or the golden tests?
- **Closure:** it *improves* it. `xyzgraph` (used only by `_xyz_to_mol`) leaves the kernel entirely,
  shrinking the kernel soft-extra list; kernel io.py reduces to a pure rdkit Mol→Mol repair.
- **Import hygiene tests:** both still pass — kernel io.py still imports no kernel module; kernel
  metal.py still doesn't reach `rxembed.embed`.
- **Behavioural tests:** none change behaviour — this is an import-path rename plus one re-export kept
  intact.

## 5. KISS weigh — recommendation

**Recommend the move, but treat it as graduation-prep, not a blocker.** In favour: it's exactly the
maintainer's instinct, it's already on the kernel's own to-do list, the two functions have no
kernel-engine caller, and it removes a whole soft-dependency (`xyzgraph`) from the kernel — a genuine
step toward "numpy + rdkit only." Against urgency: the closure isn't actually violated today, the
functions are harmless where they sit, and `repair_bond_stereo` keeps `io.py` alive regardless, so
this is a *split*, not a deletion. It is safe to defer until the kernel is split out to a standalone
package, at which point shipping path/SMILES parsing inside a drop-in that documents "hand me a Mol"
becomes the concrete wart the move removes.

---

## 6. Landed — 2026-07-22, commit `fb2bd40`

The move shipped as recommended. `_xyz_to_mol` + `parse_smiles` + `_AROMATIC_BO_TOL` now live in the new
shell leaf `src/rxembed/inputs.py` (imports rdkit + guarded xyzgraph, nothing from `rxembed`);
`repair_bond_stereo` + `_STEREO_REFS` stayed in kernel `rdkit_embed/io.py`. Callers repointed
(`embed/dispatch.py` re-binds both names, so tests importing via `rxembed.embed.dispatch` were untouched;
`isomers.py` now imports `rxembed.inputs`). Test imports updated in `test_determinism.py`/`test_mol_state.py`;
the `rxembed.embed.dispatch` re-export kept `test_frozen`/`test_connectivity`/`test_metal_charge` working with no
edit. Docs swept: CLAUDE.md "Where things live", the kernel README to-do (marked done), the placeholder
`rdkit_embed/pyproject.toml` extras (xyzgraph dropped from the kernel closure), and the stale
`io._xyz_to_mol` docstrings/comments (import-hygiene test, `test_mol_state`, `metal.py`).

Gates green on the pure move: full suite **380 passed / 2 skipped**, golden **23/23 bit-identical**,
import-hygiene green (kernel io.py still imports no kernel module; metal.py still doesn't reach
`rxembed.embed`), ruff clean. xyzgraph confirmed out of the kernel's code closure — its only remaining
mentions in `rdkit_embed/` are documentation.
