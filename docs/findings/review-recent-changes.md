# Maintainability review — clarity of the recent structural changes

Read-only review. HEAD `003084d`. Lens: **can a future maintainer who wasn't here understand the
session's structural changes** — the kernel carve (`cf338d5`), the `_mol`/`.mol` split (`521505d`), the
`xyz_to_mol` move (`fb2bd40`), the OIN adapter repoint (`b6379e4`)?

Bottom line: the *code* of these changes is clean and, in the `_mol`/`.mol` case, exceptionally well
tested. The gaps are all in the **discoverability layer** — the project's own "Where things live" map is
stale about the very change under review, one cross-cutting invariant (the input-leaf cycle guard) is
documented but not enforced, and one logger escapes the pinning convention without a comment. No
correctness trap was found in the `_mol`/`.mol` routing (I checked for it specifically — see the note at
the end).

Findings are ranked by severity.

---

## 1. [high] [safe-fix] `CLAUDE.md` "Where things live" omits the kernel entirely and actively misdirects to two moved files
`CLAUDE.md:146-168` (the canonical map), with knock-on stale prose at `:26`, `:65`, `:130`.

"Where things live" is the first place a maintainer looks to answer *"what belongs in `rxembed.rdkit_embed`
vs `rxembed`, and why?"* — the exact question this session's biggest change raises. Right now it:

- has **no entry at all** for `src/rxembed/rdkit_embed/` — the whole carved kernel (bounds, the constraint
  model, `metal`, `distance`, `donor_orient`, `mechanisms`, `polyhedron`, `sphere`, `solver`,
  `coordination`, `report`, `vecmath`, `io`, `log`, `refine/ff`) is invisible in the map; and
- **actively points at the wrong location** in two rows:
  - `:153-154` — `src/rxembed/embed/` is credited with `bounds.py`, but `bounds.py` is now
    `src/rxembed/rdkit_embed/embed/bounds.py`. The shell `embed/` holds only `dispatch.py` + `mc.py`.
  - `:155-156` — `src/rxembed/constraints/` is credited with `base.py`, `metal.py`, `builders.py`, all
    three of which now live in `src/rxembed/rdkit_embed/constraints/`. The shell `constraints/` retains
    only `nci.py` plus a re-export `__init__` shim.

Same root cause, stale prose references elsewhere (lower stakes but same fix): `:26`
`constraints/builders.py::resolve_core`, `:65` `constraints/base.py`, `:130` `constraints/metal.py`,
`:124` `ff._ff_surrogate` (module moved to `rdkit_embed/refine/ff.py`). `:63` `constraints/nci.py` is
still correct — `nci` genuinely stayed shell-side.

**Suggestion.** Add a `src/rxembed/rdkit_embed/` bullet that states the boundary rule in one sentence —
"the pure embed engine: rdkit + numpy only, driven by a `Constraints` struct; imports nothing from the
shell (locked by `tests/test_import_hygiene.py`)" — and lists the kernel modules. Correct the `embed/`
and `constraints/` rows to name only what actually remains shell-side (`dispatch`/`mc`; `nci`). The
graduate-to-standalone intent is already well recorded in `rdkit_embed/README.md` and the root
`pyproject.toml` (`:35-38`), so the map only needs a one-line pointer to it, not a repeat.

## 2. [medium] [safe-fix] `inputs.py`'s "imports nothing from `rxembed`" cycle guard is documented but nothing enforces it
`tests/test_import_hygiene.py` (gap); the invariant is asserted only in prose at `src/rxembed/inputs.py:5-6`
and `CLAUDE.md:151-152`.

The whole point of moving `_xyz_to_mol`/`parse_smiles` into `inputs.py` (`fb2bd40`) was to break the
`metal → dispatch → metal` cycle: the leaf must import nothing from `rxembed`. The *kernel* counterpart of
this rule is locked hard — `test_io_is_the_bottom_of_the_stack` asserts `_imports(io.py) == []`, and
`test_metal_does_not_reach_the_embed_dispatch` guards the cycle from the metal side. But `inputs.py` is a
**shell** module, so it is outside `_KERNEL` and no test touches it. A future edit that reaches back into
`rxembed` from `inputs.py` (e.g. `from rxembed.constraints import metal` to reuse a helper) would silently
re-form the exact cycle this commit removed, and the suite would stay green.

**Suggestion.** Add a one-line test mirroring the io one, e.g.
`assert sorted(_imports(_SRC / "inputs.py")) == []`, with a docstring noting it is the shell-side twin of
`test_io_is_the_bottom_of_the_stack` and names the `metal → dispatch → inputs` cycle it prevents.

## 3. [medium] [judgment-call] `refine/ff.py`'s logger breaks the pinning convention with no comment — a latent graduation escape and an inconsistency inviting a wrong "fix"
`src/rxembed/rdkit_embed/refine/ff.py:17` vs the four pinned siblings (`constraints/metal.py:27`,
`embed/bounds.py:18`, `constraints/builders.py:35`, `constraints/sphere.py:37`).

Four kernel loggers are pinned to literal `"rxembed.*"` names, each with the comment
`# pinned name: kept under the "rxembed" logger tree`. `ff.py` alone uses `logging.getLogger(__name__)`,
which today resolves to `rxembed.rdkit_embed.refine.ff`. Two problems:

- **Inconsistency that invites the wrong edit.** Because `ff.py`'s `__name__` name still sits under the
  `rxembed` tree today, `set_verbose()` reaches it fine — so a maintainer "harmonizing" the four pinned
  loggers to match `ff.py`'s tidier `__name__` idiom would see no breakage and conclude the pins are
  pointless boilerplate. That silently defeats what the pin is *for*.
- **What the pin is actually for is never stated.** The comment says "kept under the rxembed logger tree"
  but not *why it can't just be `__name__`*: on graduation to a standalone `rdkit_embed` package,
  `__name__` becomes `rdkit_embed.refine.ff` — **outside** the `rxembed` tree — so `set_verbose()` (which
  configures the `"rxembed"` logger) stops reaching it, while the four pinned modules keep working. `ff.py`
  is the one kernel logger that will silently go dark on graduation.

**Suggestion.** Pin `ff.py` to `logging.getLogger("rxembed.refine.ff")` (matching how the siblings mirror
the *shell* layout and drop the `rdkit_embed` segment), and add the same `# pinned name` comment. Then
strengthen one pin comment to record the rationale, e.g. `# pinned, NOT getLogger(__name__): once the
kernel graduates to standalone rdkit_embed, __name__ would escape the "rxembed" tree that set_verbose()
configures`. That turns four cryptic identical comments into one explained convention.

## 4. [medium] [judgment-call] The `.mol` finalize-on-access **cost** is undocumented, and it already runs per-conformer in a real loop
`src/rxembed/pipeline.py:493-517` (the property); live instance at `oin_adapter/reconstruct.py:108-109`.

The property docstring is excellent on *behaviour* — connectivity-only, coords byte-identical, idempotent,
a pure no-op for organics — and `test_ml_connected.py` pins all of it. What it never says is that for a
metal complex **every access recomputes** a full `Chem.Mol(...)` copy + `restore_metal` (per metal) +
`connect_metal`; there is no caching. That is easy to miss precisely because the docstring's "no-op / copy
only when needed" framing reads as cheap. It already bites: `reconstruct.py` does
`for cid in ens.ids: cand = _cand_for(dative_ref, ens.mol, cid)`, re-finalizing the surrogate on every
conformer where `m = ens.mol` hoisted above the loop would finalize once.

**Suggestion.** Add one sentence to the property docstring: "Finalizes on **every** access and is not
cached — for a metal complex, hoist `m = ens.mol` out of hot loops." Optionally hoist the `reconstruct.py`
access. (Related, and adequately covered by the field comment + docstring today, but worth keeping in mind:
the only thing stopping a *future internal* Ensemble method from typing the natural-looking `self.mol` — and
thereby paying the cost *and* handing the DG/FF engine the connected graph UFF cannot type — is the
`_mol` naming convention. The property docstring's "Internal consumers read `_mol`" line carries that
weight; keep it.)

## 5. [low] [safe-fix] `rdkit_embed/README.md` "Still to do" lists work that landed this session (and was never a kernel task)
`src/rxembed/rdkit_embed/README.md:36-37`.

The kernel README's roadmap strikes through the xyz→mol item (correctly — `fb2bd40` did it) but still lists
"**Always carry the M–L bonds**" as pending, though `521505d` completed it and `tests/test_ml_connected.py`
locks it. A maintainer reading the kernel's own roadmap would think it is unfinished. Note also that this
item is *shell* work (the `.mol` property in `pipeline.py`), not a kernel concern — its presence on the
kernel to-do list slightly blurs the boundary the carve is meant to sharpen.

**Suggestion.** Strike it through / mark Done like the xyz→mol item above it, pointing at the `.mol`
property + `test_ml_connected.py`, or drop it as out-of-scope for the kernel. The "Drop scipy" item is
genuinely still open — leave it.

## 6. [low] [judgment-call] The placeholder `pyproject.toml` lives inside the installable package tree, guarded only by a header comment
`src/rxembed/rdkit_embed/pyproject.toml:1-8`.

The file is inert and its header says so clearly, and the root `pyproject.toml:35-38` cross-references it —
so this is *well* handled, not broken. The residual smell is that an active-looking `pyproject.toml` sits
under `src/` where `[tool.setuptools.packages.find]` discovers packages; an IDE may treat it as a project
root, and a future `uv` workspace glob could re-adopt it by accident. Naming it unmistakably inert (e.g.
`pyproject.toml.graduation-marker` / `.example`) would remove any doubt without losing the design marker.
Judgment call — the comment may be deemed sufficient.

---

## Note: the `_mol`/`.mol` routing is correct and thoroughly locked (checked, not assumed)
I specifically hunted for the correctness trap the brief flagged (an internal consumer reading `.mol` where
it needs `_mol`, or vice-versa) and did not find one:

- No `self.mol = …` assignment remains anywhere — the field rename to `_mol` plus the now read-only
  property is internally consistent (a stray assignment would raise). All `mc.mol = …` writes are on the
  `_MetalCtx` dataclass, a different object.
- No `Ensemble(mol=…)` keyword construction exists; every call site uses the positional first arg, which
  `_mol` preserves.
- `score`/`optimize` both call `self.minimize()` before building their output Ensemble, so `metal_bonds`
  is populated and the output's `.mol` is connected even from a bare (un-minimized) embed — I traced this
  because it was the most plausible place for a disconnected-surrogate leak.
- The property's `if mc is not None` ("pre-minimize") branch is sound: `minimize` tears down `_metal`
  (`pipeline.py:868-869`) and records `metal_bonds` (`:895-896`), so post-minimize the property takes the
  `mc is None` path and `connect_metal` is an idempotent no-op.
- `test_ml_connected.py` (15 cases) pins the guarantee on every mol-returning path *and* the hard invariant
  that `_mol` stays the bond-less surrogate that `.mol` never mutates.
- `test_import_hygiene.py` enforces the kernel's import closure, the io-bottom rule, and the `metal →
  dispatch` cycle break — a genuinely strong guard for the carve (the one gap is the shell-leaf twin,
  finding #2).
