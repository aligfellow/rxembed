# Maintainability review — DEAD CODE / CRUFT / STALE DOCS / SIMPLIFICATION

*Read-only pass at HEAD `003084d`. Lens: concrete deletions and doc fixes. Extends
`overnight/maintainability.md` (F1–F17, captured at `ce59f34`, pre-carve) — this pass re-checks which of
those survived the kernel carve + the `xyz_to_mol`/`inputs.py` move, and adds carve-era findings. It does
**not** restate the F1–F17 analysis; §5 records only which of them are now resolved vs still-live.*

**Scope note.** All "dead" claims below are backed by whole-word grep over `src` + `tests` + `oin_adapter`.
A function-level dead-symbol scan (AST defs cross-referenced against the whole tree) found **zero** truly
dead module-level functions/classes — every candidate resolved to a dotted-attribute call (see §5). `ruff
--select F401,F811,F841` on `src` is **clean**. The debt here is stale *docs* and two tiny redundant
code artifacts, not a graveyard.

---

## RANKED FINDINGS

### C1 — `[high] [safe-fix]` scipy sphere-solver points at a `'sphere'` extra that does not exist in the shipped package
`src/rxembed/rdkit_embed/constraints/sphere.py:191-192`

**Problem.** On a missing scipy the solver raises:
```
"the coordination-sphere solver needs scipy — install the 'sphere' extra"
```
The **root** `pyproject.toml` (the installable one) declares only `ase`, `viz`, `all` extras
(`pyproject.toml:27-30`) — **no `sphere` extra, and scipy is not a direct dependency**. `pip install
rxembed[sphere]` errors. scipy is importable today only because `scikit-learn` (`pyproject.toml:18`)
drags it in transitively, so the message is misleading now and becomes a *silent no-op* the moment the
kernel graduates to standalone `numpy+rdkit` (the whole point of the placeholder pyproject). The
`sphere = ["scipy"]` extra exists **only** in the inert placeholder `src/rxembed/rdkit_embed/pyproject.toml:36-37`,
which "nothing reads" (its own header, line 1).

**Grep evidence.**
- `grep -n optional-dependencies pyproject.toml` → `ase`, `viz`, `all` only; no `sphere`.
- `grep -nE 'scipy|scikit-learn' pyproject.toml` → only `scikit-learn` (line 18); scipy never named.
- Message string: `sphere.py:192`. Placeholder-only extra: `rdkit_embed/pyproject.toml:37`.

**Fix (matches the maintainer's own decision doc `scipy-to-numpy-sphere.md` §4, recommendation (b)).**
Declare the extra in the **root** `pyproject.toml` so the message becomes true:
```toml
[project.optional-dependencies]
sphere = ["scipy"]
all = ["ase", "matplotlib", "seaborn", "xyzrender>=0.3.8", "scipy"]   # fold sphere into all
```
If the maintainer instead resolves J1 (below) toward *deleting* the solver, the message is moot — but
until then, the packaging line is the correct fix. A no-cost fallback that is independent of the
keep/delete decision: reword the message to `"install scipy (pip install scipy)"` so it never names a
non-existent extra. Prefer declaring the extra (it also matches the placeholder pyproject and the README).

---

### D1 — `[medium] [safe-fix]` CLAUDE.md "Where things live" is stale post-carve: three file locations wrong, the whole kernel + `isomers.py` unlisted
`CLAUDE.md:153-156` (+ omissions)

**Problem.** The single biggest structural change in the repo — carving the embed engine into the
`rxembed.rdkit_embed` subpackage — is not reflected in the file map a newcomer is handed:
- `CLAUDE.md:154` lists `bounds.py` under `src/rxembed/embed/`. It is at
  **`src/rxembed/rdkit_embed/embed/bounds.py`**. `src/rxembed/embed/` holds only `dispatch.py`, `mc.py`.
- `CLAUDE.md:155-156` lists `base.py`, `metal.py`, `builders.py` under `src/rxembed/constraints/`. All
  three are at **`src/rxembed/rdkit_embed/constraints/`**; `src/rxembed/constraints/` holds only `nci.py`.
- The `rxembed.rdkit_embed` kernel is **not mentioned once** (`grep -n rdkit_embed CLAUDE.md` → nothing) —
  none of `bounds/ff/base/builders/metal/distance/donor_orient/mechanisms/polyhedron/solver/sphere/coordination/io/log/report/vecmath`
  is placed.
- `isomers.py` (770 lines — the entire `rx.metal` isomer enumerator + `Isomer`/`IsomerSet` data model, the
  second-largest shell file) is **not mentioned** (`grep -n 'isomers.py' CLAUDE.md` → nothing).

**Grep evidence.** File tree vs `CLAUDE.md:153-156`; `grep -n 'isomers.py\|rdkit_embed' CLAUDE.md` → NOT MENTIONED.

**Fix.** Retarget the two bullets to `rdkit_embed/`, add a kernel bullet (one line naming
`rdkit_embed/` as the pure engine: `embed`/`bounds.py`, `restrained_uff`/`ff.py`, `constraints/{base,builders,metal,distance,donor_orient,mechanisms,polyhedron,solver,sphere}`,
`coordination.py`+`geometry.py`'s ruler, `io.py`/`vecmath.py`/`report.py`), and add an `isomers.py` bullet.
*(The `inputs.py` bullet at `CLAUDE.md:150` is already correct — the `xyz_to_mol` move was documented; only
the constraints/embed/kernel bullets lagged.)*

---

### D2 — `[medium] [safe-fix]` CLAUDE.md body attributes donor-orientation to `constraints/metal.py`; it moved to `donor_orient.py`
`CLAUDE.md:130`

**Problem.** "**Donor orientation** (`constraints/metal.py`): … `_orient_donor` … `_coplanar_donor` …
`_coplanar_donor`" — but both functions now live in **`rdkit_embed/constraints/donor_orient.py`**
(`_orient_donor` at :264, `_coplanar_donor` at :301). A maintainer following the pointer opens the wrong
file. (`_hold_donor_chirality`, `materialise_phantoms` in the surrounding prose *are* still in `metal.py`
— :333/:580 — so only the orientation attribution is wrong.)

**Grep evidence.** `grep -nE 'def _orient_donor|def _coplanar_donor' src` → both in `donor_orient.py`.

**Fix.** Change the parenthetical to `constraints/donor_orient.py`.

---

### D3 — `[low] [judgment-call]` kernel README "Still to do" recommends the numpy rewrite the maintainer's own decision doc *rejected*
`src/rxembed/rdkit_embed/README.md:29-31`

**Problem.** The README's first open item is *"Drop scipy from `constraints/sphere.py`. Replace the scipy
solver with a small numpy implementation … the numpy version must match those [rescue] cases."* But
`docs/findings/scipy-to-numpy-sphere.md` (§3–4) **measured and rejected (a) numpy-replace** (strictly less
robust on the exact contradictory cases the solver exists for) and recommends **(b) keep scipy as a
*declared* extra**. So the one forward-looking doc a newcomer reads points them at a task the investigation
already refuted. This is the doc counterpart of C1.

**Grep evidence.** `README.md:29-30` vs `scipy-to-numpy-sphere.md:8-14,122-156`.

**Fix.** Replace the "Drop scipy → numpy" bullet with the decision doc's outcome: *keep scipy as the
declared optional `sphere` extra (C1); the numpy rewrite is measured-rejected; the only live question is
keep-vs-delete the solver (J1).* Doc-only, prevents an agent re-attempting a refuted change.

---

### Dead1 — `[low] [safe-fix]` `_coplanar_donor` still carries the dead `donor_set` parameter (prior F9, survived the carve)
`src/rxembed/rdkit_embed/constraints/donor_orient.py:301` · call site `metal.py:904`

**Problem.** `def _coplanar_donor(mol, metal, d, donor_set, cons)` never reads `donor_set` anywhere in its
body; the call site passes `real_od` purely to satisfy the signature. It is threaded in exactly as into
`_orient_donor` — where `donor_set` **is** load-bearing (the APEX check, `donor_orient.py:296`) — implying
a coupling that does not exist. (The sibling F13 collision is **already fixed**: the function's own
docstring records that the `_CONJ_O_MDC_ANGLE` setdefault was "dead on every fixture, deleted.")

**Grep evidence.** `grep -n donor_set donor_orient.py` → only `:264` (`_orient_donor`, used at `:296`) and
`:301` (`_coplanar_donor` signature — never used below it). Call site: `metal.py:904`
`_coplanar_donor(mol, metal, d, real_od, c)`.

**Fix.** Drop `donor_set` from the signature (`donor_orient.py:301`) and drop `real_od` from the call
(`metal.py:904`). Pure deletion, no behaviour change.

---

### Dead2 — `[low] [safe-fix]` `GEOM` is a provably-redundant copy of `GEOM_OPTIONS[n][0]` (prior F5, survived the carve)
`src/rxembed/rdkit_embed/constraints/metal.py:80` · read only at `metal.py:755`

**Problem.** `GEOM` (metal.py:80) duplicates the first element of every `GEOM_OPTIONS` row (metal.py:92).
Re-verified by AST literal-eval at this HEAD: `GEOM == {n: GEOM_OPTIONS[n][0]}` is **True for all keys
2–8**. `GEOM` is read at exactly one site — `GEOM.get(n_donors)` (metal.py:755) — so it reads as
independent data when it is derived, adding one more table to the per-geometry alignment burden
(prior F6).

**Grep evidence.** `grep -nE '\bGEOM\b|GEOM_OPTIONS' metal.py` → def :80/:92, sole read :755; AST check
prints `GEOM == {n:OPT[n][0]} ? True`.

**Fix.** Delete `GEOM`; replace the read with `GEOM_OPTIONS.get(n_donors, (None,))[0]` (preserving the
`.get`-returns-`None`-for-unknown-CN semantics). *(Do this as part of, or consistent with, prior F6's
polyhedron-table co-location — but keep `ANGLES` hand-authored: all-pairs is measured-rejected.)*

---

### J1 — `[medium] [judgment-call]` The whole coordination-sphere solver (`sphere.py` + `solver.py`, ~340 lines + scipy) is effectively test-only — surface keep-vs-delete
`src/rxembed/rdkit_embed/constraints/sphere.py` (195 L), `solver.py` (144 L)

**Problem (surface, do not apply).** Per `scipy-to-numpy-sphere.md` §2 and the sphere memory note: 246/251
smoothing calls settle at `tol=0.0`; no real complex reaches the solver; the only real molecule that
enters `solve()` (ferrocene) solves a *feasible* sphere (a repair-to-self, not a rescue); the sole
exercisers of the scipy path are synthetic `@scipy_only` fixtures. The output is firewalled by the
`tol2 < tol` adoption guard, so a missing solver is already the shipped default. CLAUDE.md says "dead
machinery gets deleted," and an ablation over 144 structures "did not justify itself" — yet it is retained.
This is the near-dead reachability the brief asked to *flag as a decision, not delete*: it is the one place
in the tree that is arguably subtractable machinery, and it is coupled to C1's packaging bug.

**Not proposing a deletion.** Per the hard constraints, scipy keep-vs-delete is a pending maintainer call;
numpy-replacing the solver is measured-refuted. **Decision to surface:** either (b) keep + declare the
`sphere` extra (C1) and update the README (D3), or (c) delete `sphere.py`+`solver.py`+the `@scipy_only`
tests + the `_feasible_bounds` fallback branch (removes scipy from the kernel with *less* code than
keeping it). Both close C1; pick one.

---

## §5 — VERIFIED NOT-DEAD / ALREADY-RESOLVED (so a later pass does not re-flag them)

- **No function-level dead code.** An AST scan flagged 7 candidates with no non-dotted reference —
  `DGContext`, `donor_chirality_sign`, `disconnect_metal`, `lone_pair_donors`, `SphereSolver`,
  `satisfies_spec`, `enumerate_unassigned`. **All 7 are live**, reached via dotted module access
  (`_mech.DGContext` bounds.py:91; `_metal.donor_chirality_sign` dispatch.py:353/pipeline.py:974;
  `_metal.disconnect_metal` pipeline.py:564; `_metal.lone_pair_donors` dispatch.py:233;
  `_sphere.SphereSolver` solver.py:83; `_stereo.satisfies_spec` pipeline.py:986/1014/1475;
  `_stereo.enumerate_unassigned` isomers.py:230/dispatch.py:471). None is dead.
- **Prior F1 (field-by-field `Constraints` copy footgun) is RESOLVED.** `base.py` now has registry-driven
  `copy()` (`_CLONE`, :126) and `compose()` (`_MERGE`, :174) with an **import-time sync guard** (:191-195)
  that raises if a new field is added without a policy. The settle site (pipeline.py:652) and `relaxed()`
  (base.py:98) both route through `copy()`. The class of bug F1 named can no longer occur silently.
- **Prior F13 (`_CONJ_O_MDC_ANGLE` shadowed no-op) is RESOLVED** — deleted, per the `_coplanar_donor`
  docstring (donor_orient.py) and `grep -rn _CONJ_O_MDC_ANGLE src` → no matches.
- **The `xyz_to_mol` / `inputs.py` move is clean.** `_xyz_to_mol`+`parse_smiles` live in `rxembed.inputs`
  (imports nothing from `rxembed`); `rdkit_embed/io.py` retains only `repair_bond_stereo` (used by
  stereo.py:265, metal.py:427). No code imports the old `io._xyz_to_mol`; `metal.py:418` is only a comment.
  `test_import_hygiene.py:136` pins `io` importing no kernel module. No stale import paths found.
- **Logger names `rxembed.constraints.metal/.builders/.sphere` are deliberately pinned, not stale**
  (metal.py:27, builders.py:35, sphere.py:37, solver.py docstring) — `test_permutation_warning.py:59/70`
  asserts on `logger="rxembed.constraints.metal"`. Leave as-is.
- **No stray scaffolding is tracked.** `.coverage`, `coverage.xml`, `src/rxembed.egg-info/`,
  `foldwall_dump/`, `__pycache__/dcap_proto.cpython-313.pyc` (an orphan `.pyc` of a deleted scratch module)
  all exist on disk but `git ls-files` shows **none** tracked — `.gitignore` covers them. Nothing to delete
  from the repo; the orphan `.pyc` is harmless local cruft (`rm` it locally if desired).
- **No library code depends on `playground/` or `overnight/`** — `grep -rnE 'playground|overnight' src`
  → nothing. (Per brief: not proposing to touch the maintainer's `overnight/`/`playground/`.)
- **Prior F8 (`_PHANTOM_FLOOR` misleading name) is effectively moot post-carve.** The constant now sits in
  `mechanisms.py:36` beside the haptic-centroid logic and keys off the `phantoms` field (base.py) — the
  name now *matches* what it floors (the centroid dummy), so the grep-trap F8 described is gone.

---

## §6 — DID NOT PROPOSE (measured-refuted / load-bearing — per brief hard constraints)

Merging `floors`/`dg_floors` (base.py `_merge_floor`/`_merge_relief` differ by design, docstring at
:206-215) · all-pairs `metal.ANGLES` · charge-keyed M–L contraction · deleting the donor-orientation caps
(`_orient_donor`/`_coplanar_donor`) · numpy-replacing the sphere solver · re-packaging the kernel ·
re-litigating the Xe/Li surrogate. A cap/floor/angle that *looks* redundant here is load-bearing.
