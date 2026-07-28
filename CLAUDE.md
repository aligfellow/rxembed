# rxembed

`rxembed` (`src/rxembed/`) is a **unified constrained / templated conformer-embedding** package: one
composable chain that embeds conformers for anything from a bare SMILES to a frozen bimetallic transition
state with NCI-seeded binding modes, then searches, dedups, and scores them with real energies.

It is a **library, not a CLI** — you drive it from Python / notebooks. The convention is `import rxembed as rx`.

## The one chain

Every capability is *one more argument* on the same pipeline, not a new code path:

```python
import rxembed as rx
ens = rx.embed(source, ...).mc().prune()      # embed -> Monte-Carlo search -> dedup
ens.representatives()                          # the distinct modes
ens.score("gxtb").lowest(3).optimize("gxtb")   # real energies, then optimise the best
```

`rx.embed(source, *, metal, fix, constrain, template, contacts, coordinate, charge, n, seed, knowledge,
stereo)` returns an **`Ensemble`** (or an **`EnsembleSet`** when the input is inherently several candidates —
metal isomers, NCI binding modes, ambiguous coordination, **undefined stereocentres**). `rx.minimize(source,
*, fix, constrain, …)` is the search-free companion: same verbs, relax an existing geometry *toward* the
targets instead of conf-searching.

> **Constraint spec — three index-driven verbs** (`rdkit_embed/constraints/builders.py::resolve_core`, the
> clean-break replacement of the old `freeze`/`distances`/`angles`/`template`/`match`/`anchor`):
> **`fix`** (rigid — own-coords list / explicit-coords dict / exact-number dict, grafted or UFF-pulled),
> **`constrain`** (soft windows + π-stack planes, releasable by `mc(explore=)`), **`template`**
> (`(reference, {target_i: ref_i})` sugar for a coords-`fix`). Keys are **0-based atom indices** (xyz/graph
> order) — the resolver never SMARTS-matches internally; the user resolves SMARTS (two RDKit lines). This is
> a spec-layer change only; the embed mechanics below are unaffected.

| Want | Call |
|---|---|
| free / flexible | `rx.embed("CCO")` |
| NCI / vdW complex | `rx.embed("A.B", contacts="auto")` → one candidate per discovered grip |
| a specific H-bond grip | `rx.embed("A.B", contacts=rx.nci_modes(mol)["HB:…"])` |
| frozen TS core | `rx.embed(ts_xyz, fix=reacting)` → 0.000 Å graft |
| constrained TS **from SMILES** | `rx.embed("cat.substrate", fix={(i, j): d, (i, j, k): θ})` → verify `.measure()` |
| soft distance / angle / π-stack | `rx.embed(smi, constrain={(i, j): (lo, hi)})` |
| metal coordination isomers | `rx.metal("…[Pd]…", "square_planar")` → cis/trans, mer/fac… |
| the racemate (undefined centre) | `rx.embed("CC(N)C(=O)O")` → `EnsembleSet` of enantiomers (`stereo='enumerate'` keeps them separate) |
| replace a ligand on a known core | `rx.embed(analogue, template=(parent_xyz, {target_i: ref_i}))` |

**Pipeline stages** (the mutation contract is explicit): `mc`, `minimize`, `prune` build in place and chain;
`lowest`, `representatives`, `align` return a *new* ensemble; `view`, `landscape`, `cluster` never mutate. An
**`EnsembleSet` is chainable too** — `mc`/`minimize`/`prune`/`score`/`optimize`/`lowest`/`representatives`/`dump` map
over its candidates (returning an `EnsembleSet`, tags carried; `dump` writes one tagged `.xyz` per candidate), so
`rx.embed(anything).mc().prune().dump(...)` works whether the input is one molecule, a **racemate** (default
`stereo='racemic'` → an `EnsembleSet` of enantiomers), metal isomers, or NCI modes — logged per call. Each candidate
is searched, pruned (RMSD/MOI/energy/cluster), and scored **on its own constraints, WITHIN itself** — distinct
species are NEVER pooled or cross-pruned. Real-energy tiers via `score(refine=…)` (single point) and
`optimize(refine=…, level=…)` (geometry, frozen core held): **`ff` → `gfnff` → `gfn2` → `gxtb`**. An ensemble's
`energy_kind` is **`ff`** (surrogate UFF/MMFF, from `minimize`/`score('ff')` — NOT comparable across species) or
**`real`** (xtb/g-xTB, from `score`/`optimize` — comparable, since enantiomers/diastereomers/isomers share a
formula). Ranking the candidates against each other is the one deliberate cross-species step — `EnsembleSet.best(n)`
— which **refuses `ff`** energies and demands `real` ones.

## Design principles

- **KISS / no over-engineering.** Adding a capability should be a *registry row or one argument*, not a
  branch. NCI contact types live in one `KINDS` registry (`constraints/nci.py`); the enumerator is generic
  over it. Dead machinery gets deleted.
- **One struct in, one struct out.** `Constraints` (`rdkit_embed/constraints/base.py`) is the single thing every
  *builder* fills and every *stage* reads. Nothing reaches into the pipeline sideways.
- **Edit, don't replace.** The ETKDG bounds matrix is *edited* from RDKit's knowledge-derived bounds, so
  experimental-torsion / basic-knowledge seeding survives even under a tight custom core.
- **Bias the seed, let energy decide.** Constraints bias the *starting* geometry; they are not truth.
  `mc(explore=True)` fires the search twice (grip preserved, grip released), pools, and lets a real energy
  choose.
- **Surrogate the hard bits.** A metal becomes a carbon surrogate for the force field (bonds stripped), held
  by soft shape constraints; the exact frozen TS core is grafted back by Kabsch (0.000 Å).
- **Fail loud, not silent.** A calculator that produces no energies *raises* (never a silent FF fallback).
  g-xTB has no ALPB, so solvent is a GFN2 thermodynamic-cycle correction (`E_gxtb(gas) + [E_gfn2(solv) −
  E_gfn2(gas)]`) — a real solvated energy, never a silent gas-phase.
- **Physical honesty over cleverness.** σ-holes only on polarisable heavies; a thione C=S is an acceptor, not
  a σ-hole donor; a reciprocal A-H···B / B-H···A 2-cycle can't coexist in one pose. Rules grounded in
  chemistry, each a small testable change.

## Development loop

Every non-trivial change runs the validation loop — the project's working discipline:

```
assess → plan → implement → robustness review (subagent) → edge-case review (subagent)
       → reassess → regress (tests) → update memory
```

- **Adversarial subagent reviews.** After implementing, spawn independent reviewers whose job is to *break*
  the change (a robustness pass and a chemistry edge-case pass). Fold findings back before "done".
- **Tests are the gate.** `tests/` (organic, constraints, NCI, metal isomers, frozen TS, template) must stay
  green; the frozen-core distance assertions must hold. xtb-binary and openconf cases are `skipif`-gated so
  the base suite runs anywhere. `just test` runs it.
- **Proof lives in notebooks.** New capability is demonstrated in `examples/*.ipynb` with the **real inline
  API — no hidden wrappers**: logging on, every stage dumped, key geometries rendered with `xyzrender`, each
  gated by `rx.geometry.check`. Clear outputs (`jupyter nbconvert --clear-output --inplace 0*.ipynb`) before a
  lightweight commit.
- **Memory carries state across sessions** (`~/.claude/.../memory/rxembed-*.md`).

## Toolchain

Copier template with **uv / ruff / ty / pytest / pre-commit / just**. `just check` = lint + type + test;
`just setup` = `uv sync` extras + `pre-commit install`; `just fix` autofixes. Extras keep the base install
light: `mc` (openconf), `nci` (xyzgraph, scikit-learn), `viz` (matplotlib, seaborn, pandas), `ase`, `racerts`.
Version is single-sourced from `pyproject.toml` via `importlib.metadata` in `__init__.py` — never hardcode a
second copy. `ruff D` (numpy pydocstyle) is on: one-line imperative docstring summaries + only the
non-inferable specifics; comments only for load-bearing *why*.

## Real-energy caveats

`score('gxtb')` / `optimize('gxtb')` need the `xtb` executable on `$XTB_EXE`; **GFN-FF needs a standard Grimme
`xtb`** on the same path. Embedding / search / prune are pure-Python (RDKit + openconf) and need no external
calculator. `mc()` needs openconf; without it the ETKDG seeds are still returned (degrades, doesn't crash).

**openconf is git-pinned** (`[tool.uv.sources]` in `pyproject.toml`) — its transition-metal support (the
`mc(preset="transition_metal")` preset + auto metal-move budget) is on upstream `main`, not yet on PyPI. `uv
sync` resolves + locks it; `just setup-openconf-dev` swaps in a local editable clone for co-development. NB
openconf's pose-freeze is *soft* (held atoms drift ~0.1 Å), so a constraint that must truly hold across `mc`
has to be one the rxembed relax also reads (a distance/angle), not a pose-hold alone.

**Metal surrogate — two atoms, one index.** The metal is a bond-less **carbon** in the distance geometry (its
excluded volume stops a ligand folding into the centre) and a bond-less **lithium** in the force field
(`rdkit_embed/refine/ff.py::_ff_surrogate`): Li's small vdW is a soft excluded-volume sphere that keeps every non-donor — heavy *and*
hydrogen — off the metal, which no per-atom floor did. The M-donor bonds stay stripped, so `prepare`/`restore`
must hand the metal's **oxidation state** back (not just its element), or every `score`/`optimize` runs at the
wrong total charge. M-donor distance is the fitted periodic model (`ml_distance`, element/group/**delocalised**
charge/hapticity — never the raw formal charge, a Kekulé artefact) with a P/As/Se dative cap.

**Donor orientation** (`rdkit_embed/constraints/donor_orient.py`): the stripped bond removes UFF's own terms, so
the ones that matter are put back, softly. `_orient_donor` holds an **sp** donor end-on (nitrile/CO) and a slow-inverting
**pnictogen** donor's protons splayed (P/As/Sb — their inversion barrier is one a local optimiser can't cross);
`_coplanar_donor` keeps a **conjugated sp2** donor's metal in the donor's own π-plane (a soft dihedral cap, plus
a wide M-O-C angle wall for a one-neighbour O so the plane bound gets a fix). All are caps/walls, not points —
the real energy decides where inside them to sit. NB `N[Co]` writes a *covalent* N (→ NH₂); a true ammine needs
the dative `[NH3]->[Co]`. A metal-bound sp3 **carbanion/amine stereocentre** is held by a charge-neutralised
dummy-D through every embed/relax (`_hold_donor_chirality`).

**Connectivity is truth.** An optimiser can hand back a *different species* with a plausible energy;
`.filter('connectivity')` (and `prune(by=['connectivity',…])`) re-perceives the graph and drops a conformer
whose bonding changed — flagged loudly, never silent. A double bond whose E/Z the coordination locks (an
α-diimine C=N in a ring closed *through* the metal) is not enumerated as a phantom pair (`stereo.py`). The
geometry gate additionally reports a **folded donor** — a ligand pointing the wrong way off its donor —
which every distance-based check is blind to (`geometry.donor_fold`/`donor_orientation`).

## Where things live

The tree is a **kernel/shell split**: the pure DG/FF engine is the in-place subpackage `rxembed.rdkit_embed.*`
(numpy + rdkit only — no pipeline, no calculators, no perception-heavy deps), and the user-facing pipeline +
QA gate + NCI + isomers + real-energy calculators are the shell `rxembed.*`. You import as `rx` and call the
shell; the shell drives the kernel — you rarely open a kernel file. The boundary is import-closed and locked by
`tests/test_import_hygiene.py` (no kernel module reaches up into the shell). The kernel is intended to graduate
to a standalone `rdkit_embed` package once it stabilises (its `pyproject.toml` there is an inert placeholder).

**Shell — `src/rxembed/` (the pipeline you drive):**

- `pipeline.py` — `embed`, `minimize`, `wrap`, `Ensemble` (mc/minimize/prune/score/optimize/representatives),
  `EnsembleSet`.
- `inputs.py` — `_xyz_to_mol` (path→Mol, xyzgraph perception) + `parse_smiles` (str→Mol); the shell leaf that
  adapts a user source before the kernel `embed(mol, …)` engine sees it (imports nothing from `rxembed`, so it
  forms no cycle).
- `isomers.py` — the `rx.metal` entry point: `enumerate_isomers` + the `Isomer`/`IsomerSet` data model
  (name-agnostic arrangement + Λ/Δ chirality, `select()`/`filter()`).
- `embed/` — `dispatch.py` (the embed machinery: source normalisation, metal path, template path,
  `_embed_dispatch`, `_stereo_expand`/`_stereo_enumerated_embed`), `mc.py` (openconf Monte-Carlo). *(The
  edited-bounds ETKDG embed is kernel-side, `rdkit_embed/embed/bounds.py`.)*
- `constraints/` — `nci.py` only (KINDS registry, binding modes, acceptor quality, reciprocal split). *(The
  `Constraints` struct + `builders`/`metal`/etc. are kernel-side, `rdkit_embed/constraints/`.)*
- `dedup/` — `prism` moi/rmsd/descriptor prunes + the distinct energy-aware `energy_prune` (`select.py`).
- `refine/` — `xtb.py` (executable interface), `calculator.py` (`resolve`, `XTB`, `ASE`). *(The
  constraint-enforcing restrained UFF + FF energies are kernel-side, `rdkit_embed/refine/ff.py`.)*
- `geometry.py` — the TS-aware geometry gate (`check`): broken conjugation, bad H positions, clashes, moved
  core, metal over-bond, **folded donor** (`donor_orientation`/`donor_fold`, census-calibrated); frozen/metal-
  aware (dative distances aren't clashes). Shell — it reads the kernel's `coordination` perception, never the
  reverse.
- `metrics.py` — `bonding_ok`, `connectivity` (graph diff), `coordination_changed` (a donor that left / a
  non-donor that joined the metal) — behind `.filter('connectivity')`.
- `stereo.py` — chirality fingerprints + the auto-preserve gate; **`enumerate_unassigned`** (the undefined-
  stereocentre racemate/diastereomer load-in: point R/S + E/Z, `onlyUnassigned`, meso-deduped, metal-safe,
  chiral-at-P; a coordination-locked C=N is not enumerated). Wired in `embed/dispatch.py`.
- `viz.py` — landscape / cluster / render helpers (matplotlib/seaborn, the `viz` extra).

**Kernel — `src/rxembed/rdkit_embed/` (the pure embed engine, `numpy + rdkit`):**

- `embed/bounds.py` — the edited-bounds ETKDG embed (RDKit knowledge-derived bounds *edited* from the
  constraints) + seed-count scaling.
- `refine/ff.py` — `restrained_uff` (the constraint-enforcing minimiser), `ff_energies`, `_ff_surrogate` (the
  bond-less Li force-field surrogate).
- `constraints/` — `base.py` (`Constraints`, `copy`/`compose`, `relaxed()`), `builders.py` (`resolve_core` —
  the fix/constrain/template resolver), `metal.py` (the polyhedron per isomer: `surrogate_metal`, `coordination`,
  `restore_metal`/`connect_metal`, `n_sites`, `geometry_for`), `distance.py` (`ml_distance` — the fitted M–L
  periodic model), `donor_orient.py` (`_orient_donor`/`_coplanar_donor`), `mechanisms.py` (each field's
  co-located DG + FF writer), `polyhedron.py` (slot/parity descriptor), `solver.py`/`sphere.py` (the
  scipy-gated coordination-sphere solver, effectively a rare fallback).
- `coordination.py` — the coordination-sphere perception ruler (which atoms coordinate which metal; the fold /
  overbond gates the shell `geometry` reads).
- `io.py` (`repair_bond_stereo` — the bottom of the stack, imports nothing), `log.py`, `report.py`,
  `vecmath.py` — the kernel leaves.

- `examples/*.ipynb` — the 8-notebook breadth tour; `examples/structures/` — static TS `.xyz` geometries.

See `plan.md` (a local, un-checked-in roadmap file) for forward threads; the tracked reasoning lives in
`docs/findings/`.
