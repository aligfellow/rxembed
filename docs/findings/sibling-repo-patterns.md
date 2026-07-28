# Sibling-repo onboarding patterns (openconf · ML-FSM · prism_pruner)

Read-only study of three sibling packages the maintainer holds as the "rdkit-style,
easy-to-onboard" bar. Goal: extract concrete patterns rxembed (and its embed kernel)
should emulate. Nothing in any repo was modified.

Repos studied:
- `/home/ali/Documents/Codes/openconf` — conformer generator (MMFF + torsional MC)
- `/home/ali/Documents/Codes/ML-FSM` — Freezing String Method TS search (ASE + ML potentials)
- `/home/ali/Documents/Codes/prism_pruner` — ensemble similarity pruning (numpy/SVD)

---

## 1. openconf

**One-line: emulate the single `generate_conformers(mol_or_smiles, preset="…")` front door — one call, string presets for the common cases, a fat config dataclass as the escape hatch, auto-detection fills the rest.**

### Package layout
Flat package `openconf/` (13 top-level modules, ~6.6k LOC) with **one** subpackage
`propose/` holding the complex MC engine (`hybrid.py` alone is 1273 LOC). Data (torsion
library JSON) sits in `openconf/data/`. A newcomer opening `openconf/` sees a small,
labelled set of files: `api.py`, `config.py`, `perceive.py`, `relax.py`, `dedupe.py`,
`io.py`, `torsionlib.py`, `exceptions.py`. The 1000-LOC engine is tucked one level down.

### Public API surface
Tiny and obvious. `__init__.py` re-exports a curated surface; the real entry points are
**two functions**:

```python
from openconf import generate_conformers, ConformerConfig
ensemble = generate_conformers("CCCCc1ccccc1", preset="docking")   # SMILES or Chem.Mol
ensemble = generate_conformers(mol, config=ConformerConfig(max_out=100))
ensemble.to_sdf("out.sdf")
```

- `generate_conformers(mol, method="hybrid", config=None, preset=None, …)`
- `generate_conformers_from_pose(mol, constrained_atoms, …)` (FEP/analogue variant)
- Result object `ConformerEnsemble` (a dataclass) carries its own verbs:
  `.to_sdf/.to_xyz/.from_sdf`, `.boltzmann_weights()`, `.rmsd_to()`, `.pairwise_rmsd()`,
  `.summary()`, `.n_conformers`, `.energies`.
- **Seven named string presets** (`"rapid" "ensemble" "spectroscopic" "docking"
  "analogue" "macrocycle" "transition_metal"`) — the preset string is the front door;
  `preset_config(name)` returns the dataclass if you want to tweak one field.

### README / onboarding
Model of the form. Order: tagline (one italic sentence) → install → **Quick Start**
(SMILES → ensemble → save, ~10 lines) → named presets → per-use-case sections each with
a one-liner, a `<details>` full-config equivalent, and a **wall-clock benchmark table** →
a Configuration Quick-Reference matrix → "How It Works" (5 numbered stages) → benchmarks →
API reference → dependencies. A newcomer can run from the README alone. Deep parameter
science is pushed to a separate `SCIENCE.md`; agent/dev conventions to `AGENTS.md`.

### Dependency footprint
Extremely lean core: `numpy`, `rdkit`, `prism-pruner`. Dev tooling in a separate
`[dependency-groups] dev`. Nothing heavy in the base install.

### How complexity is hidden
The boundary is `api.py` (the thin front) vs `propose/hybrid.py` (the engine). One
`generate_conformers` call internally: dispatches low-flex vs full-hybrid path by rotor
count, **auto-detects metals** and grafts on a TM move budget, **auto-computes seed
count** from topology, filters stereochemistry. The 40-field `ConformerConfig`
(`config.py`, fully documented + `__post_init__` validation) exists but you never build
it — presets cover the cases and auto-tuning handles the rest. "One more capability = one
preset row or one auto-detected flag", never a new entry point.

### Docstring / comment style
`ruff` pydocstyle **google** convention, enforced. Module docstrings are one imperative
line. Every public function/method gets full `Args:/Returns:/Raises:/Examples:` with
`>>> … # doctest: +SKIP` snippets. Comments are why-only and sparse (e.g.
`# GetBestRMS aligns prbMol in place — work on a copy`).

### Test layout
`tests/` flat: `test_basic.py`, `test_constrained.py`, `test_low_mode.py`,
`test_macrocycles.py`, plus `tests/data/*.xyz` fixtures. Each test imports from the
top-level `openconf` surface and uses tiny SMILES — reads as an executable example.
`pytest --doctest-modules` runs the docstring examples too (`testpaths = ["tests", "openconf"]`).

---

## 2. ML-FSM

**One-line: emulate the calculator-agnostic plug-in boundary (ASE) + a single canonical reference driver (`examples/fsm_example.py` + a Colab) so heavy backends stay out of the deps and out of the core.**

### Package layout
src-layout `src/mlfsm/` with **7 focused modules** (~2.3k LOC): `cos.py` (the
`FreezingString` driver), `opt.py`, `coords.py`, `interp.py`, `geom.py`, `output.py`,
`utils.py`, plus `py.typed`. Notably `__init__.py` is **just the version string** — no
re-exports. Users import from submodules.

### Public API surface
More of a toolkit + driver than a one-liner. The driver class:

```python
from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer
string = FreezingString(reactant, product, nnodes_min=18, interp_method="ric")
opt = CartesianOptimizer(calc, "L-BFGS-B", maxiter, maxls, dmax)
while string.growing:
    string.grow(); string.optimize(opt); string.write(outdir)
```

Interpolation/coordinate schemes are selected by a **string** (`interp_method="ric"|"lst"|"cart"`),
not by importing different classes. Endpoints are ASE `Atoms`. The onboarding entry is not
the API but `examples/fsm_example.py` — a complete argparse driver that wires load →
calculator → FSM loop → output, plus a Colab notebook.

### README / onboarding
Badge row (incl. an **Open-in-Colab** badge) → one-sentence description → install (PyPI +
from-source) → an immediately-runnable smoke command
(`python examples/fsm_example.py data/…/06_diels_alder/ --calculator emt`) → tutorials
(Colab + the example script) → citations → a **third-party license table** → credits. The
runnable EMT command is the fastest possible "it works" for a newcomer.

### Dependency footprint
Moderate but honest: `numpy`, `ase`, `scipy`, `geometric`, `networkx`. Crucially the heavy
ML potentials (AIMNet2, MACE, UMA, xTB, QChem) are **not dependencies** — the user installs
their chosen backend and the example script **lazily imports only the selected calculator**
inside an `if calculator == …:` switch.

### How complexity is hidden
The **ASE `Calculator` interface is the boundary**: FSM is potential-agnostic, so any
NNP/QM engine plugs in without touching the core. RIC/LST/Cartesian interpolation and the
back-transformation machinery hide behind the `interp_method=` string and the optimizer
class choice.

### Docstring / comment style
`ruff` pydocstyle **numpy** convention. Very thorough class docstrings — `FreezingString`
documents every constructor `Parameters` **and** every public `Attributes` field. Sphinx
autodoc drives a readthedocs site (`docs/source/api.rst` is just `.. automodule::` blocks,
so the docstrings *are* the docs).

### Test layout
Per-topic files: `test_cos.py`, `test_coords.py`, `test_interp.py`, `test_opt.py`,
`test_stepsize.py`, `test_cartesian.py`, `test_script.py`, `test_utils.py`, plus a
`test_smoke.py` (import + `__version__`) and a dedicated `tests/regression/` harness
(fingerprint + compare). `examples/data/NN_reaction/` holds per-reaction `initial.xyz`/
`ts.xyz`/`chg`/`mult` fixtures that double as tutorial inputs.

---

## 3. prism_pruner

**One-line: emulate "numpy in, (pruned array + boolean mask) out" — plain functions with sensible defaults and a chain-all `prune(...)` wrapper; the config dataclasses stay internal plumbing the user never touches.**

### Package layout
Flat `prism_pruner/` (~12 modules, ~2.6k LOC). `pruner.py` (1003 LOC) is the engine;
math/support in `algebra.py`, `rmsd.py`, `torsion_module.py`, `graph_manipulations.py`,
`periodic_table.py`; `__main__.py` is the CLI. `__init__.py` is just a docstring (no
re-exports). Typed array aliases live in a tiny `typing.py`.

### Public API surface
The most "rdkit-style": plain functions over arrays, layered from a one-call wrapper down
to the individual metrics.

```python
from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.pruner import prune
ensemble = ConformerEnsemble.from_xyz("ensemble.xyz")
pruned, mask = prune(ensemble.coords, ensemble.atoms, rot_corr_rmsd_pruning=False, debugfunction=print)
# pruned == ensemble.coords[mask]
```

- `prune(structures, atoms, moi_pruning=True, rmsd_pruning=True, rot_corr_rmsd_pruning=False, …)`
  chains up to three metrics; three booleans toggle the stages.
- Individual metrics exposed too: `prune_by_moment_of_inertia`, `prune_by_rmsd`,
  `prune_by_rmsd_rot_corr` — every one returns `(pruned_coords, boolean_mask)`.
- `ConformerEnsemble` dataclass (`coords/atoms/energies`) with `.from_xyz/.to_xyz`.
- `debugfunction=`/`logfunction=` callables are the observability hook (default `print`).
- A one-command CLI via `__main__.py` (`prism_pruner input.xyz` → `input_pruned.xyz`).

### README / onboarding
Logo + badge row → one-paragraph description of the three metrics → install → a Usage code
block that **shows the shapes inline as comments** (`ensemble.coords.shape # (1086, 136, 3)`
→ `pruned.shape # (387, 136, 3)`) — a compelling show-don't-tell → CLI usage → credits.
Terse but sufficient for the core task; deeper cases point to `examples/`.

### Dependency footprint
Lean: `networkx`, `numpy`, `scipy`, `tqdm`. Nothing else in the core.

### How complexity is hidden
The user calls `prune(...)`; the cached iterative divide-and-conquer O(N²) pruning, the
batched-SVD Kabsch vectorization, the energy-window shortcut, and the timeout handling are
all internal. The `PrunerConfig`/`RMSDPrunerConfig`/`MOIPrunerConfig` dataclasses are
**internal plumbing the public functions build for you** — never in the signature you call.
Sensible defaults everywhere (`max_rmsd=0.25`, auto `max_dev=2*max_rmsd`, auto energy window).

### Docstring / comment style
`ruff` pydocstyle **numpy**, enforced (even `mypy --strict`). Public docstrings are a terse
one-line summary + a short paragraph. The exception that proves the rule: comments are
**dense inside the hot algorithm** — the batched-SVD RMSD formula is derived in a block
comment, and non-obvious micro-optimisations are justified inline
(`# apparently much faster than numpy array operations for such a small array`). Heavy
commenting is reserved for genuinely non-obvious math, not narration.

### Test layout
A single `tests/test_suite.py` + `conftest.py` + `*.xyz` fixtures. Each test is a tiny
readable example: `from_xyz` → `prune*` → assert the surviving count. Identical-vs-different
structure pairs make the semantics obvious at a glance.

---

## Synthesis — what makes these onboardable (the pattern list)

The 8 things all three do that rxembed should match. Where they differ, it's noted.

1. **A tiny public surface with one obvious entry point.** 1–4 functions or one driver
   class, and a result object. openconf = `generate_conformers`; prism = `prune`; ML-FSM =
   `FreezingString`. The 1000-LOC engine is exactly one module hidden behind a thin `api`.
   *rxembed already does this well* — `embed`/`minimize`/`metal`/`nci_modes` re-exported
   from `__init__`, chainable `Ensemble`/`EnsembleSet`. Keep it that tight.

2. **String presets as the front door, a config dataclass as the escape hatch.** openconf's
   `preset="docking"` is the killer pattern: 90% of users pass a string; the fat validated
   `ConformerConfig` is there only when you need it; auto-detection (metal → TM budget,
   topology → seed count) covers the rest. This is literally rxembed's own stated ethos
   ("one more argument, not a new code path"). **rxembed has no presets today — adopt them**
   (e.g. `rx.embed(src, preset="ts")` / `"metal"` / `"nci"`). prism shows the lighter
   alternative: plain kwargs with good defaults and no user-facing config object at all.

3. **A result object that carries its own verbs and round-trips.** `ConformerEnsemble` and
   prism's ensemble own `.to_xyz/.from_xyz`, `.summary`, `.boltzmann_weights`, `.rmsd_to`.
   The user never reaches into internals for I/O or analysis. *rxembed's `Ensemble` already
   matches this;* make sure `dump`/`score`/`representatives` stay discoverable on it.

4. **Lean core deps; heavy stuff optional, lazily imported, or plugged via an abstraction.**
   Base installs are numpy+rdkit or numpy+scipy+networkx. ML-FSM keeps every ML backend out
   of deps and lazily imports the chosen calculator behind ASE. *rxembed's extras split
   (`mc`/`nci`/`viz`/`ase`/`racerts`) already fits;* mirror ML-FSM's lazy-import + "fail
   loud if the backend is missing" discipline for xtb/g-xTB.

5. **The README formula.** tagline → install → runnable quick-start (≤10 lines) →
   presets/use-cases (with benchmark tables and `<details>` full-config) → "How It Works"
   numbered stages → benchmarks → API reference → deps. Deep science and dev/agent
   conventions live in **separate files** (`SCIENCE.md`, `AGENTS.md`), not the README.
   Give a newcomer a copy-pasteable one-liner in the first screen. **rxembed's README should
   lead with `rx.embed(...).mc().prune()` and one benchmark, not point straight to notebooks.**

6. **A single canonical, runnable example.** openconf's Quick Start, ML-FSM's
   `examples/fsm_example.py` + Colab (runnable with the trivial EMT backend), prism's
   README block with inline shapes. One command that proves "it works" in seconds. *rxembed
   has the notebook tour;* add one dependency-free smoke example (bare SMILES, no xtb) as
   the first thing a newcomer runs.

7. **Docstrings are the docs; comments are why-only.** All three enforce `ruff` pydocstyle
   (google for openconf, numpy for the other two), one-line imperative module summaries,
   full Args/Returns/Raises on public API, doctests where cheap (`--doctest-modules`).
   Comments explain load-bearing *why*, not narration — with prism's exception that
   genuinely non-obvious math (its SVD derivation) earns a dense block comment. *rxembed's
   CLAUDE.md already prescribes exactly this;* the siblings show the bar to hold.

8. **Tests are executable examples, and the scaffolding is identical across repos.** Tests
   import from the public surface, use tiny inputs, keep fixtures in `tests/data`, and run
   doctests. Every repo shares the same toolchain (ruff with the same lint block, ty/mypy,
   pytest `--doctest-modules`, pre-commit, the same badge row, cookiecutter/copier template).
   That cross-repo consistency is itself onboarding — learn one, you know all three. *rxembed
   already shares the copier/uv/ruff/ty toolchain;* the gap is a `test_*` file a newcomer can
   read as "how do I call this" for each capability (organic, TS, metal, NCI).

### Honest divergences among the three (so rxembed picks deliberately)
- **`__init__` re-exports:** only openconf curates a public surface in `__init__`. ML-FSM's
  `__init__` is just the version; prism's is just a docstring — both make you import from
  submodules. rxembed follows openconf here (good — it reads as more "rdkit-style").
- **One-call vs toolkit:** openconf and prism are true one-call APIs; ML-FSM is a
  toolkit + reference driver. rxembed's chain (`embed().mc().prune()`) is a fourth style —
  fluent pipeline — and is fine, but it raises the bar on making each stage self-documenting.
- **Config object:** openconf = big validated dataclass + presets; prism = no user-facing
  config at all (just kwargs); ML-FSM = constructor kwargs on the driver. For rxembed,
  openconf's preset+dataclass split is the closest fit to its "verbs + one struct
  (`Constraints`)" design.
- **Comment density:** openconf/ML-FSM keep comments sparse; prism comments its hot loops
  heavily. The rule they share: density tracks non-obviousness, not line count.

### rxembed-specific note on the embed kernel
The siblings are all **flat, single-level packages** (`openconf/x.py`, `mlfsm/x.py`,
`prism_pruner/x.py`). rxembed's kernel is nested three deep
(`rxembed/rdkit_embed/constraints/…`, `rxembed/rdkit_embed/embed/bounds.py`). Structurally
it is the outlier — the kernel would read more sibling-like if its constraints/embed layer
were shallower and each module's role were legible from the top-level listing, the way
opening `openconf/` immediately shows `api / config / perceive / relax / dedupe / io`.
</content>
