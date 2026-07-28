# rxembed: Fast, flexible molecular embedding for reactive chemistry.

**rxembed** is a unified constrained / templated conformer-embedding toolkit. One composable chain embeds
conformers for anything — a bare SMILES, a non-covalent complex with discovered binding modes, a frozen
transition state, a known TS transferred onto a fresh molecule, a metal centre across its coordination
isomers — then searches, deduplicates, and ranks them with real (xTB / g-xTB) energies. **Each capability is
one more argument on the same call, not a new code path**, and every embed can be validated against a
physical geometry gate (broken conjugation, bad H positions, clashes, a moved reacting core).

[![PyPI Downloads](https://static.pepy.tech/badge/rxembed)](https://pepy.tech/projects/rxembed)
[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

## Installation

Install from source with [uv](https://docs.astral.sh/uv/) (recommended — it resolves the git-pinned
openconf automatically):

```bash
git clone https://github.com/aligfellow/rxembed.git
cd rxembed
uv sync            # or: just setup   (also installs pre-commit)
```

For real energies, put an `xtb` (or g-xTB) binary on `$XTB_EXE`. Everything else — embed, search, prune —
is pure-Python.


## Quickstart

rxembed is a **library**, driven from Python or a notebook — everything is the same chain:

```python
import rxembed as rx

ens = rx.embed("CCO").mc().prune()               # embed -> Monte-Carlo search -> dedup
ens.representatives()                            # one conformer per distinct mode
ens.score("gxtb").lowest(3).optimize("gxtb")     # real energies, then optimise the best few
```

`rx.embed(source, ...)` takes a SMILES, an `.xyz` path, or an RDKit `Mol`, and returns an **`Ensemble`** —
or an **`EnsembleSet`** of candidates to `.select` from when the input is inherently several poses (a metal's
coordination isomers, discovered NCI binding modes). Every capability is a keyword on that one call:

| Want | Call |
|---|---|
| free / flexible | `rx.embed("CCO")` |
| a soft distance / angle / π-stack | `rx.embed(smi, constrain={(i, j): (2.6, 3.0)})` |
| NCI complex, modes discovered | `rx.embed("A.B", contacts="auto")` → one candidate per grip |
| a frozen TS core (from `.xyz`) | `rx.embed("ts.xyz", fix=reacting)` — held to 0.000 Å |
| a TS **from SMILES** | `rx.embed(smi, fix={(i, j): d, (i, j, k): θ})` — exact numbers, verify with `.measure()` |
| a known TS onto a fresh molecule | `rx.embed(smi, template=(reference, {target_i: ref_i}))` |
| relax a structure **toward** a core | `rx.minimize("mol.xyz", fix={(i, j): 2.0, (i, j, k): 178})` |
| metal coordination isomers | `rx.metal("…[Pd]…", "square_planar")` → cis / trans, mer / fac … |
| real energies | `ens.score("gxtb")` / `ens.optimize("gxtb", level="loose")` |

**Three constraint verbs** — all **index-driven** (0-based atom indices in xyz/graph order; resolve any
SMARTS yourself first, two RDKit lines): **`fix`** (rigid — the atoms *will* have this geometry: own
coords / explicit coords / exact numbers), **`constrain`** (soft — bias the seed, a real energy may win),
**`template`** (reference sugar for a coords-`fix`). The same verbs drive `rx.minimize`, the search-free
relax toward the targets.

**Mutation contract:** `mc` / `minimize` / `prune` build in place and chain; `lowest` / `representatives` /
`align` return a *new* ensemble; looking never mutates. `rx.set_verbose("INFO")` narrates every stage.

## How it flows

Three separable layers — **embedding is the core**; the other two lean on existing tools (openconf, prism).

```
  ┌─ LAYER 1 · EMBED (the core — this is what rxembed is) ───────────────────────────┐
  │  source → Mol → Constraints → edit RDKit's bounds matrix → relaxing triangle       │
  │  smoothing → ETKDG distance-geometry embed → frozen-core Kabsch graft / metal       │
  │  carbon-surrogate restore → frozen-aware UFF/MMFF cleanup → geometry gate           │
  │  free · constrained · frozen TS · templated TS · metal isomers · NCI complexes      │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │ a good embed is the seed                    │ or: many embeds ARE the conformers
        ▼                                             ▼
  ┌─ LAYER 2 · CONFORMERS ───────────────┐   ┌─ LAYER 3 · PRUNE / FILTER ───────────────┐
  │  seed an openconf MC search from the  │   │  rmsd / moi / descriptor (prism) ·       │
  │  embed(s), or embed many seeds        │   │  energy-aware dedup · binding-mode        │
  │  (bounds-biased ETKDG pool)           │   │  clustering · the geometry gate           │
  └───────────────────────────────────────┘   └───────────────────────────────────────────┘
```

Guiding principles: **edit, don't replace** (ETKDG's knowledge-based bounds are *edited*, never rebuilt);
**bias the seed, let energy decide** (constraints bias the start, a real energy chooses); **one struct in,
one struct out** (`Constraints` is the single thing builders fill and stages read); **fail loud** (a
calculator that yields no energy raises — never a silent FF fallback).

**How a constraint travels.** Every verb (`fix`/`constrain`/`template`/`metal`/`contacts`) resolves into
*one* `Constraints` struct of distance / angle / plane windows + a frozen set. That single struct then drives
three stages from different views of itself: it **edits the bounds matrix** (the embed bias), it is **held by
the restrained-UFF relax** (`minimize`), and its named atoms are **pose-frozen during the `mc` search** (so a
held contact is never broken). Bias the seed, hold it through relax and search — one struct, three readers.

## Approximations

rxembed is honest about where it trades exactness for a robust, stackable embed:

- **Metal = carbon surrogate.** For the force field a metal becomes a bond-stripped carbon (UFF-typeable);
  the coordination sphere is held by soft **shape constraints**, not real M–L bonds. M–donor distances are
  the **covalent-sum bond length**, with a cap for large soft donors (P/S — their dative bond runs shorter
  than the covalent radius implies); a **halide** donor keeps the covalent sum. A conjugated N/O donor's
  **donation angle** is held (~120°) so its rigid plane can't fold into the metal during search.
- **Exact frozen cores are grafted back.** A frozen TS core is embedded via the surrogate, then restored by
  **Kabsch superposition to 0.000 Å** — the exact input geometry, not the FF's version of it.
- **N good geometries, not N attempts.** For a metal, `embed(n=N)` re-embeds fresh seeds until N conformers
  pass the geometry gate (a bad seed the relax tears is a *re-embed*, not a kept result), then falls back
  gracefully if an arrangement is inherently strained.
- **Solvent for g-xTB is a thermodynamic cycle.** g-xTB has no implicit-solvent model, so a solvated g-xTB
  energy is `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]` — a real solvated energy, never a silent gas-phase one.
- **The search backend is openconf.** `mc()` pose-freezes the held atoms and runs openconf rotor moves around
  the bounds-biased seed; for a metal complex pass `mc(preset="transition_metal")` for metal-aware sampling.

## The geometry gate

A perfect frozen core can coexist with a chemically wrong periphery — a puckered ring, a twisted amide
(broken conjugation), a bad H position, two atoms on top of each other. `rx.geometry.check` is the physical
acceptance test, usable on any conformer (pure RDKit + NumPy, no optional deps):

```python
rep = rx.geometry.check(mol, conf_id, frozen=reacting_core, reference="ts.xyz")
rep.assert_ok()      # raises with a readable summary if any check fails
```

It is **TS-aware** (name the reacting core in `frozen=` and forming/breaking bonds aren't misread as clashes)
and **metal-aware** (a metal's dative distances aren't vdW clashes).

## Modes and the ensemble map

An ensemble is embedded in a **latent** per conformer — its dihedral angles, plus (when present) an
inter-fragment NCI-contact signature and metal-coordination features. That one latent drives both the
dedup and the picture:

- `representatives()` returns **one lowest-energy conformer per mode** — the distinct-shapes summary. A
  "mode" is whatever the latent separates: a **conformer family** (organic), a **binding-mode / contact
  pattern** (NCI), or a **ligand arrangement** (metal).
- `cluster()` labels each conformer by mode (HDBSCAN on the latent; `-1` = rare / noise).
- `landscape(method="pca"|"tsne", color="cluster"|"energy")` projects that latent to **2D** — so *mode 1, 2,
  3…* are the cluster families (distinct conformers or binding grips), coloured by cluster or by energy.
  Pruned-away duplicates are drawn faded, so you see exactly what `prune()` collapsed.

## Package structure

Each module has one job; the layout mirrors the import graph.

```
rxembed/
  pipeline.py       # user-facing: embed() + Ensemble / EnsembleSet chain
  embed/            # LAYER 1 — the core
    bounds.py       #   THE bounds matrix: edit constrained pairs -> relaxing triangle smoothing -> ETKDG
    dispatch.py     #   source routing: xyz/metal perception, Kabsch graft, isomer / template / auto-NCI
    mc.py           #   LAYER 2 — openconf Monte-Carlo search (optional)
  constraints/      # base.py (Constraints), nci.py (binding modes), metal.py (polyhedra), builders.py
  dedup/            # LAYER 3 — select.py (rmsd/moi/descriptor via prism + energy-aware), features.py
  refine/           # xtb.py / calculator.py (ff · gfnff · gfn2 · gxtb tiers), ff.py (frozen-aware UFF/MMFF)
  geometry.py       # the physical gate (TS- and metal-aware)
  stereo.py         # chirality fingerprints + auto-preserve
```

## Examples

The [`examples/`](examples/) tour is the proof of breadth — 8 notebooks, simple → complex, each executed with
real backends and gated by `geometry.check`: basics, constraints (incl. π-stacks), NCI binding modes (H-bond /
halogen / chalcogen / salt bridge), reaction TSs, templated TSs onto fresh molecules, the isothiourea
organocatalysis backbone swap, the metal space, and the g-xTB energy ladder. See
[`examples/README.md`](examples/README.md).

## Requirements

Core: **RDKit**, **NumPy**, **prism_pruner**, **openconf**, **xyzgraph**, **scikit-learn**. An **`xtb`
binary** on `$XTB_EXE` is needed only for real energies (`score`/`optimize` — g-xTB, or standard Grimme xtb
for GFN-FF); embedding / search / prune are pure-Python. The `viz` extra (matplotlib / seaborn / xyzrender)
enables `landscape()`. Each optional capability degrades gracefully if its backend is absent.

> **openconf from git.** The `mc()` search backend uses openconf's transition-metal support, which is on
> upstream `main` but not yet on a PyPI release, so `pyproject.toml` pins it via `[tool.uv.sources]` to git.
> `uv sync` resolves and locks it automatically — no manual step. To **co-develop openconf**, run
> `just setup-openconf-dev` (clones it as a sibling and installs it editable, overriding the git pin).

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
git clone https://github.com/aligfellow/rxembed.git
cd rxembed
just setup   # install dev dependencies
just check   # lint + type-check + tests
```

| Command | Description |
|---|---|
| `just check` | Run lint + type-check + tests |
| `just lint` | Format and lint with ruff |
| `just type` | Type-check with ty |
| `just test` | Run pytest with coverage |
| `just fix` | Auto-fix lint issues |
| `just build` | Build distribution |
| `just setup` | Install deps + pre-commit (openconf pulled from git automatically) |
| `just setup-openconf-dev` | Clone + editable-install openconf as a sibling (to co-develop it) |

The development discipline — every non-trivial change runs `assess → plan → implement → adversarial review →
regress` with the test suite as the gate and the `examples/` notebooks as the proof of breadth — is in
[`CLAUDE.md`](CLAUDE.md).

### CI

GitHub Actions runs lint, type-check, and tests on every push to `main` and every PR targeting `main`.
Coverage is uploaded to [Codecov](https://codecov.io).

## License

[MIT](LICENSE)

## References

**Software rxembed builds on**

- [RDKit](https://github.com/rdkit/rdkit) — distance-geometry embedding (ETKDG), the bounds matrix, UFF / MMFF cleanup, all cheminformatics.
- [openconf](https://github.com/rowansci/openconf) (rowansci) — the Monte-Carlo torsional conformer search behind `mc()`.
- [prism_pruner](https://pypi.org/project/prism-pruner/) (N. Tampellini) — the RMSD / moment-of-inertia / descriptor dedup behind `prune()`.
- [xyzgraph](https://github.com/aligfellow/xyzgraph) — bond perception for `.xyz` metal / TS inputs.
- [xyzrender](https://github.com/aligfellow/xyzrender) — the publication-quality structure rendering used in the notebooks.
- [scikit-learn](https://scikit-learn.org) — HDBSCAN clustering for `cluster()` / `landscape()`.
- [xtb](https://github.com/grimme-lab/xtb) (Grimme group) — the semiempirical engine for GFN-FF / GFN2; g-xTB for the top energy tier.

**Methods**

- ETKDG — S. Riniker, G. A. Landrum, *J. Chem. Inf. Model.* **2015**, *55*, 2562.
- GFN2-xTB — C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **2019**, *15*, 1652.
- GFN-FF — S. Spicher, S. Grimme, *Angew. Chem. Int. Ed.* **2020**, *59*, 15665.
- UFF — A. K. Rappé *et al.*, *J. Am. Chem. Soc.* **1992**, *114*, 10024.
- MMFF94 — T. A. Halgren, *J. Comput. Chem.* **1996**, *17*, 490.
- Kabsch superposition — W. Kabsch, *Acta Crystallogr. A* **1976**, *32*, 922.
- HDBSCAN — R. J. G. B. Campello, D. Moulavi, J. Sander, *PAKDD* **2013**, 160.

## Acknowledgements

Generated from [aligfellow/python-template](https://github.com/aligfellow/python-template).

<details>
<summary>Updating from the template</summary>

If this project was created with [copier](https://copier.readthedocs.io/), you can pull in upstream template improvements:

```bash
# Run from the project root
copier update --trust
```

This will:

1. Fetch the latest version of the template
2. Re-ask any questions whose defaults have changed
3. Re-render the templated files with your existing answers
4. Apply the changes as a diff — your project-specific edits are preserved via a three-way merge

If there are conflicts (e.g. you modified the `justfile` and so did the template), copier will leave standard merge conflict markers (`<<<<<<<` / `>>>>>>>`) for you to resolve manually.

The `--trust` flag is required because the template defines tasks (used for `git init` on first copy). The tasks don't run during update, but copier requires trust for any template that declares them.

Requires that the project was originally created with `copier copy`, not the plain GitHub "Use this template" button.

</details>
