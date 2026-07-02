# rxembed: Fast, flexible molecular embedding for reactive chemistry.

**rxembed** is a unified constrained / templated conformer-embedding toolkit. One composable chain embeds
conformers for anything — a bare SMILES, a non-covalent complex with discovered binding modes, a frozen
transition state, a known TS transferred onto a fresh molecule, a metal centre across its coordination
isomers — then searches, deduplicates, and ranks them with real (xTB / g-xTB) energies. **Each capability is
one more argument on the same call, not a new code path**, and every embed can be validated against a
physical geometry gate (broken conjugation, bad H positions, clashes, a moved reacting core).

[![PyPI Downloads](https://static.pepy.tech/badge/rxembed)](https://pepy.tech/projects/rxembed)
[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/blob/main/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

## Installation

> **Note:** The PyPI badge and install commands below require that you first [publish your package to PyPI](https://docs.astral.sh/uv/guides/package/#publishing-your-package) (e.g. `uv build && twine upload dist/*`).

From PyPI:

```bash
pip install rxembed
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add rxembed
```

From source:

```bash
git clone https://github.com/aligfellow/rxembed.git
cd rxembed
pip install .
# or for an editable install:
pip install -e .
```


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
| a distance / angle / π-stack | `rx.embed(smi, distances={(i, j): (2.6, 3.0)})` (index **or** SMARTS keys) |
| NCI complex, modes discovered | `rx.embed("A.B", contacts="auto")` → one candidate per grip |
| a frozen TS core (from `.xyz`) | `rx.embed("ts.xyz", freeze=reacting)` — held to 0.000 Å |
| a TS **from SMILES** | `rx.embed(smi, distances={reacting-atom windows})` |
| a known TS onto a fresh molecule | `rx.embed(smi, template="ts.xyz", match=core_SMARTS)` |
| metal coordination isomers | `rx.metal("…[Pd]…", "square_planar")` → cis / trans, mer / fac … |
| real energies | `ens.score("gxtb")` / `ens.optimize("gxtb", level="loose")` |

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

Core: **RDKit**, **NumPy**, **prism_pruner**. Optional (each capability degrades gracefully if absent):
**openconf** (`mc` search), **xyzgraph** (metal/TS `.xyz` bond perception), an **`xtb` binary** on `$XTB_EXE`
(`score`/`optimize` — g-xTB or standard Grimme xtb for GFN-FF), and the `viz` extra (matplotlib / seaborn /
scikit-learn) for `landscape()`.

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
| `just setup` | Install all dev dependencies |

### CI

GitHub Actions runs lint, type-check, and tests on every push to `main` and every PR targeting `main`. Coverage is uploaded to [Codecov](https://codecov.io).

For private repos, add your `CODECOV_TOKEN` as a repository secret under **Settings > Secrets and variables > Actions**. Public repos work without the token.

### Changing the license

This project defaults to the [MIT License](LICENSE). To change it, replace the `LICENSE` file and update the classifier in `pyproject.toml` (e.g. `"License :: OSI Approved :: Apache Software License"`).

## License

[MIT](LICENSE)

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
