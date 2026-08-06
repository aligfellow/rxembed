# Architecture

`README.md` is the API. `AGENTS.md` is the development contract. This file owns the module boundaries.

## Flow

```text
Mol | Isomer
  +-- fix= / constrain= / template= --> constraints.resolve_core --+
  +-- Isomer.coordination() ---------------------------------------+
                                                                   v
                                                              Constraints
                                                                   |
                                                            MECHANISM_ORDER
                                                              /          \
                                                    DG windows            FF terms
                                                        |                    |
                                                     bounds.py            relax.py
                                                        \                    /
                                                         +--> Conformers <--+
```

`Constraints` is the sole constraint payload. `embed.py` also carries the `Isomer` identity and temporary
graft coordinates while it drives the sequence. `bounds.py` edits RDKit's bounds matrix, `relax.py` applies
the matching restrained-UFF terms, and the result is `Conformers`.

The normal API adapts strings, paths and external tools around that core:

```text
rxembed.embed(str | path | Mol | Isomer)
    -> pipeline dispatch -> core constraint / seed / relax seam
    -> Ensemble | EnsembleSet | list[EnsembleSet] -> search / select / score
```

## Tiers

The flat implementation modules are core. They import NumPy, RDKit and siblings with relative imports and
never import `pipeline`. `core.py` is their public engine facade.

`src/rxembed/pipeline/` owns input perception, orchestration and optional tools. It imports core absolutely.
Optional dependencies are imported at the point of use.

`__init__.py` is the user facade. It re-exports the core types and helpers, but its `embed` and `minimize` are
the workflow versions from `pipeline.api`. The facade contains no logic; implementation dependencies remain
one-way from pipeline to core.

The base distribution also installs `networkx` for the vendored xyz2mol perceiver. No core module imports it;
the embed engine remains NumPy and RDKit.

`read_xyz` prefers xyzgraph. A failed perceiver warns and falls back to the other one. Without xyzgraph, a
metal complex uses vendored xyz2mol and an organic molecule uses RDKit. A fallback may not drop or reorder
atoms.

## Public surfaces

```python
import rxembed as rx

rx.embed("cat.substrate", contacts="auto").mc().prune().score("gxtb")

from rxembed import core

core.embed(mol, fix={(i, j): 2.0}).minimize()  # Mol | Isomer -> Conformers
```

There is one normal `rxembed.embed`. `rxembed.core.embed` is the explicit engine seam for another library that
already owns the molecular graph. Optional backends are enabled only by calling their operation.

## Ownership

| owner | responsibility |
|---|---|
| `__init__.py`, `core.py` | workflow and engine facades; no implementation |
| `constraints.py`, `mechanisms.py` | constraint data and its DG/FF interpretation |
| `bounds.py`, `embed.py`, `relax.py` | seed, orchestrate and relax conformers |
| `metal_polyhedron.py`, `metal_isomers.py`, `metal_coordination.py` | shapes, arrangements and coordination constraints |
| `metal_core.py`, `metal_distance.py`, `metal_donor_orient.py`, `metal_perceive.py` | metal graph surgery, distances, donor geometry and QA rulers |
| `metal_smiles.py`, `stereo.py`, `utils.py` | string round trips, organic stereo and shared RDKit geometry facts |
| `pipeline/api.py`, `pipeline/dispatch.py`, `pipeline/ensemble.py` | public pipeline verbs, routing and the chainable result |
| `pipeline/perceive.py`, `pipeline/xyz2mol_*.py` | coordinate input and bond perception |
| `pipeline/{search,select,calculators,nci,viz,geom_check,metrics,stereo_check}.py` | optional or downstream capabilities |

## Extension points

| adding | location |
|---|---|
| constraint kind | `Constraints` field and a `Mechanism` |
| coordination shape | `POLYHEDRA` row |
| NCI contact kind | `pipeline.nci.KINDS` row |
| search backend | `pipeline/search.py` |
| calculator | `pipeline/calculators.py` |
| QA check | `pipeline/geom_check.py` |
| coordinate reader | `pipeline/perceive.py` |
| optional dependency | `pyproject.toml` extra and a guarded point-of-use import |

Do not add another package directory to mark a tier.
