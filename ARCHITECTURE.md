# Architecture

`README.md` is the API. `AGENTS.md` is the development contract. This file owns the module boundaries.

## Flow

```text
Mol | Isomer
  +-- fix= / constrain= / template= --> constraints.resolve_core ---+
  +-- Isomer.cons --> metal_constraints ----------------------------+
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

The DG engine remains RDKit: `bounds.etkdg` creates `ETKDGv3`, and `seed_coordinates` supplies only an edited
native bounds matrix through `SetBoundsMat`. RDKit currently constructs signed `ChiralSet` records for
tetrahedral atoms and atropisomeric bonds, but not SP/TB/OH tags, and Python cannot supply an extra chiral set.
Metal hand and haptic winding are therefore checked on raw DG seeds before UFF. A future RDKit chiral-set input
can replace that seed filter; it does not change metal enumeration, canonical slots or constraint compilation.

`metal_core.py` owns graph surgery, perception primitives and the immutable `MetalState`: one real metal,
polyhedron, slot-ordered donors or haptic faces, and hand. `metal_stereo.py` canonicalizes site identity and
reads metal or haptic hands. `metal_slots.py` produces distinct, reachable donor-to-polyhedron slot assignments.
`metal_enumeration.py` combines ligand, haptic and per-centre choices into the `Isomer` and `IsomerSet` types in
`metal_isomer.py`, without building numerical fields for every candidate.

`Isomer.cons` asks the plain compiler in `metal_constraints.py` for a fresh `Constraints`. It compiles only that
selected isomer's active centre(s) from one private source-Mol snapshot, then composes their coordination field
with the same `Constraints` carrying a fixed TS core, NCI contacts or a retained spectator shape. The snapshot
keeps `lengths='input'` independent of later edits to the public Mol; transient centroid indices still follow
the public Mol's current atom count. A retained input geometry is already one selected state, so `from_geometry`
records its measured field immediately. Haptic centroids exist transiently while deriving a state and while
running DG/UFF; stored molecules and `MetalState` contain real atom indices only.

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
| `metal_core.py` | metal state, graph surgery and shape perception primitives |
| `metal_isomer.py`, `metal_enumeration.py` | selected isomers, collections and candidate enumeration |
| `metal_polyhedron.py`, `metal_stereo.py`, `metal_slots.py` | shape tables, canonical site identity and reachable slot assignments |
| `metal_constraints.py`, `metal_distance.py`, `metal_donor_orient.py` | compile selected states into coordination constraints |
| `metal_perceive.py` | geometry QA rulers used by perception and validation |
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
