# Architecture

How rxembed is assembled, in one page. `AGENTS.md` is how we change it.

## The flow

Every capability (a bare SMILES, a frozen TS core, an NCI grip, a coordination isomer) ends up as one
`Constraints` struct, and that struct drives both halves of the engine.

```
  fix= / constrain= / template=        enumerate_isomers(…) / an Isomer
              │                                        │
    constraints.resolve_core                  metal_coordination.py
              │                                └ metal_polyhedron · metal_distance
              │                                  metal_donor_orient · metal_sphere
              └───────────────┬────────────────────────┘
                              ▼
                     ┌────────────────┐   distances · angles · planes · coplanar · frozen · contacts
                     │  Constraints   │   + the metal fields (metals · pulls · floors · dg_floors ·
                     └────────────────┘     shapes · phantoms · spheres · haptic)
                              ▼
                     ┌────────────────┐   11 classes over those 14 fields, each holding both writers
                     │  mechanisms    │   co-located so a field cannot gain one and not the other
                     └────────────────┘
                        ╱                 ╲
               dg_windows                  ff_terms
                    ▼                         ▼
            ┌───────────────┐        ┌───────────────┐
            │   bounds.py   │        │   relax.py    │
            │ edit RDKit's  │        │ restrained UFF│
            │ bounds matrix │        │ + the Li metal│
            │ → smooth →    │        │ surrogate     │
            │ ETKDG embed   │        │               │
            └───────────────┘        └───────────────┘
                        ╲                 ╱
                         ▼               ▼
                     ┌────────────────┐   {mol, ids, cons} · .minimize() · .xyz() · .dump() · [i]
                     │  Conformers    │   `bonding_ok` is the arbiter every relax stage accepts on
                     └────────────────┘
```

`embed.py` is the front door that drives that: resolve → `bounds.embed` → `Conformers`, whose `.minimize()`
runs `relax.restrained_uff` up a stiffness ladder and never hands back a torn geometry wearing a plausible
energy.

Arrows point one way: `metal_*` produces `Constraints` and the engine consumes them. No mechanism knows
what a polyhedron is, and no coordination builder knows what a bounds matrix is. The same holds one level
up: `pipeline/` drives the core, and no core module imports `pipeline`, enforced by `tests/test_init.py`.

## The two tiers

> **Flat is core. The single directory needs extras.**

Visible from `ls src/rxembed/`: the root, `__init__.py` plus the 15 modules below, is `numpy + rdkit` and
nothing else, and the one subdirectory is the optional half. `metal/` was flattened to `metal_*.py` for
exactly this reason: a subdirectory cannot signal a tier when both tiers are subdirectories.

### Core: `src/rxembed/`, the engine

| module | job |
|---|---|
| `constraints.py` | the `Constraints` struct + `resolve_core`, the `fix`/`constrain` resolver |
| `mechanisms.py` | each class holds its DG writer and its FF writer together; 11 cover the 14 fields |
| `bounds.py` | the DG driver: edit RDKit's knowledge-derived matrix, smooth, ETKDG |
| `relax.py` | the FF driver: `restrained_uff`, `ff_energies`, the `bonding_ok` arbiter |
| `embed.py` | the front door: `embed(spec, …) -> Conformers`, and the stiffness ladder |
| `stereo.py` | undefined stereocentres → the distinct species to embed |
| `utils.py` | coordinate math, small RDKit facts, `Violation`; no domain knowledge |
| `metal_isomers.py` | `Isomer` / `IsomerSet` / `enumerate_isomers`: the arrangements |
| `metal_polyhedron.py` | the `POLYHEDRA` records + the slot/parity chirality descriptor |
| `metal_coordination.py` | the builders: an `Isomer` → its polytope `Constraints` |
| `metal_core.py` | the surrogate (`surrogate_metal` / `restore_metal` / `connect_metal`) + sphere helpers |
| `metal_distance.py` | `ml_distance`, the fitted M–L model, + the anti-overbond floors |
| `metal_donor_orient.py` | the orientation holds the stripped M–donor bond took away |
| `metal_perceive.py` | the ruler behind the QA gate: who coordinates, donor fold / overbond. Off the embed path |
| `metal_sphere.py` | the geometric sphere solver: a rare fallback, scipy, soft-guarded |

`metal_sphere.py` is the one core module that reaches past `numpy + rdkit`: its solver is a rarely-taken
fallback that guards its own scipy import and degrades with a message, which is why `sphere` is the single
core extra.

### Optional: `src/rxembed/pipeline/`, the batteries

| module | job | extra |
|---|---|---|
| `api.py` | the public verbs: `embed` · `metal` · `minimize` · `wrap`. `rx.metal` is `dispatch.enumerate_isomers` under its pipeline name, so it has no `def` of its own to grep for | via the others |
| `ensemble.py` | `Ensemble` / `EnsembleSet`: the chain | via the others |
| `dispatch.py` | source + spec → embedded conformers (metal / template / stereo / NCI routes) | via the others |
| `search.py` | openconf Monte-Carlo, `mc()` | `search` |
| `select.py` | dedup, prune, cluster, the latent | `select` |
| `calculators.py` | xtb / g-xTB / ASE, `score()` and `optimize()` | `score` |
| `metrics.py` | connectivity + coordination diffs, behind `.filter('connectivity')` | `perceive` |
| `stereo_check.py` | the chirality fingerprint + the preserve gate | `perceive` |
| `perceive.py` | `.xyz` / SMILES readers (reaches no further than `rxembed.utils`) | `perceive` |
| `nci.py` | the `KINDS` registry + binding modes | `nci` |
| `viz.py` | the 2D projection behind `landscape()`, kept-vs-pruned | `viz` |
| `geom_check.py` | the TS- and metal-aware geometry gate | none, see below |

`geom_check.py` is the one module here that needs no wheel: it is `numpy + rdkit`, and it sits in `pipeline/`
because it is a downstream QA consumer, reading the core's `metal_perceive` ruler and never the reverse. The
tier is about direction as much as dependencies.

Nothing in `pipeline/` imports its dependency at module top level. Each optional import is a plain
`try/except ImportError` at the point of use, raising a message that names the extra to `pip install`, so
`import rxembed.pipeline` works on a base install and only the path that needs a wheel asks for one
(enforced by `tests/pipeline/test_init.py`).

## The two `embed` verbs

```python
import rxembed as rx                     # core:     Mol | Isomer -> Conformers
rx.embed(mol, fix={(i, j): 2.0}).minimize()

import rxembed.pipeline as rx            # batteries: str | path | Mol -> Ensemble | EnsembleSet
rx.embed("cat.substrate", contacts="auto").mc().prune().score("gxtb")
```

Different functions, different signatures. The root verb does not change behaviour with which extras are
installed. `rxembed.pipeline` re-exports every core name, so it is a strict superset.

## Neighbours: what is not ours

```
  UPSTREAM (not ours)                rxembed                  DOWNSTREAM (optional)
  ───────────────────                ───────                  ─────────────────────
  xyzgraph  (metals, TS bonds)       Constraints              openconf      search
                                     mechanisms               prism_pruner  select
  RDKit MolFromXYZFile, SMILES  →    bounds | relax      →    xtb / g-xTB   score
  ───────────────────                Conformers               scikit-learn  cluster
  bond + charge perception                                    matplotlib    landscape
```

Perception is upstream. A source becomes an RDKit `Mol` before the engine sees it, in
`pipeline/perceive.py`, which reaches no further than `rxembed.utils` — the leaf of RDKit facts that holds no
domain knowledge. It takes exactly one name from there, `assign_stereo_from_3d`, because the door that writes
stereo from a geometry has to be the same door on both tiers: RDKit's 3D writer omits a dative bond from the
chirality basis that every reader counts, so two doors would disagree about which bonds a tag is a parity
over. `tests/test_init.py` pins that boundary, so the direction cannot quietly invert.
Bond-and-charge perception for metals and stretched TS bonds is xyzgraph's job. rxembed reads a graph, it
does not guess one.

Selection is downstream, and not locked to our embedder. `rxembed.pipeline.wrap(mol, energies=…,
minimized=True)` adopts conformers produced anywhere (a CREST or xtb run, a DFT scan, a crystal set) into an
`Ensemble` and gives them the same `prune` / `representatives` / `score` / `landscape` chain. Pass
`minimized=True` and the geometries are not touched.

## Where a new thing goes

| adding… | goes in |
|---|---|
| a new constraint kind | a `Constraints` field + a `Mechanism`, in the root |
| a new coordination shape | a `POLYHEDRA` row in `metal_polyhedron.py` |
| a new NCI contact type | a `KINDS` row in `pipeline/nci.py` |
| a new search backend | `pipeline/search.py` |
| a new calculator | `pipeline/calculators.py` |
| a new QA check | `pipeline/geom_check.py` |
| a new input format | `pipeline/perceive.py` |
| a new optional dependency | an extra in `pyproject.toml` + a guarded import at the point of use |

If a change needs a new directory, the abstraction is probably wrong.
