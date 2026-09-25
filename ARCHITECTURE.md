# Architecture

`README.md` owns the public API. `AGENTS.md` owns development rules. This file owns package boundaries and
data flow.

## Packages

`rxembed` is the normal workflow facade. It accepts strings, paths, RDKit molecules, and selected metal
isomers, then returns a chainable `Ensemble` or `EnsembleSet`. `stereo="separate"` returns a
`list[EnsembleSet]`.

`rxembed.core` is the low-level API for callers that already have an explicit-H RDKit `Mol` or an `Isomer`.
It returns `Conformers` and does not expose pipeline operations.

Core implementation lives in the flat modules under `src/rxembed/`. They use NumPy, RDKit, and relative
imports, and never import `pipeline`. `src/rxembed/pipeline/` owns source adaptation, candidate expansion,
search, selection, scoring, visualisation, and optional integrations. Pipeline imports core through `rxembed`;
optional dependencies are loaded only by the operation that needs them.

`__init__.py` composes the public facade. `core.py` is the low-level engine facade. Implementation dependencies
point from pipeline to core.

## Layers

A module imports only modules listed before it in its own row, or modules in rows above it. Imports sit at
module top, except optional dependencies at their point of use.

| layer | modules | owns |
|---|---|---|
| 0 base | `utils`, `constraints` | RDKit graph and geometry facts; the `Constraints` payload and its measurement |
| 1 shapes | `metal_polyhedron` | polyhedron registry, symmetry, fits, slot notes |
| 2 metal graph | `metal_core` | metal state records, surrogate surgery, ligand graph, charge canonicalisation |
| 3 graph chemistry | `stereo`, `metal_distance` | organic stereo; the M-L length model |
| 4 DG and force-field engine | `mechanisms`, `relax`, `bounds` | constraint-to-bounds and UFF terms; restrained UFF; RDKit bounds matrices, native ligand reach, seeding |
| 5 metal models and perception | `metal_donor_orient`, `metal_perceive`, `metal_stereo`, `metal_slots` | donor fold model; reading a sphere from coordinates; canonical metal stereo; slot assignments |
| 6 compile | `metal_constraints` | one metal state to `Constraints` |
| 7 identity | `metal_isomer` | `Isomer`, `IsomerSet` |
| 8 screen and enumeration | `metal_screen`, `metal_enumeration` | feasibility screen; candidate generation |
| 9 strings | `metal_smiles` | dative SMILES and CXSMILES |
| 10 orchestration | `embed` | seeding, acceptance and recovery (`Conformers`) |
| 11 facade | `core` | low-level API |
| 12 pipeline | `calculators`, `search`, `xyz2mol_local`, `xyz2mol_tmc`, `perceive`, `nci`, `select`, `viz`, `metrics`, `stereo_check`, `geom_check`, `ensemble`, `dispatch`, then `pipeline/__init__` and `rxembed/__init__` | workflow |

The engine sits below the metal compile on purpose: `mechanisms`, `relax` and `bounds` read `Constraints` as
data and know nothing about metal identity, while the compile and the screen use them as tools. Data still flows
compile, then engine, then embed.

## Flow

```text
source
  -> pipeline normalization and identity expansion
  -> Candidate(Isomer or Mol, fix, tags)
  -> prepare -> Constraints
  -> edited RDKit ETKDG bounds -> seed and graft -> Conformers
  -> restrained UFF -> core and workflow checks -> replacement
  -> Ensemble or EnsembleSet, or list[EnsembleSet] for stereo="separate"
  -> optional search, selection, scoring, and QA
```

`pipeline.dispatch` expands organic stereo, metal identities, vacant-site coordination, and automatic contacts
into candidates. `_execute` is the only place a pipeline candidate enters core. When a call expands into two or
more candidates, one that raises `EmbeddingError`, at seeding or at relaxation, is skipped with one warning
line and kept in `EnsembleSet.errors`, and the result stays an `EnsembleSet`; the call raises only when none
embeds. A lone candidate, such as one selected `Isomer`, raises its own error. Expanded organic stereoisomers
and automatic contact modes also record any other `ValueError` or `RuntimeError` there, as an `EmbeddingError`
naming the stereoisomer or mode.

`embed.prepare` converts a `Mol` or selected `Isomer` plus `fix` and `constrain` into one `Constraints` value.
`seed_conformers` edits RDKit's native ETKDG bounds matrix, embeds, grafts fixed coordinates, and filters raw
coordination stereo. `Conformers` owns restrained relaxation, structural acceptance, and replacement;
`Ensemble` adds workflow checks for connectivity, requested stereo, and donor hand. `embed.prepare_relax` and
`pipeline.dispatch.minimize` share that same preparation and graft for search-free restrained-UFF relaxation of
existing coordinates.

## Rules

- Select chemical identity before numerical geometry. `coordinate=` fills vacant vertices on a copied `Isomer`;
  it is not a late constraint patch.
- `Constraints` is the only geometry payload. Mechanisms translate it into matching distance-geometry bounds
  and force-field terms.
- `fix` is rigid, `constrain` is releasable, and `template` supplies coordinates to `fix`. Rigid terms win over
  inferred soft contacts; competing soft terms are errors. Torsions are owned by their central bond, and a
  coordinate graft only removes a generated term when it owns every real atom that defines it.
- Keep the selected `Isomer` attached through embedding, search, minimization, and validation. It records metal
  connectivity, occupied slots, hand, and haptic winding.
- Never put a real-metal graph into ETKDG or a force field. Core uses a surrogate graph, then restores and
  validates the selected metal state.
- Keep structural validity separate from optimizer convergence. Replace or raise on wrong identity. Report an
  unconverged conformer only when it still satisfies the structural rules.
- Reject non-finite coordinates before distance, shape, connectivity, or frozen-core checks.
- Use `constraints.constraint_value` for distances, angles, periodic dihedrals, and haptic centroids.

### Identity and enumeration

`Isomer.cons` compiles `Constraints` afresh through `metal_constraints.py` on every access; retained and
enumerated states share this compiler. Model M–L distances (`lengths='model'`) are the default; explicit
`lengths='input'` measures them from a supplied conformer instead. Angular preferences come from the
polyhedron and ligand graph in both modes.

`metal_slots.py` reduces donor-slot assignments by proper rotations and ligand equivalence. `metal_enumeration.py`
combines metal, ligand, and haptic stereo, and `metal_screen.py` owns feasibility screening: single-centre
real-atom networks are screened against compiled constraints and native ligand reach, using RDKit's native 1-4 interval for
ring-closed paths and a torsion-independent upper envelope for free acyclic paths. Triangle contradictions and
interval-certified Euclidean contradictions can reject a candidate; failed optimization and successful
screening prove neither infeasibility nor feasibility. Virtual centroids are omitted from that joint screen,
not the remaining real-atom network.

`screen=False` disables feasibility screening, not symmetry reduction or embedding validation.
`observed_only=True` is the explicit, resource-bounded choice to return only a measured coordinate assignment.
Exceeding the enumeration resource limit raises an error, and every embedding generates fresh native DG seeds,
including for a selected observed state.

### Native DG and UFF

`bounds.py` starts from native RDKit bounds; `mechanisms.py` interprets each `Constraints` field for DG and
UFF, rooted at the `Mechanism` class. Bounds construction has one ordering: generate windows, relieve obsolete
floors, commit stated windows, project dependent terms, then triangle-smooth. Passing that smoothing does not
establish 3D realizability.

The default is native `KDG()` with AIO refinement, for every molecule; an empty search retries with random
coordinates under the same model. `EmbedParams` (`bounds.py`) is the one object that reproduces a seed batch:
sampling (`seed`, `threads`, `prune_rms`, `knowledge`), an optional native `EmbedParameters` (used directly by
reference, without cloning or model-changing retries; temporary runtime scalar overrides are restored
afterward, while the generated bounds and native failure counters remain on it), and rxembed's four matrix-edit
and ablation switches. `coplanar_14` and `metal_floor_relief` filter a constraint copy only at matrix
construction, and neither changes UFF or acceptance, which retain the complete geometric payload.
`donor_orientation` and `conjugation` independently ablate rxembed's optional donor-fold and organic
conjugation cleanup terms; explicit stereo and native RDKit terms stay active either way. `Conformers.params`
carries the `EmbedParams` a result was seeded with, so replaying it reproduces the batch; `resolve_params` folds
a facade's plain `seed`/`threads` keywords into one when `params` is not given. `max_iters` changes only the
restrained-UFF iteration cap after DG; it does not alter seed construction or candidate enumeration.

`embed.py` selects metal hand and haptic winding from raw seeds and grafts fixed coordinates via
`seed_conformers`. `relax.py` owns one restrained-UFF attempt, including private typing and optimizer status;
exceptions restore the selected coordinates. Public graphs retain real atoms and bonds; surrogate metals and
haptic centroid helpers are private, and helper-member radii are DG priors, not extra UFF walls.

### Acceptance and recovery

`Conformers` owns core validation and both bounded recovery loops: same-seed stiffness retries and fresh-seed
replacements. Replacement batches reuse the same relaxation and acceptance code without recursively replacing
themselves; accepted replacements keep their conformer ids and propagate optimizer status and energy.

Core checks cover finite coordinates, numeric fixes, ligand bonds, rigid shapes, the selected metal state, and
structural coordination terms. Metal-state validation fits donor directions to their assigned polyhedron slots;
individual donor-metal-donor windows bias that fit rather than acting as separate hard gates, while M–L
windows, donor orientation, coplanarity, and umbrella caps remain direct postconditions.

Convergence and identity are kept separate. An unconverged endpoint or a restored seed may survive only if it
passes the structural contract, lands in `.unrelaxed` without a comparable energy, and is reported by one
warning. Exhausted mandatory identity requests raise `EmbeddingError` instead of publishing a wrong state.
Every `Failure` is built by the check that finds it, in chemistry words, and grouped by `Failure.key` (kind and
site). Retries log at DEBUG; an error is one sentence naming the isomer, the most common failure and one remedy,
with the counts on `EmbeddingError.failures`.

`pipeline.Ensemble` adds workflow stereo checks and, with the `workflow` extra, connectivity checks using quick
perception and distance thresholds. `.check()` reports physical QA diagnostics and `.filter("geometry")`
enforces them explicitly; ordinary metal workflows also apply `geom_check`.

### Input and optional capabilities

`pipeline/perceive.py` owns connectivity and bond-order backends. Fallbacks warn, record the backend actually
used, and may not drop or reorder atoms or change the requested charge; `fallback=False` makes the selected
backends strict.

The base distribution includes NetworkX for pipeline code; no core module imports it. Optional search,
calculator, and QA backends are imported at point of use, and core behavior must not depend on installed
extras.

## Files

| owner | responsibility |
|---|---|
| `__init__.py`, `pipeline/__init__.py` | public facade composition |
| `core.py` | low-level engine facade, logging setup, and version lookup |
| `constraints.py` | constraint data, composition, resolution, and measurement |
| `mechanisms.py` | translation of constraints into DG bounds and FF terms |
| `bounds.py`, `embed.py`, `relax.py` | ETKDG setup, orchestration, restrained UFF, and core acceptance |
| `metal_core.py` | metal state, graph surgery, and coordination primitives |
| `metal_isomer.py`, `metal_enumeration.py` | selected isomers, collections, and candidate enumeration |
| `metal_screen.py` | feasibility screen of candidate arrangements against compiled constraints and ligand reach |
| `metal_polyhedron.py`, `metal_slots.py`, `metal_stereo.py` | shape tables, reachable slot assignments, and canonical metal stereo |
| `metal_constraints.py`, `metal_distance.py`, `metal_donor_orient.py` | compile selected metal states into geometry constraints |
| `metal_perceive.py` | reading a coordination sphere from coordinates: shape, overbonding, donor orientation |
| `metal_smiles.py`, `stereo.py`, `utils.py` | metal strings, organic stereo, and shared RDKit geometry facts |
| `pipeline/dispatch.py`, `pipeline/ensemble.py` | the `embed`, `metal` and `minimize` verbs with candidate routing; chainable results and `wrap` |
| `pipeline/perceive.py`, `pipeline/xyz2mol_*.py` | coordinate input and bond perception |
| `pipeline/nci.py`, `pipeline/search.py`, `pipeline/select.py` | contacts, conformer search, pruning, and selection |
| `pipeline/calculators.py`, `pipeline/geom_check.py`, `pipeline/metrics.py`, `pipeline/stereo_check.py`, `pipeline/viz.py` | optional calculators, QA, metrics, stereo checks, and visualisation |

`read_xyz` owns perception fallback and must preserve atom count and order.

## Where changes go

| adding | existing owner |
|---|---|
| constraint kind | `Constraints` field plus a root `Mechanism` |
| coordination shape | `metal_polyhedron.POLYHEDRA` row |
| NCI contact kind | `pipeline.nci.KINDS` row |
| search or calculator backend | `pipeline/search.py` or `pipeline/calculators.py` |
| QA or input format | the owning `pipeline/` module |
| package-managed optional dependency | `pyproject.toml` extra plus guarded point-of-use import |

Keep chemistry rationale beside its owning code and measured results in local `benchmark/`. Do not add another
package directory merely to mark a tier.
