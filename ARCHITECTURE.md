# Architecture

`README.md` is the API. `AGENTS.md` is the development contract. This file owns the module boundaries.

## Flow

```text
source
  -> normalize input and dissolve template= into fix=
  -> expand organic stereo identities
  -> select ordinary, retained-metal or enumerated-metal identities
  -> seat coordinate= donors
  -> expand automatic contact modes
  -> Candidate(spec, fix, tag) x contact mode
  -> prepare + compose soft contacts
  -> Constraints
  -> MECHANISM_ORDER -> DG windows + FF terms
  -> seed -> graft -> relax -> validate -> replace failed -> finalize
  -> Ensemble | EnsembleSet | list[EnsembleSet]
```

`Constraints` is the sole geometric payload. `Isomer` is chemical identity: real metals, polyhedra, occupied
vertices, haptic faces and donor bonds. `embed.py` also carries temporary graft coordinates while it drives
the sequence. `bounds.py` edits RDKit's bounds matrix, `relax.py` applies the matching restrained-UFF terms,
and the result is `Conformers`.

Identity is complete before geometry compiles. `coordinate=` fills vacant vertices on a copied `Isomer` and
adds its donor bonds; it is not a late constraint patch. `metal=<geometry>` is pipeline sugar for
`enumerate_isomers` followed by one independent embed per isomer. A selected `Isomer` skips enumeration but
not compilation, execution or validation.

Candidate expansion and candidate execution are separate responsibilities. Stereo, metal identity,
`coordinate=` and automatic contacts only produce tagged candidate data. None calls another dispatcher or
embeds a conformer. `pipeline.dispatch._execute` is the only candidate executor and always runs `prepare`,
soft-contact composition and `seed_conformers` in that order.

User-selected metal and `coordinate=` identities are mandatory: one failing candidate fails the call instead
of disappearing because it has siblings. Generated organic stereoisomers and automatic contact modes are
alternatives and may be skipped with a warning; if none survive, the error includes the last concrete cause.

A coordinate `fix=` may participate in metal identity selection because frozen donors retain their measured
vertices. In that case enumeration consumes the raw fix and stores the resulting base `Constraints` on each
`Isomer`; a selected `Isomer` or ordinary molecule carries the raw fix to `prepare`. Both meet at the same
`Constraints` composition seam before embedding. This phase dependency does not create a second chemistry or
relaxation path.

Constraint compilation is chemistry-first. `metal_constraints.py` derives the complete coordination field,
then removes a term only when the coordinate graft owns every real atom that defines it. A measured M-L
length, a modelled newly coordinated donor and graph-derived donor orientation may coexist in one state.
Length provenance, partial freezing and soft workflow contacts therefore do not create alternate chemistry
paths.

`fix` is the rigid verb and `constrain` is the releasable verb. `template` only supplies coordinates to
`fix`; contacts are adapted through the same validated `constrain` resolver. Rigid terms win over inferred
soft contacts. Two soft sources may not claim the same coordinate. A soft window may not replace a selected
metal-state term, while an explicit numeric `fix` may override one. Torsions are owned by their central bond,
so two sources cannot state competing quartets around one bond and a soft dihedral cannot suppress
coplanarity or an umbrella restraint. Plane constraints are soft and release with other `constrain` terms
during exploratory search.

`relax.py` owns one force-field attempt and reports its optimizer status. `Conformers` in `embed.py` owns the
same-seed stiffness ladder, the sole fresh-seed replacement loop, and the core geometry contract: numeric fixes,
graph bonds, rigid shapes, selected coordination state and structural coordination terms. Acceptance returns the
failed conformer ids grouped by reason. `pipeline.Ensemble` supplies one additional validator for requested
workflow stereo, labile donor hand and, with the workflow extra, perceived connectivity; it does not restate the
core contract or seed a replacement itself. Broader free-periphery `geom_check` remains diagnostic during metal
finalization because an
explicit `fix` may intentionally violate a ground-state rule. `.check()` and `.filter("geometry")` expose that
report when the caller wants it as a gate.

Replacement batches run the same relaxation and acceptance functions as their parent, without recursively
starting another replacement controller. Successful replacements keep the original conformer id and propagate
energy and optimizer status. `Ensemble` records lifecycle as one stage value (`seeded`, `relaxed`, `minimized`),
while `Isomer` remains the only durable metal-connectivity record. Search invalidates stage-dependent energy,
optimizer, reaction and trajectory records together.

A converged force-field outlier is removed only after structural acceptance. The relative energy window is an
ensemble ranking heuristic, not a molecular-identity failure, so it does not trigger fresh embedding.

Optimizer convergence and structural identity are independent. A conformer that reaches the iteration ceiling
may remain explicitly listed in `.unrelaxed` when its geometry still passes the structural contract. A wrong
metal state may not be published: fresh seeds replace it, and an exhausted public operation fails rather than
returning fewer conformers or the wrong identity.

The selected `Isomer` remains attached when the workflow restores and connects the public metal graph. Later
search and minimization cycles therefore validate against the same occupied slots, hand and haptic winding.
Before a later search moves atoms, the workflow rebuilds the `Isomer`'s surrogate graph and copies the current
conformers onto it; the connected real-metal graph never enters ETKDG or a force field.

Every core and workflow acceptance starts by rejecting non-finite coordinates. Downstream distance, shape,
connectivity and frozen-core comparisons may therefore assume a numerical conformer; NaN cannot satisfy a
contract by making both sides of a range comparison false.

The selected metal state is validated as one correspondence-preserving fit of realised donor directions to
their occupied polyhedron slots. Individual D-M-D windows bias the ideal seat but are not separate hard gates:
chelates and haptic faces can validly distort one angle while retaining the requested state. Structural M-L
windows, donor orientation, coplanarity and umbrella caps remain direct postconditions.

`constraints.constraint_value` is the shared ruler for distances, angles, periodic dihedrals and haptic
centroids. Core acceptance, pipeline QA, diagnostics and public `.measure()` choose different policies but do
not reimplement the measurement.

The DG engine remains RDKit: `bounds.etkdg` creates `ETKDGv3`, and `seed_coordinates` supplies an edited native
bounds matrix through `SetBoundsMat`. Metal hand and haptic winding are checked on raw DG seeds because RDKit's
Python distance-geometry API cannot accept those coordination stereo records.

`metal_core.py` owns graph surgery, perception primitives and the immutable `MetalState`: one real metal,
polyhedron, slot-ordered donors or haptic faces, and hand. `metal_stereo.py` canonicalizes site identity and
reads metal or haptic hands. `metal_slots.py` produces distinct, reachable donor-to-polyhedron slot assignments.
`metal_enumeration.py` combines ligand, haptic and per-centre choices into the `Isomer` and `IsomerSet` types in
`metal_isomer.py`, without building numerical fields for every candidate.

`Isomer.cons` asks `metal_constraints.py` for a fresh `Constraints` compiled from the selected state. A private
source-Mol snapshot preserves `lengths='input'`; haptic centroids exist only while compiling and running DG/UFF.
Stored molecules and `MetalState` contain real atom indices only.

The normal API adapts strings, paths and external tools around that core:

```text
rxembed.embed(str | path | Mol | Isomer)
    -> pipeline dispatch -> core constraint / seed / relax seam
    -> Ensemble | EnsembleSet | list[EnsembleSet] -> search / select / score
```

`embed.prepare` is the core Mol/Isomer-to-`Constraints` seam. Pipeline adapters normalize sources and expand
candidate identities. Search-free relaxation uses `prepare_relax`, which calls the same preparation and applies
the returned graft directly.

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
