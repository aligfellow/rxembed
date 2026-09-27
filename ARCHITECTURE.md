# Architecture

`README.md` owns usage, `AGENTS.md` owns development rules, and each module owns its own rationale. This file
owns package boundaries and data flow.

## Packages

`rxembed` is the workflow facade. It accepts strings, paths, RDKit molecules and selected metal isomers, and
returns a chainable `Ensemble` or `EnsembleSet`, or a `list[EnsembleSet]` for `stereo="separate"`.

`rxembed.core` is the low-level API for a caller that already has an explicit-H RDKit `Mol` or an `Isomer`. It
returns `Conformers` and has no pipeline operations.

The core modules sit flat in `src/rxembed/`. They use NumPy and RDKit with relative imports and never import
`pipeline`. `src/rxembed/pipeline/` owns input adaptation, candidate expansion, search, selection, scoring, QA
and optional integrations. It imports core through `rxembed`, loads optional dependencies only in the operation
that needs them, and is the only user of NetworkX.

## Layers

A module imports only modules in rows above it, or listed before it in its own row. Imports sit at module top,
except optional dependencies at their point of use.

| layer | module | owns |
|---|---|---|
| base | `utils` | coordinate math, small RDKit facts and the QA `Violation`; nothing about metals or constraints |
| | `constraints` | the `Constraints` payload, the `fix`/`constrain`/`template` resolver, `match` and measurement |
| shapes | `metal_polyhedron` | the `POLYHEDRA` registry, proper rotations, slot notes and hand frames |
| metal graph | `metal_core` | metal state records, surrogate swap and restore, haptic sites, the ligand graph |
| graph chemistry | `stereo` | ligand stereo, including coordination-locked centres: read from 3D, written, enumerated |
| | `metal_distance` | the fitted M-L length model and the floors for non-donor atoms near the metal |
| DG and FF engine | `mechanisms` | each `Constraints` field as DG bounds and restrained-UFF terms (`Mechanism`) |
| | `relax` | one restrained UFF attempt on a private force-field graph (`restrained_uff`, `UFFRecord`) |
| | `bounds` | `EmbedParams`, native bounds and their edits, ligand reach, fragment contacts, DG seeding |
| metal models | `metal_donor_orient` | donor hybridisation and the soft holds that point each donor at the metal |
| | `metal_perceive` | read-only judgement of a sphere: shape readings and the shape gate, overbonding, donor fold |
| | `metal_stereo` | coordination-site identity, metal hand, haptic winding and face descriptors |
| | `metal_slots` | distinct donor-to-slot assignments under rotations, and chelate bite windows |
| compile | `metal_constraints` | one metal state to `Constraints`: M-L windows, L-M-L angles and chelate bites |
| identity | `metal_isomer` | `Isomer` and `IsomerSet`, and the isomer read from coordinates |
| screen and enumeration | `metal_screen` | the reach and fit screens that drop arrangements the model rules out |
| | `metal_enumeration` | ligand, haptic and coordination states combined into `Isomer` candidates |
| strings | `metal_smiles` | reading and writing dative SMILES and CXSMILES |
| orchestration | `embed` | `prepare`, seeding, acceptance and recovery: `Conformers`, `Failure`, `EmbeddingError` |
| facade | `core` | the low-level API, logging setup and the version lookup |
| pipeline | `calculators` | the xtb executable and the `Calculator` and `ASE` scoring surface |
| | `search` | Monte Carlo search through openconf |
| | `xyz2mol_local`, `xyz2mol_tmc` | vendored xyz2mol and its transition-metal bond-order and charge search |
| | `perceive` | `read_xyz`: connectivity and bond-order backends, fallbacks and the charge warning |
| | `nci` | the `KINDS` registry of contact kinds, contacts as constraints, binding-mode signatures |
| | `select` | descriptors, clustering and pruning |
| | `viz` | the ensemble landscape plot |
| | `metrics` | bond sanity and connectivity diffs |
| | `stereo_check` | stereo fingerprints for chirality-aware selection |
| | `geom_check` | the physical geometry gate (`check`, `GeometryReport`) |
| | `ensemble` | `Ensemble`, `EnsembleSet` and `wrap`: chainable search, scoring and workflow checks |
| | `dispatch` | the `embed`, `metal` and `minimize` verbs and candidate expansion |
| facades | `pipeline/__init__`, `rxembed/__init__` | public names only |

The engine sits below the metal compile on purpose: `mechanisms`, `relax` and `bounds` read `Constraints` as
data and know nothing about metal identity, while the compile and the screen use them as tools. Data still flows
compile, then engine, then embed.

## Flow

```text
source
  -> pipeline normalization and identity expansion
  -> Candidate(Isomer or Mol, fix, tags)
  -> prepare -> Constraints
  -> edited RDKit bounds -> seed and graft -> Conformers
  -> restrained UFF -> core and workflow checks -> replacement
  -> Ensemble or EnsembleSet, or list[EnsembleSet] for stereo="separate"
  -> optional search, selection, scoring and QA
```

`pipeline.dispatch` expands organic stereo, metal identities, open-site coordination and automatic contacts
into candidates. `_execute` is the only place a candidate enters core. When a call expands into two or more
candidates, one that raises `EmbeddingError` is skipped with one warning line and kept in
`EnsembleSet.errors`, and the result stays an `EnsembleSet`; the call raises only when none embeds. A lone
candidate, such as one selected `Isomer`, raises its own error. An expanded stereoisomer or contact mode also
records any other `ValueError` or `RuntimeError` there, as an `EmbeddingError` naming it.

`embed.prepare` turns a `Mol` or selected `Isomer` plus `fix` and `constrain` into one `Constraints` value.
`seed_conformers` edits RDKit's bounds matrix, embeds, grafts fixed coordinates and picks seeds with the
requested metal hand and haptic winding. `embed.prepare_relax` and `pipeline.dispatch.minimize` share the same
preparation and graft to relax existing coordinates without a search.

## Rules

- Select chemical identity before numerical geometry. `coordinate=` fills open vertices on a copied `Isomer`;
  it is not a late constraint patch.
- `Constraints` is the only geometry payload. Mechanisms translate it into matching distance-geometry bounds
  and force-field terms.
- `fix` is rigid, `constrain` is releasable, and `template` supplies coordinates to `fix`. Rigid terms win over
  inferred soft contacts; competing soft terms are errors. A torsion is owned by its central bond, and a
  coordinate graft removes a generated term only when it owns every real atom that defines it.
- Keep the selected `Isomer` attached through embedding, search, minimization and validation. It records metal
  connectivity, occupied slots, hand and haptic winding.
- Never put a real-metal graph into ETKDG or a force field. Core uses a surrogate graph, then restores and
  validates the selected metal state, including its formal charge.
- Keep structural validity separate from optimizer convergence. Replace or raise on wrong identity. Report an
  unconverged conformer only when it still satisfies the structural rules.
- Reject non-finite coordinates before distance, shape, connectivity or frozen-core checks.
- Use `constraints.constraint_value` for distances, angles, periodic dihedrals and haptic centroids.

### Identity and enumeration

`Isomer.cons` compiles `Constraints` afresh through `metal_constraints` on every access, so retained and
enumerated states share one compiler. Model M-L distances (`lengths="model"`) are the default; `lengths="input"`
measures them from the supplied conformer. Angle preferences come from the polyhedron and the ligand graph in
both modes.

`metal_slots` reduces donor-slot assignments by proper rotations and ligand equivalence. `metal_enumeration`
combines metal, ligand and haptic stereo and screens the candidates. Only the first of three screens is a proof,
and it is a proof against the compiled model, not against chemistry:

- `metal_screen` rejects a single-centre candidate whose compiled constraints contradict the ligands' native
  reach (row, triangle-closure and Euclidean certificates) or whose opposed donors exceed the acceptance gate's
  fit budget.
- The `metal_slots` edge rule holds each short chelate pair to a polyhedron hull edge, a measured claim.
- The chelate-bite fold check and the long-arc, trans and haptic span tests in `metal_screen` are model priors. A
  multi-metal candidate gets only these.

Passing the screen proves nothing. `screen=False` and any `fix=` turn all three off, but not symmetry reduction
or embedding validation, and the input conformer's own arrangement is exempt when its donor spans fit the native
reach. `observed_only=True` returns only the measured assignment. More than `MAX_EXHAUSTIVE_ORBITS` assignments
raise.

### Bounds and relaxation

`bounds` starts from native RDKit bounds and has one ordering: generate windows, relieve obsolete floors, commit
stated windows, project dependent terms, then triangle-smooth. Passing the smoothing does not establish that a
3D structure exists. An empty search is not retried.

`EmbedParams` is the one object that reproduces a seed batch. `Conformers.params` carries it, and
`bounds.resolve_params` folds a facade's plain `seed` and `threads` into one. `coplanar_14` and
`metal_floor_relief` filter a constraint copy only while the matrix is built; UFF and acceptance keep the full
payload.

`relax` owns one restrained UFF attempt, including private typing and optimizer status; an exception restores
the selected coordinates. Public graphs keep real atoms and bonds; surrogate metals and haptic centroid helpers
stay private.

### Acceptance and recovery

`Conformers` owns core validation and both bounded recovery loops: same-seed stiffness retries and up to three
fresh-seed replacement batches. Replacement batches reuse the same relaxation and acceptance code without
recursing, and accepted replacements keep their conformer ids.

Core checks cover finite coordinates, numeric fixes, ligand bonds, rigid shapes, the selected metal state and
coordination terms. The shape gate lives in `metal_perceive`: the requested polyhedron must read within
`_FIT_MARGIN` of the best reading, and each accepted conformer carries the `shape` record. Each `Failure` is
built by the check that finds it, in chemistry words, and grouped by `Failure.key` (kind and site). Retries log
at DEBUG; an `EmbeddingError` is one sentence naming the isomer, the most common failure and one remedy, with the
counts on `EmbeddingError.failures`.

`pipeline.Ensemble` adds workflow checks: requested stereo, donor hands, `geom_check` for metal workflows and,
with the `workflow` extra, connectivity. `.check()` reports QA diagnostics and `.filter()` enforces them.

### Input and optional capabilities

`pipeline/perceive.py` owns the connectivity and bond-order backends. A fallback warns, records the backend used,
and may not drop or reorder atoms or change the requested charge; `fallback=False` makes the chosen backends
strict. Optional search, calculator and QA backends are imported at point of use, and core behaviour does not
depend on installed extras.

## Where changes go

| adding | existing owner |
|---|---|
| constraint kind | `Constraints` field plus a root `Mechanism` |
| coordination shape | `metal_polyhedron.POLYHEDRA` row |
| NCI contact kind | `pipeline.nci.KINDS` row |
| search or calculator backend | `pipeline/search.py` or `pipeline/calculators.py` |
| QA or input format | the owning `pipeline/` module |
| package-managed optional dependency | `pyproject.toml` extra plus guarded point-of-use import |

Keep chemistry rationale beside its owning code and measured results in `benchmark/baseline.csv`. Do not add
another package directory merely to mark a tier.
