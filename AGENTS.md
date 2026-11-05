# rxembed development instructions

## Code

- Prefer KISS, YAGNI, deletion and existing code. Optimize a complete operation for a human reader: fewer concepts,
  representations and handoffs, with each contract visible at its owner.
- Fix a problem at its shared owner, not in each caller or a parallel path. Finish it across the callers and delete
  what it supersedes.
- Add no parameter, function or branch without a caller outside tests, and no aliases or shims. Say in the commit
  what was deleted.
- Write one function per real step and let it be long. A helper must name a step, share a decision or adapt an
  interface; no pass-through wrappers or single-use helpers of a few statements.
- Ruff enforces complexity 15, 15 branches and 6 returns. Review a signature over 6 arguments for redundant inputs.
  Never add an ignore, helper or record just to pass a limit. Split a module when its parts have different consumers.
- Prefer the standard library, RDKit and installed dependencies to custom code. Reuse a dependency's decisions, and
  report its defects to its owner instead of growing a second classifier.
- Give each decision one owner and state its contract beside it. Keep model predictions, seed targets, requested
  postconditions and QA checks distinct: shared arithmetic is not a shared tolerance, and failed heuristics do not
  prove infeasibility.
- Validate each fact once, at the first owner that can, and trust it downstream. Transformed output is new data.
- Fail with an actionable remedy. Never silently change inputs, backends or results.
- Add a record only when named fields, a lifetime or a prevented invalid state help a real caller; then use a frozen
  keyword dataclass for a value and a NamedTuple for an internal result. Encode no cases by tuple position, sentinel
  or container shape, and do not branch on `isinstance` of rxembed's own types.
- Use a registry for genuine interchangeable variants, otherwise a plain conditional; an abstract base only where a
  caller picks an implementation.
- Results own their coordinates: construction, selection and public Mol access copy, and internal numerical code
  mutates its working Mol and invalidates what it changed. A failed attempt stays retryable.
- Reuse a measurement on unchanged inputs; a persistent cache needs measured reuse and an invalidation owner.
- Name things for their responsibility in plain words. Own resources with context managers.

## Working

- Read the real code path and its tests before planning. Plan, implement, review adversarially, check edge cases and
  rerun the regressions.
- Add the smallest regression test for a defect and check that it fails without the fix.
- Give the command and result for each behaviour claim, label unmeasured ones, and report the verdict first. A claim
  whose key result cannot be reproduced is refuted.
- Ask an independent reviewer to challenge non-trivial changes. Isolate editing agents in worktrees and run tests with
  no reviewer active.
- Propose a change that needs a new concept, saying who does what, and wait for the owner. Track open work in GitHub
  issues.
- When feedback conflicts with a rule here, replace the rule in the same commit rather than adding another.

## Style and commits

- State each contract, caveat or reason once, beside its owner. Source carries no history, experiment narrative or
  TODO/NOTE markers; a measured decision keeps one line naming its rule and source, its numbers in the commit or
  `benchmark/README.md`.
- Start each docstring with one imperative line; add prose only for what the code does not show (units, a chemistry
  reason, a failure to handle), with no Parameters or Returns sections. A comment is one line saying why.
- Write plain prose: no em dashes, and no bold or capitals for emphasis. Keep log messages under about 110 characters.
- Keep commits self-contained and history linear; squash before landing. Preserve unrelated working-tree changes. Say
  in the body what was wrong, what was deleted and which check proves it. Add no AI attribution or session links.

## Layout

- A library, not a CLI: usage in `README.md`, the picture in `ARCHITECTURE.md` (current owners, flow, loops and open
  gaps, no history), rationale beside the code, evidence in commit bodies. Add no other docs.
- Shared foundations (`graph`, `stereo`, `utils`) sit flat at the top. Each subject folder imports only them and the
  folders before it: `geometry`, `metal`, `identity` and `engine` form the core (NumPy and RDKit only), `perception`
  adds xyzgraph and NetworkX, `workflow` the optional extras, and `api.py` and `ensemble.py` compose them. Imports are
  absolute, at module top, with no cycle and no private name across modules; `tests/test_dependencies.py` holds this.
- `rxembed` is the one public tier and `__init__.py` only composes it. `isomers`, `embed` and `coembed` take a SMILES
  or a Mol; every other verb reads a Mol. Public verb signatures change only with the owner's approval.
- Put an extension with its owner: a mechanism in `MECHANISM_ORDER`, a polyhedron in `POLYHEDRA`, a gate in
  `CORE_GATES`, an NCI kind in `perception/contacts.KINDS`, an embed switch in `EmbedParams`. A second policy owner or
  a new concept needs a proposal first.

## Chemistry

- rxembed's metal-aware coordination, donor and stereo contracts are authoritative; RDKit typing is a derived view and
  never redefines chemical identity.
- Use RDKit directly where it can; when it offers several routes, test the simplest on the same regression.
- Derive rules from the molecular graph, never from listed compounds. Tabulate genuine constants with their source.
- `Constraints` is the only stage-to-stage geometry payload. Keep ETKDG and edit its bounds matrix; bias the seeds and
  let energy choose the final geometry.
- Prefer ionic dative metal SMILES (`N->[Pd+2](<-[Cl-])(<-[Cl-])<-N`) and accept covalent ones. Keep ordinary donor
  hydrogens implicit and hydride, H2 and bridging-H sites explicit.

## Dependencies

- The base tier is NumPy, RDKit, NetworkX and xyzgraph, with the last two in `perception` and above. Connectivity
  validation is part of the base contract.
- Import an optional package where it is used, from an existing extra (`search`, `workflow`) named in its error, and
  keep core behaviour the same across extras. A dependency behind a caller-supplied adapter stays the caller's. A new
  base dependency or extra is a tier decision.
- Source the version through `importlib.metadata`.

## Tests and tools

- `tests/` mirrors `src/`, one module per source module plus the facade and layer tests. Name each test for its one
  contract in about 60 characters, naming chemistry where it marks a regression; the docstring carries the detail.
- Test public behaviour with the minimal example that reaches it, and keep integration tests for flows across owners.
  Do not patch private names or assert record positions. A test stays when it alone guards a contract clause or a
  fixed regression, measured by mutating its owner in a separate on-disk worktree.
- Compare rows that share an assertion as one observed table against the expected one; a registry gets one such test,
  a new entry a new row.
- Keep the default suite under a minute and mark longer tests `slow`; skip optional-tier tests on a base install.
- `just check` runs format, lint, types and tests, `just test` the tests, and `just setup` the environment. Bare
  `uv sync` is the full install, `uv sync --no-default-groups` the base tier.
- After changes to bounds, mechanisms or metal geometry run `just bench`, report every loss, and refresh
  `benchmark/baseline.csv` in the commit that accepts a measured change. `src/` and tests never import `benchmark/`;
  a shipped structure has its origin in `benchmark/README.md` and its licence in `LICENSES.md`.
- Show a new public capability in an output-cleared `examples/*.ipynb` using the public API, with results printed or
  drawn and no asserts.
