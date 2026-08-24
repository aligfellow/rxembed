# rxembed development instructions

## Scope

- Build a library, not a CLI.
- Put usage in `README.md`.
- Put structure in `ARCHITECTURE.md`.
- Put rationale beside the owning code.

## Structure

- Keep core modules flat in `src/rxembed/`.
- Keep core limited to NumPy and RDKit.
- Keep input adaptation and optional capabilities in `pipeline/`.
- Keep `__init__.py` as facade composition only.
- Keep `rxembed.embed` as the normal facade and `core.py` as the Mol/Isomer engine facade.
- Do not add a directory unless no existing abstraction fits.
- Keep core imports relative and pipeline imports from `rxembed`.
- Never import `pipeline/` from core.

## Implementation

- Prefer KISS, YAGNI, deletion, and existing abstractions.
- Add a registry row or argument, not a new code path.
- Derive chemistry rules from the graph; do not enumerate cases.
- Tabulate genuine physical constants and cite their source.
- Use `Constraints` as the only stage-to-stage constraint struct.
- Edit RDKit's ETKDG bounds matrix; do not replace it.
- Bias seed geometry; let energy decide final geometry.
- Fail loudly with an actionable remedy.
- Never silently change a graph, backend, or energy result.

## Metal SMILES

- Write ionic dative SMILES with donor-to-metal arrows and formal ligand charges.
- Preferred example: `N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]`.
- Keep ordinary donor hydrogens implicit; keep hydride, H₂, and bridging-H sites explicit.
- Accept neutral/covalent and dative inputs.
- Write neutral/covalent SMILES only when requested or when demonstrating input support.

## Dependencies

- Base tier: NumPy, RDKit, NetworkX.
- Import optional backends at point of use under `try/except ImportError`.
- Name the missing distribution and extra in dependency errors.
- Add every optional dependency to `pyproject.toml` extras.
- Treat a new base dependency as a tier decision.
- Keep core behavior invariant across installed extras.
- Keep the version sourced from `pyproject.toml` via `importlib.metadata`.

## Change placement

- Constraint kind: `Constraints` field plus root `Mechanism`.
- Coordination shape: `POLYHEDRA` row in `metal_polyhedron.py`.
- NCI contact: `KINDS` row in `pipeline/nci.py`.
- Search, calculator, QA, or input format: owning `pipeline/` module.
- Measured number: `benchmark/`.

## Comments and docs

- Start docstrings with a one-line imperative summary.
- Use NumPy docstring conventions.
- Keep only guarantees, caveats, chemistry rationale, and measured decisions.
- Explain each concept once in its owning module.
- Do not restate code in comments.
- Keep log messages under about 110 characters.
- Check logging placeholder counts after edits.
- Name fixtures for their chemistry.
- Do not use em dashes, bold emphasis, or caps emphasis.

## Evidence and tests

- Give a command and number for behavior claims.
- Label unmeasured claims as unmeasured.
- Verify measurements exercise the named path.
- Add the smallest regression test for every fixed defect.
- Mutate the named behavior to verify the test fails.
- Test contracts, not internal constants or duplicated paths.
- Keep tests with their source owner.
- Skip optional-tier tests cleanly on a base install.
- Run `benchmark/run.py` after changes to `bounds.py`, `mechanisms.py`, or `metal_*`.
- Keep `benchmark/` local, gitignored, and free of redistributed structures.
- Never read `benchmark/` from `src/`; gate test access on its existence.

## Development loop

- Run: assess, plan, implement, adversarial review, edge-case review, reassess, regress.
- After implementation, ask independent reviewers to break the change.
- Default a claim to refuted when its key number cannot be reproduced.
- Use worktree isolation for any agent that may edit.
- Run tests with no review agents active.
- Use unique scratch filenames.
- Demonstrate new capabilities in `examples/*.ipynb` with the public API.
- Gate embedded notebook structures with `rx.geom_check.check`.
- Clear notebook outputs before commit.

## Gates and toolchain

- Gate changes with `just test`; use `just check` for lint and types too.
- Run `just setup` for environment setup and pre-commit installation.
- Keep bare `uv sync` as the full dev environment.
- Do not use `uv sync --all-extras`.
- Use `uv pip install .` or `uv sync --no-default-groups` for the base tier.
- Keep extras limited to `search` and `workflow` unless a real capability requires another.

## Commits

- Keep history linear and squash before landing.
- Make each commit self-contained.
- Never add AI attribution, assistant trailers, tool names, or session URLs.
