# rxembed development instructions

## General

### Code

- Prefer KISS, YAGNI, deletion, existing code, and data over branches.
- Fix a problem at its shared owner. Do not patch each caller or add a parallel path.
- Avoid one-use abstractions and pass-through wrapper functions. Add one only when it removes duplication or
  adapts an external interface.
- Prefer the standard library, native platform features, and installed dependencies before custom code or a
  new dependency.
- Fail with an actionable remedy. Never silently change inputs, backends, or results.

### Development loop

- Inspect the real code path and its tests before planning a change.
- Assess, plan, implement, adversarially review, check edge cases, reassess, and regress.
- Add the smallest regression test for a defect and verify that it fails without the fix.
- Test public contracts, not internal constants or duplicate implementations.
- Give the command and result for behavior claims. Label unmeasured claims as unmeasured.
- Treat a claim as refuted when its key result cannot be reproduced.
- Ask an independent reviewer to challenge non-trivial changes. Isolate editing agents in worktrees, use unique
  scratch names, and run tests with no reviewers active.

### Style

- Explain each guarantee, caveat, rationale, or measured decision once, beside its owner. Do not restate code.
- Start docstrings with an imperative summary and use the configured style.
- Use plain factual prose. Do not use em dashes, bold or caps for emphasis.
- Keep log messages under about 110 characters and check placeholder counts after edits.

### Commits

- Preserve unrelated working-tree changes.
- Keep commits self-contained, history linear, and squash before landing.
- Do not add AI attribution, assistant trailers, tool names, or session URLs.

## rxembed

### Layout

- Build a library, not a CLI. Put usage in `README.md`, package structure in `ARCHITECTURE.md`, and rationale
  beside its owning code.
- Keep core modules flat in `src/rxembed/`, limited to NumPy and RDKit, with relative imports only.
- Put input adaptation and optional capabilities in `pipeline/`, importing core through `rxembed`. Never import
  `pipeline` from core.
- Keep `__init__.py` as facade composition only. `rxembed.embed` is the normal workflow; `rxembed.core` is the
  low-level Mol/Isomer API.
- Put changes with the existing owner named in `ARCHITECTURE.md`. Prefer a registry row or argument to a new
  path, and add a directory only when no owner fits.

### Chemistry

- Use RDKit directly when possible. If RDKit offers several approaches, test the simplest ones on the same
  regression or benchmark. Keep custom code only when they fail.
- Derive chemistry rules from the molecular graph. Do not enumerate compounds or special cases.
- Use `Constraints` as the only stage-to-stage geometry payload.
- Keep RDKit's ETKDG engine and edit its bounds matrix. Bias seed geometry and let energy choose final geometry.
- Tabulate genuine physical constants and cite their source.
- Prefer ionic dative metal SMILES with donor-to-metal arrows and formal ligand charges, for example
  `N->[Pd+2](<-[Cl-])(<-[Cl-])<-N`. Accept neutral/covalent and dative inputs.
- Keep ordinary donor hydrogens implicit and hydride, H2, and bridging-H sites explicit.

### Dependencies

- The base tier is NumPy, RDKit, and NetworkX. NetworkX is pipeline-only; core remains NumPy and RDKit.
- Import optional packages at point of use. Put package-managed dependencies in an existing extra and name the
  extra in errors. Dependencies behind caller-supplied adapters remain caller-managed. Keep core behavior
  invariant across installed extras.
- Source the version through `importlib.metadata`. Treat a new base dependency or extra as a tier decision.

### Tests and tools

- Never make `src/` or tests depend on `benchmark/`. After changes to bounds, mechanisms, or metal geometry,
  run `just bench` and report every loss it prints.
- Refresh `benchmark/baseline.csv` in the commit that accepts a measured change. Keep `benchmark/results/`
  untracked. Ship a structure only with its origin in `benchmark/README.md` and its licence in `LICENSES.md`.
- Skip optional-tier tests cleanly on a base install.
- Name tests and fixtures for their chemistry. Keep tests with their source owner.
- Demonstrate new public capabilities in output-cleared `examples/*.ipynb` notebooks using the public API.
- Run `just test`; use `just check` for formatting, lint, types, and tests. Run `just setup` for environment and
  pre-commit setup.
- Keep bare `uv sync` as the full development install. Use `uv pip install .` or
  `uv sync --no-default-groups` for the base tier; do not use `uv sync --all-extras`.
- Keep extras limited to `search` and `workflow` until a real capability needs another.
