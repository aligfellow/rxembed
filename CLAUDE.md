# rxembed

`rxembed` (`src/rxembed/`) is a **unified constrained / templated conformer-embedding** package: one
composable chain that embeds conformers for anything from a bare SMILES to a frozen bimetallic transition
state with NCI-seeded binding modes, then searches, dedups, and scores them with real energies.

It is a **library, not a CLI** — you drive it from Python / notebooks. Import it as `rxembed` (no alias in the
package; notebooks use the full name).

## The one chain

Every capability is *one more argument* on the same pipeline, not a new code path:

```python
import rxembed as rx
ens = rx.embed(source, ...).mc().prune()      # embed -> Monte-Carlo search -> dedup
ens.representatives()                          # the distinct modes
ens.score("gxtb").lowest(3).optimize("gxtb")   # real energies, then optimise the best
```

`rx.embed(source, *, metal, freeze, distances, angles, planes, contacts, coordinate, template, match,
anchor, charge, n, seed, knowledge, stereo)` returns an **`Ensemble`** (or an **`EnsembleSet`** when the input
is inherently several candidates — metal isomers, NCI binding modes, ambiguous coordination).

| Want | Call |
|---|---|
| free / flexible | `rx.embed("CCO")` |
| NCI / vdW complex | `rx.embed("A.B", contacts="auto")` → one candidate per discovered grip |
| a specific H-bond grip | `rx.embed("A.B", contacts=rx.nci_modes(mol)["HB:…"])` |
| frozen TS core | `rx.embed(ts_xyz, freeze=reacting)` |
| constrained TS **from SMILES** | `rx.embed("cat.substrate", distances={core role-distances})` |
| metal coordination isomers | `rx.metal("…[Pd]…", "square_planar")` → cis/trans, mer/fac… |
| replace a ligand on a known core | `rx.embed(analogue, template=parent, match=core_SMARTS)` |

**Pipeline stages** (the mutation contract is explicit): `mc`, `minimize`, `prune` build in place and chain;
`lowest`, `representatives`, `align` return a *new* ensemble; `view`, `landscape`, `cluster` never mutate.
Real-energy tiers via `score(refine=…)` (single point) and `optimize(refine=…, level=…)` (geometry, frozen core
held): **`ff` → `gfnff` → `gfn2` → `gxtb`** (force field → GFN-FF NCI-aware → GFN2 → g-xTB).

## Design principles

- **KISS / no over-engineering.** Adding a capability should be a *registry row or one argument*, not a
  branch. NCI contact types live in one `KINDS` registry (`constraints/nci.py`); the enumerator is generic
  over it. Dead machinery gets deleted.
- **One struct in, one struct out.** `Constraints` (`constraints/base.py`) is the single thing every *builder*
  fills and every *stage* reads. Nothing reaches into the pipeline sideways.
- **Edit, don't replace.** The ETKDG bounds matrix is *edited* from RDKit's knowledge-derived bounds, so
  experimental-torsion / basic-knowledge seeding survives even under a tight custom core.
- **Bias the seed, let energy decide.** Constraints bias the *starting* geometry; they are not truth.
  `mc(explore=True)` fires the search twice (grip preserved, grip released), pools, and lets a real energy
  choose.
- **Surrogate the hard bits.** A metal becomes a carbon surrogate for the force field (bonds stripped), held
  by soft shape constraints; the exact frozen TS core is grafted back by Kabsch (0.000 Å).
- **Fail loud, not silent.** A calculator that produces no energies *raises* (never a silent FF fallback);
  g-xTB + solvent *raises* rather than quietly running gas-phase.
- **Physical honesty over cleverness.** σ-holes only on polarisable heavies; a thione C=S is an acceptor, not
  a σ-hole donor; a reciprocal A-H···B / B-H···A 2-cycle can't coexist in one pose. Rules grounded in
  chemistry, each a small testable change.

## Development loop

Every non-trivial change runs the validation loop — the project's working discipline:

```
assess → plan → implement → robustness review (subagent) → edge-case review (subagent)
       → reassess → regress (tests) → update memory
```

- **Adversarial subagent reviews.** After implementing, spawn independent reviewers whose job is to *break*
  the change (a robustness pass and a chemistry edge-case pass). Fold findings back before "done".
- **Tests are the gate.** `tests/` (organic, constraints, NCI, metal isomers, frozen TS, template) must stay
  green; the frozen-core distance assertions must hold. xtb-binary and openconf cases are `skipif`-gated so
  the base suite runs anywhere. `just test` runs it.
- **Proof lives in notebooks.** New capability is demonstrated in `examples/*.ipynb` with the **real inline
  API — no hidden wrappers**: logging on, every stage dumped, key geometries rendered with `xyzrender`, each
  gated by `rx.geometry.check`. Clear outputs (`jupyter nbconvert --clear-output --inplace 0*.ipynb`) before a
  lightweight commit.
- **Memory carries state across sessions** (`~/.claude/.../memory/rxembed-*.md`).

## Toolchain

Copier template with **uv / ruff / ty / pytest / pre-commit / just**. `just check` = lint + type + test;
`just setup` = `uv sync` extras + `pre-commit install`; `just fix` autofixes. Extras keep the base install
light: `mc` (openconf), `nci` (xyzgraph, scikit-learn), `viz` (matplotlib, seaborn, pandas), `ase`, `racerts`.
Version is single-sourced from `pyproject.toml` via `importlib.metadata` in `__init__.py` — never hardcode a
second copy. `ruff D` (numpy pydocstyle) is on: one-line imperative docstring summaries + only the
non-inferable specifics; comments only for load-bearing *why*.

## Real-energy caveats

`score('gxtb')` / `optimize('gxtb')` need the `xtb` executable on `$XTB_EXE`; **GFN-FF needs a standard Grimme
`xtb`** on the same path. Embedding / search / prune are pure-Python (RDKit + openconf) and need no external
calculator. `mc()` needs openconf; without it the ETKDG seeds are still returned (degrades, doesn't crash).

## Where things live

- `src/rxembed/pipeline.py` — `embed`, `wrap`, `Ensemble` (mc/minimize/prune/score/optimize/representatives),
  `EnsembleSet`.
- `src/rxembed/embed/` — `dispatch.py` (the embed machinery: `_xyz_to_mol`, metal path, template path,
  `_embed_dispatch`), `bounds.py` (edited-bounds ETKDG embed + seed-count scaling), `mc.py`.
- `src/rxembed/constraints/` — `base.py` (`Constraints`, `relaxed()`), `nci.py` (KINDS registry, binding
  modes, acceptor quality, reciprocal split), `metal.py`, `builders.py` (`from_spec`, `from_template`).
- `src/rxembed/dedup/` — `prism` moi/rmsd/descriptor prunes + the distinct energy-aware `energy_prune`.
- `src/rxembed/refine/` — `xtb.py` (executable interface), `calculator.py` (`resolve`, `XTB`, `ASE`).
- `src/rxembed/geometry.py` — the TS-aware geometry gate (`check`): broken conjugation, bad H positions,
  clashes, moved core; frozen/metal-aware (dative distances aren't clashes).
- `src/rxembed/stereo.py` — chirality fingerprints + the auto-preserve gate.
- `examples/*.ipynb` — the 8-notebook breadth tour; `examples/structures/` — static TS `.xyz` geometries.

See `plan.md` for the forward roadmap and open threads.
