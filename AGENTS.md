# How we change rxembed

`README.md` is what it does and how to call it. `ARCHITECTURE.md` is how it fits together. Why a number is
the number it is lives next to the number, in the code.

`rxembed` is a constrained / templated conformer embedder: state what a geometry must satisfy (a distance,
an angle, a rigid reacting core, a coordination polyhedron, a hydrogen-bond grip) and get conformers that
satisfy it. It is a library, not a CLI.

---

## The one structural rule

> **Flat is core. The single directory needs extras.**

`ls src/rxembed/` is the documentation. The flat modules at the root are the embedder (`numpy + rdkit`,
nothing else); the one subdirectory, `pipeline/`, is the optional half. `metal/` was flattened to `metal_*.py`
so that the one directory means one thing, and `pipeline/` stays a directory for the same reason: the
asymmetry is the signal. Do not create a second directory to mark a tier.

```python
import rxembed as rx                     # core:      Mol | Isomer -> Conformers      numpy + rdkit
rx.embed(mol, fix={(i, j): 2.0}).minimize()

import rxembed.pipeline as rx            # batteries: str | path | Mol -> Ensemble | EnsembleSet
rx.embed("cat.substrate", contacts="auto").mc().prune().score("gxtb")
```

They are different functions with different signatures, not one call growing powers. The root verb must
never change behaviour with which extras are installed: a same-call-different-result install is the worst
option available. `rxembed.pipeline` re-exports every core name, so it is a strict superset.

**If a change needs a new directory, the abstraction is probably wrong.**

---

## The rules

**KISS.** A capability is a registry row or one argument, not a branch. If a change needs a new code path, the
abstraction is wrong. Dead machinery gets deleted.

**YAGNI.** Build for the case in front of you. No hooks for a caller that does not exist, no parameters
nothing passes, no tiers with one member.

**Delete before you add.** A change that only adds lines is suspect. Say what each change removes. When a
split moves code between files without deleting any, say so plainly: moving is not simplifying, and
sometimes it is still right.

**Separation of concerns.** The core knows nothing about `pipeline/`; perception belongs upstream (xyzgraph,
RDKit), not here. If you need a pipeline import inside a core module, that is a design error: fix the seam,
do not add a lazy import.

**Clarity over cleverness.** A name a newcomer reads correctly beats a name plus a paragraph explaining it.
Rename first, then delete the docstring that was compensating.

**Logic, not pattern matching.** State the rule that makes something true, not a list of the cases you have
seen. An element set like `{7, 8}` is a symptom: the real question was "is this donor's donation axis
conjugated?", which the graph answers for every element at once. A list is always one congener short.
*Test for this:* if adding a new element or case to a set is how you would extend a rule, the rule is pattern
matching; derive it instead. Where a genuine physical constant must be tabulated (covalent radii, census
fold windows), that is data: tabulate it and say where the numbers came from.

**One struct in, one struct out.** `Constraints` is the single thing every builder fills and every stage
reads. Nothing reaches into the pipeline sideways.

**Edit, don't replace.** The ETKDG bounds matrix is *edited* from RDKit's own knowledge-derived bounds.

**Bias the seed, let energy decide.** Constraints bias the *starting* geometry; they are not truth.

**Fail loud, not silent.** A calculator that produces no energies raises; never a silent fallback. Errors
say what the user can do about them. A warning no action can resolve is noise, and a failure that returns a
plausible-looking result silently is a defect.

---

## Comments and docstrings

One-line imperative summary, then only the non-inferable specifics: what it does, what it guarantees, and the
one caveat needed to use it correctly. `ruff D` (numpy convention) enforces the shape, not the content.

Comment the load-bearing why: a chemistry decision, a measured result, a rejected alternative. Never
restate the code.

Do not mass-delete chemistry prose. A comment recording a measurement (`all-pairs SPY: 4.64° → 4.66°, no
gain`) is what stops a settled question being re-litigated. Delete the hedging around it; keep the numbers.

Explain a thing once. The module that owns it carries the explanation; everywhere else names it and points.
Three copies of the constraint vocabulary is how they drift.

A log line is the symptom, in one line, under ~110 characters. The remedy belongs in the docstring. When you
edit one, check the `%` placeholders still match the argument count: logging swallows a mismatch silently.

Name a fixture, structure or constant for the chemistry, never for a person or the project it came from. If
you cannot name the chemistry, you do not yet know what the fixture tests, and never invent chemistry to
justify a name.

No em-dashes, no bold-for-emphasis, no CAPS-for-emphasis. State the fact and stop.

---

## Evidence

**Measure, do not assert.** A claim about behaviour needs a number and the command that produced it.
Distinguish measured from inferred, and say when you could not establish something.

**Beware the null measurement.** Verify your test exercises the code path it names. This project has produced
false "no regression" results by measuring a table through a path that never reads it, and a notebook that
reported a constraint satisfied because `frozen=` excluded the very atoms it asked about.

**Scope a cleanup from the distribution, not the aggregate.** A ratio against another codebase says something
differs, not what or how much. Count the shape first: a prose cleanup was twice mis-sized from a total before
a histogram showed 83% of the comments were already one or two lines.

**A green suite is not coverage.** 284 tests passed while `rx.embed()` returned seeds violating their own
constraints by 48°. When you fix a defect the suite missed, add the test that would have caught it. The
decisive triage signal for a test is mutation: break the code it names; if it still passes, it asserts
nothing.

**Numbers come from `benchmark/`.** 45 measured structures, a committed baseline, one command. Re-run it after
any change to `bounds.py`, `mechanisms.py` or the `metal_*` stack, and argue from the table.

## The development loop

Every non-trivial change runs it:

```
assess → plan → implement → adversarial robustness review → edge-case review → reassess → regress
```

- **Adversarial subagent reviews.** After implementing, spawn independent reviewers whose job is to *break*
  the change. Ask them to refute, not confirm; default to REFUTED when a key number cannot be reproduced.
- **Give an agent only the context its task needs.** If it needs more, that is a separate task for a fresh
  agent.
- **Run tests alone.** Review agents write to the tree despite instructions, and a concurrent run has produced
  a spurious failure here before. Use `isolation: "worktree"` for any agent that might edit, and `git diff`
  before believing a failure.
- **Agents share one scratchpad.** Name scratch files distinctively: a generic `cmp.py` has been silently
  overwritten by a concurrent agent here, and the stale script then ran under the original name.
- **Proof lives in notebooks.** New capability is demonstrated in `examples/*.ipynb` with the real inline
  API, no hidden wrappers, each gated by `rx.geom_check.check`. Clear outputs
  (`jupyter nbconvert --clear-output --inplace 0*.ipynb`) before committing.

---

## Gates

| gate | what it protects |
|---|---|
| `just test` | the suite. It must stay green. |
| `tests/test_init.py` | the tier rule, by AST walk: no core→`pipeline` edge, no absolute self-reference in core, nothing outside stdlib + numpy + rdkit, at any nesting depth. It carries its own bite check |
| `tests/pipeline/test_init.py` | every optional import in `pipeline/` is guarded and names the extra that installs it |
| `benchmark/run.py` | coordination fidelity against 45 known geometries, versus a committed baseline |

**Three kinds of test, and a fourth that does not belong here.** Unit (one module, a contract) and
integration (the chain, end to end) belong; measurement does not. A flag rate, a percentage, an average, or a
"better than before" comparison establishes a number, and numbers live in `benchmark/`. Underneath a measurement there is usually a contract worth asserting cheaply; keep that,
and prefer a bound the code itself declares over a threshold copied out of a measurement.

---

## Layout

**The tier rule.** A module at `src/rxembed/` may import `numpy`, `rdkit` and its siblings, and nothing else.
A module needing more goes in `pipeline/`, guards its import with `try/except ImportError` **at the point of
use** naming the extra, and gets that extra in `pyproject.toml`. The one core exception is `metal_sphere.py`'s
scipy solver, a rarely-reached fallback that guards its own import and degrades with a message, which is why
`sphere` is a *core* extra. Do not grow a second one without the same shape.

**Imports.** Core modules import siblings relatively, single-dot (`from .metal_core import …`), and never
name `rxembed` absolutely, so the core stays relocatable as a unit. `pipeline/` reaches core absolutely
(`from rxembed.relax import …`), so the direction of every edge reads at a glance.

**Tests mirror the source, exactly.** One test module per source module including `__init__.py`, named for
it, so `ls tests/` is only `test_*.py`, the listing is the coverage map, and `test_init.py` pins the mirror
in both directions. No `conftest.py` and no shared helper module: a test needing an extra carries its own
predicate (`@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")`), and a
helper two files both need is duplicated. One that fails on a base install rather than skipping is the
optional tier leaking into the core.

A cross-cutting test goes in the file for the module whose behaviour it pins, not in a new file. That rule
had drifted into five extra files before it was enforced.

**Where a new thing goes.**

| adding… | goes in |
|---|---|
| a new constraint kind | a `Constraints` field + a `Mechanism`, in the root |
| a new coordination shape | a `POLYHEDRA` row in `metal_polyhedron.py` |
| a new NCI contact type | a `KINDS` row in `pipeline/nci.py` |
| a new search backend / calculator / QA check / input format | `pipeline/{search,calculators,geom_check,perceive}.py` |
| a new optional dependency | an extra in `pyproject.toml` + a guarded import at the point of use |
| a measured number | `benchmark/` |

---

## Toolchain

uv / ruff / ty / pytest / pre-commit / just. `just check` = lint + type + test; `just setup` = `uv sync` +
`pre-commit install`.

**A bare `uv sync` is the full dev environment.** `uv sync` never installs an extra, so the `dev` dependency
group self-references `rxembed[all]`; that one line is what stops the habitual command leaving you with a
suite full of ImportErrors. Nothing passes `--all-extras`: the flag would mask a regression in the default.
(`[tool.uv] default-extras` is not a key uv accepts, measured on uv 0.11.32.) The base tier is
`uv pip install .`, or `uv sync --no-default-groups`; neither reads the group.

Extras: `search select score perceive nci viz sphere all`. Version is single-sourced from `pyproject.toml`
via `importlib.metadata`; never hardcode a second copy.
