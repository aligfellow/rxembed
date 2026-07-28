# Working rules

Read `CLAUDE.md` for what rxembed *is*. This file is how we change it.

## The five rules

**KISS.** A capability is a registry row or one argument, not a branch. If a change needs a new
code path, the abstraction is wrong.

**YAGNI.** Build for the case in front of you. No hooks for a caller that does not exist, no
parameters nothing passes, no tiers with one member. Speculative generality is the main source of
the mess we are undoing.

**Delete before you add.** A change that only adds lines is suspect. Say what each change removes.
When a split moves code between files without deleting any, say so plainly — moving is not
simplifying, and sometimes it is still right.

**Separation of concerns.** The kernel (bounds, constraints, mechanisms, ff, sphere, io) knows
nothing about the shell (pipeline, dedup, refine, nci, viz). Perception belongs to xyzgraph, not
here. If you need a shell import inside the kernel, that is a design error — fix the seam, do not
add a lazy import.

**Clarity over cleverness.** A name a newcomer reads correctly beats a name plus a paragraph
explaining it. Rename first, then delete the docstring that was compensating.

**Logic, not pattern matching.** State the rule that makes something true, not a list of the cases
you have seen. An element set like `{7, 8}` is a symptom: the real question was "is this donor's
donation axis conjugated?", which the graph answers for every element at once. A list is always one
congener short — xyzgraph's metal allowlist was extended P → As → Sb and still dropped Bi; the
cross-link rule that replaced it is structural (ring topology + relative distance) and needs no
element table at all.

Test for this: if adding a new element/case to a set is how you would extend a rule, the rule is
pattern matching. Derive it instead. Where a genuine physical constant must be tabulated (covalent
radii, census fold windows), that is data — tabulate it and say where the numbers came from.

## Comments and docstrings

One-line imperative summary; then only the non-inferable specifics. `ruff D` (numpy) enforces the
shape, not the content.

Comment the load-bearing **why**: a chemistry decision, a measured result, a rejected alternative.
Never restate the code.

Do **not** mass-delete chemistry prose. A comment recording a measurement (`all-pairs SPY:
4.64° → 4.66°, no gain`) is what stops a settled question being re-litigated. Delete the hedging
around it, keep the numbers.

## Interface

Public API is `rx.embed(...)` and the stages that chain off it. Every capability is one more
argument on that chain. Errors say what the user can do about them — a warning that no action can
resolve is noise, and a failure that returns a plausible-looking result silently is a defect.

## Evidence

**Measure, do not assert.** A claim about behaviour needs a number and the command that produced it.
Distinguish measured from inferred, and say when you could not establish something.

**Beware the null measurement.** Verify your test exercises the code path it names. This project has
produced false "no regression" results by measuring a table through a path that never reads it.

**A green suite is not coverage.** 284 tests passed while `rx.embed()` returned seeds violating
their own constraints by 48°. When you fix a defect the suite missed, add the test that would have
caught it.

## Gates

`just test` must stay green. The golden bit-identity harness (`tests/golden/`) gates any change to
the bounds pipeline. Frozen-core distance assertions must hold.

Run tests **alone** — review agents write to the tree despite instructions, and a concurrent run has
produced a spurious failure here before. Use `isolation: "worktree"` for any agent that might edit.

Agents share one scratchpad directory. Name scratch files distinctively — a generic `cmp.py` has
been silently overwritten by a concurrent agent here, and the stale script then ran under the
original name and produced plausible-looking output.

## Subagent tasks

Give an agent only the context its task needs. If it needs more, that is a separate task for a fresh
agent, whose finding comes back as a short written result — added to the handoff only if it changes
the plan.

Ask agents to refute, not confirm. Default to REFUTED when a key number cannot be reproduced.
