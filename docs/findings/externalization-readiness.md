# Externalization readiness: the `rdkit_embed` kernel as a separate package

*Read-only assessment, 2026-07-28, branch `rdkit-embed-kernel`. Question posed by the maintainer:
is the embed kernel (`src/rxembed/rdkit_embed/`) ready to become a SEPARATE, importable, rdkit-style
package used by both rxembed and OIN-SMILES — and where is the complexity/generalisation risk? Every
claim cites `file:line`. Nothing was edited; this doc is the only file written.*

---

## Verdict up front

The kernel is **structurally separable today but not yet consumer-ready as a standalone**. The hard
decoupling is done — 0 kernel→shell import edges, a numpy+rdkit dependency closure, an allowlist test
that locks the boundary (`tests/test_import_hygiene.py`). What is *not* done is the **usability half**:
the public surface (`rxembed/rdkit_embed/__init__.py`) covers the **organic** constrained-embed engine
cleanly, but the **metal** engine — which is the actual OIN drop-in use case — is driven entirely
through submodule reach-ins (several underscore-private) plus the *shell* pipeline. A new consumer
could drive a bare-SMILES or frozen-TS embed from the public exports; they could **not** drive a metal
coordination embed without reaching past the public API and pulling the shell back in. That is exactly
the "complex inner workings, hard to use without the wrapper" risk the maintainer named, and it is
real for the metal path specifically.

---

## 1. The drop-in surface (the crux)

### 1a. What the public surface actually is

`rxembed/rdkit_embed/__init__.py:19-30` exports exactly:

| symbol | role |
|---|---|
| `embed(mol, cons, n, seed, prune_rms, knowledge, threads)` | the bounds-matrix ETKDG embedder → conformer ids (`embed/bounds.py:174`) |
| `restrained_uff(mol, cons, distance_fc, max_iters, conf_ids)` | the FF relax that *enforces* the windows (`refine/ff.py:115`) |
| `Constraints` / `compose` | the one-struct-in/one-struct-out model (`constraints/base.py:24,198`) |
| `resolve_core(mol, *, fix, constrain, template, has_geometry)` | the fix/constrain/template → Constraints resolver (`constraints/builders.py:239`) |
| `ff_energies`, `n_confs` | FF single-points; seed-count heuristic |
| `io`, `log`, `set_verbose` | the xyz-repair leaf, logging |

Note what is **absent**: nothing metal (`surrogate_metal`, `coordination`, `Isomer`, `restore_metal`),
nothing haptic, nothing NCI, no geometry gate. The public surface is the *organic + hand-passed-constraint*
engine only.

### 1b. Minimal real end-to-end usage (organic — this genuinely works from the public API)

```python
from rdkit import Chem
from rxembed.rdkit_embed import embed, restrained_uff, resolve_core, Constraints

# free / flexible
mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
ids = embed(mol, Constraints(), n=10)          # conformer ids written onto `mol`
restrained_uff(mol, Constraints())             # FF cleanup (no windows -> plain UFF/MMFF)

# constrained TS from SMILES (exact distance + angle, index-driven)
mol = Chem.AddHs(Chem.MolFromSmiles("...reacting core..."))
cons, ref = resolve_core(mol, fix={(0, 5): 2.0, (0, 5, 9): 178.0}, has_geometry=False)
ids = embed(mol, cons, n=24)
restrained_uff(mol, cons, distance_fc=1e4)     # the pull that lands the numbers
```

Two things a consumer must know that the signatures do not tell them:

1. **It is two calls, not one.** `embed` only *biases* the constraints (the ETKDG torsion terms can
   override soft bounds — `refine/ff.py:2-6`); `restrained_uff` is what actually enforces them. A caller
   who runs only `embed` gets a seed that can violate its own windows.
2. **The frozen-core Kabsch graft is the caller's job.** `resolve_core` returns `ref = {atom: (x,y,z)}`
   and its own docstring says "The caller edits the bounds matrix from `Constraints`, embeds, **grafts
   `ref`**, then runs the restrained UFF pull" (`constraints/builders.py:243-244`). The 0.000 Å graft
   that CLAUDE.md advertises is done **in the shell** (`embed/dispatch.py`), not by the kernel. From the
   bare kernel a `fix=[atoms]` core is held only at *embedded* coords via `AddFixedPoint`, not restored to
   the exact input geometry, unless the consumer writes the Kabsch superposition themselves.

### 1c. What the rxembed SHELL adds over the bare kernel

The `rx.embed("SMILES", constrain=…).mc().prune()` one-chain wraps the kernel with everything a real user
would otherwise hand-build:

- **Source parsing** — `rxembed.inputs._xyz_to_mol` / `parse_smiles` (xyzgraph metal-aware perception).
  The kernel `embed()` takes a *Mol*; it no longer perceives one (`rdkit_embed/README.md:35-38`).
- **The Ensemble pipeline** — `mc` (openconf search), `prune` (dedup), `score`/`optimize` (real energies),
  `lowest`/`representatives`, and the `EnsembleSet` map-over-candidates. All shell (`pipeline.py`).
- **The metal front-end** — `rx.metal` = `isomers.enumerate_isomers` (shell, `isomers.py:440`): perceives
  the polyhedron, enumerates coordination isomers, builds each `Isomer` + its `Constraints`.
- **The NCI front-end** — `constraints/nci.py` (shell): the KINDS registry → binding-mode `Constraints`.
- **The stereo front-end** — `stereo.py` (shell): undefined-centre enumeration into an `EnsembleSet`.
- **The geometry QA gate** — `rx.geometry.check` (shell, `geometry.py`): the physical acceptance test.
- **The frozen-core Kabsch graft** and **metal restore / M–L reconnect** for real-energy calcs
  (`pipeline.py`, `embed/dispatch.py`).
- **The calculators** — `refine/xtb.py`, `refine/calculator.py` (shell).

So a bare-kernel user gets: *Mol in → biased seed + FF relax out*. Everything from "read my .xyz" to
"search, dedup, score, and tell me if the geometry is chemically sane" is shell.

### 1d. The complexity / generalisation RISK — honestly rated

**Organic constrained embed: LOW risk, genuinely rdkit-style.** `resolve_core` is a legible three-verb
resolver with porting guardrails (element-symbol echo, 1-based-index detection, underdetermined-angle
advisory — `constraints/builders.py:269-322`). A consumer never fills `Constraints` by hand for the
organic case; they pass `fix`/`constrain` and the resolver fills it. `Constraints` itself is a plain
dataclass with an import-time integrity check that makes "new field silently dropped" impossible
(`base.py:191-195`). For bare-SMILES, soft-window, and frozen-TS work the kernel is close to standalone.

**Metal embed: HIGH risk — this is the maintainer's concern, in the flesh.** The one real external metal
consumer, `oin_adapter/reconstruct.py`, does **not** touch a single public export. To build one fixed
isomer it reaches into (all off `__init__`'s allowlist, several underscore-private):

```
_metal.surrogate_metal, _metal._collapse_haptic (private), _metal.Isomer, _metal.n_sites,
_metal.strip_phantoms, _metal.chirality_of, _metal.VACANT,
_cbuild.coordination,
_isomers._order_label (private, and it lives in the SHELL), and in validate.py:
_coord._coordinating (private), _distance.ml_distance, M._APICAL_MIN (private), M._frag_map (private),
M.geometry_for, M.vertex_dirs
```
(`oin_adapter/reconstruct.py:15-17,29-76`; `oin_adapter/validate.py:13-17`). And even then it embeds via
the **shell** `rx.embed(iso, stereo="free").minimize()` (`reconstruct.py:107`), not the bare kernel
`embed()`. So:

- **`Constraints` by hand for a metal is expert-only.** It is 14 fields (`base.py:27-80`) with load-bearing
  co-shipping semantics the type does not express: `metals`/`pulls`/`floors`/`dg_floors` "ship together or
  not at all" (`base.py:54`), and `floors` vs `dg_floors` merge in *opposite* directions (max vs min) for a
  reason that takes a 15-line essay to state (`base.py:198-216`). A consumer cannot fill these correctly
  from the dataclass alone.
- **The metal surrogate is not usable via the public surface.** The C(DG)/Li(FF) swap, the strip-and-restore
  of M–donor bonds, and the oxidation-state restore live in `constraints/metal.py` (`surrogate_metal:255`,
  `restore_metal:286`) and `refine/ff.py:_ff_surrogate:86` — none exported. `restore_metal` is mandatory
  before any real-energy calc (else the wrong total charge, per MEMORY `metal-charge-lost`), and the consumer
  has to know to call it.
- **Perception is needed for the metal path** even though the README says perception left the kernel: the
  coordination *ruler* (`coordination._coordinating`, `metal.geometry_for`) is still consumed to build and
  validate a sphere.

**Bottom line for Part 1:** a new consumer *can* drive the bare kernel for organic / frozen-TS /
hand-passed-window embedding from the documented public surface. They *cannot* drive the metal engine
without (a) reaching past the public API into ~11 submodule symbols, (b) several of them private, and
(c) pulling the shell (`rx.embed`, `isomers`, `geometry`) back in. The metal engine is "complex inner
workings, hard to use without the wrapper" — the risk is confirmed, and it is confined to the metal path.

---

## 2. The OIN drop-in story

### 2a. How `oin_adapter/` consumes the kernel

`oin_adapter/` (4 files) is a working, corpus-validated cxSMILES→embed drop-in: it swaps rxembed's
**engine** behind OIN's **perception** (`oinsmiles.utils.cxsmiles`) and **losslessness/chirality gate**
(`oinsmiles.generation.rdkit_embed._accept`), both reused verbatim (`__init__.py:1-16`). Flow:
`parse_cxsmiles` → `reconstruct._build_isomer` (hand-builds one fixed `Isomer` from OIN's slot map) →
`rx.embed(iso).minimize(distance_fc=1e5)` → keep the first conformer OIN's `_accept` byte-matches
(`reconstruct.py:97-113`). It pins **OIN's** stiffer wall FC (`OIN_DISTANCE_FC=1e5`, `reconstruct.py:24`)
because rxembed's softer default drops otherwise-sane geometries.

Critically, the adapter depends on the **whole `rxembed`**, not a light `rdkit_embed`: it imports
`import rxembed as rx`, `rxembed.isomers`, `rxembed.geometry`, `rxembed.metrics` (all shell) *and* the
kernel submodules listed in 1d. A standalone numpy+rdkit `rdkit_embed` alone would **not** satisfy this
adapter as written — the metal enumeration (`isomers`), the geometry gate (`geometry`), and the metrics
(`metrics`) it uses are all shell.

### 2b. Fidelity result and the `rxembed-seam` gap

Corpus (`docs/findings/oin-corpus-results.md`): **105/144 pass @ 1.0 Å, median RMSD 0.211 Å, no adapter
bug**. The 39 non-passes are OIN-side encode misses (10), by-design scope declines for haptic/UNK (15),
marginal near-misses/strict-gate/Hg edge cases (13), one fixed-path perf timeout. The repoint is correct;
the swap is fidelity-preserving *by construction* (rxembed's `ml_distance`/`_PHYS_COEF` is bit-identical to
OIN's vendored copy — `plan.md:15`).

**The upstream `rxembed-seam` gap** (`oin-integration-recon.md:46-57`, `oin-corpus-results.md:112-127`):
the adapter's perception deps already live in OIN's **stable** `oinsmiles/utils/cxsmiles.py`, but `_accept`
still lives in `oinsmiles/generation/rdkit_embed.py` — a ~600-line vendored copy of an embed engine that
OIN's own backlog marks for deletion as YAGNI. If OIN deletes that file, the adapter breaks. Recommendation
(documented, **not** pushed into the OIN repo): OIN should lift `_accept` (+ `_mirror_x`) into the stable
shell. This is an OIN-repo change on a separate branch (`ali-dev` carries the perception, not `ag-dev`).

### 2c. What it would take to graduate to a real pip `rdkit_embed`

The placeholder `rxembed/rdkit_embed/pyproject.toml` is **inert** — "NOT a uv-workspace member, NOT built,
NOT installed" (line 1); it records the intended shape (numpy + rdkit base, `sphere = ["scipy"]` the one
soft extra — line 22-37). The prerequisites the separability assessment named are **already met**:

- **0 kernel→shell edges**, locked by `tests/test_import_hygiene.py` — and it is now the correct
  **allowlist** of kernel modules (`_KERNEL_PKG = "rxembed.rdkit_embed"`), not the old denylist the
  assessment flagged as blind (test docstring lines 10-18). That T15 fix is done.
- **scipy is declared** as the `sphere` extra, closing the "sphere solver silently dies" trap the
  assessment warned about (`separability-assessment.md:138-140`).

What still stands between "inert placeholder" and "separate + imported here and in OIN":

1. **The metal use case reaches the shell.** OIN's actual consumption pulls `isomers` (enumeration),
   `geometry` (the gate), and `metrics` — all shell. A light `rdkit_embed` covering only the public
   surface would serve the *organic* engine but **not** the metal drop-in that is OIN's whole point. A
   decision is owed on where the metal front-ends (isomer enumeration, coordination perception, the QA
   gate) sit relative to the package boundary — today they are shell, and the adapter depends on them.
2. **Logger names are pinned to `"rxembed.*"` inside the kernel** (`refine/ff.py:17`,
   `constraints/builders.py:35`, `embed/bounds.py:18`, `log.py:16`). `ff.py:17-19` explicitly warns that
   `set_verbose` "would go dark under a standalone `rdkit_embed` package (no longer under 'rxembed')". A
   rename (or a configurable logger root) is required at graduation.
3. **The workspace split was tried and reverted.** `cf338d5` unwound a `packages/` split back to the
   in-place subpackage (`HANDOFF.md:13-15`) — the maintainer deliberately chose in-place "until it
   stabilises." So the packaging step is prototyped, not blocked.

**Smallest real step toward "separate + imported here and in OIN":** promote the placeholder into a real
buildable member (uv-workspace member or its own dir with a live `pyproject`) re-exported by `rxembed`,
carrying the numpy+rdkit closure + the logger-root rename, so `from rdkit_embed import embed, Constraints`
resolves as an actual dependency **without moving any code**. That is small and low-risk and delivers the
organic engine as a genuine separate package. But be honest that it does **not** by itself make OIN's
*metal* drop-in light: the adapter would still import the full `rxembed` for `isomers`/`geometry`/`metrics`
until those front-ends are consciously placed on one side of the boundary.

---

## 3. Doc currency audit

Rated CURRENT / STALE / MISSING. (Findings under `docs/findings/` are point-in-time measurement records,
not onboarding docs; rated as a class with the load-bearing ones called out.)

| Doc | Tracked | Rating | Note |
|---|---|---|---|
| `README.md` (top-level) | yes | **CURRENT for users / STALE on internals** | Onboards a newcomer well: quickstart, verb table, principles, geometry gate. BUT "Package structure" (lines 152-164) still shows the **flat pre-carve layout** (`embed/bounds.py`, `constraints/{base,metal,builders}.py`) with **no kernel/shell split**, and "Requirements" lists sklearn as core. A newcomer reading README alone would not learn the kernel exists. |
| `CLAUDE.md` | yes | **CURRENT** | "Where things live" (lines 146-203) is **fully updated for the carve**: kernel/shell split, `inputs.py`, `isomers.py`, correct `rdkit_embed/*` paths, and notes `plan.md` is un-checked-in. The `review-onboarding.md` O1/O10 findings (stale map, dangling plan.md pointer) are **FIXED here** — that review predates this update. This is now the best internal map. |
| `AGENTS.md` | yes | **CURRENT** | Working rules; timeless. |
| `HANDOFF.md` | yes | **CURRENT** | Status 2026-07-23: T1–T16 DONE, carve-in-place accurate, corpus numbers match, open judgment-calls (sphere keep/delete, builders move, `rxembed-seam`) listed. The live forward-state doc. |
| `src/rxembed/rdkit_embed/README.md` | yes | **CURRENT but thin** | Correctly describes the kernel idea and "Still to do" (io._xyz_to_mol moved = DONE; sphere-solver fate + graduation = open). BUT shows only the *import line* — **no runnable end-to-end example**, and says nothing about the metal path needing shell/private reach-ins. |
| `plan.md` | **no (gitignored)** | **STALE / local-only** | Dated 2026-07-16 "Decision-ready" OIN-unification engineering doc; largely superseded by HANDOFF (which reports it done). Absent on a fresh clone. |
| `examples/README.md` | yes | **CURRENT (assumed)** | The 8-notebook tour index; user-facing, not kernel-consumer facing. |
| `docs/findings/separability-assessment.md` | yes | **CURRENT (as record)** | The definitive import-graph analysis; its packaging recommendations are now partly executed (allowlist test, scipy extra). |
| `docs/findings/oin-integration-recon.md`, `oin-corpus-results.md`, `oin-corpus-run-recipe.md` | yes | **CURRENT** | The OIN integration + fidelity record; matches the adapter code as it stands. |
| `docs/findings/adapter-repoint-scope.md` | yes | **CURRENT as history / superseded in effect** | Describes the *pre-fix* stale-symbol state; the fixes it specifies are now applied in the adapter code, so it reads as a completed to-do. |
| `docs/findings/review-onboarding.md` | yes | **PARTLY SUPERSEDED** | Its top finding O1 (CLAUDE.md stale map) and O10 (plan.md pointer) are now FIXED on disk; O2–O9 (naming, mutation-contract docstrings) may still stand. Flag when citing. |
| other `docs/findings/*` (≈30) | yes | **CURRENT as records** | Physics/measurement logs (coplanar cap, donor orientation, sphere solver, polyhedra, corpus regressions). Anti-re-litigation value; not onboarding. |
| `WORKLOG.md`, `PLAN_rxnts_integration.md`, `overnight/*.md` | **no (untracked)** | **STALE / local** | Work-in-progress logs; `PLAN_rxnts` is "redundant for OIN" per HANDOFF. Absent on a clone. |

**What a newcomer reads first, and is it good enough?**
- *As a project contributor:* README.md → CLAUDE.md. **Good enough** — they can trace
  `rx.embed(...).mc().prune()` end-to-end; CLAUDE.md's map is now accurate.
- *As a kernel consumer (the externalization lens):* `rdkit_embed/README.md` + the public `__init__`.
  **Not good enough** — no worked end-to-end example, and the metal reality (not in the public surface;
  needs shell + private symbols) is undocumented.

**Single biggest doc gap for onboarding (externalization lens):** there is **no consumer's guide to the
bare kernel** — a minimal runnable `Mol + Constraints → conformers` recipe that states the two
load-bearing facts (embed-then-restrained_uff is two steps; the frozen-core Kabsch graft is the caller's
job) AND is honest that the **metal path is not yet part of the public surface**. The kernel README gestures
at the idea but ships only an import line. Secondary gap: README.md's "Package structure" still predates the
carve and should mirror CLAUDE.md's kernel/shell split.

---

## Summary (verdict)

1. **Readiness: structurally separable, not yet consumer-ready standalone.** The decoupling is done
   (0 kernel→shell edges, numpy+rdkit closure, allowlist import-hygiene lock, scipy declared as an extra);
   the usability half is not.
2. **The public surface** (`embed`, `restrained_uff`, `Constraints`/`compose`, `resolve_core`, `ff_energies`,
   `n_confs`) covers the **organic / frozen-TS / hand-passed-window** engine cleanly and rdkit-style.
3. **Top complexity risk #1 — the metal engine is not on the public surface.** The one real external metal
   consumer (`oin_adapter`) reaches into ~11 submodule symbols (several underscore-private) and still embeds
   through the **shell** `rx.embed().minimize()`. Metal = "complex inner workings, hard to use without the
   wrapper," confirmed.
4. **Top complexity risk #2 — `Constraints` is expert-only for metals.** 14 fields with unstated
   co-shipping and opposite-direction-merge semantics (`floors`/`dg_floors`); fine via `resolve_core` for
   organics, not hand-fillable for a metal.
5. **Two smaller graduation gotchas:** loggers pinned to `"rxembed.*"` inside the kernel (`set_verbose` goes
   dark standalone — `ff.py:17` flags it); the frozen-core Kabsch graft is done shell-side, not by the kernel.
6. **OIN drop-in works today** (105/144 @1.0 Å, median 0.211 Å, no adapter bug) but consumes the **whole
   `rxembed`** (shell `isomers`/`geometry`/`metrics` + kernel), not a light `rdkit_embed`. Upstream
   `rxembed-seam`: OIN should lift `_accept` off its doomed vendored engine (documented, not pushed).
7. **Smallest real step toward "separate + imported here and in OIN":** promote the inert placeholder
   pyproject into a real buildable member re-exported by rxembed, + rename the kernel logger root — *no code
   moves*. Delivers the organic engine as a genuine package; does **not** yet make the metal drop-in light
   (isomers/geometry/metrics are still shell).
8. **Biggest doc gap:** no consumer's guide to the bare kernel (a runnable Mol→Constraints→conformers recipe
   + the honest metal caveat). CLAUDE.md's internal map is now current; README's package-structure is not.
</content>
</invoke>
