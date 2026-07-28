# `rdkit_embed` — extraction & north-star plan

The settled design from the API dialogue, and the in-place migration to reach it. Done here in `rxembed`,
but shaped so the `rdkit_embed/` subpackage can graduate to a standalone repo later (hence its own README +
pyproject). The models to match for "clean, intuitive": `../openconf` (one front door + presets + a config
escape hatch; a thin `api.py` over a hidden engine) and `../ML-FSM` (lean deps, one runnable example,
docstrings-as-docs).

## Principles (the bar — enforce these)
1. **One contract.** `Mol` or `Isomer` (+ the verbs `fix`/`constrain`) in; a `Conformers` out. Every consumer
   speaks exactly this — nothing reaches into internals.
2. **KISS / no over-abstraction.** *One* result type (`Conformers`), not a hierarchy. *Three* verbs, not a
   config DSL. `Constraints` is an internal noun the user never fills. A capability is an argument, not a class.
3. **RDKit-native, no lock-in.** We ride RDKit's ETKDG bounds + UFF; `embed`/`minimize` mirror
   `EmbedMultipleConfs`/optimize; `.mol` is a real RDKit `Mol`. Drop to raw RDKit any time.
4. **Perception is the consumer's job.** Parsing (SMILES/xyz/cxSMILES) and NCI *discovery* live in the shells;
   the kernel takes a `Mol`/`Isomer`.
5. **Hide the hard bits.** The metal surrogate, `resolve_core`, `compose`, the mechanisms — all internal.
6. **Flat & legible, every module earns its keep.** Role clear from the top-level listing; no one-file subdirs.
   A module must be a cohesive responsibility with ≥1 real consumer — **fold** a single-consumer helper into its
   consumer (`io.py`→`metal.py`), and a concern used only by the shell (QA result-types/math: `report`, `vecmath`)
   **belongs in the shell**, not the kernel. But **do NOT re-merge** the modules we deliberately split
   (metal/distance/donor_orient/coordination) — 2000-line files are equally un-KISS. Fold the trivial; don't merge
   the substantial.
7. **Comments & docstrings: load-bearing "why" only.** One-line imperative docstring summaries (ruff-D). A comment
   earns its place only when it records a non-obvious decision, a measured result, or a gotcha — **retain those**
   (the intent is the point). Cut narrative/prose/restated-signature verbosity. As a module moves, **tighten in
   flight** — never relocate verbose prose verbatim. (A prior de-verbose pass ran on the current tree, `db12751`;
   the moves re-expose it, and the loop's audit stage re-checks it.)
8. **No AI slop; human-quality prose.** Everything — code, docstrings, comments, commit messages, the README —
   reads like a careful engineer wrote it: terse, specific, no filler, no over-explaining, no restating the
   obvious. **Commit messages carry NO AI co-author trailer** and are short and factual. The review stage rejects
   slop the same way it rejects over-abstraction.

## The public surface (settled)
```python
from rdkit_embed import embed, Isomer, enumerate_isomers      # the everyday surface
from rdkit_embed import Constraints, compose                   # escape hatch (rare)

embed(spec, *, fix=None, constrain=None, n=…, seed=…, prune_rms=0.1, threads=0) -> Conformers
Isomer(mol, geometry, sites)                                   # a known isomer; sites[vertex]=donor atom (surrogate hidden)
enumerate_isomers(mol, geometry) -> IsomerSet                  # the unknown ones
```
- **`Conformers`** — thin result carrying `{mol, ids, cons}`: `.minimize(distance_fc=…) -> Conformers`,
  `.mol` (real RDKit Mol, all conformers), `.ids`, `len()`, `confs[i]`, `.dump(path)` (multi-frame xyz),
  `.xyz(i=None)`. RDKit-native core; thin ergonomic skin.
- **`IsomerSet`** — `.select(...)` / `.filter(...)` on `geometry|arrangement|chirality|index|stereo`, `.summary()`.

One spine everywhere (geometry is a 3-letter code):
```python
embed(mol, fix={(i,j): d}, constrain={(k,l): (lo,hi)}, n=24).minimize().dump("out.xyz")
embed(Isomer(mol, "OCT", sites=[...]), n=24).minimize(distance_fc=1e5).mol
[embed(iso, n=24).minimize().mol for iso in enumerate_isomers(mol, "OCT").filter(chirality="delta")]
```
`chirality` takes a typeable word (`"delta"`/`"lambda"`, normalised internally); the Δ/Λ glyph is display-only
(`.explain()` output). `sites` values are **real atom indices** (what perception hands you), not positions into a
padded donor list.

## Two collapses this bakes in (less code, not more)
- **`template` → `fix`.** A template graft is `fix` with coordinates sourced from a reference — a **frozen-core
  Kabsch graft**, *not* a coordMap (RDKit's coordMap constrains during the DG; `fix` grafts the exact core onto
  the seed post-embed and holds it with `AddFixedPoint`). `resolve_core` **drops its `template` arg**; the
  reference→coords match becomes a tiny shell resolver
  (`_template_to_fix`). The "template doesn't compose with metal" refusal is **deleted** — `fix`'s existing
  `frozen`/`core_frozen` back-off already handles a partial *or* whole-sphere pin gracefully.
- **`contacts` → `constrain`.** NCI *discovery* (`nci_candidates`/`Contact`/`KINDS`) stays in `rxembed`; the shell
  folds a `Contact` into `constrain=` windows before calling the kernel. The kernel's vocabulary is just
  `fix`/`constrain` — no `nci` import, no `contacts=` keyword.

---

## MOVE 1 — `rdkit_embed/` grows the surface (additive; golden bit-identical)
Everything here is a **relocation of existing logic** behind the new names — the physics is untouched, so
`tests/golden/` must stay 23/23.

1. **Isomer enumeration in.** Move `rxembed/isomers.py` (enumerate_isomers, IsomerSet, `_isomers_for_geometry`,
   `_distinct_orderings`/`_chelate_span_ok`, `_prepare_spectators`, `_input_ordering`, `_order_label`,
   `_select_geometries`, `_number_shared_labels`, `_resolve_center`, …) → `rdkit_embed/isomers.py`. Sever its two
   shell deps: make `enumerate_isomers` **Mol-based** (drop the `rxembed.inputs` string branch; parsing stays
   shell), and pull the ligand-stereo load-in in (2).
2. **Split `stereo.py`.** `enumerate_unassigned` + `_build_enumeration_graph` + `_coordination_locked_double_bonds`
   + `_lock_double_bond` + `_stereo_label` + `_mode` → `rdkit_embed/stereo.py` (species load-in, embed-side).
   `signature` + `satisfies_spec` (the preserve/fingerprint gate) **stay** shell.
3. **`Isomer(mol, geometry, sites)` constructor.** `sites[vertex] = donor atom index` (a list, or a
   `{vertex: atom}` dict) — real atom indices, which is what perception produces; the constructor maps them to
   the internal `order` (positions into the padded donor list) itself. Wrap today's `_build_isomer` dance
   (derive metal + donors from the Mol's DATIVE bonds → surrogate → pad to `n_sites` → seat by the sites map →
   `coordination` → `chirality_of`/`_order_label`). Add `Isomer.coordination() -> Constraints`. The surrogate
   lives here and only here.
4. **The `Conformers` result + high-level `embed`.** New `embed(spec, *, fix, constrain, n, …) -> Conformers`:
   `cons = compose(spec.coordination() if Isomer else Constraints(), resolve_core(mol, fix, constrain))` →
   `_dg_embed` (today's `bounds.embed`, unchanged) → `_graft_frozen` → wrap in `Conformers(mol, ids, cons)`.
   `Conformers.minimize()` calls today's `restrained_uff` against the carried `cons`. `.mol` restores the real
   metal. Absorbs `dispatch._bind_substrate`/`_graft_frozen`/`_embed_isomer`/`_encounter_bounds` (moved down).
5. **`resolve_core` loses `template`.** Keep `fix`/`constrain` only.
6. **Expose** in `rdkit_embed/__init__.py`: `embed`, `Isomer`, `enumerate_isomers`, `IsomerSet`, `Conformers`,
   `Constraints`, `compose` (keep `resolve_core`/`restrained_uff` importable as the escape hatch). Update
   `test_import_hygiene.py` `_KERNEL`; confirm the numpy+rdkit closure holds.
7. **Flatten & rename (Move 1.5 — see "Package layout" below).** Do it while things are already relocating.

*Transitional shims: re-export the moved names back into `rxembed.isomers`/`rxembed.stereo` so the old shell
paths keep working until Move 2 switches them. Gate: golden 23/23, full suite 383/2.*

## MOVE 2 — `rxembed/` shrinks to parse → discover → wrap → pipeline
- **`dispatch.py`** loses `_graft_frozen`, `_bind_substrate`, `_embed_isomer`, `_coordination_choices`,
  `_frozen_core_ref`, `_encounter_bounds` (moved). Keeps `_normalize` (parse), `_auto_contacts_embed` (NCI
  *discovery* routing), `_nci_windows` (folds `Contact` → `constrain`), a small new `_template_to_fix`, and the
  `Ensemble` wrap. **Deletes** the `template=/metal` refusal. `_embed_dispatch` becomes: parse → (discover NCI →
  constrain) → `rdkit_embed.embed`/`enumerate_isomers` → `Ensemble`.
- **`isomers.py`** — deleted from shell; usages import `enumerate_isomers` from `rdkit_embed` (one canonical
  name; the `rx.metal` alias is dropped in favour of `rx.enumerate_isomers`, or kept only as a documented alias
  if you prefer the noun — decide once).
- **`stereo.py`** — keeps `signature`/`satisfies_spec`; `_stereo_expand` calls the kernel `enumerate_unassigned`
  + the shell gate.
- **`pipeline.py`** — `Ensemble` **extends `Conformers`** (base = kernel `{mol, ids, cons}` + `.minimize`/`.mol`/
  `.dump`), adding `.mc`/`.prune`/`.score`/`.optimize`/`.best` + tag-aware dump. One result model, layered.
- **`nci.py`** — unchanged.

*Gate: full suite 383/2, golden 23/23, import-hygiene green. `rx.embed(...).mc().prune().score()` behaves
identically — only thinner underneath.*

## MOVE 3 — consumers re-point onto the surface
- **OIN-SMILES** `oin_adapter/reconstruct.py`: `_build_isomer`'s ~40-line private dance → `Isomer(mol, geometry,
  order)`; `rx.embed(iso).minimize()` → `rdkit_embed.embed(iso, n=n).minimize(distance_fc=1e5)`. Drops every
  private reach-in and its `import rxembed` — depends on **`rdkit_embed` only** (this is `rxembed-seam`). Its
  `slot_to_vertex` and `_accept` gate stay OIN's. *Gate: re-run the 144 corpus; 72.9% / median 0.211 Å unchanged.*
- **rxnts** consumes `embed(Isomer(...), fix={reacting core}).minimize().mol` — a metal TS with a frozen core,
  returning a connected, correctly-charged complex. Its Ask-1 (M–L bonds + charges on the returned Mol) is
  already shipped (`521505d` + the charge-restore fix). **Confirm rxnts's actual call sites** map to
  `Isomer`+`fix` before committing (seen via its wishlist, not its code).

## Sequencing, gates, reversibility
1. **Move 1** (kernel, additive, with re-export shims) — golden bit-identical + suite green; old + new both work.
2. **Move 2** (shell switches to the new door, delete shims) — suite 383/2 + golden identical.
3. **Move 3** (OIN, then rxnts) — corpus / rxnts re-validated.

Each move is independently committable and golden-gated. Golden bit-identity is the safety net throughout —
this migration **moves and renames; it does not change physics**. If a step moves golden, it changed something
it shouldn't have.

## The per-step loop (agent-based, fresh context — every Move runs this)
**Every stage is a fresh-context agent** with a narrow brief; the orchestrator only sequences and folds their
short reports — its own context stays low, and one stage's exploration never pollutes the next. No step (Move 1,
1.5, 2, 3, add-geometries) is *done* until it clears the loop; findings fold back before the next step starts.
It's the CLAUDE.md validation discipline, made concrete and pointed at KISS / naming / location / pipeline /
clarity / RDKit-native feel.

1. **Implement** (agent) — the change, one commit, tightening any docstring/comment it touches in flight (principle 7).
2. **Break it** (agent, adversarial) — its only job is to break the change: edge inputs (multi-metal, haptic,
   vacant-site pockets, undefined ligand stereo, tight chelates, low-symmetry/degenerate geometries, a bare Mol
   with no donors, a malformed `sites`) + the documented failure modes; every failure must be **loud, never silent**.
3. **Review — a fan-out of independent lenses, each its own fresh-context agent** (findings ranked, then folded):
   - **KISS / YAGNI** — anything over-built, speculative, or generalised past a real caller? Every module/class/arg
     earns its keep; cut what isn't needed *yet* — no framework, no config DSL, no unused hooks.
   - **Architecture · naming · location** — each module a cohesive role with ≥1 real consumer; nothing misplaced
     (the `report`/`vecmath` lesson); every moved/renamed symbol reads its role (dev-facing).
   - **User & dev clarity** — the surface (`embed(...).minimize().mol`) stays one obvious thing; the pipeline is
     intuitive; a newcomer can navigate the modules; docstrings are load-bearing-why, prose cut (principle 7).
   - **RDKit-native styling** — reads like RDKit to an RDKit user: `Mol` in / conf-ids-or-`.mol` out, ETKDG-shaped
     params, no surprising object where a `Mol` is expected, idiomatic names — measured against `../rdkit`.
4. **Test / regress** (agent) — golden **23/23 bit-identical** (this is a MOVE), full suite **383/2**,
   import-hygiene; Move 3 also the **144-corpus** (72.9% / median 0.211 Å unchanged). A gate, not a formality.
5. **Reassess & fold** — fold the break + all review findings, re-run step 4, and only then proceed. Update memory
   (what changed + why) so the next session inherits it.

Executes as a fan-out per step — implement → (break ∥ the four review lenses) → test → synthesize.

## Final step — the README, written last (after Move 3)
The package README is written **against the shipped surface**, so it never documents an API that isn't there.
Promote `docs/rdkit_embed-README.draft.md` → `src/rxembed/rdkit_embed/README.md`, run it through the **User & dev
clarity + RDKit-native** review lenses (openconf formula: tagline → install → ≤10-line quick-start → the small
API → how-it-works), and **verify the quick-start actually runs**. No slop; terse.

## Geometry codes & sites (simple — one flat list of shapes)
**Every shape is a 3-letter code with its own vertices. No parent/child, no "drop a vacancy."**
`LIN TPL TSH SPL TET SEE TBP SPY TPY OCT PBP SQA …` — `SEE` *is* a seesaw, `TSH` *is* a T-shape; each code carries
its own vertex directions. (We do NOT derive low-CN shapes by removing a vertex from a parent — that's ambiguous:
`TBP` minus one vertex could be a seesaw or not.) Adopt OIN's spellings for the shared codes
(`OCT/TBP/SPL/SPY/TET/PBP/SQA/LIN/TPL`) so `oin_adapter`'s `OIN_TO_RX` collapses to identity — zero translation.

- **`sites = {vertex: donor atom}`** fills the code's vertices (occupied only).
- **A code bigger than your donor count = a pocket** — one deliberate open site (`OCT` + 5 donors = an octahedron
  with a site for a substrate). That is the *only* place "vacancy" means anything, and it's a choice, not an
  inherited ambiguity.
- **Humans never type a vertex number:** `enumerate_isomers(mol, "OCT").summary()` → `.select(arrangement=…,
  chirality="delta")`. Raw `sites` is for perception-driven consumers (OIN/rxnts) that already have the mapping;
  the code's `vertex_dirs(code)` are exposed for them.
- **Handedness is internal:** `sites` fix the target Λ/Δ, `Isomer` carries `.chirality`, `embed` yields both hands
  (distance geometry is mirror-invariant), you `.filter(chirality="delta")`.

`TSH`/`SEE`/`TPY` are shapes rxembed lacks today — **one `POLYHEDRA` record each** (vertex dirs + angles): a small
"add three geometries" follow-up, cleanly separate from the extraction.

## Package layout — flat, and every module earns its keep (Move 1.5)
Exemplars are flat: prism_pruner (11 modules) and ML-FSM (7) have no subpackages; openconf is flat + **one**
*multi-file* engine subpackage. rdkit_embed today has one-file subdirs (`embed/bounds.py`, `refine/ff.py`) **and**
over-abstraction: a 44-line `io.py` (just `repair_bond_stereo`, one consumer), and `report.py` + `vecmath.py` that
**no kernel code uses at all** — only the SHELL `geometry.py` does. Fix all three: flatten, fold, and evict.

- **Evict (they're the QA gate's, not the engine's):** `report.py` (Violation/GeometryReport) and `vecmath.py` are
  used ONLY by `rxembed/geometry.py`, which is already shell → **move them to the shell** (fold into `geometry.py`
  or a shell helper). The kernel is the embed engine; QA result-types + their math are not engine.
- **Fold single-consumer helpers:** `io.py` → into `metal.py`; `log.py`'s `set_verbose` → into `__init__.py`. Stop
  exposing `io`/`log` as public modules.
- **Un-split:** `constraints/solver.py` + `sphere.py` (one fallback feature) → one `sphere.py`.

Target — **~13 flat modules, no subpackages**, every name a role and every module ≥1 real consumer:

| module | role | from |
|---|---|---|
| `__init__.py` | surface + `set_verbose` | + `log.py` |
| `embed.py` | `embed()` + `Conformers` — front door | new + high-level |
| `bounds.py` | ETKDG bounds-matrix edit (DG seed) | `embed/bounds.py` |
| `relax.py` | `minimize()` / restrained-UFF | `refine/ff.py` |
| `constraints.py` | `Constraints`, `compose`, `resolve_core` | `constraints/base`+`builders` |
| `mechanisms.py` | DG+FF constraint writers | `constraints/mechanisms` |
| `isomers.py` | `Isomer`, `enumerate_isomers`, `IsomerSet` | moved in |
| `metal.py` | surrogate/restore, `repair_bond_stereo`, metal DG/FF | `constraints/metal`+`io.py` |
| `coordination.py` | the coordination builder | `constraints/coordination_builders` |
| `distance.py` | `ml_distance` | `constraints/distance` |
| `donor_orient.py` | donor-orientation holds | `constraints/donor_orient` |
| `polyhedron.py` | `POLYHEDRA` + geometry | `constraints/polyhedron` |
| `stereo.py` | ligand-stereo load-in | moved in |
| `perceive.py` | coordination perception | `coordination.py` (renamed — clashed with the builder) |
| `sphere.py` | optional sphere fallback (scipy) | `solver.py`+`sphere.py` |

Gone from the kernel: `io.py`, `log.py`, `report.py`, `vecmath.py`, and the `embed/`/`refine/`/`constraints/`
subdirs; `report.py`+`vecmath.py` land in the shell by `geometry.py`. No file grows past ~800 lines (metal.py is
NOT re-merged with the modules we split). This is a MOVE — golden bit-identical.

## Still open (small)
- rxnts call sites (confirm before Move 3 — above).
- Whether Move 1.7 (de-nesting) is worth the churn now or deferred.
