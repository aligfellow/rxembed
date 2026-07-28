# rxembed — ONBOARDING / READABILITY / NAMING review

*Lens: could a new dev, handed this repo + `CLAUDE.md`, trace `rx.embed(source).mc().prune().score().lowest(n).optimize()`
end-to-end in an afternoon and know where to look? Scope: the user-facing shell — `pipeline.py`, `embed/dispatch.py`,
`__init__.py`, `inputs.py`, plus the constraint spec surface (`rdkit_embed/constraints/{base,builders}.py`) a newcomer must
read to understand the verbs. READ-ONLY. Extends `overnight/maintainability.md` (KISS review F1–F17, HEAD `ce59f34`) — I do
not restate its findings; where they've since been fixed I say so.*

---

## 0. Context: what changed since the prior review (so the orchestrator doesn't re-raise closed findings)

The prior KISS review's two biggest findings are **already fixed in code** — the debt has moved *from the code into the
onboarding doc*:

- **F1 (the `Constraints` hand-copy footgun) is CLOSED.** `Constraints.copy(**overrides)` and `compose(*parts)` now exist
  (`rdkit_embed/constraints/base.py:87,198`), both **field-driven** via `_CLONE`/`_MERGE` registries with an *import-time*
  sync check (`base.py:191-195`) that raises if a new field lacks a clone/merge policy. `relaxed()`, `_settle_seeds`, and the
  substrate merge all route through `copy`. The "new field silently not carried" class of bug is now structurally impossible.
  This is exemplary — a newcomer can trust it.
- **F2 (the 2042-line `metal.py` monolith) is SPLIT.** The kernel now separates `constraints/{metal,polyhedron,sphere,solver,
  mechanisms,donor_orient,distance}.py` + `coordination.py`. The concerns the prior review wanted teased apart are teased apart.

**The through-line of this review:** the *code* got substantially cleaner and moved into a `rxembed.rdkit_embed` kernel
subpackage, but **`CLAUDE.md` — which doubles as the onboarding doc — was not updated to follow.** A newcomer's single most
authoritative map now points at files that no longer exist. That is O1, and it is the finding that matters most.

**Strengths worth preserving (do not "clean up"):** docstrings here overwhelmingly say the *why*, not the signature
(`Ensemble.mol` property, `_relax_into_windows`, `compose`'s floors-vs-dg_floors essay). The `_mol`/`.mol` surrogate split is
documented at exactly the depth it needs. `builders.py::resolve_core` is a genuinely legible three-verb resolver with
porting-error guardrails (element-symbol echo, 1-based-index detection). The design *is* onboardable — the friction is in
stale signage and a handful of overloaded names, not structure.

---

## 1. RANKED FINDINGS

### O1 — `CLAUDE.md` "Where things live" (and its inline path references) predate the `rdkit_embed` carve-out **[high · safe-fix]**
`CLAUDE.md` §"Where things live" + §"Real-energy caveats" + the constraint-spec blockquote.

**Problem.** The doc a newcomer is explicitly told to onboard from points at paths that have moved, and never mentions the
kernel/shell split at all. Verified against HEAD `003084d`:

| CLAUDE.md says | Reality |
|---|---|
| `src/rxembed/constraints/` — `base.py`, `metal.py`, `builders.py` | all three now `src/rxembed/rdkit_embed/constraints/…`; only `nci.py` stays in the shell |
| `src/rxembed/embed/` — `dispatch.py` (…`_xyz_to_mol`…), `bounds.py` | `_xyz_to_mol` is in **`inputs.py`** (moved out, commit `fb2bd40`); `bounds.py` is `rdkit_embed/embed/bounds.py` |
| `ff._ff_surrogate`, `constraints/metal.py::_orient_donor`/`_coplanar_donor` | `rdkit_embed/refine/ff.py`; `_orient_donor`/`_coplanar_donor` now live in `rdkit_embed/constraints/donor_orient.py` |
| (no entry) | `isomers.py` — the `rx.metal` entry point (`enumerate_isomers`, `Isomer`) — is unlisted despite the metal prose leaning on it |
| (no mention anywhere) | the entire `rxembed.rdkit_embed` **kernel/shell split** — the biggest architectural fact in the repo |

A newcomer told "`constraints/builders.py::resolve_core` is the resolver" opens `src/rxembed/constraints/` and finds only
`nci.py`. The map is actively misleading, not merely incomplete. `MEMORY.md` already records "the rdkit_embed handoff doc is
STALE" and "geometry.py is SHELL" — that knowledge simply never reached `CLAUDE.md`.

**Fix (doc-only).** Rewrite "Where things live" around the **kernel (`rxembed.rdkit_embed.*`, numpy+rdkit DG/FF engine) vs
shell (`rxembed.*`, user-facing pipeline + QA gate + NCI + isomers)** boundary, with a 2–3 line orientation ("import as
`rx`; the pipeline you call lives in the shell; the DG/FF mechanics it drives live in the kernel — you rarely need to open
the kernel"). Fix every path in the table above. Add the `isomers.py` and `inputs.py` rows.

---

### O2 — `mc` names the *metal-context* local throughout `pipeline.py`, colliding with `.mc()` (Monte-Carlo) and the `_mc` module **[medium · safe-fix]**
`pipeline.py:685,714,861,905,932,944,992,1251,1630` (local/param `mc = self._metal`); vs `pipeline.py:37` (`from .embed import
mc as _mc`) and `pipeline.py:513` (`def mc(...)`).

**Problem.** In the one file whose headline verb is `.mc()` (Monte-Carlo search), the name `mc` is used everywhere as a local
variable holding the **metal context** (`mc = self._metal  # capture before restore`). So `_relax_and_record(self, distance_fc,
mc)` and `_drop_bad_geometries(self, mc)` take a *metal ctx* named `mc`, three methods below the *Monte-Carlo* method also
called `mc`. Compounding it, `_metal` is simultaneously the **imported kernel module** (`pipeline.py:26`) and the **`Ensemble`
field** (`pipeline.py:466`), so `_metal.restore_metal(...)` (module) and `self._metal` (field) sit in the same methods. A
newcomer holds three meanings of "mc/metal" at once. Note `embed/dispatch.py` already uses the unambiguous `metal_ctx` for the
identical object — the two shell files disagree.

**Fix (mechanical rename, grep-scoped).** Rename the metal-context locals/params `mc` → `mctx` (or `metal_ctx`, matching
dispatch). Optionally rename the module alias `_metal` → `_metalmod` so the module and the field stop sharing a name. No
behaviour change; the field's positional slot in `Ensemble(...)` is untouched.

---

### O3 — `prune()` silently runs `minimize()` (which *moves atoms*) but its docstring says only "in place" dedup **[medium · safe-fix]**
`pipeline.py:1323-1345` (`prune`); same pattern in `representatives` (1503), `cluster` (1427), `landscape` (1436).

**Problem.** `prune()`'s docstring is "Deduplicate by geometry, in place." Its **first action** is `self.minimize()`
(line 1345), which FF-relaxes every conformer — moving geometry, dropping clashed conformers, restoring the metal. A newcomer
reasonably expects `prune` to only *select among existing* geometries; instead it silently relaxes them first. The behaviour is
documented in `wrap`'s docstring ("prune()/representatives()/lowest() will run one FF minimize() first (which **moves atoms**)")
and `score`/`optimize` state it — but the verbs that *do* it don't say so at their own call site, which is where a reader looks.

**Fix (doc-only).** One line in `prune`'s (and `representatives`'s) docstring: "Relaxes once via `minimize()` first (idempotent
if already minimized), then dedups — so a raw embed is FF-settled before comparison." This is the same idempotent-minimize note
the looking-verbs already carry; extend it to the mutating ones.

---

### O4 — The `Ensemble` "Mutation contract" docstring enumerates three buckets but omits prominent verbs — `filter` (in-place), `score`/`optimize` (return-new) **[medium · judgment-call]**
`pipeline.py:452-458` (the contract) vs `filter` (1291, in-place), `score` (1091, new), `optimize` (1164, new),
`select_stereo` (1456, new).

**Problem.** The task's own question — "is the mutation contract obvious or a footgun?" The docstring teaches a clean
three-bucket model: *build-in-place* {mc, minimize, prune}; *derive-new* {lowest, representatives, align}; *looking*
{view, compare, landscape, cluster, binding_modes}. But **`filter` mutates in place** (drops reacted conformers, returns self)
and is a headline verb in CLAUDE.md (`.filter('connectivity')`) — yet it appears in no bucket. `score`/`optimize`/`select_stereo`
return new and aren't in the derive-new bucket either. A newcomer applying the stated model to `filter` has no basis to predict
whether it copies or mutates. A model presented as complete but missing its most surprising member is worse than an explicit
"not exhaustive".

**Fix (doc-only).** Add `filter` to the build-in-place bucket (next to prune) and `score`/`optimize`/`select_stereo` to the
derive-new bucket. Or state the rule generatively: "verbs that *narrow/settle* the ensemble mutate in place (mc, minimize,
prune, filter); verbs that *produce a variant* return a new one (lowest, representatives, align, score, optimize)."

---

### O5 — `embed()` can return a bare `list` (`stereo='separate'`) that does NOT chain — and the return-type summary hides it **[medium · safe-fix]**
`pipeline.py:265-266` (opening docstring), `pipeline.py:357-361` (`_relax_embedded`), `embed/dispatch.py:535-536`; CLAUDE.md
"the racemate" paragraph + the discoverability table.

**Problem.** `embed()` returns **three** shapes: `Ensemble`, `EnsembleSet`, or `list[EnsembleSet]` (for `stereo='separate'`).
The opening docstring line and CLAUDE.md both say only "an `Ensemble` or an `EnsembleSet`". The `list[EnsembleSet]` case is
buried 40 lines down in the `stereo=` section. This has teeth: a plain `list` has no `.mc()`, so `rx.embed(smi,
stereo='separate').mc()` raises `AttributeError` — directly violating CLAUDE.md's headline promise that
"`rx.embed(anything).mc().prune().dump(...)` works whether the input is one molecule, a racemate, metal isomers, or NCI modes."
The table row "`stereo='enumerate'` keeps them separate" sits right next to that promise with no "returns a list you must iterate"
caveat.

**Fix (doc-only).** State all three return shapes in the opening line, and flag on the `'separate'` mode that it returns a
`list[EnsembleSet]` you **iterate/index** (it does not chain) — mirroring the metal `EnsembleSet.select()` idiom. Optionally add
the same caveat to CLAUDE.md's stereo table row.

---

### O6 — `stereo=` canonical names vs aliases are inconsistent between code and CLAUDE.md **[low · judgment-call]**
`pipeline.py:320` (`{"auto": "racemic", "enumerate": "separate"}`), `embed()` docstring 305-309; CLAUDE.md table +
"undefined stereocentres" prose.

**Problem.** The code canonicalizes to `'racemic'` / `'separate'` (with `'auto'` / `'enumerate'` as accepted older aliases).
CLAUDE.md advertises the **aliases**: the table row uses `stereo='enumerate'` and the word "racemic" only as the default; the
canonical `'separate'` never appears in the doc. So a newcomer learns the deprecated spelling and never meets the canonical one,
and grepping the code for `'enumerate'` finds only the alias-normalization line. Two names for one mode, taught inconsistently.

**Fix (doc-only, or a one-word code comment).** Pick the canonical spelling (`'separate'`/`'racemic'`) and use it consistently in
CLAUDE.md; mention the alias once as "(alias: `'enumerate'`)". Cheap, removes a real "which one is real?" stumble.

---

### O7 — `_MetalCtx.extra` is a cryptic field name for "the other surrogated metals" **[low · judgment-call]**
`embed/dispatch.py:94` (field), consumed at `dispatch.py:738,764`, `pipeline.py:509,1254-1255,1631`.

**Problem.** `extra: list` reads as "miscellaneous extras"; it actually means *the additional surrogated metals beyond the first,
in a multi-metal complex* — `(idx, real_z, real_q)` triples. Every consumer does `metals[0]` + `extra = metals[1:]` then loops
`for mi, rz, rq in mc.extra`. The name gives a newcomer no hint that this is "the rest of the metals". The docstring comment does
explain it, but the name fights the comment.

**Fix (rename).** `extra` → `other_metals` (or `extra_metals`). Grep-scoped; ~6 sites.

---

### O8 — `inputs.py` mixes privacy conventions on its two cross-module entry points **[low · safe-fix]**
`inputs.py:16` (`_xyz_to_mol`, underscore) vs `inputs.py:74` (`parse_smiles`, public); both imported by `dispatch.py` and
`pipeline.py` / `pipeline.minimize`.

**Problem.** `inputs.py`'s module docstring frames it as "the shell leaf the embed dispatch hands a source to". Its two peer
functions — same role, same file, same cross-module callers — carry opposite privacy signals: `parse_smiles` (public) and
`_xyz_to_mol` (leading underscore, "private"). A newcomer reading the underscore assumes `_xyz_to_mol` is internal and hesitates
to call it, though it is a first-class shell API used across three modules. Inconsistent signal, no functional reason.

**Fix.** Either drop the underscore (`xyz_to_mol`) to match `parse_smiles`, or add one line to the module docstring: "these are
shell-internal but cross-module; the underscore is historical, not a privacy boundary."

---

### O9 — `embed()`'s front-door docstring (~50 lines) buries the "what" under the verb catalog **[low · judgment-call]**
`pipeline.py:265-315`.

**Problem.** `embed()` is *the* first function a newcomer reads. Its docstring opens with a one-line summary (good), then
immediately dives into the three-verb catalog, the one-surface example list, and the full `stereo=` treatise before the reader
has a mental model. It is comprehensive and accurate — but as the front door it front-loads reference material over orientation.
(Elsewhere the docstring density is a strength; this is the one spot where a reader needs the map before the detail.)

**Fix (judgment call — optional).** Lead with 2 lines: "Embed conformers for `source` (SMILES / .xyz / Mol / metal `Isomer`),
optionally constrained, and return a chainable `Ensemble` (or an `EnsembleSet`/`list` of candidates — see Returns). Every
capability is one more keyword on this one call." Then the catalog. No content lost, just re-ordered.

---

### O10 — CLAUDE.md's closing pointer targets a gitignored file **[low · safe-fix]**
CLAUDE.md final line: "See `plan.md` for the forward roadmap and open threads."

**Problem.** `plan.md` exists locally but is **gitignored** (`git check-ignore plan.md` → hit; not tracked). A newcomer on a
fresh clone has no `plan.md`, so the doc's one forward-looking pointer dangles. Minor, but it's the last thing the onboarding doc
says.

**Fix (doc-only).** Either note "(local, not checked in)" or point at the tracked `docs/findings/` corpus, which is where the
forward reasoning actually lives now.

---

## 2. Not-a-finding / personal-taste (recorded so they aren't re-litigated)

- **The `_mol` / `.mol` surrogate split** (`pipeline.py:461-511`) reads as complex but is documented at exactly the right depth
  and is genuinely essential (a metal has no valid connected graph mid-pipeline). Leave it.
- **Docstring density** across `pipeline.py`/`base.py` is high, but the prior review + `MEMORY.md` establish the comments are
  anti-re-litigation *why* (twice-reverted physics, measured trade-offs) and not inferable from code. Do not thin (O9 is a
  *re-order of one docstring*, not a call to cut comments).
- **`EnsembleSet` subclassing `list`** is a footgun (`_relax_embedded` had to order its isinstance checks to dodge it,
  `pipeline.py:357`) but it buys the "chain works on one-or-many" ergonomics and the risk is called out at the one site it
  bites. Judgment call, not worth churning.
- **`contacts` as a `Constraints` field name** colliding with the `embed(contacts=)` arg is prior-review **F10**, still live
  (`base.py:36`). Real, but already logged there; the field's docstring now at least explains it's `(dist-keys, angle-keys)`
  provenance. Renaming to `releasable`/`provenance` remains the right call — deferring to F10, not re-raising.

---

## 3. Verdict

A new dev **can** trace the one chain in an afternoon *from the code* — the pipeline verbs are well-named and well-explained,
`resolve_core` is legible, and the copy/compose footgun the prior review feared is now closed. Where they get lost is **before
they open the code**: `CLAUDE.md`, their map, describes a repo layout that the `rdkit_embed` carve-out superseded (O1), and once
inside, the `mc`/`_metal` name overloading in `pipeline.py` (O2) and the two silent behaviours — `prune` relaxing (O3), `embed`
sometimes returning an un-chainable `list` (O5) — are the sharp edges. Every finding here is a doc fix or a grep-scoped rename;
none touches the measured-refuted physics. Fix O1 first: it is the cheapest, highest-leverage change, and it is pure signage.
