# rxembed kernel constraint machinery — KISS / over-abstraction review

*Read-only review at HEAD `003084d`. Lens: over-abstraction / KISS (the maintainer's #1 concern —
"not over-abstracted, kept intuitive so one could onboard quickly"). Scope: the kernel constraint
machinery `src/rxembed/rdkit_embed/constraints/{base,builders,mechanisms,metal,distance,donor_orient,
polyhedron,solver,sphere}.py` + `embed/bounds.py`. Extends the prior 6-pass review
`overnight/maintainability.md` (F1–F17), which predates the kernel carve.*

---

## 0. Verdict — the design held up well through the carve

**The kernel is NOT over-abstracted.** The two abstractions the task asks me to judge both earn their keep,
and the recent changes *improved* the KISS story rather than eroding it:

- **`Constraints` + `compose`/`copy` (base.py).** The prior review's most-cited finding (F1: the
  field-by-field copy footgun, one instance already silently drifted) is **fixed cleanly and correctly**.
  `copy()`/`compose()` are now field-driven through the `_CLONE`/`_MERGE` registries, and a module-load
  assertion (base.py:191-195) fails at import if a new `Constraints` field is added without a clone+merge
  policy. That is exactly the "new field silently not carried" class of bug the maintainer worried about,
  made *unrepresentable*. `.copy(**overrides)` is used at 6 sites; `compose` at 3. Earning its keep.
- **The `Mechanism` phase-hook pattern (mechanisms.py).** This is a genuine co-location device, not a
  framework-for-its-own-sake: each `Constraints` field's DG writer and FF writer used to sit hundreds of
  lines apart in `bounds.py` and `ff.py` and drifted; now they are adjacent, and `MECHANISM_ORDER` is the
  single stated ordering. The two drivers (`bounds._bounds`, `ff.restrained_uff.build`) are each ~6 lines.
  A newcomer can read one class and see both halves of a field. Keep it.
- **`resolve_core` (builders.py).** Three index-driven verbs, good errors, no internal SMARTS. Clean.
- **The metal.py monolith (prior F2, 2042 lines) is largely dissolved.** Split into `distance.py`,
  `donor_orient.py`, `polyhedron.py`, `solver.py`, `sphere.py`; the isomer enumerator + `IsomerSet` moved
  out to the shell (`src/rxembed/isomers.py`). metal.py is down to 1237 lines. Good.

**What the carve fixed (do NOT re-chase these):** F1 + F3.5 (copy/merge footgun → `_CLONE`/`_MERGE`);
F2 (monolith → split); F4 (`_bounds` now has a stage-ordering docstring, mechanisms.py documents the phases);
F8 (`_PHANTOM_FLOOR` no longer collides — `_relieve_phantom_floors` is gone, the constant is consumed only by
`Haptic.dg_relief`); F13 (`_CONJ_O_MDC_ANGLE` shadowing — the dead setdefault was deleted, per
`_coplanar_donor`'s docstring).

**The debt that remains is small and mostly leftover redundancy** — plus one genuinely new structural wart
the carve introduced (the layering leak in §1). Nothing here is high-severity. The findings below are ranked
by maintainer-pain × low-risk-to-fix.

---

## 1. RANKED FINDINGS

### K1 — Coordination *builders* stranded in foundational `metal.py`, forcing 5 lazy imports to dodge a cycle **[med] [judgment-call]**
`metal.py:829, 870-871, 983, 1204` (and the same pattern in `mechanisms.py:259, 349, 394`)

- **Problem.** The carve made `distance.py` and `donor_orient.py` leaf modules that import metal's
  *foundational* constants at top level (`distance.py:14 from .metal import ...`; `donor_orient.py:15-16`).
  But the high-level coordination **builders** — `coordination`, `coordination_from_geometry`, `coordinate`,
  `_centroid_constraints`, `_chelate_bite_window` — were left *in* metal.py, and they consume distance +
  donor_orient. Metal.py is therefore simultaneously the bottom layer (constants, surrogate, polyhedron
  tables, `Isomer`) and a top layer (builders that call up into distance/donor_orient). The cycle is dodged
  with **5 lazy in-function imports** (`from .distance import ml_distance` at 829; `from .distance import
  delocalised_charges, ff_terms, ml_distance` + `from .donor_orient import _coplanar_donor, _orient_donor`
  at 870-871; `from .distance import ff_terms` at 983; `from .distance import _FLOOR_REACH, _tier_floor,
  overbond_tier` at 1204). A newcomer reading `coordination()` finds its dependencies hidden mid-body, and
  the static import graph does not show them. This is prior **F3 in its post-carve form** — the split moved
  the constants out cleanly but left the consumers below their dependencies.
- **Signpost it's misplaced:** the task itself names a `constraints/coordination.py` as if it should exist;
  the builders that belong there are marooned in metal.py.
- **Fix.** Move the five builders into a consumer-layer module that sits *above* distance/donor_orient (they
  already lazily import exactly those two, so the deps are known and small). metal.py then keeps only
  foundation (constants, surrogate_metal/restore, polyhedron tables, `Isomer`, `hold_shape`,
  `_collapse_haptic`/`materialise_phantoms`), and every lazy `from .distance`/`from .donor_orient` in it
  becomes a normal top-of-file import in the new module.
- **Risk.** Mechanical code-motion, but it threads the metal↔distance↔donor_orient triangle and `Isomer`
  is built by both metal.py (`from_geometry`) and the shell — schedule as its own gated PR with the test
  suite as the arbiter, not a drive-by. (The `mechanisms.py` lazy imports of `coordination`/`donor_orient`
  are a *different* axis — enforcement reaching the shared perception ruler — and are more defensible; this
  finding is specifically the metal.py→distance/donor_orient inversion.)

### K2 — The five index-aligned polyhedron tables still have no `Polyhedron` struct **[med] [judgment-call]**
`metal.py:80-221` (`GEOM`, `GEOM_OPTIONS`, `ANGLES`, `PERMUTATIONS`, `VERTEX_DIRS`)

- **Problem.** Adding a coordination geometry still means editing up to five separate module-level dicts and
  keeping the vertex ORDER byte-identical across `VERTEX_DIRS` / `ANGLES` / `PERMUTATIONS` by hand. The
  ordering is load-bearing but the coupling is implicit — this is exactly the "a capability is four
  synchronized edits, not one registry row" that the codebase's own design principles forbid. Prior **F6,
  unchanged by the carve.**
- **Fix.** Collapse into one dict keyed by geometry name — a small `Polyhedron` record `{options, dirs,
  angles, perms}` — so adding a polyhedron is one row and the order-alignment invariant lives in one struct
  with one docstring. **Keep `ANGLES` hand-authored** as the minimal spanning subset (all-pairs is
  measured-refuted; see the in-file note at metal.py:101-107 and `square_pyramidal` at 115-122) and keep the
  CN7/CN8 `_gen_angles` post-step (metal.py:967-969).
- **Risk.** Low, behaviour-preserving co-location — but `PERMUTATIONS`/`VERTEX_DIRS` are consumed across the
  kernel↔shell boundary (`isomers.py`, `polyhedron.py`, `solver.py`), so the record has to export the same
  views. Judgment-call because it is a real diff, not a one-liner.

### K3 — `GEOM` is a provably-redundant copy of `GEOM_OPTIONS`' first column **[low] [safe-fix]**
`metal.py:80-88`

- **Problem.** `GEOM == {n: GEOM_OPTIONS[n][0]}` for every key (verified by inspection); `GEOM` is read only
  by `geometry_for` (metal.py:755). One more table to keep in sync for zero information. Prior **F5, still
  stands.**
- **Fix.** Delete `GEOM`; `geometry_for` reads `GEOM_OPTIONS.get(n_donors, [None])[0]`. Folds naturally into
  K2 if that is done.
- **Risk.** Proven-safe (proven identical).

### K4 — `_coplanar_donor` carries a dead `donor_set` parameter **[low] [safe-fix]**
`donor_orient.py:301` (def), call site `metal.py:904`

- **Problem.** `_coplanar_donor(mol, metal, d, donor_set, cons)` never reads `donor_set` (confirmed: the
  only `donor_set` references in the module are in `_orient_donor` at lines 264/296). The call site passes
  `real_od` purely to satisfy the signature, implying a parallelism with `_orient_donor` — where the arg *is*
  load-bearing for the APEX test — that does not exist, so a reader burns time confirming a negative. Prior
  **F9, still stands** (the sibling F13 `_CONJ_O_MDC_ANGLE` shadowing was resolved by deletion; this leftover
  param was not).
- **Fix.** Drop `donor_set` from the signature and the call site.
- **Risk.** Proven-safe (pure deletion).

### K5 — Proven-safe tidy bundle: duplicated constant, dead branch, frozen knob **[low] [safe-fix]**
- **`_DISCONNECTED = 1e6`** duplicated verbatim (identical value + comment) in `mechanisms.py:35` and
  `metal.py:939` — tuning one leaves the other stale. Hoist to one leaf both import. (Prior F11 residue;
  `_RIGHT_ANGLE` is no longer duplicated — only mechanisms.py has it now.)
- **`n_sites` dead fallback** `metal.py:800`: `len(VERTEX_DIRS.get(g) or ()) or max(max(i,j) for ... in
  ANGLES[g]) + 1` — every geometry in `ANGLES` is also in `VERTEX_DIRS`, so the `or max(...)` branch never
  executes. Drop it or make it an assertion. (Prior F17.)
- **`_smooth(bm, max_tol=0.4)`** `bounds.py:29`: `max_tol` is never overridden by any caller — inline it or
  drop the parameter. (Prior F17.)
- **Risk.** All three proven-safe (behaviour-identical).

### K6 — Two `Mechanism`s are not `Constraints` fields, contradicting the "one class per field" framing **[low] [judgment-call]**
`mechanisms.py:323-403` (`Sp2Planar`, `ConjugationCap`)

- **Problem.** The module docstring and the `Mechanism` base docstring both frame the pattern as "one class
  per constraint FIELD / a constraint field's two writers." But `Sp2Planar` and `ConjugationCap` read **no**
  `Constraints` field — they perceive sp2 carbons / conjugated quartets directly off the conformer and are
  guarded by `if cons.metals: return`. They ride the registry purely to be invoked in the flat FF loop. A
  newcomer building a clean field⟷mechanism mental model trips on the two that map to nothing. (The
  `MECHANISM_ORDER` comment at :411-413 half-acknowledges this — "FF-only (no DG hook)" — but the class-level
  framing above still says "field".)
- **Fix.** Cheapest: one sentence in the module docstring naming the two as FF-only *organic caps* (not
  fields). Alternatively a tiny separate FF-only pass — but that adds a second loop for two classes, which is
  *less* KISS; prefer the doc note.
- **Risk.** Doc-only in the cheap form; proven-safe.

### K7 — Minor smells (note, don't necessarily fix) **[low]**
- **Two disagreeing "is constrained?" predicates.** `Constraints.is_constrained` (base.py:83-85) tests
  `distances | angles | planes | frozen`; the `embed()` gate (bounds.py:178) tests `distances | angles |
  planes | coplanar` — different subsets (one has `frozen`, the other `coplanar`). They answer different
  questions (pipeline-flow vs "edit the matrix at all?") so both are defensible, but the near-identical
  shape invites a reader to assume they agree. A one-line comment on each naming *which* question it answers
  would remove the trap. `is_constrained` also checks only 4 of 15 fields and is correct only because every
  metal path also sets `distances` — that invariant is still uncommented (prior review §4 asked for the
  comment; not added).
- **`_orient_donor`/`_coplanar_donor` guard the frozen core two ways.** `_orient_donor` takes `core_frozen`
  and self-returns (donor_orient.py:287); `_coplanar_donor` takes no such arg and the caller wraps it (`if
  metal not in core_frozen and d not in core_frozen`, metal.py:903-904). Same concern, two idioms — prior
  **F14, still stands.** Passing `core_frozen` into `_coplanar_donor` and self-guarding (matching
  `_orient_donor`) would make both call sites unconditional. Low; folds naturally into a K4 touch of the same
  signature.
- **`_shift_phantoms` (metal.py:301) hand-enumerates the index-keyed fields** it re-keys (`distances`,
  `pulls`, `floors`, `dg_floors`, `angles`, `haptic`, `phantoms`, `spheres`). It correctly *omits* the
  real-atom-only fields (`coplanar`, `frozen`, `planes`, `shapes`) since a centroid index never lands there —
  so this is *not* a live bug — but it is a third "touch every keyed field" site that escaped the
  `_CLONE`/`_MERGE` registry discipline, unprotected by the import-time guard. If a phantom index ever enters
  `coplanar`, it breaks silently. A one-line comment naming *why* only those eight fields (they are the only
  ones that can hold a phantom index) would pin the invariant.

---

## 2. Leave alone (load-bearing complexity — do NOT "simplify")

The prior review's §4 list is unchanged and still binding. The hard constraints from project memory hold:
do **not** merge `floors`/`dg_floors` (they differ on H/APEX membership — see the measured note in
`base.compose`'s docstring, base.py:205-215, and `distance._tier_floor`, distance.py:247-260); do **not**
derive all-pairs `metal.ANGLES` (metal.py:101-107); do **not** re-introduce a charge-keyed M–L contraction
(distance.py:57-65 documents the twice-reverted history); do **not** delete the donor-orientation caps
(`_orient_donor`/`_coplanar_donor`); do **not** numpy-replace the sphere solver.

Two items the task flagged as candidates, framed for the maintainer's call (not fixes to apply):
- **The sphere solver's near-dead reachability.** `solver.py` + `sphere.py` (`SphereSolver`, ~340 lines
  across the two) is a documented **fallback fired on ~2% of chemistry** (`sphere.py:11-13`) and is further
  gated on scipy being installed (`available()`), so on the base install it is *unreachable*. This is the
  clearest "generic machinery serving an almost-never caller" in the kernel. But it is honest, well-isolated,
  and its own docstrings say it should essentially never fire. This is the **scipy keep-vs-delete decision
  that is already a pending maintainer call** — recorded here as data, not a recommendation.
- **`compose(*parts)` is variadic but every one of its 3 call sites passes exactly 2 parts.** Harmless
  generality; reads naturally; not worth changing.

*Bottom line: the carve strengthened the two abstractions the maintainer cares about (the copy/merge footgun
is now compiler-guarded; the DG/FF writers are co-located). The remaining debt is one structural inversion
(K1, the coordination builders below their dependencies), the still-open polyhedron table co-location (K2),
and a scatter of proven-safe leftovers (K3–K5). No high-severity over-abstraction found.*
