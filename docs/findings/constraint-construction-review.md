# Constraint-construction review — do the three designs actually make the path cleaner?

**Task:** take the three completed design investigations together, judge whether the constraint-construction
path (`resolve_core` → coordination builders → polyhedron record → orientation mechanisms) genuinely becomes
cleaner / more general / more onboardable, adversarially critique each, and rank them for the maintainer.

**Scope:** READ-ONLY. No `src/`/tests edits, no commit. Every line reference below was verified against the
live `rdkit-embed-kernel` tree, not taken from the design docs on faith.

The three inputs:
- `design-polyhedron-record.md` — collapse the per-geometry tables into one keyed `Polyhedron` record.
- `design-coordination-builders.md` — move the coordination *builders* out of foundational `metal.py`.
- `design-donor-orientation-general.md` — make donor orientation one general principle (σ-aryl as an sp2
  instance, not a motif row).

---

## 0. Verification pass — the docs are accurate

Before judging, I confirmed the load-bearing claims against the code:

- **Polyhedron tables are real and scattered.** `metal.py` holds `GEOM_OPTIONS` (l.83), `ANGLES` (l.92),
  `PERMUTATIONS` (l.116), `VERTEX_DIRS` (l.169), `COPLANAR_GEOMETRIES` (l.752), `_NO_GEOMETRIC_ISOMERISM`
  (l.785), **plus** a free-floating import-time loop that mutates `ANGLES` for CN7/CN8 at l.958–960 — ~700
  lines below the dict it edits. Consumers verified: `isomers.py` (imports 5 of them), `solver.py` (`VERTEX_DIRS`),
  `polyhedron.py` (receives `dirs` as an arg), `pipeline.py:701` (`COPLANAR_GEOMETRIES`), and two test files.
- **The import cycle and its 5 lazy dodges are real.** `distance.py:14 from .metal import …`;
  `donor_orient.py:15–16 from .distance / from .metal import …`. The builders' 5 lazy in-function imports are
  exactly at `metal.py` l.820, l.861, l.862, l.974, l.1195, each with a "lazy: distance imports this module's
  constants" comment. `solver.py:16–18` is the live precedent: it imports **both** `.metal` and `.distance` at
  top level and is cycle-free. The gate module `rdkit_embed/coordination.py` exists — the naming trap the doc
  warns about is genuine.
- **Donor orientation is ALREADY generalised.** `_orient_donor` (donor_orient.py:264) is one `setdefault` loop
  over `_ORIENT_WALL`; `_coplanar_donor` (l.301) is one loop keyed on `inplane_sp2_donor`. `_ORIENT_WALL[('C',
  SP2)]` computes to `(97.0, 145.5)` exactly as the doc states (`min(85+12, 124.4−5)=97`, ceiling 145.5). The
  three old per-hybridisation rows are already collapsed. **There is no motif table left to remove.**
- **Validation harness exists.** `tests/golden/test_golden_bounds.py` locks the assembled `Constraints`, the DG
  phases, the window pairs, and the final bounds matrix at `rtol=1e-12`. `tests/test_import_hygiene.py:64–72`
  lists the kernel constraint modules in `_KERNEL` — adding one is a one-line change, as claimed.

All three docs are faithful to the tree. The disagreements below are about *judgement*, not *facts*.

---

## 1. The unified story — how a new maintainer understands "how a constraint gets built"

### BEFORE (today)

A newcomer greps `coordination` and lands in a **1228-line `metal.py` that interleaves six unrelated jobs**:
surrogate prepare/restore, the polyhedron data tables, haptic/phantom transients, donor-chirality holds,
geometry classification, isomer identity — **and** the constraint builders. To trace one metal constraint they
must:

1. find `coordination()` at l.846, buried below ~750 lines of foundation;
2. discover — only by reading *function bodies* — that it depends on `distance` and `donor_orient`, because
   those imports are **lazy, mid-body**, invisible from the file head (a cycle dodge);
3. cross-reference **six** name/CN-keyed tables scattered over 80–900 lines, *plus* a free-floating loop that
   silently rewrites `ANGLES` 700 lines away from its definition, to understand the polyhedron data the builder
   reads;
4. reverse-engineer from prose that σ-aryl orientation is "just the sp2 path".

The generic side (`builders.py::resolve_core`) lives in a *different file* from the metal side, and the
dependency graph among `metal`/`distance`/`donor_orient` is a diamond collapsed into one file — legible only by
reading every lazy import.

### AFTER (with #1 + #2 done; #3 already true)

```
resolve_core            coordination_builders.py         POLYHEDRA registry        donor_orient.py
(builders.py)     →     (the metal-polytope             (one Polyhedron record    (two general one-DOF
generic fix/            constraint sources)              per geometry: vertices    holds off one perception
constrain/template      top-level imports ARE            + angles + perms          ruler; σ-aryl = sp2
                        the dependency story             co-indexed)               instance, not a row)
```

- The two builder homes sit **side by side and self-describing**: `builders.py` = index-driven generic sources,
  `coordination_builders.py` = metal-polytope sources. Both are leaf consumers whose *top-level imports are the
  dependency story* — no more mid-body archaeology.
- `metal.py` becomes legibly **"the metal foundation"**: surrogate, primitives, and one `POLYHEDRA` registry
  where a full geometry is **one readable record** instead of six cross-referenced dicts and a spooky loop.
- The module graph is a **strict DAG** (`base ← metal ← distance ← donor_orient ← coordination_builders`), all
  edges top-level — the newcomer reads it top-to-bottom.
- Orientation is **already** one general construction; a 2–3 line doc note (see #3) makes "σ-aryl is an sp2
  instance, the skew is a deliberate known-limit" explicit so no one re-adds a motif row.

**Verdict on the unified story: yes, #1 + #2 genuinely deliver the cleaner/more-general/more-onboardable path
the maintainer asked for.** #3's generality is already shipped and should be left alone — the only *change* it
admits is a regression trap (below).

---

## 2. Adversarial critique — per proposal

### #1 Coordination builders — is it churn / relocation / the start of fragmentation?

**No — it is the one proposal that DELETES machinery rather than moving it.** The 5 lazy imports exist *only*
to dodge the cycle, and the cycle exists *only* because a consumer (`coordination`) and its dependency
(`distance`) share a file. Move the consumer out and the cause is gone: all 5 become ordinary top-level
imports, and `metal.py` ends up importing **nothing** from `distance`/`donor_orient` — an invariant it lacks
today, so it *cannot* regrow a lazy dodge.

- **Not the rejected 7-way split.** This is a **single** extraction along the exact seam the 5 lazy imports
  already trace. It is the *same* carve `distance.py` already made (its docstring: "Carved out of
  `constraints.metal`"), in the same direction. `solver.py` is the live proof the target shape works.
- **Not fragmentation.** After it, `metal.py` is a coherent foundation and `coordination_builders.py` a coherent
  consumer — each one job. It does not invite a second cut.
- **Trading clarity for cleverness?** The opposite — it *removes* the clever bit (the hand-maintained cycle
  dodge).
- **Residual risks (small, mechanical):** (a) `dispatch.py` reaches builders via the `_metal.` alias alongside
  many foundational names (`_metal.coordinate` l.307, `_metal.from_geometry` l.689) — it needs a second `_cb.`
  alias; the foundational `_metal.` names stay. (b) Naming **must** be `coordination_builders.py`, never
  `coordination.py` — the latter collides with the `rdkit_embed/coordination.py` gate and is a discoverability
  trap. (c) `from_geometry` (the `Isomer` factory) moves while the `Isomer` dataclass stays — a minor
  constructor/dataclass split, acceptable and it keeps `metal.py` builder-free.
- **Blast radius:** import-line level only — `isomers.py` (move `coordination` out of its `metal` import block),
  `dispatch.py` (+1 alias), `test_haptic.py` (2 `from_geometry` calls), `test_import_hygiene.py` (+1 `_KERNEL`
  line). **No constraint math is touched**, so the golden is bit-identical by construction.

**This improves readability/onboarding for real, at the lowest risk of the three.**

### #2 Polyhedron record — genuine single-source-of-truth, or relocation?

**Genuine win at the authoring surface; near-zero win at the read sites — and the doc is honest about exactly
that.** The value is real and it is where the maintainer feels pain:

- "Add a geometry" collapses from **up to 7 synced edits across ~700 lines** (miss one → `KeyError` deep in
  embed, or silently-achiral, or wrong default) to **one record**, with a missing required field caught at
  construction.
- The co-indexing invariant `vertex_dirs ↔ angles ↔ permutations` — today only the comment "order matches
  ANGLES" — becomes **structural** (three fields of one object).
- The spooky import-time `_gen_angles` mutation of `ANGLES` 700 lines from its dict is **deleted** (folded into
  a `resolved_angles` property / `angles=None`).

**The adversarial catch — and the hard implementation constraint:** the entire win is at the *authoring*
surface and the *one-time onboarding read*. At the ~20 consumer sites, `VERTEX_DIRS.get(g)` →
`POLYHEDRA.get(g).vertex_dirs` is **longer, not clearer**. The maximalist version that rewrites every consumer
to reach through the record **would be churn with negative readability** — and *that* is the "trading clarity
for cleverness" trap here. The doc's own verdict is the right one and must be enforced as a rule:
**keep the record as the single authoring surface, project the 5 dict names back out as read-only derived
views, and do NOT touch the consumers.** With views retained, blast radius is one block in `metal.py`.

- **Transcription risk is the real hazard** (MEDIUM): hand-moving ~10 vertex-vector rows, ~50 angle triples,
  and the 15-row octahedral/square-pyramidal permutation blocks. One flipped sign or transposed index is a
  **silent physics regression**. This is exactly what the golden net + `test_metal_chelates` +
  `test_sphere_solver` exist to catch — so this proposal **must** be prototype-validated bit-identical before
  commit (see §3).
- **Two subtleties a reviewer must check, not gloss:** `GEOM_OPTIONS` is the one non-1:1 table (CN-keyed,
  order = preference) — it needs `cn` + `default_rank` + a regroup that **must** reproduce
  `["square_planar","tetrahedral","seesaw"]` for CN4. And `COPLANAR_GEOMETRIES` / `_NO_GEOMETRIC_ISOMERISM`
  look derivable but must stay **explicit hand-set bool fields** (edge cases: `t_shape` is planar yet *not*
  in `_NO_GEOMETRIC_ISOMERISM`; deriving them changes values). Keeping them explicit is the KISS call; don't
  "tidy" the int/float mix in the vectors either (it keeps the repr and golden identical).

**Improves onboarding + authoring for real — provided the views-only discipline holds.**

### #3 Donor-orientation general — already done; the only *change* re-introduces the regression

This is the one to be careful with, because the honest answer is uncomfortable: **there is nothing to build.**

- **The generalization the maintainer wants is ALREADY LANDED** (commit `c2ae975`). `_orient_donor` is one
  perception-keyed walling loop; `_coplanar_donor` is one element-agnostic loop; the `{7,8}` element list and
  the three per-hybridisation branches are already gone. σ-aryl **already** types sp2, **already** gets the
  coplanar cap `(M, ipso, o1, o2, 180, 45)` and the `('C',SP2)` in-plane wall. It is already "just the sp2
  stuff", already not a motif row. Re-writing it for cleanliness would be **churn against an already-clean
  target.**
- **The maintainer's own question — does generalizing risk the rigid-chelate regression?** Answered directly by
  the doc and confirmed in the code: **yes, and worse.** The *only* version of #3 that is an actual code change
  is "generalize so as to also fix the σ-aryl skew." Fixing the skew requires turning the deliberately-**flat**
  in-plane floor into a **tight centred restoring pin** — and a *general* centred pin applies that
  over-determination to **every** sp2 σ-donor, including the rigid diphosphine-amidate metallacycle the
  *specific* pin already broke (`test_reembed_retry_delivers_clean_geometry` 24/24 → 22/24). Generality does
  **not** dodge the regression; it **broadens** it, because the regression is a property of *tightness*, not of
  motif-specificity. The gating predicate that would save it (global backbone rigidity) does not exist as clean
  local perception.
- **Trading clarity for cleverness?** The refuted skew-fix (a gated pin, or an invented global-rigidity
  predicate) *is* the cleverness trap. Four collapse-the-two-DOFs-into-one attempts are already
  measured-refuted — do not re-open them.

**So #3 as a construction is a no-op at best and a regression at worst.** The only worthwhile action is
**documentation**: a 2–3 line note (docstring / AGENTS) stating that σ-aryl is an sp2 instance and its
monodentate skew is a deliberate tightness known-limit that g-xTB corrects — so a future maintainer does not
"rediscover" it and re-attempt the refuted pin.

---

## 3. Ranking — readability-ROI vs risk, and the recommendation

| Rank | Proposal | Readability ROI | Risk | Decision |
|---|---|---|---|---|
| 1 | Coordination builders extraction | High (deletes the cycle dodge; DAG; discoverable home) | **Low** — import-rewire only, no math touched | **DO — first** |
| 2 | Polyhedron record (views-only) | High at authoring/onboarding; ~0 at read sites | **Medium** — transcription of vectors/angles/perms | **DO — second, MUST prototype-validate bit-identical** |
| 3 | Donor-orientation generalization | Already achieved; nothing to build | Skew-fix **re-introduces + broadens** the rigid-chelate regression | **SKIP the code change; doc-note only** |

**Order and why:** do **#1 first** — it is the lowest-risk, it is pure structure with a bit-identical golden by
construction, and it makes `metal.py` legibly the foundation *before* #2 reshapes the tables inside it. Do **#2
second**, as the views-only variant, once the file has one clear job. The two are independent, but sequencing
the zero-math structural win ahead of the data-transcription win keeps each PR's failure modes distinct.

**Must be prototype-validated (golden bit-identical + full suite) before committing:**
- **#2 unconditionally** — the transcription risk is a silent physics regression. Gate: `tests/golden` passes
  with **snapshots untouched** (regenerate nothing), plus `test_metal_chelates`, `test_sphere_solver`, and a
  throwaway `assert VERTEX_DIRS == _OLD_VERTEX_DIRS` (and the other four views) inside the PR, dropped after.
- **#1 also** — it *should* be bit-identical, so run the golden + `test_import_hygiene` (`_KERNEL` +1) + full
  suite precisely to prove no number moved. Any golden diff here means the extraction accidentally changed
  behaviour and must be investigated, not regenerated.

**Must NOT be done:** the σ-aryl **skew-fix** (the tight centred pin, general or gated) — measured to
re-introduce and broaden the rigid-chelate regression; and the maximalist **record-through-everywhere** variant
of #2 (churn, negative readability). Keep the flat general orientation hold; keep the 5 derived views.

---

## Provenance

Reads only, `rdkit-embed-kernel`. Verified against: `constraints/metal.py` (tables l.83–212, `_gen_angles`
l.958–960, the 5 lazy imports l.820/861/862/974/1195, builders `coordination` l.846 / `coordination_from_geometry`
l.963 / `from_geometry` l.994 / `coordinate` l.1182 / `_centroid_constraints` l.809), `constraints/donor_orient.py`
(`_orient_donor` l.264, `_coplanar_donor` l.301, `_ORIENT_WALL` l.82–83), `constraints/distance.py:14`,
`constraints/solver.py:16–18`, `isomers.py:20–50/97/323–330/541–549/740–767`, `pipeline.py:701`,
`embed/dispatch.py:307/689`, `tests/test_import_hygiene.py:64–72`, `tests/golden/test_golden_bounds.py`. Design
inputs: `design-polyhedron-record.md`, `design-coordination-builders.md`, `design-donor-orientation-general.md`.
