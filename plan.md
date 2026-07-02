# rxembed — roadmap

Forward-looking work on the clean base. The package is working, validated, and demonstrated; nothing below is
blocking. (The old `nci_embed`→`rxembed` migration playbook is retired — this is only what's still open.)

## Next up: embedding constraint-API redesign — see `DESIGN.md`

Collapse the six-kwarg constraint surface (`freeze`/`distances`/`angles`/`planes`/`template`/`match`/`anchor`)
into three intent-clear verbs — **`fix`** (rigid), **`constrain`** (soft), **`template`** (reference sugar) —
feeding one `resolve_core()`. Clean break, no back-compat. It is a **spec-layer change only** — the graft,
restrained-UFF, pose-freeze and bounds-matrix mechanics already consume `self.cons` unchanged. Adds a
`rx.minimize(source, fix=…, constrain=…)` entry (relax an existing structure toward the targets with the same
vocabulary). Full spec, invariants, grammar, and worked examples in **`DESIGN.md`**.

## Two bigger threads

1. **NCI seeding: keep CH-π / ring-π subordinate (don't over-prioritise).** Ring/π kinds (`CHPI`, `HBPI`,
   `CATPI`, `ANPI`, `HALPI`) are *off* by default in `_AUTO_KINDS` because a rough reference conformer surfaces
   many spurious ones. Two things need care: (a) when requested they can be over-weighted — a weak,
   geometrically-soft CH-π must never out-rank a real H-bond grip, so fold them into the acceptor-quality /
   strength ranking as genuinely weak, rewarded only when the geometry clearly supports the face contact;
   (b) leaving them entirely off means a real π interaction is invisible to the seed. Target a principled
   "weak-but-present, strictly subordinate" treatment. Watch the ranking on aromatic substrates.

2. **NCI-aware energy in the loop — the near-term win is a GFN-FF *pool re-rank*, not GFN-FF per move.**
   Neither the openconf search nor `minimize` (UFF / MMFF) has dispersion or H-bond terms, so the *search
   itself never sees NCIs* — it is geometry + constraint driven, and an NCI-aware energy only enters at
   `score`. The KISS direction (must stay one clear, not-slow call):
   ```python
   ens  = rx.embed("A.B", contacts="auto").mc(explore=True)   # geometry + constraint search, pooled
   best = ens.optimize("gfnff", level="loose").lowest(5)       # NCI-aware relax + rank of the pool
   best.optimize("gxtb")                                        # g-xTB only on the final few
   ```
   GFN-FF is already wired (`refine/xtb.py`, via `xtb --gfnff` — no separate binary). Make the pool re-rank a
   documented default tier with `level="loose"`. GFN-FF *inside* openconf's per-move loop is deferred:
   openconf has no external-energy hook and per-move xtb would be far too slow.

## Performance / clarity

- **Cache NCI detection.** `feature_matrix` and `mode_signature` each run a full `analyzer.detect(positions)`
  pass — cache per-conformer inter-fragment contacts once and reuse.
- **`stereo.signature` round-trips.** It writes a temp `.xyz` and re-runs `xyzgraph.build_graph` +
  `annotate_stereo` per conformer. Prefer an in-memory path if/when xyzgraph exposes one; until then reuse one
  temp handle and skip conformers with no managed chirality element.

## Smaller deferred items

- Provenance for SMARTS-keyed manual `distances=` on the explore path.
- `_input_ordering` reflection-blindness.
- A planar-chirality soft-guarantee; single-donor bifurcation.
- Broaden the calculator tier: the `ASE` wrapper exists (MACE / AIMNet2 / xtb-python / ORCA); a fast NNP as the
  explore-pool ranker is a natural tier below g-xTB.

## Upstream openconf PRs (not rxembed changes)

- Teach `crankshaft` / `ring_kic` / `amide_flip` the constrained-rotor filter so a ring/amide entirely outside
  the frozen core stays eligible (they currently build movable-atom sets from the whole molecule and are
  globally disabled under constraints). The ~1.6× extra ETKDG seeds cover the gap meanwhile.
- Low-mode following under constraints: project the frozen-core DOF out of the Hessian / restrain during the
  eigenvector scan so it doesn't displace pinned atoms.
