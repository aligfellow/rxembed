# Design: collapse the per-geometry polyhedron tables into one keyed record

**Task:** design how "one `Polyhedron` record per coordination geometry, in one registry" would look, and
judge honestly whether it is genuinely cleaner / more onboardable, or just relocation + churn.

**Scope constraint honoured:** this is a design doc only — no `src/` or `tests/` edits. All values quoted
below are transcribed **verbatim** from `src/rxembed/rdkit_embed/constraints/metal.py` @ `rdkit-embed-kernel`.

---

## 1. Consumer inventory — every read of every table

The task names 5 structures. The grep found **7** name/index-aligned per-geometry structures (two more that a
record would naturally absorb), plus 2 functions that encode the CN→geometry logic. All keyed by geometry
**name** except `GEOM_OPTIONS`, which is keyed by **coordination number** (the one non-1:1 case — see §5).

### `VERTEX_DIRS` — `{name: [(x,y,z), …]}` (the master table; order defines slot numbering)
| Site | Access shape |
|---|---|
| `metal.classify_geometry` :710 | **iterates all** `for name, dirs in VERTEX_DIRS.items()` |
| `metal.n_sites` :791 | `len(VERTEX_DIRS[geometry])` — hard `[]`, raises on unknown (deliberate) |
| `metal.chirality_of` :1091 | `VERTEX_DIRS.get(geometry)` → None guard |
| `metal._gen_angles` loop :959 | `VERTEX_DIRS[_g]` — **read to WRITE** `ANGLES[cn7/cn8]` at import |
| `solver.py` :52 | `VERTEX_DIRS.get(geometry)` → None guard |
| `isomers._order_label` :97 | `VERTEX_DIRS.get(geometry)` → None guard, then `dirs[vertex]` |
| `isomers._input_ordering` :541 | `VERTEX_DIRS.get(geometry)`, then `np.array(dirs)` |
| `isomers.distinct_vertex_orderings` :743 | `VERTEX_DIRS.get(geometry)` → None guard |
| `polyhedron.handedness` | receives `dirs` as an **argument** (not a table read) |
| `tests/test_sphere_solver.py` :201 | **iterates all** `m.VERTEX_DIRS.items()` |
| `tests/test_metal_chelates.py` :72,131 | `M.VERTEX_DIRS[iso.geometry][i]` — index a slot |

### `ANGLES` — `{name: [(i, j, angle_deg), …]}` (hand-authored **minimal** vertex-pair subset)
| Site | Access shape |
|---|---|
| `metal.coordination` :896 | `for i, j, a in ANGLES[geometry]` |
| `metal.coordinate` :1210 | `for i, j, a in ANGLES[iso.geometry]` |
| `metal._gen_angles` :960 | **WRITES** `ANGLES[cn7], ANGLES[cn8]` (generated from `VERTEX_DIRS`) |
| `isomers._select_geometries` :328 | `if g not in ANGLES` — validity gate |
| `isomers._select_geometries` :330 | `sorted(k for k in ANGLES if k != 'None')` — "available geometries" error text |

### `PERMUTATIONS` — `{name: [[vertex_order], …]}` (**absent** for linear / trigonal_planar / tetrahedral / CN7 / CN8)
| Site | Access shape |
|---|---|
| `isomers._input_ordering` :549 | `PERMUTATIONS.get(geometry, [tuple(range(len(donors)))])` |
| `isomers.distinct_vertex_orderings` :744 | `geometry not in PERMUTATIONS` — **load-bearing** "no canned list" branch |
| `isomers.distinct_vertex_orderings` :767 | `PERMUTATIONS[geometry]` |

### `GEOM_OPTIONS` — `{cn: [name, …]}` (**CN-keyed**, list order = preference; first = default)
| Site | Access shape |
|---|---|
| `metal.geometry_for` :746 | `GEOM_OPTIONS.get(n_donors, (None,))[0]` — first name = default for that CN |
| `isomers._select_geometries` :323 | `GEOM_OPTIONS.get(n, [])` — "options for N donors" error text |
| docstring refs | `isomers.py` :456, :323 |

### `COPLANAR_GEOMETRIES` — `frozenset({...})` (per-geometry boolean, not in the task's 5)
| Site | Access shape |
|---|---|
| `pipeline.py` :701 | `metal_ctx.geometry not in _metal.COPLANAR_GEOMETRIES` |

### `_NO_GEOMETRIC_ISOMERISM` — `frozenset({...})` (per-geometry boolean, not in the task's 5)
| Site | Access shape |
|---|---|
| `metal.label` :1040 | `if geometry in _NO_GEOMETRIC_ISOMERISM` |
| `isomers._order_label` :100 | `if geometry in _NO_GEOMETRIC_ISOMERISM` |

### `geometry_for` / `n_sites` — the CN→geometry *logic* (functions over the tables)
- `geometry_for(n, has_apical)` reads `GEOM_OPTIONS`; consumed by `metal.from_geometry` :1016 and
  `isomers._select_geometries` :319.
- `n_sites(geometry)` = `len(VERTEX_DIRS[geometry])`; consumed by `isomers._isomers_for_geometry` :381.

**Three access modes the design must preserve:** (a) `.get(name)` → `None` guard (the majority); (b) hard
`[name]`/`in` (n_sites, coordination, the validity gate); (c) **iterate all** (classify_geometry, the sphere
test). And two **writers**: the `_gen_angles` import-time loop that fills CN7/CN8 angles from their vectors.

---

## 2. The record + registry

One dataclass captures every per-geometry fact. The co-indexing invariant that is only a *comment* today
(`# order matches ANGLES`, metal.py:167) becomes structural — `vertex_dirs`, `angles`, `permutations` are three
fields of the *same object*, indexed by the same slot numbers.

```python
@dataclass(frozen=True)
class Polyhedron:
    """One coordination geometry: its vertex template + the hand-authored constraint data keyed off it.

    `vertex_dirs` defines slot numbering; `angles` and `permutations` index into those same slots. `angles`
    is the deliberate MINIMAL spanning subset (NOT all C(n,2) pairs — all-pairs is measured-refuted); `None`
    means "derive every pair from vertex_dirs" (CN7/CN8 only, was the _gen_angles loop). `permutations` is
    `None` when the geometry has no canned coordination-isomer list (linear/trigonal_planar/tetrahedral/CN7/8).
    """

    name: str
    cn: int                                   # coordination number == len(vertex_dirs); GEOM_OPTIONS is grouped on this
    default_rank: int                         # preference order within its CN (0 == the default polyhedron for that CN)
    vertex_dirs: tuple[tuple[float, float, float], ...]
    angles: tuple[tuple[int, int, float], ...] | None   # hand-authored minimal subset; None -> generate from dirs
    permutations: tuple[tuple[int, ...], ...] | None = None   # None -> "no canned list" (a load-bearing distinction)
    planar: bool = False                      # was COPLANAR_GEOMETRIES membership
    geometric_isomerism: bool = True          # False -> was _NO_GEOMETRIC_ISOMERISM membership

    @property
    def n_sites(self) -> int:
        return len(self.vertex_dirs)

    @property
    def resolved_angles(self) -> tuple[tuple[int, int, float], ...]:
        """The hand-authored subset, or every vertex pair derived from the template (CN7/CN8)."""
        if self.angles is not None:
            return self.angles
        v = self.vertex_dirs
        return tuple((i, j, _vertex_angle(v[i], v[j])) for i in range(len(v)) for j in range(i + 1, len(v)))
```

Registry — one dict, one localized block per geometry (values transcribed **verbatim**):

```python
POLYHEDRA: dict[str, Polyhedron] = {p.name: p for p in [
    # ... linear, trigonal_planar, t_shape, tetrahedral, seesaw, trigonal_bipyramidal, square_pyramidal ...

    Polyhedron(
        name="square_planar", cn=4, default_rank=0,
        vertex_dirs=((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),
        angles=((0, 2, 180), (1, 3, 180), (0, 1, 90)),
        permutations=((0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1)),
        planar=True, geometric_isomerism=True,
    ),
    Polyhedron(
        name="octahedral", cn=6, default_rank=0,
        vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        angles=((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
        permutations=(
            (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5), (0, 1, 2, 5, 3, 4),
            (0, 2, 1, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 2, 1, 5, 3, 4),
            (0, 3, 2, 1, 4, 5), (0, 3, 2, 4, 1, 5), (0, 3, 2, 5, 1, 4),
            (0, 4, 2, 3, 1, 5), (0, 4, 2, 1, 3, 5), (0, 4, 2, 5, 3, 1),
            (0, 5, 2, 3, 4, 1), (0, 5, 2, 4, 3, 1), (0, 5, 2, 1, 3, 4),
        ),
        planar=False, geometric_isomerism=True,
    ),
    # square_planar's default_rank=0 within cn=4; tetrahedral default_rank=1; seesaw default_rank=2
    # (that ordered list IS today's GEOM_OPTIONS[4] == ["square_planar","tetrahedral","seesaw"]).

    Polyhedron(  # CN7: angles=None -> derived (was the _gen_angles loop); permutations=None -> no canned list
        name="pentagonal_bipyramidal", cn=7, default_rank=0,
        vertex_dirs=(
            (0, 0, 1), (0, 0, -1), (1, 0, 0),
            (0.309017, 0.951057, 0), (-0.809017, 0.587785, 0),
            (-0.809017, -0.587785, 0), (0.309017, -0.951057, 0),
        ),
        angles=None, permutations=None,
        planar=False, geometric_isomerism=True,
    ),
]}
```

### The migration that keeps churn and transcription risk near zero

The KISS move is **not** to rewrite the ~20 consumer sites to reach through the record. It is to make the
registry the single **authoring** surface and then *project the existing 5 dict names back out as read-only
derived views* right below it:

```python
VERTEX_DIRS  = {k: p.vertex_dirs         for k, p in POLYHEDRA.items()}
ANGLES       = {k: p.resolved_angles     for k, p in POLYHEDRA.items()}   # absorbs the _gen_angles loop
PERMUTATIONS = {k: p.permutations        for k, p in POLYHEDRA.items() if p.permutations is not None}
GEOM_OPTIONS = {                                                          # regroup name-keyed -> CN-keyed
    cn: [p.name for p in sorted(g, key=lambda p: p.default_rank)]
    for cn, g in groupby_cn(POLYHEDRA.values())
}
COPLANAR_GEOMETRIES     = frozenset(k for k, p in POLYHEDRA.items() if p.planar)
_NO_GEOMETRIC_ISOMERISM = frozenset(k for k, p in POLYHEDRA.items() if not p.geometric_isomerism)
```

With the views in place, **every consumer in §1 — including all three test sites and the `isomers.py`
imports — keeps working unchanged**, and the golden is bit-identical because the views hold the same tuples.
`n_sites` / `geometry_for` stay as-is (they read the views) or forward to the record. The whole change is
confined to the definition block of `metal.py`.

---

## 3. Before / After

### (a) Access site — `n_sites` (single hard-index read)
```python
# BEFORE
def n_sites(geometry):
    return len(VERTEX_DIRS[geometry])
# AFTER (views kept)  -> IDENTICAL, no change
def n_sites(geometry):
    return len(VERTEX_DIRS[geometry])
# AFTER (record-native, optional)
def n_sites(geometry):
    return POLYHEDRA[geometry].n_sites
```

### (b) Access site — `classify_geometry` (iterate-all)
```python
# BEFORE
for name, dirs in VERTEX_DIRS.items():
    ...
# AFTER (views kept) -> unchanged.  Record-native form reads the same:
for name, p in POLYHEDRA.items():
    dirs = p.vertex_dirs
    ...
```

### (c) The CN7/CN8 angle generation — a scattered import-time mutation disappears
```python
# BEFORE  (a free-floating loop 800 lines below the ANGLES dict, mutating it in place)
for _g in ("pentagonal_bipyramidal", "square_antiprism"):
    _v = VERTEX_DIRS[_g]
    ANGLES[_g] = [(i, j, _vertex_angle(_v[i], _v[j])) for i in range(len(_v)) for j in range(i + 1, len(_v))]

# AFTER  the intent lives ON the record (angles=None) and in one property; the loop is deleted.
#   ANGLES = {k: p.resolved_angles for ...} does the derivation once, at the definition site.
```

### (d) Add a new geometry — the payoff
```python
# BEFORE: up to 7 hand-synced edits in structures scattered across ~700 lines of metal.py
GEOM_OPTIONS[5].append("pentagonal_planar")        # line ~87   (and pick the right slot for priority)
ANGLES["pentagonal_planar"] = [(0,1,72), ...]      # line ~92
PERMUTATIONS["pentagonal_planar"] = [[...], ...]   # line ~116  (or remember to OMIT it)
VERTEX_DIRS["pentagonal_planar"] = [(...), ...]    # line ~169
COPLANAR_GEOMETRIES = frozenset({..., "pentagonal_planar"})   # line ~752
# _NO_GEOMETRIC_ISOMERISM — decide, edit or not                # line ~785
# (miss any one -> KeyError deep in embed, or silently-achiral, or wrong default)

# AFTER: one record, one place, the type system flags a missing required field
Polyhedron(
    name="pentagonal_planar", cn=5, default_rank=2,
    vertex_dirs=(...), angles=((0, 1, 72), ...),
    permutations=(...),            # omit -> None -> "no canned list" path, explicit
    planar=True, geometric_isomerism=True,
)
```

---

## 4. ANGLES stays hand-authored (unchanged numerics)

The record **holds** the hand-authored minimal subset in its `angles` field verbatim — it does **not**
compute all-pairs for CN≤6 (all-pairs is measured-refuted; the minimal spanning subset is deliberate, per the
metal.py:92 note and MEMORY *under-determined-polyhedron-angles*). `angles=None` reproduces **exactly** today's
`_gen_angles` behaviour and **only** for the two geometries (CN7/CN8) that are generated today — same
`_vertex_angle` rounding, same values. No numeric value in any table changes; `resolved_angles` returns the
identical tuples the current `ANGLES` dict holds, so golden stays bit-identical.

---

## 5. Honest assessment

**Genuinely cleaner where the maintainer actually feels pain (authoring + onboarding):** yes.
- "Add a geometry" collapses from **up to 7 synced edits across ~700 lines** to **one localized record**, and a
  missing required field is a construction error instead of a `KeyError` surfacing deep inside `embed`.
- The co-indexing invariant `vertex_dirs ↔ angles ↔ permutations` (today only the comment "order matches
  ANGLES") becomes structural: three fields of one object a reader sees together.
- The scattered import-time `_gen_angles` mutation of `ANGLES` — a genuine "spooky action at a distance" 800
  lines from the dict — is deleted; the intent (`angles=None`) sits on the record.
- Onboarding: a new maintainer reads one `Polyhedron(name="octahedral", …)` block and has the *entire* spec of
  that geometry, instead of cross-referencing 5–7 dicts 80–700 lines apart.

**Where it is only relocation / where it does NOT help:**
- **The read sites gain nothing.** Most consumers read exactly one field for one geometry; `VERTEX_DIRS.get(g)`
  → `POLYHEDRA.get(g).vertex_dirs` is longer, not clearer. The entire win is at the single authoring surface +
  the one-time onboarding read — *not* at the ~20 call sites. This is why the honest design keeps the derived
  views and does **not** churn the consumers: rewriting them would be pure relocation with negative readability.
- **`GEOM_OPTIONS` is the one non-1:1 table** — CN-keyed, order = preference, an *inverse/aggregate* view, not a
  per-geometry fact. It cannot be a plain field; it needs `cn` + `default_rank` and a regroup. That regroup is a
  small, real subtlety a reviewer must check (does the sorted order reproduce
  `["square_planar","tetrahedral","seesaw"]` for cn 4?).
- **`COPLANAR_GEOMETRIES` / `_NO_GEOMETRIC_ISOMERISM`** are near-derivable (planar ⇔ all `z==0`; isomerism from
  the point group). Design keeps them as **explicit bool fields transcribed verbatim** — deriving them would
  change values via edge cases (t_shape is planar but is *not* in `_NO_GEOMETRIC_ISOMERISM`; linear is in both).
  Explicit = safe and KISS; that is the right call, but it means two fields that look computable are hand-set.

**Access pattern made worse:** none, *if* the views are retained. `permutations=None` must be preserved as the
signal for the "no canned list" branch (`isomers.distinct_vertex_orderings` :744) — a bool/empty-list would
break it — but the design already threads that through (`None` vs a list). No consumer degrades.

**Migration blast radius & transcription risk.**
- Blast radius (views-retained design): **one file, one block** — the definition region of `metal.py`. Zero
  edits to `isomers.py`, `solver.py`, `polyhedron.py`, `pipeline.py`, or the 3 test sites.
- Transcription risk is **real and the main hazard**: hand-moving ~10 vertex-vector rows and ~50 angle triples
  and the 15-row octahedral/square-pyramidal permutation blocks. A single flipped sign or transposed index is a
  silent physics regression. **Mitigation:** a one-shot equality assert (`assert VERTEX_DIRS == _OLD_VERTEX_DIRS`
  for each of the 5, dropped after) or simply the existing golden + `test_every_polytope_states_a_reachable_
  chord_target` + `test_metal_chelates` catch any drift. With that guard the risk is contained to the PR.
- Type/`int` vs `float`: today's tuples mix `(1, 0, 0)` ints and `0.965926` floats; `resolved_angles`/downstream
  do `np.array(dirs, float)`, so preserving the literal ints (not normalising to floats) keeps repr and golden
  identical. Don't "tidy" them.

**Verdict: qualified yes.** Adopt the record **as the single authoring surface with the 5 dict names kept as
derived views** — that delivers the real single-source-of-truth / onboarding win the maintainer asked for at
one-file blast radius and bit-identical golden. Do **not** adopt the maximalist version that rewrites every
consumer to reach through the record: for single-field reads that is churn and mild readability loss, not a
gain. The value is entirely in collapsing the *authoring*, not the *access*.
