# Where `Polyhedron` + `POLYHEDRA` should live, and how to de-clunk the literal

Read-only investigation. Target: `src/rxembed/rdkit_embed/constraints/metal.py` (commit `69deeb8`),
`src/rxembed/rdkit_embed/constraints/polyhedron.py`.

---

## 1. MOVE to `polyhedron.py` — VERDICT: **YES. No cycle. The "leaf" claim is backwards.**

### The actual import edges (verified against the code)

`polyhedron.py` top-level imports (lines 22-27):

```python
from functools import lru_cache
from itertools import permutations
import numpy as np
```

That is the **entire** import list — stdlib + numpy, **nothing from `rxembed`**. It is a genuine leaf.

`metal.py` imports (lines 22-25):

```python
from rxembed.rdkit_embed import io as _io
from . import polyhedron as _poly
from .base import Constraints, add_distance, add_pairwise_shape
```

So the only edge between the two is **`metal → polyhedron`** (metal imports polyhedron). The single use of
`_poly` in metal.py is one call:

```python
# metal.py:962
return _poly.handedness(dirs, list(vertices), _donor_classes(mol, donors), _chelate_edges(...))
```

### Does moving `Polyhedron` + `POLYHEDRA` + the accessors into `polyhedron.py` create a cycle?

**No.** A cycle needs a path `polyhedron → … → metal`. The moved objects are *pure data and pure functions
over that data* and reference nothing in metal.py except one numpy-only helper:

| Moved object | What it depends on | Needs metal.py? |
|---|---|---|
| `Polyhedron` (dataclass, 83-110) | `_vertex_angle` (in `resolved_angles`), `dataclass` | only `_vertex_angle` (numpy-only) |
| `POLYHEDRA` (dict, 116-271) | `Polyhedron`, `_s` (= `math.sqrt(3)/2`) | no |
| `geometries_for_cn` (274-276) | `POLYHEDRA` | no |
| `vertex_dirs` (279-282) | `POLYHEDRA` | no |
| `isomer_permutations` (285-288) | `POLYHEDRA` | no |
| `is_planar` (291-294) | `POLYHEDRA` | no |

The **one** thing that would otherwise reach back into metal.py is `_vertex_angle` (metal.py:897-899) —
used by `Polyhedron.resolved_angles` (line 110). It is numpy-only:

```python
def _vertex_angle(u, v):
    u, v = np.array(u), np.array(v)
    return round(float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))))
```

So `_vertex_angle` (and the constant `_s`, used only inside the `POLYHEDRA` literal) **must move with the
data**, or polyhedron.py would import metal.py and *that* would be the cycle. Once they move too, polyhedron.py
still imports nothing from rxembed and stays a leaf.

### Cycle proof

After the move:
- `polyhedron.py` imports: `functools`, `itertools`, `numpy`, `math` (add `import math` for `_s`). **Zero
  rxembed imports** → still a leaf.
- `metal.py` imports `polyhedron.py` (it already did) and now names more symbols from it.
- The import graph edge is **still only `metal → polyhedron`**. No `polyhedron → metal` edge exists → **no
  cycle**. ∎

### Why the implementer's reasoning is backwards

The claim was: *"polyhedron.py is a pure-algorithm leaf; putting data there would risk a cycle (metal imports
polyhedron)."* But `metal imports polyhedron` is exactly what makes the safe direction `metal → polyhedron`.
Adding data to the *imported-from* module cannot close a cycle **unless that data imports back** — and
`POLYHEDRA` is tuples/floats/ints/bools that import nothing. The cycle risk is fictional here.

It is also the more cohesive home: `polyhedron.py`'s own `point_group` / `handedness` already operate on
`vertex_dirs` templates, and its module docstring already refers to `metal.POLYHEDRA[g].vertex_dirs` (line 5).
Co-locating the templates with the symmetry algorithm that consumes them removes that cross-module reference.

### Exact plan — what moves, who re-imports

**Move into `polyhedron.py`** (verbatim): `_s`, `_vertex_angle`, `Polyhedron`, `POLYHEDRA`,
`geometries_for_cn`, `vertex_dirs`, `isomer_permutations`, `is_planar`. Add `import math` and
`from dataclasses import dataclass, field` to polyhedron.py.

**In `metal.py`, re-export** so its public surface is unchanged (KISS — zero consumer churn):

```python
from .polyhedron import (
    POLYHEDRA,
    Polyhedron,
    _vertex_angle,          # metal.label() still uses it (line 914)
    geometries_for_cn,
    is_planar,
    isomer_permutations,
    vertex_dirs,
)
```

metal.py keeps `from . import polyhedron as _poly` for the `handedness` call. `_vertex_angle` is re-imported
because `label()` (metal.py:914) still calls it.

**Consumers that need NO change** (they import from `.metal`, which re-exports):
- `isomers.py` (imports `POLYHEDRA, geometries_for_cn, geometry_for, isomer_permutations, n_sites, vertex_dirs, classify_geometry` from metal)
- `constraints/coordination_builders.py` (`POLYHEDRA, classify_geometry, geometry_for`)
- `constraints/solver.py:18` (`vertex_dirs` from `.metal`)
- `pipeline.py:701` (`_metal.is_planar`)
- tests: `test_sphere_solver.py` (`m.POLYHEDRA`), `test_metal_chelates.py` (`M.POLYHEDRA`), `test_haptic.py`

Everything keeps working through metal.py's re-export. (Optionally, a purist follow-up could repoint
`isomers.py` / `coordination_builders.py` / `solver.py` at `polyhedron` directly, but that is churn the move
does not require.)

**Note — scope boundary:** `classify_geometry`, `geometry_for`, `n_sites`, `_angle_spectrum`, `coplanar` were
**deliberately left in metal.py** for this plan. `geometry_for`/`n_sites` are thin wrappers over the accessors
(could follow, cheaply). `classify_geometry`/`coplanar` take an **rdkit `Mol`/conformer** — moving them would
add an rdkit dependency to the currently rdkit-free polyhedron.py. That is still cycle-safe, but it changes the
module's character, so keep them in metal.py unless you want polyhedron.py to own perception too.

---

## 2. DE-CLUNK the `POLYHEDRA` literal

### What's clunky today

1. **Positional magic numbers.** `Polyhedron("linear", 2, 0, …)` — the `2, 0` are `cn, default_rank`; a reader
   must consult the dataclass field order to decode them. Same `3, 1` / `4, 2` soup on every record.
2. **`cn` is redundant.** The dataclass comment already states `cn: int  # == len(vertex_dirs)`, and it holds
   for **all 13 records** (verified). It is a magic number restating the length of the tuple two args away.
3. **Long permutation blocks wedged inline.** tbp (10), square_pyramidal (15), octahedral (15) inflate each
   record so the geometry gets lost in its isomer list.

All values stay **bit-identical** below.

### Recommended form (KISS): keyword args + derive `cn` + hoist the 3 long isomer blocks

**Derive `cn`** — add a `__post_init__` to the (frozen) dataclass so it is computed, never restated:

```python
@dataclass(frozen=True)
class Polyhedron:
    name: str
    default_rank: int
    vertex_dirs: tuple
    angles: tuple | None
    permutations: tuple | None = None
    planar: bool = False
    geometric_isomerism: bool = True
    cn: int = field(init=False)  # == len(vertex_dirs), derived — never hand-typed

    def __post_init__(self):
        object.__setattr__(self, "cn", len(self.vertex_dirs))
```

`p.cn` still works everywhere (geometries_for_cn, external `.cn` reads); the value is identical to what was
written (each `cn == len(vertex_dirs)`), so it stays golden.

**Hoist** the three long isomer lists to named constants above the dict:

```python
_OCT_ISOMERS = (
    (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5), (0, 1, 2, 5, 3, 4),
    (0, 2, 1, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 2, 1, 5, 3, 4),
    (0, 3, 2, 1, 4, 5), (0, 3, 2, 4, 1, 5), (0, 3, 2, 5, 1, 4),
    (0, 4, 2, 3, 1, 5), (0, 4, 2, 1, 3, 5), (0, 4, 2, 5, 3, 1),
    (0, 5, 2, 3, 4, 1), (0, 5, 2, 4, 3, 1), (0, 5, 2, 1, 3, 4),
)  # values verbatim from metal.py:222-236
# likewise _TBP_ISOMERS, _SPY_ISOMERS
```

### Before / after — two entries

**BEFORE** (metal.py:119 and 215-238):

```python
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        Polyhedron("linear", 2, 0, ((0, 0, 1), (0, 0, -1)), ((0, 1, 180),), planar=True, geometric_isomerism=False),
        ...
        Polyhedron(
            "octahedral",
            6,
            0,
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
            ((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
            (
                (0, 1, 2, 3, 4, 5),
                (0, 1, 2, 4, 3, 5),
                (0, 1, 2, 5, 3, 4),
                (0, 2, 1, 3, 4, 5),
                (0, 2, 1, 4, 3, 5),
                (0, 2, 1, 5, 3, 4),
                (0, 3, 2, 1, 4, 5),
                (0, 3, 2, 4, 1, 5),
                (0, 3, 2, 5, 1, 4),
                (0, 4, 2, 3, 1, 5),
                (0, 4, 2, 1, 3, 5),
                (0, 4, 2, 5, 3, 1),
                (0, 5, 2, 3, 4, 1),
                (0, 5, 2, 4, 3, 1),
                (0, 5, 2, 1, 3, 4),
            ),
        ),
    ]
}
```

**AFTER** (keyword args, `cn` derived away, permutations hoisted):

```python
POLYHEDRA: dict[str, Polyhedron] = {
    p.name: p
    for p in [
        Polyhedron(
            name="linear",
            default_rank=0,
            vertex_dirs=((0, 0, 1), (0, 0, -1)),
            angles=((0, 1, 180),),
            planar=True,
            geometric_isomerism=False,
        ),
        ...
        Polyhedron(
            name="octahedral",
            default_rank=0,
            vertex_dirs=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
            angles=((0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)),
            permutations=_OCT_ISOMERS,
        ),
    ]
}
```

Every record now names its fields; the two magic numbers (`6, 0`) become `default_rank=0` (cn gone); the
15-row block is one reference. The dict-comprehension `{p.name: p …}` is kept — it auto-keys from `name` and
preserves insertion order (load-bearing: it is the `classify_geometry` tie-break, metal.py:114-115).

### Minimal alternative (if you want zero dataclass change)

Keyword args **only** — keep `cn=2, default_rank=0` explicit, no `__post_init__`, no hoisting. That alone
kills the positional magic-number soup and is a pure-formatting, obviously-golden diff. The `cn`-derivation and
isomer-hoist are the two optional extra wins on top.

---

## VERDICT

- **Move to `polyhedron.py`: YES.** No cycle — polyhedron.py imports nothing from rxembed and stays a leaf;
  the edge remains `metal → polyhedron`. Move `_s`, `_vertex_angle`, `Polyhedron`, `POLYHEDRA`,
  `geometries_for_cn`, `vertex_dirs`, `isomer_permutations`, `is_planar`; re-export them from metal.py so no
  consumer changes. The "risk a cycle" claim inverts the dependency direction.
- **De-clunk: keyword args + derive `cn` (delete the redundant field) + hoist the 3 long permutation blocks
  to named constants.** All values verbatim / golden. A keyword-args-only version is the minimal, zero-risk
  floor.
