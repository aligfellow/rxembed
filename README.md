# rxembed: Constrained Conformer Embedding for Reactive Chemistry

rxembed generates 3D conformers that satisfy geometric constraints you state up front: a frozen
transition-state core, a metal's coordination polyhedron, or a hydrogen bond, instead of embedding freely and
filtering afterwards. Constraints are written into RDKit's distance-bounds matrix before embedding, then held
again through a restrained UFF relax.

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [What You Can State](#what-you-can-state)
- [Rigid Cores](#rigid-cores)
- [Metal Complexes](#metal-complexes)
- [Approximations](#approximations)
- [Development](#development)

---

## Features

- Constraints compose. A frozen TS core, a metal sphere and an NCI contact become one `Constraints`
  struct, not three code paths that fight each other.
- Rigid cores come from three sources: a molecule's own conformer, coordinates you supply, or a template
  molecule mapped onto a fresh one. Grafted atoms are restored exactly and Kabsch-fitted onto each pose.
- Transition metals use coordination polyhedra: name `"octahedral"` or `"square_planar"` and get the
  distinct arrangements back, keyed on an unambiguous per-vertex slot map and Λ/Δ chirality. Haptic ligands
  bind through a centroid, such as a side-on bond, Cp, or an arene.
- Non-covalent grips represent hydrogen bonds, π-stacks and halogen bonds as soft constraints that survive the
  relax.
- The `geom_check` acceptance gate is TS-aware (a forming bond is not a clash) and metal-aware (a
  dative M–L distance is not a clash).
- Two tiers keep a core that reaches no further than `numpy + rdkit`, and an optional pipeline for search,
  dedup and real energies.

---

## Installation

Not on PyPI yet.

```bash
git clone https://github.com/aligfellow/rxembed.git && cd rxembed

pip install .          # core: numpy + rdkit + networkx
pip install '.[all]'   # everything below
```

| extra | brings | for |
|---|---|---|
| `search` | `openconf` | Monte-Carlo conformational search |
| `select` | `prism_pruner`, `scikit-learn` | dedup, pruning, clustering |
| `perceive` | `xyzgraph` | the default `.xyz` perceiver, metal and TS perception |
| `nci` | `xyzgraph` | binding modes |
| `viz` | `rxembed[select]`, `matplotlib`, `xyzrender` | plots and depictions |

`networkx` is a base dependency rather than an extra: the vendored xyz2mol perceiver reaches for it, and that
perceiver has to work on a base install. No core module imports it; the embedder itself is `numpy + rdkit`.

Real energies need no Python extra: they shell out to an `xtb` executable on `$XTB_EXE` (default
`~/bin/xtb`). `rxembed.pipeline.calculators.ASE(calc)` is the explicit adapter for an ASE calculator and
imports ASE only when used. A missing operation-specific extra names what to install rather than failing obscurely:

```pycon
>>> ens.prune()
ImportError: apply needs prism_pruner; pip install 'rxembed[select]'
```

---

## Quick Start

### Core: `Mol` in, conformers out

```python
from rdkit import Chem
import rxembed as rx

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))   # atoms C0 C1 C2 C3 O4
confs = rx.embed(mol, n=8).minimize()           # embed, then restrained-UFF relax
confs.energies                                  # {0: 4.29, 1: 4.89, ...}
confs.dump("out.xyz")                           # or .xyz(), or .mol
```

### Pipeline: a string in, a ranked ensemble out

```python
import rxembed.pipeline as rx

ens = rx.embed("OCCCCO").mc().prune()               # embed -> MC search -> dedup
ens.representatives()                               # one per distinct mode
ens.score("gfn2").lowest(3).optimize("gfn2")        # real energies, then optimise the best

rx.geom_check.check(ens.mol, ens.ids[0]).summary()  # 'geometry OK (no violations)'
```

> [!NOTE]
> The core verb takes an RDKit `Mol` (or a metal `Isomer`), never a SMILES string. Parsing and explicit
> hydrogens are the caller's job, so the embedder never guesses a graph it was not given. The pipeline verb
> is where parsing lives, and it accepts a SMILES, an `.xyz` path or a `Mol`. `rxembed.pipeline` re-exports
> every core name, so switching the import is the only change.

### Reading a structure in

```python
from rxembed.pipeline import read_xyz

mol = read_xyz("complex.xyz", charge=0)                      # xyzgraph, with a warned base fallback
mol = read_xyz("complex.xyz", 0, bond_orders="xyz2mol")      # xyzgraph's bonds, xyz2mol's orders
mol = read_xyz("complex.xyz", 0, connectivity="xyz2mol", bond_orders="xyz2mol")   # all xyz2mol
```

Perception is two decisions, and each is its own argument:

- `connectivity`: which atoms are bonded. `"xyzgraph"` (default), or `"xyz2mol"`, whose covalent-radius
  tolerance is widened here to 0.5 Å (Open Babel's own, which it ships with, is 0.45) and so catches a haptic
  contact xyzgraph can miss.
- `bond_orders`: what those bonds are. `"xyzgraph"` (default), or `"xyz2mol"`, which pools candidates across
  a charge ladder and ranks them instead of taking the first that fits.

`bond_orders="xyzgraph"` needs `connectivity="xyzgraph"`, since that optimiser runs inside xyzgraph's own
graph build; the combination raises. Over 131 structures (103 tmQM, 28 from the benchmark corpus), cutting
the metal out and asking whether every ligand closes its own valences without RDKit inventing a hydrogen:

| | xyzgraph orders | xyz2mol orders |
|---|---|---|
| every ligand clean | 117/131 | 128/131 |
| one canonical string over 4 shuffles of the input atom order | 103/131 | 124/131 |

The connectivity source matters far less: 125 of 129 give an identical SMILES and an identical coordination
sphere. The four that differ are the tolerance: xyz2mol finds an extra donor on WELROW (a P), ZONHOB (a B)
and SIKQIO (a C), xyzgraph an extra C on SIFJUO.

> [!NOTE]
> The default is still xyzgraph because rxembed is for reactive chemistry. xyz2mol's valence model
> allows hydrogen exactly one bond (`atomic_valence[1] = [1]`), where xyzgraph draws the real contacts of a
> side-on σ-H₂ ligand (H–H 0.859 Å, Mn···H 1.709 Å) and of a shared proton (O–H 1.203 Å, N–H 1.319 Å).
> rxembed holds the longer contact aside and writes it back as a dative, so `bond_orders="xyz2mol"` does
> work on those. xyzgraph remains the more permissive perceiver, which is what exotic and
> reactive bonding needs, while xyz2mol assigns orders and charges better on an ordinary complex.

`bond_orders="xyz2mol"` applies only to a transition-metal complex; a metal-free TS keeps xyzgraph's orders.
If a selected perceiver fails, `read_xyz` warns and tries the other one. Without the optional xyzgraph extra,
it uses vendored xyz2mol for a metal complex or RDKit for an organic molecule. If the fallback would drop or
reorder atoms, the read still raises.

---

## What You Can State

| you can state | as |
|---|---|
| a distance or angle | `fix={(i, j): 2.05}`, or `constrain=` for a soft one |
| a rigid reacting core | `fix=[i, j, k]`, or `template=(ref, map)` |
| a coordination polyhedron | `rx.metal(smiles, "octahedral")` |
| a hydrogen bond or NCI grip | `contacts="auto"` |
| a π stack | `constrain={(ring_a, ring_b): 3.5}` |

> [!IMPORTANT]
> Keys are 0-based atom indices in the molecule's own order. Resolve any pattern yourself: an index is
> unambiguous, a SMARTS match is not. `rx.match(mol, smarts)` will do it and raises if the pattern misses.

---

## Rigid Cores

`fix` is rigid, from three sources, and they compose:

```python
rx.embed(mol, fix=[3, 7, 11])                   # from mol's own conformer
rx.embed(mol, fix={3: (x, y, z)})               # from coordinates you supply
rx.embed(mol, template=(ref, {3: 11}))          # from another molecule
```

A distance or angle given instead of coordinates is a *pull*, not a graft, and is reflection-invariant, so it
may give the mirror image. Read it back with `.measure()`.

### A TS core on a fresh molecule

```python
ref  = Chem.MolFromXYZFile("ts.xyz")        # coordinates only
core = [4, 0, 5]                            # F, C, Cl, read once off the reference

new  = Chem.AddHs(Chem.MolFromSmiles("[F-].c1ccccc1CCl"))
hit  = rx.match(new, "[F-].[#6]-[Cl]")      # raises if the pattern misses

confs = rx.embed(new, template=(ref, dict(zip(hit, core))), n=4)
confs.measure(hit[:2])                      # the reference distance, held
```

The map is `{atom in the new molecule: atom in the reference}`. An empty map raises rather than embedding
with no template.

---

## Metal Complexes

```python
import rxembed.pipeline as rx

isomers = rx.metal("N->[Pt](Cl)Cl", "square_planar")   # the distinct arrangements
isomers.select(arrangement=...)                        # pick one, then embed it
```

`rx.metal` is the pipeline entry point: it adds SMILES / `.xyz` reading to core's
`rx.enumerate_isomers(mol, "square_planar")`, which does the same enumeration on a `Mol` you already have.

### Available geometries

Nameable in full or by the 3-letter code, case-insensitively. Pass a list to compare several yourself. Omit
the name and rxembed picks one: from a geometry it can measure (an `.xyz`, or a `Mol` carrying a conformer) it
classifies the shape the input already has, and from a bare SMILES it falls back to the default for that
donor count, first in each row below.

| CN | geometries (default first) |
|---|---|
| 2 | `linear` (`LIN`) |
| 3 | `trigonal_planar` (`TPL`) · `t_shape` (`TSH`) · `trigonal_pyramidal` (`TPY`) |
| 4 | `square_planar` (`SPL`) · `tetrahedral` (`TET`) · `seesaw` (`SEE`) |
| 5 | `trigonal_bipyramidal` (`TBP`) · `square_pyramidal` (`SPY`) |
| 6 | `octahedral` (`OCT`) · `trigonal_prismatic` (`TPR`) |
| 7 | `pentagonal_bipyramidal` (`PBP`) · `capped_octahedral` (`COC`) · `capped_trigonal_prismatic` (`CTP`) |
| 8 | `square_antiprism` (`SQA`) · `dodecahedral` (`DOD`) |

> [!TIP]
> Naming a geometry with more vertices than the complex has donors leaves the spare vertex empty, as a
> coordination pocket. An octahedral name on a 5-donor complex holds the sixth site open for a substrate to
> bind into.

`->` is RDKit's dative bond, written donor to metal. It keeps the donor's own valence intact and leaves the
metal no bond order to carry, which is how complexes are written throughout.

### Reading one back

```python
import rxembed as rx                  # core: these need no extra

for lig in rx.ligands(complex_mol):   # each ligand cut from the metal(s) it coordinates
    lig.mol, lig.donors, lig.atoms    # donors: {metal index in the complex: [donor indices in lig.mol]}

rx.dative_smiles(complex_mol)         # dative SMILES, checked to parse back to the same atoms
rx.canonical_smiles(isomer)           # that, plus the arrangement, as cxSMILES `<core> |atomProp:…|`
rx.enumerate_isomers(rx.parse_smiles(text))   # reads one back: the stated arrangement, not an enumeration
```

`lig.mol` is standalone and carries its own copy of the conformer. A bridging ligand has two keys in
`donors`, so it says which centres it spans. `dative_smiles` raises rather than return a string that does not
round-trip, which happens where the perceived graph is one SMILES cannot express.

> [!NOTE]
> This is the constitution layer: connectivity, charges and ligand stereocentres. The metal's own
> arrangement is deliberately not written, so cis and trans give one string, as do fac and mer, just as
> `MolToSmiles(isomericSmiles=False)` gives one string for R and S. It is not a species key.
>
> The arrangement is left out rather than half-stated. RDKit would render it as an `@TB…`/`@OH…` tag, but
> that is a permutation index over the neighbour *order*, so symmetry-equivalent ligands destabilise it
> (Fe(CO)₅ writes `@TB20`, `@TB14` or `@TB13` for one molecule), and SMILES has a class for only 3 of the 12
> arrangements rxembed distinguishes. Dropping it takes the corpus from 26 to 40 strings stable under atom
> reordering.

The arrangement lives a layer up, in `canonical_smiles`, which appends it as a cxSMILES `atomProp` block for
all 16 shapes from the coordinate-derived polyhedron descriptor: a canonical slot per donor and Λ/Δ on the
metal. Everything before the first `|` is still the constitution above, so `text.split('|', 1)[0]` is the
species key and any RDKit pipeline reads the rest. Both layers are canonical over the SPECIES, not over the
input: an M–L bond order is whatever the perceiver drew, so a covalent `M–Cl` and an ionic `[Cl-]->[M+]` give
one string. Reading needs no second verb and no extra; `enumerate_isomers` seats a stated arrangement
instead of enumerating.

---

## Approximations

Traded for a robust, stackable embed. Stated rather than hidden.

| | |
|---|---|
| metal | a bond-less carbon in the distance geometry, a bond-less lithium in the force field. M–L bonds are stripped and the sphere is held by soft constraints. M–L length is a fitted periodic model, not a radius sum |
| exact cores | grafted, not embedded. Distance geometry approximates a rigid core to ~0.2–0.4 Å, fine for a molecule and wrong for a TS |
| `n=N` on a metal | N geometries that pass the gate, not N attempts |
| g-xTB solvent | `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]`, never a silent gas-phase energy |
| `mc()` pose-freeze | soft, ~0.1 Å drift. A constraint that must hold across `mc()` has to be a distance or angle the relax also reads |

---

## Development

```bash
just setup     # uv sync + pre-commit
just check     # lint + type + test
```

`uv sync` alone is the full dev environment: the `dev` group pulls `rxembed[all]`.

- [`examples/`](examples/): 13 notebooks, from the bare bounds matrix to transition-metal catalysts searched
  and ranked. Every one that embeds a real structure is gated by `geom_check.check`; `00` and `01` are the
  two that only show the mechanism.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): how it fits together.
- [`AGENTS.md`](AGENTS.md): how we change it.

## License

[MIT](LICENSE)

## References

- [RDKit](https://github.com/rdkit/rdkit): embedding, ETKDG, UFF
- [openconf](https://github.com/rowansci/openconf): additional conformational sampling
- [prism_pruner](https://pypi.org/project/prism-pruner/): conformer pruning
- [xyzgraph](https://github.com/aligfellow/xyzgraph): `.xyz` metal and TS perception
- [xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm): vendored, the alternative bond-order perceiver
- [xtb](https://github.com/grimme-lab/g-xtb): GFN-FF, GFN2-xTB and g-xTB
