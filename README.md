# rxembed: Constrained conformer embedding for reactive chemistry

rxembed biases 3D conformers with geometric constraints stated up front: a frozen transition-state core, a
metal's coordination polyhedron, or a hydrogen bond. Constraints are written into RDKit's distance-bounds
matrix before embedding, then held through a restrained UFF relax.

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

## Features

- Constraints compose. A frozen TS core, a metal sphere and an NCI contact become one `Constraints` struct.
- Rigid cores come from three sources: a molecule's own conformer, coordinates you supply, or a template
  molecule mapped onto a fresh one. Grafted atoms are restored exactly and Kabsch-fitted onto each pose.
- Transition metals use coordination polyhedra: name `"octahedral"` or `"square_planar"` and get the
  distinct arrangements back, keyed on an unambiguous per-vertex slot map and Λ/Δ chirality. Haptic ligands
  bind through a centroid, such as a side-on bond, Cp, or an arene.
- Non-covalent grips represent hydrogen bonds, π-stacks and halogen bonds as soft constraints that survive the
  relax.
- The `geom_check` acceptance gate is TS-aware (a forming bond is not a clash) and metal-aware (a
  dative M–L distance is not a clash).
- SMILES, XYZ, RDKit Mols and metal isomers feed the same constrained engine; the returned ensemble can then
  be searched, deduplicated, checked and scored.

---

## Installation

Not on PyPI yet. A clone uses uv because the required OpenConf version is not yet on PyPI.

```bash
git clone https://github.com/aligfellow/rxembed.git && cd rxembed

pip install .                         # embedding and restrained UFF
pip install '.[workflow]'             # perception, pruning, representatives and plots
pip install '.[search,workflow]'      # add OpenConf Monte Carlo search
uv sync --locked                      # full development environment
```

| extra | brings | for |
|---|---|---|
| `workflow` | `xyzgraph`, `prism_pruner`, `scikit-learn`, `matplotlib` | perception, NCI, selection and plots |
| `search` | `openconf` | Monte-Carlo conformational search |

`xyzrender` is a notebook development dependency; neither package extra installs it.

`networkx` is a base dependency rather than an extra: the vendored xyz2mol perceiver reaches for it, and that
perceiver has to work on a base install. No core module imports it; the embedder itself is `numpy + rdkit`.

Real energies need no Python extra: they shell out to an `xtb` executable at `$XTB_EXE` (default
`~/bin/xtb`). `rxembed.pipeline.calculators.ASE(calc)` is the explicit adapter for a caller-supplied ASE
calculator and imports ASE only when used. Missing selection and search backends name the extra to install:

```pycon
>>> ens.prune()
ImportError: apply needs prism_pruner; pip install 'rxembed[workflow]'
```

---

## Quick Start

### SMILES input

```python
import rxembed as rx

ens = rx.embed("OCCCCO", n=20).minimize().prune().filter("geometry")
best = ens.lowest(3)                                  # three lowest-FF-energy conformers
```

`minimize()` is the force-field relax. `prune()` calls it if needed, but the example states the stage explicitly.
`filter("geometry")` drops conformers that fail the physical gate. Use `rx.set_verbose("INFO")` for stage
summaries or `rx.set_verbose("DEBUG")` for per-conformer diagnostics.

`embed` accepts a SMILES, `.xyz`, RDKit `Mol` or metal `Isomer`. It returns an `Ensemble`, an `EnsembleSet`
when the input expands into candidate modes, or a `list[EnsembleSet]` for `stereo="separate"`.

### XYZ input

```python
import rxembed as rx

mol = rx.read_xyz("complex.xyz", charge=0)
mol = rx.read_xyz("complex.xyz", charge=0, bond_orders="xyz2mol")
```

`read_xyz` returns an RDKit `Mol` with the input conformer and perceived bonds. Perception has two independent
choices:

- `connectivity="xyzgraph"` decides which atoms are bonded; use `"xyz2mol"` as an alternative.
- `bond_orders="xyzgraph"` assigns bond orders; use `"xyz2mol"` for better valence optimisation.

`bond_orders="xyzgraph"` needs `connectivity="xyzgraph"`, since that optimiser runs inside xyzgraph's own
graph. The default remains xyzgraph because it represents reactive contacts such as bridging hydrides,
side-on H₂ and shared protons. When xyz2mol assigns orders, rxembed temporarily removes a hydrogen's longer
second contact and restores it as dative. If a perceiver fails, `read_xyz` warns before trying the other one
and raises if the fallback changes atom count or order.

---

## Workflow Operations

### Monte Carlo search

```python
import rxembed as rx

searched = rx.embed("OCCCCO").mc(preset="rapid").minimize().prune()
```

This chain is embed, **openconf** (by [*Rowan*](https://github.com/rowansci/openconf)) Monte Carlo search, restrained UFF relax, then RMSD deduplication using **prism_pruner** by [*Nicolo Tampellini*](https://github.com/ntampellini/prism_pruner). Only `mc()`
needs the `search` extra; `prune()` needs the `workflow` extra.

### Representative modes

```python
representatives = searched.representatives()
```

`representatives()` clusters an ensemble and keeps the lowest-energy conformer in each sampled mode. Here a
mode is a torsional conformer family. For several fragments it is a relative pose or contact pattern; for a
metal complex it is a ligand arrangement. It can summarize any ensemble and does not require `mc()`.

### Single points and optimization

```python
ranked = searched.score("gxtb")                 # xTB single points; coordinates unchanged
best = ranked.lowest(3)
optimized = best.optimize("gfn2")               # xTB geometry optimization on a copy
```

`score("gxtb")`, `score("gfn2")` and `optimize(...)` need the xTB executable at `$XTB_EXE` or `~/bin/xtb`.
They need no Python extra. `score("ff")` and `minimize()` use RDKit UFF and need no executable.

---

## Embedding Engine

Use `rxembed.core` when another library already owns an explicit-H molecular graph. It is the same embedding
engine used by the normal workflow, without input parsing, search, pruning or scoring.

```python
from rdkit import Chem
from rxembed import core

mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))
confs = core.embed(mol, constrain={(0, 5): (2.6, 3.0)}, n=8).minimize()
confs.measure((0, 5))                              # terminal O...O distance
```

Transfer a reacting core from another explicit-H Mol carrying coordinates:

```python
confs = core.embed(substrate, template=(reference, "[F].[#6]-[Cl]"), n=8).minimize()
```

For a metal, enumerate the arrangements, choose one, then embed that `Isomer`:

```python
mol = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pt](<-[NH3])(Cl)Cl"))
isomers = core.enumerate_isomers(mol, "square_planar").summary()
confs = core.embed(isomers.select(index=0), n=8).minimize()
```

`core.embed` returns `Conformers`. `fix=`, `constrain=` and `template=` have the same meaning as in the normal
API. For a metal, pass a selected `Isomer`. `core.enumerate_isomers` builds it from a `Mol`; if that `Mol` came
from rxembed CXSMILES, its atom properties select the one stated arrangement.

---

## Constraints

| constraint | call |
|---|---|
| a distance or angle | `fix={(i, j): 2.05}`, or `constrain=` for a soft one |
| a rigid reacting core | `fix=[i, j, k]`, or `template=(ref, SMARTS_or_map)` |
| a coordination polyhedron | `rx.metal(smiles, "octahedral")` |
| a stated NCI grip or π stack | `constrain={(ring_a, ring_b): 3.5}` |
| automatically proposed NCI contacts | `contacts="auto"` |

> [!IMPORTANT]
> `fix` and `constrain` keys are 0-based atom indices in the molecule's own order. `rx.match(mol, smarts)`
> resolves a unique match and raises if it misses. A template may instead take the shared SMARTS directly.

---

## Rigid Cores

`fix` is rigid, from three sources, and they compose:

```python
rx.embed(mol, fix=[3, 7, 11])                   # from mol's own conformer
rx.embed(mol, fix={3: (x, y, z)})               # from coordinates you supply
rx.embed(mol, template=(ref, "CC(=O)N"))        # shared substructure on two molecular graphs
rx.embed(mol, template=(xyz, {3: 11}))          # explicit map for coordinates without a graph
```

A distance or angle given instead of coordinates is a pull, not a graft, and is reflection-invariant, so it
may give the mirror image. Read it back with `.measure()`.

### A TS core on a fresh molecule

```python
core = "[F].[#6]-[Cl]"
ref = rx.read_xyz("ts.xyz", charge=-1)
confs = rx.embed("[F-].c1ccccc1CCl", template=(ref, core), n=4)
fluoride, carbon, chloride = rx.match(confs.mol, core)
confs.measure((fluoride, carbon))             # forming bond
confs.measure((carbon, chloride))             # breaking bond
```

`template=(ref, SMARTS)` builds the atom map from one ordered match on each molecular graph. A symmetric SMARTS
or coordinates without a graph need an explicit `{target_index: reference_index}` map.

---

## Metal Complexes

```python
import rxembed as rx

isomers = rx.metal("N->[Pt](Cl)Cl", "square_planar")   # the distinct arrangements
isomers.summary()                                      # list indices, arrangements and chirality
iso = isomers.select(index=0)                          # or select(arrangement=...) from the summary
ens = rx.embed(iso)                                    # embed the selected arrangement
```

`rx.metal` reads a SMILES, `.xyz` or Mol and returns the distinct arrangements. Engine callers use
`core.enumerate_isomers(mol, "square_planar")` directly.

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
import rxembed as rx

for lig in rx.ligands(complex_mol):   # each ligand cut from the metal(s) it coordinates
    lig.mol, lig.donors, lig.atoms    # donors: {metal index in the complex: [donor indices in lig.mol]}

rx.dative_smiles(complex_mol)         # canonical dative SMILES: constitution only
text = rx.cxsmiles(isomer)            # canonical CXSMILES: constitution plus coordination arrangement
rx.embed(text)                        # reads and embeds the stated arrangement directly
```

`lig.mol` is standalone and carries its own conformer. A bridging ligand has one `donors` key per metal it
spans.

`dative_smiles` is canonical for the normalized constitution: connectivity, charges and ligand
stereocentres. It omits the metal arrangement, so cis and trans share one string. Standard SMILES metal
chirality is a neighbour-order permutation and covers only three of the geometries rxembed supports. Routine
hydrogens are implicit; hydrides, H₂ and bridging hydrogen donors remain explicit when they carry a site.

`cxsmiles` appends the arrangement as an RDKit CXSMILES `atomProp` block. For example,
`1.atomNote.s2` assigns atom position 1 in the preceding SMILES to canonical donor slot 2, while
`4.atomNote.SPL` labels atom position 4 as square planar. RDKit parses the block back into atom properties and
preserves them, but does not interpret the rxembed arrangement. `rx.embed(text)` does. The string is canonical
for the normalized constitution and stated arrangement, including Λ/Δ where applicable. Positions are canonical
output positions, not input atom indices; their numeric size has no chemical meaning. Planar-chiral haptic
winding is not yet embeddable from text; `cxsmiles` raises for it, so pass the `Mol` or `Isomer` directly.

---

## Approximations

Current limitations:

| | |
|---|---|
| metal | a bond-less carbon in the distance geometry, a bond-less lithium in the force field. M–L bonds are stripped and the sphere is held by soft constraints. M–L length is a fitted periodic model, not a radius sum |
| exact cores | grafted, not embedded. Distance geometry approximates a rigid core to ~0.2–0.4 Å, fine for a molecule and wrong for a TS |
| `n=N` | up to N seeds; `Ensemble.minimize()` drops failed geometries and retries metal seeds |
| g-xTB solvent | `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]`, never a silent gas-phase energy |
| `mc()` pose-freeze | soft, ~0.1 Å drift. A constraint that must hold across `mc()` has to be a distance or angle the relax also reads |

---

## Development

```bash
just setup     # uv sync + pre-commit
just check     # lint + type + test
```

`uv sync` alone is the full dev environment: the `dev` group pulls `rxembed[search,workflow]`.

- [`examples/`](examples/): 13 notebooks, from the bounds matrix to transition-metal catalysts.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): how it fits together.
- [`AGENTS.md`](AGENTS.md): how we change it.

## License

[MIT](LICENSE)

## References

- [RDKit](https://github.com/rdkit/rdkit): embedding, ETKDG, UFF
- [openconf](https://github.com/rowansci/openconf): Monte Carlo conformational sampling
- [prism_pruner](https://pypi.org/project/prism-pruner/): conformer pruning
- [xyzgraph](https://github.com/aligfellow/xyzgraph): `.xyz` metal and TS perception
- [xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm): vendored, the alternative bond-order perceiver
- [xtb](https://github.com/grimme-lab/g-xtb): GFN-FF, GFN2-xTB and g-xTB

Related projects:

- [MetalloGen](https://github.com/kyunghoonlee777/MetalloGen): automated transition metal complex conformer generation
- [Molassembler](https://github.com/qcscine/molassembler): molecular graphs, coordination stereochemistry and conformer generation
- [racerTS](https://github.com/digital-chemistry-laboratory/racerts): efficient conformer sampling for transition states
- [OIN-SMILES](https://github.com/tjmustard/OIN-SMILES): lossless conversion between 3D XYZ structures and 1D SMILES
