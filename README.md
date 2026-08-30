# rxembed: Constrained conformer embedding for reactive chemistry

rxembed generates 3D conformers around stated geometry: distances, angles, rigid transition-state cores,
coordination polyhedra, and non-covalent contacts. It edits RDKit's distance bounds before embedding and
holds the constraints through restrained UFF relaxation.

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

## Features

- SMILES, XYZ, RDKit `Mol`, and metal `Isomer` inputs.
- Composable numeric, rigid-core, coordination, and non-covalent constraints.
- Coordination isomers with named polyhedra, vertex slots, and Λ/Δ chirality.
- Embedding, Monte Carlo search, pruning, geometry checks, and energy ranking.

## Installation

rxembed is not yet on PyPI. Install it from a clone:

```bash
git clone https://github.com/aligfellow/rxembed.git
cd rxembed
# Base
pip install .
# Or recommended: also pruning, representatives and plots
pip install '.[workflow]'
# Or full: also OpenConf Monte Carlo search
pip install '.[search,workflow]'
```

| extra | brings | for |
|---|---|---|
| `workflow` | `xyzgraph`, `prism_pruner`, `scikit-learn`, `matplotlib` | perception, NCI, selection and plots |
| `search` | `openconf` | Monte Carlo conformational search |

xTB scoring and optimization require the [xTB executable](https://github.com/grimme-lab/g-xtb). Put it at
`~/bin/xtb` or set `$XTB_EXE` to its path.

## Quick Start

```python
import rxembed as rx

ens = rx.embed("OCCCCO", n=20).minimize()
best = ens.lowest(3)
best.dump("best.xyz")              # three conformers in one multi-frame XYZ
```

`embed` accepts SMILES, `.xyz`, RDKit `Mol`, or a selected metal `Isomer`. It normally returns one `Ensemble`.
If the request creates several choices, such as every metal isomer, it returns an `EnsembleSet` and applies
chainable methods to each member. `stereo="separate"` returns a list of those sets.

### XYZ input

```python
import rxembed as rx

mol = rx.read_xyz("complex.xyz", charge=0, bond_orders="xyz2mol")
```

`read_xyz` returns an RDKit `Mol` with coordinates and perceived bonds. The default xyzgraph backend preserves
reactive contacts such as bridging hydrides, side-on H₂, and shared protons. Choose perception separately:

- `connectivity="xyzgraph"` decides which atoms are bonded; use `"xyz2mol"` as an alternative.
- `bond_orders="xyzgraph"` assigns bond orders; use `"xyz2mol"` for transition-metal valence optimization.

`bond_orders="xyzgraph"` requires `connectivity="xyzgraph"`.

## Constraints and Rigid Cores

Ordinary molecules, reacting structures, and metal complexes all use the same `fix`, `constrain`, and
`template` arguments. For metals, supply a constraint before selection if it can decide which isomer is
possible.

| constraint | call |
|---|---|
| a numeric distance, angle or dihedral | `fix={(i, j): 2.05, (i, j, k): 170.0, (i, j, k, l): 180.0}` |
| a soft numeric window | `constrain={(i, j): (2.0, 2.2), (i, j, k, l): (170.0, 190.0)}` |
| a rigid reacting core | `fix=[i, j, k]`, or `template=(ref, SMARTS_or_map)` |
| a coordination polyhedron | `rx.metal(smiles, "octahedral")` |
| a stated NCI grip or π stack | `constrain={(ring_a, ring_b): 3.5}` |
| automatically proposed NCI contacts | `contacts="auto"` |

> [!IMPORTANT]
> `fix` and `constrain` use 0-based atom indices. `rx.match(mol, smarts)` resolves a unique match or raises.

`fix=[atoms]` keeps those atoms at their coordinates in the source. `template=(reference, mapping)` transfers
a core from another structure into that same rigid `fix` path:

```python
rx.embed(mol, fix=[3, 7, 11])
rx.embed(mol, fix={3: (x, y, z)})
rx.embed(mol, template=(ref, "CC(=O)N"))
rx.embed(mol, template=(xyz, {3: 11}))
```

### A reacting core

```python
core = "[F].[#6]-[Cl]"
ref = rx.read_xyz("ts.xyz", charge=-1)
reaction = rx.embed("[F-].c1ccccc1CCl", template=(ref, core), n=4)
fluoride, carbon, chloride = rx.match(reaction.mol, core)
reaction.measure((fluoride, carbon))          # forming bond
reaction.measure((carbon, chloride))          # breaking bond
```

`template` supplies coordinates to `fix`. A SMARTS maps one ordered match on each graph; symmetric SMARTS and
coordinate-only templates need an explicit `{target_index: reference_index}` map.

## Metal Complexes

```python
import rxembed as rx

isomers = rx.metal("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "SPL")
isomers.summary()
trans = isomers.select(label="trans")
trans_confs = rx.embed(trans, n=10)

# Or embed every square-planar isomer in one call.
all_confs = rx.embed("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", metal="SPL", n=10)
```

`rx.metal` accepts SMILES, `.xyz`, or `Mol` and returns distinct arrangements. Write ionic dative SMILES with
donor-to-metal arrows and charged anionic ligands; neutral/covalent SMILES are also accepted as input.

Use `rx.metal` when you want to inspect and select the isomer yourself. The `metal="SPL"` shortcut embeds every
enumerated isomer; `n` applies separately to each one.

If `fix` includes several atoms in the coordination sphere, pass it to `rx.metal` before selection:

```python
path = "examples/structures/mnh.xyz"
fixed = [1, 5, 63, 64, 65, 66]

states = rx.metal(path, fix=fixed)
chosen = states.filter(center="Mn", label="mer").select(hand="lambda")
mn_confs = rx.embed(chosen, n=12)
```

An off-sphere ligand core does not choose the metal arrangement, so it can be templated after selection:

```python
ligand_core = [0, 1, 2]
templated = rx.embed(trans, template=(trans_confs[0], {i: i for i in ligand_core}), n=10)
```

Other `fix`, `constrain`, `template`, `coordinate`, and `contacts` arguments can be added to the final
`rx.embed` call. A conflicting combination raises an error.

`coordinate=` fills an open metal site and applies the normal coordination constraints:

```python
pocket = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "SPL").select(index=0)
bound = rx.embed(pocket, coordinate="[OX1]", n=10)
```

Geometry inputs keep their measured ligand and haptic stereo. Several metals are handled together by default;
use `center=` to enumerate only one, or `stereo=` to vary selected stereo:

```python
n_hands = rx.metal(path, stereo={"N5": "racemic"})
all_hands = rx.metal(path, stereo="racemic")
```

`.summary(details=True)` adds trans pairs or axial/equatorial sites and distinguishes graph-non-equivalent
same-element donors. `center="Mn"` enumerates only Mn and retains the other spheres.

Ordinary donor hydrogens may be implicit. Hydrides, H₂, and bridging hydrogen donors must be explicit.

### Geometries

Names and three-letter codes are case-insensitive. If omitted, rxembed classifies existing coordinates or
uses the first geometry listed for the coordination-site count. Each haptic face is one centroid site; a CN4
complex with an η3 or higher face instead defaults to tetrahedral for a piano-stool geometry.

| CN | geometries (default first) |
|---|---|
| 1 | `monocoordinate` (`MCO`) |
| 2 | `linear` (`LIN`) |
| 3 | `trigonal_planar` (`TPL`) · `t_shape` (`TSH`) · `trigonal_pyramidal` (`TPY`) |
| 4 | `square_planar` (`SPL`) · `tetrahedral` (`TET`) · `seesaw` (`SEE`) |
| 5 | `trigonal_bipyramidal` (`TBP`) · `square_pyramidal` (`SPY`) |
| 6 | `octahedral` (`OCT`) · `trigonal_prismatic` (`TPR`) |
| 7 | `pentagonal_bipyramidal` (`PBP`) · `capped_octahedral` (`COC`) · `capped_trigonal_prismatic` (`CTP`) |
| 8 | `square_antiprism` (`SQA`) · `dodecahedral` (`DOD`) |

> [!TIP]
> Each unused polyhedron vertex remains an open coordination site.

Haptic ligands, including side-on bonds, Cp, and arenes, bind through a centroid.

### Ligands

```python
import rxembed as rx

for lig in rx.ligands(complex_mol):
    lig.mol, lig.donors, lig.atoms
```

`lig.donors` maps original metal indices to donor indices in `lig.mol`; `lig.atoms` maps ligand positions
back to original complex indices.

## Save Results

Print dative SMILES for connectivity or CXSMILES to also retain the selected metal arrangement:

```python
print(rx.dative_smiles(mn_confs.mol))
print(rx.cxsmiles(chosen))

for state in states:
    print(rx.cxsmiles(state))
```

Embedded results expose a normal RDKit `Mol`, so RDKit's writers work directly:

```python
from rdkit import Chem

Chem.MolToXYZFile(mn_confs.mol, "mn_0.xyz", confId=mn_confs.ids[0])
```

For several conformers, `dump` is the convenience method: an `Ensemble` writes one multi-frame XYZ, while an
`EnsembleSet` writes one named multi-frame XYZ per isomer.

```python
mn_confs.dump("mn.xyz")
paths = all_confs.dump("palladium.xyz")
```

CXSMILES can go directly back into `rx.embed`; plain SMILES must be enumerated again. Neither stores `fix` or
`constrain`. Every XYZ frame includes all disconnected components in the molecule.

## Workflow Operations

```python
searched = rx.embed("OCCCCO").mc(preset="rapid").minimize().prune()
representatives = searched.representatives()
ranked = searched.score("gxtb")
best = ranked.lowest(3)
optimized = best.optimize("gfn2")
```

- `mc()` searches with OpenConf and needs the `search` extra.
- `prune()` and `representatives()` need `workflow`.
- `minimize()` uses restrained UFF. `score("ff")` uses MMFF94s where possible, otherwise UFF.
- xTB scoring and optimization need the xTB executable.

Record one restrained-UFF cleanup with `trajectory=True`:

```python
walk = rx.embed(selected_isomer, n=1, trajectory=True)
cleanup = walk.trajectory
```

Caller-supplied ASE calculators work with `score()`. With `xtb` on `PATH`:

```python
from xtb_ase import XTB

ranked = rx.embed("O", n=1).score(rx.ASE(XTB()))
```

## Embedding Engine

Use `rxembed.core` when the input is already an explicit-H RDKit `Mol` and the pipeline tools are not needed:

```python
from rdkit import Chem
from rxembed import core

mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))
confs = core.embed(mol, constrain={(0, 5): (2.6, 3.0)}, n=8).minimize()
confs.measure((0, 5))
```

`core.enumerate_isomers(mol, geometry)` is the matching metal enumerator.

## Approximations

Current limitations:

| | |
|---|---|
| metal | RDKit sees a bond-less carbon during distance geometry and lithium during UFF. M–L targets come from the input geometry when present, otherwise from the fitted model. Final structures are validated |
| exact cores | distance geometry biases the core; `fix` then restores its coordinates exactly |
| `n=N` | requests N seeds. Failed structures are retried; required metal, stereo, and fixed-core requests raise if N cannot be produced |
| g-xTB solvent | `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]`, never a silent gas-phase energy |
| haptic axial pose | after face or winding identity is selected, `rx.metal` does not enumerate continuous rotation about the metal-centroid axis. ETKDG seeds may sample it incidentally; `mc()` pose-freezes each seeded face |
| `mc()` pose-freeze | soft, ~0.1 Å drift. A constraint that must hold across `mc()` has to be a distance, angle or dihedral the relax also reads |

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

[MIT](LICENSE). Vendored components retain their upstream notices in [LICENSES.md](LICENSES.md).

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
