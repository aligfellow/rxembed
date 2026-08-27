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
pip install .
pip install '.[workflow]'          # perception, pruning, representatives, plots
pip install '.[search,workflow]'   # also add OpenConf Monte Carlo search
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

ens = rx.embed("OCCCCO", n=20).minimize().prune().filter("geometry")
best = ens.lowest(3)
```

`embed` accepts SMILES, `.xyz`, RDKit `Mol`, or metal `Isomer`. It returns an `Ensemble`, an `EnsembleSet` for
multiple modes, or a `list[EnsembleSet]` for `stereo="separate"`. Use `rx.set_verbose("INFO")` for stage
summaries and `"DEBUG"` for per-conformer diagnostics.

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

## Workflow Operations

```python
import rxembed as rx

searched = rx.embed("OCCCCO").mc(preset="rapid").minimize().prune()
representatives = searched.representatives()
ranked = searched.score("gxtb")
best = ranked.lowest(3)
optimized = best.optimize("gfn2")
```

- `mc()` searches with OpenConf and needs the `search` extra.
- `prune()` removes RMSD duplicates with prism_pruner and needs `workflow`.
- `representatives()` keeps the lowest-energy member of each sampled mode.
- `score("ff")` and `minimize()` use UFF; xTB scoring and optimization need the xTB executable.

Record the restrained-UFF cleanup for one constrained conformer directly from `embed`:

```python
walk = rx.embed(selected_isomer, n=1, trajectory=True)
cleanup = walk.trajectory  # RDKit Mol: DG seed, accepted UFF snapshots, final geometry
```

Only the accepted restraint attempt is retained; rejected retries are discarded. Recording requires one
conformer because one `Mol` trajectory represents one path.

Caller-supplied ASE calculators attach without adding ASE to rxembed. With `xtb` on `PATH`,
[`xtb_ase`](https://github.com/Andrew-S-Rosen/xtb_ase) runs GFN2-xTB by default:

```python
import rxembed as rx
from xtb_ase import XTB

ranked = rx.embed("O", n=1).score(rx.ASE(XTB()))
```

This attachment supports `score()`, not `optimize()`, and passes only elements and coordinates.

## Embedding Engine

Use `rxembed.core` for the embedding engine without input parsing, search, pruning, or scoring:

```python
from rdkit import Chem
from rxembed import core

mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))
confs = core.embed(mol, constrain={(0, 5): (2.6, 3.0)}, n=8).minimize()
confs.measure((0, 5))
```

`core.embed` accepts explicit-H `Mol` or selected `Isomer` inputs and returns `Conformers`. Its constraints
match the normal API. Use `core.enumerate_isomers(mol, geometry)` to build metal isomers from a `Mol`.

## Constraints and Rigid Cores

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

Rigid cores can come from the input conformer, explicit coordinates, or a mapped template:

```python
rx.embed(mol, fix=[3, 7, 11])
rx.embed(mol, fix={3: (x, y, z)})
rx.embed(mol, template=(ref, "CC(=O)N"))
rx.embed(mol, template=(xyz, {3: 11}))
```

```python
core = "[F].[#6]-[Cl]"
ref = rx.read_xyz("ts.xyz", charge=-1)
confs = rx.embed("[F-].c1ccccc1CCl", template=(ref, core), n=4)
fluoride, carbon, chloride = rx.match(confs.mol, core)
confs.measure((fluoride, carbon))             # forming bond
confs.measure((carbon, chloride))
```

`template=(ref, SMARTS)` maps one ordered match on each graph. Symmetric SMARTS and coordinate-only templates
need an explicit `{target_index: reference_index}` map.

## Metal Complexes

```python
import rxembed as rx

isomers = rx.metal("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "SPL")
isomers.summary()
ens = rx.embed(isomers.select(label="trans"))
```

`rx.metal` accepts SMILES, `.xyz`, or `Mol` and returns distinct arrangements. Write ionic dative SMILES with
donor-to-metal arrows and charged anionic ligands; neutral/covalent SMILES are also accepted as input.

For a geometry containing several metals, all centres are handled by default. The measured ligand and haptic
stereo is retained while the coordination arrangements are enumerated. Scope a choice with `center=`, or ask
for selected stereo to vary:

```python
isomers = rx.metal("examples/structures/mnh.xyz")
isomers.summary()
mer = isomers.filter(center="Mn", label="mer")
chosen = mer.select(hand="lambda")

n_hands = rx.metal("examples/structures/mnh.xyz", stereo={"N5": "racemic"})
all_hands = rx.metal("examples/structures/mnh.xyz", stereo="racemic")
```

Reactive ligand geometry is filtered after embedding. Near a linear H-M-N relation, H-M-N-H is not a stable
dihedral, so use the H-H distance or another non-collinear coordinate.

`.summary(details=True)` adds trans pairs or axial/equatorial sites and distinguishes graph-non-equivalent
same-element donors. `center="Mn"` enumerates only Mn and retains the other spheres.

Ordinary donor hydrogens may be implicit. Hydrides, H₂, and bridging hydrogen donors must be explicit.

### Geometries

Names and three-letter codes are case-insensitive. If omitted, rxembed classifies existing coordinates or
uses the first geometry listed for the coordination-site count. Each haptic face is one centroid site; a CN4
complex with an η3 or higher face instead defaults to tetrahedral for a piano-stool geometry.

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
> Each unused polyhedron vertex remains an open coordination site.

Haptic ligands, including side-on bonds, Cp, and arenes, bind through a centroid.

### Ligands and serialization

```python
import rxembed as rx

for lig in rx.ligands(complex_mol):
    lig.mol, lig.donors, lig.atoms

rx.dative_smiles(complex_mol)   # normalized constitution
text = rx.cxsmiles(isomer)      # constitution and coordination arrangement
rx.embed(text)
```

`lig.donors` maps original metal indices to donor indices in `lig.mol`; `lig.atoms` maps ligand positions
back to original complex indices.

`dative_smiles` preserves normalized constitution but not the metal arrangement, so cis and trans share a
string. `cxsmiles` also stores geometry, canonical donor slots, and Λ/Δ chirality in atom properties that
`rx.embed` reads back. Both outputs are canonical; CX positions follow output order, not input atom indices.
An η² alkene's `re`/`si` face and an η³ or higher ligand's planar-chiral winding are stored as canonical
`+`/`-` signs on the face slot and selected immediately after distance geometry. Standard CX `c:`/`t:` fields
retain the alkene's E/Z identity. Use `stereo="free"` to leave haptic orientation unspecified.

## Approximations

Current limitations:

| | |
|---|---|
| metal | a bond-less carbon in the distance geometry, a bond-less lithium in the force field. M–L bonds are stripped and the sphere is held by soft constraints. M–L length is a fitted periodic model, not a radius sum |
| exact cores | grafted, not embedded. Distance geometry approximates a rigid core to ~0.2–0.4 Å, fine for a molecule and wrong for a TS |
| `n=N` | up to N seeds; `Ensemble.minimize()` drops failed geometries and retries metal seeds |
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
