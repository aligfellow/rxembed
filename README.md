# rxembed: Constrained conformer embedding for reactive chemistry

rxembed generates 3D conformers around stated geometry: distances, angles, rigid transition-state cores,
coordination polyhedra and non-covalent contacts. It edits RDKit's distance bounds before embedding and holds
the constraints through a restrained UFF relaxation.

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

## Features

- SMILES, XYZ, RDKit `Mol` and metal `Isomer` inputs.
- Numeric, rigid-core, coordination and non-covalent constraints that combine in one call.
- Coordination isomers with named polyhedra, vertex slots and Λ/Δ hands.
- Embedding, Monte Carlo search, pruning, geometry checks and energy ranking.

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
| `workflow` | `xyzgraph`, `prism_pruner`, `scikit-learn`, `matplotlib` | XYZ perception, pruning, representatives and plots |
| `search` | `openconf` | Monte Carlo conformational search |

xTB scoring and optimization require the [xTB executable](https://github.com/grimme-lab/g-xtb). Put it at
`~/bin/xtb` or set `$XTB_EXE` to its path.

The examples below run from the root of a clone, which holds the structures in `examples/structures/`.

## Quick Start

```python
import rxembed as rx

ens = rx.embed("OCCCCO", n=20).minimize()
best = ens.lowest(3)
best.dump("best.xyz")  # three conformers in one multi-frame XYZ
```

`rx.embed` accepts SMILES, an `.xyz` path, an RDKit `Mol` or a selected metal `Isomer`. For one molecule it
returns one `Ensemble`. When the request has several answers, such as every isomer of a metal complex, it
returns an `EnsembleSet`: a list of ensembles whose methods apply to every member.

By default rxembed prints only outcomes: an error, or a warning when the result differs from what you asked
for. [Messages and Errors](#messages-and-errors) explains them.

### Embedding settings

`n` sets the number of conformers, and `seed` and `threads` set sampling. Everything else about how seeds are
made lives in one `rx.EmbedParams` object, passed as `params=`:

| field | default | meaning |
|---|---|---|
| `seed`, `threads` | a fixed seed, all cores | sampling; `threads` changes the worker count, not the coordinates |
| `prune_rms` | `None`: 0.1 Å, or the native object's own | RMSD below which RDKit drops a duplicate seed |
| `knowledge` | `None`: on, or the native object's own | RDKit's basic-knowledge terms, such as flat aromatic rings |
| `native` | `None` | an RDKit `EmbedParameters` object to use instead of the default model |
| `coplanar_14` | `True` | rxembed's extra 1-4 distances for flat groups (RDKit's own 1-4 bounds stay) |
| `metal_floor_relief` | `True` | replace RDKit's minimum distances around the metal's placeholder atom with rxembed's |
| `donor_orientation` | `True` | terms that point each donor at the metal |
| `conjugation` | `True` | extra flattening of sp2 and conjugated organic groups |

The default model is RDKit's `KDG()` with its all-in-one (AIO) refinement, for every molecule. It has no
experimental torsion terms. To use another RDKit model, pass its parameters as `native`:

```python
from rdkit.Chem import rdDistGeom

p = rdDistGeom.srETKDGv3()  # adds torsion preferences, small rings included
ens = rx.embed("C1CCCCC1O", n=3, params=rx.EmbedParams(seed=42, threads=1, native=p))
again = rx.embed("C1CCCCC1O", n=3, params=ens.params)  # the same conformers
```

- `ens.params` records the settings a result was made with, so passing it back reproduces the result.
- rxembed owns sampling. Leave `randomSeed`, `numThreads` and `pruneRmsThresh` at RDKit's defaults on the
  native object and set `seed`, `threads` and `prune_rms` on `EmbedParams` instead; otherwise construction
  raises and names the field to use. Likewise `knowledge` must agree with `native.useBasicKnowledge`.
- rxembed rebuilds the bounds matrix for every batch of seeds from `fix`, `constrain` and the metal model.
  Set geometry through those arguments, not with `SetBoundsMat` or `SetCoordMap` on the native object.
- A native object is used by reference, so do not share one between embeds that run at the same time.
- An empty search is not retried with random starting coordinates. Ask for them with a native object that
  sets `useRandomCoords = True`.

The four switches are for studying the model; each `False` removes one of rxembed's additions:

```python
plain = rx.embed("CC(=O)Nc1ccccc1", n=3, params=rx.EmbedParams(coplanar_14=False, conjugation=False))
```

None of them changes UFF settings, and explicit stereo, `fix` and RDKit's own terms stay on.
[07_metal.ipynb](examples/07_metal.ipynb) compares the switches on a metal complex, and plain RDKit KDG + AIO
and UFF against rxembed from the same seeds.

`max_iters` caps the restrained UFF iterations after distance geometry, for example
`rx.embed(smiles, n=1, max_iters=10000)`; `rx.minimize` takes it too. It changes run time, not which
arrangements can be built.

### XYZ input

```python
mol = rx.read_xyz("examples/structures/mnh.xyz", charge=0, bond_orders="xyz2mol")
```

`read_xyz` returns an RDKit `Mol` with the coordinates and perceived bonds. By default xyzgraph decides both
which atoms are bonded and the bond orders. The two choices are separate:

- `connectivity="rdkit"` (RDKit's own connect-the-dots) or `connectivity="xyz2mol"`, each with
  `bond_orders="xyz2mol"`. xyzgraph bond orders need xyzgraph connectivity.
- `bond_orders="xyz2mol"` optimizes transition-metal valence. Keep the default when the input has reactive or
  multicentre bonds, such as a transition state or a bridging hydride.
- With `bond_orders="xyz2mol"`, a hydrogen shared by two atoms keeps one covalent bond. Its second leg is left
  out when it ends on an atom with a lone pair, since that is a hydrogen bond (hold one with `constrain`); a
  B-H-B bridge keeps it as a zero-order contact.

An XYZ file does not store a charge. `charge=0` is the default and means "assume neutral"; rxembed never
infers the charge. Pass the real charge for an ion. At `charge=0`, `read_xyz` warns when a metal ends over its
valence electron count, negative or with an odd electron count, since that is how an ion read as neutral
usually looks. For several metals whose split the total charge does not fix, name each metal's charge by atom
index, for example `metal_charges={0: 2, 1: 1}`.

If the chosen perceiver fails, `read_xyz` warns and tries the others; the atoms, their order and the total
charge never change. `fallback=False` raises instead. The returned `Mol` records what happened in its
`_rxembedConnectivity`, `_rxembedBondOrders` and `_rxembedPerceptionFallback` properties, plus:

- `_rxembedConnectivityAdded`: ligand bonds xyzgraph missed but RDKit and xyz2mol both found, restored.
- `_rxembedChargeRescue`: how a metal read past its valence electrons was read instead (empty otherwise).

## Constraints and Rigid Cores

Ordinary molecules, reacting structures and metal complexes all use the same `fix`, `constrain` and
`template` arguments.

| constraint | call |
|---|---|
| a numeric distance, angle or dihedral | `fix={(i, j): 2.05, (i, j, k): 170.0, (i, j, k, l): 180.0}` |
| a soft numeric window | `constrain={(i, j): (2.0, 2.2), (i, j, k, l): (170.0, 190.0)}` |
| a rigid core | `fix=[i, j, k]`, `fix={i: (x, y, z)}`, or `template=(ref, SMARTS_or_map)` |
| a coordination polyhedron | `rx.metal(smiles, "octahedral")` |
| a stated NCI grip or π stack | `constrain={(ring_a, ring_b): 3.5}` |
| automatically proposed NCI contacts | `contacts="auto"` |

> [!IMPORTANT]
> `fix` and `constrain` use 0-based atom indices. `rx.match(mol, smarts)` resolves a unique match or raises.

`fix=[atoms]` keeps those atoms where the input has them. `template=(reference, mapping)` copies a core from
another structure through the same rigid `fix`:

```python
ref = rx.embed("OCCCCO", n=1)
held = rx.embed(ref.mol, fix=[0, 1, 2], n=4)  # atoms 0-2 stay where ref has them
grafted = rx.embed("OCCCCCO", template=(ref, {0: 0, 1: 1, 2: 2}), n=4)  # copied onto another molecule
```

A SMARTS template maps one ordered match on each graph. A symmetric SMARTS, or a reference with coordinates
only, needs an explicit `{target_index: reference_index}` map.

### A reacting core

```python
core = "[F].[#6]-[Cl]"
ts = rx.read_xyz("examples/structures/sn2.xyz", charge=-1)
reaction = rx.embed("[F-].c1ccccc1CCl", template=(ts, core), n=4)
fluoride, carbon, chloride = rx.match(reaction.mol, core)
reaction.measure((fluoride, carbon))  # forming bond
reaction.measure((carbon, chloride))  # breaking bond
```

### Several molecules

Write separate molecules in one SMILES with a dot. rxembed embeds them in one frame: each free fragment keeps
at least van der Waals distance from the others and stays within contact range, so a counterion, solvent
molecule or substrate sits beside the rest instead of drifting away. To hold a particular contact, constrain
it:

```python
pair = rx.embed("CC(=O)O.c1ccncc1", constrain={(3, 7): (2.6, 2.9)}, n=5)  # acid O-H...N of pyridine
pair.measure((3, 7))
```

A free fragment is not bonded to anything. To bind it to an open metal site, use
[`coordinate=`](#binding-a-fragment-to-an-open-site).

## Metal Complexes

```python
isomers = rx.metal("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "SPL")
isomers.summary()
trans = isomers.select(label="trans")
trans_confs = rx.embed(trans, n=10)

# Or embed every square-planar isomer in one call.
all_confs = rx.embed("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", metal="SPL", n=10)
for err in all_confs.errors:  # the isomers that could not be built, if any
    print(err)
```

`rx.metal` takes SMILES, an `.xyz` path or a `Mol` and returns every distinct arrangement as an `IsomerSet`.
Write dative SMILES, with donor-to-metal arrows and charged anionic ligands as above; neutral or covalent SMILES
are accepted too. Ordinary donor hydrogens may stay implicit, but write hydrides, H₂ and bridging hydrogens
explicitly.

`.select()` returns one isomer by keys such as `label`, `hand`, `index`, `stereo` or `center`, and raises if
none or several match; `.filter()` takes the same keys and returns a subset. `.summary(details=True)` adds
trans pairs or axial and equatorial sites.

`metal="SPL"` embeds every isomer, with `n` conformers each, and returns the ones that embedded; the rest are
listed in `.errors` (see [Messages and Errors](#messages-and-errors)). A single selected isomer, such as
`trans`, raises its own `EmbeddingError` instead.

### Fixing part of the sphere

If `fix` holds several atoms of the coordination sphere, it decides which arrangements are possible, so pass it
to `rx.metal` before selecting:

```python
mn = rx.read_xyz("examples/structures/mn-h2.xyz", charge=-1)
core = [1, 5, 63, 64, 65, 66]  # Mn, its amine N, and the H-H and N-H...O atoms of a reacting core
states = rx.metal(mn, "OCT", center="Mn", fix=core)
chosen = states.select(label="mer", hand="lambda", index=6)
mn_confs = rx.embed(chosen, n=4)
```

A ligand core away from the metal does not decide the arrangement, so it can be templated after selection:

```python
ligand_core = [0, 1, 2]
templated = rx.embed(trans, template=(trans_confs[0], {i: i for i in ligand_core}), n=10)
```

`fix`, `constrain`, `template`, `coordinate` and `contacts` combine in the final `rx.embed` call; a conflicting
combination raises.

### Binding a fragment to an open site

Each unused polyhedron vertex is an open site. `coordinate=` binds a fragment's donor atom there, with the
normal coordination constraints. It takes a SMARTS, an atom index, or a list with one of those per open site:

```python
pocket = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "SPL").select(index=0)
bound = rx.embed(pocket, coordinate="[OX1]", n=10)
```

### Stereo

`stereo=` on `rx.embed` and `rx.metal` controls point R/S, E/Z, atropisomer M/P and bound-donor hands:

| `stereo=` | effect |
|---|---|
| omitted | a SMILES enumerates only its undefined stereo; a geometry keeps what it measures |
| `"racemic"` | enumerate every configurable element, defined ones included |
| `"separate"` | `rx.embed` only: as omitted, but return a `list[EnsembleSet]`, one per configuration |
| `"free"` | no enumeration; the seed decides |
| `{"N5": "racemic"}` | set the mode of one atom, named by element and index |
| `{"locked": "racemic"}` | enumerate both hands of every coordination-locked donor |

A dict can also set a mode per kind, with the keys `"point"`, `"ez"`, `"axial"`, `"locked"` and `"default"`
and the modes `"preserve"`, `"racemic"`, `"invert"` and `"free"`. Meso forms are dropped.

A coordination-locked donor is a stereocentre only while bound, such as an amine N or a σ-alkyl C. A geometry
input keeps its measured hand in every arrangement, and an arrangement that cannot hold it fails to embed with
a message naming the centre. `{"locked": "racemic"}` instead enumerates both hands of each such centre with
every arrangement, leaving the other stereo as measured:

```python
path = "examples/structures/mnh.xyz"
measured = rx.metal(path)
n_hands = rx.metal(path, stereo={"N5": "racemic"})
locked_hands = rx.metal(path, stereo={"locked": "racemic"})
```

A geometry input with several metals enumerates them together; `center="Mn"` enumerates only Mn and keeps the
other spheres as they are. Atropisomer CXSMILES (`wU`/`wD`, reported as `M`/`P`) round-trips with metal arrangements.

### Distances and screening

- Metal-ligand distances come from a fitted model (`lengths="model"`, the default) for SMILES and XYZ alike.
  `rx.metal(struct, lengths="input")` measures them from the input coordinates instead.
- Every `rx.embed` call makes new coordinates. To relax existing ones, use `rx.minimize(struct)`; to keep part
  of them, use `fix` or `template`.
- `rx.metal` drops arrangements the ligands cannot reach, such as a short chelate across trans sites.
  `screen=False` keeps them, with no promise that they embed. Passing the screen does not prove an arrangement
  can be built either.
- More than 1,000 distinct arrangements raise an error. Then use `rx.metal(mol, observed_only=True)` for the
  measured arrangement only, `rx.embed(mol)` to rebuild it, or `rx.metal(rx.cxsmiles(mol))` for a stated one.

### Shape check

Each accepted conformer passes a shape check. The fit of each polyhedron to the embedded sphere is a misfit
number, 0 being ideal. The requested polyhedron must read within 0.01 of the best-fitting one; a conformer
that has slid into another shape is rejected. Every accepted conformer carries the reading as the RDKit
conformer property `shape`:

```python
cid = trans_confs.ids[0]
print(trans_confs.mol.GetConformer(cid).GetProp("shape"))  # Pd4 SPL 0.070 (next SEE 0.322)
```

For a geometry input, the record also gives the input's own reading.

### Geometries

Names and three-letter codes are case-insensitive. If omitted, rxembed classifies existing coordinates or uses
the first geometry listed for the number of sites. Each haptic face is one site; a four-site complex with an η3
or larger face defaults to tetrahedral, for a piano stool.

| CN | geometries (default first) |
|---|---|
| 1 | `monocoordinate` (`MCO`) |
| 2 | `linear` (`LIN`) |
| 3 | `trigonal_planar` (`TPL`) · `t_shape` (`TSH`) · `trigonal_pyramidal` (`TPY`) |
| 4 | `square_planar` (`SPL`) · `tetrahedral` (`TET`) · `seesaw` (`SEE`) |
| 5 | `trigonal_bipyramidal` (`TBP`) · `square_pyramidal` (`SPY`) |
| 6 | `octahedral` (`OCT`) · `trigonal_prismatic` (`TPR`) · `hexagonal_planar` (`HPL`) |
| 7 | `pentagonal_bipyramidal` (`PBP`) · `capped_octahedral` (`COC`) · `capped_trigonal_prismatic` (`CTP`) |
| 8 | `square_antiprism` (`SQA`) · `dodecahedral` (`DOD`) |
| 9 | `tricapped_trigonal_prismatic` (`TCT`) |
| 10 | `bicapped_square_antiprismatic` (`BSA`) |
| 11 | `edge_contracted_icosahedral` (`ECI`) |

Haptic ligands, including side-on bonds, Cp and arenes, bind through their centroid. Checks measure the real
atoms.

### Ligands

```python
for lig in rx.ligands(trans_confs.mol):
    lig.mol, lig.donors, lig.atoms
```

`lig.donors` maps metal indices to donor indices in `lig.mol`; `lig.atoms` maps ligand positions back to
indices in the complex.

## Save Results

Dative SMILES keeps the connectivity; CXSMILES also keeps the selected metal arrangement.
`dative_smiles(mol, cx=True)` keeps ligand E/Z and atropisomer stereo without metal slot notes. Zero-order
contacts keep RDKit's CX `Z:` field in both, but stereo labels and metal hands are read without them.

```python
print(rx.dative_smiles(mn_confs.mol))
print(rx.dative_smiles(mn_confs.mol, cx=True))
print(rx.cxsmiles(chosen))

for state in states:
    print(rx.cxsmiles(state))
```

Results hold a normal RDKit `Mol`, so RDKit's writers work directly:

```python
from rdkit import Chem

Chem.MolToXYZFile(mn_confs.mol, "mn_0.xyz", confId=mn_confs.ids[0])
```

`dump` writes several conformers at once: an `Ensemble` writes one multi-frame XYZ, and an `EnsembleSet` writes
one named file per isomer.

```python
mn_confs.dump("mn.xyz")
paths = all_confs.dump("palladium.xyz")
```

A CXSMILES goes straight back into `rx.embed`; a plain SMILES is enumerated again. Neither stores `fix` or
`constrain`.

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
- `minimize()` uses restrained UFF; `stiffness=` scales rxembed's restraint terms, not UFF itself. When UFF
  cannot type the real graph, a private stand-in graph relaxes it; such results have
  `energy_kind="uff-surrogate"` and list the substitutions in `uff.surrogates` and `uff.retyped`.
  `score("ff")` is an exact MMFF94s or UFF single point on the real graph and raises when it cannot be typed.
- `check()` reports geometry problems per conformer; `filter("geometry")` drops those conformers, and
  `filter("connectivity")` (needs `workflow`) drops conformers whose bonds changed.
- xTB scoring and optimization need the xTB executable.

Record one restrained UFF cleanup with `trajectory=True`:

```python
walk = rx.embed(trans, n=1, trajectory=True)
cleanup = walk.trajectory
walk.dump_trajectory("cleanup.xyz")
```

`trajectory` is an RDKit `Mol` holding the seed and the UFF snapshots; `dump_trajectory()` writes them as a
multi-frame XYZ without alignment.

`score()` also takes a caller-supplied ASE calculator, here [xtb_ase](https://github.com/Quantum-Accelerators/xtb_ase)
with `xtb` on `PATH`:

```python
from xtb_ase import XTB

ranked = rx.embed("O", n=1).score(rx.ASE(XTB()))
```

## Messages and Errors

`rx.embed` builds a metal complex in three steps. Distance geometry (DG) makes rough 3D seeds from a table of
allowed atom-atom distances. Restrained UFF cleans each seed while restraints hold the metal polyhedron. A gate
then checks the result: ligand bonds intact, no overlapping atoms, the requested polyhedron and donor sites
(the [shape check](#shape-check)), the requested hand (Λ/Δ) and ligand R/S, and any `fix=` values. When the
gate fails, rxembed retries, first with stiffer restraints, then with up to three fresh batches of seeds.

At the default level you see only outcomes:

- `EmbeddingError`: rxembed could not build what you asked for. It is raised when no conformer passed the gate
  after every retry, or when fewer than `n` passed and a rejected one missed part of the request: the polyhedron
  or donor sites, the hand (Λ/Δ), a ligand R/S, or a `fix=` value. Fewer than `n` for any other reason, such as a
  strained bond, is a warning instead. The message is one sentence: the isomer as `summary()` prints it, the most
  common reason with its count, and one remedy, for example `Fe0 TBP cis [H1 C6 P2 P3 N5] -  -: Fe0 relaxes from
  TBP (0.211) to SPY (0.159) in 13/13 rejected seeds; try geometry= or another isomer`. `err.failures` counts
  every rejected seed by failure kind and site, and `err.isomer` is the isomer.
- A request that expands into several candidates (`metal=`, undefined stereocentres, `contacts="auto"`, an
  ambiguous `coordinate=`) returns every candidate that embedded, as an `EnsembleSet` whenever one failed. Each
  one that could not be built logs one warning line and stays in `result.errors`. With `stereo="separate"`, a
  configuration that failed stays in the list as an empty set holding its own `errors`. The call raises only
  when none embeds.
- A warning says the result differs from the request or an input changed: fewer conformers than `n`
  (`kept 3/4 conformers; ...`), conformers without a converged UFF geometry (`ens.unrelaxed`), a `read_xyz`
  perception change, a dropped constraint, or a model choice you may want to override, such as a near-tie
  between two shapes (pass `geometry=`).

The reasons use element symbols and 0-based atom indices:

- `Zn0 relaxes from SPY (0.366) to TBP (0.351)`: the donors relaxed into another polyhedron. The numbers are
  the shape misfits. Five-coordinate metals swap between these two easily (Berry pseudorotation).
- `Fe0 distorts far from OCT (0.75)`: the sphere fits no shape well enough to name it.
- `Zn0 donors swap sites (another isomer)`: the polyhedron holds, but UFF moved donors to other sites.
- `bond C9-C10 squeezed to 0.87 A`, `stretched to 2.33 A` or `C4...C9 clash at 0.90 A`: the ligand is strained
  in this arrangement.
- `N3 reads R, not S` or `N3 no longer reads S`: a ligand stereocentre inverted or lost its configuration. For
  a bound amine N, the remedy names `stereo={'locked': 'racemic'}`, since its measured hand may not fit this
  arrangement.
- `found 0/1 DG seeds with the requested metal and ligand stereo`: no seed had the requested hand together with
  the ligand R/S. With fixed ligand stereocentres some hands cannot exist.

`rx.set_verbose()` adds progress lines. `rx.set_verbose("DEBUG")` also shows every retry, grouped by failure and
site. A retry is bookkeeping, not a result, and a rejected attempt does not prove the structure impossible.
After a successful embed, `ens.relax_failures` records each rejected UFF result by conformer id, and
`ens.unrelaxed` lists the conformers returned without a converged UFF geometry.

## Embedding Engine

Use `rxembed.core` when the input is already an RDKit `Mol` with explicit hydrogens and the workflow tools are
not needed:

```python
from rdkit import Chem
from rxembed import core

mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))
confs = core.embed(mol, constrain={(0, 5): (2.6, 3.0)}, n=8).minimize()
confs.measure((0, 5))
```

`core.enumerate_isomers(mol, geometry)` is the matching metal enumerator.

## Limitations

- Rigid conjugated macrocycles, such as corroles and porphyrins, can embed in folded seatings. rxembed
  enumerates them and they can pass the shape check, but they are chemically unlikely.
- Some structures change a bond during relaxation: read back from its coordinates, the output has a bond made
  or broken. The gate does not re-read connectivity the way an independent perceiver does, so check important
  results, for example with `filter("connectivity")`.
- Charges read from an XYZ come from a Lewis-structure perception, which can disagree with reference data such
  as a published oxidation state.
- Structures with hundreds of isomers take minutes to enumerate and embed.
- The metal is a bond-less carbon placeholder during distance geometry and a lithium during UFF. The real metal
  and its charge are restored before any check or energy.
- `rx.metal` enumerates the face and winding of a haptic ligand, not its rotation about the metal-centroid
  axis. The seeds may sample that rotation; `mc()` keeps each starting pose.
- A ring bound through only some of its atoms (an eta4-arene, a bis-eta2 quinone) folds at its unbound hinge
  toward the class-median tetrahedral angle, about 10 deg past its own crystal fold.
- `mc()` holds poses softly, with about 0.1 Å drift. State a constraint that must hold through `mc()` as a
  distance, angle or dihedral.
- g-xTB with solvent is `E_gxtb(gas) + [E_gfn2(solv) - E_gfn2(gas)]`, never a silent gas-phase energy.

## Development

```bash
just setup     # uv sync + pre-commit
just check     # lint + type + test
```

`uv sync` alone is the full dev environment: the `dev` group pulls `rxembed[search,workflow]`.

Tests gated on a local [tmQMg](https://github.com/hkneiding/tmQMg) clone skip unless
`RXEMBED_TMQMG_DIR` points at its `data/` directory.

- [`examples/`](examples/): from the bounds matrix to transition-metal catalysts.
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
- [tmQMg](https://github.com/uiocompcat/tmQMg/): tmQMg dataset

Related projects:

- [MetalloGen](https://github.com/kyunghoonlee777/MetalloGen): automated transition metal complex conformer generation
- [Molassembler](https://github.com/qcscine/molassembler): molecular graphs, coordination stereochemistry and conformer generation
- [racerTS](https://github.com/digital-chemistry-laboratory/racerts): efficient conformer sampling for transition states
- [OIN-SMILES](https://github.com/tjmustard/OIN-SMILES): lossless conversion between 3D XYZ structures and 1D SMILES
