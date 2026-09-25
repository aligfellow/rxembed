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
| `workflow` | `xyzgraph`, `prism_pruner`, `scikit-learn`, `matplotlib` | XYZ perception, pruning, representatives and plots |
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

By default rxembed prints only outcomes: an error, or a warning when the result differs from the request.
[Messages and Errors](#messages-and-errors) explains them; `rx.set_verbose("DEBUG")` also shows every retry.

### EmbedParams

The default is RDKit `KDG()` with native all-in-one (AIO) refinement, followed by restrained UFF, for every
molecule including plain organics. Experimental torsions, small-ring torsions and macrocycle rules are off.
`rx.EmbedParams(native=...)`, with any RDKit `EmbedParameters`, changes the seed model:

```python
from rdkit.Chem import rdDistGeom

p = rdDistGeom.srETKDGv3()  # ordinary and small-ring torsions; ETKDGv3() adds macrocycle torsions
ens = rx.embed("C1CCCCC1O", n=3, params=rx.EmbedParams(seed=42, threads=1, native=p))
```

`native` is kept by reference and reused for replacement searches, so do not share it between concurrent
embeds; its own model, initialization, chirality and attempt-limit settings stay in force across retries.
rxembed replaces its bounds matrix on each seed batch using `fix`, `constrain` and the metal constraints,
embedding the constrained graph jointly; use those arguments for geometric restrictions, not a preloaded
`SetBoundsMat`. A native `SetCoordMap` is not merged into our edited matrix or carried into UFF; use `fix` or
`template` for coordinate restrictions through the whole workflow. rxembed owns sampling: leave
`native.randomSeed`, `.numThreads` and `.pruneRmsThresh` at RDKit's defaults and set `EmbedParams(seed=...,
threads=...)` instead, or construction raises naming the field to use. `knowledge=` combined with `native` must
agree with `native.useBasicKnowledge`, or construction raises. `ens.params` records the `EmbedParams` a result
was embedded with, so `rx.embed(m, n=k, params=ens.params)` reproduces it exactly. Coordinates do not depend on
`threads` (probed on one molecule, 12 conformers, pruning on); it only changes worker count.

For unmodified-metal controls, [the metal notebook](examples/07_metal.ipynb) calls KDG + AIO directly on a
dative graph, then compares bond-only, vdW off/on, and native UFF with added distance or polyhedron-angle
restraints from identical seeds. It exposes the native parameters and added force constants. Tagged metal
geometries guide DG but need not survive UFF. Bond-only has no clash protection; vdW off/on affects all
eligible pairs. Python does not expose a selective bond-plus-vdW builder or a single metal-radius knob.

`EmbedParams` also carries rxembed's own ablation switches, for a native-only comparison:

```python
ens = rx.embed(iso, n=1, params=rx.EmbedParams(seed=42, coplanar_14=False, metal_floor_relief=False))
```

All four switches default to `True`. `coplanar_14=False` skips our additional coplanar 1-4 projections, not
RDKit's native 1-4 bounds. `metal_floor_relief=False` retains the native surrogate exclusion floors, not zero
exclusion. `donor_orientation=False` removes rxembed's M-D-X fold and donor-plane terms; `conjugation=False`
removes its organic sp2/conjugation cleanup. None changes UFF settings, and explicit E/Z, point stereo, `fix`
and native RDKit terms stay active throughout. The choices persist through replacement searches and slices.
The notebook compares all four combinations. rxembed's additional force constants live in `mechanisms.py`;
`stiffness` scales selected restraint penalties, not native UFF or the whole objective (see
`relax.restrained_uff`).

`max_iters` changes only the restrained-UFF iteration cap after DG, for example `rx.embed(iso, n=1,
max_iters=10000)`. It is a diagnostic or workload control; it does not make an unrealizable coordination
assignment valid. The same argument is available on `rx.minimize`.

### XYZ input

```python
import rxembed as rx

mol = rx.read_xyz("complex.xyz", charge=0, bond_orders="xyz2mol")
```

`read_xyz` returns an RDKit `Mol` with coordinates and perceived bonds. By default, xyzgraph supplies both
connectivity and bond orders. Choose the two decisions separately:

- `connectivity="xyzgraph"` decides which atoms are bonded; use `connectivity="rdkit", bond_orders="xyz2mol"`
  for RDKit's native connect-the-dots graph, or `connectivity="xyz2mol", bond_orders="xyz2mol"` for the
  vendored metal-aware alternative.
- `bond_orders="xyz2mol"` requests transition-metal valence optimization. Keep the default `"xyzgraph"` when
  its reactive or multicentre bond orders are part of the input model.

The XYZ format does not carry a standard molecular charge. `charge=0` is an explicit neutral assumption, not
an inferred value; pass the dataset or calculation charge for ions. At `charge=0`, `read_xyz` warns when a metal
ends over its valence electrons, negative, or with an odd electron count, which is how an ion read without its
charge usually looks; it never infers the total.

Bond-order assignment first preserves the selected edge set. If xyz2mol cannot assign that topology, `read_xyz`
warns before retrying its jointly perceived connectivity. If both fail, it keeps the selected graph with the
reading that has fewer radical electrons: xyz2mol's, with the charge a metal cannot hold past its valence
electrons moved onto anionic donors as radicals, or xyzgraph's own charges when they reach the total and keep
each metal within its valence electrons. That rescue, like a ligand read at a lower-ranked charge to keep the
metal within its valence electrons, logs one warning and is recorded in `_rxembedChargeRescue` (empty
otherwise). `bond_orders="xyzgraph"` requires `connectivity="xyzgraph"`. Every returned graph has the requested total formal charge. Pass `fallback=False` to
require the selected perceivers. The returned Mol records the actual connectivity and bond-order sources in
`_rxembedConnectivity`, `_rxembedBondOrders`, and `_rxembedPerceptionFallback` properties. When an independent
RDKit/xyz2mol check confirms a nonmetal bond xyzgraph's graph omitted, `read_xyz` restores it and records the
affected `i-j` pairs in `_rxembedConnectivityAdded`; a stretched or contested metal contact is never added this
way.

For a multi-metal XYZ whose total charge does not determine the individual oxidation states, state them by
atom index, for example `rx.read_xyz("complex.xyz", charge=0, metal_charges={0: 2, 1: 1})`; incomplete or
charge-inconsistent assignments fail rather than being redistributed between metals.

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

For tethered sigma donors, `rx.metal` rejects an arrangement when the ligand's distance-geometry upper bound is
shorter than the required donor separation. Embedding can still fail when the full set of metal distances and
angles is inconsistent, and successful arrangements should be checked with an appropriate energy method.

Use `rx.metal` when you want to inspect and select the isomer yourself. The `metal="SPL"` shortcut embeds every
enumerated isomer; `n` applies separately to each one. It returns every isomer that embedded: an isomer that
cannot be built logs one warning line and stays in `all_confs.errors`, and the call raises only when none
embeds. A single selected isomer, such as `trans` above, raises its own `EmbeddingError`.

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
locked_hands = rx.metal(path, stereo={"locked": "racemic"})
```

The measured hand includes that of a donor which is a stereocentre only while bound, such as an amine N or a
σ-alkyl C. Every arrangement carries it, and one that cannot hold it fails to embed. `{"locked": "racemic"}`
instead enumerates both hands of each such centre with every arrangement, leaving the other stereo as measured.

XYZ and SMILES use the same model distances, polyhedron and graph-derived chelate angle preferences. Use
`rx.metal(struct, lengths="input")` to measure M–L distances from coordinates explicitly; the default,
`lengths="model"`, applies regardless of whether the input carries geometry. Every `rx.embed` call generates
new coordinates. Use `rx.minimize(struct)` to relax existing coordinates, or explicit `fix`/`template` to
preserve a selected core during embedding.

Native atropisomer CXSMILES (`wU`/`wD`, reported as `M`/`P`) round-trips and enumerates with metal
arrangements. Alkene `E`/`Z` remains independent of an η² face.

`.summary(details=True)` adds trans pairs or axial/equatorial sites and distinguishes graph-non-equivalent
same-element donors. `center="Mn"` enumerates only Mn and retains the other spheres.

`rx.metal` deduplicates graph-labelled slot assignments under proper polyhedron rotations. For a single metal,
it checks compiled real-atom coordination targets jointly against RDKit's ligand bond/angle reach. This can
reject short trans chelates and incompatible tethered donor networks before embedding. The reach layer keeps
RDKit's native 1-4 interval for ring-closed paths and replaces free-torsion 1-4/1-5 bounds with a
torsion-independent upper envelope elsewhere; compiled bite-angle priors still carry their native geometry
assumptions. Passing this distance-only screen does not prove the remaining stereo, plane or geometry
constraints realizable. A haptic complex still gets its real-atom check, but virtual-centroid rows are
omitted. Haptics, multiple centres and base constraints also retain the prior opposed-span/donor-facing
screen. Explicit `fix` and `observed_only=True` remain authoritative. Pass `screen=False` to retain
symmetry-distinct assignments beyond these reach and donor-facing model limits. This does not expand explicitly
stated CX slots, change ligand stereo or relax embedding checks:

```python
states = rx.metal(smiles, "SPL", screen=False)
```

An exact pool exceeding 1,000 proper-rotation orbits raises; use `rx.metal(mol, observed_only=True)` when only
the measured arrangement is wanted, `rx.embed(mol)` to regenerate its perceived arrangement, or
`rx.metal(rx.cxsmiles(mol))` to request a stated arrangement. The explicit `observed_only` choice does not
assert that omitted assignments are chemically impossible.

An accepted conformer's requested shape reads within `_FIT_MARGIN` (0.01) of the best reading a fresh sphere
gives, not necessarily the argmin: `rx.embed`'s acceptance gate carries both residuals on one RDKit conformer
property, read as `ens.mol.GetConformer(cid).GetProp("shape")`.

Ordinary donor hydrogens may be implicit. Hydrides, H₂, and bridging hydrogen donors must be explicit.

For two independent bidentate ligands in a tetrahedral state, the shared DG/UFF angle preferences accommodate
both bites together. Supported planar bite pairs and tridentate arrangements likewise use joint targets. These
remain soft restraints, not a guarantee of exact planarity or whole-ligand feasibility. See the checked
public-API examples in [07_metal.ipynb](examples/07_metal.ipynb).

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
| 6 | `octahedral` (`OCT`) · `trigonal_prismatic` (`TPR`) · `hexagonal_planar` (`HPL`) |
| 7 | `pentagonal_bipyramidal` (`PBP`) · `capped_octahedral` (`COC`) · `capped_trigonal_prismatic` (`CTP`) |
| 8 | `square_antiprism` (`SQA`) · `dodecahedral` (`DOD`) |
| 9 | `tricapped_trigonal_prismatic` (`TCT`) |
| 10 | `bicapped_square_antiprismatic` (`BSA`) |
| 11 | `edge_contracted_icosahedral` (`ECI`) |

> [!TIP]
> Each unused polyhedron vertex remains an open coordination site.

Haptic ligands, including side-on bonds, Cp, and arenes, bind through a centroid. During cleanup, a finite
restraint keeps the temporary helper near the moving ligand centroid; validation measures the real atoms.

### Ligands

```python
import rxembed as rx

for lig in rx.ligands(complex_mol):
    lig.mol, lig.donors, lig.atoms
```

`lig.donors` maps original metal indices to donor indices in `lig.mol`; `lig.atoms` maps ligand positions
back to original complex indices.

## Save Results

Print dative SMILES for connectivity or CXSMILES to also retain the selected metal arrangement. Use
`dative_smiles(mol, cx=True)` when the connectivity string must retain ligand E/Z or atropisomer stereo but
must remain free of metal slot notes. Zero-order contacts retain RDKit's native CX `Z:` field even in the
connectivity string; plain `~` does not preserve their bond type.

```python
print(rx.dative_smiles(mn_confs.mol))
print(rx.dative_smiles(mn_confs.mol, cx=True))
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
- `minimize()` uses restrained UFF. If native UFF typing fails or assigns an impossible multicoordinate angle
  objective, a private surrogate graph may clean up the geometry while radius-corrected bond holds preserve the
  public element's scale. Such results expose `energy_kind="uff-surrogate"`, element substitutions in
  `uff.surrogates`, and fixed-core bond substitutions in `uff.retyped`. `score("ff")` remains an exact
  public-graph MMFF94s/UFF single point and raises when that graph is not typeable.
- xTB scoring and optimization need the xTB executable.

Record one restrained-UFF cleanup with `trajectory=True`:

```python
walk = rx.embed(selected_isomer, n=1, trajectory=True)
cleanup = walk.trajectory
walk.dump_trajectory("cleanup.xyz")
```

Capture is off by default. `trajectory` is a public RDKit `Mol` containing the seed and retained restrained-UFF
snapshots; `dump_trajectory()` writes multi-frame XYZ without alignment. A path is discarded if a later
acceptance step replaces its endpoint.

Caller-supplied ASE calculators work with `score()`. With `xtb` on `PATH`:

```python
from xtb_ase import XTB

ranked = rx.embed("O", n=1).score(rx.ASE(XTB()))
```

## Messages and Errors

`rx.embed` builds a metal complex in four steps. Distance geometry (DG) makes rough 3D seeds from a table of
allowed atom-atom distances. Restrained UFF cleans each seed while restraints hold the metal polyhedron. A gate
then checks the result: ligand bonds intact, no overlapping atoms, the requested polyhedron and donor sites, the
requested hand (Λ/Δ) and ligand R/S, and any `fix=` values. When the gate fails, rxembed retries, first with
stiffer restraints, then with up to three fresh batches of seeds.

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
  one that could not be built logs one warning line, the error's sentence, and stays in `result.errors`. With
  `stereo="separate"`, a configuration that failed stays in the list as an empty set holding its own `errors`.
  The call raises only when none embeds.
- A warning says the result differs from the request or an input changed: fewer conformers than `n`
  (`kept 3/4 conformers; ...`), conformers without a converged UFF geometry (`ens.unrelaxed`), a `read_xyz`
  perception change, a dropped constraint, or a model choice you may want to override, such as a near-tie
  between two shapes (pass `geometry=`).

The reasons use element symbols and 0-based atom indices:

- `Zn0 relaxes from SPY (0.366) to TBP (0.351)`: the donors relaxed into another polyhedron. The numbers are
  shape misfits, 0 being ideal. Five-coordinate metals swap between these two easily (Berry pseudorotation).
- `Fe0 distorts far from OCT (0.75)`: the sphere fits no shape well enough to name it.
- `Zn0 donors swap sites (another isomer)`: the polyhedron holds, but UFF moved donors to other sites.
- `bond C9-C10 squeezed to 0.87 A`, `stretched to 2.33 A` or `C4...C9 clash at 0.90 A`: the ligand is strained
  in this arrangement.
- `N3 reads R, not S` or `N3 no longer reads S`: a ligand stereocentre, often a coordinated amine N, inverted or
  lost its configuration. For a bound amine N, the remedy names `stereo={'locked': 'racemic'}`: its measured
  hand may not fit this arrangement.
- `found 0/1 DG seeds with the requested metal and ligand stereo`: no seed had the requested hand together with
  the ligand R/S. With fixed ligand stereocentres some hands cannot exist.

`rx.set_verbose()` adds progress lines. `rx.set_verbose("DEBUG")` also shows every retry, grouped by failure and
site: rejected UFF results, stiffness escalation, each replacement batch, and each native DG call's
rejected-attempt counters. A retry is bookkeeping, not a result, and a rejected attempt does not prove chemical
infeasibility. After a successful embed, `ens.relax_failures` records each rejected UFF result by conformer id,
replacement batches included, and `ens.unrelaxed` lists the conformers returned without a converged UFF
geometry.

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

| area | limitation |
|---|---|
| metals | rxembed presents a bondless carbon surrogate to ETKDG and lithium to UFF. M–L targets use the fitted model unless `lengths="input"` is requested. The final metal state is validated |
| coordinate `fix` | ETKDG biases the fixed core, then rxembed restores its coordinates exactly |
| `n=N` | requests N seeds. Failed required structures are retried; an isomer whose requested metal, stereo, or fixed-core count cannot be produced raises `EmbeddingError`, or is skipped with a warning when the call expands into several candidates |
| g-xTB solvent | `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]`, never a silent gas-phase energy |
| haptic axial pose | `rx.metal` selects face and winding, not continuous rotation about the metal-centroid axis. ETKDG may sample it; `mc()` freezes each starting pose |
| `mc()` pose freeze | soft, with about 0.1 Å drift. State a persistent constraint as a distance, angle, or dihedral that relax also reads |

## Development

```bash
just setup     # uv sync + pre-commit
just check     # lint + type + test
```

`uv sync` alone is the full dev environment: the `dev` group pulls `rxembed[search,workflow]`.

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
