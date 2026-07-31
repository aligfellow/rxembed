# rxembed

Fast, flexible molecular embedding for reactive chemistry.  

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

What you can state: a distance, an angle, a rigid reacting core, a coordination polyhedron, a hydrogen-bond
grip. Each is one more argument on the same call.

The embedder needs numpy and rdkit. Conformer search, dedup, real energies, NCI discovery and the geometry
gate are the optional `rxembed.pipeline` half.

## Install

Two tiers, and the split is visible from `ls src/rxembed/`: flat is core, the one directory needs extras.
Not on PyPI yet, so both are from a checkout.

```bash
git clone https://github.com/aligfellow/rxembed.git && cd rxembed

pip install .          # the embedder: numpy + rdkit, nothing else
pip install '.[all]'   # plus the pipeline: search, dedup, real energies, NCI, QA, plots
```

Individual extras exist (`search`, `select`, `perceive`, `nci`, `viz`, `score`, `sphere`) if you want one
stage without the rest.

Real energies (`score` / `optimize` with `'gfnff'`, `'gfn2'`, `'gxtb'`) shell out to an `xtb` executable
on `$XTB_EXE` (default `~/bin/xtb`), so no extra installs them. GFN-FF needs a standard Grimme `xtb` on that
same path.

A missing extra never gives you a bare upstream traceback. It names what to type:

```
>>> ens.prune()
ImportError: this needs prism_pruner; pip install 'rxembed[select]'
>>> ens.landscape()
ImportError: this needs matplotlib; pip install 'rxembed[viz]'
```

> **openconf from git.** `mc()`'s transition-metal support (`preset='transition_metal'`) is on openconf's
> upstream `main`, not on a PyPI release, so `pyproject.toml` pins it via `[tool.uv.sources]`.
> `uv sync` resolves and locks it; a plain `pip install 'rxembed[search]'` gets the PyPI version
> without that preset. Without openconf at all, `mc()` warns and returns the ETKDG seeds; it does not crash.

## Quick start on a base install

`rxembed` is a library, driven from Python or a notebook. The core surface is three verbs. `rx.embed`
takes an RDKit `Mol` (or an `Isomer`) and returns a `Conformers`. Parsing is deliberately not the
embedder's job, so a base install never needs a perception dependency.

```python
from rdkit import Chem
import rxembed as rx

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))   # atoms C0 C1 C2 C3 O4

confs = rx.embed(mol, n=8).minimize()           # embed, then restrained-UFF relax
len(confs)                                       # 6, ETKDG having pruned duplicates at 0.1 A RMSD
confs.energies                                   # {0: 4.29, 1: 4.89, 2: 3.09, 3: 4.92, 4: 3.51, 5: 4.33}
confs.dump("out.xyz")                            # or confs.xyz(), or confs.mol for raw RDKit
```

Constraints are 0-based atom indices in xyz/graph order. The resolver never SMARTS-matches internally, so
resolve any pattern yourself first (two RDKit lines): an index is unambiguous and a match is not.

```python
held = rx.embed(mol, fix={(0, 4): 3.0}, n=8).minimize()   # hold C0···O4 at 3.0 A while the rest is free
# measured back on the first conformer: 3.02 A

relaxed = rx.minimize(confs.mol, fix={(0, 4): 3.4})       # no search: pull an existing geometry to the target
# measured back: 3.38 A
```

A `fix` stated as a number is a tight-window UFF pull rather than a snap, so read it back with
`.measure()` — that is where the numbers above come from. A `fix` stated as coordinates is different: it is
restored exactly.

```python
held.measure((0, 4))       # {'mean': 3.02, 'min': 3.00, 'max': 3.04, 'n': 6}
```

**A rigid core comes back exactly.** Name the atoms and their geometry survives a random-frame embed
untouched, which is the point of the whole exercise for a transition state, whose partial bonds distance
geometry would otherwise average away:

```python
from rdkit.Chem import rdDistGeom

ref = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))
rdDistGeom.EmbedMolecule(ref, randomSeed=1)
core = [0, 1, 2, 3]                              # the amide C-C(=O)-N, held at the source's own coordinates

confs = rx.embed(ref, fix=core, n=6).minimize()
# max deviation of any core distance from the reference, over every conformer: 0.0000 A
```

**A metal is the same `embed`, given an arrangement.** `enumerate_isomers` turns a `Mol` and a named
polyhedron into the distinct coordination isomers; each one embeds like anything else.

```python
pd = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl"))
isomers = rx.enumerate_isomers(pd, "square_planar")

[i.label for i in isomers]      # ['cis', 'trans']
isomers[0].summary()            # 'square_planar | N0 N2 Cl3 Cl4 | achiral'

confs = rx.embed(isomers[0], n=4).minimize()
Chem.MolToSmiles(confs.mol)     # '[H][N]([H])([H])->[Pd](<-[Cl])(<-[Cl])<-[N]([H])([H])[H]'
```

Note the dative bonds in the input SMILES. `N[Pd]` writes a covalent N (→ NH₂); a real ammine is
`[NH3]->[Pd]`. The metal is stripped to a bond-less surrogate for the force field, and `.mol` restores its
element, oxidation state and M–donor dative bonds.

Two reads in the other direction. `rx.ligands` cuts a complex at its M–L bonds and returns each ligand as a
standalone `Mol` carrying its own copy of the input coordinates, with `donors` keyed by metal so denticity,
hapticity and bridging all survive the cut. `rx.dative_smiles` writes the complex back out as a SMILES, so a
geometry you read can become a string you edit and re-embed.

```python
mol = rx.read_xyz("complex.xyz")
[(Chem.MolToSmiles(lig.mol), lig.donors) for lig in rx.ligands(mol)]
rx.embed(rx.metal(rx.dative_smiles(mol), "octahedral")[0], n=5)
```

Round-tripping a complex is not free: a hydrogen with a second connection (a side-on H₂, a bridging hydride,
an H-bond relay perceived as a bond) has no valence left to write, so those bonds are made dative.
`dative_smiles` raises rather than return a string that will not parse back. Enumerating one centre of a
multi-metal complex still needs the geometry, since the spectator metal is retained from it.

## A rigid core

`fix` is rigid: those atoms will have that geometry. One question, three places the coordinates can come
from, and they compose.

```python
rx.embed(mol, fix=[3, 7, 11])                      # from mol's own conformer
rx.embed(mol, fix={3: (x, y, z)})                  # from coordinates you supply
rx.embed(mol, template=(ref, {3: 11}))             # from another molecule
```

Grafted atoms are restored exactly (0.000 A) and Kabsch-fitted onto each embedded pose; everything else is
conformer-searched. Coordinates carry handedness. A distance or angle instead of coordinates is a pull rather
than a graft, and is reflection-invariant, so it may give the mirror image:

```python
rx.embed(mol, fix={(3, 11): 2.05, (3, 11, 12): 178})   # verify with .measure()
```

## Put a TS core on a fresh molecule

Base install, public API only.

```python
from rdkit import Chem
import rxembed as rx

ref  = Chem.MolFromXYZFile("ts.xyz")        # coordinates only, no perception needed
core = [4, 0, 5]                            # F, C, Cl: read once off the reference

new  = Chem.AddHs(Chem.MolFromSmiles("[F-].c1ccccc1CCl"))
hit  = rx.match(new, "[F-].[#6]-[Cl]")      # raises if the pattern misses

confs = rx.embed(new, template=(ref, dict(zip(hit, core))), n=4)
confs.measure(hit[:2])                      # {'mean': 1.717, ...}, the reference distance, held
```

- the map is `{atom in the new molecule: atom in the reference}`, both 0-based
- `zip` pairs them in order, so the core list must be in the same order as the SMARTS hit
- SMARTS builds the map; the library never matches for you, because one pattern usually has several matches
  and a resolver picking one would flip a core silently
- an empty map raises rather than quietly embedding with no template

With the `perceive` extra, `rx.read_xyz` gives the reference perceived bonds, so you can SMARTS both sides
and read no indices at all. That works when the anchor sits on a motif the reaction does not change; where
the reaction makes and breaks bonds at the core, no single pattern matches both sides.

## The full chain

Install the extras and change one word: `import rxembed.pipeline as rx`. The pipeline `embed` takes a SMILES
string, an `.xyz` path or a Mol, and returns a chainable `Ensemble`. It returns an `EnsembleSet` when the
input is inherently several candidates: metal isomers, NCI binding modes, a racemate.

```python
import rxembed.pipeline as rx

ens = rx.embed("OCCCCO").mc().prune()            # embed -> Monte-Carlo search -> dedup
ens.representatives()                             # one conformer per distinct mode
ens.score("gfn2").lowest(3).optimize("gfn2")      # real energies, then optimise the best few
```

The two `embed`s are different functions with different signatures, and the root one does not change
behaviour with which extras are installed. `rxembed.pipeline` re-exports every other core name
(`Constraints`, `enumerate_isomers`, `set_verbose`, …), so switching the import is the only change.

- free or flexible: `rx.embed("CCO")`
- a soft distance, angle or π-stack window: `rx.embed(smi, constrain={(i, j): (2.6, 3.0)})`
- NCI modes discovered for you: `rx.embed("A.B", contacts="auto")`, one candidate per grip
- one specific grip: `rx.embed("A.B", contacts=rx.nci_modes(mol)["HB:…"])`
- a frozen TS core from an .xyz: `rx.embed("ts.xyz", fix=reacting)`
- a TS from SMILES: `rx.embed(smi, fix={(i, j): d, (i, j, k): θ})`, then `ens.measure((i, j))`
- a known TS onto a fresh molecule: `rx.embed(smi, template=(reference, {target_i: ref_i}))`
- relax a structure toward a core: `rx.minimize("mol.xyz", fix={(i, j): 2.0, (i, j, k): 178})`
- metal coordination isomers: `rx.embed(smi, metal="square_planar")`, or `rx.metal(...)` for the isomers alone
- the racemate of an undefined centre: `rx.embed("CC(N)C(=O)O")`
- real energies: `ens.score("gxtb")` or `ens.optimize("gxtb", level="loose")`

**Binding modes are discovered, not guessed.** `contacts='auto'` enumerates the grips two fragments can
actually form and hands back one candidate per grip, each searched and pruned within itself. Distinct
species are never pooled or cross-pruned:

```python
>>> modes = rx.embed("O=C(N)c1ccccc1.OC(=O)C", contacts="auto")   # benzamide + acetic acid
>>> [c.tag["nci"] for c in modes]
['HB:H20->O0 + HB:H13->O9',
 'HB:H20->O0 + HB:H13->O11',
 'HB:H13->O11 + HB:H20->N2',
 'HB:H13->O9',
 'HB:H20->N2']
```

Which verbs change the ensemble:

- `mc`, `minimize`, `prune`, `filter` change it and return it, so they chain
- `lowest`, `representatives`, `align`, `score`, `optimize` return a new one
- `measure`, `cluster`, `landscape` change nothing

`rx.set_verbose("INFO")` narrates every stage. Energies are tagged `ff` (surrogate force field, not
comparable across species) or `real` (xtb or g-xTB, which is). `EnsembleSet.best(n)` is the one cross-species
ranking and refuses `ff` energies rather than ranking on them.

### The geometry gate

A perfect frozen core can coexist with a chemically wrong periphery: a puckered ring, a twisted amide, a bad
H position, two atoms on top of each other. `geom_check.check` is the physical acceptance test, and it is pure
RDKit + NumPy (no extra needed):

```python
>>> rx.geom_check.check(ens.mol, ens.ids[0]).summary()
'geometry OK (no violations)'
```

It is TS-aware: name the reacting core in `frozen=` and forming or breaking bonds are not misread as
clashes. It is metal-aware too: a dative M–L distance is not a vdW clash, and it reports a folded donor, a
ligand pointing the wrong way off its donor atom, which every distance-based check is blind to.
`rep.assert_ok()` raises with a readable summary.

### Modes and the ensemble map

Each conformer is embedded in a latent: its dihedral angles, plus, when present, an inter-fragment
NCI-contact signature and metal-coordination features. That one latent drives both the dedup and the picture.
A "mode" is whatever the latent separates: a conformer family (organic), a binding grip (NCI), or a ligand
arrangement (metal).

- `representatives()` gives one lowest-energy conformer per mode: the distinct-shapes summary.
- `cluster()` labels each conformer by mode (HDBSCAN; `-1` is rare or noise).
- `landscape(method="pca"|"tsne", color="cluster"|"energy")` projects that latent to 2D. Conformers a
  `prune()` merged away are drawn faded, so you see exactly what was collapsed.

## Approximations

Where exactness is traded for a robust, stackable embed, stated rather than hidden:

- **Metal = surrogate.** In distance geometry the metal is a bond-less carbon (its excluded volume stops a
  ligand folding into the centre); in the force field a bond-less lithium, whose small vdW radius keeps
  every non-donor off the metal. M–L bonds are stripped and the sphere is held by soft shape constraints, so
  `restore` must hand back the oxidation state, not just the element, or every real-energy call runs at the
  wrong total charge. M–donor distance is a fitted periodic model, not a covalent-radius sum.
- **Exact cores are grafted, not embedded.** Distance geometry only approximates a rigid core (~0.2–0.4 Å
  internal RMSD, fine for a normal molecule and wrong for a TS). The core is restored exactly and Kabsch-fitted
  onto the embedded periphery.
- **N good geometries, not N attempts.** For a metal, `n=N` re-embeds fresh seeds until N conformers pass the
  gate; a seed the relax tears is a re-embed, not a kept result.
- **Solvent for g-xTB is a thermodynamic cycle.** g-xTB has no implicit solvent, so a solvated g-xTB energy is
  `E_gxtb(gas) + [E_gfn2(solv) − E_gfn2(gas)]`, a real solvated energy and never a silent gas-phase one.
- **openconf's pose-freeze is soft** (held atoms drift ~0.1 Å), so a constraint that must truly hold across
  `mc()` has to be one the rxembed relax also reads, a distance or an angle, not a pose-hold alone.

## Examples

Twelve notebooks in [`examples/`](examples/), each gated by `geom_check.check`:

- `00` the bounds matrix, built by hand
- `01` embedding, and controlling it with `fix` and `constrain`
- `02` distance, angle and plane windows
- `03` NCI binding modes
- `04` organic transition states
- `05` a template on a fresh molecule
- `06` an organocatalysis backbone swap
- `07` the coordination sphere: isomers, geometries, chelates, hapticity
- `08` real energies
- `09` assembling a TS with no reference
- `10` retargeting a DFT TS by SMILES
- `11` transition-metal catalysts, searched and ranked

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
git clone https://github.com/aligfellow/rxembed.git
cd rxembed
just setup   # uv sync + pre-commit (openconf is pulled from git automatically)
just check   # lint, type-check, tests
```

- `just check` runs lint, type-check and tests
- `just lint`, `just type`, `just test`, `just build`, `just setup`

A bare `uv sync` gives the full dev environment: the `dev` dependency group self-references
`rxembed[all]`, so no recipe passes `--all-extras`, which would mask a regression in the default.

`benchmark/run.py` rebuilds 45 known coordination geometries from the graph and diffs the result against a
committed baseline. Re-run it after any change to the bounds matrix, the mechanisms or the metal stack.

CI runs lint, type-check and the suite on every push to `main` and every PR. Coverage goes to
[Codecov](https://codecov.io). The tier rule is enforced in the suite: `tests/test_init.py` walks the AST for
any core module reaching past numpy + rdkit, and `tests/pipeline/test_init.py` checks every optional import
is guarded.

[`AGENTS.md`](AGENTS.md) is the working discipline; assess → plan → implement → adversarial review → regress.
[`ARCHITECTURE.md`](ARCHITECTURE.md) is the assembly.

## License

[MIT](LICENSE)

## References

- [RDKit](https://github.com/rdkit/rdkit): embedding, ETKDG, UFF
- [openconf](https://github.com/rowansci/openconf): additional conformational sampling
- [prism_pruner](https://pypi.org/project/prism-pruner/): conformer pruning
- [xyzgraph](https://github.com/aligfellow/xyzgraph): `.xyz` metal and TS perception
- [xtb](https://github.com/grimme-lab/g-xtb): GFN-FF, GFN2-xTB and g-xTB

## Acknowledgements

Generated from [aligfellow/python-template](https://github.com/aligfellow/python-template).

<details>
<summary>Updating from the template</summary>

If this project was created with [copier](https://copier.readthedocs.io/), you can pull in upstream template improvements:

```bash
# Run from the project root
copier update --trust
```

This will:

1. Fetch the latest version of the template
2. Re-ask any questions whose defaults have changed
3. Re-render the templated files with your existing answers
4. Apply the changes as a diff; your project-specific edits are preserved via a three-way merge

If there are conflicts (e.g. you modified the `justfile` and so did the template), copier will leave standard merge conflict markers (`<<<<<<<` / `>>>>>>>`) for you to resolve manually.

The `--trust` flag is required because the template defines tasks (used for `git init` on first copy). The tasks don't run during update, but copier requires trust for any template that declares them.

Requires that the project was originally created with `copier copy`, not the plain GitHub "Use this template" button.

</details>
