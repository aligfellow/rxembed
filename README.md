# rxembed

Constrained conformer embedding for reactive chemistry. State what a geometry must satisfy; get conformers
that satisfy it.

[![License](https://img.shields.io/github/license/aligfellow/rxembed)](https://github.com/aligfellow/rxembed/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/rxembed/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/rxembed/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/rxembed)](https://codecov.io/gh/aligfellow/rxembed)

| you can state | as |
|---|---|
| a distance or angle | `fix={(i, j): 2.05}`; `constrain=` for a soft one |
| a rigid reacting core | `fix=[i, j, k]` or `template=(ref, map)` |
| a coordination polyhedron | `rx.metal(smiles, "octahedral")` |
| a hydrogen bond or NCI grip | `contacts="auto"` |
| a pi stack | `constrain={(ring_a, ring_b): 3.5}` |

They compose. A frozen TS core, a metal sphere and an NCI contact are one `Constraints` struct, not three
code paths.

## Install

Two tiers, visible from `ls src/rxembed/`: flat is core, the one directory needs extras. Not on PyPI yet.

```bash
git clone https://github.com/aligfellow/rxembed.git && cd rxembed
pip install .          # core: numpy + rdkit
pip install '.[all]'   # + pipeline: search, dedup, real energies, NCI, QA, plots
```

Extras: `search select perceive nci viz score sphere`. Real energies shell out to an `xtb` executable on
`$XTB_EXE` (default `~/bin/xtb`). A missing extra names what to install:

```
>>> ens.prune()
ImportError: apply needs prism_pruner; pip install 'rxembed[select]'
```

## Two verbs

Different functions, different signatures. The core one never changes behaviour with which extras are
installed.

| | import | takes | returns |
|---|---|---|---|
| core | `import rxembed as rx` | `Mol` or `Isomer` | `Conformers` |
| pipeline | `import rxembed.pipeline as rx` | SMILES, `.xyz` path, or `Mol` | `Ensemble` / `EnsembleSet` |

`rxembed.pipeline` re-exports every core name, so switching the import is the only change. Parsing is not the
embedder's job, which is why a SMILES or a path needs the pipeline: `read_xyz`, `metal`, `geom_check`,
`Ensemble`, `wrap` and the NCI helpers live there, not in core.

## Core

```python
from rdkit import Chem
import rxembed as rx

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))   # atoms C0 C1 C2 C3 O4
confs = rx.embed(mol, n=8).minimize()           # embed, then restrained-UFF relax
confs.energies                                  # {0: 4.29, 1: 4.89, ...}
confs.dump("out.xyz")                           # or .xyz(), or .mol
```

Constraints are 0-based atom indices in xyz/graph order. Resolve any pattern yourself: an index is
unambiguous, a SMARTS match is not.

### A rigid core

`fix` is rigid, from three sources, and they compose:

```python
rx.embed(mol, fix=[3, 7, 11])                   # from mol's own conformer
rx.embed(mol, fix={3: (x, y, z)})               # from coordinates you supply
rx.embed(mol, template=(ref, {3: 11}))          # from another molecule
```

Grafted atoms are restored exactly and Kabsch-fitted onto each pose. A distance or angle instead of
coordinates is a pull, not a graft, and is reflection-invariant, so it may give the mirror image. Read it back
with `.measure()`.

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

## Pipeline

```python
import rxembed.pipeline as rx

ens = rx.embed("OCCCCO").mc().prune()              # embed -> MC search -> dedup
ens.representatives()                              # one per distinct mode
ens.score("gfn2").lowest(3).optimize("gfn2")       # real energies, then optimise the best

isomers = rx.metal("N->[Pt](Cl)Cl", "square_planar")
rx.geom_check.check(ens.mol, ens.ids[0]).summary()  # 'geometry OK (no violations)'
```

`geom_check` is the acceptance test and needs no extra. It is TS-aware (name the core in `frozen=` and a
forming bond is not read as a clash) and metal-aware (a dative M-L distance is not a clash; it reports a
folded donor).

## Reading a complex back

```python
import rxembed as rx

for lig in rx.ligands(complex_mol):   # each ligand cut from the metal(s) it coordinates
    lig.mol, lig.donors, lig.atoms    # donors: {metal index in the complex: [donor indices in lig.mol]}

rx.dative_smiles(complex_mol)         # SMILES with dative M-L bonds, checked to parse back to the same atoms
```

`lig.mol` is standalone and carries its own copy of the conformer. A bridging ligand has two keys in
`donors`, so it says which centres it spans. `dative_smiles` raises rather than return a string that does not
round-trip, which happens where the perceived graph is one SMILES cannot express.

`->` is RDKit's dative bond, donor to metal, and it is how a complex is written throughout — including the
`rx.metal(…)` input above. It keeps the donor's own valence intact and leaves the metal no bond order to
carry.

## Approximations

Traded for a robust, stackable embed. Stated rather than hidden.

| | |
|---|---|
| metal | a bond-less carbon in the distance geometry, a bond-less lithium in the force field. M-L bonds are stripped and the sphere is held by soft constraints. M-L length is a fitted periodic model, not a radius sum |
| exact cores | grafted, not embedded. Distance geometry approximates a rigid core to ~0.2-0.4 A, fine for a molecule and wrong for a TS |
| `n=N` on a metal | N geometries that pass the gate, not N attempts |
| g-xTB solvent | `E_gxtb(gas) + [E_gfn2(solv) - E_gfn2(gas)]`, never a silent gas-phase energy |
| `mc()` pose-freeze | soft, ~0.1 A drift. A constraint that must hold across `mc()` has to be a distance or angle the relax also reads |

## More

- [`examples/`](examples/): 12 notebooks, each gated by `geom_check.check`, from the bare bounds matrix to
  transition-metal catalysts searched and ranked.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): how it fits together.
- [`AGENTS.md`](AGENTS.md): how we change it.

## Development

```bash
just setup     # uv sync + pre-commit
just check     # lint + type + test
```

`uv sync` alone is the full dev environment: the `dev` group pulls `rxembed[all]`.

## License

[MIT](LICENSE)

## References

- [RDKit](https://github.com/rdkit/rdkit): embedding, ETKDG, UFF
- [openconf](https://github.com/rowansci/openconf): additional conformational sampling
- [prism_pruner](https://pypi.org/project/prism-pruner/): conformer pruning
- [xyzgraph](https://github.com/aligfellow/xyzgraph): `.xyz` metal and TS perception
- [xtb](https://github.com/grimme-lab/g-xtb): GFN-FF, GFN2-xTB and g-xTB
