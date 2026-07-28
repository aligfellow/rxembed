<!-- DRAFT of src/rxembed/rdkit_embed/README.md — describes the TARGET surface (post Move 1). Graduates to the
     package README once embed()/Isomer/enumerate_isomers/Conformers exist. Keep it this short. -->

# rdkit_embed

Constrained, coordination-aware conformer embedding on top of RDKit's ETKDG.

It does the one thing RDKit's `EmbedMolecule` can't: hold **your constraints** (distances, angles, a frozen
core) and **metal coordination** (a real polyhedron, dative geometry, Λ/Δ) while it seeds — then relax against
the same constraints. Pure `numpy` + `rdkit`; `.mol` is always a plain RDKit `Mol`, so there's no lock-in.

## Install
```bash
pip install rdkit_embed          # numpy + rdkit;  scipy is an optional 'sphere' extra
```

## Quick start
```python
from rdkit import Chem
from rdkit_embed import embed, Isomer, enumerate_isomers

mol = Chem.AddHs(Chem.MolFromSmiles("OCCO.O=C(C)C"))

# organic: a soft distance window + a pinned pair, then relax, then write it out
embed(mol, constrain={(2, 8): (2.6, 3.2)}, fix={(0, 1): 1.43}, n=24).minimize().dump("out.xyz")

# a metal complex — usually you DON'T know the vertex numbering: enumerate, look, pick
isos = enumerate_isomers(mol_with_dative_bonds, "OCT")    # geometry is a 3-letter code (OCT, TBP, SPL, …)
isos.summary()                                            # prints:  index · arrangement · Δ/Λ  — no vertex numbers
for iso in isos.filter(chirality="delta"):                # typeable: "delta"/"lambda" (case-insensitive)
    embed(iso, n=24).minimize().mol

# if you DO know the exact seating (e.g. from your own perception), name it directly:
iso = Isomer(mol_with_dative_bonds, "OCT", sites=[3, 7, 11, 15, 19, 23])   # donor atom at each vertex
embed(iso, n=24).minimize(distance_fc=1e5).mol
```

## The API (small on purpose)
```python
embed(spec, *, fix=None, constrain=None, n=…, seed=…, prune_rms=0.1, threads=0) -> Conformers
Isomer(mol, geometry, sites)                       # geometry = a 3-letter code (OCT, TBP, SPL, …); sites[vertex] = donor atom
enumerate_isomers(mol, geometry) -> IsomerSet      # the distinct ones
```
Geometry is a 3-letter code: `LIN TPL SPL TET TBP SPY OCT PBP SQA`. `sites` values are real atom indices; the
vertex numbering follows the geometry's `vertex_dirs(code)` (exposed), and `Isomer` validates an impossible seating.

| object | what you get |
|---|---|
| **`Conformers`** | `.minimize(distance_fc=…)` (constrained UFF relax) · `.mol` (RDKit Mol, all conformers) · `.dump(path)` / `.xyz(i)` · `.ids` · `len()` · `confs[i]` |
| **`IsomerSet`** | `.select(...)` / `.filter(...)` on `arrangement` / `chirality` (`"delta"`/`"lambda"`) / `geometry` / `index` · `.summary()` |

The two verbs are all you pass:

| verb | is | e.g. |
|---|---|---|
| `fix=` | a rigid hold — an exact distance/angle, or a frozen core (Kabsch-grafted onto the seed, pinned through the relax) | `{(i,j): 1.9}` · `{(i,j,k): 90}` · `{i: (x,y,z)}` |
| `constrain=` | a soft window (biases the seed) | `{(i,j): (2.6, 3.2)}` · `{(i,j,k): (100, 140)}` |

Metal coordination rides on the `Isomer`. `Constraints`/`compose` exist as an escape hatch for hand-authoring;
you almost never need them.

## How it works
1. Your `fix`/`constrain` (+ an `Isomer`'s coordination) are resolved into edits on **RDKit's own ETKDG bounds
   matrix**, then embedded with `EmbedMultipleConfs`.
2. `.minimize()` relaxes the seeds with a **restrained UFF/MMFF** that holds those same constraints.
3. A metal is a bond-less **surrogate** during the distance-geometry and force-field steps (UFF can't type a
   transition metal) — this is entirely internal; `.mol` hands back the real element, oxidation state, and M–L
   dative bonds. Λ/Δ is a *derived* label (distance geometry is mirror-invariant), so you `filter(chirality=…)`
   the result rather than requesting a hand up front.

## Dependencies
`numpy`, `rdkit`. `scipy` is an optional `sphere` extra for a rarely-reached bounds-repair fallback.

## Consumed by
`rxembed` (conformer search: adds `.mc/.prune/.score` on top of `Conformers`), `OIN-SMILES` (coordination-string
round-trip), `rxnts` (metal reaction-TS). All speak the same `Mol`/`Isomer` + `fix`/`constrain` contract.
