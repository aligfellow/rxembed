# rdkit_embed — the rxembed embedding kernel

The pure embedding engine. It embeds conformers for anything RDKit can represent — a bare molecule, a
constrained transition state, a metal complex — driven entirely by a `Constraints` struct. It depends on
**numpy + rdkit only**; scipy (the sphere solver) and xyzgraph (perception) are optional, guarded fallbacks.

## The idea

Any embedding, or a *combination* of embeddings, is possible by passing constraints. Most constraints are set
up **automatically** — metal coordination, NCI grips, donor orientation, haptic centroids, stereo holds — but
you can also hand-pass your own: distances, angles, frozen cores, π-stack planes.

The engine **edits RDKit's ETKDG bounds matrix** from those constraints and relaxes into them. It does not
fight ETKDG's chemical knowledge — it biases it: the experimental-torsion / basic-knowledge seeding survives
even under a tight custom core.

## Use

```python
from rxembed.rdkit_embed import embed, Constraints
```

The kernel lives inside `rxembed` for now, so it imports as `rxembed.rdkit_embed`. The placeholder
`pyproject.toml` alongside this README marks the intent to graduate it to a standalone `rdkit_embed` package
once the kernel stabilises.

## Still to do

- **Decide the fate of the scipy sphere solver (`constraints/sphere.py` + `solver.py`).** The numpy-rewrite
  idea was **measured and rejected** (`docs/findings/scipy-to-numpy-sphere.md`): a small numpy solver is
  strictly less robust on the exact contradictory cases the scipy solver exists for. So the numpy rewrite is
  off the table. scipy is now a *declared* optional extra (`sphere = ["scipy"]`) — the only open question is a
  maintainer keep-vs-delete call on the whole solver (it settles ~2% of cases and is unreachable on the base
  install; deleting it removes scipy from the kernel entirely). Until then, keep + declare.
- ~~**Reconsider `io._xyz_to_mol`.**~~ **Done.** xyz→mol is user / IO adaptation, not an embed-engine concern,
  so `_xyz_to_mol` + `parse_smiles` moved out to the shell leaf `rxembed.inputs`; the kernel `embed()` already
  takes a mol. Only `repair_bond_stereo` (a Mol→Mol repair the metal surrogate calls) stays in `io.py`, and
  `xyzgraph` leaves the kernel's dependency closure with the readers.
- **Graduate the package.** Once the kernel stabilises, split `rdkit_embed` out to a standalone package (a uv
  workspace member again, or its own repo).
