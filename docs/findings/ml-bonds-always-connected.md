# The user-facing `Ensemble.mol` always carries the M–L bonds

## The requirement

The molecule a caller reads back from a metal complex must ALWAYS be a proper connected molecule — the real
metal element + its oxidation state + the M–donor **DATIVE** bonds — on **any** access, at **any** stage.
"input smiles has them, and should always be returned with them. Otherwise it's unclear what happens."

## The gap this closes

Commit `44d2828` added a connectivity finalize (`metal.connect_metal`, re-adds the surrogate-stripped M–L
bonds as DATIVE) but ran it **only inside `Ensemble.minimize()`**. So the terminal/derived paths
(minimize/score/optimize/representatives/lowest/align/dump) came back connected, but a **bare
`rx.embed(metal).mol`** (no `.minimize()`) — and `rx.embed(metal).mc().mol` — still returned the bond-less
**surrogate** (carbon at the metal, M–L bonds stripped, `GetMolFrags` > 1). "Always connected" was not
guaranteed on every access.

## The hard constraint

The **internal working mol must stay the bond-less surrogate** (real metal element off, M–L bonds stripped)
so the DG/FF embed + relax runs — UFF cannot type a bonded transition metal (that is why `mc` strips the
bonds before re-searching). So this cannot just "finalize eagerly and store" — it must **decouple** the
engine's working mol from the user-facing graph.

## The design — the `_mol` / `.mol` split

`Ensemble`'s stored field is renamed `mol` → **`_mol`**: the internal working mol, the bond-less carbon/Li
surrogate every DG/FF stage relies on. `.mol` is now a **read-only `@property`** that finalizes the connected
user-facing graph **on access**, from `_mol`:

```python
@property
def mol(self):
    mol, mc = self._mol, self._metal
    bonds = self.metal_bonds or (list(mc.donor_bonds) if mc is not None else [])
    if mc is not None:               # pre-minimize: _mol is still the carbon surrogate, M–L stripped —
        mol = Chem.Mol(mol)          # restore the real element(s)/charge on a COPY (never touch the working mol)
        for mi, rz, rq in [(mc.metal, mc.real_z, mc.real_q), *mc.extra]:
            _metal.restore_metal(mol, mi, rz, rq)
    return _metal.connect_metal(mol, bonds) if bonds else mol   # idempotent; no-op if already bonded
```

Properties of the accessor:

- **Reuses the existing machinery** — `metal.restore_metal` (element + oxidation state) and
  `metal.connect_metal` (the DATIVE re-add), the same two functions `minimize()`'s finalize runs. No logic is
  duplicated.
- **Never mutates the working mol.** When finalization changes anything (pre-minimize metal) it acts on a
  `Chem.Mol(...)` copy. The engine's `_mol` is untouched — verified by a test that reads `.mol` and then
  asserts `_mol` is still the bond-less carbon surrogate.
- **Connectivity-only.** `restore_metal`/`connect_metal` swap the element and add bonds; they never move an
  atom. A test asserts `.mol` coordinates are byte-identical to `_mol`'s.
- **Idempotent / no-op where nothing is owed.** Post-minimize `_mol` is already restored + connected, so
  `connect_metal` returns it unchanged (`mc is None`, bonds already present) — `.mol is _mol` for that object,
  as before the split. **Organic** inputs have no `_metal` and no `metal_bonds`, so `.mol` returns `_mol`
  itself — a pure no-op, identity preserved.

### Stage-by-stage

| Stage | `_metal` | `metal_bonds` | `_mol` state | `.mol` returns |
|---|---|---|---|---|
| bare embed (metal) | live | ∅ | carbon surrogate, no M–L | copy: restore + connect (via `mc.donor_bonds`) |
| embed → mc (metal) | live | ∅ | surrogate, no M–L | same as bare |
| minimize (metal) | None | set | real element, connected | `_mol` (idempotent connect) |
| minimize → mc (metal) | None | set | real element, M–L stripped by `disconnect_metal` | copy: re-connect from `metal_bonds` |
| score/optimize/derive | None | set | connected copy | idempotent |
| derived from *pre*-minimize (`align()`, `[0]`) | live (rebound to copy) | ∅ | surrogate copy | copy: restore + connect |
| organic (any stage) | None | ∅ | plain mol | `_mol` (no-op) |

## Readers re-routed to `_mol`

Every reference **inside** the `Ensemble` class was mechanically re-pointed from `self.mol` → `self._mol` (77
sites) and `batch.mol` → `batch._mol` (the re-embed sub-ensemble). These are all engine/gate consumers that
must see the surrogate — the FF relax (`restrained_uff`, `ff_energies`), the geometry gate
(`_geometry.check` in `_reembed_until_clean`), dedup/cluster/landscape, `_scan_connectivity` (which does its
**own** surrogate→real element mapping and would double-handle a finalized mol), `measure`, `align`, and the
`Chem.Mol(self._mol)` copies that seed `score`/`optimize`/`_derive`/`dump`. `dump` keeps copying from `_mol`
and doing its own element restore, so its XYZ output is byte-identical (XYZ has no bond records anyway).

External modules that read `ens.mol` (`viz`, `_reference_positions` in dispatch) only do so on post-minimize
(already-connected) ensembles or for pure coordinate reads, so the property is transparent to them.

### Two white-box **tests** re-routed to `_mol`

Two engine tests grabbed a **pre-minimize** metal ensemble's graph through `.mol` to inspect engine
internals that operate on the surrogate:

- `test_connectivity.py::test_ff_surrogate_is_a_small_soft_sphere_not_a_hole_or_a_wall` — measures the
  bond-less **Li surrogate**'s own UFF vdW.
- `test_coplanar.py::_emitted_cap_donors` (feeds `test_the_ff_torsion_is_skipped_…`) — `Coplanar.ff_terms`
  reads `conf.GetOwningMol()`, so a finalized conformer's extra DATIVE bonds + real element would change which
  coplanar caps are judged redundant.

Both unambiguously want the working surrogate, so they now read `ens._mol` (with an explanatory comment).
This is the same internal-consumer re-route applied at the test layer — not a force-green: on the old code
`.mol` *was* the surrogate for a pre-minimize ensemble, and `_mol` restores exactly that. The bit-identity
control (`test_the_controls_keep_every_ff_cap_bit_identical`) stays green.

## What the geometry gate sees is unchanged

The pipeline feeds `_geometry.check` `self._mol` (and `batch._mol`) inside `_reembed_until_clean` — the exact
mol it saw before the rename (parent: real element, no dative bonds yet; `batch`: connected, because its own
`minimize` finalized it). Golden fixtures capture the DG bounds matrix + `Constraints` at embed time and never
run `minimize`, so the accessor never fires during capture.

## Test coverage

- `tests/test_ml_connected.py` (new, 15 cases): bare `.mol` connected (real element + oxidation state + DATIVE
  donor→metal direction); the working `_mol` stays the bond-less carbon surrogate and `.mol` access does not
  mutate it; finalize is connectivity-only (coords identical); every mol-returning path parametrized (bare,
  minimize, score('ff'), representatives, lowest, align, `[0]`, prune, derived-from-pre-minimize); `.mc().mol`
  connected (openconf-gated); each bare `EnsembleSet` candidate connected; organic `.mol is _mol` no-op.
- `tests/test_output_connectivity.py` (pre-existing) continues to pin the post-`minimize` terminal paths.

## Gates

- Golden bit-identical: `tests/golden/` 23/23.
- Full suite: 380 passed / 2 skipped (365 baseline + 15 new).
- ruff check + format clean.

## Residual caveats

- The pre-minimize `.mol` builds a fresh finalized copy on **each** access (no cache) — correct and safe (it
  is a user-facing read, never a hot loop; all internal hot paths use `_mol`), but not free. A cache with
  geometry-mutation invalidation was deliberately not added to keep the invariant obvious.
- A haptic (η³+) donor's DATIVE bonds are re-added from every ring-atom→metal pair recorded in `donor_bonds`
  (e.g. η⁵-Cp → five dative bonds). That is a faithful connectivity, not a bond-order claim.
