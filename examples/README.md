# rxembed examples — the breadth tour

Ten notebooks, simple → complex. Each is the **same chain** (`embed → mc → prune → score`) with more
arguments, and each calls `rx.geometry.check(...)` as its own acceptance gate — so the tour is *live proof*
the embeds are chemically sound, not prose. Almost no other tool can do the TS / templated-TS / metal / NCI
range shown here on real chemistry.

Run them with the repo on the path (`PYTHONPATH=../src`, or `pip install -e .`). Real backends used:
**RDKit** (embed), **openconf** (`mc`), **xyzgraph** (`.xyz` perception), **g-xTB** via `~/bin/xtb`
(`score`/`optimize`). Missing optional backends degrade gracefully rather than crash.

| # | Notebook | Showcases |
|---|----------|-----------|
| 01 | [`01_basics`](01_basics.ipynb) | the `embed → mc → prune` chain; the mutation contract; all four dedup methods; the geometry gate; the ensemble landscape; `dump` |
| 02 | [`02_constraints`](02_constraints.ipynb) | soft `constrain=` windows — distance / angle / **π-stack plane** (index-driven; resolve SMARTS to indices yourself); `measure()`; incremental relaxing on a tight core; fail-loud on an impossible constraint |
| 03 | [`03_nci`](03_nci.ipynb) | `contacts='auto'` binding-mode discovery; **halogen / chalcogen / salt-bridge** contacts; the bifurcated thiourea; the gate as an acceptance filter; `mc(explore=True)` |
| 04 | [`04_organic_ts`](04_organic_ts.ipynb) | frozen TS core from `.xyz` (0.000 Å); constrained TS **from SMILES** (SN2, FLP borylation); the **TS-aware gate** (`frozen=`); **capstone** — the 172-atom bimp TS with **multiple NCI binding modes** at the frozen core |
| 05 | [`05_templated_ts`](05_templated_ts.ipynb) | **a known TS onto a fresh SMILES** — `template=(reference, {target_i: ref_i})`; a substrate **series** on one pinned core; an `Ensemble` as a template |
| 06 | [`06_organocatalysis`](06_organocatalysis.ipynb) | **TS transfer across scaffolds** — the isothiourea backbone swap (tetramisole → BTM → HyperBTM), conserved amidine core at 0.000 Å; the S···C=O chalcogen activation; a real chiral-phosphoric-acid TS |
| 07 | [`07_metal`](07_metal.ipynb) | the full metal space (24 cells): **isomers** (cis/trans, mer/fac); across geometries; **chelates** (bidentate en bite); **vacant pockets**; three real bimetallic/organometallic **TS** (mn-h2 / mn-hy / ru-co, spectator ferrocenes, core held 0.000 Å) + free-site enumeration; **seating a substrate and switching its binding mode** (coordinate → bifunctional) |
| 08 | [`08_energies`](08_energies.ipynb) | the `ff → gfnff → gfn2 → gxtb` tier ladder; the GFN-FF pool re-rank → g-xTB pipeline; honest, non-clobbering energies |
| 09 | [`09_assemble_ts`](09_assemble_ts.ipynb) | **build a TS with no reference geometry** — a core stated as **internals** (`fix={(i,j):d,(i,j,k):θ}`) or **explicit coordinates** (`fix={i:(x,y,z)}`), fresh cat/substrate/**third fragment** stitched as `.`-disconnected SMILES, core atoms resolved by SMARTS inline; the **payoff** — a real-energy-ranked **reactive-complex / TS-guess generator** screening across substrates (what no single-molecule conformer tool can assemble); contrast with a `coordMap` |
| 10 | [`10_retarget_ts`](10_retarget_ts.ipynb) | **retarget a real DFT TS by SMILES** — graft the reacting core (graphrc bond-change indices) onto a freshly-assembled `catalyst.substrate`: rebuild `cpa` (0.000 Å); **mutate the jacob_ts4 FLP catalyst / substrate / both** via conserved-anchor SMARTS (B–B, B–H held); the **diastereomer switch** (`chirality='retain'` vs `'both'`); the **charged thia-ma sulfa-Michael** TS — swap the acyl isothiouronium (HyperBTM→BTM→p-tolyl) holding the acyl, C–S held 2.266 Å; `tmc_round` as the metal-SMILES enabler; every step **overlaid on the original TS**, geometrically aligned on the reactive core |

**Rendering is direct and honestly typed.** The notebooks call `xyzrender.render(...)` **inline** (only a
one-line `xyz()` dump helper is defined). xyzrender detects ordinary bonds — *including metal–donor
coordination* — from the geometry on its own; we name only the special ones: **`ts_bonds=[(i, j)]`** (dashed)
for genuine forming/breaking TS bonds, **`nci_bonds=[(i, j)]`** (dotted) for non-covalent contacts (H-bonds,
chalcogen, cation-π), and **`labels=["i j d"]`** / `"i j k a"` for distances/angles (xyzrender is 1-indexed, so
atoms are passed `+1`). `landscape()` draws the whole ensemble: kept modes coloured (1-indexed), conformers a
prune merged away as faint points. The geometry gate is called directly as `rx.geometry.check(...)` and 3D
rendering uses `xyzrender.render(...)` inline — everything the notebooks do is the real API, no hidden helpers.
`structures/` holds the static TS geometries (generated once from graphrc trajectories): `sn2`, `spiro-ts1`,
`bimp_small` (50-atom light TS), `bimp` (172-atom bifunctional flagship — multiple NCIs + frozen core), `cpa`
(chiral-phosphoric-acid TS), `mn-h2` / `mn-hy` / `ru-co` (bimetallic/organometallic TS). The reacting-atom
indices and forming/breaking bonds are written inline in the notebooks (they'd come from a graphrc trajectory
in practice).

Outputs are committed here as executed proof; clear them with `jupyter nbconvert --clear-output --inplace
0*.ipynb` before a lightweight commit.
