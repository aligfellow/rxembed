# Metal round-trip benchmark

For each structure and seed: read the XYZ, enumerate its metal isomers (`rx.metal`), embed every one
(`rx.embed`), write each candidate's geometry back out, and independently re-read it. A candidate passes
when its heavy-atom bonds survive the round trip, its own `rx.cxsmiles` matches the isomer it was asked to
embed, and `rx.geom_check.check` reports no violation. A structure passes when its reference isomer (the
one matching the input) does. One structure and seed runs per forked child with a hard deadline, so a stuck
or crashing structure cannot stall or kill the run.

## Commands

```sh
just bench fixtures --seed 42 7 1234 2026 99      # the 100 shipped fixtures, five seeds
just bench tmqmg --size 100                        # a MaxMin-diverse sample of a local tmQMg clone
uv run python benchmark/run.py compare benchmark/baseline.csv benchmark/results/fixtures-<stamp>.csv
```

`--only ID [ID ...]` runs just those IDs. `--seed` takes one or more `rx.embed` seeds (default 42).
`--timeout` sets the per-structure-and-seed deadline (default 600 s); a timeout names the stage it was in.
`--out` sets the results CSV path (default `benchmark/results/<cohort>-<UTCstamp>.csv`), opened with mode
`x` so a run never overwrites another. `--keep-xyz` writes every candidate geometry to `<out stem>-xyz/`;
by default they live in a scratch directory removed when the run ends.

`tmqmg` needs `RXEMBED_TMQMG_DIR` set to a tmQMg clone's `data/` directory, with `tmQMg_xyz.zip` unzipped
there. `--size N` (default 100) picks N IDs by RDKit's MaxMin diversity picker over a metal/donor/ligand
fingerprint, seeded by one typical structure per metal; the pick is deterministic (seed 0) and a smaller
size is a prefix of a larger one. There is no shipped ID list.

## Output

Each results CSV starts with one `#`-prefixed JSON line (the rxembed and RDKit versions, the seeds and the
timeout), then one row per structure and seed:

| column | meaning |
|---|---|
| `status` | `pass`, `fail`, or `timeout` |
| `stage` | for a failure: `read`, `enumerate`, `embed`, `worker`, or a `validate:*` step; for a timeout, the stage reached |
| `detail` | the failure text |
| `core_rmsd` | the reference isomer's site RMSD to the input (metal and donor centroids); reported, never a gate |
| `isomers` / `embedded` / `valid` | counts over the structure's enumerated metal isomers |
| `valid_cx` | one 8-character CX hash per validated isomer, space-joined |

A candidate fails at the first check that catches it: `validate:read`, `validate:connectivity`, or
`validate:cx`, or `validate:geometry`. The structure's own verdict is `enumerate` when the reference CX is
absent or matched twice, and otherwise follows the reference candidate's own validity. Each run also prints
a short summary: passes per seed, failures grouped by stage, and the ten slowest structure-seed runs.

## compare

`run.py compare BASE.csv NEW.csv` checks a new run against a baseline by majority vote across their shared
seeds (needing more than half to count as passing or valid). A reference or isomer that held a majority in
BASE and drops below one in NEW is reported as lost, and the command exits 1. The reverse, and any other
change of count, is noise: reported as a total, not listed. A missing row counts as a non-pass. A structure
with any timeout in NEW is skipped for the isomer comparison, since isomers it never reached are not
evidence either way.

## Fixtures and licences

`fixtures.csv` + `fixtures/*.xyz` are the local, tracked cohort:

| origin | fixtures | source | licence |
|---|---|---|---|
| tmQMg | 65 | github.com/hkneiding/tmQMg | MIT, (c) 2024 Hannes Kneiding |
| OIN | 34 | github.com/tjmustard/OIN-SMILES `tests/fixtures` | MIT, (c) 2025 Thomas J. L. Mustard |
| rxembed | 1 (MnH) | built in this repo | MIT, this repo |

ASISAX, BENVOG and KAXVOX carry tmPHOTO/tmCAT dataset headers from OIN's own copy; PdCl2-RR-DPDME carries a
MetalloGen-3D header. See [../LICENSES.md](../LICENSES.md) for the licence text.
