# Metal round-trip benchmark

For each structure and seed: read the XYZ, enumerate its metal isomers (`rx.metal`), embed every
one (`rx.embed`), write each candidate's geometry back out, and independently re-perceive it. A
structure passes only when its reference isomer (the one matching the input) both embeds and
survives that check. One structure runs per forked child with a hard deadline, so a stuck or
crashing structure cannot stall or kill the run.

## Commands

```sh
just bench fixtures --seed 42 7 1234 2026 99      # 100 local fixtures, five seeds
just bench sample --size 100                       # first 100 in-scope tmQMg IDs
just bench issues                                   # the reported cases in issues.txt
uv run python benchmark/run.py compare benchmark/baseline.csv benchmark/results/fixtures-<stamp>.csv
```

`--only ID [ID ...]` runs just those IDs. `--timeout` sets the per-structure-and-seed deadline
(default 600 s); a timeout names the stage it was in, e.g. "exceeded 600 s during embed 3/8".
`--out` sets the results CSV path (default `benchmark/results/<cohort>-<UTCstamp>.csv`), opened
with mode `x` so a run never overwrites another. `--keep-xyz` writes every candidate geometry to
`<out stem>-xyz/`; by default they live in a scratch directory that is removed when the run ends.

The `sample` and `issues` cohorts need `RXEMBED_TMQMG_DIR` set to a tmQMg clone's `data/`
directory, with `tmQMg_xyz.zip` unzipped there.

## Output

Each results CSV starts with one `#`-prefixed JSON line (rxembed and RDKit versions, the exact
source hash, the run's argv and start time), then one row per structure and seed:

| column | meaning |
|---|---|
| `status` | `pass`, `fail`, or `timeout` |
| `stage` | for a failure: `read`, `enumerate`, `embed`, `worker`, or a `validate:*` step; for a timeout, the stage reached |
| `detail` | the failure text, or a leniency note on a pass (e.g. a Lewis re-read) |
| `core_rmsd` | the reference isomer's site RMSD to the input, computed once it embeds |
| `isomers` / `embedded` / `valid` | counts over the structure's enumerated metal isomers |
| `valid_cx` | one 8-character CX hash per validated isomer, space-joined |

A candidate fails at the first check that catches it: `validate:read`, `validate:fresh`,
`validate:connectivity`, `validate:constitution`, `validate:cx`, or `validate:geometry`. The
structure's own verdict is `enumerate` when the reference CX is absent or matched twice,
`validate:identity` when a different candidate reads back as the reference, `validate:core-rmsd`
when the reference embeds correctly but its own geometry sits over 0.75 A from the input, and
otherwise follows the reference candidate's own validity.

## compare

`run.py compare BASE.csv NEW.csv` checks a new run against a baseline by majority vote across
their shared seeds (needing more than half to count as passing or valid). A reference or isomer
that held a majority in BASE and drops below one in NEW is reported as lost, and the command exits
1. The reverse, and any other change of count, is noise: reported as a total, not listed. A
missing row counts as a non-pass. A structure with any timeout in NEW is skipped for the isomer
comparison, since isomers it never reached are not evidence either way.

## Fixtures and licences

`fixtures.csv` + `fixtures/*.xyz` are the local, tracked cohort:

| origin | fixtures | source | licence |
|---|---|---|---|
| tmQMg | 65 | github.com/hkneiding/tmQMg | MIT, (c) 2024 Hannes Kneiding |
| OIN | 34 | github.com/tjmustard/OIN-SMILES `tests/fixtures` | MIT, (c) 2025 Thomas J. L. Mustard |
| rxembed | 1 (MnH) | built in this repo | MIT, this repo |

ASISAX, BENVOG and KAXVOX carry tmPHOTO/tmCAT dataset headers from OIN's own copy; PdCl2-RR-DPDME
carries a MetalloGen-3D header. See [../LICENSES.md](../LICENSES.md) for the licence text.

`tmqmg.txt` pins 4,890 ordered tmQMg IDs already pruned to this benchmark's scope (no boron
cages, no all-metals-unbound structures); `sample` takes the first `--size`. `issues.txt` lists
the reported cases `issues` runs, as `#`-commented groups.

## Tests

`just bench-test` runs `benchmark/test_run.py`. Refresh `benchmark/baseline.csv` (the fixtures
cohort at seeds 42, 7, 1234, 2026 and 99) in the commit that accepts a measured change.
