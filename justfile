# Run all checks
check: lint type test

lint:
    uv run ruff format src tests
    uv run ruff check --fix src tests

type:
    uv run ty check src tests

test:
    uv run python -m pytest -v

# Regression tests for the benchmark runner; benchmark/ is maintainer-local and gitignored, so a
# clean checkout may not have it -- report that cleanly instead of failing.
bench-test:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f benchmark/test_run.py ]; then
        echo "benchmark/ absent (maintainer-local); skipping bench-test"
        exit 0
    fi
    uv run python -m pytest benchmark/test_run.py -v

build:
    uv build

setup:
    uv sync
    uv run pre-commit install

# Benchmark fixtures, first N pinned tmQMg IDs and issues; src=<tree>/src measures another tree
bench n='100' dataset='../tmQMg' src='src':
    #!/usr/bin/env bash
    set -euo pipefail
    export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_NO_SYNC=1
    export PYTHONPATH={{quote(absolute_path(src))}}
    mkdir -p benchmark/results
    out=$(mktemp -d "benchmark/results/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
    for cohort in fixtures sample issues; do
        uv run python benchmark/run.py "$cohort" "$out" --size {{quote(n)}} --dataset {{quote(dataset)}}
    done
