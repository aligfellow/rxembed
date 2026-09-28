# Run all checks
check: lint type test

lint:
    uv run ruff format src tests benchmark
    uv run ruff check --fix src tests benchmark

type:
    uv run ty check src tests

test:
    uv run python -m pytest -v

build:
    uv build

setup:
    uv sync
    uv run pre-commit install

# Run a benchmark cohort (fixtures, tmqmg or compare); see benchmark/README.md.
bench *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_NO_SYNC=1
    uv run python benchmark/run.py {{args}}
