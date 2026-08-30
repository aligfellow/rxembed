# Run all checks
check: lint type test

lint:
    uv run ruff format src tests
    uv run ruff check --fix src tests

type:
    uv run ty check src tests

test:
    uv run python -m pytest -v

build:
    uv build

setup:
    uv sync
    uv run pre-commit install
