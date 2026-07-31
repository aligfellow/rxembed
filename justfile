# Run all checks
check: lint type test

# No `--all-extras` anywhere below, deliberately: the `dev` dependency group in pyproject.toml pulls
# `rxembed[all]`, so a bare `uv run` / `uv sync` already gives the full environment. Passing the flag would
# paper over exactly the footgun we fixed — if the default ever stops installing the extras, these go red.
lint:
    uv run ruff format .
    uv run ruff check --fix .

type:
    uv run ty check

test:
    uv run python -m pytest --cov --cov-report=xml -v

build:
    uv build

# Install deps + pre-commit. `uv sync` alone is the full dev environment (see the `dev` group in
# pyproject.toml); openconf, including its unreleased transition-metal support, comes from the git pin in
# [tool.uv.sources]. No manual clone needed.
setup:
    uv sync
    uv run pre-commit install
