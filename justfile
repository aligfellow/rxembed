# Run all checks
check: lint type test

lint:
    uv run ruff format .
    uv run ruff check --fix .

type:
    uv run ty check

test:
    uv run python -m pytest --cov --cov-report=xml -v

build:
    uv build

setup:
    uv sync 
    uv run pre-commit install
