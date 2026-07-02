# Run all checks
check: lint type test

lint:
    uv run ruff format .
    uv run ruff check .

type:
    uv run --extra mc --extra nci --extra viz --extra ase --extra racerts ty check

test:
    uv run --extra mc --extra nci --extra viz --extra ase --extra racerts python -m pytest --cov --cov-report=xml -v

fix:
    uv run ruff format .
    uv run ruff check --fix .

build:
    uv build

setup:
    uv sync --extra mc --extra nci --extra viz --extra ase --extra racerts --dev
    uv run pre-commit install
