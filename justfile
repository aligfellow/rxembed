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

# Install deps + pre-commit. openconf (incl. its unreleased transition-metal support) is pulled from git by
# `uv sync` — see [tool.uv.sources] in pyproject.toml. No manual clone needed.
setup:
    uv sync
    uv run pre-commit install

# Co-develop openconf: clone it as a sibling and override the git pin with a local editable install. Re-run
# after `uv sync` (which reverts to the git pin). Point OPENCONF_DIR elsewhere if your clone isn't ../openconf.
OPENCONF_DIR := "../openconf"
setup-openconf-dev:
    test -d {{OPENCONF_DIR}} || git clone https://github.com/rowansci/openconf {{OPENCONF_DIR}}
    uv pip install -e {{OPENCONF_DIR}}
