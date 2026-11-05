"""Mark tests that need an optional extra, so a base install skips them cleanly."""

from importlib.util import find_spec

import pytest


def needs(*modules):
    """Return a skip marker for a test that imports every one of `modules`."""
    missing = [module for module in modules if find_spec(module) is None]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)} (an rxembed extra)")
