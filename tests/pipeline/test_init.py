"""Every optional import in `pipeline/` is guarded, and every guard names the extra that installs it.

The static rule is the load-bearing half. A bare `import sklearn` in a function body imports fine on any dev
box and reaches the user as ``No module named 'sklearn'``, which names no remedy. No other gate watches for
it: the layer gate enforces what CORE may import, and this is the pipeline tier.

The guards are plain `try/except ImportError` at the import itself; visible where it happens, with nothing
between the reader and the failure. These tests are what keep the fifteen of them saying the same thing.
"""

from __future__ import annotations

import ast
import builtins
import sys
import tomllib
from pathlib import Path

import pytest

import rxembed.pipeline as _pipeline
from rxembed.pipeline.select import cluster_on

_PIPELINE = Path(_pipeline.__file__).parent
_BASE = {"numpy", "rdkit", "rxembed"}  # what `pip install rxembed` gives you, plus ourselves
_REPO = Path(__file__).resolve().parents[2]  # absolute: the suite must not depend on the cwd


def _modules():
    return sorted(_PIPELINE.glob("*.py"))


def _optional(name):
    """True for an import a base install (numpy + rdkit + stdlib) would not satisfy."""
    top = name.split(".")[0]
    return top not in _BASE and top not in sys.stdlib_module_names


def _imported(node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    return [node.module or ""] if node.level == 0 else []  # a relative import is a sibling, never a wheel


def _guarded(tree):
    """ids of import nodes sitting inside a `try:` whose handler catches ImportError."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            h.type is None or "ImportError" in ast.dump(h.type) for h in node.handlers
        ):
            out |= {id(n) for stmt in node.body for n in ast.walk(stmt)}
    return out


def _type_checking_only(tree):
    """ids of imports under `if TYPE_CHECKING`; annotations, never executed."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            out |= {id(n) for stmt in node.body for n in ast.walk(stmt)}
    return out


# --- the message a user sees ------------------------------------------------------------------------------


def test_the_message_names_the_operation_and_what_pip_installs(monkeypatch):
    """Not "this needs sklearn" but "cluster_on needs scikit-learn": the two things a user has to know."""
    import numpy as np

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "sklearn":
            raise ImportError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError) as exc:
        cluster_on(np.random.rand(10, 3))

    msg = str(exc.value)
    assert "cluster_on" in msg, "the message must name the operation, not say 'this'"
    assert "scikit-learn" in msg, "it must name the DISTRIBUTION pip installs, not the import name"
    assert "pip install 'rxembed[select]'" in msg, msg
    assert isinstance(exc.value.__cause__, ImportError), "the upstream reason must stay reachable for a real bug"


# --- the static rules ---------------------------------------------------------------------------------------


def test_every_optional_import_is_guarded():
    """A bare optional import reaches the user as "No module named 'sklearn'", which names no remedy."""
    bare = []
    for path in _modules():
        tree = ast.parse(path.read_text())
        exempt = _guarded(tree) | _type_checking_only(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom) and id(node) not in exempt:
                bare += [f"{path.name}:{node.lineno}: {n}" for n in _imported(node) if _optional(n)]
    assert not bare, "wrap these in try/except ImportError naming the extra:\n  " + "\n  ".join(bare)


def test_every_guard_names_a_real_extra_and_the_function_it_is_in():
    """A typo'd extra tells the user to install something that does not exist; worse than no message.

    The function name is checked too: these messages are written by hand at fifteen sites, so the one that
    drifts is the one copied from its neighbour and never re-read.
    """
    declared = set(tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"])
    checked = 0
    for path in _modules():
        src = path.read_text()
        tree = ast.parse(src)
        fns = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                for ln in range(fn.lineno, (fn.end_lineno or fn.lineno) + 1):
                    fns[ln] = fn.name
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            if getattr(node.exc.func, "id", None) != "ImportError" or not node.exc.args:
                continue
            arg = node.exc.args[0]
            if not isinstance(arg, ast.Constant) or "pip install" not in str(arg.value):
                continue
            msg, where = str(arg.value), f"{path.name}:{node.lineno}"
            extra = msg.split("rxembed[", 1)[1].split("]", 1)[0]
            assert extra in declared, f"{where}: names extra {extra!r}, which pyproject does not declare"
            owner = fns.get(node.lineno)
            assert owner, f"{where}: guard sits outside any function"
            assert msg.startswith(owner), f"{where}: message names {msg.split(maxsplit=1)[0]!r}, not {owner!r}"
            checked += 1
    assert checked > 5, f"the AST walk found only {checked} guards: it is measuring nothing"
