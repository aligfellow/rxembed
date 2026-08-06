"""What a user hits when an optional dependency is missing: the gate `AGENTS.md` names for this tier."""

from __future__ import annotations

import builtins

import pytest

from rxembed.pipeline.select import cluster_on


def test_message_names_the_operation_and_what_pip_installs(monkeypatch):
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
    assert "pip install 'rxembed[workflow]'" in msg, msg
    assert isinstance(exc.value.__cause__, ImportError), "the upstream reason must stay reachable for a real bug"
