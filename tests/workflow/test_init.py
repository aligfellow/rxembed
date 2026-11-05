"""What a user hits when a workflow extra is missing: the gate `AGENTS.md` names for this tier."""

import sys

import numpy as np
import pytest

from rxembed.workflow.select import cluster_on


def test_missing_extra_names_its_module_and_what_pip_installs(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)  # None makes the import raise ImportError
    with pytest.raises(ImportError, match=r"sklearn\.cluster is not installed; pip install 'rxembed\[workflow\]'") as e:
        cluster_on(np.zeros((10, 3)))
    assert isinstance(e.value.__cause__, ImportError), "the upstream reason must stay reachable for a real bug"
