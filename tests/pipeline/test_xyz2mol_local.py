"""Test rxembed's changes to the vendored xyz2mol core."""

from __future__ import annotations

import numpy as np
import pytest

from rxembed.pipeline.xyz2mol_local import AC2BO


def test_element_without_a_valence_model_is_named():
    adjacency = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match=r"no valence model for Ga \(atom 0\)"):
        AC2BO(adjacency, [31, 6], 0)
