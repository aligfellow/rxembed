"""`perception/xyz2mol_local.py`: rxembed's change to the vendored xyz2mol core."""

import numpy as np
import pytest

from rxembed.perception.xyz2mol_local import AC2BO


def test_ac2bo_refuses_an_element_without_a_valence_model():
    with pytest.raises(ValueError, match=r"no valence model for Ga \(atom 0\)"):
        AC2BO(np.array([[0, 1], [1, 0]]), [31, 6], 0)
