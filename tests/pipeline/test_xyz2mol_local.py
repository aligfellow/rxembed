"""Test rxembed's changes to the vendored xyz2mol core."""

from __future__ import annotations

import pytest

from rxembed.pipeline.xyz2mol_local import AC2BO, get_proto_mol


def test_proto_mol_atoms_carry_no_query():
    mol = get_proto_mol([6, 1, 1, 1, 1])
    assert mol.GetNumAtoms() == 5
    assert not any(a.HasQuery() for a in mol.GetAtoms()), "a query atom is back in the proto mol"
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [6, 1, 1, 1, 1]


def test_overvalent_atom_raises():
    import numpy as np

    # one carbon bonded to six hydrogens: no valence the model can place
    n = 7
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        adjacency[0, i] = adjacency[i, 0] = 1
    with pytest.raises(ValueError, match="valence"):
        AC2BO(adjacency, [6] + [1] * (n - 1), 0)
