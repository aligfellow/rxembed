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


def test_porphyrin_sized_valence_search_is_refused_before_it_enumerates(monkeypatch):
    """A bare 20-carbon ring offers 3**20 valence assignments; past the bound the search must not start."""
    import numpy as np

    from rxembed.pipeline import xyz2mol_local

    n = 20
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        adjacency[i, (i + 1) % n] = adjacency[(i + 1) % n, i] = 1

    def enumerated(*_args):
        raise AssertionError("the valence product was enumerated")

    monkeypatch.setattr(xyz2mol_local.itertools, "product", enumerated)
    with pytest.raises(ValueError, match="3486784401 valence combinations"):
        AC2BO(adjacency, [6] * n, 0, max_combinations=256)
