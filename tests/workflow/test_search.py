"""`workflow/search.py`: openconf's search adopted onto rxembed's Mol, pose-frozen on the atoms constraints hold."""

import numpy as np
import pytest

from rxembed.geometry.constraints import Constraints
from rxembed.workflow import search
from tests.extras import needs
from tests.geometry_fixtures import embedded, placed

pytestmark = needs("openconf")


def test_search_adds_conformers_keeping_frozen_atoms_at_seed():
    for held in ([], [0, 1, 2]):
        butanol = embedded("CCCCO")
        seed = butanol.GetConformer().GetPositions()
        ids = search.search(butanol, Constraints(frozen=frozenset(held)), search.config("rapid", 1))
        assert ids == list(range(1, butanol.GetNumConformers())), held
        assert all(np.allclose(butanol.GetConformer(i).GetPositions()[held], seed[held]) for i in ids), held


def test_search_refuses_the_conformers_of_another_atom_set():
    """Pose-frozen, openconf adds the hydrogens a heavy-atom butanol leaves implicit."""
    butanol = placed("CCCCO", [(0, 0, 0), (1.5, 0, 0), (2, 1.4, 0), (3.5, 1.4, 0), (4, 2.8, 0)])
    with pytest.raises(RuntimeError, match="changed the atom set"):
        search.search(butanol, Constraints(frozen=frozenset({0, 1})), search.config("rapid", 1))
