"""`pipeline/viz.py`: the 2D projection of the shared latent, and the landscape drawn from it."""

from importlib.util import find_spec

import pytest

import rxembed as rx

pytestmark = pytest.mark.skipif(find_spec("matplotlib") is None, reason="needs rxembed[workflow]")


def test_landscape_projects_kept_and_pruned():
    ens = rx.embed("OC(=O)CCCCc1ccccc1", n=10, seed=1).prune(max_rmsd=2.5)
    assert ens.discarded, "the fixture needs something pruned away for the dropped series to exist"
    ax = ens.landscape(color="energy")
    drawn = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert drawn == len(ens.ids) + len(set(ens.discarded))
