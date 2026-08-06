"""`pipeline/viz.py`: the 2D projection of the shared latent, and the landscape drawn from it."""

from importlib.util import find_spec

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(find_spec("matplotlib") is None, reason="needs rxembed[workflow]")


@pytest.fixture(autouse=True)
def _headless():
    import matplotlib as mpl

    mpl.use("Agg")


def test_projection_is_2d_and_rejects_unknown_method():
    from rxembed.pipeline.viz import project

    feats = np.random.default_rng(0).normal(size=(12, 7))
    assert project(feats, "pca").shape == (12, 2)
    with pytest.raises(ValueError, match="umap"):
        project(feats, "umap")


def test_landscape_projects_kept_and_pruned():
    import rxembed as rx
    from rxembed.pipeline.viz import landscape

    ens = rx.embed("OC(=O)CCCCc1ccccc1", n=10, seed=1).prune(max_rmsd=2.5)
    assert ens.discarded, "the fixture needs something pruned away for the dropped series to exist"
    ax = landscape(ens, color="energy")
    drawn = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert drawn == len(ens.ids) + len(set(ens.discarded))
