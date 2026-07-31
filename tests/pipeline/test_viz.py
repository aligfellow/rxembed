"""`pipeline/viz.py`: the 2D projection of the shared latent, and the landscape drawn from it."""

from importlib.util import find_spec

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    find_spec("matplotlib") is None or find_spec("seaborn") is None, reason="needs rxembed[viz]"
)


@pytest.fixture(autouse=True)
def _headless():
    import matplotlib as mpl

    mpl.use("Agg")


def test_every_known_projection_lands_in_two_dimensions_and_an_unknown_one_is_refused():
    """The landscape plots x/y off this, so a projection that came back 7-wide would draw silent nonsense."""
    from rxembed.pipeline.viz import project

    feats = np.random.default_rng(0).normal(size=(12, 7))
    for method in ("pca", "tsne"):
        assert project(feats, method).shape == (12, 2)
    with pytest.raises(ValueError, match="umap"):
        project(feats, "umap")


def test_the_landscape_draws_the_kept_and_the_pruned_in_one_projection():
    """Kept and dropped are projected together, so a pruned region is visible where it actually was."""
    import rxembed.pipeline as rx
    from rxembed.pipeline.viz import landscape

    ens = rx.embed("OC(=O)CCCCc1ccccc1", n=10, seed=1).prune(max_rmsd=2.5)
    assert ens.discarded, "the fixture needs something pruned away for the dropped series to exist"
    ax = landscape(ens, color="energy")
    drawn = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert drawn == len(ens.ids) + len(set(ens.discarded))
