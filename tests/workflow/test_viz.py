"""`workflow/viz.py`: the 2D landscape of an ensemble's kept and dropped conformers on its clustering latent."""

import pytest
from rdkit.Chem import rdDistGeom

import rxembed as rx
from tests.extras import needs
from tests.geometry_fixtures import embedded

pytestmark = needs("matplotlib", "sklearn")


@pytest.fixture(scope="module")
def propanol_water():
    """Return sixteen propanol-water conformers given energies in close pairs, pruned by energy to eight."""
    mol = embedded("CCCO.O")
    rdDistGeom.EmbedMultipleConfs(mol, 16, randomSeed=1)
    return rx.wrap(mol, energies=[k // 2 + 0.01 * (k % 2) for k in range(16)]).prune(by="energy")


def test_landscape_draws_each_kept_and_dropped_conformer(propanol_water):
    for method in ("pca", "tsne"):
        drawn = sum(len(c.get_offsets()) for c in propanol_water.landscape(method=method, color="energy").collections)
        assert drawn == len(propanol_water.ids) + len(set(propanol_water.discarded)), method


def test_landscape_colours_kept_conformers_by_cluster(propanol_water):
    labels = propanol_water.cluster().tolist()
    drawn = {c.get_label(): len(c.get_offsets()) for c in propanol_water.landscape().collections}
    assert drawn == {"dropped (8)": 8} | {"noise" if k == -1 else f"mode {k + 1}": labels.count(k) for k in set(labels)}


def test_landscape_refuses_a_colour_projection_or_empty_ensemble(propanol_water):
    for ens, options, refusal in (
        (propanol_water, {"color": "mode"}, "'cluster' or 'energy'"),
        (propanol_water, {"method": "umap"}, "'pca' or 'tsne'"),
        (propanol_water[:0], {}, "draw it before the stage that emptied it"),
    ):
        with pytest.raises(ValueError, match=refusal):
            ens.landscape(**options)
