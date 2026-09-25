"""`pipeline/search.py`: the openconf Monte-Carlo backend and the `mc()` verb over it."""

from importlib.util import find_spec

import numpy as np
import pytest

import rxembed as rx
from rxembed.pipeline import search


def test_missing_mc_backend_preserves_ensemble(monkeypatch):
    monkeypatch.setattr(search, "available", lambda: False)
    ens = rx.embed("CCCCCCO", n=3, seed=1).minimize()
    before = (list(ens.ids), dict(ens.energies), ens.energy_kind, ens._stage)
    positions = {i: ens.mol.GetConformer(i).GetPositions() for i in ens.ids}

    with pytest.raises(ImportError, match=r"mc needs openconf; pip install 'rxembed\[search\]'"):
        ens.mc(preset="rapid")

    assert (ens.ids, ens.energies, ens.energy_kind, ens._stage) == before
    for i, expected in positions.items():
        np.testing.assert_array_equal(ens.mol.GetConformer(i).GetPositions(), expected)


def test_failed_search_clears_stale_energies(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(search, "available", lambda: True)
    monkeypatch.setattr(search, "search", fail)
    ens = rx.embed("CCCC", n=2, seed=1).minimize()
    assert ens.energies
    assert ens.energy_kind == "ff"

    ens.mc()

    assert not ens.energies
    assert not ens.energy_kind
    assert ens._stage == "seeded"


# --- config resolution: a preset, then single-knob overrides ----------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_unknown_passthrough_field_is_refused_by_name():
    with pytest.raises(TypeError, match="not_a_field"):
        search._config("rapid", None, None, None, None, constrained=False, not_a_field=1)


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_constrained_search_rejects_low_mode(caplog):
    with caplog.at_level("WARNING", logger="rxembed"):
        cfg = search._config("rapid", None, None, True, None, constrained=True)
    assert not cfg.use_low_mode_following
    assert any("low-mode" in r.getMessage() for r in caplog.records)


# --- the mc() verb ----------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
@pytest.mark.parametrize(
    ("spec", "seeds_survive"),
    [
        pytest.param({}, False, id="unconstrained"),
        pytest.param({"fix": {(0, 6): 4.0}}, True, id="constrained"),
    ],
)
def test_mc_preserves_only_constrained_seeds(spec, seeds_survive):
    ens = rx.embed("CCCCCCO", n=4, seed=1, **spec)
    seeds = set(ens.ids)
    ens.mc(preset="ensemble", seed=1, max_out=8)
    assert ens.n >= 1
    assert (seeds <= set(ens.ids)) is seeds_survive
    if seeds_survive:
        assert ens.n > len(seeds), "a constrained search added nothing"


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_prune_deduplicates_search_results():
    ens = rx.embed("CCCCCCO", n=6, seed=1).mc(preset="rapid", seed=1, max_out=20)
    searched = ens.n
    ens.prune(max_rmsd=1.5)
    assert ens.n < searched
    assert set(ens.duplicates()) <= set(ens.ids), "a dropped conformer was grouped under another dropped one"


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_explore_pools_released_grip_and_clears_provenance():
    es = rx.embed("CC(=O)O.n1ccccc1", contacts="auto", n=6)
    ens = es[0] if isinstance(es, rx.EnsembleSet) else es
    assert any(ens.cons.contacts), "the seeded grip must be recorded as releasable before explore"
    before = ens.n
    ens.mc(preset="rapid", explore=True)
    assert ens.n > before
    assert ens.cons.contacts == (frozenset(), frozenset()), "nothing is left to release after explore"
