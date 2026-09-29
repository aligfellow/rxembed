"""`pipeline/search.py`: the openconf Monte-Carlo backend and the `mc()` verb over it."""

from importlib.util import find_spec

import pytest

import rxembed as rx
from rxembed.pipeline import search

# --- config resolution: a preset, then single-knob overrides ----------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_constrained_search_rejects_low_mode(monkeypatch, caplog):
    ens = rx.embed("CCCCCCO", n=1, seed=1, fix={(0, 6): 4.0})
    real_config = search.config
    captured = []

    def spy(*args, **kw):
        cfg = real_config(*args, **kw)
        captured.append(cfg)
        return cfg

    monkeypatch.setattr(search, "config", spy)
    with caplog.at_level("WARNING", logger="rxembed"):
        ens.mc(preset="ensemble", low_mode=True)

    assert not captured[-1].use_low_mode_following
    assert any("low-mode" in r.getMessage() for r in caplog.records)


# --- the mc() verb ----------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
@pytest.mark.parametrize(
    ("spec", "seeds_survive"),
    [
        pytest.param({}, False, id="unconstrained"),
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
