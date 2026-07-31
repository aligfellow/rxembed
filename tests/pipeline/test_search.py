"""`pipeline/search.py`: the openconf Monte-Carlo backend and the `mc()` verb over it."""

from importlib.util import find_spec

import pytest

from rxembed.pipeline import search


def test_mc_without_the_backend_returns_the_seeds_it_was_given(monkeypatch, caplog):
    """The documented degradation: no openconf means `mc()` is a no-op with a WARNING, never an ImportError."""
    import rxembed.pipeline as rx

    monkeypatch.setattr(search, "available", lambda: False)
    ens = rx.embed("CCCCCCO", n=3, seed=1)
    before = list(ens.ids)
    with caplog.at_level("WARNING", logger="rxembed"):
        assert ens.mc(preset="rapid").ids == before
    assert any("openconf" in r.getMessage() for r in caplog.records)


# --- config resolution: a preset, then single-knob overrides ----------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_single_knobs_override_the_preset():
    cfg = search._config("rapid", seed=7, max_out=3, low_mode=None, config=None, constrained=False)
    assert (cfg.random_seed, cfg.max_out) == (7, 3)


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_an_unknown_passthrough_field_is_refused_by_name():
    """`mc(**kwargs)` is a passthrough to openconf's config: a typo must not be swallowed silently."""
    with pytest.raises(TypeError, match="not_a_field"):
        search._config("rapid", None, None, None, None, constrained=False, not_a_field=1)


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_low_mode_following_is_refused_in_constrained_search_and_says_so(caplog):
    """openconf disables low-mode following when atoms are pose-held; honouring the flag would be a lie."""
    with caplog.at_level("WARNING", logger="rxembed"):
        cfg = search._config("rapid", None, None, True, None, constrained=True)
    assert not cfg.use_low_mode_following
    assert any("low-mode" in r.getMessage() for r in caplog.records)


# --- the mc() verb ----------------------------------------------------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
@pytest.mark.parametrize(
    ("spec", "seeds_survive"),
    [
        pytest.param({}, False, id="unconstrained-supersedes-the-seeds"),
        pytest.param({"fix": {(0, 6): 4.0}}, True, id="constrained-is-added-around-them"),
    ],
)
def test_a_search_replaces_the_seeds_only_when_it_is_free_to_generate_them_itself(spec, seeds_survive):
    """Unconstrained, openconf is the whole generator (keeping seeds double-counts); pose-frozen, it explores
    only AROUND the biased seed, so discarding that seed would throw away the constraint it was given."""
    import rxembed.pipeline as rx

    ens = rx.embed("CCCCCCO", n=4, seed=1, **spec)
    seeds = set(ens.ids)
    ens.mc(preset="ensemble", seed=1, max_out=8)
    assert ens.n >= 1
    assert (seeds <= set(ens.ids)) is seeds_survive
    if seeds_survive:
        assert ens.n > len(seeds), "a constrained search added nothing"


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_prune_merges_the_searched_ensemble_rather_than_passing_it_through():
    """The canonical `mc().prune()` tail, and `mc` un-sets `_minimized`, so the prune re-relaxes first."""
    import rxembed.pipeline as rx

    ens = rx.embed("CCCCCCO", n=6, seed=1).mc(preset="rapid", seed=1, max_out=20)
    searched = ens.n
    ens.prune(max_rmsd=1.5)
    assert ens.n < searched
    assert set(ens.duplicates()) <= set(ens.ids), "a dropped conformer was grouped under another dropped one"


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_mc_explore_pools_a_second_grip_released_pass_and_clears_the_provenance():
    """'bias the seed, let energy decide': explore fires twice, grip held, grip released, and pools both."""
    import rxembed.pipeline as rx

    es = rx.embed("CC(=O)O.n1ccccc1", contacts="auto", n=6)
    ens = es[0] if isinstance(es, rx.EnsembleSet) else es
    assert any(ens.cons.contacts), "the seeded grip must be recorded as releasable before explore"
    before = ens.n
    ens.mc(preset="rapid", explore=True)
    assert ens.n > before
    assert ens.cons.contacts == (frozenset(), frozenset()), "nothing is left to release after an explore pass"
