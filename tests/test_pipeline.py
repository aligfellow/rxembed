"""Integration tests for the search + dedup tail and the explore-release provenance (openconf-gated).

Every DESIGN worked example ends ``.mc().prune()``, and "bias the seed, let energy decide" lives in
``mc(explore=True)`` releasing exactly the soft grips. Both need openconf, so these are skipped when it is
absent (the rest of the suite still runs). No xtb.
"""

from tests.conftest import needs_openconf


@needs_openconf
def test_mc_then_prune_chain():
    import rxembed as rx

    ens = rx.embed("CCCCCCO", n=6).mc(preset="rapid").prune()  # the canonical search + dedup tail
    assert ens.n >= 1
    assert ens.discarded is not None  # prune ran and recorded what it merged (may be empty)


@needs_openconf
def test_mc_explore_releases_only_soft_contacts():
    import rxembed as rx

    es = rx.embed("CC(=O)O.n1ccccc1", contacts="auto", n=6)  # a seeded NCI grip (soft, releasable)
    ens = es[0] if isinstance(es, rx.EnsembleSet) else es
    assert any(ens.cons.contacts), "the seeded grip should be recorded as releasable before explore"
    before = ens.n
    ens.mc(preset="rapid", explore=True)  # second pass with the grip released, structure kept
    assert ens.n > before  # explore pooled additional (grip-released) conformers
    assert ens.cons.contacts == (frozenset(), frozenset())  # provenance cleared: nothing left to release
