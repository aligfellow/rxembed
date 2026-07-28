"""A user spec must reach the embedder whole — no field silently dropped on the way in.

`_embed_isomer` merged a resolved spec into a metal isomer's constraints by hand-listing four fields, so a
`constrain={(ring_a, ring_b): sep}` pi-stack was parsed, index-validated, logged as accepted, and then
discarded before either the bounds writer or the force field saw it. Silent user-data loss, and a direct
violation of the project's "fail loud, never silently drop" rule. The merge is now field-driven.
"""

import rxembed as rx

_BIPY_PD = "Cl[Pd]1(Cl)<-n2ccccc2-c2ccccn->12"


def _two_six_rings(mol):
    return [tuple(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 6][:2]


def test_a_pi_stack_constrain_survives_the_metal_path():
    iso = next(iter(rx.metal(_BIPY_PD, "square_planar")))
    a, b = _two_six_rings(iso.mol)
    ens = rx.embed(iso, constrain={(a, b): 3.6}, n=2, seed=1)
    assert ens.cons.planes, "the pi-stack was accepted and then dropped before the embed"
    assert any(set(pa) == set(a) and set(pb) == set(b) for pa, pb, _sep in ens.cons.planes)


def test_the_organic_path_already_carried_it():
    """The same spec on a non-metal source — the reference behaviour the metal path now matches."""
    ens = rx.embed(
        "c1ccccc1CCCCc1ccccc1",
        constrain={((0, 1, 2, 3, 4, 5), (10, 11, 12, 13, 14, 15)): 3.6},
        n=2,
        seed=1,
    )
    assert ens.cons.planes


def test_a_spec_landing_on_a_sphere_hold_still_overrides_it():
    """Field-driven merge must stay LAST-WINS on distances: a user number beats the coordination window.

    And it must not be demoted to a releasable grip — `mc(explore=)` frees contacts, and a sphere hold that
    became releasable could let the coordination sphere dissociate.
    """
    iso = next(iter(rx.metal(_BIPY_PD, "square_planar")))
    metal = iso.metal
    donor = next(d for d in iso.donors)
    ens = rx.embed(iso, fix={(metal, donor): 2.42}, n=2, seed=1)
    lo, hi = ens.cons.distances[(min(metal, donor), max(metal, donor))]
    assert lo <= 2.42 <= hi, f"the user fix was overridden by the sphere hold: {(lo, hi)}"
    assert (min(metal, donor), max(metal, donor)) not in ens.cons.contacts[0], "a sphere hold became releasable"
