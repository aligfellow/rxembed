"""Organic — classic reactivity, transition states, and binding modes.

Scaffold (`skip`) until Phase A. The parametrize lists enumerate the reaction TS cores and NCI modes the
embed core must handle; each asserts forming-bond windows + frozen core + a clean periphery (geometry gate).
"""

import pytest

from rxembed import geometry as geom

skip_phase_a = pytest.mark.skip(reason="Phase A: rx.embed not yet ported")


@skip_phase_a
@pytest.mark.parametrize("smiles", ["CCO", "OC(=O)CCCCc1ccccc1", "C1CC1C(=O)O", "c1ccc2ccccc2c1"])
def test_baseline_flexible_rigid_strained(smiles):
    import rxembed as rx

    ens = rx.embed(smiles).prune()
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


# classic reaction TS cores: (name, forming/breaking atom pairs -> window). Embedded from .xyz freeze
# AND constrained-from-SMILES; each row is one reaction archetype.
REACTION_TS = [
    ("sn2", {"forming": (0, 5), "breaking": (0, 1)}),  # linear 3-centre
    ("e2", {"c_h": (0, 6), "c_lg": (1, 7)}),  # anti-periplanar
    ("diels_alder", {"cc_a": (0, 5), "cc_b": (3, 4)}),  # two forming C-C
    ("aldol", {"cc": (0, 4), "proton": (2, 5)}),  # C-C + proton transfer
    ("sigmatropic_33", {"break": (0, 1), "form": (5, 6)}),  # [3,3]
    ("proton_transfer", {"xh": (0, 3), "hy": (3, 4)}),
]


@skip_phase_a
@pytest.mark.parametrize(("name", "windows"), REACTION_TS)
def test_reaction_ts_core(name, windows):
    import rxembed as rx

    ens = rx.embed(f"tests/fixtures/ts_{name}.xyz", freeze="core")
    ref = rx.embed(f"tests/fixtures/ts_{name}.xyz")
    for cid in ens.ids:
        geom.check(ens.mol, cid, frozen="core", reference=ref.mol).assert_ok()


@skip_phase_a
@pytest.mark.parametrize(
    "complex_smiles",
    [
        "OC(=O)c1ccccc1.n1ccccc1",  # H-bond
        "IC(F)(F)F.n1ccccc1",  # halogen bond
        "O=S(c1ccccc1)c1ccccc1.O",  # chalcogen bond
        "CC(=O)[O-].C[NH3+]",  # salt bridge -> directional anion H-bond
    ],
)
def test_organic_binding_modes(complex_smiles):
    import rxembed as rx

    es = rx.embed(complex_smiles, contacts="auto")  # -> one candidate per discovered grip
    assert len(es) >= 1
    for ens in es:
        for cid in ens.ids:
            geom.check(ens.mol, cid).assert_ok()


@skip_phase_a
def test_ts_with_nci_grip():
    """A reacting core that also holds a non-covalent contact (template + contacts together)."""
    import rxembed as rx

    ens = rx.embed("CCCCO.c1ccccc1", template="tests/fixtures/core.xyz", contacts=[(0, 13, 3.2, 3.8)])
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()
