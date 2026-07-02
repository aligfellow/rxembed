"""Organocatalysis — transfer a known TS across scaffolds by backbone swap.

The headline capability: hold a known reacting-core TS geometry and template it onto a *series* of related
catalyst backbones (e.g. one isothiourea motif to another). Scaffold (`skip`) until Phase A; the
parametrize lists are the executable spec of which scaffolds must transfer cleanly.
"""

import pytest

from rxembed import geometry as geom

skip_phase_a = pytest.mark.skip(reason="Phase A: rx.embed / template not yet ported")


# Isothiourea acyl-ammonium / enolate TS: conserved reacting core = N-C(acyl) + S...C=O chalcogen contact.
# Transfer that core onto each backbone. (SMILES are placeholders for the real catalyst scaffolds.)
ISOTHIOUREA_BACKBONES = [
    "btm",  # benzotetramisole
    "hyperbtm",  # HyperBTM
    "tetramisole",  # tetramisole
    "modified",  # a modified backbone (the "change the backbone" case)
]


@skip_phase_a
@pytest.mark.parametrize("backbone", ISOTHIOUREA_BACKBONES)
def test_isothiourea_ts_transfer(backbone):
    import rxembed as rx

    # template = the known TS core; match = the conserved isothiourea reacting motif (SMARTS)
    ens = rx.embed(
        f"tests/fixtures/isothiourea_{backbone}.smi",
        template="tests/fixtures/isothiourea_ts_core.xyz",
        match="[#7]-[#6](=[#8])...[#16]",  # placeholder SMARTS for the conserved core
    )
    ref = rx.embed("tests/fixtures/isothiourea_ts_core.xyz")
    for cid in ens.ids:
        # reacting core transferred exactly + the S...C=O / N-C contact preserved + backbone clean
        geom.check(ens.mol, cid, frozen="core", reference=ref.mol).assert_ok()


@skip_phase_a
@pytest.mark.parametrize(
    ("family", "analogues"),
    [
        ("nhc", ["imes", "ipr", "triazolylidene"]),
        ("cinchona", ["quinine", "quinidine", "modified"]),
        ("binol_phosphoric_acid", ["trip", "stip", "modified"]),
    ],
)
def test_ts_transfer_other_families(family, analogues):
    """One known TS -> an analogue series, reacting core conserved, periphery clean."""
    import rxembed as rx

    for analogue in analogues:
        ens = rx.embed(
            f"tests/fixtures/{family}_{analogue}.smi",
            template=f"tests/fixtures/{family}_ts_core.xyz",
            match="core",
        )
        for cid in ens.ids:
            geom.check(ens.mol, cid).assert_ok()
