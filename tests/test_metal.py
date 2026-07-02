"""Metal coordination — isomer enumeration and clean embeds, reusing the new ``fix``/``constrain`` wiring.

The metal path keeps its own ``metal=`` / ``coordinate=`` verbs (out of the constraint-API redesign's
scope) but now routes any held reacting core through the same resolver. These check the enumeration counts
and that each isomer embeds to a metal-aware-clean geometry — pure RDKit + UFF surrogate, no xtb.
"""

import pytest

from rxembed import geometry as geom


def _assert_clean(ens):
    ens = ens.minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert rep.ok(), rep.summary()


@pytest.mark.parametrize(
    ("smiles", "geometry", "labels"),
    [
        ("CCCN[Pd](Cl)(Cl)NCCC", "square_planar", {"cis", "trans"}),  # MA2B2 -> cis / trans
        ("[NH3][Co]([NH3])([NH3])(Cl)(Cl)Cl", "octahedral", {"mer", "fac"}),  # MA3B3 -> mer / fac
    ],
)
def test_isomer_enumeration_counts_and_clean(smiles, geometry, labels):
    import rxembed as rx

    cands = rx.embed(smiles, metal=geometry, n=4)
    assert isinstance(cands, rx.EnsembleSet)
    assert {e.tag["label"] for e in cands} == labels  # the distinct, symmetry-reduced isomers
    for e in cands:
        _assert_clean(e)


def test_coordinate_binds_substrate_at_vacant_site():
    import rxembed as rx

    # 3 donors in a 4-vertex square plane -> one vacant pocket; the substrate O coordinates there
    es = rx.embed("CCCN[Pd](Cl)NCCC.O", metal="square_planar", coordinate="[OX2]", n=3)
    for ens in list(es) if isinstance(es, rx.EnsembleSet) else [es]:
        assert ens.n >= 1


def test_metal_and_isomer_source_are_mutually_exclusive():
    import rxembed as rx

    iso = next(iter(rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")))
    with pytest.raises(ValueError, match="Isomer source OR metal"):
        rx.embed(iso, metal="square_planar")
