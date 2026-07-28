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


def test_coordinate_relieves_the_phantom_floor_it_creates(monkeypatch):
    """Seating a substrate donor must relieve the surrogate floor on the atoms it pulls into the sphere.

    RDKit floors every M...X at the bond-less carbon surrogate's vdW contact (~3.4 Å). Seating a carbonyl O
    puts its own carbon at ~3.2 Å through the new M-O window, so the phantom contradicts the coordination and
    triangle smoothing repairs some pair — not necessarily that one. Zero crossover is the assertion: the
    constraints handed to the embed must be mutually realisable. Red whenever `coordinate()`'s output is
    cherry-picked field-by-field instead of `compose`d, since the relief rides on a field the cherry-pick drops.
    """
    import rxembed as rx
    from rxembed.rdkit_embed.embed import bounds as _b

    tols, real = [], _b._bounds

    def spy(mol, cons):
        bm, tol = real(mol, cons)
        tols.append(tol)
        return bm, tol

    monkeypatch.setattr(_b, "_bounds", spy)
    iso = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "square_planar").select(index=0)  # one vacant site + free acetone
    o = next(a.GetIdx() for a in iso.mol.GetAtoms() if a.GetSymbol() == "O")
    rx.embed(iso, coordinate=o, n=1)
    assert tols, "the embed never built a bounds matrix"
    assert max(tols) == 0.0, f"bound crossover repaired: {max(tols) * 100:.4f}%"


def test_metal_and_isomer_source_are_mutually_exclusive():
    import rxembed as rx

    iso = next(iter(rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")))
    with pytest.raises(ValueError, match="Isomer source OR metal"):
        rx.embed(iso, metal="square_planar")


_RETAIN_RELAX = "RELAXED into its constraint windows"  # the seam's clarity line for the retain-input path


def test_retain_input_geometry_is_logged_as_relaxed(tmp_path, caplog):
    """`rx.embed(metal_geometry)` retains the arrangement but RELAXES it — say so, don't imply it's pristine.

    The retain-input seam relaxes ids[0] into its coordination windows (the M-donor sphere is held <0.01 Å so
    the arrangement survives), which is what makes scoring fair — an unrelaxed input would score perfectly
    against itself. A user must be able to read that the input was relaxed, not returned as-is. The line fires
    only on the retain path, never on a plain constrained embed.
    """
    import rxembed as rx

    xyz = tmp_path / "pd.xyz"  # a small square-planar Pd geometry to feed the retain-input path (no constraints)
    rx.embed(rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0], n=1, seed=1).dump(str(xyz))

    with caplog.at_level("INFO", logger="rxembed"):
        rx.embed(str(xyz))
    assert any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text

    caplog.clear()
    with caplog.at_level("INFO", logger="rxembed"):  # a normal constrained embed has no retained input
        rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert not any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text
