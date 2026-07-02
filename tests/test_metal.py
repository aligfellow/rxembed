"""Metal — the full coordination space.

Geometries, isomer permutations, ligand swapping, metal TS, and altered/ambidentate binding modes.
Scaffold (`skip`) until Phase A; the parametrize lists enumerate what the metal path must handle.
"""

import pytest

from rxembed import geometry as geom

skip_phase_a = pytest.mark.skip(reason="Phase A: rx.metal / rx.embed not yet ported")


# --- all geometries ----------------------------------------------------------

GEOMETRIES = [
    ("linear", 2),
    ("trigonal_planar", 3),
    ("tetrahedral", 4),
    ("square_planar", 4),
    ("trigonal_bipyramidal", 5),
    ("square_pyramidal", 5),
    ("octahedral", 6),
]


@skip_phase_a
@pytest.mark.parametrize(("geometry", "coordination"), GEOMETRIES)
def test_geometry_embeds_clean(geometry, coordination):
    import rxembed as rx

    smiles = "..."  # a representative complex for this geometry
    for iso in rx.metal(smiles, geometry):
        ens = rx.embed(iso)
        for cid in ens.ids:
            geom.check(ens.mol, cid).assert_ok()


# --- isomer permutations per geometry: enumeration count + no duplicate perms ---

ISOMER_CASES = [
    ("square_planar_MA2B2", "square_planar", 2),  # cis / trans
    ("octahedral_MA3B3", "octahedral", 2),  # mer / fac
    ("tris_chelate", "octahedral", 2),  # Lambda / Delta chirality
]


@skip_phase_a
@pytest.mark.parametrize(("case", "geometry", "n_expected"), ISOMER_CASES)
def test_isomer_permutations(case, geometry, n_expected):
    import rxembed as rx

    isomers = list(rx.metal(f"tests/fixtures/{case}.smi", geometry))
    assert len(isomers) == n_expected  # correct count (symmetry-equivalent perms deduped)
    for iso in isomers:
        ens = rx.embed(iso)
        for cid in ens.ids:
            geom.check(ens.mol, cid).assert_ok()


# --- ligand swapping on a known core -----------------------------------------


@skip_phase_a
@pytest.mark.parametrize("denticity", ["monodentate", "bidentate", "tridentate"])
def test_ligand_swap_on_template(denticity):
    import rxembed as rx

    ens = rx.embed(
        f"tests/fixtures/new_{denticity}_ligand.smi",
        template="tests/fixtures/metal_core.xyz",
        match="core",
    )
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()  # retained core preserved, ligand at vacated site(s)


# --- metal TS: freeze the reacting core, enumerate free sites -----------------


@skip_phase_a
def test_metal_ts_frozen_core_free_sites():
    import rxembed as rx

    frozen = "reacting_core"  # M-H + substrate
    isomers = list(rx.metal("tests/fixtures/metal_ts.xyz", "octahedral", freeze=frozen))
    ref = rx.embed("tests/fixtures/metal_ts.xyz")
    assert len(isomers) >= 2  # mer/fac of the spectator ligands, reacting core held
    for iso in isomers:
        ens = rx.embed(iso)
        for cid in ens.ids:
            geom.check(ens.mol, cid, frozen=frozen, reference=ref.mol).assert_ok()


@skip_phase_a
def test_bimetallic_ts():
    import rxembed as rx

    ens = rx.embed("tests/fixtures/bimetallic_ts.xyz", freeze="core")
    ref = rx.embed("tests/fixtures/bimetallic_ts.xyz")
    for cid in ens.ids:
        geom.check(ens.mol, cid, frozen="core", reference=ref.mol).assert_ok()


# --- altering binding modes: one ligand, two possible sites ------------------

AMBIDENTATE = [
    ("thiocyanate", ["S_bound", "N_bound"]),  # SCN-
    ("nitrite", ["nitro", "nitrito"]),  # NO2-
    ("enolate", ["O_bound", "C_bound"]),  # O- vs C-
    ("hemilabile", ["kappa1", "kappa2"]),  # denticity change
]


@skip_phase_a
@pytest.mark.parametrize(("ligand", "modes"), AMBIDENTATE)
def test_ambidentate_binding_modes(ligand, modes):
    """One ligand with two binding sites -> an EnsembleSet with both distinct modes, each sane."""
    import rxembed as rx

    es = rx.embed(f"tests/fixtures/metal_{ligand}.smi", coordinate="auto")
    assert len(es) == len(modes)  # both modes produced
    for ens in es:
        for cid in ens.ids:
            geom.check(ens.mol, cid).assert_ok()
