"""`identity/species.py`: the `Species` record, the centres it seats or reads, and the selectors that match it."""

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.identity.reading import read_species
from rxembed.identity.species import Species, matches, misplaced_pins, observed_centre, retained_geometry, site_details
from rxembed.metal.centres import Centre, Site, canonical_metal_graph, chemical_graph
from rxembed.metal.polyhedra import POLYHEDRA
from rxembed.metal.sphere import LigandFacts
from tests.geometry_fixtures import placed
from tests.metal_fixtures import (
    FERROCENE_ONE_FACE,
    FERROCENE_TWO_FACES,
    PD2CL6,
    SPY_TBP_TIE,
    SQUASHED_SIX,
    ideal_sphere,
    seated,
)

DIAMMINE_PD = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"  # N0 Pd1 Cl2 Cl3 N4
ZINC = "[F-]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-]"  # F0 Zn1 Cl2 Br3 I4


def _on_vertices(species, radius=2.0):
    """Return ``{atom: position}`` with the metal at the origin and each donor on its polyhedron vertex."""
    ((centre,), (sites,)) = species.centres, species.sites.values()
    dirs = np.asarray(POLYHEDRA[centre.geometry].vertex_dirs, float)
    return {centre.atom: np.zeros(3)} | {atom: radius * dirs[v] for v, atom in enumerate(sites.vertices)}


def test_centre_from_slots_refuses_a_seating_of_no_one_isomer():
    pd = rx.parse_smiles(DIAMMINE_PD)
    for geometry, slots, message in (
        ("SPL", {0: 0, 1: 2, 2: 3, 9: 4}, "vertex 9 is not one of this geometry's 4 vertices"),
        ("SPL", {0: 0, 1: 2, 2: 3, 3: 0}, "donor atom 0 is seated at two vertices"),
        ("SPL", {0: 0, 1: 2, 2: 3}, r"donor\(s\) \[4\] were given no vertex"),
        ("linear", {0: 0, 1: 4}, "linear has 2 coordination sites but metal 1 has 4"),
    ):
        with pytest.raises(ValueError, match=message):
            seated(pd, geometry, slots)


def test_seat_vacancies_refuses_an_atom_that_cannot_fill_it():
    """An H on a bound N would merge into its site or split a side-on N-N site, so the whole site is bound instead."""
    wet = seated(Chem.AddHs(rx.parse_smiles(DIAMMINE_PD + ".O.O.O")), "OCT", [0, 2, 3, 4])
    hydrazine = seated(Chem.AddHs(rx.parse_smiles("N1N->[Pd+2]<-1(<-[Cl-])<-[Cl-]")), "OCT", {0: 0, 1: 3, 2: 4})
    for species, atoms, message in (
        (wet, [99], r"must be in 0-\d+; got \[99\]"),
        (wet, [5, 5], "must be distinct"),
        (wet, [0], "already donors of metal 1"),
        (wet, [1], "must be ligand donors, not metal atoms"),
        (wet, [8], "would regroup"),
        (hydrazine, [6], "would regroup"),
        (wet, [5, 6, 7], r"has 2 vacant site\(s\) but 3 atom\(s\) to coordinate"),
    ):
        with pytest.raises(ValueError, match=message):
            species.seat_vacancies(species.centres[0].atom, atoms)


def test_seat_vacancies_returns_a_new_species_bound_there():
    """Binding the free end of a kappa1 ethylenediamine closes the chelate ring; the record it came from stays."""
    kappa1 = seated(rx.parse_smiles("NCCN->[Pd+2](<-[Cl-])<-[Cl-]"), "SPL", [3, 5, 6])  # N0 C1 C2 N3 Pd4 Cl5 Cl6
    before = kappa1.mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    chelate = kappa1.seat_vacancies(4, [0])

    assert kappa1.seat_vacancies(4, []) is kappa1
    assert chelate.centres[0].vertices[3] == Site(atoms=(0,))
    assert chelate.mol.GetBondBetweenAtoms(0, 4).GetBondType() == Chem.BondType.DATIVE
    assert any({0, 4} <= set(ring) for ring in chelate.mol.GetRingInfo().AtomRings())
    assert kappa1.mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_species_prints_a_row_per_centre_then_ligand_stereo():
    surrogated = Species(mol=Chem.MolFromSmiles("[Fe+2].[Mn]"), centres=(Centre(0, 26, 2), Centre(1, 25, 0)))
    propene = rx.isomers("[Cl-]->[Pt+2]12(<-[Cl-])(<-[CH2]=[CH](C)->1)<-[CH2]=[CH](C)->2", geometry="SPL")[1]
    for species, shown in (
        (Species(mol=Chem.MolFromSmiles("CCO")), ""),
        (surrogated, "Fe0 [surrogated] ; Mn1 [surrogated]  -"),
        (seated(rx.parse_smiles(DIAMMINE_PD), "SPL", [0, 2, 4, 3]), "Pd1 SPL trans [N0 Cl2 N4 Cl3] -  -"),
        (seated(rx.parse_smiles(ZINC), "TET", [0, 2, 3, 4]), "Zn1 TET [F0 Cl2 Br3 I4] Δ  -"),
        (read_species(propene), "Pt1 SPL cis [Cl0 Cl2 η2(re) η2(si)] - meso  -"),
    ):
        assert (str(species), repr(species)) == (shown, f"Species({shown!r})"), shown


def test_misplaced_pins_names_sphere_atoms_pinned_off_vertices():
    """Swapping two donors of a tetrahedron is its mirror image, which compiled angles cannot tell apart."""
    zinc = seated(rx.parse_smiles(ZINC), "TET", [0, 2, 3, 4])
    cis = seated(rx.parse_smiles(DIAMMINE_PD), "SPL", [0, 4, 2, 3])
    zinc_core, cis_core = _on_vertices(zinc), _on_vertices(cis)
    mirrored = {{0: 2, 2: 0}.get(atom, atom): position for atom, position in zinc_core.items()}
    trans = {{2: 4, 4: 2}.get(atom, atom): position for atom, position in cis_core.items()}
    for name, species, core, angles_only, misplaced in (
        ("tetrahedron-as-seated", zinc, zinc_core, False, []),
        ("tetrahedron-mirrored", zinc, mirrored, False, [0, 2, 3, 4]),
        ("TET mirrored, angles only", zinc, mirrored, True, []),
        ("cis-palladium-pinned-trans", cis, trans, True, [0, 2, 3, 4]),
        ("cis-palladium-unpinned", cis, {}, False, []),
    ):
        assert misplaced_pins(species, core, angles_only=angles_only) == misplaced, name


def test_retained_geometry_reads_the_best_fitting_polyhedron():
    """A squashed six-site sphere fits no polyhedron and a CN5 sphere ties two; each still reads the nearest."""
    for name, dirs, geometry in (
        ("ideal-bipyramid", POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, "trigonal_bipyramidal"),
        ("squashed-six", SQUASHED_SIX, "hexagonal_planar"),
        ("bipyramid-pyramid-tie", SPY_TBP_TIE, "trigonal_bipyramidal"),
    ):
        sphere = ideal_sphere(dirs, 2.1)
        assert retained_geometry(sphere, 0, list(range(1, len(dirs) + 1)), {}) == geometry, name


def _thirteen_fluorides():
    """Return LaF13 with its fluorides on a puckered ring around the metal."""
    mol = rx.parse_smiles("[La+10]" + "(<-[F-])" * 12 + "<-[F-]")
    angles = 2 * np.pi * np.arange(13) / 13
    ring = [(2 * np.cos(a), 2 * np.sin(a), 0.3 * (k % 3 - 1)) for k, a in enumerate(angles)]
    return placed(mol, [(0, 0, 0), *ring])


def test_observed_centre_refuses_an_unreadable_sphere():
    """Ag(NH3)2+ with one ammine moved onto the silver has no site direction, so no linear reading."""
    for mol, geometry, message in (
        (_thirteen_fluorides(), None, r"no polyhedron template for '13-coordinate'"),
        (placed("[Hg].[C-]#[O+]", [(0, 0, 0), (3, 0, 0), (4.1, 0, 0)]), None, "no donor bonds"),
        (rx.parse_smiles("N->[Ag+]<-N"), None, "needs an input geometry"),
        (
            placed(rx.parse_smiles(DIAMMINE_PD), [(2, 0, 0), (0, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0)]),
            "tetrahedral",
            "do not take the requested tetrahedral geometry",
        ),
        (
            placed(rx.parse_smiles("N->[Ag+]<-N"), [(0, 0, 0), (0, 0, 0), (2, 0, 0)]),
            None,
            "do not take the linear geometry their sites are read in",
        ),
    ):
        facts = LigandFacts(chemical_graph(canonical_metal_graph(mol)))
        metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in {"La", "Hg", "Ag", "Pd"})
        with pytest.raises(ValueError, match=message):
            observed_centre(facts, metal, geometry)


def test_site_details_list_site_relations_and_ambiguous_donors():
    """A carbonyl and a methanide are both C, so each is named by its ligand; a TBP lists its axial and equatorial."""
    for smiles, choices, lines in (
        (
            "[O+]#[C-]->[Co+3](<-[CH3-])(<-[Cl-])(<-N)(<-[F-])<-P",
            {"geometry": "OCT"},
            ["Co2 trans: C1-C3, Cl4-N5, F6-P7", "Co2 C1: [C-]#[O+]", "Co2 C3: [CH3-]"],
        ),
        (
            "[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O",
            {"geometry": "TBP"},
            ["Fe2 axial: C1 F3; equatorial: Cl4 N5 O6"],
        ),
        (FERROCENE_TWO_FACES, {"stereo": "racemic"}, ["Fe2 faces: η5Rₚ(1,3,4,5,6), η5Rₚ(8,9,10,12,14)"]),
    ):
        species = read_species(rx.isomers(smiles, **choices)[0])
        assert [line.strip() for line in site_details(species)] == lines, smiles


def test_matches_holds_each_selector_against_the_species():
    """The centre selectors hold together at one centre; stereo and haptic compare ligand and face hands."""
    pd, zinc = (
        seated(rx.parse_smiles(DIAMMINE_PD), "SPL", [0, 2, 4, 3]),
        seated(rx.parse_smiles(ZINC), "TET", [0, 2, 3, 4]),
    )
    rp, sp = (read_species(one) for one in rx.isomers(FERROCENE_ONE_FACE, stereo="racemic"))
    rac, meso, _other_rac = (read_species(one) for one in rx.isomers(FERROCENE_TWO_FACES, stereo="racemic"))
    butanol = read_species(rx.isomers("C[C@@H](O)CC")[0])
    for name, species, keys, matched in (
        ("nothing-selected", pd, {}, True),
        ("polyhedron-and-label", pd, {"geometry": "square_planar", "label": "trans"}, True),
        ("another-label", pd, {"label": "cis"}, False),
        ("arrangement", pd, {"arrangement": "N0 Cl2 N4 Cl3"}, True),
        ("label-at-its-index", pd, {"center": 1, "label": "trans"}, True),
        ("an-element-it-lacks", pd, {"center": "Pt"}, False),
        ("hand-by-name", zinc, {"hand": "delta"}, True),
        ("other-hand-by-glyph", zinc, {"hand": "Λ"}, False),
        ("ligand-stereo", butanol, {"stereo": "R"}, True),
        ("other-ligand-stereo", butanol, {"stereo": "S"}, False),
        ("planar-hand", rp, {"haptic": "Rp"}, True),
        ("other-planar-hand", sp, {"haptic": "Rₚ"}, False),
        ("planar-hand-spelled-s-p", sp, {"haptic": "S_p"}, True),
        ("rac-pair", rac, {"haptic": "rac"}, True),
        ("meso-pair-asked-rac", meso, {"haptic": "rac"}, False),
        ("meso-pair-by-face-atom", meso, {"haptic": {1: "Rp", 8: "Sp"}}, True),
        ("rac-pair-by-face-atom", rac, {"haptic": {1: "Rp", 8: "Sp"}}, False),
    ):
        assert matches(species, keys) is matched, name


def test_matches_refuses_a_selector_naming_no_one_centre_or_face():
    dimer = read_species(rx.isomers(PD2CL6)[0])
    rac = read_species(rx.isomers(FERROCENE_TWO_FACES, stereo="racemic")[0])
    for species, keys, error, message in (
        (dimer, {"center": 1.5}, TypeError, "center must be an atom index or element symbol"),
        (dimer, {"center": "Pd"}, ValueError, "name one by its atom index"),
        (rac, {"haptic": "Rp"}, ValueError, r"leaves 2 orientable faces unspecified; use haptic=\{face_atom: hand\}"),
    ):
        with pytest.raises(error, match=message):
            matches(species, keys)
