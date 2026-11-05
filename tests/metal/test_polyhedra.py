"""`metal/polyhedra.py`: the `POLYHEDRA` registry, each record named, read, seated in a hand and seeded."""

from __future__ import annotations

from rdkit import Chem

from rxembed.identity.smiles_io import mol_from_species
from rxembed.metal.polyhedra import POLYHEDRA, describe, mirror_seating, resolve_geometry
from rxembed.metal.shape import classify_geometry, shape_gap
from tests.metal_fixtures import ideal_sphere, orientation_parity_of, seated, seeds


def test_each_polyhedron_reads_and_seeds_as_itself():
    """Each record resolves from its name or code in any case, and its own vertices read as it, untied.

    Uranium with a fluoride of its own isotope at each vertex, so every donor differs, seats a hand on a nonplanar
    record and its mirror seating the other; a planar one has none. A seed reads as the record and, when handed, in
    the seated hand. No record is invented for a bare coordination number.
    """
    read, expected = {}, {}
    for polyhedron in POLYHEDRA.values():
        name, code, cn = polyhedron.name, polyhedron.code, polyhedron.cn
        best = classify_geometry(ideal_sphere(polyhedron.vertex_dirs, 2.0), 0, list(range(1, cn + 1)), {})
        fluorides = Chem.MolFromSmiles("[U]" + "".join(f"(<-[{19 + vertex}F-])" for vertex in range(cn)))
        species = seated(fluorides, name, list(range(1, cn + 1)))
        vertices = [site.atoms[0] for site in species.centres[0].vertices]
        mirror = seated(mol_from_species(species), name, mirror_seating(polyhedron.vertex_dirs, vertices))
        seed = seeds(mol_from_species(species), 1, seed=7)
        read[name] = (
            {resolve_geometry(spelling) for spelling in (name, name.upper(), f" {code.title()} ", code.lower())},
            [describe(code), describe(f"{cn}-coordinate"), best.name, best.reading.verdict, best.tied],
            {species.centres[0].hand, mirror.centres[0].hand},
            shape_gap(seed.mol, 0, vertices, {}, name, seed.ids[0]).accepted,
            None if polyhedron.planar else orientation_parity_of(seed.mol, seed.ids[0], species),
        )
        hands = {""} if polyhedron.planar else {"delta", "lambda"}
        describes = [f"{name} ({code}, CN {cn})", f"{cn}-coordinate", name, "fits", False]
        expected[name] = ({name}, describes, hands, True, None if polyhedron.planar else 1)

    assert read == expected
