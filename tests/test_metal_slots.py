"""Test donor-to-polyhedron seating and arrangement labels."""

import itertools
from collections import Counter

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_slots as slots
from rxembed.metal_polyhedron import CHELATE_SPAN_ANGLE, hull_edges, vertex_angle, vertex_dirs


def _wide_narrow(dirs, pair):
    """Return a `narrow` mapping that forbids `pair` at every vertex angle >= CHELATE_SPAN_ANGLE.

    Mirrors `metal_screen.narrow_span_pairs`'s per-angle form for a fixture that only cares that a pair
    cannot span *some* wide angle, not which one.
    """
    angles = {round(vertex_angle(p, q), 6) for p, q in itertools.combinations(dirs, 2)}
    return {angle: frozenset({pair}) for angle in angles if angle >= CHELATE_SPAN_ANGLE}


def test_square_pyramidal_150_degree_pair_is_trans():
    isomers = rx.metal("[V](F)(F)(Cl)(Cl)Cl", "SPY", stereo="free")
    assert Counter(iso.label for iso in isomers) == {"cis": 2, "trans": 1}


def test_three_plus_one_square_planar_has_no_false_cis_trans_label():
    isomers = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL", stereo="free")
    assert len(isomers) == 1
    assert isomers[0].label == ""


@pytest.mark.parametrize(
    ("smiles", "donors", "expected"),
    [
        ("NCCN.[C-].[O-]", [0, 3, 4, 5], 2),
        ("NCCN(C).[C-].[O-]", [0, 3, 5, 6], 3),
    ],
    ids=("symmetric", "asymmetric"),
)
def test_ordering_dedup_respects_donor_symmetry(smiles, donors, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert len(slots.distinct_vertex_orderings(mol, donors, "square_planar")) == expected


@pytest.mark.parametrize(
    ("smiles", "donors", "geometry"),
    [
        ("NCCN.[Cl-].[Br-].[I-]", [0, 3, 4, 5, 6], "trigonal_bipyramidal"),
        ("NCCN.[F-].[Cl-].[Br-].[I-]", [0, 3, 4, 5, 6, 7], "octahedral"),
    ],
    ids=("tbp", "octahedral"),
)
def test_narrow_drops_only_the_orderings_placing_the_pair_on_a_wide_vertex_pair(smiles, donors, geometry):
    """`narrow` removes exactly the streamed orders placing donor positions 0/1 >= CHELATE_SPAN_ANGLE apart.

    All 5-6 donor classes here are distinct, so each constitutional identity has one raw representative and
    dropping it cannot be silently replaced by a donor-symmetric stand-in (unlike the equivalent-halide case).
    """
    mol = Chem.MolFromSmiles(smiles)
    dirs = vertex_dirs(geometry)
    narrow = _wide_narrow(dirs, frozenset((0, 1)))

    unpruned = slots.distinct_vertex_orderings(mol, donors, geometry)
    pruned = slots.distinct_vertex_orderings(mol, donors, geometry, narrow=narrow)

    expected = [
        order for order in unpruned if vertex_angle(dirs[order.index(0)], dirs[order.index(1)]) < CHELATE_SPAN_ANGLE
    ]
    assert pruned == expected
    assert len(pruned) < len(unpruned), f"{geometry}: narrow pair is never wide here"


def test_linked_drops_only_the_orderings_placing_the_pair_off_a_hull_edge():
    """`linked` removes exactly the streamed orders placing donor positions 0/1 off a polyhedron hull edge.

    A square-antiprism top-face diagonal (109 deg) is well under `CHELATE_SPAN_ANGLE` (135), so `narrow`
    would never catch it; the edge rule catches it because it is a diagonal, not because it is wide.
    """
    mol = Chem.MolFromSmiles("NCCN.[F-].[Cl-].[Br-].[I-].[H][At].[Se]")
    donors = [0, 3, 4, 5, 6, 7, 8, 9]
    geometry = "square_antiprism"
    dirs = vertex_dirs(geometry)
    edges = hull_edges(tuple(map(tuple, dirs)))
    linked = frozenset({frozenset((0, 1))})

    unpruned = slots.distinct_vertex_orderings(mol, donors, geometry, max_orbits=6000)
    pruned = slots.distinct_vertex_orderings(mol, donors, geometry, linked=linked, max_orbits=6000)

    expected = [order for order in unpruned if frozenset((order.index(0), order.index(1))) in edges]
    assert pruned == expected
    assert len(pruned) < len(unpruned), f"{geometry}: linked pair never lands off an edge here"


def test_explicit_perms_bypass_narrow_and_linked():
    """An explicit `perms` (a `fix=` pool, or `observed_only`'s one retained order) is authoritative and
    never dropped by the forbidden-pair filter, even when it lands on a forbidden vertex pair; streamed
    (generated) orders are still pruned.
    """
    mol = Chem.MolFromSmiles("NCCN.[Cl-].[Br-].[I-]")
    donors = [0, 3, 4, 5, 6]
    geometry = "trigonal_bipyramidal"
    dirs = vertex_dirs(geometry)
    unpruned = slots.distinct_vertex_orderings(mol, donors, geometry)
    bad = next(o for o in unpruned if vertex_angle(dirs[o.index(0)], dirs[o.index(1)]) >= CHELATE_SPAN_ANGLE)
    narrow = _wide_narrow(dirs, frozenset((0, 1)))

    assert bad not in slots.distinct_vertex_orderings(mol, donors, geometry, narrow=narrow)
    assert slots.distinct_vertex_orderings(mol, donors, geometry, perms=(bad,), narrow=narrow) == [bad]


def test_high_coordination_cap_precedes_permutation_generation(monkeypatch):
    mol = Chem.MolFromSmiles(".".join(f"[{isotope}F-]" for isotope in range(1, 11)))

    monkeypatch.setattr(slots, "isomer_permutations", lambda _geometry: pytest.fail("permutations were generated"))
    with pytest.raises(ValueError, match=r"more than 1,000 distinct constitutional.*rx\.embed.*rx\.metal"):
        slots.distinct_vertex_orderings(mol, list(range(10)), "BSA")


def test_high_coordination_does_not_disguise_one_observed_order_as_enumeration():
    mol = Chem.MolFromSmiles(".".join(f"[{isotope}F-]" for isotope in range(1, 11)))
    donors = list(range(10))

    with pytest.raises(ValueError, match="exact enumeration is required"):
        slots.distinct_vertex_orderings(mol, donors, "BSA", retained=donors)


@pytest.mark.parametrize("counts", [(9, 1), (8, 2)])
def test_high_coordination_counts_constitutional_classes_before_the_cap(counts):
    mol = Chem.MolFromSmiles(".".join(["[F-]"] * counts[0] + ["[Cl-]"] * counts[1]))

    assert slots.distinct_vertex_orderings(mol, list(range(10)), "BSA")


def test_high_coordination_identical_monodentates_keep_the_constant_time_shortcut():
    mol = Chem.MolFromSmiles(".".join(["[F-]"] * 10))
    donors = list(range(10))

    assert slots.distinct_vertex_orderings(mol, donors, "BSA") == [tuple(donors)]


def test_observed_orbit_precedes_canonical_enumeration():
    mol = Chem.MolFromSmiles("N.[P].[O-].[Cl-]")
    donors = list(range(4))
    retained = [2, 1, 0, 3]

    orderings = slots.distinct_vertex_orderings(mol, donors, "square_planar", retained=retained)

    assert orderings[0] == tuple(retained)


def test_chelate_bite_is_independent_of_equal_shortest_path_order():
    values = []
    for edges in (((0, 1), (1, 2), (0, 3), (3, 2)), ((0, 3), (3, 2), (0, 1), (1, 2))):
        rw = Chem.RWMol()
        for _ in range(4):
            rw.AddAtom(Chem.Atom(6))
        for edge in edges:
            rw.AddBond(*edge, Chem.BondType.SINGLE)
        values.append(slots.chelate_bite_window(rw.GetMol(), 0, 2, donors=(0, 1, 2)))

    assert values[0] is not None
    assert values[0] == values[1]


def test_chelate_edge_links_covers_a_bite_window_pair_and_a_bonded_pair():
    """A backbone-linked pair (a 2-4 bond chelate arm) and a directly-bonded pair (a 3-membered ring) both link."""
    rw = Chem.RWMol()
    atoms = [rw.AddAtom(Chem.Atom(7 if i in (0, 3, 6) else 6)) for i in range(7)]  # N-C-C-N-C-C-N, a dien chain
    for i in range(6):
        rw.AddBond(atoms[i], atoms[i + 1], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    assert slots.chelate_edge_links(mol, [0, 3, 6]) == frozenset({frozenset((0, 1)), frozenset((1, 2))})

    rw2 = Chem.RWMol()
    n1, n2 = rw2.AddAtom(Chem.Atom(7)), rw2.AddAtom(Chem.Atom(7))
    rw2.AddBond(n1, n2, Chem.BondType.SINGLE)  # a directly-bonded donor pair, e.g. a bridging hydrazido N-N
    mol2 = rw2.GetMol()
    mol2.UpdatePropertyCache(strict=False)

    assert slots.chelate_edge_links(mol2, [n1, n2]) == frozenset({frozenset((0, 1))})


def test_chelate_edge_links_through_donor_exclusion_drops_the_redundant_outer_link():
    """When a third donor sits astride the a-b backbone, a-b is dropped; the shorter a-c and c-b hold.

    An 8-membered N-C-N-C-N-C-C-C ring gives donors a/c/b at ring positions 0/2/4: a-c and c-b are each a
    2-bond, 4-membered chelate arm, and a-b's own donor-free backbone (the other way around the ring) is
    also 4 bonds, exactly 2 + 2, so a-b is redundant with a-c plus c-b and is dropped.
    """
    rw = Chem.RWMol()
    atoms = [rw.AddAtom(Chem.Atom(7 if i in (0, 2, 4) else 6)) for i in range(8)]
    for i in range(8):
        rw.AddBond(atoms[i], atoms[(i + 1) % 8], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    padded = [atoms[0], atoms[2], atoms[4]]

    with_exclusion = slots.chelate_edge_links(mol, padded)
    assert with_exclusion == frozenset({frozenset((0, 1)), frozenset((1, 2))})


def test_haptic_tether_dedup_is_atom_order_invariant():
    mol = rx.parse_smiles("[N]1=[CH](CCC[NH2]->2)->[Ni+2]2(<-[Cl-])(<-[Cl-])<-1")
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    def labels(graph):
        return {iso.label for iso in rx.metal(graph, "square_planar", stereo="free")}

    assert labels(mol) == labels(renumbered) == {"cis", "trans"}


def test_square_pyramid_does_not_force_a_large_haptic_face_to_its_apex():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(24))
    chain = [rw.AddAtom(Chem.Atom(6)) for _ in range(8)]
    orders = [
        Chem.BondType.DOUBLE,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.SINGLE,
        Chem.BondType.SINGLE,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
    ]
    for left, right, order in zip(chain[:-1], chain[1:], orders, strict=True):
        rw.AddBond(left, right, order)
    for donor in chain[:4] + chain[6:]:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    for _ in range(3):
        carbon, oxygen = rw.AddAtom(Chem.Atom(6)), rw.AddAtom(Chem.Atom(8))
        rw.GetAtomWithIdx(carbon).SetFormalCharge(-1)
        rw.GetAtomWithIdx(oxygen).SetFormalCharge(1)
        rw.AddBond(carbon, oxygen, Chem.BondType.TRIPLE)
        rw.AddBond(carbon, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)

    expected = None
    for graph in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
        isomers = rx.metal(graph, "square_pyramidal", stereo="free")
        positions = {
            iso.vertices.index(next(site for site, face in iso.haptic.items() if len(face) == 4)) for iso in isomers
        }
        assert 0 in positions
        assert positions - {0}
        texts = {rx.cxsmiles(iso) for iso in isomers}
        expected = texts if expected is None else expected
        assert texts == expected


def test_t_shape_seats_its_trans_pair_first():
    iso = rx.metal("CP(C)(C)->[Rh](Cl)<-P(C)(C)C", "t_shape").select(index=0)
    seated = [iso.mol.GetAtomWithIdx(vertex).GetSymbol() for vertex in iso.vertices]
    assert (seated[0], seated[2]) == ("P", "P")
    assert seated[1] == "Cl"
