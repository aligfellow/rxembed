"""Test donor-to-polyhedron seating and arrangement labels."""

import itertools
import random
from collections import Counter

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_slots as slots
from rxembed.metal_polyhedron import CHELATE_SPAN_ANGLE, hull_edges, vertex_angle, vertex_dirs


def _wide_narrow(dirs, pair):
    """Return a `narrow` mapping that forbids `pair` at every vertex angle >= CHELATE_SPAN_ANGLE.

    Mirrors `metal_screen.narrow_span_pairs`'s per-angle form while keeping the old, angle-agnostic "wide
    bite" test fixtures: those cared only that a pair could not span *some* wide angle, not which one.
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
    """An explicit `perms` (a `fix=` pool, or `observed_only`'s one retained order) is authoritative.

    Regression: the forbidden-pair filter used to run on `perms` too, so a real (tmQMg) `observed_only`
    structure whose measured order happened to land on a forbidden vertex pair was silently zeroed out
    (ROGWIW, IKOYOX, KUVQOK all went from 1 to 0). Streamed (generated) orders are still pruned.
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


def _drop_forbidden_orderings(perms, dirs, narrow, linked=frozenset()):
    """Filter a streamed order out when a constrained donor pair sits on a vertex pair its rule forbids.

    Two rules share this one filter, each mapping its constrained donor pairs to a forbidden set of vertex
    pairs: `narrow` (a mapping of vertex angle to the same-ligand pairs that cannot span it) is forbidden on
    the vertex pairs at that angle; `linked` (`metal_enumeration._isomers_for_geometry`'s chelate-backbone
    or direct-bond pairs, see `chelate_edge_links`) is forbidden on the non-edge (`hull_edges`) vertex pairs.
    A donor pair in both takes the union: it is dropped by whichever vertex-pair check it lands on. Both sets
    are computed once per geometry, and either has already proven no completion can survive its screen. Empty
    `narrow` and `linked` pass `perms` through unwrapped, so the common untethered/unpruned case pays nothing.

    Kept as the sweep-then-filter baseline `_seat_pruned_orderings` must reproduce element for element; the
    live tethered/haptic enumeration path in `distinct_vertex_orderings` calls the backtracking generator instead.
    """
    if not narrow and not linked:
        return perms
    wide, off_edge = slots._forbidden_vertex_pairs(dirs, narrow, linked)

    def forbidden(order):
        if any(frozenset((order[i], order[j])) in forbidden_pairs for i, j, forbidden_pairs in wide):
            return True
        return any(frozenset((order[i], order[j])) in linked for i, j in off_edge)

    return (order for order in perms if not forbidden(order))


@pytest.mark.parametrize(
    "geometry",
    [
        "square_planar",
        "tetrahedral",
        "trigonal_bipyramidal",
        "octahedral",
        "pentagonal_bipyramidal",
        "square_antiprism",
    ],
)
def test_pruned_backtracking_matches_sweep_then_filter(geometry):
    """`_seat_pruned_orderings` must reproduce `_drop_forbidden_orderings(isomer_permutations(...))` exactly.

    Both apply the same conjunction of independent per-vertex-pair predicates (`_forbidden_vertex_pairs`), so
    rejecting a partial assignment the moment a forbidden pair is fully seated is sound (a violated pair can
    never un-violate) and complete (every pair is tested once, at its later vertex); placing vertices 0..n-1
    with donors tried in increasing index order reproduces `itertools.permutations`' lexicographic order with
    the dead subtrees skipped. `random.Random(geometry)` seeds deterministically from the string, independent
    of PYTHONHASHSEED.
    """
    dirs = vertex_dirs(geometry)
    n = len(dirs)
    rotations = slots.point_group(tuple(map(tuple, dirs)))[0]
    rng = random.Random(geometry)
    pairs = [frozenset((i, j)) for i in range(n) for j in range(i + 1, n)]
    wide_angles = {round(vertex_angle(dirs[i], dirs[j]), 6) for i in range(n) for j in range(i + 1, n)}
    wide_angles = {angle for angle in wide_angles if angle >= CHELATE_SPAN_ANGLE}
    narrow_pairs = frozenset(rng.sample(pairs, max(1, n // 2)))
    narrow = dict.fromkeys(wide_angles, narrow_pairs)
    linked = frozenset(rng.sample(pairs, max(1, n // 2)))

    baseline = list(_drop_forbidden_orderings(slots.isomer_permutations(geometry), dirs, narrow, linked))
    pruned = list(slots._seat_pruned_orderings(dirs, rotations, narrow, linked))

    assert pruned == baseline
    assert pruned, f"{geometry}: the random narrow/linked draw forbids nothing, so this proves nothing"


def _decadentate_chain_mol(n_donors=10):
    """Build a single-fragment N-C-C-N-...-N chain: every consecutive donor pair is chelate-backbone-linked."""
    rw = Chem.RWMol()
    atoms = []
    for i in range(3 * n_donors - 2):
        atoms.append(rw.AddAtom(Chem.Atom(7 if i % 3 == 0 else 6)))
    for i in range(len(atoms) - 1):
        rw.AddBond(atoms[i], atoms[i + 1], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, atoms[::3]


def test_tethered_cn10_prune_enumerates_far_fewer_than_the_full_orbit_sweep(monkeypatch):
    """A tethered CN10 case stays well under 100k explored orderings, mirroring the cap-precedes-sweep guard.

    Regression for the brute-force sweep this replaces: `isomer_permutations('bicapped_square_antiprismatic')`
    alone iterates 10! = 3,628,800 raw permutations before any filtering. A decadentate chelate chain gives 9
    same-ligand backbone-linked donor pairs (`chelate_edge_links`), which the backtracking prune should reject
    long before most of that sweep is ever built. The `isomer_permutations` trap catches a reversion to the old
    sweep-then-filter call directly (mirroring `test_high_coordination_cap_precedes_permutation_generation`);
    the explored count separately catches a prune that runs but no longer rejects a dead branch early.
    """
    mol, donors = _decadentate_chain_mol()
    geometry = "bicapped_square_antiprismatic"
    linked = slots.chelate_edge_links(mol, donors)
    assert linked, "the chain must supply at least one backbone link for this test to exercise the prune"

    monkeypatch.setattr(
        slots, "isomer_permutations", lambda _geometry: pytest.fail("fell back to the full orbit sweep")
    )
    explored = 0
    real_seat = slots._seat_pruned_orderings

    def counting(*args, **kwargs):
        nonlocal explored
        for order in real_seat(*args, **kwargs):
            explored += 1
            yield order

    monkeypatch.setattr(slots, "_seat_pruned_orderings", counting)
    try:
        slots.distinct_vertex_orderings(mol, donors, geometry, linked=linked)
    except ValueError:
        pass  # the 1,000-orbit cap may still trip; only the explored-orderings count is under test here

    assert explored < 100_000


def test_high_coordination_enumeration_refuses_the_exact_orbit_pool():
    mol = Chem.MolFromSmiles(".".join(f"[{isotope}F-]" for isotope in range(1, 11)))
    donors = list(range(10))

    with pytest.raises(ValueError, match=r"more than 1,000 distinct constitutional.*rx\.embed.*rx\.metal"):
        slots.distinct_vertex_orderings(mol, donors, "BSA")


def test_high_coordination_cap_precedes_permutation_generation(monkeypatch):
    mol = Chem.MolFromSmiles(".".join(f"[{isotope}F-]" for isotope in range(1, 11)))

    monkeypatch.setattr(slots, "isomer_permutations", lambda _geometry: pytest.fail("permutations were generated"))
    with pytest.raises(ValueError, match="more than 1,000 distinct constitutional"):
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

    assert values == [(58.0, 81.0)] * 2


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

    # The measured effect on enumeration: forbidding a-b too (what the edge rule would do without the
    # exclusion) is strictly more restrictive on a geometry whose only non-edge is the a-b vertex pair.
    without_exclusion = frozenset({frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))})
    donors = [*padded, *(rw.AddAtom(Chem.Atom(9)) for _ in range(2))]  # pad to CN5 with 2 unrelated F- sites
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    geometry = "trigonal_bipyramidal"  # 10 vertex pairs, 9 edges: only the axial pair is a non-edge

    unpruned = slots.distinct_vertex_orderings(mol, donors, geometry)
    pruned_with = slots.distinct_vertex_orderings(mol, donors, geometry, linked=with_exclusion)
    pruned_without = slots.distinct_vertex_orderings(mol, donors, geometry, linked=without_exclusion)

    assert len(unpruned) == 6
    assert len(pruned_with) == 5, "the exclusion should drop only completions putting a-c or c-b off an edge"
    assert len(pruned_without) == 4, "without the exclusion, a-b is also forced off the one non-edge pair"


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
