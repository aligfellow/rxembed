"""Test donor-to-polyhedron seating and arrangement labels."""

import itertools
from collections import Counter

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_polyhedron as poly
from rxembed import metal_slots as slots
from rxembed.metal_core import VACANT


def _frag_of(iso, fragments, vertex):
    """Return the ligand fragment containing one real or haptic vertex."""
    return fragments[iso.haptic[vertex][0]] if vertex in iso.haptic else fragments[vertex]


def _same_ligand_vertex_angles(isomers):
    """Yield each same-ligand donor pair and its polyhedron angle."""
    fragments = {atom: i for i, frag in enumerate(Chem.GetMolFrags(isomers[0].mol)) for atom in frag}
    for iso in isomers:
        directions = poly.POLYHEDRA[iso.geometry].vertex_dirs
        for i, j in itertools.combinations(range(len(iso.vertices)), 2):
            left, right = iso.vertices[i], iso.vertices[j]
            if VACANT in (left, right) or _frag_of(iso, fragments, left) != _frag_of(iso, fragments, right):
                continue
            yield iso, left, right, poly._vertex_angle(directions[i], directions[j])


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
        ("NCCN.C[O-]", [0, 3, 4, 5], 1),
        ("NCCN(C).C[O-]", [0, 3, 5, 6], 2),
    ],
    ids=("symmetric", "asymmetric"),
)
def test_ordering_dedup_respects_donor_symmetry(smiles, donors, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert len(slots.distinct_vertex_orderings(mol, donors, "square_planar")) == expected


def test_short_chelate_excludes_trans():
    smiles = "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    short = 0
    for iso, left, right, angle in _same_ligand_vertex_angles(rx.metal(smiles, "square_planar")):
        if Chem.GetDistanceMatrix(iso.mol)[left][right] <= 4:
            short += 1
            assert angle < slots.CHELATE_SPAN_ANGLE
    assert short, "the span filter was not exercised"


def test_flexible_chelate_may_span_trans_but_short_chelate_may_not():
    smiles = (
        "Cc1cc(C)c(N2C=CN3CCN4C=CN(c5c(C)cc(C)cc5C)[C]4->[Ni+2]4(<-[O-]C(=O)C(c5ccccc5)[N-]->4c4ccccc4)<-[C]32)c(C)c1"
    )
    short = 0
    for iso, left, right, angle in _same_ligand_vertex_angles(rx.metal(smiles, "square_planar")):
        if Chem.GetDistanceMatrix(iso.mol)[left][right] <= 4:
            short += 1
            assert angle < slots.CHELATE_SPAN_ANGLE
    assert short, "the span filter was not exercised"


def test_t_shape_seats_its_trans_pair_first():
    iso = rx.metal("CP(C)(C)->[Rh](Cl)<-P(C)(C)C", "t_shape").select(index=0)
    seated = [iso.mol.GetAtomWithIdx(vertex).GetSymbol() for vertex in iso.vertices]
    assert (seated[0], seated[2]) == ("P", "P")
    assert seated[1] == "Cl"
