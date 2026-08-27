"""Test selected metal states, collections, and retained geometry."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_constraints as constraints
from rxembed import metal_core
from rxembed import metal_isomer as isomer
from rxembed import metal_polyhedron as poly
from rxembed import metal_slots as slots
from rxembed.metal_core import VACANT
from rxembed.pipeline import geom_check
from tests.metal_fixtures import ferrocene

_MA2B2 = "CCCN[Pd](Cl)(Cl)NCCC"
_PT_A2B2 = "[NH3]->[Pt](<-[NH3])(Cl)Cl"


def _pt():
    return Chem.AddHs(Chem.MolFromSmiles(_PT_A2B2))


def _assert_clean(ensemble):
    ensemble = ensemble.minimize()
    assert ensemble.n >= 1
    for cid in ensemble.ids:
        report = geom_check.check(ensemble.mol, cid)
        assert report.ok(), report.summary()


def _angle(positions, left, center, right):
    a, b = positions[left] - positions[center], positions[right] - positions[center]
    return float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))


def test_isomer_summary_uses_compact_selectable_stereo(capsys):
    isomers = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "OCT", stereo="free")
    assert isomers.summary() is None
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "  idx  centres (geometry label [slots] Δ/Λ/-)  ligand"
    assert all("OCT" in line and ("Δ" in line or "Λ" in line) for line in lines[1:])

    point = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH2]C(C)O", "SPL")
    alkene = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-NCC=CC", "SPL")
    point.summary()
    alkene.summary()
    shown = capsys.readouterr().out
    assert {candidate.stereo_label for candidate in point} == {"C5:R", "C5:S"}
    assert point.filter(stereo="C5:R") == point.filter(stereo="5R") == point.filter(stereo="R")
    assert alkene.filter(stereo="E") == alkene.filter(stereo="C6=C7:E")
    assert all(label in shown for label in ("C5:R", "C5:S", "C6=C7:E", "C6=C7:Z"))


def test_enumeration_compiles_only_the_selected_isomer(monkeypatch, capsys):
    calls = 0
    build = constraints.coordination

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return build(*args, **kwargs)

    monkeypatch.setattr(constraints, "coordination", counted)
    isomers = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "OCT", stereo="free")
    assert len(isomers) == 30
    assert calls == 0
    isomers.summary()
    capsys.readouterr()
    rx.cxsmiles(isomers[0])
    assert calls == 0
    _ = isomers[0].cons
    _ = isomers[1].cons
    assert calls == 2


def test_constraint_compilation_does_not_mutate_isomer():
    candidate = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[O-]C=O", "SPL")[0]
    expected = candidate.cons
    changed = candidate.cons
    changed.distances.clear()
    assert candidate.cons == expected


def test_isomers_do_not_share_their_public_molecule():
    first, second = rx.metal(r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "SPL")[:2]
    assert first.mol is not second.mol
    first.restore()
    assert second.mol.GetAtomWithIdx(second.metal).GetAtomicNum() != second.real_z


def test_metal_states_are_read_only():
    candidate = rx.metal("N->[Pd+2](<-[Cl-])(<-[Br-])<-[F-]", "SPL")[0]
    with pytest.raises(AttributeError):
        candidate.centres = tuple(reversed(candidate.centres))


def test_center_selector_rejects_bool_and_float():
    isomers = rx.metal(_MA2B2, "SPL")
    for center in (True, 1.5):
        with pytest.raises(TypeError, match="center must be"):
            isomers.filter(center=center)


def test_summary_details_show_site_relations_and_non_equivalent_donors(capsys):
    octahedral = rx.metal("[O+]#[C-]->[Co+3](<-[CH3-])(<-[Cl-])(<-N)(<-[F-])<-P", "OCT", stereo="free")[0]
    octahedral.summary(details=True)
    shown = capsys.readouterr().out
    assert "Co2 trans:" in shown
    assert "Co2 C1: [C-]#[O+]" in shown
    assert "Co2 C3: [CH3-]" in shown

    tbp = rx.metal("[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O", "TBP", stereo="free")[0]
    tbp.summary(details=True)
    shown = capsys.readouterr().out
    assert "Fe2 axial:" in shown
    assert "equatorial:" in shown


def test_select_accepts_geometry_code_or_name():
    isomers = rx.metal(_MA2B2, "square_planar")
    first = isomers[0]
    assert isomers.select(arrangement=isomer.arrangement(first)).vertices == first.vertices
    assert isomers.select(index=0).vertices == first.vertices
    with pytest.raises(ValueError, match="matched"):
        isomers.select(arrangement="does not exist")
    tetrahedral = rx.metal("[Zn](F)(Cl)(Br)I", "TET")
    assert {candidate.geometry for candidate in tetrahedral} == {"tetrahedral"}
    assert tetrahedral.select(hand="delta") is tetrahedral.select(hand="Δ")


def test_isomer_repr_is_the_compact_summary_row():
    candidates = rx.metal("O[Co](Cl)(C)(N)(F)P", "OCT")
    assert "Constraints(" not in repr(candidates)
    assert "Co1 OCT" in repr(candidates)


def test_shape_only_summary_lists_every_restored_metal(capsys):
    candidate = isomer.from_surrogate(Chem.MolFromSmiles("[C].[C]"), [(0, 26, 2), (1, 25, 0)], [])
    candidates = isomer.IsomerSet([candidate])
    assert candidates.filter() == [candidate]
    assert candidates.select() is candidate
    candidate.summary(details=True)
    shown = capsys.readouterr().out
    assert "Fe0 [surrogated]" in shown
    assert "Mn1 [surrogated]" in shown
    assert "Constraints(" not in repr(candidate)


def test_known_isomer_seats_real_atom_indices():
    cis = isomer.Isomer(_pt(), "SPL", {0: 0, 1: 2, 2: 3, 3: 4})
    trans = isomer.Isomer(_pt(), "square_planar", [0, 3, 2, 4])
    assert (cis.label, trans.label) == ("cis", "trans")
    assert cis.geometry == trans.geometry == "square_planar"
    assert isomer.Isomer(_pt(), "OCT", {0: 0, 1: 2, 2: 3, 3: 4}).vertices.count(VACANT) == 2
    _assert_clean(rx.embed(trans, n=2, seed=1))


@pytest.mark.parametrize(
    ("sites", "match"),
    [
        ({0: 0}, "given no vertex"),
        ({0: 0, 1: 0, 2: 3, 3: 4}, "seated at two vertices"),
        ({0: 0, 1: 2, 2: 3, 9: 4}, "not one of this geometry"),
        ({0: 5, 1: 2, 2: 3, 3: 4}, "not a donor"),
    ],
)
def test_isomer_rejects_invalid_sites(sites, match):
    with pytest.raises(ValueError, match=match):
        isomer.Isomer(_pt(), "SPL", sites)


def test_isomer_rejects_an_unordered_site_set():
    with pytest.raises(TypeError, match="vertex-ordered"):
        isomer.Isomer(_pt(), "SPL", {0, 2, 3, 7})


def test_undefined_ligand_stereocentre_warns(caplog):
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH2](C(C)CC)->[Pd](<-[NH3])(Cl)Cl"))
    with caplog.at_level("WARNING", logger="rxembed"):
        isomer.Isomer(mol, "SPL", {0: 0, 1: 6, 2: 7, 3: 8})
    assert "undefined" in caplog.text
    assert "enumerate_isomers" in caplog.text


def test_isomer_source_and_metal_are_mutually_exclusive():
    candidate = rx.metal(_MA2B2, "square_planar")[0]
    with pytest.raises(ValueError, match="Isomer source OR metal"):
        rx.embed(candidate, metal="square_planar")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_retained_input_geometry_is_logged_as_relaxed(tmp_path, caplog):
    xyz = tmp_path / "pd.xyz"
    rx.embed(rx.metal(_MA2B2, "square_planar")[0], n=1, seed=1).dump(str(xyz))
    with caplog.at_level("INFO", logger="rxembed"):
        rx.embed(str(xyz), n=1)
    assert any("relaxed into its windows" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO", logger="rxembed"):
        rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert not any("relaxed into its windows" in record.message for record in caplog.records)


def test_lazy_haptic_radius_uses_the_source_geometry():
    source = ferrocene()
    expected = rx.metal(Chem.Mol(source))[0].cons.pulls
    deferred = rx.metal(source)[0]
    deferred.mol.GetConformer().SetAtomPosition(1, Point3D(20, 20, 20))
    assert deferred.cons.pulls == expected


def test_haptic_centroids_follow_atoms_added_after_enumeration():
    mol = rx.parse_smiles(r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1")
    candidate = rx.enumerate_isomers(mol, "SPL")[0]
    candidate.mol = Chem.AddHs(candidate.mol)
    assert set(candidate.cons.haptic) == set(candidate.haptic) == {candidate.mol.GetNumAtoms()}
    assert rx.embed(candidate, n=1, seed=1).ids


def test_coordination_reads_hydrogens_added_after_enumeration():
    mol = rx.parse_smiles("N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    late = rx.enumerate_isomers(mol, "SPL", stereo="free")[0]
    late.mol = Chem.AddHs(late.mol)
    early = rx.enumerate_isomers(Chem.AddHs(mol), "SPL", stereo="free")[0]
    assert Chem.MolToSmiles(late.mol) == Chem.MolToSmiles(early.mol)
    assert late.cons == early.cons


def test_haptic_face_seats_from_any_ring_atom():
    mol = ferrocene()
    rings = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(metal_core.metal_index(mol)).GetNeighbors()]
    candidate = isomer.Isomer(mol, "LIN", {0: rings[0], 1: rings[-1]})
    assert len(candidate.haptic) == 2
    assert candidate.vertices == sorted(candidate.haptic)
    assert rx.embed(candidate, n=2, seed=1).minimize().ids


def _square_planar_pt(placement):
    mol = Chem.AddHs(Chem.MolFromSmiles(_PT_A2B2))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom, Point3D(0.0, 0.0, 1.6))
    conf.SetAtomPosition(1, Point3D(0.0, 0.0, 0.0))
    for atom, (x, y) in placement.items():
        conf.SetAtomPosition(atom, Point3D(2.05 * x, 2.05 * y, 0.0))
    mol.AddConformer(conf)
    return mol


_TRANS = {0: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
_CIS = {0: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, -1)}


def _seating_is_real(candidate, positions):
    directions = poly.vertex_dirs(candidate.geometry)
    return [
        (
            candidate.vertices[i],
            candidate.vertices[j],
            _angle(positions, candidate.vertices[i], candidate.metal, candidate.vertices[j]),
        )
        for i in range(len(directions))
        for j in range(i + 1, len(directions))
        if poly._vertex_angle(directions[i], directions[j]) > slots.TRANS_ANGLE
    ]


@pytest.mark.parametrize(("placement", "expected"), [(_TRANS, "trans"), (_CIS, "cis")])
def test_retained_isomer_uses_named_slots(placement, expected):
    mol = _square_planar_pt(placement)
    retained = isomer.from_geometry(mol)
    assert retained.label == expected
    positions = mol.GetConformer().GetPositions()
    assert all(angle > 150.0 for _left, _right, angle in _seating_is_real(retained, positions))


def test_retained_isomer_captures_geometry_and_keeps_the_metal_umbrella():
    source = _square_planar_pt(_TRANS)
    reference = isomer.from_geometry(Chem.Mol(source)).cons
    retained = isomer.from_geometry(source)
    retained.mol.GetConformer().SetAtomPosition(retained.donors[0], Point3D(20, 20, 20))
    assert retained.cons.distances == reference.distances
    assert retained.cons.angles == reference.angles
    assert retained.cons.umbrellas


def _ideal_sphere(geometry, symbol, scramble):
    directions = poly.vertex_dirs(geometry)
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom("Re"))
    donors = [rw.AddAtom(Chem.Atom(symbol)) for _ in directions]
    for donor in donors:
        rw.AddBond(metal, donor, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal, Point3D(0, 0, 0))
    for atom, vertex in zip(donors, scramble, strict=True):
        direction = np.array(directions[vertex], float)
        conf.SetAtomPosition(atom, Point3D(*(direction / np.linalg.norm(direction) * 1.95)))
    mol.AddConformer(conf)
    return mol


def test_untabulated_shape_uses_polyhedron():
    geometry = "square_antiprism"
    mol = _ideal_sphere(geometry, "F", (5, 2, 7, 0, 3, 6, 1, 4))
    retained = isomer.from_geometry(mol)
    assert retained.geometry == geometry
    positions = mol.GetConformer().GetPositions()
    directions = poly.vertex_dirs(geometry)
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            expected = poly._vertex_angle(directions[i], directions[j])
            actual = _angle(positions, retained.vertices[i], retained.metal, retained.vertices[j])
            assert actual == pytest.approx(expected, abs=1.0)
