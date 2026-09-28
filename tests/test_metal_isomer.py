"""Test selected metal states, collections, and retained geometry."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_isomer as isomer
from rxembed import metal_polyhedron as poly
from rxembed import stereo
from rxembed.metal_core import MetalState

_MA2B2 = "CCCN[Pd](Cl)(Cl)NCCC"


def test_summary_details_show_site_relations_and_non_equivalent_donors(capsys):
    # Two carbon donors (a carbonyl and a methanide) must be disambiguated by index, not merged as one symbol.
    octahedral = rx.metal("[O+]#[C-]->[Co+3](<-[CH3-])(<-[Cl-])(<-N)(<-[F-])<-P", "OCT", stereo="free")[0]
    octahedral.summary(details=True)
    shown = capsys.readouterr().out
    assert "trans: C1-C3" in shown

    tbp = rx.metal("[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O", "TBP", stereo="free")[0]
    tbp.summary(details=True)
    shown = capsys.readouterr().out
    assert "axial:" in shown
    assert "equatorial:" in shown


def test_select_accepts_geometry_code_or_name():
    isomers = rx.metal(_MA2B2, "square_planar")
    first = isomers[0]
    assert isomers.select(arrangement=first.arrangement).vertices == first.vertices
    assert isomers.select(index=0).vertices == first.vertices
    with pytest.raises(ValueError, match="matched"):
        isomers.select(arrangement="does not exist")
    tetrahedral = rx.metal("[Zn](F)(Cl)(Br)I", "TET")
    assert {candidate.geometry for candidate in tetrahedral} == {"tetrahedral"}
    assert tetrahedral.select(hand="delta") is tetrahedral.select(hand="Δ")


def test_shape_only_summary_lists_every_restored_metal(capsys):
    candidate = isomer.Isomer.from_state(Chem.MolFromSmiles("[C].[C]"), (MetalState(0, 26, 2), MetalState(1, 25, 0)))
    candidates = isomer.IsomerSet([candidate])
    assert candidates.filter() == [candidate]
    assert candidates.select() is candidate
    candidate.summary(details=True)
    shown = capsys.readouterr().out
    assert "Fe0" in shown
    assert "Mn1" in shown
    assert shown.count("surrogated") == 2, "each unresolved metal must be marked, not just named"
    assert "Constraints(" not in repr(candidate)


def test_retained_imine_chelate_does_not_add_independent_ez():
    candidate = rx.metal(r"C/C=C/C/N1=C(/F)C(/Cl)=N(/Br)->[Ni+2](<-[Cl-])(<-[I-])<-1", "SPL")[0]
    source = rx.embed(candidate, n=1, seed=7, threads=1).mol
    locked = stereo.coordination_locked_double_bonds(source, {candidate.metal})
    assert locked
    pendant = stereo.bond_stereo(candidate.stereo_label)
    assert pendant
    positions = source.GetConformer().GetPositions()
    tags = [bond.GetStereo() for bond in source.GetBonds()]

    retained = isomer.from_geometry(source)

    assert locked.isdisjoint(stereo.bond_stereo(retained.stereo_label))
    assert stereo.bond_stereo(retained.stereo_label) == pendant
    assert all(retained.mol.GetBondBetweenAtoms(*pair).GetStereo() == Chem.BondStereo.STEREONONE for pair in locked)
    assert rx.cxsmiles(retained) == rx.cxsmiles(candidate)
    np.testing.assert_array_equal(retained.mol.GetConformer().GetPositions(), positions)
    assert [bond.GetStereo() for bond in source.GetBonds()] == tags
    np.testing.assert_array_equal(source.GetConformer().GetPositions(), positions)


def test_from_geometry_names_an_unsupported_coordination_number():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom("La"))
    donors = [rw.AddAtom(Chem.Atom("F")) for _ in range(13)]
    for donor in donors:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal, Point3D(0, 0, 0))
    for i, donor in enumerate(donors):
        angle = 2 * np.pi * i / len(donors)
        conf.SetAtomPosition(donor, Point3D(float(2 * np.cos(angle)), float(2 * np.sin(angle)), 0.3 * (i % 3 - 1)))
    mol.AddConformer(conf)

    with pytest.raises(ValueError, match=r"no polyhedron template for '13-coordinate'.*POLYHEDRA"):
        isomer.from_geometry(mol)


def test_from_geometry_rejects_an_unbound_metal_cleanly():
    mol = Chem.MolFromSmiles("[Hg].[C-]#[O+]")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    with pytest.raises(ValueError, match=r"Hg0.*no donor bonds"):
        isomer.from_geometry(mol)


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


@pytest.mark.parametrize("geometry", ["monocoordinate"])
def test_distorted_source_and_stated_isomer_share_polyhedron_angles(geometry):
    directions = poly.vertex_dirs(geometry)
    mol = _ideal_sphere(geometry, "F", range(len(directions)))
    positions = mol.GetConformer().GetPositions()
    positions[1:] *= (1.04, 0.98, 1.0)
    mol.GetConformer().SetPositions(positions)
    retained = isomer.from_geometry(mol)
    assert retained.geometry == geometry
    stated = isomer.Isomer(mol, geometry, retained.vertices, lengths="model")
    assert retained.cons == stated.cons
    measured = isomer.from_geometry(mol, lengths="input")
    assert measured.cons.angles == retained.cons.angles
    for donor in retained.donors:
        distance = np.linalg.norm(positions[donor] - positions[retained.metal])
        window = measured.cons.distances[tuple(sorted((retained.metal, donor)))]
        assert 0.5 * sum(window) == pytest.approx(distance)
