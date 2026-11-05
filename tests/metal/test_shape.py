"""`metal/shape.py`: which polyhedron a sphere reads as, and which atoms collapsed onto a metal or folded a donor."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed.metal import shape
from rxembed.metal.polyhedra import POLYHEDRA, VACANT
from tests.metal_fixtures import SPY_TBP_TIE, SQUASHED_SIX, ideal_sphere


def _placed(smiles, points):
    mol = rx.parse_smiles(smiles)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetPositions(np.array(points, float))
    mol.AddConformer(conformer)
    return mol


def test_metal_overbond_floors_each_atom_by_its_tier():
    """A declared donor is floored only at collapse, a non-donor by how many donors it bonds, and a bite apex not at
    all; an undeclared metal reads its own shell, and a declared donor that left the shell leaves an atom on the
    metal no donor."""
    folded = _placed(  # the amine C folded 2.0 A from Pd, under its one-donor floor; Cl3 crushed to 0.9 A
        "C[NH2]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]",
        [(1.6, 1.2, 0), (2.05, 0, 0), (0, 0, 0), (0, 0.9, 0), (-2.3, 0, 0), (0, -2.3, 0)],
    )
    dimer = Chem.RWMol()  # Pd0 and Pd1 bridged by Cl2 and Cl3, C8 crushed onto Pd0; only Pd0 declares its donors
    for z in (46, 46, 17, 17, 17, 17, 17, 17, 6):
        dimer.AddAtom(Chem.Atom(z))
    dimer_pos = np.array(
        [
            (-1.75, 0, 0),
            (1.75, 0, 0),
            (0, 1.65, 0),
            (0, -1.65, 0),
            (-3.4, 1.6, 0),
            (-3.4, -1.6, 0),
            (3.4, 1.6, 0),
            (3.4, -1.6, 0),
            (-1.75, 0, 2.0),
        ]
    )
    left = _placed(  # every chloride 3.5 A out, a water O 2.0 A above the Pt
        "[Cl-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-].O",
        [(3.5, 0, 0), (0, 0, 0), (-3.5, 0, 0), (0, 3.5, 0), (0, -3.5, 0), (0, 0, 2)],
    )
    acetate = _placed(  # a kappa2 acetate's C, bonded to both O donors, sits 2.45 A from Pd at the bite's apex
        "CC1=[O]->[Pd+2](<-[Cl-])(<-[Cl-])<-[O-]1",
        [(3.95, 0, 0), (2.45, 0, 0), (1.72, 1.12, 0), (0, 0, 0), (-1.63, 1.63, 0), (-1.63, -1.63, 0), (1.72, -1.12, 0)],
    )
    cases = {
        "folded amine": (folded, folded.GetConformer().GetPositions(), [1, 3, 4, 5]),
        "Pd2 declaring one": (dimer.GetMol(), dimer_pos, {0: {2, 3, 4, 5}}),
        "PtCl4 chlorides left": (left, left.GetConformer().GetPositions(), {1: {0, 2, 3, 4}}),
        "acetate apex": (acetate, acetate.GetConformer().GetPositions(), [2, 4, 5, 6]),
    }

    found = {name: [(v.kind, v.atoms) for v in shape.metal_overbond(*case)] for name, case in cases.items()}

    assert found == {
        "folded amine": [("metal_overbond", (2, 0)), ("metal_collapse", (2, 3))],
        "Pd2 declaring one": [("metal_overbond", (0, 8))],
        "PtCl4 chlorides left": [("metal_overbond", (1, 5))],
        "acetate apex": [],
    }


def _palladium_at(smiles, angle, donor):
    """Return `smiles` embedded with a Pd 2.1 A from its first `donor` atom at `angle` deg from that donor's carbon."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    d = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == donor)
    c = next(atom.GetIdx() for atom in mol.GetAtomWithIdx(d).GetNeighbors() if atom.GetAtomicNum() == 6)
    rw = Chem.RWMol(mol)
    pd = rw.AddAtom(Chem.Atom(46))
    rw.AddBond(d, pd, Chem.BondType.DATIVE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    pos = np.vstack([mol.GetConformer().GetPositions(), np.zeros(3)])
    u = (pos[c] - pos[d]) / np.linalg.norm(pos[c] - pos[d])
    w = np.cross(u, (0.0, 0.0, 1.0))
    t = np.radians(angle)
    pos[pd] = pos[d] + 2.1 * (np.cos(t) * u + np.sin(t) * w / np.linalg.norm(w))
    return out, pos, (pd, d, c)


def test_donor_orientation_flags_only_a_folded_donor():
    """An amine folded to 80 deg is flagged unless its M-N-C is grafted; splayed to 160 deg, or an ether, is not."""
    folded, pos, atoms = _palladium_at("CN", 80.0, 7)
    (violation,) = shape.donor_orientation(folded, pos, [atoms[1]])
    assert (violation.kind, violation.atoms, violation.value) == ("donor_orientation", atoms, pytest.approx(80.0))
    assert not shape.donor_orientation(folded, pos, [atoms[1]], frozenset(atoms))
    unfolded = [_palladium_at(smiles, angle, donor) for smiles, angle, donor in (("CN", 160.0, 7), ("COC", 60.0, 8))]
    assert [shape.donor_orientation(mol, pos, [atoms[1]]) for mol, pos, atoms in unfolded] == [[], []]


_FLAT_PYRAMID = np.array(POLYHEDRA["trigonal_pyramidal"].vertex_dirs) * (1, 1, 0)


def _donor_on_the_metal():
    """Return an ideal octahedron whose first donor sits on the metal, so it points nowhere."""
    mol = ideal_sphere(POLYHEDRA["octahedral"].vertex_dirs, 2.1)
    mol.GetConformer().SetAtomPosition(1, (0.0, 0.0, 0.0))
    return mol


# id -> (the sphere, its vertices over the sites, the requested polyhedron)
SPHERES = {
    "octahedron": (ideal_sphere(POLYHEDRA["octahedral"].vertex_dirs, 2.1), [1, 2, 3, 4, 5, 6], "octahedral"),
    "bipyramid": (
        ideal_sphere(POLYHEDRA["pentagonal_bipyramidal"].vertex_dirs, 2.1),
        [1, 2, 3, 4, 5, 6, 7],
        "pentagonal_bipyramidal",
    ),
    "vacancy": (ideal_sphere(POLYHEDRA["square_planar"].vertex_dirs[:3], 2.1), [1, 2, 3, VACANT], "square_planar"),
    "flat pyramid": (ideal_sphere(_FLAT_PYRAMID, 2.1), [1, 2, 3], "trigonal_pyramidal"),
    "squashed six": (ideal_sphere(SQUASHED_SIX, 2.1), [1, 2, 3, 4, 5, 6], "hexagonal_planar"),
    "donor on metal": (_donor_on_the_metal(), [1, 2, 3, 4, 5, 6], "octahedral"),
}


def test_shape_gap_reads_each_sphere_against_its_request():
    """Seven sites rank upper bounds beside the bipyramid's exact fit and say so; a vacancy ranks the occupied
    subset."""
    readings = {
        name: shape.shape_gap(mol, 0, vertices, {}, requested) for name, (mol, vertices, requested) in SPHERES.items()
    }

    assert {name: (r.verdict, r.matches, r.accepted, r.exact) for name, r in readings.items()} == {
        "octahedron": ("fits", True, True, True),
        "bipyramid": ("fits", True, True, False),
        "vacancy": ("fits", True, True, True),
        "flat pyramid": ("other_shape", False, False, True),
        "squashed six": ("distorted", True, False, True),
        "donor on metal": ("unreadable", False, False, True),
    }
    assert readings["flat pyramid"].clause("TPY").startswith("TPY 0.338 (next TPL 0.000)")
    assert readings["bipyramid"].clause("PBP").endswith(", approximate")
    assert readings["donor on metal"].clause("octahedral") == "OCT unreadable"


def test_classify_geometry_names_the_least_residual():
    """A tie is named as one, and an unreadable sphere reads as no polyhedron."""
    tie = shape.classify_geometry(ideal_sphere(SPY_TBP_TIE, 2.1), 0, [1, 2, 3, 4, 5], {})

    assert (tie.name, tie.reading.next_name, tie.tied) == ("trigonal_bipyramidal", "square_pyramidal", True)
    assert shape.classify_geometry(_donor_on_the_metal(), 0, [1, 2, 3, 4, 5, 6], {}) is None


def test_geometry_for_takes_the_default_polyhedron():
    """An apical face avoids right angles."""
    assert [shape.geometry_for(n) for n in (2, 4, 6, 12)] == ["linear", "square_planar", "octahedral", None]
    assert shape.geometry_for(4, has_apical=True) == "tetrahedral"  # a piano stool
