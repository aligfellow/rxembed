"""Test graph and coordination-sphere changes after relaxation."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed.pipeline import metrics

# every check here re-perceives the graph with xyzgraph
pytestmark = pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")


def _bare_sphere(symbols, bonds, coords):
    """Return ``(mol, positions)``: a metal + ligand skeleton exactly as the surrogate leaves it.

    No sanitisation and no implicit H, so a monatomic donor really is bond-less; on the real path the M-donor
    bonds have already been stripped, and that is what makes the metal gates readable at all.
    """
    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))
    for i, j in bonds:
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for a in mol.GetAtoms():
        a.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*p))
    mol.AddConformer(conf)
    return mol, mol.GetConformer().GetPositions()


# --- connectivity: the graph diff -------------------------------------------------------------------------


def test_reperception_finds_a_terminal_hydrogen_collapsed_across_an_angle():
    # The bare H sits closer to atom 1 (0.90 A) than to its nominal bond partner atom 0 (1.09 A): real
    # geometry-driven reperception must find the new C1-H contact on its own, with no stubbed _perceive.
    mol, _pos = _bare_sphere(
        ["C", "C", "H"],
        [(0, 1), (0, 2)],
        [(0, 0, 0), (1.68, 0, 0), (0.95, 0.53, 0)],
    )

    formed, _broken = metrics.connectivity(mol, 0)

    assert formed == [(1, 2)]
    assert metrics.connectivity(mol, 0, exclude={1, 2}) == ([], [])


# --- coordination_changed: the metal's own diff -----------------------------------------------------------


def test_stated_metal_distance_outranks_the_generic_donor_cutoff():
    mol, _pos = _bare_sphere(["Zn", "O"], [], [(0, 0, 0), (2.52, 0, 0)])

    assert metrics.coordination_changed(mol, 0, 0, [1]) == ([1], [])
    assert metrics.coordination_changed(mol, 0, 0, [1], constrained={(0, 1): (2.42, 2.62)}) == ([], [])


def test_undeclared_agostic_h_neither_joins_nor_leaves():
    mol, _pos = _bare_sphere(
        ["Ru", "C", "H", "P", "P"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (1.85, 0, 0.9), (0, 2.341, 0), (0, -2.341, 0)],
    )
    assert metrics.coordination_changed(mol, mol.GetConformer().GetId(), 0, [1, 3, 4]) == ([], [])


def test_collapsed_alpha_carbon_is_reported_joined():
    mol, _pos = _bare_sphere(
        ["Pd", "N", "C", "C", "Cl", "Cl"],
        [(1, 2), (2, 3)],
        [(0, 0, 0), (2.101, 0, 0), (1.502, 1.577, 0), (2.9, 2.6, 0), (0, 2.385, 0), (0, -2.385, 0)],
    )
    assert metrics.coordination_changed(mol, mol.GetConformer().GetId(), 0, [1, 4, 5])[1] == [2]

    # ...and a real beta-agostic Ti-CH2-CH3 at the crystal's 2.554 A must not be (Dawoodi/Green, Dalton 1986)
    ti, tpos = _bare_sphere(
        ["Ti", "C", "C", "Cl", "Cl", "Cl"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (2.038, 1.539, 0), (-2.2, 0, 0), (0, -2.2, 0), (0, 0, 2.2)],
    )
    assert np.linalg.norm(tpos[2] - tpos[0]) == pytest.approx(2.554, abs=0.005)  # the fixture IS the crystal
    assert metrics.coordination_changed(ti, ti.GetConformer().GetId(), 0, [1, 3, 4, 5])[1] == []
