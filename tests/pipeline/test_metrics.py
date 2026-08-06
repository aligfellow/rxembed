"""Test graph and coordination-sphere changes after relaxation."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolTransforms
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


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(m)
    return m


def _move(conf, idx, target):
    conf.SetAtomPosition(int(idx), [float(x) for x in target])


def _push_out(conf, atom, centre, distance):
    """Place `atom` at `distance` Å from `centre` along their own axis: a dissociation, made deliberately."""
    p, pc = np.array(conf.GetAtomPosition(int(atom))), np.array(conf.GetAtomPosition(int(centre)))
    _move(conf, atom, pc + distance * (p - pc) / np.linalg.norm(p - pc))


def _ruthenium(d_ruh=1.701, d_rucl=2.233):
    """DUKPII's shape: a terminal hydride and a terminal chloride on one Ru, both bond-less after the strip.

    The controlled pair; structurally identical, differing only in atomic number. Whatever the checks say of
    the chloride they must say of the hydride.
    """
    return _bare_sphere(
        ["Ru", "H", "Cl", "P", "P"],
        [],
        [(0, 0, 0), (d_ruh, 0, 0), (-d_rucl, 0, 0), (0, 2.341, 0), (0, -2.341, 0)],
    )


# --- connectivity: the graph diff -------------------------------------------------------------------------


def test_clean_conformer_reperceives_original_graph():
    m = _mol("CC(=O)Nc1ccccc1")
    assert metrics.connectivity(m, m.GetConformers()[0].GetId()) == ([], [])


def test_frozen_core_exempts_stretched_bond():
    m = _mol("CCO")
    cid = m.GetConformer().GetId()
    rdMolTransforms.SetBondLength(m.GetConformer(), 1, 2, 2.40)  # ~1.7x the C-O covalent sum, fragment and all
    _formed, broken = metrics.connectivity(m, cid)
    assert {1, 2} in [set(p) for p in broken], f"a 2.40 A C-O was not reported broken: {broken}"
    assert metrics.connectivity(m, cid, exclude={1, 2}) == ([], [])


def test_reperception_finds_transferred_proton():
    m = _mol("[NH3+]CC(=O)[O-]")
    conf = m.GetConformer()
    n = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N")
    o = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    h = next(x.GetIdx() for x in m.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 1)
    c = next(x.GetIdx() for x in m.GetAtomWithIdx(o).GetNeighbors() if x.GetAtomicNum() == 6)
    po, pc = np.array(conf.GetAtomPosition(o)), np.array(conf.GetAtomPosition(c))
    _move(conf, h, po + 0.98 * (po - pc) / np.linalg.norm(po - pc))  # a real O-H length: a transfer, not an H-bond

    formed, broken = metrics.connectivity(m, conf.GetId())
    assert {o, h} in [set(p) for p in formed]
    assert {n, h} in [set(p) for p in broken]
    assert metrics.bonding_ok(m, conf.GetId()), "bonding_ok is heavy-atom-only: it CANNOT see this"


def test_connectivity_ignores_metal_pairs():
    import rxembed as rx

    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    ens = rx.embed(iso, n=2, seed=1).minimize()
    assert ens.ids, "no conformer survived: the covalent diff was never asked about a metal pair"
    for cid in ens.ids:
        assert metrics.connectivity(ens.mol, cid, metals={iso.metal}, elements={iso.metal: iso.real_z}) == ([], [])


# --- coordination_changed: the metal's own diff -----------------------------------------------------------


def test_metal_donor_departure_is_reported():
    import rxembed as rx

    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    cid = ens.ids[0]
    assert metrics.coordination_changed(ens.mol, cid, iso.metal, iso.donors) == ([], [])

    _push_out(ens._mol.GetConformer(cid), iso.donors[0], iso.metal, 4.0)
    left, _joined = metrics.coordination_changed(ens._mol, cid, iso.metal, iso.donors)
    assert iso.donors[0] in left
    frozen = {iso.metal, iso.donors[0]}
    assert metrics.coordination_changed(ens._mol, cid, iso.metal, iso.donors, exclude=frozen)[0] == []


def test_hydride_uses_dative_not_covalent_diff():
    held, _pos = _ruthenium()
    assert held.GetAtomWithIdx(1).GetDegree() == held.GetAtomWithIdx(2).GetDegree() == 0
    assert metrics.coordination_changed(held, held.GetConformer().GetId(), 0, [1, 2, 3, 4]) == ([], [])

    gone, _pos = _ruthenium(d_ruh=5.0)
    assert metrics.coordination_changed(gone, gone.GetConformer().GetId(), 0, [1, 2, 3, 4]) == ([1], [])


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
