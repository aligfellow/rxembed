"""Test the TS-aware and metal-aware geometry gate."""

from importlib.util import find_spec
from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdDistGeom, rdForceFieldHelpers, rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_distance as mdist
from rxembed import metal_perceive
from rxembed.constraints import Constraints
from rxembed.metal_core import COORDINATION_METALS
from rxembed.pipeline import geom_check as geom
from rxembed.utils import conjugated_quartets
from tests.conftest import EXAMPLES_DIR

_DFT = (
    pytest.param("mn-h2", {"metal_charges": {0: 2, 1: 1}}, id="mn-h2"),
)  # the shipped transition states the floors must accept unchanged
_ACID_ARENE = "OC(=O)CCCCc1ccccc1"  # flexible acid + arene: the fixture for both kwarg-driven checks


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


def _reference_conformer(smiles, seed=1, optimize=True):
    """Return a Mol with one conformer from plain ETKDG (+ MMFF): a geometry rxembed had no hand in making."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


def _shift(mol, idx, delta):
    conf = mol.GetConformer(0)
    p = conf.GetAtomPosition(int(idx))
    conf.SetAtomPosition(int(idx), Point3D(p.x + delta[0], p.y + delta[1], p.z + delta[2]))
    return mol


def _kinds(report):
    return {v.kind for v in report.violations}


# --- a clean embed passes ---------------------------------------------------------------------------------


def test_geometry_check_rejects_nonfinite_coordinates():
    mol = _reference_conformer("CCO")
    mol.GetConformer().SetAtomPosition(0, (float("nan"), 0.0, 0.0))

    report = geom.check(mol, 0)

    assert _kinds(report) == {"coordinates"}
    assert "non-finite coordinate" in report.summary()


@pytest.mark.parametrize(
    "smiles",
    [
        pytest.param("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", id="ammine"),
        pytest.param("O=[N+](=CC)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]", id="cumulated-nitro"),
    ],
)
def test_geometry_check_types_ligands_once(smiles, monkeypatch):
    iso = rx.metal(smiles, "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=1, seed=42, threads=1)
    mol = ens.mol
    cid = ens.ids[0]
    orientation = metal_perceive.donor_orientation(mol, mol.GetConformer(cid).GetPositions(), iso.donors)
    infer = Mock(wraps=metal_perceive.stripped_hybridisation)
    monkeypatch.setattr(geom, "stripped_hybridisation", infer)
    monkeypatch.setattr(metal_perceive, "stripped_hybridisation", infer)
    report = ens.check()[cid]

    assert infer.call_count == 1
    assert [v for v in report.violations if v.kind == "donor_orientation"] == orientation


def test_eta2_planarity_flex_is_metal_local():
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CC=C"))  # butadiene, no metal -> no eta2 flex
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    ci = next(a.GetIdx() for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    pos = conf.GetPositions()
    pos[ci] = pos[ci] + [0.0, 0.0, 0.35]  # shove one sp2 carbon 0.35 A out of plane (past the 0.15 default)
    conf.SetPositions(pos)

    assert any(v.kind == "planarity" for v in geom.planarity(mol, pos)), "non-metal sp2 wrongly flexed"


@pytest.mark.parametrize(
    "owner", ["donor-angle", "ligand-improper", "releasable-angle", "unrestricted-angle", "fixed-angle", "scalar-angle"]
)
def test_aromatic_attachment_planarity_respects_geometry_ownership(owner):
    iso = rx.metal("c1cc[cH](->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cc1", "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=1, seed=42, threads=1)
    mol = ens.mol
    cid = ens.ids[0]
    conf = mol.GetConformer(cid)
    donor = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "C")
    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors() if n.GetIdx() != iso.metal]
    hydrogen = next(i for i in neighbors if mol.GetAtomWithIdx(i).GetAtomicNum() == 1)
    pos = conf.GetPositions()
    ring = next(r for r in mol.GetRingInfo().AtomRings() if donor in r)
    normal = np.linalg.svd(pos[list(ring)] - pos[list(ring)].mean(axis=0))[2][-1]
    direction = pos[hydrogen] - pos[donor]
    length = np.linalg.norm(direction)
    direction += length * normal
    pos[hydrogen] = pos[donor] + direction / np.linalg.norm(direction) * length
    conf.SetPositions(pos)  # pucker the donor's ligand-side improper while keeping its ring and C-H length
    assert any(v.kind == "planarity" and donor in v.atoms for v in geom.check(mol, cid, donors=iso.donors).violations)
    if owner != "ligand-improper":
        key = (iso.metal, donor, hydrogen)
        angle = rdMolTransforms.GetAngleDeg(conf, *key)
        window = (0.0, 180.0) if owner == "unrestricted-angle" else (angle, angle)
        contacts = frozenset({key[::-1]}) if owner in {"releasable-angle", "fixed-angle"} else frozenset()
        constraints = Constraints(
            angles={key: window},
            contacts=(frozenset(), contacts),
            fixed={key: window} if owner == "fixed-angle" else {},
        )
        if owner == "scalar-angle":
            constraints = {"angles": {key: angle}}
    else:
        key = (*neighbors, donor)
        angle = rdMolTransforms.GetDihedralDeg(conf, *key)
        constraints = Constraints(dihedrals={key: (angle, angle)})
    report = geom.check(mol, cid, donors=iso.donors, constraints=constraints)
    assert ("planarity" in _kinds(report)) == (owner in {"releasable-angle", "unrestricted-angle"})
    assert "constraint" not in _kinds(report)


def test_declared_side_on_pair_keeps_its_window_at_an_early_metal_distance():
    """A declared eta2 N=N at 2.7 A from Ti is side-on; the 2.6 A cap guards only a perceived sphere."""
    mol = _reference_conformer("C=C/N=N/C")
    a, c, x, s = next(conjugated_quartets(mol))  # C=C-N=N: its twist is the side-on window's question
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), a, c, x, s, 135.0)  # 45 deg off plane: past 30, within 60
    rw = Chem.RWMol(mol)
    ti = rw.AddAtom(Chem.Atom("Ti"))
    pos = rw.GetConformer().GetPositions()
    normal = np.cross(pos[s] - pos[x], pos[c] - pos[x])
    height = (2.7**2 - (np.linalg.norm(pos[s] - pos[x]) / 2) ** 2) ** 0.5
    rw.GetConformer().SetAtomPosition(ti, ((pos[x] + pos[s]) / 2 + height * normal / np.linalg.norm(normal)).tolist())

    assert "conjugation" not in _kinds(geom.check(rw.GetMol(), 0, donors=[x, s]))
    assert "conjugation" in _kinds(geom.check(rw.GetMol(), 0)), "a perceived pair beyond the cap is not side-on"


# --- one deliberate break per violation kind --------------------------------------------------------------


def _planarity():
    mol = _reference_conformer("CC(=O)Nc1ccccc1")
    return _shift(mol, mol.GetRingInfo().AtomRings()[0][0], (0, 0, 0.8)), {}


def _conjugation():
    mol = _reference_conformer("CC(=O)NC")
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), a, b, c, d, 90.0)
    return mol, {}


def _bond_length():
    return _shift(_reference_conformer("CCO"), 0, (2.0, 0, 0)), {}


def _clash():
    mol = _reference_conformer("CCCCCCCC")
    mol.GetConformer(0).SetAtomPosition(7, mol.GetConformer(0).GetAtomPosition(0))
    return mol, {}


def _hydrogen():
    mol = _reference_conformer("CO")
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    return _shift(mol, h, (1.5, 0, 0)), {}


def _frozen_core():
    ref = _reference_conformer(_ACID_ARENE)
    return _shift(Chem.Mol(ref), 0, (0.5, 0, 0)), {"frozen": list(range(6)), "reference": ref}


def _constraint():
    return _reference_conformer(_ACID_ARENE), {"constraints": {"distances": {(1, 9): (0.5, 0.6)}}}


_BREAKS = {
    "planarity": _planarity,
    "conjugation": _conjugation,
    "bond_length": _bond_length,
    "clash": _clash,
    "hydrogen": _hydrogen,
    "frozen_core": _frozen_core,
    "constraint": _constraint,
}


@pytest.mark.parametrize("kind", ["clash", "hydrogen", "frozen_core"])
def test_each_violation_kind_fires(kind):
    mol, kw = _BREAKS[kind]()
    assert kind in _kinds(geom.check(mol, 0, **kw))


def _alanine_and_mirror():
    mol = _reference_conformer("C[C@@H](C(=O)O)N")
    mirror = Chem.Mol(mol)
    mirror.GetConformer().SetPositions(-mirror.GetConformer().GetPositions())
    return mol, mirror


def test_xyz_path_reference_still_checks_stereo(tmp_path):
    mol, mirror = _alanine_and_mirror()
    path = str(tmp_path / "alanine.xyz")
    Chem.MolToXYZFile(mol, path)

    assert "stereo" in _kinds(geom.check(mirror, 0, reference=path))


def test_kwarg_checks_accept_matching_geometry():
    ref = _reference_conformer(_ACID_ARENE)
    d = float(np.linalg.norm(ref.GetConformer(0).GetPositions()[1] - ref.GetConformer(0).GetPositions()[9]))
    assert geom.check(ref, 0, frozen=list(range(6)), reference=ref).ok(), "an identical geometry moved the core"
    assert geom.check(ref, 0, constraints={"distances": {(1, 9): (d - 0.1, d + 0.1)}}).ok()


def test_constraint_gate_checks_periodic_dihedrals():
    mol = _reference_conformer("CCCC")
    pos = mol.GetConformer().GetPositions()
    atoms = (0, 1, 2, 3)
    phi = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *atoms)

    missed = geom.check_constraints(mol, pos, {"dihedrals": {atoms: (phi + 50.0, phi + 70.0)}})
    equivalent = geom.check_constraints(mol, pos, {"dihedrals": {atoms: (phi + 350.0, phi + 370.0)}})

    assert [violation.atoms for violation in missed] == [atoms]
    assert not equivalent


# --- TS-awareness: a held core is not judged by ground-state rules ----------------------------------------


# --- the 1-3 fusion gate, and the strained rings it must not eat -------------------------------------------


# --- the metal arm: the over-bond ruler the gate reads ------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(("name", "read_kw"), _DFT)
def test_floors_accept_reference_geometries(name, read_kw):
    pt = GetPeriodicTable()
    mol = rx.read_xyz(str(EXAMPLES_DIR / f"{name}.xyz"), 0, **read_kw)
    pos = mol.GetConformer().GetPositions()
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]
    assert metals, f"{name} carries no transition metal: the fixture exercises no floor at all"

    for m in metals:
        donors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors())
        cons = Constraints()
        mdist.nondonor_floors(mol, m, mol.GetAtomWithIdx(m).GetAtomicNum(), donors, cons)
        assert cons.floors, f"metal {m} got no floors at all"
        for (i, j), floor in cons.floors.items():
            x = i if j == m else j
            assert float(np.linalg.norm(pos[x] - pos[m])) >= floor, (
                f"floor rejects {mol.GetAtomWithIdx(x).GetSymbol()}{x}"
            )
        rm = pt.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
        for d in donors:
            rs = rm + pt.GetRcovalent(mol.GetAtomWithIdx(d).GetAtomicNum())
            assert float(np.linalg.norm(pos[d] - pos[m])) / rs > mdist.DONOR_COLLAPSE_RATIO
        assert not metal_perceive.metal_overbond(mol, pos, donors), "the gate rejects a real DFT geometry"


# --- a declared donor is a donor, whatever its element ------------------------------------------------------


def _ruthenium(d_ruh=1.701, d_rucl=2.233):
    """DUKPII: a terminal hydride and a terminal chloride on one Ru, both bond-less after the surrogate's strip."""
    return _bare_sphere(
        ["Ru", "H", "Cl", "P", "P"],
        [],
        [(0, 0, 0), (d_ruh, 0, 0), (-d_rucl, 0, 0), (0, 2.341, 0), (0, -2.341, 0)],
    )


@pytest.mark.parametrize("kind", ["hydride"])
def test_buried_donor_triggers_metal_collapse(kind):
    at = 1 if kind == "hydride" else 2
    mol, pos = _ruthenium(**{"d_ruh" if kind == "hydride" else "d_rucl": 0.100})
    v = metal_perceive.metal_overbond(mol, pos, [1, 2, 3, 4])
    assert [x.kind for x in v] == ["metal_collapse"]
    assert v[0].atoms == (0, at)
    assert not geom.check(mol, mol.GetConformer().GetId(), donors=[1, 2, 3, 4]).ok()


# --- the two FF caps, seen through the gate on real complexes ------------------------------------------------
