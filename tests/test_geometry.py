"""The geometry gate — no bad geometries from (frozen) embeds.

The self-contained tests here drive `rxembed.geometry` directly on RDKit embeds, so they run today
without the rest of the package. The Phase-A scaffolds at the bottom (marked `skip`) wire the same gate
onto real `rx.embed(...)` frozen / constrained / template cases once the port lands.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed import geometry as geom

# molecules whose conjugation / H-geometry a bad embed would break
CLEAN = [
    "CCO",  # ethanol
    "OC(=O)CCCCc1ccccc1",  # flexible acid + arene
    "CC(=O)Nc1ccccc1",  # acetanilide — planar amide + arene
    "c1ccccc1/C=C/c1ccccc1",  # stilbene — extended conjugation
    "O=C(N)c1ccc(N)cc1",  # benzamide + aniline
]


@pytest.mark.parametrize("smiles", CLEAN)
def test_clean_embed_passes(embed, smiles):
    mol = embed(smiles)
    rep = geom.check(mol, 0)
    assert rep.ok(), rep.summary()


def test_puckered_ring_is_caught(embed):
    mol = embed("CC(=O)Nc1ccccc1")
    conf = mol.GetConformer(0)
    ring = mol.GetRingInfo().AtomRings()[0]
    p = conf.GetAtomPosition(ring[0])
    conf.SetAtomPosition(ring[0], Point3D(p.x, p.y, p.z + 0.8))  # shove one ring atom out of plane
    rep = geom.check(mol, 0)
    assert not rep.ok()
    assert any(v.kind == "planarity" for v in rep.violations)


def test_twisted_amide_broken_conjugation_is_caught(embed):
    from rdkit.Chem import rdMolTransforms

    mol = embed("CC(=O)NC")
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), a, b, c, d, 90.0)  # rotate amide out of plane
    rep = geom.check(mol, 0)
    assert any(v.kind == "conjugation" for v in rep.violations)


def test_hbond_not_flagged_as_clash(embed):
    # a carboxylic acid embeds with close polar contacts that must NOT read as steric clashes
    assert geom.check(embed("OC(=O)CCCCc1ccccc1"), 0).ok()


def test_frozen_core_excluded_from_ground_state_checks(embed):
    # a twisted amide is caught normally, but if those atoms are the frozen/reacting core they're
    # held to the reference by design and must NOT be flagged as ground-state violations (TS-aware gate)
    from rdkit.Chem import rdMolTransforms

    mol = embed("CC(=O)NC")
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), a, b, c, d, 90.0)
    assert any(v.kind == "conjugation" for v in geom.check(mol, 0).violations)  # caught by default
    assert not any(v.kind == "conjugation" for v in geom.check(mol, 0, frozen=(a, b, c, d)).violations)  # excluded


def test_stretched_bond_is_caught(embed):
    mol = embed("CCO")
    conf = mol.GetConformer(0)
    p = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(p.x + 2.0, p.y, p.z))  # yank atom 0 away from its bond
    rep = geom.check(mol, 0)
    assert any(v.kind == "bond_length" for v in rep.violations)


def test_clash_is_caught(embed):
    mol = embed("CCCCCCCC")
    conf = mol.GetConformer(0)
    q = conf.GetAtomPosition(0)
    conf.SetAtomPosition(7, Point3D(q.x, q.y, q.z))  # drop a far atom on top of atom 0
    rep = geom.check(mol, 0)
    assert any(v.kind == "clash" for v in rep.violations)


def test_bad_hydrogen_is_caught(embed):
    mol = embed("CO")
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    conf = mol.GetConformer(0)
    p = conf.GetAtomPosition(h)
    conf.SetAtomPosition(h, Point3D(p.x + 1.5, p.y, p.z))  # stretch an X-H far past range
    rep = geom.check(mol, 0)
    assert any(v.kind == "hydrogen" for v in rep.violations)


def test_frozen_core_rmsd(embed):
    ref = embed("OC(=O)CCCCc1ccccc1")
    mol = Chem.Mol(ref)
    frozen = list(range(6))
    # identical geometry -> zero core movement
    assert geom.check(mol, 0, frozen=frozen, reference=ref).ok()
    # displace a frozen atom -> caught
    conf = mol.GetConformer(0)
    p = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(p.x + 0.5, p.y, p.z))
    rep = geom.check(mol, 0, frozen=frozen, reference=ref)
    assert any(v.kind == "frozen_core" for v in rep.violations)


def test_constraint_window(embed):
    mol = embed("OC(=O)CCCCc1ccccc1")
    d = float(np.linalg.norm(mol.GetConformer(0).GetPositions()[1] - mol.GetConformer(0).GetPositions()[9]))
    # a window around the realised distance passes; a wrong one fails
    assert geom.check(mol, 0, constraints={"distances": {(1, 9): (d - 0.1, d + 0.1)}}).ok()
    rep = geom.check(mol, 0, constraints={"distances": {(1, 9): (0.5, 0.6)}})
    assert any(v.kind == "constraint" for v in rep.violations)


def test_report_is_falsy_and_asserts():
    rep = geom.GeometryReport([geom.Violation("clash", (0, 1), 0.5, 1.0)])
    assert not rep
    with pytest.raises(AssertionError):
        rep.assert_ok()


# The real frozen / constrained / templated embeds assert this SAME gate on rxembed's own embed() in
# test_embed_core.py (constrain windows), test_frozen.py (fix graft + template), and test_organic.py (NCI).
