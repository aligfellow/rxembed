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


def test_collapsed_ester_1_3_fusion_is_caught(embed):
    """An ester O-C-O folded until its two oxygens fuse (~1.27 Å) is caught — the silent 1-3 gate hole.

    The two oxygens are a 1-3 pair across the carbonyl carbon, and every prior gate is blind to it *for that
    structural reason*: `clashes` excludes 1-3 pairs, `metrics.bonding_ok`'s fusion floor (~0.9 Å) sits below the
    fused distance, and `metrics.connectivity`'s `_MIN_TOPO` skips a topo-2 pair. So a phantom O-C-O ring shipped
    silently until `over_compression`. This asserts both directions: the fusion IS flagged, and (red-first) the
    three prior gates stay silent on that O...O pair.
    """
    from rdkit.Chem import rdMolTransforms

    from rxembed import metrics as met

    mol = embed("CC(=O)OC")  # methyl acetate — the case-4 coordinated-ester motif, metal-free
    cc = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and sum(nb.GetAtomicNum() == 8 for nb in a.GetNeighbors()) == 2
    )
    onb = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cc).GetNeighbors() if nb.GetAtomicNum() == 8]
    o_term = next(o for o in onb if mol.GetAtomWithIdx(o).GetDegree() == 1)  # swing the terminal =O, not the methyl
    o_est = next(o for o in onb if o != o_term)
    rdMolTransforms.SetAngleDeg(mol.GetConformer(0), o_est, cc, o_term, 56.0)  # fold O-C-O toward fusion
    pos = mol.GetConformer(0).GetPositions()
    assert float(np.linalg.norm(pos[o_term] - pos[o_est])) < 1.4, "the two oxygens must have fused to set the test"

    fused = [v for v in geom.check(mol, 0).violations if v.kind == "fusion"]
    assert fused, "the collapsed ester O...O fusion must be flagged"
    assert set(fused[0].atoms) >= {o_term, o_est}, "the flagged pair must be the two fused oxygens"

    # red-first: the three gates that were the only ones looking here all stay silent on the O...O pair
    assert not [v for v in geom.clashes(mol, pos) if set(v.atoms) >= {o_term, o_est}], "clashes excludes the 1-3 pair"
    assert met.bonding_ok(mol, 0), "bonding_ok's fusion floor sits below the fused O...O distance"
    formed, _ = met.connectivity(mol, 0)
    assert {o_term, o_est} not in [set(p) for p in formed], "connectivity skips the topo-2 O...O pair"


@pytest.mark.parametrize("smiles", ["C1CO1", "C1CC1", "C1CN1"])  # epoxide, cyclopropane, aziridine
def test_genuine_three_membered_ring_is_not_a_fusion(embed, smiles):
    """A REAL strained 3-ring (real ~60° angle, real A-C bond) is never read as a 1-3 fusion — the FP guard.

    The discriminator is the graph, not the angle: the ring's two terminal atoms ARE bonded, so the pair never
    enters the over-compression test. A check that flagged real epoxides would be worse than the hole it closes.
    """
    mol = embed(smiles)
    rep = geom.check(mol, 0)
    assert not [v for v in rep.violations if v.kind == "fusion"], rep.summary()
    assert rep.ok(), rep.summary()  # a clean strained ring passes the whole gate


@pytest.mark.parametrize("smiles", ["CC(=O)OC", "CC(=O)O", "C[N+](=O)[O-]", "CC(=O)C", "O=CN(C)C"])
def test_real_tight_1_3_pairs_are_not_flagged(embed, smiles):
    """Real ester / carboxylate / nitro / ketone / amide 1-3 pairs (~2.1-2.5 Å) sit far above bonding — clean.

    These are the tight-1-3 ligand motifs the fusion check must never false-positive on: their O...O / C...O 1-3
    separations are ~1.6x the covalent sum, well clear of the `d < r_cov sum` fusion floor.
    """
    mol = embed(smiles)
    rep = geom.check(mol, 0)
    assert not [v for v in rep.violations if v.kind == "fusion"], rep.summary()


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
