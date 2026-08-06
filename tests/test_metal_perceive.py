"""Test coordination-sphere and donor-orientation perception."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_perceive as coord
from rxembed.metal_donor_orient import _FOLD_WINDOW, _stripped_hybridisation
from rxembed.pipeline import geom_check as geom

_FE_C, _C_O, _FE_H = 1.80, 1.13, 1.55  # Å: the recorded FeH2(CO)4 bond lengths
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]  # the 4 equatorial vertices; the 2 hydrides go on ±z
_STITCH = (0.0, 171.7, 171.8, 180.0)  # the fold set the rigid stitch shipped: one carbonyl flat on the metal


def feh2co4(folds=(180.0, 180.0, 180.0, 180.0)):
    """FeH2(CO)4 with its carbons pinned on octahedral vertices and each carbonyl bent to its Fe-C-O angle.

    Only the oxygens swing (180 points O straight out, 0 folds it onto the metal), so every distance-based
    check sees a textbook octahedron however far the ligands fold. Returns `(mol, donors)`.
    """
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(26))  # 0 = Fe, at the origin
    pos: list[np.ndarray] = [np.zeros(3)]
    donors: list[int] = []
    for ax, ang in zip(_OCT, folds, strict=True):
        u = np.array(ax, float)
        c, o = rw.AddAtom(Chem.Atom(6)), rw.AddAtom(Chem.Atom(8))
        rw.GetAtomWithIdx(c).SetFormalCharge(-1)  # the [C-]#[O+] carbonyl
        rw.GetAtomWithIdx(o).SetFormalCharge(1)
        rw.AddBond(c, o, Chem.BondType.TRIPLE)
        rw.AddBond(c, 0, Chem.BondType.DATIVE)  # C -> Fe
        t = np.radians(ang)
        pos += [_FE_C * u, _FE_C * u + _C_O * (np.cos(t) * (-u) + np.sin(t) * np.array([0.0, 0.0, 1.0]))]
        donors.append(c)
    for z in (1.0, -1.0):
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(0, h, Chem.BondType.SINGLE)
        pos.append(np.array([0.0, 0.0, z * _FE_H]))
        donors.append(h)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol.AddConformer(conf, assignId=True)
    return mol, donors


def _place(mol, pos):
    """Write `pos` into the mol's conformer; `donor_fold` reads the conformer, not the array."""
    conf = mol.GetConformer()
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    return mol


def _donor_at(smi, angle_deg, donor_num=7, pick=None):
    """A small molecule with a Pd placed at an exact M-donor-substituent angle, 2.1 Å from the donor."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    pick = pick or (lambda a: a.GetAtomicNum() == donor_num)
    n = next(a.GetIdx() for a in mol.GetAtoms() if pick(a))
    c = next(x.GetIdx() for x in mol.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 6)
    rw = Chem.RWMol(mol)
    pd = rw.AddAtom(Chem.Atom(46))
    rw.AddBond(n, pd, Chem.BondType.DATIVE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    pos = np.vstack([mol.GetConformer().GetPositions(), np.zeros(3)])
    u = pos[c] - pos[n]
    u /= np.linalg.norm(u)
    w = np.cross(u, [0.0, 0.0, 1.0])
    w /= np.linalg.norm(w)
    t = np.radians(angle_deg)
    pos[pd] = pos[n] + 2.1 * (np.cos(t) * u + np.sin(t) * w)
    assert geom._angle(pos[pd], pos[n], pos[c]) == pytest.approx(angle_deg, abs=0.5)
    return _place(out, pos), pos, n


def _sphere_of(ens):
    return sorted({int(d) for ds in ens.sphere.values() for d in ds})


# --- the gate is reachable, and reachable with the intended donors ---------------------------------------


def test_public_gate_receives_real_donors(monkeypatch):
    seen: list[object] = []
    real = coord.donor_orientation

    def spy(mol, pos, donors=None, frozen=frozenset()):
        seen.append(donors)
        return real(mol, pos, donors, frozen)

    monkeypatch.setattr(geom, "donor_orientation", spy)
    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=4, seed=1).minimize()
    assert seen, "donor_orientation was never called through rx.embed(...).minimize()"
    assert any(d for d in seen), "the gate was called only with donors=None, so it was a silent no-op"
    assert ens.sphere, "Ensemble.sphere is empty after minimize(): the intended donors did not survive"


def test_perceiving_the_donors_makes_the_gate_blind():
    mol, donors = feh2co4(folds=_STITCH)
    pos = mol.GetConformer().GetPositions()
    assert coord.donor_orientation(mol, pos, donors), "the intended sphere must catch the 0° carbonyl"
    assert not coord.donor_orientation(mol, pos, None)


# --- true positives: what the ruler can prove impossible -------------------------------------------------


def test_donor_orientation_reports_folded_carbonyl():
    mol, donors = feh2co4(folds=_STITCH)
    v = coord.donor_orientation(mol, mol.GetConformer().GetPositions(), donors)
    assert len(v) == 1, f"expected exactly the 0° carbonyl, got {[str(x) for x in v]}"
    assert v[0].atoms == (0, 1, 2)
    assert v[0].value == pytest.approx(0.0, abs=0.1)
    assert "folded back over the metal" in v[0].detail


def test_element_key_splits_the_carbonyl_from_the_nitrile():
    mol, donors = feh2co4(folds=(152.4, 180.0, 180.0, 180.0))
    pos = mol.GetConformer().GetPositions()
    assert float(np.linalg.norm(pos[2] - pos[0])) == pytest.approx(2.85, abs=0.01)
    v = coord.donor_orientation(mol, pos, donors)
    assert len(v) == 1, "the CARBONYL at 152.4° is below the C sp floor of 155"
    assert v[0].value == pytest.approx(152.4, abs=0.1)
    assert not coord.metal_overbond(mol, pos, donors), "the radial floor is blind to it: that is the point"

    nitrile, npos, n = _donor_at("CC#N", 152.4)
    assert not coord.donor_orientation(nitrile, npos, [n]), "the NITRILE at the same angle is above its floor"
    assert coord.donor_fold(nitrile, donors=[n]).angles, "...but it is still JUDGED, not abstained on"

    side_on, spos, n2 = _donor_at("CC#N", 75.0)
    assert len(coord.donor_orientation(side_on, spos, [n2])) == 1, "an sp N swung side-on is provably impossible"


# --- what the ruler must not prove: abstention is load-bearing --------------------------------------------


def test_uncalibrated_class_is_reported_but_never_gated():
    out, pos, o = _donor_at("COC", 60.0, donor_num=8)  # dimethyl ether
    assert _stripped_hybridisation(out)[o] == Chem.HybridizationType.SP3, "the ether O must type as sp3"
    assert ("O", Chem.HybridizationType.SP3) not in _FOLD_WINDOW, "O sp3 (n=2) must have NO threshold at all"
    assert not coord.donor_orientation(out, pos, [o]), "an UNCALIBRATED class must never be gated"
    rep = coord.donor_fold(out, donors=[o])
    assert o in rep.unknown, "...and it must be REPORTED as unknown, not silently dropped"
    assert not rep.angles, "an unknown donor is never judged"


def test_gate_fires_on_the_fold_direction_only():
    splayed, pos, n = _donor_at("CN", 160.0)  # above the N sp3 ceiling of 158.4
    assert not coord.donor_orientation(splayed, pos, [n]), "the OVERSHOOT must not be gated"
    assert coord.donor_fold(splayed, donors=[n]).outside_window, "...but the metric MUST still report it"

    folded, pos, n = _donor_at("CN", 80.0)  # below the N sp3 floor of 82: no crystal realises this
    v = coord.donor_orientation(folded, pos, [n])
    assert len(v) == 1, "the FOLD direction must be gated"
    assert v[0].value == pytest.approx(80.0, abs=0.5)


def test_kappa1_carboxylate_is_metric_only():
    anionic = lambda a: a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1  # noqa: E731
    out, pos, o = _donor_at("CC(=O)[O-]", 100.0, pick=anionic)
    assert not coord.donor_orientation(out, pos, [o]), "100° is inside the census window; must NOT be flagged"


def test_two_estimators_refuse_to_gate_when_they_disagree():
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=1, seed=1).minimize()
    rep = coord.donor_fold(ens.mol, ens.ids[0], donors=_sphere_of(ens))
    assert [d for d in rep.unknown if ens.mol.GetAtomWithIdx(d).GetAtomicNum() == 6], "the [CH-] must be UNKNOWN"
    assert all(a.donor not in rep.unknown for a in rep.angles), "an UNKNOWN donor must never be judged"


# --- the accept suite: not vacuous, and zero false positives ---------------------------------------------

_HEALTHY = [
    # the nitrile is judged but not accept-checked: with the sp-linear seed hold deleted a bare-SMILES sp donor
    # embeds side-on and the ruler correctly flags it (the real energy re-opens it to end-on).
    ("nitrile", "CC#N[Pd](Cl)Cl", False),
    ("en-chelate", "Br[Pd]1(Cl)NCCN1", True),  # the chelate's other arm is judged, not exempted
    ("depe-ni-amidate", "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1", True),
]


@pytest.mark.parametrize(("name", "smi", "accept"), _HEALTHY, ids=[h[0] for h in _HEALTHY])
def test_healthy_donor_checks_are_nonvacuous(name, smi, accept):
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=1, seed=1).minimize()
    assert ens.ids, f"{name}: embed produced nothing"
    donors = _sphere_of(ens)
    assert coord.donor_fold(ens.mol, ens.ids[0], donors=donors).angles, f"{name}: the walk judged ZERO angles"
    if not accept:
        return
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"{name}: FALSE POSITIVE on a healthy conformer; {[str(x) for x in v]}"


def test_kappa2_carboxylate_apex_is_exempt():
    from rxembed.metal_distance import APEX, overbond_tier

    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    assert ens.ids, "the κ2 acetate did not embed"
    donors = _sphere_of(ens)
    donor_set = set(donors)
    tiers = (overbond_tier(ens.mol, donor_set, i) for i in range(ens.mol.GetNumAtoms()))
    apex = [i for i, tier in enumerate(tiers) if i not in donor_set and tier == APEX]
    assert apex, "the carboxylate bridgehead must be an APEX (bonded to both donor oxygens)"
    for cid in ens.ids:
        assert not coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)


def test_side_on_eta2_donor_is_exempt():
    smi = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"  # side-on alkyne
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=4, seed=1).minimize()
    assert ens.ids, "the side-on η² did not embed"
    donors = _sphere_of(ens)
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"FALSE POSITIVE on a side-on η²; {[str(x) for x in v]}"


def test_organic_molecule_is_a_strict_no_op():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1O"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    assert not coord.donor_orientation(mol, mol.GetConformer().GetPositions())
    rep = coord.donor_fold(mol)
    assert not rep.angles
    assert not rep.unknown
    assert rep.fold == 0.0


def test_metric_is_reference_free_and_planarity_never_gates():
    healthy, donors = feh2co4()
    stitched, _ = feh2co4(folds=_STITCH)
    assert coord.donor_fold(healthy, donors=donors).fold < coord.donor_fold(stitched, donors=donors).fold
    kinds = {v.kind for v in geom.check(stitched, donors=donors).violations}
    assert "planarity_donation" not in kinds, "the planarity rung must not produce a violation"
    assert "dihedral" not in kinds


# --- the pipeline gate over these rulers ---------------------------------------------------------------------
# contract NOTE: the two below are `pipeline/geom_check` rules with no metal in them; they live here because
# they are the converse guards for the metal-side flexes above, and `tests/pipeline/test_geom_check.py` asserts
# each violation KIND but not the element-awareness that separates these two cases. Move them there if it
# claims them.


def test_eta2_planarity_flex_is_metal_local():
    m = Chem.AddHs(Chem.MolFromSmiles("C=CC=C"))  # butadiene, no metal -> no η² flex
    rdDistGeom.EmbedMolecule(m, randomSeed=1)
    c = m.GetConformer()
    ci = next(a.GetIdx() for a in m.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    p = c.GetPositions()
    p[ci] = p[ci] + [0.0, 0.0, 0.35]  # shove one sp2 carbon 0.35 Å out of plane (past the 0.15 default)
    _place(m, p)
    assert any(v.kind == "planarity" for v in geom.planarity(m, c.GetPositions())), "non-metal sp2 wrongly flexed"


def test_xh_bond_length_window_is_element_aware():
    m = Chem.AddHs(Chem.MolFromSmiles("CP"))
    rdDistGeom.EmbedMolecule(m, randomSeed=1)
    c = m.GetConformer()
    h_p = next(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetSymbol() == "P")
    p = m.GetAtomWithIdx(h_p).GetNeighbors()[0].GetIdx()
    pos = c.GetPositions()
    unit = (pos[h_p] - pos[p]) / np.linalg.norm(pos[h_p] - pos[p])
    for length, flags in ((1.42, False), (1.9, True)):
        pos[h_p] = pos[p] + unit * length
        _place(m, pos)
        assert bool(any(v.kind == "hydrogen" for v in geom.hydrogens(m, c.GetPositions()))) is flags, f"P-H {length}"
