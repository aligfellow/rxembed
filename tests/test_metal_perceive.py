"""`metal_perceive`: the coordination-sphere ruler; who coordinates, and where the ligand POINTS.

Every other geometry check measures a distance, so a carbonyl folded flat onto its iron; carbon dead on its
octahedral vertex at a perfect Fe-C distance; passes them all, and `metal_overbond`'s radial floor is blind
too (a right-angle carbonyl sits at Fe...O 2.13 Å, over the 2.04 Å floor). `donor_orientation` gates the angle
against a per-(element, hybridisation) census window and ABSTAINS on an uncalibrated class; `donor_fold` is the
report-only metric. Witnesses are built by moving atoms: no force field, no xtb.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed.pipeline as rx
from rxembed import metal_perceive as coord
from rxembed.metal_donor_orient import _FOLD_WINDOW, _stripped_hybridisation
from rxembed.pipeline import geom_check as geom

_FE_C, _C_O, _FE_H = 1.80, 1.13, 1.55  # Å: the recorded FeH2(CO)4 bond lengths
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]  # the 4 equatorial vertices; the 2 hydrides go on ±z
_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic: hydrides, η²-H2, ferrocene Cp, a frozen core
_MN_H2_RC = [1, 5, 63, 64, 65, 66]
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


def test_the_gate_is_called_through_the_public_api_with_real_donors(monkeypatch):
    """Live spy: `rx.embed(...).minimize()` reaches `donor_orientation` with a NON-EMPTY donor set.

    Called with `donors=None` the gate is a silent no-op (see the blindness test below), so "it was called" is
    not the assertion; "it was called with a sphere" is. `Ensemble.sphere` surviving minimize is what supplies
    it.
    """
    seen: list[object] = []
    real = coord.donor_orientation

    def spy(mol, pos, donors=None, frozen=frozenset()):
        seen.append(donors)
        return real(mol, pos, donors, frozen)

    monkeypatch.setattr(geom, "donor_orientation", spy)
    ens = rx.embed(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0], n=4, seed=1).minimize()
    assert seen, "donor_orientation was never called through rx.embed(...).minimize()"
    assert any(d for d in seen), "the gate was called, but ALWAYS with donors=None: it is a silent no-op"
    assert ens.sphere, "Ensemble.sphere is empty after minimize(): the intended donors did not survive"


def test_perceiving_the_donors_makes_the_gate_blind():
    """Why `donors=` is mandatory: a folded carbonyl's O swings inside the radius and exempts its own carbon.

    The fold erases its own evidence; perceived as a co-donor, the carbon is written off as haptic. If the
    PERCEIVED case ever starts catching it, the circularity is gone and this should be re-thought, not deleted.
    """
    mol, donors = feh2co4(folds=_STITCH)
    pos = mol.GetConformer().GetPositions()
    assert coord.donor_orientation(mol, pos, donors), "the intended sphere must catch the 0° carbonyl"
    assert not coord.donor_orientation(mol, pos, None)


# --- true positives: what the ruler can prove impossible -------------------------------------------------


def test_the_witness_is_rejected_and_only_the_folded_carbonyl_fires():
    """FeH2(CO)4 as the rigid stitch shipped it: one violation, naming (Fe, C, O) at the 0° carbonyl."""
    mol, donors = feh2co4(folds=_STITCH)
    v = coord.donor_orientation(mol, mol.GetConformer().GetPositions(), donors)
    assert len(v) == 1, f"expected exactly the 0° carbonyl, got {[str(x) for x in v]}"
    assert v[0].atoms == (0, 1, 2)
    assert v[0].value == pytest.approx(0.0, abs=0.1)
    assert "folded back over the metal" in v[0].detail


def test_the_ruler_is_strictly_stronger_than_the_radial_floor():
    """A right-angle carbonyl at Fe...O 2.13 Å passes `metal_overbond`'s 2.04 Å floor; the fold gate rejects it."""
    mol, donors = feh2co4(folds=(90.0, 180.0, 180.0, 180.0))
    pos = mol.GetConformer().GetPositions()
    assert float(np.linalg.norm(pos[2] - pos[0])) == pytest.approx(2.13, abs=0.01)
    assert not coord.metal_overbond(mol, pos, donors), "PINNING THE KNOWN BLINDNESS of the radial floor"
    v = coord.donor_orientation(mol, pos, donors)
    assert len(v) == 1
    assert v[0].atoms == (0, 1, 2)
    assert v[0].value == pytest.approx(90.0, abs=0.1)


def test_the_element_key_splits_the_carbonyl_from_the_nitrile():
    """152.4° is impossible for a carbonyl (C sp floor 155) and fine for a nitrile (N sp floor 140).

    Pooled into one `sp` bucket the floor would be 142; too wide to prove anything. This is why the census is
    keyed on (donor element, hybridisation) rather than hybridisation alone. The nitrile is then swung to 75°,
    which no real M-N#C reaches, to show its own floor still bites.
    """
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


def test_an_uncalibrated_class_is_reported_but_never_gated():
    """An sp3 O donor folded to 60° is not gated: the census has 2 of them (n < 6) and cannot calibrate it."""
    out, pos, o = _donor_at("COC", 60.0, donor_num=8)  # dimethyl ether
    assert _stripped_hybridisation(out)[o] == Chem.HybridizationType.SP3, "the ether O must type as sp3"
    assert ("O", Chem.HybridizationType.SP3) not in _FOLD_WINDOW, "O sp3 (n=2) must have NO threshold at all"
    assert not coord.donor_orientation(out, pos, [o]), "an UNCALIBRATED class must never be gated"
    rep = coord.donor_fold(out, donors=[o])
    assert o in rep.unknown, "...and it must be REPORTED as unknown, not silently dropped"
    assert not rep.angles, "an unknown donor is never judged"


def test_the_gate_fires_on_the_fold_direction_only():
    """An sp3 donor splayed to 160° is reported but not gated; the same donor folded to 80° is gated.

    Over-splay is the opposite failure and would false-positive a real phosphine, so the upper half of
    `_FOLD_WINDOW` is deliberately disarmed.
    """
    splayed, pos, n = _donor_at("CN", 160.0)  # above the N sp3 ceiling of 158.4
    assert not coord.donor_orientation(splayed, pos, [n]), "the OVERSHOOT must not be gated"
    assert coord.donor_fold(splayed, donors=[n]).outside_window, "...but the metric MUST still report it"

    folded, pos, n = _donor_at("CN", 80.0)  # below the N sp3 floor of 82: no crystal realises this
    v = coord.donor_orientation(folded, pos, [n])
    assert len(v) == 1, "the FOLD direction must be gated"
    assert v[0].value == pytest.approx(80.0, abs=0.5)


def test_a_kappa1_carboxylate_at_100_degrees_is_a_metric_finding_not_a_violation():
    """Real O sp2 donations reach 95.5° in the crystal (floor 90), so 100° is not provably impossible."""
    anionic = lambda a: a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1  # noqa: E731
    out, pos, o = _donor_at("CC(=O)[O-]", 100.0, pick=anionic)
    assert not coord.donor_orientation(out, pos, [o]), "100° is inside the census window; must NOT be flagged"


def test_the_two_estimators_refuse_to_gate_when_they_disagree():
    """A metal-bound carbanion is sp2 to RDKit and sp3 by π-count; both defensible, so it lands in UNKNOWN."""
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=4, seed=1).minimize()
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
def test_a_healthy_system_is_judged_and_not_flagged(name, smi, accept):
    """Every healthy fixture judges at least one angle, and (bar the nitrile) no conformer is flagged.

    The two halves belong together: a classifier that UNKNOWNs everything is the likeliest silent death and
    would pass the accept half while measuring nothing.
    """
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=4, seed=1).minimize()
    assert ens.ids, f"{name}: embed produced nothing"
    donors = _sphere_of(ens)
    assert coord.donor_fold(ens.mol, ens.ids[0], donors=donors).angles, f"{name}: the walk judged ZERO angles"
    if not accept:
        return
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"{name}: FALSE POSITIVE on a healthy conformer; {[str(x) for x in v]}"


def test_a_kappa2_carboxylate_apex_is_exempt():
    """A κ2 bridgehead is bonded to both donor oxygens, so its ~90° arms are geometrically forced, not folds."""
    from rxembed.metal_distance import APEX, overbond_tier

    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    assert ens.ids, "the κ2 acetate did not embed"
    donors = _sphere_of(ens)
    donor_set = set(donors)
    tiers = (overbond_tier(ens.mol, donor_set, i) for i in range(ens.mol.GetNumAtoms()))
    apex = [i for i, tier in enumerate(tiers) if i not in donor_set and tier == APEX]
    assert apex, "the carboxylate bridgehead must be an APEX (bonded to both donor oxygens)"
    for cid in ens.ids:
        assert not coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)


def test_a_side_on_eta2_donor_is_exempt():
    """An η² donor has no donation axis (the metal sits ~70° off the ligand axis by definition); exempt.

    A donation axis taken across a co-donor once crushed an η² imine's Ni-N to 1.42 Å.
    """
    smi = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"  # side-on alkyne
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=4, seed=1).minimize()
    assert ens.ids, "the side-on η² did not embed"
    donors = _sphere_of(ens)
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"FALSE POSITIVE on a side-on η²; {[str(x) for x in v]}"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_every_exemption_holds_on_one_real_bimetallic():
    """mn-h2: hydride / haptic / bridging / APEX / frozen-core exemptions never fire, and carbonyls stay judged."""
    frozen = set(_MN_H2_RC)
    judged = 0
    for iso in rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC):
        ens = rx.embed(iso, n=3, seed=1).minimize(_retry=False)
        if not ens.ids:
            continue
        donors = _sphere_of(ens)
        donor_set = set(donors)
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            for x in coord.donor_orientation(ens.mol, pos, donors, frozen=_MN_H2_RC):
                _m_i, d, sub = x.atoms
                a = ens.mol.GetAtomWithIdx(d)
                exempt = (
                    a.GetAtomicNum() == 1
                    or any(nb.GetIdx() in donor_set for nb in a.GetNeighbors())
                    or sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ens.sphere) >= 2
                    or sum(1 for dd in donors if ens.mol.GetBondBetweenAtoms(sub, int(dd))) >= 2
                    or d in frozen
                    or sub in frozen
                )
                assert not exempt, f"an EXEMPT donor was flagged ({x})"
            judged += len(coord.donor_fold(ens.mol, cid, donors=donors, frozen=_MN_H2_RC).angles)
    assert judged, "mn-h2 judged ZERO angles: every donor was exempted and the ruler measured nothing"


def test_an_organic_molecule_is_a_strict_no_op():
    """No metal, no walk, no violation: the ruler touches nothing outside a coordination sphere."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1O"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    assert not coord.donor_orientation(mol, mol.GetConformer().GetPositions())
    rep = coord.donor_fold(mol)
    assert not rep.angles
    assert not rep.unknown
    assert rep.fold == 0.0


def test_the_metric_needs_no_reference_and_the_planarity_rung_never_gates():
    """`fold` asks the structure about itself (so it works in generation mode) and planarity is report-only.

    Real crystals reach 87.6° out of plane, so a dihedral GATE would fire on 5-8% of them.
    """
    healthy, donors = feh2co4()
    stitched, _ = feh2co4(folds=_STITCH)
    assert coord.donor_fold(healthy, donors=donors).fold < coord.donor_fold(stitched, donors=donors).fold
    kinds = {v.kind for v in geom.check(stitched, donors=donors).violations}
    assert "planarity_donation" not in kinds, "the planarity rung must NEVER produce a Violation"
    assert "dihedral" not in kinds
    assert coord.donor_fold(stitched, donors=donors).planarity >= 0.0


# --- the pipeline gate over these rulers ---------------------------------------------------------------------
# contract NOTE: the two below are `pipeline/geom_check` rules with no metal in them; they live here because
# they are the converse guards for the metal-side flexes above, and `tests/pipeline/test_geom_check.py` asserts
# each violation KIND but not the element-awareness that separates these two cases. Move them there if it
# claims them.


def test_the_eta2_planarity_flex_does_not_leak_to_a_non_metal_sp2():
    """A twisted non-metal sp2 still flags: the η² planarity flex is metal-side-on-specific."""
    m = Chem.AddHs(Chem.MolFromSmiles("C=CC=C"))  # butadiene, no metal -> no η² flex
    rdDistGeom.EmbedMolecule(m, randomSeed=1)
    c = m.GetConformer()
    ci = next(a.GetIdx() for a in m.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    p = c.GetPositions()
    p[ci] = p[ci] + [0.0, 0.0, 0.35]  # shove one sp2 carbon 0.35 Å out of plane (past the 0.15 default)
    _place(m, p)
    assert any(v.kind == "planarity" for v in geom.planarity(m, c.GetPositions())), "non-metal sp2 wrongly flexed"


def test_the_xh_bond_length_window_is_element_aware():
    """A correct P-H (~1.42 Å) passes where a C-H ceiling of 1.3 would have flagged it; 1.9 Å still flags."""
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
