"""Test coordination-sphere and donor-orientation perception."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_perceive as coord
from rxembed.metal_donor_orient import FOLD_WINDOW, stripped_hybridisation
from rxembed.metal_polyhedron import POLYHEDRA, describe
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
    assert geom.bond_angle(pos[pd], pos[n], pos[c]) == pytest.approx(angle_deg, abs=0.5)
    return _place(out, pos), pos, n


def _sphere_of(ens):
    return sorted({int(d) for ds in ens.sphere.values() for d in ds})


# --- true positives: what the ruler can prove impossible -------------------------------------------------


def test_donor_orientation_reports_folded_carbonyl():
    mol, donors = feh2co4(folds=_STITCH)
    pos = mol.GetConformer().GetPositions()
    v = coord.donor_orientation(mol, pos, donors)
    assert len(v) == 1, f"expected exactly the 0° carbonyl, got {[str(x) for x in v]}"
    assert v[0].atoms == (0, 1, 2)
    assert v[0].value == pytest.approx(0.0, abs=0.1)
    assert "census floor" in v[0].detail
    assert "inspect donor geometry and restraints" in v[0].detail
    assert not coord.donor_orientation(mol, pos, None), "the gate must not see donors it was not given"


def test_donor_angle_warning_does_not_claim_an_enclosed_carbon_has_inverted():
    mol = Chem.MolFromSmiles("[C-]([SiH3])([SiH3])([SiH3])->[Zn+]")
    pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.3], [-0.9, 1.55, 0.7], [-0.9, -1.55, 0.7], [0.0, 0.0, -2.0]])
    weights = np.linalg.solve(np.vstack((pos[1:].T, np.ones(4))), [0.0, 0.0, 0.0, 1.0])
    assert np.all(weights > 0), "the donor must be strictly inside the four-carrier tetrahedron"
    (violation,) = coord.donor_orientation(mol, pos, donors=[0])
    assert violation.atoms == (4, 0, 1)
    assert violation.limit == FOLD_WINDOW[("C", Chem.HybridizationType.SP3)][0]
    assert "census floor" in violation.detail
    assert "inspect" in violation.detail
    assert "folded" not in violation.detail
    assert "inverted" not in violation.detail


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
    assert stripped_hybridisation(out)[o] == Chem.HybridizationType.SP3, "the ether O must type as sp3"
    assert ("O", Chem.HybridizationType.SP3) not in FOLD_WINDOW, "O sp3 (n=2) must have NO threshold at all"
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
    out, pos, o = _donor_at("CC(=O)[O-]", 100.0, pick=lambda a: a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1)
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
    ("en-chelate", "[Br-]->[Pd+2]1(<-[Cl-])<-NCCN->1", True),  # neutral en: its other arm is judged
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


def test_a_partial_declaration_perceives_the_other_metal_of_a_bridged_dimer():
    """Declaring one metal of a Pd2(mu-Cl)2Cl4 dimer judges that metal only; the other keeps its own ligands."""
    rw = Chem.RWMol()
    for z in (46, 46, 17, 17, 17, 17, 17, 17, 6):
        rw.AddAtom(Chem.Atom(z))
    mol = rw.GetMol()
    pos = np.array(
        [[-1.75, 0, 0], [1.75, 0, 0], [0, 1.65, 0], [0, -1.65, 0], [-3.4, 1.6, 0], [-3.4, -1.6, 0], [3.4, 1.6, 0]]
    )
    pos = np.vstack([pos, [3.4, -1.6, 0], [-1.75, 0, 2.0]])  # B's last terminal Cl, then a carbon crushed onto A
    got = coord.metal_overbond(mol, pos, {0: {2, 3, 4, 5}})
    assert [(v.kind, v.atoms) for v in got] == [("metal_overbond", (0, 8))]


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


# --- perception: the shape invariant ------------------------------------------------------------------


def _ideal_sphere(dirs, r):
    """A bare Mol whose atom 0 is a metal and 1..N its vertices at radius `r` along `dirs`."""
    rw = Chem.RWMol()
    for _ in range(len(dirs) + 1):
        rw.AddAtom(Chem.Atom(6))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for i, d in enumerate(dirs):
        u = np.array(d, float)
        conf.SetAtomPosition(i + 1, Point3D(*(u / np.linalg.norm(u) * r)))
    mol.AddConformer(conf, assignId=True)
    return mol


def _classify(dirs, r):
    return coord.classify_geometry(_ideal_sphere(dirs, r), 0, list(range(1, len(dirs) + 1)))


def _bailar(degrees):
    """The octahedron's own vertices with one C3 face rotated by `degrees` about the body diagonal.

    0° leaves the octahedron, 60° reaches the trigonal prism. Built from the record's own `vertex_dirs` so the
    probe cannot drift away from the shape it is distorting.
    """
    dirs = np.array(POLYHEDRA["octahedral"].vertex_dirs, float)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    face = [i for i, d in enumerate(dirs) if d @ axis > 0]
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)  # Rodrigues
    dirs[face] = dirs[face] @ rot.T
    return dirs


def _tilted_square(tilt_deg):
    """A square plane with an alternating `tilt_deg` out-of-plane bow: the tetrahedral distortion of Ni/Pd(II)."""
    t = np.radians(tilt_deg)
    return [
        (np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t) * (1 if k % 2 == 0 else -1))
        for k, phi in enumerate((0.0, np.pi / 2, np.pi, 3 * np.pi / 2))
    ]


@pytest.mark.parametrize("name", sorted(POLYHEDRA))
def test_all_records_round_trip(name):
    got = _classify(POLYHEDRA[name].vertex_dirs, 2.1)
    assert got == name, f"{describe(name)} re-perceives as {got}"


def test_unbound_metal_has_no_coordination_geometry():
    mol = Chem.MolFromSmiles("[Hg]")
    mol.AddConformer(Chem.Conformer(1))

    assert coord.classify_geometry(mol, 0, []) is None


def _shallow_pyramid(frac):
    """The CN3 pyramid record with its elevation scaled by `frac`: shallower than the ideal, still tilted."""
    dirs = np.array(POLYHEDRA["trigonal_pyramidal"].vertex_dirs, float)
    dirs[:, 2] *= frac
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("dirs", "bond_length", "expected"),
    [
        (POLYHEDRA["trigonal_pyramidal"].vertex_dirs, 1.4, "trigonal_pyramidal"),
        (_shallow_pyramid(0.8), 2.0, "trigonal_pyramidal"),
        (_tilted_square(8), 2.3, "square_planar"),
        ([(np.cos(a), np.sin(a), 0.0) for a in np.arange(6) * np.pi / 3], 2.1, "hexagonal_planar"),
        (_bailar(60.0), 2.1, "trigonal_prismatic"),
    ],
    ids=["short-bonded-pyramid", "shallow-pyramid", "bowed-square-plane", "hexagonal-plane", "bailar-twist-endpoint"],
)
def test_distorted_shell_reads_as_the_nearest_ideal(dirs, bond_length, expected):
    assert _classify(dirs, bond_length) == expected


def test_bis_chelate_zinc_tetrahedron_reads_tetrahedral():
    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral").select(index=0)
    mol = iso.restore(rx.embed(iso, n=2, seed=7).minimize().mol)
    assert [i.geometry for i in rx.metal(mol)] == ["tetrahedral"]


@pytest.mark.parametrize("warn", [True, False])
def test_poor_shape_returns_record_and_warns(warn, caplog):
    squashed = np.array([(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])])
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = coord.classify_geometry(_ideal_sphere(squashed, 2.1), 0, list(range(1, 7)), warn=warn)
    assert got is not None, "a poor fit is still the nearest record, reported loudly"
    fired = bool([r for r in caplog.records if "no shape fits" in r.message])
    assert fired == warn, caplog.text


def test_near_tie_keeps_the_argmin_and_names_the_runner_up(caplog):
    """A CN5 witness almost equidistant between trigonal_bipyramidal and square_pyramidal (residual gap
    ~3e-5, far inside `_FIT_MARGIN`) still returns one name, the argmin, and logs the runner-up with the
    `geometry=` remedy rather than silently picking either. The acceptance gate is what accepts a requested
    shape this close to the argmin (`shape_reading`, rule B); `classify_geometry` itself always names the one
    nearest reading.
    """
    sp = np.array(POLYHEDRA["square_pyramidal"].vertex_dirs, float)
    tbp = np.array(POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, float)
    dirs = 0.662 * sp + 0.338 * tbp
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = coord.classify_geometry(_ideal_sphere(dirs, 2.1), 0, list(range(1, 6)))
    assert got == "trigonal_bipyramidal"
    assert [r for r in caplog.records if "near-tie" in r.message and "pass geometry='SPY'" in r.message], caplog.text
