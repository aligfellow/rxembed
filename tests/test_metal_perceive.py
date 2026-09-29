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
from rxembed.metal_polyhedron import POLYHEDRA
from rxembed.pipeline import geom_check as geom

_FE_C, _C_O, _FE_H = 1.80, 1.13, 1.55  # Å: the recorded FeH2(CO)4 bond lengths


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


# --- the accept suite: not vacuous, and zero false positives ---------------------------------------------

_HEALTHY = [
    # the nitrile is judged but not accept-checked: with the sp-linear seed hold deleted a bare-SMILES sp donor
    # embeds side-on and the ruler correctly flags it (the real energy re-opens it to end-on).
    ("nitrile", "CC#N[Pd](Cl)Cl", False),
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


@pytest.mark.parametrize("warn", [True])
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
