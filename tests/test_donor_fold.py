"""The donor-fold ruler: the gate that asks where a ligand points, not where it is.

Every other geometry check measures a distance, so a carbonyl folded flat onto its iron — carbon dead on its
octahedral vertex at a perfect Fe-C distance — passes them all, and ``metal_overbond``'s radial floor is blind
too (a right-angle carbonyl sits at Fe...O = 2.13 A, over the 2.04 A floor). ``donor_orientation`` gates the
angle; ``donor_fold`` is the report-only metric. Witnesses are built by moving atoms — no force field, no xtb.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed import geometry as geom
from rxembed.rdkit_embed import coordination as coord
from rxembed.rdkit_embed.constraints.donor_orient import _FOLD_WINDOW, _stripped_hybridisation

_FE_C, _C_O, _FE_H = 1.80, 1.13, 1.55  # A — the recorded FeH2(CO)4 bond lengths
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]  # the 4 equatorial vertices; the 2 hydrides go on +-z


def _feh2co4(folds=(180.0, 180.0, 180.0, 180.0)):
    """FeH2(CO)4 with its carbons pinned on octahedral vertices and each carbonyl bent to its Fe-C-O angle.

    Only the oxygens swing (Fe-C-O: 180 points O straight out, 0 folds it onto the metal), so every
    distance-based check sees a textbook octahedron however far the ligands fold. Returns ``(mol, donors)``.
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
        d = np.cos(t) * (-u) + np.sin(t) * np.array([0.0, 0.0, 1.0])
        pos += [_FE_C * u, _FE_C * u + _C_O * d]
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


def _folds(mol, donors):
    """The realised donor_orientation violations for the mol's only conformer."""
    return coord.donor_orientation(mol, mol.GetConformer().GetPositions(), donors)


def _place(mol, pos):
    """Write ``pos`` into the mol's conformer and return it — ``donor_fold`` reads the conformer, so the metal
    (which ``RWMol.AddAtom`` leaves at the origin) must be positioned there too, not just in the numpy array."""
    conf = mol.GetConformer()
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    return mol


def _donor_at(smi, angle_deg, donor_num=7):
    """A small molecule with a Pd placed at an exact M-donor-substituent angle, 2.1 A from the donor."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == donor_num)
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


# --- 1. the gate is actually reachable ---------------------------------------------------------------


def test_the_gate_is_actually_called():
    """``check()`` really runs the fold gate, and ``minimize()`` really hands it the intended donors."""
    from rxembed import pipeline

    src = inspect.getsource(geom.check)
    assert "donor_orientation(" in src, "check() does not call the fold gate — it cannot reject anything"

    src = inspect.getsource(pipeline.Ensemble._reembed_until_clean)
    assert "_geometry.check(" in src, "minimize()'s acceptance predicate does not call the geometry gate"
    # without `donors=` the sphere is perceived, and a folded ligand's own donor is written off as haptic —
    # the fold erases its own evidence and the gate becomes a silent no-op (see test 3).
    assert "donors=" in src, "minimize() calls the gate WITHOUT `donors=` — the perceived-sphere silent no-op"


@pytest.mark.parametrize("smi", ["CC#N[Pd](Cl)Cl", "Br[Pd]1(Cl)NCCN1"])
def test_the_gate_is_called_through_the_public_api_with_real_donors(monkeypatch, smi):
    """Live spy: ``rx.embed(...).minimize()`` reaches the gate with a non-empty donor set surviving on ``sphere``."""
    import rxembed as rx

    seen: list[object] = []
    real = coord.donor_orientation

    def spy(mol, pos, donors=None, frozen=frozenset()):
        seen.append(donors)
        return real(mol, pos, donors, frozen)

    monkeypatch.setattr(geom, "donor_orientation", spy)

    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()

    assert seen, "donor_orientation was never called through rx.embed(...).minimize()"
    assert any(d for d in seen), "the gate was called, but ALWAYS with donors=None — it is a silent no-op"
    assert ens.sphere, "Ensemble.sphere is empty after minimize(): the intended donors did not survive"


def test_perceiving_the_donors_makes_the_gate_blind():
    """Why ``donors=`` is mandatory: a folded carbonyl's O swings inside the radius, is perceived as a co-donor,
    and its own carbon is exempted as haptic — so the intended sphere catches the 0 deg fold but a perceived one
    misses it."""
    mol, donors = _feh2co4(folds=(0.0, 171.7, 171.8, 180.0))
    pos = mol.GetConformer().GetPositions()
    assert coord.donor_orientation(mol, pos, donors), "the intended sphere must catch the 0 deg carbonyl"
    assert not coord.donor_orientation(mol, pos, None), (
        "a PERCEIVED sphere is expected to miss it — if this ever starts passing, the circularity is gone "
        "and this test should be re-thought rather than deleted"
    )


def test_end_to_end_a_folded_ligand_is_rejected_through_the_real_pipeline():
    """Fold a ligand in a real pipeline conformer and feed it to the real predicate: rejected with the intended
    ``sphere``, silently accepted with ``donors=None`` — the whole reason the sphere must survive ``minimize()``."""
    import rxembed as rx

    iso = rx.metal("CC#N[Pd](Cl)Cl", "square_planar")[0]
    ens = rx.embed(iso, n=6, seed=1).minimize()
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    assert donors, "Ensemble.sphere did not survive minimize() — the gate would be handed donors=None"

    mol, cid = ens.mol, ens.ids[0]
    m = next(iter(ens.sphere))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    c = next(x.GetIdx() for x in mol.GetAtomWithIdx(n).GetNeighbors() if x.GetSymbol() == "C")

    # the clean control — the pipeline's own conformer is end-on (the sp-linear hold); the ruler must go quiet.
    now = mol.GetConformer(cid).GetPositions()
    assert geom._angle(now[m], now[n], now[c]) > 150.0, "the pipeline no longer ships an end-on nitrile"
    assert geom.check(mol, cid, donors=donors).ok(), "an end-on nitrile is healthy — the ruler must not fire"

    # the folded case — rigidly rotate the same CH3-C#N fragment about N to side-on; the ruler must reject it.
    frag, seen, stack = [], {n}, [c]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        frag.append(i)
        stack += [x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors() if x.GetIdx() not in seen]
    u = now[m] - now[n]
    u /= np.linalg.norm(u)
    w = np.cross(u, [0.0, 0.0, 1.0])
    w /= np.linalg.norm(w)
    old = (now[c] - now[n]) / np.linalg.norm(now[c] - now[n])
    folded = Chem.Mol(mol)
    conf = folded.GetConformer(cid)
    t = np.radians(110.0)  # below the 140 deg N-sp floor, wide enough not to clash the methyl
    new = np.cos(t) * u + np.sin(t) * w
    ax = np.cross(old, new)
    ax /= np.linalg.norm(ax)
    th = float(np.arccos(np.clip(float(np.dot(old, new)), -1.0, 1.0)))
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    rot = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)  # Rodrigues
    for i in frag:
        conf.SetAtomPosition(int(i), Point3D(*map(float, now[n] + rot @ (now[i] - now[n]))))
    fpos = conf.GetPositions()
    assert geom._angle(fpos[m], fpos[n], fpos[c]) == pytest.approx(110.0, abs=0.5)
    kinds = [x.kind for x in geom.check(folded, cid, donors=donors).violations]
    assert "donor_orientation" in kinds, f"the fold gate must fire on the folded nitrile, got {kinds}"
    assert not coord.metal_overbond(folded, fpos, donors), "the radial floor is blind to this fold — that is the point"
    # the silent no-op: the fold is the only reason to reject this geometry, so with no sphere it passes.
    assert "donor_orientation" not in [x.kind for x in geom.check(folded, cid, donors=None).violations], (
        "PINNING THE SILENT NO-OP: with no sphere the fold gate cannot fire — the sphere must survive minimize()"
    )


# --- 2. true positives: what the ruler can prove impossible -------------------------------------------


def test_the_ruler_rejects_the_witness():
    """FeH2(CO)4 as the rigid stitch shipped it (Fe-C-O = 0/171.7/171.8/180): only the 0 deg carbonyl fires."""
    mol, donors = _feh2co4(folds=(0.0, 171.7, 171.8, 180.0))
    v = _folds(mol, donors)

    assert len(v) == 1, f"expected exactly the 0 deg carbonyl, got {[str(x) for x in v]}"
    assert v[0].atoms == (0, 1, 2), "the violation must NAME the folded carbonyl (Fe, C, O)"
    assert v[0].value == pytest.approx(0.0, abs=0.1)
    assert "folded back over the metal" in v[0].detail
    # the three healthy carbonyls stay inside the census window (C sp median 176.2)
    assert coord.donor_fold(mol, donors=donors).fold == pytest.approx(176.2, abs=0.5)


def test_the_ruler_is_strictly_stronger_than_the_radial_floor():
    """A right-angle carbonyl: Fe...O = 2.13 A passes ``metal_overbond``'s 2.04 A floor, the fold gate rejects."""
    mol, donors = _feh2co4(folds=(90.0, 180.0, 180.0, 180.0))
    pos = mol.GetConformer().GetPositions()

    fe_o = float(np.linalg.norm(pos[2] - pos[0]))
    assert fe_o == pytest.approx(2.13, abs=0.01), "the witness must sit at the measured 2.13 A"

    assert not coord.metal_overbond(mol, pos, donors), (
        "PINNING THE KNOWN BLINDNESS: the radial floor must pass a right-angle carbonyl at 2.13 A"
    )
    v = coord.donor_orientation(mol, pos, donors)
    assert len(v) == 1, "the fold gate must reject what the radial floor waved through"
    assert v[0].atoms == (0, 1, 2)
    assert v[0].value == pytest.approx(90.0, abs=0.1)


def test_a_side_on_nitrile_is_rejected():
    """An sp N donor swung side-on to 75 deg is provably impossible: real M-N#C never goes below 145.0 deg."""
    mol, pos, n = _donor_at("CC#N", 75.0)
    v = coord.donor_orientation(mol, pos, [n])
    assert len(v) == 1
    assert v[0].value == pytest.approx(75.0, abs=0.5)


# --- 3. the element key is load-bearing: the same angle, two elements, two verdicts --------------------


def test_the_element_key_splits_the_carbonyl_from_the_nitrile():
    """152.4 deg (Fe...O 2.85 A) is impossible for a carbonyl (C sp floor 155) but fine for a nitrile (N sp floor 140).

    Pooled into one ``sp`` bucket the floor would be 142 — too wide to prove anything; this is why the census is
    keyed on (donor element, hybridisation).
    """
    mol, donors = _feh2co4(folds=(152.4, 180.0, 180.0, 180.0))
    pos = mol.GetConformer().GetPositions()
    assert float(np.linalg.norm(pos[2] - pos[0])) == pytest.approx(2.85, abs=0.01)

    v = coord.donor_orientation(mol, pos, donors)
    assert len(v) == 1, "the CARBONYL at 152.4 deg is below the C sp floor of 155 — it must be rejected"
    assert v[0].value == pytest.approx(152.4, abs=0.1)
    assert not coord.metal_overbond(mol, pos, donors), "the radial floor is blind to it — that is the point"

    nitrile, npos, n = _donor_at("CC#N", 152.4)
    assert not coord.donor_orientation(nitrile, npos, [n]), (
        "the NITRILE at the SAME angle is above the N sp floor of 140 (crystals reach 145.0 deg) — must NOT be gated"
    )
    assert coord.donor_fold(nitrile, donors=[n]).fold == pytest.approx(26.9, abs=0.5), "but the METRIC must see it"


def test_an_uncalibrated_class_is_never_gated():
    """An sp3 O donor folded to 60 deg is not gated — the census has only 2 of them (n < 6) and cannot calibrate it.

    The ruler judges only measured (element, hybridisation) classes and abstains on the rest, reporting them in
    ``.unknown`` rather than inventing a limit.
    """
    out, pos, o = _donor_at("COC", 60.0, donor_num=8)  # dimethyl ether: an sp3 O, a 2-donation census class
    assert _stripped_hybridisation(out)[o] == Chem.HybridizationType.SP3, "the ether O must type as sp3"
    assert ("O", Chem.HybridizationType.SP3) not in _FOLD_WINDOW, "O sp3 (n=2) must have NO threshold at all"
    assert not coord.donor_orientation(out, pos, [o]), "an UNCALIBRATED class must never be gated"
    rep = coord.donor_fold(out, donors=[o])
    assert o in rep.unknown, "...and it must be REPORTED as unknown, not silently dropped"
    assert not rep.angles, "an unknown donor is never judged"


# --- 4. what the ruler cannot prove — and must therefore not flag --------------------------------------


def test_the_gate_fires_on_the_fold_direction_only():
    """An sp3 donor splayed to 160 deg is not gated (over-splay is the opposite failure and would false-positive a
    real phosphine); the same donor folded to 80 deg is. The upper half of ``_FOLD_WINDOW`` is deliberately disarmed."""
    splayed, pos, n = _donor_at("CN", 160.0)  # above the N sp3 ceiling of 158.4 — unusual, not a fold
    assert not coord.donor_orientation(splayed, pos, [n]), "the OVERSHOOT must not be gated"
    rep = coord.donor_fold(splayed, donors=[n])
    assert rep.outside_window, "...but the metric MUST still report it, or the over-splay is hidden"

    folded, pos, n = _donor_at("CN", 80.0)  # below the N sp3 floor of 82 — no crystal realises this
    v = coord.donor_orientation(folded, pos, [n])
    assert len(v) == 1, "the FOLD direction must be gated"
    assert v[0].value == pytest.approx(80.0, abs=0.5)


def test_a_kappa1_carboxylate_relaxed_to_100_deg_is_a_metric_finding_not_a_violation():
    """Real O sp2 donations reach 95.5 deg in the crystal (floor 90), so 100 deg is not provably impossible."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    rw = Chem.RWMol(mol)
    pd = rw.AddAtom(Chem.Atom(46))
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1)
    c = next(n.GetIdx() for n in mol.GetAtomWithIdx(o).GetNeighbors())
    rw.AddBond(o, pd, Chem.BondType.DATIVE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    pos = out.GetConformer().GetPositions()
    pos = np.vstack([pos, np.zeros(3)])
    u = pos[c] - pos[o]
    u /= np.linalg.norm(u)
    w = np.cross(u, [0.0, 0.0, 1.0])
    w /= np.linalg.norm(w)
    t = np.radians(100.0)
    pos[pd] = pos[o] + 2.0 * (np.cos(t) * u + np.sin(t) * w)

    assert geom._angle(pos[pd], pos[o], pos[c]) == pytest.approx(100.0, abs=0.5)
    assert not coord.donor_orientation(out, pos, [o]), "100 deg is inside the census window — must NOT be flagged"


# --- 5. the classifier: UNKNOWN is load-bearing --------------------------------------------------------


def test_hybridisation_is_classified_on_the_metal_stripped_graph():
    """The metal is stripped before typing: a carbonyl carbon is sp on its own graph, RDKit mis-types it with the
    dative bond in — else the coordination under test would decide the class used to judge it."""
    mol, donors = _feh2co4()
    hyb = _stripped_hybridisation(mol)
    for c in donors[:4]:
        assert hyb[c] == Chem.HybridizationType.SP, "a carbonyl carbon is sp on its own graph"
    for h in donors[4:]:
        assert h not in hyb, "a hydride has no donation axis and must never be classified or gated"


def test_the_two_estimators_refuse_to_gate_when_they_disagree():
    """A metal-bound carbanion is typed sp2 by RDKit and sp3 by the pi-count; both defensible, so the ruler reports
    it in UNKNOWN and gates nothing."""
    import rxembed as rx

    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    rep = coord.donor_fold(ens.mol, ens.ids[0], donors=donors)

    carbanion = [d for d in rep.unknown if ens.mol.GetAtomWithIdx(d).GetAtomicNum() == 6]
    assert carbanion, "the [CH-] carbanion donor must land in UNKNOWN, not be confidently typed"
    assert all(a.donor not in rep.unknown for a in rep.angles), "an UNKNOWN donor must never be judged"


# --- 6. the ruler is not vacuous ----------------------------------------------------------------------

_HEALTHY = [
    ("nitrile", "CC#N[Pd](Cl)Cl", "square_planar"),  # sp donor — embeds side-on (sp-linear seed hold deleted)
    ("en-chelate", "Br[Pd]1(Cl)NCCN1", "square_planar"),  # the chelate's other arm is judged, not exempted
    ("amine", "CCCN[Pd](Cl)(Cl)NCCC", "square_planar"),
    ("acac", "CC1=CC(C)=[O]->[Ni+2](Cl)(Cl)<-[O]1", "square_planar"),  # 6-ring chelate: judged at ~125 deg
    ("henry-depe", "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1", "square_planar"),
    (
        "henry-dppe",
        "O=C1[O-]->[Ni+2]2(<-[N-](c3ccccc3)C1c1ccccc1)<-[P](CC[P]->2(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1",
        "square_planar",
    ),
]


@pytest.mark.parametrize(("name", "smi", "geometry"), _HEALTHY)
def test_the_ruler_is_not_vacuous(name, smi, geometry):
    """Every healthy fixture produces at least one judged angle — a classifier that UNKNOWNs everything is the
    likeliest silent death and would pass every accept test while measuring nothing."""
    import rxembed as rx

    iso = rx.metal(smi, geometry)[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    assert ens.ids, f"{name}: embed produced nothing"
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    rep = coord.donor_fold(ens.mol, ens.ids[0], donors=donors)
    assert rep.angles, f"{name}: the walk judged ZERO angles — the ruler is measuring nothing"


# --- 7. the accept suite: zero false positives (this is the failure mode) ------------------------------


@pytest.mark.parametrize(("name", "smi", "geometry"), [h for h in _HEALTHY if h[0] != "nitrile"])
def test_no_false_positive_on_a_healthy_system(name, smi, geometry):
    """A healthy embed is never flagged.

    The nitrile is excluded: with the sp-linear seed hold deleted a bare-SMILES sp donor embeds side-on and the
    ruler correctly flags it (a true positive; the real energy re-opens it to end-on). The gate is armed inside
    ``minimize()``'s acceptance predicate, so the pipeline re-seeds away from folds instead of shipping them.
    """
    import rxembed as rx

    iso = rx.metal(smi, geometry)[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"{name}: FALSE POSITIVE on a healthy conformer — {[str(x) for x in v]}"


def test_a_kappa2_carboxylate_apex_is_exempt():
    """A κ2 carboxylate's bridgehead is bonded to both donor oxygens (an APEX), so both ~90 deg arms are exempt —
    that angle is geometrically forced by the 4-ring, not a fold, and without the exemption would flag twice."""
    import rxembed as rx
    from rxembed.rdkit_embed.constraints.distance import APEX, overbond_tier

    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    assert ens.ids, "the kappa2 acetate did not embed"
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    apex = [
        i for i in range(ens.mol.GetNumAtoms()) if i not in donors and overbond_tier(ens.mol, set(donors), i) == APEX
    ]
    assert apex, "the carboxylate bridgehead must be an APEX (bonded to both donor oxygens)"
    for cid in ens.ids:
        assert not coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)


@pytest.mark.parametrize(
    "smi",
    [
        "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",  # side-on alkyne
        "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C",  # side-on C=C
    ],
)
def test_a_side_on_eta2_donor_is_exempt(smi):
    """A side-on η² donor has no donation axis (the metal sits ~70 deg off the ligand axis by definition); the
    co-donor exclusion exempts it — a donation axis across a co-donor once crushed an η² imine's Ni-N to 1.42 A."""
    import rxembed as rx

    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    assert ens.ids, "the side-on eta2 did not embed"
    donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
    for cid in ens.ids:
        v = coord.donor_orientation(ens.mol, ens.mol.GetConformer(cid).GetPositions(), donors)
        assert not v, f"FALSE POSITIVE on a side-on eta2 — {[str(x) for x in v]}"


def test_hydrides_and_haptic_donors_are_exempt_on_a_real_bimetallic():
    """mn-h2 (a frozen-TS bimetallic): every exemption in one real structure — hydrides, η²-H₂ co-donors, ten
    haptic ferrocene Cp carbons, and the frozen reacting core — must never fire, while the carbonyls stay judged."""
    import rxembed as rx
    import tests.test_connectivity as tc

    isos = rx.metal(tc._MN_H2, "octahedral", center="Mn", fix=tc._MN_H2_RC)
    frozen = set(tc._MN_H2_RC)
    judged = 0
    for iso in isos:
        ens = rx.embed(iso, n=6, seed=1).minimize(_retry=False)
        if not ens.ids:
            continue
        donors = sorted({int(d) for ds in ens.sphere.values() for d in ds})
        donor_set = set(donors)
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            for x in coord.donor_orientation(ens.mol, pos, donors, frozen=tc._MN_H2_RC):
                _m_i, d, sub = x.atoms
                a = ens.mol.GetAtomWithIdx(d)
                # a flagged donor may be any hybridisation, but never one of the five exempt geometries.
                is_hydride = a.GetAtomicNum() == 1
                is_haptic = any(nb.GetIdx() in donor_set for nb in a.GetNeighbors())
                is_bridging = sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in ens.sphere) >= 2
                sub_is_apex = sum(1 for dd in donors if ens.mol.GetBondBetweenAtoms(sub, int(dd))) >= 2
                in_frozen = d in frozen or sub in frozen
                exempt = is_hydride or is_haptic or is_bridging or sub_is_apex or in_frozen
                assert not exempt, (
                    f"an EXEMPT donor was flagged ({x}) — a hydride / haptic / bridging / APEX / frozen-core "
                    f"exemption failed"
                )
            judged += len(coord.donor_fold(ens.mol, cid, donors=donors, frozen=tc._MN_H2_RC).angles)
    assert judged, "mn-h2 judged ZERO angles — every donor was exempted and the ruler measured nothing"


def test_an_organic_molecule_is_a_strict_no_op():
    """No metal, no walk, no violation, no cost — the ruler touches nothing outside a coordination sphere."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1O"))
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    pos = mol.GetConformer().GetPositions()
    assert not coord.donor_orientation(mol, pos)
    rep = coord.donor_fold(mol)
    assert not rep.angles
    assert not rep.unknown
    assert rep.fold == 0.0


# --- 8. the metric ------------------------------------------------------------------------------------


def test_the_metric_needs_no_reference_and_the_planarity_rung_never_gates():
    """``fold`` asks the structure about itself (no reference, so it works in generation mode); the planarity rung
    is report-only — real crystals reach 87.6 deg out of plane, so a dihedral gate would fire on 5-8% of them."""
    healthy, donors = _feh2co4()
    stitched, _ = _feh2co4(folds=(0.0, 171.7, 171.8, 180.0))

    assert coord.donor_fold(healthy, donors=donors).fold < 5.0
    assert coord.donor_fold(stitched, donors=donors).fold > 170.0, (
        "the stitch wins every distance metric and loses `fold` by ~176 deg — that is the whole point"
    )
    kinds = {v.kind for v in geom.check(stitched, donors=donors).violations}
    assert "planarity_donation" not in kinds, "the planarity rung must NEVER produce a Violation"
    assert "dihedral" not in kinds
    assert coord.donor_fold(stitched, donors=donors).planarity >= 0.0
