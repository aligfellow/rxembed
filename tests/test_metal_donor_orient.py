"""`metal_donor_orient`: the soft holds that replace the UFF terms the stripped M-donor bond removed.

Two of them. `_orient_donor` walls every substituent (heavy and proton) of a census-calibrated donor away from
the metal, and abstains entirely on an uncalibrated (element, hybridisation) class. `_coplanar_donor` keeps a
metal in an sp2 donor's own π-plane with a soft dihedral cap; element-agnostic, derived from
`inplane_sp2_donor`, and skipped in the force field (only) where a co-donor already shares that plane, because
the bite angle pins the metal there and two independent torsions fight.

RDKit + UFF, no xtb.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as T
from rdkit.Geometry import Point3D

import rxembed.pipeline as rx
from rxembed import metal_donor_orient as DO  # noqa: N812
from rxembed.constraints import Constraints

_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic
_MN_H2_RC = [1, 5, 63, 64, 65, 66]  # its reacting core

# the N-bound Ni(II) linkage isomer: a carboxylate O donor (one heavy neighbour) and an amidate N donor (two
# heavy neighbours) on the same metal: one fixture exercises both `_coplanar_donor` permutations.
NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
_ETA2_SMI = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# a Ni(II) thiosemicarbazone: the thione C=S SULFUR is a conjugated sp2 donor the old {7, 8} N/O list dropped
_THIONE_SMI = "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# an aryl-carbanion (aromatic ipso CARBON) donor; conjugated sp2, also dropped by the {7, 8} list
_ARYL_CARBANION_SMI = "[c-]1ccccc1[Pd]2(Cl)<-[NH2]CC[NH2]->2"
# an isolated acetone O donor: sp2 with an in-plane lone pair, but RDKit marks its C=O not conjugated. The
# pyridine co-donor is conjugated and capped either way, so any drop is the ketone O's alone.
_KETONE_SMI = "CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1"
_TWO = 2
_WALL_SLACK = 1.0  # deg: a UFF torsion constraint is a penalty, not a hard wall, so a minimum riding the cap
# edge settles a hair outside it. Wide enough for that, far narrower than any fold the cap exists to stop.


# duplicated from tests/pipeline/test_metal_perceive.py: two files need it, and a shared
# test-helper module is what this project does not do.
_FE_C, _C_O, _FE_H = 1.80, 1.13, 1.55  # Å: the recorded FeH2(CO)4 bond lengths
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]  # the 4 equatorial vertices; the 2 hydrides go on ±z


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


def _oop(mol, cid, i, j, k, w):
    """Angle (deg) of atom i (the metal) out of the plane through j, k, w; 0 when the metal is coplanar."""
    p = mol.GetConformer(cid).GetPositions()
    n = np.cross(p[k] - p[j], p[w] - p[j])
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return 0.0
    n /= nn
    v = p[i] - p[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(n, v))))))


def _donor(iso, symbol):
    return next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == symbol)


def _capped(iso):
    return {e[1] for e in iso.cons.coplanar}


class _TorsionSpy:
    """Record every UFF torsion constraint an `ff_terms` writer emits, as `(i, j, k, w, lo, hi, fc)`."""

    def __init__(self):
        self.torsions = []

    def UFFAddTorsionConstraint(self, i, j, k, w, relative, lo, hi, fc):  # noqa: N802 (RDKit FF API name)
        self.torsions.append((i, j, k, w, lo, hi, fc))


def _emitted_caps(iso, seed=1):
    """Every torsion `Coplanar.ff_terms` actually writes on a seed conformer of `iso`."""
    from rxembed.mechanisms import Coplanar

    ens = rx.embed(iso, n=1, seed=seed)
    # `_mol`, not `.mol`: `ff_terms` reads `conf.GetOwningMol()` to decide plane-sharing, so it must see the
    # bond-less surrogate the FF relaxes; `.mol`'s dative M-L bonds would change which caps look redundant.
    spy = _TorsionSpy()
    Coplanar().ff_terms(spy, ens.cons, ens._mol.GetConformer(ens.ids[0]), 1e4)
    return spy.torsions


def _emitted_cap_donors(iso, seed=1):
    """Donor indices for which `Coplanar.ff_terms` actually writes a torsion on a seed conformer of `iso`."""
    return {t[1] for t in _emitted_caps(iso, seed)}


# --- the orientation wall: calibrated classes only -------------------------------------------------------


@pytest.mark.parametrize(
    ("smi", "walled"),
    [
        ("CCN[Pd](Cl)Cl", True),  # amine N sp3: a calibrated class, and its PROTON is walled (the sp3-amine fix)
        ("O->[Pd](Cl)Cl", False),  # aqua O sp3 is UNcalibrated (n < 6): the wall abstains, as the fold gate does
    ],
)
def test_a_calibrated_donors_protons_are_walled_and_an_uncalibrated_class_abstains(smi, walled):
    """An (M, donor, H) angle window exists iff the donor's (element, hybridisation) class is census-calibrated.

    Walling the proton is the fix for an sp3 amine that used to fold an H onto the metal over its lone pair;
    no rule walled an N/O/C sp3 proton before.
    """
    iso = rx.metal(smi, "square_planar").select(index=0)
    protons = [k for k in iso.cons.angles if k[0] == iso.metal and iso.mol.GetAtomWithIdx(k[2]).GetAtomicNum() == 1]
    assert bool(protons) == walled, f"{smi}: {len(protons)} proton walls, expected {'some' if walled else 'none'}"


def test_an_sp3_amine_donor_does_not_fold_a_proton_onto_the_metal():
    """Every M-N-H of a metal-bound [NH2] amine stays ≥90°: no inversion onto the lone pair.

    Red-first: with the wall heavy-only, min M-N-H reached ~65° and 3-5 of 33 conformers fully inverted, and the
    fold GATE is heavy-substituent so it shipped gate-clean. This is the embed->minimize (UFF, no xtb) degrade
    path: no calculator re-splays the fold, so the seed hold must.
    """
    smi = "CCNC1N[NH2]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[S]=1"
    iso0 = rx.metal(smi, "square_planar")[0]
    n5 = next(
        d
        for d in iso0.donors
        if iso0.mol.GetAtomWithIdx(d).GetSymbol() == "N"
        and sum(1 for nb in iso0.mol.GetAtomWithIdx(d).GetNeighbors() if nb.GetAtomicNum() == 1) >= _TWO
    )
    hs = [nb.GetIdx() for nb in iso0.mol.GetAtomWithIdx(n5).GetNeighbors() if nb.GetAtomicNum() == 1]
    assert len(hs) >= _TWO, "the fixture's amine donor must carry two protons for this to mean anything"

    angs = []
    for seed in (1, 5, 8):
        ens = rx.embed(rx.metal(smi, "square_planar")[0], n=2, seed=seed).minimize()
        for cid in ens.ids:
            conf = ens.mol.GetConformer(cid)
            angs += [T.GetAngleDeg(conf, int(iso0.metal), int(n5), int(h)) for h in hs]
    assert angs, "no amine protons measured: the embed produced nothing"
    assert min(angs) >= 90.0, f"an sp3 amine proton folded onto the metal (min M-N-H {min(angs):.1f}°)"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_slow_inverting_phosphines_protons_stay_splayed_on_both_input_paths(tmp_path):
    """A primary phosphine's M-P-H stays >90° from SMILES and again after the geometry is fed back through xyz.

    P/As/Sb inversion is a barrier a local optimiser cannot cross, so the hand the seed picks is the hand that
    ships; on the from-geometry path too, where perception rather than the SMILES supplies the donor.
    """

    def m_d_h(ens, metal, donors):
        return [
            T.GetAngleDeg(ens.mol.GetConformer(cid), int(metal), int(d), h.GetIdx())
            for d in donors
            for h in ens.mol.GetAtomWithIdx(int(d)).GetNeighbors()
            if h.GetAtomicNum() == 1
            for cid in ens.ids
        ]

    iso = rx.metal("P->[Pd](Cl)Cl", "square_planar").select(index=0)
    ens = rx.embed(iso, n=6, seed=3).minimize()
    assert ens.n >= 1
    angs = m_d_h(ens, iso.metal, iso.donors)
    assert angs, "no phosphine protons to check: the fixture is wrong"
    assert min(angs) > 90.0, f"SMILES path: phosphine proton folded to {min(angs):.1f}° of the metal"

    xyz = tmp_path / "seed.xyz"
    ens.lowest(1).dump(str(xyz))
    iso2 = rx.metal(str(xyz), "square_planar", center="Pd").select(index=0)
    ens2 = rx.embed(iso2, n=6, seed=3).minimize()
    assert ens2.n >= 1
    angs2 = m_d_h(ens2, iso2.metal, iso2.donors)
    assert min(angs2) > 90.0, f"from-geometry path: phosphine proton folded to {min(angs2):.1f}°"


def test_hybridisation_is_classified_on_the_metal_stripped_graph():
    """A carbonyl carbon types sp on its own graph; a hydride gets no class at all.

    RDKit mis-types the carbon with the dative bond in, so the coordination under test would otherwise decide
    the class used to judge it.
    """
    mol, donors = feh2co4()
    hyb = DO._stripped_hybridisation(mol)
    for c in donors[:4]:
        assert hyb[c] == Chem.HybridizationType.SP
    for h in donors[4:]:
        assert h not in hyb, "a hydride has no donation axis and must never be classified"


# --- the coplanarity cap: which donors get one -----------------------------------------------------------


def test_the_two_coplanar_permutations_each_state_exactly_one_plane():
    """A 1-heavy-neighbour O uses one dihedral anti to the heaviest reference; a 2-heavy N uses one improper.

    Both are one rotational DOF, so a second entry adds no geometry and only doubles that donor's torsion force
    constant: a doubling that made a κ1 carboxylate override a co-donor's plane wherever the two couple
    through the chelate backbone. (The two carboxyl references are rigidly ~180° apart, so the "net restoring
    force" argument for a second entry is false: the windows coincide.)
    """
    iso = rx.metal(NI_N, "square_planar")[0]
    mol = iso.mol
    o, n = _donor(iso, "O"), _donor(iso, "N")
    assert {o, n} <= _capped(iso), "both conjugated donors of the N-bound isomer must be capped"

    ((_i, _d, k, w, anchor, _cap),) = [e for e in iso.cons.coplanar if e[1] == o]
    c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    assert k == c, "the plane is defined through the carboxyl carbon"
    subs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1]
    assert w == next(s for s in subs if mol.GetAtomWithIdx(s).GetSymbol() == "O"), "the 2nd O is the reference"
    assert anchor == DO._COPLANAR_ANCHOR, "a κ1 carboxylate binds ANTI to its partner O"

    ((_i, _n, k, w, _anc, _cap),) = [e for e in iso.cons.coplanar if e[1] == n]
    heavy = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1}
    assert {k, w} == heavy, "the improper's plane atoms are the N's own two DIRECT heavy substituents"


def test_only_an_inplane_sp2_donor_is_capped():
    """Every capped donor satisfies `inplane_sp2_donor` and none is haptic: the rule, not an N/O element list.

    The three fixtures are the ways a donor can fail the rule: no π-plane (sp3 amine), an sp donor held end-on
    instead (nitrile), and a face bonded to its own co-donor (side-on η²). An organic molecule has no donors at
    all, so `coplanar` is empty there for the same reason.
    """
    for name, smi, geometry in (
        ("sp3 amine", "CCN[Pd](Cl)Cl", "square_planar"),
        ("sp nitrile", "CC#N[Pd](Cl)Cl", "square_planar"),
        ("side-on eta2", _ETA2_SMI, "square_planar"),
    ):
        for iso in rx.metal(smi, geometry):
            for _i, d, _k, _w, _anc, _cap in iso.cons.coplanar:
                assert DO.inplane_sp2_donor(iso.mol, d), f"{name}: a non-sp2 donor was capped"
                haptic = any(nb.GetIdx() in set(iso.donors) for nb in iso.mol.GetAtomWithIdx(d).GetNeighbors())
                assert not haptic, f"{name}: a haptic donor was capped"


@pytest.mark.parametrize(
    ("name", "smi", "symbol", "conjugated"),
    [
        ("thione S", _THIONE_SMI, "S", True),  # element 16: the {7, 8} N/O list dropped it
        ("aryl carbanion C", _ARYL_CARBANION_SMI, "C", True),  # element 6; likewise
        ("isolated ketone O", _KETONE_SMI, "O", False),  # the case a CONJUGATION-gated predicate dropped
    ],
    ids=["thione-S", "aryl-carbanion-C", "isolated-ketone-O"],
)
def test_the_derived_predicate_caps_donors_an_element_or_conjugation_test_dropped(name, smi, symbol, conjugated):
    """A conjugated S / an aromatic ipso C / a NON-conjugated ketone O are each in-plane sp2 donors, and capped.

    A metal binds an sp2 O/N/S/C from its IN-PLANE sigma lone pair whether or not the pi system is conjugated,
    so neither an element set nor `GetIsConjugated` is the rule. Measured harm of each drop: the thione's metal
    relaxed ~26° median / ~62° max out of its C=S plane; the isolated ketone O's reached ~48°, past the census
    p95 (~40°). The `conjugated` column is a fixture assertion; if RDKit's flag changed, the case would stop
    being the one the predicate was derived for.
    """
    iso = rx.metal(smi, "square_planar")[0]
    d = next(
        x
        for x in iso.donors
        if iso.mol.GetAtomWithIdx(x).GetSymbol() == symbol
        and (symbol != "C" or iso.mol.GetAtomWithIdx(x).GetIsAromatic())
    )
    assert any(b.GetIsConjugated() for b in iso.mol.GetAtomWithIdx(d).GetBonds()) is conjugated, f"{name}: fixture"
    assert DO.inplane_sp2_donor(iso.mol, d), f"{name} must satisfy the derived in-plane sp2 predicate"
    assert d in _capped(iso), f"{name} must receive a coplanarity cap"


def test_an_uncalibrated_donor_class_still_gets_the_cap():
    """('S', SP2) has no census fold row, so the wall abstains; yet the thione's cap still fires.

    The coplanarity enforcement reads `cons.coplanar` (an FF torsion), never `_FOLD_WINDOW`; the census
    abstention and the cap must stay decoupled.
    """
    assert ("S", DO._SP2) not in DO._FOLD_WINDOW, "('S',SP2) must stay uncalibrated (corpus has no conjugated one)"
    assert ("S", DO._SP2) not in DO._FOLD_WALL_FLOOR
    iso = rx.metal(_THIONE_SMI, "square_planar")[0]
    s = _donor(iso, "S")
    assert not any(k[0] == iso.metal and k[1] == s for k in iso.cons.angles), "the fold wall must abstain on S sp2"
    assert s in _capped(iso)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_frozen_ts_core_metal_gets_no_cap():
    """A `fix=` frozen reacting core already pins the M-donor geometry, so no cap is written."""
    for iso in rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC):
        assert not iso.cons.coplanar


# --- the coplanarity cap is SOFT: a window the relax lands inside, never a pin ---------------------------
# The bound every test below reads is the cap's own declared half-width, taken off the `cons.coplanar` entry.
# The effect sizes it replaces are on-vs-off medians and tails, per donor, over 5-8 seeds.


def test_the_emitted_cap_is_a_one_sided_window_of_its_declared_width_at_its_own_force_constant():
    """`_coplanar_window` opens `cap` deg wide on the seed's own side of the nearest in-plane well.

    Three things, all load-bearing. A window rather than a point is what leaves a real energy free to choose
    inside it. Opening on the seed's side rather than straddling the well is what makes riding the wall land ON
    the plane: a symmetric ±cap window rides out to `cap` off it instead. And the emitted torsion carries
    `_COPLANAR_FC`, never the caller's stiffness: `ff_terms` is handed the relax's current force constant,
    which the stiffness ladder escalates to 1e6, and a cap that rode that ladder would stop being a cap.
    """
    from rxembed import mechanisms
    from rxembed.mechanisms import _coplanar_window

    cap = DO._COPLANAR_CAP
    for phi, well in ((3.0, 0.0), (-3.0, 0.0), (177.0, 180.0), (-177.0, -180.0), (100.0, 180.0)):
        lo, hi = _coplanar_window(phi, cap)
        assert hi - lo == pytest.approx(cap), f"phi={phi}: the window is not `cap` wide: a pin holds nothing softly"
        assert well in (lo, hi), f"phi={phi}: the well {well} is not an EDGE; riding the wall would miss the plane"
        assert (phi >= well) == (hi > well), f"phi={phi}: the window opened on the far side of the well"

    for smi in (NI_N, _KETONE_SMI):
        iso = rx.metal(smi, "square_planar")[0]
        declared = {e[1]: e[5] for e in iso.cons.coplanar}
        emitted = _emitted_caps(iso)  # called at fc=1e4
        assert emitted, f"{smi}: the fixture must emit a cap for this to mean anything"
        for _i, j, _k, _w, lo, hi, fc in emitted:
            assert hi - lo == pytest.approx(declared[j]), f"{smi}: donor {j}'s FF window is not its declared cap"
            assert fc == mechanisms._COPLANAR_FC, f"{smi}: donor {j}'s cap rode the caller's stiffness ladder"


def _cap_deviation(smi, seeds, n):
    """Per-donor deviation (deg) of each CAPPED dihedral from its nearest in-plane well, pooled over `seeds`."""
    out: dict[int, list[float]] = {}
    for seed in seeds:
        iso = rx.metal(smi, "square_planar")[0]
        entries = list(iso.cons.coplanar)
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        for i, j, k, w, _anc, _cap in entries:
            for cid in ens.ids:
                phi = abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), int(i), int(j), int(k), int(w)))
                out.setdefault(j, []).append(min(phi, 180.0 - phi))
    return {j: np.array(v) for j, v in out.items()}


@pytest.mark.parametrize("smi", [NI_N, _KETONE_SMI], ids=["amidate-chelate", "ketone-monodentate"])
def test_the_relax_lands_inside_the_caps_own_window_without_pinning_the_plane(smi):
    """Every relaxed conformer sits inside the declared cap, some reach the plane, and some reach the wall.

    Measured on the dihedral the cap actually constrains, not on a derived out-of-plane proxy, so the bound is
    the constraint's own number and no remembered measurement enters the assertion. Riding the wall is allowed
    a degree of slack: the UFF torsion is a penalty, not a hard wall, so a flat-bottomed minimum settles a hair
    outside. Uncapped, these donors fold well past the window: that is what the cap exists to stop.
    """
    cap = DO._COPLANAR_CAP
    dev = _cap_deviation(smi, (1, 7, 13), n=6)
    assert dev, "no capped donor was measured: the fixture is wrong"
    for j, arr in dev.items():
        assert arr.max() <= cap + _WALL_SLACK, f"donor {j}: the relax broke through its cap ({arr.max():.1f}°)"
        assert arr.min() < cap / 10.0, f"donor {j}: no conformer reached the plane: the window is unreachable"
        assert arr.max() > cap / 2.0, f"donor {j}: the cap delta-spiked the ensemble instead of holding a window"


def test_the_cap_does_not_reach_into_the_aryl_ring_it_anchors_on():
    """The N-aryl phenyl twist keeps a >30° range: the cap uses the ipso carbon, not the whole ring."""
    iso = rx.metal(NI_N, "square_planar")[0]
    mol = iso.mol
    n = _donor(iso, "N")
    n_nbrs = list(mol.GetAtomWithIdx(n).GetNeighbors())
    ipso = next(nb.GetIdx() for nb in n_nbrs if nb.GetIsAromatic())
    other = next(nb.GetIdx() for nb in n_nbrs if nb.GetIdx() != ipso and nb.GetAtomicNum() > 1)
    ring_nbrs = mol.GetAtomWithIdx(ipso).GetNeighbors()
    ortho = next(nb.GetIdx() for nb in ring_nbrs if nb.GetIdx() != n and nb.GetIsAromatic())
    ens = rx.embed(iso, n=8, seed=1).minimize()
    twist = [abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), other, n, ipso, ortho)) for cid in ens.ids]
    assert max(twist) - min(twist) > 30.0, "the phenyl twist was frozen"


def test_the_cap_survives_every_constraints_rebuild():
    """`Constraints.relaxed()` COPIES the field, and the mc per-bin settle relaxes with it present.

    `coplanar` is structure, not a releasable NCI grip, so `relaxed()` must keep it. `_settle_seeds` rebuilds a
    partial `Constraints` per bin and used to hand-list the fields it carried, omitting this one: so
    `rx.metal(...).mc()` ran a stiff 1e4 relax with the sp2-donor torsion absent (max 94.3° vs 90.3° out of
    plane on the N-bound Ni(II) isomer from an identical seed). Asserted on what the RELAX is handed, not on `ens.cons`.
    """
    from rxembed.pipeline.ensemble import _refine

    c = Constraints()
    c.coplanar.append((0, 1, 2, 3, 180.0, 45.0))
    assert c.relaxed().coplanar == c.coplanar, "relaxed() dropped the coplanarity cap"
    assert c.relaxed().coplanar is not c.coplanar, "relaxed() must copy, not alias, the coplanar list"

    ens = rx.embed(rx.metal(NI_N, "square_planar")[0], n=4, seed=1)
    assert ens.cons.coplanar, "the fixture must carry a cap for this to mean anything"
    seen, real = [], _refine.restrained_uff

    def spy(mol, cons, *a, **kw):
        seen.append(cons)
        return real(mol, cons, *a, **kw)

    # ty types every function literal nominally, so no stand-in is ever assignable to what it replaces.
    _refine.restrained_uff = spy  # ty: ignore[invalid-assignment]
    try:
        ens._settle_seeds(bins=2)
    finally:
        _refine.restrained_uff = real
    assert seen, "the settle never reached the relax"
    assert all(x.coplanar for x in seen), "the settle relaxed with the coplanarity cap absent"


# --- the redundant-cap skip: FF-only, where a co-donor already shares the plane --------------------------
# A plane is pinned to the metal by TWO coplanar contacts (two M-D distances + the L-M-L bite), both of which
# the polyhedron already imposes. So the improper is real information only where the metal meets the donor's
# plane at one point. `codonor_in_plane` is the predicate: a co-donor reachable through an all-sp2 backbone
# path (element-agnostic, hybridisation only). The DG bound still composes; only the FF torsion is skipped.

# a crowded Ni(II) conjugated chelate: a pyridylimine sharing one sp2 plane with an amidate.
_CASE2 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE3 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE4 = "CCOC1=[O]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[n]2c[nH]c(C)c21"  # imidazole ester
# picolinate·ethylenediamine Ni: a RIGID conjugated bidentate whose two donors are the skipped ones.
_PICO = "O=C1[O-]->[Ni+2]2(<-[NH2]CC[NH2]->2)<-n2ccccc21"


@pytest.mark.parametrize(("smi", "kept"), [(_CASE2, {17, 27}), (_CASE3, {17})], ids=["case2", "case3"])
def test_the_ff_torsion_is_skipped_only_for_a_plane_shared_donor(smi, kept):
    """The crowded chelates drop the torsions of {15, 34} and keep the sp3-hinged ones; the controls keep every cap.

    The pyridyl N (15) and the imine N (34) share one all-sp2 plane with a co-donor; the sp3-hinged O (17) and
    N (27) do not, and `codonor_in_plane` is asserted directly on all four so the predicate is pinned as well as
    its consumer. Bit-identity on the controls is the other half: NI_N (sp3 hinge) and the monodentate KETONE
    present one contact each, so the skip must be a strict no-op there and the relax unchanged.
    """
    iso = rx.metal(smi, "square_planar")[0]
    donors = set(iso.donors)
    assert {d for d in donors if DO.codonor_in_plane(iso.mol, d, donors)} == {15, 34}

    emitted = _emitted_cap_donors(iso)
    assert 15 not in emitted, "the pyridyl N's redundant (co-donor in-plane) cap was still written"
    assert 34 not in emitted, "the imine N's redundant (co-donor in-plane) cap was still written"
    assert kept <= emitted, "a one-contact cap was wrongly skipped"

    for ctrl in (NI_N, _KETONE_SMI):
        ctrl_iso = rx.metal(ctrl, "square_planar")[0]
        capped = _capped(ctrl_iso)
        assert capped, f"{ctrl}: the control must carry caps for this to mean anything"
        assert _emitted_cap_donors(ctrl_iso) == capped, f"{ctrl}: a control cap was dropped: the FF is not identical"


def test_the_skip_is_ff_only_so_the_dg_bound_still_composes():
    """A skipped donor keeps its `cons.coplanar` entry; `Coplanar.dg_post` reads that list unconditionally.

    Dropping the entry at build time would look identical in the force field and silently take the 1,4 distance
    bound with it, leaving the plane held by nothing during the embed itself. PICO's conjugated bidentate is the
    case: both its donors are skipped, and the geometry is then held by the backbone and the polyhedron alone.
    """
    iso = rx.metal(_PICO, "square_planar")[0]
    donors = set(iso.donors)
    skipped = {e[1] for e in iso.cons.coplanar if DO.codonor_in_plane(iso.mol, e[1], donors)}
    assert skipped, "PICO's conjugated-bidentate caps must be the skipped ones for this control to mean anything"
    assert not skipped & _emitted_cap_donors(iso), "a plane-shared donor still got its FF torsion"
    assert skipped <= _capped(iso), "the skip removed the constraint the DG writer reads, not just the FF term"


def test_skipping_an_ester_cap_does_not_collapse_the_ester():
    """case4's plane-locked ester O cap is skipped and its O-C-O never closes below 90° (was 0/40 collapses)."""
    iso_set = rx.metal(_CASE4, "square_planar")
    assert len(iso_set) > 3, "case4 must expose the iso3 arrangement that historically collapsed"
    iso = iso_set[3]  # the O4/N23/C16/O6 arrangement
    measured = 0
    for s in range(4):
        ens = rx.embed(iso, n=1, seed=0xF00D + s).minimize()
        if not ens.ids:
            continue
        measured += 1
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        v1, v2 = pos[2] - pos[3], pos[4] - pos[3]  # the ester O2-C3-O4 angle
        oco = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        assert oco >= 90.0, f"seed {s}: the ester collapsed to {oco:.1f}°"
    assert measured, "no seed embedded: the collapse guard measured nothing"


def test_a_frozen_metal_silences_the_wall_only_where_the_geometry_is_the_truth():
    """A `fix=`d metal drops every donor's orientation wall on a measured geometry, and keeps it on a modelled one.

    Two different questions wearing one flag. With the input conformer as the distance truth, a frozen metal
    means the reference already says where each donor points and a wall could only fight it. With the fitted
    model there is no such reference, and reading the frozen metal as a skip left a rebuilt ligand with nothing
    orienting it at all: a carbonyl added to a held sphere relaxed to M-C-O 129-148° against its own 155° gate.

    The frozen donors stay skipped either way; that skip is per-donor and is not what moved.
    """

    def walls(lengths):  # (metal, donor, substituent) windows, i.e. `_orient_donor`'s, not the polyhedron's
        iso = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC, lengths=lengths)[0]
        return {k: v for k, v in iso.cons.angles.items() if k[0] == iso.metal and k[1] != iso.metal}

    assert not walls("auto"), "a measured geometry is the orientation truth; a wall on top can only fight it"

    modelled = walls("model")
    assert modelled, "with no geometry to trust, every free donor needs its wall back"
    assert not {k for k in modelled if k[1] in _MN_H2_RC}, "a fix=d donor's own orientation is still the reference's"
    why = "the carbonyl that motivated this must be the sp-carbon wall, or the test is measuring something else"
    assert (1, 61, 3) in modelled, why
    assert modelled[(1, 61, 3)] == (165.0, 180.0), why
