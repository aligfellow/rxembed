"""The sp2-donor coplanarity cap (``donor_orient._coplanar_donor`` -> ``cons.coplanar``). RDKit + UFF.

An sp2 donor binds from an in-plane sigma lone pair, but the surrogate strips the M-donor bond so UFF lets the metal
drift out of plane. A SOFT dihedral cap pulls it back toward the census p95 (``CENSUS_OOP_P95`` = 40 deg) without
delta-spiking the spread. These pin which donors get it (element-agnostic: N/O incl. ISOLATED carbonyl/imine, C,
S; not sp3/sp/haptic), that it is soft, that it never freezes a twisting aryl, and that it is a no-op off metal.
"""

from __future__ import annotations

import numpy as np
import pytest

import rxembed as rx
from rxembed import geometry as geo
from rxembed.rdkit_embed.constraints import donor_orient as DO  # noqa: N812
from rxembed.rdkit_embed.constraints import metal as M  # noqa: N812
from rxembed.rdkit_embed.constraints.base import Constraints

# the henry Ni(II) amidate: a carboxylate O donor (perm 1, one heavy neighbour) AND an N-aryl amidate N donor
# (perm 2, two heavy neighbours) on the same metal — one fixture exercises both permutations.
HENRY = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# a side-on eta2 alkyne on the same henry scaffold — a haptic donor bonded to a co-donor
_ETA2_SMI = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# a Ni(II) thiosemicarbazone: the thione C=S SULFUR is a conjugated sp2 donor the old {7,8} N/O list dropped.
_THIONE_SMI = "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# an aryl-carbanion (aromatic ipso CARBON) donor — conjugated sp2, also dropped by the {7,8} list
_ARYL_CARBANION_SMI = "[c-]1ccccc1[Pd]2(Cl)<-[NH2]CC[NH2]->2"
# an ISOLATED acetone O donor (Lewis-acid-activated ketone): sp2, in-plane lone pair, but RDKit marks its C=O
# `GetIsConjugated()==False` — the case a conjugation-gated predicate dropped (R3). Pyridine is the conjugated
# co-donor, capped either way, so any drop is the ketone O's alone.
_KETONE_SMI = "CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1"
# a bis(acetaldimine) Zn: two ISOLATED imine N donors (RDKit: C=N not conjugated) — coplanar came back EMPTY (R3).
_IMINE_SMI = "C/C=N->[Zn](Cl)(Cl)<-N=C/C"
_TWO_N = 2  # the bis(acetaldimine) fixture's imine-donor count


def _oop(mol, cid, i, j, k, w):
    """Angle (deg) of atom i (the metal) out of the plane through j, k, w — 0 when the metal is coplanar."""
    p = mol.GetConformer(cid).GetPositions()
    n = np.cross(p[k] - p[j], p[w] - p[j])
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return 0.0
    n /= nn
    v = p[i] - p[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(n, v))))))


def _entries(iso):
    return {tuple(e[:4]): e for e in iso.cons.coplanar}


# --- 1. WHICH DONORS GET THE CAP (and the two permutations) -------------------------------------------


def test_a_carboxylate_o_and_an_amidate_n_both_get_a_cap():
    """The henry carboxylate O (perm 1) and amidate N (perm 2) each get coplanar entries on the same metal."""
    iso = rx.metal(HENRY, "square_planar")[0]
    mol = iso.mol
    o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
    n = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "N")
    donors_capped = {e[1] for e in iso.cons.coplanar}
    assert o in donors_capped, "the conjugated carboxylate O donor must get a coplanarity cap"
    assert n in donors_capped, "the conjugated amidate N donor must get a coplanarity cap"


def test_a_one_heavy_neighbour_o_states_its_one_plane_once_against_the_heaviest_reference():
    """Perm 1: the carboxylate O states ONE entry — its plane is one DOF, and a 2nd reference only restates it.

    C's heavy substituents are rigidly ~180° apart in the M-O-C-X dihedral, so an entry per substituent is the
    same rotational DOF twice: `mechanisms._coplanar_window` hands both the identical window, adding no geometric
    information and only doubling this donor's torsion force constant. That doubling is not free — it made a κ1
    carboxylate override a co-donor's own plane wherever the two couple through the chelate backbone. The
    stiffness a κ1 O has always effectively had is now stated once, in `mechanisms._COPLANAR_FC`.
    """
    iso = rx.metal(HENRY, "square_planar")[0]
    mol = iso.mol
    o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
    c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    entries = [e for e in iso.cons.coplanar if e[1] == o]
    assert len(entries) == 1, "the carboxylate O's plane is ONE DOF and must be stated exactly once"
    _i, _d, k, w, anchor, _cap = entries[0]
    assert k == c, "the plane is defined through the carboxyl carbon"
    subs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1]
    o2 = next(s for s in subs if mol.GetAtomWithIdx(s).GetSymbol() == "O")  # the heteroatom partner
    assert w == o2, "the heaviest substituent (the 2nd O) is the reference — a κ1 carboxylate binds anti to it"
    assert anchor == DO._COPLANAR_ANCHOR, "the 2nd O is the anti (180) reference"


def test_the_two_carboxylate_references_would_have_been_the_same_dof():
    """The redundancy the single entry replaces, pinned: both references give ONE window, not a two-sided trap.

    Guards the deletion above against being undone — the old code's stated reason for a second entry was that the
    two flat-bottomed walls would form "a net restoring force". They do not: they coincide.
    """
    from rdkit.Chem import rdMolTransforms as T

    from rxembed.embed.dispatch import _embed_dispatch
    from rxembed.rdkit_embed.constraints.mechanisms import _coplanar_window

    iso = rx.metal(HENRY, "square_planar")[0]
    mol = iso.mol
    o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
    c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    subs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1]
    # The DG seed, not `rx.embed` (which relaxes): the claim here is about the constraint ALGEBRA — two
    # windows on one rotational DOF — and needs a rigidly sp2 carboxyl carbon to express it on. The seed is
    # anti to within 0.1 deg on all 8 conformers; UFF then pyramidalises that carbon (measured after the embed
    # relax: 136.4-179.9), which is force-field noise against a geometric identity.
    ens = _embed_dispatch(iso, n=8, seed=1)
    for cid in ens.ids:
        conf = ens.mol.GetConformer(cid)
        phis = [T.GetDihedralDeg(conf, iso.metal, o, c, x) for x in subs]
        offset = abs((phis[0] - phis[1] + 540) % 360 - 180)
        assert offset == pytest.approx(180.0, abs=5.0), "C's substituents are not rigidly anti — the DOF differs"
        # both windows, expressed in the FIRST dihedral's coordinate, must coincide
        wins = [_coplanar_window(p, DO._COPLANAR_CAP) for p in phis]
        shift = phis[0] - phis[1]
        a = [(x + 540) % 360 - 180 for x in wins[0]]
        b = [((x + shift) + 540) % 360 - 180 for x in wins[1]]
        assert min(abs(a[0] - b[0]), 360 - abs(a[0] - b[0])) < 5.0, "the two references gave DIFFERENT windows"
        assert min(abs(a[1] - b[1]), 360 - abs(a[1] - b[1])) < 5.0, "the two references gave DIFFERENT windows"


def test_a_two_heavy_neighbour_n_uses_the_improper_of_its_own_substituents():
    """Perm 2: the amidate N is held by ONE improper M-N-X-Y over its own two DIRECT heavy substituents."""
    iso = rx.metal(HENRY, "square_planar")[0]
    mol = iso.mol
    n = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "N")
    entries = [e for e in iso.cons.coplanar if e[1] == n]
    assert len(entries) == 1, "the 2-substituent N uses a single improper, not one dihedral per substituent"
    _i, _n, k, w, _anc, _cap = entries[0]
    heavy = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1}
    assert {k, w} == heavy, "the improper's plane atoms are the N's own two DIRECT heavy substituents"


@pytest.mark.parametrize(
    ("name", "smi", "geometry"),
    [
        ("sp3 amine", "CCN[Pd](Cl)Cl", "square_planar"),  # sp3 N — no pi-plane
        ("sp nitrile", "CC#N[Pd](Cl)Cl", "square_planar"),  # sp N — linear, held end-on by _orient_donor
        # a side-on eta2 alkyne donor (bonded to a co-donor — off any lone-pair axis)
        ("side-on eta2", _ETA2_SMI, "square_planar"),
    ],
)
def test_a_non_conjugated_or_haptic_or_sp_donor_gets_no_cap(name, smi, geometry):
    """Only an sp2 in-plane donor gets a cap — an sp3 amine, sp nitrile, or side-on eta2 donor gets none.

    The rule is the derived, element-agnostic `metal.inplane_sp2_donor`, not an N/O element list: every
    capped donor must satisfy it, and no haptic donor (bonded to a co-donor) may be capped.
    """
    for iso in rx.metal(smi, geometry):
        for _i, d, _k, _w, _anc, _cap in iso.cons.coplanar:
            a = iso.mol.GetAtomWithIdx(d)
            assert DO.inplane_sp2_donor(iso.mol, d), f"{name}: a non-sp2 donor was capped"
            haptic = any(nb.GetIdx() in set(iso.donors) for nb in a.GetNeighbors())
            assert not haptic, f"{name}: a haptic donor was capped"


# --- 1b. THE DERIVED PREDICATE: donors the {7,8} N/O list dropped (carbon, sulfur) ---------------------


def test_a_thione_sulfur_donor_gets_the_cap():
    """A thione C=S sulfur (conjugated sp2, element 16) gets a coplanarity cap — the {7,8} list dropped it.

    Red-first: with `_coplanar_donor` gated on `frozenset({7, 8})` the S was never capped and its metal relaxed
    to ~26 deg median / ~62 deg max out of its own C=S plane (measured, henry thiosemicarbazone).
    """
    for iso in rx.metal(_THIONE_SMI, "square_planar"):
        s = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "S")
        assert DO.inplane_sp2_donor(iso.mol, s), "the thione S is an sp2 in-plane donor"
        assert any(e[1] == s for e in iso.cons.coplanar), "the thione S donor must receive a coplanarity cap"


def test_an_aryl_carbanion_carbon_donor_gets_the_cap():
    """An aryl-carbanion ipso CARBON (aromatic, conjugated sp2, element 6) gets a cap — the {7,8} list dropped it."""
    iso = rx.metal(_ARYL_CARBANION_SMI, "square_planar")[0]
    c = next(
        d
        for d in iso.donors
        if iso.mol.GetAtomWithIdx(d).GetSymbol() == "C" and iso.mol.GetAtomWithIdx(d).GetIsAromatic()
    )
    assert DO.inplane_sp2_donor(iso.mol, c), "the aryl-carbanion ipso C is an sp2 in-plane donor"
    entries = [e for e in iso.cons.coplanar if e[1] == c]
    assert entries, "the aryl-carbanion C donor must receive a coplanarity cap"
    _i, _c, k, w, _anc, _cap = entries[0]  # two heavy (ortho) neighbours -> the improper over its own substituents
    orthos = {nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetAtomicNum() > 1}
    assert {k, w} <= orthos, "the aryl carbanion's improper is over its own two ring neighbours"


def test_the_uncalibrated_thione_cap_is_decoupled_from_the_fold_window():
    """('S', SP2) has no `_FOLD_WINDOW` census row (corpus n=0 conjugated) — the fold gate/wall abstain, cap fires.

    The coplanarity enforcement reads only `cons.coplanar` (an FF torsion), NOT `_FOLD_WINDOW`, so an uncalibrated
    donor still gets held. Guards that the census abstention and the cap stay decoupled.
    """
    assert ("S", DO._SP2) not in DO._FOLD_WINDOW, "('S',SP2) must stay uncalibrated (corpus has no conjugated one)"
    assert ("S", DO._SP2) not in DO._FOLD_WALL_FLOOR, "an uncalibrated class gets no fold wall"
    iso = rx.metal(_THIONE_SMI, "square_planar")[0]
    s = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "S")
    # no M-S-X fold-wall angle is written (the wall abstains), yet the cap is present (decoupled)
    assert not any(k[0] == iso.metal and k[1] == s for k in iso.cons.angles), "the fold wall must abstain on S sp2"
    assert any(e[1] == s for e in iso.cons.coplanar), "the cap must still fire for the uncalibrated thione"


# --- 1c. THE DERIVED PREDICATE: ISOLATED (non-conjugated) sp2 N/O donors — the R3 regression -----------
# A metal binds an sp2 O/N from its IN-PLANE sigma lone pair whether or not the π system is conjugated. RDKit marks an
# isolated ketone/aldehyde/ketimine C=X `GetIsConjugated()==False`, so a conjugation-gated predicate silently
# dropped the cap these need — Lewis-acid activation of a simple carbonyl and simple imine ligands, in-scope motifs.


def test_an_isolated_ketone_oxygen_donor_gets_the_cap():
    """An ISOLATED ketone O (RDKit: its C=O `GetIsConjugated()==False`) gets a coplanarity cap — the R3 fix.

    Red-first: a conjugation-gated predicate ('sp2 AND in a π system') dropped this. The acetone O is sp2 with an
    in-plane lone pair but not conjugated, so its metal folded out of plane (measured max ~48 deg, past the census
    p95 ~40); the pyridine co-donor (conjugated) is capped either way, so only the isolated ketone O was lost.
    """
    iso = rx.metal(_KETONE_SMI, "square_planar")[0]
    mol = iso.mol
    o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
    conjugated = any(b.GetIsConjugated() for b in mol.GetAtomWithIdx(o).GetBonds())
    assert not conjugated, "fixture must be a NON-conjugated C=O (the case a conjugation test dropped)"
    assert DO.inplane_sp2_donor(mol, o), "an isolated sp2 ketone O is still an in-plane sp2 donor"
    assert any(e[1] == o for e in iso.cons.coplanar), "the isolated ketone O donor must receive a coplanarity cap"


def test_both_isolated_imine_nitrogen_donors_get_the_cap():
    """Both ISOLATED acetaldimine N donors (RDKit: C=N not conjugated) get a coplanarity cap — the R3 fix.

    Red-first: the conjugation test returned `cons.coplanar` EMPTY for this bis(imine); the metal folded to ~70
    deg out of an imine's plane. An isolated ketimine is sp2 with an in-plane lone pair — conjugation isn't needed.
    """
    iso = rx.metal(_IMINE_SMI, "tetrahedral")[0]
    mol = iso.mol
    imines = [d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "N"]
    assert len(imines) == _TWO_N, "the bis(acetaldimine) fixture has two N donors"
    capped = {e[1] for e in iso.cons.coplanar}
    for n in imines:
        assert not any(b.GetIsConjugated() for b in mol.GetAtomWithIdx(n).GetBonds()), "must be a NON-conjugated C=N"
        assert DO.inplane_sp2_donor(mol, n), "an isolated sp2 imine N is still an in-plane sp2 donor"
        assert n in capped, "each isolated imine N donor must receive a coplanarity cap"


def test_the_isolated_ketone_cap_pulls_the_metal_into_the_carbonyl_plane():
    """The cap actually seats the ketone O's metal near its C=O plane; without it the metal folds out (~48 deg max).

    Measures the DEFECT the cap repairs (not just the entry's presence): relaxed max out-of-plane with the cap ON
    is bounded near the census p95, and strictly below the cap-OFF fold — the harm the R3 drop caused.
    """
    seeds = (1, 7, 13, 21, 42, 99)
    on, off = [], []
    for seed in seeds:
        iso = rx.metal(_KETONE_SMI, "square_planar")[0]
        mol, m = iso.mol, iso.metal
        o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
        c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetAtomicNum() > 1)
        w = max(
            (nb for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
        ).GetIdx()
        assert iso.cons.coplanar, "fixture must carry the ketone-O cap"
        ens = rx.embed(iso, n=8, seed=seed).minimize()
        on += [_oop(ens.mol, cid, m, o, c, w) for cid in ens.ids]
        off_iso = rx.metal(_KETONE_SMI, "square_planar")[0]
        off_iso.cons.coplanar = []
        ens_off = rx.embed(off_iso, n=8, seed=seed).minimize()
        off += [_oop(ens_off.mol, cid, m, o, c, w) for cid in ens_off.ids]
    on, off = np.array(on), np.array(off)
    assert on.max() < off.max(), "the cap must remove the isolated ketone O's gross out-of-plane fold"
    assert on.max() <= 50.0, "no conformer may fold far past the census p95 (~40 deg) once capped"


def test_a_frozen_ts_core_metal_gets_no_cap():
    """A fix= frozen-TS-core metal gets no cap — the reacting core already pins the M-donor geometry."""
    import tests.test_connectivity as tc

    isos = rx.metal(tc._MN_H2, "octahedral", center="Mn", fix=tc._MN_H2_RC)
    for iso in isos:
        assert not iso.cons.coplanar, "a frozen-TS-core metal must not receive a coplanarity cap"


# --- 2. IT IS SOFT: reduces the out-of-plane but keeps the census spread -------------------------------


def _relaxed_oop(coplanar_on, seeds=(1, 7, 13, 21, 42, 99, 123, 7777)):
    """Relaxed metal out-of-plane (henry carboxylate + amidate N) over seeds; `coplanar_on=False` is the baseline.

    The out-of-plane is BIMODAL (a near-plane cluster + a ~40° cap-edge cluster), so its median and gross tail are
    unstable below ~8 seeds — sample enough to measure the cap effect, not the sampling draw (a 2-seed snapshot
    flipped the amidate median from 1.8 to 14 deg on an unrelated seed change).
    """
    carbox, amid = [], []
    for seed in seeds:
        iso = rx.metal(HENRY, "square_planar")[0]
        mol, m = iso.mol, iso.metal
        o = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "O")
        c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
        o2 = next(x.GetIdx() for x in mol.GetAtomWithIdx(c).GetNeighbors() if x.GetSymbol() == "O" and x.GetIdx() != o)
        n = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "N")
        nh = [nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1]
        if not coplanar_on:
            iso.cons.coplanar = []
        ens = rx.embed(iso, n=12, seed=seed).minimize()
        carbox += [_oop(ens.mol, cid, m, o, c, o2) for cid in ens.ids]
        amid += [_oop(ens.mol, cid, m, n, nh[0], nh[1]) for cid in ens.ids]
    return np.array(carbox), np.array(amid)


def test_the_cap_pulls_the_metal_in_plane_while_keeping_a_real_spread():
    """WITH the cap the metal sits nearer each donor's plane than without, yet the distribution keeps its spread."""
    on_c, on_n = _relaxed_oop(coplanar_on=True)
    _off_c, off_n = _relaxed_oop(coplanar_on=False)

    # the amidate N (a single improper) is the clean census-like case: the median drops and the gross tail is capped
    assert np.median(on_n) < np.median(off_n) + 1.0, "the cap must not make the amidate N worse"
    assert on_n.max() < off_n.max(), "the cap must remove the amidate N's gross out-of-plane tail"
    assert on_n.max() <= 50.0, "no conformer may sit far past the census p95 (~40 deg) once capped"

    # NOT a delta-spike: a real spread survives on BOTH donors (some conformers in-plane, some out toward the tail)
    for name, arr in (("carboxylate", on_c), ("amidate", on_n)):
        assert arr.min() < 10.0, f"{name}: the cap must let some conformers reach the plane"
        assert arr.max() > 15.0, f"{name}: the cap must NOT delta-spike the whole ensemble to the plane"
        assert arr.std() > 5.0, f"{name}: the census spread was collapsed — this is the overfit the window forbids"

    # the carboxylate's gross out-of-plane tail is capped too (no conformer rides far past the census)
    assert on_c.max() <= 50.0, "the carboxylate cap must bound the out-of-plane at ~the census p95"


def test_the_phenyl_ring_twist_stays_free():
    """The cap uses the ipso carbon only, so the N-aryl phenyl twist about the N-ipso bond keeps its full range."""
    from rdkit.Chem import rdMolTransforms as T

    iso = rx.metal(HENRY, "square_planar")[0]
    mol = iso.mol
    n = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "N")
    n_nbrs = list(mol.GetAtomWithIdx(n).GetNeighbors())
    ipso = next(nb.GetIdx() for nb in n_nbrs if nb.GetIsAromatic())
    other = next(nb.GetIdx() for nb in n_nbrs if nb.GetIdx() != ipso and nb.GetAtomicNum() > 1)
    ipso_nbrs = mol.GetAtomWithIdx(ipso).GetNeighbors()
    ortho = next(nb.GetIdx() for nb in ipso_nbrs if nb.GetIdx() != n and nb.GetIsAromatic())
    ens = rx.embed(iso, n=12, seed=1).minimize()
    twist = [abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), other, n, ipso, ortho)) for cid in ens.ids]
    assert max(twist) - min(twist) > 30.0, "the phenyl twist was frozen — the cap must not reach into the ring"


# --- 3. PLUMBING: relaxed() carries it, census provenance, organic no-op --------------------------------


def test_relaxed_carries_the_coplanar_field():
    """``Constraints.relaxed()`` copies (not aliases) ``coplanar`` — it is structure, not a releasable NCI grip."""
    c = Constraints()
    c.coplanar.append((0, 1, 2, 3, 180.0, 45.0))
    assert c.relaxed().coplanar == c.coplanar, "relaxed() dropped the coplanarity cap"
    assert c.relaxed().coplanar is not c.coplanar, "relaxed() must copy, not alias, the coplanar list"


def test_the_mc_settle_carries_the_coplanar_field():
    """The per-bin settle must relax WITH the cap — it drives the metal out of plane without it.

    `_settle_seeds` rebuilds a partial Constraints per bin (distances/angles are re-derived at that bin's
    fraction, so they start empty) and used to hand-list the fields it carried, omitting `coplanar`. Since
    `restrained_uff` reads that field, `rx.metal(...).mc()` ran a stiff 1e4 relax with the sp2-donor torsion
    absent. Measured on henry Ni(II) from an identical seed geometry (max 72.2 deg out of plane): settling
    without the cap gave max 94.3 / mean 27.2, with it max 90.3 / mean 23.0.
    """
    from rxembed.pipeline import _refine

    iso = rx.metal(HENRY, "square_planar")[0]
    ens = rx.embed(iso, n=4, seed=1)
    assert ens.cons.coplanar, "fixture must carry a cap for this test to mean anything"

    # Assert on what the RELAX is handed, not on `ens.cons`: the settle builds a fresh partial Constraints per
    # bin and passes that to `restrained_uff`, so the ensemble's own field is unchanged either way.
    seen = []
    real = _refine.restrained_uff

    def spy(mol, cons, *a, **kw):
        seen.append(cons)
        return real(mol, cons, *a, **kw)

    _refine.restrained_uff = spy
    try:
        ens._settle_seeds(bins=2)
    finally:
        _refine.restrained_uff = real

    assert seen, "the settle never reached the relax"
    assert all(c.coplanar for c in seen), "the settle relaxed with the coplanarity cap absent"


def test_the_census_provenance_is_documented():
    """The cap's census percentile provenance is in a docstring, not a bare magic number."""
    import inspect

    src = inspect.getsource(DO)
    assert "p95" in src.lower(), "the census p95 provenance of the cap must be documented"
    assert "census" in src.lower(), "the census provenance of the cap must be documented"
    assert DO.CENSUS_OOP_P95 == pytest.approx(40.0), "the cap tracks the census p95 (O 42 / N 37)"


def test_an_organic_system_is_a_strict_no_op():
    """No metal -> no conjugated-donor cap -> ``cons.coplanar`` empty, and the FF path is bit-identical."""
    ens = rx.embed("CC(=O)Oc1ccccc1C(=O)O", n=6, seed=1).minimize()
    assert not ens.cons.coplanar, "an organic system must never populate the coplanarity cap"


# --- 4. THE REDUNDANT-CAP SKIP: FF-only, a co-donor already shares the plane -------------------------
# A plane is pinned to the metal by TWO coplanar contacts (two M-D distances + the L-M-L bite angle), both of
# which the polyhedron already imposes. So the cap's improper is real information only where the metal meets the
# donor's plane at ONE point. When a co-donor of the same metal lies in the same conjugated sp2 plane (a
# conjugated bidentate: pyridylimine, picolinate, acac), the bite already pins the metal there — the improper is
# a redundant restatement, and on a crowded chelate the independent per-donor FF torsions ADD and fight. So skip
# the FF torsion (only) for such a donor; the DG bound composes (INTERSECT) and stays. `metal.codonor_in_plane`
# is the predicate: a co-donor reachable through an all-sp2 backbone path (element-agnostic, hybridisation only).

# the karoline Ni(II) crowded conjugated chelates (docs/findings/karoline-metal-embed.md): a pyridylimine sharing
# one sp2 plane with an amidate/carboxylate. Cases 2/3 flagged 59-65% from the redundant caps fighting.
_CASE2 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE3 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE4 = "CCOC1=[O]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[n]2c[nH]c(C)c21"  # imidazole ester
# picolinate·ethylenediamine·pyridine Ni: a RIGID conjugated bidentate whose two donors are skipped, yet held
# in plane by the backbone alone (the control that shows the skip is not a geometry regression).
_PICO = "O=C1[O-]->[Ni+2]2(<-[NH2]CC[NH2]->2)<-n2ccccc21"


class _TorsionSpy:
    """Record the ``(i, j, k, w)`` of every UFF torsion constraint an ``ff_terms`` writer emits."""

    def __init__(self):
        self.torsions = []

    def UFFAddTorsionConstraint(self, i, j, k, w, relative, lo, hi, fc):  # noqa: N802 (RDKit FF API name)
        self.torsions.append((i, j, k, w))


def _emitted_cap_donors(iso, seed=1):
    """Donor indices for which ``Coplanar.ff_terms`` actually writes a torsion on a seed conformer of ``iso``."""
    from rxembed.rdkit_embed.constraints.mechanisms import Coplanar

    ens = rx.embed(iso, n=1, seed=seed)
    # `_mol` (the internal working mol), NOT `.mol`: `Coplanar.ff_terms` reads `conf.GetOwningMol()` to decide
    # plane-sharing, so it must see the bond-less surrogate the FF relaxes — not `.mol`'s finalized graph (real
    # metal + DATIVE M-L bonds), whose extra bonds would change which caps are judged redundant.
    conf = ens._mol.GetConformer(ens.ids[0])
    spy = _TorsionSpy()
    Coplanar().ff_terms(spy, ens.cons, conf, 1e4)  # fc is unused by the coplanar term (it uses _COPLANAR_FC)
    return {j for _i, j, _k, _w in spy.torsions}


def test_the_predicate_flags_a_conjugated_bidentate_donor_not_a_one_contact_one():
    """`codonor_in_plane` marks case2's pyridylimine donors redundant (co-donor in-plane) but not the hinged ones.

    The pyridyl N (15) and imine N (34) share one conjugated sp2 plane, reachable through an all-sp2 backbone
    path — the bite angle already pins the metal there. The carboxylate O (17) reaches its co-donor only across
    an sp3 alpha-carbon hinge, and the anilide N (27) likewise, so each is a genuine one-contact donor: kept.
    """
    iso = rx.metal(_CASE2, "square_planar")[0]
    donors = set(iso.donors)
    redundant = {d for d in donors if DO.codonor_in_plane(iso.mol, d, donors)}
    assert redundant == {15, 34}, "only the conjugated-bidentate pyridylimine donors share a co-donor's plane"


def test_the_controls_are_not_flagged_redundant():
    """HENRY (sp3 hinge) and KETONE (monodentate) present ONE contact each — no donor is flagged redundant."""
    for smi, geometry in ((HENRY, "square_planar"), (_KETONE_SMI, "square_planar")):
        iso = rx.metal(smi, geometry)[0]
        donors = set(iso.donors)
        redundant = {d for d in donors if DO.codonor_in_plane(iso.mol, d, donors)}
        assert not redundant, f"{smi}: a one-contact control must not be flagged as sharing a co-donor's plane"


def test_the_ff_torsion_is_skipped_for_a_plane_shared_donor_and_written_otherwise():
    """RED-FIRST: `Coplanar.ff_terms` writes NO torsion for case2's redundant donors, but keeps the one-contact ones.

    Before the skip every capped donor got a torsion, so {15, 34} appeared in the emitted set. The FF-only skip
    drops exactly those (a co-donor shares their plane); the one-contact carboxylate O (17) and anilide N (27)
    still get theirs.
    """
    emitted = _emitted_cap_donors(rx.metal(_CASE2, "square_planar")[0])
    assert 15 not in emitted, "the pyridyl N cap is redundant (co-donor in-plane) and must be skipped in the FF"
    assert 34 not in emitted, "the imine N cap is redundant (co-donor in-plane) and must be skipped in the FF"
    assert {17, 27} <= emitted, "the one-contact carboxylate O / anilide N caps must still be written in the FF"


def test_the_controls_keep_every_ff_cap_bit_identical():
    """Bit-identity: HENRY and KETONE emit a torsion for EVERY capped donor — no cap dropped, relax unchanged."""
    for smi, geometry in ((HENRY, "square_planar"), (_KETONE_SMI, "square_planar")):
        iso = rx.metal(smi, geometry)[0]
        capped = {e[1] for e in iso.cons.coplanar}
        assert capped, f"{smi}: fixture must carry caps for this to mean anything"
        assert _emitted_cap_donors(iso) == capped, f"{smi}: a control cap was dropped — the FF is not bit-identical"


def _flag_rate(smi, seeds, n):
    """Fraction of minimized conformers the geometry gate / connectivity flags, pooled over seeds and isomers."""
    from rxembed import metrics as met

    flagged = total = 0
    for seed in seeds:
        for iso in rx.metal(smi, "square_planar"):
            ens = rx.embed(iso, n=n, seed=seed).minimize()
            sphere = sorted({d for ds in ens.sphere.values() for d in ds}) or list(iso.donors)
            metals = set(M.metal_indices(ens.mol))
            for cid in ens.ids:
                rep = geo.check(ens.mol, cid, donors=sphere)
                formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
                total += 1
                if not rep.ok() or formed or broken:
                    flagged += 1
    return flagged / total if total else float("nan")


@pytest.mark.parametrize(("smi", "skip_max"), [(_CASE2, 0.40), (_CASE3, 0.50)])
def test_the_flag_rate_drops_on_a_crowded_conjugated_chelate(smi, skip_max):
    """RED-FIRST: skipping the redundant caps drops the planarity/conjugation flag rate on cases 2/3.

    With the caps fighting (skip disabled) the crowded conjugated backbone twists to compromise between the
    redundant per-donor torsions -> ~59% (case2) / ~65% (case3) flagged. Removing the redundant FF torsions
    lets it relax planar: measured ~22% / ~40%. Verifies the drop AND that it is RED without the skip.
    """
    seeds = (1, 7)
    with_skip = _flag_rate(smi, seeds, n=6)

    orig = DO.codonor_in_plane
    DO.codonor_in_plane = lambda *a, **k: False  # disable the skip -> the old all-caps behaviour (the baseline)
    try:
        no_skip = _flag_rate(smi, seeds, n=6)
    finally:
        DO.codonor_in_plane = orig

    assert no_skip > 0.45, "baseline (caps fighting) must flag heavily — else the test is not exercising the defect"
    assert with_skip < skip_max, "skipping the redundant caps must drop the flag rate below the finding's level"
    assert with_skip < no_skip - 0.12, "the skip must measurably improve the flag rate, not merely not-worsen it"


def test_a_skipped_conjugated_bidentate_stays_in_plane_via_its_backbone():
    """No geometry regression: PICO's skipped donors keep the metal in plane, held by the rigid backbone alone.

    Both picolinate donors are skipped (a co-donor shares their conjugated plane), yet after `minimize` the metal
    still sits ~2 deg out of each donor's plane — the backbone + polyhedron pin it without the per-donor torsion.
    """
    iso0 = rx.metal(_PICO, "square_planar")[0]
    donors = set(iso0.donors)
    skipped = {e for e in iso0.cons.coplanar if DO.codonor_in_plane(iso0.mol, e[1], donors)}
    assert skipped, "PICO's conjugated-bidentate caps must be the skipped ones for this control to mean anything"

    dev = {e[1]: [] for e in skipped}
    for seed in (1, 7, 13):
        iso = rx.metal(_PICO, "square_planar")[0]
        ens = rx.embed(iso, n=8, seed=seed).minimize()
        for i, j, k, w, _anc, _cap in skipped:
            dev[j] += [_oop(ens.mol, cid, i, j, k, w) for cid in ens.ids]
    for j, vals in dev.items():
        assert np.median(vals) < 10.0, f"skipped donor {j}: the backbone failed to hold the metal in plane"


def test_skipping_the_ester_cap_does_not_collapse_the_ester():
    """case4's plane-locked ester O cap is skipped, and the ester O-C-O does NOT collapse (stays ~0/N, cf 0/40)."""
    iso_set = rx.metal(_CASE4, "square_planar")
    assert len(iso_set) > 3, "case4 must expose the iso3 arrangement that historically collapsed"
    iso = iso_set[3]  # the O4/N23/C16/O6 arrangement (the 'epoxide' collapse case)
    collapses = 0
    for s in range(12):
        ens = rx.embed(iso, n=1, seed=0xF00D + s).minimize()
        if not ens.ids:
            continue
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        v1, v2 = pos[2] - pos[3], pos[4] - pos[3]  # the ester O2-C3-O4 angle
        oco = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        if oco < 90.0:
            collapses += 1
    assert collapses == 0, "skipping the ester cap must not reintroduce the silent O-C-O collapse"
