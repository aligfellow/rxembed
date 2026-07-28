"""The 'no permutations tabulated' info-log must fire only when a distinct arrangement is actually lost.

A geometry with no canned permutation list (linear, tetrahedral, CN7/CN8) returns just the input/identity
ordering. Whether that *loses* anything depends on the donors: linear's two sites are swap-equivalent (a
metallocene collapses to two centroid vertices -> linear, one arrangement, nothing lost -> silent), whereas
four distinct donors on a tetrahedron are two enantiomers we do not expand (a real limitation -> warn). The
warning is gated on the distinct-arrangement count derived from the enumerator's own dedup, not a name table.
"""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from tests import test_haptic as th

S, DAT = Chem.BondType.SINGLE, Chem.BondType.DATIVE
_WARN = "coordination-isomer permutations"  # the stable substring of the guard's info-log


def _tetrahedral_four_distinct():
    """Zn(II) with four *distinct* monodentate dative donors (N, O, S, P) -> tetrahedral, two enantiomers.

    All six vertex-pairs share the 109.47 deg angle, so the pairwise element/angle signature is identical for
    both handednesses — only the centre's Λ/Δ chirality separates them, which is exactly what makes this the
    genuine 'two arrangements, none expanded' limitation the warning must still surface.
    """
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(30))
    rw.GetAtomWithIdx(me).SetFormalCharge(2)
    donors = []
    for z, n_h in [(7, 3), (8, 2), (16, 2), (15, 3)]:  # NH3, OH2, SH2, PH3 (dative M<-donor)
        d = rw.AddAtom(Chem.Atom(z))
        for _ in range(n_h):
            rw.AddBond(d, rw.AddAtom(Chem.Atom(1)), S)
        rw.AddBond(me, d, DAT)
        donors.append(d)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for d, v in zip(donors, [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], strict=True):
        p = np.array(v, float) / np.sqrt(3) * 2.1
        conf.SetAtomPosition(d, Point3D(*p))
    for a in m.GetAtoms():  # splay each donor's H's off it deterministically
        if a.GetAtomicNum() == 1:
            base = np.array(conf.GetAtomPosition(a.GetNeighbors()[0].GetIdx()))
            off = np.random.RandomState(a.GetIdx()).randn(3) * 0.3
            conf.SetAtomPosition(a.GetIdx(), Point3D(*(base * 1.4 + off)))
    m.AddConformer(conf)
    return m


def test_metallocene_geometry_does_not_warn(caplog):
    """A ferrocene collapses to two centroid sites -> linear: one arrangement, so the warning is noise -> silent.

    Red on pre-T6 code, where the guard warned unconditionally for every untabulated geometry.
    """
    with caplog.at_level("INFO", logger="rxembed.constraints.metal"):
        isos = rx.metal(th.ferrocene())
    assert isos  # the single ordering still comes back (return unchanged)
    assert isos[0].geometry == "linear"
    assert not any(_WARN in r.message for r in caplog.records), (
        "linear has one arrangement; the 'no permutations tabulated' warning is pure noise here"
    )


def test_tetrahedral_four_distinct_donors_warns(caplog):
    """Four distinct donors on a tetrahedron are two enantiomers we do not expand -> the warning must still fire."""
    with caplog.at_level("INFO", logger="rxembed.constraints.metal"):
        isos = rx.metal(_tetrahedral_four_distinct(), "tetrahedral")
    assert isos  # the single ordering still comes back (return unchanged)
    assert isos[0].geometry == "tetrahedral"
    assert any(_WARN in r.message for r in caplog.records), (
        "four distinct tetrahedral donors are a genuine un-enumerated pair of enantiomers — the user must hear it"
    )
