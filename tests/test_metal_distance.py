"""`metal_distance`: the fitted M-L bond-length model and the anti-overbond floors, measured on real embeds.

`ml_distance` is a periodic fit (element / group / delocalised charge / hapticity) with a P/As/Sb dative cap;
`nondonor_floors` keeps everything that is not a donor out of the metal's coordination sphere, which the
bond-less carbon surrogate cannot do on its own. Both are only observable in the relaxed geometry, so these
embed. The tier boundaries themselves (`overbond_tier`, the report-vs-force-field floors) are pinned in
`tests/pipeline/test_geom_check.py`. RDKit + UFF, no xtb.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem import rdMolTransforms as T

import rxembed.pipeline as rx
from rxembed import metal_distance as D  # noqa: N812
from rxembed import metal_perceive as perceive

# the N-bound Ni(II) linkage isomer, depe backbone: a P donor (capped), an anionic O and an amidate N
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"


def test_a_pnictogen_donor_takes_the_dative_cap_and_a_halide_does_not():
    """A neutral P donor is modelled at `r_M + _SOFT_DONOR_FRAC * r_P`, a contraction; an X-type Cl is not.

    A P/As/Sb covalent radius over-states its dative reach and the published fit has no dative term, so the cap
    is element-class-wide; but applying it to a halide would shorten a bond the fit already gets right. Read
    off `ml_distance` rather than off a relaxed geometry: the realised bond sits inside a ±0.05 Å window either
    way, so no embed can tell which branch the donor took.
    """
    pt = GetPeriodicTable()
    iso = rx.metal(_NI_N, "square_planar")[0]
    q = D.delocalised_charges(iso.mol)
    p = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 15)
    r_ni, r_p = pt.GetRcovalent(28), pt.GetRcovalent(15)
    got = D.ml_distance(iso.mol, iso.metal, p, 28, set(iso.donors), q)
    assert got == pytest.approx(r_ni + D._SOFT_DONOR_FRAC * r_p), "the P donor did not take the dative cap"
    assert got < r_ni + r_p, "the cap must CONTRACT the pnictogen, not lengthen it"

    pdcl = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]
    cl = next(d for d in pdcl.donors if pdcl.mol.GetAtomWithIdx(d).GetAtomicNum() == 17)
    r_pd, r_cl = pt.GetRcovalent(46), pt.GetRcovalent(17)
    got_cl = D.ml_distance(pdcl.mol, pdcl.metal, cl, 46, set(pdcl.donors), D.delocalised_charges(pdcl.mol))
    assert got_cl > r_pd + D._SOFT_DONOR_FRAC * r_cl, "a halide was wrongly given the soft-donor contraction"


def test_every_m_donor_lands_inside_the_window_the_model_stated():
    """A tight-bite bis-chelate embeds with every M-donor inside its own stated window; coordinated, not torn.

    The bound is `cons.distances`, so it moves with the model instead of restating a remembered bond length;
    the slack is the relax's, not the window's. `stereo='free'`: the amidate has an alpha-C the racemic default
    would enumerate, which is a different axis from the embeddability under test.
    """
    iso = rx.metal(_NI_N, "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=3).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        c = ens.mol.GetConformer(cid)
        for d in iso.donors:
            lo, hi = iso.cons.distances[(min(d, iso.metal), max(d, iso.metal))]
            got = T.GetBondLength(c, iso.metal, d)
            assert lo - 0.05 <= got <= hi + 0.05, f"donor {d}: {got:.3f} Å is outside its window ({lo:.3f}, {hi:.3f})"


def test_a_vacant_site_is_not_filled_by_the_ligands_own_backbone():
    """No non-donor heavy atom reaches a bonding distance of the metal, and the coordination set is unchanged.

    The tiered `nondonor_floors` is the only thing holding a backbone out of a vacant vertex: the surrogate is
    bond-less, so nothing else stops the force field folding a ligand into the pocket.
    """
    from rxembed.pipeline import metrics

    pt = GetPeriodicTable()
    for iso in rx.metal("CCN[Pd](Cl)Cl", "square_planar"):
        ens = rx.embed(iso, n=4, seed=1).minimize()
        assert ens.n >= 1
        r_m = pt.GetRcovalent(ens.mol.GetAtomWithIdx(iso.metal).GetAtomicNum())
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            for a in ens.mol.GetAtoms():
                i = a.GetIdx()
                if i == iso.metal or i in iso.donors or a.GetAtomicNum() == 1:
                    continue
                d = float(np.linalg.norm(pos[i] - pos[iso.metal]))
                ratio = d / (r_m + pt.GetRcovalent(a.GetAtomicNum()))
                assert ratio >= D.NEAR_REPORT_RATIO, (
                    f"{iso.label}/conf{cid}: non-donor {a.GetSymbol()}{i} collapsed into the vacant site "
                    f"({d:.3f} Å = {ratio:.3f} x the covalent sum)"
                )
            assert not perceive.metal_overbond(ens.mol, pos, iso.donors)
            assert metrics.coordination_changed(ens.mol, cid, iso.metal, iso.donors) == ([], [])


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_the_floors_survive_the_monte_carlo_search():
    """After `mc()`, a conjugated donor's own carboxyl carbon still never reaches a bonding distance.

    The rotor search moves atoms the embed's bounds matrix no longer constrains, so the floors have to be
    re-imposed by the post-search relax; swinging the carboxyl C into the metal is the failure that pinned it.
    """
    iso = rx.metal(_NI_N, "square_planar")[0]
    o_don = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 8)
    c_carboxyl = next(n.GetIdx() for n in iso.mol.GetAtomWithIdx(o_don).GetNeighbors() if n.GetAtomicNum() == 6)
    searched = rx.embed(iso, n=4).minimize().mc(preset="ensemble").minimize()
    assert searched.n >= 1
    donors = sorted({int(d) for ds in searched.sphere.values() for d in ds})
    pt = GetPeriodicTable()
    r_sum = pt.GetRcovalent(searched.mol.GetAtomWithIdx(iso.metal).GetAtomicNum()) + pt.GetRcovalent(6)
    for cid in searched.ids:
        pos = searched.mol.GetConformer(cid).GetPositions()
        d_mc = float(np.linalg.norm(pos[iso.metal] - pos[c_carboxyl]))
        assert d_mc / r_sum >= D.NEAR_REPORT_RATIO, f"carboxyl C collapsed to a bond at {d_mc:.2f} Å"
        assert not perceive.metal_overbond(searched.mol, pos, donors), "carboxyl C over-bonded the metal in mc"
