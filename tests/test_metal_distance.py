"""`metal_distance`: the fitted M-L bond-length model and the anti-overbond floors, measured on real embeds.

`ml_distance` is a periodic fit (element / group / delocalised charge / hapticity) with a P/As/Sb dative cap;
`nondonor_floors` keeps everything that is not a donor out of the metal's coordination sphere, which the
bond-less carbon surrogate cannot do on its own. Both are only observable in the relaxed geometry, so these
embed. The tier boundaries themselves (`overbond_tier`, the report-vs-force-field floors) are pinned in
`tests/pipeline/test_geom_check.py`. RDKit + UFF, no xtb.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem import rdMolTransforms as T

import rxembed.pipeline as rx
from rxembed import metal_core as core
from rxembed import metal_distance as D  # noqa: N812
from rxembed import metal_perceive as perceive
from rxembed.metal_core import metal_index
from rxembed.metal_donor_orient import _stripped_hybridisation

# the N-bound Ni(II) linkage isomer, depe backbone: a P donor (capped), an anionic O and an amidate N
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
_VANADYL_ACAC = "O=[V+2]12(<-O=C(C)C=C(C)[O-]1)<-O=C(C)C=C(C)[O-]2"  # VO(acac)2: one oxo, four equivalent O
_NITRIDO_OXO_DIAQUA = "N#[Mo+2](=O)(<-[OH2])<-[OH2]"  # all three cells of the ligand-valence rule, on one metal
_NITRIDO_OXO_DIAQUA_IONIC = "[N-3]->[Mo+7](<-[O-2])(<-[OH2])<-[OH2]"  # the same species and total charge
_AQUA_PROTONS = 2  # the two protons that stop `_NITRIDO_OXO_DIAQUA`'s water reading as an oxo
_ACAC_OXYGENS = 4  # what `_VANADYL_ACAC` presents besides its oxo: two bidentate acac
_NITROGEN, _OXYGEN, _MOLYBDENUM = 7, 8, 42


def _ligand_free_contraction(z, group):
    intercept, slope = D._LIGAND_FREE_CONTRACTION[z]
    return intercept + slope * group


def _as_perceived(mol):
    """The same complex as an xyz perception hands it back: every M-donor bond is single and every atom neutral.

    Neither of the two SMILES spellings, and the one production actually meets. It is what makes the terminal
    oxo undecidable from bond order or charge, and decidable only from what its ligand side leaves unsatisfied.
    """
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        if any(a.GetAtomicNum() in core.TRANSITION_METALS for a in (bond.GetBeginAtom(), bond.GetEndAtom())):
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def test_a_pnictogen_donor_takes_the_dative_cap_and_a_halide_does_not():
    pt = GetPeriodicTable()
    iso = rx.metal(_NI_N, "square_planar")[0]
    q = D.delocalised_charges(iso.mol)
    p = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 15)
    r_ni, r_p = pt.GetRcovalent(28), pt.GetRcovalent(15)
    got = D.ml_distance(iso.mol, iso.metal, p, 28, set(iso.donors), q, hyb=_stripped_hybridisation(iso.mol))
    assert got == pytest.approx(r_ni + D._SOFT_DONOR_FRAC * r_p), "the P donor did not take the dative cap"
    assert got < r_ni + r_p, "the cap must CONTRACT the pnictogen, not lengthen it"

    pdcl = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]
    cl = next(d for d in pdcl.donors if pdcl.mol.GetAtomWithIdx(d).GetAtomicNum() == 17)
    r_pd, r_cl = pt.GetRcovalent(46), pt.GetRcovalent(17)
    got_cl = D.ml_distance(
        pdcl.mol,
        pdcl.metal,
        cl,
        46,
        set(pdcl.donors),
        D.delocalised_charges(pdcl.mol),
        hyb=_stripped_hybridisation(pdcl.mol),
    )
    assert got_cl > r_pd + D._SOFT_DONOR_FRAC * r_cl, "a halide was wrongly given the soft-donor contraction"


def test_a_vanadyl_oxo_contracts_while_its_acac_oxygens_keep_one_shared_length(monkeypatch):
    iso = rx.metal(_VANADYL_ACAC, "square_pyramidal")[0]
    q, hyb = D.delocalised_charges(iso.mol), _stripped_hybridisation(iso.mol)
    dset = set(iso.donors)

    def targets():
        return {int(d): D.ml_distance(iso.mol, iso.metal, int(d), iso.real_z, dset, q, hyb=hyb) for d in iso.donors}

    got = targets()
    oxo = [d for d in got if not core.ligand_valence(iso.mol.GetAtomWithIdx(d))]
    acac = [d for d in got if d not in oxo]
    assert len(oxo) == 1, f"expected one ligand-valence-free O on the vanadyl, got {oxo}"
    assert len(acac) == _ACAC_OXYGENS, f"expected four acac oxygens, got {acac}"
    assert len({round(got[d], 9) for d in acac}) == 1, (
        f"the acac oxygens were split: {[round(got[d], 4) for d in acac]} -- the Kekulé artefact is back"
    )
    assert got[oxo[0]] < min(got[d] for d in acac)

    contraction = _ligand_free_contraction(_OXYGEN, 5)
    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {})
    off = targets()
    assert off[oxo[0]] - got[oxo[0]] == pytest.approx(contraction), "the oxo did not take the tmQM contraction"
    assert all(off[d] == got[d] for d in acac), "the term reached a donor whose ligand side already fills it"


def test_the_ligand_valence_separates_an_oxo_and_a_nitrido_from_an_aqua(monkeypatch):
    real = Chem.AddHs(Chem.MolFromSmiles(_NITRIDO_OXO_DIAQUA))
    m = metal_index(real)
    donors = [n.GetIdx() for n in real.GetAtomWithIdx(m).GetNeighbors()]
    q, hyb = D.delocalised_charges(real), _stripped_hybridisation(real)

    def targets():
        return {d: D.ml_distance(real, m, d, _MOLYBDENUM, set(donors), q, hyb=hyb) for d in donors}

    lig = {d: core.ligand_valence(real.GetAtomWithIdx(d)) for d in donors}
    z = {d: real.GetAtomWithIdx(d).GetAtomicNum() for d in donors}
    aqua = [d for d in donors if lig[d] == _AQUA_PROTONS]
    assert sorted(lig.values()) == [0, 0, _AQUA_PROTONS, _AQUA_PROTONS], f"the proton count was not read: {lig}"
    on = targets()
    oxo = next(d for d in donors if z[d] == _OXYGEN and not lig[d])
    assert on[aqua[0]] - on[oxo] == pytest.approx(_ligand_free_contraction(_OXYGEN, 6))

    nitrido = _ligand_free_contraction(_NITROGEN, 6)
    oxo_contraction = _ligand_free_contraction(_OXYGEN, 6)
    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {})
    off = targets()
    shift = {d: off[d] - on[d] for d in donors}
    assert [shift[d] for d in aqua] == [0.0, 0.0], f"a donor whose ligand side fills it was contracted: {shift}"
    assert shift[next(d for d in donors if z[d] == _NITROGEN)] == pytest.approx(nitrido)
    assert shift[oxo] == pytest.approx(oxo_contraction)

    monkeypatch.undo()
    iso = rx.metal(_NITRIDO_OXO_DIAQUA, "tetrahedral")[0]
    q, hyb = D.delocalised_charges(iso.mol), _stripped_hybridisation(iso.mol)
    for d in iso.donors:  # the surrogate has no M-donor bond to read, and must reach the same answer
        got = D.ml_distance(iso.mol, iso.metal, int(d), iso.real_z, set(iso.donors), q, hyb=hyb)
        assert got == pytest.approx(on[int(d)]), f"donor {d} moved when the metal was replaced by the surrogate"


def test_one_oxo_gets_one_target_however_the_caller_spelled_it(monkeypatch):
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in (_NITRIDO_OXO_DIAQUA, _NITRIDO_OXO_DIAQUA_IONIC)]
    mols.append(_as_perceived(mols[0]))
    before = [[a.GetFormalCharge() for a in mol.GetAtoms()] for mol in mols]

    def targets(mol):
        m = metal_index(mol)
        donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
        q, hyb = D.delocalised_charges(mol), _stripped_hybridisation(mol)
        return sorted(round(D.ml_distance(mol, m, d, _MOLYBDENUM, set(donors), q, hyb=hyb), 9) for d in donors)

    want = targets(mols[0])
    for mol, spelling in zip(mols[1:], ("the ionic spelling", "the perceived single bond"), strict=True):
        assert targets(mol) == want, f"{spelling} gave a different M-L target set: {targets(mol)} != {want}"
    assert [[a.GetFormalCharge() for a in mol.GetAtoms()] for mol in mols] == before, (
        "reading the ionic form moved a formal charge on the caller's Mol"
    )

    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {})
    assert targets(mols[1]) != targets(mols[0]), "without the ligand-valence branch the spellings should NOT agree"


def test_a_terminal_hydride_takes_its_refit_contraction(monkeypatch):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[H][Ru]([H])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", params)
    metal = metal_index(mol)
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()]
    hydride = next(d for d in donors if mol.GetAtomWithIdx(d).GetAtomicNum() == 1)
    q, hyb = D.delocalised_charges(mol), _stripped_hybridisation(mol)
    on = D.ml_distance(mol, metal, hydride, 44, set(donors), q, hyb=hyb)

    contraction = _ligand_free_contraction(1, 8)
    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {z: v for z, v in D._LIGAND_FREE_CONTRACTION.items() if z != 1})
    off = D.ml_distance(mol, metal, hydride, 44, set(donors), q, hyb=hyb)
    assert off - on == pytest.approx(contraction)


def test_an_sp_atom_in_a_haptic_face_does_not_take_the_sigma_contraction():
    rw = Chem.RWMol()
    metal, a, b = (rw.AddAtom(Chem.Atom(z)) for z in (26, 6, 6))
    rw.AddBond(a, b, Chem.BondType.TRIPLE)
    rw.AddBond(a, metal, Chem.BondType.DATIVE)
    rw.AddBond(b, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    donors = {a, b}
    assert D._hapticity(mol, a, donors) == 2
    sp = D.ml_distance(mol, metal, a, 26, donors, {}, hyb={a: Chem.HybridizationType.SP})
    sp2 = D.ml_distance(mol, metal, a, 26, donors, {}, hyb={a: Chem.HybridizationType.SP2})
    assert sp == sp2, "the sigma-only SP contraction reached a multi-atom haptic face"

    sigma_sp = D.ml_distance(mol, metal, a, 26, {a}, {}, hyb={a: Chem.HybridizationType.SP})
    sigma_sp2 = D.ml_distance(mol, metal, a, 26, {a}, {}, hyb={a: Chem.HybridizationType.SP2})
    assert sigma_sp2 - sigma_sp == pytest.approx(D._SP_CONTRACTION)


def test_every_m_donor_lands_inside_the_window_the_model_stated():
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
