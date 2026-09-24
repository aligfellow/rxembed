"""Test metal-ligand seed distances and anti-overbond floors."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem import rdMolTransforms as T

import rxembed as rx
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


def test_pnictogen_uses_dative_cap_not_halide():
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


def test_vanadyl_oxo_contracts_without_splitting_acac_lengths(monkeypatch):
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

    monkeypatch.setitem(D._LIGAND_FREE_CONTRACTION, _OXYGEN, (0.0, 0.0))
    off = targets()
    assert off[oxo[0]] > got[oxo[0]], "the oxo did not take the tmQM contraction"
    assert all(off[d] == got[d] for d in acac), "the term reached a donor whose ligand side already fills it"


def test_ligand_valence_distinguishes_oxo_nitrido_and_aqua(monkeypatch):
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
    assert on[oxo] < on[aqua[0]], "the ligand-valence-free oxo did not bind shorter than an otherwise-identical aqua O"

    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {})
    off = targets()
    shift = {d: off[d] - on[d] for d in donors}
    assert [shift[d] for d in aqua] == [0.0, 0.0], f"a donor whose ligand side fills it was contracted: {shift}"
    assert shift[next(d for d in donors if z[d] == _NITROGEN)] > 0, "removing the nitrido contraction must lengthen it"
    assert shift[oxo] > 0, "removing the oxo contraction must lengthen it"

    monkeypatch.undo()
    iso = rx.metal(_NITRIDO_OXO_DIAQUA, "tetrahedral")[0]
    q, hyb = D.delocalised_charges(iso.mol), _stripped_hybridisation(iso.mol)
    for d in iso.donors:  # the surrogate has no M-donor bond to read, and must reach the same answer
        got = D.ml_distance(iso.mol, iso.metal, int(d), iso.real_z, set(iso.donors), q, hyb=hyb)
        assert got == pytest.approx(on[int(d)]), f"donor {d} moved when the metal was replaced by the surrogate"


def test_oxo_has_one_canonical_target(monkeypatch):
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


def test_terminal_hydride_takes_its_refit_contraction(monkeypatch):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[H][Ru]([H])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", params)
    metal = metal_index(mol)
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()]
    hydride = next(d for d in donors if mol.GetAtomWithIdx(d).GetAtomicNum() == 1)
    q, hyb = D.delocalised_charges(mol), _stripped_hybridisation(mol)
    on = D.ml_distance(mol, metal, hydride, 44, set(donors), q, hyb=hyb)

    monkeypatch.setattr(D, "_LIGAND_FREE_CONTRACTION", {z: v for z, v in D._LIGAND_FREE_CONTRACTION.items() if z != 1})
    off = D.ml_distance(mol, metal, hydride, 44, set(donors), q, hyb=hyb)
    assert off > on, "removing the hydride contraction must lengthen its target"


def test_haptic_sp_atom_skips_sigma_contraction():
    rw = Chem.RWMol()
    metal, a, b = (rw.AddAtom(Chem.Atom(z)) for z in (26, 6, 6))
    rw.AddBond(a, b, Chem.BondType.TRIPLE)
    rw.AddBond(a, metal, Chem.BondType.DATIVE)
    rw.AddBond(b, metal, Chem.BondType.DATIVE)
    for atom in (a, b):
        rw.GetAtomWithIdx(atom).SetNoImplicit(True)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    donors = {a, b}
    assert D.ligand_degree(mol.GetAtomWithIdx(a)) == 1, "isolate hapticity from the terminal-donor guard"
    assert D._hapticity(mol, a, donors) == 2
    sp = D.ml_distance(mol, metal, a, 26, donors, {}, hyb={a: Chem.HybridizationType.SP})
    sp2 = D.ml_distance(mol, metal, a, 26, donors, {}, hyb={a: Chem.HybridizationType.SP2})
    assert sp == sp2, "the sigma-only SP contraction reached a multi-atom haptic face"


@pytest.mark.parametrize(
    ("smiles", "terminal"),
    [("[C-](#[O+])->[Pt+2]", True), ("CC(->[Pt+2])#CC", False), ("[CH](->[Pt+2])#C", False)],
)
def test_sigma_sp_contraction_requires_a_terminal_ligand_axis(smiles, terminal):
    source = Chem.MolFromSmiles(smiles)
    for mol in (source, Chem.AddHs(source)):
        metal = metal_index(mol)
        donor = mol.GetAtomWithIdx(metal).GetNeighbors()[0].GetIdx()
        donors = {donor}
        assert D._hapticity(mol, donor, donors) == 0
        sp = D.ml_distance(mol, metal, donor, 78, donors, {}, hyb={donor: Chem.HybridizationType.SP})
        sp2 = D.ml_distance(mol, metal, donor, 78, donors, {}, hyb={donor: Chem.HybridizationType.SP2})
        assert sp2 - sp == pytest.approx(D._SP_CONTRACTION if terminal else 0.0)


def test_all_m_donors_land_in_model_windows():
    iso = rx.metal(_NI_N, "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=3).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        c = ens.mol.GetConformer(cid)
        for d in iso.donors:
            lo, hi = iso.cons.distances[(min(d, iso.metal), max(d, iso.metal))]
            got = T.GetBondLength(c, iso.metal, d)
            assert lo - 0.05 <= got <= hi + 0.05, f"donor {d}: {got:.3f} Å is outside its window ({lo:.3f}, {hi:.3f})"


def test_chelated_m_d_windows_do_not_get_independent_uff_pulls():
    iso = rx.metal("[Pd+2]1(<-[Cl-])(<-[Cl-])(<-[NH2]CC[NH2]->1)", "square_planar")[0]
    cons = iso.cons
    nitrogen = {d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == _NITROGEN}
    chloride = {d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 17}

    assert len(nitrogen) == 2
    assert len(chloride) == 2
    assert all(tuple(sorted((iso.metal, d))) not in cons.pulls for d in nitrogen)
    assert all(tuple(sorted((iso.metal, d))) in cons.pulls for d in chloride)


def test_coordinate_backed_chelate_windows_use_the_same_radial_policy():
    rw = Chem.RWMol()
    metal, n_left, carbon, n_right, chloride, bromide = (rw.AddAtom(Chem.Atom(z)) for z in (46, 7, 6, 7, 17, 35))
    rw.AddBond(n_left, carbon, Chem.BondType.SINGLE)
    rw.AddBond(carbon, n_right, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(((0, 0, 0), (2.1, 1, 0), (2.1, 0, 0), (2.1, -1, 0), (2.1, 0, 1), (2.1, 0, -1))):
        conf.SetAtomPosition(index, point)
    mol.AddConformer(conf)
    distances = {
        (metal, n_left): (2.1, 2.3),
        (metal, n_right): (2.1, 2.3),
        (metal, chloride): (2.1, 2.3),
        (metal, bromide): (2.1, 2.3),
    }
    cons = rx.Constraints(distances=distances)

    D.ff_terms(mol, cons, {metal: (46, [n_left, n_right, chloride, bromide])})

    assert (metal, n_left) not in cons.pulls
    assert (metal, n_right) not in cons.pulls
    assert cons.pulls[(metal, chloride)] == pytest.approx(2.2)
    assert cons.pulls[(metal, bromide)] == pytest.approx(2.2)
    mol.RemoveAllConformers()
    without_coordinates = rx.Constraints(distances=distances)
    D.ff_terms(mol, without_coordinates, {metal: (46, [n_left, n_right, chloride, bromide])})
    assert cons == without_coordinates


def test_vacant_site_excludes_ligand_backbone():
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


def test_non_donor_clearance_ignores_source_coordinates():
    rw = Chem.RWMol()
    metal, donor, sulfur = (rw.AddAtom(Chem.Atom(z)) for z in (46, 6, 16))
    rw.AddBond(donor, sulfur, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(donor, (2.1, 0.0, 0.0))
    conf.SetAtomPosition(sulfur, (2.76, 0.0, 0.0))
    mol.AddConformer(conf)

    cons = rx.Constraints()
    D.nondonor_floors(mol, metal, 46, [donor], cons)
    mol.RemoveAllConformers()
    without_coordinates = rx.Constraints()
    D.nondonor_floors(mol, metal, 46, [donor], without_coordinates)
    assert cons == without_coordinates


def test_haptic_backbone_stays_exempt_from_a_sigma_bridgehead_floor():
    rw = Chem.RWMol()
    metal, left, right, nitrogen, silicon = (rw.AddAtom(Chem.Atom(z)) for z in (22, 6, 6, 7, 14))
    rw.AddBond(left, right, Chem.BondType.DOUBLE)
    rw.AddBond(silicon, left, Chem.BondType.SINGLE)
    rw.AddBond(silicon, nitrogen, Chem.BondType.SINGLE)
    for donor in (left, right, nitrogen):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    cons = rx.Constraints()

    D.nondonor_floors(mol, metal, 22, [left, right, nitrogen], cons)

    assert (metal, silicon) not in cons.floors
