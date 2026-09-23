"""Test donor orientation and coplanarity restraints."""

from __future__ import annotations

import csv
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as T

import rxembed as rx
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
# an aryl thiolate S donor: RDKit types it sp2 from the ring's aromaticity, but the S carries no pi bond of
# its own, so the pi count gives sp3 and the two estimators disagree (period-3, so {7, 8} does not override)
_ARYL_THIOLATE_SMI = "[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[S-]c1ccccc1"
_ETA1_CH_SMI = "CC(C)(C)[P](->[Pd]<-[CH]1=C(N2CCOCC2)C=CC=C1)(C(C)(C)C)C(C)(C)C"
_TWO = 2
_WALL_SLACK = 1.0  # deg: a UFF torsion constraint is a penalty, not a hard wall, so a minimum riding the cap
# edge settles a hair outside it. Wide enough for that, far narrower than any fold the cap exists to stop.


def _donor(iso, symbol):
    return next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == symbol)


def _capped(iso):
    return {e[1] for e in iso.cons.coplanar}


class _TorsionSpy:
    """Record every UFF torsion constraint an `_ff_terms` writer emits, as `(i, j, k, w, lo, hi, fc)`."""

    def __init__(self):
        self.torsions = []

    def UFFAddTorsionConstraint(self, i, j, k, w, relative, lo, hi, fc):  # noqa: N802 (RDKit FF API name)
        self.torsions.append((i, j, k, w, lo, hi, fc))


def _emitted_caps(iso, seed=1):
    """Every torsion `Coplanar._ff_terms` actually writes on a seed conformer of `iso`."""
    from rxembed.mechanisms import Coplanar

    ens = rx.embed(iso, n=1, seed=seed)
    # `_mol`, not `.mol`: `_ff_terms` reads `conf.GetOwningMol()` to decide plane-sharing, so it must see the
    # bond-less surrogate the FF relaxes; `.mol`'s dative M-L bonds would change which caps look redundant.
    spy = _TorsionSpy()
    Coplanar()._ff_terms(spy, ens.cons, ens._mol.GetConformer(ens.ids[0]), 1e4)
    return spy.torsions


def _emitted_cap_donors(iso, seed=1):
    """Donor indices for which `Coplanar._ff_terms` actually writes a torsion on a seed conformer of `iso`."""
    return {t[1] for t in _emitted_caps(iso, seed)}


# --- the orientation wall: calibrated classes only -------------------------------------------------------


@pytest.mark.parametrize(
    ("smi", "walled"),
    [
        ("CCN[Pd](Cl)Cl", True),  # amine N sp3: a calibrated class, and its PROTON is walled (the sp3-amine fix)
        ("O->[Pd](Cl)Cl", False),  # aqua O sp3 is UNcalibrated (n < 6): the wall abstains, as the fold gate does
    ],
    ids=["amine", "aqua"],
)
def test_proton_walls_require_calibrated_donor(smi, walled):
    iso = rx.metal(smi, "square_planar").select(index=0)
    protons = [k for k in iso.cons.angles if k[0] == iso.metal and iso.mol.GetAtomWithIdx(k[2]).GetAtomicNum() == 1]
    assert bool(protons) == walled, f"{smi}: {len(protons)} proton walls, expected {'some' if walled else 'none'}"


@pytest.mark.parametrize("writer", [DO._orient_donor, DO._coplanar_donor])
def test_donor_writers_use_supplied_classification_including_abstention(writer, monkeypatch):
    iso = rx.metal("[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-n1ccccc1", "SPL")[0]
    mol, donors = iso._graph, set(iso.donors)
    donor = next(d for d in donors if mol.GetAtomWithIdx(d).GetIsAromatic())
    hyb = DO._stripped_hybridisation(mol)
    expected = Constraints()
    writer(mol, iso.metal, donor, donors, expected)
    assert expected.angles or expected.coplanar

    def reclassified(_mol):
        pytest.fail("a supplied graph classification must not be recomputed")

    monkeypatch.setattr(DO, "_stripped_hybridisation", reclassified)
    actual = Constraints()
    writer(mol, iso.metal, donor, donors, actual, hyb=hyb)
    assert actual == expected
    abstained = Constraints()
    writer(mol, iso.metal, donor, donors, abstained, hyb={})
    assert abstained == Constraints()


@pytest.mark.parametrize(
    "ligand", ["n1ccccc1", "n1cc[nH]c1", "n1[nH]ccc1", "n1ccoc1", "n1nn[nH]c1", "[c-]1ccccc1", "N(C)=CC", "[C-]1=CCCC1"]
)
def test_free_sp2_donor_accepts_native_ligand_bisector(ligand):
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    from rxembed.mechanisms import Angle

    iso = rx.metal(f"[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-{ligand}", "SPL")[0]
    donor = next(
        d for d in iso.donors if sum(n.GetAtomicNum() > 1 for n in iso._graph.GetAtomWithIdx(d).GetNeighbors()) == 2
    )
    left, right = [n.GetIdx() for n in iso._graph.GetAtomWithIdx(donor).GetNeighbors()]
    maps = []
    fragments = Chem.GetMolFrags(iso._graph, asMols=True, fragsMolAtomMapping=maps)
    block, indices = next((block, indices) for block, indices in zip(fragments, maps, strict=True) if donor in indices)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed, params.numThreads = 42, 1
    assert rdDistGeom.EmbedMolecule(block, params) == 0
    assert rdForceFieldHelpers.UFFOptimizeMolecule(block, maxIters=2000) == 0
    xyz = block.GetConformer().GetPositions()
    origin = xyz[indices.index(donor)]
    rays = xyz[[indices.index(left), indices.index(right)]] - origin
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    external = -rays.sum(axis=0)
    external /= np.linalg.norm(external)
    work = Chem.Mol(iso._graph)
    work.RemoveAllConformers()
    conf = Chem.Conformer(work.GetNumAtoms())
    for atom, position in zip(indices, xyz, strict=True):
        conf.SetAtomPosition(atom, position)
    conf.SetAtomPosition(iso.metal, origin + 2.0 * external)
    work.AddConformer(conf)
    angles = {key: window for key, window in iso.cons.angles.items() if key[:2] == (iso.metal, donor)}
    assert len(angles) == 2
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(work)
    Angle()._ff_terms(ff, Constraints(angles=angles), work.GetConformer(), 1.0)
    ff.Initialize()
    assert ff.CalcEnergy() == pytest.approx(0.0, abs=1e-8), "donor walls oppose the native ligand's external bisector"
    work.GetConformer().SetAtomPosition(iso.metal, origin + 2.0 * rays[0])
    assert ff.CalcEnergy() > 0.0, "centring must still penalize donation into a substituent"

    order = list(reversed(range(work.GetNumAtoms())))
    permuted = Chem.RenumberAtoms(work, order)
    changed = Constraints()
    DO._orient_donor(permuted, order.index(iso.metal), order.index(donor), {order.index(donor)}, changed)
    remapped = {tuple(order[i] for i in key): value for key, value in changed.angles.items()}
    assert remapped == angles, "atom order or source coordinates changed the graph-derived donor windows"


def test_sp3_amine_does_not_fold_proton_to_metal():
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


def test_eta1_arene_ch_does_not_fold_hydrogen_to_metal():
    iso = rx.metal(_ETA1_CH_SMI)[0]
    ens = rx.embed(iso, n=1, seed=7).minimize()
    carbon = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 6)
    hydrogen = next(n.GetIdx() for n in iso.mol.GetAtomWithIdx(carbon).GetNeighbors() if n.GetAtomicNum() == 1)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    assert np.linalg.norm(pos[iso.metal] - pos[hydrogen]) > 2.4, "the eta1 C-H folded its hydrogen onto the metal"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_phosphine_protons_stay_splayed_on_both_inputs(tmp_path):

    def m_d_h(ens, metal, donors):
        mol = ens.mol
        return [
            T.GetAngleDeg(mol.GetConformer(cid), int(metal), int(d), h.GetIdx())
            for d in donors
            for h in mol.GetAtomWithIdx(int(d)).GetNeighbors()
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


def test_hybridisation_uses_metal_stripped_graph():
    params = Chem.SmilesParserParams()
    params.removeHs = False  # a hydride is a DONOR here, so it has to survive the parse as its own atom
    mol = Chem.MolFromSmiles("[H][Ru]([H])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", params)
    hyb = DO._stripped_hybridisation(mol)
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Ru")
    carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors() if n.GetAtomicNum() == 6]
    hydrides = [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors() if n.GetAtomicNum() == 1]
    assert len(carbons) == 4, "the fixture lost a carbonyl"
    assert len(hydrides) == 2, "the fixture lost a hydride, so the never-classified half asserts nothing"
    for c in carbons:
        assert hyb[c] == Chem.HybridizationType.SP, "the carbonyl carbon was typed through its M-C bond"
    for h in hydrides:
        assert h not in hyb, "a hydride has no donation axis and must never be classified"


def test_explicit_triple_bond_outranks_an_aromatic_flag():
    mol = Chem.MolFromSmiles("C1#CC=CC=C1")
    triple = mol.GetBondWithIdx(0)
    atoms = (triple.GetBeginAtomIdx(), triple.GetEndAtomIdx())
    assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms), "fixture premise"
    assert all(DO._stripped_hybridisation(mol)[i] == Chem.HybridizationType.SP for i in atoms)


def test_ambiguous_anionic_aryl_nitrogen_abstains_from_a_planar_donor_model():
    mol = rx.parse_smiles("C[Si](C)(C)[N-]c1ccccc1->[Pd+2](<-[Cl-])(<-[Cl-])<-Cl")
    nitrogen = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "N")

    assert nitrogen not in DO._stripped_hybridisation(mol)


# MOCQIE's N12: xyz2mol writes the cumulated O=[N+]=C Kekule form. Both estimators type the metal-bound N sp,
# but an sp centre with 2 ligand sigma bonds has used its sigma framework and both pi orbitals on those bonds,
# leaving no sigma lone pair for the dative M-N bond that is drawn; the real donor is the bent, sp2 resonance
# form. A genuine sp donor (nitrile N, carbyne C) has exactly one ligand sigma bond and is unaffected.
_CUMULATED_NITRO_SMI = "O=[N+](=CC)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"
_NITRILE_SMI = "CC#N->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"


def test_metal_bound_cumulated_sp_donor_is_retyped_sp2():
    mol = rx.parse_smiles(_CUMULATED_NITRO_SMI)
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert mol.GetAtomWithIdx(nitrogen).GetHybridization() == Chem.HybridizationType.SP, "fixture premise"
    assert DO._stripped_hybridisation(mol)[nitrogen] == Chem.HybridizationType.SP2


def test_retyped_cumulated_donor_gets_an_md_x_orientation_wall():
    mol = rx.parse_smiles(_CUMULATED_NITRO_SMI)
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pd")
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    donors = {n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()}
    cons = Constraints()
    DO._orient_donor(mol, metal, nitrogen, donors, cons)
    assert (metal, nitrogen, oxygen) in cons.angles


def test_genuine_terminal_sp_nitrile_donor_keeps_sp_and_gets_no_coplanar_wall():
    """A one-ligand-sigma-bond sp donor keeps its native class: no in-plane lone pair, so no coplanar cap."""
    mol = rx.parse_smiles(_NITRILE_SMI)
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pd")
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    donors = {n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()}
    assert DO._stripped_hybridisation(mol)[nitrogen] == Chem.HybridizationType.SP
    cons = Constraints()
    DO._coplanar_donor(mol, metal, nitrogen, donors, cons)
    assert not cons.coplanar, "a genuine sp donor has no in-plane lone pair to hold coplanar"


@pytest.mark.parametrize("smiles", ["CC#N", "CC#CC", "C=C=C", "O=C=O", "[N-]=C=S", "[N-]=C=[Se]"])
def test_internal_sp_centres_keep_their_native_class(smiles):
    source = Chem.MolFromSmiles(smiles)
    for mol in (source, Chem.AddHs(source)):
        for work in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
            native = [a.GetIdx() for a in work.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP]
            assert native
            agreed = DO._stripped_hybridisation(work)
            assert all(agreed[i] == Chem.HybridizationType.SP for i in native)


@pytest.mark.parametrize("end", ["S", "[Se]"])
def test_linear_reference_does_not_create_a_donor_plane(end):
    source = Chem.MolFromSmiles(f"[N-](=C={end})->[Pt+2](<-[Cl-])(<-[Cl-])<-N")
    for mol in (source, Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))):
        iso = rx.metal(mol, "SPL")[0]
        donor = next(
            d
            for d in iso.donors
            if iso.mol.GetAtomWithIdx(d).GetFormalCharge() == -1 and iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 7
        )
        assert DO.inplane_sp2_donor(iso.mol, donor)
        assert any(key[1] == donor for key in iso.cons.angles), "retain the donor's supported bend restraint"
        assert not iso.cons.coplanar, "a linear N-C-X reference does not define a plane"


@pytest.mark.parametrize(
    ("smiles", "terminal"),
    [
        ("CC(->[Pt+2](<-[Cl-])(<-[Cl-])<-N)#CC", False),
        ("C(->[Pt+2](<-[Cl-])(<-[Cl-])<-N)#C", False),
        ("[CH](->[Pt+2](<-[Cl-])(<-[Cl-])<-N)#C", False),
        ("[C-](#[O+])->[Pt+2](<-[Cl-])(<-[Cl-])<-N", True),
    ],
)
def test_only_terminal_sp_donors_get_end_on_restraints(smiles, terminal):
    source = Chem.MolFromSmiles(smiles)
    for mol in (source, Chem.AddHs(source)):
        metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 78)
        donors = {a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()}
        donor = next(d for d in donors if mol.GetAtomWithIdx(d).GetAtomicNum() == 6)
        hyb = DO._stripped_hybridisation(mol)
        assert hyb[donor] == Chem.HybridizationType.SP
        assert (DO.donation_axis(mol, donor, donors, hyb=hyb) is not None) == terminal
        assert DO.donation_axis(mol, donor, donors) == DO.donation_axis(mol, donor, donors, hyb=hyb)
        cons = Constraints()
        DO._orient_donor(mol, metal, donor, donors, cons)
        assert bool(cons.angles) == terminal
        if terminal:
            assert all(lo > 150 for lo, _ in cons.angles.values())


@pytest.mark.parametrize(
    ("bond_type", "expected"),
    [(Chem.BondType.SINGLE, [3]), (Chem.BondType.DOUBLE, None)],
    ids=["sigma-codonors", "eta2-face"],
)
def test_only_pi_bonded_codonors_remove_the_donation_axis(bond_type, expected):
    rw = Chem.RWMol()
    metal, left, right, left_sub, right_sub = (rw.AddAtom(Chem.Atom(z)) for z in (78, 7, 7, 6, 6))
    rw.AddBond(left, right, bond_type)
    rw.AddBond(left, left_sub, Chem.BondType.SINGLE)
    rw.AddBond(right, right_sub, Chem.BondType.SINGLE)
    rw.AddBond(left, metal, Chem.BondType.DATIVE)
    rw.AddBond(right, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    assert DO.donation_axis(mol, left, {left, right}) == expected


def test_uncalibrated_codonor_keeps_the_shared_backbone_arm_walled():
    # R_pair: a backbone arm is exempt from the wall only when the co-donor it reaches is itself calibrated
    # and small enough to hold its own arm. Swap the P,N chelate's amine N for an ether O (uncalibrated,
    # ("O", SP3) has no census entry): P's backbone arm must stay walled, since O has no fallback hold.
    smi = "C[P]1(C)CCO->[Pd+2](<-[Cl-])(<-[Cl-])<-1"
    iso = rx.metal(smi, "square_planar")[0]
    assert (iso.metal, 1, 3) in iso.cons.angles


def test_backbone_target_picks_the_nearer_donor_across_a_bridging_donor():
    # A macrocycle can route one donor's only arm past a bridging donor to a farther one beyond it (CAGJED:
    # an As donor bridges two amidate N's). R_pair must key off the NEAREST co-donor: whichever donor the
    # arm reaches first decides the exemption, not a donor further down the same ring.
    rw = Chem.RWMol()
    metal, na, ca, pmid, cb, nb, cx = (rw.AddAtom(Chem.Atom(z)) for z in (46, 7, 6, 15, 6, 7, 6))
    rw.AddBond(na, ca, Chem.BondType.SINGLE)
    rw.AddBond(ca, pmid, Chem.BondType.SINGLE)
    rw.AddBond(pmid, cb, Chem.BondType.SINGLE)
    rw.AddBond(cb, nb, Chem.BondType.SINGLE)
    rw.AddBond(pmid, cx, Chem.BondType.SINGLE)
    rw.AddBond(na, metal, Chem.BondType.DATIVE)
    rw.AddBond(pmid, metal, Chem.BondType.DATIVE)
    rw.AddBond(nb, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    donors = {na, pmid, nb}

    assert DO._backbone_targets(mol, na, donors) == {ca: pmid}
    assert DO._backbone_targets(mol, nb, donors) == {cb: pmid}


def test_symmetric_dithiolate_chelate_keeps_both_backbone_arms_walled():
    # FISCIT/CIRFIT: a benzene-1,2-dithiolate's two S donors are mutually "small" (one heavy substituent
    # each) and mutually calibrated ("S", SP3). R_pair must not exempt both arms at once, or the whole
    # chelate loses its fold wall; a ring of two equally small donors keeps both.
    smi = "Cl[Pd]1(Cl)<-[S-]c2ccccc2[S-]->1"
    iso = rx.metal(smi, "square_planar")[0]
    s1, s2 = (d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "S")
    c1 = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(s1).GetNeighbors() if nb.GetAtomicNum() > 1)
    c2 = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(s2).GetNeighbors() if nb.GetAtomicNum() > 1)
    assert (iso.metal, s1, c1) in iso.cons.angles
    assert (iso.metal, s2, c2) in iso.cons.angles


def test_orientation_wall_never_uses_a_sigma_codonor_as_a_substituent():
    rw = Chem.RWMol()
    metal, left, right, carbon = (rw.AddAtom(Chem.Atom(z)) for z in (39, 7, 7, 6))
    rw.AddBond(left, right, Chem.BondType.SINGLE)
    rw.AddBond(right, carbon, Chem.BondType.DOUBLE)
    rw.AddBond(left, metal, Chem.BondType.DATIVE)
    rw.AddBond(right, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    cons = Constraints()

    DO._orient_donor(mol, metal, left, {left, right}, cons)
    DO._orient_donor(mol, metal, right, {left, right}, cons)

    assert (metal, left, right) not in cons.angles
    assert (metal, right, left) not in cons.angles
    assert cons.angles[(metal, right, carbon)] == (108.0, 180.0)


@pytest.mark.parametrize(
    "smi",
    ["Cl[Pd](Cl)(Cl)<-[S](=O)(C)CC", "Cl[Pd](Cl)(Cl)<-[S+]([O-])(C)CC"],
    ids=["neutral-double", "charge-separated"],
)
def test_sulfoxide_lewis_forms_are_pyramidal_donors(smi):
    iso = rx.metal(smi, "square_planar", stereo="free")[0]
    sulfur = _donor(iso, "S")
    assert DO._stripped_hybridisation(iso.mol)[sulfur] == Chem.HybridizationType.SP3
    assert len([k for k in iso.cons.angles if k[1] == sulfur]) == 3
    assert sulfur not in _capped(iso)


def test_tricoordinate_aromatic_phosphorus_gets_the_pyramidal_orientation_wall():
    smi = "Cc1c[p](N(C)C)(->[Pt+2](<-[Cl-])(<-[Cl-])<-[NH3])cc1C"
    iso = rx.metal(smi, "square_planar")[0]
    phosphorus = _donor(iso, "P")

    assert DO._stripped_hybridisation(iso.mol)[phosphorus] == Chem.HybridizationType.SP3
    assert len([key for key in iso.cons.angles if key[1] == phosphorus]) == 3


def test_thioether_donor_stays_pyramidal():
    iso = rx.metal("Cl[Pd](Cl)(Cl)<-[S](C)CC", "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=1, seed=7).minimize()
    sulfur = _donor(iso, "S")
    carbon = [n.GetIdx() for n in iso.mol.GetAtomWithIdx(sulfur).GetNeighbors() if n.GetAtomicNum() > 1]
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    normal = np.cross(pos[carbon[0]] - pos[iso.metal], pos[carbon[1]] - pos[iso.metal])
    out_of_plane = abs((pos[sulfur] - pos[iso.metal]) @ normal / np.linalg.norm(normal))
    assert out_of_plane > 0.7, "the coordinated thioether sulfur flattened"


# --- the coplanarity cap: which donors get one -----------------------------------------------------------


def test_coplanar_permutations_each_define_one_plane():
    iso = rx.metal(NI_N, "square_planar")[0]
    mol = iso.mol
    o, n = _donor(iso, "O"), _donor(iso, "N")
    assert {o, n} <= _capped(iso), "both conjugated donors of the N-bound isomer must be capped"

    ((_i, _d, k, w, _anchor, _cap),) = [e for e in iso.cons.coplanar if e[1] == o]
    c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    assert k == c, "the plane is defined through the carboxyl carbon"
    subs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1]
    assert w == next(s for s in subs if mol.GetAtomWithIdx(s).GetSymbol() == "O"), "the 2nd O is the reference"
    assert _anchor is None, "a proper row defines a plane but not its periodic well"

    ((_i, _n, k, w, _anc, _cap),) = [e for e in iso.cons.coplanar if e[1] == n]
    heavy = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1}
    assert {k, w} == heavy, "the improper's plane atoms are the N's own two DIRECT heavy substituents"
    assert _anc == DO._COPLANAR_ANCHOR, "two direct substituents define the external anti sector"


def test_conjugated_bridge_chelate_gets_the_measured_ring_closure_hinge():
    # Replaces the old bridged D-X-D branch (_BRIDGED_COPLANAR_CAP): a bridging carboxylate is a conjugated
    # 4-ring, exactly `_ring_hinge`'s case now, at the measured 4-ring cap and one row per donor (no reciprocal
    # duplicate: unlike the deleted branch, the pairwise ring walk never needs a dedup guard).
    source = Chem.MolFromSmiles("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1")
    for mol in (source, Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))):
        iso = rx.metal(mol, "tetrahedral")[0]
        hinge = [row for row in iso.cons.coplanar if row[5] == DO._HINGE_CAP[4]]

        assert len(hinge) == 2, "one row per donor of the ring"
        oxygens = {donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetAtomicNum() == 8}
        for metal, d, other, bridge, anchor, _cap in hinge:
            assert metal == iso.metal
            assert {d, other} == oxygens
            assert anchor == DO._COPLANAR_ANCHOR
            assert iso.mol.GetBondBetweenAtoms(bridge, d) is not None
            assert iso.mol.GetBondBetweenAtoms(bridge, other) is not None


def test_conjugated_dithiolate_ring_gets_both_hinge_rows():
    # bis(benzene-1,2-dithiolate)Pd's FISCIT/CIRFIT ring: a 5-membered chelate whose whole metal-free S-C=C-S
    # path is aromatic-conjugated. Both donors hinge (mutation: drop the conjugation test and this still
    # passes, since ring size alone no longer distinguishes a real chelate from an unconjugated one).
    iso = rx.metal("Cl[Pd]1(Cl)<-[S-]c2ccccc2[S-]->1", "square_planar")[0]
    s1, s2 = (d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "S")
    hinge = [row for row in iso.cons.coplanar if row[5] == DO._HINGE_CAP[5]]

    assert len(hinge) == 2, "one row per donor of the ring"
    assert {row[1] for row in hinge} == {s1, s2}
    for metal, d, other, x, anchor, _cap in hinge:
        assert metal == iso.metal
        assert other == (s2 if d == s1 else s1)
        assert iso.mol.GetBondBetweenAtoms(d, x) is not None, "X is D's own ring neighbour"
        assert anchor == DO._COPLANAR_ANCHOR


@pytest.mark.parametrize(
    ("name", "smi", "geometry"),
    [
        ("en-Pd", "Cl[Pd]1(Cl)<-NCCN->1", "square_planar"),  # 5-ring, sp3 backbone: no conjugated path
        ("acac-Zn", "CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]C(C)=C1", "tetrahedral"),  # 6-ring: excluded on size alone
    ],
)
def test_nonconjugated_and_oversized_rings_get_no_hinge_row(name, smi, geometry):
    # Mutation: allow ring sizes 4-6 (drop the `_HINGE_CAP` size gate) and acac-Zn picks up a spurious row.
    iso = rx.metal(smi, geometry)[0]
    hinge = [row for row in iso.cons.coplanar if row[5] in DO._HINGE_CAP.values()]
    assert not hinge, f"{name}: no ring here is both small enough and fully conjugated"


_TMQMG_DIR = Path("/home/ali/Documents/Codes/tmQMg/data")


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.skipif(not _TMQMG_DIR.is_dir(), reason="needs a local tmQMg clone")
def test_fiscit_dithiolate_hinge_holds_the_measured_fold():
    # FISCIT (bis-benzenedithiolate-oxo-Tc): unconstrained the ring hinge fold measures ~38 deg; the hinge
    # holds it to ~34. Mutation: delete the hinge row and this fold exceeds 36 again.
    charges = {
        row["id"]: int(row["charge"])
        for row in csv.DictReader((_TMQMG_DIR / "tmQMg_properties_and_targets.csv").open())
    }
    ref = rx.read_xyz(
        str(_TMQMG_DIR / "xyz" / "FISCIT.xyz"),
        charge=charges["FISCIT"],
        connectivity="xyzgraph",
        bond_orders="xyz2mol",
    )
    isos = rx.metal(ref, lengths="model")
    ref_cx = rx.cxsmiles(ref)
    iso = next(i for i in isos if rx.cxsmiles(i) == ref_cx)
    hinge = [row for row in iso.cons.coplanar if row[5] in DO._HINGE_CAP.values()]
    assert hinge, "FISCIT's two dithiolate rings must each get a hinge row"

    ens = rx.embed(iso, n=3, seed=42, threads=1)
    fold = max(
        180.0 - abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), i, j, k, w))
        for cid in ens.ids
        for i, j, k, w, _anchor, _cap in hinge
    )
    assert fold <= 36.0, f"hinge fold {fold:.1f} deg exceeds the measured bound"


def test_kappa2_dithiocarbamate_hinge_holds_the_measured_fold_and_rereads_kappa2(tmp_path):
    # Ni(S2CNMe2)2: a real kappa2,kappa2 bis(dithiocarbamato)nickel(II), each S2CN ring a conjugated 4-ring.
    # Mutation: cap 26 deg (the deleted _BRIDGED_COPLANAR_CAP value) instead of the measured 23.
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(28))
    donors = []
    for _ligand in range(2):
        n, m1, m2, c, s1, s2 = (rw.AddAtom(Chem.Atom(z)) for z in (7, 6, 6, 6, 16, 16))
        rw.AddBond(n, m1, Chem.BondType.SINGLE)
        rw.AddBond(n, m2, Chem.BondType.SINGLE)
        rw.AddBond(n, c, Chem.BondType.SINGLE)
        rw.AddBond(c, s1, Chem.BondType.DOUBLE)
        rw.AddBond(c, s2, Chem.BondType.SINGLE)
        rw.AddBond(s1, metal, Chem.BondType.DATIVE)
        rw.AddBond(s2, metal, Chem.BondType.DATIVE)
        rw.GetAtomWithIdx(s2).SetFormalCharge(-1)
        donors.append((s1, s2))
    rw.GetAtomWithIdx(metal).SetFormalCharge(2)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, catchErrors=True)

    iso = rx.metal(mol, "square_planar")[0]
    hinge = [row for row in iso.cons.coplanar if row[5] == DO._HINGE_CAP[4]]
    assert len(hinge) == 4, "one row per donor, per ring"

    ens = rx.embed(iso, n=3, seed=42, threads=1)
    fold = max(
        180.0 - abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), i, j, k, w))
        for cid in ens.ids
        for i, j, k, w, _anchor, _cap in hinge
    )
    assert fold <= 24.0, f"hinge fold {fold:.1f} deg exceeds the measured 4-ring bound"

    cid = ens.ids[0]
    xyz_path = tmp_path / "ni_dtc2.xyz"
    Chem.MolToXYZFile(ens.mol, str(xyz_path), confId=cid)
    fresh = rx.read_xyz(str(xyz_path), charge=0, connectivity="xyzgraph", bond_orders="xyz2mol")
    fresh_metal = next(a for a in fresh.GetAtoms() if a.GetAtomicNum() == 28)
    neighbour_symbols = sorted(nb.GetSymbol() for nb in fresh_metal.GetNeighbors())
    assert neighbour_symbols == ["S", "S", "S", "S"], "kappa2,kappa2: no Ni-C bond on reread"


def test_only_an_inplane_sp2_donor_is_capped():
    for name, smi, geometry in (
        ("sp3 amine", "CCN[Pd](Cl)Cl", "square_planar"),
        ("sp nitrile", "CC#N[Pd](Cl)Cl", "square_planar"),
        ("side-on eta2", _ETA2_SMI, "square_planar"),
    ):
        isos = list(rx.metal(smi, geometry))
        assert isos, f"{name}: nothing was enumerated, so no cap was ever inspected"
        for iso in isos:
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
def test_graph_recovers_missed_donors(name, smi, symbol, conjugated):
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


def test_uncalibrated_donor_class_still_gets_the_cap():
    assert ("S", DO._SP2) not in DO._FOLD_WINDOW, "('S',SP2) must stay uncalibrated (corpus has no conjugated one)"
    iso = rx.metal(_THIONE_SMI, "square_planar")[0]
    s = _donor(iso, "S")
    assert not any(k[0] == iso.metal and k[1] == s for k in iso.cons.angles), "the fold wall must abstain on S sp2"
    assert s in _capped(iso)


def test_aryl_thiolate_sulfur_gets_the_pyramidal_orientation_wall():
    # FISCIT: an aryl thiolate S typed sp2 by RDKit's ring conjugation, sp3 by lone-pair count, got no wall
    # at all (both estimators must agree per class). Period 3+ does not planarise (unlike N/O), so the fix
    # resolves the disagreement to sp3, not sp2.
    iso = rx.metal(_ARYL_THIOLATE_SMI, "square_planar")[0]
    sulfur = _donor(iso, "S")
    carbon = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(sulfur).GetNeighbors() if nb.GetAtomicNum() > 1)

    assert DO._stripped_hybridisation(iso.mol)[sulfur] == Chem.HybridizationType.SP3
    assert (iso.metal, sulfur, carbon) in iso.cons.angles


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_ts_keeps_each_unowned_cap():
    source = rx.read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    isos = list(rx.metal(source, "octahedral", center="Mn", fix=_MN_H2_RC))
    assert isos, "nothing was enumerated: the partial-freeze claim was never tested"
    for iso in isos:
        assert iso.cons.coplanar
        assert all(not set(row[:4]) <= iso.cons.frozen for row in iso.cons.coplanar)


# --- the coplanarity cap is SOFT: a window the relax lands inside, never a pin ---------------------------
# The bound every test below reads is the cap's own declared half-width, taken off the `cons.coplanar` entry.
# The effect sizes it replaces are on-vs-off medians and tails, per donor, over 5-8 seeds.


def test_cap_window_has_declared_width_and_force_constant():
    from rxembed import mechanisms
    from rxembed.mechanisms import _coplanar_window

    cap = DO._COPLANAR_CAP
    for phi, well in ((3.0, 0.0), (-3.0, 0.0), (177.0, 180.0), (-177.0, -180.0), (100.0, 180.0)):
        lo, hi = _coplanar_window(phi, cap)
        assert hi - lo == pytest.approx(cap), f"phi={phi}: the window is not `cap` wide: a pin holds nothing softly"
        assert well in (lo, hi), f"phi={phi}: the well {well} is not an EDGE; riding the wall would miss the plane"
        assert (phi >= well) == (hi > well), f"phi={phi}: the window opened on the far side of the well"

    assert _coplanar_window(75.0, cap, 180.0) == (135.0, 180.0)
    assert _coplanar_window(-75.0, cap, 180.0) == (-180.0, -135.0)

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


def test_relax_lands_inside_cap_without_flattening():
    cap = DO._COPLANAR_CAP
    dev = _cap_deviation(NI_N, (6,), n=1)
    assert dev, "no capped donor was measured: the fixture is wrong"
    for j, arr in dev.items():
        assert arr.max() <= cap + _WALL_SLACK, f"donor {j}: the relax broke through its cap ({arr.max():.1f}°)"


def test_cap_excludes_its_aryl_anchor():
    iso = rx.metal(NI_N, "square_planar")[0]
    mol = iso.mol
    n = _donor(iso, "N")
    n_nbrs = list(mol.GetAtomWithIdx(n).GetNeighbors())
    ipso = next(nb.GetIdx() for nb in n_nbrs if nb.GetIsAromatic())
    other = next(nb.GetIdx() for nb in n_nbrs if nb.GetIdx() != ipso and nb.GetAtomicNum() > 1)
    ring_nbrs = mol.GetAtomWithIdx(ipso).GetNeighbors()
    ortho = next(nb.GetIdx() for nb in ring_nbrs if nb.GetIdx() != n and nb.GetIsAromatic())
    cap = next(row for row in iso.cons.coplanar if row[1] == n)
    assert cap[:4] == (iso.metal, n, other, ipso)
    assert ortho not in cap[:4]


def test_cap_survives_every_constraints_rebuild():
    import rxembed.pipeline.ensemble as ensemble_module

    c = Constraints()
    c.coplanar.append((0, 1, 2, 3, 180.0, 45.0))
    assert c.relaxed().coplanar == c.coplanar, "relaxed() dropped the coplanarity cap"
    assert c.relaxed().coplanar is not c.coplanar, "relaxed() must copy, not alias, the coplanar list"

    ens = rx.embed(rx.metal(NI_N, "square_planar")[0], n=2, seed=1)
    assert ens.cons.coplanar, "the fixture must carry a cap for this to mean anything"
    seen, real = [], ensemble_module.restrained_uff

    def spy(mol, cons, *a, **kw):
        seen.append(cons)
        return real(mol, cons, *a, **kw)

    # ty types every function literal nominally, so no stand-in is ever assignable to what it replaces.
    ensemble_module.restrained_uff = spy  # ty: ignore[invalid-assignment]
    try:
        ens._settle_seeds(bins=2)
    finally:
        ensemble_module.restrained_uff = real
    assert seen, "the settle never reached the relax"
    assert all(x.coplanar for x in seen), "the settle relaxed with the coplanarity cap absent"


# --- every local donor plane survives the metal-bond strip ------------------------------------------------
# Two donors in one conjugated ligand still define different M-D-X-Y inequalities. An all-sp2 path is not a
# proof that either local term is redundant.

# a crowded Ni(II) conjugated chelate: a pyridylimine sharing one sp2 plane with an amidate.
_CASE2 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE3 = "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C"
_CASE4 = "CCOC1=[O]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[n]2c[nH]c(C)c21"  # imidazole ester
# picolinate-ethylenediamine Ni: a rigid conjugated bidentate with two distinct local donor planes.
_PICO = "O=C1[O-]->[Ni+2]2(<-[NH2]CC[NH2]->2)<-n2ccccc21"


@pytest.mark.parametrize("smi", [_CASE2, _CASE3, _PICO], ids=["amidate", "carbanion", "picolinate"])
def test_ff_torsion_keeps_every_local_donor_plane(smi):
    iso = rx.metal(smi, "square_planar")[0]
    assert _capped(iso), "the fixture has no local donor-plane caps"
    assert _emitted_cap_donors(iso) == _capped(iso)


def test_local_donor_caps_do_not_collapse_the_ester():
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


def test_length_source_does_not_control_partial_freeze_walls():

    def walls(lengths):  # (metal, donor, substituent) windows, i.e. `_orient_donor`'s, not the polyhedron's
        source = rx.read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
        iso = rx.metal(source, "octahedral", center="Mn", fix=_MN_H2_RC, lengths=lengths)[0]
        return {k: v for k, v in iso.cons.angles.items() if k[0] == iso.metal and k[1] != iso.metal}

    measured, modelled = walls("input"), walls("model")
    assert measured == modelled, "M-L length provenance changed which graph-derived orientations exist"
    assert measured, "partially free donors lost every orientation wall"
    why = "the carbonyl that motivated this must be the sp-carbon wall, or the test is measuring something else"
    assert measured[(1, 61, 3)] == (165.0, 180.0), why
