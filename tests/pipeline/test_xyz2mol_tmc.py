"""Test canonical ligand ordering and bond-order ranking in the TMC perceiver."""

from __future__ import annotations

import math
import random
from itertools import permutations

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.pipeline import xyz2mol_tmc as tmc
from rxembed.pipeline.xyz2mol_tmc import _invented_hydrogens, get_lig_mol, get_tmc_mol
from tests.conftest import EXAMPLES_DIR
from tests.metal_fixtures import ferrocene


@pytest.mark.parametrize(("atomic_number", "charge"), [(1, -1), (17, -1)])
def test_monatomic_ligand_keeps_its_proposed_charge(atomic_number, charge):
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(atomic_number))
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    ligand, perceived = get_lig_mol(mol, charge, [0])

    assert perceived == charge
    assert ligand.GetAtomWithIdx(0).GetFormalCharge() == charge


@pytest.mark.parametrize("element", ["Cl", "Br"])
def test_fragment_valence_is_checked_after_native_bond_orders(element, monkeypatch):
    rw = Chem.RWMol()
    for symbol in ("Ag", element, "O", "O", "O", "O"):
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for oxygen in range(2, 6):
        rw.AddBond(1, oxygen, Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    source = rw.GetMol()
    conf = Chem.Conformer(6)
    points = (
        (5, 0, 0),
        (0, 0, 0),
        (0.866, 0.866, 0.866),
        (0.866, -0.866, -0.866),
        (-0.866, 0.866, -0.866),
        (-0.866, -0.866, 0.866),
    )
    for atom, point in enumerate(points):
        conf.SetAtomPosition(atom, Point3D(*point))
    source.AddConformer(conf)

    def no_fallback(*_args, **_kwargs):
        pytest.fail("the native bond-order path can solve this selected connectivity")

    monkeypatch.setattr(tmc, "AC2mol", no_fallback)
    for order in (list(range(6)), list(reversed(range(6)))):
        mol = Chem.RenumberAtoms(source, order)
        coords = mol.GetConformer().GetPositions()
        perceived, _ = get_tmc_mol(None, 0, graph=(mol, coords))
        Chem.SanitizeMol(perceived)
        assert [a.GetAtomicNum() for a in perceived.GetAtoms()] == [a.GetAtomicNum() for a in mol.GetAtoms()]
        assert perceived.GetConformer().GetPositions() == pytest.approx(coords)
        assert {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in perceived.GetBonds()} == {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
        }
        assert Chem.MolToSmiles(perceived) == f"[Ag+].[O-][{element}+3]([O-])([O-])[O-]"
        assert Chem.GetFormalCharge(perceived) == 0


def test_titanium_is_never_read_above_its_four_valence_electrons():
    """Ti(IV) is titanium's highest oxidation state; four chlorides asking for more must be refused."""
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    chlorines = [rw.AddAtom(Chem.Atom(17)) for _ in range(4)]
    for chlorine in chlorines:
        rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(titanium, Point3D(0, 0, 0))
    for chlorine, point in zip(chlorines, [(2.3, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, -2.3, 0)], strict=True):
        conf.SetAtomPosition(chlorine, Point3D(*point))
    mol.AddConformer(conf)
    coords = mol.GetConformer().GetPositions()

    get_tmc_mol(None, 0, graph=(mol, coords))  # Ti+4 at four Cl-: at the cap, not over it

    with pytest.raises(ValueError, match=r"Ti\+8 is an impossible oxidation state"):
        get_tmc_mol(None, 4, graph=(mol, coords))  # same ligands, asked to give up two more electrons


@pytest.mark.parametrize("element", ["Sn", "Ge", "Pb"])
def test_trimethylstannyl_platinum_trichloride_reads_a_closed_shell_stannyl_anion(element):
    """Every metal-ligand bond is cut before ligand perception, whatever the donor element.

    An element list had left the Pt-Sn bond uncut, so the stannyl never got ligand perception and read as a
    neutral Sn on Pt(I).
    """
    source = Chem.MolFromSmiles(f"C[{element}](C)(C)[Pt](Cl)(Cl)Cl", sanitize=False)
    source.UpdatePropertyCache(strict=False)
    source = Chem.AddHs(source)
    assert rdDistGeom.EmbedMolecule(source, randomSeed=7, useRandomCoords=True) == 0
    for atom in source.GetAtoms():
        atom.SetNoImplicit(True)

    out, _coords = get_tmc_mol(None, -2, graph=(source, source.GetConformer().GetPositions()))

    assert [(a.GetSymbol(), a.GetFormalCharge()) for a in out.GetAtoms() if a.GetAtomicNum() > 9] == [
        (element, -1),
        ("Pt", 2),
        *[("Cl", -1)] * 3,
    ]
    assert not any(a.GetNumRadicalElectrons() for a in out.GetAtoms())


def test_bis_silylamido_cyclopentadienyl_zirconium_chloride_reads_as_zirconium_four():
    """USUQAB's constrained-geometry ligand: a Cp ring with two Me2Si-NMe arms, all seven atoms on Zr.

    Each amido nitrogen is bonded only to saturated carbons and silicons with no lone pair to offer as
    a pi partner, so `_saturated_donor_charges` charges it before the native search runs. Charged up
    front, the native ladder reads the trianion (Cp- and two amides) that NBO gives, at Zr+4.
    """
    ligand = Chem.AddHs(Chem.MolFromSmiles("C[N-][Si](C)(C)C1=CC([Si](C)(C)[N-]C)=C[CH-]1"))
    assert rdDistGeom.EmbedMolecule(ligand, randomSeed=7) == 0
    rw = Chem.RWMol(ligand)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    donors = [atom.GetIdx() for atom in ligand.GetAtoms() if atom.IsInRing() or atom.GetAtomicNum() == 7]
    zirconium, chlorine = rw.AddAtom(Chem.Atom(40)), rw.AddAtom(Chem.Atom(17))
    for donor in [*donors, chlorine]:
        rw.AddBond(zirconium, donor, Chem.BondType.SINGLE)
    source = rw.GetMol()
    source.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(source.GetNumAtoms())  # Zr at the origin: a graph= read never uses metal positions
    for index, point in enumerate(ligand.GetConformer().GetPositions()):
        conf.SetAtomPosition(index, Point3D(*point))
    conf.SetAtomPosition(chlorine, Point3D(0, 0, 2.4))
    source.AddConformer(conf, assignId=True)

    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        perceived, _coords = get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))

        charges = {atom.GetSymbol(): 0 for atom in perceived.GetAtoms()}
        for atom in perceived.GetAtoms():
            charges[atom.GetSymbol()] += atom.GetFormalCharge()
        assert charges == {"Zr": 4, "Cl": -1, "N": -2, "C": -1, "Si": 0, "H": 0}


def _boratacyclopentadienyl_titanium_trichloride():
    """Ti bound to an eta5-borole ring (B + 4 CH) and three chlorides, borole ring not yet charged.

    Borole is non-aromatic neutral and aromatic (6 pi electrons, isoelectronic with cyclopentadienide)
    as its dianion: `_fast_bond_orders` ranks the aromatic q=-2 ring above the neutral q=0 one regardless
    of hint. At three chlorides the q=-2 ring reads Ti+5 (over the four-electron cap); only the
    lower-ranked neutral ring keeps Ti in range, so this is a real case for the rescue search, not a
    synthetic one.
    """
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    boron = rw.AddAtom(Chem.Atom(5))
    ring = [boron] + [rw.AddAtom(Chem.Atom(6)) for _ in range(4)]
    for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    hydrogens = []
    for atom in ring:
        hydrogen = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(atom, hydrogen, Chem.BondType.SINGLE)
        hydrogens.append(hydrogen)
        rw.AddBond(titanium, atom, Chem.BondType.SINGLE)
    chlorines = [rw.AddAtom(Chem.Atom(17)) for _ in range(3)]
    for chlorine in chlorines:
        rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    mol = rw.GetMol()

    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(titanium, Point3D(0, 0, 0))
    for i, atom in enumerate(ring):
        angle = 2 * math.pi * i / len(ring)
        conf.SetAtomPosition(atom, Point3D(2.2 * math.cos(angle), 2.2 * math.sin(angle), 1.9))
    for i, hydrogen in enumerate(hydrogens):
        angle = 2 * math.pi * i / len(hydrogens)
        conf.SetAtomPosition(hydrogen, Point3D(3.3 * math.cos(angle), 3.3 * math.sin(angle), 2.6))
    for i, chlorine in enumerate(chlorines):
        angle = 2 * math.pi * i / len(chlorines)
        conf.SetAtomPosition(chlorine, Point3D(2.3 * math.cos(angle), 2.3 * math.sin(angle), -1.9))
    mol.AddConformer(conf)
    return mol, mol.GetConformer().GetPositions()


def test_over_cap_borole_ring_charge_is_rescued_by_its_neutral_form():
    """The aromatic dianion ring reads Ti+5 and must be refused outright; the rescue search instead
    tries the ring's own lower-ranked neutral form, which keeps Ti+3, and takes it."""
    mol, coords = _boratacyclopentadienyl_titanium_trichloride()

    out, _xyz = get_tmc_mol(None, 0, graph=(mol, coords))

    titanium = next(a for a in out.GetAtoms() if a.GetAtomicNum() == 22)
    boron = next(a for a in out.GetAtoms() if a.GetAtomicNum() == 5)
    assert titanium.GetFormalCharge() == 3  # Ti+3: neutral ring (0) + three Cl- (-3)
    assert boron.GetFormalCharge() == 0


def _bis_borole_titanium_trichloride(chlorides=3, lower_boron_substituent=1):
    """Ti between two eta5-borole rings (B + 4 CH each, above and below) with `chlorides` in the waist.

    `lower_boron_substituent` is the atomic number on the lower ring's boron, in place of its hydrogen.
    """
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    points = {titanium: (0.0, 0.0, 0.0)}
    for side, substituent in ((1, 1), (-1, lower_boron_substituent)):
        ring = [rw.AddAtom(Chem.Atom(5))] + [rw.AddAtom(Chem.Atom(6)) for _ in range(4)]
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            rw.AddBond(a, b, Chem.BondType.SINGLE)
        for i, atom in enumerate(ring):
            angle = 2 * math.pi * i / len(ring)
            points[atom] = (1.2 * math.cos(angle), 1.2 * math.sin(angle), 2.0 * side)
            hydrogen = rw.AddAtom(Chem.Atom(substituent if i == 0 else 1))
            rw.AddBond(atom, hydrogen, Chem.BondType.SINGLE)
            points[hydrogen] = (2.2 * math.cos(angle), 2.2 * math.sin(angle), 2.3 * side)
            rw.AddBond(titanium, atom, Chem.BondType.SINGLE)
    for i in range(chlorides):
        chlorine = rw.AddAtom(Chem.Atom(17))
        rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
        points[chlorine] = (2.4 * math.cos(2 * math.pi * i / 3), 2.4 * math.sin(2 * math.pi * i / 3), 0.0)
    rw.UpdatePropertyCache(strict=False)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in points.items():
        conf.SetAtomPosition(index, Point3D(*point))
    mol.AddConformer(conf)
    return mol, mol.GetConformer().GetPositions()


def test_two_borole_rings_on_titanium_trichloride_are_both_read_neutral():
    """Each aromatic borole dianion ranks first, but two read Ti+7 and one alone still Ti+5: only both neutral fit."""
    mol, coords = _bis_borole_titanium_trichloride()

    out, _coords = get_tmc_mol(None, 0, graph=(mol, coords))

    titanium = next(a for a in out.GetAtoms() if a.GetAtomicNum() == 22)
    assert titanium.GetFormalCharge() == 3
    assert [a.GetFormalCharge() for a in out.GetAtoms() if a.GetAtomicNum() == 5] == [0, 0]
    assert not any(a.GetNumRadicalElectrons() for a in out.GetAtoms())
    assert out.GetProp("_rxembedChargeRescue").startswith("Ti+7 is over its 4 valence electrons")


def test_rescued_ring_on_borole_fluoroborole_titanium_chloride_does_not_follow_atom_order():
    """Either borole dianion alone neutralised brings Ti+5 to Ti+3; which ring is chosen must not follow file order."""
    source, _coords = _bis_borole_titanium_trichloride(chlorides=1, lower_boron_substituent=9)
    readings = set()
    for seed in range(4):
        order = list(range(source.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        mol = Chem.RenumberAtoms(source, order)
        out, _coords = get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))
        assert out.GetProp("_rxembedChargeRescue").endswith("read Ti+3 with 1 ligand charge(s) changed")
        readings.add(Chem.MolToSmiles(out))
    assert len(readings) == 1


def test_titanium_tetrachloride_takes_one_chloride_electron_per_charge_past_titanium_four():
    """Asked for radicals, each unit of charge past Ti(IV) takes one chloride's electron until none is left."""
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    chlorines = [rw.AddAtom(Chem.Atom(17)) for _ in range(4)]
    for chlorine in chlorines:
        rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for chlorine, point in zip(chlorines, [(2.3, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, -2.3, 0)], strict=True):
        conf.SetAtomPosition(chlorine, Point3D(*point))
    mol.AddConformer(conf)
    coords = mol.GetConformer().GetPositions()

    assert not get_tmc_mol(None, 0, graph=(mol, coords))[0].HasProp("_rxembedChargeRescue")
    out, _coords = get_tmc_mol(None, 4, graph=(mol, coords), radicals=True)

    assert [(a.GetSymbol(), a.GetFormalCharge(), a.GetNumRadicalElectrons()) for a in out.GetAtoms()] == [
        ("Ti", 4, 0),
        *[("Cl", 0, 1)] * 4,
    ]
    assert out.GetProp("_rxembedChargeRescue").startswith("Ti+8 is over its 4 valence electrons at charge=4")
    with pytest.raises(ValueError, match=r"Ti\+9 is an impossible oxidation state"):
        get_tmc_mol(None, 5, graph=(mol, coords), radicals=True)


def test_ligand_charge_pool_is_empty_for_a_monatomic_ligand():
    """A single-atom ligand (a halide, a hydride) has no alternate resonance form to search."""
    chloride = Chem.MolFromSmiles("[Cl-]")

    assert tmc._ligand_charge_pool(chloride, -1, [0]) == []


def test_invented_hydrogen_gate_counts_bracket_h_but_not_xyz_h_atoms():
    params = Chem.SmilesParserParams()
    params.removeHs = False
    virtual = Chem.MolFromSmiles("[SH][SH]", params)
    explicit = Chem.MolFromSmiles("[S]([H])[S][H]", params)

    assert _invented_hydrogens(virtual) == 2
    assert _invented_hydrogens(explicit) == 0


def test_direct_xyz_keeps_only_supplied_hydrogens(tmp_path, monkeypatch):
    path = tmp_path / "silver_aqua_chloride.xyz"
    path.write_text("5\ncharge=0\nAg 0 0 0\nO 2.2 0 0\nCl -2.4 0 0\nH 2.78 .76 0\nH 2.78 -.76 0\n")

    def no_fallback(*_args, **_kwargs):
        pytest.fail("native bond orders suffice for water and chloride")

    monkeypatch.setattr(tmc, "AC2mol", no_fallback)
    mol, coords = get_tmc_mol(path, 0)

    Chem.SanitizeMol(mol)
    assert [atom.GetSymbol() for atom in mol.GetAtoms()] == ["Ag", "O", "Cl", "H", "H"]
    assert coords == pytest.approx(mol.GetConformer().GetPositions())
    assert _invented_hydrogens(mol) == 0
    assert Chem.MolToSmiles(mol) == "[H][O]([H])->[Ag+]<-[Cl-]"


def test_later_clean_native_bond_order_candidate_skips_resonance(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("C=O"))  # formaldehyde keeps an unclosed valence, so the ladder runs
    calls, limits, assigned = [], [], []

    def clean(candidate, _coordinating_atoms, resonate=True):
        calls.append(resonate)
        if resonate:
            raise AssertionError("a clean native assignment entered resonance enumeration")
        dirty = assigned[-1] != 2
        return [(candidate, int(dirty), 0, 0, 0, 0)]

    def assign(*_args, **kw):
        assigned.append(kw["charge"])
        limits.append(kw.get("maxIterations", 0))

    monkeypatch.setattr(tmc.rdDetermineBonds, "DetermineBondOrders", assign)
    monkeypatch.setattr(tmc, "lig_checks", clean)

    assert tmc._fast_bond_orders(mol, 0, [1]) is not None
    assert limits
    assert all(limits)
    assert calls
    assert not any(calls)


def _flat_explicit_ligand(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def test_sulfoxonium_ylide_reads_one_lewis_form_whatever_the_bond_order():
    """A reader adds bonds in distance order; the bond-order search keeps its first form, so must not see it."""
    source = _flat_explicit_ligand("C[S+](=O)([CH2-])[CH2-]")
    ylides = [
        atom.GetIdx()
        for atom in source.GetAtoms()
        if atom.GetAtomicNum() == 6 and sum(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors()) == 2
    ]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in source.GetBonds()]
    rng = random.Random(0)
    forms = set()
    for _ in range(8):
        rng.shuffle(bonds)
        rw = Chem.RWMol(source)
        for begin, end in bonds:
            rw.RemoveBond(begin, end)
        for begin, end in bonds:
            rw.AddBond(begin, end, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        ligand, _charge = get_lig_mol(mol, -1, ylides)
        forms.add(
            frozenset(
                frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                for bond in ligand.GetBonds()
                if bond.GetBondType() == Chem.BondType.DOUBLE
            )
        )
    assert len(ylides) == 2
    assert len(forms) == 1, "the ylide's S=C bond moved with the bond insertion order"


def test_native_charge_search_keeps_a_closed_shell_donor_with_remote_charge():
    source = _flat_explicit_ligand("[O-]P(c1ccccc1)c1cccc2ccc[c-]c12")
    source_donors = [
        atom.GetIdx()
        for atom in source.GetAtoms()
        if atom.GetAtomicNum() == 15
        or (atom.GetAtomicNum() == 6 and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0)
    ]
    seen = set()
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        new_of = {old: new for new, old in enumerate(order)}
        ligand, charge = get_lig_mol(mol, 0, [new_of[donor] for donor in source_donors])

        phosphorus = next(atom for atom in ligand.GetAtoms() if atom.GetAtomicNum() == 15)
        oxygen = next(atom for atom in ligand.GetAtoms() if atom.GetAtomicNum() == 8)
        assert charge == Chem.GetFormalCharge(ligand) == -2
        assert phosphorus.GetFormalCharge() == 0
        assert ligand.GetBondBetweenAtoms(phosphorus.GetIdx(), oxygen.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        seen.add(Chem.MolToSmiles(ligand))

    assert len(seen) == 1


def test_charge_hint_only_orders_the_same_closed_shell_ligand_candidates():
    source = _flat_explicit_ligand("C=CC=C")
    seen = set()
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        donors = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        for hint in (0, -2):
            ligand, charge = get_lig_mol(mol, hint, donors)
            assert charge == 0
            assert not sum(atom.GetNumRadicalElectrons() for atom in ligand.GetAtoms())
            seen.add(Chem.MolToSmiles(ligand))
    assert len(seen) == 1


def test_graph_equivalent_nitrogen_oxide_uses_the_electronic_charge_hint():
    mol = _flat_explicit_ligand("[O-][N+](=O)[O-]")
    donors = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]

    ligand, charge = tmc._fast_bond_orders(mol, -3, donors)
    nitrogen = next(atom for atom in ligand.GetAtoms() if atom.GetAtomicNum() == 7)

    assert charge == Chem.GetFormalCharge(ligand) == -3
    assert nitrogen.GetFormalCharge() == 0


def _c7h7_ring():
    """A bare eta7-cycloheptatrienyl ring: 7 CH, single bonds only, no charge or bond order proposed yet."""
    rw = Chem.RWMol()
    ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(7)]
    for atom in ring:
        rw.GetAtomWithIdx(atom).SetNoImplicit(True)
    for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    for carbon in ring:
        hydrogen = rw.AddAtom(Chem.Atom(1))
        rw.GetAtomWithIdx(hydrogen).SetNoImplicit(True)
        rw.AddBond(carbon, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, ring


def test_eta7_cycloheptatrienyl_ring_prefers_the_least_charge_separated_ion():
    """A bare eta7-C7H7 ring (KIZNAI, KOSMAG): q=-7 and q=-3 both spread +/-1 over every ring atom, so
    invented-H, radicals, pairless donors, aromaticity and charge concentration all tie between them.
    The hint (-1, an off-ladder pi-electron count, not a candidate charge) cannot break the tie either,
    so the least charge-separated valid assignment should win: the textbook 10-pi-electron trianion,
    not the fully-carbanion heptaanion.
    """
    mol, ring = _c7h7_ring()

    ligand, charge = tmc._fast_bond_orders(mol, -1, ring)

    assert charge == Chem.GetFormalCharge(ligand) == -3


@pytest.mark.parametrize("hint", [-1, 1])
def test_ligand_charge_ladder_stays_bounded_regardless_of_hint_sign(hint):
    """Nothing should let the ladder settle for the fully-carbanion heptaanion over the textbook
    trianion just because a Hueckel hint landed on the wrong side of zero: the accepted ligand's own
    reported charge must match its atoms' formal-charge sum, and no atom should carry a charge no real
    Lewis structure would give a plain hydrocarbon donor.
    """
    mol, ring = _c7h7_ring()

    ligand, charge = get_lig_mol(mol, hint, ring)

    assert charge == Chem.GetFormalCharge(ligand)
    assert abs(charge) <= 3
    assert max(abs(atom.GetFormalCharge()) for atom in ligand.GetAtoms()) <= 2


def test_monodentate_nitrate_donor_form_is_atom_order_invariant():
    source = _flat_explicit_ligand("[O-][N+](=O)[O-]")
    for order in permutations(range(source.GetNumAtoms())):
        mol = Chem.RenumberAtoms(source, order)
        new_of = {old: new for new, old in enumerate(order)}
        donor = new_of[0]

        ligand, charge = get_lig_mol(mol, -1, [donor])
        nitrogen = next(atom.GetIdx() for atom in ligand.GetAtoms() if atom.GetAtomicNum() == 7)

        assert charge == -1
        assert ligand.GetAtomWithIdx(donor).GetFormalCharge() == -1
        assert ligand.GetBondBetweenAtoms(donor, nitrogen).GetBondType() == Chem.BondType.SINGLE


def test_graph_equivalent_charge_uses_eht_hint_or_errors(monkeypatch):
    mol = Chem.MolFromSmiles("CC")

    def assign(*_args, **kw):
        if kw["charge"] not in {0, 2}:
            raise ValueError("only two graph-equivalent candidates")

    def checks(candidate, _donors, resonate=True):
        assert not resonate
        return [(candidate, 0, 0, 0, 0, 0)]

    monkeypatch.setattr(tmc.rdDetermineBonds, "DetermineBondOrders", assign)
    monkeypatch.setattr(tmc, "lig_checks", checks)
    monkeypatch.setattr(tmc, "_donor_localised_candidates", lambda *_args: ())

    assert tmc._fast_bond_orders(mol, 0, [])[1] == 0
    assert tmc._fast_bond_orders(mol, 2, [])[1] == 2
    with pytest.raises(ValueError, match=r"graph-equivalent.*extended Hückel proposed q=-2"):
        tmc._fast_bond_orders(mol, -2, [])


@pytest.mark.parametrize(("distance", "expected_charge"), [(1.22, 0), (1.43, -2)])
def test_charge_hint_needs_bond_length_agreement(distance, expected_charge):
    mol = _flat_explicit_ligand("O=C(F)F")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((distance, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.66, 1.143, 0.0), (-0.66, -1.143, 0.0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    ligand, charge = tmc._fast_bond_orders(mol, -2, [0, 1])

    assert charge == expected_charge
    assert ligand.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == (2.0 if charge == 0 else 1.0)


def test_closed_shell_cyclopentadienyl_anion_beats_a_neutral_radical():
    mol = _flat_explicit_ligand("[CH-]1C=CC=C1")
    donors = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]

    ligand, charge = get_lig_mol(mol, 1, donors)

    assert charge == -1
    assert sum(atom.GetNumRadicalElectrons() for atom in ligand.GetAtoms()) == 0
    assert sum(abs(atom.GetFormalCharge()) for atom in ligand.GetAtoms()) == 1


def test_carbon_monoxide_avoids_a_divalent_carbanion_for_any_charge_hint():
    source = _flat_explicit_ligand("[C-]#[O+]")
    seen = set()
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        carbon = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        for hint in (-2, 0, 2):
            ligand, charge = get_lig_mol(mol, hint, [carbon])
            assert charge == 0
            seen.add(Chem.MolToSmiles(ligand))
    assert seen == {"[C-]#[O+]"}


def test_donor_localised_search_checks_the_combination_count_before_enumerating(monkeypatch):
    mol = _flat_explicit_ligand("C" * 20)

    def materialized(*_args, **_kwargs):
        raise AssertionError("the over-cap combination iterator was constructed")

    monkeypatch.setattr(tmc, "combinations", materialized)

    assert tmc._donor_localised_candidates(mol, -10, range(20)) == []


def test_equivalent_oxygen_charge_is_localized_on_the_donor_after_renumbering():
    source = Chem.MolFromSmiles("[O-]P(=O)(O)O")
    phosphorus = next(a.GetIdx() for a in source.GetAtoms() if a.GetAtomicNum() == 15)
    donor = next(
        bond.GetOtherAtomIdx(phosphorus)
        for bond in source.GetAtomWithIdx(phosphorus).GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    expected = None
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        mol = Chem.RenumberAtoms(source, order)
        new_of = {old: new for new, old in enumerate(order)}
        fixed = tmc._localise_donor_pairs(mol, [new_of[donor]])
        localized = fixed.GetAtomWithIdx(new_of[donor])

        assert localized.GetFormalCharge() == -1
        assert fixed.GetBondBetweenAtoms(new_of[phosphorus], new_of[donor]).GetBondType() == Chem.BondType.SINGLE
        smiles = Chem.MolToSmiles(fixed)
        expected = smiles if expected is None else expected
        assert smiles == expected


def test_donor_localisation_does_not_invent_an_unproven_lewis_form():
    fixtures = [
        (
            Chem.MolFromSmiles("C[S](=O)(=[CH2])[CH2-]"),
            lambda mol: [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() == 2],
        ),
        (Chem.MolFromSmiles("C[P]1(O)(=[S]->[Ni+2]<-[S-]1)"), lambda _mol: None),
    ]
    for source, donors in fixtures:
        expected = Chem.MolToSmiles(source)
        for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
            mol = Chem.RenumberAtoms(source, order)
            assert Chem.MolToSmiles(tmc._localise_donor_pairs(mol, donors(mol))) == expected


def test_bonded_codonors_are_not_pruned_by_radial_distance(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number in (46, 6, 16):
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for pair in ((0, 1), (0, 2), (1, 2)):
        rw.AddBond(*pair, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    conf = Chem.Conformer(3)
    for atom, point in enumerate(((0, 0, 0), (2, 0, 0), (2.5, 0.5, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)
    monkeypatch.setattr(tmc, "get_proposed_ligand_charge", lambda _mol: 0)
    monkeypatch.setattr(tmc, "get_lig_mol", lambda ligand, charge, _donors: (ligand, charge))

    perceived, _coords = get_tmc_mol(None, 0, graph=(mol, conf.GetPositions()))
    metal = next(atom for atom in perceived.GetAtoms() if atom.GetAtomicNum() == 46)

    assert {neighbor.GetAtomicNum() for neighbor in metal.GetNeighbors()} == {6, 16}


def test_multimetal_reassembly_preserves_supplied_oxidation_states(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number, charge in ((26, 2), (25, 1), (17, -1), (17, -1)):
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    rw.AddBond(2, 0, Chem.BondType.DATIVE)
    rw.AddBond(3, 1, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    mol.AddConformer(Chem.Conformer(4))
    monkeypatch.setattr(tmc, "get_proposed_ligand_charge", Chem.GetFormalCharge)
    monkeypatch.setattr(tmc, "get_lig_mol", lambda ligand, charge, _donors: (ligand, charge))

    perceived, _coords = get_tmc_mol(None, 1, graph=(mol, mol.GetConformer().GetPositions()))

    assert {atom.GetAtomicNum(): atom.GetFormalCharge() for atom in perceived.GetAtoms() if atom.GetIdx() < 2} == {
        26: 2,
        25: 1,
    }
    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals"):
        get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))


def test_sigma_polyhydride_bridge_is_not_pruned_as_a_haptic_face(monkeypatch):
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 1, 1, 1):
        atom = Chem.Atom(atomic_number)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    rw.GetAtomWithIdx(1).SetFormalCharge(-1)
    for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4)):
        rw.AddBond(*pair, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    conf = Chem.Conformer(5)
    for atom, point in enumerate(((0, 0, 0), (2.1, 0, 0), (1.6, 0, 0), (1.7, 0.1, 0), (3.2, 0, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)
    monkeypatch.setattr(tmc, "get_proposed_ligand_charge", Chem.GetFormalCharge)
    monkeypatch.setattr(tmc, "get_lig_mol", lambda ligand, charge, _donors: (ligand, charge))

    perceived, _coords = get_tmc_mol(None, 0, graph=(mol, conf.GetPositions()))
    metal = next(atom for atom in perceived.GetAtoms() if atom.GetAtomicNum() == 26)

    assert {neighbor.GetAtomicNum() for neighbor in metal.GetNeighbors()} == {1, 5}


def test_three_centre_borohydride_completes_a_missing_boron_hydrogen_edge():
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 1, 1, 1, 1):
        rw.AddAtom(Chem.Atom(atomic_number))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(0, 2, Chem.BondType.SINGLE)
    for hydrogen in (3, 4, 5):
        rw.AddBond(1, hydrogen, Chem.BondType.SINGLE)
    mol = rw.GetMol()

    completed = tmc._reconnect_metal_hydride_bridges(
        mol,
        ((0, 0, 0), (2.076, 0, 0), (1.307, 1.087, 0), (3.1, 0, 0), (2.1, 1.1, 0), (2.1, -1.1, 0)),
    )

    assert completed.GetBondBetweenAtoms(1, 2) is not None
    assert completed.GetAtomWithIdx(1).GetFormalCharge() == -1


def test_borohydride_bridge_assigns_one_nearest_boron_per_hydrogen():
    rw = Chem.RWMol()
    for atomic_number in (26, 5, 5, 1):
        rw.AddAtom(Chem.Atom(atomic_number))
    rw.AddBond(0, 3, Chem.BondType.SINGLE)
    mol = rw.GetMol()

    completed = tmc._reconnect_metal_hydride_bridges(
        mol,
        ((0, 0, 0), (1.5, 0, 0), (1.8, 0, 0), (0.2, 0, 0)),
    )

    assert completed.GetBondBetweenAtoms(1, 3) is not None
    assert completed.GetBondBetweenAtoms(2, 3) is None


def test_ligand_checks_stream_resonance_candidates():
    """`lig_checks(resonate=True)` yields benzene's own aromatic form with no formal charges."""
    candidates = list(tmc.lig_checks(Chem.MolFromSmiles("c1ccccc1"), ()))

    assert len(candidates) == 1
    mol, positive, negative, n_aromatic, invented, pairless = candidates[0]
    assert (positive, negative, pairless) == (0, 0, 0)
    assert (n_aromatic, invented) == (6, 6)  # 6 aromatic carbons, each with one implicit ring H
    assert Chem.MolToSmiles(mol) == "c1ccccc1"


def _perceive(path, charge=0, **kw):
    mol, _coords = get_tmc_mol(path, charge, **kw)
    return mol


def _shuffled(path, tmp_dir, seed):
    """The same molecule, same coordinates, with the atom lines in a different order."""
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    body = lines[2 : 2 + n]
    random.Random(seed).shuffle(body)
    out = tmp_dir / f"{path.stem}_{seed}.xyz"
    out.write_text("\n".join([lines[0], lines[1], *body]) + "\n")
    return out


def _ferrocene_xyz(tmp_path):
    """Write `metal_fixtures.ferrocene()` to `.xyz`, adding each Cp ring carbon's hydrogen radially.

    An `.xyz` file carries only elements and coordinates, so the source mol's own bond orders and formal
    charges (needed to build a valid dative RDKit graph, not a real Cp-H valence) do not need to sanitize
    once the hydrogens are added; xyz2mol_tmc perceives bonds and charges from the geometry alone.
    """
    rw = Chem.RWMol(ferrocene())
    conf = rw.GetConformer()
    for idx in range(1, 11):  # atom 0 is Fe; 1-5 and 6-10 are the two Cp rings
        p = conf.GetAtomPosition(idx)
        r = math.hypot(p.x, p.y)
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(idx, h, Chem.BondType.SINGLE)
        conf.SetAtomPosition(h, Point3D(p.x * (r + 1.08) / r, p.y * (r + 1.08) / r, p.z))
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    path = tmp_path / "ferrocene.xyz"
    Chem.MolToXYZFile(mol, str(path))
    return path


@pytest.mark.parametrize(
    "path",
    [_ferrocene_xyz, lambda _tmp_path: EXAMPLES_DIR / "ru-co.xyz"],
    ids=["ferrocene", "ru-co"],
)
def test_canonical_under_atom_reordering(path, tmp_path):
    path = path(tmp_path)
    seen = {Chem.MolToSmiles(_perceive(path))}
    for seed in (1, 2, 3):
        seen.add(Chem.MolToSmiles(_perceive(_shuffled(path, tmp_path, seed))))
    assert len(seen) == 1, f"{path.name} gave {len(seen)} strings for one molecule: {sorted(seen)}"


def test_ligands_close_their_own_valences(tmp_path):
    mol = _perceive(_ferrocene_xyz(tmp_path))
    rw = Chem.RWMol(mol)
    metals = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "Fe"]
    for i in sorted(metals, reverse=True):
        rw.RemoveAtom(i)
    stripped = rw.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    for frag in Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False):
        work = Chem.RWMol(frag)
        for a in work.GetAtoms():
            if a.GetAtomicNum() != 1:
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
        m = work.GetMol()
        Chem.SanitizeMol(m)  # raises on an over-valent ligand
        invented = sum(a.GetNumImplicitHs() for a in m.GetAtoms() if a.GetAtomicNum() != 1)
        assert invented == 0, f"perception left a ligand undervalent by {invented} H"


def test_amido_donor_charge_keeps_a_chelates_oxime_nitrogen_sp2(tmp_path):
    """Charging a Tc-oxime chelate's two amido nitrogens up front lets the whole ligand's bond-order search
    succeed, so its two separate oxime nitrogens read as ordinary sp2 imines rather than a charged sp form.

    A hand-built single amido donor does not reproduce the failure (needs the whole ligand graph), so this
    embeds MOCQIE's published graph fresh (no CSD coordinates) and re-reads the resulting xyz.
    """
    # a bis(dimethylglyoximato) Tc(V) chelate: two amido N donors, two oxime N=C(-O(H)) donors
    smiles = (
        "CC1=[N]([O-])->[Tc+5]23(<-[O-2])<-[N](O)=C(C)C(C)(C)[N-]->2C[C@@H](C#N)C[N-]->3C1(C)C "
        "|atomProp:2.atomNote.s1:4.atomNote.SPY-delta:5.atomNote.s0:6.atomNote.s4:13.atomNote.s3:19.atomNote.s2|"
    )
    ensemble = rx.embed(smiles, n=1, seed=42)
    path = tmp_path / "mocqie.xyz"
    Chem.MolToXYZFile(ensemble.mol, str(path), confId=ensemble.ids[0])
    mol = rx.read_xyz(str(path), charge=0, bond_orders="xyz2mol")

    for index in (13, 19):  # the two amido N donors
        atom = mol.GetAtomWithIdx(index)
        assert atom.GetFormalCharge() == -1
        assert atom.GetHybridization() == Chem.HybridizationType.SP3
    for index in (2, 6):  # the two oxime N=C donors
        atom = mol.GetAtomWithIdx(index)
        assert atom.GetFormalCharge() == 0
        assert atom.GetHybridization() == Chem.HybridizationType.SP2


def test_cyclotriphosphine_ring_donor_stays_single_bonded(tmp_path):
    """A saturated P3H3 ring (cyclotriphosphine) donating to a metal through all three phosphines must keep
    single P-P bonds, not the P#P one RDKit's compiled search promotes without this fix.
    """
    pp, ph = 2.21, 1.42  # A: a typical P-P single bond and P-H bond
    radius = pp / math.sqrt(3)
    angles = (0, 120, 240)
    p_pos = [(radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), 0.0) for a in angles]
    scale = (radius + ph) / radius
    h_pos = [(x * scale, y * scale, z * scale) for x, y, z in p_pos]

    lines = ["7", "charge=0"]
    for symbol, pos in [
        *zip(("P", "P", "P"), p_pos, strict=True),
        *zip(("H", "H", "H"), h_pos, strict=True),
        ("Ni", (0.0, 0.0, 2.2)),
    ]:
        lines.append(f"{symbol} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    path = tmp_path / "cyclotriphosphine_nickel.xyz"
    path.write_text("\n".join(lines) + "\n")

    # Pin the metal's charge: three plain phosphine donors on Ni(0) is unambiguous, and skipping the
    # oxidation-state search keeps this test about the ring's bond orders, not that separate guess.
    mol = rx.read_xyz(str(path), charge=0, bond_orders="xyz2mol", metal_charges={6: 0})

    ring_bonds = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() < 3 and b.GetEndAtomIdx() < 3]
    assert len(ring_bonds) == 3
    assert not any(b.GetBondType() == Chem.BondType.TRIPLE for b in ring_bonds)
