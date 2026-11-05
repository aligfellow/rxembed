"""`perception/xyz2mol_tmc.py`: rxembed's changes to the vendored TMC perceiver, read through its public functions."""

import math
import random

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

from rxembed.perception.xyz2mol_tmc import get_lig_mol, get_tmc_mol
from tests.geometry_fixtures import embedded


def _bare(mol):
    """Return `mol` as a perceiver graph: single bonds, no charges, aromaticity or implicit H."""
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def _on_metal(ligand, metal, donors, extra=()):
    """Return the bare `ligand` with its conformer, bonded to a `metal` at the origin through `donors`.

    `extra` are ``(symbol, point)`` atoms bonded to the metal alone.
    """
    rw = Chem.RWMol(_bare(ligand))
    centre = rw.AddAtom(Chem.Atom(metal))
    for symbol, _point in extra:
        donors = [*donors, rw.AddAtom(Chem.Atom(symbol))]
    for donor in donors:
        rw.AddBond(centre, donor, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate([*ligand.GetConformer().GetPositions(), (0, 0, 0), *(p for _s, p in extra)]):
        conformer.SetAtomPosition(index, Point3D(*point))
    mol.AddConformer(conformer, assignId=True)
    return mol


def _stannyl_platinum():
    """Return C3Pb-PtCl3 placed by RDKit: a Pt-Pb bond any element list of donors would have left uncut."""
    source = Chem.MolFromSmiles("C[Pb](C)(C)[Pt](Cl)(Cl)Cl", sanitize=False)
    source.UpdatePropertyCache(strict=False)
    source = Chem.AddHs(source)
    assert rdDistGeom.EmbedMolecule(source, randomSeed=7, useRandomCoords=True) == 0
    return _bare(source)


def _cyclotriphosphine_nickel():
    """Return a saturated P3H3 ring donating through all three phosphines to Ni(0), P-P 2.21 A and P-H 1.42 A."""
    radius = 2.21 / math.sqrt(3)
    p_points = [(radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), -2.2) for a in (0, 120, 240)]
    h_points = [(x * (radius + 1.42) / radius, y * (radius + 1.42) / radius, z) for x, y, z in p_points]
    ring = Chem.RWMol()
    for symbol in "PPPHHH":
        ring.AddAtom(Chem.Atom(symbol))
    for i in range(3):
        ring.AddBond(i, (i + 1) % 3, Chem.BondType.SINGLE)
        ring.AddBond(i, i + 3, Chem.BondType.SINGLE)
    ring = ring.GetMol()
    conformer = Chem.Conformer(6)
    for index, point in enumerate([*p_points, *h_points]):
        conformer.SetAtomPosition(index, Point3D(*point))
    ring.AddConformer(conformer)
    return _on_metal(ring, 28, [0, 1, 2])


def _bis_borole_titanium_chloride():
    """Return Ti between an eta5-borole and an eta5-fluoroborole (B + 4 CH each, above and below), and a chloride."""
    rw = Chem.RWMol()
    titanium = rw.AddAtom(Chem.Atom(22))
    points = {titanium: (0.0, 0.0, 0.0)}
    for side, substituent in ((1, 1), (-1, 9)):
        ring = [rw.AddAtom(Chem.Atom(5))] + [rw.AddAtom(Chem.Atom(6)) for _ in range(4)]
        for i, atom in enumerate(ring):
            rw.AddBond(atom, ring[(i + 1) % 5], Chem.BondType.SINGLE)
            angle = 2 * math.pi * i / 5
            points[atom] = (1.2 * math.cos(angle), 1.2 * math.sin(angle), 2.0 * side)
            outer = rw.AddAtom(Chem.Atom(substituent if i == 0 else 1))
            rw.AddBond(atom, outer, Chem.BondType.SINGLE)
            points[outer] = (2.2 * math.cos(angle), 2.2 * math.sin(angle), 2.3 * side)
            rw.AddBond(titanium, atom, Chem.BondType.SINGLE)
    chlorine = rw.AddAtom(Chem.Atom(17))
    rw.AddBond(titanium, chlorine, Chem.BondType.SINGLE)
    points[chlorine] = (2.4, 0.0, 0.0)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in points.items():
        conformer.SetAtomPosition(index, Point3D(*point))
    mol.AddConformer(conformer)
    return mol


def _read(mol, charge, order):
    """Return the Mol `get_tmc_mol` reads from `mol`'s graph renumbered to `order`, back in `mol`'s atom order."""
    renumbered = Chem.RenumberAtoms(mol, order)
    out, _coords = get_tmc_mol(None, charge, graph=(renumbered, renumbered.GetConformer().GetPositions()))
    return Chem.RenumberAtoms(out, [order.index(i) for i in range(len(order))])


def test_get_tmc_mol_reads_each_cut_ligand_in_any_order():
    """Every metal-ligand bond is cut, whatever the donor element; an amido N on saturated neighbours is charged before
    the search, so USUQAB's constrained-geometry ligand reads as the Cp- bis-amide trianion on Zr(IV)."""
    usuqab = embedded("C[N-][Si](C)(C)C1=CC([Si](C)(C)[N-]C)=C[CH-]1")
    donors = [atom.GetIdx() for atom in usuqab.GetAtoms() if atom.IsInRing() or atom.GetAtomicNum() == 7]
    for name, mol, charge, charges in (
        ("Me3Sn PtCl3", _stannyl_platinum(), -2, {"Pb": -1, "Pt": 2, "Cl": -3, "C": 0, "H": 0}),
        (
            "usuqab-zirconium-chloride",
            _on_metal(usuqab, 40, donors, [("Cl", (0, 0, 2.4))]),
            0,
            {"Zr": 4, "Cl": -1, "N": -2, "C": -1, "Si": 0, "H": 0},
        ),
    ):
        for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
            out = _read(mol, charge, order)
            found = dict.fromkeys(charges, 0)
            for atom in out.GetAtoms():
                found[atom.GetSymbol()] += atom.GetFormalCharge()
            assert (found, sum(atom.GetNumRadicalElectrons() for atom in out.GetAtoms())) == (charges, 0), name


def test_get_tmc_mol_keeps_a_saturated_ring_donor_single_bonded():
    """A P3H3 ring donating through all three phosphines keeps its P-P single bonds, not RDKit's P#P promotion."""
    out = _read(_cyclotriphosphine_nickel(), 0, list(range(7)))

    assert Chem.BondType.TRIPLE not in {out.GetBondBetweenAtoms(i, (i + 1) % 3).GetBondType() for i in range(3)}


def test_get_tmc_mol_rescues_an_over_cap_metal_in_any_order():
    """Either borole dianion alone neutralised brings Ti+5 to Ti+3; which ring is chosen must not follow file order."""
    source = _bis_borole_titanium_chloride()
    readings = set()
    for seed in range(4):
        order = list(range(source.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        mol = Chem.RenumberAtoms(source, order)
        out, _coords = get_tmc_mol(None, 0, graph=(mol, mol.GetConformer().GetPositions()))
        assert out.GetProp("_rxembedChargeRescue").endswith("read Ti+3 with 1 ligand charge(s) changed")
        readings.add(Chem.MolToSmiles(out))
    assert len(readings) == 1


def _iron_manganese_chlorides():
    """Return Fe(II) and Mn(I), each holding a chloride, as a graph with the metal charges given."""
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
    return mol


def test_get_tmc_mol_keeps_each_metals_given_oxidation_state():
    assert Chem.MolToSmiles(_read(_iron_manganese_chlorides(), 1, [0, 1, 2, 3])) == "[Cl-]->[Fe+2].[Cl-]->[Mn+]"


def test_get_tmc_mol_refuses_a_total_it_cannot_split(tmp_path):
    """The given split misses the total, or a planar [Pd2Cl6]2- file names no split at all."""
    path = tmp_path / "pd2cl6.xyz"
    path.write_text(
        "8\n\nCl -3.35 1.6 0\nPd -1.75 0 0\nCl -3.35 -1.6 0\nCl 0 1.65 0\n"
        "Pd 1.75 0 0\nCl 3.35 1.6 0\nCl 3.35 -1.6 0\nCl 0 -1.65 0\n"
    )
    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals: supplied"):
        _read(_iron_manganese_chlorides(), 0, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="cannot allocate oxidation states between multiple metals from coordinates"):
        get_tmc_mol(str(path), -2)


def test_get_tmc_mol_bonds_a_bridging_hydride_back_to_its_boron(tmp_path):
    """A kappa1-H borohydride's bridging H, bound to Fe, is left off its boron 1.6 A away by the covalent reader."""
    path = tmp_path / "borohydride.xyz"
    path.write_text("6\n\nFe 0 0 0\nH 1.7 0 0\nB 3.3 0 0\nH 3.7 1.1 0\nH 3.7 -0.55 0.95\nH 3.7 -0.55 -0.95\n")

    out, _coords = get_tmc_mol(str(path), 0)

    assert (out.GetBondBetweenAtoms(1, 2) is not None, out.GetAtomWithIdx(2).GetFormalCharge()) == (True, -1)


def test_get_lig_mol_takes_the_charge_its_hint_or_lengths_pick():
    """A nitrate graph reads as neutral N at the -3 the hint names; C(=O)F2 at -2 only where bond lengths agree."""
    nitrate = _bare(Chem.AddHs(Chem.MolFromSmiles("[O-][N+](=O)[O-]")))
    carbonyl_fluoride = _bare(Chem.MolFromSmiles("O=C(F)F"))
    conformer = Chem.Conformer(4)
    for atom, point in enumerate(((1.43, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.66, 1.143, 0.0), (-0.66, -1.143, 0.0))):
        conformer.SetAtomPosition(atom, Point3D(*point))
    carbonyl_fluoride.AddConformer(conformer)
    for ligand, hint, donors, read in (
        (nitrate, -3, [0, 2, 3], ("[O-]N([O-])[O-]", -3)),
        (carbonyl_fluoride, -2, [0, 1], ("[O-][C-](F)F", -2)),
    ):
        out, charge = get_lig_mol(ligand, hint, donors)
        assert (Chem.MolToSmiles(out), charge) == read, read


def test_get_lig_mol_reads_tropylium_at_least_separated_charge():
    """An eta7-C7H7's -1 pi-count hint is off its tied -3 and -7 readings, so the ladder takes -3, not -7."""
    ring = Chem.RWMol()
    for _ in range(7):
        ring.AddAtom(Chem.Atom(6))
    for i in range(7):
        ring.AddBond(i, (i + 1) % 7, Chem.BondType.SINGLE)
        ring.AddBond(i, ring.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)

    out, charge = get_lig_mol(_bare(ring.GetMol()), -1, list(range(7)))

    assert (charge, Chem.GetFormalCharge(out)) == (-3, -3)


def test_get_lig_mol_refuses_ties_its_hint_cannot_pick():
    """PUXZUE's bis(methyltetrazolyl)triazenide graph reads alike at -1 and +1, and the -3 hint names neither."""
    triazenide = _bare(Chem.AddHs(Chem.MolFromSmiles("Cn1nnc(N=N[N]c2nnn(C)n2)n1")))

    with pytest.raises(
        ValueError, match=r"ligand charges \[-1, 1\] are graph-equivalent.*explicit charged/dative SMILES"
    ):
        get_lig_mol(triazenide, -3, [3, 6, 9])
