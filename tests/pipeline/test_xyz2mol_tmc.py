"""Test canonical ligand ordering and bond-order ranking in the TMC perceiver."""

from __future__ import annotations

import random
from itertools import permutations
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed.pipeline import xyz2mol_tmc as tmc
from rxembed.pipeline.xyz2mol_tmc import _invented_hydrogens, get_lig_mol, get_tmc_mol

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmark" / "corpus"

# `benchmark/` is a local-only harness and is gitignored (AGENTS.md), so a clean clone does not have these
# structures. The suite has to stay fully runnable without it: skip rather than error.
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the local-only benchmark/corpus")


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
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
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


def test_native_charge_search_is_symmetric(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    charges = []

    def assign(*_args, **kw):
        charges.append(kw["charge"])
        raise ValueError("record the complete ladder")

    monkeypatch.setattr(tmc.rdDetermineBonds, "DetermineBondOrders", assign)

    assert tmc._fast_bond_orders(mol, 0, [0]) is None
    assert set(charges) == {-4, -2, 0, 2, 4}


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


def test_ligand_checks_stream_resonance_candidates(monkeypatch):
    class Supplier:
        def __init__(self, mol):
            self.mol = mol

        def __iter__(self):
            yield self.mol

        def __len__(self):
            raise AssertionError("resonance candidates must not be materialized")

    monkeypatch.setattr(tmc.rdchem, "ResonanceMolSupplier", lambda mol, **_kwargs: Supplier(mol))
    candidates = tmc.lig_checks(Chem.MolFromSmiles("c1ccccc1"), ())

    assert iter(candidates) is candidates
    assert list(candidates)


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


@pytest.mark.parametrize("name", ["Ferrocene", "CisPlatin", "FeCO5", "Cis-PtCl2(en)"])
@needs_corpus
def test_canonical_under_atom_reordering(name, tmp_path):
    path = CORPUS / f"{name}.xyz"
    seen = {Chem.MolToSmiles(_perceive(path))}
    for seed in (1, 2, 3):
        seen.add(Chem.MolToSmiles(_perceive(_shuffled(path, tmp_path, seed))))
    assert len(seen) == 1, f"{name} gave {len(seen)} strings for one molecule: {sorted(seen)}"


@needs_corpus
def test_ligands_close_their_own_valences():
    mol = _perceive(CORPUS / "Ferrocene.xyz")
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
