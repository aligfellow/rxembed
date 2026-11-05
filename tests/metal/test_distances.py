"""`metal/distances.py`: how long each metal-donor bond is held, and how close a non-donor or second metal may come."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.metal import distances
from rxembed.metal.centres import canonical_metal_graph, chemical_graph
from rxembed.metal.sphere import LigandFacts
from tests.metal_fixtures import pd2cl6

_PT = Chem.GetPeriodicTable()

# id -> (complex, {donor atom: typical crystal-structure M-L length window, A}): one row per departure from the fit
LENGTHS = {
    "PdHCl(PMe3)(MeCN)": (  # hydride, halide, soft P and terminal sp N
        "[H-]->[Pd+2](<-[Cl-])(<-[P](C)(C)C)<-N#CC",
        {0: (1.45, 1.70), 2: (2.20, 2.45), 3: (2.20, 2.40), 7: (1.90, 2.10)},
    ),
    "VOCl4": ("[O-2]->[V+4](<-[Cl-])(<-[Cl-])(<-[Cl-])<-[Cl-]", {0: (1.50, 1.70)}),  # an oxo
    "Fe(CO)5": ("[O+]#[C-]->[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", {1: (1.70, 1.90)}),
    "FeCl2(BH4)": ("[BH3-][H]->[Fe+2](<-[Cl-])<-[Cl-]", {1: (1.55, 1.85)}),  # a B-H...Fe bridge
}


def test_delocalised_charges_share_a_charge_over_alike_donors():
    """A charge is averaged over donors alike once bond orders, charges and aromaticity are cleared: a carboxylate's O,
    and a dipyrrin's two N though one is drawn in an aromatic pyrrolide and the other in a non-aromatic ring."""
    rows = {"acetate": ("CC(=O)[O-]->[Zn+2]", "O"), "dipyrrin": ("C1=CC(=Cc2ccc[n-]2->[Zn+2])N=C1", "N")}
    shared = {}
    for name, (smiles, element) in rows.items():
        mol = Chem.AddHs(rx.parse_smiles(smiles))
        charges = distances.delocalised_charges(mol)
        shared[name] = sorted(charges[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == element)
    assert shared == {"acetate": [-0.5, -0.5], "dipyrrin": [-0.5, -0.5]}


def test_ml_distance_fits_crystal_lengths_or_the_covalent_sum():
    """Each donor class lands inside its crystal length window, and an unfitted metal at the covalent radius sum."""
    found = {}
    for name, (smiles, _windows) in {**LENGTHS, "UF2": ("[F-]->[U+2]<-[F-]", {})}.items():
        facts = LigandFacts(chemical_graph(canonical_metal_graph(rx.parse_smiles(smiles))))
        for donor, metal in facts.donor_bonds:
            z = facts.graph.GetAtomWithIdx(metal).GetAtomicNum()
            found[name, donor] = distances.ml_distance(
                facts.mol, donor, z, facts.charges, hyb=facts.hybridisation, eta=0
            )

    windows = {(name, donor): window for name, (_smiles, rows) in LENGTHS.items() for donor, window in rows.items()}
    expected = {key: pytest.approx((lo + hi) / 2, abs=(hi - lo) / 2) for key, (lo, hi) in windows.items()}
    expected["UF2", 0] = pytest.approx(_PT.GetRcovalent(92) + _PT.GetRcovalent(9))
    assert {key: found[key] for key in expected} == expected


def test_ff_terms_pull_lone_donors_and_floor_non_donors():
    """Pd0 holds an N1-C2-N3 chelate, Cl4, Br5 and a hydride H7; C6 hangs off N1.

    The chelate's two N choose their own lengths unless grafted, and the hydride keeps its pull. C2 bonds both N, a bite
    apex only the DG relief floors; C6 bonds one donor and is floored in the force field too. Coordinates change none.
    """
    rw = Chem.RWMol()
    pd, n1, c2, n3, cl, br, c6, h7 = (rw.AddAtom(Chem.Atom(z)) for z in (46, 7, 6, 7, 17, 35, 6, 1))
    for a, b in ((n1, c2), (c2, n3), (n1, c6)):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    donors = [n1, n3, cl, br, h7]
    windows = rx.Constraints(distances={(pd, donor): (2.1, 2.3) for donor in donors})

    held = distances.ff_terms(mol, windows, pd, 46, donors)
    grafted = distances.ff_terms(mol, windows, pd, 46, donors, frozen={n1})
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetPositions(np.arange(3.0 * mol.GetNumAtoms()).reshape(-1, 3))
    mol.AddConformer(conformer)

    assert held.metals == {pd}
    assert set(held.pulls) == {(pd, cl), (pd, br), (pd, h7)}
    assert held.pulls[(pd, cl)] == pytest.approx((2.2, 2.2))
    assert set(grafted.pulls) == {(pd, donor) for donor in donors}
    assert (set(held.floors), set(held.dg_floors)) == ({(pd, c6)}, {(pd, c2), (pd, c6)})
    assert distances.ff_terms(mol, windows, pd, 46, donors) == held


def test_metals_sharing_a_donor_are_floored_apart():
    """Only Pd0 and Pt1, which share Cl3, are floored; the embedded Pd2Cl6 keeps its unbonded Pd pair apart."""
    floors = distances.bridge_floors([(3, 0), (3, 1), (4, 0), (5, 2)], {0: 46, 1: 78, 2: 46})
    assert set(floors.floors) == set(floors.dg_floors) == {(0, 1)}

    ens = rx.embed(rx.isomers(pd2cl6(), center=1, stereo="free")[0], n=1, seed=2, threads=1)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    assert np.linalg.norm(pos[1] - pos[4]) > 2.8  # a Pd-Pd bond would be about 2.6 A
    legs = [(pos[1] - pos[bridge], pos[4] - pos[bridge]) for bridge in (3, 7)]
    angles = [np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))) for a, b in legs]
    assert angles == [pytest.approx(90.0, abs=15.0)] * 2  # an M(mu-Cl)2M rhombus, neither folded shut nor torn open


def test_alkali_ion_embeds_at_ionic_contact_with_each_partner():
    """A chelating acetate, and waters and a chloride drawn as arrows from the sodium, all rest at contact; each lone
    partner is pulled to its window's middle and the chelate's linked arms are not."""
    found, expected = {}, {}
    for smiles, pulls in (("CC1=[O]->[Na+]<-[O-]1", 0), ("O->[Na+](<-O)(<-O)->[Cl]", 4)):
        found[smiles, "pulls"] = len(distances.ionic_contacts(Chem.AddHs(rx.parse_smiles(smiles))).pulls)
        expected[smiles, "pulls"] = pulls
        ens = rx.embed(smiles, n=1, seed=1)
        sodium = next(atom for atom in ens.mol.GetAtoms() if atom.GetAtomicNum() == 11)
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        for partner in sodium.GetNeighbors():
            contact = _PT.GetRcovalent(11) + _PT.GetRcovalent(partner.GetAtomicNum())
            found[smiles, partner.GetIdx()] = np.linalg.norm(pos[sodium.GetIdx()] - pos[partner.GetIdx()])
            expected[smiles, partner.GetIdx()] = pytest.approx(contact - 0.05, abs=0.2)  # 0.25 short to 0.15 long

    assert found == expected
