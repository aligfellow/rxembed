"""`metal/orientation.py`: each donor's class, the walls holding its substituents off the metal and its plane holds."""

from __future__ import annotations

import itertools

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed.metal import orientation
from rxembed.metal.centres import canonical_metal_graph, chemical_graph
from rxembed.metal.sphere import LigandFacts

SP, SP2, SP3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
_DMSO_PALLADIUM = "C[S](C)(=O)->[Pd+2](<-[S-]c1ccccc1)(<-[O]=C(C)C)<-N#CC"
COMPLEXES = {
    "Pd dmso": _DMSO_PALLADIUM,
    "Pd amine acetate": "C[NH2]->[Pd+2]1(<-n2ccccc2)<-[O-]C(C)=[O]->1",
    "Pd tolyl": "C[c]1(->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])ccccc1",
    "Pd2 thiolate": "C[S-]1->[Pd+2](<-[Cl-])(<-[Cl-])<-[S-](C)->[Pd+2]1(<-[Cl-])<-[Cl-]",
    "Pd nitrone": "O=[N+](=CC)->[Pd+2](<-[H-])(<-[Cl-])<-[Cl-]",  # drawn cumulated, O=N+=C
}
# (complex, donor) -> (its class, the substituents walled off the metal, its plane holds)
DONORS = {
    ("Pd dmso", 1): (SP3, {0, 2, 3}, []),  # a sulfoxide S is pyramidal
    ("Pd dmso", 5): (SP3, {6}, []),  # so is an aryl thiolate S
    ("Pd dmso", 12): (SP2, {13}, ["donor_torsion"]),
    ("Pd dmso", 16): (SP, {17}, []),
    ("Pd amine acetate", 1): (SP3, {0, 16, 17}, []),  # its protons are walled too
    ("Pd amine acetate", 3): (SP2, {4, 8}, ["donor_plane"]),
    ("Pd amine acetate", 9): (SP2, set(), ["donor_torsion"]),  # the bite's apex C is no arm
    ("Pd amine acetate", 12): (SP2, set(), ["donor_torsion"]),
    ("Pd tolyl", 1): (SP2, set(), []),  # a filled aromatic C has no axis of its own
    ("Pd2 thiolate", 1): (SP3, set(), []),  # a bridge sets its own axis
    ("Pd nitrone", 1): (SP2, {0, 2}, ["donor_plane"]),  # a bound cumulated N donates as sp2
}


def _facts(smiles):
    """Return the ligand facts an embed reads from `smiles` with explicit H."""
    return LigandFacts(chemical_graph(canonical_metal_graph(Chem.AddHs(rx.parse_smiles(smiles)))))


def _tertiary_amine_palladium():
    """Return Pd bound to an amine N, a PH two carbons along N's first arm with an ether O past it, and an ether O
    two carbons along N's second arm; N's third substituent is a methyl."""
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(46))
    amine, methyl, first, second, phosphine, third, fourth, far_ether, fifth, sixth, near_ether = (
        rw.AddAtom(Chem.Atom(z)) for z in (7, 6, 6, 6, 15, 6, 6, 8, 6, 6, 8)
    )
    chain = (amine, first, second, phosphine, third, fourth, far_ether)
    for a, b in (*itertools.pairwise(chain), (amine, methyl), (amine, fifth), (fifth, sixth), (sixth, near_ether)):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    for donor in (amine, phosphine, far_ether, near_ether):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol, metal, amine, {methyl, fifth}


def test_each_donor_class_is_walled_and_held_in_its_plane():
    """An sp donor is held straight, a free pyridine N on its ring's external bisector; a small conjugated chelate
    ring is held flat at its hinge, one row per donor. A hydride and a bridge donate along no axis of their own.

    A bulky amine's arm is freed only where its nearer donor is small and walled itself: N's arm toward a PH is, its
    arm toward an ether O is not, and the ether past the PH changes nothing.
    """
    facts = {name: _facts(smiles) for name, smiles in COMPLEXES.items()}
    found, windows = {}, {}
    for name, donor in DONORS:
        one = facts[name]
        metal = next(m for d, m in one.donor_bonds if d == donor)
        donors = {d for d, m in one.donor_bonds if m == metal}
        hyb = one.hybridisation
        walls = orientation.orient_donor(one.mol, metal, donor, donors, hyb=hyb, metals=one.bound[donor])
        caps = orientation.coplanar_donor(one.mol, metal, donor, donors, hyb=hyb)
        found[name, donor] = (hyb.get(donor), {key[2] for key in walls}, [row.kind for row in caps])
        windows[name, donor] = set(walls.values())
    hinges = {}
    for name, one in facts.items():
        rows = orientation.ring_hinge(one.mol, one.donor_bonds[0][1], {d for d, _m in one.donor_bonds})
        hinges[name] = [row.atoms for row in rows]

    assert found == DONORS
    ((low, high),) = windows["Pd dmso", 16]
    assert 170.0 <= low < high == 180.0
    ((low, high),) = windows["Pd amine acetate", 3]
    assert (low + high) / 2 == pytest.approx(120.0, abs=2.0)
    assert {name: rows for name, rows in hinges.items() if rows} == {
        "Pd amine acetate": [(2, 9, 12, 10), (2, 12, 9, 10)]
    }
    nitrone = facts["Pd nitrone"]
    assert orientation.stripped_hybridisation(nitrone.mol)[1] == SP  # undeclared, the drawn form's sp stands
    bridged = facts["Pd2 thiolate"]
    donors = {d for d, _m in nitrone.donor_bonds}, {d for d, _m in bridged.donor_bonds}
    axes = (
        orientation.donation_axis(nitrone.mol, 5, donors[0]),
        orientation.donation_axis(bridged.mol, 1, donors[1], metals=2),
    )
    assert axes == (None, None)
    mol, metal, amine, walled = _tertiary_amine_palladium()
    donors = {n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()}
    assert {key[2] for key in orientation.orient_donor(mol, metal, amine, donors)} == walled


def test_embedded_donor_relaxes_inside_its_walls_and_caps():
    """Each wall and cap is a soft window the relax lands inside, give or take the force field's 1 deg."""
    ens = rx.embed(rx.isomers(_DMSO_PALLADIUM, geometry="SPL")[0], n=1, seed=6)
    conformer = ens.mol.GetConformer(ens.ids[0])
    outside = {}
    for row in ens.cons.coplanar:
        phi = abs(rdMolTransforms.GetDihedralDeg(conformer, *map(int, row.atoms)))
        if min(phi, 180.0 - phi) > row.cap + 1.0:
            outside[row.atoms] = (phi, row.cap)
    for (a, b, c), (low, high) in ens.cons.angles.items():
        angle = rdMolTransforms.GetAngleDeg(conformer, a, b, c)
        if a in ens.cons.metals and not low - 1.0 <= angle <= high + 1.0:
            outside[a, b, c] = (angle, (low, high))
    assert outside == {}
