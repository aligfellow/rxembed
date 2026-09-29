"""Test donor-to-polyhedron seating and arrangement labels."""

from rdkit import Chem

import rxembed as rx
from rxembed import metal_slots as slots


def test_observed_orbit_precedes_canonical_enumeration():
    mol = Chem.MolFromSmiles("N.[P].[O-].[Cl-]")
    donors = list(range(4))
    retained = [2, 1, 0, 3]

    orderings = slots.distinct_vertex_orderings(slots.SeatingProblem(mol, donors, "square_planar"), retained=retained)

    assert orderings[0] == tuple(retained)


def test_chelate_edge_links_through_donor_exclusion_drops_the_redundant_outer_link():
    """When a third donor sits astride the a-b backbone, a-b is dropped; the shorter a-c and c-b hold.

    An 8-membered N-C-N-C-N-C-C-C ring gives donors a/c/b at ring positions 0/2/4: a-c and c-b are each a
    2-bond, 4-membered chelate arm, and a-b's own donor-free backbone (the other way around the ring) is
    also 4 bonds, exactly 2 + 2, so a-b is redundant with a-c plus c-b and is dropped.
    """
    rw = Chem.RWMol()
    atoms = [rw.AddAtom(Chem.Atom(7 if i in (0, 2, 4) else 6)) for i in range(8)]
    for i in range(8):
        rw.AddBond(atoms[i], atoms[(i + 1) % 8], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    padded = [atoms[0], atoms[2], atoms[4]]

    with_exclusion = slots.chelate_edge_links(mol, padded)
    assert with_exclusion == frozenset({frozenset((0, 1)), frozenset((1, 2))})


def test_haptic_tether_dedup_is_atom_order_invariant():
    mol = rx.parse_smiles("[N]1=[CH](CCC[NH2]->2)->[Ni+2]2(<-[Cl-])(<-[Cl-])<-1")
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    def labels(graph):
        return {iso.label for iso in rx.metal(graph, "square_planar", stereo="free")}

    assert labels(mol) == labels(renumbered) == {"cis", "trans"}
