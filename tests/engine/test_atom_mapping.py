"""`engine/atom_mapping.py`: putting a product on its reactant's atom order, and which atoms the reaction moves."""

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.engine.atom_mapping import ReactionAtoms, in_reactant_order, reaction_atoms
from rxembed.engine.coembed import least_motion
from rxembed.identity.reading import read_species
from tests.metal_fixtures import on_its_vertices

FAC_NOTES = (  # fac-Co(NH3)3Cl3 with the slots of ammines N0 and N2 left to fill
    "[NH3]->[Co](<-[NH3])(<-[NH3])(<-[Cl-])(<-[Cl-])<-[Cl-] "
    "|atomProp:0.atomNote.s{}:1.atomNote.OCT:2.atomNote.s{}:3.atomNote.s4:4.atomNote.s1:5.atomNote.s3:6.atomNote.s5|"
)
PROPENE_PT = "[CH2]=[CH]->1C)<-[CH2]=[CH]->2C"  # two side-on propenes on Pt


def _ordered(reactant, product):
    """Return both sides as Mols, a SMILES with explicit H, and the product on the reactant's atom order."""
    sides = [Chem.AddHs(rx.parse_smiles(side)) if isinstance(side, str) else side for side in (reactant, product)]
    return sides[0], in_reactant_order(*sides)


def _bond_changes(reactant, product):
    """Return the bonds `in_reactant_order` breaks and forms, as sets of sorted index pairs."""
    bonds = [
        {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
        for mol in _ordered(reactant, product)
    ]
    return bonds[0] - bonds[1], bonds[1] - bonds[0]


def _reaction_atoms(reactant, product, **selected):
    """Return the `ReactionAtoms` of two sources once they share an atom order and least-motion seating."""
    return reaction_atoms(*least_motion(*map(read_species, _ordered(reactant, product))), **selected)


def _reversed(mol):
    """Return `mol` with its atoms in reverse order."""
    return Chem.RenumberAtoms(Chem.Mol(mol), list(range(mol.GetNumAtoms()))[::-1])


def test_in_reactant_order_keeps_the_most_bonds():
    """A balanced O-H pair may break and re-form, two leaving H may form one H2, and atom maps pin what they label."""
    relay = (  # 2-pyridone to 2-hydroxypyridine through three waters, every atom mapped
        "[O:1]=[c:2]1[c:3]([H:11])[c:4]([H:12])[c:5]([H:13])[c:6]([H:14])[n:7]1[H:15]"
        ".[O:8]([H:16])[H:17].[O:9]([H:18])[H:19].[O:10]([H:20])[H:21]",
        "[O:1]([H:20])[c:2]1[c:3]([H:11])[c:4]([H:12])[c:5]([H:13])[c:6]([H:14])[n:7]1"
        ".[O:8]([H:15])[H:17].[O:9]([H:16])[H:19].[O:10]([H:18])[H:21]",
    )
    rows = {
        "phenoxide protonated": (("CC(=O)O.[O-]c1ccccc1", "CC(=O)[O-].Oc1ccccc1"), {(3, 14)}, {(4, 14)}),
        "isopropanol to H2": (("CC(C)O", "CC(C)=O.[H][H]"), {(1, 7), (3, 11)}, {(7, 11)}),
        "mapped claisen": (("[CH2:1]=C[CH2:2]OC=C", "[CH2:2]=C[CH2:1]CC=O"), {(2, 3)}, {(0, 5)}),
        "proton relay": (relay, {(10, 11), (12, 13), (15, 16), (18, 19)}, {(0, 19), (11, 12), (13, 15), (16, 18)}),
    }
    changes = {name: _bond_changes(*sides) for name, (sides, _broken, _formed) in rows.items()}
    assert changes == {name: (broken, formed) for name, (_sides, broken, formed) in rows.items()}


def test_in_reactant_order_refuses_unpairable_sides():
    for reactant, product, refusal in (
        ("CCCCBr.[Cl-]", "CCCCCl", "the two sides hold different atoms"),
        ("[CH3:1]Cl", "[CH3:2]Cl", "give each atom-map number to one atom on each side"),
        ("[CH3:1]O", "C[OH:1]", "atom map 1 changes element or isotope"),
        ("C=CCOC=C", "C=CCCC=O", r"overlay equally well as different reactions .*; map the reacting atoms"),
    ):
        with pytest.raises(ValueError, match=refusal):
            _ordered(reactant, product)


def test_in_reactant_order_pairs_one_identity_with_its_stereo():
    """One M biaryl axis written from the iodoaryl ring and from the bromopyrrole keeps its hand on renumbering."""
    reactant = Chem.AddHs(Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"))
    product = Chem.AddHs(Chem.MolFromSmiles("Ic1cccc(C)c1-n1c(C)ccc1Br |wU:7.6|"))

    assert rx.identity(in_reactant_order(reactant, product)).stereo == "C7-N8:M"


def test_reaction_atoms_are_the_fewest_sites_that_change():
    """Atoms react where a bond or specified arrangement changes. Equivalent sites of one species keep their places in
    any atom order. A metal moves the fewest sites within its polyhedron, and across a change of polyhedron all of them
    only when it keeps them all. A configuration change turns its smallest separable group.
    """
    propenes = rx.isomers(f"[Cl-]->[Pt+2]12(<-[Cl-])(<-{PROPENE_PT}", geometry="SPL")
    meso = next(mol for mol in propenes if rx.identity(mol).haptic == ("meso",))
    fac = on_its_vertices(
        rx.isomers("N->[Co+3](<-N)(<-N)(<-[Cl-])(<-[Cl-])<-[Cl-]", geometry="OCT").select(label="fac")
    )
    palladium = rx.isomers("N->[Pd+2](<-N)(<-[Cl-])<-[Cl-]", geometry="SPL")  # N0 Pd1 N2 Cl3 Cl4
    ammines = "[NH3:1]->[Co+3:2](<-[NH3:3])(<-[NH3:4])(<-[NH3:5])"
    iron = "[O+]#[C-]->[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[PH3]"  # C1 Fe2 C3 C5 C7 P9
    rows = {
        "chloride exchange": ("[Cl-].CCl", "ClC.[Cl-]", ()),
        "K(NH3)6 unchanged": ("N->[K+](<-N)(<-N)(<-N)(<-N)<-N", "N->[K+](<-N)(<-N)(<-N)(<-N)<-N", ()),
        "fac Co ammines swapped": (FAC_NOTES.format(0, 2), FAC_NOTES.format(2, 0), ()),
        "meso Pt reversed": (meso, _reversed(meso), ()),
        "fac Co reversed": (fac, _reversed(fac), ()),
        "cis to trans Pd": (palladium.select(label="cis"), palladium.select(label="trans"), (1, 2, 3)),
        "Co loses chloride": (
            rx.isomers(f"{ammines}(<-[NH3:6])<-[Cl-:7]", geometry="OCT")[0],
            rx.isomers(f"{ammines}<-[NH3:6].[Cl-:7]", geometry="SPY")[0],
            (1, 6),
        ),
        "Fe Berry": (rx.isomers(iron, geometry="TBP")[0], rx.isomers(iron, geometry="SPY")[0], (1, 2, 3, 5, 7, 9)),
        "Pt ammine for chloride, reversed": (
            rx.isomers(f"[Cl-]->[Pt+2]12(<-[Cl-])(<-{PROPENE_PT}.N", geometry="SPL")[1],
            _reversed(rx.isomers(f"N->[Pt+2]12(<-[Cl-])(<-{PROPENE_PT}.[Cl-]", geometry="SPL")[1]),
            (0, 1, 9),
        ),
        "CFClBrI inverted": ("[F:1][C@:2]([Cl:3])([Br:4])[I:5]", "[F:1][C@@:2]([Cl:3])([Br:4])[I:5]", (0, 1)),
        "E to Z butene": ("C/C=C/C", "C/C=C\\C", (0, 1, 2, 4, 5, 6, 7)),
    }
    reacting = {name: _reaction_atoms(reactant, product).reacting for name, (reactant, product, _atoms) in rows.items()}
    assert reacting == {name: atoms for name, (_reactant, _product, atoms) in rows.items()}


def test_reaction_atoms_align_the_selected_spectators():
    """Bromide displaced from bromoethane by chloride: C1, Br2 and Cl3 react, and with no selection every unchanged
    atom, hydrogens included, is a spectator."""
    selections = ({}, {"spectators": ["[CH3]"]}, {"reacting": [0]})
    assert [_reaction_atoms("CCBr.[Cl-]", "CCCl.[Br-]", **selected) for selected in selections] == [
        ReactionAtoms(spectators=(0, 4, 5, 6, 7, 8), reacting=(1, 2, 3)),
        ReactionAtoms(spectators=(0,), reacting=(1, 2, 3)),
        ReactionAtoms(spectators=(4, 5, 6, 7, 8), reacting=(0, 1, 2, 3)),
    ]


def test_reaction_atoms_refuse_what_they_cannot_align():
    for reactant, product, selected, refusal in (
        ("CCBr.[Cl-]", "CCCl.[Br-]", {"spectators": []}, "spectator_atoms must contain at least one atom"),
        ("CCBr.[Cl-]", "CCCl.[Br-]", {"spectators": [0, 1], "reacting": [1]}, "react_atoms must be outside"),
        ("CCBr.[Cl-]", "CCCl.[Br-]", {"spectators": {0: 0}}, "pair atoms with atom maps on both sides"),
        ("CCBr.[Cl-]", "CCCl.[Br-]", {"reacting": "[CH3]"}, "react_atoms takes reactant atoms"),
        ("[H][H].ClCl", "[H]Cl.[H]Cl", {}, "the reaction changes every atom; pass spectator_atoms="),
    ):
        with pytest.raises((TypeError, ValueError), match=refusal):
            _reaction_atoms(reactant, product, **selected)
