"""`metal/slots.py`: one species per distinct seating, its cis/trans or mer/fac label, and what limits a seating."""

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.metal import slots

# id -> (SMILES, geometry, every listed species' (label, hand), sorted)
ISOMERS = {
    "Co(en)2Cl2": (
        "C1C[NH2]->[Co+3]2(<-[NH2]1)(<-[Cl-])(<-[Cl-])<-[NH2]CC[NH2]->2",
        "OCT",
        [("cis", "delta"), ("cis", "lambda"), ("trans", "")],
    ),
    "Co(NH3)3Cl3": ("[Cl-]->[Co+3](<-[Cl-])(<-[Cl-])(<-N)(<-N)<-N", "OCT", [("fac", ""), ("mer", "")]),
    "Co(dien)Cl3": ("N1CCN2CCN->[Co+3](<-[Cl-])(<-[Cl-])(<-[Cl-])<-1<-2", "OCT", [("fac", ""), ("mer", "")]),
    "UCl2F4 prism": (  # cis on an edge, trans across a square face
        "[Cl-]->[U](<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[Cl-]",
        "TPR",
        [("cis", ""), ("cis", ""), ("trans", "delta"), ("trans", "lambda")],
    ),
    "UCl2F6 antiprism": (  # past six vertices cis and trans name nothing
        "[Cl-]->[U](<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[Cl-]",
        "SQA",
        [("", ""), ("", ""), ("", "delta"), ("", "delta"), ("", "lambda"), ("", "lambda")],
    ),
    "Pd(en)Cl2": ("N1CCN->[Pd+2](<-[Cl-])(<-[Cl-])<-1", "SPL", [("cis", "")]),  # no trans bite
    "Ni imine-amine": (  # a side-on imine face tethered to an amine donor
        "[N]1=[CH](CCC[NH2]->2)->[Ni+2]2(<-[Cl-])(<-[Cl-])<-1",
        "SPL",
        [("cis", ""), ("cis", ""), ("trans", ""), ("trans", "")],
    ),
}


def test_isomers_list_each_seating_once_in_any_atom_order():
    """Each ISOMERS row lists every distinct seating once with its label and hand, drawn or with atoms reversed."""
    listed = {}
    for case, (smiles, geometry, _labels) in ISOMERS.items():
        drawn = rx.parse_smiles(smiles)
        for mol in (drawn, Chem.RenumberAtoms(drawn, list(reversed(range(drawn.GetNumAtoms()))))):
            found = map(rx.identity, rx.isomers(mol, geometry=geometry))
            listed.setdefault(case, []).append(sorted((one.labels[0], one.centres[0].hand) for one in found))

    assert listed == {case: [labels, labels] for case, (_smiles, _geometry, labels) in ISOMERS.items()}


def test_measured_seating_comes_first_and_fix_keeps_its_pins():
    """trans-PtCl2(NH3)2 placed by hand: the trans seating comes first, and pinning Pt and both Cl keeps it alone."""
    mol = rx.parse_smiles("[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[NH3]")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetPositions(np.array([(2.3, 0, 0), (0, 0, 0), (-2.3, 0, 0), (0, 2.05, 0), (0, -2.05, 0)]))
    mol.AddConformer(conformer)

    assert [rx.identity(one).labels for one in rx.isomers(mol, geometry="SPL")] == [("trans",), ("cis",)]
    assert [rx.identity(one).labels for one in rx.isomers(mol, geometry="SPL", fix=[0, 1, 2])] == [("trans",)]
    smiles = rx.parse_smiles("[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[NH3]")
    with pytest.raises(ValueError, match="but not their metal 1, so they state no vertex; add the metal to fix="):
        rx.isomers(smiles, geometry="SPL", fix={0: (2.3, 0, 0), 2: (-2.3, 0, 0)})


def test_too_many_seatings_are_refused_before_listing():
    """Eight distinct donors on a square antiprism have 5,040 seatings, past the 1,000 the limit allows."""
    with pytest.raises(ValueError, match=r"more than 1,000 distinct .* rx\.isomers\(rx\.cxsmiles\(mol\)\)"):
        rx.isomers("[F-]->[U](<-[Cl-])(<-[Br-])(<-[I-])(<-N)(<-O)(<-S)<-P", geometry="SQA")


def test_chelate_link_spanned_by_a_third_donor_is_dropped():
    """In an N-C-N-C-N-C-C-C ring the outer N0...N4 link runs as far as the N0-N2 and N2-N4 links it would repeat."""
    ring = Chem.MolFromSmiles("N1CNCNCCC1")
    assert slots.short_chelate_pairs(ring, [0, 2, 4]) == frozenset({frozenset((0, 1)), frozenset((1, 2))})
