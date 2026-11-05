"""`identity/smiles_io.py`: parse and write dative SMILES and CX SMILES, and the notes that seat each metal."""

import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.graph import metal_indices
from rxembed.identity.reading import read_species
from rxembed.identity.smiles_io import NATIVE_TAGS, mol_from_species, write_dative
from rxembed.metal.polyhedra import vertex_dirs
from rxembed.stereo import LigandStereo, StereoElement, defined_stereo_label
from tests.geometry_fixtures import embedded, placed
from tests.metal_fixtures import (
    BUTADIENE_FE_CO3,
    FERROCENE_TWO_FACES,
    PD2CL6,
    PT2CL4_AMMINES,
    bridged_dimer,
    enumerated,
    pd2cl6,
    zinc_copies,
)

ATROP = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"  # an N-aryl pyrrole whose marker specifies the M axis
BROMO_PD = "[NH3]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Br-]"
NOTED_BROMO_PD = BROMO_PD + " |atomProp:0.atomNote.s0:1.atomNote.SPL:2.atomNote.s1:3.atomNote.s2:4.atomNote.s3|"
NOTED_ZINC = (  # each donor different, so the seating has a hand
    "[NH3]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-] |atomProp:0.atomNote.s0:1.atomNote.TET-delta:2.atomNote.s1:3.atomNote.s2:"
    "4.atomNote.s3|"
)
NOTED_PROPENE_PT = (  # the propene's prochiral face wound +
    "C[CH]1=[CH2]->[Pt+2]<-1(<-[NH3])(<-[Cl-])<-[Br-] |atomProp:1.atomNote.s0+:2.atomNote.s0+:3.atomNote.SPL:"
    "4.atomNote.s1:5.atomNote.s3:6.atomNote.s2|"
)
NOTED_PD_PT = (  # two square planes, Pd1 and Pt6
    "[NH3]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Br-].[NH3]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-] |atomProp:0.atomNote.s0:"
    "1.atomNote.SPL:2.atomNote.s1:3.atomNote.s2:4.atomNote.s3:5.atomNote.s0:6.atomNote.SPL:7.atomNote.s1:"
    "8.atomNote.s2:9.atomNote.s3|"
)
NOTED_BRIDGED_PD_PT = (  # each bridging chloride carries one slot per metal
    "[Cl-]->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]->1 |atomProp:0.atomNote.s0:1.atomNote.SPL:"
    "2.atomNote.s1:3.atomNote.s2;s0:4.atomNote.SPL:5.atomNote.s2:6.atomNote.s3:7.atomNote.s3;s1|"
)
NOTED_PYRAZINE_CO2 = (  # two graph-equivalent cobalts, one chiral; "OCT-delta" sorts before "OCT", so it is first
    "[NH3]->[Co+3](<-[NH3])(<-[Cl-])(<-[Cl-])(<-[n]1ccccc1)<-[n]1cc[n](->[Co+3](<-[NH3])(<-[NH3])(<-[Cl-])"
    "(<-[Cl-])<-[n]2ccccc2)cc1 |atomProp:0.atomNote.s0:1.atomNote.OCT-delta:2.atomNote.s2:3.atomNote.s1:"
    "4.atomNote.s4:5.atomNote.s3:11.atomNote.s5:14.atomNote.s0:15.atomNote.OCT:16.atomNote.s1:17.atomNote.s2:"
    "18.atomNote.s4:19.atomNote.s5:20.atomNote.s3|"
)
NOTED_ZEISE_LIKE = (  # ethylene, whose two faces are mirror images
    "[NH3]->[Pt+2]1(<-[Cl-])(<-[Br-])<-[CH2]=[CH2]->1 |atomProp:0.atomNote.s0:1.atomNote.SPL:2.atomNote.s2:"
    "3.atomNote.s1:4.atomNote.s3:5.atomNote.s3|"
)
PT_AMMINES = "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]"
CARBANION_NI = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"  # carbanion C23
QUINOID_YLIDE = "[Cl-]->[Au+]<-[CH2-][S+2](c1ccccc1)=C1C=C[CH-]C=C1"  # only its Lewis form splits the phenyls at S
PD2CL6_XY = ((-3.35, 1.6), (-1.75, 0.0), (-3.35, -1.6), (0.0, 1.65), (1.75, 0.0), (3.35, 1.6), (3.35, -1.6), (0, -1.65))
UP, DOWN = (3.35, 1.6, 0.0), (3.35, -1.6, 0.0)


def _renumbered(mol, *orders):
    """Return `mol` in each atom order of `orders`, the given order first."""
    return [mol, *(Chem.RenumberAtoms(mol, list(order)) for order in orders)]


def _reversed(mol):
    """Return `mol` in the given order and with its atoms reversed."""
    return _renumbered(mol, reversed(range(mol.GetNumAtoms())))


def _planar(smiles, xy):
    """Return `smiles` placed in the z = 0 plane at `xy`, one point per atom."""
    return placed(rx.parse_smiles(smiles), [(x, y, 0.0) for x, y in xy])


def _pyrazolate_dimers():
    """Return the Cs and C2 [Pd(mu-pz)Cl(NH3)]2: no donor atom is shared, so the bridges' relation is the ligands'."""
    smiles = "[Cl-]->[Pd+2]1(<-[NH3])<-[n-]2[cH][cH][cH][n]2->[Pd+2](<-[Cl-])(<-[NH3])<-[n-]3[cH][cH][cH][n]3->1"
    ring = {1: (-1.75, 0.0), 8: (1.75, 0.0), 0: (-2.99, 1.95), 2: (-2.99, -1.95)}
    ring |= {3: (-0.68, 1.68), 4: (-1.1, 2.95), 5: (0.0, 3.75), 6: (1.1, 2.95), 7: (0.68, 1.68)}
    ring |= {15: (-0.68, -1.68), 14: (-1.1, -2.95), 13: (0.0, -3.75), 12: (1.1, -2.95), 11: (0.68, -1.68)}
    return [
        _planar(smiles, [{**ring, 9: chloride, 10: ammine}[atom] for atom in range(16)])
        for chloride, ammine in (((2.99, 1.95), (2.99, -1.95)), ((2.99, -1.95), (2.99, 1.95)))
    ]


def test_parse_smiles_reads_each_drawing_as_its_graph():
    """Hydrogens drop unless kept, mapped or addressed by CX notes; a covalent metal bond is read as dative."""
    for smiles, choices, written in (
        ("[H]OC", {}, "CO"),
        ("[H]OC", {"remove_hs": False}, "[H]OC"),
        ("[H]OC |atomProp:0.atomNote.x|", {}, "[H]OC"),
        ("[NH3:1]->[Cu+:2]<-[NH2:3][H:4]", {}, "[NH3:1]->[Cu+:2]<-[NH2:3][H:4]"),
        ("[Zn](n1ccccc1)n1ccccc1", {}, "c1cc[n](->[Zn]<-[n]2ccccc2)cc1"),
    ):
        assert Chem.MolToSmiles(rx.parse_smiles(smiles, **choices)) == written, smiles


def test_parse_smiles_keeps_an_eta2_ez_drawn_through_its_metal():
    """A side-on alkene's E/Z referenced through its metal moves onto ligand atoms with the code the drawing gives."""
    drawn = {
        r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1": "C1=C2:Z",
        r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1": "C1=C2:E",
    }
    read = {smiles: str(defined_stereo_label(mol := rx.parse_smiles(smiles), metal_indices(mol))) for smiles in drawn}

    assert read == drawn


def test_parse_smiles_refuses_a_string_it_cannot_read_as_written():
    for smiles, message in (
        ("C1CC", "could not parse SMILES"),
        ("Cl[Pt@SP](Cl)(N)N", "square_planar tag on 4 neighbours has no permutation"),
        ("Cl[Pt@SP1](Cl)(N)(N)N", "square_planar tag on 5 neighbours has no permutation"),
        ("CC=CC |c:x|", "invalid CX c:/t: double-bond stereo field"),
        ("CC=CC |c:|", "invalid empty CX c:/t: double-bond stereo field"),
        ("C/C=C/C |c:1,t:1|", "contradictory or duplicate CX c:/t: stereo for bond 1"),
        ("CC=CC |c:9|", "CX c: field names missing bond 9"),
        ("CC=CC |c:0|", "CX c:0 did not define valid double-bond stereo"),
        ("CC=CC |atomProp:0._rxEZ0.E|", "CX E/Z record 0 must name two atoms"),
        ("N(->[Pd+]<-[Cl-])->[Pt+]<-[Br-] |atomProp:0.atomNote.s0|", r"has 1 slot note\(s\) for 2 adjacent metal"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.parse_smiles(smiles)


def _bridging_hydride(edges):
    """Return [BH3]-H-Fe+ with its two H bonds added in `edges` order, both covalent."""
    rw = Chem.RWMol()
    for number, charge, hydrogens in ((5, -1, 3), (1, 0, 0), (26, 1, 0)):
        atom = Chem.Atom(number)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(hydrogens)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for begin, end in edges:
        rw.AddBond(begin, end, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def _enolate_both_ways():
    """Return a prop-1-en-1-olate geometry with its C=C dihedral at 0 and at 180 degrees."""
    forms = []
    for angle in (0.0, 180.0):
        mol = embedded("CC=C[O-]")
        rdMolTransforms.SetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3, angle)
        forms.append(mol)
    return forms


def test_smiles_writes_one_dative_string_for_every_drawing():
    """A Lewis form, atom order or forged tag changes no string; an O~N contact is written but makes C1 no centre;
    a mapped or E/Z-bearing hydrogen stays."""
    forged = rx.parse_smiles("C[N-](C)->[Pd+3](<-[Cl-])<-[Cl-]")
    forged.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    carbanion = rx.parse_smiles(CARBANION_NI.replace("[CH-]", "[C@@H-]"))
    forged_beside_a_hand = Chem.Mol(carbanion)
    forged_beside_a_hand.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    imine = rx.parse_smiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]")  # its E drawn against the platinum
    imine.GetBondBetweenAtoms(1, 2).SetStereoAtoms(0, 3)
    imine.GetBondBetweenAtoms(1, 2).SetStereo(Chem.BondStereo.STEREOE)
    side_on = rx.isomers(r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", geometry="SPL")[0]
    hydrogen_bonded = rx.parse_smiles("[N]1(->[Ni]2)/O[H]~O=[N+]->2=C/C=1 |Z:3|", remove_hs=False)
    for drawings, written in (
        (("[NH3]->[Pt](<-[NH3])(Cl)Cl", PT_AMMINES), PT_AMMINES),
        (("CP(C)(C)->[Rh]Cl", "CP(C)(C)->[Rh+]<-[Cl-]"), "C[P](C)(C)->[Rh+]<-[Cl-]"),
        (("CN(C)[Pd](Cl)Cl", "C[N-](C)->[Pd+3](<-[Cl-])<-[Cl-]", forged), "C[N-](C)->[Pd+3](<-[Cl-])<-[Cl-]"),
        (("C[Pd](Cl)(Cl)C", "[CH3-]->[Pd+4](<-[Cl-])(<-[Cl-])<-[CH3-]"), "[CH3-]->[Pd+4](<-[CH3-])(<-[Cl-])<-[Cl-]"),
        ([_bridging_hydride(e) for e in (((0, 1), (1, 2)), ((1, 2), (0, 1)))], "[BH3-][H]->[Fe+]"),
        ([rx.parse_smiles("[BH3-][H+][BH3-]", remove_hs=False)], "[BH3-][H+][BH3-]"),
        (_reversed(hydrogen_bonded), "[H]1O[N]2=CC=[N+](=O~1)->[Ni]<-2 |Z:6|"),
        (_reversed(rx.parse_smiles("C[C@@H](CO)CO~N |Z:5|")), "CC(CO)CO~N |Z:5|"),
        (_enolate_both_ways(), "CC=C[O-]"),
        (["[NH3:1]->[Cu+:2]<-[NH2:3][H:4]"], "[NH3:1]->[Cu+:2]<-[NH2:3][H:4]"),
        ([imine], r"[H]/[N](=C\C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]"),
        ([side_on], "C[CH]1=[CH](F)->[Pt+2]<-1(<-[NH3])(<-[Cl-])<-[Br-]"),  # E/Z through the metal is CX's
        ([carbanion, forged_beside_a_hand], rx.smiles(carbanion)),
        ([embedded(QUINOID_YLIDE)], "[Cl-]->[Au+]<-[CH2-][S+2](c1ccccc1)=c1cc[cH-]cc1"),
    ):
        mols = [rx.parse_smiles(one) if isinstance(one, str) else one for one in drawings]
        assert {rx.smiles(mol) for mol in mols} == {written}, written


def test_smiles_refuses_what_a_plain_dative_smiles_cannot_hold():
    unreadable = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl")))
    carbon = unreadable.AddAtom(Chem.Atom(6))
    for _ in range(5):
        unreadable.AddBond(carbon, unreadable.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
    for mol, message in (
        (rx.parse_smiles(ATROP), r"cannot retain atropisomer stereo; use cxsmiles\(\)"),
        (unreadable.GetMol(), r"could not write a round-tripping SMILES .* use the Mol itself"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.smiles(mol)


def test_each_species_cx_name_reads_back_as_itself():
    """Faces, E/Z through a shared metal, axes and bridges that plain SMILES drops each tell two names apart."""
    for name, species in (
        (
            "eta2-alkene-faces-and-ez",
            rx.isomers(r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", geometry="SPL")[:2],
        ),
        ("eta2 E/Z by chiral C", rx.isomers(r"F[C@H](Cl)/[CH]1=[CH](/Br)->[Pt+2](<-[Cl-])(<-[I-])(<-[NH3])<-1")[:1]),
        ("carbanion-donor-hands", rx.isomers(CARBANION_NI, geometry="SPL")),
        ("atropisomer-axis", rx.isomers(ATROP, stereo="racemic")),
        ("rac-and-meso-ferrocenes", rx.isomers(FERROCENE_TWO_FACES, stereo="racemic")),
        ("s-cis-and-s-trans-butadiene", rx.isomers(BUTADIENE_FE_CO3)),
        ("delta-and-lambda-zinc", rx.isomers("[F-]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-]", geometry="TET")),
        ("Cl bridges cis or trans", rx.isomers(PD2CL6, screen=False)),
        ("syn and anti Pt dimers", [bridged_dimer(PT2CL4_AMMINES, pair) for pair in ((UP, DOWN), (DOWN, UP))]),
        ("cs-and-c2-pyrazolate-dimers", _pyrazolate_dimers()),
        (
            "nitrogen-bridging-pd-and-pt",
            [_planar("N(->[Pd+]<-[Cl-])->[Pt+]<-[Br-]", [(0, 0), (-2, 0), (-4, 0), (2, 0), (4, 0)])],
        ),
        ("pyrazine-bridged-cobalts", [rx.parse_smiles(NOTED_PYRAZINE_CO2)]),
    ):
        names = [rx.cxsmiles(one) for one in species]
        assert len(set(names)) == len(names), name
        assert [rx.cxsmiles(rx.parse_smiles(text)) for text in names] == names, name


def test_species_mol_reads_back_as_its_record_in_atom_order():
    """Notes are written in the record's own vertex frame, so its constraints are the record's."""
    for smiles, geometry in (
        ("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "SPL"),
        ("CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC", "OCT"),
        ("C[C@@H]1C[NH2]->[Co+3]2(<-[NH2]1)(<-[Cl-])(<-[Cl-])<-[NH2]CC[NH2]->2", "OCT"),
        ("[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH2]->1", "SPL"),
    ):
        for record in enumerated(Chem.AddHs(rx.parse_smiles(smiles)), geometry):
            written = mol_from_species(record)
            elements = [atom.GetAtomicNum() for atom in written.GetAtoms()]
            assert elements == [atom.GetAtomicNum() for atom in record.mol.GetAtoms()], smiles
            assert read_species(written).centres == record.centres, smiles


def test_cxsmiles_names_a_species_alike_in_any_order_or_drawing():
    """Two equivalent metals, three zinc copies, bridge slots bound to their bonds and two identical faces."""
    anti = _planar(
        "[Cl-]->[Pd+2]1(<-[NH3])<-[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[n]2[cH][cH][cH][n-]2->1",
        [
            *PD2CL6_XY[:5],
            (3.35, -1.6),
            (3.35, 1.6),
            (0.68, -1.68),
            (1.1, -2.95),
            (0, -3.75),
            (-1.1, -2.95),
            (-0.68, -1.68),
        ],
    )
    shuffles = ([7, 6, 5, 4, 3, 2, 1, 0], [4, 5, 6, 7, 0, 1, 2, 3], [3, 1, 7, 0, 6, 4, 2, 5])
    bridged = rx.parse_smiles(NOTED_BRIDGED_PD_PT, remove_hs=False)
    phosphate = Chem.MolFromSmiles("F[P@OH1-](F)(F)(F)(F)F.[NH3]->[Pt+2](<-[NH3])(<-[NH3])<-[NH2][C@@H](C)CC")
    plus_minus = rx.cxsmiles(rx.isomers(FERROCENE_TWO_FACES, stereo="racemic")[1])
    for name, forms in (
        ("planar-pd2cl6", _renumbered(pd2cl6(), *shuffles)),
        ("SPL and TET Pd2Cl6", _renumbered(bridged_dimer(PD2CL6, ((3.05, 0, 1.8), (3.05, 0, -1.8))), *shuffles)),
        ("syn-ammine-platinum-dimer", _renumbered(bridged_dimer(PT2CL4_AMMINES, (UP, DOWN)), *shuffles)),
        ("anti pyrazolate dimer", _renumbered(anti, reversed(range(12)), [7, 11, 4, 1, 0, 2, 3, 5, 6, 8, 9, 10])),
        (
            "three-zinc-copies",
            _renumbered(zinc_copies((1, 1, 1)), [*range(5, 15), *range(5)], [*range(10, 15), *range(10)]),
        ),
        ("bridge-slots-renumbered", [NOTED_BRIDGED_PD_PT, *_reversed(bridged)[1:]]),
        (
            "OCT phosphate tag unread",
            _renumbered(phosphate, [int(i) for i in np.random.default_rng(5).permutation(phosphate.GetNumAtoms())]),
        ),
        ("faces wound oppositely", [plus_minus, re.sub(r"s([01])([+-])", _flipped, plus_minus)]),
    ):
        mols = [rx.parse_smiles(one) if isinstance(one, str) else one for one in forms]
        assert len({rx.cxsmiles(mol) for mol in mols}) == 1, name


def _flipped(note):
    """Return a face's slot note `note` (a regex match of slot and sign) wound the other way."""
    return f"s{note[1]}{'-' if note[2] == '+' else '+'}"


def test_equivalent_zinc_centres_of_unlike_hands_have_no_name():
    """A pair swap orders two equivalent metals; nothing orders three once one is the mirror image."""
    for verb in (rx.cxsmiles, rx.identity):
        with pytest.raises(ValueError, match="three or more symmetry-equivalent metal centres carry different"):
            verb(zinc_copies((1, 1, -1)))


def test_cxsmiles_keeps_a_caller_note_the_name_leaves_out():
    """rx.cxsmiles carries an annotation through a reaction mapping; the species is named without it."""
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    mol.GetAtomWithIdx(0).SetProp("atomNote", "selected")

    assert "atomNote.selected" in rx.cxsmiles(mol)
    assert rx.identity(mol).name == rx.cxsmiles(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))


def test_contradictory_notes_are_refused():
    """A planar or achiral seating has no hand, a chiral one must name its own, and only a prochiral face winds."""
    for text, message in (
        (NOTED_BROMO_PD.replace("SPL", "SPL-delta"), "is planar and cannot carry metal chirality 'delta'"),
        (NOTED_BROMO_PD.replace("SPL", "SPL-x"), "unknown metal chirality 'x'"),
        (NOTED_BROMO_PD.replace("SPL", "XYZ"), "names 'XYZ', not a polyhedron rxembed has"),
        (NOTED_BROMO_PD.replace("4.atomNote.s3", "4.atomNote.s9"), "vertex 9 is not one of this geometry's 4"),
        (NOTED_BROMO_PD.replace("4.atomNote.s3", "4.atomNote.s2"), r"donor\(s\) \[\d\] were given no vertex"),
        (NOTED_ZINC.replace("TET-delta", "TET"), r"seating is chiral \(delta\) but the metal note omits chirality"),
        (NOTED_ZINC.replace("TET-delta", "TET-lambda"), "the metal note says lambda, but its seating is delta"),
        (NOTED_ZINC.replace("[Br-]", "[Cl-]"), "the metal note says delta, but its seating is achiral"),
        (NOTED_PROPENE_PT.replace("2.atomNote.s0+", "2.atomNote.s0-"), "the haptic face at slot s0 carries both"),
        (NOTED_PROPENE_PT.replace("4.atomNote.s1", "4.atomNote.s1+"), "slot s1\\+ gives a face winding, but that slot"),
        (NOTED_PROPENE_PT.replace("s0+", "s0P"), "gives a winding this face lacks; it has only"),
        (NOTED_ZEISE_LIKE.replace("s3:5.atomNote.s3", "s3+:5.atomNote.s3+"), "none, being a mirror-symmetric face"),
        (NOTED_PD_PT.partition(":5.atomNote")[0] + "|", r"arrange metal\(s\) \[1\] but not metal\(s\) \[6\]"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.isomers(text)


# a complex per tag class, its ligands all different; RDKit's TB angles need covalent bonds
_TAGGED = {
    Chem.ChiralType.CHI_SQUAREPLANAR: "Cl[Pt@SP{}](Br)(N)I",
    Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL: "Cl[Fe@TB{}](Br)(I)(N)P",
    Chem.ChiralType.CHI_OCTAHEDRAL: "O[Co@OH{}](Cl)(C)(N)(F)P",
}


def test_native_tags_read_back_from_their_vertices():
    """Neighbours placed in storage order on the vertices of a NATIVE_TAGS row read back as its tag and number."""
    read = {}
    for tag, (geometry, rows) in NATIVE_TAGS.items():
        dirs = np.asarray(vertex_dirs(geometry), float)
        dirs = 2.0 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        for number, vertices in rows.items():
            sphere = Chem.MolFromSmiles(_TAGGED[tag].format(number))
            metal = next(atom for atom in sphere.GetAtoms() if atom.GetChiralTag() == tag)
            conf = Chem.Conformer(sphere.GetNumAtoms())
            for bond, vertex in zip(metal.GetBonds(), vertices, strict=True):
                conf.SetAtomPosition(bond.GetOtherAtomIdx(metal.GetIdx()), Point3D(*dirs[vertex]))
            sphere.AddConformer(conf)
            metal.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            metal.ClearProp("_chiralPermutation")
            Chem.AssignStereochemistryFrom3D(sphere, replaceExistingTags=True)
            read[tag, number] = (metal.GetChiralTag(), metal.GetUnsignedProp("_chiralPermutation"))

    assert read == {key: key for key in read}


def test_native_metal_tag_specifies_the_species_its_notes_would():
    """A tag counts its bonds, then implicit hydrogens, then vacancies; it completes notes on another metal."""
    plain = {rx.identity(s).labels[0]: rx.cxsmiles(s) for s in rx.isomers(PT_AMMINES, geometry="SPL")}
    pyridine = {rx.identity(s).labels[0]: rx.cxsmiles(s) for s in rx.isomers("Cl[Pt](<-N)(n1ccccc1)Cl", geometry="SPL")}
    vacant = Chem.AddHs(Chem.MolFromSmiles("Cl[Pt@SP2](Br)N"))
    assert rdDistGeom.EmbedMolecule(vacant, randomSeed=7) == 0  # RDKit's own embed leaves the tag's vacancy
    next(a for a in vacant.GetAtoms() if a.GetSymbol() == "Pt").SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    noted_pd = NOTED_PD_PT.replace(":5.atomNote.s0:6.atomNote.SPL:7.atomNote.s1:8.atomNote.s2:9.atomNote.s3", "")
    cobalt = Chem.MolFromSmiles("O[Co@OH19](Cl)(C)(N)(F)P")
    for name, tagged, specified in (
        ("u-shape-cis", "[Cl-]->[Pt@SP1+2](<-[Cl-])(<-[NH3])<-[NH3]", plain["cis"]),
        ("4-shape-trans", Chem.MolFromSmiles("[Cl-]->[Pt@SP2+2](<-[Cl-])(<-[NH3])<-[NH3]"), plain["trans"]),
        ("z-shape-cis", "[Cl-]->[Pt@SP3+2](<-[Cl-])(<-[NH3])<-[NH3]", plain["cis"]),
        ("implicit-hydrogen", "Cl[Pt@SP1H](N)N", rx.cxsmiles(rx.parse_smiles("Cl[Pt@SP1]([H])(N)N", remove_hs=False))),
        (
            "missing-ligand-vacancy",
            "Cl[Pt@SP2](Br)N",
            rx.cxsmiles(rx.isomers(vacant, geometry="SPL", observed=True)[0]),
        ),
        ("beside-agreeing-notes", plain["cis"].replace("[Pt+2]", "[Pt@SP1+2]"), plain["cis"]),
        (
            "completes-another-metal",
            noted_pd.replace("[Pt+2]", "[Pt@SP1+2]"),
            rx.cxsmiles(rx.parse_smiles(NOTED_PD_PT)),
        ),
        ("pyridine-bond-redrawn-dative", "Cl[Pt@SP1](<-N)(n1ccccc1)Cl", pyridine["cis"]),
        ("octahedral-tag-renumbered", Chem.RenumberAtoms(cobalt, [6, 5, 4, 3, 2, 1, 0]), rx.cxsmiles(cobalt)),
    ):
        source = rx.parse_smiles(tagged) if isinstance(tagged, str) else tagged
        assert (rx.cxsmiles(source), [rx.cxsmiles(s) for s in rx.isomers(source)]) == (specified, [specified]), name


def test_native_metal_tag_naming_no_one_seating_is_refused():
    untagged = Chem.MolFromSmiles("Cl[Pt](Cl)(N)N")
    untagged.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_SQUAREPLANAR)  # as a caller may, with no number
    cis = rx.cxsmiles(rx.isomers(PT_AMMINES, geometry="SPL").select(label="cis"))
    for source, message in (
        (untagged, "on 4 neighbours has no permutation"),
        ("C1=C->[Pt@SP1]<-1(Cl)Cl", "counts a haptic face's atoms or a bonded metal"),
        (cis.replace("[Pt+2]", "[Pt@SP2+2]"), "state different arrangements; keep one"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.isomers(source)


def test_write_dative_sets_a_raw_hand_beside_the_ez_it_writes():
    """A label's raw hand wins over the one the coordinates and tags hold; the Z it asks for is written beside it."""
    mol = rx.parse_smiles("C/C=C/C[N@H](C)->[Cu+]<-[Cl-]")
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0) == 0
    expected = Chem.Mol(mol)
    expected.GetAtomWithIdx(4).InvertChirality()
    raw = "CW" if expected.GetAtomWithIdx(4).GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "CCW"
    specified = LigandStereo(
        elements=(
            StereoElement(kind="point", atoms=(4,), symbols=("N",), code=raw),
            StereoElement(kind="ez", atoms=(1, 2), symbols=("C", "C"), code="Z"),
        )
    )

    text, at, _bonds, _unwritable = write_dative(mol, specified)

    back = rx.parse_smiles(text)
    assert defined_stereo_label(back, metal_indices(back)).bonds == {frozenset((at[1], at[2])): "Z"}
    for view in (expected, back):
        for bond in view.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        view.UpdatePropertyCache(strict=False)
    assert back.HasSubstructMatch(expected, useChirality=True)
