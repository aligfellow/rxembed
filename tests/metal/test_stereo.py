"""`metal/stereo.py`: donor and site classes, metal hands, and the winding a haptic face takes on its metal."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed.metal import stereo
from tests.metal_fixtures import BUTADIENE_FE_CO3, ISOPRENE_FE_CO3


def test_sites_are_classed_by_graph_symmetry():
    """A kappa2 sulfate drawn with an expanded octet has equivalent O, as a carboxylate's are; an aqua H...Cl contact
    drawn as a zero bond is no chelate bite, so it seats the three isomers the bare complex has."""
    assert len(rx.isomers("[O-]S1(=O)=[O]->[Pd+2](<-[Cl-])(<-[P](C)(C)C)<-[O-]1", geometry="SPL")) == 1
    aqua = rx.parse_smiles("[H]O([H])->[Pt+2](<-[NH3])(<-[Br-])<-[Cl-]", remove_hs=False)
    contact = Chem.RWMol(aqua)
    contact.AddBond(0, 6, Chem.BondType.ZERO)  # H0...Cl6

    assert [len(rx.isomers(mol, geometry="SPL")) for mol in (aqua, contact.GetMol())] == [3, 3]


def test_hand_is_read_where_placed_and_selected_by_glyph():
    """Five distinct donors place the empty octahedral vertex; two donors of a trigonal pyramid lie in one plane. A
    selection takes a hand as its word or its glyph."""
    assert rx.identity(rx.isomers("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]", geometry="OCT")[0]).centres[0].hand
    assert not rx.identity(rx.isomers("[Cl-]->[Zn+2]<-[Br-]", geometry="TPY")[0]).centres[0].hand
    enantiomers = rx.isomers("[F-]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-]", geometry="TET")
    selected = [enantiomers.select(hand=hand) for hand in (" Λ ", "delta")]
    assert [rx.identity(one).centres[0].hand for one in selected] == ["lambda", "delta"]


_Z_FLUOROPROPENE_PLATINUM = (  # an eta2 face whose own E/Z orders its re/si sides
    r"C[CH]1=[CH](F)->[Pt+2]<-1(<-[NH3])(<-[Cl-])<-[Br-] "
    "|atomProp:1.atomNote.s0+:2.atomNote.s0+:4.atomNote.SPL:5.atomNote.s1:6.atomNote.s3:7.atomNote.s2,c:1|"
)


WRITTEN = {
    "delta cobaloxime": "CC1=C(C)[N-](O)->[Co+3]2(<-[CH2-]CCC#N)(<-[N]1=O)(<-[N](=O)C(C)=C(C)[N-]->2O)<-[n]1ccn(C)c1C "
    "|atomProp:4.atomNote.s0:6.atomNote.OCT-delta:7.atomNote.s2:12.atomNote.s4:14.atomNote.s5:20.atomNote.s1:"
    "22.atomNote.s3|",
    "trihalo ferrocene": "O[c-]12->[Fe]3456789(<-[cH]1[c]->3(I)[c]->4(Br)[c]->52Cl)<-[cH]1[c]->6(Cl)[c]->7(Br)"
    "[c]->8(I)[c-]->91O |atomProp:1.atomNote.s0-:2.atomNote.LIN:3.atomNote.s0-:4.atomNote.s0-:6.atomNote.s0-:"
    "8.atomNote.s0-:10.atomNote.s1-:11.atomNote.s1-:13.atomNote.s1-:15.atomNote.s1-:17.atomNote.s1-|",
    "Z-fluoropropene Pt": _Z_FLUOROPROPENE_PLATINUM,
}


def test_atom_maps_and_order_name_no_hand_or_winding():
    """Neither atom maps nor atom order change the identity or the written text of a metal hand or face winding."""
    read, expected = {}, {}
    for case, text in WRITTEN.items():
        mol = Chem.AddHs(rx.parse_smiles(text))
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        mapped = Chem.Mol(mol)
        for atom in mapped.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)  # hydrogens too: RDKit keeps a mapped H explicit
        written = rx.parse_smiles(rx.cxsmiles(mapped))
        for atom in written.GetAtoms():
            atom.SetAtomMapNum(0)
        read[case] = (rx.identity(mapped), [rx.cxsmiles(one) for one in (written, mol, reordered)])
        expected[case] = (rx.identity(mol), [text] * 3)

    assert read == expected


def _cyclopentadienyl_iron():
    """Return a 1-fluoro-2-bromocyclopentadienide on Fe, its five face carbons bound."""
    rw = Chem.RWMol(Chem.CombineMols(Chem.MolFromSmiles("[c-]1(F)c(Br)ccc1"), Chem.MolFromSmiles("[Fe+2]")))
    iron = rw.GetNumAtoms() - 1
    for atom in (atom.GetIdx() for atom in rw.GetAtoms() if atom.GetIsAromatic()):
        rw.AddBond(atom, iron, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol, iron


def _diene_token(diene, path, degrees):
    """Return the token of the diene `path` on Fe2 of an iron tricarbonyl, its C1-C2-C3-C4 torsion at `degrees`."""
    phi = np.radians(degrees)
    pos = np.zeros((diene.GetNumAtoms(), 3))
    pos[path] = [(-0.7, 1.2, 0.0), (0.0, 0.0, 0.0), (1.45, 0.0, 0.0), (2.15, 1.2 * np.cos(phi), 1.2 * np.sin(phi))]
    pos[2] = (0.7, 0.6, -1.7)
    return stereo.face_winding(diene, pos, 2, path, stereo.donor_classes(diene, path))


_E_FLUOROPROPENE_PLATINUM = r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
# id -> a haptic complex and the face configuration of each species rx.isomers lists, sorted
FACES = {
    "butadiene Fe(CO)3": (BUTADIENE_FE_CO3, ["M", "P", "c"]),  # one s-cis form, its own mirror image
    "isoprene Fe(CO)3": (ISOPRENE_FE_CO3, ["+", "-", "M", "P"]),
}


def test_face_takes_allowed_configurations_and_reads_its_side():
    """A rotatable diene adds its s-trans helicity, and a face with two distinct sides names each; an eta2 face reads
    alike with its H implicit.

    A Cp ring with Fe in its plane has no side; an eta2 alkene's side reads alike at any scale; a bound diene is
    s-cis under 90 deg and P or M by its s-trans helicity above, abstaining on either boundary.
    """
    configurations = {
        name: sorted(
            site.winding
            for identity in map(rx.identity, rx.isomers(smiles))
            for site in identity.centres[0].vertices
            if site is not None and len(site.atoms) > 1
        )
        for name, (smiles, _expected) in FACES.items()
    }
    assert configurations == {name: expected for name, (_smiles, expected) in FACES.items()}
    assert rx.cxsmiles(rx.parse_smiles(_Z_FLUOROPROPENE_PLATINUM)) == _Z_FLUOROPROPENE_PLATINUM  # its H implicit

    cyclopentadienyl, iron = _cyclopentadienyl_iron()
    face = [atom.GetIdx() for atom in cyclopentadienyl.GetAtoms() if atom.GetIsAromatic()]
    pos = np.zeros((cyclopentadienyl.GetNumAtoms(), 3))
    theta = 2 * np.pi * np.arange(len(face)) / len(face)
    pos[face, 0], pos[face, 1] = np.cos(theta), np.sin(theta)
    pos[iron] = (2.0, 0.0, 0.0)
    assert stereo.face_winding(cyclopentadienyl, pos, iron, face, stereo.donor_classes(cyclopentadienyl, face)) == ""

    alkene = Chem.AddHs(rx.parse_smiles(_E_FLUOROPROPENE_PLATINUM))
    platinum, face = 4, (1, 2)
    donors = [bond.GetOtherAtomIdx(platinum) for bond in alkene.GetAtomWithIdx(platinum).GetBonds()]
    pos = np.zeros((alkene.GetNumAtoms(), 3))
    for atom, point in {
        0: (-1.2, 0.7, 0.0),
        1: (-0.5, 0.0, 0.0),
        2: (0.5, 0.0, 0.0),
        3: (1.2, 0.7, 0.0),
        4: (0.0, -1.0, 1e-6),
        11: (-1.2, -0.7, 0.0),
        12: (1.2, -0.7, 0.0),
    }.items():
        pos[atom] = point
    ranks, cip = stereo.donor_classes(alkene, donors), list(Chem.ComputeAtomCIPRanks(alkene))
    assert {stereo.face_winding(alkene, pos * scale, platinum, face, ranks, cip) for scale in (1e-3, 1.0, 1e3)} == {"-"}

    butadiene, isoprene = Chem.MolFromSmiles(BUTADIENE_FE_CO3), Chem.MolFromSmiles(ISOPRENE_FE_CO3)
    tokens = [_diene_token(butadiene, [7, 8, 9, 10], degrees) for degrees in (0, 89, 91, 179, -91, 90, 180)]
    assert tokens == ["c", "c", "P", "P", "M", "", ""]
    assert _diene_token(isoprene, [7, 8, 10, 11], 0) in ("+", "-")  # its methyl tells the two s-cis faces apart


def test_eta2_alkene_with_any_stereo_signs_no_face_relation():
    alkene = Chem.AddHs(rx.parse_smiles(_E_FLUOROPROPENE_PLATINUM))
    alkene.GetBondBetweenAtoms(1, 2).SetStereo(Chem.BondStereo.STEREOANY)

    assert stereo.eta2_signatures(alkene, (1, 2)) == ((), ())
