"""`stereo.py`: the ligand stereo a Mol specifies or a conformer shows, with its metals cut."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor

import rxembed as rx
from rxembed import stereo
from rxembed.graph import metal_indices
from rxembed.identity.smiles_io import parse_smiles
from rxembed.utils import TETRAHEDRAL_TAGS, mirror_tag
from tests.geometry_fixtures import embedded, mirrored, placed
from tests.metal_fixtures import BINAP_PDCL2

# Six carboxylates give one fragment 64 resonance forms, past the 32 the E/Z check searches.
HEXACARBOXYLATE = "C(CC(=O)[O-])(CC(=O)[O-])C(CC(=O)[O-])(CC(=O)[O-])C(CC(=O)[O-])CC(=O)[O-]"
# A biaryl axis beside a thiazolidinone imine, its hand drawn by a wedge
IMINE_AXIS = r"C/N=C1\SCCC(=O)N1c1ccccc1C(C)C |wU:8.7|"
# Bromo-iodo phenylpyrrole: its C7-N8 axis carries a native atrop tag
PHENYLPYRROLE_AXIS = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"
NI_CHELATE_IMINE = r"C/C1=N/[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1"
PD_METHYLPROPANEDIAMINE = "[Cl-]->[Pd+2]1(<-[Cl-])<-N(C)CC(C)CN(C)->1"
CO_BIS_EN = "[Cl-]->[Co+3]12(<-[Cl-])(<-NCCN->1)<-NCCN->2"
# (BIPHEP)PdCl2: the P,P chelate alone locks the biphenyl axis, whose CH ortho atoms leave it unhindered
BIPHEP_PDCL2 = "[Cl-]->[Pd+2]1(<-[Cl-])<-P(c2ccccc2)(c2ccccc2)c2ccccc2-c2ccccc2P->1(c1ccccc1)c1ccccc1"
PD_ALKENE = "CC=C(Cl)Br.[Pd+2]"  # a free Pd beside the alkene, so its CX E/Z is applied on the metal-cut graph
_MIRROR = {"R": "S", "S": "R", "M": "P", "P": "M"}
CFCLBRI = "[C@](F)(Cl)(Br)I"  # placed with its carbon first, then F, Cl, Br and I at four carrier positions
COPLANAR = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]
TETRAHEDRON = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def _stereo(*rows):
    """Return the `LigandStereo` of ``(kind, symbols, atoms, code)`` rows."""
    return stereo.LigandStereo(
        elements=tuple(stereo.StereoElement(kind=k, symbols=s, atoms=a, code=c) for k, s, a, c in rows)
    )


_C1R, _N3S = ("point", ("C",), (1,), "R"), ("point", ("N",), (3,), "S")
_E34, _M56 = ("ez", ("C", "C"), (3, 4), "E"), ("axis", ("C", "C"), (5, 6), "M")


def _cx_ez(smiles, code="E"):
    """Return `smiles` as RDKit parses it with a CX E/Z record `code` on atoms 1 and 2, as rx.cxsmiles writes one."""
    return Chem.MolFromSmiles(f"{smiles} |atomProp:1._rxEZ0.{code}:2._rxEZ0.{code}|")


def _drawn(mol):
    """Return `mol` with its 2D drawing read as a 3D conformer."""
    rdDepictor.Compute2DCoords(mol)
    mol.GetConformer().Set3D(True)
    return mol


def _props(mol):
    """Return everything `mol` carries, to show a reader left it as given."""
    return mol.ToBinary(Chem.PropertyPickleOptions.AllProps)


def _bare(mol):
    """Return a copy of `mol` that specifies no point or E/Z, so a reader takes them from the conformer."""
    out = Chem.Mol(mol)
    Chem.RemoveStereochemistry(out)
    return out


def _v3000(mol):
    """Return `mol` through a V3000 block read as RDKit reads a metal MolFile: tags from 3D, the cleanup skipped."""
    read = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol, forceV3000=True), removeHs=False, sanitize=False)
    Chem.SanitizeMol(read)
    return read


def _geometry(smiles, isomer=0):
    """Return a bare rx.embed geometry of one listed isomer, its notes cleared as a reader leaves them."""
    mol = rx.embed(rx.isomers(smiles)[isomer], n=1, seed=1).mol
    for atom in mol.GetAtoms():
        atom.ClearProp("atomNote")
    return mol


def test_ligand_stereo_orders_points_then_bonds_then_axes():
    label = _stereo(_C1R, _E34, _M56)

    assert str(label) == "C1:R,C3=C4:E,C5-C6:M"
    assert (label.points, label.bonds, label.axes) == ({1: "R"}, {frozenset((3, 4)): "E"}, {(5, 6): "M"})
    assert (bool(label), bool(stereo.LigandStereo()), str(stereo.LigandStereo())) == (True, False, "")
    assert str(label.updated(_stereo(("point", ("C",), (1,), "S"), _N3S))) == "C1:S,C3=C4:E,C5-C6:M,N3:S"


def test_matches_stereo_reads_label_index_symbols_or_code():
    for elements, selector, matches in (
        ((_C1R,), "C1:R", True),
        ((_C1R,), "1R", True),
        ((_C1R,), "C:R", True),
        ((_C1R,), "R", True),
        ((_C1R,), "S", False),
        ((_C1R,), "C=C:R", False),
        ((_C1R,), "X", False),
        ((_C1R, _N3S), "C:R", True),
        ((_C1R, _N3S), "C:R,N:S", True),
        ((_C1R, _N3S), "C1:R,N3:R", False),
        ((_E34,), "C=C:E", True),
        ((_E34,), "3=4:E", True),
        ((_E34,), "E", True),
        ((_M56,), "M", True),
    ):
        assert stereo.matches_stereo(_stereo(*elements), selector) is matches, (elements, selector)


def test_matches_stereo_refuses_an_ambiguous_shorthand():
    for elements, selector, choices in (
        ((_C1R, ("point", ("C",), (3,), "S")), "C:R", r"C1:R.*C3:S"),
        ((_C1R, _N3S), "R", r"C1:R.*N3:S"),
        ((("ez", ("C", "C"), (1, 2), "E"), ("ez", ("C", "C"), (3, 4), "Z")), "E", r"C1=C2:E.*C3=C4:Z"),
    ):
        with pytest.raises(ValueError, match=f"is ambiguous.*use one of.*{choices}"):
            stereo.matches_stereo(_stereo(*elements), selector)


def test_ligand_misses_names_each_element_read_otherwise():
    """A point both sides tag is compared by its tag; a pseudoasymmetric one keeping its code is the other hand."""
    alkenol = "C[C@@H](O)/C=C/C"
    for specified, observed, misses in (
        (alkenol, alkenol, []),
        (alkenol, "C[C@H](O)/C=C/C", [("point", (1,), "R", "S")]),
        (alkenol, r"C[C@@H](O)/C=C\C", [("ez", (3, 4), "E", "Z")]),
        (alkenol, "CC(O)C=CC", [("point", (1,), "R", ""), ("ez", (3, 4), "E", "")]),
        (
            "C[C@@H](O)[C@@H](O)[C@H](C)O",
            "C[C@H](O)[C@H](O)[C@@H](C)O",
            [("point", (1,), "R", "S"), ("point", (3,), "s", "the other hand"), ("point", (5,), "S", "R")],
        ),
    ):
        specified_mol, observed_mol = Chem.MolFromSmiles(specified), Chem.MolFromSmiles(observed)
        labels = stereo.defined_stereo_label(specified_mol), stereo.defined_stereo_label(observed_mol)
        found = stereo.ligand_misses(*labels, specified_mol, observed_mol)
        assert [(miss.kind, miss.atoms, miss.specified, miss.observed) for miss in found] == misses, observed


def test_stereo_symmetry_maps_mirrors_only_by_graph_symmetry():
    """The unlike diamine palladium is its own mirror image with its amines exchanged; the like one and an alkenol
    are not, and an E alkene is not its Z."""
    for specified, symmetric in (
        ("C[C@@H]1C[N@@H](C)->[Pd+2](<-[Cl-])(<-[Cl-])<-[N@@H](C)C1", True),
        ("CC1C[N@@H](C)->[Pd+2](<-[Cl-])(<-[Cl-])<-[N@H](C)C1", False),
        ("C[C@@H](O)/C=C/C", False),
    ):
        mol = parse_smiles(specified)
        mirror = Chem.Mol(mol)
        for atom in mirror.GetAtoms():
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
        order = stereo.stereo_symmetry(mol, mirror, stereo.defined_stereo_label(mol, metal_indices(mol)))
        assert (order is not None) is symmetric, specified
    trans, cis = parse_smiles("CC(O)/C=C/C"), parse_smiles(r"CC(O)/C=C\C")
    label = stereo.defined_stereo_label(trans)
    assert [stereo.stereo_symmetry(trans, side, label) is None for side in (trans, cis)] == [False, True]


def test_defined_stereo_label_reads_a_coordinated_mol():
    """A metal-cut donor keeps its hand and E/Z references in any atom order, as a CX-encoded E/Z beside a metal is
    applied in any; a locked E/Z or a stale tag is none."""
    bis_imine = parse_smiles(r"[H]/[N](=C(\C))->[Pt+4](<-[Cl-])(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[N](/[H])=C(/C)")
    stale = Chem.AddHs(Chem.MolFromSmiles("CC(C)O"))
    stale.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)  # as a reader that skips cleanup tags it
    for mol, label in (
        (Chem.MolFromSmiles(r"C/C=C/[C@H](O)c1c(C)cccc1-n1c(C)ccc1Br |wU:11.11|"), "C3:S,C1=C2:E,C11-N12:P"),
        (parse_smiles("C[P@H](->[Pt](Cl)(Cl)Cl)CC"), "P1:S"),
        (parse_smiles("C[N@H](->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br"), "N1:R"),
        (bis_imine, "N0=C1:E,N8=C9:E"),
        (Chem.RenumberAtoms(bis_imine, list(reversed(range(bis_imine.GetNumAtoms())))), "N10=C9:E,N2=C1:E"),
        (_cx_ez("CC=CC"), "C1=C2:E"),
        (Chem.RenumberAtoms(_cx_ez(PD_ALKENE), [5, 4, 3, 2, 1, 0]), "C4=C3:E"),
        (_cx_ez(PD_ALKENE, "Z"), "C1=C2:Z"),
        (parse_smiles(NI_CHELATE_IMINE), ""),
        (_cx_ez("CC1=N[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1"), ""),
        (stale, ""),
    ):
        before = _props(mol)
        assert str(stereo.defined_stereo_label(mol, metal_indices(mol))) == label, Chem.MolToSmiles(mol)
        assert _props(mol) == before, Chem.MolToSmiles(mol)


def test_defined_stereo_label_refuses_an_unreadable_cx_ez():
    for cx, message in (
        ("CC=CC |atomProp:1._rxEZ0.X:2._rxEZ0.X|", "invalid CX E/Z value 'X' on atom 1"),
        ("CC=CC |atomProp:1._rxEZ0.E|", "record 0 must name two atoms with one configuration"),
        ("CC=CC |atomProp:1._rxEZ0.E:2._rxEZ0.Z|", "record 0 must name two atoms with one configuration"),
        ("CCCC |atomProp:1._rxEZ0.E:2._rxEZ0.E|", "record 0 does not name a double bond"),
        ("CC=CC |atomProp:1._rxEZ0.E:2._rxEZ0.E:1._rxEZ1.E:2._rxEZ1.E|", r"duplicate CX E/Z record for bond \(1, 2\)"),
        (r"C/C=C/C |atomProp:1._rxEZ0.Z:2._rxEZ0.Z|", r"native and CX E/Z disagree on bond \(1, 2\)"),
    ):
        with pytest.raises(ValueError, match=message):
            stereo.defined_stereo_label(Chem.MolFromSmiles(cx))


def test_apply_point_stereo_tags_each_named_centre_with_its_code():
    """An R/S code is set by CIP on the coordinated graph, a CW/CCW code as the tag itself; other centres stay."""
    for smiles, label, centres, applied in (
        ("CC(O)CC", _stereo(("point", ("C",), (1,), "R")), {1}, "C1:R"),
        ("CC(O)CC", _stereo(("point", ("C",), (1,), "S")), {1}, "C1:S"),
        ("CC(O)CC", _stereo(("point", ("C",), (1,), "S")), set(), ""),
        ("FC(Cl)(Br)I", _stereo(("point", ("C",), (1,), "CCW")), {1}, "C1:S"),
    ):
        mol = Chem.MolFromSmiles(smiles)

        stereo.apply_point_stereo(mol, label, centres)

        assert str(stereo.defined_stereo_label(mol)) == applied, (smiles, label, centres)


def test_stereo_graph_points_are_the_measurable_centres():
    """Each row: the points and, of those, the donor hands only coordination fixes.

    A bound S or N keeps a lone pair or a metal as its fourth carrier, a free amine inverts, a bridgehead N's hand is
    its cage's, equivalent arms, carriers only a Lewis form tells apart or a chloride bridging two Pd leave no centre.
    """
    for smiles, points, locked in (
        ("CC[S@](C)->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3]", {2}, {2}),
        ("CC[NH](C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", {2}, {2}),
        ("C[N@H](->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br", {1}, {1}),
        ("C[P@H](->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-])CC", {1}, set()),
        ("C[N@]1C[C@H]1C", {1, 3}, set()),
        ("CC[NH]C", set(), set()),
        ("COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", {10, 12, 14, 19}, set()),
        ("CN12->[Rh+](<-[I-])(<-[C-]#[O+])<-P3(C)CN(CN(C1)C3)C2", set(), set()),
        ("C[P@](=O)([O-])CC", set(), set()),
        ("[Cl-]->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]->1", set(), set()),
    ):
        mol = parse_smiles(smiles)
        graph = stereo.StereoGraph(mol, metal_indices(mol))
        assert (graph.points, graph.locked_centres) == (points, locked), smiles


def test_stereo_graph_locks_coordination_fixed_bonds_and_axes():
    """A chelate ring locks an endocyclic double bond, a chelate or four ortho groups a biaryl axis.

    A bound diene's central bond is its s-cis/s-trans class; a monodentate or side-on alkene, a double bond in its own
    ring, a bipyridine and a biaryl with one donor bound lock nothing.
    """
    for smiles, bonds, axes in (
        ("C[C]1->2=[C]->3(C)[CH2-]->[Zr+2]23<-[CH2-]1", {frozenset((1, 2))}, []),
        (NI_CHELATE_IMINE, {frozenset((1, 2))}, []),
        ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", set(), []),
        ("C[CH]1=[CH](F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", set(), []),
        ("[Cl-]->[Pd+2]1(<-[Cl-])<-NC2=C(CCCC2)N->1", set(), []),
        (BIPHEP_PDCL2, set(), [(21, 22)]),
        ("Cc1ccc2ccccc2c1-c1c(C)ccc2ccccc12.[Pd+2]", set(), [(10, 11)]),
        ("[Cl-]->[Pd+2]1(<-[Cl-])<-n2ccccc2-c2cccc[n]->12", set(), []),
        ("[Cl-]->[Pd+2](<-[Cl-])(<-N)<-P(c1ccccc1)(c1ccccc1)c1ccccc1-c1ccccc1P(c1ccccc1)c1ccccc1", set(), []),
    ):
        mol = parse_smiles(smiles)
        graph = stereo.StereoGraph(mol, metal_indices(mol))
        assert (graph.locked, graph.locked_axes) == (bonds, axes), smiles


def test_specified_ez_is_missed_unless_the_conformer_shows_it():
    """A conformer shows a native E/Z it was built with, not one retagged since; a CX record it never shows."""
    butene = {frozenset((1, 2))}
    retagged = embedded("C/C=C/C")
    retagged.GetBondBetweenAtoms(1, 2).SetStereo(Chem.BondStereo.STEREOZ)
    encoded = placed(_cx_ez("CC=CC"), [(0, 0, 0), (1.5, 0, 0), (2, 1.3, 0), (3.5, 1.3, 0)])
    for name, mol, specified, missed in (
        ("drawn", Chem.MolFromSmiles("C/C=C/C"), butene, butene),
        ("embedded", embedded("C/C=C/C"), butene, set()),
        ("retagged", retagged, butene, butene),
        ("encoded", encoded, butene, butene),
        ("unspecified", embedded("CC=CC"), set(), set()),
    ):
        graph = stereo.StereoGraph(mol)
        assert (graph.specified_ez, graph.missed_ez) == (specified, missed), name


def test_enumerable_ez_is_specified_or_resonance_stable():
    """An allyl anion's E/Z moves in resonance unless specified or cleared by a request; a butatriene is one element,
    none when an end carries two hydrogens."""
    for smiles, cleared, pairs in (
        ("CC=CC", (), [(1, 2)]),
        ("CC=C=C=CC", (), [(1, 2)]),
        ("CC=C=C=C", (), []),
        ("CC=C[CH2-]", (), []),
        (r"C/C=C/[CH2-]", (), [(1, 2)]),
        ("CC=C[CH2-]", [(1, 2)], [(1, 2)]),
    ):
        graph = stereo.StereoGraph(Chem.MolFromSmiles(smiles))
        work = graph.work
        potential = {e.centeredOn: e for e in stereo.potential_stereo(work) if e.type == Chem.StereoType.Bond_Double}
        found = map(work.GetBondWithIdx, graph.enumerable_ez(work, potential, cleared))
        assert sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in found) == pairs, (smiles, cleared)


def test_clear_ez_clears_selected_ez_and_says_if_any_changed():
    butene, hexadiene = frozenset((1, 2)), {frozenset((1, 2)), frozenset((3, 4))}
    for mol, pairs, changed, kept in (
        (Chem.MolFromSmiles("C/C=C/C"), None, True, set()),
        (Chem.MolFromSmiles("CC=CC"), None, False, set()),
        (Chem.MolFromSmiles("C/C=C/C"), set(), False, {butene}),
        (_cx_ez("CC=CC"), {butene}, True, set()),
        (Chem.MolFromSmiles("C/C=C/C=C/C"), {butene}, True, hexadiene - {butene}),
    ):
        found = stereo.clear_ez(mol, pairs), stereo.StereoGraph(mol).specified_ez
        assert found == (changed, kept), (Chem.MolToSmiles(mol), pairs)


def test_stereo_graph_refuses_stereo_nothing_can_hold():
    """A carbanion's three carriers and lone pair give no fourth position to read; a five-ring holds its imine cis."""
    carbanion = Chem.AddHs(Chem.MolFromSmiles("C[C-](CC)c1ccccc1"))
    carbanion.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    for mol, fact, message in (
        (carbanion, "points", r"atom\(s\) \[1\] cannot be read from 3D; remove the tag or use four neighbours"),
        (Chem.MolFromSmiles(r"C/N1=C\CN->[Pd+2](<-[Cl-])(<-[Cl-])<-1"), "locked", "N1=C2 puts Pd5 and C3 trans, but"),
        (_cx_ez("CN1=CCN->[Pd+2](<-[Cl-])(<-[Cl-])<-1", "Z"), "locked", "N1=C2 puts C0 and C3 cis, but its ring"),
        (_cx_ez("C[NH2]->[Pd+2](<-[Cl-])(<-[Cl-])<-N"), "locked", "does not name a double bond"),
    ):
        with pytest.raises(ValueError, match=message):
            getattr(stereo.StereoGraph(mol, metal_indices(mol)), fact)


def test_stereo_from_3d_reads_the_hand_its_mirror_inverts():
    """A pyramidal P or a held N reads its lone pair; an E/Z away from a conjugated group survives the resonance bound.

    The free copper's lactate, its rings perceived fast, is the input whose caches the axis query once filled.
    """
    imine_axis = embedded(IMINE_AXIS)
    imine_axis.GetBondBetweenAtoms(8, 6).SetBondDir(Chem.BondDir.NONE)  # an embedded Mol keeps the tag, not the wedge
    lactate = embedded("[Cu+].C[C@H](O)C(=O)[O-]")
    Chem.FastFindRings(lactate)
    for name, mol in (
        ("phosphine", embedded("C[P@](CC)c1ccccc1")),
        ("aziridine", embedded("C[N@]1C[C@H]1C")),
        ("imine axis", imine_axis),
        ("alkene past bound", embedded("C/C=C/C" + HEXACARBOXYLATE, seed=1)),
        ("Cu lactate", lactate),
    ):
        metals, before = metal_indices(mol), _props(mol)
        read = stereo.stereo_from_3d(mol, metals)

        assert read, name
        assert read == stereo.defined_stereo_label(mol, metals), name
        mirror_read = stereo.stereo_from_3d(mirrored(mol), metals)
        assert mirror_read.elements == tuple(replace(e, code=_MIRROR.get(e.code, e.code)) for e in read.elements), name
        assert _props(mol) == before, name


def test_stereo_from_3d_reads_no_stereo_a_conformer_cannot_show():
    """A point outside its carriers' tetrahedron, an E/Z a resonance form moves or one past the resonance bound in a
    conjugated group, and a coplanar chelated biaryl read nothing rather than an arbitrary hand."""
    binap = _drawn(parse_smiles(BINAP_PDCL2, remove_hs=False))
    point = binap.GetConformer().GetAtomPosition(0)
    binap.GetConformer().SetAtomPosition(0, (point.x, point.y, 0.01))
    for name, mol in (
        ("C above flat carriers", placed(CFCLBRI, [(0, 0, 1), *COPLANAR])),
        ("C outside tetrahedron", placed(CFCLBRI, [(2, 2, 2), *TETRAHEDRON])),
        ("allyl anion", _bare(_drawn(Chem.MolFromSmiles(r"C/C=C/[CH2-]")))),
        ("enal past bound", _bare(embedded("O=C/C=C/C" + HEXACARBOXYLATE, seed=1))),
        ("flat BINAP", binap),
    ):
        before = _props(mol)
        assert not stereo.stereo_from_3d(mol, metal_indices(mol)), name
        assert _props(mol) == before, name


def test_stereo_from_3d_with_apply_writes_what_it_reads():
    """An invalid point tag is cleared, an axis tagged the other hand is retagged as measured, and a coordination-locked
    E/Z is written on the Mol but left out of the record."""
    nickel = _bare(_geometry(NI_CHELATE_IMINE))
    imine_axis = embedded(IMINE_AXIS)
    axis = imine_axis.GetBondBetweenAtoms(8, 9)
    axis.SetStereo(stereo.ATROP_STEREO[1 - stereo.ATROP_STEREO.index(axis.GetStereo())])  # the other hand
    for name, mol, written in (
        ("difluoroethene", _bare(embedded("F/C=C/F")), [(1, 2)]),
        ("butanol", _bare(embedded("C[C@H](O)CC")), []),
        ("C outside tetrahedron", placed(CFCLBRI, [(2, 2, 2), *TETRAHEDRON]), []),
        ("axis tagged otherwise", imine_axis, [(1, 2)]),
        ("axis mirrored", mirrored(embedded(IMINE_AXIS)), [(1, 2)]),
        ("Ni chelate imine", nickel, [(1, 2)]),
    ):
        metals = metal_indices(mol)
        read = stereo.stereo_from_3d(mol, metals, apply=True)

        assert stereo.defined_stereo_label(mol, metals) == read, name
        assert all(mol.GetBondBetweenAtoms(*pair).GetStereo() != Chem.BondStereo.STEREONONE for pair in written), name


def test_without_stale_point_tags_keeps_stereocentre_tags():
    """A MolFile read tags every sp3 atom; a tag stays where the conformer shows a centre, a pseudoasymmetric carbon
    judged beside its amine hands, and where 3D cannot measure the atom RDKit's cleanup decides."""
    flat_butanol = embedded("C[C@H](O)CC")
    positions = flat_butanol.GetConformer().GetPositions()
    positions[1] = positions[[0, 2, 3]].mean(axis=0)  # the centre in the plane of its heavy carriers
    flat_butanol.GetConformer().SetPositions(positions)
    pushed = _geometry(CO_BIS_EN)  # C4 mirrored through the plane of N3, C5 and one H: outside its tetrahedron
    positions = pushed.GetConformer().GetPositions()
    plane = positions[[n.GetIdx() for n in pushed.GetAtomWithIdx(4).GetNeighbors()][:3]]
    normal = np.cross(plane[1] - plane[0], plane[2] - plane[0])
    positions[4] -= 2 * np.dot(positions[4] - plane[0], normal) / np.dot(normal, normal) * normal
    pushed.GetConformer().SetPositions(positions)
    stale = Chem.AddHs(Chem.MolFromSmiles("CC(C)O"))
    carbanion = Chem.AddHs(Chem.MolFromSmiles("C[C-](CC)c1ccccc1"))
    for mol in (stale, carbanion):
        mol.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    for name, mol, kept in (
        ("unlike Pd diamine", _v3000(_geometry(PD_METHYLPROPANEDIAMINE)), {6}),
        ("like Pd diamine", _v3000(_geometry(PD_METHYLPROPANEDIAMINE, 1)), set()),
        ("Co(en)2", _v3000(_geometry(CO_BIS_EN)), set()),
        ("Co(en)2 CH2 outside", _v3000(pushed), set()),
        ("flat butanol", flat_butanol, {1}),
        ("isopropanol unplaced", stale, set()),
        ("carbanion", carbanion, {1}),
    ):
        before, out = _props(mol), stereo.without_stale_point_tags(mol)
        assert {atom.GetIdx() for atom in out.GetAtoms() if atom.GetChiralTag() in TETRAHEDRAL_TAGS} == kept, name
        assert _props(mol) == before, name


def test_renumber_atoms_keeps_each_atropisomer_axis_hand():
    """Swapping C1 and C5 changes the lowest neighbour at one end of the C7-N8 axis; also C9 and C13, at both."""
    mol = Chem.MolFromSmiles(PHENYLPYRROLE_AXIS)
    specified = {(e.kind, frozenset(e.atoms), e.code) for e in stereo.defined_stereo_label(mol).elements}
    count = mol.GetNumAtoms()
    one_end = [0, 5, 2, 3, 4, 1, *range(6, count)]
    both_ends = [*one_end[:9], 13, *one_end[10:13], 9, *one_end[14:]]
    for name, order in (("one end", one_end), ("both ends", both_ends), ("reversed", list(reversed(range(count))))):
        label = stereo.defined_stereo_label(stereo.renumber_atoms(mol, order))
        assert {(e.kind, frozenset(order[a] for a in e.atoms), e.code) for e in label.elements} == specified, name
