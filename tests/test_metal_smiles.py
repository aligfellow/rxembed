"""Test canonical dative SMILES and arrangement-bearing CXSMILES round trips."""

from __future__ import annotations

import importlib
import logging
import re
import subprocess
import sys
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_constraints as C  # noqa: N812
from rxembed import metal_core as _metal
from rxembed import metal_enumeration as K  # noqa: N812
from rxembed import metal_isomer as I  # noqa: N812
from rxembed import metal_smiles as S  # noqa: N812
from rxembed import metal_stereo as _metal_stereo
from rxembed import stereo
from rxembed.core import embed as core_embed
from rxembed.metal_core import VACANT, materialized_state
from rxembed.metal_polyhedron import SLOT_BOND_PROP, point_group, vertex_dirs
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.perceive import read_xyz

_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MNH = "examples/structures/mnh.xyz"  # the corresponding Mn hydride minimum
_MA2B2_SEATS = {"cis": [1, 2, 3, 4], "trans": [1, 3, 2, 4]}  # square_planar: 0 and 2 are the trans pair
_MA3B3_SEATS = {"fac": [1, 4, 2, 5, 3, 6], "mer": [1, 2, 3, 4, 5, 6]}  # octahedral: 0/1, 2/3, 4/5 are trans
_CIS3 = [1, 3, 2, 5, 4, 6]  # cis,cis,cis-MA2B2C2: every same-element pair at 90 degrees, so the centre is chiral
_ETA2_ASYM_E = r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
_TWO_ETA2 = r"C/[CH]1=[CH](/F)->[Pt+2]2(<-[CH](Cl)=[CH](Br)->2)(<-[NH3])(<-[Cl-])<-1"
_ATROP_RU_COVALENT_CX = (
    "[Cl-][Ru+2]12([Cl-])([NH2][C@H](c3ccccc3)[C@H]([NH2]1)c1ccccc1)"
    "[P](c1ccccc1)(c1ccccc1)c1ccc3ccccc3c1-c1c([P]2(c2ccccc2)c2ccccc2)ccc2ccccc12 |wU:41.47|"
)
_BINAP_PD = (
    "[Pd+2]%90(<-[Cl-])(<-[Cl-])(<-P(c1ccccc1)(c2ccccc2)c3ccc4ccccc4c3-c3c(P(c4ccccc4)(c5ccccc5)->%90)ccc4ccccc34)"
)
_AZA_BIARYL_CR = "[O+]#[C-]->[Cr]1(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[n]2cccnc2-c2nccc[n]->12"
_NON_CIP_CAGE_AU = "Cn1n[n+]([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[c-](->[Au+]<-[Cl-])c1-c1ccccc1"
_PSEUDO_BIS_ETA2_RU = "CC#[N]->[Ru+]123(<-[Cl-])(<-[N]#CC)(<-[N]#CC)<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31"
_BAXFIQ_DONOR_SLOTS = "CC(C)(C)[N+]#[C-]->[Pt+2](<-[SiH-](c1ccccc1)c1ccccc1)(<-[SiH-](c1ccccc1)c1ccccc1)<-[P](C)(C)C"


def _isomer(smi, geometry, seating):
    """An `Isomer` stated as a vertex ordering: `seating[v]` is the atom sitting at vertex v.

    The intuitive door, and a hermetic one: no conformer, no embed and no `benchmark/corpus`, because a
    vertex ordering already fixes the arrangement and the handedness. Read the record in `metal_polyhedron`
    for what a vertex number means; the convention is not uniform across the shapes.
    """
    return I.Isomer(Chem.MolFromSmiles(smi), geometry, seating)


def _seated(iso):
    """Return the per-vertex donor identity: an element symbol, ``ηn`` for a face, ``·`` for a vacancy.

    Atom indices are renumbered by every write, so they cannot be compared across a round trip; what a
    vertex holds can be.
    """
    return [
        "·" if d == VACANT else (f"η{len(iso.haptic[d])}" if d in iso.haptic else iso.mol.GetAtomWithIdx(d).GetSymbol())
        for d in iso.vertices
    ]


def _planar_chiral_ferrocene(two_chiral_faces=False, point_stereo=False):
    """Return a fully dative ferrocene with one or two directionally substituted Cp faces."""
    asymmetric = "[c-]1(F)c(Br)c(C(F)Cl)cc1" if point_stereo else "[c-]1(F)c(Br)ccc1"
    faces = (Chem.MolFromSmiles(asymmetric), Chem.MolFromSmiles(asymmetric if two_chiral_faces else "[cH-]1cccc1"))
    rw = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(*faces), Chem.MolFromSmiles("[Fe+2]")))
    iron = rw.GetNumAtoms() - 1
    for atom in rw.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == "C":
            rw.AddBond(atom.GetIdx(), iron, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return Chem.AddHs(mol)


def _haptic_windings(mol, iso, cid):
    """Read every planar-chiral haptic winding from one conformer."""
    pos = mol.GetConformer(int(cid)).GetPositions()
    ranks = _metal_stereo.donor_classes(mol, iso.donors)
    return {
        dummy: sign
        for dummy, face in iso.haptic.items()
        if (sign := _metal_stereo.face_winding(mol, pos, iso.metal, face, ranks))
    }


# Run the graph round trip while a meta-path block refuses every optional dependency. The final import checks
# that the block is active; an in-process ``sys.modules`` probe alone would be a null measurement.
_BASE_INSTALL = """
import sys
BLOCKED = ("xyzgraph", "openconf", "scipy", "sklearn", "matplotlib", "prism_pruner", "ase", "xyzrender")

class BaseInstall:
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError("no module named %r (simulated base install)" % name)
        return None

sys.meta_path.insert(0, BaseInstall())

from rdkit import Chem
import rxembed as rx

mol = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pt](<-[NH3])(Cl)Cl"))
text = rx.cxsmiles(rx.enumerate_isomers(mol, "square_planar")[0])
embedded = rx.embed(text, n=2, seed=1)
assert embedded.iso is not None, "the stated arrangement was not read"
assert embedded.ids, "nothing embedded"
assert rx.cxsmiles(embedded.iso) == text, "the string is not a fixed point"
try:
    import xyzgraph
except ImportError:
    pass
else:
    raise AssertionError("the block is not blocking, so this proves nothing")
print(text)
"""


def test_graph_roundtrip_needs_no_optional_deps():
    run = subprocess.run([sys.executable, "-c", _BASE_INSTALL], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "[H]" not in run.stdout, run.stdout


# --- the parse / write contract -------------------------------------------------------------------------


def test_bad_smiles_raises():
    assert S.parse_smiles("CCO").GetNumAtoms() == 3
    with pytest.raises(ValueError, match="could not parse SMILES"):
        S.parse_smiles("C1CC")


def test_covalent_metal_input_normalizes_before_kekulization():
    mol = S.parse_smiles("[Zn](n1ccccc1)n1ccccc1")

    assert S.dative_smiles(mol) == "c1cc[n](->[Zn]<-[n]2ccccc2)cc1"


def test_writer_does_not_invent_a_radical_on_an_unbound_aromatic_sulfur():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccsc1.N->[Pt+](<-[Cl-])<-[Cl-]"))
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)

    text = S.dative_smiles(mol)

    sulfur = next(atom for atom in S.parse_smiles(text).GetAtoms() if atom.GetSymbol() == "S")
    assert sulfur.GetNumRadicalElectrons() == 0


def test_native_atrop_cx_survives_covalent_input_and_metal_enumeration():
    # Test native stereo serialization even for arrangements excluded by the ground-state model.
    isomers = rx.metal(_ATROP_RU_COVALENT_CX, "OCT", screen=False)

    assert len(isomers) == 6
    assert all(iso.stereo_label.endswith(":M") for iso in isomers)
    for iso in isomers:
        text = rx.cxsmiles(iso)
        back = rx.metal(text)
        assert [item.stereo_label for item in back] == [iso.stereo_label]
        assert rx.cxsmiles(back[0]) == text
    with pytest.raises(ValueError, match="plain dative SMILES cannot retain atropisomer stereo"):
        S.dative_smiles(S.parse_smiles(_ATROP_RU_COVALENT_CX))


def test_cxsmiles_measures_a_marked_atrop_axis_from_3d():
    atrop = Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|")
    metal = Chem.MolFromSmiles("[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]")
    mol = Chem.AddHs(Chem.CombineMols(metal, atrop))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0

    before = next(iter(stereo.axis_stereo(rx.metal(rx.cxsmiles(mol))[0].stereo_label).values()))
    positions = mol.GetConformer().GetPositions()
    positions[:, 0] *= -1
    mol.GetConformer().SetPositions(positions)
    after = next(iter(stereo.axis_stereo(rx.metal(rx.cxsmiles(mol))[0].stereo_label).values()))

    assert after == {"M": "P", "P": "M"}[before]


def test_cxsmiles_perceives_unmarked_bound_binap_axis_from_3d():
    iso = rx.metal(_BINAP_PD, "SPL")[0]
    assert not stereo.axis_stereo(iso.stereo_label)

    geometry = rx.embed(iso, n=1, seed=7).minimize().mol
    text = rx.cxsmiles(geometry)
    back = rx.metal(text)
    perceived = rx.metal(geometry, "SPL")

    assert len(stereo.axis_stereo(back[0].stereo_label)) == 1
    assert len(stereo.axis_stereo(perceived[0].stereo_label)) == 1
    assert len(rx.embed(perceived[0], n=1, seed=7)) == 1
    with pytest.raises(ValueError, match="plain dative SMILES cannot retain atropisomer stereo"):
        rx.dative_smiles(geometry)


def test_coplanar_bound_biaryl_is_not_forced_to_have_an_atrop_hand():
    mol = S.parse_smiles(_BINAP_PD, remove_hs=False)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    point = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(point.x, point.y, 0.01))
    conf.Set3D(True)

    assert not stereo.axis_stereo(stereo.stereo_from_3d(mol, S.metal_indices(mol)))


def test_unsubstituted_aza_biaryl_is_not_an_atrop_axis():
    mol = S.parse_smiles(_AZA_BIARYL_CR)
    work, _caps = stereo._build_enumeration_graph(mol, set(S.metal_indices(mol)))

    assert not stereo._coordination_atrop_bonds(mol, set(S.metal_indices(mol)), work)


def test_write_dative_returns_written_atom_order():
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl"))
    smi, at = S.write_dative(mol)
    assert smi == S.dative_smiles(mol), "the two doors disagree about the string"
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1}
    assert set(at) == heavy, "routine hydrogens should be implicit and have no string position"
    assert sorted(at.values()) == list(range(len(at))), "the order is not a permutation of the written atoms"
    params = Chem.SmilesParserParams()
    params.removeHs = False
    back = Chem.MolFromSmiles(smi, params)
    written = [back.GetAtomWithIdx(at[a]).GetAtomicNum() for a in sorted(at)]
    assert written == [mol.GetAtomWithIdx(a).GetAtomicNum() for a in sorted(at)], "a position addresses another atom"


@pytest.mark.parametrize(
    "smiles",
    [r"F[C@H](Cl)/C=C/Br.N->[Pt+2](<-[H-])(<-[Cl-])<-[Cl-] |&1:1|", _ATROP_RU_COVALENT_CX],
)
def test_empty_writer_stereo_label_does_not_reperceive_or_keep_stale_tags(smiles, monkeypatch):
    mol = S.parse_smiles(smiles, remove_hs=False)
    rdDepictor.Compute2DCoords(mol)
    mol.GetConformer().Set3D(True)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    clean = Chem.Mol(mol)
    clean.RemoveAllConformers()
    Chem.RemoveStereochemistry(clean)
    for bond in clean.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    expected = S._write_dative(clean, "")
    native = S._write_native_stereo

    def no_inference(*_args, **_kwargs):
        pytest.fail("an authoritative empty label must not trigger 3D stereo inference")

    def write(graph, wanted, wanted_bonds):
        assert not wanted
        assert not wanted_bonds
        assert not graph.GetStereoGroups()
        assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in graph.GetAtoms())
        assert all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in graph.GetBonds())
        return native(graph, wanted, wanted_bonds)

    monkeypatch.setattr(S, "stereo_from_3d", no_inference)
    monkeypatch.setattr(S, "_write_native_stereo", write)

    assert S._write_dative(mol, "") == expected
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    if any(bond.GetStereo() in stereo._ATROP_STEREO for bond in mol.GetBonds()):
        with pytest.raises(ValueError, match="plain dative SMILES cannot retain atropisomer stereo"):
            S.write_dative(mol, "")
    else:
        assert mol.GetStereoGroups()
        assert S.write_dative(mol, "") == expected[:2]
        assert "[H-]" in expected[0]


def test_dative_writer_rebases_double_bond_stereo_after_metal_normalization():
    mol = S.parse_smiles(r"C/C=N(/C)->[Fe+2](<-[Cl-])<-[Cl-]")
    assert stereo.defined_stereo_label(mol, S.metal_indices(mol)) == "C1=N2:E"

    text, _at = S.write_dative(mol, "C1=N2:Z")

    back = S.parse_smiles(text)
    assert stereo.defined_stereo_label(back, S.metal_indices(back)) == "C1=N2:Z"


def test_canonical_slots_keep_identical_donor_links_with_their_assigned_slots():
    mol = S.parse_smiles(_BAXFIQ_DONOR_SLOTS)
    expected = {rx.cxsmiles(iso) for iso in rx.metal(mol, "SPL", screen=False)}

    assert len(expected) == 2
    for permutation in (list(reversed(range(mol.GetNumAtoms()))), list(range(0, mol.GetNumAtoms(), 2))):
        order = permutation + [index for index in range(mol.GetNumAtoms()) if index not in permutation]
        renumbered = Chem.RenumberAtoms(mol, order)
        assert {rx.cxsmiles(iso) for iso in rx.metal(renumbered, "SPL", screen=False)} == expected


@pytest.mark.parametrize("unbound_metal", [False, True])
def test_zero_order_contact_preserves_native_cip_and_bond_identity(unbound_metal):
    mol = S.parse_smiles("C[C@@H](CO)CO~N |Z:5|")
    if unbound_metal:
        mol = Chem.CombineMols(mol, S.parse_smiles("[Zn+2]"))
    before = Chem.MolToCXSmiles(mol)
    assert stereo.defined_stereo_label(mol) == "C1:S"

    text, at = S.write_dative(mol)
    back = S.parse_smiles(text)

    assert "Z:" in text
    assert back.GetBondBetweenAtoms(at[5], at[6]).GetBondType() == Chem.BondType.ZERO
    assert stereo.point_stereo(stereo.defined_stereo_label(back)) == {at[1]: "S"}
    assert S.dative_smiles(back) == text
    assert S.dative_smiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))) == text
    assert Chem.MolToCXSmiles(mol) == before
    if unbound_metal:
        assert rx.cxsmiles(mol) == text


@pytest.mark.parametrize(
    ("smiles", "geometry", "marker"),
    [
        (_ETA2_ASYM_E, "SPL", r",[ct]:"),
        (_ATROP_RU_COVALENT_CX, "OCT", r",w[UD]:"),
        ("C[N@](CC)(CCC)->[Pt+2](<-[Cl-])(<-[Br-])<-[I-]", "SPL", r",atomProp:"),
    ],
)
def test_zero_order_contacts_compose_with_metal_and_native_stereo_fields(smiles, geometry, marker):
    mol = Chem.CombineMols(S.parse_smiles(smiles), S.parse_smiles("C[C@@H](CO)CO~N |Z:5|"))
    iso = rx.metal(mol, geometry)[0]

    text = rx.cxsmiles(iso)
    back = rx.parse_smiles(text)

    assert text.count("|") == 2
    assert "Z:" in text
    assert "atomNote" in text
    assert re.search(marker, text)
    assert sum(bond.GetBondType() == Chem.BondType.ZERO for bond in back.GetBonds()) == 1
    assert rx.cxsmiles(rx.metal(back)[0]) == text
    assert rx.cxsmiles(Chem.RenumberAtoms(back, list(reversed(range(back.GetNumAtoms()))))) == text


def test_hydrogen_bond_cycle_does_not_make_chelate_imine_stereo_order_dependent():
    mol = S.parse_smiles("[N]1(->[Ni]2)/O[H]~O=[N+]->2=C/C=1 |Z:3|", remove_hs=False)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    text = S.dative_smiles(mol)

    assert S.dative_smiles(reversed_mol) == text
    back = S.parse_smiles(text, remove_hs=False)
    assert not stereo.bond_stereo(stereo.defined_stereo_label(back, S.metal_indices(back)))


def test_dative_writer_omits_coordination_locked_imine_stereo():
    mol = S.parse_smiles(r"C/C1=[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-[NH2]CC1")

    text = S.dative_smiles(mol)

    assert "/" not in text
    assert "\\" not in text
    back = S.parse_smiles(text)
    assert not stereo.bond_stereo(stereo.defined_stereo_label(back, S.metal_indices(back)))


def test_cx_cis_marker_may_encode_e_after_canonical_traversal():
    mol = S.parse_smiles("ClC(F)=C(Br)I |c:2|")

    assert stereo.defined_stereo_label(mol) == "C1=C3:E"


def test_cx_fields_losslessly_retain_conjugated_imine_ez():
    mol = S.parse_smiles(
        r"CC1=c2\cccc\c2=[N]2->[Ni]34<-[N](=C5\[CH-]C=CC=C5[C@H](C)\[N]->3="
        r"c3/cc(C)c(C)c/c3=[N]->4\1)/C(=O)C\2=O"
    )
    label = stereo.defined_stereo_label(mol, S.metal_indices(mol))
    core, at, bonds, unwritable = S._write_dative(mol, label)

    fields = S._cx_bond_stereo(core, label, at, bonds)
    text = f"{core} |{','.join(fields)}|"
    back = S.parse_smiles(text)

    assert unwritable
    assert fields
    expected = {frozenset(at[idx] for idx in pair): code for pair, code in stereo.bond_stereo(label).items()}
    assert stereo.bond_stereo(stereo.defined_stereo_label(back, S.metal_indices(back))) == expected
    flipped = label.replace("C7=N8:E", "C7=N8:Z")
    assert flipped != label
    assert S._cx_bond_stereo(core, flipped, at, bonds) != fields
    iso = I.Isomer(mol, "square_planar", [8, 10, 19, 28])
    iso.stereo_label = label
    cx = rx.cxsmiles(iso)
    assert rx.cxsmiles(rx.metal(cx)[0]) == cx


@pytest.mark.parametrize(
    ("smiles", "remove_hs"),
    [
        ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", True),
        ("CC=N([H])->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", False),
    ],
)
def test_dative_writer_retains_coordinated_imine_ez_and_maps_only_source_atoms(caplog, smiles, remove_hs):
    mol = S.parse_smiles(smiles, remove_hs=remove_hs)
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Pt")
    bond = mol.GetBondBetweenAtoms(1, 2)
    bond.SetStereoAtoms(0, metal)
    bond.SetStereo(Chem.BondStereo.STEREOE)

    with caplog.at_level(logging.WARNING):
        text, at = S.write_dative(mol, "C1=N2:E")

    back = S.parse_smiles(text, remove_hs=False)
    assert set(at) == set(range(mol.GetNumAtoms()))
    assert all(back.GetAtomWithIdx(at[i]).GetAtomicNum() == mol.GetAtomWithIdx(i).GetAtomicNum() for i in at)
    label = stereo.defined_stereo_label(back, S.metal_indices(back))
    assert set(stereo.bond_stereo(label).values()) == {"E"}
    assert "cannot read E/Z" not in caplog.text


def test_monodentate_imine_ez_remains_a_ligand_configuration():
    isomers = rx.metal("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", "square_planar")
    texts = {rx.cxsmiles(iso) for iso in isomers}

    assert isomers
    assert {iso.stereo_label for iso in isomers} == {"C1=N2:E", "C1=N2:Z"}
    assert len(texts) == len(isomers)
    assert all(rx.cxsmiles(rx.metal(text)[0]) == text for text in texts)


def test_hydrogen_reduction_conserves_a_haptic_carbanion_hydrogen():
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[H][c-]1(->[Fe+])cccc1", params)
    hydrogen = next(atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
    hydrogen.GetNeighbors()[0].SetNoImplicit(False)
    mol.UpdatePropertyCache(strict=False)

    text = S.dative_smiles(mol)

    assert "[cH-]" in text
    assert S.parse_smiles(text) is not None


def test_writer_hides_routine_h_and_keeps_hydride(capfd):
    cisplatin = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pt](<-[NH3])(Cl)Cl"))
    text = rx.cxsmiles(I.Isomer(cisplatin, "square_planar", [0, 2, 3, 4]))
    assert "[H]" not in text, text

    mol = S.parse_smiles(text)
    assert mol.GetNumAtoms() < cisplatin.GetNumAtoms(), "routine hydrogens were written explicitly"
    assert Chem.AddHs(mol).GetNumAtoms() == cisplatin.GetNumAtoms(), "the implicit hydrogen count changed"
    kept = K.stated_arrangement(mol)
    assert kept is not None, "the arrangement did not survive the parse at all"
    assert sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in kept[1].values()) == ["Cl", "Cl", "N", "N"]
    assert rx.cxsmiles(rx.enumerate_isomers(mol)[0]) == text, "the string is not a fixed point"
    expected = {rx.cxsmiles(i) for i in rx.enumerate_isomers(cisplatin, "square_planar")}
    reordered = Chem.RenumberAtoms(cisplatin, list(reversed(range(cisplatin.GetNumAtoms()))))
    assert {rx.cxsmiles(i) for i in rx.enumerate_isomers(reordered, "square_planar")} == expected

    params = Chem.SmilesParserParams()
    params.removeHs = False
    hydride = Chem.MolFromSmiles("[H-]->[Pt+2](Cl)(Cl)<-[NH3]", params)
    hydride_text = rx.cxsmiles(rx.enumerate_isomers(hydride, "square_planar")[0])
    assert "[H-]" in hydride_text, "the hydrogen donor lost the atom that carries its slot"
    assert rx.cxsmiles(rx.enumerate_isomers(rx.parse_smiles(hydride_text))[0]) == hydride_text
    assert "not removing hydrogen atom without neighbors" not in capfd.readouterr().err


def test_writer_keeps_the_nonmetal_leg_of_a_bridging_hydrogen():
    written = set()
    for edges in (((0, 1), (1, 2)), ((1, 2), (0, 1))):
        rw = Chem.RWMol()
        boron = Chem.Atom(5)
        boron.SetFormalCharge(-1)
        boron.SetNumExplicitHs(3)
        boron.SetNoImplicit(True)
        hydrogen = Chem.Atom(1)
        hydrogen.SetNoImplicit(True)
        iron = Chem.Atom(26)
        iron.SetFormalCharge(1)
        iron.SetNoImplicit(True)
        for atom in (boron, hydrogen, iron):
            rw.AddAtom(atom)
        for edge in edges:
            rw.AddBond(*edge, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        conf = Chem.Conformer(3)
        for atom, point in enumerate(((-1, 0, 0), (0, 0, 0), (1, 0, 0))):
            conf.SetAtomPosition(atom, Point3D(*point))
        mol.AddConformer(conf)
        written.add(S.dative_smiles(mol))

    assert written == {"[BH3-][H]->[Fe+]"}


def test_coordinate_free_stated_arrangement_is_a_cxsmiles_fixed_point():
    text = rx.cxsmiles(_isomer("[Pt](F)(F)(Cl)Cl", "square_planar", _MA2B2_SEATS["cis"]))
    parsed = S.parse_smiles(text)

    assert parsed.GetNumConformers() == 0
    assert rx.cxsmiles(parsed) == text


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_dative_smiles_roundtrips_nonstandard_complex():
    mol = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    assert any(a.GetAtomicNum() == 1 and a.GetDegree() > 1 for a in mol.GetAtoms()), (
        "this fixture must contain an over-connected hydrogen, or it does not test the repair"
    )

    def metals(m):  # SMILES renumbers, so the multiset is the claim, not the order
        return sorted(
            (m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(i).GetFormalCharge()) for i in _metal.metal_indices(m)
        )

    text = S.dative_smiles(mol)
    assert "[H][H]->" in text
    back = Chem.AddHs(Chem.MolFromSmiles(text))
    assert back.GetNumAtoms() == mol.GetNumAtoms()
    assert metals(back) == metals(mol) == [("Fe", 2), ("Mn", 1)]
    assert S.dative_smiles(S.parse_smiles(text)) == text
    assert S.dative_smiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))) == text


def test_dative_smiles_rejects_unreadable_graph():
    rw = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl")))
    c = rw.AddAtom(Chem.Atom(6))
    for _ in range(5):
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(c, h, Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="round-tripping SMILES"):
        S.dative_smiles(rw.GetMol())


def test_dative_writer_does_not_invent_point_stereo_on_a_degree_five_atom():
    mol = Chem.MolFromSmiles("F[Si](Cl)(Br)(I)->[Pd]", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        catchErrors=True,
    )
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0

    text = S.dative_smiles(mol)

    assert "@" not in text
    assert S.dative_smiles(S.parse_smiles(text)) == text


def test_macrocycle_ez_writer_has_a_canonical_fixed_point():
    text = (
        r"C1=CC=C2/C(=[N]3->[Hf+2]456(<-[O]=[C]->4(Cc4ccccc4)Cc4ccccc4)"
        r"<-[N](=C4/C=CC=CC=C4[N-]->5CCC\3)/CCCCC[N-]->62)C=C1"
    )

    once = S.dative_smiles(S.parse_smiles(text))

    assert once == text
    assert S.dative_smiles(S.parse_smiles(once)) == once


# --- canonicality: one species, one string, whatever Lewis form described it ------------------------------

# Each pair is one species written two ways, and the total charge is matched inside the pair on purpose: a
# covalent `[Pd]Cl` is Pd(II)Cl2 only against `[Cl-]->[Pd+2]`, and comparing it to `[Pd+]` would be comparing
# two different anions.
_LEWIS_PAIRS = {
    "halide": ("[NH3]->[Pt](<-[NH3])(Cl)Cl", "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]"),
    "neutral phosphine beside an anion": ("CP(C)(C)->[Rh]Cl", "CP(C)(C)->[Rh+]<-[Cl-]"),
    "amide": ("CN(C)[Pd](Cl)Cl", "C[N-](C)->[Pd+3](<-[Cl-])<-[Cl-]"),
    "alkyl": ("C[Pd](Cl)(Cl)C", "[CH3-]->[Pd+4](<-[Cl-])(<-[Cl-])<-[CH3-]"),
}


@pytest.mark.parametrize(("kind", "pair"), _LEWIS_PAIRS.items(), ids=_LEWIS_PAIRS)
def test_lewis_forms_share_species_string(kind, pair):
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in pair]
    assert len({Chem.GetFormalCharge(m) for m in mols}) == 1, f"{kind}: the pair is not one species, fix the fixture"
    written = {S.dative_smiles(m) for m in mols}
    assert len(written) == 1, f"{kind}: two Lewis forms of one species gave {len(written)} strings: {written}"


def test_terminal_oxo_is_ionic_and_pi_face_unchanged():
    lewis, ionic = "O=[V](Cl)(Cl)Cl", "[O-2]->[V+5](<-[Cl-])(<-[Cl-])<-[Cl-]"
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in (lewis, ionic)]
    assert len({sum(a.GetFormalCharge() for a in m.GetAtoms()) for m in mols}) == 1, "not the same total charge"
    before = [[a.GetFormalCharge() for a in m.GetAtoms()] for m in mols]
    written = [S.dative_smiles(m) for m in mols]
    assert written[0] == written[1], f"one species, two strings: {written}"
    assert "[O-2]" in written[0], f"the terminal oxo was not written ionically: {written[0]}"
    assert [[a.GetFormalCharge() for a in m.GetAtoms()] for m in mols] == before, "the writer mutated its input"

    fe = S.dative_smiles(Chem.AddHs(Chem.MolFromSmiles("[cH-]1cccc1.[cH-]1cccc1.[Fe+2]")))
    assert "[Fe+2]" in fe, f"the pi face was charged and the iron took the balance: {fe}"
    assert fe.count("[cH-]") == 2, f"a Cp carbon beyond the two anionic ones was charged: {fe}"


def test_coordinated_ring_resonance_has_one_nonmutating_dative_string(capfd):
    written = set()
    for face in ("[c-]1(F)c(Br)ccc1", "c1(F)c(Br)[cH-]cc1"):
        ligands = (Chem.MolFromSmiles(face), Chem.MolFromSmiles("[cH-]1cccc1"))
        rw = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(*ligands), Chem.MolFromSmiles("[Fe+2]")))
        metal = rw.GetNumAtoms() - 1
        for atom in list(rw.GetAtoms()):
            if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                rw.AddBond(atom.GetIdx(), metal, Chem.BondType.DATIVE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
        for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
            variant = Chem.RenumberAtoms(mol, order)
            before = [atom.GetFormalCharge() for atom in variant.GetAtoms()]
            written.add(S.dative_smiles(variant))
            assert [atom.GetFormalCharge() for atom in variant.GetAtoms()] == before

    assert len(written) == 1
    assert "Can't kekulize mol" not in capfd.readouterr().err


def test_multiply_charged_aromatic_macrocycle_has_an_idempotent_dative_string():
    text = (
        "FC1=C(F)c2c(-c3c(F)c(F)c(F)c(F)c3F)c3c(F)c(F)c4cc5c(F)c(F)"
        "c6c(-c7c(F)c(F)c(F)c(F)c7F)c7[n]8->[Zn+2](<-[n]2c1cc8C(F)=C7F)"
        "(<-[n-]43)(<-[n-]56)<-[O]1CCCC1"
    )
    molecule = S.parse_smiles(text)
    expected = S.dative_smiles(molecule)
    reordered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))

    assert S.dative_smiles(S.parse_smiles(expected)) == expected
    assert S.dative_smiles(reordered) == expected


def test_neutral_carbene_remains_neutral_in_dative_output():
    text = S.dative_smiles(S.parse_smiles("C[C](C)->[Rh+](<-[Br-])<-[C-]#[O+]"))

    assert "C[C](C)->[Rh+]" in text


def test_isomer_and_mol_write_same_constitution():
    for smi in (s for pair in _LEWIS_PAIRS.values() for s in pair):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        isos = rx.enumerate_isomers(mol)
        assert isos, f"{smi} enumerated nothing"
        core = rx.cxsmiles(isos[0]).split(" |", 1)[0]
        assert core == S.dative_smiles(mol), f"{smi}: the Isomer door wrote a different constitution"


# --- the arrangement the SMILES grammar cannot say -------------------------------------------------------


def test_cxsmiles_distinguishes_metal_hands():
    smi, flipped = "[Pt](F)(F)(Cl)(Cl)(Br)Br", [_CIS3[v] for v in (0, 1, 2, 3, 5, 4)]  # the Br pair exchanged
    hands = [_isomer(smi, "octahedral", seating) for seating in (_CIS3, flipped)]
    assert {i.chirality for i in hands} == {"delta", "lambda"}, [i.chirality for i in hands]
    one, other = (rx.cxsmiles(i) for i in hands)
    assert one.split("|")[0] == other.split("|")[0], "the constitution is the same molecule"
    assert one != other, "the two hands share a canonical string"


@pytest.mark.parametrize(
    ("smi", "geometry"),
    [
        ("[NH3]->[Pt](<-[NH3])(Cl)Cl", "square_planar"),  # cis / trans
        ("[Pt](F)(F)(Cl)(Cl)(Br)Br", "octahedral"),  # MA2B2C2: six isomers, one delta / lambda pair
        ("Br[Pd]1(Cl)NCCN1", "square_planar"),  # a chelate, so a bite edge is in the fold
        ("[Co]123(OCCN1)(OCCN2)OCCN3", "octahedral"),  # three identical unsymmetrical chelates
        ("[CH2]=[CH2].Cl[Pt](Cl)Cl", "square_planar"),  # Zeise: an eta2 face is one vertex
        ("[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])<-N", "seesaw"),
        ("[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])(<-N)<-O", "trigonal_bipyramidal"),
        ("[O+]#[C-]->[Co+3](<-[F-])(<-[Cl-])(<-[Br-])(<-N)<-O", "octahedral"),
        ("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "octahedral"),
    ],
    ids=[
        "MA2B2",
        "MA2B2C2",
        "chelate",
        "tris-chelate",
        "eta2",
        "SEE-all-distinct",
        "TBP-all-distinct",
        "OH-all-distinct",
        "OH-all-distinct-PH3",
    ],
)
def test_all_enumerated_isomers_read_back(smi, geometry):
    isos = rx.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles(smi)), geometry)
    assert len(isos) >= 1
    for iso in isos:
        text = rx.cxsmiles(iso)
        back = rx.enumerate_isomers(S.parse_smiles(text))
        assert len(back) == 1, f"{iso.label}: its own string enumerated {len(back)} isomers"
        got = back[0]
        assert rx.cxsmiles(got) == text, f"{iso.label}: the string is not a fixed point"
        assert got.geometry == iso.geometry, f"{iso.label}: came back as {got.geometry}"
        assert got.chirality == iso.chirality, f"{iso.label}: {iso.chirality!r} came back {got.chirality!r}"
        was, now = _seated(iso), _seated(got)
        dirs = vertex_dirs(iso.geometry)
        rotations = None if dirs is None else point_group(tuple(map(tuple, dirs)))[0]
        assert rotations is not None, f"{iso.label}: no vertex-direction template"
        assert any([was[q[v]] for v in range(len(was))] == now for q in rotations), (
            f"{iso.label}: {was} and {now} are not the same arrangement under any rotation of the template"
        )


def test_cxsmiles_on_a_pruned_bridgehead_graph_does_not_crash():
    """A raw conformer-bearing Mol with a Class A bridgehead bond must not crash `cxsmiles`.

    Regression for a KeyError keyed on the pruned atom's own index: `cxsmiles`'s `bound` dict, read
    off the caller's Mol before it is canonicalized, named the bridgehead as a metal neighbour, while
    `centre_notes` (built from a canonicalized `Isomer` via `from_geometry`) no longer had an entry for
    it. `complexed` must be canonicalized before either is derived; see `metal_core._canonical_metal_graph`.
    """
    rw = Chem.RWMol()

    def add(symbol, charge=0):
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        return rw.AddAtom(atom)

    ni, p, c1, c2, sd, ss = add("Ni", 2), add("P"), add("C"), add("C"), add("S"), add("S", -1)
    for carbon in (c1, c2):
        rw.GetAtomWithIdx(carbon).SetNumExplicitHs(3)
    rw.AddBond(p, c1, Chem.BondType.SINGLE)
    rw.AddBond(p, c2, Chem.BondType.SINGLE)
    rw.AddBond(p, sd, Chem.BondType.DOUBLE)
    rw.AddBond(p, ss, Chem.BondType.SINGLE)
    rw.AddBond(sd, ni, Chem.BondType.DATIVE)
    rw.AddBond(ss, ni, Chem.BondType.DATIVE)
    rw.AddBond(p, ni, Chem.BondType.DATIVE)  # the wrong bridgehead bond: kappa2 dithiophosphinate P

    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in {
        ni: (0.0, 0.0, 0.0),
        p: (0.0, 0.0, 2.2),
        c1: (1.4, 0.0, 2.8),
        c2: (-1.4, 0.0, 2.8),
        sd: (1.1, 1.9, -0.6),
        ss: (-1.1, -1.9, -0.6),
    }.items():
        conf.SetAtomPosition(idx, Point3D(*xyz))
    mol.AddConformer(conf, assignId=True)

    text = rx.cxsmiles(mol)  # crashed with KeyError(p) before the fix; must not raise

    back = rx.metal(text)[0]
    donors = sorted(back.mol.GetAtomWithIdx(d).GetSymbol() for d, _m in back.donor_bonds)
    assert donors == ["S", "S"]


def test_bis_silyl_cxsmiles_keeps_distinct_trans_pairs():
    baxfiq = "CC(C)(C)[N+]#[C-]->[Pt+2](<-[SiH-](c1ccccc1)c1ccccc1)(<-[SiH-](c1ccccc1)c1ccccc1)<-[P](C)(C)C"
    isomers = rx.metal(baxfiq, "square_planar", stereo="free")
    texts = {iso.label: rx.cxsmiles(iso) for iso in isomers}
    source = S.parse_smiles(baxfiq)
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    reordered = {iso.label: rx.cxsmiles(iso) for iso in rx.metal(reversed_source, "square_planar", stereo="free")}

    assert set(texts) == {"cis", "trans"}
    assert texts["cis"] != texts["trans"]
    assert all(rx.metal(text)[0].label == label for label, text in texts.items())
    assert reordered == texts


def test_tetradentate_tetrahedron_keeps_mirror_arrangements():
    base = Chem.MolFromSmiles("SCCCCN=CC=NCCCCS")
    rw = Chem.RWMol(base)
    metal = rw.AddAtom(Chem.Atom("Cu"))
    donors = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetSymbol() in {"N", "S"}]
    for donor in donors:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    source = rw.GetMol()
    source.UpdatePropertyCache(strict=False)

    isomers = rx.metal(source, "tetrahedral", stereo="free")
    texts = {rx.cxsmiles(iso) for iso in isomers}
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))

    assert len(isomers) == len(texts) == 2
    assert {iso.chirality for iso in isomers} == {"delta", "lambda"}
    assert {rx.cxsmiles(iso) for iso in rx.metal(reversed_source, "tetrahedral", stereo="free")} == texts
    assert all(rx.cxsmiles(rx.metal(text)[0]) == text for text in texts)


def test_small_metal_closed_imine_uses_the_arrangement_instead_of_ez():
    diimine = r"C/N1=C(/F)C(/Cl)=N(/Br)->[Ni+2](<-[Cl-])(<-[I-])<-1"
    text = rx.cxsmiles(rx.metal(diimine, "square_planar")[0])

    assert not rx.metal(text)[0].stereo_label
    assert rx.cxsmiles(rx.metal(text)[0]) == text
    assert rx.cxsmiles(rx.embed(rx.metal(text)[0], n=1, seed=7).mol) == text


def test_each_chiral_octahedral_key_is_atom_order_invariant():
    source = Chem.MolFromSmiles("[O+]#[C-]->[Co+3](<-[F-])(<-[Cl-])(<-[Br-])(<-N)<-O")
    order = list(reversed(range(source.GetNumAtoms())))  # new index -> old index
    old_to_new = {old: new for new, old in enumerate(order)}
    renumbered = Chem.RenumberAtoms(source, order)
    for iso in rx.metal(source, "octahedral", stereo="free"):
        remapped = rx.Isomer(renumbered, iso.geometry, [old_to_new[donor] for donor in iso.vertices])
        assert rx.cxsmiles(remapped) == rx.cxsmiles(iso)


def test_stated_arrangement_rejects_wrong_chirality():
    chiral = next(
        i
        for i in rx.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("[Pt](F)(F)(Cl)(Cl)(Br)Br")), "octahedral")
        if i.chirality == "delta"
    )
    text = rx.cxsmiles(chiral)
    with pytest.raises(ValueError, match="omits chirality"):
        rx.enumerate_isomers(S.parse_smiles(text.replace("-delta", "")))
    with pytest.raises(ValueError, match="seating is delta"):
        rx.enumerate_isomers(S.parse_smiles(text.replace("-delta", "-lambda")))

    square = rx.cxsmiles(_isomer("[Pt](F)(F)(Cl)Cl", "square_planar", _MA2B2_SEATS["cis"]))
    with pytest.raises(ValueError, match="planar"):
        rx.enumerate_isomers(S.parse_smiles(square.replace(".SPL", ".SPL-delta")))
    with pytest.raises(ValueError, match="not a haptic face"):
        rx.enumerate_isomers(S.parse_smiles(square.replace(".s0", ".s0+", 1)))

    trans = next(
        i
        for i in rx.enumerate_isomers(
            Chem.AddHs(Chem.MolFromSmiles("Cl[Co]12(Cl)(NCCN1)NCCN2")), "octahedral", stereo="free"
        )
        if not i.chirality
    )
    forged = rx.cxsmiles(trans).replace(".OCT:", ".OCT-delta:")
    with pytest.raises(ValueError, match="seating is achiral"):
        rx.enumerate_isomers(S.parse_smiles(forged), stereo="free")


def test_planar_chiral_ferrocene_winding_roundtrips_and_selects_after_dg():
    iso = rx.metal(_planar_chiral_ferrocene(), stereo="free")[0]
    raw = core_embed(iso, n=8, seed=7, prune_rms=-1)
    by_sign = {next(iter(_haptic_windings(raw.mol, iso, cid).values())): cid for cid in raw.ids}
    assert set(by_sign) == {"+", "-"}, "the ungated DG control did not sample both windings"

    texts = {}
    for sign, cid in by_sign.items():
        realised = Chem.Mol(raw.mol, False, int(cid))
        text = rx.cxsmiles(realised)
        retained = I.from_geometry(realised)
        assert set(retained.haptic_winding.values()) == {sign}
        assert set(rx.metal(realised, stereo="free")[0].haptic_winding.values()) == {sign}
        retained_embed = core_embed(retained, n=4, seed=17, prune_rms=-1)
        assert {
            next(iter(_haptic_windings(retained_embed.mol, retained, retained_cid).values()))
            for retained_cid in retained_embed.ids
        } == {sign}
        back = rx.enumerate_isomers(Chem.AddHs(S.parse_smiles(text)), stereo="free")[0]
        assert set(back.haptic_winding.values()) == {sign}
        assert rx.cxsmiles(back) == text
        default_back = rx.enumerate_isomers(S.parse_smiles(text))
        assert len(default_back) == 1
        assert not default_back[0].stereo_label
        assert rx.cxsmiles(Chem.RenumberAtoms(realised, list(reversed(range(realised.GetNumAtoms()))))) == text

        embedded = core_embed(back, n=6, seed=11, prune_rms=-1).minimize()
        assert len(embedded) == 6
        assert all(
            geom.check(embedded.mol, embedded_cid, donors=back.donors, constraints=embedded.cons).ok()
            for embedded_cid in embedded.ids
        )
        assert {
            next(iter(_haptic_windings(embedded.mol, back, embedded_cid).values())) for embedded_cid in embedded.ids
        } == {sign}
        texts[sign] = text

    assert texts["+"].split(" |", 1)[0] == texts["-"].split(" |", 1)[0]
    assert texts["+"] != texts["-"]
    forged = S.parse_smiles(texts["+"])
    unsigned = next(
        atom
        for atom in forged.GetAtoms()
        if atom.HasProp("atomNote")
        and atom.GetProp("atomNote").startswith("s")
        and atom.GetProp("atomNote")[-1].isdigit()
    )
    unsigned.SetProp("atomNote", unsigned.GetProp("atomNote") + "+")
    with pytest.raises(ValueError, match="mirror-symmetric face"):
        rx.enumerate_isomers(forged, stereo="free")


def test_stated_haptic_winding_survives_ligand_stereo_enumeration():
    fixed = rx.cxsmiles(rx.metal(_planar_chiral_ferrocene(point_stereo=True))[0]).replace("@", "")
    back = rx.enumerate_isomers(S.parse_smiles(fixed))
    assert {iso.stereo_label for iso in back} == {"C1:R", "C1:S"}
    assert {tuple(iso.haptic_winding.values()) for iso in back} == {("+",)}


def test_carbanion_stereo_stays_in_the_smiles_core():
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
    isomers = rx.metal(smi, "square_planar")
    texts = {iso.stereo_label: rx.cxsmiles(iso) for iso in isomers}

    assert set(texts) == {"C23:R", "C23:S"}
    # Whole-shell reach excludes the doubly trans seating, not either carbanion hand.
    assert len({rx.cxsmiles(iso) for iso in isomers}) == len(isomers) == 2
    assert {iso.label for iso in isomers} == {"cis"}
    assert len(set(texts.values())) == 2
    assert all("[C@" in text for text in texts.values())
    assert all("rxStereo" not in text for text in texts.values())
    assert {rx.metal(text)[0].stereo_label for text in texts.values()} == set(texts)
    for text in texts.values():
        core = text.split(" |", 1)[0]
        mol = S.parse_smiles(text)
        assert S.dative_smiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))) == core

    # The fallback writer must not leak an unrelated, invalid tag through cleanStereo=False.
    forged = S.parse_smiles(texts["C23:S"])
    methyl = next(a for a in forged.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 1)
    methyl.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    params = Chem.SmilesWriteParams()
    params.cleanStereo = False
    assert "H3]" in Chem.MolToSmiles(forged, params)
    assert S.dative_smiles(forged) == texts["C23:S"].split(" |", 1)[0]

    target = isomers.select(stereo="R", label="cis")
    expected = rx.cxsmiles(target)
    realised = rx.embed(target, n=1, seed=7).minimize().mol
    donor = realised.GetAtomWithIdx(23)
    donor.SetChiralTag(
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        if donor.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
        else Chem.ChiralType.CHI_TETRAHEDRAL_CW
    )
    assert rx.cxsmiles(realised) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "[Cu+]1<-[N@H](C)CCN->1",
        "[O+]#[C-]->[Cu+]12<-[N@@H](C)[C@@H]3C[C@H]([N@H]->1C)C[C@H]([N@H]->2C)C3",
    ],
    ids=["secondary-amine", "coupled-triamine"],
)
def test_coordinated_point_stereo_has_a_canonical_atom_and_bond_basis(smiles):
    mol = S.parse_smiles(smiles)
    params = Chem.SmilesWriteParams()
    params.cleanStereo = False
    flags = Chem.CXSmilesFields.CX_COORDINATE_BONDS
    text = S.dative_smiles(mol)
    rng = np.random.default_rng(42)

    for _ in range(8):
        mol = Chem.RenumberAtoms(mol, rng.permutation(mol.GetNumAtoms()).tolist())
        native = Chem.MolToCXSmiles(mol, params, flags)
        written, at = S.write_dative(mol)
        assert written == text
        assert Chem.MolToCXSmiles(mol, params, flags) == native, "writer must not mutate its input"
        back = S.parse_smiles(written)
        source, target = Chem.Mol(mol), Chem.Mol(back)
        for view in (source, target):
            for bond in view.GetBonds():
                if bond.GetBondType() == Chem.BondType.DATIVE:
                    bond.SetBondType(Chem.BondType.SINGLE)  # include all carriers in RDKit's chiral match
            view.UpdatePropertyCache(strict=False)
        for old, new in at.items():
            source.GetAtomWithIdx(old).SetIsotope(1000 + old)
            target.GetAtomWithIdx(new).SetIsotope(1000 + old)
        assert target.HasSubstructMatch(source, useChirality=True), "retain the mapped stereo, not just hand counts"
        mol = back

    altered = Chem.Mol(mol)
    donor = next(a for a in altered.GetAtoms() if a.GetSymbol() == "N" and a.GetChiralTag())
    donor.InvertChirality()
    assert S.dative_smiles(altered) != text


def test_raw_donor_point_override_composes_with_an_independent_ez_override():
    mol = S.parse_smiles("C/C=C/C[N@H](C)->[Cu+]<-[Cl-]")
    expected = Chem.Mol(mol)
    donor = expected.GetAtomWithIdx(4)
    donor.InvertChirality()
    raw = "CW" if donor.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "CCW"

    text, at = S.write_dative(mol, f"N4:{raw},C1=C2:Z")
    back = S.parse_smiles(text)

    label = stereo.defined_stereo_label(back, S.metal_indices(back))
    assert stereo.bond_stereo(label) == {frozenset((at[1], at[2])): "Z"}
    for view in (expected, back):
        for bond in view.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
        view.UpdatePropertyCache(strict=False)
    assert back.HasSubstructMatch(expected, useChirality=True)


def test_non_cip_ring_fusion_tags_stay_in_the_smiles_core():
    mol = S.parse_smiles(_NON_CIP_CAGE_AU)
    before = stereo.point_stereo(stereo.defined_stereo_label(mol, S.metal_indices(mol)))
    assert sorted(before.values()) == ["CCW", "s", "s", "s"], "fixture premise"

    text = S.dative_smiles(mol)
    back = S.parse_smiles(text)

    assert S.dative_smiles(back) == text
    assert len(stereo.point_stereo(stereo.defined_stereo_label(back, S.metal_indices(back)))) == 4


def test_pseudoasymmetric_bis_eta2_cage_is_atom_order_invariant():
    source = S.parse_smiles(_PSEUDO_BIS_ETA2_RU)
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))

    expected = rx.metal(source, "OCT")
    reordered = rx.metal(reversed_source, "OCT")

    # The trans bis-η² seating is outside the native ligand reach and is screened before embedding.
    assert len(expected) == len(reordered) == 2
    assert {rx.cxsmiles(iso) for iso in expected} == {rx.cxsmiles(iso) for iso in reordered}
    assert {iso.label for iso in expected} == {"fac", "mer"}
    assert {iso.stereo_label.split(":")[-1] for iso in expected} == {"s"}
    assert {rx.cxsmiles(iso) for iso in reordered} == {rx.cxsmiles(iso) for iso in expected}


def test_tagged_chiral_amine_donor_stays_in_the_smiles_core():
    hands = {}
    for tag in ("@", "@@"):
        isomers = rx.metal(f"[Pd](Cl)(Cl)(Cl)([N{tag}H](C)O)", "square_planar")
        assert len(isomers) == 1
        hands[isomers[0].stereo_label] = rx.cxsmiles(isomers[0])

    assert set(hands) == {"N4:R", "N4:S"}
    assert all("[N@" in text for text in hands.values())
    assert all("rxStereo" not in text for text in hands.values())
    assert {rx.metal(text)[0].stereo_label.rsplit(":", 1)[1] for text in hands.values()} == {"R", "S"}
    assert all(rx.cxsmiles(rx.metal(text)[0]) == text for text in hands.values())

    for text in hands.values():
        realised = rx.embed(rx.metal(text)[0], n=1, seed=2).minimize().mol
        assert rx.cxsmiles(realised) == text


def test_coupled_chelated_amine_hands_write_together():
    text = "CC1(C)C[N@@H]2[C@@H]3CCCC[C@@H]3[N@@H]3CC(C)(C)[S-]->[Ni+2]<-2<-3<-[S-]1"

    written = rx.dative_smiles(rx.parse_smiles(text, remove_hs=False))
    back = rx.parse_smiles(written)

    assert rx.dative_smiles(back) == written
    assert len(stereo.point_stereo(stereo.defined_stereo_label(back, S.metal_indices(back)))) == 4


def test_coupled_amine_hands_are_written_after_canonical_ring_closure_rebasing():
    base = S.parse_smiles(
        "O=C1C[N]2(CC[N]34CC(=O)[O-]->[Ti+5]<-2<-3(<-[O-2])(<-[O-]1)<-[O-]c1ccccc1C4)Cc1ccccc1O",
        remove_hs=False,
    )
    rw = Chem.RWMol()
    for atom in base.GetAtoms():
        rw.AddAtom(Chem.Atom(atom))
    bonds = sorted(
        base.GetBonds(),
        key=lambda bond: (max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), bond.GetBeginAtomIdx()),
        reverse=True,
    )
    for bond in bonds:
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    centres = sorted(stereo.point_centres(mol, S.metal_indices(mol)))
    written = set()
    for left in "RS":
        for right in "RS":
            configured = Chem.Mol(mol)
            stereo.apply_point_stereo(
                configured,
                f"N{centres[0]}:{left},N{centres[1]}:{right}",
                centres,
            )
            text = S.dative_smiles(configured)
            back = S.parse_smiles(text, remove_hs=False)
            assert sorted(
                stereo.point_stereo(stereo.defined_stereo_label(back, S.metal_indices(back))).values()
            ) == sorted((left, right))
            assert (
                S.dative_smiles(Chem.RenumberAtoms(configured, list(reversed(range(configured.GetNumAtoms()))))) == text
            )
            assert S.dative_smiles(back) == text
            written.add(text)
    assert len(written) == 4


def test_embedded_phosphorus_stays_in_the_smiles_core():
    isomer = rx.metal("[Pd](Cl)(Cl)(Cl)([P@H](C)O)", "square_planar")[0]
    text = rx.cxsmiles(rx.embed(isomer, n=1, seed=7).minimize().mol)
    assert "[P@" in text
    assert "rxStereo" not in text
    assert rx.metal(text)[0].stereo_label.endswith(":R")


def test_invalid_phosphorus_tag_is_cleaned_without_losing_its_hydrogen():
    text = S.dative_smiles(S.parse_smiles("[Pd](Cl)(Cl)(Cl)([P@H](C)C)"))
    assert "[PH]" in text
    assert "@" not in text


def test_symmetric_donor_bridging_two_metals_is_not_called_chiral():
    text = S.dative_smiles(S.parse_smiles("C[N](C)(->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br"))
    assert "@" not in text


def test_tagged_amine_between_two_metals_survives_dative_normalization():
    mol = S.parse_smiles("C[N@H](->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br")

    text = S.dative_smiles(mol)

    assert "[N@" in text
    assert stereo.defined_stereo_label(S.parse_smiles(text), S.metal_indices(mol)) == "N1:R"


def test_coordinate_free_haptic_winding_enumerates_both_hands():
    isomers = rx.metal(_planar_chiral_ferrocene())
    assert len(isomers) == 2
    assert {next(iter(iso.haptic_winding.values())) for iso in isomers} == {"+", "-"}
    assert {"η5Rₚ", "η5Sₚ"} <= {part[:4] for iso in isomers for part in I.arrangement(iso).split()}
    by_sign = {next(iter(iso.haptic_winding.values())): I.arrangement(iso) for iso in isomers}
    assert "η5Rₚ" in by_sign["+"]
    assert "η5Sₚ" in by_sign["-"]
    assert all("Rₚ" not in rx.cxsmiles(iso) and "Sₚ" not in rx.cxsmiles(iso) for iso in isomers)
    for iso in isomers:
        embedded = core_embed(iso, n=4, seed=7, prune_rms=-1)
        assert {next(iter(_haptic_windings(embedded.mol, iso, cid).values())) for cid in embedded.ids} == set(
            iso.haptic_winding.values()
        )


@pytest.mark.parametrize(
    ("smiles", "count", "names"),
    [
        (r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 6, {"(re,re)", "(si,si)"}),
        (_ETA2_ASYM_E, 6, {"(re,si)", "(si,re)"}),
        (r"C/[CH]1=[CH](\C)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 3, set()),
        (r"C/[CH]1=[CH](/C)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 6, {"(re,re)", "(si,si)"}),
        (r"[CH2]1=[CH](C)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 6, {"(re)", "(si)"}),
        (r"[CH2]1=[CH2]->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 3, set()),
        ("O1C[CH]2=[CH](CC1)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-2", 6, {"(re,re)", "(si,si)"}),
    ],
    ids=["asymmetric-Z", "asymmetric-E", "symmetric-Z", "symmetric-E", "propene", "ethene", "small-ring"],
)
def test_eta2_face_orientation_matrix(smiles, count, names):
    isomers = rx.metal(smiles, "SPL")
    shown = {name for name in names if any(name in I.arrangement(iso) for iso in isomers)}
    assert len(isomers) == count
    assert shown == names
    assert {tuple(iso.haptic_winding.values()) for iso in isomers} == ({("+",), ("-",)} if names else {()})


def test_eta2_face_is_selected_after_dg_and_reflection_flips_it():
    free = rx.metal(_ETA2_ASYM_E, "SPL", stereo="free")[0]
    raw = core_embed(free, n=16, seed=7, prune_rms=-1)
    by_sign = {next(iter(_haptic_windings(raw.mol, free, cid).values())): cid for cid in raw.ids}
    assert set(by_sign) == {"+", "-"}, "the ungated DG control did not sample both alkene faces"

    cid = by_sign["+"]
    conf = raw._mol.GetConformer(cid)
    positions = conf.GetPositions()
    positions[:, 0] *= -1
    for atom, xyz in enumerate(positions):
        conf.SetAtomPosition(atom, xyz.tolist())
    assert set(_haptic_windings(raw.mol, free, cid).values()) == {"-"}

    for iso in rx.metal(_ETA2_ASYM_E, "SPL")[:2]:
        embedded = core_embed(iso, n=4, seed=7, prune_rms=-1)
        assert {next(iter(_haptic_windings(embedded.mol, iso, i).values())) for i in embedded.ids} == set(
            iso.haptic_winding.values()
        )


def test_eta2_face_and_coordinated_amine_stereo_compose_in_one_dg_seed():
    smiles = r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[N@H](C)O)<-1"
    for iso in rx.metal(smiles, "SPL")[:2]:  # one seating, its two eta2 faces
        embedded = core_embed(iso, n=2, seed=7, prune_rms=-1)
        assert {next(iter(_haptic_windings(embedded.mol, iso, cid).values())) for cid in embedded.ids} == set(
            iso.haptic_winding.values()
        )
        assert {
            stereo.stereo_from_3d(Chem.Mol(embedded.mol, False, int(cid)), exclude={iso.metal}) for cid in embedded.ids
        } == {iso.stereo_label}


def test_eta2_face_enumerates_from_a_mol_with_implicit_hydrogens():
    isomers = rx.metal(rx.parse_smiles(_ETA2_ASYM_E), "SPL")
    assert len(isomers) == 6
    assert {tuple(iso.haptic_winding.values()) for iso in isomers} == {("+",), ("-",)}


def test_eta2_stereoany_does_not_imply_a_face_relation():
    iso = rx.metal(_ETA2_ASYM_E, "SPL")[0]
    face = next(iter(iso.haptic.values()))
    iso.mol.GetBondBetweenAtoms(*face).SetStereo(Chem.BondStereo.STEREOANY)
    assert _metal_stereo.eta2_signatures(iso.mol, face) == ((), ())


def test_dative_cx_option_retains_eta2_ez_without_arrangement_notes():
    text = rx.dative_smiles(rx.parse_smiles(_ETA2_ASYM_E), cx=True)

    assert "|t:" in text
    assert "atomNote" not in text
    parsed = rx.parse_smiles(text)
    assert {iso.stereo_label for iso in rx.metal(parsed, "SPL")} == {"C1=C2:E"}


@pytest.mark.parametrize(
    ("smiles", "label"),
    [
        (_ETA2_ASYM_E, "C1=C2:E"),
        (r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "C1=C2:Z"),
    ],
)
def test_eta2_face_and_ez_are_one_cxsmiles_fixed_point(smiles, label, caplog):
    isomers = rx.metal(smiles, "SPL")[:2]
    texts = {next(iter(iso.haptic_winding.values())): rx.cxsmiles(iso) for iso in isomers}
    assert len(set(texts.values())) == 2
    assert all(",c:" in text or ",t:" in text for text in texts.values())
    for sign, text in texts.items():
        parsed = rx.parse_smiles(text)
        bond = next(bond for bond in parsed.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
        assert len(set(bond.GetStereoAtoms())) == 2
        assert rx.cxsmiles(parsed) == text
        back = rx.metal(text)
        assert len(back) == 1
        assert back[0].stereo_label == label
        assert set(back[0].haptic_winding.values()) == {sign}
        assert rx.cxsmiles(back[0]) == text
        embedded = rx.embed(back[0], n=1, seed=7)
        order = list(reversed(range(embedded.mol.GetNumAtoms())))
        assert rx.cxsmiles(Chem.RenumberAtoms(embedded.mol, order)) == text
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        plain = rx.dative_smiles(rx.embed(isomers[0], n=1, seed=7).mol)
    assert "/" not in plain
    assert "\\" not in plain
    assert "cannot read E/Z" in caplog.text
    assert {iso.stereo_label for iso in rx.metal(plain, "SPL")} == {"C1=C2:E", "C1=C2:Z"}


def test_cxsmiles_round_trips_two_eta2_bonds_and_rejects_invalid_fields():
    isomers = rx.metal(_TWO_ETA2, "SPL")
    assert len(isomers) == 24  # 3 SPL seatings x 2 and 4 configurations of the non-equivalent eta2 faces
    assert len({I.arrangement(iso) for iso in isomers}) == len(isomers)
    two_faces = next(iso for iso in isomers if iso.stereo_label.count(":E") == 2)
    text = rx.cxsmiles(two_faces)
    back = rx.metal(text)
    assert len(back) == 1
    assert back[0].stereo_label.count(":E") == 2
    assert rx.cxsmiles(back[0]) == text
    embedded = rx.embed(two_faces, n=1, seed=4)
    assert rx.cxsmiles(Chem.RenumberAtoms(embedded.mol, list(reversed(range(embedded.mol.GetNumAtoms()))))) == text

    native = rx.cxsmiles(rx.metal(_ETA2_ASYM_E, "SPL")[0])
    valid = native
    assert rx.metal(valid)[0].stereo_label == "C1=C2:E"
    for forged in (
        valid.replace("t:1", "t:999"),
        valid.replace("t:1", "t:0"),
        valid.replace("t:1", "t:"),
        valid.replace("t:1", "t:1,c:1"),
        valid.removesuffix("|") + ",t:1|",
    ):
        with pytest.raises(ValueError, match="CX"):
            rx.parse_smiles(forged)


def test_eta2_ez_and_chiral_substituent_are_one_cxsmiles_fixed_point():
    smiles = r"F[C@H](Cl)/[CH]1=[CH](/Br)->[Pt+2](<-[Cl-])(<-[I-])(<-[NH3])<-1"
    iso = rx.metal(smiles, "SPL")[0]
    text = rx.cxsmiles(iso)
    back = rx.metal(text)
    assert len(back) == 1
    assert sorted(part.rsplit(":", 1)[-1] for part in back[0].stereo_label.split(",")) == ["E", "R"]
    assert rx.cxsmiles(back[0]) == text
    embedded = rx.embed(iso, n=1, seed=7)
    order = list(reversed(range(embedded.mol.GetNumAtoms())))
    assert rx.cxsmiles(Chem.RenumberAtoms(embedded.mol, order)) == text


def test_haptic_winding_enumeration_composes_with_ligand_stereo():
    isomers = rx.metal(_planar_chiral_ferrocene(point_stereo=True))
    assert {(iso.stereo_label, next(iter(iso.haptic_winding.values()))) for iso in isomers} == {
        ("C5:R", "+"),
        ("C5:R", "-"),
        ("C5:S", "+"),
        ("C5:S", "-"),
    }


def test_identical_haptic_faces_canonicalize_opposite_windings():
    source = _planar_chiral_ferrocene(two_chiral_faces=True)
    isomers = rx.metal(source)
    assert len(isomers) == 3  # RₚRₚ, RₚSₚ, SₚSₚ; swapping identical faces removes SₚRₚ
    assert sorted(i.haptic_configuration for i in isomers) == ["meso", "rac", "rac"]
    assert isomers.select(haptic="meso").haptic_configuration == "meso"
    with pytest.raises(ValueError, match="2 orientable faces"):
        isomers.filter(haptic="Rp")
    assert len(isomers.filter(haptic={0: "Rp"})) == 2
    assert len(isomers.filter(haptic={0: "Rp", 7: "Sp"})) == 1
    iso = rx.metal(source, stereo="free")[0]
    raw = core_embed(iso, n=32, seed=7, prune_rms=-1)
    by_winding = {tuple(_haptic_windings(raw.mol, iso, cid).values()): cid for cid in raw.ids}
    assert {("+", "-"), ("-", "+")} <= set(by_winding)
    strings = {rx.cxsmiles(Chem.Mol(raw.mol, False, int(by_winding[winding]))) for winding in (("+", "-"), ("-", "+"))}
    assert len(strings) == 1
    engine = importlib.import_module("rxembed.embed")
    target = next(iso for iso in isomers if set(iso.haptic_winding.values()) == {"+", "-"})
    targets = engine._stereo_targets(target)
    ranks, eta2 = engine._winding_ranks(raw.mol, targets)
    assert all(
        engine._seed_stereo_matches(raw.mol, by_winding[winding], target, targets, ranks, eta2, False)
        for winding in (("+", "-"), ("-", "+"))
    )


def test_stated_arrangement_rejects_shape_override_and_composes_fix():
    text = rx.cxsmiles(_isomer("[Pt](F)(F)(F)(Cl)(Cl)Cl", "octahedral", _MA3B3_SEATS["fac"]))
    mol = S.parse_smiles(text)
    assert len(rx.enumerate_isomers(mol, "OCT")) == 1, "naming the shape the string states is not a contradiction"
    with pytest.raises(ValueError, match="nothing to act on"):
        rx.enumerate_isomers(mol, geometry="trigonal_prismatic")
    with pytest.raises(ValueError, match="source has no geometry"):
        rx.enumerate_isomers(mol, fix=[1, 2])

    fixed = rx.enumerate_isomers(mol, fix={(1, 2): 2.0})
    assert fixed[0].cons.fixed[(1, 2)] == (2.0, 2.0)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_multimetal_cxsmiles_roundtrips_and_gates_every_sphere_after_dg():
    source = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    direct = rx.metal(source, "OCT", center="Mn", fix=[1, 5, 63, 64, 65, 66], stereo="free").filter(label="mer")[0]
    text = rx.cxsmiles(direct)
    iso = rx.metal(text, center="Mn", stereo="free")[0]

    assert len(direct.centres) == len(iso.centres) == 2
    assert rx.cxsmiles(iso) == text
    assert len(rx.metal(rx.cxsmiles(source), center="Mn", stereo="free")[0].centres) == 2
    raw = core_embed(direct, n=4, seed=7, prune_rms=-1)
    expected_fe = I.centre_states(direct, "Fe")[0]
    expected_fe_winding = materialized_state(direct, expected_fe)[2]
    assert {
        tuple(I.from_geometry(Chem.Mol(raw.mol, False, int(cid)), center="Fe").haptic_winding.values())
        for cid in raw.ids
    } == {tuple(expected_fe_winding.values())}
    assert {
        _metal_stereo.realised_chirality(
            raw.mol, cid, direct.geometry, direct.vertices, direct.metal, direct.chirality, direct.haptic
        )
        for cid in raw.ids
    } == {direct.chirality}

    fe = rx.metal(text, center="Fe", stereo="free")[0]
    raw = core_embed(fe, n=2, seed=9, prune_rms=-1)
    expected_mn = I.centre_states(fe, "Mn")[0]
    mn_vertices, mn_haptic, _mn_winding, _mn_donors = materialized_state(fe, expected_mn)
    assert {
        tuple(I.from_geometry(Chem.Mol(raw.mol, False, int(cid)), center="Fe").haptic_winding.values())
        for cid in raw.ids
    } == {tuple(fe.haptic_winding.values())}
    assert {
        _metal_stereo.realised_chirality(
            raw.mol,
            cid,
            expected_mn.geometry,
            mn_vertices,
            expected_mn.atom,
            expected_mn.hand,
            mn_haptic,
        )
        for cid in raw.ids
    } == {expected_mn.hand}


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_multimetal_haptic_center_enumerates_each_face_before_uff():
    source = read_xyz(_MNH)
    preserved = rx.metal(source, center="Fe", stereo="preserve")
    inverted = rx.metal(source, center="Fe", stereo="invert")
    racemic = rx.metal(source, center="Fe", stereo="racemic")
    free = rx.metal(source, center="Fe", stereo="free")
    assert [tuple(iso.haptic_winding.values()) for iso in preserved] == [("-",)]
    assert [tuple(iso.haptic_winding.values()) for iso in inverted] == [("+",)]
    assert len(racemic) == 8  # 2 Fe face windings x 2 N5 hands x 2 C47 hands
    assert {tuple(iso.haptic_winding.values()) for iso in racemic} == {("+",), ("-",)}
    assert {iso.stereo_label for iso in racemic} == {
        "N5:R,C47:R",
        "N5:R,C47:S",
        "N5:S,C47:R",
        "N5:S,C47:S",
    }
    assert [tuple(iso.haptic_winding.values()) for iso in free] == [("-",)]

    representatives = [
        next(iso for iso in racemic if tuple(iso.haptic_winding.values()) == (winding,)) for winding in ("+", "-")
    ]
    for iso in representatives:
        raw = core_embed(iso, n=2, seed=3, prune_rms=-1)
        assert {
            tuple(I.from_geometry(Chem.Mol(raw.mol, False, int(cid)), center="Fe").haptic_winding.values())
            for cid in raw.ids
        } == {tuple(iso.haptic_winding.values())}

    spectator_n = racemic.filter(stereo="N5:S")[0]
    cleaned = rx.embed(spectator_n, n=2, seed=2)
    assert set(cleaned.sphere) == {0, 1}
    assert {stereo.stereo_from_3d(Chem.Mol(cleaned.mol, False, int(cid)), exclude={0, 1}) for cid in cleaned.ids} == {
        spectator_n.stereo_label
    }


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_all_centers_compiles_only_the_selected_product(monkeypatch):
    calls = []
    build = C.coordination

    def counted(*args, **kwargs):
        calls.append(args[1])
        return build(*args, **kwargs)

    monkeypatch.setattr(C, "coordination", counted)
    isomers = rx.metal(read_xyz(_MNH), screen=False)
    assert len(isomers) == 15
    assert calls == []
    _ = isomers[0].cons
    assert len(calls) == 2
    assert set(calls) == {state.atom for state in isomers[0].centres}


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_all_centers_is_the_cartesian_product_and_roundtrips():
    source = read_xyz(_MNH)
    isomers = rx.metal(source, screen=False)
    assert len(isomers) == 15  # exact Mn graph/polyhedron orbits; preserve the measured N, C, and Fe face
    assert len(isomers.filter(center="Mn", label="fac")) == 6
    assert len(isomers.filter(center="Mn", label="mer")) == 9
    assert {iso.stereo_label for iso in isomers} == {"N5:R,C47:R"}
    assert len(isomers.filter(center="Fe", haptic="Sₚ", stereo="N5:R,C47:R")) == 15
    with pytest.raises(ValueError, match="matched 15") as error:
        isomers.select(center="Fe", haptic="Sₚ")
    assert "Fe0" in str(error.value)
    assert "linear" in str(error.value)
    assert all(len(iso.centres) == 2 and not iso.cons.shapes and not iso.cons.frozen for iso in isomers)

    n_racemic = rx.metal(source, stereo={"N5": "racemic"}, screen=False)
    assert len(n_racemic) == 30
    assert {iso.stereo_label for iso in n_racemic} == {"N5:R,C47:R", "N5:S,C47:R"}
    racemic = rx.metal(source, stereo="racemic", screen=False)
    assert len(racemic) == 120
    assert {tuple(materialized_state(iso, I.centre_states(iso, "Fe")[0])[2].values()) for iso in racemic} == {
        ("+",),
        ("-",),
    }
    assert len(racemic.filter(stereo="N5:S,C47:R")) == 30

    free = rx.metal(source, stereo="free", screen=False)
    free_strings = {rx.cxsmiles(iso) for iso in free}
    assert free_strings == {rx.cxsmiles(iso) for iso in rx.metal(source, stereo={"point": "free"}, screen=False)}
    assert free_strings == {rx.cxsmiles(iso) for iso in rx.metal(source, stereo={"N5": "free"}, screen=False)}
    assert len(free) == 15
    assert {iso.stereo_label for iso in free} == {"C47:R"}
    for selector in ({"H66": "racemic"}, {"C64": "racemic"}):
        with pytest.raises(ValueError, match="not configurable point stereocentres"):
            rx.metal(source, stereo=selector)

    retained = I.from_geometry(source, center="all")
    assert len(retained.centres) == 2
    direct = rx.embed(source, n=1, seed=7)
    assert direct.ids
    assert len(direct.iso.centres) == 2

    strings = {rx.cxsmiles(iso) for iso in isomers}
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    assert strings == {rx.cxsmiles(iso) for iso in rx.metal(reversed_source, screen=False)}
    assert len(strings) == 15
    assert all(rx.cxsmiles(rx.metal(text, center="all")[0]) == text for text in strings)
    written = rx.cxsmiles(source)
    assert written in strings
    assert len(rx.metal(written)) == 1
    assert rx.embed(written, n=1, seed=7).ids
    stated = S.parse_smiles(written)
    reversed_stated = Chem.RenumberAtoms(stated, list(reversed(range(stated.GetNumAtoms()))))
    original, reversed_iso = rx.metal(stated)[0], rx.metal(reversed_stated)[0]
    assert (original.real_z, original.geometry, original.label) == (
        reversed_iso.real_z,
        reversed_iso.geometry,
        reversed_iso.label,
    )

    planar = rx.metal(written, stereo={"planar": "racemic"})
    assert len(planar) == 2
    assert {tuple(materialized_state(iso, I.centre_states(iso, "Fe")[0])[2].values()) for iso in planar} == {
        ("+",),
        ("-",),
    }
    inverted = rx.metal(written, stereo="invert")
    assert len(inverted) == 1
    assert list(stereo.point_stereo(inverted[0].stereo_label).values()) == ["S", "S"]
    assert tuple(materialized_state(inverted[0], I.centre_states(inverted[0], "Fe")[0])[2].values()) == ("+",)

    signed = next(text for text in strings if re.search(r"\.atomNote\.s\d+-", text))
    note = re.search(r"\.atomNote\.(s\d+)-", signed)
    assert note is not None
    unsigned = signed.replace(note.group(0), f".atomNote.{note.group(1)}")
    for requested in (None, "unassigned", "invert"):
        expanded = rx.metal(unsigned, stereo=requested)
        assert len(expanded) == 2
        assert {tuple(materialized_state(iso, I.centre_states(iso, "Fe")[0])[2].values()) for iso in expanded} == {
            ("+",),
            ("-",),
        }
    with pytest.raises(ValueError, match="stereo expansion produced several states"):
        rx.embed(unsigned, n=1, seed=7)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_multimetal_reach_preserves_reference_and_only_omits_opposed_short_chelates():
    source = read_xyz(_MNH)
    exact = {rx.cxsmiles(iso): iso for iso in rx.metal(source, screen=False, lengths="input")}
    screened = {rx.cxsmiles(iso) for iso in rx.metal(source, lengths="input")}
    assert len(exact) == 15
    assert len(screened) == 12
    assert screened < exact.keys()
    assert rx.cxsmiles(source) in screened
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    assert {rx.cxsmiles(iso) for iso in rx.metal(reversed_source, lengths="input")} == screened
    for identity in exact.keys() - screened:
        iso = exact[identity]
        left, right = [atom for atom in iso.donors if source.GetAtomWithIdx(atom).GetAtomicNum() == 7]
        assert len(Chem.GetShortestPath(iso._graph, left, right)) == 4
        rays = np.asarray(vertex_dirs(iso.geometry))
        np.testing.assert_allclose(rays[iso.vertices.index(left)], -rays[iso.vertices.index(right)])


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_all_centers_stacks_the_frozen_core_once():
    fixed = [1, 5, 63, 64, 65, 66]
    source = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    assert len(rx.metal(source, center="Mn", fix=fixed, stereo="preserve")) == 12
    haptic_racemic = {"planar": "racemic"}
    assert len(rx.metal(source, center="Fe", fix=fixed, stereo=haptic_racemic)) == 2
    isomers = rx.metal(source, center="all", fix=fixed, stereo=haptic_racemic)
    assert len(isomers) == 24
    assert {iso.stereo_label for iso in isomers} == {"N5:R,C47:R"}
    assert all(iso.cons.frozen == set(fixed) and not iso.cons.shapes for iso in isomers)

    reference_cx = rx.cxsmiles(source)
    target = next(iso for iso in isomers if rx.cxsmiles(iso) == reference_cx)
    embedded = rx.embed(target, n=1, seed=1)
    assert embedded.n == 1
    reference = source.GetConformer().GetPositions()[fixed]
    realised_core = embedded.mol.GetConformer(embedded.ids[0]).GetPositions()[fixed]
    reference_distances = np.linalg.norm(reference[:, None] - reference, axis=2)
    realised_distances = np.linalg.norm(realised_core[:, None] - realised_core, axis=2)
    assert np.allclose(realised_distances, reference_distances, atol=1e-12)
    expected = rx.cxsmiles(target).split("|", 1)[1]
    realised = rx.cxsmiles(Chem.Mol(embedded.mol, False, embedded.ids[0])).split("|", 1)[1]
    assert realised == expected


def test_bridging_donor_carries_one_slot_per_adjacent_metal():
    mol = S.parse_smiles("N(->[Pd+]<-[Cl-])->[Pt+]<-[Br-]")
    positions = {"N": (0, 0, 0), "Pd": (-2, 0, 0), "Cl": (-4, 0, 0), "Pt": (2, 0, 0), "Br": (4, 0, 0)}
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        conf.SetAtomPosition(atom.GetIdx(), positions[atom.GetSymbol()])
    mol.AddConformer(conf)

    text = rx.cxsmiles(mol)
    note = next(
        atom.GetProp("atomNote")
        for atom in S.parse_smiles(text).GetAtoms()
        if atom.HasProp("atomNote") and ";" in atom.GetProp("atomNote")
    )
    slots = note.split(";")
    assert len(slots) == 2
    assert all(slot in {"s0", "s1"} for slot in slots)
    assert all(len(rx.metal(text, center=metal, stereo="free")[0].centres) == 2 for metal in ("Pd", "Pt"))
    assert all(rx.cxsmiles(rx.metal(text, center=metal, stereo="free")[0]) == text for metal in ("Pd", "Pt"))

    # The bridge list is resolved onto RDKit bond properties at parse time, so renumbering cannot swap it.
    forged = S.parse_smiles(text)
    bridge = next(atom for atom in forged.GetAtoms() if atom.GetSymbol() == "N")
    metals = sorted(
        (atom for atom in bridge.GetNeighbors() if atom.GetSymbol() in {"Pd", "Pt"}), key=lambda a: a.GetIdx()
    )
    for slot, metal in enumerate(metals):
        forged.GetBondBetweenAtoms(bridge.GetIdx(), metal.GetIdx()).SetProp(SLOT_BOND_PROP, f"s{slot}")
        terminal = next(atom for atom in metal.GetNeighbors() if atom.GetIdx() != bridge.GetIdx())
        forged.GetBondBetweenAtoms(terminal.GetIdx(), metal.GetIdx()).SetProp(SLOT_BOND_PROP, f"s{1 - slot}")

    def elements(molecule, metal):
        iso = rx.metal(molecule, center=metal, stereo="free")[0]
        return tuple(iso.mol.GetAtomWithIdx(atom).GetSymbol() for atom in iso.vertices)

    expected = {metal: elements(forged, metal) for metal in ("Pd", "Pt")}
    reversed_mol = Chem.RenumberAtoms(forged, list(reversed(range(forged.GetNumAtoms()))))
    assert {metal: elements(reversed_mol, metal) for metal in ("Pd", "Pt")} == expected
    assert rx.cxsmiles(rx.metal(reversed_mol, center="Pd", stereo="free")[0]) == rx.cxsmiles(
        rx.metal(forged, center="Pd", stereo="free")[0]
    )


def test_multimetal_writer_rejects_lossy_graphs():
    with pytest.raises(ValueError, match="no transition metal"):
        rx.cxsmiles(Chem.MolFromSmiles("CC"))
    with pytest.raises(ValueError, match="ambiguous charge allocation"):
        S.dative_smiles(Chem.MolFromSmiles("[Cl][Fe]O[Mn][Br]"))

    mol = S.parse_smiles("N(->[Pd+]<-[Cl-])->[Pt+]<-[Br-]")
    rw = Chem.RWMol(mol)
    metals = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() in _metal.COORDINATION_METALS]
    rw.AddBond(metals[0], metals[1], Chem.BondType.SINGLE)
    direct = rw.GetMol()
    conf = Chem.Conformer(direct.GetNumAtoms())
    for atom in range(direct.GetNumAtoms()):
        conf.SetAtomPosition(atom, (float(2 * atom), 0.0, 0.0))
    direct.AddConformer(conf)
    with pytest.raises(ValueError, match="bond type cannot be restored"):
        I.from_geometry(direct)


def test_cxsmiles_leaves_an_unbound_metal_without_an_arrangement_note():
    mol = Chem.MolFromSmiles("[Ag+].F/C=C/F")

    assert rx.cxsmiles(mol) == S.dative_smiles(mol)


def _tied_palladium_centres(second_seating):
    """Return two graph-equivalent square-planar Pd centres with the requested second seating."""
    mol = Chem.MolFromSmiles("[Pd](F)(F)(Cl)Cl.[Pd](F)(F)(Cl)Cl")
    conf = Chem.Conformer(mol.GetNumAtoms())
    directions = np.asarray(vertex_dirs("square_planar"))
    for offset, (metal, seating) in enumerate(((0, [1, 2, 3, 4]), (5, second_seating))):
        origin = np.array([8.0 * offset, 0.0, 0.0])
        conf.SetAtomPosition(metal, tuple(map(float, origin)))
        for vertex, donor in enumerate(seating):
            conf.SetAtomPosition(donor, tuple(map(float, origin + 2.0 * directions[vertex])))
    mol.AddConformer(conf)
    return mol


def test_multimetal_writer_rejects_distinct_states_on_tied_centres():
    with pytest.raises(ValueError, match="symmetry-equivalent metal centres"):
        rx.cxsmiles(_tied_palladium_centres([6, 8, 7, 9]))


def test_identical_states_on_tied_centres_roundtrip():
    text = rx.cxsmiles(_tied_palladium_centres([6, 7, 8, 9]))
    assert rx.cxsmiles(rx.metal(text)[0]) == text


def test_cxsmiles_rejects_unknown_shape():
    bare = I.from_surrogate(Chem.MolFromSmiles("[Pt](F)(F)(Cl)Cl"), [(0, 78, 0)], [])
    with pytest.raises(ValueError, match="no polyhedron template"):
        rx.cxsmiles(bare)
