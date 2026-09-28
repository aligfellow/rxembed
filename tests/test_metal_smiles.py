"""Test canonical dative SMILES and arrangement-bearing CXSMILES round trips."""

from __future__ import annotations

import importlib
import logging
import re
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_isomer, metal_smiles, stereo
from rxembed.core import embed as core_embed
from rxembed.metal_core import VACANT, connect_metal, materialized_state
from rxembed.metal_perceive import SHAPE_PROP, SHAPE_REQUEST_PROP
from rxembed.metal_polyhedron import SLOT_BOND_PROP, point_group, record, vertex_dirs
from rxembed.metal_stereo import donor_classes, eta2_signatures, face_winding
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.perceive import read_xyz
from tests.conftest import EXAMPLES_DIR
from tests.metal_fixtures import BUTADIENE_FE_CO3

_MN_H2 = str(EXAMPLES_DIR / "mn-h2.xyz")  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MNH = str(EXAMPLES_DIR / "mnh.xyz")  # the corresponding Mn hydride minimum
_MA3B3_SEATS = {"fac": [1, 4, 2, 5, 3, 6], "mer": [1, 2, 3, 4, 5, 6]}  # octahedral: 0/1, 2/3, 4/5 are trans
_ETA2_ASYM_E = r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
_BINAP_PD = (
    "[Pd+2]%90(<-[Cl-])(<-[Cl-])(<-P(c1ccccc1)(c2ccccc2)c3ccc4ccccc4c3-c3c(P(c4ccccc4)(c5ccccc5)->%90)ccc4ccccc34)"
)
# One shared owner for a (smi, geometry) coverage set, reused by `test_all_enumerated_isomers_read_back`.


def _isomer(smi, geometry, seating):
    """An `Isomer` stated as a vertex ordering: `seating[v]` is the atom sitting at vertex v.

    The intuitive door, and a hermetic one: no conformer, no embed and no `benchmark/corpus`, because a
    vertex ordering already fixes the arrangement and the handedness. Read the record in `metal_polyhedron`
    for what a vertex number means; the convention is not uniform across the shapes.
    """
    return metal_isomer.Isomer(Chem.MolFromSmiles(smi), geometry, seating)


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
    ranks = donor_classes(mol, iso.donors)
    return {
        dummy: sign for dummy, face in iso.haptic.items() if (sign := face_winding(mol, pos, iso.metal, face, ranks))
    }


# Run the graph round trip while a meta-path block refuses every optional dependency. The final import checks
# that the block is active; an in-process ``sys.modules`` probe alone would be a null measurement.


# --- the parse / write contract -------------------------------------------------------------------------


def test_covalent_metal_input_normalizes_before_kekulization():
    mol = metal_smiles.parse_smiles("[Zn](n1ccccc1)n1ccccc1")

    assert metal_smiles.dative_smiles(mol) == "c1cc[n](->[Zn]<-[n]2ccccc2)cc1"


def test_parse_smiles_removes_explicit_hydrogens_by_default():
    mol = metal_smiles.parse_smiles("[H]OC")
    assert not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())


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
        written.add(metal_smiles.dative_smiles(mol))

    assert written == {"[BH3-][H]->[Fe+]"}


def test_hydrogen_bond_cycle_does_not_make_chelate_imine_stereo_order_dependent():
    mol = metal_smiles.parse_smiles("[N]1(->[Ni]2)/O[H]~O=[N+]->2=C/C=1 |Z:3|", remove_hs=False)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    text = metal_smiles.dative_smiles(mol)

    assert metal_smiles.dative_smiles(reversed_mol) == text
    back = metal_smiles.parse_smiles(text, remove_hs=False)
    assert not stereo.bond_stereo(stereo.defined_stereo_label(back, metal_smiles.metal_indices(back)))


def test_isomer_and_mol_write_same_constitution():
    for smi in (s for pair in _LEWIS_PAIRS.values() for s in pair):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        isos = rx.enumerate_isomers(mol)
        assert isos, f"{smi} enumerated nothing"
        core = rx.cxsmiles(isos[0]).split(" |", 1)[0]
        assert core == metal_smiles.dative_smiles(mol), f"{smi}: the Isomer door wrote a different constitution"


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


def test_coplanar_bound_biaryl_is_not_forced_to_have_an_atrop_hand():
    mol = metal_smiles.parse_smiles(_BINAP_PD, remove_hs=False)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    point = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(point.x, point.y, 0.01))
    conf.Set3D(True)

    assert not stereo.axis_stereo(stereo.stereo_from_3d(mol, metal_smiles.metal_indices(mol)))


def test_zero_order_contact_is_written_but_is_not_constitution():
    """Only the O~N contact tells C1's two CH2OH arms apart, so C1 is no stereocentre; the contact still round-trips."""
    mol = metal_smiles.parse_smiles("C[C@@H](CO)CO~N |Z:5|")
    before = Chem.MolToCXSmiles(mol)
    assert stereo.defined_stereo_label(mol) == ""

    text = rx.dative_smiles(mol)
    back = metal_smiles.parse_smiles(text)

    assert "Z:" in text
    assert "@" not in text
    assert sum(bond.GetBondType() == Chem.BondType.ZERO for bond in back.GetBonds()) == 1
    assert metal_smiles.dative_smiles(back) == text
    assert metal_smiles.dative_smiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))) == text
    assert Chem.MolToCXSmiles(mol) == before


def test_cx_fields_losslessly_retain_conjugated_imine_ez():
    mol = metal_smiles.parse_smiles(
        r"CC1=c2\cccc\c2=[N]2->[Ni]34<-[N](=C5\[CH-]C=CC=C5[C@H](C)\[N]->3="
        r"c3/cc(C)c(C)c/c3=[N]->4\1)/C(=O)C\2=O"
    )
    label = stereo.defined_stereo_label(mol, metal_smiles.metal_indices(mol))
    core, at, bonds, unwritable = metal_smiles._write_dative(mol, label)

    fields = metal_smiles._cx_bond_stereo(core, label, at, bonds)
    text = f"{core} |{','.join(fields)}|"
    back = metal_smiles.parse_smiles(text)

    assert unwritable
    assert fields
    expected = {frozenset(at[idx] for idx in pair): code for pair, code in stereo.bond_stereo(label).items()}
    assert stereo.bond_stereo(stereo.defined_stereo_label(back, metal_smiles.metal_indices(back))) == expected
    flipped = label.replace("C7=N8:E", "C7=N8:Z")
    assert flipped != label
    assert metal_smiles._cx_bond_stereo(core, flipped, at, bonds) != fields
    iso = metal_isomer.Isomer(mol, "square_planar", [8, 10, 19, 28])
    iso.stereo_label = label
    cx = rx.cxsmiles(iso)
    assert rx.cxsmiles(rx.metal(cx)[0]) == cx


@pytest.mark.parametrize(
    ("smiles", "remove_hs"),
    [
        ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", True),
    ],
)
def test_dative_writer_retains_coordinated_imine_ez_and_maps_only_source_atoms(smiles, remove_hs):
    mol = metal_smiles.parse_smiles(smiles, remove_hs=remove_hs)
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Pt")
    bond = mol.GetBondBetweenAtoms(1, 2)
    bond.SetStereoAtoms(0, metal)
    bond.SetStereo(Chem.BondStereo.STEREOE)

    text, at, _bonds, unwritable = metal_smiles._write_dative(mol, "C1=N2:E")

    back = metal_smiles.parse_smiles(text, remove_hs=False)
    assert set(at) == set(range(mol.GetNumAtoms()))
    assert all(back.GetAtomWithIdx(at[i]).GetAtomicNum() == mol.GetAtomWithIdx(i).GetAtomicNum() for i in at)
    label = stereo.defined_stereo_label(back, metal_smiles.metal_indices(back))
    assert set(stereo.bond_stereo(label).values()) == {"E"}
    assert not unwritable


def test_dative_smiles_rejects_unreadable_graph():
    rw = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl")))
    c = rw.AddAtom(Chem.Atom(6))
    for _ in range(5):
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(c, h, Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="round-tripping SMILES"):
        metal_smiles.dative_smiles(rw.GetMol())


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


# --- the arrangement the SMILES grammar cannot say -------------------------------------------------------


def test_planar_chiral_ferrocene_winding_roundtrips_and_selects_after_dg():
    racemic = rx.metal(_planar_chiral_ferrocene(), stereo="racemic")
    assert len(racemic.filter(haptic="Sₚ")) == 1
    assert len(racemic.filter(haptic="Rₚ")) == 1

    iso = rx.metal(_planar_chiral_ferrocene(), stereo="free")[0]
    raw = core_embed(iso, n=8, params=rx.EmbedParams(seed=7, prune_rms=-1))
    by_sign = {next(iter(_haptic_windings(raw.mol, iso, cid).values())): cid for cid in raw.ids}
    assert set(by_sign) == {"+", "-"}, "the ungated DG control did not sample both windings"

    texts = {}
    for sign, cid in by_sign.items():
        realised = Chem.Mol(raw.mol, False, int(cid))
        text = rx.cxsmiles(realised)
        retained = metal_isomer.from_geometry(realised)
        assert set(retained.haptic_winding.values()) == {sign}
        assert set(rx.metal(realised, stereo="free")[0].haptic_winding.values()) == {sign}
        retained_embed = core_embed(retained, n=4, params=rx.EmbedParams(seed=17, prune_rms=-1))
        assert {
            next(iter(_haptic_windings(retained_embed.mol, retained, retained_cid).values()))
            for retained_cid in retained_embed.ids
        } == {sign}
        back = rx.enumerate_isomers(Chem.AddHs(metal_smiles.parse_smiles(text)), stereo="free")[0]
        assert set(back.haptic_winding.values()) == {sign}
        assert rx.cxsmiles(back) == text
        default_back = rx.enumerate_isomers(metal_smiles.parse_smiles(text))
        assert len(default_back) == 1
        assert not default_back[0].stereo_label
        assert rx.cxsmiles(Chem.RenumberAtoms(realised, list(reversed(range(realised.GetNumAtoms()))))) == text

        embedded = core_embed(back, n=6, params=rx.EmbedParams(seed=11, prune_rms=-1)).minimize()
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
    forged = metal_smiles.parse_smiles(texts["+"])
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


def _cd_fan_near_tie_conformer():
    """Return `(iso, mol, cid)`: the notebook-07 Cd near-tie isomer, and a conformer just past its SPY/TBP
    fit-residual crossover (square_pyramidal 0.228, next trigonal_bipyramidal 0.224; gap < `_FIT_MARGIN`).

    Spherical-linear interpolation between the two idealized templates' own `vertex_dirs`, slot by slot (the
    same construction `test_metal_enumeration.py`'s Berry-pseudorotation test uses): fixed geometry, no embed
    and no seed, so the near-tie does not depend on where a relax happens to land.
    """
    fans = rx.metal("[NH2]1CC[NH]2CC[NH2]->[Cd+2](<-[Cl-])(<-[Cl-])<-1<-2", "square_pyramidal")
    iso = next(c for c in fans if c.vertices[0] == 3 and set(c.vertices[1::2]) == {0, 6})

    spy = np.array(record("square_pyramidal").vertex_dirs, float)
    tbp = np.array(record("trigonal_bipyramidal").vertex_dirs, float)
    spy /= np.linalg.norm(spy, axis=1, keepdims=True)
    tbp /= np.linalg.norm(tbp, axis=1, keepdims=True)

    def slerp(a, b, t):
        theta = np.arccos(np.clip(a @ b, -1.0, 1.0))
        return a if theta < 1e-9 else (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)

    t = 0.32  # past the ~0.312 crossover: TBP argmin, SPY a near-tie runner-up inside _FIT_MARGIN
    pts = np.array([slerp(spy[i], tbp[i], t) for i in range(5)])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    bond_length = 2.4  # an ordinary Cd-N/Cd-Cl distance; the reading depends only on direction, not scale
    # the real dative-bonded graph cxsmiles(mol) itself needs, not the internal surrogate
    mol = connect_metal(iso.restore(Chem.Mol(iso.graph)), iso.donor_bonds)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(100.0 + i, 0.0, 0.0))  # park the ligand backbone away from the sphere
    conf.SetAtomPosition(iso.metal, Point3D(0.0, 0.0, 0.0))
    for slot, donor in enumerate(iso.vertices):
        conf.SetAtomPosition(donor, Point3D(*(bond_length * pts[slot])))
    cid = mol.AddConformer(conf, assignId=True)
    return iso, mol, cid


def test_cadmium_near_tie_conformer_round_trips_in_its_requested_frame():
    emb = importlib.import_module("rxembed.embed")
    iso, mol, cid = _cd_fan_near_tie_conformer()

    assert emb._coordination_state_failure(mol, cid, iso) is None, "rule B must accept the near-tie request"
    conf = mol.GetConformer(cid)
    assert conf.HasProp(SHAPE_REQUEST_PROP), "an accepted conformer must carry the machine-readable record"

    assert rx.cxsmiles(mol) == rx.cxsmiles(iso), "an accepted near-tie conformer must write CX in its own frame"

    raw = Chem.Mol(mol)
    raw.GetConformer(cid).ClearProp(SHAPE_REQUEST_PROP)
    raw.GetConformer(cid).ClearProp(SHAPE_PROP)
    assert rx.cxsmiles(raw) != rx.cxsmiles(iso), "a raw geometry with no request must keep the argmin reading"
    assert ".TBP:" in rx.cxsmiles(raw), "the argmin at this geometry is trigonal_bipyramidal, not the request"


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
        mol = metal_smiles.parse_smiles(text)
        assert metal_smiles.dative_smiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))) == core

    # The fallback writer must not leak an unrelated, invalid tag through cleanStereo=False.
    forged = metal_smiles.parse_smiles(texts["C23:S"])
    methyl = next(a for a in forged.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 1)
    methyl.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    params = Chem.SmilesWriteParams()
    params.cleanStereo = False
    assert "H3]" in Chem.MolToSmiles(forged, params)
    assert metal_smiles.dative_smiles(forged) == texts["C23:S"].split(" |", 1)[0]

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


@pytest.mark.parametrize(("smi", "geometry"), [("[CH2]=[CH2].Cl[Pt](Cl)Cl", "square_planar")], ids=["eta2"])
def test_all_enumerated_isomers_read_back(smi, geometry):
    isos = rx.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles(smi)), geometry)
    assert len(isos) >= 1
    for iso in isos:
        text = rx.cxsmiles(iso)
        back = rx.enumerate_isomers(metal_smiles.parse_smiles(text))
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


@pytest.mark.parametrize(
    ("smiles", "tokens"), [(BUTADIENE_FE_CO3, ["M", "P", "c"])], ids=["butadiene-mirror-symmetric-s-cis"]
)
def test_bound_diene_s_cis_and_s_trans_forms_are_isomers(smiles, tokens):
    """A bound diene is s-cis, named by its face or `c` when a mirror relates both, or s-trans P or M."""
    isomers = rx.metal(smiles)
    assert sorted("".join(iso.haptic_winding.values()) for iso in isomers) == tokens
    for iso in isomers:
        (back,) = rx.metal(rx.cxsmiles(iso))
        assert rx.cxsmiles(back) == rx.cxsmiles(iso)


@pytest.mark.parametrize("embedded", [True])
def test_raw_donor_point_override_composes_with_an_independent_ez_override(embedded):
    mol = metal_smiles.parse_smiles("C/C=C/C[N@H](C)->[Cu+]<-[Cl-]")
    if embedded:  # the coordinates hold the source hand; the stated label still wins
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=0) == 0
    expected = Chem.Mol(mol)
    donor = expected.GetAtomWithIdx(4)
    donor.InvertChirality()
    raw = "CW" if donor.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "CCW"

    text, at, _bonds, _unwritable = metal_smiles._write_dative(mol, f"N4:{raw},C1=C2:Z")
    back = metal_smiles.parse_smiles(text)

    label = stereo.defined_stereo_label(back, metal_smiles.metal_indices(back))
    assert stereo.bond_stereo(label) == {frozenset((at[1], at[2])): "Z"}
    for view in (expected, back):
        for bond in view.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
        view.UpdatePropertyCache(strict=False)
    assert back.HasSubstructMatch(expected, useChirality=True)


@pytest.mark.parametrize(
    ("smiles", "count", "names"),
    [
        (r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 6, {"(re,re)", "(si,si)"}),
        (r"[CH2]1=[CH](C)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", 6, {"(re)", "(si)"}),
    ],
    ids=["asymmetric-Z", "propene"],
)
def test_eta2_face_orientation_matrix(smiles, count, names):
    isomers = rx.metal(smiles, "SPL")
    shown = {name for name in names if any(name in iso.arrangement for iso in isomers)}
    assert len(isomers) == count
    assert shown == names
    assert {tuple(iso.haptic_winding.values()) for iso in isomers} == ({("+",), ("-",)} if names else {()})


def test_eta2_stereoany_does_not_imply_a_face_relation():
    iso = rx.metal(_ETA2_ASYM_E, "SPL")[0]
    face = next(iter(iso.haptic.values()))
    iso.mol.GetBondBetweenAtoms(*face).SetStereo(Chem.BondStereo.STEREOANY)
    assert eta2_signatures(iso.mol, face) == ((), ())


def test_dative_cx_option_retains_eta2_ez_without_arrangement_notes():
    text = rx.dative_smiles(rx.parse_smiles(_ETA2_ASYM_E), cx=True)

    assert "|t:" in text
    assert "atomNote" not in text
    parsed = rx.parse_smiles(text)
    assert {iso.stereo_label for iso in rx.metal(parsed, "SPL")} == {"C1=C2:E"}


@pytest.mark.parametrize(
    ("smiles", "label"),
    [
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
    raw = core_embed(iso, n=32, params=rx.EmbedParams(seed=7, prune_rms=-1))
    by_winding = {tuple(_haptic_windings(raw.mol, iso, cid).values()): cid for cid in raw.ids}
    assert {("+", "-"), ("-", "+")} <= set(by_winding)
    strings = {rx.cxsmiles(Chem.Mol(raw.mol, False, int(by_winding[winding]))) for winding in (("+", "-"), ("-", "+"))}
    assert len(strings) == 1
    engine = importlib.import_module("rxembed.embed")
    target = next(iso for iso in isomers if set(iso.haptic_winding.values()) == {"+", "-"})
    targets = engine._stereo_targets(target)
    ranks, eta2 = engine._winding_ranks(raw.mol, targets)
    assert all(
        engine._seed_stereo_matches(raw.mol, by_winding[winding], target, targets, ranks, eta2)
        for winding in (("+", "-"), ("-", "+"))
    )


def test_stated_arrangement_rejects_shape_override_and_composes_fix():
    text = rx.cxsmiles(_isomer("[Pt](F)(F)(F)(Cl)(Cl)Cl", "octahedral", _MA3B3_SEATS["fac"]))
    mol = metal_smiles.parse_smiles(text)
    assert len(rx.enumerate_isomers(mol, "OCT")) == 1, "naming the shape the string states is not a contradiction"
    with pytest.raises(ValueError, match="nothing to act on"):
        rx.enumerate_isomers(mol, geometry="trigonal_prismatic")
    with pytest.raises(ValueError, match="source has no geometry"):
        rx.enumerate_isomers(mol, fix=[1, 2])

    fixed = rx.enumerate_isomers(mol, fix={(1, 2): 2.0})
    assert fixed[0].cons.fixed[(1, 2)] == (2.0, 2.0)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_all_centers_is_the_cartesian_product_and_roundtrips():
    source = read_xyz(_MNH)
    isomers = rx.metal(source, screen=False)
    assert len(isomers) == 15  # exact Mn graph/polyhedron orbits; preserve the measured N, C, and Fe face
    assert len(isomers.filter(center="Mn", label="fac")) == 6
    assert len(isomers.filter(center="Mn", label="mer")) == 9
    assert {iso.stereo_label for iso in isomers} == {"N5:R,C47:R"}
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
    assert {
        tuple(materialized_state(iso, metal_isomer.centre_states(iso, "Fe")[0])[2].values()) for iso in racemic
    } == {
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

    retained = metal_isomer.from_geometry(source, center="all")
    assert len(retained.centres) == 2

    strings = {rx.cxsmiles(iso) for iso in isomers}
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    assert strings == {rx.cxsmiles(iso) for iso in rx.metal(reversed_source, screen=False)}
    assert len(strings) == 15
    written = rx.cxsmiles(source)
    assert written in strings
    assert len(rx.metal(written)) == 1
    stated = metal_smiles.parse_smiles(written)
    reversed_stated = Chem.RenumberAtoms(stated, list(reversed(range(stated.GetNumAtoms()))))
    original, reversed_iso = rx.metal(stated)[0], rx.metal(reversed_stated)[0]
    assert (original.real_z, original.geometry, original.label) == (
        reversed_iso.real_z,
        reversed_iso.geometry,
        reversed_iso.label,
    )

    planar = rx.metal(written, stereo={"planar": "racemic"})
    assert len(planar) == 2
    assert {tuple(materialized_state(iso, metal_isomer.centre_states(iso, "Fe")[0])[2].values()) for iso in planar} == {
        ("+",),
        ("-",),
    }
    inverted = rx.metal(written, stereo="invert")
    assert len(inverted) == 1
    assert list(stereo.point_stereo(inverted[0].stereo_label).values()) == ["S", "S"]
    assert tuple(materialized_state(inverted[0], metal_isomer.centre_states(inverted[0], "Fe")[0])[2].values()) == (
        "+",
    )

    signed = next(text for text in strings if re.search(r"\.atomNote\.s\d+-", text))
    note = re.search(r"\.atomNote\.(s\d+)-", signed)
    assert note is not None
    unsigned = signed.replace(note.group(0), f".atomNote.{note.group(1)}")
    for requested in (None, "unassigned", "invert"):
        expanded = rx.metal(unsigned, stereo=requested)
        assert len(expanded) == 2
        assert {
            tuple(materialized_state(iso, metal_isomer.centre_states(iso, "Fe")[0])[2].values()) for iso in expanded
        } == {
            ("+",),
            ("-",),
        }
    with pytest.raises(ValueError, match="stereo expansion produced several states"):
        rx.embed(unsigned, n=1, seed=7)


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
    mol = metal_smiles.parse_smiles("N(->[Pd+]<-[Cl-])->[Pt+]<-[Br-]")
    positions = {"N": (0, 0, 0), "Pd": (-2, 0, 0), "Cl": (-4, 0, 0), "Pt": (2, 0, 0), "Br": (4, 0, 0)}
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        conf.SetAtomPosition(atom.GetIdx(), positions[atom.GetSymbol()])
    mol.AddConformer(conf)

    text = rx.cxsmiles(mol)
    note = next(
        atom.GetProp("atomNote")
        for atom in metal_smiles.parse_smiles(text).GetAtoms()
        if atom.HasProp("atomNote") and ";" in atom.GetProp("atomNote")
    )
    slots = note.split(";")
    assert len(slots) == 2
    assert all(slot in {"s0", "s1"} for slot in slots)
    assert all(len(rx.metal(text, center=metal, stereo="free")[0].centres) == 2 for metal in ("Pd", "Pt"))
    assert all(rx.cxsmiles(rx.metal(text, center=metal, stereo="free")[0]) == text for metal in ("Pd", "Pt"))

    # The bridge list is resolved onto RDKit bond properties at parse time, so renumbering cannot swap it.
    forged = metal_smiles.parse_smiles(text)
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


def test_cxsmiles_leaves_an_unbound_metal_without_an_arrangement_note():
    mol = Chem.MolFromSmiles("[Ag+].F/C=C/F")

    assert rx.cxsmiles(mol) == metal_smiles.dative_smiles(mol)
