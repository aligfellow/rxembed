"""Test canonical dative SMILES and arrangement-bearing CXSMILES round trips."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

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
from rxembed.metal_polyhedron import SLOT_BOND_PROP, rotation_group, vertex_dirs
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


def test_native_atrop_cx_survives_covalent_input_and_metal_enumeration():
    isomers = rx.metal(_ATROP_RU_COVALENT_CX, "OCT")

    assert len(isomers) == 3
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


def test_writer_hides_routine_h_and_keeps_hydride():
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


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_dative_smiles_roundtrips_nonstandard_complex():
    mol = read_xyz(_MN_H2)
    assert any(a.GetAtomicNum() == 1 and a.GetDegree() > 1 for a in mol.GetAtoms()), (
        "this fixture must contain an over-connected hydrogen, or it does not test the repair"
    )
    assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None, "a naive write must fail here, or there is nothing to fix"

    def metals(m):  # SMILES renumbers, so the multiset is the claim, not the order
        return sorted(
            (m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(i).GetFormalCharge()) for i in _metal.metal_indices(m)
        )

    text = S.dative_smiles(mol)
    back = Chem.AddHs(Chem.MolFromSmiles(text))
    assert back.GetNumAtoms() == mol.GetNumAtoms()
    assert metals(back) == metals(mol) == [("Fe", 2), ("Mn", 0)]
    assert S.dative_smiles(S.parse_smiles(text)) == text


def test_dative_smiles_rejects_unreadable_graph():
    rw = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl")))
    c = rw.AddAtom(Chem.Atom(6))
    for _ in range(5):
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(c, h, Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="round-tripping SMILES"):
        S.dative_smiles(rw.GetMol())


# --- canonicality: one species, one string, whatever Lewis form described it ------------------------------

# Each pair is one species written two ways, and the total charge is matched inside the pair on purpose: a
# covalent `[Pd]Cl` is Pd(II)Cl2 only against `[Cl-]->[Pd+2]`, and comparing it to `[Pd+]` would be comparing
# two different anions.
_LEWIS_PAIRS = {
    "halide": ("[NH3]->[Pt](<-[NH3])(Cl)Cl", "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]"),
    "neutral phosphine beside an anion": ("CP(C)(C)->[Rh]Cl", "CP(C)(C)->[Rh+]<-[Cl-]"),
}
# The same construction where the rule deliberately stops; see the boundary test below for why.
_PARTLY_FILLED = {
    "amide": ("CN(C)[Pd](Cl)Cl", "C[N-](C)->[Pd+3](<-[Cl-])<-[Cl-]"),
    "alkyl": ("C[Pd](Cl)(Cl)C", "[CH3-]->[Pd+4](<-[Cl-])(<-[Cl-])<-[CH3-]"),
}


@pytest.mark.parametrize(("kind", "pair"), _LEWIS_PAIRS.items(), ids=["halide", "phosphine"])
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


def test_isomer_and_mol_write_same_constitution():
    for smi in (s for pair in (*_LEWIS_PAIRS.values(), *_PARTLY_FILLED.values()) for s in pair):
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
        assert any([was[q[v]] for v in range(len(was))] == now for q in rotation_group(iso.geometry)), (
            f"{iso.label}: {was} and {now} are not the same arrangement under any rotation of the template"
        )


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

    realised = rx.embed(isomers.select(stereo="R"), n=1, seed=7).minimize().mol
    donor = realised.GetAtomWithIdx(23)
    donor.SetChiralTag(
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        if donor.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
        else Chem.ChiralType.CHI_TETRAHEDRAL_CW
    )
    assert rx.cxsmiles(realised) == texts["C23:R"]


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


@pytest.mark.parametrize(
    ("smiles", "label"),
    [
        (_ETA2_ASYM_E, "C1=C2:E"),
        (r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "C1=C2:Z"),
    ],
)
def test_eta2_face_and_ez_are_one_cxsmiles_fixed_point(smiles, label):
    isomers = rx.metal(smiles, "SPL")[:2]
    texts = {next(iter(iso.haptic_winding.values())): rx.cxsmiles(iso) for iso in isomers}
    assert len(set(texts.values())) == 2
    assert all("/" in text.split(" |", 1)[0] or "\\" in text.split(" |", 1)[0] for text in texts.values())
    assert all(",c:" not in text and ",t:" not in text for text in texts.values())
    for sign, text in texts.items():
        parsed = rx.parse_smiles(text.split(" |", 1)[0])
        bond = next(bond for bond in parsed.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
        assert len(set(bond.GetStereoAtoms())) == 2
        back = rx.metal(text)
        assert len(back) == 1
        assert back[0].stereo_label == label
        assert set(back[0].haptic_winding.values()) == {sign}
        assert rx.cxsmiles(back[0]) == text
        embedded = rx.embed(back[0], n=1, seed=7)
        order = list(reversed(range(embedded.mol.GetNumAtoms())))
        assert rx.cxsmiles(Chem.RenumberAtoms(embedded.mol, order)) == text
    plain = rx.dative_smiles(rx.embed(isomers[0], n=1, seed=7).mol)
    assert "/" in plain or "\\" in plain
    assert {iso.stereo_label for iso in rx.metal(plain, "SPL")} == {label}


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
    valid = native.removesuffix("|") + ",t:1|"
    assert rx.metal(valid)[0].stereo_label == "C1=C2:E"
    for forged in (
        valid.replace("t:1", "t:999"),
        valid.replace("t:1", "t:0"),
        valid.replace("t:1", "t:"),
        valid.replace("t:1", "t:1,c:1"),
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
    source = read_xyz(_MN_H2)
    direct = rx.metal(_MN_H2, "OCT", center="Mn", fix=[1, 5, 63, 64, 65, 66], stereo="free").filter(label="mer")[0]
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
    isomers = rx.metal(read_xyz(_MNH))
    assert len(isomers) == 9
    assert calls == []
    _ = isomers[0].cons
    assert len(calls) == 2
    assert set(calls) == {state.atom for state in isomers[0].centres}


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_all_centers_is_the_cartesian_product_and_roundtrips():
    source = read_xyz(_MNH)
    isomers = rx.metal(source)
    assert len(isomers) == 9  # preserve the measured N, C, and Fe face; enumerate Mn arrangements
    assert len(isomers.filter(center="Mn", label="fac")) == 6
    assert len(isomers.filter(center="Mn", label="mer")) == 3
    assert {iso.stereo_label for iso in isomers} == {"N5:R,C47:R"}
    assert len(isomers.filter(center="Fe", haptic="Sₚ", stereo="N5:R,C47:R")) == 9
    with pytest.raises(ValueError, match="matched 9") as error:
        isomers.select(center="Fe", haptic="Sₚ")
    assert "Fe0" in str(error.value)
    assert "linear" in str(error.value)
    assert all(len(iso.centres) == 2 and not iso.cons.shapes and not iso.cons.frozen for iso in isomers)

    n_racemic = rx.metal(source, stereo={"N5": "racemic"})
    assert len(n_racemic) == 18
    assert {iso.stereo_label for iso in n_racemic} == {"N5:R,C47:R", "N5:S,C47:R"}
    racemic = rx.metal(source, stereo="racemic")
    assert len(racemic) == 72
    assert {tuple(materialized_state(iso, I.centre_states(iso, "Fe")[0])[2].values()) for iso in racemic} == {
        ("+",),
        ("-",),
    }
    assert len(racemic.filter(stereo="N5:S,C47:R")) == 18

    free = rx.metal(source, stereo="free")
    free_strings = {rx.cxsmiles(iso) for iso in free}
    assert free_strings == {rx.cxsmiles(iso) for iso in rx.metal(source, stereo={"point": "free"})}
    assert free_strings == {rx.cxsmiles(iso) for iso in rx.metal(source, stereo={"N5": "free"})}
    assert len(free) == 9
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
    assert strings == {rx.cxsmiles(iso) for iso in rx.metal(reversed_source)}
    assert len(strings) == 9
    assert all(rx.cxsmiles(rx.metal(text, center="all")[0]) == text for text in strings)
    written = next(iter(strings))
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

    unsigned = written.replace(".atomNote.s1-", ".atomNote.s1")
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
def test_all_centers_stacks_the_frozen_core_once():
    fixed = [1, 5, 63, 64, 65, 66]
    source = read_xyz(_MN_H2)
    assert len(rx.metal(source, center="Mn", fix=fixed, stereo="preserve")) == 6
    haptic_racemic = {"planar": "racemic"}
    assert len(rx.metal(source, center="Fe", fix=fixed, stereo=haptic_racemic)) == 2
    isomers = rx.metal(source, center="all", fix=fixed, stereo=haptic_racemic)
    assert len(isomers) == 12
    assert {iso.stereo_label for iso in isomers} == {"N5:R,C47:R"}
    assert all(iso.cons.frozen == set(fixed) and not iso.cons.shapes for iso in isomers)

    target = isomers[9]  # this seed initially relaxed to the right hand but the wrong Mn slot arrangement
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
