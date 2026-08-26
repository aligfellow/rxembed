"""Test canonical dative SMILES and arrangement-bearing CXSMILES round trips."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_core as _metal
from rxembed import metal_isomers as K  # noqa: N812
from rxembed import metal_smiles as S  # noqa: N812
from rxembed import stereo
from rxembed.core import embed as core_embed
from rxembed.metal_core import VACANT
from rxembed.metal_polyhedron import rotation_group
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.perceive import read_xyz

_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MA2B2_SEATS = {"cis": [1, 2, 3, 4], "trans": [1, 3, 2, 4]}  # square_planar: 0 and 2 are the trans pair
_MA3B3_SEATS = {"fac": [1, 4, 2, 5, 3, 6], "mer": [1, 2, 3, 4, 5, 6]}  # octahedral: 0/1, 2/3, 4/5 are trans
_CIS3 = [1, 3, 2, 5, 4, 6]  # cis,cis,cis-MA2B2C2: every same-element pair at 90 degrees, so the centre is chiral


def _isomer(smi, geometry, seating):
    """An `Isomer` stated as a vertex ordering: `seating[v]` is the atom sitting at vertex v.

    The intuitive door, and a hermetic one: no conformer, no embed and no `benchmark/corpus`, because a
    vertex ordering already fixes the arrangement and the handedness. Read the record in `metal_polyhedron`
    for what a vertex number means; the convention is not uniform across the shapes.
    """
    return K.Isomer(Chem.MolFromSmiles(smi), geometry, seating)


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
    ranks = _metal._donor_classes(mol, iso.donors)
    return {
        dummy: sign
        for dummy, face in iso.haptic.items()
        if (sign := _metal._face_winding(mol, pos, iso.metal, face, ranks))
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
    text = rx.cxsmiles(K.Isomer(cisplatin, "square_planar", [0, 2, 3, 4]))
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

    back = Chem.AddHs(Chem.MolFromSmiles(S.dative_smiles(mol)))
    assert back.GetNumAtoms() == mol.GetNumAtoms()
    assert metals(back) == metals(mol) == [("Fe", 2), ("Mn", 0)]


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
        retained = K.from_geometry(realised)
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

    # Seed 2 makes restrained UFF cross the amine inversion barrier; rxembed keeps the correct DG seed.
    realised = rx.embed(rx.metal("[Pd](Cl)(Cl)(Cl)([N@H](C)O)", "square_planar")[0], n=1, seed=2).minimize().mol
    assert rx.cxsmiles(realised) == hands["N4:R"]


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
    assert {"η5Rₚ", "η5Sₚ"} <= {part[:4] for iso in isomers for part in K.arrangement(iso).split()}
    by_sign = {next(iter(iso.haptic_winding.values())): K.arrangement(iso) for iso in isomers}
    assert "η5Rₚ" in by_sign["+"]
    assert "η5Sₚ" in by_sign["-"]
    assert all("Rₚ" not in rx.cxsmiles(iso) and "Sₚ" not in rx.cxsmiles(iso) for iso in isomers)
    for iso in isomers:
        embedded = core_embed(iso, n=4, seed=7, prune_rms=-1)
        assert {next(iter(_haptic_windings(embedded.mol, iso, cid).values())) for cid in embedded.ids} == set(
            iso.haptic_winding.values()
        )


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
    iso = rx.metal(source, stereo="free")[0]
    raw = core_embed(iso, n=32, seed=7, prune_rms=-1)
    by_winding = {tuple(_haptic_windings(raw.mol, iso, cid).values()): cid for cid in raw.ids}
    assert {("+", "-"), ("-", "+")} <= set(by_winding)
    strings = {rx.cxsmiles(Chem.Mol(raw.mol, False, int(by_winding[winding]))) for winding in (("+", "-"), ("-", "+"))}
    assert len(strings) == 1


def test_stated_arrangement_rejects_redundant_arguments():
    text = rx.cxsmiles(_isomer("[Pt](F)(F)(F)(Cl)(Cl)Cl", "octahedral", _MA3B3_SEATS["fac"]))
    mol = S.parse_smiles(text)
    assert len(rx.enumerate_isomers(mol, "OCT")) == 1, "naming the shape the string states is not a contradiction"
    for kwargs in ({"geometry": "trigonal_prismatic"}, {"fix": [1, 2]}):
        with pytest.raises(ValueError, match="nothing to act on"):
            rx.enumerate_isomers(mol, **kwargs)


def test_cxsmiles_rejects_multiple_metals_and_unknown_shape():
    iso = _isomer("[Pt](F)(F)(Cl)Cl", "square_planar", _MA2B2_SEATS["cis"])
    rw = Chem.RWMol(iso.mol)
    rw.AddAtom(Chem.Atom(46))  # a spectator Pd bonded to nothing: still a second centre to state
    iso.mol, iso.spectator_metals = rw.GetMol(), [(rw.GetNumAtoms() - 1, 46, 0)]
    with pytest.raises(NotImplementedError, match="one metal centre"):
        rx.cxsmiles(iso)
    bare = K.from_surrogate(Chem.MolFromSmiles("[Pt](F)(F)(Cl)Cl"), [(0, 78, 0)], [])
    with pytest.raises(ValueError, match="no polyhedron template"):
        rx.cxsmiles(bare)
