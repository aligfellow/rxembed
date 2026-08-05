"""`metal_smiles`: the string a complex is written as, and the string it is read back from.

One module because one contract. The writer indexes its ``atomProp`` block by position in the string it just
wrote, and the parser must keep the hydrogens the writer counted or every index addresses a different atom,
so the two are tested against each other rather than each against a fixture.

Three claims, in the order they matter:

1. the string round-trips. A SMILES that cannot be read back is worse than none, so the failure is loud.
2. the string is CANONICAL, which means it is a property of the species and not of the input that described
   it. A perceived M-L bond order is an artefact of whoever perceived it, so a covalent `M-Cl` and an ionic
   `[Cl-]->[M+]` must give one string; if they gave two, the word canonical would mean nothing.
3. the arrangement survives. `atomProp` carries what the SMILES grammar cannot say, so an isomer written and
   read back must be the same isomer, arrangement and handedness included.

RDKit only, no embed and no `benchmark/corpus`.
"""

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
from rxembed.metal_core import VACANT
from rxembed.metal_polyhedron import rotation_group
from rxembed.pipeline.perceive import _xyz_to_mol

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


# The graph round trip on a base install, run in a subprocess so a meta-path block can refuse the optional
# tier outright: an in-process `sys.modules` probe would only show that nothing HAPPENED to import it, which
# is not the claim. The last two statements check the block is blocking, since a null result needs a control.
_BASE_INSTALL = """
import sys
BLOCKED = ("rxembed.pipeline", "xyzgraph", "openconf", "scipy", "sklearn", "matplotlib", "networkx", "ase")

class BaseInstall:
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError("no module named %r (simulated base install)" % name)
        return None

sys.meta_path.insert(0, BaseInstall())

from rdkit import Chem
import rxembed as rx

mol = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pt](<-[NH3])(Cl)Cl"))
text = rx.canonical_smiles(rx.enumerate_isomers(mol, "square_planar")[0])
back = rx.enumerate_isomers(rx.parse_smiles(text))
assert len(back) == 1, back
assert len(rx.embed(back[0], n=2, seed=1).minimize().ids) >= 1, "nothing embedded"
assert rx.canonical_smiles(back[0]) == text, "the string is not a fixed point"
assert "rxembed.pipeline" not in sys.modules, "the optional tier was imported after all"
try:
    import rxembed.pipeline
except ImportError:
    pass
else:
    raise AssertionError("the block is not blocking, so this proves nothing")
print(text)
"""


def test_the_graph_round_trip_closes_with_the_optional_tier_uninstallable():
    run = subprocess.run([sys.executable, "-c", _BASE_INSTALL], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().startswith("[H][N]"), run.stdout


# --- the parse / write contract -------------------------------------------------------------------------


def test_a_bad_smiles_raises_instead_of_returning_the_none_rdkit_gives():
    assert S.parse_smiles("CCO").GetNumAtoms() == 3
    with pytest.raises(ValueError, match="could not parse SMILES"):
        S.parse_smiles("C1CC")


def test_write_dative_reports_the_atom_order_its_own_string_was_written_in():
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl"))
    smi, at = S.write_dative(mol)
    assert smi == S.dative_smiles(mol), "the two doors disagree about the string"
    assert sorted(at.values()) == list(range(mol.GetNumAtoms())), "the order is not a permutation of the atoms"
    params = Chem.SmilesParserParams()
    params.removeHs = False  # else the parsed indices are not the written positions, which is the whole point
    back = Chem.MolFromSmiles(smi, params)
    written = [back.GetAtomWithIdx(at[a.GetIdx()]).GetAtomicNum() for a in mol.GetAtoms()]
    assert written == [a.GetAtomicNum() for a in mol.GetAtoms()], "a position does not address its own atom"


def test_the_parser_hands_back_the_molecule_the_writer_wrote():
    # A fixture with explicit hydrogens, or there is nothing for `removeHs` to keep.
    cisplatin = Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pt](<-[NH3])(Cl)Cl"))
    text = rx.canonical_smiles(K.Isomer(cisplatin, "square_planar", [0, 2, 3, 4]))

    mol = S.parse_smiles(text)
    assert mol.GetNumAtoms() == cisplatin.GetNumAtoms(), "the parse did not hand back the atoms that were written"
    kept = K.stated_arrangement(mol)
    assert kept is not None, "the arrangement did not survive the parse at all"
    assert sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in kept[1].values()) == ["Cl", "Cl", "N", "N"]
    assert rx.canonical_smiles(rx.enumerate_isomers(mol)[0]) == text, "the string is not a fixed point"

    naive = Chem.MolFromSmiles(text)  # the default parse: same string, six hydrogens short
    assert naive.GetNumAtoms() < cisplatin.GetNumAtoms(), "RDKit kept the Hs; this fixture cannot show the keying"
    with pytest.raises(ValueError, match="round-tripping SMILES"):
        rx.canonical_smiles(rx.enumerate_isomers(naive)[0])


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_dative_smiles_round_trips_a_complex_smiles_cannot_write_naively():
    mol = _xyz_to_mol(_MN_H2)
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


def test_dative_smiles_raises_rather_than_return_an_unreadable_string():
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


@pytest.mark.parametrize(("kind", "pair"), _LEWIS_PAIRS.items(), ids=list(_LEWIS_PAIRS))
def test_one_species_gives_one_string_whichever_lewis_form_described_it(kind, pair):
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in pair]
    assert len({Chem.GetFormalCharge(m) for m in mols}) == 1, f"{kind}: the pair is not one species, fix the fixture"
    written = {S.dative_smiles(m) for m in mols}
    assert len(written) == 1, f"{kind}: two Lewis forms of one species gave {len(written)} strings: {written}"


def test_a_terminal_oxo_is_written_ionically_and_a_pi_face_is_left_alone():
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
    assert fe.count("[c-]") == 2, f"a Cp carbon beyond the two anionic ones was charged: {fe}"


def test_the_isomer_door_and_the_mol_door_write_one_constitution():
    for smi in (s for pair in (*_LEWIS_PAIRS.values(), *_PARTLY_FILLED.values()) for s in pair):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        isos = rx.enumerate_isomers(mol)
        assert isos, f"{smi} enumerated nothing"
        core = rx.canonical_smiles(isos[0]).split(" |", 1)[0]
        assert core == S.dative_smiles(mol), f"{smi}: the Isomer door wrote a different constitution"


# --- the arrangement the SMILES grammar cannot say -------------------------------------------------------


def test_the_canonical_string_separates_a_chiral_centre_from_its_mirror():
    smi, flipped = "[Pt](F)(F)(Cl)(Cl)(Br)Br", [_CIS3[v] for v in (0, 1, 2, 3, 5, 4)]  # the Br pair exchanged
    hands = [_isomer(smi, "octahedral", seating) for seating in (_CIS3, flipped)]
    assert {i.chirality for i in hands} == {"delta", "lambda"}, [i.chirality for i in hands]
    one, other = (rx.canonical_smiles(i) for i in hands)
    assert one.split("|")[0] == other.split("|")[0], "the constitution is the same molecule"
    assert one != other, "the two hands share a canonical string"


@pytest.mark.parametrize(
    ("smi", "geometry"),
    [
        ("[NH3]->[Pt](<-[NH3])(Cl)Cl", "square_planar"),  # cis / trans
        ("[Pt](F)(F)(Cl)(Cl)(Br)Br", "octahedral"),  # MA2B2C2: six isomers, one delta / lambda pair
        ("Br[Pd]1(Cl)NCCN1", "square_planar"),  # a chelate, so a bite edge is in the fold
        ("[CH2]=[CH2].Cl[Pt](Cl)Cl", "square_planar"),  # Zeise: an eta2 face is one vertex
    ],
    ids=["MA2B2", "MA2B2C2", "chelate", "eta2"],
)
def test_every_enumerated_isomer_reads_back_as_itself(smi, geometry):
    isos = rx.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles(smi)), geometry)
    assert len(isos) >= 1
    for iso in isos:
        text = rx.canonical_smiles(iso)
        back = rx.enumerate_isomers(S.parse_smiles(text))
        assert len(back) == 1, f"{iso.label}: its own string enumerated {len(back)} isomers"
        got = back[0]
        assert rx.canonical_smiles(got) == text, f"{iso.label}: the string is not a fixed point"
        assert got.geometry == iso.geometry, f"{iso.label}: came back as {got.geometry}"
        assert got.chirality == iso.chirality, f"{iso.label}: {iso.chirality!r} came back {got.chirality!r}"
        was, now = _seated(iso), _seated(got)
        assert any([was[q[v]] for v in range(len(was))] == now for q in rotation_group(iso.geometry)), (
            f"{iso.label}: {was} and {now} are not the same arrangement under any rotation of the template"
        )


def test_a_stated_arrangement_refuses_an_argument_with_nothing_left_to_do():
    text = rx.canonical_smiles(_isomer("[Pt](F)(F)(F)(Cl)(Cl)Cl", "octahedral", _MA3B3_SEATS["fac"]))
    mol = S.parse_smiles(text)
    assert len(rx.enumerate_isomers(mol, "OCT")) == 1, "naming the shape the string states is not a contradiction"
    for kwargs in ({"geometry": "trigonal_prismatic"}, {"fix": [1, 2]}):
        with pytest.raises(ValueError, match="nothing to act on"):
            rx.enumerate_isomers(mol, **kwargs)


def test_the_canonical_string_refuses_what_it_cannot_state():
    iso = _isomer("[Pt](F)(F)(Cl)Cl", "square_planar", _MA2B2_SEATS["cis"])
    rw = Chem.RWMol(iso.mol)
    rw.AddAtom(Chem.Atom(46))  # a spectator Pd bonded to nothing: still a second centre to state
    iso.mol, iso.extra = rw.GetMol(), [(rw.GetNumAtoms() - 1, 46, 0)]
    with pytest.raises(NotImplementedError, match="one metal centre"):
        rx.canonical_smiles(iso)
    bare = K.from_surrogate(Chem.MolFromSmiles("[Pt](F)(F)(Cl)Cl"), [(0, 78, 0)], [])
    with pytest.raises(ValueError, match="no polyhedron template"):
        rx.canonical_smiles(bare)
