"""`identity/stereoisomers.py`: the ligand stereoisomers a graph has under a `stereo=` request."""

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.graph import metal_indices
from rxembed.identity.stereoisomers import StereoRequest, enumerate_unassigned, unassigned_centres
from tests.geometry_fixtures import embedded
from tests.metal_fixtures import BINAP_PDCL2

# Pt(II) with one bound carboxylate and both amines bound, its amine hands specified
ONE_ARM_BOUND_PT_CX = (
    "O=C([O-])C[N@@]1(CC(=O)O)CC[N@]2(CC(=O)O)CC(=O)[O-]->[Pt+2]<-1<-2<-[Cl-] "
    "|atomProp:4.atomNote.s0:11.atomNote.s1:19.atomNote.s2:20.atomNote.SPL:21.atomNote.s3|"
)


def _with_metals(smiles):
    """Return the Mol of `smiles` and its metals, as a coordination caller passes them."""
    mol = Chem.MolFromSmiles(smiles)
    return mol, set(metal_indices(mol))


def test_stereo_request_parse_reads_a_mode_per_kind_and_atom():
    """Unset, a graph enumerates and a geometry keeps, its bound diene re-enumerated; 'all' keeps; 'invert' mirrors
    every hand but keeps E/Z. Each row: the graph and the geometry mode tables, the locked mode, mirror, selectors."""
    kinds = ("point", "ez", "axial", "planar", "helical", "diene")
    unset, kept, mirrored = (dict.fromkeys(kinds, mode) for mode in ("unassigned", "preserve", "invert"))
    for given, graph, geometry, locked, mirror, selectors in (
        (None, unset, kept | {"diene": "racemic"}, None, False, {}),
        ("all", kept, kept | {"diene": "racemic"}, None, False, {}),
        ("invert", mirrored | {"ez": "preserve"}, mirrored | {"ez": "preserve"}, None, True, {}),
        (
            {"point": "racemic", "locked": "free", "N5": "invert"},
            unset | {"point": "racemic", "diene": "free"},
            kept | {"point": "racemic", "diene": "free"},
            "free",
            False,
            {"N5": ("N", 5, "invert")},
        ),
    ):
        request = StereoRequest.parse(given)
        shown = (request.graph, request.geometry, request.locked, request.mirror, request.selectors)
        assert shown == (graph, geometry, locked, mirror, selectors), given


def test_stereo_request_parse_refuses_a_request_it_cannot_take():
    for given, one_species, message in (
        ("bogus", False, "unknown stereo mode"),
        ({"point": "unassigned"}, False, "unknown stereo mode"),
        ({"chirality": "free"}, False, "unknown stereo mode"),
        ("racemic", True, r"enumerates stereoisomers; embed rx\.isomers"),
        ("all", True, r"enumerates stereoisomers; embed rx\.isomers"),
    ):
        with pytest.raises(ValueError, match=message):
            StereoRequest.parse(given, one_species=one_species)


def test_selector_naming_no_configurable_centre_is_refused():
    """C1 of 1-bromo-1-chloropropane's SMILES is its stereocentre: O1 and C99 name no atom, C3 no centre, and the
    methyl C0 of a bound ethylmethylamine none either."""
    for source, selector, message in (
        ("Cl[C@H](Br)CC", "O1", "is not an atom in this molecule"),
        ("Cl[C@H](Br)CC", "C99", "is not an atom in this molecule"),
        ("Cl[C@H](Br)CC", "C3", "not configurable point stereocentres"),
        ("CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "C0", "not configurable point stereocentres"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.isomers(source, stereo={selector: "racemic"})


def test_unassigned_centres_names_each_open_element_by_its_kind():
    """A cumulene's first bond has no E/Z of its own; coordination locks a chelate imine and equivalent donor arms;
    only a chelate locks a biaryl axis, and a five-ring bipyridine chelate holds none."""
    for smiles, kinds in (
        ("CC=C=CC", ["cumulene"]),
        ("CC=C=C(C)C=CC", ["cumulene", "ez"]),
        ("CC=CC(O)CC", ["ez", "point"]),
        ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", ["ez"]),
        (r"C/C1=N/[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1", []),
        ("CN12->[Rh+](<-[I-])(<-[C-]#[O+])<-P3(C)CN(CN(C1)C3)C2", []),
        (BINAP_PDCL2, ["axis"]),
        ("[Cl-]->[Pd+2](<-[Cl-])(<-N)<-P(c1ccccc1)(c1ccccc1)c1ccccc1-c1ccccc1P(c1ccccc1)c1ccccc1", []),
        ("[Cl-]->[Pd+2]1(<-[Cl-])<-n2ccccc2-c2cccc[n]->12", []),
    ):
        mol, metals = _with_metals(smiles)
        assert sorted(element.kind for element in unassigned_centres(mol, metals)) == kinds, smiles


def test_enumerate_unassigned_counts_elements_and_kept_variants():
    """Each row: (variants, elements expanded, variants before dropping duplicates and the cap, elements left open)."""
    pentenamine = Chem.MolFromSmiles("CC=CC(N)O")
    every_bond = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in pentenamine.GetBonds()}
    for name, given, choices, counts in (
        ("mapped-meso-butanediol", "[CH3:1][CH:2]([OH:5])[CH:3]([OH:6])[CH3:4]", {}, (3, 2, 4, 0)),
        ("pentenamine-bonds-skipped", pentenamine, {"skip_bonds": every_bond}, (2, 1, 2, 0)),
        ("pentenamine-centre-skipped", pentenamine, {"skip_points": {3}}, (2, 1, 2, 0)),
        ("pentanetriol-ends-skipped", "CC(O)C(O)C(C)O", {"skip_points": {1, 5}}, (1, 0, 1, 0)),
        ("quinine-bridgehead-nitrogen", "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", {}, (16, 4, 16, 0)),
        ("six-centre-heptane-capped", "CC(O)C(N)C(F)C(Cl)C(Br)C(I)C", {}, (32, 6, 64, 0)),
        ("pentadiene-allene", "CC=C=CC", {}, (1, 1, 2, 1)),
    ):
        mol = Chem.MolFromSmiles(given) if isinstance(given, str) else given
        expansion = enumerate_unassigned(mol, **choices)
        assert (len(expansion.variants), *expansion[1:]) == counts, name


def test_enumerate_unassigned_labels_variants_by_their_stereo():
    """A bound monodentate imine and a side-on alkene keep both E/Z, a chelate-locked imine neither; R,S and S,R
    amines on one palladium are one complex."""
    for smiles, labels in (
        ("CP(CC)c1ccccc1", ["P1:R", "P1:S"]),
        ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", ["C1=N2:E", "C1=N2:Z"]),
        (r"C/C1=N/[NH]->[Ni+2](<-[Cl-])(<-[Cl-])<-1", [""]),
        ("C[CH]1=[CH](F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", ["C1=C2:E", "C1=C2:Z"]),
        ("CC[CH](C)N->[Pd+2](<-[Cl-])(<-[Cl-])<-N[CH](C)CC", ["C2:R,C9:R", "C2:S,C9:R", "C2:S,C9:S"]),
    ):
        mol, metals = _with_metals(smiles)
        variants = enumerate_unassigned(mol, exclude=metals).variants
        assert sorted(str(variant.stereo) for variant in variants) == labels, smiles


def test_stereo_request_lists_the_ligand_stereoisomers_it_keeps():
    """A conformer is reference geometry: it fills open stereo, and 'preserve' and 'invert' keep or mirror it."""
    for source, stereo, labels in (
        ("CC(O)C=CC", None, ["C1:R,C3=C4:E", "C1:R,C3=C4:Z", "C1:S,C3=C4:E", "C1:S,C3=C4:Z"]),
        ("CC(O)C=CC", {"ez": "free"}, ["C1:R", "C1:S"]),
        ("CC(O)C=CC", "free", [""]),
        ("CC(O)C=CC", {"default": "free"}, [""]),
        ("C[C@H](O)[C@@H](C)N", "racemic", ["C1:R,C3:R", "C1:R,C3:S", "C1:S,C3:R", "C1:S,C3:S"]),
        (r"C/C=C/[CH2-]", "racemic", ["C1=C2:E", "C1=C2:Z"]),
        (r"C/C=C/[O-]->[Cu+]<-N", "racemic", ["C1=C2:E", "C1=C2:Z"]),
        ("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|", "racemic", ["C7-N8:M", "C7-N8:P"]),
        ("CC(O)C(O)C(C)O", None, ["C1:R,C3:r,C5:S", "C1:R,C3:s,C5:S", "C1:R,C5:R", "C1:S,C5:S"]),
        ("C[C@@H](O)C(O)[C@@H](C)O", None, ["C1:R,C5:R"]),
        ("C[C@@H](O)C(O)[C@H](C)O |o1:1|", None, ["C1:R,C3:r,C5:S", "C1:R,C3:s,C5:S"]),
        ("Cl[C@H](Br)CC", {"C1": "invert"}, ["C1:S"]),
        ("Cl[C@H](Br)CC", {"C1": "racemic"}, ["C1:R", "C1:S"]),
        (embedded("C[C@@H](O)CC"), "invert", ["C1:S"]),
        (embedded("C[C@@H](O)[C@H](C)O"), "invert", ["C1:S,C3:R"]),
        (embedded("C[C@H]1CC[C@@H](C)CC1"), "invert", ["C1:s,C4:s"]),
        (ONE_ARM_BOUND_PT_CX, None, ["N4:R,N11:S"]),
        (ONE_ARM_BOUND_PT_CX, {"locked": "racemic"}, ["N4:R,N11:R", "N4:R,N11:S", "N4:S,N11:R", "N4:S,N11:S"]),
        (ONE_ARM_BOUND_PT_CX, {"locked": "free"}, [""]),
        ("CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", {"locked": "racemic"}, ["N2:R", "N2:S"]),
    ):
        found = sorted(rx.identity(species).stereo for species in rx.isomers(source, stereo=stereo))
        assert found == labels, (source if isinstance(source, str) else Chem.MolToSmiles(source), stereo)


def test_stereo_request_mirrors_a_bound_amine_by_its_conformer():
    """The record decides the inverted hand, so no conformer reads the specified R back."""
    amine = rx.embed("CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", n=1, seed=1).mol
    inverted = rx.isomers(amine, stereo={"point": "invert"})

    assert [str(record.species.stereo) for record in inverted.records] == ["N2:S"]


def test_a_stereo_request_no_variant_keeps_is_refused():
    """Butan-2-ol has no hand to preserve, so every enumerated hand breaks the request."""
    with pytest.raises(ValueError, match="has no configuration compatible"):
        rx.isomers("CC(O)CC", stereo="preserve")


def test_enumerated_alkene_specifies_its_ez_to_rdkits_writer():
    for species in rx.isomers("CC(O)C=CC"):
        assert [rx.cxsmiles(back) for back in rx.isomers(Chem.MolToSmiles(species))] == [rx.cxsmiles(species)]
