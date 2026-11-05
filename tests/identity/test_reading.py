"""`identity/reading.py`: which species a Mol is or can be, whether its conformer agrees, and its name.

A species is specified by its notes and tags, and a conformer shows one. `rx.isomers` and `rx.embed` follow the
specification and take a disagreeing conformer as input geometry; `rx.minimize` cannot change identity by relaxing, so
it refuses a disagreement it cannot relax away; the describe verbs name what the conformer shows and refuse a
disagreement with the specification.
"""

import re
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.identity.reading import Specification, named_species, observed_labels, read_species
from rxembed.identity.smiles_io import mol_from_species
from rxembed.identity.species import Species
from rxembed.utils import mirror_tag
from tests.geometry_fixtures import embedded, placed
from tests.metal_fixtures import (
    FERROCENE_TWO_FACES,
    ISOPRENE_FE_CO3,
    PT2CL4_AMMINES,
    berry_intermediate,
    bridged_dimer,
    enumerated,
    on_its_vertices,
    zinc_copies,
)

PD_AMMINE = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"  # N0 Pd1 Cl2 Cl3 N4
ZINC = "[F-]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-]"
ATROP = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"  # an N-aryl pyrrole whose marker specifies the M axis
AMINE_ZINC = "[Cl-]->[Zn+2](<-[Br-])(<-[I-])<-N[C@@H](C)CC"  # Cl0 Zn1 Br2 I3 N4, its (S) carbon C5
BIS_ETHYLMETHYLAMINE_PT = "CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[N@@H](C)CC"


def _one(ensemble):
    """Return the first conformer of an embedded or minimized ensemble as its own Mol."""
    return Chem.Mol(ensemble.mol, False, int(ensemble.ids[0]))


@cache
def _pd(label):
    """Return the cis or trans PdCl2(NH3)2 species, embedded once."""
    return _one(rx.embed(rx.isomers(PD_AMMINE, geometry="SPL").select(label=label), n=1, seed=1))


@cache
def _embedded(source, hand=None, geometry=None):
    """Return one embedded conformer of `source`'s species with `hand`, else its first, embedded once."""
    found = rx.isomers(source, geometry=geometry)
    return _one(rx.embed(found.select(hand=hand) if hand else found[0], n=1, seed=1))


def _mirrored(mol, *, tags=False):
    """Return `mol` with its conformer reflected through x = 0, and its point tags too when `tags`."""
    out = Chem.Mol(mol)
    out.GetConformer().SetPositions(out.GetConformer().GetPositions() * [-1, 1, 1])
    for atom in out.GetAtoms() if tags else ():
        atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
    return out


def _without(mol, *, notes=True, tags=False):
    """Return a copy of `mol` without its metal notes, and without its point tags when `tags`."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if notes:
            atom.ClearProp("atomNote")
        if tags:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out


def _with_conformer(specified, geometry):
    """Return explicit-H `specified` carrying the conformer of `geometry`, matched on the graph without stereo."""
    specified = Chem.AddHs(specified)
    match = geometry.GetSubstructMatch(specified, useChirality=False)
    order = [0] * len(match)
    for specified_idx, geometry_idx in enumerate(match):
        order[geometry_idx] = specified_idx
    out = Chem.RenumberAtoms(specified, order)
    out.AddConformer(Chem.Conformer(geometry.GetConformer()), assignId=True)
    return out


def _carrying(mol, geometry):
    """Return a copy of `mol` carrying the conformer of `geometry`, which has the same atom order."""
    out = Chem.Mol(mol)
    out.AddConformer(Chem.Conformer(geometry.GetConformer()), assignId=True)
    return out


def _moved(mol, pairs):
    """Return a copy of `mol` whose atoms in each pair trade places in the conformer."""
    out = Chem.Mol(mol)
    pos = out.GetConformer().GetPositions()
    for left, right in pairs:
        pos[[left, right]] = pos[[right, left]]
    out.GetConformer().SetPositions(pos)
    return out


def _stacked(first, second, *, notes=True, tags=True):
    """Return the graph of `first` carrying its conformer and `second`'s as conformers 0 and 1."""
    mol = _without(first, notes=not notes, tags=not tags)
    mol.RemoveAllConformers()
    for cid, source in enumerate((first, second)):
        conf = Chem.Conformer(source.GetConformer())
        conf.SetId(cid)
        mol.AddConformer(conf, assignId=False)
    return mol


def _past_its_face(mol, centre, *, hydrogen_in_face=False):
    """Return a copy of `mol` with `centre` moved 0.15 A past a face of its neighbours, so it reads no hand."""
    out = Chem.Mol(mol)
    pos = out.GetConformer().GetPositions()
    neighbours = out.GetAtomWithIdx(centre).GetNeighbors()
    hydrogen = next(n.GetIdx() for n in neighbours if n.GetAtomicNum() == 1)
    heavy = [n.GetIdx() for n in neighbours if n.GetAtomicNum() > 1]
    face_atoms, apex = ([hydrogen, *heavy[:2]], heavy[2]) if hydrogen_in_face else (heavy, hydrogen)
    face = pos[face_atoms].mean(axis=0)
    pos[centre] = face + 0.15 * (face - pos[apex]) / np.linalg.norm(face - pos[apex])
    out.GetConformer().SetPositions(pos)
    return out


def _ammines_swapped(cis):
    """Return cis-PdCl2(NH3)2 with its two ammines, hydrogens and all, exchanged in the conformer."""
    plain = Chem.Mol(cis)
    plain.RemoveAllConformers()
    order = next(
        order
        for order in plain.GetSubstructMatches(plain, uniquify=False, useChirality=False)
        if order[0] == 4 and order[2] == 2 and order[3] == 3
    )
    swapped = Chem.Mol(cis)
    swapped.GetConformer().SetPositions(cis.GetConformer().GetPositions()[list(order)])
    return swapped


def _trans_chloride(mol, cid=-1):
    """Return each ammine N of a PdCl2(NH3)2 conformer with the chloride trans to it."""
    conf = mol.GetConformer(int(cid))
    return {n: max((2, 3), key=lambda cl, n=n: rdMolTransforms.GetAngleDeg(conf, n, 1, cl)) for n in (0, 4)}


def _without_conformers(mol):
    """Return `mol` without its conformers: what it specifies."""
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    return out


@cache
def _representations():
    """Return, per species, its specified CX name and each representation of it a verb may be given."""
    cis = rx.isomers(PD_AMMINE, geometry="SPL").select(label="cis")
    text = rx.cxsmiles(cis)
    parsed = rx.parse_smiles(text, remove_hs=False)
    explicit = Chem.AddHs(parsed)
    butanol = Chem.MolFromSmiles("C[C@@H](O)CC")
    return {
        "cis-diammine-palladium": (
            text,
            {
                "cx": text,
                "parsed": parsed,
                "renumbered": Chem.RenumberAtoms(explicit, list(reversed(range(explicit.GetNumAtoms())))),
                "embedded": _pd("cis"),
                "observed": _without(_pd("cis")),
                "disagreeing": _with_conformer(parsed, _pd("trans")),
            },
        ),
        "r-butan-2-ol": (
            rx.cxsmiles(butanol),
            {
                "parsed": butanol,
                "embedded": embedded("C[C@@H](O)CC"),
                "observed": _without(embedded("C[C@@H](O)CC"), notes=False, tags=True),
                "disagreeing": _with_conformer(butanol, embedded("C[C@H](O)CC")),
            },
        ),
    }


VERBS = {
    "cxsmiles": rx.cxsmiles,
    "isomers": lambda source: [rx.cxsmiles(_without_conformers(one)) for one in rx.isomers(source)],
    "embed": lambda source: rx.cxsmiles(_one(rx.embed(source, n=1, seed=2))),
    "minimize": lambda source: rx.cxsmiles(_one(rx.minimize(source))),
}
# the verbs each representation is read by: a text is only enumerated, and only a conformer can be minimized
CELLS = {
    "cx": ("isomers",),
    "parsed": ("cxsmiles", "isomers", "embed"),
    "renumbered": ("cxsmiles", "isomers"),
    "embedded": tuple(VERBS),
    "observed": tuple(VERBS),
    "disagreeing": tuple(VERBS),
}
# a disagreeing conformer is refused by cxsmiles, and by minimize where relaxing into the specification inverts a hand
REFUSED = {
    ("cis-diammine-palladium", "disagreeing", "cxsmiles"): ValueError,
    ("r-butan-2-ol", "disagreeing", "cxsmiles"): ValueError,
    ("r-butan-2-ol", "disagreeing", "minimize"): rx.EmbeddingError,
}


def test_each_verb_reads_one_identity_from_each_representation():
    """A conformer that disagrees is input geometry to isomers and embed, and minimize relaxes it (`REFUSED`)."""
    failed = []
    for species, (specified, reps) in _representations().items():
        for rep, source in reps.items():
            for verb in CELLS[rep]:
                refused = REFUSED.get((species, rep, verb))
                try:
                    got = VERBS[verb](source)
                except Exception as error:  # every cell is reported, whatever it raised
                    if refused is None or not isinstance(error, refused):
                        failed.append(f"{species} {rep} -> {verb}: raised {error!r}")
                    continue
                if refused is not None or got != ([specified] if verb == "isomers" else specified):
                    failed.append(f"{species} {rep} -> {verb}: read {got!r}")
    assert not failed, "\n".join(failed)


def test_each_describe_verb_refuses_a_contradicting_conformer():
    mirrored = _mirrored(embedded("C[C@@H](O)CC"))
    for describe in (rx.identity, rx.cxsmiles, rx.smiles):
        with pytest.raises(ValueError, match="this Mol's conformer and its specified stereo disagree"):
            describe(mirrored)


def test_contradicting_conformer_is_refused_naming_the_miss():
    """A code that reads the same in the mirror, such as the s of a pseudoasymmetric carbon, is no miss."""
    fluoropropylamine = _embedded("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    imine = "C[C@H](F)/C=N(/[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-])C"
    silver = on_its_vertices(rx.isomers("N->[Ag+]<-N")[0], [(1, 0, 0), (-1, 0, 0)])
    pos = silver.GetConformer().GetPositions()
    pos[0] = pos[1]  # N0 on its silver
    silver.GetConformer().SetPositions(pos)
    collapsed = on_its_vertices(rx.isomers(ZINC, geometry="TET")[0])
    pos = collapsed.GetConformer().GetPositions()
    pos[0] = pos[1]  # F0 on its zinc: no ray, so no hand
    collapsed.GetConformer().SetPositions(pos)
    noted_amines = (
        "CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Br-])<-[N@@H](C)CC |atomProp:2.atomNote.s0:4.atomNote.SPL:5.atomNote.s2:"
        "6.atomNote.s1:7.atomNote.s3|"
    )
    for name, mol, message in (
        ("mirrored-r-butan-2-ol", _mirrored(embedded("C[C@@H](O)CC")), "specified C1 R but the conformer has S"),
        ("mirrored-marked-biaryl-axis", _mirrored(embedded(ATROP)), "specified C7-N8 M but the conformer has P"),
        ("mirrored-sec-butylpentanediol", _mirrored(embedded("C[C@@H](O)[C@@H]([C@@H](C)CC)[C@H](C)O")), "disagree"),
        ("mirrored Cu lactate, free ion", _mirrored(embedded("[Cu+].C[C@H](O)C(=O)[O-]")), "C2 S but"),
        ("butanol C past face", _past_its_face(embedded("C[C@H](O)CC"), 1, hydrogen_in_face=True), "C1 S but .* none"),
        ("amine C past its face", _past_its_face(fluoropropylamine, 1), "specified C1 S but the conformer has none"),
        (
            "mirrored imine against Pt",
            _mirrored(_with_conformer(Chem.MolFromSmiles(imine), _embedded(imine))),
            r"specified C\d+ S but the conformer has R",
        ),
        ("mirrored-delta-zinc", _mirrored(on_its_vertices(rx.isomers(ZINC, geometry="TET")[0])), "disagree"),
        ("mirrored PtClBr amines", _mirrored(_embedded(noted_amines)), "disagree"),
        ("ammine onto its Ag", silver, "the donors do not take the specified polyhedron"),
        ("F collapsed on delta Zn", collapsed, "the donors do not take the specified polyhedron"),
    ):
        with pytest.raises(ValueError, match=message) as refused:
            rx.cxsmiles(mol)
        assert not re.search(r"\b(\S+) but \1\b", str(refused.value)), name


def _isoprene_carbonyls_swapped():
    """Return Fe(CO)3(isoprene) with two carbonyls exchanged beside its prochiral face, then as embedded; no notes."""
    mol = _without(_embedded(ISOPRENE_FE_CO3))
    (c1, o1), (c2, o2), _third = [
        (atom.GetIdx(), oxygen.GetIdx())
        for atom in mol.GetAtoms()
        for oxygen in atom.GetNeighbors()
        if atom.GetSymbol() == "C" and oxygen.GetSymbol() == "O"
    ]
    return _moved(mol, [(c1, c2), (o1, o2)]), mol


def _mapped(mol):
    """Return a copy of `mol` with every atom mapped, as a reaction labels it."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return out


def test_species_has_one_identity_however_numbered_or_drawn():
    """Equivalent donors, bridges or metals trading places, or an achiral species' mirror image, are that species."""
    cis = _pd("cis")
    syn = bridged_dimer(PT2CL4_AMMINES, ((3.35, 1.6, 0.0), (3.35, -1.6, 0.0)))
    zinc = rx.isomers(zinc_copies((1, -1)))[0]  # its notes in the conformer's frame
    copies = Chem.GetMolFrags(zinc)
    like = _embedded("[Cl-]->[Zn+2](<-[Br-])(<-N[C@@H](C)CC)<-N[C@@H](C)CC", geometry="tetrahedral")
    free = rx.isomers("[Cl-]->[Zn+2](<-[Br-])(<-N[C@@H](C)CC)<-N[CH](C)CC", geometry="tetrahedral", stereo="free")
    labelled = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)CC"))
    labelled.GetAtomWithIdx(1).SetProp("react_atom_idx", "1")  # a reaction label on an atom is not the species
    phosphine = rx.parse_smiles("C[P@H](->[Pt](Cl)(Cl)Cl)CC")
    rows = [
        ("cis Pd Cl exchanged", _moved(cis, [(2, 3)]), cis),
        (
            "cis-palladium-renumbered",
            Chem.RenumberAtoms(cis, [int(i) for i in np.random.default_rng(1).permutation(cis.GetNumAtoms())]),
            cis,
        ),
        ("cis-palladium-mapped", _mapped(cis), cis),
        ("syn-dimer-bridges-exchanged", _moved(syn, [(3, 7)]), syn),
        ("isoprene Fe CO exchanged", *_isoprene_carbonyls_swapped()),
        ("unlike Zn copies exchanged", _moved(zinc, zip(copies[0], copies[1], strict=True)), zinc),
        ("R-butanol relabelled", Chem.RenumberAtoms(labelled, [4, 3, 2, 1, 0, *range(5, 15)]), labelled),
        ("unnoted Pt phosphine reversed", Chem.RenumberAtoms(phosphine, list(reversed(range(8)))), phosphine),
        *(
            (f"free-sec-butylamine-zinc-{k}-on-a-like-conformer", _carrying(one, like), like)
            for k, one in enumerate(free)
        ),
    ]
    for smiles in ("C[C@H]1CC[C@@H](C)CC1", "C[C@@H](O)[C@H](C)O"):
        rows.append((f"mirrored-{smiles}", _mirrored(embedded(smiles)), embedded(smiles)))
    rows.append(
        ("mirrored Pt amines", _mirrored(_embedded(BIS_ETHYLMETHYLAMINE_PT)), _embedded(BIS_ETHYLMETHYLAMINE_PT))
    )
    for name, changed, reference in rows:
        assert rx.identity(changed) == rx.identity(reference), name


def _two_pd_differing_only_in_alanine_hands():
    """Return an alanine-palladium and its mirror image side by side, read from coordinates alone."""
    one = _embedded("C[C@H](N)C(=O)[O-]->[Pd+2](<-[Cl-])(<-[Cl-])<-N")
    return _without(Chem.CombineMols(one, _mirrored(one), Point3D(14, 0, 0)), tags=True)


def test_identity_records_each_species_field_on_the_mols_atoms():
    """Ligand stereo, then each centre's polyhedron, cis/trans label, arrangement and rac/meso face pair."""
    meso_diene = embedded("C[C@@H](O)/C=C/[C@H](C)O")
    bromo_and_chloro_faces = FERROCENE_TWO_FACES.replace("(Br)", "(Cl)")
    unlike_faces = rx.isomers(bromo_and_chloro_faces, stereo="racemic")[0]
    propene = rx.isomers("[Cl-]->[Pt+2]12(<-[Cl-])(<-[CH2]=[CH](C)->1)<-[CH2]=[CH](C)->2", geometry="SPL")
    for name, mol, expected in (
        ("r-butan-2-ol", Chem.MolFromSmiles("C[C@@H](O)CC"), ("C1:R", (), (), (), ())),
        ("cis-palladium", _pd("cis"), ("", ("square_planar",), ("cis",), ("N0 N4 Cl2 Cl3",), ("",))),
        (
            "meso-propene-platinum",
            propene[1],
            ("", ("square_planar",), ("cis",), ("η2(re) η2(si) Cl0 Cl2",), ("meso",)),
        ),
        (
            "unnoted-diammine-copper",
            rx.parse_smiles("[NH3]->[Cu+]<-[NH3]"),
            ("", ("linear",), ("",), ("N0 N2",), ("",)),
        ),
        ("mirrored meso dienediol", _mirrored(meso_diene), ("C1:S,C5:R,C3=C4:E", (), (), (), ())),
        ("unlike planar faces", unlike_faces, ("", ("linear",), ("",), ("η5Rₚ@C8 η5Rₚ@C1",), ("",))),
        (
            "Pd told apart by hands",
            _two_pd_differing_only_in_alanine_hands(),
            ("C1:S,C20:R", ("square_planar",) * 2, ("cis", "cis"), ("O24 N28 Cl26 Cl27", "O5 N9 Cl7 Cl8"), ("", "")),
        ),
    ):
        named = rx.identity(mol)
        shown = (
            named.stereo,
            tuple(centre.geometry for centre in named.centres),
            named.labels,
            named.arrangements,
            named.haptic,
        )
        assert shown == expected, name


def test_identity_seats_each_centre_as_named_in_any_order():
    """Names read the seating their CX notes write, so a renumbered cobalt seats every donor where it did."""
    species = rx.isomers("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]", geometry="OCT")[0]
    seated = []
    for seed in range(3):
        order = [int(i) for i in np.random.default_rng(seed).permutation(species.GetNumAtoms())]
        (centre,) = rx.identity(Chem.RenumberAtoms(species, order)).centres
        seated.append([site and tuple(order[atom] for atom in site.atoms) for site in centre.vertices])

    assert seated.count(seated[0]) == len(seated)


def test_identity_of_a_conf_id_names_what_that_conformer_shows():
    """A note-free graph carrying a cis and a trans conformer names each by its own coordinates."""
    both = _stacked(_pd("cis"), _pd("trans"), notes=False)

    assert [rx.identity(both, cid).labels for cid in (0, 1)] == [("cis",), ("trans",)]


def test_conformers_reading_unlike_the_first_are_refused():
    """One record's constraints hold one seating, so equivalent ammines seated two ways are refused too."""
    cis, trans = _pd("cis"), _pd("trans")
    note_free, swapped = _stacked(cis, trans, notes=False), _ammines_swapped(cis)
    butanol = embedded("C[C@@H](O)CC")
    bent = Chem.Mol(cis)  # the donors on tetrahedral directions: the shape is part of what is read
    pos = cis.GetConformer().GetPositions()
    for direction, donor in zip(
        np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3), (0, 2, 3, 4), strict=True
    ):
        moved = [donor, *(h.GetIdx() for h in cis.GetAtomWithIdx(donor).GetNeighbors() if h.GetAtomicNum() == 1)]
        pos[moved] += pos[1] + np.linalg.norm(pos[donor] - pos[1]) * direction - pos[donor]
    bent.GetConformer().SetPositions(pos)
    another = r"conformer\(s\) 1 of this Mol show another species than conformer 0"
    otherwise = r"conformer\(s\) 1 of this Mol seat equivalent donors otherwise than conformer 0"
    for verb, mol, message in (
        (rx.minimize, note_free, another),
        (rx.embed, note_free, another),
        (lambda m: rx.isomers(m, geometry="SPL", observed=True), note_free, another),
        (rx.minimize, _stacked(cis, trans), another),
        (rx.isomers, _stacked(butanol, _mirrored(butanol)), another),
        (lambda m: rx.isomers(m, observed=True), _stacked(cis, bent), another),
        (rx.minimize, _stacked(cis, swapped), otherwise),
        (rx.minimize, _stacked(swapped, cis, notes=False), otherwise),
    ):
        with pytest.raises(ValueError, match=message):
            verb(mol)


def test_each_species_keeps_only_the_conformers_that_show_it():
    """An untagged racemate specifies no hand; mirror images of a meso diol are one species, whatever its tags say;
    carbonyls seated as mirror images beside a wound face hold the same compiled angles."""
    racemic = Chem.AddHs(Chem.MolFromSmiles("CC(O)CC"))
    rdDistGeom.EmbedMultipleConfs(racemic, 6, randomSeed=3)
    meso = embedded("C[C@@H](O)[C@H](C)O")
    rr = _stacked(meso, _mirrored(meso))
    rr.GetAtomWithIdx(3).SetChiralTag(mirror_tag(rr.GetAtomWithIdx(3).GetChiralTag()))  # (R,R) tags, meso conformers
    butanols = _stacked(embedded("C[C@@H](O)CC"), _mirrored(embedded("C[C@@H](O)CC")), tags=False)
    isoprene = _stacked(_embedded(ISOPRENE_FE_CO3), _isoprene_carbonyls_swapped()[0])
    minimized = rx.minimize(racemic)
    rows = (
        (
            "minimized-racemate-hands",
            {rx.identity(minimized.mol, cid).stereo for cid in minimized.ids},
            {"C1:R", "C1:S"},
        ),
        ("racemate-member-conformers", rx.isomers(racemic)[0].GetNumConformers(), 6),
        (
            "untagged meso mirrors",
            [m.GetNumConformers() for m in rx.isomers(_stacked(meso, _mirrored(meso), tags=False))],
            [2],
        ),
        ("r-r-tags-on-meso-conformers", [m.GetNumConformers() for m in rx.isomers(rr)], [2]),
        (
            "untagged R and S racemic",
            [m.GetNumConformers() for m in rx.isomers(butanols, stereo="racemic", observed=True)],
            [0, 0],
        ),
        ("isoprene Fe mirror seatings", [m.GetNumConformers() for m in rx.isomers(isoprene)], [2]),
    )
    assert {name: found for name, found, expected in rows} == {name: expected for name, found, expected in rows}


def test_species_carries_each_input_conformer_once_with_its_id():
    geometry = Chem.Mol(_pd("cis"))
    first = geometry.GetConformer()
    first.SetId(7)
    first.SetProp("origin", "first")
    second = Chem.Conformer(first)
    second.SetId(23)
    second.SetPositions(first.GetPositions() + np.array([3.25, -1.5, 0.25]))
    second.SetProp("origin", "second")
    geometry.AddConformer(second, assignId=False)
    before = geometry.ToBinary(Chem.PropertyPickleOptions.AllProps)

    (carried,) = rx.isomers(geometry)

    assert [(c.GetId(), c.GetProp("origin")) for c in carried.GetConformers()] == [(7, "first"), (23, "second")]
    assert all(
        np.array_equal(a.GetPositions(), b.GetPositions())
        for a, b in zip(carried.GetConformers(), geometry.GetConformers(), strict=True)
    )
    assert geometry.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_pins_seat_traded_equivalent_ammines_as_they_fit_best():
    """Unpinned, minimize keeps the seating the conformer shows; pins keyed as the notes seat the ammines win."""
    cis = _pd("cis")
    swapped = _ammines_swapped(cis)
    as_noted = {i: tuple(cis.GetConformer().GetPositions()[i]) for i in range(5)}
    beside = next(cl for cl in (2, 3) if cl != _trans_chloride(cis)[0])  # the chloride the notes seat cis to N0
    assert _trans_chloride(swapped) != _trans_chloride(cis)
    for name, fix, seating in (
        ("unpinned", None, swapped),
        ("sphere-pinned-as-noted", as_noted, cis),
        ("cis-pair-held-as-noted", {(0, beside): 3.1}, cis),
        ("cis-angle-held-as-noted", {(0, 1, beside): 90.0}, cis),
        ("sphere-pinned-where-it-stands", list(range(5)), swapped),
    ):
        out = rx.minimize(swapped, fix=fix)
        assert _trans_chloride(out.mol, out.ids[0]) == _trans_chloride(seating), name
    assert len(rx.isomers(swapped, fix=as_noted)) == 1


def test_isomers_and_embed_take_one_stereo_request_alike():
    """'invert' mirrors every hand, a metal's included, and keeps E/Z, which {'ez': 'invert'} flips."""
    alcohol = embedded("C[C@H](N)O")
    for name, source, stereo, expected in (
        ("r-aminoethanol-inverted", alcohol, "invert", ("C1:S", (), ())),
        (
            "S-butylamine Zn inverted",
            _embedded(AMINE_ZINC, geometry="TET"),
            "invert",
            ("C5:R", ("lambda",), ("N4 Cl0 Br2 I3",)),
        ),
        ("e-alkene-inverted-keeps-e", embedded("C/C=C/[C@H](N)O"), "invert", ("C3:S,C1=C2:E", (), ())),
        ("e-alkene-ez-inverted", embedded("C/C=C/[C@H](N)O"), {"ez": "invert"}, ("C3:R,C1=C2:Z", (), ())),
    ):
        listed = [rx.identity(one) for one in rx.isomers(source, stereo=stereo)]
        built = rx.identity(_one(rx.embed(source, n=1, seed=1, stereo=stereo)))
        shown = (built.stereo, tuple(centre.hand for centre in built.centres), built.arrangements)
        assert (listed, shown) == ([built], expected), name


def test_stereo_request_on_a_mirror_enumerates_around_its_spec():
    """A mirrored conformer is geometry only, so `stereo=` moves no hand the Mol specifies."""
    mol = _embedded(BIS_ETHYLMETHYLAMINE_PT)
    around = [
        [rx.cxsmiles(_without_conformers(one)) for one in rx.isomers(m, stereo={"N2": "free"})]
        for m in (mol, _mirrored(mol))
    ]

    assert around[0] == around[1]


def test_open_organic_centre_is_filled_from_the_conformer():
    """(2R)-butane-2,3-diol with C3 open, carrying a (2R,3S) conformer: every embedded conformer is that species."""
    diol = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)C(C)O"))
    diol.AddConformer(Chem.Conformer(embedded("C[C@@H](O)[C@H](C)O").GetConformer()), assignId=True)
    ens = rx.embed(diol, n=3, seed=1)

    assert {rx.identity(ens.mol, cid) for cid in ens.ids} == {rx.identity(diol)}


def _cadmium_near_tie():
    """Return a Cd fan species and its sphere just past the SPY/TBP crossover of the Berry path, the rest parked."""
    fans = enumerated(Chem.AddHs(rx.parse_smiles("[NH2]1CC[NH]2CC[NH2]->[Cd+2](<-[Cl-])(<-[Cl-])<-1<-2")), "SPY")
    iso = next(
        c
        for c in fans
        if c.centres[0].vertices[0].atoms == (3,) and {s.atoms[0] for s in c.centres[0].vertices[1::2]} == {0, 6}
    )
    ((metal, sites),) = iso.sites.items()
    rays = dict(zip(sites.vertices, 2.4 * berry_intermediate(0.32), strict=True))
    points = [
        rays.get(atom, (0, 0, 0) if atom == metal else (100.0 + atom, 0, 0)) for atom in range(iso.mol.GetNumAtoms())
    ]
    return iso, placed(Chem.Mol(iso.mol), points)


def test_accepted_conformer_writes_the_name_in_its_notes_frame():
    """Square pyramidal reads within the margin of the trigonal bipyramid it is closer to, so the noted SPY stands."""
    iso, sphere = _cadmium_near_tie()
    noted = mol_from_species(iso)
    noted.AddConformer(Chem.Conformer(sphere.GetConformer()), assignId=True)

    assert rx.cxsmiles(noted) == rx.cxsmiles(_without_conformers(noted))
    assert ".TBP:" in rx.cxsmiles(sphere)  # without notes the conformer reads its best fit


def test_specification_cites_each_centres_shape_on_its_conformer():
    """Accepted records cite this reading; a species without a conformer cites none."""
    iso, sphere = _cadmium_near_tie()
    placed_iso = replace(iso, mol=sphere)

    assert Specification(species=iso).inputs == {}
    assert {atom: reading.matches for atom, reading in Specification(species=placed_iso).inputs.items()} == {7: True}


def test_named_species_names_a_record_as_noted_not_measured():
    """A cis record carrying a trans conformer is still cis; a graph with bound metals but no centre is read once."""
    cis = read_species(rx.isomers(PD_AMMINE, geometry="SPL").select(label="cis"))
    carrying_trans = replace(cis, mol=placed(Chem.Mol(cis.mol), _pd("trans").GetConformer().GetPositions()))
    copper = read_species(rx.isomers("[NH3]->[Cu+]<-[NH3]")[0])
    for name, species, expected in (
        ("cis with trans conformer", carrying_trans, rx.identity(_pd("cis")).name),
        ("Cu graph, no centre", Species(mol=copper.mol), rx.identity(rx.parse_smiles("[NH3]->[Cu+]<-[NH3]")).name),
    ):
        assert named_species(species).name == expected, name


def test_geometry_reads_a_name_or_code_in_either_case_or_a_list():
    specified = rx.cxsmiles(rx.isomers("Cl[Pt](Cl)(N)N", geometry="SPL")[0])
    for geometry in ("SPL", "square_planar", "spl", ["spl"]):
        assert [rx.cxsmiles(one) for one in rx.isomers(specified, geometry=geometry)] == [specified], geometry


def test_cxsmiles_of_a_bare_mol_writes_only_what_it_specifies():
    for smiles in ("[Ag+].F/C=C/F", "[NH3]->[Cu+]<-[NH3]", PD_AMMINE):
        mol = rx.parse_smiles(smiles)
        assert rx.cxsmiles(mol) == rx.smiles(mol), smiles


def _twisted_trans_pd(degrees):
    """Return trans-Pd(NH3)2Cl2 with its two trans pairs tilted `degrees` apart toward a tetrahedron."""
    trans = rx.isomers(PD_AMMINE, geometry="SPL").select(label="trans")
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return on_its_vertices(trans, [(c, 0, s), (0, c, -s), (-c, 0, s), (0, -c, -s)])


def test_observed_labels_read_the_noted_or_best_fit_polyhedron():
    """Twisted 15.25 deg, trans-PdCl2(NH3)2 reads best as a seesaw and within the margin as its noted square plane."""
    twisted = _twisted_trans_pd(15.25)
    rows = [
        ("twisted trans Pd noted", twisted, ["trans"]),
        ("twisted Pd read as seesaw", _without(twisted), ["cis"]),
    ]
    for smiles, geometry in (
        ("[U]1(<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[CH2]=[CH2]->1", "OCT"),
        ("N->[Co+3](<-N)(<-N)(<-[Cl-])(<-[Cl-])<-[Cl-]", "OCT"),
        ("[F-]->[Re+7](<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[F-]", "PBP"),
    ):
        for species in rx.isomers(smiles, geometry=geometry):
            rows.append((rx.cxsmiles(species), _without(on_its_vertices(species)), list(rx.identity(species).labels)))
    for name, mol, labels in rows:
        metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in {"Pd", "U", "Co", "Re"})
        assert observed_labels(mol, metal, [0]) == labels, name


def test_observed_labels_refuse_a_note_naming_no_polyhedron():
    twisted = _twisted_trans_pd(0.0)
    twisted.GetAtomWithIdx(1).SetProp("atomNote", "XYZ")

    with pytest.raises(ValueError, match="not a polyhedron"):
        observed_labels(twisted, 1, [0])


def test_locked_imine_ez_reads_as_the_arrangement_not_a_label():
    """The pendant alkene keeps its E/Z; the chelate imines are locked; reading the conformer changes nothing."""
    smiles = r"C/C=C/C/N1=C(\F)C(/Cl)=N(/Br)->[Ni+2](<-[Cl-])(<-[I-])<-1"
    source = _embedded(smiles)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    species = read_species(rx.isomers(smiles)[0])

    shown = read_species(_without(source))

    assert shown.stereo.bonds == species.stereo.bonds == {frozenset((1, 2)): "E"}
    assert rx.cxsmiles(mol_from_species(shown)) == rx.cxsmiles(_without_conformers(source))
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_flipped_bridging_enolate_reads_its_own_double_bond():
    """A delocalised charge moves only across an E/Z its conformer shows, so each conformer is read afresh."""
    mol = Chem.Mol(_embedded("CC(/[O-]->[Pd+2](<-[Cl-])(<-N)<-N)=C/C(=O->[Pd+2](<-[Cl-])(<-N)<-N)C(F)(F)F"))
    bond = next(
        b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.GetBeginAtom().GetSymbol() == "C"
    )
    ends = [bond.GetBeginAtom(), bond.GetEndAtom()]
    left, right = (
        next(n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx() and n.GetSymbol() != "H")
        for a, b in (ends, ends[::-1])
    )
    conf = Chem.Conformer(mol.GetConformer())
    angle = rdMolTransforms.GetDihedralDeg(conf, left, ends[0].GetIdx(), ends[1].GetIdx(), right)
    rdMolTransforms.SetDihedralDeg(conf, left, ends[0].GetIdx(), ends[1].GetIdx(), right, angle + 180.0)
    flipped = mol.AddConformer(conf, assignId=True)

    assert rx.changes(mol, conf_id=mol.GetConformers()[0].GetId()).stereo is None
    assert "the conformer has" in rx.changes(mol, conf_id=flipped).stereo


def test_coordinate_seats_a_substrate_once_per_distinct_vacancy():
    """Only all-cis Co(NH3)2Cl2 has two distinct vacancies, trans to N and trans to Cl; an index or a list of them."""
    for species in rx.isomers("N->[Co+3](<-[Cl-])(<-[Cl-])<-N.O", geometry="OCT")[:2]:  # an achiral cis, a lambda
        seated = []
        for form in (species, Chem.RenumberAtoms(species, list(reversed(range(species.GetNumAtoms()))))):
            water = next(a.GetIdx() for a in form.GetAtoms() if a.GetSymbol() == "O")
            cobalt = next(a.GetIdx() for a in form.GetAtoms() if a.GetSymbol() == "Co")
            found = rx.isomers(form, coordinate=[water] if form is species else water)
            assert all(one.GetBondBetweenAtoms(water, cobalt).GetBondType() == Chem.BondType.DATIVE for one in found)
            seated.append([rx.cxsmiles(one) for one in found])
        assert len(set(seated[0])) == len(seated[0]) == (2 if rx.identity(species).centres[0].hand else 1)
        assert seated[1] == seated[0]


def test_coordinate_seats_each_smarts_match_in_a_lone_vacancy():
    found = rx.isomers("N->[Pd+2](<-[Cl-])<-[Cl-].O.O", geometry="square_planar", coordinate="[OX2]")

    assert {
        next(n.GetIdx() for n in one.GetAtomWithIdx(1).GetNeighbors() if n.GetSymbol() == "O") for one in found
    } == {4, 5}


def test_reading_refuses_a_request_naming_no_one_species():
    cis_in_a_trans_frame = Chem.Mol(rx.isomers(PD_AMMINE, geometry="SPL").select(label="cis"))
    cis_in_a_trans_frame.AddConformer(Chem.Conformer(_pd("trans").GetConformer()), assignId=True)
    core = rx.Template(_pd("trans").GetConformer().GetPositions(), {i: i for i in range(5)})
    swapped = _ammines_swapped(_pd("cis"))
    as_swapped = {i: tuple(swapped.GetConformer().GetPositions()[i]) for i in range(5)}
    specified = rx.cxsmiles(rx.isomers(PD_AMMINE, geometry="SPL")[0])
    for call, message in (
        (lambda: rx.identity(rx.parse_smiles(PD_AMMINE)), r"2 species .*rx\.isomers"),
        (
            lambda: rx.embed(embedded("C[C@H](N)O"), n=1, stereo={"point": "racemic"}),
            r"makes this 2 species; pick one with rx\.isomers",
        ),
        (lambda: rx.embed("CC(O)CC", n=1), r"this source is 2 stereoisomers; pick one"),
        (lambda: rx.isomers(specified, geometry="hexagon"), "unknown geometry 'hexagon'"),
        (lambda: rx.isomers(PD_AMMINE, geometry="all"), "pass a list of names"),
        (lambda: rx.isomers(PD_AMMINE, center="Pt"), "needs center= to name one of"),
        (lambda: rx.isomers(PT2CL4_AMMINES, center="Pt"), "name one by its atom index"),
        (
            lambda: rx.isomers(_without_conformers(_pd("cis")), fix=as_swapped),
            r"pins coordination-sphere atoms \[0, 2, 3, 4\] off the arrangement this source specifies; use observed",
        ),
        (lambda: rx.embed(cis_in_a_trans_frame, fix=core, n=1, seed=1), "observed=True"),
        (lambda: rx.isomers(PD_AMMINE + ".O", geometry="OCT", coordinate="[O"), "not a valid SMARTS"),
        (lambda: rx.isomers(PD_AMMINE + ".O", geometry="OCT", coordinate="[S]"), "matched no atoms"),
        (
            lambda: rx.isomers("N->[Pd+2]<-[Cl-].O.O", geometry="SPL", coordinate="[OX2]"),
            r"matched 2 atoms for 2 vacant site",
        ),
        (lambda: rx.isomers(PT2CL4_AMMINES + ".O", coordinate=10), "coordinate= seats one metal"),
        (lambda: rx.isomers(PD_AMMINE + ".O", geometry="OCT", coordinate=[5, 5]), "must be distinct"),
    ):
        with pytest.raises(ValueError, match=message):
            call()
