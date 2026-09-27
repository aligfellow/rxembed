"""Test metal-centre handedness perception and selection."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import metal_stereo as metal
from rxembed.metal_isomer import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, orientation_parity, point_group

emb = importlib.import_module("rxembed.embed")

_ACAC = "CC(=O)C=C([O-])C"


def test_donor_classes_ignore_resonance_form():
    mol = Chem.MolFromSmiles(_ACAC)
    oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    perceived = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    assert perceived[oxygens[0]] != perceived[oxygens[1]], "fixture no longer distinguishes resonance forms"
    every = list(range(mol.GetNumAtoms()))
    canonical = metal.donor_classes(mol, every)
    assert canonical[oxygens[0]] == canonical[oxygens[1]]
    assert all(perceived[a] != perceived[b] or canonical[a] == canonical[b] for a in every for b in every)


@pytest.mark.parametrize(
    ("smiles", "donors", "equal_pairs", "unequal_pairs"),
    [
        (r"F/C=N/C.F/C=N\C", [2, 6], [], [(2, 6)]),
        ("CS(C)=O.C[S+](C)[O-]", [1, 5], [(1, 5)], []),
        ("[NH-]C(=[NH2+])N", [0, 2, 3], [(2, 3)], [(0, 2)]),
    ],
    ids=["flat-symmetry-ligand-stereo", "same-sulfoxide-drawn-in-its-two-lewis-forms", "charge-separated-resonance"],
)
def test_donor_classes_distinguish_or_merge_by_resonance_and_stereo(smiles, donors, equal_pairs, unequal_pairs):
    mol = Chem.MolFromSmiles(smiles)
    classes = metal.donor_classes(mol, donors)

    for a, b in equal_pairs:
        assert classes[a] == classes[b]
    for a, b in unequal_pairs:
        assert classes[a] != classes[b]


def test_site_identity_restores_carriers_before_removing_donor_hydrogens():
    from rxembed.metal_core import surrogate_all_metals

    donors = [1, 6]
    for tag, equivalent in (("@", True), ("@@", False)):
        mol = Chem.AddHs(rx.parse_smiles(f"C[N@](CC)([H])->[Cu+]<-[N{tag}](C)([H])CC", remove_hs=False))
        full = metal.donor_classes(mol, donors)
        assert (full[1] == full[6]) is equivalent
        base, identities = surrogate_all_metals(mol)
        before = base.ToBinary()
        roles = [(donor, *identities[0]) for donor in donors]

        classes = metal.site_classes(base, donors, coordination=roles)

        assert (classes[1] == classes[6]) is equivalent
        assert base.ToBinary() == before


def test_site_classes_preserve_inequivalent_stereo_roots():
    mol = Chem.MolFromSmiles("N[C@H](F)[C@H](F)[C@@H](F)[C@H](F)N")
    roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    classes = metal.site_classes(mol, roots)
    assert classes[roots[0]] != classes[roots[1]]

    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    new_roots = [atom.GetIdx() for atom in reordered.GetAtoms() if atom.GetAtomicNum() == 7]
    repeated_classes = metal.site_classes(reordered, new_roots)
    assert repeated_classes[new_roots[0]] != repeated_classes[new_roots[1]]


def test_site_classes_use_the_same_rooted_resonance_proof():
    mol = Chem.MolFromSmiles(_ACAC)
    oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    classes = metal.site_classes(mol, [100, 101], {100: (oxygens[0],), 101: (oxygens[1],)})

    assert classes[100] == classes[101]


_TRIP = "c2c(C(C)C)cc(C(C)C)cc2C(C)C"  # 2,4,6-triisopropylphenyl


@pytest.mark.parametrize("phosphine", ["[P](C)(C)C", f"[P]({_TRIP})({_TRIP}){_TRIP}"], ids=["PMe3", "PTrip3"])
def test_kappa2_acetate_oxygens_stay_equivalent_beside_a_bulky_phosphine(phosphine):
    isomers = rx.metal(f"CC1=[O]->[Pd+2](<-[Cl-])(<-{phosphine})<-[O-]1", "SPL")

    assert len(isomers) == 1


@pytest.mark.parametrize(
    ("smiles", "want"),
    [
        (f"CC(=O)[O-]->[Pd+2](<-[P]({_TRIP})({_TRIP}){_TRIP})(<-[Cl-])<-O=C(C)[O-]", 2),
        (r"C/C=C/C#N->[Pd+2](<-[Cl-])(<-[Br-])<-N#C/C=C\C", 3),
        ("CP1(C)=[O]->[Pd+2](<-[Cl-])(<-[P](C)(C)C)<-[O-]1", 1),
        ("[O-]S1(=O)=[O]->[Pd+2](<-[Cl-])(<-[P](C)(C)C)<-[O-]1", 1),
    ],
    ids=[
        "mixed-lewis-form-kappa1-acetates",
        "crotononitrile-ez-stays-diastereomeric",
        "kappa2-phosphinate-expanded-octet",
        "kappa2-sulfate-expanded-octet",
    ],
)
def test_resonance_identity_case_table_isomer_counts(smiles, want):
    # The kappa1 acetates sit in two separate conjugated systems drawn in opposite Lewis forms and must
    # still merge into one ligand class. The crotononitriles carry E/Z stereo on a bond inside the merged
    # conjugated system and must stay diastereomeric, so the rule must not erase stated bond stereo. The
    # phosphinate and sulfate oxygens sit on a p-block centre RDKit does not perceive as conjugated and must
    # still merge, the same drawing choice as a carboxylate's.
    assert len(rx.metal(smiles, "SPL")) == want


def test_tetraphenylporphyrinato_nitrogens_merge_into_one_site_class():
    # meso-tetraphenylporphyrinato dianion, one Lewis form (two pyrrolide N-, two pyridine-type N): all four
    # donors sit in one 48-heavy-atom macrocyclic conjugated system, so they share one resonance identity.
    smiles = "c1ccc(cc1)-c1c2ccc([n-]2)c(-c2ccccc2)c2ccc(n2)c(-c2ccccc2)c2ccc([n-]2)c(-c2ccccc2)c2ccc1n2"
    mol = Chem.MolFromSmiles(smiles)
    nitrogens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    classes = metal.donor_classes(mol, nitrogens)

    assert len(set(classes.values())) == 1


def test_site_markers_do_not_suppress_dithiocarbamate_resonance():
    mol = Chem.MolFromSmiles("CN(C)C(=S)[S-]")
    sulfurs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16]
    donors = metal.donor_classes(mol, sulfurs)
    sites = metal.site_classes(mol, sulfurs)

    assert donors[sulfurs[0]] == donors[sulfurs[1]]
    assert sites[sulfurs[0]] == sites[sulfurs[1]]


def test_bis_dithiolene_lewis_forms_give_one_isomer_set():
    """A Mo bis-dithiolene chelate enumerates alike whether one wing is drawn dithiolate or dithione.

    Same connectivity, H counts and total charge either way; the metal absorbs the difference (Mo(VI) with
    two dithiolates vs Mo(IV) with one dithiolate and one neutral dithione). Resonance identity ignores
    charge, so both readings give the two chemically identical ligands the same class and the same
    ligand-exchange symmetry.
    """
    dithiolate = "[Mo+6]12(<-[Cl-])(<-[Br-])(<-[S-]C=C[S-]->1)<-[S-]C=C[S-]->2"
    mixed = "[Mo+4]12(<-[Cl-])(<-[Br-])(<-[S-]C=C[S-]->1)<-S=CC=S->2"
    isos_a, isos_b = rx.metal(dithiolate, "OCT"), rx.metal(mixed, "OCT")

    assert len(isos_a) == len(isos_b)
    for isos in (isos_a, isos_b):
        iso = isos[0]
        classes = metal.site_classes(iso.graph, iso.donors, coordination=iso.roles)
        sulfurs = [d for d in iso.donors if iso.graph.GetAtomWithIdx(d).GetAtomicNum() == 16]
        assert len({classes[s] for s in sulfurs}) == 1, "all four dithiolene sulfurs must be one site class"


@pytest.mark.parametrize("sites", [[23, 2, 18, 12, 6], [23, 2, 6, 12, 18]])
def test_tc_oxo_amine_oxime_hand_ignores_the_oxime_hydrogen_bond_contact(sites):
    """MOCQIE's O-H~O contact closes no ring and ranks no atom, so the embedded geometry reads back its hand."""
    contact = rx.parse_smiles(
        "CC1=[N]2O[H]~[O-][N]3=C(C)C(C)(C)[N-]4C[C@H](C#N)C[N-](C1(C)C)->[Tc+5]<-2<-3<-4<-[O-2] |Z:4|"
    )
    rw = Chem.RWMol(contact)
    rw.RemoveBond(4, 5)
    free = rw.GetMol()
    free.UpdatePropertyCache(strict=False)

    assert Isomer(contact, "SPY", sites).chirality == Isomer(free, "SPY", sites).chirality != ""


def test_face_winding_abstains_at_the_plane_and_is_scale_invariant():
    face_mol = Chem.MolFromSmiles("[c-]1(F)c(Br)ccc1")
    rw = Chem.RWMol(Chem.CombineMols(face_mol, Chem.MolFromSmiles("[Fe+2]")))
    metal_idx = rw.GetNumAtoms() - 1
    face = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetIsAromatic()]
    for atom in face:
        rw.AddBond(atom, metal_idx, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    pos = np.zeros((mol.GetNumAtoms(), 3))
    theta = 2 * np.pi * np.arange(len(face)) / len(face)
    pos[face, 0], pos[face, 1] = np.cos(theta), np.sin(theta)
    pos[metal_idx] = (2.0, 0.0, 0.0)

    assert metal.face_winding(mol, pos, metal_idx, face, metal.donor_classes(mol, face)) == ""

    eta2 = Chem.AddHs(rx.parse_smiles(r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"))
    metal_idx, face = 4, (1, 2)
    donors = [
        bond.GetBeginAtomIdx()
        for bond in eta2.GetBonds()
        if bond.GetBondType() == Chem.BondType.DATIVE and bond.GetEndAtomIdx() == metal_idx
    ]
    pos = np.zeros((eta2.GetNumAtoms(), 3))
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
    ranks = metal.donor_classes(eta2, donors)
    cip = list(Chem.ComputeAtomCIPRanks(eta2))

    assert {metal.face_winding(eta2, pos * scale, metal_idx, face, ranks, cip) for scale in (1e-3, 1.0, 1e3)} == {"-"}


def test_routine_hydrogen_is_removed_when_its_bond_defines_imine_stereo():
    mol = Chem.MolFromSmiles("[H]/N=C(/C)F")

    reduced, mapping = metal.remove_routine_hydrogens(mol)

    assert Chem.MolToSmiles(reduced) == "CC(=N)F"
    assert mapping == {1: 0, 2: 1, 3: 2, 4: 3}


def test_equivalent_site_assignments_preserve_links_and_vacancy():
    links = {frozenset((0, 1)): 2}
    assignments = list(metal.equivalent_site_assignments(["N", "N", "N", None], links))

    assert assignments == [{0: 0, 1: 1, 2: 2}, {0: 1, 1: 0, 2: 2}]
    assert all(3 not in assignment and 3 not in assignment.values() for assignment in assignments)


def _orientation_parity(mol, cid, iso):
    """Fit every observed vertex to the ideal shape and return proper (+1) or mirrored (-1)."""
    pos = mol.GetConformer(int(cid)).GetPositions()

    def point(atom):
        ring = iso.haptic.get(atom)
        return np.mean(pos[list(ring)], axis=0) if ring else pos[atom]

    observed = np.asarray([point(atom) - pos[iso.metal] for atom in iso.vertices])
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    ideal = np.asarray(POLYHEDRA[iso.geometry].vertex_dirs, float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    return orientation_parity(observed, ideal)


def test_orientation_parity_covers_every_full_rank_polyhedron_symmetry():
    for name, poly in POLYHEDRA.items():
        dirs = np.asarray(poly.vertex_dirs, float)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        if np.linalg.matrix_rank(dirs) < 3:
            continue
        proper, improper = point_group(poly.vertex_dirs)
        for parity, permutations in ((1, proper), (-1, improper)):
            for q in permutations:
                assert orientation_parity(dirs[list(q)], dirs) == parity, (name, q)


def test_orientation_reader_skips_achiral_incomplete_and_collapsed_centres():
    iso = rx.metal("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", "square_planar")[0]
    assert metal.realised_chirality(iso.mol, 0, iso.geometry, iso.vertices, iso.metal, iso.chirality, iso.haptic) == ""

    incomplete = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]", "octahedral")[0]
    assert not incomplete.chirality
    assert (
        metal.realised_chirality(
            incomplete.mol,
            0,
            incomplete.geometry,
            incomplete.vertices,
            incomplete.metal,
            incomplete.chirality,
            incomplete.haptic,
        )
        == ""
    )

    chiral = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "octahedral")[0]
    mol = Chem.Mol(chiral.mol)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    assert (
        metal.realised_chirality(
            mol, 0, chiral.geometry, chiral.vertices, chiral.metal, chiral.chirality, chiral.haptic
        )
        == ""
    )


def test_chiral_ligand_filters_wrong_metal_hands_before_uff(monkeypatch):
    smiles = "O->[Co+3](<-[Cl-])(<-[CH3-])(<-[NH2][C@H](C)CC)(<-[F-])<-P"
    expected = [x for x in Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles), includeUnassigned=True) if x[1] != "?"]
    iso = rx.metal(smiles, "octahedral")[0]
    calls = []
    native = emb.seed_coordinates

    def counted(*args, **kwargs):
        calls.append(args[2])
        ids = list(native(*args, **kwargs))
        if len(calls) == 1:  # make the first batch entirely wrong-handed to exercise the retry
            mol = args[0]
            for cid in ids:
                if _orientation_parity(mol, cid, iso) == 1:
                    emb._reflect(mol, cid)
        return ids

    monkeypatch.setattr(emb, "seed_coordinates", counted)
    conformers = emb.embed(iso, n=8, params=rx.EmbedParams(seed=7, prune_rms=-1))
    assert len(conformers) == 8
    assert calls[0] > 8, "a mirror-unsafe ligand must sample both DG hands before selecting one"
    assert len(calls) > 1, "an all-wrong first batch must retry before cleanup"
    assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}
    for cid in conformers.ids:
        one = Chem.Mol(conformers.mol, False, int(cid))
        one.GetAtomWithIdx(expected[0][0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistryFrom3D(one, confId=one.GetConformer().GetId(), replaceExistingTags=True)
        assert [x for x in Chem.FindMolChiralCenters(one, includeUnassigned=True) if x[1] != "?"] == expected


def _distinct_donor_isomer(geometry):
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(92))
    donors = []
    for isotope in range(14, 14 + POLYHEDRA[geometry].cn):
        donor = Chem.Atom(7)
        donor.SetIsotope(isotope)
        index = rw.AddAtom(donor)
        for _ in range(3):
            rw.AddBond(index, rw.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
        rw.AddBond(index, metal, Chem.BondType.DATIVE)
        donors.append(index)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Isomer(mol, geometry, donors)


def test_post_dg_gate_is_geometry_derived_for_every_nonplanar_polyhedron():
    for geometry, polyhedron in POLYHEDRA.items():
        if polyhedron.planar:
            continue
        iso = _distinct_donor_isomer(geometry)
        assert iso.chirality in {"delta", "lambda"}, geometry
        conformers = emb.embed(iso, n=2, params=rx.EmbedParams(seed=7, prune_rms=-1))
        assert len(conformers) == 2, geometry
        assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}, geometry
