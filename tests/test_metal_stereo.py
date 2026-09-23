"""Test metal-centre handedness perception and selection."""

from __future__ import annotations

import importlib
import itertools
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed import metal_stereo as metal
from rxembed.metal_isomer import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, orientation_parity, point_group

emb = importlib.import_module("rxembed.embed")

_MATRIX_CONFS = 32
_RX_CONFS = 8
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
        ("CS(C)=O.C[S+](C)[O-]", [1, 5], [], [(1, 5)]),
        ("[NH-]C(=[NH2+])N", [0, 2, 3], [(2, 3)], [(0, 2)]),
    ],
    ids=["flat-symmetry-ligand-stereo", "unrelated-sulfur-forms", "charge-separated-resonance"],
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


def test_site_classes_preserve_inequivalent_stereo_roots_and_reuse_the_proof(monkeypatch):
    from rxembed import utils

    monkeypatch.setattr(utils, "_RESONANCE_CACHE", {})
    mol = Chem.MolFromSmiles("N[C@H](F)[C@H](F)[C@@H](F)[C@H](F)N")
    roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    classes = metal.site_classes(mol, roots)
    assert classes[roots[0]] != classes[roots[1]]

    def repeated(*args, **kwargs):
        raise AssertionError("re-enumerated an unchanged resonance proof")

    monkeypatch.setattr(Chem, "ResonanceMolSupplier", repeated)
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    new_roots = [atom.GetIdx() for atom in reordered.GetAtoms() if atom.GetAtomicNum() == 7]
    repeated_classes = metal.site_classes(reordered, new_roots)
    assert repeated_classes[new_roots[0]] != repeated_classes[new_roots[1]]


def test_site_classes_use_the_same_rooted_resonance_proof():
    mol = Chem.MolFromSmiles(_ACAC)
    oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    classes = metal.site_classes(mol, [100, 101], {100: (oxygens[0],), 101: (oxygens[1],)})

    assert classes[100] == classes[101]


def test_large_resonance_identity_has_a_bounded_proof(monkeypatch):
    mol = Chem.MolFromSmiles("C" * 41)
    seen = []

    def bounded(*args, **kwargs):
        seen.append(kwargs["max_forms"])
        return False, True

    monkeypatch.setattr(metal, "resonance_match", bounded)
    assert metal._root_resonance_match(mol, 0, 1) == (False, True)

    assert seen == [8]


def test_large_site_identity_does_not_run_full_molecule_resonance(monkeypatch):
    mol = Chem.MolFromSmiles("C" * 40 + "C(=O)[O-]")
    oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]

    def forbidden(*args, **kwargs):
        raise AssertionError("large site identity must not enumerate whole-molecule resonance forms")

    monkeypatch.setattr(metal, "_root_resonance_match", forbidden)
    classes = metal._root_classes(mol, oxygens)

    assert len(classes) == 2
    assert classes[oxygens[0]] != classes[oxygens[1]]


def test_site_markers_do_not_suppress_dithiocarbamate_resonance():
    mol = Chem.MolFromSmiles("CN(C)C(=S)[S-]")
    sulfurs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16]
    donors = metal.donor_classes(mol, sulfurs)
    sites = metal.site_classes(mol, sulfurs)

    assert donors[sulfurs[0]] == donors[sulfurs[1]]
    assert sites[sulfurs[0]] == sites[sulfurs[1]]


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


def _native_embed(smiles, n=_MATRIX_CONFS):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 7
        params.pruneRmsThresh = -1
        ids = list(rdDistGeom.EmbedMultipleConfs(mol, n, params))
    assert len(ids) == n, smiles
    return mol, ids


def _center_and_neighbors(mol):
    center = next(a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() >= 4)
    return center, [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()]


def _longest_pairs(mol, neighbors, n):
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    return sorted(itertools.combinations(neighbors, 2), key=lambda q: bm[max(q)][min(q)], reverse=True)[:n]


def _volume_signs(mol, ids, atoms):
    return {int(np.sign(_signed_volume(*(mol.GetConformer(cid).GetPositions()[a] for a in atoms)))) for cid in ids}


def _signed_volume(p1, p2, p3, p4):
    return float((p1 - p4) @ np.cross(p2 - p4, p3 - p4))


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
    u, _s, vt = np.linalg.svd(ideal.T @ observed)
    return int(np.sign(np.linalg.det(u @ vt)))


def test_rdkit_native_tetrahedral_tags_build_signed_chiral_sets():
    for token, tag, sign in (
        ("@", Chem.ChiralType.CHI_TETRAHEDRAL_CCW, 1),
        ("@@", Chem.ChiralType.CHI_TETRAHEDRAL_CW, -1),
    ):
        mol, ids = _native_embed(f"N[C{token}](F)(Cl)Br")
        center, neighbors = _center_and_neighbors(mol)
        assert mol.GetAtomWithIdx(center).GetChiralTag() == tag
        assert _volume_signs(mol, ids, neighbors) == {sign}


def test_rdkit_native_non_tetrahedral_tag_matrix_has_no_signed_volume():
    for permutation in range(1, 4):
        mol, ids = _native_embed(f"Cl[Pt@SP{permutation}]([35Cl])([36Cl])[37Cl]")
        _center, neighbors = _center_and_neighbors(mol)
        assert _volume_signs(mol, ids, neighbors) == {-1, 1}, f"SP{permutation} unexpectedly stayed planar"

    for permutation in range(1, 21):
        mol, ids = _native_embed(f"Cl[Pt@TB{permutation}]([35Cl])([36Cl])([37Cl])[38Cl]")
        center, neighbors = _center_and_neighbors(mol)
        axial = tuple(sorted(_longest_pairs(mol, neighbors, 1)[0]))
        equatorial = sorted(set(neighbors) - set(axial))
        atoms = (axial[0], equatorial[0], equatorial[1], center)
        assert _volume_signs(mol, ids, atoms) == {-1, 1}, f"TB{permutation} unexpectedly selected one hand"

    for permutation in range(1, 31):
        mol, ids = _native_embed(f"Cl[Th@OH{permutation}]([35Cl])([36Cl])([37Cl])([38Cl])[39Cl]")
        center, neighbors = _center_and_neighbors(mol)
        trans = sorted(tuple(sorted(q)) for q in _longest_pairs(mol, neighbors, 3))
        atoms = (*(q[0] for q in trans), center)
        assert _volume_signs(mol, ids, atoms) == {-1, 1}, f"OH{permutation} unexpectedly selected one hand"


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


def test_rxembed_post_dg_gate_covers_every_chiral_candidate_before_cleanup():
    fixtures = (
        ("seesaw", "O->[Fe+2](<-[Cl-])(<-[CH3-])<-N"),
        ("trigonal_bipyramidal", "O->[Fe+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]"),
        ("square_pyramidal", "O->[Fe+3](<-[Cl-])(<-[CH3-])(<-N)<-[F-]"),
        ("octahedral", "O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P"),
    )
    tested = Counter()
    for geometry, smiles in fixtures:
        for iso in rx.metal(smiles, geometry):
            assert iso.chirality in {"delta", "lambda"}
            conformers = emb.embed(iso, n=_RX_CONFS, seed=7, prune_rms=-1)
            assert len(conformers) == _RX_CONFS
            assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}
            tested[geometry] += len(conformers)
    assert tested == {
        "seesaw": 12 * _RX_CONFS,
        "trigonal_bipyramidal": 20 * _RX_CONFS,
        "square_pyramidal": 30 * _RX_CONFS,
        "octahedral": 30 * _RX_CONFS,
    }


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
    conformers = emb.embed(iso, n=8, seed=7, prune_rms=-1)
    assert len(conformers) == 8
    assert calls[0] > 8, "a mirror-unsafe ligand must sample both DG hands before selecting one"
    assert len(calls) > 1, "an all-wrong first batch must retry before cleanup"
    assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}
    for cid in conformers.ids:
        one = Chem.Mol(conformers.mol, False, int(cid))
        one.GetAtomWithIdx(expected[0][0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistryFrom3D(one, confId=one.GetConformer().GetId(), replaceExistingTags=True)
        assert [x for x in Chem.FindMolChiralCenters(one, includeUnassigned=True) if x[1] != "?"] == expected


def test_free_hand_reflection_preserves_distances_and_flips_chirality():
    iso = rx.metal("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", "octahedral")[0]
    conformers = emb.embed(iso, n=1, seed=7, prune_rms=-1)
    mol, cid = conformers._mol, conformers.ids[0]
    before = np.asarray(Chem.Get3DDistanceMatrix(mol, confId=cid))
    parity = _orientation_parity(mol, cid, iso)

    emb._reflect(mol, cid)

    assert np.allclose(Chem.Get3DDistanceMatrix(mol, confId=cid), before)
    assert _orientation_parity(mol, cid, iso) == -parity


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
        conformers = emb.embed(iso, n=2, seed=7, prune_rms=-1)
        assert len(conformers) == 2, geometry
        assert {_orientation_parity(conformers._mol, cid, iso) for cid in conformers.ids} == {1}, geometry
