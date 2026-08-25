"""Test native signed volumes and rxembed orientation over non-tetrahedral geometry matrices."""

from __future__ import annotations

import importlib
import itertools
from collections import Counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed import metal_core as metal
from rxembed.metal_isomers import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, orientation_parity, point_group

emb = importlib.import_module("rxembed.embed")

_MATRIX_CONFS = 32
_RX_CONFS = 8


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
