"""Test coordination and vacant-site constraint construction."""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_constraints as _metal_constraints
from rxembed import metal_core as _metal
from rxembed import stereo as _stereo
from rxembed.constraints import FIX_ANGLE_TOL, Constraints
from rxembed.metal_constraints import (
    _add_ligand_ez,
    _add_umbrella,
    _planar_bite_targets,
    _tetrahedral_cross_angle,
    coordination,
)
from rxembed.metal_core import HapticSite, MetalState, _bounds_matrix, classify_geometry
from rxembed.metal_donor_orient import _COPLANAR_CAP
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, _vertex_angle, point_group, vertex_dirs
from rxembed.metal_slots import _CHELATE_BITE, _chelate_bite_window
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.ensemble import Ensemble
from rxembed.relax import _ff_surrogate
from tests.metal_fixtures import ferrocene

_WINDOW = 9.0  # deg: the coordination angle window is ±8°; 9 is that plus slack
_NI_N_CY = (  # the same N-bound isomer on a Cy2P-arene backbone
    "O=C1[O-]->[Ni+2]2(<-[N-](c3ccccc3)C1c1ccccc1)<-[P](Cc1ccccc1[P]->2(C1CCCCC1)C1CCCCC1)(C1CCCCC1)C1CCCCC1"
)
_BIS_EN_CO = "Cl[Co]12(Cl)(NCCN1)NCCN2"  # two en chelates + 2 Cl: intra- and inter-ligand pairs on one metal
_TRIDENTATE_CD = "[NH2]1CC[NH]2CC[NH2]->[Cd+2](<-[Cl-])(<-[Cl-])<-1<-2"


def _realised(ens, iso, cid, i, j):
    """The vertex i-metal-vertex j angle realised in conformer `cid`."""
    pos = ens.mol.GetConformer(cid).GetPositions()
    return _vertex_angle(pos[iso.vertices[i]] - pos[iso.metal], pos[iso.vertices[j]] - pos[iso.metal])


def _two_centre_ensemble(primary, secondary):
    """Build two ideal, separated coordination spheres on one conformer."""
    rw = Chem.RWMol()
    states = []
    for geometry in (primary, secondary):
        metal = rw.AddAtom(Chem.Atom(6))
        donors = tuple(rw.AddAtom(Chem.Atom(7)) for _ in POLYHEDRA[geometry].vertex_dirs)
        states.append(MetalState(metal, 46, 2, geometry, donors))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for origin, state in zip((np.zeros(3), np.array([8.0, 0.0, 0.0])), states, strict=True):
        conf.SetAtomPosition(state.atom, Point3D(*origin))
        for donor, direction in zip(state.vertices, POLYHEDRA[state.geometry].vertex_dirs, strict=True):
            unit = np.asarray(direction, dtype=float)
            conf.SetAtomPosition(donor, Point3D(*(origin + 2.0 * unit / np.linalg.norm(unit))))
    mol.AddConformer(conf)
    iso = Isomer._from_state(mol, states)
    return Ensemble(mol, [0], iso=iso), iso


# --- the polyhedron angles are realised, not merely stated -----------------------------------------------

# One real complex per low-CN shape: a 14-electron T-shaped Rh(I) phosphine (its two P trans, Cl the stem) and
# Fe(CO)4 (the 16e d8 C2v sawhorse). The CN3 pyramid has its own test below: it is the shape that used to lose.
_SHAPE_CASES = [
    ("t_shape", "CP(C)(C)->[Rh](Cl)<-P(C)(C)C"),
    ("seesaw", "[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"),
]


@pytest.mark.parametrize(("geometry", "smiles"), _SHAPE_CASES, ids=[c[0] for c in _SHAPE_CASES])
def test_shape_is_realised_by_a_real_embed(geometry, smiles):
    iso = rx.metal(smiles, geometry).select(index=0)
    ens = rx.embed(iso, n=4).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        for i, j, target in POLYHEDRA[geometry].resolved_angles:
            got = _realised(ens, iso, cid, i, j)
            assert abs(got - target) <= _WINDOW, f"{geometry} vertices {i}-{j}: {got}° vs {target}°"
        found = classify_geometry(ens.mol, iso.metal, list(iso.vertices), cid)
        assert found == geometry, f"seated {geometry}, embedded a {found}"


# The donor sets a bare CN3 pyramid used to lose on (13/49 conformers survived; every other shape 100%). The
# ±8° D-M-D window is flat-bottomed, so a phosphine rode the wall to 117.5°; inside trigonal_planar's basin,
# 2.5° away, and `mechanisms.Umbrella` now holds the scale-free improper instead. Two fixtures: the trimethyl
# case is the cheap one, and PPh3's DG seed comes out exactly planar, so it proves the hold RE-FORMS a pyramid
# rather than only keeping one.
_PPH3 = "P(c1ccccc1)(c1ccccc1)c1ccccc1"


# PPh3 alone: its DG seed comes out exactly planar, so it proves the hold RE-FORMS a pyramid rather than
# only keeping one. The PMe3 case took the same branch from an already-pyramidal seed.
def test_requested_pyramid_survives_relax():
    smiles = f"c1ccccc1P(c1ccccc1)(c1ccccc1)->[Pt](<-{_PPH3})<-{_PPH3}"
    iso = rx.metal(smiles, "trigonal_pyramidal").select(index=0)
    ens = rx.embed(iso, n=1, seed=7).minimize()
    assert ens.n >= 1, "Pt(PPh3)3: no conformer survived the gates"
    got = [classify_geometry(ens.mol, iso.metal, list(iso.vertices), cid) for cid in ens.ids]
    assert got == ["trigonal_pyramidal"] * len(got), f"Pt(PPh3)3: requested a pyramid, got {got}"


def test_flattened_pyramid_is_rejected():
    ens = rx.embed("C[P](C)(C)[Fe]([P](C)(C)C)[P](C)(C)C", metal="TPY", n=4, seed=7)[0]
    iso, mol = ens.iso, ens._mol  # `_mol`: `.mol` hands back a metal-restored COPY, which the edits below lose
    verts = list(iso.vertices)
    assert ens.n >= 1
    assert [classify_geometry(mol, iso.metal, verts, c) for c in ens.ids] == ["trigonal_pyramidal"] * ens.n
    for cid in ens.ids:  # push the metal onto its donor plane: the geometry the warning exists to report
        conf = mol.GetConformer(cid)
        pos = conf.GetPositions()
        conf.SetAtomPosition(iso.metal, Point3D(*np.mean([pos[v] for v in verts], axis=0)))
    failures = ens._acceptance_failures()
    assert len(failures) == 1
    assert str(next(iter(failures))).startswith("coordination state")
    assert next(iter(failures.values())) == ens.ids


def test_secondary_planar_centre_is_checked():
    ens, iso = _two_centre_ensemble("tetrahedral", "square_planar")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    donor = secondary.vertices[0]
    conf.SetAtomPosition(donor, Point3D(*(pos[donor] + np.array([0.0, 0.0, 2.0]))))
    assert not ens._coordination_ok(0, iso), "the puckered secondary square plane passed the final gate"


def test_secondary_nonplanar_centre_flattening_is_rejected():
    ens, iso = _two_centre_ensemble("square_planar", "trigonal_pyramidal")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    conf.SetAtomPosition(secondary.atom, Point3D(*np.mean(pos[list(secondary.vertices)], axis=0)))
    assert not ens._coordination_ok(0, iso)


def test_one_haptic_site_does_not_define_a_nonplanar_state():
    ens, iso = _two_centre_ensemble("tetrahedral", "tetrahedral")
    secondary = iso.centres[1]
    face = HapticSite(tuple(secondary.vertices))
    sparse = secondary._replace(vertices=(face, None, None, None))
    iso = Isomer._from_state(iso.mol, (iso.centres[0], sparse))
    assert ens._coordination_ok(0, iso)


def test_planar_donor_keeps_rdkit_native_substituent_floor_during_relax():
    iso = rx.metal("[Pd+2](<-[n]1ccccc1)(<-[n]1ccccc1)(<-[Cl-])<-[Cl-]", "square_planar")[0]
    bounds = _bounds_matrix(iso.mol)
    ens = rx.embed(iso, n=1, seed=42)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()

    for donor in (d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "N"):
        left, right = sorted(neighbour.GetIdx() for neighbour in iso.mol.GetAtomWithIdx(donor).GetNeighbors())
        floor = float(bounds[right, left])
        assert iso.cons.floors[(left, right)] == pytest.approx(floor)
        assert np.linalg.norm(pos[left] - pos[right]) >= floor
    assert ens.check()[ens.ids[0]].ok()


def test_donor_orientation_walls_cannot_collapse_two_protons_together():
    iso = rx.metal("[NH2](c1ccccc1)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar")[0]
    donor = next(atom for atom in iso.mol.GetAtoms() if atom.GetAtomicNum() == 7)
    protons = sorted(neighbor.GetIdx() for neighbor in donor.GetNeighbors() if neighbor.GetAtomicNum() == 1)
    floor = float(_bounds_matrix(iso.mol)[protons[1], protons[0]])

    assert iso.cons.floors[tuple(protons)] == pytest.approx(floor)
    ens = rx.embed(iso, n=1, seed=42)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    # A floor is a one-sided wall, not a target: `>= floor` fails here by a measured 0.0035 A (seed 42,
    # deterministic) because the relax settles a hair inside the wall, within geom_check's own floor slop.
    # Not tightened to a strict floor; tightened only to the measured undershoot instead of the old 0.01 pad.
    assert np.linalg.norm(pos[protons[0]] - pos[protons[1]]) >= floor - 0.004
    assert ens.check()[ens.ids[0]].ok()


def test_tridentate_pyramid_finds_valid_requested_shape():
    iso = rx.metal("[Cu+]12<-n3ccccc3-c3cccc(n->13)-c1ccccn->21", "trigonal_pyramidal").select(index=0)
    ens = rx.embed(iso, n=4, seed=7)

    assert ens.n >= 1
    assert all(classify_geometry(ens.mol, iso.metal, list(iso.vertices), cid) == iso.geometry for cid in ens.ids)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("fixed", "first_present"),
    [
        ([1, 2, 5], True),
        ([1, 2, 5, 64], True),
        ([1, 2, 3, 5, 64], False),
    ],
    ids=["metal-fixed", "metal-donor-fixed", "complete-term-fixed"],
)
def test_partial_fix_drops_only_complete_carbonyl_terms(fixed, first_present):
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=fixed)[0]

    assert ((1, 64, 3) in iso.cons.angles) is first_present
    assert (1, 65, 4) in iso.cons.angles


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mnh_carbonyls_pass_the_relax_contract():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=[1, 2, 5])[0]
    ens = rx.embed(iso, n=1, seed=0xF00D)

    assert ens.ids
    assert not ens.unrelaxed
    conf = ens._mol.GetConformer(ens.ids[0])
    for atoms in ((1, 64, 3), (1, 65, 4)):
        assert ens.cons.angles[atoms] == (165.0, 180.0)
        assert GetAngleDeg(conf, *atoms) >= 165.0 - FIX_ANGLE_TOL
        pair = atoms[:2]
        value = float(np.linalg.norm(conf.GetPositions()[pair[0]] - conf.GetPositions()[pair[1]]))
        lo, hi = ens.cons.distances[pair]
        assert lo <= value <= hi


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_retained_geometry_uses_the_same_donor_orientation_compiler():
    from rxembed.metal_isomer import from_geometry

    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    mol = Chem.AddHs(rx.read_xyz(str(path)), addCoords=True)
    cons = from_geometry(mol, center="all").cons

    assert cons.angles[(1, 64, 3)] == (165.0, 180.0)
    assert cons.angles[(1, 65, 4)] == (165.0, 180.0)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_two_fixed_donors_do_not_own_their_angle_when_the_metal_is_free():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    isos = rx.metal(str(path), center="all", fix=[6, 64])

    assert any(any(key[1] == 1 and {key[0], key[2]} == {6, 64} for key in iso.cons.angles) for iso in isos)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mn_fe_state_passes_the_relax_contract():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mn-h2.xyz"
    reference = rx.read_xyz(str(path), metal_charges={0: 2, 1: 1})
    ens = rx.embed(reference, fix=[1, 5, 63, 64, 65, 66], stereo={"planar": "racemic"}, n=1, seed=0xF00D)

    assert ens.ids
    assert not ens.unrelaxed
    assert ens._metal_states() == {ens.ids[0]: True}
    assert ens.cons.angles[(1, 61, 3)] == (165.0, 180.0)
    assert ens.cons.angles[(1, 62, 4)] == (165.0, 180.0)
    conf = ens._mol.GetConformer(ens.ids[0])
    assert GetAngleDeg(conf, 1, 61, 3) >= 165.0 - FIX_ANGLE_TOL
    assert GetAngleDeg(conf, 1, 62, 4) >= 165.0 - FIX_ANGLE_TOL
    for pair in ((1, 61), (1, 62)):
        value = float(np.linalg.norm(conf.GetPositions()[pair[0]] - conf.GetPositions()[pair[1]]))
        lo, hi = ens.cons.distances[pair]
        assert lo <= value <= hi
    reference = reference.GetConformer().GetPositions()
    frozen = sorted(ens.cons.frozen)
    drift = max(
        abs(
            np.linalg.norm(reference[i] - reference[j])
            - np.linalg.norm(conf.GetPositions()[i] - conf.GetPositions()[j])
        )
        for i, j in itertools.combinations(frozen, 2)
    )
    assert drift < 1e-12
    assert not ens._scan_connectivity()


# --- the chelate bite comes from the backbone, not the polyhedron ---------------------------------------


def _planar_bite_case(geometry="trigonal_planar"):
    """Build a diamine bite and independent spectators on a model shell."""
    mol = Chem.MolFromSmiles("[Ni+2].NCCN.N.Cl.Cl")
    poly = POLYHEDRA[geometry]
    vertices = [1, 4, 5] if geometry == "trigonal_planar" else [6, 7, 1, 4, 5]
    pair = frozenset((vertices.index(1), vertices.index(4)))
    cons = Constraints(metals={0}, distances={(0, donor): (1.9, 2.1) for donor in vertices})
    for i, j in itertools.combinations(range(len(vertices)), 2):
        angle = _vertex_angle(poly.vertex_dirs[i], poly.vertex_dirs[j])
        cons.angles[vertices[i], 0, vertices[j]] = (max(0.0, angle - 8.0), min(180.0, angle + 8.0))
    cons.angles[1, 0, 4] = (75.0, 91.0)
    return mol, vertices, poly, {pair: (75.0, 91.0)}, cons


def _planar_fan_case():
    """Build two connected bites with independent spectators on a square pyramid."""
    mol = Chem.MolFromSmiles("[Co].NCCNCCN.F.Cl")
    poly = POLYHEDRA["square_pyramidal"]
    vertices = [4, 8, 1, 9, 7]
    bites = {frozenset((0, 2)): (83.0, 91.0), frozenset((0, 4)): (85.0, 93.0)}
    cons = Constraints(
        metals={0},
        distances={(0, atom): (2.1, 2.2) for atom in vertices},
        angles={
            (vertices[i], 0, vertices[j]): (angle - 8.0, min(180.0, angle + 8.0))
            for i, j, angle in poly.resolved_angles
        },
    )
    for pair, window in bites.items():
        i, j = sorted(pair)
        cons.angles[vertices[i], 0, vertices[j]] = window
    cons.angles[1, 0, 7] = (150.0, 180.0)
    return mol, vertices, poly, bites, cons


def _planar_opposed_case(second_bite=True, geometry="square_planar"):
    """Build a narrow bite opposite another bite or independent spectators."""
    mol = Chem.MolFromSmiles("[Ag+].[O-]C[O-]." + ("NCCN" if second_bite else "N.C.C.N") + ".F.Cl")
    poly = POLYHEDRA[geometry]
    vertices = [1, 3, 4, 7] if geometry == "square_planar" else [1, 4, 3, 7, 8, 9]
    bites = {frozenset((vertices.index(1), vertices.index(3))): (60.0, 69.0)}
    if second_bite:
        bites[frozenset((vertices.index(4), vertices.index(7)))] = (70.0, 91.0)
    cons = Constraints(metals={0}, distances={(0, donor): (1.9, 2.1) for donor in vertices})
    for i, j in itertools.combinations(range(len(vertices)), 2):
        angle = _vertex_angle(poly.vertex_dirs[i], poly.vertex_dirs[j])
        cons.angles[vertices[i], 0, vertices[j]] = bites.get(
            frozenset((i, j)), (max(0.0, angle - 8.0), min(180.0, angle + 8.0))
        )
    return mol, vertices, poly, bites, cons


@pytest.mark.parametrize("second_bite", [False, True])
@pytest.mark.parametrize("geometry", ["square_planar", "octahedral"])
def test_opposed_bite_targets_preserve_windows_and_share_one_planar_shell(second_bite, geometry):
    mol, vertices, poly, bites, cons = _planar_opposed_case(second_bite, geometry)
    original = cons.copy()
    _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
    assert cons.copy(angles=original.angles, pulls=original.pulls) == original
    assert len(cons.pulls) == len(vertices) * (len(vertices) - 1) // 2
    gram = np.eye(len(vertices))
    for i, j in itertools.combinations(range(len(vertices)), 2):
        key = (vertices[i], 0, vertices[j])
        target = cons.pulls[min(key, key[::-1])]
        lo, hi = cons.angles[key]
        assert lo - 1e-8 <= target <= hi + 1e-8
        assert hi - lo == pytest.approx(original.angles[key][1] - original.angles[key][0])
        if {vertices[i], vertices[j]} not in ({1, 7}, {3, 4}):
            assert cons.angles[key] == original.angles[key]
        gram[i, j] = gram[j, i] = np.cos(np.radians(target))
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert np.linalg.matrix_rank(gram) == (2 if geometry == "square_planar" else 3)
    assert cons.pulls[1, 0, 3] == pytest.approx(69.0)
    assert cons.pulls[4, 0, 7] == pytest.approx(85.0)


def test_opposed_bite_targets_abstain_for_linked_held_or_incompatible_shells():
    mol, vertices, poly, bites, original = _planar_opposed_case()
    linked = Chem.RWMol(mol)
    linked.AddBond(3, 4, Chem.BondType.SINGLE)
    for source, blocked in ((linked.GetMol(), ()), (mol, {2})):
        cons = original.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, source, blocked)
        assert cons == original
    for field, value in (
        ("fixed", {(1, 0, 4): (180.0, 180.0)}),
        ("frozen", {1}),
        ("contacts", (frozenset(), frozenset({(1, 0, 4)}))),
        ("haptic", {1: (2, 3)}),
        ("angles", {key: value for key, value in original.angles.items() if key != (1, 0, 4)}),
        ("angles", {**original.angles, (1, 0, 4): (180.0, 180.0)}),
    ):
        cons = original.copy(**{field: value})
        before = cons.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
        assert cons == before


def test_held_opposed_bites_retain_a_valid_shared_angle_witness():
    # Opposed bites need no shared bite value: the ring closure fixes cis/trans jointly from BOTH bite
    # windows (metal_constraints.coordination's opposed-bite branch), not from one shared `180 - bite`.
    # Mutating back to that per-bite formula, or to the old silent `if lo <= hi` empty-intersection
    # fallthrough, breaks this witness on these elongated (O/S kappa2-corrected) legs.
    iso = Isomer(Chem.MolFromSmiles("[Ag+]12(<-[O-]C[O-]->1)<-NCCN->2"), "square_planar", {0: 1, 1: 3, 2: 4, 3: 7})
    cons = coordination(iso.mol, iso.metal, iso.vertices, iso.geometry, 47, haptic={}, frozen={2})
    windows = {frozenset((a, b)): value for (a, m, b), value in cons.angles.items() if m == iso.metal}
    left, right = windows[frozenset((1, 3))], windows[frozenset((4, 7))]
    # b1, b2: the closest-approaching point in each bite's own window (minimising |b1 - b2|), not a shared value.
    if left[1] < right[0]:
        b1, b2 = left[1], right[0]
    elif right[1] < left[0]:
        b1, b2 = left[0], right[1]
    else:
        b1 = b2 = (max(left[0], right[0]) + min(left[1], right[1])) / 2
    c = (360 - b1 - b2) / 2
    azimuths = dict(zip(iso.vertices, (0, b1, b1 + c, b1 + c + b2), strict=True))
    for pair, (lo, hi) in windows.items():
        a, b = pair
        angle = abs(azimuths[a] - azimuths[b])
        angle = min(angle, 360 - angle)
        assert lo <= angle <= hi
    assert not any(len(key) == 3 for key in cons.pulls)


def test_planar_bite_accepts_a_haptic_spectator_without_reorienting_its_face():
    _, vertices, poly, bites, baseline = _planar_bite_case()
    mol = Chem.MolFromSmiles("[Ni+2].NCCN.c1ccccc1")
    dummy = mol.GetNumAtoms()

    def rename(atom):
        return dummy if atom == 5 else atom

    vertices = [rename(atom) for atom in vertices]
    cons = baseline.copy(
        distances={tuple(rename(i) for i in key): value for key, value in baseline.distances.items()},
        angles={tuple(rename(i) for i in key): value for key, value in baseline.angles.items()},
        haptic={dummy: tuple(range(5, dummy))},
    )
    original = cons.copy()
    _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
    assert cons.copy(angles=original.angles, pulls=original.pulls) == original
    assert cons.pulls[1, 0, 4] == pytest.approx(91.0)
    assert cons.pulls[1, 0, dummy] == cons.pulls[4, 0, dummy] == pytest.approx(134.5)


def test_planar_fan_targets_share_one_shell_without_changing_bite_or_radial_windows():
    mol, vertices, poly, bites, cons = _planar_fan_case()
    original = cons.copy()
    _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
    assert cons.copy(angles=original.angles, pulls=original.pulls) == original
    assert {min(key, key[::-1]) for key in cons.angles} <= cons.pulls.keys()
    gram = np.eye(len(vertices))
    for i, j in itertools.combinations(range(len(vertices)), 2):
        key = (vertices[i], 0, vertices[j])
        target = cons.pulls[min(key, key[::-1])]
        lo, hi = cons.angles[key]
        assert lo - 1e-8 <= target <= hi + 1e-8
        assert hi - lo == pytest.approx(original.angles[key][1] - original.angles[key][0])
        gram[i, j] = gram[j, i] = np.cos(np.radians(target))
        if frozenset((i, j)) in bites:
            assert cons.angles[key] == original.angles[key]
            assert target == pytest.approx(np.mean(bites[frozenset((i, j))]))
    # Positive-semidefinite rank three is an independent common-ray check, not a ligand-pose certificate.
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert np.linalg.matrix_rank(gram) == 3
    assert cons.pulls[8, 0, 9] == pytest.approx(180.0)
    assert cons.pulls[1, 0, 7] == pytest.approx(176.0)
    for spectator in (8, 9):
        for donor in (1, 4, 7):
            assert cons.pulls[min((donor, 0, spectator), (spectator, 0, donor))] == pytest.approx(90.0)


def test_planar_fan_abstains_for_linked_spectators_explicit_authority_and_wrong_shape():
    mol, vertices, poly, bites, original = _planar_fan_case()
    for field, value in (
        ("fixed", {(1, 0, 8): (100.0, 100.0)}),
        ("frozen", {1}),
        ("contacts", (frozenset(), frozenset({(1, 0, 8)}))),
        ("haptic", {1: (2, 3)}),
        ("angles", {key: value for key, value in original.angles.items() if key != (8, 0, 9)}),
    ):
        cons = original.copy(**{field: value})
        before = cons.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
        assert cons == before
    for edge in ((8, 9), (7, 8)):
        linked = Chem.RWMol(mol)
        linked.AddBond(*edge, Chem.BondType.SINGLE)
        cons = original.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, linked.GetMol(), ())
        assert cons == original
    for slots, template, windows in (
        ([1, 4, 7], POLYHEDRA["t_shape"], {frozenset((0, 1)): (83.0, 91.0), frozenset((1, 2)): (83.0, 91.0)}),
        (
            [8, 9, 4, 1, 7],
            POLYHEDRA["trigonal_bipyramidal"],
            {frozenset((2, 3)): (83.0, 91.0), frozenset((2, 4)): (83.0, 91.0)},
        ),
    ):
        cons = original.copy()
        _planar_bite_targets(cons, slots, template, windows, 0, mol, ())
        assert cons == original


def test_planar_fan_rejects_degenerate_rays_and_reversed_orientation(monkeypatch):
    mol, vertices, poly, bites, original = _planar_fan_case()
    degenerate = list(poly.vertex_dirs)
    degenerate[4] = (0.0, 0.0, -1.0)
    cons = original.copy()
    _planar_bite_targets(cons, vertices, replace(poly, vertex_dirs=tuple(degenerate)), bites, 0, mol, ())
    assert cons == original
    monkeypatch.setitem(_planar_bite_targets.__globals__, "orientation_parity", lambda *_args: -1)
    cons = original.copy()
    _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
    assert cons == original


@pytest.mark.parametrize("geometry", ["trigonal_planar", "trigonal_bipyramidal"])
def test_planar_bite_windows_and_preferences_share_one_shell(geometry):
    mol, vertices, poly, bites, cons = _planar_bite_case(geometry)
    original = cons.copy()
    _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
    assert cons.copy(angles=original.angles, pulls=original.pulls) == original
    assert cons.angles[1, 0, 4] == (75.0, 91.0)
    assert (1, 0, 4) in cons.pulls
    assert cons.pulls[1, 0, 4] == pytest.approx(91.0)
    for key in ((1, 0, 5), (4, 0, 5)):
        assert cons.pulls[key] == pytest.approx(134.5)
        assert sum(cons.angles[key]) / 2 == pytest.approx(cons.pulls[key])
        assert cons.angles[key][1] - cons.angles[key][0] == pytest.approx(16.0)
    assert sum(cons.pulls[key] for key in ((1, 0, 4), (1, 0, 5), (4, 0, 5))) == pytest.approx(360.0)
    assert all(lo <= cons.pulls[min(key, key[::-1])] <= hi for key, (lo, hi) in cons.angles.items())


def test_tridentate_normal_arm_keeps_a_common_shell_through_cleanup(tmp_path):
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "trigonal_bipyramidal")
        if 0 in candidate.vertices[:2] and {3, 6} <= set(candidate.vertices[2:])
    )
    cons = iso.cons
    targets = {key: value for key, value in cons.pulls.items() if len(key) == 3}
    assert len(targets) == 10
    for key, value in targets.items():
        lo, hi = cons.angles[key] if key in cons.angles else cons.angles[key[::-1]]
        assert lo <= value <= hi
    equatorial = set(iso.vertices[2:])
    assert sum(value for (a, _, b), value in targets.items() if {a, b} <= equatorial) == pytest.approx(360.0)
    assert all(value == pytest.approx(90.0) for (a, _, b), value in targets.items() if len({a, b} & equatorial) == 1)

    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    assert not ensemble.unrelaxed
    geom.check(ensemble.mol).assert_ok()
    assert ensemble.cons.pulls == cons.pulls
    pytest.importorskip("xyzgraph")
    path = tmp_path / "tridentate_normal_arm.xyz"
    Chem.MolToXYZFile(ensemble.mol, str(path))
    fresh = rx.read_xyz(str(path), charge=Chem.GetFormalCharge(ensemble.mol), bond_orders="xyz2mol")
    assert rx.cxsmiles(fresh) == rx.cxsmiles(iso)


def test_tridentate_fan_compiles_shared_targets_and_survives_cleanup(tmp_path):
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "square_pyramidal")
        if candidate.vertices[0] == 3 and set(candidate.vertices[1::2]) == {0, 6}
    )
    targets = {key: value for key, value in iso.cons.pulls.items() if len(key) == 3}
    assert len(targets) == 10
    for donor in (0, 3, 6):
        for spectator in (8, 9):
            assert targets[donor, iso.metal, spectator] == pytest.approx(90.0)
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    assert not ensemble.unrelaxed
    assert ensemble.cons == iso.cons
    geom.check(ensemble.mol).assert_ok()
    assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
    pytest.importorskip("xyzgraph")
    path = tmp_path / "tridentate_fan.xyz"
    Chem.MolToXYZFile(ensemble.mol, str(path))
    fresh = rx.read_xyz(str(path), charge=Chem.GetFormalCharge(ensemble.mol), bond_orders="xyz2mol")
    assert rx.cxsmiles(fresh) == rx.cxsmiles(iso)


@pytest.mark.parametrize("case", ["triad", "fan", "opposed", "single"])
def test_planar_bite_targets_are_atom_slot_and_rotation_invariant(case):
    mol, vertices, poly, bites, original = {
        "triad": lambda: _planar_bite_case("trigonal_bipyramidal"),
        "fan": _planar_fan_case,
        "opposed": _planar_opposed_case,
        "single": lambda: _planar_opposed_case(False),
    }[case]()
    baseline = original.copy()
    _planar_bite_targets(baseline, vertices, poly, bites, 0, mol, ())
    order = list(reversed(range(mol.GetNumAtoms())))
    mapping = {old: new for new, old in enumerate(order)}
    moved_mol = Chem.RenumberAtoms(mol, order)
    rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    for permutation in itertools.permutations(range(len(vertices))):
        slot_map = {old: new for new, old in enumerate(permutation)}
        moved_vertices = [mapping[vertices[i]] for i in permutation]
        moved_poly = replace(poly, vertex_dirs=np.asarray(poly.vertex_dirs)[list(permutation)] @ rotation)
        cons = original.copy(
            distances={tuple(sorted(mapping[i] for i in key)): value for key, value in original.distances.items()},
            angles={tuple(mapping[i] for i in key): value for key, value in original.angles.items()},
            metals={mapping[0]},
        )
        moved_bites = {frozenset(slot_map[i] for i in key): value for key, value in reversed(list(bites.items()))}
        _planar_bite_targets(cons, moved_vertices, moved_poly, moved_bites, mapping[0], moved_mol, ())
        for field in ("angles", "pulls"):
            restored = {}
            for key, value in getattr(cons, field).items():
                restored_key = tuple(order[i] for i in key)
                restored[min(restored_key, restored_key[::-1])] = value
            expected = {min(key, key[::-1]): value for key, value in getattr(baseline, field).items()}
            assert restored.keys() == expected.keys()
            np.testing.assert_allclose([restored[key] for key in expected], list(expected.values()), atol=1e-10)


def test_planar_bite_targets_abstain_for_coupled_held_or_incomplete_shells():
    mol, vertices, poly, bites, original = _planar_bite_case()
    for field, value in (
        ("fixed", {(1, 0, 5): (100.0, 100.0)}),
        ("frozen", {1}),
        ("contacts", (frozenset(), frozenset({(1, 0, 5)}))),
        ("haptic", {1: (2, 3)}),
        ("angles", {key: value for key, value in original.angles.items() if key != (1, 0, 5)}),
    ):
        cons = original.copy(**{field: value})
        before = cons.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, mol, ())
        assert cons == before
    linked = Chem.RWMol(mol)
    linked.AddBond(4, 5, Chem.BondType.SINGLE)
    for source, blocked in ((mol, {0}), (mol, {2}), (linked.GetMol(), ())):
        cons = original.copy()
        _planar_bite_targets(cons, vertices, poly, bites, 0, source, blocked)
        assert cons == original
    cons = original.copy()
    _planar_bite_targets(cons, vertices, POLYHEDRA["t_shape"], bites, 0, mol, ())
    assert cons == original
    cons.angles[1, 0, 4] = (10.0, 20.0)
    before = cons.copy()
    _planar_bite_targets(cons, vertices, poly, {next(iter(bites)): (10.0, 20.0)}, 0, mol, ())
    assert cons == before


@pytest.mark.parametrize("opposed", [False, True])
def test_public_planar_bite_compilation_respects_late_target_owners(opposed):
    iso = (
        Isomer(Chem.MolFromSmiles("[Ag+]12(<-[O-]C[O-]->1)<-NCCN->2"), "square_planar", {0: 1, 1: 3, 2: 4, 3: 7})
        if opposed
        else rx.metal("N[Ni]1NCCN1", "trigonal_planar")[0]
    )
    original = iso.cons
    assert any(len(key) == 3 for key in original.pulls)
    trans = (iso.vertices[0], iso.metal, iso.vertices[2])
    for external in (Constraints(pulls={trans: 110.0}), Constraints(fixed={trans: (180.0, 180.0)})):
        cons = iso._constraints(external)
        assert not any(len(key) == 3 for key in cons.pulls)
    assert iso.cons == original


def test_planar_diamine_cleanup_retains_targets_bonds_and_fresh_cx(tmp_path):
    iso = rx.metal("N[Ni]1NCCN1", "trigonal_planar")[0]
    targets = {key: value for key, value in iso.cons.pulls.items() if len(key) == 3}
    assert len(targets) == 3
    ens = rx.embed(iso, n=1, seed=42).minimize()
    assert ens.n == 1
    assert not ens.unrelaxed
    assert {key: value for key, value in ens.cons.pulls.items() if len(key) == 3} == targets
    assert all(check.ok for check in ens.check().values())
    path = tmp_path / "planar_diamine.xyz"
    Chem.MolToXYZFile(ens.mol, str(path))
    fresh = rx.read_xyz(
        str(path), charge=Chem.GetFormalCharge(ens.mol), connectivity="rdkit", bond_orders="xyz2mol", fallback=False
    )
    assert rx.cxsmiles(fresh) == rx.cxsmiles(ens.mol) == rx.cxsmiles(iso)
    edges = [{frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()} for mol in (fresh, ens.mol)]
    assert edges[0] == edges[1]


def _frag_of(mol):
    return {a: f for f, atoms in enumerate(Chem.GetMolFrags(mol)) for a in atoms}


def _states_an_intra_pair(iso):
    frag = _frag_of(iso.mol)
    return any(k[1] == iso.metal and frag[k[0]] == frag[k[2]] for k in iso.cons.angles)


def test_chelate_bite_uses_backbone_not_ideal():
    iso = next(
        i
        for i in rx.metal(_BIS_EN_CO, "octahedral")
        if _states_an_intra_pair(i)
        and all(
            window[1] < 135
            for key, window in i.cons.angles.items()
            if key[1] == i.metal and _frag_of(i.mol)[key[0]] == _frag_of(i.mol)[key[2]]
        )
    )
    frag = _frag_of(iso.mol)
    ideal = {a for _i, _j, a in POLYHEDRA["octahedral"].resolved_angles}

    intra = {k: v for k, v in iso.cons.angles.items() if k[1] == iso.metal and frag[k[0]] == frag[k[2]]}
    assert intra, "the fixture must state at least one intra-chelate window"
    for k, window in intra.items():
        assert _CHELATE_BITE[5][0] <= window[0] <= window[1] <= _CHELATE_BITE[5][1], (
            f"{k}: an en 5-ring must intersect its graph reach with its bite prior, got {window}"
        )
    inter = {k: v for k, v in iso.cons.angles.items() if k[1] == iso.metal and frag[k[0]] != frag[k[2]]}
    assert inter, "the fixture must also state an inter-ligand pair, or the contrast is untested"
    pads = {(max(0.0, a - 8.0), min(180.0, a + 8.0)) for a in ideal}  # ±8° about a tabulated angle, clamped
    for k, window in inter.items():
        assert window in pads, f"{k}: an inter-ligand pair must be a polyhedron angle ±8°, got {window}"

    # the window comes from the RING, so a pair with no shared backbone gets none at all
    cl = [d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "Cl"]
    assert _chelate_bite_window(iso.mol, cl[0], cl[1]) is None


def test_chelate_bites_do_not_depend_on_polyhedron_angle_representatives():
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "trigonal_bipyramidal")
        if set(candidate.vertices[:2]) == {0, 6}
    )

    bites = {
        frozenset((left, right)): window
        for (left, metal, right), window in iso.cons.angles.items()
        if metal == iso.metal
    }
    assert bites[frozenset((0, 3))] == _CHELATE_BITE[5]
    assert bites[frozenset((3, 6))] == _CHELATE_BITE[5]


@pytest.mark.parametrize(
    ("geometry", "coordination_number"),
    [("trigonal_bipyramidal", 5), ("square_pyramidal", 5), ("octahedral", 6)],
)
def test_coordination_constraints_are_invariant_under_proper_rotation(geometry, coordination_number):
    mol = Chem.MolFromSmiles("[Fe+3].[Cl-].[Br-].[I-].N.P.S")
    metal, donors = 0, tuple(range(1, coordination_number + 1))

    def sphere_angles(vertices):
        cons = coordination(mol, metal, vertices, geometry, 26, haptic={})
        return {(min(a, b), max(a, b)): window for (a, centre, b), window in cons.angles.items() if centre == metal}

    expected = sphere_angles(donors)
    for rotation in point_group(vertex_dirs(geometry))[0]:
        assert sphere_angles(tuple(donors[rotation[i]] for i in range(coordination_number))) == expected


def test_directly_bonded_donors_use_their_distance_triangle():
    iso = rx.metal("N1N->[Y+3](<-[Cl-])(<-[Cl-])<-1", "tetrahedral")[0]

    assert (0, iso.metal, 1) not in iso.cons.angles
    assert (0, 1) in iso.cons.distances


@pytest.mark.parametrize("reverse", [False, True])
def test_retained_bonded_donors_keep_distance_ownership_and_allow_explicit_angles(reverse):
    from rxembed.embed import prepare
    from rxembed.metal_isomer import from_geometry

    mol = Chem.MolFromSmiles("N1N->[Y+3](<-[Cl-])(<-[Cl-])<-1")
    native = rdDistGeom.GetMoleculeBoundsMatrix(Chem.MolFromSmiles("NN"))
    half_span = (native[0, 1] + native[1, 0]) / 4
    radius = 2.2
    height = np.sqrt(radius**2 - half_span**2)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetPositions(
        np.array(
            [
                [half_span, 0, height],
                [-half_span, 0, height],
                [0, 0, 0],
                [0, radius * np.sqrt(2 / 3), -radius / np.sqrt(3)],
                [0, -radius * np.sqrt(2 / 3), -radius / np.sqrt(3)],
            ]
        )
    )
    mol.AddConformer(conf)
    order = list(range(mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        mol = Chem.RenumberAtoms(mol, order)
    left, right = sorted((order.index(0), order.index(1)))
    metal = order.index(2)
    iso = from_geometry(mol, lengths="input")
    cons = iso.cons
    angle = (left, metal, right)

    for a, b in ((left, right), (left, metal), (right, metal)):
        lo, hi = cons.distances[tuple(sorted((a, b)))]
        distance = np.linalg.norm(mol.GetConformer().GetPositions()[a] - mol.GetConformer().GetPositions()[b])
        assert lo <= distance <= hi
    assert angle not in cons.angles
    assert angle[::-1] not in cons.angles
    stated = Isomer(mol, iso.geometry, iso.vertices, lengths="input")
    assert cons.angles == stated.cons.angles
    assert cons.distances == stated.cons.distances
    value = GetAngleDeg(mol.GetConformer(), *angle)
    _, fixed, _, _ = prepare(iso, fix={angle: value})
    assert fixed.fixed[angle] == pytest.approx((value, value))
    assert fixed.angles[angle][0] <= value <= fixed.angles[angle][1]


def test_retained_haptic_distances_are_measured_but_angles_use_the_polyhedron():
    from rxembed.metal_isomer import from_geometry

    mol = ferrocene()
    positions = mol.GetConformer().GetPositions()
    positions[6:11] += (0.6, 0, 0)
    mol.GetConformer().SetPositions(positions)
    iso = from_geometry(mol, lengths="input")
    cons = iso.cons
    left, right = cons.haptic
    angle = (left, iso.metal, right)
    rays = [positions[list(cons.haptic[site])].mean(axis=0) - positions[iso.metal] for site in (left, right)]
    measured = _vertex_angle(*rays)
    assert measured < 165
    assert cons.angles.get(angle, cons.angles.get(angle[::-1])) == (172.0, 180.0)
    for site, ray in zip((left, right), rays, strict=True):
        window = cons.distances[tuple(sorted((site, iso.metal)))]
        assert 0.5 * sum(window) == pytest.approx(np.linalg.norm(ray))
    stated = Isomer(mol, iso.geometry, [cons.haptic[site][0] for site in iso.vertices], lengths="input")
    assert cons.angles == stated.cons.angles
    assert cons.distances == stated.cons.distances


def test_three_centre_hydrogen_bridge_keeps_its_measured_xh_span():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(26))
    boron = rw.AddAtom(Chem.Atom(5))
    bridge = rw.AddAtom(Chem.Atom(1))
    nitrogen = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(boron, bridge, Chem.BondType.SINGLE)
    for donor in (boron, bridge, nitrogen):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((0, 0, 0), (2.1, 0, 0), (1.4, 1.15, 0), (0, 2.0, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    cons = coordination(
        mol, metal, (boron, bridge, nitrogen), "trigonal_planar", 26, haptic={}, source=mol, lengths="input"
    )

    distance = np.linalg.norm(np.array((2.1, 0, 0)) - np.array((1.4, 1.15, 0)))
    assert cons.distances[(boron, bridge)] == pytest.approx((distance - 0.1, distance + 0.1))
    assert cons.pulls[(boron, bridge)] == pytest.approx(distance)


def test_large_atom_chain_is_not_forced_into_an_organic_chelate_bite():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(28))
    donors = [rw.AddAtom(Chem.Atom(33)) for _ in range(6)]
    caps = [rw.AddAtom(Chem.Atom(6)) for _ in donors]
    for i, donor in enumerate(donors):
        rw.AddBond(donor, donors[(i + 1) % len(donors)], Chem.BondType.SINGLE)
        rw.AddBond(donor, caps[i], Chem.BondType.SINGLE)
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal, Point3D(0.0, 0.0, 0.0))
    for i, (donor, cap) in enumerate(zip(donors, caps, strict=True)):
        direction = np.array((np.cos(i * np.pi / 3), np.sin(i * np.pi / 3), 0.0))
        conf.SetAtomPosition(donor, Point3D(*(2.5 * direction)))
        conf.SetAtomPosition(cap, Point3D(*(3.9 * direction)))
    mol.AddConformer(conf)

    isomers = rx.metal(mol, lengths="input", stereo="free")

    assert isomers
    assert isomers[0].geometry == "hexagonal_planar"

    coordinate_free = rx.metal(rx.cxsmiles(isomers[0]), stereo="free")[0]
    assert (
        min(
            coordinate_free.cons.distances[tuple(sorted((coordinate_free.metal, donor)))][1]
            for donor in coordinate_free.donors
        )
        > 2.4
    )


def test_trans_bidentate_keeps_the_polyhedron_window():
    iso = next(
        candidate
        for candidate in rx.metal("N1CCCCCN->[Pt+2](<-[Cl-])(<-[Cl-])<-1", "square_planar", stereo="free")
        if candidate.label == "trans"
    )
    nitrogens = [donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetAtomicNum() == 7]

    assert iso.cons.angles[(nitrogens[0], iso.metal, nitrogens[1])] == (172.0, 180.0)


def test_tridentate_outer_trans_pair_uses_the_classification_window():
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "trigonal_bipyramidal")
        if set(candidate.vertices[:2]) == {0, 6}
    )

    assert iso.cons.angles[(0, iso.metal, 6)] == (150.0, 180.0)


@pytest.mark.parametrize("metal", ["Ag+", "Cd+2"])
def test_square_planar_opposed_chelate_targets_have_a_common_geometry(metal):
    mol = Chem.MolFromSmiles(f"[{metal}]12(<-[O-]C[O-]->1)<-NCCN->2")
    iso = Isomer(mol, "square_planar", {0: 1, 1: 3, 2: 4, 3: 7})
    angles = {
        frozenset((left, right)): window
        for (left, metal, right), window in iso.cons.angles.items()
        if metal == iso.metal
    }

    bites = [angles[frozenset((1, 3))], angles[frozenset((4, 7))]]
    for window, prior in zip(bites, (_CHELATE_BITE[4], _CHELATE_BITE[5]), strict=True):
        assert prior[0] <= window[0] <= window[1] <= prior[1]
    assert len(angles) == 6
    gram = np.eye(len(iso.vertices))
    for i, j in itertools.combinations(range(len(iso.vertices)), 2):
        a, b = iso.vertices[i], iso.vertices[j]
        target = iso.cons.pulls[min((a, iso.metal, b), (b, iso.metal, a))]
        lo, hi = angles[frozenset((a, b))]
        assert lo - 1e-8 <= target <= hi + 1e-8
        gram[i, j] = gram[j, i] = np.cos(np.radians(target))
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert np.linalg.matrix_rank(gram) == 2


@pytest.mark.parametrize("smiles", ["[Zn+2].NCCN.NCCCN", "[Zn+2].NCC[O-].PCCCN"])
def test_independent_tetrahedral_bite_centres_are_jointly_realizable(smiles):
    mol = Chem.MolFromSmiles(smiles)
    donors = (1, 4, 5, 9)
    expected = None
    for vertices in itertools.permutations(donors):
        cons = coordination(mol, 0, vertices, "tetrahedral", 30, haptic={})
        angles = {tuple(sorted((a, b))): window for (a, metal, b), window in cons.angles.items() if metal == 0}
        if expected is None:
            expected = angles
        assert angles == pytest.approx(expected)
        gram = np.eye(len(donors))
        for (a, b), window in angles.items():
            i, j = donors.index(a), donors.index(b)
            gram[i, j] = gram[j, i] = np.cos(np.radians(0.5 * sum(window)))
        spectrum = np.linalg.eigvalsh(gram)
        assert spectrum[0] == pytest.approx(0, abs=1e-12), "four unit rays need rank at most three"
        assert spectrum[1] > 0, "the witness must not collapse into a plane"
    order = list(reversed(range(mol.GetNumAtoms())))
    renamed = coordination(
        Chem.RenumberAtoms(mol, order),
        order.index(0),
        tuple(order.index(i) for i in donors),
        "tetrahedral",
        30,
        haptic={},
    )
    assert {
        tuple(sorted((order[a], order[b]))): window
        for (a, metal, b), window in renamed.angles.items()
        if order[metal] == 0
    } == pytest.approx(expected)


def test_tetrahedral_bite_centre_abstains_for_held_or_wrong_shape_networks():
    mol = Chem.MolFromSmiles("[Zn+2].NCCN.NCCCN")
    vertices = (1, 4, 5, 9)
    cons = coordination(mol, 0, vertices, "tetrahedral", 30, haptic={})
    bites = {frozenset((0, 1)): (70.0, 90.0), frozenset((2, 3)): (75.0, 105.0)}
    assert _tetrahedral_cross_angle(mol, 0, vertices, bites, cons.distances, ()) is not None
    for blocked in ((0,), (2,)):
        assert _tetrahedral_cross_angle(mol, 0, vertices, bites, cons.distances, blocked) is None
    almost_planar = dict.fromkeys(bites, (168.0, 172.0))
    assert _tetrahedral_cross_angle(mol, 0, vertices, almost_planar, cons.distances, ()) is None
    linked = Chem.RWMol(mol)
    linked.AddBond(3, 6, Chem.BondType.SINGLE)
    assert _tetrahedral_cross_angle(linked.GetMol(), 0, vertices, bites, cons.distances, ()) == pytest.approx(
        _tetrahedral_cross_angle(mol, 0, vertices, bites, cons.distances, ())
    )
    assert _tetrahedral_cross_angle(linked.GetMol(), 0, vertices, bites, cons.distances, (8,)) is None
    assert _tetrahedral_cross_angle(mol, 0, vertices, {frozenset((0, 1)): (70.0, 90.0)}, cons.distances, ()) is None


def test_linked_thioether_bites_preserve_macrocycle_bonds_and_fresh_hands(tmp_path):
    from rdkit.Chem import rdForceFieldHelpers

    smiles = "C1CC[S]2->[Cd+2]34<-[S](C1)CC[S]->3CCCC[S]->4CC2"
    isomers = rx.metal(smiles, "tetrahedral")
    assert len(isomers) == 2
    identities = {rx.cxsmiles(iso) for iso in isomers}
    assert len(identities) == 2
    for index, iso in enumerate(isomers):
        ens = rx.embed(iso, n=1, seed=42, threads=1)
        assert ens.n == 1
        assert not ens.unrelaxed
        assert all(check.ok for check in ens.check().values())
        work = _ff_surrogate(ens._mol, ens.cons.metals)
        pos = ens._mol.GetConformer(ens.ids[0]).GetPositions()
        for bond in work.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if min(work.GetAtomWithIdx(atom).GetAtomicNum() for atom in (a, b)) <= 1:
                continue
            _force, equilibrium = rdForceFieldHelpers.GetUFFBondStretchParams(work, a, b)
            assert abs(np.linalg.norm(pos[a] - pos[b]) / equilibrium - 1) < 0.08
        path = tmp_path / f"thioether_macrocycle_{index}.xyz"
        Chem.MolToXYZFile(ens.mol, str(path))
        fresh = rx.read_xyz(str(path), charge=2, connectivity="rdkit", bond_orders="xyz2mol", fallback=False)
        assert rx.cxsmiles(fresh) == rx.cxsmiles(ens.mol) == rx.cxsmiles(iso)


def test_bis_imino_pyridine_tetrahedron_keeps_ligand_bonds_and_fresh_cx(tmp_path):
    from rdkit.Chem import rdForceFieldHelpers

    def edges(mol):
        return {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in mol.GetBonds()}

    smiles = "C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C"
    isomers = rx.metal(smiles, "tetrahedral")
    assert len(isomers) == 2
    for index, iso in enumerate(isomers):
        ens = rx.embed(iso, n=1, seed=42, threads=1)
        assert ens.n == 1
        assert all(check.ok for check in ens.check().values())
        work = _ff_surrogate(ens._mol, ens.cons.metals)
        pos = ens._mol.GetConformer(ens.ids[0]).GetPositions()
        for bond in work.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if min(work.GetAtomWithIdx(i).GetAtomicNum() for i in (a, b)) <= 1:
                continue
            _kb, equilibrium = rdForceFieldHelpers.GetUFFBondStretchParams(work, a, b)
            assert abs(np.linalg.norm(pos[a] - pos[b]) / equilibrium - 1) < 0.05
        path = tmp_path / f"bis_imino_pyridine_{index}.xyz"
        Chem.MolToXYZFile(ens.mol, str(path))
        fresh = rx.read_xyz(str(path), charge=1, connectivity="rdkit", bond_orders="xyz2mol", fallback=False)
        assert rx.cxsmiles(fresh) == rx.cxsmiles(ens.mol) == rx.cxsmiles(iso)
        assert edges(fresh) == edges(ens.mol)


def test_late_graft_and_numeric_bite_disable_independent_target_preferences():
    from rxembed.embed import prepare

    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    frag = _frag_of(iso.mol)
    initial = iso.cons
    cross = next(k for k in initial.angles if k[1] == iso.metal and frag[k[0]] != frag[k[2]])
    bite = next(k for k in initial.angles if k[1] == iso.metal and frag[k[0]] == frag[k[2]])
    backbone = next(
        a.GetIdx()
        for a in iso.mol.GetAtoms()
        if a.GetAtomicNum() == 6 and sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors()) == 2
    )
    _, grafted, _, _ = prepare(iso, fix={backbone: (0.0, 0.0, 0.0)})
    _, fixed, _, _ = prepare(iso, fix={bite: 87.0})
    assert backbone in grafted.frozen
    assert fixed.fixed[bite] == (87.0, 87.0)
    assert fixed.angles[cross] == grafted.angles[cross] != initial.angles[cross]
    assert iso.cons.angles == initial.angles, "late authority must not mutate the caller's isomer"


def test_unnamed_spectator_donor_bond_disables_independent_bite_targets():
    from rxembed.metal_constraints import compile_constraints, compile_context

    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    assert iso._constraints(context=compile_context(iso._graph)) == iso.cons
    spectator = iso._graph.GetNumAtoms()
    mol = Chem.CombineMols(iso._graph, Chem.MolFromSmiles("[C]"))
    Chem.GetSymmSSSR(mol)
    unnamed = MetalState(spectator, 26, 0, "", ())
    cons = compile_constraints(
        mol,
        (*iso.centres, unnamed),
        length_mol=iso._length_mol,
        base=Constraints(),
        constrained_metals=iso._constrained_metals,
        lengths="model",
        stereo_label=iso.stereo_label,
        donor_bonds=[*iso.donor_bonds, (2, spectator)],
    )
    frag = _frag_of(iso.mol)
    cross = next(k for k in iso.cons.angles if k[1] == iso.metal and frag[k[0]] != frag[k[2]])
    assert cons.angles[cross] != iso.cons.angles[cross]


def test_chelate_bite_is_not_reported_as_a_steric_clash():
    ens = rx.embed(rx.metal(_NI_N_CY, "square_planar")[0], n=1).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert not [v for v in rep.violations if v.kind == "clash"], rep.summary()


@pytest.mark.parametrize(("linker", "labels"), [("CCCC", {"cis"}), ("CCCCCCC", {"cis", "trans"})])
def test_chelate_reach_preserves_embeddings_without_stretching_the_backbone(linker, labels):
    from rdkit.Chem import rdForceFieldHelpers

    isomers = rx.metal(f"[Pd+2]1(<-[Cl-])(<-[Cl-])<-N{linker}N->1", "square_planar")
    assert len(isomers) == len(labels)
    assert {iso.label for iso in isomers} == labels
    for iso in isomers:
        ensemble = rx.embed(iso, n=1, seed=1)
        assert ensemble.n == 1
        positions = ensemble.mol.GetConformer().GetPositions()
        for bond in ensemble.mol.GetBonds():
            if not all(atom.GetAtomicNum() == 6 for atom in (bond.GetBeginAtom(), bond.GetEndAtom())):
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            params = rdForceFieldHelpers.GetUFFBondStretchParams(ensemble.mol, a, b)
            assert params is not None
            assert np.linalg.norm(positions[a] - positions[b]) == pytest.approx(params[1], rel=0.08)


def test_side_on_eta2_ligand_embeds_geometry_clean():
    smi = "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C"
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=1).minimize()
    assert any(geom.check(ens.mol, c).ok() for c in ens.ids), "no geom.check-clean side-on conformer"


def test_haptic_face_and_sigma_donor_on_one_ligand_compile_without_a_virtual_bite_path():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(metal).SetFormalCharge(2)
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    linker = rw.AddAtom(Chem.Atom(6))
    donor = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    rw.AddBond(face[1], linker, Chem.BondType.SINGLE)
    rw.AddBond(linker, donor, Chem.BondType.SINGLE)
    for atom in (*face, donor):
        rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "trigonal_planar", {0: face[0], 1: donor})
    assert iso.cons.haptic


def test_input_coordinates_do_not_disable_haptic_chelate_donor_folds():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom("Y"))
    faces = []
    for _ in range(2):
        left, right = (rw.AddAtom(Chem.Atom("C")) for _ in range(2))
        rw.AddBond(left, right, Chem.BondType.DOUBLE)
        faces.append((left, right))
    donors = (rw.AddAtom(Chem.Atom("N")), rw.AddAtom(Chem.Atom("N")))
    rw.AddBond(*donors, Chem.BondType.SINGLE)
    substituents = []
    for donor in donors:
        substituent = rw.AddAtom(Chem.Atom("C"))
        substituents.append(substituent)
        rw.AddBond(donor, substituent, Chem.BondType.SINGLE)
    for donor in (*itertools.chain.from_iterable(faces), *donors):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    source = rw.GetMol()
    source.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(source, catchErrors=True)
    conf = Chem.Conformer(source.GetNumAtoms())
    points = {
        metal: (0.0, 0.0, 0.0),
        donors[0]: (2.3, 0.0, 0.0),
        donors[1]: (-2.3, 0.0, 0.0),
        substituents[0]: (2.3, 1.0, 0.0),
        substituents[1]: (-2.3, -1.0, 0.0),
        faces[0][0]: (0.0, 2.3, 0.3),
        faces[0][1]: (0.0, 2.3, -0.3),
        faces[1][0]: (0.0, -2.3, 0.3),
        faces[1][1]: (0.0, -2.3, -0.3),
    }
    for atom, point in points.items():
        conf.SetAtomPosition(atom, Point3D(*point))
    source.AddConformer(conf)

    sites = [faces[0][0], faces[1][0], *donors]
    observed = rx.Isomer(source, "tetrahedral", sites)
    model = Chem.Mol(source)
    model.RemoveAllConformers()
    unmeasured = rx.Isomer(model, "tetrahedral", sites)
    assert observed.cons == unmeasured.cons
    for donor, substituent in zip(donors, substituents, strict=True):
        fold = (metal, donor, substituent)
        assert fold in observed.cons.angles
        assert fold in unmeasured.cons.angles


def test_planar_haptic_umbrella_does_not_depend_on_the_centroid_slot():
    smiles = r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
    isomers = rx.metal(smiles, "square_planar", stereo="free")

    assert len(isomers) == 3
    assert all(iso.cons.umbrellas for iso in isomers)


@pytest.mark.parametrize("geometry", ["square_planar", "hexagonal_planar"])
def test_planar_shell_force_covers_every_donor_and_is_index_invariant(geometry):
    from rdkit.Chem import rdForceFieldHelpers

    from rxembed.mechanisms import Umbrella

    poly = POLYHEDRA[geometry]
    vertices = tuple(range(1, poly.cn + 1))
    positions = np.vstack((np.zeros(3), poly.vertex_dirs)).astype(float)
    positions[1:] *= np.linspace(0.7, 1.8, poly.cn)[:, None]

    def field(metal, slots, pose):
        mol = Chem.MolFromSmiles(".".join(["[C]"] * len(pose)))
        conf = Chem.Conformer(len(pose))
        conf.SetPositions(pose)
        mol.AddConformer(conf)
        cons = Constraints()
        _add_umbrella(cons, metal, slots, poly)
        assert sum(value[1] for value in cons.umbrellas.values()) == pytest.approx(1.0)
        ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
        Umbrella()._ff_terms(ff, cons, mol.GetConformer(), 1.0)
        ff.Initialize()
        return mol, ff

    _owner, ff = field(0, vertices, positions)
    assert ff.CalcEnergy() == pytest.approx(0.0, abs=1e-10)
    for donor in vertices:
        moved = positions.copy()
        moved[donor, 2] = 2.0
        energy = ff.CalcEnergy(moved.ravel().tolist())
        gradient = np.array(ff.CalcGrad(moved.ravel().tolist())).reshape(-1, 3)
        assert energy > 0.0
        assert np.linalg.norm(gradient[donor]) > 0.0
        plus, minus = moved.copy(), moved.copy()
        plus[donor, 2] += 1e-5
        minus[donor, 2] -= 1e-5
        difference = (ff.CalcEnergy(plus.ravel().tolist()) - ff.CalcEnergy(minus.ravel().tolist())) / 2e-5
        assert gradient[donor, 2] == pytest.approx(difference, rel=1e-5, abs=1e-5)
        for slots in (vertices[1:] + vertices[:1], vertices[::-1]):
            _rotated_owner, rotated = field(0, slots, positions)
            assert rotated.CalcEnergy(moved.ravel().tolist()) == pytest.approx(energy, abs=1e-8)
            np.testing.assert_allclose(rotated.CalcGrad(moved.ravel().tolist()), gradient.ravel(), atol=1e-7)
        order = np.arange(len(positions))[::-1]
        _renamed_owner, renamed = field(int(order[0]), tuple(int(order[i]) for i in vertices), positions[order])
        assert renamed.CalcEnergy(moved[order].ravel().tolist()) == pytest.approx(energy, abs=1e-8)
        np.testing.assert_allclose(
            np.array(renamed.CalcGrad(moved[order].ravel().tolist())).reshape(-1, 3)[order], gradient, atol=1e-7
        )


def test_pyramidal_haptic_sites_define_the_shape_umbrella():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(26))
    faces = [[rw.AddAtom(Chem.Atom(6)) for _ in range(2)] for _ in range(2)]
    donor = rw.AddAtom(Chem.Atom(7))
    for face in faces:
        rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
        for atom in face:
            rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "trigonal_pyramidal", {0: faces[0][0], 1: faces[1][0], 2: donor})

    assert len(iso.cons.umbrellas) == 1
    key, value = next(iter(iso.cons.umbrellas.items()))
    assert set(key) == {metal, donor, *iso.cons.phantoms}
    assert value == POLYHEDRA["trigonal_pyramidal"].umbrella_improper


def test_planar_haptic_shape_with_a_vacancy_keeps_its_umbrella():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(46))
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    donors = [rw.AddAtom(Chem.Atom(7)) for _ in range(2)]
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    for atom in (*face, *donors):
        rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "square_planar", {0: face[0], 1: donors[0], 2: donors[1]})

    assert len(iso.cons.umbrellas) == 1
    key, value = next(iter(iso.cons.umbrellas.items()))
    assert set(key) == {metal, *iso.cons.phantoms, *donors}
    assert value is None


def test_coordinated_nh_imine_uses_requested_ez_donor_plane():
    source = Chem.AddHs(Chem.MolFromSmiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"))
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        isomers = enumerate_isomers(Chem.RenumberAtoms(source, order), "square_planar")
        assert {_stereo.bond_stereo(iso.stereo_label).popitem()[1] for iso in isomers} == {"E", "Z"}
        for iso in isomers:
            targets = _stereo.metal_referenced_ez(iso.mol, iso.stereo_label, iso.donor_bonds)
            ((_, (donor, carbon, metal, _ref, _ligand_ref, wanted)),) = targets.items()
            restored = Chem.Mol(iso.mol)
            iso.restore(restored)
            restored = _metal.connect_metal(restored, iso.donor_bonds)
            assert _stereo.metal_referenced_ez(restored, iso.stereo_label, iso.donor_bonds) == targets
            rows = iso.cons.coplanar
            ((_, _, _, _ref, anchor, _cap),) = [row for row in rows if row[0] == metal]
            assert anchor == (180.0 if wanted == "E" else 0.0)
            assert {row[4] for row in rows if row[1:3] == (donor, carbon)} == {0.0, 180.0}


def test_coordinated_nh_imine_ez_survives_donor_orientation_ablation():
    source = Chem.AddHs(Chem.MolFromSmiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"))
    isomer = enumerate_isomers(source, "square_planar")[0]
    targets = _stereo.metal_referenced_ez(isomer.mol, isomer.stereo_label, isomer.donor_bonds)
    assert targets

    cons = isomer._constraints(donor_orientation=False)
    ((_, (_, _, metal, _ref, _ligand_ref, wanted)),) = targets.items()
    rows = [row for row in cons.coplanar if row[0] == metal]
    assert rows
    assert any(row[4] == (180.0 if wanted == "E" else 0.0) for row in rows)


def test_n_substituted_imine_keeps_ordinary_ligand_ez():
    iso = rx.metal(r"C/C=N(/C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar")[0]

    assert _stereo.bond_stereo(iso.stereo_label)
    assert not _stereo.metal_referenced_ez(iso.mol, iso.stereo_label, iso.donor_bonds)


def test_ligand_ez_uses_rdkit_reference_geometry_not_absolute_cip():
    mol = Chem.MolFromSmiles("FC(Cl)=C(Br)I")
    bond = next(bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    bond.SetStereoAtoms(0, 5)
    bond.SetStereo(Chem.BondStereo.STEREOE)
    label = _stereo.defined_stereo_label(mol)
    assert label.endswith(":Z"), "the regression requires raw-trans references whose absolute CIP label is Z"

    cons = Constraints()
    _add_ligand_ez(cons, mol, label, ())

    assert cons.coplanar == [(0, 1, 3, 5, 180.0, _COPLANAR_CAP)]


def test_eta2_alkene_keeps_stated_ez_during_relaxation():
    iso = rx.metal(r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "square_planar")[0]
    pair = next(iter(_stereo.bond_stereo(iso.stereo_label)))
    rows = [row for row in iso.cons.coplanar if frozenset(row[1:3]) == pair]

    assert len(rows) == 1
    tag = iso.mol.GetBondBetweenAtoms(*pair).GetStereo()
    cis = tag in {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}
    assert rows[0][4:] == (0.0 if cis else 180.0, _COPLANAR_CAP)


def test_fully_occupied_planar_chelate_gets_one_shape_level_plane_hold():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(47))
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    donors = [rw.AddAtom(Chem.Atom(8)) for _ in range(2)]
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    rw.AddBond(face[0], donors[0], Chem.BondType.SINGLE)
    rw.AddBond(face[1], donors[1], Chem.BondType.SINGLE)
    for atom in (*face, *donors):
        rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "trigonal_planar", {0: face[0], 1: donors[0], 2: donors[1]})

    assert set(iso.haptic.values()) == {tuple(face)}
    assert len(iso.cons.umbrellas) == 1
    assert next(iter(iso.cons.umbrellas)).count(iso.metal) == 1


def test_rejected_seeds_are_replaced_to_n_clean(monkeypatch):
    ens = rx.embed(rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0], n=2)
    donors = list(ens.iso.donors)
    workflow_failure = type(ens)._workflow_failure
    rejected = False

    def reject_one_seed(self, owner, cid):
        nonlocal rejected
        reason = workflow_failure(self, owner, cid)
        if not rejected and owner is self:
            rejected = True
            return "test rejection"
        return reason

    monkeypatch.setattr(type(ens), "_workflow_failure", reject_one_seed)
    ens._stage = "seeded"
    ens.minimize()
    assert rejected, "the test did not send a seed through the rejection path"
    assert ens.n == 2, "embed(n=2) must hand back exactly 2 geometries after re-seeding"
    assert all(geom.check(ens.mol, c, donors=donors, constraints=ens.cons).ok() for c in ens.ids)


def test_free_fragment_is_tethered_at_vdw_contact():
    embedded = 0
    for iso in rx.metal("CCCN[Pd](Cl)(Cl)NCCC.c1ccccc1", "square_planar"):
        ens = rx.embed(iso, n=2).minimize()
        if not ens.n:
            continue
        embedded += 1
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        for frag in Chem.GetMolFrags(ens.mol):
            heavy = [a for a in frag if ens.mol.GetAtomWithIdx(a).GetAtomicNum() > 1 and a != iso.metal]
            if heavy:
                assert min(float(np.linalg.norm(pos[iso.metal] - pos[a])) for a in heavy) < 8.0
    assert embedded >= 1


# --- coordinate(): seating a substrate at a vacant vertex -------------------------------------------------


def test_coordinate_binds_a_substrate_at_the_vacant_site():
    es = rx.embed("CCCN[Pd](Cl)NCCC.O", metal="square_planar", coordinate="[OX2]", n=3, seed=1)
    for ens in list(es) if isinstance(es, rx.EnsembleSet) else [es]:
        m = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "Pd")
        o = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "O")
        assert o in ens.iso.vertices
        assert (o, m) in ens.iso.donor_bonds
        assert o in ens.sphere[m]
        ens.minimize()
        assert ens.n >= 1
        assert all(report.ok() for report in ens.check().values())
        lo, hi = ens.cons.distances[(min(m, o), max(m, o))]
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            assert lo - 0.1 <= float(np.linalg.norm(pos[m] - pos[o])) <= hi + 0.1, "the substrate did not seat"


def test_coordinate_composes_with_fix_constrain_and_contacts():
    iso = rx.metal("CCCN[Pd](Cl)NCCC.O", "square_planar")[0]
    metal = iso.metal
    oxygen = next(atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "O")
    carbons = [atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "C"]
    nitrogen = next(atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "N")
    fixed = (carbons[0], carbons[1])
    soft = (carbons[-2], carbons[-1])
    contact = (nitrogen, oxygen)

    ens = rx.embed(
        iso,
        coordinate=oxygen,
        fix={fixed: 1.53},
        constrain={soft: (1.4, 1.7)},
        contacts={contact: (2.5, 4.5)},
        n=1,
        seed=1,
    )

    assert oxygen in ens.iso.vertices
    assert ens.cons.fixed[fixed] == (1.53, 1.53)
    assert ens.cons.contacts[0] == frozenset({soft, contact})
    assert tuple(sorted((metal, oxygen))) in ens.cons.distances


def test_coordinate_relieves_the_phantom_floor_it_creates(monkeypatch):
    from rxembed import bounds as _b

    tols, real = [], _b._bounds

    def spy(mol, cons, params=None):
        bm, tol = real(mol, cons, params)
        tols.append(tol)
        return bm, tol

    monkeypatch.setattr(_b, "_bounds", spy)
    iso = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "square_planar").select(index=0)  # one vacant site + free acetone
    o = next(a.GetIdx() for a in iso.mol.GetAtoms() if a.GetSymbol() == "O")
    ens = rx.embed(iso, coordinate=o, n=1)
    carbonyl = next(n.GetIdx() for n in iso.mol.GetAtomWithIdx(o).GetNeighbors())
    assert (iso.metal, o, carbonyl) in ens.cons.angles
    assert tols, "the embed never built a bounds matrix"
    assert max(tols) == 0.0, f"bound crossover repaired: {max(tols) * 100:.4f}%"


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ("donor", "already donors"),
        ("metal", "not metal atoms"),
        ("missing", "must be in"),
        ("repeated", "must be distinct"),
    ],
)
def test_coordinate_rejects_invalid_identity_expansion(coordinate, message):
    iso = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "square_planar").select(index=0)
    oxygen = next(a.GetIdx() for a in iso.mol.GetAtoms() if a.GetSymbol() == "O")
    choice = {
        "donor": iso.donors[0],
        "metal": iso.metal,
        "missing": iso.mol.GetNumAtoms(),
        "repeated": [oxygen, oxygen],
    }[coordinate]
    with pytest.raises(ValueError, match=message):
        rx.embed(iso, coordinate=choice, n=1)


# ---------------------------------------------------------------------------------------------------------
# lengths=; WHERE the M-donor window is measured from
# ---------------------------------------------------------------------------------------------------------

_SQUARE_PD = "Cl[Pd](Cl)(N)N"  # two chemically equivalent Cl and two equivalent N: the tell, below


def _fake_geometry(smiles, seed=1):
    """The same graph carrying a PLAIN ETKDG conformer: a geometry produced with no M-L parameter at all."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return Chem.RemoveHs(mol)


def _ml_windows(iso):
    return {k: v for k, v in iso.cons.distances.items() if iso.metal in k}


def test_etkdg_conformer_is_not_metal_geometry():
    mol = _fake_geometry(_SQUARE_PD)
    default = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar")[0])
    model = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar", lengths="model")[0])
    measured = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar", lengths="input")[0])

    def mids(w):
        return sorted(round((lo + hi) / 2, 3) for lo, hi in w.values())

    assert len(set(mids(model))) == 2, f"the model must give the two Cl one window and the two N another: {mids(model)}"
    assert default == model
    assert len(set(mids(measured))) == 4, "only explicit input lengths may use the metal-blind conformer"


def test_input_and_model_lengths_ignore_mol_metadata():
    mol = _fake_geometry(_SQUARE_PD)
    auto = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar")[0])
    given = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar", lengths="input")[0])
    assert given != auto, "coordinates must not silently supply model targets"

    graph = Chem.MolFromSmiles(_SQUARE_PD)
    auto_g = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar")[0])
    model_g = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar", lengths="model")[0])
    assert auto == auto_g == model_g


@pytest.mark.parametrize("lengths", ["input", "model"])
def test_constraint_compilation_classifies_donors_once(lengths, monkeypatch):
    from rxembed import metal_constraints, metal_donor_orient

    iso = rx.metal("Cl[Pt]1(F)N(C)CCCN1")[0]
    reference = rx.embed(iso, n=1, seed=42, threads=1).mol
    iso = rx.metal(reference, lengths=lengths)[0]
    expected = iso.cons
    classify = metal_donor_orient._stripped_hybridisation
    calls = []

    def counted(mol):
        calls.append(mol)
        return classify(mol)

    monkeypatch.setattr(metal_constraints, "_stripped_hybridisation", counted)
    monkeypatch.setattr(metal_donor_orient, "_stripped_hybridisation", counted)

    assert iso.cons == expected
    assert len(calls) == 1, "one coordination build must share its ligand classification across donors"


def test_input_haptic_centroid_uses_the_input_distance_width():
    iso = rx.metal(ferrocene(), lengths="input")[0]
    windows = [
        window for pair, window in iso.cons.distances.items() if iso.metal in pair and set(pair) & iso.cons.phantoms
    ]

    assert windows
    assert all(hi - lo == pytest.approx(0.2) for lo, hi in windows)


@pytest.mark.parametrize(
    ("ligand", "metal", "geometry"),
    [
        ("C=C", "Pt", "square_planar"),
        ("C=CC=C", "Fe", "octahedral"),
        ("[cH-]1cccc1", "Ti", "tetrahedral"),
        ("c1ccccc1", "Ru", "trigonal_planar"),
    ],
)
def test_model_haptic_targets_ignore_source_coordinates(ligand, metal, geometry):
    rw = Chem.RWMol(Chem.CombineMols(Chem.MolFromSmiles(f"[{metal}]"), Chem.MolFromSmiles(ligand)))
    face = tuple(range(1, rw.GetNumAtoms()))
    for donor in face:
        rw.AddBond(donor, 0, Chem.BondType.DATIVE)
    sites = [face[0]]
    for _ in POLYHEDRA[geometry].vertex_dirs[1:]:
        donor = rw.AddAtom(Chem.Atom("Cl"))
        rw.GetAtomWithIdx(donor).SetFormalCharge(-1)
        rw.AddBond(donor, 0, Chem.BondType.DATIVE)
        sites.append(donor)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    expected = Isomer(mol, geometry, sites, lengths="model").cons
    assert len(expected.haptic) == 1
    assert set(next(iter(expected.haptic.values()))) == set(face)

    for radius in (0.8, 1.2):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for offset, donor in enumerate(face):
            angle = 2 * np.pi * offset / len(face)
            conf.SetAtomPosition(donor, Point3D(radius * np.cos(angle), radius * np.sin(angle), 2.0))
        for slot, donor in enumerate(sites[1:], 1):
            conf.SetAtomPosition(donor, Point3D(*(2.0 * np.asarray(POLYHEDRA[geometry].vertex_dirs[slot]))))
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        model = Isomer(mol, geometry, sites, lengths="model")
        assert model.cons == expected
        measured = Isomer(mol, geometry, sites, lengths="input")
        centroid = next(iter(measured.haptic))
        assert measured.cons.pulls[tuple(sorted((measured.metal, centroid)))] == pytest.approx(2.0)
        assert measured.cons != expected
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), conf.GetPositions())


def test_lazy_length_source_is_fixed_when_isomers_are_built():
    graph = Chem.MolFromSmiles(_SQUARE_PD)
    expected_model = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar")[0])
    deferred_model = enumerate_isomers(Chem.Mol(graph), "square_planar")[0]
    deferred_model.mol.AddConformer(Chem.Conformer(deferred_model.mol.GetNumAtoms()))
    assert _ml_windows(deferred_model) == expected_model

    geometry = _fake_geometry(_SQUARE_PD)
    expected_input = _ml_windows(enumerate_isomers(Chem.Mol(geometry), "square_planar", lengths="input")[0])
    deferred_input = enumerate_isomers(Chem.Mol(geometry), "square_planar", lengths="input")[0]
    deferred_input.mol.GetConformer().SetAtomPosition(deferred_input.donors[0], Point3D(20, 20, 20))
    assert _ml_windows(deferred_input) == expected_input


def test_input_lengths_require_geometry():
    with pytest.raises(ValueError, match="carries no geometry"):
        enumerate_isomers(Chem.MolFromSmiles(_SQUARE_PD), "square_planar", lengths="input")


def test_input_lengths_do_not_impose_observed_non_donor_contacts():
    mol = Chem.AddHs(rx.parse_smiles("[C-]#[O+]->[Mn]"))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(((3.0, 0.0, 0.0), (1.7, 0.0, 0.0), (0.0, 0.0, 0.0))):
        conf.SetAtomPosition(index, Point3D(*point))
    mol.AddConformer(conf)

    measured = coordination(
        mol,
        2,
        (1, -1),
        "linear",
        25,
        haptic={},
        source=mol,
        lengths="input",
    )
    alternate = coordination(
        mol,
        2,
        (1, -1),
        "linear",
        25,
        haptic={},
        source=mol,
        lengths="model",
    )
    assert measured.floors == alternate.floors


def test_length_source_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        isos = enumerate_isomers(_fake_geometry(_SQUARE_PD), "square_planar", lengths="input")
    assert len(isos) > 1, "the premise: more than one ordering was built"
    said = [r for r in caplog.records if "M-donor windows from" in r.getMessage()]
    assert len(said) == 1, f"the source was announced {len(said)} times"
    assert "input conformer" in said[0].getMessage()
    assert "lengths='input'" in said[0].getMessage()


def test_screen_constraints_skip_force_field_contacts(monkeypatch):
    iso = rx.metal("Cl[Pt](Cl)(N)N", "square_planar")[0]
    context = _metal_constraints.compile_context(iso._graph)

    def fail(*_args, **_kwargs):
        raise AssertionError("screen compilation must not build force-field contacts")

    monkeypatch.setattr(_metal_constraints, "ff_terms", fail)
    screened = iso._constraints(force_field=False, context=context)

    assert screened.distances


def test_joint_shell_uses_native_reach_for_nonbite_real_donor_pairs(monkeypatch):
    from rdkit.ForceField.rdForceField import ForceField

    mol = Chem.MolFromSmiles("[Cu+2].N.N.N.N")
    vertices = (1, 2, 3, 4)
    poly = POLYHEDRA["tetrahedral"]
    cons = Constraints(
        distances={(0, donor): (1.95, 2.05) for donor in vertices},
        angles={(vertices[i], 0, vertices[j]): (90.0, 100.0) for i, j in itertools.combinations(range(4), 2)},
    )
    bounds = _bounds_matrix(mol)
    bounds[4, 3] = bounds[3, 4] = 3.5
    original = cons.copy()
    monkeypatch.setattr(ForceField, "Minimize", lambda *args, **kwargs: 0)
    _metal_constraints._joint_shell_targets(
        0,
        vertices,
        {},
        poly,
        cons,
        {frozenset((0, 1)): (70.0, 90.0)},
        bounds,
        same_ligand={tuple(sorted(pair)) for pair in itertools.combinations(vertices, 2)},
    )
    assert cons == original, "a witness outside a non-bite native span must abstain"


def test_linked_donor_compilation_supplies_the_complete_native_network(monkeypatch):
    from rdkit.ForceField.rdForceField import ForceField

    vertices = (1, 4, 7, 10)
    pairs = set()
    add = ForceField.AddDistanceConstraint

    def record_pair(ff, left, right, *args):
        if left and right:
            pairs.add(tuple(sorted((vertices[left - 1], vertices[right - 1]))))
        return add(ff, left, right, *args)

    monkeypatch.setattr(ForceField, "AddDistanceConstraint", record_pair)
    mol = Chem.MolFromSmiles("[Cu+2].NCCNCCNCCN")
    _metal_constraints.coordination(mol, 0, vertices, "tetrahedral", 29, haptic={})

    assert pairs == set(itertools.combinations(vertices, 2)), "nonadjacent linked donors also constrain reach"


def test_joint_shell_prefers_explicit_real_donor_span(monkeypatch):
    from rdkit.ForceField.rdForceField import ForceField

    mol = Chem.MolFromSmiles("[Cu+2].N.N.N.N")
    vertices = (1, 2, 3, 4)
    cons = Constraints(
        distances={(0, donor): (1.95, 2.05) for donor in vertices} | {(3, 4): (1.1, 1.2)},
        angles={(vertices[i], 0, vertices[j]): (80.0, 100.0) for i, j in itertools.combinations(range(4), 2)},
    )
    rows = []
    add = ForceField.AddDistanceConstraint
    monkeypatch.setattr(ForceField, "AddDistanceConstraint", lambda ff, *args: (rows.append(args), add(ff, *args))[1])
    monkeypatch.setattr(ForceField, "Minimize", lambda *args, **kwargs: 0)
    _metal_constraints._joint_shell_targets(
        0,
        vertices,
        {},
        POLYHEDRA["tetrahedral"],
        cons,
        {frozenset((0, 1)): (70.0, 90.0)},
        _bounds_matrix(mol),
        same_ligand={tuple(sorted(pair)) for pair in itertools.combinations(vertices, 2)},
    )
    assert any(row[:3] == (3, 4, 1.1) and row[3] == 1.2 for row in rows)


@pytest.mark.parametrize(("blocked", "overridden"), [({0}, ()), (set(), ((0, 1),))])
def test_joint_shell_authority_is_a_noop(monkeypatch, blocked, overridden):
    mol = Chem.MolFromSmiles("[Cu+2].N.N.N.N")
    vertices = (1, 2, 3, 4)
    poly = POLYHEDRA["tetrahedral"]
    cons = Constraints(
        distances={(0, donor): (1.95, 2.05) for donor in vertices},
        angles={(vertices[i], 0, vertices[j]): (80.0, 100.0) for i, j in itertools.combinations(range(4), 2)},
    )
    original = cons.copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("held or overridden shell must not be refitted")

    monkeypatch.setattr(_metal_constraints.rdForceFieldHelpers, "CreateEmptyForceFieldForMol", unexpected)
    _metal_constraints._joint_shell_targets(
        0, vertices, {}, poly, cons, {frozenset((0, 1)): (70.0, 90.0)}, _bounds_matrix(mol), blocked, overridden
    )
    assert cons == original


def test_overlapping_tetrahedral_bites_have_origin_enclosing_targets():
    mol = Chem.MolFromSmiles("[Cu+2].N.N.N.N")
    vertices = (1, 2, 3, 4)
    poly = POLYHEDRA["tetrahedral"]
    wanted = {
        (0, 1): (101.5, 117.5),
        (2, 3): (101.5, 117.5),
        (0, 2): (70.0, 91.0),
        (0, 3): (70.0, 91.0),
        (1, 2): (74.0, 104.0),
        (1, 3): (74.0, 104.0),
    }
    radii = (2.425, 2.425, 2.114, 2.088)
    cons = Constraints(
        distances={(0, donor): (radius - 0.05, radius + 0.05) for donor, radius in zip(vertices, radii, strict=True)},
        angles={(vertices[i], 0, vertices[j]): wanted[(i, j)] for i, j in itertools.combinations(range(4), 2)},
    )
    original = {key: hi - lo for key, (lo, hi) in cons.angles.items()}
    ceilings = np.eye(4)
    for (i, j), (_lo, hi) in wanted.items():
        ceilings[i, j] = ceilings[j, i] = np.cos(np.radians(hi))
    assert np.all(ceilings.sum(axis=1) > 0), "original windows force every ray into one open hemisphere"
    _metal_constraints._joint_shell_targets(
        0,
        vertices,
        {},
        poly,
        cons,
        {frozenset(pair): wanted[pair] for pair in ((0, 2), (0, 3), (1, 2), (1, 3))},
        _bounds_matrix(mol),
        same_ligand=(),
    )
    assert all(0.0 <= lo <= hi <= 180.0 for lo, hi in cons.angles.values())
    assert {key: hi - lo for key, (lo, hi) in cons.angles.items()} == pytest.approx(original)
    pulls = np.array([cons.pulls[key] for key in cons.angles], dtype=float)
    assert np.all(np.isfinite(pulls))
    assert all(lo <= cons.pulls[key] <= hi for key, (lo, hi) in cons.angles.items())
    gram = np.eye(4)
    for i, j in itertools.combinations(range(4), 2):
        gram[i, j] = gram[j, i] = np.cos(np.radians(cons.pulls[(vertices[i], 0, vertices[j])]))
    values, basis = np.linalg.eigh(gram)
    assert values[0] == pytest.approx(0.0, abs=1e-10)
    assert values[1] > 1e-6
    positions = basis[:, -3:] * np.sqrt(values[-3:]) * np.array(radii)[:, None]
    weights = np.linalg.solve(np.vstack([positions.T, np.ones(4)]), [0, 0, 0, 1])
    assert np.all(weights > 0)


def test_auxiliary_shell_reflection_preserves_published_angles(monkeypatch):
    from rdkit.ForceField.rdForceField import ForceField

    mol, vertices, poly, bites, original = _planar_bite_case("trigonal_bipyramidal")
    create = _metal_constraints.rdForceFieldHelpers.CreateEmptyForceFieldForMol
    minimize = ForceField.Minimize
    shells, parities, fitted = [], [], []

    def capture(shell):
        shells.append(shell)
        return create(shell)

    def solve(ff, *args, **kwargs):
        status = minimize(ff, *args, **kwargs)
        conf = shells[-1].GetConformer()
        positions = conf.GetPositions()
        if reflect:
            positions[:, 0] *= -1
            conf.SetPositions(positions)
        rays = positions[1:] - positions[0]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        parities.append(_metal_constraints.orientation_parity(rays, poly.vertex_dirs))
        return status

    monkeypatch.setattr(_metal_constraints.rdForceFieldHelpers, "CreateEmptyForceFieldForMol", capture)
    monkeypatch.setattr(ForceField, "Minimize", solve)
    for reflect in (False, True):  # noqa: B007 - read by the native solver callback
        cons = original.copy()
        _metal_constraints._joint_shell_targets(
            0, vertices, {}, poly, cons, bites, _bounds_matrix(mol), same_ligand={(1, 4)}
        )
        assert cons.pulls
        assert cons.copy(angles=original.angles, pulls=original.pulls) == original
        fitted.append(cons)
    assert parities[0] == -parities[1]
    assert fitted[0] == fitted[1]
