"""Test coordination and vacant-site constraint construction."""

from __future__ import annotations

import csv
import itertools
import logging
import math
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_constraints, metal_core, stereo
from rxembed.bounds import EmbedParams, bounds_matrix
from rxembed.constraints import FIX_ANGLE_TOL, Constraints
from rxembed.embed import _coordination_state_failure
from rxembed.metal_constraints import (
    CoordinationSphere,
    _add_ligand_ez,
    _add_umbrella,
    _bite_window_and_reach,
    compile_constraints,
    compile_context,
    coordination,
)
from rxembed.metal_core import VACANT, HapticSite, MetalState, metal_indices
from rxembed.metal_donor_orient import COPLANAR_CAP
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer
from rxembed.metal_perceive import classify_geometry
from rxembed.metal_polyhedron import CHELATE_SPAN_ANGLE, POLYHEDRA, point_group, vertex_angle, vertex_dirs
from rxembed.metal_slots import SPAN_TOL, chelate_bite_window
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.ensemble import Ensemble
from rxembed.relax import _ff_surrogate
from tests.conftest import EXAMPLES_DIR, TMQMG_DIR
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
    return vertex_angle(pos[iso.vertices[i]] - pos[iso.metal], pos[iso.vertices[j]] - pos[iso.metal])


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
    iso = Isomer.from_state(mol, states)
    return Ensemble(mol, [0], iso=iso), iso


# --- the polyhedron angles are realised, not merely stated -----------------------------------------------

# One real complex per low-CN shape: a 14-electron T-shaped Rh(I) phosphine (its two P trans, Cl the stem) and
# Fe(CO)4 (the 16e d8 C2v sawhorse). The CN3 pyramid needs its own hold and has its own test below.
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


# A bare CN3 pyramid's ±8° D-M-D window is flat-bottomed, with no restoring force once a phosphine rides the
# wall into trigonal_planar's basin, so `mechanisms.Umbrella` holds the scale-free improper instead. Two
# fixtures: the trimethyl case is cheap, and PPh3's DG seed comes out exactly planar, proving the hold
# re-forms a pyramid rather than only keeping one.
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
    failure = next(iter(failures))
    assert failure.kind == "coordination_shape"
    assert str(failure).endswith("donors fall into a plane")
    assert next(iter(failures.values())) == ens.ids


def test_secondary_planar_centre_is_checked():
    ens, iso = _two_centre_ensemble("tetrahedral", "square_planar")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    donor = secondary.vertices[0]
    conf.SetAtomPosition(donor, Point3D(*(pos[donor] + np.array([0.0, 0.0, 2.0]))))
    assert _coordination_state_failure(ens._mol, 0, iso) is not None, "the puckered secondary plane passed the gate"


def test_secondary_nonplanar_centre_flattening_is_rejected():
    ens, iso = _two_centre_ensemble("square_planar", "trigonal_pyramidal")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    conf.SetAtomPosition(secondary.atom, Point3D(*np.mean(pos[list(secondary.vertices)], axis=0)))
    assert _coordination_state_failure(ens._mol, 0, iso) is not None


def test_one_haptic_site_does_not_define_a_nonplanar_state():
    ens, iso = _two_centre_ensemble("tetrahedral", "tetrahedral")
    secondary = iso.centres[1]
    face = HapticSite(tuple(secondary.vertices))
    sparse = secondary._replace(vertices=(face, None, None, None))
    iso = Isomer.from_state(iso.mol, (iso.centres[0], sparse))
    assert _coordination_state_failure(ens._mol, 0, iso) is None


def test_planar_donor_keeps_rdkit_native_substituent_floor_during_relax():
    iso = rx.metal("[Pd+2](<-[n]1ccccc1)(<-[n]1ccccc1)(<-[Cl-])<-[Cl-]", "square_planar")[0]
    bounds = bounds_matrix(iso.mol)
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
    floor = float(bounds_matrix(iso.mol)[protons[1], protons[0]])

    assert iso.cons.floors[tuple(protons)] == pytest.approx(floor)
    ens = rx.embed(iso, n=1, seed=42)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    # A floor is a one-sided wall, not a target: the relax settles a measured 0.0035 A inside it at this
    # seed, within geom_check's own floor slop, so the assertion tolerance matches that undershoot.
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
    path = EXAMPLES_DIR / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=fixed)[0]

    assert ((1, 64, 3) in iso.cons.angles) is first_present
    assert (1, 65, 4) in iso.cons.angles


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mnh_carbonyls_pass_the_relax_contract():
    path = EXAMPLES_DIR / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=[1, 2, 5])[0]
    ens = rx.embed(iso, n=1, seed=0xF00D)

    assert ens.ids
    assert not ens.unrelaxed
    conf = ens._mol.GetConformer(ens.ids[0])
    for atoms in ((1, 64, 3), (1, 65, 4)):
        assert ens.cons.angles[atoms] == (174.0, 180.0)
        assert GetAngleDeg(conf, *atoms) >= 174.0 - FIX_ANGLE_TOL
        pair = atoms[:2]
        value = float(np.linalg.norm(conf.GetPositions()[pair[0]] - conf.GetPositions()[pair[1]]))
        lo, hi = ens.cons.distances[pair]
        assert lo <= value <= hi


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_retained_geometry_uses_the_same_donor_orientation_compiler():
    from rxembed.metal_isomer import from_geometry

    path = EXAMPLES_DIR / "mnh.xyz"
    mol = Chem.AddHs(rx.read_xyz(str(path)), addCoords=True)
    cons = from_geometry(mol, center="all").cons

    assert cons.angles[(1, 64, 3)] == (174.0, 180.0)
    assert cons.angles[(1, 65, 4)] == (174.0, 180.0)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_two_fixed_donors_do_not_own_their_angle_when_the_metal_is_free():
    path = EXAMPLES_DIR / "mnh.xyz"
    isos = rx.metal(str(path), center="all", fix=[6, 64])

    assert any(any(key[1] == 1 and {key[0], key[2]} == {6, 64} for key in iso.cons.angles) for iso in isos)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mn_fe_state_passes_the_relax_contract():
    path = EXAMPLES_DIR / "mn-h2.xyz"
    reference = rx.read_xyz(str(path), metal_charges={0: 2, 1: 1})
    ens = rx.embed(reference, fix=[1, 5, 63, 64, 65, 66], stereo={"planar": "racemic"}, n=1, seed=0xF00D)

    assert ens.ids
    assert not ens.unrelaxed
    assert ens._metal_states() == {ens.ids[0]: True}
    assert ens.cons.angles[(1, 61, 3)] == (174.0, 180.0)
    assert ens.cons.angles[(1, 62, 4)] == (174.0, 180.0)
    conf = ens._mol.GetConformer(ens.ids[0])
    assert GetAngleDeg(conf, 1, 61, 3) >= 174.0 - FIX_ANGLE_TOL
    assert GetAngleDeg(conf, 1, 62, 4) >= 174.0 - FIX_ANGLE_TOL
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


def test_held_opposed_bites_retain_a_valid_shared_angle_witness():
    # Opposed bites need no shared bite value: the relaxed shell fixes cis/trans jointly from BOTH bite
    # windows (metal_polyhedron.relaxed_shell), not from one shared `180 - bite`. Mutating back to that
    # per-bite formula breaks this witness on these elongated (O/S kappa2-corrected) legs.
    iso = Isomer(Chem.MolFromSmiles("[Ag+]12(<-[O-]C[O-]->1)<-NCCN->2"), "square_planar", {0: 1, 1: 3, 2: 4, 3: 7})
    cons = coordination(CoordinationSphere(iso.mol, iso.metal, 47, tuple(iso.vertices), {}, POLYHEDRA[iso.geometry]))
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
    # The one pull convention (the relaxed shell at bite-window midpoints) keeps every pull inside its window.
    for key, target in cons.pulls.items():
        if len(key) != 3 or key[1] != iso.metal:
            continue
        a, _metal, b = key
        lo, hi = windows[frozenset((a, b))]
        assert lo - 1e-8 <= target <= hi + 1e-8


def test_tridentate_normal_arm_keeps_a_common_shell_through_cleanup(tmp_path):
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "trigonal_bipyramidal")
        if 0 in candidate.vertices[:2] and {3, 6} <= set(candidate.vertices[2:])
    )
    cons = iso.cons
    targets = {key: value for key, value in cons.pulls.items() if len(key) == 3}
    # Every compiled row gets a pull toward the shared shell's own image, including a spectator row the
    # construction leaves at (near enough) ideal: that row's only restoring force is still this pull (DULPUV).
    assert len(targets) == 10
    for key, value in targets.items():
        lo, hi = cons.angles[key] if key in cons.angles else cons.angles[key[::-1]]
        assert lo <= value <= hi
    equatorial = set(iso.vertices[2:])
    # The mixed axial/equatorial bite (0 is axial) tilts the shared hinge donor slightly out of the
    # equatorial plane, so the trio no longer closes to exactly 360 the way an equatorial-only fan would.
    assert sum(value for (a, _, b), value in targets.items() if {a, b} <= equatorial) == pytest.approx(360.0, abs=0.5)

    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    assert not ensemble.unrelaxed
    geom.check(ensemble.mol).assert_ok()
    assert ensemble.cons.pulls == cons.pulls
    pytest.importorskip("xyzgraph")
    path = tmp_path / "tridentate_normal_arm.xyz"
    Chem.MolToXYZFile(ensemble.mol, str(path))
    fresh = rx.read_xyz(str(path), charge=Chem.GetFormalCharge(ensemble.mol), bond_orders="xyz2mol")
    assert rx.cxsmiles(fresh) == rx.cxsmiles(iso)


def test_tridentate_fan_compiles_shared_targets_and_survives_cleanup():
    """A hinge donor at the apex bites both trans basal arms; every compiled pull stays inside its own window
    and the four basal donors it targets are coplanar.
    """
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "square_pyramidal")
        if candidate.vertices[0] == 3 and set(candidate.vertices[1::2]) == {0, 6}
    )
    targets = {key: value for key, value in iso.cons.pulls.items() if len(key) == 3}
    assert len(targets) == 10
    for key, value in targets.items():
        lo, hi = iso.cons.angles[key] if key in iso.cons.angles else iso.cons.angles[key[::-1]]
        assert lo - 1e-8 <= value <= hi + 1e-8

    # Rebuild each basal donor's 3D point from its compiled M-metal distance and pairwise pull angle (the
    # metal at the origin), then check the four points share one plane independent of the metal's own height.
    basal = [0, 8, 6, 9]
    radius = {d: sum(iso.cons.distances[tuple(sorted((d, iso.metal)))]) / 2 for d in basal}
    gram = np.zeros((len(basal), len(basal)))
    for i, donor in enumerate(basal):
        gram[i, i] = radius[donor] ** 2
    for i, j in itertools.combinations(range(len(basal)), 2):
        a, b = basal[i], basal[j]
        angle = targets.get((a, iso.metal, b), targets.get((b, iso.metal, a)))
        gram[i, j] = gram[j, i] = radius[a] * radius[b] * np.cos(np.radians(angle))
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:3]
    points = eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0, None))
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    assert spread[-1] < 0.05 * spread[0], "the four basal donors must lie in one plane"


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
        cons = compile_constraints(iso, external)
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
    five_ring_census = (70.0, 91.0)
    for k, window in intra.items():
        assert five_ring_census[0] <= window[0] <= window[1] <= five_ring_census[1], (
            f"{k}: an en 5-ring must intersect its graph reach with its bite prior, got {window}"
        )
    inter = {k: v for k, v in iso.cons.angles.items() if k[1] == iso.metal and frag[k[0]] != frag[k[2]]}
    assert inter, "the fixture must also state an inter-ligand pair, or the contrast is untested"
    pads = {(max(0.0, a - 8.0), min(180.0, a + 8.0)) for a in ideal}  # ±8° about a tabulated angle, clamped
    for k, window in inter.items():
        # A wide intra-chelate bite (the native reach, `seated_bites`) can pull `relaxed_shell`'s corner
        # images further than the plain pad, and every already-compiled row only widens for that (`coordination`
        # never narrows one), so an inter-ligand row need only contain its pad, not equal it.
        assert any(window[0] <= lo and window[1] >= hi for lo, hi in pads), (
            f"{k}: an inter-ligand pair must contain a polyhedron angle ±8° pad, got {window}"
        )

    # the window comes from the RING, so a pair with no shared backbone gets none at all
    cl = [d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "Cl"]
    assert chelate_bite_window(iso.mol, cl[0], cl[1]) is None


def test_chelate_bite_window_intersects_a_partial_rdkit_reach_window():
    """`_bite_window_and_reach` intersects the ring-size census with the backbone reach triangle for `window`,
    and reports the wider triangle back as `reach` for `bounded_bites`'s gap box.
    """
    mol = Chem.MolFromSmiles("NCCN")
    matrix = np.zeros((4, 4))

    def span(angle):
        return math.sqrt(8.0 - 8.0 * math.cos(math.radians(angle)))

    matrix[3, 0] = span(80.0) + SPAN_TOL
    matrix[0, 3] = span(100.0) - SPAN_TOL

    window, reach = _bite_window_and_reach(mol, (0,), (3,), (0, 3), matrix, (2.0, 2.0), (0.0, 180.0))

    assert window == pytest.approx((80.0, 91.0))
    assert reach == pytest.approx((80.0, 100.0))


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
    five_ring_census = (70.0, 91.0)
    assert bites[frozenset((0, 3))] == five_ring_census
    assert bites[frozenset((3, 6))] == five_ring_census


def test_tris_pyrazolyl_borate_zinc_chloride_keeps_its_boron_nitrogen_bonds():
    """A fac tripod's spectator rows must widen with the bite (metal_polyhedron.relaxed_shell), or the three
    ideal +- 8 Cl-Zn-N rows jointly force the N-Zn-N bite outside its own window and tear the ligand's N-B
    bonds finding room. Today N-B reaches 1.878/1.538 = 1.221 x RDKit's own upper bound; the union row keeps
    it under 1.05x.
    """
    iso = rx.metal("[Cl-]->[Zn+2]12<-[n]3cccn3[BH-](n3ccc[n]->13)n3ccc[n]->23", "tetrahedral")[0]
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    mol = ensemble.mol
    positions = mol.GetConformer(ensemble.ids[0]).GetPositions()
    metals = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 30}
    rw = Chem.RWMol(mol)
    for bond in list(mol.GetBonds()):
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & metals:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    ligand = rw.GetMol()
    ligand.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(ligand)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
    for bond in ligand.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if 1 in (ligand.GetAtomWithIdx(i).GetAtomicNum(), ligand.GetAtomWithIdx(j).GetAtomicNum()):
            continue  # RDKit's ordinary X-H bounds do not describe a bridging hydrogen
        ratio = np.linalg.norm(positions[i] - positions[j]) / bounds[i][j]
        assert ratio < 1.05, f"{ligand.GetAtomWithIdx(i).GetSymbol()}-{ligand.GetAtomWithIdx(j).GetSymbol()} {ratio}"


def test_capped_tp_tungsten_tricarbonyl_holds_carbonyl_off_the_spectator_pyrazolyl_nitrogen(tmp_path):
    """A Tp tungsten tricarbonyl's spectator N-W-C rows narrow toward the bite, but never past the angle
    where a carbonyl carbon and a pyrazolyl nitrogen on different ligands would close inside their own
    van der Waals contact floor (MADROZ).
    """
    cx = (
        "CC#[N]->[W+2]12(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[n]3c(C)cc(C)n3[BH-](n3c(C)cc(C)[n]->13)"
        "n1c(C)cc(C)[n]->21 |atomProp:2.atomNote.s6:3.atomNote.COC-delta:4.atomNote.s1:6.atomNote.s4:"
        "8.atomNote.s5:10.atomNote.s0:24.atomNote.s2:31.atomNote.s3|"
    )
    iso = rx.metal(cx)[0]
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    pytest.importorskip("xyzgraph")
    path = tmp_path / "madroz.xyz"
    Chem.MolToXYZFile(ensemble.mol, str(path), confId=ensemble.ids[0])
    fresh = rx.read_xyz(str(path), charge=1, connectivity="xyzgraph", bond_orders="xyz2mol")

    def edges(mol):
        return {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in mol.GetBonds()
            if b.GetBondType() != Chem.BondType.ZERO
        }

    formed = edges(metal_core.canonical_metal_graph(fresh)) - edges(ensemble.mol)
    assert not formed, formed


# --- a haptic bite reads the native ligand reach, same as a sigma bite (metal_constraints.seated_bites) ------


def _worst_ligand_bond_ratio(mol, positions, ref_positions=None):
    """Return the worst (ratio, label) over every non-hydrogen ligand bond, metal bonds excluded.

    Without `ref_positions` the reference is RDKit's own metal-free upper bound for that bond (a SMILES input
    has no measured length); with it, the reference is that bond's length in `ref_positions` (a crystal read).
    """
    metals = set(metal_indices(mol))
    rw = Chem.RWMol(mol)
    for bond in list(mol.GetBonds()):
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & metals:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    ligand = rw.GetMol()
    ligand.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(ligand)
    worst = (0.0, "")
    if ref_positions is None:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
    for bond in ligand.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if 1 in (ligand.GetAtomWithIdx(i).GetAtomicNum(), ligand.GetAtomWithIdx(j).GetAtomicNum()):
            continue  # RDKit's ordinary X-H bounds do not describe a bridging hydrogen
        length = float(np.linalg.norm(positions[i] - positions[j]))
        target = bounds[i][j] if ref_positions is None else float(np.linalg.norm(ref_positions[i] - ref_positions[j]))
        label = f"{ligand.GetAtomWithIdx(i).GetSymbol()}{i}-{ligand.GetAtomWithIdx(j).GetSymbol()}{j}"
        worst = max(worst, (length / target, label))
    return worst


def test_norbornadiene_nickel_dicarbonyl_keeps_its_carbon_bonds():
    """A haptic diene's bite pair has no ring-size census prior to intersect with, so `_bite_window_and_reach`
    must replace the plain ideal +- 8 deg row outright wherever the native ligand reach excludes part of it.
    Today the bicycle's bridging C-C bond stretches to 1.148x RDKit's own metal-free upper bound.
    """
    smiles = (
        "[O+]#[C-]->[Ni]123(<-[C-]#[O+])<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31 "
        "|atomProp:1.atomNote.s0:2.atomNote.TET:3.atomNote.s1:5.atomNote.s2:6.atomNote.s2:10.atomNote.s3:11.atomNote.s3|"
    )
    iso = rx.metal(smiles)[0]
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    positions = ensemble.mol.GetConformer(ensemble.ids[0]).GetPositions()
    ratio, label = _worst_ligand_bond_ratio(ensemble.mol, positions)
    assert ratio < 1.05, f"{label} {ratio}"


@pytest.mark.parametrize(
    "smiles",
    [
        "[Cl-]->[Pt+2]123(<-[Cl-])<-[CH2]=[CH]->1CCCCCC[CH]->2=[CH2]->3",
        "[Cl-]->[Pt+2]123(<-[Cl-])<-[CH]4=[CH]->1C=C[CH]->2=[CH]->3C=C4",
    ],
    ids=["long_tether", "cyclooctatetraene"],
)
def test_cis_diene_faces_keep_a_cis_window_on_platinum(smiles):
    """Every cis diene isomer's face-face row must stay below `CHELATE_SPAN_ANGLE`, or a real embed can force
    the two coordinated faces toward `PtCl2`'s trans vertices and tear the ring finding room. "cis" is read
    from a 90 deg polyhedron slot pair in `iso.vertices`, independent of the compiled row this test checks.
    """
    isomers = rx.metal(smiles, "square_planar")
    dirs = POLYHEDRA["square_planar"].vertex_dirs
    checked = 0
    for iso in isomers:
        faces = list(iso.haptic)
        assert len(faces) == 2
        slots = {donor: k for k, donor in enumerate(iso.vertices) if donor != VACANT}
        ideal = vertex_angle(dirs[slots[faces[0]]], dirs[slots[faces[1]]])
        if ideal >= CHELATE_SPAN_ANGLE:
            continue  # the trans arrangement: not this test's concern
        checked += 1
        key = next(k for k in iso.cons.angles if k[1] == iso.metal and {k[0], k[2]} == set(faces))
        window = iso.cons.angles[key]
        assert window[1] < CHELATE_SPAN_ANGLE, window
    assert checked, "the fixture must enumerate at least one cis isomer, or the contrast is untested"


def _tmqmg_isomer(tmqmg_id):
    """Load one tmQMg structure's crystal-matching isomer, positions included."""
    charges = {
        row["id"]: int(row["charge"]) for row in csv.DictReader((TMQMG_DIR / "tmQMg_properties_and_targets.csv").open())
    }
    mol = rx.read_xyz(
        str(TMQMG_DIR / "xyz" / f"{tmqmg_id}.xyz"),
        charge=charges[tmqmg_id],
        connectivity="xyzgraph",
        bond_orders="xyz2mol",
    )
    return mol, rx.metal(mol, observed_only=True)[0]


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.skipif(not TMQMG_DIR.is_dir(), reason="needs a local tmQMg clone")
def test_tethered_norbornadiene_molybdenum_keeps_its_crystal_carbon_bonds():
    """EGUZIP: a norbornadiene tethered through a phosphine to the same Mo. Today the model M-L legs and a
    tight ideal +- 8 deg row on the diene bite stretch its bridging C22-C23 bond to 1.178x the crystal length.
    """
    mol, iso = _tmqmg_isomer("EGUZIP")
    crystal = mol.GetConformer().GetPositions()
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    positions = ensemble.mol.GetConformer(ensemble.ids[0]).GetPositions()
    ratio, label = _worst_ligand_bond_ratio(ensemble.mol, positions, crystal)
    assert ratio < 1.03, f"{label} {ratio}"


def test_tub_cyclooctatetraene_iridium_face_row_admits_its_crystal_bite():
    """A tub-shaped cyclooctatetraene's two eta2,eta2 faces on Ir compile a face-face window admitting the
    crystal centroid-centroid bite of 87.9 deg (OMANUL).
    """
    smiles = "CS(C)(=O)->[Ir+]123(<-[Cl-])<-C4=CC=C->1C->2=CC=C->34"
    cis_windows = set()
    for iso in rx.metal(smiles):
        faces = list(iso.haptic)
        key = next(k for k in iso.cons.angles if k[1] == iso.metal and {k[0], k[2]} == set(faces))
        window = iso.cons.angles[key]
        if window[1] < CHELATE_SPAN_ANGLE:  # the cis face-face row, not the trans spectator row
            cis_windows.add(window)
    assert cis_windows
    assert all(window[0] <= 88.0 <= window[1] for window in cis_windows)


def test_five_ring_cn_chelate_bites_admit_the_crystal_bite():
    """Three independent 5-ring C^N chelate bites on Ir each compile a window admitting the crystal's
    compressed 79.0/80.9/80.1 deg bites (FOPSOT).
    """
    smiles = "Cn1ccn2->[Ir+3]34(<-[Cl-])(<-[c-]5cc(F)ccc5-c5ccccn->35)<-[c-]3c(-c5ccccn->45)c(F)cc(F)c3-c12"
    iso = rx.metal(smiles)[0]
    bites = [
        window
        for (left, metal, right), window in iso.cons.angles.items()
        if metal == iso.metal
        and left not in iso.haptic
        and right not in iso.haptic
        and chelate_bite_window(iso.mol, left, right) is not None
    ]
    assert len(bites) == 3
    assert all(window[0] <= 79.0 <= window[1] for window in bites)


def test_diselenophosphate_four_ring_bite_follows_its_backbone():
    """WIRGOV's Se-P-Se 4-ring bite: the backbone triangle (83.9-86.3 deg at the model Au-Se leg) misses the
    ring-size census window (58-81 deg) by 0.04 deg once SPAN_TOL-widened reach is not the thing decided on.
    `_bite_window_and_reach` must then hand out the triangle, not a spurious sliver overlap, so the input's
    own arrangement stays admissible.
    """
    isomers = rx.metal("CCOP1(OCC)=[Se]->[Au+3](<-[I-])(<-[I-])<-[Se-]1", "square_planar")
    assert len(isomers) == 1


@pytest.mark.parametrize(
    ("geometry", "coordination_number"),
    [("trigonal_bipyramidal", 5), ("square_pyramidal", 5), ("octahedral", 6)],
)
def test_coordination_constraints_are_invariant_under_proper_rotation(geometry, coordination_number):
    mol = Chem.MolFromSmiles("[Fe+3].[Cl-].[Br-].[I-].N.P.S")
    metal, donors = 0, tuple(range(1, coordination_number + 1))

    def sphere_angles(vertices):
        cons = coordination(CoordinationSphere(mol, metal, 26, vertices, {}, POLYHEDRA[geometry]))
        return {(min(a, b), max(a, b)): window for (a, centre, b), window in cons.angles.items() if centre == metal}

    expected = sphere_angles(donors)
    for rotation in point_group(vertex_dirs(geometry))[0]:
        assert sphere_angles(tuple(donors[rotation[i]] for i in range(coordination_number))) == expected


def test_directly_bonded_donors_use_their_distance_triangle():
    """A bonded donor pair with a third donor neighbour on each side keeps its native distance triangle.

    ZUDWUQ's encircling cyclo-As6 ring on Ni: the pair rule (`metal_core.haptic_sites`) only merges a
    bonded donor pair with no third donor neighbour, so each As keeps its own donor slot here and its two
    ring bonds get a native distance row instead of an independent M-centred angle.
    """
    smiles = "[Ni+2](<-[AsH]16)(<-[AsH]12)(<-[AsH]23)(<-[AsH]34)(<-[AsH]45)(<-[AsH]56)"
    iso = rx.metal(smiles, "hexagonal_planar")[0]

    assert iso.haptic == {}
    for left, right in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]:
        assert (left, iso.metal, right) not in iso.cons.angles
        assert (left, right) in iso.cons.distances


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
    measured = vertex_angle(*rays)
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

    sphere = CoordinationSphere(
        mol,
        metal,
        26,
        (boron, bridge, nitrogen),
        {},
        POLYHEDRA["trigonal_planar"],
        pos=mol.GetConformer().GetPositions(),
    )
    cons = coordination(sphere)

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


def test_tridentate_outer_trans_pair_uses_the_relaxed_shell_window():
    # The two outer donors (0, 6) share no bite of their own; each is bitten only to the central donor (3).
    # Their row is the free consequence of the relaxed shell (metal_polyhedron.relaxed_shell), not the old
    # hardcoded TRANS_ANGLE fan row: both bites pulling the shared hinge inward widens this pair's window
    # well below the axial-axial ideal of 180 +- 8.
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "trigonal_bipyramidal")
        if set(candidate.vertices[:2]) == {0, 6}
    )

    lo, hi = iso.cons.angles[(0, iso.metal, 6)]
    # abs=2*FIX_ANGLE_TOL, not 1e-6: relaxed_shell now converges each bite to FIX_ANGLE_TOL, not 1e-8, and
    # this row is two bites' compounded consequence, not one of them.
    assert lo == pytest.approx(140.0, abs=2 * FIX_ANGLE_TOL)
    assert hi == pytest.approx(180.0)


def test_square_planar_opposed_chelate_targets_have_a_common_geometry():
    mol = Chem.MolFromSmiles("[Ag+]12(<-[O-]C[O-]->1)<-NCCN->2")
    iso = Isomer(mol, "square_planar", {0: 1, 1: 3, 2: 4, 3: 7})
    angles = {
        frozenset((left, right)): window
        for (left, metal, right), window in iso.cons.angles.items()
        if metal == iso.metal
    }

    four_ring_census, five_ring_census = (58.0, 81.0), (70.0, 91.0)
    bites = [angles[frozenset((1, 3))], angles[frozenset((4, 7))]]
    for window, prior in zip(bites, (four_ring_census, five_ring_census), strict=True):
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
        cons = coordination(CoordinationSphere(mol, 0, 30, vertices, {}, POLYHEDRA["tetrahedral"]))
        angles = {tuple(sorted((a, b))): window for (a, metal, b), window in cons.angles.items() if metal == 0}
        if expected is None:
            expected = angles
        # `pytest.approx` does not recurse into dict VALUES that are themselves tuples (it falls back to
        # plain equality), so compare each window individually rather than the whole dict at once.
        assert angles.keys() == expected.keys()
        for key, window in angles.items():
            assert window == pytest.approx(expected[key])
        for (a, b), window in angles.items():
            target = cons.pulls[min((a, 0, b), (b, 0, a))]
            assert window[0] - 1e-8 <= target <= window[1] + 1e-8
        # The pull convention (the relaxed shell at bite-window midpoints), not a window's own arithmetic
        # midpoint, is the joint witness: a union-widened window need not be centred on it.
        gram = np.eye(len(donors))
        for a, b in angles:
            i, j = donors.index(a), donors.index(b)
            target = cons.pulls[min((a, 0, b), (b, 0, a))]
            gram[i, j] = gram[j, i] = np.cos(np.radians(target))
        spectrum = np.linalg.eigvalsh(gram)
        assert spectrum[0] == pytest.approx(0, abs=1e-8), "four unit rays need rank at most three"
        assert spectrum[1] > 0, "the witness must not collapse into a plane"
    order = list(reversed(range(mol.GetNumAtoms())))
    renamed = coordination(
        CoordinationSphere(
            Chem.RenumberAtoms(mol, order),
            order.index(0),
            30,
            tuple(order.index(i) for i in donors),
            {},
            POLYHEDRA["tetrahedral"],
        )
    )
    assert {
        tuple(sorted((order[a], order[b]))): window
        for (a, metal, b), window in renamed.angles.items()
        if order[metal] == 0
    } == pytest.approx(expected)


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
    iso = rx.metal("C=[N]1Cc2cccc[n]2->[Cu+]<-12<-[N](Cc1cccc[n]->21)=C", "tetrahedral")[0]
    assert compile_constraints(iso, context=compile_context(iso.graph)) == iso.cons
    spectator = iso.graph.GetNumAtoms()
    mol = Chem.CombineMols(iso.graph, Chem.MolFromSmiles("[C]"))
    Chem.GetSymmSSSR(mol)
    unnamed = MetalState(spectator, 26, 0, "", ())
    cons = Isomer.from_state(
        mol,
        (*iso.centres, unnamed),
        [*iso.donor_bonds, (2, spectator)],
        constrained_metals=iso.constrained_metals,
        stereo_label=iso.stereo_label,
    ).cons
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


def test_eta4_naphthalene_hinge_folds_its_fusion_carbons_off_the_metal():
    """BAMROX: an eta4-bound naphthalene folds at its hinge, standing the unbound fusion carbons off Rh.

    Without the hinge push the fusion carbons sit at 1.11 to 1.17x the Rh-C covalent sum (below the census p5
    of 1.20 for a conjugated hinge); the push must clear that floor at every seed.
    """
    smi = "c1ccc([P]2(CCO[c]34->[Rh+]<-2567<-[cH]3[cH]->5[c]->6(OCC[P]->7(c2ccccc2)c2ccccc2)c2ccccc24)c2ccccc2)cc1"
    iso = rx.metal(smi)[0]
    r_rh_c = Chem.GetPeriodicTable().GetRcovalent(45) + Chem.GetPeriodicTable().GetRcovalent(6)
    for seed in (42, 7, 1234, 2026, 99):
        ens = rx.embed(iso, n=1, seed=seed)
        mol, pos = ens.mol, ens.mol.GetConformer(ens.ids[0]).GetPositions()
        rh = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Rh")
        face = {n.GetIdx() for n in mol.GetAtomWithIdx(rh).GetNeighbors() if n.GetSymbol() == "C"}
        rings = [set(r) for r in Chem.GetSymmSSSR(metal_core.ligand_graph(mol))]  # metal-stripped: no false hinge at Rh
        fusion_carbons = {
            x.GetIdx()
            for d in face
            for x in mol.GetAtomWithIdx(d).GetNeighbors()
            if x.GetIdx() not in face and any({d, x.GetIdx()} <= r for r in rings)
        }
        assert fusion_carbons, f"seed {seed}: no unbound fusion carbon beside the eta4 face"
        for x in fusion_carbons:
            ratio = float(np.linalg.norm(pos[x] - pos[rh])) / r_rh_c
            assert ratio >= 1.20, f"seed {seed}: fusion carbon {x} at {ratio:.3f}x the Rh-C covalent sum"


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
    # Two independent monodentate donors, not bonded to each other: a bonded N,N pair is now one haptic
    # site under the pair rule (`metal_core.haptic_sites`), which would drop their per-donor fold term.
    donors = (rw.AddAtom(Chem.Atom("N")), rw.AddAtom(Chem.Atom("N")))
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
        Umbrella().uff_terms(ff, cons, mol.GetConformer(), 1.0)
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
        assert {stereo.bond_stereo(iso.stereo_label).popitem()[1] for iso in isomers} == {"E", "Z"}
        for iso in isomers:
            targets = stereo.metal_referenced_ez(iso.mol, iso.stereo_label, iso.donor_bonds)
            ((_, (donor, carbon, metal, _ref, _ligand_ref, wanted)),) = targets.items()
            restored = Chem.Mol(iso.mol)
            iso.restore(restored)
            restored = metal_core.connect_metal(restored, iso.donor_bonds)
            assert stereo.metal_referenced_ez(restored, iso.stereo_label, iso.donor_bonds) == targets
            rows = iso.cons.coplanar
            ((_, _, _, _ref, anchor, _cap),) = [row for row in rows if row[0] == metal]
            assert anchor == (180.0 if wanted == "E" else 0.0)
            assert {row[4] for row in rows if row[1:3] == (donor, carbon)} == {0.0, 180.0}


def test_coordinated_nh_imine_ez_survives_donor_orientation_ablation():
    source = Chem.AddHs(Chem.MolFromSmiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"))
    isomer = enumerate_isomers(source, "square_planar")[0]
    targets = stereo.metal_referenced_ez(isomer.mol, isomer.stereo_label, isomer.donor_bonds)
    assert targets

    cons = compile_constraints(isomer, params=EmbedParams(donor_orientation=False))
    ((_, (_, _, metal, _ref, _ligand_ref, wanted)),) = targets.items()
    rows = [row for row in cons.coplanar if row[0] == metal]
    assert rows
    assert any(row[4] == (180.0 if wanted == "E" else 0.0) for row in rows)


def test_n_substituted_imine_keeps_ordinary_ligand_ez():
    iso = rx.metal(r"C/C=N(/C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar")[0]

    assert stereo.bond_stereo(iso.stereo_label)
    assert not stereo.metal_referenced_ez(iso.mol, iso.stereo_label, iso.donor_bonds)


def test_ligand_ez_uses_rdkit_reference_geometry_not_absolute_cip():
    mol = Chem.MolFromSmiles("FC(Cl)=C(Br)I")
    bond = next(bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    bond.SetStereoAtoms(0, 5)
    bond.SetStereo(Chem.BondStereo.STEREOE)
    label = stereo.defined_stereo_label(mol)
    assert label.endswith(":Z"), "the regression requires raw-trans references whose absolute CIP label is Z"

    cons = Constraints()
    _add_ligand_ez(cons, mol, label, ())

    assert len(cons.coplanar) == 1
    row = cons.coplanar[0]
    assert row[:4] == (0, 1, 3, 5)
    assert row[4] == 180.0


def test_eta2_alkene_keeps_stated_ez_during_relaxation():
    iso = rx.metal(r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", "square_planar")[0]
    pair = next(iter(stereo.bond_stereo(iso.stereo_label)))
    rows = [row for row in iso.cons.coplanar if frozenset(row[1:3]) == pair]

    assert len(rows) == 1
    tag = iso.mol.GetBondBetweenAtoms(*pair).GetStereo()
    cis = tag in {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}
    assert rows[0][4:] == (0.0 if cis else 180.0, COPLANAR_CAP)


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
    from rxembed import bounds

    tols, real = [], bounds._bounds

    def spy(mol, cons, params=None):
        bm, tol = real(mol, cons, params)
        tols.append(tol)
        return bm, tol

    monkeypatch.setattr(bounds, "_bounds", spy)
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


@pytest.mark.parametrize("face", ["centred-arene", "slipped-arene", "open-allyl"])
def test_haptic_centroid_target_matches_measured_geometry(face):
    if face == "open-allyl":
        mol = Chem.MolFromSmiles("C=CC")
        positions = np.array([[-1.2, 0.0, 0.0], [0.0, 0.7, 0.0], [1.2, 0.0, 0.0]])
    else:
        mol = Chem.MolFromSmiles("c1ccccc1")
        angles = np.arange(6) * np.pi / 3
        positions = 1.4 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    metal = np.array([0.0 if face == "centred-arene" else 1.2, 0.0, 1.5])
    conf = Chem.Conformer(len(positions))
    conf.SetPositions(positions)
    mol.AddConformer(conf)
    radius = metal_constraints._site_radius(mol, tuple(range(len(positions))), positions=positions)
    lengths = np.linalg.norm(positions - metal, axis=1)
    actual = np.linalg.norm(metal - positions.mean(axis=0))
    assert metal_constraints._site_height(radius, lengths) == pytest.approx(actual)


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
        CoordinationSphere(mol, 2, 25, (1, -1), {}, POLYHEDRA["linear"], pos=mol.GetConformer().GetPositions())
    )
    alternate = coordination(CoordinationSphere(mol, 2, 25, (1, -1), {}, POLYHEDRA["linear"]))
    assert measured.floors == alternate.floors


def test_length_source_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        enumerate_isomers(_fake_geometry(_SQUARE_PD), "square_planar", lengths="input")
    said = [r for r in caplog.records if "M-donor windows from" in r.getMessage()]
    assert said
    assert "input conformer" in said[0].getMessage()
    assert "lengths='input'" in said[0].getMessage()
