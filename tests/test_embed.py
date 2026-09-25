"""Test the core embed and Conformers API."""

from __future__ import annotations

import importlib
import itertools
import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, SetBondLength

import rxembed as rx
from rxembed import bounds as bnd
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL, Constraints, resolve_core
from rxembed.embed import BASE_STIFFNESS, Conformers, Failure, embed, fold_substrate, minimize
from rxembed.metal_core import COORDINATION_METALS, VACANT, donor_chirality_sign, metal_indices, state_with_winding
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer, from_geometry
from rxembed.metal_perceive import SHAPE_PROP, classify_geometry, coplanar
from rxembed.metal_polyhedron import POLYHEDRA
from rxembed.metal_smiles import cxsmiles, parse_smiles
from rxembed.relax import UFFOptimizationError, UFFRecord, UFFTypingError, bonding_failure, restrained_uff
from rxembed.stereo import axis_stereo, bond_stereo, point_stereo, stereo_from_3d
from tests.metal_fixtures import ONE_ARM_BOUND_PT

emb = importlib.import_module("rxembed.embed")  # the engine implementation module, not the public facade

_BIPY_PD = "Cl[Pd]1(Cl)<-n2ccccc2-c2ccccn->12"
_EN_PD = "Cl[Pd](Cl)(<-N(C)(C)C)<-N(C)(C)C"
# the N-bound Ni(II) isomer: the window relax tears 3 of 8 seeds at the base stiffness, the ladder's type case
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# cis-[Co(en)2Cl2], the textbook Delta/Lambda pair, and the same complex with one sp3 centre on a backbone:
# the second is the case a reflection cannot repair, because it would invert that centre too
_CO_EN = "Cl[Co]12(Cl)(NCCN1)NCCN2"
_CO_EN_ME = "Cl[Co]12(Cl)(N[C@@H](C)CN1)NCCN2"


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _with_geometry(smiles, seed=7):
    mol = _mol(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _isomer(smiles=_BIPY_PD, geometry="square_planar"):
    """The first enumerated coordination isomer of `smiles`: the metal spec `embed` accepts."""
    return next(iter(enumerate_isomers(_mol(smiles), geometry)))


def _fold(iso, **spec):
    """Fold a user `fix`/`constrain` onto the isomer's polyhedron, exactly as `embed(iso, **spec)` does."""
    sub, graft_ref = resolve_core(iso.mol, **spec, has_geometry=iso.mol.GetNumConformers() > 0)
    return fold_substrate(iso.cons.copy(), sub, graft_ref)


def _sphere_key(iso, which=0):
    """The (i, j) distance key of one M-donor hold, in `add_distance`'s sorted order."""
    return (min(iso.metal, iso.donors[which]), max(iso.metal, iso.donors[which]))


def _distance(mol, cid, i, j):
    p = mol.GetConformer(cid).GetPositions()
    return float(np.linalg.norm(p[i] - p[j]))


def _aligned_rms(a, b):
    """Return same-order Cartesian RMSD after a proper Kabsch alignment."""
    a, b = a - a.mean(0), b - b.mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    u[:, -1] *= np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((a @ (u @ vt) - b) ** 2, axis=1))))


# ---------------------------------------------------------------------------------------------------------
# the everyday spine
# ---------------------------------------------------------------------------------------------------------


def test_embed_fix_chains_minimize_and_dump(tmp_path):
    mol = _mol("OCCCN")
    confs = embed(mol, fix={(0, 4): 3.0}, n=8, seed=0xF00D)
    assert len(confs) == len(confs.ids) > 0
    confs.minimize()
    for cid in confs.ids:
        assert _distance(confs._mol, cid, 0, 4) == pytest.approx(3.0, abs=0.1)
    path = confs.dump(tmp_path / "out.xyz")
    assert path.read_text().splitlines().count(str(mol.GetNumAtoms())) == len(confs)


def test_numeric_fix_delivers_a_linear_three_centre_core():
    mol = _mol("[F-].CCl")  # F(0), C(1), Cl(2)
    fix = {(0, 1): 2.0, (1, 2): 2.2, (0, 1, 2): 178.0}
    # Two disconnected fragments tied only by the fix: DG's own eigen-embedding step needs a random start to
    # place them into the constrained arrangement at all, so ask for one explicitly (no default retry).
    native = rdDistGeom.KDG()
    native.useLegacyImplementation = False
    native.useRandomCoords = True
    confs = embed(mol, fix=fix, n=1, params=bnd.EmbedParams(seed=0xF00D, prune_rms=-1, native=native)).minimize()
    assert confs
    for cid in confs.ids:
        conf = confs.mol.GetConformer(int(cid))
        assert GetBondLength(conf, 0, 1) == pytest.approx(2.0, abs=FIX_DISTANCE_TOL)
        assert GetBondLength(conf, 1, 2) == pytest.approx(2.2, abs=FIX_DISTANCE_TOL)
        assert GetAngleDeg(conf, 0, 1, 2) == pytest.approx(178.0, abs=FIX_ANGLE_TOL)


def test_tagged_metal_donor_embedding_does_not_print_uff_typer_noise(capfd):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]

    assert embed(iso, n=1, seed=2).ids
    assert "UFFTYPER" not in capfd.readouterr().err


def test_metal_publication_keeps_the_ligand_bond_integrity_contract():
    confs = embed(_isomer("Cl[Pd](Cl)(NCCCl)N"), n=1, seed=42, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    chlorine = next(
        atom
        for atom in confs._mol.GetAtoms()
        if atom.GetAtomicNum() == 17 and any(n.GetAtomicNum() == 6 for n in atom.GetNeighbors())
    )
    pair = tuple(sorted((chlorine.GetIdx(), chlorine.GetNeighbors()[0].GetIdx())))
    SetBondLength(confs._mol.GetConformer(cid), pair[0], pair[1], 2.4)

    failure = confs._geometry_failure(cid)
    assert failure is not None
    assert failure.kind == "bonding"

    # An explicit stretched-bond request owns only that pair, including in a metal-containing TS.
    confs.cons.distances[pair] = (2.35, 2.45)
    assert confs._geometry_failure(cid) is None


def test_numeric_fix_rejects_when_cleanup_cannot_hold_it(monkeypatch, caplog):
    fix = {(1, 2): 2.026, (2, 0): 1.557}
    confs = embed(_mol("[O-].ClCCCCBr"), fix=fix, n=1, params=bnd.EmbedParams(seed=1, prune_rms=-1))
    assert confs._fixed_geometry_misses(confs.ids[0]), "the raw seed must miss for this test to exercise rejection"

    monkeypatch.setattr(
        emb,
        "restrained_uff",
        lambda mol, cons, conf_ids=None, **kw: np.zeros(len(conf_ids if conf_ids is not None else mol.GetConformers())),
    )
    seed_calls = []
    real_seed = emb.seed_conformers

    def tracked_seed(*args, **kwargs):
        params = kwargs.get("params", args[4] if len(args) > 4 else None)
        seed_calls.append(params.seed)
        return real_seed(*args, **kwargs)

    monkeypatch.setattr(emb, "seed_conformers", tracked_seed)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        with pytest.raises(ValueError, match="fix="):
            confs.minimize()

    assert seed_calls, "the hard fix failure bypassed fresh-seed replacement"
    assert not confs.ids, "an off-target numeric fix was returned after cleanup"
    assert not confs.unrelaxed, "rejected conformer ids leaked into tracked state"
    assert "requested" in caplog.text
    assert "+/- 0.001" in caplog.text
    assert "got" in caplog.text


def test_undefined_dihedral_fails_numeric_fix_gate():
    mol = _with_geometry("CCCC")
    conf = mol.GetConformer()
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0))):
        conf.SetAtomPosition(atom, xyz)
    cons = resolve_core(mol, fix={(0, 1, 2, 3): 60.0}, has_geometry=True)[0]
    misses = Conformers(mol, [0], cons)._fixed_geometry_misses(0)
    assert len(misses) == 1
    assert np.isinf(misses[0][0])


def test_coordination_gate_rejects_a_poor_nearest_shape():
    iso = _isomer()
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    metal = np.array(conf.GetAtomPosition(iso.metal))
    for donor in iso.donors:
        conf.SetAtomPosition(donor, (metal + np.array([1.0, 0.0, 0.0])).tolist())

    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_coordination_gate_rejects_another_named_shape_below_the_residual_floor():
    iso = _isomer("[Pt](F)(Cl)(Br)I", "tetrahedral")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA["seesaw"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))

    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_coordination_gate_rejects_a_nearer_cn7_shape_below_the_residual_floor():
    iso = _isomer("[Re](F)(F)(F)(F)(F)(F)F", "pentagonal_bipyramidal")
    conf = Chem.Conformer(iso.mol.GetNumAtoms())
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA["capped_trigonal_prismatic"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))
    iso.mol.AddConformer(conf)

    assert classify_geometry(iso.mol, iso.metal, list(iso.vertices), 0) == "capped_trigonal_prismatic"
    assert emb._coordination_state_failure(iso.mol, 0, iso) is not None


def test_coordination_gate_accepts_an_ideal_mixed_high_coordination_state():
    iso = _isomer("[La](F)(F)(F)(F)(F)(F)(F)Cl", "square_antiprism")
    conf = Chem.Conformer(iso.mol.GetNumAtoms())
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA[iso.geometry].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))
    iso.mol.AddConformer(conf)

    assert emb._coordination_state_failure(iso.mol, 0, iso) is None


def test_coordination_gate_rejects_a_better_exact_search_slot_assignment():
    iso = _isomer("[Fe](N)(O)(F)(Cl)Br", "square_pyramidal")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    ideal = np.asarray(POLYHEDRA["square_pyramidal"].vertex_dirs, dtype=float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    alternate = ideal[[1, 2, 3, 0, 4]]
    observed = 0.49 * ideal + 0.51 * alternate
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    for donor, direction in zip(iso.vertices, observed, strict=True):
        conf.SetAtomPosition(donor, tuple(2.0 * direction))

    assert classify_geometry(confs._mol, iso.metal, list(iso.vertices), confs.ids[0]) == "square_pyramidal"
    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_cobalt_tetraammine_vacancy_rejects_a_trigonal_bipyramidal_shell():
    """A CN6 octahedral request with one vacancy (Co(NH3)4Cl, 5 donors) is read against its 5 occupied sites.

    An ideal trigonal_bipyramidal shell, not the octahedral-minus-one square pyramid its own vacant subset
    would read as, is a clear miss, named by both shape codes.
    """
    iso = _isomer("[Co](N)(N)(N)(N)Cl", "octahedral")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    donors = [v for v in iso.vertices if v != VACANT]
    for donor, direction in zip(donors, POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))

    failure = emb._coordination_state_failure(confs._mol, confs.ids[0], iso)
    assert failure is not None
    assert re.fullmatch(rf"Co{iso.metal} relaxes from OCT \(\d\.\d{{3}}\) to TBP \(\d\.\d{{3}}\)", str(failure))


def test_fe_pentacarbonyl_square_pyramid_shell_is_rejected_with_numbers():
    """Fe(CO)5 requesting trigonal_bipyramidal, given an ideal square_pyramidal shell: residual 0.233, well
    under `FIT_FLOOR` (0.45) but not a near-tie, so an identity-only gate would accept it. The rejection
    must carry both residual numbers, not just a name.
    """
    iso = _isomer("[Fe](N)(O)(F)(Cl)Br", "trigonal_bipyramidal")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA["square_pyramidal"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))

    failure = emb._coordination_state_failure(confs._mol, confs.ids[0], iso)
    assert failure is not None
    assert len(re.findall(r"\d\.\d{3}", str(failure))) >= 2


def test_fe_pentacarbonyl_conformer_carries_its_shape_record():
    """An accepted `rx.embed` conformer carries one `shape` property naming its centre and polyhedron."""
    iso = _isomer("[Fe](N)(O)(F)(Cl)Br", "trigonal_bipyramidal")
    confs = embed(iso, n=1, seed=1).minimize()

    prop = confs._mol.GetConformer(confs.ids[0]).GetProp(SHAPE_PROP)
    assert prop.startswith("Fe0 TBP")


def test_publication_rejects_a_different_coordination_arrangement():
    isomers = enumerate_isomers(_mol("[Pt](F)(Cl)(Br)I"), "square_planar", screen=False)
    selected = embed(isomers[0], n=1, seed=1, threads=1)
    alternate = embed(isomers[1], n=1, seed=1, threads=1)

    assert emb._coordination_state_failure(alternate.mol, 0, isomers[0]) is not None
    assert emb._coordination_state_failure(selected.mol, 0, isomers[0]) is None


def test_octahedral_hand_does_not_replace_full_donor_slot_identity():
    isomers = enumerate_isomers(_mol("[Co](N)(O)(F)(Cl)(Br)P"), "octahedral", screen=False)
    assert len(isomers) == 30  # six different donors: 6! / 24 proper octahedral rotations
    assert len({cxsmiles(iso) for iso in isomers}) == len(isomers)
    requested = isomers[0]
    assert sum(iso.chirality == requested.chirality for iso in isomers) == 15
    directions = np.asarray(POLYHEDRA["octahedral"].vertex_dirs, float)
    for index, iso in enumerate(isomers):
        mol = Chem.Mol(iso.mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom, direction in zip(iso.vertices, directions, strict=True):
            conf.SetAtomPosition(atom, 2 * direction)
        mol.AddConformer(conf)
        assert (emb._coordination_state_failure(mol, 0, requested) is None) == (index == 0), (
            f"accepted the wrong seating or rejected the requested one: {iso}"
        )


def test_coordination_gate_rejects_nonfinite_donor_coordinates():
    iso = _isomer()
    confs = embed(iso, n=1, seed=1)
    confs._mol.GetConformer(confs.ids[0]).SetAtomPosition(iso.donors[0], (float("nan"), 0.0, 0.0))

    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_structural_gate_rejects_undefined_coplanarity():
    mol = _with_geometry("CCCC")
    conf = mol.GetConformer()
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))):
        conf.SetAtomPosition(atom, xyz)
    cons = Constraints(coplanar=[(0, 1, 2, 3, 180.0, 10.0)])

    assert emb._structural_failure(mol, 0, cons) == Failure(
        "structural_constraint", "coplanarity C0-C1-C2-C3 is undefined", atoms=(0, 1, 2, 3)
    )


@pytest.mark.parametrize(
    ("cons", "kind", "detail", "atoms"),
    [
        (
            Constraints(metals={0}, distances={(0, 2): (2.0, 2.1)}),
            "ml_distance",
            "M-L distance He0-He2 1.41 A, 0.58 A outside 1.99-2.11",
            (0, 2),
        ),
        (
            Constraints(coplanar=[(0, 1, 2, 3, None, 10.0)]),
            "structural_constraint",
            "coplanarity He0-He1-He2-He3 45.0 deg, 33 deg outside 0.0-12.0",
            (0, 1, 2, 3),
        ),
        (
            Constraints(umbrellas={(0, 1, 2, 3): 60.0}),
            "structural_constraint",
            "umbrella He0-He1-He2-He3 45.0 deg, 13 deg outside 58.0-92.0",
            (0, 1, 2, 3),
        ),
    ],
)
def test_geometry_failure_names_the_structural_measurement(cons, kind, detail, atoms):
    mol = Chem.MolFromSmiles("[He].[He].[He].[He]")
    conf = Chem.Conformer(4)
    conf.SetPositions(np.array(((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)), dtype=float))
    mol.AddConformer(conf)
    conformers = Conformers(mol, [0], cons)

    assert conformers._geometry_failure(0) == Failure(kind, detail, atoms=atoms)


def test_structural_failure_reports_excess_below_display_precision():
    mol = Chem.MolFromSmiles("[He].[He]")
    conf = Chem.Conformer(2)
    conf.SetPositions(np.array(((0, 0, 0), (2.1 + emb._ML_WINDOW_TOL + 0.0004, 0, 0))))
    mol.AddConformer(conf)
    cons = Constraints(metals={0}, distances={(0, 1): (2.0, 2.1)})

    failure = emb._structural_failure(mol, 0, cons)
    assert failure.kind == "ml_distance"
    assert "0.0004 A outside" in failure.detail


def test_model_ml_window_accepts_physical_slack_a_fixed_distance_does_not():
    """A model-derived M-L window tolerates 0.01 A; a user-fixed (fix=/template=) distance still needs 0.001 A.

    `cons.fixed` is the existing record of a user-stated exact distance; anything else reaching the M-L
    check is model-derived (constrain= is already excluded via `contacts[0]`). Same geometry, same window,
    different provenance: only the fixed one should reject a 0.006 A overshoot.
    """
    mol = Chem.MolFromSmiles("[He].[He]")
    conf = Chem.Conformer(2)
    conf.SetPositions(np.array(((0, 0, 0), (2.106, 0, 0))))  # 0.006 A past hi=2.1
    mol.AddConformer(conf)

    model = Constraints(metals={0}, distances={(0, 1): (2.0, 2.1)})
    assert emb._structural_failure(mol, 0, model) is None

    fixed = Constraints(metals={0}, distances={(0, 1): (2.0, 2.1)}, fixed={(0, 1): (2.0, 2.1)})
    failure = emb._structural_failure(mol, 0, fixed)
    assert failure == Failure(
        "ml_distance",
        f"M-L distance He0-He1 2.11 A, {0.006 - FIX_DISTANCE_TOL:.2g} A outside "
        f"{2.0 - FIX_DISTANCE_TOL:.2f}-{2.1 + FIX_DISTANCE_TOL:.2f}",
        atoms=(0, 1),
    )


def test_donor_orientation_seed_wall_is_not_a_structural_postcondition():
    mol = Chem.MolFromSmiles("[Li].N=C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (1.173648, 0.984808, 0))):
        conf.SetAtomPosition(atom, xyz)
    mol.AddConformer(conf)
    cons = Constraints(angles={(0, 1, 2): (108.0, 180.0)}, metals={0})

    assert emb._structural_failure(mol, 0, cons) is None


def test_donor_angle_failure_logs_without_rejecting_the_conformer(monkeypatch, caplog):
    confs = embed(_isomer(), n=1, seed=1, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    detail = "C1 (sp3) M-D-X to Si2: 99.5° < census floor 104°; inspect donor geometry and restraints"
    monkeypatch.setattr(emb, "donor_orientation", lambda *args: [SimpleNamespace(detail=detail)])

    with caplog.at_level(logging.DEBUG, logger="rxembed"):
        failure = confs._geometry_failure(cid)
    assert failure is None, "a donor-orientation floor violation must not reject the conformer"
    assert [record.levelno for record in caplog.records if detail in record.getMessage()] == [logging.DEBUG]


@pytest.mark.parametrize(("real_window", "expected"), [((0.9, 2.2), True), ((1.9, 2.2), False)])
def test_structural_gate_ignores_transient_phantom_but_checks_real_face_distances(real_window, expected):
    mol = Chem.MolFromSmiles("[Li].C.C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 1), (-1, 0, 1))):
        conf.SetAtomPosition(atom, xyz)
    mol.AddConformer(conf)
    cons = Constraints(
        distances={(0, 1): real_window, (0, 2): real_window, (0, 3): (1.9, 2.1)},
        metals={0},
        phantoms=frozenset({3}),
        haptic={3: (1, 2)},
    )

    assert (emb._structural_failure(mol, 0, cons) is None) is expected


# ---------------------------------------------------------------------------------------------------------
# Conformers: the one result type
# ---------------------------------------------------------------------------------------------------------


def test_indexing_reuses_mol_in_new_conformers():
    confs = embed(_mol("CCOCC"), n=6, params=bnd.EmbedParams(seed=7, prune_rms=-1))
    sub = confs[:2]
    assert sub.ids == confs.ids[:2]
    assert sub._mol is confs._mol
    assert confs[0].ids == [confs.ids[0]]
    assert {c.GetId() for c in sub.mol.GetConformers()} == set(sub.ids)
    with pytest.raises(ValueError, match="not one of this result's ids"):
        sub.xyz(confs.ids[-1])  # an id the slice no longer tracks


def test_dump_refuses_a_result_with_no_conformers(tmp_path):
    with pytest.raises(ValueError, match="nothing to dump"):
        Conformers(_mol("CCO"), []).dump(tmp_path / "empty.xyz")


@pytest.mark.parametrize("trajectory", [None, Chem.Mol()])
def test_dump_trajectory_refuses_without_capture_before_opening(tmp_path, trajectory):
    destination = tmp_path / "existing.xyz"
    destination.write_text("keep")
    confs = Conformers(_mol("CCO"), [0], trajectory=trajectory)

    with pytest.raises(ValueError, match="trajectory=True"):
        confs.dump_trajectory(destination)
    assert destination.read_text() == "keep"


def test_surrogate_is_internal_and_output_restores_metal():
    src = _mol(_EN_PD)
    iso = Isomer(src, "SPL", {0: 0, 1: 2, 2: 3, 3: 7})
    confs = embed(iso, n=2, seed=0xF00D).minimize()

    internal = confs._mol.GetAtomWithIdx(iso.metal)
    assert internal.GetAtomicNum() not in COORDINATION_METALS, "the working mol must still hold the surrogate"
    assert internal.GetDegree() == 0, "the surrogate must stay bond-less: a bonded Li makes UFF singular"

    restored = confs.mol.GetAtomWithIdx(iso.metal)
    assert restored.GetAtomicNum() in COORDINATION_METALS
    assert restored.GetFormalCharge() == iso.real_q
    assert Chem.GetFormalCharge(confs.mol) == Chem.GetFormalCharge(src)
    assert len(Chem.GetMolFrags(confs.mol)) == 1, "the M-donor bonds are re-added, so the complex is one fragment"


# ---------------------------------------------------------------------------------------------------------
# refusals: every one is a wrong answer the guard turned into an error
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"seed": -1}, "not reproducible"), ({"n": 0}, "positive conformer")],
    ids=["negative-seed", "zero-conformers"],
)
def test_embed_rejects_silent_misuse(kwargs, match):
    with pytest.raises(ValueError, match=match):
        embed(_mol("CCO"), **kwargs)


def test_embed_refuses_implicit_hydrogens():
    with pytest.raises(ValueError, match="AddHs"):
        embed(Chem.MolFromSmiles("CCO"), n=2)


def test_embed_refuses_a_bare_metal_mol():
    with pytest.raises(ValueError, match="Isomer"):
        embed(_mol(_EN_PD), n=2)


def test_embed_refuses_a_source_it_would_have_to_parse():
    with pytest.raises(TypeError, match="RDKit Mol or an Isomer"):
        embed("CCO", n=2)


def test_zero_conformer_embed_warns(caplog):
    mol = _mol("C1C2CC3CC1CC(C2)C3")  # adamantane: a 1.0 A fix across the cage is not embeddable
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs = embed(mol, fix={(0, 3): 1.0}, n=1, seed=0xF00D)
    assert len(confs) == 0
    assert "no conformer" in caplog.text


# ---------------------------------------------------------------------------------------------------------
# fold_substrate: a user spec must reach the embedder whole, and must not be demoted to soft
# ---------------------------------------------------------------------------------------------------------


def test_pi_stack_constrain_survives_the_fold():
    iso = _isomer()
    a, b = (tuple(r) for r in iso.mol.GetRingInfo().AtomRings() if len(r) == 6)
    cons = _fold(iso, constrain={(a, b): 3.6})
    assert any(set(pa) == set(a) and set(pb) == set(b) for pa, pb, _sep in cons.planes)


def test_fix_landing_on_a_sphere_hold_overrides_it():
    iso = _isomer()
    key = _sphere_key(iso)
    alone, _ref = resolve_core(iso.mol, fix={key: 2.42}, has_geometry=False)
    assert alone.distances[key] != iso.cons.distances[key], "premise: the two windows must differ"
    folded = _fold(iso, fix={key: 2.42})
    assert folded.distances[key] == alone.distances[key], "the sphere hold clipped the fix"
    assert key not in folded.pulls, "the sphere's approximate pull still competes with the numeric fix"


def test_sphere_hold_remains_nonreleasable():
    iso = _isomer()
    key = _sphere_key(iso)
    with pytest.raises(ValueError, match="soft bias cannot replace the selected metal state"):
        _fold(iso, constrain={key: (2.3, 2.5)})


def test_soft_dihedral_cannot_replace_structural_metal_torsion():
    base = Constraints(coplanar=[(0, 1, 2, 3, 180.0, 15.0)], umbrellas={(6, 1, 2, 7): None})
    key = (4, 1, 2, 5)
    sub = Constraints(
        dihedrals={key: (80.0, 100.0)},
        contacts=(frozenset(), frozenset({key})),
    )

    with pytest.raises(ValueError, match="soft bias cannot replace the selected metal state"):
        fold_substrate(base, sub, {})


@pytest.mark.parametrize("ideal", [None, 30.0, 0.0])
def test_substrate_rotation_preserves_umbrella_support_ownership(ideal):
    base = Constraints(umbrellas={(0, 1, 2, 3): ideal})
    key = (4, 1, 2, 5)
    sub = Constraints(dihedrals={key: (40.0, 60.0)}, contacts=(frozenset(), frozenset({key})))
    folded = fold_substrate(base, sub, {})
    assert folded.dihedrals == sub.dihedrals
    assert folded.umbrellas == base.umbrellas
    same = (1, 0, 3, 2)
    sub = Constraints(dihedrals={same: (40.0, 60.0)}, contacts=(frozenset(), frozenset({same})))
    if ideal == 0.0:
        assert fold_substrate(base, sub, {}).dihedrals == sub.dihedrals
    else:
        with pytest.raises(ValueError, match="soft bias cannot replace"):
            fold_substrate(base, sub, {})


def test_substrate_cannot_bypass_a_proper_torsion_using_different_endpoints():
    base = Constraints(dihedrals={(0, 1, 2, 3): (40.0, 60.0)})
    key = (4, 1, 2, 5)
    sub = Constraints(dihedrals={key: (80.0, 100.0)}, contacts=(frozenset(), frozenset({key})))
    with pytest.raises(ValueError, match="soft bias cannot replace"):
        fold_substrate(base, sub, {})


def test_pyramidal_gate_is_not_disabled_by_an_independent_torsion():
    mol = Chem.MolFromSmiles("C.C.C.C.C.C")
    conf = Chem.Conformer(6)
    conf.SetPositions(np.array([(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)], float))
    mol.AddConformer(conf)
    base = Constraints(umbrellas={(0, 1, 2, 3): 30.0})
    for key, accepted in (((4, 1, 2, 5), False), ((1, 0, 3, 2), True)):
        cons = base.copy(dihedrals={key: (40.0, 60.0)})
        assert (emb._structural_failure(mol, mol.GetConformer().GetId(), cons) is None) == accepted


def test_graft_allows_one_sphere_atom_not_two():
    iso = _isomer(_EN_PD)
    fix = {iso.donors[0]: (0.0, 0.0, 0.0), iso.donors[1]: (2.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="pins coordination-sphere atoms"):
        _fold(iso, fix=fix)
    assert iso.donors[0] in _fold(iso, fix={iso.donors[0]: (0.0, 0.0, 0.0)}).frozen


def test_ammonium_acetate_stays_within_contact_range():
    """A simple salt's two free ions embed together, not scattered apart.

    Every fragment repels every other one at its van der Waals floor (`bnd._cap_fragment_contacts`); nothing
    picks a contact atom, so the ions can settle anywhere within that shared range, never beyond it.
    """
    mol = _mol("CC(=O)[O-].[NH4+]")
    confs = embed(mol, n=6, seed=42)
    assert confs.ids, "the salt must actually embed for this test to say anything"
    pt = Chem.GetPeriodicTable()
    frags = Chem.GetMolFrags(confs.mol)
    assert len(frags) == 2, "acetate and ammonium stay their own fragments; nothing bonds them"
    for cid in confs.ids:
        pos = confs.mol.GetConformer(cid).GetPositions()
        for i in [a for a in frags[0] if confs.mol.GetAtomWithIdx(a).GetAtomicNum() > 1]:
            for j in [a for a in frags[1] if confs.mol.GetAtomWithIdx(a).GetAtomicNum() > 1]:
                d = float(np.linalg.norm(pos[i] - pos[j]))
                vdw = pt.GetRvdw(confs.mol.GetAtomWithIdx(i).GetAtomicNum()) + pt.GetRvdw(
                    confs.mol.GetAtomWithIdx(j).GetAtomicNum()
                )
                assert d >= vdw - 0.05, f"{i}-{j} sits inside its van der Waals floor"
                assert d < 15.0, f"{i}-{j} drifted out of contact range ({d:.1f} A)"


# ---------------------------------------------------------------------------------------------------------
# graft_frozen; distance geometry only approximates a rigid core; the graft restores it exactly
# ---------------------------------------------------------------------------------------------------------


def test_graft_restores_the_core_shape_exactly():
    mol = _with_geometry("CCCl")
    core = [0, 1, 2]
    ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.8, 0.0]])
    emb.graft_frozen(mol, [0], core, ref)

    pos = mol.GetConformer(0).GetPositions()
    for (a, b), want in (((0, 1), 1.5), ((1, 2), 1.8), ((0, 2), float(np.linalg.norm(ref[0] - ref[2])))):
        assert np.linalg.norm(pos[core[a]] - pos[core[b]]) == pytest.approx(want, abs=1e-9)


def test_core_too_small_to_orient_slides_or_stands_still():
    mol = _with_geometry("CCCl")
    before = mol.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(mol, [0], [1, 2], np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]))

    pos = mol.GetConformer(0).GetPositions()
    assert np.allclose(pos[1], before[1]), "the anchor atom moved"
    assert np.linalg.norm(pos[2] - pos[1]) == pytest.approx(2.4, abs=1e-9)
    axis_before, axis_after = before[2] - before[1], pos[2] - pos[1]
    cos = axis_after @ axis_before / (np.linalg.norm(axis_after) * np.linalg.norm(axis_before))
    assert cos == pytest.approx(1.0, abs=1e-9), "the embedded axis was rotated, not just rescaled"

    one = _with_geometry("CCCl")
    before = one.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(one, [0], [1], np.array([[9.0, 9.0, 9.0]]))
    assert np.allclose(one.GetConformer(0).GetPositions(), before)


# ---------------------------------------------------------------------------------------------------------
# minimize: the search-free companion verb
# ---------------------------------------------------------------------------------------------------------


def test_minimize_pulls_without_search():
    mol = _with_geometry("CCCl")
    before = GetBondLength(mol.GetConformer(0), 1, 2)
    confs = minimize(mol, fix={(1, 2): 2.4})

    assert len(confs) == mol.GetNumConformers(), "minimize must not add or drop conformers: it does not search"
    after = GetBondLength(confs.mol.GetConformer(confs.ids[0]), 1, 2)
    assert abs(after - 2.4) < 0.1, f"C-Cl was not pulled to the target: {before:.2f} -> {after:.2f}"
    assert GetBondLength(mol.GetConformer(0), 1, 2) == pytest.approx(before), "the caller's geometry was relaxed"


def test_minimize_refuses_a_graph_with_no_geometry():
    with pytest.raises(ValueError, match="existing geometry"):
        minimize(_mol("CCO"))


def test_minimize_holds_isomer_polyhedron():
    src = _with_geometry(_EN_PD)
    iso = Isomer(src, "SPL", {0: 0, 1: 2, 2: 3, 3: 7})
    before = iso.mol.GetConformer().GetPositions().copy()
    assert not coplanar(before, iso.metal, iso.donors), "the premise: this input does NOT realise the square plane"
    confs = minimize(iso)

    assert confs.iso is iso, "the isomer must be carried, or nothing downstream can restore the metal"
    assert set(confs.cons.distances) >= set(iso.cons.distances), "the M-donor windows never reached the relax"
    assert set(confs.cons.angles) == set(iso.cons.angles), "the polyhedron angles never reached the relax"

    pos = confs._mol.GetConformer(confs.ids[0]).GetPositions()
    for (i, j), (lo, hi) in iso.cons.distances.items():  # the isomer's OWN windows, not a tabulated length
        d = float(np.linalg.norm(pos[i] - pos[j]))
        assert lo - 0.15 <= d <= hi + 0.15, f"sphere not held: d({i},{j}) = {d:.2f}, window ({lo:.2f}, {hi:.2f})"
    assert coplanar(pos, iso.metal, iso.donors), "the relax was pulled toward the declared square plane"

    assert confs.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() in COORDINATION_METALS
    assert len(Chem.GetMolFrags(confs.mol)) == 1, "the M-donor bonds are re-added, so the complex is one fragment"
    assert confs._mol is not iso.mol, "the relax works on a copy: the caller keeps its isomer"
    assert np.allclose(iso.mol.GetConformer().GetPositions(), before)


@pytest.mark.parametrize("lengths", ["model", "input"])
def test_minimize_uses_model_distances_unless_input_requested(lengths):
    mol = _with_geometry("Cl[Pd](Cl)(N)N")
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(pd).GetNeighbors()]
    before = [GetBondLength(mol.GetConformer(0), pd, d) for d in donors]

    target = from_geometry(mol, lengths=lengths)
    confs = minimize(mol if lengths == "model" else target)
    conf = confs.mol.GetConformer(confs.ids[0])
    after = [GetBondLength(conf, pd, d) for d in donors]
    if lengths == "input":
        assert np.allclose(before, after, atol=0.15)
    else:
        assert not np.allclose(before, after, atol=0.15)
    for donor, distance in zip(donors, after, strict=True):
        lo, hi = target.cons.distances[tuple(sorted((pd, donor)))]
        assert lo - 0.15 <= distance <= hi + 0.15


# ---------------------------------------------------------------------------------------------------------
# the stiffness ladder: the front door may not hand back a torn geometry wearing a plausible energy
# ---------------------------------------------------------------------------------------------------------


def test_high_force_candidate_and_stable_coordination_use_kind_not_message_wording():
    reworded_ml = Failure("ml_distance", "a completely reworded M-L overshoot message")
    assert emb._high_force_candidate(reworded_ml)
    assert not emb._high_force_candidate(Failure("structural_constraint", "M-L distance (0, 1): 2.2 A"))
    assert not emb._high_force_candidate("M-L distance (0, 1): not a Failure")

    reworded_shape = Failure("coordination_shape", "a completely reworded shape mismatch message")
    assert emb._stable_coordination_failure(reworded_shape)
    assert not emb._stable_coordination_failure(Failure("metal_state", "coordination state at M0 is nonplanar"))
    assert not emb._stable_coordination_failure("coordination state at M0: not a Failure")


def test_stiffness_ladder_retries_only_the_unresolved_conformer(monkeypatch):
    confs = embed(_isomer(_NI_N), n=2, params=bnd.EmbedParams(seed=1, threads=1, prune_rms=-1))
    victim = int(confs.ids[0])
    calls = []
    constraints = []
    latest = {}

    def marked_uff(_mol, _cons, *, stiffness, max_iters, conf_ids, record=None, **_kw):
        ids = tuple(conf_ids)
        calls.append((ids, stiffness, max_iters))
        constraints.append(_cons)
        latest.update(dict.fromkeys(ids, stiffness))
        if record is not None:
            record.statuses.update(dict.fromkeys(ids, 0))
        return np.zeros(len(ids))

    def injected_failure(_self, cid, _iso=None):
        return "heavy-atom bonding/clash failure" if cid == victim and latest[cid] == BASE_STIFFNESS else None

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(Conformers, "_geometry_failure", injected_failure)
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [
        (tuple(confs.ids), BASE_STIFFNESS, 10),
        ((victim,), 3 * BASE_STIFFNESS, 10),
    ]
    assert all(active is confs.cons for active in constraints)
    assert victim not in confs.unrelaxed


def test_replacement_batches_reuse_one_coordination_prep_per_isomer(monkeypatch):
    """One isomer's coordination-state prep must be built once, then shared across every ladder rung.

    `_relax_constrained` copies `self` with `dataclasses.replace()` at every rung, and each rung checks the
    shape for every conformer; a per-copy cache would rebuild the same isomer's prep on each of those checks.
    """
    real_prep = emb._coordination_state_prep
    calls = []

    def spy(iso):
        calls.append(True)
        return real_prep(iso)

    monkeypatch.setattr(emb, "_coordination_state_prep", spy)

    confs = embed(_isomer(_NI_N), n=2, seed=1, threads=1).minimize()

    assert confs.ids
    assert len(calls) == 1


def test_relax_ladder_does_not_escalate_a_converged_wrong_coordination_shape(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(), params=bnd.EmbedParams(seed=1))
    calls = []

    def marked_uff(_mol, _cons, *, max_iters, conf_ids, record=None, **_kw):
        calls.append(max_iters)
        if record is not None:
            record.statuses.update(dict.fromkeys(conf_ids, 0))
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda *_args: Failure("coordination_shape", "coordination state at M0: expected SPY, found TBP"),
    )

    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [10]
    assert confs.unrelaxed == [0]
    assert confs.relax_failures[0].kind == "coordination_shape"


def test_relax_ladder_does_not_escalate_an_unconverged_wrong_coordination_shape(monkeypatch):
    """A wrong shape is a wrong basin whether or not UFF's optimizer reports convergence at it."""
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(), params=bnd.EmbedParams(seed=1))
    calls = []

    def marked_uff(_mol, _cons, *, max_iters, conf_ids, record=None, **_kw):
        calls.append(max_iters)
        if record is not None:
            record.statuses.update(dict.fromkeys(conf_ids, 1))  # not converged
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda *_args: Failure("coordination_shape", "coordination state at M0: expected SPY, found TBP"),
    )

    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [10]
    assert confs.unrelaxed == [0]
    assert confs.relax_failures[0].kind == "coordination_shape"


@pytest.mark.parametrize(
    ("requested", "recover", "planar"),
    [(True, True, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_relax_ladder_checks_requested_haptic_face_before_accepting(monkeypatch, requested, recover, planar):
    smiles = "C[CH]1=[CH]2[CH]3=[CH2]->[Fe]<-3<-2<-1(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"
    iso = enumerate_isomers(Chem.AddHs(parse_smiles(smiles)), "tetrahedral")[0]
    confs = embed(iso, n=1, seed=42, threads=1)
    cid = confs.ids[0]
    seed = confs._mol.GetConformer(cid).GetPositions().copy()
    assert confs._metal_states()[cid]
    if not requested:
        state = state_with_winding(iso.centres[0], iso.vertices, {})
        confs.iso = iso.with_stereo((state,))
        assert not emb._stereo_targets(confs.iso)
    attempts, faces = [], []

    def flip_then_recover(mol, _cons, *, stiffness, max_iters, conf_ids, record=None, **_kw):
        if max_iters:
            attempts.append(stiffness)
            assert np.array_equal(mol.GetConformer(cid).GetPositions(), seed)
            pos = seed.copy()
            if len(attempts) == 1 or not recover:
                if planar:
                    pos[:, 2] = 0.0  # metal in the face plane: no observed winding, not an opposite sign
                else:
                    pos[:, 0] *= -1
            else:
                pos[:, 0] += 0.1
            mol.GetConformer(cid).SetPositions(pos)
            assert record is not None
            assert record.snapshots is not None
            record.statuses[cid] = 0
            record.snapshots[cid] = [pos.copy()]
            faces.append(confs._metal_states()[cid])
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", flip_then_recover)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda *_args: None)
    frames = []
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10, _frames=frames)

    if requested and recover:
        assert faces == [False, True]
        assert not confs.unrelaxed
        assert len(frames) == 2
        np.testing.assert_allclose(frames[-1], seed + np.array([0.1, 0.0, 0.0]))
    elif requested:
        assert len(attempts) > 1
        assert not any(faces)
        assert confs.unrelaxed == [cid]
        assert len(frames) == 1
        np.testing.assert_array_equal(confs._mol.GetConformer(cid).GetPositions(), seed)
    else:
        assert len(attempts) == 1
        assert not confs.unrelaxed
    assert not confs._acceptance_failures()


def test_finite_gate_valid_max_iteration_endpoint_is_retained(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (1.0, 3.0)}))
    before = mol.GetConformer().GetPositions().copy()
    calls = []

    def stalled_uff(mol, _cons, *, max_iters, conf_ids, record=None, **_kw):
        calls.append(max_iters)
        if max_iters:
            assert record is not None
            moved = mol.GetConformer(0).GetPositions().copy()
            moved[0, 0] += 0.2
            mol.GetConformer(0).SetPositions(moved)
            record.statuses[0] = 1
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", stalled_uff)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [10]
    assert confs.unrelaxed == [0]
    assert not np.array_equal(mol.GetConformer().GetPositions(), before)


def test_relax_ladder_corrects_free_metal_mirrors_before_geometry_acceptance(monkeypatch):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    confs = embed(iso, n=1, seed=1, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    seed = confs._mol.GetConformer(cid).GetPositions().copy()
    attempts = []

    def reflect(mol, _cons, *, max_iters, conf_ids, record=None, **_kw):
        if max_iters:
            attempts.append(max_iters)
            emb._reflect(mol, cid)
            assert record is not None
            if record.snapshots is not None:
                record.snapshots[cid] = [mol.GetConformer(cid).GetPositions().copy()]
            record.statuses[cid] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    frames = []
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10, _frames=frames)

    assert attempts == [10]
    assert confs._metal_states()[cid]
    assert confs._geometry_failure(cid) is None
    assert not confs.unrelaxed
    np.testing.assert_allclose(confs._mol.GetConformer(cid).GetPositions(), seed)
    assert frames == []
    confs._store_trajectory(frames)
    assert confs.trajectory is None


def test_relax_ladder_leaves_inactive_metal_mirrors_untouched(monkeypatch):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    confs = embed(iso, n=1, seed=1, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    seed = confs._mol.GetConformer(cid).GetPositions().copy()
    inactive = confs._mol.AddConformer(Chem.Conformer(confs._mol.GetConformer(cid)), assignId=True)
    confs.ids.append(inactive)
    emb._reflect(confs._mol, inactive)
    before = confs._mol.GetConformer(inactive).GetPositions().copy()

    def reflect(mol, _cons, *, conf_ids, max_iters, record=None, **_kw):
        assert conf_ids == [cid]
        if max_iters:
            emb._reflect(mol, cid)
            assert record is not None
            record.statuses[cid] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    active = confs[:1]
    active._relax_constrained(BASE_STIFFNESS, 10)

    assert not active.unrelaxed
    assert confs._metal_states()[cid]
    np.testing.assert_allclose(confs._mol.GetConformer(cid).GetPositions(), seed)
    np.testing.assert_array_equal(confs._mol.GetConformer(inactive).GetPositions(), before)


def test_unavailable_uff_clears_stale_energies_and_keeps_the_seed(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (1.0, 3.0)}), energies={0: -99.0})
    before = mol.GetConformer().GetPositions().copy()

    def unavailable(*_args, **_kwargs):
        raise UFFTypingError("x")

    monkeypatch.setattr(emb, "restrained_uff", unavailable)
    assert confs._relax_constrained(BASE_STIFFNESS, max_iters=10) is None

    assert confs.energies == {}
    assert confs.unrelaxed == [0]
    assert np.array_equal(mol.GetConformer().GetPositions(), before)


@pytest.mark.parametrize("victim", [0, 1])
def test_later_uff_failure_preserves_already_accepted_endpoints(monkeypatch, victim):
    mol = _with_geometry("CCO")
    mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    confs = Conformers(mol, [0, 1], Constraints(distances={(0, 2): (1.0, 3.0)}), unrelaxed=[0, 1])
    seed = mol.GetConformer().GetPositions().copy()
    endpoint = seed + np.array([0.1, 0.0, 0.0])
    calls = []

    def fail_second_attempt(mol, _cons, *, conf_ids, record, **_kw):
        calls.append(list(conf_ids))
        if len(calls) > 1:
            raise UFFTypingError("unavailable")
        for cid in conf_ids:
            mol.GetConformer(cid).SetPositions(endpoint)
            record.statuses[cid] = 0
            record.snapshots[cid] = [endpoint.copy()]
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", fail_second_attempt)
    monkeypatch.setattr(
        Conformers, "_geometry_failure", lambda _self, cid: emb.Failure("bonding", "retry") if cid == victim else None
    )
    frames = []
    assert confs._relax_constrained(BASE_STIFFNESS, max_iters=10, _frames=frames) is None

    assert calls == [[0, 1], [victim]]
    assert confs.unrelaxed == [victim]
    assert not confs.energies
    np.testing.assert_array_equal(mol.GetConformer(victim).GetPositions(), seed)
    np.testing.assert_array_equal(mol.GetConformer(1 - victim).GetPositions(), endpoint)
    np.testing.assert_array_equal(frames[0], seed)
    np.testing.assert_array_equal(frames[-1], seed if victim == 0 else endpoint)
    assert len(frames) == (1 if victim == 0 else 2)


@pytest.mark.parametrize("constrained", [False, True])
def test_uff_optimizer_failure_is_not_relabelled_as_unavailable(monkeypatch, constrained):
    mol = _with_geometry("CCO")
    cons = Constraints(distances={(0, 2): (1.0, 3.0)}) if constrained else Constraints()
    confs = Conformers(mol, [0], cons)

    def failed(*_args, **_kwargs):
        raise UFFOptimizationError("BFGS failed")

    monkeypatch.setattr(emb, "restrained_uff", failed)
    with pytest.raises(UFFOptimizationError, match="BFGS failed"):
        confs.minimize()


def test_higher_uff_ceiling_preserves_ni_seed_diversity():
    confs = embed(_isomer(_NI_N), n=20, params=bnd.EmbedParams(seed=42, threads=1, prune_rms=-1))
    raw = {cid: confs._mol.GetConformer(cid).GetPositions().copy() for cid in confs.ids}
    record = UFFRecord()
    restrained_uff(confs._mol, confs.cons, conf_ids=confs.ids, record=record)

    assert len(confs.ids) == 20
    assert record.statuses == dict.fromkeys(confs.ids, 0)
    final = {cid: confs._mol.GetConformer(cid).GetPositions().copy() for cid in confs.ids}
    pairs = [
        (_aligned_rms(raw[i], raw[j]), _aligned_rms(final[i], final[j]))
        for i, j in itertools.combinations(confs.ids, 2)
    ]
    assert min(after for _before, after in pairs) >= 0.5
    assert not [(before, after) for before, after in pairs if before > 1.0 and after < 0.5]


def test_constrained_embed_returns_intact_or_seed():
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1).minimize()
    assert len(confs) >= 1
    for cid in confs.ids:
        assert bonding_failure(confs.mol, int(cid), constrained=confs.cons.distances) is None, (
            f"conformer {cid} came back torn"
        )


def test_minimize_preserves_count_and_energies():
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1)
    before = list(confs.ids)
    assert confs.minimize().ids == before
    assert set(confs.energies) == {int(c) for c in confs.ids}
    assert all(np.isfinite(v) for v in confs.energies.values())


def test_constrained_energies_share_one_final_objective(monkeypatch):
    confs = embed(_mol("CCC"), fix={(0, 2): (2.0, 3.0)}, n=2, seed=1)
    calls = []
    real_uff = emb.restrained_uff

    def marked_uff(mol, cons, **kw):
        result = real_uff(mol, cons, **kw)
        calls.append((kw["stiffness"], kw.get("max_iters")))
        return np.full(len(result), 7.0 if kw.get("max_iters") == 0 else 99.0)

    def replace_after_relax(self, *args, **kwargs):
        self.energies = dict.fromkeys(self.ids, -50.0)  # stand in for a hand re-seed's separately scored batch
        return {}

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(Conformers, "_accept_relaxed", replace_after_relax)
    confs.minimize()

    assert calls[-1] == (BASE_STIFFNESS, 0)
    assert set(confs.energies.values()) == {7.0}


def test_relax_result_is_rescored_on_a_common_objective(monkeypatch):
    """`_relax_constrained` leaves scoring to `_rescore_restrained`, one single point on the caller's stiffness."""
    mol = Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)"))
    iso = enumerate_isomers(mol, "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    calls = []
    real_uff = emb.restrained_uff

    def spy_uff(mol, cons, **kw):
        result = real_uff(mol, cons, **kw)
        calls.append((kw.get("max_iters"), result.copy()))
        return result

    monkeypatch.setattr(emb, "restrained_uff", spy_uff)
    ran = confs._relax_constrained(BASE_STIFFNESS)
    assert ran is not None
    assert all(max_iters != 0 for max_iters, _result in calls), "the ladder itself must not take a single point"

    confs._rescore_restrained(BASE_STIFFNESS)
    settled = [cid for cid in confs.ids if cid not in confs.unrelaxed]

    assert calls[-1][0] == 0
    assert [confs.energies[cid] for cid in settled] == list(calls[-1][1])


@pytest.mark.parametrize("operation", ["embed", "minimize"])
def test_rejected_uff_endpoint_reason_survives_seed_restoration(monkeypatch, caplog, operation):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (2.0, 3.0)}))
    seed = mol.GetConformer().GetPositions().copy()
    assert confs._geometry_failure(0) is None

    def tear_bond(mol, _cons, *, conf_ids, max_iters, record=None, **_kw):
        if max_iters:
            assert record is not None
            for cid in conf_ids:
                positions = mol.GetConformer(cid).GetPositions()
                positions[0, 0] += 100.0
                mol.GetConformer(cid).SetPositions(positions)
                record.statuses[cid] = 0  # numerical convergence does not guarantee a valid molecule
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", tear_bond)
    with caplog.at_level(logging.DEBUG, logger="rxembed"):
        confs._relax_constrained(BASE_STIFFNESS, operation=operation)

    assert np.array_equal(mol.GetConformer().GetPositions(), seed)
    assert confs._geometry_failure(0) is None
    assert confs.unrelaxed == [0]
    assert confs.relax_failures[0].kind == "bonding"
    retry = f"{operation}: rejected 1/1 UFF results (1x bond C0-C1 stretched to"
    assert [record.levelno for record in caplog.records if record.getMessage().startswith(retry)] == [logging.DEBUG]


def test_a_rejected_uff_result_that_a_fresh_seed_replaces_logs_no_warning(monkeypatch, caplog):
    """A retry is bookkeeping: a first UFF result that fails, then a fresh seed that passes, logs no WARNING."""
    confs = embed(_isomer(), n=1, seed=1)
    first, cid = confs._mol, confs.ids[0]
    first.GetConformer(cid).SetAtomPosition(0, (100.0, 0.0, 0.0))  # the restored seed must fail the gate too
    real = emb.restrained_uff

    def tear_the_first_seed(mol, cons, **kwargs):
        energies = real(mol, cons, **kwargs)
        if mol is first and kwargs.get("max_iters"):
            for conf_id in kwargs["conf_ids"]:
                mol.GetConformer(int(conf_id)).SetAtomPosition(0, (100.0, 0.0, 0.0))
        return energies

    monkeypatch.setattr(emb, "restrained_uff", tear_the_first_seed)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs.minimize()

    assert cid in confs.relax_failures, "the first UFF result was not rejected, so nothing was retried"
    assert confs.ids == [cid]
    assert not confs.unrelaxed, "the fresh seed must replace the restored one"
    assert [record.getMessage() for record in caplog.records] == []


def test_relax_retry_rejects_an_inverted_donor_hand(monkeypatch, caplog):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    seed_pos = {cid: confs._mol.GetConformer(cid).GetPositions().copy()}
    references = [iso.metal]
    hand = donor_chirality_sign(confs._mol, cid, donor, references)
    inverted = []
    confs.unrelaxed = [cid]

    def reflect(mol, _cons, *, conf_ids, max_iters, record=None, **_kw):
        if max_iters == 0:
            return np.zeros(len(conf_ids))
        for conf_id in conf_ids:
            positions = mol.GetConformer(conf_id).GetPositions()
            positions[:, 0] *= -1.0
            mol.GetConformer(conf_id).SetPositions(positions)
            assert record is not None
            record.statuses[conf_id] = 0
            inverted.append(donor_chirality_sign(mol, conf_id, donor, references))
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)
    with caplog.at_level(logging.DEBUG, logger="rxembed"):
        confs._relax_constrained(BASE_STIFFNESS, 10)

    assert inverted
    assert set(inverted) != {hand}
    assert confs.unrelaxed == [cid]
    assert np.array_equal(confs._mol.GetConformer(cid).GetPositions(), seed_pos[cid])
    assert str(confs.relax_failures[cid]) == f"coordinated donor N{donor} inverts"
    assert f"coordinated donor N{donor} inverts" in caplog.text


def test_relax_retry_can_recover_a_donor_hand_at_a_stronger_rung(monkeypatch):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    references = [iso.metal]
    wanted = donor_chirality_sign(confs._mol, cid, donor, references)
    attempts = []

    def flip_once(mol, _cons, *, stiffness, max_iters, conf_ids, record=None, **_kw):
        attempts.append(stiffness)
        for conf_id in conf_ids:
            if max_iters and stiffness == BASE_STIFFNESS:
                positions = mol.GetConformer(conf_id).GetPositions()
                positions[:, 0] *= -1.0
                mol.GetConformer(conf_id).SetPositions(positions)
            if record is not None:
                record.statuses[conf_id] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", flip_once)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)

    confs._relax_constrained(BASE_STIFFNESS, 10)

    assert attempts[:2] == [BASE_STIFFNESS, 3 * BASE_STIFFNESS]
    assert confs.unrelaxed == []
    assert donor_chirality_sign(confs._mol, cid, donor, references) == wanted


def test_relax_failure_keeps_the_first_physical_reason_when_donor_hand_also_changes(monkeypatch):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    references = [iso.metal]
    hand = donor_chirality_sign(confs._mol, cid, donor, references)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda _self, _cid, _iso=None: Failure("bonding", "heavy-atom bonding/clash failure"),
    )
    monkeypatch.setattr(
        emb,
        "donor_chirality_sign",
        lambda *_args, **_kwargs: "S" if hand != "S" else "R",
    )

    failures = confs._relax_failures({cid: {donor: (hand, references)}})

    assert next(iter(failures.values())) == Failure("bonding", "heavy-atom bonding/clash failure")


def test_trajectory_keeps_only_the_accepted_stiffness(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (2.0, 3.0)}))
    seed = mol.GetConformer(0).GetPositions().copy()
    attempts = []

    def marked_uff(mol, cons, *, stiffness, max_iters, conf_ids=None, record=None, **_kw):
        if max_iters == 0:
            return np.array([0.0])
        attempts.append(stiffness)
        cid = int(conf_ids[0]) if conf_ids else 0
        if record is not None:
            record.statuses[cid] = 0
        positions = mol.GetConformer(cid).GetPositions().copy()
        positions[0, 0] = len(attempts)
        for atom, xyz in enumerate(positions):
            mol.GetConformer(cid).SetAtomPosition(atom, xyz.tolist())
        if record is not None and record.snapshots is not None:
            record.snapshots[cid] = [positions.copy()]
        return np.array([float(stiffness)])

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda _self, _cid, _iso=None: (
            Failure("structural_constraint", "injected failure") if len(attempts) < 2 else None
        ),
    )
    frames = []
    confs._relax_constrained(BASE_STIFFNESS, _frames=frames)
    confs._store_trajectory(frames)

    assert confs.trajectory is not None
    xs = [conf.GetPositions()[0, 0] for conf in confs.trajectory.GetConformers()]
    assert attempts[:2] == [BASE_STIFFNESS, 3 * BASE_STIFFNESS]
    assert xs == pytest.approx([seed[0, 0], 2.0]), "the rejected first-attempt frame leaked into the trajectory"
    assert confs[:0].trajectory is None


# ---------------------------------------------------------------------------------------------------------
# the metal-centre handedness gate
# ---------------------------------------------------------------------------------------------------------


def _hands(confs):
    """The metal-centre hand each returned conformer actually realises, read back from its coordinates."""
    mol = confs.mol
    return [from_geometry(Chem.Mol(mol, False, int(c))).chirality for c in confs.ids]


@pytest.mark.parametrize("want", ["delta", "lambda"])
def test_named_metal_hand_is_preserved(want):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality == want)
    confs = embed(iso, n=8, seed=0xF00D).minimize()
    assert len(confs) >= 4, "the fixture must return enough conformers to be a fair sample"
    assert _hands(confs) == [want] * len(confs)


def test_unstated_metal_hand_skips_hand_check(monkeypatch):

    def boom(self):
        raise AssertionError("the handedness read ran for a caller who named no hand")

    monkeypatch.setattr(emb.Conformers, "_metal_hands", boom)
    monkeypatch.setattr(emb.Conformers, "_metal_states", boom)
    assert len(embed(_mol("OCCCN"), n=4, seed=0xF00D).minimize()) == 4  # organic: there is no `iso` at all
    achiral = next(iso for iso in enumerate_isomers(_mol(_CO_EN), "octahedral") if not iso.chirality)
    assert len(embed(achiral, n=4, seed=0xF00D).minimize()) >= 1  # a metal whose centre states no hand


def test_mirror_freedom_depends_on_metal_inversion():
    assert emb._mirror_is_free(_mol("OCCCN"))
    assert emb._mirror_is_free(_mol("C/C=C/CO")), "E/Z is reflection-invariant and must not block the mirror"
    assert not emb._mirror_is_free(_mol("C[C@H](N)CO"))
    assert not emb._mirror_is_free(_mol("CC(N)CO")), "an sp3 centre inverts whether or not it is assigned"
    assert not emb._mirror_is_free(_mol("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"))


def test_stereocentre_preserves_metal_hand_and_energies():
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    assert not emb._mirror_is_free(iso.mol), "the premise: this fixture must be beyond the free fix"
    confs = embed(iso, n=6, seed=0xF00D).minimize()
    assert _hands(confs) == [iso.chirality] * len(confs)
    assert set(confs.energies) == {int(c) for c in confs.ids}, "a re-seeded conformer must bring its own energy"


def _mirrored_input(smiles):
    """An `Isomer` stating one hand over a conformer of the other: the delta seating on a reflected geometry."""
    delta = next(i for i in enumerate_isomers(_mol(smiles), "octahedral") if i.chirality == "delta")
    confs = embed(delta, n=1, seed=0xF00D).minimize()
    mol = confs.mol
    conf = mol.GetConformer(int(confs.ids[0]))
    pos = conf.GetPositions()
    pos[:, 0] *= -1.0  # the enantiomer, by hand: nothing in the constraint set can tell the two apart
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, xyz.tolist())
    got = from_geometry(Chem.Mol(mol, False, int(confs.ids[0]))).chirality
    assert got == "lambda", f"the premise: this geometry must be the mirror of the stated hand, got {got!r}"
    return Isomer(mol, "octahedral", dict(enumerate(delta.vertices)))


def test_minimize_preserves_isomer_hand():
    confs = minimize(_mirrored_input(_CO_EN))
    assert _hands(confs) == ["delta"]


def test_unfixable_metal_hand_fails_instead_of_returning_the_wrong_identity():
    with pytest.raises(emb.EmbeddingError) as caught:
        minimize(_mirrored_input(_CO_EN_ME))
    assert caught.value.isomer is not None
    assert str(caught.value).startswith(f"{caught.value.isomer}: ")


@pytest.mark.parametrize(
    ("initial", "replacement"),
    [
        ("requested metal hand was not retained", "heavy-atom bonding/clash failure"),
        ("heavy-atom bonding/clash failure", "coordination state at Co0 crossed its donor-slot seating"),
    ],
)
def test_hard_relax_failure_before_or_after_replacement_is_required(monkeypatch, initial, replacement):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    confs = embed(iso, n=1, seed=1)
    replaced = False

    def wrong_failure(self, operation="minimize", validator=None, ids=None):
        reason = replacement if replaced else initial
        kind = "metal_state" if reason.startswith(("requested metal", "coordination state")) else "bonding"
        return {Failure(kind, reason): list(self.ids if ids is None else ids)}

    def broken_replacement(_self, failed, *_args, **_kwargs):
        nonlocal replaced
        replaced = True
        return list(failed)

    monkeypatch.setattr(Conformers, "_acceptance_failures", wrong_failure)
    monkeypatch.setattr(Conformers, "_replace_failed", broken_replacement)

    with pytest.raises(emb.EmbeddingError) as caught:
        confs.minimize()
    assert caught.value.isomer is iso
    assert replacement in str(caught.value)


def test_requested_coordination_shape_must_fill_seed_count(monkeypatch):
    confs = embed(_isomer(), n=1, seed=1)
    monkeypatch.setattr(
        Conformers,
        "_acceptance_failures",
        lambda self, *_args, **_kwargs: {
            Failure("metal_state", "coordination state at Pt0 is nonplanar"): list(self.ids)
        },
    )
    monkeypatch.setattr(Conformers, "_replace_failed", lambda _self, failed, *_args, **_kwargs: list(failed))

    with pytest.raises(emb.EmbeddingError, match=r"Pt0 is nonplanar in 1/1 rejected seeds; try the other hand"):
        confs.minimize()


@pytest.mark.parametrize(
    ("request_kind", "kind", "remedy"),
    [
        ("molecule", "bonding", "try another seed="),
        ("fixed molecule", "bonding", "try a looser fix="),
        ("smiles isomer", "bonding", "try another isomer"),
        ("smiles isomer", "ml_distance", "try another isomer"),
        ("geometry isomer", "ml_distance", "try lengths='input' or another isomer"),
        ("smiles isomer", "metal_state", "try the other hand from rx.metal"),
    ],
)
def test_embedding_error_offers_only_remedies_the_request_can_use(monkeypatch, request_kind, kind, remedy):
    if request_kind.endswith("molecule"):
        cons = Constraints(frozen={0}) if request_kind == "fixed molecule" else Constraints()
        confs = Conformers(_with_geometry("CC"), [0], cons)
    else:
        iso = _isomer()
        if request_kind == "geometry isomer":
            iso = from_geometry(embed(iso, n=1, seed=1).minimize().mol)
        confs = embed(iso, n=1, seed=1)
    monkeypatch.setattr(
        Conformers, "_acceptance_failures", lambda self, *_args, **_kw: {Failure(kind, "synthetic"): list(self.ids)}
    )

    with pytest.raises(emb.EmbeddingError) as caught:
        confs._accept_relaxed(BASE_STIFFNESS, 1, operation="embed", allow_replacement=False)
    assert str(caught.value).endswith(f": synthetic in 1/1 rejected seeds; {remedy}")


def test_coordination_gate_accepts_a_graph_equivalent_site_swap():
    iso = enumerate_isomers(_mol("[Pt](F)(F)(Cl)Br"), "square_planar", stereo="free")[0]
    conformers = embed(iso, n=1, seed=7).minimize()
    left, right = [donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetSymbol() == "F"]
    positions = conformers._mol.GetConformer().GetPositions().copy()
    positions[[left, right]] = positions[[right, left]]
    conformers._mol.GetConformer().SetPositions(positions)

    assert emb._coordination_state_failure(conformers._mol, conformers.ids[0], conformers.iso) is None


def test_handed_coordination_gate_rejects_an_improper_equivalent_reseating():
    iso = enumerate_isomers(_mol("[Co](N)(N)(P)(P)(O)(O)"), "octahedral", stereo="free")[0]
    donors = {
        atomic_number: [atom for atom in iso.donors if iso.mol.GetAtomWithIdx(atom).GetAtomicNum() == atomic_number]
        for atomic_number in (7, 8, 15)
    }
    n, o, p = donors[7], donors[8], donors[15]
    state = iso.centres[0]._replace(vertices=(o[0], n[0], n[1], p[0], p[1], o[1]), hand="delta")
    iso = iso.with_stereo((state,))
    directions = np.array(
        [
            [0.333, 0.753, -0.568],
            [-0.362, 0.546, 0.755],
            [0.957, 0.003, 0.291],
            [-0.544, -0.760, 0.356],
            [0.185, -0.732, -0.656],
            [-0.847, 0.278, -0.454],
        ]
    )
    conf = Chem.Conformer(iso.mol.GetNumAtoms())
    for slot, atom in enumerate(state.vertices):
        conf.SetAtomPosition(atom, 2 * directions[slot])
    iso.mol.AddConformer(conf)

    achiral = iso.with_stereo((state._replace(hand=""),))
    assert emb._coordination_state_failure(achiral.mol, 0, achiral) is None, "the distorted octahedron itself is valid"
    targets = emb._stereo_targets(iso)
    ranks, eta2 = emb._winding_ranks(iso.mol, targets)
    assert emb._seed_stereo_matches(iso.mol, 0, iso, targets, ranks, eta2, False)
    assert emb._coordination_state_failure(iso.mol, 0, iso) is not None, "an improper donor swap is the other hand"


def test_partial_coordination_gate_preserves_vacancy_and_equivalent_sites():
    iso = enumerate_isomers(_mol("[Pt](F)(F)Cl"), "square_planar", stereo="free")[0]
    conformers = embed(iso, n=1, seed=7)
    conf = conformers._mol.GetConformer(conformers.ids[0])
    metal = np.asarray(conf.GetAtomPosition(iso.metal))
    occupied = [slot for slot, donor in enumerate(iso.vertices) if donor >= 0]
    ideal = np.asarray([POLYHEDRA[iso.geometry].vertex_dirs[slot] for slot in occupied], float)
    donors = [iso.vertices[slot] for slot in occupied]
    fluorines = [donor for donor in donors if iso.mol.GetAtomWithIdx(donor).GetSymbol() == "F"]
    chlorine = next(donor for donor in donors if donor not in fluorines)

    for donor, direction in zip(donors, ideal, strict=True):
        conf.SetAtomPosition(donor, tuple(metal + 2.0 * direction))
    positions = conf.GetPositions().copy()
    positions[fluorines] = positions[fluorines[::-1]]
    conf.SetPositions(positions)
    assert emb._coordination_state_failure(conformers._mol, conformers.ids[0], conformers.iso) is None

    positions = conf.GetPositions().copy()
    positions[[fluorines[0], chlorine]] = positions[[chlorine, fluorines[0]]]
    conf.SetPositions(positions)
    assert emb._coordination_state_failure(conformers._mol, conformers.ids[0], conformers.iso) is not None


def test_independent_ligand_ez_is_restored_when_relaxation_changes_it():
    isomer = enumerate_isomers(_mol(r"C/N1=C(/C)CCCCCC[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-1"), "square_planar")[0]
    confs = embed(isomer, n=1, seed=7).minimize()
    realised = bond_stereo(stereo_from_3d(confs.mol, exclude=metal_indices(confs.mol)))

    assert len(confs) == 1
    assert realised == bond_stereo(isomer.stereo_label)


def test_metal_referenced_imine_ez_is_selected_before_relaxation():
    isomers = enumerate_isomers(_mol("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]"), "square_planar")

    for isomer in isomers:
        conformers = embed(isomer, n=3, seed=42)
        assert len(conformers) == 3
        wanted = bond_stereo(isomer.stereo_label)
        published = conformers.mol
        for cid in conformers.ids:
            one = Chem.Mol(published, False, int(cid))
            realised = bond_stereo(stereo_from_3d(one, exclude=metal_indices(one)))
            assert realised == wanted


def test_stereo_selection_streams_one_bounded_seed_budget(monkeypatch, caplog):
    isomer = enumerate_isomers(_mol("CC=[NH]->[Pt](Cl)(Br)I"), "tetrahedral")[0]
    seen = []

    def reject_all(_mol, _cons, count, params, **_kwargs):
        seen.append((count, params.seed))
        return []

    monkeypatch.setattr(emb, "seed_count", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(emb, "seed_coordinates", reject_all)

    with (
        caplog.at_level(logging.DEBUG, logger="rxembed"),
        pytest.raises(emb.EmbeddingError, match="found 0/1 DG seeds"),
    ):
        embed(isomer, n=1, seed=42)
    assert seen == [(4, 42), (4, 43), (2, 44)]
    assert "DG search returned 0 candidates; 0/1 satisfied requested stereo selection" in caplog.text


def test_relax_gate_rejects_inverted_ligand_point_stereo(monkeypatch):
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    conformers = embed(isomer, n=1, seed=7)
    atom, expected = next(iter(point_stereo(isomer.stereo_label).items()))
    found = {"R": "S", "S": "R"}[expected]
    symbol = isomer.mol.GetAtomWithIdx(atom).GetSymbol()

    def inverted(published, **_kwargs):
        assert published.GetBondBetweenAtoms(*isomer.donor_bonds[0]).GetBondType() == Chem.BondType.DATIVE
        return f"{symbol}{atom}:{found}"

    monkeypatch.setattr(emb, "stereo_from_3d", inverted)
    monkeypatch.setattr(emb, "_coordination_state_failure", lambda *_args, **_kwargs: None)

    assert str(conformers._geometry_failure(conformers.ids[0])) == f"{symbol}{atom} reads {found}, not {expected}"


def test_relax_gate_offers_locked_hand_enumeration_for_a_bound_amine(monkeypatch):
    """A bound amine's hand is the input arrangement's, so losing it names the option that enumerates both."""
    isomer = _isomer(ONE_ARM_BOUND_PT)
    conformers = embed(isomer, n=1, seed=7)
    wanted = point_stereo(isomer.stereo_label)
    monkeypatch.setattr(emb, "stereo_from_3d", lambda *_args, **_kwargs: f"N4:{'S' if wanted[4] == 'R' else 'R'}")
    monkeypatch.setattr(emb, "_coordination_state_failure", lambda *_args, **_kwargs: None)

    failure = conformers._geometry_failure(conformers.ids[0])

    assert failure.kind == "locked_stereo"
    assert emb.remedy(failure.kind, isomer, conformers.cons) == (
        "try rx.metal(..., stereo={'locked': 'racemic'}) or another isomer"
    )
    assert conformers._required_failure({failure: [0]}) == {"locked_stereo"}


def test_relax_gate_distinguishes_unassigned_ligand_stereo(monkeypatch):
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    monkeypatch.setattr(emb, "stereo_from_3d", lambda *_args, **_kwargs: "")

    failure = emb._ligand_stereo_failure(isomer.mol, isomer)

    assert failure.kind == "ligand_stereo_unassigned"
    assert Conformers(_with_geometry("CC"), [0], Constraints())._required_failure({failure: [0]}) == {
        "ligand_stereo_unassigned"
    }


def test_relax_gate_rejects_reflected_ligand_axial_stereo():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
    wanted = stereo_from_3d(mol)
    request = SimpleNamespace(stereo_label=wanted)
    pair, expected = next(iter(axis_stereo(wanted).items()))
    positions = mol.GetConformer().GetPositions()
    positions[:, 0] *= -1
    mol.GetConformer().SetPositions(positions)
    found = axis_stereo(stereo_from_3d(mol))[pair]

    failure = emb._ligand_stereo_failure(mol, request)
    a, b = sorted(pair)
    names = f"{mol.GetAtomWithIdx(a).GetSymbol()}{a}-{mol.GetAtomWithIdx(b).GetSymbol()}{b}"
    assert str(failure) == f"axis {names} reads {found}, not {expected}"
    assert Conformers(mol, [0], Constraints())._required_failure({failure: [0]}) == {"ligand_stereo"}


def test_required_failure_policy_uses_kind_not_message_wording():
    conformers = Conformers(_with_geometry("CC"), [0], Constraints())
    assert conformers._required_failure({Failure("ligand_stereo", "reworded completely"): [0]}) == {"ligand_stereo"}
    assert conformers._required_failure({Failure("bonding", "double bond C1=C2 reads Z, not E"): [0]}) == set()


@pytest.mark.parametrize("operation", ["embed", "minimize"])
@pytest.mark.parametrize("survivor", [False, True])
def test_total_embed_rejection_reports_original_bond_failure(operation, survivor):
    mol = _with_geometry("CC")
    if survivor:
        mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    SetBondLength(mol.GetConformer(0), 0, 1, 10.0)
    conformers = Conformers(mol, [conf.GetId() for conf in mol.GetConformers()])

    if operation == "embed" and not survivor:
        with pytest.raises(emb.EmbeddingError, match=r"embed: bond C0-C1 stretched to 10.00 A in 1/1 rejected seeds"):
            conformers._accept_relaxed(BASE_STIFFNESS, 1, operation=operation, allow_replacement=False)
    else:
        failures = conformers._accept_relaxed(BASE_STIFFNESS, 1, operation=operation, allow_replacement=False)
        assert {failure.kind for failure in failures} == {"bonding"}
    assert conformers.ids == ([1] if survivor else [])


def test_valid_unrelaxed_reseed_preserves_its_status(monkeypatch):
    confs = embed(_isomer(), n=1, seed=1)
    victim = confs.ids[0]
    confs.unrelaxed = [victim]

    monkeypatch.setattr(emb, "seed_conformers", lambda *_args, **_kwargs: (Chem.Mol(confs._mol), [victim], None))

    def leave_unrelaxed(self, *_args, **_kwargs):
        self.unrelaxed = list(self.ids)
        self.energies = dict.fromkeys(self.ids, 42.0)
        return [0.0] * len(self.ids)

    monkeypatch.setattr(Conformers, "_relax_constrained", leave_unrelaxed)
    monkeypatch.setattr(Conformers, "_acceptance_failures", lambda *_args, **_kwargs: {})

    assert confs._replace_failed([victim], BASE_STIFFNESS, 1) == []
    assert confs.unrelaxed == [victim]
    assert confs.energies == {victim: 42.0}


@pytest.mark.parametrize("count", [1, 7])
def test_replacements_rank_valid_settled_scores_without_touching_survivors(monkeypatch, count):
    mol = Chem.MolFromSmiles("[He]")
    for cid in range(8):
        conf = Chem.Conformer(1)
        conf.SetId(cid)
        conf.SetAtomPosition(0, (float(cid), 0.0, 0.0))
        mol.AddConformer(conf, assignId=False)
    target = Chem.Mol(mol)
    for conf in target.GetConformers():
        conf.SetId(conf.GetId() + 20)
    failed = list(range(20, 20 + count))
    owner = Conformers(Chem.Mol(target), [*failed, 27], params=bnd.EmbedParams(seed=42), energies={27: -77.0})
    calls = []
    marker = object()

    def seeds(*_args, **_kwargs):
        calls.append(True)
        return Chem.Mol(mol), list(range(8)), None

    def relax(batch, *_args):
        batch.energies = {0: -100.0, 1: 5.0, 2: 1.0, 3: 1.0, 4: -1000.0, 6: -np.inf, 7: np.nan}
        batch.unrelaxed = [4]

    def accept(_batch, operation, validator=None):
        assert operation == "embed"
        assert validator is marker
        return {Failure("physical_geometry", "invalid lowest energy"): [0]}

    monkeypatch.setattr(emb, "seed_conformers", seeds)
    monkeypatch.setattr(Conformers, "_relax_once", relax)
    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)
    assert owner._replace_failed(failed, BASE_STIFFNESS, 1, operation="embed", validator=marker) == []
    expected = [2, 3, 1, 4, 5, 6, 7][:count]
    assert [int(owner._mol.GetConformer(cid).GetAtomPosition(0).x) for cid in failed] == expected
    assert calls == [True]
    assert owner.ids == [*failed, 27]
    assert owner.energies[20] == 1.0
    assert owner.energies[27] == -77.0
    assert owner.unrelaxed == ([23] if count == 7 else [])
    if count == 7:
        assert 24 not in owner.energies
    np.testing.assert_array_equal(owner._mol.GetConformer(27).GetPositions(), target.GetConformer(27).GetPositions())


def test_restrained_rescore_does_not_rank_unrelaxed_seeds(monkeypatch):
    confs = embed(_mol("CCC"), n=2, params=bnd.EmbedParams(seed=1, prune_rms=-1))
    confs.unrelaxed = [confs.ids[1]]
    seen = []

    def score(_mol, _cons, *, conf_ids, **_kwargs):
        seen.extend(conf_ids)
        return [3.0] * len(conf_ids)

    monkeypatch.setattr(emb, "restrained_uff", score)
    confs._rescore_restrained(BASE_STIFFNESS)

    assert seen == [confs.ids[0]]
    assert confs.energies == {confs.ids[0]: 3.0}


@pytest.mark.parametrize(
    "success_on", [2, 3, 4], ids=("second-batch-rescue", "third-batch-rescue", "third-batch-rejected")
)
def test_replacement_searches_three_bounded_serial_batches(monkeypatch, success_on):
    confs = embed(_isomer(), n=1, seed=1, threads=3)
    from rdkit.Chem import rdDistGeom

    confs.params = bnd.EmbedParams(
        seed=1,
        threads=3,
        knowledge=False,
        prune_rms=-1,
        native=rdDistGeom.ETDG(),  # useBasicKnowledge=False by default, matching knowledge=False
        coplanar_14=False,
        metal_floor_relief=False,
    )
    victim = confs.ids[0]
    calls = []

    def seeds(source, _cons, _iso, n, trial_params, **_kwargs):
        assert trial_params.native is confs.params.native
        assert not trial_params.coplanar_14
        assert not trial_params.metal_floor_relief
        calls.append((n, trial_params.seed, trial_params.threads, trial_params.knowledge, trial_params.prune_rms))
        mol = Chem.Mol(source)
        ids = [conf.GetId() for conf in mol.GetConformers()]
        return mol, ids, None

    monkeypatch.setattr(emb, "seed_conformers", seeds)
    monkeypatch.setattr(Conformers, "_relax_once", lambda *_args, **_kwargs: None)
    constrained_relaxations = []
    monkeypatch.setattr(
        Conformers,
        "_relax_constrained",
        lambda self, *_args, **_kwargs: constrained_relaxations.append(list(self.ids)),
    )
    attempts = 0

    def accept(self, *_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        missed = Failure("structural_constraint", "missed structural constraint")
        return {missed: list(self.ids)} if attempts < success_on else {}

    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)

    expected = [] if success_on < 4 else [victim]
    assert confs._replace_failed([victim], BASE_STIFFNESS, 1) == expected
    assert calls == [(4, seed, 3, False, -1) for seed in range(2, 2 + min(success_on, 3))]
    assert len(constrained_relaxations) == min(success_on, 3)
    child = confs[0]
    assert child.params is confs.params
    assert child._mol is confs._mol
    assert child.unrelaxed is not confs.unrelaxed
    assert child.uff is not confs.uff


# ---------------------------------------------------------------------------------------------------------
# replacement's two stop rules: a single crossed seating, or the same (kind, atoms) twice in a row
# ---------------------------------------------------------------------------------------------------------


_REPLACEMENT_VICTIM = 5  # an arbitrary id distinct from the fresh batch's own conformer id (0)


def _replacement_owner(monkeypatch, *, accept, relax_failures=None, ladder=None):
    """Build a one-atom `Conformers` whose fresh-seed batches each seed exactly conformer 0.

    ``accept`` stands in for `_acceptance_failures`; every batch relaxes as a no-op that records ``ladder``,
    when given, as its seed's rejected UFF endpoint reason.
    """
    victim_mol = Chem.MolFromSmiles("[He]")
    victim_conf = Chem.Conformer(1)
    victim_conf.SetId(_REPLACEMENT_VICTIM)
    victim_mol.AddConformer(victim_conf, assignId=False)
    owner = Conformers(
        victim_mol,
        [_REPLACEMENT_VICTIM],
        params=bnd.EmbedParams(seed=1),
        relax_failures=relax_failures or {},
    )
    batch_mol = Chem.MolFromSmiles("[He]")
    batch_mol.AddConformer(Chem.Conformer(1), assignId=False)  # id 0: a fresh batch's own seed
    calls = []

    def seeds(*_args, **_kwargs):
        calls.append(True)
        return Chem.Mol(batch_mol), [0], None

    monkeypatch.setattr(emb, "seed_conformers", seeds)

    def relax(batch, *_args, **_kwargs):
        if ladder is not None:
            batch.relax_failures = dict.fromkeys(batch.ids, ladder)

    monkeypatch.setattr(Conformers, "_relax_once", relax)
    monkeypatch.setattr(Conformers, "_relax_constrained", relax)
    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)
    return owner, _REPLACEMENT_VICTIM, calls


def test_replacement_stops_after_two_batches_tear_the_same_ligand_bond(monkeypatch):
    tear = Failure("bonding", "heavy-atom bonding/clash failure (bond 3-6 2.500 A above 1.690 A)", atoms=(3, 6))
    owner, victim, calls = _replacement_owner(monkeypatch, accept=lambda self, *_a, **_k: {tear: list(self.ids)})

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 2, "a third fresh batch cannot tear the same bond differently"
    assert result == [victim]


def test_replacement_keeps_searching_after_one_batch_reads_the_same_wrong_trigonal_bipyramid(monkeypatch):
    wrong_shape = Failure("coordination_shape", "coordination state at M0: expected SPY, found TBP", atoms=(0,))
    attempts = []

    def accept(self, *_args, **_kwargs):
        attempts.append(True)
        return {wrong_shape: list(self.ids)} if len(attempts) == 1 else {}

    owner, victim, calls = _replacement_owner(monkeypatch, accept=accept)

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 2, "one bad batch alone must not stop the search (the refuted one-batch rule)"
    assert result == []


def test_replacement_keeps_searching_after_two_batches_read_the_same_wrong_shape(monkeypatch):
    """A near-tie shape residual can still cross on a third seed, so the two-batch rule must not apply to it.

    A square-pyramidal-vs-trigonal-bipyramidal near tie (VALRAE, seed 1/2 of `enumerate_isomers`) needed its
    third replacement round to succeed; a same-shape-twice stop cost that rescue.
    """
    wrong_shape = Failure("coordination_shape", "coordination state at M0: expected SPY, found TBP", atoms=(0,))
    attempts = []

    def accept(self, *_args, **_kwargs):
        attempts.append(True)
        return {wrong_shape: list(self.ids)} if len(attempts) <= 2 else {}

    owner, victim, calls = _replacement_owner(monkeypatch, accept=accept)

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 3, "a wrong labelled shape never stops on repetition, only a fresh seed can rescue it"
    assert result == []


def test_replacement_keeps_searching_when_planar_stereo_is_wrong_twice(monkeypatch):
    site_less = Failure("metal_state", "requested metal hand or haptic winding was not retained")
    attempts = []

    def accept(self, *_args, **_kwargs):
        attempts.append(True)
        return {site_less: list(self.ids)} if len(attempts) <= 2 else {}

    owner, victim, calls = _replacement_owner(monkeypatch, accept=accept)

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 3, "a whole-molecule failure names no atoms, so it never trips the same-atoms rule"
    assert result == []


def test_replacement_stops_when_every_seed_crosses_to_another_seating(monkeypatch):
    wrong_shape = Failure("coordination_shape", "coordination state at M0 is nonplanar", atoms=(0,))
    crossed = Failure("seating_crossed", "coordination state at M0 crossed its donor-slot seating", atoms=(0,))
    owner, victim, calls = _replacement_owner(
        monkeypatch,
        accept=lambda self, *_a, **_k: {crossed: list(self.ids)},
        relax_failures={_REPLACEMENT_VICTIM: wrong_shape},  # this endpoint can reach a wrong labelled shape
        ladder=wrong_shape,  # the seed's UFF endpoint read a wrong shape; its restored seed crossed seating
    )

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 1, "today's behaviour: every seed crossing seating stops after one batch"
    assert result == [victim]


def test_requested_metal_hand_must_fill_the_seed_count(monkeypatch, caplog):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: False)

    with caplog.at_level(logging.DEBUG, logger="rxembed"), pytest.raises(emb.EmbeddingError) as caught:
        embed(iso, n=1, seed=1)
    assert (
        str(caught.value) == f"{iso}: found 0/1 DG seeds with the requested metal and ligand stereo; try another isomer"
    )
    assert caught.value.isomer is iso
    summary = next(record for record in caplog.records if record.msg.startswith("DG search returned"))
    assert summary.args[0] > 0
    assert "0/1 satisfied requested stereo selection" in summary.message


def test_ligand_hard_chirality_falls_back_when_it_blocks_the_metal_hand(monkeypatch):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    mol, cons, prepared, graft = emb.prepare(iso)
    calls = []

    def seeds(candidate, _cons, n, _params, *, enforce_chirality=True, **_kwargs):
        calls.append((n, enforce_chirality, _kwargs["max_attempts"]))
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "seed_count", lambda *_args, **_kwargs: 8)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: not calls[-1][1])

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert calls == [(4, True, 30), (4, False, 30)]
    assert len(ids) == target == 1


@pytest.mark.parametrize("n", [1, 8])
def test_stereo_seed_selection_prefers_a_seed_with_the_requested_geometry(monkeypatch, n):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    mol, cons, prepared, graft = emb.prepare(iso)

    def seeds(candidate, _cons, n, _params, **_kwargs):
        candidate.RemoveAllConformers()
        for marker in range(n):
            conf = Chem.Conformer(candidate.GetNumAtoms())
            conf.SetAtomPosition(0, (float(marker), 0.0, 0.0))
            candidate.AddConformer(conf, assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(emb, "_seed_geometry_matches", lambda _mol, cid, _iso: cid == 2)

    seeded, ids, target = emb.seed_conformers(mol, cons, prepared, n, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert len(ids) == target == n
    assert len({seeded.GetConformer(cid).GetAtomPosition(0).x for cid in ids}) == n
    assert seeded.GetConformer(ids[0]).GetAtomPosition(0).x == 2.0


def test_hand_only_cobalt_ethylenediamine_seed_selection_ranks_no_windings(monkeypatch):
    """A hand-only target has no haptic face, so seed selection must never read a winding for it.

    This checks an internal call count, not a public result: `_seed_stereo_matches` compares an empty
    realised winding against an empty requested one, which is always a match, so the comparison is skipped
    rather than performed for nothing.
    """
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    real_winding_signature = emb.winding_signature
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return real_winding_signature(*args, **kwargs)

    monkeypatch.setattr(emb, "winding_signature", spy)

    confs = embed(iso, n=1, seed=1)

    assert confs.ids
    assert not calls


_PHEN_W_CO3 = "[O+]#[C-]->[W]1(<-[C-]#[O+])(<-[C-]#[O+])<-n2cccc3ccc4ccc[n]->1c4c32"


def test_a_raw_dg_seed_is_gated_after_relaxation_not_redrawn_for_its_pre_relax_shape():
    """A DG seed only approximates its metal angle windows; its relaxed endpoint, not its raw shape, is gated."""
    for iso in rx.metal(_PHEN_W_CO3, "trigonal_bipyramidal"):
        ensemble = rx.embed(iso, n=1, seed=42, threads=1)
        assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
        assert not ensemble.unrelaxed
        assert all(report.ok() for report in ensemble.check().values())


@pytest.mark.parametrize("kind", ["ez", "axial"])
def test_nonpoint_ligand_stereo_does_not_enable_the_point_fallback(monkeypatch, kind):
    iso = _isomer("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", "square_planar")
    mol, cons, prepared, graft = emb.prepare(iso)
    calls = []
    if kind == "axial":
        monkeypatch.setattr(emb, "bond_stereo", lambda _label: {})
        monkeypatch.setattr(emb, "axis_stereo", lambda _label: {(0, 1): "Ra"})
    else:
        monkeypatch.setattr(emb, "axis_stereo", lambda _label: {})

    def seeds(candidate, _cons, n, _params, *, enforce_chirality=True, **kwargs):
        calls.append((enforce_chirality, kwargs["max_attempts"]))
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert calls == [(True, 0)]
    assert len(ids) == target == 1


def test_stereo_seed_budget_scales_with_independent_metal_states(monkeypatch):
    iso = _isomer()
    mol, cons, prepared, graft = emb.prepare(iso)
    state = SimpleNamespace(hand="delta")
    targets = [(state, (), {}, {}), (state, (), {}, {})]
    calls, inspected = [], 0

    def seeds(candidate, _cons, n, _params, **_kwargs):
        calls.append(n)
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    def twelfth_seed_matches(*_args, **_kwargs):
        nonlocal inspected
        inspected += 1
        return inspected == 12

    monkeypatch.setattr(emb, "_stereo_targets", lambda _iso: targets)
    monkeypatch.setattr(emb, "_winding_ranks", lambda *_args: ({}, None))
    monkeypatch.setattr(emb, "seed_count", lambda *_args, **_kwargs: 8)
    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", twelfth_seed_matches)

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert calls == [4, 4, 4]
    assert inspected == 12
    assert len(ids) == target == 1


def test_free_chirality_seed_selection_still_rejects_the_ligand_mirror():
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    seeded = embed(iso, n=1, seed=5)
    cid = seeded.ids[0]

    no_metal_targets = (seeded._mol, cid, iso, [], {}, None, False)
    assert emb._seed_stereo_matches(*no_metal_targets, check_ligand=True)
    emb._reflect(seeded._mol, cid)
    assert not emb._seed_stereo_matches(*no_metal_targets, check_ligand=True)


def test_unconstrained_max_iteration_replaces_invalid_geometry_and_keeps_status(monkeypatch):
    confs = embed(_mol("CCC"), n=1, seed=1)
    owner = confs._mol

    def stall(mol, _cons, **kwargs):
        ids = kwargs.get("conf_ids") or [conf.GetId() for conf in mol.GetConformers()]
        if mol is owner:
            mol.GetConformer(ids[0]).SetAtomPosition(0, (99.0, 99.0, 99.0))
        record = kwargs.get("record")
        if record is not None:
            record.statuses.update(dict.fromkeys(ids, 1))
        return [0.0] * len(ids)

    monkeypatch.setattr(emb, "restrained_uff", stall)
    confs.minimize()

    assert confs.ids
    assert confs.unrelaxed == confs.ids
    assert all(confs._geometry_failure(cid) is None for cid in confs.ids)


def test_measure_reports_distance_and_angle():
    mol = _mol("CCCCO")
    confs = embed(mol, fix={(0, 4): 3.0}, n=6, seed=1).minimize()
    got = confs.measure((0, 4))
    assert set(got) == {"mean", "min", "max", "n"}
    assert got["n"] == len(confs.ids)
    assert abs(got["mean"] - 3.0) < 0.1, f"the stated 3.0 A was not realised: {got}"
    assert len(confs.measure((0, 1, 4))) == 4, "an angle (3 atoms) must work too"
    with pytest.raises(ValueError, match=r"2 \(distance\), 3 \(angle\) or 4"):
        confs.measure((0,))


def test_template_composes_with_rigid_core_forms():
    mol = _mol("CC(=O)Nc1ccccc1")
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    ref = Chem.Mol(mol)
    pos = ref.GetConformer().GetPositions()

    both = embed(mol, template=(ref, {0: 0, 1: 1}), fix=[2, 3], n=2, seed=1)
    assert sorted(both.cons.frozen) == [0, 1, 2, 3], "a list fix beside a template must not be dropped"

    mixed = embed(mol, template=(ref, {0: 0, 1: 1}), fix={2: tuple(pos[2])}, n=2, seed=1)
    assert sorted(mixed.cons.frozen) == [0, 1, 2]

    alone = embed(mol, template=(ref, {i: i for i in (0, 1, 2, 3)}), n=2, seed=1)
    assert sorted(alone.cons.frozen) == [0, 1, 2, 3]
    assert alone.cons.distances.keys() == embed(mol, fix=[0, 1, 2, 3], n=2, seed=1).cons.distances.keys(), (
        "a template and an own-coords fix over the same atoms must build the same rigid body"
    )
