"""Test the core embed and Conformers API."""

from __future__ import annotations

import importlib
import itertools
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, SetBondLength

from rxembed import bounds as bnd
from rxembed import metal_core as _metal
from rxembed import stereo as _stereo
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL, Constraints, resolve_core
from rxembed.embed import BASE_STIFFNESS, Conformers, Failure, embed, fold_substrate, minimize
from rxembed.metal_core import TRANSITION_METALS, coplanar
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer, from_geometry
from rxembed.metal_polyhedron import POLYHEDRA
from rxembed.metal_smiles import cxsmiles, parse_smiles
from rxembed.relax import UFFOptimizationError, UFFTypingError, bonding_ok, restrained_uff

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
    confs = embed(mol, fix=fix, n=1, seed=0xF00D, prune_rms=-1).minimize()
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
    confs = embed(_mol("[O-].ClCCCCBr"), fix=fix, n=1, seed=1, prune_rms=-1)
    assert not confs._fixed_geometry_ok(confs.ids[0]), "the raw seed must miss for this test to exercise rejection"

    monkeypatch.setattr(
        emb,
        "restrained_uff",
        lambda mol, cons, conf_ids=None, **kw: np.zeros(len(conf_ids if conf_ids is not None else mol.GetConformers())),
    )
    seed_calls = []
    real_seed = emb.seed_conformers

    def tracked_seed(*args, **kwargs):
        seed_calls.append(kwargs["seed"])
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

    assert not emb._coordination_state_ok(confs._mol, confs.ids[0], iso)


def test_coordination_gate_rejects_another_named_shape_below_the_residual_floor():
    iso = _isomer("[Pt](F)(Cl)(Br)I", "tetrahedral")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA["seesaw"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))

    assert not emb._coordination_state_ok(confs._mol, confs.ids[0], iso)


def test_coordination_gate_rejects_a_nearer_cn7_shape_below_the_residual_floor():
    iso = _isomer("[Re](F)(F)(F)(F)(F)(F)F", "pentagonal_bipyramidal")
    conf = Chem.Conformer(iso.mol.GetNumAtoms())
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA["capped_trigonal_prismatic"].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))
    iso.mol.AddConformer(conf)

    assert _metal.classify_geometry(iso.mol, iso.metal, list(iso.vertices), 0) == "capped_trigonal_prismatic"
    assert not emb._coordination_state_ok(iso.mol, 0, iso)


def test_coordination_gate_accepts_an_ideal_mixed_high_coordination_state():
    iso = _isomer("[La](F)(F)(F)(F)(F)(F)(F)Cl", "square_antiprism")
    conf = Chem.Conformer(iso.mol.GetNumAtoms())
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    for donor, direction in zip(iso.vertices, POLYHEDRA[iso.geometry].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))
    iso.mol.AddConformer(conf)

    assert emb._coordination_state_ok(iso.mol, 0, iso)


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

    assert _metal.classify_geometry(confs._mol, iso.metal, list(iso.vertices), confs.ids[0]) == "square_pyramidal"
    assert not emb._coordination_state_ok(confs._mol, confs.ids[0], iso)


def test_publication_rejects_a_different_coordination_arrangement():
    isomers = enumerate_isomers(_mol("[Pt](F)(Cl)(Br)I"), "square_planar", screen=False)
    selected = embed(isomers[0], n=1, seed=1, threads=1)
    alternate = embed(isomers[1], n=1, seed=1, threads=1)

    assert not emb._coordination_state_ok(alternate.mol, 0, isomers[0])
    assert emb._coordination_state_ok(selected.mol, 0, isomers[0])


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
        assert emb._coordination_state_ok(mol, 0, requested) == (index == 0), (
            f"accepted the wrong seating or rejected the requested one: {iso}"
        )


def test_coordination_gate_rejects_nonfinite_donor_coordinates():
    iso = _isomer()
    confs = embed(iso, n=1, seed=1)
    confs._mol.GetConformer(confs.ids[0]).SetAtomPosition(iso.donors[0], (float("nan"), 0.0, 0.0))

    assert not emb._coordination_state_ok(confs._mol, confs.ids[0], iso)


def test_structural_gate_rejects_undefined_coplanarity():
    mol = _with_geometry("CCCC")
    conf = mol.GetConformer()
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))):
        conf.SetAtomPosition(atom, xyz)
    cons = Constraints(coplanar=[(0, 1, 2, 3, 180.0, 10.0)])

    assert emb._structural_failure(mol, 0, cons) == Failure(
        "structural_constraint", "coplanarity deviation (0, 1, 2, 3): undefined deg; expected [0, 12] deg"
    )


@pytest.mark.parametrize(
    ("cons", "detail"),
    [
        (
            Constraints(metals={0}, distances={(0, 2): (2.0, 2.1)}),
            "M-L distance (0, 2): 1.414 (excess 0.576) A; "
            f"expected [{2.0 - emb._ML_WINDOW_TOL:g}, {2.1 + emb._ML_WINDOW_TOL:g}] A",
        ),
        (
            Constraints(coplanar=[(0, 1, 2, 3, None, 10.0)]),
            "coplanarity deviation (0, 1, 2, 3): 45.000 (excess 33) deg; expected [0, 12] deg",
        ),
        (
            Constraints(umbrellas={(0, 1, 2, 3): 60.0}),
            "absolute umbrella dihedral (0, 1, 2, 3): 45.000 (excess 13) deg; expected [58, 92] deg",
        ),
    ],
)
def test_geometry_failure_names_the_structural_measurement(cons, detail):
    mol = Chem.MolFromSmiles("[He].[He].[He].[He]")
    conf = Chem.Conformer(4)
    conf.SetPositions(np.array(((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)), dtype=float))
    mol.AddConformer(conf)
    conformers = Conformers(mol, [0], cons)

    assert conformers._geometry_failure(0) == Failure("structural_constraint", detail)


def test_structural_failure_reports_excess_below_display_precision():
    mol = Chem.MolFromSmiles("[He].[He]")
    conf = Chem.Conformer(2)
    conf.SetPositions(np.array(((0, 0, 0), (2.1 + emb._ML_WINDOW_TOL + 0.0004, 0, 0))))
    mol.AddConformer(conf)
    cons = Constraints(metals={0}, distances={(0, 1): (2.0, 2.1)})

    failure = emb._structural_failure(mol, 0, cons)
    assert failure.kind == "structural_constraint"
    assert "excess 0.0004" in failure.detail


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
        "structural_constraint",
        f"M-L distance (0, 1): 2.106 (excess {0.006 - FIX_DISTANCE_TOL:.3g}) A; "
        f"expected [{2.0 - FIX_DISTANCE_TOL:g}, {2.1 + FIX_DISTANCE_TOL:g}] A",
    )


def test_donor_orientation_seed_wall_is_not_a_structural_postcondition():
    mol = Chem.MolFromSmiles("[Li].N=C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (1.173648, 0.984808, 0))):
        conf.SetAtomPosition(atom, xyz)
    mol.AddConformer(conf)
    cons = Constraints(angles={(0, 1, 2): (108.0, 180.0)}, metals={0})

    assert emb._structural_failure(mol, 0, cons) is None


def test_donor_angle_failure_preserves_the_measurement_without_inferring_folding(monkeypatch):
    confs = embed(_isomer(), n=1, seed=1, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    detail = "C1 (sp3) M-D-X to Si2: 99.5° < census floor 104°; inspect donor geometry and restraints"
    monkeypatch.setattr(emb, "donor_orientation", lambda *args: [SimpleNamespace(detail=detail)])

    failure = confs._geometry_failure(cid)
    assert failure == Failure("metal_state", f"donor orientation: {detail}")


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
    confs = embed(_mol("CCOCC"), n=6, seed=7, prune_rms=-1)
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
    assert internal.GetAtomicNum() not in TRANSITION_METALS, "the working mol must still hold the surrogate"
    assert internal.GetDegree() == 0, "the surrogate must stay bond-less: a bonded Li makes UFF singular"

    restored = confs.mol.GetAtomWithIdx(iso.metal)
    assert restored.GetAtomicNum() in TRANSITION_METALS
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


# ---------------------------------------------------------------------------------------------------------
# encounter bounds: a probe geometry decides a discrete question, so it must not read process history
# ---------------------------------------------------------------------------------------------------------


def _two_fragments():
    return _mol("OC(=O)CCCc1ccccc1.NCCCCN")


def _burn_global_rng(n=64):
    """Consume RDKit global randomness, standing in for whatever ran before us in a real session."""
    for _ in range(n):
        m = _mol("CCCCO")
        rdDistGeom.EmbedMolecule(m, rdDistGeom.ETKDGv3())  # deliberately unseeded


def test_encounter_bounds_ignore_global_rng():
    mol = _two_fragments()
    assert len(Chem.GetMolFrags(mol)) >= 2, "fixture must be multi-fragment to exercise the encounter bounds"
    before = emb.encounter_bounds(mol)
    assert before, "no inter-fragment bound was produced: the fixture is not exercising the code"
    _burn_global_rng()
    assert emb.encounter_bounds(mol) == before


def test_probe_seed_is_forwarded(monkeypatch):
    seen = []
    real = bnd.probe_conformer
    monkeypatch.setattr(emb, "probe_conformer", lambda m, s: (seen.append(s), real(m, s))[1])
    emb.encounter_bounds(_two_fragments(), seed=4321)
    assert seen == [4321]


def test_float_bounds_apply_only_without_pins():
    mol = _two_fragments()
    assert emb.float_encounter_bounds(mol, Constraints()), "every pair of a free multi-fragment mol must be bounded"

    i, j = sorted(f[0] for f in Chem.GetMolFrags(mol))
    assert emb.float_encounter_bounds(mol, Constraints(distances={(i, j): (3.0, 3.5)})) == {}
    assert emb.float_encounter_bounds(_mol("CCO"), Constraints()) == {}


def test_encounter_bounds_treat_distance_linked_ligands_as_one_component():
    mol = Chem.MolFromSmiles("C.N.O.F")
    bounds = emb.float_encounter_bounds(mol, Constraints(distances={(0, 1): (2.0, 2.1), (0, 2): (2.0, 2.1)}))

    assert len(bounds) == 1
    assert 3 in next(iter(bounds))


def test_encounter_bounds_do_not_move_two_separately_fixed_fragments():
    mol = Chem.MolFromSmiles("C.N")

    assert emb.float_encounter_bounds(mol, Constraints(frozen={0, 1})) == {}


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

    assert confs.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() in TRANSITION_METALS
    assert len(Chem.GetMolFrags(confs.mol)) == 1, "the M-donor bonds are re-added, so the complex is one fragment"
    assert confs._mol is not iso.mol, "the relax works on a copy: the caller keeps its isomer"
    assert np.allclose(iso.mol.GetConformer().GetPositions(), before)


@pytest.mark.parametrize("lengths", ["model", "input"])
def test_minimize_uses_model_distances_unless_input_requested(lengths):
    mol = _with_geometry("Cl[Pd](Cl)(N)N")
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS)
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


def test_stiffness_ladder_retries_only_the_unresolved_conformer(monkeypatch):
    confs = embed(_isomer(_NI_N), n=2, seed=1, threads=1, prune_rms=-1)
    victim = int(confs.ids[0])
    calls = []
    constraints = []
    latest = {}

    def marked_uff(_mol, _cons, *, stiffness, max_iters, conf_ids, _statuses=None, **_kw):
        ids = tuple(conf_ids)
        calls.append((ids, stiffness, max_iters))
        constraints.append(_cons)
        latest.update(dict.fromkeys(ids, stiffness))
        if _statuses is not None:
            _statuses.update(dict.fromkeys(ids, 0))
        return np.zeros(len(ids))

    def injected_failure(_self, cid, _iso=None):
        return "heavy-atom bonding/clash failure" if cid == victim and latest[cid] == BASE_STIFFNESS else None

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(Conformers, "_geometry_failure", injected_failure)
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls[:2] == [
        (tuple(confs.ids), BASE_STIFFNESS, 10),
        ((victim,), 3 * BASE_STIFFNESS, 10),
    ]
    assert calls[-1] == (tuple(confs.ids), BASE_STIFFNESS, 0)
    assert all(active is confs.cons for active in constraints)
    assert victim not in confs.unrelaxed


def test_relax_ladder_does_not_escalate_a_converged_wrong_coordination_shape(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(), seed=1)
    calls = []

    def marked_uff(_mol, _cons, *, max_iters, conf_ids, _statuses=None, **_kw):
        calls.append(max_iters)
        if _statuses is not None:
            _statuses.update(dict.fromkeys(conf_ids, 0))
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda *_args: Failure("metal_state", "coordination state at M0: expected SPY, found TBP"),
    )

    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [10, 0]
    assert confs.unrelaxed == [0]
    assert confs.relax_failures[0].kind == "metal_state"


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
        state = _metal.state_with_winding(iso.centres[0], iso.vertices, {})
        confs.iso = iso._with_stereo((state,))
        assert not emb._stereo_targets(confs.iso)
    attempts, faces = [], []

    def flip_then_recover(mol, _cons, *, stiffness, max_iters, conf_ids, _statuses=None, _snapshots=None, **_kw):
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
            assert _statuses is not None
            assert _snapshots is not None
            _statuses[cid] = 0
            _snapshots[cid] = [pos.copy()]
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

    def stalled_uff(mol, _cons, *, max_iters, conf_ids, _statuses=None, **_kw):
        calls.append(max_iters)
        if max_iters:
            assert _statuses is not None
            moved = mol.GetConformer(0).GetPositions().copy()
            moved[0, 0] += 0.2
            mol.GetConformer(0).SetPositions(moved)
            _statuses[0] = 1
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", stalled_uff)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10)

    assert calls == [10, 0]
    assert confs.unrelaxed == [0]
    assert not np.array_equal(mol.GetConformer().GetPositions(), before)


def test_relax_ladder_corrects_free_metal_mirrors_before_geometry_acceptance(monkeypatch):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    confs = embed(iso, n=1, seed=1, threads=1).minimize()
    cid = confs.ids[0]
    assert confs._geometry_failure(cid) is None
    seed = confs._mol.GetConformer(cid).GetPositions().copy()
    attempts = []

    def reflect(mol, _cons, *, max_iters, conf_ids, _statuses=None, _snapshots=None, **_kw):
        if max_iters:
            attempts.append(max_iters)
            emb._reflect(mol, cid)
            if _snapshots is not None:
                _snapshots[cid] = [mol.GetConformer(cid).GetPositions().copy()]
            assert _statuses is not None
            _statuses[cid] = 0
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

    def reflect(mol, _cons, *, conf_ids, max_iters, _statuses=None, **_kw):
        assert conf_ids == [cid]
        if max_iters:
            emb._reflect(mol, cid)
            assert _statuses is not None
            _statuses[cid] = 0
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

    def fail_second_attempt(mol, _cons, *, conf_ids, _statuses, _snapshots, **_kw):
        calls.append(list(conf_ids))
        if len(calls) > 1:
            raise UFFTypingError("unavailable")
        for cid in conf_ids:
            mol.GetConformer(cid).SetPositions(endpoint)
            _statuses[cid] = 0
            _snapshots[cid] = [endpoint.copy()]
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
    confs = embed(_isomer(_NI_N), n=20, seed=42, threads=1, prune_rms=-1)
    raw = {cid: confs._mol.GetConformer(cid).GetPositions().copy() for cid in confs.ids}
    statuses = {}
    restrained_uff(confs._mol, confs.cons, conf_ids=confs.ids, _statuses=statuses)

    assert len(confs.ids) == 20
    assert statuses == dict.fromkeys(confs.ids, 0)
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
        assert bonding_ok(confs.mol, int(cid), constrained=confs.cons.distances), f"conformer {cid} came back torn"


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
    energies = confs._relax_constrained(BASE_STIFFNESS)

    assert calls[-1][0] == 0
    assert np.array_equal(energies, calls[-1][1])


@pytest.mark.parametrize("operation", ["embed", "minimize"])
def test_rejected_uff_endpoint_reason_survives_seed_restoration(monkeypatch, caplog, operation):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (2.0, 3.0)}))
    seed = mol.GetConformer().GetPositions().copy()
    assert confs._geometry_failure(0) is None

    def tear_bond(mol, _cons, *, conf_ids, max_iters, _statuses=None, **_kw):
        if max_iters:
            assert _statuses is not None
            for cid in conf_ids:
                positions = mol.GetConformer(cid).GetPositions()
                positions[0, 0] += 100.0
                mol.GetConformer(cid).SetPositions(positions)
                _statuses[cid] = 0  # numerical convergence does not guarantee a valid molecule
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", tear_bond)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs._relax_constrained(BASE_STIFFNESS, operation=operation)

    assert np.array_equal(mol.GetConformer().GetPositions(), seed)
    assert confs._geometry_failure(0) is None
    assert confs.unrelaxed == [0]
    assert f"{operation}: rejected 1/1 UFF endpoints" in caplog.text
    assert "heavy-atom bonding/clash failure" in caplog.text
    assert "restored seeds" in caplog.text


def test_relax_retry_rejects_an_inverted_donor_hand(monkeypatch, caplog):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    seed_pos = {cid: confs._mol.GetConformer(cid).GetPositions().copy()}
    references = [iso.metal]
    hand = _metal.donor_chirality_sign(confs._mol, cid, donor, references)
    inverted = []
    confs.unrelaxed = [cid]

    def reflect(mol, _cons, *, conf_ids, max_iters, _statuses=None, **_kw):
        if max_iters == 0:
            return np.zeros(len(conf_ids))
        for conf_id in conf_ids:
            positions = mol.GetConformer(conf_id).GetPositions()
            positions[:, 0] *= -1.0
            mol.GetConformer(conf_id).SetPositions(positions)
            assert _statuses is not None
            _statuses[conf_id] = 0
            inverted.append(_metal.donor_chirality_sign(mol, conf_id, donor, references))
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)
    with caplog.at_level(logging.INFO, logger="rxembed"):
        confs._relax_constrained(BASE_STIFFNESS, 10)

    assert inverted
    assert set(inverted) != {hand}
    assert confs.unrelaxed == [cid]
    assert np.array_equal(confs._mol.GetConformer(cid).GetPositions(), seed_pos[cid])
    assert "coordinated ligand hand changed" in caplog.text


def test_relax_retry_can_recover_a_donor_hand_at_a_stronger_rung(monkeypatch):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    references = [iso.metal]
    wanted = _metal.donor_chirality_sign(confs._mol, cid, donor, references)
    attempts = []

    def flip_once(mol, _cons, *, stiffness, max_iters, conf_ids, _statuses=None, **_kw):
        attempts.append(stiffness)
        for conf_id in conf_ids:
            if max_iters and stiffness == BASE_STIFFNESS:
                positions = mol.GetConformer(conf_id).GetPositions()
                positions[:, 0] *= -1.0
                mol.GetConformer(conf_id).SetPositions(positions)
            if _statuses is not None:
                _statuses[conf_id] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", flip_once)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda _self, _cid, _iso=None: None)

    confs._relax_constrained(BASE_STIFFNESS, 10)

    assert attempts[:2] == [BASE_STIFFNESS, 3 * BASE_STIFFNESS]
    assert confs.unrelaxed == []
    assert _metal.donor_chirality_sign(confs._mol, cid, donor, references) == wanted


def test_relax_failure_keeps_the_first_physical_reason_when_donor_hand_also_changes(monkeypatch):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    references = [iso.metal]
    hand = _metal.donor_chirality_sign(confs._mol, cid, donor, references)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda _self, _cid, _iso=None: Failure("bonding", "heavy-atom bonding/clash failure"),
    )
    monkeypatch.setattr(
        emb._metal,
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

    def marked_uff(mol, cons, *, stiffness, max_iters, conf_ids=None, _snapshots=None, _statuses=None, **_kw):
        if max_iters == 0:
            return np.array([0.0])
        attempts.append(stiffness)
        cid = int(conf_ids[0]) if conf_ids else 0
        if _statuses is not None:
            _statuses[cid] = 0
        positions = mol.GetConformer(cid).GetPositions().copy()
        positions[0, 0] = len(attempts)
        for atom, xyz in enumerate(positions):
            mol.GetConformer(cid).SetAtomPosition(atom, xyz.tolist())
        if _snapshots is not None:
            _snapshots[cid] = [positions.copy()]
        return np.array([float(stiffness)])

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda _self, _cid, _iso=None: "injected failure" if len(attempts) < 2 else None,
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
    with pytest.raises(ValueError, match="the requested metal state"):
        minimize(_mirrored_input(_CO_EN_ME))


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

    with pytest.raises(ValueError, match="the requested metal state"):
        confs.minimize()


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

    with pytest.raises(ValueError, match="could not satisfy the requested metal state"):
        confs.minimize()


def test_coordination_gate_accepts_a_graph_equivalent_site_swap():
    iso = enumerate_isomers(_mol("[Pt](F)(F)(Cl)Br"), "square_planar", stereo="free")[0]
    conformers = embed(iso, n=1, seed=7).minimize()
    left, right = [donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetSymbol() == "F"]
    positions = conformers._mol.GetConformer().GetPositions().copy()
    positions[[left, right]] = positions[[right, left]]
    conformers._mol.GetConformer().SetPositions(positions)

    assert conformers._coordination_ok(conformers.ids[0])


def test_handed_coordination_gate_rejects_an_improper_equivalent_reseating():
    iso = enumerate_isomers(_mol("[Co](N)(N)(P)(P)(O)(O)"), "octahedral", stereo="free")[0]
    donors = {
        atomic_number: [atom for atom in iso.donors if iso.mol.GetAtomWithIdx(atom).GetAtomicNum() == atomic_number]
        for atomic_number in (7, 8, 15)
    }
    n, o, p = donors[7], donors[8], donors[15]
    state = iso.centres[0]._replace(vertices=(o[0], n[0], n[1], p[0], p[1], o[1]), hand="delta")
    iso = iso._with_stereo((state,))
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

    achiral = iso._with_stereo((state._replace(hand=""),))
    assert emb._coordination_state_ok(achiral.mol, 0, achiral), "the distorted octahedron itself is valid"
    targets = emb._stereo_targets(iso)
    ranks, eta2 = emb._winding_ranks(iso.mol, targets)
    assert emb._seed_stereo_matches(iso.mol, 0, iso, targets, ranks, eta2, False)
    assert not emb._coordination_state_ok(iso.mol, 0, iso), "an improper donor swap is the other hand"


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
    assert conformers._coordination_ok(conformers.ids[0])

    positions = conf.GetPositions().copy()
    positions[[fluorines[0], chlorine]] = positions[[chlorine, fluorines[0]]]
    conf.SetPositions(positions)
    assert not conformers._coordination_ok(conformers.ids[0])


def test_independent_ligand_ez_is_restored_when_relaxation_changes_it():
    isomer = enumerate_isomers(_mol(r"C/N1=C(/C)CCCCCC[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-1"), "square_planar")[0]
    confs = embed(isomer, n=1, seed=7).minimize()
    realised = _stereo.bond_stereo(_stereo.stereo_from_3d(confs.mol, exclude=_metal.metal_indices(confs.mol)))

    assert len(confs) == 1
    assert realised == _stereo.bond_stereo(isomer.stereo_label)


def test_metal_referenced_imine_ez_is_selected_before_relaxation():
    isomers = enumerate_isomers(_mol("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]"), "square_planar")

    for isomer in isomers:
        conformers = embed(isomer, n=3, seed=42)
        assert len(conformers) == 3
        wanted = _stereo.bond_stereo(isomer.stereo_label)
        published = conformers.mol
        for cid in conformers.ids:
            one = Chem.Mol(published, False, int(cid))
            realised = _stereo.bond_stereo(_stereo.stereo_from_3d(one, exclude=_metal.metal_indices(one)))
            assert realised == wanted


def test_stereo_selection_streams_one_bounded_seed_budget(monkeypatch, caplog):
    isomer = enumerate_isomers(_mol("CC=[NH]->[Pt](Cl)(Br)I"), "tetrahedral")[0]
    seen = []

    def reject_all(_mol, _cons, count, *, seed, **_kwargs):
        seen.append((count, seed))
        return []

    monkeypatch.setattr(emb, "seed_count", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(emb, "seed_coordinates", reject_all)

    with pytest.raises(RuntimeError, match="bounded DG seed budget"):
        embed(isomer, n=1, seed=42)
    assert seen == [(4, 42), (4, 43), (2, 44)]
    assert "DG search returned 0 candidates; 0/1 satisfied requested stereo selection" in caplog.text


def test_relax_gate_rejects_inverted_ligand_point_stereo(monkeypatch):
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    conformers = embed(isomer, n=1, seed=7)
    atom, expected = next(iter(_stereo.point_stereo(isomer.stereo_label).items()))
    found = {"R": "S", "S": "R"}[expected]
    symbol = isomer.mol.GetAtomWithIdx(atom).GetSymbol()

    def inverted(published, **_kwargs):
        assert published.GetBondBetweenAtoms(*isomer.donor_bonds[0]).GetBondType() == Chem.BondType.DATIVE
        return f"{symbol}{atom}:{found}"

    monkeypatch.setattr(_stereo, "stereo_from_3d", inverted)
    monkeypatch.setattr(emb, "_coordination_state_failure", lambda *_args, **_kwargs: None)

    assert str(conformers._geometry_failure(conformers.ids[0])) == (
        f"wrong ligand point stereo at atom {atom}: expected {expected}, found {found}"
    )


def test_relax_gate_distinguishes_unassigned_ligand_stereo(monkeypatch):
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    monkeypatch.setattr(_stereo, "stereo_from_3d", lambda *_args, **_kwargs: "")

    failure = emb._ligand_stereo_failure(isomer.mol, isomer)

    assert failure.kind == "ligand_stereo_unassigned"
    assert Conformers(_with_geometry("CC"), [0], Constraints())._required_failure({failure: [0]}) == (
        "requested ligand stereo",
    )


def test_relax_gate_rejects_reflected_ligand_axial_stereo():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
    wanted = _stereo.stereo_from_3d(mol)
    request = SimpleNamespace(stereo_label=wanted)
    pair, expected = next(iter(_stereo.axis_stereo(wanted).items()))
    positions = mol.GetConformer().GetPositions()
    positions[:, 0] *= -1
    mol.GetConformer().SetPositions(positions)
    found = _stereo.axis_stereo(_stereo.stereo_from_3d(mol))[pair]

    failure = emb._ligand_stereo_failure(mol, request)
    assert str(failure) == f"wrong ligand axial stereo at bond {pair}: expected {expected}, found {found}"
    assert Conformers(mol, [0], Constraints())._required_failure({failure: [0]}) == ("requested ligand stereo",)


def test_required_failure_policy_uses_kind_not_message_wording():
    conformers = Conformers(_with_geometry("CC"), [0], Constraints())
    assert conformers._required_failure({Failure("ligand_stereo", "reworded completely"): [0]}) == (
        "requested ligand stereo",
    )
    assert conformers._required_failure({Failure("bonding", "wrong ligand E/Z at bond 1-2"): [0]}) == ()


@pytest.mark.parametrize("operation", ["embed", "minimize"])
@pytest.mark.parametrize("survivor", [False, True])
def test_total_embed_rejection_reports_original_bond_failure(operation, survivor):
    mol = _with_geometry("CC")
    if survivor:
        mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    SetBondLength(mol.GetConformer(0), 0, 1, 10.0)
    conformers = Conformers(mol, [conf.GetId() for conf in mol.GetConformers()])

    if operation == "embed" and not survivor:
        with pytest.raises(emb.EmbeddingError, match=r"heavy-atom bonding/clash failure.*publication gate"):
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
    owner = Conformers(Chem.Mol(target), [*failed, 27], seed=42, energies={27: -77.0})
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
        return {"invalid lowest energy": [0]}

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
    confs = embed(_mol("CCC"), n=2, seed=1, prune_rms=-1)
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
    confs = embed(_isomer(), n=1, seed=1, threads=3, knowledge=False, prune_rms=-1)
    from rdkit.Chem import rdDistGeom

    confs.embed_params = rdDistGeom.KDG()
    confs.coplanar_14 = False
    confs.metal_floor_relief = False
    victim = confs.ids[0]
    calls = []

    def seeds(source, _cons, _iso, n, *, seed, threads, knowledge=True, prune_rms=0.1, **_kwargs):
        assert _kwargs["embed_params"] is confs.embed_params
        assert not _kwargs["coplanar_14"]
        assert not _kwargs["metal_floor_relief"]
        calls.append((n, seed, threads, knowledge, prune_rms))
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
        return {"missed structural constraint": list(self.ids)} if attempts < success_on else {}

    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)

    expected = [] if success_on < 4 else [victim]
    assert confs._replace_failed([victim], BASE_STIFFNESS, 1) == expected
    assert calls == [(4, seed, 3, False, -1) for seed in range(2, 2 + min(success_on, 3))]
    assert len(constrained_relaxations) == min(success_on, 3)
    child = confs[0]
    assert child.embed_params is confs.embed_params
    assert not child.coplanar_14
    assert not child.metal_floor_relief
    assert (child.seed, child.threads, child.knowledge, child.prune_rms) == (1, 3, False, -1)
    assert child._mol is confs._mol
    assert child.unrelaxed is not confs.unrelaxed
    assert child.uff_surrogates is not confs.uff_surrogates


def test_requested_metal_hand_must_fill_the_seed_count(monkeypatch, caplog):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match=r"found 0/1 seeds with the requested stereo state") as caught:
        embed(iso, n=1, seed=1)
    assert str(iso) in str(caught.value)
    summary = next(record for record in caplog.records if record.msg.startswith("DG search returned"))
    assert summary.args[0] > 0
    assert "0/1 satisfied requested stereo selection" in summary.message


def test_ligand_hard_chirality_falls_back_when_it_blocks_the_metal_hand(monkeypatch):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    mol, cons, prepared, graft = emb.prepare(iso)
    calls = []

    def seeds(candidate, _cons, n, *, enforce_chirality=True, **_kwargs):
        calls.append((n, enforce_chirality, _kwargs["max_attempts"]))
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "seed_count", lambda *_args, **_kwargs: 8)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: not calls[-1][1])

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, seed=1, graft_ref=graft)

    assert calls == [(4, True, 30), (4, False, 30)]
    assert len(ids) == target == 1


@pytest.mark.parametrize("n", [1, 8])
def test_stereo_seed_selection_prefers_a_seed_with_the_requested_geometry(monkeypatch, n):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    mol, cons, prepared, graft = emb.prepare(iso)

    def seeds(candidate, _cons, n, **_kwargs):
        candidate.RemoveAllConformers()
        for marker in range(n):
            conf = Chem.Conformer(candidate.GetNumAtoms())
            conf.SetAtomPosition(0, (float(marker), 0.0, 0.0))
            candidate.AddConformer(conf, assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(emb, "_seed_geometry_matches", lambda _mol, cid, _iso: cid == 2)

    seeded, ids, target = emb.seed_conformers(mol, cons, prepared, n, seed=1, graft_ref=graft)

    assert len(ids) == target == n
    assert len({seeded.GetConformer(cid).GetAtomPosition(0).x for cid in ids}) == n
    assert seeded.GetConformer(ids[0]).GetAtomPosition(0).x == 2.0


@pytest.mark.parametrize("kind", ["ez", "axial"])
def test_nonpoint_ligand_stereo_does_not_enable_the_point_fallback(monkeypatch, kind):
    iso = _isomer("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", "square_planar")
    mol, cons, prepared, graft = emb.prepare(iso)
    calls = []
    if kind == "axial":
        monkeypatch.setattr(_stereo, "bond_stereo", lambda _label: {})
        monkeypatch.setattr(_stereo, "axis_stereo", lambda _label: {(0, 1): "Ra"})
    else:
        monkeypatch.setattr(_stereo, "axis_stereo", lambda _label: {})

    def seeds(candidate, _cons, n, *, enforce_chirality=True, **kwargs):
        calls.append((enforce_chirality, kwargs["max_attempts"]))
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, seed=1, graft_ref=graft)

    assert calls == [(True, 0)]
    assert len(ids) == target == 1


def test_stereo_seed_budget_scales_with_independent_metal_states(monkeypatch):
    iso = _isomer()
    mol, cons, prepared, graft = emb.prepare(iso)
    state = SimpleNamespace(hand="delta")
    targets = [(state, (), {}, {}), (state, (), {}, {})]
    calls, inspected = [], 0

    def seeds(candidate, _cons, n, **_kwargs):
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

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, seed=1, graft_ref=graft)

    assert calls == [4, 4, 4]
    assert inspected == 12
    assert len(ids) == target == 1


def test_free_chirality_seed_selection_still_rejects_the_ligand_mirror():
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    seeded = embed(iso, n=1, seed=5)
    cid = seeded.ids[0]

    assert emb._seed_ligand_stereo_matches(seeded._mol, cid, iso)
    emb._reflect(seeded._mol, cid)
    assert not emb._seed_ligand_stereo_matches(seeded._mol, cid, iso)


def test_unconstrained_max_iteration_replaces_invalid_geometry_and_keeps_status(monkeypatch):
    confs = embed(_mol("CCC"), n=1, seed=1)
    owner = confs._mol

    def stall(mol, _cons, **kwargs):
        ids = kwargs.get("conf_ids") or [conf.GetId() for conf in mol.GetConformers()]
        if mol is owner:
            mol.GetConformer(ids[0]).SetAtomPosition(0, (99.0, 99.0, 99.0))
        statuses = kwargs.get("_statuses")
        if statuses is not None:
            statuses.update(dict.fromkeys(ids, 1))
        return [0.0] * len(ids)

    monkeypatch.setattr(emb, "restrained_uff", stall)
    confs.minimize()

    assert confs.ids
    assert confs.unrelaxed == confs.ids
    assert all(confs._relax_ok(cid) for cid in confs.ids)


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
