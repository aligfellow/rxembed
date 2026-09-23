"""Test RDKit bounds editing, smoothing and coordinate seeding."""

from __future__ import annotations

import itertools
import json
import math
import random

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import bounds as bnd
from rxembed import mechanisms as mech
from rxembed import metal_core
from rxembed.constraints import FIX_DISTANCE_TOL, Constraints, add_distance


def _graph(smiles):
    """Return a Mol with explicit Hs and no conformer for seed-count tests."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _mol(smiles="CCO", seed=1):
    mol = _graph(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _matrix(mol):
    return rdDistGeom.GetMoleculeBoundsMatrix(mol)


# ---------------------------------------------------------------------------------------------------------
# embed_parameters: native model and reproducible sampling defaults
# ---------------------------------------------------------------------------------------------------------


def test_default_parameters_select_native_kdg_and_aio():
    on = bnd.embed_parameters(11)
    assert on.randomSeed == 11
    with pytest.raises(TypeError):
        bnd.embed_parameters()  # ty: ignore[missing-argument]

    assert (on.useExpTorsionAnglePrefs, on.useBasicKnowledge) == (False, True)
    assert not on.useLegacyImplementation
    assert not on.useSmallRingTorsions
    assert not on.useMacrocycleTorsions
    assert not on.useMacrocycle14config
    assert on.ETversion == rdDistGeom.KDG().ETversion
    off = bnd.embed_parameters(11, knowledge=False)
    assert (off.useExpTorsionAnglePrefs, off.useBasicKnowledge) == (False, False)
    assert not off.useSmallRingTorsions
    assert not off.useLegacyImplementation

    assert on.pruneRmsThresh == -1.0
    assert bnd.embed_parameters(11, prune_rms=0.0).pruneRmsThresh == 0.0
    assert bnd.embed_parameters(11, prune_rms=0.5).pruneRmsThresh == 0.5


@pytest.mark.parametrize(
    ("smiles", "force_trans"),
    [("O=C1NCCCCCCC1", True), ("O=C1OCCCCCCC1", True), ("CC(=O)NCC", False), ("CCCC", True)],
    ids=["lactam", "lactone", "free-amide", "alkane-control"],
)
def test_constrained_seed_uses_its_native_bounds_parameters(monkeypatch, smiles, force_trans):
    mol = _graph(smiles)
    params = bnd.embed_parameters(42, threads=1)
    params.forceTransAmides = force_trans
    native_bounds, native_embed = rdDistGeom.GetMoleculeBoundsMatrix, rdDistGeom.EmbedMultipleConfs
    expected = native_bounds(mol, embedParams=params)
    built = []

    def build(candidate, *args, **kwargs):
        matrix = native_bounds(candidate, *args, **kwargs)
        built.append(matrix.copy())
        return matrix

    def embed(candidate, count, used_params):
        assert used_params is params
        np.testing.assert_array_equal(built[-1], expected)
        return native_embed(candidate, count, used_params)

    monkeypatch.setattr(rdDistGeom, "GetMoleculeBoundsMatrix", build)
    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", embed)
    cons = Constraints(distances={(0, 1): (expected[1, 0], expected[0, 1])})
    assert bnd.seed_coordinates(mol, cons, n=1, seed=42, threads=1, embed_params=params)


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_native_parameters_reach_rdkit_without_model_fallback_or_stale_bounds(monkeypatch, reject, legacy):
    params = rdDistGeom.srETKDGv3()
    params.useLegacyImplementation = legacy
    params.useRandomCoords = True
    params.enforceChirality = False
    params.maxIterations = 17
    params.randomSeed, params.numThreads, params.pruneRmsThresh = 7, 2, 0.4
    params.SetCPCI({(0, 1): 0.01})
    before = json.loads(rdDistGeom.EmbedParametersToJSON(params))
    native, set_bounds = rdDistGeom.EmbedMultipleConfs, rdDistGeom.EmbedParameters.SetBoundsMat
    matrices, calls = [], []

    def handoff(used, matrix):
        assert used is params
        matrices.append(matrix.copy())
        return set_bounds(used, matrix)

    def embed(mol, n, used):
        assert used is params
        assert used.useLegacyImplementation == legacy
        assert used.useRandomCoords
        assert not used.enforceChirality
        assert used.maxIterations == 17
        assert used.useBasicKnowledge
        assert used.useExpTorsionAnglePrefs
        assert used.useSmallRingTorsions
        assert not used.useMacrocycleTorsions
        assert not used.useMacrocycle14config
        assert (used.randomSeed, used.numThreads, used.pruneRmsThresh) == (42, 1, -1)
        assert not used.embedFragmentsSeparately
        assert matrices[-1].shape == (mol.GetNumAtoms(),) * 2
        calls.append(mol.GetNumAtoms())
        return [] if reject else native(mol, n, used)

    monkeypatch.setattr(rdDistGeom.EmbedParameters, "SetBoundsMat", handoff)
    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", embed)
    for smiles in ("CCCC", "CC.CC", "CC"):
        mol = _graph(smiles)
        matrix = rdDistGeom.GetMoleculeBoundsMatrix(mol, embedParams=params)
        cons = Constraints(distances={(0, 1): (matrix[1, 0], matrix[0, 1])}) if smiles == "CCCC" else Constraints()
        ids = bnd.seed_coordinates(mol, cons, 1, seed=42, threads=1, prune_rms=-1, embed_params=params)
        assert bool(ids) != reject
        after = json.loads(rdDistGeom.EmbedParametersToJSON(params))
        assert after.pop("boundsMatrix")
        assert after == before
    assert len(calls) == len(matrices) == 3


def test_untracked_native_parameters_do_not_report_stale_failure_counts(monkeypatch, caplog):
    params = rdDistGeom.KDG()

    def stale(_params):
        raise AssertionError("untracked failure counts belong to an earlier native call")

    monkeypatch.setattr(rdDistGeom.EmbedParameters, "GetFailureCounts", stale)
    with caplog.at_level("DEBUG", logger="rxembed.bounds"):
        assert bnd.seed_coordinates(_graph("CC"), Constraints(), 1, embed_params=params)
    assert "rejected attempts not tracked" in caplog.text


def test_native_timeout_is_not_a_conformer_id_or_an_implicit_retry(monkeypatch):
    params = rdDistGeom.KDG()
    params.timeout = 1
    before = params.randomSeed, params.numThreads, params.clearConfs
    calls = []

    def timeout(_mol, _n, used):
        calls.append(used)
        return [-1]  # RDKit's timeout sentinel, not an attached conformer.

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", timeout)
    with pytest.raises(TimeoutError, match=r"native RDKit.*timeout=1s.*embed_params\.timeout"):
        bnd.seed_coordinates(_graph("CC"), Constraints(), 1, seed=42, threads=2, embed_params=params)
    assert calls == [params]
    assert (params.randomSeed, params.numThreads, params.clearConfs) == before


def test_native_parameter_keywords_have_one_precedence_rule():
    params = rdDistGeom.KDG()
    assert bnd.embedding_options(None, None, None, embed_params=params) == (bnd.DEFAULT_SEED, 1, True, -1)
    params.randomSeed, params.numThreads, params.pruneRmsThresh = 7, 3, 0.25
    assert bnd.embedding_options(None, None, None, embed_params=params) == (7, 3, True, 0.25)
    assert bnd.embedding_options(42, 1, None, -1, params) == (42, 1, True, -1)
    for knowledge in (False, True):
        with pytest.raises(ValueError, match="useBasicKnowledge"):
            bnd.embedding_options(None, None, knowledge, embed_params=params)
    with pytest.raises(TypeError, match="EmbedParameters"):
        bnd.embedding_options(None, None, None, embed_params={})


def test_aio_refines_the_edited_interfragment_distance():
    params = rdDistGeom.srETKDGv3()
    params.useLegacyImplementation = False
    for distance in (3.0, 6.0):
        mol = _graph("CC.CC")
        cons = Constraints(distances={(0, 2): (distance, distance + 0.05)})
        ids = bnd.seed_coordinates(mol, cons, 1, seed=42, threads=1, embed_params=params)
        assert ids
        positions = mol.GetConformer(ids[0]).GetPositions()
        assert distance - 0.1 < np.linalg.norm(positions[0] - positions[2]) < distance + 0.15


def test_existing_coordinates_do_not_replace_native_embedding(monkeypatch):
    mol = _graph("CC")
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = np.arange(mol.GetNumAtoms() * 3, dtype=float).reshape(mol.GetNumAtoms(), 3)
    conf.SetPositions(positions)
    mol.AddConformer(conf)
    calls = []
    native = bnd.rdDistGeom.EmbedMultipleConfs

    def generate(*args):
        calls.append(args[1])
        return native(*args)

    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMultipleConfs", generate)

    ids = bnd.seed_coordinates(mol, Constraints(), 1, seed=42)

    assert len(ids) == 1
    assert calls == [1]
    assert not np.allclose(mol.GetConformer(ids[0]).GetPositions(), positions)


# ---------------------------------------------------------------------------------------------------------
# probe_conformer: a throwaway geometry that decides a discrete question
#
# That its seed decides the answer, rather than process history, is asserted end to end on the one caller
# that consumes it: `tests/test_embed.py`'s encounter-bounds pair.
# ---------------------------------------------------------------------------------------------------------


def test_failed_probe_returns_none_rather_than_an_empty_mol(monkeypatch):
    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMolecule", lambda *_a, **_k: -1)
    assert bnd.probe_conformer(_mol(), 7) is None


def test_probe_geometry_ignores_python_and_numpy_random_state():
    """A probe with no explicit seed of its own reads RDKit's global RNG, so a structure that passed
    alone could fail inside a full suite carrying different process history and never reproduce.
    """
    mol = _graph("CCCO")  # propanol-sized

    random.seed(1)
    np.random.seed(1)
    first = bnd.probe_conformer(mol, 7)

    random.seed(99999)
    np.random.seed(99999)
    second = bnd.probe_conformer(mol, 7)

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first.GetConformer().GetPositions(), second.GetConformer().GetPositions())


def test_ligand_reach_retains_free_rotation_without_discarding_chain_geometry():
    mol = _graph("CCCN")
    before = rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=False)
    reach = bnd.ligand_reach(mol)
    assert reach[0, 3] < before[0, 3] - 0.05
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    np.testing.assert_allclose(bnd.ligand_reach(reversed_mol), reach[::-1, ::-1].T, atol=1e-10)

    # Native 1-4 construction prefers a 90-degree disulfide torsion. A free-torsion reach screen must also
    # admit an anti chain with the same native bond lengths and valence angles, irrespective of its energy.
    mol = _graph("CSSC")
    basis = rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=False, doTriangleSmoothing=False)
    lengths = [0.5 * (basis[i, i + 1] + basis[i + 1, i]) for i in range(3)]
    angles = []
    for i in range(2):
        a, b = lengths[i : i + 2]
        span = 0.5 * (basis[i, i + 2] + basis[i + 2, i])
        angles.append(math.acos((a * a + b * b - span * span) / (2 * a * b)))
    a, b, c = lengths
    alpha, beta = angles
    anti_span = math.hypot(a * math.cos(alpha) - b + c * math.cos(beta), a * math.sin(alpha) + c * math.sin(beta))
    preferred = rdDistGeom.GetMoleculeBoundsMatrix(mol, forceTransAmides=False, set15bounds=False)
    assert preferred[0, 3] < anti_span < bnd.ligand_reach(mol)[0, 3]


def test_ligand_reach_keeps_native_bounds_for_ring_closure():
    ring = Chem.MolFromSmiles("Pc1ccccc1P")
    native = rdDistGeom.GetMoleculeBoundsMatrix(ring, set14bounds=True, set15bounds=False)
    assert bnd.ligand_reach(ring)[0, 7] == pytest.approx(native[0, 7])

    # A saturated ring's central bond is not aromatic, so only the native 1-4 lower bound carries over;
    # the upper bound is free-torsion, which is looser than RDKit's sp2-sp2 cis template (4.456 -> 4.552).
    saturated = Chem.MolFromSmiles("PC1CCCCC1P")
    native = rdDistGeom.GetMoleculeBoundsMatrix(saturated, set14bounds=True, set15bounds=False)
    assert bnd.ligand_reach(saturated)[0, 7] >= native[0, 7] - 1e-9

    chain = Chem.MolFromSmiles("PCCCCP")
    free = bnd.ligand_reach(chain)
    native = rdDistGeom.GetMoleculeBoundsMatrix(chain, set14bounds=True, set15bounds=False)
    assert free[0, 5] > native[0, 5]


def test_ligand_reach_frees_a_saturated_ring_torsion_for_a_thiourea_diazepane():
    # JOYDIK: a 7-membered diazepane ring carries two exocyclic C=S groups on adjacent ring carbons.
    # RDKit's native 1-4 upper bound pins the ring's C-C central bond to an sp2-sp2 cis template
    # (hybridisation preference), underestimating the S...S reach against the crystal (3.486 A).
    # Only an aromatic central bond should pin the torsion; this saturated ring bond must not.
    mol = _graph("CN1CCCN(C)C(=S)C1=S")
    assert bnd.ligand_reach(mol)[8, 10] > 3.49


def test_three_bond_upper_encloses_arbitrary_torsions_and_interval_correlations():
    rng = np.random.default_rng(42)
    for _ in range(100):
        a, b, c = rng.uniform(0.6, 2.6, 3)
        alpha, beta = rng.uniform(0.1, math.pi - 0.1, 2)
        pos = np.array(
            [
                [a * math.cos(alpha), a * math.sin(alpha), 0],
                [0, 0, 0],
                [b, 0, 0],
                [b - c * math.cos(beta), c * math.sin(beta), 0],
            ]
        )
        matrix = np.zeros((4, 4))
        for i, j in ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)):
            span = np.linalg.norm(pos[i] - pos[j])
            slack = rng.uniform(0, 0.04)
            matrix[j, i], matrix[i, j] = span * (1 - slack), span * (1 + slack)
        upper = bnd._chain_upper(matrix, (0, 1, 2, 3))
        assert upper == pytest.approx(bnd._chain_upper(matrix, (3, 2, 1, 0)))
        for torsion in np.linspace(-math.pi, math.pi, 9):
            terminal = np.array(
                [b - c * math.cos(beta), c * math.sin(beta) * math.cos(torsion), c * math.sin(beta) * math.sin(torsion)]
            )
            assert np.linalg.norm(pos[0] - terminal) <= upper


@pytest.mark.parametrize("window", [(0.0, 1.0), (1.0, math.inf), (math.nan, 1.0), (3.0, 3.1)])
def test_three_bond_reach_abstains_without_supported_local_triangles(window):
    matrix = np.ones((4, 4))
    matrix[1, 0], matrix[0, 1] = window
    assert math.isinf(bnd._chain_upper(matrix, (0, 1, 2, 3)))


def test_inconsistent_ligand_reach_is_not_repaired(monkeypatch):
    monkeypatch.setattr(bnd.DistanceGeometry, "DoTriangleSmoothing", lambda *_args: False)
    with pytest.raises(ValueError, match="native ligand reach bounds are inconsistent"):
        bnd.ligand_reach(_graph("CCCN"))


def test_compiled_reach_uses_shared_legs_and_only_stated_floors():
    mol = Chem.MolFromSmiles("[He].[He].[He].[He].[He]")
    reach = np.triu(np.full((5, 5), np.inf), 1)
    cons = Constraints(
        metals={0},
        distances={(0, 1): (1.0, 3.0), (0, 2): (2.0, 2.0), (2, 3): (2.0, 2.0)},
        angles={(1, 0, 2): (30.0, 30.0), (1, 2, 3): (60.0, 60.0)},
        floors={(0, 4): 1.5},
    )
    matrix = bnd.coordination_reach(mol, cons, reach)
    reverse = cons.copy(angles=dict(reversed(list(cons.angles.items()))))
    np.testing.assert_array_equal(matrix, bnd.coordination_reach(mol, reverse, reach))
    native = bnd._coordination_reach_base(mol, reach, cons.metals)
    np.testing.assert_array_equal(matrix, bnd.coordination_reach(mol, cons, reach, native=native))
    # Acute minimum is inside the length interval, using the same radial tolerance as publication.
    assert matrix[2, 1] == pytest.approx((2.0 - FIX_DISTANCE_TOL) / 2)
    assert matrix[4, 0] == 1.5
    assert np.isinf(matrix[0, 4])
    assert matrix[4, 1] == 0
    assert np.isinf(matrix[1, 4])
    assert (matrix[3, 2], matrix[2, 3]) == (2.0, 2.0)
    assert cons.distances[(0, 1)] == (1.0, 3.0)


@pytest.mark.parametrize("side", [-1, 1])
def test_compiled_reach_encloses_accepted_radial_boundary_without_changing_targets(side):
    from rxembed.embed import _structural_failure

    mol = Chem.MolFromSmiles("[He].[He].[He]")
    cons = Constraints(metals={0}, distances={(0, 1): (2.0, 2.0), (0, 2): (2.0, 2.0)}, angles={(1, 0, 2): (90.0, 90.0)})
    original = cons.copy()
    conf = Chem.Conformer(3)
    radius = 2.0 + side * 0.75 * FIX_DISTANCE_TOL
    conf.SetPositions(np.array(((0.0, 0.0, 0.0), (radius, 0.0, 0.0), (0.0, radius, 0.0))))
    mol.AddConformer(conf)
    assert _structural_failure(mol, 0, cons) is None
    matrix = bnd.coordination_reach(mol, cons, np.triu(np.full((3, 3), np.inf), 1))
    positions = conf.GetPositions()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        value = np.linalg.norm(positions[i] - positions[j])
        assert matrix[j, i] <= value <= matrix[i, j]
    assert cons == original


def test_compiled_reach_omits_virtual_rows_without_dropping_real_constraints():
    mol = Chem.MolFromSmiles("[He].[He].[He].[He]")
    reach = np.triu(np.full((4, 4), np.inf), 1)
    cons = Constraints(
        metals={0},
        distances={(0, 1): (2.0, 2.0), (0, 2): (2.0, 2.0)},
        angles={(1, 0, 2): (90.0, 90.0)},
        floors={(0, 3): 1.5},
    )
    with_face = cons.copy(
        haptic={4: (2, 3)},
        phantoms={4},
        distances={**cons.distances, (0, 4): (0.1, 0.1), (2, 4): (0.1, 0.1)},
        angles={**cons.angles, (1, 0, 4): (180.0, 180.0), (1, 4, 2): (0.0, 0.0)},
        floors={**cons.floors, (3, 4): 100.0},
    )
    before = with_face.copy()

    np.testing.assert_array_equal(
        bnd.coordination_reach(mol, with_face, reach), bnd.coordination_reach(mol, cons, reach)
    )
    assert with_face == before


@pytest.mark.parametrize("refine", [False, True])
def test_interval_euclidean_certificate_is_not_a_midpoint_or_rank_test(refine):
    matrix = np.full((4, 4), 2.0)
    matrix[3, :3] = matrix[:3, 3] = 1.1
    np.fill_diagonal(matrix, 0.0)
    assert bnd.DistanceGeometry.DoTriangleSmoothing(matrix.copy())
    assert bnd._euclidean_conflict(matrix, refine=refine) is not None
    matrix[3, :3], matrix[:3, 3] = 0.1, 1.16  # Contains the valid equilateral-base centre.
    assert bnd._euclidean_conflict(matrix, refine=refine) is None
    assert bnd._euclidean_conflict(np.ones((5, 5)) - np.eye(5), refine=refine) is None  # Valid in dimension four.
    rng = np.random.default_rng(42)
    for _ in range(100):
        points = rng.normal(size=(5, 3))
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        slack = rng.uniform(0.0, 0.5, distances.shape)
        intervals = np.triu(distances + slack, 1) + np.tril(np.maximum(0.0, distances - slack), -1)
        assert bnd._euclidean_conflict(intervals, refine=refine) is None


def test_interval_euclidean_certificate_preserves_degenerate_eigenspaces():
    groups = np.arange(9) // 3
    squared = np.where(groups[:, None] == groups[None, :], 4.0, 1.21)
    np.fill_diagonal(squared, 0.0)
    lower, upper = np.sqrt(0.65 * squared), np.sqrt(1.35 * squared)
    rng = np.random.default_rng(42)
    for _ in range(20):
        order = rng.permutation(9)
        intervals = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        assert bnd.DistanceGeometry.DoTriangleSmoothing(intervals.copy())
        assert bnd._euclidean_conflict(intervals) == pytest.approx(0.2995, abs=1e-12)
    for invalid in (-0.1, np.nan, np.inf):
        intervals[1, 0] = invalid
        assert bnd._euclidean_conflict(intervals) is None


def test_refinement_certifies_coupled_modes_without_atom_order_dependence():
    n = 8
    u = np.array((1, 1, 1, 1, -1, -1, -1, -1)) / np.sqrt(n)
    v = np.array((1, 1, -1, -1, 1, 1, -1, -1)) / np.sqrt(n)
    gram = np.eye(n) - np.ones((n, n)) / n - 1.2 * np.outer(u, u) - 1.1 * np.outer(v, v)
    squared = np.diag(gram)[:, None] + np.diag(gram) - 2 * gram
    slack = np.where(np.outer(u, u) * np.outer(v, v) < 0, 0.12, 0.0)
    lower, upper = np.sqrt(np.maximum(squared - slack, 0.0)), np.sqrt(squared + slack)
    rng = np.random.default_rng(42)
    margin = None
    for _ in range(10):
        order = rng.permutation(n)
        matrix = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        before = matrix.copy()
        assert bnd._euclidean_conflict(matrix) is None
        found = bnd._euclidean_conflict(matrix, refine=True)
        assert found is not None
        if margin is not None:
            assert found == pytest.approx(margin, abs=1e-10)
        margin = found
        np.testing.assert_array_equal(matrix, before)


# ---------------------------------------------------------------------------------------------------------
# _smooth: the tolerance is the signal
# ---------------------------------------------------------------------------------------------------------


def test_unrepairable_bounds_raise():
    bm = _matrix(_mol())
    bm[0][2], bm[2][0] = 0.31, 0.30
    with pytest.raises(RuntimeError, match="triangle smoothing failed"):
        bnd._smooth(bm)


def test_stereo_carrier_edges_are_not_native_ligand_bounds():
    """Keep donor-metal edges for chirality while deriving bounds from ligand topology."""
    mol = metal_core.connect_metal(Chem.MolFromSmiles("NCCCN.[Fe+2]"), [(0, 5), (4, 5)])
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    cons = Constraints(
        metals={5},
        distances={(0, 5): (2.0, 2.0), (4, 5): (2.0, 2.0)},
        angles={(0, 5, 4): (76.0, 91.0)},
    )
    before = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()]
    before_tags = [atom.GetChiralTag() for atom in mol.GetAtoms()]
    native = metal_core._bounds_matrix(Chem.MolFromSmiles("NCCCN.[Fe+2]"))
    ctx = bnd._write(mol, cons)

    assert (ctx.bm[5, 0], ctx.bm[0, 5]) == (2.0, 2.0)
    assert (ctx.bm[5, 4], ctx.bm[4, 5]) == (2.0, 2.0)
    assert ctx.bm[0, 4] <= 4.0 * math.sin(math.radians(91.0 / 2.0)) + 1e-12
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        assert ctx.bm[a, b] == native[a, b]
        assert ctx.bm[b, a] == native[b, a]
    assert [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()] == before
    assert [atom.GetChiralTag() for atom in mol.GetAtoms()] == before_tags


def test_bounds_strip_only_selected_owned_dative_edges(monkeypatch):
    """Keep unowned spectator and metal-metal edges out of the private native edit."""
    rw = Chem.RWMol(Chem.MolFromSmiles("NCCO.[Cu+].[Zn+]"))
    for begin, end in ((0, 4), (3, 4), (2, 4), (1, 5), (4, 5)):
        rw.AddBond(begin, end, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol, includeDativeBonds=True)
    cons = Constraints(
        metals={4},
        distances={
            (0, 4): (2.0, 2.0),
            (3, 4): (2.0, 2.0),
            (1, 5): (2.0, 2.0),
            (4, 5): (2.0, 2.0),
        },
        angles={(0, 4, 3): (76.0, 91.0)},
    )
    before = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()]
    before_tags = [atom.GetChiralTag() for atom in mol.GetAtoms()]
    seen = []
    native_bounds = metal_core._bounds_matrix

    def capture(candidate, *args, **kwargs):
        seen.append(Chem.Mol(candidate))
        return native_bounds(candidate, *args, **kwargs)

    monkeypatch.setattr(metal_core, "_bounds_matrix", capture)
    bnd._write(mol, cons)
    assert len(seen) == 1
    private = seen[0]
    assert private.GetBondBetweenAtoms(0, 4) is None
    assert private.GetBondBetweenAtoms(3, 4) is None
    for pair in ((2, 4), (1, 5), (4, 5)):
        bond = private.GetBondBetweenAtoms(*pair)
        assert bond is not None
        assert bond.GetBondType() == Chem.BondType.DATIVE
    assert [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()] == before
    assert [atom.GetChiralTag() for atom in mol.GetAtoms()] == before_tags


@pytest.mark.parametrize(
    ("smiles", "donors"),
    [("c1ccncc1.[Cu+]", (3,)), ("N1CCNCC1.[Cu+]", (0, 3))],
    ids=["aromatic-ring", "ring-chelate"],
)
def test_private_native_bounds_rebuild_ring_perception(monkeypatch, smiles, donors):
    """Preserve aromatic and chelate-ring native bounds after removing owned carrier edges."""
    rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
    metal = rw.GetNumAtoms() - 1
    for donor in donors:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    cons = Constraints(
        metals={metal},
        distances={tuple(sorted((donor, metal))): (2.0, 2.0) for donor in donors},
    )
    reference = Chem.MolFromSmiles(smiles)
    expected = metal_core._bounds_matrix(reference)
    seen = []
    native_bounds = metal_core._bounds_matrix

    def capture(candidate, *args, **kwargs):
        seen.append(Chem.Mol(candidate))
        return native_bounds(candidate, *args, **kwargs)

    monkeypatch.setattr(metal_core, "_bounds_matrix", capture)
    ctx = bnd._write(mol, cons)
    private = seen[0]
    rings = private.GetRingInfo().AtomRings()
    assert rings
    assert all(metal not in ring for ring in rings)
    ring = next(ring for ring in rings if len(ring) >= 5)
    for a, b in itertools.combinations(ring, 2):
        assert ctx.bm[a, b] == expected[a, b]
        assert ctx.bm[b, a] == expected[b, a]
    assert all(private.GetBondBetweenAtoms(donor, metal) is None for donor in donors)


# ---------------------------------------------------------------------------------------------------------
# _bounds / _feasible_bounds: the edit, and what happens when it cannot be satisfied
# ---------------------------------------------------------------------------------------------------------


def test_matrix_is_edited_not_replaced():
    mol = _mol()
    before = _matrix(mol)
    assert before[1][0] > 1.0, "the C0-C1 lower bound is RDKit's own bond window: the premise"

    after, tol = bnd._bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert (after[0][2], after[2][0]) == pytest.approx((2.6, 2.5))
    assert tol == 0.0
    assert (after[0][1], after[1][0]) == (before[0][1], before[1][0]), "a bonded pair was rewritten"


@pytest.mark.parametrize(
    ("build", "hidden"),
    [
        (lambda: bnd._bounds(_graph("C[S-]"), Constraints(distances={(0, 1): (1.7, 1.9)})), "UFFTYPER"),
        (
            # the hydride's own construction warns too; build it at collection time (default-arg trick) so
            # only the seeding call itself is under test, matching the UFFTYPER row's fresh-matrix timing
            lambda mol=_graph("[H-].CC"): bnd.seed_coordinates(mol, Constraints(), n=1, seed=42),  # noqa: B008
            "not removing hydrogen atom without neighbors",
        ),
    ],
    ids=["uff_typer_on_bounds_matrix", "disconnected_hydride_warning"],
)
def test_bounds_hide_rdkit_internal_diagnostics(build, hidden, capfd):
    capfd.readouterr()
    assert build()
    assert hidden not in capfd.readouterr().err


def test_unrealisable_spec_names_failed_window(caplog):
    mol = _mol()
    with caplog.at_level("WARNING", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (1.0, 1.02)}))
    assert tol > 0.0
    assert "bounds needed smoothing" in caplog.text
    assert "distance 0-2" in caplog.text, "the tolerance alone points nowhere; the window is the point"


def test_realisable_spec_says_nothing(caplog):
    mol = _mol()
    with caplog.at_level("INFO", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert tol == 0.0
    assert caplog.records == []


# ---------------------------------------------------------------------------------------------------------
# embed: the native distance-geometry call itself
# ---------------------------------------------------------------------------------------------------------


def test_unconstrained_embed_does_not_build_a_custom_matrix(monkeypatch):
    calls = []
    monkeypatch.setattr(bnd, "_feasible_bounds", lambda *a, **k: calls.append(a) or (_matrix(a[0]), 0.0))

    bnd.seed_coordinates(_mol(), Constraints(), n=2, seed=3)
    assert calls == []

    bnd.seed_coordinates(_mol(), Constraints(distances={(0, 2): (2.5, 2.6)}), n=2, seed=3)
    assert len(calls) == 1


def test_embed_ids_are_reproducible_and_attached():
    mol = _graph("CCO")
    ids = bnd.seed_coordinates(mol, Constraints(), n=4, seed=3, prune_rms=-1)
    assert len(ids) == 4
    assert {int(c.GetId()) for c in mol.GetConformers()} == {int(i) for i in ids}

    assert len(bnd.seed_coordinates(_graph("CCO"), Constraints(), n=8, seed=3)) < 8

    a, b = _graph("CCO"), _graph("CCO")
    bnd.seed_coordinates(a, Constraints(), n=2, seed=1234)
    bnd.seed_coordinates(b, Constraints(), n=2, seed=1234)
    assert np.allclose(a.GetConformer(0).GetPositions(), b.GetConformer(0).GetPositions())


def test_seed_coordinates_can_delegate_chirality_to_a_later_accept_gate(monkeypatch):
    seen = []

    def reject(_mol, _n, params):
        seen.append((params.enforceChirality, params.maxIterations))
        return []

    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMultipleConfs", reject)
    bnd.seed_coordinates(
        _graph("F[C@H](Cl)Br"),
        Constraints(),
        n=1,
        seed=42,
        enforce_chirality=False,
        max_attempts=30,
    )

    assert seen == [(False, 30)] * 2


@pytest.mark.parametrize("knowledge", [True, False])
@pytest.mark.parametrize("rejections", [0, 1, 2])
def test_native_recovery_preserves_bounds_and_stereo(monkeypatch, knowledge, rejections):
    native = rdDistGeom.EmbedMultipleConfs
    mol = _graph("F[C@H](Cl)Br")
    cons = Constraints(distances={(0, 1): (1.3, 1.5)})
    stages, matrices, handed = [], [], {}
    build = bnd._feasible_bounds
    set_matrix = rdDistGeom.EmbedParameters.SetBoundsMat

    def matrix(*args):
        result = build(*args)
        matrices.append(result[0].copy())
        return result

    def handoff(params, matrix):
        handed[id(params)] = matrix.copy()
        return set_matrix(params, matrix)

    def embed(candidate, n, params):
        assert (n, params.randomSeed, params.numThreads, params.maxIterations) == (1, 42, 1, 30)
        assert params.enforceChirality
        assert not params.embedFragmentsSeparately
        np.testing.assert_array_equal(handed[id(params)], matrices[0])
        stages.append(
            (
                params.useLegacyImplementation,
                params.useRandomCoords,
                params.useBasicKnowledge,
                params.useExpTorsionAnglePrefs,
            )
        )
        return [] if len(stages) <= rejections else native(candidate, n, params)

    monkeypatch.setattr(bnd, "_feasible_bounds", matrix)
    monkeypatch.setattr(rdDistGeom.EmbedParameters, "SetBoundsMat", handoff)
    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", embed)
    ids = bnd.seed_coordinates(mol, cons, n=1, seed=42, threads=1, knowledge=knowledge, max_attempts=30)

    expected = [(False, False, knowledge, False), (False, True, knowledge, False)]
    assert stages == expected[: rejections + 1]
    assert len(matrices) == 1
    assert cons.distances == {(0, 1): (1.3, 1.5)}
    assert bool(ids) == (rejections < len(expected))
    for cid in ids:
        positions = mol.GetConformer(cid).GetPositions()
        assert 1.3 <= np.linalg.norm(positions[0] - positions[1]) <= 1.5
        measured = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(measured, cid, replaceExistingTags=True)
        assert measured.GetAtomWithIdx(1).GetChiralTag() == mol.GetAtomWithIdx(1).GetChiralTag()


def test_native_failure_counts_are_reported_before_each_fallback(monkeypatch, caplog):
    counts = [0] * (max(map(int, rdDistGeom.EmbedFailureCauses.names.values())) + 1)
    causes = iter(
        [
            rdDistGeom.EmbedFailureCauses.INITIAL_COORDS,
            rdDistGeom.EmbedFailureCauses.LINEAR_DOUBLE_BOND,
        ]
    )

    def reject(_mol, _n, params):
        assert params.trackFailures
        counts[:] = [0] * len(counts)
        counts[int(next(causes))] = 2
        return []

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", reject)
    monkeypatch.setattr(rdDistGeom.EmbedParameters, "GetFailureCounts", lambda _self: tuple(counts))
    with caplog.at_level("DEBUG", logger="rxembed.bounds"):
        assert not bnd.seed_coordinates(_graph("CC"), Constraints(), n=1, seed=42)

    messages = [record.message for record in caplog.records if record.name == "rxembed.bounds"]
    assert len(messages) == 2
    assert "random=False" in messages[0]
    assert "{'INITIAL_COORDS': 2}" in messages[0]
    assert "random=True" in messages[1]
    assert "{'LINEAR_DOUBLE_BOND': 2}" in messages[1]


def test_native_failure_tracking_does_not_change_seed_coordinates(monkeypatch):
    native = rdDistGeom.EmbedMultipleConfs
    tracked, untracked = _graph("CCCO"), _graph("CCCO")
    ids = bnd.seed_coordinates(tracked, Constraints(), n=3, seed=42, threads=1, prune_rms=-1)

    def without_tracking(mol, n, params):
        params.trackFailures = False
        return native(mol, n, params)

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", without_tracking)
    assert bnd.seed_coordinates(untracked, Constraints(), n=3, seed=42, threads=1, prune_rms=-1) == ids
    for cid in ids:
        np.testing.assert_array_equal(
            tracked.GetConformer(cid).GetPositions(), untracked.GetConformer(cid).GetPositions()
        )


@pytest.mark.parametrize("mode", ["knowledge", "broken", "plain"])
def test_seed_selection_prefers_intact_bonds_without_losing_candidates(monkeypatch, mode):
    mol = Chem.MolFromSmiles("CC")
    original_ids = [7, 3, 11, 5]
    lengths = [5.0] * 4 if mode == "broken" else [5.0, 1.5, 4.0, 1.4]
    calls = []

    def embed(candidate, _n, params):
        calls.append((params.useLegacyImplementation, params.useExpTorsionAnglePrefs, params.useBasicKnowledge))
        candidate.RemoveAllConformers()
        for cid, length in zip(original_ids, lengths, strict=True):
            conf = Chem.Conformer(candidate.GetNumAtoms())
            conf.SetId(cid)
            conf.SetAtomPosition(0, (float(cid), 0.0, 0.0))
            conf.SetAtomPosition(1, (cid + length, 0.0, 0.0))
            candidate.AddConformer(conf, assignId=False)
        return original_ids.copy()

    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMultipleConfs", embed)
    ids = bnd.seed_coordinates(mol, Constraints(), n=4, knowledge=mode != "plain")

    assert ids == (original_ids if mode == "broken" else [3, 5, 7, 11])
    assert {conf.GetId() for conf in mol.GetConformers()} == set(original_ids)
    for cid, length in zip(original_ids, lengths, strict=True):
        np.testing.assert_array_equal(
            mol.GetConformer(cid).GetPositions(), [(float(cid), 0.0, 0.0), (cid + length, 0.0, 0.0)]
        )
    assert calls == [(False, False, mode != "plain")]


def test_bring_real_confs_removes_phantoms():
    real = _mol()
    n = real.GetNumAtoms()
    work = Chem.RWMol(real)
    work.AddAtom(Chem.Atom(0))  # the transient centroid dummy
    work = work.GetMol()
    conf = Chem.Conformer(n + 1)
    for a in range(n + 1):
        conf.SetAtomPosition(a, (float(a), 0.0, 0.0))
    conf.SetId(5)
    work.AddConformer(conf, assignId=False)

    bnd._bring_real_confs(real, work, [5])
    assert [int(c.GetId()) for c in real.GetConformers()] == [5]
    assert real.GetConformer(5).GetPositions().shape == (n, 3)
    assert real.GetConformer(5).GetAtomPosition(0).x == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------
# seed_count: scale the seed count by flexibility rather than holding it flat
# ---------------------------------------------------------------------------------------------------------


def test_seed_count_scales_and_clamps():
    rigid, mid, floppy = _graph("CCO"), _graph("C" * 25), _graph("C" * 60)
    assert bnd.seed_count(rigid) > 10  # RDKit's own flat default, which this exists to replace
    assert bnd.seed_count(rigid) < bnd.seed_count(mid) < bnd.seed_count(floppy)
    assert bnd.seed_count(rigid) == bnd.seed_count(_graph("CC")), "the floor must clamp a rigid molecule"
    assert bnd.seed_count(floppy) == bnd.seed_count(_graph("C" * 120)), "the ceiling must clamp a long chain"
    for mol in (rigid, mid):
        assert bnd.seed_count(mol, constrained=True) > bnd.seed_count(mol)


# ---------------------------------------------------------------------------------------------------------
# The angle rules; how an angle-derived window meets the matrix
#
#   R1  a stated `cons.distances` window on the angle's 1-3 pair wins outright; the angle is discarded
#   R2  a real bond path between the end atoms -> intersect the angle-derived window with the matrix
#   R2' ...and if that intersection is disjoint, the backbone wins and the angle contributes nothing
#   R3  no bond path (the atoms meet only through the stripped metal) -> write the angle outright
#   C1  a coplanar entry whose M-D-X angle is unstated contributes nothing (a 1,4 distance carries no
#       dihedral information until that angle is pinned)
#   C2  the 1,4 edge is an EXTREMUM over the stated angle window, never a single pinned angle
#
# Each is built on a hand-made topology, because the branch it selects is chosen by the topology and no real
# molecule offers all six. C1 in particular is silent when wrong: a refactor that drops the skip, or that
# "helpfully" derives the missing angle from the matrix, changes the bounds and raises nothing.
# ---------------------------------------------------------------------------------------------------------


def _rule_mol(smi):
    return Chem.AddHs(Chem.MolFromSmiles(smi))


def _edited(mol, cons):
    """The edited bounds matrix alone; `_bounds` also returns the tolerance it settled at."""
    bm, _tol = bnd._bounds(mol, cons)
    return bm


def _window(bm, i, j):
    """(lo, hi) for a pair, in the matrix's own convention: bm[hi_idx][lo_idx] is the lower bound."""
    a, b = (i, j) if i < j else (j, i)
    return bm[b][a], bm[a][b]


def test_r1_a_stated_distance_pre_empts_the_angle():
    mol = _rule_mol("CCC")
    cons = Constraints()
    add_distance(cons.distances, 0, 2, 3.00, 3.02)
    cons.angles[(0, 1, 2)] = (60.0, 70.0)  # would imply a MUCH shorter 0..2 if it were applied
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (round(lo, 6), round(hi, 6)) == (3.00, 3.02)


@pytest.mark.parametrize(
    ("window", "note"),
    [
        ((100.0, 130.0), "wider than the backbone -> the tighter REAL bound survives untouched"),
        ((109.0, 111.0), "narrower than the backbone -> the angle tightens it"),
    ],
    ids=["wider", "narrower"],
)
def test_angle_bounds_intersect_bond_path(window, note):
    mol = _rule_mol("CCC")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] < mech._DISCONNECTED  # a real bond path: the predicate that selects INTERSECT

    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)
    ctx = mech.DGContext(mol, base)
    d01, d12 = ctx.mid(0, 1), ctx.mid(1, 2)
    alo = mech._law_of_cosines(d01, d12, window[0])
    ahi = mech._law_of_cosines(d01, d12, window[1])

    cons = Constraints()
    cons.angles[(0, 1, 2)] = window
    got = _window(_edited(mol, cons), 0, 2)
    assert got == pytest.approx((max(alo, blo), min(ahi, bhi))), note


def test_r2_disjoint_intersection_keeps_backbone():
    mol = _rule_mol("CCC")
    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = (1.0, 2.0)  # a physically impossible bite -> derived window far below the backbone
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (lo, hi) == pytest.approx((blo, bhi)), "a disjoint intersection must leave the backbone standing"


def test_r3_no_bond_path_writes_the_angle_outright():
    mol = _rule_mol("C.C.C")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] >= mech._DISCONNECTED  # no bond path: the predicate that selects WRITE OUTRIGHT

    cons = Constraints()
    add_distance(cons.distances, 0, 1, 2.00, 2.00)
    add_distance(cons.distances, 1, 2, 2.00, 2.00)
    cons.angles[(0, 1, 2)] = (90.0, 90.0)
    lo, hi = _window(_edited(mol, cons), 0, 2)
    want = math.sqrt(2.0**2 + 2.0**2)  # law of cosines at exactly 90 deg
    assert lo == pytest.approx(want, abs=1e-6)
    assert hi == pytest.approx(want, abs=1e-6)


def _coplanar_case(angle_window):
    """A 4-atom chain with an explicit coplanar cap; `angle_window` optionally pins its M-D-X angle."""
    mol = _rule_mol("C.C.C.C")  # disconnected, so nothing but our own windows reaches the matrix
    cons = Constraints()
    for i, j in ((0, 1), (1, 2), (2, 3)):
        add_distance(cons.distances, i, j, 1.40, 1.40)
    add_distance(cons.distances, 1, 3, 2.40, 2.40)  # pins th_jkw, which is derived from the matrix legs
    if angle_window is not None:
        cons.angles[(0, 1, 2)] = angle_window
    cons.coplanar = [(0, 1, 2, 3, 180.0, 45.0)]
    return mol, cons


def test_unstated_mdx_angle_adds_no_coplanar_bound():
    mol, cons = _coplanar_case(None)
    with_cap = _edited(mol, cons)

    mol2, cons2 = _coplanar_case(None)
    cons2.coplanar = []
    without_cap = _edited(mol2, cons2)

    np.testing.assert_array_equal(with_cap, without_cap)


@pytest.mark.parametrize(
    ("atoms", "overrides"),
    [((0, 1, 2, 3), True), ((3, 2, 1, 0), True), ((4, 1, 2, 5), True), ((0, 1, 4, 3), False)],
)
def test_stated_torsion_supersedes_coplanar_seed_bound(atoms, overrides):
    mol, cons = _coplanar_case((118.0, 122.0))
    without_cap = _edited(mol, cons.copy(coplanar=[]))
    with_cap = _edited(mol, cons)
    assert not np.array_equal(with_cap, without_cap), "the unoverridden cap must be active"

    stated = cons.copy(dihedrals={atoms: (89.0, 91.0)})
    np.testing.assert_array_equal(_edited(mol, stated), without_cap if overrides else with_cap)


@pytest.mark.parametrize("inactive", ["no_donor_angle", "no_well", "explicit_distance"])
def test_inactive_coplanar_leaves_the_matrix_unchanged(inactive):
    mol, cons = _coplanar_case(None if inactive == "no_donor_angle" else (118.0, 122.0))
    if inactive == "no_well":
        cons.coplanar = [(*entry[:4], None, entry[5]) for entry in cons.coplanar]
    if inactive == "explicit_distance":
        cons.distances[(0, 3)] = (1.0, 10.0)
    ctx = bnd._write(mol, cons)
    np.testing.assert_array_equal(ctx.bm, bnd._write(mol, cons.copy(coplanar=[])).bm)


def test_repeated_inactive_cap_does_not_replace_stronger_owner():
    mol, cons = _coplanar_case((118.0, 122.0))
    row = cons.coplanar[0]
    weak, strong = (*row[:5], 45.0), (*row[:5], 20.0)
    cons.coplanar = [weak, strong, weak]
    ctx = bnd._write(mol, cons)
    np.testing.assert_array_equal(ctx.bm, bnd._write(mol, cons.copy(coplanar=[strong])).bm)


def test_c2_14_edge_uses_window_extremum():
    mol, narrow = _coplanar_case((118.0, 122.0))
    _, wide = _coplanar_case((100.0, 180.0))
    lo_n, _ = _window(_edited(mol, narrow), 0, 3)
    lo_w, _ = _window(_edited(mol, wide), 0, 3)
    assert lo_w <= lo_n + 1e-9, "a wider M-D-X window must not produce a TIGHTER coplanarity floor"
