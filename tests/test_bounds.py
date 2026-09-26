"""Test RDKit bounds editing, smoothing and coordinate seeding."""

from __future__ import annotations

import itertools
import json
import math
from functools import partial

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


def _random_start():
    """Return a native KDG object with random starting coordinates, RDKit's own fallback for a hard search."""
    native = rdDistGeom.KDG()
    native.useLegacyImplementation = False
    native.useRandomCoords = True
    return native


# ---------------------------------------------------------------------------------------------------------
# embed_parameters: native model and reproducible sampling defaults
# ---------------------------------------------------------------------------------------------------------


def test_prune_rms_reaches_the_native_seed_batch():
    """EmbedParams(prune_rms=...) reaches RDKit's own duplicate-pruning threshold, thinning the seed batch."""
    assert bnd.embed_parameters(11, prune_rms=0.5).pruneRmsThresh == 0.5

    off = bnd.seed_coordinates(_graph("CCCCO"), Constraints(), 20, bnd.EmbedParams(seed=3, prune_rms=-1))
    on = bnd.seed_coordinates(_graph("CCCCO"), Constraints(), 20, bnd.EmbedParams(seed=3, prune_rms=0.5))
    assert len(on) < len(off)


def test_native_parameters_reach_rdkit_without_model_fallback_or_stale_bounds():
    """The caller's own EmbedParams(native=...) object is restored to its prior state after use."""
    native = rdDistGeom.srETKDGv3()
    native.useRandomCoords = True
    params = bnd.EmbedParams(seed=42, threads=1, prune_rms=-1, native=native)
    # Set the caller's own pre-existing native fields only after wrapping: EmbedParams requires them at
    # RDKit's defaults, but seed_coordinates must still save and restore whatever the caller had before it.
    native.randomSeed, native.numThreads, native.pruneRmsThresh = 7, 2, 0.4
    before = json.loads(rdDistGeom.EmbedParametersToJSON(native))

    assert bnd.seed_coordinates(_graph("CCCC"), Constraints(), 1, params)

    after = json.loads(rdDistGeom.EmbedParametersToJSON(native))
    assert after.pop("boundsMatrix")
    assert after == before


def test_native_timeout_is_not_a_conformer_id_or_an_implicit_retry(monkeypatch):
    native = rdDistGeom.KDG()
    native.timeout = 1
    params = bnd.EmbedParams(seed=42, threads=2, native=native)
    before = native.randomSeed, native.numThreads, native.clearConfs
    calls = []

    def timeout(_mol, _n, used):
        calls.append(used)
        return [-1]  # RDKit's timeout sentinel, not an attached conformer.

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", timeout)
    with pytest.raises(TimeoutError, match=r"native RDKit.*timeout=1s.*native\.timeout"):
        bnd.seed_coordinates(_graph("CC"), Constraints(), 1, params)
    assert calls == [native]
    assert (native.randomSeed, native.numThreads, native.clearConfs) == before


# ---------------------------------------------------------------------------------------------------------
# EmbedParams: one owner per setting
# ---------------------------------------------------------------------------------------------------------


def test_seed_or_threads_together_with_params_is_a_loud_conflict():
    with pytest.raises(ValueError, match=r"dataclasses\.replace"):
        bnd.resolve_params(bnd.EmbedParams(), 5, None)
    with pytest.raises(ValueError, match=r"dataclasses\.replace"):
        bnd.resolve_params(bnd.EmbedParams(), None, 2)
    assert bnd.resolve_params(None, 5, 2) == bnd.EmbedParams(seed=5, threads=2)
    assert bnd.resolve_params(None, None, None) == bnd.EmbedParams()


def test_a_plain_native_object_must_be_wrapped():
    with pytest.raises(TypeError, match=r"EmbedParams\(native=\.\.\.\)"):
        bnd.resolve_params(rdDistGeom.KDG(), None, None)


@pytest.mark.parametrize(
    ("rdkit_name", "field_name"), [("randomSeed", "seed"), ("numThreads", "threads"), ("pruneRmsThresh", "prune_rms")]
)
def test_native_sampling_fields_must_stay_at_rdkits_own_defaults(rdkit_name, field_name):
    native = rdDistGeom.KDG()
    setattr(native, rdkit_name, 42 if rdkit_name != "pruneRmsThresh" else 0.4)
    with pytest.raises(ValueError, match=f"native.{rdkit_name}.*EmbedParams\\({field_name}="):
        bnd.EmbedParams(native=native)


def test_knowledge_together_with_native_must_agree_or_stay_unset():
    with pytest.raises(ValueError, match="useBasicKnowledge"):
        bnd.EmbedParams(native=rdDistGeom.KDG(), knowledge=False)
    with pytest.raises(ValueError, match="useBasicKnowledge"):
        bnd.EmbedParams(native=rdDistGeom.ETDG(), knowledge=True)
    assert bnd.EmbedParams(native=rdDistGeom.KDG(), knowledge=True).knowledge
    assert bnd.EmbedParams(native=rdDistGeom.KDG()).native.useBasicKnowledge  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("name", ["coplanar_14", "metal_floor_relief", "donor_orientation", "conjugation"])
def test_a_non_bool_switch_is_a_type_error(name):
    with pytest.raises(TypeError, match=f"{name} must be True or False"):
        bnd.EmbedParams(**{name: "yes"})  # ty: ignore[invalid-argument-type]


def test_a_negative_seed_is_not_reproducible():
    with pytest.raises(ValueError, match="not reproducible"):
        bnd.EmbedParams(seed=-1)


def test_native_must_be_an_embed_parameters_object():
    with pytest.raises(TypeError, match="EmbedParameters"):
        bnd.EmbedParams(native={})  # ty: ignore[invalid-argument-type]


def test_aio_refines_the_edited_interfragment_distance():
    native = rdDistGeom.srETKDGv3()
    native.useLegacyImplementation = False
    params = bnd.EmbedParams(seed=42, threads=1, native=native)
    for distance in (3.0, 6.0):
        mol = _graph("CC.CC")
        cons = Constraints(distances={(0, 2): (distance, distance + 0.05)})
        ids = bnd.seed_coordinates(mol, cons, 1, params)
        assert ids
        positions = mol.GetConformer(ids[0]).GetPositions()
        assert distance - 0.1 < np.linalg.norm(positions[0] - positions[2]) < distance + 0.15


# ---------------------------------------------------------------------------------------------------------
# probe_conformer: a throwaway geometry that decides a discrete question
#
# That its seed decides the answer, rather than process history, is asserted end to end on the one caller
# that consumes it: `tests/test_embed.py`'s encounter-bounds pair.
# ---------------------------------------------------------------------------------------------------------


def test_failed_probe_returns_none_rather_than_an_empty_mol(monkeypatch):
    monkeypatch.setattr(bnd.rdDistGeom, "EmbedMolecule", lambda *_a, **_k: -1)
    assert bnd.probe_conformer(_mol(), 7) is None


def test_probe_geometry_ignores_rdkits_own_global_random_state():
    """A probe's explicit seed must reproduce even after another call consumes RDKit's global RNG."""
    mol = _graph("CCCO")  # propanol-sized

    first = bnd.probe_conformer(mol, 7)
    rdDistGeom.EmbedMolecule(_graph("CCCO"), randomSeed=-1)  # perturb RDKit's global RNG stream
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
    # Only an aromatic central bond should pin the torsion; a saturated ring bond must not, even with
    # exocyclic sp2 substituents. JOYDIK's crystal S...S reach is 3.486 A.
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
    native = bnd.coordination_reach_base(mol, reach, cons.metals)
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
    native = bnd.bounds_matrix(Chem.MolFromSmiles("NCCCN.[Fe+2]"))
    ctx = bnd._write(mol, cons)

    assert (ctx.bm[5, 0], ctx.bm[0, 5]) == (2.0, 2.0)
    assert (ctx.bm[5, 4], ctx.bm[4, 5]) == (2.0, 2.0)
    assert ctx.bm[0, 4] <= 4.0 * math.sin(math.radians(91.0 / 2.0)) + 1e-12
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        assert ctx.bm[a, b] == native[a, b]
        assert ctx.bm[b, a] == native[b, a]
    assert [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()] == before
    assert [atom.GetChiralTag() for atom in mol.GetAtoms()] == before_tags


def _dative_complex(edges):
    """N0-C1-C2-O3 with dative ``edges`` added to Cu4 and Zn5."""
    rw = Chem.RWMol(Chem.MolFromSmiles("NCCO.[Cu+].[Zn+]"))
    for begin, end in edges:
        rw.AddBond(begin, end, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol, includeDativeBonds=True)
    return mol


def test_bounds_strip_only_selected_owned_dative_edges():
    """Keep unowned spectator and metal-metal edges out of the private native edit."""
    all_edges = ((0, 4), (3, 4), (2, 4), (1, 5), (4, 5))
    spectator_edges = ((2, 4), (1, 5), (4, 5))  # not stated as an owned coordination bond in cons
    mol = _dative_complex(all_edges)
    before = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()]
    before_tags = [atom.GetChiralTag() for atom in mol.GetAtoms()]
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

    kept = bnd.bounds_matrix(_dative_complex(all_edges))
    stripped = bnd.bounds_matrix(_dative_complex(spectator_edges))
    ctx = bnd._write(mol, cons)

    for i, j in ((1, 4), (0, 5), (3, 5), (1, 3)):  # reachable only through an owned, stripped donor edge
        a, b = min(i, j), max(i, j)
        assert (ctx.bm[b, a], ctx.bm[a, b]) == (stripped[b, a], stripped[a, b])
    a, b = 2, 4  # the spectator dative edge is not owned, so it stays in the private edit
    assert (ctx.bm[b, a], ctx.bm[a, b]) == (kept[b, a], kept[a, b])
    assert [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in mol.GetBonds()] == before
    assert [atom.GetChiralTag() for atom in mol.GetAtoms()] == before_tags


@pytest.mark.parametrize(
    ("smiles", "donors"),
    [("c1ccncc1.[Cu+]", (3,)), ("N1CCNCC1.[Cu+]", (0, 3))],
    ids=["aromatic-ring", "ring-chelate"],
)
def test_private_native_bounds_rebuild_ring_perception(smiles, donors):
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
    expected = bnd.bounds_matrix(reference)
    rings = reference.GetRingInfo().AtomRings()
    assert rings
    ring = next(ring for ring in rings if len(ring) >= 5)

    ctx = bnd._write(mol, cons)

    for a, b in itertools.combinations(ring, 2):
        assert ctx.bm[a, b] == expected[a, b]
        assert ctx.bm[b, a] == expected[b, a]


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
            # the hydride's own construction warns too; `partial` builds it at collection time, so only the
            # seeding call itself is under test. This fragment only embeds with a random start, so ask for
            # one through a native object, same as any other caller of an unconstrained multi-fragment input.
            partial(
                bnd.seed_coordinates,
                _graph("[H-].CC"),
                Constraints(),
                1,
                bnd.EmbedParams(seed=42, native=_random_start()),
            ),
            "not removing hydrogen atom without neighbors",
        ),
    ],
    ids=["uff_typer_on_bounds_matrix", "disconnected_hydride_warning"],
)
def test_bounds_hide_rdkit_internal_diagnostics(build, hidden, capfd):
    capfd.readouterr()
    assert build()
    assert hidden not in capfd.readouterr().err


@pytest.mark.parametrize(("stated", "level"), [(False, "DEBUG"), (True, "WARNING")])
def test_unrealisable_spec_names_failed_window(caplog, stated, level):
    """The seed repair names its pair, and warns only when that pair is the caller's own fix= distance."""
    mol = _mol()
    window = {(0, 2): (1.0, 1.02)}
    with caplog.at_level("DEBUG", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances=window, fixed=window if stated else {}))
    assert tol > 0.0
    repair = next(record for record in caplog.records if "needed smoothing" in record.getMessage())
    assert repair.levelname == level
    assert "C0-O2" in repair.getMessage(), "the tolerance alone points nowhere; the window is the point"


def test_realisable_spec_says_nothing(caplog):
    mol = _mol()
    with caplog.at_level("INFO", logger="rxembed.bounds"):
        _bm, tol = bnd._feasible_bounds(mol, Constraints(distances={(0, 2): (2.5, 2.6)}))
    assert tol == 0.0
    assert caplog.records == []


# ---------------------------------------------------------------------------------------------------------
# _cap_fragment_contacts: every free component repels every other at its van der Waals floor and stays
# within a shared, formula-derived ceiling; no probe conformer, no chosen contact pair
# ---------------------------------------------------------------------------------------------------------


def test_cap_fragment_contacts_bounds_every_free_pair():
    mol = _graph("C.N")
    bm = _matrix(mol)
    assert bm[0][1] > 100.0, "the premise: two unlinked fragments start at RDKit's raw unset upper bound"

    bnd._cap_fragment_contacts(mol, Constraints(), bm)

    assert bm[1][0] <= bm[0][1] < 100.0, "the pair now has a real, finite ceiling above its own floor"


def test_cap_fragment_contacts_treats_distance_linked_atoms_as_one_component():
    mol = Chem.MolFromSmiles("C.N.O.F")
    bm = _matrix(mol)
    raw = bm[1][2]

    bnd._cap_fragment_contacts(mol, Constraints(distances={(0, 1): (2.0, 2.1), (0, 2): (2.0, 2.1)}), bm)

    assert bm[1][2] == raw, "N and O are held into one component with C; capping never touches a same-component pair"
    for atom in (0, 1, 2):
        a, b = min(atom, 3), max(atom, 3)
        assert bm[a][b] < 100.0, f"atom {atom}'s component and the free F must both get a ceiling"


def test_cap_fragment_contacts_skips_two_already_pinned_fragments():
    mol = _graph("C.N")
    bm = _matrix(mol)
    raw = bm[0][1]

    bnd._cap_fragment_contacts(mol, Constraints(frozen={0, 1}), bm)

    assert bm[0][1] == raw, "both fragments are already pinned elsewhere; forcing them together could fight that"


# ---------------------------------------------------------------------------------------------------------
# embed: the native distance-geometry call itself
# ---------------------------------------------------------------------------------------------------------


def test_embed_ids_are_reproducible_and_attached():
    mol = _graph("CCO")
    ids = bnd.seed_coordinates(mol, Constraints(), 4, bnd.EmbedParams(seed=3, prune_rms=-1))
    assert len(ids) == 4
    assert {int(c.GetId()) for c in mol.GetConformers()} == {int(i) for i in ids}

    assert len(bnd.seed_coordinates(_graph("CCO"), Constraints(), 8, bnd.EmbedParams(seed=3))) < 8

    a, b = _graph("CCO"), _graph("CCO")
    bnd.seed_coordinates(a, Constraints(), 2, bnd.EmbedParams(seed=1234))
    bnd.seed_coordinates(b, Constraints(), 2, bnd.EmbedParams(seed=1234))
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
        1,
        bnd.EmbedParams(seed=42),
        enforce_chirality=False,
        max_attempts=30,
    )

    assert seen == [(False, 30)]


def test_native_failure_counts_are_reported_once_with_a_random_start_remedy(monkeypatch, caplog):
    counts = [0] * (max(map(int, rdDistGeom.EmbedFailureCauses.names.values())) + 1)
    counts[int(rdDistGeom.EmbedFailureCauses.INITIAL_COORDS)] = 2

    def reject(_mol, _n, params):
        assert params.trackFailures
        return []

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", reject)
    monkeypatch.setattr(rdDistGeom.EmbedParameters, "GetFailureCounts", lambda _self: tuple(counts))
    with caplog.at_level("DEBUG", logger="rxembed.bounds"):
        assert not bnd.seed_coordinates(_graph("CC"), Constraints(), 1, bnd.EmbedParams(seed=42))

    messages = [record.message for record in caplog.records if record.name == "rxembed.bounds"]
    assert len(messages) == 1
    assert "random=False" in messages[0]
    assert "{'INITIAL_COORDS': 2}" in messages[0]
    assert "EmbedParams(native=...), useRandomCoords=True" in messages[0]


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
    ids = bnd.seed_coordinates(mol, Constraints(), 4, bnd.EmbedParams(knowledge=mode != "plain"))

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


def test_reversed_angle_preserves_distance_window():
    """A stated distance window on the angle's 1-3 pair wins outright, regardless of the angle key's atom order."""
    mol = _rule_mol("CCCC")

    def window(angle_key):
        cons = Constraints()
        add_distance(cons.distances, 0, 3, 1.50, 1.56)
        cons.angles[angle_key] = (95.0, 105.0)
        return _window(_edited(mol, cons), 0, 3)

    assert window((0, 1, 3)) == pytest.approx(window((3, 1, 0))), "angle index order changed the bounds"
    assert window((3, 1, 0)) == pytest.approx((1.50, 1.56)), "the explicit window was clobbered"


@pytest.mark.parametrize(
    ("window", "tightens"), [((100.0, 130.0), False), ((109.0, 111.0), True)], ids=["wider", "narrower"]
)
def test_angle_bounds_intersect_bond_path(window, tightens):
    """A real bond path intersects the angle-derived window with the backbone; a wider one leaves it untouched."""
    mol = _rule_mol("CCC")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] < mech.DISCONNECTED  # a real bond path: the predicate that selects INTERSECT

    blo, bhi = _window(_edited(mol, Constraints()), 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = window
    lo, hi = _window(_edited(mol, cons), 0, 2)

    assert blo <= lo <= hi <= bhi, "the intersection must never widen the backbone window"
    assert ((lo, hi) != (blo, bhi)) == tightens


def test_r2_disjoint_intersection_keeps_backbone():
    mol = _rule_mol("CCC")
    base = _edited(mol, Constraints())
    blo, bhi = _window(base, 0, 2)

    cons = Constraints()
    cons.angles[(0, 1, 2)] = (1.0, 2.0)  # a physically impossible bite -> derived window far below the backbone
    lo, hi = _window(_edited(mol, cons), 0, 2)
    assert (lo, hi) == pytest.approx((blo, bhi)), "a disjoint intersection must leave the backbone standing"


def test_angle_prior_keeps_the_nonbonded_floor_between_cis_donors():
    """A 1-6 pair beyond RDKit's own 1-5 topology bound has only a generic nonbonded floor, not real geometry.

    A tight metal-leg angle can imply a window disjoint from and entirely below that floor; the disjoint
    intersection must leave RDKit's bound standing, as if the angle had never been stated.
    """
    mol = _rule_mol("CCCCCC.[Ni]")
    assert Chem.GetDistanceMatrix(mech.disconnect_metal(mol), force=True)[0][5] == 5  # past RDKit's 1-5 bound

    cons_no_angle = Constraints()
    add_distance(cons_no_angle.distances, 0, 6, 1.95, 2.05)
    add_distance(cons_no_angle.distances, 6, 5, 1.95, 2.05)
    without_angle = _window(_edited(mol, cons_no_angle), 0, 5)

    cons = Constraints()
    add_distance(cons.distances, 0, 6, 1.95, 2.05)
    add_distance(cons.distances, 6, 5, 1.95, 2.05)
    cons.angles[(0, 6, 5)] = (62.0, 74.0)  # a physically tight bite: its derived window sits below RDKit's floor
    with_angle = _window(_edited(mol, cons), 0, 5)

    assert with_angle == pytest.approx(without_angle), "a disjoint intersection must leave the backbone standing"


def test_r3_no_bond_path_writes_the_angle_outright():
    mol = _rule_mol("C.C.C")
    topo = Chem.GetDistanceMatrix(mol)
    assert topo[0][2] >= mech.DISCONNECTED  # no bond path: the predicate that selects WRITE OUTRIGHT

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
