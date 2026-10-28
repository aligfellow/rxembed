"""Test paired reaction endpoints through the public coembedding API."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed import coembedding
from rxembed.bounds import EmbedParams
from rxembed.constraints import Constraints
from rxembed.embed import Conformers
from rxembed.stereo import defined_stereo_label, stereo_from_3d
from tests.conftest import EXAMPLES_DIR


def _bromobutane_and_chlorobutane():
    """Return 1-bromobutane + Cl- and 1-chlorobutane + Br- on one atom order: halides at 4 and 5, then the Hs."""
    reactant = Chem.AddHs(Chem.MolFromSmiles("CCCCBr.[Cl-]"))
    product = Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]"))
    order = [0, 1, 2, 3, 5, 4, *range(6, product.GetNumAtoms())]
    return reactant, Chem.RenumberAtoms(product, order)


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (((1.40, 1.45), (1.44, 1.50)), (1.42, 1.475)),
        (((1.40, 1.50), (1.50, 1.60)), (1.45, 1.55)),
        (((1.35, 1.60), (1.40, 1.50)), (1.375, 1.55)),
        (((1.45, 1.45), (1.40, 1.50)), (1.425, 1.475)),
        (((1.40, 1.50), (1.40, 1.50)), (1.40, 1.50)),
        (((1.45, 1.45), (1.45, 1.45)), (1.45, 1.45)),
        (((1.35, 1.40), (1.45, 1.50)), (1.35, 1.50)),
    ],
    ids=["overlapping", "touching", "contained", "point-overlap", "identical", "identical-points", "disjoint"],
)
@pytest.mark.parametrize("reorder_product", [False, True])
def test_methanol_union_sends_endpoint_bounds_to_rdkit(monkeypatch, windows, expected, reorder_product):
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    order = [0, *range(2, mol.GetNumAtoms()), 1] if reorder_product else list(range(mol.GetNumAtoms()))
    product = Chem.RenumberAtoms(mol, order)
    correspondence = tuple((i, order.index(i)) for i in range(mol.GetNumAtoms()))
    owners = [
        Conformers(Chem.Mol(mol), [], Constraints(distances={(0, 1): windows[0]})),
        Conformers(product, [], Constraints(distances={(order.index(0), order.index(1)): windows[1]})),
    ]
    native = rdDistGeom.EmbedMultipleConfs
    received = []

    def embed(graph, n, params):
        matrix = json.loads(rdDistGeom.EmbedParametersToJSON(params))["boundsMatrix"]
        received.append((float(matrix[1][0]), float(matrix[0][1])))
        return native(graph, n, params)

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", embed)
    seeded, ids = coembedding.seed_union_conformers(owners, correspondence, 3, EmbedParams(seed=42, threads=1))

    assert len(received) == 1
    assert received[0] == pytest.approx(expected)
    assert owners[0].cons.distances[(0, 1)] == windows[0]
    assert owners[1].cons.distances[(order.index(0), order.index(1))] == windows[1]
    assert ids
    for cid in ids:
        positions = seeded.GetConformer(cid).GetPositions()
        assert expected[0] - 0.01 < np.linalg.norm(positions[0] - positions[1]) < expected[1] + 0.01


def test_sn2_copies_keep_their_own_bonds_and_overlay_the_spectator_chain():
    reactant, product = _bromobutane_and_chlorobutane()

    result = rx.coembed(reactant, product, n=4, seed=3)
    left, right = result.reactant, result.product

    assert left.ids == right.ids
    assert 0 < len(left.ids) <= 4
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[3] - r[4]) < 2.1  # C-Br bonded
        assert np.linalg.norm(r[3] - r[5]) > 2.6  # Cl- free
        assert np.linalg.norm(p[3] - p[5]) < 1.9  # C-Cl bonded
        assert np.linalg.norm(p[3] - p[4]) > 2.6  # Br- free
        assert np.linalg.norm(r[:2] - p[:2], axis=1).max() < 0.5  # the far end of the chain: one shared pose


def test_ethylamine_copper_preparation_restores_a_seed_when_uff_tears_its_ligand(monkeypatch):
    core_embed = importlib.import_module("rxembed.embed")
    native = core_embed.restrained_uff
    source = rx.metal(Chem.AddHs(rx.parse_smiles("CCN->[Cu+]<-NCC")), "linear")[0]

    def torn_ligand(mol, cons, **kwargs):
        energies = native(mol, cons, **kwargs)
        if kwargs["max_iters"]:
            for cid in kwargs["conf_ids"]:
                conf = mol.GetConformer(cid)
                positions = conf.GetPositions()
                positions[0] += np.array([20.0, 0.0, 0.0])
                conf.SetPositions(positions)
        return energies

    monkeypatch.setattr(core_embed, "restrained_uff", torn_ligand)
    pair = rx.coembed(source, source, n=1, seed=42, joint_max_iters=1)

    assert len(pair) == 1
    for endpoint in (pair.reactant, pair.product):
        positions = endpoint.mol.GetConformer(pair.ids[0]).GetPositions()
        assert np.linalg.norm(positions[0] - positions[1]) < 2.0


def test_ethane_pair_keeps_its_valid_seed_when_the_single_joint_pass_tears_a_bond(monkeypatch):
    native = coembedding._coupled_uff
    calls = []

    def torn_joint(*args, **kwargs):
        positions, record = native(*args, **kwargs)
        positions[0] += np.array([20.0, 0.0, 0.0])
        calls.append(record)
        return positions, record

    monkeypatch.setattr(coembedding, "_coupled_uff", torn_joint)
    pair = rx.coembed("CC", "CC", n=1, seed=42)

    assert len(pair) == 1
    assert len(calls) == 1
    positions = pair.reactant.mol.GetConformer(pair.ids[0]).GetPositions()
    assert np.linalg.norm(positions[0] - positions[1]) < 2.0
    assert pair.joint_results[pair.ids[0]]["stopping_reason"] == "validation_adjusted"


def test_sn2_uses_union_cpci_preparation_and_one_joint_optimization_per_candidate(monkeypatch):
    reactant, product = _bromobutane_and_chlorobutane()
    native_union = coembedding.seed_union_conformers
    core_embed = importlib.import_module("rxembed.embed")
    native_uff = core_embed.restrained_uff
    native_joint = coembedding._coupled_uff
    unions, free_calls, joint_calls = [], [], []

    def union(*args, **kwargs):
        graph, ids = native_union(*args, **kwargs)
        assert graph.GetNumAtoms() == reactant.GetNumAtoms()
        assert graph.GetBondBetweenAtoms(3, 4).GetBondType() == Chem.BondType.ZERO
        assert graph.GetBondBetweenAtoms(3, 5).GetBondType() == Chem.BondType.ZERO
        assert graph.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
        unions.append(graph)
        return graph, ids

    def uff(graph, cons, **kwargs):
        assert all(b.GetBondType() != Chem.BondType.ZERO for b in graph.GetBonds())
        if kwargs["max_iters"]:
            assert kwargs["max_iters"] == 100
            assert not cons.frozen
            free_calls.append(kwargs["max_iters"])
        return native_uff(graph, cons, **kwargs)

    def joint(graphs, terms, alignment, cid, **kwargs):
        assert all(not term.frozen for term in terms)
        joint_calls.append(cid)
        return native_joint(graphs, terms, alignment, cid, **kwargs)

    monkeypatch.setattr(coembedding, "seed_union_conformers", union)
    monkeypatch.setattr(core_embed, "restrained_uff", uff)
    monkeypatch.setattr(coembedding, "_coupled_uff", joint)
    result = rx.coembed(reactant, product, n=1, seed=42, threads=1)

    assert len(result) == 1
    assert unions
    assert free_calls == [100, 100]
    assert joint_calls == [0]
    assert result.joint_results[result.ids[0]]["stage"] == "joint"
    assert result.scores[result.ids[0]] == pytest.approx(result.joint_results[result.ids[0]]["score"])


@pytest.mark.parametrize("caller_native", [False, True])
def test_sn2_uses_graph_cpci_unless_the_caller_owns_native_preferences(monkeypatch, caller_native):
    reactant, product = _bromobutane_and_chlorobutane()
    native = rdDistGeom.ETKDGv3()
    native.SetCPCI({(4, 5): 0.4})
    setter = rdDistGeom.EmbedParameters.SetCPCI
    supplied = []

    def cpci(params, values):
        supplied.append(dict(values))
        return setter(params, values)

    monkeypatch.setattr(rdDistGeom.EmbedParameters, "SetCPCI", cpci)
    params = EmbedParams(seed=42, threads=1, native=native if caller_native else None)
    result = rx.coembed(reactant, product, n=1, params=params)

    assert len(result) == 1
    if caller_native:
        assert not supplied
        assert native.randomSeed == -1
        assert native.numThreads == 1
        assert native.pruneRmsThresh == -1
    else:
        assert supplied
        assert all(values[(3, 4)] < 0 and values[(3, 5)] < 0 and values[(4, 5)] > 0 for values in supplied)


@pytest.mark.parametrize("reverse", [False, True])
def test_aromatic_hydrogenation_seeds_both_input_directions_without_broken_aromatic_flags(reverse):
    left = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[H:7][H:8]"
    right = "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]([H:7])[CH:6]1[H:8]"
    result = rx.coembed(*(right, left) if reverse else (left, right), n=1, seed=42, threads=1)
    assert len(result) == 1
    endpoints = (result.product, result.reactant) if reverse else (result.reactant, result.product)
    assert any(b.GetIsAromatic() for b in endpoints[0].mol.GetBonds())
    assert not any(b.GetIsAromatic() for b in endpoints[1].mol.GetBonds())


def test_constrain_holds_the_chloride_beside_the_chain_end_in_both_copies():
    reactant, product = _bromobutane_and_chlorobutane()

    result = rx.coembed(reactant, product, constrain={(0, 5): (3.0, 3.3)}, n=3, seed=3)
    left, right = result.reactant, result.product

    assert result.ids
    for conformers in (left, right):  # free Cl- in the reactant, bonded Cl in the product
        for cid in conformers.ids:
            pos = conformers.mol.GetConformer(cid).GetPositions()
            assert 2.95 < np.linalg.norm(pos[0] - pos[5]) < 3.35


def test_metal_copies_keep_their_own_coordination_sphere():
    reactant = Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[NH3:3].[Cl-:4]"))  # N0 Cu1 N2 Cl3
    product = Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[Cl-:4].[NH3:3]"))  # N0 Cu1 Cl2 N3
    product = Chem.RenumberAtoms(product, [0, 1, 3, 2, *range(4, product.GetNumAtoms())])

    result = rx.coembed(rx.metal(reactant, "linear")[0], rx.metal(product, "linear")[0], n=2, seed=1)
    left, right = result.reactant, result.product

    assert left.ids == right.ids
    assert 0 < len(left.ids) <= 2
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[1] - r[2]) < 2.2  # Cu-N bonded
        assert np.linalg.norm(r[1] - r[3]) > 2.6  # Cl- free
        assert np.linalg.norm(p[1] - p[3]) < 2.4  # Cu-Cl bonded
        assert np.linalg.norm(p[1] - p[2]) > 2.6  # NH3 free


def test_hexammine_potassium_co_embeds_where_every_eigenvector_start_is_rejected():
    hexammine = Chem.AddHs(rx.parse_smiles("N->[K+](<-N)(<-N)(<-N)(<-N)<-N"))  # K sits at its donors' centroid

    left = rx.coembed(hexammine, Chem.Mol(hexammine), n=2, seed=7).reactant

    assert len(left.ids) == 2


@pytest.mark.parametrize("smiles", ["C[C@H](O)CC", "Cc1cccc(O)c1-c1c(C)cccc1O |wU:7.6|"], ids=["centre", "axis"])
def test_copies_that_differ_only_in_configuration_each_keep_their_own(smiles):
    left = Chem.AddHs(Chem.MolFromSmiles(smiles))
    right = Chem.AddHs(Chem.MolFromSmiles(smiles.replace("@", "@@").replace("wU", "wD")))

    result = rx.coembed(left, right, n=4, seed=1)
    copies = result.reactant, result.product

    assert 0 < len(copies[0].ids) <= 4
    for conformers, mol in zip(copies, (left, right), strict=True):
        for cid in conformers.ids:
            assert stereo_from_3d(Chem.Mol(conformers.mol, confId=cid)) == defined_stereo_label(mol)


def test_metal_copies_that_differ_only_in_a_bound_amine_hand_co_embed():
    left, right = (
        rx.metal(Chem.AddHs(rx.parse_smiles(f"[Pd](Cl)(Cl)(Cl)([N{tag}H](C)O)")), "SPL")[0] for tag in ("@", "@@")
    )

    assert 0 < len(rx.coembed(left, right, n=4, seed=1)) <= 4


def test_inverting_sn2_copies_overlay_the_spectator_carbons():
    reactant = Chem.AddHs(Chem.MolFromSmiles("C[C@H](Br)CC.[Cl-]"))
    product = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](Cl)CC.[Br-]"))
    product = Chem.RenumberAtoms(product, [0, 1, 5, 3, 4, 2, *range(6, product.GetNumAtoms())])

    result = rx.coembed(reactant, product, n=4, seed=1)
    left, right = result.reactant, result.product

    assert left.ids
    for cid in left.ids:
        r, p = left.mol.GetConformer(cid).GetPositions(), right.mol.GetConformer(cid).GetPositions()
        assert np.linalg.norm(r[[0, 3, 4]] - p[[0, 3, 4]], axis=1).max() < 0.5


def test_sn2_endpoints_can_use_different_atom_orders():
    reactant, _product = _bromobutane_and_chlorobutane()
    product = Chem.AddHs(Chem.MolFromSmiles("CCCCCl.[Br-]"))

    result = rx.coembed(reactant, product, n=1, seed=3)

    assert result.ids
    assert len(result.correspondence) == reactant.GetNumAtoms()
    assert result.reactant.mol.GetAtomWithIdx(4).GetSymbol() == "Br"
    assert result.product.mol.GetAtomWithIdx(4).GetSymbol() == "Cl"
    assert dict(result.correspondence)[4] == 5


def test_nazarov_xyz_endpoints_keep_their_input_atom_order():
    paths = [str(EXAMPLES_DIR / f"nazarov_{side}.xyz") for side in ("start", "end")]
    originals = [Chem.MolFromXYZFile(path) for path in paths]

    result = rx.coembed(*paths, n=1, seed=1, stereo="free")

    assert len(result) == 1
    assert result.correspondence == tuple((i, i) for i in range(originals[0].GetNumAtoms()))
    for endpoint, original in zip((result.reactant, result.product), originals, strict=True):
        assert [atom.GetSymbol() for atom in endpoint.mol.GetAtoms()] == [
            atom.GetSymbol() for atom in original.GetAtoms()
        ]
        assert np.all(np.isfinite(endpoint.mol.GetConformer(result.ids[0]).GetPositions()))
    assert np.isfinite(result.alignment_scores[result.ids[0]])
    assert np.isfinite(result.react_rmsds[result.ids[0]])


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_ethane_pair_rejects_invalid_joint_cycle_limits(value):
    with pytest.raises(ValueError, match="max_cycles must be a positive integer"):
        rx.coembed("CC", "CC", max_cycles=value)


def test_ethane_pair_passes_individual_step_and_cycle_limits_to_optimizer(monkeypatch):
    native = coembedding._coupled_uff
    calls = []

    def capture(*args, **kwargs):
        calls.append((kwargs["max_iters"], kwargs["max_cycles"]))
        return native(*args, **kwargs)

    monkeypatch.setattr(coembedding, "_coupled_uff", capture)
    result = rx.coembed("CC", "CC", n=1, seed=42, joint_max_iters=7, max_cycles=3)

    assert len(result) == 1
    assert calls
    assert all(limits == (7, 3) for limits in calls)


def test_methyl_chloride_pairs_and_rdkit_conformers_are_ordered_by_combined_score():
    pair = rx.coembed(
        "[C:1]([H:2])([H:3])([H:4])[Cl:6].[F-:5]",
        "[C:1]([H:2])([H:3])([H:4])[F:5].[Cl-:6]",
        charge=-1,
        n=5,
        stereo="free",
        threads=1,
    )
    scores = pair.scores
    assert len(pair) > 1
    assert [scores[cid] for cid in pair.ids] == sorted(scores.values())
    assert pair.reactant.ids == pair.product.ids == pair.ids
    assert list(scores) == pair.ids
    for endpoint, first in zip((pair.reactant, pair.product), pair[0], strict=True):
        mol = endpoint.mol
        assert [conf.GetId() for conf in mol.GetConformers()] == pair.ids
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), first.GetConformer().GetPositions())


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ethane_pair_warns_when_too_few_starting_conformers_are_requested(caplog, n):
    with caplog.at_level("WARNING", logger="rxembed"):
        pair = rx.coembed("CC", "CC", n=n, seed=42, threads=1, seed_relax_iters=0, joint_max_iters=1, max_cycles=1)
    notices = [r for r in caplog.records if "we recommend more conformers" in r.getMessage()]
    assert len(notices) == (1 if n <= 3 else 0)
    assert pair.ids
    assert all(window == (0.0, 0.0) for window in pair.constraints.distances.values())


@pytest.mark.parametrize("name", ["max_iters", "initial_relaxation_iters", "union_seed", "ts_weight", "ts_atoms"])
def test_coembed_rejects_removed_control_names(name):
    with pytest.raises(TypeError, match="unexpected keyword"):
        rx.coembed("CC", "CC", **{name: 1})
