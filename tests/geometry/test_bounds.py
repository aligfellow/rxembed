"""`geometry/bounds.py`: the embed settings, the edited seed bounds and the native seed batches drawn from them."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields, replace
from typing import Any
from unittest import mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.geometry import bounds as bnd
from rxembed.geometry.constraints import Constraints, CoplanarRow, resolve_core
from rxembed.graph import surrogate_all_metals


def _graph(smiles):
    """Return `smiles` with explicit H and no conformer."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _settings(native):
    """Return a native parameter object's settings, without the bounds matrix seeding gives it."""
    settings = json.loads(rdDistGeom.EmbedParametersToJSON(native))
    settings.pop("boundsMatrix", None)
    return settings


# Per field: a value other than its default, and a refused value with the remedy its message names (None where the
# field takes any value). A new field adds its row here.
EMBED_PARAMS: dict[str, tuple[Any, Any, Any]] = {
    "prune_rms": (0.5, None, None),
    "knowledge": (False, "no", "knowledge must be True, False or None"),
    "native": (None, "KDG", "native must be an RDKit EmbedParameters object"),
    "coplanar_14": (False, 0, "coplanar_14 must be True or False"),
    "metal_floor_relief": (False, 0, "metal_floor_relief must be True or False"),
    "donor_orientation": (False, 0, "donor_orientation must be True or False"),
    "conjugation": (False, 0, "conjugation must be True or False"),
    "lengths": ("input", "measured", "expected one of"),
}


def test_embed_params_refuse_bad_values_or_owned_native_settings():
    """A native RDKit object leaves seed, threads and pruning to the verb, and its useBasicKnowledge to `knowledge`."""
    assert EMBED_PARAMS.keys() == {field.name for field in fields(bnd.EmbedParams)}
    for name, (_value, refused, remedy) in EMBED_PARAMS.items():
        if refused is not None:
            with pytest.raises((TypeError, ValueError), match=re.escape(remedy)):
                bnd.EmbedParams(**{name: refused})
    natives = (("randomSeed", 7, "seed="), ("numThreads", 2, "threads="), ("pruneRmsThresh", 0.4, "prune_rms"))
    for name, value, message in natives:
        native = rdDistGeom.KDG()
        setattr(native, name, value)
        with pytest.raises(ValueError, match=rf"native\.{name}={value} is set; pass .*{message}"):
            bnd.EmbedParams(native=native)
    with pytest.raises(ValueError, match="useBasicKnowledge"):
        bnd.EmbedParams(native=rdDistGeom.ETDG(), knowledge=True)
    assert bnd.EmbedParams(native=rdDistGeom.KDG(), knowledge=True).knowledge
    assert bnd.EmbedParams(**{name: value for name, (value, _refused, _remedy) in EMBED_PARAMS.items()})


def test_sampling_refuses_foreign_params_or_an_unspreadable_seed():
    """RDKit seeds conformer i with seed * (i + 1), so a seed below 1 would draw identical conformers."""
    with pytest.raises(TypeError, match=r"wrap an RDKit object as EmbedParams\(native=\.\.\.\)"):
        bnd.Sampling(params=rdDistGeom.KDG())
    seeds: tuple[Any, ...] = (0, True, 2.5)
    for seed in seeds:
        with pytest.raises(ValueError, match="give a seed >= 1"):
            bnd.Sampling(seed=seed)


def test_retry_seed_is_shared_by_no_other_stage_or_seed():
    retries = [bnd.Sampling(seed=s).retry(t, i).seed for s in range(1, 5) for t in ("seed", "swap") for i in (1, 2, 3)]
    assert len(set(retries) | set(range(1, 5))) == len(retries) + 4
    assert bnd.Sampling(seed=7).retry("seed batch", 0).seed == 7


def test_resolve_sampling_keeps_each_unset_settings_default():
    params = bnd.EmbedParams(knowledge=False)
    assert bnd.resolve_sampling(params, 5, None) == bnd.Sampling(params=params, seed=5)
    assert bnd.resolve_sampling(None, None, 2) == bnd.Sampling(threads=2)


def test_seeding_leaves_a_native_object_as_set_even_on_timeout(monkeypatch):
    native = rdDistGeom.srETKDGv3()
    native.useRandomCoords = True
    sampling = bnd.Sampling(params=bnd.EmbedParams(native=native), seed=42, threads=1)
    native.randomSeed, native.numThreads, native.pruneRmsThresh = 7, 2, 0.4  # the record checked them; seeding restores
    before = _settings(native)
    assert bnd.seed_coordinates(_graph("CCCC"), Constraints(), 1, sampling) == [0]
    assert _settings(native) == before
    timeout = mock.Mock(return_value=[-1])
    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", timeout)
    with pytest.raises(TimeoutError, match=r"native RDKit embedding exceeded timeout=0s; raise native\.timeout"):
        bnd.seed_coordinates(_graph("CCCC"), Constraints(), 10, sampling)
    assert ([call.args[1] for call in timeout.call_args_list], _settings(native)) == ([10], before)  # its one search


def test_each_seed_phase_draws_only_the_shortfall_until_timeout(monkeypatch):
    """AIO, then legacy refinement and AIO from random starts. A threaded RDKit embed can report a conformer it never
    placed, every atom at the origin; legacy refinement fails as for tmQMg NAKXIG (an eta2 alkyne on osmium)."""
    assert bnd.seed_coordinates(_graph("[I-]"), Constraints(), 1, bnd.Sampling(seed=1)) == [0]  # a lone atom is placed
    native, calls, placed = rdDistGeom.EmbedMultipleConfs, [], {}

    def draw(mol, count, p):
        calls.append((count, p.useLegacyImplementation, p.useRandomCoords, p.clearConfs, p.randomSeed))
        if len(calls) == 2:  # legacy refinement finds nothing
            return []
        if len(calls) == 4:  # the next batch times out
            return [-1]
        ids = list(native(mol, count, p))
        if len(calls) == 1:
            mol.GetConformer(ids[1]).SetPositions(np.zeros((mol.GetNumAtoms(), 3)))
            placed.update({new: mol.GetConformer(old).GetPositions() for new, old in enumerate((ids[0], ids[2]))})
        return ids

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", draw)
    mol = _graph("CCO")
    sampling = bnd.Sampling(params=bnd.EmbedParams(prune_rms=-1), seed=42, threads=1)
    assert bnd.seed_coordinates(mol, Constraints(), 3, sampling) == [0, 1, 2]
    assert calls == [(3, False, False, True, 42), (1, True, True, False, 42), (1, False, True, False, 42)]
    np.testing.assert_array_equal([mol.GetConformer(cid).GetPositions() for cid in placed], list(placed.values()))
    assert mol.GetConformer(2).GetPositions().any()
    with pytest.raises(TimeoutError, match="exceeded timeout=0s"):
        bnd.seed_coordinates(mol, Constraints(), 2, sampling)
    assert len(calls) == 4  # no later phase


def test_ligand_reach_is_torsion_free_within_each_fragment():
    aromatic, saturated, chain = (Chem.MolFromSmiles(s) for s in ("Pc1ccccc1P", "PC1CCCCC1P", "PCCCCP.O"))
    native = [
        rdDistGeom.GetMoleculeBoundsMatrix(m, set14bounds=True, set15bounds=False) for m in (aromatic, saturated, chain)
    ]
    assert bnd.ligand_reach(aromatic)[0, 7] == pytest.approx(native[0][0, 7])  # across an aromatic ring bond
    assert bnd.ligand_reach(saturated)[0, 7] > native[1][0, 7]  # not RDKit's sp2 cis template
    free = bnd.ligand_reach(chain)
    assert free[0, 5] > native[2][0, 5]
    assert (free[1, 0], free[0, 1], free[6, 0], free[0, 6]) == (native[2][1, 0], native[2][0, 1], 0.0, math.inf)
    matrices = np.ones((2, 4, 4))
    matrices[:, 1, 0], matrices[:, 0, 1] = (1.0, 3.0), (math.inf, 3.1)  # unbounded, or no unit triangle closes it
    assert [bnd.chain_upper(matrix, (0, 1, 2, 3)) for matrix in matrices] == [math.inf, math.inf]


def test_fragment_contacts_keep_free_components_within_reach():
    """Ethanol C0 C1 O2 and methane C3: a requested distance joins them, and two pinned components keep their gap."""
    mol = _graph("CCO.C")
    bm = bnd.bounds_matrix(mol)
    free = bnd.fragment_contacts(mol, Constraints(), bm)
    assert free.keys() == {(0, 3), (1, 3), (2, 3)}
    assert all(lo == bm[j][i] < hi < 10.0 for (i, j), (lo, hi) in free.items())
    assert bnd.fragment_contacts(mol, Constraints(frozen={0}), bm) == free
    joined = (Constraints(distances={(0, 3): (3.0, 4.0)}), Constraints(frozen={0, 3}))
    assert [bnd.fragment_contacts(mol, cons, bm) for cons in joined] == [{}, {}]
    assert bnd.fragment_contacts(_graph("CO"), Constraints(), bnd.bounds_matrix(_graph("CO"))) == {}
    assert bnd.seed_bounds(mol, Constraints(), bnd.Sampling())[0, 3] <= free[(0, 3)][1]


def test_coordination_reach_writes_targets_over_a_native_copy():
    """M0 holds D1 and D2 cis, D4 by a stated D1...D4 span, and a haptic centroid X3 left to acceptance."""
    native, cis = np.triu(np.full((5, 5), math.inf), 1), (85.0, 95.0)
    held = {(0, 1): (2.0, 2.1), (0, 2): (2.0, 2.1), (0, 3): (1.6, 1.7), (0, 4): (2.0, 2.1), (1, 4): (3.5, 3.6)}
    angles = {(1, 0, 2): cis, (1, 0, 4): cis, (1, 0, 3): (100.0, 120.0)}
    cons = Constraints(metals={0}, distances=held, angles=angles, floors={(1, 2): 2.5}, haptic={3: (5, 6)})
    reach = bnd.coordination_reach(cons, native)
    assert reach[1, 0] < 2.0  # a metal window widened by its acceptance slack
    assert reach[0, 1] > 2.1
    assert (reach[4, 1], reach[1, 4]) == (3.5, 3.6)  # a requested span owns its cell over the angle row
    assert (reach[3, 0], reach[0, 3], reach[3, 1], reach[1, 3]) == (0, math.inf, 0, math.inf)  # no centroid rows
    assert 2.5 < reach[2, 1] < 2.9 < reach[1, 2] < 3.2  # the cis angle spans D1...D2 above its floor
    assert np.isinf(native[np.triu_indices(5, 1)]).all()


def test_seed_bounds_read_a_held_metal_as_its_surrogate_carbon():
    """A carried nitrile-iron(II) dative bond gives the engine graph's bounds, as does the bare ion."""
    real, sampling = _graph("CC#N.[Fe+2]"), bnd.Sampling()  # C0 C1 N2 Fe3
    carrier = Chem.RWMol(real)
    carrier.AddBond(2, 3, Chem.BondType.DATIVE)
    carrier.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(carrier)
    cons = Constraints(metals={3}, distances={(2, 3): (1.9, 1.95)})
    bounds = [bnd.seed_bounds(mol, cons, sampling) for mol in (carrier.GetMol(), surrogate_all_metals(carrier), real)]
    np.testing.assert_array_equal(bounds[1:], [bounds[0]] * 2)


def test_seed_bounds_repair_a_crossed_window_not_a_contradiction(caplog):
    """Propane's C0...C2 cannot exceed its two bonds' reach of about 3.08 A: 3.1 is repaired, 5.0 contradicts."""
    propane, sampling = _graph("CCC"), bnd.Sampling()
    requested = resolve_core(propane, fix={(0, 2): 3.1}).cons
    seen = {}
    for cons, level in ((requested, "WARNING"), (Constraints(distances={(0, 2): (3.08, 3.12)}), "DEBUG")):
        caplog.clear()
        with caplog.at_level("DEBUG", logger="rxembed.bounds"):
            bm = bnd.seed_bounds(propane, cons, sampling)
        levels = [r.levelname for r in caplog.records if "C0-C2 needed smoothing" in r.message]
        seen[level] = (levels, bm[2, 0] <= bm[0, 2] < 3.12)
    assert seen == {"WARNING": (["WARNING"], True), "DEBUG": (["DEBUG"], True)}
    with pytest.raises(ValueError, match="bounds contradict each other; loosen fix=/constrain="):
        bnd.seed_coordinates(propane, resolve_core(propane, fix={(0, 2): 5.0}).cons, 1, sampling)


def test_matrix_only_switches_drop_their_terms_from_seeds_alone():
    """Four disconnected carbons held as a chain, an anti cap on C0-C1-C2-C3 and a relieved floor on C0...C3's H."""
    mol, chain = _graph("C.C.C.C"), {(0, 1): (1.4, 1.4), (1, 2): (1.4, 1.4), (2, 3): (1.4, 1.4), (1, 3): (2.4, 2.4)}
    cap = CoplanarRow(kind="ligand_ez", atoms=(0, 1, 2, 3), anchor=180.0, cap=45.0)
    full = Constraints(distances=chain, angles={(0, 1, 2): (118.0, 122.0)}, coplanar=[cap], dg_floors={(0, 16): 2.0})
    variants = (replace(full, coplanar=(), dg_floors={}), replace(full, dg_floors={}), replace(full, coplanar=()))
    neither, capped, relieved = (bnd.seed_bounds(mol, cons, bnd.Sampling()) for cons in variants)
    assert [np.array_equal(matrix, neither) for matrix in (capped, relieved)] == [False, False]
    off = bnd.Sampling(params=bnd.EmbedParams(coplanar_14=False, metal_floor_relief=False))
    np.testing.assert_array_equal(bnd.seed_bounds(mol, full, off), neither)


def test_seed_count_grows_with_rotors_or_constraint_in_clamps():
    rigid, mid, floppy = _graph("CCO"), _graph("C" * 25), _graph("C" * 60)
    assert 10 < bnd.seed_count(rigid) == bnd.seed_count(_graph("CC")) < bnd.seed_count(mid) < bnd.seed_count(floppy)
    assert bnd.seed_count(floppy) == bnd.seed_count(_graph("C" * 120))
    assert bnd.seed_count(rigid, constrained=True) > bnd.seed_count(rigid)
    assert bnd.seed_count(mid, constrained=True) > bnd.seed_count(mid)


def test_probe_embeds_a_copy_with_each_lone_ion_at_the_origin():
    pair = _graph("[Na+].[Cl-]")
    before = pair.ToBinary(Chem.PropertyPickleOptions.AllProps)
    assert not bnd.probe_conformer(pair, 42).GetConformer().GetPositions().any()
    assert pair.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_seeds_return_only_real_atoms_around_a_haptic_centroid():
    """Cr6 sits 1.65 A over benzene's centroid X13, held 1.40 A from each ring carbon and present only in the seed."""
    mol, ring = _graph("c1ccccc1.[Cr]"), tuple(range(6))
    held = {(6, 13): (1.6, 1.7)} | {(atom, 13): (1.38, 1.42) for atom in ring}
    cons = Constraints(metals={6}, haptic={13: ring}, distances=held, dg_floors={(atom, 6): 2.0 for atom in ring})
    seeds = {}
    for cid in bnd.seed_coordinates(mol, cons, 2, bnd.Sampling(seed=1, threads=1)):
        positions = mol.GetConformer(cid).GetPositions()
        over = np.linalg.norm(positions[6] - positions[list(ring)].mean(axis=0))
        seeds[cid] = (positions.shape, over < min(2.2, *(np.linalg.norm(positions[6] - positions[a]) for a in ring)))
    assert seeds == dict.fromkeys(seeds, ((13, 3), True))  # a raw seed


def test_pair_preferences_steer_rdkits_seed_refinement():
    spans = []
    for cpci in ({(0, 1): -1.0}, {}, {(0, 1): 1.0}):  # attract, leave, repel the two methane carbons
        methanes = _graph("C.C")
        held = Constraints(distances={(0, 1): (3.0, 8.0)}, seed_cpci=cpci)
        ids = bnd.seed_coordinates(methanes, held, 8, bnd.Sampling(seed=3, threads=1))
        positions = [methanes.GetConformer(cid).GetPositions() for cid in ids]
        spans.append(np.mean([np.linalg.norm(p[0] - p[1]) for p in positions]))
    assert spans[0] < spans[1] < spans[2]
