"""`engine/conformers.py`: the `Conformers` result, its relax up the stiffness ladder and fresh-seed replacement."""

import dataclasses

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import SetBondLength

import rxembed as rx
from rxembed.engine.acceptance import CORE_GATES, Failure, Gate
from rxembed.engine.conformers import Conformers
from rxembed.engine.seeding import Seeds
from rxembed.geometry.bounds import Sampling
from rxembed.geometry.constraints import Constraints, resolve_core
from rxembed.identity.species import Species
from rxembed.metal.centres import engine_graph
from tests.geometry_fixtures import embedded
from tests.metal_fixtures import conformer_species, seeds

FACE = next(gate for gate in CORE_GATES if gate.name == "metal_face")


def _organic(smiles, n, seed=7):
    """Return `n` RDKit conformers of `smiles` with explicit H as an organic `Conformers`."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(rdDistGeom.EmbedMultipleConfs(mol, n, randomSeed=seed))
    return Conformers(mol, ids, species=Species(mol=Chem.Mol(mol)))


def test_result_owns_its_coordinates():
    """Construction copies the Mol, ids and energies; a selection is a new result whose relax leaves its parent."""
    mol = embedded("CCO")
    mol.GetConformer().SetId(27)
    ids, energies, before = [27], {27: -77.0}, mol.GetConformer(27).GetPositions()
    owned = Conformers(mol, ids, species=Species(mol=Chem.Mol(mol)), energies=energies)
    mol.GetConformer(27).SetAtomPosition(0, (99.0, 99.0, 99.0))
    ids.clear()
    energies.clear()
    assert (owned.ids, owned.energies) == ([27], {27: -77.0})
    np.testing.assert_array_equal(owned.mol.GetConformer(27).GetPositions(), before)

    ether = _organic("CCOCCC", 3)
    parent = {cid: ether.mol.GetConformer(cid).GetPositions() for cid in ether.ids}
    child, sibling = ether[:2], ether[0]
    assert (child.ids, sibling.ids) == (ether.ids[:2], ether.ids[:1])
    assert [conf.GetId() for conf in child.mol.GetConformers()] == child.ids
    with pytest.raises(ValueError, match="not one of this result's ids"):
        child.xyz(ether.ids[-1])
    child.minimize()
    assert not np.array_equal(child.mol.GetConformer(ether.ids[0]).GetPositions(), parent[ether.ids[0]])
    np.testing.assert_array_equal([ether.mol.GetConformer(cid).GetPositions() for cid in parent], list(parent.values()))
    np.testing.assert_array_equal(sibling.mol.GetConformer(ether.ids[0]).GetPositions(), parent[ether.ids[0]])
    assert ether.energies == sibling.energies == {}


def test_measure_reads_mean_range_and_count():
    """A distance, an angle or a dihedral over two conformers."""
    butane = Chem.MolFromSmiles("CCCC")
    for points in (
        [(0, 0, 0), (1.5, 0, 0), (1.5, 1.5, 0), (1.5, 1.5, 1.5)],
        [(0, 0, 0), (1.7, 0, 0), (1.7, 1.5, 0), (1.7, 1.5, -1.5)],
    ):
        conformer = Chem.Conformer(4)
        conformer.SetPositions(np.array(points, float))
        butane.AddConformer(conformer, assignId=True)
    conformers = Conformers(butane, [0, 1], species=Species(mol=butane))
    rows = {(0, 1): (1.6, 1.5, 1.7), (0, 1, 2): (90.0, 90.0, 90.0), (0, 1, 2, 3): (0.0, -90.0, 90.0)}
    assert {atoms: conformers.measure(atoms) for atoms in rows} == {
        atoms: pytest.approx({"mean": mean, "min": low, "max": high, "n": 2})
        for atoms, (mean, low, high) in rows.items()
    }


def test_xyz_writers_write_each_tracked_conformer_or_kept_step(tmp_path):
    ether = _organic("CCOC", 2)
    relaxed = rx.embed("CCO", fix={(0, 2): 2.6}, n=1, seed=1, trajectory=True)
    written, expected = {}, {}
    for name, conformers, write, frames in (
        ("dump", ether, ether.dump, 2),
        ("dump_trajectory", relaxed, relaxed.dump_trajectory, relaxed.trajectory.GetNumConformers()),
    ):
        written[name] = (frames > 1, len(write(tmp_path / f"{name}.xyz").read_text().splitlines()))
        expected[name] = (True, frames * (conformers.mol.GetNumAtoms() + 2))
    assert written == expected


def test_minimize_keeps_an_input_uff_cannot_type():
    """UFF has no type for this boron network, so neither a plain nor a constrained relax moves it or its energy."""
    kept = {}
    for name, distances in (("plain", {}), ("constrained", {(0, 2): (1.0, 3.0)})):
        mol = embedded("C=[B]B")
        cons = Constraints(distances=distances)
        conformers = Conformers(mol, [0], cons, species=Species(mol=mol), energies={0: -99.0})
        conformers.minimize(max_iters=10)
        unmoved = np.array_equal(conformers.mol.GetConformer().GetPositions(), mol.GetConformer().GetPositions())
        kept[name] = (conformers.energies, conformers.unrelaxed, unmoved)
    assert kept == dict.fromkeys(kept, ({0: -99.0}, [0], True))


def test_stiffness_ladder_restarts_each_rung_from_the_seed(monkeypatch):
    """A relax that always lands on the other face of a vinyl fluoride climbs every rung, then returns to its seed."""
    conformers = seeds(rx.isomers("[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH]->1F", geometry="SPL")[0], 1, seed=1)
    cid = conformers.ids[0]
    seed = conformers.mol.GetConformer(cid).GetPositions()
    starts, rungs = [], []

    def onto_the_other_face(mol, _cons, *, stiffness, max_iters, conf_ids, record, **_kwargs):
        if max_iters:
            starts.append(mol.GetConformer(cid).GetPositions())
            rungs.append(stiffness)
            mol.GetConformer(cid).SetPositions(starts[-1] * [-1, 1, 1])
            record.statuses[cid] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr("rxembed.engine.conformers.restrained_uff", onto_the_other_face)
    conformers.gates = (FACE,)
    conformers.minimize(max_iters=10)

    assert len(rungs) > 1
    assert rungs == sorted(rungs)
    np.testing.assert_array_equal(starts, [seed] * len(starts))
    assert (conformers.ids, conformers.unrelaxed, conformers.relax_failures[cid].kind) == ([cid], [cid], "metal_face")
    np.testing.assert_array_equal(conformers.mol.GetConformer(cid).GetPositions(), seed)


def _helium_at(xs):
    """Return a helium pair with a conformer of atom 0 at each ``{id: x}`` on the x axis, atom 1 3 A further."""
    mol = Chem.MolFromSmiles("[He].[He]")
    for cid, x in xs.items():
        conformer = Chem.Conformer(2)
        conformer.SetId(cid)
        conformer.SetPositions(np.array([[x, 0.0, 0.0], [x + 3.0, 0.0, 0.0]]))
        mol.AddConformer(conformer, assignId=False)
    return mol


def _x(conformers, cid):
    """Return where conformer `cid` puts atom 0 on the x axis."""
    return conformers.mol.GetConformer(int(cid)).GetPositions()[0, 0]


def _helium_pairs(monkeypatch, owned, verdict, *, energies=None, cons=None):
    """Return helium pairs at x = `owned[cid]` and the sampling seed of each fresh batch `minimize` draws.

    Each batch seeds pairs at x = 0..7 as conformers 0..7, which relax 0.5 A along x to energies `energies[cid]`
    (0 else), all but conformer 4 converged. The one gate fails what `verdict(x)` gives for atom 0's x.
    """
    pair = _helium_at({})
    owner = Conformers(
        _helium_at(owned), list(owned), cons or Constraints(), species=Species(mol=pair), sampling=Sampling(seed=1)
    )
    drawn = []

    def fresh(*args, **_kwargs):
        drawn.append(args[4].seed)
        return Seeds(mol=_helium_at(dict(enumerate(range(8)))), ids=list(range(8)), selected=False)

    def uff(mol, _cons, *, conf_ids, max_iters, record, **_kwargs):
        for cid in conf_ids if max_iters else ():
            conformer = mol.GetConformer(int(cid))
            conformer.SetPositions(conformer.GetPositions() + np.array([0.5, 0.0, 0.0]))
        record.statuses.update({cid: int(cid == 4) for cid in conf_ids})
        return np.array([(energies or {}).get(cid, 0.0) for cid in conf_ids])

    monkeypatch.setattr("rxembed.engine.conformers.seed_conformers", fresh)
    monkeypatch.setattr("rxembed.engine.conformers.restrained_uff", uff)
    monkeypatch.setattr(
        Conformers,
        "gates",
        (Gate("stub", lambda confs, cid: verdict(_x(confs, cid))),),
    )
    return owner, drawn


def test_replacement_adopts_the_lowest_settled_finite_scores(monkeypatch):
    """A survivor is untouched. Seven pairs past x = 100 fail; one fresh batch replaces them, its x = 0 seed failing
    too. Only settled finite scores rank, lowest first: conformer 4 is unconverged and the rest score nan or -inf.
    """
    failed = list(range(20, 27))
    energies = {0: -100.0, 1: 5.0, 2: 1.0, 3: 1.0, 4: -1000.0, 5: np.nan, 6: -np.inf, 7: np.nan, 27: -77.0}
    invalid = Failure("physical_geometry", "invalid lowest energy")
    owner, drawn = _helium_pairs(
        monkeypatch,
        {**{cid: 100.0 + cid for cid in failed}, 27: 7.0},
        lambda x: invalid if x == 0.5 or x > 100 else None,
        energies=energies,
    )

    owner.minimize(max_iters=1)

    assert len(drawn) == 1
    assert owner.ids == [*failed, 27]
    assert [_x(owner, cid) for cid in [*failed, 27]] == [2.5, 3.5, 1.5, 4.5, 5.5, 6.5, 7.5, 7.5]
    assert (owner.energies[20], owner.energies[27], owner.unrelaxed) == (1.0, -77.0, [23])


def test_replacement_stops_on_a_repeat_or_crossed_seating(monkeypatch):
    """A torn bond twice has no third batch to tear it otherwise; a relax onto another shape whose seed crosses its
    seating is another arrangement after one batch. Nothing replaces the failed pair either way.
    """
    tear = Failure("bonding", "bond He0-He1 stretched", atoms=(0, 1))
    shape = Failure("coordination_shape", "He0 relaxes from TET to SPL", atoms=(0,))
    crossed = Failure("seating_crossed", "He0 donors swap sites", atoms=(0,))
    found, expected = {}, {}
    for name, verdict, cons, batches in (
        ("torn twice", lambda _x: tear, None, 2),
        ("crossed", lambda x: shape if x % 1 else crossed, Constraints(distances={(0, 1): (3.0, 3.0)}), 1),
    ):
        owner, drawn = _helium_pairs(monkeypatch, {5: 5.0}, verdict, cons=cons)
        owner.minimize(max_iters=1)
        found[name] = (drawn, owner.ids)
        expected[name] = ([owner.sampling.retry("replacement", batch).seed for batch in range(1, batches + 1)], [])
    assert found == expected


def test_result_given_new_constraints_is_gated_again():
    chloroethane = embedded("CCCl")
    SetBondLength(chloroethane.GetConformer(), 1, 2, 2.4)
    held = Conformers(
        chloroethane, [0], resolve_core(chloroethane, fix={(1, 2): 2.4}).cons, species=Species(mol=chloroethane)
    )
    held.minimize(max_iters=0)
    moved = dataclasses.replace(held, cons=resolve_core(chloroethane, fix={(1, 2): 2.9}).cons)

    with pytest.raises(rx.EmbeddingError, match="not held"):
        moved.minimize(max_iters=0)


@pytest.fixture(scope="module")
def all_cis_cobalt():
    """Return one seed of Delta all-cis Co(NH3)2(OH2)2Cl2 (N0 Co1 N2 O3 O4 Cl5 Cl6) as `Conformers`."""
    return seeds(rx.isomers("N->[Co+3](<-N)(<-O)(<-O)(<-[Cl-])<-[Cl-]", geometry="OCT").select(hand="delta"), 1, seed=7)


def test_minimize_reflects_a_mirrored_free_hand_back(all_cis_cobalt):
    """Same-class donors in each other's slots keep the Delta hand their best seating reads; any mirror is reflected."""
    species = all_cis_cobalt.species
    graph = engine_graph(species.mol)
    base = all_cis_cobalt.mol.GetConformer(all_cis_cobalt.ids[0]).GetPositions()
    slots = [site.atoms[0] for site in species.centres[0].vertices]
    hands = {}
    for name, order, mirror in (
        ("mirrored", range(6), -1),
        ("traded", [0, 4, 2, 5, 1, 3], 1),
        ("traded and mirrored", [0, 4, 2, 5, 1, 3], -1),
    ):
        pos = base.copy()
        for slot, source in enumerate(order):  # each donor moves with its hydrogens
            target, donor = slots[slot], slots[source]
            pos[[target, *(n.GetIdx() for n in graph.GetAtomWithIdx(target).GetNeighbors())]] = base[
                [donor, *(n.GetIdx() for n in graph.GetAtomWithIdx(donor).GetNeighbors())]
            ]
        mol = Chem.Mol(graph)
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
        mol.GetConformer().SetPositions(pos * [mirror, 1, 1])

        relaxed = Conformers(mol, [0], species=species).minimize(max_iters=0)
        hands[name] = conformer_species(relaxed.mol).centres[0].hand
    assert hands == dict.fromkeys(hands, "delta")


def test_planar_nickel_cleanup_keeps_pulls_bonds_and_species(tmp_path):
    """Relaxing an embedded trigonal-planar Ni diamine again keeps each held angle; its xyz reads as one species."""
    embedded_ni = rx.embed(rx.isomers("N[Ni]1NCCN1", geometry="trigonal_planar")[0], n=1, seed=42)
    pulls = dict(embedded_ni.cons.pulls)
    ens = embedded_ni.minimize()
    assert (ens.n, ens.unrelaxed, dict(ens.cons.pulls)) == (1, [], pulls)
    checks = ens.check()
    assert {name: check.ok for name, check in checks.items()} == dict.fromkeys(checks, True)
    path = tmp_path / "planar_diamine.xyz"
    Chem.MolToXYZFile(ens.mol, str(path))
    read = rx.read_xyz(
        str(path), charge=Chem.GetFormalCharge(ens.mol), connectivity="rdkit", bond_orders="xyz2mol", fallback=False
    )
    assert rx.cxsmiles(read) == rx.cxsmiles(ens.mol)
    edges = [{frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()} for mol in (read, ens.mol)]
    assert edges[0] == edges[1]
