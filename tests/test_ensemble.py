"""`ensemble.py`: the Ensemble verbs and their records, the sets that hold several results, and `wrap`."""

import logging
import re
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms

import rxembed as rx
from rxembed.engine.acceptance import Gate
from rxembed.geometry.constraints import resolve_core
from rxembed.geometry.relax import Coupling, restrained_uff
from rxembed.identity.reading import read_species
from rxembed.identity.species import Species
from rxembed.identity.stereoisomers import StereoRequest
from rxembed.perception.changes import Changes, stereo_filter
from rxembed.workflow.calculators import Calculator
from tests.conftest import EXAMPLES_DIR
from tests.extras import needs
from tests.geometry_fixtures import embedded, mirrored, placed
from tests.metal_fixtures import ferrocene, on_its_vertices, pd2cl6

EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # covalent notation for the square-planar en chelate
DIAMMINE_PD = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"
PT_PYRIDINE = "[Cl-]->[Pt+2](<-[Cl-])(<-n1ccccc1)<-n1ccccc1"
ONE_HARTREE = -627.509474  # kcal/mol, the energy a -1 Eh calculator gives
HELD_ETHANE = {(0, 1): (1.6, 1.6)}


class OneGate(rx.Ensemble):
    """Hold conformers judged only by the gate rows an instance is given."""

    gates: tuple[Gate, ...] = ()


class Model(Calculator):
    """Give each conformer -1 Eh, or `hartree[id]`, raising one that is an exception; optimising applies `moves[id]`,
    an edit of a copy of that conformer, and returns every other conformer as it is.
    """

    def __init__(self, hartree=None, moves=None):
        self.hartree, self.moves = hartree or {}, moves or {}

    def energy(self, mol, conf_id=-1):
        value = self.hartree.get(conf_id, -1.0)
        if isinstance(value, Exception):
            raise value
        return value

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        conformer = Chem.Conformer(mol.GetConformer(conf_id))
        if conf_id in self.moves:
            self.moves[conf_id](conformer)
        return conformer.GetPositions(), self.energy(mol, conf_id)


def conformers(smiles, n, seed=7):
    """Return `smiles` with explicit H and `n` RDKit conformers, numbered from 0."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert list(rdDistGeom.EmbedMultipleConfs(mol, n, randomSeed=seed)) == list(range(n))
    return mol


def water_dimers(*shifts):
    """Return a water dimer H-bonded O0-H2...O1, one conformer per shift of the second water along the O...O axis."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O.O"))
    bonded = np.array([(0, 0, 0), (2.8, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0), (3.45, 0.72, 0), (3.45, -0.72, 0)])
    for shift in shifts:
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.SetPositions(bonded + np.outer([0, 1, 0, 0, 1, 1], (shift, 0, 0)))
        mol.AddConformer(conformer, assignId=True)
    return mol


def pulled(ens, cid, atoms, anchor, distance):
    """Return a copy of `ens` with `atoms` moved as one along anchor->atoms[0], until atoms[0] is `distance` A out."""
    mol = ens.mol
    xyz = mol.GetConformer(cid).GetPositions()
    ray = xyz[atoms[0]] - xyz[anchor]
    xyz[list(atoms)] += (distance / np.linalg.norm(ray) - 1) * ray
    mol.GetConformer(cid).SetPositions(xyz)
    return replace(ens, _mol=mol)


def puckered(conformer, ring):
    """Lift the fourth atom of `ring` 0.5 A out of the plane of its first three."""
    xyz = conformer.GetPositions()
    normal = np.cross(xyz[ring[1]] - xyz[ring[0]], xyz[ring[2]] - xyz[ring[0]])
    xyz[ring[3]] += 0.5 * normal / np.linalg.norm(normal)
    conformer.SetPositions(xyz)


def atom(mol, symbol):
    """Return the index of the first `symbol` atom of `mol`."""
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol)


def ethanol_pairs(mode=None):
    """Return two ethanol reaction pairs whose product moves the hydroxyl H8 1 A along x, its reacting atom."""
    reactant = conformers("CCO", 2)
    product = Chem.Mol(reactant)
    for conformer in product.GetConformers():
        conformer.SetAtomPosition(8, conformer.GetPositions()[8] + (1.0, 0, 0))
    ends = [rx.wrap(mol, energies=[2.0, 1.0]) for mol in (reactant, product)]
    for end in ends:
        end.tag.update({} if mode is None else {"nci": mode})
    return rx.Coembedded(
        reactant=ends[0],
        product=ends[1],
        coupling=Coupling(atoms=np.arange(9), weights=np.ones(9), spectators=8),
        joint_results={0: {"seed": 11}, 1: {"seed": 12}},
        optimizer_attempts=[{"seed": 11, "iterations": 40}],
    )


@pytest.fixture(scope="module")
def refused():
    """Return, by workflow gate name, an Ensemble whose conformer 0 that gate refuses and the site its reason names."""
    torn = placed("CCCC", [(0, 0, 0), (10, 0, 0), (11, 1, 0), (12.5, 1, 0)])
    cyclophane = embedded("FC1=CC2=CC=C1CCCCCCCC2", seed=1)  # 2-fluoro[8]paracyclophane shows a planar hand
    square = [(2.3, 0, 0), (0, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, -2.3, 0)]
    chloride_on = placed(rx.parse_smiles("[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-].[Cl-]"), [*square, (0, 0, 1.8)])
    hand = stereo_filter(cyclophane, StereoRequest.parse(None))
    return {
        "connectivity": (rx.Ensemble(torn, [0], species=Species(mol=torn)), "C0-C1 broken"),
        "requested_stereo": (
            rx.Ensemble(mirrored(cyclophane), [0], species=Species(mol=cyclophane), _stereo_filter=hand),
            "planar",
        ),
        "physical_geometry": (rx.Ensemble(chloride_on, [0], species=read_species(chloride_on)), "metal_overbond"),
    }


def test_each_workflow_gate_names_and_drops_what_it_refuses(refused, caplog):
    """Each gate reads its failure at its site; first in a relax, no workflow gate restores the seed, a required kind
    raises with its headline and any other drops the conformer into `discarded` with a warning.
    """
    read, expected = {}, {}
    for gate in rx.Ensemble.workflow_gates:
        ensemble, site = refused[gate.name]
        failure = gate.check(ensemble, 0)
        assert failure is not None, gate.name
        chloroethane = embedded("CCCl")
        held = OneGate(
            chloroethane, [0], resolve_core(chloroethane, fix={(1, 2): 2.4}).cons, species=Species(mol=chloroethane)
        )
        held.gates = (Gate(gate.name, lambda _ensemble, _cid, failure=failure: failure, gate.ladder, gate.required),)
        headline = re.escape(f"{failure.detail} in 1/1 rejected seeds; try ")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rxembed"):
            try:
                held.minimize(max_iters=0)
                outcome = ("warned", held.discarded) if re.search(headline, caplog.text) else caplog.text
            except rx.EmbeddingError as error:
                outcome = ("raised", None) if re.search(headline, str(error)) else str(error)
        read[gate.name] = (failure.kind, site in failure.detail, outcome, held.ids, 0 in held.relax_failures)
        raised = failure.kind in gate.required
        expected[gate.name] = (gate.name, True, ("raised", None) if raised else ("warned", [0]), [], gate.ladder)
    assert read == expected


def test_wrap_refuses_unpairable_ids_or_energies_unchanged():
    """Ids are stored conformer ids, each once; energies are finite, one per id or a dict over them."""
    methanol = embedded("CO")
    methanol.GetConformer().SetId(23)
    before = methanol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    for name, options in (
        ("absent", {"ids": [999]}),
        ("native alias", {"ids": [-1]}),
        ("float", {"ids": [23.0]}),
        ("bool", {"ids": [True]}),
        ("list", {"ids": [[23]]}),
        ("repeated", {"ids": [23, 23], "energies": [1.0, 2.0]}),
        ("energies misaligned", {"energies": [1.0, 2.0]}),
        ("energies of another id", {"energies": {99: 1.0}}),
        ("nan", {"energies": [float("nan")]}),
        ("infinity", {"energies": [float("inf")]}),
    ):
        with pytest.raises(ValueError, match=r"wrap\(ids=\) takes distinct conformer ids"):
            rx.wrap(methanol, **options)
        assert methanol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before, name


@needs("scipy", "sklearn")
def test_wrap_keeps_energies_and_geometries_through_ranking():
    """A butanol given energies by numpy ids, in their order, is final: prune, lowest and representatives rank it as
    it stands.
    """
    butanol = conformers("CCCCO", 4, seed=1)
    before = {cid: butanol.GetConformer(cid).GetPositions() for cid in range(4)}
    given = rx.wrap(butanol, ids=np.array([3, 2, 1, 0]), energies=[30.0, 20.0, 10.0, 0.0])
    assert given.ids == [3, 2, 1, 0]
    for verb in ("prune", "lowest", "representatives"):
        out = getattr(given, verb)()

        assert out.energy_kind == "given", verb
        assert out.energies == {cid: 10.0 * cid for cid in out.ids}, verb
        assert all(np.array_equal(out.mol.GetConformer(cid).GetPositions(), before[cid]) for cid in out.ids), verb


def test_wrap_minimized_scores_two_ethanes_by_uff_in_place():
    ethane = embedded("CC")
    pair = Chem.CombineMols(ethane, ethane, Chem.rdGeometry.Point3D(4.0, 0.0, 0.0))
    positions = pair.GetConformer().GetPositions()

    wrapped = rx.wrap(pair, minimized=True)

    assert wrapped.energy_kind == "UFF"
    assert list(wrapped.energies.values()) == pytest.approx(
        restrained_uff(Chem.Mol(pair), rx.Constraints(), max_iters=0)
    )
    np.testing.assert_array_equal(wrapped.mol.GetConformer().GetPositions(), positions)
    np.testing.assert_array_equal(pair.GetConformer().GetPositions(), positions)


@needs("scipy")
def test_wrap_reads_a_covalent_pt_hydride_as_a_hydride_site():
    """[H][Pt-2](Cl)(Cl)Cl, its H moved out of the PtCl3 plane: RDKit's RemoveHs drops a neutral H as routine."""
    molecule = Chem.MolFromSmiles("[H][Pt-2](Cl)(Cl)Cl", sanitize=False)
    molecule.UpdatePropertyCache(strict=False)
    square = np.array([[1.55, 0, 0], [0, 0, 0], [-2.3, 0, 0], [0, 2.3, 0], [0, -2.3, 0]])
    tilted = square.copy()
    tilted[0] = (1.1, 0, 1.1)  # only the hydride moves
    for coordinates in (square, tilted):
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        conformer.SetPositions(coordinates)
        molecule.AddConformer(conformer, assignId=True)
    ensemble = rx.wrap(molecule, energies=[0.0, 0.1])

    assert ensemble.mol.GetAtomWithIdx(0).GetFormalCharge() == -1  # the readers' [H-]->[Pt+2], atom order kept
    assert ensemble.prune(max_rmsd=0.1).ids == [0, 1], "a hydride is a coordination site, not a routine hydrogen"


def test_selection_or_calculation_is_a_new_tagged_ensemble():
    """A butanol whose torn conformer the connectivity filter dropped: every result derived from it lists that id and
    keeps the stage, so a given geometry stays final under minimize.
    """
    butanol = conformers("CCCCO", 3, seed=1)
    rdMolTransforms.SetBondLength(butanol.GetConformer(2), 0, 1, 3.0)
    ens = rx.wrap(butanol, energies=[0.0, 1.0, 2.0]).filter("connectivity")
    ens.tag["nci"] = "HB"
    for name, out, kind in (
        ("index", ens[0], "given"),
        ("slice", ens[:2], "given"),
        ("lowest", ens.lowest(1), "given"),
        ("align", ens.align(), "given"),
        ("score", ens.score(Model()), "Model"),
        ("optimize", ens.optimize(Model()), "Model"),
    ):
        assert (type(out), out.discarded, out.tag) == (rx.Ensemble, [2], {"nci": "HB"}), name
        assert out.minimize().energy_kind == kind, name
        out.tag["nci"] = "selected"
        assert ens.tag == {"nci": "HB"}, name


def test_lowest_ranks_a_conformer_without_energy_last():
    ens = rx.wrap(conformers("CCCCO", 3, seed=1), energies={1: 5.0, 2: 3.0})

    assert (ens.lowest(2).ids, ens.lowest(2).n) == ([2, 1], 2)
    assert ens.lowest(3).ids == [2, 1, 0]


def test_minimize_retries_only_unconverged_conformers():
    """Ethyl propyl ether relaxed to part convergence, free and held: the next call relaxes only the rest and scores
    them all on one objective, and a converged result stays as it is.
    """
    for constrain, budget in ((None, 80), ({(0, 5): (2.6, 3.0)}, 60)):
        params = rx.EmbedParams(prune_rms=-1)
        ens = rx.embed("CCOCCC", n=8, seed=42, threads=1, params=params, constrain=constrain, max_iters=1)
        ens.minimize(max_iters=budget)
        done = {cid: ens.mol.GetConformer(cid).GetPositions() for cid in ens.ids if cid not in ens.unrelaxed}
        assert done, constrain
        assert ens.unrelaxed, constrain

        ens.minimize()

        assert not ens.unrelaxed, constrain
        for cid, positions in done.items():
            np.testing.assert_array_equal(ens.mol.GetConformer(cid).GetPositions(), positions)
        expected = restrained_uff(ens.mol, ens.cons, max_iters=0, conf_ids=ens.ids)
        np.testing.assert_allclose([ens.energies[cid] for cid in ens.ids], expected, atol=1e-10)
        finished, energies = ens.mol.GetConformer(ens.ids[0]).GetPositions(), dict(ens.energies)
        ens.minimize(max_iters=1)
        np.testing.assert_array_equal(ens.mol.GetConformer(ens.ids[0]).GetPositions(), finished)
        assert ens.energies == energies, constrain


def test_minimize_keeps_its_stiffness_until_another_is_given():
    ens = rx.embed("CC", constrain=HELD_ETHANE, n=1, seed=42, threads=1).minimize(stiffness=1)
    before = ens.mol.GetConformer().GetPositions()
    weak = np.linalg.norm(before[0] - before[1])

    ens.minimize(stiffness=10)

    strong = np.linalg.norm(ens.mol.GetConformer().GetPositions()[0] - ens.mol.GetConformer().GetPositions()[1])
    assert abs(strong - 1.6) < abs(weak - 1.6)
    positions, energies = ens.mol.GetConformer().GetPositions(), dict(ens.energies)
    ens.minimize()
    np.testing.assert_array_equal(ens.mol.GetConformer().GetPositions(), positions)
    assert ens.energies == energies
    ens.minimize(stiffness=1)
    assert not np.array_equal(ens.mol.GetConformer().GetPositions(), positions)


def test_an_interrupted_minimize_remains_retryable():
    """A held ethane relax that raises part way, retrying unconverged conformers or retuning a converged run."""
    for pending in (True, False):
        ens = rx.embed("CC", constrain=HELD_ETHANE, n=1, seed=42, threads=1, max_iters=1)
        ens.minimize(max_iters=1 if pending else 2000)
        assert bool(ens.unrelaxed) is pending
        before = ens.mol.GetConformer().GetPositions()
        strength = 1 if pending else 10

        with pytest.raises(TypeError):
            ens.minimize(stiffness=strength, max_iters=None)
        ens.minimize(stiffness=strength)

        assert not ens.unrelaxed, pending
        assert not np.array_equal(ens.mol.GetConformer().GetPositions(), before), pending


@needs("scipy")
def test_minimize_names_its_objective_in_energy_kind():
    """Restraints add their terms to a held relax; a stand-in UFF type for Se makes the objective a surrogate."""
    rows = (
        ("butane pruned", rx.wrap(embedded("CCCC")).prune(), "UFF"),
        ("held ethane", rx.embed("CC", constrain=HELD_ETHANE, n=1, seed=42).minimize(), "restrained UFF"),
        ("selenourea lowest", rx.wrap(embedded("NC(=[Se])N")).lowest(), "surrogate UFF"),
    )
    assert {name: ens.energy_kind for name, ens, _ in rows} == {name: kind for name, _, kind in rows}


def test_minimize_keeps_calculator_energies_uff_cannot_type():
    """UFF has no type for this boron network: its scored conformers stay unrelaxed with the calculator's energies."""
    network = conformers("C=[B]B", 2)
    positions = {cid: network.GetConformer(cid).GetPositions() for cid in (0, 1)}
    scored = rx.wrap(network).score(Model({1: -2.0}))
    for name, out in (("score", scored), ("minimize", scored.minimize()), ("lowest", scored.lowest(2))):
        assert (out.unrelaxed, out.energy_kind) == ([0, 1], "Model"), name
        assert out.energies == pytest.approx({0: ONE_HARTREE, 1: 2 * ONE_HARTREE}), name
        for cid, xyz in positions.items():
            np.testing.assert_array_equal(out.mol.GetConformer(cid).GetPositions(), xyz)


def test_score_ff_scores_the_objective_without_moving_it():
    """A Pd chelate minimized at stiffness 7, then scored and optimised outside: each scores its restrained objective at
    7, whichever model named its energies.
    """
    expanded = {cid: lambda c: c.SetPositions(1.08 * c.GetPositions()) for cid in range(2)}
    species = rx.isomers(EN_PDBRCL, geometry="square_planar")[0]
    selected = rx.embed(species, n=2, seed=1, threads=1).minimize(stiffness=7.0)[:1].minimize()
    for name, source in (
        ("selected", selected),
        ("scored", selected.score(Model())),
        ("optimised", selected.optimize(Model(moves=expanded))),
    ):
        positions, energies, kind = source.mol.GetConformer().GetPositions(), dict(source.energies), source.energy_kind
        expected = restrained_uff(source.mol, source.cons, stiffness=7.0, max_iters=0, conf_ids=source.ids)

        scored = source.score("ff")

        assert (scored.ids, scored.energy_kind) == (source.ids, "restrained UFF"), name
        assert list(scored.energies.values()) == pytest.approx(expected), name
        assert (source.energies, source.energy_kind) == (energies, kind), name
        np.testing.assert_array_equal(scored.mol.GetConformer().GetPositions(), positions)
        np.testing.assert_array_equal(source.mol.GetConformer().GetPositions(), positions)


def test_external_calculation_returns_named_kcal_energies():
    """A held ethane scored keeps its geometry and relax trajectory; optimised, it takes the calculator's geometry and
    stays final under any minimize. The source keeps its own records.
    """
    source = rx.embed("CC", constrain=HELD_ETHANE, n=1, seed=42, threads=1, trajectory=True).minimize()
    cid = source.ids[0]
    xyz = source.mol.GetConformer(cid).GetPositions()
    frames = [conf.GetPositions() for conf in source.trajectory.GetConformers()]
    energies, kind = dict(source.energies), source.energy_kind
    shifted = Model(moves={cid: lambda c: c.SetPositions(np.add(c.GetPositions(), (1.0, 0, 0)))})
    for name, out, positions, steps in (
        ("score", source.score(Model()).minimize(), xyz, frames),
        ("optimize", source.optimize(shifted).minimize(stiffness=10.0), np.add(xyz, (1.0, 0, 0)), []),
    ):
        assert (out.ids, out.energy_kind) == ([cid], "Model"), name
        assert out.energies == pytest.approx({cid: ONE_HARTREE}), name
        np.testing.assert_allclose(out.mol.GetConformer(cid).GetPositions(), positions, err_msg=name)
        kept = [] if out.trajectory is None else [conf.GetPositions() for conf in out.trajectory.GetConformers()]
        np.testing.assert_array_equal(kept, steps, err_msg=name)
    assert (source.energies, source.energy_kind) == (energies, kind)
    np.testing.assert_array_equal(source.mol.GetConformer(cid).GetPositions(), xyz)


def test_uncomputable_conformer_is_dropped_into_discarded():
    """A RuntimeError or a non-finite energy on one butanol conformer drops it alone, from score or optimize."""
    ens = rx.wrap(conformers("CCCCO", 2, seed=1), energies=[1.0, 2.0])
    for verb in ("score", "optimize"):
        for failed in (RuntimeError("SCF did not converge"), float("nan")):
            out = getattr(ens, verb)(Model({0: failed}))

            assert (out.ids, out.discarded, out.energy_kind) == ([1], [0], "Model"), (verb, failed)


def test_external_calculation_raises_when_nothing_survives():
    """A calculator's bug or missing model raises, as does failing or leaving the specified sec-butanol throughout."""
    ens = rx.wrap(conformers("CC[C@H](C)O", 2, seed=1), energies=[1.0, 2.0])
    diverged = RuntimeError("SCF did not converge")
    mirror = {cid: lambda c: c.SetPositions(c.GetPositions() * (-1, 1, 1)) for cid in ens.ids}
    for verb, refine, error, message in (
        ("score", Model({0: ValueError("a bug in the calculator")}), ValueError, "a bug in the calculator"),
        ("score", Model({0: diverged, 1: diverged}), RuntimeError, "Model failed on all 2 conformer"),
        ("optimize", Calculator(), NotImplementedError, "Calculator has no geometry optimiser"),
        ("optimize", "ff", ValueError, "optimize needs a real calculator"),
        ("optimize", Model(moves=mirror), RuntimeError, "moved every conformer off the specified species"),
    ):
        with pytest.raises(error, match=message):
            getattr(ens, verb)(refine)


def test_optimize_drops_on_drop_gates_and_keeps_flagged_ones():
    """Inverting a stereocentre loses that conformer; tearing a hexane bond, bending a linear cuprate or puckering a
    pyridine keeps it for `filter` or `check` to judge.
    """
    cuprate = rx.embed(rx.isomers("[Cl-]->[Cu+]<-[Cl-]", geometry="LIN")[0], n=1, seed=1)
    platinum = rx.embed(rx.isomers(PT_PYRIDINE, geometry="SPL")[0], n=2, seed=42, threads=1)
    ring = list(Chem.GetSymmSSSR(platinum.mol)[0])
    for name, ens, moves, kept in (
        (
            "mirrored sec-butanol",
            rx.wrap(conformers("CC[C@H](C)O", 2, seed=1), energies=[1.0, 2.0]),
            {0: lambda c: c.SetPositions(c.GetPositions() * (-1, 1, 1))},
            [1],
        ),
        (
            "torn-hexane",
            rx.wrap(conformers("CCCCCC", 2, seed=1), energies=[1.0, 2.0]),
            {0: lambda c: rdMolTransforms.SetBondLength(c, 2, 3, 3.0)},
            [0, 1],
        ),
        (
            "bent-dichlorocuprate",
            cuprate,
            {cuprate.ids[0]: lambda c: rdMolTransforms.SetAngleDeg(c, 0, 1, 2, 90.0)},
            cuprate.ids,
        ),
        ("puckered Pt pyridine", platinum, {platinum.ids[0]: partial(puckered, ring=ring)}, platinum.ids),
    ):
        out = ens.optimize(Model(moves=moves))

        assert out.ids == kept, name
        assert out.discarded == [cid for cid in ens.ids if cid not in kept], name


def test_relative_gives_energies_above_the_lowest_in_kcal():
    assert rx.wrap(conformers("CCCC", 3), energies={0: 2.0, 1: 0.5}).relative() == {0: 1.5, 1: 0.0}


def test_check_reads_the_constraints_core_and_reference():
    """By default the carried constraints own their atoms and their windows go unchecked; ``constraints=`` checks that
    record's windows and core, None keeping the carried core; ``reference=`` checks the core's place, and ``frozen=``
    names another core.
    """
    collapsed = embedded("CCO", seed=1)
    collapsed.GetConformer().SetAtomPosition(0, collapsed.GetConformer().GetAtomPosition(1))
    held = rx.wrap(collapsed)
    held.cons = rx.Constraints(frozen=frozenset({0, 1, 3, 4, 5}), distances={(0, 2): (100.0, 101.0)})
    ethanol = embedded("CCO", seed=1)
    moved = Chem.Mol(ethanol)
    moved.GetConformer().SetAtomPosition(0, ethanol.GetConformer().GetPositions()[0] + (0.5, 0, 0))
    free = rx.wrap(ethanol)
    hexanol = rx.embed("CCCCCCO", fix={(0, 6): 3.5}, n=2, seed=1)
    arene = rx.isomers("c1cc[cH](->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cc1", geometry="square_planar", stereo="free")[0]
    improper = (*(n.GetIdx() for n in arene.GetAtomWithIdx(3).GetNeighbors() if n.GetSymbol() != "Pd"), 3)
    for name, ens, options, kinds in (
        ("inside core", held, {}, set()),
        ("windows checked", held, {"check_windows": True}, {"constraint"}),
        ("under its record", held, {"constraints": held.cons}, {"constraint"}),
        ("without its record", held, {"constraints": None}, set()),
        ("without a core", held, {"frozen": ()}, {"bond_length", "fusion"}),
        (
            "C off its core",
            rx.wrap(moved),
            {"constraints": rx.Constraints(frozen=frozenset(range(3))), "reference": ethanol},
            {"frozen_core"},
        ),
        ("ethanol on itself", free, {"reference": ethanol}, set()),
        ("hexanol ends fixed", hexanol, {"reference": rx.embed("CCCCCCO", n=1, seed=1).mol}, set()),
        ("arene improper fixed", rx.embed(arene, n=1, seed=42, threads=1, fix={improper: (42.0, 48.0)}), {}, set()),
        (
            "arene improper soft",
            rx.embed(arene, n=1, seed=42, threads=1, constrain={improper: (42.0, 48.0)}),
            {},
            set(),
        ),
    ):
        reports = ens.check(**options).values()

        assert all({v.kind for v in report.defects} == kinds for report in reports), name


def test_physical_gate_fails_metal_complexes_check_fails():
    """An eta1-Cp palladium keeps every conformer its minimize published; a pyridine carbon pushed onto platinum fails
    both, whatever the request fixes. A puckered benzene passes the gate, judged by its bonds, yet fails `check`.
    """
    gate = next(gate for gate in rx.Ensemble.gates if gate.name == "physical_geometry")
    cp = rx.isomers("[cH-]1(->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cccc1", geometry="square_planar", stereo="free")[0]
    platinum = rx.isomers(PT_PYRIDINE, geometry="SPL")[0]
    pt, chloride = atom(platinum, "Pt"), atom(platinum, "Cl")
    nitrogen = atom(platinum, "N")
    ring = next(list(ring) for ring in Chem.GetSymmSSSR(platinum) if nitrogen in ring)
    para = max(ring, key=lambda i: Chem.GetDistanceMatrix(platinum)[nitrogen, i])
    published = rx.embed(cp, n=4, seed=42, threads=1, params=rx.EmbedParams(donor_orientation=False)).minimize()
    loose = rx.embed(platinum, n=1, seed=42, threads=1)
    fixed = rx.embed(platinum, n=1, seed=42, threads=1, fix={(pt, chloride): 2.30})
    for name, ens, kept in (
        ("eta1 Cp Pd", published, list(published.ids)),
        ("C on Pt", pulled(loose, loose.ids[0], [para], pt, 2.0), []),
        ("C on Pt, Cl fixed", pulled(fixed, fixed.ids[0], [para], pt, 2.0), []),
    ):
        passing = [cid for cid in ens.ids if gate.check(ens, cid) is None]
        reported = [cid for cid, report in ens.check().items() if not report.defects]

        assert passing == reported == ens.filter("geometry").ids == kept, name
    hexagon = [(1.39 * np.cos(k * np.pi / 3), 1.39 * np.sin(k * np.pi / 3), 0.5 * (k == 0)) for k in range(6)]
    benzene = rx.wrap(placed("c1ccccc1", hexagon))
    assert gate.check(benzene, 0) is None
    assert benzene.filter("geometry").ids == []


def test_bond_changes_names_the_bond_a_stretched_hydroxyl_breaks():
    ethanol = conformers("CCO", 2)
    rdMolTransforms.SetBondLength(ethanol.GetConformer(1), 2, 8, 3.0)

    assert rx.wrap(ethanol).bond_changes() == {0: Changes(), 1: Changes(broken=((2, 8),))}


def test_filter_connectivity_drops_conformers_read_otherwise():
    """A bromide or an ammine moved 4 A off palladium is a broken bond; a sodium's chloride at ionic contact and a
    frustrated Lewis pair held at its transition state are bonded as their graphs say.
    """
    chelate = rx.embed(rx.isomers(EN_PDBRCL, geometry="square_planar")[0], n=2, seed=1)
    diammine = rx.embed(rx.isomers(DIAMMINE_PD, geometry="square_planar")[0], n=2, seed=1)
    nitrogen = atom(diammine.mol, "N")
    ammine = [
        nitrogen,
        *(n.GetIdx() for n in diammine.mol.GetAtomWithIdx(nitrogen).GetNeighbors() if n.GetSymbol() == "H"),
    ]
    ts = rx.read_xyz(str(EXAMPLES_DIR / "FLP_TS.xyz"), 0).GetConformer().GetPositions()
    flp = Chem.AddHs(
        Chem.MolFromSmiles("Bc1ccccc1N1CCCCC1.CC1(C)OB(c2cccs2)OC1(C)C")
    )  # the FLP with a thienyl boronate
    (borane,) = rx.match(flp, "[#5;$([#5]c);!$([#5]O)]")
    (boronate,) = rx.match(flp, "[#5;$([#5]O)]")
    ipso = rx.match(flp, "[#5;!$([#5]O)]c")[1]
    hydride = next(n.GetIdx() for n in flp.GetAtomWithIdx(borane).GetNeighbors() if n.GetAtomicNum() == 1)
    core = {borane: tuple(ts[11]), boronate: tuple(ts[32]), hydride: tuple(ts[27]), ipso: tuple(ts[1])}
    sodium = rx.embed("O->[Na+](<-O)(<-O)->[Cl]", n=3, seed=1)  # its relax rests Na-Cl on the contact wall
    for name, ens, kept in (
        ("Br off Pd", pulled(chelate, 0, [atom(chelate.mol, "Br")], atom(chelate.mol, "Pd"), 4.0), [1]),
        ("NH3 off wrapped Pd", rx.wrap(pulled(diammine, 0, ammine, atom(diammine.mol, "Pd"), 4.0).mol), [1]),
        ("Cl at Na contact", rx.wrap(sodium.mol), sodium.ids),
        ("FLP at its TS", rx.embed(flp, fix=core, n=1, seed=1, stereo="free"), [0]),
    ):
        dropped = [cid for cid in ens.ids if cid not in kept]

        assert ens.filter("connectivity").ids == kept, name
        assert ens.discarded == dropped, name


def test_filter_refuses_a_method_or_option_it_does_not_take():
    ens = rx.wrap(embedded("CCO"))
    for by, options, error, message in (
        ("rmsd", {}, ValueError, "geometric dedup is prune"),
        ("connectivity", {"frozen": (0, 1)}, TypeError, "takes no options"),
        ("geometry", {"frozne": (0, 1)}, TypeError, "unexpected geometry option"),
    ):
        with pytest.raises(error, match=message):
            ens.filter(by, **options)


@needs("scipy")
def test_compare_fits_two_conformers_in_place_under_a_tolerance():
    """An ethanol and its copy moved 5 A fit atom for atom, unmoved; an unselected id or a bad tolerance raises."""
    ethanol = embedded("CCO")
    ethanol.AddConformer(Chem.Conformer(ethanol.GetConformer()), assignId=True)
    ethanol.GetConformer(1).SetPositions(ethanol.GetConformer(0).GetPositions() + 5.0)
    ens = rx.wrap(ethanol)

    fit = ens.compare(0, 1, max_rmsd=0.1)

    assert (fit.rmsd, fit.order) == (pytest.approx(0.0, abs=1e-6), tuple(range(9)))
    np.testing.assert_array_equal(ens.mol.GetConformer(1).GetPositions(), ethanol.GetConformer(1).GetPositions())
    for candidate, max_rmsd, refusal in (
        (9, 0.1, "choose reference and candidate"),
        (0, 0.0, "finite and positive"),
        (0, float("inf"), "finite and positive"),
    ):
        with pytest.raises(ValueError, match=refusal):
            ens.compare(0, candidate, max_rmsd=max_rmsd)


@needs("scipy")
def test_prune_runs_each_method_of_a_sequence_in_turn():
    """Butanol, a copy and a copy with C0-C1 broken: the connectivity filter drops the broken one before RMSD merges."""
    butanol = embedded("CCCCO")
    for _ in range(2):
        butanol.AddConformer(Chem.Conformer(butanol.GetConformer(0)), assignId=True)
    rdMolTransforms.SetBondLength(butanol.GetConformer(2), 0, 1, 3.0)

    ens = rx.wrap(butanol, energies=[0.0, 0.1, 0.2]).prune(by=["connectivity", "rmsd"])

    assert (ens.ids, ens.discarded) == ([0], [2, 1])


def test_binding_modes_count_inter_fragment_contact_kinds():
    """A water dimer H-bonded in one conformer and pulled apart in the other counts one HB set and one empty set."""
    ens = rx.wrap(water_dimers(0.0, 10.0))

    assert ens.binding_modes() == {("HB",): 1, (): 1}
    assert ens[:0].binding_modes() == {}


@needs("sklearn")
def test_representatives_keep_the_lowest_of_each_mode():
    """Water dimers in two H-bonded clusters, an H-bonded straggler and two pulled apart: each cluster gives its
    lowest, the straggler folds into a mode its contacts show, and the apart pair is a new mode through its lowest
    member only while that lies within 10 kcal/mol of the lowest of all.
    """
    dimers = water_dimers(0.0, 0.02, 0.04, 0.4, 0.42, 0.44, 10.0, 12.0, 0.8)
    for apart, kept in (((0.5, 50.0), [6, 0, 3]), ((40.0, 50.0), [0, 3])):
        ens = rx.wrap(dimers, energies=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, *apart, 1.0])

        assert ens.representatives().ids == kept, apart


def test_align_superposes_the_core_its_default_or_on_names():
    """A copy turned a quarter about z and shifted, then atoms off the core moved: the held atoms, else a metal and its
    donors, else the heavy atoms superpose, as do the atoms ``on=`` names by SMARTS or index.
    """
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    chelate = on_its_vertices(rx.isomers(EN_PDBRCL, geometry="SPL")[0])
    pd = atom(chelate, "Pd")
    donors = sorted(n.GetIdx() for n in chelate.GetAtomWithIdx(pd).GetNeighbors())
    backbone = atom(chelate, "C")
    for name, mol, cons, on, core, off in (
        ("held atoms", embedded("CCO"), rx.Constraints(frozen=frozenset({0, 1, 8})), None, [0, 1, 8], [2]),
        ("metal and donors", chelate, rx.Constraints(), None, [pd, *donors], [backbone]),
        ("heavy atoms", embedded("CCO"), rx.Constraints(), None, [0, 1, 2], [8]),
        ("smarts", embedded("CCO"), rx.Constraints(), "CCO", [0, 1, 2], [8]),
        ("indices", embedded("CCO"), rx.Constraints(), [0, 1, 2], [0, 1, 2], [8]),
    ):
        xyz = mol.GetConformer().GetPositions()
        copy = xyz @ turn + 5.0
        copy[off] += 1.0
        mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
        mol.GetConformer(1).SetPositions(copy)
        ens = rx.wrap(mol)
        ens.cons = cons

        aligned = ens.align(on=on).mol
        frames = [aligned.GetConformer(cid).GetPositions()[core] for cid in (0, 1)]

        np.testing.assert_allclose(frames[1], frames[0], atol=1e-6, err_msg=name)


@needs("openconf")
def test_mc_replaces_the_seeds_of_an_unconstrained_hexanol():
    ens = rx.wrap(embedded("CCCCCCO"))
    ens.mc(preset="rapid", seed=1)
    assert ens.ids
    assert 0 not in ens.ids


@needs("openconf")
def test_mc_forgets_what_was_read_on_the_old_coordinates():
    """trans-PdCl2(NH3)2, held by its metal windows, searched after minimize: the search adds to its seed, and no
    energy, status or reason outlives the move.
    """
    trans = rx.isomers(DIAMMINE_PD, geometry="SPL").select(label="trans")
    ens = rx.embed(trans, n=1, seed=1).minimize()
    seed = ens.ids[0]
    ens.relax_failures = {seed: "a reason read on the seed"}
    ens.mc(preset="rapid", seed=1)

    assert ens.ids[0] == seed
    assert (ens.energies, ens.energy_kind, ens.unrelaxed, ens.relax_failures) == ({}, "", [], {})
    ens.minimize()
    assert set(ens.energies) == set(ens.ids)


@needs("openconf")
def test_refused_openconf_search_leaves_ferrocene_as_it_was():
    ens = rx.embed(Chem.AddHs(ferrocene(), addCoords=True), n=2, seed=1).minimize()
    ids, energies, kind = list(ens.ids), dict(ens.energies), ens.energy_kind
    positions = {cid: ens.mol.GetConformer(cid).GetPositions() for cid in ens.ids}

    ens.mc()

    assert (ens.ids, ens.energies, ens.energy_kind) == (ids, energies, kind)
    assert all(np.array_equal(ens.mol.GetConformer(cid).GetPositions(), xyz) for cid, xyz in positions.items())


def test_ensemble_set_keeps_its_errors_through_each_verb():
    """Each verb runs within each candidate, in order, and returns a set of them: the filter drops a torn ethanol."""
    failed = rx.EmbeddingError("species [CCN]: no seed met the constraints")
    later = rx.EmbeddingError("species [CCS]: no seed met the constraints")
    torn = conformers("CCO", 2)
    rdMolTransforms.SetBondLength(torn.GetConformer(1), 0, 1, 3.0)
    members = [rx.wrap(embedded("CCO"), energies=[0.0]), rx.wrap(torn, energies=[0.0, 1.0])]
    ethanols = rx.EnsembleSet(members, errors=[failed])
    for index, member in enumerate(ethanols):
        member.tag["nci"] = f"grip-{index}"
    both = ["grip-0", "grip-1"]
    rows = (
        ("filter by", ethanols.filter(by="connectivity"), both, (failed,)),
        ("slice", ethanols[:1], ["grip-0"], (failed,)),
        ("added", ethanols + rx.EnsembleSet([rx.wrap(embedded("CCO"))], [later]), [*both, None], (failed, later)),
        ("minimize", ethanols.minimize(), both, (failed,)),
        ("lowest", ethanols.lowest(1), both, (failed,)),
        ("score", ethanols.score("ff"), both, (failed,)),
    )
    found = {
        name: (type(result), [member.tag.get("nci") for member in result], result.errors) for name, result, *_ in rows
    }
    assert found == {name: (rx.EnsembleSet, modes, errors) for name, _result, modes, errors in rows}
    assert [member.ids for member in ethanols] == [[0], [0]]
    with pytest.raises(TypeError):
        ethanols + list(ethanols)


def test_ensemble_set_explains_an_attribute_it_cannot_answer():
    pair = rx.EnsembleSet([rx.wrap(embedded("CCO")), rx.wrap(embedded("CCC"))])
    for name in ("ids", "energy_kind"):
        with pytest.raises(AttributeError, match=r"needs one Ensemble; this set holds 2.*pick one with \[k\]"):
            getattr(pair, name)
    assert not hasattr(pair, "not_a_verb_at_all")


def test_ensemble_set_picks_by_species_name_index_or_mode():
    """``select`` takes the one candidate ``filter`` picks, and raises listing the set when none or several match."""
    alanines = rx.EnsembleSet(rx.wrap(embedded(smiles)) for smiles in ("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O"))
    alanines[1].tag["nci"] = "HB"
    pds = rx.EnsembleSet(rx.wrap(on_its_vertices(species)) for species in rx.isomers(DIAMMINE_PD))
    dimer = rx.EnsembleSet([rx.wrap(pd2cl6())])  # each Pd seats both bridges; Pd4's terminal Cl5 and Cl6 adjacent
    rows = (
        ("alanine hand", alanines, {"stereo": "C1:R"}, [0]),
        ("numpy index -1", alanines, {"index": np.int64(-1)}, [1]),
        ("contact mode", alanines, {"nci": "HB"}, [1]),
        ("diammine label", pds, {"label": "trans"}, [1]),
        ("polyhedron code", pds, {"geometry": "spl"}, [0, 1]),
        ("dimer second metal", dimer, {"center": 4, "arrangement": "Cl3 Cl7 Cl5 Cl6"}, [0]),
        ("dimer first metal", dimer, {"center": 1, "arrangement": "Cl3 Cl7 Cl5 Cl6"}, []),
    )
    found = {name: chosen.filter(**keys) for name, chosen, keys, _ in rows}
    assert {name: (type(one), [id(e) for e in one]) for name, one in found.items()} == {
        name: (rx.EnsembleSet, [id(chosen[k]) for k in picked]) for name, chosen, _, picked in rows
    }
    assert alanines.select(stereo="C1:S") is alanines[1]
    with pytest.raises(ValueError, match=r"matched 2 candidates; have:\n  \[0\]"):
        alanines.select()


def test_best_ranks_by_one_real_model_and_empties_last():
    butanol, isobutanol = conformers("CCCCO", 2, seed=1), conformers("CC(C)CO", 2, seed=1)
    high, low = rx.wrap(butanol, energies=[5.0, 7.0]), rx.wrap(isobutanol, energies=[3.0, 9.0])

    assert rx.EnsembleSet([high, low]).best() is low
    assert rx.EnsembleSet([low[:0], high]).best() is high
    for candidates in (
        [rx.wrap(butanol, minimized=True), rx.wrap(isobutanol, minimized=True)],  # a force field
        [high, low.score(Model())],  # two models
        [low[:0]],  # no energies
        [rx.wrap(butanol, energies={0: 1.0})[1:]],  # a selection of conformers without one
    ):
        with pytest.raises(ValueError, match="compares real energies from one model"):
            rx.EnsembleSet(candidates).best()


def test_ensemble_set_names_each_candidate_by_species_and_mode(tmp_path):
    """Two zinc hands no label tells apart: each line, the repr and each file name carry the centre row read as the
    species set reads it, the file its index too.
    """
    species = rx.isomers("[NH3]->[Zn+2](<-[PH3])(<-[Cl-])<-[Br-]", geometry="TET")
    hands = rx.EnsembleSet(rx.wrap(on_its_vertices(mol)) for mol in species)
    hands[1].tag["nci"] = "HB"
    lines = species.summary().splitlines()
    names = [line.split("] ", 1)[1] + mode for line, mode in zip(lines, ("", " HB"), strict=True)]

    paths = hands.dump(str(tmp_path / "zinc.xyz"))

    assert hands.summary().splitlines() == [f"  [{k}] {name}  (1 conformers)" for k, name in enumerate(names)]
    assert all(name in repr(hands) for name in names)
    slugs = [Path(path).name.split("_", 2) for path in paths]
    assert [slug[:2] for slug in slugs] == [["zinc", "0"], ["zinc", "1"]]
    assert slugs[0][2] != slugs[1][2]
    assert [len(Path(path).read_text().splitlines()) for path in paths] == [2 + hands[0].mol.GetNumAtoms()] * 2


def test_species_sets_add_and_slice_to_species_sets():
    alcohols, alkenes = rx.isomers("CC(O)CC"), rx.isomers("CC=CC")

    both = alcohols + alkenes

    assert type(both) is type(both[:1]) is rx.SpeciesSet
    assert list(both) == [*alcohols, *alkenes]
    assert both.select(stereo="E") is alkenes.select(stereo="E")
    with pytest.raises(ValueError, match="matched 4 species"):
        both.select()
    with pytest.raises(TypeError):
        alcohols + tuple(alkenes)


def test_species_set_reads_each_member_as_it_now_is():
    """An inverted chlorofluoroethane selects by its new hand, untouched by the reading; a ferrocenyl Mn hydride whose
    amine N is left free is named by its own species, though read again alone its Mol specifies two.
    """
    found = rx.isomers("C[C@H](F)Cl")
    mol = found[0]
    original = rx.identity(mol).stereo
    mol.GetAtomWithIdx(1).InvertChirality()
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    assert found.select(stereo=rx.identity(mol).stereo) is mol
    with pytest.raises(ValueError, match=r"matched 0 species; have:\n\[ 0\]"):
        found.select(stereo=original)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before

    manganese = rx.isomers(rx.read_xyz(str(EXAMPLES_DIR / "mnh.xyz"), 0), center=0, stereo="free")

    assert manganese.summary().startswith("[ 0] Mn1 OCT fac [H67 P2 C64 N6 N5 C65] Λ ; Fe0 LIN")
    assert len(manganese.summary(details=True).splitlines()) > len(manganese)
    assert manganese.select(center="Mn", label="fac") is manganese[0]


def test_selection_refuses_an_unreadable_key_index_or_geometry():
    """Species and candidate sets select alike: a known name, an int position in range and one polyhedron."""
    species = rx.isomers(DIAMMINE_PD)  # cis and trans
    candidates = rx.EnsembleSet()
    for selection, keys, error, message in (
        (species, {"chiral": "R"}, TypeError, "unknown selector"),
        (candidates, {"mode": "HB"}, TypeError, "'nci'"),
        (species, {"index": True}, TypeError, "int position"),
        (candidates, {"index": 1.0}, TypeError, "int position"),
        (species, {"index": 2}, IndexError, "out of range for 2"),
        (species, {"index": -3}, IndexError, "out of range for 2"),
        (candidates, {"geometry": ["SPL", "OCT"]}, TypeError, "one polyhedron"),
        (species, {"geometry": "XYZ"}, ValueError, "unknown geometry 'XYZ'"),
    ):
        with pytest.raises(error, match=message):
            selection.filter(**keys)


def test_coembedded_pairs_narrow_both_endpoints_alike():
    """Two ethanol pairs whose hydroxyl H moves 1 A: indexing, iterating, slicing and lowest keep pair k of both
    endpoints with its seed; ids refuse endpoints selected apart.
    """
    pairs = ethanol_pairs()

    assert (len(pairs), pairs.ids, pairs.react_atoms, pairs.source_seeds) == (2, [0, 1], (8,), {0: 11, 1: 12})
    assert pairs.react_rmsds == pytest.approx({0: 1.0, 1: 1.0})
    assert pairs.alignment_scores == pytest.approx({0: 0.0, 1: 0.0})
    rows = {"index": (pairs[1], [1]), "iterated to the end": ([*pairs][-1], [1]), "slice": (pairs[:1], [0])}
    rows["lowest"] = (pairs.lowest(), [1])
    found = {
        name: (type(one), one.reactant.ids, one.product.ids, list(one.source_seeds)) for name, (one, _) in rows.items()
    }
    assert found == {name: (rx.Coembedded, ids, ids, ids) for name, (_, ids) in rows.items()}
    assert [one.optimizer_attempts for one, _ids in rows.values()] == [pairs.optimizer_attempts] * 4
    with pytest.raises(ValueError, match="endpoint ids differ"):
        _ = replace(pairs, product=pairs.product[:1]).ids


def test_coembedded_set_keeps_its_errors_through_slicing():
    failed = rx.EmbeddingError("contacts='auto' mode 'ionic': no pair met the constraints")
    hb, stacked = ethanol_pairs("HB"), ethanol_pairs("stacked")
    modes = rx.CoembeddedSet([hb, stacked], errors=[failed])
    found = {"slice": modes[1:], "index": modes.filter(index=-1), "mode": modes.filter(nci="HB")}
    picked = {"slice": [stacked], "index": [stacked], "mode": [hb]}
    assert {name: (type(one), one.errors, [id(p) for p in one]) for name, one in found.items()} == {
        name: (rx.CoembeddedSet, (failed,), [id(p) for p in pairs]) for name, pairs in picked.items()
    }
