"""`engine/coembed.py`: co-embedding a reaction's endpoints from one shared seed, each holding its own species."""

import contextlib
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.engine.atom_mapping import in_reactant_order
from rxembed.engine.coembed import least_motion
from rxembed.identity.reading import read_species
from rxembed.utils import without_atom_maps
from tests.conftest import EXAMPLES_DIR
from tests.metal_fixtures import specified_mol

_PT = Chem.GetPeriodicTable()
SN2_SIDES = ("CCCCBr.[Cl-]", "CCCCCl.[Br-]")  # C3 Br4 Cl5 on both sides


def _bonds(mol):
    """Return the bonds of `mol` as frozensets of their two atoms."""
    return {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in mol.GetBonds()}


def test_endpoints_keep_their_own_bonds_on_one_union_seed():
    """A bond either endpoint lacks is long there and short in the other, the spectators share one pose, and every
    pair comes from a seed of the union graph, an aromatic ring the reaction breaks included.
    """
    cu = (
        rx.isomers(Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[NH3:3].[Cl-:4]")), geometry="linear")[0],
        rx.isomers(Chem.AddHs(rx.parse_smiles("[NH3:1]->[Cu+:2]<-[Cl-:4].[NH3:3]")), geometry="linear")[0],
    )
    cycloaddition = (
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[CH2:7]=[CH2:8]",
        "[CH:1]12[CH:2]=[CH:3][CH:4]([CH:5]=[CH:6]1)[CH2:7][CH2:8]2",
    )
    found = {}
    for name, result in (
        ("SN2", rx.coembed(*SN2_SIDES, n=2, seed=3)),
        ("Cu ligand swap", rx.coembed(*cu, n=1, seed=1)),
        ("cycloaddition", rx.coembed(*cycloaddition, n=1, seed=42, threads=1)),
    ):
        ends = (result.reactant, result.product)
        bonds = [_bonds(end.mol) for end in ends]
        misread = []
        for cid in result.ids:
            for end, own in zip(ends, bonds, strict=True):
                pos = end.mol.GetConformer(cid).GetPositions()
                for i, j in map(tuple, bonds[0] ^ bonds[1]):
                    reach = 1.3 * sum(_PT.GetRcovalent(end.mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
                    if (np.linalg.norm(pos[i] - pos[j]) < reach) != ({i, j} in own):
                        misread.append((cid, i, j))
        seeds = {record["seed_method"] for record in result.joint_results.values()}
        aligned = all(result.alignment_scores[cid] < 0.5 for cid in result.ids)
        found[name] = (len(result.ids) > 0, seeds, aligned, misread)
    assert found == dict.fromkeys(found, (True, {"union"}, True, []))


def test_each_endpoint_keeps_the_configuration_it_specifies():
    """A specified centre, axis, bound donor or E/Z that differs between the endpoints turns in the one that copies
    it; a Z bond both specify stays Z in both.
    """
    amine_pd = [rx.isomers(f"[Pd](Cl)(Cl)(Cl)([N{hand}H](C)O)", geometry="SPL")[0] for hand in ("@", "@@")]
    found, expected = {}, {}
    for name, sides in (
        ("butanol inverted", [rx.parse_smiles(side) for side in ("C[C@H](O)CC", "C[C@@H](O)CC")]),
        ("biaryl turned", [rx.parse_smiles(f"Cc1cccc(O)c1-c1c(C)cccc1O |w{w}:7.6|") for w in "UD"]),
        ("Pd amine inverted", amine_pd),
        ("Z crotyl SN2", [rx.parse_smiles(f"C/C=C\\C{x}.[{y}-]") for x, y in (("Br", "Cl"), ("Cl", "Br"))]),
        ("pentene E to Z", [rx.parse_smiles(side) for side in ("C/C=C/CC", "C/C=C\\CC")]),
    ):
        result = rx.coembed(*sides, n=1, seed=1)
        found[name] = [
            rx.identity(end.mol, cid).stereo for cid in result.ids[:1] for end in (result.reactant, result.product)
        ]
        expected[name] = [rx.identity(side).stereo for side in sides]

    assert found == expected


def test_open_stereo_takes_one_configuration_in_both_ends():
    """A centre bonded alike in both endpoints reads one hand in their shared atom order, whether both graphs name it
    or only the one a nearby substitution makes it a centre in; a centre the reaction forms is the reaction's.
    """
    found, expected = {}, {}
    for name, reactant, product, centre in (
        ("carbinol beside SN2", "CC(O)CCCBr.[Cl-]", "CC(O)CCCCl.[Br-]", 1),
        # the ring inversion reflects the shared seed, so a pair reading the sec-butyl C2 two ways must be refused
        ("sec-butyl, inverting ring", "[C@]1(C(C)CC)(CCCCC)OCCCC1", "[C@@]1(C(C)CC)(CCCCC)OCCCC1", 1),
        ("centre SN2 makes", "BrCC(C)CBr.[Cl-]", "ClCC(C)CBr.[Br-]", 2),
        (
            "Diels-Alder centre",
            "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH:6][C:7](=[O:8])[O:9][CH3:10]",
            "[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH:6]1[C:7](=[O:8])[O:9][CH3:10]",
            None,
        ),
    ):
        result = rx.coembed(reactant, product, n=2, seed=1)
        alike = set()
        for cid in result.ids if centre is not None else ():
            hands = []
            for end in (result.reactant, result.product):
                pos = end.mol.GetConformer(cid).GetPositions()
                arms = sorted(n.GetIdx() for n in end.mol.GetAtomWithIdx(centre).GetNeighbors())[:3]
                hands.append(np.sign(np.linalg.det(pos[arms] - pos[centre])))
            alike.add(hands[0] == hands[1])
        found[name] = (len(result.ids) > 0, alike)
        expected[name] = (True, set() if centre is None else {True})
    assert found == expected


def _stereoisomer(species):
    """Return `species` as canonical CXSMILES without atom maps: the stereoisomer, whatever its labelling."""
    return rx.cxsmiles(specified_mol(replace(species, mol=without_atom_maps(species.mol))))


def test_least_motion_relabels_only_within_a_stereoisomer():
    """Two eta2 propenes of opposite winding are not interchangeable, nor the bridges at one palladium of a dimer."""
    faces = "(<-[CH2:7]=[CH:8]([CH3:9])->1)<-[CH2:10]=[CH:11]([CH3:12])->2"
    ammine = f"[Cl-:1]->[W:2]12(<-[C-:3]#[O+:4])(<-[PH3:5])(<-[NH3:6]){faces}.[OH2:13]"
    aqua = f"[Cl-:1]->[W:2]12(<-[C-:3]#[O+:4])(<-[PH3:5])(<-[OH2:13]){faces}.[NH3:6]"
    ammine = Chem.AddHs(rx.parse_smiles(ammine))
    tungsten = [  # the first seating of each side whose two face notes wind oppositely
        read_species(
            next(
                iso
                for iso in rx.isomers(side, geometry="OCT")
                if {atom.GetProp("atomNote")[-1] for atom in iso.GetAtoms() if atom.HasProp("atomNote")} >= {"+", "-"}
            )
        )
        for side in (ammine, in_reactant_order(ammine, Chem.AddHs(rx.parse_smiles(aqua))))
    ]
    dimer = [read_species(iso) for iso in rx.isomers("N->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pd+2](<-N)(<-[Cl-])<-[Cl-]->1")[:2]]
    rows = {"W ammine for water": tungsten, "Pd2 bridges": dimer}
    relabelled = {name: list(map(_stereoisomer, least_motion(*sides))) for name, sides in rows.items()}
    assert relabelled == {name: list(map(_stereoisomer, sides)) for name, sides in rows.items()}


def test_refilled_pair_takes_the_id_it_replaces():
    """Pruning keeps one pair of the first shared seed, so the second comes from a refill round."""
    result = rx.coembed("C[C@H](Br)CC.[Cl-]", "[Br-].CC[C@H](Cl)C", n=2, seed=5, params=rx.EmbedParams(prune_rms=100))

    assert sorted(result.ids) == [0, 1]
    assert len(set(result.source_seeds.values())) == 2


def test_pair_scores_and_records_read_one_coupled_objective():
    """N-methylacetamide twice, its methyls held 3.4 A apart: no atom reacts and the score is the last joint step."""
    pair = rx.coembed("CC(=O)NC", "CC(=O)NC", constrain={(0, 4): (3.4, 3.4)}, n=1, seed=42, threads=1)

    (cid,) = pair.ids
    record = pair.joint_results[cid]
    ends = (pair.reactant, pair.product)
    spans = [np.linalg.norm(np.subtract(*end.mol.GetConformer(cid).GetPositions()[[0, 4]])) for end in ends]
    score = pytest.approx(pair.scores[cid], rel=1e-12, abs=1e-10)

    assert not pair.react_atoms
    observed = (record["stage"], record["score_history"][-1], record["score"], spans)
    assert observed == ("joint", score, score, pytest.approx([3.4, 3.4], abs=0.05))


def test_nazarov_frames_coembed_in_their_input_order():
    """Two xyz frames of one path, mapped atom for atom, return in their input order."""
    paths = [str(EXAMPLES_DIR / f"nazarov_{side}.xyz") for side in ("start", "end")]
    sides = [rx.read_xyz(path, 0) for path in paths]
    for mol in sides:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    params = rx.CoEmbedParams(seed_relax_iters=10, joint_max_iters=10, max_cycles=1)
    result = rx.coembed(*sides, n=1, seed=1, stereo="free", coembed_params=params)

    ends = (result.reactant, result.product)
    symbols = [[atom.GetSymbol() for atom in Chem.MolFromXYZFile(path).GetAtoms()] for path in paths]
    assert len(result) == 1
    assert [[atom.GetSymbol() for atom in end.mol.GetAtoms()] for end in ends] == symbols
    assert [np.isfinite(end.mol.GetConformer(result.ids[0]).GetPositions()).all() for end in ends] == [True, True]
    assert np.isfinite(result.alignment_scores[result.ids[0]])


def test_coembed_never_reflects_an_endpoint(monkeypatch):
    """A free Delta zinc would be reflected after a lone relax; an endpoint shares its spectators with its partner."""

    def refuse(*_args):
        raise AssertionError("an endpoint was reflected")

    monkeypatch.setattr("rxembed.engine.conformers.reflect_to_specification", refuse)
    delta = rx.isomers("[NH3]->[Zn+2](<-[PH3])(<-[Cl-])<-[Br-]", geometry="TET").select(hand="delta")

    assert rx.coembed(delta, delta, n=1, seed=1).ids


def test_coembed_builds_a_partner_around_an_unmoved_endpoint():
    reactant = rx.embed(SN2_SIDES[0], n=1, seed=1).mol
    before = reactant.GetConformer().GetPositions()

    pair = rx.coembed(reactant, SN2_SIDES[1], fixed="reactant", seed=1)

    assert len(pair) == 1
    np.testing.assert_array_equal(pair.reactant.mol.GetConformer(pair.ids[0]).GetPositions(), before)
    assert pair.product.mol.GetBondBetweenAtoms(3, 5) is not None


def test_rejected_pair_returns_to_its_prepared_geometry():
    """Potassium trading an ammine for water leaves its window in the joint pass; without preparation every pair
    fails, and the refusal names each failure by its endpoint.
    """
    sides = [Chem.AddHs(rx.parse_smiles(f"N->[K+](<-N)(<-N)(<-N)(<-N)<-{x}.{y}")) for x, y in (("N", "O"), ("O", "N"))]
    unprepared = rx.CoEmbedParams(seed_relax_iters=0, max_cycles=1, joint_max_iters=1)
    for name, seed, params, refusal in (
        ("prepared", 2, None, None),
        ("unprepared", 7, unprepared, r"kept no pairs \(.*reactant: .*\); try another seed or endpoint"),
    ):
        with pytest.raises(rx.EmbeddingError, match=refusal) if refusal else contextlib.nullcontext():
            result = rx.coembed(*sides, n=1, seed=seed, coembed_params=params)
            reasons = [record["stopping_reason"] for record in result.joint_results.values()]
            assert reasons == ["validation_adjusted"], name
