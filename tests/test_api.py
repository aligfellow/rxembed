"""`api.py`: the public verbs' argument handling and the hand-offs from a source to its result, across the owners."""

import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

import rxembed as rx
from rxembed.engine.atom_mapping import in_reactant_order
from rxembed.perception.changes import signature
from tests.geometry_fixtures import embedded

DIAMMINE_PD = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"
DIAMMINE_PT = "N->[Pt+2](<-[Cl-])(<-[Cl-])<-N"
EN_PDCL2 = "Cl[Pd]1(Cl)NCCN1"  # covalent notation for dichloro(ethylenediamine)palladium
SN2 = ("CBr.[Cl-]", "CCl.[Br-]")
# The beta-hydride elimination of a (2-phenylethyl)palladium bis(trimethylphosphine): H3 moves onto Pd4
BETA_HYDRIDE = (
    "C[P:5](C)(C)->[Pd+2:4](<-[CH2-:1][CH:2]([H:3])c1ccccc1)<-[P:6](C)(C)C "
    "|atomProp:1.atomNote.s0:4.atomNote.SPL:5.atomNote.s1:14.atomNote.s3|",
    "C[P:5](C)(C)->[Pd+2:4]1(<-[CH2:1]=[CH:2]->1c1ccccc1)(<-[H-:3])<-[P:6](C)(C)C "
    "|atomProp:1.atomNote.s0:4.atomNote.SPL:5.atomNote.s1+:6.atomNote.s1+:13.atomNote.s2:14.atomNote.s3|",
)
# An aryl C-H activated at a pincer palladium by its own phenoxide, drawn without atom maps
PD_C_H_ACTIVATION = (
    "COC1=C[CH](->[Pd+2]23<-[N]4=C(C5=[N]->2C(CC2=[N]->3C([O-])=CC=C2)=CC=C5)C=CC=C4)=CC(OC)=C1OC",
    "[H]OC1=CC=CC2CC3=CC=CC4C5=[N](->[Pd+2](<-[C-]6=CC(OC)=C(OC)C(OC)=C6)(<-[N]=21)<-[N]=43)C=CC=C5",
)


@pytest.fixture(scope="module")
def beta_hydride():
    """Return the beta-hydride's endpoints, mapped atom for atom with the product in its own order, and their pair
    co-embedded with a short endpoint preparation.
    """
    mols = [Chem.AddHs(rx.parse_smiles(side, remove_hs=False)) for side in BETA_HYDRIDE]
    mols[1] = in_reactant_order(*mols)
    for mol in mols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    mols[1] = Chem.RenumberAtoms(mols[1], list(range(mols[1].GetNumAtoms()))[::-1])
    sides = tuple(rx.isomers(mol).select() for mol in mols)
    short = rx.CoEmbedParams(seed_relax_iters=50, joint_max_iters=50)
    return sides, rx.coembed(sides, n=1, seed=1, coembed_params=short, threads=1)


def _bonds_by_map(mol):
    """Return the bonds of `mol` as pairs of atom-map numbers."""
    return {
        frozenset(mol.GetAtomWithIdx(i).GetAtomMapNum() for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    }


def test_isomers_refuses_a_choice_its_source_cannot_take():
    for kwargs, error, refusal in (
        ({"observed": True}, ValueError, "observed=True reads a conformer"),
        ({"geometry": "SPL"}, ValueError, "choose a metal arrangement; this species has none"),
        ({"screen": 1}, TypeError, "screen must be a bool"),
    ):
        with pytest.raises(error, match=refusal):
            rx.isomers("CCO", **kwargs)


def test_embed_refuses_a_request_it_cannot_run():
    for source, kwargs, error, refusal in (
        ([], {}, ValueError, "empty sequence"),
        ("water.XYZ", {}, ValueError, r"rx\.read_xyz\(path, charge\)"),
        (Path("water.xyz"), {}, ValueError, r"rx\.read_xyz\(path, charge\)"),
        ("CCO", {"n": 0}, ValueError, "positive conformer count"),
        ("CCO", {"n": 1.5}, ValueError, "positive conformer count"),
        ("CCO", {"trajectory": 1, "n": 1}, TypeError, "trajectory must be True or False"),
        ("CCO", {"trajectory": True, "n": 2}, ValueError, "requires n=1"),
        ("CCO", {"max_iters": 0}, ValueError, "max_iters must be a positive integer"),
        ("CCO", {"max_iters": True}, ValueError, "max_iters must be a positive integer"),
    ):
        with pytest.raises(error, match=refusal):
            rx.embed(source, **kwargs)


def test_minimize_refuses_a_relax_cap_not_a_positive_integer():
    with pytest.raises(ValueError, match="max_iters must be a positive integer"):
        rx.minimize(embedded("CCO"), max_iters=0)


def test_coembed_refuses_a_reaction_count_or_selection():
    forms = r"a \(reactant, product\) pair or 'reactant>>product'"
    for sources, kwargs, error, refusal in (
        ((Chem.MolFromSmiles("CCO"),), {}, ValueError, forms),
        ((("CCO",),), {}, ValueError, forms),
        (("CCO>>CC>>C",), {}, ValueError, "without agents or CX extensions"),
        (("CO>>CO |a:0|",), {}, ValueError, "without agents or CX extensions"),
        (SN2, {"n": 0}, ValueError, "positive conformer count"),
        (SN2, {"n": 1.5}, ValueError, "positive conformer count"),
        (SN2, {"constrain": {(0, 100): (2.0, 3.0)}}, ValueError, "geometry terms need atom-index keys"),
        (SN2, {"spectator_atoms": []}, ValueError, "spectator_atoms must contain at least one atom"),
        (SN2, {"react_atoms": "[CH3]"}, TypeError, "react_atoms takes reactant atoms"),
    ):
        with pytest.raises(error, match=refusal):
            rx.coembed(*sources, seed=1, **kwargs)


def test_coembed_refuses_a_fixed_endpoint_it_cannot_keep():
    for edit, kwargs, refusal in (
        (None, {"fixed": "both"}, 'fixed must be "reactant" or "product"'),
        (None, {"n": 2}, "use n=1 or omit n"),
        (lambda mol: mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True), {}, "exactly one conformer"),
        (lambda mol: mol.GetConformer().SetAtomPosition(0, (float("nan"), 0, 0)), {}, "finite 3D coordinates"),
        (lambda mol: mol.GetConformer().Set3D(False), {}, "finite 3D coordinates"),
    ):
        acetone = embedded("CC(=O)C")
        if edit:
            edit(acetone)
        with pytest.raises(ValueError, match=refusal):
            rx.coembed(acetone, "C=C(O)C", **{"fixed": "reactant", "n": 1, **kwargs})


def test_embed_needs_xyzgraph_to_check_a_geometrys_hand(monkeypatch):
    geometry = embedded("CCO")
    monkeypatch.setitem(sys.modules, "xyzgraph", None)  # None makes the import raise ImportError

    with pytest.raises(ImportError, match="xyzgraph"):
        rx.embed(geometry, n=1, seed=1)


def test_specified_alkene_source_writes_its_ez():
    """E-but-2-ene built with no bond directions, and E-cyclooctene drawn by its hydrogens' directions."""
    butene = Chem.MolFromSmiles("CC=CC")
    butene.GetBondWithIdx(1).SetStereoAtoms(0, 3)
    butene.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOE)
    sources = (butene, "[H]/C1=C(/[H])CCCCCC1")
    assert [rx.cxsmiles(one) for source in sources for one in rx.isomers(source)] == ["C/C=C/C", "C1=C/CCCCCC/1"]


def test_isomers_seat_water_once_per_vacancy_in_cx_order():
    """cis-Ammine(chloro)palladium has two distinct vacancies, trans to the ammine and trans to the chloride."""
    cis = rx.isomers("N->[Pd+2]<-[Cl-].O", geometry="SPL")[0]  # N0 and Cl2 on adjacent slots

    names = [rx.cxsmiles(one) for one in rx.isomers(cis, coordinate=3)]  # the water O3

    assert names == sorted(set(names))
    assert len(names) == 2


def test_embed_seeds_from_its_seed_through_native_params():
    """A NumPy integer seed and count serve as Python ones."""
    native = rdDistGeom.KDG()
    first, again, other = (
        rx.embed("CCO", n=np.int64(2), seed=np.int64(seed), params=rx.EmbedParams(native=native)) for seed in (5, 5, 7)
    )

    assert first.sampling.seed == 5
    assert first.sampling.params.native is native
    seeded, reseeded = ([ens.mol.GetConformer(cid).GetPositions() for cid in first.ids] for ens in (first, again))
    np.testing.assert_array_equal(seeded, reseeded)
    assert not np.allclose(first.mol.GetConformer().GetPositions(), other.mol.GetConformer().GetPositions())


def test_embed_returns_an_ensemble_for_one_species_else_a_set():
    rows = (
        ("CCO", {}, rx.Ensemble),
        (Chem.MolFromSmiles("CCO"), {}, rx.Ensemble),
        (["CCO"], {}, rx.EnsembleSet),
        ("C.C", {"contacts": "auto"}, rx.EnsembleSet),
    )
    assert [type(rx.embed(source, n=1, seed=1, **kwargs)) for source, kwargs, _ in rows] == [kind for *_, kind in rows]


def test_free_stereo_leaves_an_unassigned_centre_to_the_seed():
    """sec-Butylamine on linear Cu(I): its carbon is unassigned, so only stereo='free' makes the SMILES one species."""
    smiles = "CCC(C)[NH2]->[Cu+]<-[NH3]"

    assert rx.embed(smiles, n=1, seed=1, stereo="free").n == 1
    with pytest.raises(ValueError, match=r"rx\.isomers"):
        rx.embed(smiles, n=1, seed=1)


def test_sequence_keeps_what_embeds_and_names_each_failure(caplog):
    """A member that cannot be read is named as given, a read one by its species, and one no seed builds by the error
    that says so; each failure is one WARNING line.
    """
    for name, sources, fix, named in (
        (
            "unread",
            [DIAMMINE_PT, Chem.MolFromSmiles("CC(O)CC"), "CCO"],
            None,
            [f"species [{DIAMMINE_PT}]: ", "species [CCC(C)O]: "],
        ),
        (
            "E-cyclooctene contradicted",
            rx.isomers("C1CCC=CCCC1"),
            {(2, 5): 2.0},
            ["species [C3=C4:E]: "],
        ),
        (
            "trans-spanning en",
            rx.isomers(EN_PDCL2, geometry="SPL", screen=False),
            None,
            ["Pd1 SPL trans "],
        ),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rxembed"):
            result = rx.embed(sources, fix=fix, n=1, seed=1)
        errors = [str(err) for err in result.errors]
        assert len(result) == 1, name
        assert len(errors) == len(named), (name, errors)
        assert all(map(str.startswith, errors, named)), (name, errors)
        assert [record.getMessage() for record in caplog.records] == errors, name


def test_embed_raises_when_nothing_embeds_alone_its_own_error(caplog):
    trans = rx.isomers(EN_PDCL2, geometry="SPL", screen=False).select(label="trans")
    with caplog.at_level(logging.WARNING, logger="rxembed"), pytest.raises(rx.EmbeddingError) as lone:
        rx.embed(trans, n=1, seed=1)
    assert not caplog.records
    assert "none of" not in str(lone.value)

    with pytest.raises(rx.EmbeddingError) as every:
        rx.embed([trans, trans], n=1, seed=1)

    assert str(every.value).startswith("none of 2 species")
    assert str(every.value).endswith(str(lone.value))
    assert every.value.failures == lone.value.failures + lone.value.failures


def test_max_iters_caps_the_relax_marking_seeds_unrelaxed():
    ens = rx.embed("OCCCCO", constrain={(0, 5): (2.6, 3.0)}, n=1, seed=1, max_iters=1)

    assert ens.unrelaxed == ens.ids == [0]


def test_nickel_chelate_keeps_its_uff_cleanup_on_its_own_atoms():
    smiles = "C[P]1(C)CC[P](C)(C)->[Ni+2]<-12<-[O-]C(=O)C[N-]->2C"

    ens = rx.embed(rx.isomers(smiles, geometry="SPL")[0], n=1, seed=19, trajectory=True)

    trail = ens.trajectory
    frames = [conformer.GetPositions() for conformer in trail.GetConformers()]
    assert len(frames) > 2
    assert [atom.GetSymbol() for atom in trail.GetAtoms()] == [atom.GetSymbol() for atom in ens.mol.GetAtoms()]
    assert not np.allclose(frames[0], frames[1])
    np.testing.assert_allclose(frames[-1], ens.mol.GetConformer().GetPositions())


def test_copper_macrocycle_embeds_its_tetrahedral_species():
    """Both diagonal bites widen to the relaxed shell, and the bounded bite box keeps them inside the tetrahedron."""
    species = rx.isomers("C1C[N@@H]2->[Cu+]34<-[S](CC2)CC/[C-]->3=[NH+]/CC[S]->4C1", geometry="tetrahedral")[0]

    ensemble = rx.embed(species, n=1, seed=42, threads=1)

    assert (ensemble.n, ensemble.unrelaxed, rx.cxsmiles(ensemble.mol)) == (1, [], rx.cxsmiles(species))


def test_geometry_source_keeps_its_planar_or_helical_hand(tmp_path):
    """No tag states these hands, so they come from the source conformer: a [5]helicene read from an .xyz keeps P and
    an inverted one turns M, each held by a trivial fix that relaxes it into windows; a cyclophane whose point tag is
    inverted keeps its conformer's planar hand through minimize.
    """
    helicene = Chem.AddHs(Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12"))
    assert rdDistGeom.EmbedMolecule(helicene, randomSeed=1, useRandomCoords=True) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(helicene, maxIters=5000)
    xyz = helicene.GetConformer().GetPositions()
    fix = {(0, 1): round(float(np.linalg.norm(xyz[0] - xyz[1])), 3)}
    (tmp_path / "helicene.xyz").write_text(Chem.MolToXYZBlock(helicene))
    cyclophane = rx.embed("C1Cc2ccc(cc2)CCc2ccc1c(c2)[C@@H](C)O", n=1, seed=1).mol
    planar = signature(cyclophane)["planar"]
    for atom in cyclophane.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            atom.InvertChirality()

    hands, expected = {}, {}
    for name, ens, kind, hand in (
        (
            "helicene from xyz",
            rx.embed(rx.read_xyz(str(tmp_path / "helicene.xyz")), n=1, seed=3, fix=fix),
            "helical",
            Counter({"P": 1}),
        ),
        ("helicene inverted", rx.embed(helicene, n=1, seed=1, stereo="invert", fix=fix), "helical", Counter({"M": 1})),
        ("cyclophane inverted", rx.embed(cyclophane, n=2, seed=6).minimize(), "planar", planar),
    ):
        hands[name] = (len(ens.ids) > 0, [signature(ens.mol, conf_id=cid)[kind] for cid in ens.ids])
        expected[name] = (True, [hand] * len(ens.ids))
    assert hands == expected


def test_minimize_holds_a_pd_chelate_core_and_its_arrangement():
    species = rx.isomers("Br[Pd]1(Cl)NCCN1", geometry="SPL")[0]
    geometry = rx.embed(species, n=1, seed=1).mol
    palladium = next(atom for atom in geometry.GetAtoms() if atom.GetSymbol() == "Pd")
    sphere = {palladium.GetIdx(), *(donor.GetIdx() for donor in palladium.GetNeighbors())}
    core = [atom.GetIdx() for atom in geometry.GetAtoms() if atom.GetIdx() not in sphere][:3]

    out = rx.minimize(geometry, fix=[rx.Template(geometry, {i: i for i in core[:2]}), core[2]])

    assert out.n == 1
    assert sorted(out.cons.frozen) == core
    assert rx.cxsmiles(out.mol) == rx.cxsmiles(species)


def test_coembed_reads_sources_a_pair_or_a_reaction_smiles():
    written = [rx.smiles(rx.parse_smiles(side)) for side in SN2]
    pairs = [rx.coembed(*form, n=1, seed=1) for form in (SN2, (SN2,), (">>".join(SN2),))]
    assert [[rx.smiles(end.mol) for end in (pair.reactant, pair.product)] for pair in pairs] == [written] * 3


def test_coembed_keeps_a_fixed_endpoint_and_its_source_unmoved(beta_hydride):
    """The beta-hydride pair is rebuilt around its reactant, and a cis-diamminedichloropalladium around a trans one."""
    pair = beta_hydride[1][0]
    found = rx.isomers(DIAMMINE_PD, geometry="SPL")
    trans = rx.embed(found.select(label="trans"), n=1, seed=1).mol
    for name, sides, fixed in (
        ("beta-H reactant", (pair.reactant.mol, pair.product.mol), "reactant"),
        ("trans Pd product", (found.select(label="cis"), trans), "product"),
    ):
        side = ("reactant", "product").index(fixed)
        supplied = sides[side].GetConformer().GetPositions()

        rebuilt = rx.coembed(*sides, fixed=fixed, n=1, seed=1, threads=1)

        kept, built = (rebuilt.product, rebuilt.reactant) if side else (rebuilt.reactant, rebuilt.product)
        np.testing.assert_array_equal(kept.mol.GetConformer(rebuilt.ids[0]).GetPositions(), supplied, err_msg=name)
        np.testing.assert_array_equal(sides[side].GetConformer().GetPositions(), supplied, err_msg=name)
        assert built.check()[rebuilt.ids[0]].ok, name


def test_coembed_refuses_a_failing_fixed_endpoint_unmoved():
    acetone = embedded("CC(=O)C")
    torn = acetone.GetConformer().GetPositions()
    torn[2] += (5.0, 0.0, 0.0)  # the carbonyl O pulled off its carbon
    acetone.GetConformer().SetPositions(torn)

    with pytest.raises(rx.EmbeddingError, match="fixed reactant fails"):
        rx.coembed(acetone, "C=C(O)C", fixed="reactant", n=1, seed=42)
    np.testing.assert_array_equal(acetone.GetConformer().GetPositions(), torn)


def test_beta_hydride_with_short_preparation_keeps_valid_pairs(beta_hydride):
    _sides, pairs = beta_hydride

    assert len(pairs) == 1
    checks = [
        (all(report.ok for report in end.check().values()), end.unrelaxed) for end in (pairs.reactant, pairs.product)
    ]
    assert checks == [(True, [])] * 2  # a capped preparation starts the joint pass; it is no failure to relax


def test_coembed_keeps_beta_hydride_maps_in_its_product_order(beta_hydride):
    sides, pairs = beta_hydride
    ends = (pairs.reactant.mol, pairs.product.mol)

    maps = [{a.GetAtomMapNum(): a.GetSymbol() for a in mol.GetAtoms()} for mol in (*ends, *sides)]
    assert maps[:2] == maps[2:]
    assert _bonds_by_map(ends[0]) ^ _bonds_by_map(ends[1]) == _bonds_by_map(sides[0]) ^ _bonds_by_map(sides[1])


def test_coembed_overlays_an_unmapped_pd_ch_activation():
    """The aryl C4-H proton moves onto the phenoxide O, and each endpoint holds its own bond to it."""
    result = rx.coembed(*(rx.isomers(side, geometry="SPL").select() for side in PD_C_H_ACTIVATION), n=1, seed=1)

    ends = (result.reactant.mol, result.product.mol)
    (hydrogen,) = [atom for atom in result.react_atoms if ends[0].GetAtomWithIdx(atom).GetAtomicNum() == 1]
    parents = [end.GetAtomWithIdx(hydrogen).GetNeighbors()[0] for end in ends]
    assert (parents[0].GetIdx(), parents[1].GetSymbol()) == (4, "O")
    for end, parent, reach in zip(ends, parents, (1.2, 1.1), strict=True):
        xyz = end.GetConformer(result.ids[0]).GetPositions()
        assert np.linalg.norm(xyz[parent.GetIdx()] - xyz[hydrogen]) < reach


def test_coembed_names_each_endpoint_failure_of_a_pt_insertion():
    hydride = rx.isomers("[Pt+2:1]1(<-[H-:2])(<-[Cl-:3])(<-[PH3:4])<-[CH2:5]=[CH2:6]->1", geometry="SPL")[0]
    ethyl = rx.isomers("[Pt+2:1](<-[Cl-:3])(<-[PH3:4])<-[CH2-:6][CH2:5][H:2]", geometry="SPL")[0]  # H slot vacant
    starved = rx.CoEmbedParams(seed_relax_iters=1, joint_max_iters=1, max_cycles=1)

    with pytest.raises(rx.EmbeddingError) as raised:
        rx.coembed((rx.cxsmiles(hydride), rx.cxsmiles(ethyl)), n=1, seed=1, coembed_params=starved, threads=1)

    assert raised.value.failures
    for failure in raised.value.failures:
        assert failure.atoms, failure
        assert failure.detail.startswith(("reactant:", "product:")), failure


def test_wet_pd_diammine_coembeds_modes_keeping_failed_counts():
    """'auto' and the same modes passed from `rx.contacts` give one pair or one error per mode."""
    found = rx.isomers(DIAMMINE_PD, geometry="SPL")
    wet = [rx.cxsmiles(found.select(label=label)).replace(" |", ".O |") for label in ("cis", "trans")]
    modes = rx.contacts(rx.parse_smiles(wet[0]))

    pairs, chosen = (rx.coembed(*wet, contacts=request, n=1, seed=1, threads=1) for request in ("auto", modes))

    assert (type(pairs), type(chosen)) == (rx.CoembeddedSet, rx.CoembeddedSet)
    assert [pair.reactant.tag for pair in chosen] == [pair.reactant.tag for pair in pairs]
    assert pairs
    assert pairs.errors
    assert len(pairs) + len(pairs.errors) == len(modes)
    assert all(error.failures for error in pairs.errors)
    for pair in pairs:
        assert pair.reactant.tag == pair.product.tag
        assert pair.reactant.tag["nci"] in modes
        for endpoint in (pair.reactant, pair.product):
            assert all(report.ok for report in endpoint.check().values())


def test_coembed_holds_contacts_unless_the_reaction_bonds_them():
    """The acid proton H9 moves onto the amine N5, so an H9...N5 contact holds in the reactant alone."""
    results = rx.coembed("CC(=O)O.CN>>CC(=O)[O-].C[NH3+]", contacts="auto", n=1, seed=1)

    labels = [result.reactant.tag["nci"] for result in results]
    assert labels == [result.product.tag["nci"] for result in results]
    assert any("O3->N5" in label for label in labels)
    for result in results:
        ends = (result.reactant.mol, result.product.mol)
        held = [set(end.cons.contacts.distances) for end in (result.reactant, result.product)]
        for i, j in held[0]:
            formed = ends[0].GetBondBetweenAtoms(i, j) is None and ends[1].GetBondBetweenAtoms(i, j) is not None
            assert ((i, j) in held[1]) is not formed, (labels, i, j)
