"""`engine/seeding.py`: preparing a species, drawing seeds with the stereo it specifies, grafting a fixed core."""

import contextlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg

import rxembed as rx
from rxembed.engine.acceptance import conformer_misses
from rxembed.engine.seeding import Seeds, graft_frozen, prepare, reflect, reflect_to_specification, reflection_is_free
from rxembed.geometry.bounds import Sampling
from rxembed.geometry.constraints import Constraints, resolve_core
from rxembed.identity.reading import Specification, read_species
from rxembed.metal.polyhedra import POLYHEDRA
from rxembed.utils import mirror_tag
from tests.geometry_fixtures import embedded, mirrored, placed
from tests.metal_fixtures import on_its_vertices, orientation_parity_of, seated, seeds

CO_ALL_CIS = "N->[Co+3](<-N)(<-O)(<-O)(<-[Cl-])<-[Cl-]"  # N0 Co1 N2 O3 O4 Cl5 Cl6
CO_AMINE = "O->[Co+3](<-[Cl-])(<-[CH3-])(<-[NH2][C@H](C)CC)(<-[F-])<-P"  # C5 the amine's stereocentre
TAGGED_AMINE_PD = "[Pd](Cl)(Cl)(Cl)([N@H](C)O)"  # Pd0, amine N4 tagged but no stereocentre of the species
OCTAHEDRON = np.asarray(POLYHEDRA["octahedral"].vertex_dirs, float)


def _drawing(monkeypatch, edit):
    """Pass each RDKit conformer draw through `edit(mol, ids, batch)`; return the parameters of every draw."""
    native, drawn = rdDistGeom.EmbedMultipleConfs, []

    def draw(mol, n, params):
        ids = list(native(mol, n, params))
        drawn.append(params)
        edit(mol, ids, len(drawn))
        return ids

    monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", draw)
    return drawn


def _volume(points):
    """Return the signed volume the next three points span from the first: its sign is the core's hand."""
    return float(np.linalg.det(points[1:4] - points[0]))


def test_graft_frozen_rotates_a_core_never_mirroring_it():
    """It rotates the core onto each seed. A pair keeps its first atom and axis and takes the grafted distance; one atom
    has nothing to restore."""
    mol = embedded("C(F)(Cl)Br")  # C0 F1 Cl2 Br3
    seed = mol.GetConformer().GetPositions()
    mirror = seed * [-1, 1, 1] + 5.0
    graft_frozen(mol, [0], dict(enumerate(mirror[:4])))
    core = mol.GetConformer().GetPositions()[:4]
    spans = [np.linalg.norm(points[:4, None] - points[:4], axis=2) for points in (core, mirror)]
    np.testing.assert_allclose(*spans, atol=1e-9)
    assert np.sign(_volume(core)) == np.sign(_volume(mirror)) != np.sign(_volume(seed))
    np.testing.assert_allclose(core.mean(axis=0), seed[:4].mean(axis=0), atol=1e-6)

    pair = embedded("CCCl")
    seed = pair.GetConformer().GetPositions()
    graft_frozen(pair, [0], {1: (0.0, 0.0, 0.0), 2: (2.4, 0.0, 0.0)})
    moved = pair.GetConformer().GetPositions()
    np.testing.assert_array_equal(moved[1], seed[1])
    np.testing.assert_allclose(moved[2] - moved[1], 2.4 * (seed[2] - seed[1]) / np.linalg.norm(seed[2] - seed[1]))

    graft_frozen(pair, [0], {1: (9.0, 9.0, 9.0)})
    np.testing.assert_array_equal(pair.GetConformer().GetPositions(), moved)


def test_fixed_core_keeps_its_geometry_in_seeds_and_results():
    """An index fix holds the input's own core, a coordinate fix the grafted one, here stretched 5 percent."""
    ether = embedded("CCOC")
    own = ether.GetConformer().GetPositions()[:4]
    stretched = dict(enumerate(1.05 * own))
    seeded = seeds(ether, 1, fix=[0, 1, 2, 3], seed=1)
    relaxed = rx.minimize(ether, fix=stretched)
    held = [result.mol.GetConformer(result.ids[0]).GetPositions()[:4] for result in (seeded, relaxed)]
    found, expected = ([np.linalg.norm(p[:, None] - p, axis=2) for p in cores] for cores in (held, (own, 1.05 * own)))
    np.testing.assert_allclose(found, expected, atol=1e-6)


def test_seeds_require_names_why_no_seed_was_kept():
    """It never says the species cannot exist."""
    platinum = read_species(rx.isomers("N->[Pt+2](<-[Cl-])(<-[Br-])<-P", geometry="SPL")[0])
    for selected, reason in (
        (True, "no DG seed has the requested metal and ligand stereo; try another isomer"),
        (False, "no conformer satisfied the constraints; try another isomer"),
    ):
        with pytest.raises(rx.EmbeddingError, match=f"{reason}$") as caught:
            Seeds(mol=platinum.mol, ids=[], selected=selected).require(platinum, Constraints())
        assert str(caught.value).count(str(platinum)) == 1, selected


def test_reflection_is_free_only_for_a_lone_metal_hand():
    """Only one metal hand that nothing else tells from its mirror reflects freely."""
    delta = read_species(rx.isomers(CO_ALL_CIS, geometry="OCT").select(hand="delta"))
    amine = read_species(rx.isomers(CO_AMINE, geometry="OCT")[0])
    tagged = seated(Chem.AddHs(rx.parse_smiles(TAGGED_AMINE_PD)), "SPL", [1, 2, 3, 4])
    cis = read_species(rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL").select(label="cis"))
    rows = {
        "delta Co": (delta, Constraints(), True),
        "Co frozen core": (delta, Constraints(frozen={1}), False),
        "Co held dihedral": (delta, Constraints(dihedrals={(0, 1, 2, 3): (0.0, 90.0)}), False),
        "Co chiral amine": (amine, Constraints(), False),
        "Pd tagged amine": (tagged, Constraints(), False),
        "cis Pd": (cis, Constraints(), False),
    }
    free = {name: reflection_is_free(Specification(species=species), cons) for name, (species, cons, _) in rows.items()}
    assert free == {name: row[2] for name, row in rows.items()}


def _onto(species_mol, frame):
    """Return a copy of `species_mol` with the conformer of `frame`, a Mol of the same atom order."""
    mol = Chem.Mol(species_mol)
    mol.RemoveAllConformers()
    mol.AddConformer(Chem.Conformer(frame.GetConformer()), assignId=True)
    return mol


def test_reflect_to_specification_reflects_only_full_mirrors():
    """A conformer is reflected only where its mirror shows the whole specification. Every metal hand, ligand point and
    axis must miss and nothing else may; a mirror that misses too is put back."""
    delta_mol = rx.isomers(CO_ALL_CIS, geometry="OCT").select(hand="delta")
    delta = read_species(delta_mol)
    collapsed = on_its_vertices(delta_mol)
    pos = collapsed.GetConformer().GetPositions()
    pos[:7] = 0.0  # the metal and its donors on one point, which shows no hand; the hydrogens stay apart
    collapsed.GetConformer().SetPositions(pos)
    diol = embedded("C[C@@H](O)[C@@H](C)O")  # (R,R): two ligand points
    pentenol = embedded("C/C=C/[C@@H](C)O")  # one point beside an E bond
    found, expected = {}, {}
    for name, species, mol, reflected in (
        ("Co as specified", delta, on_its_vertices(delta_mol), False),
        ("Co mirrored", delta, on_its_vertices(delta_mol, OCTAHEDRON * [-1, 1, 1]), True),
        ("Co swapped", delta, on_its_vertices(delta_mol, OCTAHEDRON[[3, 1, 2, 0, 4, 5]]), False),
        ("Co collapsed", delta, collapsed, False),  # its mirror misses too
        ("diol mirrored", read_species(diol), mirrored(diol), True),
        ("diol one inverted", read_species(diol), _onto(diol, embedded("C[C@@H](O)[C@H](C)O")), False),
        ("pentenol Z", read_species(pentenol), _onto(pentenol, embedded("C/C=C\\[C@@H](C)O")), False),
    ):
        before, cid = mol.GetConformer().GetPositions(), mol.GetConformer().GetId()
        returned = reflect_to_specification(Specification(species=species), mol, [cid])
        moved = np.array_equal(mol.GetConformer().GetPositions(), before * ([-1, 1, 1] if reflected else 1))
        found[name] = (returned, moved)
        expected[name] = ([cid] * reflected, True)
    assert found == expected


def test_cobalt_amine_seeds_reflect_or_redraw_wrong_hands(monkeypatch):
    """A first batch of mirror images is reflected back to the specified Co hand and C5; wrong Co hands on the
    specified C5 have no mirror that fixes them, nor has any seed under a signed dihedral hold, so a second batch is
    drawn instead.
    """
    species = rx.isomers(CO_AMINE, geometry="OCT")[0]
    centre = read_species(species)
    specification = Specification(species=read_species(species))
    found, expected = {}, {}
    for name, carbon_turned_back, dihedral, redrawn in (
        ("mirror images", False, None, False),
        ("wrong Co hands", True, None, True),
        ("held dihedral", False, {(4, 5, 7, 8): (-180.0, 180.0)}, True),
    ):

        def mirror_first_batch(mol, ids, batch, carbon_turned_back=carbon_turned_back):
            if batch > 1:
                return
            hydrogen = next(n.GetIdx() for n in mol.GetAtomWithIdx(5).GetNeighbors() if n.GetAtomicNum() == 1)
            for cid in ids:
                if orientation_parity_of(mol, cid, centre) == 1:
                    reflect(mol, cid)
                    if carbon_turned_back:  # the H through its C turns C5 back, leaving the Co hand wrong
                        pos = mol.GetConformer(cid).GetPositions()
                        pos[hydrogen] = 2 * pos[5] - pos[hydrogen]
                        mol.GetConformer(cid).SetPositions(pos)

        with monkeypatch.context() as patch:
            drawn = _drawing(patch, mirror_first_batch)
            conformers = seeds(species, 1, constrain=dihedral, seed=4, prune_rms=-1)
        misses = [conformer_misses(specification, conformers.mol, cid) for cid in conformers.ids]
        found[name] = (len(conformers), len(drawn) > 1, drawn[1].randomSeed if len(drawn) > 1 else None, misses)
        retry = Sampling(seed=4).retry("seed batch", 1).seed if redrawn else None
        expected[name] = (1, redrawn, retry, [()])
    assert found == expected


def test_seeds_whose_only_miss_is_their_seating_fill_in(monkeypatch):
    """A Delta all-cis cobalt none of whose draws seats its donors as specified keeps them: relaxation can seat them."""
    species = rx.isomers(CO_ALL_CIS, geometry="OCT").select(hand="delta")
    specification = Specification(species=read_species(species))

    def unseat(mol, ids, _batch):  # every draw seats amine N0 and chloride Cl5 in each other's slots
        for cid in ids:
            pos = mol.GetConformer(cid).GetPositions()
            pos[[0, 3, 2, 5, 4, 6]] = pos[1] + 2.0 * OCTAHEDRON[[3, 1, 2, 0, 4, 5]]
            mol.GetConformer(cid).SetPositions(pos)

    _drawing(monkeypatch, unseat)
    conformers = seeds(species, 1, seed=1)

    kinds = [{miss.kind for miss in conformer_misses(specification, conformers.mol, cid)} for cid in conformers.ids]
    assert kinds == [{"seating"}]


def test_rdkit_chirality_is_unchecked_only_for_ligand_points(monkeypatch):
    """Seeding leaves RDKit's chirality check off only for ligand point stereo. RDKit's check also tests sphere windows
    only the relax meets, so a specified E/Z imine keeps it on."""
    drawn = _drawing(monkeypatch, lambda _mol, _ids, _batch: None)
    checked = {}
    for name, species in (
        ("E imine Pt", rx.isomers("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", geometry="SPL")[0]),
        ("chiral amine Pt", rx.isomers("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")[0]),
    ):
        drawn.clear()
        kept = len(seeds(species, 1, seed=1).ids) > 0
        checked[name] = (kept, {params.enforceChirality for params in drawn})
    assert checked == {"E imine Pt": (True, {True}), "chiral amine Pt": (True, {False})}


def test_seed_conformers_hold_terminal_carbonyls_straight():
    """ETKDG keeps M-C-O straight only across an M-C bond it sees, so the seed graph carries it."""
    molybdenum = rx.isomers("[O+]#[C-]->[Mo](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-P(C)(C)C")[0]
    conformers = seeds(molybdenum, 1, seed=1)
    conf = conformers.mol.GetConformer(conformers.ids[0])
    carbonyls = [
        (atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()) for atom in molybdenum.GetAtoms() if atom.GetSymbol() == "O"
    ]
    assert min(GetAngleDeg(conf, 2, carbon, oxygen) for oxygen, carbon in carbonyls) >= 178.0


def _pinned(frame, atoms):
    """Return a coordinate fix of `atoms` at their places on conformer 0 of `frame`."""
    return {atom: tuple(frame.GetConformer().GetPositions()[atom]) for atom in atoms}


def _other_hand(mol, centre):
    """Return a copy of `mol` with the tag of `centre` mirrored."""
    other = Chem.Mol(mol)
    other.GetAtomWithIdx(centre).SetChiralTag(mirror_tag(other.GetAtomWithIdx(centre).GetChiralTag()))
    return other


def test_prepare_refuses_a_graft_pinning_another_species():
    """The refusal comes before any seed. A graft must seat the sphere on the species' own vertices, and three pinned
    neighbours of a specified centre must stand in its hand, read from their own tetrahedron: dative bonds as single,
    partners' hands unread.
    """
    frame = [(2.3, 0, 0), (0, 0, 0), (0, 2.3, 0), (-2.05, 0, 0), (0, -2.05, 0)]  # Cl0 Pt1 Cl2 N3 N4, cis
    cis = placed(rx.parse_smiles("[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[NH3]"), frame)
    trans = Chem.RemoveHs(rx.isomers("[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[NH3]", geometry="SPL").select(label="trans"))
    trans.AddConformer(Chem.Conformer(cis.GetConformer()), assignId=True)
    triol = embedded("C[C@@H](O)[C@H](O)[C@H](C)O")  # C3 pseudoasymmetric: its code needs its partners' hands
    triol_pins = _pinned(triol, [3, *(n.GetIdx() for n in triol.GetAtomWithIdx(3).GetNeighbors())])
    phosphine = embedded("C[P@](CC)c1ccccc1")  # the atom order of the bound phosphine below, P1 its lone pair Pt
    bound = rx.isomers("C[P@](CC)(c1ccccc1)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")[0]
    alcohol = embedded("C[C@H](N)O")
    for name, species_mol, fix, refusal in (
        ("cis onto cis", cis, _pinned(cis, [0, 2, 3]), None),
        ("cis onto trans", trans, _pinned(cis, [0, 2, 3]), r"pins sphere atoms \[0, 2, 3\] off"),
        ("triol own hand", triol, triol_pins, None),
        ("triol other hand", _other_hand(triol, 3), triol_pins, r"stereocentre\(s\) \[3\] in"),
        ("phosphine own hand", bound, _pinned(phosphine, [0, 1, 2, 4]), None),
        ("phosphine mirror hand", bound, _pinned(mirrored(phosphine), [0, 1, 2, 4]), r"\[1\] in the other"),
        ("aminoethanol two pins", _other_hand(alcohol, 1), _pinned(alcohol, [0, 3]), None),
    ):
        species = read_species(species_mol)
        with pytest.raises(ValueError, match=refusal) if refusal else contextlib.nullcontext():
            prepare(species, resolve_core(species.mol, fix=fix))
            assert refusal is None, name


def test_prepare_keeps_contacts_no_structural_term_owns():
    """A water held to an ammine is kept as a contact; a soft Pd-Cl window is the coordination term's to set."""
    wet = read_species(rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N.O", geometry="SPL").select(label="cis"))  # O5
    model = prepare(wet, resolve_core(wet.mol)).cons

    prepared = prepare(wet, resolve_core(wet.mol), contacts={(1, 2): (2.6, 2.8), (0, 5): (2.8, 3.2)})

    assert prepared.cons.contacts.distances == {(0, 5)}
    assert prepared.cons.distances[0, 5] == (2.8, 3.2)
    assert prepared.cons.distances[1, 2] == model.distances[1, 2]
