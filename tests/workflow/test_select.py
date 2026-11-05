"""`workflow/select.py`: the per-conformer latent, its binding modes, the RMSD comparison and the `DEDUP` registry."""

import math
import warnings

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

import rxembed as rx
from rxembed.workflow import select
from tests.extras import needs
from tests.geometry_fixtures import embedded, placed
from tests.metal_fixtures import on_its_vertices

TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # a quarter turn about z
BUTANOL = embedded("CCCCO")
CIS_PD = on_its_vertices(rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL").select(label="cis"))
# O0 and O1, then O0's H2 (donating to O1) and H3, then O1's H4 and H5
_WATERS = [(0, 0, 0), (2.9, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0), (3.2, 0.5, 0.8), (3.2, 0.5, -0.8)]
WATER_DIMER = placed(Chem.AddHs(Chem.MolFromSmiles("O.O")), _WATERS)
WATERS_APART = placed(Chem.AddHs(Chem.MolFromSmiles("O.O")), np.add(_WATERS, [(0, 0, 0), (6, 0, 0), (0, 0, 0)] * 2))
_ISOPROPANOL = np.array([[1.6, 0, 0], [0, 0, 0], [-0.6, 1.3, 0.2], [-0.5, -0.8, 1.1]])
_BENZENE = np.array([[1.39 * np.cos(k * np.pi / 3), 1.39 * np.sin(k * np.pi / 3), 0.0] for k in range(6)])


def _conformers(mol, *frames):
    """Return a copy of `mol`, or a SMILES as drawn, holding one conformer per frame of positions, numbered from 0."""
    out = Chem.MolFromSmiles(mol) if isinstance(mol, str) else Chem.Mol(mol)
    out.RemoveAllConformers()
    for frame in frames:
        conformer = Chem.Conformer(out.GetNumAtoms())
        conformer.SetPositions(np.asarray(frame, float))
        out.AddConformer(conformer, assignId=True)
    return out


def _swapped_methylamines():
    """Return two methylamines and a water, then the two amines swapped, turned and moved, atoms interspersed.

    Returns the Mol, its conformers 17 and 29, and their positions.
    """
    mol = embedded("CN.CN.O", seed=17)
    fragments = Chem.GetMolFrags(mol)
    reference = mol.GetConformer().GetPositions()
    for number, fragment in enumerate(fragments):
        reference[list(fragment)] += [5.0 * number, number**2, 0]
    swapped = np.arange(mol.GetNumAtoms())
    swapped[list(fragments[0])], swapped[list(fragments[1])] = fragments[1], fragments[0]
    candidate = reference[swapped] @ TURN + 7.0
    mol = _conformers(mol, reference, candidate)
    for conformer, cid in zip(mol.GetConformers(), (17, 29), strict=True):
        conformer.SetId(cid)
    interspersed = [*range(1, mol.GetNumAtoms(), 2), *range(0, mol.GetNumAtoms(), 2)]
    return Chem.RenumberAtoms(mol, interspersed), reference[interspersed], candidate[interspersed]


def _relabelled_turned(mol, count, seed):
    """Return `count` copies of `mol`'s conformer, each relabelled by a random automorphism, turned and jittered."""
    rng, positions = np.random.default_rng(seed), mol.GetConformer().GetPositions()
    relabellings = mol.GetSubstructMatches(mol, uniquify=False, maxMatches=5000)
    frames = []
    for _ in range(count):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        turn = q * np.sign(np.diag(r))
        turn = turn if np.linalg.det(turn) > 0 else -turn
        relabelled = positions[list(relabellings[rng.integers(len(relabellings))])]
        frames.append(relabelled @ turn + rng.normal(scale=0.01, size=positions.shape))
    return _conformers(mol, *frames)


def _ethanol_rotamers():
    """Return ethanol at hydroxyl dihedrals 60, 180 and 180 deg, the last with two methyl hydrogens exchanged."""
    ethanol = embedded("CCO", seed=17)
    hydroxyl = next(a.GetIdx() for a in ethanol.GetAtomWithIdx(2).GetNeighbors() if a.GetAtomicNum() == 1)
    methyl = [a.GetIdx() for a in ethanol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    frames = []
    for angle in (60.0, 180.0, 180.0):
        rdMolTransforms.SetDihedralDeg(ethanol.GetConformer(), 0, 1, 2, hydroxyl, angle)
        frames.append(ethanol.GetConformer().GetPositions())
    frames[2][methyl[:2]] = frames[2][methyl[1::-1]]
    return _conformers(ethanol, *frames)


def _sodium_cluster(*frames):
    return _conformers(".".join(["[Na+]"] * len(frames[0])), *frames)


# Its closest fit swaps the sodiums (0.489 A) but moves one 1.03 A; the stored order fits to 0.491 A within 0.98 A.
_FAR = [[6.0, 0, 0], [-6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
SODIUM_PAIR = _conformers(
    "[Na+].[Na+].[K+].[Rb+].[Cs+].[Ca+2]",
    [[0.0, 0.0, 0.0], [-0.33877, 0.69414, -0.41281], *_FAR],
    [[0.25204, 0.88520, 0.54940], [-0.26140, 0.07012, -0.37305], *_FAR],
)


def test_dihedrals_reads_each_rotatable_bond_as_its_cos_and_sin():
    """Butanol has two C-C rotors, its methyl and hydroxyl none; a rotor-free ethane reads one zero."""
    butanol = Chem.Mol(BUTANOL)
    rdMolTransforms.SetDihedralDeg(butanol.GetConformer(), 0, 1, 2, 3, 60.0)
    rdMolTransforms.SetDihedralDeg(butanol.GetConformer(), 1, 2, 3, 4, 180.0)
    for name, mol, expected in (("butanol", butanol, [0.5, np.sqrt(0.75), -1.0, 0.0]), ("ethane", embedded("CC"), [0])):
        assert np.allclose(select.dihedrals(mol, 0), expected), name


def test_metal_donors_read_the_graph_or_a_bondless_metals_shell():
    """A bonded metal names its non-metal neighbours, even none; only a bondless one is read from its shell."""
    rows = {
        "butanol": (BUTANOL, None),
        "cis-diammine-palladium": (CIS_PD, [0, 2, 3, 4]),
        "bondless-lanthanum-between-two-selenium": (
            placed("[La].[Se].[Se]", [(0, 0, 0), (3, 0, 0), (-3, 0, 0)]),
            [1, 2],
        ),
        "lanthanum-bonded-to-iron-alone": (
            placed("[La][Fe].[Se].[Se]", [(0, 0, 0), (2.5, 2.5, 0), (3, 0, 0), (-3, 0, 0)]),
            [],
        ),
    }
    found = {name: select.metal_donors(mol, [0]) for name, (mol, _donors) in rows.items()}
    assert found == {name: donors for name, (_mol, donors) in rows.items()}


def test_features_kind_names_the_most_specific_block():
    """Dihedrals always; a relative pose for several fragments, contacts where they touch, L-M-L angles at a metal."""
    rows = {
        "butanol": (BUTANOL, ("dihedral",), "conformer family"),
        "two-waters-apart": (WATERS_APART, ("dihedral", "relpose"), "relative arrangement"),
        "water-dimer": (WATER_DIMER, ("dihedral", "relpose", "nci"), "contact pattern"),
        "cis-diammine-palladium": (CIS_PD, ("dihedral", "metal"), "ligand arrangement"),
    }
    for name, (mol, blocks, kind) in rows.items():
        snapshot = select.features(mol, [0])
        assert (snapshot.blocks, snapshot.kind) == (blocks, kind), name


def test_features_read_statistics_on_fitted_conformers_alone():
    """Water dimers pulled apart step by step, four fitted and two placed: the fitted rows stay as they were."""
    frames = [np.add(_WATERS, [(0, 0, 0), (dx, 0, 0), (0, 0, 0)] * 2) for dx in (0, 0.5, 1.1, 2.1, -0.2, 3.1)]
    waters = _conformers(WATER_DIMER, *frames)

    fitted = select.features(waters, [0, 1, 2, 3]).matrix

    assert np.allclose(select.features(waters, [0, 1, 2, 3], [4, 5]).matrix[:4], fitted)


def test_mode_signature_names_contacts_and_metal_arrangement():
    """Contact kinds with the fragments they bridge and the metal's arrangement label; None for torsions alone."""
    rows = {
        "butanol": (BUTANOL, None),
        "water-dimer": (WATER_DIMER, [(("nci", frozenset({("HB", (0, 1))})),)]),
        "cis-diammine-palladium": (CIS_PD, [(("metal", "cis"),)]),
    }
    found = {name: select.mode_signature(mol, [0], select.features(mol, [0])) for name, (mol, _sig) in rows.items()}
    assert found == {name: signature for name, (_mol, signature) in rows.items()}


def test_cluster_on_calls_fewer_than_three_conformers_one_mode():
    assert select.cluster_on(np.zeros((2, 4))).tolist() == [0, 0]


@needs("scipy")
def test_compare_returns_the_order_superimposing_the_candidate():
    """Methylamines swapped, turned and moved: every reference atom finds its image, the hydrogens left out or not."""
    mol, reference, candidate = _swapped_methylamines()
    for hydrogens in (False, True):
        comparison = select.compare(mol, 17, 29, max_rmsd=1e-5, hydrogens=hydrogens)
        superimposed = (candidate[list(comparison.order)] - 7.0) @ TURN.T
        assert np.allclose(superimposed, reference), hydrogens
        assert max(comparison.rmsd, comparison.maximum) < 1e-5, hydrogens
    assert np.array_equal(mol.GetConformer(29).GetPositions(), candidate)


@needs("scipy")
def test_compare_takes_the_closest_automorphism_passing_gates():
    """A sodium pair fits by its stored order, not the closer swap; a stretched benzene fits by none."""
    rows = {
        "sodium-pair": (SODIUM_PAIR, (0, 1), 0.5, tuple(range(6))),
        "benzene-scaled-by-1.08": (_conformers("c1ccccc1", _BENZENE, 1.08 * _BENZENE), (0, 1), 0.1, None),
    }
    for name, (mol, pair, max_rmsd, order) in rows.items():
        comparison = select.compare(mol, *pair, max_rmsd=max_rmsd)
        assert (comparison and comparison.order) == order, name


@needs("scipy")
def test_rmsd_dedup_merges_only_by_symmetry_rotation_and_gates():
    """A copy merges within max_rmsd with every atom within twice it, by bond-preserving orders and proper rotations.

    Ordinary hydrogens are left out unless `hydrogens`, a hydride and H2 never; each regression row is a symmetric top,
    frame or radius bound that once merged or split the wrong pair.
    """
    swap = (_ISOPROPANOL, _ISOPROPANOL[[2, 1, 0, 3]])
    chain = np.array([[1.25 * i, 0.85 * (i % 2), 0.0] for i in range(16)])
    swung, stretched = chain.copy(), chain.copy()  # C0 moved within max_rmsd overall: beyond twice it, or between
    swung[0] += (0, 0, 1.5)
    stretched[0] += (-0.7, 0, 0)
    hydride = rx.parse_smiles("[H-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    square_pt = np.array([[1.55, 0, 0], [0, 0, 0], [-2.3, 0, 0], [0, 2.3, 0], [0, -2.3, 0]])
    tilted_pt = np.vstack([[1.1, 0, 1.1], square_pt[1:]])  # the hydride alone out of the PtCl3 plane
    h2 = [[[0, 0, 0], [length, 0, 0]] for length in (0.74, 1.4, 0.74)]
    glycol = np.array([[-1.9, 0.6, 0.0], [-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [1.9, 0.6, 0.0]])
    cot = embedded("C1=CC=CC=CC=C1")
    bond_shift = [(a + 1) % 8 if a < 8 else 8 + (a - 7) % 8 for a in range(16)]  # each double bond onto a single
    cot_frames = [cot.GetConformer().GetPositions()[order] for order in (range(16), bond_shift)]
    phenylethanol = embedded("C[C@H](O)c1ccccc1")
    hands = [phenylethanol.GetConformer().GetPositions() * side for side in (1, [-1, 1, 1])]
    tetraphenylmethane = embedded("c1ccc(cc1)C(c1ccccc1)(c1ccccc1)c1ccccc1", seed=3)
    rdForceFieldHelpers.MMFFOptimizeMolecule(tetraphenylmethane)  # its para C-H bonds point at its centre
    phenyls = tetraphenylmethane.GetConformer().GetPositions()
    relabelling = list(tetraphenylmethane.GetSubstructMatches(tetraphenylmethane, uniquify=False, maxMatches=1000)[19])
    turned = phenyls[relabelling] @ TURN + np.random.default_rng(19).normal(scale=0.005, size=phenyls.shape)
    triad = ([[-3.0, 0, 0], [0, 0, 0], [3.0, 0, 0]], [[0, -3.0, 0], [0, 0, 0], [0, 3.0, 0]])
    crowded = [[0, 3, 0], [0, -3, 0], [0.5, 0, 3], [-0.5, 0, 3], [0.5, 0, -3], [-0.5, 0, -3]]  # two ions per vertex
    octahedron = [[3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0], [0, 0, 3], [0, 0, -3]]
    hexamer = np.array([[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5], [-2.5, 0.4, 0], [0.3, -2.5, 0.6], [0.2, 0.5, -2.5]])
    copy = hexamer[[1, 2, 0, 4, 5, 3]] @ TURN
    copy[[0, 4]] += (0.03, 0, 0)  # its mirror image shares its radii exactly, so the radius bound ranks it nearer
    rows = {
        "isopropanol-methyls-swapped": (_conformers("CC(C)O", *swap), 1e-5, False, [0]),
        "atom-mapped-isopropanol-methyls-swapped": (
            _conformers("[CH3:1][CH:2]([CH3:3])[OH:4]", *swap),
            1e-5,
            False,
            [0],
        ),
        "propanol-ends-swapped": (_conformers("CCCO", *swap), 1e-5, False, [0, 1]),
        "13c-isopropanol-methyls-swapped": (_conformers("[13CH3]C(C)O", *swap), 1e-5, False, [0, 1]),
        "isopropanol-turned-and-moved": (_conformers("CC(C)O", swap[0], swap[0] @ TURN + 5.0), 1e-3, False, [0]),
        "dihydrogen-stretched": (_conformers("[H][H]", *h2), 0.1, False, [0, 1]),
        "platinum-hydride-tilted": (_conformers(hydride, square_pt, tilted_pt), 0.1, False, [0, 1]),
        "ethanol-hydroxyl-rotamers-hydrogens-left-out": (_ethanol_rotamers(), 0.1, False, [0]),
        "ethanol-hydroxyl-rotamers-hydrogens-compared": (_ethanol_rotamers(), 0.1, True, [0, 1]),
        "hexadecane-end-swung-out-of-plane": (_conformers("C" * 16, chain, swung), 0.5, False, [0, 1]),
        "hexadecane-end-stretched-outward": (_conformers("C" * 16, chain, stretched), 0.5, False, [0]),
        "benzene-scaled-by-1.07": (_conformers("c1ccccc1", _BENZENE, 1.07 * _BENZENE), 0.1, False, [0]),
        "benzene-scaled-by-1.08": (_conformers("c1ccccc1", _BENZENE, 1.08 * _BENZENE), 0.1, False, [0, 1]),
        "glycol-oxygens-trading-carbons": (_conformers("OCCO", glycol, glycol[[3, 1, 2, 0]]), 0.5, False, [0, 1]),
        "cyclooctatetraene-bond-shifted": (_conformers(cot, *cot_frames), 0.02, False, [0, 1]),
        "1-phenylethanol-mirrored": (_conformers(phenylethanol, *hands), 0.1, False, [0, 1]),
        "sodium-pair-swap-fits-closest-but-moves-one-ion-too-far": (SODIUM_PAIR, 0.5, False, [0]),
        "sodium-triad-turned-about-its-middle-ion": (_sodium_cluster(*triad), 0.5, False, [0]),
        "sodium-hexamer-crowding-two-octahedron-vertices": (_sodium_cluster(crowded, octahedron), 0.5, False, [0, 1]),
        "relabelled-sodium-hexamer-beside-its-mirror-image": (
            _sodium_cluster(hexamer, copy * [-1, 1, 1], copy),
            0.1,
            False,
            [0, 1],
        ),
        "relabelled-turned-neopentane": (_relabelled_turned(embedded("CC(C)(C)C"), 2, 17), 0.5, True, [0]),
        "relabelled-turned-tetraphenylmethane": (_conformers(tetraphenylmethane, phenyls, turned), 0.5, True, [0]),
    }
    for name, (mol, max_rmsd, hydrogens, kept) in rows.items():
        ids = [conformer.GetId() for conformer in mol.GetConformers()]
        energies = [0.1 * k for k in range(len(ids))]
        assert select.apply(mol, ids, energies, max_rmsd=max_rmsd, hydrogens=hydrogens) == kept, name


@needs("scipy")
def test_nearest_kept_maps_each_dropped_conformer():
    """Benzene scaled by 1.15 sits nearer the 1.2 copy than the unscaled ring: 1.39 x 0.05 A."""
    benzenes = _conformers("c1ccccc1", _BENZENE, 1.2 * _BENZENE, 1.15 * _BENZENE)
    assert select.nearest_kept(benzenes, [0, 1], [2]) == {2: (1, pytest.approx(0.0695, abs=1e-3))}


@needs("scipy")
def test_each_dedup_method_keeps_the_lower_copy_in_its_window():
    """A copy merges into the lower-energy one, stays 20 kcal/mol apart or unscored, and kept ids come energy-sorted."""
    position = BUTANOL.GetConformer().GetPositions()
    copies = _conformers(BUTANOL, position, position)
    rows = {"degenerate": ([0.0, 0.01], [0]), "20 apart": ([20.0, 0.0], [1, 0]), "unscored": ([math.inf] * 2, [0, 1])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # inf - inf lies outside every window, without a warning
        kept = {
            (m, name): select.apply(copies, [0, 1], energies, m)
            for m in select.DEDUP
            for name, (energies, _) in rows.items()
        }
    assert kept == {(m, name): ids for m in select.DEDUP for name, (_, ids) in rows.items()}


def test_apply_refuses_an_unknown_method_naming_the_choices():
    for method in ("cluster", ["rmsd"]):
        with pytest.raises(ValueError, match=r"'rmsd'.*representatives\(\)"):
            select.apply(BUTANOL, [0], [0.0], method=method)


def test_energy_prune_drops_energies_near_a_kept_one():
    """0.04 falls within 0.05 of the kept 0.0; 0.08 does not, though it lies within 0.05 of the dropped 0.04."""
    assert select.energy_prune([0.0, 0.04, 0.08], energy_tol=0.05).tolist() == [True, False, True]


@needs("scipy")
@pytest.mark.slow
def test_rmsd_dedup_finishes_on_pd_tetrakis_triphenylphosphine():
    phosphine = "P(c1ccccc1)(c1ccccc1)c1ccccc1"
    species = rx.isomers(f"{phosphine}->[Pd](<-{phosphine})(<-{phosphine})<-{phosphine}")[0]
    position = rx.embed(species, n=1, seed=1).mol.GetConformer().GetPositions()
    assert select.apply(_conformers(species, position, position @ TURN), [0, 1], [0.0, 0.1]) == [0]
