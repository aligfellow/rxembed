"""`geometry/constraints.py`: the `Constraints` record, its merges, coplanar rows and `fix=`/`constrain=` resolution."""

import pickle
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.geometry.constraints import (
    COPLANAR_KINDS,
    Constraints,
    CoplanarRow,
    Improper,
    SoftKeys,
    compose,
    donor_directions,
    resolve_atom,
    resolve_core,
    retained_soft,
)
from rxembed.metal.sphere import compile_constraints
from tests.geometry_fixtures import embedded
from tests.metal_fixtures import enumerated

# C9 is a haptic centroid over H4-H6: a key reaches it, yet it is no atom MC must hold
_HELD = Constraints(
    distances={(0, 1): (1.4, 1.6), (0, 2): (2.0, 3.0), (8, 9): (1.6, 1.8)},
    angles={(0, 1, 2): (100.0, 120.0)},
    dihedrals={(0, 1, 2, 3): (50.0, 70.0)},
    planes=[((0, 1, 2), (3, 4, 5), 3.5)],
    pulls={(0, 1, 2): (110.0, 110.0), (0, 2): (2.5, 2.5)},
    fixed={(0, 3): (2.0, 3.0)},
    frozen={7},
    haptic={9: (4, 5, 6)},
    contacts=SoftKeys(distances={(0, 2), (0, 3)}, angular={(0, 1, 2), (0, 1, 2, 3)}),
)


def test_constraints_own_frozen_copies_of_their_fields():
    window, ring = np.array([2.0, 3.0]), [3, 4, 5]
    cons = Constraints(distances={(0, 2): window}, planes=[((0, 1, 2), ring, 3.5)])
    window[:], ring[0] = (5.0, 6.0), 9
    assert (cons.distances[(0, 2)], cons.planes) == ((2.0, 3.0), (((0, 1, 2), (3, 4, 5), 3.5),))
    with pytest.raises(TypeError, match="frozen"):
        cons.fixed[(0, 3)] = (3.0, 3.0)
    assert pickle.loads(pickle.dumps(_HELD)) == _HELD
    for pulls, message in (
        ({(0, 1, 2, 3): (1, 1)}, r"write a target t as \(t, t\)"),
        ({(0, 1, 2): (181, 181)}, "0 and 180"),
    ):
        with pytest.raises(ValueError, match=message):
            Constraints(pulls=pulls)


def test_relaxed_releases_only_the_soft_contacts():
    relaxed = _HELD.relaxed()
    assert relaxed.distances.keys() == {(0, 1), (0, 3), (8, 9)}
    assert (relaxed.angles, relaxed.dihedrals, relaxed.planes, relaxed.pulls) == ({}, {}, (), {(0, 2): (2.5, 2.5)})
    assert (bool(_HELD.contacts), bool(relaxed.contacts)) == (True, False)


def test_constrained_atoms_are_held_atoms_but_a_haptic_centroid():
    assert _HELD.constrained_atoms() == {0, 1, 2, 3, 4, 5, 7, 8}


def test_donor_directions_come_from_held_angles_or_donor_rows():
    window, rows = (100.0, 120.0), [CoplanarRow(kind="donor_torsion", atoms=(9, 10, 11, 12), cap=15.0)]
    angles = {(9, 0, 1): window, (9, 2, 3): (0.0, 180.0), (9, 4, 5): window, (9, 6, 7): window, (1, 8, 2): window}
    released = {(9, 4, 5), (7, 6, 9)}  # (9, 6, 7) is also fixed, which keeps its authority
    assert donor_directions(angles, rows, {9}, releasable=released, fixed={(9, 6, 7)}) == {0, 6, 10}


def test_compose_merges_each_field_by_its_rule():
    bond, angle, torsion, window, pull = (0, 1), (0, 1, 2), (0, 1, 2, 3), (2.3, 2.5), (120.0, 120.0)
    fixed = Constraints(fixed={bond: (2.0, 2.0), angle: (110.0, 110.0), torsion: (-180.0, -180.0)})
    derived = Constraints(distances={bond: window}, angles={angle[::-1]: window}, dihedrals={torsion[::-1]: window})
    both = [compose(fixed, replace(derived, pulls={bond: window})), compose(derived, fixed)]  # a fix wins either way
    held = ({bond: pytest.approx((1.98, 2.02))}, {angle: (108, 112)}, {torsion: (178, 182)}, {})
    assert [(merged.distances, merged.angles, merged.dihedrals, merged.pulls) for merged in both] == [held] * 2
    assert compose(fixed, Constraints(fixed={torsion: (180.0, 180.0)})).fixed[torsion] == (180.0, 180.0)
    first = Constraints(distances={bond: (1, 2)}, floors={bond: 3.0}, dg_floors={bond: 2.0}, conjugation=False)
    second = Constraints(distances={bond: (3, 4)}, floors={bond: 2.5}, dg_floors={bond: 1.5}, pulls={angle[::-1]: pull})
    merged = compose(first, second, Constraints(pulls={angle: pull}))
    assert (merged.distances[bond], merged.conjugation) == ((3.0, 4.0), False)
    assert (merged.floors[bond], merged.dg_floors[bond], merged.pulls) == (3.0, 1.5, {angle: pull})
    for parts, message in (
        ((fixed, Constraints(fixed={bond: (2.1, 2.1)})), "conflicting fixed"),
        ((Constraints(haptic={9: (0, 1, 2)}), Constraints(haptic={9: (3, 4, 5)})), "conflicting haptic"),
        ((Constraints(pulls={angle: pull}), Constraints(pulls={angle: (110.0, 110.0)})), "conflicting pulls"),
    ):
        with pytest.raises(ValueError, match=message):
            compose(*parts)


def test_retained_soft_keeps_terms_only_where_the_base_is_free():
    """A fixed torsion owns its bond axis, a coplanar row its j-k bond and a shell improper only its own four atoms."""
    base = Constraints(
        distances={(0, 1): (1.4, 1.6), (0, 2): (2.0, 3.0)},
        angles={(0, 1, 2): (100.0, 120.0), (15, 16, 17): (100.0, 110.0)},
        fixed={(0, 1, 2, 3): (-60.0, -60.0)},
        coplanar=[CoplanarRow(kind="ligand_ez", atoms=(9, 10, 11, 12), cap=15.0, anchor=180.0)],
        umbrellas={(5, 6, 7, 8): Improper(kind="plane")},
        contacts=SoftKeys(distances={(0, 2)}, angular={(15, 16, 17)}),
    )
    window = (80.0, 100.0)
    pulls = {(1, 0): (1.5, 1.5), (3, 4): (2.5, 2.5), (2, 1, 0): (110.0, 110.0), (1, 2, 3): (95.0, 95.0)}
    keeps = {(3, 4): True, (1, 0): False, (1, 2, 3): True, (2, 1, 0): False, (1, 2, 3, 6): True, (13, 6, 7, 14): True}
    keeps |= {(4, 1, 2, 5): False, (13, 10, 11, 14): False, (8, 7, 6, 5): False}
    terms = {size: {key: window for key in keeps if len(key) == size} for size in (2, 3, 4)}
    soft = SoftKeys(distances=set(terms[2]), angular={*terms[3], *terms[4]})
    incoming = Constraints(distances=terms[2], angles=terms[3], dihedrals=terms[4], pulls=pulls, contacts=soft)
    retained = retained_soft(base, incoming)
    assert {*retained.distances, *retained.angles, *retained.dihedrals} == {k for k, kept in keeps.items() if kept}
    assert retained.pulls.keys() == {(3, 4), (1, 2, 3)}
    assert retained.contacts == SoftKeys(distances={(3, 4)}, angular={(1, 2, 3), (1, 2, 3, 6), (13, 6, 7, 14)})
    for overlap in (Constraints(distances={(2, 0): (2.5, 2.8)}), Constraints(angles={(17, 16, 15): (90.0, 95.0)})):
        with pytest.raises(ValueError, match="state each degree of freedom once"):
            retained_soft(base, overlap)


# A species whose compile writes rows of each coplanar kind (a pyridine N's plane, a ketone O's torsion). A new kind
# adds its row here.
_PYRIDINE_KETONE = ("CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1", "square_planar")
COPLANAR_SPECIES = {
    "donor_plane": _PYRIDINE_KETONE,
    "donor_torsion": _PYRIDINE_KETONE,
    "ring_hinge": ("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral"),
    "metal_ez": ("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar"),
    "ligand_ez": (r"C/C=C/C(=O)[O-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar"),
}


def test_each_coplanar_kind_reads_its_rows_as_their_graph_does():
    assert COPLANAR_SPECIES.keys() == COPLANAR_KINDS.keys()
    read, expected = {}, {}
    for kind, (smiles, geometry) in COPLANAR_SPECIES.items():
        iso = enumerated(Chem.AddHs(rx.parse_smiles(smiles)), geometry)[0]
        cons, bond = compile_constraints(iso), iso.mol.GetBondBetweenAtoms
        rows = [row for row in cons.coplanar if row.kind == kind]
        for row in rows:
            i, j, *far = row.atoms
            read[kind, row.atoms] = (row.out_of_plane, row.holds_donor, row.seeded)
            expected[kind, row.atoms] = (all(bond(j, atom) for atom in far), i in cons.metals, row.anchor is None)
        with pytest.raises(ValueError, match="anchor"):
            replace(rows[0], anchor=180.0 if rows[0].seeded else None)
    assert read == expected


def test_improper_holds_the_well_or_side_its_kind_names():
    rows = (("plane", 170, 180), ("plane", -20, 0), ("pyramid", -40, -40), ("side", -5, -90))
    windows = [Improper(kind=kind, minimum=20.0).window(phi) for kind, phi, _well in rows]
    assert [lo <= well <= hi for (lo, hi), (*_row, well) in zip(windows, rows, strict=True)] == [True] * 4
    assert Improper(kind="pyramid", minimum=20.0).window(40.0) == (20.0, 90.0)
    with pytest.raises(ValueError, match="'plane', 'pyramid' or 'side'"):
        Improper(kind="tilt")


def test_fix_resolves_to_a_shape_held_graft_or_canonical_values(tmp_path):
    mol = embedded("CCO")
    own = mol.GetConformer().GetPositions()
    held = resolve_core(mol, fix=[0, 1, 2])
    assert (held.graft, held.cons.frozen, held.cons.fixed) == ({i: tuple(own[i]) for i in (0, 1, 2)}, {0, 1, 2}, {})
    assert all(lo < np.linalg.norm(own[i] - own[j]) < hi for (i, j), (lo, hi) in held.cons.distances.items())
    reference, path, core = Chem.Mol(mol), tmp_path / "ethanol.xyz", {0: 0, 1: 1, 2: 2}
    reference.GetConformer().SetPositions(own + 1.0)
    Chem.MolToXYZFile(reference, str(path))
    shifted = {i: pytest.approx(own[i] + 1.0, abs=1e-5) for i in core}  # an .xyz keeps six decimals
    sources = (reference, own + 1.0, path, rx.wrap(reference))
    assert [resolve_core(mol, fix=rx.Template(source, core)).graft for source in sources] == [shifted] * 4
    assert resolve_core(mol, fix=rx.Template(reference, "[CH3][CH2][OH]")).graft == shifted
    assert resolve_core(mol, fix=[rx.Template(reference, core), 2]).graft == {**shifted, 2: tuple(own[2])}
    numbers = resolve_core(mol, fix={(2, 1, 0): (100.0, 120.0), (3, 0, 1, 2): 190.0, (1, 0): 1.6, 4: (0, 0, 9)})
    assert numbers.cons.fixed == {(0, 1, 2): (100.0, 120.0), (2, 1, 0, 3): (-170.0, -170.0), (0, 1): (1.6, 1.6)}
    assert numbers.graft == {4: (0.0, 0.0, 9.0)}


def test_constrain_resolves_to_releasable_soft_windows():
    mol = embedded("CCO")
    soft = resolve_core(mol, constrain={(0, 2): 2.8, (0, 1, 2): 120.0, (3, 0, 1, 2): -60.0}).cons
    assert (soft.distances[(0, 2)], soft.angles[(0, 1, 2)]) == (pytest.approx((2.7, 2.9)), (115.0, 125.0))
    assert soft.dihedrals[(2, 1, 0, 3)] == (-65.0, -55.0)
    assert soft.contacts == SoftKeys(distances={(0, 2)}, angular={(0, 1, 2), (2, 1, 0, 3)})
    assert resolve_core(mol, constrain={((0, 1, 2), (3, 4, 5)): 3.7}).cons.planes == (((0, 1, 2), (3, 4, 5), 3.7),)


def test_unholdable_fix_or_constrain_is_refused_with_a_remedy():
    ethanol, flat, toluene = embedded("CCO"), Chem.MolFromSmiles("CCO"), Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    for target, fix, constrain, message in (
        (ethanol, {}, None, "fix= is empty"),
        (ethanol, {(1, 1): 1.5}, None, "names the same atom twice"),
        (ethanol, [99], None, "atom index 99 .*outside"),
        (ethanol, {(0, 1, 2): 100.0, (2, 1, 0): 110.0}, None, "conflicting values"),
        (ethanol, {(0, 1, 2, 3): (0.0, 361.0)}, None, "no wider than 360"),
        (ethanol, {(0, 1, 2): 190.0}, None, "angle in 0-180"),
        (ethanol, {(0, 1): (2.0, 1.0)}, None, "ordered"),
        (flat, [0, 1, 2], None, "a SMILES has none"),
        (ethanol, {(0, 2): 2.4}, {(2, 0): 2.5}, "both name"),
        (ethanol, rx.Template(flat, {0: 0}), None, "needs a conformer"),
        (ethanol, rx.Template(np.zeros((3, 2)), {0: 0}), None, "reference must be"),
        (ethanol, rx.Template("no-such-file.xyz", {0: 0}), None, "could not read coordinates"),
        (ethanol, rx.Template(ethanol, {0: 100}), None, "out of range"),
        (ethanol, rx.Template(np.zeros((3, 3)), "CCO"), None, "needs two molecular graphs"),
        (ethanol, rx.Template(ethanol, "CCN"), None, "matched nothing"),
        (ethanol, rx.Template(ethanol, "[CX4]-[#6]"), None, "symmetry-equivalent orderings"),
        (toluene, rx.Template(ethanol, "[CX4]-[#6]"), None, "symmetry-equivalent orderings"),  # ordered once here
    ):
        with pytest.raises(ValueError, match=message):
            resolve_core(target, fix=fix, constrain=constrain)


def test_fix_leaving_a_freedom_unheld_logs_an_advisory(caplog):
    two = {(0, 1): 1.5, (1, 2): 1.4}
    rows = (([0, 1], "2 grafted atom(s) hold no orientation"), (two, "C0-C1-O2 is free"), ({**two, (0, 2): 2.4}, ""))
    for fix, message in rows:
        caplog.clear()
        with caplog.at_level("WARNING", logger="rxembed"):
            resolve_core(embedded("CCO"), fix=fix)
        assert message in caplog.text if message else not caplog.records, fix


def test_window_inside_a_grafted_core_is_dropped_and_warned(caplog):
    collinear = {**{i: (float(i), 0.0, 0.0) for i in (0, 1, 2)}, (1, 2): 3.4}
    with caplog.at_level("WARNING", logger="rxembed"):
        cons = resolve_core(embedded("CCO"), fix=collinear, constrain={(0, 1, 2): 100.0, (2, 1, 0, 3): 60.0}).cons
    assert caplog.text.count("inside the fixed core") == 2
    assert cons.distances[(1, 2)] == pytest.approx((0.95, 1.05))  # back to the graft's own shape
    assert (cons.fixed, cons.angles, cons.contacts) == ({}, {}, SoftKeys(angular={(2, 1, 0, 3)}))


def test_match_names_one_atom_set_in_any_resonance_form():
    mol = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1CCl"))  # 2-chlorobenzyl chloride: [Cl] names both chlorines
    assert rx.match(mol, "[Cl]C[c]") == (8, 7, 6)
    thioureas = (Chem.AddHs(Chem.MolFromSmiles(s)) for s in ("C[N-]C(=S)NCC", "CN=C([S-])NCC"))
    assert {rx.match(thiourea, "NC(=S)N") for thiourea in thioureas} == {(1, 2, 3, 4)}
    attack = Chem.RWMol(Chem.MolFromSmiles("[F-].CCl"))  # an SN2 fluoride bonded to its carbon has no Lewis form
    attack.AddBond(0, 1, Chem.BondType.SINGLE)
    attack.UpdatePropertyCache(strict=False)
    refused = ((mol, "[Cl]", "matched 2 times"), (mol, "[not a smarts", "did not parse"))
    for target, smarts, message in (*refused, (mol, "[Br]", "matched nothing"), (attack, "[Br]", "matched nothing")):
        with pytest.raises(ValueError, match=message):
            rx.match(target, smarts)


def test_resolve_atom_takes_an_index_or_a_smarts_first_atom():
    mol = Chem.AddHs(Chem.MolFromSmiles("Clc1ccccc1CCl"))
    assert (resolve_atom(mol, "[Cl]C[c]"), resolve_atom(mol, 3)) == (8, 3)
