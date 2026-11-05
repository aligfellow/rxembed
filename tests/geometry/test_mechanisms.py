"""`geometry/mechanisms.py`: the `MECHANISM_ORDER` registry and the bounds and force-field terms of each mechanism."""

import itertools
import math
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed.geometry import mechanisms as mech
from rxembed.geometry.constraints import Constraints, CoplanarRow, Improper, SoftKeys
from rxembed.graph import materialise_phantoms
from tests.geometry_fixtures import embedded, placed


def _written(mol, cons, *hooks, matrix=None):
    """Return a DG context over `mol`'s native bounds, or a copy of `matrix`, after each of `hooks` writes `cons`."""
    ctx = mech.DGContext(mol, rdDistGeom.GetMoleculeBoundsMatrix(mol) if matrix is None else matrix.copy())
    for hook in hooks:
        hook(cons, ctx)
    return ctx


def _terms(mechanism, cons, mol):
    """Return the force-field calls `mechanism` makes for `cons` on `mol`'s conformer, as (name, *arguments)."""
    ff = mock.MagicMock()
    mechanism.uff_terms(ff, cons, mol.GetConformer(), 1.0)
    return [(name, *args) for name, args, _kwargs in ff.mock_calls]


def _energies(mechanism, cons, mol, frames):
    """Return the energy of `mechanism`'s terms alone at each frame of `mol`'s coordinates."""
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(mol)
    mechanism.uff_terms(ff, cons, mol.GetConformer(), 1.0)
    ff.Initialize()
    return [ff.CalcEnergy(np.ravel(frame).tolist()) for frame in frames]


AMIDE = embedded("CC(=O)NC")  # C0 C1 O2 N3 C4, H5-H11: C1 is sp2 and O2=C1-N3-C4 a conjugated quartet
# Each mechanism's own field on the amide and whether it writes bounds; None marks a UFF repair (the molecule alone).
_H5_ANTI_N3 = CoplanarRow(kind="ligand_ez", atoms=(5, 0, 1, 3), anchor=180.0, cap=15.0)
OWN_FIELD = {
    "Frozen": (Constraints(frozen={0, 1, 2}), False),
    "Distance": (Constraints(distances={(0, 4): (2.5, 3.5)}), True),
    "Pull": (Constraints(pulls={(0, 4): (3.0, 3.0)}), False),
    "Floor": (Constraints(floors={(0, 4): 3.0}, dg_floors={(0, 4): 1.0}), True),
    "Angle": (Constraints(angles={(0, 1, 3): (100.0, 110.0)}), True),
    "Dihedral": (Constraints(dihedrals={(0, 1, 3, 4): (170.0, 190.0)}), False),
    "Coplanar": (Constraints(coplanar=[_H5_ANTI_N3], angles={(5, 0, 1): (105.0, 115.0)}), True),
    "Plane": (Constraints(planes=[((0, 1, 2), (3, 4, 5), 3.6)]), True),
    "Haptic": (Constraints(haptic={11: (0, 1)}), True),
    "TrigonalAngle": (None, False),
    "Sp2Planar": (None, False),
    "ConjugationCap": (None, False),
    "Umbrella": (Constraints(umbrellas={(0, 2, 3, 1): Improper(kind="side")}), False),
}


def test_each_mechanism_writes_only_from_its_own_field():
    wrote = {}
    for mechanism in mech.MECHANISM_ORDER:
        own = OWN_FIELD[type(mechanism).__name__][0]
        for cons in (Constraints(), own or Constraints()):
            ctx = _written(AMIDE, cons, mechanism.dg_windows, mechanism.dg_relief, mechanism.dg_post)
            bounds = bool(ctx.pairs) or not np.array_equal(ctx.bm, _written(AMIDE, cons).bm)
            wrote.setdefault(type(mechanism).__name__, []).append((bounds, bool(_terms(mechanism, cons, AMIDE))))
    assert wrote == {name: [(False, own is None), (bounds, True)] for name, (own, bounds) in OWN_FIELD.items()}


def test_angle_prior_intersects_backbone_bounds_or_owns_its_span():
    """It owns its span across a stripped metal bond: propane C0-C1-C2, and methane C3 held 2.0 A from C1."""
    mol, held = Chem.AddHs(Chem.MolFromSmiles("CCC.C")), {(0, 2): (2.6, 2.7)}
    within, disjoint, owned, across = (
        _written(mol, cons, mech.Distance().dg_windows, mech.Angle().dg_windows).pairs
        for cons in (
            Constraints(angles={(0, 1, 2): (110.0, 115.0)}),
            Constraints(angles={(0, 1, 2): (60.0, 70.0)}),
            Constraints(distances=held, angles={(0, 1, 2): (60.0, 70.0)}),
            Constraints(distances={(1, 3): (2.0, 2.0)}, angles={(2, 1, 3): (90.0, 90.0)}),
        )
    )
    native = (_written(mol, held).bm[2, 0], _written(mol, held).bm[0, 2])
    assert native[0] < within[(0, 2)][0] < within[(0, 2)][1] <= native[1]
    assert (disjoint, owned) == ({(0, 2): native}, held)
    assert across[(2, 3)][0] == across[(2, 3)][1] < 3.0  # no carbon contact floor across a stripped bond


def test_held_contact_relieves_only_unowned_neighbour_floors():
    """Bromomethane's C0 holds the chloride Cl2 at 2.12 A; of its neighbours only H4 shares the deficit."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CBr.[Cl-]"))  # C0 Br1 Cl2, H3-H5
    held = {(0, 1): (1.9, 1.95), (0, 2): (2.1, 2.14), (1, 2): (3.9, 4.1)}
    cons = Constraints(distances=held, frozen={0, 1, 2}, floors={(2, 3): 3.1}, dg_floors={(2, 5): 2.0})
    before = _written(mol, cons, mech.Distance().dg_windows).bm
    after = _written(mol, cons, mech.Distance().dg_windows, mech.Distance().dg_relief).bm
    assert np.argwhere(after < before).tolist() == np.argwhere(after != before).tolist() == [[4, 2]]


def test_floor_relief_lowers_only_a_native_floor_no_window_owns():
    mol = Chem.AddHs(Chem.MolFromSmiles("CBr.[Cl-]"))
    native = _written(mol, Constraints()).bm[2, 1]
    relieved = {}
    for floor, explicit in ((2.0, {}), (4.0, {}), (1.0, {(1, 2): (3.9, 4.1)})):
        cons = Constraints(dg_floors={(1, 2): floor}, distances=explicit)
        relieved[floor] = _written(mol, cons, mech.Distance().dg_windows, mech.Floor().dg_relief).bm[2, 1]
    assert relieved == {2.0: 2.0, 4.0: native, 1.0: native}


def test_haptic_centroid_seats_in_its_face_then_holds_any_shape():
    """Allylamine's C0=C1 binds Cr4 through centroid X, at right angles to N3 and to a separate chloride Cl5."""
    mol, x = Chem.AddHs(Chem.MolFromSmiles("C=CCN.[Cr].Cl")), 14  # X follows the 14 real atoms
    held = {(0, 4): (2.1, 2.1), (1, 4): (2.1, 2.1), (3, 4): (2.1, 2.1), (4, 5): (2.3, 2.3), (4, x): (2.0, 2.0)}
    angles = {(x, 4, 5): (90.0, 95.0), (x, 4, 3): (90.0, 95.0)}
    cons = Constraints(haptic={x: (0, 1)}, metals={4}, distances=held, angles=angles)
    before = _written(materialise_phantoms(mol, cons.haptic), cons, mech.Distance().dg_windows).bm
    bm = _written(materialise_phantoms(mol, cons.haptic), cons, mech.Distance().dg_windows, mech.Haptic().dg_relief).bm
    assert bm[x, 0] < 0.5
    assert (bm[x, 4], bm[3, 0], bm[3, 1]) == (before[x, 4], before[3, 0], before[3, 1])
    assert bm[5, 0] == bm[5, 1] == pytest.approx(math.hypot(2.1, 2.3))
    face = placed("C.C.C.C.C", [(1, 0, 0), (0, 2, 0), (-1, 0, 0), (0, -1.5, 0), (0, 0, 0)])  # C4 is the dummy
    corners = face.GetConformer().GetPositions()[:4]
    lifted = [np.vstack([corners, corners.mean(axis=0) + np.array([0, 0, z])]) for z in (0.0, 0.1, 0.2)]
    energies = _energies(mech.Haptic(), Constraints(haptic={4: (0, 1, 2, 3)}), face, lifted)
    assert energies == pytest.approx([0.0, energies[1], 4 * energies[1]], abs=1e-6)
    assert energies[1] > 0.0


def test_coplanar_row_holds_a_1_4_bound_then_a_soft_torsion():
    mol, matrix = Chem.MolFromSmiles("CCCC"), np.triu(np.full((4, 4), 10.0), 1)
    for (a, b), window in {(0, 1): (1.9, 2.1), (1, 2): (1.39, 1.41), (2, 3): (1.39, 1.41), (1, 3): (2.4, 2.45)}.items():
        matrix[b, a], matrix[a, b] = window
    syn = CoplanarRow(kind="ligand_ez", atoms=(0, 1, 2, 3), anchor=0.0, cap=15.0)
    anti, cons = replace(syn, anchor=180.0), Constraints(angles={(0, 1, 2): (110.0, 120.0)}, coplanar=[syn])
    spans = {}
    for name, case in {
        "syn": cons,
        "anti": replace(cons, coplanar=[anti]),
        "anti among looser": replace(cons, coplanar=[replace(anti, cap=45.0), anti, replace(anti, cap=45.0)]),
        "out of plane": replace(cons, coplanar=[replace(syn, kind="donor_plane")]),
        "past a right angle": replace(cons, coplanar=[replace(anti, cap=91.0)]),
        "seeded": replace(cons, coplanar=[replace(syn, kind="donor_torsion", anchor=None)]),
        "a requested dihedral": replace(cons, dihedrals={(0, 1, 2, 3): (89.0, 91.0)}),
        "no angle": replace(cons, angles={}),
        "a requested span": replace(cons, distances={(0, 3): (3.0, 3.2)}),
    }.items():
        bm = _written(mol, case, mech.Coplanar().dg_post, matrix=matrix).bm
        spans[name] = (bm[3, 0], bm[0, 3])
    lo, hi = spans.pop("syn")
    s, tilt = math.sin(math.radians(120.0)), math.radians(15.0)  # i and the far atom at 120 degrees, 15 off-plane
    near, far = np.array([-1.05, 2.1 * s, 0.0]), np.array([2.1, 1.4 * s * math.cos(tilt), 1.4 * s * math.sin(tilt)])
    assert lo <= np.linalg.norm(near - far) <= hi < 10.0
    assert spans.pop("anti among looser") == spans["anti"]
    assert spans.pop("anti")[0] > lo
    assert spans.pop("out of plane")[1] > hi
    assert set(spans.values()) == {(0.0, 10.0)}
    row = CoplanarRow(kind="donor_plane", atoms=(4, 1, 2, 3), anchor=0.0, cap=15.0)
    free = [Constraints(coplanar=[replace(row, kind=kind)]) for kind in ("donor_plane", "ligand_ez")]
    grafted = [replace(cons, frozen={1, 2, 3, 4}) for cons in free]  # an out-of-plane row reads both dihedrals of C1
    assert [len(_terms(mech.Coplanar(), cons, AMIDE)) for cons in free + grafted] == [2, 1, 0, 0]


def test_triangle_angles_enclose_the_widest_side_range_angle():
    widest = max(mech.opposite_angle(a, 2.0, 1.0) for a in np.linspace(1.0, 3.0, 201))
    assert mech.triangle_angles((1.0, 3.0), (2.0, 2.0), (1.0, 1.0))[1] >= widest


def test_distance_terms_hold_all_windows_but_haptic_seed_radii():
    """H11 stands in for a haptic centroid over H5 and H6, whose seed radii shape only the bounds."""
    windows = {(0, 3): (2.0, 4.0), (0, 4): (2.0, 4.0), (1, 5): (2.0, 2.2), (5, 11): (1.2, 1.3), (6, 11): (1.2, 1.3)}
    cons = Constraints(distances=windows, fixed={(1, 2): (1.2, 1.3), (2, 3): (2.2, 2.2)}, haptic={11: (5, 6)})
    held = {(call[1], call[2]): call[3:5] for call in _terms(mech.Distance(), cons, AMIDE)}
    assert (held.keys(), held[(2, 3)]) == ({(0, 3), (0, 4), (1, 5), (1, 2), (2, 3)}, (2.2, 2.2))
    assert 1.2 < held[(1, 2)][0] < held[(1, 2)][1] < 1.3  # a ranged fix is an inset wall


def test_pull_biases_to_its_window_or_a_contact_to_its_middle():
    contacts = {(0, 3), (0, 4)}
    cons = Constraints(distances=dict.fromkeys(contacts, (2.0, 4.0)), pulls={(3, 0): (2.5, 2.5), (2, 1, 3): (120, 125)})
    cons = replace(cons, contacts=SoftKeys(distances=contacts))
    pulled = [(0, 3, 2.5, 2.5), (2, 1, 3, False), (0, 4, 3.0, 3.0)]
    assert [call[1:5] for call in _terms(mech.Pull(), cons, AMIDE)] == pulled


def test_trigonal_angle_repair_skips_each_owned_pair():
    free = len(_terms(mech.TrigonalAngle(), Constraints(), AMIDE))
    owners = Constraints(angles={(0, 1, 2): (110.0, 130.0)}, distances={(0, 3): (2.1, 2.5)}, floors={(2, 3): 2.0})
    owned = (owners, Constraints(frozen={0, 1, 2, 3}))  # each owns one, or all, of C1's three pairs
    assert [free - len(_terms(mech.TrigonalAngle(), owner, AMIDE)) for owner in owned] == [3, 3]


def test_sp2_carbon_holds_its_seed_improper_unless_owned():
    """Benzene's C0-H is bound to C6, a bondless metal stand-in; a carbanion bridging two metals owns its carbon."""
    planar = mech.Sp2Planar()
    owners = (  # a held improper, the whole carbon grafted, and the graft touching it: C1 and O2 held, C0 and N3 free
        Constraints(dihedrals={(0, 2, 3, 1): (170.0, 190.0)}),
        Constraints(frozen={0, 1, 2, 3}),
        Constraints(frozen={1, 2}),
    )
    assert max(len(_terms(planar, owner, AMIDE)) for owner in owners) < len(_terms(planar, Constraints(), AMIDE))
    assert not _terms(planar, Constraints(conjugation=False), AMIDE)
    benzene, bound = embedded("c1ccccc1.C"), Constraints(metals={6}, distances={(0, 6): (2.0, 2.2)})
    rows = [  # bound, its angle held, through a haptic centroid, as a contact
        bound,
        replace(bound, angles={(6, 0, 1): (115.0, 125.0)}),
        Constraints(metals={6}, haptic={17: (0, 1)}, distances={(6, 17): (2.0, 2.1)}),
        replace(bound, contacts=SoftKeys(distances={(0, 6)})),
    ]
    assert [0 not in {call[4] for call in _terms(planar, cons, benzene)} for cons in rows] == [False, True, True, False]
    held = Constraints(metals={5, 6}, distances={(0, 5): (2.0, 2.2), (0, 6): (2.0, 2.2)})
    assert {call[4] for call in _terms(planar, held, embedded("[cH-]1cccc1.C.C"))} == {1, 2, 3, 4}


def test_conjugation_cap_leaves_a_requested_or_grafted_quartet():
    free = len(_terms(mech.ConjugationCap(), Constraints(), AMIDE))
    owners = (Constraints(dihedrals={(2, 1, 3, 4): (170.0, 190.0)}), Constraints(frozen={1, 2, 3, 4}))
    assert [free - len(_terms(mech.ConjugationCap(), owner, AMIDE)) for owner in owners] == [1, 1]
    assert not _terms(mech.ConjugationCap(), Constraints(conjugation=False), AMIDE)


def test_umbrella_holds_its_side_unless_grafted_or_requested():
    mol = placed("C(F)(Cl)Br", [(0, 0, 0), (1, 1, 4.0), (1, -1, -4.0), (-1, 1, -4.0)])  # the implicit H carries too
    frames = [mol.GetConformer().GetPositions() * (sign, 1, 1) for sign in (1, -1)]
    sides = [Constraints(umbrellas={(*order, 0): Improper(kind="side")}) for order in itertools.permutations((1, 2, 3))]
    energies = np.array([_energies(mech.Umbrella(), side, mol, frames) for side in sides])
    np.testing.assert_allclose(energies, [[0.0, energies[0, 1]]] * 6, atol=1e-10)
    assert energies[0, 1] > 0.0
    hexane = embedded("CCCCCC")
    held = {}
    for key in itertools.permutations((0, 1, 2, 3)):
        sharing = {(4, key[1], key[2], 5): (10.0, 30.0)}  # a torsion sharing only the axis leaves the improper its own
        cons = Constraints(umbrellas={key: Improper(kind="plane", anchor=180.0)}, dihedrals=sharing)
        owners = ({}, {"frozen": set(key[:3])}, {"frozen": set(key)}, {"dihedrals": {(1, 0, 3, 2): (10.0, 30.0)}})
        held[key] = tuple(bool(_terms(mech.Umbrella(), replace(cons, **owner), hexane)) for owner in owners)
    assert held == dict.fromkeys(held, (True, True, False, False))
