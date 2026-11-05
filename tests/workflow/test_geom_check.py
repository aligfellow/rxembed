"""`workflow/geom_check.py`: the defects one conformer shows, which of them fail it, and what the request owns."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed.geometry.constraints import Constraints, SoftKeys
from rxembed.workflow import geom_check as geom
from tests.geometry_fixtures import embedded, mirrored, placed
from tests.metal_fixtures import bound_donors

ETHANOL = embedded("CCO")
BUTANE = embedded("CCCC")
# C0 of the methylamine folded 80 deg off the Pd2-N1 axis, under its donor_orientation floor; chlorides in xz
_FOLD = np.radians(80.0)
FOLDED_AMINE_PD = placed(
    rx.parse_smiles("C[NH2]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"),
    [
        (2.05 - 1.47 * np.cos(_FOLD), 1.47 * np.sin(_FOLD), 0),
        (2.05, 0, 0),
        (0, 0, 0),
        (-2.3, 0, 0),
        (0, 0, 2.3),
        (0, 0, -2.3),
    ],
)


def _kinds(report):
    return {v.kind for v in report.violations}


def _twisted_amide():
    """Return N-methylacetamide with its O=C-N-C dihedral turned to 90 deg."""
    amide = embedded("CC(=O)NC")
    o, c, n, methyl = amide.GetSubstructMatch(Chem.MolFromSmarts("O=C-N-C"))
    rdMolTransforms.SetDihedralDeg(amide.GetConformer(), o, c, n, methyl, 90.0)
    return amide


def test_check_reports_each_defect_on_its_minimal_break():
    """Each row breaks one physical rule, and the report names that rule alone."""
    nan, core_moved = Chem.Mol(ETHANOL), Chem.Mol(ETHANOL)
    nan.GetConformer().SetAtomPosition(0, (float("nan"), 0.0, 0.0))
    core_moved.GetConformer().SetAtomPosition(0, tuple(ETHANOL.GetConformer().GetPositions()[0] + (0.5, 0, 0)))
    hexagon = [(1.39 * np.cos(k * np.pi / 3), 1.39 * np.sin(k * np.pi / 3), 0.5 * (k == 0)) for k in range(6)]
    square_pd = [(2.3, 0, 0), (0, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, -2.3, 0)]  # Cl0 Pd1 Cl2 Cl3 Cl4
    bromochlorofluoromethane = embedded("F[C@H](Cl)Br")
    rows = {
        "non-finite ethanol": (nan, {}, {"coordinates"}),
        "ethane-stretched": (placed("CC", [(0, 0, 0), (2.5, 0, 0)]), {}, {"bond_length"}),
        "methanes overlapping": (placed("C.C", [(0, 0, 0), (1.0, 0, 0)]), {}, {"clash"}),
        "propane folded": (placed("CCC", [(1.5, 0, 0), (0, 0, 0), (0.86, 1.23, 0)]), {}, {"fusion"}),
        "isobutylene lifted": (
            placed("C=C(C)C", [(1.4, 0, 0), (0, 0, 0.3), (-0.7, 1.21, 0), (-0.7, -1.21, 0)]),
            {},
            {"planarity"},
        ),
        "benzene-puckered": (placed("c1ccccc1", hexagon), {}, {"planarity"}),
        "amide twisted": (_twisted_amide(), {}, {"conjugation"}),
        "Cl on PdCl4": (
            placed(rx.parse_smiles("[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-].[Cl-]"), [*square_pd, (0, 0, 1.8)]),
            {},
            {"metal_overbond"},
        ),
        "Ru hydride declared": (
            placed("[Ru].[H].[Cl].[P].[P]", [(0, 0, 0), (0.1, 0, 0), (-2.233, 0, 0), (0, 2.341, 0), (0, -2.341, 0)]),
            {"donors": [1, 2, 3, 4]},
            {"metal_collapse"},
        ),
        "amine folded on Pd": (FOLDED_AMINE_PD, {}, {"donor_orientation"}),
        "core moved": (
            core_moved,
            {"reference": ETHANOL, "constraints": Constraints(frozen=frozenset(range(9)))},
            {"frozen_core"},
        ),
        "window missed": (ETHANOL, {"constraints": Constraints(distances={(0, 2): (0.5, 0.6)})}, {"constraint"}),
        "mirrored-bromochlorofluoromethane": (
            mirrored(bromochlorofluoromethane),
            {"reference": bromochlorofluoromethane},
            {"stereo"},
        ),
    }
    found = {name: _kinds(geom.check(mol, 0, **options)) for name, (mol, options, _) in rows.items()}
    assert found == {name: kinds for name, (_mol, _options, kinds) in rows.items()}


def test_check_passes_a_sound_conformer():
    """A bonded pair is no clash, a small ring's 1-3 pair no fusion, and a core on its own reference has not moved."""
    rows = {
        "ethanol": (ETHANOL, {}),
        "dihydrogen": (placed("[H][H]", [(0, 0, 0), (0.74, 0, 0)]), {}),
        "cyclopropane": (placed("C1CC1", [(0, 0, 0), (1.51, 0, 0), (0.755, 1.308, 0)]), {}),
        "own core": (ETHANOL, {"reference": ETHANOL, "constraints": Constraints(frozen=frozenset(range(9)))}),
    }
    found = {name: geom.check(mol, 0, **options).violations for name, (mol, options) in rows.items()}
    assert found == {name: [] for name in rows}


def test_check_leaves_a_held_terms_violation_to_the_request():
    """Two pairs of methanes overlapping: a fix, a contact or a frozen core owns its pair, the other is judged."""
    methanes = placed("C.C.C.C", [(0, 0, 0), (1.0, 0, 0), (10, 0, 0), (11, 0, 0)])
    rows = {
        "unheld": (Constraints(), {(0, 1), (2, 3)}),
        "fixed": (Constraints(fixed={(0, 1): (1.0, 1.0)}), {(2, 3)}),
        "contact": (
            Constraints(distances={(0, 1): (0.9, 1.1)}, contacts=SoftKeys(distances=frozenset({(0, 1)}))),
            {(2, 3)},
        ),
        "frozen": (Constraints(frozen=frozenset({0, 1})), {(2, 3)}),
    }
    for name, (constraints, judged) in rows.items():
        report = geom.check(methanes, 0, constraints=constraints, check_windows=False)
        assert {v.atoms for v in report.violations} == judged, name


def test_check_fails_conjugation_only_where_constraints_hold_it():
    """Donor orientation and released conjugation inform without failing; held conjugation fails the conformer."""
    rows = {
        "conjugation held": (_twisted_amide(), Constraints(), "conjugation", False),
        "conjugation released": (_twisted_amide(), Constraints(conjugation=False), "conjugation", True),
        "folded-amine": (FOLDED_AMINE_PD, Constraints(), "donor_orientation", True),
    }
    for name, (mol, constraints, kind, ok) in rows.items():
        report = geom.check(mol, 0, constraints=constraints)
        assert (_kinds(report), report.ok) == ({kind}, ok), name


def test_check_constraints_reports_each_window_a_geometry_misses():
    """A dihedral window is periodic, a scalar is a point window, and a term on a haptic centroid is not measured."""
    pos = BUTANE.GetConformer().GetPositions()
    phi = rdMolTransforms.GetDihedralDeg(BUTANE.GetConformer(), 0, 1, 2, 3)
    d = float(np.linalg.norm(pos[0] - pos[3]))
    rows = {
        "dihedral-missed": (Constraints(dihedrals={(0, 1, 2, 3): (phi + 50.0, phi + 70.0)}), [(0, 1, 2, 3)]),
        "dihedral a turn on": (Constraints(dihedrals={(0, 1, 2, 3): (phi + 350.0, phi + 370.0)}), []),
        "scalar distance": (Constraints(distances={(0, 3): d}), []),
        "centroid-term": (Constraints(distances={(0, 3): (0.5, 0.6)}, haptic={3: (1, 2)}), []),
    }
    found = {name: [v.atoms for v in geom.check_constraints(BUTANE, pos, cons)] for name, (cons, _) in rows.items()}
    assert found == {name: missed for name, (_cons, missed) in rows.items()}


def test_check_constraints_refuses_a_term_of_five_atoms():
    pos = BUTANE.GetConformer().GetPositions()
    with pytest.raises(ValueError, match="needs 2, 3 or 4 atoms"):
        geom.check_constraints(BUTANE, pos, Constraints(distances={(0, 1, 2, 3, 4): (1.0, 2.0)}))


def test_check_refuses_constraints_that_are_not_the_record():
    with pytest.raises(TypeError, match=r"rx\.Constraints record"):
        geom.check(ETHANOL, 0, constraints={"distances": {(0, 1): (1.0, 2.0)}})


@pytest.fixture(scope="module")
def tipped():
    """Return, by ligand, its metal-bound carbon's species embedded with one C-H tipped out of that carbon's plane."""
    out = {}
    for name, smiles in (
        ("phenyl", "c1cc[cH](->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cc1"),
        ("eta1-cyclopentadienyl", "[cH-]1(->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cccc1"),
        ("eta2-ethylene", "[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH2]->1"),
    ):
        ens = rx.embed(rx.isomers(smiles, geometry="square_planar", stereo="free")[0], n=1, seed=42, threads=1)
        mol = Chem.Mol(ens.mol, confId=ens.ids[0])
        donor = next(d for d in bound_donors(mol) if mol.GetAtomWithIdx(d).GetSymbol() == "C")
        frame = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors() if n.GetSymbol() not in ("Pd", "Pt")]
        hydrogen = next(i for i in frame if mol.GetAtomWithIdx(i).GetAtomicNum() == 1)
        a, b = (i for i in frame if i != hydrogen)
        pos = mol.GetConformer().GetPositions()
        normal = np.cross(pos[a] - pos[donor], pos[b] - pos[donor])
        bond = pos[hydrogen] - pos[donor]
        tip = bond + np.linalg.norm(bond) * normal / np.linalg.norm(normal)
        pos[hydrogen] = pos[donor] + tip * np.linalg.norm(bond) / np.linalg.norm(tip)
        mol.GetConformer().SetPositions(pos)
        out[name] = (mol, donor, (*frame, donor))
    return out


def test_bound_carbons_use_the_aryl_rule_unless_oriented(tipped):
    """A sigma-aryl carbon is judged; a haptic or carbanion carbon pyramidalises and is spared.

    A held M-C-H angle or framework improper owns its geometry; a releasable or unrestricted angle orients nothing.
    """
    mol, donor, improper = tipped["phenyl"]
    (metal,) = (a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pd")
    angle_key = (metal, donor, next(i for i in improper if mol.GetAtomWithIdx(i).GetAtomicNum() == 1))
    angle = rdMolTransforms.GetAngleDeg(mol.GetConformer(), *angle_key)
    twist = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *improper)
    released = SoftKeys(angular=frozenset({angle_key[::-1]}))
    rows = {
        "phenyl": ("phenyl", Constraints(), True),
        "angle held": ("phenyl", Constraints(angles={angle_key: (angle, angle)}), False),
        "angle fixed": ("phenyl", Constraints(fixed={angle_key: (angle, angle)}, contacts=released), False),
        "improper held": ("phenyl", Constraints(dihedrals={improper: (twist, twist)}), False),
        "angle releasable": ("phenyl", Constraints(angles={angle_key: (angle, angle)}, contacts=released), True),
        "angle unrestricted": ("phenyl", Constraints(angles={angle_key: (0.0, 180.0)}), True),
        "eta1-cyclopentadienyl": ("eta1-cyclopentadienyl", Constraints(), False),
        "eta2-ethylene": ("eta2-ethylene", Constraints(), False),
    }
    for name, (ligand, constraints, judged) in rows.items():
        mol, donor, _ = tipped[ligand]
        report = geom.check(mol, 0, donors=bound_donors(mol), constraints=constraints, check_windows=False)
        assert any(v.kind == "planarity" and donor in v.atoms for v in report.violations) == judged, name


def test_check_widens_a_side_on_pairs_conjugation_window():
    """Ti 2.7 A from a twisted C=C-N=N: declared donors widen its window, a perceived pair that far does not."""
    mol = embedded("C=C/N=N/C")  # C0=C1-N2=N3: the twist is the side-on window's question
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3, 135.0)
    titanium = Chem.RWMol(mol)
    ti = titanium.AddAtom(Chem.Atom("Ti"))
    pos = titanium.GetConformer().GetPositions()
    normal = np.cross(pos[3] - pos[2], pos[1] - pos[2])
    height = (2.7**2 - (np.linalg.norm(pos[3] - pos[2]) / 2) ** 2) ** 0.5
    titanium.GetConformer().SetAtomPosition(ti, (pos[2] + pos[3]) / 2 + height * normal / np.linalg.norm(normal))
    donors = {"declared": [2, 3], "perceived": None}
    twisted = {name: "conjugation" in _kinds(geom.check(titanium.GetMol(), 0, donors=d)) for name, d in donors.items()}
    assert twisted == {"declared": False, "perceived": True}
