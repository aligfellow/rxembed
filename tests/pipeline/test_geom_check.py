"""Test the TS-aware and metal-aware geometry gate."""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdDistGeom, rdForceFieldHelpers, rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_distance as mdist
from rxembed import metal_perceive
from rxembed.constraints import Constraints
from rxembed.metal_core import COORDINATION_METALS
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline import metrics
from rxembed.utils import conjugated_quartets
from tests.conftest import EXAMPLES_DIR

_DFT = (
    pytest.param("mn-h2", {"metal_charges": {0: 2, 1: 1}}, id="mn-h2"),
    pytest.param("ru-co", {"bond_orders": "xyz2mol"}, id="ru-co"),
)  # the shipped transition states the floors must accept unchanged
_ACID_ARENE = "OC(=O)CCCCc1ccccc1"  # flexible acid + arene: the fixture for both kwarg-driven checks


def _bare_sphere(symbols, bonds, coords):
    """Return ``(mol, positions)``: a metal + ligand skeleton exactly as the surrogate leaves it.

    No sanitisation and no implicit H, so a monatomic donor really is bond-less; on the real path the M-donor
    bonds have already been stripped, and that is what makes the metal gates readable at all.
    """
    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))
    for i, j in bonds:
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for a in mol.GetAtoms():
        a.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*p))
    mol.AddConformer(conf)
    return mol, mol.GetConformer().GetPositions()


def _reference_conformer(smiles, seed=1, optimize=True):
    """Return a Mol with one conformer from plain ETKDG (+ MMFF): a geometry rxembed had no hand in making."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


def _shift(mol, idx, delta):
    conf = mol.GetConformer(0)
    p = conf.GetAtomPosition(int(idx))
    conf.SetAtomPosition(int(idx), Point3D(p.x + delta[0], p.y + delta[1], p.z + delta[2]))
    return mol


def _kinds(report):
    return {v.kind for v in report.violations}


# --- a clean embed passes ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        _ACID_ARENE,  # close polar contacts that must not read as clashes
    ],
    ids=["ethanol", "acid-arene"],
)
def test_clean_conformer_passes_geometry_check(smiles):
    rep = geom.check(_reference_conformer(smiles), 0)
    assert rep.ok(), rep.summary()


def test_ensemble_checks_every_tracked_conformer():
    ens = rx.embed("CCO", n=2, seed=1).minimize()
    reports = ens.check()
    assert set(reports) == set(ens.ids)
    assert all(report.ok() for report in reports.values())


def test_geom_check_prints_the_shape_record():
    """`check()`'s report carries the acceptance gate's shape record, explicit "ungated" before it ever ran."""
    iso = next(
        iter(rx.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("[Fe](N)(O)(F)(Cl)Br")), "trigonal_bipyramidal"))
    )
    confs = rx.core.embed(iso, n=1, seed=1)

    seed_report = geom.check(confs.mol, confs.ids[0], donors=iso.donors)
    assert seed_report.shape == "no shape record (ungated)"

    confs.minimize()
    report = geom.check(confs.mol, confs.ids[0], donors=iso.donors)
    assert report.shape is not None
    assert report.shape.startswith("Fe0 TBP")


def test_geometry_check_rejects_nonfinite_coordinates():
    mol = _reference_conformer("CCO")
    mol.GetConformer().SetAtomPosition(0, (float("nan"), 0.0, 0.0))

    report = geom.check(mol, 0)

    assert _kinds(report) == {"coordinates"}
    assert "non-finite coordinate" in report.summary()


def test_clashes_checks_geminal_hydrogens_but_not_h2_or_metal_hydrides():
    methane = _reference_conformer("C", optimize=False)
    hydrogens = [atom.GetIdx() for atom in methane.GetAtoms() if atom.GetAtomicNum() == 1]
    methane.GetConformer().SetAtomPosition(hydrogens[1], methane.GetConformer().GetAtomPosition(hydrogens[0]))
    assert any(v.detail == "H...H clash" for v in geom.clashes(methane, methane.GetConformer().GetPositions()))
    assert not any(
        v.detail == "H...H clash" for v in geom.clashes(methane, methane.GetConformer().GetPositions(), exclude={0})
    )

    for symbols, bonds in ((["H", "H"], [(0, 1)]), (["Fe", "H", "H"], [(0, 1), (0, 2)])):
        mol, pos = _bare_sphere(symbols, bonds, [(0, 0, 0), (1, 0, 0), (1.1, 0, 0)][: len(symbols)])
        assert not any(v.detail == "H...H clash" for v in geom.clashes(mol, pos))


def test_planarity_ignores_a_fastfindrings_perimeter_cycle():
    mol = _reference_conformer("c1ccc2ccccc2c1")
    rings = [set(ring) for ring in Chem.GetSymmSSSR(mol)]
    shared = sorted(rings[0] & rings[1])
    moving = sorted(rings[1] - set(shared))
    pos = mol.GetConformer().GetPositions()
    origin = pos[shared[0]]
    axis = pos[shared[1]] - origin
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    pos[moving] = origin + (pos[moving] - origin) @ rotation.T
    Chem.FastFindRings(mol)
    assert any(len(ring) == 10 for ring in mol.GetRingInfo().AtomRings()), "fixture premise"

    violations = geom.planarity(mol, pos)

    assert not any(violation.detail == "aromatic ring puckered" for violation in violations)


def test_planarity_checks_local_fused_rings_not_their_super_ring():
    mol = Chem.MolFromSmiles("c1c2c3cc3c12")
    pos = np.array(
        [[-0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 1.5, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    )
    Chem.GetSymmSSSR(mol)
    perimeter = next(ring for ring in mol.GetRingInfo().AtomRings() if len(ring) == 4)
    points = pos[list(perimeter)]
    rms = float(np.sqrt(np.linalg.svd(points - points.mean(0))[1][2] ** 2 / len(perimeter)))
    assert rms > 0.1, "the super-ring must be puckered enough to expose an unfiltered global RMS gate"

    violations = geom.planarity(mol, pos)

    assert not any(violation.detail == "aromatic ring puckered" for violation in violations)


def test_eta2_planarity_flex_is_metal_local():
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CC=C"))  # butadiene, no metal -> no eta2 flex
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    ci = next(a.GetIdx() for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    pos = conf.GetPositions()
    pos[ci] = pos[ci] + [0.0, 0.0, 0.35]  # shove one sp2 carbon 0.35 A out of plane (past the 0.15 default)
    conf.SetPositions(pos)

    assert any(v.kind == "planarity" for v in geom.planarity(mol, pos)), "non-metal sp2 wrongly flexed"


def test_declared_side_on_pair_keeps_its_window_at_an_early_metal_distance():
    """A declared eta2 N=N at 2.7 A from Ti is side-on; the 2.6 A cap guards only a perceived sphere."""
    mol = _reference_conformer("C=C/N=N/C")
    a, c, x, s = next(conjugated_quartets(mol))  # C=C-N=N: its twist is the side-on window's question
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), a, c, x, s, 135.0)  # 45 deg off plane: past 30, within 60
    rw = Chem.RWMol(mol)
    ti = rw.AddAtom(Chem.Atom("Ti"))
    pos = rw.GetConformer().GetPositions()
    normal = np.cross(pos[s] - pos[x], pos[c] - pos[x])
    height = (2.7**2 - (np.linalg.norm(pos[s] - pos[x]) / 2) ** 2) ** 0.5
    rw.GetConformer().SetAtomPosition(ti, ((pos[x] + pos[s]) / 2 + height * normal / np.linalg.norm(normal)).tolist())

    assert "conjugation" not in _kinds(geom.check(rw.GetMol(), 0, donors=[x, s]))
    assert "conjugation" in _kinds(geom.check(rw.GetMol(), 0)), "a perceived pair beyond the cap is not side-on"


def test_xh_bond_length_window_is_element_aware():
    mol = Chem.AddHs(Chem.MolFromSmiles("CP"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetSymbol() == "P")
    p = mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
    pos = conf.GetPositions()
    unit = (pos[h] - pos[p]) / np.linalg.norm(pos[h] - pos[p])
    for length, flagged in ((1.42, False), (1.9, True)):
        pos[h] = pos[p] + unit * length
        conf.SetPositions(pos)
        assert bool(any(v.kind == "hydrogen" for v in geom.hydrogens(mol, pos))) is flagged, f"P-H {length}"


# --- one deliberate break per violation kind --------------------------------------------------------------


def _planarity():
    mol = _reference_conformer("CC(=O)Nc1ccccc1")
    return _shift(mol, mol.GetRingInfo().AtomRings()[0][0], (0, 0, 0.8)), {}


def _conjugation():
    mol = _reference_conformer("CC(=O)NC")
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), a, b, c, d, 90.0)
    return mol, {}


def _bond_length():
    return _shift(_reference_conformer("CCO"), 0, (2.0, 0, 0)), {}


def _clash():
    mol = _reference_conformer("CCCCCCCC")
    mol.GetConformer(0).SetAtomPosition(7, mol.GetConformer(0).GetAtomPosition(0))
    return mol, {}


def _hydrogen():
    mol = _reference_conformer("CO")
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    return _shift(mol, h, (1.5, 0, 0)), {}


def _frozen_core():
    ref = _reference_conformer(_ACID_ARENE)
    return _shift(Chem.Mol(ref), 0, (0.5, 0, 0)), {"frozen": list(range(6)), "reference": ref}


def _constraint():
    return _reference_conformer(_ACID_ARENE), {"constraints": {"distances": {(1, 9): (0.5, 0.6)}}}


_BREAKS = {
    "planarity": _planarity,
    "conjugation": _conjugation,
    "bond_length": _bond_length,
    "clash": _clash,
    "hydrogen": _hydrogen,
    "frozen_core": _frozen_core,
    "constraint": _constraint,
}


@pytest.mark.parametrize("kind", list(_BREAKS), ids=list(_BREAKS))
def test_each_violation_kind_fires(kind):
    mol, kw = _BREAKS[kind]()
    assert kind in _kinds(geom.check(mol, 0, **kw))


def _alanine_and_mirror():
    mol = _reference_conformer("C[C@@H](C(=O)O)N")
    mirror = Chem.Mol(mol)
    mirror.GetConformer().SetPositions(-mirror.GetConformer().GetPositions())
    return mol, mirror


def test_xyz_path_reference_still_checks_stereo(tmp_path):
    mol, mirror = _alanine_and_mirror()
    path = str(tmp_path / "alanine.xyz")
    Chem.MolToXYZFile(mol, path)

    assert "stereo" in _kinds(geom.check(mirror, 0, reference=path))


def test_stereo_check_reads_the_conformer_it_is_asked_for():
    mol, mirror = _alanine_and_mirror()
    both = Chem.Mol(mol)
    inverted = Chem.Conformer(mirror.GetConformer())
    inverted.SetId(5)
    both.AddConformer(inverted)

    assert "stereo" not in _kinds(geom.check(both, reference=mol)), "the default id is the first conformer"
    assert "stereo" in _kinds(geom.check(both, 5, reference=mol))


def test_kwarg_checks_accept_matching_geometry():
    ref = _reference_conformer(_ACID_ARENE)
    d = float(np.linalg.norm(ref.GetConformer(0).GetPositions()[1] - ref.GetConformer(0).GetPositions()[9]))
    assert geom.check(ref, 0, frozen=list(range(6)), reference=ref).ok(), "an identical geometry moved the core"
    assert geom.check(ref, 0, constraints={"distances": {(1, 9): (d - 0.1, d + 0.1)}}).ok()


def test_constraint_gate_checks_periodic_dihedrals():
    mol = _reference_conformer("CCCC")
    pos = mol.GetConformer().GetPositions()
    atoms = (0, 1, 2, 3)
    phi = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *atoms)

    missed = geom.check_constraints(mol, pos, {"dihedrals": {atoms: (phi + 50.0, phi + 70.0)}})
    equivalent = geom.check_constraints(mol, pos, {"dihedrals": {atoms: (phi + 350.0, phi + 370.0)}})

    assert [violation.atoms for violation in missed] == [atoms]
    assert not equivalent


@pytest.mark.parametrize(
    "spec",
    [
        {"distances": {(0, -1): (1.0, 2.0)}},
        {"distances": {(0, 999): (1.0, 2.0)}, "haptic": {999: [0, 999]}},
    ],
    ids=["negative-atom", "invalid-haptic-face"],
)
def test_constraint_gate_rejects_an_unmeasurable_virtual_term(spec):
    mol = _reference_conformer("CC")
    pos = mol.GetConformer().GetPositions()

    violations = geom.check_constraints(mol, pos, spec)

    assert len(violations) == 1
    assert np.isnan(violations[0].value)
    assert "could not be measured" in violations[0].detail


def test_constraint_gate_ignores_a_transient_phantom_term():
    mol = _reference_conformer("CCC")
    pos = mol.GetConformer().GetPositions()
    spec = rx.Constraints(
        distances={(0, 99): (20.0, 21.0)},
        haptic={99: (1, 2)},
        phantoms=frozenset({99}),
    )

    assert not geom.check_constraints(mol, pos, spec)


# --- TS-awareness: a held core is not judged by ground-state rules ----------------------------------------


def test_frozen_core_skips_ground_state_checks():
    mol, _kw = _conjugation()
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    assert "conjugation" in _kinds(geom.check(mol, 0))
    assert "conjugation" not in _kinds(geom.check(mol, 0, frozen=(a, b, c, d)))


# --- the 1-3 fusion gate, and the strained rings it must not eat -------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_gate_catches_collapsed_ester_fusion():
    mol = _reference_conformer("CC(=O)OC")
    cc = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and sum(nb.GetAtomicNum() == 8 for nb in a.GetNeighbors()) == 2
    )
    onb = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cc).GetNeighbors() if nb.GetAtomicNum() == 8]
    o_term = next(o for o in onb if mol.GetAtomWithIdx(o).GetDegree() == 1)  # swing the terminal =O, not the methyl
    o_est = next(o for o in onb if o != o_term)
    rdMolTransforms.SetAngleDeg(mol.GetConformer(0), o_est, cc, o_term, 56.0)
    pos = mol.GetConformer(0).GetPositions()
    assert float(np.linalg.norm(pos[o_term] - pos[o_est])) < 1.4, "the two oxygens must have fused to set the test"

    fused = [v for v in geom.check(mol, 0).violations if v.kind == "fusion"]
    assert fused
    assert set(fused[0].atoms) >= {o_term, o_est}
    # red-first: the three gates that were the only ones looking here all stay silent on that pair
    assert not [v for v in geom.clashes(mol, pos) if set(v.atoms) >= {o_term, o_est}], "clashes excludes a 1-3 pair"
    assert metrics.bonding_failure(mol, 0) is None, "bonding_failure's fusion floor sits below the fused O...O distance"
    formed, _broken = metrics.connectivity(mol, 0)
    assert {o_term, o_est} not in [set(p) for p in formed], "connectivity skips a topo-2 pair"


@pytest.mark.parametrize("smiles", ["C1CO1", "CC(=O)OC", "C[N+](=O)[O-]"], ids=["epoxide", "ester", "nitro"])
def test_tight_1_3_pair_is_not_fusion(smiles):
    rep = geom.check(_reference_conformer(smiles), 0)
    assert "fusion" not in _kinds(rep)
    assert rep.ok(), rep.summary()


# --- the metal arm: the over-bond ruler the gate reads ------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_overbond_gate_accepts_clean_isomers():
    seen = 0
    for iso in rx.metal("Cl[Pd](Cl)(N)N", "square_planar"):
        ens = rx.embed(iso, n=2, seed=1).minimize()
        for cid in ens.ids:
            seen += 1
            assert not metal_perceive.metal_overbond(ens.mol, ens.mol.GetConformer(cid).GetPositions(), iso.donors)
    assert seen, "no conformer was judged: the gate was never asked anything"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_donor_sets_are_per_metal():
    reference = rx.read_xyz(str(EXAMPLES_DIR / "mn-h2.xyz"), metal_charges={0: 2, 1: 1})
    isos = rx.metal(reference, "octahedral", center="Mn", fix=[1, 5, 63, 64, 65, 66])
    iso = next(candidate for candidate in isos if rx.cxsmiles(candidate) == rx.cxsmiles(reference))
    ens = rx.embed(iso, n=1, seed=1)
    assert ens.ids, "no conformer was judged: the gate was never asked anything"
    for cid in ens.ids:
        assert not metal_perceive.metal_overbond(ens.mol, ens.mol.GetConformer(cid).GetPositions(), iso.donors)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_gate_catches_third_sphere_overbond():
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    mol, m, donors = ens.mol, iso.metal, set(iso.donors)
    topo = Chem.GetDistanceMatrix(mol)
    hops = {i: 1 + min(topo[d][i] for d in donors) for i in range(mol.GetNumAtoms())}
    c = next(i for i, h in hops.items() if h >= 3 and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

    clean = mol.GetConformer(ens.ids[0]).GetPositions().copy()
    assert not metal_perceive.metal_overbond(mol, clean, donors)

    def crushed_to(distance):  # the clean M...c separation is embed-dependent, so aim at an absolute distance
        pos = clean.copy()
        u = pos[c] - pos[m]
        pos[c] = pos[m] + u / float(np.linalg.norm(u)) * distance
        return pos

    buried = crushed_to(1.2)
    v = metal_perceive.metal_overbond(mol, buried, donors)
    assert [x.kind for x in v] == ["metal_overbond"]
    assert c in v[0].atoms

    # perceiving the donors instead of declaring them is circular: the collapse itself makes the atom "a donor",
    # so it is judged by the donor floor (may it be this close?) not the non-donor one (may it be here at all?).
    near = crushed_to(1.75)  # a plausible BONDING length; isolates the circularity from the donor floor
    assert metal_perceive.metal_overbond(mol, near, donors), "the declared-donor gate must still see this collapse"
    assert not metal_perceive.metal_overbond(mol, near, None), "perceived-donor mode is expected to be circular"
    # ...but the circularity is bounded: past the donor floor even a perceived donor is judged. Wrong atom, not silence.
    assert [x.kind for x in metal_perceive.metal_overbond(mol, buried, None)] == ["metal_collapse"]


def test_overbond_tier_counts_bonded_donors():
    ac, pos = _bare_sphere(  # Pd | O O (donors) | C carboxyl | C methyl: the CMD/AMLA motif
        ["Pd", "O", "O", "C", "C"],
        [(1, 3), (2, 3), (3, 4)],
        [(0, 0, 0), (1.12, 1.68, 0), (-1.12, 1.68, 0), (0, 2.46, 0), (0, 3.96, 0)],
    )
    assert np.linalg.norm(pos[3] - pos[0]) == pytest.approx(2.460, abs=0.005)
    assert not metal_perceive.metal_overbond(ac, pos, [1, 2])

    ti, tpos = _bare_sphere(["Ti", "C", "C"], [(1, 2)], [(0, 0, 0), (2.15, 0, 0), (1.60, 1.99, 0)])
    assert np.linalg.norm(tpos[2] - tpos[0]) == pytest.approx(2.554, abs=0.01)
    assert not metal_perceive.metal_overbond(ti, tpos, [1])

    assert mdist.overbond_tier(ac, [1, 2], 3) == mdist.APEX  # bonded to both donors: a chelate bite, forced
    assert mdist.overbond_tier(ac, [1, 2], 4) == mdist.OUTER  # bonded to neither: third sphere
    assert mdist.overbond_tier(ac, [1], 3) == mdist.NEAR  # bonded to one: second sphere, floored


def test_second_sphere_floor_rejects_collapse_not_agostic():
    col, pos = _bare_sphere(
        ["Pd", "N", "C", "C", "Cl", "Cl"],
        [(1, 2), (2, 3)],
        [(0, 0, 0), (2.101, 0, 0), (1.502, 1.577, 0), (2.9, 2.6, 0), (0, 2.385, 0), (0, -2.385, 0)],
    )
    cons = Constraints()
    mdist.nondonor_floors(col, 0, 46, [1, 4, 5], cons)
    assert np.linalg.norm(pos[2] - pos[0]) < cons.floors[(0, 2)], "the alpha-C is floored, not exempt"
    assert metal_perceive.metal_overbond(col, pos, [1, 4, 5])

    ti, pos = _bare_sphere(
        ["Ti", "C", "C", "Cl", "Cl", "Cl"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (2.038, 1.539, 0), (-2.2, 0, 0), (0, -2.2, 0), (0, 0, 2.2)],
    )
    cons = Constraints()
    mdist.nondonor_floors(ti, 0, 22, [1, 3, 4, 5], cons)
    assert np.linalg.norm(pos[2] - pos[0]) > cons.floors[(0, 2)]
    assert not metal_perceive.metal_overbond(ti, pos, [1, 3, 4, 5])


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(("name", "read_kw"), _DFT)
def test_floors_accept_reference_geometries(name, read_kw):
    pt = GetPeriodicTable()
    mol = rx.read_xyz(str(EXAMPLES_DIR / f"{name}.xyz"), 0, **read_kw)
    pos = mol.GetConformer().GetPositions()
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]
    assert metals, f"{name} carries no transition metal: the fixture exercises no floor at all"

    for m in metals:
        donors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors())
        cons = Constraints()
        mdist.nondonor_floors(mol, m, mol.GetAtomWithIdx(m).GetAtomicNum(), donors, cons)
        assert cons.floors, f"metal {m} got no floors at all"
        for (i, j), floor in cons.floors.items():
            x = i if j == m else j
            assert float(np.linalg.norm(pos[x] - pos[m])) >= floor, (
                f"floor rejects {mol.GetAtomWithIdx(x).GetSymbol()}{x}"
            )
        rm = pt.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
        for d in donors:
            rs = rm + pt.GetRcovalent(mol.GetAtomWithIdx(d).GetAtomicNum())
            assert float(np.linalg.norm(pos[d] - pos[m])) / rs > mdist.DONOR_COLLAPSE_RATIO
        assert not metal_perceive.metal_overbond(mol, pos, donors), "the gate rejects a real DFT geometry"


# --- a declared donor is a donor, whatever its element ------------------------------------------------------


def _ruthenium(d_ruh=1.701, d_rucl=2.233):
    """DUKPII: a terminal hydride and a terminal chloride on one Ru, both bond-less after the surrogate's strip."""
    return _bare_sphere(
        ["Ru", "H", "Cl", "P", "P"],
        [],
        [(0, 0, 0), (d_ruh, 0, 0), (-d_rucl, 0, 0), (0, 2.341, 0), (0, -2.341, 0)],
    )


def test_declared_hydride_passes_without_covalent_ruler():
    mol, pos = _ruthenium()
    donors = [1, 2, 3, 4]
    assert 1 in metal_perceive.spheres(mol, pos, donors)[0]
    assert not geom.hydrogens(mol, pos, donors=frozenset(donors))
    assert geom.check(mol, mol.GetConformer().GetId(), donors=donors).ok()


def test_undeclared_agostic_h_is_not_promoted_to_a_donor():
    mol, pos = _bare_sphere(
        ["Ru", "C", "H", "P", "P"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (1.85, 0, 0.9), (0, 2.341, 0), (0, -2.341, 0)],
    )
    donors = [1, 3, 4]
    assert 2 not in metal_perceive.spheres(mol, pos, donors)[0]
    assert not metal_perceive.metal_overbond(mol, pos, donors)


@pytest.mark.parametrize("kind", ["hydride", "chloride"])
def test_buried_donor_triggers_metal_collapse(kind):
    at = 1 if kind == "hydride" else 2
    mol, pos = _ruthenium(**{"d_ruh" if kind == "hydride" else "d_rucl": 0.100})
    v = metal_perceive.metal_overbond(mol, pos, [1, 2, 3, 4])
    assert [x.kind for x in v] == ["metal_collapse"]
    assert v[0].atoms == (0, at)
    assert not geom.check(mol, mol.GetConformer().GetId(), donors=[1, 2, 3, 4]).ok()


# --- the two FF caps, seen through the gate on real complexes ------------------------------------------------

_SCHREINER = "c1ccccc1NC(=S)Nc1ccccc1.CC(C)=O"  # diphenylthiourea + acetone: a Schreiner-type H-bond donor
_CHB = "C1CSC2=NCCN12.CC(=O)OC(C)=O"  # tetramisole's bicyclic thiazoline/imidazolidine core + acetic anhydride


def _seeded(smi, seed=1, pick=None):
    """Embed an NCI complex on its first (or `pick`-matched) discovered mode -> the ensembles to check."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if pick is None:
        contact = next(iter(rx.nci_modes(mol).values()))
    else:
        cands = rx.nci_candidates(mol)
        contact = cands[next(k for k in cands if k.startswith(pick))]
    res = rx.embed(smi, seed=seed, n=4, contacts=contact)
    return list(res) if isinstance(res, rx.EnsembleSet) else [res]


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_conjugation_cap_holds_thiourea_plane():
    seen = 0
    for ens in _seeded(_SCHREINER):
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            seen += 1
            assert "conjugation" not in _kinds(geom.check(ens.mol, int(cid), frozen=frozen))
    assert seen, "no conformer was produced: the gate assertion never ran"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[workflow]")
def test_chb_caps_keep_sp2_carbons_planar():
    seen = 0
    for ens in _seeded(_CHB, pick="ChB"):
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            seen += 1
            assert "planarity" not in _kinds(geom.check(ens.mol, int(cid), frozen=frozen))
    assert seen, "no conformer was produced: the gate assertion never ran"
