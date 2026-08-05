"""`pipeline/geom_check.py`, the TS-aware geometry gate: every violation kind, and what it must not flag.

The gate is the pipeline's; the rulers it reads for a metal (``metal_perceive.metal_overbond``, the
``metal_distance`` floors and tier boundaries) are the core's, and are pinned here because that is where they
are observable; building their subject needs ``rx.metal`` / xyz perception, i.e. the optional tier.
"""

from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolTransforms
from rdkit.Geometry import Point3D

from rxembed import metal_distance as mdist
from rxembed import metal_perceive as perceive
from rxembed.pipeline import geom_check as geom

_DFT = ("mn-h2", "ru-co")  # the shipped transition states the floors must accept unchanged
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
)
def test_a_clean_conformer_passes_the_whole_gate(smiles):
    rep = geom.check(_reference_conformer(smiles), 0)
    assert rep.ok(), rep.summary()


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
def test_each_violation_kind_fires_on_its_own_deliberate_break(kind):
    mol, kw = _BREAKS[kind]()
    assert kind in _kinds(geom.check(mol, 0, **kw))


def test_the_two_kwarg_driven_checks_are_silent_when_the_geometry_agrees_with_what_was_stated():
    ref = _reference_conformer(_ACID_ARENE)
    d = float(np.linalg.norm(ref.GetConformer(0).GetPositions()[1] - ref.GetConformer(0).GetPositions()[9]))
    assert geom.check(ref, 0, frozen=list(range(6)), reference=ref).ok(), "an identical geometry moved the core"
    assert geom.check(ref, 0, constraints={"distances": {(1, 9): (d - 0.1, d + 0.1)}}).ok()


# --- TS-awareness: a held core is not judged by ground-state rules ----------------------------------------


def test_a_frozen_core_is_exempt_from_the_ground_state_checks():
    mol, _kw = _conjugation()
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[O]=[C]-[N]-[C]"))
    assert "conjugation" in _kinds(geom.check(mol, 0))
    assert "conjugation" not in _kinds(geom.check(mol, 0, frozen=(a, b, c, d)))


# --- the 1-3 fusion gate, and the strained rings it must not eat -------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_collapsed_ester_o_c_o_fusion_is_caught_where_every_prior_gate_was_blind():
    from rxembed.pipeline import metrics

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
    assert metrics.bonding_ok(mol, 0), "bonding_ok's fusion floor sits below the fused O...O distance"
    formed, _broken = metrics.connectivity(mol, 0)
    assert {o_term, o_est} not in [set(p) for p in formed], "connectivity skips a topo-2 pair"


@pytest.mark.parametrize("smiles", ["C1CO1", "CC(=O)OC", "C[N+](=O)[O-]"])
def test_a_real_tight_1_3_pair_is_never_a_fusion(smiles):
    rep = geom.check(_reference_conformer(smiles), 0)
    assert "fusion" not in _kinds(rep)
    assert rep.ok(), rep.summary()


# --- the metal arm: the over-bond ruler the gate reads ------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_the_over_bond_gate_is_silent_on_every_isomer_of_a_clean_complex():
    import rxembed.pipeline as rx

    seen = 0
    for iso in rx.metal("Cl[Pd](Cl)(N)N", "square_planar"):
        ens = rx.embed(iso, n=2, seed=1).minimize()
        for cid in ens.ids:
            seen += 1
            assert not perceive.metal_overbond(ens.mol, ens.mol.GetConformer(cid).GetPositions(), iso.donors)
    assert seen, "no conformer was judged: the gate was never asked anything"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_donor_sets_are_per_metal_so_a_spectator_ferrocene_is_not_an_over_bond():
    import rxembed.pipeline as rx

    isos = rx.metal("examples/structures/mn-h2.xyz", "octahedral", center="Mn", fix=[1, 5, 63, 64, 65, 66])
    ens = rx.embed(isos[0], n=1, seed=1)
    assert ens.ids, "no conformer was judged: the gate was never asked anything"
    for cid in ens.ids:
        assert not perceive.metal_overbond(ens.mol, ens.mol.GetConformer(cid).GetPositions(), isos[0].donors)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_third_sphere_atom_crushed_onto_the_metal_is_an_over_bond():
    import rxembed.pipeline as rx

    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    mol, m, donors = ens.mol, iso.metal, set(iso.donors)
    topo = Chem.GetDistanceMatrix(mol)
    hops = {i: 1 + min(topo[d][i] for d in donors) for i in range(mol.GetNumAtoms())}
    c = next(i for i, h in hops.items() if h >= 3 and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

    clean = mol.GetConformer(ens.ids[0]).GetPositions().copy()
    assert not perceive.metal_overbond(mol, clean, donors)

    def crushed_to(distance):  # the clean M...c separation is embed-dependent, so aim at an absolute distance
        pos = clean.copy()
        u = pos[c] - pos[m]
        pos[c] = pos[m] + u / float(np.linalg.norm(u)) * distance
        return pos

    buried = crushed_to(1.2)
    v = perceive.metal_overbond(mol, buried, donors)
    assert [x.kind for x in v] == ["metal_overbond"]
    assert c in v[0].atoms

    # perceiving the donors instead of declaring them is circular: the collapse itself makes the atom "a donor",
    # so it is judged by the donor floor (may it be this close?) not the non-donor one (may it be here at all?).
    near = crushed_to(1.75)  # a plausible BONDING length; isolates the circularity from the donor floor
    assert perceive.metal_overbond(mol, near, donors), "the declared-donor gate must still see this collapse"
    assert not perceive.metal_overbond(mol, near, None), "perceived-donor mode is expected to be circular"
    # ...but the circularity is bounded: past the donor floor even a perceived donor is judged. Wrong atom, not silence.
    assert [x.kind for x in perceive.metal_overbond(mol, buried, None)] == ["metal_collapse"]


def test_the_over_bond_tier_is_decided_by_how_many_donors_the_atom_is_bonded_to():
    ac, pos = _bare_sphere(  # Pd | O O (donors) | C carboxyl | C methyl: the CMD/AMLA motif
        ["Pd", "O", "O", "C", "C"],
        [(1, 3), (2, 3), (3, 4)],
        [(0, 0, 0), (1.12, 1.68, 0), (-1.12, 1.68, 0), (0, 2.46, 0), (0, 3.96, 0)],
    )
    assert np.linalg.norm(pos[3] - pos[0]) == pytest.approx(2.460, abs=0.005)
    assert not perceive.metal_overbond(ac, pos, [1, 2])

    ti, tpos = _bare_sphere(["Ti", "C", "C"], [(1, 2)], [(0, 0, 0), (2.15, 0, 0), (1.60, 1.99, 0)])
    assert np.linalg.norm(tpos[2] - tpos[0]) == pytest.approx(2.554, abs=0.01)
    assert not perceive.metal_overbond(ti, tpos, [1])

    assert mdist.overbond_tier(ac, [1, 2], 3) == mdist.APEX  # bonded to both donors: a chelate bite, forced
    assert mdist.overbond_tier(ac, [1, 2], 4) == mdist.OUTER  # bonded to neither: third sphere
    assert mdist.overbond_tier(ac, [1], 3) == mdist.NEAR  # bonded to one: second sphere, floored


def test_the_second_sphere_floor_rejects_a_collapse_but_clears_a_real_agostic():
    from rxembed.constraints import Constraints

    col, pos = _bare_sphere(
        ["Pd", "N", "C", "C", "Cl", "Cl"],
        [(1, 2), (2, 3)],
        [(0, 0, 0), (2.101, 0, 0), (1.502, 1.577, 0), (2.9, 2.6, 0), (0, 2.385, 0), (0, -2.385, 0)],
    )
    cons = Constraints()
    mdist.nondonor_floors(col, 0, 46, [1, 4, 5], cons)
    assert np.linalg.norm(pos[2] - pos[0]) < cons.floors[(0, 2)], "the alpha-C is floored, not exempt"
    assert perceive.metal_overbond(col, pos, [1, 4, 5])

    ti, pos = _bare_sphere(
        ["Ti", "C", "C", "Cl", "Cl", "Cl"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (2.038, 1.539, 0), (-2.2, 0, 0), (0, -2.2, 0), (0, 0, 2.2)],
    )
    cons = Constraints()
    mdist.nondonor_floors(ti, 0, 22, [1, 3, 4, 5], cons)
    assert np.linalg.norm(pos[2] - pos[0]) > cons.floors[(0, 2)]
    assert not perceive.metal_overbond(ti, pos, [1, 3, 4, 5])


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
@pytest.mark.parametrize("name", _DFT)
def test_every_floor_accepts_the_real_dft_geometry_it_exists_to_reproduce(name):
    from rdkit.Chem import GetPeriodicTable

    from rxembed.constraints import Constraints
    from rxembed.metal_core import TRANSITION_METALS
    from rxembed.pipeline.perceive import _xyz_to_mol

    pt = GetPeriodicTable()
    mol = _xyz_to_mol(f"examples/structures/{name}.xyz", 0)
    pos = mol.GetConformer().GetPositions()
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]
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
        assert not perceive.metal_overbond(mol, pos, donors), "the gate rejects a real DFT geometry"


# --- a declared donor is a donor, whatever its element ------------------------------------------------------


def _ruthenium(d_ruh=1.701, d_rucl=2.233):
    """DUKPII: a terminal hydride and a terminal chloride on one Ru, both bond-less after the surrogate's strip."""
    return _bare_sphere(
        ["Ru", "H", "Cl", "P", "P"],
        [],
        [(0, 0, 0), (d_ruh, 0, 0), (-d_rucl, 0, 0), (0, 2.341, 0), (0, -2.341, 0)],
    )


def test_a_declared_hydride_passes_the_gate_that_has_no_covalent_ruler_for_it():
    mol, pos = _ruthenium()
    donors = [1, 2, 3, 4]
    assert 1 in perceive._spheres(mol, pos, donors)[0]
    assert not geom.hydrogens(mol, pos, donors=frozenset(donors))
    assert geom.check(mol, mol.GetConformer().GetId(), donors=donors).ok()


def test_an_undeclared_agostic_h_is_not_promoted_to_a_donor():
    mol, pos = _bare_sphere(
        ["Ru", "C", "H", "P", "P"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (1.85, 0, 0.9), (0, 2.341, 0), (0, -2.341, 0)],
    )
    donors = [1, 3, 4]
    assert 2 not in perceive._spheres(mol, pos, donors)[0]
    assert not perceive.metal_overbond(mol, pos, donors)


@pytest.mark.parametrize("kind", ["hydride", "chloride"])
def test_a_declared_donor_buried_in_the_metal_is_still_caught(kind):
    at = 1 if kind == "hydride" else 2
    mol, pos = _ruthenium(**{"d_ruh" if kind == "hydride" else "d_rucl": 0.100})
    v = perceive.metal_overbond(mol, pos, [1, 2, 3, 4])
    assert [x.kind for x in v] == ["metal_collapse"]
    assert v[0].atoms == (0, at)
    assert not geom.check(mol, mol.GetConformer().GetId(), donors=[1, 2, 3, 4]).ok()


# --- the two FF caps, seen through the gate on real complexes ------------------------------------------------

_SCHREINER = "FC(F)(F)c1cc(cc(c1)C(F)(F)F)NC(=S)Nc1cc(cc(c1)C(F)(F)F)C(F)(F)F.CC(C)=O"
_CHB = "C1CSC2=NC(CN12)c1ccccc1.CC(=O)OC(C)=O"  # tetramisole isothiourea + acetic anhydride


def _seeded(smi, seed=1, pick=None):
    """Embed an NCI complex on its first (or `pick`-matched) discovered mode -> the ensembles to check."""
    import rxembed.pipeline as rx

    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if pick is None:
        contact = next(iter(rx.nci_modes(mol).values()))
    else:
        cands = rx.nci_candidates(mol)
        contact = cands[next(k for k in cands if k.startswith(pick))]
    res = rx.embed(smi, seed=seed, n=4, contacts=contact)
    return list(res) if isinstance(res, rx.EnsembleSet) else [res]


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_the_conjugation_cap_holds_the_thiourea_c_s_n_plane_through_the_relax():
    seen = 0
    for ens in _seeded(_SCHREINER):
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            seen += 1
            assert "conjugation" not in _kinds(geom.check(ens.mol, int(cid), frozen=frozen))
    assert seen, "no conformer was produced: the gate assertion never ran"


@pytest.mark.skipif(find_spec("xyzgraph") is None or find_spec("networkx") is None, reason="needs rxembed[nci]")
def test_both_caps_active_leave_the_chb_complexes_sp2_carbons_planar():
    seen = 0
    for ens in _seeded(_CHB, pick="ChB"):
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            seen += 1
            assert "planarity" not in _kinds(geom.check(ens.mol, int(cid), frozen=frozen))
    assert seen, "no conformer was produced: the gate assertion never ran"


def test_the_sp2_hold_rides_its_window_rather_than_pinning_flat_or_freeing_the_bowl():
    from rdkit.Chem import rdDistGeom

    from rxembed import mechanisms
    from rxembed.constraints import Constraints
    from rxembed.relax import restrained_uff

    mol = Chem.AddHs(Chem.MolFromSmiles("c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=1) == 0

    def worst_improper():
        conf = mol.GetConformer()
        sp2 = [
            (a.GetIdx(), [n.GetIdx() for n in a.GetNeighbors()])
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP2 and a.GetDegree() == 3
        ]
        # indexed, not *nb: the degree-3 filter above is what makes the length 3, and a star-unpack hides that
        return max((abs(rdMolTransforms.GetDihedralDeg(conf, nb[0], nb[1], nb[2], c)) for c, nb in sp2), default=0.0)

    assert worst_improper() < 1.0, "ETKDG did not seed corannulene flat: the R4 premise is void"
    restrained_uff(mol, Constraints())
    held = worst_improper()
    assert 0.5 * mechanisms._SP2_HOLD_WIN < held < mechanisms._SP2_HOLD_WIN + 3.0, (
        f"worst sp2 improper {held:.1f} deg; expected it to ride the +/-{mechanisms._SP2_HOLD_WIN:.0f} deg window"
    )
