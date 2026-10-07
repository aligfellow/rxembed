"""Test donor orientation and coplanarity restraints."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem import rdMolTransforms as T

import rxembed as rx
from rxembed import metal_donor_orient
from rxembed.constraints import Constraints
from rxembed.mechanisms import Angle

# the N-bound Ni(II) linkage isomer: a carboxylate O donor (one heavy neighbour) and an amidate N donor (two
# heavy neighbours) on the same metal: one fixture exercises both `coplanar_donor` permutations.
NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
_ETA2_SMI = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# a Ni(II) thiosemicarbazone: the thione C=S SULFUR is a conjugated sp2 donor the old {7, 8} N/O list dropped
# an aryl-carbanion (aromatic ipso CARBON) donor; conjugated sp2, also dropped by the {7, 8} list
# an isolated acetone O donor: sp2 with an in-plane lone pair, but RDKit marks its C=O not conjugated. The
# pyridine co-donor is conjugated and capped either way, so any drop is the ketone O's alone.
_KETONE_SMI = "CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1"
# an aryl thiolate S donor: RDKit types it sp2 from the ring's aromaticity, but the S carries no pi bond of
# its own, so the pi count gives sp3 and the two estimators disagree (period-3, so {7, 8} does not override)
_ARYL_THIOLATE_SMI = "[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[S-]c1ccccc1"
SP2 = Chem.HybridizationType.SP2
_WALL_SLACK = 1.0  # deg: a UFF torsion constraint is a penalty, not a hard wall, so a minimum riding the cap
# edge settles a hair outside it. Wide enough for that, far narrower than any fold the cap exists to stop.


def _donor(iso, symbol):
    return next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == symbol)


def _capped(iso):
    return {e[1] for e in iso.cons.coplanar}


# --- the orientation wall: calibrated classes only -------------------------------------------------------


@pytest.mark.parametrize("ligand", ["n1nn[nH]c1"])
def test_free_sp2_donor_accepts_native_ligand_bisector(ligand):
    iso = rx.metal(f"[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-{ligand}", "SPL")[0]
    donor = next(
        d for d in iso.donors if sum(n.GetAtomicNum() > 1 for n in iso.graph.GetAtomWithIdx(d).GetNeighbors()) == 2
    )
    left, right = [n.GetIdx() for n in iso.graph.GetAtomWithIdx(donor).GetNeighbors()]
    maps = []
    fragments = Chem.GetMolFrags(iso.graph, asMols=True, fragsMolAtomMapping=maps)
    block, indices = next((block, indices) for block, indices in zip(fragments, maps, strict=True) if donor in indices)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed, params.numThreads = 42, 1
    assert rdDistGeom.EmbedMolecule(block, params) == 0
    assert rdForceFieldHelpers.UFFOptimizeMolecule(block, maxIters=2000) == 0
    xyz = block.GetConformer().GetPositions()
    origin = xyz[indices.index(donor)]
    rays = xyz[[indices.index(left), indices.index(right)]] - origin
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    external = -rays.sum(axis=0)
    external /= np.linalg.norm(external)
    work = Chem.Mol(iso.graph)
    work.RemoveAllConformers()
    conf = Chem.Conformer(work.GetNumAtoms())
    for atom, position in zip(indices, xyz, strict=True):
        conf.SetAtomPosition(atom, position)
    conf.SetAtomPosition(iso.metal, origin + 2.0 * external)
    work.AddConformer(conf)
    angles = {key: window for key, window in iso.cons.angles.items() if key[:2] == (iso.metal, donor)}
    assert len(angles) == 2
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(work)
    Angle().uff_terms(ff, Constraints(angles=angles), work.GetConformer(), 1.0)
    ff.Initialize()
    assert ff.CalcEnergy() == pytest.approx(0.0, abs=1e-8), "donor walls oppose the native ligand's external bisector"
    work.GetConformer().SetAtomPosition(iso.metal, origin + 2.0 * rays[0])
    assert ff.CalcEnergy() > 0.0, "centring must still penalize donation into a substituent"

    order = list(reversed(range(work.GetNumAtoms())))
    permuted = Chem.RenumberAtoms(work, order)
    changed = Constraints()
    metal_donor_orient.orient_donor(permuted, order.index(iso.metal), order.index(donor), {order.index(donor)}, changed)
    remapped = {tuple(order[i] for i in key): value for key, value in changed.angles.items()}
    assert remapped == angles, "atom order or source coordinates changed the graph-derived donor windows"


# MOCQIE's N12: xyz2mol writes the cumulated O=[N+]=C Kekule form. Both estimators type the metal-bound N sp,
# but an sp centre with 2 ligand sigma bonds has used its sigma framework and both pi orbitals on those bonds,
# leaving no sigma lone pair for the dative M-N bond that is drawn; the real donor is the bent, sp2 resonance
# form. A genuine sp donor (nitrile N, carbyne C) has exactly one ligand sigma bond and is unaffected.
_CUMULATED_NITRO_SMI = "O=[N+](=CC)->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"


def test_declared_cumulated_sp_donor_is_retyped_sp2():
    """The retype reads the caller's declaration, never the graph's own metal bond."""
    mol = rx.parse_smiles(_CUMULATED_NITRO_SMI)
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert mol.GetAtomWithIdx(nitrogen).GetHybridization() == Chem.HybridizationType.SP, "fixture premise"
    assert metal_donor_orient.stripped_hybridisation(mol)[nitrogen] == Chem.HybridizationType.SP
    assert metal_donor_orient.stripped_hybridisation(mol, {nitrogen: 1})[nitrogen] == Chem.HybridizationType.SP2


def test_a_bridging_donor_gets_no_terminal_orientation_wall():
    """A donor declared on two metals takes its axis from the bridge, in enforcement as in the gate."""
    mol = rx.parse_smiles("C[S-]1->[Pd+2](<-[Cl-])(<-[Cl-])<-[S-](C)->[Pd+2]1(<-[Cl-])<-[Cl-]")
    sulfur = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")
    metal = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(sulfur).GetNeighbors() if nb.GetSymbol() == "Pd")
    donors = {nb.GetIdx() for nb in mol.GetAtomWithIdx(metal).GetNeighbors()}
    cons = Constraints()
    metal_donor_orient.orient_donor(mol, metal, sulfur, donors, cons, metals=2)
    assert metal_donor_orient.donation_axis(mol, sulfur, donors, metals=2) is None
    assert not cons.angles


@pytest.mark.parametrize("smi", ["Cl[Pd](Cl)(Cl)<-[S](=O)(C)CC"], ids=["neutral-double"])
def test_sulfoxide_lewis_forms_are_pyramidal_donors(smi):
    iso = rx.metal(smi, "square_planar", stereo="free")[0]
    sulfur = _donor(iso, "S")
    assert metal_donor_orient.stripped_hybridisation(iso.mol)[sulfur] == Chem.HybridizationType.SP3
    assert len([k for k in iso.cons.angles if k[1] == sulfur]) == 3
    assert sulfur not in _capped(iso)


def test_aryl_thiolate_sulfur_gets_the_pyramidal_orientation_wall():
    # FISCIT: an aryl thiolate S typed sp2 by RDKit's ring conjugation, sp3 by lone-pair count, got no wall
    # at all (both estimators must agree per class). Period 3+ does not planarise (unlike N/O), so the fix
    # resolves the disagreement to sp3, not sp2.
    iso = rx.metal(_ARYL_THIOLATE_SMI, "square_planar")[0]
    sulfur = _donor(iso, "S")
    carbon = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(sulfur).GetNeighbors() if nb.GetAtomicNum() > 1)

    assert metal_donor_orient.stripped_hybridisation(iso.mol)[sulfur] == Chem.HybridizationType.SP3
    assert (iso.metal, sulfur, carbon) in iso.cons.angles


@pytest.mark.parametrize(
    ("name", "smi", "symbol", "conjugated"),
    [("isolated ketone O", _KETONE_SMI, "O", False)],
    ids=["isolated-ketone-O"],
)
def test_graph_recovers_missed_donors(name, smi, symbol, conjugated):
    iso = rx.metal(smi, "square_planar")[0]
    d = next(
        x
        for x in iso.donors
        if iso.mol.GetAtomWithIdx(x).GetSymbol() == symbol
        and (symbol != "C" or iso.mol.GetAtomWithIdx(x).GetIsAromatic())
    )
    assert any(b.GetIsConjugated() for b in iso.mol.GetAtomWithIdx(d).GetBonds()) is conjugated, f"{name}: fixture"
    assert metal_donor_orient.stripped_hybridisation(iso.mol).get(d) == SP2, f"{name} must type sp2"
    assert d in _capped(iso), f"{name} must receive a coplanarity cap"


@pytest.mark.parametrize(
    "smi",
    [
        "COc1c[cH](->[Pd+]23<-[O-]CCOc4ccc5ccc6ccc[n]->2c6c5[n]->34)cc(OC)c1OC",
        "C[c]1(->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])ccccc1",
    ],
    ids=["tethered-arene-CH", "arene-ipso-carbon"],
)
def test_aromatic_carbon_attachment_keeps_ligand_geometry(smi):
    iso = rx.metal(smi, "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=1, seed=42, threads=1)
    mol = ens.mol
    donor = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "C")
    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors() if n.GetIdx() != iso.metal]
    pos = mol.GetConformer(ens.ids[0]).GetPositions()
    left, middle, right = pos[neighbors]
    normal = np.cross(middle - left, right - left)
    height = abs(float(np.dot(pos[donor] - left, normal))) / np.linalg.norm(normal)
    assert height < 0.1, f"aromatic donor pyramidalised by {height:.3f} Å"
    for neighbor in neighbors:
        if mol.GetAtomWithIdx(neighbor).GetAtomicNum() == 1:
            length = float(np.linalg.norm(pos[donor] - pos[neighbor]))
            assert 1.05 < length < 1.12, f"aromatic C-H stretched to {length:.3f} Å"
    assert rx.cxsmiles(mol) == rx.cxsmiles(iso)
    ens.check()[ens.ids[0]].assert_ok()


@pytest.mark.parametrize("orientation", [True, False])
def test_eta1_cyclopentadienyl_keeps_coordination_geometry(orientation):
    iso = rx.metal("[cH-]1(->[Pd+2](<-[Cl-])(<-[Cl-])<-[NH3])cccc1", "square_planar", stereo="free")[0]
    ens = rx.embed(iso, n=1, params=rx.EmbedParams(seed=42, threads=1, donor_orientation=orientation))
    mol = ens.mol
    donor = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "C")
    hydrogen = next(n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors() if n.GetAtomicNum() == 1)
    assert 1.05 < T.GetBondLength(mol.GetConformer(ens.ids[0]), donor, hydrogen) < 1.17
    assert all(v.kind == "donor_orientation" for v in ens.check()[ens.ids[0]].violations)


def test_only_an_inplane_sp2_donor_is_capped():
    for name, smi, geometry in (
        ("sp3 amine", "CCN[Pd](Cl)Cl", "square_planar"),
        ("sp nitrile", "CC#N[Pd](Cl)Cl", "square_planar"),
        ("side-on eta2", _ETA2_SMI, "square_planar"),
    ):
        isos = list(rx.metal(smi, geometry))
        assert isos, f"{name}: nothing was enumerated, so no cap was ever inspected"
        for iso in isos:
            for _i, d, _k, _w, _anc, _cap in iso.cons.coplanar:
                assert metal_donor_orient.stripped_hybridisation(iso.mol).get(d) == SP2, (
                    f"{name}: a non-sp2 donor was capped"
                )
                haptic = any(nb.GetIdx() in set(iso.donors) for nb in iso.mol.GetAtomWithIdx(d).GetNeighbors())
                assert not haptic, f"{name}: a haptic donor was capped"


def test_backbone_target_picks_the_nearer_donor_across_a_bridging_donor():
    """R_pair must exempt a bulky donor's arm off the nearer co-donor down its own backbone, not a donor
    reached only by continuing past it (a macrocycle can route one arm to either).
    """
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(46))
    na, methyl_a, methyl_b = (rw.AddAtom(Chem.Atom(z)) for z in (7, 6, 6))  # a bulky tertiary amine donor
    arm_a, arm_b = (rw.AddAtom(Chem.Atom(6)) for _ in range(2))
    pmid = rw.AddAtom(Chem.Atom(15))  # the nearer donor: a small, calibrated phosphine
    arm_c, arm_d = (rw.AddAtom(Chem.Atom(6)) for _ in range(2))
    nb = rw.AddAtom(Chem.Atom(8))  # the farther donor: an uncalibrated ether, reached only past pmid
    for a, b in (
        (na, methyl_a),
        (na, methyl_b),
        (na, arm_a),
        (arm_a, arm_b),
        (arm_b, pmid),
        (pmid, arm_c),
        (arm_c, arm_d),
        (arm_d, nb),
    ):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    for donor in (na, pmid, nb):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    donors = {na, pmid, nb}

    cons = Constraints()
    metal_donor_orient.orient_donor(mol, metal, na, donors, cons)
    assert (metal, na, methyl_a) in cons.angles
    assert (metal, na, methyl_b) in cons.angles
    assert (metal, na, arm_a) not in cons.angles, "the nearer donor pmid is small and calibrated: R_pair exempts it"


# --- the coplanarity cap: which donors get one -----------------------------------------------------------


def test_conjugated_bridge_chelate_gets_the_measured_ring_closure_hinge():
    # A bridging carboxylate is a conjugated 4-ring, `ring_hinge`'s case: one coplanar row per donor of the
    # ring, walked pairwise through the bridge (never needs a dedup guard for a reciprocal duplicate).
    source = Chem.MolFromSmiles("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1")
    for mol in (source, Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))):
        iso = rx.metal(mol, "tetrahedral")[0]
        oxygens = {donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetAtomicNum() == 8}
        hinge = [row for row in iso.cons.coplanar if {row[1], row[2]} == oxygens]

        assert len(hinge) == 2, "one row per donor of the ring"
        for metal, d, other, bridge, _anchor, _cap in hinge:
            assert metal == iso.metal
            assert iso.mol.GetBondBetweenAtoms(bridge, d) is not None
            assert iso.mol.GetBondBetweenAtoms(bridge, other) is not None


# --- the coplanarity cap is SOFT: a window the relax lands inside, never a pin ---------------------------
# The bound every test below reads is the cap's own declared half-width, taken off the `cons.coplanar` entry.
# The effect sizes it replaces are on-vs-off medians and tails, per donor, over 5-8 seeds.


def _cap_deviation(smi, seeds, n):
    """Per-donor deviation (deg) of each CAPPED dihedral from its nearest in-plane well, pooled over `seeds`."""
    out: dict[int, list[float]] = {}
    for seed in seeds:
        iso = rx.metal(smi, "square_planar")[0]
        entries = list(iso.cons.coplanar)
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        for i, j, k, w, _anc, _cap in entries:
            for cid in ens.ids:
                phi = abs(T.GetDihedralDeg(ens.mol.GetConformer(cid), int(i), int(j), int(k), int(w)))
                out.setdefault(j, []).append(min(phi, 180.0 - phi))
    return {j: np.array(v) for j, v in out.items()}


def test_relax_lands_inside_cap_without_flattening():
    cap = metal_donor_orient.COPLANAR_CAP
    dev = _cap_deviation(NI_N, (6,), n=1)
    assert dev, "no capped donor was measured: the fixture is wrong"
    for j, arr in dev.items():
        assert arr.max() <= cap + _WALL_SLACK, f"donor {j}: the relax broke through its cap ({arr.max():.1f}°)"


# --- every local donor plane survives the metal-bond strip ------------------------------------------------
# Two donors in one conjugated ligand still define different M-D-X-Y inequalities. An all-sp2 path is not a
# proof that either local term is redundant.

# a crowded Ni(II) conjugated chelate: a pyridylimine sharing one sp2 plane with an amidate.
# picolinate-ethylenediamine Ni: a rigid conjugated bidentate with two distinct local donor planes.
