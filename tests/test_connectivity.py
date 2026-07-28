"""Connectivity: is the conformer still the molecule we asked for?

``bonding_ok`` is heavy-atoms-only (blind to proton transfer) and only fires below 0.7x the covalent sum, so a
new C-C bond at 1.54 A passes it. These pin the hole-closers: the ``connectivity`` diff, the ``metal_overbond``
gate, ``.filter('connectivity')``, and the ``bounds.py`` key-order bug.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import rxembed as rx
from rxembed import geometry as geom
from rxembed import metrics
from rxembed.embed.dispatch import _xyz_to_mol
from rxembed.rdkit_embed import coordination as coord
from rxembed.rdkit_embed.constraints import metal as _metal
from rxembed.rdkit_embed.constraints.base import Constraints, add_distance
from rxembed.rdkit_embed.embed import bounds


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _sphere(symbols, bonds, coords):
    """A bare metal + ligand skeleton with an explicit conformer (no sanitisation, no implicit H)."""
    from rdkit.Geometry import Point3D

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


# --- the connectivity diff ---------------------------------------------------


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)Nc1ccccc1", "OC(=O)CCCCc1ccccc1"])
def test_clean_conformer_has_no_diff(smiles):
    """A sane conformer re-perceives to exactly the graph it came from."""
    m = _mol(smiles)
    formed, broken = metrics.connectivity(m, m.GetConformers()[0].GetId())
    assert not formed
    assert not broken


def test_detects_a_broken_bond():
    """Pull a fragment away: the bond must be reported broken."""
    m = _mol("CCO")
    conf = m.GetConformer()
    o = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "O")
    p = conf.GetAtomPosition(o)
    conf.SetAtomPosition(o, (p.x + 6.0, p.y, p.z))
    _formed, broken = metrics.connectivity(m, conf.GetId())
    assert any(o in pair for pair in broken), f"a 6 A-displaced O was not reported broken: {broken}"


def test_detects_a_proton_transfer_which_bonding_ok_cannot_see():
    """Move an H from N to O: bonding_ok is heavy-only and misses it."""
    m = _mol("[NH3+]CC(=O)[O-]")
    conf = m.GetConformer()
    n = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N")
    o = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    h = next(x.GetIdx() for x in m.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 1)
    c = next(x.GetIdx() for x in m.GetAtomWithIdx(o).GetNeighbors() if x.GetAtomicNum() == 6)
    po, pc = np.array(conf.GetAtomPosition(o)), np.array(conf.GetAtomPosition(c))
    away = (po - pc) / np.linalg.norm(po - pc)
    conf.SetAtomPosition(h, (po + 0.98 * away).tolist())  # a real O-H length away from C: a transfer, not an H-bond
    formed, broken = metrics.connectivity(m, conf.GetId())
    assert (o, h) in formed or (h, o) in formed, f"proton transfer not seen as formed: {formed}"
    assert (n, h) in broken or (h, n) in broken, f"proton transfer not seen as broken: {broken}"
    assert metrics.bonding_ok(m, conf.GetId()), "bonding_ok is heavy-atom-only — it CANNOT see this"


def test_frozen_core_partial_bonds_are_exempt():
    """A TS core's partial bonds are held to the reference by design and must not be judged."""
    ens = rx.embed("examples/structures/bimp.xyz", fix=[14, 15, 16, 17], n=1, seed=1)
    for cid in ens.ids:
        formed, broken = metrics.connectivity(ens.mol, cid, exclude=ens.cons.frozen)
        assert not formed
        assert not broken


def test_metal_dative_bonds_are_not_judged_as_covalent():
    """Metal pairs are excluded (a dative bond has no covalent yardstick) — coordination_changed judges them."""
    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    ens = rx.embed(iso, n=2, seed=1).minimize()
    for cid in ens.ids:
        formed, broken = metrics.connectivity(ens.mol, cid, metals={iso.metal}, elements={iso.metal: iso.real_z})
        assert not formed
        assert not broken


def test_the_metal_branch_actually_fires_through_the_pipeline():
    """`.filter('connectivity')`'s metal branch fires through the public API (Ensemble.sphere carries the donors)."""
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    ens = rx.embed(iso, n=6, seed=1).minimize()
    assert ens.sphere, "the coordination sphere was forgotten by minimize()"
    assert not ens._scan_connectivity(), "a healthy metal ensemble must not be flagged"

    conf = ens.mol.GetConformer(ens.ids[0])  # rip one ligand off
    d = iso.donors[0]
    p, pm = np.array(conf.GetAtomPosition(d)), np.array(conf.GetAtomPosition(iso.metal))
    conf.SetAtomPosition(d, (pm + 4.0 * (p - pm) / np.linalg.norm(p - pm)).tolist())

    before = ens.n
    assert ens._scan_connectivity(), "a dissociated ligand was not seen through the pipeline"
    ens.filter("connectivity")
    assert ens.n == before - 1, "filter must drop the dissociated conformer and ONLY that one"


def test_coordination_changed_sees_a_dissociating_ligand():
    """A donor dragged off the metal is reported as having left the coordination sphere."""
    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    cid = ens.ids[0]
    left, joined = metrics.coordination_changed(ens.mol, cid, iso.metal, iso.donors)
    assert not left  # intact to begin with
    assert not joined

    conf = ens.mol.GetConformer(cid)
    d = iso.donors[0]
    p, pm = np.array(conf.GetAtomPosition(d)), np.array(conf.GetAtomPosition(iso.metal))
    conf.SetAtomPosition(d, (pm + 4.0 * (p - pm) / np.linalg.norm(p - pm)).tolist())  # 4 A out
    left, _ = metrics.coordination_changed(ens.mol, cid, iso.metal, iso.donors)
    assert d in left


@pytest.mark.parametrize(
    "smi",
    [
        # a 1.71 A C=P phosphaalkene: xyzgraph refuses to perceive it and calls the bond broken
        "Cc1cc(C)c([CH]2=[PH]->[Ni+2]<-23<-[O-]C(=O)C(c2ccccc2)[N-]->3c2ccccc2)c(C)c1",
        # a 1,3 geminal pair at 2.02 A: xyzgraph calls that separation a newly formed bond
        "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    ],
)
def test_no_false_positives_on_healthy_metal_catalysts(smi):
    """A perceived bond change counts only when the geometry agrees: neither healthy Henry catalyst may be flagged."""
    for iso in rx.metal(smi, "square_planar", stereo="free"):
        ens = rx.embed(iso, n=3, seed=1).minimize()
        if not ens.n:
            continue
        assert not ens._scan_connectivity(), "a healthy catalyst was flagged as having reacted"
        assert ens.filter("connectivity").n == ens.n  # ...and filter must not eat the ensemble
        break


# --- the metal over-bond gate ------------------------------------------------


def test_metal_overbond_is_silent_on_good_geometry():
    """Zero violations on clean embeds and real DFT geometries — the gate must not fire on real chemistry."""
    for smi, geo in [("Cl[Pd](Cl)(N)N", "square_planar"), ("Br[Pd]1(Cl)NCCN1", "square_planar")]:
        for iso in rx.metal(smi, geo):
            ens = rx.embed(iso, n=2, seed=1).minimize()
            for cid in ens.ids:
                pos = ens.mol.GetConformer(cid).GetPositions()
                assert not coord.metal_overbond(ens.mol, pos, iso.donors)


def test_metal_overbond_does_not_false_positive_on_a_bimetallic_spectator():
    """Donor sets are per-metal: the spectator ferrocene's 10 Cp carbons must not read as collapsed onto its Fe."""
    rc = [1, 5, 63, 64, 65, 66]
    isos = rx.metal("examples/structures/mn-h2.xyz", "octahedral", center="Mn", fix=rc)
    ens = rx.embed(isos[0], n=1, seed=1)
    for cid in ens.ids:
        pos = ens.mol.GetConformer(cid).GetPositions()
        assert not coord.metal_overbond(ens.mol, pos, isos[0].donors)


def test_metal_overbond_fires_when_a_third_sphere_atom_collapses_onto_the_metal():
    """It must fire — and only on a third-sphere atom."""
    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1).minimize()
    mol, m, donors = ens.mol, iso.metal, set(iso.donors)
    topo = Chem.GetDistanceMatrix(mol)
    hops = {i: 1 + min(topo[d][i] for d in donors) for i in range(mol.GetNumAtoms())}
    c = next(i for i, h in hops.items() if h >= 3 and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

    clean = mol.GetConformer(ens.ids[0]).GetPositions().copy()
    assert not coord.metal_overbond(mol, clean, donors)  # clean to begin with

    pos = clean.copy()
    _u = pos[c] - pos[m]  # crush a third-sphere heavy atom onto the Ni — a deep burial past the donor floor,
    pos[c] = pos[m] + _u / float(np.linalg.norm(_u)) * 1.2  # computed (the clean M...c distance is embed-dependent)
    v = coord.metal_overbond(mol, pos, donors)
    assert v
    assert v[0].kind == "metal_overbond"
    assert c in v[0].atoms

    # Perceiving the donors is circular: the collapse itself makes the atom "a donor", so it is judged by the
    # donor floor (may it be this close to a metal?) instead of the non-donor one (may it be here at all?) —
    # and the over-bond is never reported. Shown at a crush onto a plausible BONDING distance, which isolates
    # the circularity from the donor floor: nothing here is short enough for a collapse.
    near = clean.copy()
    u = near[c] - near[m]  # crush to a fixed 1.75 A bond length — the clean M...c distance is embed-dependent, so a
    near[c] = near[m] + u / float(np.linalg.norm(u)) * 1.75  # fixed *multiplier* lands at a distance that drifts
    assert 1.7 < float(np.linalg.norm(near[c] - near[m])) < 1.8  # a bond length, not a burial
    assert coord.metal_overbond(mol, near, donors), "the declared-donor gate must still see this collapse"
    assert not coord.metal_overbond(mol, near, None), "perceived-donor mode is expected to be circular"

    # The circularity is bounded, not total: past the donor floor even a "perceived donor" is judged, since a
    # donor's licence is a bonding window, not a half-line to zero. It is the wrong ATOM, but not silence.
    assert [x.kind for x in coord.metal_overbond(mol, pos, None)] == ["metal_collapse"]


def test_metal_overbond_accepts_real_crystal_geometry_2_bonds_out():
    """An atom bonded to a donor (2 hops out) is placed by the ligand backbone, not an over-bond, and must pass."""
    # Pd kappa2-acetate bite apex (CMD/AMLA C-H activation): Pd | O O (donors) | C carboxyl at 2.460 A | C methyl
    ac, pos = _sphere(
        ["Pd", "O", "O", "C", "C"],
        [(1, 3), (2, 3), (3, 4)],
        [(0, 0, 0), (1.12, 1.68, 0), (-1.12, 1.68, 0), (0, 2.46, 0), (0, 3.96, 0)],
    )
    assert np.linalg.norm(pos[3] - pos[0]) == pytest.approx(2.460, abs=0.005)
    assert not coord.metal_overbond(ac, pos, [1, 2]), "the gate rejects a real kappa2-acetate crystal"

    # Ti beta-agostic ethyl (Dawoodi/Green, JCS Dalton 1986): Ti | C_alpha (donor) | C_beta at 2.554 A
    ti, pos = _sphere(["Ti", "C", "C"], [(1, 2)], [(0, 0, 0), (2.15, 0, 0), (1.60, 1.99, 0)])
    assert np.linalg.norm(pos[2] - pos[0]) == pytest.approx(2.554, abs=0.01)
    assert not coord.metal_overbond(ti, pos, [1]), "the gate rejects a real beta-agostic crystal"


# --- the .filter() verb ------------------------------------------------------


def test_filter_keeps_an_intact_ensemble_untouched():
    ens = rx.embed("CCO", n=4, seed=1).minimize()
    before = list(ens.ids)
    assert ens.filter("connectivity").ids == before


def test_filter_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="connectivity"):
        rx.embed("CCO", n=1, seed=1).filter("rmsd")


def test_prune_composes_connectivity_in_the_cascade():
    """prune(by=['connectivity', 'rmsd']) must run the validity filter, then the dedup."""
    ens = rx.embed("OC(=O)CCCCc1ccccc1", n=6, seed=1)
    ens.prune(by=["connectivity", "rmsd"])
    assert ens.ids


def test_ensembleset_filter_keeps_both_meanings():
    """`set.filter(tag=...)` selects candidates (pre-existing); `set.filter('connectivity')` drops conformers."""
    es = rx.embed("CC(N)C(=O)O", n=2, seed=1)  # a racemate -> EnsembleSet
    assert len(es.filter(stereo="1R")) == 1  # the tag selector must not have been broken
    out = es.filter("connectivity")  # the conformer filter, mapped over candidates
    assert len(out) == len(es)


def test_a_reacted_conformer_is_flagged_and_never_dropped_silently():
    """The geometry-only scan optimize() runs flags a transferred proton, never silently shrinks, then filter raises."""
    ens = rx.embed("[NH3+]CC(=O)[O-]", n=1, seed=1).minimize()
    cid = ens.ids[0]
    conf = ens.mol.GetConformer(cid)
    n = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "N")
    o = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    h = next(x.GetIdx() for x in ens.mol.GetAtomWithIdx(n).GetNeighbors() if x.GetAtomicNum() == 1)
    cc = next(x.GetIdx() for x in ens.mol.GetAtomWithIdx(o).GetNeighbors() if x.GetAtomicNum() == 6)
    po, pc = np.array(conf.GetAtomPosition(o)), np.array(conf.GetAtomPosition(cc))
    away = (po - pc) / np.linalg.norm(po - pc)
    conf.SetAtomPosition(h, (po + 0.98 * away).tolist())  # N-H...O -> N...H-O (a real transfer)

    changed = ens._scan_connectivity()  # exactly what optimize() calls on its output geometry
    assert cid in changed, "a transferred proton must be flagged"
    formed, broken = changed[cid]
    assert formed
    assert broken
    assert ens.ids == [cid], "flagging must NOT shrink the ensemble behind the user's back"
    with pytest.raises(RuntimeError, match="different species"):
        ens.filter("connectivity")


# --- the zero-vdW force-field surrogate --------------------------------------


def test_ff_surrogate_is_a_bounded_vdw_sphere():
    """The bond-less Li surrogate must be UFF-typeable (vdW sphere keeps non-donor H off the metal) yet small."""
    from rdkit.Chem import rdForceFieldHelpers

    from rxembed.rdkit_embed.constraints.metal import FF_SURROGATE

    m = Chem.RWMol(Chem.MolFromSmiles("CC"))
    m.GetAtomWithIdx(0).SetAtomicNum(FF_SURROGATE)
    m.GetAtomWithIdx(0).SetNoImplicit(True)
    mol = m.GetMol()
    mol.UpdatePropertyCache(strict=False)
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(mol), (
        f"RDKit's UFF no longer types Z={FF_SURROGATE}. The FF surrogate must be typeable to carry a vdW sphere; "
        "an untypeable element (Xe) leaves the metal a HOLE and an amine N-H collapses onto it."
    )
    # the sphere keeps a non-donor proton off the metal, which no enumerated floor did
    iso = rx.metal("CCN[Pd](Cl)Cl", "square_planar").select(index=0)
    ens = rx.embed(iso, n=8).minimize()
    md, mp = iso.metal, ens.mol.GetConformer
    worst_h = min(
        float(np.linalg.norm(mp(c).GetPositions()[a.GetIdx()] - mp(c).GetPositions()[md]))
        for c in ens.ids
        for a in ens.mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetIdx() not in iso.donors
    )
    assert worst_h > 2.0, (
        f"an amine N-H collapsed to {worst_h:.2f} A of the metal — the FF surrogate's vdW sphere is not holding "
        "non-donor protons off. Under Xe (zero vdW) this was 1.3 A."
    )


def test_a_bonded_typeable_surrogate_is_the_trap_that_looks_like_the_fix():
    """A bonded UFF-typeable Li can't work: Li types linear (theta0=180), whose angle term is singular -> ~1e10."""
    from rdkit import rdBase
    from rdkit.Chem import rdForceFieldHelpers

    rw = Chem.RWMol()
    li = rw.AddAtom(Chem.Atom(3))  # a bonded Li, square-planar CN4, donors at a perfect 2.00 A
    rw.GetAtomWithIdx(li).SetNoImplicit(True)
    verts = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    for _ in verts:
        d = rw.AddAtom(Chem.Atom(7))
        rw.GetAtomWithIdx(d).SetNoImplicit(True)
        rw.AddBond(li, d, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    for i, v in enumerate(verts):
        conf.SetAtomPosition(i + 1, tuple(2.0 * c for c in v))
    mol.AddConformer(conf)

    with rdBase.BlockLogs():
        assert rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)  # it types — that is the trap
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=False)
        ff.Initialize()
    assert ff.CalcEnergy() > 1e6, (
        "RDKit's UFF no longer explodes on a bonded Li at CN4. The linear-theta0 singularity was the whole "
        "reason a bonded typeable surrogate is unusable — re-open the question and re-measure the sweep."
    )


def test_ff_surrogate_is_a_small_soft_sphere_not_a_hole_or_a_wall():
    """The bond-less Li surrogate must inject a small positive vdW: not zero (a hole) nor a carbon-like wall."""
    from rdkit import rdBase
    from rdkit.Chem import rdForceFieldHelpers

    from rxembed.rdkit_embed.refine.ff import _ff_surrogate

    iso = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")[0]
    ens = rx.embed(iso, n=1, seed=1)
    # `_mol` (the internal working mol), NOT `.mol`: this measures the bond-less Li surrogate's own FF vdW, so
    # it needs the surrogate the engine sees — `.mol` is the finalized public graph (real metal + dative bonds).
    mol, cid, m = ens._mol, ens.ids[0], iso.metal

    def energy(x):
        with rdBase.BlockLogs():
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(x, confId=cid, ignoreInterfragInteractions=False)
            ff.Initialize()
            return ff.CalcEnergy()

    rw = Chem.RWMol(Chem.Mol(mol))
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)
    rw.RemoveAtom(m)
    gone = rw.GetMol()
    gone.UpdatePropertyCache(strict=False)
    injected = energy(_ff_surrogate(mol, {m})) - energy(gone)
    assert 0.1 < injected < 50.0, (
        f"the FF surrogate injected {injected:.1f} kcal/mol — it must be a SMALL soft sphere. Zero means the "
        "metal is a hole (non-donors collapse onto it); tens-to-hundreds means a carbon-like wall that inflates "
        "M-donor distances. Bond-less Li measures ~5.5."
    )


def test_ff_surrogate_is_a_strict_noop_without_a_metal():
    """An organic system must get back the identical object — the metal path has zero blast radius."""
    from rxembed.rdkit_embed.refine.ff import _ff_surrogate

    mol = rx.embed("CCO", n=1, seed=1).mol
    assert _ff_surrogate(mol, set()) is mol


def test_organic_constraints_carry_no_metal_fields():
    """The three metal fields must be empty for an organic system — that is the zero-blast-radius guarantee."""
    ens = rx.embed("CCO", n=1, seed=1)
    assert not ens.cons.metals
    assert not ens.cons.pulls
    assert not ens.cons.floors


def test_relaxed_carries_the_metal_fields():
    """mc(explore=) releases the soft NCI grips — it must not release the metal's structural FF holds."""
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    r = iso.cons.relaxed()
    assert r.metals == iso.cons.metals
    assert r.pulls == iso.cons.pulls
    assert r.floors == iso.cons.floors
    assert r.shapes == iso.cons.shapes


# --- a pull is for a modelled window, never for a member of a rigid shape -----

_MN_H2 = "examples/structures/mn-h2.xyz"
_MN_H2_RC = [1, 5, 63, 64, 65, 66]


def test_a_shape_held_sphere_gets_no_pull_but_a_modelled_one_does():
    """A shape-held M-donor pair (rigid all-pairs body) gets no pull; a degenerate `coordination` window does."""
    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    iso = isos[0]
    spectators = {m for m in iso.cons.metals if m != iso.metal}
    assert spectators, "mn-h2 is bimetallic: the ferrocene Fe must be surrogated as a spectator"
    held = set().union(*iso.cons.shapes)
    assert spectators <= held, "the spectator's sphere is the rigid body hold_shape pinned"
    assert not [k for k in iso.cons.pulls if spectators & set(k)], (
        "a member of a rigid shape was pulled: the relax buys that pull by tearing the pairs it did not pull"
    )
    for d in iso.donors:  # ...while the enumerated centre's modelled window still gets its pull
        assert (min(iso.metal, d), max(iso.metal, d)) in iso.cons.pulls

    smiles_iso = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]  # no input geometry -> nothing shape-held
    assert not smiles_iso.cons.shapes
    assert len(smiles_iso.cons.pulls) == len(smiles_iso.donors)


def test_the_spectator_ferrocenes_shape_is_not_traded_for_its_metal_pulls():
    """Regression: a rigid-body member gets no pull, else the relax satisfies it to 0.000 A by tearing its windows."""
    # find the spectator from the molecule (not cons.shapes) so re-adding the pull fails on the tear, not a missing rec
    ref = _xyz_to_mol(_MN_H2, 0)
    fe = next(
        a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() in _metal.TRANSITION_METALS and a.GetSymbol() != "Mn"
    )
    shape = {fe, *(n.GetIdx() for n in ref.GetAtomWithIdx(fe).GetNeighbors())}

    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    kept = 0
    for k, iso in enumerate(isos):
        windows = {kk: v for kk, v in iso.cons.distances.items() if set(kk) <= shape}
        assert len(windows) > 50, "the spectator ferrocene's rigid body is all pairs of {Fe, *10 Cp carbons}"
        # _retry=False: deterministic path — seed starvation is a separate defect; the subject here is the shape
        ens = rx.embed(iso, n=12, seed=1).minimize(_retry=False)
        kept += ens.n
        for cid in ens.ids:  # every kept conformer, not just the best: a torn body must never survive
            worst = max(
                max(lo - (d := rdMolTransforms.GetBondLength(ens.mol.GetConformer(cid), i, j)), d - hi, 0.0)
                for (i, j), (lo, hi) in windows.items()
            )
            assert worst < 0.15, f"isomer {k} conf {cid}: the spectator's shape tore by {worst:.3f} A"
    assert kept >= 6, f"non-vacuous: the 6 isomers kept only {kept} conformers between them"


# --- the bounds-matrix key-order bug ----------------------------------------


def test_angle_does_not_clobber_an_explicit_distance_window():
    """An angle written (k, j, i) must not clobber the explicit distance window on (i, k): _bounds must sort its key."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0

    def window(angle_key):
        c = Constraints()
        add_distance(c.distances, 0, 3, 1.50, 1.56)
        c.angles[angle_key] = (95.0, 105.0)
        bm, _tol = bounds._bounds(mol, c)  # `_bounds` also returns the tolerance smoothing settled at
        return bm[3][0], bm[0][3]  # (lo, hi) for the pair (0, 3)

    assert window((0, 1, 3)) == pytest.approx(window((3, 1, 0))), "angle index ORDER changed the bounds"
    assert window((3, 1, 0)) == pytest.approx((1.50, 1.56), abs=1e-6), "the explicit window was clobbered"


# --- the tiered anti-overbond floor -----------------------------------------


def test_overbond_tier_discriminates_by_donor_neighbour_count():
    """The one rule the FF floor, the geometry gate and the coordination check all key off."""
    from rxembed.rdkit_embed.constraints import distance as M  # noqa: N812

    # Pd | O O (donors) | C apex bonded to both | C methyl bonded to neither
    ac, _ = _sphere(
        ["Pd", "O", "O", "C", "C"],
        [(1, 3), (2, 3), (3, 4)],
        [(0, 0, 0), (1.12, 1.68, 0), (-1.12, 1.68, 0), (0, 2.46, 0), (0, 3.96, 0)],
    )
    assert M.overbond_tier(ac, [1, 2], 3) == M.APEX  # a chelate bite apex: geometrically forced
    assert M.overbond_tier(ac, [1, 2], 4) == M.OUTER  # third sphere
    assert M.overbond_tier(ac, [1], 3) == M.NEAR  # ...only one donor -> second sphere, floored


def test_the_reporters_sit_strictly_below_the_force_field_floor():
    """A relax comes to rest on its floor — so the gates must report below it, or they flag their own output."""
    from rxembed.rdkit_embed.constraints import distance as M  # noqa: N812

    assert M.NEAR_REPORT_RATIO < M._NEAR_FLOOR_RATIO
    assert M.OUTER_REPORT_MARGIN < M._OVERBOND_MARGIN


def test_second_sphere_floor_rejects_a_collapse_but_accepts_a_real_agostic():
    """The bonded-to-a-donor tier is floored, not exempt: an alpha-C collapse rejects, a beta-agostic accepts."""
    from rxembed.rdkit_embed.constraints.base import Constraints
    from rxembed.rdkit_embed.constraints.distance import nondonor_floors

    # Pd(N donor) + alpha-C at the measured collapse distance, in the empty vertex
    col, pos = _sphere(
        ["Pd", "N", "C", "C", "Cl", "Cl"],
        [(1, 2), (2, 3)],
        [(0, 0, 0), (2.101, 0, 0), (1.502, 1.577, 0), (2.9, 2.6, 0), (0, 2.385, 0), (0, -2.385, 0)],
    )
    assert np.linalg.norm(pos[2] - pos[0]) == pytest.approx(2.178, abs=0.005)
    cons = Constraints()
    nondonor_floors(col, 0, 46, [1, 4, 5], cons)
    assert cons.floors[(0, 2)] == pytest.approx(1.05 * 2.15, abs=0.01)  # floored, not exempt
    assert np.linalg.norm(pos[2] - pos[0]) < cons.floors[(0, 2)]  # ...and the collapse breaches it
    assert coord.metal_overbond(col, pos, [1, 4, 5])
    assert metrics.coordination_changed(col, col.GetConformer().GetId(), 0, [1, 4, 5])[1] == [2]

    # Ti-CH2-CH3: the beta-carbon at the crystal's 2.554 A must survive the same floor
    ti, pos = _sphere(
        ["Ti", "C", "C", "Cl", "Cl", "Cl"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (2.038, 1.539, 0), (-2.2, 0, 0), (0, -2.2, 0), (0, 0, 2.2)],
    )
    assert np.linalg.norm(pos[2] - pos[0]) == pytest.approx(2.554, abs=0.005)
    cons = Constraints()
    nondonor_floors(ti, 0, 22, [1, 3, 4, 5], cons)
    assert np.linalg.norm(pos[2] - pos[0]) > cons.floors[(0, 2)]  # accepted, with headroom
    assert not coord.metal_overbond(ti, pos, [1, 3, 4, 5])
    assert metrics.coordination_changed(ti, ti.GetConformer().GetId(), 0, [1, 3, 4, 5])[1] == []


def test_real_dft_geometries_have_zero_floor_violations_against_their_own_input():
    """The floors must accept the structures rxembed is built to reproduce — its own DFT transition states."""
    from rxembed.embed.dispatch import _xyz_to_mol
    from rxembed.rdkit_embed.constraints.base import Constraints
    from rxembed.rdkit_embed.constraints.distance import nondonor_floors
    from rxembed.rdkit_embed.constraints.metal import TRANSITION_METALS

    for name in ("mn-h2", "ru-co", "mn-hy"):
        mol = _xyz_to_mol(f"examples/structures/{name}.xyz", 0)
        pos = mol.GetConformer().GetPositions()
        for m in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]:
            donors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors())
            cons = Constraints()
            nondonor_floors(mol, m, mol.GetAtomWithIdx(m).GetAtomicNum(), donors, cons)
            assert cons.floors, f"{name}: metal {m} got no floors at all"
            for (i, j), floor in cons.floors.items():
                x = i if j == m else j
                d = float(np.linalg.norm(pos[x] - pos[m]))
                assert d >= floor, f"{name}: floor rejects the real geometry at {mol.GetAtomWithIdx(x).GetSymbol()}{x}"
            assert not coord.metal_overbond(mol, pos, donors), f"{name}: the gate rejects a real DFT geometry"


# --- a declared donor is a donor, whatever its element ------------------------


def _ruthenium(d_ruh=1.701, d_rucl=2.233):
    """The real DUKPII shape: a terminal hydride + a terminal chloride on one Ru, M-D bonds stripped.

    The controlled pair — BOTH donors are monatomic and bond-less after the strip, structurally identical
    and differing only in atomic number. Whatever the gates say of the chloride they must say of the hydride.
    """
    return _sphere(
        ["Ru", "H", "Cl", "P", "P"],
        [],  # the surrogate has stripped every M-donor bond: H and Cl are both orphans
        [(0, 0, 0), (d_ruh, 0, 0), (-d_rucl, 0, 0), (0, 2.341, 0), (0, -2.341, 0)],
    )


def test_a_monatomic_hydride_donor_is_not_a_broken_molecule():
    """A hydride reads as a donor, not a dissociation — the strip leaves it with no bonds to read."""
    mol, pos = _ruthenium()
    donors = [1, 2, 3, 4]
    cid = mol.GetConformer().GetId()
    assert mol.GetAtomWithIdx(1).GetDegree() == 0  # the hydride is orphaned...
    assert mol.GetAtomWithIdx(2).GetDegree() == 0  # ...and so is the chloride, identically

    assert metrics.coordination_changed(mol, cid, 0, donors) == ([], [])  # neither left, nothing joined
    assert 1 in coord._spheres(mol, pos, donors)[0]  # the declared hydride is in the sphere
    assert not geom.hydrogens(mol, pos, donors=frozenset(donors))  # no covalent X-H ruler on a metal-held H
    assert geom.check(mol, cid, donors=donors).ok()


def test_a_hydride_that_really_dissociated_is_still_caught():
    """The exemption must not blind the gate: the dative yardstick still judges the M-H distance."""
    mol, _pos = _ruthenium(d_ruh=5.0)  # the H flew off
    left, joined = metrics.coordination_changed(mol, mol.GetConformer().GetId(), 0, [1, 2, 3, 4])
    assert left == [1]  # the hydride, and only it
    assert joined == []


def test_an_undeclared_agostic_h_is_not_a_donor_and_not_an_overbond():
    """The element screen still does its real job: it classifies atoms nobody declared."""
    mol, pos = _sphere(
        ["Ru", "C", "H", "P", "P"],
        [(1, 2)],
        [(0, 0, 0), (2.10, 0, 0), (1.85, 0, 0.9), (0, 2.341, 0), (0, -2.341, 0)],  # a beta-agostic C-H
    )
    donors = [1, 3, 4]
    assert 2 not in coord._spheres(mol, pos, donors)[0]  # close to Ru, but nobody declared it
    assert metrics.coordination_changed(mol, mol.GetConformer().GetId(), 0, donors) == ([], [])
    assert not coord.metal_overbond(mol, pos, donors)  # ...and it is not an over-bond either


@pytest.mark.parametrize("kind", ["hydride", "chloride"])
def test_a_declared_donor_collapsed_into_the_metal_is_caught(kind):
    """A donor's licence is a bonding WINDOW, not a half-line to zero — the collapse direction is judged.

    Being declared exempts a donor from the *non-donor* floor ("have you reached bonding distance"), which is
    the wrong question to ask it. It must not exempt it from the only question left: is it closer than any
    bond can be. A monatomic donor has no other judge — the strip leaves it bond-less, so every covalent gate
    skips it by construction, and without this it passes the gate buried in the metal.
    """
    at = 1 if kind == "hydride" else 2
    mol, pos = _ruthenium(**{"d_ruh" if kind == "hydride" else "d_rucl": 0.100})
    v = coord.metal_overbond(mol, pos, [1, 2, 3, 4])
    assert [x.kind for x in v] == ["metal_collapse"]
    assert v[0].atoms == (0, at)
    assert not geom.check(mol, mol.GetConformer().GetId(), donors=[1, 2, 3, 4]).ok()


def test_the_donor_collapse_floor_clears_every_real_m_donor_bond():
    """The floor must sit below the tightest bond chemistry allows, or it rejects real structures.

    ``rcov`` is a SINGLE-bond radius, so the shortest real M-D is a triple bond — ~0.82 x the sum on Pyykko's
    triple-bond radii, which is exactly where the corpus's tightest (a back-bonded Mn-CO) sits.
    """
    from rdkit.Chem import GetPeriodicTable

    from rxembed.embed.dispatch import _xyz_to_mol
    from rxembed.rdkit_embed.constraints.distance import DONOR_COLLAPSE_RATIO
    from rxembed.rdkit_embed.constraints.metal import TRANSITION_METALS

    pt = GetPeriodicTable()

    assert DONOR_COLLAPSE_RATIO < 0.814  # the measured tightest real M-D (mn-hy Mn1-C65, 1.750 A)
    tightest = 1e9
    for name in ("mn-h2", "ru-co", "mn-hy"):
        mol = _xyz_to_mol(f"examples/structures/{name}.xyz", 0)
        pos = mol.GetConformer().GetPositions()
        for m in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]:
            donors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors())
            assert not coord.metal_overbond(mol, pos, donors), f"{name}: the collapse floor rejects real DFT"
            rm = pt.GetRcovalent(mol.GetAtomWithIdx(m).GetAtomicNum())
            for d in donors:
                rs = rm + pt.GetRcovalent(mol.GetAtomWithIdx(d).GetAtomicNum())
                tightest = min(tightest, float(np.linalg.norm(pos[d] - pos[m])) / rs)
    assert tightest > DONOR_COLLAPSE_RATIO  # ...with real headroom, not by luck
    assert tightest == pytest.approx(0.814, abs=0.01)


# --- a user-stated bond length is not a broken bond ---------------------------


def test_a_fixed_stretched_bond_survives_the_gate_end_to_end():
    """`rx.embed('CCCl', fix={(1,2): 2.4})` — the documented "constrained TS from SMILES" headline.

    2.4 A is 1.36x the C-Cl covalent sum, past `bonding_ok`'s 1.3x break test: before the constrained-pair
    exemption `minimize()` dropped it as a "broken bond" and handed back an EMPTY ensemble.
    """
    ens = rx.embed("CCCl", fix={(1, 2): 2.4}, n=4, seed=42).minimize()
    assert ens.ids, "the requested dissociating C-Cl was thrown away for being what was asked for"
    for i in ens.ids:
        pos = ens.mol.GetConformer(i).GetPositions()
        assert float(np.linalg.norm(pos[1] - pos[2])) == pytest.approx(2.4, abs=0.1)


def test_the_exemption_is_per_pair_so_a_tear_beside_a_constraint_is_still_caught():
    """A constrained pair buys no amnesty for the rest of the molecule — the over-widening guard."""
    m = _mol("CCCl")
    conf = m.GetConformer()
    stated = {(1, 2): (2.38, 2.42)}

    def stretch(anchor, moved, target):  # place `moved` at `target` A along its own bond axis
        p = conf.GetPositions()
        u = (p[moved] - p[anchor]) / np.linalg.norm(p[moved] - p[anchor])
        conf.SetAtomPosition(moved, (p[anchor] + u * target).tolist())

    stretch(1, 2, 2.4)  # the FIXED C1-Cl2, exactly as the user asked
    assert not metrics.bonding_ok(m, conf.GetId()), "unexempted, the stretched C-Cl reads as broken"
    assert metrics.bonding_ok(m, conf.GetId(), constrained=stated), "a stated length is not a broken bond"

    stretch(1, 0, 2.6)  # now tear the UNCONSTRAINED C0-C1 as well (1.7x the covalent sum)
    assert not metrics.bonding_ok(m, conf.GetId(), constrained=stated), "a torn C-C beside a fixed C-Cl"
