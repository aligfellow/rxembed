"""Metal bis-chelate embedding + the name-agnostic geometric identity (arrangement / chirality).

Regression for the Henry Ni(II) catalysts (dative-bond SMILES, tight 3-membered / side-on bites): the
old ideal-angle chelate window forced ~45° bites toward 90° and tore the ligand bond, so *nothing* embedded.
The fix lets the ligand backbone set the bite (`coordination` drops the intra-chelate angle) and filters
trans-spanning chelate arrangements the backbone can't reach (`isomers._chelate_span_ok`); the clash gate
no longer flags cis coordination partners (`geometry._coordination_pairs`). These lock that in and check the
chirality descriptor on textbook cases. Pure RDKit + UFF, no xtb.
"""

import pytest

from rxembed import geometry as geom
from rxembed.constraints import metal as M  # noqa: N812

# A representative slice of the Henry Ni(II) bis-chelate set — the tight-bite cases that used to fail
# entirely (a side-on C=P phosphaalkene, a direct N=C, a normal O,N amidate + diphosphine/malonate).
HENRY = [
    "Cc1cc(C)c([CH]2=[PH]->[Ni+2]<-23<-[O-]C(=O)C(c2ccccc2)[N-]->3c2ccccc2)c(C)c1",
    "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    "O=C1[O-]->[Ni+2]2(<-[N-](c3ccccc3)C1c1ccccc1)<-[P](Cc1ccccc1[P]->2(C1CCCCC1)C1CCCCC1)(C1CCCCC1)C1CCCCC1",
    "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C",
]


@pytest.mark.parametrize("smi", HENRY)
def test_henry_bischelate_embeds_with_proper_coordination(smi):
    """At least one enumerated isomer embeds with sane dative M-donor distances (the complex is embeddable).

    Some enumerated arrangements are legitimately-infeasible phantoms (a chelate placed *trans* its backbone
    can't span) and correctly embed nothing — the test is that the *feasible* isomer(s) come out clean, not
    that every arrangement does.
    """
    from rdkit.Chem import rdMolTransforms as T

    import rxembed as rx

    isos = rx.metal(smi, "square_planar")
    embedded = 0
    for iso in isos:
        ens = rx.embed(iso, n=3).minimize()
        if not ens.n:  # an infeasible phantom arrangement — fine, skip it
            continue
        embedded += 1
        for cid in ens.ids:
            c = ens.mol.GetConformer(cid)
            for d in iso.donors:  # dative M-donor bonds land ~1.7-2.5 Å (proper coordination, not blown apart)
                assert 1.6 <= T.GetBondLength(c, iso.metal, d) <= 2.6
    assert embedded >= 1, f"{smi} embedded no isomer at all"


def test_normal_chelate_bite_from_backbone_not_forced_ideal():
    """An en chelate on Pd keeps its backbone bite (~78-90°), not the square-planar 90° ideal ± a wide pad."""
    import rxembed as rx

    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    ens = rx.embed(iso, n=6).mc(preset="rapid")
    nn = [d for d in iso.donors if ens.mol.GetAtomWithIdx(d).GetSymbol() == "N"]
    bite = ens.measure((nn[0], iso.metal, nn[1]))["mean"]
    assert 72 <= bite <= 92, f"en bite {bite:.0f}° off the backbone value"


def test_trans_span_filter_drops_impossible_bidentate_trans():
    """A short-backbone bis-chelate square plane enumerates only cis-cis isomers (no trans-spanning bite)."""
    from rdkit import Chem

    import rxembed as rx

    smi = "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    isos = rx.metal(smi, "square_planar")
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(isos[0].mol)) for a in f}
    for iso in isos:  # no enumerated isomer may place a same-ligand donor pair trans (span filter removed those)
        for i in range(len(iso.vertices)):
            for j in range(i + 1, len(iso.vertices)):
                a, b = iso.vertices[i], iso.vertices[j]
                if M.VACANT not in (a, b) and frag[a] == frag[b]:
                    ang = M._vertex_angle(M.VERTEX_DIRS[iso.geometry][i], M.VERTEX_DIRS[iso.geometry][j])
                    assert ang < M._SPAN_ANGLE, f"chelate placed trans ({ang}°) survived the span filter"


@pytest.mark.parametrize(
    ("smi", "geometry"),
    [
        ("CCCN[Pd](Cl)(Cl)NCCC.c1ccccc1", "square_planar"),  # + free benzene
        ("CCN[Pd](Cl)Cl.O.C1CCOC1", "square_planar"),  # + free water + THF
        ("N[Co](N)(N)(Cl)(Cl)Cl.CC#N", "octahedral"),  # + free acetonitrile
    ],
)
def test_free_fragment_embeds_at_vdw_contact_never_infinite(smi, geometry):
    """A metal complex embedded with a separate (`.`) fragment tethers it at vdW contact, not infinity.

    The metal isomer path used to skip the encounter-bounds the general path applies, so an un-bonded
    fragment drifted to hundreds of Å. Whatever the input, separate fragments embedded together must never
    land at infinite separation — a vdW-contact complex is the floor.
    """
    import numpy as np
    from rdkit import Chem

    import rxembed as rx

    embedded = 0
    for iso in rx.metal(smi, geometry):
        ens = rx.embed(iso, n=2).minimize()
        if not ens.n:
            continue
        embedded += 1
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        m = pos[iso.metal]
        for frag in Chem.GetMolFrags(ens.mol):
            heavy = [a for a in frag if ens.mol.GetAtomWithIdx(a).GetAtomicNum() > 1 and a != iso.metal]
            if heavy:
                assert min(float(np.linalg.norm(m - pos[a])) for a in heavy) < 8.0  # vdW contact, not adrift
    assert embedded >= 1


def test_isomer_summary_gives_geometric_identity():
    """iso.summary() returns 'geometry | arrangement | chirality' — the convenient per-isomer identity."""
    import rxembed as rx

    iso = rx.metal("[Zn](F)(Cl)(Br)I", "tetrahedral")[0]
    s = iso.summary()
    assert s.startswith("tetrahedral | ")
    assert iso.chirality in s  # Δ or Λ tetrahedral centre


def test_two_bidentates_no_both_trans_phantom_enumerated():
    """Two bidentate ligands (one short 5-membered) enumerate only cis-cis — never the illogical both-trans.

    A flexible bis-NHC *can* span trans, but the 5-membered amidate can't, so an arrangement placing the
    amidate trans is impossible and must not be enumerated (the span filter drops it).
    """
    from rdkit import Chem

    import rxembed as rx

    smi = "Cc1cc(C)c(N2C=CN3CCN4C=CN(c5c(C)cc(C)cc5C)[C]4->[Ni+2]4(<-[O-]C(=O)C(c5ccccc5)[N-]->4c4ccccc4)<-[C]32)c(C)c1"
    isos = rx.metal(smi, "square_planar")
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(isos[0].mol)) for a in f}
    for iso in isos:  # the short amidate (its two donors ≤4 bonds apart) may never sit at trans vertices
        for i in range(len(iso.vertices)):
            for j in range(i + 1, len(iso.vertices)):
                a, b = iso.vertices[i], iso.vertices[j]
                if M.VACANT in (a, b) or frag[a] != frag[b]:
                    continue
                ang = M._vertex_angle(M.VERTEX_DIRS[iso.geometry][i], M.VERTEX_DIRS[iso.geometry][j])
                dmat = Chem.GetDistanceMatrix(iso.mol)
                if dmat[a][b] <= 4 and ang >= M._SPAN_ANGLE:  # a short chelate forced trans — the phantom
                    raise AssertionError(f"short chelate {a}-{b} enumerated trans ({ang}°)")


def test_en_bischelate_no_trans_phantom_embeds():
    """A trans-spanning en placement is impossible: [Pd(en)Cl2] has 1 embeddable isomer, not a distorted 2nd.

    Regression for the chelate-bite change: freeing the intra-chelate angle must not let an en (which can
    only bite cis) survive embedding at trans vertices as a metal-out-of-plane phantom. The span filter
    pre-drops it; the restored trans-assigned angle is the embed-time backstop.
    """
    import rxembed as rx

    embed_counts = [rx.embed(iso, n=3).minimize().n for iso in rx.metal("Cl[Pd]1(Cl)NCCN1", "square_planar")]
    assert sum(1 for n in embed_counts if n) == 1, f"expected exactly 1 embeddable isomer, got {embed_counts}"


def test_bis_en_octahedral_gives_the_three_real_stereoisomers():
    """[Co(en)2Cl2] enumerates exactly trans / cis-Δ / cis-Λ (chirality-aware), each embeddable."""
    import rxembed as rx

    isos = rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")
    embeddable = [iso for iso in isos if rx.embed(iso, n=2).minimize().n]
    assert {i.chirality for i in embeddable} == {"", "Δ", "Λ"}, [i.chirality for i in embeddable]


@pytest.mark.parametrize(
    ("smi", "geometry", "chirality"),
    [
        ("CCCN[Pd](Cl)(Cl)NCCC", "square_planar", {""}),  # MA2B2 cis/trans — both achiral
        ("N[Co](N)(N)(Cl)(Cl)Cl", "octahedral", {""}),  # mer/fac — both achiral
        ("[Zn](F)(Cl)(Br)I", "tetrahedral", {"Δ", "Λ"}),  # MABCD — a genuine stereocentre
    ],
)
def test_chirality_descriptor_on_textbook_cases(smi, geometry, chirality):
    """The metal-centre Λ/Δ tag is '' for an achiral arrangement and Δ/Λ for a real stereocentre."""
    import rxembed as rx

    isos = rx.metal(smi, geometry)
    assert {i.chirality for i in isos} <= chirality or {i.chirality for i in isos} == chirality


def test_select_is_name_agnostic():
    """select() keys on arrangement / chirality / index, never requiring the cis/trans name."""
    import rxembed as rx

    isos = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")
    by_arr = isos.select(arrangement=M.arrange(isos[0]))
    by_idx = isos.select(index=0)
    assert by_arr.vertices == isos[0].vertices == by_idx.vertices
    with pytest.raises(ValueError, match="matched"):
        isos.select(arrangement="does not exist")


@pytest.mark.parametrize(
    "smi",
    [
        "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",  # side-on alkyne
        "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C",  # C=C
    ],
)
def test_side_on_eta2_embeds_geometry_clean(smi):
    """A side-on η² ligand (π-bonded donor pair) embeds a geom.check-clean conformer.

    Holding the π bond at its natural length (`coordination`) stops the two M-donor pulls stretching the
    C≡C/C=C apart and blowing the sphere out; the flex planarity window (`geometry._eta2_pi_atoms`) tolerates
    the small out-of-plane the side-on binding legitimately imposes on the sp2 atoms.
    """
    import rxembed as rx

    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=4).minimize()
    assert any(geom.check(ens.mol, c).ok() for c in ens.ids), "no geom.check-clean side-on conformer"


def test_eta2_flex_window_does_not_leak_to_non_metal_sp2():
    """The η² planarity flex is metal-side-on-specific: a twisted non-metal sp2 still flags."""
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    m = Chem.AddHs(Chem.MolFromSmiles("C=CC=C"))  # butadiene, no metal -> no η² flex
    rdDistGeom.EmbedMolecule(m, randomSeed=1)
    c = m.GetConformer()
    ci = next(a.GetIdx() for a in m.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    p = c.GetPositions()
    p[ci] = p[ci] + [0.0, 0.0, 0.35]  # shove one sp2 carbon 0.35 A out of plane (past the 0.15 default)
    for i, x in enumerate(p):
        c.SetAtomPosition(i, x.tolist())
    assert any(v.kind == "planarity" for v in geom.planarity(m, c.GetPositions())), "non-metal sp2 wrongly flexed"


def test_heavy_atom_xh_bond_length_is_element_aware():
    """A correct P-H (~1.42 A) passes; C-H stays tight at ~1.3 A ceiling; a grossly long P-H still flags."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    m = Chem.AddHs(Chem.MolFromSmiles("CP"))
    rdDistGeom.EmbedMolecule(m, randomSeed=1)
    c = m.GetConformer()
    h_p = next(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetSymbol() == "P")
    p = m.GetAtomWithIdx(h_p).GetNeighbors()[0].GetIdx()
    pos = c.GetPositions()
    unit = (pos[h_p] - pos[p]) / np.linalg.norm(pos[h_p] - pos[p])
    pos[h_p] = pos[p] + unit * 1.42  # a correct P-H length — must NOT flag (was false-flagged at hi=1.3)
    for i, x in enumerate(pos):
        c.SetAtomPosition(i, x.tolist())
    assert not any(v.kind == "hydrogen" for v in geom.hydrogens(m, c.GetPositions()))
    pos[h_p] = pos[p] + unit * 1.9  # a grossly stretched P-H — must still flag
    for i, x in enumerate(pos):
        c.SetAtomPosition(i, x.tolist())
    assert any(v.kind == "hydrogen" for v in geom.hydrogens(m, c.GetPositions()))


def test_henry_geometry_gate_no_false_clash_on_chelate_bite():
    """The clash gate does not flag two cis donors of one metal (a bite ~2 Å) as a steric overlap."""
    import rxembed as rx

    iso = rx.metal(HENRY[2], "square_planar")[0]  # the O,N + diphosphine case (clean donors)
    ens = rx.embed(iso, n=3).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        clashes = [v for v in rep.violations if v.kind == "clash"]
        assert not clashes, rep.summary()


def test_reembed_retry_delivers_clean_geometry():
    """A metal whose default seed the relax tears still yields a geom.check-clean geometry: `minimize`
    re-embeds fresh seeds until N are clean. The diphosphine on a rigid fused diene tears ~half its seeds,
    so the single-seed (henry-loop) call used to render a torn geometry; now it is clean, and n=N is all-clean.
    """
    import rxembed as rx

    smi = (
        "CCC1=C2CCCCC2=C(CC)[P](c2ccccc2)(c2ccccc2)->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)"
        "[N-]->2c2ccccc2)<-[P]1(c1ccccc1)c1ccccc1"
    )
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=1).minimize()  # the henry single-seed loop — must be clean, not a torn default seed
    assert ens.n >= 1
    assert geom.check(ens.mol, ens.ids[0]).ok(), geom.check(ens.mol, ens.ids[0]).summary()
    big = rx.embed(iso, n=6).minimize()  # embed(n=N) -> N GOOD geometries, every one clean
    assert big.n >= 6
    assert all(geom.check(big.mol, c).ok() for c in big.ids)


def test_soft_donor_distance_is_capped():
    """A P donor gets a realistic M-P (~2.1-2.3 A), not the raw covalent sum (~2.4) that fights a rigid
    backbone into a tear; a HALIDE donor is X-type (not dative) and must NOT be capped — its M-X sits at the
    covalent sum (capping to ~2.24 would be too short and manufacture tears); O/N/C stay at the covalent sum.
    """
    from rdkit.Chem import rdMolTransforms as T

    import rxembed as rx

    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    ens = rx.embed(iso, n=3).minimize()
    assert ens.n >= 1
    c = ens.mol.GetConformer(ens.ids[0])
    for d in iso.donors:
        z = ens.mol.GetAtomWithIdx(d).GetAtomicNum()
        dist = T.GetBondLength(c, iso.metal, d)
        if z == 15:  # phosphorus — soft dative, capped well under the ~2.4 raw covalent sum
            assert 2.0 < dist < 2.35, f"Ni-P {dist:.2f} outside the capped/realistic window"
        else:  # O / N / C — below the cap, so unchanged and realistic
            assert 1.8 < dist < 2.25, f"Ni-donor(Z={z}) {dist:.2f} off"
    pdcl = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]  # a halide donor must keep the covalent sum
    e2 = rx.embed(pdcl, n=3).minimize()
    assert e2.n >= 1
    c2 = e2.mol.GetConformer(e2.ids[0])
    cl = [d for d in pdcl.donors if e2.mol.GetAtomWithIdx(d).GetAtomicNum() == 17]
    assert cl, "no Cl donor perceived"
    assert all(T.GetBondLength(c2, pdcl.metal, d) > 2.35 for d in cl), "Pd-Cl wrongly capped short"


def test_mc_does_not_fold_conjugated_donor_into_metal():
    """openconf's rotor search must not swing a conjugated N/O donor's rigid plane into the metal.

    The amidate O-C(=O) carboxylate can rotate about its single bond, but its M-O-C donation angle is held
    (~120°, never acute), so the carbonyl carbon stays out of the coordination sphere (>~2.5 Å from the
    metal) through the whole embed -> minimize -> mc -> minimize chain. Regression for the mc fold-in.
    """
    import numpy as np

    from tests.conftest import _openconf_available

    if not _openconf_available():
        import pytest

        pytest.skip("openconf not installed")
    import rxembed as rx

    smi = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(smi, "square_planar")[0]
    # the carboxylate carbon: the carbon bonded to both the coordinating O (donor) and the =O
    o_don = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetAtomicNum() == 8)
    c_carboxyl = next(n.GetIdx() for n in iso.mol.GetAtomWithIdx(o_don).GetNeighbors() if n.GetAtomicNum() == 6)
    searched = rx.embed(iso, n=8).minimize().mc(preset="ensemble").minimize()
    assert searched.n >= 1
    for cid in searched.ids:  # the non-donor carboxyl carbon never intrudes into the coordination sphere
        pos = searched.mol.GetConformer(cid).GetPositions()
        d_mc = float(np.linalg.norm(pos[iso.metal] - pos[c_carboxyl]))
        assert d_mc > 2.4, f"carboxyl C folded to {d_mc:.2f} Å from the metal (conjugated donor swung in)"
