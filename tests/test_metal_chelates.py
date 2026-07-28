"""Metal bis-chelate embedding + the name-agnostic geometric identity (arrangement / chirality). RDKit + UFF."""

import pytest

from rxembed import geometry as geom
from rxembed.rdkit_embed import coordination as coord
from rxembed.rdkit_embed.constraints import distance as D  # noqa: N812
from rxembed.rdkit_embed.constraints import metal as M  # noqa: N812

# Henry Ni(II) bis-chelate tight-bite cases: side-on C=P, direct N=C, O,N amidate + diphosphine/malonate.
HENRY = [
    "Cc1cc(C)c([CH]2=[PH]->[Ni+2]<-23<-[O-]C(=O)C(c2ccccc2)[N-]->3c2ccccc2)c(C)c1",
    "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    "O=C1[O-]->[Ni+2]2(<-[N-](c3ccccc3)C1c1ccccc1)<-[P](Cc1ccccc1[P]->2(C1CCCCC1)C1CCCCC1)(C1CCCCC1)C1CCCCC1",
    "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C",
]
_TWO_H = 2  # a primary amine donor carries two protons


@pytest.mark.parametrize("smi", HENRY)
def test_henry_bischelate_embeds_with_proper_coordination(smi):
    """At least one enumerated isomer embeds with sane dative M-donor distances (the complex is embeddable)."""
    from rdkit.Chem import rdMolTransforms as T

    import rxembed as rx

    # stereo="free": these amidates have an alpha-C the racemic default would enumerate; this checks embeddability
    isos = rx.metal(smi, "square_planar", stereo="free")
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

    def frag_of(iso, v):
        """Fragment of a vertex — a haptic face's centroid is a dummy, so resolve it via its own ring atoms."""
        return frag[iso.haptic[v][0]] if v in iso.haptic else frag[v]

    for iso in isos:  # no enumerated isomer may place a same-ligand donor pair trans (span filter removed those)
        for i in range(len(iso.vertices)):
            for j in range(i + 1, len(iso.vertices)):
                a, b = iso.vertices[i], iso.vertices[j]
                if M.VACANT not in (a, b) and frag_of(iso, a) == frag_of(iso, b):
                    vd = M.POLYHEDRA[iso.geometry].vertex_dirs
                    ang = M._vertex_angle(vd[i], vd[j])
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
    """A metal complex embedded with a separate (`.`) fragment tethers it at vdW contact, not infinity."""
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
    """A short 5-membered amidate never enumerates trans, even alongside a flexible bis-NHC that can span trans."""
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
                vd = M.POLYHEDRA[iso.geometry].vertex_dirs
                ang = M._vertex_angle(vd[i], vd[j])
                dmat = Chem.GetDistanceMatrix(iso.mol)
                if dmat[a][b] <= 4 and ang >= M._SPAN_ANGLE:  # a short chelate forced trans — the phantom
                    raise AssertionError(f"short chelate {a}-{b} enumerated trans ({ang}°)")


def test_side_on_eta2_donors_are_not_screened_for_end_on_donation():
    """An η² face has no lone-pair axis, so the trans-span *orientation* screen must abstain, not enumerate zero.

    IYIJAE (Rh-NHC-CO-COD): the η² alkene carbons of 1,5-cyclooctadiene are co-donors of the same metal, which
    donate their π face side-on — the metal sits ~70° off any M-C-X axis by construction. Screening them as if
    they donated end-on rejected every arrangement and made the documented `rx.metal(...)[0]` raise IndexError.
    """
    import rxembed as rx

    smi = (
        "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)[c-]1->[Rh+]123(<-[C-]#[O+])"
        "<-[CH]4=[CH]->1CC[CH]->2=[CH]->3CC4"
    )
    assert len(rx.metal(smi)) > 0, "an η² donor has no donation axis — the orientation screen must abstain"


def test_an_untabulated_geometrys_only_ordering_is_never_filtered_away():
    """CN7 has no permutation table, so the identity ordering is the only candidate — filtering it returns none.

    Both pre-filters exist to *prefer* feasible arrangements over infeasible ones within an enumeration; with a
    single un-enumerated ordering there is nothing to prefer it over, so `isomers` returns it BEFORE either
    filter runs (else the branch's own "the geometry still embeds" contract breaks and `rx.metal(...)[0]` raises
    IndexError). A homoleptic MoCl7 is the cleanest CN7 — pentagonal-bipyramidal, one ordering, no haptic face.
    """
    import rxembed as rx

    isos = rx.metal("Cl[Mo](Cl)(Cl)(Cl)(Cl)(Cl)Cl")
    assert len(isos) > 0, "the single identity ordering of an untabulated geometry must survive enumeration"
    assert isos[0].geometry == "pentagonal_bipyramidal"


def test_eta2_pyridine_arene_collapses_a_cn7_miscount_to_octahedral():
    """KASFIU (W-Tp-NO-PMe3-eta2pyridine): the eta2 C=C is ONE vertex, not two sigma donors, so the sphere is CN6.

    Its two adjacent pyridine carbons both bind W at ~2.3 A (crystal) — a genuine side-on eta2. Counting them as
    two separate sigma donors gave CN7 (a pentagonal-bipyramidal miscount); collapsing the face to one centroid
    vertex gives the physically-right octahedron, which embeds geometry-clean where the miscount did not.
    """
    import rxembed as rx
    from rxembed import geometry as geom

    smi = "CN(C)c1ncc[cH]2->[W+2]34(<-[N-]=O)(<-[cH]12)(<-[n]1cccn1[BH-](n1ccc[n]->31)n1ccc[n]->41)<-[P](C)(C)C"
    isos = rx.metal(smi)
    assert isos, "the eta2 complex must enumerate at least one isomer"
    assert all(iso.geometry == "octahedral" for iso in isos)  # CN6, not the CN7 eta2-as-2sigma miscount
    assert all(len(iso.haptic) == 1 for iso in isos)  # exactly one eta2 face collapsed to a centroid vertex
    ens = rx.embed(isos[0], n=4).minimize()
    assert any(geom.check(ens.mol, c).ok() for c in ens.ids), "no geom.check-clean eta2-pyridine conformer"


def test_en_bischelate_no_trans_phantom_embeds():
    """A trans-spanning en placement is impossible: [Pd(en)Cl2] has 1 embeddable isomer, not a distorted 2nd."""
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
    """A side-on η² ligand (π-bonded donor pair) embeds a geom.check-clean conformer."""
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
    """`minimize` re-embeds fresh seeds until N are geom.check-clean (diphosphine on a rigid diene tears ~half)."""
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
    """A P donor gets a realistic capped M-P (~2.1-2.3 A); an X-type halide keeps the uncapped covalent sum."""
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
        else:  # O / N / C — below the cap, so unchanged (a mild anionic-O shortfall is the accepted model cost)
            assert 1.8 < dist < 2.25, f"Ni-donor(Z={z}) {dist:.2f} off"
    pdcl = rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0]  # a halide donor must keep the covalent sum
    e2 = rx.embed(pdcl, n=3).minimize()
    assert e2.n >= 1
    c2 = e2.mol.GetConformer(e2.ids[0])
    cl = [d for d in pdcl.donors if e2.mol.GetAtomWithIdx(d).GetAtomicNum() == 17]
    assert cl, "no Cl donor perceived"
    assert all(T.GetBondLength(c2, pdcl.metal, d) > 2.35 for d in cl), "Pd-Cl wrongly capped short"


def test_mc_does_not_fold_conjugated_donor_into_metal():
    """openconf's rotor search must not swing a conjugated N/O donor's carbon into a bond with the metal."""
    import numpy as np
    from rdkit.Chem import GetPeriodicTable

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
    donors = sorted({int(d) for ds in searched.sphere.values() for d in ds})
    pt = GetPeriodicTable()
    r_sum = pt.GetRcovalent(searched.mol.GetAtomWithIdx(iso.metal).GetAtomicNum()) + pt.GetRcovalent(6)
    for cid in searched.ids:  # the non-donor carboxyl carbon never reaches a bonding distance
        pos = searched.mol.GetConformer(cid).GetPositions()
        d_mc = float(np.linalg.norm(pos[iso.metal] - pos[c_carboxyl]))
        assert d_mc / r_sum >= D.NEAR_REPORT_RATIO, f"carboxyl C collapsed to a bond at {d_mc:.2f} Å from the metal"
        assert not coord.metal_overbond(searched.mol, pos, donors), "carboxyl C over-bonded the metal in the mc search"


def test_vacant_coordination_site_is_not_filled_by_the_ligands_own_backbone():
    """A vacant vertex stays vacant: the tiered floor (`distance.nondonor_floors`) holds a non-donor backbone out."""
    import numpy as np
    from rdkit.Chem import GetPeriodicTable

    import rxembed as rx
    from rxembed import metrics

    pt = GetPeriodicTable()
    for iso in rx.metal("CCN[Pd](Cl)Cl", "square_planar"):
        ens = rx.embed(iso, n=4, seed=1).minimize()
        assert ens.n >= 1
        r_m = pt.GetRcovalent(ens.mol.GetAtomWithIdx(iso.metal).GetAtomicNum())
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            for a in ens.mol.GetAtoms():
                i = a.GetIdx()
                if i == iso.metal or i in iso.donors or a.GetAtomicNum() == 1:
                    continue
                d = float(np.linalg.norm(pos[i] - pos[iso.metal]))
                r_sum = r_m + pt.GetRcovalent(a.GetAtomicNum())
                assert d / r_sum >= D.NEAR_REPORT_RATIO, (
                    f"{iso.label}/conf{cid}: non-donor {a.GetSymbol()}{i} collapsed into the vacant site "
                    f"({d:.3f} Å = {d / r_sum:.3f} x the covalent sum)"
                )
            assert not coord.metal_overbond(ens.mol, pos, iso.donors)
            assert metrics.coordination_changed(ens.mol, cid, iso.metal, iso.donors) == ([], [])


@pytest.mark.parametrize(
    ("smi", "held"),
    [
        ("P->[Pd](Cl)Cl", True),  # a primary phosphine — P sp3 is a calibrated class
        ("CCN[Pd](Cl)Cl", True),  # amine N — sp3 calibrated: its proton is now walled too (the sp3-amine fix)
        ("[NH3]->[Pd](Cl)Cl", True),  # ammine N — sp3 calibrated, likewise
        ("O->[Pd](Cl)Cl", False),  # aqua O — sp3 UNcalibrated (n < 6): the wall abstains, as the fold gate does
    ],
)
def test_a_calibrated_donors_protons_are_walled_and_an_uncalibrated_class_abstains(smi, held):
    """An (M, donor, H) orientation wall exists iff the donor's (element, hyb) class is census-calibrated.

    The unified `_orient_donor` walls EVERY substituent (heavy AND proton) of a calibrated donor at its census
    window, so an amine / ammine N and a phosphine P all get a proton wall — walling the proton is the fix for an
    sp3 amine that used to fold an H onto the metal over its lone pair (no rule walled an N/O/C sp3 proton before).
    An uncalibrated class (aqua O sp3, n < 6) gets no window and abstains, exactly as `donor_orientation` does.
    """
    import rxembed as rx

    iso = rx.metal(smi, "square_planar").select(index=0)
    proton_windows = [
        k for k in iso.cons.angles if k[0] == iso.metal and iso.mol.GetAtomWithIdx(k[2]).GetAtomicNum() == 1
    ]
    assert bool(proton_windows) == held, (
        f"{smi}: proton walls {'expected but absent' if held else 'present but should be gone'} "
        f"({len(proton_windows)} found)"
    )


def test_a_slow_inverting_phosphines_protons_stay_splayed_on_both_paths(tmp_path):
    """A primary phosphine's protons stay splayed away from the metal on both the SMILES and the from-xyz path."""
    import numpy as np
    from rdkit.Chem import rdMolTransforms as T

    import rxembed as rx

    def m_d_h(ens, metal, donors):
        return [
            T.GetAngleDeg(ens.mol.GetConformer(cid), int(metal), int(d), h.GetIdx())
            for d in donors
            for h in ens.mol.GetAtomWithIdx(int(d)).GetNeighbors()
            if h.GetAtomicNum() == 1
            for cid in ens.ids
        ]

    smi = "P->[Pd](Cl)Cl"
    iso = rx.metal(smi, "square_planar").select(index=0)
    ens = rx.embed(iso, n=6, seed=3).minimize()
    assert ens.n >= 1
    angs = m_d_h(ens, iso.metal, iso.donors)
    assert angs, f"{smi}: no phosphine protons to check — the fixture is wrong"
    assert min(angs) > 90.0, f"{smi} (SMILES): phosphine proton folded to {min(angs):.1f} deg of the metal"

    xyz = tmp_path / "seed.xyz"  # ...and now feed that geometry back in through the from-geometry path
    ens.lowest(1).dump(str(xyz))
    iso2 = rx.metal(str(xyz), "square_planar", center="Pd").select(index=0)
    ens2 = rx.embed(iso2, n=6, seed=3).minimize()
    assert ens2.n >= 1
    angs2 = m_d_h(ens2, iso2.metal, iso2.donors)
    assert min(angs2) > 90.0, f"{smi} (from geometry): phosphine proton folded to {min(angs2):.1f} deg of the metal"
    assert abs(float(np.median(angs2)) - 109.5) < 20.0, (
        f"{smi} (from geometry): M-P-H median {np.median(angs2):.1f} deg is not a hybridisation angle"
    )


def test_an_sp3_amine_donor_does_not_fold_a_proton_onto_the_metal():
    """A metal-bound sp3 [NH2] amine keeps its protons off the metal — the sp3-amine fix, inside `_orient_donor`.

    RED-first: with the fold wall heavy-only (an N sp3 proton walled by NO rule) the amine N folds a proton onto
    the metal over its lone pair — min M-N-H ~65 deg, ~3-5/33 conformers fully inverted, and the fold gate is
    heavy-substituent so it ships gate-clean (measured, seeds 1-8, minimized). The unified wall holds the proton
    at the census ('N',SP3) window, lifting min M-N-H to ~94 deg with zero inversions. This is the embed->minimize
    (UFF, no g-xTB) degrade path: no calculator re-splays the fold, so the seed hold must.
    """
    from rdkit.Chem import rdMolTransforms as T

    import rxembed as rx

    smi = "CCNC1N[NH2]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[S]=1"
    iso0 = rx.metal(smi, "square_planar")[0]
    n5 = next(  # the primary [NH2] amine donor: an sp3 N carrying two protons
        d
        for d in iso0.donors
        if iso0.mol.GetAtomWithIdx(d).GetSymbol() == "N"
        and sum(1 for nb in iso0.mol.GetAtomWithIdx(d).GetNeighbors() if nb.GetAtomicNum() == 1) >= _TWO_H
    )
    hs = [nb.GetIdx() for nb in iso0.mol.GetAtomWithIdx(n5).GetNeighbors() if nb.GetAtomicNum() == 1]
    assert len(hs) >= _TWO_H, "the fixture's amine donor must carry two protons for this to mean anything"

    angs = []
    for seed in range(1, 9):
        iso = rx.metal(smi, "square_planar")[0]
        ens = rx.embed(iso, n=2, seed=seed).minimize()
        for cid in ens.ids:
            conf = ens.mol.GetConformer(cid)
            angs += [T.GetAngleDeg(conf, int(iso.metal), int(n5), int(h)) for h in hs]
    assert angs, "no amine protons measured — the embed produced nothing"
    assert min(angs) >= 90.0, f"an sp3 amine proton folded onto the metal (min M-N-H {min(angs):.1f} deg)"


def test_representatives_on_metal_complex_does_not_raise():
    """The dedup descriptor path (representatives/landscape/cluster) must survive the metal restore.

    Regression (found by examples/07_metal + henry): `mc()`'s `disconnect_metal` clears RingInfo (RemoveBond),
    and the follow-up `minimize`'s `connect_metal` re-added the DATIVE M-donor bonds without re-perceiving it, so
    the restored graph reached `rotatable_quads`' ring-aware SMARTS with RingInfo uninitialised -> RuntimeError.
    """
    import rxembed as rx

    iso = rx.metal("Br[Pd]1(Cl)<-NCC<-N1", "square_planar").select()  # Pd(II) ethylenediamine chelate
    ens = rx.embed(iso, n=4, seed=1).minimize()
    ens._mol = M.disconnect_metal(ens._mol)  # reproduce mc()'s effect (RemoveBond clears RingInfo) sans openconf
    ens._minimized = False  # so representatives()'s minimize() re-connects via connect_metal
    ens.minimize()

    reps = ens.representatives()  # was: RuntimeError "RingInfo not initialized"
    assert reps.n >= 1
    assert len(ens.cluster()) == ens.n  # cluster() shares the descriptor latent

    from rxembed.dedup.descriptors import rotatable_quads

    # the en backbone C-C sits in the ring closed THROUGH the metal, so it is not a rotatable dihedral axis
    assert rotatable_quads(ens.mol) == []
