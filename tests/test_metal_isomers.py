"""`metal_isomers`: the three doors onto `Isomer` and the distinct-arrangement enumeration.

`Isomer(mol, geometry, sites)` seats a known arrangement, `from_geometry` retains the input's own, and
`enumerate_isomers` produces the unknown ones as an `IsomerSet`. All three must agree on the same identity;
a name-agnostic arrangement plus a handedness tag, and none may enumerate an arrangement the ligands
cannot reach (a short chelate forced trans). An η≥2 face is one vertex, held by a transient centroid dummy
that lives in no stored Mol. RDKit + UFF surrogate, no xtb.
"""

from __future__ import annotations

import itertools
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed.pipeline as rx
from rxembed import metal_core as M  # noqa: N812
from rxembed import metal_isomers as K  # noqa: N812
from rxembed import metal_polyhedron as P  # noqa: N812
from rxembed.pipeline import geom_check as geom

_MA2B2 = "CCCN[Pd](Cl)(Cl)NCCC"  # square-planar MA2B2 -> the cis / trans pair
_PT_A2B2 = "[NH3]->[Pt](<-[NH3])(Cl)Cl"  # donors, in atom order: N0 N2 Cl3 Cl4
S, D, DAT = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.DATIVE


def _assert_clean(ens):
    """Every conformer of a minimized ensemble passes the metal-aware geometry gate."""
    ens = ens.minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert rep.ok(), rep.summary()


def _pt():
    return Chem.AddHs(Chem.MolFromSmiles(_PT_A2B2))


def _frag_of(iso, frag, v):
    """Fragment index of a vertex: a haptic face's centroid is a dummy, so resolve it via its ring atoms."""
    return frag[iso.haptic[v][0]] if v in iso.haptic else frag[v]


def _same_ligand_vertex_angles(isos):
    """Every (isomer, vertex-separation angle) for a donor pair of one ligand, across an enumerated set."""
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(isos[0].mol)) for a in f}
    out = []
    for iso in isos:
        dirs = P.POLYHEDRA[iso.geometry].vertex_dirs
        for vi, vj in itertools.combinations(range(len(iso.vertices)), 2):
            a, b = iso.vertices[vi], iso.vertices[vj]
            if M.VACANT in (a, b) or _frag_of(iso, frag, a) != _frag_of(iso, frag, b):
                continue
            out.append((iso, a, b, P._vertex_angle(dirs[vi], dirs[vj])))
    return out


def _angle(pos, i, j, k):
    a, b = pos[i] - pos[j], pos[k] - pos[j]
    return float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))


# --- sandwich / piano-stool builders (test_mol_state.py imports these) ----------------------------------


def _sandwich(ring_n, metal_z, metal_q, kekule, anion, zoff, rad):
    """A bis(η-n) sandwich Mol with a seed geometry: metal + two n-membered carbocycles, dative M<-C."""
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(metal_z))
    rw.GetAtomWithIdx(me).SetFormalCharge(metal_q)
    rings = []
    for _r in range(2):
        cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(ring_n)]
        for k in range(ring_n):
            rw.AddBond(cs[k], cs[(k + 1) % ring_n], kekule[k])
        if anion:
            rw.GetAtomWithIdx(cs[0]).SetFormalCharge(-1)  # Cp-: the ring's delocalised -1 on one carbon
        for c in cs:
            rw.AddBond(me, c, DAT)  # dative M<-C so each ring keeps its valence when stripped
        rings.append(cs)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for ri, cs in enumerate(rings):
        z = zoff if ri == 0 else -zoff
        for k, c in enumerate(cs):
            ang = 2 * np.pi * k / ring_n + (0.2 if ri else 0.0)  # stagger the second ring
            conf.SetAtomPosition(c, Point3D(rad * np.cos(ang), rad * np.sin(ang), z))
    m.AddConformer(conf)
    return m


def ferrocene():
    """Ferrocene: two Cp- rings on Fe(II), an η5 sandwich (M-C ~2.05 Å)."""
    return _sandwich(5, 26, 2, [S, D, S, D, S], anion=True, zoff=1.66, rad=1.21)


def dibenzenechromium():
    """Bis(benzene)chromium: two benzene rings on Cr(0), an η6 sandwich (M-C ~2.13 Å)."""
    return _sandwich(6, 24, 0, [S, D, S, D, S, D], anion=False, zoff=1.61, rad=1.40)


def cp_ticl3():
    """CpTiCl3: a Cp- ring + three chlorides on Ti(IV): a C3v piano stool (one face + 3 sigma donors, CN4)."""
    rw = Chem.RWMol()
    ti = rw.AddAtom(Chem.Atom(22))
    rw.GetAtomWithIdx(ti).SetFormalCharge(4)
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    for k, b in enumerate([S, D, S, D, S]):
        rw.AddBond(cs[k], cs[(k + 1) % 5], b)
    rw.GetAtomWithIdx(cs[0]).SetFormalCharge(-1)
    for c in cs:
        rw.AddBond(ti, c, DAT)
    cls = [rw.AddAtom(Chem.Atom(17)) for _ in range(3)]
    for cl in cls:
        rw.AddBond(ti, cl, S)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(ti, Point3D(0, 0, 0))
    for k, c in enumerate(cs):
        a = 2 * np.pi * k / 5
        conf.SetAtomPosition(c, Point3D(1.2 * np.cos(a), 1.2 * np.sin(a), 1.9))  # ring up
    for k, cl in enumerate(cls):
        a = 2 * np.pi * k / 3
        conf.SetAtomPosition(cl, Point3D(1.9 * np.cos(a), 1.9 * np.sin(a), -1.3))  # tripod down
    m.AddConformer(conf)
    return m


def tetrahedral_four_distinct():
    """Zn(II) with four distinct monodentate dative donors (N, O, S, P) -> tetrahedral, two enantiomers.

    All six vertex pairs share 109.47°, so the pairwise element/angle signature is identical for both hands;
    only the centre's handedness separates them, which is what makes this the genuine un-enumerated limitation.
    """
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(30))
    rw.GetAtomWithIdx(me).SetFormalCharge(2)
    donors = []
    for z, n_h in [(7, 3), (8, 2), (16, 2), (15, 3)]:  # NH3, OH2, SH2, PH3 (dative M<-donor)
        d = rw.AddAtom(Chem.Atom(z))
        for _ in range(n_h):
            rw.AddBond(d, rw.AddAtom(Chem.Atom(1)), S)
        rw.AddBond(me, d, DAT)
        donors.append(d)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for d, v in zip(donors, [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], strict=True):
        conf.SetAtomPosition(d, Point3D(*(np.array(v, float) / np.sqrt(3) * 2.1)))
    for a in m.GetAtoms():  # splay each donor's H's off it deterministically
        if a.GetAtomicNum() == 1:
            base = np.array(conf.GetAtomPosition(a.GetNeighbors()[0].GetIdx()))
            conf.SetAtomPosition(a.GetIdx(), Point3D(*(base * 1.4 + np.random.RandomState(a.GetIdx()).randn(3) * 0.3)))
    m.AddConformer(conf)
    return m


# --- enumeration: the textbook isomers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "geometry", "labels"),
    [
        (_MA2B2, "square_planar", {"cis", "trans"}),  # MA2B2
        ("[NH3][Co]([NH3])([NH3])(Cl)(Cl)Cl", "octahedral", {"mer", "fac"}),  # MA3B3
    ],
)
def test_enumeration_gives_the_distinct_isomers_and_each_embeds_clean(smiles, geometry, labels):
    """The symmetry-reduced isomers come back tagged, and every one embeds to a metal-aware-clean geometry."""
    cands = rx.embed(smiles, metal=geometry, n=4)
    assert isinstance(cands, rx.EnsembleSet)
    assert {e.tag["label"] for e in cands} == labels
    for e in cands:
        _assert_clean(e)


@pytest.mark.parametrize(
    ("smi", "geometry", "chirality"),
    [
        (_MA2B2, "square_planar", {""}),  # MA2B2 cis/trans: both achiral
        # MABCD is a genuine chiral centre, but tetrahedral has no permutation table, so only the input
        # ordering is returned: the un-enumerated partner is what the permutation info-log below announces.
        ("[Zn](F)(Cl)(Br)I", "tetrahedral", {"delta"}),
    ],
)
def test_the_chirality_tag_is_empty_for_an_achiral_arrangement(smi, geometry, chirality):
    """The handedness tag is '' for an achiral arrangement and a word for a real stereocentre."""
    assert {i.chirality for i in rx.metal(smi, geometry)} == chirality


def test_bis_en_octahedral_gives_the_three_real_stereoisomers():
    """[Co(en)2Cl2] enumerates exactly trans / cis-delta / cis-lambda (chirality-aware), each embeddable."""
    isos = rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")
    embeddable = [iso for iso in isos if rx.embed(iso, n=2).minimize().n]
    assert {i.chirality for i in embeddable} == {"", "delta", "lambda"}, [i.chirality for i in embeddable]


def test_select_is_name_agnostic_and_accepts_a_code_or_a_word():
    """`select()` keys on arrangement / chirality / index: never on the cis/trans name."""
    isos = rx.metal(_MA2B2, "square_planar")
    assert isos.select(arrangement=K.arrangement(isos[0])).vertices == isos[0].vertices == isos.select(index=0).vertices
    with pytest.raises(ValueError, match="matched"):
        isos.select(arrangement="does not exist")
    tet = rx.metal("[Zn](F)(Cl)(Br)I", "TET")  # the code is an input alias, not the identity
    assert {i.geometry for i in tet} == {"tetrahedral"}
    assert tet.select(chirality="delta") is tet.select(chirality="Δ")


# --- enumeration: arrangements the ligands cannot reach -------------------------------------------------


@pytest.mark.parametrize(
    "smi",
    [
        # a short-backbone bis-chelate (5-ring amidate + direct N=C)
        "CC(C)(C)[N]1=[CH](Cc2ccccc2)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
        # the same short amidate alongside a FLEXIBLE bis-NHC that genuinely can span trans
        "Cc1cc(C)c(N2C=CN3CCN4C=CN(c5c(C)cc(C)cc5C)[C]4->[Ni+2]4(<-[O-]C(=O)C(c5ccccc5)[N-]->4c4ccccc4)<-[C]32)c(C)c1",
    ],
    ids=["bis-chelate", "amidate+flexible-NHC"],
)
def test_a_short_chelate_is_never_enumerated_trans(smi):
    """No enumerated isomer places a same-ligand donor pair at a trans-span vertex separation.

    The span filter reads the ligand backbone's own reach, so a flexible co-ligand that CAN span trans does not
    license the short one beside it.
    """
    for iso, a, b, ang in _same_ligand_vertex_angles(rx.metal(smi, "square_planar")):
        dmat = Chem.GetDistanceMatrix(iso.mol)
        if dmat[a][b] <= 4:  # a short chelate: its two donors are at most 4 bonds apart
            assert ang < M._SPAN_ANGLE, f"short chelate {a}-{b} enumerated trans ({ang}°)"


def test_an_untabulated_geometrys_only_ordering_is_never_filtered_away():
    """CN7 has no permutation table, so the identity ordering is returned before either pre-filter runs.

    Both filters exist to *prefer* feasible arrangements within an enumeration; with a single ordering there is
    nothing to prefer it over, and filtering it would make the documented `rx.metal(...)[0]` raise IndexError.
    """
    isos = rx.metal("Cl[Mo](Cl)(Cl)(Cl)(Cl)(Cl)Cl")  # homoleptic MoCl7: one ordering, no haptic face
    assert len(isos) > 0
    assert isos[0].geometry == "pentagonal_bipyramidal"


def test_a_side_on_eta2_donor_pair_does_not_fail_the_orientation_screen():
    """An η² face has no lone-pair axis, so the trans-span orientation screen abstains rather than enumerate zero.

    IYIJAE (Rh-NHC-CO-COD): the η² alkene carbons donate their π face side-on, so the metal sits ~70° off any
    M-C-X axis by construction. Screening them as end-on donors rejected every arrangement and made the
    documented `rx.metal(...)[0]` raise IndexError.
    """
    smi = (
        "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)[c-]1->[Rh+]123(<-[C-]#[O+])"
        "<-[CH]4=[CH]->1CC[CH]->2=[CH]->3CC4"
    )
    assert len(rx.metal(smi)) > 0


def test_the_t_shape_seats_its_trans_pair_first():
    """[RhCl(PR3)2] puts the two P on t_shape's TRANS vertices (0, 2) and Cl on the stem.

    Realising the angles proves the polytope; only the donor identities prove the SEATING, and a P-at-stem
    arrangement passes every angle row identically.
    """
    iso = rx.metal("CP(C)(C)->[Rh](Cl)<-P(C)(C)C", "t_shape").select(index=0)
    seated = [iso.mol.GetAtomWithIdx(v).GetSymbol() for v in iso.vertices]
    assert (seated[0], seated[2]) == ("P", "P"), f"trans vertices got {seated}"
    assert seated[1] == "Cl"


# --- the "no permutations tabulated" info-log: only when an arrangement is really lost -------------------

_PERM_WARN = "no isomer permutations tabulated"  # the stable substring of the guard's info-log


def test_an_untabulated_geometry_with_one_arrangement_stays_silent(caplog):
    """A ferrocene collapses to two centroid sites -> linear: one arrangement, so the warning would be noise."""
    with caplog.at_level("INFO", logger="rxembed.metal"):
        isos = rx.metal(ferrocene())
    assert isos, "the single ordering must still come back"
    assert isos[0].geometry == "linear"
    assert not any(_PERM_WARN in r.message for r in caplog.records), caplog.text


def test_four_distinct_tetrahedral_donors_warn(caplog):
    """Four distinct donors on a tetrahedron are two enantiomers we do not expand: the user must hear it."""
    with caplog.at_level("INFO", logger="rxembed.metal"):
        isos = rx.metal(tetrahedral_four_distinct(), "tetrahedral")
    assert isos, "the single ordering must still come back"
    assert isos[0].geometry == "tetrahedral"
    assert any(_PERM_WARN in r.message for r in caplog.records), caplog.text


# --- Isomer(mol, geometry, sites): the known-isomer front door -------------------------------------------


def test_a_known_isomer_seats_real_atom_indices():
    """`sites` names donor ATOMS (what perception has); which vertices they share decides cis vs trans."""
    mol = _pt()
    cis = K.Isomer(mol, "SPL", {0: 0, 1: 2, 2: 3, 3: 4})  # the two N on adjacent vertices
    trans = K.Isomer(mol, "square_planar", [0, 3, 2, 4])  # ...and across (a list is vertex-ordered)
    assert (cis.label, trans.label) == ("cis", "trans")
    assert cis.geometry == trans.geometry == "square_planar"  # the code is an input alias, not the identity
    assert cis.coordination() is cis.cons
    _assert_clean(rx.embed(trans, n=2, seed=1))  # ...and a hand-seated Isomer embeds


def test_a_known_isomer_leaves_a_pocket_when_the_geometry_is_bigger():
    """Four donors seated in a 6-vertex shell leave two VACANT vertices, not a re-counted CN4."""
    iso = K.Isomer(_pt(), "OCT", {0: 0, 1: 2, 2: 3, 3: 4})
    assert iso.vertices.count(M.VACANT) == 2


@pytest.mark.parametrize(
    ("sites", "match"),
    [
        ({0: 0}, "given no vertex"),  # an unseated donor would silently embed a different isomer
        ({0: 0, 1: 0, 2: 3, 3: 4}, "seated at two vertices"),
        ({0: 0, 1: 2, 2: 3, 9: 4}, "not one of this geometry"),
        ({0: 5, 1: 2, 2: 3, 3: 4}, "not a donor"),  # atom 5 is an ammine H
    ],
)
def test_a_malformed_sites_map_is_rejected_loudly(sites, match):
    """Every way of mis-seating a donor raises with the reason, rather than embedding a different isomer."""
    with pytest.raises(ValueError, match=match):
        K.Isomer(_pt(), "SPL", sites)


def test_an_isomer_source_and_metal_are_mutually_exclusive():
    """`rx.embed(iso, metal=...)` refuses: the Isomer already fixes the arrangement `metal=` would enumerate."""
    iso = next(iter(rx.metal(_MA2B2, "square_planar")))
    with pytest.raises(ValueError, match="Isomer source OR metal"):
        rx.embed(iso, metal="square_planar")


# --- the template graft over a coordination sphere -------------------------------------------------------


def _cis_reference(n=2):
    """Embed the MA2B2 pair and return `(cis ensemble, its positions, {symbol: [idx]})`."""
    pair = rx.embed(_MA2B2, metal="square_planar", n=n)
    cis = next(e for e in pair if e.tag["label"] == "cis")
    where: dict = {}
    for a in cis.mol.GetAtoms():
        where.setdefault(a.GetSymbol(), []).append(a.GetIdx())
    return cis, cis.mol.GetConformer(cis.ids[0]).GetPositions(), where


def test_a_template_off_the_sphere_grafts_exactly_and_leaves_the_arrangement_to_the_polyhedron():
    """A core pinning no coordination pair grafts at 0.000 Å while each enumerated arrangement stays itself."""
    ref, pos, where = _cis_reference()
    pd, cl = where["Pd"][0], where["Cl"]
    core = [0, 1, 2]  # the propyl backbone of one ligand: no metal, no donor pair

    out = rx.embed(_MA2B2, metal="square_planar", n=2, template=(ref.mol, {i: i for i in core}))
    assert {e.tag["label"] for e in out} == {"cis", "trans"}
    for ens in out:
        assert sorted(ens.cons.frozen) == core  # the templated atoms, and not the metal sphere
        for cid in ens.ids:
            p = ens.mol.GetConformer(cid).GetPositions()
            for a, b in [(0, 1), (0, 2), (1, 2)]:  # the whole grafted core, not just one distance
                assert np.linalg.norm(p[a] - p[b]) == pytest.approx(np.linalg.norm(pos[a] - pos[b]), abs=1e-6)
            got = _angle(p, cl[0], pd, cl[1])  # the arrangement the label promises SURVIVED the graft
            assert got > 150 if ens.tag["label"] == "trans" else got < 120, (ens.tag["label"], got)


def test_a_graft_over_the_coordination_sphere_is_refused():
    """Pinning two sphere atoms fixes the arrangement, so it must FAIL rather than relabel the reference.

    Regression: templating the two Cl of the MA2B2 pair pinned Cl...Cl, so both enumerated isomers came back at
    the reference's 91.9°: the `trans` candidate was a cis geometry wearing the `trans` label. Refused on the
    enumerating `metal=` path and on a single chosen `Isomer` alike.
    """
    ref, pos, where = _cis_reference()
    sphere = [where["Pd"][0], *where["Cl"], *where["N"]]
    for core in ([*where["Cl"]], sphere):
        with pytest.raises(ValueError, match="coordination-sphere atoms"):
            rx.embed(_MA2B2, metal="square_planar", n=2, template=(ref.mol, {i: i for i in core}))
    iso = next(iter(rx.metal(_MA2B2, "square_planar")))
    with pytest.raises(ValueError, match="coordination-sphere atoms"):
        rx.embed(iso, n=2, fix={i: tuple(pos[i]) for i in where["Cl"]})


# --- from_geometry: retaining the input's own arrangement -------------------------------------------------

_RETAIN_RELAX = "relaxed into its windows"  # the seam's clarity line for the retain-input path


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_retained_input_geometry_is_logged_as_relaxed(tmp_path, caplog):
    """`rx.embed(metal_geometry)` retains the arrangement but relaxes it, and says so.

    The seam relaxes ids[0] into its coordination windows (the M-donor sphere is held <0.01 Å so the
    arrangement survives), which is what makes scoring fair; an unrelaxed input would score perfectly against
    itself. The line fires only on the retain path, never on a plain constrained embed.
    """
    xyz = tmp_path / "pd.xyz"
    rx.embed(rx.metal(_MA2B2, "square_planar")[0], n=1, seed=1).dump(str(xyz))

    with caplog.at_level("INFO", logger="rxembed"):
        rx.embed(str(xyz))
    assert any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text

    caplog.clear()
    with caplog.at_level("INFO", logger="rxembed"):  # a normal constrained embed has no retained input
        rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert not any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text


# --- haptic faces: one vertex, a transient centroid -------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_sandwich_is_one_isomer_whose_stored_mol_is_phantom_free(tmp_path):
    """A bis(η5) sandwich enumerates one isomer, embeds clean at the crystal M-C, and stores no centroid dummy.

    The centroid is embed scaffolding: it lives in `cons.phantoms` at indices past the real atoms, never in a
    stored Mol, so the ring atoms remain the donors through every stage: the arrangement key, the connectivity
    diff and the dumped xyz all have to be written from the real ring.
    """
    isos = rx.metal(ferrocene())
    assert len(isos) == 1  # two identical faces on one metal: a single achiral identity
    iso = isos[0]

    assert iso.mol.GetNumAtoms() == 11
    assert set(iso.donors) == set(range(1, 11))  # every ring atom IS a donor, not a centroid dummy
    assert iso.cons.phantoms, "the constraints must name the centroid dummies (the embed scaffolding)"
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms), "a dummy sits inside the real atoms"
    assert len(iso.cons.haptic) == 2  # one centroid per face
    arr = K.arrangement(iso)
    assert arr.count("η5") == 2, "the vertex must render as a hapticity tag, from the ring the mol really has"
    assert isos.select(arrangement=arr) is not None

    ens = rx.embed(iso, n=4).minimize()
    assert ens.mol.GetNumAtoms() == 11  # still real after the full relax
    assert ens.ids
    cid = next(iter(ens.ids))
    assert geom.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    mc = sorted(float(np.linalg.norm(pos[iso.metal] - pos[c])) for c in iso.donors)
    assert mc[0] >= 1.9, "the closest ring atom collapsed onto the metal"
    assert mc[-1] <= 2.3, "the farthest ring atom left the η5 shell"
    ens.filter("connectivity")  # re-perceives the graph; must not choke on (or find) a phantom

    with open(ens.dump(str(tmp_path / "ferrocene.xyz"))) as f:
        assert int(f.readline()) == 11, "a centroid dummy reached the dumped xyz"


def test_a_haptic_complex_survives_the_mc_search():
    """`.mc()` and its explore pass survive a sandwich: the seed-settle materialises the centroid.

    Regression: `_settle_seeds` rebuilt a `Constraints` without `phantoms`/`haptic`, so the seeded M->centroid
    distance named an index the relax never materialised.
    """
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc().ids
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc(explore=True).ids


def test_a_known_isomer_seats_a_haptic_face_by_any_of_its_ring_atoms():
    """`Isomer(mol, geometry, sites)`: a face is one vertex, named by any single ring atom it contains."""
    mol = ferrocene()
    rings = [n.GetIdx() for n in mol.GetAtomWithIdx(M.metal_index(mol)).GetNeighbors()]
    iso = K.Isomer(mol, "LIN", {0: rings[0], 1: rings[-1]})  # one atom per Cp, not all five
    assert len(iso.haptic) == 2
    assert iso.vertices == sorted(iso.haptic)  # both vertices are centroid dummies, not raw ring atoms
    assert rx.embed(iso, n=2, seed=1).minimize().ids


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_from_geometry_collapses_each_haptic_face_to_one_vertex():
    """The retain-input path collapses each Cp face to one centroid vertex, exactly as the enumerate path does.

    Without the collapse, raw ring atoms flow into `cons`/`vertices`/`haptic`; haptic {}, ten vertices and 45
    angles across raw ring atoms, and the transient-centroid mechanism never fires on `rx.embed(xyz)`.
    """
    iso = K.from_geometry(ferrocene())
    assert len(iso.haptic) == 2
    assert len(iso.vertices) == 2  # TWO coordination sites, not ten raw ring atoms
    assert all(v in iso.haptic for v in iso.vertices)
    assert iso.cons.haptic == dict(iso.haptic)
    assert len(iso.cons.phantoms) == 2
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms)
    assert iso.mol.GetNumAtoms() == 11  # the stored mol is REAL (Fe + 2 Cp)
    assert set(iso.donors) == set(range(1, 11))  # every ring atom is still a real donor
    assert len(iso.cons.angles) == 1  # one centroid-M-centroid angle (was 45 across raw ring atoms)

    ens = rx.embed(ferrocene(), n=4, seed=1).minimize()  # ...and the retained arrangement embeds end to end
    assert ens.mol.GetNumAtoms() == 11
    assert ens.ids
    cid = next(iter(ens.ids))
    assert geom.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    fe = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 26)
    for ring in ens.cons.haptic.values():
        d = float(np.linalg.norm(pos[fe] - np.mean([pos[a] for a in ring], axis=0)))
        assert 1.5 < d < 1.85, "Fe->Cp-centroid must sit at the ~1.66 Å crystal distance"
    ens.filter("connectivity")
    assert ens.ids


@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_a_half_sandwich_is_a_piano_stool_not_a_flat_square(door):
    """A haptic CN4 (Cp + 3 sigma) takes the TETRAHEDRAL default: a face is apical, never an in-plane vertex.

    `square_planar` is the CN4 vertex-count default and would seat a chloride trans through the ring.
    """
    iso = rx.metal(cp_ticl3())[0] if door == "enumerate" else K.from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1


def test_the_piano_stools_chlorides_never_sit_trans_through_the_ring():
    """The embedded CpTiCl3 keeps every Cl off the ring axis and off each other: the flat-square absurdity."""
    iso = rx.metal(cp_ticl3())[0]
    ens = rx.embed(iso, n=6).minimize()
    assert ens.ids
    pos = ens.mol.GetConformer(next(iter(ens.ids))).GetPositions()
    ring = next(iter(iso.cons.haptic.values()))
    centroid = np.mean([pos[a] for a in ring], axis=0)
    cls = [a.GetIdx() for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 17]
    for cl in cls:
        u, v = centroid - pos[iso.metal], pos[cl] - pos[iso.metal]
        assert np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))) < 150.0
    for a, b in itertools.combinations(cls, 2):
        assert _angle(pos, a, iso.metal, b) < 150.0


def test_an_eta2_face_collapses_a_cn7_miscount_to_octahedral():
    """KASFIU (W-Tp-NO-PMe3-η²pyridine): the η² C=C is one vertex, so the sphere is CN6, not a CN7 miscount.

    Its two adjacent pyridine carbons both bind W at ~2.3 Å in the crystal. Counting them as two sigma donors gave
    a pentagonal bipyramid that did not embed geometry-clean.
    """
    smi = "CN(C)c1ncc[cH]2->[W+2]34(<-[N-]=O)(<-[cH]12)(<-[n]1cccn1[BH-](n1ccc[n]->31)n1ccc[n]->41)<-[P](C)(C)C"
    isos = rx.metal(smi)
    assert isos
    assert all(iso.geometry == "octahedral" for iso in isos)
    assert all(len(iso.haptic) == 1 for iso in isos)
    ens = rx.embed(isos[0], n=4).minimize()
    assert any(geom.check(ens.mol, c).ok() for c in ens.ids), "no geom.check-clean η²-pyridine conformer"


# ---------------------------------------------------------------------------------------------------------
# from_geometry seats its donors on the polyhedron: the ROUND TRIP between the two Isomer doors
# ---------------------------------------------------------------------------------------------------------


def _square_planar_pt(placement):
    """`[NH3]2PtCl2` carrying an exact square-planar conformer; `placement` maps atom index -> unit direction."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_PT_A2B2))  # donors, in atom order: N0 N2 Cl3 Cl4
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(a, Point3D(0.0, 0.0, 1.6))  # every H parked off-plane; only the sphere is read
    conf.SetAtomPosition(1, Point3D(0.0, 0.0, 0.0))  # the Pt
    for atom, (x, y) in placement.items():
        conf.SetAtomPosition(atom, Point3D(2.05 * x, 2.05 * y, 0.0))
    mol.AddConformer(conf)
    return mol


_TRANS = {0: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}  # N up/down, Cl left/right -> both pairs TRANS
_CIS = {0: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, -1)}  # N adjacent, Cl adjacent -> CIS


def _seating_is_real(iso, pos):
    """Report each (vertex pair the polyhedron calls trans, angle its two donors ACTUALLY subtend at the metal).

    The one thing `vertices` has to satisfy to be a seating: a pair the record says is opposite really is.
    Rotation-free and permutation-free, so it needs neither a canonical arrangement string nor the enumerator.
    """
    dirs = P.vertex_dirs(iso.geometry)
    return [
        (iso.vertices[i], iso.vertices[j], _angle(pos, iso.vertices[i], iso.metal, iso.vertices[j]))
        for i in range(len(dirs))
        for j in range(i + 1, len(dirs))
        if P._vertex_angle(dirs[i], dirs[j]) > M._TRANS_ANGLE
    ]


@pytest.mark.parametrize(("placement", "expected"), [(_TRANS, "trans"), (_CIS, "cis")])
def test_a_retained_isomer_seats_its_donors_on_the_polyhedron_it_names(placement, expected):
    """`from_geometry`'s `vertices` must be a SEATING, not the donors in perception order.

    `vertices[v]` is documented as the donor at vertex `v`, and everything downstream reads it that way: the
    `arrangement` selector, the chirality descriptor, any trans/cis question asked of the record. Left in
    perception order it is none of those, and nothing says so: trans `[NH3]2PtCl2` came back rendering
    byte-identically to enumerated *cis*, which inverts the project's own guidance that the arrangement is the
    reliable key and the label the coarse one.

    The assertion is against the CONFORMER, not against the enumerator: two seatings of one isomer differ by a
    rotation of the polyhedron and by swapping identical ligands, so neither the arrangement string nor the
    atom-index partition is invariant. "Opposite vertices hold donors that are really opposite" is.
    """
    mol = _square_planar_pt(placement)
    retained = K.from_geometry(mol)
    assert retained.label == expected, "the premise: the MEASURED label reads the conformer correctly"

    pos = mol.GetConformer().GetPositions()
    for a, b, angle in _seating_is_real(retained, pos):
        assert angle > 150.0, f"vertices say atoms {a} and {b} are trans, but they subtend {angle:.0f} deg"


def _ideal_sphere(geometry, symbol, scramble):
    """A perfect `geometry` of identical donors, listed in `scramble` order so perception order != vertex order."""
    dirs = P.vertex_dirs(geometry)
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom("Re"))
    ds = [rw.AddAtom(Chem.Atom(symbol)) for _ in dirs]
    for d in ds:
        rw.AddBond(me, d, DAT)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for atom, v in zip(ds, scramble, strict=True):  # donor k sits at vertex scramble[k], not vertex k
        conf.SetAtomPosition(atom, Point3D(*(np.array(dirs[v], float) / np.linalg.norm(dirs[v]) * 1.95)))
    mol.AddConformer(conf)
    return mol


@pytest.mark.parametrize(
    ("geometry", "scramble"),
    [
        ("pentagonal_bipyramidal", (3, 6, 0, 4, 1, 5, 2)),
        ("square_antiprism", (5, 2, 7, 0, 3, 6, 1, 4)),
    ],
)
def test_a_geometry_with_no_canned_isomer_list_is_still_seated_on_its_polyhedron(geometry, scramble):
    """CN7/8 have no `permutations`, and the ordering search used to fall back to the IDENTITY for them.

    That is not a fallback, it is perception order, so `vertices` was arbitrary for exactly the records where
    a full n! search is unaffordable (8! = 40 320). Measured on the corpus before the fix: the as-seated
    alignment scored 4.73 against a true optimum of 7.90, and the benchmark read that as a 1.05 Å rebuild
    error which was then mis-explained as a physics gap. It is a seating bug.

    The donors here sit on a PERFECT template, so a correct seating is exact: every pair the record calls
    trans must subtend 180°.
    """
    mol = _ideal_sphere(geometry, "F", scramble)
    iso = K.from_geometry(mol)
    assert iso.geometry == geometry, f"the premise: an ideal {geometry} must be perceived as one, got {iso.geometry}"

    pos = mol.GetConformer().GetPositions()
    for a, b, angle in _seating_is_real(iso, pos):
        assert angle == pytest.approx(180.0, abs=1.0), f"vertices call {a} and {b} trans; they subtend {angle:.0f}°"


def test_the_seating_search_finds_the_same_answer_a_full_permutation_search_would():
    """The approximation has to be exact where it can be checked, or it is just a different wrong answer.

    Seeding from every ordered donor TRIPLE is what makes it so: three correspondences fix a rotation, and one
    greedy pass from a meaningless start converges to a local optimum (5.63 against 7.90 on a real antiprism).
    """
    dirs = np.array(P.vertex_dirs("square_antiprism"), float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rng = np.random.RandomState(0)
    for trial in range(3):
        perm = rng.permutation(len(dirs))
        dd = dirs[perm] + rng.randn(len(dirs), 3) * 0.05  # scrambled, and distorted so it is not a clean puzzle
        dd /= np.linalg.norm(dd, axis=1, keepdims=True)

        def fit(order, dd=dd):
            return float(np.linalg.svd(dd[list(order)].T @ dirs, compute_uv=False).sum())

        got = K._seat_by_alignment(dd, dirs)
        best = max(itertools.permutations(range(len(dirs))), key=fit)
        assert fit(got) == pytest.approx(fit(best), abs=1e-6), f"trial {trial}: {fit(got):.3f} vs {fit(best):.3f}"


def test_the_seating_search_may_match_an_achiral_template_through_a_reflection():
    """The ordering search must score with reflections ALLOWED, or a real octahedron seats trans donors CIS.

    Naming a shape is not asking which enantiomer. Most templates here are achiral; their point group already
    contains the improper operation, and handedness is answered separately over that point group by
    `metal_polyhedron.handedness`. The two conventions can diverge because `isomer_permutations` is
    symmetry-REDUCED (15 orderings for an octahedron, not 720), so the correct member of that list is
    sometimes only reachable through an improper alignment.

    Scoring by proper rotation alone looks principled and is measured wrong: on DUGVUX, CSBRHB and BESJUE it
    put donor pairs subtending 93-95° into the record's TRANS slots, and their rebuilt spheres moved 0.5-1.6 Å
    from the crystals. The fixture below is the smallest synthetic case found where the two disagree.
    """
    ideal = np.array(P.vertex_dirs("octahedral"), float)
    rng = np.random.RandomState(0)
    for mag in (0.10, 0.18):  # the first draw is consumed on purpose: this exact state is the divergent one
        dd = ideal[rng.permutation(6)] * np.array([1.0, 1.0, -1.0]) + rng.randn(6, 3) * mag
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)

    order = max(P.isomer_permutations("octahedral"), key=lambda o: P._fit_trace(dd[list(o)].T @ ideal))
    trans = [
        float(np.degrees(np.arccos(np.clip(dd[order[i]] @ dd[order[j]], -1, 1)))) for i, j in ((0, 1), (2, 3), (4, 5))
    ]
    assert min(trans) > 140.0, f"the record's trans slots hold pairs at {[f'{a:.0f}' for a in trans]}°"


def test_an_enumeration_that_filters_everything_out_says_so(caplog, monkeypatch):
    """An empty IsomerSet reads as "this geometry has no isomers", a different claim from what happened.

    `_isomers_for_geometry` logs how many candidate orderings it will dedup, then drops any placing a chelate
    trans across a span its backbone cannot reach. When that removed all of them the count went to zero with
    no further line, so the caller saw a bare empty set. Found while swapping a ligand on a real complex:
    rx.metal reported "2 free-site arrangement(s) to dedup" and then returned nothing at all.
    """
    import logging

    monkeypatch.setattr(K, "distinct_vertex_orderings", lambda *a, **kw: [])  # every candidate rejected
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = K.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("Br[Pd]1(Cl)NCCN1")), "square_planar")
    assert not out, "the monkeypatch must leave the enumeration empty"
    assert any("no arrangement survived" in r.getMessage() for r in caplog.records), caplog.text
