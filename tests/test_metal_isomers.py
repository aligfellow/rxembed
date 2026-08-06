"""Test metal-isomer construction, retention and enumeration."""

from __future__ import annotations

import itertools
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
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


# --- sandwich / piano-stool builders (tests/test_metal_core.py imports `ferrocene`) ---------------------


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
    ids=["square-planar-ma2b2", "octahedral-ma3b3"],
)
def test_metal_isomers_embed_clean(smiles, geometry, labels):
    cands = rx.embed(smiles, metal=geometry, n=4)
    assert {e.tag["label"] for e in cands} == labels
    for e in cands:
        _assert_clean(e)


def test_bis_en_octahedral_has_three_stereoisomers():
    isos = rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")
    embeddable = [iso for iso in isos if rx.embed(iso, n=2).minimize().n]
    assert {i.chirality for i in embeddable} == {"", "delta", "lambda"}, [i.chirality for i in embeddable]


def test_select_accepts_geometry_code_or_name():
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
def test_short_chelate_excludes_trans(smi):
    short = 0
    for iso, a, b, ang in _same_ligand_vertex_angles(rx.metal(smi, "square_planar")):
        dmat = Chem.GetDistanceMatrix(iso.mol)
        if dmat[a][b] <= 4:  # a short chelate: its two donors are at most 4 bonds apart
            short += 1
            assert ang < M._SPAN_ANGLE, f"short chelate {a}-{b} enumerated trans ({ang}°)"
    assert short, "no same-ligand pair was within 4 bonds: the span filter was never exercised"


def test_untabulated_single_ordering_is_retained():
    isos = rx.metal("Cl[Mo](Cl)(Cl)(Cl)(Cl)(Cl)Cl")  # homoleptic MoCl7: one ordering, no haptic face
    assert isos[0].geometry == "pentagonal_bipyramidal"


def test_eta2_pair_passes_orientation_screen():
    smi = (
        "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)[c-]1->[Rh+]123(<-[C-]#[O+])"
        "<-[CH]4=[CH]->1CC[CH]->2=[CH]->3CC4"
    )
    assert len(rx.metal(smi)) > 0


def test_t_shape_seats_its_trans_pair_first():
    iso = rx.metal("CP(C)(C)->[Rh](Cl)<-P(C)(C)C", "t_shape").select(index=0)
    seated = [iso.mol.GetAtomWithIdx(v).GetSymbol() for v in iso.vertices]
    assert (seated[0], seated[2]) == ("P", "P"), f"trans vertices got {seated}"
    assert seated[1] == "Cl"


# --- the "no permutations tabulated" info-log: only when an arrangement is really lost -------------------

_PERM_WARN = "no isomer permutations tabulated"  # the stable substring of the guard's info-log


def test_untabulated_single_arrangement_is_quiet(caplog):
    with caplog.at_level("INFO", logger="rxembed.metal"):
        isos = rx.metal(ferrocene())
    assert isos, "the single ordering must still come back"
    assert isos[0].geometry == "linear"
    assert not any(_PERM_WARN in r.message for r in caplog.records), caplog.text


def test_four_distinct_tetrahedral_donors_warn(caplog):
    with caplog.at_level("INFO", logger="rxembed.metal"):
        isos = rx.metal(tetrahedral_four_distinct(), "tetrahedral")
    assert isos, "the single ordering must still come back"
    assert isos[0].geometry == "tetrahedral"
    assert any(_PERM_WARN in r.message for r in caplog.records), caplog.text


# --- Isomer(mol, geometry, sites): the known-isomer front door -------------------------------------------


def test_known_isomer_seats_real_atom_indices():
    mol = _pt()
    cis = K.Isomer(mol, "SPL", {0: 0, 1: 2, 2: 3, 3: 4})  # the two N on adjacent vertices
    trans = K.Isomer(mol, "square_planar", [0, 3, 2, 4])  # ...and across (a list is vertex-ordered)
    assert (cis.label, trans.label) == ("cis", "trans")
    assert cis.geometry == trans.geometry == "square_planar"  # the code is an input alias, not the identity
    assert cis.coordination() is cis.cons
    assert K.Isomer(_pt(), "OCT", {0: 0, 1: 2, 2: 3, 3: 4}).vertices.count(M.VACANT) == 2  # a bigger shell
    _assert_clean(rx.embed(trans, n=2, seed=1))  # ...and a hand-seated Isomer embeds


@pytest.mark.parametrize(
    ("sites", "match"),
    [
        ({0: 0}, "given no vertex"),  # an unseated donor would silently embed a different isomer
        ({0: 0, 1: 0, 2: 3, 3: 4}, "seated at two vertices"),
        ({0: 0, 1: 2, 2: 3, 9: 4}, "not one of this geometry"),
        ({0: 5, 1: 2, 2: 3, 3: 4}, "not a donor"),  # atom 5 is an ammine H
    ],
    ids=["missing-vertex", "duplicate-vertex", "wrong-geometry", "not-donor"],
)
def test_isomer_rejects_invalid_sites(sites, match):
    with pytest.raises(ValueError, match=match):
        K.Isomer(_pt(), "SPL", sites)


def test_isomer_source_and_metal_are_mutually_exclusive():
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


def test_offsphere_template_leaves_arrangement_free():
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


def test_graft_over_the_coordination_sphere_is_refused():
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


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_retained_input_geometry_is_logged_as_relaxed(tmp_path, caplog):
    xyz = tmp_path / "pd.xyz"
    rx.embed(rx.metal(_MA2B2, "square_planar")[0], n=1, seed=1).dump(str(xyz))

    with caplog.at_level("INFO", logger="rxembed"):
        rx.embed(str(xyz), n=1)
    assert any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text

    caplog.clear()
    with caplog.at_level("INFO", logger="rxembed"):  # a normal constrained embed has no retained input
        rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1)
    assert not any(_RETAIN_RELAX in r.message for r in caplog.records), caplog.text


# --- haptic faces: one vertex, a transient centroid -------------------------------------------------------


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_sandwich_uses_two_centroids(door, tmp_path):
    if door == "enumerate":
        isos = rx.metal(ferrocene())
        assert len(isos) == 1  # two identical faces on one metal: a single achiral identity
        iso = isos[0]
        ens = rx.embed(iso, n=1, seed=1)
        arr = K.arrangement(iso)
        assert arr.count("η5") == 2, "the vertex must render as a hapticity tag, from the ring the mol really has"
        assert isos.select(arrangement=arr) is not None
    else:
        ens = rx.embed(ferrocene(), n=1, seed=1)  # the public geometry-retention door, not its internal builder
        iso = K.from_geometry(ferrocene())
        assert iso.cons.haptic == dict(iso.haptic)
        assert len(iso.cons.angles) == 1  # one centroid-M-centroid angle (was 45 across raw ring atoms)

    assert len(iso.haptic) == 2  # one centroid per face
    assert len(iso.vertices) == 2  # TWO coordination sites, not ten raw ring atoms
    assert all(v in iso.haptic for v in iso.vertices)
    assert iso.mol.GetNumAtoms() == 11  # the stored mol is REAL (Fe + 2 Cp)
    assert set(iso.donors) == set(range(1, 11))  # every ring atom IS a donor, not a centroid dummy
    assert len(iso.cons.phantoms) == 2, "the constraints must name the centroid dummies (the scaffolding)"
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms), "a dummy sits inside the real atoms"

    ens.minimize()
    assert ens.mol.GetNumAtoms() == 11  # still real after the full relax
    assert ens.ids
    cid = next(iter(ens.ids))
    assert geom.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    mc = sorted(float(np.linalg.norm(pos[iso.metal] - pos[c])) for c in iso.donors)
    assert mc[0] >= 1.9, "the closest ring atom collapsed onto the metal"
    assert mc[-1] <= 2.3, "the farthest ring atom left the η5 shell"
    for ring in ens.cons.haptic.values():
        d = float(np.linalg.norm(pos[iso.metal] - np.mean([pos[a] for a in ring], axis=0)))
        assert 1.5 < d < 1.85, "Fe->Cp-centroid must sit at the ~1.66 Å crystal distance"
    ens.filter("connectivity")  # re-perceives the graph; must not choke on (or find) a phantom
    assert ens.ids

    with open(ens.dump(str(tmp_path / "ferrocene.xyz"))) as f:
        assert int(f.readline()) == 11, "a centroid dummy reached the dumped xyz"


def test_haptic_complex_survives_the_mc_search():
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc().ids
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc(explore=True).ids


def test_haptic_face_seats_from_any_ring_atom():
    mol = ferrocene()
    rings = [n.GetIdx() for n in mol.GetAtomWithIdx(M.metal_index(mol)).GetNeighbors()]
    iso = K.Isomer(mol, "LIN", {0: rings[0], 1: rings[-1]})  # one atom per Cp, not all five
    assert len(iso.haptic) == 2
    assert iso.vertices == sorted(iso.haptic)  # both vertices are centroid dummies, not raw ring atoms
    assert rx.embed(iso, n=2, seed=1).minimize().ids


@pytest.mark.parametrize("door", ["enumerate", "from_geometry"], ids=["rx.metal", "from_geometry"])
def test_half_sandwich_uses_piano_stool_shape(door):
    iso = rx.metal(cp_ticl3())[0] if door == "enumerate" else K.from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1


def test_piano_stool_chlorides_avoid_trans_ring():
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


def test_eta2_face_collapses_a_cn7_miscount_to_octahedral():
    smi = "CN(C)c1ncc[cH]2->[W+2]34(<-[N-]=O)(<-[cH]12)(<-[n]1cccn1[BH-](n1ccc[n]->31)n1ccc[n]->41)<-[P](C)(C)C"
    isos = rx.metal(smi)
    assert isos
    assert all(iso.geometry == "octahedral" for iso in isos)
    assert all(len(iso.haptic) == 1 for iso in isos)


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


@pytest.mark.parametrize(("placement", "expected"), [(_TRANS, "trans"), (_CIS, "cis")], ids=["trans", "cis"])
def test_retained_isomer_uses_named_slots(placement, expected):
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


def test_untabulated_shape_uses_polyhedron():
    geometry = "square_antiprism"
    scramble = (5, 2, 7, 0, 3, 6, 1, 4)  # CN7 takes the same branch and adds no case
    mol = _ideal_sphere(geometry, "F", scramble)
    iso = K.from_geometry(mol)
    assert iso.geometry == geometry, f"the premise: an ideal {geometry} must be perceived as one, got {iso.geometry}"

    pos = mol.GetConformer().GetPositions()
    dirs = P.vertex_dirs(geometry)
    pairs = [(i, j) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
    assert len(pairs) == len(dirs) * (len(dirs) - 1) // 2, "the probe must reach every vertex pair"
    for i, j in pairs:
        ideal = P._vertex_angle(dirs[i], dirs[j])
        got = _angle(pos, iso.vertices[i], iso.metal, iso.vertices[j])
        assert got == pytest.approx(ideal, abs=1.0), (
            f"vertices {i}/{j} sit {ideal:.0f}° apart on the record; their donors "
            f"{iso.vertices[i]}/{iso.vertices[j]} subtend {got:.0f}°"
        )


def test_seating_finds_distorted_antiprism_optimum():
    dirs = np.array(P.vertex_dirs("square_antiprism"), float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rng = np.random.RandomState(0)
    # Exhaustive 8! optima for these fixed distorted fixtures, computed once rather than on every test run.
    optima = (7.9744270801213695, 7.988318361026678, 7.989758899386766)
    for trial, optimum in enumerate(optima):
        dd = dirs[rng.permutation(len(dirs))] + rng.randn(len(dirs), 3) * 0.05
        dd /= np.linalg.norm(dd, axis=1, keepdims=True)
        got = K._seat_by_alignment(dd, dirs)
        score = float(np.linalg.svd(dd[list(got)].T @ dirs, compute_uv=False).sum())
        assert score == pytest.approx(optimum, abs=1e-12), f"trial {trial}: {score:.6f} vs {optimum:.6f}"


def test_seating_allows_reflection_for_achiral_template():
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


def test_empty_isomer_enumeration_warns(caplog, monkeypatch):
    import logging

    monkeypatch.setattr(K, "distinct_vertex_orderings", lambda *a, **kw: [])  # every candidate rejected
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = K.enumerate_isomers(Chem.AddHs(Chem.MolFromSmiles("Br[Pd]1(Cl)NCCN1")), "square_planar")
    assert not out, "the monkeypatch must leave the enumeration empty"
    assert any("no arrangement survived" in r.getMessage() for r in caplog.records), caplog.text
