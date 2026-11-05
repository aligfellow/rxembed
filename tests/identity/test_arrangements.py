"""`identity/arrangements.py`: every coordination and ligand stereoisomer of a metal graph, screened by ligand reach."""

from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem, DistanceGeometry
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.geometry.constraints import Constraints
from rxembed.identity.arrangements import compiled_reach_conflict, euclidean_conflict, haptic_span_conflict
from rxembed.metal.centres import Centre
from tests.geometry_fixtures import placed
from tests.metal_fixtures import (
    BINAP_PDCL2,
    BUTADIENE_FE_CO3,
    PD2CL6,
    PT2CL4_AMMINES,
    berry_intermediate,
    on_its_vertices,
    pd2cl6,
)

MO6CL14 = (  # an octahedron of Mo(II), each face capped by a chloride, one terminal chloride per metal
    "[Cl-]->[Mo+2]123<-[Cl-]4->[Mo+2]56(<-[Cl-])<-[Cl-]->1->[Mo+2]17(<-[Cl-])<-[Cl-]->2->[Mo+2]28(<-[Cl-])"
    "<-[Cl-]->3->[Mo+2]<-43(<-[Cl-])<-[Cl-]->5->[Mo+2](<-[Cl-])(<-[Cl-]->6->1)(<-[Cl-]->3->2)<-[Cl-]->7->8"
)
NOTED_S_CIS_BUTADIENE_FE = (
    "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[CH]->1[CH]->2=[CH2]->3 |atomProp:1.atomNote.s0:"
    "2.atomNote.TET:3.atomNote.s1:5.atomNote.s2:7.atomNote.s3c:8.atomNote.s3c:9.atomNote.s3c:10.atomNote.s3c|"
)
PYRIDYLETHYL_NI = "[Cl-]->[Ni+2]12(<-[Cl-])(<-[n]3ccccc3CCc3cccc[n]->13)<-[n]3ccccc3CCc3cccc[n]->23"
NORBORNADIENE_RU = "CC#[N]->[Ru+2]123(<-[Cl-])(<-[Cl-])(<-[N]#CC)<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31"
POCOP_NI = "COC(=O)c1cc2O[P](C(C)C)(C(C)C)->[Ni+2]3(<-[Cl-])<-[c-]2c(O[P]->3(C(C)C)C(C)C)c1"
CO_EN_PY2 = "[Cl-]->[Co+3]1(<-[Cl-])(<-NCCN->1)(<-n1ccccc1)<-n1ccccc1"  # Co1, pyridine N7 and N13, para C10 and C16
AGOSTIC_PD = "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1"
PYRAZINE_PAIR = "{}->[Pd+2](<-[Cl-])(<-[Cl-])<-n1ccn(cc1)->{}(<-[Cl-])(<-[Cl-])<-{}"
CIS_PT_PLACED = (  # cis-PtCl2(NH3)2 by hand: Cl0 Pt1 Cl2 N3 N4
    "[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[NH3]",
    [(2.3, 0, 0), (0, 0, 0), (0, 2.3, 0), (-2.05, 0, 0), (0, -2.05, 0)],
)


def test_enumerate_isomers_lists_each_arrangement_once():
    """Vacancies are a pocket; joint metals count once per species; faces and their E/Z multiply the seatings."""
    ethylenediamine = on_its_vertices(rx.isomers("N1CCN->[Pd+2](<-[Cl-])(<-[Cl-])<-1", geometry="SPL")[0])
    for atom in ethylenediamine.GetAtoms():
        atom.ClearProp("atomNote")  # a geometry input, its seating one the ligand reaches
    for name, smiles, choices, count in (
        ("Mo(en)ClBr bipyramid", "[NH2]1CC[NH2]->[Mo+3]<-1(<-[Cl-])<-[Br-]", {"geometry": "PBP", "stereo": "free"}, 30),
        ("T-shaped Pd in SPL", "N->[Pd+2](<-[Cl-])<-[Br-]", {"geometry": "SPL"}, 3),
        ("bent ZnX2 vacancy", "[Cl-]->[Zn+2]<-[Br-]", {"geometry": "TPY"}, 1),
        ("seesaw Fe, one vacancy", "N->[Fe+2](<-[Cl-])<-[Br-]", {"geometry": "SEE"}, 9),
        (
            "propenes cis or trans on Pt",
            "[Cl-]->[Pt+2]12(<-[Cl-])(<-[CH2]=[CH](C)->1)<-[CH2]=[CH](C)->2",
            {"geometry": "SPL"},
            6,
        ),
        ("Z-fluoropropene on Pt", r"C/[CH]1=[CH](\F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1", {"geometry": "SPL"}, 6),
        ("s-cis and s-trans diene Fe", BUTADIENE_FE_CO3, {}, 3),
        ("s-cis-butadiene-iron-noted", NOTED_S_CIS_BUTADIENE_FE, {}, 1),
        ("diene Fe without class", NOTED_S_CIS_BUTADIENE_FE.replace("s3c", "s3"), {}, 3),
        (
            "locked imine Cd chelates",
            "O[N]1=Cc2cccc[n]2->[Cd+2]<-12(<-[I-])(<-[I-])<-[N](O)=Cc1cccc[n]->21 "
            "|t:1,atomProp:12._rxEZ0.E:14._rxEZ0.E|",
            {"screen": False},
            11,
        ),
        ("pyrazine Pd and Pt", PYRAZINE_PAIR.format("N", "[Pt+2]", "N"), {}, 4),
        ("pyrazine equal Pd", PYRAZINE_PAIR.format("N", "[Pd+2]", "N"), {}, 3),
        (
            "pyrazine Pd R and S amines",
            PYRAZINE_PAIR.format("C[C@H](c2ccccc2)N", "[Pd+2]", "N[C@@H](C)c3ccccc3"),
            {},
            4,
        ),
        (
            "pyrazine pair beside Ag ammines",
            PYRAZINE_PAIR.format("N", "[Pt+2]", "N.N->[Ag+]<-N.N->[Ag+]<-N.N->[Ag+]<-N"),
            {},
            4,
        ),
        (
            "bipy Pd bridged to Pd",
            "[NH3]->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pd+2]2(<-[Cl-]->1)<-[n]1ccccc1-c1cccc[n]->21",
            {},
            1,
        ),
        ("lone Cp Cu", "[Cu+]1234<-[CH-]5[CH]->1=[CH]->2[CH]->3=[CH]5->4", {}, 1),
        ("Zeise anion default", "[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH2]->1", {}, 1),
        ("one Pd of placed Pd2Cl6", pd2cl6(), {"center": 1, "observed": False}, 2),
        ("placed Pd(en) in its square", ethylenediamine, {"geometry": "SPL"}, 1),
        ("placed Pd(en) asked TET", ethylenediamine, {"geometry": "TET"}, 1),
    ):
        assert len(rx.isomers(smiles, **choices)) == count, name


def test_reach_screen_drops_unreachable_arrangements_unless_off():
    """A chelate never spans trans; tethered faces, a pincer's bites and shared bridges must close at model lengths."""
    for name, smiles, geometry, counts in (
        ("bis-pyridylethyl-nickel", PYRIDYLETHYL_NI, "OCT", (3, 5)),
        (
            "oxalate capping a face",
            "[F-]->[Zr+4]1(<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[O-]C(=O)C(=O)[O-]->1",
            "capped_octahedral",
            (6, 7),
        ),
        ("norbornadiene-faces-on-prism-edges", NORBORNADIENE_RU, "TPR", (10, 18)),
        ("en cannot span linear Cu", "[Cu+]1<-[NH2]CC[NH2]->1", "linear", (0, 1)),
        ("pocop-pincer-nickel", POCOP_NI, "SPL", (1, 2)),
        ("hexachlorodipalladate-bridges", PD2CL6, None, (1, 3)),
        ("Cl-bridged Pt dimer", PT2CL4_AMMINES, None, (2, 4)),
    ):
        counts_found = tuple(len(rx.isomers(smiles, geometry=geometry, screen=screen)) for screen in (True, False))
        assert counts_found == counts, name


def test_fix_keeps_only_the_arrangements_its_pins_agree_with():
    """Para carbons pinned 9.5 A apart leave the pyridines trans; a numeric pair or ligand bond pins no vertex."""
    bridge = next(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in rx.parse_smiles(PYRIDYLETHYL_NI).GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and all(a.GetAtomicNum() == 6 and not a.GetIsAromatic() for a in (b.GetBeginAtom(), b.GetEndAtom()))
    )
    agostic = AGOSTIC_PD + ".N->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"
    mol = Chem.AddHs(rx.parse_smiles(AGOSTIC_PD))
    hydrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() == 2)
    carbon = next(n.GetIdx() for n in mol.GetAtomWithIdx(hydrogen).GetNeighbors() if n.GetAtomicNum() == 6)
    for name, smiles, choices, count in (
        ("para-carbons-pinned-apart", CO_EN_PY2, {"fix": {10: (-4.75, 0, 0), 16: (4.75, 0, 0)}}, 1),
        ("Co pyridines pinned trans", CO_EN_PY2, {"fix": {1: (0, 0, 0), 7: (2, 0, 0), 13: (-2, 0, 0)}}, 2),
        ("pyridine N held 3.15 A apart", CO_EN_PY2, {"fix": {(7, 13): 3.15}}, 4),
        ("chelate-backbone-bond-held", PYRIDYLETHYL_NI, {"geometry": "OCT", "fix": {bridge: 1.5}}, 3),
        (
            "agostic C-H, second metal",
            agostic,
            {"center": "all", "stereo": "free", "fix": {(carbon, hydrogen): 2.0}},
            2,
        ),
    ):
        assert len(rx.isomers(smiles, **choices)) == count, name


def test_enumerate_isomers_refuses_an_unanswerable_request():
    cluster = rx.parse_smiles(MO6CL14)
    rdDepictor.Compute2DCoords(cluster)  # six equivalent metals refuse, whatever the conformer
    nine = "[F-]->[La+6](<-[Cl-])(<-[Br-])(<-[I-])(<-N)(<-O)(<-S)(<-P)<-[CH3-]"  # La1, nine different donors
    pd_pt = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N.N->[Pt+2](<-[Cl-])(<-[Cl-])<-N"
    noted = (
        "[NH3]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Br-] "
        "|atomProp:0.atomNote.s0:1.atomNote.SPL:2.atomNote.s1:3.atomNote.s2:4.atomNote.s3|"
    )
    for source, choices, message in (
        (MO6CL14, {}, "joined by bridges, which no name records for three or more"),
        (cluster, {"center": 1, "observed": False}, "one of three or more symmetry-equivalent metals"),
        (
            "N->[Nb+5](<-O)(<-[F-])(<-[Cl-])(<-[Br-])(<-[I-])<-n1ccn(cc1)->[Nb+5](<-O)(<-[F-])(<-[Cl-])(<-[Br-])(<-[I-])<-N",
            {},
            r"combine into 254,016 arrangements \(504 x 504",
        ),
        (pd_pt, {"center": "Pd"}, "needs a geometry for the others; read one with rx.read_xyz"),
        (pd_pt, {"center": "all", "geometry": "SPL"}, "geometry= names one polyhedron, but every metal is enumerated"),
        ("[Hg].[C-]#[O+]", {}, "Hg0 has no donor bonds.*use the Mol directly or pass geometry="),
        ("[Ag+].N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", {}, "Ag0 has no donor bonds.*enumerate the other metals without it"),
        (
            "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N",
            {"geometry": "linear"},
            "linear has 2 coordination sites but the metal has 4",
        ),
        ("[La+10]" + "(<-[F-])" * 12 + "<-[F-]", {}, "no default geometry for 13 donors; pass geometry="),
        (noted, {"geometry": "TET"}, r"already specifies a square_planar arrangement, so geometry=\['tetrahedral'\]"),
        (nine, {"geometry": "TCT", "fix": {1: (0, 0, 0), 0: (2, 0, 0), 2: (0, 2, 0)}}, "fix= leaves 5,040 free-site"),
    ):
        with pytest.raises(ValueError, match=message):
            rx.isomers(source, **choices)


def test_each_arrangement_carries_its_variants_ligand_stereo():
    """A chelate locks the BINAP axis into two hands, which a request may leave to the seed."""
    carbanions = placed(
        rx.parse_smiles("[C-](F)(Cl)(Br)->[Pt+2](<-[C-](F)(Cl)Br)(<-[I-])<-P"),
        [
            *((2, 0, 0), (2.4, 1, 1), (2.4, -1, 1), (2.4, 0, -1), (0, 0, 0)),  # a carbanion C0, then Pt4
            *((-2, 0, 0), (-2.4, 1, 1), (-2.4, -1, 1), (-2.4, 0, -1), (0, 2, 0), (0, -2, 0)),
        ],
    )
    for name, source, choices, labels in (
        ("amine beside open C", "[Pd](Cl)(Cl)(Cl)([N@H](C)C(O)C)", {"geometry": "SPL"}, ["N4:R,C6:R", "N4:R,C6:S"]),
        ("carbanions keep shown hands", carbanions, {"geometry": "SPL"}, ["C0:S,C5:R"] * 3),
        ("chelate-locked-binap-axis", BINAP_PDCL2, {}, ["C25-C26:M", "C25-C26:P"]),
        ("binap-axis-left-to-the-seed", BINAP_PDCL2, {"stereo": "free"}, [""]),
        (
            "RS and SR amines one species",
            "CC[CH](C)N->[Pd+2](<-[Cl-])(<-[Cl-])<-N[CH](C)CC",
            {"geometry": "SPL"},
            ["C2:R,C9:R", "C2:R,C9:R", "C2:S,C9:R", "C2:S,C9:R", "C2:S,C9:S", "C2:S,C9:S"],
        ),
    ):
        assert sorted(rx.identity(one).stereo for one in rx.isomers(source, **choices)) == labels, name


def test_atom_maps_label_isomers_leaving_symmetric_donors_alike():
    for smiles, geometry in (
        ("N->[Pt+2](<-N)(<-[Cl-])<-[Cl-]", "SPL"),
        ("[O+]#[C-]->[Fe]1(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-C=C->1", "TBP"),
    ):
        mol = Chem.AddHs(rx.parse_smiles(smiles))
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        found = rx.isomers(mol, geometry=geometry)
        assert len(found) == 2, smiles  # cis and trans, or the ethylene axial and equatorial
        assert all(a.GetAtomMapNum() == a.GetIdx() + 1 for one in found for a in one.GetAtoms()), smiles


def test_invert_on_an_unnoted_geometry_lists_its_mirror():
    """A delta zinc, one with a vacancy and an s-trans (P) diene read from coordinates; three donors read as a
    tetrahedron only when asked."""
    zinc = rx.isomers("[F-]->[Zn+2](<-[Cl-])(<-[Br-])<-[I-]", geometry="TET")[0]
    vacant = rx.isomers("N->[Zn+2](<-[Cl-])<-[Br-]", geometry="TET")[0]
    diene = next(one for one in rx.isomers(BUTADIENE_FE_CO3) if "η4P" in rx.identity(one).arrangements[0])
    for name, specified, geometry, shown in (
        ("delta-zinc", zinc, None, on_its_vertices(zinc)),
        ("delta-zinc-with-a-vacancy", vacant, "TET", on_its_vertices(vacant)),
        ("s-trans-butadiene-iron", diene, None, rx.embed(diene, n=1, seed=3).mol),
    ):
        read = Chem.Mol(shown)
        for atom in read.GetAtoms():
            atom.ClearProp("atomNote")
        if geometry is not None:
            read = rx.isomers(read, geometry=geometry, observed=True)[0]
        mirror = [rx.cxsmiles(one) for one in rx.isomers(specified, stereo="invert")]
        assert [rx.cxsmiles(one) for one in rx.isomers(read, stereo="invert")] == mirror, name


def _berry_midpoint_iron_beside_cis_platinum():
    """Return Fe(CO)5 (Fe2) halfway between its bipyramid and pyramid beside a placed cis-PtCl2(NH3)2."""
    iron = rx.parse_smiles("[O+]#[C-]->[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]")
    tbp = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]])
    spy = np.array([[0, 0, 1], [0, 0.97, -0.26], [0.97, 0, -0.26], [0, -0.97, -0.26], [-0.97, 0, -0.26]])
    rays = (tbp + spy) / np.linalg.norm(tbp + spy, axis=1, keepdims=True)
    conf = Chem.Conformer(iron.GetNumAtoms())
    for ray, carbon in zip(rays, [n.GetIdx() for n in iron.GetAtomWithIdx(2).GetNeighbors()], strict=True):
        oxygen = next(n.GetIdx() for n in iron.GetAtomWithIdx(carbon).GetNeighbors() if n.GetIdx() != 2)
        conf.SetAtomPosition(carbon, Point3D(*(1.8 * ray)))
        conf.SetAtomPosition(oxygen, Point3D(*(2.95 * ray)))
    iron.AddConformer(conf)
    return Chem.CombineMols(iron, placed(rx.parse_smiles(CIS_PT_PLACED[0]), CIS_PT_PLACED[1]), Point3D(14, 0, 0))


def _cp_ticl3():
    """Return CpTiCl3 placed as a piano stool, its ring above Ti0 and its chlorides below."""
    mol = rx.parse_smiles("[Ti+4]1234(<-[Cl-])(<-[Cl-])(<-[Cl-])<-[cH-]5[cH]->1[cH]->2[cH]->3[cH]->45")
    ring = [(1.2 * np.cos(a), 1.2 * np.sin(a), 1.9) for a in 2 * np.pi * np.arange(5) / 5]
    tripod = [(1.9 * np.cos(a), 1.9 * np.sin(a), -1.3) for a in 2 * np.pi * np.arange(3) / 3]
    return placed(mol, [(0, 0, 0), *tripod, *ring])


def test_observed_reads_each_centre_in_its_named_or_best_shape():
    """Fe(CO)5 halfway along the Berry path fits SPY best, so only its TBP note or request seats it TBP."""
    noted = rx.isomers(_berry_midpoint_iron_beside_cis_platinum(), center=2, geometry="TBP", observed=True)[0]
    platinum = next(atom.GetIdx() for atom in noted.GetAtoms() if atom.GetSymbol() == "Pt")
    pts = berry_intermediate(0.32)
    berry = placed(
        rx.parse_smiles("[Cl-]->[Fe+3](<-[Cl-])(<-[Cl-])(<-[Cl-])<-[Cl-]"), [2.3 * pts[0], (0, 0, 0), *(2.3 * pts[1:])]
    )
    for name, source, choices, metal, geometry in (
        ("noted Fe, every metal", noted, {"center": "all"}, 2, "trigonal_bipyramidal"),
        ("noted Fe beside read Pt", noted, {"center": platinum}, 2, "trigonal_bipyramidal"),
        ("Berry asked as SPY", berry, {"geometry": "SPY"}, 1, "square_pyramidal"),
        ("piano-stool-titanium", _cp_ticl3(), {}, 0, "tetrahedral"),
    ):
        (centre,) = (c for c in rx.identity(rx.isomers(source, observed=True, **choices)[0]).centres if c.atom == metal)
        assert centre.geometry == geometry, name


def test_an_embedded_vacancy_species_reads_back_as_itself():
    """A T-shape ties a full plane with a vertex empty; only the empty vertex tells a pyramidal zinc's mirror pair."""
    for smiles, geometry in (("N->[Pd+2](<-[Cl-])<-[Br-]", "SPL"), ("N->[Zn+2](<-[Cl-])<-[Br-]", "TET")):
        species = rx.isomers(smiles, geometry=geometry)[0]
        assert rx.cxsmiles(rx.embed(species, n=1, seed=42, threads=1).mol) == rx.cxsmiles(species), smiles


def test_euclidean_conflict_certifies_only_unrealizable_boxes():
    """Three triangles 2 A wide, 1.1 A apart, each distance within 35%: degenerate eigenspaces must not hide it."""
    groups = np.arange(9) // 3
    squared = np.where(groups[:, None] == groups[None, :], 4.0, 1.21)
    np.fill_diagonal(squared, 0.0)
    lower, upper = np.sqrt(0.65 * squared), np.sqrt(1.35 * squared)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(9)
        box = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        assert DistanceGeometry.DoTriangleSmoothing(box.copy())
        assert euclidean_conflict(box) == pytest.approx(0.2995, abs=1e-12)
    for invalid in (-0.1, np.nan, np.inf):
        box[1, 0] = invalid
        assert euclidean_conflict(box) is None, invalid


def test_compiled_reach_conflict_names_an_unspannable_angle_row():
    """A 136-152 deg row on 2 A legs needs 3.7 A between the donors, and the ligand reaches 2.5 A."""
    mol = Chem.MolFromSmiles("C.[He].[He].[He].[He]")
    constraints = Constraints(
        metals={4}, distances={(0, 4): (2.0, 2.0), (2, 4): (2.0, 2.0)}, angles={(0, 4, 2): (136.0, 152.0)}
    )
    native = np.triu(np.full((5, 5), np.inf), 1)
    native[0, 2] = 2.5
    species = SimpleNamespace(mol=mol, sites={4: SimpleNamespace(donors=(0, 2))})
    facts = SimpleNamespace(mol=mol, native_reach=native, memo={})

    assert "native ligand reach" in compiled_reach_conflict(species, Centre(4, 2, 0), constraints, facts)


def test_haptic_span_conflict_judges_trans_faces_by_centroids():
    """Side-on faces trans across metal 5: their member rays need not be trans, but their centroids must reach.

    Equal faces take the best matching of their atoms; an eta2 and an eta3 face the mean over every pair.
    """
    positions = np.array([[2, 0.7, 0], [2, -0.7, 0], [-2, 0.7, 0], [-2, -0.7, 0], [-2.4, 0, 0.6], [0, 0, 0]])
    reach = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    constraints = Constraints(distances={(5, 6): (2.0, 2.0), (5, 7): (2.0, 2.0)})
    metal = Centre(5, 2, 0)
    equal, unequal = (
        (SimpleNamespace(sites={5: SimpleNamespace(haptic={6: (0, 1), 7: right})})) for right in ((2, 3), (2, 3, 4))
    )

    assert haptic_span_conflict(equal, metal, reach, (6, 7), 180.0, constraints) is None
    reach[:2, 2:5] = reach[2:5, :2] = 3.5
    for species in (equal, unequal):
        assert "centroid reach" in haptic_span_conflict(species, metal, reach, (6, 7), 180.0, constraints)


def test_failed_native_reach_leaves_everything_unscreened(monkeypatch):
    """Without native reach the pincer keeps its unreachable cis seating and both chloride bridges stay open."""
    monkeypatch.setattr(DistanceGeometry, "DoTriangleSmoothing", lambda *_args: False)

    assert [len(rx.isomers(POCOP_NI, geometry="SPL")), len(rx.isomers(PD2CL6))] == [2, 3]
