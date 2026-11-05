"""`metal/sphere.py`: the distance, angle, plane and stereo holds compiled for each seated coordination sphere."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.geometry.bounds import EmbedParams
from rxembed.geometry.constraints import resolve_core
from rxembed.graph import ligand_graph
from rxembed.identity.reading import read_species
from rxembed.metal.polyhedra import POLYHEDRA
from rxembed.metal.slots import chelate_bite_window
from rxembed.metal.sphere import (
    CoordinationSphere,
    LigandFacts,
    bite_window_and_reach,
    compile_constraints,
    coordination,
)
from tests.metal_fixtures import enumerated, ferrocene, seated

_TETRAHEDRAL = math.degrees(math.acos(-1.0 / 3.0))


def _species(smiles, geometry=None, **choices):
    """Return every species of `smiles` with explicit H that enumeration lists in `geometry`."""
    return enumerated(Chem.AddHs(rx.parse_smiles(smiles)), geometry, **choices)


def _noted(smiles):
    """Return the one species a SMILES with CX slot notes specifies."""
    mol = Chem.AddHs(rx.parse_smiles(smiles))
    return read_species(mol, pins=resolve_core(mol))


def _dioxoethylene_silver():
    """Return Ag0 bound side-on to C1=C2 and by O3 and O4 on its two ends, a trigonal plane every site of which a
    chelate fills."""
    rw = Chem.RWMol()
    silver, c1, c2, o3, o4 = (rw.AddAtom(Chem.Atom(z)) for z in (47, 6, 6, 8, 8))
    for a, b, order in ((c1, c2, Chem.BondType.DOUBLE), (c1, o3, Chem.BondType.SINGLE), (c2, o4, Chem.BondType.SINGLE)):
        rw.AddBond(a, b, order)
    for atom in (c1, c2, o3, o4):
        rw.AddBond(atom, silver, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return seated(mol, "trigonal_planar", {0: c1, 1: o3, 2: o4})


def test_donor_pairs_and_shells_are_held_by_their_polyhedron():
    """Rows are centred on the unrounded vertex angle, a trans pair held straight; a vacancy's angles stay free.

    A square plane with a vacancy keeps one plane hold through its three donors, as does a trigonal plane whose
    chelate no relaxed shell can read; a trigonal pyramid holds its metal off the donor plane.
    """
    found, expected = {}, {}
    for geometry, smiles in (
        ("seesaw", "[Cl-]->[Pt+2](<-[Cl-])(<-[Br-])<-[Br-]"),
        ("SPL", "[Cl-]->[Pd+2](<-[Cl-])<-N"),
    ):
        for index, species in enumerate(_species(smiles, geometry)):
            (centre,) = species.centres
            slot = {site.atoms[0]: k for k, site in enumerate(centre.vertices) if site is not None}
            rows = compile_constraints(species).angles.items()
            found[geometry, index] = {
                frozenset((a, b)): w if w == (172.0, 180.0) else sum(w) / 2 for (a, m, b), w in rows if m == centre.atom
            }
            for pair in map(frozenset, itertools.combinations(slot, 2)):
                ideal = POLYHEDRA[centre.geometry].pair_angles[frozenset(map(slot.get, pair))]
                held = (172.0, 180.0) if ideal == 180.0 else pytest.approx(ideal)  # a trans pair is held straight
                expected.setdefault((geometry, index), {})[pair] = held
    assert found == expected
    (plane,) = compile_constraints(_species("[Cl-]->[Pd+2](<-[Cl-])<-N", "SPL")[0]).umbrellas.values()
    (pyramid,) = compile_constraints(_species("[Cl-]->[Zn+2](<-[Br-])<-[I-]", "TPY")[0]).umbrellas.values()
    assert (plane.kind, pyramid.kind) == ("plane", "pyramid")
    assert pyramid.minimum == pytest.approx(POLYHEDRA["trigonal_pyramidal"].umbrella_improper)
    (chelate,) = compile_constraints(_dioxoethylene_silver()).umbrellas.items()
    assert chelate[1].kind == "plane"
    assert chelate[0].count(0) == 1  # the metal and three sites, a side-on face at its centroid


def test_embed_params_choose_lengths_and_donor_walls():
    """Ferrocene's two centroids and ten ring carbons are each held +-0.1 A about the measured length.

    An embed of cis-PdCl2(NH3)2 with both Pd-Cl stretched to 2.6 A keeps them under input lengths and returns them to
    the model's under model lengths. Without donor orientation no donor is walled or held in its plane.
    """
    sandwich = enumerated(Chem.AddHs(ferrocene(), addCoords=True))[0]
    iron = sandwich.centres[0].atom
    held = compile_constraints(sandwich, params=EmbedParams(lengths="input"))
    assert sorted(round(hi - lo, 6) for pair, (lo, hi) in held.distances.items() if iron in pair) == [0.2] * 12
    unplaced = _species("[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[NH3]", "SPL")[0]
    with pytest.raises(ValueError, match="lengths='input' requires a conformer"):
        compile_constraints(unplaced, params=EmbedParams(lengths="input"))

    cis = rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL").select(label="cis")
    geometry = rx.embed(cis, n=1, seed=1).mol
    for chloride in (2, 3):
        rdMolTransforms.SetBondLength(geometry.GetConformer(), 1, chloride, 2.6)
    chlorides = {}
    for lengths in ("model", "input"):
        placed = rx.embed(geometry, n=1, seed=1, params=rx.EmbedParams(lengths=lengths)).minimize()
        conformer = placed.mol.GetConformer(placed.ids[0])
        chlorides[lengths] = [rdMolTransforms.GetBondLength(conformer, 1, x) for x in (2, 3)]
    assert chlorides == {"model": pytest.approx([2.29] * 2, abs=0.03), "input": pytest.approx([2.6] * 2, abs=0.03)}
    acetate = _species("C[NH2]->[Pd+2]1(<-n2ccccc2)<-[O-]C(C)=[O]->1", "SPL")[0]
    free = compile_constraints(acetate, params=EmbedParams(donor_orientation=False))
    assert not free.coplanar
    assert not [key for key in free.angles if key[0] in free.metals]


def _ethylenediamine_reach(low, high):
    """Return the ligand facts of NCCN whose N0...N3 reach spans 2 A legs at `low` to `high` deg, plus `SPAN_TOL`."""
    facts = LigandFacts(Chem.MolFromSmiles("NCCN"))
    matrix = np.zeros((4, 4))
    matrix[3, 0], matrix[0, 3] = (math.sqrt(8.0 - 8.0 * math.cos(math.radians(angle))) for angle in (low, high))
    matrix[3, 0], matrix[0, 3] = matrix[3, 0] + 0.1, matrix[0, 3] - 0.1
    facts.native_reach = matrix
    return facts


def test_chelate_bite_window_is_cut_to_the_backbone_reach():
    """A 5-ring N,N bite's calibrated window is cut to the reach two 2 A legs allow, and the reach replaces a window it
    misses; without a reach the calibrated window stands, and a pair on no 4-6 ring has no bite."""
    calibrated = chelate_bite_window(Chem.MolFromSmiles("NCCN"), 0, 3)
    unreached = LigandFacts(Chem.MolFromSmiles("NCCN"))
    unreached.native_reach = None
    legs = ((2.0, 2.0), (0, 180))
    found = {
        "cut": bite_window_and_reach(_ethylenediamine_reach(80.0, 100.0), (0,), (3,), (0, 3), *legs),
        "replaced": bite_window_and_reach(_ethylenediamine_reach(100.0, 120.0), (0,), (3,), (0, 3), *legs),
        "unreached": bite_window_and_reach(unreached, (0,), (3,), (0, 3), *legs),
        "no ring": bite_window_and_reach(LigandFacts(Chem.MolFromSmiles("NCCCCCN")), (0,), (6,), (0, 6), *legs),
    }
    assert found == {
        "cut": (pytest.approx((80.0, calibrated[1])), pytest.approx((80.0, 100.0))),
        "replaced": (pytest.approx((100.0, 120.0)), pytest.approx((100.0, 120.0))),
        "unreached": (calibrated, calibrated),
        "no ring": None,
    }


def test_crystal_chelate_windows_admit_their_geometry():
    """FOPSOT's three 5-ring C,N bites on Ir measure 79.0 to 80.9 deg; a hexagonal-planar ring of six As donors
    bonded to each other around Ni needs legs past 2.4 A to span each As-As bond, rather than an organic bite."""
    iridium = _noted(
        "Cn1cc[n]2->[Ir+3]34(<-[Cl-])(<-[c-]5cc(F)ccc5-c5cccc[n]->35)<-[c-]3c(-c5cccc[n]->45)c(F)cc(F)c3-c12 "
        "|atomProp:4.atomNote.s0:5.atomNote.OCT-lambda:6.atomNote.s1:7.atomNote.s2:19.atomNote.s4:20.atomNote.s3:"
        "27.atomNote.s5|"
    )
    ((metal, sites),) = iridium.sites.items()
    facts = LigandFacts(iridium.mol)
    bites = [
        window
        for (left, centre, right), window in compile_constraints(iridium).angles.items()
        if centre == metal and chelate_bite_window(facts.mol, left, right) is not None
    ]
    assert [lo <= 79.0 <= hi for lo, hi in bites] == [True, True, True]

    specified = _noted(
        "C[As]12->[Ni]3456<-[As]1(C)[As]->3(C)[As]->4(C)[As]->5(C)[As]->62C "
        "|atomProp:1.atomNote.s0:2.atomNote.HPL:3.atomNote.s1:5.atomNote.s2:7.atomNote.s3:9.atomNote.s4:11.atomNote.s5|"
    )
    windows = compile_constraints(specified).distances
    ((metal, sites),) = specified.sites.items()
    assert min(windows[min(metal, d), max(metal, d)][1] for d in sites.donors) > 2.4


_SNS_PINCER = (  # PIVPOB: a rigid six-ring S,N,S pincer on the equator of a vanadium dioxo bipyramid
    "[O-2]->[V+5]12(<-[O-2])<-[S-]CCc3cccc(CC[S-]->1)[n]->23 "
    "|atomProp:0.atomNote.s0:1.atomNote.TBP:2.atomNote.s1:3.atomNote.s2:13.atomNote.s3:14.atomNote.s4|"
)
_SNS_PINCER_VACANT = (  # the same pincer with one axial oxo removed
    "[O-2]->[V+5]12<-[S-]CCc3cccc(CC[S-]->1)[n]->23 "
    "|atomProp:0.atomNote.s0:1.atomNote.TBP:2.atomNote.s2:12.atomNote.s3:13.atomNote.s4|"
)
_ACAC_ZIRCONIUM_ANTIPRISM = (  # a square antiprism, whose shape reading is only an upper bound
    "CC1=CC(C)=[O]->[Zr+4](<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])(<-[F-])<-[O-]1 "
    "|atomProp:5.atomNote.s0:6.atomNote.SQA-lambda:7.atomNote.s1:8.atomNote.s2:9.atomNote.s3:10.atomNote.s4:"
    "11.atomNote.s5:12.atomNote.s6:13.atomNote.s7|"
)


def _bites(species, pair, request=None):
    """Return the windows of the bite rows between donors of the element `pair` that `species` compiles."""
    mol = species.mol
    return {
        window
        for (left, _metal, right), window in compile_constraints(species, request).angles.items()
        if {mol.GetAtomWithIdx(left).GetSymbol(), mol.GetAtomWithIdx(right).GetSymbol()} == set(pair)
    }


def test_chelate_bite_opens_only_where_shells_read_the_sphere():
    """PIVPOB's equatorial S-V-N bites measure 105-108 deg, past the 74-104 deg six-ring census: the relaxed shells
    open them to the backbone's reach. A grafted metal, a held backbone, a vacancy or a polyhedron read only
    approximately leaves the census window."""
    free = _noted(_SNS_PINCER)
    census = chelate_bite_window(LigandFacts(free.mol).mol, 3, 14)
    kept = {
        "grafted metal": _bites(free, "SN", rx.Constraints(frozen=frozenset({1}))),
        "held backbone": _bites(free, "SN", resolve_core(free.mol, constrain={(4, 5): (1.4, 1.6)}).cons),
        "vacancy": _bites(_noted(_SNS_PINCER_VACANT), "SN"),
        "acac antiprism": _bites(_noted(_ACAC_ZIRCONIUM_ANTIPRISM), "OO"),
    }

    assert all(lo <= 105.0 and hi >= 108.0 for lo, hi in _bites(free, "SN"))
    assert kept == {name: {census} for name in kept}  # acac's six-ring census is the pincer's


_STYRENE_PALLADIUM = (
    "C[P:5](C)(C)->[Pd+2:4]1(<-[CH2:1]=[CH:2]->1c1ccccc1)(<-[H-:3])<-[P:6](C)(C)C "
    "|atomProp:1.atomNote.s0:4.atomNote.SPL:5.atomNote.s1+:6.atomNote.s1+:13.atomNote.s2:14.atomNote.s3|"
)
_NAPHTHALENE_RHODIUM = (  # BAMROX: an eta4 naphthalene, its unbound fusion carbons on the folded hinge
    "c1ccc([P]2(CCO[c]34->[Rh+]<-2567<-[cH]3[cH]->5[c]->6(OCC[P]->7(c2ccccc2)c2ccccc2)c2ccccc24)c2ccccc2)cc1 "
    "|atomProp:4.atomNote.s0:8.atomNote.s2:9.atomNote.TPL:10.atomNote.s2:11.atomNote.s2:12.atomNote.s2:16.atomNote.s1|"
)
_NORBORNADIENE_NICKEL = (
    "[O+]#[C-]->[Ni]123(<-[C-]#[O+])<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31 "
    "|atomProp:1.atomNote.s0:2.atomNote.TET:3.atomNote.s1:5.atomNote.s2:6.atomNote.s2:10.atomNote.s3:11.atomNote.s3|"
)


def _pushed(species):
    """Return the face substituents each metal pushes toward a tetrahedral angle off it."""
    cons = compile_constraints(species)
    return {key[2] for key, window in cons.pulls.items() if len(key) == 3 and window == (_TETRAHEDRAL, 180.0)}


def test_haptic_face_is_held_through_its_centroid():
    """Its hinge and side-on substituents are pushed off the metal: a side-on styrene's four, an eta4 naphthalene's two
    fusion carbons. A norbornadiene's centroids take what its bicycle reaches, short of tetrahedral; Cp is a cone."""
    pushed, expected = {}, {}
    for name, species in (("styrene", _noted(_STYRENE_PALLADIUM)), ("naphthalene", _noted(_NAPHTHALENE_RHODIUM))):
        ((metal, sites),) = species.sites.items()
        (face,) = (set(atoms) for atoms in sites.haptic.values())
        beside = {x.GetIdx() for a in face for x in species.mol.GetAtomWithIdx(a).GetNeighbors()} - face - {metal}
        rings = [set(ring) for ring in Chem.GetSymmSSSR(ligand_graph(species.mol)) if face < set(ring)]
        pushed[name] = _pushed(species)
        expected[name] = beside if len(face) == 2 else set().union(*rings) - face
    assert pushed == expected

    norbornadiene = _noted(_NORBORNADIENE_NICKEL)
    ((nickel, sites),) = norbornadiene.sites.items()
    left, right = sites.haptic
    hi = compile_constraints(norbornadiene).angles[left, nickel, right][1]
    assert hi < _TETRAHEDRAL - 5.0

    sandwich = enumerated(Chem.AddHs(ferrocene(), addCoords=True))[0]
    ((_iron, sites),) = sandwich.sites.items()
    cone = compile_constraints(sandwich).distances
    assert all((min(c, a), max(c, a)) in cone for c, face in sites.haptic.items() for a in face)


def _hydrogen_bridge(element):
    """Return Fe0 bound by an X1-H2 bond's two atoms and N3, placed in one plane, X1 of `element`."""
    rw = Chem.RWMol()
    metal, x, bridge, nitrogen = (rw.AddAtom(Chem.Atom(z)) for z in (26, element, 1, 7))
    rw.AddBond(x, bridge, Chem.BondType.SINGLE)
    for donor in (x, bridge, nitrogen):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((0, 0, 0), (2.1, 0, 0), (1.4, 1.15, 0), (0, 2.0, 0))):
        conformer.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conformer)
    return mol


def test_only_a_hydrogen_bridge_keeps_its_measured_span():
    """Under input lengths a three-centre B-H or C-H bonded to the metal at both ends keeps its measured span and pull,
    as the M-H length reads it; an Si-H keeps RDKit's native bond. Model legs need no widening to span a B-H."""
    span = float(np.linalg.norm(np.subtract((2.1, 0, 0), (1.4, 1.15, 0))))
    trigonal = POLYHEDRA["trigonal_planar"]
    held, expected = {}, {}
    for element, bridged in ((5, True), (6, True), (14, False)):
        mol = _hydrogen_bridge(element)
        facts = LigandFacts(mol)
        pos = mol.GetConformer().GetPositions()
        cons = coordination(CoordinationSphere(facts.mol, 0, 26, (1, 2, 3), {}, trigonal, pos=pos, facts=facts))
        held[element] = (cons.distances[1, 2], cons.pulls.get((1, 2)))
        native = (facts.bounds[2, 1], facts.bounds[1, 2])
        bridge = (pytest.approx((span - 0.1, span + 0.1)), pytest.approx((span, span)))
        expected[element] = bridge if bridged else (pytest.approx(native), None)
    assert held == expected
    facts = LigandFacts(_hydrogen_bridge(5))
    model = coordination(CoordinationSphere(facts.mol, 0, 26, (1, 2, 3), {}, trigonal, facts=facts)).distances
    widths = {donor: model[0, donor][1] - model[0, donor][0] for donor in (1, 2, 3)}
    assert [widths[1], widths[2]] == pytest.approx([widths[3]] * 2)  # the legs already span the B-H: neither widens


def test_tridentate_fan_pulls_each_angle_inside_its_window():
    """A diethylenetriamine Cd on a square pyramid, its central N apical and its arms trans: each pull sits inside its
    row, and the four basal donors' targets put them in one plane."""
    species = _noted(
        "[Cl-]->[Cd+2]12(<-[Cl-])<-[NH2]CC[NH]->1CC[NH2]->2 "
        "|atomProp:0.atomNote.s1:1.atomNote.SPY:2.atomNote.s3:3.atomNote.s2:6.atomNote.s0:9.atomNote.s4|"
    )
    metal = species.centres[0].atom
    cons = compile_constraints(species)
    targets = {key: target for key, (target, _target) in cons.pulls.items() if len(key) == 3}
    windows = {key: cons.angles[key] if key in cons.angles else cons.angles[key[::-1]] for key in targets}
    assert len(targets) == 10
    assert targets == {key: pytest.approx((lo + hi) / 2, abs=(hi - lo) / 2 + 1e-8) for key, (lo, hi) in windows.items()}

    basal = [site.atoms[0] for site in species.centres[0].vertices[1:]]
    radius = {d: sum(cons.distances[tuple(sorted((d, metal)))]) / 2 for d in basal}
    gram = np.diag([radius[d] ** 2 for d in basal])
    for i, j in itertools.combinations(range(4), 2):
        a, b = basal[i], basal[j]
        angle = targets.get((a, metal, b), targets.get((b, metal, a)))
        gram[i, j] = gram[j, i] = radius[a] * radius[b] * np.cos(np.radians(angle))
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:3]
    points = eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0, None))
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    assert spread[-1] < 0.05 * spread[0]


def test_ligand_stereo_and_donor_protons_are_held():
    """An N-H imine's E/Z takes the metal as its reference with the H held opposite; an N-methyl imine's and a free
    double bond's E/Z keep their own references; a ligand stereocentre keeps its signed volume; an aniline donor's
    two H stay above RDKit's native 1-3 floor."""
    anchors = {
        "".join(imine.stereo.bonds.values()): {row.kind: row.anchor for row in compile_constraints(imine).coplanar}
        for imine in _species("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL")
    }
    assert anchors == {"E": {"metal_ez": 180.0, "ligand_ez": 0.0}, "Z": {"metal_ez": 0.0, "ligand_ez": 180.0}}

    own = {}
    for name, smiles in (
        ("N-methyl imine", "C/C=N(\\C)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"),
        ("free alkene", "C/C=C/C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"),
    ):
        rows = compile_constraints(_species(smiles, "SPL")[0]).coplanar
        own[name] = [(row.atoms, row.anchor) for row in rows if row.kind.endswith("_ez")]
    assert own == {"N-methyl imine": [((0, 1, 2, 3), 0.0)], "free alkene": [((0, 1, 2, 3), 180.0)]}

    amine = _species("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL")[0]
    carbon = amine.mol.GetAtomWithIdx(1)
    held = {key for key, umbrella in compile_constraints(amine).umbrellas.items() if umbrella.kind == "side"}
    assert held == {tuple(n.GetIdx() for n in carbon.GetNeighbors())}

    aniline = _species("[NH2](c1ccccc1)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL")[0]
    protons = sorted(n.GetIdx() for n in aniline.mol.GetAtomWithIdx(0).GetNeighbors() if n.GetAtomicNum() == 1)
    native = LigandFacts(aniline.mol).bounds
    assert compile_constraints(aniline).floors[tuple(protons)] == pytest.approx(float(native[protons[1], protons[0]]))


def test_only_fix_and_graft_may_override_sphere_terms():
    """A soft distance, angle or plane dihedral on a sphere term is refused; a graft keeps a side-on face's centroid
    while a free donor's row needs it."""
    species = _species("[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[NH3]", "SPL")[0]  # Cl0 Pd1 Cl2 N3 N4

    plane = next(iter(compile_constraints(species).umbrellas))
    for overlap in ({(0, 1): (2.0, 2.5)}, {(0, 1, 2): (80.0, 100.0)}, {plane: (40.0, 60.0)}):
        with pytest.raises(ValueError, match=r"constrain= overlaps the coordination term.*override with fix="):
            compile_constraints(species, resolve_core(species.mol, constrain=overlap).cons)
    fixed = compile_constraints(species, resolve_core(species.mol, fix={(0, 1): 2.4}).cons)
    assert fixed.distances[0, 1] == pytest.approx((2.38, 2.42))
    grafted = compile_constraints(species, rx.Constraints(frozen=frozenset({0, 1, 2})))
    assert {(0, 1), (1, 2)}.isdisjoint(grafted.distances)
    assert (0, 1, 2) not in grafted.angles
    assert (1, 3) in grafted.distances

    zeise = _species("[CH2]1=[CH2]->[Pt+2](<-[Cl-])(<-[Cl-])(<-n2ccccc2)<-1", "SPL")[0]  # C0=C1 on Pt2, pyridine N5
    face_and_metal = compile_constraints(zeise, rx.Constraints(frozen=frozenset({0, 1, 2})))
    sphere = compile_constraints(zeise, rx.Constraints(frozen=frozenset(range(6))))
    assert face_and_metal.haptic  # its centroid still anchors the rows to the free donors
    assert not sphere.haptic
    assert not [key for key in sphere.distances if 2 in key] + [key for key in sphere.angles if key[1] == 2]
    assert [row.kind for row in sphere.coplanar] == ["donor_plane"]  # the pyridine ring is not grafted
