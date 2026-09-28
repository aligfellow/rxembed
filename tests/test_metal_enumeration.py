"""Test metal-isomer enumeration and embedding integration."""

from __future__ import annotations

import itertools
from collections import Counter
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem, DistanceGeometry
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_enumeration
from rxembed import metal_slots as slots
from rxembed.bounds import coordination_reach_base, ligand_reach
from rxembed.constraints import Constraints
from rxembed.embed import embed as core_embed
from rxembed.mechanisms import law_of_cosines
from rxembed.metal_core import VACANT, HapticSite
from rxembed.metal_isomer import from_geometry
from rxembed.metal_perceive import shape_gap
from rxembed.metal_polyhedron import POLYHEDRA, record
from rxembed.metal_stereo import chelate_links
from rxembed.pipeline import geom_check as geom
from tests.metal_fixtures import ferrocene, one_arm_bound_pt

_MA2B2 = "CCCN[Pd](Cl)(Cl)NCCC"  # square-planar MA2B2 -> the cis / trans pair
S, D, DAT = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.DATIVE


def _assert_clean(ens):
    """Every conformer of a minimized ensemble passes the metal-aware geometry gate."""
    ens = ens.minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert rep.ok(), rep.summary()


def _assert_embeds_as(iso, *, extra=None, seed=42, threads=1):
    """Embed one screened isomer and assert it relaxes clean and keeps its identity; return the ensemble."""
    ensemble = rx.embed(iso, n=1, seed=seed, threads=threads)
    assert not ensemble.unrelaxed
    for report in ensemble.check().values():
        assert report.ok(), report.summary()
    assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
    if extra is not None:
        extra(ensemble, iso)
    return ensemble


def test_encoded_locked_ez_keeps_equivalent_chelate_sites_equivalent():
    text = "O[N]1=Cc2cccc[n]2->[Cd+2]<-12(<-[I-])(<-[I-])<-[N](O)=Cc1cccc[n]->21 |t:1,atomProp:12._rxEZ0.E:14._rxEZ0.E|"
    mol = rx.parse_smiles(text)
    candidates = rx.metal(mol, screen=False)

    assert len(candidates) == 11
    assert len({rx.cxsmiles(iso) for iso in candidates}) == len(candidates)


def test_unbound_metal_has_no_coordination_isomer():
    source = rx.parse_smiles("[Hg].[C-]#[O+]")

    with pytest.raises(ValueError, match="Hg0 has no donor bonds"):
        rx.metal(source)


def _closo_b6h6_bound_to_iron():
    """Build a closo-B6H6 octahedron, each boron bonding four borons and one hydrogen, dative-bound to Fe2+.

    RDKit will not sanitise a five-bonded boron from SMILES, so this is built as a raw graph.
    """
    rw = Chem.RWMol()
    borons = [rw.AddAtom(Chem.Atom(5)) for _ in range(6)]
    hydrogens = [rw.AddAtom(Chem.Atom(1)) for _ in range(6)]
    antipode = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    for i, j in itertools.combinations(range(6), 2):
        if antipode[i] != j:
            rw.AddBond(borons[i], borons[j], Chem.BondType.SINGLE)
    for boron, hydrogen in zip(borons, hydrogens, strict=True):
        rw.AddBond(boron, hydrogen, Chem.BondType.SINGLE)
    iron = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(iron).SetFormalCharge(2)
    rw.AddBond(borons[0], iron, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def test_boron_cage_fails_at_enumeration_boundary():
    source = _closo_b6h6_bound_to_iron()

    with pytest.raises(ValueError, match="two-centre donor model"):
        rx.metal(source)
    assert rx.metal("[BH3-][H]->[Fe+]")

    conformer = Chem.Mol(Chem.AddHs(source))
    coordinates = Chem.Conformer(conformer.GetNumAtoms())
    for index in range(conformer.GetNumAtoms()):
        coordinates.SetAtomPosition(index, (float(index), 0.0, 0.0))
    coordinates.Set3D(True)
    conformer.AddConformer(coordinates)
    with pytest.raises(ValueError, match="two-centre donor model"):
        rx.embed(conformer)


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


# --- enumeration: the textbook isomers ------------------------------------------------------------------


def test_input_geometry_preserves_the_indexed_hand_of_equivalent_donors():
    mol = rx.parse_smiles("[C-](F)(Cl)(Br)->[Pt+2](<-[C-](F)(Cl)Br)(<-[I-])<-P")
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = [
        (2, 0, 0),
        (2.4, 1, 1),
        (2.4, -1, 1),
        (2.4, 0, -1),
        (0, 0, 0),
        (-2, 0, 0),
        (-2.4, 1, 1),
        (-2.4, -1, 1),
        (-2.4, 0, -1),
        (0, 2, 0),
        (0, -2, 0),
    ]
    for atom, position in enumerate(positions):
        conf.SetAtomPosition(atom, Point3D(*position))
    mol.AddConformer(conf)

    isomers = rx.metal(mol, "square_planar")

    assert isomers
    assert {iso.stereo_label for iso in isomers} == {"C0:S,C5:R"}
    assert all(sum(value == 0.0 for value in iso.cons.umbrellas.values()) == 2 for iso in isomers)


@pytest.mark.parametrize(
    ("smiles", "geometry", "labels"),
    [
        ("[NH3][Co]([NH3])([NH3])(Cl)(Cl)Cl", "octahedral", {"mer", "fac"}),  # MA3B3
    ],
    ids=["octahedral-ma3b3"],
)
def test_metal_isomers_embed_clean(smiles, geometry, labels):
    cands = rx.embed(smiles, metal=geometry, n=4)
    assert {e.tag["label"] for e in cands} == labels
    for e in cands:
        _assert_clean(e)


@pytest.mark.parametrize(
    ("geometry", "smiles", "per_hand"),
    [
        ("seesaw", "[O+]#[C-]->[Fe+2](<-[F-])(<-[Cl-])<-N", 6),
    ],
    ids=["seesaw"],
)
def test_chiral_polyhedra_enumerate_both_hands(geometry, smiles, per_hand):
    isos = rx.metal(smiles, geometry, stereo="free")
    assert Counter(i.chirality for i in isos) == {"delta": per_hand, "lambda": per_hand}
    assert {i.label for i in isos} == {""}  # all donors differ, so cis/trans has no meaning


def test_defined_and_enumerated_ligand_stereo_share_one_label():
    isomers = rx.metal("[Pd](Cl)(Cl)(Cl)([N@H](C)C(O)C)", "SPL")
    assert {iso.stereo_label for iso in isomers} == {"N4:R,C6:R", "N4:R,C6:S"}


def test_bound_amine_hands_are_enumerated_only_on_request():
    """A geometry input keeps its measured bound-N hands; stereo={'locked': 'racemic'} adds every other pair."""
    mol = one_arm_bound_pt()
    measured = rx.cxsmiles(mol)
    identities = [rx.cxsmiles(iso) for iso in rx.metal(mol, stereo={"locked": "racemic"})]

    assert [rx.cxsmiles(iso) for iso in rx.metal(mol)] == [measured]
    assert measured in identities
    assert len(set(identities)) == len(identities) == 4


def test_dodecahedral_bipyridyl_screen_rejects_unspanned_whole_ligand_assignments():
    smiles = "[F-]->[Nb+4]12(<-[F-])(<-[F-])(<-[F-])(<-[n]3ccccc3-c3cccc[n]->13)<-[n]1ccccc1-c1cccc[n]->21"
    mol = rx.parse_smiles(smiles)
    coordinate_free = Chem.Mol(mol)
    assert len(rx.metal(coordinate_free, "dodecahedral", screen=False)) == 66
    assert len(rx.metal(coordinate_free, "dodecahedral")) == 29
    metal = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 41)
    donors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetAtomPosition(metal, (0.0, 0.0, 0.0))
    directions = np.asarray(POLYHEDRA["dodecahedral"].vertex_dirs, float)
    for slot, donor in enumerate(donors):
        # Deliberately exceed native bipyridyl reach, beyond publication's M-L slack, only when these radii
        # are requested: at 3.0 A the lower window edge sat within 0.01 A of reach.
        radius = 3.1 if mol.GetAtomWithIdx(donor).GetAtomicNum() == 7 else 1.9
        conformer.SetAtomPosition(donor, tuple(radius * directions[slot]))
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in {metal, *donors}:
            conformer.SetAtomPosition(atom.GetIdx(), (10.0 + atom.GetIdx(), 0.0, 0.0))
    mol.AddConformer(conformer)

    unrestricted = rx.metal(mol, "dodecahedral", screen=False)
    screened = rx.metal(mol, "dodecahedral")

    assert len(unrestricted) == 66
    assert {rx.cxsmiles(iso) for iso in screened} == {
        rx.cxsmiles(iso) for iso in rx.metal(coordinate_free, "dodecahedral")
    }
    assert len(rx.metal(mol, "dodecahedral", lengths="input", screen=False)) == 66
    assert len(rx.metal(mol, "dodecahedral", lengths="input")) == 0


def test_refined_chelate_screen_keeps_explicit_authority_and_embeddable_states(monkeypatch):
    smiles = "[Cl-]->[Ni+2]12(<-[Cl-])(<-[n]3ccccc3CCc3cccc[n]->13)<-[n]3ccccc3CCc3cccc[n]->23"
    mol = rx.parse_smiles(smiles)
    retained = rx.metal(mol, "OCT")
    assert len(retained) == 3
    for iso in retained:
        _assert_embeds_as(iso)
    unrestricted = rx.metal(mol, "OCT", screen=False)
    assert len(unrestricted) == 5
    for iso in unrestricted:
        stated = rx.metal(rx.cxsmiles(iso))
        assert len(stated) == 1
        assert rx.cxsmiles(stated[0]) == rx.cxsmiles(iso)
    bridge = next(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == S
        and all(
            atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
    )
    assert len(rx.metal(mol, "OCT", fix={(bridge.GetBeginAtomIdx(), bridge.GetEndAtomIdx()): 1.5})) == 5
    certificate = metal_enumeration._euclidean_conflict
    monkeypatch.setattr(metal_enumeration, "_euclidean_conflict", lambda matrix, **_kwargs: certificate(matrix))
    assert len(rx.metal(mol, "OCT")) == 5


def test_interval_euclidean_certificate_preserves_degenerate_eigenspaces():
    groups = np.arange(9) // 3
    squared = np.where(groups[:, None] == groups[None, :], 4.0, 1.21)
    np.fill_diagonal(squared, 0.0)
    lower, upper = np.sqrt(0.65 * squared), np.sqrt(1.35 * squared)
    rng = np.random.default_rng(42)
    for _ in range(20):
        order = rng.permutation(9)
        intervals = np.tril(lower[np.ix_(order, order)], -1) + np.triu(upper[np.ix_(order, order)], 1)
        assert DistanceGeometry.DoTriangleSmoothing(intervals.copy())
        assert metal_enumeration._euclidean_conflict(intervals) == pytest.approx(0.2995, abs=1e-12)
    for invalid in (-0.1, np.nan, np.inf):
        intervals[1, 0] = invalid
        assert metal_enumeration._euclidean_conflict(intervals) is None


def test_trans_reach_screen_uses_the_shared_150_degree_slot_boundary():
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.C.C.C.[He]")
    iso = SimpleNamespace(graph=mol, metal=4, vertices=(0, 1, 2, 3), haptic={}, geometry="square_pyramidal")
    compiled = Constraints(distances={(donor, 4): (2.0, 2.1) for donor in range(4)})
    # the trans slot boundary at each donor's lower M-L bound (2.0, not the upper 2.1), widened by SPAN_TOL
    boundary = law_of_cosines(2.0, 2.0, slots.TRANS_ANGLE) - slots.SPAN_TOL

    reach = np.full((5, 5), 10.0)
    reach[1, 3] = reach[3, 1] = boundary - 0.02
    assert metal_enumeration._trans_span_conflict(iso, reach, compiled) is not None

    reach[1, 3] = reach[3, 1] = boundary + 0.02
    assert metal_enumeration._trans_span_conflict(iso, reach, compiled) is None


def test_compiled_span_uses_local_triangle_before_global_certificate(monkeypatch):
    from types import SimpleNamespace

    mol = Chem.MolFromSmiles("C.[He].[He].[He].[He]")
    constraints = Constraints(
        metals={4},
        distances={(0, 4): (2.0, 2.0), (2, 4): (2.0, 2.0)},
        angles={(0, 4, 2): (136.0, 152.0)},
    )
    iso = SimpleNamespace(graph=mol, metal=4, donors=(0, 2))
    native = np.full((5, 5), np.inf)
    native[2, 0], native[0, 2] = 0.0, 2.5
    monkeypatch.setattr(
        metal_enumeration,
        "coordination_reach",
        lambda *_args, **_kwargs: pytest.fail("the local contradiction should short-circuit the global certificate"),
    )

    assert "native ligand reach" in metal_enumeration._compiled_reach_conflict(iso, native, constraints, native, {})


def test_tris_dien_lanthanum_stays_under_the_orbit_cap():
    """A tris-dien La(III) sphere (CN9, all one fragment) enumerates without tripping the exact-orbit cap."""
    smiles = "C1C[NH]2CC[NH2]->[La+3]<-23456(<-[NH2]1)(<-[NH2]CC[NH]->3CC[NH2]->4)<-[NH2]CC[NH]->5CC[NH2]->6"
    mol = rx.parse_smiles(smiles)

    isomers = rx.metal(mol)

    assert len(isomers) == 62
    with pytest.raises(ValueError, match="more than 1,000 distinct constitutional"):
        rx.metal(mol, screen=False)


def test_tethered_haptic_faces_reject_an_unreachable_trans_state():
    smiles = "CC#[N]->[Ru+2]123(<-[Cl-])(<-[Cl-])(<-[N]#CC)<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31"

    def conflict(iso):
        reach = ligand_reach(iso.length_mol)
        native = coordination_reach_base(iso.graph, reach, {iso.metal})
        links = chelate_links(iso.graph, iso.vertices, iso.haptic)
        return metal_enumeration._reach_conflict(iso, reach, links, native, {})

    isomers = rx.metal(smiles, "octahedral", screen=False)
    trans, cis = isomers[0], isomers[1]
    seed = rx.embed(cis, n=1, seed=42, threads=1).mol
    trans.length_mol.AddConformer(Chem.Conformer(seed.GetConformer(0)))

    assert len(isomers) == 6
    assert "haptic faces" in conflict(trans)
    assert "haptic faces" in conflict(rx.metal(smiles, "octahedral", screen=False)[0])


@pytest.mark.parametrize("measured", [True])
@pytest.mark.parametrize("reordered", [False])
def test_haptic_span_keeps_realizable_member_angles(measured, reordered):
    from types import SimpleNamespace

    positions = np.array([[2, 0.7, 0], [2, -0.7, 0], [-2, 0.7, 0], [-2, -0.7, 0], [0, 0, 0]])
    mol = Chem.MolFromSmiles("C.C.C.C.[Mo]")
    metal, left, right = 4, (0, 1), (2, 3)
    if reordered:
        mol = Chem.RenumberAtoms(mol, [4, 3, 2, 1, 0])
        positions = positions[::-1, [1, 2, 0]] + np.array([1.0, 2.0, 3.0])
        metal, left, right = 0, (4, 3), (2, 1)
    conformer = Chem.Conformer(5)
    conformer.SetPositions(positions)
    mol.AddConformer(conformer)
    iso = SimpleNamespace(
        haptic={5: left, 6: right}, length_mol=mol, metal=metal, lengths="input" if measured else "model"
    )
    reach = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    cons = Constraints(distances={(metal, 5): (2.0, 2.0), (metal, 6): (2.0, 2.0)})
    for atom in (*left, *right):
        radius = float(np.linalg.norm(positions[atom] - positions[metal]))
        cons.distances[tuple(sorted((metal, atom)))] = (radius, radius)

    # The centroids are trans, but their individual member rays are not.
    assert metal_enumeration._haptic_span_conflict(iso, reach, 5, 6, 180.0, cons) is None
    for a, b in itertools.product(left, right):
        reach[a, b] = reach[b, a] = 3.5
    assert "centroid reach" in metal_enumeration._haptic_span_conflict(iso, reach, 5, 6, 180.0, cons)


@pytest.mark.parametrize(
    ("smiles", "geometry", "count"),
    [
        ("[Cl-]->[Cu+2]1<-n2cccc3ccc4ccc[n]->1c4c32", "trigonal_planar", 1),
    ],
    ids=["cu_trigonal_planar"],
)
def test_fused_chelate_is_screened_at_its_bite_on_a_trigonal_site(smiles, geometry, count):
    """A fused (phenanthroline-like) 5-ring chelate is screened at its native backbone bite
    (`metal_slots.chelate_bite_window`), not the ideal 120 vertex angle (`screen=False` gives 1 and 3).
    """
    isomers = rx.metal(smiles, geometry)
    assert len(isomers) == count
    for iso in isomers:
        _assert_embeds_as(iso)


_POCOP_NI = "COC(=O)c1cc2O[P](C(C)C)(C(C)C)->[Ni+2]3(<-[Cl-])<-[c-]2c(O[P]->3(C(C)C)C(C)C)c1"


def test_a_failed_ligand_reach_warns_that_no_arrangement_is_screened(monkeypatch, caplog):
    """Without native reach the pincer keeps its unreachable cis state, so the result must say it is unscreened."""
    import logging

    def inconsistent(_mol):
        raise ValueError("native ligand reach bounds are inconsistent")

    monkeypatch.setattr(metal_enumeration, "ligand_reach", inconsistent)
    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        assert len(rx.metal(_POCOP_NI, "square_planar")) == 2
    assert "native ligand reach failed (native ligand reach bounds are inconsistent)" in caplog.text


def test_screened_chelate_with_vacant_sites_enumerates_every_site_arrangement():
    """En, Cl and Br on a pentagonal bipyramid leave three vacant sites, so the raw pool exceeds 4!."""
    assert len(rx.metal("[NH2]1CC[NH2]->[Mo+3]<-1(<-[Cl-])<-[Br-]", "PBP", stereo="free")) == 30


@pytest.mark.parametrize("stereo", ["bogus"])
def test_core_enumeration_rejects_an_unknown_stereo_mode(stereo):
    with pytest.raises(ValueError, match="unknown stereo mode"):
        rx.enumerate_isomers(rx.parse_smiles(_MA2B2), stereo=stereo)


@pytest.mark.parametrize("screen", [True])
def test_multimetal_numeric_fix_bypasses_ground_state_reach_screen(screen, caplog):
    import logging

    smiles = "CC(C)(C)[P]1(C(C)(C)C)C(C)(C)C[H]->[Pd+2]<-1(<-[Br-])<-[c-]1cscn1"
    first = rx.embed(rx.metal(smiles, "SPL")[0], n=1, seed=42, threads=1).mol
    second = rx.embed(rx.metal("N->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "SPL")[0], n=1, seed=42, threads=1).mol
    combined = Chem.CombineMols(first, second, Point3D(8, 0, 0))
    hydrogen = next(
        a for a in first.GetAtoms() if a.GetAtomicNum() == 1 and any(n.GetAtomicNum() == 46 for n in a.GetNeighbors())
    )
    carbon = next(a for a in hydrogen.GetNeighbors() if a.GetAtomicNum() == 6)

    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        ordinary = rx.metal(combined, center="all", stereo="free", screen=screen)
    assert len(ordinary) == 3
    assert len({rx.cxsmiles(iso) for iso in ordinary}) == len(ordinary)
    assert ("multi-metal sphere has no reach certificate or edge rule" in caplog.text) == screen
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        isomers = rx.metal(
            combined, center="all", stereo="free", fix={(carbon.GetIdx(), hydrogen.GetIdx()): 2.0}, screen=screen
        )
    assert len(isomers) == 3
    assert ("fix= is set" in caplog.text) == screen


def test_observed_only_requires_coordinates():
    with pytest.raises(ValueError, match="observed_only=True requires an input conformer"):
        rx.metal(_MA2B2, observed_only=True)


@pytest.mark.parametrize(
    ("smiles", "geometry"),
    [
        ("N->[Pd+2](<-[Cl-])<-[Br-]", "SPL"),
        ("N->[Zn+2](<-[Cl-])<-[Br-]", "TET"),
        ("N->[Co+3](<-N)(<-N)(<-[Cl-])<-[Br-]", "OCT"),
        ("N->[Fe+2](<-N)(<-[Cl-])<-[Br-]", "TBP"),
        ("N->[Pd+2](<-N)(<-N)<-[Cl-]", "SPY"),
    ],
    ids=[
        "t-shaped-pd-in-square-plane",
        "pyramidal-zn-in-tetrahedron",
        "co-in-octahedron",
        "fe-in-bipyramid",
        "pd-in-square-pyramid",
    ],
)
def test_embedded_vacancy_isomers_write_their_own_cx(smiles, geometry):
    """A vacancy request reads back on its occupied vertices, not as the full-polyhedron record at that CN.

    A T-shape ties a square plane with a vertex empty, so it must still write SPL. Each other request has a
    mirror pair that only the vacant vertex tells apart, and the square pyramid has an empty apex.
    """
    for iso in rx.metal(smiles, geometry):
        embedded = rx.embed(iso, n=1, seed=42, threads=1).mol
        assert rx.cxsmiles(embedded) == rx.cxsmiles(iso)


@pytest.mark.parametrize(
    ("smiles", "geometry", "count"),
    [("[Cl-]->[Zn+2]<-[Br-]", "TPY", 1), ("N->[Fe+2](<-[Cl-])<-[Br-]", "SEE", 9)],
    ids=["bent-zncl-br", "seesaw-fe-with-one-empty-site"],
)
def test_donors_coplanar_with_the_metal_have_no_vacancy_mirror_isomer(smiles, geometry, count):
    """Two donors, or a seesaw's two axial donors and one equatorial, share a plane with the metal, so the
    mirror image through that plane is the same molecule with its empty site on the other side.
    """
    assert len(rx.metal(smiles, geometry)) == count


# --- vertex-derived permutation pools -------------------------------------------------------------------


# --- the template graft over a coordination sphere -------------------------------------------------------


def _cis_reference(n=2):
    """Embed the MA2B2 pair and return `(cis ensemble, its positions, {symbol: [idx]})`."""
    pair = rx.embed(_MA2B2, metal="square_planar", n=n)
    cis = next(e for e in pair if e.tag["label"] == "cis")
    where: dict = {}
    for a in cis.mol.GetAtoms():
        where.setdefault(a.GetSymbol(), []).append(a.GetIdx())
    return cis, cis.mol.GetConformer(cis.ids[0]).GetPositions(), where


def test_graft_over_the_coordination_sphere_is_refused():
    ref, pos, where = _cis_reference()
    sphere = [where["Pd"][0], *where["Cl"], *where["N"]]
    for core in ([*where["Cl"]], sphere):
        with pytest.raises(ValueError, match="coordination-sphere atoms"):
            rx.embed(_MA2B2, metal="square_planar", n=2, template=(ref.mol, {i: i for i in core}))
    iso = next(iter(rx.metal(_MA2B2, "square_planar")))
    with pytest.raises(ValueError, match="coordination-sphere atoms"):
        rx.embed(iso, n=2, fix={i: tuple(pos[i]) for i in where["Cl"]})


def test_retained_source_graft_authority_is_independent_of_constraint_compilation():
    from rxembed.embed import prepare

    ref, pos, where = _cis_reference(n=1)
    source = ref.mol
    fix = {i: tuple(pos[i]) for i in [*where["Cl"], where["N"][0]]}
    for spec in (source, from_geometry(source)):
        _, cons, _, graft = prepare(spec, fix=fix)
        assert set(fix) <= cons.frozen
        assert set(graft) == set(fix)
    for selected in rx.metal(source):
        with pytest.raises(ValueError, match="coordination-sphere atoms"):
            prepare(selected, fix=fix)
    ens = rx.embed(source, fix=fix, contacts={(0, 2): (1.0, 5.0)}, n=1, seed=42, threads=1)
    assert ens.cons.contacts[0] == {(0, 2)}
    actual = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    for left, right in itertools.combinations(fix, 2):
        assert np.linalg.norm(actual[left] - actual[right]) == pytest.approx(
            np.linalg.norm(pos[left] - pos[right]), abs=1e-6
        )
    assert rx.cxsmiles(ens.mol) == rx.cxsmiles(source)


# --- haptic faces: one vertex, a transient centroid -------------------------------------------------------


@pytest.mark.skipif(find_spec("openconf") is None, reason="openconf not installed")
def test_haptic_complex_mc_search_warns_and_keeps_the_seeded_conformers(caplog):
    """openconf's pose generator refuses a haptic centroid mol ("changed the atom set"); mc() must warn
    instead of raising and must leave the seeded conformers exactly as they were.
    """
    ens = rx.embed(rx.metal(ferrocene())[0], n=3, seed=1)
    seeded = list(ens.ids)

    with caplog.at_level("WARNING", logger="rxembed"):
        ens.mc()

    assert list(ens.ids) == seeded
    assert any("openconf could not search this system" in r.getMessage() for r in caplog.records)


def test_haptic_spectator_uses_the_same_centroid_constraints():
    combined = Chem.CombineMols(ferrocene(), ferrocene(), Point3D(6, 0, 0))
    iso = rx.metal(combined, center=0, stereo="free")[0]
    expected = list(range(iso.mol.GetNumAtoms(), iso.mol.GetNumAtoms() + 4))
    assert sorted(iso.cons.phantoms) == sorted(iso.cons.haptic) == expected
    assert [sum(isinstance(site, HapticSite) for site in state.vertices) for state in iso.centres] == [2, 2]
    assert core_embed(iso, n=1, seed=1).ids
    positions = combined.GetConformer().GetPositions()
    second = combined.GetNumAtoms() // 2
    positions[second + 1 :] = positions[second] + 1.1 * (positions[second + 1 :] - positions[second])
    combined.GetConformer().SetPositions(positions)
    assert rx.metal(combined, center=0, stereo="free")[0].cons == iso.cons


@pytest.mark.parametrize("door", ["from_geometry"], ids=["from_geometry"])
def test_half_sandwich_uses_piano_stool_shape(door):
    iso = rx.metal(cp_ticl3())[0] if door == "enumerate" else from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1


def test_empty_isomer_enumeration_from_a_pruned_pool_names_the_screen_remedy(caplog):
    """`linear` has no hull edge at all, so `chelate_edge_links` prunes every streamed order before the
    per-candidate reach screen ever runs; the warning must still name the `screen=False` remedy, not the
    generic message that `unreachable` alone used to gate.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="rxembed.metal"):
        out = rx.metal("[Cu+]1<-[NH2]CC[NH2]->1", "linear")
    assert not out
    warnings = [r.getMessage() for r in caplog.records if "no model-compatible arrangement" in r.getMessage()]
    assert warnings, caplog.text
    # screen=False is the wrong remedy for a model-length contradiction (it returns non-embeddable states);
    # lengths='input' must be named too.
    assert "screen=False" in warnings[0]
    assert "lengths='input'" in warnings[0]


def test_six_ring_pincer_opens_to_a_trigonal_bipyramid_equator():
    """PIVPOB's SNS pincer (a rigid 6-ring bite each side) reads 105-108 deg at its equator, above the 74-104
    deg ring-size census. The census-only window's ideal-clamped anchor misreads square_pyramidal, so a
    census-only wall refused this isomer outright. The gap rule (`metal_constraints.bounded_bites`) instead
    opens a box from the census edge out to the backbone-triangle reach, which this arrangement's relaxed
    shell does read as trigonal_bipyramidal, so it is enumerated and embeds.
    """
    smiles = "[O-2]->[V+5]12(<-[O-2])<-[S-]CCc3cccc(CC[S-]->1)[n]->23"
    isomers = rx.metal(smiles, "trigonal_bipyramidal")
    mol = isomers[0].graph  # one shared graph; atom symbols identify the pincer regardless of vertex order
    equatorial = set(dict(record("trigonal_bipyramidal").site_groups)["equatorial"])

    def is_pincer_equatorial(iso):
        pincer = {v for v, d in enumerate(iso.vertices) if d != VACANT and mol.GetAtomWithIdx(d).GetSymbol() != "O"}
        return pincer == equatorial

    matches = [iso for iso in isomers if is_pincer_equatorial(iso)]
    assert len(matches) == 1

    ensemble = rx.embed(matches[0], n=1, seed=42, threads=1)
    assert ensemble.n == 1


def _berry_intermediate_positions(t):
    """Return 5 unit vectors on the SPY->TBP Berry pseudorotation path at parameter `t` (0 = SPY, 1 = TBP).

    Spherical-linear interpolation, slot by slot, between the two idealized templates' own `vertex_dirs`.
    Not the literature pivot/turnstile pairing (this codebase's own fit is seating-invariant, so any
    continuous path between the two templates crosses their residual tie the same way); only the residual
    gap it lands at matters here.
    """
    spy = np.array(record("square_pyramidal").vertex_dirs, float)
    tbp = np.array(record("trigonal_bipyramidal").vertex_dirs, float)
    spy /= np.linalg.norm(spy, axis=1, keepdims=True)
    tbp /= np.linalg.norm(tbp, axis=1, keepdims=True)

    def slerp(a, b):
        theta = np.arccos(np.clip(a @ b, -1.0, 1.0))
        if theta < 1e-9:  # SPY and TBP already share this slot's direction (both templates' axial/apex)
            return a
        return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)

    pts = np.array([slerp(spy[i], tbp[i]) for i in range(5)])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_berry_pseudorotation_intermediate_reads_back_in_its_requested_frame():
    """A five-coordinate [FeCl5]2- Berry pseudorotation intermediate, close enough past the SPY/TBP fit-residual
    crossover that trigonal_bipyramidal reads as the argmin (0.224) with square_pyramidal a near-tie runner-up
    (0.228), inside `_FIT_MARGIN`: rule B accepts the requested square_pyramidal, where the pre-rule-B gate
    could only ever publish a strict argmin match or raise. `rx.metal(..., geometry='square_pyramidal',
    observed_only=True)` must still round-trip it, since `observed_only` shares the acceptance gate's rule B
    predicate, not a strict argmin equality. Fixed geometry, no embed and no seed: the premise does not depend
    on where a relax happens to land.
    """
    mol = Chem.MolFromSmiles("[Cl-]->[Fe+3](<-[Cl-])(<-[Cl-])(<-[Cl-])<-[Cl-]")
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Fe")
    donors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"]
    bond_length = 2.3  # Å, an ordinary Fe(III)-Cl distance; the reading depends only on direction, not scale
    positions = _berry_intermediate_positions(0.32)  # past the ~0.312 crossover, TBP wins by ~0.003 < _FIT_MARGIN

    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetAtomPosition(metal, Point3D(0.0, 0.0, 0.0))
    for donor, direction in zip(donors, positions, strict=True):
        conformer.SetAtomPosition(donor, Point3D(*(bond_length * direction)))
    mol.AddConformer(conformer, assignId=True)

    requested, next_name, next_residual, accepted = shape_gap(mol, metal, donors, {}, "square_pyramidal")
    assert next_name == "trigonal_bipyramidal"
    assert next_residual < requested, "premise: this shell's argmin is the OTHER shape, not the requested one"
    assert accepted, "rule B must accept the near-tie requested shape"

    reread = rx.metal(mol, geometry="square_pyramidal", observed_only=True)
    assert len(reread) >= 1
    assert reread[0].geometry == "square_pyramidal"
