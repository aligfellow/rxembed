"""Test coordination and vacant-site constraint construction."""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import metal_core, stereo
from rxembed.bounds import bounds_matrix
from rxembed.embed import _coordination_state_failure
from rxembed.metal_constraints import (
    CoordinationSphere,
    _bite_window_and_reach,
    coordination,
)
from rxembed.metal_core import MetalState, metal_indices
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer
from rxembed.metal_perceive import classify_geometry
from rxembed.metal_polyhedron import POLYHEDRA
from rxembed.metal_slots import SPAN_TOL, chelate_bite_window
from rxembed.pipeline.ensemble import Ensemble
from tests.metal_fixtures import ferrocene

_TRIDENTATE_CD = "[NH2]1CC[NH]2CC[NH2]->[Cd+2](<-[Cl-])(<-[Cl-])<-1<-2"


def _two_centre_ensemble(primary, secondary):
    """Build two ideal, separated coordination spheres on one conformer."""
    rw = Chem.RWMol()
    states = []
    for geometry in (primary, secondary):
        metal = rw.AddAtom(Chem.Atom(6))
        donors = tuple(rw.AddAtom(Chem.Atom(7)) for _ in POLYHEDRA[geometry].vertex_dirs)
        states.append(MetalState(metal, 46, 2, geometry, donors))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for origin, state in zip((np.zeros(3), np.array([8.0, 0.0, 0.0])), states, strict=True):
        conf.SetAtomPosition(state.atom, Point3D(*origin))
        for donor, direction in zip(state.vertices, POLYHEDRA[state.geometry].vertex_dirs, strict=True):
            unit = np.asarray(direction, dtype=float)
            conf.SetAtomPosition(donor, Point3D(*(origin + 2.0 * unit / np.linalg.norm(unit))))
    mol.AddConformer(conf)
    iso = Isomer.from_state(mol, states)
    return Ensemble(mol, [0], iso=iso), iso


# --- the polyhedron angles are realised, not merely stated -----------------------------------------------

# One real complex per low-CN shape: a 14-electron T-shaped Rh(I) phosphine (its two P trans, Cl the stem) and
# Fe(CO)4 (the 16e d8 C2v sawhorse). The CN3 pyramid needs its own hold and has its own test below.


# A bare CN3 pyramid's ±8° D-M-D window is flat-bottomed, with no restoring force once a phosphine rides the
# wall into trigonal_planar's basin, so `mechanisms.Umbrella` holds the scale-free improper instead. Two
# fixtures: the trimethyl case is cheap, and PPh3's DG seed comes out exactly planar, proving the hold
# re-forms a pyramid rather than only keeping one.


# PPh3 alone: its DG seed comes out exactly planar, so it proves the hold RE-FORMS a pyramid rather than
# only keeping one. The PMe3 case took the same branch from an already-pyramidal seed.


def test_flattened_pyramid_is_rejected():
    ens = rx.embed("C[P](C)(C)[Fe]([P](C)(C)C)[P](C)(C)C", metal="TPY", n=4, seed=7)[0]
    iso, mol = ens.iso, ens._mol  # `_mol`: `.mol` hands back a metal-restored COPY, which the edits below lose
    verts = list(iso.vertices)
    assert ens.n >= 1
    assert [classify_geometry(mol, iso.metal, verts, c) for c in ens.ids] == ["trigonal_pyramidal"] * ens.n
    for cid in ens.ids:  # push the metal onto its donor plane: the geometry the warning exists to report
        conf = mol.GetConformer(cid)
        pos = conf.GetPositions()
        conf.SetAtomPosition(iso.metal, Point3D(*np.mean([pos[v] for v in verts], axis=0)))
    failures = ens._acceptance_failures()
    assert len(failures) == 1
    failure = next(iter(failures))
    assert failure.kind == "coordination_shape"
    assert str(failure).endswith("donors fall into a plane")
    assert next(iter(failures.values())) == ens.ids


def test_secondary_planar_centre_is_checked():
    ens, iso = _two_centre_ensemble("tetrahedral", "square_planar")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    donor = secondary.vertices[0]
    conf.SetAtomPosition(donor, Point3D(*(pos[donor] + np.array([0.0, 0.0, 2.0]))))
    assert _coordination_state_failure(ens._mol, 0, iso) is not None, "the puckered secondary plane passed the gate"


def test_donor_orientation_walls_cannot_collapse_two_protons_together():
    iso = rx.metal("[NH2](c1ccccc1)->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]", "square_planar")[0]
    donor = next(atom for atom in iso.mol.GetAtoms() if atom.GetAtomicNum() == 7)
    protons = sorted(neighbor.GetIdx() for neighbor in donor.GetNeighbors() if neighbor.GetAtomicNum() == 1)
    floor = float(bounds_matrix(iso.mol)[protons[1], protons[0]])

    assert iso.cons.floors[tuple(protons)] == pytest.approx(floor)
    ens = rx.embed(iso, n=1, seed=42)
    pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
    # A floor is a one-sided wall, not a target: the relax settles a measured 0.0035 A inside it at this
    # seed, within geom_check's own floor slop, so the assertion tolerance matches that undershoot.
    assert np.linalg.norm(pos[protons[0]] - pos[protons[1]]) >= floor - 0.004
    assert ens.check()[ens.ids[0]].ok()


# --- the chelate bite comes from the backbone, not the polyhedron ---------------------------------------


def test_tridentate_fan_compiles_shared_targets_and_survives_cleanup():
    """A hinge donor at the apex bites both trans basal arms; every compiled pull stays inside its own window
    and the four basal donors it targets are coplanar.
    """
    iso = next(
        candidate
        for candidate in rx.metal(_TRIDENTATE_CD, "square_pyramidal")
        if candidate.vertices[0] == 3 and set(candidate.vertices[1::2]) == {0, 6}
    )
    targets = {key: value for key, value in iso.cons.pulls.items() if len(key) == 3}
    assert len(targets) == 10
    for key, value in targets.items():
        lo, hi = iso.cons.angles[key] if key in iso.cons.angles else iso.cons.angles[key[::-1]]
        assert lo - 1e-8 <= value <= hi + 1e-8

    # Rebuild each basal donor's 3D point from its compiled M-metal distance and pairwise pull angle (the
    # metal at the origin), then check the four points share one plane independent of the metal's own height.
    basal = [0, 8, 6, 9]
    radius = {d: sum(iso.cons.distances[tuple(sorted((d, iso.metal)))]) / 2 for d in basal}
    gram = np.zeros((len(basal), len(basal)))
    for i, donor in enumerate(basal):
        gram[i, i] = radius[donor] ** 2
    for i, j in itertools.combinations(range(len(basal)), 2):
        a, b = basal[i], basal[j]
        angle = targets.get((a, iso.metal, b), targets.get((b, iso.metal, a)))
        gram[i, j] = gram[j, i] = radius[a] * radius[b] * np.cos(np.radians(angle))
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:3]
    points = eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0, None))
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    assert spread[-1] < 0.05 * spread[0], "the four basal donors must lie in one plane"


def test_planar_diamine_cleanup_retains_targets_bonds_and_fresh_cx(tmp_path):
    iso = rx.metal("N[Ni]1NCCN1", "trigonal_planar")[0]
    targets = {key: value for key, value in iso.cons.pulls.items() if len(key) == 3}
    assert len(targets) == 3
    ens = rx.embed(iso, n=1, seed=42).minimize()
    assert ens.n == 1
    assert not ens.unrelaxed
    assert {key: value for key, value in ens.cons.pulls.items() if len(key) == 3} == targets
    assert all(check.ok for check in ens.check().values())
    path = tmp_path / "planar_diamine.xyz"
    Chem.MolToXYZFile(ens.mol, str(path))
    fresh = rx.read_xyz(
        str(path), charge=Chem.GetFormalCharge(ens.mol), connectivity="rdkit", bond_orders="xyz2mol", fallback=False
    )
    assert rx.cxsmiles(fresh) == rx.cxsmiles(ens.mol) == rx.cxsmiles(iso)
    edges = [{frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()} for mol in (fresh, ens.mol)]
    assert edges[0] == edges[1]


def test_norbornadiene_nickel_dicarbonyl_keeps_its_carbon_bonds():
    """A haptic diene's bite pair has no ring-size census prior to intersect with, so `_bite_window_and_reach`
    must replace the plain ideal +- 8 deg row outright wherever the native ligand reach excludes part of it.
    Today the bicycle's bridging C-C bond stretches to 1.148x RDKit's own metal-free upper bound.
    """
    smiles = (
        "[O+]#[C-]->[Ni]123(<-[C-]#[O+])<-[CH]4=[CH]->1[C@H]1C[C@@H]4[CH]->2=[CH]->31 "
        "|atomProp:1.atomNote.s0:2.atomNote.TET:3.atomNote.s1:5.atomNote.s2:6.atomNote.s2:10.atomNote.s3:11.atomNote.s3|"
    )
    iso = rx.metal(smiles)[0]
    ensemble = rx.embed(iso, n=1, seed=42, threads=1)
    positions = ensemble.mol.GetConformer(ensemble.ids[0]).GetPositions()
    ratio, label = _worst_ligand_bond_ratio(ensemble.mol, positions)
    assert ratio < 1.05, f"{label} {ratio}"


def test_eta4_naphthalene_hinge_folds_its_fusion_carbons_off_the_metal():
    """BAMROX: an eta4-bound naphthalene folds at its hinge, standing the unbound fusion carbons off Rh.

    Without the hinge push the fusion carbons sit at 1.11 to 1.17x the Rh-C covalent sum (below the census p5
    of 1.20 for a conjugated hinge); the push must clear that floor at every seed.
    """
    smi = "c1ccc([P]2(CCO[c]34->[Rh+]<-2567<-[cH]3[cH]->5[c]->6(OCC[P]->7(c2ccccc2)c2ccccc2)c2ccccc24)c2ccccc2)cc1"
    iso = rx.metal(smi)[0]
    r_rh_c = Chem.GetPeriodicTable().GetRcovalent(45) + Chem.GetPeriodicTable().GetRcovalent(6)
    for seed in (42, 7, 1234, 2026, 99):
        ens = rx.embed(iso, n=1, seed=seed)
        mol, pos = ens.mol, ens.mol.GetConformer(ens.ids[0]).GetPositions()
        rh = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Rh")
        face = {n.GetIdx() for n in mol.GetAtomWithIdx(rh).GetNeighbors() if n.GetSymbol() == "C"}
        rings = [set(r) for r in Chem.GetSymmSSSR(metal_core.ligand_graph(mol))]  # metal-stripped: no false hinge at Rh
        fusion_carbons = {
            x.GetIdx()
            for d in face
            for x in mol.GetAtomWithIdx(d).GetNeighbors()
            if x.GetIdx() not in face and any({d, x.GetIdx()} <= r for r in rings)
        }
        assert fusion_carbons, f"seed {seed}: no unbound fusion carbon beside the eta4 face"
        for x in fusion_carbons:
            ratio = float(np.linalg.norm(pos[x] - pos[rh])) / r_rh_c
            assert ratio >= 1.20, f"seed {seed}: fusion carbon {x} at {ratio:.3f}x the Rh-C covalent sum"


def test_chelate_bite_window_intersects_a_partial_rdkit_reach_window():
    """`_bite_window_and_reach` intersects the ring-size census with the backbone reach triangle for `window`,
    and reports the wider triangle back as `reach` for `bounded_bites`'s gap box.
    """
    mol = Chem.MolFromSmiles("NCCN")
    matrix = np.zeros((4, 4))

    def span(angle):
        return math.sqrt(8.0 - 8.0 * math.cos(math.radians(angle)))

    matrix[3, 0] = span(80.0) + SPAN_TOL
    matrix[0, 3] = span(100.0) - SPAN_TOL

    window, reach = _bite_window_and_reach(mol, (0,), (3,), (0, 3), matrix, (2.0, 2.0), (0.0, 180.0))

    assert window == pytest.approx((80.0, 91.0))
    assert reach == pytest.approx((80.0, 100.0))


def test_chelate_bite_window_reads_the_four_ring_census():
    """A donor pair 2 bonds apart (a carboxylate's O-C-O) takes the 4-ring bite census verbatim."""
    mol = Chem.MolFromSmiles("OCO")
    assert chelate_bite_window(mol, 0, 2) == (58.0, 81.0)


# --- a haptic bite reads the native ligand reach, same as a sigma bite (metal_constraints.seated_bites) ------


def _worst_ligand_bond_ratio(mol, positions, ref_positions=None):
    """Return the worst (ratio, label) over every non-hydrogen ligand bond, metal bonds excluded.

    Without `ref_positions` the reference is RDKit's own metal-free upper bound for that bond (a SMILES input
    has no measured length); with it, the reference is that bond's length in `ref_positions` (a crystal read).
    """
    metals = set(metal_indices(mol))
    rw = Chem.RWMol(mol)
    for bond in list(mol.GetBonds()):
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & metals:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    ligand = rw.GetMol()
    ligand.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(ligand)
    worst = (0.0, "")
    if ref_positions is None:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
    for bond in ligand.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if 1 in (ligand.GetAtomWithIdx(i).GetAtomicNum(), ligand.GetAtomWithIdx(j).GetAtomicNum()):
            continue  # RDKit's ordinary X-H bounds do not describe a bridging hydrogen
        length = float(np.linalg.norm(positions[i] - positions[j]))
        target = bounds[i][j] if ref_positions is None else float(np.linalg.norm(ref_positions[i] - ref_positions[j]))
        label = f"{ligand.GetAtomWithIdx(i).GetSymbol()}{i}-{ligand.GetAtomWithIdx(j).GetSymbol()}{j}"
        worst = max(worst, (length / target, label))
    return worst


def test_five_ring_cn_chelate_bites_admit_the_crystal_bite():
    """Three independent 5-ring C^N chelate bites on Ir each compile a window admitting the crystal's
    compressed 79.0/80.9/80.1 deg bites (FOPSOT).
    """
    smiles = "Cn1ccn2->[Ir+3]34(<-[Cl-])(<-[c-]5cc(F)ccc5-c5ccccn->35)<-[c-]3c(-c5ccccn->45)c(F)cc(F)c3-c12"
    iso = rx.metal(smiles)[0]
    bites = [
        window
        for (left, metal, right), window in iso.cons.angles.items()
        if metal == iso.metal
        and left not in iso.haptic
        and right not in iso.haptic
        and chelate_bite_window(iso.mol, left, right) is not None
    ]
    assert len(bites) == 3
    assert all(window[0] <= 79.0 <= window[1] for window in bites)


def test_three_centre_hydrogen_bridge_keeps_its_measured_xh_span():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(26))
    boron = rw.AddAtom(Chem.Atom(5))
    bridge = rw.AddAtom(Chem.Atom(1))
    nitrogen = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(boron, bridge, Chem.BondType.SINGLE)
    for donor in (boron, bridge, nitrogen):
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, point in enumerate(((0, 0, 0), (2.1, 0, 0), (1.4, 1.15, 0), (0, 2.0, 0))):
        conf.SetAtomPosition(atom, Point3D(*point))
    mol.AddConformer(conf)

    sphere = CoordinationSphere(
        mol,
        metal,
        26,
        (boron, bridge, nitrogen),
        {},
        POLYHEDRA["trigonal_planar"],
        pos=mol.GetConformer().GetPositions(),
    )
    cons = coordination(sphere)

    distance = np.linalg.norm(np.array((2.1, 0, 0)) - np.array((1.4, 1.15, 0)))
    assert cons.distances[(boron, bridge)] == pytest.approx((distance - 0.1, distance + 0.1))
    assert cons.pulls[(boron, bridge)] == pytest.approx(distance)


def test_large_atom_chain_is_not_forced_into_an_organic_chelate_bite():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(28))
    donors = [rw.AddAtom(Chem.Atom(33)) for _ in range(6)]
    caps = [rw.AddAtom(Chem.Atom(6)) for _ in donors]
    for i, donor in enumerate(donors):
        rw.AddBond(donor, donors[(i + 1) % len(donors)], Chem.BondType.SINGLE)
        rw.AddBond(donor, caps[i], Chem.BondType.SINGLE)
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal, Point3D(0.0, 0.0, 0.0))
    for i, (donor, cap) in enumerate(zip(donors, caps, strict=True)):
        direction = np.array((np.cos(i * np.pi / 3), np.sin(i * np.pi / 3), 0.0))
        conf.SetAtomPosition(donor, Point3D(*(2.5 * direction)))
        conf.SetAtomPosition(cap, Point3D(*(3.9 * direction)))
    mol.AddConformer(conf)

    isomers = rx.metal(mol, lengths="input", stereo="free")

    assert isomers
    assert isomers[0].geometry == "hexagonal_planar"

    coordinate_free = rx.metal(rx.cxsmiles(isomers[0]), stereo="free")[0]
    assert (
        min(
            coordinate_free.cons.distances[tuple(sorted((coordinate_free.metal, donor)))][1]
            for donor in coordinate_free.donors
        )
        > 2.4
    )


def test_trans_bidentate_keeps_the_polyhedron_window():
    iso = next(
        candidate
        for candidate in rx.metal("N1CCCCCN->[Pt+2](<-[Cl-])(<-[Cl-])<-1", "square_planar", stereo="free")
        if candidate.label == "trans"
    )
    nitrogens = [donor for donor in iso.donors if iso.mol.GetAtomWithIdx(donor).GetAtomicNum() == 7]

    assert iso.cons.angles[(nitrogens[0], iso.metal, nitrogens[1])] == (172.0, 180.0)


def test_coordinated_nh_imine_uses_requested_ez_donor_plane():
    source = Chem.AddHs(Chem.MolFromSmiles("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]"))
    for order in (list(range(source.GetNumAtoms())), list(reversed(range(source.GetNumAtoms())))):
        isomers = enumerate_isomers(Chem.RenumberAtoms(source, order), "square_planar")
        assert {stereo.bond_stereo(iso.stereo_label).popitem()[1] for iso in isomers} == {"E", "Z"}
        for iso in isomers:
            targets = stereo.metal_referenced_ez(iso.mol, iso.stereo_label, iso.donor_bonds)
            ((_, (donor, carbon, metal, _ref, _ligand_ref, wanted)),) = targets.items()
            restored = Chem.Mol(iso.mol)
            iso.restore(restored)
            restored = metal_core.connect_metal(restored, iso.donor_bonds)
            assert stereo.metal_referenced_ez(restored, iso.stereo_label, iso.donor_bonds) == targets
            rows = iso.cons.coplanar
            ((_, _, _, _ref, anchor, _cap),) = [row for row in rows if row[0] == metal]
            assert anchor == (180.0 if wanted == "E" else 0.0)
            assert {row[4] for row in rows if row[1:3] == (donor, carbon)} == {0.0, 180.0}


def test_fully_occupied_planar_chelate_gets_one_shape_level_plane_hold():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(47))
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    donors = [rw.AddAtom(Chem.Atom(8)) for _ in range(2)]
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    rw.AddBond(face[0], donors[0], Chem.BondType.SINGLE)
    rw.AddBond(face[1], donors[1], Chem.BondType.SINGLE)
    for atom in (*face, *donors):
        rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "trigonal_planar", {0: face[0], 1: donors[0], 2: donors[1]})

    assert set(iso.haptic.values()) == {tuple(face)}
    assert len(iso.cons.umbrellas) == 1
    assert next(iter(iso.cons.umbrellas)).count(iso.metal) == 1


# --- coordinate(): seating a substrate at a vacant vertex -------------------------------------------------


def test_coordinate_binds_a_substrate_at_the_vacant_site():
    es = rx.embed("CCCN[Pd](Cl)NCCC.O", metal="square_planar", coordinate="[OX2]", n=3, seed=1)
    for ens in list(es) if isinstance(es, rx.EnsembleSet) else [es]:
        m = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "Pd")
        o = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetSymbol() == "O")
        assert o in ens.iso.vertices
        assert (o, m) in ens.iso.donor_bonds
        assert o in ens.sphere[m]
        ens.minimize()
        assert ens.n >= 1
        assert all(report.ok() for report in ens.check().values())
        lo, hi = ens.cons.distances[(min(m, o), max(m, o))]
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            assert lo - 0.1 <= float(np.linalg.norm(pos[m] - pos[o])) <= hi + 0.1, "the substrate did not seat"


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ("donor", "already donors"),
        ("repeated", "must be distinct"),
    ],
)
def test_coordinate_rejects_invalid_identity_expansion(coordinate, message):
    iso = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "square_planar").select(index=0)
    oxygen = next(a.GetIdx() for a in iso.mol.GetAtoms() if a.GetSymbol() == "O")
    choice = {
        "donor": iso.donors[0],
        "metal": iso.metal,
        "missing": iso.mol.GetNumAtoms(),
        "repeated": [oxygen, oxygen],
    }[coordinate]
    with pytest.raises(ValueError, match=message):
        rx.embed(iso, coordinate=choice, n=1)


# ---------------------------------------------------------------------------------------------------------
# lengths=; WHERE the M-donor window is measured from
# ---------------------------------------------------------------------------------------------------------

_SQUARE_PD = "Cl[Pd](Cl)(N)N"  # two chemically equivalent Cl and two equivalent N: the tell, below


def _fake_geometry(smiles, seed=1):
    """The same graph carrying a PLAIN ETKDG conformer: a geometry produced with no M-L parameter at all."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return Chem.RemoveHs(mol)


def test_input_haptic_centroid_uses_the_input_distance_width():
    iso = rx.metal(ferrocene(), lengths="input")[0]
    windows = [
        window for pair, window in iso.cons.distances.items() if iso.metal in pair and set(pair) & iso.cons.phantoms
    ]

    assert windows
    assert all(hi - lo == pytest.approx(0.2) for lo, hi in windows)


def test_length_source_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        enumerate_isomers(_fake_geometry(_SQUARE_PD), "square_planar", lengths="input")
    said = [r for r in caplog.records if "M-donor windows from" in r.getMessage()]
    assert said
    assert "input conformer" in said[0].getMessage()
    assert "lengths='input'" in said[0].getMessage()
