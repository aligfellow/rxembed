"""Test coordination and vacant-site constraint construction."""

from __future__ import annotations

import itertools
import logging
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.constraints import FIX_ANGLE_TOL
from rxembed.metal_constraints import _CHELATE_BITE, _chelate_bite_window
from rxembed.metal_core import HapticSite, MetalState, classify_geometry
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer
from rxembed.metal_polyhedron import POLYHEDRA, _vertex_angle
from rxembed.pipeline import geom_check as geom
from rxembed.pipeline.ensemble import Ensemble

_WINDOW = 9.0  # deg: the coordination angle window is ±8°; 9 is that plus slack
_NI_N_CY = (  # the same N-bound isomer on a Cy2P-arene backbone
    "O=C1[O-]->[Ni+2]2(<-[N-](c3ccccc3)C1c1ccccc1)<-[P](Cc1ccccc1[P]->2(C1CCCCC1)C1CCCCC1)(C1CCCCC1)C1CCCCC1"
)
_BIS_EN_CO = "Cl[Co]12(Cl)(NCCN1)NCCN2"  # two en chelates + 2 Cl: intra- and inter-ligand pairs on one metal


def _realised(ens, iso, cid, i, j):
    """The vertex i-metal-vertex j angle realised in conformer `cid`."""
    pos = ens.mol.GetConformer(cid).GetPositions()
    return _vertex_angle(pos[iso.vertices[i]] - pos[iso.metal], pos[iso.vertices[j]] - pos[iso.metal])


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
    iso = Isomer._from_state(mol, states)
    return Ensemble(mol, [0], iso=iso), iso


# --- the polyhedron angles are realised, not merely stated -----------------------------------------------

# One real complex per low-CN shape: a 14-electron T-shaped Rh(I) phosphine (its two P trans, Cl the stem) and
# Fe(CO)4 (the 16e d8 C2v sawhorse). The CN3 pyramid has its own test below: it is the shape that used to lose.
_SHAPE_CASES = [
    ("t_shape", "CP(C)(C)->[Rh](Cl)<-P(C)(C)C"),
    ("seesaw", "[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"),
]


@pytest.mark.parametrize(("geometry", "smiles"), _SHAPE_CASES, ids=[c[0] for c in _SHAPE_CASES])
def test_shape_is_realised_by_a_real_embed(geometry, smiles):
    iso = rx.metal(smiles, geometry).select(index=0)
    ens = rx.embed(iso, n=4).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        for i, j, target in POLYHEDRA[geometry].resolved_angles:
            got = _realised(ens, iso, cid, i, j)
            assert abs(got - target) <= _WINDOW, f"{geometry} vertices {i}-{j}: {got}° vs {target}°"
        found = classify_geometry(ens.mol, iso.metal, list(iso.vertices), cid)
        assert found == geometry, f"seated {geometry}, embedded a {found}"


# The donor sets a bare CN3 pyramid used to lose on (13/49 conformers survived; every other shape 100%). The
# ±8° D-M-D window is flat-bottomed, so a phosphine rode the wall to 117.5°; inside trigonal_planar's basin,
# 2.5° away, and `mechanisms.Umbrella` now holds the scale-free improper instead. Two fixtures: the trimethyl
# case is the cheap one, and PPh3's DG seed comes out exactly planar, so it proves the hold RE-FORMS a pyramid
# rather than only keeping one.
_PPH3 = "P(c1ccccc1)(c1ccccc1)c1ccccc1"


# PPh3 alone: its DG seed comes out exactly planar, so it proves the hold RE-FORMS a pyramid rather than
# only keeping one. The PMe3 case took the same branch from an already-pyramidal seed.
def test_requested_pyramid_survives_relax():
    smiles = f"c1ccccc1P(c1ccccc1)(c1ccccc1)->[Pt](<-{_PPH3})<-{_PPH3}"
    iso = rx.metal(smiles, "trigonal_pyramidal").select(index=0)
    ens = rx.embed(iso, n=1, seed=7).minimize()
    assert ens.n >= 1, "Pt(PPh3)3: no conformer survived the gates"
    got = [classify_geometry(ens.mol, iso.metal, list(iso.vertices), cid) for cid in ens.ids]
    assert got == ["trigonal_pyramidal"] * len(got), f"Pt(PPh3)3: requested a pyramid, got {got}"


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
    assert failures == {"wrong coordination state": ens.ids}


def test_secondary_planar_centre_is_checked():
    ens, iso = _two_centre_ensemble("tetrahedral", "square_planar")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    donor = secondary.vertices[0]
    conf.SetAtomPosition(donor, Point3D(*(pos[donor] + np.array([0.0, 0.0, 2.0]))))
    assert not ens._coordination_ok(0, iso), "the puckered secondary square plane passed the final gate"


def test_secondary_nonplanar_centre_flattening_is_rejected():
    ens, iso = _two_centre_ensemble("square_planar", "trigonal_pyramidal")
    secondary = iso.centres[1]
    conf = ens._mol.GetConformer()
    pos = conf.GetPositions()
    conf.SetAtomPosition(secondary.atom, Point3D(*np.mean(pos[list(secondary.vertices)], axis=0)))
    assert not ens._coordination_ok(0, iso)


def test_one_haptic_site_does_not_define_a_nonplanar_state():
    ens, iso = _two_centre_ensemble("tetrahedral", "tetrahedral")
    secondary = iso.centres[1]
    face = HapticSite(tuple(secondary.vertices))
    sparse = secondary._replace(vertices=(face, None, None, None))
    iso = Isomer._from_state(iso.mol, (iso.centres[0], sparse))
    assert ens._coordination_ok(0, iso)


def test_infeasible_rigid_pyramid_is_rejected(caplog):
    iso = rx.metal("[Cu+]12<-n3ccccc3-c3cccc(n->13)-c1ccccn->21", "trigonal_pyramidal").select(index=0)
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        ens = rx.embed(iso, n=4, seed=7).minimize()
    assert ens.n == 0
    assert "arrangement may be infeasible" in caplog.text


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
@pytest.mark.parametrize(
    ("fixed", "first_present"),
    [
        ([1, 2, 5], True),
        ([1, 2, 5, 64], True),
        ([1, 2, 3, 5, 64], False),
    ],
    ids=["metal-fixed", "metal-donor-fixed", "complete-term-fixed"],
)
def test_partial_fix_drops_only_complete_carbonyl_terms(fixed, first_present):
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=fixed)[0]

    assert ((1, 64, 3) in iso.cons.angles) is first_present
    assert (1, 65, 4) in iso.cons.angles


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mnh_carbonyls_pass_the_relax_contract():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    iso = rx.metal(str(path), center="all", fix=[1, 2, 5])[0]
    ens = rx.embed(iso, n=1, seed=0xF00D)

    assert ens.ids
    assert not ens.unrelaxed
    conf = ens._mol.GetConformer(ens.ids[0])
    for atoms in ((1, 64, 3), (1, 65, 4)):
        assert ens.cons.angles[atoms] == (165.0, 180.0)
        assert GetAngleDeg(conf, *atoms) >= 165.0 - FIX_ANGLE_TOL
        pair = atoms[:2]
        value = float(np.linalg.norm(conf.GetPositions()[pair[0]] - conf.GetPositions()[pair[1]]))
        lo, hi = ens.cons.distances[pair]
        assert lo <= value <= hi


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_retained_geometry_uses_the_same_donor_orientation_compiler():
    from rxembed.metal_isomer import from_geometry

    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    mol = Chem.AddHs(rx.read_xyz(str(path)), addCoords=True)
    cons = from_geometry(mol, center="all").cons

    assert cons.angles[(1, 64, 3)] == (165.0, 180.0)
    assert cons.angles[(1, 65, 4)] == (165.0, 180.0)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_two_fixed_donors_do_not_own_their_angle_when_the_metal_is_free():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mnh.xyz"
    isos = rx.metal(str(path), center="all", fix=[6, 64])

    assert any(any(key[1] == 1 and {key[0], key[2]} == {6, 64} for key in iso.cons.angles) for iso in isos)


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_partial_frozen_mn_fe_state_passes_the_relax_contract():
    path = Path(__file__).parents[1] / "examples" / "structures" / "mn-h2.xyz"
    states = rx.metal(str(path), center="all", fix=[1, 5, 63, 64, 65, 66], stereo={"planar": "racemic"})
    iso = states.filter(center="Mn", label="mer", hand="lambda").select(center="Fe", haptic="Rₚ")
    ens = rx.embed(iso, n=1, seed=0xF00D)

    assert len(states) == 12
    assert ens.ids
    assert not ens.unrelaxed
    assert ens._metal_states() == {ens.ids[0]: True}
    assert ens.cons.angles[(1, 61, 3)] == (165.0, 180.0)
    assert ens.cons.angles[(1, 62, 4)] == (165.0, 180.0)
    conf = ens._mol.GetConformer(ens.ids[0])
    assert GetAngleDeg(conf, 1, 61, 3) >= 165.0 - FIX_ANGLE_TOL
    assert GetAngleDeg(conf, 1, 62, 4) >= 165.0 - FIX_ANGLE_TOL
    for pair in ((1, 61), (1, 62)):
        value = float(np.linalg.norm(conf.GetPositions()[pair[0]] - conf.GetPositions()[pair[1]]))
        lo, hi = ens.cons.distances[pair]
        assert lo <= value <= hi
    reference = iso.mol.GetConformer().GetPositions()
    frozen = sorted(ens.cons.frozen)
    drift = max(
        abs(
            np.linalg.norm(reference[i] - reference[j])
            - np.linalg.norm(conf.GetPositions()[i] - conf.GetPositions()[j])
        )
        for i, j in itertools.combinations(frozen, 2)
    )
    assert drift < 1e-12
    assert not ens._scan_connectivity()


# --- the chelate bite comes from the backbone, not the polyhedron ---------------------------------------


def _frag_of(mol):
    return {a: f for f, atoms in enumerate(Chem.GetMolFrags(mol)) for a in atoms}


def _states_an_intra_pair(iso):
    frag = _frag_of(iso.mol)
    return any(k[1] == iso.metal and frag[k[0]] == frag[k[2]] for k in iso.cons.angles)


def test_chelate_bite_uses_backbone_not_ideal():
    iso = next(i for i in rx.metal(_BIS_EN_CO, "octahedral") if _states_an_intra_pair(i))
    frag = _frag_of(iso.mol)
    ideal = {a for _i, _j, a in POLYHEDRA["octahedral"].resolved_angles}

    intra = {k: v for k, v in iso.cons.angles.items() if k[1] == iso.metal and frag[k[0]] == frag[k[2]]}
    assert intra, "the fixture must state at least one intra-chelate window"
    for k, window in intra.items():
        assert window == _CHELATE_BITE[5], f"{k}: an en 5-ring must be given its own bite, got {window}"
        assert not any(a - 8.0 <= window[0] and window[1] <= a + 8.0 for a in ideal), f"{k}: a polyhedron pad"

    inter = {k: v for k, v in iso.cons.angles.items() if k[1] == iso.metal and frag[k[0]] != frag[k[2]]}
    assert inter, "the fixture must also state an inter-ligand pair, or the contrast is untested"
    pads = {(max(0.0, a - 8.0), min(180.0, a + 8.0)) for a in ideal}  # ±8° about a tabulated angle, clamped
    for k, window in inter.items():
        assert window in pads, f"{k}: an inter-ligand pair must be a polyhedron angle ±8°, got {window}"

    # the window comes from the RING, so a pair with no shared backbone gets none at all
    cl = [d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "Cl"]
    assert _chelate_bite_window(iso.mol, cl[0], cl[1]) is None


def test_chelate_bite_is_not_reported_as_a_steric_clash():
    ens = rx.embed(rx.metal(_NI_N_CY, "square_planar")[0], n=1).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        rep = geom.check(ens.mol, cid)
        assert not [v for v in rep.violations if v.kind == "clash"], rep.summary()


def test_reachable_trans_chelate_embeds():
    iso = rx.metal("[Pd+2]1(<-[Cl-])(<-[Cl-])<-NCCCCN->1", "square_planar").filter(label="trans").select()
    assert rx.embed(iso, n=1, seed=1).n == 1


def test_side_on_eta2_ligand_embeds_geometry_clean():
    smi = "COC(=O)[C]12->[Ni+2]3(<-[O-]C(=O)C(c4ccccc4)[N-]->3c3ccccc3)<-[C]=1(C(=O)OC)C2(C)C(C)(C)C"
    ens = rx.embed(rx.metal(smi, "square_planar")[0], n=1).minimize()
    assert any(geom.check(ens.mol, c).ok() for c in ens.ids), "no geom.check-clean side-on conformer"


def test_haptic_face_and_sigma_donor_on_one_ligand_compile_without_a_virtual_bite_path():
    rw = Chem.RWMol()
    metal = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(metal).SetFormalCharge(2)
    face = [rw.AddAtom(Chem.Atom(6)) for _ in range(2)]
    linker = rw.AddAtom(Chem.Atom(6))
    donor = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(face[0], face[1], Chem.BondType.DOUBLE)
    rw.AddBond(face[1], linker, Chem.BondType.SINGLE)
    rw.AddBond(linker, donor, Chem.BondType.SINGLE)
    for atom in (*face, donor):
        rw.AddBond(atom, metal, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    iso = rx.Isomer(mol, "trigonal_planar", {0: face[0], 1: donor})
    assert iso.cons.haptic


def test_planar_haptic_umbrella_does_not_depend_on_the_centroid_slot():
    smiles = r"C/[CH]1=[CH](/F)->[Pt+2](<-[Cl-])(<-[Br-])(<-[NH3])<-1"
    isomers = rx.metal(smiles, "square_planar", stereo="free")

    assert len(isomers) == 3
    assert all(iso.cons.umbrellas for iso in isomers)


def test_rejected_seeds_are_replaced_to_n_clean(monkeypatch):
    ens = rx.embed(rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar")[0], n=2)
    donors = list(ens.iso.donors)
    workflow_failure = type(ens)._workflow_failure
    rejected = False

    def reject_one_seed(self, owner, cid):
        nonlocal rejected
        reason = workflow_failure(self, owner, cid)
        if not rejected and owner is self:
            rejected = True
            return "test rejection"
        return reason

    monkeypatch.setattr(type(ens), "_workflow_failure", reject_one_seed)
    ens._stage = "seeded"
    ens.minimize()
    assert rejected, "the test did not send a seed through the rejection path"
    assert ens.n == 2, "embed(n=2) must hand back exactly 2 geometries after re-seeding"
    assert all(geom.check(ens.mol, c, donors=donors, constraints=ens.cons).ok() for c in ens.ids)


def test_free_fragment_is_tethered_at_vdw_contact():
    embedded = 0
    for iso in rx.metal("CCCN[Pd](Cl)(Cl)NCCC.c1ccccc1", "square_planar"):
        ens = rx.embed(iso, n=2).minimize()
        if not ens.n:
            continue
        embedded += 1
        pos = ens.mol.GetConformer(ens.ids[0]).GetPositions()
        for frag in Chem.GetMolFrags(ens.mol):
            heavy = [a for a in frag if ens.mol.GetAtomWithIdx(a).GetAtomicNum() > 1 and a != iso.metal]
            if heavy:
                assert min(float(np.linalg.norm(pos[iso.metal] - pos[a])) for a in heavy) < 8.0
    assert embedded >= 1


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


def test_coordinate_composes_with_fix_constrain_and_contacts():
    iso = rx.metal("CCCN[Pd](Cl)NCCC.O", "square_planar")[0]
    metal = iso.metal
    oxygen = next(atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "O")
    carbons = [atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "C"]
    nitrogen = next(atom.GetIdx() for atom in iso.mol.GetAtoms() if atom.GetSymbol() == "N")
    fixed = (carbons[0], carbons[1])
    soft = (carbons[-2], carbons[-1])
    contact = (nitrogen, oxygen)

    ens = rx.embed(
        iso,
        coordinate=oxygen,
        fix={fixed: 1.53},
        constrain={soft: (1.4, 1.7)},
        contacts={contact: (2.5, 4.5)},
        n=1,
        seed=1,
    )

    assert oxygen in ens.iso.vertices
    assert ens.cons.fixed[fixed] == (1.53, 1.53)
    assert ens.cons.contacts[0] == frozenset({soft, contact})
    assert tuple(sorted((metal, oxygen))) in ens.cons.distances


def test_coordinate_relieves_the_phantom_floor_it_creates(monkeypatch):
    from rxembed import bounds as _b

    tols, real = [], _b._bounds

    def spy(mol, cons):
        bm, tol = real(mol, cons)
        tols.append(tol)
        return bm, tol

    monkeypatch.setattr(_b, "_bounds", spy)
    iso = rx.metal("N->[Pt](Cl)Cl.CC(C)=O", "square_planar").select(index=0)  # one vacant site + free acetone
    o = next(a.GetIdx() for a in iso.mol.GetAtoms() if a.GetSymbol() == "O")
    ens = rx.embed(iso, coordinate=o, n=1)
    carbonyl = next(n.GetIdx() for n in iso.mol.GetAtomWithIdx(o).GetNeighbors())
    assert (iso.metal, o, carbonyl) in ens.cons.angles
    assert tols, "the embed never built a bounds matrix"
    assert max(tols) == 0.0, f"bound crossover repaired: {max(tols) * 100:.4f}%"


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ("donor", "already donors"),
        ("metal", "not metal atoms"),
        ("missing", "must be in"),
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


def _ml_windows(iso):
    return {k: v for k, v in iso.cons.distances.items() if iso.metal in k}


def test_etkdg_conformer_is_not_metal_geometry():
    mol = _fake_geometry(_SQUARE_PD)
    auto = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar")[0])
    model = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar", lengths="model")[0])

    def mids(w):
        return sorted(round((lo + hi) / 2, 3) for lo, hi in w.values())

    assert len(set(mids(model))) == 2, f"the model must give the two Cl one window and the two N another: {mids(model)}"
    assert len(set(mids(auto))) == 4, f"the premise: a metal-blind conformer splits every donor: {mids(auto)}"
    short = np.mean(mids(model)) - np.mean(mids(auto))  # and it is SHORT, not merely different
    assert short > 0.15, f"the metal-blind conformer should sit well inside the model, got {short:.3f} A"


def test_input_and_model_lengths_ignore_mol_metadata():
    mol = _fake_geometry(_SQUARE_PD)
    auto = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar")[0])
    given = _ml_windows(enumerate_isomers(Chem.Mol(mol), "square_planar", lengths="input")[0])
    assert given == auto, "'auto' on a Mol WITH a conformer is 'input'"

    graph = Chem.MolFromSmiles(_SQUARE_PD)
    auto_g = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar")[0])
    model_g = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar", lengths="model")[0])
    assert auto_g == model_g, "'auto' on a Mol WITHOUT one is 'model'"


def test_lazy_length_source_is_fixed_when_isomers_are_built():
    graph = Chem.MolFromSmiles(_SQUARE_PD)
    expected_model = _ml_windows(enumerate_isomers(Chem.Mol(graph), "square_planar")[0])
    deferred_model = enumerate_isomers(Chem.Mol(graph), "square_planar")[0]
    deferred_model.mol.AddConformer(Chem.Conformer(deferred_model.mol.GetNumAtoms()))
    assert _ml_windows(deferred_model) == expected_model

    geometry = _fake_geometry(_SQUARE_PD)
    expected_input = _ml_windows(enumerate_isomers(Chem.Mol(geometry), "square_planar")[0])
    deferred_input = enumerate_isomers(Chem.Mol(geometry), "square_planar")[0]
    deferred_input.mol.GetConformer().SetAtomPosition(deferred_input.donors[0], Point3D(20, 20, 20))
    assert _ml_windows(deferred_input) == expected_input


def test_input_lengths_require_geometry():
    with pytest.raises(ValueError, match="carries no geometry"):
        enumerate_isomers(Chem.MolFromSmiles(_SQUARE_PD), "square_planar", lengths="input")


def test_length_source_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="rxembed.metal"):
        isos = enumerate_isomers(_fake_geometry(_SQUARE_PD), "square_planar")
    assert len(isos) > 1, "the premise: more than one ordering was built"
    said = [r for r in caplog.records if "M-donor windows from" in r.getMessage()]
    assert len(said) == 1, f"the source was announced {len(said)} times"
    assert "input conformer" in said[0].getMessage()
    assert "lengths='model'" in said[0].getMessage(), "and the message must name the way out"
