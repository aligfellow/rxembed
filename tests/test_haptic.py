"""eta>=3 haptic coordination (Cp/arene sandwiches): the centroid-dummy vertex + the transient-phantom contract.

A Cp/arene face collapses to ONE polyhedron vertex (its centroid dummy) for the embed, but that dummy is embed
scaffolding only — it lives in NO stored Mol. These tests pin the contract: the Isomer/ensemble mol is real (the
ring atoms ARE the donors), every stage runs clean, and no phantom ever reaches the gate, a graph diff, or a dump.
Pure RDKit + UFF surrogate, no xtb.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import geometry as geo
from rxembed.rdkit_embed.constraints import coordination_builders as _cbuild
from rxembed.rdkit_embed.constraints import metal as _metal

S, D = Chem.BondType.SINGLE, Chem.BondType.DOUBLE


def _sandwich(ring_n, metal_z, metal_q, kekule, anion, zoff, rad):
    """Build a bis(eta-n) sandwich Mol with a seed geometry: `metal` + two n-membered carbocycles, dative M<-C."""
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
            rw.AddBond(me, c, Chem.BondType.DATIVE)  # dative M<-C so each ring keeps its valence when stripped
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
    """Ferrocene: two Cp- rings on Fe(II), an eta5 sandwich (M-C ~2.05 A)."""
    return _sandwich(5, 26, 2, [S, D, S, D, S], anion=True, zoff=1.66, rad=1.21)


def dibenzenechromium():
    """Bis(benzene)chromium: two benzene rings on Cr(0), an eta6 sandwich (M-C ~2.13 A)."""
    return _sandwich(6, 24, 0, [S, D, S, D, S, D], anion=False, zoff=1.61, rad=1.40)


def cp_ticl3():
    """CpTiCl3: a Cp- ring + three chlorides on Ti(IV) — a C3v piano stool (ONE face + 3 sigma donors, CN4)."""
    rw = Chem.RWMol()
    ti = rw.AddAtom(Chem.Atom(22))
    rw.GetAtomWithIdx(ti).SetFormalCharge(4)
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    for k, b in enumerate([S, D, S, D, S]):
        rw.AddBond(cs[k], cs[(k + 1) % 5], b)
    rw.GetAtomWithIdx(cs[0]).SetFormalCharge(-1)
    for c in cs:
        rw.AddBond(ti, c, Chem.BondType.DATIVE)
    cls = [rw.AddAtom(Chem.Atom(17)) for _ in range(3)]
    for cl in cls:
        rw.AddBond(ti, cl, Chem.BondType.SINGLE)
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


def test_half_sandwich_is_a_piano_stool_not_a_flat_square():
    """A haptic CN4 (Cp + 3 sigma) defaults to tetrahedral (piano stool), NEVER square_planar with a ligand trans
    THROUGH the ring — a haptic face is apical, not an in-plane vertex. Regression on `geometry_for(has_haptic=)`."""
    iso = rx.metal(cp_ticl3())[0]
    assert iso.geometry == "tetrahedral"  # NOT square_planar (the vertex-count default)
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    ens = rx.embed(iso, n=6).minimize()
    assert ens.ids
    cid = next(iter(ens.ids))
    pos = ens.mol.GetConformer(cid).GetPositions()
    ring = next(iter(iso.cons.haptic.values()))
    centroid = np.mean([pos[a] for a in ring], axis=0)
    cls = [a.GetIdx() for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 17]

    def angle(a, b, c):
        u, v = a - b, c - b
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))

    for cl in cls:  # no chloride sits trans (through) the coordinated ring — that is the absurd flat-square outcome
        assert angle(centroid, pos[iso.metal], pos[cl]) < 150.0
    for i in range(len(cls)):  # ...and no two chlorides mutually trans either
        for j in range(i + 1, len(cls)):
            assert angle(pos[cls[i]], pos[iso.metal], pos[cls[j]]) < 150.0


@pytest.mark.parametrize(
    ("build", "ring_n", "n_atoms", "mc_lo", "mc_hi"),
    [(ferrocene, 5, 11, 1.9, 2.3), (dibenzenechromium, 6, 13, 2.0, 2.4)],
)
def test_haptic_sandwich(build, ring_n, n_atoms, mc_lo, mc_hi):
    """A bis(eta-n) sandwich enumerates one real isomer, embeds + minimises clean, at the right M-C distance."""
    isos = rx.metal(build())
    assert len(isos) == 1  # two identical faces on one metal — a single achiral identity
    iso = isos[0]

    # --- the transient-phantom contract: the stored Isomer is REAL --------------------------------------------
    assert iso.mol.GetNumAtoms() == n_atoms  # phantom-free (only the metal + the 2 rings)
    assert set(iso.donors) == set(range(1, 2 * ring_n + 1))  # every ring atom IS a donor (not a centroid dummy)
    assert iso.cons.phantoms  # the constraints DO name centroid dummies (the embed scaffolding)...
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms)  # ...but at indices past the real atoms
    assert len(iso.cons.haptic) == 2  # one centroid per face
    assert "η" in _metal.arrangement(iso)  # the vertex renders as a hapticity tag, not a crash

    # --- embed -> minimize: clean, and the ring seats at the coordination distance -----------------------------
    ens = rx.embed(iso, n=4).minimize()
    assert ens.mol.GetNumAtoms() == n_atoms  # still real after the full relax
    assert ens.ids  # at least one geometry survived the gate
    cid = next(iter(ens.ids))
    assert geo.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    mc = sorted(float(np.linalg.norm(pos[iso.metal] - pos[c])) for c in iso.donors)
    assert mc[0] >= mc_lo  # closest ring atom not collapsed onto the metal
    assert mc[-1] <= mc_hi  # farthest ring atom still within the eta-n coordination shell

    # --- no phantom leaks past the ensemble boundary -----------------------------------------------------------
    ens.filter("connectivity")  # re-perceives the graph — must not choke on (or find) a phantom


def test_haptic_mc_seed_settle():
    """`.mc()` (and its explore pass) survive a haptic complex — the seed-settle relax materialises the centroid.

    Regression: `_settle_seeds` rebuilt a `Constraints` without carrying `phantoms`/`haptic`, so the seeded
    M->centroid distance/angle named an index the relax never materialised — an RDKit range error (caught, but the
    settle silently failed and spewed noise). Guards that every rebuilt `Constraints` carries the haptic map.
    """
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc().ids
    assert rx.embed(rx.metal(ferrocene())[0], n=3).mc(explore=True).ids


def test_haptic_dump_is_phantom_free(tmp_path):
    """`dump` writes only the real atoms — no haptic centroid dummy reaches the xyz."""
    ens = rx.embed(rx.metal(ferrocene())[0], n=3).minimize()
    path = ens.dump(str(tmp_path / "ferrocene.xyz"))
    with open(path) as f:
        count = int(f.readline())
    assert count == ens.mol.GetNumAtoms() == 11  # the 2 Cp rings + Fe, nothing else


def test_haptic_arrangement_selectable():
    """The eta-n vertex is a selectable arrangement key, rendered from the ring (the real mol has no such atom)."""
    iso = rx.metal(ferrocene())[0]
    arr = _metal.arrangement(iso)
    assert arr.count("η5") == 2  # two eta5 faces
    assert rx.metal(ferrocene()).select(arrangement=arr) is not None  # round-trips through select()


def test_from_geometry_collapses_haptic_faces():
    """`rx.embed(mol)`'s retain-input path (`from_geometry`) collapses each Cp face to ONE centroid vertex.

    Without the `_collapse_haptic` call, the transient-centroid mechanism NEVER fires on the `rx.embed(xyz)`
    path: raw ring atoms flow into `cons`/`vertices`/`haptic`, so this is RED there (haptic {}, ten vertices,
    45 angles across raw ring atoms). Pins parity with the `rx.metal` enumerate path.
    """
    iso = _cbuild.from_geometry(ferrocene())  # a real-Fe Mol with a conformer -> retain-input path
    assert len(iso.haptic) == 2  # one centroid per Cp face (was {} before the collapse)
    assert len(iso.vertices) == 2  # TWO coordination sites, not ten raw ring atoms
    assert all(v in iso.haptic for v in iso.vertices)  # both vertices are centroid dummies
    assert iso.cons.haptic == dict(iso.haptic)  # the constraints carry the same centroid map
    assert len(iso.cons.phantoms) == 2  # two centroid dummies named as embed scaffolding
    assert all(p >= iso.mol.GetNumAtoms() for p in iso.cons.phantoms)  # ...at indices past the real atoms
    assert iso.mol.GetNumAtoms() == 11  # the stored mol is REAL (Fe + 2 Cp), phantom-free
    assert set(iso.donors) == set(range(1, 11))  # every ring atom is still a real donor
    assert len(iso.cons.angles) == 1  # one centroid-M-centroid angle (was 45 across raw ring atoms)

    # embed the retained arrangement end-to-end: sane ferrocene, both rings at the crystal distance, one molecule
    ens = rx.embed(ferrocene(), n=4, seed=1).minimize()
    assert ens.mol.GetNumAtoms() == 11  # still real after the full relax
    assert ens.ids  # at least one geometry survived the gate
    cid = next(iter(ens.ids))
    assert geo.check(ens.mol, cid, donors=iso.donors, constraints=ens.cons).ok()
    pos = ens.mol.GetConformer(cid).GetPositions()
    fe = next(a.GetIdx() for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 26)
    for ring in ens.cons.haptic.values():
        d = float(np.linalg.norm(pos[fe] - np.mean([pos[a] for a in ring], axis=0)))
        assert 1.5 < d < 1.85  # Fe->Cp-centroid ~1.66 A (ferrocene crystal)
    ens.filter("connectivity")  # re-perceives the graph — must not choke on a phantom
    assert ens.ids  # ...and the molecule stayed connected (one species)


def test_from_geometry_piano_stool_is_tetrahedral():
    """A haptic CN4 (Cp + 3 sigma) via the retain-input path is a piano stool = tetrahedral, not a flat square.

    Folds the apical divergence: `from_geometry` now passes `has_apical=` to `geometry_for`, so an eta>=3 face
    takes the tetrahedral count default instead of the square_planar the vertex count alone would pick (which
    would seat a chloride trans THROUGH the ring). Also RED without the collapse (four sites, not eight atoms).
    """
    iso = _cbuild.from_geometry(cp_ticl3())
    assert iso.geometry == "tetrahedral"  # NOT square_planar
    assert len(iso.vertices) == 4  # centroid + 3 Cl
    assert len(iso.haptic) == 1  # the one Cp face collapsed to a single vertex
