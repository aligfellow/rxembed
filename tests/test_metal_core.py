"""`metal_core`: shape perception (`classify_geometry`, `geometry_for`) and the metal surrogate round-trip.

Two contracts, both about the metal atom itself rather than the ligands around it.

Perception: the shape invariant. No shape may be constructable but not perceivable, so every `POLYHEDRA`
record must re-perceive as itself, and the planarity exclusion runs one way (a flat sphere is not a 3D shape;
an out-of-plane sphere is a distorted planar shape as readily as a 3D one).

The surrogate. `surrogate_metal` swaps the metal for a bond-less neutral carbon so the DG/FF can embed it;
`restore_metal`/`connect_metal` must hand back the element, the oxidation state and the M-donor dative bonds.
Losing the charge sends every real-energy calculation to xtb at the wrong total; losing the bonds hands the
user a dissociated complex. Asserted here on `metal_core`'s own functions, and once through the public embed
so the call sites are known to use them. Which stage returns a connected mol is `Ensemble`'s accessor
contract, tested in `tests/pipeline/test_ensemble.py`. RDKit + UFF throughout, no xtb.
"""

from __future__ import annotations

import logging
import pathlib
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

import rxembed.pipeline as rx
from rxembed import metal_core as _metal
from rxembed.metal_core import classify_geometry, geometry_for, repair_bond_stereo
from rxembed.metal_polyhedron import POLYHEDRA, describe
from rxembed.pipeline.perceive import _xyz_to_mol

_MN_H2 = "examples/structures/mn-h2.xyz"  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe
_MN_H2_RC = [1, 5, 63, 64, 65, 66]  # its reacting core
_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # neutral en-PdBrCl, which reliably embeds: the connectivity fixture
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"  # net 0, Ni(II)


def _ideal_sphere(dirs, r):
    """A bare Mol whose atom 0 is a metal and 1..N its vertices at radius `r` along `dirs`."""
    rw = Chem.RWMol()
    for _ in range(len(dirs) + 1):
        rw.AddAtom(Chem.Atom(6))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for i, d in enumerate(dirs):
        u = np.array(d, float)
        conf.SetAtomPosition(i + 1, Point3D(*(u / np.linalg.norm(u) * r)))
    mol.AddConformer(conf, assignId=True)
    return mol


def _classify(dirs, r):
    return classify_geometry(_ideal_sphere(dirs, r), 0, list(range(1, len(dirs) + 1)))


def _bailar(degrees):
    """The octahedron's own vertices with one C3 face rotated by `degrees` about the body diagonal.

    0° leaves the octahedron, 60° reaches the trigonal prism. Built from the record's own `vertex_dirs` so the
    probe cannot drift away from the shape it is distorting.
    """
    dirs = np.array(POLYHEDRA["octahedral"].vertex_dirs, float)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    face = [i for i, d in enumerate(dirs) if d @ axis > 0]
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)  # Rodrigues
    dirs[face] = dirs[face] @ rot.T
    return dirs


def _tilted_square(tilt_deg):
    """A square plane with an alternating `tilt_deg` out-of-plane bow: the tetrahedral distortion of Ni/Pd(II)."""
    t = np.radians(tilt_deg)
    return [
        (np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t) * (1 if k % 2 == 0 else -1))
        for k, phi in enumerate((0.0, np.pi / 2, np.pi, 3 * np.pi / 2))
    ]


# --- perception: the shape invariant ------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(POLYHEDRA))
def test_every_record_round_trips(name):
    """The invariant: build each record's ideal sphere, re-perceive it, and the name comes back.

    A record that cannot round-trip makes `Isomer.summary()` lie: it names a shape `from_geometry` would
    rebuild as a different polytope. Every record, at one representative M-L length; the one record whose
    answer depends on bond length has its own test below. The constructable-but-unperceivable case this
    catches is a spectrum clone; a record that steals real distorted spheres is caught by
    `test_a_bis_chelate_tetrahedron_is_not_reclassified` instead.
    """
    got = _classify(POLYHEDRA[name].vertex_dirs, 2.1)
    assert got == name, f"{describe(name)} re-perceives as {got}"


def test_a_short_bonded_pyramid_is_not_excluded_by_its_own_flatness():
    """The CN3 pyramid still perceives at a 1.4 Å bond, where its own ideal is flatter than `COPLANAR_TOL`.

    `COPLANAR_TOL` is an absolute out-of-plane RMS while an ideal pyramid measures 0.1443 x r, so below ~1.7 Å
    the record's own ideal sphere reads as flat. `_too_flat_for` therefore bounds the exclusion by each
    record's own ideal; with a bare `COPLANAR_TOL` this sphere comes back `trigonal_planar`, and the
    invariant is broken exactly where real short M-L bonds live.
    """
    dirs = POLYHEDRA["trigonal_pyramidal"].vertex_dirs
    assert _metal._ideal_plane_rms(POLYHEDRA["trigonal_pyramidal"], 1.4) < _metal.COPLANAR_TOL, "fixture premise"
    assert _classify(dirs, 1.4) == "trigonal_pyramidal"


def test_a_bowed_square_plane_is_not_reclassified():
    """A tetrahedrally-distorted square still perceives as square_planar; planarity never overrules a better fit.

    The converse of the flatness exclusion, and why it runs one way: at M-L 2.3 Å an 8° bow measures a
    coplanarity RMS past `COPLANAR_TOL`, so excluding `square_planar` for "not being flat" handed the name to
    `seesaw` at a 19.1° spectrum RMS against square_planar's own 9.3°. Above ~14° the spectrum itself prefers
    seesaw and that answer stands.
    """
    assert _classify(_tilted_square(8), 2.3) == "square_planar"


def test_a_bis_chelate_tetrahedron_is_not_reclassified():
    """A tetrahedrally-seated 4-ring bis-chelate still perceives as tetrahedral, not as a CN4 pyramid.

    The guard against re-adding the deleted CN4 monopyramid. Its own round trip was self-consistent; its defect
    was stealing real, distorted tetrahedra: a small-bite chelate pinches one row to ~71°, which scored nearer
    the monopyramid (16.16) than tetrahedral (16.70), silently.
    """
    iso = rx.metal("CC1=[O]->[Zn+2](Cl)(Cl)<-[O-]1", "tetrahedral").select(index=0)
    mol = iso.restore(rx.embed(iso, n=2, seed=7).minimize().mol)
    assert [i.geometry for i in rx.metal(mol)] == ["tetrahedral"]


@pytest.mark.parametrize(("twist", "expected"), [(0.0, "octahedral"), (60.0, "trigonal_prismatic")])
def test_the_bailar_twist_is_named_at_both_ends(twist, expected, caplog):
    """Both ends of the octahedron->prism interconversion are real shapes, and both are named silently.

    This test used to assert the opposite: that a 60 deg twist WARNED, because `octahedral` was alone at CN6
    and so won the argmin outright at 29.2 deg of misfit. Adding `trigonal_prismatic` is exactly the fix: the
    argmin now has an opponent, the crossover lands between 30 and 35 deg where the geometry really is
    ambiguous, and neither end needs a warning to be honest.
    """
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(_bailar(twist), 2.1), 0, list(range(1, 7)))
    assert got == expected
    assert not [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_the_fit_floor_still_flags_a_sphere_no_record_fits(caplog):
    """The floor is for a sphere that matches nothing, which a competitor cannot rescue.

    `classify_geometry` is an unconditional argmin at every CN, so something has to say "this name is the
    nearest record, not a reading". Here six donors are crushed onto one hemisphere: no CN6 record is close,
    and the answer must be loud rather than reclassified: a sorted spectrum discards vertex roles, so a poor
    fit's identity is not trustworthy either.
    """
    squashed = np.array([(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])])
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        got = classify_geometry(_ideal_sphere(squashed, 2.1), 0, list(range(1, 7)))
    assert got is not None, "the argmin is still returned; loud, never reclassified"
    assert [r for r in caplog.records if "no shape fits" in r.message], caplog.text


def test_the_cn_defaults_are_the_common_shapes():
    """A CN with no requested shape defaults to the common one: a new record must not steal it."""
    assert geometry_for(3) == "trigonal_planar"
    assert geometry_for(4) == "square_planar"


# --- the surrogate round-trip: oxidation state ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "smiles"),
    [
        ("[PdCl4]2-", "[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]"),
        # a genuinely neutral metal: a phantom charge on it would cancel against the ligands in any net total
        ("Fe(CO)5", "[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"),
    ],
)
def test_the_surrogate_captures_the_oxidation_state_and_restore_hands_it_back(name, smiles):
    """`surrogate_metal` neutralises the metal but CAPTURES its charge; `restore_metal` hands element + charge back.

    The surrogate must be neutral, a charged bond-less carbon is not a valid DG atom, so the oxidation state
    can only survive by being carried out of band, and it is what reaches the calculator as the total charge.
    """
    m0 = Chem.MolFromSmiles(smiles)
    metal_idx = _metal.metal_index(m0)
    q0 = m0.GetAtomWithIdx(metal_idx).GetFormalCharge()

    surrogate, m, _donors, real_z, real_q = _metal.surrogate_metal(m0)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == _metal.SURROGATE
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == 0, f"{name}: the DG surrogate must be neutral"
    assert real_q == q0, f"{name}: the oxidation state was thrown away, not captured"

    _metal.restore_metal(surrogate, m, real_z, real_q)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == real_z
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == q0


def _metal_charges(mol):
    """`{atom index: formal charge}` for every metal: a neutral metal must come back 0, not gain a phantom."""
    return {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() in _metal._METAL_Z}


def _assert_charge_roundtrip(name, mol_charge, ens):
    """After minimize: the mol charge and the number sent to xtb both equal the input's net charge."""
    ens = (ens.candidates[0] if hasattr(ens, "candidates") else ens).minimize()
    assert Chem.GetFormalCharge(ens.mol) == mol_charge, f"{name}: the metal's oxidation state was lost"
    # the number that reaches xtb; pinned explicitly so a _calc_charge refactor cannot silently re-break it
    assert ens._calc_charge(None) == mol_charge, f"{name}: _calc_charge sends {ens._calc_charge(None):+d}"
    return ens


@pytest.mark.parametrize("route", ["isomer", "constrain"])
def test_the_metal_comes_back_at_its_oxidation_state_through_the_public_embed(route):
    """Both surrogate doors, an enumerated `Isomer` and the `constrain=` `from_surrogate` path, restore Ni(II).

    The N-bound Ni is net-neutral with a +2 metal, so a lost oxidation state is invisible in the total until the
    per-atom charges are read; both are asserted.
    """
    want = _metal_charges(Chem.MolFromSmiles(_NI_N))
    spec = rx.metal(_NI_N)[0] if route == "isomer" else _NI_N
    kw = {} if route == "isomer" else {"constrain": {(0, 1): (1.4, 1.6)}}
    ens = _assert_charge_roundtrip(route, 0, rx.embed(spec, n=2, seed=1, **kw))
    assert _metal_charges(ens.mol) == want, f"{route}: the metal came back at the wrong oxidation state"


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_a_spectator_metals_charge_roundtrips_on_both_restore_sites():
    """mn-h2 (Mn + a spectator ferrocene Fe): both metals restore, through `embed(iso)` and through `minimize`."""
    q_in = Chem.GetFormalCharge(_xyz_to_mol(_MN_H2, 0))
    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    _assert_charge_roundtrip("mn-h2 (embed isomer)", q_in, rx.embed(isos[0], n=2, seed=1))
    _assert_charge_roundtrip("mn-h2 (minimize)", q_in, rx.minimize(_MN_H2))


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_isomer_restore_hands_every_metal_its_oxidation_state():
    """`Isomer.restore` restores the centre and each spectator (the `extra` 3-tuples), not a silent zero."""
    iso = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)[0]
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == _metal.SURROGATE  # still the neutral carbon
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == 0
    iso.restore()
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == iso.real_q
    for mi, _rz, rq in iso.extra:  # the spectator ferrocene Fe
        assert iso.mol.GetAtomWithIdx(mi).GetFormalCharge() == rq


# --- the surrogate round-trip: connectivity -------------------------------------------------------------


def test_connect_metal_restores_the_stripped_bonds_and_disconnect_is_its_inverse():
    """`connect_metal` re-adds every M-donor bond as a donor->metal DATIVE and RE-PERCEIVES RingInfo.

    Regression: `RemoveBond` clears RingInfo but `AddBond` does not restore it, so a chelate ring closed
    through the metal reached the ring-aware SMARTS of the dedup descriptor uninitialised -> RuntimeError. The
    dative direction matters as much: counted toward the metal's valence, never the donor's, so no ligand
    gains an H or a charge.
    """
    mol = Chem.MolFromSmiles(_EN_PDBRCL)  # the en chelate closes its ring through the Pd
    stripped, m, donors, real_z, real_q = _metal.surrogate_metal(mol)
    assert len(Chem.GetMolFrags(stripped)) > 1, "the surrogate must leave the metal a separate fragment"

    _metal.restore_metal(stripped, m, real_z, real_q)
    connected = _metal.connect_metal(stripped, [(d, m) for d in donors])
    assert len(Chem.GetMolFrags(connected)) == 1
    for d in donors:
        b = connected.GetBondBetweenAtoms(int(d), int(m))
        assert b is not None, f"donor {d} has no bond to the metal"
        assert b.GetBondType() == Chem.BondType.DATIVE, f"donor {d}'s bond to the metal is not DATIVE"
        assert b.GetBeginAtomIdx() == int(d), f"donor {d}'s dative bond runs the wrong way"
    assert connected.GetRingInfo().NumRings() >= 1, "the chelate ring closed through the metal was not re-perceived"
    assert connected.GetSubstructMatches(Chem.MolFromSmarts("[R]"))  # a ring-aware query must not raise

    assert len(Chem.GetMolFrags(_metal.disconnect_metal(connected))) > 1, "disconnect is not connect's inverse"


def test_the_bare_embed_finalizes_a_connected_copy_without_touching_the_working_mol():
    """`.mol` before any `minimize()` is already the connected complex, and reading it moves nothing.

    The bare stage is the gap the accessor closes (every later stage is `tests/pipeline/test_ensemble.py`'s).
    The working `_mol` must STAY the bond-less carbon the engine needs; UFF cannot type a bonded transition
    metal, so the finalize has to be a copy, at identical coordinates.
    """
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1)
    assert not ens._minimized, "the fixture must be a bare embed, or this measures nothing"

    out = ens.mol
    assert len(Chem.GetMolFrags(out)) == 1, "bare .mol is not connected"
    assert out.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z
    assert "Pd" in Chem.MolToSmiles(out), "the connected graph must round-trip to a real complex SMILES"

    assert ens._mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == _metal.SURROGATE, "the finalize mutated `_mol`"
    assert len(Chem.GetMolFrags(ens._mol)) > 1, "the working mol was re-bonded"
    for cid in ens.ids:
        assert np.allclose(ens._mol.GetConformer(cid).GetPositions(), out.GetConformer(cid).GetPositions())


# --- the metal-bound donor stereocentre hold (`_hold_donor_chirality` / `_release_donor_chirality`) -------

_CARBANION_NI = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)N(c1ccccc1)[CH-]->2c1ccccc1"
_CARBANION_C = 23  # the metal-bound sp3 carbanion donor: a stereocentre only WHILE bound


def _chirality_volume(mol, cid, centre):
    conf = mol.GetConformer(cid)
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()][:3]
    p = np.array([list(conf.GetAtomPosition(i)) for i in [centre, *nbrs]])
    return float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))


def test_a_metal_bound_carbanion_donor_embeds_two_distinct_hands():
    """The hold (neutralise + dummy-D) survives the whole embed + relax, so the two hands stay opposite."""
    en = rx.metal(_CARBANION_NI, "square_planar")
    assert {i.stereo_label for i in en} == {"23R", "23S"}  # metal-priority CIP labels
    hands = []
    for iso in en:
        e = rx.embed(iso, n=4).minimize()
        assert e.n >= 1
        assert e.mol.GetNumAtoms() == 65  # the dummy deuterium is removed after embed + relax
        assert not any(a.GetIsotope() == 2 for a in e.mol.GetAtoms())  # no leaked D
        assert any(a.GetSymbol() == "Ni" for a in e.mol.GetAtoms())  # the surrogate is switched back
        signs = {np.sign(_chirality_volume(e.mol, i, _CARBANION_C)) for i in e.ids}
        assert len(signs) == 1  # every conformer of this isomer has the same donor hand
        hands.append(signs.pop())
    assert len(set(hands)) == 2  # and the two isomers are opposite


def test_the_donor_charge_is_restored_after_the_hold():
    # the hold NEUTRALISES the carbanion during the embed (a -1 C can't take a 4th bond) then RESTORES it;
    # a leak would corrupt the donor charge, and the total charge sent to xtb downstream.
    for iso in rx.metal(_CARBANION_NI, "square_planar"):
        assert rx.embed(iso, n=1).mol.GetAtomWithIdx(_CARBANION_C).GetFormalCharge() == -1


# ---------------------------------------------------------------------------------------------------------
# RDKit STATE across the metal-bond surgery (was test_mol_state.py)
#
# RDKit state must stay valid across the metal-bond surgery.
#
# Stripping the M-donor bonds can invalidate RDKit state that was derived while the metal was still bonded, and
# RDKit does not always notice. Two ways it bit us, both found on the tmQM corpus:
#
#   * a double bond left FLAGGED stereo with its two reference atoms dropped; because the metal was one of them.
#     RDKit's own ETKDG then indexes the empty vector and SEGFAULTS (rc 139), which no try/except can catch.
#   * `metal.surrogate_metal` re-imposing a STRICT sanitize on a structure the reader deliberately admitted leniently,
#     rejecting chemistry (an unkekulisable quinoid ring, a BPh4- boron) that perception had already accepted.
#
# These are asserted as INVARIANTS rather than as "structure X does not crash": the invariant is what the next
# surgery site has to honour, and it is what makes the guarantee checkable on any input.
# ---------------------------------------------------------------------------------------------------------


# rxembed's own fixtures. The invariants below must hold on any metal complex, so the gate runs on structures
# this repo ships rather than reaching into a sibling checkout: a unit suite that depends on an absolute path
# outside the project is not portable and is not a gate. The 144-structure corpus SWEEP that originally found
# these defects is a measurement, not a gate, and lives in `benchmark/` where the corpus is in scope.
_CORPUS = sorted(str(p) for p in pathlib.Path("examples/structures").glob("*.xyz"))
corpus_only = pytest.mark.skipif(not _CORPUS, reason="no structure fixtures found")


def _orphaned(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != 2]


@corpus_only
def test_the_surgery_preserves_stereo_rather_than_blanket_clearing_it():
    """Repair must RE-DERIVE where a reference survives, not just drop every damaged flag.

    On DEYMIE the metal is itself a stereo reference atom for two C=N bonds; stripping it orphans both. The
    E/Z is still definable from the substituents that remain, so it must survive; re-expressed against them
    (a bond that read "Z relative to the metal" becomes "E relative to the other ring atom": same geometry,
    new reference). Blanket-clearing would pass the orphan invariant while silently discarding real
    stereochemistry, so this test is what stops the cheap fix.
    """
    path = next((p for p in _CORPUS if p.endswith("DEYMIE.xyz")), None)
    if path is None:
        pytest.skip("DEYMIE.xyz is a corpus structure: the benchmark/ sweep covers it")
    mol = _xyz_to_mol(path, 0)
    raw_flags = sum(1 for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    assert raw_flags, "fixture must carry bond stereo before the surgery"

    prepared, *_ = _metal.surrogate_metal(mol)
    kept = sum(1 for b in prepared.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    assert not _orphaned(prepared), "an orphaned flag survived surrogate_metal()"
    assert kept, "the surgery discarded ALL bond stereo instead of re-deriving what was still definable"


def test_repair_is_a_noop_on_a_clean_mol():
    mol = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C/C"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    Chem.AssignStereochemistryFrom3D(mol)
    stereo_before = [(b.GetIdx(), b.GetStereo()) for b in mol.GetBonds()]
    assert repair_bond_stereo(mol) == 0
    assert [(b.GetIdx(), b.GetStereo()) for b in mol.GetBonds()] == stereo_before


@corpus_only
@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[perceive]")
def test_no_prepared_mol_carries_a_stereo_flag_without_its_references():
    """THE INVARIANT. A flag without two live reference atoms is what segfaults ETKDG (DAJXOD, DEYMIE)."""
    violations = []
    for path in _CORPUS:
        mol = _xyz_to_mol(path, 0)
        if not _metal.metal_indices(mol):
            continue
        prepared, *_ = _metal.surrogate_metal(mol)
        if _orphaned(prepared):
            violations.append(path.rsplit("/", 1)[-1])
    assert violations == [], f"orphaned stereo flags survive surrogate_metal(): {violations}"


@corpus_only
def test_surrogate_metal_admits_everything_the_reader_admits():
    """`surrogate_metal` must not re-impose strictness `inputs._xyz_to_mol` waived (WACJET, WIMCAA, XAQDUS)."""
    rejected = []
    for path in _CORPUS:
        try:
            mol = _xyz_to_mol(path, 0)
        except Exception:
            continue  # perception itself declined it: not surrogate_metal's business
        if not _metal.metal_indices(mol):
            continue
        try:
            _metal.surrogate_metal(mol)
        except Exception as exc:
            rejected.append((path.rsplit("/", 1)[-1], type(exc).__name__))
    assert rejected == [], f"surrogate_metal() rejects structures the reader accepted: {rejected}"


def test_a_haptic_face_and_a_chirality_cap_can_compose():
    """The two transients must not compete for one reserved index block.

    A haptic face reserves centroid-dummy indices from the real atom count, and a labile (carbanion/amine)
    donor's chirality D-cap is appended from the same count, so whichever lands second collides. rxembed
    appended the cap first and then REFUSED the combination with a hard ValueError, which turned a normal
    ligand class into a total failure: an eta2/eta3/eta5 face whose atoms are also anionic sp3 stereocentres
    is exactly a Cp / allyl / ylide, and it is 31% of the haptic structures in the tmQM corpus. It names the
    same atom ("COJKAO's C46 is both a carbanion and an eta2 atom") and treats it as routine.

    The caps now take the low indices and the centroid block slides above them.
    """
    import rxembed.pipeline as rx
    from tests.test_metal_isomers import ferrocene

    iso = next(iter(rx.metal(ferrocene())))
    cons = iso.cons
    assert cons.haptic, "fixture must carry a haptic face"
    before = sorted(cons.haptic)

    _metal._shift_phantoms(cons, 2)  # as if two D-caps had been appended ahead of the centroids
    after = sorted(cons.haptic)
    assert after == [i + 2 for i in before]
    assert sorted(cons.phantoms) == after, "phantoms must move with haptic"
    # the COMPLETENESS CHECK: no field may still name an old index. This is what stops a future field from
    # being silently left behind: the same defect class as the hand-listed Constraints copies.
    stale = set(before) - set(after)
    for name in ("distances", "angles", "pulls", "floors", "dg_floors"):
        for k in getattr(cons, name):
            assert not (stale & set(k)), f"{name} still names a pre-shift dummy index {k}"
    for s in cons.spheres:
        assert not (stale & {d for d, _ring in s.haptic}), "the sphere recipe still names a pre-shift dummy"


@corpus_only
def test_a_shipped_metal_fixture_with_both_transients_embeds():
    """End-to-end over the shipped fixtures: a face and a labile donor must not refuse each other.

    Filtered to metal structures that actually carry both: the organic TS fixtures have no metal, and a metal
    without a labile donor never exercised the collision. `tests/test_metal_isomers.py` builds the synthetic pair that
    guarantees coverage even if no shipped .xyz qualifies.
    """
    import rxembed.pipeline as rx

    failures, exercised = [], 0
    for path in _CORPUS:
        try:
            mol = _xyz_to_mol(path, 0)
        except Exception:
            continue
        if not _metal.metal_indices(mol):
            continue  # an organic TS fixture; nothing to enumerate
        mol.RemoveAllConformers()
        try:
            isomers = list(rx.metal(mol))
        except Exception:
            continue  # a geometry rxembed does not enumerate is a different concern
        if not isomers:
            continue
        iso = isomers[0]
        if not (iso.cons.haptic and _metal._labile_donors(iso.mol, iso.donors)):
            continue
        exercised += 1
        try:
            rx.embed(iso, n=2, seed=1)
        except Exception as exc:
            failures.append((path.rsplit("/", 1)[-1], type(exc).__name__, str(exc)[:60]))
    if not exercised:
        # SKIP, never pass: no shipped fixture carries both a haptic face and a labile donor, so a green result
        # here would be vacuous. The condition needs a real corpus structure (COJKAO, ILONON, NUKHEG, the TiCat
        # series; 14 in tmQM), which the `benchmark/` sweep covers. Recorded rather than faked: a synthetic
        # carbanion does not survive `surrogate_metal`'s tag handling, so building one here would test something else.
        pytest.skip("no shipped fixture carries BOTH a haptic face and a labile donor; see benchmark/")
    assert failures == [], f"a face and a chirality cap still refuse each other: {failures}"


def test_ligands_separates_a_complex_and_reports_denticity_per_metal():
    """Each ligand comes back standalone with the atoms that coordinate, keyed by which metal they reach.

    Per metal, not pooled: on the Mn/Fe bimetallic one backbone bridges both centres, and pooling its donors
    reports a meaningless kappa8 where the truth is kappa5 to one and kappa3 to the other. Denticity is the
    thing a caller swaps a ligand ON, so it has to be right.
    """
    mol = _xyz_to_mol(_MN_H2)
    ligs = _metal.ligands(mol)
    assert ligs, "the fixture must have ligands"

    for lig in ligs:
        assert lig.mol.GetNumConformers(), "a ligand must carry its own geometry, or it cannot be re-placed"
        assert lig.mol.GetNumAtoms() == len(lig.atoms), "`atoms` must index the original positionally"
        for donors in lig.donors.values():
            assert donors, "a metal with no donors must not appear as a key"
            assert all(0 <= d < lig.mol.GetNumAtoms() for d in donors), "donors index the LIGAND, not the complex"

    bridging = [lig for lig in ligs if len(lig.donors) > 1]
    assert bridging, "this fixture's backbone bridges both metals; without one the per-metal split is untested"
    assert sorted(len(d) for d in bridging[0].donors.values()) == [3, 5]

    # every donor of every metal is accounted for, exactly once
    seen = sorted(lig.atoms[d] for lig in ligs for ds in lig.donors.values() for d in ds)
    want = sorted(n.GetIdx() for m in _metal.metal_indices(mol) for n in mol.GetAtomWithIdx(m).GetNeighbors())
    assert seen == want, "the ligands must partition the coordination sphere"


def test_ligands_refuses_a_molecule_with_no_metal():
    """It reads a coordination sphere, so a plain organic is a caller error, not an empty list."""
    with pytest.raises(ValueError, match="no transition metal"):
        _metal.ligands(Chem.AddHs(Chem.MolFromSmiles("CCO")))


def test_dative_smiles_round_trips_a_complex_smiles_cannot_write_naively():
    """The whole point is a SMILES you can read back: same atoms, same metals, same charges.

    The Mn/Fe fixture is the case that forces the work. Two of its hydrogens have a second connection - a
    side-on H2 on the Mn and an H-bond relay perceived as a bond - and a hydrogen has no valence left for a
    second single bond, so a naive `MolToSmiles` emits a string that will not parse. Charge is asserted
    because a metal that comes back neutral sends every downstream real-energy call to the wrong total.
    """
    mol = _xyz_to_mol(_MN_H2)
    assert any(a.GetAtomicNum() == 1 and a.GetDegree() > 1 for a in mol.GetAtoms()), (
        "this fixture must contain an over-connected hydrogen, or it does not test the repair"
    )
    assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None, "a naive write must fail here, or there is nothing to fix"

    def metals(m):  # SMILES renumbers, so the multiset is the claim, not the order
        return sorted(
            (m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(i).GetFormalCharge()) for i in _metal.metal_indices(m)
        )

    back = Chem.AddHs(Chem.MolFromSmiles(_metal.dative_smiles(mol)))
    assert back.GetNumAtoms() == mol.GetNumAtoms()
    assert metals(back) == metals(mol) == [("Fe", 2), ("Mn", 0)]


def test_dative_smiles_raises_rather_than_return_an_unreadable_string():
    """A SMILES that does not round-trip is worse than none, so the failure has to be loud.

    Forced with a perceived valence-5 carbon, which SMILES cannot express; a silent return would hand the
    caller a string that parses to a different molecule or to nothing.
    """
    rw = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("[NH3]->[Pd](<-[NH3])(Cl)Cl")))
    c = rw.AddAtom(Chem.Atom(6))
    for _ in range(5):
        h = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(c, h, Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="round-tripping SMILES"):
        _metal.dative_smiles(rw.GetMol())
