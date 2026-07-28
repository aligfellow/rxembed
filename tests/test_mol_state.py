"""RDKit state must stay valid across the metal-bond surgery.

Stripping the M-donor bonds can invalidate RDKit state that was derived while the metal was still bonded, and
RDKit does not always notice. Two ways it bit us, both found on OIN's tmQM corpus:

  * a double bond left FLAGGED stereo with its two reference atoms dropped — because the METAL was one of them.
    RDKit's own ETKDG then indexes the empty vector and SEGFAULTS (rc 139), which no try/except can catch.
  * `metal.surrogate_metal` re-imposing a STRICT sanitize on a structure the reader deliberately admitted leniently,
    rejecting chemistry (an unkekulisable quinoid ring, a BPh4- boron) that perception had already accepted.

These are asserted as INVARIANTS rather than as "structure X does not crash": the invariant is what the next
surgery site has to honour, and it is what makes the guarantee checkable on any input.
"""

import pathlib

import pytest
from rdkit import Chem

from rxembed.inputs import _xyz_to_mol
from rxembed.rdkit_embed.constraints import metal as _metal
from rxembed.rdkit_embed.io import repair_bond_stereo

# rxembed's OWN fixtures. The invariants below must hold on any metal complex, so the gate runs on structures
# this repo ships rather than reaching into a sibling checkout — a unit suite that depends on an absolute path
# outside the project is not portable and is not a gate. The 144-structure corpus SWEEP that originally found
# these defects is a measurement, not a gate, and lives in `oin_adapter/` where the corpus is in scope.
_CORPUS = sorted(str(p) for p in pathlib.Path("examples/structures").glob("*.xyz"))
corpus_only = pytest.mark.skipif(not _CORPUS, reason="no structure fixtures found")


def _orphaned(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != 2]


@corpus_only
def test_the_surgery_preserves_stereo_rather_than_blanket_clearing_it():
    """Repair must RE-DERIVE where a reference survives, not just drop every damaged flag.

    On DEYMIE the metal is itself a stereo reference atom for two C=N bonds; stripping it orphans both. The
    E/Z is still definable from the substituents that remain, so it must survive — re-expressed against them
    (a bond that read "Z relative to the metal" becomes "E relative to the other ring atom": same geometry,
    new reference). Blanket-clearing would pass the orphan invariant while silently discarding real
    stereochemistry, so this test is what stops the cheap fix.
    """
    path = next((p for p in _CORPUS if p.endswith("DEYMIE.xyz")), None)
    if path is None:
        pytest.skip("DEYMIE.xyz is a corpus structure — the sweep in oin_adapter/ covers it")
    mol = _xyz_to_mol(path, 0)
    raw_flags = sum(1 for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    assert raw_flags, "fixture must carry bond stereo before the surgery"

    prepared, *_ = _metal.surrogate_metal(mol)
    kept = sum(1 for b in prepared.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    assert not _orphaned(prepared), "an orphaned flag survived surrogate_metal()"
    assert kept, "the surgery discarded ALL bond stereo instead of re-deriving what was still definable"


def test_repair_is_a_noop_on_a_clean_mol():
    mol = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C/C"))
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    Chem.AssignStereochemistryFrom3D(mol)
    stereo_before = [(b.GetIdx(), b.GetStereo()) for b in mol.GetBonds()]
    assert repair_bond_stereo(mol) == 0
    assert [(b.GetIdx(), b.GetStereo()) for b in mol.GetBonds()] == stereo_before


@corpus_only
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
            continue  # perception itself declined it — not surrogate_metal's business
        if not _metal.metal_indices(mol):
            continue
        try:
            _metal.surrogate_metal(mol)
        except Exception as exc:
            rejected.append((path.rsplit("/", 1)[-1], type(exc).__name__))
    assert rejected == [], f"surrogate_metal() rejects structures the reader accepted: {rejected}"


def test_a_haptic_face_and_a_chirality_cap_can_compose():
    """The two transients must not compete for one reserved index block.

    A haptic face reserves centroid-dummy indices from the REAL atom count, and a labile (carbanion/amine)
    donor's chirality D-cap is appended from the same count — so whichever lands second collides. rxembed
    appended the cap first and then REFUSED the combination with a hard ValueError, which turned a normal
    ligand class into a total failure: an eta2/eta3/eta5 face whose atoms are also anionic sp3 stereocentres
    is exactly a Cp / allyl / ylide, and it is 31% of the haptic structures in the tmQM corpus. OIN names the
    same atom ("COJKAO's C46 is BOTH a carbanion and an eta2 atom") and treats it as routine.

    The caps now take the low indices and the centroid block slides above them — OIN's ordering.
    """
    import sys

    sys.path.insert(0, "tests")
    import rxembed as rx
    from tests import test_haptic as th

    iso = next(iter(rx.metal(th.ferrocene())))
    cons = iso.cons
    assert cons.haptic, "fixture must carry a haptic face"
    before = sorted(cons.haptic)

    _metal._shift_phantoms(cons, 2)  # as if two D-caps had been appended ahead of the centroids
    after = sorted(cons.haptic)
    assert after == [i + 2 for i in before]
    assert sorted(cons.phantoms) == after, "phantoms must move with haptic"
    # THE COMPLETENESS CHECK: no field may still name an old index. This is what stops a future field from
    # being silently left behind — the same defect class as the hand-listed Constraints copies.
    stale = set(before) - set(after)
    for name in ("distances", "angles", "pulls", "floors", "dg_floors"):
        for k in getattr(cons, name):
            assert not (stale & set(k)), f"{name} still names a pre-shift dummy index {k}"
    for s in cons.spheres:
        assert not (stale & {d for d, _ring in s.haptic}), "the sphere recipe still names a pre-shift dummy"


@corpus_only
def test_a_shipped_metal_fixture_with_both_transients_embeds():
    """End-to-end over the shipped fixtures: a face AND a labile donor must not refuse each other.

    Filtered to metal structures that actually carry both — the organic TS fixtures have no metal, and a metal
    without a labile donor never exercised the collision. `tests/test_haptic.py` builds the synthetic pair that
    guarantees coverage even if no shipped .xyz qualifies.
    """
    import rxembed as rx

    failures, exercised = [], 0
    for path in _CORPUS:
        try:
            mol = _xyz_to_mol(path, 0)
        except Exception:
            continue
        if not _metal.metal_indices(mol):
            continue  # an organic TS fixture — nothing to enumerate
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
        # series — 14 in tmQM), which the sweep in `oin_adapter/` covers. Recorded rather than faked: a synthetic
        # carbanion does not survive `surrogate_metal`'s tag handling, so building one here would test something else.
        pytest.skip("no shipped fixture carries BOTH a haptic face and a labile donor — see oin_adapter/")
    assert failures == [], f"a face and a chirality cap still refuse each other: {failures}"
