"""Test metal identification, shape perception and surrogate restoration."""

from __future__ import annotations

import importlib
from importlib.util import find_spec

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed import bounds, core, metal_core, stereo
from rxembed.bounds import bounds_matrix
from rxembed.pipeline.perceive import read_xyz
from tests.conftest import EXAMPLES_DIR

emb = importlib.import_module("rxembed.embed")

_MN_H2 = str(EXAMPLES_DIR / "mn-h2.xyz")  # a frozen-TS bimetallic: Mn centre + a spectator ferrocene Fe


@pytest.mark.parametrize("operation", ["materialise"])
def test_haptic_centroids_rebuild_cached_topology(operation):
    mol = Chem.AddHs(Chem.MolFromSmiles("C=C.C=C"))
    count = mol.GetNumAtoms()
    haptic = {count: (0, 1), count + 1: (2, 3)}
    if operation == "strip":
        mol = metal_core.materialise_phantoms(mol, haptic)
    fresh = Chem.Mol(mol)
    cached = Chem.GetDistanceMatrix(mol).copy()

    def apply(candidate):
        if operation == "collapse":
            return metal_core.collapse_haptic(candidate, [0, 1, 2, 3])[0]
        if operation == "strip":
            return metal_core.strip_phantoms(candidate, set(haptic))
        return metal_core.materialise_phantoms(candidate, haptic)

    out, expected = apply(mol), apply(fresh)
    actual = Chem.GetDistanceMatrix(out).copy()
    np.testing.assert_array_equal(actual, Chem.GetDistanceMatrix(out, force=True))
    np.testing.assert_array_equal(bounds_matrix(out), bounds_matrix(expected))
    np.testing.assert_array_equal(Chem.GetDistanceMatrix(mol), cached)


@pytest.mark.parametrize(
    ("build", "check"),
    [
        (lambda: (Chem.MolFromSmiles("NN"), [0, 1]), lambda s, d: s == [(0, 1)]),
    ],
    ids=[
        "bonded-donor-pair-is-one-site-whatever-its-hydrogens-or-bond-order",
    ],
)
def test_haptic_sites_group_donors_into_pi_faces_and_isolated_sigma_sites(build, check):
    mol, donors = build()

    sites = metal_core.haptic_sites(mol, donors)

    assert check(sites, donors)


# --- which atoms are metal centres: one predicate behind every gate ---------------------------------------


# CN4 rather than CN3: `COPLANAR_TOL` is absolute, so a trigonal-planar sphere at the ~3.1 A the covalent-sum
# fallback gives an f-block centre pyramidalises past the accept gate (CeCl3 relaxes to 0.383 A out-of-plane
# against the 0.25 tol, and `minimize` says so). That is the tolerance's known scaling, not this predicate's.


# --- the surrogate round-trip: oxidation state ---------------------------------------------------------


# --- the surrogate round-trip: connectivity -------------------------------------------------------------


# --- metal-bound donor stereochemistry ------------------------------------------------------------------


# --- the donor's hand across the strip: a tag is a parity over the DONOR's bond order, not a symbol to copy -

# The M-L bond written FIRST at the P is the odd slot the surrogate strip mirrors; written LAST is the
# even control. `<-` is the dative arrow the README's complexes use; a bare bond is the covalent alternative.


_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _hand(mol, centre, order, cid=-1):
    """The tag naming this conformer's hand at `centre`, read in the fixed bond `order` given.

    RDKit's convention, pinned against RDKit itself by the first test below: negative volume is CW.
    """
    p = mol.GetConformer(cid).GetPositions()
    v = float(np.dot(np.cross(p[order[0]] - p[centre], p[order[1]] - p[centre]), p[order[2]] - p[centre]))
    return _TETRAHEDRAL[0] if v < 0 else _TETRAHEDRAL[1]


# the odd slot and the even control; `covalent_first` is the odd slot again, and is kept at the strip above


@pytest.mark.parametrize("tag", ["[S@@]"])
def test_external_dative_donor_stereo_survives_strip(tag):
    mol = Chem.AddHs(Chem.MolFromSmiles(f"Cl[Pd](Cl)(Cl)<-{tag}(=O)(C)CC"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    # sanitize=False because a strict sanitize rejects the Pd complex; the reader writes the tag either way
    back = Chem.MolFromMolBlock(Chem.MolToV3KMolBlock(mol), sanitize=False, removeHs=False)
    back.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(back, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    donor = next(a.GetIdx() for a in back.GetAtoms() if a.GetSymbol() == "S")
    bond = next(
        b
        for b in back.GetAtomWithIdx(donor).GetBonds()
        if back.GetAtomWithIdx(b.GetOtherAtomIdx(donor)).GetSymbol() == "Pd"
    )
    assert bond.GetBondType() == Chem.BondType.DATIVE, "the round trip did not keep the coordination bond"
    assert bond.GetBeginAtomIdx() == donor, "the dative bond does not leave the donor, so this asserts nothing"

    stripped, _m, _donors, _z, _q = metal_core.surrogate_metal(back)
    order = [b.GetOtherAtomIdx(donor) for b in stripped.GetAtomWithIdx(donor).GetBonds()]
    assert stripped.GetAtomWithIdx(donor).GetChiralTag() == _hand(stripped, donor, order)


@pytest.mark.parametrize("tag", ["@"])
@pytest.mark.parametrize("second_metal", ["Pt"])
def test_two_metal_bridge_uses_its_absolute_label_after_the_surrogate_strip(tag, second_metal):
    mol = Chem.AddHs(rx.parse_smiles(f"C[N{tag}H](->[Pd](Cl)(Cl)Cl)->[{second_metal}](Br)(Br)Br"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    iso = rx.metal(mol, center="all")[0]
    assert iso.mol.GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    embedded = core.embed(iso, n=2, params=rx.EmbedParams(seed=2, prune_rms=-1)).minimize()
    metals = set(metal_core.metal_indices(embedded.mol))
    donor = embedded.mol.GetAtomWithIdx(1)

    assert embedded.unrelaxed == []
    assert donor.GetChiralTag() in _TETRAHEDRAL
    assert stereo.defined_stereo_label(embedded.mol, metals) == iso.stereo_label
    assert {stereo.stereo_from_3d(Chem.Mol(embedded.mol, False, int(cid)), exclude=metals) for cid in embedded.ids} == {
        iso.stereo_label
    }


def test_multimetal_surrogate_repairs_stereo_orphaned_by_the_strip():
    mol = rx.parse_smiles("CC(O)=[S]->[Zn]")
    bond = mol.GetBondBetweenAtoms(1, 3)
    bond.SetStereoAtoms(0, 4)  # Zn is the sulfur-side E/Z reference before the coordination bond is stripped
    bond.SetStereo(Chem.BondStereo.STEREOZ)

    stripped, _metals = metal_core.surrogate_all_metals(mol)

    assert not _orphaned(stripped)


def test_direct_isomer_constructor_protects_a_tagged_amine_from_cleanup(monkeypatch):
    source = Chem.AddHs(rx.parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)"))
    expected = stereo.defined_stereo_label(source, {0})
    iso = rx.Isomer(source, "SPL", [1, 2, 3, 4])
    conformers = core.embed(iso, n=1, seed=2)

    assert iso.stereo_label == ""
    assert iso.mol.GetAtomWithIdx(4).GetChiralTag() in _TETRAHEDRAL
    assert emb.stereo_donor_bonds(iso.mol, iso) == [(4, 0)]

    def reflect(mol, _cons, *, conf_ids, max_iters, record=None, **_kwargs):
        for cid in conf_ids:
            if max_iters:
                positions = mol.GetConformer(cid).GetPositions()
                positions[:, 0] *= -1.0
                mol.GetConformer(cid).SetPositions(positions)
            if record is not None:
                record.statuses[cid] = 0
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", reflect)
    conformers._relax_constrained(emb.BASE_STIFFNESS, max_iters=10)

    assert conformers.unrelaxed == conformers.ids
    assert {
        stereo.stereo_from_3d(Chem.Mol(conformers.mol, False, int(cid)), exclude={iso.metal}) for cid in conformers.ids
    } == {expected}


@pytest.mark.parametrize("linker", ["[P@](C)(CC)CC[P@@](C)(CC)->2"])
def test_haptic_helpers_preserve_the_transient_donor_stereo_ring(linker, monkeypatch, tmp_path):
    isomers = rx.metal(f"[Pt+2]12(<-[Cl-])(<-[CH2]=[CH2]->1)<-{linker}", "SPL")
    captured = []
    original = metal_core.materialise_phantoms

    def observe(mol, haptic):
        out = original(mol, haptic)
        bonds = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE
        ]
        if haptic and bonds:
            assert len(bonds) == 2
            assert any({a for pair in bonds for a in pair} <= set(ring) for ring in out.GetRingInfo().AtomRings())
            oracle = Chem.Mol(mol)
            oracle.ClearComputedProps()
            oracle.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(oracle, includeDativeBonds=True)
            builder = Chem.RWMol(mol)
            for index in sorted(haptic):
                assert builder.AddAtom(Chem.Atom(6)) == index
            removed = metal_core.strip_phantoms(builder.GetMol(), set(haptic))
            with rdBase.BlockLogs():
                expected = rdDistGeom.GetMoleculeBoundsMatrix(oracle, doTriangleSmoothing=False)
                for candidate in (out, removed):
                    actual = rdDistGeom.GetMoleculeBoundsMatrix(candidate, doTriangleSmoothing=False)
                    np.testing.assert_allclose(actual[: len(expected), : len(expected)], expected, atol=1e-12, rtol=0)
            captured.append(bonds)
        return out

    monkeypatch.setattr(bounds, "materialise_phantoms", observe)
    assert isomers
    for iso in isomers:
        assert len(iso.haptic) == 1
        assert len(next(iter(iso.haptic.values()))) == 2
        assert len(emb.stereo_donor_bonds(iso.mol, iso)) == 2
        ensemble = rx.embed(iso, n=1, seed=2, threads=1)
        assert not ensemble.unrelaxed
        assert ensemble.check()[ensemble.ids[0]].ok()
        realised = stereo.stereo_from_3d(ensemble.mol, exclude=metal_core.metal_indices(ensemble.mol))
        assert stereo.point_stereo(realised) == stereo.point_stereo(iso.stereo_label)
        assert rx.cxsmiles(ensemble.mol) == rx.cxsmiles(iso)
        if find_spec("xyzgraph") is not None:
            path = tmp_path / "donor-stereo.xyz"
            Chem.MolToXYZFile(ensemble.mol, str(path))
            fresh = rx.read_xyz(str(path), charge=Chem.GetFormalCharge(ensemble.mol), bond_orders="xyz2mol")
            assert rx.cxsmiles(fresh) == rx.cxsmiles(iso)
    assert captured


# ---------------------------------------------------------------------------------------------------------
# RDKit state must stay valid across the metal-bond surgery. Stripping the M-donor bonds can invalidate RDKit
# state that was derived while the metal was still bonded, and RDKit does not always notice:
#
#   * a double bond left FLAGGED stereo with its two reference atoms dropped, because the metal was one of them.
#     RDKit's own ETKDG then indexes the empty vector and SEGFAULTS (rc 139), which no try/except can catch.
#   * `metal.surrogate_metal` re-imposing a STRICT sanitize on a structure the reader deliberately admitted leniently,
#     rejecting chemistry (an unkekulisable quinoid ring, a BPh4- boron) that perception had already accepted.
#
# These are asserted as INVARIANTS rather than as "structure X does not crash": the invariant is what the next
# surgery site has to honour, and it is what makes the guarantee checkable on any input.
# ---------------------------------------------------------------------------------------------------------


# rxembed's own fixtures. The invariants below must hold on any metal complex, so the gate runs on structures
# this repo ships rather than reaching into a sibling checkout: a unit suite that depends on an absolute path
# outside the project is not portable and is not a gate. A local benchmark corpus sweep is a measurement, not
# a gate, and lives in `benchmark/` where the corpus is in scope.


def _orphaned(mol):
    return [b for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE and len(b.GetStereoAtoms()) != 2]


@pytest.mark.skipif(find_spec("xyzgraph") is None, reason="needs rxembed[workflow]")
def test_ligands_reports_denticity_per_metal():
    mol = read_xyz(_MN_H2, metal_charges={0: 2, 1: 1})
    ligs = metal_core.ligands(mol)
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
    want = sorted(n.GetIdx() for m in metal_core.metal_indices(mol) for n in mol.GetAtomWithIdx(m).GetNeighbors())
    assert seen == want, "the ligands must partition the coordination sphere"
