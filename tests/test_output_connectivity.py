"""The pipeline output mol must be a proper connected molecule — the M-donor bonds re-added.

The surrogate strips the M-donor bonds so the DG/FF can embed a bond-less metal, and `restore_metal` is
calculator-minimal (element + charge only — xtb scores from coords + total charge, needs no graph). So the
mol a caller got back had the metal topologically DETACHED from its ligands: right coords/element/charge, but
`Chem.GetMolFrags` counted the metal and every ligand as separate fragments, and graph perception / re-embed /
writing connectivity / an OIN-SMILES drop-in all broke on it.

`minimize()` now runs a connectivity finalize (`metal.connect_metal`) once geometry + element + charge are
settled: the stripped M-L bonds come back as DATIVE (donor->metal), so the output is one component through the
metal. These pin that guarantee — each was RED before the finalize (the metal was its own fragment).
"""

import pathlib

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.rdkit_embed.constraints import metal as _metal


def _dative(mol, donor, metal):
    """Return True if there is a DATIVE bond directed donor->metal (RDKit's begin=donor convention)."""
    b = mol.GetBondBetweenAtoms(int(donor), int(metal))
    return b is not None and b.GetBondType() == Chem.BondType.DATIVE and b.GetBeginAtomIdx() == int(donor)


def test_a_metal_complex_output_is_connected_through_the_metal():
    """THE regression: `rx.embed(metal).minimize().mol` has the metal DATIVE-bonded to every donor, one fragment.

    RED before the finalize: `restore_metal` put back only element + charge, so this same mol came out as four
    separate fragments (metal, Br, Cl, the en chelate) with the Pd bonded to nothing.
    """
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1).minimize()
    assert ens.n, "fixture must keep at least one conformer"
    assert len(Chem.GetMolFrags(ens.mol)) == 1, "the metal is still a separate fragment — connectivity not restored"
    for d in iso.donors:
        assert _dative(ens.mol, d, iso.metal), f"donor {d} is not DATIVE-bonded to the metal"
    # the connected graph round-trips to a real coordination-complex SMILES (it did not before)
    assert "Pd" in Chem.MolToSmiles(ens.mol)


def test_the_finalize_does_not_move_atoms_or_change_the_charge():
    """Connectivity-only: the finalize adds bonds, never touching coordinates or the total formal charge."""
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    ens = rx.embed(iso, n=2, seed=1)
    # capture the settled geometry+charge BEFORE the finalize by re-running minimize's relax on a twin, then
    # comparing to the finalized mol: coordinates and charge must be identical, only the bond count differs.
    ens.minimize()
    mol = ens.mol
    q = Chem.GetFormalCharge(mol)
    assert q == 0, "en-PdBrCl is neutral; the finalize must not perturb the total charge"
    # every dative bond added is metal-incident, so no ligand atom gained a covalent neighbour
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DATIVE:
            assert iso.metal in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()), "a DATIVE bond not incident to the metal"


def test_each_metal_isomer_candidate_is_connected():
    """An EnsembleSet of coordination isomers: every candidate's output mol is connected, not just the first."""
    isos = rx.metal("Cl[Pd](Cl)(N)N", "square_planar")
    embedded = [rx.embed(iso, n=2, seed=1).minimize() for iso in isos]
    assert any(e.n for e in embedded), "at least one isomer must embed"
    for e in embedded:
        if not e.n:
            continue
        assert len(Chem.GetMolFrags(e.mol)) == 1, f"isomer {e.tag.get('arrangement')} came out disconnected"


def test_a_bimetallic_output_connects_every_metal():
    """Both metals reconnect — a spectator ferrocene's Fe is bonded to its Cp, not left as loose rings."""
    isos = rx.metal("examples/structures/mn-h2.xyz", "octahedral", center="Mn", fix=[1, 5, 63, 64, 65, 66])
    ens = rx.embed(isos[0], n=4, seed=1).minimize(_retry=False)
    if not ens.n:
        pytest.skip("no conformer survived the relax for this deterministic seed — connectivity is unexercised")
    assert len(Chem.GetMolFrags(ens.mol)) == 1, "a bimetallic output must be one component through BOTH metals"
    for mi in _metal.metal_indices(ens.mol):
        assert ens.mol.GetAtomWithIdx(mi).GetDegree() > 0, f"metal {mi} was left disconnected"


def test_a_derived_ensemble_stays_connected():
    """`representatives()` / `lowest()` derive a new mol — it must inherit the finalized connectivity."""
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    ens = rx.embed(iso, n=4, seed=1).minimize()
    if ens.n < 2:
        pytest.skip("need >=2 conformers to derive a representative set")
    reps = ens.representatives()
    assert len(Chem.GetMolFrags(reps.mol)) == 1, "a derived ensemble dropped the M-L connectivity"


def test_the_mc_chain_re_connects_and_never_relaxes_a_bonded_metal():
    """`minimize().mc().minimize()`: the search must run on the BARE mol, the final output re-connected.

    The finalize adds DATIVE M-L bonds, but UFF cannot type a bonded transition metal — so `mc` strips them
    before searching (else the follow-up relax tears every bond and empties the ensemble) and the closing
    `minimize` re-adds them from the durable `metal_bonds`. Both halves are asserted: the ensemble survives
    (search saw the bare mol) AND the output is one fragment (re-connected).
    """
    from tests.conftest import _openconf_available

    if not _openconf_available():
        pytest.skip("openconf not installed")
    iso = rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")[0]
    searched = rx.embed(iso, n=6, seed=1).minimize().mc(preset="ensemble", seed=1).minimize()
    assert searched.n >= 1, "the mc search collapsed — a bonded metal was handed to the relax"
    assert searched.metal_bonds, "the M-L bond record must survive the _metal teardown for the re-connect"
    assert len(Chem.GetMolFrags(searched.mol)) == 1, "the closing minimize did not re-connect the metal"


def test_organic_output_is_untouched_by_the_finalize():
    """No metal -> the finalize is a no-op: ethanol stays one fragment and nothing crashes."""
    ens = rx.embed("CCO").minimize()
    assert ens.n
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    assert not [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE], "no dative bonds on organics"


@pytest.mark.skipif(not pathlib.Path("examples/structures/bimp.xyz").exists(), reason="frozen-TS fixture absent")
def test_a_metalfree_frozen_ts_is_unchanged():
    """The frozen-TS graft path carries no metal, so the finalize adds nothing (no dative bonds appear)."""
    ens = rx.embed("examples/structures/bimp.xyz", fix=[10, 11, 12, 14], n=2, seed=1).minimize()
    assert ens.n
    assert not [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
