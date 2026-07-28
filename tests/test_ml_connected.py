"""The user-facing ``Ensemble.mol`` must ALWAYS be a connected molecule — on ANY access, at ANY stage.

The input SMILES has the M-L bonds; they must always come back. The surrogate strips them (and swaps the
metal to a carbon) so the DG/FF can embed a bond-less metal, so the *internal working mol* (``_mol``) is that
bond-less surrogate — but the public ``.mol`` property finalizes the connected graph on access (real metal
element + oxidation state + the M-donor DATIVE bonds), reusing ``minimize``'s own finalize machinery.

``test_output_connectivity.py`` pins the terminal (post-``minimize``) paths. THESE pin the gap that was still
open after commit 44d2828: a **bare** ``rx.embed(metal).mol`` (no ``.minimize()``), the ``.mc()`` output, and
every derived accessor — plus the hard invariant that the engine still sees the bond-less surrogate.
"""

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.rdkit_embed.constraints import metal as _metal

_EN_PDBRCL = "Br[Pd]1(Cl)NCCN1"  # neutral en-PdBrCl — reliably embeds, the fixture the connectivity suite uses


def _dative(mol, donor, metal):
    """True if a DATIVE bond runs donor->metal (RDKit's begin=donor convention)."""
    b = mol.GetBondBetweenAtoms(int(donor), int(metal))
    return b is not None and b.GetBondType() == Chem.BondType.DATIVE and b.GetBeginAtomIdx() == int(donor)


def _assert_connected(mol, iso):
    """The finalized mol is one fragment, real metal element, every donor DATIVE-bonded to it."""
    assert len(Chem.GetMolFrags(mol)) == 1, "the metal is a separate fragment — .mol is not connected"
    assert mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z, "the metal is not its real element"
    assert mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == iso.real_q, "the metal lost its oxidation state"
    for d in iso.donors:
        assert _dative(mol, d, iso.metal), f"donor {d} is not DATIVE-bonded to the metal"


def test_bare_embed_mol_is_already_connected():
    """THE gap: ``rx.embed(metal).mol`` — no ``.minimize()`` — is a connected complex, not the surrogate."""
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1)
    assert not ens._minimized, "fixture must be a bare, un-minimized embed"
    _assert_connected(ens.mol, iso)
    assert "Pd" in Chem.MolToSmiles(ens.mol), "the connected graph must round-trip to a real complex SMILES"


def test_the_internal_working_mol_stays_the_bond_less_surrogate():
    """The hard constraint: ``_mol`` is the bond-less carbon surrogate the engine needs, decoupled from ``.mol``.

    Restoring the element or re-adding the M-L bonds on the working mol would break the DG/FF (UFF cannot type
    a bonded transition metal). So the pre-minimize ``_mol`` must still be a bond-less carbon at the metal.
    """
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1)
    m = ens._metal.metal
    assert ens._mol.GetAtomWithIdx(m).GetAtomicNum() == _metal.SURROGATE, "the working mol lost its carbon surrogate"
    assert len(Chem.GetMolFrags(ens._mol)) > 1, "the working mol must stay the bond-less surrogate (M-L stripped)"
    # and the public accessor did NOT mutate that working mol
    _ = ens.mol
    assert ens._mol.GetAtomWithIdx(m).GetAtomicNum() == _metal.SURROGATE, ".mol access mutated the working surrogate"
    assert len(Chem.GetMolFrags(ens._mol)) > 1, ".mol access re-bonded the working surrogate"


def test_finalize_is_connectivity_only_never_moves_an_atom():
    """The accessor adds bonds / swaps the element on a COPY; coordinates are byte-for-byte the working mol's."""
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1)
    for cid in ens.ids:
        work = ens._mol.GetConformer(cid).GetPositions()
        pub = ens.mol.GetConformer(cid).GetPositions()
        assert np.allclose(work, pub), "the finalize moved atoms — it must be connectivity-only"


@pytest.mark.parametrize(
    "access",
    [
        lambda e: e,  # bare embed
        lambda e: e.minimize(),
        lambda e: e.minimize().score("ff"),  # FF single point — no xtb binary needed
        lambda e: e.minimize().representatives(),
        lambda e: e.minimize().lowest(1),
        lambda e: e.minimize().align(),
        lambda e: e.minimize()[0],
        lambda e: e.minimize().prune(),
        lambda e: e.align(),  # derived from a PRE-minimize ensemble (carries the live _metal ctx)
    ],
    ids=["bare", "minimize", "score_ff", "representatives", "lowest", "align", "getitem", "prune", "bare_align"],
)
def test_every_mol_returning_path_is_connected(access):
    """Every accessor that hands back a mol — terminal or derived — yields the connected complex."""
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    out = access(rx.embed(iso, n=3, seed=1))
    if out.n == 0:
        pytest.skip("no conformer survived for this deterministic seed")
    _assert_connected(out.mol, iso)


def test_mc_output_is_connected():
    """``rx.embed(metal).mc().mol`` (pre-minimize, openconf-driven) is a connected complex."""
    from tests.conftest import _openconf_available

    if not _openconf_available():
        pytest.skip("openconf not installed")
    iso = rx.metal(_EN_PDBRCL, "square_planar")[0]
    ens = rx.embed(iso, n=3, seed=1).mc(seed=1)
    _assert_connected(ens.mol, iso)


def test_ensembleset_candidates_are_each_connected_bare():
    """Every candidate of a bare ``EnsembleSet`` (metal isomers, no ``.minimize()``) finalizes connected."""
    es = rx.embed("CCCN[Pd](Cl)(Cl)NCCC", metal="square_planar", n=3, seed=1)
    assert isinstance(es, rx.EnsembleSet)
    for cand in es:
        if cand.n == 0:
            continue
        assert len(Chem.GetMolFrags(cand.mol)) == 1, f"candidate {cand.tag} came out disconnected on bare .mol"
        assert [b for b in cand.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE], "no DATIVE M-L bonds"


def test_organic_mol_is_an_untouched_no_op():
    """No metal -> ``.mol`` passes ``_mol`` through unchanged: same object, one fragment, no dative bonds."""
    ens = rx.embed("CCO", n=2, seed=1)
    assert ens.mol is ens._mol, "organic .mol must be the working mol itself (a pure no-op, identity preserved)"
    assert len(Chem.GetMolFrags(ens.mol)) == 1
    assert not [b for b in ens.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
    mini = ens.minimize()
    assert len(Chem.GetMolFrags(mini.mol)) == 1
    assert not [b for b in mini.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
