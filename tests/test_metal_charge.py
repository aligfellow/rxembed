"""The metal's oxidation state must survive the carbon-surrogate round-trip (``prepare``/``restore``).

``prepare`` swaps the metal for a neutral carbon surrogate; ``restore`` must hand back both element AND formal
charge, else ``_calc_charge`` sends xtb a total wrong by the oxidation state. Pinned at both restore sites
(pipeline ``_MetalCtx`` and ``Isomer``) and at the number that reaches the calculator. RDKit + UFF, no xtb.
"""

from rdkit import Chem

import rxembed as rx
from rxembed.embed.dispatch import _xyz_to_mol
from rxembed.rdkit_embed.constraints import metal as _metal

_MN_H2 = "examples/structures/mn-h2.xyz"
_MN_H2_RC = [1, 5, 63, 64, 65, 66]  # the reacting core to hold (see test_connectivity)
_HENRY = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"

# (name, SMILES, net charge) — a spread of oxidation states and a genuinely neutral metal that must stay 0
_COMPLEXES = [
    ("henry Ni(II) amidate/carboxylate (net 0)", _HENRY, 0),
    ("MeCN-PdCl2 (net 0)", "CC#N[Pd](Cl)Cl", 0),
    ("[Pd(NH3)4]2+ (cation)", "[NH3]->[Pd+2](<-[NH3])(<-[NH3])<-[NH3]", 2),
    ("[PdCl4]2- (anion)", "[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]", -2),
    ("Fe(CO)5 (neutral metal)", "[Fe](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]", 0),
]


def _one(ens):
    """Collapse an EnsembleSet-or-Ensemble down to a single Ensemble."""
    return ens.candidates[0] if hasattr(ens, "candidates") else ens


def test_prepare_restore_is_a_charge_identity_on_the_metal_atom():
    """``restore`` hands the metal back element + formal charge; ``prepare`` neutralises the surrogate in between."""
    m0 = Chem.MolFromSmiles("[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    metal_idx = _metal.metal_index(m0)
    q0 = m0.GetAtomWithIdx(metal_idx).GetFormalCharge()
    assert q0 == 2  # Pd(II)

    surrogate, m, _donors, real_z, real_q = _metal.surrogate_metal(m0)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == _metal.SURROGATE
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == 0  # neutral in the DG — this is deliberate
    assert real_q == q0  # ...but the oxidation state was captured, not thrown away

    _metal.restore_metal(surrogate, m, real_z, real_q)
    assert surrogate.GetAtomWithIdx(m).GetAtomicNum() == real_z
    assert surrogate.GetAtomWithIdx(m).GetFormalCharge() == q0  # handed back


def _assert_roundtrip(name, mol_charge, ens):
    """After minimize, the ensemble mol and the charge sent to the calculator both equal the input charge."""
    ens = _one(ens).minimize()
    assert Chem.GetFormalCharge(ens.mol) == mol_charge, (
        f"{name}: mol charge {Chem.GetFormalCharge(ens.mol):+d} != input {mol_charge:+d} "
        f"— the metal's oxidation state was lost across the surrogate round-trip"
    )
    # the number that reaches xtb — pinned explicitly so a _calc_charge refactor can't silently re-break it
    assert ens._calc_charge(None) == mol_charge, (
        f"{name}: _calc_charge sends {ens._calc_charge(None):+d}, not the true total {mol_charge:+d}"
    )
    return ens


def test_charge_roundtrips_through_embed_isomer_minimize():
    """Every complex: ``rx.embed(rx.metal(...)[0]).minimize`` returns the input's net charge (dispatch _MetalCtx)."""
    for name, smiles, q in _COMPLEXES:
        iso = rx.metal(smiles)[0]  # default geometry for the donor count
        _assert_roundtrip(name, q, rx.embed(iso, n=2, seed=1))


def test_charge_roundtrips_through_smiles_constrain_path():
    """The ``rx.embed(smiles, constrain=...)`` surrogate path (also dispatch _MetalCtx) preserves net charge."""
    name, smiles, q = _COMPLEXES[0]  # henry
    ens = rx.embed(smiles, constrain={(0, 1): (1.4, 1.6)}, n=2, seed=1)
    _assert_roundtrip(name, q, ens)


def test_neutral_metal_stays_neutral():
    """A charge-free metal (Fe(CO)5, a Pd(0)) must round-trip to 0 — not gain a phantom charge."""
    ens = _assert_roundtrip("Fe(CO)5", 0, rx.embed(rx.metal(_COMPLEXES[4][1])[0], n=2, seed=1))
    fe = next(a for a in ens.mol.GetAtoms() if a.GetAtomicNum() == 26)
    assert fe.GetFormalCharge() == 0


def test_multi_metal_spectator_charge_roundtrips_via_embed_isomer():
    """mn-h2 (Mn + spectator ferrocene Fe): both metals' oxidation states restore (the ``extra`` 3-tuple path)."""
    mol0 = _xyz_to_mol(_MN_H2, 0)
    q_in = Chem.GetFormalCharge(mol0)
    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    _assert_roundtrip("mn-h2 (embed isomer)", q_in, rx.embed(isos[0], n=2, seed=1))


def test_multi_metal_charge_roundtrips_via_minimize():
    """``rx.minimize(mn-h2.xyz)`` exercises the pipeline's own ``_MetalCtx`` restore (the second code site)."""
    mol0 = _xyz_to_mol(_MN_H2, 0)
    q_in = Chem.GetFormalCharge(mol0)
    _assert_roundtrip("mn-h2 (minimize)", q_in, rx.minimize(_MN_H2))


def test_isomer_restore_method_restores_charge_directly():
    """``Isomer.restore`` hands every metal (incl. the spectator) back its oxidation state, not a silent zero."""
    isos = rx.metal(_MN_H2, "octahedral", center="Mn", fix=_MN_H2_RC)
    iso = isos[0]
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == _metal.SURROGATE  # still the neutral carbon
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == 0
    iso.restore()
    assert iso.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() == iso.real_z
    assert iso.mol.GetAtomWithIdx(iso.metal).GetFormalCharge() == iso.real_q
    for mi, _rz, rq in iso.extra:  # the spectator ferrocene Fe
        assert iso.mol.GetAtomWithIdx(mi).GetFormalCharge() == rq
