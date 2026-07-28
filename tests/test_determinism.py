"""`rx.embed` at a fixed seed must not depend on how much randomness the process consumed earlier.

A scaffolding embed that forgets its seed does not fail loudly — it makes results depend on process history,
so a test passes alone and fails in a full suite, and a "bit-identical" refactor claim becomes unfalsifiable.
`_encounter_bounds` had exactly this defect: RDKit's `ETKDGv3().randomSeed` defaults to -1 (draw from the
global RNG), and the probe conformer it embeds decides which heavy-atom pair every inter-fragment bound is
keyed on. It reached any multi-fragment source — a TS complex, an NCI pair, a salt.
"""

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed.embed.dispatch import _encounter_bounds
from rxembed.inputs import _xyz_to_mol

_MULTI_FRAGMENT = "examples/structures/bimp.xyz"


def _burn_global_rng(n=64):
    """Consume RDKit global randomness, standing in for whatever ran before us in a real session."""
    for _ in range(n):
        m = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
        rdDistGeom.EmbedMolecule(m, rdDistGeom.ETKDGv3())  # deliberately unseeded


def test_encounter_bounds_are_independent_of_global_rng_state():
    mol = _xyz_to_mol(_MULTI_FRAGMENT, 0)
    assert len(Chem.GetMolFrags(mol)) >= 2, "fixture must be multi-fragment to exercise the encounter bounds"

    before = _encounter_bounds(mol)
    _burn_global_rng()
    after = _encounter_bounds(mol)
    assert before == after, "encounter bounds moved after unrelated randomness was consumed"


def test_encounter_bounds_are_reproducible_across_calls():
    mol = _xyz_to_mol(_MULTI_FRAGMENT, 0)
    assert _encounter_bounds(mol) == _encounter_bounds(mol)


def test_a_different_probe_seed_is_allowed_to_differ():
    """The seed is a real parameter, not a constant baked past the signature — otherwise the guard is vacuous."""
    mol = _xyz_to_mol(_MULTI_FRAGMENT, 0)
    fixed = _encounter_bounds(mol)
    assert _encounter_bounds(mol, seed=0xF00D) == fixed
