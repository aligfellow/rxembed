"""Organocatalysis — transfer a known reacting-core geometry across a backbone series with ``template``.

The headline capability: hold one known core geometry and place it on a family of related scaffolds. Here
a conserved amide motif's geometry (read from one embedded reference) is templated onto three substituted
analogues; each transfer must preserve the core exactly and leave a clean periphery. No fixtures, no xtb —
the reference is generated in-process so the test is self-contained.
"""

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import geometry as geom

_GRAFT_TOL = 0.01
_CORE = [0, 1, 2, 3]  # the conserved C-C(=O)-N motif — same leading indices in every analogue below


def _reference_core_positions(seed=5):
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m.GetConformer().GetPositions()


@pytest.mark.parametrize(
    "analogue",
    [
        "CC(=O)Nc1ccccc1",  # parent
        "CC(=O)Nc1ccc(C)cc1",  # para-methyl
        "CC(=O)Nc1ccc(C(C)(C)C)cc1",  # para-tert-butyl (the "change the backbone" case)
    ],
)
def test_core_transfers_across_backbones(analogue):
    import rxembed as rx

    ref_pos = _reference_core_positions()
    ens = rx.embed(analogue, template=(ref_pos, {i: i for i in _CORE}), n=6)
    assert ens.n >= 1
    for cid in ens.ids:
        pos = ens.mol.GetConformer(cid).GetPositions()
        drift = max(
            abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j]))
            for i, j in itertools.combinations(_CORE, 2)
        )
        assert drift < _GRAFT_TOL  # reacting core transferred exactly
        geom.check(ens.mol, cid).assert_ok()  # backbone clean
