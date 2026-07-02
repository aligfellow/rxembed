"""Per-conformer geometric descriptors for dedup.

Dihedrals (geometry) live here; the NCI binding-mode signature lives in
constraints.nci (xyzgraph). The rotatable-bond atom quads are computed once so
every conformer is described on the same axes.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

_ROT = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")


def _ref(mol, a, b):
    """Pick a reference neighbour of ``a`` (not ``b``) for the dihedral, preferring a heavy atom."""
    nbrs = [x.GetIdx() for x in mol.GetAtomWithIdx(a).GetNeighbors() if x.GetIdx() != b]
    heavy = [i for i in nbrs if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    return (heavy or nbrs or [None])[0]


def rotatable_quads(mol):
    """Heavy-atom dihedral 4-tuples (i, a, b, j), one per rotatable bond — stable across conformers."""
    quads = []
    for a, b in mol.GetSubstructMatches(_ROT):
        i, j = _ref(mol, a, b), _ref(mol, b, a)
        if i is not None and j is not None:
            quads.append((i, a, b, j))
    return quads


def dihedrals(mol, conf_id, quads=None):
    """(cos, sin) of each rotatable-bond dihedral — frame-invariant feature vector."""
    quads = rotatable_quads(mol) if quads is None else quads
    conf = mol.GetConformer(conf_id)
    v = []
    for q in quads:
        t = np.radians(rdMolTransforms.GetDihedralDeg(conf, *q))
        v += [np.cos(t), np.sin(t)]
    return np.array(v) if v else np.zeros(1)
