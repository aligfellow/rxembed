"""Geometry sanity check: is a generated conformer a real molecule rather than nonsense.

bonding_ok perceives bonds from the geometry (covalent-radius cutoffs) and checks
they match the molecular graph: no stretched/broken bond, no spurious fusion/clash.
Used by the pipeline to drop conformers that minimised into garbage.
"""

from __future__ import annotations

import numpy as np
from rdkit.Chem import GetPeriodicTable

_PT = GetPeriodicTable()
# d- and f-block metals: their coordinate/dative bonds are not governed by covalent-radius cutoffs.
_METAL_Z = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))


def bonding_ok(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset()):
    """Return True if geometry-perceived connectivity matches the graph (heavy atoms).

    Two things are skipped so *valid* geometries aren't rejected: pairs *inside* a frozen/reacting core
    (``exclude``) — a partial forming/breaking bond is held to the reference, not a ground-state bond — and
    any pair involving a metal, whose dative/coordinate distances covalent radii don't describe. A
    genuinely broken free-periphery bond still fails (the ensemble may legitimately empty).
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    metals = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in _METAL_Z}
    exclude = set(exclude)
    rcov = {i: _PT.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy}
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    for n, i in enumerate(heavy):
        for j in heavy[n + 1 :]:
            if (i in exclude and j in exclude) or i in metals or j in metals:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cut = rcov[i] + rcov[j]
            if frozenset((i, j)) in bonded:
                if d > bond_tol * cut or d < clash_tol * cut:  # bonded pair stretched/broken OR crushed
                    return False
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                return False
    return True
