"""Coordinate, periodic-table and vector math shared by the geometry gate and the coordination gate.

A dependency-free leaf: both ``geometry`` (the general checks) and ``coordination`` (the metal-sphere
gates) lean on these, and keeping them here is what lets the two gate modules import in one direction
without a cycle. Pure ``rdkit`` + ``numpy``, no domain knowledge beyond covalent/vdW radii.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()
_CARBON_Z = 6  # sp2-planarity is checked on carbons only (N is physically pyramidal — see geometry.planarity())
_EPS = 1e-9  # numerical floor for a degenerate cross-product / near-zero norm


def _positions(mol_or_pos, conf_id: int = -1) -> np.ndarray:
    """(N, 3) coordinates from a Mol conformer or a pass-through array."""
    if isinstance(mol_or_pos, np.ndarray):
        return mol_or_pos
    return mol_or_pos.GetConformer(conf_id).GetPositions()


def _rcov(z: int) -> float:
    return _PT.GetRcovalent(z)


def _rvdw(z: int) -> float:
    return _PT.GetRvdw(z)


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of ``a`` onto ``b`` after optimal rigid superposition (Kabsch)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt(((a @ r.T - b) ** 2).sum(1).mean()))


def _plane_offset(center: np.ndarray, neighbors: np.ndarray) -> float:
    """Distance of ``center`` from the best-fit plane of its 3 ``neighbors`` (0 = planar sp2)."""
    n = np.cross(neighbors[1] - neighbors[0], neighbors[2] - neighbors[0])
    norm = np.linalg.norm(n)
    if norm < _EPS:
        return 0.0
    return float(abs(np.dot(center - neighbors[0], n / norm)))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, w = a - b, c - b
    cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral p0-p1-p2-p3 in degrees."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
