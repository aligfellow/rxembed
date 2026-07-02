"""Geometry gate -- validate that an embedded conformer is chemically sound.

The point of freezing a core (or grafting a reacting TS back by Kabsch) is that the *free periphery*
still has to come out right. A perfect frozen core and perfect constraint distances can coexist with a
puckered aromatic ring, a twisted amide (broken conjugation), an out-of-plane N-H, a stretched X-H, or
two atoms embedded on top of each other. Distance-window checks miss all of that.

``check()`` runs the physical checks and returns a ``GeometryReport``: truthy when clean, printable, with
``.assert_ok()`` for tests. Pure ``rdkit`` + ``numpy`` — no optional deps, usable on any conformer.

    rep = rxembed.geometry.check(mol, conf_id, frozen=core, reference=ts_mol,
                                   constraints={"distances": {(1, 9): (2.6, 3.0)}})
    rep.assert_ok()          # raises with a readable summary if any check fails
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()
# transition / lanthanide / actinide metals — their coordination distances are dative, not vdW clashes,
# and their "bonds" aren't covalent (same set metrics.bonding_ok uses); check() excludes them so a metal
# centre isn't read as clashing with its own ligands (ligand-ligand geometry is still fully checked).
_METAL_Z = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
_CARBON_Z = 6  # sp2-planarity is checked on carbons only (N is physically pyramidal — see planarity())
_SP2_DEGREE = 3  # a planar sp2 centre has exactly three neighbours
_EPS = 1e-9  # numerical floor for a degenerate cross-product / near-zero norm


@dataclass(frozen=True)
class Violation:
    """One failed check. ``value`` breached ``limit`` for the atoms named."""

    kind: str
    atoms: tuple[int, ...]
    value: float
    limit: float
    detail: str = ""

    def __str__(self) -> str:
        """Format the violation as a readable one-liner."""
        a = "-".join(map(str, self.atoms))
        return f"[{self.kind}] atoms {a}: {self.value:.3f} vs {self.limit:.3f} {self.detail}".rstrip()


@dataclass
class GeometryReport:
    """Result of ``check()``. Falsy/``ok()`` when there are no violations."""

    violations: list[Violation] = field(default_factory=list)

    def ok(self) -> bool:
        """Return True when the conformer passed every check."""
        return not self.violations

    def __bool__(self) -> bool:
        """Truthy when there are no violations."""
        return self.ok()

    def summary(self) -> str:
        """Format the violations one per line (or an all-clear message)."""
        if self.ok():
            return "geometry OK (no violations)"
        lines = [f"{len(self.violations)} geometry violation(s):"]
        lines += [f"  - {v}" for v in self.violations]
        return "\n".join(lines)

    def assert_ok(self) -> None:
        """Raise ``AssertionError`` with the summary if any check failed (for tests)."""
        if not self.ok():
            raise AssertionError(self.summary())


# --- geometry helpers --------------------------------------------------------


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


# --- the checks --------------------------------------------------------------


def bond_lengths(mol, pos, lo: float = 0.7, hi: float = 1.3, exclude=frozenset()) -> list[Violation]:
    """Every bond within ``[lo, hi]`` x the sum of covalent radii (no stretched/compressed bonds).

    ``exclude`` atoms (a frozen/reacting TS core) are skipped — their partial bonds are held to the
    reference by design, not ground-state bonds.
    """
    out = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in exclude or j in exclude:
            continue
        d = float(np.linalg.norm(pos[i] - pos[j]))
        ideal = _rcov(mol.GetAtomWithIdx(i).GetAtomicNum()) + _rcov(mol.GetAtomWithIdx(j).GetAtomicNum())
        if d < lo * ideal:
            out.append(Violation("bond_length", (i, j), d, lo * ideal, "compressed"))
        elif d > hi * ideal:
            out.append(Violation("bond_length", (i, j), d, hi * ideal, "stretched"))
    return out


def hydrogens(mol, pos, lo: float = 0.8, hi: float = 1.3, exclude=frozenset()) -> list[Violation]:
    """Each H bonded to exactly one heavy atom at a sane X-H length (bad-H-position detector)."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = atom.GetNeighbors()
        h = atom.GetIdx()
        if h in exclude or (nbrs and nbrs[0].GetIdx() in exclude):
            continue
        if len(nbrs) != 1:
            out.append(Violation("hydrogen", (h,), float(len(nbrs)), 1.0, "H not bonded to exactly one atom"))
            continue
        x = nbrs[0].GetIdx()
        d = float(np.linalg.norm(pos[h] - pos[x]))
        if d < lo or d > hi:
            out.append(Violation("hydrogen", (x, h), d, hi if d > hi else lo, "X-H length"))
    return out


def clashes(
    mol, pos, heavy_vdw: float = 0.7, hh_floor: float = 1.5, xh_cov: float = 1.1, exclude=frozenset()
) -> list[Violation]:
    """No non-bonded pair (excluding 1-3 angle pairs) in steric overlap.

    Thresholds differ by kind so real non-covalent contacts (H-bonds, halogen/π) are not flagged:
    heavy-heavy overlap at `heavy_vdw` x sum-of-vdW (a genuine steric clash sits well inside vdW); H...H
    below `hh_floor` Å; and an H buried inside a non-bonded heavy atom at `xh_cov` x sum-of-covalent
    (small enough that a 1.8 Å H-bond H...acceptor passes, large enough to catch a buried H).
    ``exclude`` atoms (a frozen/reacting core) are skipped — a forming/breaking bond reads as a clash.
    """
    excluded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        excluded.add((min(i, j), max(i, j)))
    for atom in mol.GetAtoms():  # 1-3 pairs (share a neighbour) are governed by angles, not clashes
        nb = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(nb)):
            for c in range(a + 1, len(nb)):
                excluded.add((min(nb[a], nb[c]), max(nb[a], nb[c])))
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    out = []
    n = mol.GetNumAtoms()
    for i in range(n):
        if i in exclude:
            continue
        for j in range(i + 1, n):
            if j in exclude or (i, j) in excluded:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if z[i] == 1 and z[j] == 1:
                limit, why = hh_floor, "H...H clash"
            elif z[i] == 1 or z[j] == 1:
                limit, why = xh_cov * (_rcov(z[i]) + _rcov(z[j])), "H buried in heavy atom"
            else:
                limit, why = heavy_vdw * (_rvdw(z[i]) + _rvdw(z[j])), "heavy-atom steric overlap"
            if d < limit:
                out.append(Violation("clash", (i, j), d, limit, why))
    return out


def planarity(mol, pos, oop: float = 0.15, ring_rms: float = 0.10, exclude=frozenset()) -> list[Violation]:
    """sp2 carbons stay planar and aromatic rings stay flat (broken-conjugation detector).

    Catches puckered aromatic rings and twisted sp2 carbons (C=C / C=O / aromatic C). Nitrogen is left
    out on purpose: an amine / aniline N is *physically* pyramidal even when RDKit labels it sp2, so a
    blanket N-planarity rule false-positives. The twisted-amide / broken-conjugation case (the true signal
    at a conjugating N/O) is handled by ``conjugation()`` via the O=C-N dihedral.
    """
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != _CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        if atom.GetIdx() in exclude:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) != _SP2_DEGREE:
            continue
        off = _plane_offset(pos[atom.GetIdx()], pos[nbrs])
        if off > oop:
            out.append(Violation("planarity", (atom.GetIdx(), *nbrs), off, oop, "sp2 out of plane"))
    for ring in mol.GetRingInfo().AtomRings():
        if any(i in exclude for i in ring) or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        p = pos[list(ring)]
        rms = float(np.sqrt((np.linalg.svd(p - p.mean(0))[1][2] ** 2) / len(ring)))
        if rms > ring_rms:
            out.append(Violation("planarity", tuple(ring), rms, ring_rms, "aromatic ring puckered"))
    return out


def conjugation(mol, pos, tol_deg: float = 30.0, exclude=frozenset()) -> list[Violation]:
    """Amide / ester / enamine / enol-ether conjugation stays planar (broken-conjugation detector).

    For each single bond X-C where X is N/O and C bears a double bond (C=O or C=C), the substituent must
    be coplanar with the π system: the A=C-X-S dihedral near 0 or 180 deg. Catches a twisted amide (incl.
    an out-of-plane amide N-H, S=H) or enamine that local sp2-planarity misses. Rings and biaryls are left
    out — a lactam is locked by its ring and biaryls legitimately twist (atropisomers).
    """
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != _CARBON_Z:
                continue
            if x_atom.GetIdx() in exclude or c_atom.GetIdx() in exclude:
                continue
            dbl = [
                n
                for n in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [n for n in x_atom.GetNeighbors() if n.GetIdx() != c_atom.GetIdx()]
            if not dbl or not subs:
                continue
            a, x, c, s = dbl[0].GetIdx(), x_atom.GetIdx(), c_atom.GetIdx(), subs[0].GetIdx()
            dih = abs(_dihedral(pos[a], pos[c], pos[x], pos[s]))
            dev = min(dih, abs(180.0 - dih))
            if dev > tol_deg:
                out.append(Violation("conjugation", (a, c, x, s), dev, tol_deg, "twisted out of plane"))
    return out


def frozen_core(mol, pos, frozen, reference, tol: float = 0.05) -> list[Violation]:
    """Frozen atoms held to ``reference`` within ``tol`` Å after Kabsch superposition."""
    frozen = list(frozen)
    ref = _positions(reference)
    rmsd = _kabsch_rmsd(pos[frozen], ref[frozen])
    if rmsd > tol:
        return [Violation("frozen_core", tuple(frozen), rmsd, tol, "core moved")]
    return []


def check_constraints(mol, pos, spec, dist_slack: float = 0.15, ang_slack: float = 5.0) -> list[Violation]:
    """Seeded ``distances`` / ``angles`` realised within their window (+ slack).

    ``spec`` is a mapping or Constraints-like object exposing ``distances`` / ``angles``.
    """
    dists = _attr(spec, "distances", {})
    angles = _attr(spec, "angles", {})
    out = []
    for (i, j), win in dists.items():
        lo, hi = win if isinstance(win, tuple) else (win, win)
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if not (lo - dist_slack <= d <= hi + dist_slack):
            out.append(Violation("constraint", (i, j), d, hi + dist_slack, f"distance window [{lo}, {hi}]"))
    for (i, j, k), (lo, hi) in angles.items():
        a = _angle(pos[i], pos[j], pos[k])
        if not (lo - ang_slack <= a <= hi + ang_slack):
            out.append(Violation("constraint", (i, j, k), a, hi + ang_slack, f"angle window [{lo}, {hi}]"))
    return out


def stereo(mol, pos, reference, conf_id: int = -1) -> list[Violation]:
    """No stereocentre inverted vs ``reference`` (CIP tags from 3D)."""
    got = _cip(mol, conf_id)
    want = _cip(reference)
    out = []
    for idx, tag in want.items():
        if got.get(idx) != tag:
            out.append(Violation("stereo", (idx,), 0.0, 0.0, f"{tag} -> {got.get(idx)}"))
    return out


# --- top-level entry ---------------------------------------------------------


def check(mol, conf_id: int = -1, *, frozen=None, reference=None, constraints=None) -> GeometryReport:
    """Run the full physical gate on one conformer and collect all violations.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with at least one conformer.
    conf_id : int
        Conformer id to check (default: the last / active one).
    frozen : sequence of int, optional
        The frozen / reacting TS core. Held to ``reference`` by an RMSD check and **excluded from the
        ground-state checks** (a forming/breaking bond or a distorted reacting sp2 is correct by design,
        not a bad geometry) — only the free periphery is vetted for clashes / conjugation / H-placement.
    reference : rdkit.Chem.Mol or .xyz path, optional
        Reference geometry for the frozen-core RMSD and stereo checks.
    constraints : mapping or Constraints-like, optional
        ``distances`` / ``angles`` to confirm the seed was realised.
    """
    if isinstance(reference, str):
        reference = Chem.MolFromXYZFile(reference)  # coords only; atom order must match `mol`
    pos = _positions(mol, conf_id)
    exclude = set(frozen) if frozen is not None else set()
    exclude |= {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z}  # dative, not vdW
    exclude = frozenset(exclude)
    v: list[Violation] = []
    v += bond_lengths(mol, pos, exclude=exclude)
    v += hydrogens(mol, pos, exclude=exclude)
    v += clashes(mol, pos, exclude=exclude)
    v += planarity(mol, pos, exclude=exclude)
    v += conjugation(mol, pos, exclude=exclude)
    if frozen is not None and reference is not None:
        v += frozen_core(mol, pos, frozen, reference)
    if constraints is not None:
        v += check_constraints(mol, pos, constraints)
    if reference is not None:
        v += stereo(mol, pos, reference, conf_id)
    return GeometryReport(v)


# --- small private utilities -------------------------------------------------


def _attr(obj, name, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


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


def _cip(mol, conf_id: int = -1) -> dict[int, str]:
    m = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(m, confId=conf_id if conf_id >= 0 else m.GetNumConformers() - 1)
    out = {}
    for atom in m.GetAtoms():
        if atom.HasProp("_CIPCode"):
            out[atom.GetIdx()] = atom.GetProp("_CIPCode")
    return out
