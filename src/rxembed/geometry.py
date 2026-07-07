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
_COORD_FACTOR = 1.3  # a heavy atom within this x covalent-sum of a metal is a coordinating donor
_FLEX_CONJ = 60.0  # deg: the wider conjugation-dihedral window a metal-coordinated / side-on π atom gets (vs 30)
_XH_TOL = 0.2  # Å slack over an X-H covalent-radius sum (P-H/Si-H/S-H run longer than the flat C-H ceiling)
_SIDEON_SYM = 0.5  # Å: max |d(M,a) - d(M,b)| for a pi pair to count as symmetric side-on (else donor + backbone)
_SIDEON_MAX = 2.6  # Å: both eta-2 atoms must bind within this — beyond it a pi atom is a backbone atom, not a donor


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
    """Each H bonded to exactly one heavy atom at a sane X-H length (bad-H-position detector).

    The upper limit is element-aware: a P-H (~1.42 A), Si-H (~1.48), or S-H (~1.34) bond is genuinely
    longer than a C/N/O-H, so the heavy atom's covalent radius sets its own ceiling (else a correct P-H
    false-flags). C/N/O-H keep the tight default `hi`.
    """
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
        hi_x = max(hi, _PT.GetRcovalent(nbrs[0].GetAtomicNum()) + _PT.GetRcovalent(1) + _XH_TOL)
        d = float(np.linalg.norm(pos[h] - pos[x]))
        if d < lo or d > hi_x:
            out.append(Violation("hydrogen", (x, h), d, hi_x if d > hi_x else lo, "X-H length"))
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

    Two donors of the same metal are a **1-3 pair through the metal** — cis coordination partners sit at the
    bite distance (~2 Å), which is not a clash. The metal-donor bonds are stripped on the surrogate, so
    these are detected by distance (`_coordination_pairs`) and excluded like any angle pair.
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
    excluded |= _coordination_pairs(mol, pos)  # cis donors of one metal are 1-3 through it, not a clash
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
    at a conjugating N/O) is handled by ``conjugation()`` via the O=C-N dihedral. A metal-coordinated carbon
    is passed in ``exclude`` by ``check()`` — the coordination sets its geometry, not an sp2 rule.
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


def conjugation(mol, pos, tol_deg: float = 30.0, exclude=frozenset(), flex=frozenset()) -> list[Violation]:
    """Amide / ester / enamine / enol-ether conjugation stays planar (broken-conjugation detector).

    For each single bond X-C where X is N/O and C bears a double bond (C=O or C=C), the substituent must
    be coplanar with the π system: the A=C-X-S dihedral near 0 or 180 deg. Catches a twisted amide (incl.
    an out-of-plane amide N-H, S=H) or enamine that local sp2-planarity misses. Rings and biaryls are left
    out — a lactam is locked by its ring and biaryls legitimately twist (atropisomers). A `flex` atom (a
    side-on η² π atom) gets the wider ``_FLEX_CONJ`` window.
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
            limit = _FLEX_CONJ if (c in flex or x in flex or a in flex) else tol_deg  # side-on η² twists further
            dih = abs(_dihedral(pos[a], pos[c], pos[x], pos[s]))
            dev = min(dih, abs(180.0 - dih))
            if dev > limit:
                out.append(Violation("conjugation", (a, c, x, s), dev, limit, "twisted out of plane"))
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
    # a metal-coordinated carbon (carbanion / carbene / side-on η² carbon) is NOT held to sp2 planarity — the
    # coordination sets its geometry (it can be sp3-pyramidal or bent; gfn2 keeps a coordinating [CH-] ~0.3 A
    # out of plane), so it's EXEMPT from the flat-sp2 check. A gross error still shows in bond/clash, and the
    # metal SQUARE PLANE is the separate coplanarity gate — this is only the atom's own local sp2.
    coord_c = _coordinating_carbons(mol, pos)
    v: list[Violation] = []
    v += bond_lengths(mol, pos, exclude=exclude)
    v += hydrogens(mol, pos, exclude=exclude)
    v += clashes(mol, pos, exclude=exclude)
    v += planarity(mol, pos, exclude=exclude | coord_c)
    v += conjugation(mol, pos, exclude=exclude, flex=_eta2_pi_atoms(mol, pos) | coord_c)
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


def _eta2_pi_atoms(mol, pos) -> set[int]:
    """Return atoms in a genuine side-on **η²** unit — a π-bonded pair binding one metal *symmetrically*.

    A side-on π ligand (alkene / alkyne / imine / phosphaalkene) binds *through* its π bond, so the metal
    sits above the bond and legitimately pulls the sp2 atoms a little out of plane. That out-of-plane is a
    real feature of the (angle/distance-restricted) side-on coordination, not a broken geometry, so these
    atoms get a wider planarity/conjugation window.

    The signal is a pi-bonded (double/triple) pair BOTH near the metal AND at **~equal** metal distance —
    side-on binding is symmetric. Equal-distance is what rejects a false positive the flat 1.3x donor shell
    lets through: an alpha-diimine's imine *carbon* drifts just inside the shell behind its sigma-donor N,
    but that N (~2.0 A) / C (~2.8 A) pair is lopsided, not side-on, so its backbone stays strictly planar.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z]
    out: set[int] = set()
    for a, b in _coordination_pairs(mol, pos):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondTypeAsDouble() < 2:  # noqa: PLR2004 — need double/triple for side-on pi
            continue
        for m in metals:  # symmetric binding: both atoms at ~equal distance from the SAME metal
            da, db = float(np.linalg.norm(pos[m] - pos[a])), float(np.linalg.norm(pos[m] - pos[b]))
            if abs(da - db) <= _SIDEON_SYM and max(da, db) <= _SIDEON_MAX:
                out.update((a, b))
                break
    return out


def _coordinating_carbons(mol, pos) -> set[int]:
    """Return carbons bound to a metal — a carbanion / carbene / eta-2 carbon is sp3-ish, not sp2-planar.

    A carbon that coordinates a metal (within the ``_COORD_FACTOR`` donor shell) legitimately pyramidalises
    out of the flat sp2 plane RDKit assigns it — gfn2 confirms a coordinating ``[CH-]`` amidate carbanion
    stays ~0.3 A out of plane — so it gets the wider planarity window, not the strict flat-sp2 one.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z]
    out: set[int] = set()
    for m in metals:
        rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == _CARBON_Z and (
                float(np.linalg.norm(pos[m] - pos[a.GetIdx()])) <= _COORD_FACTOR * (rm + _rcov(_CARBON_Z))
            ):
                out.add(a.GetIdx())
    return out


def _coordination_pairs(mol, pos) -> set[tuple[int, int]]:
    """Return donor-donor index pairs of each metal (a heavy atom within ``_COORD_FACTOR`` x covalent-sum).

    These are 1-3 pairs *through* the metal — cis coordination partners at the bite distance, not a steric
    clash. The metal-donor bonds are stripped on the rxembed surrogate, so donors are found geometrically.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z]
    out: set[tuple[int, int]] = set()
    for m in metals:
        rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
        donors = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1
            and a.GetIdx() != m
            and a.GetAtomicNum() not in _METAL_Z
            and float(np.linalg.norm(pos[m] - pos[a.GetIdx()])) <= _COORD_FACTOR * (rm + _rcov(a.GetAtomicNum()))
        ]
        for x in range(len(donors)):
            for y in range(x + 1, len(donors)):
                out.add((min(donors[x], donors[y]), max(donors[x], donors[y])))
    return out


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
