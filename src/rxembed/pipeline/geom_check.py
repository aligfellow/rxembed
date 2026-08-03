"""Geometry gate -- validate that an embedded conformer is chemically sound.

A perfect frozen core and perfect constraint distances can still coexist with a puckered aromatic ring, a
twisted amide, an out-of-plane N-H, a stretched X-H, or two atoms on top of each other: the free periphery
that distance-window checks miss.

``check()`` runs the physical checks and returns a ``GeometryReport``: truthy when clean, printable, with
``.assert_ok()`` for tests. Pure ``rdkit`` + ``numpy``.

    rep = geom_check.check(mol, conf_id, frozen=core, reference=ts_mol,
                           constraints={"distances": {(1, 9): (2.6, 3.0)}})
    rep.assert_ok()          # raises with a readable summary if any check fails

The two coordination-sphere gates it calls live in the core's ``rxembed.metal_perceive``; the
``GeometryReport`` that collects them is this module's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

# metals excluded from every ground-state check (dative, not vdW):
from rxembed.metal_core import _METAL_Z
from rxembed.metal_perceive import (
    _coordinating_carbons,
    _coordination_pairs,
    _eta2_pi_atoms,
    donor_orientation,
    metal_overbond,
)
from rxembed.utils import (
    _CARBON_Z,
    _PT,
    _SP2_DEGREE,
    Violation,
    _angle,
    _dihedral,
    _positions,
    _rcov,
    assign_stereo_from_3d,
    conjugated_quartets,
)

_FLEX_CONJ = 60.0  # deg: the wider conjugation-dihedral window a metal-coordinated / side-on π atom gets (vs 30)
_XH_TOL = 0.2  # Å slack over an X-H covalent-radius sum (P-H/Si-H/S-H run longer than the flat C-H ceiling)
_FUSE_RATIO = (
    1.0  # a NON-bonded 1-3 pair has fused when its separation drops to the covalent sum: a ring closed by the relax
)
# rather than by the graph. Bonded 1-3 pairs are excluded, being legitimately this close.


# --- the result -------------------------------------------------------------


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


# --- the checks --------------------------------------------------------------


def bond_lengths(mol, pos, lo: float = 0.7, hi: float = 1.3, exclude=frozenset()) -> list[Violation]:
    """Every bond within ``[lo, hi]`` x the sum of covalent radii (no stretched/compressed bonds).

    ``exclude`` atoms (a frozen or reacting TS core) are skipped: their partial bonds are held to the
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


def hydrogens(mol, pos, lo: float = 0.8, hi: float = 1.3, exclude=frozenset(), donors=frozenset()) -> list[Violation]:
    """Each H bonded to exactly one heavy atom at a sane X-H length (bad-H-position detector).

    The upper limit is element-aware: a P-H/Si-H/S-H bond is genuinely longer than a C/N/O-H, so the heavy
    atom's covalent radius sets its own ceiling (else a correct P-H false-flags); C/N/O-H keep the tight `hi`.

    A metal-held H is not judged here, a dative M-H not being a covalent X-H. It is metal-held if it is a
    ``donors`` declaration (a terminal hydride is left bond-less by the strip) or its bonds reach a metal; its
    M-H distance is judged by `metal_overbond`'s dative floor.
    """
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = atom.GetNeighbors()
        h = atom.GetIdx()
        # `any`, not `nbrs[0]`: an eta2-H2 donor's neighbours are the metal *and* its H partner, in whatever
        # order RDKit stored them; order must not decide whether this H is judged.
        if h in exclude or h in donors or any(nb.GetIdx() in exclude for nb in nbrs):
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

    Thresholds differ by kind so real non-covalent contacts (H-bonds, halogen/π) are not flagged: heavy-heavy
    at `heavy_vdw` x sum-of-vdW; H...H below `hh_floor` Å; an H buried in a heavy atom at `xh_cov` x
    sum-of-covalent (a 1.8 Å H-bond passes, a buried H caught). ``exclude`` atoms (a frozen/reacting core) are
    skipped.

    Two donors of the same metal are a 1-3 pair through the metal (cis partners at the bite distance ~2 Å,
    not a clash); the surrogate strips the M-donor bonds, so these are found by distance and excluded.
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


def over_compression(mol, pos, exclude=frozenset(), ratio: float = _FUSE_RATIO) -> list[Violation]:
    """Non-bonded 1-3 pairs crushed to bonding distance: a phantom ring no other gate catches.

    A 1-3 pair is normally held apart by its bridging angle, so ``clashes`` excludes it and
    ``metrics.connectivity`` (``_MIN_TOPO``) never reports it. But a hard relax CAN fold that angle until the
    terminals fuse: a coordinated ester's O-C-O collapsing ~122->56° fuses its O to 1.27 Å and a perceiver reads
    a 3-ring the molecule never had (below ``bonding_ok``'s ~0.9 Å fusion floor).

    The discriminator against a genuine small ring (epoxide, cyclopropane) is the graph, not the angle: there the
    terminals are BONDED so the pair never enters this test. Only a not-bonded pair below ``ratio`` x covalent sum
    is a fusion. Frozen-core / metal atoms come in via ``exclude``.
    """
    out: list[Violation] = []
    for mid_atom in mol.GetAtoms():
        mid = mid_atom.GetIdx()
        if mid in exclude or mid_atom.GetAtomicNum() == 1:  # a frozen/metal bridge, or an H (no bridging angle to
            continue  # an H bonded to two heavies is only a broken or 3-centre graph, not a fusion
        nbrs = [n.GetIdx() for n in mid_atom.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in exclude]
        for u in range(len(nbrs)):
            for w in range(u + 1, len(nbrs)):
                a, c = nbrs[u], nbrs[w]
                if mol.GetBondBetweenAtoms(a, c) is not None:  # a genuine small ring: the two terminals ARE bonded
                    continue
                r_sum = _rcov(mol.GetAtomWithIdx(a).GetAtomicNum()) + _rcov(mol.GetAtomWithIdx(c).GetAtomicNum())
                floor = ratio * r_sum
                d = float(np.linalg.norm(pos[a] - pos[c]))
                if d < floor:
                    ang = _angle(pos[a], pos[mid], pos[c])
                    a_s = f"{mol.GetAtomWithIdx(a).GetSymbol()}{a}"
                    c_s = f"{mol.GetAtomWithIdx(c).GetSymbol()}{c}"
                    m_s = f"{mol.GetAtomWithIdx(mid).GetSymbol()}{mid}"
                    out.append(
                        Violation(
                            "fusion",
                            (a, mid, c),
                            d,
                            floor,
                            f"{a_s}...{c_s} fused to {d:.2f} A across {m_s} (angle {ang:.0f}°): "
                            "a bond the graph does not have",
                        )
                    )
    return out


def planarity(mol, pos, oop: float = 0.15, ring_rms: float = 0.10, exclude=frozenset()) -> list[Violation]:
    """sp2 carbons stay planar and aromatic rings stay flat (broken-conjugation detector).

    Catches puckered aromatic rings and twisted sp2 carbons (C=C / C=O / aromatic C). Nitrogen is left
    out on purpose: an amine / aniline N is *physically* pyramidal even when RDKit labels it sp2, so a
    blanket N-planarity rule false-positives. The twisted-amide / broken-conjugation case (the true signal
    at a conjugating N/O) is handled by ``conjugation()`` via the O=C-N dihedral. A metal-coordinated carbon
    is passed in ``exclude`` by ``check()``: the coordination sets its geometry, not an sp2 rule.
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
    out: a lactam is locked by its ring and biaryls legitimately twist (atropisomers). A `flex` atom (a
    side-on η² π atom) gets the wider ``_FLEX_CONJ`` window.
    """
    out = []
    for a, c, x, s in conjugated_quartets(mol, exclude):
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
    n = len(pos)  # a constraint on a transient (a haptic centroid dummy) has no counterpart in the real geometry:
    out = []  # its index is >= n, so skip it. The equivalent real check (M -> each ring atom) has real indices.
    for (i, j), win in dists.items():
        if i >= n or j >= n:
            continue
        lo, hi = win if isinstance(win, tuple) else (win, win)
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if not (lo - dist_slack <= d <= hi + dist_slack):
            out.append(Violation("constraint", (i, j), d, hi + dist_slack, f"distance window [{lo}, {hi}]"))
    for (i, j, k), (lo, hi) in angles.items():
        if i >= n or j >= n or k >= n:
            continue
        a = _angle(pos[i], pos[j], pos[k])
        if not (lo - ang_slack <= a <= hi + ang_slack):
            out.append(Violation("constraint", (i, j, k), a, hi + ang_slack, f"angle window [{lo}, {hi}]"))
    return out


def stereo_violations(mol, pos, reference, conf_id: int = -1) -> list[Violation]:
    """No stereocentre inverted vs ``reference`` (CIP tags from 3D)."""
    got = _cip(mol, conf_id)
    want = _cip(reference)
    out = []
    for idx, tag in want.items():
        if got.get(idx) != tag:
            out.append(Violation("stereo", (idx,), 0.0, 0.0, f"{tag} -> {got.get(idx)}"))
    return out


# --- top-level entry ---------------------------------------------------------


def check(mol, conf_id: int = -1, *, frozen=None, reference=None, constraints=None, donors=None) -> GeometryReport:
    """Run the full physical gate on one conformer and collect all violations.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with at least one conformer.
    conf_id : int
        Conformer id to check (default: the last / active one).
    frozen : sequence of int, optional
        The frozen or reacting TS core. Held to ``reference`` by an RMSD check and excluded from the
        ground-state checks, since a forming or breaking bond and a distorted reacting sp2 are correct by
        design; only the free periphery is vetted for clashes, conjugation and H-placement.
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
    # A metal-coordinated carbon (carbanion, carbene, side-on η²) is not held to organic valence: the dative
    # bond is not covalent, so counting it would flag every real one.
    coord_c = _coordinating_carbons(mol, pos)
    v: list[Violation] = []
    v += bond_lengths(mol, pos, exclude=exclude)
    v += hydrogens(mol, pos, exclude=exclude, donors=frozenset(donors or ()))
    v += clashes(mol, pos, exclude=exclude)
    v += over_compression(mol, pos, exclude=exclude)  # a 1-3 pair fused across a collapsed bridging angle:
    #     phantom ring `clashes` (1-3 excluded) and `metrics.connectivity` (`_MIN_TOPO`) are both blind to
    v += planarity(mol, pos, exclude=exclude | coord_c)
    v += conjugation(mol, pos, exclude=exclude, flex=_eta2_pi_atoms(mol, pos) | coord_c)
    v += metal_overbond(mol, pos, donors)  # the two gates that look INTO the sphere (every other check excludes
    v += donor_orientation(mol, pos, donors, frozen or frozenset())  # metals): has an atom reached bonding distance,
    #     and does a ligand still point the right way (a short D-X can go side-on under the floor). Pass `donors`.
    if frozen is not None and reference is not None:
        v += frozen_core(mol, pos, frozen, reference)
    if constraints is not None:
        v += check_constraints(mol, pos, constraints)
    if reference is not None:
        v += stereo_violations(mol, pos, reference, conf_id)
    return GeometryReport(v)


# --- small private utilities -------------------------------------------------

_EPS = 1e-9  # numerical floor for a degenerate cross-product / near-zero norm


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


def _attr(obj, name, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cip(mol, conf_id: int = -1) -> dict[int, str]:
    m = Chem.Mol(mol)
    # through `utils`: the CIP labeller counts a dative bond leaving the centre and RDKit's 3D writer does
    # not, so reading a label off a raw write mirrors the R/S at every dative-bonded donor.
    assign_stereo_from_3d(m, conf_id if conf_id >= 0 else m.GetNumConformers() - 1)
    out = {}
    for atom in m.GetAtoms():
        if atom.HasProp("_CIPCode"):
            out[atom.GetIdx()] = atom.GetProp("_CIPCode")
    return out
