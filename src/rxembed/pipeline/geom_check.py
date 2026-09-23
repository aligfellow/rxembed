"""Geometry gate: validate that an embedded conformer is chemically sound.

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
from rdkit.Numerics import rdAlignment

# metals excluded from every ground-state check (dative, not vdW):
from rxembed.constraints import constraint_value, within_window
from rxembed.metal_core import COORDINATION_METALS, metal_indices
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
        if b.GetBondType() == Chem.BondType.ZERO:
            continue  # explicit non-covalent contact, not a bond-length contract
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in exclude or j in exclude:
            continue
        d = float(np.linalg.norm(pos[i] - pos[j]))
        ideal = _rcov(mol.GetAtomWithIdx(i).GetAtomicNum()) + _rcov(mol.GetAtomWithIdx(j).GetAtomicNum())
        if d < lo * ideal:
            out.append(
                Violation(
                    kind="bond_length",
                    atoms=(i, j),
                    value=d,
                    limit=lo * ideal,
                    detail="compressed",
                )
            )
        elif d > hi * ideal:
            out.append(
                Violation(
                    kind="bond_length",
                    atoms=(i, j),
                    value=d,
                    limit=hi * ideal,
                    detail="stretched",
                )
            )
    return out


def hydrogens(
    mol, pos, *, lo: float = 0.8, hi: float = 1.3, exclude=frozenset(), donors=frozenset()
) -> list[Violation]:
    """Each H bonded to exactly one heavy atom at a sane X-H length (bad-H-position detector).

    The upper limit is element-aware: a P-H/Si-H/S-H bond is genuinely longer than a C/N/O-H, so the heavy
    atom's covalent radius sets its own ceiling instead of the tight `hi`. A metal-held H is not judged here
    (a dative M-H is not a covalent X-H; `metal_overbond` owns its distance), where metal-held means a
    ``donors`` declaration (a terminal hydride left bond-less by the strip) or a bond reaching a metal.
    """
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h = atom.GetIdx()
        nbrs = [
            nb
            for nb in atom.GetNeighbors()
            if mol.GetBondBetweenAtoms(h, nb.GetIdx()).GetBondType() != Chem.BondType.ZERO
        ]
        if not nbrs and any(
            mol.GetBondBetweenAtoms(h, nb.GetIdx()).GetBondType() == Chem.BondType.ZERO for nb in atom.GetNeighbors()
        ):
            continue  # an explicitly disconnected/bridging H has no covalent X-H contract
        # `any`, not `nbrs[0]`: an eta2-H2 donor's neighbours are the metal and its H partner in RDKit's order.
        if h in exclude or h in donors or any(nb.GetIdx() in exclude for nb in nbrs):
            continue
        if len(nbrs) != 1:
            out.append(
                Violation(
                    kind="hydrogen",
                    atoms=(h,),
                    value=float(len(nbrs)),
                    limit=1.0,
                    detail="H not bonded to exactly one atom",
                )
            )
            continue
        x = nbrs[0].GetIdx()
        hi_x = max(hi, _PT.GetRcovalent(nbrs[0].GetAtomicNum()) + _PT.GetRcovalent(1) + _XH_TOL)
        d = float(np.linalg.norm(pos[h] - pos[x]))
        if d < lo or d > hi_x:
            out.append(
                Violation(
                    kind="hydrogen",
                    atoms=(x, h),
                    value=d,
                    limit=hi_x if d > hi_x else lo,
                    detail="X-H length",
                )
            )
    return out


def clashes(
    mol, pos, *, heavy_vdw: float = 0.7, hh_floor: float = 1.5, xh_cov: float = 1.1, exclude=frozenset()
) -> list[Violation]:
    """No non-bonded pair (excluding 1-3 angle pairs) in steric overlap.

    Thresholds differ by kind so real non-covalent contacts (H-bonds, halogen/π) are not flagged: heavy-heavy
    at `heavy_vdw` x sum-of-vdW; H...H below `hh_floor` Å; an H buried in a heavy atom at `xh_cov` x
    sum-of-covalent (a 1.8 Å H-bond passes, a buried H caught). ``exclude`` atoms (a frozen/reacting core) are
    skipped.
    """
    excluded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        excluded.add((min(i, j), max(i, j)))
    for atom in mol.GetAtoms():  # 1-3 pairs (share a neighbour) are governed by angles, not clashes
        nb = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(nb)):
            for c in range(a + 1, len(nb)):
                if (
                    atom.GetAtomicNum() not in COORDINATION_METALS
                    and atom.GetIdx() not in exclude
                    and mol.GetAtomWithIdx(nb[a]).GetAtomicNum() == 1
                    and mol.GetAtomWithIdx(nb[c]).GetAtomicNum() == 1
                ):
                    continue  # test H-X-H collapse only where X is not an excluded reacting/coordination centre
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
                limit, why = heavy_vdw * (_PT.GetRvdw(z[i]) + _PT.GetRvdw(z[j])), "heavy-atom steric overlap"
            if d < limit:
                out.append(Violation(kind="clash", atoms=(i, j), value=d, limit=limit, detail=why))
    return out


def over_compression(mol, pos, exclude=frozenset(), ratio: float = _FUSE_RATIO) -> list[Violation]:
    """Non-bonded 1-3 pairs crushed to bonding distance: a phantom ring no other gate catches.

    A 1-3 pair is normally held apart by its bridging angle, so ``clashes`` excludes it and
    ``metrics.connectivity`` never reports it. But a hard relax can fold that angle until the terminals
    fuse: a coordinated ester's O-C-O collapsing ~122->56° fuses its O to 1.27 Å, below ``bonding_ok``'s
    ~0.9 Å fusion floor. The discriminator against a genuine small ring (epoxide, cyclopropane) is the
    graph, not the angle: a bonded pair never enters this test.
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
                            kind="fusion",
                            atoms=(a, mid, c),
                            value=d,
                            limit=floor,
                            detail=f"{a_s}...{c_s} fused to {d:.2f} A across {m_s} (angle {ang:.0f}°): "
                            "a bond the graph does not have",
                        )
                    )
    return out


def planarity(mol, pos, oop: float = 0.15, ring_rms: float = 0.10, exclude=frozenset()) -> list[Violation]:
    """sp2 carbons stay planar and aromatic rings stay flat (broken-conjugation detector).

    Catches puckered aromatic rings and twisted sp2 carbons (C=C / C=O / aromatic C). Nitrogen is left out
    on purpose: an amine/aniline N is physically pyramidal even when RDKit labels it sp2, so a blanket rule
    false-positives. The real twisted-amide signal is caught by ``conjugation()``'s O=C-N dihedral instead.
    A metal-coordinated carbon is passed in ``exclude`` by ``check()``: coordination sets its geometry, not
    an sp2 rule.
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
            out.append(
                Violation(
                    kind="planarity",
                    atoms=(atom.GetIdx(), *nbrs),
                    value=off,
                    limit=oop,
                    detail="sp2 out of plane",
                )
            )
    # SymmSSSR rewrites RingInfo, so keep this read-only gate off the caller's graph. Composite fused-system
    # perimeters reuse several edges whose smaller local rings already carry the planarity requirement.
    graph = Chem.Mol(mol)
    Chem.GetSymmSSSR(graph)
    rings = graph.GetRingInfo()
    for ring, bonds in zip(rings.AtomRings(), rings.BondRings(), strict=True):
        if any(i in exclude for i in ring) or not all(graph.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        smaller_edges = sum(rings.MinBondRingSize(bond) < len(ring) for bond in bonds)
        if smaller_edges > 1:
            continue
        p = pos[list(ring)]
        rms = float(np.sqrt((np.linalg.svd(p - p.mean(0))[1][2] ** 2) / len(ring)))
        if rms > ring_rms:
            out.append(
                Violation(
                    kind="planarity",
                    atoms=tuple(ring),
                    value=rms,
                    limit=ring_rms,
                    detail="aromatic ring puckered",
                )
            )
    return out


def conjugation(mol, pos, tol_deg: float = 30.0, exclude=frozenset(), flex=frozenset()) -> list[Violation]:
    """Amide / ester / enamine / enol-ether conjugation stays planar (broken-conjugation detector).

    For each single bond X-C where X is N/O and C bears a double bond (C=O or C=C), the substituent must be
    coplanar with the π system: the A=C-X-S dihedral near 0 or 180 deg. Catches a twisted amide or enamine
    that local sp2-planarity misses. Rings and biaryls are left out (a lactam is locked by its ring; a
    biaryl legitimately twists as an atropisomer); a `flex` atom (side-on η² π) gets a wider window.
    """
    out = []
    for a, c, x, s in conjugated_quartets(mol, exclude):
        limit = _FLEX_CONJ if (c in flex or x in flex or a in flex) else tol_deg  # side-on η² twists further
        dih = abs(_dihedral(pos[a], pos[c], pos[x], pos[s]))
        dev = min(dih, abs(180.0 - dih))
        if dev > limit:
            out.append(
                Violation(
                    kind="conjugation",
                    atoms=(a, c, x, s),
                    value=dev,
                    limit=limit,
                    detail="twisted out of plane",
                )
            )
    return out


def frozen_core(mol, pos, frozen, reference, tol: float = 0.05) -> list[Violation]:
    """Frozen atoms held to ``reference`` within ``tol`` Å after Kabsch superposition."""
    frozen = list(frozen)
    ref = _positions(reference)
    ssd, _ = rdAlignment.GetAlignmentTransform(ref[frozen], pos[frozen])
    rmsd = (ssd / len(frozen)) ** 0.5
    if rmsd > tol:
        return [Violation(kind="frozen_core", atoms=tuple(frozen), value=rmsd, limit=tol, detail="core moved")]
    return []


def check_constraints(mol, pos, spec, dist_slack: float = 0.15, ang_slack: float = 5.0) -> list[Violation]:
    """Return distance, angle and dihedral terms missed by one geometry.

    ``spec`` is a mapping or Constraints-like object exposing geometric windows.
    """
    haptic = _attr(spec, "haptic", {})
    phantoms = set(_attr(spec, "phantoms", ()))
    out = []
    for name, windows, slack in (
        ("distance", _attr(spec, "distances", {}), dist_slack),
        ("angle", _attr(spec, "angles", {}), ang_slack),
        ("dihedral", _attr(spec, "dihedrals", {}), ang_slack),
    ):
        for atoms, given in windows.items():
            if phantoms.intersection(atoms):
                continue  # private DG/UFF scaffold, not a geometric term on the stored molecule
            window = tuple(given) if isinstance(given, (tuple, list)) else (given, given)
            value = constraint_value(pos, atoms, haptic, window)
            if not within_window(value, window, slack):
                lo, hi = window
                out.append(
                    Violation(
                        kind="constraint",
                        atoms=atoms,
                        value=float("nan") if value is None else value,
                        limit=hi + slack,
                        detail=(
                            f"{name} could not be measured; check atom indices and haptic metadata"
                            if value is None or not np.isfinite(value)
                            else f"{name} window [{lo}, {hi}]"
                        ),
                    )
                )
    return out


def stereo_violations(mol, pos, reference, conf_id: int = -1) -> list[Violation]:
    """No stereocentre inverted vs ``reference`` (CIP tags from 3D)."""
    got = _cip(mol, conf_id)
    want = _cip(reference)
    out = []
    for idx, tag in want.items():
        if got.get(idx) != tag:
            out.append(
                Violation(
                    kind="stereo",
                    atoms=(idx,),
                    value=0.0,
                    limit=0.0,
                    detail=f"{tag} -> {got.get(idx)}",
                )
            )
    return out


# --- top-level entry ---------------------------------------------------------


def check(mol, conf_id: int = -1, *, frozen=None, reference=None, constraints=None, donors=None) -> GeometryReport:
    """Run the full physical gate on one conformer and collect all violations.

    Parameters
    ----------
    frozen : sequence of int, optional
        The frozen or reacting TS core: excluded from the ground-state checks (a forming/breaking bond
        or distorted reacting sp2 is correct by design) and held to ``reference`` by an RMSD check.
    reference : rdkit.Chem.Mol or .xyz path, optional
        Reference geometry for the frozen-core RMSD and stereo checks.
    constraints : mapping or Constraints-like, optional
        ``distances`` / ``angles`` to confirm the seed was realised.
    """
    if isinstance(reference, str):
        reference = Chem.MolFromXYZFile(reference)  # coords only; atom order must match `mol`
    pos = _positions(mol, conf_id)
    if not np.all(np.isfinite(pos)):
        return GeometryReport(
            [
                Violation(
                    kind="coordinates",
                    atoms=(),
                    value=float("nan"),
                    limit=0.0,
                    detail="non-finite coordinate",
                )
            ]
        )
    exclude = set(frozen) if frozen is not None else set()
    exclude |= set(metal_indices(mol))  # dative, not vdW
    exclude = frozenset(exclude)
    coord_c = _coordinating_carbons(mol, pos)  # dative, not covalent: an organic valence rule would false-flag it
    v: list[Violation] = []
    v += bond_lengths(mol, pos, exclude=exclude)
    v += hydrogens(mol, pos, exclude=exclude, donors=frozenset(donors or ()))
    v += clashes(mol, pos, exclude=exclude)
    v += over_compression(mol, pos, exclude=exclude)
    v += planarity(mol, pos, exclude=exclude | coord_c)
    v += conjugation(mol, pos, exclude=exclude, flex=_eta2_pi_atoms(mol, pos) | coord_c)
    v += metal_overbond(mol, pos, donors)  # the two gates that look INTO the sphere, unlike every check above
    v += donor_orientation(mol, pos, donors, frozen or frozenset())
    if frozen is not None and reference is not None:
        v += frozen_core(mol, pos, frozen, reference)
    if constraints is not None:
        v += check_constraints(mol, pos, constraints)
    if reference is not None:
        v += stereo_violations(mol, pos, reference, conf_id)
    return GeometryReport(v)


# --- small private utilities -------------------------------------------------

_EPS = 1e-9  # numerical floor for a degenerate cross-product / near-zero norm


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
