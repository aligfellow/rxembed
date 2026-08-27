"""Metal coordination-sphere gates -- has an atom collapsed onto the metal, and does a ligand still point right.

Read-only and off the embed path: these gates judge a finished geometry, and `pipeline.geom_check` consumes
them. They look *into* the coordination sphere, which every ``geometry`` check excludes because a dative
distance is not covalent: ``metal_overbond`` asks whether an atom reached bonding distance,
``donor_orientation`` whether a ligand still donates along its axis, and ``donor_fold`` reports the metric
behind the same walk. The rest is the sphere perception both resolve on; `pipeline.select` also reads its
`_coordinating` leaf when a wrapped metal has no bonds.

Enforcement (``_orient_donor`` / ``_coplanar_donor``) lives in ``metal_donor_orient``; both paths share its
donation-axis rules. Shared coordinate math and ``Violation`` live in ``utils``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from .metal_core import COORDINATION_METALS
from .metal_distance import (
    APEX,
    DONOR_COLLAPSE_RATIO,
    NEAR,
    NEAR_REPORT_RATIO,
    OUTER_REPORT_MARGIN,
    overbond_tier,
)
from .metal_donor_orient import (
    _FOLD_MEDIAN,
    _FOLD_WINDOW,
    _stripped_hybridisation,
    donation_axis,
)
from .utils import _CARBON_Z, Violation, _angle, _dihedral, _positions, _rcov

_COORD_FACTOR = 1.3  # a heavy atom within this x covalent-sum of a metal is a coordinating donor
_SIDEON_SYM = 0.5  # Å: max |d(M,a) - d(M,b)| for a pi pair to count as symmetric side-on (else donor + backbone)
_SIDEON_MAX = 2.6  # Å: both eta-2 atoms must bind within this; beyond it a pi atom is backbone, not a donor.
# Absolute, where `_COORD_FACTOR` above is a ratio of the covalent sum on the same axis, and the ratio form is
# measured and refuted over the 57 corpus geometries. WELROW's La...Se=P at 3.10/3.45 Å is only 1.10 x the
# covalent sum (La's radius alone is 2.07 Å), so any ratio loose enough to keep GODNOD's genuine Ni eta2-C=S
# (1.23 x) admits it too, and that is wrong: the P is the backbone behind the Se donor, the alpha-diimine
# false positive one shell out. COJKAO pins the inversion from the other side, its Pd...S=O rejected at
# 1.226 x against GODNOD's accepted 1.233 x, two verdicts 0.007 apart in ratio and 0.27 Å apart here


def metal_overbond(mol, pos, donors=None, margin: float | None = None) -> list[Violation]:
    """Atoms that have collapsed onto a metal; over-short is the silently-failing direction.

    Every other gate excludes metals, so an atom crushed into the sphere is otherwise invisible. Every
    non-metal atom is floored and the tier picks which floor: a non-donor is asked whether it reached bonding
    distance, a *donor* only whether it is closer than a bond can be (``DONOR_COLLAPSE_RATIO``, reported as
    ``metal_collapse``) -- the one ruler that can judge a monatomic hydride or halide.

    ``donors`` is the intended set and the caller must pass it: perceiving it is circular, since a collapsed
    atom enters the radius, re-classifies as a donor, and is then asked the donor's question instead of its
    own. ``donors=None`` falls back to perception only for a genuinely unknown sphere.

    Non-donors are floored by ``overbond_tier``: an APEX bite is forced and exempt, a second-sphere atom gets
    a ratio loose enough for a real β-agostic contact, and the rest the covalent sum + margin, below the FF's
    own floor so a floored relax never trips this. An *undeclared* H is skipped.
    """
    margin = OUTER_REPORT_MARGIN if margin is None else margin
    out: list[Violation] = []
    for m, sphere in _spheres(mol, pos, donors).items():
        rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i == m or a.GetAtomicNum() in COORDINATION_METALS:
                continue
            r_sum = rm + _rcov(a.GetAtomicNum())
            if i in sphere:  # a donor gets a floor too (see docstring): its licence is a bonding window, not a
                # half-line to zero, so only collapse is left to judge: the one ruler for a monatomic donor
                floor, what = DONOR_COLLAPSE_RATIO * r_sum, "donor"
            elif a.GetAtomicNum() == 1:
                continue  # an undeclared H: a beta-agostic contact reaches this distance and is not an over-bond
            else:
                tier = overbond_tier(mol, sphere, i)
                if tier == APEX:  # a chelate bite apex / eta-n backbone: not free to collapse, never flagged
                    continue
                floor, what = (NEAR_REPORT_RATIO * r_sum if tier == NEAR else r_sum + margin), "non-donor"
            d = float(np.linalg.norm(pos[i] - pos[m]))
            if d < floor:
                out.append(
                    Violation(
                        kind="metal_collapse" if what == "donor" else "metal_overbond",
                        atoms=(m, i),
                        value=d,
                        limit=floor,
                        detail=f"{what} {a.GetSymbol()}{i} is {d:.2f} A from the metal (floor {floor:.2f} A): it has "
                        + ("collapsed into it" if what == "donor" else "effectively bonded to it"),
                    )
                )
    return out


@dataclass(frozen=True)
class DonorAngle:
    """One judged M-D-X donation angle: the donor's class, the angle, and how far it folds."""

    metal: int
    donor: int
    sub: int  # X, a heavy substituent of the donor
    element: str = field(kw_only=True)  # the donor's element; thiolate at 103°, carboxylate at
    hyb: Chem.HybridizationType = field(kw_only=True)  # 126°: pooling by hyb alone voids the gate
    angle: float = field(kw_only=True)
    deviation: float = field(kw_only=True)  # |angle - class median|: the reference-free `fold`
    planarity: float | None = field(default=None, kw_only=True)  # metal out-of-plane in ligand plane (None
    # if not conjugated); report-only: crystals reach 87.6° and it can never gate

    @property
    def cls(self) -> tuple[str, Chem.HybridizationType]:
        """The census class this donation is judged against: ``(donor element, hybridisation)``."""
        return (self.element, self.hyb)


@dataclass
class FoldReport:
    """Result of ``donor_fold()``: the report-only donation metric, which never gates."""

    angles: list[DonorAngle] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)  # donors the ruler refused to judge: the two estimators
    # disagreed on the hybridisation, or the (element, hyb) class has n < 6 in the census (`_FOLD_WINDOW`). A
    # rising count is a free perception-bug detector.

    @property
    def fold(self) -> float:
        """Max |M-D-X - class median| over every judged donation (0.0 when nothing was judged).

        Needs no reference structure: it asks the geometry about itself, so unlike an RMSD-to-crystal metric
        it works in generation mode.
        """
        return max((a.deviation for a in self.angles), default=0.0)

    @property
    def planarity(self) -> float:
        """Max deg the metal lies out of a conjugated donor's ligand plane (0.0 if none). REPORT only."""
        return max((a.planarity for a in self.angles if a.planarity is not None), default=0.0)

    @property
    def outside_window(self) -> list[DonorAngle]:
        """Donations outside the census 99% window, including the overshoot the gate declines to judge.

        The gate fires on the fold floor only, a ceiling being a false positive on a healthy phosphine, so a
        real over-splay would otherwise be invisible rather than merely unjudged.
        """
        return [a for a in self.angles if not (_FOLD_WINDOW[a.cls][0] <= a.angle <= _FOLD_WINDOW[a.cls][1])]

    def summary(self) -> str:
        """Format the fold / planarity maxima and the unjudged donors."""
        return (
            f"fold {self.fold:.1f}° (max |M-D-X - class median|, n={len(self.angles)}), "
            f"planarity {self.planarity:.1f}° out of plane (report-only; real crystals reach 87.6°), "
            f"{len(self.outside_window)} outside the census window, {len(self.unknown)} donor(s) unclassified"
        )


def donor_fold(mol, conf_id: int = -1, *, donors=None, frozen=frozenset()) -> FoldReport:
    """Measure how far every ligand folds off its donation axis. Reports; never rejects.

    The same walk `donor_orientation` gates on, reported in full, so a merely unusual geometry stays visible
    where the gate would pass it. Read ``.fold``, ``.planarity``, ``.unknown``.
    """
    angles, unknown = _donor_walk(mol, _positions(mol, conf_id), donors, frozen)
    return FoldReport(angles, unknown)


def donor_orientation(mol, pos, donors=None, frozen=frozenset()) -> list[Violation]:
    """Ligands folded back over the metal: the one gate that asks where a ligand points, not where it is.

    ``metal_overbond`` is blind to the commonest fold, a short D-X bond being unable to reach the over-bond
    floor even swung fully side-on (a carbonyl bent to a right angle holds its O at 2.13 Å over a 2.04 Å
    floor). So this reads the M-D-X angle, which folding collapses. The floor per class is the 0.5th
    percentile of 721 crystal donations minus 5°, keyed on (element, hybridisation) because a thiolate donates
    at 103° where a carboxylate donates at 126°. It fires only on an angle no reference of that class
    realises; ``donor_fold`` is the metric that reports the rest.

    Five exemptions, each acute *by construction*:

    * a haptic donor bonded to a co-donor has no donation axis, sitting ~70° off it;
    * a hydride, sigma-complex or agostic H has no lone-pair axis;
    * a bridging donor takes its geometry from the bridge;
    * an APEX substituent bonded to >=2 donors sits at ~90° geometrically; a chelate's other arm is not exempt;
    * a D-X unit inside the frozen core takes its orientation from the reference TS.

    A donor the two estimators disagree on, or an uncalibrated class (n < 6), is never gated. ``donors`` is
    the intended sphere and the caller must pass it: perceiving it is circular, since a folded atom enters the
    radius, reads as a co-donor, and writes its own donor off as haptic.
    """
    out: list[Violation] = []
    for a in _donor_walk(mol, pos, donors, frozen)[0]:
        lo, hi = _FOLD_WINDOW[a.cls]
        if a.angle >= lo:  # floor only: gating the ceiling false-positives on a healthy phosphine at 153.9°
            continue  # (see `_FOLD_WINDOW`); the overshoot is reported by `FoldReport.outside_window`
        xsym = mol.GetAtomWithIdx(a.sub).GetSymbol()
        out.append(
            Violation(
                kind="donor_orientation",
                atoms=(a.metal, a.donor, a.sub),
                value=a.angle,
                limit=lo,
                detail=f"{a.element}{a.donor} ({str(a.hyb).lower()}) donates at {a.angle:.1f}° to {xsym}{a.sub} "
                f"(real {a.element} {str(a.hyb).lower()} donations: {lo:.0f}-{hi:.0f}°, median "
                f"{_FOLD_MEDIAN[a.cls]:.0f}°): the ligand has folded back over the metal",
            )
        )
    return out


def _eta2_pi_atoms(mol, pos) -> set[int]:
    """Return atoms in a genuine side-on η² unit: a π-bonded pair binding one metal symmetrically.

    A side-on π ligand binds *through* its π bond, so the metal sits above the bond and legitimately pulls the
    sp2 atoms a little out of plane, a real feature rather than broken geometry, so these atoms get a wider
    planarity/conjugation window.

    The signal is a pi-bonded pair both near the metal and at roughly equal metal distance: side-on is symmetric.
    Equal-distance rejects a false positive the flat 1.3x shell lets through: an alpha-diimine's imine C drifts
    inside the shell behind its sigma-donor N, but that N/C pair is lopsided, not side-on.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]
    out: set[int] = set()
    for a, b in _coordination_pairs(mol, pos):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondTypeAsDouble() < 2:  # noqa: PLR2004  double/triple needed for side-on pi
            continue
        for m in metals:  # symmetric binding: both atoms at ~equal distance from the same metal
            da, db = float(np.linalg.norm(pos[m] - pos[a])), float(np.linalg.norm(pos[m] - pos[b]))
            if abs(da - db) <= _SIDEON_SYM and max(da, db) <= _SIDEON_MAX:
                out.update((a, b))
                break
    return out


def _donor_walk(mol, pos, donors=None, frozen=frozenset()) -> tuple[list[DonorAngle], list[int]]:
    """Walk M -> D -> X: one judged angle per heavy donor substituent, plus the donors it refused to judge.

    The single source of truth for both the gate (``donor_orientation``) and the metric (``donor_fold``), so the
    two can never disagree about what was measured. See ``donor_orientation`` for the exemptions.
    """
    spheres = _spheres(mol, pos, donors)
    all_donors = {d for s in spheres.values() for d in s}  # every metal's donors: the co-donor exclusion set
    hyb = _stripped_hybridisation(mol)
    out: list[DonorAngle] = []
    unknown: list[int] = []
    for m, sphere in spheres.items():
        for d in sorted(sphere):
            subs = donation_axis(mol, d, all_donors, sphere=sphere, frozen=frozen)
            if subs is None:  # hydride, bridging or haptic: the donation question does not apply
                continue
            cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None  # None when estimators disagree
            if cls not in _FOLD_WINDOW:  # estimators disagreed, or n < 6 for this (element, hyb) class (an sp3
                if subs:  # ether O, an sp2 P): a threshold read off < 6 donations is noise, so abstain. Record
                    unknown.append(d)  # only if it had a judgeable substituent: an honest unknown, not a bare
                continue  # skip.
            for x in subs:
                ang = _angle(pos[m], pos[d], pos[x])
                dev = abs(ang - _FOLD_MEDIAN[cls])
                out.append(
                    DonorAngle(
                        m,
                        d,
                        x,
                        element=cls[0],
                        hyb=cls[1],
                        angle=ang,
                        deviation=dev,
                        planarity=_planarity_dev(mol, pos, hyb=hyb, m=m, d=d, x=x),
                    )
                )
    return out, unknown


def _planarity_dev(mol, pos, *, hyb, m, d, x) -> float | None:
    """Deg the metal lies out of a CONJUGATED donor's ligand plane (None when the D-X bond is not conjugated).

    A carboxylate/pyridine plane rotated into the coordination sphere with an acceptable M-D-X angle. Report-only
    It can never gate: the census p95 is 54.8° and real crystals reach 87.6° out of plane.
    """
    bond = mol.GetBondBetweenAtoms(d, x)
    sp2 = Chem.HybridizationType.SP2
    if bond is None or not bond.GetIsConjugated() or hyb.get(d) != sp2 or hyb.get(x) != sp2:
        return None
    best = None
    for y in mol.GetAtomWithIdx(x).GetNeighbors():
        j = y.GetIdx()
        if j in (d, m) or y.GetAtomicNum() == 1 or y.GetAtomicNum() in COORDINATION_METALS:
            continue
        t = abs(_dihedral(pos[m], pos[d], pos[x], pos[j]))
        dev = min(t, abs(180.0 - t))  # 0 = metal lies in the ligand plane
        best = dev if best is None else min(best, dev)
    return best


def _coordinating_carbons(mol, pos) -> set[int]:
    """Return carbons in a metal's coordination shell, which get the wider planarity window.

    A carbanion / carbene / eta2 carbon legitimately pyramidalises out of the flat sp2 plane RDKit assigns it,
    so judging it by the strict sp2 rule reports a defect that is not one.
    """
    metals = (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
    return {i for m in metals for i in _coordinating(mol, pos, m) if mol.GetAtomWithIdx(i).GetAtomicNum() == _CARBON_Z}


def _coordinating(mol, pos, m, declared=frozenset()) -> set[int]:
    """Non-metal atoms sitting inside metal ``m``'s coordination shell (``_COORD_FACTOR`` x covalent-sum).

    A heavy atom at coordination distance is unambiguously a donor. A hydrogen is not, since a hydride and an
    agostic C-H sit at the same M-H distance, so an H counts only when in ``declared``: the sole way an H
    enters a sphere; the element screen never overrides a declaration.
    """
    rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
    out = set()
    for a in mol.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        if i == m or z in COORDINATION_METALS or (z == 1 and i not in declared):
            continue
        if float(np.linalg.norm(pos[m] - pos[i])) <= _COORD_FACTOR * (rm + _rcov(z)):
            out.add(i)
    return out


def _spheres(mol, pos, donors=None) -> dict[int, set[int]]:
    """``{metal: its coordination sphere}``: intended donors where known, perceived where not.

    Resolved per metal, because in a bimetallic complex the caller usually knows only the reacting centre's
    sphere and the other must fall back to perception rather than flag its own ligands. A known donor that has
    left the shell is dropped; `coordination_changed` is what gates that.
    """
    known = set() if donors is None else {int(d) for d in donors}
    out: dict[int, set[int]] = {}
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS):
        shell = _coordinating(mol, pos, m, known)  # `known` also admits a declared hydride the element screen
        out[m] = (known & shell) or shell  # would drop; else `known & shell` silently loses it
    return out


def _coordination_pairs(mol, pos) -> set[tuple[int, int]]:
    """Return donor-donor index pairs of each metal (a heavy atom within ``_COORD_FACTOR`` x covalent-sum).

    These are 1-3 pairs through the metal: cis coordination partners at the bite distance, not a steric
    clash. The metal-donor bonds are stripped on the rxembed surrogate, so donors are found geometrically.
    """
    out: set[tuple[int, int]] = set()
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS):
        donors = sorted(_coordinating(mol, pos, m))
        for x in range(len(donors)):
            for y in range(x + 1, len(donors)):
                out.add((donors[x], donors[y]))
    return out
