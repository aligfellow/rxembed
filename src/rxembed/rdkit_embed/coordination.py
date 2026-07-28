"""Metal coordination-sphere gates -- has an atom collapsed onto the metal, and does a ligand still point right.

The two gates look *into* the coordination sphere (which every ``geometry`` check excludes, a dative distance
not being covalent): ``metal_overbond`` asks whether an atom reached bonding distance, ``donor_orientation``
whether a ligand still donates along its axis or has folded back. ``donor_fold`` is the report-only metric
behind the same walk. The rest is the sphere perception both gates resolve on (``_spheres`` / ``_coordinating``,
``_coordination_pairs``, ``_eta2_pi_atoms``, coordinating carbons).

This is the *gate* side; the *enforcement* side (``_orient_donor`` / ``_coplanar_donor`` + the census) lives in
``constraints.donor_orient``, imported below so the two cannot drift.

The module also holds the two pieces of *organic* conjugation perception the QA gate and the FF caps share --
``conjugated_quartets`` (for ``conjugation`` / ``ConjugationCap``) and ``_SP2_DEGREE`` (for ``planarity`` /
``Sp2Planar``) -- one definition each, so enforcement and gate cannot name different atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from rxembed.rdkit_embed.constraints.distance import (
    APEX,
    DONOR_COLLAPSE_RATIO,
    NEAR,
    NEAR_REPORT_RATIO,
    OUTER_REPORT_MARGIN,
    overbond_tier,
)
from rxembed.rdkit_embed.constraints.donor_orient import (
    _FOLD_MEDIAN,
    _FOLD_WINDOW,
    _stripped_hybridisation,
    donation_axis,
)
from rxembed.rdkit_embed.constraints.metal import _METAL_Z
from rxembed.rdkit_embed.report import Violation
from rxembed.rdkit_embed.vecmath import _CARBON_Z, _angle, _dihedral, _positions, _rcov

_COORD_FACTOR = 1.3  # a heavy atom within this x covalent-sum of a metal is a coordinating donor
_SIDEON_SYM = 0.5  # Å: max |d(M,a) - d(M,b)| for a pi pair to count as symmetric side-on (else donor + backbone)
_SIDEON_MAX = 2.6  # Å: both eta-2 atoms must bind within this — beyond it a pi atom is a backbone atom, not a donor
_PLANARITY_P95 = 54.8  # deg: metal out of a conjugated donor's plane; report-only (crystals reach 87.6, no gate holds)
_SP2_DEGREE = 3  # a planar sp2 centre has exactly three neighbours


def metal_overbond(mol, pos, donors=None, margin: float | None = None) -> list[Violation]:
    """Atoms that have collapsed onto a metal — over-short is the silently-failing direction.

    Every other gate excludes metals, so an atom crushed into the sphere is otherwise invisible.

    **Every** non-metal atom is floored; the tier picks which floor. A non-donor is asked "have you reached
    bonding distance"; a *donor*, licensed to be there, only "are you closer than a bond can be"
    (``DONOR_COLLAPSE_RATIO``, reported as ``metal_collapse``) — the one ruler that can judge a monatomic donor
    (a bond-less hydride/halide every covalent gate skips).

    ``donors`` is the intended set; the caller must pass it — perceiving it is circular (a collapsed atom enters
    the radius, is re-classified a donor, and is then asked the donor's question, not its own). ``donors=None``
    falls back to perception only for a genuinely unknown sphere.

    Non-donors are floored by ``overbond_tier``: an APEX bite (>= 2 donors) is forced and exempt; a second-sphere
    atom (1 donor) by a ratio loose enough for a real beta-agostic / CMD contact; else covalent sum + margin
    (below the FF's own floor, so a floored relax never trips this). An *undeclared* H is skipped (an agostic
    contact reaches the same M-H distance as a bond).
    """
    margin = OUTER_REPORT_MARGIN if margin is None else margin
    out: list[Violation] = []
    for m, sphere in _spheres(mol, pos, donors).items():
        rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i == m or a.GetAtomicNum() in _METAL_Z:
                continue
            r_sum = rm + _rcov(a.GetAtomicNum())
            if i in sphere:  # a donor gets a floor too (see docstring): its licence is a bonding window, not a
                # half-line to zero, so only collapse is left to judge — the one ruler for a monatomic donor
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
                        "metal_collapse" if what == "donor" else "metal_overbond",
                        (m, i),
                        d,
                        floor,
                        f"{what} {a.GetSymbol()}{i} is {d:.2f} A from the metal (floor {floor:.2f} A) — it has "
                        + ("collapsed into it" if what == "donor" else "effectively bonded to it"),
                    )
                )
    return out


@dataclass(frozen=True)
class DonorAngle:
    """One judged M-D-X donation angle: the donor's class, the angle, and how far it folds."""

    metal: int
    donor: int
    sub: int  # X — a heavy substituent of the donor
    element: str  # the donor's element — half the class key: a thiolate donates at 103 deg, a carboxylate at
    hyb: Chem.HybridizationType  # 126 deg, so pooling by hybridisation alone voids the gate
    angle: float
    deviation: float  # |angle - class census median| — the reference-free `fold`
    planarity: float | None  # deg the metal sits out of a conjugated donor's ligand plane (None if not
    # conjugated); report-only — real crystals reach 87.6 deg out of plane, so it can never gate (`_PLANARITY_P95`)

    @property
    def cls(self) -> tuple[str, Chem.HybridizationType]:
        """The census class this donation is judged against: ``(donor element, hybridisation)``."""
        return (self.element, self.hyb)


@dataclass
class FoldReport:
    """Result of ``donor_fold()`` — the report-only donation metric; never gates."""

    angles: list[DonorAngle] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)  # donors the ruler refused to judge: the two estimators
    # disagreed on the hybridisation, or the (element, hyb) class has n < 6 in the census (`_FOLD_WINDOW`). A
    # rising count is a free perception-bug detector.

    @property
    def fold(self) -> float:
        """Max |M-D-X - class median| over every judged donation (0.0 when nothing was judged).

        Needs no reference structure — it asks the geometry about itself (a 0° carbonyl folds 176° off its class
        median), so unlike an RMSD-to-crystal metric it works in generation mode too.
        """
        return max((a.deviation for a in self.angles), default=0.0)

    @property
    def planarity(self) -> float:
        """Max deg the metal lies out of a conjugated donor's ligand plane (0.0 if none). REPORT ONLY."""
        return max((a.planarity for a in self.angles if a.planarity is not None), default=0.0)

    @property
    def outside_window(self) -> list[DonorAngle]:
        """Donations outside the census 99% window, including the overshoot the gate declines to judge.

        The gate fires on the fold floor only (see ``_FOLD_WINDOW``: a ceiling would false-positive on a healthy
        phosphine at 153.9°, which the surrogate FF has no restoring force against). Anything the census never
        realises but the gate won't reject shows up here, so a real over-splay is visible rather than dropped.
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
    """Measure how far every ligand folds off its donation axis — the metric, which never rejects.

    The gate (``donor_orientation``) fires only on the provably impossible; this reports the same walk in full,
    so a merely unusual geometry — a κ1 carboxylate relaxed to 100°, a nitrile bent to 152° — is still visible.
    Read ``.fold`` (deg off the class median), ``.planarity``, and ``.unknown``.
    """
    angles, unknown = _donor_walk(mol, _positions(mol, conf_id), donors, frozen)
    return FoldReport(angles, unknown)


def donor_orientation(mol, pos, donors=None, frozen=frozenset()) -> list[Violation]:
    """Ligands folded back over the metal — the one gate that asks where a ligand points, not where it is.

    ``metal_overbond`` is blind to the commonest fold: a short D-X bond cannot reach the over-bond floor even
    swung fully side-on (a carbonyl bent to a right angle holds its O at 2.13 Å over the 2.04 Å floor). So this
    looks at the M-D-X angle, which folding back collapses. The floor per class is the 0.5th percentile of 721
    crystal donations minus 5° (``_FOLD_WINDOW``); the class is (element, hybridisation), since a thiolate
    donates at 103° where a carboxylate donates at 126°. It fires only on an angle no reference of that class
    realises (a κ1 carboxylate at 100° deliberately isn't one — see ``donor_fold`` for the metric that reports it).

    Five exemptions, each acute *by construction*:

    * a **haptic** donor (bonded to a co-donor — side-on η² alkene, η-n ring) has no donation axis (~70° off it);
    * a **hydride / sigma-complex / agostic H** donor: no lone-pair axis;
    * a **bridging** donor (≥ 2 metals): geometry set by the bridge;
    * an **APEX** substituent (bonded to ≥ 2 donors — κ2 bite, η-n backbone): M-O-C ~90°, geometrically forced
      (the same ``overbond_tier`` exemption; a chelate's *other* arm is not exempt — `en` M-N-C ~108° is judged);
    * a D-X unit inside the **frozen** core: its orientation is the reference TS's own.

    A donor the two estimators disagree on, or an uncalibrated (element, hyb) class (n < 6), is never gated.
    ``donors`` is the intended sphere and the caller must pass it — perceiving it is circular (a folded atom
    enters the radius, is perceived as a co-donor, and its own donor is written off haptic).
    """
    out: list[Violation] = []
    for a in _donor_walk(mol, pos, donors, frozen)[0]:
        lo, hi = _FOLD_WINDOW[a.cls]
        if a.angle >= lo:  # floor only: gating the ceiling false-positives on a healthy phosphine at 153.9°
            continue  # (see `_FOLD_WINDOW`); the overshoot is reported by `FoldReport.outside_window`
        xsym = mol.GetAtomWithIdx(a.sub).GetSymbol()
        out.append(
            Violation(
                "donor_orientation",
                (a.metal, a.donor, a.sub),
                a.angle,
                lo,
                f"{a.element}{a.donor} ({str(a.hyb).lower()}) donates at {a.angle:.1f}° to {xsym}{a.sub} "
                f"(real {a.element} {str(a.hyb).lower()} donations: {lo:.0f}-{hi:.0f}°, median "
                f"{_FOLD_MEDIAN[a.cls]:.0f}°) — the ligand has folded back over the metal",
            )
        )
    return out


def _eta2_pi_atoms(mol, pos) -> set[int]:
    """Return atoms in a genuine side-on **η²** unit — a π-bonded pair binding one metal *symmetrically*.

    A side-on π ligand binds *through* its π bond, so the metal sits above the bond and legitimately pulls the
    sp2 atoms a little out of plane — a real feature, not broken geometry, so these atoms get a wider
    planarity/conjugation window.

    The signal is a pi-bonded pair BOTH near the metal AND at **~equal** metal distance (side-on is symmetric).
    Equal-distance rejects a false positive the flat 1.3x shell lets through: an alpha-diimine's imine C drifts
    inside the shell behind its sigma-donor N, but that N/C pair is lopsided, not side-on.
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
            subs = donation_axis(mol, d, all_donors, sphere, frozen)
            if subs is None:  # hydride / bridging / haptic — the donation question does not apply
                continue
            cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None  # None when estimators disagree
            if cls not in _FOLD_WINDOW:  # estimators disagreed, or n < 6 for this (element, hyb) class (an sp3
                if subs:  # ether O, an sp2 P): a threshold read off < 6 donations is noise, so abstain. Record
                    unknown.append(d)  # only if it had a judgeable substituent — an honest unknown, not a bare
                continue  # skip.
            for x in subs:
                ang = _angle(pos[m], pos[d], pos[x])
                dev = abs(ang - _FOLD_MEDIAN[cls])
                out.append(DonorAngle(m, d, x, cls[0], cls[1], ang, dev, _planarity_dev(mol, pos, hyb, m, d, x)))
    return out, unknown


def _planarity_dev(mol, pos, hyb, m, d, x) -> float | None:
    """Deg the metal lies out of a CONJUGATED donor's ligand plane (None when the D-X bond is not conjugated).

    A carboxylate/pyridine plane rotated into the coordination sphere with an acceptable M-D-X angle. Report-only
    — it can never gate: the census measured real crystals up to 87.6° out of plane (see ``_PLANARITY_P95``).
    """
    bond = mol.GetBondBetweenAtoms(d, x)
    sp2 = Chem.HybridizationType.SP2
    if bond is None or not bond.GetIsConjugated() or hyb.get(d) != sp2 or hyb.get(x) != sp2:
        return None
    best = None
    for y in mol.GetAtomWithIdx(x).GetNeighbors():
        j = y.GetIdx()
        if j in (d, m) or y.GetAtomicNum() == 1 or y.GetAtomicNum() in _METAL_Z:
            continue
        t = abs(_dihedral(pos[m], pos[d], pos[x], pos[j]))
        dev = min(t, abs(180.0 - t))  # 0 = metal lies in the ligand plane
        best = dev if best is None else min(best, dev)
    return best


def _coordinating_carbons(mol, pos) -> set[int]:
    """Return carbons in a metal's coordination shell — a carbanion / carbene / eta-2 carbon is sp3-ish.

    Such a carbon (within the ``_COORD_FACTOR`` donor shell of a metal) legitimately pyramidalises out of the
    flat sp2 plane RDKit assigns it — gfn2 confirms a coordinating ``[CH-]`` amidate carbanion stays ~0.3 A out
    of plane — so it gets the wider planarity window, not the strict flat-sp2 one.
    """
    metals = (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z)
    return {i for m in metals for i in _coordinating(mol, pos, m) if mol.GetAtomWithIdx(i).GetAtomicNum() == _CARBON_Z}


def _coordinating(mol, pos, m, declared=frozenset()) -> set[int]:
    """Non-metal atoms sitting inside metal ``m``'s coordination shell (``_COORD_FACTOR`` x covalent-sum).

    A *heavy* atom at coordination distance is unambiguously a donor. A **hydrogen** is not (a hydride and an
    agostic C-H sit at the same M-H distance), so an H counts only when in ``declared`` — the sole way an H
    enters a sphere; the element screen never overrides a declaration.
    """
    rm = _rcov(mol.GetAtomWithIdx(m).GetAtomicNum())
    out = set()
    for a in mol.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        if i == m or z in _METAL_Z or (z == 1 and i not in declared):
            continue
        if float(np.linalg.norm(pos[m] - pos[i])) <= _COORD_FACTOR * (rm + _rcov(z)):
            out.add(i)
    return out


def _spheres(mol, pos, donors=None) -> dict[int, set[int]]:
    """``{metal: its coordination sphere}`` — the intended donors where known, perceived where not.

    The one place the two in-sphere gates resolve whose donor is whose. Per metal: in a bimetallic complex the
    caller usually knows only the reacting metal's sphere, so a metal we were told nothing about falls back to
    perception rather than flagging its own ligands. A known donor no longer in the shell is dropped (it
    dissociated; ``coordination_changed`` gates that).
    """
    known = set() if donors is None else {int(d) for d in donors}
    out: dict[int, set[int]] = {}
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z):
        shell = _coordinating(mol, pos, m, known)  # `known` also admits a declared hydride the element screen
        out[m] = (known & shell) or shell  # would drop — else `known & shell` silently loses it
    return out


def _coordination_pairs(mol, pos) -> set[tuple[int, int]]:
    """Return donor-donor index pairs of each metal (a heavy atom within ``_COORD_FACTOR`` x covalent-sum).

    These are 1-3 pairs *through* the metal — cis coordination partners at the bite distance, not a steric
    clash. The metal-donor bonds are stripped on the rxembed surrogate, so donors are found geometrically.
    """
    out: set[tuple[int, int]] = set()
    for m in (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z):
        donors = sorted(_coordinating(mol, pos, m))
        for x in range(len(donors)):
            for y in range(x + 1, len(donors)):
                out.add((donors[x], donors[y]))
    return out


# --- organic conjugation perception (shared by the QA gate and the FF caps) --------------------------------


def conjugated_quartets(mol, exclude=frozenset()):
    """Yield each ``(a, c, x, s)`` quartet whose A=C-X-S dihedral measures a conjugation plane.

    A single, non-ring bond X-C (X in N/O, C bearing a double bond to A) where X carries a substituent S. The
    sole perception both ``geometry.conjugation`` and ``ConjugationCap`` read, so they cannot name different
    atoms. ``exclude`` drops a quartet whose X or C is a frozen / metal atom, as the gate does.
    """
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
            yield dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()
