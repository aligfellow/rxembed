"""The metal-donor bond length model and the surrogate's anti-overbond floors.

The fitted periodic M-L distance (`ml_distance`), the delocalised-charge input it reads, and the tiered
non-donor floors (`nondonor_floors` / `overbond_tier` / `ff_terms`) that replace the van der Waals the
carbon/lithium surrogate deletes. Carved out of `metal`; imports only its foundational constants.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from .metal_core import TRANSITION_METALS, VACANT

_PT = GetPeriodicTable()


_OVERBOND_MARGIN = 0.55  # Å over the covalent sum: how close a third-sphere non-donor may sit before it is a
# bond. Above the 0.45 reporting floor, so a relax resting on its floor never trips the gate.
_VDW_FLOOR_SCALE = 0.90  # x (r_vdw(M) + r_vdw(X)) for an outer atom, replacing the sterics the zero-vdW
# surrogate deleted. NEAR/APEX keep the covalent scale: their position is fixed by the M-donor bond.
_FLOOR_REACH = 5  # bonds from the metal, counted through the donors: the surrogate disconnects it topologically

# --- the anti-overbond tiers (`overbond_tier`) --------------------------------------------------------
# The second sphere fits neither blanket rule: floored it rejects a β-agostic ethyl, exempted the zero-vdW FF
# folds it into a vacant vertex. Split on donor count: >=2 is a bite apex fixed by its own windows, exactly 1 is
# free to rotate into an empty vertex and is the only one floored.
_NEAR_FLOOR_RATIO = 1.05  # x (rcov_M + rcov_X): the FF floor for a one-donor-neighbour atom
# Gates report strictly below the FF floors, so a relax resting on one is not flagged.
NEAR_REPORT_RATIO = 1.03
OUTER_REPORT_MARGIN = 0.45  # Å over the covalent sum, for a third-sphere non-donor
# A donor's licence is a window, not a half-line to zero. `rcov` is single-bond, so the shortest real M-D is a
# triple bond (Pyykkö Mn≡C = 0.82 x the sum; the tightest measured, back-bonded Mn-CO, 0.814). Below 0.70 the
# nuclei interpenetrate: this catches a donor buried in the metal, not merely a short one.
DONOR_COLLAPSE_RATIO = 0.70
_APEX_DONORS = 2  # bonded to >= this many donors of this metal -> a geometrically forced bite apex: never floored
APEX, NEAR, OUTER = "apex", "near", "outer"

_SOFT_DATIVE_DONORS = frozenset({15, 33, 51})  # P/As/Sb, whose covalent radius over-states the dative M-bond.
# Relative, so it scales the group without over-contracting As/Sb. Halides/chalcogens keep the covalent sum.
_SOFT_DONOR_FRAC = 0.82  # x donor covalent radius, neutral non-haptic pnictogen only; 1.0 disables it


# --- the M-donor bond length -------------------------------------------------------------------------
# A fitted periodic model plus two guards a naive fit lacks. A formal charge is a Lewis artefact that splits
# acac's equivalent oxygens, so charge is delocalised over symmetry classes first and its contraction bounded.
# Known residual: nothing here reads bond ORDER, so an anionic O runs short and an M=O long.


_PHYS_COEF = (0.29574, 0.95161, 0.91467, -0.10418, 0.00875, -0.12043, 0.03590)
_METAL_GROUP = {  # Z -> group (= d-electron count); the covalent radius already carries the period, so
    **{z: z - 18 for z in range(21, 31)},  # c3*g + c4*g² is the d-electron parabola: the part of the
    **{z: z - 36 for z in range(39, 49)},  # bond length the radii alone do not capture
    **{z: z - 68 for z in range(72, 81)},
}
_PAULING_EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28,
    46: 2.20, 47: 1.93, 48: 1.69, 51: 2.05, 52: 2.10, 53: 2.66, 57: 1.10, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90,
    76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
}  # fmt: skip
_AGOSTIC_ELONGATION = 0.55  # Å: an agostic C-H...M is a 3c-2e sigma-complex, not a hydride, so the fit's
# 2-centre M-H cannot describe it. The tell is a metal-bound H whose other neighbour is carbon.
_PI_ACCEPTOR_OFFSET = 0.099  # Å: a carbonyl / isocyanide C binds shorter than the fit predicts, π-backbonding
# pulling the metal in. The tell is a carbon donor with a C≡O or C≡N.
_CARBON = 6
_ETA2 = 2  # mutually-bonded donors at one site -> haptic. η is a count, and M-L grows monotonically with it
# (the `c6*eta` term), so it enters the model as the island size.
_MAX_CONTRACTION = 0.12  # Å bound on the ionic contraction: a correction, not an annihilation. Unbounded, a
# late-TM anionic O comes out ~0.2 Å short, and nothing gates over-short. An interim guard; the fix is a refit.


def delocalised_charges(mol):
    """Formal charge spread over the atoms it is actually delocalised across. ``{atom: q}``, possibly fractional.

    A raw formal charge is a Lewis artefact: a carboxylate's two O are equivalent but only one carries the (-1),
    depending on the Kekulé structure typed. The charge is instead averaged over each connectivity-symmetry class
    of the metal-cut graph (canonical ranking, bond orders and charges cleared, metals removed): a carboxylate's
    O, acac's, an amidinate's N share one charge. Cutting the metal makes it path-INDEPENDENT (a real-metal call
    and the surrogate-stripped call agree) and re-merges a kappa1 carboxylate's two O. A genuinely inequivalent
    pair (an oxo + an acac O on one vanadium) differs in connectivity and keeps its own. Degrades to ``{}`` if the
    metal-cut graph cannot be ranked.
    """
    metals = sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS), reverse=True)
    flat = Chem.RWMol(Chem.Mol(mol))
    for m in metals:  # cut the metal so the delocalisation is identical with or without a bonded metal present
        flat.RemoveAtom(m)
    keep = [i for i in range(mol.GetNumAtoms()) if i not in set(metals)]  # flat-local index -> parent (mol) index
    for b in flat.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)  # resonance-blind: only the connectivity and the elements remain
    for a in flat.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetIsAromatic(False)
    fm = flat.GetMol()
    try:  # charge is achiral / isotope-blind; a pathological cut graph degrades to {} rather than aborting the embed
        fm.UpdatePropertyCache(strict=False)
        klass = list(Chem.CanonicalRankAtoms(fm, breakTies=False, includeChirality=False, includeIsotopes=False))
    except Exception:
        return {}
    members: dict = {}
    for local, parent in enumerate(keep):
        members.setdefault(klass[local], []).append(parent)
    out = {}
    for group in members.values():
        q = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in group) / len(group)
        for i in group:
            out[i] = q
    return out


def _hapticity(mol, d, donor_set):
    """Size of the mutually-bonded donor island containing ``d`` (eta^n), or 0 if it is a lone sigma donor."""
    seen, stack = {d}, [d]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            i = nb.GetIdx()
            if i in donor_set and i not in seen:
                seen.add(i)
                stack.append(i)
    return len(seen) if len(seen) >= _ETA2 else 0


def ml_distance(mol, metal, d, real_z, donor_set, charges=None):
    """Return the ideal metal-donor bond length: a periodic model fitted to TM crystal structures.

    ``c0 + c1*r_M + c2*r_D + c3*group + c4*group^2 + c5*(q x dEN) + c6*eta``, OLS-fitted over tmQM/Kulik. Every
    input is chemical (element, group, charge, hapticity), never an atom index, so it is order-invariant and
    extrapolates. Beats the plain covalent-radius sum, most on anionic-O donors.

    Three deliberate departures from the published fit:

    * charge is delocalised over the metal-CUT graph, not the raw formal charge (``delocalised_charges``);
    * a neutral, non-haptic pnictogen (P/As/Sb) contracts to ``_SOFT_DONOR_FRAC`` (the fit has no dative term);
    * a carbonyl / isocyanide C binds ``_PI_ACCEPTOR_OFFSET`` shorter (pi-backbonding).

    Falls back to the covalent sum outside the fitted tables.
    """
    a = mol.GetAtomWithIdx(d)
    z_d = a.GetAtomicNum()
    r_m, r_d = _PT.GetRcovalent(real_z), _PT.GetRcovalent(z_d)
    g = _METAL_GROUP.get(real_z)
    if g is None or z_d not in _PAULING_EN or real_z not in _PAULING_EN:
        base = r_m + r_d  # outside the fitted tables -> the covalent sum, gracefully
    else:
        if charges is None:  # never fall back to the raw formal charge: the twice-reverted Kekulé split
            charges = delocalised_charges(mol)
        q = charges.get(d, a.GetFormalCharge())
        q = min(-q, 2.0) if q < 0 else 0.0  # anionic multiplicity, clamped; may be fractional (see docstring)
        c = _PHYS_COEF
        ionic = c[5] * q * (_PAULING_EN[z_d] - _PAULING_EN[real_z])  # the ionic character of this bond
        ionic = max(ionic, -_MAX_CONTRACTION)  # a bounded correction, not an annihilation
        eta = _hapticity(mol, d, donor_set)
        base = c[0] + c[1] * r_m + c[2] * r_d + c[3] * g + c[4] * g * g + ionic + c[6] * eta
        if z_d in _SOFT_DATIVE_DONORS and eta == 0 and q == 0:  # neutral, non-haptic pnictogen binds shorter than fit
            base = r_m + _SOFT_DONOR_FRAC * r_d
    if z_d == 1 and any(n.GetAtomicNum() == _CARBON for n in a.GetNeighbors()):  # agostic C-H...M, not a hydride
        return base + _AGOSTIC_ELONGATION
    if z_d == _CARBON and any(  # carbonyl / isocyanide C (C#O or C#N): pi-backbonding pulls the metal in
        b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(a).GetAtomicNum() in (7, 8) for b in a.GetBonds()
    ):
        return base - _PI_ACCEPTOR_OFFSET
    return base


def ff_terms(mol, cons, spheres):
    """Configure the force field for every metal in ``spheres``: the one place this happens.

    ``spheres`` is ``{metal index: (real_z, [donor indices])}`` for every metal (a spectator ferrocene too, not
    just the enumerated one; wiring this per-path was the bug that left entry points on the carbon fiction).
    Fills three ``Constraints`` fields:

    * ``metals``: re-typed to a zero-vdW element, else UFF shoves every M-donor pair out with fictitious LJ;
    * ``pulls``: a soft harmonic onto the wall midpoint so the donor cannot ride a wall (a ``hold_shape`` sphere
      gets none);
    * ``floors``: the anti-overbond guard replacing the deleted vdW.

    A ``hold_shape`` sphere (`cons.shapes`) is an all-pairs body with no wall degeneracy; pulling only its
    M-donor subset tears the un-pulled donor-donor pairs, so those spheres are pulled for none of their pairs.
    """
    shape_held = set().union(*cons.shapes) if cons.shapes else set()  # metals whose sphere is an all-pairs body
    for m, (real_z, donors) in spheres.items():
        real = [d for d in donors if d != VACANT]
        if not real:
            continue
        cons.metals.add(m)
        if m not in shape_held:  # a modelled window: pull the donor off the wall it would otherwise ride
            for d in real:
                key = (min(m, d), max(m, d))
                if key in cons.distances:  # the wall the caller wrote -> its midpoint is the target
                    lo, hi = cons.distances[key]
                    cons.pulls[key] = 0.5 * (lo + hi)
        nondonor_floors(mol, m, real_z, real, cons)


def overbond_tier(mol, donors, i):
    """Anti-overbond tier of NON-donor atom ``i`` against a metal whose donor set is ``donors``.

    The one place the second-sphere rule is written; `nondonor_floors` (force field), `coordination.metal_overbond`
    (gate) and `metrics.coordination_changed` (connectivity check) all key off this, so the three cannot drift.
    See the tier commentary at the top of this module for the ratios that set it.

    * ``APEX``: bonded to 2 or more donors of this metal, a chelate bite apex or eta-n backbone. Forced,
      never floored.
    * ``NEAR``: bonded to exactly 1 donor, the second sphere. Free to swing into a vacant vertex, so it is
      floored, but only at a ratio loose enough to admit a real agostic / CMD contact.
    * ``OUTER``: bonded to no donor, the third sphere and beyond. Nothing but a collapse puts it near the metal.
    """
    n = sum(1 for d in donors if mol.GetBondBetweenAtoms(int(i), int(d)) is not None)
    if n >= _APEX_DONORS:
        return APEX
    return NEAR if n else OUTER


def _tier_floor(z, tier, r_m, real_z):
    """Minimum M...z distance for a non-donor at ``tier``: the value the FF wall and DG relief share.

    NEAR = the covalent guard (a 1,3 atom legitimately sits inside the vdW contact); outer = the vdW contact that
    replaces the surrogate's deleted Lennard-Jones; APEX = the bare covalent sum. The two callers (`floors`,
    `dg_floors`) differ only in membership (see `nondonor_floors`): the FF skips H and APEX, the DG keeps both,
    never in this value.
    """
    r_sum = r_m + _PT.GetRcovalent(z)
    if tier == NEAR:
        return _NEAR_FLOOR_RATIO * r_sum
    if tier == APEX:
        return r_sum
    return max(r_sum + _OVERBOND_MARGIN, _VDW_FLOOR_SCALE * (_PT.GetRvdw(real_z) + _PT.GetRvdw(z)))


def nondonor_floors(mol, metal, real_z, donors, cons):
    """Record the minimum M...X for every at-risk non-donor heavy atom: the metal's steric identity.

    The other half of the zero-vdW force field: the surrogate's Lennard-Jones was the only thing keeping a
    non-donor off the metal (every clash gate excludes metals), so deleting it without replacement lets ligands
    collapse inward.

    Two physics, tiered by ``overbond_tier``: ``OUTER`` (bonded to nothing) is held off by a vdW contact
    (``_VDW_FLOOR_SCALE``); ``NEAR`` (1,3 through its donor, position fixed by the M-donor bond/angle) keeps the
    looser covalent guard (``1.05 x r_cov sum``, forbidding a bond but admitting a beta-agostic C-H); ``APEX``
    (>= 2 donors) cannot move and is not floored.

    Reach is measured through the donors, not the metal: ``surrogate_metal()`` strips the M-donor bonds, so the
    metal's topological distance to its own ligands is infinite.
    """
    real = [d for d in donors if d != VACANT]
    if not real:
        return
    topo = Chem.GetDistanceMatrix(mol)
    r_m = _PT.GetRcovalent(real_z)
    committed = {metal, *real}
    # A floor is for a modelled atom, never a rigid-body member (a `hold_shape` body already states every internal
    # distance, so an outside floor over-determines it and the relax tears the body) nor an atom whose M...X is
    # already an explicit window (a frozen TS core, whose window is the truth). Same rule as ff_terms' pulls.
    rigid = set().union(*cons.shapes) if cons.shapes else set()
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in committed or a.GetAtomicNum() == 1:  # H is not what re-perception over-bonds
            continue
        if i in rigid or (min(metal, i), max(metal, i)) in cons.distances:
            continue
        hops = 1 + min(topo[d][i] for d in real)  # M -> donor -> ... -> X
        if hops > _FLOOR_REACH:
            continue
        tier = overbond_tier(mol, real, i)
        if tier == APEX:  # a bite apex cannot move: its distance is already fixed by the two M-donor windows, so
            continue  # the FF needs no wall here; the DG still relieves it below, RDKit flooring it regardless
        cons.floors[(min(metal, i), max(metal, i))] = _tier_floor(a.GetAtomicNum(), tier, r_m, real_z)

    # ...and the same distance into the bounds matrix, where it LOWERS a floor: RDKit floors every M...X at the
    # surrogate's ~3.4 Å carbon-vdW contact, forbidding real 2.8-3.0 Å second-sphere geometry. Same value as the
    # FF wall, different membership -- this keeps the hydrogen and the APEX atom the FF drops.
    dg_reach = {i for d in real for i in np.flatnonzero(topo[d] <= _FLOOR_REACH - 1)}
    for i in map(int, dg_reach):
        if i in committed:
            continue
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        cons.dg_floors[(min(metal, i), max(metal, i))] = _tier_floor(z, overbond_tier(mol, real, i), r_m, real_z)
