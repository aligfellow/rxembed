"""The metal-donor bond length model and the surrogate's anti-overbond floors.

The fitted periodic M-L distance (`ml_distance`), its delocalised-charge input, and the tiered non-donor
floors. The FF's bondless lithium surrogate retains native vdW interactions; these additional floors
encode real-metal clearance assumptions, not missing native repulsion or bond-admission criteria.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from .metal_core import COORDINATION_METALS, VACANT, _frag_map, _haptic_sites, ligand_degree, ligand_valence

_PT = GetPeriodicTable()


_OVERBOND_MARGIN = 0.55  # Å over the covalent sum: a model clearance above the 0.45 reporting boundary.
_VDW_FLOOR_SCALE = 0.90  # x (r_vdw(M) + r_vdw(X)): extra OUTER clearance, not the surrogate's native vdW.
_FLOOR_REACH = 5  # bonds from the metal, counted through the donors: the surrogate disconnects it topologically
_INPUT_HALF_WIDTH = 0.1  # Å: lower/upper room around explicitly requested input lengths

# --- the anti-overbond tiers (`overbond_tier`) --------------------------------------------------------
# Second-sphere clearance is smaller than OUTER clearance to admit beta-agostic geometry. Haptic-backed
# apices are exempt; sigma-backed atoms retain the covalent guard, including bridges between chelate arms.
_NEAR_FLOOR_RATIO = 1.05  # x (rcov_M + rcov_X): the FF floor for a one-donor-neighbour atom
# Gates report strictly below the FF floors, so a relax resting on one is not flagged.
NEAR_REPORT_RATIO = 1.03
OUTER_REPORT_MARGIN = 0.45  # Å over the covalent sum, for a third-sphere non-donor
# A donor's licence is a window, not a half-line to zero. `rcov` is single-bond, so the shortest real M-D is a
# triple bond (Pyykkö Mn≡C = 0.82 x the sum; the tightest measured, back-bonded Mn-CO, 0.814). Below 0.70 the
# nuclei interpenetrate: this catches a donor buried in the metal, not merely a short one.
DONOR_COLLAPSE_RATIO = 0.70
_APEX_DONORS = 2  # enough attached donors to test whether a haptic face forces the backbone position
APEX, NEAR, OUTER = "apex", "near", "outer"

_SOFT_DATIVE_DONORS = frozenset({15, 33, 51})  # P/As/Sb, whose covalent radius over-states the dative M-bond.
# Relative, so it scales the group without over-contracting As/Sb. Halides/chalcogens keep the covalent sum.
_SOFT_DONOR_FRAC = 0.82  # x donor covalent radius, neutral non-haptic pnictogen only; 1.0 disables it


# --- the M-donor bond length -------------------------------------------------------------------------
# A fitted periodic model plus the guards a naive fit lacks. A formal charge is a Lewis artefact that splits
# acac's equivalent oxygens, so charge is delocalised over symmetry classes first. Known residual: nothing
# here reads bond order, so an anionic O still runs short. Terminal multiple bonds are keyed on ligand valence
# instead (`_LIGAND_FREE_CONTRACTION`).


_PHYS_COEF = (0.29574, 0.95161, 0.91467, -0.10418, 0.00875, -0.12581, 0.03590)
logger = logging.getLogger("rxembed.metal")

_METAL_GROUP = {  # Z -> group (= d-electron count); the covalent radius already carries the period, so
    **{z: z - 18 for z in range(21, 31)},  # c3*g + c4*g² is the d-electron parabola: the part of the
    **{z: z - 36 for z in range(39, 49)},  # bond length the radii alone do not capture
    **{z: z - 68 for z in range(72, 81)},
    57: 3,  # La and Lu are group 3 and reach this fit as coordination centres, so leaving them out sent every La
    71: 3,  # pair to the covalent-sum fallback: 3483 tmQM pairs at a median -0.250 A, a lookup gap not a fit error
}
# La is an extrapolation (r_cov 2.07 A against the 1.20-1.75 the fit was trained on, and no f-shell term), and
# it is still the worst-fit metal in the census, running a median 0.117 A long: that is an f-block term's job,
# not a relabelling's. `benchmark/ml_refit.py` measures held-out MAE 0.164 against 0.261 for the covalent sum,
# better in every donor cell with at least 20 pairs. Lu has zero census pairs and inherits only La's group.
_UNFITTED = set()  # elements already warned about, so a corpus run reports each gap once, not per bond
_PAULING_EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28,
    46: 2.20, 47: 1.93, 48: 1.69, 51: 2.05, 52: 2.10, 53: 2.66, 57: 1.10, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90,
    76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
}  # fmt: skip
_AGOSTIC_ELONGATION = 0.55  # Å: an agostic C-H...M is a 3c-2e sigma-complex, not a hydride, so the fit's
# 2-centre M-H cannot describe it. The tell is a metal-bound H whose other neighbour is carbon.
_SP_CONTRACTION = 0.128  # Å: an sp donor binds shorter than the fit predicts. High s-character shortens the
# sigma bond and the empty pi* takes back-donation, so one rule covers CO, isocyanide, nitrile, nitrosyl and acetylide.
# One value for every metal: making it two by gating it off for a d0 metal is measured and refuted (a d0 sp
# donor needs less, a median 0.071 Å, but less is not none, and the cell is 165 of 49410 pairs).
# `benchmark/ml_refit.py` owns the held-out fit.
_CARBON = 6
_ETA2 = 2  # mutually-bonded donors at one site -> haptic. η is a count, and M-L grows monotonically with it
# (the `c6*eta` term), so it enters the model as the island size.
_LIGAND_FREE_CONTRACTION = {  # Z -> (constant, metal-group slope), in Å
    1: (0.14589, 0.0),  # hydride
    7: (0.35500, 0.0),  # nitrido: the residual refit was worse, so retain the existing held-out value
    8: (0.67730, -0.04421),  # oxo
}
# These are metal-bound atoms with no ligand-side valence, not an element-class shortcut: aqua, ammine and
# agostic H all have a ligand-side neighbour and never enter. The group slope is earned only by oxo: over
# three structure splits it cuts held-out oxo MAE from 0.055 to 0.038-0.040 A. The hydride constant cuts
# 0.137 to 0.032-0.034 A on those splits, but the six non-overlap local pairs oppose it; nitrido keeps its old
# constant because refitting shifted its median and slightly worsened every split. Terminal sulfide and imido
# remain refuted. Measured by `benchmark/ml_refit.py` over 612,774 tmQM pairs.


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
    metals = sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS), reverse=True)
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
    """Return the shared haptic-site size for ``d``, or 0 for a sigma donor."""
    site = next(site for site in _haptic_sites(mol, donor_set) if d in site)
    return len(site) if len(site) >= _ETA2 else 0


def ml_distance(mol, metal, d, real_z, donor_set, charges=None, *, hyb):
    """Return the ideal metal-donor bond length: a periodic model fitted to TM crystal structures.

    ``c0 + c1*r_M + c2*r_D + c3*group + c4*group^2 + c5*tanh(q x dEN) + c6*eta``, fitted over tmQM/Kulik.
    Every input is chemical (element, group, charge, hapticity), never an atom index, so it is order-invariant
    and extrapolates. The tanh makes the ionic contraction saturate instead of requiring an inline clamp.

    Four deliberate departures from the published fit:

    * charge is delocalised over the metal-CUT graph, not the raw formal charge (``delocalised_charges``);
    * a neutral, non-haptic pnictogen (P/As/Sb) contracts to ``_SOFT_DONOR_FRAC`` (the fit has no dative term);
    * a donor with no ligand-side valence (``metal_core.ligand_valence``) is hydride / nitrido / oxo-like and
      takes its fitted contraction instead of a charge term;
    * a terminal non-haptic sp donor binds ``_SP_CONTRACTION`` shorter (s-character plus pi back-donation).

    Falls back to the covalent sum outside the fitted tables.
    """
    a = mol.GetAtomWithIdx(d)
    z_d = a.GetAtomicNum()
    r_m, r_d = _PT.GetRcovalent(real_z), _PT.GetRcovalent(z_d)
    g = _METAL_GROUP.get(real_z)
    if g is None or z_d not in _PAULING_EN or real_z not in _PAULING_EN:
        base = r_m + r_d  # outside the fitted tables -> the covalent sum, gracefully
        missing = real_z if g is None or real_z not in _PAULING_EN else z_d
        if missing not in _UNFITTED:  # once per element: silence here is how the La gap survived
            _UNFITTED.add(missing)
            logger.warning(
                "M-L length for %s falls back to the covalent radius sum: it is outside the fitted tables",
                _PT.GetElementSymbol(int(missing)),
            )
    else:
        if charges is None:  # never fall back to the raw formal charge: the twice-reverted Kekulé split
            charges = delocalised_charges(mol)
        # A ligand-valence-free donor owns its contraction and drops charge rather than adding it. The fitted
        # rows were terminal; a bridge inherits this as an uncalibrated extrapolation because the stripped
        # surrogate cannot retain its metal-neighbour count.
        ligand_free = z_d in _LIGAND_FREE_CONTRACTION and not ligand_valence(a)
        q = 0.0 if ligand_free else charges.get(d, a.GetFormalCharge())
        q = min(-q, 2.0) if q < 0 else 0.0  # anionic multiplicity, clamped; may be fractional (see docstring)
        c = _PHYS_COEF
        ionic = c[5] * math.tanh(q * (_PAULING_EN[z_d] - _PAULING_EN[real_z]))
        eta = _hapticity(mol, d, donor_set)
        base = c[0] + c[1] * r_m + c[2] * r_d + c[3] * g + c[4] * g * g + ionic + c[6] * eta
        # Three contractions, one chain: no donor can earn two. A haptic donor has ligand-side valence and also
        # excludes the SP correction explicitly because hybridisation is not a donation-axis descriptor for a
        # multi-atom face.
        if z_d in _SOFT_DATIVE_DONORS and eta == 0 and q == 0:  # neutral, non-haptic pnictogen binds shorter than fit
            base = r_m + _SOFT_DONOR_FRAC * r_d
        elif ligand_free:
            intercept, slope = _LIGAND_FREE_CONTRACTION[z_d]
            base -= intercept + slope * g
        elif eta == 0 and hyb.get(d) is Chem.HybridizationType.SP and ligand_degree(a) == 1:
            base -= _SP_CONTRACTION
    if z_d == 1 and any(n.GetAtomicNum() == _CARBON for n in a.GetNeighbors()):  # agostic C-H...M, not a hydride
        return base + _AGOSTIC_ELONGATION
    return base


def ff_terms(mol, cons, spheres, *, frozen=(), fragments=None, topology=None):
    """Configure the force field for every metal in ``spheres``: the one place this happens.

    ``spheres`` is ``{metal index: (real_z, [donor indices])}`` for every metal (a spectator ferrocene too, not
    just the enumerated one; wiring this per-path was the bug that left entry points on the carbon fiction).
    Fills three ``Constraints`` fields:

    * ``metals``: use a bondless Li FF surrogate with weak native vdW and no bonded metal terms;
    * ``pulls``: a soft harmonic onto the wall midpoint so the donor cannot ride a wall (a metal held as an
      all-pairs rigid body via ``cons.shapes`` gets none);
    * ``floors``: additional real-metal clearance guards.

    A metal's sphere stated as an all-pairs rigid body (`cons.shapes`) has no wall degeneracy; pulling only its
    M-donor subset tears the un-pulled donor-donor pairs, so those spheres are pulled for none of their pairs.

    Connected heavy-donor networks use distance windows so their bite geometry and native ligand force field
    can choose compatible radial distances. This policy is independent of the source of those windows.
    """
    shape_held = set().union(*cons.shapes) if cons.shapes else set()  # metals whose sphere is an all-pairs body
    frozen = set(cons.frozen) | set(frozen)
    for m, (real_z, donors) in spheres.items():
        real = [d for d in donors if d != VACANT]
        if not real:
            continue
        cons.metals.add(m)
        if m not in shape_held:  # a radial shell: pull independent donors off their walls
            fragments = _frag_map(mol) if fragments is None else fragments
            by_fragment = {}
            for donor in real:
                by_fragment.setdefault(fragments.get(donor), []).append(donor)
            # Release a radial midpoint only for an all-heavy donor network. An explicit H is a
            # three-centre donor/connection, not a flexible chelate arm: keep its pull as a connectivity guard.
            linked = {
                donor
                for group in by_fragment.values()
                if len(group) > 1
                and not (set(group) & frozen)
                and all(mol.GetAtomWithIdx(donor).GetAtomicNum() != 1 for donor in group)
                for donor in group
            }
            for d in real:
                key = (min(m, d), max(m, d))
                if key not in cons.distances:
                    continue
                if d in linked:
                    continue
                lo, hi = cons.distances[key]
                cons.pulls[key] = 0.5 * (lo + hi)
        nondonor_floors(mol, m, real_z, real, cons, topology=topology)


def overbond_tier(mol, donors, i):
    """Anti-overbond tier of NON-donor atom ``i`` against a metal whose donor set is ``donors``.

    The one place the second-sphere rule is written; `nondonor_floors` (force field), `coordination.metal_overbond`
    (gate) and `metrics.coordination_changed` (connectivity check) all key off this, so the three cannot drift.
    See the tier commentary at the top of this module for the ratios that set it.

    * ``APEX``: bonded to 2 or more donors of one and the same haptic site. That face's geometry fixes it.
    * ``NEAR``: bonded to one or more sigma donors; the covalent guard allows agostic or CMD geometry.
    * ``OUTER``: bonded to no donor; receives the larger model clearance, not a proof of absent coordination.
    """
    attached = {int(d) for d in donors if mol.GetBondBetweenAtoms(int(i), int(d)) is not None}
    if len(attached) >= _APEX_DONORS:
        # Two donors from two DIFFERENT faces (e.g. a bicyclic diene's bridgehead) leave no single face
        # geometry to fix the atom, so every attached donor must share one site, not just any haptic site.
        for site in _haptic_sites(mol, donors):
            if len(site) > 1 and len(attached & set(site)) >= _APEX_DONORS:
                return APEX
    return NEAR if attached else OUTER


def _tier_floor(z, tier, r_m, real_z):
    """Minimum M...z distance for a non-donor at ``tier``: the value the FF wall and DG relief share.

    NEAR = the covalent guard (a 1,3 atom legitimately sits inside the vdW contact); OUTER includes the
    additional real-metal vdW-scale clearance; APEX = the bare covalent sum. The two callers (`floors`,
    `dg_floors`) differ in membership (see `nondonor_floors`): the FF skips H and haptic APEX atoms, while the
    DG keeps both.
    """
    r_sum = r_m + _PT.GetRcovalent(z)
    if tier == NEAR:
        return _NEAR_FLOOR_RATIO * r_sum
    if tier == APEX:
        return r_sum
    return max(r_sum + _OVERBOND_MARGIN, _VDW_FLOOR_SCALE * (_PT.GetRvdw(real_z) + _PT.GetRvdw(z)))


def nondonor_floors(mol, metal, real_z, donors, cons, *, topology=None):
    """Record the minimum M...X for every at-risk non-donor heavy atom: the metal's steric identity.

    Tiered model clearances supplement native surrogate vdW: ``OUTER`` uses ``_VDW_FLOOR_SCALE``;
    ``NEAR`` (1,3 through its donor, position fixed by the M-donor bond/angle) keeps the
    looser covalent guard (``1.05 x r_cov sum``, forbidding a bond but admitting a beta-agostic C-H); ``APEX``
    is a haptic-backed scaffold whose position is fixed by the face geometry and remains exempt.

    Reach is measured through the donors, not the metal: ``surrogate_metal()`` strips the M-donor bonds, so the
    metal's topological distance to its own ligands is infinite.
    """
    real = [d for d in donors if d != VACANT]
    if not real:
        return
    topo = Chem.GetDistanceMatrix(mol) if topology is None else topology
    r_m = _PT.GetRcovalent(real_z)
    committed = {metal, *real}
    # A floor is for a modelled atom, never a rigid-body member (an all-pairs body already states every internal
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
        if tier == APEX:
            continue  # a pi-face scaffold owns this backbone geometry
        key = (min(metal, i), max(metal, i))
        floor = _tier_floor(a.GetAtomicNum(), tier, r_m, real_z)
        cons.floors[key] = max(cons.floors.get(key, 0.0), floor)

    # ...and the same distance into the bounds matrix, where it LOWERS a floor: RDKit floors every M...X at the
    # surrogate's ~3.4 Å carbon-vdW contact, forbidding real 2.8-3.0 Å second-sphere geometry. Same value as the
    # FF wall, different membership -- this keeps the hydrogen and the APEX atom the FF drops.
    dg_reach = {i for d in real for i in np.flatnonzero(topo[d] <= _FLOOR_REACH - 1)}
    for i in map(int, dg_reach):
        if i in committed:
            continue
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        floor = _tier_floor(z, overbond_tier(mol, real, i), r_m, real_z)
        cons.dg_floors[(min(metal, i), max(metal, i))] = floor
