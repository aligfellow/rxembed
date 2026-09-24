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

from .metal_core import _ETA2, COORDINATION_METALS, VACANT, _frag_map, _haptic_sites, ligand_degree, ligand_valence
from .utils import _CARBON_Z

_PT = GetPeriodicTable()


_OVERBOND_MARGIN = 0.55  # Å over the covalent sum: a model clearance above the 0.45 reporting boundary.
_VDW_FLOOR_SCALE = 0.90  # x (r_vdw(M) + r_vdw(X)): extra OUTER clearance, not the surrogate's native vdW.
_FLOOR_REACH = 5  # bonds from the metal, counted through the donors: the surrogate disconnects it topologically
_INPUT_HALF_WIDTH = 0.1  # Å: lower/upper room around explicitly requested input lengths

# --- the anti-overbond tiers (`overbond_tier`) --------------------------------------------------------
# Second-sphere clearance is smaller than OUTER clearance to admit beta-agostic geometry. An atom bonded to
# 2 or more donors of the metal is a bite apex or eta-n backbone, fixed by its own windows, and stays exempt.
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
# A fitted periodic model plus the guards a naive fit lacks. A formal charge is a Lewis artefact that splits
# acac's equivalent oxygens, so charge is delocalised over symmetry classes first. Known residual: nothing
# here reads bond order, so an anionic O still runs short. Terminal multiple bonds are keyed on ligand valence
# instead (`_LIGAND_FREE_CONTRACTION`).


_PHYS_COEF = (0.29574, 0.95161, 0.91467, -0.10418, 0.00875, -0.12581, 0.03590)
logger = logging.getLogger("rxembed.metal")

_METAL_GROUP = {  # Z -> group (= d-electron count); the covalent radius already carries the period, so
    **{z: z - 18 for z in range(21, 31)},  # c3*g + c4*g^2 is the d-electron parabola, the part of the
    **{z: z - 36 for z in range(39, 49)},  # bond length the radii alone do not capture
    **{z: z - 68 for z in range(72, 81)},
    57: 3,  # La and Lu are group 3 as coordination centres
    71: 3,
}
# La extrapolates past the fit's trained covalent-radius range (2.07 A against 1.20-1.75) and has no f-shell
# term, so it is the worst-fit metal in the census, a median 0.117 A too long. Lu has no census pairs of its
# own and inherits La's group.
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
_SP_CONTRACTION = 0.128  # A: an sp donor (CO, isocyanide, nitrile, nitrosyl, acetylide) binds this much
# shorter than the fit predicts, from high s-character in the sigma bond and pi back-donation into the empty pi*.
_LIGAND_FREE_CONTRACTION = {  # Z -> (constant, metal-group slope), in A; fitted over the tmQM/Kulik census
    1: (0.14589, 0.0),  # hydride
    7: (0.35500, 0.0),  # nitrido
    8: (0.67730, -0.04421),  # oxo
}
# A donor here has no ligand-side neighbour at all: aqua, ammine and agostic H all keep one and never enter.


def delocalised_charges(mol):
    """Return `{atom: charge}`, spreading each formal charge over the atoms it is actually delocalised across.

    A raw formal charge is a Lewis artefact: a carboxylate's two O are equivalent, but only one carries the
    charge depending which Kekule form was typed. Average it instead over each connectivity-symmetry class of
    the metal-cut graph (canonical ranking, bond orders and charges cleared): a carboxylate's O, acac's O and
    an amidinate's N share one charge, while a genuinely inequivalent pair (an oxo plus an acac O on one
    vanadium) keeps its own. Cutting the metal first makes the result the same with or without a bonded metal,
    and re-merges a kappa1 carboxylate's two O. The charge may come back fractional. Returns `{}` if the
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


def _hapticity(mol, d, donor_set, sites=None):
    """Return the shared haptic-site size for ``d``, or 0 for a sigma donor.

    ``sites`` reuses an already-grouped `_haptic_sites` call instead of regrouping the whole donor set.
    """
    sites = _haptic_sites(mol, donor_set) if sites is None else sites
    site = next(site for site in sites if d in site)
    return len(site) if len(site) >= _ETA2 else 0


def ml_distance(mol, metal, d, real_z, donor_set, charges=None, *, hyb, eta=None):
    """Return the ideal metal-donor bond length: a periodic model fitted to TM crystal structures.

    `c0 + c1*r_M + c2*r_D + c3*group + c4*group^2 + c5*tanh(q x dEN) + c6*eta`, fitted over tmQM/Kulik.
    Falls back to the covalent radius sum outside the fitted tables.

    Four departures from the published fit:

    * charge is delocalised over the metal-cut graph, not the raw formal charge (`delocalised_charges`);
    * a neutral, non-haptic pnictogen (P/As/Sb) contracts to `_SOFT_DONOR_FRAC` (the fit has no dative term);
    * a donor with no ligand-side valence (hydride, nitrido, oxo) takes its own fitted contraction instead of
      a charge term (see `metal_core.ligand_valence`);
    * a terminal, non-haptic sp donor binds `_SP_CONTRACTION` shorter (s-character plus pi back-donation).

    `eta` reuses a caller-precomputed hapticity (`_hapticity`) instead of regrouping `donor_set` per donor;
    M-L grows monotonically with it, so it enters the fit linearly as the donor-island size.
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
        if charges is None:  # never the raw formal charge: it is a Lewis artefact (see delocalised_charges)
            charges = delocalised_charges(mol)
        # A ligand-valence-free donor owns its contraction and drops charge rather than adding it. The fitted
        # rows were terminal; a bridge inherits this as an uncalibrated extrapolation because the stripped
        # surrogate cannot retain its metal-neighbour count.
        ligand_free = z_d in _LIGAND_FREE_CONTRACTION and not ligand_valence(a)
        q = 0.0 if ligand_free else charges.get(d, a.GetFormalCharge())
        q = min(-q, 2.0) if q < 0 else 0.0  # anionic multiplicity, clamped; may be fractional (see docstring)
        c = _PHYS_COEF
        ionic = c[5] * math.tanh(q * (_PAULING_EN[z_d] - _PAULING_EN[real_z]))
        eta = _hapticity(mol, d, donor_set) if eta is None else eta
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
    if z_d == 1 and any(n.GetAtomicNum() == _CARBON_Z for n in a.GetNeighbors()):  # agostic C-H...M, not a hydride
        return base + _AGOSTIC_ELONGATION
    return base


def ff_terms(mol, cons, spheres, *, frozen=(), fragments=None, topology=None, sites=None):
    """Set up the force field for every metal in `spheres`: the one place this happens.

    Fills three `Constraints` fields: `metals` (a bondless Li surrogate with weak native vdW and no bonded
    metal terms), `pulls` (a soft pull to the wall midpoint so a donor cannot ride its wall) and `floors`
    (extra real-metal clearance). A metal held as an all-pairs rigid body (`cons.shapes`) already states every
    internal distance and gets no pulls: pulling only its M-donor subset would tear the un-pulled donor-donor
    pairs. An all-heavy donor network also gets no pull, left free to let its own bite geometry and the native
    ligand force field pick compatible radial distances.

    `spheres` is `{metal index: (real_z, [donor indices])}` for every metal, including a spectator that is
    not the one being enumerated. `sites` reuses an already-grouped `_haptic_sites` call; pass it only when
    `spheres` holds one metal, since it goes to every sphere's floors unchanged.
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
        nondonor_floors(mol, m, real_z, real, cons, topology=topology, sites=sites)


def overbond_tier(mol, donors, i, sites=None):
    """Anti-overbond tier of non-donor atom `i` against a metal whose donor set is `donors`.

    The one place the second-sphere rule is written; `nondonor_floors` (force field), `coordination.metal_overbond`
    (gate) and `metrics.coordination_changed` (connectivity check) all key off this, so the three cannot drift.
    See the tier ratios at the top of this module. `sites` is accepted for call-signature compatibility with
    callers that precompute `_haptic_sites`; unused here.

    * `APEX`: bonded to 2 or more donors, a chelate bite apex or eta-n backbone. Forced, never floored.
    * `NEAR`: bonded to exactly 1 donor, the second sphere. Floored, but loosely enough to admit a real
      agostic or CMD contact.
    * `OUTER`: bonded to no donor, the third sphere and beyond. Only a collapse puts it this close.
    """
    attached = {int(d) for d in donors if mol.GetBondBetweenAtoms(int(i), int(d)) is not None}
    if len(attached) >= _APEX_DONORS:
        return APEX
    return NEAR if attached else OUTER


def _tier_floor(z, tier, r_m, real_z):
    """Minimum M...z distance for a non-donor at ``tier``: the value the FF wall and DG relief share.

    NEAR = the covalent guard (a 1,3 atom legitimately sits inside the vdW contact); OUTER includes the
    additional real-metal vdW-scale clearance; APEX = the bare covalent sum. The two callers (`floors`,
    `dg_floors`) differ in membership (see `nondonor_floors`): the FF skips H and APEX atoms, while the
    DG keeps both.
    """
    r_sum = r_m + _PT.GetRcovalent(z)
    if tier == NEAR:
        return _NEAR_FLOOR_RATIO * r_sum
    if tier == APEX:
        return r_sum
    return max(r_sum + _OVERBOND_MARGIN, _VDW_FLOOR_SCALE * (_PT.GetRvdw(real_z) + _PT.GetRvdw(z)))


def nondonor_floors(mol, metal, real_z, donors, cons, *, topology=None, sites=None):
    """Record the minimum M...X for every at-risk non-donor heavy atom: the metal's steric identity.

    `OUTER` gets `_VDW_FLOOR_SCALE` vdW clearance; `NEAR` (1,3 through its donor, position fixed by the
    M-donor bond/angle) keeps the looser covalent guard (`1.05 x r_cov sum`), forbidding a bond but admitting
    a real beta-agostic contact; `APEX` (>= 2 donors) cannot move and is not floored.

    Reach is measured through the donors, not the metal: the FF surrogate strips the M-donor bonds, so the
    metal's own topological distance to its ligands is otherwise infinite. `sites` reuses an already-grouped
    `_haptic_sites` call, shared with every `overbond_tier` call this makes.
    """
    real = [d for d in donors if d != VACANT]
    if not real:
        return
    sites = _haptic_sites(mol, real) if sites is None else sites
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
        tier = overbond_tier(mol, real, i, sites=sites)
        if tier == APEX:  # a bite apex cannot move: its distance is already fixed by the two M-donor windows, so
            continue  # the FF needs no wall here; the DG still relieves it below, RDKit flooring it regardless
        key = (min(metal, i), max(metal, i))
        floor = _tier_floor(a.GetAtomicNum(), tier, r_m, real_z)
        cons.floors[key] = max(cons.floors.get(key, 0.0), floor)

    # The same distance also lowers a floor in the bounds matrix: RDKit floors every M...X at the surrogate's
    # ~3.4 A carbon-vdW contact, forbidding real 2.8-3.0 A second-sphere geometry. Same value as the FF wall,
    # different membership: this keeps the hydrogen and the APEX atom that the FF drops.
    dg_reach = {i for d in real for i in np.flatnonzero(topo[d] <= _FLOOR_REACH - 1)}
    for i in map(int, dg_reach):
        if i in committed:
            continue
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        floor = _tier_floor(z, overbond_tier(mol, real, i, sites=sites), r_m, real_z)
        cons.dg_floors[(min(metal, i), max(metal, i))] = floor
