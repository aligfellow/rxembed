"""Say how long each metal-donor bond is, and how close a non-donor may come to a metal.

`ml_distance` is the fitted periodic M-L length. The tiered non-donor floors (`overbond_tier`) give the force field's
bondless surrogate the clearance a real metal has; they are not a bond criterion.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from rxembed.geometry.constraints import Constraints, add_distance, canonical_key, graft_owns
from rxembed.graph import COORDINATION_METALS, frag_map, ligand_degree, ligand_graph
from rxembed.metal.centres import ligand_valence
from rxembed.utils import flat_ranks

_PT = GetPeriodicTable()


_OVERBOND_MARGIN = 0.55  # Å over the covalent sum: a model clearance above the 0.45 reporting boundary.
_VDW_FLOOR_SCALE = 0.90  # x (r_vdw(M) + r_vdw(X)): extra OUTER clearance, not the surrogate's native vdW.
_FLOOR_REACH = 5  # bonds from the metal, counted through the donors: the surrogate disconnects it topologically
INPUT_HALF_WIDTH = 0.1  # Å: lower/upper room around explicitly requested input lengths

# --- the anti-overbond tiers (`overbond_tier`) --------------------------------------------------------
_NEAR_FLOOR_RATIO = 1.05  # x (rcov_M + rcov_X): the FF floor for a one-donor-neighbour atom
# Gates report strictly below the FF floors, so a relax resting on one is not flagged.
NEAR_REPORT_RATIO = 1.03
OUTER_REPORT_MARGIN = 0.45  # Å over the covalent sum, for a third-sphere non-donor
# the shortest real M-D, a triple bond, is about 0.82 x the single-bond sum; below 0.70 nuclei overlap
DONOR_COLLAPSE_RATIO = 0.70
APEX_DONORS = 2  # donors of one metal an atom bonds to that make it a bite apex (`overbond_tier`)
APEX, NEAR, OUTER = "apex", "near", "outer"

_SOFT_DATIVE_DONORS = frozenset({15, 33, 51})  # P/As/Sb, whose covalent radius over-states the dative M-bond.
# Relative, so it scales the group without over-contracting As/Sb. Halides/chalcogens keep the covalent sum.
_SOFT_DONOR_FRAC = 0.82  # x donor covalent radius, neutral non-haptic pnictogen only; 1.0 disables it


# the M-donor bond length: a fitted periodic model plus the guards `ml_distance` lists


_PHYS_COEF = (0.29574, 0.95161, 0.91467, -0.10418, 0.00875, -0.12581, 0.03590)
logger = logging.getLogger("rxembed.metal")

_METAL_GROUP = {  # Z -> group (= d-electron count); the covalent radius already carries the period, so
    **{z: z - 18 for z in range(21, 31)},  # c3*g + c4*g^2 is the d-electron parabola, the part of the
    **{z: z - 36 for z in range(39, 49)},  # bond length the radii alone do not capture
    **{z: z - 68 for z in range(72, 81)},
    57: 3,  # La is group 3 as a coordination centre
}
# La lies past the fitted radius range with no f-shell term, so it runs long, yet beats the covalent sum
_UNFITTED = set()  # elements already warned about, so each gap is reported once, not per bond
_PAULING_EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28,
    46: 2.20, 47: 1.93, 48: 1.69, 51: 2.05, 52: 2.10, 53: 2.66, 57: 1.10, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90,
    76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
}  # fmt: skip
# a ligand C-H or B-H on a metal is a 3c-2e bridge, not a hydride: X -> (M-H offset, X-H leg), Å
BRIDGE_H = {6: (0.152, 1.199), 5: (0.069, 1.333)}


def bridge_partners(mol, atom):
    """Return the ligand neighbours that make metal-bound `atom` an X-H...M bridging H, else ``()``.

    The one test of a bridge: an H whose X is a `BRIDGE_H` element. Any other metal-bound H is a hydride.
    """
    found = mol.GetAtomWithIdx(atom)
    return tuple(n for n in found.GetNeighbors() if n.GetAtomicNum() in BRIDGE_H) if found.GetAtomicNum() == 1 else ()


_SP_CONTRACTION = 0.128  # A: an sp donor (CO, isocyanide, nitrile, nitrosyl, acetylide) binds this much
# shorter than the fit, from s-character and pi back-donation; one value for every metal
_LIGAND_FREE_CONTRACTION = {  # Z -> (constant, metal-group slope), in A, fitted to crystal structures
    1: (0.14589, 0.0),  # hydride
    7: (0.35500, 0.0),  # nitrido: the prior constant, not refitted
    8: (0.67730, -0.04421),  # oxo: only oxo carries a group slope
}
# A donor here has no ligand-side neighbour at all: aqua, ammine and agostic H all keep one and never enter.


def delocalised_charges(mol):
    """Return ``{atom: charge}`` for every non-metal atom, each formal charge averaged over its symmetry class.

    Which carboxylate O carries the charge is a Lewis artefact, so charges are averaged over the `flat_ranks` classes
    of the metal-cut graph: a carboxylate's O, acac's O, an amidinate's N and a porphyrin dianion's four N share one
    charge, while an oxo and an acac O on one vanadium keep their own. The charge may be fractional.
    """
    metals = sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS), reverse=True)
    cut = Chem.RWMol(mol)
    for m in metals:
        cut.RemoveAtom(m)
    keep = [i for i in range(mol.GetNumAtoms()) if i not in set(metals)]  # cut index -> parent (mol) index
    klass = flat_ranks(cut.GetMol())
    members: dict = {}
    for local, parent in enumerate(keep):
        members.setdefault(klass[local], []).append(parent)
    out = {}
    for group in members.values():
        q = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in group) / len(group)
        for i in group:
            out[i] = q
    return out


def ml_distance(mol, d, real_z, charges, *, hyb, eta):
    """Return the ideal metal-donor bond length from a periodic model fitted to transition-metal crystal structures.

    `c0 + c1*r_M + c2*r_D + c3*group + c4*group^2 + c5*tanh(q x dEN) + c6*eta`; outside the fitted tables it falls back
    to the covalent radius sum. Five departures from the plain fit:

    * charge is delocalised over the metal-cut graph (`delocalised_charges`); * a neutral non-haptic P, As or Sb
    contracts to `_SOFT_DONOR_FRAC`, since the fit has no dative term; * a donor with no ligand-side valence (hydride,
    nitrido, oxo) takes its own contraction instead of a charge term; * a terminal non-haptic sp donor binds
    `_SP_CONTRACTION` shorter (s-character and pi back-donation); * an X-H...M bridging H adds its `BRIDGE_H` offset.

    `eta` is 0 for a sigma donor, else the site size. Bond order is not read, so an anionic O donor still runs short.
    """
    a = mol.GetAtomWithIdx(d)
    z_d = a.GetAtomicNum()
    r_m, r_d = _PT.GetRcovalent(real_z), _PT.GetRcovalent(z_d)
    g = _METAL_GROUP.get(real_z)
    if g is None or z_d not in _PAULING_EN or real_z not in _PAULING_EN:
        base = r_m + r_d  # outside the fitted tables -> the covalent sum, gracefully
        missing = real_z if g is None or real_z not in _PAULING_EN else z_d
        if missing not in _UNFITTED:  # once per element
            _UNFITTED.add(missing)
            logger.warning(
                "M-L length for %s falls back to the covalent radius sum: it is outside the fitted tables",
                _PT.GetElementSymbol(int(missing)),
            )
    else:
        # a ligand-valence-free donor takes its own contraction; a bridge extrapolates the terminal fit
        ligand_free = z_d in _LIGAND_FREE_CONTRACTION and not ligand_valence(a)
        q = 0.0 if ligand_free else charges[d]
        q = min(-q, 2.0) if q < 0 else 0.0  # anionic multiplicity, clamped; may be fractional (see docstring)
        c = _PHYS_COEF
        ionic = c[5] * math.tanh(q * (_PAULING_EN[z_d] - _PAULING_EN[real_z]))
        base = c[0] + c[1] * r_m + c[2] * r_d + c[3] * g + c[4] * g * g + ionic + c[6] * eta
        # at most one contraction: a haptic donor has ligand valence, and eta == 0 excludes faces from the others
        if z_d in _SOFT_DATIVE_DONORS and eta == 0 and q == 0:  # neutral, non-haptic pnictogen binds shorter than fit
            base = r_m + _SOFT_DONOR_FRAC * r_d
        elif ligand_free:
            intercept, slope = _LIGAND_FREE_CONTRACTION[z_d]
            base -= intercept + slope * g
        elif eta == 0 and hyb.get(d) is Chem.HybridizationType.SP and ligand_degree(a) == 1:
            base -= _SP_CONTRACTION
    if partners := bridge_partners(mol, d):  # an X-H...M bridge, not a hydride
        return base + BRIDGE_H[partners[0].GetAtomicNum()][0]
    return base


def ff_terms(mol, cons, metal, real_z, donors, *, frozen=(), fragments=None, topology=None):
    """Return `cons` with the force field set up for one metal and its real coordinating atoms.

    It adds `metal` as a bondless surrogate, a pull to the middle of each donor's distance window so no donor rides its
    wall, and the non-donor floors. Two or more heavy unfrozen donors of one fragment get no pull: their bite and the
    ligand force field choose compatible distances.
    """
    if not donors:
        return cons
    fragments = frag_map(mol) if fragments is None else fragments
    by_fragment = {}
    for donor in donors:
        by_fragment.setdefault(fragments.get(donor), []).append(donor)
    # An H donor is a three-centre connection, not a flexible chelate arm, so its pull stays.
    linked = {
        donor
        for group in by_fragment.values()
        if len(group) > 1
        and not (set(group) & set(frozen))
        and all(mol.GetAtomWithIdx(donor).GetAtomicNum() != 1 for donor in group)
        for donor in group
    }
    pulls = dict(cons.pulls)
    for d in donors:
        key = canonical_key((metal, d))
        if key in cons.distances and d not in linked:
            lo, hi = cons.distances[key]
            pulls[key] = (0.5 * (lo + hi),) * 2
    cons = replace(cons, metals=cons.metals | {metal}, pulls=pulls)
    return nondonor_floors(mol, metal, real_z, donors, cons, topology=topology)


def overbond_tier(mol, donors, i):
    """Return the tier of non-donor atom `i` against a metal with donors `donors`: the one second-sphere rule.

    * `APEX`: bonded to two or more donors, a bite apex or eta-n backbone; never floored. * `NEAR`: bonded to one donor,
    the second sphere; floored loosely enough for a real agostic or CMD contact. * `OUTER`: bonded to no donor; only a
    collapse brings it this close.
    """
    attached = {int(d) for d in donors if mol.GetBondBetweenAtoms(int(i), int(d)) is not None}
    if len(attached) >= APEX_DONORS:
        return APEX
    return NEAR if attached else OUTER


def _tier_floor(z, tier, r_m, real_z):
    """Return the minimum M...z distance for a non-donor at `tier`, shared by the FF wall and the DG relief.

    `NEAR` is a covalent guard, since a 1,3 atom sits inside the vdW contact; `OUTER` adds vdW-scale clearance; `APEX`
    is the bare covalent sum.
    """
    r_sum = r_m + _PT.GetRcovalent(z)
    if tier == NEAR:
        return _NEAR_FLOOR_RATIO * r_sum
    if tier == APEX:
        return r_sum
    return max(r_sum + _OVERBOND_MARGIN, _VDW_FLOOR_SCALE * (_PT.GetRvdw(real_z) + _PT.GetRvdw(z)))


def nondonor_floors(mol, metal, real_z, real, cons, *, topology=None):
    """Return `cons` with a `_tier_floor` from `metal` to each non-donor within `_FLOOR_REACH` bonds.

    The FF wall takes the heavy `NEAR` and `OUTER` atoms; the DG relief takes every atom in reach. Reach counts through
    the donors `real`, since the surrogate strips the metal's bonds.
    """
    topo = Chem.GetDistanceMatrix(mol) if topology is None else topology
    r_m = _PT.GetRcovalent(real_z)
    committed = {metal, *real}
    floors, dg_floors = dict(cons.floors), dict(cons.dg_floors)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in committed or a.GetAtomicNum() == 1:  # H is not what re-perception over-bonds
            continue
        hops = 1 + min(topo[d][i] for d in real)  # M -> donor -> ... -> X
        if hops > _FLOOR_REACH:
            continue
        tier = overbond_tier(mol, real, i)
        if tier == APEX:  # fixed by its two M-donor windows
            continue
        key = canonical_key((metal, i))
        floor = _tier_floor(a.GetAtomicNum(), tier, r_m, real_z)
        floors[key] = max(floors.get(key, 0.0), floor)

    # RDKit floors every M...X at the carbon surrogate's vdW contact, which forbids real second-sphere geometry.
    dg_reach = {i for d in real for i in np.flatnonzero(topo[d] <= _FLOOR_REACH - 1)}
    for i in map(int, dg_reach):
        if i in committed:
            continue
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        floor = _tier_floor(z, overbond_tier(mol, real, i), r_m, real_z)
        dg_floors[canonical_key((metal, i))] = floor
    return replace(cons, floors=floors, dg_floors=dg_floors)


def bridge_floors(donor_bonds, real_z, frozen=frozenset()):
    """Return `Constraints` flooring every two metals that share a donor at the `NEAR` tier.

    The stripped graph cannot reach one metal from the other and states no M-M bond, so nothing else holds them apart
    and UFF would fold the bridge shut.
    """
    sharing = defaultdict(set)
    for donor, metal in donor_bonds:
        sharing[donor].add(metal)
    floors = {
        (a, b): _tier_floor(real_z[b], NEAR, _PT.GetRcovalent(real_z[a]), real_z[a])
        for metals in sharing.values()
        for a, b in itertools.combinations(sorted(metals), 2)
        if not graft_owns((a, b), frozen)
    }
    return Constraints(floors=floors, dg_floors=floors)


_ALKALI = frozenset({3, 11, 19, 37, 55, 87})  # Li, Na, K, Rb, Cs, Fr
_ION_CONTACT = (-0.2, 0.1)  # Å around the covalent radius sum; a relaxed ion rests on the upper wall


def ionic_contacts(mol):
    """Return `Constraints` holding each alkali ion within contact of its dative partners, as a bond-less atom."""
    distances, partners = {}, defaultdict(list)
    for bond in mol.GetBonds():
        ions = [atom for atom in (bond.GetBeginAtom(), bond.GetEndAtom()) if atom.GetAtomicNum() in _ALKALI]
        if bond.GetBondType() != Chem.BondType.DATIVE or len(ions) != 1:
            continue
        ion, partner = ions[0], bond.GetOtherAtom(ions[0])
        r_sum = _PT.GetRcovalent(ion.GetAtomicNum()) + _PT.GetRcovalent(partner.GetAtomicNum())
        add_distance(distances, partner.GetIdx(), ion.GetIdx(), r_sum + _ION_CONTACT[0], r_sum + _ION_CONTACT[1])
        partners[ion.GetIdx()].append(partner.GetIdx())
    cons = Constraints(distances=distances)
    # fragments with the ions' own bonds cut, so lone partners are not one chelate's linked arms
    fragments = {atom: k for k, frag in enumerate(Chem.GetMolFrags(ligand_graph(mol, set(partners)))) for atom in frag}
    for ion, bound in partners.items():
        cons = ff_terms(mol, cons, ion, mol.GetAtomWithIdx(ion).GetAtomicNum(), bound, fragments=fragments)
    return cons
