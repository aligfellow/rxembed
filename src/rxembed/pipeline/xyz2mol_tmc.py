"""Module for the xyz2mol functionality for TMCs.

Modifications:
- ranked assignment (``_fast_bond_orders`` + ``_donor_localised_candidates`` + ``_lig_rank_key``)
- the invented-H gate (``_invented_hydrogens``)
- the canonical numbering boundary (``blind_canonical_order``, applied in ``get_lig_mol``)
- hydride-bridge and agostic reconnection
- ``_sanitized`` as the single sanitize, so every candidate is ranked on one rule

The atom order of the returned mol is the file's, not the canonical one.
Canonicalisation happens inside, and is mapped back before returning.
"""

import logging
from itertools import combinations
from math import comb

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import (
    GetPeriodicTable,
    rdchem,
    rdDetermineBonds,
    rdEHTTools,
    rdForceFieldHelpers,
    rdmolops,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from rxembed.metal_core import haptic_sites
from rxembed.utils import flat_ranks, lone_pair_electrons

from .xyz2mol_local import AC2mol, chiral_stereo_check, read_xyz_file, xyz2AC_obabel

# Narrower than metal_core.COORDINATION_METALS, which includes the f-block. The two are not
# interchangeable; changing this list changes which atoms are disconnected as metals.
# fmt: off
TRANSITION_METALS: list[str] = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "La", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]

TRANSITION_METALS_NUM: list[int] = [
    21, 22, 23, 24, 25, 26, 27, 57, 28, 29, 30, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
]
# fmt: on

_UNSANITIZABLE = 999  # sorts above any real implicit-H count, so such a candidate loses
_BOND_ORDER_MAX_ITERS = 10_000  # bound each native charge hypothesis; failure advances the existing ladder
_H, _B, _C = 1, 5, 6  # atomic numbers used by the reconnection and agostic checks
_PT = GetPeriodicTable()


def blind_canonical_order(mol: Chem.Mol, coordinating_atoms=()) -> list:
    """Return a canonical atom order that ignores Lewis form but retains the donor set.

    The order is given as order[new] = old, which is what Chem.RenumberAtoms takes. It is used to
    canonicalize the input to a bond order search, which returns the first valence consistent
    assignment it reaches and so depends on the numbering. Ranking the perceived molecule would be
    circular, so the ranks come from a copy with all bonds single, charges zeroed and aromatic
    flags cleared.
    """
    marked = Chem.Mol(mol)
    coordinating_atoms = {int(index) for index in coordinating_atoms}
    for atom in marked.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetIdx() in coordinating_atoms else 0)
    ranks = flat_ranks(marked, break_ties=True)
    order = [0] * len(ranks)
    for idx, rank in enumerate(ranks):
        order[rank] = idx
    return order


logger = logging.getLogger(__name__)

params = rdMolStandardize.MetalDisconnectorOptions()
# rdkit C-extension option attributes are untyped; mypy misreads the setter.
params.splitAromaticC = True  # type: ignore[assignment]
params.splitGrignards = True  # type: ignore[assignment]
params.adjustCharges = False  # type: ignore[assignment]

# Metal-nonmetal disconnection; extended to the full lanthanide series (#57-#71).
MetalNon_Hg = (
    "[#3,#11,#12,#19,#13,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,"
    "#42,#43,#44,#45,#46,#47,#48,#57,#58,#59,#60,#61,#62,#63,#64,#65,#66,#67,"
    "#68,#69,#70,#71,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    "~[B,#6,#14,#15,#33,#51,#16,#34,#52,Cl,Br,I,#85,#1;!$([#1]~[#6])]"
)
# Metal-N/O/F disconnection (RDKit's default MetalNof omits the lanthanides); set via SetMetalNof.
MetalNof_TM = (
    "[#3,#4,#11,#12,#13,#19,#20,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#31,"
    "#37,#38,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#49,#50,#55,#56,#57,#58,"
    "#59,#60,#61,#62,#63,#64,#65,#66,#67,#68,#69,#70,#71,#72,#73,#74,#75,#76,"
    "#77,#78,#79,#80,#81,#82,#83]~[#7,#8,#9]"
)


def _sanitized(mol):
    """Sanitize a copy of the molecule and return it.

    Raises ValueError if the perceived bond orders do not form a valid molecule. Kept as the single
    sanitization point, since _invented_hydrogens scores a candidate by what this does to it and a
    laxer sanitization elsewhere would rank candidates on different rules.
    """
    work = Chem.RWMol(mol) if isinstance(mol, Chem.RWMol) else Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception as exc:
        raise ValueError(f"perceived bond orders do not form a valid molecule: {exc}") from exc
    return work


def fix_NO2(mol):
    """Localize nitro groups mis-assigned a charge of -2.

    Such groups are a neutral nitrogen bound to two negatively charged oxygen
    atoms. They are changed to reflect the correct neutral configuration of a
    nitro group. The oxidation state on the transition metal is changed
    accordingly.
    """
    # Create RWMol if not already
    if isinstance(mol, Chem.RWMol):
        emol = mol
    else:
        emol = Chem.RWMol(mol)

    patt = Chem.MolFromSmarts(
        "[#8-]-[#7+0]-[#8-].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    )
    matches = emol.GetSubstructMatches(patt)
    for a1, a2, a3, a4 in matches:
        if not emol.GetBondBetweenAtoms(a1, a4) and not emol.GetBondBetweenAtoms(a3, a4):
            tm = emol.GetAtomWithIdx(a4)
            o1 = emol.GetAtomWithIdx(a1)
            n = emol.GetAtomWithIdx(a2)
            tm_charge = tm.GetFormalCharge()
            new_charge = tm_charge - 2
            tm.SetFormalCharge(new_charge)
            n.SetFormalCharge(+1)
            o1.SetFormalCharge(0)
            emol.RemoveBond(a1, a2)
            emol.AddBond(a1, a2, rdchem.BondType.DOUBLE)

    return _sanitized(emol)


def _localise_donor_pairs(mol, coordinating_atoms=None):
    """Move an existing delocalised negative charge onto the coordinating donor.

    ``coordinating_atoms`` supplies ligand-local donor indices during bond-order ranking. Once a metal is
    present, omitting it derives the same donor sets from the metal bonds.
    """
    if isinstance(mol, Chem.RWMol):
        emol = mol
    else:
        emol = Chem.RWMol(mol)

    patt = Chem.MolFromSmarts("[#6-,#7-,#8-,#15-,#16-]-[*]=[#6,#7,#8,#15,#16]")

    matches = emol.GetSubstructMatches(patt)
    donor_sets = (
        [set(map(int, coordinating_atoms))]
        if coordinating_atoms is not None
        else [
            {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
            for atom in emol.GetAtoms()
            if atom.GetAtomicNum() in TRANSITION_METALS_NUM
        ]
    )
    used_atom_ids_1 = []
    used_atom_ids_3 = []
    for donors in donor_sets:
        for a1, a2, a3 in matches:
            if a3 in donors and a1 not in donors and a1 not in used_atom_ids_1 and a3 not in used_atom_ids_3:
                used_atom_ids_1.append(a1)
                used_atom_ids_3.append(a3)

                emol.RemoveBond(a1, a2)
                emol.AddBond(a1, a2, Chem.rdchem.BondType.DOUBLE)
                emol.RemoveBond(a2, a3)
                emol.AddBond(a2, a3, Chem.rdchem.BondType.SINGLE)
                # Move one negative charge a1->a3 RELATIVE to a3's charge; an absolute set
                # mis-charges an already-cationic a3 (a coordinating phosphonium P+ -> neutral
                # phosphine donor, not P- + a phantom H).
                at1, at3 = emol.GetAtomWithIdx(a1), emol.GetAtomWithIdx(a3)
                at1.SetFormalCharge(at1.GetFormalCharge() + 1)
                at3.SetFormalCharge(at3.GetFormalCharge() - 1)

    # This is usually the first full sanitize to touch the assembled TMC, so an
    # unkekulizable ring perceived upstream by AC2mol surfaces here even when this
    # function rewrote nothing.
    return _sanitized(emol)


def get_proposed_ligand_charge(ligand_mol, cutoff=-10):
    """Run an extended Hückel calculation for the ligand in ligand_mol.

    A suggested charge is found by filling electrons in orbitals <-10eV and
    comparing with total number of valence electrons. If charge is >= 1 (<-1)
    and the LUMO (HOMO) is low (high) in energy, two additional electrons are
    added (removed). The suggested charge is returned.
    """
    valence_electrons = sum(_PT.GetNOuterElecs(a.GetAtomicNum()) for a in ligand_mol.GetAtoms())

    passed, result = rdEHTTools.RunMol(ligand_mol)
    if not passed:
        raise ValueError("RDKit extended-Hueckel charge estimation failed for this ligand")
    N_occ_orbs = sum(1 for i in result.GetOrbitalEnergies() if i < cutoff)
    charge = valence_electrons - 2 * N_occ_orbs
    percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]
    if N_occ_orbs == len(result.GetOrbitalEnergies()):
        percieved_lumo = np.nan
    else:
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge >= 1 and percieved_lumo < -9:
        N_occ_orbs += 1
        charge += -2
        logger.debug("added two more electrons: charge %s, LUMO %s", charge, percieved_lumo)
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge < -1 and percieved_homo > -10.2:
        N_occ_orbs -= 1
        charge += 2
        logger.debug("removed two electrons: charge %s, HOMO %s", charge, percieved_homo)
        percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]

    return charge


# A tetra-coordinate N or B, or a tri-coordinate O, reads no other way than a formal charge; a period-2
# "fewer than four valence electrons, more sigma bonds than that" formula would also charge a five-bonded
# TS carbon, so this stays a literal table. Shared with perceive._flatten_for_xyz2mol, which re-applies it after
# clearing charges for its own bond-order search.
SEEDED_STRUCTURAL_CHARGES = {(7, 4): 1, (8, 3): 1, (5, 4): -1}


def get_basic_mol(xyz_file, overall_charge):
    """Build a basic mol object for an extended Hückel calculation.

    The object is constructed from the adjacency matrix evaluated from the
    xyz-coordinates. All bonds are single bonds, and charges are only assigned
    if necessary to work with it, i.e. a nitrogen with four neighbors gets a
    +1 charge, boron with 4 neighbors gets a -1 charge and oxygen with three
    neighbors gets a +1 charge. Hydrogen atoms come only from the XYZ records.
    """
    atoms, _, xyz_coords = read_xyz_file(xyz_file)

    # AC, mol = xyz2AC_huckel(atoms, xyz_coords, overall_charge)
    AC, mol = xyz2AC_obabel(atoms, xyz_coords, tolerance=0.5)  # Modified tolerance to capture haptic bonds
    tm_indxs = [atoms.index(tm) for tm in TRANSITION_METALS_NUM if tm in atoms]

    rwMol = Chem.RWMol(mol)
    length_ac = len(AC)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for i in range(length_ac):
        for j in range(i + 1, length_ac):
            bo = int(round(AC[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    for i, a in enumerate(mol.GetAtoms()):
        a.SetNoImplicit(True)
        explicit_valence = sum(ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs)
        a.SetFormalCharge(SEEDED_STRUCTURAL_CHARGES.get((a.GetAtomicNum(), explicit_valence), 0))

    return mol, xyz_coords


def _invented_hydrogens(res_mol) -> int:
    """Count hydrogens encoded on heavy atoms but absent from the XYZ graph.

    The xyz file carries every hydrogen explicitly, so a correct perception gains none. A form that
    only kekulizes by de-aromatizing a ring, such as a phosphonium ylide with a pentavalent ipso
    carbon, gains one for each carbon demoted to sp3. Both implicit H and bracket-style explicit-H
    counts are virtual here; real XYZ hydrogens are separate neighbour atoms. A candidate that cannot
    be sanitized scores _UNSANITIZABLE, which is higher than any real count and loses the ranking.
    """
    try:
        k = _sanitized(Chem.Mol(res_mol))
        return sum(a.GetTotalNumHs() for a in k.GetAtoms())
    except Exception:
        return _UNSANITIZABLE


def _lig_rank_key(cand):
    """Sort key for a candidate from lig_checks.

    The candidate is (mol, n_pos, n_neg, n_aromatic, invented_H, pairless_sigma_donors). Candidates are
    ordered by fewest invented hydrogens, radicals and pairless sigma donors, then most aromatic atoms,
    fewest formal charges away from donors, lowest charge concentration, lowest total ligand charge
    magnitude and canonical SMILES.

    A fully delocalised charge-separated ring (every atom +/-1) ties the concentration field at any odd
    total charge, since that field sums individual atom charges squared and every atom already carries
    the same magnitude: an eta7-C7H7 ring reads identically whether the ring holds a net -3 or a net -7.
    The ligand's net charge is the sum of every atom's formal charge (a molecular invariant), so its
    magnitude is read off the candidate mol directly rather than threaded through as a parameter.

    This field sits below concentration and above SMILES, but _fast_bond_orders' cross-charge ranking
    excludes it from the quality prefix it compares: a charge the extended-Hueckel hint picks out
    uniquely still wins even when a less charge-separated alternative also fits the graph (see there).
    Magnitude only breaks a tie the hint itself cannot, favouring the less charge-separated candidate.
    """
    mol = cand[0]
    return (
        cand[4],
        sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
        cand[5],
        -cand[3],
        cand[1] + cand[2],
        sum(atom.GetFormalCharge() ** 2 for atom in mol.GetAtoms()),
        abs(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        _canonical_smiles(mol),
    )


def _canonical_smiles(res_mol) -> str:
    """Canonical SMILES for a resonance form.

    A form that cannot be written returns a character that sorts after any SMILES, so it loses a
    tie in _lig_rank_key rather than winning one.
    """
    try:
        return Chem.MolToSmiles(res_mol)
    except Exception:
        return "￿"


def _uff_bond_length_rms(mol):
    """Return the RMS residual from observed bonds to RDKit's UFF equilibrium lengths."""
    if not mol.GetNumConformers():
        return None
    pos = mol.GetConformer().GetPositions()
    residuals = []
    with rdBase.BlockLogs():
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            params = rdForceFieldHelpers.GetUFFBondStretchParams(mol, i, j)
            if params is None:
                return None
            residuals.append((float(np.linalg.norm(pos[i] - pos[j])) - params[1]) ** 2)
    return float(np.sqrt(np.mean(residuals))) if residuals else 0.0


def lig_checks(lig_mol, coordinating_atoms, resonate=True):
    """Yield ``(resonance form, ranking stats)`` for each candidate resonance form of a proposed ligand.

    The stats (positive/negative charge counts, aromatic atom count, invented hydrogens, pairless sigma
    donors) let the caller pick the resonance form whose coordinating atoms carry the fewest bad partial
    charges -- a positive charge on a donor atom is the red flag a correct Lewis structure should avoid.

    Enumeration runs on a blind-canonical atom numbering, not the caller's: ResonanceMolSupplier's order
    and pool size depend on input numbering (measured on RERHEB: 16 candidates one order, 17 shuffled, the
    shuffle reaching a strictly better form the original never enumerates). Since the picker below is
    stable, canonicalising first makes the winner invariant too -- shuffling the un-canonicalised input
    flipped a metal's oxidation state ([Hf+2] <-> [Hf+4]), migrated charges, and flipped stereo on 31 of
    142 fixtures. The chosen form is mapped back to the caller's own numbering before it is yielded.
    """
    order = blind_canonical_order(lig_mol, coordinating_atoms)  # order[new] = old
    new_of = {old: new for new, old in enumerate(order)}
    canon = Chem.RenumberAtoms(lig_mol, order)
    coord_canon = {new_of[int(a)] for a in coordinating_atoms if int(a) in new_of}
    back = [new_of[i] for i in range(lig_mol.GetNumAtoms())]  # canon -> original numbering

    # _donor_localised_candidates has already placed the charges, so it passes resonate=False. The
    # enumeration is also combinatorial on a large porphyrin, where each meso substituent is its own
    # conjugated group and only the core resonance matters.
    def candidates():
        if not resonate:
            yield canon
            return
        for flags in (0, Chem.ALLOW_INCOMPLETE_OCTETS):
            yielded = False
            for candidate in rdchem.ResonanceMolSupplier(canon, flags=flags):
                if candidate is not None:
                    yielded = True
                    yield candidate
            if yielded:
                return
        yield canon

    for res_mol in candidates():
        # A formal +/- on two bonded atoms is a legitimate charge-separated motif (a nitro N+-O-,
        # an N-oxide, an azide): net-neutral, not a mis-perceived charge. Pair each + with an
        # adjacent - and exclude both from the "bad charge" tally, so a clean solution can carry it
        # (else a nitro forces the fast compiled path to fall to the combinatorial AC2mol search).
        paired: set = set()
        for a in res_mol.GetAtoms():
            if a.GetFormalCharge() > 0 and a.GetIdx() not in paired:
                for nb in a.GetNeighbors():
                    if nb.GetFormalCharge() < 0 and nb.GetIdx() not in paired:
                        paired.update((a.GetIdx(), nb.GetIdx()))
                        break
        positive_atoms = []
        negative_atoms = []
        N_aromatic = 0
        for a in res_mol.GetAtoms():
            if a.GetIsAromatic():
                N_aromatic += 1
            if a.GetIdx() in paired:
                continue
            if a.GetFormalCharge() > 0:
                positive_atoms.append(a.GetIdx())
            if a.GetFormalCharge() < 0 and a.GetIdx() not in coord_canon:
                negative_atoms.append(a.GetIdx())

        # Lewis bookkeeping (pairless-donor count below): read only the old pi-seeded face rule, not the site
        # pair rule, so grouping a bonded donor pair into one site never changes this reader's charge decision.
        haptic = {atom for site in haptic_sites(res_mol, coord_canon, pairs=False) if len(site) > 1 for atom in site}
        # A sigma site needs a lone pair; multicentre sites remain ranked fallbacks above (no metal bond
        # on this ligand fragment, so lone_pair_electrons' metal exclusion is a no-op here).
        pairless = sum(lone_pair_electrons(res_mol.GetAtomWithIdx(index), ()) < 2 for index in coord_canon - haptic)

        # back to the caller's numbering (the enumeration ran on the blind-canonical one)
        yield (
            Chem.RenumberAtoms(res_mol, back),
            len(positive_atoms),
            len(negative_atoms),
            N_aromatic,
            _invented_hydrogens(res_mol),
            pairless,
        )


#: Cap on the donor subsets tried by _donor_localised_candidates. C(n, k) grows fast and every
#: subset costs a DetermineBondOrders call; a real ligand needs a handful (a porphyrin is C(4,2)=6).
#: If a ligand exceeds this we do not silently truncate to an arbitrary subset; we skip the
#: generator entirely and fall back to the blind search, so the behaviour stays explainable.
_DONOR_LOCALISED_MAX = 20


def _donor_localised_candidates(mol, charge, coordinating_atoms):
    """Generate candidates for an anionic ligand with the charge placed on the coordinating atoms.

    The bond order search returns the first assignment that satisfies every valence, which for a
    delocalised ligand is often chemically wrong: on a metal porphyrin it places carbocations and
    carbanions around the macrocycle, and the resonance supplier cannot reach the porphyrinato
    dianion from there. Since the ligand charge and its coordinating atoms are both known, the
    charge is instead seeded on the donors, -1 on each of |charge| of them, with implicit hydrogens
    disabled so the valences are closed with bonds. All subsets are returned as candidates, and
    lig_checks and _lig_rank_key choose between them.
    """
    donors = sorted(int(a) for a in coordinating_atoms)
    k = -int(charge)
    if k <= 0 or k > len(donors):  # neutral or cationic ligand: no anion to place
        return []
    if comb(len(donors), k) > _DONOR_LOCALISED_MAX:
        return []  # explainable fallback, never a silent truncation to an arbitrary subset

    out = []
    for subset in combinations(donors, k):
        work = Chem.RWMol(mol)
        for a in work.GetAtoms():
            a.SetNoImplicit(True)  # the H count is given by the xyz; do not invent one
            a.SetNumExplicitHs(0)
        for i in subset:
            work.GetAtomWithIdx(i).SetFormalCharge(-1)
        try:
            rdDetermineBonds.DetermineBondOrders(
                work,
                charge=int(charge),
                embedChiral=False,
                maxIterations=_BOND_ORDER_MAX_ITERS,
            )
            cand = _localise_donor_pairs(work.GetMol(), coordinating_atoms)
            # resonate=False: the charge placement is the seed's, so no resonance search is needed.
            out.extend(lig_checks(cand, coordinating_atoms, resonate=False))
        except Exception:
            continue
    return out


def _fast_bond_orders(mol, charge, coordinating_atoms, return_pool=False):
    """Perceive bond orders with RDKit's compiled implementation.

    rdDetermineBonds.DetermineBondOrders runs the same algorithm as AC2mol in C++ and handles most
    ligand fragments; the wider valence ligands it declines, such as dithiolenes and imidos, fall
    through to AC2mol. Returns (mol, charge), or None when nothing usable is found.

    Solutions from a graph-derived charge ladder are collected and ranked rather than returning the first
    clean one, since a clean solution is not necessarily the right one: a metal porphyrin is clean both as a
    neutral macrocycle and as a tetra-anion. ``return_pool`` skips the single-winner selection below (the
    UFF bond-length vote, the Hueckel-charge tie-break, the raise on an unresolved tie) and instead returns
    every charge that reached any invented-H-free candidate (not only ``strict``'s clean-donor tier) as a
    ``[(mol, charge), ...]`` list ranked by ``_lig_rank_key`` alone, for a caller that needs an alternative
    to the winner rather than the winner itself.
    """
    # The molecule is already in the canonical order set by get_lig_mol. Renumbering it again here
    # would let this path and the AC2mol fallback disagree.
    valence = sum(_PT.GetNOuterElecs(atom.GetAtomicNum()) for atom in mol.GetAtoms())
    limit = max(4, len(coordinating_atoms))
    charges = [q for q in range(-limit, limit + 1) if (valence - q) % 2 == 0]
    charges.sort(key=lambda q: (abs(q - charge), abs(q), q))

    native = []
    strict, relaxed = {}, {}

    def consider(candidate, candidate_charge):
        if candidate[4] != 0:
            return
        if candidate_charge not in relaxed or _lig_rank_key(candidate) < _lig_rank_key(relaxed[candidate_charge]):
            relaxed[candidate_charge] = candidate
        if (
            candidate[1] + candidate[2] == 0
            and candidate[5] == 0
            and (candidate_charge not in strict or _lig_rank_key(candidate) < _lig_rank_key(strict[candidate_charge]))
        ):
            strict[candidate_charge] = candidate

    for c in charges:
        try:
            work = Chem.RWMol(mol)
            rdDetermineBonds.DetermineBondOrders(
                work,
                charge=int(c),
                embedChiral=False,
                maxIterations=_BOND_ORDER_MAX_ITERS,
            )
            cand = _localise_donor_pairs(work.GetMol(), coordinating_atoms)
            possible = tuple(lig_checks(cand, coordinating_atoms, resonate=False))
        except Exception:
            continue
        native.append((c, cand))
        for candidate in possible:
            consider(candidate, c)

    # ...and the candidates the blind search cannot reach.
    for c in charges:
        if c < 0:
            for candidate in _donor_localised_candidates(mol, c, coordinating_atoms):
                consider(candidate, c)

    if not strict:
        # A later charge-ladder candidate can be clean even when the first is not. Search the complete native
        # ladder before expanding any resonance pool: on a macrocycle that expansion is combinatorial, while a
        # clean blind-canonical assignment already meets the acceptance gate and is deterministic.
        for c, cand in native:
            try:
                for candidate in lig_checks(cand, coordinating_atoms):
                    consider(candidate, c)
            except Exception:
                continue
    if return_pool:
        # Every charge with any invH==0 candidate, not just strict's clean-donor tier: a rescue search
        # wants the full breadth the ladder reached, not only the quality bar the normal winner clears.
        ranked = sorted(relaxed.items(), key=lambda item: _lig_rank_key(item[1]))
        return [(candidate[0], q) for q, candidate in ranked]
    pool = strict or relaxed
    if not pool:
        return None

    # Canonical SMILES orders resonance forms inside one fixed charge. It cannot decide oxidation state,
    # and neither can charge magnitude below: both sit outside the quality prefix here, so a charge the
    # Hueckel hint picks out uniquely (below) still wins over a less charge-separated alternative.
    ranked = [(candidate, q, _lig_rank_key(candidate)) for q, candidate in pool.items()]
    best_quality = min(key[:-2] for _candidate, _q, key in ranked)
    tied = [(candidate, q, key) for candidate, q, key in ranked if key[:-2] == best_quality]
    best = min(tied, key=lambda item: item[2])

    # Formal-charge concentration is a representation score, not a measurement. Let the proposed electronic
    # charge override it only when RDKit's independent bond-length model picks the same charge from candidates
    # that passed every harder graph gate. Missing UFF parameters make this comparison abstain.
    prefix = min(key[:4] for _candidate, _q, key in ranked)
    comparable = [(candidate, q) for candidate, q, key in ranked if key[:4] == prefix]
    scored = [(_uff_bond_length_rms(candidate[0]), candidate, q) for candidate, q in comparable]
    geometry_decided = False
    if scored and all(score is not None for score, _candidate, _q in scored):
        ordered = sorted(scored, key=lambda item: item[0])
        if ordered[0][2] == charge and (len(ordered) == 1 or ordered[0][0] < ordered[1][0] - 1e-9):
            best = (ordered[0][1], ordered[0][2], _lig_rank_key(ordered[0][1]))
            geometry_decided = True

    ambiguous = sorted(q for _candidate, q, _key in tied)
    if len(ambiguous) > 1 and not geometry_decided:
        hinted = [item for item in tied if item[1] == charge]
        if len(hinted) == 1:
            best = hinted[0]
        else:
            # The hint itself does not pick one out (it is off the tied ladder, as an eta7-C7H7 ring's
            # q=-1 pi-count hint is for its q=-3/q=-7 ionic candidates). Fall back to the least
            # charge-separated tied candidate; only raise when two charges also share that magnitude.
            least_charge = min(key[-2] for _candidate, _q, key in tied)
            by_magnitude = [item for item in tied if item[2][-2] == least_charge]
            if len(by_magnitude) != 1:
                raise ValueError(
                    f"bond-order perception: ligand charges {ambiguous} are graph-equivalent but "
                    f"extended Hückel proposed q={charge:+d}; use an explicit charged/dative SMILES"
                )
            best = by_magnitude[0]
    return best[0][0], best[1]


def get_lig_mol(mol, charge, coordinating_atoms):
    """Build a sanitizable ligand mol, ranked by the `lig_checks` criteria.

    The charge/carbene ladder runs only when `_fast_bond_orders` declines, ranking its candidates by
    `_lig_rank_key` rather than keeping the first good one. Atoms are renumbered to `blind_canonical_order`
    before perception and mapped back after: a bond-order search returns the first valence-consistent
    assignment it reaches, so its result depends on numbering (the same imidazolium places [n+] on either
    nitrogen, and this cannot be repaired downstream since the two numberings perceive different molecules,
    not different renderings of one). The returned mol keeps the caller's original numbering.
    """
    if mol.GetNumAtoms() == 1:  # no bond order exists to solve for a hydride, halide, oxo, or nitrido ligand
        out = Chem.Mol(mol)
        atom = out.GetAtomWithIdx(0)
        atom.SetFormalCharge(int(charge))
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(True)
        out.UpdatePropertyCache(strict=False)
        return out, charge

    order = blind_canonical_order(mol, coordinating_atoms)  # order[new] = old
    new_of = {old: new for new, old in enumerate(order)}
    canon = Chem.RenumberAtoms(mol, order)
    coord = [new_of[int(a)] for a in coordinating_atoms if int(a) in new_of]
    back = [new_of[i] for i in range(mol.GetNumAtoms())]  # canon -> the caller's numbering
    atoms = [a.GetAtomicNum() for a in canon.GetAtoms()]
    AC = Chem.rdmolops.GetAdjacencyMatrix(canon)

    def ladder():
        """Jensen's charge/carbene ladder over the canonical numbering; returns (mol, charge)."""
        # Fast path: RDKit-native bond perception, taken only when it is a clean solution; it has
        # already searched the charge itself, so there is no ladder left to run.
        fast = _fast_bond_orders(canon, charge, coord)
        if fast is not None:
            return fast

        q = charge
        lig_mol = AC2mol(canon, AC, atoms, q, allow_charged_fragments=True, use_atom_maps=False)
        if not lig_mol and q >= 0:
            q += -2
            lig_mol = AC2mol(canon, AC, atoms, q, allow_charged_fragments=True, use_atom_maps=False)
            if not lig_mol:
                return None, q
        if not lig_mol and q < 0:
            q += 2
            lig_mol = AC2mol(canon, AC, atoms, q, allow_charged_fragments=True, use_atom_maps=False)
            if not lig_mol:
                q += -4
                lig_mol = AC2mol(canon, AC, atoms, q, allow_charged_fragments=True, use_atom_maps=False)
                if not lig_mol:
                    return None, q

        # A clean candidate has no invented H, pairless sigma donor or stray charge. Keep pairless
        # multicentre donors as ranked fallbacks, but do not let one stop the existing charge search.
        best = min(lig_checks(lig_mol, coord), key=_lig_rank_key)
        if best[4] == best[5] == 0 and best[1] + best[2] == 0:
            return best[0], q

        no_carbene = AC2mol(
            canon,
            AC,
            atoms,
            q,
            allow_charged_fragments=True,
            use_atom_maps=False,
            allow_carbenes=False,
        )
        allow_carbenes = True
        if no_carbene:
            nc = min(lig_checks(no_carbene, coord), key=_lig_rank_key)
            if _lig_rank_key(nc) < _lig_rank_key(best):
                best, allow_carbenes = nc, False
        if best[4] == best[5] == 0 and best[1] + best[2] == 0:
            logger.debug("found opt solution without carbenes")
            return best[0], q

        new_charge = q + 2 if best[1] - best[2] + q < 0 else q - 2
        retry = AC2mol(
            canon,
            AC,
            atoms,
            new_charge,
            allow_charged_fragments=True,
            use_atom_maps=False,
            allow_carbenes=allow_carbenes,
        )
        if not retry:
            return best[0], q
        nc = min(lig_checks(retry, coord), key=_lig_rank_key)
        if _lig_rank_key(nc) < _lig_rank_key(best):
            best, q = nc, new_charge
        return best[0], q

    lig_mol, final_charge = ladder()
    if lig_mol is None:
        return None, final_charge
    return Chem.RenumberAtoms(lig_mol, back), final_charge


def _ligand_charge_pool(mol, charge, coordinating_atoms):
    """Return every ligand-charge candidate the fast native search finds for `mol`, ranked best first.

    Mirrors get_lig_mol's canonicalize / search / map-back steps but returns the whole pool instead of
    narrowing to one winner, for a caller that needs an alternative to get_lig_mol's own choice. A
    single-atom ligand and one the fast path declines (dithiolenes, imidos: get_lig_mol's AC2mol ladder)
    have no pool and return no candidates; only the fast native search exposes one.
    """
    if mol.GetNumAtoms() == 1:
        return []
    order = blind_canonical_order(mol, coordinating_atoms)  # order[new] = old
    new_of = {old: new for new, old in enumerate(order)}
    canon = Chem.RenumberAtoms(mol, order)
    coord = [new_of[int(a)] for a in coordinating_atoms if int(a) in new_of]
    back = [new_of[i] for i in range(mol.GetNumAtoms())]  # canon -> the caller's numbering
    pool = _fast_bond_orders(canon, charge, coord, return_pool=True)
    return [(Chem.RenumberAtoms(candidate, back), q) for candidate, q in pool]


def _rescue_over_cap_ligand_charge(lig_sources, total_lig_charge, overall_charge, limit):
    """Find one ligand substitution that brings the metal charge within its valence-electron cap.

    Each ligand fragment is resolved to its own best charge independently of what that implies for the
    metal, so the sum can push the metal over its cap when a different, ranked-but-not-best candidate for
    one ligand would have kept it under. Try each ligand's alternative charges, best-ranked first, in
    place of its current pick, holding every other ligand fixed; return the first substitution (index,
    ligand mol, ligand charge, new ligand-charge total) that satisfies the cap, or None if no single
    substitution does. Ties across ligands favour fragment order, since only one ligand carries the
    ambiguity in every case measured (see the row-C follow-up report).
    """
    for index, (m, coord, current_charge) in enumerate(lig_sources):
        for candidate_mol, candidate_charge in _ligand_charge_pool(m, current_charge, coord):
            if candidate_charge == current_charge:
                continue
            new_total = total_lig_charge - current_charge + candidate_charge
            if overall_charge - new_total <= limit:
                return index, candidate_mol, candidate_charge, new_total
    return None


# A kappa-H metal borohydride's bridging B-H is elongated (~1.6 A) vs a terminal B-H (~1.18), so the
# covalent-radius connectivity in get_basic_mol drops it: B is left a free BH3 fragment and the
# bridge H a lone metal-hydride, so the boron floats off in the 3D reconstruction (Y-B 6.6 vs real
# 3.4 A). Re-form the B-H (making a BH4- unit) up to this length.
_BRIDGE_BH_MAX = 1.85  # A; longest bridging B-H to reconnect (terminal ~1.18, bridge ~1.6-1.7)
_AGOSTIC_CH_MAX = 2.4  # A; longest agostic C-H...M contact to capture (opt-in, see `agostic=`)


def _reconnect_metal_hydride_bridges(mol, xyz_coords):
    """Join each metal-bound H to its nearest under-coordinated B within the bridge cutoff."""
    coords = np.asarray(xyz_coords)
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM]
    borons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == _B]
    if not metals or not borons:
        return mol
    capacity = {
        boron: 4
        - sum(
            neighbor.GetAtomicNum() not in TRANSITION_METALS_NUM
            for neighbor in mol.GetAtomWithIdx(boron).GetNeighbors()
        )
        for boron in borons
    }
    candidates = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        hydrogen = atom.GetIdx()
        if not any(mol.GetBondBetweenAtoms(hydrogen, metal) is not None for metal in metals):
            continue
        for boron in borons:
            distance = float(np.linalg.norm(coords[hydrogen] - coords[boron]))
            if capacity[boron] > 0 and mol.GetBondBetweenAtoms(hydrogen, boron) is None and distance <= _BRIDGE_BH_MAX:
                candidates.append((distance, hydrogen, boron))
    rw = Chem.RWMol(mol)
    assigned = set()
    for _distance, hydrogen, boron in sorted(candidates):
        if hydrogen in assigned or capacity[boron] <= 0:
            continue
        rw.AddBond(hydrogen, boron, Chem.BondType.SINGLE)
        capacity[boron] -= 1
        assigned.add(hydrogen)
        atom = rw.GetAtomWithIdx(boron)
        if atom.GetFormalCharge() == 0 and capacity[boron] == 0:
            atom.SetFormalCharge(-1)
    return rw.GetMol() if assigned else mol


def _agostic_ch_metal_pairs(mol, xyz_coords, capture=False):
    """Return (hydrogen, metal) index pairs for agostic C-H contacts, when capture is set.

    An agostic C-H points its hydrogen at the metal without being a metal hydride. The dative bond
    cannot be added here: hydrogen matches neither the MetalNon nor the MetalNof pattern, so
    MetalDisconnector would not cut it and the ligand would stay fused to the metal fragment and
    skip per-fragment perception. The contacts are only recorded, and get_tmc_mol adds the datives
    once the ligands are perceived and reassembled. Indices are those before disconnection.
    """
    if not capture:
        return []
    coords = np.asarray(xyz_coords)
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM]
    pairs = []
    for h in mol.GetAtoms():
        if h.GetAtomicNum() != _H or not any(n.GetAtomicNum() == _C for n in h.GetNeighbors()):
            continue  # only a C-H
        hi = h.GetIdx()
        if any(mol.GetBondBetweenAtoms(hi, m) is not None for m in metals):
            continue  # a real metal-hydride, not agostic
        for m in metals:
            if float(np.linalg.norm(coords[hi] - coords[m])) <= _AGOSTIC_CH_MAX:
                pairs.append((hi, m))
    return pairs


def get_tmc_mol(xyz_file, overall_charge, with_stereo=False, agostic=False, graph=None):
    """Perceive bond orders and charges for a transition-metal complex from ``xyz_file`` or a given ``graph``.

    ``agostic`` also writes a dative H->M for a C-H pointing at the metal within ``_AGOSTIC_CH_MAX``; off by
    default, since an agostic contact is real but not a bond every consumer expects. ``graph`` is an
    ``(mol, xyz_coords)`` pair to use in place of perceiving connectivity from the file: the bonds are taken
    as given and only their orders and formal charges are assigned, so it is expected to already carry
    ``get_basic_mol``'s seeded structural charges (N with four neighbours, B with four, O with three).
    A metal charge above its valence electron count is an impossible oxidation state; a single-metal read
    first tries a ranked-but-not-best charge for one ligand before raising (`_rescue_over_cap_ligand_charge`).
    """
    mol, xyz_coords = graph if graph is not None else get_basic_mol(xyz_file, overall_charge)
    if graph is None:
        mol = _reconnect_metal_hydride_bridges(mol, xyz_coords)  # xyz2mol's connectivity omitted an elongated B-H
    # Capture agostic C-H...M contacts by index now, but add the datives only after perception
    # (see _agostic_ch_metal_pairs) -- here idx == __origIdx, set in the loop just below.
    agostic_pairs = _agostic_ch_metal_pairs(mol, xyz_coords, agostic)

    tmc_indices = []
    for a in mol.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            tmc_indices.append(a.GetIdx())

    if not tmc_indices:
        raise ValueError("Found no TM in the input file. Please supply an xyz file with a TM")
    if graph is None and len(tmc_indices) > 1:
        raise ValueError(
            "xyz2mol cannot allocate oxidation states between multiple metals from coordinates alone; "
            "supply an already perceived graph or use read_xyz(..., metal_charges={atom_index: charge, ...})"
        )

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
    coordinating_atoms = {
        metal: set(map(int, np.nonzero(adjacency[metal, :])[0])) - set(tmc_indices) for metal in tmc_indices
    }
    all_coordinating_atoms = set().union(*coordinating_atoms.values())

    mdis = rdMolStandardize.MetalDisconnector(params)
    mdis.SetMetalNon(Chem.MolFromSmarts(MetalNon_Hg))
    mdis.SetMetalNof(Chem.MolFromSmarts(MetalNof_TM))
    frags = mdis.Disconnect(mol)
    # Bond orders and charges are not assigned yet. Sanitize completed candidates, not the single-bond graph.
    frag_mols = rdmolops.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

    total_lig_charge = 0
    metal_frags = []
    lig_list = []
    lig_sources = []  # (fragment, coordinating atoms, chosen charge), aligned with lig_list; rescue input
    for i, f in enumerate(frag_mols):
        m = Chem.Mol(f)
        atoms = m.GetAtoms()
        if any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in atoms):
            metal_frags.append(m)
            continue
        lig_charge = get_proposed_ligand_charge(f)

        lig_coordinating_atoms = [
            a.GetIdx() for a in m.GetAtoms() if a.GetIntProp("__origIdx") in all_coordinating_atoms
        ]
        lig_mol, lig_charge = get_lig_mol(m, lig_charge, lig_coordinating_atoms)
        if not lig_mol:
            try:
                frag_smiles = Chem.MolToSmiles(m)
            except Exception:
                frag_smiles = f"<{m.GetNumAtoms()} atoms>"
            raise ValueError(
                f"get_lig_mol failed for ligand fragment #{i} (SMILES: {frag_smiles!r}); cannot build TMC mol"
            )

        if lig_mol.GetNumAtoms() == m.GetNumAtoms():
            for a_lig, a_orig in zip(lig_mol.GetAtoms(), m.GetAtoms()):
                if a_orig.HasProp("__origIdx"):
                    a_lig.SetIntProp("__origIdx", a_orig.GetIntProp("__origIdx"))

        total_lig_charge += lig_charge
        lig_list.append(lig_mol)
        lig_sources.append((m, lig_coordinating_atoms, lig_charge))

    if not metal_frags:
        raise ValueError("Found no TM in the input file. Please supply an xyz file with a TM")

    tm = Chem.Mol(metal_frags[0])
    for fragment in metal_frags[1:]:
        tm = Chem.CombineMols(tm, fragment)
    tm = Chem.RWMol(tm)
    metals = [a for a in tm.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM]
    if len(tmc_indices) == 1:
        metals[0].SetFormalCharge(overall_charge - total_lig_charge)
        limit = _PT.GetNOuterElecs(metals[0].GetAtomicNum())
        if metals[0].GetFormalCharge() > limit:
            # Each ligand was resolved to its own best charge without regard to what that implies for the
            # metal; before refusing, see whether a ranked-but-not-best candidate for one ligand brings
            # the metal back under its cap.
            rescue = _rescue_over_cap_ligand_charge(lig_sources, total_lig_charge, overall_charge, limit)
            if rescue is not None:
                index, candidate_mol, _candidate_charge, total_lig_charge = rescue
                source = lig_sources[index][0]
                if candidate_mol.GetNumAtoms() == source.GetNumAtoms():
                    for a_lig, a_orig in zip(candidate_mol.GetAtoms(), source.GetAtoms()):
                        if a_orig.HasProp("__origIdx"):
                            a_lig.SetIntProp("__origIdx", a_orig.GetIntProp("__origIdx"))
                lig_list[index] = candidate_mol
                metals[0].SetFormalCharge(overall_charge - total_lig_charge)
    else:
        perceived = total_lig_charge + sum(a.GetFormalCharge() for a in metals)
        if perceived != overall_charge:
            raise ValueError(
                "xyz2mol cannot allocate oxidation states between multiple metals: supplied metal charges and "
                f"perceived ligands total {perceived}, requested {overall_charge}; supply a charge-consistent "
                "graph or use read_xyz(..., metal_charges={atom_index: charge, ...})"
            )

    for metal in metals:
        limit = _PT.GetNOuterElecs(metal.GetAtomicNum())
        if metal.GetFormalCharge() > limit:
            symbol = metal.GetSymbol()
            raise ValueError(
                f"{symbol}+{metal.GetFormalCharge()} is an impossible oxidation state ({symbol} has only "
                f"{limit} valence electrons); pass charge= or metal_charges= to read_xyz with the intended split"
            )

    for lmol in lig_list:
        tm = Chem.CombineMols(tm, lmol)

    emol = Chem.RWMol(tm)
    orig_to_emol = {a.GetIntProp("__origIdx"): a.GetIdx() for a in emol.GetAtoms()}
    for metal, donors in coordinating_atoms.items():
        mi = orig_to_emol[metal]
        for donor in donors:
            di = orig_to_emol[donor]
            if emol.GetBondBetweenAtoms(di, mi) is None:
                emol.AddBond(di, mi, Chem.BondType.DATIVE)

    # Agostic C-H...M: now that every ligand is perceived and reassembled, reconnect the recorded
    # H->M datives (begin=H donor). Doing it here, not before disconnection, is what keeps the
    # ligand a normal fragment through bond-order perception (see _agostic_ch_metal_pairs).
    for h_orig, m_orig in agostic_pairs:
        hi, mi = orig_to_emol.get(h_orig), orig_to_emol.get(m_orig)
        if hi is not None and mi is not None and emol.GetBondBetweenAtoms(hi, mi) is None:
            emol.AddBond(hi, mi, Chem.BondType.DATIVE)

    # Fix specific cases
    # Operate on emol directly to preserve properties
    emol = _localise_donor_pairs(emol)
    emol = fix_NO2(emol)

    tmc_mol = _sanitized(emol.GetMol())
    # Atom-count invariant: the source XYZ has every hydrogen explicit, so a correct perception
    # invents none. Any H encoded on a heavy atom means it was mis-typed (an aromatic C read as sp3, a ligand
    # that skipped bond-order perception) and the formula no longer matches the real structure --
    # fail loudly rather than ship a byte-wrong descriptor.
    invented = sum(a.GetTotalNumHs() for a in tmc_mol.GetAtoms())
    if invented:
        raise ValueError(
            f"perception invented {invented} H (formula no longer matches the XYZ); "
            "a coordinated ligand was likely mis-typed / skipped bond-order perception"
        )
    if with_stereo:
        chiral_stereo_check(tmc_mol)
    # Disconnecting the metal and reassembling the fragments reorders the atoms. __origIdx was set
    # before the disconnection, so it maps them back to the order of the xyz file, which is the
    # order the coordinates are in and the one a caller addressing atoms by index expects.
    order = [i for _, i in sorted((a.GetIntProp("__origIdx"), a.GetIdx()) for a in tmc_mol.GetAtoms())]
    tmc_mol = Chem.RenumberAtoms(tmc_mol, order)
    return tmc_mol, xyz_coords
