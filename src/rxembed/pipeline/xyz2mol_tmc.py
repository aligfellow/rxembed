"""Module for the xyz2mol functionality for TMCs.

Modifications:
- ranked assignment (``_fast_bond_orders`` + ``_donor_localised_candidates`` + ``_lig_rank_key``)
- the invented-H gate (``_invented_implicit_h``)
- the canonical numbering boundary (``blind_canonical_order``, applied in ``get_lig_mol``)
- delocalised-charge canonicalisation (``_canonicalise_delocalised_charge``)
- hydride-bridge and agostic reconnection
- ``_sanitized`` as the single sanitize, so every candidate is ranked on one rule

The atom order of the returned mol is the file's, not the canonical one.
Canonicalisation happens inside, and is mapped back before returning.
"""

import logging
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    GetPeriodicTable,
    rdchem,
    rdDetermineBonds,
    rdEHTTools,
    rdmolops,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from .xyz2mol_local import AC2mol, chiral_stereo_check, read_xyz_file, xyz2AC_obabel

# Narrower than metal_core.TRANSITION_METALS, which includes the f-block. The two are not
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
_H, _B, _C = 1, 5, 6  # atomic numbers used by the reconnection and agostic checks


def canonical_ranks(mol: Chem.Mol) -> dict:
    """Return a total priority over the atoms, used where a CIP rank is unavailable.

    RDKit only sets _CIPRank as a side effect of assigning stereocentres, and leaves it unset on
    many dative metal complexes. The ranks are taken from a copy with all bonds single, all charges
    zero and no aromatic flags, so they are the same for every resonance form and do not change
    with the order of the input file. breakTies is set, so the ranking is total.
    """
    flat = Chem.RWMol(mol)
    total_h = [a.GetTotalNumHs() for a in flat.GetAtoms()]
    for b in flat.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a, h in zip(flat.GetAtoms(), total_h, strict=True):
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNumExplicitHs(h)
        a.SetNoImplicit(True)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    m = flat.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return dict(enumerate(Chem.CanonicalRankAtoms(m, breakTies=True)))


def blind_canonical_order(mol: Chem.Mol) -> list:
    """Return a canonical atom order that ignores bond orders, charges and aromaticity.

    The order is given as order[new] = old, which is what Chem.RenumberAtoms takes. It is used to
    canonicalize the input to a bond order search, which returns the first valence consistent
    assignment it reaches and so depends on the numbering. Ranking the perceived molecule would be
    circular, so the ranks come from a copy with all bonds single, charges zeroed and aromatic
    flags cleared.
    """
    blind = Chem.RWMol(mol)
    for a in blind.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
    for b in blind.GetBonds():
        b.SetBondType(Chem.rdchem.BondType.SINGLE)
        b.SetIsAromatic(False)
    m = blind.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=True))
    order = [0] * len(ranks)
    for idx, rank in enumerate(ranks):
        order[rank] = idx
    return order


ALLOWED_OXIDATION_STATES = {
    "Sc": [3],
    "Ti": [3, 4],
    "V": [2, 3, 4, 5],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 6, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2],
    "Cu": [1, 2],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [3, 4, 5],
    "Mo": [2, 3, 4, 5, 6],
    "Tc": [2, 3, 4, 5, 6, 7],
    "Ru": [2, 3, 4, 5, 6, 7, 8],
    "Rh": [1, 3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [3, 4, 5],
    "W": [2, 3, 4, 5, 6],
    "Re": [2, 3, 4, 5, 6, 7],
    "Os": [3, 4, 5, 6, 7, 8],
    "Ir": [1, 3],
    "Pt": [2, 4],
    "Au": [1, 3],
    "Hg": [1, 2],
}
# fmt: on

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

pt = GetPeriodicTable

global atomic_valence_electrons

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[13] = 3
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[18] = 8
atomic_valence_electrons[32] = 4
atomic_valence_electrons[33] = 5  # As
atomic_valence_electrons[35] = 7
atomic_valence_electrons[34] = 6
atomic_valence_electrons[53] = 7

# TMs
atomic_valence_electrons[21] = 3  # Sc
atomic_valence_electrons[22] = 4  # Ti
atomic_valence_electrons[23] = 5  # V
atomic_valence_electrons[24] = 6  # Cr
atomic_valence_electrons[25] = 7  # Mn
atomic_valence_electrons[26] = 8  # Fe
atomic_valence_electrons[27] = 9  # Co
atomic_valence_electrons[28] = 10  # Ni
atomic_valence_electrons[29] = 11  # Cu
atomic_valence_electrons[30] = 12  # Zn

atomic_valence_electrons[39] = 3  # Y
atomic_valence_electrons[40] = 4  # Zr
atomic_valence_electrons[41] = 5  # Nb
atomic_valence_electrons[42] = 6  # Mo
atomic_valence_electrons[43] = 7  # Tc
atomic_valence_electrons[44] = 8  # Ru
atomic_valence_electrons[45] = 9  # Rh
atomic_valence_electrons[46] = 10  # Pd
atomic_valence_electrons[47] = 11  # Ag
atomic_valence_electrons[48] = 12  # Cd

atomic_valence_electrons[57] = 3  # La
atomic_valence_electrons[72] = 4  # Hf
atomic_valence_electrons[73] = 5  # Ta
atomic_valence_electrons[74] = 6  # W
atomic_valence_electrons[75] = 7  # Re
atomic_valence_electrons[76] = 8  # Os
atomic_valence_electrons[77] = 9  # Ir
atomic_valence_electrons[78] = 10  # Pt
atomic_valence_electrons[79] = 11  # Au
atomic_valence_electrons[80] = 12  # Hg


def _sanitized(mol):
    """Sanitize a copy of the molecule and return it.

    Raises ValueError if the perceived bond orders do not form a valid molecule. Kept as the single
    sanitization point, since _invented_implicit_h scores a candidate by what this does to it and a
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


def fix_equivalent_Os(mol):
    """Fix a neutral coordinating atom linked to a charged atom via resonance.

    The charge is moved to the coordinating atom and charges fixed accordingly.
    """
    if isinstance(mol, Chem.RWMol):
        emol = mol
    else:
        emol = Chem.RWMol(mol)

    patt = Chem.MolFromSmarts("[#6-,#7-,#8-,#15-,#16-]-[*]=[#6,#7,#8,#15,#16]")

    matches = emol.GetSubstructMatches(patt)
    used_atom_ids_1 = []
    used_atom_ids_3 = []
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
            neighbor_idxs = [a.GetIdx() for a in atom.GetNeighbors()]
            for a1, a2, a3 in matches:
                if (
                    a3 in neighbor_idxs
                    and a1 not in neighbor_idxs
                    and a1 not in used_atom_ids_1
                    and a3 not in used_atom_ids_3
                ):
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
    valence_electrons = 0
    passed, result = rdEHTTools.RunMol(ligand_mol)
    for a in ligand_mol.GetAtoms():
        valence_electrons += atomic_valence_electrons[a.GetAtomicNum()]

    passed, result = rdEHTTools.RunMol(ligand_mol)
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


def get_basic_mol(xyz_file, overall_charge):
    """Build a basic mol object for an extended Hückel calculation.

    The object is constructed from the adjacency matrix evaluated from the
    xyz-coordinates. All bonds are single bonds, and charges are only assigned
    if necessary to work with it, i.e. a nitrogen with four neighbors gets a
    +1 charge, boron with 4 neighbors gets a -1 charge and oxygen with three
    neighbors gets a +1 charge.
    """
    atoms, _, xyz_coords = read_xyz_file(xyz_file)

    # AC, mol = xyz2AC_huckel(atoms, xyz_coords, overall_charge)
    AC, mol = xyz2AC_obabel(
        atoms, xyz_coords, tolerance=0.5
    )  # Modified tolerance to capture haptic bonds
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
        if a.GetAtomicNum() == 7:
            # explicit_valence = np.sum(AC[i])
            explicit_valence = sum([ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs])
            if explicit_valence == 4:
                a.SetFormalCharge(1)
        if a.GetAtomicNum() == 5:
            # Boron with 4 explicit bonds should be negative
            explicit_valence = sum([ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs])
            if explicit_valence == 4:
                a.SetFormalCharge(-1)
        if a.GetAtomicNum() == 8:
            explicit_valence = sum([ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs])
            if explicit_valence == 3:
                a.SetFormalCharge(1)

    return mol, xyz_coords


def _invented_implicit_h(res_mol) -> int:
    """Count the implicit hydrogens a candidate would gain when sanitized.

    The xyz file carries every hydrogen explicitly, so a correct perception gains none. A form that
    only kekulizes by de-aromatizing a ring, such as a phosphonium ylide with a pentavalent ipso
    carbon, gains one for each carbon demoted to sp3. A candidate that cannot be sanitized at all
    scores _UNSANITIZABLE, which is higher than any real count and so loses the ranking.
    """
    try:
        k = _sanitized(Chem.Mol(res_mol))
        return sum(a.GetNumImplicitHs() for a in k.GetAtoms())
    except Exception:
        return _UNSANITIZABLE


def _lig_rank_key(cand):
    """Sort key for a candidate from lig_checks.

    The candidate is (res_mol, n_pos, n_neg, n_aromatic, invented_H). Candidates are ordered by
    fewest invented hydrogens, then most aromatic atoms, then fewest formal charges away from the
    coordinating atoms, and finally by canonical SMILES. The last term settles ties between
    equivalent resonance forms, which otherwise depend on the order the supplier emits them in.
    """
    return (cand[4], -cand[3], cand[1] + cand[2], _canonical_smiles(cand[0]))


def _canonical_smiles(res_mol) -> str:
    """Canonical SMILES for a resonance form.

    A form that cannot be written returns a character that sorts after any SMILES, so it loses a
    tie in _lig_rank_key rather than winning one.
    """
    try:
        return Chem.MolToSmiles(res_mol)
    except Exception:
        return "￿"


def lig_checks(lig_mol, coordinating_atoms, resonate=True):
    """Sending proposed ligand mol object through series of checks.

    - neighbouring coordinating atoms must be connected by pi-bond, aromatic
      bond (haptic), conjugated system
    - If I have two neighbouring identical charges -> fail, I would rather
      change the charge and make a bond
     -> suggest new charge adding/subtracting electrons based on these neighbouring charges
    - count partial charges: partial charges that are not negative on ligand
      coordinating atoms count against this ligand
      -> loop through resonance forms to see if any live up to this, then choose that one.
      -> partial positive charge on coordinating atom is big red flag
      -> If "bad" partial charges still exists suggest a new charge:
         add/subtract electrons based on the values of the partial charges
    """
    # Canonicalise the supplier's input. ResonanceMolSupplier's enumeration order and pool size depend on the
    # input atom numbering (measured: RERHEB yields 16 candidates on one atom order and 17 on another; the
    # shuffled order reaches a strictly better form the original never enumerates). `min` below is stable, so
    # a pool whose order follows the xyz line order makes the winner follow it too: shuffling the file flipped
    # a metal's oxidation state
    # ([Hf+2] <-> [Hf+4]), migrated charges and flipped stereo across 31 of 142 fixtures.
    # Enumerating on a blind-canonical numbering makes both the pool and the pick invariant; the
    # chosen form is mapped straight back, so callers still see the original numbering.
    order = blind_canonical_order(lig_mol)  # order[new] = old
    new_of = {old: new for new, old in enumerate(order)}
    canon = Chem.RenumberAtoms(lig_mol, order)
    coord_canon = {new_of[int(a)] for a in coordinating_atoms if int(a) in new_of}
    back = [new_of[i] for i in range(lig_mol.GetNumAtoms())]  # canon -> original numbering

    # _donor_localised_candidates has already placed the charges, so it passes resonate=False. The
    # enumeration is also combinatorial on a large porphyrin, where each meso substituent is its own
    # conjugated group and only the core resonance matters.
    if resonate:
        res_mols = rdchem.ResonanceMolSupplier(canon)
        if len(res_mols) == 0:
            res_mols = rdchem.ResonanceMolSupplier(canon, flags=Chem.ALLOW_INCOMPLETE_OCTETS)
        # ResonanceMolSupplier is a stateful iterator: len() runs the enumeration and leaves the
        # cursor at the end, so iterating afterwards can yield None. Index instead, drop any None,
        # and fall back to the un-resonated ligand so a supplier that yields nothing degrades
        # gracefully.
        candidates = [res_mols[i] for i in range(len(res_mols))]
        candidates = [m for m in candidates if m is not None] or [canon]
    else:
        candidates = [canon]

    # Check for neighbouring coordinating atoms:
    possible_lig_mols = []

    for res_mol in candidates:
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

        # back to the caller's numbering (the enumeration ran on the blind-canonical one)
        possible_lig_mols.append(
            (
                Chem.RenumberAtoms(res_mol, back),
                len(positive_atoms),
                len(negative_atoms),
                N_aromatic,
                _invented_implicit_h(res_mol),
            )
        )
    return possible_lig_mols


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
    subsets = list(combinations(donors, k))
    if len(subsets) > _DONOR_LOCALISED_MAX:
        return []  # explainable fallback, never a silent truncation to an arbitrary subset

    out = []
    for subset in subsets:
        work = Chem.RWMol(mol)
        for a in work.GetAtoms():
            a.SetNoImplicit(True)  # the H count is given by the xyz; do not invent one
            a.SetNumExplicitHs(0)
        for i in subset:
            work.GetAtomWithIdx(i).SetFormalCharge(-1)
        try:
            rdDetermineBonds.DetermineBondOrders(work, charge=int(charge), embedChiral=False)
            cand = work.GetMol()
            Chem.SanitizeMol(cand)
            # resonate=False: the charge placement is the seed's, so no resonance search is needed.
            out.extend(lig_checks(cand, coordinating_atoms, resonate=False))
        except Exception:
            continue
    return out


def _fast_bond_orders(mol, charge, coordinating_atoms):
    """Perceive bond orders with RDKit's compiled implementation.

    rdDetermineBonds.DetermineBondOrders runs the same algorithm as AC2mol in C++ and handles most
    ligand fragments; the wider valence ligands it declines, such as dithiolenes and imidos, fall
    through to AC2mol. Returns (mol, charge), or None when nothing usable is found.

    Solutions from the whole charge ladder are collected and ranked rather than returning the first
    clean one, since a clean solution is not necessarily the right one: a metal porphyrin is clean
    both as a neutral macrocycle and as a tetra-anion.
    """
    # The molecule is already in the canonical order set by get_lig_mol. Renumbering it again here
    # would let this path and the AC2mol fallback disagree.
    step = [-2, 2, -4, 4] if charge >= 0 else [2, -2, 4, -4]
    pools = [(c, None) for c in (charge, *(charge + s for s in step))]

    solutions = []
    for c, _ in pools:
        try:
            work = Chem.RWMol(mol)
            rdDetermineBonds.DetermineBondOrders(work, charge=int(c), embedChiral=False)
            cand = work.GetMol()
            Chem.SanitizeMol(cand)
            possible = lig_checks(cand, coordinating_atoms)
        except Exception:
            continue
        solutions.extend((p, c) for p in possible)

    # ...and the candidates the blind search cannot reach.
    solutions.extend(
        (p, charge) for p in _donor_localised_candidates(mol, charge, coordinating_atoms)
    )

    clean = [
        (p, c)
        for (p, c) in solutions
        if p[4] == 0 and p[1] + p[2] == 0  # no invented H, no charge off the donors
    ]
    if not clean:
        return None
    (best, *_), c = min(
        clean,
        key=lambda s: (
            s[0][4],  # invented H  (0 for everything in `clean`, kept for intent)
            -s[0][3],  # most aromatic
            s[0][1] + s[0][2],  # fewest stray charges
            abs(s[1] - charge),  # closest to the guessed ligand charge
            _canonical_smiles(s[0][0]),  # canonical representative of the resonance hybrid
        ),
    )
    return best, c


def get_lig_mol(mol, charge, coordinating_atoms):
    """A sanitizable mol object is created for the ligand, taking into account the checks defined
    in lig_checks.

    The charge and carbene ladder is reached only when _fast_bond_orders declines, and the
    candidates it produces are ranked by _lig_rank_key rather than the first good one being kept.

    The atoms are renumbered into a canonical order before perception and mapped back afterwards.
    Every bond order search returns the first valence consistent assignment it reaches, so its
    result depends on the numbering: the same imidazolium places [n+] on either nitrogen. This
    cannot be repaired downstream, because the two numberings perceive different molecules rather
    than different renderings of one. The order comes from blind_canonical_order, so it is the same
    for every resonance form. The returned molecule is numbered as it arrived, since the caller's
    numbering matches the coordinates.
    """
    order = blind_canonical_order(mol)  # order[new] = old
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
                lig_mol = AC2mol(
                    canon, AC, atoms, q, allow_charged_fragments=True, use_atom_maps=False
                )
                if not lig_mol:
                    return None, q

        # best = (res_mol, n_pos, n_neg, n_aromatic, invented_H). "Clean" means zero invented H and
        # clean charges; a clean-charge form that still gains phantom H is not clean (the NAXDOI
        # ylide), which is why the early-out tests both.
        best = min(lig_checks(lig_mol, coord), key=_lig_rank_key)
        if best[4] == 0 and best[1] + best[2] == 0:
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
        if best[4] == 0 and best[1] + best[2] == 0:
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


# A kappa-H metal borohydride's bridging B-H is elongated (~1.6 A) vs a terminal B-H (~1.18), so the
# covalent-radius connectivity in get_basic_mol drops it: B is left a free BH3 fragment and the
# bridge H a lone metal-hydride, so the boron floats off in the 3D reconstruction (Y-B 6.6 vs real
# 3.4 A). Re-form the B-H (making a BH4- unit) up to this length.
_BRIDGE_BH_MAX = 1.85  # A; longest bridging B-H to reconnect (terminal ~1.18, bridge ~1.6-1.7)
_AGOSTIC_CH_MAX = 2.4  # A; longest agostic C-H...M contact to capture (opt-in, see `agostic=`)


def _reconnect_metal_hydride_bridges(mol, xyz_coords):
    """Reconnect a borohydride bridging the metal through hydrogen, so the BH4 stays one unit.

    Where a metal coordinated hydrogen is within bonding range of a boron that is not already its
    neighbour, the H-B bond is added and the boron becomes a BH4- anion. The metal keeps its
    hydrogen as a dative bond. A terminal M-H with no boron in range is left untouched.
    """
    coords = np.asarray(xyz_coords)
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM]
    borons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == _B]
    if not metals:
        return mol
    rw = Chem.RWMol(mol)
    added = False
    for h in mol.GetAtoms():
        if h.GetAtomicNum() != 1:
            continue
        hi = h.GetIdx()
        coordinated = any(mol.GetBondBetweenAtoms(hi, m) is not None for m in metals)
        if coordinated:  # borohydride: reconnect the bridge H to its boron
            for b in borons:
                if mol.GetBondBetweenAtoms(hi, b) is not None:
                    continue
                if float(np.linalg.norm(coords[hi] - coords[b])) <= _BRIDGE_BH_MAX:
                    rw.AddBond(hi, b, Chem.BondType.SINGLE)
                    rw.GetAtomWithIdx(b).SetFormalCharge(-1)  # neutral B + 4 bonds -> BH4- anion
                    added = True
    return rw.GetMol() if added else mol


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


def _canonicalise_delocalised_charge(mol) -> None:
    """Move the charge of a delocalised ring anion onto a canonical atom, in place.

    SMILES carries the charge on a single atom, and which atom it was placed on followed the atom
    index, so reordering the xyz file moved an indenyl carbanion around its ring. A symmetric ring
    such as a plain Cp is unaffected, since every placement is isomorphic, but a benzo-fused or
    bridged ring is not.

    The candidates are the ring atoms bound to the metal. For a fully bound ring that is every ring
    atom and the choice is conventional; for a slipped or partially bound ring it keeps the charge
    on the coordinated part, which is both the correct Lewis structure and what the distance model
    expects. The highest CIP rank wins, computed on a neutralized copy so that the ranking does not
    depend on the charge being placed. A tie means the atoms are equivalent, and the lowest index
    settles it.
    """
    metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM}
    if not metals:
        return
    bound = {b.GetOtherAtomIdx(m) for m in metals for b in mol.GetAtomWithIdx(m).GetBonds()}
    # Rank a neutralised copy, so the ranking does not depend on the charge it is placing.
    # `canonical_ranks` is the fallback, never the atom index: RDKit sets `_CIPRank` only as a side
    # effect of assigning stereocentres and leaves it unset on plenty of dative complexes, and an
    # unset rank used to tie every candidate and fall through to index order.
    probe = Chem.RWMol(mol)
    for a in probe.GetAtoms():
        if a.GetAtomicNum() not in TRANSITION_METALS_NUM:
            a.SetFormalCharge(0)
    neutral = probe.GetMol()
    cip = {}
    try:
        Chem.SanitizeMol(neutral, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        Chem.AssignStereochemistry(neutral, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        cip = {
            a.GetIdx(): r
            for a in neutral.GetAtoms()
            if (r := a.GetPropsAsDict().get("_CIPRank")) is not None
        }
    except (ValueError, RuntimeError):
        cip = {}
    cip = cip or canonical_ranks(mol)
    moved = False
    for ring in mol.GetRingInfo().AtomRings():
        anions = [
            i
            for i in ring
            if mol.GetAtomWithIdx(i).GetFormalCharge() < 0 and mol.GetAtomWithIdx(i).GetIsAromatic()
        ]
        if len(anions) != 1:
            continue  # one delocalised anion per ring, or we are not the right tool
        cur = anions[0]
        # `cur` is not privileged among the candidates: a sigma-bonded Cp is a carbanion at the
        # donor, and the search sometimes parks the anion on an unbound ring carbon instead, so the
        # charge must be free to move onto a bound atom and not merely between bound atoms.
        cands = [i for i in ring if i in bound and mol.GetAtomWithIdx(i).GetIsAromatic()]
        if not cands:
            continue  # ring not coordinated through an aromatic C: not ours, leave the chemistry

        # A candidate must kekulize, not merely sanitize: SanitizeMol accepts an aromatic ring with
        # no Kekule structure, which then fails downstream where the caller kekulizes.
        legal = []
        for cand in sorted(cands, key=lambda i: (-cip.get(i, -1), i)):
            if cand == cur:
                legal.append(cand)  # what perception produced: legal by construction
                continue
            trial = Chem.RWMol(mol)
            trial.GetAtomWithIdx(cur).SetFormalCharge(0)
            trial.GetAtomWithIdx(cand).SetFormalCharge(-1)
            probe = trial.GetMol()
            try:
                Chem.SanitizeMol(probe)
                Chem.Kekulize(Chem.Mol(probe), clearAromaticFlags=True)  # must be WRITABLE too
            except Exception:
                continue
            legal.append(cand)
        if not legal:
            continue
        best = legal[0]  # already in (highest CIP, lowest index) order
        if best != cur:
            mol.GetAtomWithIdx(cur).SetFormalCharge(0)
            mol.GetAtomWithIdx(best).SetFormalCharge(-1)
            moved = True

    if moved:
        # Moving a charge invalidates the aromaticity and implicit-H counts that depend on it, and
        # RDKit does not recompute them; left stale, the ring kekulizes here and fails downstream.
        # A fused system also visits a shared atom twice, so the mol is mutated more than once.
        Chem.SanitizeMol(mol)


def get_tmc_mol(xyz_file, overall_charge, with_stereo=False, agostic=False, graph=None):
    """Get TMC mol object from given xyz file.

    Args:
        xyz_file (str) : Path to TMC xyz file
        overall_charge (int): Overall charge of TMC
        with_stereo (bool): Whether to percieve stereochemistry from the 3D data
        agostic (bool): Also write a dative H->M for a C-H that points at the metal within
            _AGOSTIC_CH_MAX. Off by default, since an agostic contact is a real interaction but
            not a bond every consumer expects.
        graph (tuple): An (mol, xyz_coords) pair to use in place of perceiving the connectivity
            from the file. The bonds are taken as given and only their orders and formal charges
            are assigned, which lets another perceiver supply the connectivity. Note that
            get_basic_mol also seeds the structural charges (N with four neighbours, B with four,
            O with three), so a graph supplied here is expected to carry them already.

    Returns:
        tmc_mol (rdkit.Chem.rdchem.Mol): TMC mol object
    """
    mol, xyz_coords = graph if graph is not None else get_basic_mol(xyz_file, overall_charge)
    mol = _reconnect_metal_hydride_bridges(mol, xyz_coords)  # keep a kappa-H BH4 connected
    # Capture agostic C-H...M contacts by index now, but add the datives only after perception
    # (see _agostic_ch_metal_pairs) -- here idx == __origIdx, set in the loop just below.
    agostic_pairs = _agostic_ch_metal_pairs(mol, xyz_coords, agostic)

    tmc_idx = None
    for a in mol.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            # tm_atom = a.GetSymbol()
            tmc_idx = a.GetIdx()

    if tmc_idx is None:
        raise Exception("Found no TM in the input file. Please supply an xyz file with a TM")

    coordinating_atoms = np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol)[tmc_idx, :])[0]

    # Exclude aromatic ring carbons that are backbone atoms in bidentate chelates
    # (e.g., the phenylene bridge in bidentate phosphine ligands).
    # If a carbon is in an aromatic ring and has a neighbour that is a coordinating heteroatom donor
    # (P, N, S, O), exclude the carbon: the heteroatom is the
    # real donor. This is safe for eta-ligands (Cp, arene) where no ring carbons neighbor
    # heteroatom donors that are coordinating.
    HETEROATOM_DONORS = {7, 8, 15, 16}  # N, O, P, S
    coordinating_set = set(int(idx) for idx in coordinating_atoms)
    filtered_atoms = []
    for atom_idx in coordinating_atoms:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        if atom.GetAtomicNum() == 6 and atom.IsInRing():  # Carbon in a ring
            neighbors = atom.GetNeighbors()
            has_heteroatom_donor_neighbor = any(
                n.GetAtomicNum() in HETEROATOM_DONORS and n.GetIdx() in coordinating_set
                for n in neighbors
            )
            if has_heteroatom_donor_neighbor:
                continue  # Skip this ring carbon; the neighboring heteroatom is the donor
        filtered_atoms.append(atom_idx)
    coordinating_atoms = np.array(filtered_atoms, dtype=int)

    # frags = rdMolStandardize.DisconnectOrganometallics(mol, params)
    mdis = rdMolStandardize.MetalDisconnector(params)
    mdis.SetMetalNon(Chem.MolFromSmarts(MetalNon_Hg))
    mdis.SetMetalNof(Chem.MolFromSmarts(MetalNof_TM))
    frags = mdis.Disconnect(mol)
    frag_mols = rdmolops.GetMolFrags(frags, asMols=True)

    total_lig_charge = 0
    tm_idx = None
    lig_list = []
    for i, f in enumerate(frag_mols):
        m = Chem.Mol(f)
        atoms = m.GetAtoms()
        for atom in atoms:
            if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
                tm_idx = i
                break
        else:
            lig_charge = get_proposed_ligand_charge(f)

            lig_coordinating_atoms = [
                a.GetIdx() for a in m.GetAtoms() if a.GetIntProp("__origIdx") in coordinating_atoms
            ]
            lig_mol, lig_charge = get_lig_mol(m, lig_charge, lig_coordinating_atoms)
            if not lig_mol:
                try:
                    frag_smiles = Chem.MolToSmiles(m)
                except Exception:
                    frag_smiles = f"<{m.GetNumAtoms()} atoms>"
                raise ValueError(
                    f"get_lig_mol failed for ligand fragment #{i} "
                    f"(SMILES: {frag_smiles!r}); cannot build TMC mol"
                )

            # Restore __origIdx from m to lig_mol
            if lig_mol.GetNumAtoms() == m.GetNumAtoms():
                for a_lig, a_orig in zip(lig_mol.GetAtoms(), m.GetAtoms()):
                    if a_orig.HasProp("__origIdx"):
                        a_lig.SetIntProp("__origIdx", a_orig.GetIntProp("__origIdx"))

            total_lig_charge += lig_charge
            lig_list.append(lig_mol)

    if tm_idx is None:
        raise Exception("Found no TM in the input file. Please supply an xyz file with a TM")

    tm = Chem.RWMol(frag_mols[tm_idx])
    tm_ox = overall_charge - total_lig_charge

    for a in tm.GetAtoms():
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            a.SetFormalCharge(tm_ox)

    for lmol in lig_list:
        tm = Chem.CombineMols(tm, lmol)

    emol = Chem.RWMol(tm)
    coordinating_atoms_idx = [
        a.GetIdx() for a in emol.GetAtoms() if a.GetIntProp("__origIdx") in coordinating_atoms
    ]
    tm_idx = [a.GetIdx() for a in emol.GetAtoms() if a.GetIntProp("__origIdx") == tmc_idx][0]
    dMat = Chem.Get3DDistanceMatrix(emol)
    cut_atoms = []
    for i, j in combinations(coordinating_atoms_idx, 2):
        bond = emol.GetBondBetweenAtoms(int(i), int(j))
        if bond and abs(dMat[i, tm_idx] - dMat[j, tm_idx]) >= 0.4:
            logger.debug(
                "Haptic bond pattern with too great distance: %s vs %s",
                dMat[i, tm_idx],
                dMat[j, tm_idx],
            )
            if dMat[i, tm_idx] > dMat[j, tm_idx] and i in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(i)
                cut_atoms.append(i)
            if dMat[j, tm_idx] > dMat[i, tm_idx] and j in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(j)
                cut_atoms.append(j)
    for j in cut_atoms:
        for i in coordinating_atoms_idx:
            bond = emol.GetBondBetweenAtoms(int(i), int(j))
            if bond and dMat[i, tm_idx] - dMat[j, tm_idx] >= -0.1 and i in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(i)

    for i in coordinating_atoms_idx:
        if emol.GetBondBetweenAtoms(i, tm_idx):
            continue
        emol.AddBond(i, tm_idx, Chem.BondType.DATIVE)

    # Agostic C-H...M: now that every ligand is perceived and reassembled, reconnect the recorded
    # H->M datives (begin=H donor). Doing it here, not before disconnection, is what keeps the
    # ligand a normal fragment through bond-order perception (see _agostic_ch_metal_pairs).
    orig_to_emol = {a.GetIntProp("__origIdx"): a.GetIdx() for a in emol.GetAtoms()}
    for h_orig, m_orig in agostic_pairs:
        hi, mi = orig_to_emol.get(h_orig), orig_to_emol.get(m_orig)
        if hi is not None and mi is not None and emol.GetBondBetweenAtoms(hi, mi) is None:
            emol.AddBond(hi, mi, Chem.BondType.DATIVE)

    # Fix specific cases
    # Operate on emol directly to preserve properties
    emol = fix_equivalent_Os(emol)
    emol = fix_NO2(emol)

    tmc_mol = _sanitized(emol.GetMol())
    # Atom-count invariant: the source XYZ has every hydrogen explicit, so a correct perception
    # invents none. Any implicit H means an atom was mis-typed (an aromatic C read as sp3, a ligand
    # that skipped bond-order perception) and the formula no longer matches the real structure --
    # fail loudly rather than ship a byte-wrong descriptor.
    invented = sum(a.GetNumImplicitHs() for a in tmc_mol.GetAtoms())
    if invented:
        raise ValueError(
            f"perception invented {invented} implicit H (formula no longer matches the XYZ); "
            "a coordinated ligand was likely mis-typed / skipped bond-order perception"
        )
    _canonicalise_delocalised_charge(tmc_mol)
    if with_stereo:
        chiral_stereo_check(tmc_mol)
    # Disconnecting the metal and reassembling the fragments reorders the atoms. __origIdx was set
    # before the disconnection, so it maps them back to the order of the xyz file, which is the
    # order the coordinates are in and the one a caller addressing atoms by index expects.
    order = [i for _, i in sorted((a.GetIntProp("__origIdx"), a.GetIdx()) for a in tmc_mol.GetAtoms())]
    tmc_mol = Chem.RenumberAtoms(tmc_mol, order)
    return tmc_mol, xyz_coords
