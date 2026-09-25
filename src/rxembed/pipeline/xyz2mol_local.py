"""Module for generating rdkit molobj/smiles/molecular graph from free atoms.

Main implementation by Jan H. Jensen, based on the paper

    Yeonjoon Kim and Woo Youn Kim
    "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
    to Three-Dimensional Geometry"
    Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777
    DOI: 10.1002/bkcs.10334

Modified by Maria Harris Rasmussen 2024

Modifications here:
- ``AC2BO`` raised ``SystemExit`` on an over-valent atom now raises a ``ValueError`` here.
- ``get_proto_mol`` seeded atom 0 from ``Chem.MolFromSmarts``, making it a query atom while every
  other atom was plain. ``RemoveHs`` would not remove a query atom, so an explicit H could survive
  based on the index ordering of the .xyz file.
- ``AC2BO`` takes ``max_combinations`` and raises a ``ValueError`` before enumerating more valence
  combinations than that.

Only the path used by rxembed (obabel-derived connectivity through AC2mol) is kept; upstream's
Huckel and van-der-Waals connectivity routes and its CLI entry point are deleted.
"""

import copy
import itertools
import logging
import math
from collections import defaultdict

import networkx as nx
import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

global __ATOM_LIST__
__ATOM_LIST__ = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4, 2]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]  # [2,1,3]
atomic_valence[9] = [1]
atomic_valence[13] = [3, 4]
atomic_valence[14] = [4]
atomic_valence[15] = [3, 5]  # [5,4,3]
atomic_valence[16] = [2, 4, 6]  # [6,3,2]
atomic_valence[17] = [1]
atomic_valence[18] = [0]
atomic_valence[32] = [4]
atomic_valence[33] = [5, 3]
atomic_valence[35] = [1]
atomic_valence[34] = [2]
atomic_valence[52] = [2]
atomic_valence[53] = [1]

atomic_valence[21] = [20]
atomic_valence[22] = [20]
atomic_valence[23] = [20]
atomic_valence[24] = [20]
atomic_valence[25] = [20]
atomic_valence[26] = [20]
atomic_valence[27] = [20]
atomic_valence[28] = [20]
atomic_valence[29] = [20]
atomic_valence[30] = [20]

atomic_valence[39] = [20]
atomic_valence[40] = [20]
atomic_valence[41] = [20]
atomic_valence[42] = [20]
atomic_valence[43] = [20]
atomic_valence[44] = [20]
atomic_valence[45] = [20]
atomic_valence[46] = [20]
atomic_valence[47] = [20]
atomic_valence[48] = [20]


atomic_valence[57] = [20]
atomic_valence[72] = [20]
atomic_valence[73] = [20]
atomic_valence[74] = [20]
atomic_valence[75] = [20]
atomic_valence[76] = [20]
atomic_valence[77] = [20]
atomic_valence[78] = [20]
atomic_valence[79] = [20]
atomic_valence[80] = [20]


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
atomic_valence_electrons[33] = 5
atomic_valence_electrons[35] = 7
atomic_valence_electrons[34] = 6
atomic_valence_electrons[52] = 6
atomic_valence_electrons[53] = 7


def int_atom(atom):
    """Convert str atom to integer atom."""
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_UA(maxValence_list, valence_list):
    """"""
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """"""
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, DU, use_graph=use_graph)[0]
    return BO


def valences_not_too_large(BO, valences):
    """"""
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def charge_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    Q = 0
    q_list = []

    if allow_charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if not allow_carbenes and number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    # A carbene-pattern C in the no-carbene charge sweep: charge it instead. This is
                    # a normal branch of the perception ladder, not an error -- no log noise.
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1
            if q != 0:
                q_list.append(q)
    return charge == Q


def BO_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    """Return whether a bond-order matrix has the right total charge and no over-valent atom."""
    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments,
        allow_carbenes=True,
    )

    if check_charge and check_sum:
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """"""
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 6 and BO_valence == 2:
        charge = 0
    elif atom == 13:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    elif atom == 16 and BO_valence == 4:  # testing for sulphur
        charge = 0
    elif atom == 16 and BO_valence == 5:
        charge = 1

    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
    use_atom_maps=True,
):
    """Write a bond-order matrix onto ``mol`` and assign atomic charges or radicals."""
    length_bo = len(BO_matrix)
    length_atoms = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if length_bo != length_atoms:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(length_bo, length_atoms)
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(length_bo):
        for j in range(i + 1, length_bo):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.RemoveBond(i, j)  # added this for TMC procedure
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps=use_atom_maps,
        )
    else:
        mol = set_atomic_radicals(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            use_atom_maps=use_atom_maps,
        )

    Chem.SanitizeMol(mol)

    return mol


def set_atomic_charges(
    mol,
    atoms,
    atomic_valence_electrons,
    BO_valences,
    BO_matrix,
    mol_charge,
    use_atom_maps=True,
):
    """"""
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if BO_valences[i] == 2:
                # q += 1
                a.SetNumRadicalElectrons(2)
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    # mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps=True):
    """The number of radical electrons = absolute atomic charge."""
    atomic_valence[8] = [2, 1]
    atomic_valence[7] = [3, 2]
    atomic_valence[6] = [4, 2]

    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """"""
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, DU, use_graph=True):
    """Return pairing(s) of unsaturated atoms to add one bond order each."""
    N_UA = 10000
    matching_ids = dict()
    matching_ids2 = dict()
    for i, du in zip(UA, DU):
        if du > 1:
            matching_ids[i] = N_UA
            matching_ids2[N_UA] = i
            N_UA += 1

    bonds = get_bonds(UA, AC)
    for i, j in bonds:
        if i in matching_ids:
            bonds.append(tuple(sorted([matching_ids[i], j])))

        elif j in matching_ids:
            bonds.append(tuple(sorted([i, matching_ids[j]])))

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        UA_pair = UA_pairs[0]

        remove_pairs = []
        add_pairs = []
        for i, j in UA_pair:
            if i in matching_ids2 and j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], matching_ids2[j]]))
            elif i in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], j]))
            elif j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([i, matching_ids2[j]]))
        for p1, p2 in zip(remove_pairs, add_pairs):
            UA_pair.remove(p1)
            UA_pair.append(p2)
        return [UA_pair]

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(
    AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True, max_combinations=None
):
    """Search bond orders and charges for the assignment with the fewest formal charges (Kim & Kim, Fig. 2).

    UA is the unsaturated-atom list, DU their degree of unsaturation, and best_BO the running-best bond
    order matrix; these are the paper's own names. The search visits every combination of per-atom
    valences, so ``max_combinations`` raises before it starts when there are more.
    """
    global atomic_valence
    global atomic_valence_electrons

    valences_list_of_lists = []  # e.g. for CO: [[4],[2,1]]
    AC_valence = list(AC.sum(axis=1))

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]  # >= neighbour count
        if atomicNum == 6 and valence == 1:
            possible_valence.remove(2)
        if atomicNum == 6 and not allow_carbenes and valence == 2:
            possible_valence.remove(2)
        if atomicNum == 6 and valence == 2:
            possible_valence.append(3)
        if atomicNum == 16 and valence == 1:
            possible_valence = [1, 2]

        if not possible_valence:
            # An over-valent atom the two-centre model cannot place: raise a CATCHABLE error rather
            # than sys.exit() -- perception callers (get_lig_mol / the charge sweep) handle it as a
            # defer, so a hard cage/hypervalent ligand never crashes the whole backend.
            raise ValueError(
                f"valence of atom {i} is {valence}, greater than the allowed max "
                f"{max(atomic_valence[atomicNum])}"
            )
        valences_list_of_lists.append(possible_valence)

    combinations = math.prod(len(options) for options in valences_list_of_lists)
    if max_combinations is not None and combinations > max_combinations:
        raise ValueError(f"{combinations} valence combinations exceed the search bound of {max_combinations}")
    valences_list = itertools.product(*valences_list_of_lists)  # e.g. [[4],[2,1]] -> [[4,2],[4,1]]

    best_BO = AC.copy()

    O_valences = [
        v_list for v_list, atomicNum in zip(valences_list_of_lists, atoms) if atomicNum == 8
    ]
    N_valences = [
        v_list for v_list, atomicNum in zip(valences_list_of_lists, atoms) if atomicNum == 7
    ]
    C_valences = [
        v_list for v_list, atomicNum in zip(valences_list_of_lists, atoms) if atomicNum == 6
    ]
    P_valences = [
        v_list for v_list, atomicNum in zip(valences_list_of_lists, atoms) if atomicNum == 15
    ]
    S_valences = [
        v_list for v_list, atomicNum in zip(valences_list_of_lists, atoms) if atomicNum == 16
    ]

    O_sums = []
    for v_list in itertools.product(*O_valences):
        O_sums.append(v_list)

    N_sums = []
    for v_list in itertools.product(*N_valences):
        N_sums.append(v_list)

    C_sums = []
    for v_list in itertools.product(*C_valences):
        C_sums.append(v_list)

    P_sums = []
    for v_list in itertools.product(*P_valences):
        P_sums.append(v_list)

    S_sums = []
    for v_list in itertools.product(*S_valences):
        S_sums.append(v_list)

    order_dict = dict()
    for i, v_list in enumerate(itertools.product(*[O_sums, N_sums, C_sums, P_sums, S_sums])):
        order_dict[v_list] = i

    valence_order_list = []
    for valence_list in valences_list:
        C_sum = []
        N_sum = []
        O_sum = []
        P_sum = []
        S_sum = []
        for v, atomicNum in zip(valence_list, atoms):
            if atomicNum == 6:
                C_sum.append(v)
            if atomicNum == 7:
                N_sum.append(v)
            if atomicNum == 8:
                O_sum.append(v)
            if atomicNum == 15:
                P_sum.append(v)
            if atomicNum == 16:
                S_sum.append(v)

        order_idx = order_dict[
            (tuple(O_sum), tuple(N_sum), tuple(C_sum), tuple(P_sum), tuple(S_sum))
        ]
        valence_order_list.append(order_idx)

    sorted_valences_list = [
        y
        for x, y in sorted(
            zip(valence_order_list, list(itertools.product(*valences_list_of_lists)))
        )
    ]

    for valences in sorted_valences_list:  # valences_list:
        UA, DU_from_AC = get_UA(valences, AC_valence)
        check_len = len(UA) == 0
        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, DU_from_AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
            charge_OK = charge_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )

            if status:
                return BO, atomic_valence_electrons
            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences) and charge_OK:
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(
    mol,
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    use_atom_maps=True,
    allow_carbenes=True,
    max_combinations=None,
):
    """Assign bond orders and charges to ``mol`` from its adjacency matrix ``AC``."""
    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=allow_carbenes,
        max_combinations=max_combinations,
    )
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps,
    )
    if Chem.GetFormalCharge(mol) != charge:
        return None

    return mol  # one arbitrary resonance form, not the full set


def get_proto_mol(atoms):
    """Return an empty-bonded RWMol with one plain atom per element in *atoms*.

    Atom 0 must not be a query atom. Upstream seeded this with ``Chem.MolFromSmarts("[#%d]" % atoms[0])``,
    which yields a query atom for index 0 only; `RemoveHs` then refuses to remove a hydrogen there
    (`RemoveHsParameters.removeWithQuery` defaults to False). Invisible unless atom 0 happens to be a
    hydrogen, which depends on the xyz file's line order: on FIHTIZ shuffled, the ferrocenyl C-H that
    landed at index 0 survived as a stray ``[H]``, and comparing the ligand SMILES as a string sorted the
    Cp ahead of the phosphine by that leading ``[`` -- swapping their coordination slots on a reshuffle.
    """
    rwMol = Chem.RWMol()
    for z in atoms:
        rwMol.AddAtom(Chem.Atom(z))
    return rwMol.GetMol()


def read_xyz_file(filename, look_for_charge=True):
    """"""
    atomic_symbols = []
    xyz_coordinates = []
    charge = 0

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def remove_weakest_bond(mol, atom_idx, AC, dMat, pt):
    """Drop ``atom_idx``'s bond with the largest slack over its covalent-radius sum."""
    extra_bond_lengths = []
    bond_atoms = np.nonzero(AC[atom_idx, :])[0]
    a_i = mol.GetAtomWithIdx(atom_idx)
    rcovi = pt.GetRcovalent(a_i.GetAtomicNum())
    for j in bond_atoms:
        a_j = mol.GetAtomWithIdx(int(j))
        rcovj = pt.GetRcovalent(a_j.GetAtomicNum())
        extra_bond_length = dMat[atom_idx, j] - rcovj - rcovi
        extra_bond_lengths.append(extra_bond_length)

    longest_bond_index = bond_atoms[np.argmax(extra_bond_lengths)]
    AC[atom_idx, longest_bond_index] = 0
    AC[longest_bond_index, atom_idx] = 0

    return AC


def xyz2AC_obabel(atoms, xyz, tolerance=0.45):
    """Return the adjacency matrix connecting atoms within their covalent-radius sum plus ``tolerance``.

    ``tolerance`` defaults to 0.45 A, the Open Babel paper's bond-perception cutoff. An atom left over
    its element's max valence has its longest bond dropped until it is not (see `remove_weakest_bond`).
    """
    global atomic_valence
    atomic_valence[6] = [4, 2]

    mol = get_proto_mol(atoms)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum())
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum())
            if dMat[i, j] <= Rcov_i + Rcov_j + tolerance:
                AC[i, j] = 1
                AC[j, i] = 1

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        N_con = np.sum(AC[i, :])
        while N_con > max(atomic_valence[a_i.GetAtomicNum()]):
            AC = remove_weakest_bond(mol, i, AC, dMat, pt)
            N_con = np.sum(AC[i, :])

    return AC, mol


def chiral_stereo_check(mol):
    """Find and embed chiral information into the model based on the
    coordinates.

    Args:
        mol - rdkit molecule, with embeded conformer
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.FindPotentialStereo(mol, cleanIt=True, flagPossible=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return
