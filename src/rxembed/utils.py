"""Coordinate math, small RDKit facts, and the QA result type: the core's shared leaf.

Admission rule: if it needs to know what a metal, a donor or a constraint is, it does not belong here.
Everything in this module reads a molecule or an array and answers a question that has no domain in it, which
is what lets the coordination perception and the pipeline QA gate measure an angle the same way without a
second definition or an import cycle.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()
CARBON_Z = 6
SP2_DEGREE = 3  # a planar sp2 centre has exactly three neighbours
DISCONNECTED = 1e6  # RDKit's topological distance for atoms in different fragments (it returns ~1e8)
_TETRAHEDRAL_DEGREE = 4  # the only degree at which a CW/CCW tag has a bond-order parity: see `bond_removal_mirrors`
_MIRRORED = {  # the two tetrahedral tags; no other ChiralType is a parity over the bond order
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


def cip_cache_key(mol, centers=()):
    """Return a hashable key for `mol`'s chemical content, plus an optional labelled-centre tuple.

    A wrong key here silently hands back another molecule's cached answer, so it must cover every field two
    graphs could differ on: each atom's element, isotope, charge, H count, radical count, aromaticity,
    chiral tag and explicit/implicit-H flag; each bond's endpoints, type, aromaticity, stereo flag and
    stereo reference atoms; and each enhanced stereo group's type and member atoms/bonds. The explicit/
    implicit-H flag has to sit alongside the H count itself: two graphs can agree on the total H count while
    disagreeing on whether it is fixed or open to recomputation.
    """
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetIsotope(),
            a.GetFormalCharge(),
            a.GetTotalNumHs(),
            a.GetNumRadicalElectrons(),
            a.GetIsAromatic(),
            a.GetChiralTag(),
            a.GetNoImplicit(),
        )
        for a in mol.GetAtoms()
    )
    bonds = tuple(
        (
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            b.GetBondType(),
            b.GetIsAromatic(),
            b.GetStereo(),
            tuple(b.GetStereoAtoms()),
        )
        for b in mol.GetBonds()
    )
    groups = tuple(
        (int(g.GetGroupType()), tuple(a.GetIdx() for a in g.GetAtoms()), tuple(b.GetIdx() for b in g.GetBonds()))
        for g in mol.GetStereoGroups()
    )
    return atoms, bonds, groups, tuple(sorted(set(centers)))


def flat_ranks(mol, *, break_ties=False):
    """Rank a graph with bond order, charge, aromaticity, and stereo removed."""
    rw = Chem.RWMol(mol)
    hydrogens = [atom.GetTotalNumHs() for atom in rw.GetAtoms()]
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom, count in zip(rw.GetAtoms(), hydrogens, strict=True):
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(count)
        atom.SetNoImplicit(True)
    flat = rw.GetMol()
    flat.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(flat)
    return list(Chem.CanonicalRankAtoms(flat, breakTies=break_ties))


def without_atom_maps(mol):
    """Return `mol` without atom maps, a copy only when it has some; RDKit's ranks read maps as identity."""
    if not any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        return mol
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


def hydrogen_neighbor_order(mol, hydrogen, *, metals, positions=None, ranks=None):
    """Order a multibound hydrogen's neighbours with its nonmetal ligand leg first."""
    if ranks is None:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    return sorted(
        (neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(hydrogen).GetNeighbors()),
        key=lambda neighbor: (
            mol.GetAtomWithIdx(neighbor).GetAtomicNum() in metals,
            0.0
            if positions is None
            else float(np.dot(positions[neighbor] - positions[hydrogen], positions[neighbor] - positions[hydrogen])),
            ranks[neighbor],
            neighbor,
        ),
    )


@dataclass(frozen=True)
class Violation:
    """One failed check. ``value`` breached ``limit`` for the atoms named."""

    kind: str
    atoms: tuple[int, ...]
    value: float = field(kw_only=True)
    limit: float = field(kw_only=True)
    detail: str = field(default="", kw_only=True)

    def __str__(self) -> str:
        """Format the violation as a readable one-liner."""
        a = "-".join(map(str, self.atoms))
        return f"[{self.kind}] atoms {a}: {self.value:.3f} vs {self.limit:.3f} {self.detail}".rstrip()


def conjugated_quartets(mol, exclude=frozenset()):
    """Yield each ``(a, c, x, s)`` quartet whose A=C-X-S dihedral measures a conjugation plane.

    A single, non-ring bond X-C (X in N/O, C bearing a double bond to A) where X carries a substituent S. The
    sole perception both ``pipeline.geom_check`` and ``ConjugationCap`` read, so they cannot name different
    atoms. ``exclude`` drops a quartet whose X or C is a frozen / metal atom, as the gate does.
    """
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != CARBON_Z:
                continue
            if x_atom.GetIdx() in exclude or c_atom.GetIdx() in exclude:
                continue
            dbl = [
                n
                for n in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [
                n
                for n in x_atom.GetNeighbors()
                if n.GetIdx() != c_atom.GetIdx()
                and mol.GetBondBetweenAtoms(x_atom.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.DATIVE
            ]
            if not dbl or not subs:
                continue
            yield dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()


def as_positions(mol_or_pos, conf_id: int = -1) -> np.ndarray:
    """Return (N, 3) coordinates from a Mol conformer, passing an array through unchanged."""
    if isinstance(mol_or_pos, np.ndarray):
        return mol_or_pos
    return mol_or_pos.GetConformer(conf_id).GetPositions()


def atom_label(mol, idx):
    """Return an atom's element symbol followed by its index, as messages print it: ``'C12'``."""
    return f"{mol.GetAtomWithIdx(int(idx)).GetSymbol()}{int(idx)}"


def lone_pair_electrons(atom, metals):
    """Return `atom`'s nonbonding valence: outer electrons minus formal charge minus bonded valence.

    A bond to a metal is stripped from the bonded-valence term first (`Bond.GetValenceContrib`, zero
    for a dative donor bond, the bond order for a covalent one), so this reads the same whether `atom`
    is itself dative- or covalent-bonded to the metal, and the same as passing `metals=()` on a graph
    that has none. Not a per-element list, so it holds for any main-group atom. A result of at least
    two means the atom keeps a lone pair it can donate.
    """
    to_metal = sum(
        bond.GetValenceContrib(atom) for bond in atom.GetBonds() if bond.GetOtherAtomIdx(atom.GetIdx()) in metals
    )
    return _PT.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() - (atom.GetTotalValence() - to_metal)


def hydrogen_bond(hydrogen, partner, metals) -> bool:
    """Return whether a hydrogen's extra contact to `partner` is a hydrogen bond rather than a bond.

    A partner that keeps a lone pair (X-H...Y) accepts the hydrogen noncovalently, so the contact is an NCI a
    caller states, not constitution. A partner without one (B-H-B) shares the hydrogen's pair in a three-centre
    bond, and a metal partner makes a bridging hydride; both are bonding.
    """
    return (
        hydrogen.GetAtomicNum() == 1 and partner.GetIdx() not in metals and lone_pair_electrons(partner, metals) >= 2  # noqa: PLR2004  one lone pair
    )


def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the a-b-c angle in degrees, or NaN when a leg has zero length."""
    u, w = a - b, c - b
    scale = np.linalg.norm(u) * np.linalg.norm(w)
    if scale == 0.0:
        return float("nan")
    cos = np.dot(u, w) / scale
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the signed dihedral p0-p1-p2-p3 in degrees, or NaN when it is undefined."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    axis = np.linalg.norm(b1)
    if axis == 0.0:
        return float("nan")
    b1 /= axis
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    eps = np.finfo(float).eps
    if np.linalg.norm(v) <= eps * np.linalg.norm(b0) or np.linalg.norm(w) <= eps * np.linalg.norm(b2):
        return float("nan")
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def mirror_tag(tag):
    """Return the tetrahedral tag naming the mirror image of `tag`; anything else passes through unchanged."""
    return _MIRRORED.get(tag, tag)


def bond_removal_mirrors(atom, partner, degrees=(_TETRAHEDRAL_DEGREE,)) -> bool:
    """Return whether removing the bond to `partner` changes `atom`'s tetrahedral-tag parity.

    RDKit's CW/CCW tag is relative to ``atom.GetBonds()`` order. Removing slot ``p`` changes that basis by
    ``n - 1 - p`` swaps, so an odd count needs the tag mirrored. Call this on the graph that still has the
    bond, even before a tag exists; the same correction applies inversely when grafting one. A plain
    ``AddBond`` needs no correction, since RDKit appends the new bond last. Defined by default only at degree
    four: higher degrees are left unchanged, and lower degrees have no representable tetrahedral chirality.
    Replacing the neighbour by an appended one moves the same slot, and a lone pair holds the fourth position
    of a degree-three centre, so that caller passes ``degrees=(3, 4)``.
    """
    partners = [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]
    if len(partners) not in degrees or partner not in partners:
        return False
    return (len(partners) - 1 - partners.index(partner)) % 2 == 1


def combined(left, right):
    """Return `left` and `right` as one Mol, `right`'s atoms numbered after `left`'s, with ring perception done."""
    mol = Chem.CombineMols(left, right)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return mol


def remove_bond(rw, i, j) -> None:
    """Remove a bond while preserving the geometry named by degree-four tetrahedral tags.

    RDKit does not update chiral tags on bond removal, so a direct `RemoveBond` can silently invert a
    centre's handedness; an edit whose tags must survive uses this wrapper instead.
    """
    for a, other in ((int(i), int(j)), (int(j), int(i))):
        atom = rw.GetAtomWithIdx(a)
        if bond_removal_mirrors(atom, other):
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
    rw.RemoveBond(int(i), int(j))


def without_zero_bonds(mol):
    """Return `mol` without its zero-order contacts: the graph an identity ranking reads.

    A zero-order contact (the XYZ reader's second leg of a shared proton, a CX ``Z:`` bond) is not constitution:
    no geometry check holds it, and a re-read of the embedded geometry does not have it. RDKit is not consistent
    about it either: SSSR skips it, while FastFindRings, CanonicalRankAtoms, the CIP labeller and
    GetDistanceMatrix count it as an edge.
    """
    zero = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.ZERO]
    if not zero:
        return mol
    rw = Chem.RWMol(mol)
    for i, j in zero:
        remove_bond(rw, i, j)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def assign_stereo_from_3d(mol, conf_id: int = -1) -> None:
    """Assign 3D stereochemistry in the bond-order basis used by rxembed readers.

    RDKit's 3D writer omits a donor-originating dative bond from the centre's neighbour basis, while its CIP
    and SMILES readers include it: the same sulfoxide geometry parses as CIP R but assigns from raw 3D as
    CIP S. Mirror a degree-four tag when that difference changes its parity, then refresh CIP labels.
    Higher-degree tags are left unchanged, since the parity rule does not hold there. `stereo` calls RDKit
    directly on its metal-free enumeration graph, which leaves a ligand-internal dative bond unrebased.
    """
    Chem.AssignStereochemistryFrom3D(mol, confId=conf_id)
    rebased = False
    for atom in mol.GetAtoms():
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() == atom.GetIdx():
                if bond_removal_mirrors(atom, bond.GetOtherAtomIdx(atom.GetIdx())):
                    atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
                    rebased = True
    if rebased:
        # AssignStereochemistryFrom3D also writes `_CIPCode`; refresh it after mirroring a tag.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def repair_bond_stereo(mol) -> int:
    """Re-derive or drop any bond stereo whose reference atoms were lost to bond surgery; return how many.

    Removing a bond can leave a double bond still flagged E/Z while RDKit silently drops one of its two
    stereo reference atoms, because that atom was the removed partner. The flag then carries no information,
    and RDKit's own ETKDG indexes the empty reference vector and segfaults: a crash no ``try``/``except``
    can catch.

    Where a conformer survives the surgery, the geometry is re-perceived from it, so the E/Z is kept,
    re-expressed against the substituents that remain rather than discarded. Only where no reference
    survives is the flag dropped, since the information itself did not survive the surgery.
    """
    # Metal-ring SMILES can make RDKit choose the metal as both E/Z references. Once bond surgery removes
    # the metal, the original slash bonds are still the least ambiguous source of the ligand's E/Z. Preserve
    # already-valid E/Z verbatim: SetBondStereoFromDirections otherwise rewrites E/Z as TRANS/CIS.
    stated = {
        b.GetIdx(): (b.GetStereo(), tuple(b.GetStereoAtoms()))
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetStereo() != Chem.BondStereo.STEREONONE
        and len(b.GetStereoAtoms()) == 2  # noqa: PLR2004
        and len(set(b.GetStereoAtoms())) == 2  # noqa: PLR2004
    }
    Chem.SetBondStereoFromDirections(mol)
    for idx, (tag, refs) in stated.items():
        bond = mol.GetBondWithIdx(idx)
        bond.SetStereoAtoms(*refs)
        bond.SetStereo(tag)
    orphaned = [
        b
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetStereo() != Chem.BondStereo.STEREONONE
        and len(b.GetStereoAtoms()) != 2  # noqa: PLR2004
    ]
    if not orphaned:
        return 0
    if mol.GetNumConformers():  # re-perceive from the geometry: keeps the E/Z, re-referenced
        assign_stereo_from_3d(mol)
    for b in orphaned:  # whatever re-perception could not re-reference carries no information, so drop it
        if len(b.GetStereoAtoms()) != 2:  # noqa: PLR2004
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return len(orphaned)


# --- the reaction graph overlay ----------------------------------------------------------------------
_MAX_SPARE = 2  # bond pairs broken and re-formed beyond the element-pair balance; unmeasured
_MAX_REMAINDERS = 20_000  # remainders canonicalised in one overlay search; unmeasured


def map_reaction(reactant, product, frames=False):
    """Return both sides on the reactant's atom order; any atom maps stay on both.

    Two `.xyz` `frames` on one element order keep it: an .xyz carries no atom map, frames of one path already share
    their atoms, and only that order tells a degenerate exchange apart. Otherwise the product is renumbered by its
    maximal overlay onto the reactant's graph (`_overlay_order`).
    """
    sides = [Chem.Mol(reactant), Chem.Mol(product)]
    elements = [[(atom.GetAtomicNum(), atom.GetIsotope()) for atom in mol.GetAtoms()] for mol in sides]
    if sorted(elements[0]) != sorted(elements[1]):
        raise ValueError("the two sides hold different atoms; a reaction's endpoints share one formula")
    maps = [Counter(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()) for mol in sides]
    if maps[0].keys() != maps[1].keys() or any(uses > 1 for numbers in maps for uses in numbers.values()):
        raise ValueError("give each atom-map number to one atom on each side, and the same numbers on both sides")
    if not (frames and elements[0] == elements[1]):
        sides[1] = Chem.RenumberAtoms(sides[1], _overlay_order(*sides))
    return sides


def _overlay_order(reactant, product):
    """Return the product's atom order onto the reactant's: the map that keeps the most bonds.

    The overlay reads connectivity alone. Elements, isotopes and atom-map numbers label atoms, so a mapped atom
    overlays only its partner; bond orders, charges and stereo are ignored. As in RMSD-PP, it drops the fewest
    bonds from each side until both remainders are one graph and pairs atoms by their identical canonical SMILES.
    The element-pair bond counts fix how many bonds of each pair change; a pair whose count balances (a proton
    moved O to O) breaks and re-forms up to `_MAX_SPARE` more. A bond between two mapped atoms is not searched: it
    stays where both sides bond the pair and changes where one does not.
    """
    (r_free, r_changed), (p_free, p_changed) = _bond_classes(reactant, product), _bond_classes(product, reactant)
    skeletons = [_without(_skeleton(mol), changed) for mol, changed in ((reactant, r_changed), (product, p_changed))]
    classes = sorted(r_free.keys() | p_free.keys())
    balance = {pair: len(r_free[pair]) - len(p_free[pair]) for pair in classes}
    shared = [pair for pair in classes if r_free[pair] and p_free[pair]]
    budget = _MAX_REMAINDERS
    for spare in range(_MAX_SPARE + 1):
        drops = [
            [{pair: max(sign * balance[pair], 0) + extra.count(pair) for pair in classes} for sign in (1, -1)]
            for extra in itertools.combinations_with_replacement(shared, spare)
        ]
        budget -= sum(
            math.prod(math.comb(len(free[pair]), count) for pair, count in drop.items())
            for sides in drops
            for free, drop in zip((r_free, p_free), sides, strict=True)
        )
        if budget < 0:
            raise ValueError("too many ways to overlay these two graphs; map the reacting atoms on both sides")
        overlays = []
        for r_drop, p_drop in drops:
            left = _remainders(skeletons[0], r_free, r_drop, r_changed)
            right = _remainders(skeletons[1], p_free, p_drop, p_changed)
            overlays += [(r, p) for key in left.keys() & right.keys() for r in left[key] for p in right[key]]
        if overlays:
            return _one_reaction(reactant, overlays)
    raise ValueError(
        f"the two graphs overlay only once more than {_MAX_SPARE} extra bonds break and re-form; check the atom "
        "maps, or map the reacting atoms on both sides"
    )


def _one_reaction(reactant, overlays):
    """Return the first overlay's order, or raise when tied overlays are different reactions.

    A reaction is its ITS graph: the kept, broken and formed bonds, each its own bond type. Two overlays, or an
    overlay and its formed bonds moved by the remainder's symmetry (`_images`), are one reaction when their ITS
    graphs share a canonical SMILES. Canonical ranks stand in for symmetry orbits; on a rare regular graph they
    merge too much, which raises rather than picks.
    """
    skeleton = _skeleton(reactant)
    reactions = {}
    for (broken, r_order), (dropped, p_order) in overlays:
        onto = dict(zip(p_order, r_order, strict=True))  # product atom -> reactant atom
        formed = [(onto[i], onto[j]) for i, j in dropped]
        remainder = _without(skeleton, broken)
        atoms = sorted({atom for pair in formed for atom in pair})
        for image in _images(remainder, atoms):
            sent = dict(zip(atoms, image, strict=True))
            its = Chem.RWMol(remainder)
            for i, j in broken:
                its.AddBond(i, j, Chem.BondType.DOUBLE)
            for i, j in formed:
                its.AddBond(sent[i], sent[j], Chem.BondType.TRIPLE)
            reactions.setdefault(Chem.MolToSmiles(its), (broken, [(sent[i], sent[j]) for i, j in formed]))
        if len(reactions) > 1:
            readings = [
                f"breaking {_bond_labels(reactant, broken)} and forming {_bond_labels(reactant, formed)}"
                for broken, formed in list(reactions.values())[:2]
            ]
            raise ValueError(
                f"the two graphs overlay equally well as different reactions ({' or '.join(readings)}); map the "
                "reacting atoms on both sides"
            )
    (_broken, r_order), (_dropped, p_order) = overlays[0]
    order = [0] * len(r_order)
    for r, p in zip(r_order, p_order, strict=True):
        order[r] = p
    return order


def _images(skeleton, atoms):
    """Return each tuple of atoms the skeleton's symmetry can send `atoms` to, fixing one atom at a time."""

    def ranks(fixed):
        marked = Chem.RWMol(skeleton)
        for label, atom in enumerate(fixed, 1):
            marked.GetAtomWithIdx(atom).SetAtomMapNum(1_000_000 + label)  # above any input map number
        return Chem.CanonicalRankAtoms(marked, breakTies=False)

    images = [()]
    for i, atom in enumerate(atoms):
        target = ranks(atoms[:i])[atom]
        images = [(*image, other) for image in images for other, rank in enumerate(ranks(image)) if rank == target]
    return images


def _bond_labels(mol, bonds):
    """Return `bonds` as message text, such as ``'C0-C5'``."""
    return ", ".join(f"{atom_label(mol, i)}-{atom_label(mol, j)}" for i, j in bonds) or "nothing"


def _bond_classes(mol, partner):
    """Return `mol`'s bonds an overlay may drop, by element and isotope pair, and its mapped bonds `partner` lacks."""
    where = {atom.GetAtomMapNum(): atom.GetIdx() for atom in partner.GetAtoms() if atom.GetAtomMapNum()}
    free, changed = defaultdict(list), []
    for bond in mol.GetBonds():
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if all(atom.GetAtomMapNum() for atom in ends):
            if partner.GetBondBetweenAtoms(*(where[atom.GetAtomMapNum()] for atom in ends)) is None:
                changed.append(pair)
            continue
        free[tuple(sorted((atom.GetAtomicNum(), atom.GetIsotope()) for atom in ends))].append(pair)
    return free, changed


def _remainders(skeleton, free, drops, changed):
    """Return `skeleton`'s remainders without `drops[pair]` of each pair's `free` bonds, by canonical SMILES.

    Each carries its dropped bonds, `changed` included, and its atom output order.
    """
    out = defaultdict(list)
    for picks in itertools.product(*(itertools.combinations(free[pair], count) for pair, count in drops.items())):
        dropped = [bond for pick in picks for bond in pick]
        remainder = _without(skeleton, dropped)
        key = Chem.MolToSmiles(remainder)
        out[key].append(([*changed, *dropped], list(remainder.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])))
    return out


def _skeleton(mol):
    """Return `mol`'s connectivity: atoms labelled by element, isotope and atom map, joined by plain bonds."""
    skeleton = Chem.RWMol()
    for atom in mol.GetAtoms():
        label = Chem.Atom(atom.GetAtomicNum())
        label.SetIsotope(atom.GetIsotope())
        label.SetAtomMapNum(atom.GetAtomMapNum())
        label.SetNoImplicit(True)
        skeleton.AddAtom(label)
    for bond in mol.GetBonds():
        skeleton.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return skeleton


def _without(skeleton, bonds):
    """Return a copy of `skeleton` without `bonds`, ready to canonicalise."""
    remainder = Chem.RWMol(skeleton)
    for i, j in bonds:
        remainder.RemoveBond(i, j)
    remainder.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(remainder)
    return remainder
