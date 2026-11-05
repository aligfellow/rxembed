"""Put a reaction's product on its reactant's atom order, and tell which atoms the reaction changes, for `rx.coembed`.

The reactant's order is never changed. A caller with a mapper of its own passes atom-mapped input instead. Which metal
sites move (`site_motions`) and which atoms turn to change a configuration (`turned_atoms`) are read from the two
endpoints alone, with no coordinate.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from rdkit import Chem

from rxembed.geometry.constraints import resolve_atom
from rxembed.graph import metal_indices
from rxembed.identity.smiles_io import distinct
from rxembed.identity.species import graph_with_stereo
from rxembed.metal.polyhedra import VACANT, seating_frames, vertex_dirs
from rxembed.stereo import renumber_atoms
from rxembed.utils import atom_label, skeleton

_MAX_SPARE = 2  # bond pairs broken and re-formed beyond the element-pair balance
_MAX_REMAINDERS = 20_000  # remainders canonicalised in one overlay search


def in_reactant_order(reactant, product):
    """Return `product` renumbered onto the reactant's atom order, so atom i is one atom on both sides.

    An atom-map number pairs its two atoms. Two sides of one identity pair by RDKit's canonical atom order, which
    reads stereo, bond orders and charges; any other two by their maximal overlay (`_overlay_order`), in which a
    mapped atom overlays only its partner, so mapping both sides atom for atom keeps an order the overlay cannot tell
    apart, such as two frames of a degenerate exchange. Atom maps, notes, stereo and conformers move with their atoms
    (`renumber_atoms`).
    """
    sides = [Chem.Mol(reactant), Chem.Mol(product)]
    if sorted(map(_element, reactant.GetAtoms())) != sorted(map(_element, product.GetAtoms())):
        raise ValueError("the two sides hold different atoms; embed endpoints of different formulas separately")
    maps = [Counter(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()) for mol in sides]
    if maps[0].keys() != maps[1].keys() or any(uses > 1 for numbers in maps for uses in numbers.values()):
        raise ValueError("give each atom-map number to one atom on each side, and the same numbers on both sides")
    named = [{atom.GetAtomMapNum(): _element(atom) for atom in mol.GetAtoms() if atom.GetAtomMapNum()} for mol in sides]
    if changed := sorted(number for number in named[0] if named[0][number] != named[1][number]):
        raise ValueError(f"atom map {changed[0]} changes element or isotope; correct the atom mapping")
    if Chem.MolToSmiles(sides[0]) != Chem.MolToSmiles(sides[1]):
        return renumber_atoms(product, _overlay_order(*sides))
    outputs = [mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"] for mol in sides]
    return renumber_atoms(product, [p for _r, p in sorted(zip(*outputs, strict=True))])


def _element(atom):
    """Return an atom's element and isotope, which every pairing keeps."""
    return atom.GetAtomicNum(), atom.GetIsotope()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactionAtoms:
    """Hold which atoms a reaction's endpoints align by and which it moves, as indices of their one atom order."""

    spectators: tuple
    reacting: tuple


def reaction_atoms(reactant, product, spectators=None, reacting=None):
    """Return the `ReactionAtoms` of the reactant and product `Species`, which share one atom order.

    Their equivalent sites are labelled as the co-embed seats them (`coembed.least_motion`). `spectators` and
    `reacting` select atoms by index or SMARTS on the reactant. An atom reacts when its charge, bonds (partner, order,
    dative direction) or specified stereo change, a metal's arrangement included (the metal and the donors of its
    `site_motions`, unless the sides are one species), when a specified configuration change turns it
    (`turned_atoms`), or when `reacting` selects it. The spectators are the selected ones, else every unchanged atom,
    hydrogens included. Identical endpoints have no reacting atoms and align their common scaffold.
    """
    graphs = [graph_with_stereo(species) for species in (reactant, product)]
    aligned, moved = (_picked(graphs[0], *row) for row in ((spectators, "spectator_atoms"), (reacting, "react_atoms")))
    if aligned == ():
        raise ValueError("spectator_atoms must contain at least one atom")
    if set(aligned or ()) & set(moved or ()):
        raise ValueError(
            "react_atoms must be outside spectator_atoms; remove reacting atoms from the spectator selection"
        )
    every = range(graphs[0].GetNumAtoms())
    bonds = {atom for atom in every if _bonds(graphs[0], atom) != _bonds(graphs[1], atom)}
    elements = [_stereo_at(species.stereo) for species in (reactant, product)]
    changed = bonds | {atom for atom in every if elements[0].get(atom) != elements[1].get(atom)}
    if changed or len(distinct([reactant, product])) > 1:  # one species twice moves no site, however it is labelled
        for metal, sites in site_motions(reactant.centres, product.centres).items():
            changed |= {metal}.union(*sites)
    stereo = (reactant.stereo, product.stereo)
    changed |= set(moved or ()) | turned_atoms(graphs, stereo, set(aligned or ()), bonds, chosen=aligned is not None)
    if aligned is None:
        aligned = tuple(atom for atom in every if atom not in changed)
    if not aligned:
        raise ValueError("coembed cannot align the endpoints: the reaction changes every atom; pass spectator_atoms=")
    reacting = tuple(a for a in every if a in changed and a not in aligned)
    return ReactionAtoms(spectators=aligned, reacting=reacting)


def _picked(graph, picked, name):
    """Return the `name` selection of reactant atoms (indices or SMARTS) as indices, or None.

    A ``{reactant atom: product atom}`` dict is refused: the product shares the reactant's order, and atom maps pair
    atoms the overlay cannot tell apart.
    """
    if picked is None:
        return None
    if isinstance(picked, (dict, str)):
        raise TypeError(f"{name} takes reactant atoms, as indices or SMARTS; pair atoms with atom maps on both sides")
    return tuple(dict.fromkeys(resolve_atom(graph, atom) for atom in picked))


def _bonds(mol, atom):
    """Return an atom's charge and its bonds as (partner, order, whether the atom donates it)."""
    found = mol.GetAtomWithIdx(atom)
    bonds = sorted(
        (
            b.GetOtherAtomIdx(atom),
            b.GetBondTypeAsDouble(),
            b.GetBondType() == Chem.BondType.DATIVE and b.GetBeginAtomIdx() == atom,
        )
        for b in found.GetBonds()
    )
    return found.GetFormalCharge(), bonds


def _stereo_at(stereo):
    """Return each atom's specified stereo elements, ``{atom: {(kind, atoms, code)}}``."""
    out = defaultdict(set)
    for element in stereo.elements:
        named = (element.kind, tuple(sorted(element.atoms)), element.code)
        for atom in named[1]:
            out[atom].add(named)
    return out


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
    skeletons = [_without(skeleton(mol), changed) for mol, changed in ((reactant, r_changed), (product, p_changed))]
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
        f"the endpoints overlay only if over {_MAX_SPARE} extra bonds break; check or add atom maps on both sides"
    )


def _one_reaction(reactant, overlays):
    """Return the first overlay's order, or raise when tied overlays are different reactions.

    A reaction is its ITS graph: the kept, broken and formed bonds, each its own bond type. Two overlays, or an
    overlay and its formed bonds moved by the remainder's symmetry (`_images`), are one reaction when their ITS
    graphs share a canonical SMILES. Canonical ranks stand in for symmetry orbits; on a rare regular graph they
    merge too much, which raises rather than picks.
    """
    graph = skeleton(reactant)
    reactions = {}
    for (broken, r_order), (dropped, p_order) in overlays:
        onto = dict(zip(p_order, r_order, strict=True))  # product atom -> reactant atom
        formed = [(onto[i], onto[j]) for i, j in dropped]
        remainder = _without(graph, broken)
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


def _without(skeleton, bonds):
    """Return a copy of `skeleton` without `bonds`, ready to canonicalise."""
    remainder = Chem.RWMol(skeleton)
    for i, j in bonds:
        remainder.RemoveBond(i, j)
    remainder.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(remainder)
    return remainder


def site_motions(left, right):
    """Return ``{metal: sites}``, each metal's sites that move between the centres `left` and `right`.

    Both share one atom order. A site both sides bind keeps its place when a relabelling of the polyhedron
    (`seating_frames`) sends its vertex and winding in `right` to those in `left`; the relabelling taken moves the
    fewest sites, each a frozenset of atoms, then the shortest summed chord between their two slot directions, the
    given frame first. A metal that changes polyhedron moves every site when it keeps them all, and none when its sites
    change, which is a bond change; no coordinate is read.
    """
    seated = {centre.atom: centre for centre in right}
    pairs = [(one, seated[one.atom]) for one in left if one.atom in seated]
    moved = [(one, other) for one, other in pairs if (one.geometry, one.vertices) != (other.geometry, other.vertices)]
    motions = {one.atom: _motion(one, other) for one, other in moved}
    return {metal: sites for metal, sites in motions.items() if sites}


def _motion(one, other):
    """Return the sites that move between two arrangements `one` and `other` of one metal (`site_motions`)."""
    slots = [
        {frozenset(site.atoms): (k, site.winding) for k, site in enumerate(c.vertices) if site is not VACANT}
        for c in (one, other)
    ]
    shared = slots[0].keys() & slots[1].keys()
    if one.geometry != other.geometry:
        return frozenset(shared) if slots[0].keys() == slots[1].keys() else frozenset()
    dirs = vertex_dirs(one.geometry)
    best, rank = frozenset(), None
    for q in sorted(seating_frames(dirs, tuple(sorted(k for k, _winding in slots[1].values())))):
        seats = {site: (q[slots[1][site][0]], slots[1][site][1]) for site in shared}
        sites = frozenset(site for site in shared if seats[site] != slots[0][site])
        chord = sum(math.dist(dirs[slots[0][site][0]], dirs[seats[site][0]]) for site in sites)
        if rank is None or (len(sites), round(chord, 6)) < rank:
            best, rank = sites, (len(sites), round(chord, 6))
    return best


class Inversion(NamedTuple):
    """Hold how a point centre inverts: the atoms that move, and the three neighbours whose plane they reflect through.

    `plane` is None when no acyclic branch can carry the inversion.
    """

    moving: set
    plane: list | None


def bond_sides(mol, pair):
    """Return the atoms on each side of bond `pair`, or None when the bond is in a ring."""
    cut = Chem.RWMol(mol)
    cut.RemoveBond(*pair)
    fragments = Chem.GetMolFrags(cut)
    sides = [set(next(part for part in fragments if atom in part)) for atom in pair]
    return None if pair[0] in sides[1] else sides


def point_inversion_branches(mol, centre):
    """Return separable inversion groups and their stationary neighbour planes, excluding metal-containing groups."""
    neighbours = [atom.GetIdx() for atom in mol.GetAtomWithIdx(centre).GetNeighbors()]
    if len(neighbours) == 3:  # noqa: PLR2004 - a pyramid's neighbours define its inversion plane
        return [Inversion(moving={centre}, plane=neighbours)]
    branches = []
    for neighbour in neighbours if len(neighbours) == 4 else ():  # noqa: PLR2004 - explicit tetrahedron
        cut = Chem.RWMol(mol)
        cut.RemoveBond(centre, neighbour)
        branch = set(next(part for part in Chem.GetMolFrags(cut) if neighbour in part))
        if centre not in branch and not branch.intersection(metal_indices(mol)):
            branches.append(
                Inversion(moving={centre, *branch}, plane=[atom for atom in neighbours if atom != neighbour])
            )
    return branches


def turned_atoms(graphs, stereo, held, bonds, *, chosen):
    """Return the atoms a specified configuration change turns: one group per changed element, across both endpoints.

    An element both endpoints specify with other codes, on atoms whose `bonds` stay, can turn any of its separable
    groups: a centre's branches or either side of an axis or double bond. The same group on both endpoints is one
    candidate, and the one moving fewest `held`, heavy and total atoms turns. An atropisomer axis in a ring has no
    separable side, so without `chosen` spectators it raises.
    """
    keyed = [{(e.kind, tuple(sorted(e.atoms))): e for e in side.elements} for side in stereo]
    turned = set()
    for key in keyed[0].keys() & keyed[1].keys():
        pair = keyed[0][key], keyed[1][key]
        if pair[0].code == pair[1].code or bonds & set(key[1]):
            continue
        groups = defaultdict(set)
        for graph, element in zip(graphs, pair, strict=True):
            for label, group in _turning_groups(graph, element, chosen=chosen):
                groups[label] |= group
        heavy = {atom.GetIdx() for atom in graphs[0].GetAtoms() if atom.GetAtomicNum() != 1}
        if groups:
            turned |= min(groups.values(), key=lambda g: (len(g & held), len(g & heavy), len(g), tuple(sorted(g))))
    return turned


def _turning_groups(mol, element, *, chosen):
    """Return ``(label, atoms)`` for each group that can turn `element`, labelled alike on either endpoint."""
    if element.kind == "point":
        centre = element.atoms[0]
        neighbours = {atom.GetIdx() for atom in mol.GetAtomWithIdx(centre).GetNeighbors()}
        return [
            (tuple(sorted(neighbours - set(item.plane))), item.moving) for item in point_inversion_branches(mol, centre)
        ]
    sides = bond_sides(mol, element.atoms)
    if sides is None:
        if element.kind == "axis" and not chosen:
            raise ValueError(
                "an atropisomeric axis lies in a ring; name the moving atoms with react_atoms= and the rest "
                "spectator_atoms="
            )
        return []
    return [((atom,), side) for atom, side in zip(element.atoms, sides, strict=True)]
