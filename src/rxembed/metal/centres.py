"""Normalise metal graphs, derive the engine's surrogate graphs and hold the centre records.

The engine holds a metal by distance and angle constraints alone, its bonds removed and a UFF-typeable surrogate atom in
its place, so the whole path is plain RDKit and UFF.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from rdkit import Chem, rdBase

from rxembed.geometry.constraints import FrozenDict
from rxembed.graph import (
    COORDINATION_METALS,
    ETA2,
    SURROGATE_SANITIZE,
    clear_labile_donor_stereo,
    haptic_sites,
    ligand_degree,
    materialise_phantoms,
    metal_indices,
    open_valence,
    surrogate_all_metals,
)
from rxembed.metal.polyhedra import SLOT_BOND_PROP, VACANT
from rxembed.stereo import StereoGraph, clear_ez
from rxembed.utils import flat_ranks, remove_bond, require_mol

logger = logging.getLogger("rxembed.metal")  # shared by the metal modules, under "rxembed" (`set_verbose`)

_PT = Chem.GetPeriodicTable()
EPS_LEN = 1e-9  # a donor sitting on the metal has no direction, so the sphere cannot be read at all


@dataclass(frozen=True, slots=True, kw_only=True)
class Site:
    """Name one coordination site by its real donor atoms: one for a sigma donor, the face for a haptic one.

    `winding` is a haptic face's specified winding; a sigma donor has none.
    """

    atoms: tuple[int, ...]
    winding: str = ""


class Centre(NamedTuple):
    """Identify one real metal and its coordination arrangement: a `Site` per vertex, `VACANT` at a vacancy."""

    atom: int
    atomic_num: int
    charge: int
    geometry: str = ""
    vertices: tuple = ()
    hand: str = ""


def seated_centre(centre, geometry, vertices, haptic, winding, hand):
    """Return `centre` seated in `geometry` at materialized `vertices`, its faces wound as `winding` maps them."""
    sites = tuple(
        VACANT if donor is VACANT else Site(atoms=tuple(haptic.get(donor, (donor,))), winding=winding.get(donor, ""))
        for donor in vertices
    )
    return centre._replace(geometry=geometry, vertices=sites, hand=hand)


class Sites(NamedTuple):
    """Hold one centre with transient centroid indices: per-slot vertices, haptic faces, windings, donors."""

    vertices: tuple
    haptic: FrozenDict  # centroid index -> face atoms
    winding: FrozenDict  # centroid index -> winding sign
    donors: tuple


def centroid_indices(mol, faces):
    """Return each of `faces` keyed to its centroid index: after the real atoms of `mol`, in sorted face order.

    A face is keyed by its sorted atoms, with its metal first where several metals share a call, so an index never
    depends on how the donors are seated.
    """
    return {face: mol.GetNumAtoms() + i for i, face in enumerate(sorted(faces))}


def materialized_states(mol, centres):
    """Return each centre's frozen `Sites` by metal, each face of several atoms at its `centroid_indices` index.

    A site of one atom is that atom's vertex.
    """
    index = centroid_indices(
        mol,
        [(c.atom, s.atoms) for c in centres for s in c.vertices if s is not VACANT and len(s.atoms) > 1],
    )
    out = {}
    for centre in centres:
        vertices, haptic, winding = [], {}, {}
        for site in centre.vertices:
            if site is VACANT or len(site.atoms) == 1:
                vertices.append(VACANT if site is VACANT else site.atoms[0])
                continue
            dummy = index[centre.atom, site.atoms]
            vertices.append(dummy)
            haptic[dummy] = site.atoms
            if site.winding:
                winding[dummy] = site.winding
        donors = tuple(atom for site in centre.vertices if site is not VACANT for atom in site.atoms)
        out[centre.atom] = Sites(tuple(vertices), FrozenDict(haptic), FrozenDict(winding), donors)
    return FrozenDict(out)


def state_with_winding(centre, vertices, winding):
    """Return `centre` carrying `winding` on its materialized haptic vertices."""
    sites = tuple(
        VACANT if site is VACANT else Site(atoms=site.atoms, winding=winding.get(vertices[position], ""))
        for position, site in enumerate(centre.vertices)
    )
    return centre._replace(vertices=sites)


def vertex_atom(haptic, v):
    """Return a real atom for a coordination vertex; a centroid dummy maps to its ring.

    The bondless dummy has no fragment or backbone, so without this a tethered face (an ansa Cp) would read as a
    separate ligand.
    """
    return haptic[v][0] if haptic and v in haptic else v


def donor_bonds(mol):
    """Return donor-to-metal edges in their chemical graph's bond order."""
    metals = set(metal_indices(mol))
    return tuple(
        (bond.GetOtherAtomIdx(metal), metal)
        for bond in mol.GetBonds()
        for metal in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if metal in metals and bond.GetOtherAtomIdx(metal) not in metals
    )


def _bridge_notes(mol, metals):
    """Put metal-indexed bridge slots on the donor before its coordination bonds are reordered or removed."""
    for atom in mol.GetAtoms():
        bonds = [bond for bond in atom.GetBonds() if bond.GetOtherAtomIdx(atom.GetIdx()) in metals]
        if len(bonds) > 1 and all(bond.HasProp(SLOT_BOND_PROP) for bond in bonds):
            bonds.sort(key=lambda bond: bond.GetOtherAtomIdx(atom.GetIdx()))
            atom.SetProp("atomNote", ";".join(bond.GetProp(SLOT_BOND_PROP) for bond in bonds))


def chemical_graph(mol):
    """Copy a canonical metal graph into the chemical bond order a `Species` holds.

    Metal bonds follow ligand bonds in metal and neighbour order. Donor hydrogen counts and point tags survive the
    reordering, and donors that lose their stereo carriers get their tags back. A coordination-locked E/Z belongs to the
    arrangement, not the ligand stereo. The metal's hybridization flags follow the surrogate's typing and choose no
    arrangement.
    """
    metals = metal_indices(mol)
    if not metals:
        return Chem.Mol(mol)
    numerical = surrogate_all_metals(mol)
    rw = Chem.RWMol(mol)
    clear_ez(rw, StereoGraph(mol, metals).locked)
    _bridge_notes(rw, metals)
    edges = [(atom.GetIdx(), metal) for metal in metals for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
    for donor, metal in edges:
        remove_bond(rw, donor, metal)
        atom = rw.GetAtomWithIdx(donor)
        clear_labile_donor_stereo(atom)
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    for metal in metals:
        atom = rw.GetAtomWithIdx(metal)
        atom.SetNoImplicit(True)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    hands = {donor: rw.GetAtomWithIdx(donor).GetChiralTag() for donor, _metal in edges}
    Chem.SanitizeMol(rw, SURROGATE_SANITIZE, catchErrors=True)
    rw.UpdatePropertyCache(strict=False)
    for metal in metals:
        rw.GetAtomWithIdx(metal).SetHybridization(numerical.GetAtomWithIdx(metal).GetHybridization())
    for donor, tag in hands.items():
        rw.GetAtomWithIdx(donor).SetChiralTag(tag)
    for donor, metal in edges:
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    out = rw.GetMol()
    out.ClearComputedProps()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def engine_graph(mol):
    """Project a chemical graph onto the bondless surrogate graph the bounds and coordination models use.

    Atom order and conformers are kept. Bridge notes move onto donors before their bonds go; a haptic projection
    refreshes rings with its dative bonds counted.
    """
    mol = Chem.Mol(mol)
    metals = metal_indices(mol)
    if not metals:
        return mol
    clear_ez(mol, StereoGraph(mol, metals).locked)
    _bridge_notes(mol, metals)
    base = surrogate_all_metals(mol)
    if any(
        len(site) > 1
        for metal in metals
        for site in haptic_sites(base, [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()])
    ):
        base.ClearComputedProps()
        base.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(base, includeDativeBonds=True)
    return base


def seed_graph(chemical, working, carried):
    """Build the DG graph that carries donor stereo: the working conformers with only the `carried` metal bonds.

    The real metals and those bonds let RDKit read donor chirality and terminal linearity.
    """
    rw = Chem.RWMol(chemical)
    for donor, metal in donor_bonds(chemical):
        rw.RemoveBond(donor, metal)
    for donor, metal in carried:
        if rw.GetBondBetweenAtoms(donor, metal) is None:
            rw.AddBond(donor, metal, Chem.BondType.DATIVE)
    metals = set(metal_indices(chemical))
    for atom in working.GetAtoms():
        if atom.GetIdx() not in metals:
            rw.ReplaceAtom(atom.GetIdx(), Chem.Atom(atom))
    out = rw.GetMol()
    out.RemoveAllConformers()
    for conf in working.GetConformers():
        out.AddConformer(Chem.Conformer(conf), assignId=False)
    out.ClearComputedProps()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


class Collapsed(NamedTuple):
    """Hold `donors` with each haptic face collapsed to one centroid vertex (`collapse_haptic`)."""

    mol: Chem.Mol  # the Mol with a bond-less carbon centroid appended per face, at the ring centroid
    sites: list  # sigma donors, then centroid indices
    haptic: dict  # centroid index -> face atoms


def collapse_haptic(mol, donors):
    """Return `donors` of `mol` with each haptic face collapsed to one centroid vertex; sigma donors pass through.

    Indices are unchanged, and the centroid is scaffolding stored in no Mol. A Mol without a face is returned untouched.
    """
    sites = haptic_sites(mol, donors)
    faces = [s for s in sites if len(s) > 1]  # a face is any mutually-bonded donor group, eta2 included
    vertices = [d for site in sites if len(site) == 1 for d in site]
    if not faces:
        return Collapsed(mol, vertices, {})
    haptic = {dummy: face for face, dummy in centroid_indices(mol, faces).items()}
    return Collapsed(materialise_phantoms(mol, haptic), [*vertices, *haptic], haptic)


class Ligand(NamedTuple):
    """Hold one ligand of a metal complex, as a standalone Mol plus which of its atoms coordinate."""

    mol: Chem.Mol  # standalone, carrying its own copy of the complex's conformer
    donors: dict  # {metal index in the original complex: [donor indices in `mol`]}; a bridging ligand has 2 keys
    atoms: tuple  # this ligand's indices in the original complex, positionally matching `mol`


def ligands(mol):
    """Return the complex's ligands, each separated from the metals it binds.

    Donors are reported per metal, so a bridging ligand says so: on a Mn/Fe complex its backbone is kappa3 to one metal
    and kappa5 to the other, not kappa8 to neither. A ligand is a connected fragment, so a hydrogen bond read as a bond
    fuses two ligands.
    """
    metals = metal_indices(require_mol(mol, "rx.ligands"))
    if not metals:
        raise ValueError("no metal centre found: this reads a metal complex's coordination sphere")
    bound = {m: {n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()} for m in metals}
    base = surrogate_all_metals(canonical_metal_graph(mol))
    mapping = []
    frags = Chem.GetMolFrags(base, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)

    out = []
    for frag, atoms in zip(frags, mapping, strict=True):
        if set(atoms) & set(metals):  # a surrogated metal is not a ligand of itself
            continue
        donors = {m: [k for k, a in enumerate(atoms) if a in ds] for m, ds in bound.items()}
        out.append(Ligand(frag, {m: d for m, d in donors.items() if d}, tuple(atoms)))
    return out


def ligand_valence(atom):
    """Return the bond orders `atom` spends on its ligand side, hydrogens included, metal bonds excluded.

    It reads the same on the complex and on its bondless surrogate. Hydrogens count, since they tell an aqua from an
    oxo.
    """
    return atom.GetTotalNumHs() + sum(
        b.GetBondTypeAsDouble()
        for b in atom.GetBonds()
        if b.GetOtherAtom(atom).GetAtomicNum() not in COORDINATION_METALS
    )


def _ionic_donor_charge(donor, haptic):
    """Return the formal charge needed after removing a covalent M-L valence contribution."""
    if donor.GetFormalCharge() or donor.GetNumRadicalElectrons() or donor.GetIdx() in haptic:
        return 0
    default = _PT.GetDefaultValence(donor.GetAtomicNum())
    deficit = default - ligand_valence(donor)
    if default <= 0 or deficit <= 0 or not np.isclose(deficit, round(deficit)):
        return 0
    return -round(deficit)


def _canonicalise_delocalised_charge(mol):
    """Move each delocalised -1 onto the best metal-bound atom its conjugated system lets carry it.

    A resonance move redraws bond orders and one charge within a conjugated system, X=O and X#Y included, and never
    moves a hydrogen; RDKit's `ResonanceMolSupplier` moves no charge across X=O, so the paths are rxembed's. A charge
    moves only onto an uncharged metal-bound atom and only to a better seat, so the moves end; a charge no metal-bound
    atom can take stays. A move never redraws a bond whose E/Z the Mol specifies beyond its conformer, and a form that
    changes any atom's aromaticity or any metal's haptic sites is skipped. Each move freezes the H counts along its
    path, since `SanitizeMol` would otherwise recompute implicit H.
    """
    metals = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS)
    bound = frozenset(n.GetIdx() for metal in metals for n in mol.GetAtomWithIdx(metal).GetNeighbors()) - metals
    anion = any(atom.GetFormalCharge() == -1 and atom.GetIdx() not in metals for atom in mol.GetAtoms())
    kekule = _kekule(mol) if bound and anion else None
    held_ez = frozenset(StereoGraph(mol, metals).missed_ez) if kekule is not None else frozenset()
    out, drawn = Chem.Mol(mol), None
    while kekule is not None and (seats := _open_seats(kekule, metals, bound, held_ez)):
        drawn = drawn or _Drawn.of(mol, metals, held_ez)  # ranked only once a charge can move
        if (kekule := _better_form(kekule, seats, drawn)) is not None:
            out = kekule.GetMol()
            Chem.SanitizeMol(out)
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class _Drawn:
    """Hold what a charge move compares with the drawn form `mol`: its flat ranks, aromaticity and haptic sites.

    `held_atoms` holds the atoms of the double bonds whose E/Z `mol` specifies beyond its conformer; `donors` the
    non-metal neighbours of each of `metals`, and `sites` their haptic sites in `mol`.
    """

    mol: Chem.Mol
    metals: frozenset[int]
    held_atoms: frozenset[int]
    tiers: tuple[int, ...]
    order: tuple[int, ...]
    aromatic: tuple[bool, ...]
    donors: tuple[tuple[int, ...], ...]
    sites: tuple[list, ...]

    @classmethod
    def of(cls, mol, metals, held_ez):
        """Return the record of `mol`, `metals` its metals and `held_ez` its `StereoGraph.missed_ez` pairs."""
        donors = tuple(
            tuple(n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors() if n.GetIdx() not in metals)
            for metal in metals
        )
        return cls(
            mol=mol,
            metals=metals,
            held_atoms=frozenset().union(*held_ez),
            tiers=tuple(flat_ranks(mol)),
            order=tuple(flat_ranks(mol, break_ties=True)),
            aromatic=tuple(atom.GetIsAromatic() for atom in mol.GetAtoms()),
            donors=donors,
            sites=tuple(haptic_sites(mol, atoms) for atoms in donors),
        )

    def holders(self, atom):
        """Return the metals bound to `atom`."""
        return {n.GetIdx() for n in self.mol.GetAtomWithIdx(atom).GetNeighbors()} & self.metals

    def beats(self, seat, anion):
        """Return whether metal-bound `seat` is a better place than `anion` for its -1.

        Any seat beats an atom no metal holds; between bound atoms a higher flat rank wins. Seats the flat graph cannot
        tell apart trade the charge by atom index only between different metals, since `cxsmiles` writes symmetric
        metals as the least string over their swap. On one metal the drawn seat stays: a centre whose neighbours only
        the Lewis form tells apart, such as a ylide S, takes its label from that seat.
        """
        held, holding = self.holders(seat), self.holders(anion)
        mine, theirs = (bool(held), self.tiers[seat]), (bool(holding), self.tiers[anion])
        return mine > theirs or (mine == theirs and self.order[seat] > self.order[anion] and held != holding)


_RAISE = {Chem.BondType.SINGLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.TRIPLE}
_LOWER = {Chem.BondType.DOUBLE: Chem.BondType.SINGLE, Chem.BondType.TRIPLE: Chem.BondType.DOUBLE}
_EZ_DIRECTIONS = (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)


def _kekule(mol):
    """Return a Kekule `RWMol` copy of `mol`, or None when RDKit cannot kekulize it."""
    kekule = Chem.RWMol(mol)
    try:
        with rdBase.BlockLogs():
            Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Chem.MolSanitizeException:
        return None
    return kekule


def _resonance_paths(kekule, anion, held_ez):
    """Return, for each atom that can take the charge of `anion`, the path that moves it there.

    A path leaves `anion` by raising one bond order, then alternately lowers and raises ligand bonds of the Kekule form,
    which keeps each inner atom's valence (Berge's alternating path). It never moves a bond of `held_ez`. Only simple
    paths are kept; their atoms are the conjugated system.
    """
    seen, paths, frontier = {(anion, True)}, {}, [[anion]]
    for path in frontier:
        up = len(path) % 2 == 1
        for bond in kekule.GetAtomWithIdx(path[-1]).GetBonds():
            atom = bond.GetOtherAtomIdx(path[-1])
            moves = bond.GetBondType() in (_RAISE if up else _LOWER) and frozenset((path[-1], atom)) not in held_ez
            if not moves or atom in path or (atom, not up) in seen:
                continue
            seen.add((atom, not up))
            frontier.append([*path, atom])
            if not up:
                paths.setdefault(atom, frontier[-1])
    return paths


def _open_seats(kekule, metals, bound, held_ez):
    """Return ``{anion: {seat: path}}``: each uncharged atom of `bound` a ligand -1 of `kekule` reaches."""
    charged = {atom.GetIdx() for atom in kekule.GetAtoms() if atom.GetFormalCharge()}
    anions = [atom for atom in charged - metals if kekule.GetAtomWithIdx(atom).GetFormalCharge() == -1]
    seats = {
        anion: {
            seat: path for seat, path in _resonance_paths(kekule, anion, held_ez).items() if seat in bound - charged
        }
        for anion in anions
    }
    return {anion: found for anion, found in seats.items() if found}


def _better_form(kekule, seats, drawn):
    """Return `kekule` with one -1 moved to the best of its `seats` that beats it (`_Drawn.beats`), or None."""
    for anion in sorted(seats, key=lambda atom: -drawn.order[atom]):
        better = [seat for seat in seats[anion] if drawn.beats(seat, anion)]
        for seat in sorted(better, key=lambda atom: -drawn.order[atom]):
            if (moved := _moved_charge(kekule, seats[anion][seat], drawn)) is not None:
                return moved
    return None


def _moved_charge(kekule, path, drawn):
    """Return a Kekule copy of `kekule` with the -1 moved from ``path[0]`` to ``path[-1]``, or None if it may not.

    Each bond along `path` moves one order, up first. Bond directions that carry no E/Z are cleared, so a new double
    bond reads none from them; a move that would give a new double bond a specified E/Z, a form RDKit rejects, or other
    aromaticity or haptic sites than `drawn` is refused.
    """
    trial = Chem.RWMol(kekule)
    for atom in map(trial.GetAtomWithIdx, path):  # a resonance move never changes a hydrogen count
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        for bond in atom.GetBonds():  # a direction beside a held E/Z double bond carries that E/Z
            if (
                bond.GetBondDir() in _EZ_DIRECTIONS
                and not {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & drawn.held_atoms
            ):
                bond.SetBondDir(Chem.BondDir.NONE)
    for step, (begin, end) in enumerate(itertools.pairwise(path)):
        bond = trial.GetBondBetweenAtoms(begin, end)
        bond.SetBondType((_LOWER if step % 2 else _RAISE)[bond.GetBondType()])
        bond.SetStereo(Chem.BondStereo.STEREONONE)  # a conformer's own E/Z, read from it again on the moved form
    trial.GetAtomWithIdx(path[0]).SetFormalCharge(0)
    trial.GetAtomWithIdx(path[-1]).SetFormalCharge(-1)
    directed = {
        atom
        for bond in trial.GetBonds()
        if bond.GetBondDir() in _EZ_DIRECTIONS
        for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    }
    if any({begin, end} <= directed for begin, end in zip(path[::2], path[1::2], strict=False)):
        return None
    probe = trial.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(probe)
    except Chem.MolSanitizeException:
        return None
    if tuple(atom.GetIsAromatic() for atom in probe.GetAtoms()) != drawn.aromatic:
        return None
    return (
        trial
        if all(haptic_sites(probe, atoms) == s for atoms, s in zip(drawn.donors, drawn.sites, strict=True))
        else None
    )


def _donor_charges(mol, metals, haptic):
    """Return ``{donor: (metal, charge)}`` for each donor `canonical_metal_graph` charges; a bridging one raises."""
    charged = {}
    for donor in mol.GetAtoms():
        adjacent = [neighbor.GetIdx() for neighbor in donor.GetNeighbors() if neighbor.GetIdx() in metals]
        if not adjacent or not (charge := _ionic_donor_charge(donor, haptic)):
            continue
        if len(adjacent) > 1:
            raise ValueError(
                f"neutral bridging donor {donor.GetSymbol()}{donor.GetIdx()} cannot say which of metals "
                f"{adjacent} balances it; give the charges"
            )
        charged[donor.GetIdx()] = (adjacent[0], charge)
    return charged


def canonical_metal_graph(mol):
    """Return a copy with every M-L bond in the canonical ionic donor-to-metal form.

    The total charge, charge magnitudes and non-resonant charges stay as given; the only charge moved is a delocalised
    -1, put on one atom of its conjugated system as a convention. A neutral underfilled sigma donor takes the charge its
    ligand-side valence implies, balanced on the metal; a neutral bridge cannot say which metal balances it, so it needs
    an explicit charge. A graph with open bond orders or a multi-centre cage is refused. A native SP, TB or OH tag
    counts its metal's bonds in order, so such a metal redraws all its M-L bonds or none.

    Each verb runs it once on the Mol it is given and the readers trust the result; it is idempotent. A charge never
    moves across a specified E/Z the conformer does not show, so a graph read with another conformer is read afresh.
    """
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    reject_boron_cages(rw)
    metals = {atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    faces = [
        site
        for metal in metals
        for site in haptic_sites(
            rw,
            [
                neighbor.GetIdx()
                for neighbor in rw.GetAtomWithIdx(metal).GetNeighbors()
                if neighbor.GetIdx() not in metals
            ],
            pairs=False,  # Lewis bookkeeping (donor charge balancing): read only the pi-seeded face rule.
        )
        if len(site) > 1
    ]
    charged = _donor_charges(rw, metals, {donor for face in faces for donor in face})
    reject_open_valence(rw, metals, faces)

    rows = []  # each M-L bond in bond order, and whether it is already donor-to-metal
    for bond in rw.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin in metals) == (end in metals):
            continue  # not an M-L bond
        donor, metal = (begin, end) if end in metals else (end, begin)
        note = bond.GetProp(SLOT_BOND_PROP) if bond.HasProp(SLOT_BOND_PROP) else None
        rows.append((donor, metal, note, bond.GetBondType() == Chem.BondType.DATIVE and end in metals))
    tagged = {
        metal for _d, metal, _n, dative in rows if not dative and rw.GetAtomWithIdx(metal).HasProp("_chiralPermutation")
    }
    replace = [(donor, metal, note) for donor, metal, note, dative in rows if not dative or metal in tagged]

    for donor, (metal, charge) in charged.items():
        donor_atom, metal_atom = rw.GetAtomWithIdx(donor), rw.GetAtomWithIdx(metal)
        old_donor, old_metal = donor_atom.GetFormalCharge(), metal_atom.GetFormalCharge()
        donor_atom.SetFormalCharge(charge)
        metal_atom.SetFormalCharge(old_metal - charge)
        logger.debug(
            "metal graph: charge %s%d %+d->%+d, %s%d %+d->%+d",
            donor_atom.GetSymbol(),
            donor,
            old_donor,
            charge,
            metal_atom.GetSymbol(),
            metal,
            old_metal,
            old_metal - charge,
        )
    for donor, metal, note in replace:
        remove_bond(rw, donor, metal)
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
        if note is not None:
            rw.GetBondBetweenAtoms(donor, metal).SetProp(SLOT_BOND_PROP, note)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    if replace:
        # a covalent R2PH-M bond can leave a cached P radical; recompute only on donors made dative, given radicals kept
        probe = Chem.Mol(out)
        result = Chem.SanitizeMol(probe, Chem.SanitizeFlags.SANITIZE_FINDRADICALS, catchErrors=True)
        if result == Chem.SanitizeFlags.SANITIZE_NONE:
            for donor in {donor for donor, _metal, _note in replace}:
                out.GetAtomWithIdx(donor).SetNumRadicalElectrons(probe.GetAtomWithIdx(donor).GetNumRadicalElectrons())
    return _canonicalise_delocalised_charge(out)


def symmetric_metals(mol):
    """Return each class of symmetric metals of `mol`, with the order of a symmetry taking the first to each.

    Each metal is marked in turn on a stereo-free copy before its canonical graph is drawn, so a charge that symmetric
    donors share is seated relative to the mark, not by atom index; equal canonical SMILES make two metals symmetric.
    """
    metals = metal_indices(mol)
    if len(metals) <= 1:
        return []
    written = defaultdict(dict)
    mark = 1 + max(atom.GetIsotope() for atom in mol.GetAtoms())  # an isotope no atom carries
    for metal in metals:
        marked = Chem.Mol(mol)
        marked.RemoveAllConformers()
        Chem.RemoveStereochemistry(marked)
        marked.GetAtomWithIdx(metal).SetIsotope(mark)
        marked = canonical_metal_graph(marked)
        key = Chem.MolToSmiles(marked)  # sets the output order read next
        written[key][metal] = marked.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
    classes = []
    for group in written.values():
        first = next(iter(group.values()))
        if len(group) > 1:
            classes.append(
                {m: [target for _, target in sorted(zip(first, order, strict=True))] for m, order in group.items()}
            )
    return classes


def reject_open_valence(mol, metals, faces):
    """Reject a graph that leaves bond orders open, since it is not one species.

    Two bonded atoms with open valence hold a pi bond, charge or radical pair the graph does not write; a lone one is
    the radical a bracket atom gives (``[CH3]``). Metals and their donors are settled, except the atoms of a face of
    three or more (`faces`), which `graph.haptic_sites` seeds from that open valence.
    """
    open_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if open_valence(atom)}
    bound = {n.GetIdx() for metal in metals for n in mol.GetAtomWithIdx(metal).GetNeighbors()}
    seeded = {atom for face in faces if len(face) > ETA2 for atom in face}
    unsettled = [
        f"{atom.GetSymbol()}{atom.GetIdx()}"
        for atom in mol.GetAtoms()
        if atom.GetIdx() in open_atoms - metals - (bound - seeded)
        and any(n.GetIdx() in open_atoms for n in atom.GetNeighbors())
    ]
    if unsettled:
        more = f" and {len(unsettled) - 4} more" if len(unsettled) > 4 else ""  # noqa: PLR2004  name a few
        raise ValueError(
            f"this graph leaves bond orders or charges open at {', '.join(unsettled[:4])}{more}; specify them or "
            "use rx.read_graph"
        )


def reject_boron_cages(mol):
    """Reject a period-2 atom bonded past its octet, the multi-centre cage case.

    Li, Be and B fill their octet with four two-centre bonds; a fifth or sixth sigma bond exists only through
    multi-centre bonding (a closo-borane cage), outside rxembed's two-centre donor model.
    """
    mol.UpdatePropertyCache(strict=False)  # ligand_degree needs implicit valence; callers may hand a raw graph
    vertices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if _PT.GetRow(atom.GetAtomicNum()) == 2  # noqa: PLR2004  period 2: Li through Ne
        and _PT.GetNOuterElecs(atom.GetAtomicNum()) < 4  # noqa: PLR2004  fewer than carbon's four
        and ligand_degree(atom) > 4  # noqa: PLR2004  more sigma bonds than an octet allows
    ]
    if vertices:
        raise ValueError(
            f"multi-centre cage vertex atom(s) {sorted(vertices)} lie outside the two-centre donor model; give a "
            "donor graph"
        )
