"""Enumerate the ligand stereoisomers a graph has under a `stereo=` request; a metal's arrangement is never one."""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from rxembed.graph import haptic_sites, metal_indices
from rxembed.stereo import (
    ATROP_STEREO,
    LigandStereo,
    StereoGraph,
    assign_atrop_from_3d,
    bond_references,
    clear_atrop,
    clear_ez,
    cumulene_component,
    defined_stereo_label,
    potential_stereo,
    stereo_from_3d,
)
from rxembed.utils import mirror_tag, with_conformers, without_atom_maps

logger = logging.getLogger("rxembed.stereo")  # under the "rxembed" tree `set_verbose` configures
_REQUEST_MODES = {"unassigned", "racemic", "free", "preserve", "all", "invert"}
_ENUMERATING = {"unassigned", "racemic", "all"}  # the modes that list stereoisomers, which one species cannot take
_REQUEST_KINDS = {"point", "ez", "axial", "planar", "helical", "locked", "default"}
_KIND_MODES = {"free", "preserve", "invert", "racemic"}
_SELECTOR = re.compile(r"([A-Z][a-z]?)(\d+)")


def _mode_table(kinds, has_geometry):
    """Return each kind's mode under the `stereo=` keys `kinds`, with or without a conformer.

    A kind not given takes ``default``; otherwise a geometry keeps what it shows and a graph enumerates what is
    unassigned. ``locked`` sets a bound diene's s-cis/s-trans class. A geometry enumerates that class by default, unlike
    a locked donor hand, since it adds few isomers, and inverts it where the faces invert.
    """
    default = kinds.get("default", "preserve" if has_geometry else "unassigned")
    table = {kind: kinds.get(kind, default) for kind in ("point", "ez", "axial", "planar", "helical")}
    side = table["planar"]
    return table | {
        "diene": kinds.get("locked", "racemic" if has_geometry and side not in ("free", "invert") else side)
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class StereoRequest:
    """Hold one validated ``stereo=`` request, parsed once at the facade.

    A mode string applies to every kind, with 'all' read as 'preserve'. A dict sets a mode per kind (point, ez, axial,
    planar, helical, locked, default) or per atom selector such as "N5", which `on` resolves on the Mol each verb reads.
    Without a mode, a geometry keeps what it shows and a graph enumerates what is unassigned. 'invert' asks for the
    mirror image: every hand inverted, each metal's included, and every E/Z kept, which a mirror never changes; ``{'ez':
    'invert'}`` flips E/Z.
    """

    given: str | dict | None  # as passed; refusals and messages quote it
    graph: dict  # {kind: mode} on a source without a conformer, a bound diene's as 'diene' (`_mode_table`)
    geometry: dict  # {kind: mode} on a source with one
    locked: str | None  # the mode ``{'locked': ...}`` gives a coordination-locked donor hand; None when not given
    mirror: bool  # the string 'invert', which mirrors each metal's hand too
    selectors: dict  # {selector as given: (element symbol, atom index, mode)} a dict names
    atoms: dict = field(default_factory=dict)  # {atom index: mode} the selectors name on the Mol read (`on`)

    @classmethod
    def parse(cls, stereo, *, one_species=False):
        """Return `stereo` validated; a request for `one_species` (an embed's) refuses a mode that enumerates."""
        if stereo is None or (isinstance(stereo, str) and stereo in _REQUEST_MODES):
            if one_species and stereo in _ENUMERATING:
                raise ValueError(
                    f"stereo={stereo!r} enumerates stereoisomers; embed rx.isomers(source, stereo=...) instead"
                )
            kinds = {} if stereo is None else {"default": "preserve" if stereo == "all" else stereo}
            if stereo == "invert":
                kinds["ez"] = "preserve"  # a mirror never changes an E/Z
            selectors = {}
        elif (
            isinstance(stereo, dict)
            and all(kind in _REQUEST_KINDS or _SELECTOR.fullmatch(str(kind)) for kind in stereo)
            and all(mode in _KIND_MODES for mode in stereo.values())
        ):
            kinds = {kind: mode for kind, mode in stereo.items() if kind in _REQUEST_KINDS}
            selectors = {
                selector: (match.group(1), int(match.group(2)), mode)
                for selector, mode in stereo.items()
                if (match := _SELECTOR.fullmatch(str(selector)))
            }
        else:
            raise ValueError(
                f"unknown stereo mode {stereo!r}; use one of {sorted(_REQUEST_MODES)} or "
                f"{{kind: mode}} with modes {sorted(_KIND_MODES)}"
            )
        return cls(
            given=stereo,
            graph=_mode_table(kinds, has_geometry=False),
            geometry=_mode_table(kinds, has_geometry=True),
            locked=kinds.get("locked"),
            mirror=stereo == "invert",
            selectors=selectors,
        )

    def on(self, mol):
        """Return this request on `mol`, its atom selectors resolved; a selector naming no atom raises."""
        atoms = {}
        for selector, (symbol, index, mode) in self.selectors.items():
            if not 0 <= index < mol.GetNumAtoms() or mol.GetAtomWithIdx(index).GetSymbol() != symbol:
                raise ValueError(f"stereo selector {selector!r} is not an atom in this molecule")
            atoms[index] = mode
        return replace(self, atoms=atoms)

    def modes(self, mol):
        """Return the {kind: mode} table for `mol`: `geometry` when it has a conformer, else `graph`."""
        return self.geometry if mol.GetNumConformers() else self.graph

    def kept(self, kind):
        """Return the mode the post-embed filter holds `kind` to: its geometry mode."""
        return self.geometry[kind]


DEFAULT_STEREO = StereoRequest.parse(None)


def _lock_double_bond(work, fb):
    """Pin locked double bond `fb` of `work` to an arbitrary hand so RDKit skips it; return whether it could."""
    a, c = tuple(fb)
    wb = work.GetBondBetweenAtoms(a, c)
    if wb is None:
        return False
    bi, ei = wb.GetBeginAtomIdx(), wb.GetEndAtomIdx()
    nb_b = next((n.GetIdx() for n in work.GetAtomWithIdx(bi).GetNeighbors() if n.GetIdx() != ei), None)
    nb_e = next((n.GetIdx() for n in work.GetAtomWithIdx(ei).GetNeighbors() if n.GetIdx() != bi), None)
    if nb_b is None or nb_e is None:
        return False
    wb.SetStereoAtoms(nb_b, nb_e)
    wb.SetStereo(Chem.BondStereo.STEREOCIS)
    return True


def _mask_skipped_stereo(work, potential, skip_points, skip_bonds):
    """Pin each unspecified centre and double bond of `work` the caller skips, so RDKit does not expand it."""
    for element in potential:
        if element.specified != Chem.StereoSpecified.Unspecified:
            continue
        if element.type == Chem.StereoType.Atom_Tetrahedral and element.centeredOn in skip_points:
            work.GetAtomWithIdx(element.centeredOn).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        elif element.type == Chem.StereoType.Bond_Double:
            bond = work.GetBondWithIdx(element.centeredOn)
            left, right = bond_references(work, element)
            pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if pair in skip_bonds and left is not None and right is not None:
                bond.SetStereoAtoms(left, right)
                bond.SetStereo(Chem.BondStereo.STEREOCIS)


def _unassigned_elements(graph, skip_points=(), skip_bonds=frozenset(), racemized=(), skip_axes=False):
    """Return ``(work, locked, enumerable, atrop, unmeasurable)`` of a `StereoGraph` for enumeration.

    `work` is the enumeration graph, `locked` the coordination-locked double bonds pinned on it, `enumerable` the point
    and double-bond elements to expand, `atrop` the atropisomer axes, and `unmeasurable` how many unskipped point
    centres RDKit cannot read from 3D. `skip_bonds` are double bonds left alone. `racemized` are pairs whose hand the
    request cleared: a double bond among them still counts as specified, and an axis is enumerated, as is every
    chelate-locked axis unless `skip_axes`.
    """
    work = graph.work
    cleared_bonds = {
        pair for pair in racemized if graph.mol.GetBondBetweenAtoms(*pair).GetBondType() == Chem.BondType.DOUBLE
    }
    forced_atrop = {tuple(sorted(pair)) for pair in set(racemized) - cleared_bonds}
    if not skip_axes:
        forced_atrop.update(graph.locked_axes)
    # a ring through the metal fixes this E/Z: the other isomer cannot close the ring
    locked = {fb for fb in graph.locked if _lock_double_bond(work, fb)}  # keep only the ones we could actually pin
    supported_points = graph.points
    potential = potential_stereo(work)
    work_points = {element.centeredOn for element in potential if element.type == Chem.StereoType.Atom_Tetrahedral}
    masked_points = set(skip_points) | (work_points - supported_points)
    if masked_points or skip_bonds:
        _mask_skipped_stereo(work, potential, masked_points, skip_bonds)
        potential = potential_stereo(work)
    ez = graph.enumerable_ez(
        work, {e.centeredOn: e for e in potential if e.type == Chem.StereoType.Bond_Double}, cleared_bonds
    )
    enumerable = []  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
    for e in potential:
        if e.specified != Chem.StereoSpecified.Unspecified:
            continue
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            if e.centeredOn in supported_points and e.centeredOn not in skip_points:
                enumerable.append(e)
        elif e.type == Chem.StereoType.Bond_Double:  # skip a double bond the coordination has already locked
            bond = work.GetBondWithIdx(e.centeredOn)
            pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if e.centeredOn in ez and pair not in locked and pair not in skip_bonds:
                enumerable.append(e)
    atrop = sorted(
        pair
        for pair in forced_atrop
        if (bond := work.GetBondBetweenAtoms(*pair)) is not None and bond.GetStereo() == Chem.BondStereo.STEREONONE
    )
    return work, locked, enumerable, atrop, len(graph.unmeasurable - set(skip_points))


@dataclass(frozen=True, slots=True, kw_only=True)
class Unassigned:
    """Name one stereo element a graph leaves unassigned: its `kind` and its `atoms`, in index order.

    `kind` is 'point', 'ez', 'axis' (an atropisomer bond) or 'cumulene' for the first bond of a cumulated chain, whose
    configuration lies between the chain's ends.
    """

    kind: str
    atoms: tuple[int, ...]


def unassigned_centres(mol, exclude=()):
    """Return the unspecified stereo elements of `mol`, one `Unassigned` each, without expanding them.

    A caller that embeds one species uses it to refuse to pool two enantiomers. `mol` is a Mol with its metals in
    `exclude`, or a `StereoGraph`.
    """
    work, _locked, elements, atrop, _unsupported = _unassigned_elements(StereoGraph.of(mol, exclude))
    out = []
    for e in elements:  # `work` only appends caps, so every index here is a real atom of `mol`
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            out.append(Unassigned(kind="point", atoms=(e.centeredOn,)))
            continue
        bond = work.GetBondWithIdx(e.centeredOn)
        kind = "cumulene" if len(cumulene_component(work, e.centeredOn)) > 1 else "ez"
        out.append(Unassigned(kind=kind, atoms=tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))))
    return [*out, *(Unassigned(kind="axis", atoms=pair) for pair in atrop)]


def _enumerate_atrop(work_species, atrop_centers, cap):
    """Return each of `work_species` with every hand of the axes `atrop_centers`, once each, at most `cap`."""
    if not atrop_centers:
        return work_species
    expanded, seen = [], set()
    for base in work_species:
        for tags in itertools.product(ATROP_STEREO, repeat=len(atrop_centers)):
            variant = Chem.Mol(base)
            for pair, tag in zip(atrop_centers, tags, strict=True):
                variant.GetBondBetweenAtoms(*pair).SetStereo(tag)
            key = Chem.MolToCXSmiles(with_conformers(variant, variant, ()))  # a conformer writes the axis it shows
            if key in seen:
                continue
            seen.add(key)
            expanded.append(variant)
            if len(expanded) == cap:
                return expanded
    return expanded


class LigandVariant(NamedTuple):
    """Hold one ligand stereoisomer: its Mol, and the `LigandStereo` its enumeration or input geometry decided."""

    mol: Chem.Mol
    decided: LigandStereo

    @property
    def stereo(self):
        """Return the `LigandStereo` this variant specifies: what its Mol defines, `decided` over it."""
        return defined_stereo_label(self.mol, metal_indices(self.mol)).updated(self.decided)


class Expansion(NamedTuple):
    """Hold the `LigandVariant`s `enumerate_unassigned` expands, with the counts behind them."""

    variants: list
    unassigned: int  # elements expanded
    total: int  # variants before duplicates and the cap are dropped
    unresolved: int  # elements RDKit could not set, each left at one arbitrary hand for the caller to warn about


_STEREO_CAP = 32  # variants enumerated per graph before truncating with a warning that names the remedy


def _complex_key(variant, metal_bonds, faces):
    """Return the key that tells ligand variants of one complex apart.

    A sigma donor's metal bond is made single, so RDKit ranks the donor's hand; a haptic face's bonds are cut and its
    atoms marked instead, since the three-membered rings would hide the face's E/Z.
    """
    keyed = Chem.RWMol(without_atom_maps(variant))
    keyed.RemoveAllConformers()  # coordinates would break the symmetry the key must see
    for metal, donor in metal_bonds:
        if donor in faces:
            keyed.RemoveBond(metal, donor)
            keyed.GetAtomWithIdx(donor).SetAtomMapNum(variant.GetAtomWithIdx(metal).GetAtomicNum())
        else:
            keyed.GetBondBetweenAtoms(metal, donor).SetBondType(Chem.BondType.SINGLE)
    keyed = keyed.GetMol()
    keyed.UpdatePropertyCache(strict=False)
    Chem.SetDoubleBondNeighborDirections(keyed)  # the SMILES writer reads E/Z only from bond directions
    return Chem.MolToCXSmiles(keyed)


def _rdkit_isomers(work, points, opts):
    """Return RDKit's stereoisomers of `work` with `points` as the point centres it flips.

    RDKit picks centres with `FindPotentialStereo` without cleanIt: each of `points` it misses is added as its own OR
    group, and each extra centre it finds is pinned. The input's own stereo groups are dropped, so a specified hand is
    never flipped.
    """
    rdkit_points = {
        element.centeredOn
        for element in Chem.FindPotentialStereo(work)
        if element.type == Chem.StereoType.Atom_Tetrahedral and element.specified == Chem.StereoSpecified.Unspecified
    }
    work = Chem.RWMol(work)
    for atom in rdkit_points ^ set(points):
        work.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    missed = [
        Chem.CreateStereoGroup(Chem.StereoGroupType.STEREO_OR, work, [atom])
        for atom in sorted(set(points) - rdkit_points)
    ]
    work.SetStereoGroups(missed)
    return list(EnumerateStereoisomers(work.GetMol(), opts))


def _reported(expansion):
    """Log what `expansion` expanded, truncated or could not set, and return it."""
    if expansion.unassigned:
        logger.info("stereo: %d unassigned element(s) -> %d variant(s)", expansion.unassigned, len(expansion.variants))
    if expansion.total > _STEREO_CAP:
        logger.warning(
            "stereo: enumerating %d of %d stereoisomers; assign the stereo in the input or pass stereo='free'",
            len(expansion.variants),
            expansion.total,
        )
    if expansion.unresolved:
        logger.warning(
            "stereo: %d element(s) RDKit cannot enumerate or measure; embedded without a hand", expansion.unresolved
        )
    return expansion


def enumerate_unassigned(mol, exclude=(), skip_points=(), skip_bonds=frozenset(), racemized=(), skip_axes=False):
    """Enumerate unspecified point, double-bond and atropisomer stereo as an `Expansion`.

    Variants drop meso duplicates and stop at `_STEREO_CAP` with a warning. Atom order is kept, so index-based ``fix``
    and ``constrain`` stay valid. `exclude` holds the metals, whose hands `arrangements` owns; a ligand centre,
    including a chiral P or carbanion donor, is still enumerated.
    """
    graph = StereoGraph(without_atom_maps(mol), exclude)
    work, locked, unassigned, atrop_centers, unsupported = _unassigned_elements(
        graph, set(skip_points), skip_bonds, racemized, skip_axes
    )
    if not unassigned and not atrop_centers:
        variant = Chem.Mol(mol) if locked else mol
        clear_ez(variant, locked)
        return _reported(Expansion([LigandVariant(variant, LigandStereo())], 0, 1, unsupported))
    atom_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Atom_Tetrahedral]
    bond_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Bond_Double]
    # RDKit's `unique` keys on `work`, where a bound arm reads like a free one once the metal is cut.
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=not exclude, maxIsomers=_STEREO_CAP)
    metal_bonds = [(metal, n.GetIdx()) for metal in exclude for n in mol.GetAtomWithIdx(metal).GetNeighbors()]
    faces = {
        atom
        for metal in exclude
        for site in haptic_sites(mol, [donor for m, donor in metal_bonds if m == metal])
        if len(site) > 1
        for atom in site
    }
    total = 2 ** (len(unassigned) + len(atrop_centers))
    work_species = _rdkit_isomers(work, atom_centers, opts) if unassigned else [work]
    work_species = _enumerate_atrop(work_species, atrop_centers, _STEREO_CAP)
    variants, seen = [], set()
    for wv in work_species:
        full = Chem.Mol(mol)
        clear_ez(full, locked)  # a locked bond keeps no hand, so the metal embed builds the ring-feasible one
        if bond_centers:
            graph.apply_bond_stereo(full, wv, bond_centers)
        for pair in atrop_centers:
            full.GetBondBetweenAtoms(*pair).SetStereo(wv.GetBondBetweenAtoms(*pair).GetStereo())
        label = graph.label(full, wv, atom_centers, bond_centers, atrop_centers)
        key = _complex_key(full, metal_bonds, faces)
        if key in seen:
            continue
        seen.add(key)
        variants.append(LigandVariant(full, label))
    # unset in every isomer means RDKit cannot set it (an allene); a pseudoasymmetric centre is unset in some
    for probe in work_species:
        Chem.FindPotentialStereo(probe, cleanIt=True, flagPossible=False)
    unresolved = (
        unsupported
        + sum(
            all(p.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for p in work_species)
            for i in atom_centers
        )
        + sum(
            all(p.GetBondWithIdx(b).GetStereo() == Chem.BondStereo.STEREONONE for p in work_species)
            for b in bond_centers
        )
    )
    return _reported(Expansion(variants, len(unassigned) + len(atrop_centers), total, unresolved))


def _ligand_stereo_request(graph, request, point_mode):
    """Return each point centre's mode, and the centres enumeration clears and skips.

    An atom selector, then ``locked`` for a coordination-locked donor, overrides `point_mode`. A 'free' centre is not
    enumerated: a specified hand stays and the seed decides an open one; one a selector or ``locked`` names loses its
    tag.
    """
    exact = dict(request.atoms)
    if request.locked is not None:  # an atom selector still wins, as it does over "point"
        for index in graph.locked_centres:
            exact.setdefault(index, request.locked)
    centres = graph.points if exact or point_mode != "unassigned" else set()
    invalid = sorted(set(exact) - centres)
    if invalid:
        raise ValueError(f"stereo atom(s) {invalid} are not configurable point stereocentres")
    modes = dict.fromkeys(centres, point_mode) | exact
    skip = {index for index, mode in modes.items() if mode == "free"}
    clear = {index for index, mode in modes.items() if mode == "racemic"} | (skip & exact.keys())
    return modes, clear, skip


def _keeps_reference_stereo(stereo, observed, points, modes):
    """Return whether each element of `stereo` keeps its requested relation to the `observed` stereo."""
    for element in stereo.elements:
        kind = element.kind
        mode = points.get(element.atoms[0], modes["point"]) if kind == "point" else modes[kind]
        if mode not in ("preserve", "invert") or (kind == "axis" and element.atoms not in observed.axes):
            continue  # an inferred axis has no reference hand to preserve or invert
        if kind == "point" and mode == "invert" and element.atoms[0] in observed.points:
            continue  # mirrored where it stands (`ligand_variants`); a pseudoasymmetric code reads the same
        if (mode == "preserve") != (element in observed.elements):
            return False
    return True


def ligand_variants(mol, request):
    """Return the `LigandVariant`s of `mol` under `request`; a metal's own hand is not a variant.

    A conformer is reference geometry: it fills what the tags leave unassigned, and 'preserve' and 'invert' keep or
    mirror what it shows.
    """
    has_geometry = bool(mol.GetNumConformers())
    table = request.modes(mol)
    modes = {"point": table["point"], "ez": table["ez"], "axis": table["axial"]}  # by element kind
    graph = StereoGraph(mol, metal_indices(mol))
    points, clear, skip = _ligand_stereo_request(graph, request, modes["point"])
    referenced = has_geometry or not {"preserve", "invert"}.isdisjoint((*modes.values(), *points.values()))
    specified = (
        defined_stereo_label(graph) if referenced else LigandStereo()
    )  # only a reference reads the specification
    observed = specified
    mol = Chem.Mol(mol)
    changed = False
    unstable_bonds = set()
    if has_geometry:
        observed_mol = Chem.Mol(mol)  # retain local tags too: a pseudoasymmetric code needs its partner
        observed = stereo_from_3d(observed_mol, graph.metals, apply=True).updated(specified)
        skip.update(index for index, mode in points.items() if mode == "preserve" and index not in observed.points)
        unstable_bonds = {
            pair
            for element in unassigned_centres(graph)
            if element.kind in ("ez", "cumulene") and (pair := frozenset(element.atoms)) not in observed.bonds
        }
        changed |= bool(assign_atrop_from_3d(mol, observed.axes))
    for index in clear:
        atom = mol.GetAtomWithIdx(index)
        changed |= atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if has_geometry:  # a point the request holds and the tags leave open takes the tag its conformer shows
        for index in points.keys() & observed.points.keys() - specified.points.keys():
            if points[index] in ("preserve", "invert"):
                changed = True
                mol.GetAtomWithIdx(index).SetChiralTag(observed_mol.GetAtomWithIdx(index).GetChiralTag())
    mirrored = {index for index, mode in points.items() if mode == "invert"}
    for index in mirrored:  # a mirror image inverts each hand where it stands, so a meso centre pair stays meso
        mol.GetAtomWithIdx(index).SetChiralTag(mirror_tag(mol.GetAtomWithIdx(index).GetChiralTag()))
    skip_bonds = (
        {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        if modes["ez"] == "free"
        else unstable_bonds
    )
    racemized = clear_atrop(mol) if modes["axis"] in ("racemic", "invert") else set()
    changed |= bool(racemized or mirrored)
    if modes["ez"] in ("racemic", "invert"):
        racemized |= graph.specified_ez
        changed |= clear_ez(mol)
    variants, n_unassigned, _total, _unresolved = enumerate_unassigned(
        mol,
        exclude=graph.metals,
        skip_points=skip,
        skip_bonds=skip_bonds,
        racemized=racemized,
        skip_axes=modes["axis"] == "free",
    )
    if not n_unassigned:
        decided = (
            LigandStereo(elements=tuple(e for e in observed.elements if e.kind != "point" or e.atoms[0] not in skip))
            if has_geometry
            else LigandStereo()
        )
        variants = [LigandVariant(mol if changed else graph.mol, decided)]
    variants = (
        [  # a mirrored point is decided as it now reads, so no conformer re-reads it
            variant._replace(
                decided=variant.decided.updated(
                    LigandStereo(
                        elements=tuple(
                            e
                            for e in defined_stereo_label(variant.mol, graph.metals).elements
                            if e.kind == "point" and e.atoms[0] in mirrored
                        )
                    )
                )
            )
            for variant in variants
        ]
        if mirrored
        else variants
    )
    kept = [variant for variant in variants if _keeps_reference_stereo(variant.decided, observed, points, modes)]
    if not kept:
        raise ValueError(
            f"stereo={request.given!r} has no configuration compatible with what the input specifies or shows"
        )
    return kept
