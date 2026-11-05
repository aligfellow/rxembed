"""Read which species a Mol is, or can be, whether its conformer agrees, and its name.

A species is a Mol whose stereo is specified: chiral tags and bond stereo for ligands, and CX notes giving each metal's
arrangement and hand. `read_species` reads the one species a verb embeds, `possible_species` every species `rx.isomers`
lists, and `read` compares one conformer with what its Mol specifies (`Specification`, `Reading`, `compared`).
`smiles`, `cxsmiles` and `identity` read a Mol and then write it with `smiles_io`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import permutations

import numpy as np
from rdkit import Chem

from rxembed.geometry.constraints import Resolved, resolve_atom, resolve_core, substruct_matches
from rxembed.graph import bound_metals, metal_indices
from rxembed.identity.arrangements import enumerate_isomers
from rxembed.identity.smiles_io import (
    arrangement_notes,
    canonical_notes,
    distinct,
    logger,
    mol_from_species,
    own_rankings,
    read_native_tags,
    specified_arrangement,
    specified_species,
    specified_windings_only,
    write_cxsmiles,
    write_name,
    write_notes,
    write_organic,
    write_smiles,
)
from rxembed.identity.species import (
    ObservedStereo,
    Species,
    centre_arrangement,
    centre_from_slots,
    centre_label,
    centre_name,
    fitted_centre,
    graph_with_stereo,
    haptic_configuration,
    in_canonical_order,
    misplaced_pins,
    observe_stereo,
    observed_centre,
    selected_centres,
    shape_readings,
    takes_shape,
    without_conformers,
)
from rxembed.identity.stereoisomers import DEFAULT_STEREO, StereoRequest, ligand_variants, unassigned_centres
from rxembed.metal.centres import (
    Centre,
    canonical_metal_graph,
    chemical_graph,
    collapse_haptic,
    engine_graph,
    materialized_states,
    symmetric_metals,
)
from rxembed.metal.polyhedra import POLYHEDRA, VACANT, mirror_seating, resolve_geometry, vertex_dirs
from rxembed.metal.shape import classify_geometry, site_rays
from rxembed.metal.slots import order_label
from rxembed.metal.sphere import LigandFacts
from rxembed.metal.stereo import donor_classes
from rxembed.stereo import (
    ELEMENT_KINDS,
    Miss,
    StereoGraph,
    clear_ez,
    defined_stereo_label,
    ligand_misses,
    renumber_atoms,
    stereo_from_3d,
    stereo_symmetry,
    without_stale_point_tags,
)
from rxembed.utils import atom_label, require_mol, with_conformers, without_atom_maps

_KEEP_ALL = StereoRequest.parse({"default": "preserve", "locked": "preserve"})  # a conformer fills the rest


def read(mol):
    """Return the `Reading` of the first conformer of `mol` against what `mol` specifies, or of `mol` alone.

    Only specified elements count: unassigned ligand stereo and metals without notes specify nothing. The metal graph is
    rebuilt for this conformer, since its canonical form depends on it.
    """
    mol = canonical_metal_graph(mol)
    if not mol.GetNumConformers():
        species = Species(mol=mol, stereo=defined_stereo_label(mol, metal_indices(mol)))
        return Reading(given=mol, specification=Specification(species=species, given=mol))
    conf = mol.GetConformer()
    return compared(Specification.from_mol(mol), conf, observe_stereo(mol, conf))


def described(mol):
    """Return the `Reading` a describe verb names, native metal tags read as notes; refuse a contradiction."""
    reading = read(read_native_tags(mol))
    if reading.problem:
        raise ValueError(
            f"this Mol's conformer and its specified stereo disagree ({reading.problem}); clear the notes and "
            "tags, or embed it"
        )
    return reading


def in_conformer_frame(mol, species):
    """Return `mol` with its notes rewritten in its conformer's frame, or `mol` itself when they fit.

    `species` is what `mol` specifies. A conformer that contradicts it keeps the notes. One that agrees after a graph
    symmetry, or with equivalent donors swapped, gets notes for the seating it shows, an open face winding left open.
    """
    if not mol.GetNumConformers() or (reading := read(mol)).problem:
        return mol
    specification, observed = reading.specification, reading.observed
    if not reading.moved and not _seated_otherwise(species, mol.GetConformer()):
        return mol
    framed = Chem.Mol(specification.mol if reading.moved else mol)
    seated = specified_windings_only(observed, specification.species)
    return canonical_metal_graph(write_notes(framed, seated) if seated.centres else framed)


def conformers_alike(mol, species=None, *, reading=None):
    """Raise unless every later conformer of `mol` reads as `species`, by default what the first shows.

    When the first conformer contradicts `species`, the others are compared with what the first shows instead, so
    conformers that contradict it alike stay one species. When the first agrees, the others must also seat equivalent
    donors as it does, since one record's constraints cannot hold two seatings. A given `reading` is of the first
    conformer.
    """
    if mol.GetNumConformers() < 2:  # noqa: PLR2004 - two conformers to compare
        return
    first, *rest = (conf.GetId() for conf in mol.GetConformers())
    first_species = (read(Chem.Mol(mol, False, first)) if reading is None else reading).species
    species = first_species if species is None else species
    if not (species.centres or species.stereo):
        return
    specification = Specification(species=species, given=graph_with_stereo(species))
    agrees = compared(specification, mol.GetConformer(first)).agrees
    if not agrees:
        specification = Specification(species=first_species, given=graph_with_stereo(first_species))
    other, seated = [], []
    for cid in rest:
        conf = mol.GetConformer(cid)
        if not compared(specification, conf).agrees:
            other.append(str(cid))
        elif agrees and _seated_otherwise(first_species, conf, angles_only=True):
            seated.append(str(cid))
    if other:
        raise ValueError(
            f"conformer(s) {', '.join(other)} of this Mol show another species than conformer {first}; keep one per Mol"
        )
    if seated:
        raise ValueError(
            f"conformer(s) {', '.join(seated)} of this Mol seat equivalent donors otherwise than conformer "
            f"{first}; pass one Mol per conformer"
        )


def _candidates(mol, request, *, selected=()):
    """Return the species `request` enumerates for `mol`; `possible_species` decides which keep conformers.

    An organic without a stereo request is enumerated without its conformers.
    """
    if metal_indices(mol):
        if request.observed:
            conformers_alike(mol)
        return enumerate_isomers(mol, request, selected=selected)
    graph = Chem.Mol(mol)
    if request.stereo.given is None:
        graph.RemoveAllConformers()
    return [Species(mol=variant.mol, stereo=variant.stereo) for variant in ligand_variants(graph, request.stereo)]


def read_species(mol, stereo=DEFAULT_STEREO, *, pins=None):
    """Return the one species `mol` is, read once where a verb takes it, in its conformer's frame.

    An organic fills the stereo its tags leave open from its conformer. A metal Mol with notes on every metal is what
    they specify, with unassigned ligand stereo read from the conformer; without a full set of notes it is what the
    conformer shows. The notes are rewritten in the conformer's frame unless `pins` fit the noted seating better.
    Without a conformer the Mol must be exactly one species under `stereo`, else it raises naming `rx.isomers`. Every
    later conformer must read as the first.
    """
    metals = metal_indices(mol)
    if not mol.GetNumConformers():
        return _enumerated(mol, stereo)
    if not metals:
        mol = fill_unspecified_from_3d(mol)  # its tags now agree with the conformer, so its frame stands
        species = _specified(mol, metals)
    elif not all(specified_arrangement(mol, m) is not None for m in metals):
        observed = observed_species(observe_stereo(mol, mol.GetConformer()))  # the conformer is the specification
        species = replace(observed, mol=with_conformers(observed.mol, mol, [c.GetId() for c in mol.GetConformers()]))
    else:
        species = _specified(mol, metals)
        if (framed := in_conformer_frame(mol, species)) is not mol:
            reframed = _specified(framed, metals)
            if not (pins is not None and _misfit(species, pins.cons) < _misfit(reframed, pins.cons)):
                mol, species = framed, reframed
    conformers_alike(mol, species)
    return (
        species if stereo.given is None else _enumerated(graph_with_stereo(species), stereo, selected=species.centres)
    )


def _specified(mol, metals):
    """Return the species `mol` specifies, its conformer filling ligand stereo the notes leave open."""
    if not metals:
        graph = StereoGraph(mol)
        _ = graph.points  # refuse a specified point no gate could read back from 3D, before any embedding
        return Species(mol=mol, stereo=defined_stereo_label(graph))
    return _enumerated(mol, _KEEP_ALL)  # so a locked class is read, not enumerated


def _enumerated(mol, stereo, *, selected=()):
    """Return the one species `stereo` gives `mol` and its `selected` centres; several or none raise."""
    request = IsomerRequest(stereo=stereo, center=requested_centre(mol, None), pins=resolve_core(mol))
    found = _candidates(mol, request, selected=selected)
    if len(found) != 1:
        if mol.GetNumConformers():
            detail = f"stereo={stereo.given!r} makes this {len(found)} species"
        else:
            detail = (
                f"this Mol specifies {len(found)} species (unassigned stereo or arrangement)"
                if request.center is not None
                else f"this source is {len(found)} stereoisomers"
            )
        raise ValueError(f"{detail}; pick one with rx.isomers or pass stereo='free' to leave ligand stereo to the seed")
    return found[0]


def _seated_otherwise(species, conf, *, angles_only=False):
    """Return the sphere atoms conformer `conf` seats away from the vertices `species` specifies.

    Every atom is pinned where `conf` puts it, and each haptic face at its centroid.
    """
    pos = conf.GetPositions()
    core = dict(enumerate(pos))
    for sites in species.sites.values():
        core |= {dummy: pos[list(atoms)].mean(axis=0) for dummy, atoms in sites.haptic.items()}
    return misplaced_pins(species, core, angles_only=angles_only)


def _misfit(species, pins):
    """Return how far, in degrees, the metal angles `pins` fixes sit from `species`' seating.

    A pinned distance between two donors counts through the first conformer's metal-donor lengths; other terms fit every
    seating alike.
    """
    pos = species.mol.GetConformer().GetPositions()
    misfit = 0.0
    for centre in species.centres:
        vertex = {site: v for v, site in enumerate(species.sites[centre.atom].vertices)}
        angles, metal = POLYHEDRA[centre.geometry].pair_angles, pos[centre.atom]
        pinned = {(a, b): np.mean(window) for (a, m, b), window in pins.angles.items() if m == centre.atom}
        for (a, b), window in pins.distances.items():
            if a in vertex and b in vertex:
                ra, rb = (np.linalg.norm(pos[atom] - metal) for atom in (a, b))
                cos = (ra * ra + rb * rb - np.mean(window) ** 2) / (2 * ra * rb)
                pinned[a, b] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        misfit += sum(
            abs(angle - angles[frozenset((vertex[a], vertex[b]))])
            for (a, b), angle in pinned.items()
            if a in vertex and b in vertex
        )
    return misfit


@dataclass(frozen=True, slots=True, kw_only=True)
class IsomerRequest:
    """Hold one `rx.isomers` request, each choice read once on its Mol."""

    stereo: StereoRequest
    center: int | str | None  # `requested_centre`: one metal's atom index, 'all', or None for a species with no metal
    pins: Resolved  # fix= resolved on the Mol (`resolve_core`)
    geometry: tuple[str, ...] | None = None  # `requested_geometry`: the POLYHEDRA names, or None for each default
    coordinate: CoordinateRequest | None = None  # `coordinate_request`, seated by `vacancy_seatings`
    observed: bool | None = None
    screen: bool = True


def requested_centre(mol, center):
    """Return the metal ``center=`` names on `mol`: its atom index, 'all', or None without a metal.

    Unset, it names the only metal, or 'all' of several. A choice naming no metal, or a symbol several share, raises.
    """
    metals = metal_indices(mol)
    if center is None:
        return metals[0] if len(metals) == 1 else ("all" if metals else None)
    if center == "all":
        return center
    atoms = map(mol.GetAtomWithIdx, metals)
    centres = [Centre(atom=a.GetIdx(), atomic_num=a.GetAtomicNum(), charge=a.GetFormalCharge()) for a in atoms]
    named = selected_centres(centres, center)
    if len(named) != 1:
        raise ValueError(
            f"enumerating one centre needs center= to name one of {[centre_name(c) for c in centres]} by atom index"
            " or element symbol, or center='all'"
        )
    return named[0].atom


def requested_geometry(geometry):
    """Return the `POLYHEDRA` names ``geometry=`` gives, in order, or None when unset.

    It takes a name, a 3-letter code in either case, or a list of them; an unknown name raises listing the choices.
    """
    if geometry is None:
        return None
    names = tuple(map(resolve_geometry, geometry if isinstance(geometry, (list, tuple)) else [geometry]))
    for name in names:
        if name not in POLYHEDRA:
            hint = "; pass a list of names, e.g. ['square_planar', 'tetrahedral']" if name == "all" else ""
            raise ValueError(
                f"unknown geometry {name!r}; available: {sorted(POLYHEDRA)} "
                f"(or a code: {sorted(p.code for p in POLYHEDRA.values() if p.code)}){hint}"
            )
    return names


def possible_species(mol, request):
    """Return every species `mol` can be under `request`, as `rx.isomers` lists them.

    With a conformer and no geometry or stereo request, the one species it shows keeps the conformer, as input geometry
    where it contradicts the notes. Under ``observed=True`` a species keeps its conformers only where each agrees with
    it. Every other species is enumerated without conformers. `fix=` pins keep only the species they agree with.
    """
    metals = metal_indices(mol)
    pins = request.pins
    stereo = request.stereo
    from_conformer = (
        request.observed is not False and mol.GetNumConformers() and request.geometry is None and stereo.given is None
    )
    one = from_conformer and (request.observed is None or not metals or read(mol).problem is None)
    found = [read_species(mol, pins=pins)] if one else _candidates(mol, request)
    kept = [species for species in found if not misplaced_pins(species, pins.graft)]
    if found and not kept:
        raise ValueError(
            f"fix= pins coordination-sphere atoms {misplaced_pins(found[0], pins.graft)} off the arrangement "
            "this source specifies; "
            "use observed=True"
        )
    if request.observed:  # conformers stay only where each agrees with the species
        agreed = []
        for species in kept:
            specification = Specification(species=species, given=graph_with_stereo(species))
            shown = all(compared(specification, conf).agrees for conf in species.mol.GetConformers())
            agreed.append(species if shown else without_conformers(species))
        return agreed
    return (
        kept if one else [without_conformers(species) for species in kept]
    )  # an enumerated species carries no conformer


def fill_unspecified_from_3d(mol):
    """Return a copy of organic `mol` with the stereo its tags leave open filled from its conformers.

    Tags a conformer contradicts stay as they are. Conformers that contradict the tags differently raise; a centre they
    fill differently stays open.
    """
    first, *rest = list(mol.GetConformers())  # RDKit's iterator restarts on direct starred unpacking.
    reading = read(Chem.Mol(mol, False, first.GetId()) if rest else mol)
    if rest:
        conformers_alike(mol, Species(mol=mol, stereo=defined_stereo_label(mol)), reading=reading)
    if reading.problem:
        return Chem.Mol(mol)
    out = with_conformers(reading.stereo_mol, mol, [conf.GetId() for conf in mol.GetConformers()])
    for conf in rest if rest and unassigned_centres(mol) else ():
        if not read(with_conformers(out, mol, [conf.GetId()])).problem:  # a meso mirror agrees too
            continue
        other = Chem.Mol(mol, False, conf.GetId())
        stereo_from_3d(other, apply=True)
        for atom in out.GetAtoms():
            if atom.GetChiralTag() != other.GetAtomWithIdx(atom.GetIdx()).GetChiralTag():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        kept, theirs = defined_stereo_label(out).bonds, defined_stereo_label(other).bonds
        clear_ez(out, {pair for pair, code in kept.items() if theirs.get(pair) != code})
    return out


def smiles(mol):
    """Return the canonical dative SMILES of `mol`: its constitution and ligand stereo, without metal notes.

    Cis and trans give one string, so it is not a species key; `cxsmiles` is.
    """
    mol = without_stale_point_tags(require_mol(mol, "rx.smiles"))
    if not mol.GetNumConformers() and (written := write_organic(mol, cx=False)) is not None:
        return written
    reading = described(mol)
    return write_smiles(reading.stereo_mol, reading.ligand_stereo)


def cxsmiles(mol):
    """Return the canonical dative CX SMILES of `mol` with its metal notes, atom maps kept; `identity` names a species.

    Metal notes hold polyhedron and hand; donor notes hold canonical slots and haptic winding. E/Z uses bond directions,
    CX ``c:``/``t:`` fields where RDKit supports them, and paired ``_rxEZ`` atom properties otherwise. A bridging donor
    holds one slot per metal; an organic Mol writes no notes.

    The name is what the first conformer shows, else what the Mol specifies. RDKit may write either metal of a symmetric
    pair first, so the notes are the least over both orders. Raises rather than lose identity: a missing polyhedron
    template, contradictory stereo, or three or more equivalent metals with different arrangements. How bridges join
    three or more centres is not written.
    """
    mol = without_stale_point_tags(require_mol(mol, "rx.cxsmiles"))
    if not mol.GetNumConformers() and (written := write_organic(mol, cx=True)) is not None:
        return written
    reading = described(mol)
    symmetric = reading.metal_classes if reading.species.centres else ()
    return write_cxsmiles(reading.stereo_mol, reading.species, symmetric)


@dataclass(frozen=True, slots=True, kw_only=True)
class Identity:
    """Name one species: its canonical CX SMILES, and on its Mol's atoms its ligand stereo and every centre.

    Two Identities are equal exactly when their names are: one species however its atoms are numbered or mapped.
    `centres` holds one `Centre` per metal in the name's order, seated as its notes seat it; a vertex is a `Site`, or
    None when vacant. An organic species has only `name` and `stereo`.
    """

    name: str  # the species' `cxsmiles`, written without atom maps or other atom properties
    stereo: str = field(default="", compare=False)  # ligand stereo label, e.g. "C1:R,C3=C4:E"
    centres: tuple = field(default=(), compare=False)
    labels: tuple[str, ...] = field(default=(), compare=False)  # each coarse cis/trans/fac/mer label
    arrangements: tuple[str, ...] = field(default=(), compare=False)  # each vertex's ligand, "·" if vacant
    haptic: tuple[str, ...] = field(default=(), compare=False)  # each relative face configuration: "rac", "meso" or ""


def named(mol):
    """Name what `mol`'s first conformer shows, else what it specifies, refusing a contradiction."""
    reading = described(without_stale_point_tags(without_atom_maps(mol)))
    if not reading.species.centres and bound_metals(reading.given):
        species = read_species(reading.given)
        graph = graph_with_stereo(species)
        # Reordering donor bonds changes the neighbour order used by stereo labels.
        species = replace(species, stereo=defined_stereo_label(graph, metal_indices(graph)))
        return write_name(graph, species, symmetric_metals(graph))
    symmetric = reading.metal_classes if reading.species.centres else ()
    return write_name(reading.stereo_mol, reading.species, symmetric)


def named_species(species):
    """Name a `Species` without reading coordinates, seated as its written notes seat it.

    A record with bound metals but no centres is named as the one species `read_species` gives it.
    """
    graph = without_atom_maps(graph_with_stereo(species))
    graph.RemoveAllConformers()
    if not species.centres and bound_metals(graph):
        return named(graph)
    return write_name(graph, species, symmetric_metals(graph))


def identity(mol, conf_id=None):
    """Return the `Identity` of the species `mol` is.

    With a conformer (`conf_id`, default the first) it is what the conformer shows, and notes it contradicts raise;
    without one it is what the notes and tags specify.
    """
    mol = require_mol(mol, "rx.identity", geometry=conf_id is not None)
    given = conf_id is not None or mol.GetNumConformers()
    one = Chem.Mol(mol, False, mol.GetConformer(-1 if conf_id is None else int(conf_id)).GetId()) if given else mol
    name, species = named(one)
    facts = LigandFacts(species.mol)
    return Identity(
        name=name,
        stereo=str(species.stereo),
        centres=species.centres,
        labels=tuple(centre_label(facts, centre) for centre in species.centres),
        arrangements=tuple(centre_arrangement(facts, centre) for centre in species.centres),
        haptic=tuple(haptic_configuration(facts, centre) for centre in species.centres),
    )


def _own_notes(species):
    """Return the canonical notes of each centre of `species`, keyed by its own atom indices."""
    return arrangement_notes(
        species, LigandFacts(species.mol), {atom: atom for atom in range(species.mol.GetNumAtoms())}
    )


def _unwound(species):
    """Return `species` without face windings, the form seatings and hands are compared in."""
    unwind = (
        c._replace(vertices=tuple(VACANT if site is VACANT else replace(site, winding="") for site in c.vertices))
        for c in species.centres
    )
    return replace(species, centres=tuple(unwind))


def _reflected(species, centre):
    """Return `species` with `centre`'s seating mirrored, windings removed."""
    vertices = mirror_seating(vertex_dirs(centre.geometry), centre.vertices)
    slots = {slot: site.atoms[0] for slot, site in enumerate(vertices) if site is not VACANT}
    image = centre_from_slots(LigandFacts(species.mol), centre.atom, centre.geometry, slots)
    return _unwound(replace(species, centres=tuple(image if c.atom == centre.atom else c for c in species.centres)))


def _on(facts, species):
    """Return the centres of `species` rebuilt on the graph of `facts`, each hand read there."""
    centres = []
    for c in species.centres:
        slots = {slot: site.atoms[0] for slot, site in enumerate(c.vertices) if site is not VACANT}
        centres.append(centre_from_slots(facts, c.atom, c.geometry, slots)._replace(vertices=c.vertices))
    return Species(mol=facts.graph, centres=tuple(centres), stereo=species.stereo)


@dataclass(frozen=True, kw_only=True, eq=False)
class Specification:
    """Hold what one species specifies, prepared once and compared with each conformer (`compared`).

    `given` is the Mol it was read from, if any. `misses` caches each conformer's comparison with its coordinates.
    """

    species: Species
    given: Chem.Mol | None = None
    images: dict = field(default_factory=dict, repr=False)  # graph symmetry -> the specification it moves this onto
    misses: dict = field(default_factory=dict, repr=False)  # conformer id -> (coordinates, `conformer_misses` result)

    @classmethod
    def from_mol(cls, mol):
        """Return what canonical metal graph `mol` specifies through its notes and tags.

        Notes on every bound metal arrange each centre; notes on some or none specify no arrangement, so only its
        ligand stereo is compared.
        """
        stereo = defined_stereo_label(mol, metal_indices(mol))
        bound = bound_metals(mol)
        if bound and all(specified_arrangement(mol, m) is not None for m in bound):
            return cls(species=replace(specified_species(mol), stereo=stereo), given=mol)
        return cls(species=Species(mol=mol, stereo=stereo), given=mol)

    @cached_property
    def mol(self):
        """Return the Mol it was read from, else the species Mol with notes; messages label atoms on it."""
        return mol_from_species(self.species) if self.given is None else self.given

    @cached_property
    def metal_classes(self):
        """Return its symmetry-equivalent metals."""
        return symmetric_metals(self.mol)

    @cached_property
    def rankings(self):
        """Return the atom rankings its notes compare under: its own and each equivalent-metal swap."""
        return own_rankings(self.metal_classes, self.species.stereo, self.species)

    @cached_property
    def inputs(self):
        """Return each centre's shape reading on the species' own conformer, which accepted records cite."""
        if not self.species.mol.GetNumConformers():
            return {}
        readings = shape_readings(self.species.mol, self.species)
        return {atom: reading for atom, reading in readings.items() if reading.verdict != "unreadable"}

    @cached_property
    def centres(self):
        """Return ``(centre, sites)`` for each centre with a hand or a face winding."""
        rows = ((state, self.species.sites[state.atom]) for state in self.species.centres)
        return tuple((state, sites) for state, sites in rows if state.hand or sites.winding)

    @cached_property
    def face_classes(self):
        """Return the donor symmetry classes of each centre with a face winding."""
        graph = engine_graph(self.species.mol)
        return {
            state.atom: donor_classes(
                graph,
                [d for d in vertices if d is not VACANT and d not in haptic]
                + [a for face in haptic.values() for a in face],
            )
            for state, (vertices, haptic, winding, _donors) in self.centres
            if winding
        }

    @cached_property
    def locked(self):
        """Return the point centres whose hand coordination fixes."""
        return StereoGraph(self.mol, metal_indices(self.mol)).locked_centres

    @cached_property
    def rows(self):
        """Return its canonical notes over every equivalent-metal swap that keeps its stereo."""
        return canonical_notes(self.species, self.rankings, LigandFacts(self.species.mol)).rows

    @cached_property
    def notes(self):
        """Return each centre's notes in its own atom order, windings included."""
        return _own_notes(self.species)

    @cached_property
    def seating(self):
        """Return each centre's notes in its own atom order, windings removed."""
        return _own_notes(_unwound(self.species))

    @cached_property
    def mirrored(self):
        """Return the notes of each handed centre with its seating mirrored, windings removed."""
        species = self.species
        return {c.atom: _own_notes(_reflected(species, c))[c.atom] for c in species.centres if c.hand}

    def moved(self, order):
        """Return this specification renumbered by graph symmetry `order`, conformers kept, cached."""
        if (image := self.images.get(key := tuple(order))) is not None:
            return image
        bare = write_notes(Chem.Mol(self.mol), self.species)
        bare.RemoveAllConformers()
        bare.ClearComputedProps(includeRings=False)  # stereo perception caches atom indices renumbering would not move
        ids = [conf.GetId() for conf in self.mol.GetConformers()]
        out = with_conformers(renumber_atoms(bare, order), self.mol, ids)
        self.images[key] = Specification.from_mol(canonical_metal_graph(out))
        return self.images[key]


# How a problem names each `Miss` kind: by stereo codes, or by face windings.
_MISSED = dict.fromkeys(("point", "ez", "axis", "hand"), "{specified} but the conformer has {observed}") | {
    "seating": "seating but the conformer has another arrangement",
    "face": "faces {specified} but the conformer has {observed}",
}


@dataclass(frozen=True, kw_only=True, eq=False)
class Reading:
    """Hold one read of a Mol, or of one of its conformers, against what it specifies.

    With a conformer, `observed_stereo` is its ligand stereo from 3D, `observed` the species seated where the conformer
    puts it, and `misses` each specified element it shows otherwise. `moved` says a graph symmetry moved `specification`
    onto the conformer, so its Mol is no longer `given`. Without a conformer only `given` and `specification` are set.
    """

    given: Chem.Mol
    specification: Specification
    observed_stereo: ObservedStereo | None = None
    observed: Species | None = None
    misses: tuple = ()
    moved: bool = False

    @property
    def misshapen(self):
        """Return whether a centre's donors miss its polyhedron."""
        return self.observed is not None and not all(takes_shape(self.observed.mol, c) for c in self.observed.centres)

    @property
    def agrees(self):
        """Return whether the conformer agrees with the specification: no miss, every centre in shape."""
        return not self.misses and not self.misshapen

    @property
    def stereo_mol(self):
        """Return the Mol tagged with the conformer's stereo, else the Mol given."""
        return self.given if self.observed_stereo is None else self.observed_stereo.mol

    @property
    def ligand_stereo(self):
        """Return the ligand stereo: the conformer's, else the Mol's."""
        return self.specification.species.stereo if self.observed_stereo is None else self.observed_stereo.stereo

    @property
    def metal_classes(self):
        """Return the symmetry-equivalent metals of the graph read."""
        return self.specification.metal_classes

    @cached_property
    def problem(self):
        """Return how the conformer contradicts the specification, else None."""
        ligand = [miss for miss in self.misses if miss.kind in ELEMENT_KINDS]
        if not ligand and self.misshapen:
            return "the donors do not take the specified polyhedron"
        named = [
            f"{ELEMENT_KINDS.get(miss.kind, ('',))[0].join(atom_label(self.given, atom) for atom in miss.atoms)} "
            + _MISSED[miss.kind].format(specified=miss.specified or "none", observed=miss.observed or "none")
            for miss in ligand or self.misses
        ]
        return f"specified {'; '.join(named)}" if named else None

    @cached_property
    def species(self):
        """Return the `Species` the conformer shows, else the one the Mol specifies.

        Without notes the species is read on the Mol given, so its labels follow that Mol's bonds.
        """
        stereo, observed = self.observed_stereo, self.observed
        if stereo is None or observed is None:
            return replace(specified_species(self.given), stereo=self.ligand_stereo)
        if (specified := self.specification.species).centres:
            return (
                replace(observed, stereo=stereo.stereo)
                if stereo.stereo == specified.stereo
                else replace(_on(LigandFacts(chemical_graph(stereo.mol)), observed), stereo=stereo.stereo)
            )
        if self.moved:
            stereo = observe_stereo(self.given, self.given.GetConformer())
        return observed_species(stereo, bound_metals(stereo.mol))


def compared(specification, conf, observed_stereo=None):
    """Return the `Reading` of conformer `conf` against `specification`.

    Ligand stereo is compared element by element, and each centre by canonical notes over equivalent-metal swaps. Where
    the conformer fixes ligand stereo the specification leaves open, centres are compared on the graph that stereo
    gives, since donors it tells apart may trade places. When a graph symmetry maps the specification onto the
    conformer, such as the mirror image of an achiral species, the comparison is made in that frame.
    """
    comparison = _compared_in_frame(specification, conf, observed_stereo)
    if not any(miss.kind in ELEMENT_KINDS for miss in comparison.misses):
        return comparison
    order = stereo_symmetry(specification.mol, comparison.observed_stereo.mol, specification.species.stereo)
    if order is None:
        return comparison
    in_frame = _compared_in_frame(specification.moved(order), conf, None)
    return comparison if in_frame.misses else replace(in_frame, given=specification.mol, moved=True)


def _compared_in_frame(specification, conf, stereo):
    """Return the `Reading` of conformer `conf` against `specification` in its own atom order."""
    species = specification.species
    pos = conf.GetPositions()
    undefined = {c.atom for c in species.centres if site_rays(pos, c.atom, *species.sites[c.atom][:2]) is None}
    observed, facts = species, None
    if species.centres:
        mol = Chem.Mol(species.mol)
        mol.RemoveAllConformers()
        mol.AddConformer(Chem.Conformer(conf), assignId=False)
        facts = LigandFacts(mol)
        read = (c if c.atom in undefined else fitted_centre(facts, c.atom, c.geometry) for c in species.centres)
        observed = replace(species, mol=mol, centres=tuple(read))
    ligand = ()
    if species.stereo:
        stereo = stereo or observe_stereo(specification.mol, conf)
        ligand = ligand_misses(species.stereo, stereo.stereo, specification.mol, stereo.mol)
    centres = _centre_misses(specification, observed, undefined, facts)
    if centres and not ligand and not undefined:
        stereo = stereo or observe_stereo(specification.mol, conf)
        if stereo.stereo != species.stereo and _agrees_on_observed(specification, observed, stereo):
            centres = ()
    return Reading(
        given=specification.mol,
        specification=specification,
        observed_stereo=stereo,
        observed=observed,
        misses=(*centres, *ligand),
    )


def _centre_misses(specification, observed, undefined, facts):
    """Return a `Miss` for each centre `observed` arranges otherwise than `specification`.

    Windings aside, the mirror image is a 'hand' miss and any other seating a 'seating' miss; the same seating with a
    face wound otherwise is a 'face' miss. A centre in `undefined` has no readable hand.
    """
    species = specification.species
    if not species.centres:
        return ()
    seated = specified_windings_only(observed, species)
    at = {atom: atom for atom in range(observed.mol.GetNumAtoms())}
    notes = arrangement_notes(seated, facts, at)
    if not undefined and (
        notes == specification.notes
        or canonical_notes(seated, specification.rankings, facts).rows == specification.rows
    ):
        return ()  # the notes agree in atom order, or after swapping equivalent metals
    seating = arrangement_notes(_unwound(seated), facts, at)
    misses = []
    for centre, found in zip(species.centres, seated.centres, strict=True):
        atom = centre.atom
        if atom in undefined and centre.hand:
            misses.append(Miss(kind="hand", atoms=(atom,), specified=centre.hand, observed=""))
        elif atom not in undefined and seating[atom] != specification.seating[atom]:
            kind = "hand" if seating[atom] == specification.mirrored.get(atom) else "seating"
            misses.append(Miss(kind=kind, atoms=(atom,), specified=centre.hand, observed=found.hand))
        elif atom not in undefined and notes[atom] != specification.notes[atom]:
            misses.append(Miss(kind="face", atoms=(atom,), specified=_faces(centre), observed=_faces(found)))
    return tuple(misses)


def _faces(centre):
    """Return the windings of `centre`'s haptic faces in vertex order."""
    return ",".join(site.winding for site in centre.vertices if site is not VACANT and len(site.atoms) > 1)


def _agrees_on_observed(specification, observed, stereo):
    """Return whether both sides seat the centres alike on the graph `stereo` tags."""
    facts = LigandFacts(chemical_graph(stereo.mol))
    specified, observed = _on(facts, specification.species), _on(facts, observed)
    rankings = own_rankings(specification.metal_classes, stereo.stereo, specified)
    return (
        canonical_notes(specified, rankings, facts).rows
        == canonical_notes(specified_windings_only(observed, specified), rankings, facts).rows
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinateRequest:
    """Hold one ``coordinate=`` request: the donor atoms each choice seats."""

    given: int | str | list | tuple  # as passed; messages quote it
    choices: tuple  # one tuple of donor atoms per choice; an ambiguous SMARTS gives one choice per atom it matched


def coordinate_request(mol, coordinate):
    """Return ``coordinate=`` read on the engine graph of `mol`, or None when unset.

    It is an atom index, a SMARTS, or a list of either, one per vacancy. A SMARTS matching several atoms gives one
    choice per atom, which only a centre with one vacancy accepts.
    """
    if coordinate is None:
        return None
    graph = engine_graph(mol)
    if isinstance(coordinate, (list, tuple)):
        return CoordinateRequest(given=coordinate, choices=(tuple(resolve_atom(graph, spec) for spec in coordinate),))
    if not isinstance(coordinate, str):
        return CoordinateRequest(given=coordinate, choices=((resolve_atom(graph, coordinate),),))
    pattern = Chem.MolFromSmarts(coordinate)
    if pattern is None:
        raise ValueError(
            f"coordinate={coordinate!r} is not a valid SMARTS; pass a SMARTS pattern, an atom "
            "index, or a list with one spec per vacancy"
        )
    if not (hits := substruct_matches(graph, pattern)):
        raise ValueError(f"coordinate={coordinate!r} matched no atoms")
    return CoordinateRequest(given=coordinate, choices=tuple((match[0],) for match in hits))


def _choices(coordinate, nvac):
    """Return the donor choices of `coordinate` for a centre with `nvac` vacancies."""
    count = len(coordinate.choices)
    if count > 1 and nvac != 1:  # several matches and several vacancies have no one pairing
        raise ValueError(
            f"coordinate={coordinate.given!r} matched {count} atoms for {nvac} vacant site(s); "
            f"name them explicitly as a list, e.g. ['[OX1]', '[OX2]']"
        )
    if count > 1:
        logger.info("coordinate=%r matched %d atoms; embedding %d modes, one per donor", coordinate.given, count, count)
    return coordinate.choices


def vacancy_seatings(species, coordinate, center):
    """Return, per donor choice of `coordinate`, each distinct species seating it in `center`'s vacancies.

    The species carry no conformers. `center` must be a centre of `species`.
    """
    if center not in species.sites:
        metals = [centre_name(c) for c in in_canonical_order(species, graph_with_stereo(species)).centres]
        raise ValueError(
            f"coordinate= seats one metal; name one of {metals} with center=<atom index or element symbol>"
        )
    vacancies = [position for position, site in enumerate(species.sites[center].vertices) if site is VACANT]
    bare = without_conformers(species)
    return [
        # min(): more atoms than vacancies still reaches seat_vacancies, which refuses them
        distinct(
            [
                bare.seat_vacancies(center, atoms, positions)
                for positions in permutations(vacancies, min(len(atoms), len(vacancies)))
            ]
        )
        for atoms in _choices(coordinate, len(vacancies))
    ]


def observed_species(stereo, centres=None):
    """Return the species a conformer shows at `centres`, default every metal, from its `ObservedStereo`."""
    mol = Chem.Mol(stereo.mol)
    metals = metal_indices(mol)
    clear_ez(mol, StereoGraph(mol, metals).locked)  # a coordination-locked E/Z is the arrangement's, not a label's
    facts = LigandFacts(chemical_graph(mol))
    noted = {m: specified_arrangement(mol, m) for m in (metals if centres is None else centres)}
    read = [observed_centre(facts, m, arrangement and arrangement.geometry) for m, arrangement in noted.items()]
    species = Species(mol=facts.graph, centres=tuple(read), stereo=stereo.stereo)
    return species if centres is not None else in_canonical_order(species, mol)


def observed_labels(mol, metal, ids):
    """Return the `centre_label` each conformer in `ids` shows at `metal`, '' where none can be read."""
    try:
        facts = LigandFacts(chemical_graph(mol))
    except ValueError:
        return [""] * len(ids)
    noted = specified_arrangement(mol, metal)
    sphere = collapse_haptic(facts.mol, [d for d, m in facts.donor_bonds if m == metal])
    labels = []
    for cid in map(int, ids):
        one = LigandFacts(Chem.Mol(facts.graph, False, cid))
        try:
            if noted is not None:
                geometry = noted.geometry
            else:
                best = classify_geometry(sphere.mol, metal, sphere.sites, sphere.haptic, cid) if sphere.sites else None
                geometry = None if best is None else best.name
            centre = None if geometry is None else observed_centre(one, metal, geometry)
        except ValueError:  # the donors take no polyhedron
            centre = None
        if centre is None:
            labels.append("")
        else:
            sites = materialized_states(one.graph, (centre,))[metal]
            labels.append(order_label(one.mol, sites.vertices, centre.geometry, sites.haptic))
    return labels
