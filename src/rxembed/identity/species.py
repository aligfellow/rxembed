"""Hold the `Species` record: one graph with its metal arrangements and ligand stereo.

It builds the centres a species seats (`centre_from_slots`, `observed_centre`), orders equivalent metals canonically,
and selects centres by their keys. Its notes are `smiles_io`'s, and comparing it with a conformer is `reading`'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from rxembed.geometry.constraints import is_index
from rxembed.graph import haptic_sites, metal_indices
from rxembed.metal.centres import (
    Centre,
    Site,
    Sites,
    collapse_haptic,
    donor_bonds,
    engine_graph,
    logger,
    materialized_states,
    seated_centre,
    state_with_winding,
    symmetric_metals,
)
from rxembed.metal.polyhedra import (
    HAND_GLYPHS,
    POLYHEDRA,
    VACANT,
    SlotForm,
    chirality_tag,
    describe,
    least_frames,
    record,
    relabels,
    resolve_geometry,
    slot_form,
    vertex_dirs,
)
from rxembed.metal.shape import (
    APICAL_MIN,
    classify_geometry,
    geometry_for,
    shape_gap,
)
from rxembed.metal.slots import input_ordering, order_label, pinned_vertices
from rxembed.metal.sphere import LigandFacts
from rxembed.metal.stereo import (
    MIRROR,
    chelate_links,
    chirality_of,
    donor_classes,
    face_descriptors,
    face_winding,
    site_classes,
)
from rxembed.stereo import (
    LigandStereo,
    apply_point_stereo,
    matches_stereo,
    stereo_from_3d,
)
from rxembed.utils import with_conformers, without_atom_maps

_PT = GetPeriodicTable()


def seat_order(padded, haptic, slots):
    """Return the order that seats each donor of `slots` (vertex -> donor atom) at its vertex.

    A haptic face can be named by any of its ring atoms; unnamed vertices stay vacant. A vertex out of range, a donor
    seated twice or a donor left without a vertex raises, since each would embed a different isomer.
    """
    slot = {d: k for k, d in enumerate(padded) if d is not VACANT}
    for dummy, ring in haptic.items():
        slot.update(dict.fromkeys(ring, slot[dummy]))  # a face is named by any ring atom, not by its centroid dummy
    n = len(padded)
    order = [None] * n
    for v, a in slots.items():
        if not is_index(v) or not 0 <= v < n:  # perception hands numpy ints; a bool is not a vertex
            raise ValueError(f"vertex {v!r} is not one of this geometry's {n} vertices (0-{n - 1})")
        if slot[a] in order:
            raise ValueError(f"donor atom {a} is seated at two vertices")
        order[v] = slot[a]
    spare = [k for k in range(n) if k not in order]
    unseated = [padded[k] for k in spare if padded[k] is not VACANT]
    if unseated:
        raise ValueError(f"donor(s) {unseated} were given no vertex; every donor must be seated ({len(slots)} given)")
    for v in range(n):
        if order[v] is None:
            order[v] = spare.pop(0)  # a VACANT padding slot
    return order


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Species:
    """Hold one chemical graph with its metal centres and ligand stereo.

    `centres` gives each metal's polyhedron, which donor sits at which vertex, and its hand. Records share `mol`, which
    nothing writes to; use `dataclasses.replace` for another hand or seating. Records compare by identity.
    """

    mol: Chem.Mol  # chemical graph, with real elements and donor bonds; input conformers kept
    centres: tuple[Centre, ...] = ()  # one per metal, in the order its producer gives; none for an organic
    stereo: LigandStereo = field(default_factory=LigandStereo)  # ligand stereo ('C16:R'), apart from each `hand`
    sites: dict[int, Sites] = field(init=False)  # each centre's frozen `Sites` (`materialized_states`)

    def __post_init__(self):
        """Derive each centre's sites once."""
        object.__setattr__(self, "sites", materialized_states(self.mol, self.centres))

    @property
    def held_stereo(self):
        """Return the ligand stereo the engine selects seeds for: all of it when there is a centre.

        A centre's sphere windows switch off RDKit's chirality check, so the engine holds that stereo itself; without a
        centre ETKDG keeps the tags.
        """
        return self.stereo if self.centres else LigandStereo()

    def seat_vacancies(self, metal, atoms, positions=None):
        """Return this species with `atoms` bound into the vacancies of `metal`'s centre.

        `positions` names the vacant vertices in atom order; the default is the first vacancies. Each atom must form a
        site of its own: one that would merge with a bound neighbour into a haptic site raises.
        """
        atoms = [int(atom) if is_index(atom) else atom for atom in atoms]
        if not atoms:
            return self
        n_atoms = self.mol.GetNumAtoms()
        invalid = [atom for atom in atoms if not is_index(atom) or not 0 <= atom < n_atoms]
        if invalid:
            raise ValueError(f"coordinate atom indices must be in 0-{n_atoms - 1}; got {invalid}")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"coordinate atoms must be distinct; got {atoms}")
        donors = self.sites[metal].donors
        if set(atoms) & set(donors):
            raise ValueError(f"coordinate atoms are already donors of metal {metal}: {atoms}")
        if set(atoms) & {centre.atom for centre in self.centres}:
            raise ValueError(f"coordinate atoms must be ligand donors, not metal atoms: {atoms}")
        kept = {*haptic_sites(self.mol, donors), *((a,) for a in atoms)}
        if regrouped := sorted(set(haptic_sites(self.mol, [*donors, *atoms])) - kept):
            raise ValueError(
                f"coordinate atom(s) {atoms} would regroup into {regrouped} with a bound neighbour; bind the "
                "whole site in the input"
            )

        centres = list(self.centres)
        k = next(k for k, centre in enumerate(centres) if centre.atom == metal)
        centre = centres[k]
        vacancies = [position for position, site in enumerate(centre.vertices) if site is VACANT]
        if len(atoms) > len(vacancies):
            raise ValueError(
                f"{centre.geometry} {centre_label(LigandFacts(self.mol), centre)} has {len(vacancies)} vacant site(s) "
                f"but {len(atoms)} atom(s) to coordinate"
            )
        sites = list(centre.vertices)
        for atom, position in zip(atoms, vacancies[: len(atoms)] if positions is None else positions, strict=True):
            sites[position] = Site(atoms=(atom,))
        centres[k] = centre._replace(vertices=tuple(sites))
        graph = Chem.RWMol(self.mol)
        for atom in atoms:
            graph.AddBond(atom, metal, Chem.BondType.DATIVE)
        graph.ClearComputedProps()
        graph.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(graph)
        seated = replace(self, mol=graph.GetMol(), centres=tuple(centres))
        facts = LigandFacts(seated.mol)
        vertices, haptic, _winding, _donors = seated.sites[metal]
        centres[k] = centres[k]._replace(hand=chirality_of(facts.mol, centre.geometry, vertices, haptic, facts.roles))
        return replace(seated, centres=tuple(centres))

    def __str__(self):
        """Return one row per centre: element, polyhedron, label, arrangement, hand, then ligand stereo."""
        parts = []
        facts = LigandFacts(self.mol)
        for centre in self.centres:
            if centre.geometry not in POLYHEDRA:
                parts.append(f"{centre_name(centre)} [surrogated]")
                continue
            code = POLYHEDRA[centre.geometry].code
            coarse = centre_label(facts, centre)
            label = f" {coarse}" if coarse else ""
            hand = HAND_GLYPHS.get(centre.hand, "-")
            haptic = haptic_configuration(facts, centre)
            configuration = f" {haptic}" if haptic else ""
            parts.append(
                f"{centre_name(centre)} {code}{label} [{centre_arrangement(facts, centre)}] {hand}{configuration}"
            )
        return f"{' ; '.join(parts)}  {self.stereo or '-'}" if parts else ""

    def __repr__(self):
        """Return a compact notebook representation."""
        return f"Species({str(self)!r})"


def misplaced_pins(species, core, *, angles_only=False):
    """Return the sphere atoms `core` pins away from the vertices `species` specifies, else [].

    Atoms are compared, not symmetry classes, so swapping two equivalent donors counts as another seating. A mirror
    image also counts unless `angles_only`, since compiled constraints hold only angles.
    """
    for centre in species.centres:
        if not centre.geometry:
            continue
        sites = species.sites[centre.atom].vertices
        pinned = pinned_vertices(species.mol, core, centre.atom, sites, centre.geometry)
        dirs = tuple(map(tuple, POLYHEDRA[centre.geometry].vertex_dirs))
        occupied = tuple(v for v, site in enumerate(sites) if site is not VACANT)
        # the mirror image is another seating when the centre has a hand or a chiral winding
        winding = species.sites[centre.atom].winding.values()
        handed = not angles_only and (bool(centre.hand) or any(MIRROR[token] != token for token in winding))
        if pinned and not relabels(dirs, occupied, pinned, handed):
            return sorted(sites[k] for k in pinned.values())
    return []


def centre_name(centre):
    """Return the element and atom index naming `centre`, e.g. ``'Pd0'``."""
    return f"{_PT.GetElementSymbol(centre.atomic_num)}{centre.atom}"


def centre_from_slots(facts, metal, geometry, slots):
    """Return the `Centre` of `metal` with each donor at the slot `slots` gives it, its hand derived.

    Slots left out stay vacant; slot numbers follow `polyhedra.vertex_dirs`. Face windings are read from the conformer,
    if there is one.
    """
    donors = [donor for donor, owner in facts.donor_bonds if owner == metal]
    base, sites, haptic = collapse_haptic(facts.mol, donors)  # each haptic face -> one centroid vertex
    if not sites:
        logger.warning("Species: the metal has no donor bonds, so this %s sphere is vacant; use dative bonds", geometry)
    n = POLYHEDRA[geometry].cn
    if len(sites) > n:
        raise ValueError(f"{geometry} has {n} coordination sites but metal {metal} has {len(sites)} donor site(s)")
    padded = [*sites, *[VACANT] * (n - len(sites))]
    vertices = [padded[k] for k in seat_order(padded, haptic, slots)]
    winding = observed_haptic_windings(facts.mol, metal, donors, haptic)
    hand = chirality_of(base, geometry, vertices, haptic, facts.roles)
    centre = facts.centre(metal)
    return seated_centre(centre, geometry, vertices, haptic, winding, hand)


def graph_with_stereo(species):
    """Return a copy of `species.mol` with each selected donor hand tagged, conformers kept."""
    mol = Chem.Mol(species.mol)
    points = species.stereo.points
    missing = {
        donor
        for donor, _metal in donor_bonds(mol)
        if donor in points and mol.GetAtomWithIdx(donor).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    }
    apply_point_stereo(mol, species.stereo, missing)
    return mol


def centre_label(facts, centre):
    """Return the coarse isomer label of one centre, such as cis or fac."""
    if not centre.geometry:
        return ""
    vertices, haptic, _winding, _donors = materialized_states(facts.graph, (centre,))[centre.atom]
    return order_label(facts.mol, vertices, centre.geometry, haptic)


def centre_arrangement(facts, centre):
    """Return which ligand sits at each vertex of one centre, ``·`` at a vacancy."""
    vertices, haptic, winding, donors = materialized_states(facts.graph, (centre,))[centre.atom]
    descriptors = face_descriptors(facts.mol, donors, haptic, winding)
    base_labels = [_site_symbol(facts.mol, donor, haptic, winding, descriptors) for donor in vertices]
    labels = list(base_labels)
    classes = site_classes(facts.mol, vertices, haptic, facts.roles)
    for position, donor in enumerate(vertices):
        if donor not in haptic:
            continue
        same_shape = [site for site, face in haptic.items() if len(face) == len(haptic[donor])]
        if len({classes[site] for site in same_shape}) > 1:
            anchor = min(haptic[donor])
            labels[position] += f"@{facts.mol.GetAtomWithIdx(anchor).GetSymbol()}{anchor}"
    return " ".join(labels)


def _site_symbol(mol, donor, haptic, winding, descriptors):
    """Return the short label of one site: element and index, ηn for a haptic face, ``·`` when vacant."""
    if donor is VACANT:
        return "·"
    if donor in haptic:
        ring = haptic[donor]
        return f"η{len(ring)}{descriptors.get(donor, winding.get(donor, ''))}"
    return f"{mol.GetAtomWithIdx(donor).GetSymbol()}{donor}"


def haptic_configuration(facts, centre):
    """Return 'rac' or 'meso' for a centre with one pair of equivalent named faces, else ''."""
    vertices, haptic, winding, donors = materialized_states(facts.graph, (centre,))[centre.atom]
    faces = list(face_descriptors(facts.mol, donors, haptic, winding))
    if len(faces) != 2 or set(faces) != set(winding):  # noqa: PLR2004 - a pair of faces
        return ""
    left, right = faces
    # the faces are equivalent when swapping their windings leaves the centre's state unchanged
    swapped = [
        centre_state(facts, state_with_winding(centre, vertices, wound))
        for wound in ({left: "+", right: "-"}, {left: "-", right: "+"})
    ]
    if swapped[0] != swapped[1]:
        return ""
    mirrored = [
        centre_state(facts, state_with_winding(centre, vertices, wound))
        for wound in (winding, {face: MIRROR[value] for face, value in winding.items()})
    ]
    return "meso" if mirrored[0] == mirrored[1] else "rac"


def observed_haptic_windings(mol, metal, donors, haptic):
    """Return the winding of each haptic face read from the first conformer.

    Pass the real ligand graph: centroid dummies change RDKit's symmetry classes and give a winding the output Mol
    cannot reproduce.
    """
    if not haptic or mol.GetNumConformers() == 0:
        return {}
    pos = mol.GetConformer().GetPositions()
    ranks = donor_classes(mol, donors)
    eta2 = any(len(face) == 2 for face in haptic.values())  # noqa: PLR2004
    eta2_ranks = list(Chem.ComputeAtomCIPRanks(without_atom_maps(mol))) if eta2 else None
    return {
        dummy: sign
        for dummy, face in haptic.items()
        if (sign := face_winding(mol, pos, metal, face, ranks, eta2_ranks))
    }


def canonical_metals(mol, metals, state=None):
    """Return `metals` in canonical order on graph `mol`: graph rank, then `state(metal)` on a tie.

    Equivalent metals share their lowest rank, since a delocalised charge can split their ranks. Metals tied on rank and
    state are copies of one arrangement, so either order names the same species.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeAtomMaps=False))
    for group in symmetric_metals(mol):
        least = min(ranks[metal] for metal in group)
        for metal in group:
            ranks[metal] = least
    tied = {metal for metal in metals if [ranks[other] for other in metals].count(ranks[metal]) > 1}
    states = {metal: state(metal) for metal in tied} if state else {}
    return sorted(metals, key=lambda metal: (ranks[metal], states.get(metal, ()), metal))


class CentreState(NamedTuple):
    """Hold one centre's arrangement without atom indices, so equivalent centres compare equal."""

    geometry: str
    hand: str
    form: SlotForm  # the least seating of its site classes and chelate links


def centre_state(facts, centre):
    """Return the `CentreState` of `centre`."""
    vertices, haptic, winding, _donors = materialized_states(facts.graph, (centre,))[centre.atom]
    keys = site_keys(facts, vertices, haptic, winding)
    links = chelate_links(facts.mol, vertices, haptic)
    form = slot_form(least_frames(vertex_dirs(centre.geometry), keys, links)[0], keys, links)
    return CentreState(geometry=centre.geometry, hand=centre.hand, form=form)


def in_canonical_order(species, mol):
    """Return `species` with its centres in canonical order on its graph `mol`.

    The engine and its messages use this order; names use the order the CX writer gives (`written_seating`).
    """
    by_atom = {centre.atom: centre for centre in species.centres}
    facts = LigandFacts(species.mol)
    order = canonical_metals(mol, list(by_atom), lambda metal: centre_state(facts, by_atom[metal]))
    return replace(species, centres=tuple(by_atom[metal] for metal in order))


def retained_geometry(mol, metal, sites, haptic):
    """Return the polyhedron the conformer's donors take, else the site count's default, or None.

    Logs which, and warns of a poor fit or a near tie.
    """
    apical = any(len(face) >= APICAL_MIN for face in haptic.values())
    best = classify_geometry(mol, metal, sites, haptic) if mol.GetNumConformers() else None
    if best is not None:
        clause = best.reading.clause(best.name)
        if best.reading.verdict == "distorted":
            logger.warning("geometry: no shape fits; nearest %s; pass geometry= to state it", clause)
        elif best.tied:
            other = record(best.reading.next_name).code
            logger.warning(
                "geometry: metal %d is a near-tie, %s; pass geometry='%s' for the other", metal, clause, other
            )
        logger.info("geometry: input is %s", describe(best.name))
        return best.name
    name = geometry_for(len(sites), has_apical=apical)
    reason = "an apical eta>=3 face fills more than one site" if apical else "no polyhedron read from a conformer"
    logger.info("metal: %s -> CN %d default %s", reason, len(sites), describe(name))
    return name


def fitted_centre(facts, metal, geometry):
    """Return the `Centre` of `metal` in `geometry`, seated where the conformer puts its donors.

    The donors are fitted in either hand, put in the canonical frame of their site classes and chelate links, and the
    hand and face windings are read from that seating. Above `polyhedra.seat_by_alignment`'s exhaustive limit the fit is
    approximate. None when the donors do not fill the slots.
    """
    donors = [donor for donor, owner in facts.donor_bonds if owner == metal]
    base, sites, haptic = collapse_haptic(facts.mol, donors)
    padded = [*sites, *[VACANT] * (POLYHEDRA[geometry].cn - len(sites))]
    classes = site_classes(base, padded, haptic, facts.roles)
    order = input_ordering(base, metal, padded, geometry, haptic, classes=classes)
    if order is None:
        return None
    vertices = [padded[k] for k in order]
    hand = chirality_of(base, geometry, vertices, haptic, classes=classes)
    winding = observed_haptic_windings(facts.mol, metal, donors, haptic)
    return seated_centre(facts.centre(metal), geometry, vertices, haptic, winding, hand)


def observed_centre(facts, metal, geometry=None):
    """Return the `Centre` the conformer shows at `metal`, in `geometry` or the best-fitting polyhedron.

    Raises when the donors do not take that polyhedron, the metal has no donor, or no template exists.
    """
    donors = [donor for donor, owner in facts.donor_bonds if owner == metal]
    centre = facts.centre(metal)
    if not donors:
        label = [centre_name(centre)]
        raise ValueError(f"metal centre(s) {label} have no donor bonds, so there is no coordination geometry to retain")
    if not facts.mol.GetNumConformers():
        raise ValueError(f"reading the arrangement of {centre_name(centre)} needs an input geometry (a conformer)")
    requested = geometry = resolve_geometry(geometry)
    if geometry is None:
        base, sites, haptic = collapse_haptic(facts.mol, donors)
        geometry = retained_geometry(base, metal, sites, haptic) or f"{len(sites)}-coordinate"
    if geometry not in POLYHEDRA:
        raise ValueError(f"no polyhedron template for {geometry!r}; add a POLYHEDRA row")
    fitted = fitted_centre(facts, metal, geometry)
    if fitted is None or not takes_shape(facts.mol, fitted):
        taken = (
            f"the requested {geometry} geometry" if requested else f"the {geometry} geometry their sites are read in"
        )
        raise ValueError(
            f"the donors of {centre_name(centre)} do not take {taken}; rx.isomers(mol, observed=False) lists its "
            "arrangements"
        )
    return fitted


def shape_readings(mol, species, cid=-1):
    """Return each centre's `ShapeReading` of its polyhedron on conformer `cid` of `mol`, by metal."""
    return {
        c.atom: shape_gap(mol, c.atom, species.sites[c.atom].vertices, species.sites[c.atom].haptic, c.geometry, cid)
        for c in species.centres
    }


def takes_shape(mol, centre):
    """Return whether the conformer of `mol` reads as `centre`'s polyhedron."""
    vertices, haptic, _winding, _donors = materialized_states(mol, (centre,))[centre.atom]
    return shape_gap(mol, centre.atom, vertices, haptic, centre.geometry).matches


class ObservedStereo(NamedTuple):
    """Hold the ligand stereo one conformer shows, read once (`observe_stereo`)."""

    mol: Chem.Mol  # the graph with the conformer, tagged with the stereo it shows
    stereo: LigandStereo  # that stereo, from `stereo_from_3d`


def observe_stereo(graph, conf):
    """Return the ligand stereo conformer `conf` shows on `graph`."""
    mol = Chem.Mol(graph)
    mol.RemoveAllConformers()
    mol.AddConformer(Chem.Conformer(conf), assignId=False)
    return ObservedStereo(mol=mol, stereo=stereo_from_3d(mol, metal_indices(mol), apply=True))


def selected_centres(centres, center=None):
    """Return the `centres` that `center` names: all for None, else the one with that index or element.

    An element symbol that names several centres raises.
    """
    centres = list(centres)
    if center is None:
        return centres
    if not (is_index(center) or isinstance(center, str)):
        raise TypeError(f"center must be an atom index or element symbol; got {center!r}")
    hits = [centre for centre in centres if centre.atom == center or _PT.GetElementSymbol(centre.atomic_num) == center]
    if len(hits) > 1:
        raise ValueError(f"center={center!r} names {[centre_name(c) for c in hits]}; name one by its atom index")
    return hits


def _ligand_smiles(mol, donor):
    """Return the SMILES of a donor's ligand fragment, rooted at the donor, without hydrogens."""
    mappings = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mappings)
    fragment, mapping = next(
        (Chem.Mol(without_atom_maps(f)), m) for f, m in zip(fragments, mappings, strict=True) if donor in m
    )
    marker = max((atom.GetAtomMapNum() for atom in fragment.GetAtoms()), default=0) + 1
    fragment.GetAtomWithIdx(mapping.index(donor)).SetAtomMapNum(marker)
    fragment = Chem.RemoveHs(fragment, sanitize=False)
    root = next(atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomMapNum() == marker)
    fragment.GetAtomWithIdx(root).SetAtomMapNum(0)
    return Chem.MolToSmiles(fragment, rootedAtAtom=root, canonical=True, isomericSmiles=True)


def site_details(species):
    """Return one line per site relation of `species`: groups or trans pairs, faces, ambiguous donors."""
    lines = []
    mol = engine_graph(species.mol)
    for centre in species.centres:
        if centre.geometry not in POLYHEDRA:
            continue
        vertices, haptic, winding, donors = species.sites[centre.atom]
        descriptors = face_descriptors(mol, donors, haptic, winding)
        prefix = f"       {centre_name(centre)}"
        polyhedron = POLYHEDRA[centre.geometry]
        if polyhedron.site_groups:
            groups = []
            for name, positions in polyhedron.site_groups:
                labels = (_site_symbol(mol, vertices[v], haptic, winding, descriptors) for v in positions)
                groups.append(f"{name}: {' '.join(labels)}")
            lines.append(f"{prefix} {'; '.join(groups)}")
        elif polyhedron.cis_trans:
            pairs = [
                f"{_site_symbol(mol, vertices[a], haptic, winding, descriptors)}-"
                f"{_site_symbol(mol, vertices[b], haptic, winding, descriptors)}"
                for a, b in sorted(polyhedron.trans_pairs)
            ]
            if pairs:
                lines.append(f"{prefix} trans: {', '.join(pairs)}")

        if len(haptic) > 1:
            faces = [
                f"{_site_symbol(mol, dummy, haptic, winding, descriptors)}({','.join(map(str, face))})"
                for dummy, face in haptic.items()
            ]
            lines.append(f"{prefix} faces: {', '.join(faces)}")

        ring_atoms = {atom for face in haptic.values() for atom in face}
        donors = [donor for donor in donors if donor not in ring_atoms]
        classes = donor_classes(mol, donors)
        by_symbol = {}
        for donor in donors:
            by_symbol.setdefault(mol.GetAtomWithIdx(donor).GetSymbol(), []).append(donor)
        ambiguous = {
            donor
            for same_element in by_symbol.values()
            if len({classes[donor] for donor in same_element}) > 1
            for donor in same_element
        }
        for donor in sorted(ambiguous):
            label = _site_symbol(mol, donor, haptic, winding, descriptors)
            lines.append(f"{prefix} {label}: {_ligand_smiles(mol, donor)}")
    return lines


def _haptic_matches(facts, centres, haptic):
    """Return whether a selected centre's faces match the `haptic` selector."""
    if haptic is None:
        return True
    aliases = {"Rp": "Rₚ", "Sp": "Sₚ", "R_p": "Rₚ", "S_p": "Sₚ"}
    if isinstance(haptic, dict):
        for centre in centres:
            _vertices, faces, winding, donors = materialized_states(facts.graph, (centre,))[centre.atom]
            descriptors = face_descriptors(facts.mol, donors, faces, winding)
            if all(
                len(matches := [dummy for dummy, face in faces.items() if atom in face]) == 1
                and descriptors.get(matches[0]) == aliases.get(wanted, wanted)
                for atom, wanted in haptic.items()
            ):
                return True
        return False
    wanted = aliases.get(haptic, haptic)
    for centre in centres:
        _vertices, faces, winding, donors = materialized_states(facts.graph, (centre,))[centre.atom]
        if haptic_configuration(facts, centre) == wanted:
            return True
        descriptors = face_descriptors(facts.mol, donors, faces, winding)
        if wanted in {"Rₚ", "Sₚ"} and len(descriptors) > 1:
            anchors = [min(faces[dummy]) for dummy in descriptors]
            raise ValueError(
                f"haptic={haptic!r} leaves {len(descriptors)} orientable faces unspecified; "
                f"use haptic={{face_atom: hand}} with face atoms {anchors}, or rac/meso"
            )
        if wanted in descriptors.values():
            return True
    return False


SELECTORS = frozenset({"geometry", "label", "arrangement", "hand", "haptic", "stereo", "center"})


def matches(species, keys):
    """Return whether `species` matches the validated selectors `keys`; None matches anything.

    `geometry`, `label`, `arrangement` and `hand` must hold together at one centre `center` names. `stereo` compares
    ligand stereo. `haptic` takes ``'Rₚ'``/``'Sₚ'`` for a named face, a ``{face atom: hand}`` dict, or
    ``'rac'``/``'meso'`` for a pair of equivalent faces.
    """
    geometry, hand = keys.get("geometry"), chirality_tag(keys.get("hand"))
    label, arrangement, stereo, center = (keys.get(key) for key in ("label", "arrangement", "stereo", "center"))
    centres = selected_centres(species.centres, center)
    facts = LigandFacts(species.mol)
    if center is not None and not centres:
        return False
    return (
        (
            all(name is None for name in (geometry, label, arrangement, hand))
            or any(
                geometry in (None, centre.geometry)
                and (label is None or centre_label(facts, centre) == label)
                and (arrangement is None or centre_arrangement(facts, centre) == arrangement)
                and (hand is None or centre.hand == hand)
                for centre in centres
            )
        )
        and (stereo is None or matches_stereo(species.stereo, stereo))
        and _haptic_matches(facts, centres, keys.get("haptic"))
    )


def site_keys(facts, vertices, haptic, winding):
    """Return one key per vertex, None when vacant, that ties exactly for equivalent sites.

    A donor's key is its symmetry class; a haptic face adds its winding, so identical Cp rings tie. Keys hold no atom
    index, so every writer agrees.
    """
    classes = site_classes(facts.mol, vertices, haptic, facts.roles)
    keys = []
    for d in vertices:
        if d is VACANT:
            keys.append(None)
        elif d in haptic:
            wind = winding.get(d, "")
            keys.append((classes[d], wind))
        else:
            keys.append((classes[d], ""))
    return keys


def without_conformers(species):
    """Return `species` with its Mol carrying no conformer."""
    return replace(species, mol=with_conformers(species.mol, species.mol, ()))
