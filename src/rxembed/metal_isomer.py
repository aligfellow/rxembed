"""Represent one selected coordination isomer and an indexable set of candidates."""

from __future__ import annotations

from copy import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from . import metal_core as _core
from . import metal_slots as _slots
from . import metal_stereo as _coord_stereo
from . import stereo as _stereo
from .constraints import Constraints, _is_index, add_distance, compose
from .metal_constraints import _model_distance_window, compile_constraints, coordination_from_geometry, resolve_lengths
from .metal_core import (
    _APICAL_MIN,
    VACANT,
    _collapse_haptic,
    _reject_metal_bonds,
    classify_geometry,
    geometry_for,
    logger,
    metal_indices,
    n_sites,
    restore_metal,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_polyhedron import (
    POLYHEDRA,
    _vertex_angle,
    canonical_slots,
    chirality_tag,
    describe,
    resolve_geometry,
    vertex_dirs,
)

_PT = GetPeriodicTable()
_PAIR = 2
_HAPTIC_FACE_MIN = 3


def length_source(mol, lengths):
    """Report and resolve where the M-donor windows come from."""
    positions, note = resolve_lengths(mol, lengths)
    if note:
        logger.info("metal: M-donor windows from %s", note)
    return "input" if positions is not None else "model"


def seat_order(padded, haptic, sites):
    """Map a `sites` (vertex -> donor ATOM index) onto the internal order (vertex -> position in `padded`).

    `sites` is a ``{vertex: atom}`` dict or a vertex-ordered list. A haptic face is one vertex, named by any
    of its ring atoms. Vertices left unnamed take the VACANT padding (a coordination pocket). Every failure is
    loud, because an unknown, repeated or unseated donor would otherwise embed a different isomer silently.
    """
    if not isinstance(sites, dict):
        if not isinstance(sites, (list, tuple, np.ndarray)):  # a set seats every donor exactly once, so the
            raise TypeError(  # guard below passes, and its iteration order then picks an arbitrary isomer
                f"sites must be a {{vertex: atom}} dict or a vertex-ordered list, got {type(sites).__name__}: "
                f"an unordered collection cannot say which donor sits at which vertex"
            )
        sites = {v: a for v, a in enumerate(sites) if a is not None and a != VACANT}
    slot = {d: k for k, d in enumerate(padded) if d != VACANT}
    for dummy, ring in haptic.items():
        slot.update(dict.fromkeys(ring, slot[dummy]))  # a face is named by any ring atom, not by its centroid dummy
    n = len(padded)
    order = [None] * n
    for v, a in sites.items():
        if not _is_index(v) or not 0 <= v < n:  # perception hands numpy ints; a bool is not a vertex
            raise ValueError(f"vertex {v!r} is not one of this geometry's {n} vertices (0-{n - 1})")
        if a not in slot:
            raise ValueError(f"sites[{v}] = atom {a} is not a donor of the metal; its donors are {sorted(slot)}")
        if order[v] is not None:
            raise ValueError(f"vertex {v} is given two donors ({padded[order[v]]} and {a})")
        if slot[a] in order:
            raise ValueError(f"donor atom {a} is seated at two vertices")
        order[v] = slot[a]
    spare = [k for k in range(n) if k not in order]
    unseated = [padded[k] for k in spare if padded[k] != VACANT]
    if unseated:
        raise ValueError(f"donor(s) {unseated} were given no vertex; every donor must be seated ({len(sites)} given)")
    for v in range(n):
        if order[v] is None:
            order[v] = spare.pop(0)  # a VACANT padding slot
    return order


def _warn_undefined_ligand_stereo(mol):
    """Warn when a ligand carries an undefined stereocentre, which one `Isomer` would pool both hands of.

    `Isomer` is a single species, but an unspecified centre embeds as a mixture: RDKit assigns each seed a
    hand at random, so one result carries both enantiomers and any RMSD/energy prune downstream cross-prunes
    two distinct species. `enumerate_isomers` expands them into separate candidates instead.
    """
    centres = _stereo.unassigned_centres(mol, exclude=metal_indices(mol))
    if centres:
        logger.warning(
            "Isomer: ligand stereo element(s) %s undefined; seeds mix both hands (use enumerate_isomers)",
            [c[0] if len(c) == 1 else f"{c[0]}={c[1]}" for c in centres],
        )


class Isomer:
    """Carry real-atom metal identity separately from its physical constraints.

    `centres` is the chemical state: metal identities, polyhedra, vertex seating and stereo. Numerical
    ETKDG/UFF constraints are compiled from that state only when `cons` is requested. Build a known isomer
    with ``Isomer(mol, geometry, sites)``; `enumerate_isomers` builds unknown and multi-centre states.
    """

    mol: Chem.Mol
    _centres: tuple  # one real-atom `MetalState` per metal, primary first
    donor_bonds: list  # stripped donor-metal bonds, re-added dative on output
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')
    stereo_label: str = ""  # ligand stereoisomer tag ('C16:R'), distinct from the metal-centre `chirality`

    @property
    def mol(self):
        """Return this candidate's independent public molecule, copying a shared graph on first access."""
        if self._mol is None:
            self._mol = Chem.Mol(self._shared_mol)
        return self._mol

    @mol.setter
    def mol(self, value):
        self._mol = self._shared_mol = value

    @property
    def _graph(self):
        return self._shared_mol if self._mol is None else self._mol

    def __init__(self, mol, geometry, sites, lengths="auto"):
        """Build a known isomer by assigning donor atom indices to polyhedron slots.

        `geometry` accepts a registry name or 3-letter code. `sites` is a ``{slot: atom}`` mapping or a
        slot-ordered list; omitted slots remain vacant. `lengths` selects input or model M-donor windows.
        Slot numbering is defined by `metal_polyhedron.vertex_dirs`.
        """
        geom = resolve_geometry(geometry)
        if geom not in POLYHEDRA:
            raise ValueError(
                f"unknown geometry {geometry!r}; available: {sorted(POLYHEDRA)} "
                f"(or a code: {sorted(p.code for p in POLYHEDRA.values() if p.code)})"
            )
        if len(metal_indices(mol)) > 1:
            raise NotImplementedError(
                "Isomer() seats one metal centre; enumerate a multi-metal complex with "
                "enumerate_isomers(center=), which retains the spectator metal(s)"
            )
        _warn_undefined_ligand_stereo(mol)
        base, m, donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = _collapse_haptic(base, donors)  # each haptic face -> one centroid vertex
        if not donors:
            logger.warning(
                "Isomer: the metal has no donor bonds, so this %s sphere is vacant; use dative bonds",
                geom,
            )
        n = n_sites(geom)
        if len(donors) > n:
            raise ValueError(f"{geom} has {n} coordination sites but the metal has {len(donors)} donor site(s)")
        padded = list(donors) + [VACANT] * (n - len(donors))
        order = seat_order(padded, haptic, sites)
        vertices = [padded[k] for k in order]
        # the centroid dummy is transient embed scaffolding, so the stored mol/donors are real (see `haptic`)
        real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
        stored = strip_phantoms(base, set(haptic))
        lengths = length_source(stored, lengths)
        roles = coordination_roles([(d, m) for d in real_donors], [(m, real_z, real_q)])
        hand = _coord_stereo.chirality_of(base, geom, vertices, haptic, roles)
        winding = measured_haptic_windings(base, m, real_donors, haptic)
        self.mol = stored
        self.donor_bonds = [(d, m) for d in real_donors]
        self._centres = (_core.from_vertices(m, real_z, real_q, geom, vertices, haptic, winding.items(), hand),)
        self._base_cons = Constraints()
        self._constrained_metals = frozenset({m})
        self._lengths = lengths
        self._length_mol = Chem.Mol(stored)  # retain measured input lengths if the public Mol is edited
        self._graft_ref = {}
        self.stereo_ref, self.stereo_label = None, ""

    @classmethod
    def _from_state(
        cls,
        mol,
        centres,
        donor_bonds=(),
        *,
        constraints=None,
        constrained_metals=(),
        lengths="model",
        graft_ref=None,
        stereo_ref=None,
        stereo_label="",
        shared_mol=False,
    ):
        """Build an isomer from real-atom states and optional pre-composed constraints."""
        iso = cls.__new__(cls)
        if shared_mol:
            iso._mol, iso._shared_mol = None, mol
        else:
            iso.mol = Chem.Mol(mol)
        iso._centres, iso.donor_bonds = tuple(centres), list(donor_bonds)
        if not iso._centres:
            raise ValueError("an Isomer needs at least one metal state")
        atoms = [state.atom for state in iso._centres]
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"an Isomer has repeated metal states: {atoms}")
        constrained_metals = frozenset(constrained_metals)
        unknown = set(constrained_metals) - set(atoms)
        if unknown:
            raise ValueError(f"constraints name unknown metal(s): {sorted(unknown)}")
        iso._base_cons = Constraints() if constraints is None else constraints
        iso._constrained_metals = constrained_metals
        iso._lengths = lengths
        iso._length_mol = iso._graph if shared_mol else Chem.Mol(iso._graph)
        iso._graft_ref = dict(graft_ref or {})
        iso.stereo_ref, iso.stereo_label = stereo_ref, stereo_label
        return iso

    @property
    def centres(self):
        """Return immutable metal states in primary-centre order."""
        return self._centres

    @property
    def metals(self):
        """Return restore identities in primary-centre order."""
        return tuple((state.atom, state.atomic_num, state.charge) for state in self.centres)

    @property
    def cons(self):
        """Compile physical coordination constraints from this immutable state."""
        return compile_constraints(
            self._graph,
            self.centres,
            length_mol=self._length_mol,
            base=self._base_cons,
            constrained_metals=self._constrained_metals,
            lengths=self._lengths,
        )

    @property
    def metal(self):
        """Return the primary metal atom index."""
        return self.centres[0].atom

    @property
    def real_z(self):
        """Return the primary metal's real atomic number."""
        return self.centres[0].atomic_num

    @property
    def real_q(self):
        """Return the primary metal's formal charge."""
        return self.centres[0].charge

    @property
    def spectator_metals(self):
        """Return restore identities for every non-primary metal."""
        return [(state.atom, state.atomic_num, state.charge) for state in self.centres[1:]]

    @property
    def geometry(self):
        """Return the primary coordination geometry."""
        return self.centres[0].geometry

    @property
    def vertices(self):
        """Return the primary sphere's donors in vertex order."""
        return _core.materialized_state(self, self.centres[0])[0]

    @property
    def haptic(self):
        """Return primary haptic centroid-to-face mappings."""
        return _core.materialized_state(self, self.centres[0])[1]

    @property
    def haptic_winding(self):
        """Return primary canonical haptic winding signs."""
        return _core.materialized_state(self, self.centres[0])[2]

    @property
    def chirality(self):
        """Return primary metal-centre handedness."""
        return self.centres[0].hand

    @property
    def donors(self):
        """Return the primary sphere's real donor atoms."""
        state = self.centres[0]
        if not state.geometry:
            return [donor for donor, metal in self.donor_bonds if metal == self.metal]
        donors = [site for site in state.vertices if isinstance(site, int)]
        return donors + sorted(
            {atom for site in state.vertices if isinstance(site, _core.HapticSite) for atom in site.atoms}
        )

    @property
    def label(self):
        """Return the primary sphere's conventional coarse isomer label."""
        return _state_label(self, self.centres[0])

    def _with_stereo(self, centres):
        """Return a shallow copy with hand or winding metadata replaced on unchanged seatings."""
        centres = tuple(centres)
        by_metal = {state.atom: state for state in centres}
        if set(by_metal) != {state.atom for state in self.centres}:
            raise ValueError("replacement metal states do not match this isomer")
        out = copy(self)
        if self._mol is not None:
            out.mol = Chem.Mol(self._mol)
        out._centres = centres
        return out

    def _seat_vacancies(self, atoms):
        """Return this identity with real donor atoms seated in primary vacancies, in slot order."""
        atoms = [int(atom) if _is_index(atom) else atom for atom in atoms]
        if not atoms:
            return self
        n_atoms = self.mol.GetNumAtoms()
        invalid = [atom for atom in atoms if not _is_index(atom) or not 0 <= atom < n_atoms]
        if invalid:
            raise ValueError(f"coordinate atom indices must be in 0-{n_atoms - 1}; got {invalid}")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"coordinate atoms must be distinct; got {atoms}")
        if set(atoms) & set(self.donors):
            raise ValueError(f"coordinate atoms are already donors of metal {self.metal}: {atoms}")
        if set(atoms) & {state.atom for state in self.centres}:
            raise ValueError(f"coordinate atoms must be ligand donors, not metal atoms: {atoms}")

        state = self.centres[0]
        vacancies = [position for position, site in enumerate(state.vertices) if site is None]
        if len(atoms) > len(vacancies):
            raise ValueError(
                f"{self.geometry} {self.label} has {len(vacancies)} vacant site(s) "
                f"but {len(atoms)} atom(s) to coordinate"
            )
        sites = list(state.vertices)
        for atom, position in zip(atoms, vacancies[: len(atoms)], strict=True):
            sites[position] = atom
        active = state._replace(vertices=tuple(sites))

        out = copy(self)
        out.mol = Chem.Mol(self.mol)
        out.donor_bonds = [*self.donor_bonds, *((atom, self.metal) for atom in atoms)]
        out._base_cons = self._base_cons.copy()
        out._constrained_metals = self._constrained_metals | {self.metal}
        out._length_mol = Chem.Mol(self._length_mol)
        out._graft_ref = dict(self._graft_ref)
        out._centres = (active, *self.centres[1:])
        vertices, haptic, _winding, donors = _core.materialized_state(out, active)
        hand = _coord_stereo.chirality_of(out.mol, active.geometry, vertices, haptic, isomer_roles(out))
        out._centres = (active._replace(hand=hand), *self.centres[1:])

        # Existing donors may retain measured lengths; a newly seated donor has no measured M-L state.
        for atom in atoms:
            add_distance(
                out._base_cons.distances,
                self.metal,
                atom,
                *_model_distance_window(out.mol, self.metal, atom, self.real_z, donors),
            )
        return out

    def restore(self, mol=None):
        """Swap this isomer's surrogated metal(s) back to their real element and formal charge; return `mol`.

        Acts on any molecule sharing this isomer's atom indexing, defaulting to its own. The charge matters:
        restoring only Z leaves an M(0) among anionic ligands and every real energy runs at the wrong total.
        """
        mol = self.mol if mol is None else mol
        for state in self.centres:
            restore_metal(mol, state.atom, state.atomic_num, state.charge)
        return mol

    @property
    def haptic_configuration(self):
        """Return ``'rac'``/``'meso'`` for an interchangeable pair of named haptic faces, else ``''``."""
        return _state_haptic_configuration(self, self.centres[0])

    def summary(self, details=False):
        """Print this isomer in the same table used by `IsomerSet.summary`."""
        return IsomerSet([self]).summary(details=details)

    def __str__(self):
        """Return the compact state row without its table index."""
        return _state_text(self)

    def __repr__(self):
        """Return a compact notebook representation."""
        return f"Isomer({str(self)!r})"


def coordination_roles(donor_bonds, metals):
    """Return donor-to-metal identities for constitutional ranking on the stripped graph."""
    identity = {metal: (atomic_num, charge) for metal, atomic_num, charge in metals}
    return [(donor, metal, *identity[metal]) for donor, metal in donor_bonds]


def primary_first(metals, primary):
    """Return metal identities with `primary` first and every other order retained."""
    return tuple(item for item in metals if item[0] == primary) + tuple(item for item in metals if item[0] != primary)


def isomer_roles(iso):
    """Return every stated coordination role carried by one isomer."""
    return coordination_roles(iso.donor_bonds, iso.metals)


def _state_label(iso, state):
    """Return one centre's conventional coarse isomer label."""
    if not state.geometry:
        return ""
    vertices, haptic, _winding, _donors = _core.materialized_state(iso, state)
    return _slots.order_label(iso._graph, vertices, state.geometry, range(len(vertices)), haptic)


def _state_arrangement(iso, state):
    """Format one centre's readable per-vertex ligand arrangement."""
    vertices, haptic, winding, donors = _core.materialized_state(iso, state)
    descriptors = _coord_stereo.face_descriptors(iso._graph, donors, haptic, winding)
    base_labels = [_site_symbol(iso._graph, donor, haptic, winding, descriptors) for donor in vertices]
    labels = list(base_labels)
    classes = _coord_stereo.site_classes(iso._graph, vertices, haptic, isomer_roles(iso))
    for position, donor in enumerate(vertices):
        if donor not in haptic:
            continue
        same_shape = [site for site, face in haptic.items() if len(face) == len(haptic[donor])]
        if len({classes[site] for site in same_shape}) > 1:
            anchor = min(haptic[donor])
            labels[position] += f"@{iso._graph.GetAtomWithIdx(anchor).GetSymbol()}{anchor}"
    return " ".join(labels)


def _site_symbol(mol, donor, haptic, winding, descriptors):
    """Return one compact coordination-site label."""
    if donor == VACANT:
        return "·"
    if donor in haptic:
        ring = haptic[donor]
        return f"η{len(ring)}{descriptors.get(donor, winding.get(donor, ''))}"
    return f"{mol.GetAtomWithIdx(donor).GetSymbol()}{donor}"


def winding_signature(iso, state, winding):
    """Return canonical-slot identity for one centre's haptic winding assignment."""
    vertices, haptic, _stored, _donors = _core.materialized_state(iso, state)
    classes = _coord_stereo.site_classes(iso._graph, vertices, haptic, isomer_roles(iso))
    keys = [
        None if donor == VACANT else (classes[donor], winding.get(donor, "") if donor in haptic else "")
        for donor in vertices
    ]
    slots = canonical_slots(
        vertex_dirs(state.geometry), keys, _coord_stereo.chelate_edges(iso._graph, vertices, haptic)
    )
    return tuple(key for _slot, key in sorted(zip(slots, keys, strict=True)))


def _state_haptic_configuration(iso, state):
    """Return rac/meso for one interchangeable pair of named haptic faces."""
    _vertices, haptic, winding, donors = _core.materialized_state(iso, state)
    descriptors = _coord_stereo.face_descriptors(iso._graph, donors, haptic, winding)
    windings = {d: w for d, w in winding.items() if len(haptic.get(d, ())) >= _HAPTIC_FACE_MIN}
    faces = [d for d in descriptors if d in windings]
    if len(faces) != _PAIR or set(faces) != set(windings):
        return ""
    left, right = faces
    if winding_signature(iso, state, {left: "+", right: "-"}) != winding_signature(iso, state, {left: "-", right: "+"}):
        return ""
    mirror = {face: "-" if value == "+" else "+" for face, value in winding.items()}
    current = winding_signature(iso, state, winding)
    return "meso" if current == winding_signature(iso, state, mirror) else "rac"


def arrangement(iso):
    """Format a readable per-vertex ligand arrangement, e.g. ``'N3 Cl5 Cl6 ·'`` (``·`` = a vacant site).

    The unambiguous identity of an isomer, since the cis/trans label only describes a same-element pair and
    says nothing about where a vacancy sits. Order follows the polyhedron's `vertex_dirs`.
    """
    return _state_arrangement(iso, iso.centres[0])


def measured_haptic_windings(mol, metal, donors, haptic):
    """Return haptic face-orientation signs measured from a molecule's first conformer."""
    if not haptic or mol.GetNumConformers() == 0:
        return {}
    pos = mol.GetConformer().GetPositions()
    ranks = _coord_stereo.donor_classes(mol, donors)
    eta2_ranks = list(Chem.ComputeAtomCIPRanks(mol)) if any(len(face) == _PAIR for face in haptic.values()) else None
    return {
        dummy: sign
        for dummy, face in haptic.items()
        if (sign := _coord_stereo.face_winding(mol, pos, metal, face, ranks, eta2_ranks))
    }


def canonical_metals(mol, metals, *, allow_ties=False):
    """Return metal indices in graph-canonical order, optionally retaining tied identical centres."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    if not allow_ties and len({ranks[metal] for metal in metals}) != len(metals):
        raise ValueError("cannot canonicalize symmetry-equivalent metal centres; select one centre")
    return sorted(metals, key=lambda metal: (ranks[metal], metal))


def retained_state(mol, base, metal_info, metals, m):
    """Measure one retained metal state on a shared all-metal surrogate graph."""
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals]
    _idx, real_z, real_q = next(info for info in metal_info if info[0] == m)
    base, sites, haptic = _collapse_haptic(base, donors)
    apical = any(len(ring) >= _APICAL_MIN for ring in haptic.values())
    measured = None if apical else classify_geometry(base, m, sites)
    geometry = measured or geometry_for(len(sites), has_apical=apical) or f"{len(sites)}-coordinate"
    if measured is None:
        logger.info(
            "metal: no polyhedron perceived (%s) -> CN %d default %s",
            "an apical eta>=3 face fills more than one site"
            if apical
            else "no template has this vertex count / planarity",
            len(sites),
            describe(geometry),
        )
    order = _slots.input_ordering(base, m, sites, geometry)
    order = list(order) if order else list(range(len(sites)))
    vertices = [sites[k] for k in order]
    donor_bonds = [
        (neighbor.GetIdx(), metal)
        for metal in metals
        for neighbor in mol.GetAtomWithIdx(metal).GetNeighbors()
        if neighbor.GetIdx() not in metals
    ]
    roles = coordination_roles(donor_bonds, metal_info)
    chirality = _coord_stereo.chirality_of(base, geometry, vertices, haptic=haptic, coordination=roles)
    winding = measured_haptic_windings(base, m, donors, haptic)
    return base, _core.from_vertices(m, real_z, real_q, geometry, vertices, haptic, winding.items(), chirality)


def from_geometry(mol, center=None):
    """Build one selected state from a conformer-bearing Mol without enumeration.

    Realised distances and angles are retained. Donors are fitted to canonical polyhedron slots, and each
    haptic face becomes one transient centroid site. `center` selects one sphere; ``center='all'`` retains all
    spheres on the shared surrogate graph.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("from_geometry needs an input geometry (a Mol with a conformer)")
    _reject_metal_bonds(mol)
    metals = metal_indices(mol)
    ligand_stereo = _stereo.stereo_from_3d(mol, exclude=metals)
    selected = canonical_metals(mol, metals) if center == "all" else [resolve_center(mol, metals, center)]
    base, metal_info = surrogate_all_metals(mol)
    states = {}
    for m in selected:
        base, states[m] = retained_state(mol, base, metal_info, metals, m)
    donor_bonds = [
        (n.GetIdx(), m) for m in metals for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals
    ]
    real_base = strip_phantoms(base, set(range(mol.GetNumAtoms(), base.GetNumAtoms())))
    identities = selected if center == "all" else [metal for metal, _z, _q in primary_first(metal_info, selected[0])]
    info = {metal: (z, q) for metal, z, q in metal_info}
    centres = tuple(states.get(m, _core.MetalState(m, *info[m])) for m in identities)
    parts = _core.materialized_states(real_base, centres)
    retained = [
        coordination_from_geometry(
            real_base,
            state.atom,
            parts[state.atom][0],
            state.geometry,
            state.atomic_num,
            parts[state.atom][1],
        )
        for state in centres
        if state.atom in selected
    ]
    return Isomer._from_state(
        real_base,
        centres,
        donor_bonds,
        constraints=compose(*retained),
        lengths="input",
        stereo_label=ligand_stereo,
    )


def from_surrogate(mol, metals, donor_bonds):
    """Record an already-surrogated complex as an `Isomer` carrying no polyhedron.

    On the fix/constrain path the sphere is held from the input geometry, not from a named record, so
    `geometry`/`vertices`/`cons` stay empty and only the restore payload is real: element, formal charge,
    spectator metals, and the stripped M-donor bonds a consumer must re-add.
    """
    return Isomer._from_state(
        mol,
        tuple(_core.MetalState(*identity) for identity in metals),
        donor_bonds,
    )


def centre_states(iso, center=None):
    """Return an isomer's metal states, optionally narrowed to one centre."""
    states = list(iso.centres)
    if center is None:
        return states
    if not (_is_index(center) or isinstance(center, str)):
        raise TypeError(f"center must be an atom index or element symbol; got {center!r}")
    hits = [state for state in states if state.atom == center or _PT.GetElementSymbol(state.atomic_num) == center]
    if len(hits) != 1:
        have = [f"{_PT.GetElementSymbol(state.atomic_num)}{state.atom}" for state in states]
        raise ValueError(f"center={center!r} matched {len(hits)} centre(s); have {have}")
    return hits


def _state_text(iso):
    """Return one compact multi-centre state row."""
    parts = []
    for state in iso.centres:
        symbol = _PT.GetElementSymbol(state.atomic_num)
        if state.geometry not in POLYHEDRA:
            parts.append(f"{symbol}{state.atom} [surrogated]")
            continue
        code = POLYHEDRA[state.geometry].code
        coarse = _state_label(iso, state)
        label = f" {coarse}" if coarse else ""
        hand = {"delta": "Δ", "lambda": "Λ"}.get(state.hand, "-")
        haptic = _state_haptic_configuration(iso, state)
        configuration = f" {haptic}" if haptic else ""
        parts.append(f"{symbol}{state.atom} {code}{label} [{_state_arrangement(iso, state)}] {hand}{configuration}")
    return f"{' ; '.join(parts)}  {iso.stereo_label or '-'}"


def _ligand_smiles(mol, donor):
    """Return the donor-rooted heavy-atom SMILES of its ligand fragment."""
    mappings = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mappings)
    fragment, mapping = next((Chem.Mol(f), m) for f, m in zip(fragments, mappings, strict=True) if donor in m)
    marker = max((atom.GetAtomMapNum() for atom in fragment.GetAtoms()), default=0) + 1
    fragment.GetAtomWithIdx(mapping.index(donor)).SetAtomMapNum(marker)
    fragment = Chem.RemoveHs(fragment, sanitize=False)
    root = next(atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomMapNum() == marker)
    fragment.GetAtomWithIdx(root).SetAtomMapNum(0)
    return Chem.MolToSmiles(fragment, rootedAtAtom=root, canonical=True, isomericSmiles=True)


def _print_details(iso):
    """Print optional site relations and ambiguous donor identities below one state row."""
    for state in iso.centres:
        if state.geometry not in POLYHEDRA:
            continue
        vertices, haptic, winding, donors = _core.materialized_state(iso, state)
        descriptors = _coord_stereo.face_descriptors(iso._graph, donors, haptic, winding)
        metal_symbol = _PT.GetElementSymbol(state.atomic_num)
        prefix = f"       {metal_symbol}{state.atom}"
        polyhedron = POLYHEDRA[state.geometry]
        if polyhedron.site_groups:
            groups = []
            for name, positions in polyhedron.site_groups:
                labels = (_site_symbol(iso._graph, vertices[v], haptic, winding, descriptors) for v in positions)
                groups.append(f"{name}: {' '.join(labels)}")
            print(f"{prefix} {'; '.join(groups)}")
        else:
            dirs = polyhedron.vertex_dirs
            pairs = [
                f"{_site_symbol(iso._graph, vertices[a], haptic, winding, descriptors)}-"
                f"{_site_symbol(iso._graph, vertices[b], haptic, winding, descriptors)}"
                for a in range(len(dirs))
                for b in range(a + 1, len(dirs))
                if _vertex_angle(dirs[a], dirs[b]) >= _slots.TRANS_ANGLE
            ]
            if pairs:
                print(f"{prefix} trans: {', '.join(pairs)}")

        if len(haptic) > 1:
            faces = [
                f"{_site_symbol(iso._graph, dummy, haptic, winding, descriptors)}({','.join(map(str, face))})"
                for dummy, face in haptic.items()
            ]
            print(f"{prefix} faces: {', '.join(faces)}")

        ring_atoms = {atom for face in haptic.values() for atom in face}
        donors = [donor for donor in donors if donor not in ring_atoms]
        classes = _coord_stereo.donor_classes(iso._graph, donors)
        by_symbol = {}
        for donor in donors:
            by_symbol.setdefault(iso._graph.GetAtomWithIdx(donor).GetSymbol(), []).append(donor)
        ambiguous = {
            donor
            for same_element in by_symbol.values()
            if len({classes[donor] for donor in same_element}) > 1
            for donor in same_element
        }
        for donor in sorted(ambiguous):
            label = _site_symbol(iso._graph, donor, haptic, winding, descriptors)
            print(f"{prefix} {label}: {_ligand_smiles(iso._graph, donor)}")


class IsomerSet(list):
    """Store selectable coordination-isomer candidates.

    The exact identity is the canonical slot assignment plus metal, ligand and haptic stereo. Conventional
    cis/trans/mer/fac labels are intentionally coarse. Enumeration does not compile embedding constraints.
    """

    def select(
        self, geometry=None, label=None, arrangement=None, hand=None, index=None, stereo=None, haptic=None, center=None
    ):
        """Return the single `Isomer` matching the given keys.

        Key on `arrangement` (the unambiguous per-vertex slot map), metal `hand`, `geometry`, `index`, the
        ligand `stereo` tag (e.g. ``'C16:R'``, ``'16R'``, or unambiguous ``'C:R'``/``'R'``), haptic
        configuration (``'Rₚ'``/``'Sₚ'`` or ``'rac'``/``'meso'``), or the coarse `label`. Raises if zero or
        several match.
        """
        hits = self.filter(
            geometry=geometry,
            label=label,
            arrangement=arrangement,
            hand=hand,
            index=index,
            stereo=stereo,
            haptic=haptic,
            center=center,
        )
        if len(hits) != 1:
            have = [
                (
                    k,
                    f"{_PT.GetElementSymbol(state.atomic_num)}{state.atom}",
                    state.geometry,
                    state.hand or "-",
                    _state_haptic_configuration(i, state) or "-",
                    i.stereo_label or "-",
                    _state_arrangement(i, state),
                )
                for k, i in enumerate(self)
                for state in centre_states(i, center)
            ]
            raise ValueError(
                f"select(geometry={geometry!r}, label={label!r}, arrangement={arrangement!r}, "
                f"hand={hand!r}, index={index!r}, stereo={stereo!r}, haptic={haptic!r}, center={center!r}) "
                f"matched {len(hits)} isomer(s): "
                f"{'narrow it or pick by index' if hits else 'no match'}; have {have}"
            )
        return hits[0]

    def filter(
        self, geometry=None, label=None, arrangement=None, hand=None, index=None, stereo=None, haptic=None, center=None
    ):
        """Return the subset matching the given keys, as an `IsomerSet` (keep several / pick by index).

        `label`, `arrangement`, metal `hand` and `geometry` match exactly. `stereo` accepts ``C5:R`` or ``5R``; ``C:R``
        and ``R`` are available when they identify one point centre, and E/Z work the same way for one double
        bond. Ambiguous shorthand raises with the indexed choices. `index` selects positionally. `geometry`
        also takes a 3-letter code, `hand` accepts ``achiral`` and the Δ/Λ glyphs as well as the stored words, and
        `haptic` accepts ``'Rₚ'``/``'Sₚ'`` (or ``'Rp'``/``'Sp'``) for a named face and ``'rac'``/``'meso'``
        where two equivalent faces define that relative configuration.
        """
        geometry, hand = resolve_geometry(geometry), chirality_tag(hand)

        def states(i):
            return centre_states(i, center)

        def haptic_matches(i):
            if haptic is None:
                return True
            aliases = {"Rp": "Rₚ", "Sp": "Sₚ", "R_p": "Rₚ", "S_p": "Sₚ"}
            if isinstance(haptic, dict):
                for state in states(i):
                    _vertices, faces, winding, donors = _core.materialized_state(i, state)
                    descriptors = _coord_stereo.face_descriptors(i._graph, donors, faces, winding)
                    if all(
                        len(matches := [dummy for dummy, face in faces.items() if atom in face]) == 1
                        and descriptors.get(matches[0]) == aliases.get(wanted, wanted)
                        for atom, wanted in haptic.items()
                    ):
                        return True
                return False
            wanted = aliases.get(haptic, haptic)
            for state in states(i):
                _vertices, faces, winding, donors = _core.materialized_state(i, state)
                if _state_haptic_configuration(i, state) == wanted:
                    return True
                descriptors = _coord_stereo.face_descriptors(i._graph, donors, faces, winding)
                if wanted in {"Rₚ", "Sₚ"} and len(descriptors) > 1:
                    anchors = [min(faces[dummy]) for dummy in descriptors]
                    raise ValueError(
                        f"haptic={haptic!r} leaves {len(descriptors)} orientable faces unspecified; "
                        f"use haptic={{face_atom: hand}} with face atoms {anchors}, or rac/meso"
                    )
                if wanted in descriptors.values():
                    return True
            return False

        def ok(k, i):
            state_match = any(
                geometry in (None, state.geometry)
                and (label is None or _state_label(i, state) == label)
                and (arrangement is None or _state_arrangement(i, state) == arrangement)
                and (hand is None or state.hand == hand)
                for state in states(i)
            )
            return (
                state_match
                and (index is None or index == k)
                and (stereo is None or _stereo.matches_stereo(i.stereo_label, stereo))
                and haptic_matches(i)
            )

        return IsomerSet(i for k, i in enumerate(self) if ok(k, i))

    def summary(self, details=False):
        """Print every state in one compact format; optionally add site and ligand detail."""
        print("  idx  centres (geometry label [slots] Δ/Λ/-)  ligand")
        for k, iso in enumerate(self):
            print(f"  [{k:>2}] {_state_text(iso)}")
            if details:
                _print_details(iso)


def resolve_center(mol, metals, center):
    """Pick which transition metal to enumerate.

    `center` is None (the sole metal, else an error asking you to choose), an atom index, or an element
    symbol such as ``'Mn'``. The caller handles ``'all'`` before this single-centre resolver.
    """
    if center is None:
        if len(metals) == 1:
            return metals[0]
        raise ValueError(
            f"{len(metals)} transition metals present "
            f"({[mol.GetAtomWithIdx(x).GetSymbol() + str(x) for x in metals]}); choose which to "
            f"enumerate with center=<atom index or element symbol>, or use center='all'"
        )
    if _is_index(center):
        if center not in metals:
            raise ValueError(f"center={center} is not a transition-metal atom; metals are at {metals}")
        return center
    if not isinstance(center, str):
        raise TypeError(f"center must be an atom index, element symbol, or 'all'; got {center!r}")
    hits = [x for x in metals if mol.GetAtomWithIdx(x).GetSymbol() == center]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no {center!r} centre; metals present: {[mol.GetAtomWithIdx(x).GetSymbol() for x in metals]}")
    raise ValueError(f"{len(hits)} {center} centres ({hits}); disambiguate with center=<atom index>")
