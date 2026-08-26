"""The `Isomer` itself and the metal load-in that turns a `Mol` into ready-to-embed `Isomer`s.

Three doors onto one class: ``Isomer(mol, geometry, sites)`` for a known arrangement, `from_geometry` to
retain the input's own, and `enumerate_isomers` for the unknown ones as an `IsomerSet`.

This is the layer over the metal engine, composing the polytope tables, surrogate, chirality tag and
constraint builders into the distinct-arrangement enumeration a caller selects from. Parsing is the
consumer's job: this takes a `Mol`, and the input geometry's chirality fingerprint arrives as an opaque
`stereo_ref` the pipeline computes and only the pipeline reads.
"""

from __future__ import annotations

import itertools
import math
import re
from copy import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdDistGeom

from . import stereo as _stereo
from .constraints import Constraints, SphereRecipe, _is_index, compose, resolve_core
from .metal_coordination import coordination, coordination_from_geometry, resolve_lengths
from .metal_core import (
    _APICAL_MIN,
    _COLINEAR_TOL,
    _PAIR,
    _SPAN_ANGLE,
    _SPAN_TOL,
    _TRANS_ANGLE,
    _TRIAD,
    VACANT,
    _chelate_edges,
    _collapse_haptic,
    _donor_classes,
    _face_descriptors,
    _face_has_orientation,
    _face_winding,
    _frag_map,
    _reject_metal_bonds,
    _site_classes,
    _vertex_atom,
    chirality_of,
    classify_geometry,
    geometry_for,
    hold_shape,
    logger,
    metal_indices,
    n_sites,
    restore_metal,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_distance import ff_terms
from .metal_donor_orient import _FOLD_WINDOW, _stripped_hybridisation, donation_axis
from .metal_polyhedron import (
    POLYHEDRA,
    SLOT_BOND_PROP,
    _fit_trace,
    _seat_by_alignment,
    _vertex_angle,
    canonical_slots,
    chirality_tag,
    describe,
    geometries_for_cn,
    isomer_permutations,
    read_slot_notes,
    resolve_geometry,
    seat_properly,
    vertex_dirs,
)

_PT = GetPeriodicTable()


def _say_length_source(mol, lengths):
    """Report where the M-donor windows came from, when that is not the default anyone would assume."""
    note = resolve_lengths(mol, lengths)[1]
    if note:
        logger.info("metal: M-donor windows from %s", note)


def _octahedral_triad(mol, od, haptic=None):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else ``None``.

    Either a tridentate chelate (exactly 3 donors of one ligand fragment) or exactly 3 monodentate donors of
    one element (an MA3B3 set); ``None`` otherwise, and then cis/trans is used. The exactly-3 and monodentate
    conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate (en2, en3)
    have no mer/fac, so neither must be forced into a triad.
    """
    real = [(p, _vertex_atom(haptic, od[p])) for p in range(len(od)) if od[p] != VACANT]
    if len(real) < _TRIAD:
        return None
    frag = _frag_map(mol)
    by_frag = {}
    for p, d in real:
        by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():  # a tridentate chelate (one ligand, exactly 3 donors)
        if len(ps) == _TRIAD:
            return tuple(ps)
    by_elem = {}
    for p, d in real:
        if len(by_frag[frag[d]]) == 1:  # else exactly three monodentate same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == _TRIAD:
            return tuple(ps)
    return None


def _order_label(mol, donors, geometry, order, haptic=None):
    """Build the isomer label from the ideal polyhedron; no conformer needed.

    Vacant vertices are ignored. Octahedral with a donor triad is mer/fac (one trans pair in the triad means
    mer, none means fac); otherwise cis/trans, judged on the minority same-element donor pair, which is the
    set whose placement defines the isomerism: the 2 Cl of an MA4B2, not the 4 A that always have a trans
    pair.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return f"isomer{order}"
    if not POLYHEDRA[geometry].geometric_isomerism:
        return ""  # no cis/trans distinction for this geometry: a single arrangement
    od = [donors[k] for k in order]
    if geometry == "octahedral":
        tri = _octahedral_triad(mol, od, haptic)
        if tri is not None:
            trans = sum(
                1 for i in range(3) for j in range(i + 1, 3) if _vertex_angle(dirs[tri[i]], dirs[tri[j]]) > _TRANS_ANGLE
            )
            return "fac" if trans == 0 else "mer"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] != VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(_vertex_atom(haptic, od[p])).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) == _PAIR}
    if not pairs:
        return ""  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    ps = pairs[e]
    trans = any(
        _vertex_angle(dirs[ps[i]], dirs[ps[j]]) >= _TRANS_ANGLE for i in range(len(ps)) for j in range(i + 1, len(ps))
    )
    return "trans" if trans else "cis"


# --- the Isomer itself: the known-isomer front door, and the retain-the-input builder ------------------


def _seat_order(padded, haptic, sites):
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
    """Carry one composed metal state: surrogate `mol`, `Constraints`, metal identities, and donor bonds.

    `Constraints.spheres` is the source of truth for every centre's geometry, seating and handedness. The
    familiar single-centre attributes (`metal`, `geometry`, `vertices`, `chirality`, `haptic`) are derived
    from its first recipe. Build a known isomer with ``Isomer(mol, geometry, sites)``;
    `enumerate_isomers` builds unknown and multi-centre states.
    """

    mol: Chem.Mol
    cons: Constraints
    metals: tuple  # ((index, real atomic number, formal charge), ...), primary centre first
    donor_bonds: list  # stripped donor-metal bonds, re-added dative on output
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')
    stereo_label: str = ""  # ligand stereoisomer tag ('C16:R'), distinct from the metal-centre `chirality`

    def __init__(self, mol, geometry, sites, lengths="auto"):
        """Seat `sites` on `geometry`'s polyhedron: the known-isomer front door (`enumerate_isomers` is the rest).

        `geometry` is a polyhedron name or its 3-letter code (``'OCT'``). `sites` maps vertex -> donor atom
        index, as a ``{vertex: atom}`` dict or a vertex-ordered list; real atom indices, because that is what
        perception hands a consumer, and mapping them onto the internal padded-donor order happens here. The
        metal, its donors and the surrogate come from `mol`'s own bonds, dative or covalent. A geometry with
        more vertices than donors leaves the spare one a coordination pocket.

        A vertex number means whatever that polyhedron's `vertex_dirs` says, and the convention is not uniform
        across them: square-planar 0 and 1 are cis (its trans partner is 2), octahedral 0 and 1 are trans.
        Read the record in `metal_polyhedron.py`, and check a seating with ``iso.label`` / ``iso.summary()``.

        `lengths` says where the M-donor windows are measured from: ``'auto'`` / ``'input'`` / ``'model'``,
        see `metal_coordination.resolve_lengths`.
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
        order = _seat_order(padded, haptic, sites)
        vertices = [padded[k] for k in order]
        # the centroid dummy is transient embed scaffolding, so the stored mol/donors are real (see `haptic`)
        real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
        stored = strip_phantoms(base, set(haptic))
        _say_length_source(base, lengths)
        cons = coordination(base, m, padded, geom, order, real_z, haptic=haptic, lengths=lengths)
        roles = _coordination_roles([(d, m) for d in real_donors], [(m, real_z, real_q)])
        hand = chirality_of(base, donors, geom, vertices, haptic, roles)
        winding = _measured_haptic_windings(base, m, real_donors, haptic)
        cons.spheres = tuple(
            recipe._replace(chirality=hand, winding=tuple(sorted(winding.items()))) for recipe in cons.spheres
        )
        self.mol, self.cons = stored, cons
        self.metals, self.donor_bonds = ((m, real_z, real_q),), [(d, m) for d in real_donors]
        self.stereo_ref, self.stereo_label = None, ""

    @classmethod
    def _from_state(
        cls,
        mol,
        cons,
        metals,
        donor_bonds=(),
        *,
        stereo_ref=None,
        stereo_label="",
    ):
        """Build an isomer from canonical constraints and restore metadata."""
        iso = cls.__new__(cls)
        iso.mol = mol
        iso.metals, iso.donor_bonds = tuple(metals), list(donor_bonds)
        if not iso.metals:
            raise ValueError("an Isomer needs at least one metal identity")
        primary = iso.metals[0][0]
        if cons.spheres and not any(recipe.metal == primary for recipe in cons.spheres):
            raise ValueError(f"primary metal {primary} has no sphere recipe")
        spheres = tuple(recipe for recipe in cons.spheres if recipe.metal == primary) + tuple(
            recipe for recipe in cons.spheres if recipe.metal != primary
        )
        iso.cons = cons if spheres == cons.spheres else cons.copy(spheres=spheres)
        iso.stereo_ref, iso.stereo_label = stereo_ref, stereo_label
        return iso

    @property
    def metal(self):
        """Return the primary metal atom index."""
        return self.metals[0][0]

    @property
    def real_z(self):
        """Return the primary metal's real atomic number."""
        return self.metals[0][1]

    @property
    def real_q(self):
        """Return the primary metal's formal charge."""
        return self.metals[0][2]

    @property
    def spectator_metals(self):
        """Return restore identities for every non-primary metal."""
        return list(self.metals[1:])

    @property
    def _sphere(self):
        """Return the primary sphere recipe, or ``None`` for a shape-only surrogate."""
        return self.cons.spheres[0] if self.cons.spheres else None

    @property
    def geometry(self):
        """Return the primary coordination geometry."""
        return self._sphere.geometry if self._sphere else ""

    @property
    def vertices(self):
        """Return the primary sphere's donors in vertex order."""
        return [self._sphere.donors[k] for k in self._sphere.order] if self._sphere else []

    @property
    def haptic(self):
        """Return primary haptic centroid-to-face mappings."""
        return self.cons.sphere_haptic(self._sphere) if self._sphere else {}

    @property
    def haptic_winding(self):
        """Return primary canonical haptic winding signs."""
        return dict(self._sphere.winding) if self._sphere else {}

    @property
    def chirality(self):
        """Return primary metal-centre handedness."""
        return self._sphere.chirality if self._sphere else ""

    @property
    def donors(self):
        """Return the primary sphere's real donor atoms."""
        if self._sphere is None:
            return [donor for donor, metal in self.donor_bonds if metal == self.metal]
        haptic = self.haptic
        donors = [d for d in self._sphere.donors if d != VACANT and d not in haptic]
        return donors + sorted({atom for face in haptic.values() for atom in face})

    @property
    def label(self):
        """Return the primary sphere's conventional coarse isomer label."""
        if self._sphere is None:
            return ""
        return _order_label(self.mol, self._sphere.donors, self.geometry, self._sphere.order, self.haptic)

    def _with_sphere(self, metal=None, **changes):
        """Return a shallow isomer copy with one canonical sphere recipe replaced."""
        target = self.metal if metal is None else metal
        found = any(recipe.metal == target for recipe in self.cons.spheres)
        if not found:
            raise ValueError(f"metal {target} has no sphere recipe")
        spheres = tuple(
            recipe._replace(**changes) if recipe.metal == target else recipe for recipe in self.cons.spheres
        )
        out = copy(self)
        out.cons = self.cons.copy(spheres=spheres)
        return out

    def coordination(self):
        """Return the polyhedron `Constraints` holding this arrangement: what `embed` composes onto."""
        return self.cons

    def restore(self, mol=None):
        """Swap this isomer's surrogated metal(s) back to their real element and formal charge; return `mol`.

        Acts on any molecule sharing this isomer's atom indexing, defaulting to its own. The charge matters:
        restoring only Z leaves an M(0) among anionic ligands and every real energy runs at the wrong total.
        """
        mol = self.mol if mol is None else mol
        for mi, rz, rq in self.metals:
            restore_metal(mol, mi, rz, rq)
        return mol

    @property
    def haptic_configuration(self):
        """Return ``'rac'``/``'meso'`` for an interchangeable pair of named haptic faces, else ``''``."""
        descriptors = _face_descriptors(self.mol, self.donors, self.haptic, self.haptic_winding)
        windings = {d: w for d, w in self.haptic_winding.items() if len(self.haptic.get(d, ())) >= _TRIAD}
        faces = [d for d in descriptors if d in windings]
        if len(faces) != _PAIR or set(faces) != set(windings):
            return ""
        left, right = faces
        forward = _winding_signature(self, {left: "+", right: "-"})
        reverse = _winding_signature(self, {left: "-", right: "+"})
        if forward != reverse:
            return ""
        mirror = {face: "-" if winding == "+" else "+" for face, winding in self.haptic_winding.items()}
        current = _winding_signature(self, self.haptic_winding)
        return "meso" if current == _winding_signature(self, mirror) else "rac"

    def summary(self, details=False):
        """Print this isomer in the same table used by `IsomerSet.summary`."""
        return IsomerSet([self]).summary(details=details)

    def __str__(self):
        """Return the compact state row without its table index."""
        return _state_text(self)

    def __repr__(self):
        """Return a compact notebook representation."""
        return f"Isomer({str(self)!r})"


def _coordination_roles(donor_bonds, metals):
    """Return donor-to-metal identities for constitutional ranking on the stripped graph."""
    identity = {metal: (atomic_num, charge) for metal, atomic_num, charge in metals}
    return [(donor, metal, *identity[metal]) for donor, metal in donor_bonds]


def _primary_first(metals, primary):
    """Return metal identities with `primary` first and every other order retained."""
    return tuple(item for item in metals if item[0] == primary) + tuple(item for item in metals if item[0] != primary)


def _isomer_roles(iso):
    """Return every stated coordination role carried by one isomer."""
    return _coordination_roles(iso.donor_bonds, iso.metals)


def arrangement(iso):
    """Format a readable per-vertex ligand arrangement, e.g. ``'N3 Cl5 Cl6 ·'`` (``·`` = a vacant site).

    The unambiguous identity of an isomer, since the cis/trans label only describes a same-element pair and
    says nothing about where a vacancy sits. Order follows the polyhedron's `vertex_dirs`.
    """
    descriptors = _face_descriptors(iso.mol, iso.donors, iso.haptic, iso.haptic_winding)
    base_labels = [_site_symbol(iso, donor, descriptors) for donor in iso.vertices]
    labels = list(base_labels)
    classes = _site_classes(iso.mol, iso.vertices, iso.haptic, _isomer_roles(iso))
    for position, donor in enumerate(iso.vertices):
        same = [i for i, label in enumerate(base_labels) if label == base_labels[position]]
        if donor in iso.haptic and len({classes[iso.vertices[i]] for i in same}) > 1:
            anchor = min(iso.haptic[donor])
            labels[position] += f"@{iso.mol.GetAtomWithIdx(anchor).GetSymbol()}{anchor}"
    return " ".join(labels)


arrange = arrangement  # alias so IsomerSet.filter(arrangement=…) can still call the formatter (param shadows it)


def _site_symbol(iso, donor, descriptors=None):
    """Return one compact coordination-site label."""
    if donor == VACANT:
        return "·"
    if donor in iso.haptic:
        ring = iso.haptic[donor]
        descriptors = descriptors or _face_descriptors(iso.mol, iso.donors, iso.haptic, iso.haptic_winding)
        return f"η{len(ring)}{descriptors.get(donor, iso.haptic_winding.get(donor, ''))}"
    return f"{iso.mol.GetAtomWithIdx(donor).GetSymbol()}{donor}"


def _winding_signature(iso, winding):
    """Return the existing canonical-slot identity for one haptic winding assignment."""
    classes = _site_classes(iso.mol, iso.vertices, iso.haptic, _isomer_roles(iso))
    keys = [
        None if donor == VACANT else (classes[donor], winding.get(donor, "") if donor in iso.haptic else "")
        for donor in iso.vertices
    ]
    slots = canonical_slots(vertex_dirs(iso.geometry), keys, _chelate_edges(iso.mol, iso.vertices, iso.haptic))
    return tuple(key for _slot, key in sorted(zip(slots, keys, strict=True)))


def _measured_haptic_windings(mol, metal, donors, haptic):
    """Return haptic face-orientation signs measured from a molecule's first conformer."""
    if not haptic or mol.GetNumConformers() == 0:
        return {}
    pos = mol.GetConformer().GetPositions()
    ranks = _donor_classes(mol, donors)
    eta2_ranks = list(Chem.ComputeAtomCIPRanks(mol)) if any(len(face) == _PAIR for face in haptic.values()) else None
    return {
        dummy: sign
        for dummy, face in haptic.items()
        if (sign := _face_winding(mol, pos, metal, face, ranks, eta2_ranks))
    }


def _canonical_metals(mol, metals, *, allow_ties=False):
    """Return metal indices in graph-canonical order, optionally retaining tied identical centres."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    if not allow_ties and len({ranks[metal] for metal in metals}) != len(metals):
        raise ValueError("cannot canonicalize symmetry-equivalent metal centres; select one centre")
    return sorted(metals, key=lambda metal: (ranks[metal], metal))


def _retained_sphere(mol, base, metal_info, metals, m, ligand_stereo):
    """Measure one retained coordination sphere on a shared all-metal surrogate graph."""
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals]
    _idx, real_z, _real_q = next(info for info in metal_info if info[0] == m)
    base, sites, haptic = _collapse_haptic(base, donors)
    cons = coordination_from_geometry(base, m, sites, real_z, haptic)
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
    order = _input_ordering(base, m, sites, geometry)
    order = list(order) if order else list(range(len(sites)))
    vertices = [sites[k] for k in order]
    donor_bonds = [
        (neighbor.GetIdx(), metal)
        for metal in metals
        for neighbor in mol.GetAtomWithIdx(metal).GetNeighbors()
        if neighbor.GetIdx() not in metals
    ]
    roles = _coordination_roles(donor_bonds, metal_info)
    chirality = chirality_of(base, sites, geometry, vertices, haptic=haptic, coordination=roles)
    winding = _measured_haptic_windings(base, m, donors, haptic)
    if vertex_dirs(geometry) is not None:
        cons.spheres = (
            SphereRecipe(
                m,
                tuple(sites),
                geometry,
                tuple(order),
                chirality,
                tuple(sorted(winding.items())),
            ),
        )
    return base, Isomer._from_state(
        base,
        cons,
        _primary_first(metal_info, m),
        donor_bonds,
        stereo_label=ligand_stereo,
    )


def from_geometry(mol, center=None):
    """Build an `Isomer` that retains the input ligand arrangement, with no enumeration.

    Coordination constraints come from the Mol's actual conformer (which ligand sits where, at the realised
    distances and angles) and the metal is swapped to the surrogate. A haptic face (Cp, arene, η²) collapses
    to one centroid vertex by the same transient-centroid mechanism `enumerate_isomers` uses, so the retained
    arrangement matches every other metal path: a Cp is a single site, not five sigma donors. `mol` must carry
    a conformer.

    `vertices` is seated on the named polyhedron by `_input_ordering`, the same Procrustes match a ``fix=``
    uses to hold a frozen donor at its real vertex, rather than left in perception order. It has to be:
    `vertices` and the `arrangement` rendered from it are what `IsomerSet.select` keys on, so a
    perception-ordered list makes a real structure's arrangement match a different enumerated isomer. Measured
    on TransPlatin, whose as-perceived order reads identically to enumerated cis.

    `center` selects one retained sphere; ``center='all'`` composes every sphere on one shared surrogate graph.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("from_geometry needs an input geometry (a Mol with a conformer)")
    _reject_metal_bonds(mol)
    metals = metal_indices(mol)
    ligand_stereo = _stereo.stereo_from_3d(mol, exclude=metals)
    selected = _canonical_metals(mol, metals) if center == "all" else [_resolve_center(mol, metals, center)]
    base, metal_info = surrogate_all_metals(mol)
    records = []
    for m in selected:
        base, retained = _retained_sphere(mol, base, metal_info, metals, m, ligand_stereo)
        records.append(retained)
    donor_bonds = [
        (n.GetIdx(), m) for m in metals for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals
    ]
    real_base = strip_phantoms(base, {phantom for record in records for phantom in record.cons.phantoms})
    if center != "all":
        records[0].mol = real_base
        return records[0]
    if any(not record.cons.spheres for record in records):
        missing = [record.metal for record in records if not record.cons.spheres]
        raise ValueError(f"cannot retain all metal centres: no polyhedron template for metal(s) {missing}")
    primary = records[0]
    return Isomer._from_state(
        real_base,
        compose(*(record.cons for record in records)),
        _primary_first(metal_info, primary.metal),
        donor_bonds,
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
        Constraints(),
        metals,
        donor_bonds,
    )


def _sphere_views(iso, center=None):
    """Return canonical per-centre views, optionally narrowed to one metal."""
    views = [
        Isomer._from_state(
            iso.mol,
            Constraints(spheres=(recipe,), haptic=iso.cons.sphere_haptic(recipe)),
            _primary_first(iso.metals, recipe.metal),
            iso.donor_bonds,
            stereo_label=iso.stereo_label,
        )
        for recipe in iso.cons.spheres
    ]
    if center is None:
        return views
    if not (_is_index(center) or isinstance(center, str)):
        raise TypeError(f"center must be an atom index or element symbol; got {center!r}")
    hits = [sphere for sphere in views if sphere.metal == center or _PT.GetElementSymbol(sphere.real_z) == center]
    if len(hits) != 1:
        have = [f"{_PT.GetElementSymbol(s.real_z)}{s.metal}" for s in views]
        raise ValueError(f"center={center!r} matched {len(hits)} centre(s); have {have}")
    return hits


def _state_text(iso):
    """Return one compact multi-centre state row."""
    parts = []
    if not iso.cons.spheres:
        parts = [f"{_PT.GetElementSymbol(z)}{metal} [surrogated]" for metal, z, _charge in iso.metals]
        return f"{' ; '.join(parts)}  {iso.stereo_label or '-'}"
    spheres = _sphere_views(iso) if len(iso.cons.spheres) > 1 else [iso]
    for sphere in spheres:
        symbol = _PT.GetElementSymbol(sphere.real_z)
        if sphere.geometry not in POLYHEDRA:
            parts.append(f"{symbol}{sphere.metal} [surrogated]")
            continue
        code = POLYHEDRA[sphere.geometry].code
        label = f" {sphere.label}" if sphere.label else ""
        hand = {"delta": "Δ", "lambda": "Λ"}.get(sphere.chirality, "-")
        configuration = f" {sphere.haptic_configuration}" if sphere.haptic_configuration else ""
        parts.append(f"{symbol}{sphere.metal} {code}{label} [{arrange(sphere)}] {hand}{configuration}")
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
    spheres = _sphere_views(iso) if len(iso.cons.spheres) > 1 else [iso]
    for sphere in spheres:
        if sphere.geometry not in POLYHEDRA:
            continue
        symbol = _PT.GetElementSymbol(sphere.real_z)
        prefix = f"       {symbol}{sphere.metal}"
        polyhedron = POLYHEDRA[sphere.geometry]
        if polyhedron.site_groups:
            groups = [
                f"{name}: {' '.join(_site_symbol(sphere, sphere.vertices[v]) for v in vertices)}"
                for name, vertices in polyhedron.site_groups
            ]
            print(f"{prefix} {'; '.join(groups)}")
        else:
            dirs = polyhedron.vertex_dirs
            pairs = [
                f"{_site_symbol(sphere, sphere.vertices[a])}-{_site_symbol(sphere, sphere.vertices[b])}"
                for a in range(len(dirs))
                for b in range(a + 1, len(dirs))
                if _vertex_angle(dirs[a], dirs[b]) >= _TRANS_ANGLE
            ]
            if pairs:
                print(f"{prefix} trans: {', '.join(pairs)}")

        if len(sphere.haptic) > 1:
            faces = [
                f"{_site_symbol(sphere, dummy)}({','.join(map(str, face))})" for dummy, face in sphere.haptic.items()
            ]
            print(f"{prefix} faces: {', '.join(faces)}")

        ring_atoms = {atom for face in sphere.haptic.values() for atom in face}
        donors = [donor for donor in sphere.donors if donor not in ring_atoms]
        classes = _donor_classes(sphere.mol, donors)
        by_symbol = {}
        for donor in donors:
            by_symbol.setdefault(sphere.mol.GetAtomWithIdx(donor).GetSymbol(), []).append(donor)
        ambiguous = {
            donor
            for same_element in by_symbol.values()
            if len({classes[donor] for donor in same_element}) > 1
            for donor in same_element
        }
        for donor in sorted(ambiguous):
            print(f"{prefix} {_site_symbol(sphere, donor)}: {_ligand_smiles(sphere.mol, donor)}")


class IsomerSet(list):
    """The coordination isomers of a metal centre: a ``list`` of `Isomer` to iterate, index, or pick from.

    The identity is geometric, not a chemistry name: select on the per-vertex `arrangement`, the metal-centre
    `hand` (``'delta'``/``'lambda'``/``''``), the `geometry`, or the plain index. The cis/trans/mer/fac
    `label` is the conventional coarse tag; `arrangement` is the exact identity:

        isos = rx.metal('CCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-NCCC',
                         ['square_planar', 'tetrahedral']); isos.summary()
        ens  = rx.embed(isos.select(arrangement='N3 Cl6 N7 Cl5')).mc().prune()
        ens  = rx.embed(isos[0]).mc().prune()

    Enumeration is cheap; the expensive MC search runs only on the `Isomer` you pick.
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
                    f"{_PT.GetElementSymbol(sphere.real_z)}{sphere.metal}",
                    sphere.geometry,
                    sphere.chirality or "-",
                    sphere.haptic_configuration or "-",
                    i.stereo_label or "-",
                    arrange(sphere),
                )
                for k, i in enumerate(self)
                for sphere in _sphere_views(i, center)
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

        def spheres(i):
            if len(i.cons.spheres) <= 1 and center is None:
                return [i]
            return _sphere_views(i, center)

        def haptic_matches(i):
            if haptic is None:
                return True
            aliases = {"Rp": "Rₚ", "Sp": "Sₚ", "R_p": "Rₚ", "S_p": "Sₚ"}
            if isinstance(haptic, dict):
                for sphere in spheres(i):
                    descriptors = _face_descriptors(sphere.mol, sphere.donors, sphere.haptic, sphere.haptic_winding)
                    if all(
                        len(faces := [dummy for dummy, face in sphere.haptic.items() if atom in face]) == 1
                        and descriptors.get(faces[0]) == aliases.get(wanted, wanted)
                        for atom, wanted in haptic.items()
                    ):
                        return True
                return False
            wanted = aliases.get(haptic, haptic)
            for sphere in spheres(i):
                if sphere.haptic_configuration == wanted:
                    return True
                descriptors = _face_descriptors(sphere.mol, sphere.donors, sphere.haptic, sphere.haptic_winding)
                if wanted in {"Rₚ", "Sₚ"} and len(descriptors) > 1:
                    anchors = [min(sphere.haptic[dummy]) for dummy in descriptors]
                    raise ValueError(
                        f"haptic={haptic!r} leaves {len(descriptors)} orientable faces unspecified; "
                        f"use haptic={{face_atom: hand}} with face atoms {anchors}, or rac/meso"
                    )
                if wanted in descriptors.values():
                    return True
            return False

        def ok(k, i):
            sphere_match = any(
                geometry in (None, sphere.geometry)
                and (label is None or sphere.label == label)
                and (arrangement is None or arrange(sphere) == arrangement)
                and (hand is None or sphere.chirality == hand)
                for sphere in spheres(i)
            )
            return (
                sphere_match
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


def _resolve_center(mol, metals, center):
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


def _ligand_stereo_request(mol, stereo):
    """Resolve ligand stereo modes and validate atom-specific selectors."""
    has_geometry = bool(mol.GetNumConformers())
    source_default = "preserve" if has_geometry else "unassigned"
    default_mode = stereo.get("default", source_default) if isinstance(stereo, dict) else stereo
    point_mode = stereo.get("point", default_mode) if isinstance(stereo, dict) else stereo
    ez_mode = stereo.get("ez", default_mode) if isinstance(stereo, dict) else stereo
    point_mode = "preserve" if point_mode == "all" else point_mode
    ez_mode = "preserve" if ez_mode == "all" else ez_mode
    exact = {}
    if isinstance(stereo, dict):
        for selector, mode in stereo.items():
            match = re.fullmatch(r"([A-Z][a-z]?)(\d+)", str(selector))
            if match is None:
                continue
            symbol, index = match.group(1), int(match.group(2))
            if not 0 <= index < mol.GetNumAtoms() or mol.GetAtomWithIdx(index).GetSymbol() != symbol:
                raise ValueError(f"stereo selector {selector!r} is not an atom in this molecule")
            exact[index] = mode
    centres = _stereo.point_centres(mol, exclude=metal_indices(mol))
    invalid = sorted(set(exact) - centres)
    if invalid:
        raise ValueError(f"stereo atom(s) {invalid} are not configurable point stereocentres")
    broad = not isinstance(stereo, dict) or "point" in stereo or "default" in stereo
    clear = set(centres) if broad and point_mode in ("racemic", "invert") else set()
    skip = set(centres) if broad and point_mode == "free" else set()
    for index, mode in exact.items():
        if mode in ("racemic", "invert"):
            clear.add(index)
        else:
            clear.discard(index)
        if mode == "free":
            skip.add(index)
        else:
            skip.discard(index)
    return has_geometry, point_mode, ez_mode, exact, clear, skip


def _clear_ez(mol):
    """Clear native double-bond stereo and its adjacent slash bonds; return whether anything changed."""
    changed = False
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        changed |= bond.GetStereo() != Chem.BondStereo.STEREONONE
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for adjacent in atom.GetBonds():
                if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondDir() != Chem.BondDir.NONE:
                    adjacent.SetBondDir(Chem.BondDir.NONE)
                    changed = True
    return changed


def _geometry_stereo_variants(mol, variants, stereo, point_mode, ez_mode, exact):
    """Keep the requested ligand configurations relative to an input geometry."""
    measured = (
        _stereo.stereo_from_3d(mol, exclude=metal_indices(mol))
        if mol.GetNumConformers()
        else _stereo.defined_stereo_label(mol, exclude=metal_indices(mol))
    )

    def keep(label):
        for part in label.split(",") if label else ():
            point = re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|CW|CCW)", part)
            mode = exact.get(int(point.group(1)), point_mode) if point else ez_mode
            same = _stereo.matches_stereo(measured, part)
            if mode == "preserve" and not same:
                return False
            if mode == "invert" and same:
                return False
        return True

    kept = [(variant, label) for variant, label in variants if keep(label)]
    if not kept:
        raise ValueError(f"stereo={stereo!r} has no configuration compatible with the input geometry")
    return kept


def _haptic_stereo_mode(stereo, has_geometry):
    """Resolve the haptic orientation mode from one public stereo request."""
    if isinstance(stereo, dict):
        return stereo.get("planar", stereo.get("default", "preserve" if has_geometry else "racemic"))
    if stereo == "unassigned":
        return "unassigned"
    if stereo in ("racemic", "separate"):
        return "racemic"
    if stereo in ("all", "preserve", "invert"):
        return "preserve" if stereo == "all" else stereo
    return "free"


def _ligand_stereo_variants(mol, stereo):
    """Return ligand variants, their haptic mode, and enumeration counts."""
    has_geometry, point_mode, ez_mode, exact, clear, skip = _ligand_stereo_request(mol, stereo)
    source = mol
    mol = Chem.Mol(mol)
    changed = False
    for index in clear:
        atom = mol.GetAtomWithIdx(index)
        changed |= atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    skip_bonds = ez_mode == "free"
    if ez_mode in ("racemic", "invert"):
        changed |= _clear_ez(mol)
    variants, n_unassigned, _total, unresolved = _stereo.enumerate_unassigned(
        mol, exclude=set(metal_indices(mol)), skip_points=skip, skip_bonds=skip_bonds
    )
    if not n_unassigned:
        variant = mol if changed else source
        variants = [(variant, _stereo.defined_stereo_label(variant, exclude=metal_indices(variant)))]
    variants = _geometry_stereo_variants(source, variants, stereo, point_mode, ez_mode, exact)
    return variants, _haptic_stereo_mode(stereo, has_geometry), n_unassigned, unresolved


def _prepare_spectators(mol, metals, center):
    """Return one surrogated active-centre state while retaining every spectator sphere."""
    m = _resolve_center(mol, metals, center)
    spectators = [s for s in metals if s != m]
    if spectators and mol.GetNumConformers() == 0:
        raise ValueError(
            f"enumerating one centre of a {len(metals)}-metal complex needs an input "
            f"geometry (an .xyz) to retain the other metal(s); got a coordinate-free input"
        )

    metal_set = set(metals)
    donor_map = {
        metal: [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors() if n.GetIdx() not in metal_set]
        for metal in metals
    }
    donor_bonds = [(donor, metal) for metal, donors in donor_map.items() for donor in donors]
    base, metal_info = surrogate_all_metals(mol)
    base, donors, haptic = _collapse_haptic(base, donor_map[m])
    retain = Constraints()
    for s in spectators:
        hold_shape(base, [s, *donor_map[s]], retain)
        base, held = _retained_sphere(mol, base, metal_info, metals, s, "")
        phantoms = set(held.cons.phantoms)
        # The rigid shape owns real-atom distances. Keep only the shared centroid scaffold and stereo recipe;
        # composing the full coordination field would add M-L pulls that tear a shape-held spectator.
        scaffold = Constraints(
            distances={key: value for key, value in held.cons.distances.items() if phantoms & set(key)},
            angles={key: value for key, value in held.cons.angles.items() if phantoms & set(key)},
            phantoms=held.cons.phantoms,
            spheres=held.cons.spheres,
            haptic=held.cons.haptic,
        )
        retain = compose(retain, scaffold)
    identity = {metal: atomic_num for metal, atomic_num, _charge in metal_info}
    ff_terms(base, retain, {s: (identity[s], donor_map[s]) for s in spectators})
    logger.info(
        "metal: enumerating %s%d; retaining %d spectator sphere(s)",
        mol.GetAtomWithIdx(m).GetSymbol(),
        m,
        len(spectators),
    )
    state = from_surrogate(base, _primary_first(metal_info, m), donor_bonds)
    return state, donors, haptic, retain


def _select_geometries(base, m, donors, haptic, geometry, n):
    """Resolve `geometry` to the list of polyhedron names to enumerate (default from donor count, or as given)."""
    measured = None
    if geometry is None:
        # an apical (eta>=3) face fills more than one site, so a CN4 carrying one is a piano stool, not the
        # square_planar that would seat a ligand trans through the ring. An eta2 face is a single-site vertex.
        apical = any(len(r) >= _APICAL_MIN for r in haptic.values())
        if base.GetNumConformers() and not apical:  # a retained geometry names itself; an apical face is a
            sites = [haptic.get(v, v) for v in donors]  # site-count question the templates do not model
            measured = classify_geometry(base, m, sites)
        geoms = [measured or geometry_for(n, has_apical=apical)]
        if geoms == [None]:
            raise ValueError(
                f"no default geometry for {n} donors; pass geometry= a name or list "
                f"(options for {n} donors: {[p.name for p in geometries_for_cn(n)]})"
            )
        if measured is None:  # a count default, not a measurement: the user must be able to tell them apart
            logger.info(
                "metal: no geometry= and %s -> CN %d default %s",
                "an apical eta>=3 face fills more than one site"
                if apical
                else ("no input conformer to measure" if not base.GetNumConformers() else "nothing perceived"),
                n,
                describe(geoms[0]),
            )
    else:
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
        geoms = [resolve_geometry(g) for g in geoms]  # a name or its 3-letter code ('OCT'), case-insensitive
    for g in geoms:
        if g not in POLYHEDRA:
            hint = "; pass a list of names, e.g. ['square_planar', 'tetrahedral']" if g == "all" else ""
            raise ValueError(
                f"unknown geometry {g!r}; available: {sorted(k for k in POLYHEDRA if k != 'None')} "
                f"(or a code: {sorted(p.code for p in POLYHEDRA.values() if p.code)}){hint}"
            )
    if measured is not None:
        logger.info("geometry: input is %s", describe(measured))
    if geometry is not None:
        logger.debug("geometry: requested %s", ", ".join(describe(g) for g in geoms))
    return geoms


def _frozen_permutations(base, m, padded, geom, frozen_donors, sites):
    """Generate every free-donor vertex permutation with each frozen donor pinned at its input vertex.

    Returns the explicit permutation list, bypassing the symmetry-reduced canned `isomer_permutations`, which
    would miss the representative ordering a valid frozen isomer needs; or ``None`` if the input vertex
    ordering can't be read.
    """
    base_order = _input_ordering(base, m, padded, geom)
    if base_order is None:
        return None
    frozen_v = {v: di for v, di in enumerate(base_order) if padded[di] in frozen_donors}
    free_v = [v for v in range(sites) if v not in frozen_v]
    free_di = [di for di in range(len(padded)) if padded[di] not in frozen_donors]
    perms = []
    for fp in itertools.permutations(free_di):
        o = [None] * sites
        for v, di in frozen_v.items():
            o[v] = di
        for v, di in zip(free_v, fp, strict=False):
            o[v] = di
        perms.append(o)
    logger.info(
        "metal[%s]: %d frozen donors; deduplicating %d free-site arrangements",
        geom,
        len(frozen_v),
        len(perms),
    )
    return perms


def _isomers_for_geometry(state, geom, *, donors, haptic, frozen_donors, fix_cons, retain, ref_sig, lengths="auto"):
    """Enumerate every distinct `Isomer` of one polyhedron `geom` (frozen core held, spectators retained)."""
    base, m, real_z = state.mol, state.metal, state.real_z
    n = len(donors)
    sites = n_sites(geom)
    if n > sites:
        raise ValueError(f"{geom} has {sites} coordination sites but the metal has {n} donors")
    padded = list(donors) + [VACANT] * (sites - n)  # leave empty vertices as a pocket
    if sites - n:
        logger.info(
            "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)", geom, sites, n, sites - n
        )
    perms = None
    if frozen_donors and sites == n:  # pin each frozen donor at its input vertex, then generate every
        perms = _frozen_permutations(base, m, padded, geom, frozen_donors, sites)  # free-donor arrangement
    real_donors = state.donors
    roles = _isomer_roles(state)
    out = []
    for order in distinct_vertex_orderings(
        base,
        padded,
        geom,
        perms=perms,
        r_metal=_PT.GetRcovalent(real_z),
        haptic=haptic,
        coordination=roles,
    ):
        cons = coordination(
            base,
            m,
            padded,
            geom,
            order,
            real_z,
            frozen=frozen_donors,
            core_frozen=fix_cons.frozen,
            haptic=haptic,
            lengths=lengths,
        )
        # Hold any spectator metal's shape and give it the same field. Field-driven, so a floor can never
        # arrive without its `dg_floors` twin, which would leave RDKit's phantom ~3.4 Å floor standing.
        cons = compose(cons, retain, fix_cons)
        od = [padded[k] for k in order]  # vertex -> donor atom (or VACANT); a centroid dummy for an eta>=3 face
        hand = chirality_of(base, donors, geom, od, haptic, roles)
        winding = _measured_haptic_windings(base, m, real_donors, haptic)
        cons.spheres = tuple(
            recipe._replace(chirality=hand, winding=tuple(sorted(winding.items()))) if recipe.metal == m else recipe
            for recipe in cons.spheres
        )
        # The stored Isomer is real: the centroid is transient scaffolding the embed materialises from
        # `cons.haptic`, so strip it and report the real coordinating atoms. `vertices` keeps the centroid index.
        out.append(
            Isomer._from_state(
                strip_phantoms(Chem.Mol(base), cons.phantoms),
                cons,
                state.metals,
                state.donor_bonds,
                stereo_ref=ref_sig,
            )
        )
    if not out:  # every candidate ordering was rejected: silence here reads as "this geometry has no isomers"
        logger.warning(
            "metal[%s]: no arrangement survived the feasibility filters; try another geometry",
            geom,
        )
    return out


def stated_slots(mol, metal):
    """Return donor slot notes for one metal, resolving bridge lists by canonical metal order."""
    out = {}
    for atom in mol.GetAtomWithIdx(metal).GetNeighbors():
        donor = atom.GetIdx()
        bond = mol.GetBondBetweenAtoms(metal, donor)
        if bond.HasProp(SLOT_BOND_PROP):
            values = read_slot_notes(bond.GetProp(SLOT_BOND_PROP))
        elif atom.HasProp("atomNote"):
            values = read_slot_notes(atom.GetProp("atomNote"))
            centres = sorted(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in metal_indices(mol))
            if values is not None and len(values) != len(centres):
                raise ValueError(
                    f"donor atom {donor} has {len(values)} slot note(s) for {len(centres)} adjacent metal(s)"
                )
            values = None if values is None or metal not in centres else [values[centres.index(metal)]]
        else:
            continue
        if values is not None:
            out[donor] = values[0]
    return out


def stated_arrangement(mol, center=None):
    """Return the stated ``(geometry, sites, chirality, haptic winding)`` on `mol`, or ``None``.

    The reading half of `metal_smiles.cxsmiles`, and it lives here rather than there because what it
    reads is RDKit atom properties on a `Mol`: an arrangement, which is this module's subject, not a string,
    which is that one's. `metal_smiles.parse_smiles` has already kept the block's indices addressing the
    atoms they were written for, so a string that carries an arrangement arrives as an ordinary `Mol` wearing
    it and `enumerate_isomers` seats it instead of enumerating. Returns ``None`` for a plain SMILES, which is
    the signal that there is nothing to seat.

    A vacant vertex simply has no donor, so a coordination pocket survives the round trip.
    """
    # One key for both notes, so the VALUE says which it is: a slot is `s<n>` with an optional winding sign
    # (`metal_polyhedron` owns that grammar, since it owns the slots), and anything else on a noted atom is
    # the metal's geometry code.
    noted = {a.GetIdx(): a.GetProp("atomNote") for a in mol.GetAtoms() if a.HasProp("atomNote")}
    geom = [i for i in noted if i in metal_indices(mol)]
    if center is None and len(geom) != 1:
        return None
    metal = geom[0] if center is None else _resolve_center(mol, metal_indices(mol), center)
    if metal not in geom:
        return None
    geometry, separator, chirality = noted[metal].partition("-")
    try:
        name = resolve_geometry(geometry)
    except ValueError:
        return None  # atomNote is general CXSMILES metadata; an unrelated note on a metal is not our arrangement
    if separator and chirality not in {"delta", "lambda"}:
        raise ValueError(f"the arrangement on this string has unknown metal chirality {chirality!r}")
    slots = stated_slots(mol, metal)
    sites, windings = {}, {}
    for atom_idx, (slot, winding) in slots.items():
        if winding:
            if slot in windings and windings[slot] != winding:
                raise ValueError(f"the haptic face at slot s{slot} carries both winding signs")
            windings[slot] = winding
        sites.setdefault(slot, atom_idx)  # a haptic face writes one slot on every ring atom; any names it
    if name not in POLYHEDRA:
        raise ValueError(f"the arrangement on this string names {name!r}, which is not a polyhedron rxembed has")
    return name, sites, chirality, windings


def _stated_windings(iso, windings):
    """Return validated haptic winding signs read from canonical slots."""
    stated = iso.haptic_winding
    if not windings:
        return stated
    ranks = _donor_classes(iso.mol, iso.donors)
    for slot, winding in windings.items():
        donor = iso.vertices[slot]
        face = iso.haptic.get(donor)
        if face is None:
            raise ValueError(f"slot s{slot}{winding} states haptic winding, but that slot is not a haptic face")
        if not _face_has_orientation(iso.mol, face, ranks):
            raise ValueError(f"slot s{slot}{winding} states orientation on a mirror-symmetric face")
        stated[donor] = winding
    return stated


def _validate_stated_chirality(iso, chirality):
    """Reject a metal hand that contradicts the stated seating."""
    derived = iso.chirality
    if chirality and POLYHEDRA[iso.geometry].planar:
        raise ValueError(f"{iso.geometry} is planar and cannot carry metal chirality {chirality!r}")
    if not chirality and derived:
        raise ValueError(f"the stated {iso.geometry} seating is chiral ({derived}) but the metal note omits chirality")
    if chirality and derived and chirality != derived:
        raise ValueError(f"the metal note says {chirality}, but its stated seating is {derived}")
    if chirality and not derived and chirality not in _possible_stated_hands(iso):
        raise ValueError(f"the metal note says {chirality}, but its stated seating is achiral")


def _possible_stated_hands(iso):
    """Return metal hands possible before the canonical writer's tied-site pairing."""
    roles = _isomer_roles(iso)
    classes = _site_classes(iso.mol, iso.vertices, iso.haptic, roles)
    groups = {}
    for position, donor in enumerate(iso.vertices):
        if donor == VACANT:
            continue
        groups.setdefault(classes[donor], []).append(position)
    choices = [itertools.permutations(iso.vertices[p] for p in positions) for positions in groups.values()]
    hands = set()
    for assignment in itertools.product(*choices):
        vertices = list(iso.vertices)
        for positions, donors in zip(groups.values(), assignment, strict=True):
            for position, donor in zip(positions, donors, strict=True):
                vertices[position] = donor
        hands.add(chirality_of(iso.mol, iso.donors, iso.geometry, vertices, iso.haptic, roles))
    return hands


def _from_stated_arrangements(mol, center, lengths):
    """Seat every CX-noted metal sphere and return the selected centre as one composable `Isomer`."""
    metals = metal_indices(mol)
    selected = _resolve_center(mol, metals, center)
    stated = {m: stated_arrangement(mol, center=m) for m in metals}
    missing = [m for m, value in stated.items() if value is None]
    if missing:
        raise ValueError(
            f"the input states an arrangement for metal {selected}, but not for metal(s) {missing}; "
            "a coordinate-free multi-metal embed needs one geometry note per centre"
        )

    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        centres = sorted(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in metals)
        bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), m) for m in centres]
        if len(bonds) > 1 and all(bond.HasProp(SLOT_BOND_PROP) for bond in bonds):
            atom.SetProp("atomNote", ";".join(bond.GetProp(SLOT_BOND_PROP) for bond in bonds))
    donors = {m: [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals] for m in metals}
    base, metal_info = surrogate_all_metals(mol)
    info = {m: (z, q) for m, z, q in metal_info}
    donor_bonds = [(donor, metal) for metal in metals for donor in donors[metal]]
    roles = _coordination_roles(donor_bonds, metal_info)
    _say_length_source(base, lengths)
    constraints, phantoms = [], set()
    for m in metals:
        name, sites, chirality, windings = stated[m]
        base, vertices, haptic = _collapse_haptic(base, donors[m])
        count = n_sites(name)
        if len(vertices) > count:
            raise ValueError(f"{name} has {count} sites but metal {m} has {len(vertices)} donor sites")
        padded = list(vertices) + [VACANT] * (count - len(vertices))
        order = _seat_order(padded, haptic, sites)
        seated = [padded[k] for k in order]
        cons = coordination(base, m, padded, name, order, info[m][0], haptic=haptic, lengths=lengths)
        derived = chirality_of(base, vertices, name, seated, haptic, roles)
        cons.spheres = tuple(recipe._replace(chirality=derived) for recipe in cons.spheres)
        iso = Isomer._from_state(base, cons, _primary_first(metal_info, m), donor_bonds)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        cons.spheres = tuple(
            recipe._replace(chirality=chirality, winding=tuple(sorted(winding.items()))) for recipe in cons.spheres
        )
        constraints.append(cons)
        phantoms.update(haptic)

    return Isomer._from_state(
        strip_phantoms(base, phantoms),
        compose(*constraints),
        _primary_first(metal_info, selected),
        donor_bonds,
    )


def _enumerate_haptic_windings(isomers):
    """Expand undefined haptic face orientations and drop symmetry-equivalent sign assignments."""
    out = IsomerSet()
    for iso in isomers:
        ranks = _donor_classes(iso.mol, iso.donors)
        faces = [
            dummy
            for dummy, face in iso.haptic.items()
            if dummy not in iso.haptic_winding and _face_has_orientation(iso.mol, face, ranks)
        ]
        if not faces:
            out.append(iso)
            continue
        seen = set()
        for signs in itertools.product("+-", repeat=len(faces)):
            winding = iso.haptic_winding | dict(zip(faces, signs, strict=True))
            signature = _winding_signature(iso, winding)
            if signature in seen:
                continue
            seen.add(signature)
            out.append(iso._with_sphere(winding=tuple(sorted(winding.items()))))
    return out


def _haptic_mode(isomers, mode):
    """Apply the requested mode to haptic orientation read from an input geometry."""
    if mode in ("free", "preserve"):
        return isomers
    out = IsomerSet()
    for iso in isomers:
        spheres = _sphere_views(iso) if len(iso.cons.spheres) > 1 else [iso]
        choices = []
        for sphere in spheres:
            if mode == "invert":
                winding = {face: "+" if sign == "-" else "-" for face, sign in sphere.haptic_winding.items()}
                candidate = sphere._with_sphere(winding=tuple(sorted(winding.items())))
                choices.append(_enumerate_haptic_windings([candidate]))
            elif mode == "unassigned":
                choices.append(_enumerate_haptic_windings([sphere]))
            else:  # racemic / separate: discard the measured hand, then reuse the graph enumerator
                candidate = sphere._with_sphere(winding=())
                choices.append(_enumerate_haptic_windings([candidate]))
        for selected in itertools.product(*choices):
            windings = {sphere.metal: sphere.haptic_winding for sphere in selected}
            candidate = copy(iso)
            candidate.cons = iso.cons.copy(
                spheres=tuple(
                    recipe._replace(winding=tuple(sorted(windings[recipe.metal].items())))
                    for recipe in iso.cons.spheres
                )
            )
            out.append(candidate)
    return out


def _enumerate_all_centers(mol, geometry, fix, haptic_mode, stereo_ref, lengths):
    """Stack independently enumerated coordination spheres into their Cartesian product."""
    metals = metal_indices(mol)
    if len(metals) < _PAIR:
        raise ValueError("center='all' needs at least two transition-metal centres")
    stated = stated_arrangement(mol, center=metals[0])
    metals = _canonical_metals(mol, metals, allow_ties=stated is not None)
    if stated is not None:
        if geometry is not None or fix:
            raise ValueError(
                f"this input already states every metal arrangement, so "
                f"{'fix=' if fix else f'geometry={geometry!r}'} has nothing to act on"
            )
        return _haptic_mode(IsomerSet([_from_stated_arrangements(mol, metals[0], lengths)]), haptic_mode)
    if mol.GetNumConformers() == 0:
        raise ValueError("center='all' needs an input geometry to infer one polyhedron per metal centre")
    if geometry is not None:
        raise ValueError("center='all' infers each centre's geometry; omit the single geometry= argument")

    base, metal_info = surrogate_all_metals(mol)
    sphere, donor_bonds, phantoms = {}, [], set()
    metal_set = set(metals)
    for m in metals:
        real = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metal_set]
        donor_bonds.extend((d, m) for d in real)
        base, donors, haptic = _collapse_haptic(base, real)
        sphere[m] = (donors, haptic)
        phantoms.update(haptic)
    real_base = strip_phantoms(Chem.Mol(base), phantoms)
    _say_length_source(base, lengths)

    fix_cons = Constraints()
    if fix:
        fix_cons, _ = resolve_core(base, fix=fix, has_geometry=True)
    frozen_core = Constraints(frozen=set(fix_cons.frozen))
    choices = []
    for m in metals:
        donors, haptic = sphere[m]
        state = from_surrogate(base, _primary_first(metal_info, m), donor_bonds)
        candidates = IsomerSet()
        for geom in _select_geometries(base, m, donors, haptic, None, len(donors)):
            candidates.extend(
                _isomers_for_geometry(
                    state,
                    geom,
                    donors=donors,
                    haptic=haptic,
                    frozen_donors=fix_cons.frozen & set(donors),
                    fix_cons=frozen_core,
                    retain=Constraints(),
                    ref_sig=stereo_ref,
                    lengths=lengths,
                )
            )
        for candidate in candidates:
            candidate.mol = real_base
        choices.append(_haptic_mode(candidates, haptic_mode))

    out = IsomerSet()
    for selected in itertools.product(*choices):
        primary = selected[0]
        out.append(
            Isomer._from_state(
                real_base,
                compose(*(candidate.cons for candidate in selected), fix_cons),
                _primary_first(metal_info, primary.metal),
                donor_bonds,
                stereo_ref=stereo_ref,
            )
        )
    return out


def _source_defaults(mol, center, stereo):
    """Choose source-aware centre and stereo defaults."""
    if stereo is None:
        stereo = "all" if mol.GetNumConformers() else "unassigned"
    metals = metal_indices(mol)
    if center is None and len(metals) > 1:
        stated = all(stated_arrangement(mol, center=metal) is not None for metal in metals)
        if mol.GetNumConformers() or stated:
            center = "all"
    return center, stereo


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo=None, stereo_ref=None, lengths="auto"):
    """Enumerate distinct coordination and ligand stereoisomers of an RDKit Mol.

    `geometry` accepts a registry name or 3-letter code. `center=` selects one metal while retaining the
    others; ``center='all'`` composes every centre's independently enumerated constraints. A geometry input
    retains measured ligand and haptic stereo by default, while a graph enumerates undefined elements.
    `lengths` selects input or model M-donor distances (`'auto'`, `'input'`, or `'model'`).
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError(
            f"enumerate_isomers() takes an RDKit Mol, got {type(mol).__name__}. Parse SMILES with "
            "rxembed.parse_smiles, or call rxembed.metal for a SMILES or .xyz source"
        )
    _reject_metal_bonds(mol)
    center, stereo = _source_defaults(mol, center, stereo)
    variants, haptic_mode, n_unassigned, unresolved = _ligand_stereo_variants(mol, stereo)
    out = IsomerSet()
    for variant, stated_label in variants:
        built = _enumerate_coordination(variant, geometry, center, fix, haptic_mode, stereo_ref, lengths)
        label = _stereo.defined_stereo_label(variant, exclude=metal_indices(variant)) or stated_label
        point = _stereo.point_stereo(stated_label)
        tag_source = None
        if point:
            tag_source = Chem.Mol(variant)
            tag_source.RemoveAllConformers()
            tag_source, _metal_info = surrogate_all_metals(tag_source)
        for iso in built:
            if tag_source is not None:
                iso.mol = Chem.Mol(iso.mol)
                for donor in point:
                    iso.mol.GetAtomWithIdx(donor).SetChiralTag(tag_source.GetAtomWithIdx(donor).GetChiralTag())
            iso.stereo_label = label
            out.append(iso)
    if n_unassigned:
        logger.info(
            "metal: %d ligand stereo element(s) -> coordination x %d stereoisomer(s) = %d candidate(s)",
            n_unassigned,
            len(variants),
            len(out),
        )
    if unresolved:
        logger.warning(
            "metal: %d stereo axis(es) not enumerable from a flat SMILES; embedded as one arbitrary hand",
            unresolved,
        )
    return out


def _enumerate_coordination(mol, geometry, center, fix, haptic_mode, stereo_ref, lengths):
    """Enumerate coordination arrangements for one ligand stereoisomer."""
    if center == "all":
        return _enumerate_all_centers(mol, geometry, fix, haptic_mode, stereo_ref, lengths)
    stated = stated_arrangement(mol, center=center)
    if stated is not None:
        name, sites, chirality, windings = stated
        if (geometry is not None and resolve_geometry(geometry) != name) or fix:
            raise ValueError(
                f"this input already states a {name} arrangement, so "
                f"{'fix=' if fix else f'geometry={geometry!r}'} has nothing to act on; drop it to use what "
                f"the input says, or strip the arrangement to enumerate"
            )
        logger.info("using stated %s arrangement", describe(name))
        if len(metal_indices(mol)) > 1:
            return _haptic_mode(IsomerSet([_from_stated_arrangements(mol, center, lengths)]), haptic_mode)
        iso = Isomer(mol, name, sites, lengths=lengths)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        iso = iso._with_sphere(chirality=chirality, winding=tuple(sorted(winding.items())))
        return _haptic_mode(IsomerSet([iso]), haptic_mode)
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no transition metal found")
    haptic = {}  # centroid dummies, keyed dummy -> its face atoms; set by _collapse_haptic on the single path
    # One metal means no spectators, so this is the single-centre path whether or not center= names it. An
    # explicit center= is still validated against the sole metal, raising on a wrong index or element.
    if len(metals) == 1:
        if center is not None:
            _resolve_center(mol, metals, center)
        base, m, real_donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = _collapse_haptic(base, real_donors)
        state = from_surrogate(base, ((m, real_z, real_q),), [(donor, m) for donor in real_donors])
        retain = Constraints()
    else:
        state, donors, haptic, retain = _prepare_spectators(mol, metals, center)
        base, m = state.mol, state.metal
    _say_length_source(base, lengths)  # once per molecule, not per ordering
    fix_cons = Constraints()
    frozen_donors = set()
    # Hold a reacting TS core while the rest of the coordination sphere is enumerated: coordinate forms need
    # an input geometry, while numeric distances and angles are complete on a coordinate-free graph.
    if fix:
        fix_cons, _ = resolve_core(base, fix=fix, has_geometry=base.GetNumConformers() > 0)
        frozen_donors = fix_cons.frozen & set(donors)
        logger.info(
            "metal: fixed %d input atoms; enumerating free coordination sites",
            len(fix_cons.constrained_atoms()),
        )
    geoms = _select_geometries(base, m, donors, haptic, geometry, len(donors))
    out = IsomerSet()
    for geom in geoms:
        out.extend(
            _isomers_for_geometry(
                state,
                geom,
                donors=donors,
                haptic=haptic,
                frozen_donors=frozen_donors,
                fix_cons=fix_cons,
                retain=retain,
                ref_sig=stereo_ref,
                lengths=lengths,
            )
        )
    if mol.GetNumConformers():
        out = _haptic_mode(out, haptic_mode)
    elif haptic_mode != "free":
        out = _enumerate_haptic_windings(out)
    return out


def _input_ordering(mol, metal, donors, geometry):
    """Find the vertex ordering that best matches the input geometry: which donor sits at which vertex.

    ``od[vertex] = donors[order[vertex]]``, read from the conformer by orthogonal Procrustes over the
    candidate vertex orderings. This lets a ``fix=`` hold each frozen donor at its real vertex so only the
    free sites are enumerated; otherwise the enumeration permutes a frozen donor into a vertex it cannot
    occupy, producing isomers that contradict the frozen core, such as a hydride forced off its TS site.

    The fit deliberately allows reflection, so a mirror pair scores identically; `seat_properly` turns the
    winner into the proper seating, without which `chirality_of` hands both input geometries the same tag.
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([np.zeros(3) if d == VACANT else pos[d] - pos[metal] for d in donors], float)
    dd /= np.where((norm := np.linalg.norm(dd, axis=1, keepdims=True)) > 0, norm, 1.0)
    v_ideal = np.array(dirs_ref, float)
    canned = isomer_permutations(geometry)
    if canned is None:  # no canned list (CN7/8): searching only the identity would seat donors in PERCEPTION
        return seat_properly(dd, dirs_ref, _seat_by_alignment(dd, v_ideal))  # order, not a seating at all
    best_score, best_order = -1.0, list(range(len(donors)))
    for order in canned:
        score = _fit_trace(dd[list(order)].T @ v_ideal)  # best alignment; see `_fit_trace` on reflections
        if score > best_score:
            best_score, best_order = score, order
    return seat_properly(dd, dirs_ref, best_order)


def _central_trans(od, frag, dmat, dirs, haptic=None):
    """Return True if a tridentate chelate's central donor is placed trans to one of its own arms.

    The central donor is the one on the backbone path between the other two (``d(a,c)+d(c,b)==d(a,b)``), which
    a pincer cannot do: central is cis to both arms in every real mer/fac. A flexible chelate can stretch to
    ~155° without a formally torn bond, so this drops it at enumeration rather than leaving it to `bonding_ok`.
    A haptic vertex is resolved to its ring atom so its backbone path is real.
    """
    ra = [_vertex_atom(haptic, d) if d != VACANT else VACANT for d in od]  # vertex -> representative real atom
    by_frag = {}
    for p, d in enumerate(ra):
        if d != VACANT:
            by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():
        if len(ps) != _TRIAD:
            continue
        for ci in range(3):
            c, a, b = ps[ci], ps[(ci + 1) % 3], ps[(ci + 2) % 3]
            if abs(dmat[ra[a]][ra[c]] + dmat[ra[c]][ra[b]] - dmat[ra[a]][ra[b]]) < _COLINEAR_TOL:  # c is central
                if _vertex_angle(dirs[c], dirs[a]) > _TRANS_ANGLE or _vertex_angle(dirs[c], dirs[b]) > _TRANS_ANGLE:
                    return True
                break
    return False


def _span_bounds(mol):
    """Return the ligands' own bounds matrix, or None if it cannot be built, in which case never filter.

    Bounds come from the ligand's own connectivity, i.e. how far this backbone can actually reach: the metal
    is bond-less here, so every path runs through the backbone rather than across the centre. Read as
    ``bm[i][j]`` for a pair's upper bound and ``bm[j][i]`` for its lower (i < j).
    """
    try:
        return rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except Exception:  # pragma: no cover; the bounds matrix is robust, but never let the gate crash here
        return None


def _reach(bm, i, j):
    """Return the pair's upper-bound (max reachable) distance: the bounds matrix's i<j triangle."""
    return float(bm[min(i, j)][max(i, j)])


def _donor_faces_metal(mol, d, *, other, d_md, d_mo, need, bm, hyb, donors):
    """Return False if donor ``d``, stretched to span trans, cannot still point its lone pair at the metal.

    The reach test asks only whether the donors can get ``need`` apart; this asks whether the backbone that
    achieves it can still donate. At a trans span the metal lies on the D···D' line, so the fold ruler's M-D-X
    angle is fixed by the ligand's own X-D···D' angle, and the anti backbone that reaches furthest points D's
    substituent straight at the metal. Taking the census fold floor as the minimum M-D-X, each heavy X is
    forced out to a matching X···D' distance; if the backbone can't reach that, no conformer spans and donates.

    Abstains via `donation_axis` (a hydride, bridging or haptic donor has no axis) and for an uncalibrated
    (element, hyb) class, reading the ruler's own rules rather than a copy of them.
    """
    subs = donation_axis(mol, d, donors)
    if subs is None:  # hydride / bridging / haptic: no donation axis, so "does it face the metal" is meaningless
        return True
    cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None
    if cls not in _FOLD_WINDOW:  # estimators disagreed, or n < 6 for the class: the ruler abstains, so do we
        return True
    beta = math.degrees(  # angle(M, D, D') in the M-D-D' triangle: how far off the D···D' line the metal sits
        math.acos(max(-1.0, min(1.0, (d_md**2 + need**2 - d_mo**2) / (2 * d_md * need))))
    )
    alpha = _FOLD_WINDOW[cls][0] - beta  # the X-D···D' angle the fold floor forces (M is `beta` off that line)
    if alpha <= 0:  # the metal already sits far enough off the line, so the floor costs the backbone nothing
        return True
    for x in subs:
        r = float(bm[max(x, d)][min(x, d)])  # the D-X bond length (a bond's bounds coincide to < 0.05 Å)
        out = math.sqrt(r**2 + need**2 - 2 * r * need * math.cos(math.radians(alpha)))  # X···D' this forces
        if _reach(bm, x, other) < out - _SPAN_TOL:  # the backbone can't hold X that far off the metal
            return False
    return True


def _chelate_span_ok(mol, od, *, frag, dirs, bm, r_metal, hyb, donors, haptic=None):
    """Return False if a chelate is placed trans across a metal its backbone can't reach, or can't donate to.

    A cis bidentate always folds in, so only a wide separation (`_SPAN_ANGLE` or more, i.e. trans) is tested.
    Both questions read the ligand's own bounds matrix, so a genuine long-bridge ligand that can span trans is
    allowed: this is geometry, not a topological guess. Generalises `_central_trans` to any denticity.

    Reach: the donors would sit on opposite sides, needing a donor-donor distance
    ``law_of_cosines(d_Ma, d_Mb, θ)`` a short backbone cannot span.

    Orientation (`_donor_faces_metal`): reach models where the donors are, not where they point. An extended
    anti backbone aims both lone pairs along the chain; without this a diphosphine passed reach and relaxed to
    P-M-P 155° against its 74-104° bite.
    """
    if bm is None:  # no bounds matrix -> never filter; defer to `bonding_ok` downstream
        return True
    for p in range(len(od)):
        for q in range(p + 1, len(od)):
            a, b = od[p], od[q]
            # A haptic centroid is bond-less, so resolve each vertex to a representative ring atom and let the
            # same-ligand test, the covalent reach and the backbone bounds all read the face's real chemistry.
            # Otherwise a face tethered to a co-donor reads as a separate ligand and its trans is never dropped.
            ra, rb = _vertex_atom(haptic, a), _vertex_atom(haptic, b)
            if VACANT in (a, b) or frag[ra] != frag[rb]:  # only a same-ligand (chelate) pair
                continue
            theta = _vertex_angle(dirs[p], dirs[q])
            if theta < _SPAN_ANGLE:  # cis / adjacent -> the chelate folds in, always feasible
                continue
            d_ma = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(ra).GetAtomicNum())  # real M-donor covalent sums,
            d_mb = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(rb).GetAtomicNum())  # not a fixed 2.0 (Pd-N ~2.1)
            need = math.sqrt(d_ma**2 + d_mb**2 - 2 * d_ma * d_mb * math.cos(math.radians(theta)))  # law of cosines
            if _reach(bm, ra, rb) < need - _SPAN_TOL:  # backbone can't reach
                return False
            # The orientation test asks whether the donor can still aim its lone pair at the metal, which is
            # meaningless for a haptic face: it donates a π face and has no axis, so a centroid abstains.
            if a not in (haptic or {}) and not _donor_faces_metal(
                mol, ra, other=rb, d_md=d_ma, d_mo=d_mb, need=need, bm=bm, hyb=hyb, donors=donors
            ):
                return False
            if b not in (haptic or {}) and not _donor_faces_metal(
                mol, rb, other=ra, d_md=d_mb, d_mo=d_ma, need=need, bm=bm, hyb=hyb, donors=donors
            ):
                return False
    return True


def _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, coordination=(), *, limit=None):
    """Dedup vertex orderings by their connectivity-aware signature; stop once `limit` distinct ones are found.

    Two geometric impossibilities are pruned: a tridentate's central donor trans to its own arm
    (`_central_trans`), and a chelate placed trans whose backbone can't reach or, stretched, can't donate
    inward (`_chelate_span_ok`). A cis chelate is always kept; any residual is dropped by `bonding_ok`.

    The dedup signature carries, per donor pair, the donor symmetry classes, the vertex angle and a
    same-ligand pair's intra-ligand bond distance, plus the centre's handedness, so enantiomers, asymmetric
    ligand ends and a tridentate's mer vs central-trans stay distinct. `limit` short-circuits at the k-th
    distinct arrangement, which is all the warning caller needs.
    """
    frag = _frag_map(mol)  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances
    real_donors = [d for d in donors if d != VACANT]
    classes = _site_classes(mol, donors, haptic, coordination)
    bm = _span_bounds(mol)
    hyb = _stripped_hybridisation(mol)  # the fold ruler's own (element, hyb) class: graph-only, no coords
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))]
    angle = {(p, q): _vertex_angle(dirs[p], dirs[q]) for p, q in pairs}  # a vertex-pair's angle is donor-independent

    def link(od, p, q):  # intra-ligand bond distance of a same-ligand pair
        a, b = _vertex_atom(haptic, od[p]), _vertex_atom(haptic, od[q])  # resolve a centroid to its ring atom
        if VACANT in (od[p], od[q]) or frag[a] != frag[b]:  # distinguishes a chelate's central from its
            return -1  # terminal donor; -1 for different ligands or a vacancy
        return int(dmat[a][b])

    def donor_class(d):
        return ("vacant",) if d == VACANT else ("donor", classes[d])

    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        if _central_trans(od, frag, dmat, dirs, haptic):  # a tridentate's central donor trans to its own arm
            continue
        if not _chelate_span_ok(
            mol, od, frag=frag, dirs=dirs, bm=bm, r_metal=r_metal, hyb=hyb, donors=real_donors, haptic=haptic
        ):  # can't span/donate trans
            continue
        sig = tuple(
            sorted(
                (tuple(sorted((donor_class(od[p]), donor_class(od[q])))), link(od, p, q), angle[(p, q)])
                for p, q in pairs
            )
        )
        sig = (sig, chirality_of(mol, real_donors, geometry, od, haptic, coordination))
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if limit is not None and len(out) >= limit:
                break
    return out


def distinct_vertex_orderings(mol, donors, geometry, perms=None, r_metal=1.4, haptic=None, coordination=()):
    """Enumerate distinct coordination isomers: every distinct vertex arrangement, minimally pre-filtered.

    Dedup and the two feasibility pre-filters live in `_distinct_orderings`. `perms` overrides the candidate
    vertex orderings (default ``isomer_permutations(geometry)``); a ``fix=`` enumeration passes the subset that
    keeps each frozen donor pinned to its input vertex.
    """
    dirs = vertex_dirs(geometry)
    if perms is None and isomer_permutations(geometry) is None:  # CN7/8, tetrahedral and linear have no
        # canned list, so the input ordering is the only candidate. Warn only where that loses something: a
        # linear or all-identical geometry has exactly one arrangement and the warning would be noise.
        orbit = itertools.permutations(range(len(donors)))
        if (
            dirs is not None
            and len(_distinct_orderings(mol, donors, geometry, orbit, dirs, r_metal, haptic, coordination, limit=2)) > 1
        ):
            logger.info(
                "metal[%s]: no isomer permutations tabulated; enumerating the input ordering only",
                describe(geometry),
            )
        # Unfiltered: with a single ordering there is nothing to prefer it over, so filtering could only
        # return empty and make `rx.metal(...)[0]` raise. `bonding_ok` and the geometry gate judge it.
        return [list(range(len(donors)))]
    perms = perms if perms is not None else isomer_permutations(geometry)
    if dirs is None:
        return perms
    return _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, coordination)
