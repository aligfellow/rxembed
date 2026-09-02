"""Enumerate ligand, haptic and coordination states into ready-to-embed `Isomer` candidates."""

from __future__ import annotations

import itertools
import re

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from . import metal_core as _core
from . import metal_isomer as _isomer
from . import metal_slots as _slots
from . import metal_stereo as _coord_stereo
from . import stereo as _stereo
from .constraints import Constraints, compose, resolve_core
from .metal_core import (
    _APICAL_MIN,
    VACANT,
    _collapse_haptic,
    _reject_metal_bonds,
    classify_geometry,
    geometry_for,
    hold_shape,
    logger,
    metal_indices,
    n_sites,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_distance import ff_terms
from .metal_polyhedron import (
    POLYHEDRA,
    SLOT_BOND_PROP,
    describe,
    geometries_for_cn,
    read_slot_notes,
    resolve_geometry,
)

_PT = GetPeriodicTable()
_MULTI_METAL = 2


def _ligand_stereo_request(mol, stereo):
    """Resolve ligand stereo modes and validate atom-specific selectors."""
    has_geometry = bool(mol.GetNumConformers())
    source_default = "preserve" if has_geometry else "unassigned"
    default_mode = stereo.get("default", source_default) if isinstance(stereo, dict) else stereo
    point_mode = stereo.get("point", default_mode) if isinstance(stereo, dict) else stereo
    ez_mode = stereo.get("ez", default_mode) if isinstance(stereo, dict) else stereo
    axial_mode = stereo.get("axial", default_mode) if isinstance(stereo, dict) else stereo
    point_mode = "preserve" if point_mode == "all" else point_mode
    ez_mode = "preserve" if ez_mode == "all" else ez_mode
    axial_mode = "preserve" if axial_mode == "all" else axial_mode
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
    return has_geometry, point_mode, ez_mode, axial_mode, exact, clear, skip


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


def _geometry_stereo_variants(mol, variants, stereo, point_mode, ez_mode, axial_mode, exact):
    """Keep the requested ligand configurations relative to an input geometry."""
    measured = (
        _stereo.stereo_from_3d(mol, exclude=metal_indices(mol))
        if mol.GetNumConformers()
        else _stereo.defined_stereo_label(mol, exclude=metal_indices(mol))
    )
    measured_axes = _stereo.axis_stereo(measured)

    def keep(label):
        for part in label.split(",") if label else ():
            point = re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|CW|CCW)", part)
            axis = re.fullmatch(r"[A-Z][a-z]?(\d+)-[A-Z][a-z]?(\d+):(M|P)", part)
            mode = exact.get(int(point.group(1)), point_mode) if point else axial_mode if axis else ez_mode
            if axis and tuple(sorted(map(int, axis.group(1, 2)))) not in measured_axes:
                continue  # an inferred axis has no reference hand to preserve or invert
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
    has_geometry, point_mode, ez_mode, axial_mode, exact, clear, skip = _ligand_stereo_request(mol, stereo)
    source = mol
    mol = Chem.Mol(mol)
    changed = False
    if has_geometry:
        axes = _stereo.axis_stereo(_stereo.stereo_from_3d(mol, exclude=metal_indices(mol)))
        _stereo._assign_atrop_from_3d(mol, axes)
        changed |= bool(axes)
    for index in clear:
        atom = mol.GetAtomWithIdx(index)
        changed |= atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    skip_bonds = ez_mode == "free"
    if ez_mode in ("racemic", "invert"):
        changed |= _clear_ez(mol)
    include_atrop = _stereo._clear_atrop(mol) if axial_mode in ("racemic", "invert") else set()
    changed |= bool(include_atrop)
    variants, n_unassigned, _total, unresolved = _stereo.enumerate_unassigned(
        mol,
        exclude=set(metal_indices(mol)),
        skip_points=skip,
        skip_bonds=skip_bonds,
        skip_atrop=axial_mode == "free",
        include_atrop=include_atrop,
    )
    if not n_unassigned:
        variant = mol if changed else source
        variants = [(variant, _stereo.defined_stereo_label(variant, exclude=metal_indices(variant)))]
    variants = _geometry_stereo_variants(source, variants, stereo, point_mode, ez_mode, axial_mode, exact)
    return variants, _haptic_stereo_mode(stereo, has_geometry), n_unassigned, unresolved


def _prepare_spectators(mol, metals, center):
    """Return one surrogated base isomer while retaining every spectator sphere."""
    m = _isomer.resolve_center(mol, metals, center)
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
    retained_states = {}
    spectator_phantoms = set()
    for s in spectators:
        hold_shape(base, [s, *donor_map[s]], retain)
        before = base.GetNumAtoms()
        base, retained_states[s] = _isomer.retained_state(mol, base, metal_info, metals, s)
        spectator_phantoms.update(range(before, base.GetNumAtoms()))
    identity = {metal: atomic_num for metal, atomic_num, _charge in metal_info}
    real_base = strip_phantoms(Chem.Mol(base), {*haptic, *spectator_phantoms})
    ff_terms(real_base, retain, {s: (identity[s], donor_map[s]) for s in spectators})
    base = strip_phantoms(base, spectator_phantoms)  # keep only the active centre's enumeration centroids
    logger.info(
        "metal: enumerating %s%d; retaining %d spectator sphere(s)",
        mol.GetAtomWithIdx(m).GetSymbol(),
        m,
        len(spectators),
    )
    centres = tuple(
        retained_states.get(metal, _core.MetalState(metal, atomic_num, charge))
        for metal, atomic_num, charge in _isomer.primary_first(metal_info, m)
    )
    base_iso = _isomer.Isomer._from_state(base, centres, donor_bonds)
    return base_iso, donors, haptic, retain


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

    Returns the explicit permutation list, bypassing the symmetry-reduced `isomer_permutations`, which
    would miss the representative ordering a valid frozen isomer needs; or ``None`` if the input vertex
    ordering can't be read.
    """
    base_order = _slots.input_ordering(base, m, padded, geom)
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


def _isomers_for_geometry(
    base_iso,
    geom,
    *,
    donors,
    haptic,
    frozen_donors,
    fix_cons,
    retain,
    ref_sig,
    lengths,
    source,
    graft_ref=None,
):
    """Enumerate every distinct `Isomer` of one polyhedron `geom` (frozen core held, spectators retained)."""
    base, m, real_z = base_iso._graph, base_iso.metal, base_iso.real_z
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
    real_donors = base_iso.donors
    roles = _isomer.isomer_roles(base_iso)
    base_cons = compose(retain, fix_cons)
    out = []
    for order in _slots.distinct_vertex_orderings(
        base,
        padded,
        geom,
        perms=perms,
        r_metal=_PT.GetRcovalent(real_z),
        haptic=haptic,
        coordination=roles,
    ):
        od = [padded[k] for k in order]  # vertex -> donor atom, haptic centroid, or VACANT
        hand = _coord_stereo.chirality_of(base, geom, od, haptic, roles)
        winding = _isomer.measured_haptic_windings(base, m, real_donors, haptic)
        active = _core.from_vertices(m, base_iso.real_z, base_iso.real_q, geom, od, haptic, winding.items(), hand)
        centres = (active, *(centre for centre in base_iso.centres if centre.atom != m))
        out.append(
            _isomer.Isomer._from_state(
                source,
                centres,
                base_iso.donor_bonds,
                constraints=base_cons,
                constrained_metals={m},
                lengths=lengths,
                graft_ref=graft_ref,
                stereo_ref=ref_sig,
                shared_mol=True,
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
    metal = geom[0] if center is None else _isomer.resolve_center(mol, metal_indices(mol), center)
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
    ranks = _coord_stereo.donor_classes(iso._graph, iso.donors)
    for slot, winding in windings.items():
        donor = iso.vertices[slot]
        face = iso.haptic.get(donor)
        if face is None:
            raise ValueError(f"slot s{slot}{winding} states haptic winding, but that slot is not a haptic face")
        if not _coord_stereo.face_has_orientation(iso._graph, face, ranks):
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
    roles = _isomer.isomer_roles(iso)
    classes = _coord_stereo.site_classes(iso._graph, iso.vertices, iso.haptic, roles)
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
        hands.add(_coord_stereo.chirality_of(iso._graph, iso.geometry, vertices, iso.haptic, roles))
    return hands


def _from_stated_arrangements(mol, center, lengths):
    """Seat every CX-noted metal sphere and return the selected centre as one composable `Isomer`."""
    metals = metal_indices(mol)
    selected = _isomer.resolve_center(mol, metals, center)
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
    roles = _isomer.coordination_roles(donor_bonds, metal_info)
    states, phantoms = [], set()
    for m in metals:
        name, sites, chirality, windings = stated[m]
        base, vertices, haptic = _collapse_haptic(base, donors[m])
        count = n_sites(name)
        if len(vertices) > count:
            raise ValueError(f"{name} has {count} sites but metal {m} has {len(vertices)} donor sites")
        padded = list(vertices) + [VACANT] * (count - len(vertices))
        order = _isomer.seat_order(padded, haptic, sites)
        seated = [padded[k] for k in order]
        derived = _coord_stereo.chirality_of(base, name, seated, haptic, roles)
        state = _core.from_vertices(m, *info[m], name, seated, haptic, hand=derived)
        identities = _isomer.primary_first(metal_info, m)
        centres = tuple(state if metal == m else _core.MetalState(metal, z, q) for metal, z, q in identities)
        iso = _isomer.Isomer._from_state(base, centres, donor_bonds)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        materialized = iso.vertices
        states.append(_core.state_with_winding(state, materialized, winding)._replace(hand=chirality))
        phantoms.update(haptic)

    identities = _isomer.primary_first(metal_info, selected)
    ordered = tuple(next(state for state in states if state.atom == metal) for metal, _z, _q in identities)
    source = strip_phantoms(base, phantoms)
    lengths = _isomer.length_source(source, lengths)
    return _isomer.Isomer._from_state(
        source,
        ordered,
        donor_bonds,
        constrained_metals={state.atom for state in ordered},
        lengths=lengths,
    )


def _with_fix(iso, fix):
    """Compose a geometric fix onto an already stated coordination identity."""
    if not fix:
        return iso
    cons, graft_ref = resolve_core(iso._graph, fix=fix, has_geometry=bool(iso._graph.GetNumConformers()))
    if not isinstance(fix, dict):
        graft_ref = {}
    return _isomer.Isomer._from_state(
        iso._graph,
        iso.centres,
        iso.donor_bonds,
        constraints=compose(iso._base_cons, cons),
        constrained_metals=iso._constrained_metals,
        lengths=iso._lengths,
        graft_ref=graft_ref,
        stereo_ref=iso.stereo_ref,
        stereo_label=iso.stereo_label,
    )


def _winding_variants(iso, state):
    """Return the symmetry-distinct assignments of one state's undefined haptic windings."""
    vertices, haptic, stored, donors = _core.materialized_state(iso, state)
    ranks = _coord_stereo.donor_classes(iso._graph, donors)
    faces = [
        dummy
        for dummy, face in haptic.items()
        if dummy not in stored and _coord_stereo.face_has_orientation(iso._graph, face, ranks)
    ]
    if not faces:
        return [state]
    variants, seen = [], set()
    for signs in itertools.product("+-", repeat=len(faces)):
        winding = stored | dict(zip(faces, signs, strict=True))
        signature = _isomer.winding_signature(iso, state, winding)
        if signature not in seen:
            seen.add(signature)
            variants.append(_core.state_with_winding(state, vertices, winding))
    return variants


def _enumerate_haptic_windings(isomers):
    """Expand the primary centre's undefined haptic orientations."""
    return _isomer.IsomerSet(
        iso._with_stereo((state, *iso.centres[1:]))
        for iso in isomers
        for state in _winding_variants(iso, iso.centres[0])
    )


def _haptic_mode(isomers, mode):
    """Apply the requested mode to haptic orientation read from an input geometry."""
    if mode in ("free", "preserve"):
        return isomers
    out = _isomer.IsomerSet()
    for iso in isomers:
        choices = []
        for state in iso.centres:
            vertices, _haptic, stored, _donors = _core.materialized_state(iso, state)
            candidate = state
            if mode == "invert":
                winding = {face: "+" if sign == "-" else "-" for face, sign in stored.items()}
                candidate = _core.state_with_winding(state, vertices, winding)
            elif mode != "unassigned":  # racemic / separate: discard the measured orientation first
                candidate = _core.state_with_winding(state, vertices, {})
            choices.append(_winding_variants(iso, candidate))
        out.extend(iso._with_stereo(states) for states in itertools.product(*choices))
    return out


def _enumerate_all_centers(mol, geometry, fix, haptic_mode, stereo_ref, lengths):
    """Stack independently enumerated metal states into their Cartesian product."""
    metals = metal_indices(mol)
    if len(metals) < _MULTI_METAL:
        raise ValueError("center='all' needs at least two transition-metal centres")
    stated = stated_arrangement(mol, center=metals[0])
    metals = _isomer.canonical_metals(mol, metals, allow_ties=stated is not None)
    if stated is not None:
        if geometry is not None:
            raise ValueError(
                f"this input already states every metal arrangement, so geometry={geometry!r} has nothing to act on"
            )
        iso = _with_fix(_from_stated_arrangements(mol, metals[0], lengths), fix)
        return _haptic_mode(_isomer.IsomerSet([iso]), haptic_mode)
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
    lengths = _isomer.length_source(real_base, lengths)

    fix_cons = Constraints()
    graft_ref = {}
    if fix:
        fix_cons, graft_ref = resolve_core(base, fix=fix, has_geometry=True)
        if not isinstance(fix, dict):
            graft_ref = {}  # frozen atoms are grafted from the retained source conformer; no external frame owns them
    frozen_core = Constraints(frozen=set(fix_cons.frozen))
    choices = []
    for m in metals:
        donors, haptic = sphere[m]
        base_iso = _isomer.from_surrogate(base, _isomer.primary_first(metal_info, m), donor_bonds)
        candidates = _isomer.IsomerSet()
        for geom in _select_geometries(base, m, donors, haptic, None, len(donors)):
            candidates.extend(
                _isomers_for_geometry(
                    base_iso,
                    geom,
                    donors=donors,
                    haptic=haptic,
                    frozen_donors=fix_cons.frozen & set(donors),
                    fix_cons=frozen_core,
                    retain=Constraints(),
                    ref_sig=stereo_ref,
                    lengths=lengths,
                    source=real_base,
                    graft_ref=graft_ref,
                )
            )
        choices.append(_haptic_mode(candidates, haptic_mode))

    out = _isomer.IsomerSet()
    for selected in itertools.product(*choices):
        centres = tuple(candidate.centres[0] for candidate in selected)
        out.append(
            _isomer.Isomer._from_state(
                real_base,
                centres,
                donor_bonds,
                constraints=fix_cons,
                constrained_metals={state.atom for state in centres},
                lengths=lengths,
                graft_ref=graft_ref,
                stereo_ref=stereo_ref,
                shared_mol=True,
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
    others; ``center='all'`` composes every centre's independently enumerated state. A geometry input
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
    out = _isomer.IsomerSet()
    for variant, stated_label in variants:
        built = _enumerate_coordination(variant, geometry, center, fix, haptic_mode, stereo_ref, lengths)
        label = _stereo.defined_stereo_label(variant, exclude=metal_indices(variant)) or stated_label
        point = _stereo.point_stereo(stated_label)
        tag_source = None
        if point:
            tag_source = Chem.Mol(variant)
            tag_source.RemoveAllConformers()
            tag_source, _metal_info = surrogate_all_metals(tag_source)
        if tag_source is not None and built:
            shared = Chem.Mol(built[0]._graph)
            for donor in point:
                shared.GetAtomWithIdx(donor).SetChiralTag(tag_source.GetAtomWithIdx(donor).GetChiralTag())
            for iso in built:
                iso._mol, iso._shared_mol = None, shared
        for iso in built:
            iso.stereo_label = label
            out.append(iso)
    if n_unassigned:
        logger.info(
            "metal: %d unassigned ligand stereo element(s) -> %d ligand variant(s) -> %d candidate(s)",
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
        if geometry is not None and resolve_geometry(geometry) != name:
            raise ValueError(
                f"this input already states a {name} arrangement, so geometry={geometry!r} has nothing to act on; "
                f"drop it to use what the input says, or strip the arrangement to enumerate"
            )
        logger.info("using stated %s arrangement", describe(name))
        if len(metal_indices(mol)) > 1:
            iso = _with_fix(_from_stated_arrangements(mol, center, lengths), fix)
            return _haptic_mode(_isomer.IsomerSet([iso]), haptic_mode)
        iso = _isomer.Isomer(mol, name, sites, lengths=lengths)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        vertices = iso.vertices
        state = iso.centres[0]
        iso = iso._with_stereo((_core.state_with_winding(state, vertices, winding)._replace(hand=chirality),))
        iso = _with_fix(iso, fix)
        return _haptic_mode(_isomer.IsomerSet([iso]), haptic_mode)
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no transition metal found")
    haptic = {}  # centroid dummies, keyed dummy -> its face atoms; set by _collapse_haptic on the single path
    # One metal means no spectators, so this is the single-centre path whether or not center= names it. An
    # explicit center= is still validated against the sole metal, raising on a wrong index or element.
    if len(metals) == 1:
        if center is not None:
            _isomer.resolve_center(mol, metals, center)
        base, m, real_donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = _collapse_haptic(base, real_donors)
        base_iso = _isomer.from_surrogate(base, ((m, real_z, real_q),), [(donor, m) for donor in real_donors])
        retain = Constraints()
    else:
        base_iso, donors, haptic, retain = _prepare_spectators(mol, metals, center)
        base, m = base_iso._graph, base_iso.metal
    source = strip_phantoms(Chem.Mol(base), set(haptic))
    lengths = _isomer.length_source(source, lengths)  # once per molecule, not per ordering
    fix_cons = Constraints()
    graft_ref = {}
    frozen_donors = set()
    # Hold a reacting TS core while the rest of the coordination sphere is enumerated: coordinate forms need
    # an input geometry, while numeric distances and angles are complete on a coordinate-free graph.
    if fix:
        fix_cons, graft_ref = resolve_core(base, fix=fix, has_geometry=base.GetNumConformers() > 0)
        if not isinstance(fix, dict):
            graft_ref = {}  # `frozen` plus the retained conformer carries a same-source list fix
        frozen_donors = fix_cons.frozen & set(donors)
        logger.info(
            "metal: fixed %d input atoms; enumerating free coordination sites",
            len(fix_cons.constrained_atoms()),
        )
    geoms = _select_geometries(base, m, donors, haptic, geometry, len(donors))
    out = _isomer.IsomerSet()
    for geom in geoms:
        out.extend(
            _isomers_for_geometry(
                base_iso,
                geom,
                donors=donors,
                haptic=haptic,
                frozen_donors=frozen_donors,
                fix_cons=fix_cons,
                retain=retain,
                ref_sig=stereo_ref,
                lengths=lengths,
                source=source,
                graft_ref=graft_ref,
            )
        )
    if mol.GetNumConformers():
        out = _haptic_mode(out, haptic_mode)
    elif haptic_mode != "free":
        out = _enumerate_haptic_windings(out)
    return out
