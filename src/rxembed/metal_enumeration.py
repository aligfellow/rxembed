"""Enumerate ligand, haptic and coordination states into ready-to-embed `Isomer` candidates."""

from __future__ import annotations

import functools
import itertools
import math
import re

import numpy as np
from rdkit import Chem, DistanceGeometry

from . import bounds as _bounds
from . import mechanisms as _mech
from . import metal_constraints as _constraints
from . import metal_core as _core
from . import metal_donor_orient as _orient
from . import metal_isomer as _isomer
from . import metal_slots as _slots
from . import metal_stereo as _coord_stereo
from . import stereo as _stereo
from .constraints import FIX_DISTANCE_TOL, Constraints, compose, resolve_core
from .metal_core import (
    VACANT,
    _collapse_haptic,
    _reject_metal_bonds,
    logger,
    metal_indices,
    n_sites,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_polyhedron import (
    CHELATE_SPAN_ANGLE,
    POLYHEDRA,
    SLOT_BOND_PROP,
    _vertex_angle,
    describe,
    geometries_for_cn,
    read_slot_notes,
    resolve_geometry,
)

_MULTI_METAL = 2
_BOND_CENTRE = 2
_CHELATE_PATH_MIN = 3
_CHELATE_SPAN_MIN = 120.0
_HAPTIC_MATCHING_CAP = 7  # factorial matching remains tiny for ordinary pi faces; larger faces use the safe mean bound.
_FIT_ROUNDOFF = 1e-10  # Numerical comparison on unit rays, not an angular or chemical tolerance.
_SCREEN_SITES = 8  # Keep the finite raw pool bounded before applying whole-network reach screening.
_EUCLIDEAN_SUBSET_BUDGET = 128  # Keep local route certificates bounded; the full route remains authoritative.
_ROUTE_ATOMS_KEY = "route_atoms"  # context cache: _route_certificate_paths is candidate-independent, see there.
_COMPILED_CONSTRAINTS_KEY = "_span_compiled"  # context scratch slot: this candidate's compiled constraints,
# written by _compiled_span_failure and consumed once by _unreachable_span; never read stale (see both).


def _screen_limit(site_count, reach):
    """Bound the raw tethered pool before applying the normal limit to surviving candidates."""
    if reach is not None and site_count <= _SCREEN_SITES:
        return math.factorial(site_count)
    return None


def _keep_screened_candidate(out, candidate, geometry):
    """Append one surviving screened candidate without exceeding the public orbit bound."""
    out.append(candidate)
    if len(out) > _slots._MAX_EXHAUSTIVE_ORBITS:
        raise _slots._assignment_cap_error(geometry)


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
    metal_donors = {
        neighbor.GetIdx() for metal in metal_indices(mol) for neighbor in mol.GetAtomWithIdx(metal).GetNeighbors()
    }
    clear.update(
        index
        for index in skip
        if exact.get(index) == "free" or (broad and point_mode == "free" and index in metal_donors)
    )
    return has_geometry, point_mode, ez_mode, axial_mode, exact, clear, skip


def _clear_ez(mol, pairs=None):
    """Clear selected double-bond stereo and adjacent slash bonds; return whether anything changed."""
    return _stereo._clear_ez(mol, pairs)


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
            point = re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|r|s|CW|CCW)", part)
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
    # RDKit decides whether a metal-closed ring still has independent E/Z. Strip only the remainder, whose
    # apparent bond geometry is already encoded by the selected coordination arrangement.
    locked_ez = _stereo._coordination_locked_double_bonds(mol, metal_indices(mol))
    changed = False
    unstable_bonds = set()
    if has_geometry:
        measured = _stereo.stereo_from_3d(mol, exclude=metal_indices(mol))
        measured_point_labels = _stereo.point_stereo(measured)
        measured_points = set(measured_point_labels)
        stated_points = _stereo.point_stereo(_stereo.defined_stereo_label(mol, exclude=metal_indices(mol)))
        centres = _stereo.point_centres(mol, exclude=metal_indices(mol))
        skip.update(
            index for index in centres if exact.get(index, point_mode) == "preserve" and index not in measured_points
        )
        axes = _stereo.axis_stereo(measured)
        _stereo._assign_atrop_from_3d(mol, axes)
        changed |= bool(axes)
        unstable_bonds = {
            frozenset(centre)
            for centre in _stereo.unassigned_centres(mol, exclude=metal_indices(mol))
            if len(centre) == _BOND_CENTRE and frozenset(centre) not in _stereo.bond_stereo(measured)
        }
    for index in clear:
        atom = mol.GetAtomWithIdx(index)
        changed |= atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if has_geometry:
        fixed = {index for index in centres & measured_points if exact.get(index, point_mode) in ("preserve", "invert")}
        rewrite = {index for index in fixed if stated_points.get(index) != measured_point_labels[index]}
        _stereo.apply_point_stereo(mol, measured, rewrite)
        for index in fixed:
            if exact.get(index, point_mode) == "invert":
                atom = mol.GetAtomWithIdx(index)
                atom.SetChiralTag(_stereo.mirror_tag(atom.GetChiralTag()))
        changed |= bool(rewrite) or any(exact.get(index, point_mode) == "invert" for index in fixed)
    skip_bonds = True if ez_mode == "free" else unstable_bonds
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
        label = (
            ",".join(part for part in measured.split(",") if not set(_stereo.point_stereo(part)) & skip)
            if has_geometry
            else _stereo.defined_stereo_label(variant, exclude=metal_indices(variant))
        )
        variants = [(variant, label)]
    variants = [(variant, _stereo._without_bond_stereo(label, locked_ez)) for variant, label in variants]
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
    retained_states = {}
    spectator_phantoms = set()
    for s in spectators:
        before = base.GetNumAtoms()
        base, retained_states[s] = _isomer.retained_state(mol, base, metal_info, metals, s)
        spectator_phantoms.update(range(before, base.GetNumAtoms()))
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
    return base_iso, donors, haptic


def _select_geometries(base, m, donors, haptic, geometry, n):
    """Resolve `geometry` to the list of polyhedron names to enumerate (default from donor count, or as given)."""
    measured = None
    if geometry is None:
        selected, measured, apical = _isomer.retained_geometry(base, m, donors, haptic)
        if selected is None:
            raise ValueError(
                f"no default geometry for {n} donors; pass geometry= a name or list "
                f"(options for {n} donors: {[p.name for p in geometries_for_cn(n)]})"
            )
        geoms = [selected]
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


def _frozen_permutations(base, m, padded, geom, frozen_donors, sites, haptic=None, coordination=()):
    """Generate every free-donor vertex permutation with each frozen donor pinned at its input vertex.

    Returns a lazy permutation iterator, bypassing the symmetry-reduced `isomer_permutations`, which would
    miss the representative ordering a valid frozen isomer needs; or ``None`` if the input ordering can't be
    read.
    """
    base_order = _slots.input_ordering(base, m, padded, geom, haptic, coordination)
    if base_order is None:
        return None
    frozen_v = {v: di for v, di in enumerate(base_order) if padded[di] in frozen_donors}
    free_v = [v for v in range(sites) if v not in frozen_v]
    free_di = [di for di in range(len(padded)) if padded[di] not in frozen_donors]
    count = math.factorial(len(free_di))
    if count > _slots._MAX_EXHAUSTIVE_ORBITS:
        raise ValueError(
            f"metal[{geom}]: fix= leaves exactly {count:,} free-site arrangements to scan "
            f"(limit {_slots._MAX_EXHAUSTIVE_ORBITS:,}); retain more donor positions or retain the "
            "coordinate arrangement with rx.embed(mol)"
        )
    logger.info(
        "metal[%s]: %d frozen donors; deduplicating %d free-site arrangements",
        geom,
        len(frozen_v),
        count,
    )

    def generate():
        for fp in itertools.permutations(free_di):
            order = [None] * sites
            for vertex, donor_index in frozen_v.items():
                order[vertex] = donor_index
            for vertex, donor_index in zip(free_v, fp, strict=False):
                order[vertex] = donor_index
            yield order

    return generate()


def _radial_distance_windows(mol, metal, real_z, donors, atoms, positions, base_distances=None):
    """Return each donor atom's model M-donor distance window and hybridisation, shared by both screens.

    `positions` is resolved by the caller: `_unreachable_span` measures from `iso._length_mol`, which need
    not be `mol` itself.
    """
    hyb = _constraints._stripped_hybridisation(mol)
    charges = _constraints.delocalised_charges(mol) if positions is None else None
    radial = Constraints()
    base_distances = base_distances or {}
    for donor in atoms:
        key = tuple(sorted((metal, donor)))
        radial.distances[key] = base_distances.get(key) or _constraints._donor_distance_window(
            mol, metal, donor, real_z, donors, positions=positions, charges=charges, hyb=hyb
        )
    return radial, hyb


def _narrow_span_pairs(base, source, padded, haptic, base_iso, lengths, native_reach):
    """Return same-ligand donor-position pairs that cannot span `CHELATE_SPAN_ANGLE` (135 deg) or more.

    No-loss: a compiled angle row this wide keeps a floor >= CHELATE_SPAN_ANGLE - _ANGLE_PAD (127 deg). Any
    path that could instead recentre that row (`_joint_shell_targets`, `_planar_bite_targets`'s fan branch)
    only commits a witness whose span also fits the same native reach within 1e-7, which this test has
    already shown a pair this wide cannot do.
    """
    metal = base_iso.metal
    frag = _core._frag_map(base)
    real_slots = [i for i, donor in enumerate(padded) if donor != VACANT and donor not in haptic]
    atoms = {padded[i] for i in real_slots}
    positions, _ = _constraints.resolve_lengths(source, lengths)
    radial, _hyb = _radial_distance_windows(source, metal, base_iso.real_z, set(base_iso.donors), atoms, positions)
    span = (CHELATE_SPAN_ANGLE - _constraints._ANGLE_PAD, 180.0)
    narrow = set()
    for i, j in itertools.combinations(real_slots, 2):
        a, b = padded[i], padded[j]
        if frag[a] != frag[b] or base.GetBondBetweenAtoms(a, b) is not None:
            continue
        left, right = radial.distances[tuple(sorted((metal, a)))], radial.distances[tuple(sorted((metal, b)))]
        x, y = sorted((a, b))
        if _mech._triangle_distances(left, right, span)[0] > float(native_reach[x, y]) + _slots._SPAN_TOL:
            narrow.add(frozenset((i, j)))
    return frozenset(narrow)


def _isomers_for_geometry(
    base_iso,
    geom,
    *,
    donors,
    haptic,
    frozen_donors,
    fix_cons,
    ref_sig,
    lengths,
    source,
    graft_ref=None,
    screen_reach=True,
    observed_only=False,
):
    """Enumerate every distinct `Isomer` of one polyhedron `geom` (frozen core held, spectators retained)."""
    base, m = base_iso._graph, base_iso.metal
    n = len(donors)
    sites = n_sites(geom)
    if n > sites:
        raise ValueError(f"{geom} has {sites} coordination sites but the metal has {n} donors")
    padded = list(donors) + [VACANT] * (sites - n)  # leave empty vertices as a pocket
    if sites - n:
        logger.info(
            "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)", geom, sites, n, sites - n
        )
    roles = _isomer.isomer_roles(base_iso)
    perms = None
    if frozen_donors and sites == n:  # pin each frozen donor at its input vertex, then generate every
        perms = _frozen_permutations(
            base, m, padded, geom, frozen_donors, sites, haptic, roles
        )  # free-donor arrangement
    real_donors = base_iso.donors
    tethered = _slots._has_tether(padded, _core._frag_map(base), haptic)
    classes = _coord_stereo.site_classes(base, padded, haptic, roles)
    distances = _core._ligand_distance_matrix(base) if tethered else None
    retained = (
        _slots.input_ordering(base, m, padded, geom, haptic, roles, classes=classes)
        if observed_only and base.GetNumConformers() and _isomer.retained_geometry(base, m, donors, haptic)[0] == geom
        else None
    )
    if observed_only:
        if retained is None:
            raise ValueError(
                "observed_only=True requires an input conformer whose donor arrangement matches the requested "
                f"{geom} geometry"
            )
        # The measured order is an explicit user choice, not an approximation to exhaustive enumeration.
        # Passing it as the permutation stream avoids constructing an oversized constitutional orbit pool.
        perms = (tuple(retained),)
    reach = None
    if tethered and screen_reach and not fix_cons.constrained_atoms():
        try:
            reach = _bounds.ligand_reach(base)
        except (ValueError, RuntimeError):
            logger.debug("metal[%s]: native ligand reach unavailable; retaining all arrangements", geom)
    native_reach = (
        _bounds._coordination_reach_base(source, reach, {m})
        if reach is not None and len(base_iso.centres) == 1
        else None
    )
    screen_context = _constraints.compile_context(source) if native_reach is not None else None
    # Prune same-ligand pairs the compiled screen would reject wholesale before the streamed tethered pool
    # (metal_slots.distinct_vertex_orderings' uncapped else-branch) can raise its resource cap on them.
    narrow = (
        _narrow_span_pairs(base, source, padded, haptic, base_iso, lengths, native_reach)
        if native_reach is not None
        else frozenset()
    )
    # A same-ligand chelate-backbone or direct-bond pair must sit on a polyhedron hull edge (measured claim;
    # see metal_slots._chelate_edge_links). Same gate as `narrow`: a single screened centre with no fix=.
    linked = _slots._chelate_edge_links(base, padded, haptic, distances) if native_reach is not None else frozenset()
    out, unreachable = [], 0
    raw_limit = _screen_limit(n, reach)
    # Reading a ring's winding depends on the input geometry, not on which vertex a donor lands at,
    # so it is the same for every candidate below.
    winding = _isomer.measured_haptic_windings(source, m, real_donors, haptic)
    for order in _slots.distinct_vertex_orderings(
        base,
        padded,
        geom,
        perms=perms,
        haptic=haptic,
        classes=classes,
        distances=distances,
        retained=retained,
        max_orbits=raw_limit,
        narrow=narrow,
        linked=linked,
    ):
        od = [padded[k] for k in order]  # vertex -> donor atom, haptic centroid, or VACANT
        links = _coord_stereo.chelate_links(base, od, haptic, distances) if tethered else ()
        hand = _coord_stereo.chirality_of(base, geom, od, haptic, roles, classes=classes, links=links)
        active = _core.from_vertices(m, base_iso.real_z, base_iso.real_q, geom, od, haptic, winding.items(), hand)
        centres = (active, *(centre for centre in base_iso.centres if centre.atom != m))
        candidate = _isomer.Isomer._from_state(
            source,
            centres,
            base_iso.donor_bonds,
            constraints=fix_cons,
            constrained_metals={state.atom for state in centres if state.geometry},
            lengths=lengths,
            graft_ref=graft_ref,
            stereo_ref=ref_sig,
            shared_mol=True,
        )
        if reach is not None and (retained is None or tuple(order) != tuple(retained)):
            failure = _unreachable_span(candidate, reach, classes, links, native_reach, screen_context)
            if failure:
                logger.debug("metal[%s]: omit arrangement: %s", geom, failure)
                unreachable += 1
                continue
        _keep_screened_candidate(out, candidate, geom)
    if unreachable:
        logger.info(
            "metal[%s]: omitted %d arrangements with incompatible coordination/ligand targets", geom, unreachable
        )
    if not out and (unreachable or narrow or linked):
        # narrow/linked can prune every streamed order (e.g. a linear pocket has no hull edge at all), which
        # never reaches the per-candidate screen below and so never increments `unreachable` either.
        logger.warning(
            "metal[%s]: no model-compatible arrangement; use screen=False, lengths='input' if a model "
            "M-donor distance is the contradiction, or fix= for bond changes",
            geom,
        )
    elif not out:
        logger.warning("metal[%s]: exact symmetry enumeration produced no arrangement", geom)
    return out


def _chelate_span_failure(iso, reach, radial, links, compiled=None):
    """Reject a linked target whose slot separation exceeds its native upper reach."""
    mol, metal, vertices, haptic = iso._graph, iso.metal, iso.vertices, iso.haptic
    directions = POLYHEDRA[iso.geometry].vertex_dirs
    measured = iso._lengths == "input"
    # Acute or coupled bites are soft DG priors; only independent long arcs are an outer bound.
    independent = len({vertex for pair in links for vertex in pair}) == 2 * len(links)
    for pair in links or ():
        i, j = sorted(pair)
        left, right = vertices[i], vertices[j]
        if VACANT in (left, right):
            continue
        angle = _vertex_angle(directions[i], directions[j])
        haptic_pair = bool({left, right} & haptic.keys())
        if haptic_pair:
            if (not measured and compiled is None) or angle < _slots.TRANS_ANGLE:
                continue
            failure = _haptic_span_failure(iso, reach, metal, left, right, angle, measured, compiled)
            if failure is not None:
                return failure
            continue
        if mol.GetBondBetweenAtoms(left, right) is not None:
            continue
        if not independent or links[pair] < _CHELATE_PATH_MIN or angle < _CHELATE_SPAN_MIN:
            continue
        radii = tuple(radial.distances[tuple(sorted((metal, donor)))][0] for donor in (left, right))
        needed = _mech._law_of_cosines(radii[0], radii[1], angle)
        available = float(reach[min(left, right), max(left, right)])
        if math.isfinite(available) and needed > available + _slots._SPAN_TOL:
            return f"chelate donors {left}/{right} need >= {needed:.3f} A; ligand reach <= {available:.3f} A"
    return None


def _haptic_span_failure(iso, reach, metal, left, right, angle, measured, compiled):
    """Check centroid reach; member rays need not share the angle between centroid sites."""
    left_face, right_face = iso.haptic.get(left, (left,)), iso.haptic.get(right, (right,))
    try:
        if measured:
            radii = tuple(
                _measured_radius(iso, vertex, face) for vertex, face in ((left, left_face), (right, right_face))
            )
        else:
            assert compiled is not None
            radii = tuple(compiled.distances[tuple(sorted((metal, vertex)))][0] for vertex in (left, right))
        available = _centroid_reach(reach, left_face, right_face)
    except (KeyError, IndexError, ZeroDivisionError):
        return None
    needed = _mech._law_of_cosines(radii[0], radii[1], angle)
    if math.isfinite(available) and needed > available + _slots._SPAN_TOL:
        return f"haptic faces {left}/{right} need >= {needed:.3f} A; centroid reach <= {available:.3f} A"
    return None


def _measured_radius(iso, vertex, face):
    """Return the observed metal-to-site radius for a real atom or haptic centroid."""
    positions = iso._length_mol.GetConformer().GetPositions()
    point = np.mean(positions[list(face)], axis=0) if vertex in iso.haptic else positions[vertex]
    return float(np.linalg.norm(point - positions[iso.metal]))


def _centroid_reach(reach, left, right):
    """Return a safe upper bound for the distance between two donor centroids.

    For any matching of equal-size faces, the centroid difference is the mean of the matched
    displacement vectors.  The triangle inequality therefore makes the matching mean an upper
    bound; taking the smallest matching preserves that certificate while removing the loose
    all-pairs mean.  Unequal or unusually large faces retain the uniform coupling bound.
    """
    values = [[float(reach[min(a, b), max(a, b)]) for b in right] for a in left]
    if not values or not values[0] or not all(math.isfinite(value) for row in values for value in row):
        return math.inf
    if len(left) == len(right) and len(left) <= _HAPTIC_MATCHING_CAP:
        return min(
            sum(row[index] for row, index in zip(values, order, strict=True))
            for order in itertools.permutations(range(len(right)))
        ) / len(left)
    return sum(map(sum, values)) / (len(left) * len(right))


def _unreachable_span(iso, reach, classes, links, native=None, context=None):
    """Screen jointly compiled targets, retaining the prior screen for unsupported relations.

    Haptics retain their separate real-atom network; centroid feasibility remains an embedding/FF concern.
    The prior directional test covers multi-centre states and base constraints, where this selected-sphere
    certificate cannot yet own the spectator geometry. That test permits distortion down to TRANS_ANGLE;
    above 90 degrees, separation increases with both M-L lengths. Neither screen proves chemical impossibility.
    """
    mol, metal, vertices, haptic = iso._graph, iso.metal, iso.vertices, iso.haptic
    directions = POLYHEDRA[iso.geometry].vertex_dirs
    # A multi-centre candidate has a selected sphere plus a spectator state; keep its prior outer screen until
    # one certificate owns both spheres jointly.
    compiled = None
    if len(iso.centres) == 1:
        failure = _compiled_span_failure(iso, reach, None, native, context)
        if failure is not None:
            return failure
        if links or iso.haptic:
            # `_compiled_span_failure` already compiled this exact candidate's full constraints (it must
            # succeed to reach here); reuse them instead of compiling the same candidate a second time.
            compiled = context.pop(_COMPILED_CONSTRAINTS_KEY, None) if context is not None else None
            if compiled is None:
                compiled = iso._constraints(context=context)
        if not iso.haptic and not links:
            return None
    pairs = [
        (left, right)
        for (i, left), (j, right) in itertools.combinations(enumerate(vertices), 2)
        if VACANT not in (left, right)
        and not {left, right} & haptic.keys()
        and mol.GetBondBetweenAtoms(left, right) is None
        and _vertex_angle(directions[i], directions[j]) >= _slots.TRANS_ANGLE
    ]
    if not pairs and not links:
        return None
    donors = set(iso.donors)
    atoms = set(vertices) - haptic.keys() - {VACANT}
    positions, _ = _constraints.resolve_lengths(iso._length_mol, iso._lengths)
    radial, hyb = _radial_distance_windows(mol, metal, iso.real_z, donors, atoms, positions, iso._base_cons.distances)
    if failure := _chelate_span_failure(iso, reach, radial, links, compiled):
        return failure
    for left, right in pairs:
        a, b = (radial.distances[tuple(sorted((metal, donor)))][0] for donor in (left, right))
        needed = _mech._law_of_cosines(a, b, _slots.TRANS_ANGLE)
        available = float(reach[min(left, right), max(left, right)])
        if needed > available + _slots._SPAN_TOL:
            return f"donors {left}/{right} need >= {needed:.3f} A; ligand reach <= {available:.3f} A"
    return _donor_facing_failure(iso, reach, radial, hyb, classes, links)


def _route_certificate_subsets(path):
    """Return a bounded set of local route witnesses after the complete route."""
    subset_count = sum(math.comb(len(path), n) for n in (3, 4) if n < len(path))
    if subset_count > _EUCLIDEAN_SUBSET_BUDGET:
        return (path,)
    return itertools.chain((path,), *(itertools.combinations(path, n) for n in (3, 4) if n < len(path)))


def _route_has_donor_bond(mol, path, donors):
    """Return whether a shortest donor route contains a bond between two metal donors.

    The route certificate assumes ordinary two-centre ligand connectivity. A donor-donor edge is a bridge or
    multicentre donor network, where its native bond bounds are not a valid proxy for the independent metal-ray
    geometry. Leave that route to the normal embedding and acceptance gates.
    """
    return any(
        left in donors and right in donors and mol.GetBondBetweenAtoms(int(left), int(right)) is not None
        for left, right in itertools.combinations(path, 2)
    )


def _route_certificate_paths(mol, donors, topology):
    """Return each donor-pair route's deduped, bond-filtered atom path, independent of vertex seating.

    A donor pair's shortest route (ties included, via the topology union) and whether that route crosses
    a donor-donor bond depend only on molecular topology and which atoms are donors, never on which
    polyhedron vertex a candidate seats a donor at. `_compiled_span_failure` computes this list once per
    (mol, donor set) through its `context` cache and reuses it for every screened seating. Each path's
    subset certificates stay lazy in the caller, not expanded here, so a whole-route conflict still skips
    ever generating its smaller diagnostic subsets (see `_route_certificate_subsets`).
    """
    tested_paths = set()
    paths = []
    for left, right in itertools.combinations(sorted(donors), 2):
        separation = topology[left, right]
        if not 0 < separation < mol.GetNumAtoms():
            continue
        # The union includes every equally short route through a cyclic ligand. Selecting one route
        # depends on atom numbering; skipping ties instead loses the reach check for the entire chelate.
        path = tuple(map(int, np.flatnonzero(topology[left] + topology[right] == separation)))
        if path in tested_paths:
            continue
        tested_paths.add(path)
        if _route_has_donor_bond(mol, path, donors):
            continue
        paths.append(path)
    return paths


def _compiled_span_failure(iso, reach, compiled=None, native=None, context=None):
    """Reject a jointly inconsistent compiled donor network, not an unsuccessful embedding attempt."""
    mol = iso._graph
    if compiled is None:
        constraints = iso._constraints(force_field=False, context=context) if context is not None else iso.cons
    else:
        constraints = compiled
    if constraints is not None:
        native = _bounds._coordination_reach_base(mol, reach, {iso.metal}) if native is None else native
        for (left, centre, right), angles in constraints.angles.items():
            if centre != iso.metal or constraints.haptic.keys() & {left, right}:
                continue
            legs = (
                constraints.distances.get(tuple(sorted((left, centre)))),
                constraints.distances.get(tuple(sorted((centre, right)))),
            )
            if legs[0] is None or legs[1] is None:
                continue
            a, b = sorted((left, right))
            lower, upper = _mech._triangle_distances(*legs, angles)
            native_lower, native_upper = native[b, a], native[a, b]
            if lower > native_upper + _bounds._CROSS_EPS or upper < native_lower - _bounds._CROSS_EPS:
                return "compiled coordination distances conflict with native ligand reach"
        if context is not None and compiled is None:
            constraints = iso._constraints(context=context)
            # Stash this candidate's full compile for `_unreachable_span`'s chelate/haptic follow-up check,
            # which otherwise recompiles the identical (mol, candidate) constraints a second time.
            context[_COMPILED_CONSTRAINTS_KEY] = constraints
    matrix = (
        _bounds.coordination_reach(mol, constraints, reach)
        if native is None
        else _bounds.coordination_reach(mol, constraints, reach, native=native)
    )
    closed = matrix.copy()
    if not DistanceGeometry.DoTriangleSmoothing(closed):
        upper, lower = np.triu(matrix, 1), np.tril(matrix, -1)
        upper = _bounds._upper_closure(upper + upper.T)
        lower = lower + lower.T
        raised = _bounds._lower_reach(lower, upper)
        if np.any(np.maximum(np.maximum(raised, raised.T), lower) - upper > _bounds._CROSS_EPS):
            return "compiled coordination distances conflict with native ligand reach"
        return None  # An unexplained native failure is not evidence against this arrangement.
    donors = set(iso.donors)
    if context is None:
        route_paths = _route_certificate_paths(mol, donors, Chem.GetDistanceMatrix(mol, force=True))
    else:
        # The donor set is fixed for every candidate one `_isomers_for_geometry` call screens (only each
        # donor's assigned vertex varies), so this key never collides within one screen.
        route_key = (_ROUTE_ATOMS_KEY, frozenset(donors))
        route_paths = context.get(route_key)
        if route_paths is None:
            topology = Chem.GetDistanceMatrix(mol, force=True)
            route_paths = context[route_key] = _route_certificate_paths(mol, donors, topology)
    for path in route_paths:
        # Store only route unions, not their fourth-order number of subsets; stream those once per union.
        for subset in _route_certificate_subsets(path):
            atoms = sorted((iso.metal, *subset))
            # Refine the whole route once; its smaller diagnostic subsets retain the cheap midpoint test.
            if _bounds._euclidean_conflict(closed[np.ix_(atoms, atoms)], refine=subset == path) is not None:
                return f"compiled coordination distances have no Euclidean realization at atoms {atoms}"
    return None


def _donor_pair_fit_cost(reach, radii, axes, floors, donors):
    """Lower-bound opposed-pair fit cost using length-free donor cones."""

    def gap(theta):
        a, b = radii[0][0], radii[1][0]
        span = _mech._law_of_cosines(a, b, math.degrees(theta))
        gaps = [0.0]
        for k, other in enumerate(reversed(donors)):
            beta = math.atan2(radii[1 - k][1] * math.sin(theta), radii[k][0] - radii[1 - k][1] * math.cos(theta))
            away = max(0.0, floors[k] - beta)
            for atom in axes[k]:
                available = float(reach[min(atom, other), max(atom, other)])
                if math.isfinite(available) and available > 0:
                    gaps.append(span * math.sin(min(away, math.pi / 2)) - available - _slots._SPAN_TOL)
        return max(gaps)

    if gap(math.pi) <= 0:
        return 0.0
    lo, hi = math.pi / 2, math.pi
    if gap(lo) > 0:
        hi = lo  # Any unexamined acute angle spends at least this much opposed-pair error.
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            if gap(mid) > 0:
                hi = mid
            else:
                lo = mid
    return 4 - 4 * math.sin(hi / 2)  # Use the upper bracket: a conservative lower fit cost.


def _donor_facing_failure(iso, reach, radial, hyb, classes, links):
    """Reject only when donor-facing reach exhausts the shared fit budget in every equivalent seating.

    An ideal opposed pair capped at angle theta costs at least 4-4*sin(theta/2) squared unit-ray error.
    Sum disjoint pairs against the one n*FIT_FLOOR^2 budget, not a fresh budget for each chelate. Caps
    follow from the existing single-endpoint cones without assuming a donor-substituent bond length.
    Native ligand reach intervals remain model priors, not proof of chemical impossibility.
    """
    vertices, mol, metal = iso.vertices, iso._graph, iso.metal
    occupied = sum(vertex != VACANT for vertex in vertices)
    if len(iso.centres) != 1:
        return None
    hyb = _orient._stripped_hybridisation(mol) if hyb is None else hyb
    donors = set(iso.donors)
    # Coordination can expand radial caps for bonded co-donors; the uncompiled windows cannot bound them.
    bonded_donors = {d for d in donors if any(n.GetIdx() in donors for n in mol.GetAtomWithIdx(d).GetNeighbors())}
    directions = [
        tuple(value / math.hypot(*direction) for value in direction)
        for direction in POLYHEDRA[iso.geometry].vertex_dirs
    ]
    opposed = [
        (i, j)
        for i, j in itertools.combinations(range(len(vertices)), 2)
        if VACANT not in (vertices[i], vertices[j])
        and math.dist(directions[i], tuple(-value for value in directions[j])) <= _FIT_ROUNDOFF
    ]
    used = [slot for pair in opposed for slot in pair]
    if not opposed or len(used) != len(set(used)):
        return None  # A nonstandard template cannot spend one donor's fit error twice.

    @functools.cache
    def pair_cost(left, right):
        if {left, right} & (iso.haptic.keys() | bonded_donors):
            return 0.0
        if not Chem.GetShortestPath(mol, left, right):
            return 0.0
        radii = [radial.distances[tuple(sorted((metal, atom)))] for atom in (left, right)]
        radii = [(lo - FIX_DISTANCE_TOL, hi + FIX_DISTANCE_TOL) for lo, hi in radii]
        if any(not (0 < lo <= hi) or not all(map(math.isfinite, (lo, hi))) for lo, hi in radii):
            return 0.0
        axes, floors = [], []
        for atom in (left, right):
            floor = _orient._FOLD_WINDOW.get((mol.GetAtomWithIdx(atom).GetSymbol(), hyb.get(atom)))
            floors.append(math.radians(floor[0]) if floor else 0.0)
            axes.append(tuple(_orient.donation_axis(mol, atom, donors, hyb=hyb, network=False) or ()) if floor else ())
        if not any(axes):
            return 0.0

        return _donor_pair_fit_cost(reach, radii, axes, floors, (left, right))

    keys = [None if vertex == VACANT else classes[vertex] for vertex in vertices]
    budget = occupied * _core._FIT_FLOOR**2
    minimum = math.inf
    for assignment in _coord_stereo.equivalent_site_assignments(keys, links):
        cost = sum(pair_cost(*sorted((vertices[assignment[i]], vertices[assignment[j]]))) for i, j in opposed)
        if cost <= budget + _FIT_ROUNDOFF:
            return None
        minimum = min(minimum, cost)
    if not math.isfinite(minimum):
        return None
    return f"donor-facing network needs squared fit error >= {minimum:.3f}; shape budget is {budget:.3f}"


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
    if chirality and not derived:
        raise ValueError(f"the metal note says {chirality}, but its stated seating is achiral")


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
        protect_arrangement=iso._protect_arrangement,
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


def _enumerate_all_centers(mol, geometry, fix, haptic_mode, stereo_ref, lengths, screen, observed_only):
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
                    ref_sig=stereo_ref,
                    lengths=lengths,
                    source=real_base,
                    graft_ref=graft_ref,
                    screen_reach=screen and not bool(fix),
                    observed_only=observed_only,
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


def enumerate_isomers(
    mol,
    geometry=None,
    center=None,
    fix=None,
    stereo=None,
    stereo_ref=None,
    lengths="model",
    *,
    screen=True,
    observed_only=False,
):
    """Enumerate distinct coordination and ligand stereoisomers of an RDKit Mol.

    `geometry` accepts a registry name or 3-letter code. `center=` selects one metal while retaining the
    others; ``center='all'`` composes every centre's independently enumerated state. A geometry input
    retains measured ligand and haptic stereo by default, while a graph enumerates undefined elements.
    `lengths='model'` uses model M-donor distances; ``lengths='input'`` explicitly measures them from coordinates.

    Parameters
    ----------
    screen : bool, optional
        Apply the native ligand-reach and donor-facing screens, by default True. False retains distinct
        assignments beyond those model limits, not guaranteed embeddable states. Symmetry, stated slots,
        ligand stereo, explicit fixes and embedding validation are unchanged.
    observed_only : bool, optional
        With coordinates, return only the measured donor arrangement. This is an explicit resource-bounded
        choice for high-coordinate inputs; it does not claim that unmeasured assignments are impossible.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError(
            f"enumerate_isomers() takes an RDKit Mol, got {type(mol).__name__}. Parse SMILES with "
            "rxembed.parse_smiles, or call rxembed.metal for a SMILES or .xyz source"
        )
    if not isinstance(screen, bool):
        raise TypeError("screen must be a bool")
    if not isinstance(observed_only, bool):
        raise TypeError("observed_only must be a bool")
    _reject_metal_bonds(mol)
    _core._reject_boron_cages(mol)
    mol = _core._canonical_metal_graph(mol)
    center, stereo = _source_defaults(mol, center, stereo)
    variants, haptic_mode, n_unassigned, unresolved = _ligand_stereo_variants(mol, stereo)
    out = _isomer.IsomerSet()
    for variant, stated_label in variants:
        built = _enumerate_coordination(
            variant, geometry, center, fix, haptic_mode, stereo_ref, lengths, screen, observed_only
        )
        defined = _stereo.defined_stereo_label(variant, exclude=metal_indices(variant))
        labels = {}
        for item in (*defined.split(","), *stated_label.split(",")):
            if item:
                labels[item.rsplit(":", 1)[0]] = item
        label = ",".join(labels.values())
        locked = _stereo._coordination_locked_double_bonds(variant, metal_indices(variant))
        label = _stereo._without_bond_stereo(label, locked)
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
            "metal: %d stereo element(s) not enumerable and 3D-measurable by RDKit; embedded without a hand",
            unresolved,
        )
    return out


def _enumerate_coordination(mol, geometry, center, fix, haptic_mode, stereo_ref, lengths, screen, observed_only):
    """Enumerate coordination arrangements for one ligand stereoisomer."""
    if center == "all":
        return _enumerate_all_centers(mol, geometry, fix, haptic_mode, stereo_ref, lengths, screen, observed_only)
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
    else:
        base_iso, donors, haptic = _prepare_spectators(mol, metals, center)
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
    if not donors and geometry is None:
        atom = mol.GetAtomWithIdx(m)
        raise ValueError(
            f"{atom.GetSymbol()}{m} has no donor bonds, so there is no coordination isomer to enumerate; "
            "use the Mol directly or pass geometry= to define a vacant coordination pocket"
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
                ref_sig=stereo_ref,
                lengths=lengths,
                source=source,
                graft_ref=graft_ref,
                screen_reach=screen,
                observed_only=observed_only,
            )
        )
    if mol.GetNumConformers():
        out = _haptic_mode(out, haptic_mode)
    elif haptic_mode != "free":
        out = _enumerate_haptic_windings(out)
    return out
