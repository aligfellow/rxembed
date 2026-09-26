"""Enumerate ligand, haptic and coordination states into ready-to-embed `Isomer` candidates."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem

from .bounds import coordination_reach_base, ligand_reach
from .constraints import Constraints, compose, resolve_core
from .metal_constraints import compile_context
from .metal_core import (
    VACANT,
    MetalState,
    canonical_metal_graph,
    collapse_haptic,
    frag_map,
    from_vertices,
    ligand_distance_matrix,
    logger,
    materialized_state,
    metal_indices,
    reject_boron_cages,
    reject_metal_bonds,
    state_with_winding,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_isomer import (
    Isomer,
    IsomerSet,
    canonical_metals,
    coordination_roles,
    length_source,
    measured_haptic_windings,
    primary_first,
    resolve_center,
    retained_geometry,
    retained_state,
    seat_order,
    winding_signature,
)
from .metal_perceive import shape_gap
from .metal_polyhedron import (
    MAX_EXHAUSTIVE_ORBITS,
    POLYHEDRA,
    SLOT_BOND_PROP,
    describe,
    geometries_for_cn,
    read_slot_notes,
    resolve_geometry,
)
from .metal_screen import narrow_span_pairs, unreachable_span
from .metal_slots import (
    SPAN_TOL,
    assignment_cap_error,
    chelate_edge_links,
    distinct_vertex_orderings,
    has_tether,
    input_ordering,
)
from .metal_stereo import chelate_links, chirality_of, donor_classes, face_has_orientation, site_classes
from .stereo import (
    apply_point_stereo,
    assign_atrop_from_3d,
    axis_stereo,
    bond_stereo,
    clear_atrop,
    clear_ez,
    coordination_locked_centres,
    coordination_locked_double_bonds,
    defined_stereo_label,
    enumerate_unassigned,
    matches_stereo,
    point_centres,
    point_stereo,
    stereo_from_3d,
    unassigned_centres,
    without_bond_stereo,
)
from .utils import mirror_tag

_MULTI_METAL = 2
_SCREEN_SITES = 8  # Keep the finite raw pool bounded before applying whole-network reach screening.


def _screen_limit(site_count, reach):
    """Bound the raw tethered pool before applying the normal limit to surviving candidates."""
    if reach is not None and site_count <= _SCREEN_SITES:
        return math.factorial(site_count)
    return None


def _keep_screened_candidate(out, candidate, geometry):
    """Append one surviving screened candidate without exceeding the public orbit bound."""
    out.append(candidate)
    if len(out) > MAX_EXHAUSTIVE_ORBITS:
        raise assignment_cap_error(geometry)


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
        if "locked" in stereo:  # an atom selector still wins, as it does over "point"
            for index in coordination_locked_centres(mol, metal_indices(mol)):
                exact.setdefault(index, stereo["locked"])
    centres = point_centres(mol, exclude=metal_indices(mol))
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


def _keeps_reference_stereo(label, measured, measured_axes, exact, modes):
    """Return whether each element of `label` keeps its requested relation to the measured stereo."""
    for part in label.split(",") if label else ():
        point = re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|r|s|CW|CCW)", part)
        axis = re.fullmatch(r"[A-Z][a-z]?(\d+)-[A-Z][a-z]?(\d+):(M|P)", part)
        mode = exact.get(int(point.group(1)), modes["point"]) if point else modes["axial"] if axis else modes["ez"]
        if axis and tuple(sorted(map(int, axis.group(1, 2)))) not in measured_axes:
            continue  # an inferred axis has no reference hand to preserve or invert
        same = matches_stereo(measured, part)
        if mode == "preserve" and not same:
            return False
        if mode == "invert" and same:
            return False
    return True


def _geometry_stereo_variants(mol, variants, stereo, modes, exact):
    """Keep the requested ligand configurations relative to an input geometry."""
    measured = (
        stereo_from_3d(mol, exclude=metal_indices(mol))
        if mol.GetNumConformers()
        else defined_stereo_label(mol, exclude=metal_indices(mol))
    )
    measured_axes = axis_stereo(measured)
    kept = [
        (variant, label)
        for variant, label in variants
        if _keeps_reference_stereo(label, measured, measured_axes, exact, modes)
    ]
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
    locked_ez = coordination_locked_double_bonds(mol, metal_indices(mol))
    changed = False
    unstable_bonds = set()
    if has_geometry:
        measured = stereo_from_3d(mol, exclude=metal_indices(mol))
        measured_point_labels = point_stereo(measured)
        measured_points = set(measured_point_labels)
        stated_points = point_stereo(defined_stereo_label(mol, exclude=metal_indices(mol)))
        centres = point_centres(mol, exclude=metal_indices(mol))
        skip.update(
            index for index in centres if exact.get(index, point_mode) == "preserve" and index not in measured_points
        )
        axes = axis_stereo(measured)
        assign_atrop_from_3d(mol, axes)
        changed |= bool(axes)
        unstable_bonds = {
            frozenset(centre)
            for centre in unassigned_centres(mol, exclude=metal_indices(mol))
            # a bond stereocentre is two atoms
            if len(centre) == 2 and frozenset(centre) not in bond_stereo(measured)  # noqa: PLR2004
        }
    for index in clear:
        atom = mol.GetAtomWithIdx(index)
        changed |= atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if has_geometry:
        fixed = {index for index in centres & measured_points if exact.get(index, point_mode) in ("preserve", "invert")}
        rewrite = {index for index in fixed if stated_points.get(index) != measured_point_labels[index]}
        apply_point_stereo(mol, measured, rewrite)
        for index in fixed:
            if exact.get(index, point_mode) == "invert":
                atom = mol.GetAtomWithIdx(index)
                atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
        changed |= bool(rewrite) or any(exact.get(index, point_mode) == "invert" for index in fixed)
    skip_bonds = True if ez_mode == "free" else unstable_bonds
    if ez_mode in ("racemic", "invert"):
        changed |= clear_ez(mol)
    include_atrop = clear_atrop(mol) if axial_mode in ("racemic", "invert") else set()
    changed |= bool(include_atrop)
    variants, n_unassigned, _total, unresolved = enumerate_unassigned(
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
            ",".join(part for part in measured.split(",") if not set(point_stereo(part)) & skip)
            if has_geometry
            else defined_stereo_label(variant, exclude=metal_indices(variant))
        )
        variants = [(variant, label)]
    variants = [(variant, without_bond_stereo(label, locked_ez)) for variant, label in variants]
    modes = {"point": point_mode, "ez": ez_mode, "axial": axial_mode}
    variants = _geometry_stereo_variants(source, variants, stereo, modes, exact)
    return variants, _haptic_stereo_mode(stereo, has_geometry), n_unassigned, unresolved


def _prepare_spectators(mol, metals, center):
    """Return one surrogated base isomer while retaining every spectator sphere."""
    m = resolve_center(mol, metals, center)
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
    base, donors, haptic = collapse_haptic(base, donor_map[m])
    retained_states = {}
    spectator_phantoms = set()
    for s in spectators:
        before = base.GetNumAtoms()
        base, retained_states[s] = retained_state(mol, base, metal_info, metals, s)
        spectator_phantoms.update(range(before, base.GetNumAtoms()))
    base = strip_phantoms(base, spectator_phantoms)  # keep only the active centre's enumeration centroids
    logger.info(
        "metal: enumerating %s%d; retaining %d spectator sphere(s)",
        mol.GetAtomWithIdx(m).GetSymbol(),
        m,
        len(spectators),
    )
    centres = tuple(
        retained_states.get(metal, MetalState(metal, atomic_num, charge))
        for metal, atomic_num, charge in primary_first(metal_info, m)
    )
    base_iso = Isomer.from_state(base, centres, donor_bonds)
    return base_iso, donors, haptic


def _select_geometries(base, m, donors, haptic, geometry, n):
    """Resolve `geometry` to the list of polyhedron names to enumerate (default from donor count, or as given)."""
    measured = None
    if geometry is None:
        selected, measured, apical = retained_geometry(base, m, donors, haptic)
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


def _pinned_orders(frozen_v, free_v, free_di, sites):
    """Yield every vertex order that keeps each frozen donor at its vertex and permutes the free donors."""
    for fp in itertools.permutations(free_di):
        order = [None] * sites
        for vertex, donor_index in frozen_v.items():
            order[vertex] = donor_index
        for vertex, donor_index in zip(free_v, fp, strict=False):
            order[vertex] = donor_index
        yield order


def _frozen_permutations(base, m, padded, geom, frozen_donors, haptic=None, coordination=()):
    """Generate every free-donor vertex permutation with each frozen donor pinned at its input vertex.

    Returns a lazy permutation iterator, bypassing the symmetry-reduced `isomer_permutations`, which would
    miss the representative ordering a valid frozen isomer needs; or ``None`` if the input ordering can't be
    read.
    """
    base_order = input_ordering(base, m, padded, geom, haptic, coordination)
    if base_order is None:
        return None
    sites = len(padded)
    frozen_v = {v: di for v, di in enumerate(base_order) if padded[di] in frozen_donors}
    free_v = [v for v in range(sites) if v not in frozen_v]
    free_di = [di for di in range(len(padded)) if padded[di] not in frozen_donors]
    count = math.factorial(len(free_di))
    if count > MAX_EXHAUSTIVE_ORBITS:
        raise ValueError(
            f"metal[{geom}]: fix= leaves exactly {count:,} free-site arrangements to scan "
            f"(limit {MAX_EXHAUSTIVE_ORBITS:,}); retain more donor positions or retain the "
            "coordinate arrangement with rx.embed(mol)"
        )
    logger.info(
        "metal[%s]: %d frozen donors; deduplicating %d free-site arrangements",
        geom,
        len(frozen_v),
        count,
    )
    return _pinned_orders(frozen_v, free_v, free_di, sites)


@dataclass(frozen=True)
class _Request:
    """Carry one `enumerate_isomers` call's options, plus the source and fix each molecule resolves once."""

    geometry: object
    center: object
    fix: object
    stereo_ref: object
    lengths: str
    screen: bool
    observed_only: bool
    haptic_mode: str
    source: Chem.Mol | None = None  # the real-atom graph every candidate shares
    fix_cons: Constraints | None = None
    graft_ref: dict | None = None


def _witnessed_retention(retained, observed_only, base, source, donors, haptic, native_reach):
    """Return `retained`, or None where it is not a witness to its own arrangement.

    The input proves its own arrangement only by realising it: `observed_only` names it explicitly, and
    otherwise every measured same-ligand donor span must fit within the native reach the screen itself reads.
    A hand-placed conformer can put donors where the ligand cannot reach, and then proves nothing, so only a
    witness may exempt its arrangement from the reach screen below.
    """
    if retained is None or observed_only or native_reach is None:
        return retained
    at = source.GetConformer().GetPositions()
    real = sorted(d for d in donors if d not in haptic)
    frag = frag_map(base)
    witnessed = not any(
        frag[a] == frag[b] and np.linalg.norm(at[a] - at[b]) > float(native_reach[a, b]) + SPAN_TOL
        for a, b in itertools.combinations(real, 2)
    )
    return retained if witnessed else None


def _isomers_for_geometry(request, base_iso, geom, donors, haptic):
    """Enumerate every distinct `Isomer` of one polyhedron `geom` (frozen core held, spectators retained)."""
    base, m = base_iso.graph, base_iso.metal
    source, fix_cons = request.source, request.fix_cons
    frozen_donors = fix_cons.frozen & set(donors)
    n = len(donors)
    sites = POLYHEDRA[geom].cn
    if n > sites:
        raise ValueError(f"{geom} has {sites} coordination sites but the metal has {n} donors")
    padded = list(donors) + [VACANT] * (sites - n)  # leave empty vertices as a pocket
    if sites - n:
        logger.info(
            "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)", geom, sites, n, sites - n
        )
    roles = base_iso.roles
    perms = None
    # Pin each frozen donor at its input vertex, then generate every free-donor arrangement.
    if frozen_donors and sites == n:
        perms = _frozen_permutations(base, m, padded, geom, frozen_donors, haptic, roles)
    real_donors = base_iso.donors
    tethered = has_tether(padded, frag_map(base), haptic)
    classes = site_classes(base, padded, haptic, roles)
    distances = ligand_distance_matrix(base) if tethered else None
    retained = (
        input_ordering(base, m, padded, geom, haptic, roles, classes=classes)
        if base.GetNumConformers() and shape_gap(base, m, donors, haptic, geom)[3]
        else None
    )
    if request.observed_only:
        if retained is None:
            raise ValueError(
                "observed_only=True requires an input conformer whose donor arrangement matches the requested "
                f"{geom} geometry"
            )
        # The measured order is an explicit user choice, not an approximation to exhaustive enumeration.
        # Passing it as the permutation stream avoids constructing an oversized constitutional orbit pool.
        perms = (tuple(retained),)
    reach = None
    if tethered and request.screen and not fix_cons.constrained_atoms():
        try:
            reach = ligand_reach(base)
        except (ValueError, RuntimeError):
            logger.debug("metal[%s]: native ligand reach unavailable; retaining all arrangements", geom)
    native_reach = (
        coordination_reach_base(source, reach, {m}) if reach is not None and len(base_iso.centres) == 1 else None
    )
    screen_context = compile_context(source) if native_reach is not None else None
    if screen_context is not None:
        screen_context["native_reach"] = native_reach  # metal_constraints.seated_bites' own fact, seeded once
    retained = _witnessed_retention(retained, request.observed_only, base, source, donors, haptic, native_reach)
    # Prune same-ligand pairs the compiled screen would reject wholesale before the streamed tethered pool
    # (metal_slots.distinct_vertex_orderings' uncapped else-branch) can raise its resource cap on them.
    narrow = (
        narrow_span_pairs(base_iso, source, padded, geom, haptic, request.lengths, native_reach, screen_context)
        if native_reach is not None
        else {}
    )
    # A same-ligand chelate-backbone or direct-bond pair must sit on a polyhedron hull edge (a chemistry
    # claim; see metal_slots.chelate_edge_links). Same gate as `narrow`: a single screened centre with no fix=.
    linked = chelate_edge_links(base, padded, haptic, distances) if native_reach is not None else frozenset()
    out, unreachable = [], 0
    raw_limit = _screen_limit(n, reach)
    # Reading a ring's winding depends on the input geometry, not on which vertex a donor lands at,
    # so it is the same for every candidate below.
    winding = measured_haptic_windings(source, m, real_donors, haptic)
    for order in distinct_vertex_orderings(
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
        links = chelate_links(base, od, haptic, distances) if tethered else ()
        hand = chirality_of(base, geom, od, haptic, roles, classes=classes, links=links)
        active = from_vertices(m, base_iso.real_z, base_iso.real_q, geom, od, haptic, winding.items(), hand)
        centres = (active, *(centre for centre in base_iso.centres if centre.atom != m))
        candidate = Isomer.from_state(
            source,
            centres,
            base_iso.donor_bonds,
            constraints=fix_cons,
            constrained_metals={state.atom for state in centres if state.geometry},
            lengths=request.lengths,
            graft_ref=request.graft_ref,
            stereo_ref=request.stereo_ref,
            shared_mol=True,
        )
        if reach is not None and (retained is None or tuple(order) != tuple(retained)):
            failure = unreachable_span(candidate, reach, classes, links, native_reach, screen_context)
            if failure:
                logger.debug("metal[%s]: omit arrangement: %s", geom, failure)
                unreachable += 1
                continue
        _keep_screened_candidate(out, candidate, geom)
    if unreachable:
        logger.info(
            "metal[%s]: omitted %d arrangements the ligands cannot reach (screen=False keeps them)", geom, unreachable
        )
    if not out and (unreachable or narrow or linked):
        # narrow/linked can prune every streamed order (e.g. a linear pocket has no hull edge at all), which
        # never reaches the per-candidate screen below and so never increments `unreachable` either.
        logger.warning("metal[%s]: no model-compatible arrangement; try lengths='input', fix= or screen=False", geom)
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

    The reading half of `metal_smiles.cxsmiles`. It lives here, not there, because it reads RDKit atom
    properties on a `Mol`, an arrangement, which is this module's subject rather than a string. By the time a
    `Mol` carries an arrangement, `metal_smiles.parse_smiles` has already kept the block's indices addressing
    the atoms they were written for, so `enumerate_isomers` can seat it directly instead of enumerating.
    ``None`` means there is nothing to seat: a plain SMILES.

    A vacant vertex has no donor, so a coordination pocket survives the round trip.
    """
    # One key for both notes, so the VALUE says which it is: a slot is `s<n>` with an optional winding sign
    # (`metal_polyhedron` owns that grammar, since it owns the slots), and anything else on a noted atom is
    # the metal's geometry code.
    noted = {a.GetIdx(): a.GetProp("atomNote") for a in mol.GetAtoms() if a.HasProp("atomNote")}
    geom = [i for i in noted if i in metal_indices(mol)]
    if center is None and len(geom) != 1:
        return None
    metal = geom[0] if center is None else resolve_center(mol, metal_indices(mol), center)
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
        sites.setdefault(slot, atom_idx)  # a haptic face writes one slot on every ring atom; the first seen names it
    if name not in POLYHEDRA:
        raise ValueError(f"the arrangement on this string names {name!r}, which is not a polyhedron rxembed has")
    return name, sites, chirality, windings


def _stated_windings(iso, windings):
    """Return validated haptic winding signs read from canonical slots."""
    stated = iso.haptic_winding
    if not windings:
        return stated
    ranks = donor_classes(iso.graph, iso.donors)
    for slot, winding in windings.items():
        donor = iso.vertices[slot]
        face = iso.haptic.get(donor)
        if face is None:
            raise ValueError(f"slot s{slot}{winding} states haptic winding, but that slot is not a haptic face")
        if not face_has_orientation(iso.graph, face, ranks):
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


def from_stated_arrangements(mol, center, lengths):
    """Seat every CX-noted metal sphere and return the selected centre as one composable `Isomer`."""
    metals = metal_indices(mol)
    selected = resolve_center(mol, metals, center)
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
    roles = coordination_roles(donor_bonds, metal_info)
    states, phantoms = [], set()
    for m in metals:
        name, sites, chirality, windings = stated[m]
        base, vertices, haptic = collapse_haptic(base, donors[m])
        count = POLYHEDRA[name].cn
        if len(vertices) > count:
            raise ValueError(f"{name} has {count} sites but metal {m} has {len(vertices)} donor sites")
        padded = list(vertices) + [VACANT] * (count - len(vertices))
        order = seat_order(padded, haptic, sites)
        seated = [padded[k] for k in order]
        derived = chirality_of(base, name, seated, haptic, roles)
        state = from_vertices(m, *info[m], name, seated, haptic, hand=derived)
        identities = primary_first(metal_info, m)
        centres = tuple(state if metal == m else MetalState(metal, z, q) for metal, z, q in identities)
        iso = Isomer.from_state(base, centres, donor_bonds)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        materialized = iso.vertices
        states.append(state_with_winding(state, materialized, winding)._replace(hand=chirality))
        phantoms.update(haptic)

    identities = primary_first(metal_info, selected)
    ordered = tuple(next(state for state in states if state.atom == metal) for metal, _z, _q in identities)
    source = strip_phantoms(base, phantoms)
    lengths = length_source(source, lengths)
    return Isomer.from_state(
        source,
        ordered,
        donor_bonds,
        constrained_metals={state.atom for state in ordered},
        lengths=lengths,
    )


def _resolve_fix(graph, fix, has_geometry):
    """Return `fix` as constraints and a graft frame.

    A list fix freezes atoms that graft from the retained source conformer, so no external frame owns them.
    """
    if not fix:
        return Constraints(), {}
    cons, graft_ref = resolve_core(graph, fix=fix, has_geometry=has_geometry)
    return cons, graft_ref if isinstance(fix, dict) else {}


def _with_fix(iso, fix):
    """Compose a geometric fix onto an already stated coordination identity."""
    if not fix:
        return iso
    cons, graft_ref = _resolve_fix(iso.graph, fix, bool(iso.graph.GetNumConformers()))
    return Isomer.from_state(
        iso.graph,
        iso.centres,
        iso.donor_bonds,
        constraints=compose(iso.base_cons, cons),
        constrained_metals=iso.constrained_metals,
        lengths=iso.lengths,
        graft_ref=graft_ref,
        stereo_ref=iso.stereo_ref,
        stereo_label=iso.stereo_label,
        protect_arrangement=iso.protect_arrangement,
    )


def _winding_variants(iso, state):
    """Return the symmetry-distinct assignments of one state's undefined haptic windings."""
    vertices, haptic, stored, donors = materialized_state(iso, state)
    ranks = donor_classes(iso.graph, donors)
    faces = [
        dummy for dummy, face in haptic.items() if dummy not in stored and face_has_orientation(iso.graph, face, ranks)
    ]
    if not faces:
        return [state]
    variants, seen = [], set()
    for signs in itertools.product("+-", repeat=len(faces)):
        winding = stored | dict(zip(faces, signs, strict=True))
        signature = winding_signature(iso, state, winding)
        if signature not in seen:
            seen.add(signature)
            variants.append(state_with_winding(state, vertices, winding))
    return variants


def _haptic_mode(isomers, mode):
    """Apply the requested mode to haptic orientation read from an input geometry."""
    if mode in ("free", "preserve"):
        return isomers
    out = IsomerSet()
    for iso in isomers:
        choices = []
        for state in iso.centres:
            vertices, _haptic, stored, _donors = materialized_state(iso, state)
            candidate = state
            if mode == "invert":
                winding = {face: "+" if sign == "-" else "-" for face, sign in stored.items()}
                candidate = state_with_winding(state, vertices, winding)
            elif mode != "unassigned":  # racemic / separate: discard the measured orientation first
                candidate = state_with_winding(state, vertices, {})
            choices.append(_winding_variants(iso, candidate))
        out.extend(iso.with_stereo(states) for states in itertools.product(*choices))
    return out


def _enumerate_all_centers(mol, request):
    """Stack independently enumerated metal states into their Cartesian product."""
    metals = metal_indices(mol)
    if len(metals) < _MULTI_METAL:
        raise ValueError("center='all' needs at least two transition-metal centres")
    stated = stated_arrangement(mol, center=metals[0])
    metals = canonical_metals(mol, metals, allow_ties=stated is not None)
    if stated is not None:
        if request.geometry is not None:
            raise ValueError(
                f"this input already states every metal arrangement, so geometry={request.geometry!r} has nothing "
                "to act on"
            )
        iso = _with_fix(from_stated_arrangements(mol, metals[0], request.lengths), request.fix)
        return _haptic_mode(IsomerSet([iso]), request.haptic_mode)
    if mol.GetNumConformers() == 0:
        raise ValueError("center='all' needs an input geometry to infer one polyhedron per metal centre")
    if request.geometry is not None:
        raise ValueError("center='all' infers each centre's geometry; omit the single geometry= argument")

    base, metal_info = surrogate_all_metals(mol)
    sphere, donor_bonds, phantoms = {}, [], set()
    metal_set = set(metals)
    for m in metals:
        real = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metal_set]
        donor_bonds.extend((d, m) for d in real)
        base, donors, haptic = collapse_haptic(base, real)
        sphere[m] = (donors, haptic)
        phantoms.update(haptic)
    real_base = strip_phantoms(Chem.Mol(base), phantoms)
    lengths = length_source(real_base, request.lengths)
    fix_cons, graft_ref = _resolve_fix(base, request.fix, True)
    # Each centre holds only the frozen core and skips the reach screen under a fix; the product carries the fix.
    per_centre = replace(
        request,
        source=real_base,
        lengths=lengths,
        fix_cons=Constraints(frozen=set(fix_cons.frozen)),
        graft_ref=graft_ref,
        screen=request.screen and not bool(request.fix),
    )
    choices = []
    for m in metals:
        donors, haptic = sphere[m]
        identities = primary_first(metal_info, m)
        base_iso = Isomer.from_state(base, tuple(MetalState(*item) for item in identities), donor_bonds)
        candidates = IsomerSet()
        for geom in _select_geometries(base, m, donors, haptic, None, len(donors)):
            candidates.extend(_isomers_for_geometry(per_centre, base_iso, geom, donors, haptic))
        choices.append(_haptic_mode(candidates, request.haptic_mode))

    out = IsomerSet()
    for selected in itertools.product(*choices):
        centres = tuple(candidate.centres[0] for candidate in selected)
        out.append(
            Isomer.from_state(
                real_base,
                centres,
                donor_bonds,
                constraints=fix_cons,
                constrained_metals={state.atom for state in centres},
                lengths=lengths,
                graft_ref=graft_ref,
                stereo_ref=request.stereo_ref,
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
    `lengths='model'` uses model M-donor distances; ``lengths='input'`` measures them from coordinates instead.

    Parameters
    ----------
    screen : bool, optional
        Apply the native ligand-reach and donor-facing screens, by default True. False retains distinct
        assignments beyond those model limits, with no guarantee they embed. Symmetry, stated slots, ligand
        stereo, explicit fixes and embedding validation are unchanged.
    observed_only : bool, optional
        With coordinates, return only the measured donor arrangement. An explicit resource-bounded choice for
        high-coordinate inputs, not a claim that unmeasured assignments are impossible.
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
    reject_metal_bonds(mol)
    reject_boron_cages(mol)
    mol = canonical_metal_graph(mol)
    center, stereo = _source_defaults(mol, center, stereo)
    variants, haptic_mode, n_unassigned, unresolved = _ligand_stereo_variants(mol, stereo)
    request = _Request(geometry, center, fix, stereo_ref, lengths, screen, observed_only, haptic_mode)
    out = IsomerSet()
    for variant, stated_label in variants:
        built = _enumerate_coordination(variant, request)
        defined = defined_stereo_label(variant, exclude=metal_indices(variant))
        labels = {}
        for item in (*defined.split(","), *stated_label.split(",")):
            if item:
                labels[item.rsplit(":", 1)[0]] = item
        label = ",".join(labels.values())
        locked = coordination_locked_double_bonds(variant, metal_indices(variant))
        label = without_bond_stereo(label, locked)
        point = point_stereo(stated_label)
        tag_source = None
        if point:
            tag_source = Chem.Mol(variant)
            tag_source.RemoveAllConformers()
            tag_source, _metal_info = surrogate_all_metals(tag_source)
        if tag_source is not None and built:
            shared = Chem.Mol(built[0].graph)
            for donor in point:
                shared.GetAtomWithIdx(donor).SetChiralTag(tag_source.GetAtomWithIdx(donor).GetChiralTag())
            for iso in built:
                iso.graph = shared
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


def _enumerate_coordination(mol, request):
    """Enumerate coordination arrangements for one ligand stereoisomer."""
    center, geometry = request.center, request.geometry
    if center == "all":
        return _enumerate_all_centers(mol, request)
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
            iso = _with_fix(from_stated_arrangements(mol, center, request.lengths), request.fix)
            return _haptic_mode(IsomerSet([iso]), request.haptic_mode)
        iso = Isomer(mol, name, sites, lengths=request.lengths)
        _validate_stated_chirality(iso, chirality)
        winding = _stated_windings(iso, windings)
        vertices = iso.vertices
        state = iso.centres[0]
        iso = iso.with_stereo((state_with_winding(state, vertices, winding)._replace(hand=chirality),))
        iso = _with_fix(iso, request.fix)
        return _haptic_mode(IsomerSet([iso]), request.haptic_mode)
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no transition metal found")
    # One metal means no spectators, so this is the single-centre path whether or not center= names it. An
    # explicit center= is still validated against the sole metal, raising on a wrong index or element.
    if len(metals) == 1:
        if center is not None:
            resolve_center(mol, metals, center)
        base, m, real_donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = collapse_haptic(base, real_donors)
        base_iso = Isomer.from_state(base, (MetalState(m, real_z, real_q),), [(donor, m) for donor in real_donors])
    else:
        base_iso, donors, haptic = _prepare_spectators(mol, metals, center)
        base, m = base_iso.graph, base_iso.metal
    source = strip_phantoms(Chem.Mol(base), set(haptic))
    lengths = length_source(source, request.lengths)  # once per molecule, not per ordering
    # Hold a reacting TS core while the rest of the coordination sphere is enumerated: coordinate forms need
    # an input geometry, while numeric distances and angles are complete on a coordinate-free graph.
    fix_cons, graft_ref = _resolve_fix(base, request.fix, base.GetNumConformers() > 0)
    if request.fix:
        logger.info(
            "metal: fixed %d input atoms; enumerating free coordination sites",
            len(fix_cons.constrained_atoms()),
        )
    request = replace(request, source=source, lengths=lengths, fix_cons=fix_cons, graft_ref=graft_ref)
    if not donors and geometry is None:
        atom = mol.GetAtomWithIdx(m)
        raise ValueError(
            f"{atom.GetSymbol()}{m} has no donor bonds, so there is no coordination isomer to enumerate; "
            "use the Mol directly or pass geometry= to define a vacant coordination pocket"
        )
    geoms = _select_geometries(base, m, donors, haptic, geometry, len(donors))
    out = IsomerSet()
    for geom in geoms:
        out.extend(_isomers_for_geometry(request, base_iso, geom, donors, haptic))
    if mol.GetNumConformers():
        out = _haptic_mode(out, request.haptic_mode)
    elif request.haptic_mode != "free":
        out = IsomerSet(
            iso.with_stereo((state, *iso.centres[1:]))
            for iso in out
            for state in _winding_variants(iso, iso.centres[0])
        )
    return out
