"""An `Isomer` -> the `Constraints` that hold its coordination sphere: M-donor windows and L-M-L angles.

The top of the metal stack: it reads `metal_distance` and `metal_donor_orient`, and nothing reads it back.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
from rdkit import Chem

from . import bounds as _bounds
from .constraints import Constraints, _graft_owns, add_distance, compose
from .mechanisms import _triangle_angles
from .metal_core import (
    _ETA2,
    VACANT,
    _bounds_matrix,
    _frag_map,
    _haptic_sites,
    _ligand_graph,
    _plane_rms,
    _regular_face,
    _site_height,
    _site_radius,
    _site_span,
    _vertex_atom,
    logger,
    materialized_states,
    metal_indices,
    rank_shapes,
    shape_reading,
)
from .metal_distance import _INPUT_HALF_WIDTH, delocalised_charges, ff_terms, ml_distance
from .metal_donor_orient import (
    _COPLANAR_CAP,
    _coplanar_donor,
    _orient_donor,
    _ring_hinge,
    _stripped_hybridisation,
)
from .metal_polyhedron import (
    _IMPROPER_VERTICES,
    CHELATE_SPAN_ANGLE,
    POLYHEDRA,
    _improper,
    _vertex_angle,
    record,
    relaxed_shell,
    seating_is_exhaustive,
)
from .metal_slots import _SPAN_TOL, _chelate_bite_window
from .stereo import bond_stereo, metal_referenced_ez, point_stereo

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval
_TRIGONAL_CARRIERS = 3
_TETRAHEDRAL_CARRIERS = 4
_BOND_STEREO_ATOMS = 2
_STRAIGHT = 180.0
_SHELL_ATOL = 1e-6  # numerical tolerance for template coplanarity and sector closure
# Slack on an otherwise-unconstrained compiled L-M-L angle row: wide enough for ordinary distortion, narrow
# enough that metal_enumeration._narrow_span_pairs can prove a pair's whole orbit will fail this angle wall.
_ANGLE_PAD = 8.0


def _centroid_constraints(mol, metal, dummy, ring, real_z, *, qdel, c, pos, hyb):
    """Constrain one haptic face through a transient centroid vertex.

    A regular face adds radius priors to the DG cone; irregular faces keep only metal-member distances.
    During UFF, Haptic replaces these radius priors with a moving-centroid penalty, without flattening the face.
    """
    r = _site_radius(mol, ring, positions=pos)
    if pos is not None:  # realised metal->ring-atom distances from the input geometry
        mcs = [float(np.linalg.norm(pos[metal] - pos[a])) for a in ring]
    else:  # the fitted model; the ring is its own co-donor set so the hapticity term sees the whole face
        mcs = [ml_distance(mol, metal, a, real_z, set(ring), charges=qdel, hyb=hyb) for a in ring]
    d_c = _site_height(r, mcs)
    add_distance(  # M -> centroid vertex, both modes
        c.distances,
        metal,
        dummy,
        d_c - (_INPUT_HALF_WIDTH if pos is not None else _ML_SEED_HALF_WIDTH),
        d_c + (_INPUT_HALF_WIDTH if pos is not None else _ML_SEED_HALF_WIDTH),
    )
    c.pulls[(min(metal, dummy), max(metal, dummy))] = d_c
    cone = _regular_face(mol, ring)
    for a, mca in zip(ring, mcs, strict=True):
        add_distance(
            c.distances, metal, a, mca - _ML_SEED_HALF_WIDTH, mca + _ML_SEED_HALF_WIDTH
        )  # hold each ring atom at its metal distance
        if cone:
            add_distance(c.distances, dummy, a, r - 0.1, r + 0.1)
    c.phantoms = c.phantoms | {dummy}
    c.haptic[dummy] = tuple(ring)  # embed scaffolding: materialised transiently in the DG/UFF, stored in no real Mol


LENGTHS = ("model", "input")


def compile_context(mol):
    """Return RDKit-derived data shared by repeated candidate constraint compilations."""
    return {
        "fragments": _frag_map(mol),
        "hybridisation": _stripped_hybridisation(mol),
        "charges": delocalised_charges(mol),
        "bounds": _bounds_matrix(mol),
        "topology": Chem.GetDistanceMatrix(mol),
        "stripped": _ligand_graph(mol),
    }


def _fact(context, key, compute):
    """Return a molecule fact, computing it once into the shared context."""
    if key not in context:
        context[key] = compute()
    return context[key]


def resolve_lengths(mol, lengths="model"):
    """Use model M-L distances unless input measurements are explicitly requested."""
    if lengths not in LENGTHS:
        raise ValueError(f"lengths={lengths!r}; expected one of {LENGTHS}")
    if lengths == "model":
        return None, ""
    if not mol.GetNumConformers():
        raise ValueError("lengths='input' requires a conformer; a SMILES carries no geometry; use lengths='model'")
    return mol.GetConformer().GetPositions(), "the input conformer (lengths='input')"


def compile_constraints(
    mol,
    centres,
    *,
    length_mol,
    base,
    constrained_metals,
    lengths,
    stereo_label,
    donor_bonds,
    external=(),
    donor_orientation=True,
    conjugation=True,
    force_field=True,
    context=None,
):
    """Compile selected metal states onto a copy of `base`.

    `mol` supplies current topology and atom indexing; `length_mol` preserves the input geometry used for lengths.
    `external` informs model independence without merging or bypassing the caller's later validation.
    `force_field=False` keeps the same coordination
    targets for a reach screen while omitting derived contact walls; it is not an alternative embedding model.
    """
    context = {} if context is None else context
    base = base.copy(donor_orientation=donor_orientation, conjugation=conjugation)
    built = []
    if constrained_metals:
        parts = materialized_states(mol, centres)
        held = base.constrained_atoms().union(*(cons.constrained_atoms() for cons in external if cons is not None))
        for state in centres:
            if state.atom not in constrained_metals:
                continue
            vertices, haptic, _winding, _donors = parts[state.atom]
            built.append(
                coordination(
                    mol,
                    state.atom,
                    vertices,
                    state.geometry,
                    state.atomic_num,
                    haptic=haptic,
                    frozen=base.frozen,
                    lengths=lengths,
                    source=length_mol,
                    distance_overrides=base.distances,
                    coupled_atoms=held | {donor for donor, other in donor_bonds if other != state.atom},
                    donor_orientation=donor_orientation,
                    force_field=force_field,
                    context=context,
                )
            )
    out = compose(*built, base)
    _add_donor_angle_floors(out, mol, context.get("bounds"))
    _add_point_umbrellas(out, mol, stereo_label, donor_bonds)
    _add_ligand_ez(out, mol, stereo_label, donor_bonds)
    _add_metal_ez(out, mol, stereo_label, donor_bonds)
    return out


def _add_native_pair_floors(cons, mol, pairs, bounds=None):
    """Keep unowned nonbonded pairs above RDKit's native lower bounds during restrained UFF."""
    pairs = {
        tuple(sorted(pair))
        for pair in pairs
        if mol.GetBondBetweenAtoms(*pair) is None
        and tuple(sorted(pair)) not in cons.distances
        and not _graft_owns(pair, cons.frozen)
    }
    if not pairs:
        return
    bounds = _bounds_matrix(mol) if bounds is None else bounds
    for left, right in pairs:
        cons.floors[(left, right)] = max(cons.floors.get((left, right), 0.0), float(bounds[right, left]))


def _add_donor_angle_floors(cons, mol, bounds=None):
    """Keep constrained donor substituents inside RDKit's native 1-3 lower bounds during UFF."""
    pairs = set()
    for metal, donor, left, right, anchor, _cap in cons.coplanar:
        pair = tuple(sorted((left, right)))
        if (
            anchor != _STRAIGHT
            or metal not in cons.metals
            or any(mol.GetBondBetweenAtoms(donor, i) is None for i in pair)
        ):
            continue
        pairs.add((donor, *pair))
    substituents = {}
    for metal, donor, substituent in cons.angles:
        if metal in cons.metals and mol.GetBondBetweenAtoms(donor, substituent) is not None:
            substituents.setdefault((metal, donor), set()).add(substituent)
    pairs.update(
        (donor, *sorted(pair))
        for (_metal, donor), members in substituents.items()
        for pair in itertools.combinations(members, 2)
    )
    pairs = {
        (left, right)
        for donor, left, right in pairs
        if (left, donor, right) not in cons.angles and (right, donor, left) not in cons.angles
    }
    _add_native_pair_floors(cons, mol, pairs, bounds)


def _add_point_umbrellas(cons, mol, stereo_label, donor_bonds):
    """Keep each retained tetrahedral point in its seeded signed-volume half-space during UFF."""
    metals = {}
    for donor, metal in donor_bonds:
        metals.setdefault(donor, []).append(metal)
    for centre in point_stereo(stereo_label):
        carriers = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(centre).GetNeighbors()]
        carriers.extend(metal for metal in metals.get(centre, ()) if metal not in carriers)
        key = (
            tuple(carriers)
            if len(carriers) == _TETRAHEDRAL_CARRIERS
            else (*carriers, centre)
            if len(carriers) == _TRIGONAL_CARRIERS
            else None
        )
        if key is not None and not _graft_owns(key, cons.frozen):
            cons.umbrellas.setdefault(key, 0.0)


def _add_ligand_ez(cons, mol, stereo_label, donor_bonds):
    """Keep stated ligand double bonds inside their RDKit-reference half-space during UFF."""
    metal_owned = set(metal_referenced_ez(mol, stereo_label, donor_bonds))
    for pair in bond_stereo(stereo_label):
        if pair in metal_owned:
            continue
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None or len(bond.GetStereoAtoms()) != _BOND_STEREO_ATOMS:
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        refs = bond.GetStereoAtoms()
        tag = bond.GetStereo()
        if tag in {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}:
            anchor = 0.0
        elif tag in {Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE}:
            anchor = _STRAIGHT
        else:
            continue
        row = (refs[0], begin, end, refs[1], anchor, _COPLANAR_CAP)
        if not _graft_owns(row[:4], cons.frozen):
            cons.coplanar.append(row)


def _add_metal_ez(cons, mol, stereo_label, donor_bonds):
    """Choose the donor-plane well when a metal is an imine's missing E/Z reference."""
    for donor, other, metal, ref, ligand_ref, wanted in metal_referenced_ez(mol, stereo_label, donor_bonds).values():
        anchor = 0.0 if wanted == "Z" else _STRAIGHT
        existing = [row for row in cons.coplanar if row[:3] == (metal, donor, other) and row[4] is None]
        cap = existing[0][5] if existing else _COPLANAR_CAP
        if existing:
            cons.coplanar.remove(existing[0])
        metal_row = (metal, donor, other, ref, anchor, cap)
        if not _graft_owns(metal_row[:4], cons.frozen):
            cons.coplanar.append(metal_row)
        if ligand_ref is not None:
            ligand_row = (ligand_ref, donor, other, ref, _STRAIGHT - anchor, cap)
            if not _graft_owns(ligand_row[:4], cons.frozen):
                cons.coplanar.append(ligand_row)


_EXHAUSTIVE_BITE_CORNERS = 4  # ponytail: 2**k corners is a resource ceiling above this many simultaneous bites;
# widen only if a real structure needs more than the extremes-plus-single-flips fallback below.


def _bite_reach(left, right, span):
    """Return the donor-donor angle range (degrees) two fixed model M-D legs support at a `span` (Å) range.

    `_triangle_angles` encloses angle by interval side lengths; a fixed leg is passed as a degenerate
    (x, x) interval. It can return ``None`` when the legs and span ranges admit no triangle at all (never
    observed at these near-point legs, but the fixed-leg law of cosines this replaces never rejected either,
    so fall back to it rather than lose the row).
    """
    angles = _triangle_angles((left, left), (right, right), span)
    if angles is None:

        def angle(d):
            cosine = (left * left + right * right - d * d) / (2.0 * left * right)
            return math.acos(max(-1.0, min(1.0, cosine)))

        angles = (angle(span[0]), angle(span[1]))
    return math.degrees(angles[0]), math.degrees(angles[1])


def _bite_window_and_reach(mol, left, right, donors, matrix, lengths, row):
    """Return (window, reach) for a same-ligand bite pair, one site each, or ``None`` where neither narrows it.

    `reach` is the triangle the two model M-D legs and the native ligand reach (`matrix`, a `_site_span`
    pair-interval matrix; ``None`` when `bounds.ligand_reach` could not close it for this graph) support
    (`_SPAN_TOL` numerical slack, `_bite_reach`'s law of cosines). A sigma pair (both sites one atom) keeps
    the ring-size census (`_CHELATE_BITE`, via `_chelate_bite_window`'s existence test) as its `window`,
    intersected with `reach`, or `reach` alone where the two are disjoint; with no census row it has no bite
    at all. A pair with a haptic end has no census row: lower bounds beyond 1-3 are free-ligand preferences
    that chelation overrides, so `reach` replaces `row` (the pair's own compiled polyhedron angle) outright
    wherever `reach` excludes part of it, and otherwise the row stands (no bite entry).
    """
    prior = None
    if len(left) == 1 and len(right) == 1:
        prior = _chelate_bite_window(mol, left[0], right[0], donors)
        if prior is None:
            return None
    if matrix is None:  # ligand_reach could not close a triangle for this graph
        return (prior, prior) if prior is not None else None
    left_leg, right_leg = map(float, lengths)
    lower, upper = _site_span(matrix, left, right)
    reach = _bite_reach(left_leg, right_leg, (max(0.0, lower - _SPAN_TOL), upper + _SPAN_TOL))
    if prior is not None:
        if reach[1] < prior[0] or reach[0] > prior[1]:
            return reach, reach
        return (max(prior[0], reach[0]), min(prior[1], reach[1])), reach
    if reach[0] <= row[0] and reach[1] >= row[1]:
        return None
    return reach, reach


def _seated_bites(mol, metal, od, haptic, frag_of, ideal_angles, distances, real_od, context, angle_rows=None):
    """Return each same-ligand donor pair's (window, reach) angle windows, keyed by its seated site pair.

    The one bite owner: `coordination`'s angle-row compilation and `metal_enumeration`'s chelate screens
    (`_chelate_span_failure`, `_unreachable_span`) all read these same windows, never a second copy of the
    bite condition or the relaxation it feeds (`metal_polyhedron.relaxed_shell`). `window` is the census wall
    `_bounded_bites` shrinks inside; `reach` is the backbone triangle it may open a gap box into instead. A
    haptic end joins on the same terms as a sigma donor: `_bite_window_and_reach` tells them apart by site
    size. `angle_rows` is `coordination`'s stated polyhedron subset; a pair missing from it falls back to
    `ideal_angles`, since a haptic pair has no census row of its own to compare `reach` against.
    """
    angle_rows = angle_rows or {}
    bites = {}
    for i, j in itertools.combinations(range(len(od)), 2):
        left, right = od[i], od[j]
        pair = frozenset((i, j))
        if VACANT in (left, right) or frag_of(left) != frag_of(right) or ideal_angles[pair] >= CHELATE_SPAN_ANGLE:
            continue
        keys = ((min(metal, left), max(metal, left)), (min(metal, right), max(metal, right)))
        if any(key not in distances for key in keys):  # the screen's sigma radial legs carry no haptic leg
            continue
        lengths = (0.5 * sum(distances[keys[0]]), 0.5 * sum(distances[keys[1]]))
        left_site, right_site = haptic.get(left, (left,)), haptic.get(right, (right,))
        base = angle_rows.get(pair, ideal_angles[pair])
        row = (max(0.0, base - _ANGLE_PAD), min(_STRAIGHT, base + _ANGLE_PAD))
        bite_key = ("bite", tuple(sorted(left_site)), tuple(sorted(right_site)), lengths, row)
        if bite_key not in context:
            try:
                native = _fact(
                    context,
                    "native_reach",
                    lambda: _bounds._coordination_reach_base(mol, _bounds.ligand_reach(mol), set(metal_indices(mol))),
                )
            except (ValueError, RuntimeError):
                logger.debug("metal[%s]: native ligand reach unavailable for a bite pair", metal)
                context["native_reach"] = native = None
            context[bite_key] = _bite_window_and_reach(mol, left_site, right_site, real_od, native, lengths, row)
        if (bite := context[bite_key]) is not None:
            bites[pair] = bite
    return bites


def _bite_corners(bites):
    """Yield each bite-target combination: every window corner for <= 4 bites, else the extremes plus flips."""
    keys = list(bites)
    windows = [bites[key] for key in keys]
    if len(keys) <= _EXHAUSTIVE_BITE_CORNERS:
        combos = set(itertools.product(*windows))
    else:
        low, high = tuple(window[0] for window in windows), tuple(window[1] for window in windows)
        combos = {low, high}
        for i in range(len(keys)):
            combos.add((*low[:i], high[i], *low[i + 1 :]))
            combos.add((*high[:i], low[i], *high[i + 1 :]))
    for combo in combos:
        yield dict(zip(keys, combo, strict=True))


_BITE_BOX_STEPS = 12  # bisection rounds shrinking a bite box toward its anchor; halves the gap each round


def _perceived_geometry(rays, radius, name):
    """Return whether `name` reads within `_FIT_MARGIN` of the best fit for rays scaled to `radius`.

    A synthetic witness carries no chemistry; only the ray directions and the M-L radius decide the reading,
    checked by the same rule B the acceptance gate applies to a real conformer (`metal_core.shape_reading`,
    `metal_isomer.retained_geometry`): `name` need not be the strict argmin, only within `_FIT_MARGIN` of it,
    so enumeration and acceptance agree on a tie. The radius is load-bearing, not decorative: `rank_shapes`'s
    flatness exclusion compares an absolute Å RMS (`metal_core.COPLANAR_TOL`) against a record's own ideal
    plane RMS at this witness's bond length. Built at unit length, an ideal seesaw's own plane RMS (0.2449)
    already undercuts that 0.25 Å tolerance, so ANY symmetric bite closure reads as the flatter square_planar
    regardless of the actual (much larger) model M-L distance (YEGNAA).
    """
    rms = _plane_rms(np.zeros(3), rays * radius)
    _, _, _, accepted = shape_reading(rank_shapes(rays, rms, radius), name)
    return accepted


def _bite_radius(od, bites, distances, metal):
    """Return the mean model M-L midpoint (Å) over every donor named in a bite pair.

    The scale `_bounded_bites`'s synthetic witness must be built at, so its flatness test reads against the
    sphere's real size instead of an arbitrary unit ray (YEGNAA). Shared by `coordination`'s own compiled
    windows and `metal_enumeration._chelate_span_failure`'s pre-compile screen, each keyed off its own
    donor-index/window pair.
    """
    donors = {od[i] for pair in bites for i in pair}
    return float(np.mean([0.5 * sum(distances[(min(metal, d), max(metal, d))]) for d in donors]))


def _bounded_bites(directions, name, ideal, bites, radius):
    """Shrink each of `bites`' census windows toward its ideal-clamped anchor until every corner reads as `name`.

    A same-ligand pair's model bite window (`_bite_window_and_reach`) is a ring-size prior, not a promise
    that every corner of the compiled box still perceives as the requested polyhedron: a macrocycle's
    tetrahedron can fold into another shape once several bites drift together (a joint effect `_bite_corners`
    already enumerates jointly, so the bound below checks it jointly too, not one bite at a time). `radius`
    (Å, from `_bite_radius`) is the scale the synthetic witness reads its flatness bound against; it widens
    no tolerance, it only makes the witness self-consistent with the model M-L distance. Returns the
    unmodified census windows when the full box already reads as `name`, or the narrowed windows when
    shrinking is needed.

    When even the census-clamped anchor does not read as `name`, this is not an immediate reject: it opens a
    gap box from that anchor out to `far`, the same ideal clamped into each pair's wider backbone-triangle
    `reach` instead. If the joint `far` corner reads as `name`, each pair's compiled row is the span between
    its anchor and its far point, un-shrunk (the flexible backbone, not the ring-size census, is trusted
    there). Returns ``None`` only when even that wider reach cannot support `name`: the backbone itself
    cannot hold the requested polyhedron, not merely the chelate's own ring-size bite prior.
    """
    if not bites:
        return bites
    if not _exact_reading(len(directions)):  # an approximate seating ranks shapes; it cannot narrow a wall
        return {pair: window for pair, (window, _reach) in bites.items()}
    key = tuple(
        sorted(
            ((pair, *bites[pair], ideal[pair]) for pair in bites),
            key=lambda item: tuple(sorted(item[0])),
        )
    )
    return _bounded_bites_cached(tuple(map(tuple, directions)), name, key, radius)


@lru_cache(maxsize=None)
def _exact_reading(cn):
    """Return whether every record of `cn` vertices is read by an exhaustive seating search.

    Above that bound `fit_residual` only ranks candidate shapes approximately, so the squeeze must leave the
    census window standing rather than narrow it against an approximate reading.
    """
    return all(seating_is_exhaustive(tuple(map(tuple, p.vertex_dirs))) for p in POLYHEDRA.values() if p.cn == cn)


def _corner_key(corner):
    """Return `corner`'s content key, in the same form `metal_polyhedron.relaxed_shell` builds internally."""
    return tuple(sorted((tuple(sorted(pair)), float(target)) for pair, target in corner.items()))


@lru_cache(maxsize=None)
def _reads_as_name_cached(directions, corner_key, name, radius):
    """Return whether `corner_key`'s bite targets relax to a shell that still reads as `name`.

    A same-content probe recurs often: every bisection round in `_bounded_bites_cached` re-tests corners
    that share most of their targets with the last round, and unrelated candidates frequently share a bite
    pair's exact model target too. `relaxed_shell` already memoises the relaxation itself on this same
    content key; this wraps `_perceived_geometry`'s acceptance check (never cached, until now recomputed on
    every probe including a `relaxed_shell` cache hit) in the same memo, not a second one.
    """
    bites = {frozenset(pair): target for pair, target in corner_key}
    rays = relaxed_shell(directions, bites)
    return rays is not None and _perceived_geometry(rays, radius, name)


@lru_cache(maxsize=None)
def _bounded_bites_cached(directions, name, key, radius):
    anchor = {pair: min(max(ideal, lo), hi) for pair, (lo, hi), _reach, ideal in key}
    windows = {pair: (lo, hi) for pair, (lo, hi), _reach, _ideal in key}
    far = {pair: min(max(ideal, rlo), rhi) for pair, _window, (rlo, rhi), ideal in key}

    def reads_as_name(corner):
        return _reads_as_name_cached(directions, _corner_key(corner), name, radius)

    if not reads_as_name(anchor):
        if reads_as_name(far):
            return {pair: (min(anchor[pair], far[pair]), max(anchor[pair], far[pair])) for pair in windows}
        return None
    if all(reads_as_name(corner) for corner in _bite_corners(windows)):
        return windows

    def box(t):
        return {
            pair: (lo + t * (anchor[pair] - lo), hi - t * (hi - anchor[pair])) for pair, (lo, hi) in windows.items()
        }

    good, bad = 1.0, 0.0  # t=1 is the anchor itself (degenerate, always reads correctly); t=0 the full window
    for _ in range(_BITE_BOX_STEPS):
        mid = 0.5 * (good + bad)
        if all(reads_as_name(corner) for corner in _bite_corners(box(mid))):
            good = mid
        else:
            bad = mid
    return box(good)


def coordination(  # noqa: C901 - compile each coordination term in one linear transaction
    mol,
    metal,
    vertices,
    geometry,
    real_z,
    *,
    haptic,
    frozen=(),
    lengths="model",
    source=None,
    distance_overrides=None,
    coupled_atoms=(),
    donor_orientation=True,
    force_field=True,
    context=None,
):
    """Build one metal state's distance, angle, floor and umbrella constraints.

    Cis chelates use calibrated ring-size bite windows. Disjoint tetrahedral bites and supported planar
    shells adjust their cross angles jointly; remaining pairs use polyhedron or complementary priors. Compile chemistry
    before `_drop_graft_owned` removes complete terms determined by `fix`. `source` preserves input-length
    measurements while `mol` supplies topology; `coupled_atoms` marks externally held fragments. A false
    `force_field` flag omits only derived nonbonded contact terms for enumeration screening.
    """
    context = {} if context is None else context
    frag = _fact(context, "fragments", lambda: _frag_map(mol))  # same ligand = same fragment
    poly = record(geometry)
    if poly is None:
        raise ValueError(
            f"no polyhedron template for {geometry!r}; add a POLYHEDRA row before embedding this coordination number"
        )

    def frag_of(v):  # a face tethered to a co-donor (η²-alkyne + alkyl of one metallacycle; an ansa Cp) must read
        return frag[_vertex_atom(haptic, v)]  # as one ligand: the bond-less centroid carries no fragment of its own

    pos, _note = resolve_lengths(source if source is not None else mol, lengths)
    distance_overrides = distance_overrides or {}
    c = Constraints()
    od = list(vertices)
    sigma_od = {x for x in od if x != VACANT and x not in haptic}  # single-point donors: the hinge's own ring set
    real_od = set(sigma_od)
    real_od.update(a for face in haptic.values() for a in face)  # real co-donors; centroid keys are not Mol atoms
    # Group the whole sphere into sigma/pi sites ONCE: `ml_distance` and `overbond_tier` (via `ff_terms`) each
    # used to regroup it per donor / per non-donor atom, an O(donors x non-donors) whole-molecule cost.
    haptic_sites = _haptic_sites(mol, real_od)
    eta = {d: len(site) for site in haptic_sites if len(site) >= _ETA2 for d in site}
    qdel = _fact(context, "charges", lambda: delocalised_charges(mol))
    # A Lewis charge is an artefact, so spread it.
    # The model uses ligand-only classes for either input bond convention. Input lengths do not need typing.
    hyb = _fact(context, "hybridisation", lambda: _stripped_hybridisation(mol))
    # `_orient_donor` rebuilds this per donor unless it is handed one already built; share it across the sphere.
    stripped = _fact(context, "stripped", lambda: _ligand_graph(mol)) if donor_orientation else None
    for d in od:
        if d == VACANT:
            continue
        if d in haptic:  # a centroid vertex: pin its whole ring, not a single donor (no orient/coplanar)
            _centroid_constraints(
                mol,
                metal,
                d,
                haptic[d],
                real_z,
                qdel=qdel,
                c=c,
                pos=pos,
                hyb=hyb,
            )
            continue
        if hyb is None:  # Measured lengths still need graph-derived donor orientation, shared across the sphere.
            hyb = _stripped_hybridisation(mol)
        key = (min(metal, d), max(metal, d))
        if key in distance_overrides:
            add_distance(c.distances, metal, d, *distance_overrides[key])
        else:
            add_distance(
                c.distances,
                metal,
                d,
                *_donor_distance_window(
                    mol, metal, d, real_z, real_od, positions=pos, charges=qdel, hyb=hyb, eta=eta.get(d, 0)
                ),
            )
        # Wall each donor substituent off the metal, the orientation hold a real energy cannot supply itself.
        # Length provenance is independent: measured M-L distances do not determine a partially free M-D-X axis.
        if donor_orientation:
            _orient_donor(mol, metal, d, real_od, c, hyb=hyb, stripped=stripped)
            # cap an sp2 donor's metal at the donor's own sp2 plane: the improper the stripped bond removed.
            _coplanar_donor(mol, metal, d, real_od, c, hyb=hyb)
    if donor_orientation:
        # A small, fully conjugated chelate ring hinges flat as a unit (metal_donor_orient._ring_hinge). Called
        # once per metal, after the per-donor loop above, since it walks donor PAIRS, not one donor at a time.
        _ring_hinge(mol, metal, sigma_od, c)
    pairs = list(itertools.combinations(range(len(od)), 2))
    angle_key = ("angles", geometry)
    if angle_key not in context:
        context[angle_key] = (
            {frozenset((i, j)): _vertex_angle(poly.vertex_dirs[i], poly.vertex_dirs[j]) for i, j in pairs},
            {frozenset((i, j)): a for i, j, a in poly.resolved_angles},
        )
    ideal_angles, angle_rows = (dict(values) for values in context[angle_key])
    stated_pairs = set(angle_rows)
    bond_bounds = None
    expanded = {}
    if pos is None:
        for i, j in pairs:
            left, right = od[i], od[j]
            if (
                VACANT in (left, right)
                or left in haptic
                or right in haptic
                or mol.GetBondBetweenAtoms(left, right) is None
            ):
                continue
            left_key = (min(left, metal), max(left, metal))
            right_key = (min(right, metal), max(right, metal))
            if left_key in distance_overrides or right_key in distance_overrides:
                continue
            bond_bounds = _fact(context, "bounds", lambda: _bounds_matrix(mol))
            left_window, right_window = c.distances[left_key], c.distances[right_key]
            angle = np.radians(ideal_angles[frozenset((i, j))])
            span = np.sqrt(
                left_window[1] ** 2 + right_window[1] ** 2 - 2 * left_window[1] * right_window[1] * np.cos(angle)
            )
            native_upper = float(bond_bounds[min(left, right)][max(left, right)])
            if native_upper > span > 0:
                scale = native_upper / span
                expanded[left_key] = max(expanded.get(left_key, 0.0), left_window[1] * scale)
                expanded[right_key] = max(expanded.get(right_key, 0.0), right_window[1] * scale)
    for key, upper in expanded.items():
        c.distances[key] = (c.distances[key][0], upper)
    bites = _seated_bites(mol, metal, od, haptic, frag_of, ideal_angles, c.distances, real_od, context, angle_rows)
    # A planar sphere needs every pair to stay in its plane. Sparse template rows are sufficient only at the
    # exact ideal; finite windows leave the omitted cis pairs free to pucker. Chelate bites remain graph-derived.
    if poly.planar:
        angle_rows.update({pair: angle for pair, angle in ideal_angles.items() if pair not in angle_rows})
    else:
        angle_rows.update({pair: ideal_angles[pair] for pair in bites if pair not in angle_rows})
    if metal not in frozen:  # fixed donors do not fix their angle about a free metal
        for i, j in itertools.combinations(range(len(od)), 2):
            if (
                VACANT not in (od[i], od[j])
                and frozenset((i, j)) not in stated_pairs
                and _graft_owns((od[i], od[j]), frozen, haptic)
            ):
                pair = frozenset((i, j))
                angle_rows.setdefault(pair, ideal_angles[pair])
    # A tethered chelate's bite pair takes its backbone-derived window outright, bounded so it cannot reach a
    # corner that would no longer read as this polyhedron (`_bounded_bites`); every other already-compiled row
    # only widens, by a union with the free, minimum-displacement consequence of rotating just the bite rays
    # to each bite-window corner (metal_polyhedron.relaxed_shell). A corner whose relaxed shell would change
    # the template's oriented type simply contributes no image; `metal_enumeration` rejects that arrangement
    # outright rather than silently keeping the plain ideal +- pad row here. Abstain only when the metal or a
    # bitten ligand's own atom is externally held: an unrelated frozen atom elsewhere leaves this free (an
    # opposed planar closure never depended on it either).
    blocked = set(frozen) | set(coupled_atoms)
    bite_fragments = {frag_of(od[i]) for pair in bites for i in pair}
    corner_images, mid_angles = [], None
    windows = {pair: window for pair, (window, _reach) in bites.items()}
    if bites and metal not in blocked and not bite_fragments & {frag[atom] for atom in blocked} and VACANT not in od:
        directions = poly.vertex_dirs
        radius = _bite_radius(od, bites, c.distances, metal)
        bites = _bounded_bites(directions, poly.name, ideal_angles, bites, radius) or windows
        corners = []
        for corner in _bite_corners(bites):
            rays = relaxed_shell(directions, corner)
            if rays is not None:
                corners.append((corner, rays))
                corner_images.append(np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0))))
        midpoint = {pair: 0.5 * sum(window) for pair, window in bites.items()}
        mid_rays = relaxed_shell(directions, midpoint)
        if mid_rays is None and corners:
            # The exact midpoint changes this template's oriented type; the pull is a weak bias (not a wall),
            # so use the surviving corner closest to it rather than leave every row in this sphere pull-free.
            mid_rays = min(
                corners, key=lambda item: sum((item[0][pair] - target) ** 2 for pair, target in midpoint.items())
            )[1]
        if mid_rays is not None:
            mid_angles = np.degrees(np.arccos(np.clip(mid_rays @ mid_rays.T, -1.0, 1.0)))
    else:
        bites = windows
    for pair, a in angle_rows.items():
        i, j = sorted(pair)
        if (
            od[i] == VACANT or od[j] == VACANT
        ):  # an angle to an empty vertex is unconstrained. NB a shape reached AS a vacancy is stated more weakly than
            # one with its own record: dropping a vertex drops every row naming it. Still right, because a tripod pulls
            # its geometry through its backbone, which `_chelate_bite_window` models for ring sizes 4/5/6.
            continue
        # The public D-D bond and two M-D windows already define this triangle. Retain RDKit's native bond
        # window as a stronger FF hold; an independent ideal angle can only tear this small coordination ring.
        if od[i] not in haptic and od[j] not in haptic and mol.GetBondBetweenAtoms(od[i], od[j]) is not None:
            left, right = sorted((od[i], od[j]))
            if pos is not None and 1 in (
                mol.GetAtomWithIdx(left).GetAtomicNum(),
                mol.GetAtomWithIdx(right).GetAtomicNum(),
            ):
                # An X-H-M three-centre bridge has an elongated X-H bond. RDKit's ordinary X-H bounds describe
                # a terminal bond and make UFF contract the bridge, so a measured source owns this one span.
                distance = float(np.linalg.norm(pos[left] - pos[right]))
                add_distance(c.distances, left, right, distance - _INPUT_HALF_WIDTH, distance + _INPUT_HALF_WIDTH)
                c.pulls[(left, right)] = distance
            else:
                bond_bounds = _fact(context, "bounds", lambda: _bounds_matrix(mol))
                add_distance(c.distances, left, right, bond_bounds[right, left], bond_bounds[left, right])
            continue
        key = (od[i], metal, od[j])
        if pair in bites:
            c.angles[key] = bites[pair]
        else:
            lo, hi = max(0.0, a - _ANGLE_PAD), min(_STRAIGHT, a + _ANGLE_PAD)
            if corner_images:
                values = [image[i, j] for image in corner_images]
                lo, hi = min(lo, *values), max(hi, *values)
            c.angles[key] = (lo, hi)
        # Every in-window FF force here is a flat-bottomed wall except Pull, so a compiled row's only
        # restoring force toward the construction's own arrangement is this pull: gating it to rows the
        # shell "moves" left the rest riding no bias at all, wide enough for a competing reading to win
        # (DULPUV).
        if mid_angles is not None:
            canonical = min(key, key[::-1])
            c.pulls.pop(canonical, None)
            c.pulls.pop(canonical[::-1], None)
            c.pulls[canonical] = float(mid_angles[i, j])
    # NB an η² π bond needs no hold of its own: the face is a centroid vertex, so one axial pull plus the cone
    # pins both π atoms at the face radius. Two separate M-donor pulls tore C≡C from 1.2 to 1.7 Å.
    coord = [d for d in od if d != VACANT and d not in haptic]
    coord += [a for site in haptic.values() for a in site]  # real coordinating atoms; centroid keys are reserved
    if force_field:
        ff_terms(  # coordinating atoms, so nondonor_floors never floors a ring atom
            mol,
            c,
            {metal: (real_z, coord)},
            frozen=frozen,
            fragments=frag,
            topology=context.get("topology"),
            sites=haptic_sites,  # same donor population as `coord`/`real_od`: grouped once above
        )
    _add_umbrella(c, metal, od, poly)
    return _drop_graft_owned(c, frozen, haptic)


def _donor_distance_window(mol, metal, donor, real_z, donors, *, positions=None, charges=None, hyb=None, eta=None):
    """Return the measured or model M-donor window shared by enumeration, coordination and site filling."""
    if positions is not None:
        target = float(np.linalg.norm(positions[metal] - positions[donor]))
        return target - _INPUT_HALF_WIDTH, target + _INPUT_HALF_WIDTH
    target = ml_distance(
        mol,
        metal,
        donor,
        real_z,
        donors,
        charges=delocalised_charges(mol) if charges is None else charges,
        hyb=_stripped_hybridisation(mol) if hyb is None else hyb,
        eta=eta,
    )
    return target - _ML_SEED_HALF_WIDTH, target + _ML_SEED_HALF_WIDTH


def _drop_graft_owned(cons, frozen, haptic):
    """Drop derived terms whose complete real geometry is restored by the graft."""
    if not frozen:
        return cons

    virtual = set(haptic)
    derived = (cons.distances, cons.angles, cons.dihedrals, cons.pulls, cons.umbrellas)
    active = set()
    for terms in derived:
        for key in terms:
            if not _graft_owns(key, frozen, haptic):
                active.update(virtual.intersection(key))
    for row in cons.coplanar:
        if not _graft_owns(row[:4], frozen, haptic):
            active.update(virtual.intersection(row[:4]))

    def keep(key):
        # A live virtual site is not itself grafted. Keep its transient numerical scaffold intact.
        return bool(active.intersection(key)) or not _graft_owns(key, frozen, haptic)

    for terms in (*derived, cons.floors, cons.dg_floors):
        for key in [key for key in terms if not keep(key)]:
            terms.pop(key)
    cons.coplanar = [row for row in cons.coplanar if keep(row[:4])]
    cons.haptic = {dummy: face for dummy, face in cons.haptic.items() if dummy in active}
    cons.phantoms = frozenset(dummy for dummy in cons.phantoms if dummy in active)
    return cons


def _add_umbrella(cons, metal, vertices, poly):
    """Cover a planar shell with radial impropers, or retain the three-donor pyramid hold."""
    if poly.umbrella_improper is None and not poly.planar:
        return
    occupied = [d for d in vertices if d != VACANT]
    if poly.planar and len(occupied) > _IMPROPER_VERTICES:
        directions = {atom: np.asarray(poly.vertex_dirs[i], float) for i, atom in enumerate(vertices) if atom != VACANT}
        directions = {atom: ray / np.linalg.norm(ray) for atom, ray in directions.items()}
        triples = list(itertools.combinations(occupied, _IMPROPER_VERTICES))
        for triple in triples:
            axes = []
            for middle in triple:
                left, right = (atom for atom in triple if atom != middle)
                u, v, w = (directions[atom] for atom in (left, middle, right))
                conditioning = min(np.linalg.norm(np.cross(u, v)), np.linalg.norm(np.cross(v, w)))
                axes.append(((left, metal, middle, right), conditioning))
            best = max(value for _, value in axes)
            axes = [key for key, value in axes if np.isclose(value, best, rtol=0.0, atol=_SHELL_ATOL)]
            # Both planes contain M: positive radial rescaling cannot change the template-side well.
            # Cover every triple and average tied axes. Fix/frozen removal must not reweight surviving rows.
            for key in axes:
                left, _, middle, right = key
                phi = _improper(directions[left], np.zeros(3), directions[middle], directions[right])
                cons.umbrellas[key] = (0.0 if abs(phi) < _STRAIGHT / 2 else _STRAIGHT, 1.0 / (len(triples) * len(axes)))
        return
    if poly.planar and len(occupied) >= _IMPROPER_VERTICES:
        cons.umbrellas[(*occupied[:_IMPROPER_VERTICES], metal)] = None
        return
    if len(occupied) == len(vertices) == _IMPROPER_VERTICES:
        cons.umbrellas[(*occupied, metal)] = poly.umbrella_improper
