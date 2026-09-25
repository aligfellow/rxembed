"""An `Isomer` -> the `Constraints` that hold its coordination sphere: M-donor windows and L-M-L angles.

The top of the metal stack: it reads `metal_distance` and `metal_donor_orient`, and nothing reads it back.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np
from rdkit import Chem

from .bounds import EmbedParams, bounds_matrix, coordination_reach_base, ligand_reach
from .constraints import Constraints, add_distance, compose, graft_owns
from .mechanisms import opposite_angle, triangle_angles
from .metal_core import (
    ETA2,
    VACANT,
    frag_map,
    haptic_sites,
    ligand_graph,
    logger,
    materialized_states,
    metal_indices,
    vertex_atom,
)
from .metal_distance import INPUT_HALF_WIDTH, delocalised_charges, ff_terms, ml_distance
from .metal_donor_orient import (
    COPLANAR_CAP,
    coplanar_donor,
    orient_donor,
    ring_hinge,
    stripped_hybridisation,
)
from .metal_perceive import rank_shapes, shape_reading
from .metal_polyhedron import (
    CHELATE_SPAN_ANGLE,
    IMPROPER_VERTICES,
    POLYHEDRA,
    Polyhedron,
    improper_dihedral,
    record,
    relaxed_shell,
    seating_is_exhaustive,
    vertex_angle,
)
from .metal_slots import SPAN_TOL, chelate_bite_window
from .stereo import bond_stereo, metal_referenced_ez, point_stereo

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval
_STRAIGHT = 180.0
_SHELL_ATOL = 1e-6  # numerical tolerance for template coplanarity and sector closure
# Slack on an otherwise-unconstrained compiled L-M-L angle row: wide enough for ordinary distortion, narrow
# enough that metal_screen.narrow_span_pairs can prove a pair's whole orbit will fail this angle wall.
ANGLE_PAD = 8.0


def _site_radius(mol, site, *, positions=None):
    """Return the root-mean-square distance of haptic members from their centroid.

    The identity ``R^2 = sum(i<j, d_ij^2)/n^2`` holds for any point set, including irregular faces.
    Without explicit positions, RDKit's pair-bound midpoints estimate those distances. They need not jointly
    describe a realizable point set. The caller owns length provenance; a conformer on `mol` is not authority
    to measure it. A two-atom eta2 site reduces to half the edge.
    """
    if positions is not None:
        pos = np.asarray(positions)[list(site)]
        return float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, axis=1))))

    bm = bounds_matrix(mol)
    tot = sum(
        (0.5 * (bm[max(a, b)][min(a, b)] + bm[min(a, b)][max(a, b)])) ** 2 for a, b in itertools.combinations(site, 2)
    )
    return float(np.sqrt(tot)) / len(site)


def _pair_bound(matrix, a, b, upper):
    """Return a pair's upper (above the diagonal) or lower (below it) bounds-matrix cell, zero for one atom."""
    if a == b:
        return 0.0
    return float(matrix[min(a, b)][max(a, b)] if upper else matrix[max(a, b)][min(a, b)])


def _site_radius_sq(matrix, site, upper):
    """Return a site's squared radius from one side of the bounds matrix, by `_site_radius`'s identity."""
    return sum(_pair_bound(matrix, a, b, upper) ** 2 for a, b in itertools.combinations(site, 2)) / len(site) ** 2


def _site_span(matrix, left, right):
    """Return the (lower, upper) centroid-to-centroid distance between two donor sites.

    ``|cA - cB|^2 = mean cross d^2 - R_A^2 - R_B^2`` holds for any two point sets (`_site_radius`'s same
    identity, one radius per site): the lower distance bound pairs the cross term's lower bound with each
    site's own upper-bound radius, and the upper bound pairs the reverse. A one-atom site has zero radius,
    so a sigma donor pair reduces to `matrix`'s own cell, unchanged by this.
    """
    pairs = len(left) * len(right)
    cross_lo = sum(_pair_bound(matrix, a, b, upper=False) ** 2 for a in left for b in right) / pairs
    cross_hi = sum(_pair_bound(matrix, a, b, upper=True) ** 2 for a in left for b in right) / pairs
    radius_hi = _site_radius_sq(matrix, left, upper=True) + _site_radius_sq(matrix, right, upper=True)
    radius_lo = _site_radius_sq(matrix, left, upper=False) + _site_radius_sq(matrix, right, upper=False)
    return float(np.sqrt(max(0.0, cross_lo - radius_hi))), float(np.sqrt(max(0.0, cross_hi - radius_lo)))


def _site_height(radius, member_lengths):
    """Estimate centroid distance using ``|M-c|^2 = mean(|M-member|^2) - R^2``.

    Retain the existing 0.5 A scaffold floor; it is not a feasibility proof for fitted member distances.
    """
    mean_squared_length = float(np.mean(np.square(member_lengths)))
    return float(np.sqrt(max(mean_squared_length - radius * radius, 0.25)))


def _regular_face(mol, site):
    """Return True if `site`'s own bond graph is regular: every face atom has the same face-neighbour count.

    Regularity is what makes one shared centroid radius true, so it is the rigidity test. Degree 1 is an edge
    (eta2), degree 2 a cycle (Cp/arene), while an open allyl is irregular and its centroid sits nearer the
    inner atoms. One rule for every hapticity: a bond is the smallest rigid face, so eta2 needs no special case.
    """
    face = set(site)
    deg = {sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in face) for a in face}
    return len(deg) == 1


def _centroid_constraints(sphere, dummy, qdel, hyb, c):
    """Constrain one haptic face through a transient centroid vertex.

    A regular face adds radius priors to the DG cone; irregular faces keep only metal-member distances.
    During UFF, Haptic replaces these radius priors with a moving-centroid penalty, without flattening the face.
    """
    mol, metal, pos, ring = sphere.mol, sphere.metal, sphere.pos, sphere.haptic[dummy]
    r = _site_radius(mol, ring, positions=pos)
    if pos is not None:  # realised metal->ring-atom distances from the input geometry
        mcs = [float(np.linalg.norm(pos[metal] - pos[a])) for a in ring]
    else:  # the fitted model; the ring is its own co-donor set so the hapticity term sees the whole face
        mcs = [ml_distance(mol, metal, a, sphere.real_z, set(ring), charges=qdel, hyb=hyb) for a in ring]
    d_c = _site_height(r, mcs)
    add_distance(  # M -> centroid vertex, both modes
        c.distances,
        metal,
        dummy,
        d_c - (INPUT_HALF_WIDTH if pos is not None else _ML_SEED_HALF_WIDTH),
        d_c + (INPUT_HALF_WIDTH if pos is not None else _ML_SEED_HALF_WIDTH),
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
        "fragments": frag_map(mol),
        "hybridisation": stripped_hybridisation(mol),
        "charges": delocalised_charges(mol),
        "bounds": bounds_matrix(mol),
        "topology": Chem.GetDistanceMatrix(mol),
        "stripped": ligand_graph(mol),
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


@dataclass(frozen=True, eq=False)
class CoordinationSphere:
    """Hold one metal's coordination sphere: the metal, its polyhedron, and the donor seated at each vertex.

    A vertex holds a donor atom, a transient haptic centroid key (`haptic` maps it to its face atoms) or
    VACANT. `pos` holds measured input positions for input lengths and is None for model lengths; `mol`
    supplies topology either way. `overrides` are M-D windows the caller already states, `frozen` the grafted
    atoms and `coupled` atoms held by other constraints. `context` is the compile cache shared across spheres
    and candidates, so it stays mutable.
    """

    mol: Chem.Mol
    metal: int
    real_z: int
    vertices: tuple
    haptic: dict
    poly: Polyhedron
    frozen: set | frozenset = frozenset()
    pos: np.ndarray | None = None
    overrides: dict = field(default_factory=dict)
    coupled: set | frozenset = frozenset()
    donor_orientation: bool = True
    context: dict = field(default_factory=dict)

    @cached_property
    def sigma(self):
        """Return the single-point donors, the chelate hinge's own ring set."""
        return {x for x in self.vertices if x != VACANT and x not in self.haptic}

    @cached_property
    def real(self):
        """Return the real co-donors: single-point donors plus face atoms, since centroid keys are not atoms."""
        real = set(self.sigma)
        real.update(a for face in self.haptic.values() for a in face)
        return real

    def frag_of(self, v):
        """Return vertex `v`'s ligand fragment, reading a centroid through its face atoms.

        A bond-less centroid carries no fragment of its own, so a face tethered to a co-donor (an η²-alkyne
        plus alkyl of one metallacycle, an ansa Cp) still reads as one ligand.
        """
        frag = _fact(self.context, "fragments", lambda: frag_map(self.mol))
        return frag[vertex_atom(self.haptic, v)]


def compile_constraints(iso, *external, params=None, force_field=True, context=None):
    """Compile `iso`'s constrained metal states onto a copy of its base constraints.

    `iso.graph` supplies current topology and atom indexing; `iso.length_mol` preserves the input geometry
    used for lengths. `external` informs model independence without merging or bypassing the caller's later
    validation. `params` (an `EmbedParams`) supplies the `donor_orientation` and `conjugation` switches, both
    on when omitted. `force_field=False` keeps the same coordination targets for a reach screen while omitting
    derived contact walls; it is not an alternative embedding model.
    """
    params = EmbedParams() if params is None else params
    context = {} if context is None else context
    mol = iso.graph
    base = iso.base_cons.copy(donor_orientation=params.donor_orientation, conjugation=params.conjugation)
    built = []
    if iso.constrained_metals:
        pos, _note = resolve_lengths(iso.length_mol, iso.lengths)
        parts = materialized_states(mol, iso.centres)
        held = base.constrained_atoms().union(*(cons.constrained_atoms() for cons in external if cons is not None))
        for state in iso.centres:
            if state.atom not in iso.constrained_metals:
                continue
            poly = record(state.geometry)
            if poly is None:
                raise ValueError(
                    f"no polyhedron template for {state.geometry!r}; add a POLYHEDRA row before embedding this "
                    "coordination number"
                )
            vertices, haptic, _winding, _donors = parts[state.atom]
            sphere = CoordinationSphere(
                mol,
                state.atom,
                state.atomic_num,
                tuple(vertices),
                haptic,
                poly,
                frozen=base.frozen,
                pos=pos,
                overrides=base.distances,
                coupled=held | {donor for donor, other in iso.donor_bonds if other != state.atom},
                donor_orientation=params.donor_orientation,
                context=context,
            )
            built.append(coordination(sphere, force_field))
    out = compose(*built, base)
    _add_donor_angle_floors(out, mol, context.get("bounds"))
    _add_point_umbrellas(out, mol, iso.stereo_label, iso.donor_bonds)
    _add_ligand_ez(out, mol, iso.stereo_label, iso.donor_bonds)
    _add_metal_ez(out, mol, iso.stereo_label, iso.donor_bonds)
    return out


def _add_native_pair_floors(cons, mol, pairs, bounds=None):
    """Keep unowned nonbonded pairs above RDKit's native lower bounds during restrained UFF."""
    pairs = {
        tuple(sorted(pair))
        for pair in pairs
        if mol.GetBondBetweenAtoms(*pair) is None
        and tuple(sorted(pair)) not in cons.distances
        and not graft_owns(pair, cons.frozen)
    }
    if not pairs:
        return
    bounds = bounds_matrix(mol) if bounds is None else bounds
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
            if len(carriers) == 4  # noqa: PLR2004 - a tetrahedral centre has four carriers
            else (*carriers, centre)
            if len(carriers) == 3  # noqa: PLR2004 - a trigonal centre has three carriers plus the centre itself
            else None
        )
        if key is not None and not graft_owns(key, cons.frozen):
            cons.umbrellas.setdefault(key, 0.0)


def _add_ligand_ez(cons, mol, stereo_label, donor_bonds):
    """Keep stated ligand double bonds inside their RDKit-reference half-space during UFF."""
    metal_owned = set(metal_referenced_ez(mol, stereo_label, donor_bonds))
    for pair in bond_stereo(stereo_label):
        if pair in metal_owned:
            continue
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None or len(bond.GetStereoAtoms()) != 2:  # noqa: PLR2004 - a double bond has two stereo atoms
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
        row = (refs[0], begin, end, refs[1], anchor, COPLANAR_CAP)
        if not graft_owns(row[:4], cons.frozen):
            cons.coplanar.append(row)


def _add_metal_ez(cons, mol, stereo_label, donor_bonds):
    """Choose the donor-plane well when a metal is an imine's missing E/Z reference."""
    for donor, other, metal, ref, ligand_ref, wanted in metal_referenced_ez(mol, stereo_label, donor_bonds).values():
        anchor = 0.0 if wanted == "Z" else _STRAIGHT
        existing = [row for row in cons.coplanar if row[:3] == (metal, donor, other) and row[4] is None]
        cap = existing[0][5] if existing else COPLANAR_CAP
        if existing:
            cons.coplanar.remove(existing[0])
        metal_row = (metal, donor, other, ref, anchor, cap)
        if not graft_owns(metal_row[:4], cons.frozen):
            cons.coplanar.append(metal_row)
        if ligand_ref is not None:
            ligand_row = (ligand_ref, donor, other, ref, _STRAIGHT - anchor, cap)
            if not graft_owns(ligand_row[:4], cons.frozen):
                cons.coplanar.append(ligand_row)


_EXHAUSTIVE_BITE_CORNERS = 4  # ponytail: 2**k corners is a resource ceiling above this many simultaneous bites;
# widen only if a real structure needs more than the extremes-plus-single-flips fallback below.


def _bite_reach(left, right, span):
    """Return the donor-donor angle range (degrees) two fixed model M-D legs support at a `span` (Å) range.

    `triangle_angles` encloses angle by interval side lengths; a fixed leg is passed as a degenerate
    (x, x) interval. It can return ``None`` when the legs and span ranges admit no triangle at all (never
    observed at these near-point legs, but the fixed-leg law of cosines this replaces never rejected either,
    so fall back to it rather than lose the row).
    """
    angles = triangle_angles((left, left), (right, right), span)
    if angles is None:
        angles = (opposite_angle(left, right, span[0]), opposite_angle(left, right, span[1]))
    return math.degrees(angles[0]), math.degrees(angles[1])


_CONTACT_SCALE = 0.7  # x (r_vdw + r_vdw): RDKit's own nonbonded 1-5 floor, VDW_SCALE_15 in BoundsMatrixBuilder.cpp


def _contact_angle(mol, metal, left, right, distances):
    """Return the donor-donor angle at which two model M-D legs meet their scaled van der Waals contact floor.

    Reuses `_bite_reach`'s law of cosines: the floor is just the lower end of its distance span.
    """
    table = Chem.GetPeriodicTable()
    floor = _CONTACT_SCALE * sum(table.GetRvdw(mol.GetAtomWithIdx(a).GetAtomicNum()) for a in (left, right))
    legs = [0.5 * sum(distances[(min(a, metal), max(a, metal))]) for a in (left, right)]
    return _bite_reach(legs[0], legs[1], (floor, legs[0] + legs[1]))[0]


def _bite_window_and_reach(mol, left, right, donors, matrix, lengths, row):
    """Return (window, reach) for a same-ligand bite pair, or None where neither narrows the row.

    `reach` is the triangle the two model M-D legs and the native ligand reach (`matrix`) support. A sigma
    pair (one atom each side) also has a ring-size census window (`chelate_bite_window`); `window` is that
    census clamped inside `reach`, or `reach` alone where the two ranges miss each other, and no census row
    means no bite. A haptic pair has no census row either, so `reach` alone replaces the compiled
    polyhedron angle (`row`) wherever it narrows it.
    """
    prior = None
    if len(left) == 1 and len(right) == 1:
        prior = chelate_bite_window(mol, left[0], right[0], donors)
        if prior is None:
            return None
    if matrix is None:  # ligand_reach could not close a triangle for this graph
        return (prior, prior) if prior is not None else None
    left_leg, right_leg = map(float, lengths)
    lower, upper = _site_span(matrix, left, right)
    reach = _bite_reach(left_leg, right_leg, (max(0.0, lower - SPAN_TOL), upper + SPAN_TOL))
    if prior is not None:
        # Decide agreement on the backbone triangle itself, not the SPAN_TOL-widened reach: the slack is
        # numerical room for the returned window, not licence to call a miss an overlap.
        exact = _bite_reach(left_leg, right_leg, (lower, upper))
        if exact[1] < prior[0] or exact[0] > prior[1]:
            return reach, reach
        return (max(prior[0], reach[0]), min(prior[1], reach[1])), reach
    if reach[0] <= row[0] and reach[1] >= row[1]:
        return None
    return reach, reach


def seated_bites(sphere, ideal_angles, distances, angle_rows=None):
    """Return each same-ligand donor pair's (window, reach) angle windows, keyed by its seated site pair.

    The one place `coordination` and `metal_screen`'s chelate screens both read, so the bite condition
    is defined once, not copied. `angle_rows` is `coordination`'s stated polyhedron subset; a pair missing
    from it falls back to `ideal_angles`.
    """
    mol, metal, od, haptic, context = sphere.mol, sphere.metal, sphere.vertices, sphere.haptic, sphere.context
    angle_rows = angle_rows or {}
    bites = {}
    for i, j in itertools.combinations(range(len(od)), 2):
        left, right = od[i], od[j]
        pair = frozenset((i, j))
        if (
            VACANT in (left, right)
            or sphere.frag_of(left) != sphere.frag_of(right)
            or ideal_angles[pair] >= CHELATE_SPAN_ANGLE
        ):
            continue
        keys = ((min(metal, left), max(metal, left)), (min(metal, right), max(metal, right)))
        if any(key not in distances for key in keys):  # the screen's sigma radial legs carry no haptic leg
            continue
        lengths = (0.5 * sum(distances[keys[0]]), 0.5 * sum(distances[keys[1]]))
        left_site, right_site = haptic.get(left, (left,)), haptic.get(right, (right,))
        base = angle_rows.get(pair, ideal_angles[pair])
        row = (max(0.0, base - ANGLE_PAD), min(_STRAIGHT, base + ANGLE_PAD))
        bite_key = ("bite", tuple(sorted(left_site)), tuple(sorted(right_site)), lengths, row)
        if bite_key not in context:
            try:
                native = _fact(
                    context,
                    "native_reach",
                    lambda: coordination_reach_base(mol, ligand_reach(mol), set(metal_indices(mol))),
                )
            except (ValueError, RuntimeError):
                logger.debug("metal[%s]: native ligand reach unavailable for a bite pair", metal)
                context["native_reach"] = native = None
            context[bite_key] = _bite_window_and_reach(mol, left_site, right_site, sphere.real, native, lengths, row)
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


def bounded_bites(directions, name, ideal, bites):
    """Shrink each bite's census window toward its ideal-clamped anchor until every corner reads as `name`.

    Several bites drifting together, not just one, can flip the perceived shape, so every combination of
    window corners (`_bite_corners`) is checked jointly. When even the clamped anchor fails, the window
    opens instead toward each bite's wider backbone-triangle `reach`; if that still fails too, this returns
    None, meaning the backbone itself cannot hold `name`, not just the ring-size prior.
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
    return _bounded_bites_cached(tuple(map(tuple, directions)), name, key)


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
def _reads_as_name_cached(directions, corner_key, name):
    """Return whether `corner_key`'s bite targets relax to a shell that still reads as `name`.

    The reading applies the near-tie margin the acceptance gate applies to a real conformer
    (`metal_perceive.shape_reading`). Corners recur often across bisection rounds and across candidates sharing a
    bite target, so this memoises the check alongside `relaxed_shell`'s own memoised relaxation.
    """
    bites = {frozenset(pair): target for pair, target in corner_key}
    rays = relaxed_shell(directions, bites)
    return rays is not None and shape_reading(rank_shapes(rays), name)[3]


def _shrunk_windows(windows, anchor, t):
    """Return each bite window shrunk toward its anchor target by the fraction `t`."""
    return {pair: (lo + t * (anchor[pair] - lo), hi - t * (hi - anchor[pair])) for pair, (lo, hi) in windows.items()}


@lru_cache(maxsize=None)
def _bounded_bites_cached(directions, name, key):
    anchor = {pair: min(max(ideal, lo), hi) for pair, (lo, hi), _reach, ideal in key}
    windows = {pair: (lo, hi) for pair, (lo, hi), _reach, _ideal in key}
    far = {pair: min(max(ideal, rlo), rhi) for pair, _window, (rlo, rhi), ideal in key}
    if not _reads_as_name_cached(directions, _corner_key(anchor), name):
        if _reads_as_name_cached(directions, _corner_key(far), name):
            return {pair: (min(anchor[pair], far[pair]), max(anchor[pair], far[pair])) for pair in windows}
        return None
    if all(_reads_as_name_cached(directions, _corner_key(corner), name) for corner in _bite_corners(windows)):
        return windows
    good, bad = 1.0, 0.0  # t=1 is the anchor itself (degenerate, always reads correctly); t=0 the full window
    for _ in range(_BITE_BOX_STEPS):
        mid = 0.5 * (good + bad)
        box = _shrunk_windows(windows, anchor, mid)
        if all(_reads_as_name_cached(directions, _corner_key(corner), name) for corner in _bite_corners(box)):
            good = mid
        else:
            bad = mid
    return _shrunk_windows(windows, anchor, good)


def _radial_windows(sphere, c):
    """Fill each donor's M-D window, wall its substituent off the metal, and hinge a small chelate ring flat."""
    mol, metal, context = sphere.mol, sphere.metal, sphere.context
    # Group the whole sphere into sigma/pi sites once: `ml_distance` (via `eta`) reuses it instead of
    # regrouping per donor.
    sites = haptic_sites(mol, sphere.real)
    eta = {d: len(site) for site in sites if len(site) >= ETA2 for d in site}
    qdel = _fact(context, "charges", lambda: delocalised_charges(mol))
    # A Lewis charge is an artefact, so spread it.
    # The model uses ligand-only classes for either input bond convention. Input lengths do not need typing.
    hyb = _fact(context, "hybridisation", lambda: stripped_hybridisation(mol))
    # `orient_donor` rebuilds this per donor unless it is handed one already built; share it across the sphere.
    stripped = _fact(context, "stripped", lambda: ligand_graph(mol)) if sphere.donor_orientation else None
    for d in sphere.vertices:
        if d == VACANT:
            continue
        if d in sphere.haptic:  # a centroid vertex: pin its whole ring, not a single donor (no orient/coplanar)
            _centroid_constraints(sphere, d, qdel, hyb, c)
            continue
        if hyb is None:  # Measured lengths still need graph-derived donor orientation, shared across the sphere.
            hyb = stripped_hybridisation(mol)
        key = (min(metal, d), max(metal, d))
        if key in sphere.overrides:
            add_distance(c.distances, metal, d, *sphere.overrides[key])
        else:
            add_distance(
                c.distances,
                metal,
                d,
                *donor_distance_window(
                    mol,
                    metal,
                    d,
                    sphere.real_z,
                    sphere.real,
                    positions=sphere.pos,
                    charges=qdel,
                    hyb=hyb,
                    eta=eta.get(d, 0),
                ),
            )
        # Wall each donor substituent off the metal, the orientation hold a real energy cannot supply itself.
        # Length provenance is independent: measured M-L distances do not determine a partially free M-D-X axis.
        if sphere.donor_orientation:
            orient_donor(mol, metal, d, sphere.real, c, hyb=hyb, stripped=stripped)
            # cap an sp2 donor's metal at the donor's own sp2 plane: the improper the stripped bond removed.
            coplanar_donor(mol, metal, d, sphere.real, c, hyb=hyb)
    if sphere.donor_orientation:
        # A small, fully conjugated chelate ring hinges flat as a unit (metal_donor_orient.ring_hinge). Called
        # once per metal, after the per-donor loop above, since it walks donor pairs, not one donor at a time.
        ring_hinge(mol, metal, sphere.sigma, c)


def _bonded_donor_windows(sphere, c):
    """Return this geometry's ideal and stated angle rows, widening a bonded donor pair to fit its own bond.

    A model-length donor pair joined by a real bond gets both M-D windows widened wherever the compiled
    angle would pull that bond past RDKit's native upper bound.
    """
    mol, metal, od, poly, context = sphere.mol, sphere.metal, sphere.vertices, sphere.poly, sphere.context
    pairs = list(itertools.combinations(range(len(od)), 2))
    angle_key = ("angles", poly.name)
    if angle_key not in context:
        context[angle_key] = (
            {frozenset((i, j)): vertex_angle(poly.vertex_dirs[i], poly.vertex_dirs[j]) for i, j in pairs},
            {frozenset((i, j)): a for i, j, a in poly.resolved_angles},
        )
    ideal_angles, angle_rows = (dict(values) for values in context[angle_key])
    stated_pairs = set(angle_rows)
    expanded = {}
    if sphere.pos is None:
        for i, j in pairs:
            left, right = od[i], od[j]
            if (
                VACANT in (left, right)
                or left in sphere.haptic
                or right in sphere.haptic
                or mol.GetBondBetweenAtoms(left, right) is None
            ):
                continue
            left_key = (min(left, metal), max(left, metal))
            right_key = (min(right, metal), max(right, metal))
            if left_key in sphere.overrides or right_key in sphere.overrides:
                continue
            bond_bounds = _fact(context, "bounds", lambda: bounds_matrix(mol))
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
    return ideal_angles, angle_rows, stated_pairs


def _fill_angle_rows(sphere, ideal_angles, angle_rows, bites, stated_pairs):
    """Extend the stated angle rows to cover a planar shell's remaining pairs and every graft-owned pair.

    A planar shell needs every pair held, or the pairs left out of the sparse ideal-angle subset pucker out
    of plane once the windows have finite width. A non-planar shell only needs a row for its own bite pairs,
    to have something for the bite window to narrow later.
    """
    od = sphere.vertices
    if sphere.poly.planar:
        angle_rows.update({pair: angle for pair, angle in ideal_angles.items() if pair not in angle_rows})
    else:
        angle_rows.update({pair: ideal_angles[pair] for pair in bites if pair not in angle_rows})
    if sphere.metal not in sphere.frozen:  # fixed donors do not fix their angle about a free metal
        for i, j in itertools.combinations(range(len(od)), 2):
            if (
                VACANT not in (od[i], od[j])
                and frozenset((i, j)) not in stated_pairs
                and graft_owns((od[i], od[j]), sphere.frozen, sphere.haptic)
            ):
                pair = frozenset((i, j))
                angle_rows.setdefault(pair, ideal_angles[pair])


def _chelate_bite_images(sphere, ideal_angles, bites):
    """Narrow each bite pair to its census window, then relax the whole shell to every bite-window corner.

    A bite pair takes its own backbone window, clamped so it cannot reach a corner that would read as a
    different polyhedron (`bounded_bites`; the face rule lives in `relaxed_shell`). The resulting corner
    shells feed `_assign_angle_rows`, which pulls every other row toward them. Skipped when the metal or a
    bitten ligand's own atom is held elsewhere.
    """
    od = sphere.vertices
    blocked = set(sphere.frozen) | set(sphere.coupled)
    bite_fragments = {sphere.frag_of(od[i]) for pair in bites for i in pair}
    corner_images, mid_angles = [], None
    windows = {pair: window for pair, (window, _reach) in bites.items()}
    if (
        bites
        and sphere.metal not in blocked
        and not bite_fragments & {sphere.frag_of(atom) for atom in blocked}
        and VACANT not in od
    ):
        directions = sphere.poly.vertex_dirs
        bites = bounded_bites(directions, sphere.poly.name, ideal_angles, bites) or windows
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
    return bites, corner_images, mid_angles


def _assign_angle_rows(sphere, angle_rows, bites, corner_images, mid_angles, c):
    """Compile each angle row to a bonded D-D window, a bite window, or a spectator window, plus the shell pull.

    An angle to an empty vertex is left unconstrained; that states the shape more weakly, but a tripod still
    pulls its own geometry through its backbone (`chelate_bite_window`, ring sizes 4/5/6), so it stays right
    without the missing row. A spectator row (no bite of its own) is the union of the shells the bites can
    build, padded the ordinary amount past the nearest shell; between two different ligands it can still
    narrow, but never past the point where their donors would clash inside their contact floor.
    """
    mol, metal, od, haptic, pos = sphere.mol, sphere.metal, sphere.vertices, sphere.haptic, sphere.pos
    for pair, a in angle_rows.items():
        i, j = sorted(pair)
        if od[i] == VACANT or od[j] == VACANT:
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
                add_distance(c.distances, left, right, distance - INPUT_HALF_WIDTH, distance + INPUT_HALF_WIDTH)
                c.pulls[(left, right)] = distance
            else:
                bond_bounds = _fact(sphere.context, "bounds", lambda: bounds_matrix(mol))
                add_distance(c.distances, left, right, bond_bounds[right, left], bond_bounds[left, right])
            continue
        key = (od[i], metal, od[j])
        if pair in bites:
            c.angles[key] = bites[pair]
        else:
            lo, hi = max(0.0, a - ANGLE_PAD), min(_STRAIGHT, a + ANGLE_PAD)
            if corner_images:  # union with the shells, then no further than the pad from the nearest shell
                values = [image[i, j] for image in corner_images]
                union = min(lo, *values), max(hi, *values)
                lo = max(union[0], min(values) - ANGLE_PAD)
                hi = min(union[1], max(values) + ANGLE_PAD)
                if od[i] not in haptic and od[j] not in haptic and sphere.frag_of(od[i]) != sphere.frag_of(od[j]):
                    # A spectator row from a different ligand can narrow, but never below the angle where
                    # its two donors would clash inside their contact floor; never past the union either.
                    hi = max(hi, min(union[1], _contact_angle(mol, metal, od[i], od[j], c.distances)))
            c.angles[key] = (lo, hi)
        # Every other in-window FF term is a flat-bottomed wall, so this pull is a row's only restoring force.
        # Pulling only the rows the shell moves lets a competing reading win (DULPUV).
        if mid_angles is not None:
            canonical = min(key, key[::-1])
            c.pulls.pop(canonical, None)
            c.pulls.pop(canonical[::-1], None)
            c.pulls[canonical] = float(mid_angles[i, j])


def _apply_force_field(sphere, c):
    """Set the force field for this sphere's real coordinating atoms, excluding any haptic centroid dummy."""
    # NB an η² π bond needs no hold of its own: the face is a centroid vertex, so one axial pull plus the cone
    # pins both π atoms at the face radius. Two separate M-donor pulls tore C≡C from 1.2 to 1.7 Å.
    haptic = sphere.haptic
    coord = [d for d in sphere.vertices if d != VACANT and d not in haptic]
    coord += [a for site in haptic.values() for a in site]  # real coordinating atoms; centroid keys are reserved
    ff_terms(  # coordinating atoms, so nondonor_floors never floors a ring atom
        sphere.mol,
        c,
        {sphere.metal: (sphere.real_z, coord)},
        frozen=sphere.frozen,
        fragments=sphere.context.get("fragments"),
        topology=sphere.context.get("topology"),
    )


def coordination(sphere, force_field=True):
    """Build one metal state's distance, angle, floor and umbrella constraints.

    Cis chelates use calibrated ring-size bite windows (`seated_bites`); every other angle row narrows
    toward the shells those bites can build, or otherwise takes the polyhedron's own ideal angle
    (`_chelate_bite_images`, `_assign_angle_rows`). This compiles the full model before `_drop_graft_owned`
    removes the terms `sphere.frozen` already covers. A false `force_field` flag omits only derived nonbonded
    contact terms, for enumeration screening.

    Each step below reads and writes the same `c` in dependency order: the radial M-D windows first, since
    the bite, angle and force-field steps that follow all read them.
    """
    c = Constraints()
    _radial_windows(sphere, c)
    ideal_angles, angle_rows, stated_pairs = _bonded_donor_windows(sphere, c)
    bites = seated_bites(sphere, ideal_angles, c.distances, angle_rows)
    _fill_angle_rows(sphere, ideal_angles, angle_rows, bites, stated_pairs)
    bites, corner_images, mid_angles = _chelate_bite_images(sphere, ideal_angles, bites)
    _assign_angle_rows(sphere, angle_rows, bites, corner_images, mid_angles, c)
    if force_field:
        _apply_force_field(sphere, c)
    _add_umbrella(c, sphere.metal, sphere.vertices, sphere.poly)
    return _drop_graft_owned(c, sphere.frozen, sphere.haptic)


def donor_distance_window(mol, metal, donor, real_z, donors, *, positions=None, charges=None, hyb=None, eta=None):
    """Return the measured or model M-donor window shared by enumeration, coordination and site filling."""
    if positions is not None:
        target = float(np.linalg.norm(positions[metal] - positions[donor]))
        return target - INPUT_HALF_WIDTH, target + INPUT_HALF_WIDTH
    target = ml_distance(
        mol,
        metal,
        donor,
        real_z,
        donors,
        charges=delocalised_charges(mol) if charges is None else charges,
        hyb=stripped_hybridisation(mol) if hyb is None else hyb,
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
            if not graft_owns(key, frozen, haptic):
                active.update(virtual.intersection(key))
    for row in cons.coplanar:
        if not graft_owns(row[:4], frozen, haptic):
            active.update(virtual.intersection(row[:4]))

    def keep(key):
        # A live virtual site is not itself grafted. Keep its transient numerical scaffold intact.
        return bool(active.intersection(key)) or not graft_owns(key, frozen, haptic)

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
    if poly.planar and len(occupied) > IMPROPER_VERTICES:
        directions = {atom: np.asarray(poly.vertex_dirs[i], float) for i, atom in enumerate(vertices) if atom != VACANT}
        directions = {atom: ray / np.linalg.norm(ray) for atom, ray in directions.items()}
        triples = list(itertools.combinations(occupied, IMPROPER_VERTICES))
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
                phi = improper_dihedral(directions[left], np.zeros(3), directions[middle], directions[right])
                cons.umbrellas[key] = (0.0 if abs(phi) < _STRAIGHT / 2 else _STRAIGHT, 1.0 / (len(triples) * len(axes)))
        return
    if poly.planar and len(occupied) >= IMPROPER_VERTICES:
        cons.umbrellas[(*occupied[:IMPROPER_VERTICES], metal)] = None
        return
    if len(occupied) == len(vertices) == IMPROPER_VERTICES:
        cons.umbrellas[(*occupied, metal)] = poly.umbrella_improper
