"""Own the ligand facts every sphere shares and compile the constraints that hold selected coordination spheres."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache

import numpy as np
from rdkit import Chem

from rxembed.geometry.bounds import EmbedParams, bounds_matrix, ligand_reach
from rxembed.geometry.constraints import (
    DIHEDRAL_ATOMS,
    Constraints,
    CoplanarRow,
    Improper,
    add_distance,
    canonical_key,
    compose,
    graft_owns,
    structural_dihedral_owned,
)
from rxembed.geometry.mechanisms import law_of_cosines, opposite_angle, triangle_angles
from rxembed.graph import ETA2, frag_map, haptic_sites, ligand_graph
from rxembed.metal.centres import Centre, donor_bonds, engine_graph, logger, vertex_atom
from rxembed.metal.distances import (
    BRIDGE_H,
    INPUT_HALF_WIDTH,
    bridge_floors,
    bridge_partners,
    delocalised_charges,
    ff_terms,
    ml_distance,
)
from rxembed.metal.orientation import (
    COPLANAR_CAP,
    coplanar_donor,
    orient_donor,
    ring_hinge,
    stripped_hybridisation,
)
from rxembed.metal.polyhedra import (
    FLOAT_CACHE,
    IMPROPER_VERTICES,
    VACANT,
    Polyhedron,
    improper_dihedral,
    record,
    relaxed_shell,
)
from rxembed.metal.shape import shape_reading
from rxembed.metal.slots import SPAN_TOL, chelate_bite_window
from rxembed.metal.stereo import DonorRole
from rxembed.utils import EZ_RELATION, STRAIGHT, without_atom_maps

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval
_CONE_HALF_WIDTH = 0.1  # Å: room around a regular face's centroid-to-member radius prior, in either length mode
_NOTHING = Constraints()
_SHELL_ATOL = 1e-6  # numerical tolerance for template coplanarity and sector closure
# slack on a free L-M-L row: room for distortion, narrow enough for the reach screen to prove a conflict
ANGLE_PAD = 8.0


def _site_radius(facts, site, *, positions=None):
    """Return the root-mean-square distance of a haptic site's members from their centroid.

    ``R^2 = sum(i<j, d_ij^2)/n^2`` holds for any point set. Without positions the distances are the native pair-bound
    midpoints in `facts.bounds`, which need not form a real point set. A two-atom site gives half the edge.
    """
    if positions is not None:
        pos = np.asarray(positions)[list(site)]
        return float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, axis=1))))

    bm = facts.bounds
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

    ``|cA - cB|^2 = mean cross d^2 - R_A^2 - R_B^2``: the lower bound pairs the cross term's lower bound with each
    site's upper radius, the upper bound the reverse. A one-atom site has zero radius, so a sigma pair reduces to
    `matrix`.
    """
    pairs = len(left) * len(right)
    cross_lo = sum(_pair_bound(matrix, a, b, upper=False) ** 2 for a in left for b in right) / pairs
    cross_hi = sum(_pair_bound(matrix, a, b, upper=True) ** 2 for a in left for b in right) / pairs
    radius_hi = _site_radius_sq(matrix, left, upper=True) + _site_radius_sq(matrix, right, upper=True)
    radius_lo = _site_radius_sq(matrix, left, upper=False) + _site_radius_sq(matrix, right, upper=False)
    return float(np.sqrt(max(0.0, cross_lo - radius_hi))), float(np.sqrt(max(0.0, cross_hi - radius_lo)))


def _centroid_constraints(sphere, dummy, qdel, hyb):
    """Return the constraints holding one haptic face through a transient centroid vertex.

    A regular face adds radius priors to the DG cone; an irregular one keeps only metal-member distances. UFF replaces
    the priors with a moving-centroid penalty that does not flatten the face.
    """
    mol, metal, ring = sphere.mol, sphere.metal, sphere.haptic[dummy]
    r = _site_radius(sphere.facts, ring, positions=sphere.pos)
    sites = haptic_sites(mol, ring) if sphere.pos is None else ()
    eta = {a: len(site) for site in sites if len(site) >= ETA2 for a in site}
    mcs = [_donor_length(sphere, a, qdel, hyb, eta.get(a, 0)) for a in ring]
    # the centroid height from |M-c|^2 = mean(|M-member|^2) - R^2, floored at 0.5 A to keep a degenerate face off M
    d_c = float(np.sqrt(max(np.mean(np.square(mcs)) - r * r, 0.25)))
    distances = {}
    add_distance(distances, metal, dummy, *_donor_window(sphere, d_c))  # M -> centroid vertex
    # only a regular face (an eta2 edge or a ring, not an open allyl) has one centroid radius to add as priors
    cone = len({sum(nb.GetIdx() in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()) for a in ring}) == 1
    for a, mca in zip(ring, mcs, strict=True):  # each ring atom at its metal distance, as every metal site is held
        add_distance(distances, metal, a, *_donor_window(sphere, mca))
        if cone:
            add_distance(distances, dummy, a, r - _CONE_HALF_WIDTH, r + _CONE_HALF_WIDTH)
    return Constraints(
        distances=distances,
        pulls={canonical_key((metal, dummy)): (d_c, d_c), **_hinge_push(sphere, ring)},
        haptic={dummy: tuple(ring)},  # embed scaffolding: materialised transiently in the DG/UFF, stored in no Mol
    )


# deg: a partly bound ring's hinge or side-on atom goes sp3, its free neighbour tetrahedral to the metal
_HINGE_ANGLE = math.degrees(math.acos(-1.0 / 3.0))


def _hinge_push(sphere, face):
    """Return pulls pushing the face substituents its distances leave free toward the tetrahedral M-a-x angle.

    A partly bound ring can fold at its conjugated hinge, and a two-atom face can roll about its axis. The one-sided
    window bends each such substituent away from the metal without dragging an in-plane side-on pair inward; a larger
    face's sp3 flap (cyclohexadienyl) folds freely and is not pushed.
    """
    mol, metal = sphere.mol, sphere.metal
    rings = sphere.facts.rings
    face = set(face)
    hinge_rings = [ring for ring in rings if face < ring]
    pulls = {}
    for a in face:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            x = bond.GetOtherAtomIdx(a)
            if x == metal or x in sphere.real:
                continue
            hinge = bond.GetIsConjugated() and any(x in ring for ring in hinge_rings)
            if hinge or len(face) == ETA2:
                pulls[(metal, a, x)] = (_HINGE_ANGLE, STRAIGHT)
    return pulls


@dataclass(eq=False)
class LigandFacts:
    """Hold the molecule-level data every sphere and candidate compiled from one graph shares.

    `graph` is the chemical graph and never changes; its surrogate graph and each fact are computed once, on first
    use. `memo` holds per-site-pair entries (bite windows, angle rows, reach certificates) for the facts' lifetime.
    """

    graph: Chem.Mol
    memo: dict = field(default_factory=dict)

    @cached_property
    def mol(self):
        """Return the surrogate graph shared by the operation's topology and bounds readers."""
        return engine_graph(self.graph)

    @cached_property
    def donor_bonds(self):
        """Return the chemical graph's ordered coordination edges."""
        return donor_bonds(self.graph)

    @cached_property
    def roles(self):
        """Return coordination roles for ranking sites on the surrogate graph."""
        return [
            DonorRole(donor=donor, metal=metal, atomic_num=self.graph.GetAtomWithIdx(metal).GetAtomicNum())
            for donor, metal in self.donor_bonds
        ]

    def centre(self, metal):
        """Return the chemical identity of one metal before its arrangement is selected."""
        atom = self.graph.GetAtomWithIdx(metal)
        return Centre(metal, atom.GetAtomicNum(), atom.GetFormalCharge())

    @cached_property
    def fragments(self):
        """Return each atom's ligand fragment."""
        return frag_map(self.mol)

    @cached_property
    def bound(self):
        """Return the declared metal count per donor, read by `stripped_hybridisation` and the bridge exemption."""
        return Counter(d for d, _metal in self.donor_bonds)

    @cached_property
    def hybridisation(self):
        """Return the ligand-only hybridisation classes."""
        return stripped_hybridisation(self.mol, self.bound)

    @cached_property
    def charges(self):
        """Return the delocalised formal charges."""
        return delocalised_charges(self.mol)

    @cached_property
    def bounds(self):
        """Return RDKit's native bounds matrix."""
        return bounds_matrix(self.mol)

    @cached_property
    def topology(self):
        """Return the bond-count distance matrix."""
        return Chem.GetDistanceMatrix(self.mol)

    @cached_property
    def stripped(self):
        """Return the ligand graph without the metals."""
        return ligand_graph(self.mol)

    @cached_property
    def rings(self):
        """Return the ligand graph's smallest rings."""
        return [frozenset(r) for r in Chem.GetSymmSSSR(self.stripped)]

    @cached_property
    def native_reach(self):
        """Return native ligand intervals, or None where RDKit cannot give a consistent reach.

        Its one build, read by the bite windows and by both enumeration screens of a ligand variant.
        """
        try:
            return ligand_reach(self.mol)
        except (ValueError, RuntimeError) as exc:
            logger.debug("native ligand reach unavailable (%s)", exc)
            return None


def input_positions(mol, lengths):
    """Return the conformer positions M-donor windows are measured from under ``lengths='input'``, else None."""
    if lengths == "model":
        return None
    if not mol.GetNumConformers():
        raise ValueError("lengths='input' requires a conformer; a SMILES carries no geometry; use lengths='model'")
    logger.info("metal: M-donor windows from the input conformer (lengths='input')")
    return mol.GetConformer().GetPositions()


@dataclass(frozen=True, eq=False)
class CoordinationSphere:
    """Hold one metal's coordination sphere: the metal, its polyhedron and the donor seated at each vertex.

    A vertex holds a donor atom, a transient haptic centroid key (`haptic` maps it to its face atoms) or VACANT. `pos`
    holds measured positions for input lengths, None for model lengths. `frozen` holds the grafted atoms and `coupled`
    the atoms other constraints hold.
    """

    mol: Chem.Mol
    metal: int
    real_z: int
    vertices: tuple
    haptic: dict
    poly: Polyhedron
    frozen: set | frozenset = frozenset()
    pos: np.ndarray | None = None
    coupled: set | frozenset = frozenset()
    donor_orientation: bool = True
    facts: LigandFacts = field(kw_only=True)

    @cached_property
    def sigma(self):
        """Return the single-point donors, the chelate hinge's own ring set."""
        return {x for x in self.vertices if x is not VACANT and x not in self.haptic}

    @cached_property
    def real(self):
        """Return the real co-donors: single-point donors plus face atoms, since centroid keys are not atoms."""
        real = set(self.sigma)
        real.update(a for face in self.haptic.values() for a in face)
        return real

    def frag_of(self, v):
        """Return vertex `v`'s ligand fragment, reading a bond-less centroid through its face atoms.

        A face tethered to a co-donor (an ansa Cp, an eta2-alkyne plus alkyl) thus reads as one ligand.
        """
        return self.facts.fragments[vertex_atom(self.haptic, v)]


def compile_constraints(species, request=None, contacts=None, *, params=None, facts=None):
    """Compile `request` together with the constraints that hold each centre of `species` that has a geometry.

    `species.mol` supplies topology and, under ``params.lengths='input'``, the measured M-donor distances. `contacts`
    informs model independence only. The frozen atoms of both are grafted, so the sphere drops the terms they own.
    `request` is composed last: its soft terms may not replace a structural sphere hold, while a numeric fix may.
    `params` (an `EmbedParams`) supplies the lengths source and the `donor_orientation` and `conjugation` switches.
    """
    external = [cons for cons in (request, contacts) if cons is not None]
    params = EmbedParams() if params is None else params
    facts = LigandFacts(species.mol) if facts is None else facts
    mol = facts.mol
    frozen = frozenset().union(*(cons.frozen for cons in external))
    base = Constraints(conjugation=params.conjugation, frozen=frozen)
    held = base.constrained_atoms().union(*(cons.constrained_atoms() for cons in external))
    held_centres = [centre for centre in species.centres if centre.geometry]
    pos = input_positions(mol, params.lengths) if held_centres else None  # only a held sphere measures lengths
    built = []
    for centre in held_centres:
        poly = record(centre.geometry)
        vertices, haptic, _winding, _donors = species.sites[centre.atom]
        sphere = CoordinationSphere(
            mol,
            centre.atom,
            centre.atomic_num,
            tuple(vertices),
            haptic,
            poly,
            frozen=base.frozen,
            pos=pos,
            coupled=held | {donor for donor, other in facts.donor_bonds if other != centre.atom},
            donor_orientation=params.donor_orientation,
            facts=facts,
        )
        built.append(coordination(sphere))
    bridges = bridge_floors(facts.donor_bonds, {c.atom: c.atomic_num for c in species.centres}, base.frozen)
    out = compose(*built, bridges, base) if built or bridges.floors else base  # nothing held composes to the base
    out = _add_donor_angle_floors(out, mol, facts)
    stereo = species.held_stereo
    out = _add_point_umbrellas(out, mol, stereo, facts.donor_bonds)
    out = _add_ligand_ez(out, mol, stereo, facts.donor_bonds)
    out = _add_metal_ez(out, mol, stereo, facts.donor_bonds)
    if request is None or request == _NOTHING:  # an empty request composes to `out`; skip its rebuild
        return out
    soft_distances, soft_angular = request.contacts.distances, request.contacts.angular
    sphere_angular = {canonical_key(k) for k in out.angles}
    overlap = sorted(
        (soft_distances & set(out.distances))
        | {
            k
            for k in soft_angular
            if canonical_key(k) in sphere_angular or (len(k) == DIHEDRAL_ATOMS and structural_dihedral_owned(out, k))
        }
    )
    if overlap:
        raise ValueError(
            f"constrain= overlaps the coordination term(s) {overlap}; drop them from constrain= or override with fix="
        )
    return compose(out, request)


def _add_donor_angle_floors(cons, mol, facts):
    """Return `cons` keeping constrained donor substituents inside RDKit's native 1-3 lower bounds (`facts.bounds`)."""
    pairs = {(row.atoms[1], *sorted(row.atoms[2:])) for row in cons.coplanar if row.out_of_plane}
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
    unowned = {
        pair
        for pair in pairs
        if mol.GetBondBetweenAtoms(*pair) is None and pair not in cons.distances and not graft_owns(pair, cons.frozen)
    }
    if not unowned:
        return cons
    bounds = facts.bounds
    floors = dict(cons.floors)
    for left, right in unowned:  # keep each unowned 1-3 pair above RDKit's native lower bound during UFF
        floors[(left, right)] = max(floors.get((left, right), 0.0), float(bounds[right, left]))
    return replace(cons, floors=floors)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetalEZ:
    """Name a specified E/Z bond whose `donor` end takes its one `metal` as reference in place of a ligand atom.

    `other` is the bond's far end and `ref` that end's highest-ranked substituent; `wanted` is the specified ``E`` or
    ``Z``. `ligand_ref` is the donor's one hydrogen, held opposite the metal, or None when the donor carries none.
    """

    donor: int
    other: int
    metal: int
    ref: int
    ligand_ref: int | None
    wanted: str


def metal_referenced_ez(mol, stereo, donor_bonds):
    """Return each `MetalEZ` by bond: E/Z bonds whose donor endpoint uses one metal in place of a heavy substituent."""
    if not donor_bonds:
        return {}
    metals = {}
    for donor, metal in donor_bonds:
        metals.setdefault(donor, set()).add(metal)
    metal_atoms = {metal for bound in metals.values() for metal in bound}
    ranks = list(Chem.ComputeAtomCIPRanks(without_atom_maps(mol)))
    out = {}
    for pair, wanted in stereo.bonds.items():
        donors = [atom for atom in pair if atom in metals]
        if len(donors) != 1:
            continue
        donor = donors[0]
        if len(metals[donor]) != 1:
            continue
        other = next(iter(pair - {donor}))
        ligand_refs = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(donor).GetNeighbors()
            if nb.GetIdx() != other and nb.GetIdx() not in metal_atoms
        ]
        if len(ligand_refs) > 1 or any(mol.GetAtomWithIdx(ref).GetAtomicNum() > 1 for ref in ligand_refs):
            continue
        other_refs = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(other).GetNeighbors()
            if nb.GetIdx() != donor and nb.GetIdx() not in metal_atoms
        ]
        if not other_refs:
            continue
        high = max(ranks[ref] for ref in other_refs)
        controls = [ref for ref in other_refs if ranks[ref] == high]
        if len(controls) == 1:
            out[pair] = MetalEZ(
                donor=donor,
                other=other,
                metal=next(iter(metals[donor])),
                ref=controls[0],
                ligand_ref=ligand_refs[0] if ligand_refs else None,
                wanted=wanted,
            )
    return out


def _add_point_umbrellas(cons, mol, stereo, donor_bonds):
    """Return `cons` keeping each retained tetrahedral point in its seeded signed-volume half-space during UFF."""
    metals, umbrellas = {}, dict(cons.umbrellas)
    for donor, metal in donor_bonds:
        metals.setdefault(donor, []).append(metal)
    for centre in stereo.points:
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
            umbrellas.setdefault(key, Improper(kind="side"))
    return cons if umbrellas == cons.umbrellas else replace(cons, umbrellas=umbrellas)


def _add_ligand_ez(cons, mol, stereo, donor_bonds):
    """Return `cons` keeping specified ligand double bonds inside their RDKit-reference half-space during UFF."""
    metal_owned = set(metal_referenced_ez(mol, stereo, donor_bonds))
    coplanar = list(cons.coplanar)
    for pair in stereo.bonds:
        if pair in metal_owned:
            continue
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None or len(bond.GetStereoAtoms()) != 2:  # noqa: PLR2004 - a double bond has two stereo atoms
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        refs = bond.GetStereoAtoms()
        if (relation := EZ_RELATION.get(bond.GetStereo())) is None:
            continue
        anchor = 0.0 if relation == "cis" else STRAIGHT
        row = CoplanarRow(kind="ligand_ez", atoms=(refs[0], begin, end, refs[1]), anchor=anchor, cap=COPLANAR_CAP)
        if not graft_owns(row.atoms, cons.frozen):
            coplanar.append(row)
    return cons if tuple(coplanar) == cons.coplanar else replace(cons, coplanar=coplanar)


def _add_metal_ez(cons, mol, stereo, donor_bonds):
    """Return `cons` choosing the donor-plane well when a metal is an imine's missing E/Z reference."""
    coplanar = list(cons.coplanar)
    for ez in metal_referenced_ez(mol, stereo, donor_bonds).values():
        anchor = 0.0 if ez.wanted == "Z" else STRAIGHT
        existing = [row for row in coplanar if row.seeded and row.atoms[:3] == (ez.metal, ez.donor, ez.other)]
        cap = existing[0].cap if existing else COPLANAR_CAP
        if existing:
            coplanar.remove(existing[0])
        rows = [CoplanarRow(kind="metal_ez", atoms=(ez.metal, ez.donor, ez.other, ez.ref), anchor=anchor, cap=cap)]
        if ez.ligand_ref is not None:
            atoms = (ez.ligand_ref, ez.donor, ez.other, ez.ref)
            rows.append(CoplanarRow(kind="ligand_ez", atoms=atoms, anchor=STRAIGHT - anchor, cap=cap))
        coplanar += [row for row in rows if not graft_owns(row.atoms, cons.frozen)]
    return cons if tuple(coplanar) == cons.coplanar else replace(cons, coplanar=coplanar)


_EXHAUSTIVE_BITE_CORNERS = 4  # 2**k corners is a resource ceiling; above it only extremes plus single flips are tried


def _bite_reach(left, right, span):
    """Return the donor-donor angle range (degrees) two fixed model M-D legs allow at a `span` (Å) range.

    A fixed leg is a degenerate (x, x) interval for `triangle_angles`; where no triangle exists the law of cosines gives
    the row instead.
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
    legs = [0.5 * sum(distances[canonical_key((a, metal))]) for a in (left, right)]
    return _bite_reach(legs[0], legs[1], (floor, legs[0] + legs[1]))[0]


def bite_window_and_reach(facts, left, right, donors, lengths, row):
    """Return (window, reach) for a same-ligand bite pair, or None where neither narrows the row.

    `reach` is the triangle the two model M-D legs and the native ligand reach allow. A sigma pair also has a calibrated
    ring-size window (`chelate_bite_window`), clamped inside `reach`, or `reach` alone where the two miss each other; no
    calibrated row means no bite. A haptic pair has none either, so `reach` replaces the polyhedron angle where
    narrower.
    """
    prior = None
    if len(left) == 1 and len(right) == 1:
        prior = chelate_bite_window(facts.mol, left[0], right[0], donors)
        if prior is None:
            return None
    matrix = facts.native_reach
    if matrix is None:  # ligand_reach could not close a triangle for this graph
        return (prior, prior) if prior is not None else None
    left_leg, right_leg = map(float, lengths)
    lower, upper = _site_span(matrix, left, right)
    reach = _bite_reach(left_leg, right_leg, (max(0.0, lower - SPAN_TOL), upper + SPAN_TOL))
    if prior is not None:
        # judge agreement on the backbone triangle itself; SPAN_TOL is numerical room, not an overlap
        exact = _bite_reach(left_leg, right_leg, (lower, upper))
        if exact[1] < prior[0] or exact[0] > prior[1]:
            return reach, reach
        return (max(prior[0], reach[0]), min(prior[1], reach[1])), reach
    if reach[0] <= row[0] and reach[1] >= row[1]:
        return None
    return reach, reach


def seated_bites(sphere, distances):
    """Return each same-ligand donor pair's (window, reach) angle windows, keyed by its seated site pair.

    Each is drawn about the polyhedron's own pair angle; a trans pair (`Polyhedron.trans_pairs`) gets none.
    """
    metal, od, haptic, facts = sphere.metal, sphere.vertices, sphere.haptic, sphere.facts
    wide, real = sphere.poly.trans_pairs, frozenset(sphere.real)  # a bite's calibrated window reads the co-donors
    bites = {}
    for i, j in itertools.combinations(range(len(od)), 2):
        left, right = od[i], od[j]
        pair = frozenset((i, j))
        if VACANT in (left, right) or sphere.frag_of(left) != sphere.frag_of(right) or (i, j) in wide:
            continue
        keys = (canonical_key((metal, left)), canonical_key((metal, right)))
        lengths = (0.5 * sum(distances[keys[0]]), 0.5 * sum(distances[keys[1]]))
        left_site, right_site = haptic.get(left, (left,)), haptic.get(right, (right,))
        base = sphere.poly.pair_angles[pair]
        row = (max(0.0, base - ANGLE_PAD), min(STRAIGHT, base + ANGLE_PAD))
        bite_key = ("bite", tuple(sorted(left_site)), tuple(sorted(right_site)), lengths, row, real)
        if bite_key not in facts.memo:
            facts.memo[bite_key] = bite_window_and_reach(facts, left_site, right_site, real, lengths, row)
        if (bite := facts.memo[bite_key]) is not None:
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


def bounded_bites(poly, bites):
    """Bound bite windows by relaxed shells that still read as `poly`, or return None.

    Ideal angles clamped to the initial bite windows give an anchor; if it reads as `poly`, the windows shrink toward it
    using the joint corner samples. Otherwise ideal angles clamped to the wider backbone reach are tried, opening the
    windows between the anchors. None does not prove the backbone infeasible, and an approximate shape reading leaves
    the windows unchanged. These readings guide construction; the generated coordinates are checked separately.
    """
    if not bites:
        return bites
    if not _shell_reading(poly, ()).exact:  # an approximate reading ranks shapes; it cannot narrow a wall
        return {pair: window for pair, (window, _reach) in bites.items()}
    key = tuple(
        sorted(
            ((pair, *bites[pair], poly.pair_angles[pair]) for pair in bites),
            key=lambda item: tuple(sorted(item[0])),
        )
    )
    return _bounded_bites_cached(poly, key)


def _corner_key(corner):
    """Return `corner`'s content key, in the same form `polyhedra.relaxed_shell` builds internally."""
    return tuple(sorted((tuple(sorted(pair)), float(target)) for pair, target in corner.items()))


@lru_cache(maxsize=FLOAT_CACHE)
def _shell_reading(poly, corner_key):
    """Return how the shell its bite targets `corner_key` relax to reads against `poly`; cached across rounds."""
    return shape_reading(relaxed_shell(poly, {frozenset(pair): target for pair, target in corner_key}), poly.name)


def _shrunk_windows(windows, anchor, t):
    """Return each bite window shrunk toward its anchor target by the fraction `t`."""
    return {pair: (lo + t * (anchor[pair] - lo), hi - t * (hi - anchor[pair])) for pair, (lo, hi) in windows.items()}


@lru_cache(maxsize=FLOAT_CACHE)
def _bounded_bites_cached(poly, key):
    anchor = {pair: min(max(ideal, lo), hi) for pair, (lo, hi), _reach, ideal in key}
    windows = {pair: (lo, hi) for pair, (lo, hi), _reach, _ideal in key}
    far = {pair: min(max(ideal, rlo), rhi) for pair, _window, (rlo, rhi), ideal in key}
    if not _shell_reading(poly, _corner_key(anchor)).matches:
        if _shell_reading(poly, _corner_key(far)).matches:
            return {pair: (min(anchor[pair], far[pair]), max(anchor[pair], far[pair])) for pair in windows}
        return None
    if all(_shell_reading(poly, _corner_key(corner)).matches for corner in _bite_corners(windows)):
        return windows
    good, bad = 1.0, 0.0  # t=1 is the anchor itself (degenerate, always reads correctly); t=0 the full window
    for _ in range(_BITE_BOX_STEPS):
        mid = 0.5 * (good + bad)
        box = _shrunk_windows(windows, anchor, mid)
        if all(_shell_reading(poly, _corner_key(corner)).matches for corner in _bite_corners(box)):
            good = mid
        else:
            bad = mid
    return _shrunk_windows(windows, anchor, good)


def _radial_windows(sphere):
    """Return each donor's M-D window, its substituent walls off the metal, and a small chelate ring's flat hinge."""
    mol, metal, facts = sphere.mol, sphere.metal, sphere.facts
    sites = haptic_sites(mol, sphere.real)  # once per sphere, for `ml_distance`'s `eta`
    eta = {d: len(site) for site in sites if len(site) >= ETA2 for d in site}
    qdel = facts.charges
    bound = facts.bound
    hyb = facts.hybridisation
    stripped = facts.stripped if sphere.donor_orientation else None  # one graph for every donor's `orient_donor`
    distances, pulls, angles, coplanar, haptic = {}, {}, {}, [], {}
    for d in sphere.vertices:
        if d is VACANT:
            continue
        if d in sphere.haptic:  # a centroid vertex: pin its whole ring, not a single donor (no orient/coplanar)
            face = _centroid_constraints(sphere, d, qdel, hyb)
            distances.update(face.distances)
            pulls.update(face.pulls)
            haptic.update(face.haptic)
            continue
        length = _donor_length(sphere, d, qdel, hyb, eta.get(d, 0))
        add_distance(distances, metal, d, *_donor_window(sphere, length))
        # a bridging H's X-H is pulled to `BRIDGE_H` in the FF only: a window would contradict its native bound
        for nb in bridge_partners(mol, d):
            x = nb.GetIdx()
            measured = sphere.pos is not None
            leg = np.linalg.norm(sphere.pos[x] - sphere.pos[d]) if measured else BRIDGE_H[nb.GetAtomicNum()][1]
            pulls[canonical_key((x, d))] = (float(leg),) * 2
        # wall each donor substituent off the metal, whatever the lengths: no real energy holds that axis
        if sphere.donor_orientation:
            angles.update(orient_donor(mol, metal, d, sphere.real, hyb=hyb, stripped=stripped, metals=bound[d]))
            coplanar += coplanar_donor(mol, metal, d, sphere.real, hyb=hyb)
    if sphere.donor_orientation:
        coplanar += ring_hinge(mol, metal, sphere.sigma)
    return Constraints(distances=distances, pulls=pulls, angles=angles, coplanar=coplanar, haptic=haptic)


def _bonded_donor_windows(sphere, distances):
    """Widen a bonded donor pair's M-D windows so the two legs can span its own bond.

    Both upper bounds scale until, at the polyhedron's pair angle, the legs reach RDKit's native upper bound for the
    bond.
    """
    if sphere.pos is not None:
        return
    mol, metal, od, angles = sphere.mol, sphere.metal, sphere.vertices, sphere.poly.pair_angles
    expanded = {}
    for i, j in itertools.combinations(range(len(od)), 2):
        left, right = od[i], od[j]
        if (
            VACANT in (left, right)
            or left in sphere.haptic
            or right in sphere.haptic
            or mol.GetBondBetweenAtoms(left, right) is None
        ):
            continue
        left_key = canonical_key((left, metal))
        right_key = canonical_key((right, metal))
        bond_bounds = sphere.facts.bounds
        left_window, right_window = distances[left_key], distances[right_key]
        span = law_of_cosines(left_window[1], right_window[1], angles[frozenset((i, j))])
        native_upper = float(bond_bounds[min(left, right)][max(left, right)])
        if native_upper > span > 0:
            scale = native_upper / span
            expanded[left_key] = max(expanded.get(left_key, 0.0), left_window[1] * scale)
            expanded[right_key] = max(expanded.get(right_key, 0.0), right_window[1] * scale)
    for key, upper in expanded.items():
        distances[key] = (distances[key][0], upper)


def _chelate_bite_images(sphere, bites):
    """Return bite windows, sampled corner images and shell pull targets for angle compilation.

    `bounded_bites` runs when every vertex is occupied and no metal or bitten ligand atom is held elsewhere; otherwise,
    or when it fails, the original windows stay. Usable corner shells give the spectator windows; the midpoint shell
    gives weak pulls, else the usable corner nearest it, else none.
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
        bites = bounded_bites(sphere.poly, bites) or windows
        corners = []
        for corner in _bite_corners(bites):
            rays = relaxed_shell(sphere.poly, corner)
            if rays is not None:
                corners.append((corner, rays))
                corner_images.append(np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0))))
        midpoint = {pair: 0.5 * sum(window) for pair, window in bites.items()}
        mid_rays = relaxed_shell(sphere.poly, midpoint)
        if mid_rays is None and corners:
            # the midpoint changes this template's type; pull to the nearest surviving corner rather than none
            mid_rays = min(
                corners, key=lambda item: sum((item[0][pair] - target) ** 2 for pair, target in midpoint.items())
            )[1]
        if mid_rays is not None:
            mid_angles = np.degrees(np.arccos(np.clip(mid_rays @ mid_rays.T, -1.0, 1.0)))
    else:
        bites = windows
    return bites, corner_images, mid_angles


def _assign_angle_rows(sphere, bites, corner_images, mid_angles, c):
    """Return `c` with each vertex pair compiled to a bonded D-D, bite or spectator window, plus the shell pull.

    An angle to an empty vertex stays free: a tripod still holds its own geometry through its backbone. A spectator row
    is the union of the shells the bites can build, padded past the nearest one; between two ligands it can narrow, but
    never so far that their donors clash inside their contact floor.
    """
    mol, metal, od, haptic, pos = sphere.mol, sphere.metal, sphere.vertices, sphere.haptic, sphere.pos
    distances, pulls, angles = dict(c.distances), dict(c.pulls), dict(c.angles)
    for pair, a in sphere.poly.pair_angles.items():
        i, j = sorted(pair)
        if od[i] is VACANT or od[j] is VACANT:
            continue
        # the D-D bond and two M-D windows define this triangle; an ideal angle could only tear the small ring
        if od[i] not in haptic and od[j] not in haptic and mol.GetBondBetweenAtoms(od[i], od[j]) is not None:
            left, right = sorted((od[i], od[j]))
            bridged = {nb.GetIdx() for atom in (left, right) for nb in bridge_partners(mol, atom)} & {left, right}
            if pos is not None and bridged:
                # an X-H-M bridge has a long X-H; RDKit's terminal bounds would contract it, so the measured span wins
                distance = float(np.linalg.norm(pos[left] - pos[right]))
                add_distance(distances, left, right, *_donor_window(sphere, distance))
                pulls[(left, right)] = (distance, distance)
            else:
                bond_bounds = sphere.facts.bounds
                add_distance(distances, left, right, bond_bounds[right, left], bond_bounds[left, right])
            continue
        key = (od[i], metal, od[j])
        if pair in bites:
            angles[key] = bites[pair]
        else:
            lo, hi = max(0.0, a - ANGLE_PAD), min(STRAIGHT, a + ANGLE_PAD)
            if corner_images:  # union with the shells, then no further than the pad from the nearest shell
                values = [image[i, j] for image in corner_images]
                union = min(lo, *values), max(hi, *values)
                lo = max(union[0], min(values) - ANGLE_PAD)
                hi = min(union[1], max(values) + ANGLE_PAD)
                if od[i] not in haptic and od[j] not in haptic and sphere.frag_of(od[i]) != sphere.frag_of(od[j]):
                    # a row from another ligand narrows, but not below its donors' contact floor nor past the union
                    hi = max(hi, min(union[1], _contact_angle(mol, metal, od[i], od[j], distances)))
            angles[key] = (lo, hi)
        # other in-window terms are flat walls, so pull every row: pulling only moved rows lets another reading win
        if mid_angles is not None:
            canonical = min(key, key[::-1])
            pulls.pop(canonical, None)
            pulls.pop(canonical[::-1], None)
            pulls[canonical] = (float(mid_angles[i, j]),) * 2
    return replace(c, distances=distances, pulls=pulls, angles=angles)


def coordination(sphere):
    """Build one centre's distance, angle, floor and umbrella constraints.

    Cis chelates take calibrated ring-size bite windows; every other angle row narrows toward the shells those bites can
    build, else takes the polyhedron's ideal angle. The full model is compiled before the terms the frozen atoms own are
    dropped.
    """
    c = _radial_windows(sphere)
    distances = dict(c.distances)
    _bonded_donor_windows(sphere, distances)
    bites = seated_bites(sphere, distances)
    bites, corner_images, mid_angles = _chelate_bite_images(sphere, bites)
    c = _assign_angle_rows(sphere, bites, corner_images, mid_angles, replace(c, distances=distances))
    # real atoms, never a centroid, so no face atom is floored; an eta2 bond needs no hold beyond pull and cone
    coord = [d for d in sphere.vertices if d is not VACANT and d not in sphere.haptic]
    coord += [a for site in sphere.haptic.values() for a in site]
    c = ff_terms(
        sphere.mol,
        c,
        sphere.metal,
        sphere.real_z,
        coord,
        frozen=sphere.frozen,
        fragments=sphere.facts.fragments,
        topology=sphere.facts.topology,
    )
    c = replace(c, umbrellas=_umbrellas(sphere.metal, sphere.vertices, sphere.poly))
    return _drop_graft_owned(c, sphere.frozen, sphere.haptic)


def _donor_length(sphere, donor, charges, hyb, eta):
    """Return the M-donor length a sigma donor or face atom is held about: measured, else `ml_distance`'s model.

    The length is measured on `sphere.pos` under ``lengths='input'``; the model takes the supplied hapticity `eta`.
    """
    if sphere.pos is not None:
        return float(np.linalg.norm(sphere.pos[sphere.metal] - sphere.pos[donor]))
    return ml_distance(sphere.mol, donor, sphere.real_z, charges=charges, hyb=hyb, eta=eta)


def _donor_window(sphere, target):
    """Return the window `sphere` holds a `target` length in: a donor's, face atom's, centroid's or bridge X-H's.

    A measured length (``lengths='input'``) gets `INPUT_HALF_WIDTH` either side, a model one `_ML_SEED_HALF_WIDTH`.
    """
    half = _ML_SEED_HALF_WIDTH if sphere.pos is None else INPUT_HALF_WIDTH
    return target - half, target + half


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
        if not graft_owns(row.atoms, frozen, haptic):
            active.update(virtual.intersection(row.atoms))

    def keep(key):
        # A live virtual site is not itself grafted. Keep the terms of its transient centroid intact.
        return bool(active.intersection(key)) or not graft_owns(key, frozen, haptic)

    names = ("distances", "angles", "dihedrals", "pulls", "umbrellas", "floors", "dg_floors")
    return replace(
        cons,
        **{name: {key: value for key, value in getattr(cons, name).items() if keep(key)} for name in names},
        coplanar=[row for row in cons.coplanar if keep(row.atoms)],
        haptic={dummy: face for dummy, face in cons.haptic.items() if dummy in active},
    )


def _umbrellas(metal, vertices, poly):
    """Return the radial impropers covering a planar shell, or the three-donor pyramid hold."""
    umbrellas = {}
    if poly.umbrella_improper is None and not poly.planar:
        return umbrellas
    occupied = [d for d in vertices if d is not VACANT]
    if poly.planar and len(occupied) > IMPROPER_VERTICES:
        directions = {
            atom: np.asarray(poly.vertex_dirs[i], float) for i, atom in enumerate(vertices) if atom is not VACANT
        }
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
            # both planes contain M, so rescaling keeps the well; average tied axes, and a fix reweights no row
            for key in axes:
                left, _, middle, right = key
                phi = improper_dihedral(directions[left], np.zeros(3), directions[middle], directions[right])
                anchor = 0.0 if abs(phi) < STRAIGHT / 2 else STRAIGHT
                umbrellas[key] = Improper(kind="plane", anchor=anchor, weight=1.0 / (len(triples) * len(axes)))
    elif poly.planar and len(occupied) >= IMPROPER_VERTICES:
        umbrellas[(*occupied[:IMPROPER_VERTICES], metal)] = Improper(kind="plane")
    elif len(occupied) == len(vertices) == IMPROPER_VERTICES:
        umbrellas[(*occupied, metal)] = Improper(kind="pyramid", minimum=poly.umbrella_improper)
    return umbrellas
