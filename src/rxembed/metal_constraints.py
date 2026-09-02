"""An `Isomer` -> the `Constraints` that hold its coordination sphere: M-donor windows and L-M-L angles.

The top of the metal stack: it reads `metal_distance` and `metal_donor_orient`, and nothing reads it back.
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem

from .constraints import Constraints, _graft_owns, add_distance, compose
from .metal_core import (
    VACANT,
    _frag_map,
    _regular_face,
    _site_radius,
    _vertex_atom,
    materialized_states,
)
from .metal_distance import delocalised_charges, ff_terms, ml_distance
from .metal_donor_orient import _coplanar_donor, _orient_donor, _stripped_hybridisation
from .metal_polyhedron import _IMPROPER_VERTICES, CHELATE_SPAN_ANGLE, POLYHEDRA, _vertex_angle, resolve_geometry
from .utils import _DISCONNECTED

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval


def _centroid_constraints(mol, metal, dummy, ring, real_z, *, qdel, c, pos, hyb, source=None):
    """Constrain one haptic face through a transient centroid vertex.

    A regular face uses its circumradius and metal-ring distances to define a cone. An irregular face keeps only
    its metal-ring distances so a bent allyl or diene can relax without being flattened.
    """
    r = _site_radius(source if source is not None else mol, ring)  # circumradius of the face
    if pos is not None:  # realised metal->ring-atom distances from the input geometry
        mcs = [float(np.linalg.norm(pos[metal] - pos[a])) for a in ring]
    else:  # the fitted model; the ring is its own co-donor set so the hapticity term sees the whole face
        mcs = [ml_distance(mol, metal, a, real_z, set(ring), charges=qdel, hyb=hyb) for a in ring]
    mc = float(np.mean(mcs))
    d_c = float(np.sqrt(max(mc * mc - r * r, 0.25)))
    add_distance(  # M -> centroid vertex, both modes
        c.distances, metal, dummy, d_c - _ML_SEED_HALF_WIDTH, d_c + _ML_SEED_HALF_WIDTH
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


LENGTHS = ("auto", "input", "model")  # where an M-L window comes from; see `resolve_lengths`


def resolve_lengths(mol, lengths="auto"):
    """Resolve M-L windows to input coordinates or the fitted distance model.

    ``'auto'`` uses a conformer when present, otherwise the model. ``'input'`` requires coordinates and
    ``'model'`` ignores them. The note makes an automatic coordinate choice visible to the caller.
    """
    if lengths not in LENGTHS:
        raise ValueError(f"lengths={lengths!r}; expected one of {LENGTHS}")
    has_geometry = mol.GetNumConformers() > 0
    if lengths == "model":
        return None, "the fitted periodic model (lengths='model')"
    if lengths == "input":
        if not has_geometry:
            raise ValueError(
                "lengths='input' measures each M-L window off the input conformer, and this molecule has none "
                "(a SMILES carries no geometry); use lengths='model' for the fitted periodic model"
            )
        return mol.GetConformer().GetPositions(), "the input conformer (lengths='input')"
    if not has_geometry:
        return None, ""  # the ordinary case (a SMILES, the fitted model); announcing the default is noise
    return mol.GetConformer().GetPositions(), "the input conformer; use lengths='model' if it is not metal-aware"


def compile_constraints(mol, centres, *, length_mol, base, constrained_metals, lengths):
    """Compile selected metal states onto a copy of `base`.

    `mol` supplies current topology and atom indexing; `length_mol` preserves the input geometry used for lengths.
    """
    if not constrained_metals:
        return compose(base)
    parts = materialized_states(mol, centres)
    built = []
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
            )
        )
    return compose(*built, base)


def coordination(
    mol,
    metal,
    vertices,
    geometry,
    real_z,
    *,
    haptic,
    frozen=(),
    lengths="auto",
    source=None,
    distance_overrides=None,
):
    """Build one metal state's distance, angle, floor and umbrella constraints.

    Inter-ligand pairs and trans chelates follow the ideal polyhedron; cis chelates use calibrated ring-size bite
    windows. Chemistry is compiled before `_drop_graft_owned` removes only complete terms already determined by
    `fix`. `source` preserves input-length measurements while `mol` supplies topology.
    """
    frag = _frag_map(mol)  # same ligand = same fragment
    poly = POLYHEDRA[resolve_geometry(geometry)]

    def frag_of(v):  # a face tethered to a co-donor (η²-alkyne + alkyl of one metallacycle; an ansa Cp) must read
        return frag[_vertex_atom(haptic, v)]  # as one ligand: the bond-less centroid carries no fragment of its own

    pos, _note = resolve_lengths(source if source is not None else mol, lengths)
    distance_overrides = distance_overrides or {}
    c = Constraints()
    od = list(vertices)
    real_od = {x for x in od if x != VACANT and x not in haptic}
    real_od.update(a for face in haptic.values() for a in face)  # real co-donors; centroid keys are not Mol atoms
    qdel = delocalised_charges(mol)  # a Lewis charge is an artefact, so spread it
    # The model needs the metal-stripped class because RDKit counts a dative bond. Input lengths do not.
    hyb = _stripped_hybridisation(mol) if pos is None else None
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
                source=source,
            )
            continue
        key = (min(metal, d), max(metal, d))
        if key in distance_overrides:
            add_distance(c.distances, metal, d, *distance_overrides[key])
        elif pos is not None:  # measured: this structure's value, and wider (±0.1) because it is one sample
            d_md = float(np.linalg.norm(pos[metal] - pos[d]))
            add_distance(c.distances, metal, d, d_md - 0.1, d_md + 0.1)
        else:  # the fitted periodic model (element, group, delocalised charge, hapticity)
            add_distance(
                c.distances,
                metal,
                d,
                *_model_distance_window(mol, metal, d, real_z, real_od, charges=qdel, hyb=hyb),
            )
        # Wall each donor substituent off the metal, the orientation hold a real energy cannot supply itself.
        # Length provenance is independent: measured M-L distances do not determine a partially free M-D-X axis.
        _orient_donor(mol, metal, d, real_od, c)
        # cap an sp2 donor's metal at the donor's own sp2 plane: the improper the stripped bond removed.
        _coplanar_donor(mol, metal, d, c)
    angle_rows = list(poly.resolved_angles)
    stated_pairs = {frozenset((i, j)) for i, j, _a in angle_rows}
    if metal not in frozen:  # fixed donors do not fix their angle about a free metal
        for i, j in itertools.combinations(range(len(od)), 2):
            if (
                VACANT not in (od[i], od[j])
                and frozenset((i, j)) not in stated_pairs
                and _graft_owns((od[i], od[j]), frozen, haptic)
            ):
                angle_rows.append((i, j, _vertex_angle(poly.vertex_dirs[i], poly.vertex_dirs[j])))
    for i, j, a in angle_rows:
        if (
            od[i] == VACANT or od[j] == VACANT
        ):  # an angle to an empty vertex is unconstrained. NB a shape reached AS a vacancy is stated more weakly than
            # one with its own record: dropping a vertex drops every row naming it. Still right, because a tripod pulls
            # its geometry through its backbone, which `_chelate_bite_window` models for ring sizes 4/5/6.
            continue
        intra = frag_of(od[i]) == frag_of(od[j])  # two donors of one chelating ligand
        if intra and a < CHELATE_SPAN_ANGLE:
            # A cis chelate reads its bite from calibrated 4/5/6-membered backbones. Smaller, larger and haptic
            # partners stay unconstrained because the polyhedron angle is a poor model for those ligands.
            bite = _chelate_bite_window(mol, od[i], od[j])
            if bite is not None:
                c.angles[(od[i], metal, od[j])] = bite
            continue
        # Trans feasibility belongs to slot enumeration; a retained or explicitly stated pair keeps the shape.
        pad = 8.0
        c.angles[(od[i], metal, od[j])] = (max(0.0, a - pad), min(180.0, a + pad))
    # NB an η² π bond needs no hold of its own: the face is a centroid vertex, so one axial pull plus the cone
    # pins both π atoms at the face radius. Two separate M-donor pulls tore C≡C from 1.2 to 1.7 Å.
    coord = [d for d in od if d != VACANT and d not in haptic]
    coord += [a for site in haptic.values() for a in site]  # real coordinating atoms; centroid keys are reserved
    ff_terms(mol, c, {metal: (real_z, coord)})  # coordinating atoms, so nondonor_floors never floors a ring atom
    _add_umbrella(c, metal, od, poly, haptic, frag)
    return _drop_graft_owned(c, frozen, haptic)


def _model_distance_window(mol, metal, donor, real_z, donors, *, charges=None, hyb=None):
    """Return the fitted M-donor seed window, using shared graph-derived model inputs."""
    target = ml_distance(
        mol,
        metal,
        donor,
        real_z,
        donors,
        charges=delocalised_charges(mol) if charges is None else charges,
        hyb=_stripped_hybridisation(mol) if hyb is None else hyb,
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


def _add_umbrella(cons, metal, vertices, poly, haptic, fragments):
    """Add the shared planar or pyramidal metal improper when three independent donors define it."""
    if poly.umbrella_improper is None and not poly.planar:
        return
    real = (d for d in vertices if d != VACANT and d not in haptic)
    for base in itertools.combinations(real, _IMPROPER_VERTICES):
        if len({fragments[d] for d in base}) > 1:
            cons.umbrellas[(*base, metal)] = poly.umbrella_improper
            return


_CHELATE_BITE = {4: (58.0, 81.0), 5: (70.0, 91.0), 6: (74.0, 104.0)}  # chelate ring size -> M-D-D bite window (deg):
#   the census range that admits every real bite and forbids the fold. The LIGAND's ring sets the bite, not the
#   metal's polytope: a 5-ring bites ~80°, not the ideal 90°, so the polyhedron vertex angle predicts poorly.


def _chelate_bite_window(mol, a, b):
    """Return a chelate bite window from the backbone path plus the two metal-donor bonds.

    The surrogate strips the M-donor bonds, so the chelate ring is open, the metal is bond-less, and the two donors
    relate only through the backbone: ring size = that backbone path + 2, read straight off `mol`'s own (cached)
    topological distances. Returns ``None`` for a virtual site, non-chelate, or ring outside the calibrated 4/5/6
    domain (3-membered / floppy 7+).
    """
    if not 0 <= a < mol.GetNumAtoms() or not 0 <= b < mol.GetNumAtoms():
        return None  # a haptic centroid is a coordination site, not an atom with a ligand-backbone path
    path = float(Chem.GetDistanceMatrix(mol)[a][b])
    if path > _DISCONNECTED:  # different fragments -> not a chelate
        return None
    return _CHELATE_BITE.get(int(path) + 2)


def coordination_from_geometry(mol, metal, vertices, geometry, real_z, haptic):
    """Build coordination constraints from the actual input geometry: the specific ligand arrangement.

    Compile the same graph-derived chemistry as an enumerated state, then replace only its polyhedral angle
    windows with the realised donor-metal-donor angles. Retaining a measured arrangement changes the numbers,
    not whether donor orientation, coplanarity, floors or force-field terms exist.

    A haptic face remains one centroid site, so measured angles follow coordination vertices rather than each
    ring atom independently.
    """
    c = coordination(
        mol,
        metal,
        vertices,
        geometry,
        real_z,
        haptic=haptic,
        lengths="input",
        source=mol,
    )
    pos = mol.GetConformer().GetPositions()

    def position(vertex):
        return np.mean(pos[list(haptic[vertex])], axis=0) if vertex in haptic else pos[vertex]

    occupied = [vertex for vertex in vertices if vertex != VACANT]
    for a, b in itertools.combinations(occupied, 2):
        ang = _vertex_angle(position(a) - pos[metal], position(b) - pos[metal])
        c.angles[(a, metal, b)] = (max(0.0, ang - 8), min(180.0, ang + 8))
    return c
