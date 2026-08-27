"""An `Isomer` -> the `Constraints` that hold its coordination sphere: M-donor windows and L-M-L angles.

The top of the metal stack: it reads `metal_distance` and `metal_donor_orient`, and nothing reads it back.
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem

from .constraints import Constraints, add_distance, compose
from .metal_core import (
    VACANT,
    _frag_map,
    _regular_face,
    _site_radius,
    _vertex_atom,
    materialized_states,
)
from .metal_distance import (
    _FLOOR_REACH,
    _tier_floor,
    delocalised_charges,
    ff_terms,
    ml_distance,
    overbond_tier,
)
from .metal_donor_orient import _coplanar_donor, _orient_donor, _stripped_hybridisation
from .metal_polyhedron import _IMPROPER_VERTICES, CHELATE_SPAN_ANGLE, POLYHEDRA, _vertex_angle, resolve_geometry
from .utils import _DISCONNECTED, _PT

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval


def _centroid_constraints(mol, metal, dummy, ring, real_z, *, qdel, c, pos, hyb, source=None):
    """Pin a haptic face by its centroid dummy: the one polyhedron vertex an η² alkene, Cp or arene presents.

    The metal->ring-atom distance is ``mc`` and the dummy sits ``d_c = sqrt(mc^2 - r^2)`` up the axis, exact for any
    face whose atoms share a circumradius, η² included (two points lie on a circle about their midpoint, and equal
    M-C puts M on the perpendicular; measured on Zeise's salt: model 2.018 Å vs true 2.018). A REGULAR face
    (`_regular_face`: a bond, Cp, arene, cyclobutadiene) is held in cone mode, each ring atom also pinned to the
    centroid at the radius ``r``. An irregular face (allyl / diene / pentadienyl) drops those pins (METAL mode) so
    the bent face relaxes freely; the rigid cone over-constrains it into a torn geometry (the metal-mode
    fallback). The dummy is a ``phantom``, zero vdW in the FF with its DG floor relaxed: it lives inside its ring.
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
    cone = _regular_face(mol, ring)  # a regular face is equidistant from its own centroid -> CONE mode; an
    #   irregular one (allyl/diene/pentadienyl) is floppy AND not centroid-equidistant -> METAL mode.
    for a, mca in zip(ring, mcs, strict=True):
        add_distance(
            c.distances, metal, a, mca - _ML_SEED_HALF_WIDTH, mca + _ML_SEED_HALF_WIDTH
        )  # hold each ring atom at its metal distance
        if cone:  # a rigid regular n-gon (Cp/arene): also pin each ring atom to the centroid at the ring radius
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
                frozen=base.frozen & set(vertices),
                core_frozen=base.frozen,
                lengths=lengths,
                source=length_mol,
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
    core_frozen=(),
    lengths="auto",
    source=None,
):
    """Build one metal state's distance, angle, floor and umbrella constraints.

    Frozen donor pairs need no duplicate angle. Inter-ligand pairs follow the ideal polyhedron; cis chelates
    use calibrated ring-size bite windows and trans chelates remain wide. `source` preserves input-length
    measurements while `mol` supplies the current topology.
    """
    frag = _frag_map(mol)  # same ligand = same fragment
    poly = POLYHEDRA[resolve_geometry(geometry)]

    def frag_of(v):  # a face tethered to a co-donor (η²-alkyne + alkyl of one metallacycle; an ansa Cp) must read
        return frag[_vertex_atom(haptic, v)]  # as one ligand: the bond-less centroid carries no fragment of its own

    pos, _note = resolve_lengths(source if source is not None else mol, lengths)
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
        if pos is not None:  # measured: this structure's value, and wider (±0.1) because it is one sample
            d_md = float(np.linalg.norm(pos[metal] - pos[d]))
            add_distance(c.distances, metal, d, d_md - 0.1, d_md + 0.1)
        else:  # the fitted periodic model (element, group, delocalised charge, hapticity)
            t = ml_distance(mol, metal, d, real_z, real_od, charges=qdel, hyb=hyb)  # see `ml_distance`
            add_distance(c.distances, metal, d, t - _ML_SEED_HALF_WIDTH, t + _ML_SEED_HALF_WIDTH)
        # Wall each donor substituent off the metal, the orientation hold a real energy cannot supply itself.
        # `pos` decides whether to: the input geometry is the ORIENTATION truth exactly when it is the DISTANCE
        # truth. Measured, a frozen metal already states where its donors point and a wall would fight it;
        # modelled, it states nothing, and skipping leaves a rebuilt ligand unoriented (a carbonyl added to a
        # held sphere relaxed to M-C-O 129-148°, against its own 155° gate).
        if pos is None or metal not in core_frozen:
            _orient_donor(mol, metal, d, real_od, c, core_frozen)
        # cap an sp2 donor's metal at the donor's own sp2 plane: the improper the stripped bond removed.
        # Skip when the metal or this donor is frozen: a fix= TS core already pins that M-donor geometry at the
        # input, and biasing it toward coplanar only fights the freeze.
        if metal not in core_frozen and d not in core_frozen:
            _coplanar_donor(mol, metal, d, c)
    for i, j, a in poly.resolved_angles:
        if (
            od[i] == VACANT or od[j] == VACANT
        ):  # an angle to an empty vertex is unconstrained. NB a shape reached AS a vacancy is stated more weakly than
            # one with its own record: dropping a vertex drops every row naming it. Still right, because a tripod pulls
            # its geometry through its backbone, which `_chelate_bite_window` models for ring sizes 4/5/6.
            continue
        if od[i] in frozen and od[j] in frozen:  # both held by freeze -> don't over-determine the core
            continue
        intra = frag_of(od[i]) == frag_of(od[j])  # two donors of one chelating ligand
        if intra and a < CHELATE_SPAN_ANGLE:  # a *cis* chelate: the ideal polyhedron angle predicts the bite poorly,
            # and a backbone-only distance lets the DG fold it shut, so pin it from the ring SIZE where
            # calibrated (4/5/6). Outside that domain (η², 3-membered, floppy 7+) there is no window, and a
            # HAPTIC partner is always outside it: the centroid is bond-less, so there is no ring to size.
            # NUKHEG is that case (an η³ allyl and a sigma C of its own ligand, which `seat_properly` made a
            # stated vertex pair, 11 windows -> 10). Measured over 48 conformers at six seeds: restoring the
            # window moves the median of the very angle it names by 0.0°, since the backbone already sets it.
            bite = _chelate_bite_window(mol, od[i], od[j])
            if bite is not None:
                c.angles[(od[i], metal, od[j])] = bite
            continue
        # inter-ligand pairs (±8°) and any *trans*-assigned chelate (kept wide, ±25°) are held: forcing a
        # trans span makes an unreachable chelate (an en placed trans) tear -> dropped by `bonding_ok`, the
        # embed-time safety net for anything the `isomers` span filter doesn't pre-drop.
        pad = 25.0 if intra else 8.0
        c.angles[(od[i], metal, od[j])] = (max(0.0, a - pad), min(180.0, a + pad))
    # NB an η² π bond needs no hold of its own: the face is a centroid vertex, so one axial pull plus the cone
    # pins both π atoms at the face radius. Two separate M-donor pulls tore C≡C from 1.2 to 1.7 Å.
    coord = [d for d in od if d != VACANT and d not in haptic]
    coord += [a for site in haptic.values() for a in site]  # real coordinating atoms; centroid keys are reserved
    ff_terms(mol, c, {metal: (real_z, coord)})  # coordinating atoms, so nondonor_floors never floors a ring atom
    _add_umbrella(c, metal, od, poly, haptic, frag)
    return c


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

    The realised metal-donor distances and donor-metal-donor angles, so the *specific* arrangement is held
    (vs an ideal polyhedron). Used to *retain* an input metal complex instead of enumerating isomers.

    `vertices` are the polyhedron VERTICES (sigma-donor atoms + haptic centroid dummies from `_collapse_haptic`):
    a Cp/arene/eta2 face is pinned by its centroid at the realised M-ring distances (`_centroid_constraints`) and
    the angles run over vertices, so a haptic face is one site, not one distance or angle per ring atom. `haptic`
    maps each centroid dummy to its ring.
    """
    pos = mol.GetConformer().GetPositions()
    c = Constraints()
    for v in vertices:
        if v in haptic:  # a centroid vertex: pin its whole ring at the realised M-ring distances (no orient)
            _centroid_constraints(mol, metal, v, haptic[v], real_z, qdel=None, c=c, pos=pos, hyb=None)
            continue
        dist = float(np.linalg.norm(pos[metal] - pos[v]))
        add_distance(c.distances, metal, v, dist - _ML_SEED_HALF_WIDTH, dist + _ML_SEED_HALF_WIDTH)

    def position(vertex):
        return np.mean(pos[list(haptic[vertex])], axis=0) if vertex in haptic else pos[vertex]

    for a, b in itertools.combinations(vertices, 2):
        ang = _vertex_angle(position(a) - pos[metal], position(b) - pos[metal])
        c.angles[(a, metal, b)] = (max(0.0, ang - 8), min(180.0, ang + 8))
    coord = [d for d in vertices if d not in haptic]
    coord += [a for ring in haptic.values() for a in ring]  # real donors only; reserved centroids are not Mol atoms
    ff_terms(mol, c, {metal: (real_z, coord)})  # never floors a ring atom; same force field as every other path
    poly = POLYHEDRA.get(resolve_geometry(geometry))
    if poly is not None:
        _add_umbrella(c, metal, vertices, poly, haptic, _frag_map(mol))
    return c


def coordinate(iso, atoms):
    """Constrain substrate atoms into an isomer's vacant slots.

    Atoms fill vacancies in slot order. Covalent-radius distance windows and existing polyhedron angles
    orient them around the surrogate; excess atoms raise.
    """
    od = list(iso.vertices)
    vac = [v for v in range(len(od)) if od[v] == VACANT]
    if len(atoms) > len(vac):
        raise ValueError(
            f"{iso.geometry} {iso.label} has {len(vac)} vacant site(s) but {len(atoms)} atom(s) to coordinate"
        )
    r_m = _PT.GetRcovalent(iso.real_z)
    c = Constraints()
    for at, v in zip(atoms, vac, strict=False):
        od[v] = at
        d_md = r_m + _PT.GetRcovalent(iso.mol.GetAtomWithIdx(at).GetAtomicNum())  # dative ≈ covalent sum
        add_distance(c.distances, iso.metal, at, d_md - 0.2, d_md + 0.15)
    seated = set(atoms)
    poly = POLYHEDRA[resolve_geometry(iso.geometry)]
    for i, j, a in poly.resolved_angles:  # orient each newly-seated donor at its vertex
        if od[i] != VACANT and od[j] != VACANT and (od[i] in seated or od[j] in seated):
            c.angles[(od[i], iso.metal, od[j])] = (max(0.0, a - 8), min(180.0, a + 8))
    # Seating a donor pulls its neighbours into the sphere, where RDKit still floors them at the surrogate's
    # ~3.4 Å carbon-vdW contact -- a phantom that contradicts the new M-donor window. A Pt-O=C triangle needs
    # M...C ~3.2 Å; smoothing repaired the crossover by rewriting the C=O bond. `nondonor_floors` ran before
    # this atom was a donor, so re-read the tier against the augmented donor set. DG half only.
    donors = [d for d in iso.donors if d != VACANT] + list(atoms)
    topo = Chem.GetDistanceMatrix(iso.mol)
    reach = {i for at in atoms for i in np.flatnonzero(topo[at] <= _FLOOR_REACH - 1)}
    for i in map(int, reach):
        if i == iso.metal or i in seated:
            continue
        z = iso.mol.GetAtomWithIdx(i).GetAtomicNum()
        tier = overbond_tier(iso.mol, donors, i)
        c.dg_floors[(min(iso.metal, i), max(iso.metal, i))] = _tier_floor(z, tier, r_m, iso.real_z)
    return c
