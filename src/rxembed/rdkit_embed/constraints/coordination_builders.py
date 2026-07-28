"""Metal coordination-constraint builders — the polytope constraints, per isomer.

The consumer layer of the metal stack (it reads `distance` + `donor_orient`), carved up out of
`constraints.metal` so those become clean top-level imports and `metal.py` stays foundational
(`base ← metal ← distance ← donor_orient ← coordination_builders`, one strict DAG, no lazy dodges).
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem

from .base import Constraints, SphereRecipe, add_distance
from .distance import (
    _FLOOR_REACH,
    _tier_floor,
    delocalised_charges,
    ff_terms,
    ml_distance,
    overbond_tier,
)
from .donor_orient import _coplanar_donor, _orient_donor
from .metal import (
    _APICAL_MIN,
    _DISCONNECTED,
    _PT,
    _SPAN_ANGLE,
    POLYHEDRA,
    VACANT,
    Isomer,
    _collapse_haptic,
    _frag_map,
    _regular_face,
    _site_radius,
    _vertex_angle,
    _vertex_atom,
    chirality_of,
    classify_geometry,
    geometry_for,
    label,
    strip_phantoms,
    surrogate_metal,
)


def _centroid_constraints(mol, metal, dummy, ring, real_z, qdel, c, pos):
    """Pin a haptic face by its centroid dummy — the ONE polyhedron vertex an η² alkene / Cp / arene presents.

    The metal->ring-atom distance is ``mc`` and the dummy sits ``d_c = sqrt(mc^2 - r^2)`` up the axis — EXACT for any
    face whose atoms share a circumradius, η² included (two points lie on a circle about their midpoint, and equal
    M-C puts M on the perpendicular; measured on Zeise's salt: model 2.018 Å vs true 2.018). A REGULAR face
    (`_regular_face`: a bond, Cp, arene, cyclobutadiene) is held in CONE mode — each ring atom also pinned to the
    centroid at the radius ``r``. An irregular face (allyl / diene / pentadienyl) drops those pins (METAL mode) so
    the bent face relaxes freely; the rigid cone over-constrains it into a torn geometry (OIN's metal-mode
    fallback). The dummy is a ``phantom`` (zero vdW in the FF, DG floor relaxed) — it lives inside its ring.
    """
    r = _site_radius(mol, ring)  # circumradius of the face
    if pos is not None:  # realised metal->ring-atom distances from the input geometry
        mcs = [float(np.linalg.norm(pos[metal] - pos[a])) for a in ring]
    else:  # the fitted model; the ring is its own co-donor set so the hapticity term sees the whole face
        mcs = [ml_distance(mol, metal, a, real_z, set(ring), charges=qdel) for a in ring]
    mc = float(np.mean(mcs))
    d_c = float(np.sqrt(max(mc * mc - r * r, 0.25)))
    add_distance(c.distances, metal, dummy, d_c - 0.05, d_c + 0.05)  # M -> centroid vertex (both modes)
    cone = _regular_face(mol, ring)  # a regular face is equidistant from its own centroid -> CONE mode; an
    #   irregular one (allyl/diene/pentadienyl) is floppy AND not centroid-equidistant -> METAL mode.
    for a, mca in zip(ring, mcs, strict=True):
        add_distance(c.distances, metal, a, mca - 0.05, mca + 0.05)  # hold each ring atom at its metal distance
        if cone:  # a rigid regular n-gon (Cp/arene): also pin each ring atom to the centroid at the ring radius
            add_distance(c.distances, dummy, a, r - 0.1, r + 0.1)
    c.phantoms = c.phantoms | {dummy}
    c.haptic[dummy] = tuple(ring)  # embed scaffolding: materialised transiently in the DG/UFF, stored in no real Mol


def coordination(mol, metal, donors, geometry, order, real_z, frozen=(), core_frozen=(), haptic=None):
    """Build one isomer's constraints: metal-donor distances + donor-metal-donor angles.

    `donors` may be padded with ``VACANT`` for empty vertices (a coordination pocket) — those get no constraints.

    `frozen` is the donor set held rigid by a ``fix=`` reacting core: an angle between two frozen donors is
    omitted (the freeze already pins that sub-triangle exactly, so an ideal-polyhedron angle only over-determines
    and distorts it); a free-vs-frozen angle is kept, to seat the free donor at its vertex.

    An intra-chelate L-M-L angle (two donors of one ligand) is not constrained at all: the donor-donor distance
    already in the bounds matrix plus the two M-donor distances fixes the bite, so imposing the ideal angle only
    contradicts the backbone (a side-on chelate bites ~45°, and even ±25° tore the bond). Only inter-ligand angles
    are held (±8°). Impossible placements (a short backbone forced trans) are dropped in
    `distinct_vertex_orderings` / `bonding_ok`.
    """
    frag = _frag_map(mol)  # same ligand = same fragment

    def frag_of(v):  # a face tethered to a co-donor (η²-alkyne + alkyl of one metallacycle; an ansa Cp) must read
        return frag[_vertex_atom(haptic, v)]  # as ONE ligand: the bond-less centroid carries no fragment of its own

    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    c = Constraints()
    od = [donors[k] for k in order]  # od[vertex] = donor atom there, or VACANT
    real_od = {x for x in od if x != VACANT}  # the co-donors, for the haptic (side-on) check in _orient_donor
    qdel = delocalised_charges(mol) if pos is None else None  # a Lewis charge is an artefact — spread it
    for d in od:
        if d == VACANT:
            continue
        if haptic and d in haptic:  # a centroid vertex: pin its whole ring, not a single donor (no orient/coplanar)
            _centroid_constraints(mol, metal, d, haptic[d], real_z, qdel, c, pos)
            continue
        if pos is not None:  # use the REALISED metal-donor bond length from the input
            d_md = float(np.linalg.norm(pos[metal] - pos[d]))  # geometry (a covalent-radius guess is wrong for
            add_distance(c.distances, metal, d, d_md - 0.1, d_md + 0.1)  # a hydride/carbonyl and breaks the bounds)
        else:  # no input geometry: the fitted periodic model (element, group, delocalised charge, hapticity)
            t = ml_distance(mol, metal, d, real_z, real_od, charges=qdel)  # see `ml_distance`
            add_distance(c.distances, metal, d, t - 0.05, t + 0.05)
        # splay a slow-inverting pnictogen donor's (P/As/Sb) protons away from the metal so its lone pair points
        # at M — the one orientation hold g-xTB cannot supply for itself. Skipped for a side-on η². See
        # `_orient_donor`; an sp2 donor's coplanarity is the sibling hold below, and every other donor's
        # orientation is left for the real energy to decide.
        _orient_donor(mol, metal, d, real_od, c, core_frozen)
        # cap an sp2 donor's metal at the donor's own sp2 plane — the improper the stripped bond removed.
        # Skip when the metal or this donor is frozen: a fix= TS core already pins that M-donor geometry at the
        # input, and biasing it toward coplanar only fights the freeze.
        if metal not in core_frozen and d not in core_frozen:
            _coplanar_donor(mol, metal, d, c)
    for i, j, a in POLYHEDRA[geometry].resolved_angles:
        if od[i] == VACANT or od[j] == VACANT:  # an angle to an empty vertex is unconstrained
            continue
        if od[i] in frozen and od[j] in frozen:  # both held by freeze -> don't over-determine the core
            continue
        intra = frag_of(od[i]) == frag_of(od[j])  # two donors of one chelating ligand
        if intra and a < _SPAN_ANGLE:  # a *cis* chelate: the ideal polyhedron angle is a poor bite predictor and
            # the backbone-only distance lets the DG fold a tight bite shut, so pin the bite from the ring SIZE
            # (topology), where calibrated (4/5/6). Out of that domain (eta2 / 3-membered / floppy 7+): no window.
            bite = _chelate_bite_window(mol, metal, od[i], od[j])
            if bite is not None:
                c.angles[(od[i], metal, od[j])] = bite
            continue
        # inter-ligand pairs (±8°) and any *trans*-assigned chelate (kept wide, ±25°) are held: forcing a
        # trans span makes an unreachable chelate (an en placed trans) tear -> dropped by `bonding_ok`, the
        # embed-time safety net for anything the `isomers` span filter doesn't pre-drop.
        pad = 25.0 if intra else 8.0
        c.angles[(od[i], metal, od[j])] = (max(0.0, a - pad), min(180.0, a + pad))
    # NB the η² π bond needs no hold of its own: the face is a centroid vertex, so the metal pulls the CENTROID
    # (one axial pull) and the cone pins both π atoms to it at the face radius — the two separate M-donor pulls
    # that used to tear C≡C from 1.2 to 1.7 Å no longer exist. See `_regular_face`: a bond is the smallest face.
    coord = od + [a for site in (haptic or {}).values() for a in site]  # vertices + haptic RING atoms: the real
    ff_terms(mol, c, {metal: (real_z, coord)})  # coordinating atoms, so nondonor_floors never floors a ring atom
    # Record how these targets were derived, so the engine can RE-DERIVE them on a realisable point set if the
    # distance model, the polytope and the ligand's reach turn out mutually impossible (`sphere.py`).
    c.spheres = (
        SphereRecipe(metal, tuple(donors), geometry, tuple(order), real_z, tuple(sorted((haptic or {}).items()))),
    )
    return c


_CHELATE_BITE = {4: (58.0, 81.0), 5: (70.0, 91.0), 6: (74.0, 104.0)}  # chelate ring size -> M-D-D bite window (deg):
#   the census range that admits every real bite and forbids the fold. The LIGAND's ring sets the bite, not the
#   metal's polytope — a 5-ring bites ~80°, not the ideal 90°, so the polyhedron vertex angle is a poor predictor.


def _chelate_bite_window(mol, metal, a, b):
    """Bite window for two donors of one chelate, from the ring SIZE (backbone bond-path + the 2 M-donor bonds).

    The surrogate strips the M-donor bonds, so the chelate ring is open and the two donors relate only through the
    backbone; a dative bond is not a ring bond to RDKit, so ring size = that backbone path + 2. Returns ``None`` for
    a non-chelate (no shared path) or a ring outside the calibrated 4/5/6 domain (haptic / 3-membered / floppy 7+).
    """
    try:
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(metal)
        topo = Chem.GetDistanceMatrix(rw.GetMol())
    except Exception:  # a malformed graph simply yields no bite constraint
        return None
    ia, ib = (a - 1 if a > metal else a), (b - 1 if b > metal else b)  # RemoveAtom renumbers atoms past the metal
    path = float(topo[ia][ib])
    if path > _DISCONNECTED:  # different fragments -> not a chelate
        return None
    return _CHELATE_BITE.get(int(path) + 2)


def coordination_from_geometry(mol, metal, donors, real_z, vertices=None, haptic=None, cid=-1):
    """Build coordination constraints from the **actual** input geometry (the specific ligand arrangement).

    The realised metal-donor distances and donor-metal-donor angles, so the *specific* arrangement is held
    (vs an ideal polyhedron). Used to *retain* an input metal complex instead of enumerating isomers.

    `vertices` are the polyhedron VERTICES (sigma-donor atoms + haptic centroid dummies from `_collapse_haptic`,
    defaulting to `donors` for a sigma-only sphere): a Cp/arene/eta2 face is pinned by its centroid at the
    realised M-ring distances (`_centroid_constraints`) and the angles run over vertices, so a haptic face is
    ONE site — not one distance/angle per ring atom. `haptic` maps each centroid dummy -> its ring.
    """
    haptic = haptic or {}
    vertices = list(donors) if vertices is None else list(vertices)
    pos = mol.GetConformer(cid).GetPositions()
    c = Constraints()
    for v in vertices:
        if v in haptic:  # a centroid vertex: pin its whole ring at the realised M-ring distances (no orient)
            _centroid_constraints(mol, metal, v, haptic[v], real_z, None, c, pos)  # pos given -> realised, qdel unused
            continue
        dist = float(np.linalg.norm(pos[metal] - pos[v]))
        add_distance(c.distances, metal, v, dist - 0.05, dist + 0.05)
    for a, b in itertools.combinations(vertices, 2):
        ang = _vertex_angle(pos[a] - pos[metal], pos[b] - pos[metal])
        c.angles[(a, metal, b)] = (max(0.0, ang - 8), min(180.0, ang + 8))
    coord = list(vertices) + [a for ring in haptic.values() for a in ring]  # + ring atoms so nondonor_floors
    ff_terms(mol, c, {metal: (real_z, coord)})  # never floors a ring atom; same force field as every other path
    return c


def from_geometry(mol):
    """Build an `Isomer` that **retains the input ligand arrangement** (no enumeration).

    Coordination constraints come from the Mol's *actual* conformer (which ligand sits where, at the
    realised distances/angles), the metal is swapped to the surrogate, and `vertices` records the as-given
    donor order. A haptic face (Cp / arene / eta2) collapses to ONE centroid vertex — the same
    transient-centroid mechanism `enumerate_isomers` uses — so the retained arrangement matches every other
    metal path (a Cp is a single site, not five sigma donors). `mol` must carry a conformer (e.g. from an xyz).
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("from_geometry needs an input geometry (a Mol with a conformer)")
    base, m, donors, real_z, real_q = surrogate_metal(mol)  # surrogate; conformer is preserved
    base, vertices, haptic = _collapse_haptic(base, donors)  # each haptic face -> one centroid vertex (sigma pass thru)
    cons = coordination_from_geometry(base, m, donors, real_z, vertices=vertices, haptic=haptic)
    # MEASURE the polytope from the conformer (this path always has one) — naming it from the vertex count alone
    # would be guessing when the answer is in the coordinates; the count is the fallback for a coordination
    # number no template covers. An apical (eta>=3) face fills more than one site, so a CN4 piano stool is a
    # distorted tetrahedron the templates do not model: skip classify and take the apical count default
    # (`geometry_for(has_apical=)`), never the flat square_planar the vertex count alone would pick.
    apical = any(len(r) >= _APICAL_MIN for r in haptic.values())
    geom = (
        (None if apical else classify_geometry(base, m, vertices))
        or geometry_for(len(vertices), has_apical=apical)
        or f"{len(vertices)}-coordinate"
    )
    return Isomer(
        strip_phantoms(base, set(haptic)),  # the stored mol is REAL — the centroid dummy is transient scaffolding
        cons,
        m,
        donors,
        real_z,
        real_q,
        label(base, m, vertices, base.GetConformer().GetId(), geom),
        geom,
        list(vertices),
        chirality=chirality_of(base, vertices, geom, vertices, haptic=haptic),
        haptic=dict(haptic),
        donor_bonds=[(d, m) for d in donors],  # the M-donor bonds surrogate_metal stripped — re-added on output
    )


def coordinate(iso, atoms, pad=0.15):
    """Build constraints seating substrate `atoms` into the metal's **vacant vertices**.

    The metal-donor coordinative distance + the polyhedron angles that orient each donor at its empty
    vertex. Reuses the real-ligand machinery: a substrate atom simply fills a VACANT slot. Returns a
    `Constraints` to merge.

    The window is ``(r_M + r_donor - 0.2, … + pad)``. The realised distance lands at the **upper bound**
    (the carbon surrogate has no metal-donor attraction — its vdW pushes them apart — so the restraint
    only caps the separation), so the upper bound is set to a realistic dative length ≈ covalent-sum +
    0.15 Å (a true distance needs the xTB calculator on the real metal). Raises if there are more atoms
    than vacant sites; `atoms` fill vacant vertices in order.
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
        add_distance(c.distances, iso.metal, at, d_md - 0.2, d_md + pad)
    seated = set(atoms)
    for i, j, a in POLYHEDRA[iso.geometry].resolved_angles:  # orient each newly-seated donor at its vertex
        if od[i] != VACANT and od[j] != VACANT and (od[i] in seated or od[j] in seated):
            c.angles[(od[i], iso.metal, od[j])] = (max(0.0, a - 8), min(180.0, a + 8))
    # Seating a donor pulls its neighbours into the sphere, where RDKit still floors them at the bond-less carbon
    # surrogate's vdW contact (~3.4 A) — a phantom that now CONTRADICTS the new M-donor window. A Pt-O=C triangle
    # needs M...C at ~3.2 A: smoothing repaired the 0.000545 A crossover by rewriting the C=O bond and fired the
    # sphere solver at a non-problem. `nondonor_floors` ran before this atom was a donor (so it called the carbonyl
    # C third-sphere); re-read the tier against the AUGMENTED donor set and hand the DG the same `_tier_floor` the
    # real ligands get. Reach and membership mirror its DG loop. DG half only: the FF surrogate keeps its own vdW.
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
