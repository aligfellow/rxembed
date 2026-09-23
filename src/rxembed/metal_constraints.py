"""An `Isomer` -> the `Constraints` that hold its coordination sphere: M-donor windows and L-M-L angles.

The top of the metal stack: it reads `metal_distance` and `metal_donor_orient`, and nothing reads it back.
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from .constraints import Constraints, _graft_owns, add_distance, compose
from .mechanisms import _ANGLE_TARGET_FC, PIN_FC
from .metal_core import (
    _FIT_FLOOR,
    VACANT,
    _bounds_matrix,
    _frag_map,
    _regular_face,
    _site_height,
    _site_radius,
    _vertex_atom,
    classify_geometry,
    materialized_states,
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
    _improper,
    _vertex_angle,
    fit_residual,
    ordered_fit_residual,
    orientation_parity,
    record,
)
from .metal_slots import _SPAN_TOL, TRANS_ANGLE, _chelate_bite_window
from .stereo import bond_stereo, metal_referenced_ez, point_stereo

_ML_SEED_HALF_WIDTH = 0.05  # Å: numerical room around an M-L seed target, not a prediction interval
_TRIGONAL_CARRIERS = 3
_TETRAHEDRAL_CARRIERS = 4
_BOND_STEREO_ATOMS = 2
_OPPOSED_BITES = 2  # two disjoint chelate bites spanning a planar 4-shell: the ring closure's special case
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


def _joint_shell_targets(  # noqa: C901 - fit and validate one witness before committing the shared targets
    metal,
    vertices,
    haptic,
    poly,
    cons,
    bites,
    bounds,
    blocked=(),
    overridden=(),
    same_ligand=(),
):
    """Fit one native shell witness, or leave existing angle targets unchanged.

    Native donor-pair distances constrain reach for the whole real donor network;
    explicit distance constraints take precedence, otherwise RDKit's native bounds
    are used for every supplied same-ligand real donor pair. Angle windows bias the fit without
    imposing inconsistent midpoints. An already admitted shell is a no-op. Preserve
    each prior's angular flexibility when recentering it on a joint witness. A failed
    search does not establish infeasibility. No centroid/member or cross-fragment
    van der Waals spans are invented here.
    """
    if blocked or VACANT in vertices or not bites or not cons.angles:
        return
    directions = np.array(poly.vertex_dirs, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    slot = {donor: i for i, donor in enumerate(vertices)}
    radial = []
    for donor in vertices:
        if tuple(sorted((metal, donor))) in overridden:
            return
        window = cons.distances.get(tuple(sorted((metal, donor))))
        if window is None or not np.all(np.isfinite(window)) or window[0] <= 0 or window[0] > window[1]:
            return
        radial.append(window)
    lengths = tuple(float(np.mean(window)) for window in radial)
    spans = {}
    for i, j in itertools.combinations(range(len(vertices)), 2):
        pair = frozenset((i, j))
        left, right = vertices[i], vertices[j]
        pair_key = tuple(sorted((left, right)))
        span = cons.distances.get(pair_key)
        if span is None:
            if left in haptic or right in haptic or pair_key not in same_ligand:
                continue
            span = (
                max(0.0, float(bounds[max(left, right)][min(left, right)]) - _SPAN_TOL),
                float(bounds[min(left, right)][max(left, right)]) + _SPAN_TOL,
            )
        spans[pair] = span
    shell = Chem.MolFromSmiles(".".join(["*"] * (len(vertices) + 1)))
    conf = Chem.Conformer(shell.GetNumAtoms())
    conf.SetPositions(np.vstack([np.zeros(3), directions * np.asarray(lengths)[:, None]]))
    shell.AddConformer(conf)
    ff = rdForceFieldHelpers.CreateEmptyForceFieldForMol(shell)
    ff.AddFixedPoint(0)
    for index, window in enumerate(radial, 1):
        ff.AddDistanceConstraint(0, index, *window, PIN_FC)
    for i, j in itertools.combinations(range(len(vertices)), 2):
        pair = frozenset((i, j))
        atoms = (i + 1, 0, j + 1)
        if pair in spans:
            ff.AddDistanceConstraint(i + 1, j + 1, *spans[pair], PIN_FC)
        key = (vertices[i], metal, vertices[j])
        window = cons.angles.get(key, cons.angles.get(key[::-1]))
        if window is not None:
            ff.UFFAddAngleConstraint(*atoms, False, *window, _ANGLE_TARGET_FC)
    try:
        ff.Initialize()
        if ff.CalcEnergy() == 0.0:
            return
        if ff.Minimize(maxIts=2000, forceTol=1e-8) != 0:
            return
    except (RuntimeError, ValueError):
        return
    positions = shell.GetConformer().GetPositions()
    if not np.all(np.isfinite(positions)):
        return
    rays = positions[1:] - positions[0]
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 0):
        return
    rays /= norms[:, None]
    for length, window in zip(norms, radial, strict=True):
        if not window[0] - 1e-7 <= length <= window[1] + 1e-7:
            return
    values = np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
    for pair, span in spans.items():
        i, j = sorted(pair)
        value = float(np.linalg.norm(positions[i + 1] - positions[j + 1]))
        if not span[0] - 1e-7 <= value <= span[1] + 1e-7:
            return
    ordered = ordered_fit_residual(rays, directions)
    if (
        classify_geometry(shell, 0, tuple(range(1, len(vertices) + 1)), warn=False) != poly.name
        or ordered > _FIT_FLOOR
        or not np.isclose(ordered, fit_residual(rays, poly), atol=1e-8, rtol=0.0)
    ):
        return
    # Only reflection-invariant angles leave this auxiliary fit; the isomer's actual hand is checked on embedding.
    updates = {}
    for key in tuple(cons.angles):
        a, centre, b = key
        if centre != metal or a not in slot or b not in slot:
            continue
        value = float(values[slot[a], slot[b]])
        existing = cons.angles[key]
        updates[key] = (existing, value)
    if all(lo - 1e-7 <= value <= hi + 1e-7 for (lo, hi), value in updates.values()):
        return
    final = {}
    for key, ((lo, hi), value) in updates.items():
        shift = 0.0 if lo <= value <= hi else float(np.clip(value - (lo + hi) / 2, -lo, _STRAIGHT - hi))
        final[key] = ((lo + shift, hi + shift), value)
    for key, (window, value) in final.items():
        cons.angles[key] = window
        canonical = min(key, key[::-1])
        cons.pulls.pop(canonical, None)
        cons.pulls.pop(canonical[::-1], None)
        cons.pulls[canonical] = value


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


def _tetrahedral_cross_angle(mol, metal, vertices, bites, distances, blocked):
    """Centre disjoint chelate bites on a realizable tetrahedral ray arrangement, or abstain.

    Opposite pair bisectors and orthogonal pair planes preserve the tetrahedral construction while each
    bite takes its own midpoint. Their cross-angle cosine is -cos(bite1/2)*cos(bite2/2). These soft angular
    targets do not certify whole-ligand feasibility. Joining backbones remain ligand restraints:
    fragment membership does not change this shell identity. Externally held ligands do not qualify.
    """
    if len(bites) != 2:  # noqa: PLR2004 - two disjoint donor pairs
        return None
    first, second = bites
    if first & second or len(first | second) != _TETRAHEDRAL_CARRIERS:
        return None
    frag = _frag_map(mol)
    groups = {frag[vertices[next(iter(pair))]] for pair in bites}
    if metal in blocked or any(frag[atom] in groups for atom in blocked):
        return None
    half = np.radians([0.25 * sum(window) for window in bites.values()])
    (sa, sb), (ca, cb) = np.sin(half), np.cos(half)
    rays = np.array([(sa, 0, ca), (-sa, 0, ca), (0, sb, -cb), (0, -sb, -cb)])
    donors = [vertices[i] for pair in bites for i in sorted(pair)]
    lengths = np.array([0.5 * sum(distances[tuple(sorted((metal, donor)))]) for donor in donors])
    witness = Chem.MolFromSmiles("[*].[*].[*].[*].[*]")
    conf = Chem.Conformer(witness.GetNumAtoms())
    conf.SetPositions(np.vstack([np.zeros(3), rays * lengths[:, None]]))
    witness.AddConformer(conf)
    # Native reach may lie outside the census bite prior. A realizable but almost planar witness must not
    # redefine the requested shape; use the same perception and fit owners as final geometry validation.
    if (
        classify_geometry(witness, 0, (1, 2, 3, 4), warn=False) != "tetrahedral"
        or fit_residual(rays, record("tetrahedral")) > _FIT_FLOOR
    ):
        return None
    return float(np.degrees(np.arccos(-ca * cb)))


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
    qdel = _fact(context, "charges", lambda: delocalised_charges(mol))
    # A Lewis charge is an artefact, so spread it.
    # The model uses ligand-only classes for either input bond convention. Input lengths do not need typing.
    hyb = _fact(context, "hybridisation", lambda: _stripped_hybridisation(mol))
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
                *_donor_distance_window(mol, metal, d, real_z, real_od, positions=pos, charges=qdel, hyb=hyb),
            )
        # Wall each donor substituent off the metal, the orientation hold a real energy cannot supply itself.
        # Length provenance is independent: measured M-L distances do not determine a partially free M-D-X axis.
        if donor_orientation:
            _orient_donor(mol, metal, d, real_od, c, hyb=hyb)
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
    bites = {}
    for i, j in pairs:
        left, right = od[i], od[j]
        if (
            VACANT in (left, right)
            or left in haptic
            or right in haptic
            or frag_of(left) != frag_of(right)
            or ideal_angles[frozenset((i, j))] >= CHELATE_SPAN_ANGLE
        ):
            continue
        bond_bounds = _fact(context, "bounds", lambda: _bounds_matrix(mol))
        lengths = (
            0.5 * sum(c.distances[(min(left, metal), max(left, metal))]),
            0.5 * sum(c.distances[(min(right, metal), max(right, metal))]),
        )
        bite_key = ("bite", min(left, right), max(left, right), tuple(lengths))
        if bite_key not in context:
            context[bite_key] = _chelate_bite_window(mol, left, right, real_od, bounds=bond_bounds, lengths=lengths)
        if bite := context[bite_key]:
            bites[frozenset((i, j))] = bite
    cross_angle = (
        _tetrahedral_cross_angle(mol, metal, od, bites, c.distances, set(frozen) | set(coupled_atoms))
        if poly.name == "tetrahedral"
        else None
    )
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
        if pair in bites:
            c.angles[(od[i], metal, od[j])] = bites[pair]
            continue
        through_bites = any(
            frozenset((i, k)) in bites and frozenset((j, k)) in bites for k in range(len(od)) if k not in pair
        )
        if through_bites and a >= TRANS_ANGLE:
            c.angles[(od[i], metal, od[j])] = (float(TRANS_ANGLE), _STRAIGHT)
            continue
        # Exact enumeration keeps every distinct seating; realised geometry decides whether this pair is feasible.
        centre = a if cross_angle is None else cross_angle
        c.angles[(od[i], metal, od[j])] = (max(0.0, centre - _ANGLE_PAD), min(_STRAIGHT, centre + _ANGLE_PAD))
    shared = _planar_bite_targets(c, od, poly, bites, metal, mol, set(frozen) | set(coupled_atoms), frag)
    if poly.planar and not shared:
        real = {i for i, d in enumerate(od) if d != VACANT and d not in haptic}
        bite_pairs = list(bites)
        opposed = (
            len(bite_pairs) == _OPPOSED_BITES
            and not (bite_pairs[0] & bite_pairs[1])
            and bite_pairs[0] | bite_pairs[1] == real
        )
        if opposed:
            # Two disjoint bites spanning every real slot of a planar 4-shell: the ring closure leaves no
            # free parameter beyond the two bites, so cis (the two connecting edges) and trans (the two
            # diagonals) must be derived from BOTH bites jointly. The old per-bite `180 - bite` complement is
            # this closure's b1 == b2 special case; TRANS_ANGLE (the fan-case row) plays no part here.
            (lo1, hi1), (lo2, hi2) = bites.values()
            cis = (360.0 - hi1 - hi2) / 2.0, (360.0 - lo1 - lo2) / 2.0
            trans = 180.0 - (max(hi1, hi2) - min(lo1, lo2)) / 2.0, 180.0
            for pair in angle_rows.keys() - bites.keys():
                i, j = sorted(pair)
                key = (od[i], metal, od[j])
                if key not in c.angles:
                    continue
                window = trans if ideal_angles[pair] == _STRAIGHT else cis
                if pair in stated_pairs:  # the template's own idealised default: the closure supersedes it,
                    lo0, hi0 = c.angles[key]  # UNLESS the two are disjoint -- a real conflict, not this fix's
                    if window[1] < lo0 or window[0] > hi0:  # own "ideal 180" assumption reasserting itself
                        raise ValueError(
                            f"metal[{metal}]: opposed-bite closure {window} for donors {od[i]},{od[j]} is "
                            f"disjoint from their stated angle window {(lo0, hi0)}"
                        )
                c.angles[key] = window
        else:
            # Retain the existing complementary priors for a planar shell without an opposed-bite closure.
            # ponytail: linked/held networks still use pairwise priors; replace when a joint model covers them.
            for pair in angle_rows.keys() - stated_pairs - bites.keys():
                i, j = sorted(pair)
                key = (od[i], metal, od[j])
                if key not in c.angles:
                    continue
                complements = [
                    (_STRAIGHT - hi, _STRAIGHT - lo)
                    for bite_pair, (lo, hi) in bites.items()
                    if len(pair & bite_pair) == 1 and ideal_angles.get(pair ^ bite_pair) == _STRAIGHT
                ]
                if complements:
                    lo, hi = max(window[0] for window in complements), min(window[1] for window in complements)
                    if lo <= hi:
                        c.angles[key] = (lo, hi)
    if bites and not shared and cross_angle is None:
        _joint_shell_targets(
            metal,
            od,
            haptic,
            poly,
            c,
            bites,
            bond_bounds,
            set(frozen) | set(coupled_atoms),
            distance_overrides,
            {
                tuple(sorted((left, right)))
                for left, right in itertools.combinations(od, 2)
                if left not in haptic
                and right not in haptic
                and VACANT not in (left, right)
                and frag_of(left) == frag_of(right)
            },
        )
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
        )
    _add_umbrella(c, metal, od, poly)
    return _drop_graft_owned(c, frozen, haptic)


def _planar_bite_targets(cons, vertices, poly, bites, metal, mol, blocked, fragments=None):  # noqa: C901 - two constructions, one target owner
    """Derive linked planar bites and their cross windows from one shared shell.

    Rotate independent bites around their template bisectors, retaining the opposite pair's window
    and the trans windows. For a three-donor fan, keep the central ray and move independent spectators
    to the plane normals. Neither construction certifies an intact whole-ligand pose; explicit
    constraints and linked spectator networks remain authoritative. Return True only after committing
    both the windows and their preferences; otherwise leave them untouched.
    """
    if not bites or blocked or VACANT in vertices or cons.fixed or cons.frozen or cons.shapes or any(cons.contacts):
        return
    directions = np.array(poly.vertex_dirs, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    candidates = []
    for pair, window in bites.items():
        i, j = sorted(pair)
        normal = np.cross(directions[i], directions[j])
        if np.isclose(np.linalg.norm(normal), 0.0, atol=1e-8):
            continue
        normal /= np.linalg.norm(normal)
        planar = set(np.flatnonzero(np.abs(directions @ normal) < _SHELL_ATOL))
        if len(planar) in (_TRIGONAL_CARRIERS, _TETRAHEDRAL_CARRIERS) and all(
            abs(abs(directions[k] @ normal) - 1.0) <= _SHELL_ATOL for k in range(len(vertices)) if k not in planar
        ):
            candidates.append((pair, window, planar))
    rays = directions.copy()
    if candidates and all(
        plane == candidates[0][2] and (pair == candidates[0][0] or not pair & candidates[0][0])
        for pair, _, plane in candidates
    ):
        pair, window, planar = candidates[0]
        i, j = sorted(pair)
        if {vertices[i], vertices[j]} & cons.haptic.keys():
            return
        fragments = _frag_map(mol) if fragments is None else fragments
        group = fragments[vertices[i]]
        attached = {slot for slot, donor in enumerate(vertices) if fragments[_vertex_atom(cons.haptic, donor)] == group}
        if fragments[vertices[j]] != group or attached & planar != set(pair):
            return
        angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0)))
        pairs, windows = [pair], [window]
        targets = [float(np.clip(angles[i, j], *window))]
        moved = {frozenset(key) for key in itertools.combinations(planar, 2)} - {pair}
        if len(planar) == _TRIGONAL_CARRIERS:
            sectors = [angles[a, b] for a, b in itertools.combinations(planar, 2)]
            if not all(_SHELL_ATOL < value < 180.0 - _SHELL_ATOL for value in sectors) or not np.isclose(
                sum(sectors), 360.0, atol=1e-6, rtol=0.0
            ):
                return
        else:
            if {vertices[k] for k in planar} & cons.haptic.keys():
                return
            opposite = frozenset(planar - pair)
            groups = {
                frozenset(
                    k for k, donor in enumerate(vertices) if fragments[_vertex_atom(cons.haptic, donor)] == fragment
                )
                for fragment in set(fragments.values())
            }
            if any(len(group) > 1 and group not in (pair, opposite) for group in groups):
                return
            if opposite in groups and opposite not in bites:
                return
            slots = {donor: k for k, donor in enumerate(vertices)}
            limits = {
                frozenset((slots[a], slots[b])): value
                for (a, centre, b), value in cons.angles.items()
                if centre == metal and a in slots and b in slots
            }
            trans = {key for key in moved if np.isclose(angles[tuple(sorted(key))], 180.0)}
            if len(trans) != 2 or opposite not in limits or not trans <= limits.keys():  # noqa: PLR2004 - two trans pairs
                return
            pairs.append(opposite)
            windows.append(limits[opposite])
            targets.append(float(np.clip(angles[tuple(sorted(opposite))], *windows[1])))
            # With opposite bisectors, trans = 180 - |alpha-beta|/2. Project the two closest-to-ideal
            # bite targets onto this strip; no independent exact-complement assumptions are needed.
            gap = 2.0 * (180.0 - max(limits[key][0] for key in trans))
            low, high = sorted(range(2), key=targets.__getitem__)
            if targets[high] - targets[low] > gap:
                lo = max(windows[low][0], windows[high][0] - gap)
                hi = min(windows[low][1], windows[high][1] - gap)
                if lo > hi:
                    return
                targets[low] = float(np.clip((sum(targets) - gap) / 2.0, lo, hi))
                targets[high] = targets[low] + gap
            moved -= {opposite, *trans}
        if all(
            np.isclose(target, angles[tuple(sorted(pair))], atol=1e-8, rtol=0.0)
            for pair, target in zip(pairs, targets, strict=True)
        ):
            return
        for (i, j), target in zip((sorted(pair) for pair in pairs), targets, strict=True):
            bisector, transverse = directions[i] + directions[j], directions[i] - directions[j]
            bisector /= np.linalg.norm(bisector)
            transverse /= np.linalg.norm(transverse)
            half = np.radians(target / 2.0)
            rays[i] = np.cos(half) * bisector + np.sin(half) * transverse
            rays[j] = np.cos(half) * bisector - np.sin(half) * transverse
    elif len(bites) == 2 and not cons.haptic:  # noqa: PLR2004 - two bites share the fan's central donor
        first, second = map(set, bites)
        if len(first & second) != 1:
            return
        (centre,) = first & second
        fan = first | second
        spectators = set(range(len(vertices))) - fan
        if not spectators:
            return
        members = {vertices[i] for i in fan}
        groups = [set(vertices).intersection(fragment) for fragment in Chem.GetMolFrags(mol)]
        if members not in groups or any(len(group) > 1 and group != members for group in groups):
            return
        outer = sorted(fan - {centre})
        normal = np.cross(directions[centre], directions[outer[0]])
        if np.isclose(np.linalg.norm(normal), 0.0, atol=1e-8):
            return
        normal /= np.linalg.norm(normal)
        if any(abs(directions[i] @ normal) > _SHELL_ATOL for i in fan):
            return
        for i in outer:
            tangent = directions[i] - (directions[i] @ directions[centre]) * directions[centre]
            if np.isclose(np.linalg.norm(tangent), 0.0, atol=1e-8):
                return
            tangent /= np.linalg.norm(tangent)
            angle = np.radians(np.mean(bites[frozenset((centre, i))]))
            rays[i] = np.cos(angle) * directions[centre] + np.sin(angle) * tangent
        for i in spectators:
            component = directions[i] @ normal
            if np.isclose(component, 0.0, atol=1e-8):
                return
            rays[i] = np.sign(component) * normal
        if np.linalg.matrix_rank(directions) == directions.shape[1] and (
            np.linalg.matrix_rank(rays) < directions.shape[1] or orientation_parity(rays, directions) < 0
        ):
            return
        moved = {frozenset(key) for key in itertools.combinations(range(len(vertices)), 2)} - bites.keys()
    else:
        return
    fitted = np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
    work = Chem.MolFromSmiles(".".join(["[*]"] * (len(vertices) + 1)))
    conf = Chem.Conformer(work.GetNumAtoms())
    lengths = [np.mean(cons.distances[tuple(sorted((metal, donor)))]) for donor in vertices]
    conf.SetPositions(np.vstack([np.zeros(3), rays * np.asarray(lengths)[:, None]]))
    work.AddConformer(conf)
    ordered = ordered_fit_residual(rays, directions)
    if (
        classify_geometry(work, 0, tuple(range(1, len(vertices) + 1)), warn=False) != poly.name
        or ordered > _FIT_FLOOR
        or not np.isclose(ordered, fit_residual(rays, poly), atol=1e-8, rtol=0.0)
    ):
        return
    slots = {donor: slot for slot, donor in enumerate(vertices)}
    updated, targets = cons.angles.copy(), {}
    cross = set()
    for key, (lo, hi) in cons.angles.items():
        a, centre, b = key
        if centre != metal or a not in slots or b not in slots:
            continue
        left, right = slots[a], slots[b]
        value = float(fitted[left, right])
        if frozenset((left, right)) in moved:
            shift = float(np.clip(value - (lo + hi) / 2.0, -lo, 180.0 - hi))
            updated[key] = (lo + shift, hi + shift)
            cross.add(frozenset((left, right)))
        if not updated[key][0] - 1e-8 <= value <= updated[key][1] + 1e-8:
            return
        targets[min(key, key[::-1])] = value
    if cross != moved:
        return
    # Flat walls admit competing shapes. The weak FF preference uses this same shell, not unrelated midpoints.
    cons.angles = updated
    cons.pulls.update(targets)
    return True


def _donor_distance_window(mol, metal, donor, real_z, donors, *, positions=None, charges=None, hyb=None):
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
