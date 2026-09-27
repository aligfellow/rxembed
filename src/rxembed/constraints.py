"""The `Constraints` struct every builder fills and every stage reads, plus the ``fix``/``constrain`` resolver.

Two verbs, one resolver (`resolve_core`). ``fix`` is rigid: the named atoms get this geometry, as a list
(own coords, Kabsch graft), ``{i: (x, y, z)}`` (explicit coords), or ``{(i, j): d, ...}`` (scalar
distance/angle/dihedral values within 0.001 A / 0.005 deg, or explicit windows; a dict may mix all three).
``constrain`` is soft: a wider window a real energy may overrule, and the home of pi-stacks
(``(ring_a, ring_b): separation``). Held is ``fix``; a bias the search can move off is ``constrain``.
``template=(reference, map_or_smarts)`` is not a third verb: it dissolves into a coords-``fix`` before this
resolver sees it. See README.md for the full vocabulary and worked examples.

Keys are 0-based atom indices in xyz/graph order. Strictness is the fix/constrain axis, not a second force
constant: a scalar fix carries a point restraint and an acceptance gate; a soft window carries neither.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from functools import partial

import numpy as np
from rdkit import Chem

from .utils import atom_label, bond_angle, dihedral_angle

_STRAIGHT = 180.0  # degrees: angle ceiling and one half-turn of a periodic dihedral


@dataclass(kw_only=True)
class Constraints:
    """Carry named geometric windows and rigid holds between stages; construct every field by keyword."""

    distances: dict = field(default_factory=dict)  # (i, j) -> (lo, hi) Angstrom
    angles: dict = field(default_factory=dict)  # (i, j, k) -> (lo, hi) degrees
    planes: list = field(default_factory=list)  # soft (ring_a, ring_b, separation) parallel stack
    coplanar: list = field(default_factory=list)  # (i,j,k,l,anchor,cap): hold within cap deg of the graph's
    #   in-plane anchor; None leaves the periodic well seed-selected.
    frozen: set = field(default_factory=set)  # pin to embedded coords
    contacts: tuple = field(default_factory=lambda: (frozenset(), frozenset()))  # soft user/NCI distance and
    #   angular keys; `relaxed()` releases exactly these.

    # --- Coordination fields. All empty for an organic system, so the shared relax is bit-identical there
    # rather than branching. They exist because the metal is embedded as a bond-less surrogate: stripping the
    # M-donor bonds also strips every UFF term that came with them, and these put the missing ones back.
    metals: set = field(default_factory=set)  # metal indices, re-typed to bondless Li on the FF copy (ligand
    #   UFF stays native; Li gives weaker nonbonded contacts than the DG carbon surrogate)
    pulls: dict = field(default_factory=dict)  # pair -> target Angstrom, triple -> preferred angle in degrees;
    #   soft preferences that supplement windows (numeric fixes have separate strict records below)
    floors: dict = field(default_factory=dict)  # (i, j) -> minimum Angstrom: the FF anti-overbond wall
    dg_floors: dict = field(default_factory=dict)  # (i, j) -> the same distance, lowering a bounds-matrix cell:
    #   the bond-less carbon floors every M...X at a carbon vdW contact, forbidding the real geometry (see `compose`)
    shapes: list = field(default_factory=list)  # atom sets held as an all-pairs rigid body (a spectator sphere);
    #   distinguishes a modelled window, which needs a `pull`, from a rigid-body member, which must not get one
    phantoms: frozenset = field(default_factory=frozenset)  # zero-volume dummies: one haptic face's centroid,
    #   standing in for the vertex a Cp/arene presents. Transient (never in a stored Mol; materialised only
    #   inside the embed/relax), so nothing downstream (gate, metrics, dump, calculator) sees one.
    haptic: dict = field(default_factory=dict)  # {centroid dummy -> its ring atoms}: transient scaffolding;
    #   the stored mol and donor list stay real, the face atoms are the donors
    dihedrals: dict = field(default_factory=dict)  # (i, j, k, l) -> periodic (lo, hi) degrees
    fixed: dict = field(default_factory=dict)  # numeric fix pair/triple/quartet -> requested (lo, hi);
    #   lo == hi is a scalar target
    umbrellas: dict = field(default_factory=dict)  # four atoms -> improper: None is planar, 0 keeps only the
    #   seed-side half-space, a positive value the minimum magnitude (soft walls; validate final stereo);
    #   (anchor, weight) is a template planar well with its fraction of the complete shell's force coefficient
    donor_orientation: bool = True  # retain rxembed's M-D-X fold and sp2 donor-plane terms
    conjugation: bool = True  # retain rxembed's organic sp2/conjugation UFF cleanup terms

    @property
    def is_constrained(self) -> bool:
        """True when any geometric or coordinate constraint is set."""
        return bool(self.distances or self.angles or self.dihedrals or self.planes or self.frozen)

    def copy(self, **overrides) -> "Constraints":
        """Return a field-complete deep copy, with any keyword replacing that field outright."""
        return replace(deepcopy(self), **overrides)

    def relaxed(self) -> "Constraints":
        """Return a copy with the seeded NCI/user contacts released, for the exploratory search.

        Released atoms are no longer pose-frozen, so a strained contact may break or a new one form. Every
        structural hold (frozen core, metal sphere, coplanarity cap, centroid dummies) is carried
        through by `copy`. The caller adds fragment contact bounds if releasing leaves a fragment unconstrained.
        """
        dk, angular = self.contacts
        return self.copy(
            distances={k: v for k, v in self.distances.items() if k not in dk or k in self.fixed},
            angles={k: v for k, v in self.angles.items() if k not in angular or k in self.fixed},
            dihedrals={k: v for k, v in self.dihedrals.items() if k not in angular or k in self.fixed},
            planes=[],
            pulls={
                key: value
                for key, value in self.pulls.items()
                if len(key) != ANGLE_ATOMS or not {key, key[::-1]} & angular or {key, key[::-1]} & self.fixed.keys()
            },
            contacts=(frozenset(), frozenset()),
        )

    def constrained_atoms(self) -> set:
        """Return the atoms MC pose-mode must hold so NCI / TS / metal contacts stay intact."""
        s = set(self.frozen)
        for t in (*self.distances, *self.angles, *self.dihedrals, *self.fixed, *self.pulls):
            s |= set(t)
        for ring_a, ring_b, _ in self.planes:
            s |= set(ring_a) | set(ring_b)
        return s - set(self.phantoms)  # a haptic centroid dummy is transient embed scaffolding, never a real atom


def graft_owns(atoms, frozen, haptic=None):
    """Return whether every real atom defining a term belongs to the coordinate graft."""
    real = set()
    haptic = haptic or {}
    for atom in atoms:
        real.update(haptic.get(atom, (atom,)))
    return real.issubset(frozen)


def canonical_key(key):
    """Return an atom-index key in whichever direction sorts first, so a path and its reverse compare equal."""
    key = tuple(key)
    return min(key, key[::-1])


def _central_bond(atoms):
    """Return the bond that owns a torsional degree of freedom."""
    return frozenset(atoms[1:3])


def stated_dihedral(cons, *atoms, improper=False):
    """Match a stated torsion by axis, or an improper by its four represented points."""
    owner = frozenset if improper else _central_bond
    return any(owner(key) == owner(atoms) for key in cons.dihedrals)


def structural_dihedral_owned(cons, atoms):
    """Protect torsional axes and umbrella support from soft replacement.

    A shell improper does not own a ligand rotation merely sharing its numerical torsion axis.
    Point handedness (zero) reserves no soft torsion. Coplanar rows retain their existing axis policy;
    that field mixes donor-plane impropers and proper conjugated torsions.
    """
    axis = _central_bond(atoms)
    return (
        any(_central_bond(key) == axis for key in (*cons.dihedrals, *cons.coplanar))
        or any(len(key) == DIHEDRAL_ATOMS and _central_bond(key) == axis for key in cons.fixed)
        or any(ideal != 0.0 and frozenset(key) == frozenset(atoms) for key, ideal in cons.umbrellas.items())
    )


def _merge_floor(a, b):  # a wall is a physical minimum: the stricter (higher) of two claims on one pair wins
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out[k], v) if k in out else v
    return out


def _merge_relief(a, b):  # a relief is permission to lower, so the fullest (lowest) claim wins; see `compose`
    out = dict(a)
    for k, v in b.items():
        out[k] = min(out[k], v) if k in out else v
    return out


def _merge_exclusive(name, a, b):
    """Merge two dicts, refusing a key collision, for fields where two claims cannot be reconciled."""
    clash = {k for k in b if k in a and a[k] != b[k]}
    if clash:
        raise ValueError(f"compose: conflicting {name} on {sorted(clash)}; give each key one value")
    return {**a, **b}


def periodic_window(window):
    """Return an equivalent dihedral interval centred in (-180, 180], choosing +180 at the boundary."""
    lo, hi = window
    middle = 0.5 * (lo + hi)
    wrapped = (middle + _STRAIGHT) % (2 * _STRAIGHT) - _STRAIGHT
    if wrapped == -_STRAIGHT:
        wrapped = _STRAIGHT
    shift = wrapped - middle
    return lo + shift, hi + shift


def _periodic_keys(d):
    """Return a copy of a fixed record with every dihedral window centred by `periodic_window`."""
    return {k: periodic_window(v) if len(k) == DIHEDRAL_ATOMS else v for k, v in d.items()}


def merge_pulls(a, b):
    """Merge distance targets and canonical angle targets, rejecting conflicting preferences."""
    out = {}
    for terms in (a, b):
        for atoms, value in terms.items():
            key = atoms
            if len(key) not in (DIST_ATOMS, ANGLE_ATOMS):
                raise ValueError(f"pull {key}: expected a distance pair or angle triple")
            if len(key) == ANGLE_ATOMS:
                key = min(key, key[::-1])
                if not all(np.isfinite(v) and 0 <= v <= _STRAIGHT for v in np.atleast_1d(value)):
                    raise ValueError(f"pull {key}: expected finite angles between 0 and 180 degrees")
            if key in out and out[key] != value:
                raise ValueError(f"compose: conflicting pulls on {key}; give each target one value")
            out[key] = value
    return out


_MERGE = {  # field -> how two sources combine. See `compose`.
    "distances": lambda a, b: {**a, **b},  # a spec landing ON a structural hold is a user override (dispatch)
    "angles": lambda a, b: {**a, **b},
    "dihedrals": lambda a, b: {**a, **b},
    "planes": lambda a, b: [*a, *b],  # never de-duplicated: two identical pi-stacks are the caller's business
    "coplanar": lambda a, b: [*a, *b],
    "frozen": lambda a, b: a | b,
    "contacts": lambda a, b: (a[0] | b[0], a[1] | b[1]),
    # equivalent periodic dihedral spellings of one fixed value compare equal
    "fixed": lambda a, b: _merge_exclusive("fixed", _periodic_keys(a), _periodic_keys(b)),
    "metals": lambda a, b: a | b,
    "pulls": merge_pulls,
    "floors": _merge_floor,
    "dg_floors": _merge_relief,
    "shapes": lambda a, b: [*a, *(set(s) for s in b)],
    "phantoms": lambda a, b: a | b,
    "haptic": partial(_merge_exclusive, "haptic"),  # a shared key = two faces claiming one reserved index = corruption
    "umbrellas": partial(_merge_exclusive, "umbrellas"),
    "donor_orientation": lambda a, b: a and b,
    "conjugation": lambda a, b: a and b,
}

_names = {f.name for f in fields(Constraints)}  # a new field must be given a merge policy, not defaulted
if set(_MERGE) != _names:
    raise RuntimeError(f"_MERGE is out of sync with Constraints: {_names ^ set(_MERGE)}")
del _names


def compose(*parts: Constraints) -> Constraints:
    """Merge constraint sets left to right into a new one, per-field, without mutating any input.

    Field-driven via ``_MERGE``, so a new field cannot be silently dropped at a merge site. Fixed numeric
    values win over derived builder windows regardless of part order; conflicting fixed values, pulls and
    haptic claims raise. `floors` takes the max (the stricter wall) and `dg_floors` the min (the fullest
    relief); the two differ in membership, not in which is stricter (see `metal_distance._tier_floor`).
    """
    out = Constraints()
    for part in parts:
        for name, merge in _MERGE.items():
            setattr(out, name, merge(getattr(out, name), getattr(part, name)))
    for key, value in out.fixed.items():
        if len(key) == DIST_ATOMS:
            out.distances[key] = _seed_window(value, _FIX_PAD)
            out.pulls.pop(key, None)
        elif len(key) == ANGLE_ATOMS:
            out.angles.pop(key[::-1], None)
            out.angles[key] = _seed_window(value, _FIX_ANG_PAD)
            out.pulls.pop(key, None)
            out.pulls.pop(key[::-1], None)
        else:
            out.dihedrals.pop(key[::-1], None)
            out.dihedrals[key] = _seed_window(value, _FIX_ANG_PAD)
    return out


def _angular_owner(key):
    """Return the degree of freedom an angle or dihedral key claims: its angle, or its central bond."""
    return ("dihedral", _central_bond(key)) if len(key) == DIHEDRAL_ATOMS else ("angle", canonical_key(key))


def compose_soft(base: Constraints, soft: Constraints) -> Constraints:
    """Compose incoming soft terms without replacing a structural term or another soft owner."""
    base_d, base_a = base.contacts
    base_soft_d = {canonical_key(key) for key in base_d}
    structural_d = {canonical_key(key) for key in base.distances} - base_soft_d
    structural_d |= {canonical_key(key) for key in base.fixed if len(key) == DIST_ATOMS}
    base_soft_a = {_angular_owner(key) for key in base_a}
    incoming_a = [*soft.angles, *soft.dihedrals]
    structural_a = {_angular_owner(key) for key in base.angles} - base_soft_a
    structural_a |= {_angular_owner(key) for key in base.fixed if len(key) == ANGLE_ATOMS}
    overlap = {key for key in soft.distances if canonical_key(key) in base_soft_d} | {
        key for key in incoming_a if _angular_owner(key) in base_soft_a
    }
    if overlap:
        raise ValueError(
            f"soft constraints overlap at {sorted(overlap)}; state each degree of freedom once with either "
            "constrain= or contacts="
        )
    distances = {key: value for key, value in soft.distances.items() if canonical_key(key) not in structural_d}
    angles = {key: value for key, value in soft.angles.items() if _angular_owner(key) not in structural_a}
    dihedrals = {key: value for key, value in soft.dihedrals.items() if not structural_dihedral_owned(base, key)}
    d_soft, a_soft = soft.contacts
    return compose(
        base,
        soft.copy(
            distances=distances,
            angles=angles,
            dihedrals=dihedrals,
            pulls={
                key: value
                for key, value in soft.pulls.items()
                if (
                    canonical_key(key) not in structural_d
                    if len(key) == DIST_ATOMS
                    else _angular_owner(key) not in structural_a
                )
            },
            contacts=(frozenset(d_soft & distances.keys()), frozenset(a_soft & (angles.keys() | dihedrals.keys()))),
        ),
    )


def _seed_window(window, pad):
    """Pad only a scalar target for distance-geometry seeding; preserve an explicit range verbatim."""
    lo, hi = window
    return (lo - pad, hi + pad) if lo == hi else window


def add_distance(d: dict, i: int, j: int, lo: float, hi: float) -> None:
    """Add an order-independent distance window ``(i, j) -> (lo, hi)`` to a distance dict."""
    d[(min(i, j), max(i, j))] = (lo, hi)


def add_pairwise_shape(cons, atoms, positions, pad):
    """Pin the shape of ``atoms`` by all their pairwise distances: a frame-independent rigid hold.

    Each pair becomes a ``(d-pad, d+pad)`` window in ``cons.distances``, which survives a random-frame embed
    where absolute coordinates would not. ``positions`` is anything indexable by atom index: an ``[N,3]``
    conformer array, or a ``{atom: (x, y, z)}`` map when the coords come from another molecule.
    """
    for a, b in itertools.combinations(atoms, 2):
        d = float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
        add_distance(cons.distances, a, b, d - pad, d + pad)


# --- the spec resolver: `fix` / `constrain` -> a `Constraints` ---------------------------------------------

logger = logging.getLogger("rxembed.constraints")  # under the "rxembed" tree `set_verbose` configures

# window pads: fix is tight, with a little slack because ETKDG needs it; constrain is a soft target
_FIX_PAD = 0.02  # numbers-fix distance half-window (Angstrom)
_FIX_ANG_PAD = 2.0  # numbers-fix angular half-window (degrees)
FIX_DISTANCE_TOL = 0.001  # Angstrom: scalar fix input and output agree to three decimal places
# A model M-L window is a fitted target, so FIX_DISTANCE_TOL's 0.001 A would only measure UFF convergence. On
# RITCIG's limiting pair the residual falls as C/fc with C = 0.56 A: 0.01 A needs the 100x rung, 0.001 A fc ~ 560.
ML_WINDOW_TOL = 0.01  # Angstrom: slack on a model-derived (not user-fixed) M-L window
FIX_ANGLE_TOL = 0.005  # angle/dihedral degrees: below two-decimal reporting precision without float equality
_CON_PAD = 0.1  # constrain distance half-window when a scalar target is given
_CON_ANG_PAD = 5.0  # constrain angular half-window
_SHAPE_PAD = 0.05  # graft pairwise-shape half-window (a rigid hold; the exact graft does the real work)
_MIN_SHAPE_ATOMS = 3  # below this a core has only a distance to pin, not an orientable 3-D shape
DIST_ATOMS = 2  # a distance key names two atoms
ANGLE_ATOMS = 3  # an angle key names three atoms
DIHEDRAL_ATOMS = 4  # a dihedral key names four atoms
_FIX_FIELD = {
    DIST_ATOMS: ("distances", _FIX_PAD),
    ANGLE_ATOMS: ("angles", _FIX_ANG_PAD),
    DIHEDRAL_ATOMS: ("dihedrals", _FIX_ANG_PAD),
}


def _site_point(positions, haptic, atom):
    """Return an atom's position, or its haptic face centroid, or None when either is out of range."""
    if 0 <= atom < len(positions):
        return positions[atom]
    face = haptic.get(atom)
    if not face or any(index < 0 or index >= len(positions) for index in face):
        return None
    return np.mean(positions[list(face)], axis=0)


def constraint_value(positions, atoms, haptic=None, window=None):
    """Measure one distance, angle or dihedral, resolving haptic centroids when present."""
    points = [_site_point(positions, haptic or {}, atom) for atom in atoms]
    if any(value is None for value in points):
        return None
    if len(atoms) == DIST_ATOMS:
        return float(np.linalg.norm(points[0] - points[1]))
    if len(atoms) == ANGLE_ATOMS:
        return bond_angle(*points)
    if len(atoms) != DIHEDRAL_ATOMS:
        raise ValueError("a geometric term needs 2, 3 or 4 atoms")
    value = dihedral_angle(*points)
    if window is not None:
        middle = 0.5 * sum(window)
        value = middle + (value - middle + _STRAIGHT) % (2 * _STRAIGHT) - _STRAIGHT
    return value


def out_of_plane_row(mol, row):
    """Return whether a coplanar row caps its first atom out of the donor's own plane: both far atoms bond the donor."""
    donor = row[1]
    return all(mol.GetBondBetweenAtoms(donor, atom) is not None for atom in row[2:4])


def out_of_plane(positions, metal, donor, left, right):
    """Return the angle in degrees between the donor-metal bond and the left-donor-right plane."""
    normal = np.cross(positions[left] - positions[donor], positions[right] - positions[donor])
    bond = positions[metal] - positions[donor]
    scale = np.linalg.norm(normal) * np.linalg.norm(bond)
    if scale == 0.0:
        return float("nan")
    return float(np.degrees(np.arcsin(min(1.0, abs(np.dot(normal, bond)) / scale))))


def plane_torsion_cap(cap, angle):
    """Return the dihedral about one donor bond that puts the metal ``cap`` degrees out of plane at that M-D-X angle.

    The metal's height over the plane is sin(M-D-X) * sin(dihedral offset) bond lengths, so one axis-free cap
    maps to a different dihedral on each donor bond; 90 means the angle leaves no dihedral to cap.
    """
    ratio = math.sin(math.radians(cap)) / max(math.sin(math.radians(angle)), np.finfo(float).eps)
    return 90.0 if ratio >= 1.0 else math.degrees(math.asin(ratio))


def metal_distance_tolerance(atoms, cons):
    """Return the slack publication, and every screen it bounds, allows a metal distance window."""
    return FIX_DISTANCE_TOL if atoms in cons.fixed else ML_WINDOW_TOL


def within_window(value, window, slack=0.0):
    """Return whether a finite measured value lies within a window and tolerance."""
    lo, hi = window
    return value is not None and np.isfinite(value) and lo - slack <= value <= hi + slack


_SHOWN_HITS = 4  # how many ambiguous matches to name before trailing off


def resolve_atom(mol, ref):
    """Resolve an atom index, or the first atom of a SMARTS with exactly one match (`match`), to an index."""
    if isinstance(ref, (int, np.integer)):
        return int(ref)
    return match(mol, ref)[0]


def match(mol, smarts):
    """Return the atom indices of the one SMARTS match, raising if there is not exactly one.

    `GetSubstructMatch` returns () for no match and silently returns the first hit for several, so a
    mistyped or ambiguous pattern can key a constraint to a wrong-but-plausible atom: on 2-chlorobenzyl
    chloride, `[Cl]` hands back the unreactive aryl chloride. Both raises catch that before it looks right.
    """
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"SMARTS {smarts!r} did not parse; check its syntax with Chem.MolFromSmarts")
    hits = mol.GetSubstructMatches(query)
    if not hits:
        raise ValueError(f"SMARTS {smarts!r} matched nothing; check it against this molecule's atoms")
    if len(hits) > 1:
        raise ValueError(
            f"SMARTS {smarts!r} matched {len(hits)} times ({hits[:_SHOWN_HITS]}...); "
            f"narrow the pattern, or choose yourself with mol.GetSubstructMatches(...)"
        )
    return hits[0]


def _window(val, pad, key=None, angle=False, dihedral=False):
    """Return a ``(lo, hi)`` window: a 2-sequence verbatim, a scalar as ``(val-pad, val+pad)``.

    The range is checked here, where the offending key is still in hand. Dihedral windows may cross the
    periodic boundary as ``(170, 190)``; their midpoint remains in the conventional -180 to 180 degree range.
    """
    if isinstance(val, (tuple, list, np.ndarray)):
        given = [float(v) for v in val]
        if len(given) != DIST_ATOMS:
            raise ValueError(
                f"a distance/angle/dihedral value must be a scalar target or a (lo, hi) window; got {val!r} "
                f"(a coordinate belongs under an int key: fix={{i: (x, y, z)}})"
            )
        lo, hi = given
    else:
        given = [float(val)]
        lo, hi = given[0] - pad, given[0] + pad
    if dihedral:
        valid = lo <= hi and hi - lo <= 2 * _STRAIGHT
        label = "dihedral window no wider than 360 degrees"
    else:
        top = _STRAIGHT if angle else None
        valid = lo <= hi and all(v >= 0.0 and (top is None or v <= top) for v in given)
        label = "angle in 0-180 degrees" if angle else "distance in Angstrom"
    if not all(np.isfinite(given)) or not valid:
        raise ValueError(
            f"constraint {key!r}: {val!r} is not a valid {label}; give a scalar target or an ordered (lo, hi) window"
        )
    if dihedral:
        return periodic_window((lo, hi))
    return max(0.0, lo), hi if top is None else min(top, hi)


def is_index(x):
    """Return whether `x` is an integer atom index; a bool is not one."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _index_tuple(key):
    """Return a numeric-coordinate key as atom indices; error clearly if a SMARTS/str slipped in."""
    key = tuple(key)
    for a in key:
        if isinstance(a, str):
            raise ValueError(
                f"constraint key {key!r} contains the string {a!r}: fix=/constrain= are index-driven "
                f"(0-based, xyz/graph order); resolve your SMARTS first, e.g. "
                f"i = mol.GetSubstructMatch(Chem.MolFromSmarts('[F-]'))[0], then key by i"
            )
        if not is_index(a):
            raise ValueError(f"constraint key {key!r} must be integer atom indices, got {type(a).__name__}")
    out = tuple(int(a) for a in key)
    if len(set(out)) != len(out):  # (i, i) is the (i, j) typo: it writes the bounds-matrix diagonal and no-ops
        raise ValueError(f"constraint key {key!r} names the same atom twice; geometric coordinates need distinct atoms")
    if len(out) == ANGLE_ATOMS and out[0] > out[2]:
        out = (out[2], out[1], out[0])
    elif len(out) == DIHEDRAL_ATOMS and out > out[::-1]:
        out = out[::-1]
    return out


def _is_plane_key(key):
    """Return True for a π-stack key: a 2-tuple of ring atom-groups (lists/tuples/None), not two indices."""
    if not (isinstance(key, tuple) and len(key) == DIST_ATOMS):
        return False
    return all(isinstance(e, (list, tuple)) or e is None for e in key)


def _as_coord(val):
    v = tuple(float(x) for x in val)
    if len(v) != 3:  # noqa: PLR2004
        raise ValueError(f"a coordinate fix value must be (x, y, z); got {val!r}")
    return v


def _resolve_ring(mol, ring):
    """Return a π-stack ring atom-index tuple; ``None`` is not auto-detected, so pass the atoms."""
    if ring is None:
        raise ValueError(
            "constrain plane: auto ring detection (None) is not supported; pass the ring atom indices "
            "explicitly, e.g. constrain={(tuple(ring_a), tuple(ring_b)): 3.7} "
            "(get them with mol.GetSubstructMatch / GetRingInfo)"
        )
    atoms = tuple(ring)
    if len(atoms) < 3:  # noqa: PLR2004
        raise ValueError(f"constrain plane needs at least 3 atoms per ring; got {atoms}")
    if any(not is_index(atom) for atom in atoms):
        raise ValueError(f"constrain plane ring atoms must be integer indices; got {atoms}")
    atoms = tuple(int(atom) for atom in atoms)
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"constrain plane ring atoms must be distinct; got {atoms}")
    return atoms


def _validate_indices(mol, atoms):
    n = mol.GetNumAtoms()
    for a in atoms:
        if not (0 <= a < n):
            raise ValueError(f"atom index {a} out of range (molecule has {n} atoms, 0..{n - 1})")


def _apply_fix(mol, fix, cons, coord_fix, has_geometry):
    """Resolve ``fix`` into `cons`/`coord_fix`; return the numbers-fix keys it wrote (normalised).

    A dict mixes coordinate pins (int key, grafted) and fixed numeric coordinates (tuple key, UFF-restrained);
    a list, tuple or set holds the named atoms at the source's own coords, which needs a geometry.
    """
    written: set = set()
    if fix is None:
        return written
    if not isinstance(fix, (dict, list, tuple, set)):
        raise ValueError(
            f"fix= takes a list of atom indices (own-coords graft) or a dict "
            f"({{i: (x,y,z)}} coords / {{(i,j): d}} numbers); got {type(fix).__name__} {fix!r}"
            + (f"; did you mean fix=[{fix}]?" if is_index(fix) else "")
        )
    if isinstance(fix, dict):
        for key, val in fix.items():
            if is_index(key):  # int key -> a coordinate pin (graft)
                coord_fix[int(key)] = _as_coord(val)
            else:  # tuple key -> a scalar target or explicit allowed range
                idx = _index_tuple(key)
                if len(idx) not in _FIX_FIELD:
                    raise ValueError(f"fix number key {key!r} needs 2 (distance), 3 (angle) or 4 (dihedral) atoms")
                name, pad = _FIX_FIELD[len(idx)]
                idx = (min(idx), max(idx)) if len(idx) == DIST_ATOMS else idx
                requested = _window(val, 0.0, key, angle=len(idx) == ANGLE_ATOMS, dihedral=len(idx) == DIHEDRAL_ATOMS)
                if cons.fixed.get(idx, requested) != requested:
                    raise ValueError(f"fix= gives conflicting values for {idx}; state it once")
                getattr(cons, name)[idx] = _seed_window(requested, pad)
                cons.fixed[idx] = requested
                written.add(idx)
        return written
    # a list/tuple/set of atom indices -> hold at the source's own coords (graft)
    if not has_geometry:
        raise ValueError(
            "fix=[atoms] holds them at the source's own coordinates, but this source has no "
            "geometry (a SMILES). Give numeric distances/angles/dihedrals or explicit "
            "coordinates from a reference (fix={i: (x, y, z)})."
        )
    positions = mol.GetConformer().GetPositions()
    for a in fix:
        if isinstance(a, str) or not is_index(a):
            raise ValueError(
                f"fix=[...] takes atom indices; got {a!r}. Resolve SMARTS yourself first "
                f"(fix=/constrain= are index-driven)."
            )
        coord_fix[int(a)] = tuple(positions[int(a)])
    return written


def _graft_core(mol, cons, coord_fix):
    """Turn the collected coordinate pins into a rigid graftable core: pairwise-shape bias + freeze + advisory."""
    if not coord_fix:
        return
    _validate_indices(mol, coord_fix)
    add_pairwise_shape(cons, list(coord_fix), coord_fix, _SHAPE_PAD)
    cons.frozen |= set(coord_fix)
    if len(coord_fix) < _MIN_SHAPE_ATOMS:
        logger.warning(
            "fix: %d coordinate atom(s); orientation needs at least 3 (%s)",
            len(coord_fix),
            "one atom fixes nothing (held wherever the embed lands it)"
            if len(coord_fix) == 1
            else "two atoms fix only the bond length",
        )


def _apply_constrain(mol, constrain, cons):
    """Resolve ``constrain`` into `cons`; return its releasable distance and angular keys."""
    d_soft: set = set()  # releasable soft distance keys: provenance for mc(explore=)
    angular_soft: set = set()  # releasable soft angle and dihedral keys
    if not constrain:
        return d_soft, angular_soft
    for key, val in constrain.items():
        if _is_plane_key(key):
            ra, rb = key
            ra, rb, separation = _resolve_ring(mol, ra), _resolve_ring(mol, rb), float(val)
            if set(ra) & set(rb):
                raise ValueError("constrain plane rings must not share atoms")
            if not np.isfinite(separation) or separation <= 0.0:
                raise ValueError(f"constrain plane separation must be a positive finite distance; got {val!r}")
            cons.planes.append((ra, rb, separation))
            continue
        idx = _index_tuple(key)
        if len(idx) == DIST_ATOMS:
            add_distance(cons.distances, *idx, *_window(val, _CON_PAD, key))
            d_soft.add((min(idx), max(idx)))
        elif len(idx) == ANGLE_ATOMS:
            cons.angles[idx] = _window(val, _CON_ANG_PAD, key, angle=True)
            angular_soft.add(idx)
        elif len(idx) == DIHEDRAL_ATOMS:
            cons.dihedrals[idx] = _window(val, _CON_ANG_PAD, key, dihedral=True)
            angular_soft.add(idx)
        else:
            raise ValueError(f"constrain key {key!r} needs 2 (distance), 3 (angle) or 4 (dihedral) atoms")
    return d_soft, angular_soft


def reference_positions(reference):
    """Return a template reference's ``(N, 3)`` coordinates, from a Mol, an ``.xyz`` path, or an array.

    A template needs coordinates, not bonds, so an ``.xyz`` is read with `Chem.MolFromXYZFile` and never
    perceived. That is what lets a TS with a hypervalent reacting carbon serve as a reference at all.
    """
    if isinstance(reference, Chem.Mol):
        if reference.GetNumConformers() == 0:
            raise ValueError("a template reference Mol needs a conformer (a 3D geometry)")
        return reference.GetConformer().GetPositions()
    if isinstance(reference, (str, os.PathLike)):
        path = os.fspath(reference)
        mol = Chem.MolFromXYZFile(path)
        if mol is None:
            raise ValueError(f"could not read coordinates from {path!r}; give an .xyz file path")
        return mol.GetConformer().GetPositions()
    arr = np.asarray(reference, float)
    if arr.ndim == 2 and arr.shape[1] == 3:  # noqa: PLR2004  a 2D (N, 3) array
        return arr
    raise ValueError("a template reference must be a Mol with a conformer, an .xyz path, or an (N, 3) array")


def template_to_fix(template, fix=None, own=None, target=None):
    """Fold ``template=(reference, map_or_smarts)`` into a coordinate ``fix``; sugar, not a mechanism.

    Every rigid spec answers one question, where do these atoms' coordinates come from: your own geometry
    (`fix=[i, j]`, resolved against `own`), coordinates you supply (`fix={i: (x, y, z)}`), or another
    molecule (`template=`). They compose, with an explicit `fix` on the same atom winning.

    Coordinates carry handedness, which a distance/angle spec cannot and may give the mirror image
    silently; negate one axis of the reference to embed the other diastereomer deliberately. A SMARTS needs
    one ordered correspondence on each graph, so a symmetric query needs an explicit map instead.
    """
    if not (
        isinstance(template, (tuple, list))
        and len(template) == 2  # noqa: PLR2004  template=(reference, mapping)
        and isinstance(template[1], (dict, str))
    ):
        raise ValueError(
            "template= must be (reference, SMARTS) or (reference, {target_index: reference_index}). "
            "Use a SMARTS when both molecules carry the same graph; an .xyz needs an explicit map."
        )
    reference, mapping = template
    if isinstance(mapping, str):
        if target is None or not isinstance(reference, Chem.Mol):
            raise ValueError(
                "template=(reference, SMARTS) needs target and reference molecular graphs; "
                "use an explicit index map for an .xyz or coordinate array"
            )
        target_match, reference_match = match(target, mapping), match(reference, mapping)
        query = Chem.MolFromSmarts(mapping)
        if (
            len(target.GetSubstructMatches(query, uniquify=False, maxMatches=2)) > 1
            or len(reference.GetSubstructMatches(query, uniquify=False, maxMatches=2)) > 1
        ):
            raise ValueError(
                f"template SMARTS {mapping!r} has symmetry-equivalent atom orderings; "
                "give an explicit {target_index: reference_index} map"
            )
        mapping = dict(zip(target_match, reference_match, strict=True))
    if not mapping:
        raise ValueError("template= was given an empty atom map, which grafts nothing; map at least one atom")
    positions = reference_positions(reference)
    coords = {}
    for ti, ri in mapping.items():
        if not (0 <= int(ri) < len(positions)):
            raise ValueError(
                f"template map {ti!r}->{ri!r}: reference index {ri} out of range "
                f"(the reference has {len(positions)} atoms, 0..{len(positions) - 1})"
            )
        coords[int(ti)] = tuple(positions[int(ri)])
    if isinstance(fix, dict):
        return {**coords, **fix}
    if fix:  # a list of atoms held at the source's own coordinates: another coordinate source, so merge it
        if own is None:
            raise ValueError(
                "template= with fix=[atoms] needs the source's own geometry to resolve that list. Give those "
                "atoms as coordinates instead (fix={i: (x, y, z)}), or embed without the list."
            )
        coords.update({int(i): tuple(own[int(i)]) for i in fix})
    return coords


def resolve_core(mol, *, fix=None, constrain=None, has_geometry=False):
    """Resolve ``fix`` / ``constrain`` into ``(Constraints, ref)``.

    ``ref`` is ``{atom_idx: (x, y, z)}``: the atoms to Kabsch-graft onto their exact geometry after the
    embed, empty when nothing is grafted (a numbers-only fix, or a constrain-only spec). The caller edits the
    bounds matrix from `Constraints`, embeds, grafts ``ref``, then runs the restrained UFF pull.
    """
    cons = Constraints()
    coord_fix: dict[int, tuple] = {}  # atoms grafted to exact coords (own / explicit / from a reference)

    fixed = _apply_fix(mol, fix, cons, coord_fix, has_geometry)
    _graft_core(mol, cons, coord_fix)
    d_soft, angular_soft = _apply_constrain(mol, constrain, cons)
    clash = sorted(fixed & (d_soft | angular_soft))
    if clash:  # `constrain` is applied last, so it would overwrite the fix and make it releasable
        raise ValueError(
            f"fix= and constrain= both name {clash}: one key, two contradictory intents. Keep the rigid "
            f"one in fix= (or move it to constrain= if the search may move off it)."
        )
    dropped = _drop_determined_by_graft(cons, coord_fix, fixed | d_soft | angular_soft)
    fixed, d_soft, angular_soft = fixed - dropped, d_soft - dropped, angular_soft - dropped
    for key in dropped:
        cons.fixed.pop(key, None)

    _validate_indices(
        mol,
        {a for k in cons.distances for a in k}
        | {a for k in cons.angles for a in k}
        | {a for k in cons.dihedrals for a in k}
        | {a for ra, rb, _ in cons.planes for a in (*ra, *rb)},
    )
    cons.contacts = (frozenset(d_soft), frozenset(angular_soft))
    _warn_underdetermined(cons, coord_fix, len(fixed & set(cons.distances)))
    _echo(mol, cons, coord_fix)
    return cons, coord_fix


def _drop_determined_by_graft(cons, coord_fix, keys):
    """Drop each user window whose atoms all sit in the grafted core; return the keys dropped.

    The Kabsch graft restores those atoms exactly after the embed, so such a window can never be realised
    and, worse, would seed the embed against a distance the graft then contradicts. Restore the shape
    window (a distance) or drop it (an angle/dihedral), loudly, so the user is not left thinking it applied.
    """
    dropped = {key for key in keys if graft_owns(key, coord_fix)}
    for key in sorted(dropped):
        if len(key) == DIST_ATOMS:
            add_pairwise_shape(cons, key, coord_fix, _SHAPE_PAD)  # back to what the graft implies
        elif len(key) == ANGLE_ATOMS:
            cons.angles.pop(key, None)
        else:
            cons.dihedrals.pop(key, None)
        logger.warning(
            "constraint %s is inside the fixed core, where the graft wins; dropped",
            key,
        )
    return dropped


def _warn_underdetermined(cons, coord_fix, n_fix_d):
    """Warn when two fix distances sharing an atom over exactly 3 atoms leave their angle free.

    The classic mis-fix is an SN2 pinned as ``fix={(f, c): d1, (c, cl): d2}`` with no ``(f, c, cl)`` angle,
    which comes out bent rather than linear. A richer network (more atoms, coordinates or angles) is left alone.
    """
    if coord_fix or cons.angles or cons.dihedrals or n_fix_d < DIST_ATOMS:
        return
    keys = list(cons.distances)
    atoms = {a for k in keys for a in k}
    shares = any(len(set(a) & set(b)) == 1 for i, a in enumerate(keys) for b in keys[i + 1 :])
    if len(atoms) <= _MIN_SHAPE_ATOMS and shares:
        logger.warning(
            "fix has two distances sharing an atom over %d atom(s); no angle is fixed, so the angle is free",
            len(atoms),
        )


def _echo(mol, cons, coord_fix):
    """Echo what was pinned, with element symbols, so a wrong (e.g. 1-based) index is obvious.

    The safety net for the commonest porting error, a 1-based index or a picked hydrogen: ``set_verbose`` at
    INFO, then read back ``fix d(C0,F5)->2.02A`` and check the atoms at a glance. A numbers-``fix`` is
    confirmed exactly with ``.measure()`` after the embed.
    """
    if not cons.is_constrained:
        return
    d_soft, angular_soft = cons.contacts
    a_soft, t_soft = angular_soft & cons.angles.keys(), angular_soft & cons.dihedrals.keys()

    parts = []
    if coord_fix:
        parts.append(f"graft {', '.join(atom_label(mol, i) for i in sorted(coord_fix))}")
    for atoms, (lo, hi) in cons.fixed.items():
        tol, unit = (FIX_DISTANCE_TOL, "A") if len(atoms) == DIST_ATOMS else (FIX_ANGLE_TOL, "deg")
        requested = f"{lo:.6f}+/-{tol:g}" if lo == hi else f"[{lo:.6f},{hi:.6f}]"
        name = {2: "d", 3: "angle", 4: "dihedral"}[len(atoms)]
        parts.append(f"fix {name}({','.join(atom_label(mol, i) for i in atoms)})->{requested}{unit}")
    for (i, j), (lo, hi) in cons.distances.items():
        if (i, j) in cons.fixed or (i, j) in d_soft or (i in cons.frozen and j in cons.frozen):
            continue  # already logged canonically, soft (below), or implied by the coordinate graft
        parts.append(f"fix d({atom_label(mol, i)},{atom_label(mol, j)})->{lo:.2f}-{hi:.2f}A")
    for name, windows, soft in (("angle", cons.angles, a_soft), ("dihedral", cons.dihedrals, t_soft)):
        for atoms, (lo, hi) in windows.items():
            if atoms in cons.fixed or atoms in soft:
                continue
            parts.append(f"fix {name}({','.join(atom_label(mol, i) for i in atoms)})->{lo:.1f}-{hi:.1f}deg")
    for i, j in d_soft:
        lo, hi = cons.distances[(i, j)]
        parts.append(f"soft d({atom_label(mol, i)},{atom_label(mol, j)})->{lo:.2f}-{hi:.2f}A")
    for name, windows, soft in (("angle", cons.angles, a_soft), ("dihedral", cons.dihedrals, t_soft)):
        for atoms in soft:
            lo, hi = windows[atoms]
            parts.append(f"soft {name}({','.join(atom_label(mol, i) for i in atoms)})->{lo:.1f}-{hi:.1f}deg")
    for ra, rb, sep in cons.planes:
        parts.append(f"stack {len(ra)}x{len(rb)} ring @ {sep:.1f}A")
    for part in parts:
        logger.info("resolve: %s", part)
