"""Hold the `Constraints` record every builder fills and every stage reads, and resolve ``fix`` and ``constrain``.

``fix`` is rigid: the named atoms get this geometry, given as a list (their own coordinates, grafted by Kabsch), ``{i:
(x, y, z)}`` (explicit coordinates) or ``{(i, j): d, ...}`` (a distance, angle or dihedral held within
`FIX_DISTANCE_TOL` or `FIX_ANGLE_TOL`, or an explicit window); a dict may mix all three. ``constrain`` is soft: a wider
window a real energy may overrule, and the home of pi stacks (``(ring_a, ring_b): separation``).
``fix=Template(reference, map_or_smarts)`` copies a reference's coordinates. A scalar fix carries a restraint and an
acceptance gate; a soft window carries neither. Keys are 0-based atom indices; README.md has worked examples.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Any, NamedTuple

import numpy as np
from rdkit import Chem

from rxembed.utils import STRAIGHT, atom_label, bond_angle, dihedral_angle, require_mol

# deg: wide enough for a real crystal plane's own spread, narrow enough to stay outside the pyramid basin
_PLANAR_CAP = 15.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Improper:
    """Hold one umbrella improper: its kind, which only `window` and `side_only` read, and its share of the force.

    A ``plane`` holds the improper within `_PLANAR_CAP` of `anchor` (0 or 180, else the nearer); a ``pyramid`` holds its
    magnitude between `minimum` and 90 on the side it stands; a ``side`` holds only that side.
    """

    kind: str
    anchor: float | None = None
    minimum: float = 0.0
    weight: float = 1.0

    def __post_init__(self):
        """Refuse a kind that is not a plane, pyramid or side."""
        if self.kind not in ("plane", "pyramid", "side"):
            raise ValueError(f"improper kind {self.kind!r} is not 'plane', 'pyramid' or 'side'")

    @property
    def side_only(self):
        """Return whether the row holds only the side of the plane, so it reserves no torsion value."""
        return self.kind == "side"

    def window(self, phi):
        """Return the ``(lo, hi)`` degrees this row holds an improper measured at `phi` within."""
        if self.kind == "plane":
            anchor = self.anchor
            if anchor is None:
                anchor = 0.0 if abs(phi) <= STRAIGHT / 2 else math.copysign(STRAIGHT, phi)
            return anchor - _PLANAR_CAP, anchor + _PLANAR_CAP  # RDKit wraps periodic bounds, so no clip at 180
        if self.kind == "pyramid":
            return (self.minimum, STRAIGHT / 2) if phi >= 0 else (-STRAIGHT / 2, -self.minimum)
        return (0.0, STRAIGHT) if phi >= 0 else (-STRAIGHT, 0.0)  # a side row

    def torsions(self, key):
        """Return the orders of `key` the row holds: `key`, or one per central pair for a side-only row.

        Holding a side about every axis means carrier order cannot change the penalty.
        """
        if not self.side_only:
            return (key,)
        return tuple({frozenset(order[1:3]): order for order in itertools.permutations(key)}.values())


@dataclass(frozen=True, slots=True, kw_only=True)
class _PlaneKind:
    """Name what the coplanar rows of one kind measure and hold (`CoplanarRow`)."""

    out_of_plane: bool = False
    holds_donor: bool = False
    seeded: bool = False


COPLANAR_KINDS = {  # kind -> what its rows measure and hold; each writer names its rows' kind
    "donor_plane": _PlaneKind(out_of_plane=True, holds_donor=True),  # M out of an sp2 donor's own X-D-Y plane
    "donor_torsion": _PlaneKind(holds_donor=True, seeded=True),  # M-D-C-X off a one-neighbour sp2 donor
    "ring_hinge": _PlaneKind(holds_donor=True),  # M-D-D'-X across a small conjugated chelate ring
    "metal_ez": _PlaneKind(holds_donor=True),  # M-D=C-X, the metal standing in as an imine's E/Z reference
    "ligand_ez": _PlaneKind(),  # X-C=C-Y, a specified ligand double bond
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CoplanarRow:
    """Hold four atoms ``(i, j, k, l)`` within `cap` degrees of one plane, read as its `kind` says.

    An ``out_of_plane`` kind caps the angle between the i-j bond and j's own j-k-l plane, read through the dihedrals
    about both j-k and j-l; any other kind caps the i-j-k-l dihedral. A ``holds_donor`` kind's i is a metal and j the
    donor it orients. `anchor` is the well, 0 (syn) or 180 (anti); a ``seeded`` kind has none and holds the seed's
    nearer well.
    """

    kind: str
    atoms: tuple[int, int, int, int]
    cap: float
    anchor: float | None = None

    def __post_init__(self):
        """Refuse an anchor a seeded kind states or another kind leaves out."""
        if (self.anchor is None) != COPLANAR_KINDS[self.kind].seeded:
            raise ValueError(f"a {self.kind} coplanar row {'takes no' if self.seeded else 'needs an'} anchor")

    @property
    def out_of_plane(self):
        """Return whether the row caps the metal's angle out of the donor's own plane rather than one dihedral."""
        return COPLANAR_KINDS[self.kind].out_of_plane

    @property
    def holds_donor(self):
        """Return whether the row's first atom is a metal and its second the donor whose direction it fixes."""
        return COPLANAR_KINDS[self.kind].holds_donor

    @property
    def seeded(self):
        """Return whether the seed, not the row, picks the well it holds."""
        return COPLANAR_KINDS[self.kind].seeded

    @property
    def axes(self):
        """Return the ``(axis, far)`` atom pairs whose j-axis-far dihedrals the row is read through.

        An out-of-plane cap belongs to neither donor bond, so it is shared between both.
        """
        _i, _j, k, w = self.atoms
        return ((k, w), (w, k)) if self.out_of_plane else ((k, w),)


class FrozenDict(dict):
    """Hold one mapping of a frozen record: a dict that refuses every in-place write and pickles as a copy."""

    __slots__ = ()

    def _refuse(self, *_args, **_kwargs):
        raise TypeError("this record is frozen; build the new mapping and pass it to dataclasses.replace")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):
        """Rebuild from a plain copy, since unpickling a dict writes its items one at a time."""
        return type(self), (dict(self),)


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftKeys:
    """Name the soft keys a search may release: `distances` pairs and `angular` angle and dihedral keys."""

    distances: set | frozenset = frozenset()
    angular: set | frozenset = frozenset()

    def __post_init__(self):
        """Freeze both key sets."""
        object.__setattr__(self, "distances", frozenset(self.distances))
        object.__setattr__(self, "angular", frozenset(self.angular))

    def __bool__(self):
        """Return whether any key is soft."""
        return bool(self.distances or self.angular)


@dataclass(frozen=True, slots=True, kw_only=True)
class Constraints:
    """Carry named geometric windows and rigid holds between stages; construct every field by keyword.

    A record never changes once built: mappings refuse writes, sequences are tuples and sets frozensets. A stage builds
    a new record with `dataclasses.replace` or `compose`.
    """

    __hash__ = None  # frozen, but its mappings are not hashable

    distances: dict = field(default_factory=dict)  # (i, j) -> (lo, hi) Angstrom
    angles: dict = field(default_factory=dict)  # (i, j, k) -> (lo, hi) degrees
    planes: list | tuple = ()  # soft (ring_a, ring_b, separation) parallel stack
    coplanar: list | tuple = ()  # `CoplanarRow`s
    frozen: set | frozenset = frozenset()  # pin to embedded coords
    contacts: SoftKeys = SoftKeys()  # the soft user/NCI keys `relaxed()` releases

    # coordination fields, empty for an organic: they restore the UFF terms the stripped metal bonds took
    metals: set | frozenset = frozenset()  # metal indices, re-typed to bondless Li on the FF copy (ligand
    #   UFF stays native; Li gives weaker nonbonded contacts than the DG carbon surrogate)
    pulls: dict = field(default_factory=dict)  # pair or triple -> preferred (lo, hi) Angstrom or degrees, a point
    #   target t as (t, t): soft preferences that supplement windows (numeric fixes have their own record below)
    floors: dict = field(default_factory=dict)  # (i, j) -> minimum Angstrom: the FF anti-overbond wall
    dg_floors: dict = field(default_factory=dict)  # (i, j) -> the same distance, lowering a bounds-matrix cell:
    #   the bond-less carbon floors every M...X at a carbon vdW contact, forbidding the real geometry (see `compose`)
    haptic: dict = field(default_factory=dict)  # {centroid dummy -> its ring atoms}: transient scaffolding;
    #   the stored mol and donor list stay real, the face atoms are the donors
    dihedrals: dict = field(default_factory=dict)  # (i, j, k, l) -> periodic (lo, hi) degrees
    fixed: dict = field(default_factory=dict)  # numeric fix pair/triple/quartet -> requested (lo, hi);
    #   lo == hi is a scalar target
    umbrellas: dict = field(default_factory=dict)  # four atoms -> `Improper` (soft walls; validate final stereo)
    conjugation: bool = True  # retain rxembed's organic sp2/conjugation UFF cleanup terms
    seed_cpci: dict = field(default_factory=dict)  # numerical pair preferences used only by native DG refinement

    def __post_init__(self):
        """Centre each fixed dihedral, give each fixed key its seed window in place of any pull, then freeze each field.

        Centring makes equivalent periodic spellings of one fixed dihedral compare equal when records compose.
        """
        distances, angles, dihedrals, pulls = (
            dict(getattr(self, name)) for name in ("distances", "angles", "dihedrals", "pulls")
        )
        fixed = {key: periodic_window(v) if len(key) == DIHEDRAL_ATOMS else v for key, v in self.fixed.items()}
        for key, value in fixed.items():
            if len(key) == DIST_ATOMS:
                distances[key] = _seed_window(value, _FIX_PAD)
            elif len(key) == ANGLE_ATOMS:
                angles.pop(key[::-1], None)
                angles[key] = _seed_window(value, _FIX_ANG_PAD)
            else:
                dihedrals.pop(key[::-1], None)
                dihedrals[key] = _seed_window(value, _FIX_ANG_PAD)
            pulls.pop(key, None)
            pulls.pop(key[::-1], None)
        changed = {"distances": distances, "angles": angles, "dihedrals": dihedrals, "pulls": pulls, "fixed": fixed}
        for name, value in changed.items():
            object.__setattr__(self, name, value)
        for f in fields(self):
            object.__setattr__(self, f.name, _FREEZE.get(f.name, FrozenDict)(getattr(self, f.name)))

    @property
    def phantoms(self) -> frozenset:
        """Return the zero-volume centroid dummies, one per haptic face; transient, never in a stored Mol."""
        return frozenset(self.haptic)

    @property
    def is_constrained(self) -> bool:
        """Return whether any geometric or coordinate constraint is set."""
        return bool(self.distances or self.angles or self.dihedrals or self.planes or self.frozen)

    def relaxed(self) -> "Constraints":
        """Return a copy with the seeded contacts released, for an exploratory search.

        Released atoms are no longer frozen, so a strained contact may break or a new one form; every structural hold
        stays. The caller bounds any fragment releasing leaves free.
        """
        dk, angular = self.contacts.distances, self.contacts.angular
        return replace(
            self,
            distances={k: v for k, v in self.distances.items() if k not in dk or k in self.fixed},
            angles={k: v for k, v in self.angles.items() if k not in angular or k in self.fixed},
            dihedrals={k: v for k, v in self.dihedrals.items() if k not in angular or k in self.fixed},
            planes=(),
            pulls={
                key: value
                for key, value in self.pulls.items()
                if len(key) != ANGLE_ATOMS or not {key, key[::-1]} & angular or {key, key[::-1]} & self.fixed.keys()
            },
            contacts=SoftKeys(),
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


def donor_directions(angles, coplanar, metals, *, releasable=(), fixed=()):
    """Return the donors whose structural direction owns their coordination geometry.

    A releasable angle biases only itself, and an unrestricted one gives no direction; fixed records keep authority.
    Requested dihedrals own their axis or improper separately (`requested_dihedral`).
    """
    released = {canonical_key(key) for key in releasable} - {canonical_key(key) for key in fixed}
    donors = {
        centre
        for (left, centre, right), window in angles.items()
        if (left in metals or right in metals)
        and canonical_key((left, centre, right)) not in released
        and (np.min(window) > 0 or np.max(window) < STRAIGHT)
    }
    donors.update(row.atoms[1] for row in coplanar if row.holds_donor)
    return donors


def requested_dihedral(cons, *atoms, improper=False):
    """Return whether `cons` states a dihedral about the j-k axis of `atoms`, or with `improper`, on its four atoms."""
    if improper:
        return any(frozenset(key) == frozenset(atoms) for key in cons.dihedrals)
    return any(frozenset(key[1:3]) == frozenset(atoms[1:3]) for key in cons.dihedrals)


def structural_dihedral_owned(cons, atoms):
    """Return whether a structural term owns the torsion `atoms`, so no soft term may replace it.

    A shell improper does not own a ligand rotation that merely shares its axis, a side-only improper reserves nothing,
    and every coplanar row reserves its j-k bond.
    """
    axis = frozenset(atoms[1:3])  # a torsion's degree of freedom is its j-k bond
    return (
        any(frozenset(key[1:3]) == axis for key in (*cons.dihedrals, *(row.atoms for row in cons.coplanar)))
        or any(len(key) == DIHEDRAL_ATOMS and frozenset(key[1:3]) == axis for key in cons.fixed)
        or any(not row.side_only and frozenset(key) == frozenset(atoms) for key, row in cons.umbrellas.items())
    )


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
    wrapped = (middle + STRAIGHT) % (2 * STRAIGHT) - STRAIGHT
    if wrapped == -STRAIGHT:
        wrapped = STRAIGHT
    shift = wrapped - middle
    return lo + shift, hi + shift


def merge_pulls(a, b):
    """Merge distance and angle preference windows on canonical keys, refusing a malformed row or a conflict.

    Every pull is a ``(lo, hi)`` window on a distance pair or an angle triple; a point target t is ``(t, t)``.
    """
    out = {}
    for terms in (a, b):
        for atoms, window in terms.items():
            if len(atoms) not in (DIST_ATOMS, ANGLE_ATOMS) or np.shape(window) != (2,):
                raise ValueError(
                    f"pull {atoms}: {window!r} must be a (lo, hi) window on a distance pair or angle triple; "
                    "write a target t as (t, t)"
                )
            key = canonical_key(atoms)
            if len(key) == ANGLE_ATOMS and not all(np.isfinite(v) and 0 <= v <= STRAIGHT for v in window):
                raise ValueError(f"pull {key}: expected finite angles between 0 and 180 degrees")
            if key in out and out[key] != tuple(window):
                raise ValueError(f"conflicting pulls on {key}; give each target one value")
            out[key] = tuple(window)
    return out


_MERGE = {  # field -> how two sources combine. See `compose`.
    "distances": lambda a, b: {**a, **b},  # a spec landing ON a structural hold is a user override (dispatch)
    "angles": lambda a, b: {**a, **b},
    "dihedrals": lambda a, b: {**a, **b},
    "planes": lambda a, b: [*a, *b],  # never de-duplicated: two identical pi-stacks are the caller's business
    "coplanar": lambda a, b: [*a, *b],
    "frozen": lambda a, b: a | b,
    "contacts": lambda a, b: SoftKeys(distances=a.distances | b.distances, angular=a.angular | b.angular),
    "fixed": partial(_merge_exclusive, "fixed"),
    "metals": lambda a, b: a | b,
    "pulls": merge_pulls,
    # a wall is a physical minimum, so the stricter (higher) claim on one pair wins
    "floors": lambda a, b: {**a, **{k: max(a.get(k, v), v) for k, v in b.items()}},
    # a relief is permission to lower, so the fullest (lowest) claim wins
    "dg_floors": lambda a, b: {**a, **{k: min(a.get(k, v), v) for k, v in b.items()}},
    "haptic": partial(_merge_exclusive, "haptic"),  # a shared key = two faces claiming one reserved index = corruption
    "umbrellas": partial(_merge_exclusive, "umbrellas"),
    "conjugation": lambda a, b: a and b,
    "seed_cpci": partial(_merge_exclusive, "seed_cpci"),
}


def _frozen_sequences(mapping):
    """Own sequence values as tuples and retain scalar window shorthand as immutable numbers."""
    return FrozenDict({key: tuple(value) if np.ndim(value) else float(value) for key, value in mapping.items()})


_FREEZE = {  # field -> its immutable form (`Constraints.__post_init__`); a field without a row is a mapping
    **dict.fromkeys(("distances", "angles", "dihedrals", "fixed", "haptic"), _frozen_sequences),
    "planes": lambda rows: tuple((tuple(a), tuple(b), separation) for a, b, separation in rows),
    "coplanar": tuple,
    "frozen": frozenset,
    "metals": frozenset,
    "contacts": lambda keys: keys,  # a `SoftKeys`, frozen by its constructor
    "pulls": lambda pulls: FrozenDict(merge_pulls(pulls, {})),
    "conjugation": bool,
}


def compose(*parts: Constraints) -> Constraints:
    """Merge constraint sets left to right into a new one, field by field, without changing any input.

    Fields merge by `_MERGE`, so a new field cannot be dropped silently. Fixed values win over derived windows in any
    order, and conflicting fixed values, pulls or haptic claims raise. `floors` keeps the stricter wall and `dg_floors`
    the fuller relief.
    """
    empty = Constraints()
    merged: dict[str, Any] = {f.name: getattr(empty, f.name) for f in fields(Constraints)}
    for part in parts:  # a field without a _MERGE row raises KeyError rather than being dropped
        merged = {name: _MERGE[name](value, getattr(part, name)) for name, value in merged.items()}
    return Constraints(**merged)


def _angular_owner(key):
    """Return the degree of freedom an angle or dihedral key claims: its angle, or its central bond."""
    return ("dihedral", frozenset(key[1:3])) if len(key) == DIHEDRAL_ATOMS else ("angle", canonical_key(key))


def retained_soft(base: Constraints, soft: Constraints) -> Constraints:
    """Return incoming soft terms whose degrees of freedom `base` leaves free, refusing another soft owner."""
    base_d, base_a = base.contacts.distances, base.contacts.angular
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
    d_soft, a_soft = soft.contacts.distances, soft.contacts.angular
    return replace(
        soft,
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
        contacts=SoftKeys(distances=d_soft & distances.keys(), angular=a_soft & (angles.keys() | dihedrals.keys())),
    )


def _seed_window(window, pad):
    """Pad only a scalar target for distance-geometry seeding; preserve an explicit range verbatim."""
    lo, hi = window
    return (lo - pad, hi + pad) if lo == hi else window


def add_distance(d: dict, i: int, j: int, lo: float, hi: float) -> None:
    """Add an order-independent distance window ``(i, j) -> (lo, hi)`` to a distance dict."""
    d[canonical_key((i, j))] = (lo, hi)


def add_pairwise_shape(distances, atoms, positions, pad):
    """Pin the shape of `atoms` by all their pairwise distances, a hold that survives any embedding frame.

    Each pair becomes a ``(d-pad, d+pad)`` distance window. `positions` is an ``[N, 3]`` array or ``{atom: xyz}``.
    """
    for a, b in itertools.combinations(atoms, 2):
        d = float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
        add_distance(distances, a, b, d - pad, d + pad)


# --- the spec resolver: `fix` / `constrain` -> a `Constraints` ---------------------------------------------

logger = logging.getLogger("rxembed.constraints")  # under the "rxembed" tree `set_verbose` configures

# window pads: fix is tight, with a little slack because ETKDG needs it; constrain is a soft target
_FIX_PAD = 0.02  # numbers-fix distance half-window (Angstrom)
_FIX_ANG_PAD = 2.0  # numbers-fix angular half-window (degrees)
FIX_DISTANCE_TOL = 0.001  # Angstrom: scalar fix input and output agree to three decimal places
# a model M-L window is a fitted target, so its slack is wider than FIX_DISTANCE_TOL's convergence test
ML_WINDOW_TOL = 0.01  # Angstrom: slack on a model-derived (not user-fixed) M-L window
FIX_ANGLE_TOL = 0.005  # angle/dihedral degrees: below two-decimal reporting precision without float equality
_CON_PAD = 0.1  # constrain distance half-window when a scalar target is given
_CON_ANG_PAD = 5.0  # constrain angular half-window
_SHAPE_PAD = 0.05  # graft pairwise-shape half-window (a rigid hold; the exact graft does the real work)
_MIN_SHAPE_ATOMS = 3  # below this a core has only a distance to pin, not an orientable 3-D shape
DIST_ATOMS = 2  # a distance key names two atoms
ANGLE_ATOMS = 3  # an angle key names three atoms
DIHEDRAL_ATOMS = 4  # a dihedral key names four atoms


def _site_point(positions, haptic, atom):
    """Return an atom's position, or its haptic face centroid, or None when either is out of range."""
    if 0 <= atom < len(positions):
        return positions[atom]
    face = haptic.get(atom)
    if not face or any(index < 0 or index >= len(positions) for index in face):
        return None
    return np.mean(positions[list(face)], axis=0)


def constraint_value(positions, atoms, haptic=None, window=None):
    """Return the distance, angle or dihedral `atoms` measure at `positions`, or None for an atom out of range.

    A haptic centroid key reads as its face's mean position; a dihedral is re-centred on `window` when one is given.
    """
    points = [_site_point(positions, haptic or {}, atom) for atom in atoms]
    if any(value is None for value in points):
        return None
    if len(atoms) == DIST_ATOMS:
        return float(np.linalg.norm(points[0] - points[1]))
    if len(atoms) == ANGLE_ATOMS:
        return bond_angle(*points)
    value = dihedral_angle(*points)
    if window is not None:
        middle = 0.5 * sum(window)
        value = middle + (value - middle + STRAIGHT) % (2 * STRAIGHT) - STRAIGHT
    return value


def out_of_plane(positions, metal, donor, left, right):
    """Return the angle in degrees between the donor-metal bond and the left-donor-right plane."""
    normal = np.cross(positions[left] - positions[donor], positions[right] - positions[donor])
    bond = positions[metal] - positions[donor]
    scale = np.linalg.norm(normal) * np.linalg.norm(bond)
    if scale == 0.0:
        return float("nan")
    return float(np.degrees(np.arcsin(min(1.0, abs(np.dot(normal, bond)) / scale))))


def plane_torsion_cap(cap, angle):
    """Return the dihedral about one donor bond that puts the metal `cap` degrees out of plane at that M-D-X angle.

    The metal's height over the plane is sin(M-D-X) * sin(offset), so one cap maps to a different dihedral on each donor
    bond; 90 means no dihedral is left to cap.
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
    """Resolve an atom index of `mol`, or the first atom of a SMARTS with exactly one match (`match`), to an index."""
    return match(mol, ref)[0] if isinstance(ref, str) else atom_index(mol, ref, ref)


def atom_index(mol, atom, key):
    """Return `atom` of `key` as an index of `mol`, refusing a SMARTS string, a non-integer or an atom past the end."""
    if isinstance(atom, str):
        raise ValueError(f"{key!r} names {atom!r}; keys are 0-based atom indices, so match a SMARTS to an index first")
    if not is_index(atom):
        raise ValueError(f"{key!r} needs integer atom indices, got {type(atom).__name__}")
    if not 0 <= atom < mol.GetNumAtoms():
        where = "" if key is atom else f" in {key!r}"
        raise ValueError(f"atom index {atom}{where} is outside this molecule's atoms 0-{mol.GetNumAtoms() - 1}")
    return int(atom)


def substruct_matches(mol, query, **params):
    """Return the query's matches in `mol` as drawn or, when that form has none, in any resonance form.

    A graph carries one Lewis form, so `NC(=S)N` misses a thiourea anion drawn N=C([S-])N. RDKit's resonance forms are
    Kekule, where `c1ccccc1` hits nothing, so the drawn form is searched first and keeps every match it has.
    """
    params = {"uniquify": True, **params}
    if hits := mol.GetSubstructMatches(query, **params):
        return hits
    try:
        return Chem.ResonanceMolSupplier(mol).GetSubstructMatches(query, **params)
    except Chem.AtomValenceException:  # a hypervalent TS centre has no Lewis form to enumerate
        return ()


def match(mol, smarts):
    """Return the atom indices of the one SMARTS match, raising for none or several.

    `GetSubstructMatch` returns () for none and the first hit for several, so a typo or an ambiguous pattern could key a
    constraint to a plausible wrong atom.
    """
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"SMARTS {smarts!r} did not parse; check its syntax with Chem.MolFromSmarts")
    hits = substruct_matches(require_mol(mol, "rx.match"), query)
    if not hits:
        raise ValueError(f"SMARTS {smarts!r} matched nothing; check it against this molecule's atoms")
    if len(hits) > 1:
        raise ValueError(
            f"SMARTS {smarts!r} matched {len(hits)} times ({hits[:_SHOWN_HITS]}...); "
            f"narrow the pattern, or choose yourself with mol.GetSubstructMatches(...)"
        )
    return hits[0]


def _window(val, pad, key=None, angle=False, dihedral=False):
    """Return a ``(lo, hi)`` window: a pair as given, a scalar as ``(val-pad, val+pad)``, its range checked.

    A dihedral window may cross the periodic boundary, as ``(170, 190)``.
    """
    if isinstance(val, (tuple, list, np.ndarray)):
        given = [float(v) for v in val]
        if len(given) != DIST_ATOMS:
            raise ValueError(
                f"a distance, angle or dihedral takes a value or a (lo, hi) window, got {val!r}; coordinates "
                "take an int key"
            )
        lo, hi = given
    else:
        given = [float(val)]
        lo, hi = given[0] - pad, given[0] + pad
    if dihedral:
        valid = lo <= hi and hi - lo <= 2 * STRAIGHT
        label = "dihedral window no wider than 360 degrees"
    else:
        top = STRAIGHT if angle else None
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
    """Return whether `x` is an integer value; a bool is not one."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _index_tuple(mol, key):
    """Return a numeric-term key as distinct atom indices in `canonical_key` direction, else raise."""
    key = tuple(key)
    out = tuple(atom_index(mol, a, key) for a in key)
    if len(set(out)) != len(out):  # (i, i) is the (i, j) typo: it writes the bounds-matrix diagonal and no-ops
        raise ValueError(f"constraint key {key!r} names the same atom twice; geometric coordinates need distinct atoms")
    return canonical_key(out)


def _resolve_ring(mol, ring):
    """Return a π-stack ring atom-index tuple; ``None`` is not auto-detected, so pass the atoms."""
    if ring is None:
        raise ValueError("constrain plane needs the ring atom indices, e.g. {(tuple(ring_a), tuple(ring_b)): 3.7}")
    if len(ring) < 3:  # noqa: PLR2004
        raise ValueError(f"constrain plane needs at least 3 atoms per ring; got {tuple(ring)}")
    atoms = tuple(atom_index(mol, atom, tuple(ring)) for atom in ring)
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"constrain plane ring atoms must be distinct; got {atoms}")
    return atoms


def _apply_fix(mol, fix, terms, coord_fix):
    """Resolve ``fix`` into `terms` and `coord_fix`; an empty fix is refused.

    A dict mixes coordinate pins (int key, grafted) and numeric values (tuple key, restrained). A `Template`, alone or
    in a list, copies a reference's coordinates; the other items of a list hold their atoms at the source's own
    coordinates and win over the template.
    """
    if fix is None:
        return
    if isinstance(fix, Template):
        fix = [fix]
    if not isinstance(fix, (dict, list, tuple, set)):
        raise ValueError(
            f"fix= takes a list of atom indices (own-coords graft), a dict "
            f"({{i: (x,y,z)}} coords / {{(i,j): d}} numbers) or a Template; got {type(fix).__name__} {fix!r}"
            + (f"; did you mean fix=[{fix}]?" if is_index(fix) else "")
        )
    if not fix:
        raise ValueError("fix= is empty, so it holds nothing; leave fix unset, or name the atoms or values to hold")
    if isinstance(fix, dict):
        for key, val in fix.items():
            if is_index(key):  # int key -> a coordinate pin (graft)
                xyz = tuple(float(x) for x in val)
                if len(xyz) != 3:  # noqa: PLR2004 - x, y, z
                    raise ValueError(f"a coordinate fix value must be (x, y, z); got {val!r}")
                coord_fix[atom_index(mol, key, key)] = xyz
            else:
                _fix_number(mol, key, val, terms)
        return
    templates = [item for item in fix if isinstance(item, Template)]
    if len(templates) > 1:
        raise ValueError("fix= takes one Template; merge the references or embed in two steps")
    coord_fix.update(_template_coords(templates[0], mol) if templates else {})
    if atoms := [item for item in fix if not isinstance(item, Template)]:  # held at the source's own coordinates
        if not mol.GetNumConformers():
            raise ValueError(
                "fix=[atoms] holds the source's own coordinates, and a SMILES has none; give values or {i: (x, y, z)}"
            )
        positions = mol.GetConformer().GetPositions()
        for index in (atom_index(mol, atom, "fix=[...]") for atom in atoms):
            coord_fix[index] = tuple(positions[index])


def _fix_number(mol, key, val, terms):
    """Write one numbers-``fix`` term into `terms`: a tuple key with a scalar target or an explicit allowed range."""
    idx = _index_tuple(mol, key)
    if len(idx) not in (DIST_ATOMS, ANGLE_ATOMS, DIHEDRAL_ATOMS):
        raise ValueError(f"fix number key {key!r} needs 2 (distance), 3 (angle) or 4 (dihedral) atoms")
    requested = _window(val, 0.0, key, angle=len(idx) == ANGLE_ATOMS, dihedral=len(idx) == DIHEDRAL_ATOMS)
    if terms["fixed"].get(idx, requested) != requested:
        raise ValueError(f"fix= gives conflicting values for {idx}; state it once")
    terms["fixed"][idx] = requested


def _apply_constrain(mol, constrain, terms):
    """Resolve ``constrain`` into `terms`; return its releasable distance and angular keys."""
    d_soft: set = set()  # releasable soft distance keys: provenance for mc(explore=)
    angular_soft: set = set()  # releasable soft angle and dihedral keys
    if not constrain:
        return d_soft, angular_soft
    for key, val in constrain.items():
        if (  # a pi-stack key: two ring atom groups (or None), not two atom indices
            isinstance(key, tuple)
            and len(key) == DIST_ATOMS
            and all(isinstance(e, (list, tuple)) or e is None for e in key)
        ):
            ra, rb = key
            ra, rb, separation = _resolve_ring(mol, ra), _resolve_ring(mol, rb), float(val)
            if set(ra) & set(rb):
                raise ValueError("constrain plane rings must not share atoms")
            if not np.isfinite(separation) or separation <= 0.0:
                raise ValueError(f"constrain plane separation must be a positive finite distance; got {val!r}")
            terms["planes"].append((ra, rb, separation))
            continue
        idx = _index_tuple(mol, key)
        if len(idx) == DIST_ATOMS:
            add_distance(terms["distances"], *idx, *_window(val, _CON_PAD, key))
            d_soft.add(idx)
        elif len(idx) == ANGLE_ATOMS:
            terms["angles"][idx] = _window(val, _CON_ANG_PAD, key, angle=True)
            angular_soft.add(idx)
        elif len(idx) == DIHEDRAL_ATOMS:
            terms["dihedrals"][idx] = _window(val, _CON_ANG_PAD, key, dihedral=True)
            angular_soft.add(idx)
        else:
            raise ValueError(f"constrain key {key!r} needs 2 (distance), 3 (angle) or 4 (dihedral) atoms")
    return d_soft, angular_soft


def _reference_positions(reference):
    """Return a template reference's ``(N, 3)`` coordinates, from a Mol, an ``.xyz`` path or an array.

    An ``.xyz`` is read for coordinates only, never perceived, so a TS with a hypervalent carbon can serve as reference.
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


class Template(NamedTuple):
    """Name a reference whose coordinates ``fix=`` copies onto the matching atoms of a target.

    `mapping` is ``{target index: reference index}``, or a SMARTS with one ordered match on both graphs (a symmetric
    core needs the explicit map). `reference` is a Mol with a conformer, an (N, 3) array, an .xyz path or an `Ensemble`.
    """

    reference: object
    mapping: dict | str


def _template_coords(template, target):
    """Return ``{target atom: xyz}``: the coordinates `template` copies onto `target`.

    Coordinates carry handedness, so negate one axis of the reference to embed the other diastereomer on purpose.
    """
    reference, mapping = template
    if hasattr(reference, "_restored_mol"):  # an Ensemble; this module sits below the run that defines it
        if not reference.ids:
            raise ValueError("a template reference Ensemble needs a tracked conformer; this one has none")
        reference = reference._restored_mol(reference.ids[0])
    if isinstance(mapping, str):
        mapping = _smarts_mapping(target, reference, mapping)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "a Template maps atoms by SMARTS or a {target_index: reference_index} dict; an .xyz or array needs the dict"
        )
    positions = _reference_positions(reference)
    coords = {}
    for ti, ri in mapping.items():
        if not (0 <= int(ri) < len(positions)):
            raise ValueError(
                f"template map {ti!r}->{ri!r}: reference index {ri} out of range "
                f"(the reference has {len(positions)} atoms, 0..{len(positions) - 1})"
            )
        coords[atom_index(target, ti, "Template map")] = tuple(positions[int(ri)])
    return coords


def _smarts_mapping(target, reference, smarts):
    """Return ``{target index: reference index}`` from the one ordered match of `smarts` on each graph."""
    if target is None or not isinstance(reference, Chem.Mol):
        raise ValueError(
            "Template(reference, SMARTS) needs two molecular graphs; map an .xyz or array by index instead"
        )
    target_match, reference_match = match(target, smarts), match(reference, smarts)
    query = Chem.MolFromSmarts(smarts)
    if (
        len(substruct_matches(target, query, uniquify=False, maxMatches=2)) > 1
        or len(substruct_matches(reference, query, uniquify=False, maxMatches=2)) > 1
    ):
        raise ValueError(
            f"template SMARTS {smarts!r} has symmetry-equivalent orderings; "
            "give an explicit {target_index: reference_index} map"
        )
    return dict(zip(target_match, reference_match, strict=True))


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Resolved:
    """Hold a resolved request: its `Constraints`, and the coordinates `fix` grafts (atom to xyz, empty for none)."""

    cons: Constraints
    graft: dict


def resolve_core(mol, *, fix=None, constrain=None):
    """Resolve ``fix`` and ``constrain`` on `mol` into a `Resolved`; a ``fix`` list reads `mol`'s own conformer.

    A key named by both raises, and a window inside the grafted core is dropped with a warning.
    """
    terms: dict[str, Any] = {"distances": {}, "angles": {}, "dihedrals": {}, "fixed": {}, "planes": [], "frozen": set()}
    coord_fix: dict[int, tuple] = {}  # atoms grafted to exact coords (own / explicit / from a reference)

    _apply_fix(mol, fix, terms, coord_fix)
    fixed = set(terms["fixed"])
    # the grafted core is rigid: its pairwise shape biases the seed and its atoms are frozen
    add_pairwise_shape(terms["distances"], list(coord_fix), coord_fix, _SHAPE_PAD)
    terms["frozen"] |= set(coord_fix)
    if 0 < len(coord_fix) < _MIN_SHAPE_ATOMS:
        logger.warning("fix: %d grafted atom(s) hold no orientation; graft at least 3 to hold one", len(coord_fix))
    d_soft, angular_soft = _apply_constrain(mol, constrain, terms)
    clash = sorted(fixed & (d_soft | angular_soft))
    if clash:  # `constrain` is applied last, so it would overwrite the fix and make it releasable
        raise ValueError(
            f"fix= and constrain= both name {clash}; keep a rigid hold in fix= or a movable bias in constrain="
        )
    dropped = _drop_determined_by_graft(terms, coord_fix, fixed | d_soft | angular_soft)
    fixed, d_soft, angular_soft = fixed - dropped, d_soft - dropped, angular_soft - dropped
    for key in dropped:
        terms["fixed"].pop(key, None)

    cons = Constraints(**terms, contacts=SoftKeys(distances=d_soft, angular=angular_soft))
    _warn_underdetermined(mol, cons, coord_fix, len(fixed & set(cons.distances)))
    _echo(mol, cons, coord_fix)
    return Resolved(cons=cons, graft=FrozenDict(coord_fix))


def _drop_determined_by_graft(terms, coord_fix, keys):
    """Drop each user window whose atoms all sit in the grafted core; return the keys dropped.

    The graft restores those atoms exactly, so such a window could never be reached. A distance falls back to the
    graft's shape window, and an angle or dihedral is dropped, each with a warning.
    """
    dropped = {key for key in keys if graft_owns(key, coord_fix)}
    for key in sorted(dropped):
        if len(key) == DIST_ATOMS:
            add_pairwise_shape(terms["distances"], key, coord_fix, _SHAPE_PAD)  # back to what the graft implies
        elif len(key) == ANGLE_ATOMS:
            terms["angles"].pop(key, None)
        else:
            terms["dihedrals"].pop(key, None)
        logger.warning(
            "constraint %s is inside the fixed core, where the graft wins; dropped",
            key,
        )
    return dropped


def _warn_underdetermined(mol, cons, coord_fix, n_fix_d):
    """Warn when the only two fix distances share an atom and leave their angle free.

    An SN2 pinned as ``fix={(f, c): d1, (c, cl): d2}`` without the ``(f, c, cl)`` angle comes out bent, not linear.
    """
    keys = list(cons.distances)
    if coord_fix or cons.angles or cons.dihedrals or not n_fix_d == len(keys) == 2:  # noqa: PLR2004
        return
    shared = set(keys[0]) & set(keys[1])
    if len(shared) == 1:
        (centre,) = shared
        angle = (*(set(keys[0]) - shared), centre, *(set(keys[1]) - shared))
        logger.warning(
            "fix holds two distances at %s and no angle is fixed, so %s is free; add fix={%s: degrees}",
            atom_label(mol, centre),
            "-".join(atom_label(mol, i) for i in angle),
            angle,
        )


def _echo(mol, cons, coord_fix):
    """Log at INFO what was pinned, by element and index, so a 1-based index or a picked hydrogen is obvious."""
    if not cons.is_constrained:
        return
    d_soft, angular_soft = cons.contacts.distances, cons.contacts.angular

    parts = []
    if coord_fix:
        parts.append(f"graft {', '.join(atom_label(mol, i) for i in sorted(coord_fix))}")
    for atoms, (lo, hi) in cons.fixed.items():
        tol, unit = (FIX_DISTANCE_TOL, "A") if len(atoms) == DIST_ATOMS else (FIX_ANGLE_TOL, "deg")
        requested = f"{lo:.6f}+/-{tol:g}" if lo == hi else f"[{lo:.6f},{hi:.6f}]"
        name = {2: "d", 3: "angle", 4: "dihedral"}[len(atoms)]
        parts.append(f"fix {name}({','.join(atom_label(mol, i) for i in atoms)})->{requested}{unit}")
    for i, j in d_soft:
        lo, hi = cons.distances[(i, j)]
        parts.append(f"soft d({atom_label(mol, i)},{atom_label(mol, j)})->{lo:.2f}-{hi:.2f}A")
    for name, windows in (("angle", cons.angles), ("dihedral", cons.dihedrals)):
        for atoms in angular_soft & windows.keys():
            lo, hi = windows[atoms]
            parts.append(f"soft {name}({','.join(atom_label(mol, i) for i in atoms)})->{lo:.1f}-{hi:.1f}deg")
    for ra, rb, sep in cons.planes:
        parts.append(f"stack {len(ra)}x{len(rb)} ring @ {sep:.1f}A")
    for part in parts:
        logger.info("resolve: %s", part)
