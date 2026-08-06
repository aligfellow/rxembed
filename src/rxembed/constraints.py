"""The `Constraints` struct every builder fills and every stage reads, plus the ``fix``/``constrain`` resolver.

Two verbs, one resolver (`resolve_core`).

``fix`` is rigid: the named atoms will have this geometry.

- a list of atom indices holds them at the source's own coords (Kabsch graft, needs a geometry).
- ``{i: (x, y, z)}`` holds them at explicit coords (graft).
- ``{(i, j): d, (i, j, k): θ}`` states numbers as tight windows, UFF-pulled toward the value. A pull, not
  a snap: a clean target lands within ~0.05 Å, so read it back with ``.measure()``. A dict may mix both.

``constrain`` is soft: a wider window a real energy may overrule, and the home of π-stacks (a plane key
``(ring_a, ring_b): separation``). Held? ``fix``. A bias the search can move off? ``constrain``.

A template is not a third verb: ``template=(reference, map_or_smarts)`` is dissolved into a coords-``fix``
before this resolver sees it. SMARTS is the short form when both sides have molecular graphs; an xyz or
coordinate array needs an explicit index map.

Keys are 0-based atom indices in xyz/graph order. Strictness is the fix/constrain axis, not a second force
constant, since
``AddDistanceConstraint`` is flat-bottomed: one shared ``distance_fc`` pulls a tight window to its target
and lets a wide one float.
"""

from __future__ import annotations

import itertools
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np
from rdkit import Chem


class SphereRecipe(NamedTuple):
    """One metal centre's polytope as it was seated: which donor sits at which vertex, and of which shape.

    The windows themselves are already in `distances` / `angles`; this is the seating behind them, which a
    pairwise window cannot express. `Umbrella` is its one reader, needing the three base vertices in vertex
    order to state a pyramid's improper.
    """

    metal: int
    donors: tuple
    geometry: str
    order: tuple
    real_z: int
    haptic: tuple  # ((centroid-dummy, ring atoms), ...), sorted so the recipe stays hashable and comparable


@dataclass
class Constraints:
    """The one struct every builder fills and every stage reads: distances, angles, planes, frozen set."""

    distances: dict = field(default_factory=dict)  # (i, j) -> (lo, hi) Angstrom
    angles: dict = field(default_factory=dict)  # (i, j, k) -> (lo, hi) degrees
    planes: list = field(default_factory=list)  # (ring_a, ring_b, separation) parallel stack
    coplanar: list = field(default_factory=list)  # (i, j, k, l, anchor, cap): hold i-j-k-l within `cap` deg of
    #   its in-plane `anchor` (0 syn / 180 anti). A window, not a point, so real out-of-plane scatter survives.
    frozen: set = field(default_factory=set)  # pin to embedded coords
    contacts: tuple = field(default_factory=lambda: (frozenset(), frozenset()))  # the (distance, angle) keys
    #   from seeded NCI / user contacts rather than a structural hold. `relaxed()` releases exactly these.

    # --- Coordination fields. All empty for an organic system, so the shared relax is bit-identical there
    # rather than branching. They exist because the metal is embedded as a BOND-LESS surrogate: stripping the
    # M-donor bonds also strips every UFF term that came with them, and these put the missing ones back.
    metals: set = field(default_factory=set)  # metal indices, re-typed to a zero-vdW element on the FF copy;
    #   otherwise UFF reads each M-donor pair as non-bonded and applies a ~3.85 A LJ against a real 2.0-2.4 A bond
    pulls: dict = field(default_factory=dict)  # (i, j) -> target Angstrom: where in the flat-bottomed distance
    #   window to sit. M-donor pairs only, since in a TS the metal may be in the user's own reacting core
    floors: dict = field(default_factory=dict)  # (i, j) -> minimum Angstrom: the FF anti-overbond wall
    dg_floors: dict = field(default_factory=dict)  # (i, j) -> the same distance, lowering a bounds-matrix cell:
    #   the bond-less carbon floors every M...X at a carbon vdW contact, forbidding the real geometry. See `compose`.
    shapes: list = field(default_factory=list)  # atom sets held as an all-pairs rigid body (a spectator sphere).
    #   Distinguishes a modelled window, which needs a `pull`, from a rigid-body member, which must not get one.
    phantoms: frozenset = field(default_factory=frozenset)  # zero-volume dummies: an eta>=3 face's centroid,
    #   standing in as the one vertex a Cp/arene presents. Transient: materialised inside the embed and the
    #   relax, never in a stored Mol, so nothing downstream (gate, metrics, dump, calculator) sees one.
    spheres: tuple = field(default_factory=tuple)  # one `SphereRecipe` per metal centre: which donor sits at
    #   which vertex of which polytope, which no pairwise window records. Read by `Umbrella`, never by a writer.
    haptic: dict = field(default_factory=dict)  # {centroid dummy -> its ring atoms}; the source of truth for
    #   eta>=3 faces. The stored mol and donor list stay real: the ring atoms are the donors.

    @property
    def is_constrained(self) -> bool:
        """True when any distance / angle / plane / frozen constraint is set."""
        return bool(self.distances or self.angles or self.planes or self.frozen)

    def copy(self, **overrides) -> "Constraints":
        """Return a field-complete deep copy, with any keyword replacing that field outright."""
        return replace(deepcopy(self), **overrides)

    def relaxed(self) -> "Constraints":
        """Return a copy with the seeded NCI/user contacts released, for the exploratory search.

        Released atoms are no longer pose-frozen, so a strained contact may break or a new one form. Every
        structural hold (frozen core, metal sphere, coplanarity cap, pi planes, centroid dummies) is carried
        through by `copy`. The caller adds encounter bounds if releasing leaves a fragment unconstrained.
        """
        dk, ak = self.contacts
        return self.copy(
            distances={k: v for k, v in self.distances.items() if k not in dk},
            angles={k: v for k, v in self.angles.items() if k not in ak},
            contacts=(frozenset(), frozenset()),
        )

    def constrained_atoms(self) -> set:
        """Return the atoms MC pose-mode must hold so NCI / TS / metal contacts stay intact."""
        s = set(self.frozen)
        for i, j in self.distances:
            s |= {i, j}
        for t in self.angles:
            s |= set(t)
        for ring_a, ring_b, _ in self.planes:
            s |= set(ring_a) | set(ring_b)
        return s - set(self.phantoms)  # a haptic centroid dummy is transient embed scaffolding, never a real atom


def _merge_last_wins(a, b):
    return {**a, **b}


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


def _merge_exclusive(name):
    """Build a dict merge that refuses a key collision, for fields where two claims cannot be reconciled."""

    def merge(a, b):
        clash = {k for k in b if k in a and a[k] != b[k]}
        if clash:
            raise ValueError(f"compose: conflicting {name} on {sorted(clash)}; two sources claim one key")
        return {**a, **b}

    return merge


_MERGE = {  # field -> how two sources combine. See `compose`.
    "distances": _merge_last_wins,  # a spec landing ON a structural hold is a user override of it (dispatch)
    "angles": _merge_last_wins,
    "planes": lambda a, b: [*a, *b],  # never de-duplicated: two identical pi-stacks are the caller's business
    "coplanar": lambda a, b: [*a, *b],
    "frozen": lambda a, b: a | b,
    "contacts": lambda a, b: (a[0] | b[0], a[1] | b[1]),
    "metals": lambda a, b: a | b,
    "pulls": _merge_exclusive("pulls"),  # two harmonic targets on one pair is unresolvable, not a last-wins
    "floors": _merge_floor,
    "dg_floors": _merge_relief,
    "shapes": lambda a, b: [*a, *(set(s) for s in b)],
    "phantoms": lambda a, b: a | b,
    "spheres": lambda a, b: (*a, *b),  # concat: a bimetallic complex carries one recipe per centre
    "haptic": _merge_exclusive("haptic"),  # a shared key = two faces claiming one reserved index = corruption
}

_names = {f.name for f in fields(Constraints)}  # a new field must be given a merge policy, not defaulted
if set(_MERGE) != _names:
    raise RuntimeError(f"_MERGE is out of sync with Constraints: {_names ^ set(_MERGE)}")
del _names


def compose(*parts: Constraints) -> Constraints:
    """Merge constraint sets left to right into a new one, per-field, without mutating any input.

    Field-driven via ``_MERGE``, so a new field cannot be silently dropped at a merge site. Later parts win on
    `distances`/`angles`; `pulls`/`haptic` raise on a conflicting key rather than guess.

    `floors` takes the max and `dg_floors` the min, because they are opposite mechanisms sharing one number.
    A floor is a wall the force field raises, so stricter is safer; a dg_floor is a relief that only ever
    lowers a bounds-matrix cell, where the max would keep more of the phantom it exists to cut. Both stay
    order-independent.
    """
    out = Constraints()
    for part in parts:
        for name, merge in _MERGE.items():
            setattr(out, name, merge(getattr(out, name), getattr(part, name)))
    return out


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
_FIX_ANG_PAD = 2.0  # numbers-fix angle half-window (degrees)
_CON_PAD = 0.1  # constrain distance half-window when a scalar target is given
_CON_ANG_PAD = 5.0  # constrain angle half-window
_SHAPE_PAD = 0.05  # graft pairwise-shape half-window (a rigid hold; the exact graft does the real work)
_MIN_SHAPE_ATOMS = 3  # below this a core has only a distance to pin, not an orientable 3-D shape
_COORD_LEN = 3  # an (x, y, z) coordinate
_DIST_ATOMS = 2  # a distance key names two atoms
_ANGLE_ATOMS = 3  # an angle key names three atoms
_STRAIGHT = 180.0  # degrees: a bond angle cannot exceed it
_SHOWN_HITS = 4  # how many ambiguous matches to name before trailing off


def resolve_atom(mol, ref):
    """Resolve an atom reference (an int index, or a SMARTS matching one atom) to an index; a query helper."""
    if isinstance(ref, (int, np.integer)):
        return int(ref)
    hit = mol.GetSubstructMatch(Chem.MolFromSmarts(ref))
    if not hit:
        raise ValueError(f"SMARTS {ref!r} matched nothing")
    return hit[0]


def match(mol, smarts):
    """Return the atom indices of the one SMARTS match, raising if there is not exactly one.

    Both raises are the point, and the second matters more. RDKit's `GetSubstructMatch` returns () for no
    match, so a mistyped pattern flows on as an empty index set and the caller embeds with no constraint.
    It also returns whichever match it found first when there are several, which is how a core silently
    flips: on 2-chlorobenzyl chloride, `[Cl]` hands back the unreactive aryl chloride. A constraint keyed on
    the wrong atom is a wrong answer that looks right, so an ambiguous pattern is the caller's to resolve.
    """
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"SMARTS {smarts!r} did not parse")
    hits = mol.GetSubstructMatches(query)
    if not hits:
        raise ValueError(f"SMARTS {smarts!r} matched nothing")
    if len(hits) > 1:
        raise ValueError(
            f"SMARTS {smarts!r} matched {len(hits)} times ({hits[:_SHOWN_HITS]}...); "
            f"narrow the pattern, or choose yourself with mol.GetSubstructMatches(...)"
        )
    return hits[0]


def _window(val, pad, key=None, angle=False):
    """Return a ``(lo, hi)`` window: a 2-sequence verbatim, a scalar as ``(val-pad, val+pad)``.

    The range is checked here, where the offending key is still in hand. An out-of-range angle, a negative
    distance or an inverted window otherwise survives the embed and dies inside RDKit's C++ AngleConstraint
    at minimize time, with a precondition message naming no atom. The padded window is clamped to 0-180°, so
    a legitimate ``fix={(i,j,k): 180}`` still resolves.
    """
    if isinstance(val, (tuple, list, np.ndarray)):
        given = [float(v) for v in val]
        if len(given) != _DIST_ATOMS:
            raise ValueError(
                f"a distance/angle value must be a scalar target or a (lo, hi) window; got {val!r} "
                f"(a coordinate belongs under an int key: fix={{i: (x, y, z)}})"
            )
        lo, hi = given
    else:
        given = [float(val)]
        lo, hi = given[0] - pad, given[0] + pad
    top = _STRAIGHT if angle else None
    if lo > hi or any(v < 0.0 or (top is not None and v > top) for v in given):
        raise ValueError(
            f"constraint {key!r}: {val!r} is not a valid "
            + ("angle in 0-180 degrees" if angle else "distance in Angstrom")
            + "; give a scalar target or an ordered (lo, hi) window"
        )
    return (max(0.0, lo), hi if top is None else min(top, hi))


def _is_index(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _index_tuple(key):
    """Return a pair/triple key as a tuple of ``int`` atom indices; error clearly if a SMARTS/str slipped in."""
    key = tuple(key)
    for a in key:
        if isinstance(a, str):
            raise ValueError(
                f"constraint key {key!r} contains the string {a!r}: fix=/constrain= are index-driven "
                f"(0-based, xyz/graph order); resolve your SMARTS first, e.g. "
                f"i = mol.GetSubstructMatch(Chem.MolFromSmarts('[F-]'))[0], then key by i"
            )
        if not _is_index(a):
            raise ValueError(f"constraint key {key!r} must be integer atom indices, got {type(a).__name__}")
    out = tuple(int(a) for a in key)
    if len(set(out)) != len(out):  # (i, i) is the (i, j) typo: it writes the bounds-matrix diagonal and no-ops
        raise ValueError(f"constraint key {key!r} names the same atom twice; a distance/angle needs two")
    return out


def _is_plane_key(key):
    """Return True for a π-stack key: a 2-tuple of ring atom-groups (lists/tuples/None), not two indices."""
    if not (isinstance(key, tuple) and len(key) == _DIST_ATOMS):
        return False
    return all(isinstance(e, (list, tuple)) or e is None for e in key)


def _as_coord(val):
    v = tuple(float(x) for x in val)
    if len(v) != _COORD_LEN:
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
    return tuple(int(a) for a in ring)


def _validate_indices(mol, atoms):
    n = mol.GetNumAtoms()
    for a in atoms:
        if not (0 <= a < n):
            raise ValueError(f"atom index {a} out of range (molecule has {n} atoms, 0..{n - 1})")


def _apply_fix(mol, fix, cons, coord_fix, has_geometry):
    """Resolve ``fix`` into `cons`/`coord_fix`; return the numbers-fix keys it wrote (normalised).

    A dict mixes coordinate pins (int key, grafted) and tight distance/angle windows (tuple key, UFF-pulled);
    a list, tuple or set holds the named atoms at the source's own coords, which needs a geometry. The
    returned keys are what `resolve_core` checks a later `constrain` against.
    """
    written: set = set()
    if fix is None:
        return written
    if not isinstance(fix, (dict, list, tuple, set)):
        raise ValueError(
            f"fix= takes a list of atom indices (own-coords graft) or a dict "
            f"({{i: (x,y,z)}} coords / {{(i,j): d}} numbers); got {type(fix).__name__} {fix!r}"
            + (f"; did you mean fix=[{fix}]?" if _is_index(fix) else "")
        )
    if isinstance(fix, dict):
        for key, val in fix.items():
            if _is_index(key):  # int key -> a coordinate pin (graft)
                coord_fix[int(key)] = _as_coord(val)
            else:  # tuple key -> a number: tight window, UFF-pulled to the target
                idx = _index_tuple(key)
                if len(idx) == _DIST_ATOMS:
                    add_distance(cons.distances, *idx, *_window(val, _FIX_PAD, key))
                    written.add((min(idx), max(idx)))
                elif len(idx) == _ANGLE_ATOMS:
                    cons.angles[idx] = _window(val, _FIX_ANG_PAD, key, angle=True)
                    written.add(idx)
                else:
                    raise ValueError(f"fix number key {key!r} needs 2 atoms (distance) or 3 (angle)")
        return written
    # a list/tuple/set of atom indices -> hold at the source's own coords (graft)
    if not has_geometry:
        raise ValueError(
            "fix=[atoms] holds them at the source's own coordinates, but this source has no "
            "geometry (a SMILES). Give numbers (fix={(i, j): d, (i, j, k): θ}) or explicit "
            "coordinates from a reference (fix={i: (x, y, z)})."
        )
    positions = mol.GetConformer().GetPositions()
    for a in fix:
        if isinstance(a, str) or not _is_index(a):
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
    """Resolve ``constrain`` (soft windows + π-stack planes) into `cons`; return the releasable ``(d, a)`` keys."""
    d_soft: set = set()  # releasable soft distance keys: provenance for mc(explore=)
    a_soft: set = set()  # releasable soft angle keys
    if not constrain:
        return d_soft, a_soft
    for key, val in constrain.items():
        if _is_plane_key(key):
            ra, rb = key
            cons.planes.append((_resolve_ring(mol, ra), _resolve_ring(mol, rb), float(val)))
            continue
        idx = _index_tuple(key)
        if len(idx) == _DIST_ATOMS:
            add_distance(cons.distances, *idx, *_window(val, _CON_PAD, key))
            d_soft.add((min(idx), max(idx)))
        elif len(idx) == _ANGLE_ATOMS:
            cons.angles[idx] = _window(val, _CON_ANG_PAD, key, angle=True)
            a_soft.add(idx)
        else:
            raise ValueError(f"constrain key {key!r} needs 2 atoms (distance) or 3 (angle)")
    return d_soft, a_soft


_XYZ_DIM = 3  # an (x, y, z) row
_TEMPLATE_LEN = 2  # template=(reference, mapping)


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
            raise ValueError(f"could not read coordinates from {path!r}")
        return mol.GetConformer().GetPositions()
    arr = np.asarray(reference, float)
    if arr.ndim == _TEMPLATE_LEN and arr.shape[1] == _XYZ_DIM:
        return arr
    raise ValueError("a template reference must be a Mol with a conformer, an .xyz path, or an (N, 3) array")


def template_to_fix(template, fix=None, own=None, target=None):
    """Fold ``template=(reference, map_or_smarts)`` into a coordinate ``fix``; sugar, not a mechanism.

    A template graft is a `fix` with coordinates read off a reference, so the resolver knows only
    `fix`/`constrain` and this is where the sugar dissolves. Every rigid spec is one question, where do these
    atoms' coordinates come from: your own geometry (`fix=[i, j]`, resolved against `own`), coordinates you
    supply (`fix={i: (x, y, z)}`), or another molecule (`template=`). They compose, and an explicit `fix` on
    the same atom wins.

    Coordinates carry handedness, which a distance/angle spec cannot: that spec is reflection-invariant, so
    it may give the mirror image silently. To embed the other diastereomer deliberately, negate one axis of
    the reference positions. A SMARTS must also have one ordered correspondence on each graph; a symmetric
    query needs an explicit map rather than an atom-order-dependent automorphism.
    """
    if not (
        isinstance(template, (tuple, list)) and len(template) == _TEMPLATE_LEN and isinstance(template[1], (dict, str))
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
        raise ValueError(
            "template= was given an empty atom map, which would graft nothing and embed as if no template "
            "were passed. If a SMARTS built the map, it matched nothing on one side; rxembed.match raises "
            "where GetSubstructMatch returns () silently."
        )
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
    d_soft, a_soft = _apply_constrain(mol, constrain, cons)
    clash = sorted(fixed & (d_soft | a_soft))
    if clash:  # `constrain` is applied last, so it would overwrite the fix and make it releasable
        raise ValueError(
            f"fix= and constrain= both name {clash}: one key, two contradictory intents. Keep the rigid "
            f"one in fix= (or move it to constrain= if the search may move off it)."
        )
    dropped = _drop_determined_by_graft(cons, coord_fix, fixed | d_soft | a_soft)
    fixed, d_soft, a_soft = fixed - dropped, d_soft - dropped, a_soft - dropped

    _validate_indices(
        mol,
        {a for k in cons.distances for a in k}
        | {a for k in cons.angles for a in k}
        | {a for ra, rb, _ in cons.planes for a in (*ra, *rb)},
    )
    cons.contacts = (frozenset(d_soft), frozenset(a_soft))  # only constrain= is releasable; fix is structural
    _warn_underdetermined(cons, coord_fix, len(fixed & set(cons.distances)))
    _echo(mol, cons, coord_fix)
    return cons, coord_fix


def _drop_determined_by_graft(cons, coord_fix, keys):
    """Drop each user window whose atoms all sit in the grafted core; return the keys dropped.

    The Kabsch graft restores those atoms to their exact coordinates after the embed, so such a window can
    never be realised. Worse, it overwrote the graft's own pairwise-shape bound, seeding the whole embed
    against a distance the graft then contradicts. Restore the shape window (a distance) or drop it (an
    angle), loudly: silently ignoring it would leave the user thinking the constraint applied.
    """
    dropped = {key for key in keys if all(a in coord_fix for a in key)}
    for key in sorted(dropped):
        if len(key) == _DIST_ATOMS:
            add_pairwise_shape(cons, key, coord_fix, _SHAPE_PAD)  # back to what the graft implies
        else:
            cons.angles.pop(key, None)
        logger.warning(
            "constraint %s is inside the fixed core, where the graft wins; dropped",
            key,
        )
    return dropped


def _warn_underdetermined(cons, coord_fix, n_fix_d):
    """Advisory (invariant 5): two shared-atom fix distances over exactly 3 atoms leave the angle free.

    The classic mis-fix is an SN2 pinned as ``fix={(f, c): d1, (c, cl): d2}`` with no ``(f, c, cl)`` angle,
    which comes out bent rather than the linear TS the user meant. A richer network (more than 3 atoms, or
    any coords or angles) is left alone as a deliberate distance web. Advisory, not fatal.
    """
    if coord_fix or cons.angles or n_fix_d < _DIST_ATOMS:  # coords/angles orient; <2 distances make no angle
        return
    keys = list(cons.distances)
    atoms = {a for k in keys for a in k}
    shares = any(len(set(a) & set(b)) == 1 for i, a in enumerate(keys) for b in keys[i + 1 :])
    if len(atoms) <= _MIN_SHAPE_ATOMS and shares:
        logger.warning(
            "fix: two distances share an atom over %d atom(s) and no angle is fixed, so the angle is free",
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
    d_soft, a_soft = cons.contacts

    def s(i):
        return f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}"

    parts = []
    if coord_fix:
        parts.append(f"graft {', '.join(s(i) for i in sorted(coord_fix))}")
    for (i, j), (lo, hi) in cons.distances.items():
        if (i, j) in d_soft or (i in cons.frozen and j in cons.frozen):
            continue  # soft (below) or the frozen-core shape (structural, implied by the graft)
        parts.append(f"fix d({s(i)},{s(j)})->{lo:.2f}-{hi:.2f}A")
    for (i, j, k), (lo, hi) in cons.angles.items():
        if (i, j, k) in a_soft:
            continue
        parts.append(f"fix angle({s(i)},{s(j)},{s(k)})->{lo:.1f}-{hi:.1f}deg")
    for i, j in d_soft:
        lo, hi = cons.distances[(i, j)]
        parts.append(f"soft d({s(i)},{s(j)})->{lo:.2f}-{hi:.2f}A")
    for i, j, k in a_soft:
        lo, hi = cons.angles[(i, j, k)]
        parts.append(f"soft angle({s(i)},{s(j)},{s(k)})->{lo:.1f}-{hi:.1f}deg")
    for ra, rb, sep in cons.planes:
        parts.append(f"stack {len(ra)}x{len(rb)} ring @ {sep:.1f}A")
    for part in parts:
        logger.info("resolve: %s", part)
