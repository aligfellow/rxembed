"""The Constraints struct every builder fills and every stage reads."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, fields
from typing import NamedTuple

import numpy as np


class SphereRecipe(NamedTuple):
    """How one metal centre's coordination targets were derived — enough to re-derive them on a solved sphere."""

    metal: int
    donors: tuple
    geometry: str
    order: tuple
    real_z: int
    haptic: tuple  # ((centroid-dummy, ring atoms), ...) — sorted, so the recipe stays hashable and comparable


@dataclass
class Constraints:
    """The one struct every builder fills and every stage reads: distances, angles, planes, frozen set."""

    distances: dict = field(default_factory=dict)  # (i, j) -> (lo, hi) Angstrom
    angles: dict = field(default_factory=dict)  # (i, j, k) -> (lo, hi) degrees
    planes: list = field(default_factory=list)  # (ring_a, ring_b, separation) parallel stack
    coplanar: list = field(default_factory=list)  # (i, j, k, l, anchor, cap): a soft dihedral cap holding
    #   i-j-k-l within `cap`° of its in-plane `anchor` (0 syn / 180 anti). The surrogate strips the M-donor
    #   bond, so UFF loses the improper that keeps the metal in a conjugated donor's sp2 plane. Applied in both
    #   the bounds matrix (`mechanisms.Coplanar.dg_post`, a 1,4 distance window) and the FF (a soft
    #   UFFAddTorsionConstraint). A window, not a point, so real out-of-plane scatter survives. Empty for non-metals.
    frozen: set = field(default_factory=set)  # pin to embedded coords
    contacts: tuple = field(default_factory=lambda: (frozenset(), frozenset()))
    #   (distance-keys, angle-keys) that came from SEEDED NCI / user contacts — NOT the structural holds (a
    #   frozen-core shape, a metal sphere, fragment encounter bounds). `relaxed()` releases exactly these so
    #   the exploratory MC pass can let a strained grip break (mc(explore=True)).

    # --- metal FF-only fields, all empty for an organic system so `restrained_uff` is bit-identical there;
    # composes with the TS / NCI / template constraints instead of branching the shared relax.
    metals: set = field(default_factory=set)  # metal atom indices, re-typed to a zero-vdW element on the FF
    #   copy. The embed surrogate is a bond-less carbon, so UFF sees every M-donor pair as non-bonded and
    #   applies a C...X Lennard-Jones (min 3.85 A) against a real 2.0-2.4 A M-L bond — repulsive fiction that
    #   shoves every donor past its window.
    pulls: dict = field(default_factory=dict)  # (i, j) -> target Angstrom: a soft harmonic inside the
    #   flat-bottomed distance wall. `AddDistanceConstraint(lo, hi, fc)` exerts zero force between lo and hi,
    #   so without a target the donor just rides whichever wall it was pushed to. Keyed on the M-donor pairs
    #   only, never "pair contains a metal" — in a `rx.metal(xyz, fix=rc)` TS the metal is in the reacting core
    #   and those pairs are TS constraints.
    floors: dict = field(default_factory=dict)  # (i, j) -> minimum Angstrom: the anti-overbond guard.
    #   Zeroing the surrogate's vdW removes the only thing (incidentally) keeping a non-donor off the metal;
    #   without this floor ligands collapse inward. These fields ship together or not at all.
    dg_floors: dict = field(default_factory=dict)  # (i, j) -> physical minimum M...X, for the bounds matrix.
    #   Same distance as `floors`, but lowers a bound instead of raising one. The bond-less carbon surrogate
    #   makes RDKit floor every M...X pair at a carbon vdW contact (~3.4 Å to C, ~2.9 to H) while real
    #   second-sphere atoms sit at 2.8-3.0 Å, so the true geometry is forbidden and `_smooth` must widen the
    #   physical windows to compensate. H is included (it is the most over-floored) even though `floors` skips it.
    shapes: list = field(default_factory=list)  # atom sets pinned as an all-pairs rigid body by
    #   `metal.hold_shape` (a retained/spectator coordination sphere). Lets `distance.ff_terms` tell a modelled
    #   coordination window (which needs a `pull`) from a rigid-body member (which must not be pulled: singling
    #   out a subset of its windows makes the relax tear the rest). A frozen TS core also goes through
    #   `add_pairwise_shape` but is not recorded here — it is held by `AddFixedPoint`, not by these windows.
    phantoms: frozenset = field(default_factory=frozenset)  # massless reference-point atoms with ZERO excluded
    #   volume: an eta>=3 haptic ring's centroid dummy, which stands in as the ONE coordination vertex a Cp/arene face
    #   presents. It is TRANSIENT — materialised (`metal.materialise_phantoms`) only inside `bounds.embed` /
    #   `restrained_uff` from `haptic`, never part of a stored Mol. There the bounds matrix lowers its M...X floors to
    #   ~0.3 Å so the DG may seat it INSIDE its own ring, and the FF retypes it to an untypeable element (Xe) so UFF
    #   omits all its terms — a real carbon 0.8 Å inside the ring would give an r^-12 that blows the relax up. These
    #   indices name the dummies for those two primitives; they never appear in the Isomer/ensemble mol. Empty for a
    #   sigma/eta2 sphere. (A D-cap chirality dummy is bonded to its donor and KEEPS its terms — not a phantom.)
    spheres: tuple = field(default_factory=tuple)  # one `SphereRecipe` per metal centre — how its coordination
    #   targets were derived, so the ENGINE can re-derive them on a realisable point set when the raw ones turn out
    #   non-metric (`bounds._feasible_bounds`). No constraint WRITER reads it. Empty for an organic system.
    haptic: dict = field(default_factory=dict)  # {centroid-dummy index -> its ring atoms}. The single source of truth
    #   for the eta>=3 faces: `bounds.embed` / `restrained_uff` read it to materialise the transient centroid dummy at
    #   its reserved index (seated at the ring centroid), embed/relax the face as one rigid vertex, then discard it.
    #   The stored mol and donor list are always REAL (the ring atoms ARE the donors), so nothing downstream — the
    #   gate, metrics, prune, dump, a calculator — ever sees a phantom. Empty for a sigma/eta2 sphere.

    @property
    def is_constrained(self) -> bool:
        """True when any distance / angle / plane / frozen constraint is set."""
        return bool(self.distances or self.angles or self.planes or self.frozen)

    def copy(self, **overrides) -> "Constraints":
        """Return a field-complete copy, with any keyword replacing that field outright.

        Field-driven via ``_CLONE``, never hand-listed: a field added to the dataclass and not to the registry
        fails at import, so it can never be silently dropped by a copy site (the class of bug that left the
        exploratory and settle paths running without their metal fields).
        """
        out = {name: _CLONE[name](getattr(self, name)) for name in _CLONE}
        out.update(overrides)
        return Constraints(**out)

    def relaxed(self) -> "Constraints":
        """Return a copy with the seeded NCI/user contacts released, for the exploratory search.

        The released contacts' atoms are no longer pose-frozen, so a strained contact may break or a new
        one (e.g. pi) form — while every STRUCTURAL hold (a frozen TS core, a metal sphere with its FF
        fields, the coplanarity cap, pi planes, the eta>=3 centroid dummies) is carried by `copy`, or the
        exploratory pass relaxes under the bare carbon fiction. The caller adds fragment encounter bounds if
        releasing leaves it unconstrained. Provenance is cleared (nothing left to release).
        """
        dk, ak = self.contacts
        return self.copy(
            distances={k: v for k, v in self.distances.items() if k not in dk},
            angles={k: v for k, v in self.angles.items() if k not in ak},
            contacts=(frozenset(), frozenset()),
        )

    def constrained_atoms(self) -> set:
        """Atoms MC pose-mode must hold so NCI / TS / metal contacts stay intact."""
        s = set(self.frozen)
        for i, j in self.distances:
            s |= {i, j}
        for t in self.angles:
            s |= set(t)
        for ring_a, ring_b, _ in self.planes:
            s |= set(ring_a) | set(ring_b)
        return s - set(self.phantoms)  # a haptic centroid dummy is transient embed scaffolding, never a real atom


_CLONE = {  # field -> how to clone it. `shapes` is the only field whose ELEMENTS are mutable; the frozenset/tuple
    "distances": dict,  # fields are immutable and shared deliberately.
    "angles": dict,
    "planes": list,
    "coplanar": list,
    "frozen": set,
    "contacts": lambda v: v,
    "metals": set,
    "pulls": dict,
    "floors": dict,
    "dg_floors": dict,
    "shapes": lambda v: [set(s) for s in v],
    "phantoms": lambda v: v,
    "spheres": lambda v: v,  # a tuple of immutable recipe tuples
    "haptic": dict,
}


def _merge_last_wins(a, b):
    return {**a, **b}


def _merge_floor(a, b):  # a WALL is a physical MINIMUM: the stricter (higher) of two claims on one pair wins
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out[k], v) if k in out else v
    return out


def _merge_relief(a, b):  # a RELIEF is a permission to lower, so the FULLEST (lowest) claim wins — see `compose`
    out = dict(a)
    for k, v in b.items():
        out[k] = min(out[k], v) if k in out else v
    return out


def _merge_exclusive(name):
    """Build a dict merge that refuses a key collision — for fields where two claims cannot be reconciled."""

    def merge(a, b):
        clash = {k for k in b if k in a and a[k] != b[k]}
        if clash:
            raise ValueError(f"compose: conflicting {name} on {sorted(clash)} — two sources claim one key")
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

for _reg, _what in ((_CLONE, "_CLONE"), (_MERGE, "_MERGE")):  # a new field must be given a policy, not defaulted
    _names = {f.name for f in fields(Constraints)}
    if set(_reg) != _names:
        raise RuntimeError(f"{_what} is out of sync with Constraints: {_names ^ set(_reg)}")
del _reg, _what, _names


def compose(*parts: Constraints) -> Constraints:
    """Merge constraint sets left to right into a new one, per-field and without mutating any input.

    Field-driven via ``_MERGE`` so a new field cannot be silently lost at a merge site. Later parts override
    earlier ones on `distances`/`angles` (a spec landing on a structural hold is a deliberate override);
    `pulls`/`haptic` refuse a conflicting key rather than guess.

    `floors` and `dg_floors` merge in OPPOSITE directions, and that is not an inconsistency — they are opposite
    mechanisms sharing one number. A `floors` entry is a WALL the force field raises, so the stricter (max)
    claim is the safe one. A `dg_floors` entry is a RELIEF: `mechanisms.Floor.dg_relief` only ever LOWERS a
    bound with it, cutting RDKit's phantom bond-less-carbon floor (~3.4 Å) down to the real distance. Taking
    the max of two reliefs does not "keep the stricter physics" — it keeps more of the PHANTOM, which is not
    physics at all. Two sources differ on one pair only because one of them saw a coordination the other did
    not, and `_tier_floor` is monotone in exactly that (APEX < NEAR < OUTER: the more the atom is tied into the
    sphere, the lower its floor), so the lower claim is always the better-informed one. Hence min. Both merges
    stay order-independent, and min is bounded below by the bare covalent sum, so it can never license a
    collapse. (Measured: seating an alkoxide O on `OCCCN->[Pd](Cl)Cl` leaves the pre-relax Pd...C riding the
    phantom at 3.339 Å with Pd-O-C forced open to 130.6° under max, vs 3.227 Å / 126.1° under min.)
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
    """Pin the SHAPE of ``atoms`` by all their pairwise input distances (a frame-independent rigid hold).

    Each pair becomes a ``(d-pad, d+pad)`` window in ``cons.distances`` — used to keep a rigid sub-structure
    intact through a random-frame embed: a fixed TS core (`resolve_core`) or a metal coordination sphere
    (`metal.hold_shape`). ``positions`` is indexable by atom index — an ``[N,3]`` conformer array, or a
    ``{atom: (x,y,z)}`` map (for a coords-`fix` / `template`, coords from a different molecule).
    """
    for a, b in itertools.combinations(atoms, 2):
        d = float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
        add_distance(cons.distances, a, b, d - pad, d + pad)
