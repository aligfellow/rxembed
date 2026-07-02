"""The Constraints struct every builder fills and every stage reads."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Constraints:
    """The one struct every builder fills and every stage reads: distances, angles, planes, frozen set."""

    distances: dict = field(default_factory=dict)  # (i, j) -> (lo, hi) Angstrom
    angles: dict = field(default_factory=dict)  # (i, j, k) -> (lo, hi) degrees
    planes: list = field(default_factory=list)  # (ring_a, ring_b, separation) parallel stack
    frozen: set = field(default_factory=set)  # pin to embedded coords
    contacts: tuple = field(default_factory=lambda: (frozenset(), frozenset()))
    #   (distance-keys, angle-keys) that came from SEEDED NCI / user contacts — NOT the structural holds (a
    #   frozen-core shape, a metal sphere, fragment encounter bounds). `relaxed()` releases exactly these so
    #   the exploratory MC pass can let a strained grip break (mc(explore=True)).

    @property
    def is_constrained(self) -> bool:
        """True when any distance / angle / plane / frozen constraint is set."""
        return bool(self.distances or self.angles or self.planes or self.frozen)

    def relaxed(self) -> "Constraints":
        """Return a copy with the seeded NCI/user contacts released, for the exploratory search.

        The released contacts' atoms are no longer pose-frozen, so a strained contact may break or a new
        one (e.g. pi) form — while the STRUCTURAL holds (a frozen TS core, a metal sphere, pi planes) are
        kept so the system can't dissociate. The caller adds fragment encounter bounds if releasing leaves
        it unconstrained. Provenance is cleared (nothing left to release).
        """
        dk, ak = self.contacts
        return Constraints(
            distances={k: v for k, v in self.distances.items() if k not in dk},
            angles={k: v for k, v in self.angles.items() if k not in ak},
            planes=list(self.planes),
            frozen=set(self.frozen),
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
        return s


def add_distance(d: dict, i: int, j: int, lo: float, hi: float) -> None:
    """Add an order-independent distance window ``(i, j) -> (lo, hi)`` to a distance dict."""
    d[(min(i, j), max(i, j))] = (lo, hi)


def add_pairwise_shape(cons, atoms, positions, pad):
    """Pin the SHAPE of ``atoms`` by all their pairwise input distances (a frame-independent rigid hold).

    Each pair becomes a ``(d-pad, d+pad)`` window in ``cons.distances`` — used to keep a rigid sub-structure
    intact through a random-frame embed: a frozen TS core (`from_spec`/`from_template`) or a metal
    coordination sphere (`metal.hold_shape`). ``positions`` is indexable by atom index — an ``[N,3]``
    conformer array, or a ``{atom: (x,y,z)}`` map (for `from_template`, coords from a different molecule).
    """
    for a, b in itertools.combinations(atoms, 2):
        d = float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
        add_distance(cons.distances, a, b, d - pad, d + pad)
