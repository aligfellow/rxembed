"""The golden fixture set, chosen so every branch of the DG writer is exercised by a named case.

Branch coverage was MEASURED by instrumenting the DG writer, not inferred — the obvious guesses are wrong.
Square-planar en-Pd never reaches the angle INTERSECT branch, for instance: a `Polyhedron.angles` states only
a minimal vertex-pair subset, so its N-Pd-N bite is not stated at all and all its angles are topologically
disconnected. The comment on each metal fixture names the branch it was picked to pin.

`R1_UNCOVERED` records the one branch no fixture reaches, so it is a stated gap rather than a silent one.
"""

from __future__ import annotations

import rxembed as rx

# `mechanisms.Angle.dg_windows`'s explicit-distance pre-emption (a stated `cons.distances` window on an
# angle's 1-3 pair, which discards the angle outright) fires ZERO times across every fixture below,
# including the frozen-TS metal case.
# It is covered by `test_angle_rules.py` on a hand-built Constraints instead of a real molecule.
R1_UNCOVERED = "mechanisms.Angle explicit-distance pre-emption"

_PI_STACK = ((0, 1, 2, 3, 4, 5), (10, 11, 12, 13, 14, 15))
_HENRY = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
_ETA2 = "CC(C)(C)[C]1#[C](C#C[Si](C)(C)C)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"


def _embed_all(isomer_set):
    """Embed EVERY isomer, not just the first — each has its own coordination windows and its own matrix.

    `rx.metal(...)` only perceives and enumerates; nothing reaches the bounds writer until an embed runs.
    """
    return [rx.embed(iso, n=2, seed=1) for iso in isomer_set]


def _organic():
    return {
        # No metal: proves the organic path is untouched by any metal mechanism, and that the
        # coplanar/centroid relievers never fire (their snapshots must be ABSENT from the record).
        "ethanol": lambda: rx.embed("CCO", n=2, seed=1),
        "acid_arene": lambda: rx.embed("OC(=O)CCCCc1ccccc1", constrain={(1, 9): (2.6, 3.0)}, n=2, seed=1),
        # The only fixture covering the plane path (`cons.planes`, setdefault + the +-0.3 A pad).
        "pi_stack": lambda: rx.embed("c1ccccc1CCCCc1ccccc1", constrain={_PI_STACK: 3.6}, n=2, seed=1),
        # A from-SMILES TS: exact distance + angle numbers on a reacting core.
        "sn2_smiles": lambda: rx.embed("[F-].CCl", fix={(0, 1): 2.0, (1, 2): 2.2, (0, 1, 2): 178.0}, n=2, seed=1),
        # A frozen-xyz TS: the Kabsch graft path. NB two different reacting cores for bimp.xyz exist in the
        # suite (test_frozen uses [10,11,12,14], test_connectivity uses [14,15,16,17]); this pins the former.
        "frozen_ts_bimp": lambda: rx.embed("examples/structures/bimp.xyz", fix=[10, 11, 12, 14], n=2, seed=1),
    }


def _metal():
    return {
        # All 5 angles WRITE OUTRIGHT (disconnected through the stripped metal) — the dominant path.
        "en_pd": lambda: _embed_all(rx.metal("Br[Pd]1(Cl)NCCN1", "square_planar")),
        # The INTERSECT branch: key (7,1,10), topological distance 3, derived (2.20,2.73) meets the
        # backbone's (2.48,3.69). The bis-chelate is what puts a real bond path across a stated angle.
        "bis_en_co": lambda: _embed_all(rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")),
        # Two equivalent carboxylate-type O donors — the case a charge-keyed contraction has twice split.
        "acac_ni": lambda: _embed_all(rx.metal("CC1=CC(C)=[O]->[Ni+2](Cl)(Cl)<-[O]1", "square_planar")),
        # THE COPLANAR FIXTURE, and the only one: henry Ni(II) carries both permutations of the cap — a
        # kappa1 carboxylate O (one heavy neighbour, a proper dihedral) and an N-aryl amidate N (two, an
        # improper). Without it the `coplanar` phase never fires and `d_coplanar` is never captured at all.
        # It also reaches the INTERSECT branch.
        "henry_ni": lambda: _embed_all(rx.metal(_HENRY, "square_planar")),
        # A side-on eta2 alkyne on the same scaffold: a haptic donor bonded to a co-donor, which must get
        # NO cap. Pins the negative side of the coplanar perception.
        "henry_eta2": lambda: _embed_all(rx.metal(_ETA2, "square_planar")),
    }


def _haptic():
    # Imported as a package member, never via a sys.path insert: `all_fixtures()` runs at COLLECTION time
    # (the parametrize decorator calls it), so putting `tests/` on sys.path would let the same test module be
    # imported under two names — once as `test_connectivity`, once as `tests.test_connectivity` — giving two
    # copies of its module-level state to suites that share fixtures across files.
    from tests import test_haptic as th

    return {
        # PHANTOMS: a transient centroid dummy, so `Haptic.dg_relief` fires and the matrix is
        # larger than the stored mol. These are the only fixtures where that snapshot exists.
        "ferrocene": lambda: _embed_all(rx.metal(th.ferrocene())),
        "cp_ticl3": lambda: _embed_all(rx.metal(th.cp_ticl3())),
        "dibenzene_cr": lambda: _embed_all(rx.metal(th.dibenzenechromium())),
    }


def all_fixtures():
    """{name: thunk}. A thunk returns an Ensemble, an EnsembleSet, or an IsomerSet — all carry `.cons`."""
    return {**_organic(), **_metal(), **_haptic()}
