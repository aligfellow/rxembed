"""Golden bit-identity net for the embed kernel's distance-geometry writer.

This is the reference every later "bit-identical" claim is measured against. It exists because the rest of
the suite is BEHAVIOURAL — it asserts that a geometry is good, not that a number is unchanged — so a silent
numerical regression in the bounds matrix passes 233 tests undetected.

Each fixture stores its phases separately (see `capture.py`), so a failure says WHICH phase moved: the
angle-derived windows, a floor relief, the commit, the coplanar 1,4 bound, or triangle smoothing. That
distinction is the whole point — an angle regression and a smoothing regression need different fixes, and a
single end-to-end matrix hash cannot tell them apart.

Regenerate deliberately, never to make a failure go away:

    RXEMBED_GOLDEN_UPDATE=1 uv run pytest tests/golden -q

A diff here is a behaviour change. If it is intended, the regenerated snapshot belongs in its own commit
with the measurement that justifies it.
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib

import numpy as np
import pytest

from tests.golden import capture as cap
from tests.golden import fixtures as fx

_STORE = pathlib.Path(__file__).resolve().parent / "snapshots"
_UPDATE = os.environ.get("RXEMBED_GOLDEN_UPDATE") == "1"

# Every quantity here is pure deterministic Python/numpy arithmetic over RDKit's bounds matrix plus
# math.cos/sqrt — no iterative solver — so it is compared with EXACT float equality. The one exception is
# the post-smooth matrix: DoTriangleSmoothing is C++ float arithmetic, bit-stable on one build but not
# guaranteed across RDKit versions or BLAS. It gets a tight tolerance and is compared separately.
_SMOOTH_RTOL = 1e-12


def _records_for(name, thunk):
    """Run a fixture and return its `_bounds` records — possibly zero, which is itself meaningful.

    `bounds.embed` builds a custom matrix only when the Constraints carry a distance, angle, plane or
    coplanar entry, so an unconstrained organic takes the plain-ETKDG path and never reaches the writer.
    Pinning a zero-call fixture is what would catch a change that starts (or stops) editing the matrix.
    """
    with cap.capture() as records:
        result = thunk()
        if not hasattr(result, "cons"):
            list(result)  # force any generator / lazy candidate set
    return records


def _signature(records):
    """The stored form: per _bounds call, the assembled Constraints, the phase deltas, and the tolerance."""
    out = []
    for rec in records:
        out.append(
            {
                "cons": rec["cons"],
                "pairs": rec.get("pairs"),
                "phases_present": [p for p in cap.PHASES if p in rec],
                "deltas": cap.deltas(rec),
                "tol": rec.get("tol"),
                "shape": list(rec["final"].shape),
            }
        )
    return out


def _final_matrices(records):
    return [r["final"] for r in records]


@pytest.mark.parametrize("name", sorted(fx.all_fixtures()))
def test_bounds_are_bit_identical(name):
    thunk = fx.all_fixtures()[name]
    records = _records_for(name, thunk)
    # Round-trip the live signature through JSON before comparing: the stored side has necessarily lost the
    # tuple/list distinction, so comparing raw Python would report a difference on every run.
    sig = json.loads(json.dumps(_signature(records), sort_keys=True))
    sig_path, mat_path = _STORE / f"{name}.json", _STORE / f"{name}.npz"

    if _UPDATE:
        _STORE.mkdir(exist_ok=True)
        sig_path.write_text(json.dumps(sig, indent=1, sort_keys=True))
        np.savez_compressed(mat_path, **{str(i): m for i, m in enumerate(_final_matrices(records))})
        pytest.skip(f"regenerated golden snapshot for {name} ({len(sig)} _bounds call(s))")

    if not sig_path.exists():
        pytest.fail(f"no golden snapshot for {name!r} — run with RXEMBED_GOLDEN_UPDATE=1 to create it")

    want = json.loads(sig_path.read_text())
    assert len(sig) == len(want), f"{name}: _bounds was called {len(sig)}x, golden has {len(want)}"
    for i, (got, exp) in enumerate(zip(sig, want, strict=True)):
        assert got["cons"] == exp["cons"], f"{name}[{i}]: the assembled Constraints changed"
        assert got["phases_present"] == exp["phases_present"], f"{name}[{i}]: a DG phase stopped firing"
        assert got["shape"] == exp["shape"], f"{name}[{i}]: bounds-matrix shape changed"
        assert got["tol"] == exp["tol"], f"{name}[{i}]: triangle-smoothing tolerance rung changed"
        assert got["pairs"] == exp["pairs"], f"{name}[{i}]: the assembled WINDOW pairs changed"
        # Report in PIPELINE order, never alphabetical: the two anti-correlate ("presmooth->final" sorts before
        # "rdkit->relief"), so a defect in an early phase was being announced as a late-phase one — a no-op
        # reliever reported "presmooth->final moved", which reads as RDKit/BLAS drift and primes the reader to
        # loosen the tolerance instead of finding the real cause. The FIRST phase to move is the causal one.
        order = [f"{a}->{b}" for a, b in itertools.pairwise(exp["phases_present"])]
        seen = set(order)
        for phase in [*order, *sorted((set(got["deltas"]) | set(exp["deltas"])) - seen)]:
            assert got["deltas"].get(phase) == exp["deltas"].get(phase), f"{name}[{i}]: phase {phase} moved"

    stored = np.load(mat_path)
    for i, m in enumerate(_final_matrices(records)):
        np.testing.assert_allclose(m, stored[str(i)], rtol=_SMOOTH_RTOL, atol=0, err_msg=f"{name}[{i}] post-smooth")


def test_the_coplanar_phase_is_actually_covered():
    """A net that never exercises the coplanar phase cannot catch a coplanar regression.

    The 1,4 dihedral bound is the most fragile thing in the writer (it reads angles ANOTHER builder emitted,
    after commit), so assert at least one fixture reaches it rather than trusting the fixture list.
    """
    records = _records_for("henry_ni", fx.all_fixtures()["henry_ni"])
    assert any("coplanar" in r for r in records), "no fixture fires Coplanar.dg_post — the net has a hole"


def test_the_centroid_relief_phase_is_actually_covered():
    """Likewise for the haptic centroid dummy, whose matrix is larger than the stored molecule."""
    records = _records_for("ferrocene", fx.all_fixtures()["ferrocene"])
    assert any("centroid" in r for r in records), "no fixture fires Haptic.dg_relief"
