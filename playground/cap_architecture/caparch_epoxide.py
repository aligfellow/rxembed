"""case4 ester-collapse ('epoxide') rate under the structural cap rule.

capsweep_lib.epoxide: iso3, 40 seeds, n=1, fraction with O2-C3-O4 < 90 deg (+ median angle).
The structural predicate skips case4's plane-locked ester O4 cap -> should reduce the collapse
(diagnosis: O4 cap off -> 12/40 -> 0/40).

Usage: uv run --no-sync python playground/cap_architecture/caparch_epoxide.py <mode>   (base|ff|both)
"""

from __future__ import annotations

import sys

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")
sys.path.insert(0, "playground/cap_architecture")

import capsweep_lib as L  # noqa: E402
import caparch_spike as S  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "base"
if mode != "base":
    S.install(mode)
coll, tot, med = L.epoxide(nseeds=40)
print(f"mode={mode}: ester collapse {coll}/{tot}  median O-C-O angle {med:.1f} deg  skip={dict(S.SKIP_STATS)}")
S.reset()
