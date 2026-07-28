"""The FC>=10 pin: test_reembed_retry_delivers_clean_geometry's rigid-diene diphosphine case.

Measure its capped-donor count (does the crowding predicate exempt it?) and its post-minimize
geom.check-clean delivery at baseline vs global-FC-reduction vs crowding-conditional.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/tmp/claude-1000/-home-ali-Documents-Codes-rxembed/7cb588a7-0edf-4fc6-8fb2-d291e28d68b7/scratchpad")

import capsweep_lib as L
import rxembed as rx
from rxembed import geometry as geo

# the exact SMILES from tests/test_metal_chelates.py::test_reembed_retry_delivers_clean_geometry
RDP = (
    "CCC1=C2CCCCC2=C(CC)[P](c2ccccc2)(c2ccccc2)->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[P]1(c1ccccc1)c1ccccc1"
)

iso = rx.metal(RDP, "square_planar")[0]
n_capped = len({e[1] for e in iso.cons.coplanar})
syms = {e[1]: iso.mol.GetAtomWithIdx(e[1]).GetSymbol() for e in iso.cons.coplanar}
print(f"RDP pin-test case: capped_sp2_donors={n_capped}  {syms}")


def delivers(label, **kw):
    """The test's own bar: embed(n=6).minimize() must return >=6 all-clean geometries."""
    L.set_levers(**kw)
    iso = rx.metal(RDP, "square_planar")[0]
    ens = rx.embed(iso, n=6, seed=0xF00D).minimize()
    clean = sum(1 for c in ens.ids if geo.check(ens.mol, c).ok())
    # the single-seed bar too (n=1 must be clean)
    iso1 = rx.metal(RDP, "square_planar")[0]
    ens1 = rx.embed(iso1, n=1, seed=0xF00D).minimize()
    single_ok = bool(ens1.ids) and geo.check(ens1.mol, ens1.ids[0]).ok()
    L.reset_levers()
    passes = ens.n >= 6 and clean == len(ens.ids) and single_ok
    print(
        f"{label:22}: n=6 -> {len(ens.ids)} kept, {clean} clean; n=1 clean={single_ok}  "
        f"=> TEST {'PASS' if passes else 'FAIL'}"
    )


delivers("baseline fc10", cap=45.0, fc=10.0)
delivers("global fc3", cap=45.0, fc=3.0)
delivers("global fc1", cap=45.0, fc=1.0)
delivers("global cap75", cap=75.0, fc=10.0)
delivers("crowd>=3 fc1", cap=45.0, fc=10.0, crowd_fc=1.0, crowd_n=3)
