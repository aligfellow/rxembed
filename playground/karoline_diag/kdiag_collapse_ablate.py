"""Ablation on the case4 ester collapse (O2-C3-O4 < 90 deg after minimize).

Counts, per ablation, how many seeds end with a collapsed ester on iso3 (the arrangement that shows it).
Ablations: none, seam, caps(sp2+conj), relief, and (extra) the metal Coplanar cap + _coplanar_donor.
Each is a fresh subprocess-free monkeypatch; we run one ablation per invocation to keep them isolated.
"""

from __future__ import annotations

import sys

import numpy as np

import rxembed as rx
import kdiag_harness as H

rx.set_verbose("CRITICAL")


def oco_angle(pos):
    v1, v2 = pos[2] - pos[3], pos[4] - pos[3]
    return np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))


def extra_ablation(what):
    """Beyond the harness set: disable the metal Coplanar mechanism, or _coplanar_donor writer."""
    from rxembed.constraints import mechanisms as mech
    from rxembed.constraints import metal as _metal

    if what == "coplanar_mech":
        mech.Coplanar.ff_terms = lambda self, ff, cons, conf, fc: None
        mech.Coplanar.dg_post = lambda self, cons, ctx: None
    if what == "coplanar_donor":
        _metal._coplanar_donor = lambda *a, **k: None
    if what == "orient_donor":
        _metal._orient_donor = lambda *a, **k: None


def run(ablate, nseeds=40):
    if ablate in ("coplanar_mech", "coplanar_donor", "orient_donor"):
        extra_ablation(ablate)
    elif ablate != "none":
        H.apply_ablation(ablate)
    iso_set = rx.metal(H.CASES["case4"], "square_planar")
    collapses = 0
    total = 0
    angles = []
    for k in (3,):  # iso3 is the arrangement that collapses; check it specifically
        iso = iso_set[k]
        for s in range(nseeds):
            seed = 0xF00D + s
            ens = rx.embed(iso, n=1, seed=seed).minimize()
            if not ens.ids:
                continue
            total += 1
            a = oco_angle(ens.mol.GetConformer(ens.ids[0]).GetPositions())
            angles.append(a)
            if a < 90.0:
                collapses += 1
    med = float(np.median(angles)) if angles else float("nan")
    print(f"ablate={ablate:16} iso3: {collapses}/{total} collapsed (O-C-O<90); median O-C-O={med:.1f} deg")


if __name__ == "__main__":
    ablate = sys.argv[1] if len(sys.argv) > 1 else "none"
    run(ablate)
