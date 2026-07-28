"""Spike the structural cap rule and measure flag rates on cases 1-4 + controls.

RULE (structural, element-agnostic, NOT a count):
  Cap donor D only if its plane is a FREE roll DOF about the M-D axis.  Skip the cap when D sits in a
  RIGID conjugated unit that also carries a co-donor of the same metal — i.e. some co-donor D' is reachable
  from D through a path whose every bond is aromatic / double / (single between two sp2 atoms).  Then the
  chelate bite + the rigid all-sp2 backbone already pin D's plane, so the cap is redundant (and, when several
  such donors crowd one metal, mutually over-determining).

Two spike modes:
  ff   : skip only the FF torsion for plane-locked donors (leaves the DG seed bound -> does NOT move golden)
  both : skip the whole cap for plane-locked donors, at the builder (no DG, no FF -> moves golden)

Baseline (no patch) is measured for comparison.

Usage: uv run --no-sync python playground/cap_architecture/caparch_spike.py <mode> [case ...]
       mode = base | ff | both
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")

from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import mechanisms as mech
from rxembed.constraints import metal as M  # noqa: N812

import capsweep_lib as L  # noqa: E402
import kdiag_harness as H  # noqa: E402

rx.set_verbose("CRITICAL")

CASES = {"case1": H.CASES["case1"], "case2": H.CASES["case2"], "case3": H.CASES["case3"], "case4": H.CASES["case4"]}
CONTROLS = {"HENRY": L.HENRY, "KETONE": L.KETONE, "PICO": L.PICO}

_SP2 = Chem.HybridizationType.SP2


def _rigid_bond(mol, u, v, hyb):
    b = mol.GetBondBetweenAtoms(int(u), int(v))
    if b is None:
        return False
    if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.SINGLE:
        return True
    return hyb.get(u) == _SP2 and hyb.get(v) == _SP2  # conjugated sp2-sp2 single bond is planar/rigid


def plane_locked(mol, d, donors, hyb):
    """True if donor d's plane is already fixed: a co-donor is reachable through an all-rigid conjugated path.

    The pipeline mol has the metal already bond-stripped, so GetShortestPath runs through the ligand backbone.
    """
    for dd in donors:
        if dd == d:
            continue
        path = Chem.GetShortestPath(mol, int(d), int(dd))
        if len(path) >= 2 and all(_rigid_bond(mol, u, v, hyb) for u, v in zip(path, path[1:])):
            return True
    return False


# ---------------- levers ----------------
_orig_coplanar_donor = M._coplanar_donor
_orig_ff = mech.Coplanar.ff_terms
SKIP_STATS = Counter()


def install(mode):
    """Install the structural skip in FF-only or full (both) mode."""
    if mode == "both":

        def patched_donor(mol, metal, d, donor_set, cons):
            hyb = geo._stripped_hybridisation(mol)
            if plane_locked(mol, d, donor_set, hyb):
                SKIP_STATS["skipped"] += 1
                return
            SKIP_STATS["capped"] += 1
            return _orig_coplanar_donor(mol, metal, d, donor_set, cons)

        M._coplanar_donor = patched_donor

    elif mode == "ff":
        from rdkit.Chem import rdMolTransforms as T

        from rxembed.constraints.mechanisms import _COPLANAR_FC, _coplanar_window

        def patched_ff(self, ff_obj, cons, conf, fc_dist):
            mol = conf.GetOwningMol()
            donors = {e[1] for e in cons.coplanar}
            # recover ALL donors of the metal from the M-donor distance keys
            all_don = set()
            for a, b in cons.distances:
                if a in cons.metals:
                    all_don.add(b)
                elif b in cons.metals:
                    all_don.add(a)
            hyb = geo._stripped_hybridisation(mol)
            for i, j, k, w, _anchor, cap in cons.coplanar:
                if plane_locked(mol, j, all_don or donors, hyb):
                    SKIP_STATS["skipped"] += 1
                    continue
                SKIP_STATS["capped"] += 1
                phi = T.GetDihedralDeg(conf, i, j, k, w)
                lo, hi = _coplanar_window(phi, cap)
                ff_obj.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, _COPLANAR_FC)

        mech.Coplanar.ff_terms = patched_ff


def reset():
    M._coplanar_donor = _orig_coplanar_donor
    mech.Coplanar.ff_terms = _orig_ff


# ---------------- measurement ----------------
def flag_rate(smi, n=8, seeds=(1, 7, 13, 21)):
    kinds = Counter()
    flagged = nconf = 0
    for seed in seeds:
        iso_set = rx.metal(smi, "square_planar")
        for iso in iso_set:
            ens = rx.embed(iso, n=n, seed=seed).minimize()
            sphere = sorted({d for ds in ens.sphere.values() for d in ds}) or list(iso.donors)
            metals = set(M.metal_indices(ens.mol))
            for cid in ens.ids:
                rep = geo.check(ens.mol, cid, donors=sphere)
                formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
                nconf += 1
                if not rep.ok() or formed or broken:
                    flagged += 1
                kinds.update(v.kind for v in rep.violations)
    return (flagged / nconf if nconf else float("nan")), nconf, dict(kinds)


def oop(mol, cid, i, j, k, w):
    p = mol.GetConformer(cid).GetPositions()
    nrm = np.cross(p[k] - p[j], p[w] - p[j])
    nn = np.linalg.norm(nrm)
    if nn < 1e-6:
        return 0.0
    nrm /= nn
    v = p[i] - p[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(nrm, v))))))


def control_hold(smi, ref_planes, n=8, seeds=(1, 7, 13, 21)):
    """Metal-out-of-plane per donor, measured against BASELINE cap planes (works in any mode)."""
    per = {d: [] for d in ref_planes}
    for seed in seeds:
        iso = rx.metal(smi, "square_planar")[0]
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        for d, (mi, k, w) in ref_planes.items():
            for cid in ens.ids:
                per[d].append(oop(ens.mol, cid, mi, d, k, w))
    return {d: (float(np.median(v)), float(np.max(v))) for d, v in per.items() if v}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    which = sys.argv[2:] or list(CASES) + list(CONTROLS)
    # capture baseline cap planes (unpatched) for the controls BEFORE installing the spike
    ref = {}
    for name in [c for c in which if c in CONTROLS]:
        iso = rx.metal(CONTROLS[name], "square_planar")[0]
        ref[name] = {e[1]: (e[0], e[2], e[3]) for e in iso.cons.coplanar}
    if mode != "base":
        install(mode)
    print(f"##### mode={mode} #####")
    for name in which:
        smi = (CASES | CONTROLS)[name]
        rate, nconf, kinds = flag_rate(smi)
        print(f"  {name:8} flag_rate={rate:5.1%}  n={nconf:3}  kinds={kinds}")
    print("  -- control in-plane hold (median/max metal-out-of-plane, deg, vs baseline planes) --")
    for name in [c for c in which if c in CONTROLS]:
        res = control_hold(CONTROLS[name], ref[name])
        for d, (medv, mx) in res.items():
            sym = rx.metal(CONTROLS[name], "square_planar")[0].mol.GetAtomWithIdx(d).GetSymbol()
            print(f"     {name} D={d}({sym}) median={medv:.1f} max={mx:.1f}")
    print(f"  skip stats: {dict(SKIP_STATS)}")
    reset()
