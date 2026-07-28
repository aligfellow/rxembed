"""Coplanar-cap softening sweep — measurement library.

Two independent levers, both monkeypatched (src/ untouched):
  - metal._COPLANAR_CAP   : the dihedral half-window (deg), baked into cons.coplanar tuples at rx.metal() time
                            -> enters BOTH the DG bound (Coplanar.dg_post) and the FF torsion window (Coplanar.ff_terms)
  - mechanisms._COPLANAR_FC : the FF torsion force constant (kcal/rad^2), read by Coplanar.ff_terms at relax time

A third "shape": crowding-conditional, applied by wrapping Coplanar.ff_terms so that when a metal carries
>= CROWD conjugated sp2 donors (len{e[1] for e in cons.coplanar}) the FF force and/or window are softened,
while <= 2-donor controls keep the baseline.

Metrics per setting:
  - flag_rate(case): fraction of post-minimize conformers that geom.check flags, + kind histogram (all isomers)
  - epoxide(case4): fraction of iso3 seeds with O2-C3-O4 < 90 deg (the silent ester collapse) + median angle
  - thione(case1): Ni-S-C-N in-plane dihedral folded to [0,90] — median / p95 / max spread
  - control_oop(name): relaxed metal out-of-plane (deg) on each capped donor — median / max
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
from rdkit.Chem import rdMolTransforms as T

sys.path.insert(0, "playground/karoline_diag")

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import mechanisms as mech
from rxembed.constraints import metal as M  # noqa: N812

import kdiag_harness as H

rx.set_verbose("CRITICAL")

HENRY = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
KETONE = "CC(C)=O->[Pd](Cl)(Cl)<-n1ccccc1"
PICO = "O=C1[O-]->[Ni+2]2(<-[NH2]CC[NH2]->2)<-n2ccccc21"

_CROWD_DEFAULT = 3  # a metal carrying >= this many capped sp2 donors is "crowded"


# ---------------- lever control ----------------
_orig_ff = mech.Coplanar.ff_terms
_orig_dg = mech.Coplanar.dg_post


def set_levers(cap=45.0, fc=10.0, crowd_fc=None, crowd_cap=None, crowd_n=_CROWD_DEFAULT, crowd_dg_off=False):
    """Install the cap/fc levers. crowd_fc/crowd_cap (if set) override fc/cap ONLY on crowded metals (>= crowd_n).

    crowd_dg_off: also SKIP the DG 1,4 seed bound (Coplanar.dg_post) on crowded metals.
    """
    M._COPLANAR_CAP = cap
    mech._COPLANAR_FC = fc

    def dg_post(self, cons, ctx):
        if crowd_dg_off and len({e[1] for e in cons.coplanar}) >= crowd_n:
            return  # skip the seed bound on a crowded metal
        return _orig_dg(self, cons, ctx)

    mech.Coplanar.dg_post = dg_post

    def ff(self, ff_obj, cons, conf, fc_dist):
        n_capped = len({e[1] for e in cons.coplanar})
        crowded = n_capped >= crowd_n
        use_fc = crowd_fc if (crowded and crowd_fc is not None) else fc
        # rebuild the torsion terms with per-metal fc + optional per-metal cap widening
        from rdkit.Chem import rdMolTransforms as _T

        from rxembed.constraints.mechanisms import _coplanar_window

        use_cap = crowd_cap if (crowded and crowd_cap is not None) else None
        for i, j, k, w, _anchor, cap_t in cons.coplanar:
            phi = _T.GetDihedralDeg(conf, i, j, k, w)
            lo, hi = _coplanar_window(phi, use_cap if use_cap is not None else cap_t)
            ff_obj.UFFAddTorsionConstraint(i, j, k, w, False, lo, hi, use_fc)

    mech.Coplanar.ff_terms = ff


def reset_levers():
    M._COPLANAR_CAP = 45.0
    mech._COPLANAR_FC = 10.0
    mech.Coplanar.ff_terms = _orig_ff
    mech.Coplanar.dg_post = _orig_dg


# ---------------- metrics ----------------
def flag_rate(case, n=6, seed=0xF00D):
    """Post-minimize geom.check flag rate across ALL isomers of a case, + violation-kind histogram."""
    iso_set = rx.metal(H.CASES[case] if case in H.CASES else case, "square_planar")
    kinds = Counter()
    flagged = nconf = 0
    for iso in iso_set:
        donors = list(iso.donors)
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        sphere = sorted({d for ds in ens.sphere.values() for d in ds}) or donors
        metals = set(M.metal_indices(ens.mol))
        for cid in ens.ids:
            rep = geo.check(ens.mol, cid, donors=sphere)
            formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
            nconf += 1
            k = Counter(v.kind for v in rep.violations)
            if not rep.ok() or formed or broken:
                flagged += 1
            kinds.update(k)
    rate = flagged / nconf if nconf else float("nan")
    return rate, nconf, dict(kinds)


def _oco_angle_dist(pos):
    v1, v2 = pos[2] - pos[3], pos[4] - pos[3]
    a = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    return a, float(np.linalg.norm(pos[2] - pos[4]))


def epoxide(nseeds=40):
    """case4 iso3: fraction of seeds ending with the ester O2-C3-O4 collapsed (< 90 deg); + median angle."""
    iso_set = rx.metal(H.CASES["case4"], "square_planar")
    iso = iso_set[3]  # iso3 = the O4 N23 C16 O6 arrangement that collapses
    collapses = total = 0
    angles = []
    for s in range(nseeds):
        ens = rx.embed(iso, n=1, seed=0xF00D + s).minimize()
        if not ens.ids:
            continue
        total += 1
        a, _d = _oco_angle_dist(ens.mol.GetConformer(ens.ids[0]).GetPositions())
        angles.append(a)
        if a < 90.0:
            collapses += 1
    med = float(np.median(angles)) if angles else float("nan")
    return collapses, total, med


def _fold_acute(x):
    """Fold a dihedral to [0, 90]: distance from the nearest in-plane well (0 or 180)."""
    x = abs(x) % 180.0
    return min(x, 180.0 - x)


def thione(n=8, seed=0xF00D):
    """case1: the thione Ni-S=C ride out of plane. Ni-S-C-N dihedral folded to [0,90] over all isomers."""
    iso_set = rx.metal(H.CASES["case1"], "square_planar")
    devs = []
    for iso in iso_set:
        mol, m = iso.mol, iso.metal
        s = next(d for d in iso.donors if mol.GetAtomWithIdx(d).GetSymbol() == "S")
        c = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(s).GetNeighbors() if nb.GetAtomicNum() > 1)
        ref = max(
            (nb for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != s and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
            default=None,
        )
        if ref is None:
            continue
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        for cid in ens.ids:
            devs.append(_fold_acute(T.GetDihedralDeg(ens.mol.GetConformer(cid), m, s, c, ref.GetIdx())))
    devs = np.array(devs) if devs else np.array([float("nan")])
    return float(np.median(devs)), float(np.percentile(devs, 95)), float(devs.max()), len(devs)


def _oop(mol, cid, i, j, k, w):
    """Angle (deg) of atom i (metal) out of the plane through j,k,w — 0 when coplanar (from test_coplanar)."""
    p = mol.GetConformer(cid).GetPositions()
    nrm = np.cross(p[k] - p[j], p[w] - p[j])
    nn = np.linalg.norm(nrm)
    if nn < 1e-6:
        return 0.0
    nrm /= nn
    v = p[i] - p[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(nrm, v))))))


def control_oop(smi, n=8, seeds=(1, 7, 13, 21)):
    """Relaxed metal out-of-plane per capped donor on a control complex. Returns {donor_idx: (median, max, sym)}."""
    per = {}
    for seed in seeds:
        iso = rx.metal(smi, "square_planar")[0]
        mol, m = iso.mol, iso.metal
        for e in iso.cons.coplanar:
            _i, d, k, w, _anc, _cap = e
            per.setdefault(d, {"vals": [], "sym": mol.GetAtomWithIdx(d).GetSymbol(), "kw": (k, w)})
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        for e in iso.cons.coplanar:
            _i, d, k, w, _anc, _cap = e
            for cid in ens.ids:
                per[d]["vals"].append(_oop(ens.mol, cid, m, d, k, w))
    out = {}
    for d, rec in per.items():
        vals = np.array(rec["vals"]) if rec["vals"] else np.array([float("nan")])
        out[d] = (float(np.median(vals)), float(vals.max()), rec["sym"], len(vals))
    return out
