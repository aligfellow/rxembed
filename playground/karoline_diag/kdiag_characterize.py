"""Per-case characterization at post-minimize (the rendered stage): flag rate, kinds, empties, metrics.

Case-specific:
  case1: thione coplanarity dihedral Ni-S6-C4-N3 (in-plane at 0/180); spread = the +-45 window tail
  case4: ester O2-C3-O4 collapse fraction (< 90 deg = the epoxide)
  all:   donor_fold max, geom.check kinds

Usage: .venv/bin/python kdiag_characterize.py <case> [nseeds] [--ablate what]
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from rdkit.Chem import rdMolTransforms as T

import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import metal as _metal

import kdiag_harness as H

rx.set_verbose("CRITICAL")

# (metal, donor_S, thione_C, sub_N) for case1 coplanarity
THIONE = {"case1": (7, 6, 4, 3)}
ESTER = {"case4": (2, 3, 4)}  # O2-C3-O4


def extra_ablation(what):
    from rxembed.constraints import mechanisms as mech
    from rxembed.constraints import metal as _m

    if what == "coplanar_mech":
        mech.Coplanar.ff_terms = lambda self, ff, cons, conf, fc: None
        mech.Coplanar.dg_post = lambda self, cons, ctx: None
    elif what == "coplanar_donor":
        _m._coplanar_donor = lambda *a, **k: None
    elif what == "orient_donor":
        _m._orient_donor = lambda *a, **k: None
    elif what == "caps_on_metal":
        # FORCE the organic caps to run even on a metal system (remove the `if cons.metals: return` guard)
        import types

        def sp2(self, ff, cons, conf, fc):
            from rdkit.Chem import rdMolTransforms
            from rxembed import geometry as _geo

            mol = conf.GetOwningMol()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != _geo._CARBON_Z or atom.GetHybridization().name != "SP2":
                    continue
                if atom.GetIdx() in cons.frozen:
                    continue
                nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(nbrs) != _geo._SP2_DEGREE:
                    continue
                phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], atom.GetIdx())
                ff.UFFAddTorsionConstraint(nbrs[0], nbrs[1], nbrs[2], atom.GetIdx(), False, phi - 5, phi + 5, 10.0)

        mech.Sp2Planar.ff_terms = types.MethodType(sp2, mech.Sp2Planar())
        # bind as instance method on the registry instance
        from rxembed.constraints.mechanisms import REGISTRY

        for m in REGISTRY:
            if type(m).__name__ == "Sp2Planar":
                m.ff_terms = sp2.__get__(m)


def run(case, nseeds, ablate):
    if ablate in ("coplanar_mech", "coplanar_donor", "orient_donor", "caps_on_metal"):
        extra_ablation(ablate)
    elif ablate != "none":
        H.apply_ablation(ablate)
    smi = H.CASES[case]
    iso_set = rx.metal(smi, "square_planar")
    nconf = flagged = empties = attempts = 0
    kinds = Counter()
    folds = []
    thione_dih = []
    ester_collapse = 0
    for k, iso in enumerate(iso_set):
        for s in range(nseeds):
            seed = 0xF00D + s
            attempts += 1
            ens = rx.embed(iso, n=2, seed=seed).minimize()
            if not ens.ids:
                empties += 1
                continue
            sphere = sorted({d for ds in ens.sphere.values() for d in ds}) or list(iso.donors)
            for cid in ens.ids:
                nconf += 1
                rep = geo.check(ens.mol, cid, donors=sphere)
                if not rep.ok():
                    flagged += 1
                kinds.update(v.kind for v in rep.violations)
                fr = geo.donor_fold(ens.mol, cid, donors=sphere)
                folds.append(fr.fold)
                conf = ens.mol.GetConformer(cid)
                if case in THIONE:
                    m, d, c, n = THIONE[case]
                    dih = abs(T.GetDihedralDeg(conf, m, d, c, n))
                    thione_dih.append(min(dih, abs(180 - dih)))  # deviation from in-plane
                if case in ESTER:
                    o2, c3, o4 = ESTER[case]
                    pos = conf.GetPositions()
                    v1, v2 = pos[o2] - pos[c3], pos[o4] - pos[c3]
                    a = np.degrees(np.arccos(np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                    if a < 90:
                        ester_collapse += 1
    print(f"\n##### {case}  (ablate={ablate}, {nseeds} seeds x {len(iso_set)} isomers) #####")
    print(
        f"  attempts={attempts} empties={empties} conformers={nconf} flagged(geom)={flagged} ({100 * flagged / max(nconf, 1):.0f}%)"
    )
    print(f"  geom kinds: {dict(kinds)}")
    if folds:
        fa = np.array(folds)
        print(f"  donor_fold: median={np.median(fa):.1f} p95={np.percentile(fa, 95):.1f} max={fa.max():.1f} deg")
    if thione_dih:
        td = np.array(thione_dih)
        print(
            f"  THIONE Ni-S-C-N in-plane deviation: median={np.median(td):.1f} p95={np.percentile(td, 95):.1f} max={td.max():.1f} deg (0=in-plane)"
        )
    if case in ESTER:
        print(f"  ESTER collapse (O-C-O<90): {ester_collapse}/{nconf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("nseeds", nargs="?", type=int, default=12)
    ap.add_argument("--ablate", default="none")
    args = ap.parse_args()
    run(args.case, args.nseeds, args.ablate)
