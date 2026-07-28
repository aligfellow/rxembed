"""Measure the thione S=C-vs-coordination-plane angle for case1, across seeds.

For isomer 0, embed n conformers per seed, minimize, and per conformer measure:
  - oop_S    : metal out of the thione S's own sp2 plane (the cap's DOF), deg. 0 = coplanar.
  - dih_MSCN : the Ni-S6-C4-N3 dihedral the cap targets (anchor 180 anti), and its
               deviation from the nearest in-plane well (0/180), deg.
  - sc_coord : elevation of the S=C4 bond out of the coordination plane (best-fit plane
               through Ni + the 4 donors), deg. 0 = S=C lies in the coordination plane.

The real Ni is restored on a copy before any perception; the raw angles are pure Cartesian
(element-independent) but we confirm restore does not move atoms.

Usage: uv run --no-sync python playground/thione_diag/thione_angle.py [n] [seeds...]
"""

from __future__ import annotations

import sys

import numpy as np
from rdkit.Chem import rdMolTransforms as T

import rxembed as rx

rx.set_verbose("CRITICAL")

SMI = "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"


def oop(pos, i, j, k, w):
    """Angle (deg) of atom i out of the plane through j,k,w (0 = coplanar)."""
    nrm = np.cross(pos[k] - pos[j], pos[w] - pos[j])
    nn = np.linalg.norm(nrm)
    if nn < 1e-6:
        return 0.0
    nrm /= nn
    v = pos[i] - pos[j]
    v /= np.linalg.norm(v)
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(float(np.dot(nrm, v))))))


def bond_elevation(pos, a, b, plane_atoms):
    """Elevation (deg) of bond a-b out of the best-fit plane through plane_atoms. 0 = bond lies in plane."""
    P = pos[plane_atoms]
    c = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - c)
    n = vh[2]  # plane normal = smallest-singular-value direction
    bond = pos[b] - pos[a]
    bond /= np.linalg.norm(bond)
    return np.degrees(np.arcsin(min(1.0, abs(float(np.dot(bond, n))))))


def well_dev(phi):
    """Deviation of dihedral phi from its nearest in-plane well (0 syn or 180 anti), deg."""
    p = abs(phi)
    return min(p, abs(180.0 - p))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seeds = [int(x) for x in sys.argv[2:]] or [1, 7, 13, 21, 0xF00D]

    iso = rx.metal(SMI, "square_planar")[0]
    donors = sorted(iso.donors)  # [1, 6, 8, 18]
    metal = sorted(iso.cons.metals)[0]  # 7
    plane_atoms = [metal, *donors]
    # thione cap plane: donor S6, k=4 (C), w=3 (N3)
    S, C4, N3 = 6, 4, 3

    print(f"metal={metal}  donors={donors}  plane_atoms={plane_atoms}")
    print(f"S={S} C4={C4} N3={N3}   dihedral cap target = Ni-S-C4-N3 anchored 180 (anti), cap ±45")
    print(f"seeds={seeds}  n={n}\n")

    rows = {"oop_S": [], "dih_dev": [], "sc_coord": []}
    per_seed = {}
    for seed in seeds:
        ens = rx.embed(iso, n=n, seed=seed).minimize()
        s_oop, s_dev, s_el = [], [], []
        for cid in ens.ids:
            pos = ens.mol.GetConformer(cid).GetPositions()
            o = oop(pos, metal, S, C4, N3)
            dih = T.GetDihedralDeg(ens.mol.GetConformer(cid), metal, S, C4, N3)
            dev = well_dev(dih)
            el = bond_elevation(pos, S, C4, plane_atoms)
            s_oop.append(o)
            s_dev.append(dev)
            s_el.append(el)
        rows["oop_S"] += s_oop
        rows["dih_dev"] += s_dev
        rows["sc_coord"] += s_el
        per_seed[seed] = (np.median(s_oop), np.median(s_dev), np.median(s_el), len(ens.ids))
        print(
            f"  seed {seed:>7}: nconf={len(ens.ids):>2}  "
            f"oop_S med={np.median(s_oop):5.1f} max={np.max(s_oop):5.1f} | "
            f"dih_dev med={np.median(s_dev):5.1f} max={np.max(s_dev):5.1f} | "
            f"sc_coord med={np.median(s_el):5.1f} max={np.max(s_el):5.1f}"
        )

    print("\n--- pooled across all seeds ---")
    for k, v in rows.items():
        a = np.array(v)
        print(f"  {k:>9}: median={np.median(a):5.1f}  mean={a.mean():5.1f}  max={a.max():5.1f}  n={len(a)}")

    # seed sensitivity: spread of per-seed medians
    print("\n--- seed sensitivity (per-seed medians) ---")
    for lbl, idx in (("oop_S", 0), ("dih_dev", 1), ("sc_coord", 2)):
        meds = [per_seed[s][idx] for s in seeds]
        print(f"  {lbl:>9}: {[round(m, 1) for m in meds]}  spread={max(meds) - min(meds):.1f}")


if __name__ == "__main__":
    main()
