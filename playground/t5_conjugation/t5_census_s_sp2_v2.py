"""T5 census v2: ('S', SP2) donor fold, applying the SAME exemptions _donor_walk uses (donation_axis).

Splits the population into:
  (a) ALL non-exempt ('S', SP2) M-S-X angles the fold gate would see;
  (b) the CONJUGATED subset (planarity measurable — i.e. the thione-like case henry S6 represents).
Also reports each S donor's environment (n heavy neighbours, whether its D-X bond is conjugated).
"""

from __future__ import annotations
import sys
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/corpus_regression")
from cr_corpus import CORPUS, crystal_full  # noqa: E402
from rxembed.io import _xyz_to_mol
from rxembed import geometry as geo
from rxembed.constraints.metal import metal_indices

SP2 = Chem.HybridizationType.SP2


def main():
    all_recs = []  # (name, msym, s, x, angle, oop, conj)
    per_donor = []  # (name, s, n_heavy, arom, any_conj)
    for name, path, _half in CORPUS:
        try:
            _s, _x, q, _k = crystal_full(path)
            mol = _xyz_to_mol(path, charge=q)
        except Exception:
            continue
        metals = set(metal_indices(mol))
        if not metals:
            continue
        pos = mol.GetConformer().GetPositions()
        hyb = geo._stripped_hybridisation(mol)
        all_donors = set()
        for m in metals:
            all_donors |= {nb.GetIdx() for nb in mol.GetAtomWithIdx(m).GetNeighbors()}
        for m in metals:
            sphere = {nb.GetIdx() for nb in mol.GetAtomWithIdx(m).GetNeighbors()}
            for d in sorted(sphere):
                if mol.GetAtomWithIdx(d).GetAtomicNum() != 16 or hyb.get(d) != SP2:
                    continue
                subs = geo.donation_axis(mol, d, all_donors, sphere)  # None=exempt (bridging/haptic/H)
                heavy = [nb.GetIdx() for nb in mol.GetAtomWithIdx(d).GetNeighbors() if nb.GetAtomicNum() > 1]
                anyconj = any(
                    (b := mol.GetBondBetweenAtoms(d, h)) is not None and b.GetIsConjugated() and hyb.get(h) == SP2
                    for h in heavy
                )
                per_donor.append(
                    (
                        name,
                        d,
                        len(heavy),
                        mol.GetAtomWithIdx(d).GetIsAromatic(),
                        anyconj,
                        "EXEMPT" if subs is None else "judged",
                    )
                )
                if subs is None:
                    continue
                for x in subs:
                    ang = geo._angle(pos[m], pos[d], pos[x])
                    oop = geo._planarity_dev(mol, pos, hyb, m, d, x)
                    conj = (
                        (b := mol.GetBondBetweenAtoms(d, x)) is not None and b.GetIsConjugated() and hyb.get(x) == SP2
                    )
                    all_recs.append(
                        (
                            name,
                            mol.GetAtomWithIdx(m).GetSymbol(),
                            d,
                            x,
                            round(ang, 1),
                            None if oop is None else round(oop, 1),
                            conj,
                        )
                    )

    print("=== per sp2-S-donor environment ===")
    for r in per_donor:
        print(f"  {r[0]:14s} S{r[1]:<4d} heavy_nbrs={r[2]} aromatic={r[3]} has_conj_sp2_nbr={r[4]}  {r[5]}")

    print("\n=== non-exempt M-S-X angle records (what the fold gate would judge) ===")
    for r in all_recs:
        print(f"  {r[0]:14s} {r[1]:2s} S{r[2]}-X{r[3]}  M-S-X={r[4]:6.1f}  oop={r[5]}  conj_bond={r[6]}")

    ang = np.array([r[4] for r in all_recs], float)
    conj_ang = np.array([r[4] for r in all_recs if r[6]], float)
    oop = np.array([r[5] for r in all_recs if r[5] is not None], float)
    ns = len({r[0] for r in all_recs})
    print(f"\nALL non-exempt ('S',SP2): n_angles={len(ang)} from {ns} structures")
    if len(ang):
        print(
            f"  M-S-X: min={ang.min():.1f} p0.5={np.percentile(ang, 0.5):.1f} median={np.median(ang):.1f} "
            f"p95={np.percentile(ang, 95):.1f} max={ang.max():.1f}  (spread={ang.max() - ang.min():.1f})"
        )
        print(
            f"  candidate floor=min(p0.5,min)-5={min(np.percentile(ang, 0.5), ang.min()) - 5:.1f} ceiling={ang.max():.1f}"
        )
    print(
        f"CONJUGATED subset (thione-like, the henry S6 population): n_angles={len(conj_ang)} "
        f"from {len({r[0] for r in all_recs if r[6]})} structures; planarity-measurable n={len(oop)}"
    )
    if len(conj_ang):
        print(f"  conj M-S-X: min={conj_ang.min():.1f} median={np.median(conj_ang):.1f} max={conj_ang.max():.1f}")


if __name__ == "__main__":
    main()
