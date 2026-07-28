"""Q1: is the torsion-improper the best UFF shape for holding an sp2 carbon planar?

Replicate restrained_uff's organic UFF build from the planar ETKDG seed on chb-tetramisole (the T3d case),
then add the sp2-planarity hold four ways and measure the resulting worst sp2-carbon off-plane (gate 0.15 A):

  none          restrained_uff, no sp2 hold                                    (baseline, should FAIL)
  torsion_win   UFFAddTorsionConstraint(n0,n1,n2,C, seed +-5 deg, fc=10)       <- SHIPPED Sp2Planar
  torsion_pt    UFFAddTorsionConstraint(n0,n1,n2,C, seed, seed, fc=10)         (point target, real spring)
  angle_triple  3x UFFAddAngleConstraint(ni,C,nj, 120,120, fc=10)             (planarity via sum-to-360)

The improper (out-of-plane) coordinate is a 4-body quantity; UFF's constraint API is {Distance, Angle,
Torsion, Position}. This asks whether the 1-term torsion improper is at least as effective as the 3-term
angle alternative (and Position/Distance do not encode an out-of-plane coordinate at all).

Usage:  uv run python playground/ff_handling/ffh_q1_improper_shapes.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

import rxembed as rx  # noqa: E402
from rxembed import geometry as geo  # noqa: E402
from rxembed.constraints import mechanisms as _mech  # noqa: E402
from rxembed.pipeline import Ensemble  # noqa: E402

FC = 10.0
WIN = 5.0
_SNAP: list = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    _SNAP.append({int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids})
    return _orig(self)


Ensemble._relax_into_windows = _patched


def sp2_carbons(mol):
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) == 3:
            out.append((atom.GetIdx(), nbrs))
    return out


def worst_planar(mol, pos, exclude):
    return max((geo._plane_offset(pos[c], pos[nb]) for c, nb in sp2_carbons(mol) if c not in exclude), default=0.0)


def relax(mol, cons, seed_pos, cid, shape):
    m = Chem.Mol(mol)
    conf = m.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, confId=cid, ignoreInterfragInteractions=False)
    for mech in _mech.REGISTRY:
        if mech.field == "sp2_planar":  # replace the shipped hold with the shape under test
            continue
        mech.ff_terms(ff, cons, m.GetConformer(cid), 1e4)
    frozen = set(cons.frozen)
    for centre, nbrs in sp2_carbons(m):
        if centre in frozen:
            continue
        if shape == "torsion_win":
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], centre)
            ff.UFFAddTorsionConstraint(nbrs[0], nbrs[1], nbrs[2], centre, False, phi - WIN, phi + WIN, FC)
        elif shape == "torsion_pt":
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], centre)
            ff.UFFAddTorsionConstraint(nbrs[0], nbrs[1], nbrs[2], centre, False, phi, phi, FC)
        elif shape == "angle_triple":
            for u, v in ((0, 1), (1, 2), (0, 2)):
                ff.UFFAddAngleConstraint(nbrs[u], centre, nbrs[v], False, 120.0, 120.0, FC)
    ff.Initialize()
    ff.Minimize(maxIts=500)
    return m.GetConformer(cid).GetPositions()


def run(case, seed):
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    _SNAP.clear()
    ens = rx.embed(seed=seed, **resolve(entry))
    if not hasattr(ens, "cons"):
        ens = list(ens)[0]
    mol = ens.mol
    cid = int(next(iter(ens.ids)))
    seed_pos = _SNAP[0][cid]
    frozen = {int(f) for f in ens.cons.frozen}
    metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z}
    exclude = frozen | metals
    row = {"seed": worst_planar(mol, seed_pos, exclude)}
    for shape in ("none", "torsion_win", "torsion_pt", "angle_triple"):
        row[shape] = worst_planar(mol, relax(mol, ens.cons, seed_pos, cid, shape), exclude)
    return row


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    print(f"chb-tetramisole worst sp2-carbon off-plane (A), gate 0.15  [P! = fails]")
    print(f"  {'seed':>4s} {'seed':>8s} {'none':>10s} {'torsion_win':>12s} {'torsion_pt':>11s} {'angle_triple':>13s}")
    for seed in (1, 2, 3):
        r = run("chb-tetramisole", seed)

        def f(k):
            return f"{r[k]:.3f}" + (" P!" if r[k] > 0.15 else "   ")

        print(
            f"  {seed:>4d} {r['seed']:>8.3f} {f('none'):>10s} {f('torsion_win'):>12s} {f('torsion_pt'):>11s} {f('angle_triple'):>13s}"
        )
