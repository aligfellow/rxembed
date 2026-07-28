"""Q3 safety proof (fast, no xtb): hold-at-seed cannot flatten a genuinely puckered start.

Start every relax from a PUCKERED geometry -- UFF's own bowl of corannulene (~9 deg) -- not the flat ETKDG
seed. hold-at-seed is a +-5 deg window CENTRED ON THE START, so it is mathematically incapable of
flattening the pucker; hold-flat (target 0) drives it to ~0. This isolates the preserve-vs-target-0
difference that the flat ETKDG seed masked, and shows why hold-at-seed is the safe choice for a
"genuinely-pyramidal sp2 carbon".
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms

_HOLD_FC, _HOLD_WIN = 10.0, 5.0
CORANNULENE = "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51"


def _sp2_carbons(mol):
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) == 3:  # noqa: PLR2004
            out.append((atom.GetIdx(), nbrs))
    return out


def _pyr(pos, centre, nbrs):
    n = np.cross(pos[nbrs[1]] - pos[nbrs[0]], pos[nbrs[2]] - pos[nbrs[0]])
    n /= np.linalg.norm(n) + 1e-12
    return float(
        np.mean(
            [
                abs(
                    np.degrees(
                        np.arcsin(
                            np.clip(
                                np.dot((pos[centre] - pos[j]) / (np.linalg.norm(pos[centre] - pos[j]) + 1e-12), n),
                                -1,
                                1,
                            )
                        )
                    )
                )
                for j in nbrs
            ]
        )
    )


def worst_pyr(mol, pos):
    v = [(_pyr(pos, c, nb), c) for c, nb in _sp2_carbons(mol)]
    return max(v) if v else (0.0, None)


def relax(mol, start, hold=None):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, xyz in enumerate(start):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, ignoreInterfragInteractions=False)
    if hold is not None:
        for centre, nbrs in _sp2_carbons(m):
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], centre)
            target = phi if hold == "seed" else (0.0 if abs(phi) < 90 else (180.0 if phi >= 0 else -180.0))  # noqa: PLR2004
            ff.UFFAddTorsionConstraint(
                nbrs[0], nbrs[1], nbrs[2], centre, False, target - _HOLD_WIN, target + _HOLD_WIN, _HOLD_FC
            )
    ff.Initialize()
    ff.Minimize(maxIts=2000)
    return m.GetConformer().GetPositions().copy()


def main():
    mol = Chem.AddHs(Chem.MolFromSmiles(CORANNULENE))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    seed = mol.GetConformer().GetPositions()
    bowl = relax(mol, seed)  # UFF's own puckered minimum (the "genuinely pyramidal" geometry)
    wb, cb = worst_pyr(mol, bowl)
    cbn = next(nb for c, nb in _sp2_carbons(mol) if c == cb)

    print("=" * 72)
    print(f"corannulene: UFF bowl worst sp2-C pyramidalisation = {wb:.1f} deg (a genuine pucker)")
    print("relax RE-STARTED from that bowl -- does the hold preserve it?")
    print("=" * 72)
    print(f"  {'variant':14s} {'worst_pyr(deg)':>14s} {'hub_C_pyr(deg)':>15s}")
    print(f"  {'bowl (start)':14s} {wb:14.1f} {_pyr(bowl, cb, cbn):15.1f}")
    for name, hold in (("holdC_seed", "seed"), ("holdC_flat", "flat")):
        pos = relax(mol, bowl, hold)
        print(f"  {name:14s} {worst_pyr(mol, pos)[0]:14.1f} {_pyr(pos, cb, cbn):15.1f}")
    print("\n  READ: holdC_seed preserves the ~9 deg pucker (window is around the start);")
    print("        holdC_flat collapses it toward 0 (would wrongly freeze a should-pucker centre flat).")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    main()
