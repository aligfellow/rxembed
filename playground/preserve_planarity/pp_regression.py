"""Q3: does the sp2-planarity HOLD wrongly freeze a centre that SHOULD pucker?

The danger: an sp2-perceived carbon whose true minimum is non-planar (a bowl PAH, a strained ring). A
hold-at-SEED window can only ever PREVENT the relax adding pucker (it is a window around the seed's own
improper) -- so it can never flatten a genuinely-pyramidal seed. A hold-at-FLAT (target 0) window WOULD
force such a centre flat. This measures both against GFN2 truth on curved-sp2 molecules.

For each molecule: ETKDG seed -> {plain UFF, +holdC_seed, +holdC_flat, MMFF, GFN2} and report the worst
and the named-centre sp2-carbon improper (deg pyramidalisation) under each. GFN2 is ground truth.

Usage:  uv run python pp_regression.py
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms

from rxembed import geometry as geo

XTB = "/home/ali/bin/g-xtb/binaries/xtb-6.7.1/bin/xtb"
_HOLD_FC, _HOLD_WIN = 10.0, 5.0

# curved / strained sp2-carbon systems (true minimum is non-planar)
MOLS = {
    "corannulene": "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51",
    "sumanene-ish": "C1c2ccc3Cc4ccc5Cc6ccc(c1c2c36)c4c56",  # bowl
    "cyclopropene-vinyl": "C1=CC1/C=C/C=O",  # strained sp2 next to conjugation
    "acenaphthylene": "c1ccc2cccc3C=Cc1c23",  # a strained sp2 C=C in a 5-ring on a naphthalene
}


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
    """Improper pyramidalisation (deg): mean |asin(bond . plane_normal)| over the 3 bonds. 0 = planar."""
    n = np.cross(pos[nbrs[1]] - pos[nbrs[0]], pos[nbrs[2]] - pos[nbrs[0]])
    n /= np.linalg.norm(n) + 1e-12
    devs = []
    for j in nbrs:
        v = pos[centre] - pos[j]
        v /= np.linalg.norm(v) + 1e-12
        devs.append(abs(np.degrees(np.arcsin(np.clip(np.dot(v, n), -1, 1)))))
    return float(np.mean(devs))


def worst_pyr(mol, pos):
    c = _sp2_carbons(mol)
    if not c:
        return 0.0, None
    vals = [(_pyr(pos, ce, nb), ce) for ce, nb in c]
    return max(vals)


def relax(mol, seed_pos, hold=None):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, xyz in enumerate(seed_pos):
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
    ff.Minimize(maxIts=1000)
    return m.GetConformer().GetPositions().copy()


def mmff(mol, seed_pos):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(m):
        return None
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m, props, ignoreInterfragInteractions=False)
    ff.Minimize(maxIts=1000)
    return m.GetConformer().GetPositions().copy()


def gfn2(mol, seed_pos):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    with tempfile.TemporaryDirectory() as d:
        Chem.MolToXYZFile(m, str(Path(d) / "in.xyz"))
        subprocess.run(
            [XTB, "in.xyz", "--gfn", "2", "--opt", "--chrg", str(Chem.GetFormalCharge(m))],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=600,
        )
        op = Path(d) / "xtbopt.xyz"
        return Chem.MolFromXYZFile(str(op)).GetConformer().GetPositions() if op.exists() else None


def run(name, smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"\n{name}: SMILES parse failed")
        return
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
        print(f"\n{name}: embed failed")
        return
    seed_pos = mol.GetConformer().GetPositions()
    rows = [
        ("seed", seed_pos),
        ("uff_plain", relax(mol, seed_pos)),
        ("+holdC_seed", relax(mol, seed_pos, "seed")),
        ("+holdC_flat", relax(mol, seed_pos, "flat")),
    ]
    mm = mmff(mol, seed_pos)
    if mm is not None:
        rows.append(("mmff94s", mm))
    g2 = gfn2(mol, seed_pos)
    if g2 is not None:
        rows.append(("gfn2_opt (truth)", g2))

    # pick the centre GFN2 keeps most pyramidal (the "should-pucker" atom)
    truth = dict(rows).get("gfn2_opt (truth)")
    tgt_centre = worst_pyr(mol, truth)[1] if truth is not None else worst_pyr(mol, seed_pos)[1]
    tgt_nbrs = next(nb for ce, nb in _sp2_carbons(mol) if ce == tgt_centre)

    print(f"\n{'=' * 78}\n{name}   {smi}\n  truth's most-pyramidal sp2 C = atom {tgt_centre}\n{'=' * 78}")
    print(f"  {'method':18s} {'worst_pyr(deg)':>14s} {'C%d_pyr(deg)' % tgt_centre:>14s}")
    for nm, pos in rows:
        wp, _ = worst_pyr(mol, pos)
        cp = _pyr(pos, tgt_centre, tgt_nbrs)
        print(f"  {nm:18s} {wp:14.1f} {cp:14.1f}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    for name, smi in MOLS.items():
        try:
            run(name, smi)
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n{name} EXC {type(e).__name__}: {e}")
            traceback.print_exc()
