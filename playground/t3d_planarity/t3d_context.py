"""Context for the planarity finding: cpa constraint adjacency, the bulk contributor, seed sweep.

1. cpa: print the constraints and confirm C73's neighbour C74 is in the frozen core (the distortion
   source: bare UFF keeps C73 planar, only the constrained relax pushes it out).
2. bimp-smiles-auto is the LARGEST planarity contributor (study: 0 -> 216). Characterise which atoms
   its relax flags -- are they the same S-C-N2 thiourea class as chb-tetramisole's amidine C3? Run the
   bare-UFF vs MMFF vs GFN-FF ablation on the top one to confirm the same UFF defect.
3. Seed sweep: confirm C73 / C3 flag on all five seeds (seed-independence of the mode).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

XTB = "/home/ali/bin/g-xtb/binaries/xtb-6.7.1/bin/xtb"

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def off(pos, c, nbrs):
    return geo._plane_offset(pos[c], pos[list(nbrs)])


def embed_case(case, seed):
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    _SNAP.clear()
    res = rx.embed(seed=seed, **resolve(entry))
    ens_list = list(res) if isinstance(res, (EnsembleSet, list)) else [res]
    snaps = {id(r["ens"]): r for r in _SNAP}
    return [(e, snaps[id(e)]) for e in ens_list if id(e) in snaps]


def sp2_class(mol, i):
    a = mol.GetAtomWithIdx(i)
    nb = sorted(n.GetSymbol() for n in a.GetNeighbors())
    return "".join(nb)


print("=" * 92)
print("1. cpa constraints and C73 frozen-core adjacency")
print("=" * 92)
for ens, snap in embed_case("cpa", 1):
    frozen = sorted(int(f) for f in ens.cons.frozen)
    print(f"  frozen core atoms: {frozen}")
    print(f"  n distance windows: {len(ens.cons.distances)}   n angle windows: {len(ens.cons.angles)}")
    c73 = ens.mol.GetAtomWithIdx(73)
    nbrs = [n.GetIdx() for n in c73.GetNeighbors()]
    print(f"  C73 neighbours: {nbrs}  -> frozen among them: {[n for n in nbrs if n in frozen]}")
    # any distance window touching 73?
    dw = [k for k in ens.cons.distances if 73 in k]
    print(f"  distance windows touching atom 73: {dw}")
    break

print()
print("=" * 92)
print("2. bimp-smiles-auto: which atoms its relax newly flags on planarity (the 216-count bulk)")
print("=" * 92)
flagged_classes: dict[str, int] = {}
examples = []
cands = embed_case("bimp-smiles-auto", 1)
print(f"  candidates: {len(cands)}")
for k, (ens, snap) in enumerate(cands):
    mol = ens.mol
    frozen = frozenset(int(f) for f in ens.cons.frozen)
    metals = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z)
    exclude = frozen | metals
    for c in ens.ids:
        cd = int(c)
        if cd not in snap["pre"]:
            continue
        relax_pos = mol.GetConformer(cd).GetPositions()
        for v in geo.planarity(mol, relax_pos, exclude=exclude):
            if "sp2 out of plane" not in v.detail:
                flagged_classes[f"RING:{v.detail}"] = flagged_classes.get(f"RING:{v.detail}", 0) + 1
                continue
            cls = sp2_class(mol, v.atoms[0])
            flagged_classes[cls] = flagged_classes.get(cls, 0) + 1
            if len(examples) < 6:
                a = mol.GetAtomWithIdx(v.atoms[0])
                examples.append((k, cd, v.atoms[0], cls, str(a.GetHybridization()), v.value))
print("  flagged sp2-centre neighbour-element classes (count over all cands/confs, seed 1):")
for cls, n in sorted(flagged_classes.items(), key=lambda x: -x[1]):
    print(f"    {cls:12s} {n}")
print("  examples (cand, conf, atom, nbr-elements, hyb, off):")
for e in examples:
    print(f"    {e}")

# ablation on one SCN-class centre from bimp-smiles-auto
scn = next((e for e in examples if set(e[3]) >= {"S", "N"}), None)
if scn is None:
    scn = examples[0] if examples else None
if scn is not None:
    k, cd, centre, cls, hyb, _ = scn
    ens, snap = cands[k]
    mol = ens.mol
    seed_pos = snap["pre"][cd]
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()]

    def relax_measure(kind):
        m = Chem.Mol(mol)
        conf = m.GetConformer(cd)
        for i, xyz in enumerate(seed_pos):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        if kind == "uff":
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, confId=cd, ignoreInterfragInteractions=False)
            ff.Minimize(maxIts=1000)
        elif kind == "mmff":
            if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(m):
                return None
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m, props, confId=cd, ignoreInterfragInteractions=False)
            ff.Minimize(maxIts=1000)
        elif kind == "gfnff":
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "in.xyz"
                Chem.MolToXYZFile(m, str(p), confId=cd)
                subprocess.run(
                    [XTB, "in.xyz", "--gfnff", "--opt", "--chrg", str(Chem.GetFormalCharge(m))],
                    cwd=d,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                op = Path(d) / "xtbopt.xyz"
                if not op.exists():
                    return None
                return off(Chem.MolFromXYZFile(str(op)).GetConformer().GetPositions(), centre, nbrs)
        return off(m.GetConformer(cd).GetPositions(), centre, nbrs)

    print(f"\n  ablation on bimp-smiles-auto atom {centre} (nbr-elements {cls}, hyb {hyb}):")
    print(f"    seed          {off(seed_pos, centre, nbrs):.3f}")
    print(f"    embed_relax   {off(mol.GetConformer(cd).GetPositions(), centre, nbrs):.3f}")
    print(f"    uff_plain     {relax_measure('uff'):.3f}")
    mm = relax_measure("mmff")
    print(f"    mmff94s       {('%.3f' % mm) if mm is not None else 'no params'}")
    gf = relax_measure("gfnff")
    print(f"    gfnff_opt     {('%.3f' % gf) if gf is not None else 'FAILED'}")

print()
print("=" * 92)
print("3. seed sweep: off-plane at cpa C73 and chb C3 across five seeds (first conf of first cand)")
print("=" * 92)
for case, centre, nbrs in [("cpa", 73, [72, 74, 76]), ("chb-tetramisole", 3, [2, 4, 7])]:
    print(f"  {case}  centre {centre}:")
    for sd in (1, 2, 3, 7, 0xF00D):
        vals = []
        for ens, snap in embed_case(case, sd):
            mol = ens.mol
            for c in ens.ids:
                cd = int(c)
                if cd in snap["pre"]:
                    vals.append(off(mol.GetConformer(cd).GetPositions(), centre, nbrs))
        n_flag = sum(v > 0.15 for v in vals)
        print(f"    seed {sd:<7} relax off-plane: min={min(vals):.3f} max={max(vals):.3f} flagged {n_flag}/{len(vals)}")
