"""Q2: which minimal mechanism clears BOTH the planarity (T3d) and the conjugation (T3c) mode?

Take the PLANAR ETKDG seed and relax it several ways, measuring for the WHOLE molecule (frozen/metal
excluded, exactly as geo.check does) both the worst sp2-carbon off-plane (planarity gate, 0.15 A) and the
worst conjugation quartet twist (conjugation gate, 30 deg):

  seed              raw ETKDG seed
  restrained_uff    the shipped relax (REGISTRY terms on a UFF FF)          <- baseline
  +holdC_seed       restrained_uff + sp2 CARBON improper held at seed value
  +holdCNO_seed     restrained_uff + ALL sp2 (C/N/O, 3-nbr) improper held at seed value  (element-agnostic)
  +holdCNO_flat     restrained_uff + ALL sp2 improper driven to nearest flat (0/180)      (target-0 variant)
  mmff94s           RDKit MMFF94s, no constraints                          <- the finding's alternative (c)

The improper is UFFAddTorsionConstraint(n0, n1, n2, centre) -- a soft flat-bottomed torsion window that
pins the centre in its 3-neighbour plane. `_HOLD_FC` matches mechanisms._COPLANAR_FC (10 kcal/rad^2).

Also asserts (null-measurement guard) that the +holdC variant genuinely reduces off-plane vs baseline,
and that the seed side is the pre-relax snapshot.

Usage:  uv run python pp_mechanism.py [--gfn2]
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

import rxembed as rx
from rxembed.constraints import mechanisms as _mech
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

from rxembed import geometry as geo  # noqa: E402

XTB = "/home/ali/bin/g-xtb/binaries/xtb-6.7.1/bin/xtb"
_HOLD_FC = 10.0  # kcal/rad^2 — same as mechanisms._COPLANAR_FC (soft: remove gross pucker, don't pin flat)
_HOLD_WIN = 5.0  # deg half-window around the target improper
CASES = ("chb-tetramisole", "bimp", "takemoto-acetone", "schreiner-acetone")

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def _sp2_centres(mol, elements):
    """[(centre, [n0,n1,n2])] for every 3-neighbour sp2 atom of an allowed element."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in elements or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) == 3:  # noqa: PLR2004 — a planar sp2 centre has exactly three neighbours
            out.append((atom.GetIdx(), nbrs))
    return out


def _improper(pos, centre, nbrs):
    return geo._dihedral(pos[nbrs[0]], pos[nbrs[1]], pos[nbrs[2]], pos[centre])


def _nearest_flat(phi):
    return 0.0 if abs(phi) < 90.0 else (180.0 if phi >= 0 else -180.0)  # noqa: PLR2004


def worst_planarity(mol, pos, exclude):
    vals = [geo._plane_offset(pos[c], pos[nb]) for c, nb in _sp2_centres(mol, (6,)) if c not in exclude]
    return max(vals, default=0.0)


def worst_conj(mol, pos, exclude):
    vs = geo.conjugation(mol, pos, exclude=frozenset(exclude))
    return max((v.value for v in vs), default=0.0)


def relax(mol, cons, seed_pos, cid, hold=None):
    """Replicate restrained_uff's organic (no-metal) build from the seed; hold = (elements, mode) or None."""
    m = Chem.Mol(mol)
    conf = m.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, confId=cid, ignoreInterfragInteractions=False)
    for mech in _mech.REGISTRY:
        mech.ff_terms(ff, cons, m.GetConformer(cid), 1e4)
    if hold is not None:
        elements, mode = hold
        for centre, nbrs in _sp2_centres(m, elements):
            phi = rdMolTransforms.GetDihedralDeg(conf, nbrs[0], nbrs[1], nbrs[2], centre)
            target = phi if mode == "seed" else _nearest_flat(phi)
            ff.UFFAddTorsionConstraint(
                nbrs[0], nbrs[1], nbrs[2], centre, False, target - _HOLD_WIN, target + _HOLD_WIN, _HOLD_FC
            )
    ff.Initialize()
    ff.Minimize(maxIts=500)
    return m.GetConformer(cid).GetPositions().copy()


def mmff(mol, seed_pos, cid):
    m = Chem.Mol(mol)
    conf = m.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(m):
        return None
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m, props, confId=cid, ignoreInterfragInteractions=False)
    ff.Minimize(maxIts=1000)
    return m.GetConformer(cid).GetPositions().copy()


def xtb_opt(mol, cid, method):
    with tempfile.TemporaryDirectory() as d:
        Chem.MolToXYZFile(mol, str(Path(d) / "in.xyz"), confId=cid)
        flag = ["--gfnff"] if method == "gfnff" else ["--gfn", "2"]
        subprocess.run(
            [XTB, "in.xyz", *flag, "--opt", "--chrg", str(Chem.GetFormalCharge(mol))],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=600,
        )
        op = Path(d) / "xtbopt.xyz"
        return Chem.MolFromXYZFile(str(op)).GetConformer().GetPositions() if op.exists() else None


def run(case, seed, do_gfn2):
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    _SNAP.clear()
    result = rx.embed(seed=seed, **resolve(entry))
    ens_list = list(result) if isinstance(result, (EnsembleSet, list)) else [result]
    snaps = {id(r["ens"]): r for r in _SNAP}
    ens = ens_list[0]
    snap = snaps[id(ens)]
    mol = ens.mol
    cid = int(next(c for c in ens.ids))
    seed_pos = snap["pre"][cid]
    relax_pos = mol.GetConformer(cid).GetPositions()
    assert not np.array_equal(seed_pos, relax_pos), "seed == relax (vacuous)"
    frozen = {int(f) for f in ens.cons.frozen}
    metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z}
    exclude = frozen | metals

    variants = [
        ("seed", seed_pos),
        ("restrained_uff", relax(mol, ens.cons, seed_pos, cid)),
        ("+holdC_seed", relax(mol, ens.cons, seed_pos, cid, ((6,), "seed"))),
        ("+holdCNO_seed", relax(mol, ens.cons, seed_pos, cid, ((6, 7, 8), "seed"))),
        ("+holdCNO_flat", relax(mol, ens.cons, seed_pos, cid, ((6, 7, 8), "flat"))),
    ]
    mm = mmff(mol, seed_pos, cid)
    if mm is not None:
        variants.append(("mmff94s", mm))
    gf = xtb_opt_from(mol, seed_pos, cid, "gfnff")
    if gf is not None:
        variants.append(("gfnff_opt", gf))
    if do_gfn2:
        g2 = xtb_opt_from(mol, seed_pos, cid, "gfn2")
        if g2 is not None:
            variants.append(("gfn2_opt", g2))

    print(f"\n{'=' * 84}\n{case}  seed={seed}  conf={cid}   (frozen {sorted(frozen)})\n{'=' * 84}")
    print(f"  {'method':16s} {'worst_planar(A)':>16s} {'worst_conj(deg)':>16s}")
    base_plan = None
    for name, pos in variants:
        wp, wc = worst_planarity(mol, pos, exclude), worst_conj(mol, pos, exclude)
        if name == "restrained_uff":
            base_plan = wp
        pflag = " P!" if wp > 0.15 else "   "  # noqa: PLR2004
        cflag = " C!" if wc > 30.0 else "   "  # noqa: PLR2004
        print(f"  {name:16s} {wp:16.3f}{pflag} {wc:16.1f}{cflag}")
    # null-measurement guard: the carbon hold must actually lower planarity vs baseline where baseline fails
    if base_plan is not None and base_plan > 0.15:  # noqa: PLR2004
        held = worst_planarity(mol, dict(variants)["+holdC_seed"], exclude)
        assert held < base_plan, f"hold did not reduce planarity ({held} !< {base_plan})"


def xtb_opt_from(mol, seed_pos, cid, method):
    m = Chem.Mol(mol)
    conf = m.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    return xtb_opt(m, cid, method)


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    do_gfn2 = "--gfn2" in sys.argv
    for case in CASES:
        for seed in (1, 2, 3):
            try:
                run(case, seed, do_gfn2)
            except Exception as e:  # noqa: BLE001
                import traceback

                print(f"\n{case} seed={seed} EXC {type(e).__name__}: {e}")
                traceback.print_exc()
