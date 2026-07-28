"""Decisive test: is the planarity pyramidalisation a UFF defect (real) or a gate false positive?

For the two named centres -- cpa C73 (sp2 C, 3 carbons) and chb-tetramisole C3 (the isothiourea
S-C(=N)-N amidine carbon) -- take the PLANAR ETKDG seed and relax it several ways, measuring the sp2
out-of-plane offset in each:

  seed            the raw ETKDG seed (what embed starts from)
  uff_plain       RDKit UFFGetMoleculeForceField, NO constraints  -> is bare UFF the culprit?
  mmff            RDKit MMFF94s, NO constraints                    -> does a proper FF keep it planar?
  restrained      the pipeline's refine.restrained_uff w/ the real constraints -> the shipped relax
  gfnff           xtb --gfnff --opt  (real energy)                 -> ground truth
  gfn2            xtb --gfn 2 --opt  (real energy)                 -> ground truth

Plus: cpa has a real DFT reference geometry -- its C73 off-plane is measured directly (true ground truth).
And each centre is described chemically (element/hyb/neighbours + a SMILES-ish environment).

Logic:
  * uff_plain pyramidalises but mmff/gfn stay planar  => UFF term defect, REAL geometric defect, gate right
  * gfn ALSO pyramidalises to ~0.20                    => the flat sp2 is not the real minimum, gate FALSE +ve
  * restrained >> uff_plain                            => the constraint walls (flat-bottomed) add the pucker

Usage:  uv run python t3d_ff_compare.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

import rxembed as rx
from rxembed import geometry as geo
from rxembed import refine as _refine
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

XTB = "/home/ali/bin/g-xtb/binaries/xtb-6.7.1/bin/xtb"

# (case, centre atom, its three sp2 neighbours) -- from t3d_diag.py
TARGETS = {"cpa": (73, [72, 74, 76]), "chb-tetramisole": (3, [2, 4, 7])}

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def off_plane(pos, centre, nbrs):
    return geo._plane_offset(pos[centre], pos[list(nbrs)])


def pyr_angle(pos, centre, nbrs):
    """Improper pyramidalisation: 90 - angle(centre->plane-normal). 0 = planar. In degrees."""
    n = np.cross(pos[nbrs[1]] - pos[nbrs[0]], pos[nbrs[2]] - pos[nbrs[0]])
    n /= np.linalg.norm(n) + 1e-12
    # mean bond direction dotted with normal
    devs = []
    for j in nbrs:
        v = pos[centre] - pos[j]
        v /= np.linalg.norm(v) + 1e-12
        devs.append(abs(np.degrees(np.arcsin(np.clip(np.dot(v, n), -1, 1)))))
    return float(np.mean(devs))


def xtb_opt(mol, cid, method):
    """Optimise conformer `cid` with xtb `method` ('gfnff' or 'gfn2'); return relaxed positions or None."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "in.xyz"
        Chem.MolToXYZFile(mol, str(p), confId=cid)
        chg = Chem.GetFormalCharge(mol)
        flag = ["--gfnff"] if method == "gfnff" else ["--gfn", "2"]
        cmd = [XTB, "in.xyz", *flag, "--opt", "tight", "--chrg", str(chg)]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=600)
        opt = Path(d) / "xtbopt.xyz"
        if not opt.exists():
            print(f"      xtb {method} FAILED rc={r.returncode}: {r.stderr.strip()[-200:]}")
            return None
        m2 = Chem.MolFromXYZFile(str(opt))
        return m2.GetConformer().GetPositions()


def describe(mol, centre):
    a = mol.GetAtomWithIdx(centre)
    env = []
    for n in a.GetNeighbors():
        b = mol.GetBondBetweenAtoms(centre, n.GetIdx())
        env.append(f"{n.GetSymbol()}{n.GetIdx()}({b.GetBondTypeAsDouble()},{'ar' if n.GetIsAromatic() else 'al'})")
    return (
        f"C{centre}: hyb={a.GetHybridization()} arom={a.GetIsAromatic()} inRing={a.IsInRing()} "
        f"charge={a.GetFormalCharge()} conj_nbrbonds={[b.GetIsConjugated() for b in a.GetBonds()]}\n"
        f"       bonds: {env}"
    )


def run(case, seed=1, do_gfn2=True):
    centre, nbrs = TARGETS[case]
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    kw = resolve(entry)
    ref_pos = None
    if entry["ref"]:
        from rxembed.embed.dispatch import _xyz_to_mol

        rm = _xyz_to_mol(entry["ref"], 0)
        ref_pos = rm.GetConformer().GetPositions()

    _SNAP.clear()
    result = rx.embed(seed=seed, **kw)
    ens_list = list(result) if isinstance(result, (EnsembleSet, list)) else [result]
    snaps = {id(r["ens"]): r for r in _SNAP}
    ens = ens_list[0]
    snap = snaps[id(ens)]
    mol = ens.mol
    cid = int(next(c for c in ens.ids))
    seed_pos = snap["pre"][cid]

    print(f"\n{'=' * 92}\n{case}  seed={seed}  centre atom {centre}\n{'=' * 92}")
    print("  " + describe(mol, centre))
    if ref_pos is not None:
        print(
            f"  DFT REFERENCE off-plane at C{centre}: {off_plane(ref_pos, centre, nbrs):.3f} A  "
            f"(pyr {pyr_angle(ref_pos, centre, nbrs):.1f} deg)   [ground truth]"
        )

    rows = []
    rows.append(("seed", seed_pos))

    # relaxed shipped geometry (already computed by embed)
    rows.append(("embed_relax", mol.GetConformer(cid).GetPositions().copy()))

    # plain UFF, no constraints
    m_uff = Chem.Mol(mol)
    c = m_uff.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        c.SetAtomPosition(i, [float(v) for v in xyz])
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m_uff, confId=cid, ignoreInterfragInteractions=False)
    ff.Minimize(maxIts=1000)
    rows.append(("uff_plain", m_uff.GetConformer(cid).GetPositions().copy()))

    # MMFF94s, no constraints
    m_mmff = Chem.Mol(mol)
    c = m_mmff.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        c.SetAtomPosition(i, [float(v) for v in xyz])
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(m_mmff):
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m_mmff, mmffVariant="MMFF94s")
        mff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            m_mmff, props, confId=cid, ignoreInterfragInteractions=False
        )
        mff.Minimize(maxIts=1000)
        rows.append(("mmff94s", m_mmff.GetConformer(cid).GetPositions().copy()))
    else:
        print("  (MMFF has no params for this molecule)")

    # the pipeline's restrained relax with the REAL constraints, from the seed
    m_res = Chem.Mol(mol)
    c = m_res.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        c.SetAtomPosition(i, [float(v) for v in xyz])
    _refine.restrained_uff(m_res, ens.cons, distance_fc=1e4, conf_ids=[cid])
    rows.append(("restrained_uff", m_res.GetConformer(cid).GetPositions().copy()))

    # xtb ground truth from the seed
    m_x = Chem.Mol(mol)
    c = m_x.GetConformer(cid)
    for i, xyz in enumerate(seed_pos):
        c.SetAtomPosition(i, [float(v) for v in xyz])
    gff = xtb_opt(m_x, cid, "gfnff")
    if gff is not None:
        rows.append(("gfnff_opt", gff))
    if do_gfn2:
        g2 = xtb_opt(m_x, cid, "gfn2")
        if g2 is not None:
            rows.append(("gfn2_opt", g2))

    print(f"\n  {'method':16s} {'off-plane(A)':>12s} {'pyr(deg)':>9s}   thr=0.15 A")
    for name, pos in rows:
        o = off_plane(pos, centre, nbrs)
        flag = "  <-- FLAGGED" if o > 0.15 else ""
        print(f"  {name:16s} {o:12.3f} {pyr_angle(pos, centre, nbrs):9.1f}{flag}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    for case in ("chb-tetramisole", "cpa"):
        run(case, seed=1)
