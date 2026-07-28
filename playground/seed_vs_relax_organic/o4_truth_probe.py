"""Ground-truth probe for the conjugation axis: is the relax's twist real, and is it recoverable?

The organic corpus has no crystal reference for a SMILES input, so the seed/relax comparison on §4's
conjugation axis rests on the chemistry claim that an amide / thiourea C(=X)-N is planar. This script
tests that claim two ways that do NOT depend on any reference geometry:

  (A) REFERENCE CHECK (.xyz cases only). Score the shipped real geometry's own conjugation quartets. If
      the real TS geometry is planar and the relaxed embed is not, the relax is measurably wrong -- not
      merely different.

  (B) REAL-ENERGY CHECK (all cases). Take the SAME conformer's seed and relaxed geometry, and
        1. single-point GFN-FF -- which geometry does a real, conjugation-aware energy prefer?
        2. GFN-FF geometry optimisation -- does the twist survive relaxation to a real minimum, or does
           a downstream `optimize()` repair it?
      (1) says whether the relax moved uphill. (2) says whether the damage is permanent or transient.

Needs a Grimme `xtb` on PATH / $XTB_EXE. Usage:
  uv run python o4_truth_probe.py <out.json> [case ...]
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from o2_corpus import corpus, resolve  # noqa: E402
from o3_seed_vs_relax_organic import conj_quartets  # noqa: E402

CASES = ["bimp", "thia-ma", "cpa", "schreiner-acetone", "takemoto-acetone", "chb-tetramisole"]
SEED = 1
NCONF = 4

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def conj_max(mol, quartets, pos):
    ds = []
    for a, c, x, s in quartets:
        d = abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s]))
        ds.append(min(d, abs(180.0 - d)))
    return float(max(ds)) if ds else 0.0


def one(entry):
    _SNAP.clear()
    kw = resolve(entry)
    res = rx.embed(seed=SEED, **kw)
    ens = res[0] if isinstance(res, list) else res
    snap = next(r for r in _SNAP if r["ens"] is ens)
    mol, pre = ens.mol, snap["pre"]
    quartets = conj_quartets(mol)
    ids = [int(c) for c in ens.ids if int(c) in pre]

    out = {"id": entry["id"], "n_quartets": len(quartets), "confs": []}

    # (A) the shipped real geometry, if one exists
    if entry["ref"]:
        from rxembed.embed.dispatch import _xyz_to_mol

        rp = _xyz_to_mol(entry["ref"], 0).GetConformer().GetPositions()
        out["ref_conj_max"] = conj_max(mol, quartets, rp)

    # (B) real energy: build a 2-conformer Mol per pair (seed, relax) and let rxembed score/optimise it
    for cid in ids:
        post = mol.GetConformer(cid).GetPositions().copy()
        seed_pos = pre[cid]
        rec = {
            "cid": cid,
            "seed_conj": conj_max(mol, quartets, seed_pos),
            "relax_conj": conj_max(mol, quartets, post),
        }
        for label, pos in (("seed", seed_pos), ("relax", post)):
            m = Chem.Mol(mol)
            m.RemoveAllConformers()
            conf = Chem.Conformer(mol.GetNumAtoms())
            for a, xyz in enumerate(pos):
                conf.SetAtomPosition(a, [float(v) for v in xyz])
            cid2 = m.AddConformer(conf, assignId=True)
            e = rx.wrap(m, [cid2], minimized=True)
            try:
                sp = e.score("gfnff")
                rec[f"{label}_E"] = float(next(iter(sp.energies.values())))
            except Exception as ex:
                rec[f"{label}_E"] = f"EXC {type(ex).__name__}"
            try:
                op = rx.wrap(Chem.Mol(m), [cid2], minimized=True).optimize("gfnff")
                op_pos = op.mol.GetConformer(op.ids[0]).GetPositions()
                rec[f"{label}_conj_after_opt"] = conj_max(mol, quartets, op_pos)
                rec[f"{label}_opt_rmsd"] = float(
                    geo._kabsch_rmsd(
                        op_pos[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]],
                        pos[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]],
                    )
                )
            except Exception as ex:
                rec[f"{label}_conj_after_opt"] = f"EXC {type(ex).__name__}: {str(ex)[:60]}"
        out["confs"].append(rec)
    return out


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    outp = sys.argv[1]
    want = sys.argv[2:] or CASES
    by_id = {e["id"]: e for e in corpus(n=NCONF)}
    rows = []
    for cid in want:
        try:
            r = one(by_id[cid])
        except Exception as ex:
            r = {"id": cid, "skip": f"EXC {type(ex).__name__}: {ex}"}
        rows.append(r)
        print(json.dumps(r)[:400], flush=True)
        if "ref_conj_max" in r:
            print(f"  {cid}: REAL geometry conj_max = {r['ref_conj_max']:.1f} deg", flush=True)
        for c in r.get("confs", []):
            es, er = c.get("seed_E"), c.get("relax_E")
            # `Ensemble.score` stores kcal/mol already (pipeline.py:1050) -- no Hartree conversion here
            de = (er - es) if isinstance(es, float) and isinstance(er, float) else None
            print(
                f"  cid {c['cid']}: conj {c['seed_conj']:.1f} -> {c['relax_conj']:.1f} deg | "
                f"GFN-FF E(relax)-E(seed) = {de if de is None else round(de, 1)} kcal/mol | "
                f"after opt: seed {c.get('seed_conj_after_opt')} relax {c.get('relax_conj_after_opt')}",
                flush=True,
            )
    json.dump(rows, open(outp, "w"), indent=1)
    print("wrote", outp)
