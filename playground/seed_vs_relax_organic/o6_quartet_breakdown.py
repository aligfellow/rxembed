"""Which conjugated fragment does the relax twist? Classify every quartet the relax pushes past the gate.

The concern names a thiourea H-N-C=S specifically, and reports that "amides and small thioureas are
clean". If the harm really is C=S-specific the answer is to narrow the relax's torsion handling; if the
same twist appears on C=O amides at comparable rates the answer is a general gate. This resolves that.

A quartet is labelled by (double-bond element, X element, substituent element, ring/acyclic context and
whether X carries another aryl). Reported as seed-side and relax-side deviation per class.

Usage:  uv run python o6_quartet_breakdown.py <out.json> [seed ...]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from o2_corpus import corpus, resolve  # noqa: E402

NCONF = 4
_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {
        "constrained": bool(self.cons.is_constrained),
        "pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids},
    }
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched
_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def labelled_quartets(mol):
    """(indices, label) for every conjugation quartet the gate would examine."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != 6:
                continue
            dbl = [
                nb
                for nb in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [nb for nb in x_atom.GetNeighbors() if nb.GetIdx() != c_atom.GetIdx()]
            if not dbl or not subs:
                continue
            a, x, c, s = dbl[0].GetIdx(), x_atom.GetIdx(), c_atom.GetIdx(), subs[0].GetIdx()
            # is the C the carbon of a (thio)urea -- i.e. does it bear a SECOND N?
            other_n = sum(1 for nb in c_atom.GetNeighbors() if nb.GetAtomicNum() == 7) >= 2
            aryl_x = any(nb.GetIsAromatic() for nb in x_atom.GetNeighbors())
            motif = (
                "thiourea"
                if (other_n and dbl[0].GetAtomicNum() == 16)
                else "urea"
                if other_n
                else "thioamide"
                if dbl[0].GetAtomicNum() == 16
                else "amide/ester/enamine"
            )
            lbl = (
                f"{motif} | {_SYM.get(dbl[0].GetAtomicNum(), '?')}={_SYM.get(c_atom.GetAtomicNum(), '?')}-"
                f"{_SYM.get(x_atom.GetAtomicNum(), '?')}-{_SYM.get(mol.GetAtomWithIdx(s).GetAtomicNum(), '?')}"
                f"{' | N-aryl' if aryl_x else ''}"
            )
            out.append(((a, c, x, s), lbl))
    return out


def dev(pos, q):
    a, c, x, s = q
    d = abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s]))
    return min(d, abs(180.0 - d))


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    outp = sys.argv[1]
    seeds = [int(s, 0) for s in sys.argv[2:]] or [1, 2, 3]
    acc = defaultdict(lambda: {"seed": [], "relax": [], "cases": set()})
    for entry in corpus(n=NCONF):
        for sd in seeds:
            _SNAP.clear()
            try:
                res = rx.embed(seed=sd, **resolve(entry))
            except Exception as e:
                print(f"{entry['id']} seed={sd}: EXC {type(e).__name__}", flush=True)
                continue
            ens_list = list(res) if isinstance(res, (EnsembleSet, list)) else [res]
            snaps = {id(r["ens"]): r for r in _SNAP}
            for ens in ens_list:
                snap = snaps.get(id(ens))
                if snap is None or not snap["constrained"]:
                    continue
                lq = labelled_quartets(ens.mol)
                for cid in [int(c) for c in ens.ids if int(c) in snap["pre"]]:
                    post = ens.mol.GetConformer(cid).GetPositions()
                    for q, lbl in lq:
                        acc[lbl]["seed"].append(dev(snap["pre"][cid], q))
                        acc[lbl]["relax"].append(dev(post, q))
                        acc[lbl]["cases"].add(entry["id"])
        print(f"done {entry['id']}", flush=True)

    print(f"\n{'quartet class':52s} {'n':>6} {'seed':>8} {'relax':>8} {'delta':>8} {'>30 seed':>9} {'>30 relax':>10}")
    rows = []
    for lbl, v in sorted(acc.items(), key=lambda kv: -(np.mean(kv[1]["relax"]) - np.mean(kv[1]["seed"]))):
        s, r = np.array(v["seed"]), np.array(v["relax"])
        rows.append(
            {
                "class": lbl,
                "n": len(s),
                "seed_mean": float(s.mean()),
                "relax_mean": float(r.mean()),
                "seed_over30": int((s > 30).sum()),
                "relax_over30": int((r > 30).sum()),
                "seed_max": float(s.max()),
                "relax_max": float(r.max()),
                "cases": sorted(v["cases"]),
            }
        )
        print(
            f"{lbl:52s} {len(s):6d} {s.mean():8.1f} {r.mean():8.1f} {r.mean() - s.mean():+8.1f} "
            f"{(s > 30).sum():9d} {(r > 30).sum():10d}"
        )
    json.dump(rows, open(outp, "w"), indent=1)
    print("wrote", outp)
