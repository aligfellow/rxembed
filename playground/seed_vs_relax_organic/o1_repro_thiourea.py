"""Reproduce (or refute) the named thiourea claim from tests/test_frozen.py:60.

Claim: on `examples/structures/bimp.xyz` with fix=[10,11,12,14], n=4, the geometry gate goes
seed 4/4, 3/4, 4/4 -> embed 1/4, 0/4, 2/4 over seeds 1-3, because the relax twists H-N-C=S 32-68 deg.

Method: monkeypatch `Ensemble._relax_into_windows` to snapshot the PRE-relax coordinates of every
conformer id, then delegate to the real method. So both sides come from ONE `rx.embed()` call on the
shipped code path -- there is no second embed whose seeds might differ, and the path under test is
literally the one being measured (no null measurement).
"""

from __future__ import annotations

import logging
import sys

import numpy as np

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble

PATH = "examples/structures/bimp.xyz"
CORE = [10, 11, 12, 14]

_SNAP: dict[int, dict] = {}
_orig = Ensemble._relax_into_windows


def _patched(self):
    key = id(self)
    _SNAP[key] = {
        "constrained": bool(self.cons.is_constrained),
        "pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids},
    }
    out = _orig(self)
    _SNAP[key]["post_ids"] = [int(c) for c in out.ids]
    return out


Ensemble._relax_into_windows = _patched


def conj_devs(mol, pos):
    """Every conjugation-quartet deviation from planarity (deg), as the gate computes them."""
    from rdkit import Chem

    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != 6:
                continue
            dbl = [
                n
                for n in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [n for n in x_atom.GetNeighbors() if n.GetIdx() != c_atom.GetIdx()]
            if not dbl or not subs:
                continue
            a, x, c, s = dbl[0].GetIdx(), x_atom.GetIdx(), c_atom.GetIdx(), subs[0].GetIdx()
            dih = abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s]))
            out.append(((a, c, x, s), min(dih, abs(180.0 - dih))))
    return out


def run(seed):
    _SNAP.clear()
    from rxembed.embed.dispatch import _xyz_to_mol

    ref = _xyz_to_mol(PATH, 0)
    ens = rx.embed(PATH, fix=CORE, n=4, seed=seed)
    assert len(_SNAP) == 1, f"expected exactly one relax call, got {len(_SNAP)}"
    snap = next(iter(_SNAP.values()))
    assert snap["constrained"], "GUARD FAILED: cons.is_constrained is False -> the relax never ran"

    mol = ens.mol
    pre = snap["pre"]
    rows = []
    for cid in ens.ids:
        cid = int(cid)
        post = mol.GetConformer(cid).GetPositions().copy()
        moved = not np.array_equal(post, pre[cid])
        # score BOTH sides on the SAME mol by driving its conformer coordinates
        conf = mol.GetConformer(cid)

        def at(p):
            for a, xyz in enumerate(p):
                conf.SetAtomPosition(a, [float(v) for v in xyz])
            rep = geo.check(mol, cid, frozen=CORE, reference=ref)
            cd = conj_devs(mol, p)
            return {
                "ok": rep.ok(),
                "kinds": sorted({v.kind for v in rep.violations}),
                "conj_max": max((d for _, d in cd), default=0.0),
                "conj_n_over_30": sum(d > 30.0 for _, d in cd),
            }

        s = at(pre[cid])
        r = at(post)
        at(post)  # leave the ensemble as it was
        rows.append((cid, moved, s, r))
    return rows


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    for seed in (1, 2, 3):
        rows = run(seed)
        sok = sum(r[2]["ok"] for r in rows)
        rok = sum(r[3]["ok"] for r in rows)
        print(f"\nseed={seed}:  gate clean  seed {sok}/{len(rows)}  ->  embed(relaxed) {rok}/{len(rows)}")
        for cid, moved, s, r in rows:
            print(
                f"   cid {cid}  moved={moved}  conj_max {s['conj_max']:6.1f} -> {r['conj_max']:6.1f} deg  "
                f"| seed {'OK ' if s['ok'] else 'FAIL'} {s['kinds']}  -> relax {'OK ' if r['ok'] else 'FAIL'} {r['kinds']}"
            )
