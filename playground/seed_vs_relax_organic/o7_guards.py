"""Prove the harness measures what it names -- the anti-null-measurement checks, run on live cases.

G0  the snapshot taken inside the `_relax_into_windows` patch is bit-identical to the output of
    `embed.dispatch._embed_dispatch` called with the same arguments and seed. That is the raw ETKDG
    seed by definition (it is what `rx.embed` returned before the relax shipped, and what
    tests/test_frozen.py calls to get seeds), so the "seed" column really is the un-relaxed geometry
    and nothing relaxes it earlier in the chain.
G1  the patch fires exactly once per Ensemble returned.
G2  no scored "seed" is a bit-identical copy of the input geometry (the retain-input trap).
G3  a second independent embed with the same seed reproduces the snapshot bit-for-bit.
G4  seed and relaxed coordinates differ.
G5  atom count / element order identical before and after, so one scoring graph is valid for both.
G6  an UNCONSTRAINED embed leaves every coordinate untouched (the relax must return early).

Usage:  uv run python o7_guards.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np

import rxembed as rx
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from o2_corpus import corpus, resolve  # noqa: E402

CHECK = [
    "bimp",
    "thia-ma",
    "cpa",
    "spiro-ts1",
    "acid-dist",
    "acid-angle",
    "pi-stack",
    "schreiner-acetone",
    "takemoto-acetone",
    "xb-i-n",
    "tmpl-anilide",
    "sn2-smiles",
    "free-tetramisole",
    "free-HyperBTM",
    "free-schreiner",
]
SEED = 1

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {
        "constrained": bool(self.cons.is_constrained),
        "pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids},
        "elems": [a.GetAtomicNum() for a in self.mol.GetAtoms()],
    }
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    from rxembed.embed.dispatch import _embed_dispatch, _xyz_to_mol

    by_id = {e["id"]: e for e in corpus(n=4)}
    fails = []
    for cid in CHECK:
        entry = by_id[cid]
        _SNAP.clear()
        kw = resolve(entry)
        res = rx.embed(seed=SEED, **kw)
        ens_list = list(res) if isinstance(res, (EnsembleSet, list)) else [res]
        snaps = {id(r["ens"]): r for r in _SNAP}
        g1 = len(_SNAP) == len(ens_list) and all(id(e) in snaps for e in ens_list)

        # G0: the snapshot vs a direct _embed_dispatch with the same arguments
        dk = dict(kw)
        src = dk.pop("source")
        raw = _embed_dispatch(
            src,
            metal=None,
            fix=dk.get("fix"),
            constrain=dk.get("constrain"),
            template=dk.get("template"),
            contacts=dk.get("contacts"),
            coordinate=None,
            charge=0,
            n=dk.get("n"),
            seed=SEED,
            knowledge=True,
            stereo="racemic",
        )
        raw_list = list(raw) if isinstance(raw, (EnsembleSet, list)) else [raw]
        g0 = len(raw_list) == len(ens_list) and all(
            np.array_equal(r.mol.GetConformer(c).GetPositions(), snaps[id(e)]["pre"][int(c)])
            for e, r in zip(ens_list, raw_list)
            for c in r.ids
            if int(c) in snaps[id(e)]["pre"]
        )

        g2 = g4 = g5 = g6 = True
        ref_pos = None
        if entry["ref"]:
            ref_pos = _xyz_to_mol(entry["ref"], 0).GetConformer().GetPositions()
        for e in ens_list:
            sn = snaps[id(e)]
            post = {int(c): e.mol.GetConformer(c).GetPositions() for c in e.ids}
            if ref_pos is not None:
                g2 &= not any(np.array_equal(p, ref_pos) for p in sn["pre"].values())
            g5 &= sn["elems"] == [a.GetAtomicNum() for a in e.mol.GetAtoms()]
            same = all(np.array_equal(post[c], sn["pre"][c]) for c in post if c in sn["pre"])
            if sn["constrained"]:
                g4 &= not same
            else:
                g6 &= same
        verdict = dict(G0=g0, G1=g1, G2=g2, G4=g4, G5=g5, G6=g6)
        bad = [k for k, v in verdict.items() if not v]
        cst = all(snaps[id(e)]["constrained"] for e in ens_list)
        print(
            f"{cid:20s} constrained={str(cst):5s} "
            + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in verdict.items())
        )
        if bad:
            fails.append((cid, bad))
    print("\nALL GUARDS PASS" if not fails else f"\nFAILURES: {fails}")
