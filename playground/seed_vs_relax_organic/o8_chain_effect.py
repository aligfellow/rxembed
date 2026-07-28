"""Does the embed relax change what the documented CHAIN produces, or only what `embed` returns?

`o3` measures `rx.embed()`'s own output. But the documented pipeline is `embed -> (mc) -> minimize`,
and `minimize` runs the same restrained UFF. If `minimize` lands in the same place either way, the
embed relax is a no-op for anyone who follows the chain and the regression is confined to the raw
`embed()` output -- which the notebooks and `geometry.check` DO inspect, but which is not the end state.

Three geometries per conformer, all from ONE embed so the pairing is exact:
  A  SEED      -- the raw ETKDG geometry (snapshot from inside the relax seam)
  B  EMBED     -- what `rx.embed` returns today (seed -> `_relax_into_windows`)
  C  B.min     -- B put through `minimize()` (the chain as it runs today)
  D  A.min     -- the SEED put through `minimize()` with `_seeds_relaxed` cleared, i.e. exactly the
                  chain as it ran BEFORE the embed relax shipped

C vs D is the question. `_retry=False` disables `_reembed_until_clean` so no fresh seed is substituted
mid-minimize and the A/B/C/D pairing survives (the same argument the metal study used).

Usage:  uv run python o8_chain_effect.py <out.json> [seed ...]
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
from o3_seed_vs_relax_organic import conj_quartets  # noqa: E402

NCONF = 4
SKIP = {"bimp-smiles-auto"}  # 32 candidates x 4 confs x 2 minimises -- excluded for runtime, not for result

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


def conj_max(quartets, pos):
    ds = [
        min(
            abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s])),
            abs(180.0 - abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s]))),
        )
        for a, c, x, s in quartets
    ]
    return float(max(ds)) if ds else 0.0


def place(mol, cid, pos):
    conf = mol.GetConformer(cid)
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, [float(v) for v in xyz])


def run(entry, sd):
    _SNAP.clear()
    kw = resolve(entry)
    ref_mol = None
    if entry["ref"]:
        from rxembed.embed.dispatch import _xyz_to_mol

        ref_mol = _xyz_to_mol(entry["ref"], 0)

    res = rx.embed(seed=sd, **kw)
    ens = res[0] if isinstance(res, (EnsembleSet, list)) else res
    snap = next(r for r in _SNAP if r["ens"] is ens)
    if not snap["constrained"]:
        return None
    quartets = conj_quartets(ens.mol)
    frozen = sorted(ens.cons.frozen) or None
    pre = snap["pre"]
    ids = [int(c) for c in ens.ids if int(c) in pre]
    embed_pos = {c: ens.mol.GetConformer(c).GetPositions().copy() for c in ids}

    # C: the chain as it runs TODAY
    ens.minimize(_retry=False)
    c_pos = {int(c): ens.mol.GetConformer(c).GetPositions().copy() for c in ens.ids if int(c) in pre}

    # D: the chain as it ran BEFORE the embed relax shipped -- same Ensemble, seeds restored,
    #    `_seeds_relaxed`/`_minimized` cleared so minimize genuinely relaxes rather than single-points
    ens2 = rx.embed(seed=sd, **resolve(entry))
    ens2 = ens2[0] if isinstance(ens2, (EnsembleSet, list)) else ens2
    for c in ids:
        if c in [int(x) for x in ens2.ids]:
            place(ens2.mol, c, pre[c])
    ens2._seeds_relaxed = False
    ens2._minimized = False
    ens2.minimize(_retry=False)
    d_pos = {int(c): ens2.mol.GetConformer(c).GetPositions().copy() for c in ens2.ids if int(c) in pre}

    scoring = Chem.Mol(ens.mol)
    out = []
    for c in ids:
        row = {"cid": c}
        for label, src, mol in (
            ("A_seed", pre, ens.mol),
            ("B_embed", embed_pos, ens.mol),
            ("C_embed_min", c_pos, ens.mol),
            ("D_seed_min", d_pos, ens2.mol),
        ):
            if c not in src:
                row[label] = None  # dropped by minimize
                continue
            place(scoring, c, src[c])
            rep = geo.check(scoring, c, frozen=frozen, reference=ref_mol)
            row[label] = {
                "gate_ok": bool(rep.ok()),
                "kinds": sorted({v.kind for v in rep.violations}),
                "conj_max": conj_max(quartets, src[c]),
            }
        out.append(row)
    return {
        "id": entry["id"],
        "family": entry["family"],
        "seed": sd,
        "n_seeds": len(ids),
        "n_C": len(c_pos),
        "n_D": len(d_pos),
        "confs": out,
    }


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    outp = sys.argv[1]
    seeds = [int(s, 0) for s in sys.argv[2:]] or [1, 2, 3]
    rows = []
    for entry in corpus(n=NCONF):
        if entry["id"] in SKIP:
            continue
        for sd in seeds:
            try:
                r = run(entry, sd)
            except Exception as e:
                r = {"id": entry["id"], "seed": sd, "skip": f"EXC {type(e).__name__}: {e}"}
            if r is None:
                continue
            rows.append(r)
            if "skip" in r:
                print(f"{entry['id']:24s} seed={sd:<6} {r['skip'][:80]}", flush=True)
            else:
                g = {
                    k: sum(1 for cf in r["confs"] if cf[k] and cf[k]["gate_ok"])
                    for k in ("A_seed", "B_embed", "C_embed_min", "D_seed_min")
                }
                print(
                    f"{entry['id']:24s} seed={sd:<6} n={r['n_seeds']} gate "
                    f"A(seed)={g['A_seed']} B(embed)={g['B_embed']} C(embed.min)={g['C_embed_min']} "
                    f"D(seed.min)={g['D_seed_min']} | survived C={r['n_C']} D={r['n_D']}",
                    flush=True,
                )
    json.dump(rows, open(outp, "w"), indent=1)

    tot = defaultdict(int)
    conj = defaultdict(list)
    for r in rows:
        if "skip" in r:
            continue
        for cf in r["confs"]:
            for k in ("A_seed", "B_embed", "C_embed_min", "D_seed_min"):
                if cf[k]:
                    tot[k] += cf[k]["gate_ok"]
                    tot[k + "_n"] += 1
                    conj[k].append(cf[k]["conj_max"])
    print("\n=== pooled ===")
    for k in ("A_seed", "B_embed", "C_embed_min", "D_seed_min"):
        print(
            f"  {k:14s} gate clean {tot[k]:4d}/{tot[k + '_n']:<4d} ({100 * tot[k] / max(tot[k + '_n'], 1):5.1f}%)  "
            f"conj_max mean {np.mean(conj[k]):5.1f} deg"
        )
    print("wrote", outp)
