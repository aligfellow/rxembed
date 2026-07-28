"""Score one TREE against the 144-structure corpus — the per-tree half of the T3/T4 regression study.

Run once per tree with PYTHONPATH pointed at that tree's src, then diff the JSONs with cr_compare.py.
Nothing here is tree-aware: the SAME script measures all three, so a difference in the output is a
difference in the library, not in the measurement.

What is measured, per structure x seed:
  * embed outcome      — conformer count, and ZERO-CONFORMER returns (silent failures) called out
  * embed-output geometry vs the crystal — M-donor MAE, donor fold, bond-length MAE, heavy RMSD
  * constraint-window satisfaction (distance + angle) of the geometry `embed` actually returns
  * `rx.geometry.check` pass rate AND the per-kind violation census
  * bound-crossover incidence — the smoothing tolerance `bounds._smooth` actually used (T4's target)
  * the chain endpoint after `.minimize(_retry=False)`

Null-measurement guards, each ASSERTED rather than assumed (inherited from playground/seed_vs_relax):
  G1  keep_input prepends the crystal conformer, so ids[0] is bit-identical to the input BY
      CONSTRUCTION. Scoring it scores the crystal against itself. Excluded.
  G2  a structure whose ONLY conformer is that input never embedded — a zero-conformer silent
      failure (TUXRUZ, XAQDUS). Recorded as a FAILURE, never as a perfect score.
  G3  the metal must come back as the real element AND its real oxidation state after restore.
  G4  atom count / element order must match pre and post, so one scoring graph serves both.
  G5  the smoothing spy must be installed in the module embed actually calls (asserted at import).

Usage: PYTHONPATH=<tree>/src uv run --no-sync python cr_measure.py <out.json> [seed ...]
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import sys
import time

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_corpus import CORPUS, crystal_full  # noqa: E402

NCONF = 8  # fixed budget per structure so every structure contributes equally
SEEDS = [0xF00D, 0xBEEF, 0x1234]

# --- G5: crossover spy -------------------------------------------------------------------------
# `bounds._smooth` returns the tolerance it needed; > 0 means the bounds matrix contained a
# contradiction that had to be repaired. This is exactly what T4 claims to relieve.
_bounds = importlib.import_module("rxembed.embed.bounds")
_orig_smooth = _bounds._smooth
TOLS: list[float] = []


def _spy_smooth(bm, max_tol=0.4):
    tol = _orig_smooth(bm, max_tol=max_tol)
    TOLS.append(float(tol) if tol is not None else 0.0)
    return tol


_bounds._smooth = _spy_smooth
assert _bounds._smooth is not _orig_smooth, "smoothing spy not installed"


def set_positions(mol, cid, pos):
    """Overwrite conformer `cid`'s coordinates in place."""
    conf = mol.GetConformer(cid)
    for a in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(a, [float(x) for x in pos[a]])


def score(mol, cid, pos, cry, metal, donors, cons):
    """Every axis for one geometry against the crystal — same axes as playground/seed_vs_relax."""
    set_positions(mol, cid, pos)
    ml = [abs(float(np.linalg.norm(pos[metal] - pos[d])) - float(np.linalg.norm(cry[metal] - cry[d]))) for d in donors]
    fr = geo.donor_fold(mol, cid, donors=set(donors))
    gate = geo.donor_orientation(mol, pos, donors=set(donors))
    bd = [
        abs(
            float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
            - float(np.linalg.norm(cry[b.GetBeginAtomIdx()] - cry[b.GetEndAtomIdx()]))
        )
        for b in mol.GetBonds()
        if metal not in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    ]
    dv = []
    for (i, j), (lo, hi) in cons.distances.items():
        d = float(np.linalg.norm(pos[i] - pos[j]))
        dv.append(max(0.0, lo - d, d - hi))
    av = []
    for (i, j, k), (lo, hi) in cons.angles.items():
        a = geo._angle(pos[i], pos[j], pos[k])
        av.append(max(0.0, lo - a, a - hi))
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    rmsd = geo._kabsch_rmsd(pos[heavy], cry[heavy])

    # the geometry gate + WHICH kind fired
    try:
        rep = geo.check(mol, cid, constraints=cons, donors=set(donors))
        kinds: dict[str, int] = {}
        for v in rep.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        # `ok` is a METHOD on GeometryReport, so `bool(rep.ok)` is a bound method and ALWAYS True —
        # it reported a 100% pass rate next to 2640 recorded violations. Call it.
        gate_ok, gate_kinds = bool(rep.ok()), kinds
    except Exception as e:  # a gate crash is data, not a pass
        gate_ok, gate_kinds = False, {f"EXC:{type(e).__name__}": 1}

    return {
        "ml_mae": float(np.mean(ml)) if ml else None,
        "ml_max": float(np.max(ml)) if ml else None,
        "fold": float(fr.fold),
        "gate_fold_viol": len(gate),
        "bond_mae": float(np.mean(bd)) if bd else None,
        "bond_max": float(np.max(bd)) if bd else None,
        "dwin_mean": float(np.mean(dv)) if dv else 0.0,
        "dwin_max": float(np.max(dv)) if dv else 0.0,
        "dwin_nviol": int(sum(v > 1e-6 for v in dv)),
        "awin_mean": float(np.mean(av)) if av else 0.0,
        "awin_max": float(np.max(av)) if av else 0.0,
        "awin_nviol": int(sum(v > 1e-6 for v in av)),
        "rmsd": float(rmsd),
        "gate_ok": gate_ok,
        "gate_kinds": gate_kinds,
    }


def run(name, path, half, seed):
    """Embed one structure at one seed; return its record or a skip reason."""
    sym, cry, q, q_known = crystal_full(path)
    TOLS.clear()
    t0 = time.time()

    ens = rx.embed(path, charge=q, n=NCONF, seed=seed)
    tols_embed = list(TOLS)
    if isinstance(ens, list) or hasattr(ens, "candidates"):
        return {"name": name, "half": half, "seed": seed, "skip": "EnsembleSet (not the retain-input path)"}
    if ens._metal is None:
        return {"name": name, "half": half, "seed": seed, "skip": "no metal context"}
    mol0 = ens.mol
    if mol0.GetNumAtoms() != len(sym):
        return {"name": name, "half": half, "seed": seed, "skip": f"atom count {mol0.GetNumAtoms()} != {len(sym)}"}

    metal = ens._metal.metal
    donors = sorted(ens._metal.donors or [])
    real_z, real_q = ens._metal.real_z, ens._metal.real_q
    ids = list(ens.ids)

    # G1 / G2
    if not ids:
        return {"name": name, "half": half, "seed": seed, "zero_conf": True, "skip": "ZERO CONFORMERS returned"}
    p0 = mol0.GetConformer(ids[0]).GetPositions()
    input_is_first = bool(np.array_equal(p0, cry))
    seeds_ = ids[1:] if input_is_first else ids
    pre = {c: mol0.GetConformer(c).GetPositions().copy() for c in seeds_}
    identical = [c for c, p in pre.items() if np.array_equal(p, cry)]
    for c in identical:
        pre.pop(c)
    if not pre:
        return {
            "name": name,
            "half": half,
            "seed": seed,
            "zero_conf": True,
            "skip": "ZERO CONFORMERS: output is the input geometry only (silent embed failure)",
        }

    cons = copy.deepcopy(ens.cons)
    elems_pre = [a.GetAtomicNum() for a in mol0.GetAtoms()]
    surrogate_z = int(mol0.GetAtomWithIdx(metal).GetAtomicNum())

    ens.minimize(_retry=False)  # _retry=False keeps the pairing (no fresh re-embedded seeds)
    molr = ens.mol
    post_ids = [c for c in ens.ids if c in pre]

    # G3 / G4
    restored_z = int(molr.GetAtomWithIdx(metal).GetAtomicNum())
    restored_q = int(molr.GetAtomWithIdx(metal).GetFormalCharge())
    elems_post = [a.GetAtomicNum() for a in molr.GetAtoms()]
    same_graph = len(elems_pre) == len(elems_post) and all(
        a == b for i, (a, b) in enumerate(zip(elems_pre, elems_post)) if i != metal
    )

    rec = {
        "name": name,
        "half": half,
        "seed": seed,
        "q": q,
        "q_known": q_known,
        "natoms": len(sym),
        "metal_idx": metal,
        "n_donors": len(donors),
        "surrogate_z": surrogate_z,
        "restored_z": restored_z,
        "restored_q": restored_q,
        "real_z": int(real_z),
        "real_q": int(real_q),
        "charge_ok": bool(restored_z == real_z and restored_q == real_q),
        "input_is_first_conf": input_is_first,
        "n_embed_conf": len(pre),
        "n_survived_minimize": len(post_ids),
        "same_graph": bool(same_graph),
        "n_dist_windows": len(cons.distances),
        "n_angle_windows": len(cons.angles),
        "n_dg_floors": len(cons.dg_floors),
        "tols_embed": tols_embed,
        "tol_max": max(tols_embed) if tols_embed else 0.0,
        "tol_nonzero": int(sum(t > 1e-12 for t in tols_embed)),
        "tol_calls": len(tols_embed),
        "secs": round(time.time() - t0, 1),
        "confs": [],
    }
    if not same_graph or restored_z != real_z:
        rec["skip"] = f"graph/restore check failed (same_graph={same_graph}, z {restored_z} vs {real_z})"
        return rec

    scoring = Chem.Mol(molr)
    # the crystal scored against itself on the same windows — the physical floor for every axis
    ref_cid = (post_ids or list(pre))[0]
    rec["crystal"] = score(scoring, ref_cid, cry, cry, metal, donors, cons)
    # EVERY embedded conformer is scored, whether or not minimize kept it: dropping the ones a tree
    # discards would flatter whichever tree discards more.
    for cid in sorted(pre):
        rec["confs"].append(
            {
                "cid": int(cid),
                "kept_by_minimize": cid in post_ids,
                "embed": score(scoring, cid, pre[cid], cry, metal, donors, cons),
                "post": (
                    score(scoring, cid, molr.GetConformer(cid).GetPositions().copy(), cry, metal, donors, cons)
                    if cid in post_ids
                    else None
                ),
            }
        )
    return rec


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    out_path = sys.argv[1]
    seeds = [int(s, 0) for s in sys.argv[2:]] or SEEDS
    out = []
    for name, path, half in CORPUS:
        for sd in seeds:
            try:
                rec = run(name, path, half, sd)
            except Exception as e:  # a hard failure is data, not a crash
                rec = {"name": name, "half": half, "seed": sd, "error": f"{type(e).__name__}: {e}"}
            out.append(rec)
            tag = rec.get("skip") or rec.get("error")
            print(
                f"{name:24s} {half:8s} s={sd:#x} "
                + (
                    tag
                    if tag
                    else f"n={rec['n_embed_conf']} kept={rec['n_survived_minimize']} "
                    f"tol>0={rec['tol_nonzero']}/{rec['tol_calls']} q_ok={rec['charge_ok']} {rec['secs']}s"
                ),
                flush=True,
            )
            json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)
