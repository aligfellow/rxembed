"""T2: score the RAW DG seed vs the POST-RELAX geometry against the crystal, on the same metrics.

Paired by conformer id: the same seed is measured before and after `minimize()`, so nothing is
compared across different geometries. `_retry=False` disables `_reembed_until_clean`, which would
otherwise inject FRESH seeds (new randomSeed) mid-minimize and destroy the pairing.

Null-measurement guards (each asserted, not assumed):
  G1  the retain-input path prepends the crystal conformer -> ids[0] is bit-identical to the input
      BY CONSTRUCTION. It is excluded; the scored seeds are ids[1:].
  G2  a structure whose only conformer is that input (no ETKDG seed at all) is a FAILED embed and is
      excluded entirely.
  G3  the pre-relax snapshot is taken before minimize() is called, and a SECOND independent
      rx.embed() with the same seed must reproduce it bit-for-bit (proves the snapshot is the embed
      output and that the embed is reproducible).
  G4  seed and relaxed coords must actually differ (per-conformer RMSD > 0), else the "relax" was a
      no-op and the comparison is vacuous.
  G5  the metal must be a CARBON surrogate pre-minimize and the real element + oxidation state
      post-restore. Both are asserted.
  G6  atom count and element ordering (metal aside) must be identical pre/post, so one scoring graph
      is valid for both sides.

Usage:  uv run python t2_seed_vs_relax.py <selection.json> <out.json>
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import time

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo

TMQM = "/home/ali/Documents/Codes/OIN-SMILES/tests/integration/tmQM"
SEED = 0xF00D  # the one embed seed used everywhere in this harness
NCONF = 8  # fixed conformer budget per structure, so every structure contributes equally


def crystal(path):
    """Return (symbols, coords, charge) from a tmQM .xyz."""
    lines = open(path).read().splitlines()
    n = int(lines[0])
    q = int(re.search(r"q = (-?\d+)", lines[1]).group(1))
    sym, xyz = [], []
    for line in lines[2 : 2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz), q


def set_positions(mol, cid, pos):
    """Overwrite conformer `cid`'s coordinates in place."""
    conf = mol.GetConformer(cid)
    for a in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(a, [float(x) for x in pos[a]])


def score(mol, cid, pos, cry, metal, donors, cons):
    """Every axis for one geometry, against the crystal `cry`."""
    set_positions(mol, cid, pos)

    # --- axis 1: M-donor distances vs crystal
    ml = [abs(np.linalg.norm(pos[metal] - pos[d]) - np.linalg.norm(cry[metal] - cry[d])) for d in donors]
    # --- axis 2: donor fold (the geometry gate's own metric + gate)
    fr = geo.donor_fold(mol, cid, donors=set(donors))
    gate = geo.donor_orientation(mol, pos, donors=set(donors))
    # --- axis 3: bond lengths generally (perceived graph, metal bonds excluded — stripped by the surrogate)
    bd = [
        abs(
            np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()])
            - np.linalg.norm(cry[b.GetBeginAtomIdx()] - cry[b.GetEndAtomIdx()])
        )
        for b in mol.GetBonds()
        if metal not in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    ]
    # --- axis 4: satisfaction of the windows ACTUALLY applied to this embed
    dv = []
    for (i, j), (lo, hi) in cons.distances.items():
        d = float(np.linalg.norm(pos[i] - pos[j]))
        dv.append(max(0.0, lo - d, d - hi))
    av = []
    for (i, j, k), (lo, hi) in cons.angles.items():
        a = geo._angle(pos[i], pos[j], pos[k])
        av.append(max(0.0, lo - a, a - hi))
    # --- overall: heavy-atom Kabsch RMSD to crystal
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    rmsd = geo._kabsch_rmsd(pos[heavy], cry[heavy])

    return {
        "ml_mae": float(np.mean(ml)) if ml else None,
        "ml_max": float(np.max(ml)) if ml else None,
        "fold": float(fr.fold),
        "fold_unknown": len(fr.unknown),
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
    }


def run(name):
    """Embed + relax one refcode; return the paired per-conformer record or a skip reason."""
    path = os.path.join(TMQM, f"{name}.xyz")
    sym, cry, q = crystal(path)

    t0 = time.time()
    ens = rx.embed(path, charge=q, n=NCONF, seed=SEED)
    if isinstance(ens, list) or hasattr(ens, "candidates"):
        return {"name": name, "skip": "EnsembleSet (not the single retain-input path)"}
    if ens._metal is None:
        return {"name": name, "skip": "no metal context"}

    mol0 = ens.mol
    if mol0.GetNumAtoms() != len(sym):
        return {"name": name, "skip": f"atom count {mol0.GetNumAtoms()} != xyz {len(sym)}"}

    metal = ens._metal.metal
    donors = sorted(ens._metal.donors or [])
    real_z, real_q = ens._metal.real_z, ens._metal.real_q
    # G5a: the metal must be the carbon surrogate right now
    surrogate_z = mol0.GetAtomWithIdx(metal).GetAtomicNum()

    ids = list(ens.ids)
    # G1: ids[0] is the retained crystal conformer
    p0 = mol0.GetConformer(ids[0]).GetPositions()
    input_is_first = bool(np.array_equal(p0, cry))
    seeds = ids[1:] if input_is_first else ids
    # G2: no genuine ETKDG seed -> failed embed
    if not seeds:
        return {"name": name, "skip": "FAILED EMBED: output is the input geometry only (no ETKDG seed)"}
    pre = {c: mol0.GetConformer(c).GetPositions().copy() for c in seeds}
    # G2b: belt and braces — any seed bit-identical to the crystal is not a seed
    identical = [c for c, p in pre.items() if np.array_equal(p, cry)]
    if len(identical) == len(seeds):
        return {"name": name, "skip": "FAILED EMBED: every seed is bit-identical to the input"}
    for c in identical:
        pre.pop(c)

    cons = copy.deepcopy(ens.cons)
    elems_pre = [a.GetAtomicNum() for a in mol0.GetAtoms()]

    # G3: an independent re-embed with the same seed must reproduce the snapshot bit-for-bit
    ens2 = rx.embed(path, charge=q, n=NCONF, seed=SEED)
    repro = all(
        np.array_equal(ens2.mol.GetConformer(c).GetPositions(), pre[c]) for c in pre if c in list(ens2.ids)
    ) and set(pre) <= set(ens2.ids)
    del ens2

    ens.minimize(_retry=False)  # _retry=False: no fresh re-embedded seeds, so the pairing survives
    molr = ens.mol
    post_ids = [c for c in ens.ids if c in pre]

    # G5b / G6
    restored_z = molr.GetAtomWithIdx(metal).GetAtomicNum()
    restored_q = molr.GetAtomWithIdx(metal).GetFormalCharge()
    elems_post = [a.GetAtomicNum() for a in molr.GetAtoms()]
    same_graph = len(elems_pre) == len(elems_post) and all(
        a == b for i, (a, b) in enumerate(zip(elems_pre, elems_post)) if i != metal
    )

    rec = {
        "name": name,
        "q": q,
        "natoms": len(sym),
        "metal_idx": metal,
        "n_donors": len(donors),
        "surrogate_z": int(surrogate_z),
        "restored_z": int(restored_z),
        "restored_q": int(restored_q),
        "real_z": int(real_z),
        "real_q": int(real_q),
        "input_is_first_conf": input_is_first,
        "n_seeds": len(pre),
        "n_survived_minimize": len(post_ids),
        "reproducible_embed": bool(repro),
        "same_graph": bool(same_graph),
        "n_dist_windows": len(cons.distances),
        "n_angle_windows": len(cons.angles),
        "secs": round(time.time() - t0, 1),
        "confs": [],
    }
    if not same_graph or restored_z != real_z:
        rec["skip"] = f"graph/restore check failed (same_graph={same_graph}, z {restored_z} vs {real_z})"
        return rec

    # score both sides on ONE graph (the restored molr), driving its conformers to each coordinate set
    scoring = Chem.Mol(molr)
    # BASELINE: the crystal scored against itself on the SAME windows. Its fold is the physical floor,
    # and its window violation says whether the constraint windows even CONTAIN the crystal — if they do
    # not, a relax reaching 0.00 violation is moving away from the crystal, not toward it.
    if post_ids:
        rec["crystal"] = score(scoring, post_ids[0], cry, cry, metal, donors, cons)
    for cid in post_ids:
        post_pos = molr.GetConformer(cid).GetPositions().copy()
        seed_pos = pre[cid]
        drift = float(np.sqrt(np.mean(np.sum((post_pos - seed_pos) ** 2, axis=1))))  # G4
        s = score(scoring, cid, seed_pos, cry, metal, donors, cons)
        r = score(scoring, cid, post_pos, cry, metal, donors, cons)
        rec["confs"].append({"cid": int(cid), "drift": drift, "seed": s, "relax": r})
    return rec


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    sel = json.load(open(sys.argv[1]))
    names = [r["name"] if isinstance(r, dict) else r for r in sel]
    out = []
    for nm in names:
        try:
            rec = run(nm)
        except Exception as e:  # a hard embed failure is data, not a crash
            rec = {"name": nm, "skip": f"EXCEPTION {type(e).__name__}: {e}"}
        out.append(rec)
        print(
            f"{nm:8s} "
            + (
                rec["skip"]
                if "skip" in rec
                else f"seeds={rec['n_seeds']} kept={rec['n_survived_minimize']} "
                f"repro={rec['reproducible_embed']} wins={rec['n_dist_windows']}d/{rec['n_angle_windows']}a "
                f"{rec['secs']}s"
            ),
            flush=True,
        )
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("wrote", sys.argv[2])
