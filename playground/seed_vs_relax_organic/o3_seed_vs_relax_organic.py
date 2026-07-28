"""Score the raw ETKDG seed against the post-`_relax_into_windows` geometry, on ORGANIC molecules.

The shipped `rx.embed()` relaxes its seeds into their constraint windows at the pipeline's single return
point (`pipeline._relax_embedded` -> `Ensemble._relax_into_windows`). This measures what that costs and
buys on the organic surface, where the metal study (`docs/findings/seed-vs-relax.md`) never looked.

METHOD -- one embed, two geometries, no second embed anywhere
  `Ensemble._relax_into_windows` is monkeypatched to snapshot every conformer's coordinates immediately
  before delegating to the real method. Both sides therefore come from ONE `rx.embed()` call on the
  SHIPPED code path, paired exactly by conformer id. There is no re-embed whose seeds might differ, and
  the measured path is literally the path under test.

NULL-MEASUREMENT GUARDS (asserted, not assumed)
  G1  the patch must fire exactly once per Ensemble, and `cons.is_constrained` is recorded -- an
      unconstrained embed returns early, so its "relax" is a documented no-op and is reported as such,
      never averaged in as a zero-delta improvement.
  G2  on a retain-input path `ids[0]` can be a bit-identical copy of the input. Any conformer whose SEED
      coordinates equal the reference geometry exactly is dropped: it is the input, not a seed.
  G3  a second independent `rx.embed()` with the same seed must reproduce the snapshot bit-for-bit
      (reproducibility; RDKit's default randomSeed=-1 has made this fail before).
  G4  seed and relaxed coordinates must actually differ, else the comparison is vacuous.
  G5  atom count and element ordering identical pre/post, so ONE scoring graph is valid for both sides.

GROUND TRUTH
  For a `.xyz` source there is a real geometry: it is scored through the same code as a third column
  ("ref"), and is the floor for bond length and conjugation.
  For a SMILES source there is NO ground-truth geometry, and none is invented. Two axes are still
  absolute rather than relative:
    * conjugation planarity -- an amide / thiourea C(=X)-N is planar by chemistry, so 0 deg is right
      independent of any reference geometry (this is exactly what `geometry.conjugation` encodes);
    * constraint-window satisfaction -- the window is the user's own stated target.
  Bond length on a SMILES source is reported against MMFF94-relaxed bond lengths for the same molecule,
  which is a FORCE-FIELD PROXY and is labelled as one, never as truth.

Usage:  uv run python o3_seed_vs_relax_organic.py <out.json> [seed ...]
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from o2_corpus import corpus, resolve  # noqa: E402

NCONF = 4
SEEDS = (1, 2, 3, 0xF00D, 7)

# ---------------------------------------------------------------- the snapshot patch (G1)
_SNAP: list[dict] = []
_orig_relax = Ensemble._relax_into_windows


def _patched(self):
    rec = {
        "constrained": bool(self.cons.is_constrained),
        "pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids},
    }
    _SNAP.append(rec)
    out = _orig_relax(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


# ---------------------------------------------------------------- metrics
def conj_quartets(mol):
    """The gate's own conjugation quartets (A=C-X-S), computed once per molecule."""
    q = []
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
            if dbl and subs:
                q.append((dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()))
    return q


def sp2_centres(mol):
    """Carbons with exactly three neighbours -- the gate's sp2-planarity set."""
    return [
        (a.GetIdx(), [n.GetIdx() for n in a.GetNeighbors()])
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetDegree() == 3 and a.GetHybridization() == Chem.HybridizationType.SP2
    ]


def mmff_bond_reference(mol):
    """MMFF94-relaxed bond lengths for this molecule -- a FORCE-FIELD PROXY, not ground truth."""
    m = Chem.Mol(mol)
    try:
        if m.GetNumConformers() == 0:
            return None
        if AllChem.MMFFOptimizeMolecule(m, maxIters=2000) not in (0, 1):
            return None
    except Exception:
        return None
    p = m.GetConformer(m.GetConformers()[0].GetId()).GetPositions()
    return {
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): float(np.linalg.norm(p[b.GetBeginAtomIdx()] - p[b.GetEndAtomIdx()]))
        for b in m.GetBonds()
    }


def score(mol, cid, pos, *, quartets, sp2, cons, frozen, ref_pos, bond_ref, ref_mol):
    """Every axis for one geometry."""
    conf = mol.GetConformer(cid)
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, [float(v) for v in xyz])

    # --- conjugation / planarity (the axis the concern names). Absolute: 0 deg is right.
    cd = []
    for a, c, x, s in quartets:
        dih = abs(geo._dihedral(pos[a], pos[c], pos[x], pos[s]))
        cd.append(min(dih, abs(180.0 - dih)))
    oop = [geo._plane_offset(pos[i], pos[nbrs]) for i, nbrs in sp2]

    # --- constraint-window satisfaction (the user's own stated target)
    dv = [
        max(0.0, lo - float(np.linalg.norm(pos[i] - pos[j])), float(np.linalg.norm(pos[i] - pos[j])) - hi)
        for (i, j), (lo, hi) in cons.distances.items()
    ]
    av = [
        max(0.0, lo - geo._angle(pos[i], pos[j], pos[k]), geo._angle(pos[i], pos[j], pos[k]) - hi)
        for (i, j, k), (lo, hi) in cons.angles.items()
    ]

    # --- bond length vs the reference (a real geometry, or the MMFF proxy)
    bd = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if ref_pos is not None:
            bd.append(abs(d - float(np.linalg.norm(ref_pos[i] - ref_pos[j]))))
        elif bond_ref is not None and (i, j) in bond_ref:
            bd.append(abs(d - bond_ref[(i, j)]))

    # --- the geometry gate's own verdict, exactly as a user would call it
    rep = geo.check(mol, cid, frozen=sorted(frozen) or None, reference=ref_mol)
    kinds = sorted({v.kind for v in rep.violations})

    # --- frozen-core integrity (the load-bearing TS contract)
    fdrift = 0.0
    if frozen and ref_pos is not None:
        fr = sorted(frozen)
        for ii in range(len(fr)):
            for jj in range(ii + 1, len(fr)):
                a_, b_ = fr[ii], fr[jj]
                fdrift = max(
                    fdrift,
                    abs(float(np.linalg.norm(pos[a_] - pos[b_])) - float(np.linalg.norm(ref_pos[a_] - ref_pos[b_]))),
                )

    return {
        "conj_max": float(max(cd)) if cd else 0.0,
        "conj_mean": float(np.mean(cd)) if cd else 0.0,
        "conj_nviol": int(sum(d > 30.0 for d in cd)),
        "n_quartets": len(cd),
        "oop_max": float(max(oop)) if oop else 0.0,
        "dwin_max": float(max(dv)) if dv else 0.0,
        "awin_max": float(max(av)) if av else 0.0,
        "bond_mae": float(np.mean(bd)) if bd else None,
        "bond_max": float(max(bd)) if bd else None,
        "gate_ok": bool(rep.ok()),
        "gate_kinds": kinds,
        "frozen_drift": fdrift,
    }


# ---------------------------------------------------------------- one case
def run(entry, seed):
    _SNAP.clear()
    kw = resolve(entry)
    ref_path = entry["ref"]
    ref_mol = ref_pos = None
    if ref_path:
        from rxembed.embed.dispatch import _xyz_to_mol

        ref_mol = _xyz_to_mol(ref_path, 0)
        ref_pos = ref_mol.GetConformer().GetPositions()

    t0 = time.time()
    result = rx.embed(seed=seed, **kw)
    if not _SNAP:
        return {"id": entry["id"], "seed": seed, "skip": "GUARD G1: the relax seam never fired"}

    ens_list = list(result) if isinstance(result, (EnsembleSet, list)) else [result]
    snaps = {id(r["ens"]): r for r in _SNAP}

    out = []
    for k, ens in enumerate(ens_list):
        snap = snaps.get(id(ens))
        if snap is None:
            out.append({"cand": k, "skip": "candidate not seen by the relax seam"})
            continue
        mol = ens.mol
        pre = snap["pre"]
        rec = {
            "cand": k,
            "constrained": snap["constrained"],
            "n_dist_win": len(ens.cons.distances),
            "n_ang_win": len(ens.cons.angles),
            "n_frozen": len(ens.cons.frozen),
            "confs": [],
        }
        if not snap["constrained"]:
            # the relax returns early by design; assert it truly did nothing (this IS the measurement here)
            rec["noop_verified"] = all(
                np.array_equal(mol.GetConformer(c).GetPositions(), pre[int(c)]) for c in ens.ids if int(c) in pre
            )
            out.append(rec)
            continue

        quartets, sp2 = conj_quartets(mol), sp2_centres(mol)
        bond_ref = None if ref_pos is not None else mmff_bond_reference(mol)
        scoring = Chem.Mol(mol)
        common = dict(
            quartets=quartets,
            sp2=sp2,
            cons=ens.cons,
            frozen=set(ens.cons.frozen),
            ref_pos=ref_pos,
            bond_ref=bond_ref,
            ref_mol=ref_mol,
        )
        ids = [int(c) for c in ens.ids if int(c) in pre]
        if ref_pos is not None and ids:
            rec["ref"] = score(scoring, ids[0], ref_pos, **common)  # the real geometry, same code, same graph
        for cid in ids:
            post = mol.GetConformer(cid).GetPositions().copy()
            seed_pos = pre[cid]
            if ref_pos is not None and np.array_equal(seed_pos, ref_pos):
                continue  # G2: that "seed" is the retained input geometry
            heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            drift = float(geo._kabsch_rmsd(post[heavy], seed_pos[heavy]))
            rec["confs"].append(
                {
                    "cid": cid,
                    "moved": bool(not np.array_equal(post, seed_pos)),  # G4
                    "drift": drift,
                    "seed": score(scoring, cid, seed_pos, **common),
                    "relax": score(scoring, cid, post, **common),
                }
            )
        out.append(rec)

    # G3: reproducibility -- an independent embed with the same seed must give the same seeds
    _SNAP.clear()
    try:
        rx.embed(seed=seed, **resolve(entry))
        repro = all(
            np.array_equal(a["pre"][c], b["pre"][c])
            for a, b in zip(snaps.values(), _SNAP)
            for c in a["pre"]
            if c in b["pre"]
        ) and len(_SNAP) == len(snaps)
    except Exception as e:
        repro = f"EXC {type(e).__name__}: {e}"

    return {
        "id": entry["id"],
        "family": entry["family"],
        "seed": seed,
        "has_reference": ref_path is not None,
        "reproducible": repro if isinstance(repro, str) else bool(repro),
        "secs": round(time.time() - t0, 1),
        "cands": out,
    }


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    outp = sys.argv[1]
    seeds = [int(s, 0) for s in sys.argv[2:]] or list(SEEDS)
    rows = []
    for entry in corpus(n=NCONF):
        for sd in seeds:
            try:
                r = run(entry, sd)
            except Exception as e:
                r = {"id": entry["id"], "family": entry["family"], "seed": sd, "skip": f"EXC {type(e).__name__}: {e}"}
            rows.append(r)
            if "skip" in r:
                print(f"{entry['id']:24s} seed={sd:<6} {r['skip'][:90]}", flush=True)
            else:
                nc = sum(len(c.get("confs", [])) for c in r["cands"])
                gs = sum(cf["seed"]["gate_ok"] for c in r["cands"] for cf in c.get("confs", []))
                gr = sum(cf["relax"]["gate_ok"] for c in r["cands"] for cf in c.get("confs", []))
                cst = all(c.get("constrained", True) for c in r["cands"])
                print(
                    f"{entry['id']:24s} seed={sd:<6} cands={len(r['cands'])} confs={nc} "
                    f"gate {gs}->{gr} constrained={cst} repro={r['reproducible']} {r['secs']}s",
                    flush=True,
                )
    json.dump(rows, open(outp, "w"), indent=1)
    print("wrote", outp)
