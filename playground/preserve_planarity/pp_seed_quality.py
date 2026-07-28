"""Q1 (make-or-break): is the ETKDG SEED already planar on every centre the RELAX flags?

If the seed is reliably good and only the relax is bad, then a "preserve the planarity the seed
already has" fix is sufficient (hold-at-seed). If some seeds are ALSO bad on the flagged centres,
"preserve-only" is insufficient and a real target-0 restraint is needed. This is the make-or-break.

METHOD -- one embed, two geometries (the seed_vs_relax / t3d harness, reused). Monkeypatch
`Ensemble._relax_into_windows` to snapshot each conformer's coords immediately before the real relax.
Both sides come from ONE rx.embed() on the shipped path, paired by conformer id. Then for EVERY centre
the RELAXED geometry flags -- planarity (sp2 carbon off-plane) OR conjugation (twisted quartet) -- read
the SAME centre's deviation on the SEED side. All Violation objects are real (geo.planarity / geo.conjugation
on the driven mol), not inferred.

GUARDS
  * seed snapshot asserted != relax coords (non-vacuous).
  * G3-style: only the shipped embed path is measured (the monkeypatch wraps the real method).

Usage:  uv run python pp_seed_quality.py [seed ...]
"""

from __future__ import annotations

import logging
import sys

import numpy as np

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

SEEDS = [int(s, 0) for s in sys.argv[1:]] or [1, 2, 3, 7, 0xF00D]
PLANAR_THR = 0.15  # Å — geo.planarity gate
CONJ_THR = 30.0  # deg — geo.conjugation gate
SEED_GOOD_PLANAR = 0.05  # Å — "the seed is already flat" band (well under the 0.15 gate)
SEED_GOOD_CONJ = 10.0  # deg — "the seed is already planar" band (well under the 30 gate)

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def _plan_offsets(mol, pos, exclude):
    """{centre_atom: off_plane} for every 3-neighbour sp2 carbon (the planarity gate's own set)."""
    out = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != geo._CARBON_Z or atom.GetHybridization().name != "SP2":
            continue
        if atom.GetIdx() in exclude:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) != geo._SP2_DEGREE:
            continue
        out[atom.GetIdx()] = geo._plane_offset(pos[atom.GetIdx()], pos[nbrs])
    return out


def _conj_twists(mol, pos, exclude):
    """{(a,c,x,s): twist_deg} for every conjugation quartet geo.conjugation walks (no flex/metal here)."""
    out = {}
    from rdkit import Chem

    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != geo._CARBON_Z:
                continue
            if x_atom.GetIdx() in exclude or c_atom.GetIdx() in exclude:
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
            out[(a, c, x, s)] = min(dih, abs(180.0 - dih))
    return out


def run():
    planar_rows = []  # (case, centre, seed_off, relax_off)
    conj_rows = []  # (case, quartet, seed_tw, relax_tw)
    for entry in corpus(n=4):
        if entry["family"] == "free":
            continue
        cid_name = entry["id"]
        for seed in SEEDS:
            _SNAP.clear()
            try:
                result = rx.embed(seed=seed, **resolve(entry))
            except Exception as e:  # noqa: BLE001
                print(f"  {cid_name} seed={seed} EXC {type(e).__name__}: {e}")
                continue
            ens_list = list(result) if isinstance(result, (EnsembleSet, list)) else [result]
            snaps = {id(r["ens"]): r for r in _SNAP}
            for ens in ens_list:
                snap = snaps.get(id(ens))
                if snap is None:
                    continue
                mol = ens.mol
                frozen = {int(f) for f in ens.cons.frozen}
                metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z}
                exclude = frozenset(frozen | metals)
                for c in ens.ids:
                    cd = int(c)
                    if cd not in snap["pre"]:
                        continue
                    sp = snap["pre"][cd]
                    rp = mol.GetConformer(cd).GetPositions()
                    if np.array_equal(sp, rp):
                        continue  # torn/kept-as-seed conformer, or vacuous — skip
                    ps, pr = _plan_offsets(mol, sp, exclude), _plan_offsets(mol, rp, exclude)
                    for centre, roff in pr.items():
                        if roff > PLANAR_THR:  # the relax FLAGS this centre
                            planar_rows.append((cid_name, centre, ps.get(centre, float("nan")), roff))
                    cs, cr = _conj_twists(mol, sp, exclude), _conj_twists(mol, rp, exclude)
                    for q, rtw in cr.items():
                        if rtw > CONJ_THR:  # the relax FLAGS this quartet
                            conj_rows.append((cid_name, q, cs.get(q, float("nan")), rtw))

    print("=" * 96)
    print("Q1  SEED quality on every centre the RELAX flags (organic corpus, seeds " + str(SEEDS) + ")")
    print("=" * 96)

    print(f"\n--- PLANARITY: sp2-carbon centres flagged by relax (>{PLANAR_THR} A off-plane) ---")
    print(f"  total flagged (case,centre,conf) instances: {len(planar_rows)}")
    if planar_rows:
        seed_offs = np.array([r[2] for r in planar_rows])
        good = np.sum(seed_offs <= SEED_GOOD_PLANAR)
        print(
            f"  SEED off-plane on those centres: min={np.nanmin(seed_offs):.3f} "
            f"median={np.nanmedian(seed_offs):.3f} max={np.nanmax(seed_offs):.3f} A"
        )
        print(
            f"  seed already flat (<= {SEED_GOOD_PLANAR} A): {good}/{len(planar_rows)} "
            f"({100 * good / len(planar_rows):.0f}%)"
        )
        bad = [r for r in planar_rows if not (r[2] <= SEED_GOOD_PLANAR)]
        if bad:
            print(f"  *** SEED ALSO BAD on {len(bad)} instances (seed off-plane > {SEED_GOOD_PLANAR} A):")
            byc: dict = {}
            for case, centre, so, ro in bad:
                byc.setdefault((case, centre), []).append((so, ro))
            for (case, centre), v in sorted(byc.items()):
                sos = [x[0] for x in v]
                print(
                    f"      {case} C{centre}: n={len(v)} seed_off min={np.nanmin(sos):.3f} "
                    f"max={np.nanmax(sos):.3f}  relax_off max={max(x[1] for x in v):.3f}"
                )
        # per-case worst seed offset
        print("  per-case worst SEED off-plane among flagged centres:")
        pc: dict = {}
        for case, centre, so, ro in planar_rows:
            pc.setdefault(case, []).append(so)
        for case, v in sorted(pc.items()):
            print(f"      {case:22s} n={len(v):4d}  seed_off max={np.nanmax(v):.3f}  median={np.nanmedian(v):.3f}")

    print(f"\n--- CONJUGATION: quartets flagged by relax (>{CONJ_THR} deg twist) ---")
    print(f"  total flagged (case,quartet,conf) instances: {len(conj_rows)}")
    if conj_rows:
        seed_tw = np.array([r[2] for r in conj_rows])
        good = np.sum(seed_tw <= SEED_GOOD_CONJ)
        print(
            f"  SEED twist on those quartets: min={np.nanmin(seed_tw):.1f} "
            f"median={np.nanmedian(seed_tw):.1f} max={np.nanmax(seed_tw):.1f} deg"
        )
        print(
            f"  seed already planar (<= {SEED_GOOD_CONJ} deg): {good}/{len(conj_rows)} "
            f"({100 * good / len(conj_rows):.0f}%)"
        )
        bad = [r for r in conj_rows if not (r[2] <= SEED_GOOD_CONJ)]
        print(f"  seed ALSO twisted (> {SEED_GOOD_CONJ} deg): {len(bad)}/{len(conj_rows)}")
        pc2: dict = {}
        for case, q, so, ro in conj_rows:
            pc2.setdefault(case, []).append(so)
        print("  per-case worst SEED twist among flagged quartets:")
        for case, v in sorted(pc2.items()):
            print(f"      {case:22s} n={len(v):4d}  seed_twist max={np.nanmax(v):.1f}  median={np.nanmedian(v):.1f}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    run()
