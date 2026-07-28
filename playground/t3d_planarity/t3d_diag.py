"""Diagnose the `planarity` regression the organic seed-vs-relax study named but did not diagnose.

The study (docs/findings/seed-vs-relax-organic.md) found `planarity` violations rise 15 -> 281 after
`rx.embed()`'s relax, driven by `cpa` (17/20 -> 0/20) and `chb-tetramisole` (40/40 -> 0/20), with
conjugation FLAT at 0 deg -- so this is NOT the C=S/thiourea conjugation mode. This script locates the
tripping atoms/rings, measures seed-vs-relax out-of-plane deviation against the gate threshold, and
compares against a real reference where one exists.

METHOD -- one embed, two geometries (the study's harness, reused)
  `Ensemble._relax_into_windows` is monkeypatched to snapshot each conformer's coords immediately before
  the real relax. Both sides come from ONE `rx.embed()` on the shipped path, paired by conformer id.
  Then geo.check is run on the SAME mol driven to seed / relaxed coords in turn, and every `planarity`
  Violation is printed verbatim (not inferred). For each tripping atom the element / hybridisation /
  neighbours are dumped, and for `cpa` the same atom's off-plane value in the real DFT geometry is shown.

GUARDS
  * seed snapshot is asserted != relaxed (the comparison is non-vacuous).
  * we print the actual Violation objects, so the `planarity` kind is really what fires.
  * `keep_input` copy trap: for the .xyz cases we skip any conformer whose seed == the reference exactly.

Usage:  uv run python t3d_diag.py [seed ...]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

CASES = ("cpa", "chb-tetramisole")
SEEDS = [int(s, 0) for s in sys.argv[1:]] or [1, 2, 3, 7, 0xF00D]

_SNAP: list[dict] = []
_orig = Ensemble._relax_into_windows


def _patched(self):
    rec = {"pre": {int(c): self.mol.GetConformer(c).GetPositions().copy() for c in self.ids}}
    _SNAP.append(rec)
    out = _orig(self)
    rec["ens"] = out
    return out


Ensemble._relax_into_windows = _patched


def _planarity_viols(mol, cid, pos, exclude):
    """Drive `mol`'s conformer to `pos`, return the planarity Violations geo.check would emit."""
    conf = mol.GetConformer(cid)
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, [float(v) for v in xyz])
    return [v for v in geo.planarity(mol, pos, exclude=exclude) if v.kind == "planarity"]


def _atom_desc(mol, i):
    a = mol.GetAtomWithIdx(i)
    nbrs = [(n.GetIdx(), n.GetSymbol()) for n in a.GetNeighbors()]
    return (
        f"{a.GetSymbol()}{i} hyb={str(a.GetHybridization())} arom={a.GetIsAromatic()} "
        f"deg={a.GetDegree()} inRing={a.IsInRing()} nbrs={nbrs}"
    )


def run_case(cid_name, seed):
    entry = next(e for e in corpus(n=4) if e["id"] == cid_name)
    kw = resolve(entry)
    ref_pos = ref_mol = None
    if entry["ref"]:
        from rxembed.embed.dispatch import _xyz_to_mol

        ref_mol = _xyz_to_mol(entry["ref"], 0)
        ref_pos = ref_mol.GetConformer().GetPositions()

    _SNAP.clear()
    result = rx.embed(seed=seed, **kw)
    ens_list = list(result) if isinstance(result, (EnsembleSet, list)) else [result]
    snaps = {id(r["ens"]): r for r in _SNAP}

    print(f"\n{'=' * 90}\n{cid_name}  seed={seed}  cands={len(ens_list)}\n{'=' * 90}")
    for k, ens in enumerate(ens_list):
        snap = snaps.get(id(ens))
        if snap is None:
            continue
        mol = ens.mol
        frozen = set(int(f) for f in ens.cons.frozen)
        metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z}
        exclude = frozenset(frozen | metals)
        pre = snap["pre"]
        # reference off-plane on the offending atoms (cpa only)
        ref_off = {}
        if ref_pos is not None:
            for v in geo.planarity(ref_mol, ref_pos, exclude=exclude):
                # key by the sp2 centre (first atom) or by the ring tuple
                ref_off[v.atoms] = v.value

        for c in ens.ids:
            cd = int(c)
            if cd not in pre:
                continue
            seed_pos = pre[cd]
            relax_pos = mol.GetConformer(cd).GetPositions().copy()
            if ref_pos is not None and np.array_equal(seed_pos, ref_pos):
                print(f"  cand{k} conf{cd}: SKIP (seed == reference, keep_input copy)")
                continue
            assert not np.array_equal(seed_pos, relax_pos), "seed == relax (vacuous)"
            vs = _planarity_viols(mol, cd, seed_pos, exclude)
            vr = _planarity_viols(mol, cd, relax_pos, exclude)
            seed_keys = {v.atoms for v in vs}
            new = [v for v in vr if v.atoms not in seed_keys]
            print(f"  cand{k} conf{cd}: planarity seed={len(vs)} relax={len(vr)}  new={len(new)}")
            # per seed-side value for the atoms that newly trip
            seed_vals = {v.atoms: v.value for v in vs}
            for v in vr:
                mark = "NEW " if v.atoms not in seed_keys else "    "
                sval = seed_vals.get(v.atoms)
                sval_s = f"{sval:.3f}" if sval is not None else "  -  "
                print(
                    f"    {mark}[{v.detail}] atoms={v.atoms}  seed_off={sval_s}  "
                    f"relax_off={v.value:.3f}  thr={v.limit:.2f}"
                )
                # describe the centre atom
                centre = v.atoms[0]
                if "sp2 out of plane" in v.detail:
                    print(f"         centre: {_atom_desc(mol, centre)}")
                    # measure seed-side off explicitly at the same atom even if not a viol
                    off_seed = geo._plane_offset(seed_pos[centre], seed_pos[list(v.atoms[1:])])
                    off_relax = geo._plane_offset(relax_pos[centre], relax_pos[list(v.atoms[1:])])
                    ro = ref_off.get(v.atoms)
                    print(
                        f"         off: seed={off_seed:.3f}  relax={off_relax:.3f}  "
                        f"ref={('%.3f' % ro) if ro is not None else 'n/a'}"
                    )
    return


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    for name in CASES:
        for sd in SEEDS:
            try:
                run_case(name, sd)
            except Exception as e:  # noqa: BLE001
                import traceback

                print(f"\n{name} seed={sd} EXC {type(e).__name__}: {e}")
                traceback.print_exc()
