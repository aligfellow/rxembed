"""UFF-only T3c spike: does a target-flat torsion cap on the conjugated C=X-N bond clear the twist?

The cap REUSES the metal coplanarity machinery verbatim -- `mechanisms._coplanar_window(phi, cap)` +
`ff.UFFAddTorsionConstraint(...)` at `_COPLANAR_FC` -- with an ORGANIC perception: it walks exactly the
quartets `geometry.conjugation` scores (the a-c-x-s dihedral of a single bond X-C, X in {N,O}, C double-bonded),
so enforcement and gate agree by construction (same discipline as Sp2Planar <-> planarity).

Monkeypatches `rxembed.refine.restrained_uff` (what pipeline.py calls) -- KEEPS UFF, adds the cap to the FF
walk on organic mols only. No MMFF. No src edit. Compares, per conformer through the real pipeline:
`geometry.check(...).ok()`, worst sp2-carbon off-plane, worst conjugation twist.

  BASELINE   shipped code (UFF + Sp2Planar)
  +conjcap20 baseline + conjugation torsion cap, cap=20 deg (inside the 30 deg gate)
  +conjcap30 baseline + conjugation torsion cap, cap=30 deg (== gate)

Cases: bimp / takemoto-acetone / schreiner-acetone (T3c), chb-tetramisole / cpa (T3d -- must not regress).
Seeds 1,2,3.

Usage:  uv run python playground/ff_handling/ffh_t3c_conjcap.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
from o2_corpus import corpus, resolve  # noqa: E402

import rxembed as rx  # noqa: E402
from rxembed import geometry as geo  # noqa: E402
from rxembed import refine as _refine  # noqa: E402
import rxembed.refine.ff as _ffmod  # noqa: E402
from rxembed.constraints import mechanisms as _mech  # noqa: E402

CASES = ("chb-tetramisole", "cpa", "bimp", "takemoto-acetone", "schreiner-acetone")
SEEDS = (1, 2, 3)
_COPLANAR_FC = _mech._COPLANAR_FC
_shipped = _refine.restrained_uff


def conj_quartets(mol):
    """Every (a, c, x, s) quartet geometry.conjugation scores -- the SAME perception the gate uses."""
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
            out.append((dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()))
    return out


def make_patched(cap):
    """restrained_uff clone: UFF everywhere, + an organic conjugation torsion cap (target-flat, width `cap`)."""

    def restrained(mol, cons, distance_fc=500.0, max_iters=500, conf_ids=None):
        from rxembed.constraints.metal import materialise_phantoms
        from rxembed.refine.ff import _bond_pruned, _ff_surrogate

        frozen = set(cons.frozen)
        work = materialise_phantoms(mol, cons.haptic)
        work = _ff_surrogate(work, cons.metals, cons.phantoms)
        organic = not cons.metals and not cons.phantoms and not cons.haptic
        quartets = conj_quartets(work) if organic else []

        def build(target, conf_id):
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target, confId=conf_id, ignoreInterfragInteractions=False)
            conf = target.GetConformer(conf_id)
            for m in _mech.REGISTRY:
                m.ff_terms(ff, cons, conf, distance_fc)
            for a, c, x, s in quartets:  # the T3c cap
                if any(i in frozen for i in (a, c, x, s)):
                    continue
                phi = rdMolTransforms.GetDihedralDeg(conf, a, c, x, s)
                lo, hi = _mech._coplanar_window(phi, cap)
                ff.UFFAddTorsionConstraint(a, c, x, s, False, lo, hi, _COPLANAR_FC)
            ff.Initialize()
            return ff

        def bring_home(src_mol, conf):
            if src_mol is mol:
                return
            src = src_mol.GetConformer(conf.GetId())
            for a in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(a, src.GetAtomPosition(a))

        pruned = None
        energies = []
        confs = mol.GetConformers() if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
        for conf in confs:
            cid = conf.GetId()
            ff = build(work, cid)
            try:
                ff.Minimize(maxIts=max_iters)
                energies.append(ff.CalcEnergy())
                bring_home(work, conf)
            except RuntimeError:
                if pruned is None:
                    pruned = _bond_pruned(work, frozen)
                relaxed = False
                if pruned is not None:
                    try:
                        pff = build(pruned, cid)
                        pff.Minimize(maxIts=max_iters)
                        src = pruned.GetConformer(cid)
                        for a in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(a, src.GetAtomPosition(a))
                        energies.append(pff.CalcEnergy())
                        relaxed = True
                    except RuntimeError:
                        pass
                if not relaxed:
                    try:
                        energies.append(float(ff.CalcEnergy()))
                    except RuntimeError:
                        energies.append(float("nan"))
        return np.array(energies)

    return restrained


def _sp2_carbons(mol):
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) == 3:
            out.append((atom.GetIdx(), nbrs))
    return out


def worst_planar(mol, pos, exclude):
    return max((geo._plane_offset(pos[c], pos[nb]) for c, nb in _sp2_carbons(mol) if c not in exclude), default=0.0)


def worst_conj(mol, pos, exclude):
    return max((v.value for v in geo.conjugation(mol, pos, exclude=frozenset(exclude))), default=0.0)


def measure(case, seed):
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    result = rx.embed(seed=seed, **resolve(entry))
    ens_list = list(result) if not hasattr(result, "cons") else [result]
    clean = tot = 0
    wp = wc = 0.0
    for ens in ens_list:
        mol = ens.mol
        frozen = {int(f) for f in ens.cons.frozen}
        metals = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z}
        exclude = frozen | metals
        for cid in ens.ids:
            cid = int(cid)
            pos = mol.GetConformer(cid).GetPositions()
            tot += 1
            if geo.check(mol, cid).ok():
                clean += 1
            wp = max(wp, worst_planar(mol, pos, exclude))
            wc = max(wc, worst_conj(mol, pos, exclude))
    return clean, tot, wp, wc


def run_all(label, fn):
    _refine.restrained_uff = fn
    _ffmod.restrained_uff = fn
    print(f"\n===== {label} =====")
    print(f"  {'case':20s} {'seed':>4s} {'gate_clean':>11s} {'worstPlan(A)':>13s} {'worstConj(deg)':>15s}")
    ac = at = 0
    for case in CASES:
        for seed in SEEDS:
            c, t, wp, wc = measure(case, seed)
            ac += c
            at += t
            pflag = " P!" if wp > 0.15 else "   "
            cflag = " C!" if wc > 30.0 else "   "
            print(f"  {case:20s} {seed:>4d} {f'{c}/{t}':>11s} {wp:>10.3f}{pflag} {wc:>12.1f}{cflag}")
    print(f"  ----> TOTAL gate-clean {ac}/{at}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    run_all("BASELINE  (shipped: UFF + Sp2Planar)", _shipped)
    run_all("+conjcap20  (UFF + Sp2Planar + conj cap, cap=20 deg)", make_patched(20.0))
    run_all("+conjcap30  (UFF + Sp2Planar + conj cap, cap=30 deg)", make_patched(30.0))
