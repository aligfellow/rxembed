"""Karoline Ni square-planar diagnosis harness.

Measures, per case/isomer/conformer:
  - geom.check violation kinds on the RESTORED metal (post-embed-relax and post-minimize)
  - connectivity diff (formed/broken bonds) vs the input graph, via metrics.connectivity
Fixed seeds, several conformers, metal oxidation restored before every check.

Usage:
  .venv/bin/python playground/karoline_diag/kdiag_harness.py <case> [n] [--ablate <what>]
    case  = case1|case2|case3|case4|all
    what  = none|seam|sp2|conj|caps|relief   (monkeypatch ablation)
"""

from __future__ import annotations

import argparse
from collections import Counter

from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed import metrics as met
from rxembed.constraints import metal as _metal

rx.set_verbose("ERROR")

CASES = {
    "case1": "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    "case2": "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C",
    "case3": "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C",
    "case4": "CCOC1=[O]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[n]2c[nH]c(C)c21",
}


def restored_mol(mol, mc):
    """Copy `mol` and restore the real metal element+charge (so geom.check sees a metal, not the surrogate)."""
    m = Chem.Mol(mol)
    if mc is not None:
        _metal.restore(m, mc.metal, mc.real_z, mc.real_q)
        for mi, rz, rq in mc.extra:
            _metal.restore(m, mi, rz, rq)
    return m


def elements_override(mc):
    """{idx: real_z} so metrics.connectivity treats a still-surrogate metal as its real element."""
    if mc is None:
        return None
    z = {mc.metal: mc.real_z}
    z.update({mi: rz for mi, rz, _ in mc.extra})
    return z


# ---- ablation monkeypatches (all reverted at process exit; we never touch src/) ----------
def apply_ablation(what):
    from rxembed.constraints import mechanisms as mech

    from rxembed import pipeline as pipe

    orig = {}
    if what in ("seam",):
        orig["seam"] = pipe.Ensemble._relax_into_windows
        pipe.Ensemble._relax_into_windows = lambda self: self  # return raw seeds
    if what in ("sp2", "caps"):
        orig["sp2"] = mech.Sp2Planar.ff_terms
        mech.Sp2Planar.ff_terms = lambda self, ff, cons, conf, fc: None
    if what in ("conj", "caps"):
        orig["conj"] = mech.ConjugationCap.ff_terms
        mech.ConjugationCap.ff_terms = lambda self, ff, cons, conf, fc: None
    if what in ("relief",):
        orig["relief"] = mech.Floor.dg_relief
        mech.Floor.dg_relief = lambda self, cons, ctx: None  # neutralise the phantom-floor relief
    return orig


def report_conf(mol_r, cid, donors, elements, ref_mol):
    rep = geo.check(mol_r, cid, donors=donors)
    kinds = Counter(v.kind for v in rep.violations)
    formed, broken = met.connectivity(
        ref_mol, cid, metals=set(_metal.metal_indices(mol_r)) | set(elements or {}), charge=0, elements=elements
    )
    return rep.ok(), kinds, formed, broken, rep.violations


def run_case(case, n, ablate):
    smi = CASES[case]
    print(f"\n########## {case}  (ablate={ablate}) ##########")
    print(smi)
    iso_set = rx.metal(smi, "square_planar")
    print(f"isomers: {len(iso_set)}")

    total = Counter()
    flagged = 0
    nconf = 0
    for k, iso in enumerate(iso_set):
        donors = list(iso.donors)
        ens = rx.embed(iso, n=n, seed=0xF00D)
        mc = ens._metal
        elements = elements_override(mc)
        mol_r = restored_mol(ens.mol, mc)
        print(f"\n[iso {k}] {iso.summary()} donors={donors}  embedded={len(ens.ids)}")
        for cid in ens.ids:
            ok, kinds, formed, broken, viols = report_conf(mol_r, cid, donors, elements, ens.mol)
            nconf += 1
            if not ok or formed or broken:
                flagged += 1
            total.update(kinds)
            cc = ""
            if formed:
                cc += f" FORMED={formed}"
            if broken:
                cc += f" BROKEN={broken}"
            print(f"   conf {cid}: geom_ok={ok} kinds={dict(kinds)}{cc}")
            if formed or broken:
                for v in viols:
                    print(f"        {v}")
    print(f"\n== {case} summary: {flagged}/{nconf} conformers flagged (geom or connectivity)")
    print(f"   violation-kind histogram: {dict(total)}")
    return flagged, nconf, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--ablate", default="none")
    args = ap.parse_args()
    if args.ablate != "none":
        apply_ablation(args.ablate)
    cases = list(CASES) if args.case == "all" else [args.case]
    grand = Counter()
    gf = gn = 0
    for c in cases:
        f, nn, t = run_case(c, args.n, args.ablate)
        gf += f
        gn += nn
        grand.update(t)
    print(f"\n===== GRAND: {gf}/{gn} flagged; kinds {dict(grand)} =====")
