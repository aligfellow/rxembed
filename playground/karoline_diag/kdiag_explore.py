"""Exploration: reproduce the karoline Ni cases and inspect embed output + geom.check.

Run:  .venv/bin/python playground/karoline_diag/kdiag_explore.py
"""

import sys

import rxembed as rx
from rxembed import geometry as geo

rx.set_verbose("WARNING")

CASES = {
    "case1_thiourea": "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1",
    "case2_pyridylimine_amidate_N": "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)C(c3ccccc3)[N-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C",
    "case3_pyridylimine_amidate_C": "CC1N(Cc2ccccc2)c2cccc[n]2->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[N]=1c1c(C(C)C)cccc1C(C)C",
    "case4_epoxide": "CCOC1=[O]->[Ni+2]2(<-[O-]C(=O)N(c3ccccc3)[CH-]->2c2ccccc2)<-[n]2c[nH]c(C)c21",
}

name = sys.argv[1] if len(sys.argv) > 1 else "case4_epoxide"
smi = CASES[name]
print(f"=== {name} ===\n{smi}\n")

iso_set = rx.metal(smi, "square_planar")
print("isomers:", len(iso_set))
for k, iso in enumerate(iso_set):
    print(f"  [{k}] {iso.summary()}  donors={iso.donors}")

# embed the first isomer, a few conformers, fixed seed
iso = iso_set[0]
print("\nembedding isomer 0 ...")
ens = rx.embed(iso, n=4, seed=0xF00D)
print("type:", type(ens).__name__, "n ids:", len(ens.ids))
donors = list(iso.donors)
print("donors passed to geom.check:", donors)
for cid in ens.ids:
    rep = geo.check(ens.mol, cid, donors=donors)
    kinds = [v.kind for v in rep.violations]
    print(f"  post-embed conf {cid}: ok={rep.ok()} kinds={kinds}")
    for v in rep.violations:
        print("      ", v)
