"""Confirm the Sp2Planar SPIKE clears T3d through the REAL rx.embed pipeline (gate pass rate)."""

import logging, sys

sys.path.insert(0, "/home/ali/Documents/Codes/rxembed/playground/seed_vs_relax_organic")
logging.disable(logging.WARNING)
import rxembed as rx
from rxembed import geometry as geo
from rxembed.pipeline import Ensemble, EnsembleSet
from o2_corpus import corpus, resolve

for case in ("chb-tetramisole", "bimp", "cpa", "takemoto-acetone"):
    entry = next(e for e in corpus(n=4) if e["id"] == case)
    for seed in (1, 2, 3):
        res = rx.embed(seed=seed, **resolve(entry))
        ens_list = list(res) if isinstance(res, (EnsembleSet, list)) else [res]
        clean = tot = 0
        plan = conj = 0
        for ens in ens_list:
            frozen = [int(f) for f in ens.cons.frozen] or None
            for c in ens.ids:
                tot += 1
                rep = geo.check(ens.mol, int(c), frozen=frozen)
                if rep.ok():
                    clean += 1
                plan += sum(v.kind == "planarity" for v in rep.violations)
                conj += sum(v.kind == "conjugation" for v in rep.violations)
        print(f"  {case:18s} seed={seed}  gate_clean {clean}/{tot}   planarity_viol={plan}  conjugation_viol={conj}")
