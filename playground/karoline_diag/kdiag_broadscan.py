"""Broad scan: for each case, any 3-membered ring / connectivity change at post-embed, post-min, post-mc.

Raw xyzgraph triangle detection (catches 1-3 fusions metrics.connectivity's _MIN_TOPO=3 hides).
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter

import networkx as nx
import xyzgraph

import rxembed as rx
from rxembed import metrics as met
from rxembed.constraints import metal as _metal

import kdiag_harness as H

rx.set_verbose("CRITICAL")


def triangles(mol_or_ens, cid):
    p = tempfile.mktemp(suffix=".xyz")
    mol_or_ens.dump(p, align=False) if hasattr(mol_or_ens, "dump") else None
    g = xyzgraph.build_graph(p, quick=True)
    tris = []
    for c in nx.cycle_basis(g):
        if len(c) == 3:
            elems = tuple(sorted(g.nodes[i].get("element", "?") for i in c))
            tris.append((tuple(sorted(c)), elems))
    return tris


def scan(case, nseeds, do_mc):
    smi = H.CASES[case]
    iso_set = rx.metal(smi, "square_planar")
    tri_hist = Counter()
    conn_hist = Counter()
    runs = 0
    for k, iso in enumerate(iso_set):
        for s in range(nseeds):
            seed = 0xF00D + s
            ens = rx.embed(iso, n=2, seed=seed).minimize()
            if do_mc and ens.ids:
                try:
                    ens = ens.mc(preset="transition_metal", seed=seed).minimize()
                except Exception as e:
                    print("  mc failed:", type(e).__name__, e)
            if not ens.ids:
                continue
            metals = set(_metal.metal_indices(ens.mol))
            for cid in ens.ids:
                runs += 1
                for tri, elems in triangles(ens, cid):
                    # only NEW triangles (not aromatic rings): a 3-ring is never aromatic here
                    tri_hist[elems] += 1
                formed, broken = met.connectivity(ens.mol, cid, metals=metals, charge=0)
                if formed or broken:
                    conn_hist[(tuple(formed), tuple(broken))] += 1
    print(f"\n### {case}: {runs} conformers (mc={do_mc}) ###")
    print("  raw 3-membered ring element-sets:", dict(tri_hist) or "NONE")
    print("  metrics.connectivity changes:", dict(conn_hist) or "NONE")


if __name__ == "__main__":
    cases = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] != "all" else list(H.CASES)
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    do_mc = "--mc" in sys.argv
    for c in cases:
        scan(c, nseeds, do_mc)
