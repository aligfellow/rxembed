"""Analyse the maintainer's produced k*.xyz outputs: formula, 3-membered rings (epoxide), donor geometry."""

import glob
import os

import networkx as nx
import xyzgraph

for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "k*.xyz"))):
    name = os.path.basename(path)
    with open(path) as f:
        n = int(f.readline().split()[0])
    g = xyzgraph.build_graph(path, quick=True)
    # count elements
    elems = {}
    for _, d in g.nodes(data=True):
        e = d.get("element") or d.get("symbol") or d.get("Z")
        elems[e] = elems.get(e, 0) + 1
    tri = [c for c in nx.cycle_basis(g) if len(c) == 3]
    print(f"\n{name}: {n} atoms, elements={elems}")
    print(f"   3-membered rings: {len(tri)}")
    for c in tri:
        syms = [(i, g.nodes[i].get("element")) for i in c]
        print(f"      triangle {syms}")
