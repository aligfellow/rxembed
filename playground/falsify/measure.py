"""Tabulate the asymmetry of every CROSS-LINK atom: one bridging >=2 other metal-bonded atoms.

Tier-1 rejects on topology alone; these numbers decide whether an asymmetry guard can spare
the haptic cases (allyl/Cp/eta2) while still dropping the chelate bridges.
"""

import io
import logging
import re

import numpy as np
import xyzgraph

CASES = [  # file, charge, verdict, note
    ("ferrocene.xyz", 0, "KEEP", "eta5 Cp"),
    ("zeise.xyz", -1, "KEEP", "eta2 C=C"),
    ("bisallyl_ni.xyz", 0, "KEEP", "eta3 allyl (OPT)"),
    ("../perception_diag/GODNOD_crystal.xyz", 0, "REJECT", "4-ring chelate"),
    ("../perception_diag/ZOPJOG_crystal.xyz", 0, "REJECT", "5-ring chelate"),
    ("../perception_diag/NUDXOC_crystal.xyz", 0, "REJECT", "4-ring chelate"),
    ("../perception_diag/WELROW_crystal.xyz", 0, "REJECT", "PSe2 bridge (already dropped)"),
]
METALS = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Hf Ta W Re Os Ir Pt Au Hg".split())


def graph_and_conf(path, charge):
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    old, lvl = root.handlers[:], root.level
    root.handlers, _ = [h], root.setLevel(logging.DEBUG)
    try:
        G = xyzgraph.build_graph(path, charge=charge, kekule=True)
    finally:
        root.handlers, _ = old, root.setLevel(lvl)
    conf = {}
    for m in re.finditer(r"Evaluating bond (\D+)(\d+)-(\D+)(\d+) \(d=([\d.]+).*?conf=([\d.]+)\)", buf.getvalue()):
        _, i, _, j, d, c = m.groups()
        conf[frozenset((int(i), int(j)))] = float(c)
    return G, conf


print(
    f"{'structure':22} {'verdict':7} {'bridge':>8} {'flankers':>18} "
    f"{'conf_X':>7} {'conf_fl':>8} {'ratio':>6} {'dX':>6} {'d_fl':>6} {'dd':>6}"
)
print("-" * 116)
for path, chg, verdict, note in CASES:
    try:
        G, conf = graph_and_conf(path, chg)
    except Exception as e:
        print(f"{path:22} ERROR {type(e).__name__}: {e}")
        continue
    pos = {n: np.asarray(d["position"]) for n, d in G.nodes(data=True)}
    sym = {n: d["symbol"] for n, d in G.nodes(data=True)}
    name = path.split("/")[-1].replace("_crystal.xyz", "").replace(".xyz", "")
    for m in [n for n in G if sym[n] in METALS]:
        bonded = set(G.neighbors(m))
        for X in sorted(bonded):
            fl = sorted(set(G.neighbors(X)) & bonded - {m})
            if len(fl) < 2:
                continue
            cX = conf.get(frozenset((m, X)), float("nan"))
            cF = [conf.get(frozenset((m, f)), float("nan")) for f in fl]
            dX = float(np.linalg.norm(pos[m] - pos[X]))
            dF = [float(np.linalg.norm(pos[m] - pos[f])) for f in fl]
            mnF, mdF = float(np.nanmin(cF)), float(np.mean(dF))
            print(
                f"{name:22} {verdict:7} {sym[X] + str(X):>8} "
                f"{','.join(sym[f] + str(f) for f in fl)[:18]:>18} "
                f"{cX:7.2f} {mnF:8.2f} {cX / mnF if mnF else 0:6.2f} "
                f"{dX:6.3f} {mdF:6.3f} {dX - mdF:+6.3f}"
            )
