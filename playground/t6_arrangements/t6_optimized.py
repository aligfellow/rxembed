"""T6: dedup-scan with (a) precomputed fixed per-position angles, (b) short-circuit at 2 distinct.

The vertex angle at position-pair (p,q) is constant (independent of which donor sits there), so it is
precomputed once. Measures whether this alone makes the all-identical silent high-CN scan acceptable,
avoiding any donor-multiset special-case.
"""

import itertools
import time

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed.constraints import metal as M
from rxembed.constraints.metal import (
    _central_trans,
    _chelate_span_ok,
    _frag_map,
    _span_bounds,
    _vertex_angle,
    chirality_of,
)


def surrogate_star(donor_zs, dirs):
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    donors = [rw.AddAtom(Chem.Atom(z)) for z in donor_zs]
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0, 0, 0))
    for d, v in zip(donors, dirs, strict=True):
        p = np.array(v, float)
        p = p / np.linalg.norm(p) * 2.1
        conf.SetAtomPosition(d, Point3D(*p))
    m.AddConformer(conf)
    return m, donors


def count(mol, donors, geometry, r_metal, haptic=None, limit=None):
    dirs = M.VERTEX_DIRS[geometry]
    n = len(donors)
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != M.VACANT else "X") for d in donors}
    frag = _frag_map(mol)
    dmat = Chem.GetDistanceMatrix(mol)
    real_donors = [d for d in donors if d != M.VACANT]
    bm = _span_bounds(mol)
    from rxembed.geometry import _stripped_hybridisation

    hyb = _stripped_hybridisation(mol)
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    ang = {(p, q): _vertex_angle(dirs[p], dirs[q]) for p, q in pairs}  # FIXED per position-pair

    def link(od, p, q):
        a, b = M._vertex_atom(haptic, od[p]), M._vertex_atom(haptic, od[q])
        if M.VACANT in (od[p], od[q]) or frag[a] != frag[b]:
            return -1
        return int(dmat[a][b])

    seen = set()
    scanned = 0
    for order in itertools.permutations(range(n)):
        scanned += 1
        od = [donors[k] for k in order]
        if _central_trans(od, frag, dmat, dirs, haptic):
            continue
        if not _chelate_span_ok(mol, od, frag, dirs, bm, r_metal, hyb, real_donors, haptic):
            continue
        sig = tuple(sorted((tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), ang[(p, q)]) for p, q in pairs))
        sig = (sig, chirality_of(mol, real_donors, geometry, od, haptic))
        seen.add(sig)
        if limit and len(seen) >= limit:
            break
    return len(seen), scanned


cases = [
    ("tetrahedral 4-distinct", [7, 8, 16, 15], "tetrahedral"),
    ("tetrahedral 4-identical", [7, 7, 7, 7], "tetrahedral"),
    ("linear 2-distinct", [7, 8], "linear"),
    ("pentagonal_bipyramidal 7-distinct", [7, 8, 15, 16, 9, 17, 35], "pentagonal_bipyramidal"),
    ("pentagonal_bipyramidal 7-identical(MoCl7)", [17] * 7, "pentagonal_bipyramidal"),
    ("square_antiprism 8-distinct", [7, 8, 15, 16, 9, 17, 35, 53], "square_antiprism"),
    ("square_antiprism 8-identical", [9] * 8, "square_antiprism"),
]

print("== short-circuit at 2 (warn decision) ==")
for name, zs, geom in cases:
    mol, donors = surrogate_star(zs, M.VERTEX_DIRS[geom])
    t0 = time.perf_counter()
    c, scanned = count(mol, donors, geom, M._PT.GetRcovalent(30), limit=2)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {name:42s} n={len(donors)} warn={str(c > 1):5s} scanned={scanned:6d} {dt:9.2f} ms")
