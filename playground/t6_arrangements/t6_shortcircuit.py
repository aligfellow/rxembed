"""T6: short-circuit variant — stop at the 2nd distinct arrangement. Measures cost vs full scan.

Reuses the exact per-permutation signature computation from isomers(), stopping the moment a
second distinct arrangement appears (all we need to decide the warning).
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


def admits_multiple(mol, donors, geometry, r_metal, haptic=None, cap=2):
    """True iff >1 distinct arrangement — reuses isomers()'s signature, short-circuits at `cap` distinct."""
    dirs = M.VERTEX_DIRS.get(geometry)
    if dirs is None:
        return False
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != M.VACANT else "X") for d in donors}
    frag = _frag_map(mol)
    dmat = Chem.GetDistanceMatrix(mol)
    real_donors = [d for d in donors if d != M.VACANT]
    bm = _span_bounds(mol)
    from rxembed.geometry import _stripped_hybridisation

    hyb = _stripped_hybridisation(mol)

    def link(od, p, q):
        a, b = M._vertex_atom(haptic, od[p]), M._vertex_atom(haptic, od[q])
        if M.VACANT in (od[p], od[q]) or frag[a] != frag[b]:
            return -1
        return int(dmat[a][b])

    n = len(donors)
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    seen = set()
    scanned = 0
    for order in itertools.permutations(range(n)):
        scanned += 1
        od = [donors[k] for k in order]
        if _central_trans(od, frag, dmat, dirs, haptic):
            continue
        if not _chelate_span_ok(mol, od, frag, dirs, bm, r_metal, hyb, real_donors, haptic):
            continue
        sig = tuple(
            sorted(
                (tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), _vertex_angle(dirs[p], dirs[q]))
                for p, q in pairs
            )
        )
        sig = (sig, chirality_of(mol, real_donors, geometry, od, haptic))
        seen.add(sig)
        if len(seen) >= cap:
            return True, scanned
    return len(seen) > 1, scanned


cases = [
    ("tetrahedral 4-distinct", [7, 8, 16, 15], "tetrahedral"),
    ("tetrahedral 4-identical", [7, 7, 7, 7], "tetrahedral"),
    ("linear 2-distinct", [7, 8], "linear"),
    ("pentagonal_bipyramidal 7-distinct", [7, 8, 15, 16, 9, 17, 35], "pentagonal_bipyramidal"),
    ("pentagonal_bipyramidal 7-identical", [8] * 7, "pentagonal_bipyramidal"),
    ("square_antiprism 8-distinct", [7, 8, 15, 16, 9, 17, 35, 53], "square_antiprism"),
    ("square_antiprism 8-identical", [8] * 8, "square_antiprism"),
]

for name, zs, geom in cases:
    dirs = M.VERTEX_DIRS[geom]
    mol, donors = surrogate_star(zs, dirs)
    t0 = time.perf_counter()
    warn, scanned = admits_multiple(mol, donors, geom, M._PT.GetRcovalent(30))
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {name:38s} {geom:24s} n={len(donors)}  warn={str(warn):5s}  scanned={scanned:6d}  {dt:9.2f} ms")
