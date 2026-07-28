"""T6 edge cases: identical-donor tetrahedral, CN7/CN8 distinct, and worst-case timing.

Calls isomers() directly on hand-built surrogate mols (metal already a carbon surrogate,
bonds to donors stripped) to isolate the orbit-count. Measures, does not assert.
"""

import itertools
import time

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed.constraints import metal as M


def surrogate_star(donor_zs, dirs):
    """Bond-less carbon surrogate metal at origin + monodentate donors along `dirs` (no M-donor bonds).

    Mirrors the state `isomers` sees: the metal is a bond-less C, each donor a lone heavy atom.
    """
    rw = Chem.RWMol()
    me = rw.AddAtom(Chem.Atom(6))  # carbon surrogate
    donors = []
    for z in donor_zs:
        donors.append(rw.AddAtom(Chem.Atom(z)))
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(m.GetNumAtoms())
    conf.SetAtomPosition(me, Point3D(0, 0, 0))
    for d, v in zip(donors, dirs, strict=True):
        p = np.array(v, float)
        p = p / np.linalg.norm(p) * 2.1
        conf.SetAtomPosition(d, Point3D(*p))
    m.AddConformer(conf)
    return m, donors


def count(mol, donors, geometry, r_metal=1.4):
    orbit = list(itertools.permutations(range(len(donors))))
    t0 = time.perf_counter()
    distinct = M.isomers(mol, donors, geometry, perms=orbit, r_metal=r_metal)
    dt = time.perf_counter() - t0
    return len(distinct), len(orbit), dt


cases = [
    ("tetrahedral 4-distinct", [7, 8, 16, 15], "tetrahedral"),
    ("tetrahedral 4-identical", [7, 7, 7, 7], "tetrahedral"),
    ("tetrahedral 2+2", [7, 7, 8, 8], "tetrahedral"),
    ("linear 2-distinct", [7, 8], "linear"),
    ("linear 2-identical", [7, 7], "linear"),
    ("trigonal_planar 3-distinct", [7, 8, 16], "trigonal_planar"),
    ("pentagonal_bipyramidal 7-distinct", [7, 8, 15, 16, 9, 17, 35], "pentagonal_bipyramidal"),
    ("pentagonal_bipyramidal 7-identical", [8] * 7, "pentagonal_bipyramidal"),
    ("square_antiprism 8-distinct", [7, 8, 15, 16, 9, 17, 35, 53], "square_antiprism"),
    ("square_antiprism 8-identical", [8] * 8, "square_antiprism"),
]

for name, zs, geom in cases:
    dirs = M.VERTEX_DIRS[geom]
    mol, donors = surrogate_star(zs, dirs)
    c, o, dt = count(mol, donors, geom, r_metal=M._PT.GetRcovalent(30))
    warn = "WARN" if c > 1 else "silent"
    print(f"  {name:38s} {geom:24s} n={len(donors)} orbit={o:6d}  distinct={c:3d}  {warn:6s}  {dt * 1000:8.1f} ms")
