"""The 144-structure corpus, both halves, with the charge parser each half needs.

Two different .xyz comment conventions live here:
  tmQM half (103)      ``CSD_code = ABEZAJ | q = 0 | S = 0 | ...``
  fixtures half (41)   ``Refcode: ABUVUP_comp_0 | ... | Charge: 0 | ...`` or
                       ``Optimized by MACE GPU, Charge: 0, Multiplicity: 1``
and the fixtures half also writes ``Charge: unknown`` on some entries. An unknown charge is
recorded as such (``q_known=False``) and defaulted to 0, so a verdict can be re-read without it.

Borane/carborane cages are EXCLUDED by maintainer instruction: UFF has no ``B_5``/``B_6`` atom
types, so these fail for a parameterisation reason unrelated to either change under test.
"""

from __future__ import annotations

import os
import re

import numpy as np

TMQM = "/home/ali/Documents/Codes/OIN-SMILES/tests/integration/tmQM"
FIXTURES = "/home/ali/Documents/Codes/OIN-SMILES/tests/fixtures"

# Out of scope: UFF lacks B_5/B_6 types. Detected structurally below as well (>=5 borons), so a
# cage that is not on this list is still caught and reported rather than silently scored.
BORANE_NAMES = {"WIMCAA"}


def _parse_charge(comment):
    """Return (charge, known) from either half's comment convention."""
    m = re.search(r"\bq\s*=\s*(-?\d+)", comment)
    if m:
        return int(m.group(1)), True
    m = re.search(r"Charge:\s*(-?\d+)", comment)
    if m:
        return int(m.group(1)), True
    return 0, False


def crystal(path):
    """Return (symbols, coords, charge) — charge 0 when the file says 'unknown'."""
    sym, xyz, q, _known = crystal_full(path)
    return sym, xyz, q


def crystal_full(path):
    """Return (symbols, coords, charge, charge_known)."""
    lines = open(path).read().splitlines()
    n = int(lines[0])
    q, known = _parse_charge(lines[1])
    sym, xyz = [], []
    for line in lines[2 : 2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz), q, known


def is_borane(path):
    """True if the structure carries a boron cage UFF cannot type (>=5 B)."""
    sym, _xyz, _q, _k = crystal_full(path)
    return sum(s == "B" for s in sym) >= 5


def _listing():
    out = []
    for d, half in ((TMQM, "tmQM"), (FIXTURES, "fixtures")):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".xyz"):
                continue
            name = fn[:-4]
            out.append((name, os.path.join(d, fn), half))
    return out


ALL = _listing()
CORPUS = [(n, p, h) for n, p, h in ALL if n.split("_")[0] not in BORANE_NAMES and not is_borane(p)]
EXCLUDED = [(n, p, h) for n, p, h in ALL if (n, p, h) not in set(CORPUS)]

if __name__ == "__main__":
    from collections import Counter

    print("total files:", len(ALL), Counter(h for _n, _p, h in ALL))
    print("in corpus  :", len(CORPUS), Counter(h for _n, _p, h in CORPUS))
    print("excluded   :", [(n, h) for n, _p, h in EXCLUDED])
    unknown = [n for n, p, _h in ALL if not crystal_full(p)[3]]
    print("charge unknown:", len(unknown), unknown)
