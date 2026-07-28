"""T2: survey the tmQM corpus -> (name, metal, CN, charge, natoms, donor elements)."""

import glob, os, re, json
import numpy as np

TM = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))
SYM2Z = {}
from rdkit.Chem import GetPeriodicTable

pt = GetPeriodicTable()


def read(p):
    lines = open(p).read().splitlines()
    n = int(lines[0])
    hdr = lines[1]
    sym, xyz = [], []
    for l in lines[2 : 2 + n]:
        f = l.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return hdr, sym, np.array(xyz)


rows = []
for p in sorted(glob.glob("/home/ali/Documents/Codes/OIN-SMILES/tests/integration/tmQM/*.xyz")):
    hdr, sym, xyz = read(p)
    name = os.path.basename(p)[:-4]
    q = int(re.search(r"q = (-?\d+)", hdr).group(1))
    mnd = int(re.search(r"MND = (\d+)", hdr).group(1))
    z = [pt.GetAtomicNumber(s) for s in sym]
    mi = [i for i, zz in enumerate(z) if zz in TM]
    if len(mi) != 1:
        rows.append(
            dict(name=name, metal="MULTI" if mi else "NONE", nmetal=len(mi), cn=mnd, q=q, n=len(sym), donors="")
        )
        continue
    m = mi[0]
    d = np.linalg.norm(xyz - xyz[m], axis=1)
    # donor set: nearest heavies within a generous 2.9 A (just for the element census)
    don = sorted({sym[i] for i in range(len(sym)) if i != m and d[i] < 2.9 and z[i] > 1})
    rows.append(dict(name=name, metal=sym[m], nmetal=1, cn=mnd, q=q, n=len(sym), donors="".join(don)))

json.dump(rows, open(os.path.join(os.path.dirname(__file__), "t2_survey.json"), "w"), indent=0)
print(f"{len(rows)} structures")
from collections import Counter

print("metals:", Counter(r["metal"] for r in rows).most_common())
print("CN:", sorted(Counter(r["cn"] for r in rows).items()))
print("q:", sorted(Counter(r["q"] for r in rows).items()))
print(
    "natoms: min %d med %d max %d"
    % (min(r["n"] for r in rows), int(np.median([r["n"] for r in rows])), max(r["n"] for r in rows))
)
print("multi/none metal:", [r["name"] for r in rows if r["nmetal"] != 1])
