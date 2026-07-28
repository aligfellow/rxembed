"""Map the per-donor coplanar-cap constraint stack for each Karoline case + the controls.

For every capped donor D on the metal, print:
  - the cap tuple (metal, D, k, w, anchor, cap) and D's element/hybridisation
  - heavy-neighbour count (perm 1 = proper dihedral vs perm 2 = improper)
  - is D in a chelate ring closed THROUGH the metal? (a metallacycle -> backbone-locked)
  - does the cap's reference atom w reach ANOTHER donor of this metal through the LIGAND backbone
    (metal-cut graph)?  == "is the M-D-k-w dihedral closed into a metallacycle" -> plane already fixed
  - the polyhedron: coordination geometry + n donors

No src edits; read-only perception.  Usage: uv run --no-sync python playground/cap_architecture/caparch_map.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "playground/karoline_diag")
sys.path.insert(0, "playground/cap_softening")

from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo
from rxembed.constraints import metal as M  # noqa: N812

import capsweep_lib as L  # noqa: E402
import kdiag_harness as H  # noqa: E402

rx.set_verbose("CRITICAL")

COMPLEXES = {
    "case1_thione": H.CASES["case1"],
    "case2_pyimine_amidateN": H.CASES["case2"],
    "case3_pyimine_amidateC": H.CASES["case3"],
    "case4_ester": H.CASES["case4"],
    "HENRY": L.HENRY,
    "KETONE": L.KETONE,
    "PICO": L.PICO,
}


def metal_cut_topo(mol, metal):
    """Bond-path distance matrix on the metal-CUT graph (donors only reach each other via backbone)."""
    rw = Chem.RWMol(Chem.Mol(mol))
    # strip every metal-donor bond so paths go through the ligand backbone only
    for nb in [n.GetIdx() for n in mol.GetAtomWithIdx(metal).GetNeighbors()]:
        if rw.GetBondBetweenAtoms(metal, nb) is not None:
            rw.RemoveBond(metal, nb)
    m2 = rw.GetMol()
    m2.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m2)
    return Chem.GetDistanceMatrix(m2)


def reaches_codonor(topo, atom, donor_self, donors):
    """Nearest OTHER donor reachable from `atom` through the backbone; (donor, dist) or None."""
    best = None
    for d in donors:
        if d == donor_self:
            continue
        dist = topo[atom][d]
        if dist < 1e5 and (best is None or dist < best[1]):
            best = (d, int(dist))
    return best


def in_metallacycle(mol, metal, donor):
    """True if `donor` sits on a ring that also contains the metal (a genuine chelate ring)."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if metal in ring and donor in ring:
            return True
    return False


for name, smi in COMPLEXES.items():
    iso = rx.metal(smi, "square_planar")[0]
    mol, metal, donors = iso.mol, iso.metal, list(iso.donors)
    hyb = geo._stripped_hybridisation(mol)
    topo = metal_cut_topo(mol, metal)
    print(f"\n===== {name} =====")
    print(f"  metal idx={metal} donors={donors} geom={iso.summary()}")
    print(f"  n coplanar tuples = {len(iso.cons.coplanar)}")
    for i, d, k, w, anchor, cap in iso.cons.coplanar:
        a = mol.GetAtomWithIdx(d)
        heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
        perm = "proper(1heavy)" if len(heavy) == 1 else "improper(2heavy)"
        cyc = in_metallacycle(mol, metal, d)
        w_reach = reaches_codonor(topo, w, d, donors)
        k_reach = reaches_codonor(topo, k, d, donors)
        print(
            f"   D={d}({a.GetSymbol()},{str(hyb.get(d)).split('.')[-1]}) k={k} w={w}"
            f"  {perm}  metallacycle={cyc}"
            f"  w->codonor={w_reach}  k->codonor={k_reach}"
        )
