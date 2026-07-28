"""Crux measurement for the thione-coplanarity question.

Builds case1 (Ni thiosemicarbazone/thiourea.amidate.carboxylate), restores the real Ni,
and prints:
  - the donor set the metal has (index + element + stripped hybridisation)
  - the coplanar-cap entries (which donors are capped, and their plane refs)
  - for the thione S donor: codonor_in_plane verdict, and the shortest path +
    per-atom stripped hybridisation from S to EACH co-donor.

Read-only: no minimize, no monkeypatch of src.
Usage: uv run --no-sync python playground/thione_diag/thione_paths.py
"""

from __future__ import annotations

from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geo

rx.set_verbose("CRITICAL")

SMI = "C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"


def hname(h):
    return str(h).split(".")[-1] if h is not None else "UNKNOWN"


def main():
    iso_set = rx.metal(SMI, "square_planar")
    print(f"n isomers = {len(iso_set)}")
    iso = iso_set[0]
    mol = iso.mol  # surrogate metal in place
    cons = iso.cons

    # metal + donors
    print(f"metals = {sorted(cons.metals)}")
    donors = {b if a in cons.metals else a for a, b in cons.distances if a in cons.metals or b in cons.metals}
    print(f"iso.donors attr = {sorted(iso.donors)}")
    print(f"donors from distances = {sorted(donors)}")

    hyb = geo._stripped_hybridisation(mol)

    print("\n--- donor set (idx, element, stripped-hyb, degree) ---")
    for d in sorted(donors):
        a = mol.GetAtomWithIdx(d)
        print(f"  D={d:>3}  {a.GetSymbol():>2}  hyb={hname(hyb.get(d)):>7}  deg={a.GetDegree()}")

    print("\n--- coplanar cap entries (metal, donor, k, w, anchor, cap) ---")
    for e in cons.coplanar:
        m, dd, k, w, anchor, cap = e
        print(f"  donor={dd:>3}({mol.GetAtomWithIdx(dd).GetSymbol()})  k={k}  w={w}  anchor={anchor}  cap={cap}")
    capped = {e[1] for e in cons.coplanar}
    print(f"  capped donors = {sorted(capped)}")

    # find the thione S donor
    s_donors = [d for d in donors if mol.GetAtomWithIdx(d).GetSymbol() == "S"]
    print(f"\nthione S donor(s) = {s_donors}")
    for s in s_donors:
        print(f"\n===== S donor {s} =====")
        print(f"  in-plane sp2 donor? {geo.inplane_sp2_donor(mol, s, hyb)}")
        verdict = geo.codonor_in_plane(mol, s, donors, hyb)
        print(f"  codonor_in_plane -> {verdict}  ({'SKIP' if verdict else 'KEEP'})")
        for dd in sorted(donors):
            if dd == s:
                continue
            path = Chem.GetShortestPath(mol, int(s), int(dd))
            if not path:
                print(
                    f"  S{s} -> D{dd}({mol.GetAtomWithIdx(dd).GetSymbol()}): NO backbone path (separate fragment / via metal only)"
                )
                continue
            allsp2 = all(hyb.get(a) == Chem.HybridizationType.SP2 for a in path)
            chain = " - ".join(f"{a}{mol.GetAtomWithIdx(a).GetSymbol()}[{hname(hyb.get(a))}]" for a in path)
            print(f"  S{s} -> D{dd}({mol.GetAtomWithIdx(dd).GetSymbol()}): all-sp2={allsp2}")
            print(f"      path: {chain}")


if __name__ == "__main__":
    main()
