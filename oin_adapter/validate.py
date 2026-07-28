"""Index-free sanity verdict for one embedded metal complex — rxembed-only, no OIN needed.

Every quantity is perceived from THIS one structure (metal by element, donors by graph/geometry);
no atom index ever crosses to a second structure. Judges the six internal-chemistry criteria of the
validator map: M-donor distance band, ideal-polyhedron angles, metal over-bond, ligand clash,
connectivity, and folded donors.
"""

from __future__ import annotations

import numpy as np

import rxembed.geometry as geo
from rxembed import metrics
from rxembed.rdkit_embed import coordination as _coord
from rxembed.rdkit_embed.constraints import distance as _distance
from rxembed.rdkit_embed.constraints import metal as M  # noqa: N812 — rxembed's own module alias convention

_DIST_TOL = 0.30  # Å around the fitted ml_distance target
_ANG_TOL = 15.0  # deg around an ideal polyhedron vertex angle


def _metal_idx(mol) -> int:
    ms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in geo._METAL_Z]
    if len(ms) != 1:
        raise ValueError(f"expected exactly one metal, found {len(ms)}")
    return ms[0]


def _dative_donors(mol, m):
    """Donors from THIS mol's own DATIVE M-bonds — the exact sphere, no geometric guess needed."""
    from rdkit import Chem

    return [b.GetOtherAtomIdx(m) for b in mol.GetAtomWithIdx(m).GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]


def _sites(mol, pos, m, donors):
    """[(tag, unit_vec, dist, group, band)] with each haptic face collapsed to one centroid vertex.

    ``dist`` is the centroid->metal distance (the vertex position, used for the angle spectrum);
    ``band`` is the distance judged by the C1 model band — for a haptic face that is the MEAN
    metal->ring-atom distance, not the centroid distance (the centroid sits below the ring plane,
    so a per-ring-atom ml_distance target does not describe it).
    """
    remaining, groups = list(donors), []
    while remaining:  # mutually-bonded donors = one haptic site
        g = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for b in list(remaining):
                if any(mol.GetBondBetweenAtoms(b, x) for x in g):
                    g.append(b)
                    remaining.remove(b)
                    changed = True
        groups.append(g)
    out = []
    for g in groups:
        v = pos[g].mean(0) - pos[m]
        d = float(np.linalg.norm(v))
        band = float(np.mean([np.linalg.norm(pos[a] - pos[m]) for a in g]))
        els = sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in g)
        tag = f"{els[0]}{len(g)}" if len(g) > 1 else els[0]
        out.append((tag, v / d if d else v, d, g, band))
    return out


def _ideal_angles(geometry: str):
    dirs = M.vertex_dirs(geometry)
    if not dirs:
        return None
    u = [np.array(x, float) / np.linalg.norm(x) for x in dirs]
    return sorted(
        float(np.degrees(np.arccos(np.clip(u[i] @ u[j], -1, 1)))) for i in range(len(u)) for j in range(i + 1, len(u))
    )


def validate_geometry(
    mol, metal=None, donors=None, cid=-1, *, real_z=None, dist_tol=_DIST_TOL, ang_tol=_ANG_TOL
) -> dict:
    """Judge one embedded metal complex's internal chemistry. Returns a dict verdict.

    Keys: ``ok`` (bool), ``violations`` (list of chemistry-rendered strings), ``n_sites``,
    ``geometry`` (the CN-implied polyhedron). Perception is single-structure and index-free.
    """
    pos = mol.GetConformer(cid).GetPositions()
    m = metal if metal is not None else _metal_idx(mol)
    real_z = real_z or mol.GetAtomWithIdx(m).GetAtomicNum()
    if donors is None:
        # Prefer the mol's OWN dative bonds (the exact sphere); fall back to geometric perception.
        donors = _dative_donors(mol, m) or list(_coord._coordinating(mol, pos, m))
    donors = sorted(donors)
    fails: list[str] = []

    # C3/C4/C6 + bonds/H — the structural gate (metal perceived by element, passed donor set)
    fails += [str(v) for v in geo.check(mol, cid, donors=donors).violations]

    # C5 — graph + coordination-sphere diff, both against THIS mol's own bonds
    formed, broken = metrics.connectivity(mol, cid)
    if formed or broken:
        fails.append("connectivity: " + metrics.describe(mol, formed, broken))
    left, joined = metrics.coordination_changed(mol, cid, m, donors)
    if left or joined:
        fails.append(f"coordination changed: left={sorted(left)} joined={sorted(joined)}")

    sites = _sites(mol, pos, m, donors)
    has_haptic = any(len(g) > 1 for _t, _v, _d, g, _b in sites)  # any multi-atom face (eta2 included)
    has_apical = any(len(g) >= M._APICAL_MIN for _t, _v, _d, g, _b in sites)  # only eta>=3 fills >1 site

    # C1 — per-element M-donor distance band vs the fitted model (haptic: mean M-ring-atom distance)
    for tag, _v, _d, g, band in sites:
        target = _distance.ml_distance(mol, m, g[0], real_z, set(donors))
        if abs(band - target) > dist_tol:
            fails.append(f"M-{tag} distance {band:.2f} A vs model {target:.2f}+-{dist_tol}")

    # C2 — sorted measured L-M-L spectrum vs the sorted ideal-polyhedron spectrum (vertex-free).
    # TODO(haptic): an ideal polyhedron is a poor model for a piano-stool (a η-face centroid vertex
    # gives ~125 deg centroid-M-L / ~90 deg L-M-L, not the ideal tetrahedron's 109.5). The structural
    # gate (C3-C6) + the C1 band already vet a haptic sphere, so skip the ideal-angle multiset there.
    geometry = M.geometry_for(len(sites), has_apical=has_apical) or ""
    ideal = None if has_haptic else _ideal_angles(geometry)
    if ideal is not None:
        frag = M._frag_map(mol)
        meas = sorted(
            (float(np.degrees(np.arccos(np.clip(sites[i][1] @ sites[j][1], -1, 1)))), sites[i][3][0], sites[j][3][0])
            for i in range(len(sites))
            for j in range(i + 1, len(sites))
        )
        pool = list(ideal)
        for ang, ai, bi in meas:
            k = min(range(len(pool)), key=lambda t: abs(pool[t] - ang)) if pool else None
            if k is None or abs(pool[k] - ang) > ang_tol:
                if frag[ai] != frag[bi]:  # a chelate/eta2 bite is allowed to be off-ideal
                    fails.append(f"L-M-L {ang:.0f} deg: no ideal {geometry} vertex within {ang_tol} deg")
            else:
                pool.pop(k)

    return {"ok": not fails, "violations": fails, "n_sites": len(sites), "geometry": geometry}
