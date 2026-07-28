"""The coordination-sphere solver — re-centre a metal's windows on a point set that exists in R^3.

A FALLBACK reached only when a sphere's raw `ml_distance` targets are not mutually realisable (`bounds` detects
`tol > 0`); measured, that is ~2% of real chemistry. Carved out of `constraints.metal`; composes the polytope
vertex directions and the fitted distance model it re-centres. Reuses metal's logger so the convergence
warnings keep their `rxembed.constraints.metal` name.
"""

from __future__ import annotations

import itertools

import numpy as np
from rdkit.Chem import rdDistGeom

from .base import add_distance
from .distance import delocalised_charges, ml_distance
from .metal import VACANT, _frag_map, _site_radius, logger, vertex_dirs

_SOLVED_WIN = 0.05  # Å half-window around a SOLVED M-donor distance — the same width the raw model gets
#   (`_DIST_WIN`). Deliberately NOT widened: OIN triples this to 0.15 while its own comment claims the widths are
#   untouched, which blends a second mechanism (loosening) into what should be pure re-centring. Feasibility comes
#   from the centre being a real point set, and holds for any width >= 0, so widening buys nothing and only starves
#   the distance geometry.


def solve_targets(mol, cons):
    """Re-centre each metal's coordination windows on a point set that exists in R³; return a new Constraints.

    The fallback for a sphere whose raw targets are not mutually realisable — see `sphere.py` for why that
    happens and why this is never routine. Returns ``None`` when there is nothing to solve, when scipy is
    absent, or when any centre fails to converge: the caller then keeps the raw targets, which is strictly
    what it would have done anyway.

    Only the window CENTRES move. Widths, angles, floors, pulls and every non-metal constraint are carried
    through untouched by `Constraints.copy`.
    """
    from . import sphere as _sphere

    if not cons.spheres:
        return None
    if not _sphere.available():
        logger.info("sphere solver unavailable (scipy not installed) — keeping the raw coordination targets")
        return None

    out = cons.copy()
    solved_any = False
    for recipe in cons.spheres:
        metal, real_z = recipe.metal, recipe.real_z
        haptic = dict(recipe.haptic)
        geometry = recipe.geometry
        dirs = vertex_dirs(geometry)
        if dirs is None:
            continue
        od = [recipe.donors[k] for k in recipe.order]  # vertex -> donor atom (or VACANT), as `coordination`
        sites, u_star = [], []
        for v, d in enumerate(od):
            if d == VACANT:
                continue
            atoms = tuple(haptic[d]) if d in haptic else (d,)  # a haptic face is ONE site with many atoms
            sites.append(atoms)
            u_star.append(dirs[v])
        row_atom = [a for s in sites for a in s]  # the solver's rows: one per coordinating ATOM, flat
        atom_rows = {a: r for r, a in enumerate(row_atom)}  # ...so this must key on the ROW, not the site ordinal
        if len(sites) < _MIN_SOLVE_SITES:
            continue
        n = len(row_atom)
        # `ml_distance` reads hapticity off the DONOR SET, so it must be the real coordinating atoms — `od` holds
        # centroid dummies, whose ring siblings are not in it, and every ring atom would come back as a sigma
        # carbanion (eta 0) ~0.18 A short. `charges=qdel` for the same reason `_centroid_constraints` uses it: a
        # raw formal charge is a Lewis artefact.
        real_set = {a for s in sites for a in s}
        qdel = delocalised_charges(mol)
        targets = np.array([ml_distance(mol, metal, a, real_z, real_set, qdel) for a in row_atom])
        site_rows = [[atom_rows[a] for a in s] for s in sites]
        rings = [
            (rows, _site_radius(mol, tuple(row_atom[r] for r in rows)))
            for rows in site_rows
            if len(rows) >= _RIGID_FACE
        ]
        ring_rows = {r for rows, _ in rings for r in rows}
        free = [r for r in range(n) if r not in ring_rows]
        solver = _sphere.SphereSolver(
            n, targets, free, rings, site_rows, np.array(u_star, float), _ligand_pairs(mol, row_atom, metal)
        )
        z0 = _solve_start(targets, site_rows, u_star, rings, free, row_atom)
        try:
            x, ok = solver.solve(z0)
        except _sphere.SolverUnavailableError:
            return None
        if not ok or not np.all(np.isfinite(x)):
            logger.warning("sphere solver did not converge for metal %d — keeping the raw targets", metal)
            continue
        for r, a in enumerate(row_atom):
            d = float(np.linalg.norm(x[r]))
            add_distance(out.distances, metal, a, max(0.5, d - _SOLVED_WIN), d + _SOLVED_WIN)
        for dummy, rows in zip(_dummies(od, haptic), site_rows, strict=False):
            if dummy is None:  # a sigma site has no centroid scaffolding to keep in step
                continue
            # The face's M->centroid window is a SECOND description of the same geometry. Left at its raw value
            # beside re-centred M->ring-atom windows it contradicts them, so it moves onto the solved centre too.
            d = float(np.linalg.norm(x[rows].mean(axis=0)))
            add_distance(out.distances, metal, dummy, max(0.5, d - _SOLVED_WIN), d + _SOLVED_WIN)
        solved_any = True
    return out if solved_any else None


def _dummies(od, haptic):
    """Return the centroid-dummy index per occupied vertex, or None for a sigma donor (parallel to `sites`)."""
    return [d if d in haptic else None for d in od if d != VACANT]


_MIN_SOLVE_SITES = 2  # below two vertices there is no polytope to satisfy
_RIGID_FACE = 3  # a site of this many atoms or more is a rigid regular n-gon, not free points


def _ligand_pairs(mol, rows, metal):
    """RDKit's own donor-donor bounds, as ``(row_i, row_j, lo, hi)`` — the ligand's reach, no custom radii."""
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    frag = _frag_map(mol)
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if frag.get(a) != frag.get(b):  # only atoms of ONE ligand constrain each other's reach
            continue
        lo, hi = bm[max(a, b)][min(a, b)], bm[min(a, b)][max(a, b)]
        out.append((i, j, float(lo), float(hi)))
    return out


def _solve_start(targets, site_rows, u_star, rings, free, row_atom):
    """Start the solve at the IDEAL sphere: each site on its polytope vertex at its target distance."""
    x = np.zeros((len(row_atom), 3))
    for s, rows in enumerate(site_rows):
        u = np.array(u_star[s], float)
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        for r in rows:
            x[r] = u * targets[r]
    z0 = [x[r] for r in free]
    for rows, _radius in rings:
        centre = x[rows].mean(axis=0)
        normal = centre / max(float(np.linalg.norm(centre)), 1e-9)  # a face points along its own metal vector
        z0.extend([centre, normal])
    return np.concatenate(z0) if z0 else np.zeros(0)
