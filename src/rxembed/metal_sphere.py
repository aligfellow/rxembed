"""Solve a coordination sphere for donor positions that actually exist in R³.

A fallback, never routine: `bounds` calls it only where smoothing had to repair a crossed bound
(`tol > 0`), 5 of 251 calls. It costs fidelity, abandoning the fitted `ml_distance` numbers for a jointly
realisable compromise, so routine firing means something upstream regressed. Needs scipy (the `sphere`
extra) and degrades with a message without it.

The M-donor distance model, the polyhedron's vertex directions and the ligand's reach are three
independently derived statements about one point set, so they can be mutually impossible. Solving for a
point set that satisfies all three makes the windows feasible by construction: a real point set's distance
matrix is a metric, so ``tol == 0`` becomes a theorem rather than a hope.

Variables are the free donors (3 dof each) plus, per symmetric haptic ring, a centroid and a normal. A ring
of one topology class is a rigid n-gon; solving its atoms freely would let the optimiser break a symmetry
that is chemistry. Residuals, each scaled by its sigma: radial, chord, ligand hinge, and |normal| = 1.

Departures from the formulation this was adapted from: the windows are not widened, and scipy is guarded.
`solve_targets` is the driver, returning a `Constraints` whose M-donor window centres moved onto the solve.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from rdkit.Chem import rdDistGeom

from .constraints import add_distance
from .metal_core import VACANT, _frag_map, _regular_face, _site_radius
from .metal_distance import delocalised_charges, ml_distance
from .metal_polyhedron import vertex_dirs

logger = logging.getLogger("rxembed.sphere")  # under the "rxembed" tree `set_verbose` configures

# Residual weights: each sigma is a divisor, so a smaller sigma weights its term more heavily.
SIGMA_R = 0.10  # radial (M-donor distance), the tighter term: keep the bond
SIGMA_C = 0.20  # chord (polytope vertex separation), the looser term: let the angle give
SIGMA_H = 0.05  # ligand reach, from RDKit's own bounds
SIGMA_N = 0.01  # |normal| = 1, removing the ring parametrisation's null direction
_BETA = 20.0  # softplus sharpness for the one-sided hinge
_EPS = 1e-9


class SolverUnavailableError(RuntimeError):
    """scipy is not installed, so the sphere cannot be solved."""


def available() -> bool:
    """Report whether the optional solver dependency (scipy) is importable."""
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return False
    return True


def _perp(nh):
    """Branch-free orthonormal basis perpendicular to each unit normal."""
    e = np.eye(3)[np.argmin(np.abs(nh), axis=1)]
    p1 = np.cross(nh, e)
    p1 /= np.maximum(np.linalg.norm(p1, axis=1, keepdims=True), 1e-12)
    return p1, np.cross(nh, p1)


class SphereSolver:
    """Least-squares solve for donor points satisfying the M-L targets, the polytope, and the ligand's reach."""

    def __init__(self, n, targets, free_rows, rings, site_rows, u_star, ligand_pairs, chord_override=None):
        """Precompute every index array once so the residual is pure vectorised numpy.

        ``targets`` is the per-donor M-L distance; ``rings`` is ``[(rows, radius)]`` for each rigid face;
        ``site_rows`` groups donor rows into coordination *sites* (a haptic face is one site, many donors);
        ``u_star`` is the ideal unit direction per site; ``ligand_pairs`` is ``[(i, j, lo, hi)]`` from RDKit's
        own bounds. ``chord_override`` replaces a site pair's polytope chord: the ligand sets a chelate bite,
        not the polyhedron, which is the same rule the bounds writer applies by intersecting rather than
        overwriting.
        """
        self.n = n
        self.d = np.asarray(targets, float)
        self.free = np.asarray(free_rows, dtype=int)
        self.nf = len(self.free)
        self.K = len(rings)
        self.nz = 3 * self.nf + 6 * self.K

        rr, ro, rc, rs, rad = [], [], [], [], []
        for k, (rows, radius) in enumerate(rings):
            m = len(rows)
            ph = 2 * np.pi * np.arange(m) / m
            rr.extend(rows)
            ro.extend([k] * m)
            rc.extend(np.cos(ph))
            rs.extend(np.sin(ph))
            rad.extend([radius] * m)
        self.rr, self.ro = np.asarray(rr, dtype=int), np.asarray(ro, dtype=int)
        self.rc, self.rs, self.rad = np.asarray(rc, float), np.asarray(rs, float), np.asarray(rad, float)

        self.ns = len(site_rows)
        site = np.zeros((self.ns, n))
        for i, rows in enumerate(site_rows):
            site[i, rows] = 1.0 / len(rows)
        self.S = site
        self.cp = np.asarray(list(itertools.combinations(range(self.ns), 2)), dtype=int).reshape(-1, 2)
        # Normalise here, not in the template: `residual` compares the chord against a unit-vector target, so an
        # unnormalised template silently rescales every angle it is asked about.
        u = np.asarray(u_star, float)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), _EPS)
        self.chord_t = np.linalg.norm(u[self.cp[:, 0]] - u[self.cp[:, 1]], axis=1) if len(self.cp) else np.zeros(0)
        for e, (i, j) in enumerate(self.cp):
            c = (chord_override or {}).get((int(i), int(j)), (chord_override or {}).get((int(j), int(i))))
            if c is not None:
                self.chord_t[e] = c

        ringset = [set(rows) for rows, _ in rings]  # a rigid ring meets its own internal distances exactly
        keep = [p for p in ligand_pairs if not any(p[0] in rs_ and p[1] in rs_ for rs_ in ringset)]
        self.LP = np.asarray([(i, j) for i, j, _, _ in keep], dtype=int).reshape(-1, 2)
        self.LO = np.asarray([lo for _, _, lo, _ in keep], float)
        self.HI = np.asarray([hi for _, _, _, hi in keep], float)
        self.nres = n + len(self.cp) + len(self.LP) + self.K

    def unpack(self, z):
        """Expand the variable vector into donor coordinates ``X`` (n, 3), metal at the origin."""
        x = np.zeros((self.n, 3))
        if self.nf:
            x[self.free] = z[: 3 * self.nf].reshape(self.nf, 3)
        if self.K:
            r = z[3 * self.nf :].reshape(self.K, 6)
            centre, normal = r[:, :3], r[:, 3:]
            nh = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), _EPS)
            p1, p2 = _perp(nh)
            x[self.rr] = centre[self.ro] + self.rad[:, None] * (
                self.rc[:, None] * p1[self.ro] + self.rs[:, None] * p2[self.ro]
            )
        return x

    def residual(self, z):
        """Residual vector: radial, polytope chord, ligand hinge, |normal| = 1."""
        x = self.unpack(z)
        r = np.maximum(np.linalg.norm(x, axis=1), _EPS)
        rad = (r - self.d) / SIGMA_R
        centre = self.S @ x
        cn = np.maximum(np.linalg.norm(centre, axis=1), _EPS)
        u = centre / cn[:, None]
        if len(self.cp):
            v = u[self.cp[:, 0]] - u[self.cp[:, 1]]
            chord = (np.maximum(np.linalg.norm(v, axis=1), 1e-12) - self.chord_t) / SIGMA_C
        else:
            chord = np.zeros(0)
        if len(self.LP):
            dv = x[self.LP[:, 0]] - x[self.LP[:, 1]]
            dij = np.maximum(np.linalg.norm(dv, axis=1), _EPS)
            # A softplus box: zero inside [lo, hi], smoothly rising outside. Differentiable, unlike a clip.
            hinge = (np.logaddexp(0.0, _BETA * (self.LO - dij)) + np.logaddexp(0.0, _BETA * (dij - self.HI))) / (
                _BETA * SIGMA_H
            )
        else:
            hinge = np.zeros(0)
        if self.K:
            normal = z[3 * self.nf :].reshape(self.K, 6)[:, 3:]
            unit = (np.linalg.norm(normal, axis=1) - 1.0) / SIGMA_N
        else:
            unit = np.zeros(0)
        return np.concatenate([rad, chord, hinge, unit])

    def solve(self, z0, max_nfev=800):
        """Least-squares solve from start point ``z0``; return ``(X, converged)``.

        The Jacobian is numerical. An analytic one is ~50x faster but is a correctness liability that only
        pays off in a hot loop, and this path is a rare fallback; write one if it ever becomes routine.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # the core's floor is rdkit + numpy; scipy is an extra
            raise SolverUnavailableError(
                "the coordination-sphere solver needs scipy; install the 'sphere' extra"
            ) from exc
        sol = least_squares(self.residual, z0, method="trf", max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        return self.unpack(sol.x), bool(sol.success)


# --- the driver: re-centre one metal's windows on the solved point set ------------------------------------


_SOLVED_WIN = (
    0.05  # Å half-window around a solved M-donor distance: the solve already reconciled the three statements, so
)
# this only needs to leave the DG room to land on it.


def solve_targets(mol, cons):
    """Re-centre each metal's coordination windows on a point set that exists in R³; return a new Constraints.

    The fallback for a sphere whose raw targets are not mutually realisable; see the module header for why that
    happens and why this is never routine. Returns ``None`` when there is nothing to solve, when scipy is
    absent, or when any centre fails to converge: the caller then keeps the raw targets, which is strictly
    what it would have done anyway.

    Only the window CENTRES move. Widths, angles, floors, pulls and every non-metal constraint are carried
    through untouched by `Constraints.copy`.
    """
    if not cons.spheres:
        return None
    if not available():
        logger.info("sphere solver unavailable (scipy not installed); keeping the raw coordination targets")
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
            atoms = tuple(haptic[d]) if d in haptic else (d,)  # a haptic face is one site with many atoms
            sites.append(atoms)
            u_star.append(dirs[v])
        row_atom = [a for s in sites for a in s]  # the solver's rows: one per coordinating ATOM, flat
        atom_rows = {a: r for r, a in enumerate(row_atom)}  # ...so this must key on the ROW, not the site ordinal
        if len(sites) < _MIN_SOLVE_SITES:
            continue
        n = len(row_atom)
        # `ml_distance` reads hapticity off the donor set, so it must be the real ring atoms, not the centroid
        real_set = {a for s in sites for a in s}
        qdel = delocalised_charges(mol)
        targets = np.array([ml_distance(mol, metal, a, real_z, real_set, qdel) for a in row_atom])
        site_rows = [[atom_rows[a] for a in s] for s in sites]
        rings = [
            (rows, _site_radius(mol, tuple(row_atom[r] for r in rows)))
            for rows in site_rows
            if _rigid_ring(mol, [row_atom[r] for r in rows])
        ]
        ring_rows = {r for rows, _ in rings for r in rows}
        free = [r for r in range(n) if r not in ring_rows]
        solver = SphereSolver(n, targets, free, rings, site_rows, np.array(u_star, float), _ligand_pairs(mol, row_atom))
        z0 = _solve_start(targets, site_rows, u_star, rings, free, row_atom)
        try:
            x, ok = solver.solve(z0)
        except SolverUnavailableError:
            return None
        if not ok or not np.all(np.isfinite(x)):
            logger.warning("sphere solver did not converge for metal %d; keeping the raw targets", metal)
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
_RING_PARAM_MIN = 3  # below 3 atoms the (centre, normal) form costs the same 6 dof as free atoms; eta2 stays free


def _rigid_ring(mol, atoms):
    """Return True if this face is solved as one rigid n-gon rather than as free atoms.

    Rigidity is `_regular_face`, the same shared-radius test `_centroid_constraints` reads for cone mode, so
    the solver and the bounds matrix cannot disagree about a face. The count is the separate dof question.
    """
    return len(atoms) >= _RING_PARAM_MIN and _regular_face(mol, tuple(atoms))


def _ligand_pairs(mol, rows):
    """RDKit's own donor-donor bounds, as ``(row_i, row_j, lo, hi)``: the ligand's reach, no custom radii."""
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    frag = _frag_map(mol)
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if frag.get(a) != frag.get(b):  # only atoms of one ligand constrain each other's reach
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
