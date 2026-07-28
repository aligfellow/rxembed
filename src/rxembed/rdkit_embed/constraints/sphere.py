"""Solve a coordination sphere for donor positions that actually exist in R³.

**Why this exists.** The M-donor distance model, the polyhedron's ideal vertex directions, and the ligand's
own reach are three INDEPENDENTLY derived statements about one point set. Nothing reconciles them, so they
can be mutually impossible — and when they are, triangle smoothing repairs the contradiction by rewriting a
crossed bound, which the embedded geometry then pays for. Solving for a point set that satisfies all three at
once makes the windows drawn around it feasible **by construction**: a real point set's distance matrix is a
metric, so it satisfies every triangle inequality and ``tol == 0`` becomes a theorem rather than a hope.

**This is a fallback, never routine.** It is reached only when the raw targets are measurably non-metric
(`bounds.embed` detects `tol > 0`), because it costs fidelity: it abandons the fitted `ml_distance` numbers
in favour of a jointly realisable compromise. On this corpus the raw targets are metric everywhere, so it
should essentially never fire — if it starts firing routinely, something upstream regressed.

**Formulation.** Variables are the free donors (3 dof each) plus, per symmetric haptic ring, a centroid (3)
and a normal (3). A ring whose atoms are one topology class is a RIGID regular n-gon: solving its atoms as
free dof lets the optimiser break a symmetry that is chemistry, not slack. The ring TILT is the normal's 2
dof, so this does not impose equidistance — that falls out, or does not, which is the honest treatment.

Residuals, each scaled by its own sigma: radial (hit the M-L target), chord (hit the polytope's vertex
separation), ligand hinge (stay inside RDKit's own donor-donor bounds, a soft box), and |normal| = 1.

Adapted from OIN-SMILES `generation/sphere_solver.py`. Two deliberate departures, both recorded in
`docs/` terms: this port does NOT widen the resulting windows (OIN triples the M-donor half-window from
0.05 to 0.15 Å while its comment claims widths are untouched — that is a second mechanism blended into the
re-centring, and it is left out), and scipy is a GUARDED optional import so the kernel keeps its
rdkit + numpy floor.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger("rxembed.constraints.sphere")  # pinned name: kept under the "rxembed" logger tree
#   (set_verbose configures it), unchanged by the carve to rdkit_embed.

# Residual weights. Each sigma is a DIVISOR (`(value - target) / SIGMA`), so a SMALLER sigma weights its term
# MORE. Trust the bond length and let the polyhedron absorb: for a chelate the ideal vertex angle and the
# ligand's reach are incompatible at the true bond length, and a real molecule resolves that by keeping the
# bond and distorting the polyhedron, not by inflating the sphere to preserve an ideal angle.
# Measured on a bite the ideal 90 deg cannot meet, against a 2.00 A target: 0.10/0.20 keeps the bond at 1.99
# and gives on the angle (80.3 deg); 0.20/0.10 sacrifices the bond to 1.91 to hold 85.0 deg. NB this is a
# DEPARTURE from OIN, which ships 0.20/0.10 — its own comment describes that ratio as the bug and says the
# soft pull, not sigma, was its lever. rxembed's soft pull lives in the force field, so it cannot compensate
# here and the ratio has to be right on its own.
SIGMA_R = 0.10  # radial (M-donor distance) — the tighter term: keep the bond
SIGMA_C = 0.20  # chord (polytope vertex separation) — the looser term: let the angle give
SIGMA_H = 0.05  # ligand reach, from RDKit's own bounds
SIGMA_N = 0.01  # |normal| = 1 — removes the ring parametrisation's null direction
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
        own bounds. ``chord_override`` replaces a site pair's polytope chord — the LIGAND sets a chelate bite,
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
        # Normalise HERE, not in the template. `residual` compares the chord against UNIT site centroids, so a
        # non-unit template states a target that cannot be reached: tetrahedral is written (1,1,1) (norm sqrt 3),
        # giving 2.83 against a maximum achievable 2.0, so its chord term could never be satisfied. `vertex_dirs`
        # keeps its readable form for its other consumers, and `_solve_start` already normalises — the start
        # point was right and only the target was wrong.
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

        A numerical Jacobian is used rather than OIN's analytic one: the analytic form is ~50x faster, but
        this path is a rare fallback, and a hand-derived Jacobian is a correctness liability that only pays
        off in a hot loop. If the solver ever becomes routine, port the analytic Jacobian and its 5e-8
        verification with it.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # the kernel's floor is rdkit + numpy; scipy is an extra
            raise SolverUnavailableError(
                "the coordination-sphere solver needs scipy — install the 'sphere' extra"
            ) from exc
        sol = least_squares(self.residual, z0, method="trf", max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        return self.unpack(sol.x), bool(sol.success)
