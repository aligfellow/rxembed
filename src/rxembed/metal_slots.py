"""Enumerate distinct, reachable donor assignments to metal-polyhedron slots."""

from __future__ import annotations

import itertools
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdDistGeom

from .metal_core import (
    VACANT,
    _frag_map,
    _vertex_atom,
    logger,
)
from .metal_donor_orient import _FOLD_WINDOW, _stripped_hybridisation, donation_axis
from .metal_polyhedron import (
    CHELATE_SPAN_ANGLE,
    POLYHEDRA,
    _fit_trace,
    _seat_by_alignment,
    _vertex_angle,
    describe,
    isomer_permutations,
    seat_properly,
    vertex_dirs,
)
from .metal_stereo import chirality_of, site_classes

_PT = GetPeriodicTable()
TRANS_ANGLE = 150  # same-element donor pairs beyond this are trans
_PAIR = 2
_TRIAD = 3
_COLINEAR_TOL = 0.5
_SPAN_TOL = 0.1  # Å slack drops a short chelate forced trans while a real long backbone passes
# At 0.2 Å the amidate-trans phantom returns.


def _octahedral_triad(mol, od, haptic=None):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else ``None``.

    Either a tridentate chelate (exactly 3 donors of one ligand fragment) or exactly 3 monodentate donors of
    one element (an MA3B3 set); ``None`` otherwise, and then cis/trans is used. The exactly-3 and monodentate
    conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate (en2, en3)
    have no mer/fac, so neither must be forced into a triad.
    """
    real = [(p, _vertex_atom(haptic, od[p])) for p in range(len(od)) if od[p] != VACANT]
    if len(real) < _TRIAD:
        return None
    frag = _frag_map(mol)
    by_frag = {}
    for p, d in real:
        by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():  # a tridentate chelate (one ligand, exactly 3 donors)
        if len(ps) == _TRIAD:
            return tuple(ps)
    by_elem = {}
    for p, d in real:
        if len(by_frag[frag[d]]) == 1:  # else exactly three monodentate same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == _TRIAD:
            return tuple(ps)
    return None


def order_label(mol, donors, geometry, order, haptic=None):
    """Build the isomer label from the ideal polyhedron; no conformer needed.

    Vacant vertices are ignored. Octahedral with a donor triad is mer/fac (one trans pair in the triad means
    mer, none means fac); otherwise cis/trans, judged on the minority same-element donor pair, which is the
    set whose placement defines the isomerism: the 2 Cl of an MA4B2, not the 4 A that always have a trans
    pair.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return f"isomer{order}"
    if not POLYHEDRA[geometry].geometric_isomerism:
        return ""  # no cis/trans distinction for this geometry: a single arrangement
    od = [donors[k] for k in order]
    if geometry == "octahedral":
        tri = _octahedral_triad(mol, od, haptic)
        if tri is not None:
            trans = sum(
                1 for i in range(3) for j in range(i + 1, 3) if _vertex_angle(dirs[tri[i]], dirs[tri[j]]) > TRANS_ANGLE
            )
            return "fac" if trans == 0 else "mer"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] != VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(_vertex_atom(haptic, od[p])).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) == _PAIR}
    if not pairs:
        return ""  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    ps = pairs[e]
    trans = any(
        _vertex_angle(dirs[ps[i]], dirs[ps[j]]) >= TRANS_ANGLE for i in range(len(ps)) for j in range(i + 1, len(ps))
    )
    return "trans" if trans else "cis"


def realised_label(mol, metal, donors, cid, geometry=None):
    """Label a realised coordination arrangement as cis or trans."""
    polyhedron = POLYHEDRA.get(geometry)
    if polyhedron is not None and not polyhedron.geometric_isomerism:
        return ""
    positions = mol.GetConformer(cid).GetPositions()
    for a in range(len(donors)):
        for b in range(a + 1, len(donors)):
            same = mol.GetAtomWithIdx(donors[a]).GetSymbol() == mol.GetAtomWithIdx(donors[b]).GetSymbol()
            angle = _vertex_angle(positions[donors[a]] - positions[metal], positions[donors[b]] - positions[metal])
            if same and angle >= TRANS_ANGLE:
                return "trans"
    return "cis"


def input_ordering(mol, metal, donors, geometry):
    """Fit input donors to ideal polyhedron slots.

    Orthogonal Procrustes finds ``donors[order[slot]]``. Reflection is allowed for the fit, then
    `seat_properly` restores the correct hand. Frozen donors therefore retain their realised slots.
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([np.zeros(3) if d == VACANT else pos[d] - pos[metal] for d in donors], float)
    dd /= np.where((norm := np.linalg.norm(dd, axis=1, keepdims=True)) > 0, norm, 1.0)
    v_ideal = np.array(dirs_ref, float)
    canned = isomer_permutations(geometry)
    if canned is None:  # no canned list (CN7/8): searching only the identity would seat donors in PERCEPTION
        return seat_properly(dd, dirs_ref, _seat_by_alignment(dd, v_ideal))  # order, not a seating at all
    best_score, best_order = -1.0, list(range(len(donors)))
    for order in canned:
        score = _fit_trace(dd[list(order)].T @ v_ideal)  # best alignment; see `_fit_trace` on reflections
        if score > best_score:
            best_score, best_order = score, order
    return seat_properly(dd, dirs_ref, best_order)


def _central_trans(od, frag, dmat, dirs, haptic=None):
    """Return True if a tridentate chelate's central donor is placed trans to one of its own arms.

    The central donor is the one on the backbone path between the other two (``d(a,c)+d(c,b)==d(a,b)``), which
    a pincer cannot do: central is cis to both arms in every real mer/fac. A flexible chelate can stretch to
    ~155° without a formally torn bond, so this drops it at enumeration rather than leaving it to `bonding_ok`.
    A haptic vertex is resolved to its ring atom so its backbone path is real.
    """
    ra = [_vertex_atom(haptic, d) if d != VACANT else VACANT for d in od]  # vertex -> representative real atom
    by_frag = {}
    for p, d in enumerate(ra):
        if d != VACANT:
            by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():
        if len(ps) != _TRIAD:
            continue
        for ci in range(3):
            c, a, b = ps[ci], ps[(ci + 1) % 3], ps[(ci + 2) % 3]
            if abs(dmat[ra[a]][ra[c]] + dmat[ra[c]][ra[b]] - dmat[ra[a]][ra[b]]) < _COLINEAR_TOL:  # c is central
                if _vertex_angle(dirs[c], dirs[a]) > TRANS_ANGLE or _vertex_angle(dirs[c], dirs[b]) > TRANS_ANGLE:
                    return True
                break
    return False


def _span_bounds(mol):
    """Return the ligands' own bounds matrix, or None if it cannot be built, in which case never filter.

    Bounds come from the ligand's own connectivity, i.e. how far this backbone can actually reach: the metal
    is bond-less here, so every path runs through the backbone rather than across the centre. Read as
    ``bm[i][j]`` for a pair's upper bound and ``bm[j][i]`` for its lower (i < j).
    """
    try:
        return rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except Exception:  # pragma: no cover; the bounds matrix is robust, but never let the gate crash here
        return None


def _reach(bm, i, j):
    """Return the pair's upper-bound (max reachable) distance: the bounds matrix's i<j triangle."""
    return float(bm[min(i, j)][max(i, j)])


def _donor_faces_metal(mol, d, *, other, d_md, d_mo, need, bm, hyb, donors):
    """Return whether a trans-spanning donor can still face the metal.

    The fold ruler converts its M-D-X floor into the required X···D' reach. Hydride, bridging, haptic and
    uncalibrated donors abstain because they have no supported donation axis.
    """
    subs = donation_axis(mol, d, donors)
    if subs is None:  # hydride / bridging / haptic: no donation axis, so "does it face the metal" is meaningless
        return True
    cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None
    if cls not in _FOLD_WINDOW:  # estimators disagreed, or n < 6 for the class: the ruler abstains, so do we
        return True
    beta = math.degrees(  # angle(M, D, D') in the M-D-D' triangle: how far off the D···D' line the metal sits
        math.acos(max(-1.0, min(1.0, (d_md**2 + need**2 - d_mo**2) / (2 * d_md * need))))
    )
    alpha = _FOLD_WINDOW[cls][0] - beta  # the X-D···D' angle the fold floor forces (M is `beta` off that line)
    if alpha <= 0:  # the metal already sits far enough off the line, so the floor costs the backbone nothing
        return True
    for x in subs:
        r = float(bm[max(x, d)][min(x, d)])  # the D-X bond length (a bond's bounds coincide to < 0.05 Å)
        out = math.sqrt(r**2 + need**2 - 2 * r * need * math.cos(math.radians(alpha)))  # X···D' this forces
        if _reach(bm, x, other) < out - _SPAN_TOL:  # the backbone can't hold X that far off the metal
            return False
    return True


def _chelate_span_ok(mol, od, *, frag, dirs, bm, r_metal, hyb, donors, haptic=None):
    """Reject wide chelate assignments that cannot span and donate.

    The ligand bounds matrix and law of cosines test donor separation; `_donor_faces_metal` tests orientation.
    Long bridges remain valid, while cis pairs bypass this trans-span gate.
    """
    if bm is None:  # no bounds matrix -> never filter; defer to `bonding_ok` downstream
        return True
    for p in range(len(od)):
        for q in range(p + 1, len(od)):
            a, b = od[p], od[q]
            # A haptic centroid is bond-less, so resolve each vertex to a representative ring atom and let the
            # same-ligand test, the covalent reach and the backbone bounds all read the face's real chemistry.
            # Otherwise a face tethered to a co-donor reads as a separate ligand and its trans is never dropped.
            ra, rb = _vertex_atom(haptic, a), _vertex_atom(haptic, b)
            if VACANT in (a, b) or frag[ra] != frag[rb]:  # only a same-ligand (chelate) pair
                continue
            theta = _vertex_angle(dirs[p], dirs[q])
            if theta < CHELATE_SPAN_ANGLE:  # cis / adjacent -> the chelate folds in, always feasible
                continue
            d_ma = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(ra).GetAtomicNum())  # real M-donor covalent sums,
            d_mb = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(rb).GetAtomicNum())  # not a fixed 2.0 (Pd-N ~2.1)
            need = math.sqrt(d_ma**2 + d_mb**2 - 2 * d_ma * d_mb * math.cos(math.radians(theta)))  # law of cosines
            if _reach(bm, ra, rb) < need - _SPAN_TOL:  # backbone can't reach
                return False
            # The orientation test asks whether the donor can still aim its lone pair at the metal, which is
            # meaningless for a haptic face: it donates a π face and has no axis, so a centroid abstains.
            if a not in (haptic or {}) and not _donor_faces_metal(
                mol, ra, other=rb, d_md=d_ma, d_mo=d_mb, need=need, bm=bm, hyb=hyb, donors=donors
            ):
                return False
            if b not in (haptic or {}) and not _donor_faces_metal(
                mol, rb, other=ra, d_md=d_mb, d_mo=d_ma, need=need, bm=bm, hyb=hyb, donors=donors
            ):
                return False
    return True


def _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, coordination=(), *, limit=None):
    """Deduplicate reachable slot assignments by constitutional signature.

    Pair classes, ideal angles, same-ligand path lengths and metal hand distinguish candidates. Central-trans
    tridentates and unreachable wide chelates are removed; `limit` permits an early ambiguity check.
    """
    frag = _frag_map(mol)  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances
    real_donors = [d for d in donors if d != VACANT]
    classes = site_classes(mol, donors, haptic, coordination)
    bm = _span_bounds(mol)
    hyb = _stripped_hybridisation(mol)  # the fold ruler's own (element, hyb) class: graph-only, no coords
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))]
    angle = {(p, q): _vertex_angle(dirs[p], dirs[q]) for p, q in pairs}  # a vertex-pair's angle is donor-independent

    def link(od, p, q):  # intra-ligand bond distance of a same-ligand pair
        a, b = _vertex_atom(haptic, od[p]), _vertex_atom(haptic, od[q])  # resolve a centroid to its ring atom
        if VACANT in (od[p], od[q]) or frag[a] != frag[b]:  # distinguishes a chelate's central from its
            return -1  # terminal donor; -1 for different ligands or a vacancy
        return int(dmat[a][b])

    def donor_class(d):
        return ("vacant",) if d == VACANT else ("donor", classes[d])

    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        if _central_trans(od, frag, dmat, dirs, haptic):  # a tridentate's central donor trans to its own arm
            continue
        if not _chelate_span_ok(
            mol, od, frag=frag, dirs=dirs, bm=bm, r_metal=r_metal, hyb=hyb, donors=real_donors, haptic=haptic
        ):  # can't span/donate trans
            continue
        sig = tuple(
            sorted(
                (tuple(sorted((donor_class(od[p]), donor_class(od[q])))), link(od, p, q), angle[(p, q)])
                for p, q in pairs
            )
        )
        sig = (sig, chirality_of(mol, geometry, od, haptic, coordination))
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if limit is not None and len(out) >= limit:
                break
    return out


def distinct_vertex_orderings(mol, donors, geometry, perms=None, r_metal=1.4, haptic=None, coordination=()):
    """Enumerate distinct coordination isomers: every distinct vertex arrangement, minimally pre-filtered.

    Dedup and the two feasibility pre-filters live in `_distinct_orderings`. `perms` overrides the candidate
    vertex orderings (default ``isomer_permutations(geometry)``); a ``fix=`` enumeration passes the subset that
    keeps each frozen donor pinned to its input vertex.
    """
    dirs = vertex_dirs(geometry)
    if perms is None and isomer_permutations(geometry) is None:  # CN7/8, tetrahedral and linear have no
        # canned list, so the input ordering is the only candidate. Warn only where that loses something: a
        # linear or all-identical geometry has exactly one arrangement and the warning would be noise.
        orbit = itertools.permutations(range(len(donors)))
        if (
            dirs is not None
            and len(_distinct_orderings(mol, donors, geometry, orbit, dirs, r_metal, haptic, coordination, limit=2)) > 1
        ):
            logger.info(
                "metal[%s]: no isomer permutations tabulated; enumerating the input ordering only",
                describe(geometry),
            )
        # Unfiltered: with a single ordering there is nothing to prefer it over, so filtering could only
        # return empty and make `rx.metal(...)[0]` raise. `bonding_ok` and the geometry gate judge it.
        return [list(range(len(donors)))]
    perms = perms if perms is not None else isomer_permutations(geometry)
    if dirs is None:
        return perms
    return _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, coordination)
