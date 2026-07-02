"""Resolve a user constraint spec (``fix`` / ``constrain`` / ``template``) into a `Constraints`.

**Three verbs, one resolver** (`resolve_core`), replacing the old six-kwarg surface:

- **``fix``** — *rigid*. The named atoms **will** have this geometry.
  - a **list** of atom indices → hold them at the source's **own** coords (Kabsch graft, needs a geometry).
  - a dict **``{i: (x, y, z)}``** → hold them at **explicit** coords (graft).
  - a dict **``{(i, j): d, (i, j, k): θ}``** → **numbers**: tight distance/angle windows, UFF-*pulled toward*
    the value (a clean target lands within ~0.05 Å / a few degrees — **confirm with ``.measure()``**, it is a
    pull, not a hard snap). Not grafted, not pose-frozen. A ``fix`` dict may **mix** coord and number entries.
- **``constrain``** — *soft*. Bias the seed with a wider window; a real energy may overrule it. Also carries
  π-stacks (a plane key ``(ring_a, ring_b): separation``). Rule of thumb: want it held? ``fix``. Want a bias
  the search/energy can move off? ``constrain``.
- **``template``** — *sugar* for a coords-``fix`` from a reference: ``(positions, {target_i: ref_i})``.

**Index-driven.** Keys are 0-based atom indices in xyz/graph order — the resolver never SMARTS-matches
internally (that is the user's two RDKit lines, shown in the notebooks); it kills the symmetric-core
automorphism flip and keeps the spec transparent. `resolve_atom` / `match` remain for the *query* helpers
(`measure`, `align`), not the constraint spec.

**Strictness = the fix/constrain axis, not a second force constant.** ``AddDistanceConstraint`` is
flat-bottom (zero penalty inside the window, quadratic outside), so a *tight* window (fix) is pulled to the
target and a *wide* window (constrain) floats within it under one shared ``distance_fc`` — nothing new in
``restrained_uff``.
"""

from __future__ import annotations

import numpy as np

from rxembed.log import logger

from .base import Constraints, add_distance, add_pairwise_shape

# window pads: fix is tight (a little slack — ETKDG needs it, not hard equality), constrain is a soft target
_FIX_PAD = 0.02  # numbers-fix distance half-window (Angstrom)
_FIX_ANG_PAD = 2.0  # numbers-fix angle half-window (degrees)
_CON_PAD = 0.1  # constrain distance half-window when a scalar target is given
_CON_ANG_PAD = 5.0  # constrain angle half-window
_SHAPE_PAD = 0.05  # graft pairwise-shape half-window (a rigid hold; the exact graft does the real work)
_MIN_SHAPE_ATOMS = 3  # below this a core has only a distance to pin, not an orientable 3-D shape
_COORD_LEN = 3  # an (x, y, z) coordinate
_DIST_ATOMS = 2  # a distance key names two atoms
_ANGLE_ATOMS = 3  # an angle key names three atoms


def resolve_atom(mol, ref):
    """Resolve an atom reference to an index (an int index, or a SMARTS matching one atom) — a query helper."""
    from rdkit import Chem

    if isinstance(ref, (int, np.integer)):
        return int(ref)
    hit = mol.GetSubstructMatch(Chem.MolFromSmarts(ref))
    if not hit:
        raise ValueError(f"SMARTS {ref!r} matched nothing")
    return hit[0]


def match(mol, smarts):
    """All atom indices of the first SMARTS match (for picking several reactive atoms at once) — a query helper."""
    from rdkit import Chem

    hit = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    if not hit:
        raise ValueError(f"SMARTS {smarts!r} matched nothing")
    return hit


def _window(val, pad):
    """Return a ``(lo, hi)`` window: a 2-sequence verbatim, a scalar as ``(val-pad, val+pad)``."""
    if isinstance(val, (tuple, list, np.ndarray)):
        vals = [float(v) for v in val]
        if len(vals) != _DIST_ATOMS:
            raise ValueError(
                f"a distance/angle value must be a scalar target or a (lo, hi) window; got {val!r} "
                f"(a coordinate belongs under an int key: fix={{i: (x, y, z)}})"
            )
        return (vals[0], vals[1])
    return (val - pad, val + pad)


def _is_index(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _index_tuple(key):
    """Return a pair/triple key as a tuple of ``int`` atom indices; error clearly if a SMARTS/str slipped in."""
    key = tuple(key)
    for a in key:
        if isinstance(a, str):
            raise ValueError(
                f"constraint key {key!r} contains the string {a!r}: fix=/constrain= are index-driven "
                f"(0-based, xyz/graph order) — resolve your SMARTS first, e.g. "
                f"i = mol.GetSubstructMatch(Chem.MolFromSmarts('[F-]'))[0], then key by i"
            )
        if not _is_index(a):
            raise ValueError(f"constraint key {key!r} must be integer atom indices, got {type(a).__name__}")
    return tuple(int(a) for a in key)


def _is_plane_key(key):
    """Return True for a π-stack key: a 2-tuple of ring atom-groups (lists/tuples/None), not two indices."""
    if not (isinstance(key, tuple) and len(key) == _DIST_ATOMS):
        return False
    return all(isinstance(e, (list, tuple)) or e is None for e in key)


def _as_coord(val):
    v = tuple(float(x) for x in val)
    if len(v) != _COORD_LEN:
        raise ValueError(f"a coordinate fix value must be (x, y, z); got {val!r}")
    return v


def _resolve_ring(mol, ring):
    """Return a π-stack ring atom-index tuple; ``None`` is not auto-detected — pass the atoms explicitly."""
    if ring is None:
        raise ValueError(
            "constrain plane: auto ring detection (None) is not supported — pass the ring atom indices "
            "explicitly, e.g. constrain={(tuple(ring_a), tuple(ring_b)): 3.7} "
            "(get them with mol.GetSubstructMatch / GetRingInfo)"
        )
    return tuple(int(a) for a in ring)


def _validate_indices(mol, atoms):
    n = mol.GetNumAtoms()
    for a in atoms:
        if not (0 <= a < n):
            raise ValueError(f"atom index {a} out of range (molecule has {n} atoms, 0..{n - 1})")


def resolve_core(mol, *, fix=None, constrain=None, template=None, has_geometry=False):
    """Resolve ``fix`` / ``constrain`` / ``template`` into ``(Constraints, ref)``.

    ``ref`` is ``{atom_idx: (x, y, z)}`` — the atoms to Kabsch-graft onto their exact geometry *after* the
    embed (empty when nothing is grafted: a numbers-only fix or a constrain-only spec). The caller edits the
    bounds matrix from `Constraints`, embeds, grafts ``ref``, then runs the restrained UFF pull.

    ``template`` is pre-resolved by the caller to ``(positions, {target_i: ref_i})`` — an ``N x 3`` array plus
    an explicit target→reference index map. It is folded into the coords-``fix`` path (same machinery).
    """
    cons = Constraints()
    coord_fix: dict[int, tuple] = {}  # atoms grafted to exact coords (own / explicit / template)
    d_soft: set = set()  # releasable soft distance keys (constrain) — provenance for mc(explore=)
    a_soft: set = set()  # releasable soft angle keys

    if template is not None:
        positions, mapping = template
        positions = np.asarray(positions, float)
        for ti, ri in mapping.items():
            if not (0 <= int(ri) < len(positions)):  # a bad (e.g. 1-based / off-by-one) reference index
                raise ValueError(
                    f"template map {ti!r}->{ri!r}: reference index {ri} out of range "
                    f"(reference has {len(positions)} atoms, 0..{len(positions) - 1})"
                )
            coord_fix[int(ti)] = tuple(positions[int(ri)])

    n_fix_d = n_fix_a = 0
    if fix is not None:
        if not isinstance(fix, (dict, list, tuple, set)):
            raise ValueError(
                f"fix= takes a list of atom indices (own-coords graft) or a dict "
                f"({{i: (x,y,z)}} coords / {{(i,j): d}} numbers); got {type(fix).__name__} {fix!r}"
                + (f" — did you mean fix=[{fix}]?" if _is_index(fix) else "")
            )
        if isinstance(fix, dict):
            for key, val in fix.items():
                if _is_index(key):  # int key -> a coordinate pin (graft)
                    coord_fix[int(key)] = _as_coord(val)
                else:  # tuple key -> a number: tight window, UFF-pulled to the target
                    idx = _index_tuple(key)
                    if len(idx) == _DIST_ATOMS:
                        add_distance(cons.distances, *idx, *_window(val, _FIX_PAD))
                        n_fix_d += 1
                    elif len(idx) == _ANGLE_ATOMS:
                        cons.angles[idx] = _window(val, _FIX_ANG_PAD)
                        n_fix_a += 1
                    else:
                        raise ValueError(f"fix number key {key!r} needs 2 atoms (distance) or 3 (angle)")
        else:  # a list/tuple/set of atom indices -> hold at the source's OWN coords (graft)
            if not has_geometry:
                raise ValueError(
                    "fix=[atoms] holds them at the source's own coordinates, but this source has no "
                    "geometry (a SMILES). Give numbers (fix={(i, j): d, (i, j, k): θ}) or a reference "
                    "(template=(ref, {target_i: ref_i}) / fix={i: (x, y, z)})."
                )
            positions = mol.GetConformer().GetPositions()
            for a in fix:
                if isinstance(a, str) or not _is_index(a):
                    raise ValueError(
                        f"fix=[...] takes atom indices; got {a!r}. Resolve SMARTS yourself first "
                        f"(fix=/constrain= are index-driven)."
                    )
                coord_fix[int(a)] = tuple(positions[int(a)])

    if coord_fix:  # a rigid graftable core: pairwise-shape bias + frozen + graft ref
        _validate_indices(mol, coord_fix)
        add_pairwise_shape(cons, list(coord_fix), coord_fix, _SHAPE_PAD)
        cons.frozen |= set(coord_fix)
        if len(coord_fix) < _MIN_SHAPE_ATOMS:
            logger.warning(
                "fix: %d atom(s) with coordinates — the embed frame is arbitrary, so a rigid oriented "
                "graft needs >=3 atoms; %s. Give >=3 core atoms for an exact placed pose.",
                len(coord_fix),
                "one atom fixes nothing (held wherever the embed lands it)"
                if len(coord_fix) == 1
                else "two atoms fix only the bond length",
            )

    if constrain:
        for key, val in constrain.items():
            if _is_plane_key(key):
                ra, rb = key
                cons.planes.append((_resolve_ring(mol, ra), _resolve_ring(mol, rb), float(val)))
                continue
            idx = _index_tuple(key)
            if len(idx) == _DIST_ATOMS:
                add_distance(cons.distances, *idx, *_window(val, _CON_PAD))
                d_soft.add((min(idx), max(idx)))
            elif len(idx) == _ANGLE_ATOMS:
                cons.angles[idx] = _window(val, _CON_ANG_PAD)
                a_soft.add(idx)
            else:
                raise ValueError(f"constrain key {key!r} needs 2 atoms (distance) or 3 (angle)")

    _validate_indices(
        mol,
        {a for k in cons.distances for a in k}
        | {a for k in cons.angles for a in k}
        | {a for ra, rb, _ in cons.planes for a in (*ra, *rb)},
    )
    cons.contacts = (frozenset(d_soft), frozenset(a_soft))  # only constrain= is releasable; fix is structural
    _warn_underdetermined(cons, coord_fix, n_fix_d)
    _echo(mol, cons, coord_fix)
    return cons, coord_fix


def _warn_underdetermined(cons, coord_fix, n_fix_d):
    """Advisory (invariant 5): two shared-atom fix distances over exactly 3 atoms leave the angle free.

    The classic mis-fix — an SN2 pinned as ``fix={(f, c): d1, (c, cl): d2}`` with no ``(f, c, cl)`` angle —
    is bent, not the linear TS the user meant. A richer network (>3 atoms, or any coords/angles) is left
    alone: it is a deliberate distance web, not the ambiguous 3-atom case. Advisory, not fatal.
    """
    if coord_fix or cons.angles or n_fix_d < _DIST_ATOMS:  # coords/angles orient; <2 distances make no angle
        return
    keys = list(cons.distances)
    atoms = {a for k in keys for a in k}
    shares = any(len(set(a) & set(b)) == 1 for i, a in enumerate(keys) for b in keys[i + 1 :])
    if len(atoms) <= _MIN_SHAPE_ATOMS and shares:
        logger.warning(
            "fix: two distances share an atom over %d atom(s) but no angle is fixed — the angle between "
            "them is free (a bent, not linear, core). Add the angle to fix={...} if it must be exact.",
            len(atoms),
        )


def _echo(mol, cons, coord_fix):
    """Echo what was pinned WITH element symbols (invariant 1) so a wrong (e.g. 1-based) index is obvious.

    Logged at INFO — this is the safety net for the commonest porting error (1-based indices, or picking a
    hydrogen): ``rx.set_verbose('INFO')`` then read back e.g. ``fixed d(C0, F5) -> 2.02 A`` and the atoms are
    checkable at a glance. A numbers-``fix`` is confirmed exactly with ``.measure()`` after the embed.
    """
    if not cons.is_constrained:
        return
    d_soft, a_soft = cons.contacts

    def s(i):
        return f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}"

    parts = []
    if coord_fix:
        parts.append(f"graft {', '.join(s(i) for i in sorted(coord_fix))}")
    for (i, j), (lo, hi) in cons.distances.items():
        if (i, j) in d_soft or (i in cons.frozen and j in cons.frozen):
            continue  # soft (below) or the frozen-core shape (structural, implied by the graft)
        parts.append(f"fix d({s(i)},{s(j)})->{lo:.2f}-{hi:.2f}A")
    for (i, j, k), (lo, hi) in cons.angles.items():
        if (i, j, k) in a_soft:
            continue
        parts.append(f"fix angle({s(i)},{s(j)},{s(k)})->{lo:.1f}-{hi:.1f}deg")
    for i, j in d_soft:
        lo, hi = cons.distances[(i, j)]
        parts.append(f"soft d({s(i)},{s(j)})->{lo:.2f}-{hi:.2f}A")
    for i, j, k in a_soft:
        lo, hi = cons.angles[(i, j, k)]
        parts.append(f"soft angle({s(i)},{s(j)},{s(k)})->{lo:.1f}-{hi:.1f}deg")
    for ra, rb, sep in cons.planes:
        parts.append(f"stack {len(ra)}x{len(rb)} ring @ {sep:.1f}A")
    logger.info("resolve: %s", "; ".join(parts))
