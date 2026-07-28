"""Coordination-isomer enumeration: the metal load-in that turns a SMILES / xyz into ready-to-embed `Isomer`s.

The shell layer over the metal kernel (`constraints.metal`): it composes the kernel's polytope tables,
surrogate, chirality tag and constraint builders into the distinct-arrangement enumeration and the
`IsomerSet` a caller selects from. It lives outside the kernel because it reaches the `stereo` shell to
expand an undefined *ligand* stereocentre — the two edges that kept `metal.py` bound to the shell.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from rxembed import inputs as _inputs
from rxembed.rdkit_embed.constraints.base import Constraints, compose
from rxembed.rdkit_embed.constraints.coordination_builders import coordination
from rxembed.rdkit_embed.constraints.distance import ff_terms
from rxembed.rdkit_embed.constraints.donor_orient import _FOLD_WINDOW, _stripped_hybridisation, donation_axis
from rxembed.rdkit_embed.constraints.metal import (
    _APICAL_MIN,
    _COLINEAR_TOL,
    _PAIR,
    _SPAN_ANGLE,
    _SPAN_TOL,
    _TRANS_ANGLE,
    _TRIAD,
    POLYHEDRA,
    TRANSITION_METALS,
    VACANT,
    Isomer,
    _collapse_haptic,
    _frag_map,
    _haptic_sites,
    _vertex_angle,
    _vertex_atom,
    arrange,
    chirality_of,
    classify_geometry,
    geometries_for_cn,
    geometry_for,
    hold_shape,
    isomer_permutations,
    logger,
    metal_indices,
    n_sites,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
    vertex_dirs,
)

_PT = GetPeriodicTable()


def _octahedral_triad(mol, od):
    """Return the vertex positions of a donor **triad** for which mer/fac is meaningful, else ``None``.

    A **tridentate chelate** (exactly 3 donors of one ligand fragment), else **exactly 3 monodentate**
    donors of one element (an MA3B3 set); ``None`` otherwise (then cis/trans is used). The exactly-3 and
    monodentate conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate
    (en2, en3) have no mer/fac, so neither must be forced into a triad.
    """
    real = [(p, od[p]) for p in range(len(od)) if od[p] != VACANT]
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
        if len(by_frag[frag[d]]) == 1:  # else exactly three MONODENTATE same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == _TRIAD:
            return tuple(ps)
    return None


def _order_label(mol, donors, geometry, order):
    """Build the isomer label from the IDEAL polyhedron (no conformer needed).

    Vacant vertices are ignored. Octahedral with a donor triad is **mer/fac** (one trans pair in the triad
    -> mer; none -> fac); otherwise **cis/trans** judged on the **minority same-element donor pair** -- the
    ligands whose placement defines the isomerism (the 2 Cl of an MA4B2, not the 4 A which always have a
    trans pair).
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return f"isomer{order}"
    if not POLYHEDRA[geometry].geometric_isomerism:
        return ""  # no cis/trans distinction for this geometry — a single arrangement
    od = [donors[k] for k in order]
    if geometry == "octahedral":
        tri = _octahedral_triad(mol, od)
        if tri is not None:
            trans = sum(
                1 for i in range(3) for j in range(i + 1, 3) if _vertex_angle(dirs[tri[i]], dirs[tri[j]]) > _TRANS_ANGLE
            )
            return "fac" if trans == 0 else "mer"
    by_elem = {}  # group vertex positions by donor element
    for p in range(len(od)):
        if od[p] != VACANT:
            by_elem.setdefault(mol.GetAtomWithIdx(od[p]).GetSymbol(), []).append(p)
    pairs = {e: ps for e, ps in by_elem.items() if len(ps) >= _PAIR}
    if not pairs:
        return "cis"  # all donors distinct: nothing to be cis/trans about
    e = min(pairs, key=lambda e: (len(pairs[e]), e))  # the minority same-element set defines cis/trans
    ps = pairs[e]
    trans = any(
        _vertex_angle(dirs[ps[i]], dirs[ps[j]]) > _TRANS_ANGLE for i in range(len(ps)) for j in range(i + 1, len(ps))
    )
    return "trans" if trans else "cis"


class IsomerSet(list):
    """The coordination isomers of a metal centre: a ``list`` of `Isomer` to iterate, index, or pick from.

    The identity is **geometric, not a chemistry name**: select on the per-vertex `arrangement`, the metal-centre
    `chirality` (``'Δ'``/``'Λ'``/``''``), the `geometry`, or the plain **index** — the cis/trans/mer/fac `label`
    is a coarse, sometimes-wrong tag, never required to select:

        isos = rx.metal('CCCN[Pd](Cl)(Cl)NCCC', ['square_planar', 'tetrahedral']); isos.summary()
        ens  = rx.embed(isos.select(arrangement='N3 Cl6 N7 Cl5')).mc().prune()
        ens  = rx.embed(isos[0]).mc().prune()

    Enumeration is cheap; the expensive MC search runs only on the `Isomer` you pick.
    """

    def select(self, geometry=None, label=None, arrangement=None, chirality=None, index=None, stereo=None):
        """Return the single `Isomer` matching the given keys.

        Key on `arrangement` (the unambiguous per-vertex slot map), `chirality` (``'Δ'``/``'Λ'``/``''``),
        `geometry`, `index`, the ligand `stereo` tag (e.g. ``'16R'``), or the coarse `label`. **Raises** if
        zero or several match — the message lists every isomer so you can narrow it.
        """
        hits = self.filter(
            geometry=geometry, label=label, arrangement=arrangement, chirality=chirality, index=index, stereo=stereo
        )
        if len(hits) != 1:
            have = [(k, i.geometry, i.chirality or "-", i.stereo_label or "-", arrange(i)) for k, i in enumerate(self)]
            raise ValueError(
                f"select(geometry={geometry!r}, label={label!r}, arrangement={arrangement!r}, "
                f"chirality={chirality!r}, index={index!r}, stereo={stereo!r}) matched {len(hits)} isomer(s) — "
                f"{'narrow it or pick by index' if hits else 'no match'}; have {have}"
            )
        return hits[0]

    def filter(self, geometry=None, label=None, arrangement=None, chirality=None, index=None, stereo=None):
        """Return the subset matching the given keys, as an `IsomerSet` (keep several / pick by index).

        `label` matches the **base** tag (``'fac'`` also matches auto-numbered ``fac1``/``fac2``);
        `arrangement`/`chirality`/`geometry`/`stereo` (the ligand stereoisomer tag) match exactly;
        `index` selects positionally.
        """

        def ok(k, i):
            return (
                geometry in (None, i.geometry)
                and (index is None or index == k)
                and (arrangement is None or arrange(i) == arrangement)
                and (chirality is None or i.chirality == chirality)
                and (stereo is None or i.stereo_label == stereo)
                and (label is None or i.label == label or i.label.rstrip("0123456789") == label)
            )

        return IsomerSet(i for k, i in enumerate(self) if ok(k, i))

    def summary(self):
        """Print each isomer (index, geometry, per-vertex arrangement, metal chirality) so you can pick one.

        The arrangement (which donor sits at which vertex) is the unambiguous identity; chirality is
        ``'Δ'``/``'Λ'`` or ``'(achiral)'``. The coarse cis/trans/mer/fac name is deliberately *not* shown —
        select on ``arrangement=`` / ``chirality=`` / index. Returns self (chainable).
        """
        for k, i in enumerate(self):
            stereo = f"  stereo {i.stereo_label}" if i.stereo_label else ""
            print(f"  [{k}] {i.geometry:16s} {arrange(i):26s} {i.chirality or '(achiral)'}{stereo}")
        return self


def _resolve_center(mol, metals, center):
    """Pick which transition metal to enumerate.

    `center` is None (the sole metal -- else an error asking you to choose), an **atom index**, or an
    **element symbol** (e.g. ``'Mn'``).
    """
    if center is None:
        if len(metals) == 1:
            return metals[0]
        raise ValueError(
            f"{len(metals)} transition metals present "
            f"({[mol.GetAtomWithIdx(x).GetSymbol() + str(x) for x in metals]}) — choose which to "
            f"enumerate with center=<atom index or element symbol>"
        )
    if isinstance(center, (int,)) and not isinstance(center, bool):
        if center not in metals:
            raise ValueError(f"center={center} is not a transition-metal atom; metals are at {metals}")
        return center
    hits = [x for x in metals if mol.GetAtomWithIdx(x).GetSymbol() == center]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no {center!r} centre; metals present: {[mol.GetAtomWithIdx(x).GetSymbol() for x in metals]}")
    raise ValueError(f"{len(hits)} {center} centres ({hits}) — disambiguate with center=<atom index>")


def _load_in_ligand_stereo(mol, geometry, center, fix, stereo):
    """Expand any UNDEFINED ligand stereocentre of a coordinate-free input, recursing per variant with stereo='free'.

    Returns the enumerated `IsomerSet` (coordination x ligand stereo), or ``None`` when there is nothing to load
    in (stereo='free', a geometry input, or no undefined centre) so the caller runs the normal path.
    """
    if stereo == "free" or mol.GetNumConformers() != 0:
        return None
    from rxembed import stereo as _stereo  # UNDEFINED ligand stereocentre so the set spans coordination x

    variants, n_unassigned, _total, unresolved = _stereo.enumerate_unassigned(mol, exclude=set(metal_indices(mol)))
    # exclude the metal's own centre — its Λ/Δ is enumerated below, not as RDKit point stereo
    if not n_unassigned:
        return None
    out = IsomerSet()
    for vmol, slabel in variants:  # each variant is stereo-DEFINED -> recurse with stereo='free' so its
        for iso in enumerate_isomers(vmol, geometry, center, fix, stereo="free"):  # own load-in is a no-op
            iso.stereo_label = slabel
            out.append(iso)
    logger.info(
        "metal: %d undefined ligand stereocentre(s) -> enumerating coordination x %d ligand "
        "stereoisomer(s) = %d candidate(s) (select stereo=)",
        n_unassigned,
        len(variants),
        len(out),
    )
    if unresolved:
        logger.warning(
            "metal: %d ligand stereo axis(es) (allene/atropisomer) cannot be enumerated from a flat "
            "SMILES — embedded as a single arbitrary hand; pass a geometry to fix it",
            unresolved,
        )
    return out


def _prepare_spectators(mol, metals, center):
    """Surrogate a multi-metal complex, enumerating `center` while holding each spectator metal's shape.

    Returns ``(base, m, donors, real_z, real_q, extra, retain, spectator_bonds)``. A spectator keeps its oxidation
    state and gets the same force field as the enumerated centre (its bond-less carbon fires the same fictitious
    LJ); ``spectator_bonds`` are its stripped M-donor pairs, re-added DATIVE on the output so it too is connected.
    """
    m = _resolve_center(mol, metals, center)
    spectators = [s for s in metals if s != m]
    if spectators and mol.GetNumConformers() == 0:
        raise ValueError(
            f"enumerating one centre of a {len(metals)}-metal complex needs an input "
            f"geometry (an .xyz) to retain the other metal(s) — got a coordinate-free input"
        )

    def non_metal(nbrs):
        return [a.GetIdx() for a in nbrs if a.GetAtomicNum() not in TRANSITION_METALS]

    donors = non_metal(mol.GetAtomWithIdx(m).GetNeighbors())  # a partner metal is not a coordination donor
    if any(len(s) > 1 for s in _haptic_sites(mol, donors)):  # a haptic face on the enumerated centre is only
        raise NotImplementedError(  # collapsed in the single-metal path; fail loud, don't mis-count donors
            "the enumerated centre carries a haptic face (eta2 / Cp / arene), which is only supported for a "
            "single-metal complex — a multi-metal complex with a haptic centre is not yet handled (the ring "
            "atoms would be mis-counted as separate sigma donors). Enumerate the single-metal fragment instead"
        )
    real_z = mol.GetAtomWithIdx(m).GetAtomicNum()
    real_q = mol.GetAtomWithIdx(m).GetFormalCharge()  # oxidation state — restored so xtb gets the right charge
    spec = {
        s: (
            non_metal(mol.GetAtomWithIdx(s).GetNeighbors()),
            mol.GetAtomWithIdx(s).GetAtomicNum(),
            mol.GetAtomWithIdx(s).GetFormalCharge(),  # a spectator keeps its oxidation state too
        )
        for s in spectators
    }
    base, _metals_info, _ = surrogate_all_metals(mol)  # surrogate EVERY metal (conformer preserved)
    extra = [(s, spec[s][1], spec[s][2]) for s in spectators]
    retain = Constraints()  # hold each spectator's SHAPE (relative pairwise,
    for s in spectators:  # frame-independent) — achiral, so chirality stays
        hold_shape(base, [s, *spec[s][0]], retain)  # random and is fixed by select_stereo afterward
    # A spectator is still a metal: its bond-less carbon surrogate fires the same fictitious Lennard-Jones at
    # every ligand around it, so it gets the same force field as the metal being enumerated.
    ff_terms(base, retain, {s: (spec[s][1], list(spec[s][0])) for s in spectators})
    logger.info(
        "metal: enumerating %s%d; holding %d spectator metal(s) by %d shape constraints",
        mol.GetAtomWithIdx(m).GetSymbol(),
        m,
        len(spectators),
        len(retain.distances),
    )
    spectator_bonds = [(d, s) for s in spectators for d in spec[s][0]]  # re-added DATIVE so the spectator connects too
    return base, m, donors, real_z, real_q, extra, retain, spectator_bonds


def _select_geometries(base, m, donors, haptic, geometry, n):
    """Resolve `geometry` to the list of polyhedron names to enumerate (default from donor count, or as given)."""
    if geometry is None:
        # an APICAL (eta>=3) face fills more than one site -> a CN4 carrying one is a piano stool, not square_planar
        # (which would seat a ligand trans through the ring). An eta2 face is a single-site vertex: no flip.
        apical = any(len(r) >= _APICAL_MIN for r in haptic.values())
        measured = None
        if base.GetNumConformers() and not apical:  # a retained geometry names itself; an apical face is a
            sites = [haptic.get(v, v) for v in donors]  # site-count question the templates do not model
            measured = classify_geometry(base, m, sites)
        geoms = [measured or geometry_for(n, has_apical=apical)]
        if geoms == [None]:
            raise ValueError(
                f"no default geometry for {n} donors — pass geometry= a name or list "
                f"(options for {n} donors: {[p.name for p in geometries_for_cn(n)]})"
            )
    else:
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
    for g in geoms:
        if g not in POLYHEDRA:
            hint = " — pass a list of names, e.g. ['square_planar', 'tetrahedral']" if g == "all" else ""
            raise ValueError(f"unknown geometry {g!r}; available: {sorted(k for k in POLYHEDRA if k != 'None')}{hint}")
    return geoms


def _frozen_permutations(base, m, padded, geom, frozen_donors, sites):
    """Generate every free-donor vertex permutation with each frozen donor pinned at its input vertex.

    Returns the explicit permutation list (bypassing the symmetry-reduced canned `isomer_permutations`, which would miss
    the representative ordering a valid frozen isomer needs), or ``None`` if the input vertex ordering can't be read.
    """
    base_order = _input_ordering(base, m, padded, geom)
    if base_order is None:
        return None
    frozen_v = {v: di for v, di in enumerate(base_order) if padded[di] in frozen_donors}
    free_v = [v for v in range(sites) if v not in frozen_v]
    free_di = [di for di in range(len(padded)) if padded[di] not in frozen_donors]
    perms = []
    for fp in itertools.permutations(free_di):
        o = [None] * sites
        for v, di in frozen_v.items():
            o[v] = di
        for v, di in zip(free_v, fp, strict=False):
            o[v] = di
        perms.append(o)
    logger.info(
        "metal[%s]: %d frozen donor(s) pinned at input vertices; %d free-site arrangement(s) to dedup",
        geom,
        len(frozen_v),
        len(perms),
    )
    return perms


def _isomers_for_geometry(
    base,
    geom,
    *,
    m,
    donors,
    real_z,
    real_q,
    haptic,
    frozen_donors,
    fix_cons,
    retain,
    extra,
    ref_sig,
    spectator_bonds=(),  # each held spectator metal's stripped M-donor pairs, re-added DATIVE on the output mol
):
    """Enumerate every distinct `Isomer` of one polyhedron `geom` (frozen core held, spectators retained)."""
    n = len(donors)
    sites = n_sites(geom)
    if n > sites:
        raise ValueError(f"{geom} has {sites} coordination sites but the metal has {n} donors")
    padded = list(donors) + [VACANT] * (sites - n)  # leave empty vertices as a pocket
    if sites - n:
        logger.info(
            "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)", geom, sites, n, sites - n
        )
    perms = None
    if frozen_donors and sites == n:  # pin each frozen donor at its input vertex and generate every free-donor
        perms = _frozen_permutations(base, m, padded, geom, frozen_donors, sites)  # arrangement around it
    out = []
    for order in distinct_vertex_orderings(
        base, padded, geom, perms=perms, r_metal=_PT.GetRcovalent(real_z), haptic=haptic
    ):
        cons = coordination(
            base, m, padded, geom, order, real_z, frozen=frozen_donors, core_frozen=fix_cons.frozen, haptic=haptic
        )
        cons = compose(cons, retain)  # hold any spectator metal(s)' shape and give it the same force field
        #   (zero-vdW + both halves of every floor). Field-driven, so a floor can never arrive without its
        #   `dg_floors` twin — which would leave RDKit's phantom carbon-vdW floor (~3.4 A) standing on every
        #   pair of the spectator's sphere while the force field floors the same pair at its real distance.
        cons.distances.update(fix_cons.distances)  # hold the frozen reacting core (relative pairwise)
        cons.frozen |= fix_cons.frozen  # + pin it in the relax
        od = [padded[k] for k in order]  # vertex -> donor atom (or VACANT); a centroid dummy for an eta>=3 face
        # The Isomer is REAL: the centroid dummy is transient embed scaffolding (`bounds.embed`/`restrained_uff`
        # materialise it from `cons.haptic`), so strip it from the stored mol and report the real coordinating
        # atoms (sigma donors + every ring atom) as `donors`. `vertices`/`chirality` keep the centroid index —
        # they describe the polyhedron, and `base` (which still has it) is where chirality is read.
        real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
        out.append(
            Isomer(
                strip_phantoms(Chem.Mol(base), set(haptic)),
                cons,
                m,
                real_donors,
                real_z,
                real_q,
                _order_label(base, padded, geom, order),
                geom,
                od,
                chirality=chirality_of(base, donors, geom, od, haptic),
                extra=extra,
                stereo_ref=ref_sig,
                haptic=dict(haptic),
                # centre's real donors (sigma + every haptic ring atom) + each spectator's — re-added DATIVE on output
                donor_bonds=[(d, m) for d in real_donors] + list(spectator_bonds),
            )
        )
    return out


def _number_shared_labels(out):
    """Disambiguate isomers that share a (geometry, label) by numbering them cis1, cis2… — each uniquely selectable."""
    counts = Counter((i.geometry, i.label) for i in out)  # several heteroleptic isomers can share a label
    nth = Counter()
    for i in out:
        key = (i.geometry, i.label)
        if counts[key] > 1:
            nth[key] += 1
            i.label = f"{i.label}{nth[key]}"


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo="racemic"):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    Returns an `IsomerSet` (pick with ``IsomerSet.select(…)``). The metal **load-in**: for a coordinate-free
    input `stereo='racemic'` also enumerates any UNDEFINED *ligand* stereocentre, so the set spans coordination
    x ligand-stereo (each `Isomer` carries a `.stereo_label`); `stereo='free'` opts out; a geometry input is
    untouched.

    `geometry` selects the polyhedron(a): ``None`` → the donor-count default; a name → just that one; a **list**
    → each (to compare energies yourself). Supported names come from ``geometries_for_cn``. `center=` picks which
    metal to enumerate in a multi-metal complex (the others retained at the input geometry). **Vacant sites:** a
    geometry with more vertices than donors leaves the empty vertex as a coordination pocket.
    """
    if isinstance(mol, str):
        if mol.lower().endswith(".xyz"):  # a path -> perceived Mol with a geometry, so rx.metal('complex.xyz',
            mol = _inputs._xyz_to_mol(mol, 0)  # center=…) works and not just rx.embed; needed anyway to retain a
        else:  # spectator metal
            mol = Chem.AddHs(_inputs.parse_smiles(mol))  # clear error on a bad SMILES, not a cryptic AddHs(None)
    loaded = _load_in_ligand_stereo(mol, geometry, center, fix, stereo)
    if loaded is not None:
        return loaded
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no transition metal found")
    ref_sig = None  # chirality fingerprint of the INPUT geometry
    if mol.GetNumConformers() > 0:  # (real metals) — lets stereo='preserve' hold a
        from rxembed import stereo as _stereo  # spectator's planar/axial/helical handedness

        try:
            ref_sig = _stereo.signature(mol)
        except Exception:
            ref_sig = None
    haptic = {}  # haptic centroid dummies, keyed dummy -> its face atoms (set by _collapse_haptic in the single path)
    if len(metals) == 1:  # one metal -> no spectators, so this is the single-centre path whether or not center= names
        if center is not None:  # it. Validate an explicit center= against the sole metal (raises on a wrong idx/elem);
            _resolve_center(mol, metals, center)  # a haptic face here is collapsed, never sent to the multi-metal else
        base, m, donors, real_z, real_q = surrogate_metal(mol)  # the common single-metal case, unchanged
        base, donors, haptic = _collapse_haptic(base, donors)  # collapse each haptic face to one centroid vertex
        extra, retain, spectator_bonds = [], Constraints(), []
    else:
        base, m, donors, real_z, real_q, extra, retain, spectator_bonds = _prepare_spectators(mol, metals, center)
    fix_cons = Constraints()
    frozen_donors = set()
    if fix:  # hold a reacting TS core at the input geometry
        # while the rest of the coordination sphere is
        from rxembed.rdkit_embed.constraints.builders import resolve_core

        if base.GetNumConformers() == 0:  # enumerated (mer/fac of a tridentate while a
            raise ValueError(
                "fix= needs an input geometry (an .xyz / a Mol with a conformer) to hold "
                "the reacting core; got a coordinate-free input (e.g. a SMILES)"
            )
        fix_cons, _ = resolve_core(base, fix=fix, has_geometry=True)  # reacting donor + substrate stay put)
        frozen_donors = fix_cons.frozen & set(donors)  # while a reacting donor + substrate stay put)
        logger.info(
            "metal: fixing %d atom(s) at the input geometry; enumerating the free coordination sites around them",
            len(fix_cons.frozen),
        )
    geoms = _select_geometries(base, m, donors, haptic, geometry, len(donors))
    out = IsomerSet()
    for geom in geoms:
        out.extend(
            _isomers_for_geometry(
                base,
                geom,
                m=m,
                donors=donors,
                real_z=real_z,
                real_q=real_q,
                haptic=haptic,
                frozen_donors=frozen_donors,
                fix_cons=fix_cons,
                retain=retain,
                extra=extra,
                ref_sig=ref_sig,
                spectator_bonds=spectator_bonds,
            )
        )
    _number_shared_labels(out)
    return out


def _input_ordering(mol, metal, donors, geometry):
    """Find the vertex ordering that best matches the **input geometry** (which donor at which vertex).

    ``od[vertex] = donors[order[vertex]]``, read from the conformer (orthogonal Procrustes over the
    candidate vertex orderings). Lets a ``fix=`` hold each frozen donor at its *real* vertex so only the
    free sites are enumerated -- otherwise the enumeration permutes a frozen donor into a vertex it cannot
    occupy, producing isomers that contradict the frozen core (a hydride forced off its TS site).
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([pos[d] - pos[metal] for d in donors], float)
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    v_ideal = np.array(dirs_ref, float)
    best_score, best_order = -1.0, list(range(len(donors)))
    for order in isomer_permutations(geometry) or [tuple(range(len(donors)))]:  # CN7/8 have a template but no canned
        h = dd[list(order)].T @ v_ideal  # cross-covariance of (donor-at-vertex) vs ideal vertex
        score = float(np.linalg.svd(h, compute_uv=False).sum())  # max alignment after the optimal rotation
        if score > best_score:
            best_score, best_order = score, order
    return best_order


def _central_trans(od, frag, dmat, dirs, haptic=None):
    """Return True if a tridentate chelate's *central* donor is placed **trans** to one of its own arms.

    The central donor is the one on the backbone path *between* the other two (``d(a,c)+d(c,b)==d(a,b)``), which
    is impossible for a pincer (central is cis to both arms in every real mer/fac); a flexible chelate can stretch
    to ~155° without a formally torn bond, so this drops it at enumeration rather than leave `bonding_ok` to. A
    haptic vertex is resolved to its ring atom so its backbone path is real.
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
                if _vertex_angle(dirs[c], dirs[a]) > _TRANS_ANGLE or _vertex_angle(dirs[c], dirs[b]) > _TRANS_ANGLE:
                    return True
                break
    return False


def _span_bounds(mol):
    """Return the ligands' own distance-geometry bounds matrix, or None if it can't be built (→ never filter).

    RDKit's ``GetMoleculeBoundsMatrix`` gives bounds derived from the ligand's own connectivity + knowledge —
    i.e. how far *this* backbone can actually reach (the chemistry of the isolated ligand; the metal is
    bond-less here, so every path it bounds runs through the backbone, never across the centre). Read by the
    trans-span gate as ``bm[i][j]`` for a pair's **upper** bound and ``bm[j][i]`` for its lower (i < j).
    """
    try:
        from rdkit.Chem import rdDistGeom

        return rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except Exception:  # pragma: no cover — bounds matrix is robust, but never let the gate crash enumeration
        return None


def _reach(bm, i, j):
    """Return the pair's upper-bound (max reachable) distance — the bounds matrix's i<j triangle."""
    return float(bm[min(i, j)][max(i, j)])


def _donor_faces_metal(mol, d, other, d_md, d_mo, need, bm, hyb, donors):
    """Return False if donor ``d``, stretched to span *trans*, cannot still point its lone pair at the metal.

    The reach test asks only whether the donors can get ``need`` apart; this asks whether the backbone that
    achieves it can still *donate*. At a trans span the metal lies on the D···D' line, so the fold ruler's M-D-X
    angle is fixed by the ligand's own X-D···D' angle — and the anti backbone that reaches furthest points D's
    substituent straight at the metal. Using the census fold **floor** as the minimum M-D-X, each heavy X is
    forced out to a matching X···D' distance; if the backbone can't reach that, no conformer spans and donates.

    Abstains via `donation_axis` (a hydride/bridging/haptic donor has no axis) and for an uncalibrated (element,
    hyb) class — from the ruler's own rules, not a copy.
    """
    subs = donation_axis(mol, d, donors)
    if subs is None:  # hydride / bridging / haptic: no donation axis, so "does it face the metal" is meaningless
        return True
    cls = (mol.GetAtomWithIdx(d).GetSymbol(), hyb[d]) if d in hyb else None
    if cls not in _FOLD_WINDOW:  # estimators disagreed, or n < 6 for the class — the ruler abstains, so do we
        return True
    beta = math.degrees(  # angle(M, D, D') in the M-D-D' triangle: how far off the D···D' line the metal sits
        math.acos(max(-1.0, min(1.0, (d_md**2 + need**2 - d_mo**2) / (2 * d_md * need))))
    )
    alpha = _FOLD_WINDOW[cls][0] - beta  # the X-D···D' angle the fold floor forces (M is `beta` off that line)
    if alpha <= 0:  # the metal already sits far enough off the line — the floor costs the backbone nothing
        return True
    for x in subs:
        r = float(bm[max(x, d)][min(x, d)])  # the D-X bond length (a bond's bounds coincide to < 0.05 Å)
        out = math.sqrt(r**2 + need**2 - 2 * r * need * math.cos(math.radians(alpha)))  # X···D' this forces
        if _reach(bm, x, other) < out - _SPAN_TOL:  # the backbone can't hold X that far off the metal
            return False
    return True


def _chelate_span_ok(mol, od, frag, dirs, bm, r_metal, hyb, donors, haptic=None):
    """Return False if a chelate is placed *trans* across the metal its backbone can't reach — or can't donate to.

    A cis bidentate always folds in, so only a **wide** separation (`_SPAN_ANGLE`+, trans) is tested. Two
    questions, both from the ligand's own bounds matrix so a genuine long-bridge ligand that *can* span trans is
    allowed (geometry, not a topological guess). Generalises `_central_trans` to any denticity.

    * **Reach** — the donors would sit on opposite sides, needing a donor-donor distance
      ``law_of_cosines(d_Ma, d_Mb, θ)`` a short backbone cannot span.
    * **Orientation** (`_donor_faces_metal`) — reach models where the donors *are*, not where they *point*: an
      extended anti backbone aims both lone pairs along the chain (without this a diphosphine passed reach and
      relaxed to P-M-P 155° against its 74-104° bite).
    """
    if bm is None:  # no bounds matrix -> never filter; defer to `bonding_ok` downstream
        return True
    for p in range(len(od)):
        for q in range(p + 1, len(od)):
            a, b = od[p], od[q]
            # A haptic centroid is bond-less: resolve each vertex to a representative RING atom so the fragment
            # (same-ligand) test, the covalent reach, and the backbone bounds all read the face's real chemistry —
            # else a face tethered to a co-donor reads as a separate ligand and its impossible trans is never dropped.
            ra, rb = _vertex_atom(haptic, a), _vertex_atom(haptic, b)
            if VACANT in (a, b) or frag[ra] != frag[rb]:  # only a same-ligand (chelate) pair
                continue
            theta = _vertex_angle(dirs[p], dirs[q])
            if theta < _SPAN_ANGLE:  # cis / adjacent -> the chelate folds in, always feasible
                continue
            d_ma = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(ra).GetAtomicNum())  # real M-donor covalent sums,
            d_mb = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(rb).GetAtomicNum())  # not a fixed 2.0 (Pd-N ~2.1)
            need = math.sqrt(d_ma**2 + d_mb**2 - 2 * d_ma * d_mb * math.cos(math.radians(theta)))  # law of cosines
            if _reach(bm, ra, rb) < need - _SPAN_TOL:  # backbone can't reach
                return False
            # The orientation test asks whether the donor can still aim its lone pair at the metal — meaningless for
            # a haptic face (it donates a π face, no axis), so a centroid vertex abstains (as `donation_axis` does).
            if a not in (haptic or {}) and not _donor_faces_metal(mol, ra, rb, d_ma, d_mb, need, bm, hyb, donors):
                return False
            if b not in (haptic or {}) and not _donor_faces_metal(mol, rb, ra, d_mb, d_ma, need, bm, hyb, donors):
                return False
    return True


def _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, *, limit=None):
    """Dedup vertex orderings by their connectivity-aware signature; stop once `limit` distinct ones are found.

    Two geometric impossibilities are pruned: a tridentate's central donor trans to its own arm (`_central_trans`),
    and a chelate placed trans whose backbone can't reach or, stretched, can't donate inward (`_chelate_span_ok`).
    A cis chelate is always kept; any residual is dropped downstream by `bonding_ok`.

    The dedup signature carries per donor pair the elements, the vertex angle, and a same-ligand pair's
    intra-ligand bond distance, plus the centre's Λ/Δ chirality — so enantiomers and a tridentate's mer vs
    central-trans stay distinct. `limit` short-circuits at the k-th distinct arrangement (the warning caller needs
    only "more than one?").
    """
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != VACANT else "X") for d in donors}  # vacancy = "X"
    frag = _frag_map(mol)  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances
    real_donors = [d for d in donors if d != VACANT]
    bm = _span_bounds(mol)
    hyb = _stripped_hybridisation(mol)  # the fold ruler's own (element, hyb) class — graph-only, no coords
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))]
    angle = {(p, q): _vertex_angle(dirs[p], dirs[q]) for p, q in pairs}  # a vertex-pair's angle is donor-independent

    def link(od, p, q):  # intra-ligand bond distance of a same-ligand pair
        a, b = _vertex_atom(haptic, od[p]), _vertex_atom(haptic, od[q])  # resolve a centroid to its ring atom
        if VACANT in (od[p], od[q]) or frag[a] != frag[b]:  # (distinguishes a chelate's central vs terminal
            return -1  # donor); -1 for different ligands / a vacancy
        return int(dmat[a][b])

    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        if _central_trans(od, frag, dmat, dirs, haptic):  # a tridentate's central donor trans to its arm -> impossible
            continue
        if not _chelate_span_ok(mol, od, frag, dirs, bm, r_metal, hyb, real_donors, haptic):  # can't span/donate trans
            continue
        sig = tuple(
            sorted((tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), angle[(p, q)]) for p, q in pairs)
        )
        sig = (sig, chirality_of(mol, real_donors, geometry, od, haptic))  # keep Λ/Δ enantiomers distinct (else merged)
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if limit is not None and len(out) >= limit:
                break
    return out


def distinct_vertex_orderings(mol, donors, geometry, perms=None, r_metal=1.4, haptic=None):
    """Enumerate distinct coordination isomers: **every** distinct vertex arrangement, minimally pre-filtered.

    Dedup and the two feasibility pre-filters live in `_distinct_orderings`. `perms` overrides the candidate
    vertex orderings (default ``isomer_permutations(geometry)``); a ``fix=`` enumeration passes the subset that
    keeps each frozen donor pinned to its input vertex.
    """
    dirs = vertex_dirs(geometry)
    if perms is None and isomer_permutations(geometry) is None:  # CN7/8, tetrahedral, linear: no canned list, so
        # the input ordering is the only candidate. Warn ONLY when it *loses* something — when these donors admit
        # more than one distinct arrangement we are not expanding (a linear / all-identical geometry has exactly
        # one, so the warning would be noise). The count comes from the dedup below, short-circuited at the second.
        orbit = itertools.permutations(range(len(donors)))
        if (
            dirs is not None
            and len(_distinct_orderings(mol, donors, geometry, orbit, dirs, r_metal, haptic, limit=2)) > 1
        ):
            logger.info(
                "metal[%s]: these donors admit more than one distinct site arrangement but no coordination-isomer "
                "permutations are tabulated for this geometry — enumerating the single input/identity ordering only "
                "(the geometry still embeds; the other arrangement(s) are not expanded)",
                geometry,
            )
        return [list(range(len(donors)))]  # UNFILTERED: the pre-filters prune an *enumeration*, so with a single
        # ordering there is nothing to prefer it over and filtering could only return empty — breaking the "the
        # geometry still embeds" contract and making `rx.metal(...)[0]` raise IndexError. `bonding_ok` + the
        # geometry gate judge it downstream.
    perms = perms if perms is not None else isomer_permutations(geometry)
    if dirs is None:
        return perms
    return _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic)
