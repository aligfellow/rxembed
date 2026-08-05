"""The `Isomer` itself and the metal load-in that turns a `Mol` into ready-to-embed `Isomer`s.

Three doors onto one class: ``Isomer(mol, geometry, sites)`` for a known arrangement, `from_geometry` to
retain the input's own, and `enumerate_isomers` for the unknown ones as an `IsomerSet`.

This is the layer over the metal engine, composing the polytope tables, surrogate, chirality tag and
constraint builders into the distinct-arrangement enumeration a caller selects from. Parsing is the
consumer's job: this takes a `Mol`, and the input geometry's chirality fingerprint arrives as an opaque
`stereo_ref` the pipeline computes and only the pipeline reads.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdDistGeom

from . import stereo as _stereo
from .constraints import Constraints, _is_index, compose, resolve_core
from .metal_coordination import coordination, coordination_from_geometry, resolve_lengths
from .metal_core import (
    _APICAL_MIN,
    _COLINEAR_TOL,
    _PAIR,
    _SPAN_ANGLE,
    _SPAN_TOL,
    _TRANS_ANGLE,
    _TRIAD,
    COORDINATION_METALS,
    VACANT,
    _collapse_haptic,
    _frag_map,
    _haptic_sites,
    _vertex_atom,
    chirality_of,
    classify_geometry,
    geometry_for,
    hold_shape,
    label,
    logger,
    metal_indices,
    n_sites,
    restore_metal,
    strip_phantoms,
    surrogate_all_metals,
    surrogate_metal,
)
from .metal_distance import ff_terms
from .metal_donor_orient import _FOLD_WINDOW, _stripped_hybridisation, donation_axis
from .metal_polyhedron import (
    POLYHEDRA,
    _fit_trace,
    _seat_by_alignment,
    _vertex_angle,
    chirality_tag,
    describe,
    geometries_for_cn,
    isomer_permutations,
    read_slot_note,
    resolve_geometry,
    seat_properly,
    vertex_dirs,
)

_PT = GetPeriodicTable()


def _say_length_source(mol, lengths):
    """Report where the M-donor windows came from, when that is not the default anyone would assume."""
    note = resolve_lengths(mol, lengths)[1]
    if note:
        logger.info("metal: M-donor windows from %s", note)


def _octahedral_triad(mol, od):
    """Return the vertex positions of a donor triad for which mer/fac is meaningful, else ``None``.

    Either a tridentate chelate (exactly 3 donors of one ligand fragment) or exactly 3 monodentate donors of
    one element (an MA3B3 set); ``None`` otherwise, and then cis/trans is used. The exactly-3 and monodentate
    conditions matter: MA4B2 (4 of an element) is cis/trans not mer/fac, and bis-/tris-bidentate (en2, en3)
    have no mer/fac, so neither must be forced into a triad.
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
        if len(by_frag[frag[d]]) == 1:  # else exactly three monodentate same-element donors
            by_elem.setdefault(mol.GetAtomWithIdx(d).GetSymbol(), []).append(p)
    for ps in by_elem.values():
        if len(ps) == _TRIAD:
            return tuple(ps)
    return None


def _order_label(mol, donors, geometry, order):
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


# --- the Isomer itself: the known-isomer front door, and the retain-the-input builder ------------------


def _seat_order(padded, haptic, sites):
    """Map a `sites` (vertex -> donor ATOM index) onto the internal order (vertex -> position in `padded`).

    `sites` is a ``{vertex: atom}`` dict or a vertex-ordered list. A haptic face is one vertex, named by any
    of its ring atoms. Vertices left unnamed take the VACANT padding (a coordination pocket). Every failure is
    loud, because an unknown, repeated or unseated donor would otherwise embed a different isomer silently.
    """
    if not isinstance(sites, dict):
        if not isinstance(sites, (list, tuple, np.ndarray)):  # a set seats every donor exactly once, so the
            raise TypeError(  # guard below passes, and its iteration order then picks an arbitrary isomer
                f"sites must be a {{vertex: atom}} dict or a vertex-ordered list, got {type(sites).__name__}: "
                f"an unordered collection cannot say which donor sits at which vertex"
            )
        sites = {v: a for v, a in enumerate(sites) if a is not None and a != VACANT}
    slot = {d: k for k, d in enumerate(padded) if d != VACANT}
    for dummy, ring in haptic.items():
        slot.update(dict.fromkeys(ring, slot[dummy]))  # a face is named by any ring atom, not by its centroid dummy
    n = len(padded)
    order = [None] * n
    for v, a in sites.items():
        if not _is_index(v) or not 0 <= v < n:  # perception hands numpy ints; a bool is not a vertex
            raise ValueError(f"vertex {v!r} is not one of this geometry's {n} vertices (0-{n - 1})")
        if a not in slot:
            raise ValueError(f"sites[{v}] = atom {a} is not a donor of the metal; its donors are {sorted(slot)}")
        if order[v] is not None:
            raise ValueError(f"vertex {v} is given two donors ({padded[order[v]]} and {a})")
        if slot[a] in order:
            raise ValueError(f"donor atom {a} is seated at two vertices")
        order[v] = slot[a]
    spare = [k for k in range(n) if k not in order]
    unseated = [padded[k] for k in spare if padded[k] != VACANT]
    if unseated:
        raise ValueError(f"donor(s) {unseated} were given no vertex; every donor must be seated ({len(sites)} given)")
    for v in range(n):
        if order[v] is None:
            order[v] = spare.pop(0)  # a VACANT padding slot
    return order


def _warn_undefined_ligand_stereo(mol):
    """Warn when a ligand carries an undefined stereocentre, which one `Isomer` would pool both hands of.

    `Isomer` is a single species, but an unspecified centre embeds as a mixture: RDKit assigns each seed a
    hand at random, so one result carries both enantiomers and any RMSD/energy prune downstream cross-prunes
    two distinct species. `enumerate_isomers` expands them into separate candidates instead.
    """
    centres = _stereo.unassigned_centres(mol, exclude=metal_indices(mol))
    if centres:
        logger.warning(
            "Isomer: ligand stereo element(s) %s undefined; seeds mix both hands (use enumerate_isomers)",
            [c[0] if len(c) == 1 else f"{c[0]}={c[1]}" for c in centres],
        )


@dataclass(init=False)
class Isomer:
    """One coordination isomer ready to embed: surrogate `mol`, polyhedron `cons`, `label`, `restore()`.

    Build a known isomer with ``Isomer(mol, geometry, sites)``; `enumerate_isomers` builds the unknown ones.
    `vertices[v]` is the donor atom seated at polyhedron vertex `v`, or ``VACANT`` for an empty pocket, which
    ``coordinate=`` uses to seat a substrate donor.
    """

    mol: Chem.Mol
    cons: Constraints
    metal: int
    donors: list
    real_z: int
    real_q: int  # the metal's oxidation state; the surrogate is neutral and `restore_metal` hands it back
    label: str
    geometry: str
    # No field has a dataclass default: `init=False` generates no __init__, so both builders set every field,
    # and the dataclass-field-order ignores are false positives -- there is no argument list to shadow.
    vertices: list
    chirality: str = ""  # metal-centre handedness, 'delta'/'lambda'/'': the name-agnostic stereo identity
    extra: list  # ty: ignore[dataclass-field-order]  # spectator metals (idx, real_z, real_q), restored with
    # `metal`
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')
    stereo_label: str = ""  # ligand stereoisomer tag ('16R'), distinct from the metal-centre `chirality`
    haptic: dict  # ty: ignore[dataclass-field-order]  # {centroid vertex -> its face's atoms}. A vertex is not
    # always an atom of `mol` -- an η² alkene, Cp or arene is one vertex -- so resolve it through here first.
    donor_bonds: list  # ty: ignore[dataclass-field-order]  # stripped M-donor bonds, re-added dative by
    # `connect_metal`

    def __init__(self, mol, geometry, sites, lengths="auto"):
        """Seat `sites` on `geometry`'s polyhedron: the known-isomer front door (`enumerate_isomers` is the rest).

        `geometry` is a polyhedron name or its 3-letter code (``'OCT'``). `sites` maps vertex -> donor atom
        index, as a ``{vertex: atom}`` dict or a vertex-ordered list; real atom indices, because that is what
        perception hands a consumer, and mapping them onto the internal padded-donor order happens here. The
        metal, its donors and the surrogate come from `mol`'s own bonds, dative or covalent. A geometry with
        more vertices than donors leaves the spare one a coordination pocket.

        A vertex number means whatever that polyhedron's `vertex_dirs` says, and the convention is not uniform
        across them: square-planar 0 and 1 are cis (its trans partner is 2), octahedral 0 and 1 are trans.
        Read the record in `metal_polyhedron.py`, and check a seating with ``iso.label`` / ``iso.summary()``.

        `lengths` says where the M-donor windows are measured from: ``'auto'`` / ``'input'`` / ``'model'``,
        see `metal_coordination.resolve_lengths`.
        """
        geom = resolve_geometry(geometry)
        if geom not in POLYHEDRA:
            raise ValueError(
                f"unknown geometry {geometry!r}; available: {sorted(POLYHEDRA)} "
                f"(or a code: {sorted(p.code for p in POLYHEDRA.values() if p.code)})"
            )
        if len(metal_indices(mol)) > 1:
            raise NotImplementedError(
                "Isomer() seats one metal centre; enumerate a multi-metal complex with "
                "enumerate_isomers(center=), which retains the spectator metal(s)"
            )
        _warn_undefined_ligand_stereo(mol)
        base, m, donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = _collapse_haptic(base, donors)  # each haptic face -> one centroid vertex
        if not donors:
            logger.warning(
                "Isomer: the metal has no donor bonds, so this %s sphere is vacant; use dative bonds",
                geom,
            )
        n = n_sites(geom)
        if len(donors) > n:
            raise ValueError(f"{geom} has {n} coordination sites but the metal has {len(donors)} donor site(s)")
        padded = list(donors) + [VACANT] * (n - len(donors))
        order = _seat_order(padded, haptic, sites)
        vertices = [padded[k] for k in order]
        # the centroid dummy is transient embed scaffolding, so the stored mol/donors are real (see `haptic`)
        real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
        self.mol = strip_phantoms(base, set(haptic))
        _say_length_source(base, lengths)
        self.cons = coordination(base, m, padded, geom, order, real_z, haptic=haptic, lengths=lengths)
        self.metal, self.donors, self.real_z, self.real_q = m, real_donors, real_z, real_q
        self.label = _order_label(base, padded, geom, order)
        self.geometry, self.vertices = geom, vertices
        self.chirality = chirality_of(base, donors, geom, vertices, haptic)
        self.extra, self.stereo_ref, self.stereo_label = [], None, ""
        self.haptic = dict(haptic)
        self.donor_bonds = [(d, m) for d in real_donors]

    @classmethod
    def _from_parts(
        cls,
        mol,
        cons,
        metal,
        donors,
        real_z,
        real_q,
        label,
        geometry,
        vertices,
        *,
        chirality="",
        extra=(),
        stereo_ref=None,
        stereo_label="",
        haptic=(),
        donor_bonds=(),
    ):
        """Build an Isomer from already-derived parts, for a builder that surrogated the Mol itself.

        `enumerate_isomers` surrogates once and seats many orderings, and `from_geometry` reads the arrangement
        off a conformer; neither can go through `__init__`, which does the derivation itself.
        """
        iso = cls.__new__(cls)
        iso.mol, iso.cons, iso.metal, iso.donors = mol, cons, metal, donors
        iso.real_z, iso.real_q, iso.label, iso.geometry = real_z, real_q, label, geometry
        iso.vertices, iso.chirality, iso.extra = list(vertices), chirality, list(extra)
        iso.stereo_ref, iso.stereo_label = stereo_ref, stereo_label
        iso.haptic, iso.donor_bonds = dict(haptic), list(donor_bonds)
        return iso

    def coordination(self):
        """Return the polyhedron `Constraints` holding this arrangement: what `embed` composes onto."""
        return self.cons

    def restore(self, mol=None):
        """Swap this isomer's surrogated metal(s) back to their real element and oxidation state; return `mol`.

        Acts on any molecule sharing this isomer's atom indexing, defaulting to its own. The charge matters:
        restoring only Z leaves an M(0) among anionic ligands and every real energy runs at the wrong total.
        """
        mol = self.mol if mol is None else mol
        for mi, rz, rq in [(self.metal, self.real_z, self.real_q), *self.extra]:
            restore_metal(mol, mi, rz, rq)
        return mol

    def summary(self):
        """Return this isomer's geometric identity string: ``geometry | per-vertex arrangement | chirality``.

        The one-liner for a single isomer, in the name-agnostic keys you would ``select`` on, e.g.
        ``'square_planar | C25 C44 O27 N37 | achiral'``. Mirrors what `IsomerSet.summary` prints per row.
        A `from_surrogate` record has no polyhedron to name, and says so rather than printing empty fields.
        """
        if not self.geometry:
            return f"{_PT.GetElementSymbol(self.real_z)}{self.metal} (surrogated, no polyhedron)"
        stereo = f" | stereo {self.stereo_label}" if self.stereo_label else ""
        return f"{self.geometry} | {arrangement(self)} | {self.chirality or 'achiral'}{stereo}"


def arrangement(iso):
    """Format a readable per-vertex ligand arrangement, e.g. ``'N3 Cl5 Cl6 ·'`` (``·`` = a vacant site).

    The unambiguous identity of an isomer, since the cis/trans label only describes a same-element pair and
    says nothing about where a vacancy sits. Order follows the polyhedron's `vertex_dirs`.
    """

    def sym(d):
        if d == VACANT:
            return "·"
        if d in iso.haptic:  # a haptic face's centroid vertex: the ring, not a real atom index
            ring = iso.haptic[d]
            return f"η{len(ring)}({min(ring)})"  # e.g. 'η5(1)': hapticity + the lowest-index ring atom, unambiguous
        return f"{iso.mol.GetAtomWithIdx(d).GetSymbol()}{d}"

    return " ".join(sym(d) for d in iso.vertices)


arrange = arrangement  # alias so IsomerSet.filter(arrangement=…) can still call the formatter (param shadows it)


def from_geometry(mol):
    """Build an `Isomer` that retains the input ligand arrangement, with no enumeration.

    Coordination constraints come from the Mol's actual conformer (which ligand sits where, at the realised
    distances and angles) and the metal is swapped to the surrogate. A haptic face (Cp, arene, η²) collapses
    to one centroid vertex by the same transient-centroid mechanism `enumerate_isomers` uses, so the retained
    arrangement matches every other metal path: a Cp is a single site, not five sigma donors. `mol` must carry
    a conformer.

    `vertices` is seated on the named polyhedron by `_input_ordering`, the same Procrustes match a ``fix=``
    uses to hold a frozen donor at its real vertex, rather than left in perception order. It has to be:
    `vertices` and the `arrangement` rendered from it are what `IsomerSet.select` keys on, so a
    perception-ordered list makes a real structure's arrangement match a different enumerated isomer. Measured
    on TransPlatin, whose as-perceived order reads identically to enumerated cis.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("from_geometry needs an input geometry (a Mol with a conformer)")
    base, m, donors, real_z, real_q = surrogate_metal(mol)  # surrogate; conformer is preserved
    base, sites, haptic = _collapse_haptic(base, donors)  # each haptic face -> one centroid vertex (sigma pass thru)
    cons = coordination_from_geometry(base, m, sites, real_z, haptic)
    # Measure the polytope from the conformer rather than guess it from the vertex count, which is only the
    # fallback for a CN no template covers. An apical (eta>=3) face fills more than one site, so a CN4 piano
    # stool is a distorted tetrahedron: take the apical default, never the flat square_planar.
    apical = any(len(r) >= _APICAL_MIN for r in haptic.values())
    measured = None if apical else classify_geometry(base, m, sites)  # logs its own perceived line
    geom = measured or geometry_for(len(sites), has_apical=apical) or f"{len(sites)}-coordinate"
    if measured is None:  # say why the name is a default rather than a measurement
        logger.info(
            "metal: no polyhedron perceived (%s) -> CN %d default %s",
            "an apical eta>=3 face fills more than one site"
            if apical
            else "no template has this vertex count / planarity",
            len(sites),
            describe(geom),
        )
    order = _input_ordering(base, m, sites, geom)  # seat each donor on the polyhedron it was just named as
    vertices = [sites[k] for k in order] if order else list(sites)  # no template (or a mismatched CN): as given
    return Isomer._from_parts(
        strip_phantoms(base, set(haptic)),  # the stored mol is real; the centroid dummy is transient
        cons=cons,
        metal=m,
        donors=donors,
        real_z=real_z,
        real_q=real_q,
        label=label(base, m, sites, base.GetConformer().GetId(), geom),  # measured from the conformer, not the seating
        geometry=geom,
        vertices=vertices,
        chirality=chirality_of(base, sites, geom, vertices, haptic=haptic),
        haptic=dict(haptic),
        donor_bonds=[(d, m) for d in donors],  # the M-donor bonds surrogate_metal stripped, re-added on output
    )


def from_surrogate(mol, metals, donor_bonds, donors=()):
    """Record an already-surrogated complex as an `Isomer` carrying no polyhedron.

    On the fix/constrain path the sphere is held from the input geometry, not from a named record, so
    `geometry`/`vertices`/`cons` stay empty and only the restore payload is real: element, oxidation state,
    spectator metals, and the stripped M-donor bonds a consumer must re-add.
    """
    (m, real_z, real_q), extra = metals[0], metals[1:]
    return Isomer._from_parts(
        mol,
        cons=Constraints(),
        metal=m,
        donors=list(donors),
        real_z=real_z,
        real_q=real_q,
        label="",
        geometry="",
        vertices=[],
        extra=extra,
        donor_bonds=donor_bonds,
    )


class IsomerSet(list):
    """The coordination isomers of a metal centre: a ``list`` of `Isomer` to iterate, index, or pick from.

    The identity is geometric, not a chemistry name: select on the per-vertex `arrangement`, the metal-centre
    `chirality` (``'delta'``/``'lambda'``/``''``), the `geometry`, or the plain index. The cis/trans/mer/fac
    `label` is a coarse, sometimes-wrong tag, never required to select:

        isos = rx.metal('CCCN[Pd](Cl)(Cl)NCCC', ['square_planar', 'tetrahedral']); isos.summary()
        ens  = rx.embed(isos.select(arrangement='N3 Cl6 N7 Cl5')).mc().prune()
        ens  = rx.embed(isos[0]).mc().prune()

    Enumeration is cheap; the expensive MC search runs only on the `Isomer` you pick.
    """

    def select(self, geometry=None, label=None, arrangement=None, chirality=None, index=None, stereo=None):
        """Return the single `Isomer` matching the given keys.

        Key on `arrangement` (the unambiguous per-vertex slot map), `chirality`, `geometry`, `index`, the
        ligand `stereo` tag (e.g. ``'16R'``), or the coarse `label`. Raises if zero or several match, listing
        every isomer so you can narrow it.
        """
        hits = self.filter(
            geometry=geometry, label=label, arrangement=arrangement, chirality=chirality, index=index, stereo=stereo
        )
        if len(hits) != 1:
            have = [(k, i.geometry, i.chirality or "-", i.stereo_label or "-", arrange(i)) for k, i in enumerate(self)]
            raise ValueError(
                f"select(geometry={geometry!r}, label={label!r}, arrangement={arrangement!r}, "
                f"chirality={chirality!r}, index={index!r}, stereo={stereo!r}) matched {len(hits)} isomer(s): "
                f"{'narrow it or pick by index' if hits else 'no match'}; have {have}"
            )
        return hits[0]

    def filter(self, geometry=None, label=None, arrangement=None, chirality=None, index=None, stereo=None):
        """Return the subset matching the given keys, as an `IsomerSet` (keep several / pick by index).

        `label` matches the base tag, so ``'fac'`` also matches auto-numbered ``fac1``/``fac2``.
        `arrangement`, `chirality`, `geometry` and `stereo` (the ligand stereoisomer tag) match exactly, and
        `index` selects positionally. `geometry` also takes a 3-letter code, and `chirality` accepts the Δ/Λ
        glyphs as well as the stored words.
        """
        geometry, chirality = resolve_geometry(geometry), chirality_tag(chirality)

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
        ``'delta'``/``'lambda'`` or ``'(achiral)'``. The coarse cis/trans/mer/fac name is not shown, because
        selecting on it is unreliable: use ``arrangement=`` / ``chirality=`` / index. Returns self.
        """
        for k, i in enumerate(self):
            stereo = f"  stereo {i.stereo_label}" if i.stereo_label else ""
            print(f"  [{k}] {i.geometry:16s} {arrange(i):26s} {i.chirality or '(achiral)'}{stereo}")
        return self


def _resolve_center(mol, metals, center):
    """Pick which transition metal to enumerate.

    `center` is None (the sole metal, else an error asking you to choose), an atom index, or an element
    symbol such as ``'Mn'``.
    """
    if center is None:
        if len(metals) == 1:
            return metals[0]
        raise ValueError(
            f"{len(metals)} transition metals present "
            f"({[mol.GetAtomWithIdx(x).GetSymbol() + str(x) for x in metals]}); choose which to "
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
    raise ValueError(f"{len(hits)} {center} centres ({hits}); disambiguate with center=<atom index>")


def _load_in_ligand_stereo(mol, geometry, center, fix, stereo, lengths="auto"):
    """Expand any undefined ligand stereocentre of a coordinate-free input, recursing per variant.

    Returns the enumerated `IsomerSet` (coordination x ligand stereo), or ``None`` when there is nothing to
    load in (stereo='free', a geometry input, or no undefined centre) so the caller runs the normal path.
    """
    if stereo == "free" or mol.GetNumConformers() != 0:
        return None
    variants, n_unassigned, _total, unresolved = _stereo.enumerate_unassigned(mol, exclude=set(metal_indices(mol)))
    # exclude the metal's own centre: its handedness is enumerated below, not as RDKit point stereo
    if not n_unassigned:
        return None
    out = IsomerSet()
    for vmol, slabel in variants:  # each variant is stereo-defined, so recurse with stereo='free' and the
        for iso in enumerate_isomers(vmol, geometry, center, fix, stereo="free", lengths=lengths):  # a no-op
            iso.stereo_label = slabel
            out.append(iso)
    logger.info(
        "metal: %d undefined stereocentre(s) -> coordination x %d stereoisomer(s) = %d candidate(s)",
        n_unassigned,
        len(variants),
        len(out),
    )
    if unresolved:
        logger.warning(
            "metal: %d stereo axis(es) not enumerable from a flat SMILES; embedded as one arbitrary hand",
            unresolved,
        )
    return out


def _prepare_spectators(mol, metals, center):
    """Surrogate a multi-metal complex, enumerating `center` while holding each spectator metal's shape.

    Returns ``(base, m, donors, real_z, real_q, extra, retain, spectator_bonds)``. A spectator keeps its
    oxidation state and gets the same force field as the enumerated centre, its bond-less carbon firing the
    same fictitious LJ. ``spectator_bonds`` are its stripped M-donor pairs, re-added dative on the output.
    """
    m = _resolve_center(mol, metals, center)
    spectators = [s for s in metals if s != m]
    if spectators and mol.GetNumConformers() == 0:
        raise ValueError(
            f"enumerating one centre of a {len(metals)}-metal complex needs an input "
            f"geometry (an .xyz) to retain the other metal(s); got a coordinate-free input"
        )

    def non_metal(nbrs):
        return [a.GetIdx() for a in nbrs if a.GetAtomicNum() not in COORDINATION_METALS]

    donors = non_metal(mol.GetAtomWithIdx(m).GetNeighbors())  # a partner metal is not a coordination donor
    # Only this path skips `_collapse_haptic`, so a Cp would count as five sigma donors and pick the wrong
    # polyhedron. Refused rather than built: a haptic centre enumerates to one isomer anyway.
    if any(len(s) > 1 for s in _haptic_sites(mol, donors)):
        raise NotImplementedError(
            "the enumerated centre carries a haptic face (eta2 / Cp / arene), which is only supported for a "
            "single-metal complex; a multi-metal complex with a haptic centre is not handled (the ring atoms "
            "would be mis-counted as separate sigma donors). Enumerate the single-metal fragment instead"
        )
    real_z = mol.GetAtomWithIdx(m).GetAtomicNum()
    real_q = mol.GetAtomWithIdx(m).GetFormalCharge()  # oxidation state, restored so xtb gets the right charge
    spec = {
        s: (
            non_metal(mol.GetAtomWithIdx(s).GetNeighbors()),
            mol.GetAtomWithIdx(s).GetAtomicNum(),
            mol.GetAtomWithIdx(s).GetFormalCharge(),  # a spectator keeps its oxidation state too
        )
        for s in spectators
    }
    base, _metals_info = surrogate_all_metals(mol)  # surrogate every metal; the conformer is preserved
    extra = [(s, spec[s][1], spec[s][2]) for s in spectators]
    # Hold each spectator's shape by relative pairwise distances, which are frame-independent and therefore
    # achiral: its handedness stays random here and is fixed by select_stereo afterward.
    retain = Constraints()
    for s in spectators:
        hold_shape(base, [s, *spec[s][0]], retain)
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
    spectator_bonds = [(d, s) for s in spectators for d in spec[s][0]]  # re-added dative so a spectator connects
    return base, m, donors, real_z, real_q, extra, retain, spectator_bonds


def _select_geometries(base, m, donors, haptic, geometry, n):
    """Resolve `geometry` to the list of polyhedron names to enumerate (default from donor count, or as given)."""
    if geometry is None:
        # an apical (eta>=3) face fills more than one site, so a CN4 carrying one is a piano stool, not the
        # square_planar that would seat a ligand trans through the ring. An eta2 face is a single-site vertex.
        apical = any(len(r) >= _APICAL_MIN for r in haptic.values())
        measured = None
        if base.GetNumConformers() and not apical:  # a retained geometry names itself; an apical face is a
            sites = [haptic.get(v, v) for v in donors]  # site-count question the templates do not model
            measured = classify_geometry(base, m, sites)  # logs its own perceived line
        geoms = [measured or geometry_for(n, has_apical=apical)]
        if geoms == [None]:
            raise ValueError(
                f"no default geometry for {n} donors; pass geometry= a name or list "
                f"(options for {n} donors: {[p.name for p in geometries_for_cn(n)]})"
            )
        if measured is None:  # a count default, not a measurement: the user must be able to tell them apart
            logger.info(
                "metal: no geometry= and %s -> CN %d default %s",
                "an apical eta>=3 face fills more than one site"
                if apical
                else ("no input conformer to measure" if not base.GetNumConformers() else "nothing perceived"),
                n,
                describe(geoms[0]),
            )
    else:
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
        geoms = [resolve_geometry(g) for g in geoms]  # a name or its 3-letter code ('OCT'), case-insensitive
    for g in geoms:
        if g not in POLYHEDRA:
            hint = "; pass a list of names, e.g. ['square_planar', 'tetrahedral']" if g == "all" else ""
            raise ValueError(
                f"unknown geometry {g!r}; available: {sorted(k for k in POLYHEDRA if k != 'None')} "
                f"(or a code: {sorted(p.code for p in POLYHEDRA.values() if p.code)}){hint}"
            )
    if geometry is not None:
        logger.info("metal: requested %s", ", ".join(describe(g) for g in geoms))
    return geoms


def _frozen_permutations(base, m, padded, geom, frozen_donors, sites):
    """Generate every free-donor vertex permutation with each frozen donor pinned at its input vertex.

    Returns the explicit permutation list, bypassing the symmetry-reduced canned `isomer_permutations`, which
    would miss the representative ordering a valid frozen isomer needs; or ``None`` if the input vertex
    ordering can't be read.
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
    spectator_bonds=(),  # each held spectator metal's stripped M-donor pairs, re-added dative on the output
    lengths="auto",
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
    if frozen_donors and sites == n:  # pin each frozen donor at its input vertex, then generate every
        perms = _frozen_permutations(base, m, padded, geom, frozen_donors, sites)  # free-donor arrangement
    out = []
    for order in distinct_vertex_orderings(
        base, padded, geom, perms=perms, r_metal=_PT.GetRcovalent(real_z), haptic=haptic
    ):
        cons = coordination(
            base,
            m,
            padded,
            geom,
            order,
            real_z,
            frozen=frozen_donors,
            core_frozen=fix_cons.frozen,
            haptic=haptic,
            lengths=lengths,
        )
        # Hold any spectator metal's shape and give it the same field. Field-driven, so a floor can never
        # arrive without its `dg_floors` twin, which would leave RDKit's phantom ~3.4 Å floor standing.
        cons = compose(cons, retain)
        cons.distances.update(fix_cons.distances)  # hold the frozen reacting core (relative pairwise)
        cons.frozen |= fix_cons.frozen  # and pin it in the relax
        od = [padded[k] for k in order]  # vertex -> donor atom (or VACANT); a centroid dummy for an eta>=3 face
        # The stored Isomer is real: the centroid is transient scaffolding the embed materialises from
        # `cons.haptic`, so strip it and report the real coordinating atoms. `vertices` keeps the centroid index.
        real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
        out.append(
            Isomer._from_parts(
                strip_phantoms(Chem.Mol(base), set(haptic)),
                cons=cons,
                metal=m,
                donors=real_donors,
                real_z=real_z,
                real_q=real_q,
                label=_order_label(base, padded, geom, order),
                geometry=geom,
                vertices=od,
                chirality=chirality_of(base, donors, geom, od, haptic),
                extra=extra,
                stereo_ref=ref_sig,
                haptic=dict(haptic),
                # the centre's real donors plus each spectator's, re-added dative on the output
                donor_bonds=[(d, m) for d in real_donors] + list(spectator_bonds),
            )
        )
    if not out:  # every candidate ordering was rejected: silence here reads as "this geometry has no isomers"
        logger.warning(
            "metal[%s]: no arrangement survived the feasibility filters; try another geometry",
            geom,
        )
    return out


def _number_shared_labels(out):
    """Disambiguate isomers sharing a (geometry, label) as cis1, cis2… so each stays selectable."""
    counts = Counter((i.geometry, i.label) for i in out)  # several heteroleptic isomers can share a label
    nth = Counter()
    for i in out:
        key = (i.geometry, i.label)
        if counts[key] > 1:
            nth[key] += 1
            i.label = f"{i.label}{nth[key]}"


def stated_arrangement(mol):
    """Return the ``(geometry, {vertex: donor atom})`` a canonical string put on `mol`, or ``None``.

    The reading half of `metal_smiles.canonical_smiles`, and it lives here rather than there because what it
    reads is RDKit atom properties on a `Mol`: an arrangement, which is this module's subject, not a string,
    which is that one's. `metal_smiles.parse_smiles` has already kept the block's indices addressing the
    atoms they were written for, so a string that carries an arrangement arrives as an ordinary `Mol` wearing
    it and `enumerate_isomers` seats it instead of enumerating. Returns ``None`` for a plain SMILES, which is
    the signal that there is nothing to seat.

    A vacant vertex simply has no donor, so a coordination pocket survives the round trip.
    """
    # One key for both notes, so the VALUE says which it is: a slot is `s<n>` with an optional winding sign
    # (`metal_polyhedron` owns that grammar, since it owns the slots), and anything else on a noted atom is
    # the metal's geometry code.
    noted = {a.GetIdx(): a.GetProp("atomNote") for a in mol.GetAtoms() if a.HasProp("atomNote")}
    slots = {i: read for i, v in noted.items() if (read := read_slot_note(v)) is not None}
    geom = [i for i in noted if i not in slots]
    if len(geom) != 1:
        return None
    name = resolve_geometry(noted[geom[0]].split("-")[0])
    sites = {}
    for atom_idx, (slot, _winding) in slots.items():
        sites.setdefault(slot, atom_idx)  # a haptic face writes one slot on every ring atom; any names it
    if name not in POLYHEDRA:
        raise ValueError(f"the arrangement on this string names {name!r}, which is not a polyhedron rxembed has")
    return name, sites


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo="racemic", stereo_ref=None, lengths="auto"):
    """Enumerate all distinct coordination isomers of a `Mol` as ready-to-embed `Isomer` objects (metal surrogated).

    Returns an `IsomerSet` to pick from with ``IsomerSet.select(…)``. For a coordinate-free input,
    `stereo='racemic'` also enumerates any undefined ligand stereocentre, so the set spans coordination x
    ligand-stereo and each `Isomer` carries a `.stereo_label`; `stereo='free'` opts out and a geometry input
    is untouched.

    `geometry` selects the polyhedra: ``None`` takes the donor-count default, a name takes one, a list takes
    each so you can compare energies yourself. Names come from ``geometries_for_cn``, each also nameable by
    its 3-letter code. `center=` picks which metal to enumerate, the others retained at the input geometry. A
    geometry with more vertices than donors leaves the empty one as a coordination pocket.

    `stereo_ref` is the input's chirality fingerprint, computed by the caller because it needs a perception
    the engine does not carry, and passed to each `Isomer` for a ``stereo='preserve'`` gate.

    A `Mol` that already states an arrangement (one read from a `canonical_smiles` string) has nothing to
    enumerate: that one `Isomer` comes back, seated as written.

    `lengths` says where the M-donor windows are measured from: ``'auto'`` (the input conformer if there is
    one, else the fitted model), ``'input'``, or ``'model'``. Set it when `mol` carries a geometry that is not
    metal-aware, since a plain ETKDG conformer has no M-L parameter and ``'auto'`` would embed toward it.

    Takes a `Mol`: parsing is the consumer's job (see the module docstring), and `rxembed.pipeline.metal` is
    the same enumeration with the SMILES / ``.xyz`` reader and the `stereo_ref` fingerprint in front of it.
    """
    if not isinstance(mol, Chem.Mol):  # else the first `mol.GetNumConformers()` below raises a bare
        raise TypeError(  # AttributeError, on exactly the string `rxembed.pipeline.metal` accepts
            f"enumerate_isomers() takes an RDKit Mol, got {type(mol).__name__}. Perception is upstream of the "
            f"engine: parse a SMILES with rxembed.parse_smiles, or call rxembed.pipeline.metal, which reads a "
            f"SMILES or an .xyz path and computes the stereo fingerprint the 'preserve' gate compares against"
        )
    loaded = _load_in_ligand_stereo(mol, geometry, center, fix, stereo, lengths)
    if loaded is not None:
        return loaded  # each ligand-stereo variant recurses, so a stated arrangement is seated on every one
    stated = stated_arrangement(mol)
    if stated is not None:
        name, sites = stated
        if (geometry is not None and resolve_geometry(geometry) != name) or fix:
            raise ValueError(
                f"this input already states a {name} arrangement, so "
                f"{'fix=' if fix else f'geometry={geometry!r}'} has nothing to act on; drop it to use what "
                f"the input says, or strip the arrangement to enumerate"
            )
        logger.info("metal: %s arrangement stated by the input; enumerating nothing", describe(name))
        return IsomerSet([Isomer(mol, name, sites, lengths=lengths)])
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no transition metal found")
    haptic = {}  # centroid dummies, keyed dummy -> its face atoms; set by _collapse_haptic on the single path
    # One metal means no spectators, so this is the single-centre path whether or not center= names it. An
    # explicit center= is still validated against the sole metal, raising on a wrong index or element.
    if len(metals) == 1:
        if center is not None:
            _resolve_center(mol, metals, center)
        base, m, donors, real_z, real_q = surrogate_metal(mol)
        base, donors, haptic = _collapse_haptic(base, donors)  # collapse each haptic face to one centroid vertex
        extra, retain, spectator_bonds = [], Constraints(), []
    else:
        base, m, donors, real_z, real_q, extra, retain, spectator_bonds = _prepare_spectators(mol, metals, center)
    _say_length_source(base, lengths)  # once per molecule, not per ordering
    fix_cons = Constraints()
    frozen_donors = set()
    # Hold a reacting TS core at the input geometry while the rest of the coordination sphere is enumerated:
    # the mer/fac of a tridentate, say, while the reacting donor and substrate stay put.
    if fix:
        if base.GetNumConformers() == 0:
            raise ValueError(
                "fix= needs an input geometry (an .xyz / a Mol with a conformer) to hold "
                "the reacting core; got a coordinate-free input (e.g. a SMILES)"
            )
        fix_cons, _ = resolve_core(base, fix=fix, has_geometry=True)
        frozen_donors = fix_cons.frozen & set(donors)
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
                ref_sig=stereo_ref,
                spectator_bonds=spectator_bonds,
                lengths=lengths,
            )
        )
    _number_shared_labels(out)
    return out


def _input_ordering(mol, metal, donors, geometry):
    """Find the vertex ordering that best matches the input geometry: which donor sits at which vertex.

    ``od[vertex] = donors[order[vertex]]``, read from the conformer by orthogonal Procrustes over the
    candidate vertex orderings. This lets a ``fix=`` hold each frozen donor at its real vertex so only the
    free sites are enumerated; otherwise the enumeration permutes a frozen donor into a vertex it cannot
    occupy, producing isomers that contradict the frozen core, such as a hydride forced off its TS site.

    The candidate list holds one representative per FULL-group orbit, so the winner is only a seating up to
    a reflection and a mirror image scores identically at every candidate; `seat_properly` turns it into the
    reflection-free one, without which `chirality_of` hands both hands the same tag.
    """
    dirs_ref = vertex_dirs(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([pos[d] - pos[metal] for d in donors], float)
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
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
                if _vertex_angle(dirs[c], dirs[a]) > _TRANS_ANGLE or _vertex_angle(dirs[c], dirs[b]) > _TRANS_ANGLE:
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
    """Return False if donor ``d``, stretched to span trans, cannot still point its lone pair at the metal.

    The reach test asks only whether the donors can get ``need`` apart; this asks whether the backbone that
    achieves it can still donate. At a trans span the metal lies on the D···D' line, so the fold ruler's M-D-X
    angle is fixed by the ligand's own X-D···D' angle, and the anti backbone that reaches furthest points D's
    substituent straight at the metal. Taking the census fold floor as the minimum M-D-X, each heavy X is
    forced out to a matching X···D' distance; if the backbone can't reach that, no conformer spans and donates.

    Abstains via `donation_axis` (a hydride, bridging or haptic donor has no axis) and for an uncalibrated
    (element, hyb) class, reading the ruler's own rules rather than a copy of them.
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
    """Return False if a chelate is placed trans across a metal its backbone can't reach, or can't donate to.

    A cis bidentate always folds in, so only a wide separation (`_SPAN_ANGLE` or more, i.e. trans) is tested.
    Both questions read the ligand's own bounds matrix, so a genuine long-bridge ligand that can span trans is
    allowed: this is geometry, not a topological guess. Generalises `_central_trans` to any denticity.

    Reach: the donors would sit on opposite sides, needing a donor-donor distance
    ``law_of_cosines(d_Ma, d_Mb, θ)`` a short backbone cannot span.

    Orientation (`_donor_faces_metal`): reach models where the donors are, not where they point. An extended
    anti backbone aims both lone pairs along the chain; without this a diphosphine passed reach and relaxed to
    P-M-P 155° against its 74-104° bite.
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
            if theta < _SPAN_ANGLE:  # cis / adjacent -> the chelate folds in, always feasible
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


def _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic, *, limit=None):
    """Dedup vertex orderings by their connectivity-aware signature; stop once `limit` distinct ones are found.

    Two geometric impossibilities are pruned: a tridentate's central donor trans to its own arm
    (`_central_trans`), and a chelate placed trans whose backbone can't reach or, stretched, can't donate
    inward (`_chelate_span_ok`). A cis chelate is always kept; any residual is dropped by `bonding_ok`.

    The dedup signature carries, per donor pair, the elements, the vertex angle and a same-ligand pair's
    intra-ligand bond distance, plus the centre's handedness, so enantiomers and a tridentate's mer vs
    central-trans stay distinct. `limit` short-circuits at the k-th distinct arrangement, which is all the
    warning caller needs.
    """
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != VACANT else "X") for d in donors}  # vacancy = "X"
    frag = _frag_map(mol)  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances
    real_donors = [d for d in donors if d != VACANT]
    bm = _span_bounds(mol)
    hyb = _stripped_hybridisation(mol)  # the fold ruler's own (element, hyb) class: graph-only, no coords
    pairs = [(p, q) for p in range(len(dirs)) for q in range(p + 1, len(dirs))]
    angle = {(p, q): _vertex_angle(dirs[p], dirs[q]) for p, q in pairs}  # a vertex-pair's angle is donor-independent

    def link(od, p, q):  # intra-ligand bond distance of a same-ligand pair
        a, b = _vertex_atom(haptic, od[p]), _vertex_atom(haptic, od[q])  # resolve a centroid to its ring atom
        if VACANT in (od[p], od[q]) or frag[a] != frag[b]:  # distinguishes a chelate's central from its
            return -1  # terminal donor; -1 for different ligands or a vacancy
        return int(dmat[a][b])

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
            sorted((tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), angle[(p, q)]) for p, q in pairs)
        )
        sig = (sig, chirality_of(mol, real_donors, geometry, od, haptic))  # keep enantiomers distinct (else merged)
        if sig not in seen:
            seen.add(sig)
            out.append(order)
            if limit is not None and len(out) >= limit:
                break
    return out


def distinct_vertex_orderings(mol, donors, geometry, perms=None, r_metal=1.4, haptic=None):
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
            and len(_distinct_orderings(mol, donors, geometry, orbit, dirs, r_metal, haptic, limit=2)) > 1
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
    return _distinct_orderings(mol, donors, geometry, perms, dirs, r_metal, haptic)
