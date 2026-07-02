"""Metal-complex coordination constraints — the polyhedron, per isomer.

Coordination geometries, L-M-L angles, and isomer permutations are adapted from TMC_embed code by
Maria H. Rasmussen, **TMC_embed** (https://github.com/jensengroup/TMC_embed).

The metal is held purely by distance + angle constraints (its bonds removed) and embedded/relaxed with a
UFF-typeable surrogate atom in its place — so the whole path is plain RDKit + UFF, no xtb.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from rxembed.log import logger  # relative, like builders.py — package convention

from .base import Constraints, add_distance, add_pairwise_shape

_PT = GetPeriodicTable()
TRANSITION_METALS = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    57,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
}
SURROGATE = 6  # carbon — UFF-typeable, fine as a bondless constrained anchor for embed + the main relax
RELAX_SURROGATE = 15  # phosphorus — for the donor-proton relax only: UFF-types the metal *with bonds*
# through CN6 (carbon's types stop at 4 bonds), so a bonded donor regains correct hybridisation and its
# protons sit right. It is used ONLY in a second pass with every heavy atom frozen — donor H's are the
# sole movers — NOT as a single bonded-P relax of the whole complex. That distinction matters: a free
# bonded-P relax lets phosphorus's own UFF bond/angle terms drag every metal-donor distance to the P-N
# length (~1.8 A, wrong for real metals) and, at CN>=5, bend the polyhedron (its angle terms don't know
# the target geometry). Freezing the heavy frame confines P's influence to the protons; the polyhedron
# and distances stay exactly as the bondless-carbon first pass (constraint-driven) set them.

GEOM = {2: "linear", 3: "trigonal_planar", 4: "square_planar", 5: "trigonal_bipyramidal", 6: "octahedral"}
# catalogue of supported polyhedra per coordination number (the first is the GEOM default) — for
# reference and "did you mean" suggestions. Pick geometries explicitly, e.g.
# rx.metal(smi, ['square_planar', 'tetrahedral']), and run your own energy comparison across them.
GEOM_OPTIONS = {
    2: ["linear"],
    3: ["trigonal_planar", "t_shape"],
    4: ["square_planar", "tetrahedral", "seesaw"],
    5: ["trigonal_bipyramidal", "square_pyramidal"],
    6: ["octahedral"],
}
ANGLES = {  # adapted from TMC_embed angle_dict
    "linear": [(0, 1, 180)],
    "trigonal_planar": [(0, 1, 120), (1, 2, 120), (0, 2, 120)],
    "t_shape": [(0, 1, 90), (1, 2, 90), (0, 2, 180)],
    "square_planar": [(0, 2, 180), (1, 3, 180), (0, 1, 90)],
    "tetrahedral": [(0, 1, 109.5), (0, 2, 109.5), (0, 3, 109.5), (1, 2, 109.5), (1, 3, 109.5), (2, 3, 109.5)],
    "seesaw": [(0, 1, 180), (2, 3, 120), (0, 3, 90), (1, 2, 90)],
    "trigonal_bipyramidal": [(0, 1, 180), (1, 2, 90), (0, 3, 90), (2, 3, 120), (3, 4, 120), (2, 4, 120)],
    "square_pyramidal": [(0, 1, 90), (0, 3, 90), (1, 3, 180), (2, 4, 180), (1, 4, 90), (2, 3, 90)],
    "octahedral": [(0, 1, 180), (2, 3, 180), (4, 5, 180), (0, 2, 90), (1, 4, 90), (3, 5, 90)],
}
PERMUTATIONS = {  # adapted from TMC_embed coord_permutations
    "t_shape": [[0, 1, 2], [1, 0, 2], [0, 2, 1]],
    "square_planar": [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1]],
    "seesaw": [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1]],
    "trigonal_bipyramidal": [
        [0, 1, 2, 3, 4],
        [0, 2, 1, 3, 4],
        [0, 3, 1, 2, 4],
        [0, 4, 1, 2, 3],
        [1, 2, 0, 3, 4],
        [1, 3, 0, 2, 4],
        [1, 4, 0, 2, 3],
        [2, 3, 0, 1, 4],
        [2, 4, 0, 1, 3],
        [3, 4, 0, 1, 2],
    ],
    "square_pyramidal": [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [0, 1, 4, 2, 3],
        [1, 0, 2, 3, 4],
        [1, 0, 3, 2, 4],
        [1, 0, 4, 2, 3],
        [2, 1, 0, 3, 4],
        [2, 1, 3, 0, 4],
        [2, 1, 4, 0, 3],
        [3, 1, 2, 0, 4],
        [3, 1, 0, 2, 4],
        [3, 1, 4, 2, 0],
        [4, 1, 2, 3, 0],
        [4, 1, 3, 2, 0],
        [4, 1, 0, 2, 3],
    ],
    "octahedral": [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 4, 3, 5],
        [0, 1, 2, 5, 3, 4],
        [0, 2, 1, 3, 4, 5],
        [0, 2, 1, 4, 3, 5],
        [0, 2, 1, 5, 3, 4],
        [0, 3, 2, 1, 4, 5],
        [0, 3, 2, 4, 1, 5],
        [0, 3, 2, 5, 1, 4],
        [0, 4, 2, 3, 1, 5],
        [0, 4, 2, 1, 3, 5],
        [0, 4, 2, 5, 3, 1],
        [0, 5, 2, 3, 4, 1],
        [0, 5, 2, 4, 3, 1],
        [0, 5, 2, 1, 3, 4],
    ],
}
# polyhedron vertex unit vectors (order matches ANGLES), for distinct-isomer detection
_s = math.sqrt(3) / 2  # sin 60° = cos 30°, the y of a 120°-spaced vertex
VERTEX_DIRS = {  # rxembed's, derived to match the TMC angles
    "linear": [(0, 0, 1), (0, 0, -1)],
    "trigonal_planar": [(1, 0, 0), (-0.5, _s, 0), (-0.5, -_s, 0)],
    "t_shape": [(0, 0, 1), (1, 0, 0), (0, 0, -1)],  # 0,2 trans; 1 perpendicular
    "square_planar": [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
    "tetrahedral": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
    "seesaw": [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, _s, 0)],  # 0,1 axial; 2,3 equatorial 120°
    "trigonal_bipyramidal": [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, _s, 0), (-0.5, -_s, 0)],
    "square_pyramidal": [(0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],  # 0 apex; 1-4 basal
    "octahedral": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
}


def metal_index(mol):
    """Index of the first transition-metal atom, or None."""
    return next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS), None)


def prepare(mol):
    """Remove metal-donor bonds and swap the metal to a UFF surrogate. Returns (mol, metal, donors, real_Z)."""
    m = metal_index(mol)
    if m is None:
        raise ValueError("no transition metal found")
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
    em = Chem.RWMol(mol)
    for d in donors:
        em.RemoveBond(d, m)
        em.GetAtomWithIdx(d).SetNoImplicit(True)  # freeze donor H count so MC (openconf) adds none
    real_z = em.GetAtomWithIdx(m).GetAtomicNum()
    a = em.GetAtomWithIdx(m)
    a.SetAtomicNum(SURROGATE)
    a.SetNoImplicit(True)
    a.SetFormalCharge(0)
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return out, m, donors, real_z


def restore(mol, metal, real_z):
    """Swap `metal` back from the surrogate to its real element `real_z`."""
    mol.GetAtomWithIdx(metal).SetAtomicNum(real_z)


def metal_indices(mol):
    """Return the indices of all transition-metal atoms."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]


def prepare_all(mol):
    """Surrogate **every** transition metal (bonds removed, carbon) for a multi-metal complex.

    UFF must type the whole complex -- e.g. a bimetallic TS (ferrocene Fe + a reactive Mn). `prepare` only
    does the first metal; the second would stay a real metal that won't UFF-type. Returns ``(mol, metals,
    donors)`` where ``metals`` is ``[(idx, real_z), ...]`` (restore each) and ``donors`` is every
    metal-donor atom (so the caller can freeze the coordination cores -- bond-stripped surrogates would
    otherwise let the ligands drift). Lenient sanitise (a stripped η⁵-Cp is a radical fragment, fine here).
    """
    idxs = metal_indices(mol)
    if not idxs:
        raise ValueError("no transition metal found")
    em = Chem.RWMol(mol)
    metals, donors = [], []
    for m in idxs:
        metals.append((m, em.GetAtomWithIdx(m).GetAtomicNum()))
        for d in [n.GetIdx() for n in em.GetAtomWithIdx(m).GetNeighbors()]:
            if em.GetBondBetweenAtoms(d, m) is not None:
                em.RemoveBond(d, m)
            em.GetAtomWithIdx(d).SetNoImplicit(True)
            donors.append(d)
        a = em.GetAtomWithIdx(m)
        a.SetAtomicNum(SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
    out = em.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out, metals, sorted(set(donors))


def geometry_for(n_donors):
    """Default coordination polyhedron name for `n_donors`, or None."""
    return GEOM.get(n_donors)


VACANT = -1  # a coordination vertex left empty (donors < sites)
_TRANS_ANGLE = 150  # degrees: a same-element donor pair beyond this is trans (else cis)
_TRIAD = 3  # a mer/fac triad is exactly three donors
_PAIR = 2  # a same-element pair that can be cis/trans
# geometries with no cis/trans (or mer/fac) isomerism under monodentate substitution: linear (its two sites
# are always trans — no choice), trigonal_planar and tetrahedral (no trans vertex pair exists, so all donors
# are mutually cis). One arrangement only; it carries no stereo-label.
_NO_GEOMETRIC_ISOMERISM = frozenset({"linear", "trigonal_planar", "tetrahedral"})
_COLINEAR_TOL = 0.5  # topological-distance tolerance for the "central donor" collinearity test


def n_sites(geometry):
    """Return the number of coordination vertices the geometry has."""
    return len(VERTEX_DIRS.get(geometry) or ()) or max(max(i, j) for i, j, _ in ANGLES[geometry]) + 1


def hold_shape(mol, atoms, cons, pad=0.1, cid=-1):
    """Fix the SHAPE of `atoms` at their input-geometry mutual distances (all pairwise, ``±pad``).

    A relative, frame-independent constraint, so a bondless surrogate metal's coordination sphere (a
    ferrocene, a retained spectator centre) is held INTACT when the molecule is re-embedded, without pinning
    it to an absolute frame (which would tear it from the rest).
    """
    add_pairwise_shape(cons, atoms, mol.GetConformer(cid).GetPositions(), pad)


def coordination(mol, metal, donors, geometry, order, real_z, frozen=()):
    """Build one isomer's constraints: metal-donor distances + donor-metal-donor angles.

    `donors` may be padded with ``VACANT`` for empty vertices (a coordination pocket) -- those get no
    constraints.

    `frozen` is the set of donor atoms held rigid by a ``fix=`` reacting core: an angle between **two
    frozen donors** is *omitted* — the freeze already pins that sub-triangle exactly (Mn-d1, Mn-d2, d1-d2),
    so an ideal-polyhedron angle there only over-determines it and makes triangle-smoothing widen (then
    lock in) a distorted core. A free-vs-frozen angle is kept — it's what seats a free donor at its vertex.

    The L-M-L angle window is **tight (±8°) for an inter-ligand pair** (its angle is free to take the
    ideal polyhedron value) but **wide (±25°) for an intra-chelate pair** (two donors of one ligand): a
    chelate's bite is set by its rigid backbone and is distorted well off the ideal (a meridional pincer
    sits near 80°/165°, not 90°/180°), so forcing the ideal there contradicts the backbone and the
    distance-geometry bounds become infeasible. The wide window still separates cis from trans (so mer/fac
    is preserved) while letting the bite be whatever the backbone needs.
    """
    r_m = _PT.GetRcovalent(real_z)
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}  # same ligand = same fragment
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    c = Constraints()
    od = [donors[k] for k in order]  # od[vertex] = donor atom there, or VACANT
    for d in od:
        if d == VACANT:
            continue
        if pos is not None:  # use the REALISED metal-donor bond length from the input
            d_md = float(np.linalg.norm(pos[metal] - pos[d]))  # geometry (a covalent-radius guess is wrong for
            add_distance(c.distances, metal, d, d_md - 0.1, d_md + 0.1)  # a hydride/carbonyl and breaks the bounds)
        else:
            d_md = 0.9 * (r_m + _PT.GetRcovalent(mol.GetAtomWithIdx(d).GetAtomicNum()))
            add_distance(c.distances, metal, d, d_md - 0.05, d_md + 0.05)
    for i, j, a in ANGLES[geometry]:
        if od[i] == VACANT or od[j] == VACANT:  # an angle to an empty vertex is unconstrained
            continue
        if od[i] in frozen and od[j] in frozen:  # both held by freeze -> don't over-determine the core
            continue
        pad = 25.0 if frag[od[i]] == frag[od[j]] else 8.0  # a chelate bite is distorted off the ideal
        c.angles[(od[i], metal, od[j])] = (max(0.0, a - pad), min(180.0, a + pad))
    return c


def _vertex_angle(u, v):
    u, v = np.array(u), np.array(v)
    return round(float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))))


def coordination_from_geometry(mol, metal, donors, real_z, cid=-1):
    """Build coordination constraints from the **actual** input geometry (the specific ligand arrangement).

    The realised metal-donor distances and donor-metal-donor angles, so the *specific* arrangement is held
    (vs an ideal polyhedron). Used to *retain* an input metal complex instead of enumerating isomers.
    """
    pos = mol.GetConformer(cid).GetPositions()
    c = Constraints()
    for d in donors:
        dist = float(np.linalg.norm(pos[metal] - pos[d]))
        add_distance(c.distances, metal, d, dist - 0.05, dist + 0.05)
    for a, b in itertools.combinations(donors, 2):
        ang = _vertex_angle(pos[a] - pos[metal], pos[b] - pos[metal])
        c.angles[(a, metal, b)] = (max(0.0, ang - 8), min(180.0, ang + 8))
    return c


def from_geometry(mol):
    """Build an `Isomer` that **retains the input ligand arrangement** (no enumeration).

    Coordination constraints come from the Mol's *actual* conformer (which ligand sits where, at the
    realised distances/angles), the metal is swapped to the surrogate, and `vertices` records the as-given
    donor order. `mol` must carry a conformer (e.g. perceived from an xyz).
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("from_geometry needs an input geometry (a Mol with a conformer)")
    base, m, donors, real_z = prepare(mol)  # surrogate; conformer is preserved
    cons = coordination_from_geometry(base, m, donors, real_z)
    geom = geometry_for(len(donors)) or f"{len(donors)}-coordinate"
    return Isomer(
        base, cons, m, donors, real_z, label(base, m, donors, base.GetConformer().GetId(), geom), geom, list(donors)
    )


def label(mol, metal, donors, cid, geometry=None):
    """Coordination-isomer label of one conformer: 'trans' if a same-element donor pair is trans, else 'cis'.

    Geometries with no geometric isomerism (linear / trigonal-planar / tetrahedral) get no label ('').
    """
    if geometry in _NO_GEOMETRIC_ISOMERISM:
        return ""
    pos = mol.GetConformer(cid).GetPositions()
    for a in range(len(donors)):
        for b in range(a + 1, len(donors)):
            same = mol.GetAtomWithIdx(donors[a]).GetSymbol() == mol.GetAtomWithIdx(donors[b]).GetSymbol()
            if same and _vertex_angle(pos[donors[a]] - pos[metal], pos[donors[b]] - pos[metal]) > _TRANS_ANGLE:
                return "trans"
    return "cis"


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
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}
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
    dirs = VERTEX_DIRS.get(geometry)
    if dirs is None:
        return f"isomer{order}"
    if geometry in _NO_GEOMETRIC_ISOMERISM:
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


@dataclass
class Isomer:
    """One coordination isomer ready to embed: surrogate `mol`, polyhedron `cons`, `label`, `restore()`.

    `vertices[v]` is the donor atom seated at polyhedron vertex `v`, or ``VACANT`` for an empty pocket (used
    by ``coordinate=`` to seat a substrate donor there).
    """

    mol: Chem.Mol
    cons: Constraints
    metal: int
    donors: list
    real_z: int
    label: str
    geometry: str
    vertices: list = field(default_factory=list)
    extra: list = field(default_factory=list)  # other surrogated metals (idx, real_z) — spectators in a
    # multi-metal complex, restored alongside `metal`
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')

    def restore(self):
        """Restore this isomer's metal(s) from the surrogate back to their real elements."""
        restore(self.mol, self.metal, self.real_z)
        for mi, rz in self.extra:
            restore(self.mol, mi, rz)


def arrangement(iso):
    """Format a readable per-vertex ligand arrangement, e.g. ``'N3 Cl5 Cl6 ·'`` (``·`` = a vacant site).

    The *unambiguous* identity of an isomer, since the cis/trans label only describes a same-element pair
    and says nothing about where a vacancy sits. Order follows the polyhedron's `VERTEX_DIRS`.
    """

    def sym(d):
        return f"{iso.mol.GetAtomWithIdx(d).GetSymbol()}{d}" if d != VACANT else "·"

    return " ".join(sym(d) for d in iso.vertices)


class IsomerSet(list):
    """The coordination isomers of a metal centre: a ``list`` of `Isomer` to iterate, index, or pick from.

    Pick one to conf-search just the one you want:

        isos = rx.metal('CCCN[Pd](Cl)(Cl)NCCC', ['square_planar', 'tetrahedral']); isos.summary()
        ens  = rx.embed(isos.select(geometry='square_planar', label='trans')).mc().prune()
        ens  = rx.embed(isos[0]).mc().prune()                    # or just by index

    Enumeration is cheap (just the coordination constraints, no embedding); the expensive MC conformer
    search runs only on the `Isomer` you pick. The cis/trans `label` is a *coarse* tag (a same-element
    trans pair) — for an unambiguous choice use the **index** or the `arrangement` shown by `summary()`.
    """

    def select(self, geometry=None, label=None):
        """Return the single `Isomer` matching `geometry`/`label`.

        **Raises** if zero or several match (the label is coarse -- narrow it, or pick by index/`filter`).
        """
        hits = self.filter(geometry=geometry, label=label)
        if len(hits) != 1:
            raise ValueError(
                f"select(geometry={geometry!r}, label={label!r}) matched {len(hits)} isomer(s) — "
                f"{'narrow it or pick by index' if hits else 'no match'}; have "
                f"{[(i.geometry, i.label, arrangement(i)) for i in self]}"
            )
        return hits[0]

    def filter(self, geometry=None, label=None):
        """Return the subset matching `geometry`/`label`, as an `IsomerSet` (keep several / pick by index).

        `label` matches the **base** tag: ``'fac'`` matches the auto-numbered ``fac1``/``fac2`` (several
        heteroleptic isomers share a base label and get numbered) as well as a bare ``fac`` -- so ``select``
        then tells you to narrow, rather than ``filter('fac')`` silently missing ``fac1``.
        """

        def lab(i):
            return geometry in (None, i.geometry) and (
                label is None or i.label == label or i.label.rstrip("0123456789") == label
            )

        return IsomerSet(i for i in self if lab(i))

    def summary(self):
        """Print each isomer (index, geometry, cis/trans, arrangement) so you can pick one unambiguously.

        Returns self (chainable).
        """
        for k, i in enumerate(self):
            print(f"  [{k}] {i.geometry:18s} {i.label:6s}  {arrangement(i)}")
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


def enumerate_isomers(mol, geometry=None, center=None, fix=None):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    Returns an `IsomerSet`; each `Isomer` carries `.geometry` + `.label` -- pick with ``IsomerSet.select(…)``.

    `mol` is a SMILES or Mol. `geometry` selects the polyhedron(a): ``None`` → the default for the donor
    count (4 → square_planar); a name (``'octahedral'``, …) → just that one; a **list** of names → each, to
    embed several and compare energies (xTB/DFT) yourself. Supported names per coordination number are in
    ``GEOM_OPTIONS``.

    `center=` picks which metal to enumerate in a multi-metal complex (atom index or element symbol); the
    other metal(s) are retained at the input geometry (needs a conformer). **Vacant sites:** name a geometry
    with *more* vertices than donors and the empty vertex is left as a coordination pocket — present donors
    held at their polyhedron angles, the vacancy free (bind a substrate there with an explicit metal-donor
    distance).
    """
    if isinstance(mol, str):
        if mol.lower().endswith(".xyz"):  # a path -> perceived Mol with a geometry (so
            from rxembed.embed.dispatch import _xyz_to_mol  # rx.metal('complex.xyz', center=…) works, not

            mol = _xyz_to_mol(mol, 0)  # just rx.embed); the geometry is needed anyway
        else:  # to retain a spectator metal
            mol = Chem.AddHs(Chem.MolFromSmiles(mol))
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
    if len(metals) == 1 and center is None:
        base, m, donors, real_z = prepare(mol)  # the common single-metal case, unchanged
        extra, retain = [], Constraints()
    else:
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
        real_z = mol.GetAtomWithIdx(m).GetAtomicNum()
        spec = {
            s: (non_metal(mol.GetAtomWithIdx(s).GetNeighbors()), mol.GetAtomWithIdx(s).GetAtomicNum())
            for s in spectators
        }
        base, _metals_info, _ = prepare_all(mol)  # surrogate EVERY metal (conformer preserved)
        extra = [(s, spec[s][1]) for s in spectators]
        retain = Constraints()  # hold each spectator's SHAPE (relative pairwise,
        for s in spectators:  # frame-independent) — achiral, so chirality stays
            hold_shape(base, [s, *spec[s][0]], retain)  # random and is fixed by select_stereo afterward
        logger.info(
            "metal: enumerating %s%d; holding %d spectator metal(s) by %d shape constraints",
            mol.GetAtomWithIdx(m).GetSymbol(),
            m,
            len(spectators),
            len(retain.distances),
        )
    fix_cons = Constraints()
    frozen_donors = set()
    if fix:  # hold a reacting TS core at the input geometry
        from .builders import resolve_core  # while the rest of the coordination sphere is

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
    n = len(donors)
    if geometry is None:
        geoms = [geometry_for(n)]
        if geoms == [None]:
            raise ValueError(
                f"no default geometry for {n} donors — pass geometry= a name or list "
                f"(options for {n} donors: {GEOM_OPTIONS.get(n, [])})"
            )
    else:
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
    for g in geoms:
        if g not in ANGLES:
            hint = " — pass a list of names, e.g. ['square_planar', 'tetrahedral']" if g == "all" else ""
            raise ValueError(f"unknown geometry {g!r}; available: {sorted(k for k in ANGLES if k != 'None')}{hint}")
    out = IsomerSet()
    for geom in geoms:
        sites = n_sites(geom)
        if n > sites:
            raise ValueError(f"{geom} has {sites} coordination sites but the metal has {n} donors")
        padded = list(donors) + [VACANT] * (sites - n)  # leave empty vertices as a pocket
        if sites - n:
            logger.info(
                "metal[%s]: %d sites, %d donors -> %d vacant site(s) (coordination pocket)", geom, sites, n, sites - n
            )
        perms = None
        if frozen_donors and sites == n:  # pin each frozen donor at its input vertex and
            base_order = _input_ordering(base, m, padded, geom)  # GENERATE every free-donor permutation around
            if base_order is not None:  # them (filtering the canned, symmetry-reduced
                frozen_v = {v: di for v, di in enumerate(base_order) if padded[di] in frozen_donors}
                free_v = [v for v in range(sites) if v not in frozen_v]  # PERMUTATIONS list would miss the
                free_di = [di for di in range(len(padded)) if padded[di] not in frozen_donors]  # representative
                perms = []  # ordering a valid isomer happens to need)
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
        for order in isomers(base, padded, geom, perms=perms):
            cons = coordination(base, m, padded, geom, order, real_z, frozen=frozen_donors)
            cons.distances.update(retain.distances)  # hold any spectator metal(s)' shape (relative)
            cons.distances.update(fix_cons.distances)  # hold the frozen reacting core (relative pairwise)
            cons.frozen |= fix_cons.frozen  # + pin it in the relax
            od = [padded[k] for k in order]  # vertex -> donor atom (or VACANT)
            out.append(
                Isomer(
                    Chem.Mol(base),
                    cons,
                    m,
                    donors,
                    real_z,
                    _order_label(base, padded, geom, order),
                    geom,
                    od,
                    extra=extra,
                    stereo_ref=ref_sig,
                )
            )
    counts = Counter((i.geometry, i.label) for i in out)  # make every isomer uniquely selectable:
    nth = Counter()  # several heteroleptic isomers can share a
    for i in out:  # cis/trans label -> number them cis1, cis2…
        key = (i.geometry, i.label)
        if counts[key] > 1:
            nth[key] += 1
            i.label = f"{i.label}{nth[key]}"
    return out


_LONE_PAIR_Z = {7, 8, 15, 16}  # N O P S — substrate atoms that can coordinate


def lone_pair_donors(mol, metal, exclude=()):
    """Return substrate lone-pair donors (N/O/P/S) that could coordinate a vacant site.

    Excludes the metal and every atom in a **ligand fragment** -- a fragment containing the metal or one of
    `exclude` (the isomer's real donors). `prepare()` detaches the ligands into their own fragments, so
    without this a ligand-backbone heteroatom (an ether O on a phosphine, etc.) would look like a free
    substrate donor.
    """
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}
    ligand_frags = {frag[metal]} | {frag[d] for d in exclude}
    return [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _LONE_PAIR_Z and frag[a.GetIdx()] not in ligand_frags
    ]


def coordinate(iso, atoms, pad=0.15):
    """Build constraints seating substrate `atoms` into the metal's **vacant vertices**.

    The metal-donor coordinative distance + the polyhedron angles that orient each donor at its empty
    vertex. Reuses the real-ligand machinery: a substrate atom simply fills a VACANT slot. Returns a
    `Constraints` to merge.

    The window is ``(r_M + r_donor - 0.2, … + pad)``. The realised distance lands at the **upper bound**
    (the carbon surrogate has no metal-donor attraction — its vdW pushes them apart — so the restraint
    only caps the separation), so the upper bound is set to a realistic dative length ≈ covalent-sum +
    0.15 Å (a true distance needs the xTB calculator on the real metal). Raises if there are more atoms
    than vacant sites; `atoms` fill vacant vertices in order.
    """
    od = list(iso.vertices)
    vac = [v for v in range(len(od)) if od[v] == VACANT]
    if len(atoms) > len(vac):
        raise ValueError(
            f"{iso.geometry} {iso.label} has {len(vac)} vacant site(s) but {len(atoms)} atom(s) to coordinate"
        )
    r_m = _PT.GetRcovalent(iso.real_z)
    c = Constraints()
    for at, v in zip(atoms, vac, strict=False):
        od[v] = at
        d_md = r_m + _PT.GetRcovalent(iso.mol.GetAtomWithIdx(at).GetAtomicNum())  # dative ≈ covalent sum
        add_distance(c.distances, iso.metal, at, d_md - 0.2, d_md + pad)
    seated = set(atoms)
    for i, j, a in ANGLES[iso.geometry]:  # orient each newly-seated donor at its vertex
        if od[i] != VACANT and od[j] != VACANT and (od[i] in seated or od[j] in seated):
            c.angles[(od[i], iso.metal, od[j])] = (max(0.0, a - 8), min(180.0, a + 8))
    return c


def _input_ordering(mol, metal, donors, geometry):
    """Find the vertex ordering that best matches the **input geometry** (which donor at which vertex).

    ``od[vertex] = donors[order[vertex]]``, read from the conformer (orthogonal Procrustes over the
    candidate vertex orderings). Lets a ``fix=`` hold each frozen donor at its *real* vertex so only the
    free sites are enumerated -- otherwise the enumeration permutes a frozen donor into a vertex it cannot
    occupy, producing isomers that contradict the frozen core (a hydride forced off its TS site).
    """
    dirs_ref = VERTEX_DIRS.get(geometry)
    if dirs_ref is None or mol.GetNumConformers() == 0 or len(donors) != len(dirs_ref):
        return None
    pos = mol.GetConformer().GetPositions()
    dd = np.array([pos[d] - pos[metal] for d in donors], float)
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    v_ideal = np.array(dirs_ref, float)
    best_score, best_order = -1.0, list(range(len(donors)))
    for order in PERMUTATIONS[geometry]:
        h = dd[list(order)].T @ v_ideal  # cross-covariance of (donor-at-vertex) vs ideal vertex
        score = float(np.linalg.svd(h, compute_uv=False).sum())  # max alignment after the optimal rotation
        if score > best_score:
            best_score, best_order = score, order
    return best_order


def _central_trans(od, frag, dmat, dirs):
    """Return True if a tridentate chelate's *central* donor is placed **trans** to one of its own arms.

    The central donor is the one on the backbone path *between* the other two (``d(a,c)+d(c,b)==d(a,b)``).
    That is topologically impossible for a pincer (the central donor is cis to both arms in *every* real
    mer/fac), yet a flexible chelate can stretch to ~155° without a formally torn bond, so `bonding_ok`
    alone lets it through -- this drops it at enumeration instead of leaving a nonsensical isomer in the set.
    """
    by_frag = {}
    for p, d in enumerate(od):
        if d != VACANT:
            by_frag.setdefault(frag[d], []).append(p)
    for ps in by_frag.values():
        if len(ps) != _TRIAD:
            continue
        for ci in range(3):
            c, a, b = ps[ci], ps[(ci + 1) % 3], ps[(ci + 2) % 3]
            if abs(dmat[od[a]][od[c]] + dmat[od[c]][od[b]] - dmat[od[a]][od[b]]) < _COLINEAR_TOL:  # c is central
                if _vertex_angle(dirs[c], dirs[a]) > _TRANS_ANGLE or _vertex_angle(dirs[c], dirs[b]) > _TRANS_ANGLE:
                    return True
                break
    return False


def isomers(mol, donors, geometry, perms=None):
    """Enumerate distinct coordination isomers: **every** distinct vertex arrangement, no geometric pre-filter.

    Bar one topological impossibility: a tridentate's central donor trans to its own arm (see
    `_central_trans`).

    `perms` overrides the candidate vertex orderings (default ``PERMUTATIONS[geometry]``) — a ``fix=``
    enumeration passes the subset that keeps each frozen donor pinned to its input vertex.

    Whether a chelate can physically reach a given arrangement is a question of *geometry*, not topology, so
    we do not guess it here (a "bidentate can't span trans" / "tridentate can" rule mis-handles long-bridge
    ligands either way). Instead every arrangement is enumerated and the geometrically impossible ones (a
    chelate forced to span a bite it cannot reach) embed with a torn ligand bond and are dropped downstream
    by `bonding_ok` in `Ensemble.minimize` — geometry is the arbiter.

    Dedup is **connectivity-aware**: the signature carries, per donor pair, the elements, the vertex angle,
    **and** the intra-ligand bond distance for a same-ligand pair. So genuinely distinct arrangements that
    share an element/angle pattern — a tridentate's terminal-trans (mer) vs the central-trans (impossible)
    one — are kept apart rather than merged (merging them would let the impossible one mask the real mer).
    """
    perms = perms if perms is not None else PERMUTATIONS.get(geometry, [list(range(len(donors)))])
    dirs = VERTEX_DIRS.get(geometry)
    if dirs is None:
        return perms
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != VACANT else "X") for d in donors}  # vacancy = "X"
    frag = {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances

    def link(od, p, q):  # intra-ligand bond distance of a same-ligand pair
        a, b = od[p], od[q]  # (distinguishes a chelate's central vs terminal
        if VACANT in (a, b) or frag[a] != frag[b]:  # donor); -1 for different ligands / a vacancy
            return -1
        return int(dmat[a][b])

    seen, out = set(), []
    for order in perms:
        od = [donors[k] for k in order]  # od[position] = donor atom (or VACANT) at that polyhedron vertex
        if _central_trans(od, frag, dmat, dirs):  # a tridentate's central donor trans to its arm -> impossible
            continue
        pairs = [(p, q) for p in range(len(od)) for q in range(p + 1, len(od))]
        sig = tuple(
            sorted(
                (tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), _vertex_angle(dirs[p], dirs[q]))
                for p, q in pairs
            )
        )
        if sig not in seen:
            seen.add(sig)
            out.append(order)
    return out
