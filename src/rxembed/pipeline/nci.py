"""Non-covalent contacts, as constraints and as a binding-mode signature.

Contacts become `constrain` windows; the signature is what dedups two poses that grip the same way. The one
`KINDS` registry drives both, and the enumerator is generic over it: a new contact type is a row, not a
branch. This is the only module that touches xyzgraph's NCI detection.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from rdkit import Chem

from rxembed.bounds import probe_conformer
from rxembed.metal_core import frag_map, metal_indices
from rxembed.utils import CARBON_Z, atom_label

if TYPE_CHECKING:
    from xyzgraph.nci import NCIAnalyzer


def _graph(mol: Chem.Mol, conf_id: int = -1) -> nx.Graph:
    conf = mol.GetConformer(conf_id)
    g = nx.Graph()
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        g.add_node(
            a.GetIdx(),
            symbol=a.GetSymbol(),
            atomic_number=a.GetAtomicNum(),
            position=(p.x, p.y, p.z),
            formal_charge=a.GetFormalCharge(),
        )
    for b in mol.GetBonds():
        g.add_edge(
            b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_order=1.5 if b.GetIsAromatic() else b.GetBondTypeAsDouble()
        )
    for i in g.nodes:
        g.nodes[i]["valence"] = sum(g[i][j]["bond_order"] for j in g.neighbors(i))
        g.nodes[i]["agg_charge"] = float(g.nodes[i]["formal_charge"])
    g.graph["aromatic_rings"] = [tuple(map(int, r)) for r in Chem.GetSymmSSSR(mol)]
    return g


def analyzer(mol: Chem.Mol) -> NCIAnalyzer:
    """Build an xyzgraph ``NCIAnalyzer`` from ``mol`` (bonds/positions from its conformer)."""
    try:
        from xyzgraph.nci import NCIAnalyzer
    except ImportError as exc:
        raise ImportError("analyzer needs xyzgraph; pip install 'rxembed[workflow]'") from exc

    return NCIAnalyzer(_graph(mol))


_HB_DONOR_Z = {7, 8, 16}  # N O S: polar H-bond donor heavies; a charge-assisted
# H-bond is promoted from IONIC only off one of these (a
# C-H on an iminium/tropylium cation is not a real donor)
_ACCEPTOR_Z = {7, 8, 9}  # N O F: lone-pair H-bond acceptors (for M-H; the
# (1.6,2.2) window is tuned for these; S/Cl need a
# longer window, to add later)

# A sigma-hole needs a polarisable heavy donor, so the element list is the physics, not a convenience:
# F has no usable hole and O/N are the acceptors, not the donors.
_SIGMA_HOLE_Z = {"XB": {17, 35, 53, 85}, "ChB": {16, 34, 52}, "PnB": {33, 51, 83}}


@dataclass(frozen=True)
class ContactKind:
    """One non-covalent contact *type*: the single place its behaviour is described.

    Adding a kind is one registry row, not a new code branch.

    - ``family``: how its candidates are enumerated. ``'atom'`` (an xyzgraph atom-pair donor→acceptor),
      ``'ring'`` (an atom placed over a ring centroid), or ``'hydride'`` (a terminal metal hydride).
    - ``window``: the contact-distance window ``(lo, hi)`` Å for atom/hydride; the atom-centroid distance
      (a single float) for ring.
    - ``orient``: the orientation-angle window ``(lo, hi)`` ° that makes a *directional* contact real
      (H-bond bends to ~140°, sigma-holes are sharply linear); ``None`` for a non-directional contact.
    - ``apex``: how to find the orientation apex. ``'donor'`` (HB: the donor-heavy atom), ``'sigma'``
      (the anti-periplanar R of an R-X···A sigma-hole), ``'metal'`` (M-H), else ``None``.
    - ``anchor``: the near atom. Atom family: ``'donor_h'`` (HB) | ``'heavy'`` (sigma-hole/ionic);
      ring family: ``'ring_h'`` (the donor H) | ``'ring_ion'`` (the ion); hydride family: the hydride.
    """

    name: str
    family: str
    window: object
    orient: tuple | None = None
    apex: str | None = None
    anchor: str = "heavy"


# One row per contact type. Ring values are centroid heights, expanded to atom windows downstream.
KINDS = {
    k.name: k
    for k in [
        ContactKind("HB", "atom", (1.6, 2.2), orient=(140.0, 180.0), apex="donor", anchor="donor_h"),
        ContactKind("XB", "atom", (2.5, 3.1), orient=(160.0, 180.0), apex="sigma"),  # GFN-FF-surviving on I...pyridine
        ContactKind("ChB", "atom", (3.0, 3.6), orient=(155.0, 180.0), apex="sigma"),
        ContactKind("PnB", "atom", (3.0, 3.6), orient=(155.0, 180.0), apex="sigma"),
        ContactKind("IONIC", "atom", (2.6, 3.8)),
        ContactKind("CATLP", "atom", (2.6, 3.4)),
        ContactKind("CHPI", "ring", 3.0, anchor="ring_h"),
        ContactKind("HBPI", "ring", 3.0, anchor="ring_h"),
        ContactKind("CATPI", "ring", 3.5, anchor="ring_ion"),  # generic: GFN-FF optima span 1.55-4.41 A by cation
        ContactKind("ANPI", "ring", 3.5, anchor="ring_ion"),
        ContactKind("HALPI", "ring", 3.5, anchor="ring_ion"),
        ContactKind("MH", "hydride", (1.6, 2.2), orient=(140.0, 180.0), apex="metal", anchor="hydride"),
    ]
}


@dataclass
class Contact:
    """A seeded non-covalent contact: a distance window, plus an orientation angle for a directional one.

    A held distance alone gives a bent, weakly-detected contact, so a directional contact (H-bond,
    sigma-hole) also carries an orientation-angle window: ``distances`` is ``{(i, j): (lo, hi)}`` and
    ``angles`` is ``{(i, j, k): (lo, hi)}`` in degrees (donor-heavy, H, acceptor). Plugs into
    ``rx.embed(contacts=...)``. ``near``/``far`` (the donor/anchor atom; the acceptor atom or ring tuple)
    carry no constraint themselves, but group contacts into compatible binding modes.
    """

    distances: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)
    label: str = ""
    near: int | None = None
    far: int | tuple[int, ...] | None = None


def _donor_h(mol, heavy):
    return next((a.GetIdx() for a in mol.GetAtomWithIdx(heavy).GetNeighbors() if a.GetAtomicNum() == 1), None)


def _sigma_apex(mol, x, acceptor, pos):
    """Pick the heavy R bonded to `x` that best caps the R-x...acceptor sigma-hole cone.

    The most *anti-periplanar* neighbour to the x->acceptor direction. A chalcogen/pnictogen bears several
    sigma-holes (one anti to each X-R bond), so pick by geometry in the rough conformer `pos`, not
    topological order. None if `x` has no heavy neighbour other than the acceptor.
    """
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(x).GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != acceptor]
    if not nbrs:
        return None
    va = pos[acceptor] - pos[x]
    cones = []
    for r in nbrs:
        vr = pos[r] - pos[x]
        cosine = vr @ va / (np.linalg.norm(vr) * np.linalg.norm(va) + 1e-9)
        cones.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    return nbrs[cones.index(max(cones))]  # R most opposite the acceptor (R-X-A nearest 180 deg)


def _atom_over_ring(pos, atom, ring, d_centroid):
    """Constrain `atom` over the ring centroid via per-ring-atom distances (no centroid dummy needed).

    Each ring-atom distance is sqrt(d^2 + r^2), placing the atom ``d_centroid`` above the ring plane.
    """
    c = pos[list(ring)].mean(0)
    out = []
    for k in ring:
        dk = (d_centroid**2 + float(np.linalg.norm(pos[k] - c)) ** 2) ** 0.5
        out.append((min(atom, k), max(atom, k), round(dk - 0.3, 2), round(dk + 0.3, 2)))
    return out


def candidate_contacts(mol, kinds=tuple(KINDS), inter_fragment=True, seed=0xC0FFEE):
    """Candidate NCI contacts enumerated from topology (xyzgraph `_pairs`), not a random geometry.

    Finds the real H-bond / halogen-bond / cation-pi / CH-pi candidates (every thiourea N-H to every
    substrate O) independent of the pose, plus terminal metal hydrides (M-H). Returns ``{label: Contact}``,
    plug into ``embed(contacts=...)``; every kind is one ``KINDS`` row. A rough conformer is embedded
    (deterministically, `seed`) only when the input has none, since its geometry decides the sigma-hole
    apex, the M-H nearest acceptor, and ring radii.
    """
    work = Chem.Mol(mol)
    if work.GetNumConformers() == 0:
        # `seed` differs from the embed default on purpose: moving it would move every found contact.
        work = probe_conformer(mol, seed) or work
    pos = work.GetConformer().GetPositions()
    an = analyzer(work)
    frag = frag_map(work)
    handler = {"atom": _atom_contacts, "ring": _ring_contacts, "hydride": _metal_hydride_contacts}
    out, seen = {}, set()
    for name in kinds:
        kind = KINDS.get(name)
        if kind is not None:
            out.update(handler[kind.family](work, an, pos, frag, seen, inter_fragment, kind))
    return out


# auto considers the directional/atom + ionic + M-H kinds; ring/π (CHPI…) are geometrically soft and a
# rough reference conformer surfaces many spurious ones, so they are off by default (request explicitly).
_AUTO_KINDS = ("HB", "XB", "ChB", "PnB", "IONIC", "CATLP", "MH")
# anchors: strong enough to define a binding mode on their own; the rest only ride along a strong combo.
_ANCHOR_KINDS = {"HB", "IONIC", "CATLP", "MH"}
# H-bonds (incl. charge-assisted ones promoted from IONIC in _atom_contacts) lead: directional, the real
# structure-definers. A bare ion pair with no bridging H (R4N+···Cl-) is strong but non-directional, so it
# ranks below HB rather than above it: a salt bridge that is really an anion-H-bond is described as the HB.
_STRENGTH = {"CATLP": 4, "HB": 4, "MH": 4, "IONIC": 3, "XB": 3, "ChB": 2, "PnB": 2}
# only a bent contact (H-bond / electrostatic) can share an acceptor (a bifurcated clamp); a sigma-hole or a
# metal-hydride is sharply linear and several cannot all aim at one acceptor, so those cap at 1.
_BIFURCATING = {"HB", "IONIC", "CATLP"}


def _kind(c):
    return c.label.split(":", 1)[0]


def _donor_heavy(mol, c):
    """Return the heavy atom at the donor end of `c` (for the reciprocal-H-bond check).

    The heavy neighbour of the donor H for an HB/M-H, else the near atom itself (a bare ion / sigma-hole donor).
    """
    a = mol.GetAtomWithIdx(int(c.near))
    if a.GetAtomicNum() == 1:
        return next((n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1), None)
    return int(c.near)


def _is_reverse(c, o, dh):
    """Return whether `c` and `o` are a reciprocal H-bond pair (c is A-H...B and o is B-H...A).

    An impossible 2-cycle in one pose (the two donor Hs can't both sit in the single A...B channel). `dh`
    maps id->donor-heavy.
    """
    dc, ac, do, ao = dh.get(id(c)), c.far, dh.get(id(o)), o.far
    return dc is not None and do is not None and isinstance(ac, int) and isinstance(ao, int) and dc == ao and ac == do


def _resolve_reciprocal(sk, dh):
    """Split a skeleton gripping a reciprocal H-bond pair into its two physical one-directional poses.

    A 2-cycle (A-H...B and B-H...A) can't coexist in one geometry, but each direction is a real binding
    mode, so each must survive as its own; larger H-bond rings (A->B, B->C, C->A) pass through untouched.
    Done as a post-step on the assignments, not an in-branch guard, so a singly-optioned mutual pair still
    yields both directions.
    """
    pairs = [(c, o) for i, c in enumerate(sk) for o in sk[i + 1 :] if _is_reverse(c, o, dh)]
    if not pairs:
        return [sk]
    variants = [list(sk)]
    for c, o in pairs:  # drop one side of each reciprocal pair (product)
        nxt = []
        for v in variants:
            if any(x is c for x in v) and any(x is o for x in v):
                nxt.append([x for x in v if x is not o])  # keep c's direction
                nxt.append([x for x in v if x is not c])  # keep o's direction
            else:
                nxt.append(v)
        variants = nxt
    return variants


def _acc_key(c):
    if isinstance(c.far, tuple):
        return ("ring", *c.far)
    return ("atom", int(c.far))


def _acceptor_quality(mol, ak):
    """Return the relative H-bond basicity of an acceptor (higher = better), used only to rank modes.

    A grip onto a genuinely basic lone pair beats one onto a spent one: an amine/imine/iminophosphorane N
    (very basic) accepts strongly; a carbonyl/phosphoryl O or thione S accepts well; but an amide/
    (thio)urea/aromatic N accepts poorly (delocalised into the adjacent pi system, it is the very N that is
    itself a good H-bond donor).
    """
    if ak[0] != "atom":
        return 1
    a = mol.GetAtomWithIdx(ak[1])
    if a.GetAtomicNum() == 7:  # noqa: PLR2004  nitrogen
        if a.GetIsAromatic():
            return 0  # pyrrole/amide-like aromatic N: lone pair in ring
        for nb in a.GetNeighbors():  # amide / amidine / (thio)urea: N-C(=O/=S/=N)
            if nb.GetAtomicNum() == CARBON_Z and any(
                b.GetBondTypeAsDouble() >= 2 and b.GetOtherAtom(nb).GetAtomicNum() in (7, 8, 16)  # noqa: PLR2004
                for b in nb.GetBonds()
            ):
                return 0
        return 2  # amine / imine / iminophosphorane N: strong base
    return 1  # O / S / halide lone-pair acceptor: decent


def _accepts(c, count, cap):
    """Return whether the acceptor of `c` can take it (bent contacts up to `cap`, linear ones only one).

    Bent contacts bifurcate up to `cap`; linear ones (sigma-hole, M-H) take only one. `count[acc]` is
    ``[#bent, #linear]`` already seated there.
    """
    bent, lin = count[_acc_key(c)]
    return bent < cap if _kind(c) in _BIFURCATING else lin < 1


def _seat(c, count, delta):
    cls = 0 if _kind(c) in _BIFURCATING else 1
    count[_acc_key(c)][cls] += delta


def _maximal_assignments(by_near, cap, hard_cap=256):
    """Enumerate every maximal donor->acceptor assignment of the anchor contacts.

    Each donor (a `by_near` group of its mutually-exclusive contacts) takes one contact whenever an acceptor
    still has spare capacity (so a donor is only left unplaced when it geometrically cannot be, keeping
    each assignment maximal), branching on *which* acceptor it grips. `by_near` is an ordered
    ``[(near, [contacts]), ...]``. Reciprocal 2-cycles are resolved afterwards (``_resolve_reciprocal``).
    """
    out = []

    def bt(i, chosen, count):
        if len(out) >= hard_cap:
            return
        if i == len(by_near):
            out.append(list(chosen))
            return
        placeable = [c for c in by_near[i][1] if _accepts(c, count, cap)]
        if not placeable:  # this donor can't be placed -> skip, still maximal
            bt(i + 1, chosen, count)
            return
        for c in placeable:
            _seat(c, count, +1)
            chosen.append(c)
            bt(i + 1, chosen, count)
            chosen.pop()
            _seat(c, count, -1)

    bt(0, [], defaultdict(lambda: [0, 0]))
    return out


def _augment_with_aux(combo, aux, cap):
    """Ride every geometrically-compatible auxiliary sigma-hole (XB/ChB/PnB) onto an anchor grip.

    An aux rides along only where its donor is still free and the acceptor has spare capacity, so a weak
    halogen bond strengthens a real clamp but never stands alone. Mutates and returns `combo`.
    """
    used = {c.near for c in combo}
    count = defaultdict(lambda: [0, 0])
    for c in combo:
        _seat(c, count, +1)
    for c in aux:
        if c.near not in used and _accepts(c, count, cap):
            combo.append(c)
            used.add(c.near)
            _seat(c, count, +1)
    return combo


def _mode_score(mol, contacts):
    """Return a binding mode's sort key: contact count, then strength, acceptor basicity and bifurcated clamps."""
    accs = defaultdict(int)
    for c in contacts:
        accs[_acc_key(c)] += 1
    return (
        len(contacts),  # most contacts (best grip)
        sum(_STRENGTH.get(_kind(c), 0) for c in contacts),  # then strongest contact types
        sum(_acceptor_quality(mol, _acc_key(c)) for c in contacts),  # then each H-bond onto a basic acceptor
        sum(1 for v in accs.values() if v > 1),  # then bifurcated clamps
    )


def _mode_label(mol, contacts):
    """Name a binding mode by its contacts, grouped by acceptor, as ``kinds:donors->acceptor`` parts."""
    groups = defaultdict(list)
    for c in contacts:
        groups[_acc_key(c)].append(c)
    parts = []
    for ak, gs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ks = "/".join(sorted({_kind(c) for c in gs}))
        donors = ",".join(atom_label(mol, c.near) for c in gs)
        acc = f"ring{ak[1]}" if ak[0] == "ring" else atom_label(mol, ak[1])
        parts.append(f"{ks}:{donors}->{acc}")
    return " + ".join(parts)


def auto_binding_modes(mol, kinds=_AUTO_KINDS, inter_fragment=True, acceptor_cap=2, max_modes=8, seed=0xC0FFEE):
    """Enumerate cooperative binding modes, each a maximal combination of compatible contacts.

    Multipoint binding, not any single weak contact, stabilises these complexes. Anchors (H-bond, ionic,
    cation-lone-pair, metal-hydride) are matched to acceptors first, up to `acceptor_cap` each; compatible
    auxiliary sigma-holes then ride a strong grip without standing alone, so with no anchor present each
    auxiliary is its own weak mode. Returns the top `max_modes`, ranked by contact count then strength, as
    ``{mode_label: Contact}``; drive with ``embed(contacts='auto')`` or pass one to ``embed(contacts=)``.
    """
    # Bootstraps a rough conformer itself if `mol` has none, so SMILES input works.
    cands = candidate_contacts(mol, kinds=kinds, inter_fragment=inter_fragment, seed=seed)
    anchors = [c for c in cands.values() if _kind(c) in _ANCHOR_KINDS]
    aux = sorted(
        (c for c in cands.values() if _kind(c) not in _ANCHOR_KINDS), key=lambda c: -_STRENGTH.get(_kind(c), 0)
    )

    dh = {id(c): _donor_heavy(mol, c) for c in anchors}  # donor-heavy of each anchor, for reciprocal split
    by_near = OrderedDict()
    for c in anchors:
        by_near.setdefault(c.near, []).append(c)
    skeletons = _maximal_assignments(list(by_near.items()), acceptor_cap) if anchors else [[c] for c in aux]
    skeletons = [r for sk in skeletons for r in _resolve_reciprocal(sk, dh)]  # 2-cycle -> two 1-way modes
    augment = bool(anchors)  # anchor combos get aux riders; weak-only don't

    seen, ranked = set(), []
    for sk in sorted(skeletons, key=partial(_mode_score, mol), reverse=True):
        key = frozenset(id(c) for c in sk)
        if sk and key not in seen:
            seen.add(key)
            ranked.append(sk)

    modes = OrderedDict()
    for sk in ranked[:max_modes]:
        combo = _augment_with_aux(list(sk), aux, acceptor_cap) if augment else list(sk)
        m = Contact()
        for c in combo:
            m.distances.update(c.distances)
            m.angles.update(c.angles)
        m.label = _mode_label(mol, combo)
        modes[m.label] = m
    return modes


def _atom_contacts(work, an, pos, frag, seen, inter_fragment, kind):
    """Build atom-pair contacts (HB / sigma-hole / ionic) from xyzgraph donor->acceptor pairs.

    Each pair gives a distance and, for a directional kind, the linear orientation angle.
    """
    out = {}
    for pair in an._pairs.get(kind.name, []):
        a, b = pair[0], pair[1]
        if not (isinstance(a, int) and isinstance(b, int)) or (inter_fragment and frag[a] == frag[b]):
            continue
        eff = kind
        if kind.name == "IONIC":
            # A salt bridge whose ion bears an H is a charge-assisted (anion) H-bond: describe it as the
            # directional HB (collapses onto the real HB pair if xyzgraph typed it, so no double count).
            # A bare ion pair with no polar donor H either side stays IONIC.
            if work.GetAtomWithIdx(a).GetAtomicNum() in _HB_DONOR_Z and _donor_h(work, a) is not None:
                eff = KINDS["HB"]
            elif work.GetAtomWithIdx(b).GetAtomicNum() in _HB_DONOR_Z and _donor_h(work, b) is not None:
                a, b = b, a  # the H-bearing ion is the H-bond donor
                eff = KINDS["HB"]
        allow = _SIGMA_HOLE_Z.get(eff.name)  # sigma-hole only on a polarisable heavy donor;
        if allow is not None:
            da = work.GetAtomWithIdx(a)
            if da.GetAtomicNum() not in allow:
                continue  # drop F (XB), O (ChB), N/P (PnB) false positives
            if any(bd.GetBondType() == Chem.BondType.DOUBLE for bd in da.GetBonds()):
                # a pi-bonded chalcogen (a thione/thiourea C=S) is electron-rich (an H-bond acceptor),
                # not a sigma-hole donor: its constraint would just pucker the sp2 group
                continue
        anchor = _donor_h(work, a) if eff.anchor == "donor_h" else a
        if anchor is None or (eff.name, anchor, b) in seen:
            continue
        seen.add((eff.name, anchor, b))
        label = f"{eff.name}:{atom_label(work, a)}->{atom_label(work, b)}"
        angles = {}
        if eff.orient:  # hold the contact linear, not just close
            apex = a if eff.apex == "donor" else _sigma_apex(work, a, b, pos) if eff.apex == "sigma" else None
            if apex is not None and apex not in (anchor, b) and frag[apex] != frag[b]:
                angles = {(apex, anchor, b): eff.orient}  # inter-fragment only (else smoothing can fail)
        out[label] = Contact(
            distances={(min(anchor, b), max(anchor, b)): eff.window}, angles=angles, label=label, near=anchor, far=b
        )
    return out


def _ring_contacts(work, an, pos, frag, seen, inter_fragment, kind):
    """Build atom-over-ring contacts (CH-pi / cation-pi / ...) as per-ring-atom distances.

    The distances place the atom over the centroid: the geometry itself orients it, so no separate angle.
    """
    out = {}
    for pair in an._pairs.get(kind.name, []):
        atom = pair[0][1] if kind.anchor == "ring_h" else pair[0]  # (donor,H)->H, or the ion
        ring = tuple(pair[1])
        if not isinstance(atom, int) or (inter_fragment and any(frag[atom] == frag[r] for r in ring)):
            continue
        if (kind.name, atom, ring) in seen:
            continue
        seen.add((kind.name, atom, ring))
        label = f"{kind.name}:{atom_label(work, atom)}->ring{ring[0]}"
        dists = {(i, j): (lo, hi) for i, j, lo, hi in _atom_over_ring(pos, atom, ring, kind.window)}
        out[label] = Contact(distances=dists, label=label, near=atom, far=ring)
    return out


def _metal_hydride_contacts(work, an, pos, frag, seen, inter_fragment, kind):
    """Build terminal metal-hydride (M-H) H-bond donor contacts, one per hydride to its nearest acceptor.

    Nearest N/O/F acceptor in another fragment (so a substrate with many heteroatoms doesn't spawn a
    contradictory pile). Structural detection (a terminal H on a coordination centre), not SMARTS, so it
    works on an xyz-perceived Mol too. Dev-stage simplification: an M-H is always a donor (the metal is the 'donor
    heavy', orientation M-H...A linear); the di-hydrogen-bond and S/Cl-acceptor cases are left for later.
    xyzgraph does not type these, so they are *seeded* but not *re-detected* by binding_modes().
    """
    out = {}
    metals = metal_indices(work)
    hydrides = [
        (metal, nbr.GetIdx())
        for metal in metals
        for nbr in work.GetAtomWithIdx(metal).GetNeighbors()
        if nbr.GetAtomicNum() == 1 and nbr.GetDegree() == 1
    ]
    for metal, h in hydrides:
        accs = [
            a.GetIdx()
            for a in work.GetAtoms()
            if a.GetAtomicNum() in _ACCEPTOR_Z
            and a.GetIdx() != metal
            and not (inter_fragment and frag[h] == frag[a.GetIdx()])
        ]
        if not accs:
            continue
        ai = min(accs, key=lambda x: float(np.linalg.norm(pos[h] - pos[x])))  # nearest acceptor only
        label = f"{kind.name}:{atom_label(work, metal)}H{h}->{atom_label(work, ai)}"
        out[label] = Contact(
            distances={(min(h, ai), max(h, ai)): kind.window},
            angles={(metal, h, ai): kind.orient},
            label=label,
            near=h,
            far=ai,
        )
    return out
