"""Metal-complex coordination constraints — the polyhedron, per isomer.

Coordination geometries, L-M-L angles, and isomer permutations are adapted from TMC_embed code by
Maria H. Rasmussen, **TMC_embed** (https://github.com/jensengroup/TMC_embed).

The metal is held purely by distance + angle constraints (its bonds removed) and embedded/relaxed with a
UFF-typeable surrogate atom in its place — so the whole path is plain RDKit + UFF, no xtb.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from rdkit.Geometry import Point3D

from rxembed.rdkit_embed import io as _io

from . import polyhedron as _poly
from .base import Constraints, add_distance, add_pairwise_shape

# The polyhedron registry + accessors now live in the leaf `polyhedron.py`; re-export them so metal.py stays
# their import home for every consumer (isomers, coordination_builders, solver, pipeline, tests).
from .polyhedron import (  # noqa: F401 — re-exports; Polyhedron/is_planar/isomer_permutations aren't used inside metal.py
    POLYHEDRA,
    Polyhedron,
    _vertex_angle,
    geometries_for_cn,
    is_planar,
    isomer_permutations,
    vertex_dirs,
)

logger = logging.getLogger("rxembed.constraints.metal")  # pinned name: kept under the "rxembed" logger tree
#   (set_verbose configures) and the exact child caplog filters on, unchanged by the carve to rdkit_embed.

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
# transition / lanthanide / actinide metals — their coordination distances are dative, not vdW clashes, and their
# "bonds" aren't covalent (the same set `metrics.bonding_ok` uses); `geometry.check` excludes them so a metal centre
# isn't read as clashing with its own ligands, and the donor-perception ruler below strips exactly these bonds first.
# Broader than `TRANSITION_METALS` above (it spans the full f-block), because a fold/donor question is meaningful for
# any coordination centre, not only the ones `rx.metal(...)` enumerates polyhedra for.
_METAL_Z = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
SURROGATE = 6  # carbon: its excluded volume stops a ligand folding into the metal, so the distance geometry keeps it
FF_SURROGATE = 3  # lithium: a bond-less, UFF-typeable force-field surrogate. With the M-donor bonds stripped Li carries
# only a vdW term — a soft excluded-volume sphere holding every non-donor (heavy and H) off the metal, while donors are
# held explicitly (`cons.distances` + `pulls`), stiff enough to win against it. Must stay bond-less: bonded, UFF types
# Li linear (theta0=180) and its angle term 1/(4 sin^2 theta0) is singular, injecting ~1e9 kcal/mol into a CN>=3 sphere.
UFF_GHOST = 54  # xenon: the FF type for a haptic centroid dummy (`cons.phantoms`). It sits ~0.8 A inside its own ring,
# where a real element's r^-12 vdW is astronomical (carbon ~6e7 kcal), so it must carry ZERO terms — reached the only
# way RDKit allows: an element the UFF typer cannot type, whose terms it then omits. Kept in the mol so its restraints
# (M->centroid, centroid->ring) survive. Only metal-side phantoms are ghosted; a D-cap keeps its FF terms.


def _frag_map(mol):
    """Map each atom index -> its fragment id (same ligand = same fragment)."""
    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}


def _vertex_atom(haptic, v):
    """Resolve a coordination vertex to its representative REAL atom — a centroid dummy maps through its ring.

    A haptic centroid is a bond-less dummy carrying none of the fragment identity, topological distance or
    backbone reach its ring atoms do. EVERY vertex-keyed enumeration lookup (the dedup signature, the
    trans-span / central-trans pre-filters, the chelate bite graph) must resolve through here first, or a face
    TETHERED to a co-donor (an ansa/constrained-geometry Cp) reads as a separate ligand and its impossible trans
    placement is never dropped. Any one ring atom carries the identity the dummy lacks.
    """
    return haptic[v][0] if haptic and v in haptic else v


def metal_index(mol):
    """Index of the first transition-metal atom, or None."""
    return next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS), None)


_MIN_STEREO_NEIGHBOURS = 3  # a tetrahedral stereocentre needs >=3 explicit neighbours (else ETKDG raises)
_TETRAVALENT = 4  # a fully-substituted (already degree-4) donor is not a candidate for the D-cap chirality hold


def _clear_labile_donor_stereo(atom):
    """Drop a chiral tag on a metal-donor atom that has fallen below 3 neighbours after the sphere strip.

    A donor that is a stereocentre only *while bonded to the metal* — a planar amidate N- (metal + aryl +
    alpha-C = 3 neighbours, 2 without) — leaves a stale tag that crashes ETKDG (``nbrs.size() >= 3``). A genuine
    donor stereocentre keeps 3 real substituents after the strip (a chiral-at-P, a carbanion C⁻ donor) and is
    untouched, so its enantiomers still embed distinctly.
    """
    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and atom.GetDegree() < _MIN_STEREO_NEIGHBOURS:
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)


def _labile_donors(mol, donors):
    """Return the metal-bound donors whose config the surrogate can't hold natively: sp3 **C/N** with a chiral tag.

    A carbanion-C / amine-N donor is a stereocentre only WHILE metal-bound; the surrogate strips that bond, so
    it becomes a bare degree-3 centre RDKit/UFF will invert. A heavy pnictogen (P/As/Sb) stays configurationally
    stable as degree-3 and needs no help, so it is excluded. `donors` may be ``None`` (a `_MetalCtx` from the
    frozen-core / general metal path doesn't track them) -> no labile donors.
    """
    return [
        d
        for d in (donors or ())
        if mol.GetAtomWithIdx(d).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        and mol.GetAtomWithIdx(d).GetAtomicNum() in (6, 7)
        and mol.GetAtomWithIdx(d).GetTotalDegree() < _TETRAVALENT
    ]


def donor_chirality_sign(mol, cid, donor):
    """Geometric hand (+1 / -1, or None) of a donor: the signed volume of its first three neighbours.

    Neighbour order is stable for a fixed mol, so the sign is comparable across that mol's conformers — used to
    cull a conformer whose labile donor inverted (a relax / re-embed / mc stray) back to the enumerated hand.
    """
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors()]
    if len(nbrs) < _MIN_STEREO_NEIGHBOURS:
        return None
    conf = mol.GetConformer(cid)
    p = np.array([list(conf.GetAtomPosition(i)) for i in [donor, nbrs[0], nbrs[1], nbrs[2]]])
    v = float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))
    return int(np.sign(v)) if abs(v) > 1e-6 else None  # noqa: PLR2004 — a near-planar (racemising) centre: no hand


_DUMMY_M_LO, _DUMMY_M_HI = 0.8, 1.8  # Å: pin the hold-dummy D near the metal (~ the coordinate-bond / lone-pair side)


def _shift_phantoms(cons, offset):
    """Move every reserved haptic-centroid index up by ``offset``, re-keying every field that names one.

    Two transients are appended from the REAL atom count — a haptic centroid dummy and a labile donor's D-cap —
    but `materialise_phantoms` requires the centroid keys to be the consecutive block after that count, so
    whichever is appended second collides. This makes the caps take the low indices and the centroid block slide
    above them (the composition is 31% of the tmQM haptic corpus — a Cp face whose atoms are also anionic sp3
    stereocentres).
    """
    if not cons.haptic or not offset:
        return
    old = set(cons.haptic)
    bump = {i: i + offset for i in old}  # every reserved index moves; a real atom index never does

    def key(k):
        return tuple(bump.get(i, i) for i in k)

    cons.haptic = {bump[d]: ring for d, ring in cons.haptic.items()}
    cons.phantoms = frozenset(bump[p] for p in cons.phantoms)
    for name in ("distances", "pulls", "floors", "dg_floors"):
        setattr(cons, name, {key(k): v for k, v in getattr(cons, name).items()})
    cons.angles = {key(k): v for k, v in cons.angles.items()}
    cons.spheres = tuple(  # the re-solve recipe names the dummies too (donors/order index REAL atoms only)
        s._replace(haptic=tuple((bump[d], ring) for d, ring in s.haptic)) for s in cons.spheres
    )


def _hold_donor_chirality(mol, metal, donors, cons):
    """Cap each labile (sp3 C/N) metal-bound donor carrying a chiral tag with a dummy D, so the hand is enforced.

    A degree-3 carbanion/amine donor (no M-C bond in the surrogate) is not a stereocentre RDKit/UFF perceives, so
    its two enumerated hands relax to the same geometry. Neutralising its charge and adding a 4th bond to a
    **deuterium** makes it a proper tetrahedral centre in the same appended-D basis the enumeration labelled it.
    With conformers (relax/mc) each D goes at the 4th vertex of THIS conformer's hand; without (initial embed) the
    hand comes from ETKDG + the tag. A **(metal, D) distance** pins D on the coordinate-bond side (without it ETKDG
    thrashes ~200x slower); `_release_donor_chirality` drops that key. Returns ``(capped_mol, held)``.
    """
    labile = _labile_donors(mol, donors)
    if not labile:  # the common case (no carbanion/amine stereocentre) — skip the RWMol copy + sanitize entirely
        return mol, []
    _shift_phantoms(cons, len(labile))  # the caps append HERE, so the haptic block reserved after them moves up
    rw = Chem.RWMol(mol)
    held = []
    for d in labile:
        a = rw.GetAtomWithIdx(d)
        held.append((None, d, a.GetFormalCharge()))
        a.SetFormalCharge(0)  # neutralise: a 4th bond on an anion would be hypervalent
        a.SetNoImplicit(True)
        dm = rw.AddAtom(Chem.Atom(1))
        rw.GetAtomWithIdx(dm).SetIsotope(2)  # deuterium — distinct from any real H, lowest CIP priority
        rw.AddBond(d, dm, Chem.BondType.SINGLE)
        add_distance(cons.distances, metal, dm, _DUMMY_M_LO, _DUMMY_M_HI)  # pin D near M -> fast, correct ETKDG place
        held[-1] = (dm, d, held[-1][2])
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    for conf in out.GetConformers():  # place each D at the 4th vertex of this conformer's current hand
        for dm, d, _q in held:
            pd = np.array(conf.GetAtomPosition(d))
            nbrs = [n.GetIdx() for n in out.GetAtomWithIdx(d).GetNeighbors() if n.GetIdx() != dm][:3]
            units = [(v := np.array(conf.GetAtomPosition(i)) - pd) / (np.linalg.norm(v) or 1.0) for i in nbrs]
            fourth = -sum(units)  # opposite the three real substituents (~ the lone-pair / M direction)
            conf.SetAtomPosition(dm, Point3D(*(pd + fourth / (np.linalg.norm(fourth) or 1.0))))
    return out, held


def _release_donor_chirality(mol, held, cons):
    """Remove the hold dummy D's + their ``(metal, D)`` cons keys, restore donor charges, keep the tag.

    Dropping the cons key is critical: `cons` is the Ensemble's, reused by `minimize`/`_reembed`/`mc`, and a
    distance to a now-removed atom would index past the mol (an IndexError in the bounds matrix).
    """
    if not held:
        return mol
    dummies = {dm for dm, _d, _q in held}
    for key in [k for k in cons.distances if k[0] in dummies or k[1] in dummies]:
        del cons.distances[key]
    rw = Chem.RWMol(mol)
    for dm in sorted(dummies, reverse=True):  # high indices first so the remaining atoms don't shift
        rw.RemoveAtom(dm)
    for _dm, d, q in held:
        rw.GetAtomWithIdx(d).SetFormalCharge(q)
    out = rw.GetMol()
    donor_tags = {d: out.GetAtomWithIdx(d).GetChiralTag() for _dm, d, _q in held}  # sanitize drops the now-degree-3
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    for d, t in donor_tags.items():  # carbanion/amine tag -> keep it so the NEXT hold (relax, re-embed, mc) fires
        out.GetAtomWithIdx(d).SetChiralTag(t)
    return out


def surrogate_metal(mol):
    """Remove metal-donor bonds and swap the metal to a UFF surrogate. Returns (mol, metal, donors, real_Z, real_q)."""
    m = metal_index(mol)
    if m is None:
        raise ValueError("no transition metal found")
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
    em = Chem.RWMol(mol)
    for d in donors:
        em.RemoveBond(d, m)
        _clear_labile_donor_stereo(em.GetAtomWithIdx(d))  # a donor that's a stereocentre only WHILE metal-bound
        em.GetAtomWithIdx(d).SetNoImplicit(True)  # freeze donor H count so MC (openconf) adds none
    real_z = em.GetAtomWithIdx(m).GetAtomicNum()
    a = em.GetAtomWithIdx(m)
    real_q = a.GetFormalCharge()  # the oxidation state — zeroed for the surrogate, handed back by `restore_metal`
    a.SetAtomicNum(SURROGATE)
    a.SetNoImplicit(True)
    a.SetFormalCharge(0)
    a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # a bondless surrogate is never a stereocentre (a stray tag
    out = em.GetMol()  # from a metal RDKit mis-flagged as tetrahedral would crash ETKDG: 'nbrs.size() >= 3')
    donor_tags = {d: out.GetAtomWithIdx(d).GetChiralTag() for d in donors}  # re-applied below (sanitize drops the tag)
    # Lenient (no valence checks), the SAME tolerance the reader admitted this structure under and as
    # `surrogate_all_metals`; a strict sanitize would reject what perception waived (a quinoid ring, a BPh4- boron).
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    _io.repair_bond_stereo(out)  # the strip can orphan a C=N whose reference atom WAS the metal — see there
    for d, t in donor_tags.items():
        if t != Chem.ChiralType.CHI_UNSPECIFIED:
            out.GetAtomWithIdx(d).SetChiralTag(t)
    return out, m, donors, real_z, real_q


def restore_metal(mol, metal, real_z, real_q):
    """Swap `metal` back from the surrogate to its real element `real_z` and its oxidation state `real_q`.

    The charge is load-bearing: `Chem.GetFormalCharge` is what `_calc_charge` hands the calculator, so restoring
    only the element leaves an M(0) among anionic ligands and xtb gets the total charge wrong by the oxidation
    state. (The surrogate must stay neutral — a charged bond-less carbon is not a valid DG atom.)
    """
    a = mol.GetAtomWithIdx(metal)
    a.SetAtomicNum(real_z)
    a.SetFormalCharge(real_q)


def connect_metal(mol, donor_bonds):
    """Re-add the surrogate-stripped M-donor bonds as DATIVE (donor->metal), returning a connected Mol.

    `restore_metal` is calculator-minimal (xtb needs no graph), so it leaves the metal topologically DETACHED;
    anything that USES the mol (perception, re-embedding, an OIN drop-in) needs the M-L bonds, so this re-adds
    them once geometry/element/charge are settled.

    DATIVE rather than covalent: it counts toward the metal's valence, never the donor's, so it restores
    connectivity without touching any ligand's valence / H-count / charge. Coordinates untouched; idempotent.
    """
    rw = Chem.RWMol(mol)
    added = False
    for d, m in donor_bonds:
        if rw.GetBondBetweenAtoms(int(d), int(m)) is None:
            rw.AddBond(int(d), int(m), Chem.BondType.DATIVE)
            added = True
    if not added:
        return mol
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)  # recompute implicit valence (never raises, never moves a formal charge)
    Chem.FastFindRings(out)  # the DATIVE bonds close chelate rings THROUGH the metal — re-perceive RingInfo so
    # downstream ring-aware queries are correct (and don't raise "not initialized" when mc's disconnect cleared it):
    # the dedup descriptor SMARTS (`rotatable_quads`) and the geometry gate's AtomRings()
    return out


def disconnect_metal(mol):
    """Strip the DATIVE M-donor bonds `connect_metal` added, returning the bare working mol (inverse of it).

    A stage that MOVES atoms again (`mc`, a re-`minimize`) must see the bond-less mol the surrogate needs (UFF
    cannot type a bonded metal), so the finalize's bonds come off before atoms move and the follow-up minimize
    re-adds them. Any dative-to-metal bond on a pipeline mol is a `connect_metal` artefact. No-op when none.
    """
    dative = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DATIVE
        and (
            mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() in TRANSITION_METALS
            or mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() in TRANSITION_METALS
        )
    ]
    if not dative:
        return mol
    rw = Chem.RWMol(mol)
    for a, b in dative:
        rw.RemoveBond(a, b)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def metal_indices(mol):
    """Return the indices of all transition-metal atoms."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]


def _haptic_sites(mol, donors):
    """Group `donors` into coordination SITES — a directly-bonded pi-face (Cp/arene/allyl) collapses to one.

    A lone sigma donor is its own 1-tuple; a chelate's donors relate only THROUGH the backbone, so they stay
    separate — only a genuine haptic face is grouped (ferrocene's ten ring donors reduce to two sites). Run on the
    surrogate (M-donor bonds stripped), so the walk follows only ligand bonds.
    """
    dset = set(donors)
    seen, sites = set(), []
    for d in sorted(dset):
        if d in seen:
            continue
        stack, group = [d], []
        while stack:  # flood-fill over donor-donor ligand bonds
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            group.append(a)
            stack += [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in dset]
        sites.append(tuple(sorted(group)))
    return sorted(sites)


def _regular_face(mol, site):
    """Return True if `site`'s own bond graph is REGULAR — every face atom has the same face-neighbour count.

    The cone test: one shared centroid radius is only true of a vertex-transitive (regular-polygon) face. Degree 1
    is a single EDGE (an eta2 alkene), degree 2 a CYCLE (Cp / arene). An open allyl/diene is degree {1,2,...,1},
    irregular — its centroid is nearer the inner atoms, so it takes METAL mode and relaxes free. One rule for every
    hapticity, no size threshold: a bond is the smallest rigid face, so eta2 needs no side-on special case.
    """
    face = set(site)
    deg = {sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in face) for a in face}
    return len(deg) == 1


def _collapse_haptic(mol, donors):
    """Collapse each haptic face (any mutually-bonded donor group) to ONE centroid vertex; sigma donors pass through.

    APPENDS a bond-less carbon centroid per face (existing indices unchanged), seated at the ring centroid if the
    mol has a conformer. Returns ``(mol, vertices, haptic)``; `haptic` maps each dummy -> its ring atoms. The dummy
    is embed scaffolding that lives in no stored Mol (`enumerate_isomers` strips it before storing the real
    `Isomer`; only `bounds.embed` / `restrained_uff` re-materialise it). No haptic face -> returned untouched.
    """
    sites = _haptic_sites(mol, donors)
    faces = [s for s in sites if len(s) > 1]  # a face is ANY mutually-bonded donor group — eta2 included
    if not faces:
        return mol, donors, {}
    em = Chem.RWMol(mol)
    conf = em.GetConformer() if em.GetNumConformers() else None
    vertices = [d for s in sites if len(s) == 1 for d in s]  # a lone sigma donor is its own vertex
    haptic = {}
    for site in faces:
        idx = em.AddAtom(Chem.Atom(SURROGATE))  # a bond-less carbon: excluded volume in the DG, Xe-ghosted in the FF
        em.GetAtomWithIdx(idx).SetNoImplicit(True)  # a point, not a valence — no implicit H (like the metal)
        if conf is not None:
            conf.SetAtomPosition(idx, Point3D(*np.mean([list(conf.GetAtomPosition(a)) for a in site], axis=0)))
        vertices.append(idx)
        haptic[idx] = site
    out = em.GetMol()
    out.UpdatePropertyCache(strict=False)  # bond-less centroids' valence (GetMoleculeBoundsMatrix needs it)
    Chem.FastFindRings(out)  # + RingInfo (the RWMol edit cleared it), which GetMoleculeBoundsMatrix requires
    return out, vertices, haptic


def materialise_phantoms(mol, haptic):
    """Return a copy of `mol` with each haptic centroid dummy appended (bond-less carbon) at its reserved index.

    A TRANSIENT of the embed (it lives in NO persistent Mol). `bounds.embed` / `restrained_uff` materialise it
    (from `Constraints.haptic`) so a Cp/arene face embeds as one rigid vertex, then discard it. Reserved indices
    are consecutive from the real atom count, so append order reproduces them; each dummy is seated at its ring
    centroid. No-op when there is no haptic face.
    """
    if not haptic:
        return mol
    base_n = mol.GetNumAtoms()  # the dummies are appended, so their reserved keys must be the next consecutive indices
    if sorted(haptic) != list(range(base_n, base_n + len(haptic))):  # a D-cap or other transient was appended first
        raise ValueError(
            f"haptic centroid indices {sorted(haptic)} are not the {len(haptic)} indices after {base_n} atoms — "
            f"another transient atom (a donor-chirality D-cap?) was appended first; a haptic face and a "
            f"carbanion/amine-donor chirality hold cannot compose"
        )
    rw = Chem.RWMol(mol)
    for _dummy in sorted(haptic):
        a = rw.GetAtomWithIdx(rw.AddAtom(Chem.Atom(SURROGATE)))
        a.SetNoImplicit(True)  # a point, not a valence
        a.SetHybridization(Chem.HybridizationType.SP3)  # match the sanitised metal surrogate (else UFF/bounds-matrix
        #   emit 'unrecognized hybridization' on the bond-less carbon — UpdatePropertyCache leaves it UNSPECIFIED)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)  # GetMoleculeBoundsMatrix / UFF need the new bond-less atoms' valence
    Chem.FastFindRings(out)  # + RingInfo (an RWMol edit clears it), which GetMoleculeBoundsMatrix requires
    for conf in out.GetConformers():
        for dummy, ring in haptic.items():
            conf.SetAtomPosition(dummy, Point3D(*np.mean([list(conf.GetAtomPosition(a)) for a in ring], axis=0)))
    return out


def strip_phantoms(mol, phantoms):
    """Return a copy of `mol` with the haptic centroid dummies removed; real indices unchanged.

    Embed scaffolding (`Constraints.phantoms`), never part of a stored Mol; `enumerate_isomers` strips it before
    storing the real `Isomer` so nothing downstream sees it. Dummies are the highest indices, so removing them from
    the end leaves every real index untouched. No-op when there are no phantoms.
    """
    if not phantoms:
        return mol
    rw = Chem.RWMol(mol)
    for p in sorted(phantoms, reverse=True):  # remove from the end so the real (lower) indices stay put
        rw.RemoveAtom(int(p))
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)  # RemoveAtom clears RingInfo; downstream (embed's bounds matrix, perception) needs it
    return out


def _site_radius(mol, site, cid=-1):
    """Circumradius of a haptic ring: the mean distance of its atoms from their own centroid.

    From the conformer if the mol has one (a retained crystal geometry), else from RDKit's own bounds midpoints
    (``R = sqrt(sum d_ij^2)/n``, exact for a regular n-gon — correct for Cp, arene, boratabenzene's longer B-C
    edge alike, with no hand-fitted radii). A 2-atom (eta2) site reduces to half the edge.
    """
    if mol.GetNumConformers():
        pos = np.array([list(mol.GetConformer(cid).GetAtomPosition(a)) for a in site])
        return float(np.linalg.norm(pos - pos.mean(0), axis=1).mean())
    from rdkit.Chem import rdDistGeom

    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    tot = sum(
        (0.5 * (bm[max(a, b)][min(a, b)] + bm[min(a, b)][max(a, b)])) ** 2 for a, b in itertools.combinations(site, 2)
    )
    return float(np.sqrt(tot)) / len(site)


def surrogate_all_metals(mol):
    """Surrogate **every** transition metal (bonds removed, carbon) for a multi-metal complex.

    UFF must type the whole complex (a bimetallic TS); `surrogate_metal` only does the first metal. Returns
    ``(mol, metals, donors)`` where ``metals`` is ``[(idx, real_z, real_q), ...]`` (element + oxidation state for
    `restore_metal`) and ``donors`` is every metal-donor atom (so the caller can freeze the cores). Lenient
    sanitise (a stripped η⁵-Cp is a radical fragment, fine here).
    """
    idxs = metal_indices(mol)
    if not idxs:
        raise ValueError("no transition metal found")
    em = Chem.RWMol(mol)
    metals, donors = [], []
    for m in idxs:
        metals.append((m, em.GetAtomWithIdx(m).GetAtomicNum(), em.GetAtomWithIdx(m).GetFormalCharge()))
        for d in [n.GetIdx() for n in em.GetAtomWithIdx(m).GetNeighbors()]:
            if em.GetBondBetweenAtoms(d, m) is not None:
                em.RemoveBond(d, m)
            _clear_labile_donor_stereo(em.GetAtomWithIdx(d))  # stale tag on a now-<3-nbr donor crashes ETKDG
            em.GetAtomWithIdx(d).SetNoImplicit(True)
            donors.append(d)
        a = em.GetAtomWithIdx(m)
        a.SetAtomicNum(SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # bondless surrogate — a stray metal tag crashes ETKDG
    out = em.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out, metals, sorted(set(donors))


_APICAL_MIN = 3  # a face of >= this many atoms is APICAL/facial (site occupancy, not geometry): an eta2 alkene
#   fills ONE site (Zeise's flat square plane, C=C midpoint an in-plane vertex, 4.2° from the crystal), an eta5-Cp
#   / eta6-arene caps a whole FACE (a piano stool). Steers only the default-polyhedron GUESS below; the embed
#   mechanics (`_regular_face`) are hapticity-free.


def classify_geometry(mol, metal, sites, cid=-1):
    """Name the coordination polytope from the ACTUAL geometry: the closest template by angle spectrum.

    ``sites`` are coordination SITES, not atoms (a haptic face is one vertex via its centroid). Returns ``None``
    when no template has that vertex count.

    The discriminator is the sorted spectrum of vertex-metal-vertex angles, invariant to rotation and donor
    ordering (no Kabsch, no permutation search), which is what separates the near-degenerate pairs: a trigonal
    bipyramid and a square pyramid share donor count and connectivity and differ ONLY in these angles. Counting
    donors alone mis-named an ideal square pyramid `trigonal_bipyramidal` and a tetrahedron `square_planar`.
    """
    pos = mol.GetConformer(cid).GetPositions()
    vectors = []
    for site in sites:
        atoms = [site] if isinstance(site, (int, np.integer)) else list(site)
        vectors.append(np.mean([pos[a] for a in atoms], axis=0) - pos[metal])
    obs = _angle_spectrum(vectors)
    if obs is None:
        return None
    best, best_err = None, None
    for name, p in POLYHEDRA.items():
        dirs = p.vertex_dirs
        if len(dirs) != len(vectors):
            continue
        ideal = _angle_spectrum([np.array(d, float) for d in dirs])
        err = float(np.sqrt(np.mean((obs - ideal) ** 2)))
        if best_err is None or err < best_err:
            best, best_err = name, err
    return best


def _angle_spectrum(vectors):
    """Sorted vertex-metal-vertex angles (deg) — a rotation- and order-invariant shape signature."""
    unit = []
    for v in vectors:
        n = float(np.linalg.norm(v))
        if n < _EPS_VEC:
            return None
        unit.append(np.asarray(v, float) / n)
    return np.array(
        sorted(math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b))))) for a, b in itertools.combinations(unit, 2))
    )


_EPS_VEC = 1e-9  # a donor sitting on the metal has no direction to classify by


def geometry_for(n_donors, has_apical=False):
    """Default coordination polyhedron name for `n_donors`, or None.

    An APICAL face (eta>=3) is an axial cone, so a CN4 carrying one is a PIANO STOOL / bent metallocene (a
    distorted tetrahedron), never the flat `square_planar` the vertex count would pick (which would seat a ligand
    trans THROUGH the ring). Only CN4 flips. An eta2 face is NOT apical — an ordinary single-site vertex that must
    keep the vertex-count default, or Zeise's square-planar salt embeds as a tetrahedron (30.5° from crystal vs 4.2).
    """
    gs = geometries_for_cn(n_donors)  # best-default first
    g = gs[0].name if gs else None
    if has_apical and g == "square_planar":
        return "tetrahedral"
    return g


_COPLANAR_TOL = 0.25  # Å: RMS out-of-plane of {metal + donors} above which a *planar* geometry isn't planar


def coplanar(pos, metal, donors, tol=_COPLANAR_TOL, haptic=None):
    """Return True if the metal + coordination VERTICES lie in one plane — the test of a declared *planar* geometry.

    A square-planar / T-shape / trigonal-planar complex **is** coplanar by definition (a bite squeezing the
    in-plane angles is still planar), but an arrangement that can only satisfy its ligands by twisting out of
    plane (an impossible trans-chelate) is not — separating a distorted-but-planar square (RMS ~0.05 Å) from a
    puckered phantom (RMS ~0.35 Å). Fewer than 4 points are always coplanar.

    An eta>=3 haptic face is ONE vertex (its centroid), so `haptic` collapses each face to a centroid point first.
    """
    ring_atoms = {a for ring in (haptic or {}).values() for a in ring}
    verts = [pos[d] for d in donors if d not in ring_atoms]  # each sigma/eta2 donor is its own vertex
    verts += [np.mean([pos[a] for a in ring], axis=0) for ring in (haptic or {}).values()]  # each face -> centroid
    pts = np.array([pos[metal], *verts])
    if len(pts) < 4:  # noqa: PLR2004 — a plane needs >=3 points; <4 total is trivially coplanar
        return True
    dev = (pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0))[2][2]  # signed distance from best-fit plane
    return float(np.sqrt(np.mean(dev**2))) <= tol


VACANT = -1  # a coordination vertex left empty (donors < sites)
_TRANS_ANGLE = 150  # degrees: a same-element donor pair beyond this is trans (else cis)
_TRIAD = 3  # a mer/fac triad is exactly three donors
_PAIR = 2  # a same-element pair that can be cis/trans
_COLINEAR_TOL = 0.5  # topological-distance tolerance for the "central donor" collinearity test


def n_sites(geometry):
    """Return the number of coordination vertices the geometry has."""
    return len(POLYHEDRA[geometry].vertex_dirs)  # unknown geometry -> KeyError (deliberate)


def hold_shape(mol, atoms, cons, pad=0.1, cid=-1):
    """Fix the SHAPE of `atoms` at their input-geometry mutual distances (all pairwise, ``±pad``).

    A relative, frame-independent constraint, so a surrogate metal's sphere (a spectator ferrocene) is held
    INTACT when re-embedded without pinning it to an absolute frame. The atom set is recorded in ``cons.shapes``:
    these windows are one rigid body, so `ff_terms` must not single out the M-donor ones for a `pull` (it would
    tear the rest — see there).
    """
    add_pairwise_shape(cons, atoms, mol.GetConformer(cid).GetPositions(), pad)
    cons.shapes.append(set(atoms))


_SPAN_ANGLE = 135  # a same-ligand donor pair at a vertex separation this wide is a *trans*-type span
_SPAN_TOL = 0.1  # Å slack on the backbone-reach test — tight enough to drop a 5-membered chelate (e.g. an
# amidate, backbone ~3.7 Å) forced *trans* (need ~3.85 Å), while a genuine long bridge (backbone >> need,
# e.g. a flexible bis-NHC at ~6.5 Å) still passes. Bigger slack (0.2) let the amidate-trans phantom through.


_DISCONNECTED = 1e6  # RDKit's topological distance for atoms in different fragments (it returns ~1e8)


def label(mol, metal, donors, cid, geometry=None):
    """Coordination-isomer label of one conformer: 'trans' if a same-element donor pair is trans, else 'cis'.

    Geometries with no geometric isomerism (linear / trigonal-planar / tetrahedral) get no label ('').
    """
    p = POLYHEDRA.get(geometry)
    if p is not None and not p.geometric_isomerism:
        return ""
    pos = mol.GetConformer(cid).GetPositions()
    for a in range(len(donors)):
        for b in range(a + 1, len(donors)):
            same = mol.GetAtomWithIdx(donors[a]).GetSymbol() == mol.GetAtomWithIdx(donors[b]).GetSymbol()
            if same and _vertex_angle(pos[donors[a]] - pos[metal], pos[donors[b]] - pos[metal]) > _TRANS_ANGLE:
                return "trans"
    return "cis"


def _donor_classes(mol, donors):
    """Map each donor atom -> its symmetry-equivalence class (equal for interchangeable donors).

    RDKit canonical rank with ``breakTies=False`` gives the graph automorphism classes, so the two N of one
    en, or three equivalent chloride, share a class — exactly what the handedness parity must key on. The
    metal-donor bonds are stripped on the surrogate, so identical ligands sit in identical fragments and
    rank equal. Falls back to the atomic symbol if canonical ranking is unavailable.
    """
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        return {d: ranks[d] for d in donors}
    except Exception:  # pragma: no cover - canonical ranking is robust, but never let chirality crash embed
        return {d: mol.GetAtomWithIdx(d).GetSymbol() for d in donors}


def _chelate_edges(mol, vertices, haptic=None):
    """Return ``{frozenset({vertex_i, vertex_j})}`` for vertex pairs whose donors chelate one ligand.

    Two occupied vertices are a *bite* when their donors sit in the same fragment (same ligand). A
    tris/bis-chelate's Λ/Δ handedness lives in this bite graph, not the per-vertex donor class. A haptic
    face's centroid is resolved through its ring (`_vertex_atom`) so a tethered face bites its co-donor.
    """
    frag = _frag_map(mol)
    occ = [v for v in range(len(vertices)) if vertices[v] != VACANT]
    return frozenset(
        frozenset((a, b))
        for i, a in enumerate(occ)
        for b in occ[i + 1 :]
        if frag[_vertex_atom(haptic, vertices[a])] == frag[_vertex_atom(haptic, vertices[b])]
    )


def chirality_of(mol, donors, geometry, vertices, haptic=None):
    """Return the metal centre's Λ/Δ chirality tag (``'Δ'`` / ``'Λ'`` / ``''`` achiral) for one arrangement.

    `vertices[v]` is the donor seated at polyhedron vertex `v` (or ``VACANT``). Name-agnostic and
    order-invariant: the parity of the frame canonicalising the (donor-class + chelate-bite) labelling over
    the geometry's point group (see `polyhedron.handedness`). ``''`` when the geometry has no template or a
    vertex is vacant. `haptic` resolves a centroid vertex to its ring for the chelate-bite graph.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return ""
    return _poly.handedness(dirs, list(vertices), _donor_classes(mol, donors), _chelate_edges(mol, vertices, haptic))


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
    real_q: int  # the metal's oxidation state — the surrogate is neutral; `restore_metal` hands it back
    label: str
    geometry: str
    vertices: list = field(default_factory=list)
    chirality: str = ""  # metal-centre handedness 'Δ'/'Λ'/'' — the name-agnostic stereo identity
    extra: list = field(default_factory=list)  # other surrogated metals (idx, real_z, real_q) — spectators in a
    # multi-metal complex, restored (element and charge) alongside `metal`
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')
    stereo_label: str = ""  # LIGAND stereoisomer tag (e.g. '16R') when rx.metal enumerated an undefined ligand
    # stereocentre — the coordination x ligand-stereo load-in; distinct from the metal-centre `chirality`
    haptic: dict = field(default_factory=dict)  # {centroid-vertex index -> its face's atoms}. `mol`/`donors` are
    # REAL (the centroid dummy is transient embed scaffolding, never stored), but `vertices` records the coordination
    # identity honestly — an η² alkene / Cp / arene each present ONE vertex — so `arrangement` renders it from this
    # map. A vertex in `vertices` is therefore NOT always an atom of `mol`: resolve it through here first.
    donor_bonds: list = field(default_factory=list)  # the stripped M-donor bonds as (donor, metal) pairs, EVERY
    # metal's (centre + spectators) — `connect_metal` re-adds them as DATIVE so the pipeline output is connected.

    def restore(self):
        """Restore this isomer's metal(s) from the surrogate back to their real elements and oxidation states."""
        restore_metal(self.mol, self.metal, self.real_z, self.real_q)
        for mi, rz, rq in self.extra:
            restore_metal(self.mol, mi, rz, rq)

    def summary(self):
        """Return this isomer's geometric identity string: ``geometry | per-vertex arrangement | chirality``.

        The convenient one-liner for a single isomer (the name-agnostic keys you'd ``select`` on), e.g.
        ``'square_planar | C25 C44 O27 N37 | achiral'``. Mirrors what `IsomerSet.summary` prints per row.
        """
        stereo = f" | stereo {self.stereo_label}" if self.stereo_label else ""
        return f"{self.geometry} | {arrangement(self)} | {self.chirality or 'achiral'}{stereo}"


def arrangement(iso):
    """Format a readable per-vertex ligand arrangement, e.g. ``'N3 Cl5 Cl6 ·'`` (``·`` = a vacant site).

    The *unambiguous* identity of an isomer, since the cis/trans label only describes a same-element pair
    and says nothing about where a vacancy sits. Order follows the polyhedron's `vertex_dirs`.
    """

    def sym(d):
        if d == VACANT:
            return "·"
        if d in iso.haptic:  # a haptic face's centroid vertex — the ring, not a real atom index (mol is stripped)
            ring = iso.haptic[d]
            return f"η{len(ring)}({min(ring)})"  # e.g. 'η5(1)': hapticity + the lowest-index ring atom, unambiguous
        return f"{iso.mol.GetAtomWithIdx(d).GetSymbol()}{d}"

    return " ".join(sym(d) for d in iso.vertices)


arrange = arrangement  # alias so IsomerSet.filter(arrangement=…) can still call the formatter (param shadows it)


_LONE_PAIR_Z = {7, 8, 15, 16, 33, 34, 51, 52}  # N O P S As Se Sb Te — p-block groups 15/16 (pnictogens + chalcogens)


def lone_pair_donors(mol, metal, exclude=()):
    """Return substrate lone-pair donors (p-block group-15/16 heteroatoms) that could coordinate a vacant site.

    Excludes the metal and every atom in a **ligand fragment** (one containing the metal or an `exclude` donor).
    `surrogate_metal()` detaches the ligands into their own fragments, so without this a ligand-backbone
    heteroatom (an ether O on a phosphine) would look like a free substrate donor.
    """
    frag = _frag_map(mol)
    ligand_frags = {frag[metal]} | {frag[d] for d in exclude}
    return [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _LONE_PAIR_Z and frag[a.GetIdx()] not in ligand_frags
    ]
