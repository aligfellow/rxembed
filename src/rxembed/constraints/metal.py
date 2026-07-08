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
from rdkit.Geometry import Point3D

from rxembed.log import logger  # relative, like builders.py — package convention

from . import polyhedron as _poly
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

_SOFT_DATIVE_DONORS = frozenset({15, 16, 33, 34, 51, 52})  # P, S, As, Se, Sb, Te — soft lone-pair donors whose
# dative M-bond runs SHORTER than their (large) single-bond covalent radius. A HALIDE (Cl/Br/I) donor is X-type
# covalent, NOT dative — its M-X sits AT the covalent sum (Ni-Br 2.44) and must NOT be capped (capping would give
# ~2.09, too short, and manufacture the very tears the retry then wastes rounds on). So the cap is soft-donor-only.
_DONOR_RCOV_CAP = 0.85  # Å: the raw covalent sum gives Ni-P 2.40 (real ~2.15); capping the donor radius to ~0.85
# (just above carbon's 0.76) fixes P/S/… while C/N/O — already below it — are untouched at their covalent sum.
_DONOR_DONATION_MIN, _DONOR_DONATION_MAX = 100.0, 150.0  # deg: the M-donor-neighbour window for a conjugated N/O
# donor (donation along the sp2 lone-pair axis, ~120°) — wide enough for real variation, tight enough to forbid
# the rigid-unit fold-in (M-O-C collapsing to ~83° swings a carboxylate/pyridine plane into the coordination sphere).
_DONOR_ORIENT = {  # M-donor-substituent angle window by DONOR hybridisation: the substituents splay away from the
    Chem.HybridizationType.SP3: (95.0, 130.0),  # metal (lone pair / coordinate bond toward it) — ~109.5 tetrahedral,
    Chem.HybridizationType.SP2: (_DONOR_DONATION_MIN, _DONOR_DONATION_MAX),  # ~120 trigonal (the conjugated N/O hold)
}

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


def _frag_map(mol):
    """Map each atom index -> its fragment id (same ligand = same fragment)."""
    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}


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


def _hold_donor_chirality(mol, metal, donors, cons):
    """Cap each labile (sp3 C/N) metal-bound donor carrying a chiral tag with a dummy D, so the hand is enforced.

    A degree-3 carbanion/amine donor (no M-C bond in the surrogate) is not a stereocentre RDKit/UFF perceives,
    so its two enumerated hands relax to the *same* geometry. Neutralising its charge (a carbanion cannot be
    pentavalent) and adding a 4th bond to a **deuterium** makes it a proper, enforced tetrahedral centre — the
    same appended-D basis the enumeration labelled it in, so the embedded hand matches the tag. When the mol
    already has conformers (the relax / mc path) each D is placed at the 4th tetrahedral vertex of THIS
    conformer's current hand (preserving it); a conformer-free mol (the initial embed) gets the hand from ETKDG +
    the chiral tag. A **(metal, D) distance is added to `cons`** so D is pinned on the coordinate-bond side —
    WITHOUT it ETKDG thrashes on the free 4th atom's chiral volume (~200x slower); `_release_donor_chirality`
    drops that key. Returns ``(capped_mol, held)`` where `held` is ``[(dummy_idx, donor_idx, original_charge)]``.
    """
    labile = _labile_donors(mol, donors)
    if not labile:  # the common case (no carbanion/amine stereocentre) — skip the RWMol copy + sanitize entirely
        return mol, []
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


def prepare(mol):
    """Remove metal-donor bonds and swap the metal to a UFF surrogate. Returns (mol, metal, donors, real_Z)."""
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
    a.SetAtomicNum(SURROGATE)
    a.SetNoImplicit(True)
    a.SetFormalCharge(0)
    a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # a bondless surrogate is never a stereocentre (a stray tag
    out = em.GetMol()  # from a metal RDKit mis-flagged as tetrahedral would crash ETKDG: 'nbrs.size() >= 3')
    donor_tags = {d: out.GetAtomWithIdx(d).GetChiralTag() for d in donors}  # sanitize drops an enumerated
    Chem.SanitizeMol(out)  # degree-3 carbanion/amine donor's tag -> re-apply it so `_hold_donor_chirality` sees it
    for d, t in donor_tags.items():
        if t != Chem.ChiralType.CHI_UNSPECIFIED:
            out.GetAtomWithIdx(d).SetChiralTag(t)
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


def geometry_for(n_donors):
    """Default coordination polyhedron name for `n_donors`, or None."""
    return GEOM.get(n_donors)


COPLANAR_GEOMETRIES = frozenset({"linear", "trigonal_planar", "t_shape", "square_planar"})
_COPLANAR_TOL = 0.25  # Å: RMS out-of-plane of {metal + donors} above which a *planar* geometry isn't planar


def coplanar(pos, metal, donors, tol=_COPLANAR_TOL):
    """Return True if the metal + donors lie in one plane — the defining test of a declared *planar* geometry.

    A square-planar (or T-shape / trigonal-planar) complex **is** coplanar by definition; a bite that squeezes
    the in-plane angles (a 47° chelate forcing others to 125°) is still planar and fine, but an arrangement
    that can only satisfy its ligands by *twisting out of plane* (an impossible trans-chelate the stiff relax
    forces through) is not this geometry. This separates a legitimate distorted-but-planar square (RMS ~0.05 Å)
    from a puckered phantom (RMS ~0.35 Å). Fewer than 4 points are always coplanar.
    """
    pts = np.array([pos[metal]] + [pos[d] for d in donors])
    if len(pts) < 4:  # noqa: PLR2004 — a plane needs >=3 points; <4 total is trivially coplanar
        return True
    dev = (pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0))[2][2]  # signed distance from best-fit plane
    return float(np.sqrt(np.mean(dev**2))) <= tol


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


def _orient_donor(mol, metal, d, donor_set, cons, orient_protons):
    """Hold a donor's substituent directions relative to the metal with M-D-X angles.

    Two parts. (1) ALWAYS: a conjugated N/O donor's rigid **sp2 heavy** plane is held ~120° so it can't fold acute
    into the metal — the long-standing donation hold. (2) Only when embedding **from scratch** (`orient_protons`,
    i.e. no input geometry — a geometry / frozen-core embed already places the H's, and piling extra angles onto its
    already-tight bounds over-constrains it): splay **any donor's protons** to the hybridisation angle (~109.5° sp3)
    so a methyl / ammine / amine's X-H bonds point away and the lone pair points at M, and SKIP a **haptic** donor
    (directly bonded to a co-donor = a side-on eta2 / two-sigma unit, not a lone-pair splay). Each angle is one 1,3
    bounds entry that biases the embed, is held by the relax, and registers the atoms for mc's pose-hold.
    """
    a = mol.GetAtomWithIdx(d)
    conjugated = a.GetAtomicNum() in (7, 8)
    proton_window = _DONOR_ORIENT.get(a.GetHybridization()) if orient_protons else None
    haptic = any(nb.GetIdx() in donor_set for nb in a.GetNeighbors())
    for nb in a.GetNeighbors():
        if conjugated and nb.GetAtomicNum() > 1 and nb.GetHybridization() == Chem.HybridizationType.SP2:
            cons.angles.setdefault((metal, d, nb.GetIdx()), (_DONOR_DONATION_MIN, _DONOR_DONATION_MAX))  # (1)
        elif proton_window and not haptic and nb.GetAtomicNum() == 1:  # (2) — from-scratch, non-haptic donor protons
            cons.angles.setdefault((metal, d, nb.GetIdx()), proton_window)


def coordination(mol, metal, donors, geometry, order, real_z, frozen=()):
    """Build one isomer's constraints: metal-donor distances + donor-metal-donor angles.

    `donors` may be padded with ``VACANT`` for empty vertices (a coordination pocket) -- those get no
    constraints.

    `frozen` is the set of donor atoms held rigid by a ``fix=`` reacting core: an angle between **two
    frozen donors** is *omitted* — the freeze already pins that sub-triangle exactly (Mn-d1, Mn-d2, d1-d2),
    so an ideal-polyhedron angle there only over-determines it and makes triangle-smoothing widen (then
    lock in) a distorted core. A free-vs-frozen angle is kept — it's what seats a free donor at its vertex.

    An **intra-chelate** L-M-L angle (two donors of one ligand fragment) is **not constrained at all**: a
    chelate's bite is set by its rigid backbone, so the donor-donor distance already in the bounds matrix
    plus the two M-donor distances *fixes* the bite — imposing the ideal polyhedron angle on top only
    contradicts the backbone and makes the geometry infeasible (a 3-membered / side-on chelate bites ~45°,
    nowhere near 90°; even ±25° tore the ligand bond). Only **inter-ligand** angles are held (±8°), which is
    what seats the separate ligands at their relative polyhedron positions. Impossible chelate placements (a
    short backbone forced to span *trans*) are dropped upstream in `isomers` and downstream by `bonding_ok`.
    """
    r_m = _PT.GetRcovalent(real_z)
    frag = _frag_map(mol)  # same ligand = same fragment
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    c = Constraints()
    od = [donors[k] for k in order]  # od[vertex] = donor atom there, or VACANT
    real_od = {x for x in od if x != VACANT}  # the co-donors, for the haptic (side-on) check in _orient_donor
    for d in od:
        if d == VACANT:
            continue
        if pos is not None:  # use the REALISED metal-donor bond length from the input
            d_md = float(np.linalg.norm(pos[metal] - pos[d]))  # geometry (a covalent-radius guess is wrong for
            add_distance(c.distances, metal, d, d_md - 0.1, d_md + 0.1)  # a hydride/carbonyl and breaks the bounds)
        else:  # the covalent-sum bond length — the natural M-donor distance. A tighter guess (the old 0.9x)
            z_d = mol.GetAtomWithIdx(d).GetAtomicNum()  # pulls the whole sphere in, so a donor's aryl intrudes;
            r_d = _PT.GetRcovalent(z_d)  # a large SOFT donor (P/S/…) is capped — its dative bond runs shorter than
            if z_d in _SOFT_DATIVE_DONORS:  # its covalent radius implies; a halide (X-type) keeps the covalent sum
                r_d = min(r_d, _DONOR_RCOV_CAP)
            add_distance(c.distances, metal, d, r_m + r_d - 0.05, r_m + r_d + 0.05)
        # orient the donor's substituents away from the metal (its protons splay, lone pair / coordinate bond
        # points at M): a conjugated N/O donor's rigid plane is held near 120° (else it folds acute, swinging a
        # carboxylate/amide/pyridine plane into the sphere), and ANY donor's PROTONS are held so a methyl/ammine/
        # amine's X-H bonds don't point every which way. Skipped for a side-on η² pair. See `_orient_donor`.
        _orient_donor(mol, metal, d, real_od, c, orient_protons=pos is None)
    for i, j, a in ANGLES[geometry]:
        if od[i] == VACANT or od[j] == VACANT:  # an angle to an empty vertex is unconstrained
            continue
        if od[i] in frozen and od[j] in frozen:  # both held by freeze -> don't over-determine the core
            continue
        intra = frag[od[i]] == frag[od[j]]  # two donors of one chelating ligand
        if intra and a < _SPAN_ANGLE:  # a *cis* chelate bite -> the ligand backbone folds it (tight bites embed);
            continue  # imposing the ideal 90° tore side-on / 3-membered ligands
        # inter-ligand pairs (±8°) and any *trans*-assigned chelate (kept wide, ±25°) are held: forcing a
        # trans span makes an unreachable chelate (an en placed trans) tear -> dropped by `bonding_ok`, the
        # embed-time safety net for anything the `isomers` span filter doesn't pre-drop.
        pad = 25.0 if intra else 8.0
        c.angles[(od[i], metal, od[j])] = (max(0.0, a - pad), min(180.0, a + pad))
    # A **side-on η² pair** — two donors π-bonded to each other (a double/triple bond), both coordinating —
    # must be held at its natural bond length: the two M-donor pulls otherwise stretch the C≡C/C=C apart
    # (1.2 → 1.7 A) and blow the whole sphere out. Holding just this one bond keeps the side-on unit compact
    # (no CN reduction, no dummy). A single bond between two coordinating atoms is a two-sigma chelate, not side-on.
    eta_pairs = [
        (od[i], od[j])
        for i in range(len(od))
        for j in range(i + 1, len(od))
        if od[i] != VACANT and od[j] != VACANT and _pi_bonded(mol, od[i], od[j])
    ]
    if eta_pairs:
        from rdkit.Chem import rdDistGeom

        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)  # the accurate π-bond length for this bond order
        for a, b in eta_pairs:
            lo, hi = (a, b) if a < b else (b, a)
            bl = 0.5 * (bm[lo][hi] + bm[hi][lo])
            add_distance(c.distances, a, b, bl - 0.03, bl + 0.03)
    return c


def _pi_bonded(mol, a, b):
    """Return True if `a`,`b` are directly bonded by a **multiple** bond (a side-on pi unit, not two sigma)."""
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() >= 2  # noqa: PLR2004 — double/triple = pi


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
        base,
        cons,
        m,
        donors,
        real_z,
        label(base, m, donors, base.GetConformer().GetId(), geom),
        geom,
        list(donors),
        chirality=chirality_of(base, donors, geom, list(donors)),
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


def _chelate_edges(mol, vertices):
    """Return ``{frozenset({vertex_i, vertex_j})}`` for vertex pairs whose donors chelate one ligand.

    Two occupied vertices are a *bite* when their donors sit in the same fragment (same ligand). A
    tris/bis-chelate's Λ/Δ handedness lives in this bite graph, not the per-vertex donor class.
    """
    frag = _frag_map(mol)
    occ = [v for v in range(len(vertices)) if vertices[v] != VACANT]
    return frozenset(
        frozenset((a, b)) for i, a in enumerate(occ) for b in occ[i + 1 :] if frag[vertices[a]] == frag[vertices[b]]
    )


def chirality_of(mol, donors, geometry, vertices):
    """Return the metal centre's Λ/Δ chirality tag (``'Δ'`` / ``'Λ'`` / ``''`` achiral) for one arrangement.

    `vertices[v]` is the donor seated at polyhedron vertex `v` (or ``VACANT``). Name-agnostic and
    order-invariant: the parity of the frame canonicalising the (donor-class + chelate-bite) labelling over
    the geometry's point group (see `polyhedron.handedness`). ``''`` when the geometry has no template or a
    vertex is vacant.
    """
    dirs = VERTEX_DIRS.get(geometry)
    if dirs is None:
        return ""
    return _poly.handedness(dirs, list(vertices), _donor_classes(mol, donors), _chelate_edges(mol, vertices))


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
    chirality: str = ""  # metal-centre handedness 'Δ'/'Λ'/'' — the name-agnostic stereo identity
    extra: list = field(default_factory=list)  # other surrogated metals (idx, real_z) — spectators in a
    # multi-metal complex, restored alongside `metal`
    stereo_ref: object = None  # input-geometry chirality fingerprint (for stereo='preserve')
    stereo_label: str = ""  # LIGAND stereoisomer tag (e.g. '16R') when rx.metal enumerated an undefined ligand
    # stereocentre — the coordination x ligand-stereo load-in; distinct from the metal-centre `chirality`

    def restore(self):
        """Restore this isomer's metal(s) from the surrogate back to their real elements."""
        restore(self.mol, self.metal, self.real_z)
        for mi, rz in self.extra:
            restore(self.mol, mi, rz)

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
    and says nothing about where a vacancy sits. Order follows the polyhedron's `VERTEX_DIRS`.
    """

    def sym(d):
        return f"{iso.mol.GetAtomWithIdx(d).GetSymbol()}{d}" if d != VACANT else "·"

    return " ".join(sym(d) for d in iso.vertices)


arrange = arrangement  # alias so IsomerSet.filter(arrangement=…) can still call the formatter (param shadows it)


class IsomerSet(list):
    """The coordination isomers of a metal centre: a ``list`` of `Isomer` to iterate, index, or pick from.

    Pick one to conf-search just the one you want. The identity is **geometric, not a chemistry name**:
    select on the per-vertex `arrangement` (which donor sits where), the metal-centre `chirality`
    (``'Δ'``/``'Λ'``/``''``), the `geometry`, or the plain **index** — the cis/trans/mer/fac `label` is a
    coarse, sometimes-wrong convenience tag and is *never* required to select:

        isos = rx.metal('CCCN[Pd](Cl)(Cl)NCCC', ['square_planar', 'tetrahedral']); isos.summary()
        ens  = rx.embed(isos.select(arrangement='N3 Cl6 N7 Cl5')).mc().prune()   # unambiguous
        ens  = rx.embed(isos.select(geometry='square_planar', chirality='Δ')).mc().prune()
        ens  = rx.embed(isos[0]).mc().prune()                                    # or just by index

    Enumeration is cheap (just the coordination constraints, no embedding); the expensive MC conformer
    search runs only on the `Isomer` you pick.
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


def enumerate_isomers(mol, geometry=None, center=None, fix=None, stereo="racemic"):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    Returns an `IsomerSet`; each `Isomer` carries `.geometry` + `.label` -- pick with ``IsomerSet.select(…)``.
    This is the metal **load-in**: for a coordinate-free input (SMILES) `stereo='racemic'` also enumerates any
    UNDEFINED *ligand* stereocentre (the alpha-carbon of an amino-acidate, a chiral-at-P donor, …), so the set
    spans coordination x ligand-stereo — each such `Isomer` carries a `.stereo_label` (select on ``stereo=``).
    `stereo='free'` opts out (one arbitrary hand). A geometry input (.xyz) has 3D-defined stereo — untouched.

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
            from rxembed.embed.dispatch import parse_smiles

            mol = Chem.AddHs(parse_smiles(mol))  # clear error on a bad SMILES, not a cryptic AddHs(None)
    if stereo != "free" and mol.GetNumConformers() == 0:  # coordinate-free ligand-stereo load-in: expand any
        from rxembed import stereo as _stereo  # UNDEFINED ligand stereocentre so the set spans coordination x

        variants, n_unassigned, _total, unresolved = _stereo.enumerate_unassigned(
            mol, exclude=set(metal_indices(mol))
        )  # exclude the metal's own centre — its Λ/Δ is enumerated below, not as RDKit point stereo
        if n_unassigned:
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
        for order in isomers(base, padded, geom, perms=perms, r_metal=_PT.GetRcovalent(real_z)):
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
                    chirality=chirality_of(base, donors, geom, od),
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
    frag = _frag_map(mol)
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


_SPAN_ANGLE = 135  # a same-ligand donor pair at a vertex separation this wide is a *trans*-type span
_SPAN_TOL = 0.1  # Å slack on the backbone-reach test — tight enough to drop a 5-membered chelate (e.g. an
# amidate, backbone ~3.7 Å) forced *trans* (need ~3.85 Å), while a genuine long bridge (backbone >> need,
# e.g. a flexible bis-NHC at ~6.5 Å) still passes. Bigger slack (0.2) let the amidate-trans phantom through.


def _donor_span_bounds(mol, donors, frag):
    """Return each intra-ligand donor pair's **upper-bound** donor-donor distance (Å), from the bounds matrix.

    RDKit's ``GetMoleculeBoundsMatrix`` gives distance-geometry bounds derived from the ligand's own
    connectivity + knowledge — i.e. how far *this* ligand's two donors can actually reach (the chemistry of
    the isolated backbone). Used only by the trans-span feasibility gate. Empty (→ never filter, defer to
    `bonding_ok`) if the matrix can't be built.
    """
    pairs = [
        (min(a, b), max(a, b))
        for i, a in enumerate(donors)
        for b in donors[i + 1 :]
        if frag[a] == frag[b]  # same ligand only — an inter-ligand pair has no fixed backbone distance
    ]
    if not pairs:
        return {}
    try:
        from rdkit.Chem import rdDistGeom

        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except Exception:  # pragma: no cover — bounds matrix is robust, but never let the gate crash enumeration
        return {}
    return {(a, b): float(bm[a][b]) for a, b in pairs}  # bm[i<j] is the pair's upper bound


def _chelate_span_ok(mol, od, frag, dirs, span_bounds, r_metal):
    """Return False if a chelate is placed *trans* across the metal its backbone can't physically reach.

    A bidentate at *adjacent* (cis) vertices always folds in — no lower span to meet — so only a **wide**
    vertex separation (`_SPAN_ANGLE`+, i.e. trans) is tested: the two donors would sit on opposite sides of
    the metal, needing a donor-donor distance ``law_of_cosines(d_Ma, d_Mb, θ)`` (with each ``d_M`` the real
    metal + donor covalent-radii sum) that a short backbone cannot span. `span_bounds[(a, b)]` is the
    ligand's own upper-bound donor-donor distance (from the bounds matrix), so a genuine long-bridge /
    macrocyclic ligand that *can* reach trans is still allowed — geometry, not a topological "bidentate can't
    span trans" guess. Generalises `_central_trans` to any denticity.
    """
    for p in range(len(od)):
        for q in range(p + 1, len(od)):
            a, b = od[p], od[q]
            if VACANT in (a, b) or frag[a] != frag[b]:  # only a same-ligand (chelate) pair
                continue
            theta = _vertex_angle(dirs[p], dirs[q])
            if theta < _SPAN_ANGLE:  # cis / adjacent -> the chelate folds in, always feasible
                continue
            d_ma = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(a).GetAtomicNum())  # real M-donor covalent sums,
            d_mb = r_metal + _PT.GetRcovalent(mol.GetAtomWithIdx(b).GetAtomicNum())  # not a fixed 2.0 (Pd-N ~2.1)
            need = math.sqrt(d_ma**2 + d_mb**2 - 2 * d_ma * d_mb * math.cos(math.radians(theta)))  # law of cosines
            if span_bounds.get((min(a, b), max(a, b)), math.inf) < need - _SPAN_TOL:  # backbone can't reach
                return False
    return True


def isomers(mol, donors, geometry, perms=None, r_metal=1.4):
    """Enumerate distinct coordination isomers: **every** distinct vertex arrangement, minimally pre-filtered.

    Two geometric impossibilities are dropped here so they don't sit in the isomer set as spurious
    candidates: a tridentate's central donor placed trans to its own arm (`_central_trans`), and a chelate
    placed *trans* across the metal whose backbone cannot span that far (`_chelate_span_ok` — a
    bounds-matrix reach test, so a real long-bridge ligand that *can* span trans is kept). Both are
    geometry-driven, not a topological "bidentate can't span trans" guess.

    `perms` overrides the candidate vertex orderings (default ``PERMUTATIONS[geometry]``) — a ``fix=``
    enumeration passes the subset that keeps each frozen donor pinned to its input vertex.

    A *cis* chelate placement is always kept (its bite folds to whatever the backbone dictates, tight or
    wide — a 3-membered / side-on ligand embeds fine); any residual infeasible arrangement still embeds with
    a torn ligand bond and is dropped downstream by `bonding_ok` in `Ensemble.minimize`.

    Dedup is **connectivity-aware**: the signature carries, per donor pair, the elements, the vertex angle,
    **and** the intra-ligand bond distance for a same-ligand pair, plus the centre's Λ/Δ chirality (so
    enantiomers stay distinct). Genuinely distinct arrangements that share an element/angle pattern — a
    tridentate's terminal-trans (mer) vs the central-trans one — are kept apart rather than merged.
    """
    perms = perms if perms is not None else PERMUTATIONS.get(geometry, [list(range(len(donors)))])
    dirs = VERTEX_DIRS.get(geometry)
    if dirs is None:
        return perms
    elem = {d: (mol.GetAtomWithIdx(d).GetSymbol() if d != VACANT else "X") for d in donors}  # vacancy = "X"
    frag = _frag_map(mol)  # same ligand = same fragment
    dmat = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances
    real_donors = [d for d in donors if d != VACANT]
    span_bounds = _donor_span_bounds(mol, real_donors, frag)

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
        if not _chelate_span_ok(mol, od, frag, dirs, span_bounds, r_metal):  # chelate can't span an unreachable trans
            continue
        pairs = [(p, q) for p in range(len(od)) for q in range(p + 1, len(od))]
        sig = tuple(
            sorted(
                (tuple(sorted((elem[od[p]], elem[od[q]]))), link(od, p, q), _vertex_angle(dirs[p], dirs[q]))
                for p, q in pairs
            )
        )
        sig = (sig, chirality_of(mol, real_donors, geometry, od))  # keep Λ/Δ enantiomers distinct (else merged)
        if sig not in seen:
            seen.add(sig)
            out.append(order)
    return out
