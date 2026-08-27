"""Metal-complex coordination primitives: the surrogate, the sphere perception helpers, the shape holds.

The metal is held purely by distance and angle constraints, its bonds removed, and embedded or relaxed with a
UFF-typeable surrogate atom in its place, so the whole path is plain RDKit and UFF with no xtb.
`repair_bond_stereo` lives here too, being the Mol->Mol cleanup that same bond surgery needs.
"""

from __future__ import annotations

import itertools
import logging
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

from .constraints import add_distance, add_pairwise_shape
from .metal_polyhedron import (
    POLYHEDRA,
    fit_residual,
    geometries_for_cn,
    resolve_geometry,
)
from .metal_polyhedron import describe as _describe
from .utils import bond_removal_mirrors, remove_bond, repair_bond_stereo

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__ ("rxembed.metal_core"): this is
#   the name `set_verbose` configures and every caplog filter in the suite matches.

# The d-block proper: Sc-Zn, Y-Cd, La, Lu, Hf-Hg. A chemistry set, not the centre predicate. It names the
# elements the tmQM-fitted tables were trained on (exactly the keys of `metal_distance._METAL_GROUP`), which
# is why La and Lu are in it and Ce-Yb are not. "Is this atom a coordination centre" is `COORDINATION_METALS`
# below; every reader that asked the centre question through this name (`metal_distance`, `metal_enumeration`,
# `metal_smiles`, `pipeline/nci`, `pipeline/ensemble`) now reads that one instead, so no module in `src/`
# reads this set. It states what the fit covers, and the suite reads it to find the metal in a d-block input.
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
# Any coordination centre, f-block included: the question is meaningful wherever ligands coordinate, and
# M-L bonds are dative, so a clash gate must exclude them or a metal reads as clashing with its own sphere.
#
# One question, one set. Every gate acting on "there is a metal here" reads this one, or two of them disagree
# about the same atom: `embed._check_bare_mol` refused an un-surrogated centre on the narrow d-block set while
# `relax.bonding_ok` exempted it from the clash gate on this one, so a lanthanide walked past the guard and
# then lost the gate that would have caught the result (measured: `[Ce](Cl)(Cl)Cl` embedded, RDKit printing
# "UFFTYPER: Unrecognized atom type: Ce2+3", while `[Fe](Cl)(Cl)Cl` was correctly refused).
COORDINATION_METALS = (
    frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
)
SURROGATE = 6  # carbon: its excluded volume stops a ligand folding into the metal, so the distance geometry keeps it
VACANT = -1  # materialized empty vertex; immutable MetalState stores it as None


class HapticSite(NamedTuple):
    """Identify one haptic coordination site by its real face atoms and stated winding."""

    atoms: tuple
    winding: str = ""


class MetalState(NamedTuple):
    """Identify one real metal and its coordination arrangement."""

    atom: int
    atomic_num: int
    charge: int
    geometry: str = ""
    vertices: tuple = ()
    hand: str = ""


def from_vertices(metal, atomic_num, charge, geometry, vertices, haptic, winding=(), hand=""):
    """Build one real-atom state from materialized coordination vertices."""
    winding = dict(winding)
    sites = tuple(
        None
        if donor == VACANT
        else HapticSite(tuple(haptic[donor]), winding.get(donor, ""))
        if donor in haptic
        else donor
        for donor in vertices
    )
    return MetalState(metal, atomic_num, charge, geometry, sites, hand)


def materialized_states(mol, centres):
    """Assign transient centroid indices to states in centre and vertex order."""
    dummy = mol.GetNumAtoms()
    out = {}
    for state in centres:
        vertices, haptic, winding, donors = [], {}, {}, []
        for site in state.vertices:
            if site is None:
                vertices.append(VACANT)
            elif isinstance(site, HapticSite):
                vertices.append(dummy)
                haptic[dummy] = site.atoms
                donors.extend(site.atoms)
                if site.winding:
                    winding[dummy] = site.winding
                dummy += 1
            else:
                vertices.append(site)
                donors.append(site)
        out[state.atom] = (vertices, haptic, winding, donors)
    return out


def materialized_state(iso, state):
    """Return transient vertices, haptic faces, windings and donors for one state."""
    centres = tuple(state if current.atom == state.atom else current for current in iso.centres)
    return materialized_states(iso.mol, centres)[state.atom]


def state_with_winding(state, vertices, winding):
    """Return a state carrying `winding` on its materialized haptic vertices."""
    sites = tuple(
        HapticSite(site.atoms, winding.get(vertices[position], "")) if isinstance(site, HapticSite) else site
        for position, site in enumerate(state.vertices)
    )
    return state._replace(vertices=sites)


def _frag_map(mol):
    """Map each atom index -> its fragment id (same ligand = same fragment)."""
    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}


def _vertex_atom(haptic, v):
    """Resolve a coordination vertex to a representative real atom; a centroid dummy maps through its ring.

    Every vertex-keyed enumeration lookup must go through here. A centroid dummy is bond-less, so it carries
    none of the fragment identity or backbone reach its ring atoms do, and a tethered face (an ansa Cp) would
    otherwise read as a separate ligand.
    """
    return haptic[v][0] if haptic and v in haptic else v


def metal_index(mol):
    """Index of the first metal coordination centre (d- or f-block), or None."""
    return next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS), None)


_MIN_STEREO_NEIGHBOURS = 3  # a tetrahedral stereocentre needs >=3 explicit neighbours (else ETKDG raises)
_TETRAVALENT = 4  # a fully-substituted (already degree-4) donor is not a candidate for the D-cap chirality hold


def _clear_labile_donor_stereo(atom):
    """Drop a chiral tag on a donor left below 3 neighbours by the sphere strip.

    A donor that is a stereocentre only while metal-bound (a planar amidate N-) keeps a stale tag that crashes
    ETKDG. A genuine one (chiral-at-P, a carbanion C-) still has 3 substituents after the strip and is
    untouched, so its enantiomers embed distinctly.
    """
    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and atom.GetDegree() < _MIN_STEREO_NEIGHBOURS:
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)


def _labile_donors(mol, donors):
    """Return the metal-bound donors the surrogate can't hold natively: sp3 C or N with a chiral tag.

    A carbanion-C or amine-N donor is a stereocentre only while metal-bound. The surrogate strips that bond,
    so it becomes a bare degree-3 centre RDKit and UFF will invert. A heavy pnictogen (P/As/Sb) stays
    configurationally stable at degree 3 and is excluded. `donors` may be empty, since a `from_surrogate`
    isomer from the frozen-core path does not track them, and then there are no labile donors.
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

    Neighbour order is stable for a fixed mol, so the sign is comparable across that mol's conformers. Used
    to cull a conformer whose labile donor inverted, via a relax, re-embed or mc stray, off its enumerated hand.

    The sign is RDKit's own convention, negative for `CHI_TETRAHEDRAL_CW`, and holds at degree 3 as at degree
    4 (the fourth reference is then the centre itself, which leaves the triple product unchanged).
    """
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors()]
    if len(nbrs) < _MIN_STEREO_NEIGHBOURS:
        return None
    conf = mol.GetConformer(cid)
    p = np.array([list(conf.GetAtomPosition(i)) for i in [donor, nbrs[0], nbrs[1], nbrs[2]]])
    v = float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))
    return int(np.sign(v)) if abs(v) > 1e-6 else None  # noqa: PLR2004  a near-planar centre has no hand


_HAND_TAG = {  # `donor_chirality_sign` -> the tag naming that hand in the atom's current bond order
    -1: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    +1: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}


def _retag(mol, hands, ambiguous):
    """Re-apply each stripped donor's tetrahedral tag, in the bond order the strip left behind.

    `ambiguous` names the donors whose carried symbol does not determine a hand; there the geometry decides.
    """
    # Needed at all because the lenient sanitize drops a tag off any donor it types non-SP3, and
    # `repair_bond_stereo`'s AssignStereochemistryFrom3D wipes one it then refuses to re-derive below degree
    # 4. So the caller runs this last.
    #
    # `remove_bond` re-based every tag in `hands` as it took the M-L bond out, which is the whole answer
    # wherever the incoming basis was known. It is not known for a DATIVE M-L bond at an odd slot, and only
    # there: RDKit's 3D writer leaves that bond out of the basis and its SMILES parser counts it, so the two
    # bases name opposite hands and the graph does not record which was used. `utils.assign_stereo_from_3d`
    # settles it for every tag rxembed writes, but RDKit's own molblock reader writes tags too, so a Mol can
    # arrive already mis-based and no predicate over the graph can tell. A conformer can. Consulting it HERE
    # and only here costs the caller a declared hand at that one shape instead of at every donor, which is
    # what the previous unconditional measure did, silently and with no log line.
    for d, carried in hands.items():
        atom = mol.GetAtomWithIdx(d)
        if carried == Chem.ChiralType.CHI_UNSPECIFIED or atom.GetDegree() < _MIN_STEREO_NEIGHBOURS:
            continue
        decided = carried
        if d in ambiguous and mol.GetNumConformers():
            decided = _HAND_TAG.get(donor_chirality_sign(mol, -1, d), carried)
        atom.SetChiralTag(decided)


def _basis_is_ambiguous(mol, donor, metal) -> bool:
    """Whether `donor`'s carried tag could equally be a parity with or without its bond to `metal`.

    True exactly when that bond is DATIVE, leaves the donor, and sits at a slot the removal would mirror.
    Anywhere else both of RDKit's conventions agree, so the symbol names one hand and is honoured as given.
    """
    bond = mol.GetBondBetweenAtoms(int(donor), int(metal))
    if bond is None or bond.GetBondType() != Chem.BondType.DATIVE or bond.GetBeginAtomIdx() != int(donor):
        return False
    return bond_removal_mirrors(mol.GetAtomWithIdx(int(donor)), int(metal))


_DUMMY_M_LO, _DUMMY_M_HI = 0.8, 1.8  # Å: pin the hold-dummy D near the metal (~ the coordinate-bond / lone-pair side)


def _shift_phantoms(cons, offset):
    """Shift every reserved haptic-centroid index by ``offset`` and re-key its constraint fields.

    Two transients are appended from the real atom count, a haptic centroid dummy and a labile donor's D-cap,
    but `materialise_phantoms` requires the centroid keys to be the consecutive block after that count, so
    whichever is appended second collides. This makes the caps take the low indices and the centroid block
    slide above them, so haptic centroids and donor-chirality caps can coexist.
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


def _hold_donor_chirality(mol, metal, donors, cons):
    """Cap each labile (sp3 C/N) metal-bound donor carrying a chiral tag with a dummy D, so the hand is enforced.

    A degree-3 carbanion or amine donor has no M-C bond in the surrogate, so RDKit and UFF do not perceive a
    stereocentre and its two enumerated hands relax to the same geometry. Neutralising its charge and adding a
    4th bond to a deuterium makes it a proper tetrahedral centre in the same appended-D basis the enumeration
    labelled it. With conformers (relax/mc) each D goes at the 4th vertex of that conformer's hand; without
    them, on the initial embed, the hand comes from ETKDG and the tag. A (metal, D) distance pins D on the
    coordinate-bond side, without which ETKDG thrashes ~200x slower, and `_release_donor_chirality` drops that
    key. Returns ``(capped_mol, held)``.
    """
    labile = _labile_donors(mol, donors)
    if not labile:  # the common case: no carbanion/amine stereocentre, so skip the RWMol copy and sanitize
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
        rw.GetAtomWithIdx(dm).SetIsotope(2)  # deuterium: distinct from any real H, lowest CIP priority
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
    for d, t in donor_tags.items():  # carbanion/amine tag -> keep it so the next hold (relax, re-embed, mc) fires
        out.GetAtomWithIdx(d).SetChiralTag(t)
    _shift_phantoms(cons, -len(held))
    return out


def surrogate_metal(mol):
    """Remove metal-donor bonds and swap the metal to a UFF surrogate. Returns (mol, metal, donors, real_Z, real_q)."""
    m = metal_index(mol)
    if m is None:
        raise ValueError("no metal centre found")
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
    em = Chem.RWMol(mol)
    hands = {}  # donor -> the tag it must carry in the bond order the strip leaves behind
    ambiguous = {d for d in donors if _basis_is_ambiguous(em, d, m)}  # read BEFORE the bond goes
    for d in donors:
        remove_bond(em, d, m)  # re-bases the tag: an M-L bond at an odd slot mirrors the symbol it leaves
        a = em.GetAtomWithIdx(d)
        _clear_labile_donor_stereo(a)  # a donor that's a stereocentre only while bound
        a.SetNumExplicitHs(a.GetTotalNumHs())  # freeze the count, rather than deleting it: `SetNoImplicit`
        a.SetNoImplicit(True)  # alone zeroes an implicit H, and an aqua that loses its two protons then reads
        # as a terminal oxo (`ligand_valence`). A no-op on the explicit-H mol `embed` requires.
        hands[d] = a.GetChiralTag()
    real_z = em.GetAtomWithIdx(m).GetAtomicNum()
    a = em.GetAtomWithIdx(m)
    real_q = a.GetFormalCharge()  # zeroed here, handed back by `restore_metal`
    a.SetAtomicNum(SURROGATE)
    a.SetNoImplicit(True)
    a.SetFormalCharge(0)
    # A bondless surrogate is never a stereocentre, and a stray tag from a metal RDKit mis-flagged as
    # tetrahedral would crash ETKDG with 'nbrs.size() >= 3'.
    a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    out = em.GetMol()
    # Lenient (no valence checks), the same tolerance the reader admitted this structure under and the same as
    # `surrogate_all_metals`; a strict sanitize would reject what perception waived (a quinoid ring, a BPh4-).
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    repair_bond_stereo(out)  # the strip can orphan a C=N whose reference atom was the metal; see there
    _retag(out, hands, ambiguous)  # last: the lenient sanitize and `repair_bond_stereo` both wipe atom tags
    return out, m, donors, real_z, real_q


def restore_metal(mol, metal, real_z, real_q):
    """Swap `metal` back from the surrogate to its real element and formal charge.

    The charge is what reaches the calculator, so restoring only the element leaves an M(0) among anionic
    ligands and every real energy runs at the wrong total charge. The surrogate itself must stay neutral: a
    charged bond-less carbon is not a valid DG atom.
    """
    a = mol.GetAtomWithIdx(metal)
    a.SetAtomicNum(real_z)
    a.SetFormalCharge(real_q)


def connect_metal(mol, donor_bonds, *, order=Chem.BondType.DATIVE):
    """Re-add the surrogate-stripped M-donor bonds as dative (donor->metal), returning a connected Mol.

    `restore_metal` is calculator-minimal, since xtb needs no graph, so it leaves the metal topologically
    detached. Anything that uses the mol (perception, re-embedding, a downstream tool) needs the M-L bonds, so
    this re-adds them once geometry, element and charge are settled.

    Dative rather than covalent: it counts toward the metal's valence, never the donor's, so it restores
    connectivity without touching any ligand's valence, H-count or charge. Coordinates untouched, idempotent.
    A bond the input drew covalent comes back dative, so the metal picks up radical electrons RDKit would
    otherwise pair; element, formal charge and total charge, which is what a calculator reads, are unaffected.

    `order` exists for the one caller that needs the opposite: `metal_smiles.cxsmiles` asks for single
    on the sigma donors, because writing a string has to re-derive the ionic form from the donor's valence,
    and a dative bond is already the answer to that question. Every other caller wants the default.
    """
    rw = Chem.RWMol(mol)
    added = False
    for d, m in donor_bonds:
        if rw.GetBondBetweenAtoms(int(d), int(m)) is None:
            rw.AddBond(int(d), int(m), order)
            added = True
    if not added:
        return mol
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)  # recompute implicit valence (never raises, never moves a formal charge)
    Chem.FastFindRings(
        out
    )  # the dative bonds close chelate rings through the metal, so re-perceive rings or the dedup SMARTS and the
    # geometry gate read the wrong ones
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
            mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() in COORDINATION_METALS
            or mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() in COORDINATION_METALS
        )
    ]
    if not dative:
        return mol
    rw = Chem.RWMol(mol)
    for a, b in dative:
        remove_bond(rw, a, b)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def metal_indices(mol):
    """Return the indices of every metal coordination centre (d- or f-block).

    The one answer to "which atoms does the surrogate own", so every door reads it: the `Isomer` enumeration,
    `embed.prepare`'s perceived-complex path, and the guard that refuses an un-surrogated centre.
    """
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]


_ETA2 = 2


def _haptic_sites(mol, donors):
    """Group `donors` into coordination sites: a directly-bonded pi-face (Cp/arene/allyl) collapses to one.

    A lone sigma donor is its own 1-tuple. A chelate's donors relate only through the backbone, so they stay
    separate and only a genuine haptic face is grouped: ferrocene's ten ring donors reduce to two sites. Run
    on the surrogate, whose M-donor bonds are stripped, so the walk follows only ligand bonds.
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
    """Return True if `site`'s own bond graph is regular: every face atom has the same face-neighbour count.

    Regularity is what makes one shared centroid radius true, so it is the rigidity test. Degree 1 is an edge
    (eta2), degree 2 a cycle (Cp/arene), while an open allyl is irregular and its centroid sits nearer the
    inner atoms. One rule for every hapticity: a bond is the smallest rigid face, so eta2 needs no special case.
    """
    face = set(site)
    deg = {sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in face) for a in face}
    return len(deg) == 1


def _collapse_haptic(mol, donors):
    """Collapse each haptic face to one centroid vertex; sigma donors pass through.

    Appends a bond-less carbon centroid per face, leaving existing indices unchanged, seated at the ring
    centroid if the mol has a conformer. Returns ``(mol, vertices, haptic)``, where `haptic` maps each dummy to
    its ring atoms. The dummy is embed scaffolding that lives in no stored Mol: `enumerate_isomers` strips it
    before storing the real `Isomer`, and only `bounds.seed_coordinates` / `restrained_uff` re-materialise it.
    A mol with no haptic face is returned untouched.
    """
    sites = _haptic_sites(mol, donors)
    faces = [s for s in sites if len(s) > 1]  # a face is any mutually-bonded donor group, eta2 included
    if not faces:
        return mol, donors, {}
    em = Chem.RWMol(mol)
    conf = em.GetConformer() if em.GetNumConformers() else None
    vertices = [d for s in sites if len(s) == 1 for d in s]  # a lone sigma donor is its own vertex
    haptic = {}
    for site in faces:
        idx = em.AddAtom(Chem.Atom(SURROGATE))  # a bond-less carbon: excluded volume in the DG, Xe-ghosted in the FF
        dummy = em.GetAtomWithIdx(idx)
        dummy.SetNoImplicit(True)  # a point, not a valence: no implicit H, as for the metal
        dummy.SetHybridization(Chem.HybridizationType.SP3)  # keep the transient point typable by UFF
        if conf is not None:
            conf.SetAtomPosition(idx, Point3D(*np.mean([list(conf.GetAtomPosition(a)) for a in site], axis=0)))
        vertices.append(idx)
        haptic[idx] = site
    out = em.GetMol()
    out.UpdatePropertyCache(strict=False)  # bond-less centroids' valence (GetMoleculeBoundsMatrix needs it)
    Chem.FastFindRings(out)  # + RingInfo (the RWMol edit cleared it), which GetMoleculeBoundsMatrix requires
    return out, vertices, haptic


class Ligand(NamedTuple):
    """One ligand of a metal complex, as a standalone Mol plus which of its atoms coordinate."""

    mol: Chem.Mol  # standalone, carrying its own copy of the complex's conformer
    donors: dict  # {metal index in the ORIGINAL complex: [donor indices in `mol`]}; a bridging ligand has 2 keys
    atoms: tuple  # this ligand's indices in the original complex, positionally matching `mol`


def ligands(mol):
    """Return the complex's ligands, each separated from the metal(s) it coordinates.

    A read, not a builder: it reports which atoms coordinate and which ligand each belongs to, and hands back
    plain RDKit Mols. What you do with them is yours.

    Donors are reported per metal rather than pooled, so a ligand bridging two centres says so: on a Mn/Fe
    complex its backbone is kappa3 to one and kappa5 to the other, not kappa8 to nothing in particular. A
    ligand is a connected fragment, so a hydrogen bond perceived as a bond fuses two of them into one.
    """
    metals = metal_indices(mol)
    if not metals:
        raise ValueError("no metal centre found: this reads a metal complex's coordination sphere")
    bound = {m: {n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()} for m in metals}
    base, _info = surrogate_all_metals(mol)
    mapping = []
    frags = Chem.GetMolFrags(base, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)

    out = []
    for frag, atoms in zip(frags, mapping, strict=True):
        if set(atoms) & set(metals):  # a surrogated metal is not a ligand of itself
            continue
        donors = {m: [k for k, a in enumerate(atoms) if a in ds] for m, ds in bound.items()}
        out.append(Ligand(frag, {m: d for m, d in donors.items() if d}, tuple(atoms)))
    return out


_PT = Chem.GetPeriodicTable()


def ligand_valence(atom):
    """Bond orders `atom` spends on its LIGAND side, protons included; every bond to a metal is excluded.

    Metal-blind by construction, so it reads the same on the real complex and on `surrogate_metal`'s bond-less
    one. Implicit and explicit hydrogens both count, since they are exactly what separates an aqua from an oxo.
    """
    return atom.GetTotalNumHs() + sum(
        b.GetBondTypeAsDouble()
        for b in atom.GetBonds()
        if b.GetOtherAtom(atom).GetAtomicNum() not in COORDINATION_METALS
    )


def donated_charge(donor):
    """Give the charge a donor carries once its bond to the metal is written the ionic way, donor -> metal.

    Decided from VALENCE, never from the perceived M-L bond order, because no two perceivers agree on that one:
    xyzgraph types a terminal oxo as a neutral single-bonded `[O]` and the nitrido beside it as `[N-3]`, while
    a SMILES writes the same oxo `O=[M]`. What survives all three is what the LIGAND side leaves unsatisfied.
    A donor with nothing but metals on it has no other way to fill its shell, so it donates its whole valence
    and carries the matching charge: `M=O` is `[M2+]<-[O2-]`, `M#N` is `[M3+]<-[N3-]`, `M-Cl` is `[M+]<-[Cl-]`.
    A donor its ligand side already satisfies (a phosphine, an aqua, an ether) donates a lone pair and stays
    neutral, which is what a dative bond means; charging a coordinated PR3 would invent an ion.

    Zero for anything a metal is not the whole of: the alkoxide / amide / carbanion case, where a partly-filled
    donor's charge is a Kekule accident and belongs to `metal_distance.delocalised_charges` instead. Reaching
    past the empty ligand side is measured and refuted (see `ml_distance`), and it must never reach a pi face:
    counting a Cp carbon short is what wrote ferrocene as `[Fe+10]` with ten `[c-]`.

    Call it on a donor. A bare counterion has an empty ligand side too, and would come back charged.
    """
    valence = _PT.GetDefaultValence(donor.GetAtomicNum())
    if valence <= 0 or ligand_valence(donor):  # a metal (no default valence), or a donor its ligand already fills
        return 0
    return -valence


def materialise_phantoms(mol, haptic):
    """Return a copy of `mol` with each haptic centroid dummy appended (bond-less carbon) at its reserved index.

    A transient of the embed, living in no persistent Mol. `bounds.seed_coordinates` and `restrained_uff` materialise it
    from `Constraints.haptic` so a Cp/arene face embeds as one rigid vertex, then discard it. Reserved indices
    are consecutive from the real atom count, so append order reproduces them, and each dummy is seated at its
    ring centroid. A no-op when there is no haptic face.
    """
    if not haptic:
        return mol
    base_n = mol.GetNumAtoms()  # dummies are appended, so their reserved keys are the next consecutive indices
    if sorted(haptic) != list(range(base_n, base_n + len(haptic))):  # another transient was appended first
        raise ValueError(
            f"haptic centroid indices {sorted(haptic)} are not the {len(haptic)} indices after {base_n} atoms: "
            f"another transient atom (a donor-chirality D-cap?) was appended first, and a haptic face cannot "
            f"compose with a carbanion/amine-donor chirality hold"
        )
    rw = Chem.RWMol(mol)
    for _dummy in sorted(haptic):
        a = rw.GetAtomWithIdx(rw.AddAtom(Chem.Atom(SURROGATE)))
        a.SetNoImplicit(True)  # a point, not a valence
        a.SetHybridization(Chem.HybridizationType.SP3)  # match the sanitised metal surrogate; otherwise UFF and
        #   the bounds matrix emit 'unrecognized hybridization', UpdatePropertyCache leaving it UNSPECIFIED
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)  # GetMoleculeBoundsMatrix / UFF need the new bond-less atoms' valence
    Chem.FastFindRings(out)  # + RingInfo (an RWMol edit clears it), which GetMoleculeBoundsMatrix requires
    for conf in out.GetConformers():
        for dummy, ring in haptic.items():
            conf.SetAtomPosition(dummy, Point3D(*np.mean([list(conf.GetAtomPosition(a)) for a in ring], axis=0)))
    return out


def strip_phantoms(mol, phantoms):
    """Return a copy of `mol` with the haptic centroid dummies removed; real indices unchanged.

    Embed scaffolding (`Constraints.phantoms`), never part of a stored Mol: `enumerate_isomers` strips it
    before storing the real `Isomer` so nothing downstream sees it. Dummies are the highest indices, so
    removing them from the end leaves every real index untouched. A no-op when there are no phantoms.
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

    From the conformer if the mol has one (a retained crystal geometry), else from RDKit's own bounds
    midpoints as ``R = sqrt(sum d_ij^2)/n``, which is exact for a regular n-gon and so serves Cp, arene and
    boratabenzene's longer B-C edge alike with no hand-fitted radii. A 2-atom eta2 site reduces to half the edge.
    """
    if mol.GetNumConformers():
        pos = np.array([list(mol.GetConformer(cid).GetAtomPosition(a)) for a in site])
        return float(np.linalg.norm(pos - pos.mean(0), axis=1).mean())

    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    tot = sum(
        (0.5 * (bm[max(a, b)][min(a, b)] + bm[min(a, b)][max(a, b)])) ** 2 for a, b in itertools.combinations(site, 2)
    )
    return float(np.sqrt(tot)) / len(site)


def _reject_metal_bonds(mol):
    """Reject direct metal-metal bonds until their bond type can be restored losslessly."""
    metals = set(metal_indices(mol))
    direct = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in metals and bond.GetEndAtomIdx() in metals
    ]
    if direct:
        raise ValueError(
            f"direct metal-metal bonds {direct} are not supported because their bond type cannot be restored"
        )


def surrogate_all_metals(mol):
    """Surrogate every metal centre (bonds removed, carbon) for a multi-metal complex.

    UFF must type the whole complex of a bimetallic TS, and `surrogate_metal` only does the first metal.
    Returns ``(mol, metals)`` where ``metals`` is ``[(idx, real_z, real_q), ...]``, the element and formal
    charge `restore_metal` needs. Sanitised leniently, since a stripped η⁵-Cp is a radical fragment.
    """
    idxs = metal_indices(mol)
    if not idxs:
        raise ValueError("no metal centre found")
    _reject_metal_bonds(mol)
    em = Chem.RWMol(mol)
    metals, hands, ambiguous = [], {}, set()
    for m in idxs:
        metals.append((m, em.GetAtomWithIdx(m).GetAtomicNum(), em.GetAtomWithIdx(m).GetFormalCharge()))
        for d in [n.GetIdx() for n in em.GetAtomWithIdx(m).GetNeighbors()]:
            if _basis_is_ambiguous(em, d, m):  # read BEFORE the bond goes, as in `surrogate_metal`
                ambiguous.add(d)
            remove_bond(em, d, m)  # as in `surrogate_metal`; a bridging donor's two strips compose here
            a = em.GetAtomWithIdx(d)
            _clear_labile_donor_stereo(a)  # stale tag on a now-<3-nbr donor crashes ETKDG
            a.SetNoImplicit(True)
            hands[d] = a.GetChiralTag()
        a = em.GetAtomWithIdx(m)
        a.SetAtomicNum(SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # bondless surrogate: a stray metal tag crashes ETKDG
    out = em.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    _retag(out, hands, ambiguous)
    return out, metals


_APICAL_MIN = 3  # a face of this many atoms caps a face (a piano stool); an eta2 alkene fills one ordinary
#   in-plane site. Site occupancy, not geometry: it steers the default-polyhedron guess, never the embed.


_EPS_LEN = 1e-9  # a donor sitting on the metal has no direction, so the sphere cannot be read at all
_FIT_FLOOR = (
    0.45  # Procrustes residual above which no record really fits. `classify_geometry` is an unconditional argmin and
)
# at 4 of 12 CNs one record is alone, so it wins by default; this only WARNS, never reclassifies. Sized on
# 45 corpus centres (median 0.069, 90th 0.30), clearing the real band by ~50%.
COPLANAR_TOL = (
    0.25  # Å RMS out-of-plane of {metal + vertices} above which a sphere is not planar. One constant for two jobs,
)
# the accept gate on a declared-`planar` record and the `classify_geometry` exclusion. The price: an
# absolute RMS means a shallow real pyramid (M[N(SiMe3)2]3) reads `trigonal_planar`.


def _plane_rms(metal_pos, verts):
    """Return the RMS distance of {metal + coordination vertices} from their best-fit plane (Å).

    Fewer than 4 points define a plane exactly, so they score 0: trivially coplanar.
    """
    pts = np.array([metal_pos, *verts])
    if len(pts) < 4:  # noqa: PLR2004  a plane needs >=3 points; <4 total is trivially coplanar
        return 0.0
    dev = (pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0))[2][2]  # signed distance from best-fit plane
    return float(np.sqrt(np.mean(dev**2)))


def _ideal_plane_rms(p, r):
    """Return the out-of-plane RMS the record's own ideal sphere measures at M-L bond length `r` (Å)."""
    return _plane_rms(np.zeros(3), [r * np.array(d, float) / np.linalg.norm(d) for d in p.vertex_dirs])


def _too_flat_for(p, rms, r):
    """Return True if an out-of-plane RMS of `rms` at bond length `r` rules the non-planar record `p` out.

    The bound is the QA tolerance or the record's own ideal, whichever is smaller, so a record can never be
    excluded by a sphere that is that record at any bond length. An ideal sphere lands exactly on its bound
    and a float round-trip moves it by ~1e-17, hence the closeness guard rather than a bare ``<``.
    """
    bound = min(COPLANAR_TOL, _ideal_plane_rms(p, r))
    return rms < bound and not np.isclose(rms, bound)


def classify_geometry(mol, metal, sites, cid=-1):
    """Name the coordination polytope from the actual geometry: a flatness exclusion, then the angle spectrum.

    ``sites`` are coordination sites, not atoms: a haptic face is one vertex via its centroid. Returns
    ``None`` only when no record has that vertex count, and warns above `_FIT_FLOOR`, where the name is the
    nearest record rather than a reading of the sphere.

    Flatness excludes one way only: a flat sphere cannot be a record whose metal sits off its vertex plane,
    but the converse says nothing, since an out-of-plane sphere is a distorted planar shape as readily as a
    3-D one. This is what separates `trigonal_planar` from the CN3 pyramid, where an angle boundary would
    have to be fitted and this has a natural zero.

    Then the sorted vertex-metal-vertex angle spectrum, invariant to rotation and donor ordering, so no
    Kabsch and no permutation search. It separates a trigonal bipyramid from a square pyramid, which differ
    only in these angles. The sort discards vertex roles: lossless at CN3, but at CN4 it is 720 orderings
    against 24 realisable, so an apex/base pair cannot be told apart here.
    """
    pos = mol.GetConformer(cid).GetPositions()
    points = []
    for site in sites:
        atoms = [site] if isinstance(site, (int, np.integer)) else list(site)
        points.append(np.mean([pos[a] for a in atoms], axis=0))
    obs = np.array([p - pos[metal] for p in points], float)
    if not np.all(np.linalg.norm(obs, axis=1) > _EPS_LEN):
        return None
    obs = obs / np.linalg.norm(obs, axis=1, keepdims=True)
    rms = _plane_rms(pos[metal], points)
    r = float(np.mean([np.linalg.norm(p - pos[metal]) for p in points]))
    same_cn = [(n, p) for n, p in POLYHEDRA.items() if p.cn == len(points)]
    kept = [(n, p) for n, p in same_cn if p.planar or not _too_flat_for(p, rms, r)]
    dropped = [n for n, _ in same_cn if n not in {k for k, _ in kept}]
    if (
        not kept
    ):  # CN>=5 has no planar record, so a flat sphere there excludes everything: rank them all rather than name
        # nothing (stable sort, so POLYHEDRA order still breaks an exact tie)
        kept, dropped = same_cn, []  # nothing left to discriminate, rank them all rather than name nothing
    # stable sort, so POLYHEDRA insertion order still breaks an exact tie
    ranked = sorted(((fit_residual(obs, p), name) for name, p in kept), key=lambda t: t[0])
    if not ranked:
        return None
    best_err, best = ranked[0]
    poor = best_err > _FIT_FLOOR  # no record fits; the argmin is still returned, but it is a name, not a reading
    runner = f"; next {_describe(ranked[1][1])} {ranked[1][0]:.3f}" if len(ranked) > 1 else ""
    if poor:
        logger.warning(
            "geometry: no shape fits; nearest %s (residual %.3f > %.2f). Pass geometry= to state it",
            _describe(best),
            best_err,
            _FIT_FLOOR,
        )
    else:
        logger.debug("geometry: %s residual %.3f%s", _describe(best), best_err, runner)
    logger.debug(
        "sphere: %s; coplanarity RMS %.3f A vs %.2f tol; M-L %.2f A%s",
        "in-plane" if rms <= COPLANAR_TOL else "out-of-plane",
        rms,
        COPLANAR_TOL,
        r,
        f"; flatness excluded {', '.join(_describe(n) for n in dropped)}" if dropped else "",
    )
    return best


def geometry_for(n_donors, has_apical=False):
    """Default coordination polyhedron name for `n_donors`, or None.

    An apical (eta>=3) face is an axial cone, so a CN4 carrying one is a piano stool, never the flat
    `square_planar` the vertex count would pick, which would seat a ligand trans through the ring. Only CN4
    flips. An eta2 face is not apical: it is an ordinary single-site vertex and keeps the default.
    """
    gs = geometries_for_cn(n_donors)  # best-default first
    g = gs[0].name if gs else None
    if has_apical and g == "square_planar":
        return "tetrahedral"
    return g


def coplanar(pos, metal, donors, tol=COPLANAR_TOL, haptic=None):
    """Return True if the metal and its coordination vertices lie in one plane.

    The feasibility test for a declared planar record: a bite squeezing the in-plane angles is still planar,
    but an arrangement that can only satisfy its ligands by twisting out of plane (an impossible trans-chelate)
    is not. An eta>=3 face is one vertex, so `haptic` collapses each face to its centroid first; that
    bookkeeping is all this adds over `_plane_rms`.
    """
    ring_atoms = {a for ring in (haptic or {}).values() for a in ring}
    verts = [pos[d] for d in donors if d not in ring_atoms]  # each sigma/eta2 donor is its own vertex
    verts += [np.mean([pos[a] for a in ring], axis=0) for ring in (haptic or {}).values()]  # each face -> centroid
    return _plane_rms(pos[metal], verts) <= tol


def n_sites(geometry):
    """Return the number of coordination vertices the geometry has (name or 3-letter code)."""
    return len(POLYHEDRA[resolve_geometry(geometry)].vertex_dirs)  # unknown geometry -> KeyError (deliberate)


def hold_shape(mol, atoms, cons, pad=0.1, cid=-1):
    """Fix the shape of `atoms` at their input mutual distances (all pairwise, ``+/-pad``).

    Relative and frame-independent, so a spectator sphere survives a re-embed without being pinned to an
    absolute frame. Recorded in ``cons.shapes`` because these windows are one rigid body: pulling the M-donor
    subset alone would tear the rest.
    """
    add_pairwise_shape(cons, atoms, mol.GetConformer(cid).GetPositions(), pad)
    cons.shapes.append(set(atoms))


_LONE_PAIR_Z = {7, 8, 15, 16, 33, 34, 51, 52}  # N O P S As Se Sb Te: p-block groups 15 and 16


def lone_pair_donors(mol, metal, exclude=()):
    """Return substrate lone-pair donors (p-block group 15/16) that could take a vacant site.

    Excludes every atom in a ligand fragment, meaning one holding the metal or an `exclude` donor. The
    surrogate detaches ligands into their own fragments, so without that a ligand-backbone heteroatom (an
    ether O on a phosphine) would read as a free substrate donor.
    """
    frag = _frag_map(mol)
    ligand_frags = {frag[metal]} | {frag[d] for d in exclude}
    return [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _LONE_PAIR_Z and frag[a.GetIdx()] not in ligand_frags
    ]
