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
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

from .metal_polyhedron import (
    POLYHEDRA,
    SLOT_BOND_PROP,
    fit_residual,
    geometries_for_cn,
    resolve_geometry,
)
from .metal_polyhedron import describe as _describe
from .utils import bond_removal_mirrors, flat_ranks, remove_bond, repair_bond_stereo, resonance_match

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
_BORON_Z = 5
_BORON_CAGE_MIN = 5
_PT = Chem.GetPeriodicTable()
SURROGATE = 6  # carbon: its excluded volume stops a ligand folding into the metal, so the distance geometry keeps it
_SURROGATE_SANITIZE = (
    Chem.SanitizeFlags.SANITIZE_ALL
    ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
)
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
    return materialized_states(iso._graph, centres)[state.atom]


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
    return next(iter(metal_indices(mol)), None)


_MIN_STEREO_NEIGHBOURS = 3  # a tetrahedral stereocentre needs >=3 explicit neighbours (else ETKDG raises)


def _clear_labile_donor_stereo(atom):
    """Drop a chiral tag that cannot describe a tetrahedral donor after the sphere strip.

    A donor that is a stereocentre only while metal-bound (a planar amidate N-) keeps a stale tag that crashes
    ETKDG. An aromatic donor is not tetrahedral even when metal perception left stale SP3 hybridization. A
    genuine centre (chiral-at-P, a carbanion C-) still has 3 substituents and remains untouched.
    """
    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and (
        atom.GetIsAromatic() or atom.GetDegree() < _MIN_STEREO_NEIGHBOURS
    ):
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)


def donor_chirality_sign(mol, cid, donor, references=()):
    """Geometric hand (+1 / -1, or None) of a donor: the signed volume of its first three neighbours.

    ``references`` supplies stripped metal neighbours. With four carriers the fourth, rather than the centre,
    is the volume origin, so a bridge is measured as the actual tetrahedron. Neighbour and reference order is
    stable for a fixed mol, making the sign comparable across its conformers.

    Without extra references the sign is RDKit's own convention, negative for `CHI_TETRAHEDRAL_CW`.
    """
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors()]
    nbrs.extend(int(i) for i in references if int(i) not in nbrs)
    if len(nbrs) < _MIN_STEREO_NEIGHBOURS:
        return None
    conf = mol.GetConformer(cid)
    origin = nbrs[3] if len(nbrs) > _MIN_STEREO_NEIGHBOURS else donor
    p = np.array([list(conf.GetAtomPosition(i)) for i in [origin, nbrs[0], nbrs[1], nbrs[2]]])
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


def surrogate_metal(mol):
    """Remove metal-donor bonds and swap the metal to a UFF surrogate. Returns (mol, metal, donors, real_Z, real_q)."""
    mol = _canonical_metal_graph(mol)
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
    Chem.SanitizeMol(out, _SURROGATE_SANITIZE, catchErrors=True)
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


def connect_metal(mol, donor_bonds):
    """Re-add the surrogate-stripped M-donor bonds as dative (donor->metal), returning a connected Mol.

    `restore_metal` is calculator-minimal, since xtb needs no graph, so it leaves the metal topologically
    detached. Anything that uses the mol (perception, re-embedding, a downstream tool) needs the M-L bonds, so
    this re-adds them once geometry, element and charge are settled.

    Dative rather than covalent: it counts toward the metal's valence, never the donor's, so it restores
    connectivity without touching any ligand's valence, H-count or charge. Coordinates untouched, idempotent.
    A bond the input drew covalent comes back dative; element, formal charge and total charge are unaffected.
    """
    rw = Chem.RWMol(mol)
    added = False
    for donor, metal in donor_bonds:
        d, m = int(donor), int(metal)
        if rw.GetBondBetweenAtoms(d, m) is None:
            rw.AddBond(d, m, Chem.BondType.DATIVE)
            added = True
    if not added:
        return mol
    out = rw.GetMol()
    out.ClearComputedProps()  # AddBond preserves cached path matrices from the disconnected graph.
    out.UpdatePropertyCache(strict=False)  # recompute implicit valence (never raises, never moves a formal charge)
    Chem.FastFindRings(
        out
    )  # the dative bonds close chelate rings through the metal, so re-perceive rings or the dedup SMARTS and the
    # geometry gate read the wrong ones
    return out


def disconnect_metal(mol):
    """Remove metal-donor dative bonds from the working graph; return the input when there are none.

    Constraints own the M-L geometry during DG and UFF, independently of native bonded-metal terms.
    Public coordination bonds are restored after relaxation by `connect_metal`.
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
    out.ClearComputedProps()  # Removing bonds also leaves RDKit's cached path matrix behind.
    out.UpdatePropertyCache(strict=False)
    return out


def metal_indices(mol):
    """Return the indices of every metal coordination centre (d- or f-block).

    The one answer to "which atoms does the surrogate own", so every door reads it: the `Isomer` enumeration,
    `embed.prepare`'s perceived-complex path, and the guard that refuses an un-surrogated centre.
    """
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]


def _ligand_distance_matrix(mol):
    """Return graph distances after removing coordination-centre edges."""
    metals = set(metal_indices(mol))
    if not metals:
        return Chem.GetDistanceMatrix(mol)
    rw = Chem.RWMol(mol)
    for bond in list(rw.GetBonds()):
        if bond.GetBeginAtomIdx() in metals or bond.GetEndAtomIdx() in metals:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    out = rw.GetMol()
    out.ClearComputedProps()
    return Chem.GetDistanceMatrix(out)


_ETA2 = 2


def _haptic_sites(mol, donors):
    """Group `donors` into sigma sites and connected pi faces.

    A lone sigma donor is its own 1-tuple. A chelate's donors relate only through the backbone, so they stay
    separate even when directly bonded. A face starts at a donor-donor multiple or aromatic bond and extends
    across adjacent charged or radical donor endpoints; bonds between face atoms then join a diene, allyl,
    Cp, or arene into one site. A true isolated diatomic ligand is one side-on site regardless of its perceived
    Lewis bond order; implicit hydrogens still count as substituents, so hydrazine is not mistaken for a face.
    """
    dset = set(donors)
    pi = set()
    pi_hybridization = {
        Chem.HybridizationType.UNSPECIFIED,
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
    }
    for ring in mol.GetRingInfo().AtomRings():
        if set(ring) <= dset and all(mol.GetAtomWithIdx(atom).GetHybridization() in pi_hybridization for atom in ring):
            pi.update(ring)
    for bond in mol.GetBonds():
        ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        conjugated = bond.GetBondTypeAsDouble() > 1 and all(
            atom.GetHybridization() in pi_hybridization for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
        if ends <= dset and (bond.GetIsAromatic() or conjugated):
            pi.update(ends)
    while True:
        extended = {
            d
            for d in dset - pi
            if (atom := mol.GetAtomWithIdx(d)).GetFormalCharge() or atom.GetNumRadicalElectrons()
            if any(neighbor.GetIdx() in pi for neighbor in atom.GetNeighbors())
        }
        if not extended:
            break
        pi.update(extended)

    def joined(a, b):
        if a in pi and b in pi:
            return True
        return all(
            mol.GetAtomWithIdx(i).GetTotalNumHs() == 0
            and all(
                neighbor.GetIdx() in {a, b} or neighbor.GetAtomicNum() in COORDINATION_METALS
                for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
            )
            for i in (a, b)
        )

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
            stack += [
                nb.GetIdx()
                for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() in dset and joined(a, nb.GetIdx())
            ]
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
    vertices = [d for site in sites if len(site) == 1 for d in site]
    if not faces:
        return mol, vertices, {}
    haptic = {mol.GetNumAtoms() + i: site for i, site in enumerate(faces)}
    return materialise_phantoms(mol, haptic), [*vertices, *haptic], haptic


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


def ligand_degree(atom):
    """Count ligand-side neighbours, including implicit and explicit hydrogen."""
    return atom.GetTotalDegree() - sum(nb.GetAtomicNum() in COORDINATION_METALS for nb in atom.GetNeighbors())


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


def _ionic_donor_charge(donor, haptic):
    """Return the formal charge needed after removing a covalent M-L valence contribution."""
    if donor.GetFormalCharge() or donor.GetNumRadicalElectrons() or donor.GetIdx() in haptic:
        return 0
    default = _PT.GetDefaultValence(donor.GetAtomicNum())
    deficit = default - ligand_valence(donor)
    if default <= 0 or deficit <= 0 or not np.isclose(deficit, round(deficit)):
        return 0
    return -round(deficit)


def _canonicalise_delocalised_charge(mol):
    """Canonicalize one aromatic anion per component when RDKit proves the requested resonance form."""
    rw = Chem.RWMol(mol)
    metals = {atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    if not metals:
        return rw.GetMol()
    # RDKit's resonance proof can expose another form after a proved move changes the cached aromatic
    # representation. Iterate to a fixed point so repeated graph normalization is itself a normal form.
    for _ in range(max(1, rw.GetNumAtoms())):
        bound = {bond.GetOtherAtomIdx(metal) for metal in metals for bond in rw.GetAtomWithIdx(metal).GetBonds()}
        ranks = flat_ranks(rw, break_ties=True)
        aromatic = Chem.RWMol(rw)
        for bond in list(aromatic.GetBonds()):
            if not bond.GetIsAromatic():
                aromatic.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        moved = False
        for component in Chem.GetMolFrags(aromatic):
            charged = [atom for atom in component if rw.GetAtomWithIdx(atom).GetFormalCharge()]
            # Several charges need joint normalization. Moving them ring by ring is order-dependent and can
            # change the string again on the next write, even when every individual move has a resonance proof.
            if len(charged) != 1 or rw.GetAtomWithIdx(charged[0]).GetFormalCharge() != -1:
                continue
            current = charged[0]
            candidates = [atom for atom in component if atom in bound and rw.GetAtomWithIdx(atom).GetIsAromatic()]
            for candidate in sorted(candidates, key=lambda atom: (-ranks[atom], atom)):
                if candidate != current:
                    trial = Chem.RWMol(rw)
                    trial.GetAtomWithIdx(current).SetFormalCharge(0)
                    trial.GetAtomWithIdx(candidate).SetFormalCharge(-1)
                    probe = trial.GetMol()
                    try:
                        with rdBase.BlockLogs():
                            Chem.SanitizeMol(probe)
                            Chem.Kekulize(Chem.Mol(probe), clearAromaticFlags=True)
                            matched, capped = resonance_match(probe, rw.GetMol())
                    except (RuntimeError, ValueError):
                        continue
                    if not matched:
                        if capped:
                            logger.warning(
                                "metal graph: resonance search exceeded 32 forms; preserving aromatic charge placement"
                            )
                            break
                        continue
                if candidate != current:
                    rw.GetAtomWithIdx(current).SetFormalCharge(0)
                    rw.GetAtomWithIdx(candidate).SetFormalCharge(-1)
                    moved = True
                break
        if not moved:
            break
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        rw = Chem.RWMol(out)
    return rw.GetMol()


_BRIDGEHEAD_SIGMA_MIN = 4  # a kappa2 chelate bridgehead (P, Si, B) bonds >=4 non-metal sigma neighbours
_BRIDGEHEAD_DONORS_MIN = 2  # fewer is a sigma-silane/borane bridgehead (one metal-bound neighbour), not this rule
_TRIGONAL_SIGMA = 3  # Class B's bridgehead: exactly 3 non-metal sigma bonds (carboxylate/amidinate C, N-B-N B)
_TRIGONAL_NONDONOR_Z = {1, 6}  # H and C: a TS contact or a genuine eta-n face carbon, never this rule's donor


def _lone_pair(atom, metals):
    """Return `atom`'s nonbonding valence: outer electrons minus formal charge minus bonded valence.

    A bond to a metal is stripped from the bonded-valence term first (`Bond.GetValenceContrib`, zero
    for a dative donor bond, the bond order for a covalent one), so this reads the same whether `atom`
    is itself dative- or covalent-bonded to the metal. Not a per-element list, so it holds for any
    main-group atom; shared by the bridgehead X (Class A and B) and, for Class B, its flanking donors.
    """
    to_metal = sum(
        bond.GetValenceContrib(atom) for bond in atom.GetBonds() if bond.GetOtherAtomIdx(atom.GetIdx()) in metals
    )
    return _PT.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() - (atom.GetTotalValence() - to_metal)


def _metal_free_rings(rw, metals):
    """Return each ring of `rw` with every metal atom removed, as a list of atom-index frozensets.

    Built once per graph for Class B's ring test. `RingInfo.AtomRings()` is a view into its owning
    Mol's C++ memory, so the ring atoms are read out here, before the stripped copy is dropped, rather
    than handing back the live RingInfo (that use-after-free crashed the measurement with a MemoryError).
    A fresh `Chem.Atom` is used per kept atom rather than a copy of the original: copying carries over
    cached valence/implicit-H state from the metal-bonded graph, which corrupted ring perception the
    same way once that atom sat in a differently-bonded graph.
    """
    em = Chem.RWMol()
    kept = {}
    for atom in rw.GetAtoms():
        if atom.GetIdx() in metals:
            continue
        fresh = Chem.Atom(atom.GetAtomicNum())
        fresh.SetFormalCharge(atom.GetFormalCharge())
        kept[atom.GetIdx()] = em.AddAtom(fresh)
    for bond in rw.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in metals or b in metals:
            continue
        em.AddBond(kept[a], kept[b], Chem.BondType.SINGLE)  # bond order is irrelevant to ring membership
    stripped = em.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(stripped)
    new_to_old = {new: old for old, new in kept.items()}
    return [frozenset(new_to_old[i] for i in ring) for ring in stripped.GetRingInfo().AtomRings()]


def _trigonal_donors_qualify(rw, x, donors, metals, rings):
    """Return True when Class B's two extra conditions hold for a trigonal bridgehead's `donors`.

    Every donor must be a non-carbon heteroatom that itself has a lone pair (a formal-charge carbanion
    never qualifies -- what keeps a Cp, indenyl, or pyrrolyl ring intact, since ring aromaticity puts a
    delocalised charge on a ring carbon too). And X must not share a metal-free ring with a donor --
    what keeps a phosphole, thiazole, or imidazolyl face intact, where the flanking donors are joined
    to X by a real organic ring bond, not only by both separately reaching the same metal.
    """
    xi = x.GetIdx()
    for d in donors:
        atom = rw.GetAtomWithIdx(d)
        if atom.GetAtomicNum() in _TRIGONAL_NONDONOR_Z or _lone_pair(atom, metals) <= 0:
            return False
        if any(xi in ring and d in ring for ring in rings):
            return False
    return True


def _prune_donorless_bridgeheads(rw, metals):
    """Remove an M-X bond where X is a chelate bridgehead with no donor orbital of its own.

    A metal-aware reader can bond the metal to a chelate bridgehead X -- the P of a kappa2 S2PR2 or
    N-P-N/O-P-O ligand, the Si of S-Si-S, the B of kappa2-BH4 -- instead of, or besides, that bridgehead's
    real donor neighbours. X has no donor orbital, and the bond is graph-provably wrong, exactly when: two
    or more of X's own neighbours are themselves bonded to that same metal (the real donors) and are not
    bonded to each other; X carries four or more sigma bonds to non-metal atoms (Class A); and X has no
    lone pair (`_lone_pair`) -- not a per-element list, so it holds for any main-group bridgehead.

    Class B extends the same X-has-no-lone-pair test to a TRIGONAL bridgehead (exactly three sigma bonds
    to non-metal atoms: a carboxylate, amidinate, or dithiocarbamate C, or an N-B-N B). A geometric test
    cannot see this one -- xyzgraph 1.6.14's mis-bonded M-C sits at an ordinary M-C distance -- so it
    needs two further graph-only conditions (`_trigonal_donors_qualify`) before X loses its bond: every
    one of its metal-bound neighbours must itself be a non-carbon heteroatom with a lone pair, and X must
    not share a metal-free ring with one of them. A sigma-only macrocycle (each ring member independently
    donating its own lone pair, e.g. a cyclo-As6 crown) is excluded by X's own lone-pair test, since each
    of its members is a real donor in its own right, not a bridgehead.

    A sigma-silane or sigma-borane bridgehead (eta2-Si-H, B-H: one metal-bound neighbour) fails the first
    test and is left for the reader -- it is not graph-provable this way. Logs one warning naming every
    removed bond; the reader should never form one. This guard mirrors xyzgraph's own `_prune_crosslinks`
    and can be dropped once a fixed xyzgraph is installed.
    """
    rings = _metal_free_rings(rw, metals) if metals else []
    bad = []
    for metal in metals:
        for x in rw.GetAtomWithIdx(metal).GetNeighbors():
            xi = x.GetIdx()
            if xi in metals:
                continue
            donors = [
                n.GetIdx()
                for n in x.GetNeighbors()
                if n.GetIdx() != metal and rw.GetBondBetweenAtoms(n.GetIdx(), metal) is not None
            ]
            if len(donors) < _BRIDGEHEAD_DONORS_MIN or any(
                rw.GetBondBetweenAtoms(a, b) is not None for a, b in itertools.combinations(donors, 2)
            ):
                continue
            sigma_to_nonmetal = x.GetTotalDegree() - sum(1 for n in x.GetNeighbors() if n.GetIdx() in metals)
            trigonal = sigma_to_nonmetal == _TRIGONAL_SIGMA
            if sigma_to_nonmetal < _BRIDGEHEAD_SIGMA_MIN and not trigonal:
                continue
            if _lone_pair(x, metals) > 0:
                continue
            if trigonal and not _trigonal_donors_qualify(rw, x, donors, metals, rings):
                continue
            bad.append((xi, x.GetSymbol(), metal))
    for xi, _sym, metal in bad:
        remove_bond(rw, xi, metal)
    if bad:
        logger.warning(
            "metal graph: dropped bridgehead bond(s) %s (no lone pair, not a donor); omit it from the input",
            ", ".join(f"{sym}{xi}-{metal}" for xi, sym, metal in bad),
        )
    return bad


def _canonical_metal_graph(mol):
    """Return a copy with every M-L bond in the canonical ionic donor-to-metal form.

    Stated total charge, charge magnitude, and non-resonant charges remain authoritative. Only the atom carrying
    a delocalised aromatic -1 is canonicalized as a representation convention. Neutral underfilled sigma donors
    receive the integral charge implied by their ligand-side valence, balanced on the adjacent metal. A neutral
    bridge cannot say which metal owns that balance and therefore requires explicit charges.

    Every metal graph, however it entered (an XYZ read, a parsed SMILES, or rxembed's own ligand-bond
    restore after a swap), is canonicalized here before its donors are used, so a Class A bridgehead bond
    (see `_prune_donorless_bridgeheads`) is caught once at this one choke point rather than per entry route.
    """
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    metals = {atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    _prune_donorless_bridgeheads(rw, metals)  # a no-op when `metals` is empty
    rw.UpdatePropertyCache(strict=False)
    haptic = {
        donor
        for metal in metals
        for site in _haptic_sites(
            rw,
            [
                neighbor.GetIdx()
                for neighbor in rw.GetAtomWithIdx(metal).GetNeighbors()
                if neighbor.GetIdx() not in metals
            ],
        )
        if len(site) > 1
        for donor in site
    }
    charged = {}
    for donor in rw.GetAtoms():
        adjacent = [neighbor.GetIdx() for neighbor in donor.GetNeighbors() if neighbor.GetIdx() in metals]
        if not adjacent:
            continue
        charge = _ionic_donor_charge(donor, haptic)
        if len(adjacent) > 1 and charge:
            raise ValueError(
                f"neutral bridging donor {donor.GetSymbol()}{donor.GetIdx()} has ambiguous charge allocation "
                f"between metals {adjacent}; state the donor and metal formal charges explicitly"
            )
        if charge:
            charged[donor.GetIdx()] = (adjacent[0], charge)

    replace = []
    for bond in rw.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in metals and end not in metals:
            donor, metal = end, begin
        elif end in metals and begin not in metals:
            donor, metal = begin, end
        else:
            continue
        if bond.GetBondType() == Chem.BondType.DATIVE and begin == donor:
            continue
        note = bond.GetProp(SLOT_BOND_PROP) if bond.HasProp(SLOT_BOND_PROP) else None
        replace.append((donor, metal, note))

    for donor, (metal, charge) in charged.items():
        rw.GetAtomWithIdx(donor).SetFormalCharge(charge)
        atom = rw.GetAtomWithIdx(metal)
        atom.SetFormalCharge(atom.GetFormalCharge() - charge)
    for donor, metal, note in replace:
        remove_bond(rw, donor, metal)
        rw.AddBond(donor, metal, Chem.BondType.DATIVE)
        if note is not None:
            rw.GetBondBetweenAtoms(donor, metal).SetProp(SLOT_BOND_PROP, note)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    if replace:
        # A covalent R2PH-M bond can make RDKit cache one P radical. The equivalent dative bond does not spend
        # donor valence, so recompute only that property and only on donors whose notation changed. Radicals
        # explicitly supplied on already-dative input remain authoritative.
        probe = Chem.Mol(out)
        result = Chem.SanitizeMol(probe, Chem.SanitizeFlags.SANITIZE_FINDRADICALS, catchErrors=True)
        if result == Chem.SanitizeFlags.SANITIZE_NONE:
            for donor in {donor for donor, _metal, _note in replace}:
                out.GetAtomWithIdx(donor).SetNumRadicalElectrons(probe.GetAtomWithIdx(donor).GetNumRadicalElectrons())
    return _canonicalise_delocalised_charge(out)


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
    if sorted(haptic) != list(range(base_n, base_n + len(haptic))):
        raise ValueError(f"haptic centroid indices {sorted(haptic)} are not consecutive after {base_n} real atoms")
    rw = Chem.RWMol(mol)
    for _dummy in sorted(haptic):
        a = rw.GetAtomWithIdx(rw.AddAtom(Chem.Atom(SURROGATE)))
        a.SetNoImplicit(True)  # a point, not a valence
        a.SetHybridization(Chem.HybridizationType.SP3)  # match the sanitised metal surrogate; otherwise UFF and
        #   the bounds matrix emit 'unrecognized hybridization', UpdatePropertyCache leaving it UNSPECIFIED
    out = rw.GetMol()
    # AddAtom retains cached path matrices with the old atom count, corrupting subsequent native DG bounds.
    out.ClearComputedProps()
    out.UpdatePropertyCache(strict=False)  # GetMoleculeBoundsMatrix / UFF need the new bond-less atoms' valence
    # FastFindRings guarantees membership, not the ring sizes consumed by native DG. Retain the temporary
    # donor-stereo dative cycles while rebuilding a symmetric small-ring basis after helper edits.
    Chem.GetSymmSSSR(out, includeDativeBonds=True)
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
    Chem.GetSymmSSSR(out, includeDativeBonds=True)  # RemoveAtom invalidates the native ring-size cache.
    return out


def _bounds_matrix(mol, params=None, *, set14bounds=True):
    """Build RDKit bounds without leaking its internal UFF-typing diagnostics for likely noisy graphs."""
    noisy = any(atom.GetFormalCharge() or atom.GetAtomicNum() in COORDINATION_METALS for atom in mol.GetAtoms())
    options = {} if params is None else {"embedParams": params}
    if noisy:
        with rdBase.BlockLogs():
            return rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=set14bounds, **options)
    return rdDistGeom.GetMoleculeBoundsMatrix(mol, set14bounds=set14bounds, **options)


def _site_radius(mol, site, *, positions=None):
    """Return the root-mean-square distance of haptic members from their centroid.

    The identity ``R^2 = sum(i<j, d_ij^2)/n^2`` holds for any point set, including irregular faces.
    Without explicit positions, RDKit's pair-bound midpoints estimate those distances. They need not jointly
    describe a realizable point set. The caller owns length provenance; a conformer on `mol` is not authority
    to measure it. A two-atom eta2 site reduces to half the edge.
    """
    if positions is not None:
        pos = np.asarray(positions)[list(site)]
        return float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, axis=1))))

    bm = _bounds_matrix(mol)
    tot = sum(
        (0.5 * (bm[max(a, b)][min(a, b)] + bm[min(a, b)][max(a, b)])) ** 2 for a, b in itertools.combinations(site, 2)
    )
    return float(np.sqrt(tot)) / len(site)


def _site_height(radius, member_lengths):
    """Estimate centroid distance using ``|M-c|^2 = mean(|M-member|^2) - R^2``.

    Retain the existing 0.5 A scaffold floor; it is not a feasibility proof for fitted member distances.
    """
    mean_squared_length = float(np.mean(np.square(member_lengths)))
    return float(np.sqrt(max(mean_squared_length - radius * radius, 0.25)))


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


def _reject_boron_cages(mol):
    """Reject connected five-boron cages outside the two-centre donor model."""
    metals = set(metal_indices(mol))
    seen, cages = set(), []
    for atom in mol.GetAtoms():
        start = atom.GetIdx()
        if start in metals or start in seen:
            continue
        stack, component = [start], []
        while stack:
            index = stack.pop()
            if index in seen or index in metals:
                continue
            seen.add(index)
            component.append(index)
            stack.extend(
                neighbor.GetIdx()
                for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
                if neighbor.GetIdx() not in metals
            )
        borons = [index for index in component if mol.GetAtomWithIdx(index).GetAtomicNum() == _BORON_Z]
        if len(borons) >= _BORON_CAGE_MIN:
            cages.append((tuple(sorted(borons)), tuple(sorted(component))))
    if cages:
        details = ", ".join(
            f"{len(borons)} boron atoms ({component[0]}..{component[-1]})" for borons, component in cages
        )
        raise ValueError(
            f"boron cage ligand(s) detected: {details}; multi-centre B-H/B-B bonding is outside rxembed's "
            "two-centre donor model; supply an explicit donor graph or use a cage-capable backend"
        )


def surrogate_all_metals(mol):
    """Surrogate every metal centre (bonds removed, carbon) for a multi-metal complex.

    UFF must type the whole complex of a bimetallic TS, and `surrogate_metal` only does the first metal.
    Returns ``(mol, metals)`` where ``metals`` is ``[(idx, real_z, real_q), ...]``, the element and formal
    charge `restore_metal` needs. Sanitised leniently, since a stripped η⁵-Cp is a radical fragment.
    """
    mol = _canonical_metal_graph(mol)
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
            a.SetNumExplicitHs(a.GetTotalNumHs())
            a.SetNoImplicit(True)
            hands[d] = a.GetChiralTag()
        a = em.GetAtomWithIdx(m)
        a.SetAtomicNum(SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # bondless surrogate: a stray metal tag crashes ETKDG
    out = em.GetMol()
    Chem.SanitizeMol(out, _SURROGATE_SANITIZE, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    repair_bond_stereo(out)
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


def classify_geometry(mol, metal, sites, cid=-1, *, warn=True):
    """Name the coordination polytope by flatness exclusion and best orthogonal vertex fit.

    ``sites`` are coordination sites, not atoms: a haptic face is one vertex via its centroid. Returns
    ``None`` only when no record has that vertex count, and warns above `_FIT_FLOOR`, where the name is the
    nearest record rather than a reading of the sphere. Set ``warn=False`` for repeated internal validation.

    Flatness excludes one way only: a flat sphere cannot be a record whose metal sits off its vertex plane,
    but the converse says nothing, since an out-of-plane sphere is a distorted planar shape as readily as a
    3-D one. This is what separates `trigonal_planar` from the CN3 pyramid, where an angle boundary would
    have to be fitted and this has a natural zero.

    The fit preserves vertex correspondence while allowing rotation, reflection and donor reordering.
    `metal_polyhedron.fit_residual` owns the bounded-exact seating search and its high-CN approximation.
    """
    pos = mol.GetConformer(cid).GetPositions()
    points = []
    for site in sites:
        atoms = [site] if isinstance(site, (int, np.integer)) else list(site)
        points.append(np.mean([pos[a] for a in atoms], axis=0))
    if not points:
        return None
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
    if poor and warn:
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
