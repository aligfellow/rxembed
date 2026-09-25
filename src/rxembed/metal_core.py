"""Metal-complex coordination primitives: the metal state records, the surrogate and the ligand graph.

The metal is held purely by distance and angle constraints, its bonds removed, and embedded or relaxed with a
UFF-typeable surrogate atom in its place, so the whole path is plain RDKit and UFF with no xtb.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

from .metal_polyhedron import SLOT_BOND_PROP
from .utils import bond_removal_mirrors, flat_ranks, remove_bond, repair_bond_stereo

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__ ("rxembed.metal_core"): this is
#   the name `set_verbose` configures and every caplog filter in the suite matches.

# Any coordination centre, f-block included, because M-L bonds are dative wherever ligands coordinate. Every
# gate that asks whether an atom is a metal reads this one set, so no two gates can disagree about a centre.
COORDINATION_METALS = (
    frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
)
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
EPS_LEN = 1e-9  # a donor sitting on the metal has no direction, so the sphere cannot be read at all


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
    return materialized_states(iso.graph, centres)[state.atom]


def state_with_winding(state, vertices, winding):
    """Return a state carrying `winding` on its materialized haptic vertices."""
    sites = tuple(
        HapticSite(site.atoms, winding.get(vertices[position], "")) if isinstance(site, HapticSite) else site
        for position, site in enumerate(state.vertices)
    )
    return state._replace(vertices=sites)


def frag_map(mol):
    """Map each atom index -> its fragment id (same ligand = same fragment)."""
    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}


def vertex_atom(haptic, v):
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


HAND_TAG = {  # `donor_chirality_sign` -> the tag naming that hand in the atom's current bond order
    -1: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    +1: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}


def _retag(mol, hands, ambiguous):
    """Re-apply each stripped donor's tetrahedral tag, in the bond order the strip left behind.

    `ambiguous` names the donors whose carried symbol does not determine a hand; there the geometry decides.

    Runs last: the lenient sanitize drops a tag off any donor it types non-SP3, and
    `repair_bond_stereo`'s 3D stereo assignment wipes one it then refuses to re-derive below degree 4.

    `remove_bond` already re-based every tag in `hands` when it took the M-L bond out, wherever the incoming
    basis was known. A DATIVE M-L bond at an odd slot is the one case it is not: RDKit's 3D writer leaves
    that bond out of the basis while its SMILES parser counts it, so the two bases name opposite hands and
    nothing in the graph says which was meant. Only a conformer can settle it, so an ambiguous donor is
    resolved here, from the geometry, rather than guessed for every donor.
    """
    for d, carried in hands.items():
        atom = mol.GetAtomWithIdx(d)
        if carried == Chem.ChiralType.CHI_UNSPECIFIED or atom.GetDegree() < _MIN_STEREO_NEIGHBOURS:
            continue
        decided = carried
        if d in ambiguous and mol.GetNumConformers():
            decided = HAND_TAG.get(donor_chirality_sign(mol, -1, d), carried)
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
    mol = canonical_metal_graph(mol)
    m = metal_index(mol)
    if m is None:
        raise ValueError("no metal centre found")
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
    em = Chem.RWMol(mol)
    hands = {}  # donor -> the tag it must carry in the bond order the strip leaves behind
    ambiguous = {d for d in donors if _basis_is_ambiguous(em, d, m)}  # read before the bond goes
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
    # the dative bonds close chelate rings through the metal, so re-perceive rings or the dedup SMARTS and the
    # geometry gate read the wrong ones
    Chem.FastFindRings(out)
    return out


def disconnect_metal(mol):
    """Remove metal-donor dative bonds from the working graph; return the input when there are none.

    Constraints own the M-L geometry during DG and UFF, independently of native bonded-metal terms.
    Public coordination bonds are restored after relaxation by `connect_metal`.

    Only dative bonds go: a covalent M-X bond is a real backbone path, so DG topology (`DGContext.topo`)
    keeps it. `ligand_graph` strips every metal bond, dative or covalent, because it needs ligands read as
    separate fragments regardless of bond type.
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


def ligand_graph(mol, metals=None):
    """Return the metal-stripped ligand graph: index-stable, so callers reuse `mol`'s atom indices.

    The one strip every consumer routes through: `remove_bond` mirrors a chiral tag when removing a bond
    flips its carrier's parity, so a stereo-reading consumer and a topology-only one see the same graph.
    ``metals`` defaults to every coordination-metal atom in ``mol``; pass an explicit subset to strip only
    those centres.
    """
    metals = (
        {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS} if metals is None else metals
    )
    rw = Chem.RWMol(mol)
    for m in metals:
        for nb in [n.GetIdx() for n in rw.GetAtomWithIdx(int(m)).GetNeighbors()]:
            remove_bond(rw, int(m), nb)
    out = rw.GetMol()
    out.ClearComputedProps()
    return out


def ligand_distance_matrix(mol):
    """Return graph distances after removing coordination-centre edges."""
    if not metal_indices(mol):
        return Chem.GetDistanceMatrix(mol)
    return Chem.GetDistanceMatrix(ligand_graph(mol))


ETA2 = 2  # atoms in the smallest haptic face, an eta2 bond


def _face_joined(mol, a, b, pi, dset, pairs):
    """Return whether two bonded donors share one site: both on a pi face, or an isolated pair (see `haptic_sites`)."""
    if a in pi and b in pi:
        return True
    if pairs:
        return all(sum(nb.GetIdx() in dset for nb in mol.GetAtomWithIdx(i).GetNeighbors()) == 1 for i in (a, b))
    return all(
        mol.GetAtomWithIdx(i).GetTotalNumHs() == 0
        and all(
            neighbor.GetIdx() in {a, b} or neighbor.GetAtomicNum() in COORDINATION_METALS
            for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
        )
        for i in (a, b)
    )


def haptic_sites(mol, donors, *, pairs=True):
    """Group `donors` into sigma sites and connected pi faces.

    A lone sigma donor is its own 1-tuple. A face starts at a donor-donor multiple or aromatic bond and
    extends across adjacent charged or radical donor endpoints; bonds between face atoms then join a diene,
    allyl, Cp, or arene into one site. A bonded pair of donors with no third donor neighbour on either end is
    also one site regardless of its perceived Lewis bond order (`pairs=True`, the default): a metallaoxirane's
    C-O, a kappa2-hydrazide's N-N. A ring of three or more donors keeps the pi-seeded rule instead, so a
    sigma-only macrocycle stays one sigma site per atom. `pairs=False` keeps only the isolated-diatomic clause
    (no hydrogens, no other ligand neighbour), for the two Lewis-bookkeeping callers (`canonical_metal_graph`'s
    donor-charge count, `xyz2mol_tmc.lig_checks`'s pairless-sigma-donor count) that must not let site grouping
    change a reader's bond-order or charge decision.
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
                if nb.GetIdx() in dset and _face_joined(mol, a, nb.GetIdx(), pi, dset, pairs)
            ]
        sites.append(tuple(sorted(group)))
    return sorted(sites)


def collapse_haptic(mol, donors):
    """Collapse each haptic face to one centroid vertex; sigma donors pass through.

    Appends a bond-less carbon centroid per face, leaving existing indices unchanged, seated at the ring
    centroid if the mol has a conformer. Returns ``(mol, vertices, haptic)``, where `haptic` maps each dummy to
    its ring atoms. The dummy is embed scaffolding that lives in no stored Mol: `enumerate_isomers` strips it
    before storing the real `Isomer`, and only `bounds.seed_coordinates` / `restrained_uff` re-materialise it.
    A mol with no haptic face is returned untouched.
    """
    sites = haptic_sites(mol, donors)
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
    """Canonicalize a delocalised aromatic anion's charge onto its highest-ranked metal-bound ring atom.

    A resonance move redraws bond orders and charge within one conjugated system; it never moves a hydrogen.
    Freezing that system's atoms' H counts and retrying `SanitizeMol`'s Kekulization is therefore an exact,
    uncapped proof: it succeeds only when some alternating bond-order pattern seats the charge at the
    candidate with every atom's H count unchanged, which is what a resonance form is. No search over the
    rest of the molecule is needed, since Kekulization is a polynomial matching, not a form-by-form
    enumeration. The commit must reuse the validated, still-frozen molecule the proof produced: copying only
    the two charges onto an unfrozen working copy lets `SanitizeMol` recompute implicit H on its own and
    silently pick a different, sometimes unkekulizable count instead of the pattern the proof found.
    """
    rw = Chem.RWMol(mol)
    metals = {atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    if not metals:
        return rw.GetMol()
    # A move can expose another delocalised anion once the cached aromatic representation changes. Iterate
    # to a fixed point so repeated graph normalization is itself a normal form.
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
            # change the string again on the next write, even when every individual move is independently valid.
            if len(charged) != 1 or rw.GetAtomWithIdx(charged[0]).GetFormalCharge() != -1:
                continue
            current = charged[0]
            candidates = [atom for atom in component if atom in bound and rw.GetAtomWithIdx(atom).GetIsAromatic()]
            for candidate in sorted(candidates, key=lambda atom: (-ranks[atom], atom)):
                if candidate == current:
                    break
                trial = Chem.RWMol(rw)
                for atom in component:  # freeze H on this system only: a resonance move never changes one
                    a = trial.GetAtomWithIdx(atom)
                    a.SetNumExplicitHs(a.GetTotalNumHs())
                    a.SetNoImplicit(True)
                trial.GetAtomWithIdx(current).SetFormalCharge(0)
                trial.GetAtomWithIdx(candidate).SetFormalCharge(-1)
                probe = trial.GetMol()
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(probe)
                except Chem.MolSanitizeException:
                    continue
                rw = Chem.RWMol(probe)  # adopt the validated mol itself, not a re-derivation of its charges
                moved = True
                break
        if not moved:
            break
    return rw.GetMol()


def canonical_metal_graph(mol):
    """Return a copy with every M-L bond in the canonical ionic donor-to-metal form.

    Stated total charge, charge magnitude and non-resonant charges stay authoritative. The only charge this
    moves is a delocalised aromatic -1, canonicalized onto one ring atom as a representation convention.
    A neutral underfilled sigma donor gets the integral charge its ligand-side valence implies, balanced on
    the adjacent metal; a neutral bridge cannot say which metal owns that balance, so it needs an explicit
    charge instead. Every charge this rewrites is logged at debug level with the atom and its old and new
    value.

    Runs before any donor is read, on every metal graph however it arrived (an XYZ read, a parsed SMILES, or
    rxembed's own bond restore after a swap). A bridgehead M-X bond with no donor orbital of its own is a
    separate, connectivity-only fault the XYZ reader fixes first; see
    `pipeline.perceive._prune_donorless_bridgeheads`.
    """
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    metals = {atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in COORDINATION_METALS}
    haptic = {
        donor
        for metal in metals
        for site in haptic_sites(
            rw,
            [
                neighbor.GetIdx()
                for neighbor in rw.GetAtomWithIdx(metal).GetNeighbors()
                if neighbor.GetIdx() not in metals
            ],
            pairs=False,  # Lewis bookkeeping (donor charge balancing): read only the pi-seeded face rule.
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
        donor_atom, metal_atom = rw.GetAtomWithIdx(donor), rw.GetAtomWithIdx(metal)
        old_donor, old_metal = donor_atom.GetFormalCharge(), metal_atom.GetFormalCharge()
        donor_atom.SetFormalCharge(charge)
        metal_atom.SetFormalCharge(old_metal - charge)
        logger.debug(
            "metal graph: charge %s%d %+d->%+d, %s%d %+d->%+d",
            donor_atom.GetSymbol(),
            donor,
            old_donor,
            charge,
            metal_atom.GetSymbol(),
            metal,
            old_metal,
            old_metal - charge,
        )
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


def reject_metal_bonds(mol):
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


def reject_boron_cages(mol):
    """Reject a period-2 atom bonded past its octet, the multi-centre cage case.

    A period-2 atom with fewer than four valence electrons (Li, Be, B) fills its octet in four two-centre
    bonds; a fifth or sixth sigma bond only exists through multi-centre bonding (a closo-borane cage, for
    example), which is outside rxembed's two-centre donor model.
    """
    mol.UpdatePropertyCache(strict=False)  # ligand_degree needs implicit valence; callers may hand a raw graph
    vertices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if _PT.GetRow(atom.GetAtomicNum()) == 2  # noqa: PLR2004  period 2: Li through Ne
        and _PT.GetNOuterElecs(atom.GetAtomicNum()) < 4  # noqa: PLR2004  fewer than carbon's four
        and ligand_degree(atom) > 4  # noqa: PLR2004  more sigma bonds than an octet allows
    ]
    if vertices:
        raise ValueError(
            f"multi-centre cage vertex atom(s) detected: {sorted(vertices)}; a period-2 atom with more than "
            "four two-centre bonds is outside rxembed's two-centre donor model; supply an explicit donor "
            "graph or use a cage-capable backend"
        )


def surrogate_all_metals(mol):
    """Surrogate every metal centre (bonds removed, carbon) for a multi-metal complex.

    UFF must type the whole complex of a bimetallic TS, and `surrogate_metal` only does the first metal.
    Returns ``(mol, metals)`` where ``metals`` is ``[(idx, real_z, real_q), ...]``, the element and formal
    charge `restore_metal` needs. Sanitised leniently, since a stripped η⁵-Cp is a radical fragment.
    """
    mol = canonical_metal_graph(mol)
    idxs = metal_indices(mol)
    if not idxs:
        raise ValueError("no metal centre found")
    reject_metal_bonds(mol)
    em = Chem.RWMol(mol)
    metals, hands, ambiguous = [], {}, set()
    for m in idxs:
        metals.append((m, em.GetAtomWithIdx(m).GetAtomicNum(), em.GetAtomWithIdx(m).GetFormalCharge()))
        for d in [n.GetIdx() for n in em.GetAtomWithIdx(m).GetNeighbors()]:
            if _basis_is_ambiguous(em, d, m):  # read before the bond goes, as in `surrogate_metal`
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
