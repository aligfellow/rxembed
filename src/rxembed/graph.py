"""Read the graph facts every layer shares: which atoms are metals and how ligands bond around them."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from rxembed.utils import CARBON_Z, SP2_DEGREE, flat_ranks, remove_bond

# Any coordination centre, f-block included: the one set every check that asks "is this a metal" reads.
COORDINATION_METALS = (
    frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
)
SURROGATE = 6  # carbon: its excluded volume stops a ligand folding into the metal, so the distance geometry keeps it
_PT = Chem.GetPeriodicTable()
SURROGATE_SANITIZE = (
    Chem.SanitizeFlags.SANITIZE_ALL
    ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
)
_MIN_STEREO_NEIGHBOURS = 3  # a tetrahedral stereocentre needs >=3 neighbours, implicit H included (else ETKDG raises)


def metal_indices(mol):
    """Return the indices of every metal coordination centre (`COORDINATION_METALS`), in atom order."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]


def metal_index(mol):
    """Return the index of the first metal coordination centre, or None."""
    return next(iter(metal_indices(mol)), None)


def bound_metals(mol):
    """Return the metals of canonical metal graph `mol` that bind a donor."""
    metals = metal_indices(mol)
    return [m for m in metals if any(n.GetIdx() not in metals for n in mol.GetAtomWithIdx(m).GetNeighbors())]


def hydrogen_legs(mol, metals):
    """Map each hydrogen bonded to two or more atoms to its neighbours, its ligand leg first.

    A nonmetal comes before one of `metals`, then the nearer atom when `mol` has a conformer, then the lower
    constitution rank, so a B-H-M or C-H-M bridge keeps its ligand bond first and a true tie reads the same every time.
    """
    hydrogens = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetDegree() > 1]
    if not hydrogens:
        return {}
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    ranks = flat_ranks(mol)
    legs = {}
    for hydrogen in hydrogens:
        h = hydrogen.GetIdx()
        keyed = sorted(
            (
                n.GetAtomicNum() in metals,
                0.0 if pos is None else float(np.sum((pos[n.GetIdx()] - pos[h]) ** 2)),
                ranks[n.GetIdx()],
                n.GetIdx(),
            )
            for n in hydrogen.GetNeighbors()
        )
        legs[h] = [neighbor for *_key, neighbor in keyed]
    return legs


def lone_pair_electrons(atom, metals):
    """Return `atom`'s nonbonding valence: outer electrons minus formal charge minus bonded valence.

    Bonds to `metals` count as dative, so dative and covalent input read alike; two or more means a lone pair to donate.
    """
    to_metal = sum(
        bond.GetValenceContrib(atom) for bond in atom.GetBonds() if bond.GetOtherAtomIdx(atom.GetIdx()) in metals
    )
    return _PT.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() - (atom.GetTotalValence() - to_metal)


def hydrogen_bond(hydrogen, partner, metals) -> bool:
    """Return whether a hydrogen's extra contact to `partner` is a hydrogen bond rather than a bond.

    A partner that keeps a lone pair (X-H...Y) is an NCI, not constitution; one without (B-H-B) shares the pair in a
    three-centre bond, and a metal partner makes a bridging hydride.
    """
    return (
        hydrogen.GetAtomicNum() == 1 and partner.GetIdx() not in metals and lone_pair_electrons(partner, metals) >= 2  # noqa: PLR2004  one lone pair
    )


def ligand_graph(mol, metals=None):
    """Return a copy of `mol` without the bonds of `metals` (default every metal), atom indices unchanged.

    Every strip goes through here, keeping each tetrahedral and E/Z tag's meaning. A zero-order hydrogen bond goes too;
    a B-H-B contact stays.
    """
    rw = Chem.RWMol(mol)
    zero = [bond for bond in rw.GetBonds() if bond.GetBondType() == Chem.BondType.ZERO]
    present = set(metal_indices(mol)) if metals is None or zero else set()
    metals = present if metals is None else metals
    if zero:
        rw.UpdatePropertyCache(strict=False)  # lone-pair counts read valence
    contacts = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in zero
        if hydrogen_bond(bond.GetBeginAtom(), bond.GetEndAtom(), present)
        or hydrogen_bond(bond.GetEndAtom(), bond.GetBeginAtom(), present)
    ]
    for m in metals:
        for nb in [n.GetIdx() for n in rw.GetAtomWithIdx(int(m)).GetNeighbors()]:
            remove_bond(rw, int(m), nb)
    for i, j in contacts:
        if rw.GetBondBetweenAtoms(i, j) is not None:
            remove_bond(rw, i, j)
    out = rw.GetMol()
    out.ClearComputedProps()
    return out


def frag_map(mol):
    """Map each atom index -> its fragment id (same ligand = same fragment); a hydrogen bond joins nothing."""
    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(ligand_graph(mol, ()))) for a in f}


def ligand_degree(atom):
    """Count ligand-side neighbours, including implicit and explicit hydrogen."""
    return atom.GetTotalDegree() - sum(nb.GetAtomicNum() in COORDINATION_METALS for nb in atom.GetNeighbors())


def open_valence(atom):
    """Return whether the graph leaves `atom`'s valence open: a pi bond, charge or radical it does not write.

    A bonds-only graph writes every bond single, so such an atom shows as fewer bonds than its valence, or an odd count
    of nonbonding electrons.
    """
    ligand = [bond for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() not in COORDINATION_METALS]
    held = len(ligand) + atom.GetNumExplicitHs()
    return bool(
        atom.GetNoImplicit()
        and not atom.GetFormalCharge()
        and not atom.GetNumRadicalElectrons()
        and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in ligand)
        and (held < _PT.GetDefaultValence(atom.GetAtomicNum()) or (_PT.GetNOuterElecs(atom.GetAtomicNum()) - held) % 2)
    )


def no_aromatic_axis(atom):
    """Return whether an aromatic donor lacks an in-plane sigma direction.

    An sp2 framework with three ligand neighbours has no in-plane position left; an aromatic heteroatom's conjugating
    pair creates none.
    """
    if not atom.GetIsAromatic() or ligand_degree(atom) < SP2_DEGREE:
        return False
    metals = {nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() in COORDINATION_METALS}
    return atom.GetAtomicNum() != CARBON_Z or lone_pair_electrons(atom, metals) < 2  # noqa: PLR2004  one pair


def disconnect_metal(mol):
    """Return a copy of `mol` without its metal dative bonds, or `mol` itself when it has none.

    A covalent M-X bond is a real backbone path and stays; `ligand_graph` strips both kinds.
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


ETA2 = 2  # atoms in the smallest haptic face, an eta2 bond
_PI_HYBRIDISATION = {Chem.HybridizationType.UNSPECIFIED, Chem.HybridizationType.SP, Chem.HybridizationType.SP2}


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


def _pi_face_donors(mol, dset):
    """Return the donors of `dset` on a pi face: the seeds `haptic_sites` names, then their charged or radical ends."""
    unpaired = {d for d in dset if open_valence(atom := mol.GetAtomWithIdx(d)) or atom.GetNumRadicalElectrons()}
    pi = set()
    for d in unpaired:  # the middle of a chain or ring of unpaired donors seeds it with its unpaired neighbours
        if len(chained := {n.GetIdx() for n in mol.GetAtomWithIdx(d).GetNeighbors()} & unpaired) > 1:
            pi.update(chained, (d,))
    for ring in mol.GetRingInfo().AtomRings():
        if set(ring) <= dset and all(mol.GetAtomWithIdx(atom).GetHybridization() in _PI_HYBRIDISATION for atom in ring):
            pi.update(ring)
    for bond in mol.GetBonds():
        ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        conjugated = bond.GetBondTypeAsDouble() > 1 and all(
            atom.GetHybridization() in _PI_HYBRIDISATION for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
        if ends <= dset and (bond.GetIsAromatic() or conjugated):
            pi.update(ends)
    while extended := {
        d
        for d in dset - pi
        if (atom := mol.GetAtomWithIdx(d)).GetFormalCharge() or atom.GetNumRadicalElectrons()
        if any(neighbor.GetIdx() in pi for neighbor in atom.GetNeighbors())
    }:
        pi.update(extended)
    return pi


def haptic_sites(mol, donors, *, pairs=True):
    """Group `donors` into sorted site tuples: each sigma donor alone, each connected pi face as one.

    A face starts at a donor-donor multiple or aromatic bond, or at three or more bonded donors with an unpaired
    electron, and spreads across adjacent charged or radical donors, so a diene, allyl, Cp or arene is one site. With
    ``pairs=True`` a bonded donor pair with no third donor neighbour is one site whatever its order (a metallaoxirane's
    C-O), while cyclo-As6 stays one site per atom. ``pairs=False`` joins only an isolated diatomic, for callers whose
    bond orders and charges grouping must not change.
    """
    dset = set(donors)
    pi = _pi_face_donors(mol, dset)
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


_DIENE_FACE = 4  # atoms in an eta4 diene face: two bonds either side of the central bond
_RIGID_RING = 8  # RDKit's FindPotentialStereo ring size: a smaller ring fixes a bond's configuration


def torsion_path(mol, face):
    """Return a face's atoms in path order when it is an open four-atom chain about a rotatable bond, else None.

    Such a face (an eta4 diene) is s-cis or s-trans once bound. The central bond rotates unless a ligand ring under
    eight atoms holds it; the metal never closes that ring.
    """
    if len(face) != _DIENE_FACE:
        return None
    inside = set(face)
    neighbours = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in inside] for a in face}
    if sorted(map(len, neighbours.values())) != [1, 1, 2, 2]:  # the only four-atom graph with these degrees is a path
        return None
    path = [min(a for a in face if len(neighbours[a]) == 1)]
    while len(path) < _DIENE_FACE:
        path.append(next(a for a in neighbours[path[-1]] if a not in path))
    ligand = ligand_graph(mol)
    Chem.GetSymmSSSR(ligand)
    size = ligand.GetRingInfo().MinBondRingSize(ligand.GetBondBetweenAtoms(path[1], path[2]).GetIdx())
    return None if 0 < size < _RIGID_RING else path


def materialise_phantoms(mol, haptic):
    """Return a copy of `mol` with a bondless carbon dummy at each `haptic` face's centroid, or `mol` when none.

    Each dummy takes its reserved index after the real atoms; no stored Mol holds one.
    """
    if not haptic:
        return mol
    rw = Chem.RWMol(mol)
    for _dummy in sorted(haptic):
        a = rw.GetAtomWithIdx(rw.AddAtom(Chem.Atom(SURROGATE)))
        a.SetNoImplicit(True)  # a point, not a valence
        a.SetHybridization(Chem.HybridizationType.SP3)  # as the metal surrogate; UNSPECIFIED warns in UFF and DG bounds
    out = rw.GetMol()
    # AddAtom retains cached path matrices with the old atom count, corrupting subsequent native DG bounds.
    out.ClearComputedProps()
    out.UpdatePropertyCache(strict=False)  # GetMoleculeBoundsMatrix / UFF need the new bond-less atoms' valence
    # Native DG reads ring sizes, which FastFindRings does not give; dative rings count, for donor stereo.
    Chem.GetSymmSSSR(out, includeDativeBonds=True)
    for conf in out.GetConformers():
        for dummy, ring in haptic.items():
            conf.SetAtomPosition(dummy, Point3D(*np.mean([list(conf.GetAtomPosition(a)) for a in ring], axis=0)))
    return out


def strip_phantoms(mol, phantoms):
    """Return a copy of `mol` without the centroid dummies `phantoms`, or `mol` when none; real indices unchanged."""
    if not phantoms:
        return mol
    rw = Chem.RWMol(mol)
    for p in sorted(phantoms, reverse=True):  # remove from the end so the real (lower) indices stay put
        rw.RemoveAtom(int(p))
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(out, includeDativeBonds=True)  # RemoveAtom invalidates the native ring-size cache.
    return out


def clear_labile_donor_stereo(atom):
    """Drop the chiral tag of a donor that cannot be tetrahedral once its metal is stripped.

    A donor that is a stereocentre only while bound (a planar amidate N-) or aromatic keeps a stale tag that crashes
    ETKDG; a real centre (chiral P, a carbanion) is untouched.
    """
    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and (
        atom.GetIsAromatic() or atom.GetTotalDegree() < _MIN_STEREO_NEIGHBOURS
    ):
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)


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


def surrogate_all_metals(mol):
    """Return a copy of a canonical metal graph with every metal a bondless neutral carbon, its surrogate graph.

    Each donor keeps its hydrogen count and any hand it can hold. The sanitise is lenient, since a stripped Cp is a
    radical fragment, and a metal-metal bond is refused.
    """
    idxs = metal_indices(mol)
    reject_metal_bonds(mol)
    em = Chem.RWMol(mol)
    hands = {}
    for m in idxs:
        for d in [n.GetIdx() for n in em.GetAtomWithIdx(m).GetNeighbors()]:
            remove_bond(em, d, m)  # a bridging donor's two strips compose here
            a = em.GetAtomWithIdx(d)
            clear_labile_donor_stereo(a)  # stale tag on a now-<3-nbr donor crashes ETKDG
            a.SetNumExplicitHs(a.GetTotalNumHs())
            a.SetNoImplicit(True)
            hands[d] = a.GetChiralTag()
        a = em.GetAtomWithIdx(m)
        a.SetAtomicNum(SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)  # bondless surrogate: a stray metal tag crashes ETKDG
    out = em.GetMol()
    Chem.SanitizeMol(out, SURROGATE_SANITIZE, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    for donor, tag in hands.items():  # the specified hand: the lenient sanitize drops one off a donor typed non-SP3
        out.GetAtomWithIdx(donor).SetChiralTag(tag)
    return out
