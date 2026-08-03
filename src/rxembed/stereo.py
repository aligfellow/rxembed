"""Ligand-stereo load-in: expand a `Mol`'s UNDEFINED stereocentres into the distinct species to embed.

The embed-side half of stereochemistry: graph-only (RDKit's `EnumerateStereoisomers` over the
unspecified elements), metal-safe (the centre's handedness is the coordination-isomer path's job). The
conformer-side half, the coordinate-derived chirality fingerprint that filters embedded conformers,
is perception-driven and stays with the consumer (`rxembed.stereo`).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from .utils import bond_removal_mirrors, mirror_tag, remove_bond, repair_bond_stereo


def _stereo_label(mol, atom_centers, bond_centers, cap_to_metal=None):
    """Build a readable, index-keyed configuration tag over the enumerated centres, e.g. ``'1S'`` or ``'1R,3S,5=6:E'``.

    CIP R/S where RDKit assigns it (falls back to the raw CW/CCW tag for a centre it won't CIP-rank, e.g. some
    P), plus E/Z for each enumerated double bond. Keyed only on the *enumerated* atoms/bonds so distinct
    variants always get distinct, stable labels. ``cap_to_metal`` maps each donor's D-cap atom index to its
    metal's atomic number: the CIP is then computed with the METAL (highest priority) in the cap position, not
    the D (lowest), so a metal-bound donor's R/S names the coordinated centre correctly (the D->M priority
    flip is not a fixed R<->S swap; it depends on the donor's other substituents, e.g. whether it carries an H).
    """
    if cap_to_metal:  # temporarily give each D-cap the metal's atomic number for a coordinated-complex CIP
        rw = Chem.RWMol(mol)
        for d_idx, z in cap_to_metal.items():
            rw.GetAtomWithIdx(d_idx).SetAtomicNum(z)
            rw.GetAtomWithIdx(d_idx).SetIsotope(0)
            for nb in rw.GetAtomWithIdx(d_idx).GetNeighbors():  # neutralise the donor so metal+donor isn't hypervalent
                nb.SetFormalCharge(0)  # (an anionic carbanion C would be pentavalent with a real M bonded)
                nb.SetNoImplicit(True)
        mol = rw.GetMol()
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
        )
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    parts = []
    for idx in atom_centers:
        a = mol.GetAtomWithIdx(idx)
        code = a.GetPropsAsDict().get("_CIPCode") or {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW",
        }.get(a.GetChiralTag())
        if code:  # an unresolved centre (an allene axis RDKit can't set) is dropped, never given a '?' tag
            parts.append(f"{idx}{code}")
    for bidx in bond_centers:
        b = mol.GetBondWithIdx(bidx)
        tag = {
            Chem.BondStereo.STEREOE: "E",
            Chem.BondStereo.STEREOZ: "Z",
            Chem.BondStereo.STEREOTRANS: "E",
            Chem.BondStereo.STEREOCIS: "Z",
        }.get(b.GetStereo())
        if tag:
            parts.append(f"{b.GetBeginAtomIdx()}={b.GetEndAtomIdx()}:{tag}")
    return ",".join(parts)


def _coordination_locked_double_bonds(mol, metals):
    """Double bonds whose E/Z is fixed by the coordination: endocyclic in a ring closed through the metal.

    Such a bond has one buildable geometry (decided by the coordination isomer, the polyhedron path's job), so
    enumerating both E and Z is a phantom: the wrong hand forces a bite the chelate can't span and the pipeline
    burns seeds relaxing it into broken bonds. An alpha-diimine (N=C-C=N chelate) is the type case: both C=N
    sit in the 5-membered metal ring and were enumerated 2x2.

    RDKit ignores dative M-donor bonds in ring perception, so the metal-closed ring is invisible natively;
    upgrade the datives to single to reveal it. A double bond still in a ring once the metal is removed is a
    genuine organic ring bond (RDKit already handles its E/Z) and left alone; only a bond cyclic because of the
    metal is locked here.
    """
    metals = set(metals)
    if not metals:
        return set()
    up = Chem.RWMol(mol)  # dative -> single so RDKit sees the metal ring; FastFindRings avoids a valence sanitize
    for b in up.GetBonds():
        if b.GetBondType() == Chem.BondType.DATIVE:
            b.SetBondType(Chem.BondType.SINGLE)
    up = up.GetMol()
    Chem.FastFindRings(up)
    metal_rings = [set(r) for r in up.GetRingInfo().AtomRings() if metals & set(r)]
    free = Chem.RWMol(mol)  # the metal-free graph: which double bonds are still cyclic without the metal?
    for m in sorted(metals, reverse=True):
        for nb in [n.GetIdx() for n in free.GetAtomWithIdx(m).GetNeighbors()]:
            remove_bond(free, m, nb)
    free = free.GetMol()
    Chem.FastFindRings(free)
    locked = set()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        in_metal_ring = any({a, c} <= r for r in metal_rings)
        fb = free.GetBondBetweenAtoms(a, c)
        if in_metal_ring and not (fb is not None and fb.IsInRing()):  # cyclic only because of the metal
            locked.add(frozenset((a, c)))
    return locked


def _lock_double_bond(work, fb):
    """Pin a coordination-locked double bond to an arbitrary definite stereo on ``work``; return True on success.

    Stops ``onlyUnassigned`` from enumerating it. The value is never grafted onto the full mol (``graft`` skips
    locked bonds), so the metal embed builds the one ring-feasible hand.

    ``SetStereoAtoms`` requires the two reference atoms in the bond's own begin/end order (each a neighbour of
    the corresponding end), so read the order off the bond, not off the unordered ``fb``.
    """
    a, c = tuple(fb)
    wb = work.GetBondBetweenAtoms(a, c)
    if wb is None:
        return False
    bi, ei = wb.GetBeginAtomIdx(), wb.GetEndAtomIdx()
    nb_b = next((n.GetIdx() for n in work.GetAtomWithIdx(bi).GetNeighbors() if n.GetIdx() != ei), None)
    nb_e = next((n.GetIdx() for n in work.GetAtomWithIdx(ei).GetNeighbors() if n.GetIdx() != bi), None)
    if nb_b is None or nb_e is None:
        return False
    try:
        wb.SetStereoAtoms(nb_b, nb_e)
        wb.SetStereo(Chem.BondStereo.STEREOCIS)
    except (RuntimeError, ValueError):  # degrade to "not locked" rather than crash; the phantom just enumerates
        return False
    return True


def _build_enumeration_graph(mol, exclude):
    """Disconnect each metal and D-cap each freed sp3 donor so RDKit enumerates only ligand stereo.

    Returns ``(work, cap_to_metal)``: the cap index -> its metal's atomic number. With no `exclude` there is
    nothing to disconnect, so `mol` is returned unchanged.
    """
    if not exclude:
        return mol, {}
    # Disconnect each metal first: a metal-bound donor is a stereocentre only while bound, so RDKit would
    # enumerate hands the surrogate cannot hold.
    cap_to_metal = {}  # D-cap atom index -> its metal's atomic number (for the coordinated-complex CIP label)
    work = Chem.RWMol(mol)
    for mi in exclude:
        z_metal = mol.GetAtomWithIdx(mi).GetAtomicNum()
        for nb in [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]:
            remove_bond(work, mi, nb)  # re-base the donor's tag onto the stripped order; `graft` inverts it
            if mol.GetAtomWithIdx(nb).GetHybridization() == Chem.HybridizationType.SP3:
                d = work.AddAtom(Chem.Atom(1))
                work.GetAtomWithIdx(d).SetIsotope(2)  # deuterium
                work.AddBond(nb, d, Chem.BondType.SINGLE)
                work.GetAtomWithIdx(nb).SetNoImplicit(True)
                cap_to_metal[d] = z_metal
    work = work.GetMol()
    Chem.SanitizeMol(work, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    # The strip above can orphan a C=N whose stereo reference atom WAS the metal, and a flagged bond with no
    # references makes `FindPotentialStereo` below raise ("only can support 2 stereo neighbors"). The
    # tolerant sanitize happens to scrub most of them, but that is luck rather than a contract.
    repair_bond_stereo(work)
    return work, cap_to_metal


def _unassigned_elements(mol, exclude=()):
    """Return ``(work, cap_to_metal, locked, elements)``: the enumeration graph and its unspecified elements."""
    exclude = set(exclude)
    work, cap_to_metal = _build_enumeration_graph(mol, exclude)

    # A C=N / C=C whose E/Z the coordination fixes must not be enumerated: the metal closes the ring, so only
    # one geometry exists and the other embeds as a strained impossibility.
    locked = _coordination_locked_double_bonds(mol, exclude)
    locked = {fb for fb in locked if _lock_double_bond(work, fb)}  # keep only the ones we could actually pin

    def enumerable(e):  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
        if e.specified != Chem.StereoSpecified.Unspecified:
            return False
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            return True
        if e.type == Chem.StereoType.Bond_Double:  # skip a double bond the coordination has already locked
            wb = work.GetBondWithIdx(e.centeredOn)
            return frozenset((wb.GetBeginAtomIdx(), wb.GetEndAtomIdx())) not in locked
        return False

    return work, cap_to_metal, locked, [e for e in Chem.FindPotentialStereo(work) if enumerable(e)]


def unassigned_centres(mol, exclude=()):
    """Return the unspecified stereo elements as atom-index tuples: ``(atom,)``, or ``(i, j)`` for a double bond.

    The cheap predicate behind `enumerate_unassigned`: what WOULD be expanded, without expanding it. A caller
    that embeds one species (`Isomer`) uses it to refuse to pool two enantiomers silently.
    """
    work, _caps, _locked, elements = _unassigned_elements(mol, exclude)
    out = []
    for e in elements:  # `work` only APPENDS D-caps, so every index here is a real atom of `mol`
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            out.append((e.centeredOn,))
        else:
            b = work.GetBondWithIdx(e.centeredOn)
            out.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    return out


def enumerate_unassigned(mol, cap=32, exclude=()):
    """Enumerate stereoisomers over only the *unspecified* stereo elements (point R/S + double-bond E/Z).

    Returns ``(variants, n_unassigned, total, unresolved)``: ``variants`` a list of ``(variant_mol, label)``
    with defined centres held (`onlyUnassigned`), meso/duplicates dropped (`unique`), truncated to ``cap`` of
    ``total``; ``unresolved`` counts elements RDKit could not enumerate -- an allene axis or a biaryl
    atropisomer, which stay one arbitrary hand for the caller to warn about. Atom order is preserved, so
    index-based ``fix``/``constrain`` stay valid.

    A metal complex is safe: the metal is excluded, its handedness being the coordination-isomer path's job
    and RDKit's dative-metal stereo not order-canonical. A ligand stereocentre is still enumerated, including
    a chiral-at-P or carbanion donor that drops to degree 3 after the strip. `exclude` is the metal indices.
    """
    n_real = mol.GetNumAtoms()
    work, cap_to_metal, locked, unassigned = _unassigned_elements(mol, exclude)
    if not unassigned:
        return [(mol, "")], 0, 1, 0
    atom_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Atom_Tetrahedral]
    bond_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Bond_Double]
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=cap)
    total = GetStereoisomerCount(work, opts)
    work_isos = list(EnumerateStereoisomers(work, opts))

    def graft(wv):  # copy the enumerated ligand stereo (atom parity + E/Z) onto the FULL mol; skip the D caps
        full = Chem.Mol(mol)
        for a in wv.GetAtoms():
            if a.GetIdx() >= n_real:  # an appended D cap has no counterpart on the full mol
                continue
            fa = full.GetAtomWithIdx(a.GetIdx())
            # `work` is `mol` with each M-donor bond removed, so a tag enumerated there is in the STRIPPED
            # bond order; writing it back across a bond the full mol still has is the inverse re-basing. The
            # D cap does not enter it: appended last, it stands in the slot the metal's removal vacated.
            tag = a.GetChiralTag()
            for mi in exclude:
                if bond_removal_mirrors(fa, mi):
                    tag = mirror_tag(tag)
            fa.SetChiralTag(tag)
        for b in wv.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            # skip a coordination-locked bond: its `work` stereo is the arbitrary lock value, not a real hand;
            # the full mol keeps it unspecified so the metal embed builds the ring-feasible geometry.
            if frozenset((i, j)) in locked:
                continue
            if b.GetStereo() != Chem.BondStereo.STEREONONE and i < n_real and j < n_real:
                fb = full.GetBondBetweenAtoms(i, j)
                if fb is not None:
                    fb.SetStereoAtoms(*b.GetStereoAtoms())
                    fb.SetStereo(b.GetStereo())
        return full

    variants = [(graft(wv), _stereo_label(wv, atom_centers, bond_centers, cap_to_metal)) for wv in work_isos]
    probe = work_isos[0] if work_isos else work  # centres still UNSPECIFIED after enumeration = axial (allene)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    unresolved = sum(
        1 for i in atom_centers if probe.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    ) + sum(1 for b in bond_centers if probe.GetBondWithIdx(b).GetStereo() == Chem.BondStereo.STEREONONE)
    return variants, len(unassigned), total, unresolved
