"""Ligand-stereo load-in: expand a `Mol`'s UNDEFINED stereocentres into the distinct species to embed.

The embed-side half of stereochemistry: graph-only (RDKit's `EnumerateStereoisomers` over the
unspecified elements), metal-safe (the centre's handedness is the coordination-isomer path's job). The
conformer-side half, the coordinate-derived chirality fingerprint that filters embedded conformers,
is perception-driven and stays with the consumer (`rxembed.stereo`).
"""

from __future__ import annotations

import re

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from .metal_core import _haptic_sites
from .utils import bond_removal_mirrors, bond_replacement_mirrors, mirror_tag, remove_bond, repair_bond_stereo

_MIN_POINT_BRANCHES = 3


def point_stereo(label):
    """Return the point-centre codes in a ligand-stereo label, keyed by atom index."""
    return {
        int(match.group(1)): match.group(2)
        for part in label.split(",")
        if label
        if (match := re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|CW|CCW)", part))
    }


def _stereo_label(mol, atom_centers, bond_centers, cap_to_metal=None):
    """Build an atom-qualified configuration tag, e.g. ``'C1:S'`` or ``'C1:R,C3:S,C5=C6:E'``.

    CIP R/S where RDKit assigns it (falls back to the raw CW/CCW tag for a centre it won't CIP-rank, e.g. some
    P), plus E/Z for each enumerated double bond. Keyed only on the *enumerated* atoms/bonds so distinct
    variants always get distinct, stable labels. ``cap_to_metal`` maps each donor's D-cap atom index to its
    metal's atomic number: the CIP is then computed with the METAL (highest priority) in the cap position, not
    the D (lowest), so a metal-bound donor's R/S names the coordinated centre correctly (the D->M priority
    flip is not a fixed R<->S swap; it depends on the donor's other substituents, e.g. whether it carries an H).
    """
    if cap_to_metal:  # temporarily give each D-cap the metal's atomic number for a coordinated-complex CIP
        rw = Chem.RWMol(mol)
        charged = set()
        for d_idx, (z, donated, _mirrored) in cap_to_metal.items():
            rw.GetAtomWithIdx(d_idx).SetAtomicNum(z)
            rw.GetAtomWithIdx(d_idx).SetIsotope(0)
            for nb in rw.GetAtomWithIdx(d_idx).GetNeighbors():
                # Replacing donor->M by donor-M makes that donation covalent for CIP: C- -> C, N -> N+.
                # A covalent M-D bond is only replaced, so its donor charge does not move.
                if donated and nb.GetIdx() not in charged:
                    nb.SetFormalCharge(nb.GetFormalCharge() + 1)
                    charged.add(nb.GetIdx())
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
            parts.append(f"{a.GetSymbol()}{idx}:{code}")
    for bidx in bond_centers:
        b = mol.GetBondWithIdx(bidx)
        tag = {
            Chem.BondStereo.STEREOE: "E",
            Chem.BondStereo.STEREOZ: "Z",
            Chem.BondStereo.STEREOTRANS: "E",
            Chem.BondStereo.STEREOCIS: "Z",
        }.get(b.GetStereo())
        if tag:
            begin, end = b.GetBeginAtom(), b.GetEndAtom()
            parts.append(f"{begin.GetSymbol()}{begin.GetIdx()}={end.GetSymbol()}{end.GetIdx()}:{tag}")
    return ",".join(parts)


def matches_stereo(label, selector):
    """Match an indexed stereo selector or an unambiguous configuration shorthand."""
    if selector == label:
        return True
    items = []
    for part in label.split(",") if label else ():
        point = re.fullmatch(r"([A-Z][a-z]?)(\d+):(R|S|CW|CCW)", part)
        bond = re.fullmatch(r"([A-Z][a-z]?)(\d+)=([A-Z][a-z]?)(\d+):(E|Z)", part)
        if point:
            symbol, index, code = point.groups()
            items.append(("point", (symbol,), code, part, f"{index}{code}"))
        elif bond:
            left, i, right, j, code = bond.groups()
            items.append(("bond", tuple(sorted((left, right))), code, part, f"{i}={j}:{code}"))
    if selector in {item[3] for item in items} | {item[4] for item in items}:
        return True
    if selector == ",".join(item[4] for item in items):
        return True

    kind = symbols = wanted = None
    if selector in {"R", "S", "CW", "CCW"}:
        kind, wanted = "point", selector
    elif selector in {"E", "Z"}:
        kind, wanted = "bond", selector
    elif match := re.fullmatch(r"([A-Z][a-z]?):(R|S|CW|CCW)", selector):
        kind, symbols, wanted = "point", (match.group(1),), match.group(2)
    elif match := re.fullmatch(r"([A-Z][a-z]?)=([A-Z][a-z]?):(E|Z)", selector):
        kind, symbols, wanted = "bond", tuple(sorted(match.group(1, 2))), match.group(3)
    if kind is None:
        return False
    candidates = [item for item in items if item[0] == kind and (symbols is None or item[1] == symbols)]
    if len(candidates) > 1:
        raise ValueError(
            f"stereo={selector!r} is ambiguous for {label!r}; use one of {[item[3] for item in candidates]}"
        )
    return len(candidates) == 1 and candidates[0][2] == wanted


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
    """Disconnect each metal and D-cap each freed sp3 sigma donor so RDKit enumerates only ligand stereo.

    Returns ``(work, cap_to_metal)``: the cap index -> ``(metal atomic number, was donor->metal dative,
    replacement changed parity)``.
    With no `exclude` there is nothing to disconnect, so `mol` is returned unchanged.
    """
    if not exclude:
        return mol, {}
    # Disconnect each metal first: a metal-bound donor is a stereocentre only while bound, so RDKit would
    # enumerate hands the surrogate cannot hold.
    cap_to_metal = {}
    work = Chem.RWMol(mol)
    for mi in sorted(exclude):
        z_metal = mol.GetAtomWithIdx(mi).GetAtomicNum()
        donors = [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]
        haptic = {d for site in _haptic_sites(mol, donors) if len(site) > 1 for d in site}
        for nb in donors:
            bond = mol.GetBondBetweenAtoms(mi, nb)
            donated = bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() == nb
            replacement_mirrors = bond_replacement_mirrors(work.GetAtomWithIdx(nb), mi)
            removal_mirrors = bond_removal_mirrors(work.GetAtomWithIdx(nb), mi)
            remove_bond(work, mi, nb)  # re-base the donor's tag onto the stripped order; `graft` inverts it
            donor = mol.GetAtomWithIdx(nb)
            sigma_only = all(
                b.GetOtherAtomIdx(nb) in exclude or b.GetBondType() == Chem.BondType.SINGLE for b in donor.GetBonds()
            )
            # Two identical H rule out tetrahedral chirality; a D cap makes RDKit misclassify bracket `[PH3]`.
            # A haptic atom belongs to a pi face, not one sigma-donor point centre; capping it creates a
            # phantom R/S centre when RDKit parses a fully dative Cp ring as locally sp3.
            if (
                nb not in haptic
                and donor.GetDegree() + donor.GetTotalNumHs() >= _MIN_POINT_BRANCHES
                and (donor.GetHybridization() == Chem.HybridizationType.SP3 or sigma_only)
                and donor.GetTotalNumHs() <= 1
            ):
                if replacement_mirrors != removal_mirrors:
                    atom = work.GetAtomWithIdx(nb)
                    atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
                d = work.AddAtom(Chem.Atom(1))
                work.GetAtomWithIdx(d).SetIsotope(2 + z_metal)  # preserve equal/different metal identity
                work.AddBond(nb, d, Chem.BondType.SINGLE)
                work.GetAtomWithIdx(nb).SetNoImplicit(True)
                cap_to_metal[d] = (z_metal, donated, replacement_mirrors)
                for conf in work.GetConformers():
                    conf.SetAtomPosition(d, conf.GetAtomPosition(mi))
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


def defined_stereo_label(mol, exclude=()):
    """Label the ligand stereo already defined on a coordinated molecule."""
    work, cap_to_metal = _build_enumeration_graph(mol, set(exclude))
    atom_centers = [
        atom.GetIdx()
        for atom in work.GetAtoms()
        if atom.GetIdx() < mol.GetNumAtoms()
        and atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    ]
    bond_centers = [
        bond.GetIdx()
        for bond in work.GetBonds()
        if bond.GetStereo()
        in {
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOCIS,
        }
    ]
    return _stereo_label(work, atom_centers, bond_centers, cap_to_metal)


def stereo_from_3d(mol, exclude=()):
    """Label ligand stereo measured from the first conformer."""
    if not mol.GetNumConformers():
        raise ValueError("stereo_from_3d needs a conformer")
    work, cap_to_metal = _build_enumeration_graph(mol, set(exclude))
    Chem.AssignStereochemistryFrom3D(work, confId=work.GetConformer().GetId(), replaceExistingTags=True)
    atom_centers = [
        atom.GetIdx()
        for atom in work.GetAtoms()
        if atom.GetIdx() < mol.GetNumAtoms()
        and atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    ]
    bond_centers = [bond.GetIdx() for bond in work.GetBonds() if bond.GetStereo() != Chem.BondStereo.STEREONONE]
    return _stereo_label(work, atom_centers, bond_centers, cap_to_metal)


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
            if a.GetIdx() >= n_real or a.GetIdx() not in atom_centers:
                continue
            fa = full.GetAtomWithIdx(a.GetIdx())
            # `work` is `mol` with each M-donor bond removed, so a tag enumerated there is in the STRIPPED
            # bond order; writing it back across a bond the full mol still has is the inverse re-basing. The
            # D cap does not enter it: appended last, it stands in the slot the metal's removal vacated.
            tag = a.GetChiralTag()
            for cap_idx, (_z, _donated, mirrored) in cap_to_metal.items():
                if mirrored and wv.GetBondBetweenAtoms(a.GetIdx(), cap_idx) is not None:
                    tag = mirror_tag(tag)
            fa.SetChiralTag(tag)
        for b in wv.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            # skip a coordination-locked bond: its `work` stereo is the arbitrary lock value, not a real hand;
            # the full mol keeps it unspecified so the metal embed builds the ring-feasible geometry.
            if b.GetIdx() not in bond_centers or frozenset((i, j)) in locked:
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
