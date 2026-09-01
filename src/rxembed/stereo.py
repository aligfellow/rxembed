"""Expand undefined ligand stereo, including native atrop tags RDKit does not enumerate."""

from __future__ import annotations

import itertools
import re

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from .metal_core import _haptic_sites
from .utils import (
    _STEREO_REFS,
    bond_removal_mirrors,
    bond_replacement_mirrors,
    mirror_tag,
    remove_bond,
    repair_bond_stereo,
)

_MIN_POINT_BRANCHES = 3
_ATROP_STEREO = (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
_ATROP_WEDGE = (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH)


def point_stereo(label):
    """Return the point-centre codes in a ligand-stereo label, keyed by atom index."""
    return {
        int(match.group(1)): match.group(2)
        for part in label.split(",")
        if label
        if (match := re.fullmatch(r"[A-Z][a-z]?(\d+):(R|S|CW|CCW)", part))
    }


def axis_stereo(label):
    """Return native M/P atropisomer descriptors keyed by their axis atom pair."""
    return {
        tuple(sorted((int(match.group(1)), int(match.group(2))))): match.group(3)
        for part in label.split(",")
        if label
        if (match := re.fullmatch(r"[A-Z][a-z]?(\d+)-[A-Z][a-z]?(\d+):(M|P)", part))
    }


def _stereo_label(mol, atom_centers, bond_centers, atrop_centers=(), cap_to_metal=None):
    """Build an atom-qualified configuration tag, e.g. ``'C1:R,C3=C4:E,C5-C6:M'``.

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
    # A standard CX c:/t: field directly sets a valid CIS/TRANS tag but no slash bond directions. RDKit's
    # clean assignment erases that tag, so retain stated bond geometry while assigning point-centre CIP.
    stated_bonds = {
        b.GetIdx(): (b.GetStereo(), tuple(b.GetStereoAtoms()))
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetStereo() != Chem.BondStereo.STEREONONE
        and len(b.GetStereoAtoms()) == _STEREO_REFS
        and len(set(b.GetStereoAtoms())) == _STEREO_REFS
    }
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for idx, (tag, refs) in stated_bonds.items():
        bond = mol.GetBondWithIdx(idx)
        bond.SetStereoAtoms(*refs)
        bond.SetStereo(tag)
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
    for i, j in atrop_centers:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetStereo() not in _ATROP_STEREO:
            continue
        Chem.AssignCIPLabels(mol, bondsToLabel=[bond.GetIdx()])
        code = bond.GetPropsAsDict().get("_CIPCode")
        if code not in {"M", "P"}:
            raise ValueError(f"RDKit could not assign M/P to atropisomer bond {i}-{j}")
        first, second = sorted((i, j))
        left, right = mol.GetAtomWithIdx(first), mol.GetAtomWithIdx(second)
        parts.append(f"{left.GetSymbol()}{first}-{right.GetSymbol()}{second}:{code}")
    return ",".join(parts)


def _label_item(part):
    """Parse one indexed ligand-stereo label into its selector fields."""
    if match := re.fullmatch(r"([A-Z][a-z]?)(\d+):(R|S|CW|CCW)", part):
        symbol, index, code = match.groups()
        return "point", (symbol,), code, part, f"{index}{code}"
    if match := re.fullmatch(r"([A-Z][a-z]?)(\d+)=([A-Z][a-z]?)(\d+):(E|Z)", part):
        left, i, right, j, code = match.groups()
        return "bond", tuple(sorted((left, right))), code, part, f"{i}={j}:{code}"
    if match := re.fullmatch(r"([A-Z][a-z]?)(\d+)-([A-Z][a-z]?)(\d+):(M|P)", part):
        left, i, right, j, code = match.groups()
        return "axis", tuple(sorted((left, right))), code, part, f"{i}-{j}:{code}"
    return None


def matches_stereo(label, selector):
    """Match an indexed stereo selector or an unambiguous configuration shorthand."""
    if selector == label:
        return True
    if "," in selector:
        return all(matches_stereo(label, part.strip()) for part in selector.split(","))
    items = [item for part in label.split(",") if (item := _label_item(part)) is not None] if label else []
    if selector in {item[3] for item in items} | {item[4] for item in items}:
        return True
    if selector == ",".join(item[4] for item in items):
        return True

    kind = symbols = wanted = None
    if selector in {"R", "S", "CW", "CCW"}:
        kind, wanted = "point", selector
    elif selector in {"E", "Z"}:
        kind, wanted = "bond", selector
    elif selector in {"M", "P"}:
        kind, wanted = "axis", selector
    elif match := re.fullmatch(r"([A-Z][a-z]?):(R|S|CW|CCW)", selector):
        kind, symbols, wanted = "point", (match.group(1),), match.group(2)
    elif match := re.fullmatch(r"([A-Z][a-z]?)=([A-Z][a-z]?):(E|Z)", selector):
        kind, symbols, wanted = "bond", tuple(sorted(match.group(1, 2))), match.group(3)
    elif match := re.fullmatch(r"([A-Z][a-z]?)-([A-Z][a-z]?):(M|P)", selector):
        kind, symbols, wanted = "axis", tuple(sorted(match.group(1, 2))), match.group(3)
    if kind is None:
        return False
    candidates = [item for item in items if item[0] == kind and (symbols is None or item[1] == symbols)]
    if len(candidates) > 1:
        raise ValueError(
            f"stereo={selector!r} is ambiguous for {label!r}; use one of {[item[3] for item in candidates]}"
        )
    return len(candidates) == 1 and candidates[0][2] == wanted


def _clear_atrop(mol):
    """Clear native atropisomer tags and their signaling wedges; return the axis atom pairs."""
    axes = set()
    for bond in mol.GetBonds():
        if bond.GetStereo() not in _ATROP_STEREO:
            continue
        axes.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for adjacent in atom.GetBonds():
                if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondDir() in _ATROP_WEDGE:
                    adjacent.SetBondDir(Chem.BondDir.NONE)
    return axes


def _assign_atrop_from_3d(mol, atrop_centers):
    """Assign selected native atrop bonds from 3D coordinates through RDKit's MolBlock parser."""
    if not atrop_centers:
        return
    probe = Chem.Mol(mol)
    for pair in atrop_centers:
        probe.GetBondBetweenAtoms(*pair).SetStereo(Chem.BondStereo.STEREOATROPCW)
    block = Chem.MolToMolBlock(probe, confId=probe.GetConformer().GetId(), includeStereo=True)
    perceived = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if perceived is None or not perceived.GetConformer().Is3D():
        raise ValueError("RDKit could not perceive atropisomer stereo from the 3D MolBlock")
    for pair in atrop_centers:
        tag = perceived.GetBondBetweenAtoms(*pair).GetStereo()
        if tag not in _ATROP_STEREO:
            raise ValueError(f"RDKit could not perceive atropisomer axis {pair} from 3D coordinates")
        bond = mol.GetBondBetweenAtoms(*pair)
        bond.SetStereo(tag)
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")


def _metal_closed_rings(mol, metals):
    """Return atom and bond sets for rings visible after treating dative bonds as single."""
    up = Chem.RWMol(mol)
    for bond in up.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
    up = up.GetMol()
    Chem.FastFindRings(up)
    return [
        (set(atoms), set(bonds))
        for atoms, bonds in zip(up.GetRingInfo().AtomRings(), up.GetRingInfo().BondRings(), strict=True)
        if set(metals) & set(atoms)
    ]


def _coordination_atrop_bonds(mol, metals, work):
    """Return native-eligible, ortho-blocked axes inside a metal-closed chelate."""
    ring_bonds = set().union(*(bonds for _atoms, bonds in _metal_closed_rings(mol, metals))) if metals else set()
    axes = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            bond.GetBondType() != Chem.BondType.SINGLE
            or not all(atom.GetIsAromatic() for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
            or bond.GetIdx() not in ring_bonds
            or bond.IsInRing()
        ):
            continue
        if any(
            neighbor.GetTotalNumHs(includeNeighbors=True)
            for end in (i, j)
            for neighbor in mol.GetAtomWithIdx(end).GetNeighbors()
            if neighbor.GetIdx() not in (i, j)
        ):
            continue  # RDKit validates an axis but does not decide whether its ortho groups block rotation
        probe = Chem.Mol(work)
        candidate = probe.GetBondBetweenAtoms(i, j)
        candidate.SetStereo(Chem.BondStereo.STEREOATROPCW)
        Chem.CleanupAtropisomers(probe)
        if candidate.GetStereo() == Chem.BondStereo.STEREOATROPCW:
            axes.append(tuple(sorted((i, j))))
    return axes


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
    haptic = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for metal in metals
        for site in _haptic_sites(mol, [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()])
        if len(site) > 1
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in site and bond.GetEndAtomIdx() in site
    }
    metal_rings = _metal_closed_rings(mol, metals)
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
        if frozenset((a, c)) in haptic:
            continue  # an eta2 C=C keeps its ligand E/Z; coordination chooses a face, not a bond geometry
        in_metal_ring = any({a, c} <= atoms for atoms, _bonds in metal_rings)
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


def _unassigned_elements(
    mol,
    exclude=(),
    include=(),
    skip_points=(),
    skip_bonds=False,
    skip_atrop=False,
    include_atrop=(),
):
    """Return the enumeration graph, ordinary elements, and native atropisomer axes."""
    exclude = set(exclude)
    work, cap_to_metal = _build_enumeration_graph(mol, exclude)
    forced_atrop = {tuple(sorted(pair)) for pair in include_atrop}
    if include == "all":
        forced_atrop.update(_clear_atrop(work))
        Chem.RemoveStereochemistry(work)
    else:
        for atom in include:
            work.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    # A C=N / C=C whose E/Z the coordination fixes must not be enumerated: the metal closes the ring, so only
    # one geometry exists and the other embeds as a strained impossibility.
    locked = _coordination_locked_double_bonds(mol, exclude)
    locked = {fb for fb in locked if _lock_double_bond(work, fb)}  # keep only the ones we could actually pin

    def enumerable(e):  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
        if e.specified != Chem.StereoSpecified.Unspecified:
            return False
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            return e.centeredOn not in skip_points
        if e.type == Chem.StereoType.Bond_Double:  # skip a double bond the coordination has already locked
            wb = work.GetBondWithIdx(e.centeredOn)
            return not skip_bonds and frozenset((wb.GetBeginAtomIdx(), wb.GetEndAtomIdx())) not in locked
        return False

    atrop = set() if skip_atrop else forced_atrop
    atrop = sorted(
        pair
        for pair in atrop
        if (bond := work.GetBondBetweenAtoms(*pair)) is not None and bond.GetStereo() == Chem.BondStereo.STEREONONE
    )
    return work, cap_to_metal, locked, [e for e in Chem.FindPotentialStereo(work) if enumerable(e)], atrop


def point_centres(mol, exclude=()):
    """Return atom indices that can carry tetrahedral ligand stereo on the metal-free enumeration graph."""
    work, _caps = _build_enumeration_graph(mol, set(exclude))
    return {
        element.centeredOn
        for element in Chem.FindPotentialStereo(work)
        if element.type == Chem.StereoType.Atom_Tetrahedral and element.centeredOn < mol.GetNumAtoms()
    }


def unassigned_centres(mol, exclude=()):
    """Return the unspecified stereo elements as atom-index tuples: ``(atom,)``, or ``(i, j)`` for a double bond.

    The cheap predicate behind `enumerate_unassigned`: what WOULD be expanded, without expanding it. A caller
    that embeds one species (`Isomer`) uses it to refuse to pool two enantiomers silently.
    """
    work, _caps, _locked, elements, atrop = _unassigned_elements(mol, exclude)
    out = []
    for e in elements:  # `work` only APPENDS D-caps, so every index here is a real atom of `mol`
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            out.append((e.centeredOn,))
        else:
            b = work.GetBondWithIdx(e.centeredOn)
            out.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    return [*out, *atrop]


def _stereo_centres(mol, n_real):
    """Return defined point, E/Z, and native atrop centres."""
    points = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() < n_real
        and atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    ]
    doubles = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    axes = [
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
        if bond.GetStereo() in _ATROP_STEREO
    ]
    return points, doubles, axes


def defined_stereo_label(mol, exclude=()):
    """Label the ligand stereo already defined on a coordinated molecule."""
    work, cap_to_metal = _build_enumeration_graph(mol, set(exclude))
    atom_centers, bond_centers, atrop_centers = _stereo_centres(work, mol.GetNumAtoms())
    return _stereo_label(work, atom_centers, bond_centers, atrop_centers, cap_to_metal)


def stereo_from_3d(mol, exclude=()):
    """Label ligand stereo measured from the first conformer."""
    if not mol.GetNumConformers():
        raise ValueError("stereo_from_3d needs a conformer")
    exclude = set(exclude)
    work, cap_to_metal = _build_enumeration_graph(mol, exclude)
    _, _, stated = _stereo_centres(work, mol.GetNumAtoms())
    atrop_centers = sorted(set(stated) | set(_coordination_atrop_bonds(mol, exclude, work)))
    Chem.AssignStereochemistryFrom3D(work, confId=work.GetConformer().GetId(), replaceExistingTags=True)
    _assign_atrop_from_3d(work, atrop_centers)
    atom_centers, bond_centers, _ = _stereo_centres(work, mol.GetNumAtoms())
    return _stereo_label(work, atom_centers, bond_centers, atrop_centers, cap_to_metal)


def _enumerate_atrop(work_isos, atrop_centers, cap):
    """Expand and native-canonicalize the selected atropisomer bonds."""
    if not atrop_centers:
        return work_isos
    expanded, seen = [], set()
    for base in work_isos:
        for tags in itertools.product(_ATROP_STEREO, repeat=len(atrop_centers)):
            variant = Chem.Mol(base)
            for pair, tag in zip(atrop_centers, tags, strict=True):
                variant.GetBondBetweenAtoms(*pair).SetStereo(tag)
            key = Chem.MolToCXSmiles(variant)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(variant)
            if len(expanded) == cap:
                return expanded
    return expanded


def enumerate_unassigned(
    mol,
    cap=32,
    exclude=(),
    include=(),
    skip_points=(),
    skip_bonds=False,
    skip_atrop=False,
    include_atrop=(),
):
    """Enumerate unspecified point, double-bond, and native atropisomer stereo.

    Returns ``(variants, n_unassigned, total, unresolved)``: ``variants`` a list of ``(variant_mol, label)``
    with defined centres held (`onlyUnassigned`), meso/duplicates dropped (`unique`), truncated to ``cap`` of
    ``total``; ``unresolved`` counts elements RDKit could not enumerate, such as an allene axis, which stays
    one arbitrary hand for the caller to warn about. Atom order is preserved, so index-based
    ``fix``/``constrain`` stay valid.

    A metal complex is safe: the metal is excluded, its handedness being the coordination-isomer path's job
    and RDKit's dative-metal stereo not order-canonical. A ligand stereocentre is still enumerated, including
    a chiral-at-P or carbanion donor that drops to degree 3 after the strip. `exclude` is the metal indices.
    """
    n_real = mol.GetNumAtoms()
    work, cap_to_metal, locked, unassigned, atrop_centers = _unassigned_elements(
        mol,
        exclude,
        include,
        set(skip_points),
        skip_bonds,
        skip_atrop,
        include_atrop,
    )
    if not unassigned and not atrop_centers:
        return [(mol, "")], 0, 1, 0
    atom_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Atom_Tetrahedral]
    bond_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Bond_Double]
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=cap)
    total = GetStereoisomerCount(work, opts) * 2 ** len(atrop_centers)
    work_isos = list(EnumerateStereoisomers(work, opts)) if unassigned else [work]
    work_isos = _enumerate_atrop(work_isos, atrop_centers, cap)

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
        for pair in atrop_centers:
            wb = wv.GetBondBetweenAtoms(*pair)
            fb = full.GetBondBetweenAtoms(*pair)
            if wb is not None and fb is not None:
                fb.SetStereo(wb.GetStereo())
        return full

    variants = [
        (graft(wv), _stereo_label(wv, atom_centers, bond_centers, atrop_centers, cap_to_metal)) for wv in work_isos
    ]
    probe = work_isos[0] if work_isos else work  # centres still UNSPECIFIED after enumeration = axial (allene)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    unresolved = sum(
        1 for i in atom_centers if probe.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    ) + sum(1 for b in bond_centers if probe.GetBondWithIdx(b).GetStereo() == Chem.BondStereo.STEREONONE)
    return variants, len(unassigned) + len(atrop_centers), total, unresolved
