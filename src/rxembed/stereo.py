"""Expand undefined ligand stereo, including native atrop tags RDKit does not enumerate."""

from __future__ import annotations

import itertools
import logging
import re

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from .metal_core import haptic_sites, ligand_graph, torsion_path
from .utils import (
    DISCONNECTED,
    bond_removal_mirrors,
    cip_cache_key,
    flat_ranks,
    mirror_tag,
    remove_bond,
    repair_bond_stereo,
    without_zero_bonds,
)

_MIN_POINT_BRANCHES = 3
_MIN_BLOCKED_ORTHO_CONNECTIONS = 3
_MIN_BRIDGE_METALS = 2
_ISOTOPE_ELEMENT_STRIDE = 128  # exceeds the periodic table, so equal-element bridge caps stay distinct
ATROP_STEREO = (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
_ATROP_WEDGE = (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH)
_TETRAHEDRAL_TAGS = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
_POINT_CIP = {"R", "S", "r", "s"}
_THREE_COORDINATE_3D = {16, 34}  # RDKit's assignChiralTypesFrom3D special case: S and Se use a lone pair
_RESONANCE_EZ_CAP = 32  # forms searched per fragment: independent conjugated groups multiply its form count
_RX_EZ_PROP = re.compile(r"_rxEZ(\d+)")
# One ligand-stereo label item each: a point centre, a double bond, an atropisomer axis.
_POINT_ITEM = re.compile(r"([A-Z][a-z]?)(\d+):(R|S|r|s|CW|CCW)")
_BOND_ITEM = re.compile(r"([A-Z][a-z]?)(\d+)=([A-Z][a-z]?)(\d+):(E|Z)")
_AXIS_ITEM = re.compile(r"([A-Z][a-z]?)(\d+)-([A-Z][a-z]?)(\d+):(M|P)")
logger = logging.getLogger("rxembed.stereo")  # under the "rxembed" tree `set_verbose` configures
_STEREO_SANITIZE = (
    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
)


def point_stereo(label):
    """Return the point-centre codes in a ligand-stereo label, keyed by atom index."""
    return {
        int(match.group(2)): match.group(3)
        for part in label.split(",")
        if label
        if (match := _POINT_ITEM.fullmatch(part))
    }


def bond_stereo(label):
    """Return the E/Z codes in a ligand-stereo label, keyed by atom pair."""
    return {
        frozenset((int(match.group(2)), int(match.group(4)))): match.group(5)
        for part in label.split(",")
        if label
        if (match := _BOND_ITEM.fullmatch(part))
    }


def metal_referenced_ez(mol, label, donor_bonds):
    """Return E/Z bonds whose donor endpoint uses one metal in place of a heavy ligand substituent."""
    metals = {}
    for donor, metal in donor_bonds:
        metals.setdefault(donor, set()).add(metal)
    metal_atoms = {metal for bound in metals.values() for metal in bound}
    ranks = list(Chem.ComputeAtomCIPRanks(mol))
    out = {}
    for pair, wanted in bond_stereo(label).items():
        donors = [atom for atom in pair if atom in metals]
        if len(donors) != 1:
            continue
        donor = donors[0]
        if len(metals[donor]) != 1:
            continue
        other = next(iter(pair - {donor}))
        ligand_refs = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(donor).GetNeighbors()
            if nb.GetIdx() != other and nb.GetIdx() not in metal_atoms
        ]
        if len(ligand_refs) > 1 or any(mol.GetAtomWithIdx(ref).GetAtomicNum() > 1 for ref in ligand_refs):
            continue
        other_refs = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(other).GetNeighbors()
            if nb.GetIdx() != donor and nb.GetIdx() not in metal_atoms
        ]
        if not other_refs:
            continue
        high = max(ranks[ref] for ref in other_refs)
        controls = [ref for ref in other_refs if ranks[ref] == high]
        if len(controls) == 1:
            out[pair] = (
                donor,
                other,
                next(iter(metals[donor])),
                controls[0],
                ligand_refs[0] if ligand_refs else None,
                wanted,
            )
    return out


def encoded_bond_stereo(mol):
    """Return E/Z pairs stored by the CX fallback, rejecting malformed records."""
    records = {}
    for atom in mol.GetAtoms():
        for name in atom.GetPropNames(includePrivate=True, includeComputed=False):
            if (match := _RX_EZ_PROP.fullmatch(name)) is None:
                continue
            code = atom.GetProp(name)
            if code not in {"E", "Z"}:
                raise ValueError(f"invalid CX E/Z value {code!r} on atom {atom.GetIdx()}")
            records.setdefault(int(match.group(1)), []).append((atom.GetIdx(), code))
    out = {}
    for token, entries in records.items():
        atoms = [idx for idx, _code in entries]
        codes = {code for _idx, code in entries}
        if len(entries) != 2 or len(set(atoms)) != 2 or len(codes) != 1:  # noqa: PLR2004
            raise ValueError(f"CX E/Z record {token} must name two atoms with one configuration")
        pair = frozenset(atoms)
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"CX E/Z record {token} does not name a double bond")
        if pair in out:
            raise ValueError(f"duplicate CX E/Z record for bond {tuple(sorted(pair))}")
        out[pair] = codes.pop()
    return out


def apply_encoded_bond_stereo(mol, skip=()):
    """Apply CX-encoded E/Z against the references RDKit perceives, verifying each code by CIP.

    E is TRANS only relative to the CIP-leading substituents, so a code set against arbitrary neighbours can land
    as the other isomer. ``skip`` names atom pairs to leave unassigned.
    """
    skip = {frozenset(pair) for pair in skip}
    encoded = {pair: code for pair, code in encoded_bond_stereo(mol).items() if pair not in skip}
    if not encoded:
        return
    potential = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())): element
        for element in Chem.FindPotentialStereo(mol)
        if element.type == Chem.StereoType.Bond_Double
        for bond in (mol.GetBondWithIdx(element.centeredOn),)
    }
    for pair, wanted in encoded.items():
        element = potential.get(pair)
        if element is None:
            raise ValueError(f"could not restore CX E/Z on bond {tuple(sorted(pair))}")
        bond = mol.GetBondWithIdx(element.centeredOn)
        controls = list(element.controllingAtoms)
        left = next((idx for idx in controls[:2] if idx < mol.GetNumAtoms()), None)
        right = next((idx for idx in controls[2:] if idx < mol.GetNumAtoms()), None)
        if left is None or right is None or left == right:
            raise ValueError(f"could not restore CX E/Z references on bond {tuple(sorted(pair))}")
        bond.SetStereoAtoms(left, right)
        bond.SetStereo(Chem.BondStereo.STEREOE)
        if bond_stereo_code(mol, bond.GetIdx()) != wanted:
            bond.SetStereo(Chem.BondStereo.STEREOZ)
        if bond_stereo_code(mol, bond.GetIdx()) != wanted:
            raise ValueError(f"could not restore CX {wanted} on bond {tuple(sorted(pair))}")


def without_bond_stereo(label, pairs):
    """Remove selected E/Z entries from a ligand-stereo label without changing its graph."""
    return ",".join(part for part in label.split(",") if pairs.isdisjoint(bond_stereo(part)))


def axis_stereo(label):
    """Return native M/P atropisomer descriptors keyed by their axis atom pair."""
    return {
        tuple(sorted((int(match.group(2)), int(match.group(4))))): match.group(5)
        for part in label.split(",")
        if label
        if (match := _AXIS_ITEM.fullmatch(part))
    }


def apply_point_stereo(mol, label, centers):
    """Apply absolute point labels as local tags in ``mol``'s current full-graph bond order."""
    centers = set(centers)
    for idx, wanted in point_stereo(label).items():
        if idx not in centers:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if wanted in {"CW", "CCW"}:
            atom.SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CW if wanted == "CW" else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            )
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
            continue
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
        Chem.AssignCIPLabels(mol, atomsToLabel=[idx])
        actual = atom.GetPropsAsDict().get("_CIPCode")
        if actual not in _POINT_CIP:
            raise ValueError(f"could not apply {wanted} ligand stereo at atom {idx} on the coordinated graph")
        if actual != wanted:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        atom.ClearProp("_CIPCode")


def bond_stereo_code(mol, bond_index):
    """Return a double bond's CIP E/Z code, independent of its traversal-relative cis/trans tag."""
    probe = Chem.Mol(mol)
    bond = probe.GetBondWithIdx(bond_index)
    try:
        Chem.AssignCIPLabels(probe, bondsToLabel=[bond_index])
    except (RuntimeError, ValueError):
        pass
    code = bond.GetPropsAsDict().get("_CIPCode")
    if code in {"E", "Z"}:
        return code
    return {
        Chem.BondStereo.STEREOE: "E",
        Chem.BondStereo.STEREOZ: "Z",
        Chem.BondStereo.STEREOTRANS: "E",
        Chem.BondStereo.STEREOCIS: "Z",
    }.get(bond.GetStereo())


def _stereo_label(mol, atom_centers, bond_centers, atrop_centers=(), cap_to_metal=None, point_codes=None):
    """Build an atom-qualified configuration tag, e.g. ``'C1:R,C3=C4:E,C5-C6:M'``.

    CIP R/S or pseudoasymmetric r/s where RDKit assigns it (falls back to the raw CW/CCW tag for a centre it
    won't CIP-rank, e.g. some P), plus E/Z for each enumerated double bond. Keyed only on the *enumerated*
    atoms/bonds so distinct variants always get distinct, stable labels. Full-graph ``point_codes`` take
    precedence over capped-ligand labels: removing a metal can change priorities even at a remote centre.
    ``cap_to_metal`` supplies each cap's metal atomic number for centres RDKit cannot label on the full graph.
    """
    if cap_to_metal:  # temporarily give each cap the metal's atomic number for coordinated-complex CIP
        rw = Chem.RWMol(mol)
        charged = set()
        for d_idx, (z, donated, _mirrored, _metal) in cap_to_metal.items():
            rw.GetAtomWithIdx(d_idx).SetAtomicNum(z)
            rw.GetAtomWithIdx(d_idx).SetIsotope(0)
            donor = next(iter(rw.GetAtomWithIdx(d_idx).GetNeighbors()))
            # The temporary single bond represents donor -> metal as a normal CIP edge. Preserve the
            # corresponding Lewis electron count while RDKit ranks that edge: N -> N+, C- -> C, etc.
            if donated and donor.GetIdx() not in charged:
                donor.SetFormalCharge(donor.GetFormalCharge() + 1)
                charged.add(donor.GetIdx())
            donor.SetNoImplicit(True)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol, _STEREO_SANITIZE, catchErrors=True)
    # Perception and absolute labelling are separate native operations. Legacy cleanup misses some
    # para-stereocentres and its approximate CIP properties depend on RDKit's process-global stereo mode.
    Chem.FindPotentialStereo(mol, cleanIt=True, flagPossible=False)
    point_codes = {**_point_cip_codes(mol, atom_centers), **(point_codes or {})}
    parts = []
    for idx in atom_centers:
        a = mol.GetAtomWithIdx(idx)
        code = point_codes.get(idx) or {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW",
        }.get(a.GetChiralTag())
        if code:  # an unresolved centre (an allene axis RDKit can't set) is dropped, never given a '?' tag
            parts.append(f"{a.GetSymbol()}{idx}:{code}")
    for bidx in bond_centers:
        b = mol.GetBondWithIdx(bidx)
        tag = bond_stereo_code(mol, bidx)
        if tag:
            begin, end = b.GetBeginAtom(), b.GetEndAtom()
            parts.append(f"{begin.GetSymbol()}{begin.GetIdx()}={end.GetSymbol()}{end.GetIdx()}:{tag}")
    for i, j in atrop_centers:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetStereo() not in ATROP_STEREO:
            continue
        Chem.AssignCIPLabels(mol, bondsToLabel=[bond.GetIdx()])
        code = bond.GetPropsAsDict().get("_CIPCode")
        if code not in {"M", "P"}:
            raise ValueError(f"RDKit could not assign M/P to atropisomer bond {i}-{j}")
        first, second = sorted((i, j))
        left, right = mol.GetAtomWithIdx(first), mol.GetAtomWithIdx(second)
        parts.append(f"{left.GetSymbol()}{first}-{right.GetSymbol()}{second}:{code}")
    return ",".join(parts)


_CIP_CACHE = {}
_CIP_CACHE_MAX = 256


def _point_cip_codes(mol, centers):
    """Return absolute R/S or pseudoasymmetric r/s labels RDKit assigns on the current full graph.

    Memoised on the graph's content (atoms, bonds and the centre list, in index order), because the same
    coordination graph is asked about many times and RDKit's CIP labeller goes superlinear on a metal-closed
    ring. Chiral tags must be part of the key: a centre's label depends on every other centre's configuration,
    not just which atoms are bonded to which.
    """
    centers = list(centers)
    try:
        key = cip_cache_key(mol, centers)
    except RuntimeError:
        # An unsanitized/malformed graph (e.g. implicit valence never calculated) fails the same way
        # AssignCIPLabels would below; skip the cache and let the ordinary except path handle it.
        key = None
    if key is not None:
        cached = _CIP_CACHE.get(key)
        if cached is not None:
            return dict(cached)
    probe = Chem.Mol(without_zero_bonds(mol))
    for idx in centers:
        atom = probe.GetAtomWithIdx(idx)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    try:
        with rdBase.BlockLogs():
            Chem.AssignCIPLabels(probe, atomsToLabel=centers)
    except RuntimeError:
        result = {}
    else:
        result = {
            idx: code
            for idx in centers
            if (code := probe.GetAtomWithIdx(idx).GetPropsAsDict().get("_CIPCode")) in _POINT_CIP
        }
    if key is not None:
        if len(_CIP_CACHE) >= _CIP_CACHE_MAX:
            _CIP_CACHE.clear()
        _CIP_CACHE[key] = dict(result)
    return dict(result)


def _rdkit_3d_point_capable(atom):
    """Return whether RDKit can assign this capped atom's tetrahedral tag from 3D."""
    degree = sum(bond.GetBondType() != Chem.BondType.ZERO for bond in atom.GetBonds())
    total = degree + atom.GetTotalNumHs()
    return (
        degree >= _MIN_POINT_BRANCHES
        and total <= 4  # noqa: PLR2004  four tetrahedral carriers
        and (total == 4 or atom.GetAtomicNum() in _THREE_COORDINATE_3D)  # noqa: PLR2004
    )


def _tetrahedral_centres(mol):
    """Return the atoms RDKit's potential-stereo perception flags as tetrahedral centres."""
    return {
        element.centeredOn
        for element in Chem.FindPotentialStereo(mol)
        if element.type == Chem.StereoType.Atom_Tetrahedral
    }


def _distinct_carriers(work, ranks, index, count=4):
    """Return whether an atom carries `count` substituents, hydrogens included, in distinct graph classes."""
    atom = work.GetAtomWithIdx(index)
    classes = [ranks[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()]
    classes.extend([-1] * atom.GetTotalNumHs())
    return len(classes) == len(set(classes)) == count


def _free_graph(mol, metals):
    """Return the metal-free ligand graph with any E/Z reference the strip orphaned repaired or dropped."""
    free = ligand_graph(mol, metals)
    free.UpdatePropertyCache(strict=False)
    repair_bond_stereo(free)
    return free


def _bridgehead_partners(free, atom):
    """Return ``(partner, arms, ends)`` for each nearest atom joined to `atom` by three disjoint bridges.

    ``arms`` are `atom`'s neighbours and ``ends`` the partner's neighbours on the same bridges, in the same order.
    """
    arms = [neighbor.GetIdx() for neighbor in free.GetAtomWithIdx(atom).GetNeighbors()]
    cut = Chem.RWMol(free)
    for arm in arms:
        cut.RemoveBond(atom, arm)
    distances = Chem.GetDistanceMatrix(cut)
    found = []
    for partner in range(free.GetNumAtoms()):
        if partner == atom or partner in arms or any(distances[arm][partner] >= DISCONNECTED for arm in arms):
            continue
        paths = [Chem.GetShortestPath(cut, arm, partner) for arm in arms]
        inner = [set(path[:-1]) for path in paths]
        if any(left & right for left, right in itertools.combinations(inner, 2)):
            continue
        found.append((sum(map(len, paths)), partner, arms, [path[-2] for path in paths]))
    shortest = min((row[0] for row in found), default=None)
    return [row[1:] for row in found if row[0] == shortest]


def _with_hand(mol, atom, order, tag):
    """Tag `atom` so that its neighbours listed in `order` turn as `tag` names."""
    bonded = [bond.GetOtherAtomIdx(atom) for bond in mol.GetAtomWithIdx(atom).GetBonds()]
    rank = [bonded.index(neighbor) for neighbor in order]
    swaps = sum(left > right for left, right in itertools.combinations(rank, 2))
    mol.GetAtomWithIdx(atom).SetChiralTag(mirror_tag(tag) if swaps % 2 else tag)


def _cage_fixed_nitrogens(free, centres):
    """Return the bridgehead nitrogens among `centres` whose hand is no stereo element of their own.

    A bridgehead N cannot invert alone. Its three bridges end at a partner bridgehead and, in any buildable
    cage, both point out of it, so bridge by bridge the partner turns the opposite way. The pair carries one
    element, the cage's hand: a partner other than N holds it as its own centre, and it is none at all when the
    mirrored cage is the same molecule.
    """
    fixed = set()
    for atom in centres:
        arms = [neighbor.GetIdx() for neighbor in free.GetAtomWithIdx(atom).GetNeighbors()]
        if free.GetAtomWithIdx(atom).GetAtomicNum() != 7 or len(arms) != 3:  # noqa: PLR2004  a tertiary N
            continue
        if any(free.GetBondBetweenAtoms(*pair) for pair in itertools.combinations(arms, 2)):
            continue  # a three-ring N, not a bridgehead: it inverts slowly on its own
        partners = _bridgehead_partners(free, atom)
        if any(free.GetAtomWithIdx(partner).GetAtomicNum() != 7 for partner, _arms, _ends in partners):  # noqa: PLR2004
            fixed.add(atom)
            continue
        for partner, arms, ends in partners[:1]:
            if free.GetAtomWithIdx(partner).GetDegree() != 3:  # noqa: PLR2004  the ends must be its only carriers
                continue
            cage, mirror = Chem.Mol(free), Chem.Mol(free)
            for graph, tag in (
                (cage, Chem.ChiralType.CHI_TETRAHEDRAL_CW),
                (mirror, Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
            ):
                _with_hand(graph, atom, arms, tag)
                _with_hand(graph, partner, ends, mirror_tag(tag))
            if Chem.MolToSmiles(cage) == Chem.MolToSmiles(mirror):
                fixed.update((atom, partner))
    return fixed


def _point_capability(mol, work, cap_to_metal=(), exclude=()):
    """Return supported and unmeasurable point centres on the authoritative graph."""
    cap_to_metal = dict(cap_to_metal)
    metals = set(exclude) | {record[3] for record in cap_to_metal.values()}
    stated = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in metals and atom.GetChiralTag() in _TETRAHEDRAL_TAGS
    }
    cap_donors = {
        cap: work.GetAtomWithIdx(cap).GetNeighbors()[0].GetIdx()
        for cap in cap_to_metal
        if work.GetAtomWithIdx(cap).GetDegree() == 1
    }
    _closed, rings = _metal_closed_rings(mol, metals)
    chelated_nitrogen = {
        cap_donors[cap]
        for cap, record in cap_to_metal.items()
        if cap in cap_donors and any({cap_donors[cap], record[3]} <= atoms for atoms, _bonds in rings)
    }
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False))
    # A neutral N inverts unless RDKit's own rule holds it on the metal-free graph (a bridgehead or three-ring N)
    # with three distinct carriers and, for a bridgehead, a hand the cage leaves free. The coordination
    # convention also retains a bound amine hand for a metal-closed chelate with four distinct carriers. The
    # graph cannot infer inversion kinetics further; stereo='free' is the explicit opt-out.
    free = _free_graph(mol, metals)
    free_ranks = list(Chem.CanonicalRankAtoms(free, breakTies=False, includeChirality=False))
    held = {index for index in _tetrahedral_centres(free) if _distinct_carriers(free, free_ranks, index, 3)}
    held -= _cage_fixed_nitrogens(free, held)
    held.update(index for index in chelated_nitrogen if _distinct_carriers(work, ranks, index))
    potential = _tetrahedral_centres(mol) | {index for index in _tetrahedral_centres(work) if index < mol.GetNumAtoms()}
    donors = list(cap_donors.values())
    bridged = {idx for idx in set(donors) if donors.count(idx) >= _MIN_BRIDGE_METALS}
    potential.update(bridged)
    potential.difference_update(metals)
    cap_created = set(donors)
    # A temporary metal cap is only a fourth carrier. It cannot distinguish symmetry-equivalent ligand arms.
    potential.difference_update(
        index
        for index in potential - stated - bridged
        if index in cap_created and not _distinct_carriers(work, ranks, index)
    )
    potential.difference_update(
        index
        for index in potential - held - bridged
        if work.GetAtomWithIdx(index).GetAtomicNum() == 7  # noqa: PLR2004  nitrogen
        and work.GetAtomWithIdx(index).GetFormalCharge() == 0
    )
    # Carriers only the drawn Lewis form tells apart, such as a phenyl drawn as a quinoid carbanion, are one
    # group: bond orders and charges move no atom, so the centre has no hand, whether tagged or perceived.
    flat = flat_ranks(work)
    lewis_only = {
        index
        for index in stated | potential
        if len({flat[n.GetIdx()] for n in work.GetAtomWithIdx(index).GetNeighbors()})
        < len({ranks[n.GetIdx()] for n in work.GetAtomWithIdx(index).GetNeighbors()})
    }
    stated -= lewis_only
    potential -= lewis_only
    candidates = stated | potential
    supported = {
        idx for idx in candidates if idx < work.GetNumAtoms() and _rdkit_3d_point_capable(work.GetAtomWithIdx(idx))
    }
    impossible = sorted(stated - supported)
    if impossible:
        raise ValueError(
            f"point stereo at atom(s) {impossible} cannot be measured from 3D by RDKit; "
            "remove the tag or use a degree-four representation"
        )
    return supported, potential - supported


def _graft_point_tags(full, work, centers, cap_to_metal):
    """Copy enumerated point tags from the capped graph onto the coordinated graph."""
    centers = set(centers)
    for atom in work.GetAtoms():
        if atom.GetIdx() >= full.GetNumAtoms() or atom.GetIdx() not in centers:
            continue
        tag = atom.GetChiralTag()
        for cap_idx, (_z, _donated, mirrored, _metal) in cap_to_metal.items():
            if mirrored and work.GetBondBetweenAtoms(atom.GetIdx(), cap_idx) is not None:
                tag = mirror_tag(tag)
        full.GetAtomWithIdx(atom.GetIdx()).SetChiralTag(tag)


def _label_item(part):
    """Parse one indexed ligand-stereo label into its selector fields."""
    if match := _POINT_ITEM.fullmatch(part):
        symbol, index, code = match.groups()
        return "point", (symbol,), code, part, f"{index}{code}"
    if match := _BOND_ITEM.fullmatch(part):
        left, i, right, j, code = match.groups()
        return "bond", tuple(sorted((left, right))), code, part, f"{i}={j}:{code}"
    if match := _AXIS_ITEM.fullmatch(part):
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
    if selector in _POINT_CIP | {"CW", "CCW"}:
        kind, wanted = "point", selector
    elif selector in {"E", "Z"}:
        kind, wanted = "bond", selector
    elif selector in {"M", "P"}:
        kind, wanted = "axis", selector
    elif match := re.fullmatch(r"([A-Z][a-z]?):(R|S|r|s|CW|CCW)", selector):
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


def clear_atrop(mol):
    """Clear native atropisomer tags and their signaling wedges; return the axis atom pairs."""
    axes = set()
    for bond in mol.GetBonds():
        if bond.GetStereo() not in ATROP_STEREO:
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


def atrop_code(mol, bond):
    """Return RDKit's sequence-rule descriptor for one assigned atropisomer bond."""
    Chem.AssignCIPLabels(mol, bondsToLabel=[bond.GetIdx()])
    return bond.GetPropsAsDict().get("_CIPCode")


def _apply_atrop_stereo(mol, axes):
    """Apply absolute M/P labels as RDKit native atrop bond tags."""
    clear_atrop(mol)
    for pair, target in axes.items():
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None:
            raise ValueError(f"could not locate atropisomer bond {pair}")
        for tag in ATROP_STEREO:
            candidate = Chem.Mol(mol)
            assigned = candidate.GetBondBetweenAtoms(*pair)
            assigned.SetStereo(tag)
            Chem.CleanupAtropisomers(candidate)
            if assigned.GetStereo() in ATROP_STEREO and atrop_code(candidate, assigned) == target:
                bond.SetStereo(tag)
                break
        else:
            raise ValueError(f"could not apply {target} atropisomer stereo on bond {pair}")


def clear_ez(mol, pairs=None):
    """Clear selected double-bond stereo and adjacent slash bonds; return whether anything changed."""
    if pairs is not None and not pairs:
        return False
    changed = False
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if pairs is not None and pair not in pairs:
            continue
        changed |= bond.GetStereo() != Chem.BondStereo.STEREONONE
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    directions = (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)
    for bond in mol.GetBonds():
        if bond.GetBondDir() in directions:
            bond.SetBondDir(Chem.BondDir.NONE)
            changed = True
    if pairs is not None:
        Chem.SetDoubleBondNeighborDirections(mol)
    return changed


def assign_atrop_from_3d(mol, atrop_centers, *, required=True):
    """Assign selected native atrop bonds from 3D, returning those with a perceived hand."""
    if not atrop_centers:
        return []
    probe = Chem.Mol(mol)
    for pair in atrop_centers:
        probe.GetBondBetweenAtoms(*pair).SetStereo(Chem.BondStereo.STEREOATROPCW)
    block = Chem.MolToMolBlock(probe, confId=probe.GetConformer().GetId(), includeStereo=True)
    perceived = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if perceived is None or not perceived.GetConformer().Is3D():
        raise ValueError("RDKit could not perceive atropisomer stereo from the 3D MolBlock")
    assigned = []
    for pair in atrop_centers:
        tag = perceived.GetBondBetweenAtoms(*pair).GetStereo()
        if tag not in ATROP_STEREO:
            if required:
                raise ValueError(f"RDKit could not perceive atropisomer axis {pair} from 3D coordinates")
            continue
        bond = mol.GetBondBetweenAtoms(*pair)
        bond.SetStereo(tag)
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
        assigned.append(pair)
    return assigned


def _metal_closed_rings(mol, metals):
    """Return the single-bonded coordination graph and its metal-containing rings."""
    up = Chem.RWMol(mol)
    for bond in up.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
    up = up.GetMol()
    up.UpdatePropertyCache(strict=False)
    # Stereo queries use smallest rings. A DFS cycle basis can omit a metal-closed path after renumbering.
    Chem.GetSymmSSSR(up)
    rings = [
        (set(atoms), set(bonds))
        for atoms, bonds in zip(up.GetRingInfo().AtomRings(), up.GetRingInfo().BondRings(), strict=True)
        if set(metals) & set(atoms)
    ]
    return up, rings


def _coordination_atrop_bonds(mol, metals, work):
    """Return native-eligible, ortho-blocked axes inside a metal-closed chelate."""
    if not metals:
        return []
    _closed, rings = _metal_closed_rings(mol, metals)
    ring_bonds = set().union(*(bonds for _atoms, bonds in rings))
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=True))
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
        if _native_atrop_candidate(work, (i, j), ranks):
            axes.append(tuple(sorted((i, j))))
    return axes


def _native_atrop_candidate(mol, pair, ranks):
    """Return whether RDKit accepts an ortho-blocked single bond as a native atrop axis, given atom `ranks`."""
    bond = mol.GetBondBetweenAtoms(*pair)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return False
    sides = [
        [ranks[n.GetIdx()] for n in mol.GetAtomWithIdx(end).GetNeighbors() if n.GetIdx() not in pair] for end in pair
    ]
    if any(len(side) not in (1, 2) or len(side) != len(set(side)) for side in sides):
        return False
    ortho = [
        neighbor for end in pair for neighbor in mol.GetAtomWithIdx(end).GetNeighbors() if neighbor.GetIdx() not in pair
    ]
    if any(
        sum(adjacent.GetAtomicNum() > 1 for adjacent in atom.GetNeighbors()) < _MIN_BLOCKED_ORTHO_CONNECTIONS
        for atom in ortho
    ):
        return False  # an unsubstituted ortho C or N does not block rotation merely because it has no hydrogen
    probe = Chem.Mol(mol)
    candidate = probe.GetBondBetweenAtoms(*pair)
    candidate.SetStereo(Chem.BondStereo.STEREOATROPCW)
    Chem.CleanupAtropisomers(probe)
    return candidate.GetStereo() == Chem.BondStereo.STEREOATROPCW


def coordination_locked_double_bonds(mol, metals):
    """Return double bonds whose E/Z the coordination fixes: endocyclic in a ring closed through the metal.

    Such a bond has one buildable geometry, decided by the coordination isomer (the polyhedron path's job).
    Enumerating both E and Z is a phantom: the wrong hand forces a bite the chelate cannot span, and the
    pipeline burns seeds relaxing it into broken bonds.

    RDKit ignores dative M-donor bonds in ring perception, so the metal-closed ring is invisible natively;
    upgrade the datives to single to reveal it. A double bond still in a ring once the metal is removed is a
    genuine organic ring bond, already handled by RDKit, and is left alone; only a bond cyclic because of the
    metal is locked here.
    """
    metals = set(metals)
    if not metals:
        return set()
    faces = [
        site
        for metal in metals
        for site in haptic_sites(mol, [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()])
        if len(site) > 1
    ]
    haptic = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for site in faces
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in site and bond.GetEndAtomIdx() in site
    }
    # A diene face's central bond, drawn double in its metallacyclopentene form, is the face's s-cis/s-trans class.
    central = {frozenset(path[1:3]) for site in faces if (path := torsion_path(mol, site)) is not None}
    closed, metal_rings = _metal_closed_rings(mol, metals)
    ring_stereo = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for element in Chem.FindPotentialStereo(closed, cleanIt=False, flagPossible=True)
        if element.type == Chem.StereoType.Bond_Double
        for bond in (closed.GetBondWithIdx(element.centeredOn),)
    }
    free = ligand_graph(mol, metals)  # the metal-free graph: which double bonds are still cyclic without it?
    Chem.FastFindRings(free)
    locked = set()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pair = frozenset((a, c))
        if pair in central:
            locked.add(pair)
            continue
        if pair in haptic:
            continue  # an eta2 C=C keeps its ligand E/Z; coordination chooses a face, not a bond geometry
        in_metal_ring = any({a, c} <= atoms for atoms, _bonds in metal_rings)
        fb = free.GetBondBetweenAtoms(a, c)
        if (
            in_metal_ring and pair not in ring_stereo and not (fb is not None and fb.IsInRing())
        ):  # cyclic only because of the metal
            locked.add(pair)
    return locked


def coordination_locked_centres(mol, metals):
    """Return the donor stereocentres that exist only while bound: labile in the free ligand, fixed by coordination.

    RDKit's potential-stereo rule on the metal-free graph decides lability: a free amine N inverts, so it is no
    stereocentre there, while a phosphine P or a ring carbon is. A locked hand belongs to the coordination
    arrangement, as a metal-closed ring's E/Z does (`coordination_locked_double_bonds`), yet no arrangement fixes
    it: a chelate ring can twist far enough to reverse it.
    """
    metals = set(metals)
    if not metals:
        return set()
    donors = {n.GetIdx() for m in metals for n in mol.GetAtomWithIdx(m).GetNeighbors()} - metals
    return (point_centres(mol, exclude=metals) & donors) - _tetrahedral_centres(_free_graph(mol, metals))


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
    wb.SetStereoAtoms(nb_b, nb_e)
    wb.SetStereo(Chem.BondStereo.STEREOCIS)
    return True


def _free_double_pairs(mol, metal_neighbors):
    """Return the double bonds RDKit reads as potential E/Z once every excluded metal is disconnected."""
    free = Chem.RWMol(mol)
    for donor, metals in metal_neighbors.items():
        for metal in metals:
            free.RemoveBond(donor, metal)
    free = free.GetMol()
    free.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(free)
    repair_bond_stereo(free)
    return {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for element in Chem.FindPotentialStereo(free)
        if element.type == Chem.StereoType.Bond_Double
        for bond in (free.GetBondWithIdx(element.centeredOn),)
    }


def _stereogenic_bridges(mol, metal_neighbors):
    """Return the donors bridging two or more metals that RDKit still ranks as CIP point stereocentres."""
    bridges = set()
    for donor, metals in metal_neighbors.items():
        if len(metals) < _MIN_BRIDGE_METALS:
            continue
        probe = Chem.Mol(mol)
        probe.GetAtomWithIdx(donor).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        if donor in _point_cip_codes(probe, [donor]):
            bridges.add(donor)
    return bridges


def _donor_caps(mol, exclude, metal_neighbors):
    """Return the cap element each metal-bound donor needs to keep its ligand stereo once the metal is cut.

    A dummy atom (0) stands in for the metal as the second E/Z reference of a double-bond end; a hydrogen (1)
    keeps a point centre's fourth carrier. Donors without an entry need no cap.
    """
    free_double_pairs = _free_double_pairs(mol, metal_neighbors)
    stereogenic_bridges = _stereogenic_bridges(mol, metal_neighbors)
    caps = {}
    for nb, metals in metal_neighbors.items():
        if not metals:
            continue
        donor = mol.GetAtomWithIdx(nb)
        ligand_bonds = [b for b in donor.GetBonds() if b.GetOtherAtomIdx(nb) not in exclude]
        double_bonds = [b for b in ligand_bonds if b.GetBondType() == Chem.BondType.DOUBLE]
        heavy_ligand_bonds = [b for b in ligand_bonds if b.GetOtherAtom(donor).GetAtomicNum() > 1]
        # C=[NH]->M needs M as its second explicit E/Z reference even after H is materialized for SMILES.
        # An N-substituted imine already has that ligand-side reference and must not be re-ranked by M.
        double_end = (
            len(double_bonds) == 1
            and len(heavy_ligand_bonds) == 1
            and frozenset((double_bonds[0].GetBeginAtomIdx(), double_bonds[0].GetEndAtomIdx())) in free_double_pairs
        )
        sigma_only = all(b.GetBondType() == Chem.BondType.SINGLE for b in ligand_bonds)
        # Two identical H rule out tetrahedral chirality; an H cap makes RDKit misclassify bracket `[PH3]`.
        point_cap = (
            (len(metals) < _MIN_BRIDGE_METALS or nb in stereogenic_bridges)
            and not donor.GetIsAromatic()
            and donor.GetDegree() + donor.GetTotalNumHs() >= _MIN_POINT_BRANCHES
            and (donor.GetHybridization() == Chem.HybridizationType.SP3 or sigma_only)
            and donor.GetTotalNumHs() <= 1
        )
        if double_end or point_cap:
            caps[nb] = 0 if double_end else 1
    return caps


def _build_enumeration_graph(mol, exclude):
    """Disconnect each metal and single-bond-cap donors whose ligand stereo needs that neighbour.

    Returns ``(work, cap_to_metal)``: cap index -> ``(metal atomic number, was donor->metal dative,
    replacement changed parity, metal index)``.
    With no `exclude` there is nothing to disconnect, so the contact-free `mol` is returned.
    """
    mol = without_zero_bonds(mol)
    if not exclude:
        return mol, {}
    # Disconnect each metal first: a metal-bound donor is a stereocentre only while bound, so RDKit would
    # enumerate hands the surrogate cannot hold.
    cap_to_metal = {}
    metal_neighbors = {
        atom.GetIdx(): [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in exclude] for atom in mol.GetAtoms()
    }
    caps = _donor_caps(mol, exclude, metal_neighbors)
    work = Chem.RWMol(mol)
    for mi in sorted(exclude):
        z_metal = mol.GetAtomWithIdx(mi).GetAtomicNum()
        donors = [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]
        # A haptic atom belongs to a pi face, not one sigma-donor point centre; capping it creates a
        # phantom R/S centre when RDKit parses a fully dative Cp ring as locally sp3.
        haptic = {d for site in haptic_sites(mol, donors) if len(site) > 1 for d in site}
        for nb in donors:
            bond = mol.GetBondBetweenAtoms(mi, nb)
            donated = bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() == nb
            replacement_mirrors = bond_removal_mirrors(work.GetAtomWithIdx(nb), mi, degrees=(3, 4))
            removal_mirrors = bond_removal_mirrors(work.GetAtomWithIdx(nb), mi)
            remove_bond(work, mi, nb)  # re-base the donor's tag onto the stripped order; `graft` inverts it
            if nb in haptic or nb not in caps:
                continue
            if replacement_mirrors != removal_mirrors:
                atom = work.GetAtomWithIdx(nb)
                atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
            cap = Chem.Atom(caps[nb])
            cap.SetNoImplicit(True)
            d = work.AddAtom(cap)
            same_element = [m for m in metal_neighbors[nb] if mol.GetAtomWithIdx(m).GetAtomicNum() == z_metal]
            isotope = 2 + z_metal + _ISOTOPE_ELEMENT_STRIDE * same_element.index(mi)
            work.GetAtomWithIdx(d).SetIsotope(isotope)
            # This is a disposable RDKit stereo graph, not the public chemical graph. RDKit excludes a
            # donor-originating dative edge from tetrahedral and alkene stereo, so the cap must be a normal
            # neighbour here. ``cap_to_metal`` retains the real bond kind for grafting and CIP charge.
            work.AddBond(nb, d, Chem.BondType.SINGLE)
            for original in mol.GetAtomWithIdx(nb).GetBonds():
                if original.GetBondType() != Chem.BondType.DOUBLE or original.GetOtherAtomIdx(nb) in exclude:
                    continue
                copied = work.GetBondBetweenAtoms(original.GetBeginAtomIdx(), original.GetEndAtomIdx())
                refs = tuple(d if ref == mi else ref for ref in original.GetStereoAtoms())
                if mi in original.GetStereoAtoms() and len(refs) == 2:  # noqa: PLR2004
                    copied.SetStereoAtoms(*refs)
                    copied.SetStereo(original.GetStereo())
            work.GetAtomWithIdx(nb).SetNoImplicit(True)
            cap_to_metal[d] = (z_metal, donated, replacement_mirrors, mi)
            for conf in work.GetConformers():
                conf.SetAtomPosition(d, conf.GetAtomPosition(mi))
    work = work.GetMol()
    Chem.SanitizeMol(work, _STEREO_SANITIZE, catchErrors=True)
    # The strip above can orphan a C=N whose stereo reference atom was the metal, and a flagged bond with no
    # references makes `FindPotentialStereo` below raise ("only can support 2 stereo neighbors"). The
    # tolerant sanitize happens to scrub most of them, but that is luck rather than a contract.
    repair_bond_stereo(work)
    apply_encoded_bond_stereo(work)
    return work, cap_to_metal


def _mask_skipped_stereo(work, potential, skip_points, skip_bonds):
    """Mark skipped centres as handled on the temporary graph so RDKit does not expand them anyway."""
    for element in potential:
        if element.specified != Chem.StereoSpecified.Unspecified:
            continue
        if element.type == Chem.StereoType.Atom_Tetrahedral and element.centeredOn in skip_points:
            work.GetAtomWithIdx(element.centeredOn).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        elif element.type == Chem.StereoType.Bond_Double and _skip_bond(work, element.centeredOn, skip_bonds):
            bond = work.GetBondWithIdx(element.centeredOn)
            controls = list(element.controllingAtoms)
            left = next((atom for atom in controls[:2] if atom < work.GetNumAtoms()), None)
            right = next((atom for atom in controls[2:] if atom < work.GetNumAtoms()), None)
            if left is not None and right is not None:
                bond.SetStereoAtoms(left, right)
                bond.SetStereo(Chem.BondStereo.STEREOCIS)


def _skip_bond(work, bond_index, skip_bonds):
    """Return whether a boolean or atom-pair selection masks one double bond."""
    if isinstance(skip_bonds, bool):
        return skip_bonds
    bond = work.GetBondWithIdx(bond_index)
    return frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in skip_bonds


def _cumulene_component(work, bond_index):
    """Return the maximal connected component of cumulated double bonds."""
    component, frontier = {bond_index}, [bond_index]
    while frontier:
        bond = work.GetBondWithIdx(frontier.pop())
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for adjacent in atom.GetBonds():
                if adjacent.GetBondType() == Chem.BondType.DOUBLE and adjacent.GetIdx() not in component:
                    component.add(adjacent.GetIdx())
                    frontier.append(adjacent.GetIdx())
    return component


def _cumulene_terminal_controls(work, component, potential_double):
    """Return each cumulene terminal and its endpoint-normalized controlling atoms, or ``None``."""
    incidence = {}
    for index in component:
        bond = work.GetBondWithIdx(index)
        for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
            incidence.setdefault(atom, []).append(index)
    terminals = [atom for atom, bonds in incidence.items() if len(bonds) == 1]
    if len(terminals) != 2:  # noqa: PLR2004
        return None
    out = []
    for atom in terminals:
        index = incidence[atom][0]
        info = potential_double.get(index)
        if info is None:
            return None
        bond = work.GetBondWithIdx(index)
        controls = list(info.controllingAtoms)
        pair = controls[:2] if bond.GetBeginAtomIdx() == atom else controls[2:]
        if pair[0] == pair[1]:
            return None
        out.append((atom, tuple(sorted(pair))))
    return tuple(sorted(out))


def _potential_ez_signature(work, pair):
    """Return the resonance-sensitive graph signature of one potential E/Z element."""
    probe = Chem.Mol(work)
    Chem.RemoveStereochemistry(probe)
    bond = probe.GetBondBetweenAtoms(*pair)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return None
    bond_index = bond.GetIdx()
    potential = {
        element.centeredOn: element
        for element in Chem.FindPotentialStereo(probe)
        if element.type == Chem.StereoType.Bond_Double
    }
    component = _cumulene_component(probe, bond_index)
    element = potential.get(bond_index)
    controls = _cumulene_terminal_controls(probe, component, potential)
    if element is None or bond_index != min(component) or controls is None:
        return None
    edges = tuple(
        sorted(
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for index in component
            for bond in (probe.GetBondWithIdx(index),)
        )
    )
    if len(component) > 1:
        return edges, controls

    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    refs = []
    for atom, other in ((begin, end), (end, begin)):
        choices = sorted(
            neighbor.GetIdx() for neighbor in probe.GetAtomWithIdx(atom).GetNeighbors() if neighbor.GetIdx() != other
        )
        if not choices:
            return None
        refs.append(choices[0])
    bond.SetStereoAtoms(*refs)
    bond.SetStereo(Chem.BondStereo.STEREOCIS)
    Chem.AssignCIPLabels(probe, bondsToLabel=[bond_index])
    code = bond.GetPropsAsDict().get("_CIPCode")
    return (edges, code) if code in {"E", "Z"} else None


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
    work = Chem.Mol(work)  # masking skipped elements below must never mutate the caller's graph
    forced_atrop = {tuple(sorted(pair)) for pair in include_atrop}
    if include == "all":
        forced_atrop.update(clear_atrop(work))
        Chem.RemoveStereochemistry(work)
    else:
        for atom in include:
            work.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    # A C=N / C=C whose E/Z the coordination fixes must not be enumerated: the metal closes the ring, so only
    # one geometry exists and the other embeds as a strained impossibility.
    locked = coordination_locked_double_bonds(mol, exclude)
    locked = {fb for fb in locked if _lock_double_bond(work, fb)}  # keep only the ones we could actually pin
    supported_points, unsupported_points = _point_capability(mol, work, cap_to_metal, exclude)
    potential = list(Chem.FindPotentialStereo(work))
    work_points = {element.centeredOn for element in potential if element.type == Chem.StereoType.Atom_Tetrahedral}
    _mask_skipped_stereo(work, potential, set(skip_points) | (work_points - supported_points), skip_bonds)
    potential = list(Chem.FindPotentialStereo(work))
    potential_double = {e.centeredOn: e for e in potential if e.type == Chem.StereoType.Bond_Double}
    stated_pairs = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() != Chem.BondStereo.STEREONONE
        and len(bond.GetStereoAtoms()) == 2  # noqa: PLR2004
    } | set(encoded_bond_stereo(mol))
    stated_bonds = {bond.GetIdx() for pair in stated_pairs if (bond := work.GetBondBetweenAtoms(*pair)) is not None}
    stable_bonds = set(_resonance_stable_ez(work, potential_double, stated_bonds, structural=True))
    enumerable = []  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
    for e in potential:
        if e.specified != Chem.StereoSpecified.Unspecified:
            continue
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            if e.centeredOn in supported_points and e.centeredOn not in skip_points:
                enumerable.append(e)
        elif e.type == Chem.StereoType.Bond_Double:  # skip a double bond the coordination has already locked
            wb = work.GetBondWithIdx(e.centeredOn)
            pair = frozenset((wb.GetBeginAtomIdx(), wb.GetEndAtomIdx()))
            component = _cumulene_component(work, e.centeredOn)
            if (
                not _skip_bond(work, e.centeredOn, skip_bonds)
                and pair not in locked
                and e.centeredOn in stable_bonds
                and e.centeredOn == min(component)
                and _cumulene_terminal_controls(work, component, potential_double) is not None
            ):
                enumerable.append(e)
    atrop = set() if skip_atrop else forced_atrop
    atrop = sorted(
        pair
        for pair in atrop
        if (bond := work.GetBondBetweenAtoms(*pair)) is not None and bond.GetStereo() == Chem.BondStereo.STEREONONE
    )
    unresolved_points = unsupported_points - set(skip_points)
    return work, cap_to_metal, locked, enumerable, atrop, len(unresolved_points)


def point_centres(mol, exclude=()):
    """Return atom indices that can carry tetrahedral ligand stereo on the metal-free enumeration graph."""
    exclude = set(exclude)
    work, caps = _build_enumeration_graph(mol, exclude)
    supported, _unsupported = _point_capability(mol, work, caps, exclude)
    return supported


def unassigned_centres(mol, exclude=()):
    """Return the unspecified stereo elements as atom-index tuples: ``(atom,)``, or ``(i, j)`` for a double bond.

    The cheap predicate behind `enumerate_unassigned`: what would be expanded, without expanding it. A caller
    that embeds one species (`Isomer`) uses it to refuse to pool two enantiomers silently.
    """
    work, _caps, _locked, elements, atrop, _unsupported = _unassigned_elements(mol, exclude)
    out = []
    for e in elements:  # `work` only appends caps, so every index here is a real atom of `mol`
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            out.append((e.centeredOn,))
        else:
            b = work.GetBondWithIdx(e.centeredOn)
            out.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    return [*out, *atrop]


def _stereo_centres(mol, n_real):
    """Return defined point, E/Z, and native atrop centres."""
    points = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() < n_real and atom.GetChiralTag() in _TETRAHEDRAL_TAGS
    ]
    doubles = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    axes = [
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
        if bond.GetStereo() in ATROP_STEREO
    ]
    return points, doubles, axes


def _ez_description(form, pair, structural):
    """Return a double bond's E/Z-defining graph signature, with its stereo code unless `structural`."""
    graph = _potential_ez_signature(form, pair)
    if structural or graph is None:
        return graph
    bond = form.GetBondBetweenAtoms(*pair)
    code = bond_stereo_code(form, bond.GetIdx())
    return (graph, code) if code else None


def _resonance_stable_ez(work, bond_centers, stated, *, structural=False):
    """Keep inferred E/Z whose label or defining graph is invariant across RDKit resonance forms.

    A delocalised charge can swap the CIP-leading path without moving an atom, turning one geometry from E
    into Z. It can also move a coordinate-free double bond or extend it into a cumulene. Such labels are
    Lewis-form artefacts. Explicit input is authoritative. Independent conjugated groups multiply a fragment's
    forms, so the search is bounded; past the bound, or where RDKit cannot enumerate a fragment, a bond in a
    conjugated group abstains with a warning. A bond outside every group keeps its order in every form.
    """
    keep = set(stated)
    candidates = [idx for idx in bond_centers if idx not in keep]
    if not candidates:
        return list(bond_centers)
    unproven = set()
    mappings = []
    fragments = Chem.GetMolFrags(work, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mappings)
    for fragment, atoms in zip(fragments, mappings, strict=True):
        local = {original: idx for idx, original in enumerate(atoms)}
        selected = {}
        for bond_idx in candidates:
            bond = work.GetBondWithIdx(bond_idx)
            pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if pair[0] in local:
                selected[bond_idx] = (local[pair[0]], local[pair[1]])
        if not selected:
            continue
        try:
            expected = {idx: _ez_description(fragment, pair, structural) for idx, pair in selected.items()}
            stable = {idx for idx, value in expected.items() if value is not None}
            forms = Chem.ResonanceMolSupplier(fragment, maxStructs=_RESONANCE_EZ_CAP + 1)
            forms.SetNumThreads(1)
            for count, form in enumerate(forms, 1):
                if count > _RESONANCE_EZ_CAP:
                    capped = {
                        idx
                        for idx in stable
                        if forms.GetBondConjGrpIdx(fragment.GetBondBetweenAtoms(*selected[idx]).GetIdx()) >= 0
                    }
                    stable -= capped
                    unproven |= capped
                    break
                if form is None:
                    continue
                if not structural:
                    Chem.RemoveStereochemistry(form)
                    Chem.AssignStereochemistryFrom3D(
                        form,
                        confId=form.GetConformer().GetId(),
                        replaceExistingTags=True,
                    )
                for bond_idx in tuple(stable):
                    if _ez_description(form, selected[bond_idx], structural) != expected[bond_idx]:
                        stable.remove(bond_idx)
        except (RuntimeError, ValueError):
            unproven.update(selected)
            continue
        keep.update(stable)
    if unproven:
        logger.warning(
            "stereo: E/Z at bond(s) %s left unassigned: resonance invariance unproven in %d forms; state it to keep it",
            ", ".join(
                f"{bond.GetBeginAtomIdx()}={bond.GetEndAtomIdx()}"
                for bond in map(work.GetBondWithIdx, sorted(unproven))
            ),
            _RESONANCE_EZ_CAP,
        )
    return [idx for idx in bond_centers if idx in keep]


def defined_stereo_label(mol, exclude=()):
    """Label the ligand stereo already defined on a coordinated molecule."""
    work, cap_to_metal = _build_enumeration_graph(mol, set(exclude))
    atom_centers, bond_centers, atrop_centers = _stereo_centres(work, mol.GetNumAtoms())
    full = Chem.Mol(mol)
    _graft_point_tags(full, work, atom_centers, cap_to_metal)
    label = _stereo_label(
        work,
        atom_centers,
        bond_centers,
        atrop_centers,
        cap_to_metal,
        _point_cip_codes(full, atom_centers),
    )
    native = bond_stereo(label)
    parts = label.split(",") if label else []
    for pair, code in encoded_bond_stereo(mol).items():
        if pair in native and native[pair] != code:
            raise ValueError(f"native and CX E/Z disagree on bond {tuple(sorted(pair))}")
        if pair not in native:
            i, j = sorted(pair)
            parts.append(f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}={mol.GetAtomWithIdx(j).GetSymbol()}{j}:{code}")
    return ",".join(parts)


def _non_tetrahedral_points(mol, centres):
    """Find four-carrier points whose centre is outside the open neighbour tetrahedron."""
    positions = mol.GetConformer().GetPositions()
    invalid = set()
    for idx in centres:
        atom = mol.GetAtomWithIdx(idx)
        # Match RDKit Chirality.cpp::bondAffectsAtomChirality on the capped stereo graph.
        carriers = [
            bond.GetOtherAtomIdx(idx)
            for bond in atom.GetBonds()
            if bond.GetBondType() not in {Chem.BondType.ZERO, Chem.BondType.UNSPECIFIED}
            and not (bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() == idx)
        ]
        if len(carriers) != 4:  # noqa: PLR2004
            continue  # an implicit H or lone pair supplies no fourth physical position
        points = positions[carriers]
        try:
            weights = np.linalg.solve((points[:3] - points[3]).T, positions[idx] - points[3])
        except np.linalg.LinAlgError:
            invalid.add(idx)
            continue
        if not (np.all(weights > 0.0) and weights.sum() < 1.0):
            invalid.add(idx)
    return invalid


def stereo_from_3d(mol, exclude=(), *, apply=False):
    """Label measured ligand stereo, leaving non-tetrahedral points unassigned.

    RDKit's 3D assignment uses three neighbours in bond order; outside the neighbour tetrahedron those
    choices can disagree. Apply the same centre-inside-volume condition as native DG before trusting CIP,
    without a fitted volume margin or ideal angle. ``apply=True`` also removes such invalid point tags.
    """
    if not mol.GetNumConformers():
        raise ValueError("stereo_from_3d needs a conformer")
    exclude = set(exclude)
    work, cap_to_metal = _build_enumeration_graph(mol, exclude)
    if work is mol:
        work = Chem.Mol(work)
    supported_points, _unsupported = _point_capability(mol, work, cap_to_metal, exclude)
    _, stated_bonds, stated_axes = _stereo_centres(work, mol.GetNumAtoms())
    inferred = set(_coordination_atrop_bonds(mol, exclude, work)) - set(stated_axes)
    Chem.AssignStereochemistryFrom3D(work, confId=work.GetConformer().GetId(), replaceExistingTags=True)
    invalid = _non_tetrahedral_points(work, supported_points)
    full = Chem.Mol(mol)
    for idx in invalid:
        for graph in (work, full):
            atom = graph.GetAtomWithIdx(idx)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.ClearProp("_CIPCode")
    assign_atrop_from_3d(work, stated_axes)
    atrop_centers = sorted(set(stated_axes) | set(assign_atrop_from_3d(work, inferred, required=False)))
    atom_centers, bond_centers, _ = _stereo_centres(work, mol.GetNumAtoms())
    atom_centers = [idx for idx in atom_centers if idx in supported_points]
    bond_centers = _resonance_stable_ez(work, bond_centers, stated_bonds)
    _graft_point_tags(full, work, atom_centers, cap_to_metal)
    label = _stereo_label(
        work,
        atom_centers,
        bond_centers,
        atrop_centers,
        cap_to_metal,
        _point_cip_codes(full, atom_centers),
    )
    if apply:
        for idx in set(atom_centers) | invalid:
            mol.GetAtomWithIdx(idx).SetChiralTag(full.GetAtomWithIdx(idx).GetChiralTag())
            if idx in invalid:
                mol.GetAtomWithIdx(idx).ClearProp("_CIPCode")
        clear_ez(mol)
        for idx in bond_centers:
            source = work.GetBondWithIdx(idx)
            target = mol.GetBondBetweenAtoms(source.GetBeginAtomIdx(), source.GetEndAtomIdx())
            refs = [cap_to_metal.get(ref, (None, None, None, ref))[3] for ref in source.GetStereoAtoms()]
            if target is not None and len(refs) == 2:  # noqa: PLR2004
                target.SetStereoAtoms(*refs)
                target.SetStereo(source.GetStereo())
        Chem.SetDoubleBondNeighborDirections(mol)
        _apply_atrop_stereo(mol, axis_stereo(label))
    return label


def _enumerate_atrop(work_isos, atrop_centers, cap):
    """Expand and native-canonicalize the selected atropisomer bonds."""
    if not atrop_centers:
        return work_isos
    expanded, seen = [], set()
    for base in work_isos:
        for tags in itertools.product(ATROP_STEREO, repeat=len(atrop_centers)):
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

    Returns ``(variants, n_unassigned, total, unresolved)``: ``variants`` is a list of ``(variant_mol, label)``
    with defined centres held, meso/duplicates dropped, and truncated to ``cap`` of ``total``. ``unresolved``
    counts elements RDKit could not enumerate, such as an allene axis, which stays at one arbitrary hand for
    the caller to warn about. Atom order is preserved, so index-based ``fix``/``constrain`` stay valid.

    `exclude` is the metal indices. Excluding them is safe: a metal's own handedness is the coordination-
    isomer path's job, and RDKit's dative-metal stereo is not order-canonical anyway. A ligand stereocentre,
    including a chiral-at-P or carbanion donor that drops to degree 3 after the strip, is still enumerated.
    """
    n_real = mol.GetNumAtoms()
    work, cap_to_metal, locked, unassigned, atrop_centers, unsupported = _unassigned_elements(
        mol,
        exclude,
        include,
        set(skip_points),
        skip_bonds,
        skip_atrop,
        include_atrop,
    )
    if not unassigned and not atrop_centers:
        if not locked:
            return [(mol, "")], 0, 1, unsupported
        variant = Chem.Mol(mol)
        clear_ez(variant, locked)
        return [(variant, "")], 0, 1, unsupported
    atom_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Atom_Tetrahedral]
    bond_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Bond_Double]
    # RDKit's `unique` keys on `work`, where a bound arm reads like a free one once the metal is cut. Key on the
    # complex: a sigma donor keeps a single bond, so RDKit ranks every carrier of its hand, and a haptic face
    # marks its atoms instead, since the three-membered rings of its bonds would hide the face's E/Z.
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=not exclude, maxIsomers=cap)
    metal_bonds = [(metal, n.GetIdx()) for metal in exclude for n in mol.GetAtomWithIdx(metal).GetNeighbors()]
    faces = {
        atom
        for metal in exclude
        for site in haptic_sites(mol, [donor for m, donor in metal_bonds if m == metal])
        if len(site) > 1
        for atom in site
    }
    total = 2 ** (len(unassigned) + len(atrop_centers))
    work_isos = list(EnumerateStereoisomers(work, opts)) if unassigned else [work]
    work_isos = _enumerate_atrop(work_isos, atrop_centers, cap)
    variants, seen = [], set()
    # Copy enumerated ligand stereo (atom parity + E/Z) onto the full mol; skip isotope caps.
    for wv in work_isos:
        full = Chem.Mol(mol)
        clear_ez(full, locked)  # the arbitrary lock belongs only to the disposable enumeration graph
        # `work` is `mol` with each M-donor bond removed, so its tags must be re-based onto the full bond order.
        _graft_point_tags(full, wv, atom_centers, cap_to_metal)
        for b in wv.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            # skip a coordination-locked bond: its `work` stereo is the arbitrary lock value, not a real hand;
            # the full mol keeps it unspecified so the metal embed builds the ring-feasible geometry.
            if b.GetIdx() not in bond_centers or frozenset((i, j)) in locked:
                continue
            if b.GetStereo() != Chem.BondStereo.STEREONONE and i < n_real and j < n_real:
                fb = full.GetBondBetweenAtoms(i, j)
                if fb is not None:
                    refs = tuple(cap_to_metal[ref][3] if ref in cap_to_metal else ref for ref in b.GetStereoAtoms())
                    fb.SetStereoAtoms(*refs)
                    fb.SetStereo(b.GetStereo())
        for pair in atrop_centers:
            wb = wv.GetBondBetweenAtoms(*pair)
            fb = full.GetBondBetweenAtoms(*pair)
            if wb is not None and fb is not None:
                fb.SetStereo(wb.GetStereo())
        label = _stereo_label(
            wv,
            atom_centers,
            bond_centers,
            atrop_centers,
            cap_to_metal,
            _point_cip_codes(full, atom_centers),
        )
        keyed = Chem.RWMol(full)
        keyed.RemoveAllConformers()  # coordinates would break the symmetry the key must see
        for metal, donor in metal_bonds:
            if donor in faces:
                keyed.RemoveBond(metal, donor)
                keyed.GetAtomWithIdx(donor).SetAtomMapNum(mol.GetAtomWithIdx(metal).GetAtomicNum())
            else:
                keyed.GetBondBetweenAtoms(metal, donor).SetBondType(Chem.BondType.SINGLE)
        keyed = keyed.GetMol()
        keyed.UpdatePropertyCache(strict=False)
        Chem.SetDoubleBondNeighborDirections(keyed)  # the SMILES writer reads E/Z only from bond directions
        key = Chem.MolToCXSmiles(keyed)
        if key in seen:
            continue
        seen.add(key)
        variants.append((full, label))
    probe = work_isos[0] if work_isos else work  # centres still UNSPECIFIED after enumeration = axial (allene)
    Chem.FindPotentialStereo(probe, cleanIt=True, flagPossible=False)
    unresolved = (
        unsupported
        + sum(1 for i in atom_centers if probe.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED)
        + sum(1 for b in bond_centers if probe.GetBondWithIdx(b).GetStereo() == Chem.BondStereo.STEREONONE)
    )
    return variants, len(unassigned) + len(atrop_centers), total, unresolved
