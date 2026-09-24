"""Canonicalize coordination-site identity and read metal-centred stereochemistry."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from . import metal_polyhedron as _poly
from .metal_core import (
    _EPS_LEN,
    _ETA2,
    COORDINATION_METALS,
    VACANT,
    _frag_map,
    _ligand_distance_matrix,
    _vertex_atom,
    metal_indices,
)
from .metal_polyhedron import vertex_dirs
from .stereo import _apply_encoded_bond_stereo, _coordination_locked_double_bonds
from .utils import bond_removal_mirrors, mirror_tag

_FACE_EPS = 1e-8
_HALF_TURN = 180
_CIS_STEREO = {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}
_TRANS_STEREO = {Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE}
_PT = Chem.GetPeriodicTable()
_P_BLOCK_OUTER = frozenset(range(3, 8))  # RDKit's group index for main-group 13-17; d/f-block excluded below
_CHALCOGEN_OUTER = 6  # group 16: the terminal donor atom of a p-block hypervalent centre


def remove_routine_hydrogens(mol, keep=()):
    """Return a hydrogen-reduced Mol and its old-to-new atom-index mapping."""
    out, keep = Chem.Mol(mol), set(keep)
    out.UpdatePropertyCache(strict=False)
    point_tags = {}
    for atom in out.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIdx() not in keep:
                if bond_removal_mirrors(atom, neighbor.GetIdx()):
                    tag = mirror_tag(tag)
        point_tags[atom.GetIdx()] = tag
    hydrogens = {
        atom.GetIdx(): atom.GetTotalNumHs() + sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
        for atom in out.GetAtoms()
        if atom.GetAtomicNum() != 1
    }
    for atom in out.GetAtoms():
        atom.SetIntProp("_rxembedOriginalIndex", atom.GetIdx())
        if atom.GetIdx() in keep and atom.GetAtomicNum() == 1 and not atom.GetIsotope():
            atom.SetBoolProp("_rxembedCoordinationH", True)
            atom.SetIsotope(1)
    params = Chem.RemoveHsParameters()
    params.removeDegreeZero = True
    params.removeDefiningBondStereo = True
    params.showWarnings = False  # protected donor hydrides are deliberately isotope-marked and retained
    out = Chem.RemoveHs(out, params, sanitize=False)
    at = {}
    deficits = []
    for atom in out.GetAtoms():
        original = atom.GetIntProp("_rxembedOriginalIndex")
        at[original] = atom.GetIdx()
        if original in point_tags:
            atom.SetChiralTag(point_tags[original])
        if atom.GetAtomicNum() != 1:
            current = atom.GetTotalNumHs() + sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
            deficits.append((atom, hydrogens[original] - current))
        atom.ClearProp("_rxembedOriginalIndex")
        if atom.HasProp("_rxembedCoordinationH"):
            atom.SetIsotope(0)
            atom.ClearProp("_rxembedCoordinationH")
    for atom, deficit in deficits:
        if deficit > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + deficit)
    out.UpdatePropertyCache(strict=False)
    return out, at


def _hypervalent_bond(bond):
    """Return whether a bond joins a p-block centre to one of its terminal chalcogen donors.

    RDKit's conjugation perception never marks an expanded-octet X=O bond conjugated, so a terminal chalcogen
    (group 16, one heavy neighbour) bonded to a p-block centre (groups 13-17, not a coordination metal) is
    added by this same graph fact: X=O and X(+)-O(-) are one drawing choice, exactly as a carboxylate's are.
    """

    def centre(atom):
        z = atom.GetAtomicNum()
        return z not in COORDINATION_METALS and _PT.GetNOuterElecs(z) in _P_BLOCK_OUTER

    def terminal(atom):
        heavy_degree = sum(neighbor.GetAtomicNum() != 1 for neighbor in atom.GetNeighbors())
        return _PT.GetNOuterElecs(atom.GetAtomicNum()) == _CHALCOGEN_OUTER and heavy_degree == 1

    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    return (centre(begin) and terminal(end)) or (centre(end) and terminal(begin))


# Resonance moves bond orders and charges only inside one conjugated system, so ranking a graph that gives
# every conjugated bond one type and every atom of a system that system's total charge, leaving everything
# else exact, proves the same identity as form enumeration: two roots share a rank exactly when a resonance
# form maps one onto the other, at any molecule size and with no cap. `_hypervalent_bond` extends the same
# rule to a p-block centre's expanded-octet donors, which RDKit's own conjugation perception cannot reach,
# and the two kinds of bond share one systems graph, so a hypervalent group merges with an adjoining
# conjugated system wherever they touch with no extra code.
def _root_classes(mol, roots):
    """Classify roots by graph symmetry after erasing each conjugated system's drawn Lewis form."""
    roots = list(dict.fromkeys(roots))
    chemical = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() in (Chem.BondType.ZERO, Chem.BondType.DATIVE):
            chemical.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    try:
        chemical.UpdatePropertyCache(strict=False)
        Chem.SetConjugation(chemical)
        systems = Chem.RWMol(chemical)
        flat = Chem.RWMol(mol)
        for bond in chemical.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if not (bond.GetIsConjugated() or _hypervalent_bond(bond)):
                systems.RemoveBond(begin, end)
                continue
            flat.GetBondBetweenAtoms(begin, end).SetBondType(Chem.BondType.AROMATIC)
            flat.GetBondBetweenAtoms(begin, end).SetIsAromatic(False)
        for atoms in Chem.GetMolFrags(systems, sanitizeFrags=False):
            charge = sum(mol.GetAtomWithIdx(index).GetFormalCharge() for index in atoms)
            for index in atoms:
                atom = flat.GetAtomWithIdx(index)
                atom.SetNumExplicitHs(mol.GetAtomWithIdx(index).GetTotalNumHs())
                atom.SetNoImplicit(True)
                atom.SetFormalCharge(charge)
                atom.SetIsAromatic(False)
        flat.UpdatePropertyCache(strict=False)
        ranks = list(Chem.CanonicalRankAtoms(flat, breakTies=False))
        exact = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("RDKit could not canonicalize coordination-site identity") from exc
    labels = {}
    for root in roots:
        labels[ranks[root]] = min(labels.get(ranks[root], exact[root]), exact[root])
    return {root: labels[ranks[root]] for root in roots}


def donor_classes(mol, donors):
    """Map donor atoms to graph or RDKit-proven resonance symmetry classes.

    Coordination identity follows graph automorphism, not one localized charge or bond-order assignment.
    """
    ranked, at = remove_routine_hydrogens(mol, donors)
    classes = _root_classes(ranked, [at[donor] for donor in donors])
    return {donor: classes[at[donor]] for donor in donors}


def site_classes(mol, sites, haptic=None, coordination=()):
    """Map occupied coordination sites to graph-symmetry classes.

    A marker joined to every atom of a haptic face ranks the rooted atom set, while temporary zero bonds retain
    terminal and bridging donor roles on the stripped surrogate graph.
    """
    haptic = haptic or {}
    occupied = [site for site in sites if site != VACANT]
    atoms = {atom for site in occupied for atom in (haptic.get(site) or (site,))}
    atoms.update(donor for donor, _metal, _atomic_num, _charge in coordination)
    rw = Chem.RWMol(mol)
    for donor, metal, atomic_num, charge in coordination:
        if min(donor, metal) < 0 or max(donor, metal) >= mol.GetNumAtoms():
            continue
        atom = rw.GetAtomWithIdx(metal)
        atom.SetAtomicNum(atomic_num)
        atom.SetFormalCharge(charge)
        atom.SetIsotope(1000 + charge)
        if rw.GetBondBetweenAtoms(donor, metal) is None:
            rw.AddBond(donor, metal, Chem.BondType.ZERO)
    work = rw.GetMol()
    ring = Chem.RWMol(work)
    for bond in ring.GetBonds():
        if bond.GetBondType() == Chem.BondType.ZERO:
            bond.SetBondType(Chem.BondType.SINGLE)
    ring = ring.GetMol()
    locked = _coordination_locked_double_bonds(ring, metal_indices(ring))
    for pair in locked:
        bond = work.GetBondBetweenAtoms(*pair)
        if bond is not None:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    _apply_encoded_bond_stereo(work, skip=locked)
    # Complete the donor's stereo carriers before removing H and rebasing its point tag.
    ranked, at = remove_routine_hydrogens(work, atoms)
    rw = Chem.RWMol(ranked)
    markers = {}
    for site in occupied:
        marker = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(marker).SetNoImplicit(True)
        for atom in haptic.get(site) or (site,):
            rw.AddBond(marker, at[atom], Chem.BondType.ZERO)
        markers[site] = marker
    marked = rw.GetMol()
    marked.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(marked)
    classes = _root_classes(marked, markers.values())
    return {site: (classes[marker],) for site, marker in markers.items()}


def equivalent_site_assignments(classes, links=None, *, targets=None, sources=None):
    """Yield class- and chelate-link-preserving target-to-source vertex maps."""
    link_labels = {} if links is None else links
    occupied = [vertex for vertex, value in enumerate(classes) if value is not None]
    targets = occupied if targets is None else list(targets)
    sources = occupied if sources is None else list(sources)
    remaining, assigned = set(sources), {}

    def place(position):
        if position == len(targets):
            yield dict(assigned)
            return
        target = targets[position]
        for source in sources:
            if source not in remaining or classes[target] != classes[source]:
                continue
            if any(
                link_labels.get(frozenset((target, other))) != link_labels.get(frozenset((source, mapped)))
                for other, mapped in assigned.items()
            ):
                continue
            assigned[target] = source
            remaining.remove(source)
            yield from place(position + 1)
            remaining.add(source)
            del assigned[target]

    yield from place(0)


def _face_walk(mol, face):
    """Order a haptic face that is a simple path or cycle, or return ``None``."""
    inside = set(face)
    neighbours = {
        atom: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors() if n.GetIdx() in inside)
        for atom in face
    }
    ends = [atom for atom in face if len(neighbours[atom]) == 1]
    # A path or cycle has at most two neighbours per atom, and 0 open ends if a cycle, else 2.
    if any(len(neighbours[atom]) > 2 for atom in face) or len(ends) not in (0, 2):  # noqa: PLR2004
        return None
    walk = [min(ends) if ends else min(face)]
    while len(walk) < len(face):
        previous = walk[-2] if len(walk) > 1 else None
        step = [atom for atom in neighbours[walk[-1]] if atom != previous]
        if not step:
            return None
        walk.append(min(step))
    return walk, not ends


def _canonical_face_walk(mol, face, ranks):
    """Return the canonical direction around a planar-chiral haptic face."""
    if len(face) < 3 or (walked := _face_walk(mol, face)) is None:  # noqa: PLR2004 - eta2 has its own signature path
        return None
    order, closed = walked
    n = len(order)
    classes = [ranks[atom] for atom in order]
    if closed:
        forward = min(tuple(classes[(start + i) % n] for i in range(n)) for start in range(n))
        reverse = min(tuple(classes[(start - i) % n] for i in range(n)) for start in range(n))
    else:
        forward, reverse = tuple(classes), tuple(reversed(classes))
    if forward == reverse:
        return None
    return (order if forward < reverse else order[::-1]), closed


def _eta2_centres(mol, face, ranks, metal=None):
    """Return CIP-orderable trigonal centres on an eta2 face."""
    if len(face) != _ETA2 or mol.GetBondBetweenAtoms(*face) is None:
        return []
    out = []
    for atom in face:
        neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors() if n.GetIdx() != metal]
        ordered = sorted(neighbours, key=ranks.__getitem__, reverse=True)
        if len(ordered) == 3 and len({ranks[n] for n in ordered}) == len(ordered):  # noqa: PLR2004 - CIP needs 3
            key = (ranks[atom], tuple(sorted((ranks[n] for n in ordered), reverse=True)))
            out.append((atom, key, ordered))
    return out


def eta2_signatures(mol, face, ranks=None):
    """Return the two mirror-related re/si signatures available to an eta2 face.

    Empty signatures mean the ligand graph does not define an independent, priority-orderable face choice.
    """
    if any(mol.GetAtomWithIdx(atom).GetTotalNumHs() for atom in face):
        mol = Chem.AddHs(mol, onlyOnAtoms=list(face))
        ranks = None
    try:
        ranks = list(Chem.ComputeAtomCIPRanks(mol)) if ranks is None else ranks
    except (RuntimeError, ValueError):
        return (), ()
    centres = _eta2_centres(mol, face, ranks)
    if not centres:
        return (), ()
    if len(centres) == 1:
        signature = ((centres[0][1], "re"),)
    else:
        a, b = face
        bond = mol.GetBondBetweenAtoms(a, b)
        refs = list(bond.GetStereoAtoms())
        if len(centres) != 2:  # noqa: PLR2004 - a two-atom face has at most two CIP centres
            return (), ()
        if bond.GetStereo() in _CIS_STEREO | _TRANS_STEREO and len(refs) == 2:  # noqa: PLR2004 - two stereo refs
            cis = bond.GetStereo() in _CIS_STEREO
            if bond.GetBeginAtomIdx() != a:
                refs.reverse()  # RDKit stores stereo references in bond begin/end order, not `face` order
        elif (
            bond.GetStereo() == Chem.BondStereo.STEREONONE
            and bond.IsInRing()
            and not any(
                info.type == Chem.StereoType.Bond_Double and info.centeredOn == bond.GetIdx()
                for info in Chem.FindPotentialStereo(mol)
            )
        ):
            cis = True
            ring = min((set(r) for r in mol.GetRingInfo().AtomRings() if {a, b} <= set(r)), key=len)
            refs = [
                next(
                    n.GetIdx()
                    for n in mol.GetAtomWithIdx(atom).GetNeighbors()
                    if n.GetIdx() != other and n.GetIdx() in ring
                )
                for atom, other in ((a, b), (b, a))
            ]
        else:
            return (), ()
        ra, rb = refs
        try:
            other_a = next(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() not in {b, ra})
            other_b = next(n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() not in {a, rb})
        except StopIteration:
            return (), ()
        angle = {b: 0, ra: 120, other_a: 240, a: 180}
        angle[rb], angle[other_b] = (60, -60) if cis else (-60, 60)
        signature = tuple(
            sorted(
                (
                    key,
                    "si" if 0 < (angle[ordered[1]] - angle[ordered[0]]) % 360 < _HALF_TURN else "re",
                )
                for _atom, key, ordered in centres
            )
        )
    mirror = tuple(sorted((key, "si" if name == "re" else "re") for key, name in signature))
    return signature, mirror


def face_has_orientation(mol, face, ranks):
    """Return whether a haptic face has two distinguishable mirror orientations."""
    if len(face) == _ETA2:
        signature, mirror = eta2_signatures(mol, face)
        return bool(signature and signature != mirror)
    return _canonical_face_walk(mol, face, ranks) is not None


def _face_side(normal, direction, area_scale):
    """Return a scale-invariant side of a face, or zero for a degenerate placement."""
    value = float(normal @ direction)
    scale = float(area_scale * np.linalg.norm(direction))
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0.0 or abs(value) <= _FACE_EPS * scale:
        return 0
    return 1 if value > 0.0 else -1


def face_winding(mol, pos, metal, face, ranks, eta2_ranks=None):
    """Return the canonical ``'+'`` or ``'-'`` orientation of a haptic face.

    Proper rotation preserves the sign and reflection flips it. The sign remains authoritative when RDKit's
    CIP ranks cannot supply a conventional re/si or planar descriptor.
    """
    if len(face) == _ETA2:
        try:
            eta2_ranks = list(Chem.ComputeAtomCIPRanks(mol)) if eta2_ranks is None else eta2_ranks
        except (RuntimeError, ValueError):
            return ""
        signature = []
        for atom, key, ordered in _eta2_centres(mol, face, eta2_ranks, metal):
            centre = pos[atom]
            left, right = pos[ordered[0]] - centre, pos[ordered[1]] - centre
            side = _face_side(np.cross(left, right), pos[metal] - centre, np.linalg.norm(left) * np.linalg.norm(right))
            if not side:
                return ""
            signature.append((key, "si" if side > 0 else "re"))
        signature = tuple(sorted(signature))
        mirror = tuple(sorted((key, "si" if name == "re" else "re") for key, name in signature))
        return "+" if signature < mirror else "-" if signature > mirror else ""
    canonical = _canonical_face_walk(mol, face, ranks)
    if canonical is None:
        return ""
    sequence, closed = canonical
    centre = np.mean([pos[atom] for atom in sequence], axis=0)
    following = sequence[1:] + sequence[:1] if closed else sequence[1:]
    terms = [np.cross(pos[a] - centre, pos[b] - centre) for a, b in zip(sequence, following, strict=False)]
    side = _face_side(sum(terms, start=np.zeros(3)), centre - pos[metal], sum(np.linalg.norm(term) for term in terms))
    return "+" if side > 0 else "-" if side < 0 else ""


def face_descriptors(mol, donors, haptic, windings):
    """Return re/si or ``Rₚ``/``Sₚ`` labels for priority-orderable haptic faces.

    Tied or stereo-dependent priorities deliberately remain unnamed; their canonical winding is still lossless.
    """
    if not windings:
        return {}

    def reference(ranks, sequence):
        highest = max(ranks[atom] for atom in sequence)
        pilots = [atom for atom in sequence if ranks[atom] == highest]
        if len(pilots) != 1:
            return None
        pilot = pilots[0]
        i = sequence.index(pilot)
        previous, following = sequence[i - 1], sequence[(i + 1) % len(sequence)]
        if ranks[previous] == ranks[following]:
            return None
        return pilot, following if ranks[following] > ranks[previous] else previous

    try:
        priorities = Chem.ComputeAtomCIPRanks(mol)
        unmarked = Chem.Mol(mol)
        Chem.RemoveStereochemistry(unmarked)
        constitutional = Chem.ComputeAtomCIPRanks(unmarked)
    except (RuntimeError, ValueError):
        return {}
    classes = donor_classes(mol, donors)
    out = {}
    for dummy, winding in windings.items():
        face = haptic.get(dummy, ())
        if len(face) == _ETA2:
            signature, mirror = eta2_signatures(mol, face, priorities)
            if winding in "+-" and signature and signature != mirror:
                selected = min(signature, mirror) if winding == "+" else max(signature, mirror)
                out[dummy] = f"({','.join(name for _key, name in selected)})"
            continue
        canonical = _canonical_face_walk(mol, face, classes)
        if winding not in "+-" or canonical is None or not canonical[1]:
            continue
        sequence = canonical[0]
        choice = reference(priorities, sequence)
        if choice is None or choice != reference(constitutional, sequence):
            continue
        pilot, toward_second = choice
        following = sequence[(sequence.index(pilot) + 1) % len(sequence)]
        sense = (1 if winding == "+" else -1) * (1 if toward_second == following else -1)
        out[dummy] = "Rₚ" if sense < 0 else "Sₚ"
    return out


def chelate_links(mol, vertices, haptic=None, distances=None):
    """Return same-ligand vertex pairs labelled by their shortest graph distance."""
    haptic = haptic or {}
    frag = _frag_map(mol)
    occupied = [vertex for vertex in range(len(vertices)) if vertices[vertex] != VACANT]
    pairs = [
        (a, b)
        for i, a in enumerate(occupied)
        for b in occupied[i + 1 :]
        if frag[_vertex_atom(haptic, vertices[a])] == frag[_vertex_atom(haptic, vertices[b])]
    ]
    if not pairs:
        return {}
    distances = _ligand_distance_matrix(mol) if distances is None else distances
    return {
        frozenset((a, b)): min(
            int(distances[left][right])
            for left in haptic.get(vertices[a], (vertices[a],))
            for right in haptic.get(vertices[b], (vertices[b],))
        )
        for a, b in pairs
    }


def chirality_of(mol, geometry, vertices, haptic=None, coordination=(), *, classes=None, links=None):
    """Return the canonical metal-centre hand, or empty when achiral or undecidable.

    Point-group parity uses canonical site classes plus graph-distance-labelled donor links, so atom order,
    resonance localization and proper rotation cannot change the result.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return ""
    return _poly.handedness(
        dirs,
        list(vertices),
        site_classes(mol, vertices, haptic, coordination) if classes is None else classes,
        chelate_links(mol, vertices, haptic) if links is None else links,
    )


def realised_chirality(mol, cid, geometry, vertices, metal, chirality, haptic=None):
    """Read the stated metal-centre hand from one 3D conformer."""
    dirs = vertex_dirs(geometry) if chirality else None
    if dirs is None or len(vertices) != len(dirs) or any(atom < 0 for atom in vertices):
        return ""
    pos = mol.GetConformer(int(cid)).GetPositions()
    haptic = haptic or {}

    def point(atom):
        face = haptic.get(atom)
        return np.mean(pos[list(face)], axis=0) if face else pos[atom]

    observed = np.asarray([point(atom) - pos[metal] for atom in vertices])
    lengths = np.linalg.norm(observed, axis=1, keepdims=True)
    if not np.all(np.isfinite(observed)) or not np.all(lengths > _EPS_LEN):
        return ""
    if _poly.orientation_parity(observed / lengths, dirs) > 0:
        return chirality
    return _poly.LAMBDA if chirality == _poly.DELTA else _poly.DELTA
