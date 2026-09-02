"""Canonicalize coordination-site identity and read metal-centred stereochemistry."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from . import metal_polyhedron as _poly
from .metal_core import _EPS_LEN, VACANT, _frag_map, _vertex_atom
from .metal_polyhedron import vertex_dirs

_MIN_STEREO_NEIGHBOURS = 3
_ETA2 = 2
_FACE_MIN = 3
_PATH_ENDS = 2
_FACE_EPS = 1e-8
_HALF_TURN = 180
_CIS_STEREO = {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}
_TRANS_STEREO = {Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE}


def _flat_ranks(mol):
    """Rank the resonance-insensitive skeleton with charge, bond order and stereo removed."""
    rw = Chem.RWMol(mol)
    total_h = [a.GetTotalNumHs() for a in rw.GetAtoms()]
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom, h in zip(rw.GetAtoms(), total_h, strict=True):
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(h)
        atom.SetNoImplicit(True)
    flat = rw.GetMol()
    flat.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(flat)
    return list(Chem.CanonicalRankAtoms(flat, breakTies=False))


def remove_routine_hydrogens(mol, keep=()):
    """Return a hydrogen-reduced Mol and its old-to-new atom-index mapping."""
    out, keep = Chem.Mol(mol), set(keep)
    for atom in out.GetAtoms():
        atom.SetIntProp("_rxembedOriginalIndex", atom.GetIdx())
        if atom.GetIdx() in keep and atom.GetAtomicNum() == 1 and not atom.GetIsotope():
            atom.SetBoolProp("_rxembedCoordinationH", True)
            atom.SetIsotope(1)
    params = Chem.RemoveHsParameters()
    params.removeDegreeZero = True
    out = Chem.RemoveHs(out, params, sanitize=False)
    at = {}
    for atom in out.GetAtoms():
        at[atom.GetIntProp("_rxembedOriginalIndex")] = atom.GetIdx()
        atom.ClearProp("_rxembedOriginalIndex")
        if atom.HasProp("_rxembedCoordinationH"):
            atom.SetIsotope(0)
            atom.ClearProp("_rxembedCoordinationH")
    out.UpdatePropertyCache(strict=False)
    return out, at


def donor_classes(mol, donors):
    """Map donor atoms to graph-symmetry classes, coarsened over resonance forms.

    Coordination identity follows graph automorphism, not one localized charge or bond-order assignment.
    The resonance-flat ranking may merge perceived classes but never split them.
    """
    try:
        ranked, at = remove_routine_hydrogens(mol, donors)
        ranks = list(Chem.CanonicalRankAtoms(ranked, breakTies=False))
        flat = _flat_ranks(ranked)
    except Exception:  # pragma: no cover - ranking must never make embedding fail
        return {d: mol.GetAtomWithIdx(d).GetSymbol() for d in donors}
    root = {}

    def find(x):
        while root.setdefault(x, x) != x:
            x = root[x] = root[root[x]]
        return x

    for donor in donors:
        perceived = find(("perceived", ranks[at[donor]]))
        resonance = find(("flat", flat[at[donor]]))
        root[max(perceived, resonance)] = min(perceived, resonance)
    return {donor: find(("perceived", ranks[at[donor]])) for donor in donors}


def site_classes(mol, sites, haptic=None, coordination=()):
    """Map occupied coordination sites to graph-symmetry classes.

    A marker joined to every atom of a haptic face ranks the rooted atom set, while temporary zero bonds retain
    terminal and bridging donor roles on the stripped surrogate graph.
    """
    haptic = haptic or {}
    occupied = [site for site in sites if site != VACANT]
    atoms = {atom for site in occupied for atom in (haptic.get(site) or (site,))}
    atoms.update(donor for donor, _metal, _atomic_num, _charge in coordination)
    ranked, at = remove_routine_hydrogens(mol, atoms)
    rw = Chem.RWMol(ranked)
    for donor, metal, atomic_num, charge in coordination:
        if donor not in at or metal not in at:
            continue
        atom = rw.GetAtomWithIdx(at[metal])
        atom.SetAtomicNum(atomic_num)
        atom.SetFormalCharge(charge)
        atom.SetIsotope(1000 + charge)
        if rw.GetBondBetweenAtoms(at[donor], at[metal]) is None:
            rw.AddBond(at[donor], at[metal], Chem.BondType.ZERO)
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
    perceived = list(Chem.CanonicalRankAtoms(marked, breakTies=False))
    flat = _flat_ranks(marked)
    root = {}

    def find(x):
        while root.setdefault(x, x) != x:
            x = root[x] = root[root[x]]
        return x

    for marker in markers.values():
        ranked_marker = find(("perceived", perceived[marker]))
        flat_marker = find(("flat", flat[marker]))
        root[max(ranked_marker, flat_marker)] = min(ranked_marker, flat_marker)
    return {site: (find(("perceived", perceived[marker])),) for site, marker in markers.items()}


def _face_walk(mol, face):
    """Order a haptic face that is a simple path or cycle, or return ``None``."""
    inside = set(face)
    neighbours = {
        atom: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors() if n.GetIdx() in inside)
        for atom in face
    }
    ends = [atom for atom in face if len(neighbours[atom]) == 1]
    if any(len(neighbours[atom]) > _PATH_ENDS for atom in face) or len(ends) not in (0, _PATH_ENDS):
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
    if len(face) < _FACE_MIN or (walked := _face_walk(mol, face)) is None:
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
        if len(ordered) == _MIN_STEREO_NEIGHBOURS and len({ranks[n] for n in ordered}) == len(ordered):
            key = (ranks[atom], tuple(sorted((ranks[n] for n in ordered), reverse=True)))
            out.append((atom, key, ordered))
    return out


def eta2_signatures(mol, face, ranks=None):
    """Return the two mirror-related re/si signatures available to an eta2 face.

    Empty signatures mean the ligand graph does not define an independent, priority-orderable face choice.
    """
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
        if len(centres) != _ETA2:
            return (), ()
        if bond.GetStereo() in _CIS_STEREO | _TRANS_STEREO and len(refs) == _ETA2:
            cis = bond.GetStereo() in _CIS_STEREO
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
        if bond.GetBeginAtomIdx() != a:
            refs.reverse()
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
            volume = float(np.cross(pos[ordered[0]] - centre, pos[ordered[1]] - centre) @ (pos[metal] - centre))
            if abs(volume) <= _FACE_EPS:
                return ""
            signature.append((key, "si" if volume > 0 else "re"))
        signature = tuple(sorted(signature))
        mirror = tuple(sorted((key, "si" if name == "re" else "re") for key, name in signature))
        return "+" if signature < mirror else "-" if signature > mirror else ""
    canonical = _canonical_face_walk(mol, face, ranks)
    if canonical is None:
        return ""
    sequence, closed = canonical
    centre = np.mean([pos[atom] for atom in sequence], axis=0)
    following = sequence[1:] + sequence[:1] if closed else sequence[1:]
    circulation = sum(
        (np.cross(pos[a] - centre, pos[b] - centre) for a, b in zip(sequence, following, strict=False)),
        start=np.zeros(3),
    )
    return "+" if float(circulation @ (centre - pos[metal])) > 0 else "-"


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


def chelate_edges(mol, vertices, haptic=None):
    """Return vertex pairs whose donors chelate through one ligand."""
    frag = _frag_map(mol)
    occupied = [vertex for vertex in range(len(vertices)) if vertices[vertex] != VACANT]
    return frozenset(
        frozenset((a, b))
        for i, a in enumerate(occupied)
        for b in occupied[i + 1 :]
        if frag[_vertex_atom(haptic, vertices[a])] == frag[_vertex_atom(haptic, vertices[b])]
    )


def chirality_of(mol, geometry, vertices, haptic=None, coordination=(), *, classes=None, edges=None):
    """Return the canonical metal-centre hand, or empty when achiral or undecidable.

    The point-group parity is computed from canonical site classes plus the chelate-bite graph, so atom order,
    resonance localization and proper rotation cannot change the result.
    """
    dirs = vertex_dirs(geometry)
    if dirs is None:
        return ""
    return _poly.handedness(
        dirs,
        list(vertices),
        site_classes(mol, vertices, haptic, coordination) if classes is None else classes,
        chelate_edges(mol, vertices, haptic) if edges is None else edges,
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
