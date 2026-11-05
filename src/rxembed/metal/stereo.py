"""Read the stereo at a metal: which donors and sites are alike, each bound face's configuration, the metal's hand.

These readers keep the graph they receive. `LigandFacts` projects a chemical graph once onto its surrogate graph
before they run, and centroid atoms may be added to that surrogate graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from rdkit import Chem

from rxembed.graph import COORDINATION_METALS, ETA2, frag_map, ligand_graph, torsion_path
from rxembed.metal.centres import vertex_atom
from rxembed.metal.polyhedra import DELTA, LAMBDA, VACANT, handedness, vertex_dirs
from rxembed.stereo import apply_encoded_bond_stereo
from rxembed.utils import EZ_RELATION, STRAIGHT, dihedral_angle, without_atom_maps, without_zero_bonds

_FACE_EPS = 1e-8
_PT = Chem.GetPeriodicTable()
_P_BLOCK_OUTER = frozenset(range(3, 8))  # RDKit's group index for main-group 13-17; d/f-block excluded below
_CHALCOGEN_OUTER = 6  # group 16: the terminal donor atom of a p-block hypervalent centre
# The mirror image of each metal hand and haptic face token; a symmetric s-cis face ('c') is its own.
MIRROR = {DELTA: LAMBDA, LAMBDA: DELTA, "+": "-", "-": "+", "P": "M", "M": "P", "c": "c"}


@dataclass(frozen=True, slots=True, kw_only=True)
class DonorRole:
    """Name one donor bond the surrogate strip removed: its `donor`, its `metal` and that metal's `atomic_num`."""

    donor: int
    metal: int
    atomic_num: int


class Reduced(NamedTuple):
    """Hold a hydrogen-reduced Mol and where each kept atom moved."""

    mol: Chem.Mol
    at: dict  # old atom index -> new


def remove_routine_hydrogens(mol, keep=()):
    """Return `mol` `Reduced` by its routine hydrogens; `keep` hydrogens stay explicit."""
    out, keep = Chem.Mol(mol), set(keep)
    for atom in out.GetAtoms():
        atom.SetIntProp("_rxembedOriginalIndex", atom.GetIdx())
        if atom.GetIdx() in keep and atom.GetAtomicNum() == 1 and not atom.GetIsotope():
            atom.SetBoolProp("_rxembedCoordinationH", True)
            atom.SetIsotope(1)
    params = Chem.RemoveHsParameters()
    params.removeDegreeZero = True
    params.removeDefiningBondStereo = True
    params.removeMapped = False  # as `parse_smiles`
    params.updateExplicitCount = True
    params.showWarnings = False  # protected donor hydrides are deliberately isotope-marked and retained
    out = Chem.RemoveHs(out, params, sanitize=False)
    at = {}
    for atom in out.GetAtoms():
        at[atom.GetIntProp("_rxembedOriginalIndex")] = atom.GetIdx()
        atom.ClearProp("_rxembedOriginalIndex")
        if atom.HasProp("_rxembedCoordinationH"):
            atom.SetIsotope(0)
            atom.ClearProp("_rxembedCoordinationH")
    out.UpdatePropertyCache(strict=False)
    return Reduced(out, at)


def _terminal_chalcogen_bond(bond):
    """Return whether a bond joins a main-group 13-17 centre to a group 16 donor with one heavy neighbour.

    RDKit never marks an expanded-octet X=O bond conjugated, so these bonds are added by this graph fact: X=O and
    X(+)-O(-) are one drawing choice, as a carboxylate's are.
    """
    for centre, donor in ((bond.GetBeginAtom(), bond.GetEndAtom()), (bond.GetEndAtom(), bond.GetBeginAtom())):
        p_block = (
            centre.GetAtomicNum() not in COORDINATION_METALS
            and _PT.GetNOuterElecs(centre.GetAtomicNum()) in _P_BLOCK_OUTER
        )
        terminal = sum(neighbor.GetAtomicNum() != 1 for neighbor in donor.GetNeighbors()) == 1
        if p_block and terminal and _PT.GetNOuterElecs(donor.GetAtomicNum()) == _CHALCOGEN_OUTER:
            return True
    return False


def _root_classes(mol, roots):
    """Classify roots by graph symmetry after erasing each conjugated system's drawn Lewis form.

    Resonance and metal-ligand redox move only bond orders and charges, so ranking with every conjugated bond one type
    and every charge zero gives the same identity as enumerating every form, at any size.
    """
    chemical = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() in (Chem.BondType.ZERO, Chem.BondType.DATIVE):
            chemical.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    try:
        chemical.UpdatePropertyCache(strict=False)
        Chem.SetConjugation(chemical)
        flat = Chem.RWMol(mol)
        for bond in chemical.GetBonds():
            if bond.GetIsConjugated() or _terminal_chalcogen_bond(bond):
                flat_bond = flat.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                flat_bond.SetBondType(Chem.BondType.AROMATIC)
                flat_bond.SetIsAromatic(False)
        for atom, source in zip(flat.GetAtoms(), mol.GetAtoms(), strict=True):
            atom.SetNumExplicitHs(source.GetTotalNumHs())
            atom.SetNoImplicit(True)
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
        flat.UpdatePropertyCache(strict=False)
        ranks = list(Chem.CanonicalRankAtoms(flat, breakTies=False, includeAtomMaps=False))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("RDKit could not canonicalize coordination-site identity") from exc
    # the flat rank names each class, so a hand never flips with the drawn Lewis form
    return {root: ranks[root] for root in roots}


def donor_classes(mol, donors):
    """Return `{donor: class}` by graph symmetry with each conjugated system's Lewis form erased (`_root_classes`)."""
    ranked, at = remove_routine_hydrogens(without_atom_maps(without_zero_bonds(mol)), donors)
    classes = _root_classes(ranked, [at[donor] for donor in donors])
    return {donor: classes[at[donor]] for donor in donors}


def site_classes(mol, sites, haptic=None, coordination=()):
    """Map occupied coordination sites to graph-symmetry classes.

    A marker joined to every atom of a haptic face ranks the rooted atom set, while temporary zero bonds retain
    terminal and bridging donor roles (`coordination`, `DonorRole`s) on the stripped surrogate graph.
    """
    haptic = haptic or {}
    occupied = [site for site in sites if site is not VACANT]
    atoms = {atom for site in occupied for atom in (haptic.get(site) or (site,))}
    atoms.update(role.donor for role in coordination)
    rw = Chem.RWMol(without_zero_bonds(mol))
    for role in coordination:
        atom = rw.GetAtomWithIdx(role.metal)
        atom.SetAtomicNum(role.atomic_num)
        # formal charge stays out, as in _root_classes: redox only moves charge between metal and ligand
        atom.SetFormalCharge(0)
        atom.SetIsotope(1000)
        if rw.GetBondBetweenAtoms(role.donor, role.metal) is None:
            rw.AddBond(role.donor, role.metal, Chem.BondType.ZERO)
    work = rw.GetMol()
    apply_encoded_bond_stereo(work)
    # Complete the donor's stereo carriers before removing H and rebasing its point tag.
    ranked, at = remove_routine_hydrogens(without_atom_maps(work), atoms)
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
    occupied = [vertex for vertex, value in enumerate(classes) if value is not None]
    targets = occupied if targets is None else list(targets)
    sources = occupied if sources is None else list(sources)
    yield from _place(classes, {} if links is None else links, targets, sources, {})


def _place(classes, links, targets, sources, assigned):
    """Extend `assigned` over the remaining targets, yielding each complete class- and link-preserving map."""
    if len(assigned) == len(targets):
        yield dict(assigned)
        return
    target = targets[len(assigned)]
    for source in sources:
        if source in assigned.values() or classes[target] != classes[source]:
            continue
        if any(
            links.get(frozenset((target, other))) != links.get(frozenset((source, mapped)))
            for other, mapped in assigned.items()
        ):
            continue
        assigned[target] = source
        yield from _place(classes, links, targets, sources, assigned)
        del assigned[target]


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
    if len(face) != ETA2 or mol.GetBondBetweenAtoms(*face) is None:
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
        ranks = list(Chem.ComputeAtomCIPRanks(without_atom_maps(mol))) if ranks is None else ranks
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
        if bond.GetStereo() in EZ_RELATION and len(refs) == 2:  # noqa: PLR2004 - two stereo refs
            cis = EZ_RELATION[bond.GetStereo()] == "cis"
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
                    "si" if 0 < (angle[ordered[1]] - angle[ordered[0]]) % 360 < STRAIGHT else "re",
                )
                for _atom, key, ordered in centres
            )
        )
    mirror = tuple(sorted((key, "si" if name == "re" else "re") for key, name in signature))
    return signature, mirror


def face_orientations(mol, face, ranks):
    """Return the tokens that name one haptic face's configurations, or none when it has one configuration.

    A face side is ``'+'`` or ``'-'``. A rotatable diene face (`torsion_path`) adds its s-trans helicity
    ``'P'`` or ``'M'``, and its s-cis form is a face side, or ``'c'`` when a mirror relates the two sides.
    """
    if len(face) == ETA2:  # a face has a side to name only when its two mirror orientations differ
        signature, mirror = eta2_signatures(mol, face)
        sides = ("+", "-") if signature and signature != mirror else ()
    else:
        sides = ("+", "-") if _canonical_face_walk(mol, face, ranks) is not None else ()
    if torsion_path(mol, face) is None:
        return sides
    return (*(sides or ("c",)), "P", "M")


def _face_side(normal, direction, area_scale):
    """Return a scale-invariant side of a face, or zero for a degenerate placement."""
    value = float(normal @ direction)
    scale = float(area_scale * np.linalg.norm(direction))
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0.0 or abs(value) <= _FACE_EPS * scale:
        return 0
    return 1 if value > 0.0 else -1


def _eta2_winding(mol, pos, metal, face, eta2_ranks):
    """Return an eta2 face's token from the re/si side the metal takes at each CIP-orderable centre, or ``''``."""
    try:
        eta2_ranks = list(Chem.ComputeAtomCIPRanks(without_atom_maps(mol))) if eta2_ranks is None else eta2_ranks
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


def face_winding(mol, pos, metal, face, ranks, eta2_ranks=None):
    """Return the canonical configuration token of a haptic face (see `face_orientations`), or ''.

    Proper rotation keeps the token and reflection mirrors it. A rotatable diene face is s-cis when its C1-C2-C3-C4
    torsion is under 90 degrees in magnitude and s-trans above, with no fitted margin; s-trans is named by that
    torsion's helicity, P when positive, so it needs no priorities.
    """
    if len(face) == ETA2:
        return _eta2_winding(mol, pos, metal, face, eta2_ranks)
    canonical = _canonical_face_walk(mol, face, ranks)
    if (path := torsion_path(mol, face)) is not None:
        phi = np.radians(dihedral_angle(*pos[path]))
        cos, sin = np.cos(phi), np.sin(phi)
        if not np.isfinite(phi) or abs(cos) <= _FACE_EPS:
            return ""
        if cos < 0.0:
            return "" if abs(sin) <= _FACE_EPS else "P" if sin > 0.0 else "M"
        if canonical is None:
            return "c"
    if canonical is None:
        return ""
    sequence, closed = canonical
    centre = np.mean([pos[atom] for atom in sequence], axis=0)
    following = sequence[1:] + sequence[:1] if closed else sequence[1:]
    terms = [np.cross(pos[a] - centre, pos[b] - centre) for a, b in zip(sequence, following, strict=False)]
    side = _face_side(sum(terms, start=np.zeros(3)), centre - pos[metal], sum(np.linalg.norm(term) for term in terms))
    return "+" if side > 0 else "-" if side < 0 else ""


def _face_reference(ranks, sequence):
    """Return a face's unique top-ranked atom and its higher-ranked ring neighbour, or None on a tie."""
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


def face_descriptors(mol, donors, haptic, windings):
    """Return re/si or ``Rₚ``/``Sₚ`` labels for priority-orderable haptic faces.

    Tied or stereo-dependent priorities deliberately remain unnamed; their canonical winding is still lossless.
    """
    if not windings:
        return {}
    try:
        priorities = Chem.ComputeAtomCIPRanks(without_atom_maps(mol))
        unmarked = Chem.Mol(mol)
        Chem.RemoveStereochemistry(unmarked)
        constitutional = Chem.ComputeAtomCIPRanks(without_atom_maps(unmarked))
    except (RuntimeError, ValueError):
        return {}
    classes = donor_classes(mol, donors)
    out = {}
    for dummy, winding in windings.items():
        face = haptic.get(dummy, ())
        if len(face) == ETA2:
            signature, mirror = eta2_signatures(mol, face, priorities)
            if winding in "+-" and signature and signature != mirror:
                selected = min(signature, mirror) if winding == "+" else max(signature, mirror)
                out[dummy] = f"({','.join(name for _key, name in selected)})"
            continue
        canonical = _canonical_face_walk(mol, face, classes)
        if winding not in "+-" or canonical is None or not canonical[1]:
            continue
        sequence = canonical[0]
        choice = _face_reference(priorities, sequence)
        if choice is None or choice != _face_reference(constitutional, sequence):
            continue
        pilot, toward_second = choice
        following = sequence[(sequence.index(pilot) + 1) % len(sequence)]
        sense = (1 if winding == "+" else -1) * (1 if toward_second == following else -1)
        out[dummy] = "Rₚ" if sense < 0 else "Sₚ"
    return out


def chelate_links(mol, vertices, haptic=None, distances=None):
    """Return same-ligand vertex pairs labelled by their shortest graph distance."""
    haptic = haptic or {}
    frag = frag_map(mol)
    occupied = [vertex for vertex in range(len(vertices)) if vertices[vertex] is not VACANT]
    pairs = [
        (a, b)
        for i, a in enumerate(occupied)
        for b in occupied[i + 1 :]
        if frag[vertex_atom(haptic, vertices[a])] == frag[vertex_atom(haptic, vertices[b])]
    ]
    if not pairs:
        return {}
    distances = Chem.GetDistanceMatrix(ligand_graph(mol)) if distances is None else distances
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
    return handedness(
        dirs,
        list(vertices),
        site_classes(mol, vertices, haptic, coordination) if classes is None else classes,
        chelate_links(mol, vertices, haptic) if links is None else links,
    )
