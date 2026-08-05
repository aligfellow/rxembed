"""Read a metal complex from a SMILES, and write one back as a canonical dative CXSMILES.

Both halves of one contract, which is why they share a module. The writer indexes its ``atomProp`` block by
POSITION in the string it has just written, and the reader has to parse with exactly the hydrogens the writer
counted or every index in the block addresses a different atom. Two modules with two owners is how that
agreement rots silently.

Two layers, and the split between them is the point:

- `dative_smiles` / `write_dative`: the CONSTITUTION. Connectivity, charges and ligand stereocentres, with
  every M-donor bond normalised to the ionic dative form. cis and trans give one string here, as do fac and
  mer.
- `canonical_smiles`: that string plus ``|atomProp:…|``, carrying the arrangement the SMILES grammar cannot
  say. OpenSMILES has a chirality class for 3 of the 12 rxembed polyhedra with geometric isomerism, so the
  block is the only carrier for the other nine.

Written here, read in `metal_isomers`: `stated_arrangement` puts an arrangement back onto a `Mol`, which is
an RDKit-property job on an isomer, not a string job, so the import runs one way only (this module imports
`metal_isomers`, never the reverse). The `s<n>±` grammar the two exchange is `metal_polyhedron`'s
`slot_note` / `read_slot_note`, since that leaf owns the slots being spelled and neither end can own both
halves.

`metal_polyhedron` keeps the slot scheme itself (`canonical_slots`, `rotation_group`, `seat_properly`): a
canonical vertex ordering is a property of the polyhedron, not of a string format. Only the rendering of one
is here.

`numpy + rdkit` like every module at this level, so the whole graph round trip
(``CXSMILES -> Isomer -> CXSMILES``) closes on a base install. Only turning *coordinates* into a Mol needs
perception, and that is `pipeline/perceive.py`'s job.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem

from .metal_core import (
    _PT,
    COORDINATION_METALS,
    VACANT,
    _chelate_edges,
    _donor_classes,
    connect_metal,
    donated_charge,
    ligand_valence,
    metal_indices,
)
from .metal_isomers import from_geometry
from .metal_polyhedron import canonical_slots, record, slot_note, vertex_dirs
from .utils import assign_stereo_from_3d, remove_bond

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__: the name `set_verbose` configures


def parse_smiles(smi):
    """Parse a SMILES to a Mol, raising a clear error instead of returning ``None`` (which crashes downstream).

    A CXSMILES carrying an `atomProp` block keeps its explicit hydrogens, because a block index is a position
    in the written atom order (`write_dative` returns exactly that order) and the default parser renumbers
    every atom underneath it by removing them. Only that case, so a plain SMILES is read exactly as before.

    The one SMILES door for both tiers, for the same reason `utils.assign_stereo_from_3d` is the one stereo
    door: a second parser that did not know about the block would read the arrangement onto the wrong atoms.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = "atomProp" not in smi
    mol = Chem.MolFromSmiles(smi, params)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smi!r}")
    return mol


# --- the constitution layer: dative M-L bonds, and the atom order the string was written in ---------------

_H_VALENCE = 1  # all SMILES will spend on a hydrogen; a second connection has to be dative
_METAL_STEREO_TAGS = frozenset(  # the non-tetrahedral classes perception leaves on a metal; see `write_dative`
    {Chem.ChiralType.CHI_SQUAREPLANAR, Chem.ChiralType.CHI_OCTAHEDRAL, Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL}
)


def _donate_to_metal(rw):
    """Rewrite every M-L bond a SMILES cannot spell covalently as a dative one, donor -> metal, in place.

    Two cases, one rule. An anionic donor held by a covalent bond is the charge counted twice: `[Cl-]` has a
    full octet, so `[Cl-][Ti+4]` will not sanitize, and donating spends the metal's valence instead. 18 of 45
    corpus structures, CisPlatin and TiCl4 among them, could not be written at all without that. And a donor
    with nothing but the metal to fill its shell is the same species written the other way round, so it takes
    the charge `donated_charge` reads off its valence and the metal takes the balance: `M=O` -> `[M2+]<-[O2-]`,
    where perception otherwise leaves an undervalent neutral `[O]` in the canonical string. `ml_distance` keys
    the M=O bond length off that same rule, so the written string and the embedded geometry cannot disagree.

    A donor whose ligand side is already full - an ammine, a phosphine, an aqua - moves no charge either way,
    but a covalent bond is still one valence more than it has: `[NH3][Pt]` will not parse. Perception writes
    those dative to begin with, so that third branch acts only on a sphere rebuilt from an `Isomer`, which
    hands its sigma donors back single on purpose (`_rebuild`). A partly filled donor is left exactly as it
    arrived, because `CN(C)[Pd]` is a legal neutral nitrogen and charging it is the Kekule artefact
    `donated_charge` refuses; the cost is that the amide and the alkyl are the two donor classes whose two
    Lewis forms still write two strings.
    """
    ionic, charged = [], {}
    for bond in rw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for m, d in ((i, j), (j, i)):
            if rw.GetAtomWithIdx(m).GetAtomicNum() not in COORDINATION_METALS:
                continue
            donor = rw.GetAtomWithIdx(d)
            single = bond.GetBondType() == Chem.BondType.SINGLE
            if donor.GetFormalCharge() < 0 and single:
                ionic.append((d, m))
            elif not donor.GetFormalCharge() and donated_charge(donor):
                ionic.append((d, m))
                charged.setdefault(d, m)  # a bridged donor charges once, against the first metal holding it
            elif single and ligand_valence(donor) >= _PT.GetDefaultValence(donor.GetAtomicNum()) > 0:
                ionic.append((d, m))  # a full shell: no charge to move, but no valence left to hold a bond
            break
    for d, m in charged.items():
        q = donated_charge(rw.GetAtomWithIdx(d))
        rw.GetAtomWithIdx(d).SetFormalCharge(q)
        rw.GetAtomWithIdx(m).SetFormalCharge(rw.GetAtomWithIdx(m).GetFormalCharge() - q)
    for d, m in ionic:
        remove_bond(rw, d, m)  # re-seats the bond LAST at the metal, so the donor's tag moves basis with it
        rw.AddBond(d, m, Chem.BondType.DATIVE)


def dative_smiles(mol):
    """Write the complex as a SMILES with dative M-donor bonds, checked to parse back to the same atoms.

    The constitution layer: connectivity, charges and ligand stereocentres. The metal's arrangement is not
    written, so cis and trans give one string, as do fac and mer; `canonical_smiles` is the layer that adds
    it. Not a species key on its own.
    """
    return write_dative(mol)[0]


def write_dative(mol):
    """Return ``(smiles, {atom: its position in that string's atom order})``: `dative_smiles` plus the order.

    The positions are what an `atomProp` block indexes, so the CXSMILES writer needs them and cannot get
    them from the string.

    Perception already writes M-donor bonds as dative, so this is `MolToSmiles` plus the two repairs SMILES
    needs. A hydrogen with more than one connection - a side-on H2, a bridging hydride, an H-bond relay
    perceived as a bond - has no valence left for a second single bond, so it keeps its shortest bond and
    donates through the rest. That is the real electron flow for sigma-donated H2 and a convention for the
    relay, which SMILES has no way to say otherwise.

    Raises rather than hand back a string that does not round-trip, since a SMILES you cannot read back is
    worse than none.
    """
    rw = Chem.RWMol(mol)
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() <= _H_VALENCE:
            continue
        h = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if pos is not None:  # without a geometry the graph order is all there is to go on
            nbrs.sort(key=lambda n: float(np.sum((pos[n] - pos[h]) ** 2)))
        for n in nbrs[1:]:
            remove_bond(rw, h, n)  # re-seats the bond LAST at `n`, so the partner's tag moves basis with it
            rw.AddBond(h, n, Chem.BondType.DATIVE)  # H donates: a dative bond spends the end atom's valence

    _donate_to_metal(rw)

    out = rw.GetMol()
    # E/Z references atoms picked from the bond's neighbour order, which a renumber does not update: 4 of the
    # 5 corpus structures unstable under reordering differed only in `/` and `\`. Geometry has no such order.
    if out.GetNumConformers():
        assign_stereo_from_3d(out)  # the one door, never the raw call: see its docstring on the dative basis

    # The metal's own tag is a permutation index over the neighbour order too, but no geometry settles it:
    # Fe(CO)5 writes @TB20, @TB14 or @TB13 for one molecule, and SMILES has a class for 3 of the 12
    # arrangements here. Dropping it takes the corpus 26 -> 40 stable under reordering; see the docstring
    # for what that costs.
    for atom in out.GetAtoms():
        if atom.GetChiralTag() in _METAL_STEREO_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_chiralPermutation"):
                atom.ClearProp("_chiralPermutation")
    out.UpdatePropertyCache(strict=False)
    smi = Chem.MolToSmiles(out)
    back = Chem.MolFromSmiles(smi)
    if back is None or Chem.AddHs(back).GetNumAtoms() != mol.GetNumAtoms():
        got = "does not parse" if back is None else f"parses back as {Chem.AddHs(back).GetNumAtoms()} atoms"
        raise ValueError(
            f"could not write a round-tripping SMILES for this complex ({mol.GetNumAtoms()} atoms): the "
            f"result {got}. The perceived graph is likely one SMILES cannot express (a hypervalent or "
            f"partial-bond centre); work from the Mol itself."
        )
    written = out.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]  # MolToSmiles sets it: position -> atom
    return smi, {int(a): p for p, a in enumerate(written)}


# --- the arrangement layer: the canonical slot note ------------------------------------------------------

_FACE_MIN = 3  # an eta2 face is one bond, and a bond's two ends are its own mirror: nothing to wind
_PATH_ENDS = 2  # an open face (an allyl) has exactly two atoms with a single face-neighbour


def _face_walk(mol, face):
    """Return ``(atoms in bond order, closed)`` for a haptic face, or ``None`` if it branches.

    A face is a cycle (Cp, arene) or an open chain (an allyl); `_collapse_haptic` admits both, so this
    cannot assume a ring. An open chain has to start at an END, or the walk dead-ends halfway and the answer
    would depend on whether the lowest atom index happened to land in the middle: measured, that made JIWHOQ
    and NUKHEG give two strings over four atom orderings.
    """
    inside = set(face)
    nbrs = {a: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in inside) for a in face}
    ends = [a for a in face if len(nbrs[a]) == 1]
    if any(len(nbrs[a]) > _PATH_ENDS for a in face) or len(ends) not in (0, _PATH_ENDS):
        return None
    walk = [min(ends) if ends else min(face)]
    while len(walk) < len(face):
        prev = walk[-2] if len(walk) > 1 else None
        step = [a for a in nbrs[walk[-1]] if a != prev]
        if not step:
            return None
        walk.append(min(step))
    return walk, not ends


def _face_winding(mol, pos, metal, face, ranks):
    """Return ``'+'``/``'-'`` for a planar-chiral haptic face, ``''`` when the face is its own mirror.

    A face is planar-chiral iff its sequence of atom classes differs from its own reverse (over every
    rotation too, when the face is a closed ring). Then one direction of travel is canonically first, and
    the sign of that direction's circulation about the metal-to-centroid axis is a mirror-sensitive fact
    that a slot number alone cannot hold: measured over `benchmark/corpus`, 6 of its 14 faces of eta>=3 are
    planar-chiral and the sign flips on every one of them under a mirror. A symmetric Cp's direction says
    only where the walk started, so keying on it would split two identical rings into two classes.

    An eta2 face is one bond and its two ends are its own mirror, so it has no winding. Its orientation
    about the metal-to-centroid axis still is a stereo element (DEYMIE's three side-on oximates make a
    propeller); nothing here carries that, and no slot number can.
    """
    if len(face) < _FACE_MIN:
        return ""
    walked = _face_walk(mol, face)
    if walked is None:
        return ""
    order, closed = walked
    n = len(order)
    cls = [ranks[a] for a in order]
    if closed:  # a ring reads from any atom, a chain only from its two ends
        fwd = min(tuple(cls[(s + i) % n] for i in range(n)) for s in range(n))
        rev = min(tuple(cls[(s - i) % n] for i in range(n)) for s in range(n))
    else:
        fwd, rev = tuple(cls), tuple(reversed(cls))
    if fwd == rev:
        return ""
    seq = order if fwd < rev else order[::-1]
    c = np.mean([pos[a] for a in order], axis=0)
    edges = zip(seq, seq[1:] + seq[:1] if closed else seq[1:], strict=False)  # an open face has no closing edge
    return "+" if float(sum(np.cross(pos[a] - c, pos[b] - c) for a, b in edges) @ (c - pos[metal])) > 0 else "-"


def _site_keys(iso, ranks, pos):
    """Return one order-invariant key per vertex (``None`` at a vacancy): what decides which sites tie.

    A sigma donor is its symmetry class. A haptic face is the sorted multiset of its ring atoms' classes,
    so two identical Cp rings tie and a Cp and an arene do not, plus its winding where the ring has one.
    Never an atom index and never a position in a string, so any writer folding on these agrees.

    A ring's winding needs a geometry; with ``pos`` ``None`` (an `Isomer` stated as a vertex ordering, not
    measured from a conformer) the face keeps its classes and drops the hand, which is honest: nothing in a
    vertex ordering says which way an unseen ring is wound.
    """
    keys = []
    for d in iso.vertices:
        if d == VACANT:
            keys.append(None)
        elif d in iso.haptic:
            face = iso.haptic[d]
            wind = "" if pos is None else _face_winding(iso.mol, pos, iso.metal, face, ranks)
            keys.append((tuple(sorted(ranks[a] for a in face)), wind))
        else:
            keys.append(((ranks[d],), ""))
    return keys


def _slot_notes(iso, keys, slots, at):
    """Return ``{atom: 's<n>'}``, pairing sorted sites with sorted slots inside each key class.

    The fold leaves interchangeable sites tied by construction, so which of them a tied frame numbered
    first is arbitrary; pairing within the class makes the rendered note independent of that choice. The
    sites are paired, not their atoms: a haptic face is several atoms holding one slot, and pairing atoms
    would hand two rings of one class five slots each.
    """
    by_key = {}
    for v, d in enumerate(iso.vertices):
        if d == VACANT:
            continue
        atoms = tuple(iso.haptic[d]) if d in iso.haptic else (d,)
        by_key.setdefault(keys[v], []).append((min(at[a] for a in atoms), atoms))
    notes = {}
    for key, group in by_key.items():
        for (_p, atoms), slot in zip(
            sorted(group), sorted(slots[v] for v, k in enumerate(keys) if k == key), strict=True
        ):
            for a in atoms:
                notes[a] = slot_note(slot, key[1])
    return notes


def _rebuild(iso):
    """Re-connect an `Isomer`'s stripped metal, in the Lewis form `dative_smiles` would derive from valence.

    An `Isomer` carries the metal as a bond-less surrogate, so its constitution has to be rebuilt before it
    can be written, and that rebuild decides whose charge is whose. Restoring every M-donor bond as dative keeps
    the input's charges exactly where they were, which is what a calculator wants and the opposite of what a
    canonical string wants: a complex drawn `[Pt](Cl)Cl` then came back `[Pt](<-[Cl])<-[Cl]`, a Pt(0) with
    two neutral chlorides, where the same molecule handed to `dative_smiles` directly gave `[Pt+2]` with two
    `[Cl-]`. Two doors, two strings, one species.

    So the sigma donors go back single and `_donate_to_metal` re-derives the ionic form from each donor's own
    valence, exactly as it does for a `Mol`. A haptic face does not: its ring atoms are one vertex sharing one
    donation, not n donors each one electron short, and a single bond per ring atom reads a Cp as five
    carbanions.
    """
    ring = {a for face in iso.haptic.values() for a in face}
    whole = connect_metal(iso.restore(Chem.Mol(iso.mol)), [b for b in iso.donor_bonds if b[0] in ring])
    return connect_metal(whole, [b for b in iso.donor_bonds if b[0] not in ring], order=Chem.BondType.SINGLE)


def canonical_smiles(source):
    """Write a metal complex as a canonical CXSMILES: a dative-SMILES core plus its arrangement.

    ``<dative core> |atomProp:...|``. Everything before the first ``|`` is a valid canonical dative SMILES
    that any RDKit pipeline reads, so ``text.split('|', 1)[0]`` is a constitution key; the block carries what
    the grammar cannot say. That is load-bearing rather than decorative: of the 12 rxembed polyhedra with
    geometric isomerism, OpenSMILES has a chirality class for 3, so the block is the only carrier of the
    arrangement for the other nine, and `dative_smiles` deliberately drops the metal's own tag.

    The block states, on the metal, the 3-letter geometry code and the Lambda/Delta word where the centre is
    chiral; on each donor, ``s<n>``, its canonical slot, with a ``+``/``-`` for a planar-chiral haptic ring's
    winding. A slot is a fact about the molecule only modulo the template's proper rotations, so it is
    minimised over those and no more: the full point group is ``proper x Z2`` and that Z2 is the handedness.

    `source` is an `Isomer`, whose arrangement is already stated, or a `Mol` with a conformer, whose
    arrangement is measured off it by `from_geometry`. Reading the string back needs no second verb:
    `enumerate_isomers` returns the one arrangement `metal_isomers.stated_arrangement` finds on it instead
    of enumerating.

    One metal centre: a note is per centre by construction, but no corpus of bridged complexes has been
    measured, so that path raises rather than claim a losslessness nothing has checked. A coordination
    number with no `POLYHEDRA` template raises for the same reason: a string with no arrangement in it would
    merge every isomer of that centre silently.
    """
    given = not isinstance(source, Chem.Mol)  # an Isomer; tested this way round so the Mol type is the import
    iso = source if given else from_geometry(source)
    dirs = vertex_dirs(iso.geometry)
    if dirs is None:
        raise ValueError(
            f"no polyhedron template for {iso.geometry!r}, so there is no slot scheme to write and the "
            f"arrangement would be lost silently; add a POLYHEDRA row for this coordination number"
        )
    complexed = _rebuild(iso) if given else source  # a Mol is already its own constitution
    if len(metal_indices(complexed)) != 1:
        raise NotImplementedError(
            f"canonical_smiles writes one metal centre; this complex has {len(metal_indices(complexed))}. A "
            f"per-centre note is what the format states, but no corpus of bridged complexes has been "
            f"measured, so this raises rather than write a string whose losslessness is unchecked"
        )
    core, at = write_dative(complexed)
    work = iso.mol  # the surrogate: its ligands are separate fragments, so a bite and a class both read right
    coordinating = [a for d in iso.vertices if d != VACANT for a in (iso.haptic.get(d) or (d,))]
    ranks = _donor_classes(work, coordinating)
    keys = _site_keys(iso, ranks, work.GetConformer().GetPositions() if work.GetNumConformers() else None)
    slots = canonical_slots(dirs, keys, _chelate_edges(work, iso.vertices, iso.haptic))
    geom = record(iso.geometry).code + (f"-{iso.chirality}" if iso.chirality else "")
    notes = {iso.metal: geom}
    notes.update(_slot_notes(iso, keys, slots, at))
    # `atomNote` rather than a key of our own: RDKit reads, writes and DRAWS it, so the arrangement is visible
    # in a depiction and survives a round trip through `MolToCXSmiles` without special handling. Atom index
    # order is not a choice either; RDKit re-emits the block sorted by index whatever order it was built in.
    block = ":".join(f"{at[a]}.atomNote.{v}" for a, v in sorted(notes.items(), key=lambda kv: at[kv[0]]))
    return f"{core} |atomProp:{block}|"
