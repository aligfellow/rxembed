"""Parse and write the SMILES and CX SMILES of a species, organic or metal, with the atom notes CX SMILES carry.

SMILES are written in canonical ionic-dative form with constitution, point stereo and E/Z where the graph allows. CX
fields add zero-order contacts, atropisomer stereo, metal polyhedron slots and haptic winding. The notes (a metal's
polyhedron and hand, a donor's slot) are read from a Mol (`specified_species`), written onto one (`mol_from_species`)
and made canonical over equivalent metals (`canonical_notes`); they address the atom order just written, so parsing
and writing share this module. Naming a Mol, which first reads it, is `reading`'s.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import NamedTuple

from rdkit import Chem, rdBase

from rxembed.graph import COORDINATION_METALS, bound_metals, haptic_sites, hydrogen_legs, metal_indices
from rxembed.identity.species import (
    Species,
    centre_from_slots,
    centre_state,
    graph_with_stereo,
    site_keys,
)
from rxembed.metal.centres import (
    canonical_metal_graph,
    chemical_graph,
    materialized_states,
    state_with_winding,
    symmetric_metals,
)
from rxembed.metal.polyhedra import (
    HAND_GLYPHS,
    POLYHEDRA,
    SLOT_BOND_PROP,
    VACANT,
    least_frames,
    read_slot_notes,
    record,
    resolve_geometry,
    slot_note,
    vertex_dirs,
)
from rxembed.metal.sphere import LigandFacts
from rxembed.metal.stereo import (
    chelate_links,
    donor_classes,
    equivalent_site_assignments,
    face_orientations,
    remove_routine_hydrogens,
)
from rxembed.stereo import (
    ATROP_STEREO,
    atrop_code,
    atrop_tagged,
    bond_stereo_code,
    defined_stereo_label,
    encoded_bond_stereo,
)
from rxembed.utils import (
    EZ_RELATION,
    POINT_TAGS,
    TETRAHEDRAL_TAGS,
    mirror_tag,
    remove_bond,
    with_conformers,
)

logger = logging.getLogger("rxembed.metal")  # shared with the metal modules, under "rxembed" (`set_verbose`)

_NOTED_BONDS = frozenset({Chem.BondType.ZERO, Chem.BondType.DATIVE})
_TAG_CODES = {tag: code for code, tag in POINT_TAGS.items()}
_CX_BOND_FIELD = re.compile(r"(?:^|,)([ct]):((?:\d+(?:,\d+)*)?)(?=,|$)")
_CX_BOND_PREFIX = re.compile(r"(?:^|,)[ct]:")


def _repair_haptic_bond_stereo(mol):
    """Replace shared-metal E/Z references with ligand-side references without changing CIP."""
    for bond in mol.GetBonds():
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in EZ_RELATION or len(refs) != 2:  # noqa: PLR2004 - two stereo refs
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        shared = {n.GetIdx() for n in begin.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS} & {
            n.GetIdx() for n in end.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS
        }
        left = [n.GetIdx() for n in begin.GetNeighbors() if n.GetIdx() != end.GetIdx() and n.GetIdx() not in shared]
        right = [n.GetIdx() for n in end.GetNeighbors() if n.GetIdx() != begin.GetIdx() and n.GetIdx() not in shared]
        if not shared or not left or not right:
            continue
        expected = bond_stereo_code(mol, bond.GetIdx()) if len(set(refs)) == 2 else None  # noqa: PLR2004
        bond.SetStereoAtoms(min(left), min(right))
        if expected is None:
            continue
        for tag in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            bond.SetStereo(tag)
            if bond_stereo_code(mol, bond.GetIdx()) == expected:
                break


def _validate_cx_bond_stereo(smi, mol):
    """Refuse a malformed, repeated or unaddressable CX c:/t: double-bond field of `smi`, read on `mol`."""
    block = smi.partition("|")[2].rpartition("|")[0]
    if not block:
        return
    matches = _CX_BOND_FIELD.findall(block)
    if len(_CX_BOND_PREFIX.findall(block)) != len(matches):
        raise ValueError("invalid CX c:/t: double-bond stereo field")
    markers = {}
    for marker, values in matches:
        if not values:
            raise ValueError("invalid empty CX c:/t: double-bond stereo field")
        for value in values.split(","):
            idx = int(value)
            if idx in markers:
                raise ValueError(f"contradictory or duplicate CX c:/t: stereo for bond {idx}")
            markers[idx] = marker
    params = Chem.SmilesWriteParams()
    params.canonical = False  # preserve the parsed traversal: CX bond positions address that written order
    params.cleanStereo = False
    Chem.MolToSmiles(mol, params)
    order = list(mol.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
    for idx, marker in markers.items():
        if idx >= len(order):
            raise ValueError(f"CX {marker}: field names missing bond {idx}")
        bond = mol.GetBondWithIdx(order[idx])
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in EZ_RELATION or len(refs) != 2 or len(set(refs)) != 2:  # noqa: PLR2004
            raise ValueError(f"CX {marker}:{idx} did not define valid double-bond stereo")


def parse_smiles(smi, *, remove_hs=True):
    """Return the sanitised Mol of one SMILES, metal or organic, keeping coordinated ligand stereo.

    Sanitising before removing hydrogens keeps tags such as ``[N@H]`` beside a dative bond. A metal's SP, TB or OH tag
    is checked before RDKit's cleanup can drop it. CX inputs keep explicit coordination hydrogens, since ``atomProp``
    indices address the written atom order. Invalid input raises.
    """
    params = Chem.SmilesParserParams()
    params.sanitize = False  # RDKit's integrated cleanup erases `[N@H]` when its fourth neighbour is dative.
    params.removeHs = False
    mol = Chem.MolFromSmiles(smi, params)
    if mol is not None:
        raw = Chem.Mol(mol)
        for atom in raw.GetAtoms():  # RDKit's cleanup drops a tag it numbers 0, so it is judged here first
            if atom.GetAtomicNum() in COORDINATION_METALS and atom.GetChiralTag() in NATIVE_TAGS:
                native_tag_vertices(atom, atom.GetDegree() + atom.GetNumExplicitHs())
        raw_tags = {
            atom.GetIdx(): atom.GetChiralTag() for atom in raw.GetAtoms() if atom.GetChiralTag() in TETRAHEDRAL_TAGS
        }
        try:
            Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            try:
                raw.UpdatePropertyCache(strict=False)
                Chem.AssignRadicals(raw)  # a bracket atom's unfilled valence is the radical the SMILES states
                mol = canonical_metal_graph(raw)
                Chem.SanitizeMol(mol)
                logger.info("parse: normalized invalid covalent metal bonds to dative")
            except (RuntimeError, ValueError):
                mol = None
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smi!r}")
    for index, tag in raw_tags.items():
        mol.GetAtomWithIdx(index).SetChiralTag(tag)
    Chem.SetBondStereoFromDirections(mol)
    _repair_haptic_bond_stereo(mol)
    _validate_cx_bond_stereo(smi, mol)
    encoded_bond_stereo(mol)  # raises on a malformed CX E/Z record
    bind_slot_notes(mol)
    if remove_hs and "atomProp" not in smi:
        keep = Chem.RemoveHsParameters()
        keep.removeMapped = False  # a mapped hydrogen is one the caller names
        mol = Chem.RemoveHs(mol, keep)
    return mol


# --- the constitution layer: dative M-L bonds, and the atom order the string was written in ---------------

_H_VALENCE = 1  # ordinary hydrogen; [H+] can encode a two-leg, three-centre bridge
_H_BRIDGE_VALENCE = 2


def _bond_positions(graph):
    """Return each bond's written position, keyed by its end atoms' written positions, after `MolToSmiles`."""
    atoms = list(graph.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    bonds = list(graph.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
    at = {idx: pos for pos, idx in enumerate(atoms)}
    return {
        frozenset((at[bond.GetBeginAtomIdx()], at[bond.GetEndAtomIdx()])): pos
        for pos, idx in enumerate(bonds)
        for bond in (graph.GetBondWithIdx(idx),)
    }


def _flip_stereo(graph, points, pairs):
    """Mirror each listed point centre and invert each listed double bond in place, for the next write trial."""
    for idx in points:
        atom = graph.GetAtomWithIdx(idx)
        tag = atom.GetChiralTag()
        atom.SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CW if tag == Chem.ChiralType.CHI_UNSPECIFIED else mirror_tag(tag)
        )
    for i, j in pairs:
        bond = graph.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        left, right = (
            next(
                (
                    adjacent.GetOtherAtomIdx(end.GetIdx())
                    for adjacent in end.GetBonds()
                    if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondType() == Chem.BondType.SINGLE
                ),
                None,
            )
            for end in (bond.GetBeginAtom(), bond.GetEndAtom())
        )
        if left is None or right is None or left == right:
            continue
        bond.SetStereoAtoms(left, right)
        flipped = Chem.BondStereo.STEREOZ if EZ_RELATION.get(bond.GetStereo()) == "trans" else Chem.BondStereo.STEREOE
        bond.SetStereo(flipped)


def _writes(graph, params, rebase):
    """Yield each trial SMILES with the copy that holds its output order; canonical writes try RDKit's default first."""
    for writer in (None, params) if rebase else (params,):
        written = Chem.Mol(graph)  # MolToSmiles may rebase tags, so never write the next trial in place
        smi = Chem.MolToSmiles(written) if writer is None else Chem.MolToSmiles(written, writer)
        # Z: indexes this writer's bond order; MolToCXSmiles writes datives single, so its core cannot replace ours
        zero = [
            str(pos)
            for pos, idx in enumerate(written.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
            if written.GetBondWithIdx(idx).GetBondType() == Chem.BondType.ZERO
        ]
        if zero:
            smi = _append_cx(smi, [f"Z:{','.join(zero)}"])
        yield smi, written


@dataclass(frozen=True)
class _StereoTargets:
    """Hold the proved ligand stereo that each parse-back of a written SMILES must reproduce."""

    mol: Chem.Mol  # the source graph, carrying every wanted CW/CCW tag
    wanted: dict  # atom -> R/S/r/s or CW/CCW
    wanted_bonds: dict  # frozenset atom pair -> E/Z
    tags: dict  # wanted centres whose CIP code the source already proves; CIP runs once, not per trial
    native: dict  # wanted centres carrying a tetrahedral tag on the source

    @classmethod
    def of(cls, mol, wanted, wanted_bonds):
        """Return the targets for `mol`, reading its CIP codes before any trial is written."""
        defined = defined_stereo_label(mol, metal_indices(mol)).points if wanted else {}
        tags = {idx: mol.GetAtomWithIdx(idx).GetChiralTag() for idx, code in wanted.items() if defined.get(idx) == code}
        native = {
            idx: mol.GetAtomWithIdx(idx).GetChiralTag()
            for idx in wanted
            if mol.GetAtomWithIdx(idx).GetChiralTag() in POINT_TAGS.values()
        }
        return cls(mol, wanted, wanted_bonds, tags, native)

    def _native_tag_matches(self, graph, points, index, target):
        """Return whether `target` keeps the explicit tag of a centre RDKit cannot CIP-rank, such as [S@+2].

        RDKit keeps such a tag but cannot label it or match it chirally, so the tag itself is compared.
        """
        return (
            index in self.native
            and points.get(target) is None
            and graph.GetAtomWithIdx(target).GetChiralTag() == self.native[index]
        )

    def _same_hand(self, graph, mapped, idx, tag):
        """Return whether `graph` gives centre `idx` the hand `tag` under the atom map `mapped`.

        The local permutation is compared with every carrier present; correcting CIP labels instead can oscillate when
        one fix changes a neighbouring pseudoasymmetric label.
        """
        source, target = Chem.Mol(self.mol), Chem.Mol(graph)
        for view in (source, target):
            for bond in view.GetBonds():
                if bond.GetBondType() == Chem.BondType.DATIVE:
                    bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
            view.UpdatePropertyCache(strict=False)
        for atom in source.GetAtoms():
            atom.SetChiralTag(tag if atom.GetIdx() == idx else Chem.ChiralType.CHI_UNSPECIFIED)
        for atom in target.GetAtoms():
            if atom.GetIdx() != mapped[idx]:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for old, new in mapped.items():
            isotope = 1000 + old  # force RDKit's chiral match onto this known atom mapping
            source.GetAtomWithIdx(old).SetIsotope(isotope)
            target.GetAtomWithIdx(new).SetIsotope(isotope)
        return target.HasSubstructMatch(source, useChirality=True)

    def wrong(self, graph, mapped):
        """Return the source centres and double bonds a parse-back `graph` gets wrong under `mapped`."""
        stereo = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = stereo.points, stereo.bonds
        wrong_points = [
            idx
            for idx, code in self.wanted.items()
            if idx not in self.tags
            and code in {"R", "S", "r", "s"}
            and points.get(mapped[idx]) != code
            and not self._native_tag_matches(graph, points, idx, mapped[idx])
        ]
        coded = {idx: POINT_TAGS[code] for idx, code in self.wanted.items() if code in POINT_TAGS}
        wrong_points += [
            idx
            for idx, tag in (coded | self.native | self.tags).items()
            if len(mapped) != self.mol.GetNumAtoms()
            or idx not in mapped
            or not (
                self._same_hand(graph, mapped, idx, tag) or self._native_tag_matches(graph, points, idx, mapped[idx])
            )
        ]
        wrong_bonds = [
            pair
            for pair, code in self.wanted_bonds.items()
            if bonds.get(frozenset(mapped[idx] for idx in pair)) != code
        ]
        return wrong_points, wrong_bonds

    def compatible_order(self, graph):
        """Return the source-to-written atom order when `graph` matches the source with every wanted code, else None."""
        query = Chem.Mol(self.mol)
        for bond in query.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            for adjacent in (*bond.GetBeginAtom().GetBonds(), *bond.GetEndAtom().GetBonds()):
                adjacent.SetBondDir(Chem.BondDir.NONE)
        match = graph.GetSubstructMatch(query, useChirality=True)
        if len(match) != self.mol.GetNumAtoms():
            return None
        stereo = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = stereo.points, stereo.bonds
        if any(code in {"R", "S", "r", "s"} and points.get(match[idx]) != code for idx, code in self.wanted.items()):
            return None
        if any(bonds.get(frozenset(match[idx] for idx in pair)) != code for pair, code in self.wanted_bonds.items()):
            return None
        order = [None] * len(match)
        for old, new in enumerate(match):
            order[new] = old
        return order


def _rebase_native_stereo(mol, wanted, wanted_bonds, params):
    """Write through RDKit's coordinate-bond canonical order, so a tag beside a dative bond keeps its hand.

    RDKit's arrow writer ignores outgoing dative bonds when it rebases a point tag; its coordinate-bond CX writer counts
    every carrier. That canonical order is parsed once and arrows written in it, since canonicalising corrected tags
    again cycles.
    """
    for bond in mol.GetBonds():
        if bond.GetBondDir() in (Chem.BondDir.UNKNOWN, Chem.BondDir.EITHERDOUBLE):
            bond.SetBondDir(Chem.BondDir.NONE)
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    params.includeDativeBonds = False
    Chem.MolToSmiles(mol, params)
    order = list(mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    positions = {idx: pos for pos, idx in enumerate(order)}
    canonical = Chem.MolToCXSmiles(
        mol, params, Chem.CXSmilesFields.CX_COORDINATE_BONDS | Chem.CXSmilesFields.CX_ZERO_BONDS
    )
    back = parse_smiles(canonical, remove_hs=False)
    mapped_wanted = {positions[idx]: code for idx, code in wanted.items()}
    for idx, code in wanted.items():
        if code not in POINT_TAGS:
            continue
        tag = back.GetAtomWithIdx(positions[idx]).GetChiralTag()
        if tag not in POINT_TAGS.values():
            raise ValueError(f"native canonicalization lost point stereo at atom {idx}")
        mapped_wanted[positions[idx]] = _TAG_CODES[tag]
    mapped_bonds = {frozenset(positions[idx] for idx in pair): code for pair, code in wanted_bonds.items()}
    smi, rebased_order, rebased_bonds = _write_native_stereo(back, mapped_wanted, mapped_bonds, rebase=False)
    return smi, [order[idx] for idx in rebased_order], rebased_bonds


def _write_native_stereo(mol, wanted, wanted_bonds, *, rebase=True):
    """Return ``(smiles, written atom order, bond positions)``, keeping only stereo proved on the ligand graph.

    RDKit's writer can rebase a tag, so each trial is parsed back and its wrong centres and bonds flipped for the next.
    ``rebase=False`` writes the atom order as given.
    """
    mol = Chem.Mol(mol)
    params = Chem.SmilesWriteParams()
    params.cleanStereo = False  # every retained tag was proved upstream; RDKit otherwise deletes chiral amines
    params.canonical = rebase
    if rebase and any(
        bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() in wanted for bond in mol.GetBonds()
    ):
        return _rebase_native_stereo(mol, wanted, wanted_bonds, params)
    targets = _StereoTargets.of(mol, wanted, wanted_bonds)
    candidate = Chem.Mol(mol)
    for _ in range(max(3, len(wanted) + len(wanted_bonds) + 2)):
        for bond in candidate.GetBonds():
            if bond.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT):
                bond.SetBondDir(Chem.BondDir.NONE)
        if wanted_bonds:
            Chem.SetDoubleBondNeighborDirections(candidate)
        for smi, written_mol in _writes(candidate, params, rebase):
            order = list(written_mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
            if not wanted and not wanted_bonds:
                return smi, order, _bond_positions(written_mol)
            with rdBase.BlockLogs():
                back = parse_smiles(smi, remove_hs=False)
            if rebase and (compatible := targets.compatible_order(back)) is not None:
                return smi, compatible, _bond_positions(written_mol)
            wrong_points, wrong_bonds = targets.wrong(back, {idx: pos for pos, idx in enumerate(order)})
            if not wrong_points and not wrong_bonds:
                return smi, order, _bond_positions(written_mol)
        _flip_stereo(candidate, wrong_points, wrong_bonds)
    detail = {"atom(s)": sorted(wrong_points), "bond(s)": sorted(tuple(sorted(pair)) for pair in wrong_bonds)}
    where = " and ".join(f"{name} {items}" for name, items in detail.items() if items)
    raise ValueError(f"could not write ligand stereo at {where}")


def _writable_ez(mol, pair):
    """Return whether plain SMILES can write this double bond's E/Z: a single bond at each end, no shared metal."""
    bond = mol.GetBondBetweenAtoms(*pair)
    if bond is None:
        return False
    ends = (bond.GetBeginAtom(), bond.GetEndAtom())
    shared_metals = set.intersection(
        *(
            {neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() in COORDINATION_METALS}
            for atom in ends
        )
    )
    return not shared_metals and all(
        any(adj.GetIdx() != bond.GetIdx() and adj.GetBondType() == Chem.BondType.SINGLE for adj in atom.GetBonds())
        for atom in ends
    )


def _clear_bond_stereo(mol, pairs):
    """Clear selected E/Z tags and their adjacent directional bonds in place."""
    for pair in pairs:
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None:
            continue
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for adjacent in atom.GetBonds():
                if adjacent.GetIdx() != bond.GetIdx():
                    adjacent.SetBondDir(Chem.BondDir.NONE)


def _atom_key(atom):
    """Return the atom fields a SMILES round trip must keep."""
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetIsAromatic(),
    )


def _bond_table(mol, mapping):
    """Return each mapped bond's order, aromaticity and dative direction, keyed by its mapped end atoms."""
    out = {}
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in mapping or bond.GetEndAtomIdx() not in mapping:
            continue  # a routine H may be represented by its neighbour's total-H count
        begin, end = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        pair = tuple(sorted((begin, end)))
        direction = (begin, end) if bond.GetBondType() == Chem.BondType.DATIVE else None
        out[pair] = (bond.GetBondType(), bond.GetIsAromatic(), direction)
    return out


def _round_trip_graph_error(expected, parsed, written):
    """Return the first constitutional change a parsed SMILES makes, else None.

    RDKit's own `RemoveHs` decides which hydrogens a round trip keeps.
    """
    if parsed is None:
        return "does not parse"
    if len(set(written)) != len(written) or parsed.GetNumAtoms() != len(written):
        return "has an inconsistent written atom map"
    for atom, idx in zip(parsed.GetAtoms(), written, strict=True):
        atom.SetIntProp("_rxembedWritten", int(idx))
    actual = Chem.RemoveHs(parsed)
    at = {atom.GetIntProp("_rxembedWritten"): atom.GetIdx() for atom in actual.GetAtoms()}
    omitted = set(range(expected.GetNumAtoms())) - set(at)
    if any(
        expected.GetAtomWithIdx(idx).GetAtomicNum() != 1
        or expected.GetAtomWithIdx(idx).GetDegree() != 1
        or any(
            neighbor.GetAtomicNum() in COORDINATION_METALS for neighbor in expected.GetAtomWithIdx(idx).GetNeighbors()
        )
        for idx in omitted
    ):
        return "drops a non-routine explicit atom"
    for old, new in at.items():
        if _atom_key(expected.GetAtomWithIdx(old)) != _atom_key(actual.GetAtomWithIdx(new)):
            return f"changes atom {old}"
    if _bond_table(expected, at) != _bond_table(actual, {idx: idx for idx in range(actual.GetNumAtoms())}):
        return "changes connectivity, bond order, or dative direction"
    return None


def _dative_hydrogens(mol):
    """Return `mol` with the excess legs of an ordinary hydrogen made dative; a two-leg [H+] bridge stays.

    The first leg by `hydrogen_legs` stays covalent. Retyping a leg of an [H+] bridge would make a radical.
    """
    rw = Chem.RWMol(mol)
    for h, ordered in hydrogen_legs(mol, COORDINATION_METALS).items():
        atom = mol.GetAtomWithIdx(h)
        nbrs = [bond.GetOtherAtomIdx(h) for bond in atom.GetBonds() if bond.GetValenceContrib(atom) > 0]
        if len(nbrs) <= _H_VALENCE or (len(nbrs) == _H_BRIDGE_VALENCE and atom.GetFormalCharge() == 1):
            continue
        for n in [neighbor for neighbor in ordered if neighbor in nbrs][1:]:
            remove_bond(rw, h, n)  # re-seats the bond last at `n`, so the partner's tag moves basis with it
            rw.AddBond(h, n, Chem.BondType.DATIVE)  # H donates: a dative bond spends the end atom's valence
    return rw.GetMol()


def _ez_end_hydrogens(mol, wanted_bonds):
    """Return `mol` with explicit hydrogens on each wanted E/Z end whose only possible reference is hydrogen."""
    ends = set()
    for pair in wanted_bonds:
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None:
            continue
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if atom.GetTotalNumHs() and not any(
                adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondType() == Chem.BondType.SINGLE
                for adjacent in atom.GetBonds()
            ):
                ends.add(atom.GetIdx())
    if not ends:
        return mol
    return Chem.AddHs(mol, onlyOnAtoms=sorted(ends), addCoords=bool(mol.GetNumConformers()))


def _drop_unproved_stereo(mol, wanted, wanted_bonds):
    """Clear in place every point tag, E/Z and SP, TB or OH tag that `wanted` and `wanted_bonds` do not name."""
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in POINT_TAGS.values() and atom.GetIdx() not in wanted:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    _clear_bond_stereo(
        mol,
        [
            pair
            for bond in mol.GetBonds()
            if bond.GetStereo() in EZ_RELATION
            and (pair := frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) not in wanted_bonds
        ],
    )
    # bound metals' SP/TB/OH tags are already notes; any other is an atom-order permutation nothing reads
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in NATIVE_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.ClearProp("_chiralPermutation")


def _kept_hydrogens(mol, wanted_bonds):
    """Return the hydrogens the SMILES keeps: metal-bound ones, which can own a slot, and sole E/Z references."""
    keep = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 1 and any(n.GetAtomicNum() in COORDINATION_METALS for n in atom.GetNeighbors())
    }
    for pair in wanted_bonds:
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is None:
            continue
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if not any(
                adjacent.GetIdx() != bond.GetIdx()
                and adjacent.GetBondType() == Chem.BondType.SINGLE
                and adjacent.GetOtherAtom(atom).GetAtomicNum() != 1
                for adjacent in atom.GetBonds()
            ):
                keep.update(neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
    return keep


def _atoms_with_h(mol):
    """Return the atom count with every hydrogen explicit."""
    return mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)


def _check_round_trip(mol, smi, written, expected):
    """Raise unless `smi` parses back to `mol`'s graph with `expected` atoms including hydrogen."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    back = Chem.MolFromSmiles(smi, params)
    actual = None if back is None else _atoms_with_h(back)
    graph_error = _round_trip_graph_error(mol, back, written)
    if actual == expected and not graph_error:
        return
    got = "does not parse" if actual is None else f"parses back as {actual} atoms"
    if graph_error and actual == expected:
        got = graph_error
    raise ValueError(
        f"could not write a round-tripping SMILES for this complex ({expected} atoms with H): the result {got}; "
        "use the Mol itself"
    )


def write_dative(mol, stereo):
    """Return canonical dative SMILES, its atom and bond position maps, and the E/Z it could not write.

    `mol` is a canonical metal graph whose tags carry `stereo`; excess hydrogen legs become dative first, each edit
    carrying the tags. Only atoms that stay explicit have a position. Raises rather than return a SMILES that will not
    round-trip.
    """
    source = Chem.Mol(mol)
    source.UpdatePropertyCache(strict=False)
    expected = _atoms_with_h(source)
    # set raw CW/CCW codes as tags now, so the edits below carry them as they carry any tag
    raw = {idx: POINT_TAGS[code] for idx, code in stereo.points.items() if code in POINT_TAGS}
    for idx, tag in raw.items():
        source.GetAtomWithIdx(idx).SetChiralTag(tag)
    out = _dative_hydrogens(source)
    out.UpdatePropertyCache(strict=False)
    wanted = stereo.points
    wanted_bonds = stereo.bonds
    out = _ez_end_hydrogens(out, wanted_bonds)
    if not stereo:
        # no stereo to write: drop what RDKit would infer
        Chem.RemoveStereochemistry(out)
        for bond in out.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)  # RDKit leaves single-bond atrop tags behind.
    _drop_unproved_stereo(out, wanted, wanted_bonds)
    out, reduced = remove_routine_hydrogens(out, _kept_hydrogens(out, wanted_bonds))
    original = {new: old for old, new in reduced.items()}
    wanted = {
        reduced[idx]: _TAG_CODES[out.GetAtomWithIdx(reduced[idx]).GetChiralTag()] if idx in raw else code
        for idx, code in wanted.items()
        if idx in reduced
    }
    wanted_bonds = {
        frozenset(reduced[idx] for idx in pair): code for pair, code in wanted_bonds.items() if pair <= reduced.keys()
    }
    unwritable = {pair for pair in wanted_bonds if not _writable_ez(out, pair)}
    _clear_bond_stereo(out, unwritable)
    wanted_bonds = {pair: code for pair, code in wanted_bonds.items() if pair not in unwritable}
    smi, written, bonds = _write_native_stereo(out, wanted, wanted_bonds)
    _check_round_trip(out, smi, written, expected)
    return (
        smi,
        {original[int(a)]: p for p, a in enumerate(written) if original[int(a)] < source.GetNumAtoms()},
        bonds,
        unwritable,
    )


def _atrop_bond_stereo(core, stereo, at):
    """Return the native CX wU/wD field retaining every assigned atropisomer axis of `stereo`."""
    axes = stereo.axes
    if not axes:
        return []
    parsed = parse_smiles(core, remove_hs=False)
    targets = []
    for (i, j), target in axes.items():
        if i not in at or j not in at:
            raise ValueError(f"could not name atropisomer stereo on bond {i}-{j}")
        mapped = (at[i], at[j])
        if parsed.GetBondBetweenAtoms(*mapped) is None:
            raise ValueError(f"could not locate atropisomer bond {i}-{j} in the written CXSMILES")
        if (parsed := atrop_tagged(parsed, mapped, target)) is None:
            raise ValueError(f"could not retain atropisomer stereo on bond {i}-{j}")
        targets.append((mapped, target))
    params = Chem.SmilesWriteParams()
    params.canonical = False  # CX bond indices must address the already-canonical `core` traversal
    native = Chem.MolToCXSmiles(parsed, params, int(Chem.CXSmilesFields.CX_BOND_ATROPISOMER))
    block = native.partition("|")[2].rpartition("|")[0]
    if not block:
        raise ValueError("RDKit did not write the assigned atropisomer stereo")
    back = parse_smiles(_append_cx(core, [block]), remove_hs=False)
    for mapped, target in targets:
        bond = back.GetBondBetweenAtoms(*mapped)
        if bond is None or bond.GetStereo() not in ATROP_STEREO or atrop_code(back, bond) != target:
            raise ValueError("could not retain atropisomer stereo with native CX wU/wD fields")
    return [block]


def write_organic(mol, *, cx):
    """Write a metal-free Mol with RDKit's own writer, or return None when only the dative writer can.

    `mol` carries the stereo to write as tags. Hydrogens drop as RDKit's `RemoveHs` drops them; mapped ones stay. A
    metal, an atropisomer axis or a zero-order or dative bond needs the dative writer.
    """
    if metal_indices(mol) or any(
        bond.GetStereo() in ATROP_STEREO or bond.GetBondType() in _NOTED_BONDS for bond in mol.GetBonds()
    ):
        return None
    keep = Chem.RemoveHsParameters()
    keep.removeMapped = False
    mol = Chem.RemoveHs(mol, keep, sanitize=False)
    return (
        Chem.MolToCXSmiles(mol, Chem.SmilesWriteParams(), Chem.CXSmilesFields.CX_ALL_BUT_COORDS)
        if cx
        else Chem.MolToSmiles(mol)
    )


# --- the arrangement layer: the canonical slot note ------------------------------------------------------


def write_smiles(mol, stereo):
    """Return the canonical dative SMILES of `mol`, tagged with its ligand `stereo`, metal notes left out.

    RDKit writes a metal-free Mol. Atropisomer stereo, which a plain SMILES cannot hold, raises.
    """
    if (written := write_organic(mol, cx=False)) is not None:
        return written
    if stereo.axes or any(bond.GetStereo() in ATROP_STEREO for bond in mol.GetBonds()):
        raise ValueError("plain dative SMILES cannot retain atropisomer stereo; use cxsmiles()")
    smi, _at, _bonds, unwritable = write_dative(mol, stereo)
    if unwritable:
        detail = sorted(tuple(sorted(pair)) for pair in unwritable)
        logger.warning("dative SMILES: RDKit cannot read E/Z at bond(s) %s; use cxsmiles() to retain it", detail)
    return smi


def write_cxsmiles(mol, species, symmetric):
    """Return the canonical CX SMILES of `species` on `mol`, its notes the least over `symmetric` metal swaps.

    A metal-free Mol keeps the caller's atom properties as CX fields (rxts carries annotations through a mapping);
    `write_name`, which names a species, leaves them out.
    """
    if (written := write_organic(mol, cx=True)) is not None:
        return written
    return _dative_cx(mol, species, symmetric).cx


def _append_cx(core, fields):
    """Append CX fields without discarding the core's native constitutional bond fields."""
    body, _, existing = core.partition("|")
    block = ",".join(field for field in (existing.removesuffix("|"), *fields) if field)
    return f"{body.rstrip()} |{block}|" if block else body.rstrip()


def _cx_bond_stereo(core, stereo, at, bonds):
    """Return CX fields for the E/Z of `stereo` that the plain SMILES `core` cannot carry."""
    fields = {"c": set(), "t": set()}
    parsed = parse_smiles(core, remove_hs=False)
    wanted = stereo.bonds
    parsed_ez = defined_stereo_label(parsed, metal_indices(parsed)).bonds if wanted else {}
    encoded = []
    for pair, expected in wanted.items():
        if not pair <= at.keys():
            raise ValueError(f"could not name double-bond stereo on bond {tuple(sorted(pair))}")
        key = frozenset(at[a] for a in pair)
        written = parsed.GetBondBetweenAtoms(*key)
        if written is not None and parsed_ez.get(key) == expected:
            continue
        bond_index = bonds.get(key)
        if bond_index is None:
            raise ValueError("could not locate the double bond in the written CXSMILES")
        for marker in ("c", "t"):
            try:
                probe = parse_smiles(_append_cx(core, [f"{marker}:{bond_index}"]), remove_hs=False)
            except ValueError:
                continue
            candidate = probe.GetBondBetweenAtoms(*key)
            if candidate is not None and defined_stereo_label(probe, metal_indices(probe)).bonds.get(key) == expected:
                fields[marker].add(bond_index)
                break
        else:
            encoded.append((pair, expected))
    out = [f"{marker}:{','.join(map(str, sorted(indices)))}" for marker, indices in fields.items() if indices]
    if encoded:
        props = []
        ordered = sorted(encoded, key=lambda row: tuple(sorted(at[idx] for idx in row[0])))
        for token, (pair, code) in enumerate(ordered):
            props.extend(f"{at[idx]}._rxEZ{token}.{code}" for idx in sorted(pair, key=at.get))
        out.append(f"atomProp:{':'.join(props)}")
    return out


class Named(NamedTuple):
    """Hold one species' CX name and its `Species` seated as that name's notes seat it."""

    name: str
    species: Species


class _Written(NamedTuple):
    """Hold one dative CXSMILES and the `CanonicalNotes` it writes, None where it arranges no centre."""

    cx: str
    notes: CanonicalNotes | None


def _dative_cx(graph, species, symmetric):
    """Return the dative CX SMILES of `species` on `graph`, with its centres' notes."""
    stereo = species.stereo
    core, at, bond_positions, _unwritable = write_dative(graph, stereo)
    fields, notes = [], None
    if bound_metals(graph) and species.centres:
        facts = LigandFacts(species.mol)
        refuse_unordered_metals(species, symmetric, facts)
        notes = canonical_notes(species, symmetric_rankings(symmetric, stereo, at), facts)
        # atomNote, since RDKit reads, writes and draws it; RDKit sorts the block by atom index
        fields.append("atomProp:" + ":".join(f"{position}.atomNote.{note}" for position, note in notes.rows))
    fields += [*_cx_bond_stereo(core, stereo, at, bond_positions), *_atrop_bond_stereo(core, stereo, at)]
    return _Written(cx=_append_cx(core, fields), notes=notes)


def _without_atom_props(mol):
    """Return a copy of `mol` without its public atom properties, which RDKit's CX writer would write as fields."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        for name in atom.GetPropNames():
            atom.ClearProp(name)
    return out


def write_name(graph, species, symmetric):
    """Return the `Named` canonical name of `species` on `graph`, seated as those notes seat it."""
    if (name := write_organic(_without_atom_props(graph), cx=True)) is not None:
        return Named(name=name, species=species)
    written = _dative_cx(graph, species, symmetric)
    species = species if written.notes is None else written_seating(species, written.notes)
    return Named(name=written.cx, species=species)


def own_rankings(classes, stereo, species):
    """Return the rankings `species`' notes compare under: its own order and each equivalent-metal swap."""
    return symmetric_rankings(classes, stereo, {atom: atom for atom in range(species.mol.GetNumAtoms())})


def specified_slots(mol, metal):
    """Return ``{donor: (slot, winding)}`` the notes give at `metal`.

    A note on the bond wins over the donor's; a bridging donor's own note lists one slot per metal, in atom order.
    """
    out, metals = {}, set(metal_indices(mol))
    for atom in mol.GetAtomWithIdx(metal).GetNeighbors():
        donor = atom.GetIdx()
        bond = mol.GetBondBetweenAtoms(metal, donor)
        if bond.HasProp(SLOT_BOND_PROP):
            values = read_slot_notes(bond.GetProp(SLOT_BOND_PROP))
        elif atom.HasProp("atomNote"):
            values = read_slot_notes(atom.GetProp("atomNote"))
            centres = sorted(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in metals)
            if values is not None and len(values) != len(centres):
                raise ValueError(
                    f"donor atom {donor} has {len(values)} slot note(s) for {len(centres)} adjacent metal(s)"
                )
            values = None if values is None or metal not in centres else [values[centres.index(metal)]]
        else:
            continue
        if values is not None:
            out[donor] = values[0]
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecifiedArrangement:
    """Hold one metal arrangement read from notes; a vacant slot names no donor."""

    geometry: str
    slots: dict  # slot -> donor atom, a haptic face named by its first ring atom read
    hand: str  # 'delta', 'lambda' or ''
    windings: dict  # slot -> haptic winding token


def metal_note(centre):
    """Return the note a metal writes: its polyhedron code, then ``-<hand>`` if it has one."""
    return record(centre.geometry).code + (f"-{centre.hand}" if centre.hand else "")


def specified_arrangement(mol, metal):
    """Return the `SpecifiedArrangement` the notes of `mol` give `metal`, or None without notes."""
    # one atomNote key holds a donor's slot `s<n>` (`slot_note`) and a metal's polyhedron code (`metal_note`)
    atom = mol.GetAtomWithIdx(metal)
    if not atom.HasProp("atomNote"):
        return None
    geometry, separator, chirality = atom.GetProp("atomNote").partition("-")
    name = resolve_geometry(geometry)
    if separator and chirality not in HAND_GLYPHS:
        raise ValueError(f"the arrangement on this string has unknown metal chirality {chirality!r}")
    slots = specified_slots(mol, metal)
    sites, windings = {}, {}
    for atom_idx, (slot, winding) in slots.items():
        if winding:
            if slot in windings and windings[slot] != winding:
                raise ValueError(f"the haptic face at slot s{slot} carries both winding signs")
            windings[slot] = winding
        sites.setdefault(slot, atom_idx)  # a haptic face writes one slot on every ring atom; the first read names it
    if name not in POLYHEDRA:
        raise ValueError(
            f"the metal atomNote names {name!r}, not a polyhedron rxembed has; use a polyhedron name or code"
        )
    return SpecifiedArrangement(geometry=name, slots=sites, hand=chirality, windings=windings)


def specified_centres(arrangements, facts):
    """Build each centre from its `SpecifiedArrangement`, checking its hand and windings.

    Every metal in `arrangements` needs one: notes on some metals but not all raise.
    """
    if missing := [metal for metal, value in arrangements.items() if value is None]:
        noted = [metal for metal in arrangements if metal not in missing]
        raise ValueError(
            f"the notes arrange metal(s) {noted} but not metal(s) {missing}; give every metal a geometry note, or none"
        )
    return tuple(_specified_centre(facts, metal, arrangement) for metal, arrangement in arrangements.items())


def _specified_centre(facts, metal, arrangement):
    """Return the `Centre` `arrangement` gives `metal`, its hand and windings checked."""
    centre = centre_from_slots(facts, metal, arrangement.geometry, arrangement.slots)
    _check_specified_hand(centre, arrangement.hand)
    return _with_specified_windings(facts, centre, arrangement.windings)


def _check_specified_hand(centre, hand):
    """Raise when a metal note's hand contradicts the hand its seating gives."""
    derived = centre.hand
    if hand and POLYHEDRA[centre.geometry].planar:
        raise ValueError(f"{centre.geometry} is planar and cannot carry metal chirality {hand!r}")
    if not hand and derived:
        raise ValueError(
            f"the noted {centre.geometry} seating is chiral ({derived}) but the metal note omits chirality"
        )
    if hand and derived and hand != derived:
        raise ValueError(f"the metal note says {hand}, but its seating is {derived}")
    if hand and not derived:
        raise ValueError(f"the metal note says {hand}, but its seating is achiral")


def _with_specified_windings(facts, centre, windings):
    """Return `centre` with each slot's face winding from the notes, checked against the face."""
    vertices, haptic, winding, donors = materialized_states(facts.graph, (centre,))[centre.atom]
    signs = dict(winding)
    ranks = donor_classes(facts.mol, donors) if windings else None
    for slot, sign in windings.items():
        face = haptic.get(vertices[slot])
        if face is None:
            raise ValueError(f"slot s{slot}{sign} gives a face winding, but that slot is not a haptic face")
        if sign not in (tokens := face_orientations(facts.mol, face, ranks)):
            allowed = f"only {'/'.join(tokens)}" if tokens else "none, being a mirror-symmetric face"
            raise ValueError(f"slot s{slot}{sign} gives a winding this face lacks; it has {allowed}")
        signs[vertices[slot]] = sign
    return state_with_winding(centre, vertices, signs)


def specified_species(mol):
    """Return the species the notes of `mol` specify, ignoring coordinates.

    Notes must arrange every bound metal or none; a free ion is no centre.
    """
    bare = Chem.Mol(mol)
    bare.RemoveAllConformers()
    graph = chemical_graph(bare)
    facts = LigandFacts(graph)
    arrangements = {metal: specified_arrangement(bare, metal) for metal in bound_metals(bare)}
    return Species(mol=graph, centres=specified_centres(arrangements if any(arrangements.values()) else {}, facts))


def bind_slot_notes(mol, written=None):
    """Copy each bridging donor's slot notes onto its metal bonds, so renumbering cannot swap them.

    The note lists slots in its metals' `written` order, else atom order.
    """
    metals = set(metal_indices(mol))
    for atom in mol.GetAtoms():
        if not atom.HasProp("atomNote") or (values := read_slot_notes(atom.GetProp("atomNote"))) is None:
            continue
        neighbours = (n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in metals)
        centres = sorted(neighbours, key=None if written is None else written.get)
        if len(values) != len(centres):
            raise ValueError(
                f"donor atom {atom.GetIdx()} has {len(values)} slot note(s) for {len(centres)} adjacent metal(s)"
            )
        if len(values) == 1:
            continue
        for metal, value in zip(centres, values, strict=True):
            mol.GetBondBetweenAtoms(atom.GetIdx(), metal).SetProp(SLOT_BOND_PROP, slot_note(*value))


# RDKit's SP/TB/OH permutation numbers (OpenSMILES 1.0, 3.8.5-3.8.7) as the vertex of each neighbour in
# storage order: the square runs 0-3, the TBP axis is 0-1, the OCT axis 4-5, equators wound as "@" lists them
NATIVE_TAGS = {
    Chem.ChiralType.CHI_SQUAREPLANAR: (
        "square_planar",
        {1: (0, 1, 2, 3), 2: (0, 2, 1, 3), 3: (0, 1, 3, 2)},  # the U, 4 and Z shapes
    ),
    Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL: (
        "trigonal_bipyramidal",
        {  # the axis, then whether the equator lists anticlockwise (@) or clockwise (@@) seen from its start
            1: (0, 2, 3, 4, 1),  # a-e @
            2: (0, 2, 4, 3, 1),  # a-e @@
            3: (0, 2, 3, 1, 4),  # a-d @
            4: (0, 2, 4, 1, 3),  # a-d @@
            5: (0, 2, 1, 3, 4),  # a-c @
            6: (0, 2, 1, 4, 3),  # a-c @@
            7: (0, 1, 2, 3, 4),  # a-b @
            8: (0, 1, 2, 4, 3),  # a-b @@
            9: (2, 0, 3, 4, 1),  # b-e @
            10: (2, 0, 3, 1, 4),  # b-d @
            11: (2, 0, 4, 3, 1),  # b-e @@
            12: (2, 0, 4, 1, 3),  # b-d @@
            13: (2, 0, 1, 3, 4),  # b-c @
            14: (2, 0, 1, 4, 3),  # b-c @@
            15: (2, 3, 0, 4, 1),  # c-e @
            16: (2, 3, 0, 1, 4),  # c-d @
            17: (2, 3, 4, 0, 1),  # d-e @
            18: (2, 4, 3, 0, 1),  # d-e @@
            19: (2, 4, 0, 1, 3),  # c-d @@
            20: (2, 4, 0, 3, 1),  # c-e @@
        },
    ),
    Chem.ChiralType.CHI_OCTAHEDRAL: (
        "octahedral",
        {  # the axis from a, then the equator's shape and winding seen from a
            1: (4, 0, 2, 1, 3, 5),  # a-f U @
            2: (4, 0, 3, 1, 2, 5),  # a-f U @@
            3: (4, 0, 2, 1, 5, 3),  # a-e U @
            4: (4, 0, 2, 3, 1, 5),  # a-f Z @
            5: (4, 0, 2, 3, 5, 1),  # a-e Z @
            6: (4, 0, 2, 5, 1, 3),  # a-d U @
            7: (4, 0, 2, 5, 3, 1),  # a-d Z @
            8: (4, 0, 1, 2, 3, 5),  # a-f 4 @@
            9: (4, 0, 1, 2, 5, 3),  # a-e 4 @@
            10: (4, 0, 1, 3, 2, 5),  # a-f 4 @
            11: (4, 0, 1, 3, 5, 2),  # a-e 4 @
            12: (4, 0, 1, 5, 2, 3),  # a-d 4 @@
            13: (4, 0, 1, 5, 3, 2),  # a-d 4 @
            14: (4, 0, 3, 2, 1, 5),  # a-f Z @@
            15: (4, 0, 3, 2, 5, 1),  # a-e Z @@
            16: (4, 0, 3, 1, 5, 2),  # a-e U @@
            17: (4, 0, 3, 5, 2, 1),  # a-d Z @@
            18: (4, 0, 3, 5, 1, 2),  # a-d U @@
            19: (4, 0, 5, 2, 1, 3),  # a-c U @
            20: (4, 0, 5, 2, 3, 1),  # a-c Z @
            21: (4, 0, 5, 1, 2, 3),  # a-c 4 @@
            22: (4, 0, 5, 1, 3, 2),  # a-c 4 @
            23: (4, 0, 5, 3, 2, 1),  # a-c Z @@
            24: (4, 0, 5, 3, 1, 2),  # a-c U @@
            25: (4, 5, 0, 2, 1, 3),  # a-b U @
            26: (4, 5, 0, 2, 3, 1),  # a-b Z @
            27: (4, 5, 0, 1, 2, 3),  # a-b 4 @@
            28: (4, 5, 0, 1, 3, 2),  # a-b 4 @
            29: (4, 5, 0, 3, 2, 1),  # a-b Z @@
            30: (4, 5, 0, 3, 1, 2),  # a-b U @@
        },
    ),
}


def read_native_tags(mol):
    """Return `mol` with each metal's RDKit SP, TB or OH tag replaced by notes for the same seating.

    A tag's number depends on neighbour order, so the notes become the one record and the tags are removed. RDKit counts
    a tagged metal's neighbours in storage order: bonds, then implicit hydrogens, then missing ligands, which are
    vacancies. A Mol with a conformer is read from 3D instead, since RDKit writes those tags from 3D. Notes that
    disagree with the tag, or a haptic face or bonded metal counted as a site, raise. Returns `mol` itself when no metal
    is tagged.
    """
    tagged = [m for m in metal_indices(mol) if mol.GetAtomWithIdx(m).GetChiralTag() in NATIVE_TAGS]
    if not tagged:
        return mol
    out = Chem.Mol(mol) if mol.GetNumConformers() else _tag_notes(mol, tagged)
    for m in tagged:
        out.GetAtomWithIdx(m).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        out.GetAtomWithIdx(m).ClearProp("_chiralPermutation")
    return out


def native_tag_vertices(atom, neighbours):
    """Return the vertex of each neighbour that `atom`'s native tag gives (`NATIVE_TAGS`).

    A permutation outside the table raises; RDKit numbers it 0 when none is written or the tag counts too many
    neighbours.
    """
    geometry, rows = NATIVE_TAGS[atom.GetChiralTag()]
    number = atom.GetUnsignedProp("_chiralPermutation") if atom.HasProp("_chiralPermutation") else 0
    cn = POLYHEDRA[geometry].cn
    if number not in rows or neighbours > cn:
        raise ValueError(
            f"{atom.GetSymbol()}{atom.GetIdx()}'s {geometry} tag on {neighbours} neighbours has no permutation "
            f"(1-{len(rows)}); use CX notes"
        )
    return rows[number]


def _tag_notes(mol, tagged):
    """Return the canonical metal graph of `mol` with notes for every tagged or noted metal."""
    if hydrogens := [m for m in tagged if mol.GetAtomWithIdx(m).GetTotalNumHs()]:
        mol = Chem.AddHs(mol, onlyOnAtoms=hydrogens)
    mol = canonical_metal_graph(mol)
    facts = LigandFacts(chemical_graph(mol))

    centres = []
    for metal in bound_metals(mol):
        told = _tag_centre(facts, mol.GetAtomWithIdx(metal)) if metal in tagged else None
        noted = (arrangement := specified_arrangement(mol, metal)) and _specified_centre(facts, metal, arrangement)
        if told and noted and centre_state(facts, told) != centre_state(facts, noted):
            raise ValueError(
                f"metal {metal} carries a native tag and notes that state different arrangements; keep one"
            )
        if told or noted:
            centres.append(told or noted)
    return write_notes(mol, Species(mol=facts.graph, centres=tuple(centres)))


def _tag_centre(facts, atom):
    """Return the `Centre` metal `atom`'s native tag gives."""
    donors = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds()]
    geometry = NATIVE_TAGS[atom.GetChiralTag()][0]
    vertices = native_tag_vertices(atom, len(donors))
    if len(haptic_sites(facts.mol, [d for d, m in facts.donor_bonds if m == atom.GetIdx()])) != len(donors):
        raise ValueError(
            f"{atom.GetSymbol()}{atom.GetIdx()}'s {geometry} tag counts a haptic face's atoms or a bonded metal "
            "as sites; use CX notes"
        )
    return centre_from_slots(facts, atom.GetIdx(), geometry, dict(zip(vertices, donors, strict=False)))


def _canonical_site_keys(facts, vertices, haptic, winding, at):
    """Return `site_keys` with each class named by its atoms' written positions `at`."""
    keys = site_keys(facts, vertices, haptic, winding)
    positions = {}
    for donor, key in zip(vertices, keys, strict=True):
        if key is not None:
            positions.setdefault(key[0], []).extend(at[atom] for atom in haptic.get(donor, (donor,)))
    return [None if key is None else (tuple(sorted(positions[key[0]])), key[1]) for key in keys]


def _ways(facts, centre, sites, at, coupled):
    """Return each way to write `centre`'s notes, least first, with the donor each `coupled` donor stands for.

    A centre without coupled donors has only its least way.
    """
    vertices, haptic, winding, _donors = sites
    keys = _canonical_site_keys(facts, vertices, haptic, winding, at)
    links = chelate_links(facts.mol, vertices, haptic)
    occupied = [v for v, donor in enumerate(vertices) if donor is not VACANT]
    atoms = {v: haptic.get(vertices[v], (vertices[v],)) for v in occupied}
    targets = sorted(occupied, key=lambda v: min(at[atom] for atom in atoms[v]))
    classes = [None if key is None else key[0] for key in keys]
    geometry = {centre.atom: metal_note(centre)}
    ways = []
    for slots in least_frames(vertex_dirs(centre.geometry), keys, links):
        sources = sorted(occupied, key=lambda v, slots=slots: (slots[v], keys[v][1]))
        for assigned in equivalent_site_assignments(classes, links, targets=targets, sources=sources):
            notes = {atom: slot_note(slots[s], keys[s][1]) for t, s in assigned.items() for atom in atoms[t]}
            ways.append(
                (geometry | notes, {atoms[t][0]: atoms[s][0] for t, s in assigned.items() if atoms[t][0] in coupled})
            )
            if coupled.isdisjoint(atom for face in atoms.values() for atom in face):
                return ways
    return ways


def _least_notes(ways, at, distance, roles):
    """Return the least notes of each centre in turn that keep earlier centres' bridges in place, or None.

    `roles` maps each coupled donor an earlier centre seated to the donor it stands for there. A later way must keep
    every graph distance between coupled donors, so a bridge takes the same place among its equivalents at both centres.
    """
    if not ways:
        return []
    least = {}  # per set of roles left for the later centres, the least notes that set them
    for notes, played in ways[0]:
        if any(distance[x][y] != distance[roles[x]][played[y]] for x in roles for y in played):
            continue
        ranked = sorted((at[atom], note) for atom, note in notes.items())
        if (group := frozenset(played.items())) not in least or ranked < least[group][0]:
            least[group] = (ranked, notes, played)
    options = [
        [notes, *tail]
        for _, notes, played in least.values()
        if (tail := _least_notes(ways[1:], at, distance, roles | played)) is not None
    ]
    return min(
        options,
        key=lambda option: [sorted((at[atom], note) for atom, note in one.items()) for one in option],
        default=None,
    )


def specified_windings_only(observed, specified):
    """Return `observed` with the winding cleared on every face `specified` leaves unwound."""
    winds = {
        (centre.atom, frozenset(site.atoms)): site.winding
        for centre in specified.centres
        for site in centre.vertices
        if site is not VACANT
    }

    def specified_winding(atom, site):
        if site is VACANT:
            return VACANT
        return replace(site, winding=site.winding if winds.get((atom, frozenset(site.atoms))) else "")

    return replace(
        observed,
        centres=tuple(
            centre._replace(vertices=tuple(specified_winding(centre.atom, site) for site in centre.vertices))
            for centre in observed.centres
        ),
    )


def arrangement_notes(species, facts, at):
    """Return the canonical notes of each centre and its donors, by metal, keyed by atom index.

    Equivalent donors take their least slots. A ligand bridging exactly two centres takes the same place among its
    equivalents at both, so syn and anti bridges stay distinct; bridges among three or more centres are seated centre by
    centre and their relation is not written.
    """
    owner = facts.fragments
    metals = defaultdict(set)  # the centres each ligand binds, by its `frag_map` id
    for donor, metal in facts.donor_bonds:
        metals[owner[donor]].add(metal)
    bridging = {donor: owner[donor] for donor, _metal in facts.donor_bonds if len(metals[owner[donor]]) > 1}
    groups = []  # the centres bridging ligands join
    for found in (found for found in metals.values() if len(found) > 1):
        joined = set(found).union(*(group for group in groups if group & found))
        groups = [group for group in groups if not group & found] + [joined]
    pairs = [group for group in groups if len(group) == 2]  # noqa: PLR2004  a bridged pair
    coupled = {donor for donor, ligand in bridging.items() if any(metals[ligand] <= pair for pair in pairs)}
    distance = facts.topology
    notes = {}
    for group in [*pairs, *({centre.atom} for centre in species.centres if not any(centre.atom in p for p in pairs))]:
        centres = sorted((centre for centre in species.centres if centre.atom in group), key=lambda c: at[c.atom])
        ways = [_ways(facts, centre, species.sites[centre.atom], at, coupled) for centre in centres]
        notes |= dict(zip((c.atom for c in centres), _least_notes(ways, at, distance, {}), strict=True))
    return notes


def refuse_unordered_metals(species, symmetric, facts):
    """Refuse three or more equivalent metals with different arrangements: a pair swap orders two, nothing orders three.

    `symmetric` is `symmetric_metals` of the graph and `facts` its `LigandFacts`.
    """
    by_atom = {centre.atom: centre for centre in species.centres}
    for group in symmetric:
        if len(group) > 2 and len({centre_state(facts, by_atom[m]) for m in group if m in by_atom}) > 1:  # noqa: PLR2004
            raise ValueError(
                "three or more symmetry-equivalent metal centres carry different arrangements; take each complex alone"
            )


def symmetric_rankings(classes, stereo, at):
    """Return ranking `at` and its image under each equivalent-metal pair swap that keeps `stereo`.

    Such a swap is a graph symmetry, so a species and its image are one species. A class of three or more metals keeps
    its ranking.
    """
    exchanges = [list(group.values())[1] for group in classes if len(group) == 2]  # noqa: PLR2004  a pair
    if not exchanges:
        return [at]
    orders = [list(range(len(exchanges[0])))]
    for exchange in exchanges:
        orders += [[exchange[atom] for atom in order] for order in orders]
    elements = {(e.kind, frozenset(e.atoms), e.code) for e in stereo.elements}
    return [
        {atom: at[order[atom]] for atom in at}
        for order in orders
        if {(e.kind, frozenset(order[a] for a in e.atoms), e.code) for e in stereo.elements} == elements
    ]


class CanonicalNotes(NamedTuple):
    """Hold the canonical notes of one species: the rows a CX string writes, and their ranking."""

    rows: tuple  # sorted (position, note) rows; equal rows mean the same species
    at: dict  # the ranking that writes them: the position each atom's notes are written at


def canonical_notes(species, rankings, facts):
    """Return the least `CanonicalNotes` of `species` over `rankings`.

    Two species of one graph and ligand stereo are the same exactly when their rows agree. Rows compare as the text they
    write, so the CX string is the least one ("OCT-delta" sorts before "OCT").
    """
    written = []
    for at in rankings:
        centre_notes = arrangement_notes(species, facts, at)
        notes = joined_notes(
            centre_notes, {metal: set(by_atom) - {metal} for metal, by_atom in centre_notes.items()}, at
        )
        written.append(CanonicalNotes(rows=tuple(sorted((at[atom], note) for atom, note in notes.items())), at=at))
    return min(written, key=lambda one: ":".join(f"{position}.{note}" for position, note in one.rows))


def written_seating(species, notes):
    """Return `species` seated as its notes `notes` seat it, centres in written order.

    The notes are read back onto the atoms of `species` itself, so names read from the result match the CX string.
    """
    graph = with_conformers(species.mol, species.mol, ())
    noted = Chem.Mol(graph)
    atoms = {position: atom for atom, position in notes.at.items()}
    for position, note in notes.rows:
        noted.GetAtomWithIdx(atoms[position]).SetProp("atomNote", note)
    bind_slot_notes(noted, notes.at)
    arrangements = {centre.atom: specified_arrangement(noted, centre.atom) for centre in species.centres}
    centres = sorted(specified_centres(arrangements, LigandFacts(graph)), key=lambda centre: notes.at[centre.atom])
    return replace(species, centres=tuple(centres))


def distinct(built):
    """Return each species of `built` once, in order; they share one graph and ligand stereo."""
    if not built:
        return []
    graph = graph_with_stereo(built[0])
    stereo = defined_stereo_label(graph, metal_indices(graph))
    rankings = own_rankings(symmetric_metals(graph), stereo, built[0])
    facts = LigandFacts(built[0].mol)
    out = {}
    for species in built:
        if facts.graph is not species.mol:
            facts = LigandFacts(species.mol)
        out.setdefault(canonical_notes(species, rankings, facts).rows, species)
    return list(out.values())


def joined_notes(centre_notes, bound, at):
    """Return one note per atom: each metal's polyhedron and hand, each donor's slot at every metal it binds."""
    notes = {m: notes_of_m[m] for m, notes_of_m in centre_notes.items()}
    for donor in set().union(*(bound[m] for m in centre_notes)):
        owners = sorted((m for m in centre_notes if donor in bound[m]), key=at.get)
        notes[donor] = ";".join(centre_notes[m][donor] for m in owners)
    return notes


def write_notes(mol, species):
    """Write each centre of `species` onto `mol` as notes numbered by its own vertices, and return `mol`.

    Reading the notes back seats every donor where the record has it, so the species compiles to the same constraints.
    """
    centre_notes = {}
    for centre in species.centres:
        vertices, haptic, winding, _donors = species.sites[centre.atom]
        notes = {centre.atom: metal_note(centre)}
        for slot, donor in enumerate(vertices):
            if donor is not VACANT:
                notes.update(dict.fromkeys(haptic.get(donor, (donor,)), slot_note(slot, winding.get(donor, ""))))
        centre_notes[centre.atom] = notes
    bound = {m: set(notes) - {m} for m, notes in centre_notes.items()}
    for atom, note in joined_notes(centre_notes, bound, {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms()}).items():
        mol.GetAtomWithIdx(atom).SetProp("atomNote", note)
    for bond in mol.GetBonds():
        bond.ClearProp(SLOT_BOND_PROP)
    for donor in set().union(*bound.values()):  # a bridge keeps each metal's slot on its bond; no other note is read
        owners = [metal for metal in bound if donor in bound[metal]]
        for metal in owners if len(owners) > 1 else ():
            mol.GetBondBetweenAtoms(donor, metal).SetProp(SLOT_BOND_PROP, centre_notes[metal][donor])
    return mol


def mol_from_species(species):
    """Return the species Mol: the graph with stereo tags, conformers and notes.

    Notes are written in the record's own vertex frame, so reading the Mol back gives the same record.
    """
    return write_notes(graph_with_stereo(species), species)
