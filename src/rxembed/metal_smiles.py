"""Read and write canonical ionic-dative metal SMILES.

Dative SMILES carries constitution, point stereo, and E/Z where its graph permits. Zero-order contacts retain
their native CX Z: field. Further CX fields add atropisomer stereo, canonical polyhedron slots, and haptic
winding. Notes address the atom order just written, so parsing and writing share this module; the slot grammar
itself belongs to `metal_polyhedron`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rdkit import Chem, rdBase

from .metal_core import (
    COORDINATION_METALS,
    HAND_TAG,
    VACANT,
    canonical_metal_graph,
    connect_metal,
    donor_chirality_sign,
    materialized_state,
    metal_indices,
)
from .metal_enumeration import enumerate_isomers, from_stated_arrangements, stated_arrangement
from .metal_isomer import centre_states, from_geometry
from .metal_perceive import SHAPE_REQUEST_PROP, decode_shape_request
from .metal_polyhedron import SLOT_BOND_PROP, canonical_slots, read_slot_notes, record, slot_note, vertex_dirs
from .metal_stereo import chelate_links, equivalent_site_assignments, remove_routine_hydrogens, site_classes
from .stereo import (
    ATROP_STEREO,
    atrop_code,
    axis_stereo,
    bond_stereo,
    bond_stereo_code,
    coordination_locked_double_bonds,
    defined_stereo_label,
    encoded_bond_stereo,
    point_stereo,
    stereo_from_3d,
    without_bond_stereo,
)
from .utils import hydrogen_neighbor_order, mirror_tag, remove_bond

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__: the name `set_verbose` configures

_E_BOND = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
_Z_BOND = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
_POINT_TAGS = {"CW": Chem.ChiralType.CHI_TETRAHEDRAL_CW, "CCW": Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
_TAG_CODES = {tag: code for code, tag in _POINT_TAGS.items()}
_CX_BOND_FIELD = re.compile(r"(?:^|,)([ct]):((?:\d+(?:,\d+)*)?)(?=,|$)")
_CX_BOND_PREFIX = re.compile(r"(?:^|,)[ct]:")


def _repair_haptic_bond_stereo(mol):
    """Replace shared-metal E/Z references with ligand-side references without changing CIP."""
    for bond in mol.GetBonds():
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in _E_BOND | _Z_BOND or len(refs) != 2:  # noqa: PLR2004 - two stereo refs
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
    """Reject invalid standard CX c:/t: fields."""
    block = smi.partition("|")[2].rpartition("|")[0]
    if not block:
        return
    matches = _CX_BOND_FIELD.findall(block)
    if len(_CX_BOND_PREFIX.findall(block)) != len(matches):
        raise ValueError("invalid CX c:/t: double-bond stereo field")
    stated = {}
    for marker, values in matches:
        if not values:
            raise ValueError("invalid empty CX c:/t: double-bond stereo field")
        for value in values.split(","):
            idx = int(value)
            if idx in stated:
                raise ValueError(f"contradictory or duplicate CX c:/t: stereo for bond {idx}")
            stated[idx] = marker
    params = Chem.SmilesWriteParams()
    params.canonical = False  # preserve the parsed traversal: CX bond positions address that written order
    params.cleanStereo = False
    Chem.MolToSmiles(mol, params)
    order = list(mol.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
    for idx, marker in stated.items():
        if idx >= len(order):
            raise ValueError(f"CX {marker}: field names missing bond {idx}")
        bond = mol.GetBondWithIdx(order[idx])
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in _E_BOND | _Z_BOND or len(refs) != 2 or len(set(refs)) != 2:  # noqa: PLR2004
            raise ValueError(f"CX {marker}:{idx} did not define valid double-bond stereo")


def _bind_slot_notes(mol):
    """Attach each parsed donor slot to its metal-donor bond so renumbering cannot swap a bridge list."""
    metals = set(metal_indices(mol))
    for atom in mol.GetAtoms():
        if not atom.HasProp("atomNote") or (values := read_slot_notes(atom.GetProp("atomNote"))) is None:
            continue
        centres = sorted(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in metals)
        if len(values) != len(centres):
            raise ValueError(
                f"donor atom {atom.GetIdx()} has {len(values)} slot note(s) for {len(centres)} adjacent metal(s)"
            )
        if len(values) == 1:
            continue
        for metal, value in zip(centres, values, strict=True):
            mol.GetBondBetweenAtoms(atom.GetIdx(), metal).SetProp(SLOT_BOND_PROP, slot_note(*value))


def parse_smiles(smi, *, remove_hs=True):
    """Parse one metal SMILES and preserve coordinated ligand stereo.

    Sanitising before removing hydrogens retains tags such as ``[N@H]`` beside a dative bond. CX inputs keep
    explicit coordination hydrogens because ``atomProp`` indices address the written atom order. Invalid
    input raises instead of returning ``None``.
    """
    params = Chem.SmilesParserParams()
    params.sanitize = False  # RDKit's integrated cleanup erases `[N@H]` when its fourth neighbour is dative.
    params.removeHs = False
    mol = Chem.MolFromSmiles(smi, params)
    if mol is not None:
        raw = Chem.Mol(mol)
        raw_tags = {
            atom.GetIdx(): atom.GetChiralTag()
            for atom in raw.GetAtoms()
            if atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
        }
        try:
            Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            try:
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
    encoded_bond_stereo(mol)
    _bind_slot_notes(mol)
    if remove_hs and "atomProp" not in smi:
        mol = Chem.RemoveHs(mol)
    return mol


# --- the constitution layer: dative M-L bonds, and the atom order the string was written in ---------------

_H_VALENCE = 1  # all SMILES will spend on a hydrogen; a second connection has to be dative
_METAL_STEREO_TAGS = frozenset(  # the non-tetrahedral classes perception leaves on a metal
    {Chem.ChiralType.CHI_SQUAREPLANAR, Chem.ChiralType.CHI_OCTAHEDRAL, Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL}
)


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
        bond.SetStereo(Chem.BondStereo.STEREOZ if bond.GetStereo() in _E_BOND else Chem.BondStereo.STEREOE)


def _writes(graph, params, rebase):
    """Yield each trial SMILES with the written copy that holds its output order.

    A canonical write tries RDKit's default writer before `params`.
    """
    for writer in (None, params) if rebase else (params,):
        written = Chem.Mol(graph)  # MolToSmiles may rebase tags, so never write the next trial in place
        smi = Chem.MolToSmiles(written) if writer is None else Chem.MolToSmiles(written, writer)
        # Native Z: indices address this arrow writer's bond order. MolToCXSmiles uses single bonds for
        # dative edges, so its core cannot replace ours. This is RDKit's get_zerobonds_block convention.
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
        defined = point_stereo(defined_stereo_label(mol, metal_indices(mol))) if wanted else {}
        tags = {idx: mol.GetAtomWithIdx(idx).GetChiralTag() for idx, code in wanted.items() if defined.get(idx) == code}
        native = {
            idx: mol.GetAtomWithIdx(idx).GetChiralTag()
            for idx in wanted
            if mol.GetAtomWithIdx(idx).GetChiralTag() in _POINT_TAGS.values()
        }
        return cls(mol, wanted, wanted_bonds, tags, native)

    def _native_tag_matches(self, graph, points, index, target):
        """Return whether `target` keeps the explicit tag of a centre RDKit cannot CIP-rank.

        RDKit can preserve an explicit tag on hypervalent centres while it cannot assign a CIP label or use
        that centre in a chiral substructure match (for example [S@+2]). The tag is still a meaningful parser
        round-trip contract; ordinary centres take the stronger CIP and match path.
        """
        return (
            index in self.native
            and points.get(target) is None
            and graph.GetAtomWithIdx(target).GetChiralTag() == self.native[index]
        )

    def _same_hand(self, graph, mapped, idx, tag):
        """Return whether `graph` gives centre `idx` the hand `tag` under the atom map `mapped`.

        Compare the local permutation with all carriers present. Simultaneously correcting CIP labels can
        oscillate when fixing one donor changes a neighbouring pseudoasymmetric descriptor.
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
        label = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = point_stereo(label), bond_stereo(label)
        wrong_points = [
            idx
            for idx, code in self.wanted.items()
            if idx not in self.tags
            and code in {"R", "S", "r", "s"}
            and points.get(mapped[idx]) != code
            and not self._native_tag_matches(graph, points, idx, mapped[idx])
        ]
        stated = {idx: _POINT_TAGS[code] for idx, code in self.wanted.items() if code in _POINT_TAGS}
        wrong_points += [
            idx
            for idx, tag in (stated | self.native | self.tags).items()
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
        label = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = point_stereo(label), bond_stereo(label)
        if any(code in {"R", "S", "r", "s"} and points.get(match[idx]) != code for idx, code in self.wanted.items()):
            return None
        if any(bonds.get(frozenset(match[idx] for idx in pair)) != code for pair, code in self.wanted_bonds.items()):
            return None
        order = [None] * len(match)
        for old, new in enumerate(match):
            order[new] = old
        return order


def _rebase_native_stereo(mol, wanted, wanted_bonds, params):
    """Write through RDKit's coordinate-bond canonical order, so a tag beside an outgoing dative bond keeps its hand.

    RDKit's arrow writer excludes outgoing dative bonds when deciding whether to rebase a point tag. Its
    coordinate-bond CX writer includes every carrier. Parse that canonical atom and bond order once, then emit
    arrows in a fixed traversal; repeatedly canonicalizing corrected tags cycles.
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
        if code not in _POINT_TAGS:
            continue
        tag = back.GetAtomWithIdx(positions[idx]).GetChiralTag()
        if tag not in _POINT_TAGS.values():
            raise ValueError(f"native canonicalization lost point stereo at atom {idx}")
        mapped_wanted[positions[idx]] = _TAG_CODES[tag]
    mapped_bonds = {frozenset(positions[idx] for idx in pair): code for pair, code in wanted_bonds.items()}
    smi, rebased_order, rebased_bonds = _write_native_stereo(back, mapped_wanted, mapped_bonds, rebase=False)
    return smi, [order[idx] for idx in rebased_order], rebased_bonds


def _write_native_stereo(mol, wanted, wanted_bonds, *, rebase=True):
    """Write SMILES, restoring only point and double-bond stereo already proved on the ligand graph.

    Returns ``(smiles, written atom order, bond positions)``. RDKit's writer can rebase a tag, so each trial is
    parsed back and its wrong centres and bonds are flipped for the next one. ``rebase=False`` writes the
    atom order as given, for the second pass of `_rebase_native_stereo`.
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
    """Return whether plain RDKit SMILES has a directional bond at both ends of this double bond."""
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


def _round_trip_graph_error(expected, smi, written):
    """Return the first constitutional change made by parsing a written SMILES.

    Parses with every atom explicit, then applies RDKit's own default `RemoveHs` and reads which written
    atoms survive: RDKit decides which hydrogens a round trip keeps, not a model of that rule (a modelled
    rule mismatched RDKit's own choice for a hydroxycarbene O-H that fixes the bond's E/Z).
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    parsed = Chem.MolFromSmiles(smi, params)
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
    """Return `mol` with each multi-leg hydrogen donating through every leg but its first.

    The first leg by `hydrogen_neighbor_order` stays covalent: the nonmetal leg, or for H2 or a nonmetal relay
    the one distance and canonical graph rank choose.
    """
    rw = Chem.RWMol(mol)
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    for atom in mol.GetAtoms():
        h = atom.GetIdx()
        nbrs = [bond.GetOtherAtomIdx(h) for bond in atom.GetBonds() if bond.GetValenceContrib(atom) > 0]
        if atom.GetAtomicNum() != 1 or len(nbrs) <= _H_VALENCE:
            # already normalised: keep the one covalent bond the string explicitly states
            continue
        ordered = hydrogen_neighbor_order(
            mol,
            h,
            metals=COORDINATION_METALS,
            positions=pos,
            ranks=ranks,
        )
        nbrs = [neighbor for neighbor in ordered if neighbor in nbrs]
        for n in nbrs[1:]:
            remove_bond(rw, h, n)  # re-seats the bond LAST at `n`, so the partner's tag moves basis with it
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
    """Clear in place every point tag, E/Z and metal tag that `wanted` and `wanted_bonds` do not name."""
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in _POINT_TAGS.values() and atom.GetIdx() not in wanted:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    _clear_bond_stereo(
        mol,
        [
            pair
            for bond in mol.GetBonds()
            if bond.GetStereo() in _E_BOND | _Z_BOND
            and (pair := frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) not in wanted_bonds
        ],
    )
    # The metal's own tag is a neighbour-order permutation that no geometry settles (Fe(CO)5 writes @TB20,
    # @TB14 or @TB13). Dropping it takes the corpus from 26 to 40 strings stable under reordering.
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in _METAL_STEREO_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_chiralPermutation"):
                atom.ClearProp("_chiralPermutation")


def _kept_hydrogens(mol, wanted_bonds):
    """Return the hydrogens the SMILES keeps explicit: metal-bound ones, and each E/Z end's only reference.

    A hydrogen bound to the metal remains explicit because it can own a slot.
    """
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
    back = Chem.MolFromSmiles(smi)
    actual = None if back is None else _atoms_with_h(back)
    graph_error = _round_trip_graph_error(mol, smi, written)
    if actual == expected and not graph_error:
        return
    got = "does not parse" if actual is None else f"parses back as {actual} atoms"
    if graph_error and actual == expected:
        got = graph_error
    raise ValueError(
        f"could not write a graph-preserving round-tripping SMILES for this complex "
        f"({expected} atoms including H): the "
        f"result {got}. The perceived graph is likely one SMILES cannot express (a hypervalent or "
        f"partial-bond centre); work from the Mol itself."
    )


def _write_dative(mol, stereo_label, *, stated=False):
    """Return canonical dative SMILES plus atom-position and bond-position maps.

    The atom-position map is what an `atomProp` block indexes; only atoms that remain explicit have one.
    Every accepted M-donor bond is normalized to dative first, including every extra leg of a hydrogen.
    ``stated=True`` marks `mol`'s point tags as an isomer's own hands, which win over its carried coordinates.

    Raises instead of returning a SMILES that will not round-trip: an unreadable string is worse than none.
    """
    source = Chem.Mol(mol)
    source.UpdatePropertyCache(strict=False)
    expected = _atoms_with_h(source)
    # A raw CW/CCW code (a centre CIP cannot rank) is a tag in `source`'s bond order. Set it there, so each edit
    # below carries it into the written order as it carries any tag: removing a hydrogen or re-seating a bond
    # can mirror a tag, and the bare code cannot follow that.
    raw = {idx: _POINT_TAGS[code] for idx, code in point_stereo(stereo_label).items() if code in _POINT_TAGS}
    for idx, tag in raw.items():
        source.GetAtomWithIdx(idx).SetChiralTag(tag)
    # An isomer can state a hand against its carried coordinates: an enumerated locked donor or stereo='invert'.
    # Parity reads the tag in the all-bonds basis rxembed stores. A tag left in RDKit's 3D basis, which omits a
    # dative bond at an odd slot (`metal_core._retag`), would read as stated against the coordinates.
    against = (
        {
            atom.GetIdx()
            for atom in source.GetAtoms()
            if atom.GetDegree() == 4  # noqa: PLR2004  four explicit carriers fix the parity without an implicit H
            and atom.GetChiralTag() in HAND_TAG.values()
            and HAND_TAG.get(donor_chirality_sign(source, -1, atom.GetIdx())) == mirror_tag(atom.GetChiralTag())
        }
        if stated and source.GetNumConformers()
        else set()
    )
    stereo_label = without_bond_stereo(
        stereo_label,
        coordination_locked_double_bonds(source, metal_indices(source)),
    )
    out = canonical_metal_graph(_dative_hydrogens(source))
    out.UpdatePropertyCache(strict=False)
    wanted = point_stereo(stereo_label)
    wanted_bonds = bond_stereo(stereo_label)
    out = _ez_end_hydrogens(out, wanted_bonds)
    if not stereo_label:
        # The caller already resolved stereo. Do not infer and then discard it again.
        Chem.RemoveStereochemistry(out)
        for bond in out.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)  # RDKit leaves single-bond atrop tags behind.
    elif out.GetNumConformers():
        # Normalizing metal bonds changes neighbour order; rebase retained stereo from coordinates.
        stated_raw = {idx: out.GetAtomWithIdx(idx).GetChiralTag() for idx in raw}  # the label wins, as R/S does
        stereo_from_3d(out, metal_indices(out), apply=True)
        for idx in against:
            atom = out.GetAtomWithIdx(idx)
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
        for idx, tag in stated_raw.items():
            out.GetAtomWithIdx(idx).SetChiralTag(tag)
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


def _atrop_bond_stereo(core, stereo_label, at):
    """Return the native CX wU/wD field retaining every assigned atropisomer axis."""
    axes = axis_stereo(stereo_label)
    if not axes:
        return []
    plain = parse_smiles(core, remove_hs=False)
    targets = []
    for (i, j), target in axes.items():
        if i not in at or j not in at:
            raise ValueError(f"could not name atropisomer stereo on bond {i}-{j}")
        mapped = (at[i], at[j])
        if plain.GetBondBetweenAtoms(*mapped) is None:
            raise ValueError(f"could not locate atropisomer bond {i}-{j} in the written CXSMILES")
        for tag in (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW):
            candidate = Chem.Mol(plain)
            bond = candidate.GetBondBetweenAtoms(*mapped)
            bond.SetStereo(tag)
            Chem.CleanupAtropisomers(candidate)
            if bond.GetStereo() in ATROP_STEREO and atrop_code(candidate, bond) == target:
                plain = candidate
                targets.append((mapped, target))
                break
        else:
            raise ValueError(f"could not retain atropisomer stereo on bond {i}-{j}")
    params = Chem.SmilesWriteParams()
    params.canonical = False  # CX bond indices must address the already-canonical `core` traversal
    native = Chem.MolToCXSmiles(plain, params, int(Chem.CXSmilesFields.CX_BOND_ATROPISOMER))
    block = native.partition("|")[2].rpartition("|")[0]
    if not block:
        raise ValueError("RDKit did not write the assigned atropisomer stereo")
    back = parse_smiles(_append_cx(core, [block]), remove_hs=False)
    for mapped, target in targets:
        bond = back.GetBondBetweenAtoms(*mapped)
        if bond is None or bond.GetStereo() not in ATROP_STEREO or atrop_code(back, bond) != target:
            raise ValueError("could not retain atropisomer stereo with native CX wU/wD fields")
    return [block]


def dative_smiles(mol, *, cx=False):
    """Return canonical dative SMILES without the metal arrangement, so cis and trans give one string.

    Use `cxsmiles` when the arrangement matters; this string alone is not a species key.
    """
    stereo_label = (
        stereo_from_3d(mol, metal_indices(mol))
        if mol.GetNumConformers()
        else defined_stereo_label(mol, metal_indices(mol))
    )
    if not cx and (axis_stereo(stereo_label) or any(bond.GetStereo() in ATROP_STEREO for bond in mol.GetBonds())):
        raise ValueError("plain dative SMILES cannot retain atropisomer stereo; use cxsmiles()")
    smi, at, bonds, unwritable = _write_dative(mol, stereo_label)
    if cx:
        fields = [*_cx_bond_stereo(smi, stereo_label, at, bonds), *_atrop_bond_stereo(smi, stereo_label, at)]
        return _append_cx(smi, fields)
    if unwritable:
        detail = sorted(tuple(sorted(pair)) for pair in unwritable)
        logger.warning("dative SMILES: RDKit cannot read E/Z at bond(s) %s; use cxsmiles() to retain it", detail)
    return smi


# --- the arrangement layer: the canonical slot note ------------------------------------------------------


def _site_keys(iso, vertices, haptic, winding):
    """Return one order-invariant key per vertex (``None`` at a vacancy), for deciding which sites tie.

    A sigma donor's key is its symmetry class; a haptic face's key adds the symmetry class of its whole atom
    set and its winding, so identical Cp rings tie and constitutionally different faces do not. Never an atom
    index or a string position, so every writer folding on these agrees.

    `Isomer` owns the winding: `from_geometry` has already measured and stored it, while a vertex-only isomer
    leaves it empty.
    """
    classes = site_classes(iso.graph, vertices, haptic, iso.roles)
    keys = []
    for d in vertices:
        if d == VACANT:
            keys.append(None)
        elif d in haptic:
            wind = winding.get(d, "")
            keys.append((classes[d], wind))
        else:
            keys.append((classes[d], ""))
    return keys


def _canonical_site_keys(iso, vertices, haptic, winding, at):
    """Relabel site classes by their canonical dative-SMILES atom positions."""
    keys = _site_keys(iso, vertices, haptic, winding)
    positions = {}
    for donor, key in zip(vertices, keys, strict=True):
        if key is not None:
            positions.setdefault(key[0], []).extend(at[atom] for atom in haptic.get(donor, (donor,)))
    return [None if key is None else (tuple(sorted(positions[key[0]])), key[1]) for key in keys]


def _slot_notes(vertices, haptic, keys, slots, at, links):
    """Return the least donor-slot assignment allowed by the labelled donor graph."""
    occupied = [v for v, donor in enumerate(vertices) if donor != VACANT]
    atoms = {v: tuple(haptic.get(vertices[v], (vertices[v],))) for v in occupied}
    targets = sorted(occupied, key=lambda v: min(at[atom] for atom in atoms[v]))
    assigned = next(
        equivalent_site_assignments(
            [None if key is None else key[0] for key in keys],
            links,
            targets=targets,
            sources=sorted(occupied, key=lambda v: (slots[v], keys[v][1])),
        ),
        None,
    )
    if assigned is None:  # identity is always a colour- and link-preserving assignment
        raise ValueError("could not canonicalize the donor-slot assignment")
    return {
        atom: slot_note(slots[assigned[target]], keys[assigned[target]][1])
        for target in targets
        for atom in atoms[target]
    }


def _append_cx(core, fields):
    """Append CX fields without discarding the core's native constitutional bond fields."""
    body, _, existing = core.partition("|")
    block = ",".join(field for field in (existing.removesuffix("|"), *fields) if field)
    return f"{body.rstrip()} |{block}|" if block else body.rstrip()


def _realised_bond_stereo(graph, pair):
    """Return the E/Z code RDKit reads on `graph` for the double bond between the atoms in `pair`."""
    return bond_stereo(defined_stereo_label(graph, metal_indices(graph))).get(pair)


def _cx_bond_stereo(core, stereo_label, at, bonds):
    """Return lossless CX fields for E/Z that plain SMILES cannot retain."""
    fields = {"c": set(), "t": set()}
    plain = parse_smiles(core, remove_hs=False)
    encoded = []
    for pair, expected in bond_stereo(stereo_label).items():
        if not pair <= at.keys():
            raise ValueError(f"could not name double-bond stereo on bond {tuple(sorted(pair))}")
        key = frozenset(at[a] for a in pair)
        written = plain.GetBondBetweenAtoms(*key)
        if written is not None and _realised_bond_stereo(plain, key) == expected:
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
            if candidate is not None and _realised_bond_stereo(probe, key) == expected:
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


def _arrangement_notes(iso, state, at):
    """Return the metal and donor notes for one centre, keyed by source atom index."""
    dirs = vertex_dirs(state.geometry)
    if dirs is None:
        raise ValueError(
            f"no polyhedron template for {state.geometry!r}, so there is no slot scheme to write and the "
            f"arrangement would be lost silently; add a POLYHEDRA row for this coordination number"
        )
    work = iso.graph
    vertices, haptic, winding, _donors = materialized_state(iso, state)
    keys = _canonical_site_keys(iso, vertices, haptic, winding, at)
    links = chelate_links(work, vertices, haptic)
    slots = canonical_slots(dirs, keys, links)
    geom = record(state.geometry).code + (f"-{state.hand}" if state.hand else "")
    return {state.atom: geom} | _slot_notes(vertices, haptic, keys, slots, at, links)


def cxsmiles(source):
    """Write canonical dative CXSMILES carrying the selected metal state.

    Metal notes store geometry and hand; donor notes store canonical slots and haptic winding. E/Z uses
    native bond directions and CX ``c:``/``t:`` fields where RDKit supports them, and paired ``_rxEZ`` atom
    properties otherwise. A bridging donor stores one slot per adjacent metal.

    `source` is an `Isomer` or a conformer-bearing `Mol`. Fails rather than lose identity: a missing
    polyhedron template, contradictory stereo, or symmetry-equivalent metals carrying different states.
    """
    iso = None if isinstance(source, Chem.Mol) else source
    # A Mol is already its own constitution; an Isomer is reconnected with its authoritative donor bonds.
    complexed = source if iso is None else connect_metal(iso.restore(Chem.Mol(iso.graph)), iso.donor_bonds)
    # Donor bookkeeping below must see the same canonical graph the centre notes were derived from.
    complexed = canonical_metal_graph(complexed)
    metals = metal_indices(complexed)
    if not metals:
        raise ValueError("no transition metal found")
    if iso is not None:
        ligand_stereo = iso.stereo_label
    elif complexed.GetNumConformers():
        ligand_stereo = stereo_from_3d(complexed, metals)
    else:
        ligand_stereo = defined_stereo_label(complexed, metals)
    locked = coordination_locked_double_bonds(complexed, metals)
    ligand_stereo = without_bond_stereo(ligand_stereo, locked)
    bound = {
        m: {n.GetIdx() for n in complexed.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals} for m in metals
    }
    centres = tuple(m for m in metals if bound[m])
    if not centres:
        core, at, bond_positions, _unwritable = _write_dative(complexed, ligand_stereo, stated=iso is not None)
        fields = [
            *_cx_bond_stereo(core, ligand_stereo, at, bond_positions),
            *_atrop_bond_stereo(core, ligand_stereo, at),
        ]
        return _append_cx(core, fields)
    if iso is not None:
        records = [(iso, state) for state in centre_states(iso)]
    elif complexed.GetNumConformers():
        # An accepted conformer's requested frame (SHAPE_REQUEST_PROP) wins over the argmin, which can read the
        # other shape of a near-tie; the acceptance gate already verified it within _FIT_MARGIN. A raw geometry
        # has no such record and keeps the argmin reading.
        conf = source.GetConformer() if source.GetNumConformers() else None
        requested = (
            decode_shape_request(conf.GetProp(SHAPE_REQUEST_PROP))
            if conf is not None and conf.HasProp(SHAPE_REQUEST_PROP)
            else {}
        )
        records = []
        for m in centres:
            built = (
                enumerate_isomers(source, geometry=requested[m], center=m, observed_only=True)
                if m in requested
                else (from_geometry(source, center=m),)
            )
            records.append((built[0], built[0].centres[0]))
    else:
        if not all(stated_arrangement(complexed, center=m) is not None for m in centres):
            raise ValueError("cxsmiles needs an Isomer, coordinates, or one stated arrangement per metal")
        stated = from_stated_arrangements(complexed, centres[0], "model")
        records = [(stated, state) for state in centre_states(stated)]
    if {state.atom for _record, state in records} != set(centres):
        raise ValueError("the isomer does not carry one state per metal")
    core, at, bond_positions, _unwritable = _write_dative(complexed, ligand_stereo, stated=iso is not None)
    centre_notes = {}
    for record_iso, state in records:
        centre_notes[state.atom] = _arrangement_notes(record_iso, state, at)
    ranks = list(Chem.CanonicalRankAtoms(complexed, breakTies=False))
    by_rank = {}
    for m in centres:
        signature = (centre_notes[m][m], tuple(sorted((ranks[d], centre_notes[m][d]) for d in bound[m])))
        by_rank.setdefault(ranks[m], set()).add(signature)
    if any(len(signatures) > 1 for signatures in by_rank.values()):
        raise ValueError(
            "symmetry-equivalent metal centres carry different arrangements; global multi-metal "
            "canonicalization is not supported"
        )
    notes = {m: centre_notes[m][m] for m in centres}
    for donor in set().union(*(bound[m] for m in centres)):
        owners = sorted((m for m in centres if donor in bound[m]), key=at.get)
        notes[donor] = ";".join(centre_notes[m][donor] for m in owners)
    # `atomNote` rather than a key of our own: RDKit reads, writes and DRAWS it, so the arrangement is visible
    # in a depiction and survives a round trip through `MolToCXSmiles` without special handling. Atom index
    # order is not a choice either; RDKit re-emits the block sorted by index whatever order it was built in.
    block = ":".join(f"{at[a]}.atomNote.{value}" for a, value in sorted(notes.items(), key=lambda x: at[x[0]]))
    fields = [
        f"atomProp:{block}",
        *_cx_bond_stereo(core, ligand_stereo, at, bond_positions),
        *_atrop_bond_stereo(core, ligand_stereo, at),
    ]
    return _append_cx(core, fields)
