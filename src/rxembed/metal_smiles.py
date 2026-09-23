"""Read and write canonical ionic-dative metal SMILES.

Dative SMILES carries constitution, point stereo, and E/Z where its graph permits. Zero-order contacts retain
their native CX Z: field. Further CX fields add atropisomer stereo, canonical polyhedron slots, and haptic
winding. Notes address the atom order just written, so parsing and writing share this module; the slot grammar
itself belongs to `metal_polyhedron`.
"""

from __future__ import annotations

import logging
import re

from rdkit import Chem, rdBase

from . import metal_isomer as _isomer
from . import metal_stereo as _coord_stereo
from .metal_core import (
    COORDINATION_METALS,
    VACANT,
    _canonical_metal_graph,
    connect_metal,
    materialized_state,
    metal_indices,
)
from .metal_polyhedron import SLOT_BOND_PROP, canonical_slots, read_slot_notes, record, slot_note, vertex_dirs
from .stereo import (
    _ATROP_STEREO,
    _atrop_code,
    _bond_stereo_code,
    _coordination_locked_double_bonds,
    _encoded_bond_stereo,
    _without_bond_stereo,
    axis_stereo,
    bond_stereo,
    defined_stereo_label,
    point_stereo,
    stereo_from_3d,
)
from .utils import hydrogen_neighbor_order, mirror_tag, remove_bond

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__: the name `set_verbose` configures

_E_BOND = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
_Z_BOND = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
_STEREO_REFS = 2
_CX_BOND_FIELD = re.compile(r"(?:^|,)([ct]):((?:\d+(?:,\d+)*)?)(?=,|$)")
_CX_BOND_PREFIX = re.compile(r"(?:^|,)[ct]:")


def _repair_haptic_bond_stereo(mol):
    """Replace shared-metal E/Z references with ligand-side references without changing CIP."""
    for bond in mol.GetBonds():
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in _E_BOND | _Z_BOND or len(refs) != _STEREO_REFS:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        shared = {n.GetIdx() for n in begin.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS} & {
            n.GetIdx() for n in end.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS
        }
        left = [n.GetIdx() for n in begin.GetNeighbors() if n.GetIdx() != end.GetIdx() and n.GetIdx() not in shared]
        right = [n.GetIdx() for n in end.GetNeighbors() if n.GetIdx() != begin.GetIdx() and n.GetIdx() not in shared]
        if not shared or not left or not right:
            continue
        expected = _bond_stereo_code(mol, bond.GetIdx()) if len(set(refs)) == _STEREO_REFS else None
        bond.SetStereoAtoms(min(left), min(right))
        if expected is None:
            continue
        for tag in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            bond.SetStereo(tag)
            if _bond_stereo_code(mol, bond.GetIdx()) == expected:
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
        if bond.GetStereo() not in _E_BOND | _Z_BOND or len(refs) != _STEREO_REFS or len(set(refs)) != _STEREO_REFS:
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
                mol = _canonical_metal_graph(raw)
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
    _encoded_bond_stereo(mol)
    _bind_slot_notes(mol)
    if remove_hs and "atomProp" not in smi:
        mol = Chem.RemoveHs(mol)
    return mol


# --- the constitution layer: dative M-L bonds, and the atom order the string was written in ---------------

_H_VALENCE = 1  # all SMILES will spend on a hydrogen; a second connection has to be dative
_METAL_STEREO_TAGS = frozenset(  # the non-tetrahedral classes perception leaves on a metal; see `write_dative`
    {Chem.ChiralType.CHI_SQUAREPLANAR, Chem.ChiralType.CHI_OCTAHEDRAL, Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL}
)


def dative_smiles(mol, *, cx=False):
    """Write canonical SMILES with dative M-donor bonds and ordinary hydrogens implicit.

    The constitution layer: connectivity, charges and ligand stereocentres. The metal's arrangement is not
    written, so cis and trans give one string, as do fac and mer; `cxsmiles` is the layer that adds
    it. Set ``cx=True`` to retain ligand E/Z and atropisomer fields without writing metal slot notes. Zero-order
    contacts require a native CX ``Z:`` appendix; plain ``~`` loses their bond type and CIP assignability. Not
    a species key on its own.
    """
    return write_dative(mol, cx=cx)[0]


def _write_native_stereo(mol, wanted, wanted_bonds, *, _rebase=True):  # noqa: C901 - validates both native stereo kinds in one pass
    """Write SMILES, restoring only point and double-bond stereo already proved on the ligand graph."""
    raw_tags = {
        idx: Chem.ChiralType.CHI_TETRAHEDRAL_CW if code == "CW" else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        for idx, code in wanted.items()
        if code in {"CW", "CCW"}
    }
    mol = Chem.Mol(mol)
    for idx, tag in raw_tags.items():
        mol.GetAtomWithIdx(idx).SetChiralTag(tag)
    params = Chem.SmilesWriteParams()
    params.cleanStereo = False  # every retained tag was proved upstream; RDKit otherwise deletes chiral amines
    if _rebase and any(
        bond.GetBondType() == Chem.BondType.DATIVE and bond.GetBeginAtomIdx() in wanted for bond in mol.GetBonds()
    ):
        # RDKit's arrow writer excludes outgoing dative bonds when deciding whether to rebase a point tag.
        # Its coordinate-bond CX writer includes every carrier. Parse that canonical atom and bond order
        # once, then emit arrows in the fixed traversal below; repeatedly canonicalizing corrected tags cycles.
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
        for idx in raw_tags:
            tag = back.GetAtomWithIdx(positions[idx]).GetChiralTag()
            if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                raise ValueError(f"native canonicalization lost point stereo at atom {idx}")
            mapped_wanted[positions[idx]] = "CW" if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "CCW"
        mapped_bonds = {frozenset(positions[idx] for idx in pair): code for pair, code in wanted_bonds.items()}
        smi, rebased_order, rebased_bonds = _write_native_stereo(back, mapped_wanted, mapped_bonds, _rebase=False)
        return smi, [order[idx] for idx in rebased_order], rebased_bonds
    if not _rebase:
        params.canonical = False
    defined = point_stereo(defined_stereo_label(mol, metal_indices(mol))) if wanted else {}
    tags = {idx: mol.GetAtomWithIdx(idx).GetChiralTag() for idx, code in wanted.items() if defined.get(idx) == code}
    native_tags = {
        idx: mol.GetAtomWithIdx(idx).GetChiralTag()
        for idx in wanted
        if mol.GetAtomWithIdx(idx).GetChiralTag()
        in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    }

    def bond_positions(graph):
        atoms = list(graph.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
        bonds = list(graph.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
        at = {idx: pos for pos, idx in enumerate(atoms)}
        return {
            frozenset((at[bond.GetBeginAtomIdx()], at[bond.GetEndAtomIdx()])): pos
            for pos, idx in enumerate(bonds)
            for bond in (graph.GetBondWithIdx(idx),)
        }

    def wrong(graph, mapped):
        label = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = point_stereo(label), bond_stereo(label)

        def native_tag_matches(index, target):
            # RDKit can preserve an explicit tag on hypervalent centres while it cannot assign a CIP label
            # or use that centre in a chiral substructure match (for example [S@+2]). The tag is still a
            # meaningful parser round-trip contract; ordinary centres take the stronger CIP/match path below.
            return (
                index in native_tags
                and points.get(target) is None
                and graph.GetAtomWithIdx(target).GetChiralTag() == native_tags[index]
            )

        wrong_points = [
            idx
            for idx, code in wanted.items()
            if idx not in tags
            and code in {"R", "S", "r", "s"}
            and points.get(mapped[idx]) != code
            and not native_tag_matches(idx, mapped[idx])
        ]
        for idx, tag in (raw_tags | native_tags | tags).items():
            if len(mapped) != mol.GetNumAtoms() or idx not in mapped:
                wrong_points.append(idx)
                continue
            source, target = Chem.Mol(mol), Chem.Mol(graph)
            # Compare the local permutation with all carriers present. Simultaneously correcting CIP labels
            # can oscillate when fixing one donor changes a neighbouring pseudoasymmetric descriptor.
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
            if not target.HasSubstructMatch(source, useChirality=True) and not native_tag_matches(idx, mapped[idx]):
                wrong_points.append(idx)
        wrong_bonds = [
            pair for pair, code in wanted_bonds.items() if bonds.get(frozenset(mapped[idx] for idx in pair)) != code
        ]
        return wrong_points, wrong_bonds

    def compatible_order(graph):
        query = Chem.Mol(mol)
        for bond in query.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            for adjacent in (*bond.GetBeginAtom().GetBonds(), *bond.GetEndAtom().GetBonds()):
                adjacent.SetBondDir(Chem.BondDir.NONE)
        match = graph.GetSubstructMatch(query, useChirality=True)
        if len(match) != mol.GetNumAtoms():
            return None
        label = defined_stereo_label(graph, metal_indices(graph))
        points, bonds = point_stereo(label), bond_stereo(label)
        if any(code in {"R", "S", "r", "s"} and points.get(match[idx]) != code for idx, code in wanted.items()):
            return None
        if any(bonds.get(frozenset(match[idx] for idx in pair)) != code for pair, code in wanted_bonds.items()):
            return None
        order = [None] * len(match)
        for old, new in enumerate(match):
            order[new] = old
        return order

    def flip_bond(graph, pair):
        i, j = pair
        bond = graph.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            return
        left = next(
            (
                adjacent.GetOtherAtomIdx(bond.GetBeginAtomIdx())
                for adjacent in bond.GetBeginAtom().GetBonds()
                if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondType() == Chem.BondType.SINGLE
            ),
            None,
        )
        right = next(
            (
                adjacent.GetOtherAtomIdx(bond.GetEndAtomIdx())
                for adjacent in bond.GetEndAtom().GetBonds()
                if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondType() == Chem.BondType.SINGLE
            ),
            None,
        )
        if left is None or right is None or left == right:
            return
        bond.SetStereoAtoms(left, right)
        bond.SetStereo(Chem.BondStereo.STEREOZ if bond.GetStereo() in _E_BOND else Chem.BondStereo.STEREOE)

    candidate = Chem.Mol(mol)

    def writes(graph):
        for writer in (params,) if not _rebase else (None, params):
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

    attempts = max(3, len(wanted) + len(wanted_bonds) + 2)
    for _ in range(attempts):
        for bond in candidate.GetBonds():
            if bond.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT):
                bond.SetBondDir(Chem.BondDir.NONE)
        if wanted_bonds:
            Chem.SetDoubleBondNeighborDirections(candidate)
        for smi, written_mol in writes(candidate):
            order = list(written_mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
            positions = {idx: pos for pos, idx in enumerate(order)}
            if not wanted and not wanted_bonds:
                return smi, order, bond_positions(written_mol)
            with rdBase.BlockLogs():
                back = parse_smiles(smi, remove_hs=False)
            if _rebase and (compatible := compatible_order(back)) is not None:
                return smi, compatible, bond_positions(written_mol)
            wrong_points, wrong_bonds = wrong(back, positions)
            if not wrong_points and not wrong_bonds:
                return smi, order, bond_positions(written_mol)
        for idx in wrong_points:
            atom = candidate.GetAtomWithIdx(idx)
            tag = atom.GetChiralTag()
            atom.SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CW if tag == Chem.ChiralType.CHI_UNSPECIFIED else mirror_tag(tag)
            )
        for pair in wrong_bonds:
            flip_bond(candidate, pair)

    detail = []
    if wrong_points:
        detail.append(f"atom(s) {sorted(wrong_points)}")
    if wrong_bonds:
        detail.append(f"bond(s) {sorted(tuple(sorted(pair)) for pair in wrong_bonds)}")
    raise ValueError(f"could not write ligand stereo at {' and '.join(detail)}")


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


def _round_trip_graph_error(expected, actual, written):
    """Return the first constitutional change made by parsing a written SMILES."""
    if actual is None:
        return "does not parse"
    if len(set(written)) != len(written):
        return "has an inconsistent written atom map"

    def implicit_h(idx):
        atom = expected.GetAtomWithIdx(int(idx))
        return (
            atom.GetAtomicNum() == 1
            and atom.GetIsotope() == atom.GetFormalCharge() == atom.GetNumRadicalElectrons() == 0
            and atom.GetDegree() == 1
            and atom.GetNeighbors()[0].GetAtomicNum() not in {*COORDINATION_METALS, 1}
        )

    explicit = [int(idx) for idx in written if not implicit_h(idx)]
    if actual.GetNumAtoms() != len(explicit):
        return "has an inconsistent written atom map"
    at = {old: new for new, old in enumerate(explicit)}
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

    def atom_key(atom):
        return (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetIsAromatic(),
        )

    for old, new in at.items():
        if atom_key(expected.GetAtomWithIdx(old)) != atom_key(actual.GetAtomWithIdx(new)):
            return f"changes atom {old}"

    def bonds(mol, mapping):
        out = {}
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() not in mapping or bond.GetEndAtomIdx() not in mapping:
                continue  # a routine H may be represented by its neighbour's total-H count
            begin, end = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
            pair = tuple(sorted((begin, end)))
            direction = (begin, end) if bond.GetBondType() == Chem.BondType.DATIVE else None
            out[pair] = (bond.GetBondType(), bond.GetIsAromatic(), direction)
        return out

    if bonds(expected, at) != bonds(actual, {idx: idx for idx in range(actual.GetNumAtoms())}):
        return "changes connectivity, bond order, or dative direction"
    return None


def _write_dative(mol, stereo_label):  # noqa: C901 - one graph-normalisation transaction
    """Return canonical dative SMILES plus atom-position and bond-position maps.

    The positions are what an `atomProp` block indexes, so the CXSMILES writer needs them and cannot get
    them from the string. Only atoms that remain explicit have a position.

    Every accepted M-donor bond is normalized to dative before writing. A hydrogen with more than one
    connection keeps its nonmetal ligand leg covalent and donates through metal legs. For H2 or a nonmetal
    relay, distance and canonical graph rank break the tie.

    Raises rather than hand back a string that does not round-trip, since a SMILES you cannot read back is
    worse than none.
    """
    expected_mol = Chem.Mol(mol)
    expected_mol.UpdatePropertyCache(strict=False)
    source_atoms = expected_mol.GetNumAtoms()
    expected = expected_mol.GetNumAtoms() + sum(
        a.GetTotalNumHs() for a in expected_mol.GetAtoms() if a.GetAtomicNum() != 1
    )
    stereo_label = _without_bond_stereo(
        stereo_label,
        _coordination_locked_double_bonds(expected_mol, metal_indices(expected_mol)),
    )
    rw = Chem.RWMol(expected_mol)
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    ranks = list(Chem.CanonicalRankAtoms(expected_mol, breakTies=False))
    for atom in mol.GetAtoms():
        h = atom.GetIdx()
        nbrs = [bond.GetOtherAtomIdx(h) for bond in atom.GetBonds() if bond.GetValenceContrib(atom) > 0]
        if atom.GetAtomicNum() != 1 or len(nbrs) <= _H_VALENCE:
            # already normalised: keep the one covalent bond the string explicitly states
            continue
        ordered = hydrogen_neighbor_order(
            expected_mol,
            h,
            metals=COORDINATION_METALS,
            positions=pos,
            ranks=ranks,
        )
        nbrs = [neighbor for neighbor in ordered if neighbor in nbrs]
        for n in nbrs[1:]:
            remove_bond(rw, h, n)  # re-seats the bond LAST at `n`, so the partner's tag moves basis with it
            rw.AddBond(h, n, Chem.BondType.DATIVE)  # H donates: a dative bond spends the end atom's valence

    out = _canonical_metal_graph(rw.GetMol())
    out.UpdatePropertyCache(strict=False)
    wanted = point_stereo(stereo_label)
    wanted_bonds = bond_stereo(stereo_label)
    materialize_h = set()
    for pair in wanted_bonds:
        bond = out.GetBondBetweenAtoms(*pair)
        if bond is None:
            continue
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if atom.GetTotalNumHs() and not any(
                adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondType() == Chem.BondType.SINGLE
                for adjacent in atom.GetBonds()
            ):
                materialize_h.add(atom.GetIdx())
    if materialize_h:
        out = Chem.AddHs(out, onlyOnAtoms=sorted(materialize_h), addCoords=bool(out.GetNumConformers()))
    if not stereo_label:
        # The caller already resolved stereo. Do not infer and then discard it again.
        Chem.RemoveStereochemistry(out)
        for bond in out.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)  # RDKit leaves single-bond atrop tags behind.
    elif out.GetNumConformers():
        # Normalizing metal bonds changes neighbour order; rebase retained stereo from coordinates.
        stereo_from_3d(out, metal_indices(out), apply=True)

    point_tags = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    for atom in out.GetAtoms():
        if atom.GetChiralTag() in point_tags and atom.GetIdx() not in wanted:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in out.GetBonds():
        pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetStereo() in _E_BOND | _Z_BOND and pair not in wanted_bonds:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                for adjacent in atom.GetBonds():
                    if adjacent.GetIdx() != bond.GetIdx():
                        adjacent.SetBondDir(Chem.BondDir.NONE)

    # The metal's own tag is a permutation index over the neighbour order too, but no geometry settles it:
    # Fe(CO)5 writes @TB20, @TB14 or @TB13 for one molecule, and SMILES has a class for 3 of the 12
    # arrangements here. Dropping it takes the corpus 26 -> 40 stable under reordering; see the docstring
    # for what that costs.
    for atom in out.GetAtoms():
        if atom.GetChiralTag() in _METAL_STEREO_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_chiralPermutation"):
                atom.ClearProp("_chiralPermutation")

    # Suppress routine AddHs output. A hydrogen bound to the metal remains explicit because it can own a slot.
    keep_h = {
        atom.GetIdx()
        for atom in out.GetAtoms()
        if atom.GetAtomicNum() == 1 and any(n.GetAtomicNum() in COORDINATION_METALS for n in atom.GetNeighbors())
    }
    for pair in wanted_bonds:
        bond = out.GetBondBetweenAtoms(*pair)
        if bond is None:
            continue
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if any(
                adjacent.GetIdx() != bond.GetIdx()
                and adjacent.GetBondType() == Chem.BondType.SINGLE
                and adjacent.GetOtherAtom(atom).GetAtomicNum() != 1
                for adjacent in atom.GetBonds()
            ):
                continue
            keep_h.update(neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
    out, reduced = _coord_stereo.remove_routine_hydrogens(out, keep_h)
    original = {new: old for old, new in reduced.items()}
    wanted = {reduced[idx]: code for idx, code in wanted.items() if idx in reduced}
    wanted_bonds = {
        frozenset(reduced[idx] for idx in pair): code for pair, code in wanted_bonds.items() if pair <= reduced.keys()
    }
    unwritable = {pair for pair in wanted_bonds if not _writable_ez(out, pair)}
    if unwritable:
        _clear_bond_stereo(out, unwritable)
        wanted_bonds = {pair: code for pair, code in wanted_bonds.items() if pair not in unwritable}
    smi, written, bonds = _write_native_stereo(out, wanted, wanted_bonds)
    back = Chem.MolFromSmiles(smi)
    actual = (
        None
        if back is None
        else back.GetNumAtoms() + sum(a.GetTotalNumHs() for a in back.GetAtoms() if a.GetAtomicNum() != 1)
    )
    graph_error = _round_trip_graph_error(out, back, written)
    if actual != expected or graph_error:
        got = "does not parse" if actual is None else f"parses back as {actual} atoms"
        if graph_error and actual == expected:
            got = graph_error
        raise ValueError(
            f"could not write a graph-preserving round-tripping SMILES for this complex "
            f"({expected} atoms including H): the "
            f"result {got}. The perceived graph is likely one SMILES cannot express (a hypervalent or "
            f"partial-bond centre); work from the Mol itself."
        )
    return (
        smi,
        {original[int(a)]: p for p, a in enumerate(written) if original[int(a)] < source_atoms},
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
            if bond.GetStereo() in _ATROP_STEREO and _atrop_code(candidate, bond) == target:
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
        if bond is None or bond.GetStereo() not in _ATROP_STEREO or _atrop_code(back, bond) != target:
            raise ValueError("could not retain atropisomer stereo with native CX wU/wD fields")
    return [block]


def write_dative(mol, stereo_label=None, *, cx=False):
    """Return canonical dative SMILES and its original-atom to output-position map.

    Plain SMILES cannot carry E/Z on an eta2 alkene bonded to the metal at both ends. Use `cxsmiles` for that
    graph; ``cx=True`` writes the standard CX ``c:``/``t:`` and atropisomer fields without metal arrangement
    notes.
    """
    if stereo_label is None:
        stereo_label = (
            stereo_from_3d(mol, metal_indices(mol))
            if mol.GetNumConformers()
            else defined_stereo_label(mol, metal_indices(mol))
        )
    if not cx and (axis_stereo(stereo_label) or any(bond.GetStereo() in _ATROP_STEREO for bond in mol.GetBonds())):
        raise ValueError("plain dative SMILES cannot retain atropisomer stereo; use cxsmiles()")
    smi, at, _bonds, unwritable = _write_dative(mol, stereo_label)
    if cx:
        fields = [
            *_cx_bond_stereo(smi, stereo_label, at, _bonds),
            *_atrop_bond_stereo(smi, stereo_label, at),
        ]
        return _append_cx(smi, fields), at
    if unwritable:
        detail = sorted(tuple(sorted(pair)) for pair in unwritable)
        logger.warning("dative SMILES: RDKit cannot read E/Z at bond(s) %s; use cxsmiles() to retain it", detail)
    return smi, at


# --- the arrangement layer: the canonical slot note ------------------------------------------------------


def _site_keys(iso, vertices, haptic, winding):
    """Return one order-invariant key per vertex (``None`` at a vacancy): what decides which sites tie.

    A sigma donor is its symmetry class. A haptic face is the symmetry class of the complete atom set,
    so two identical Cp rings tie and constitutionally different faces do not, plus its winding where present.
    Never an atom index and never a position in a string, so any writer folding on these agrees.

    The `Isomer` owns the winding. A geometry measured by `from_geometry` has already stored it; a vertex-only
    isomer honestly leaves it empty.
    """
    classes = _coord_stereo.site_classes(iso._graph, vertices, haptic, _isomer.isomer_roles(iso))
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
        _coord_stereo.equivalent_site_assignments(
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


def _rebuild(iso):
    """Reconnect an `Isomer` with its authoritative donor-to-metal bonds before canonical writing."""
    return connect_metal(iso.restore(Chem.Mol(iso._graph)), iso.donor_bonds)


def _append_cx(core, fields):
    """Append CX fields without discarding the core's native constitutional bond fields."""
    body, _, existing = core.partition("|")
    block = ",".join(field for field in (existing.removesuffix("|"), *fields) if field)
    return f"{body.rstrip()} |{block}|" if block else body.rstrip()


def _cx_bond_stereo(core, stereo_label, at, bonds):
    """Return lossless CX fields for E/Z that plain SMILES cannot retain."""
    fields = {"c": set(), "t": set()}
    plain = parse_smiles(core, remove_hs=False)
    encoded = []

    def realised(graph, pair):
        label = defined_stereo_label(graph, metal_indices(graph))
        return bond_stereo(label).get(pair)

    for pair, expected in bond_stereo(stereo_label).items():
        if not pair <= at.keys():
            raise ValueError(f"could not name double-bond stereo on bond {tuple(sorted(pair))}")
        key = frozenset(at[a] for a in pair)
        written = plain.GetBondBetweenAtoms(*key)
        if written is not None and realised(plain, key) == expected:
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
            if candidate is not None and realised(probe, key) == expected:
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
    work = iso._graph
    vertices, haptic, winding, _donors = materialized_state(iso, state)
    keys = _canonical_site_keys(iso, vertices, haptic, winding, at)
    links = _coord_stereo.chelate_links(work, vertices, haptic)
    slots = canonical_slots(dirs, keys, links)
    geom = record(state.geometry).code + (f"-{state.hand}" if state.hand else "")
    return {state.atom: geom} | _slot_notes(vertices, haptic, keys, slots, at, links)


def cxsmiles(source):
    """Write canonical dative CXSMILES carrying the selected metal state.

    The plain core is a constitution key. Metal notes store geometry and hand; donor notes store canonical
    slots and haptic winding. Native bond directions and standard CX ``c:``/``t:`` fields retain E/Z where
    RDKit supports them; paired ``_rxEZ`` atom properties are the lossless fallback. Multiple centres use
    dative adjacency, and a bridge stores one slot per adjacent metal.

    `source` is an `Isomer` or a conformer-bearing `Mol`. Missing polyhedron templates, contradictory stereo,
    and symmetry-equivalent metals carrying different states fail rather than lose identity.
    """
    iso = None if isinstance(source, Chem.Mol) else source
    complexed = source if iso is None else _rebuild(iso)  # a Mol is already its own constitution
    # Donor bookkeeping below must see the same canonical graph the centre notes were derived from.
    complexed = _canonical_metal_graph(complexed)
    metals = metal_indices(complexed)
    if not metals:
        raise ValueError("no transition metal found")
    if iso is not None:
        ligand_stereo = iso.stereo_label
    elif complexed.GetNumConformers():
        ligand_stereo = stereo_from_3d(complexed, metals)
    else:
        ligand_stereo = defined_stereo_label(complexed, metals)
    locked = _coordination_locked_double_bonds(complexed, metals)
    ligand_stereo = _without_bond_stereo(ligand_stereo, locked)
    bound = {
        m: {n.GetIdx() for n in complexed.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals} for m in metals
    }
    centres = tuple(m for m in metals if bound[m])
    if not centres:
        core, at, bond_positions, _unwritable = _write_dative(complexed, ligand_stereo)
        fields = [
            *_cx_bond_stereo(core, ligand_stereo, at, bond_positions),
            *_atrop_bond_stereo(core, ligand_stereo, at),
        ]
        return _append_cx(core, fields)
    if iso is not None:
        records = [(iso, state) for state in _isomer.centre_states(iso)]
    elif complexed.GetNumConformers():
        records = [(record, record.centres[0]) for record in (_isomer.from_geometry(source, center=m) for m in centres)]
    else:
        from .metal_enumeration import _from_stated_arrangements, stated_arrangement

        if not all(stated_arrangement(complexed, center=m) is not None for m in centres):
            raise ValueError("cxsmiles needs an Isomer, coordinates, or one stated arrangement per metal")
        stated = _from_stated_arrangements(complexed, centres[0], "model")
        records = [(stated, state) for state in _isomer.centre_states(stated)]
    if {state.atom for _record, state in records} != set(centres):
        raise ValueError("the isomer does not carry one state per metal")
    core, at, bond_positions, _unwritable = _write_dative(complexed, ligand_stereo)
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
