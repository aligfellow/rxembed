"""Read and write canonical ionic-dative metal SMILES.

Plain dative SMILES carries constitution, point stereo, and E/Z where its graph permits. CX fields add native
atropisomer stereo, canonical polyhedron slots, and haptic winding. Notes address the atom order just written,
so parsing and writing share this module; the slot grammar itself belongs to `metal_polyhedron`.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from rdkit import Chem

from . import metal_isomer as _isomer
from . import metal_stereo as _coord_stereo
from .metal_core import (
    _ETA2,
    _PT,
    COORDINATION_METALS,
    VACANT,
    HapticSite,
    connect_metal,
    donated_charge,
    ligand_valence,
    materialized_state,
    metal_indices,
)
from .metal_polyhedron import SLOT_BOND_PROP, canonical_slots, read_slot_notes, record, slot_note, vertex_dirs
from .stereo import _ATROP_STEREO, axis_stereo, defined_stereo_label, point_stereo, stereo_from_3d
from .utils import assign_stereo_from_3d, mirror_tag, remove_bond

logger = logging.getLogger("rxembed.metal")  # spelled out, not __name__: the name `set_verbose` configures

_E_BOND = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
_Z_BOND = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
_STEREO_REFS = 2
_CX_BOND_FIELD = re.compile(r"(?:^|,)([ct]):((?:\d+(?:,\d+)*)?)(?=,|$)")
_CX_BOND_PREFIX = re.compile(r"(?:^|,)[ct]:")


def _repair_haptic_bond_stereo(mol):
    """Replace a duplicated shared-metal stereo reference with the two ligand-side references."""
    for bond in mol.GetBonds():
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in _E_BOND | _Z_BOND or len(refs) != _STEREO_REFS or len(set(refs)) == _STEREO_REFS:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        shared = {n.GetIdx() for n in begin.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS} & {
            n.GetIdx() for n in end.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS
        }
        left = [n.GetIdx() for n in begin.GetNeighbors() if n.GetIdx() != end.GetIdx() and n.GetIdx() not in shared]
        right = [n.GetIdx() for n in end.GetNeighbors() if n.GetIdx() != begin.GetIdx() and n.GetIdx() not in shared]
        if len(shared) == len(left) == len(right) == 1:
            bond.SetStereoAtoms(left[0], right[0])


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
        expected = _Z_BOND if marker == "c" else _E_BOND
        refs = list(bond.GetStereoAtoms())
        if bond.GetStereo() not in expected or len(refs) != _STEREO_REFS or len(set(refs)) != _STEREO_REFS:
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


def parse_smiles(smi):
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
        try:
            Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            try:
                rw = Chem.RWMol(raw)
                rw.UpdatePropertyCache(strict=False)
                _donate_to_metal(rw)
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
                logger.info("parse: normalized invalid covalent metal bonds to dative")
            except (RuntimeError, ValueError):
                mol = None
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smi!r}")
    Chem.SetBondStereoFromDirections(mol)
    _repair_haptic_bond_stereo(mol)
    _validate_cx_bond_stereo(smi, mol)
    _bind_slot_notes(mol)
    if "atomProp" not in smi:
        mol = Chem.RemoveHs(mol)
    return mol


# --- the constitution layer: dative M-L bonds, and the atom order the string was written in ---------------

_H_VALENCE = 1  # all SMILES will spend on a hydrogen; a second connection has to be dative
_METAL_STEREO_TAGS = frozenset(  # the non-tetrahedral classes perception leaves on a metal; see `write_dative`
    {Chem.ChiralType.CHI_SQUAREPLANAR, Chem.ChiralType.CHI_OCTAHEDRAL, Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL}
)


def _donate_to_metal(rw):
    """Normalize M-L bonds from ligand-side valence, in place.

    Full-shell donors become dative; anionic donors retain their stated charge. An undervalent neutral donor
    receives the charge implied by `donated_charge`, with the balance placed on the metal. The same rule owns
    distance typing, so written constitution and embedded geometry agree. Ambiguous neutral bridges fail.
    """
    for donor in rw.GetAtoms():
        adjacent = [n.GetIdx() for n in donor.GetNeighbors() if n.GetAtomicNum() in COORDINATION_METALS]
        if len(adjacent) > 1 and donor.GetFormalCharge() == 0 and donated_charge(donor):
            raise ValueError(
                f"neutral bridging donor {donor.GetSymbol()}{donor.GetIdx()} has ambiguous charge allocation "
                f"between metals {adjacent}; state the donor and metal formal charges explicitly"
            )

    ionic, covalent, charged = [], [], {}
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
            elif not single and not donor.GetFormalCharge():
                valence = ligand_valence(donor)
                other_donors = [
                    n.GetIdx()
                    for n in rw.GetAtomWithIdx(m).GetNeighbors()
                    if n.GetIdx() != d and n.GetAtomicNum() not in COORDINATION_METALS
                ]
                haptic = any(rw.GetBondBetweenAtoms(d, other) is not None for other in other_donors)
                if not haptic and 0 < valence < _PT.GetDefaultValence(donor.GetAtomicNum()):
                    covalent.append((d, m))
            break
    for d, m in charged.items():
        q = donated_charge(rw.GetAtomWithIdx(d))
        rw.GetAtomWithIdx(d).SetFormalCharge(q)
        rw.GetAtomWithIdx(m).SetFormalCharge(rw.GetAtomWithIdx(m).GetFormalCharge() - q)
    for d, m in ionic:
        remove_bond(rw, d, m)  # re-seats the bond LAST at the metal, so the donor's tag moves basis with it
        rw.AddBond(d, m, Chem.BondType.DATIVE)
    for d, m in covalent:
        remove_bond(rw, d, m)
        rw.AddBond(d, m, Chem.BondType.SINGLE)


def dative_smiles(mol):
    """Write canonical SMILES with dative M-donor bonds and ordinary hydrogens implicit.

    The constitution layer: connectivity, charges and ligand stereocentres. The metal's arrangement is not
    written, so cis and trans give one string, as do fac and mer; `cxsmiles` is the layer that adds
    it. Not a species key on its own.
    """
    return write_dative(mol)[0]


def _write_native_stereo(mol, wanted):
    """Write SMILES, restoring only point and double-bond stereo already proved on the ligand graph."""

    def bond_positions(graph):
        atoms = list(graph.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
        bonds = list(graph.GetPropsAsDict(True, True)["_smilesBondOutputOrder"])
        at = {idx: pos for pos, idx in enumerate(atoms)}
        return {
            frozenset((at[bond.GetBeginAtomIdx()], at[bond.GetEndAtomIdx()])): pos
            for pos, idx in enumerate(bonds)
            for bond in (graph.GetBondWithIdx(idx),)
        }

    has_bond_stereo = any(
        bond.GetStereo() in _E_BOND | _Z_BOND
        and len(bond.GetStereoAtoms()) == _STEREO_REFS
        and len(set(bond.GetStereoAtoms())) == _STEREO_REFS
        for bond in mol.GetBonds()
    )
    if has_bond_stereo:
        Chem.SetDoubleBondNeighborDirections(mol)
        params = Chem.SmilesWriteParams()
        params.cleanStereo = False
        smi = Chem.MolToSmiles(mol, params)
    else:
        smi = Chem.MolToSmiles(mol)
    order = list(mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    if not wanted:
        return smi, order, bond_positions(mol)

    clean = parse_smiles(smi)
    actual = point_stereo(defined_stereo_label(clean, metal_indices(clean)))
    positions = {idx: pos for pos, idx in enumerate(order)}
    missing = {idx: code for idx, code in wanted.items() if actual.get(positions[idx]) != code}
    if not missing:
        return smi, order, bond_positions(mol)

    for idx in missing:
        clean.GetAtomWithIdx(positions[idx]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    params = Chem.SmilesWriteParams()
    params.cleanStereo = False  # `clean` came from RDKit's clean writer; only proved stereo was restored.
    for _ in range(2):
        smi = Chem.MolToSmiles(clean, params)
        clean_order = list(clean.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
        clean_positions = {idx: pos for pos, idx in enumerate(clean_order)}
        back = parse_smiles(smi)
        actual = point_stereo(defined_stereo_label(back, metal_indices(back)))
        wrong = [idx for idx, code in wanted.items() if actual.get(clean_positions[positions[idx]]) != code]
        if not wrong:
            return smi, [order[idx] for idx in clean_order], bond_positions(clean)
        for idx in wrong:
            atom = clean.GetAtomWithIdx(positions[idx])
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
    raise ValueError(f"could not write ligand stereo at atom(s) {sorted(wrong)}")


def _write_dative(mol, stereo_label=None):
    """Return canonical dative SMILES plus atom-position and bond-position maps.

    The positions are what an `atomProp` block indexes, so the CXSMILES writer needs them and cannot get
    them from the string. Only atoms that remain explicit have a position.

    Perception already writes M-donor bonds as dative, so this is `MolToSmiles` plus the two repairs SMILES
    needs. A hydrogen with more than one connection - a side-on H2, a bridging hydride, an H-bond relay
    perceived as a bond - has no valence left for a second single bond, so it keeps its shortest bond and
    donates through the rest. That is the real electron flow for sigma-donated H2 and a convention for the
    relay, which SMILES has no way to say otherwise.

    Raises rather than hand back a string that does not round-trip, since a SMILES you cannot read back is
    worse than none.
    """
    expected_mol = Chem.Mol(mol)
    expected_mol.UpdatePropertyCache(strict=False)
    expected = expected_mol.GetNumAtoms() + sum(
        a.GetTotalNumHs() for a in expected_mol.GetAtoms() if a.GetAtomicNum() != 1
    )
    if stereo_label is None:
        if mol.GetNumConformers():
            stereo_label = stereo_from_3d(mol, metal_indices(mol))
        else:
            stereo_label = defined_stereo_label(mol, metal_indices(mol))
    rw = Chem.RWMol(mol)
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    for atom in mol.GetAtoms():
        h = atom.GetIdx()
        nbrs = [bond.GetOtherAtomIdx(h) for bond in atom.GetBonds() if bond.GetBondType() != Chem.BondType.DATIVE]
        if atom.GetAtomicNum() != 1 or len(nbrs) <= _H_VALENCE:
            # already normalised: keep the one covalent bond the string explicitly states
            continue
        if pos is not None:  # without a geometry the graph order is all there is to go on
            nbrs.sort(key=lambda n: float(np.sum((pos[n] - pos[h]) ** 2)))
        for n in nbrs[1:]:
            remove_bond(rw, h, n)  # re-seats the bond LAST at `n`, so the partner's tag moves basis with it
            rw.AddBond(h, n, Chem.BondType.DATIVE)  # H donates: a dative bond spends the end atom's valence

    _donate_to_metal(rw)

    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    bond_stereo = {
        bond.GetIdx(): (bond.GetStereo(), tuple(bond.GetStereoAtoms()))
        for bond in out.GetBonds()
        if bond.GetStereo() in _E_BOND | _Z_BOND
        and len(bond.GetStereoAtoms()) == _STEREO_REFS
        and len(set(bond.GetStereoAtoms())) == _STEREO_REFS
    }
    wanted = {idx: code for idx, code in point_stereo(stereo_label).items() if code in {"R", "S"}}
    # E/Z references atoms picked from the bond's neighbour order, which a renumber does not update: 4 of the
    # 5 corpus structures unstable under reordering differed only in `/` and `\`. Geometry has no such order.
    if out.GetNumConformers():
        assign_stereo_from_3d(out)  # the one door, never the raw call: see its docstring on the dative basis
        for idx, (tag, refs) in bond_stereo.items():
            bond = out.GetBondWithIdx(idx)
            bond.SetStereoAtoms(*refs)
            bond.SetStereo(tag)

    point_tags = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    for atom in out.GetAtoms():
        if atom.GetChiralTag() in point_tags and atom.GetIdx() not in wanted:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

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
    keep_h = [
        atom.GetIdx()
        for atom in out.GetAtoms()
        if atom.GetAtomicNum() == 1 and any(n.GetAtomicNum() in COORDINATION_METALS for n in atom.GetNeighbors())
    ]
    out, reduced = _coord_stereo.remove_routine_hydrogens(out, keep_h)
    original = {new: old for old, new in reduced.items()}
    wanted = {reduced[idx]: code for idx, code in wanted.items() if idx in reduced}
    smi, written, bonds = _write_native_stereo(out, wanted)
    back = Chem.MolFromSmiles(smi)
    actual = (
        None
        if back is None
        else back.GetNumAtoms() + sum(a.GetTotalNumHs() for a in back.GetAtoms() if a.GetAtomicNum() != 1)
    )
    if actual != expected:
        got = "does not parse" if actual is None else f"parses back as {actual} atoms"
        raise ValueError(
            f"could not write a round-tripping SMILES for this complex ({expected} atoms including H): the "
            f"result {got}. The perceived graph is likely one SMILES cannot express (a hypervalent or "
            f"partial-bond centre); work from the Mol itself."
        )
    return smi, {original[int(a)]: p for p, a in enumerate(written)}, bonds


def _same_bond_stereo(left, right):
    """Return whether two RDKit double-bond tags name the same E/Z geometry."""
    return (left in _E_BOND and right in _E_BOND) or (left in _Z_BOND and right in _Z_BOND)


def _atrop_code(mol, bond):
    """Return RDKit's sequence-rule descriptor for one assigned atropisomer bond."""
    Chem.AssignCIPLabels(mol, bondsToLabel=[bond.GetIdx()])
    return bond.GetPropsAsDict().get("_CIPCode")


def _atrop_bond_stereo(core, stereo_label, at):
    """Return the native CX wU/wD field retaining every assigned atropisomer axis."""
    axes = axis_stereo(stereo_label)
    if not axes:
        return []
    plain = parse_smiles(core)
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
    back = parse_smiles(f"{core} |{block}|")
    for mapped, target in targets:
        bond = back.GetBondBetweenAtoms(*mapped)
        if bond is None or bond.GetStereo() not in _ATROP_STEREO or _atrop_code(back, bond) != target:
            raise ValueError("could not retain atropisomer stereo with native CX wU/wD fields")
    return [block]


def write_dative(mol, stereo_label=None):
    """Return canonical dative SMILES and its original-atom to output-position map.

    Plain SMILES cannot carry E/Z on an eta2 alkene bonded to the metal at both ends. Use `cxsmiles` for that
    graph; it writes the standard CX ``c:``/``t:`` bond field.
    """
    smi, at, _bonds = _write_dative(mol, stereo_label)
    if any(bond.GetStereo() in _ATROP_STEREO for bond in mol.GetBonds()):
        raise ValueError("plain dative SMILES cannot retain atropisomer stereo; use cxsmiles()")
    back = parse_smiles(smi)
    for bond in mol.GetBonds():
        if bond.GetStereo() not in _E_BOND | _Z_BOND or not {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= at.keys():
            continue
        written = back.GetBondBetweenAtoms(at[bond.GetBeginAtomIdx()], at[bond.GetEndAtomIdx()])
        if written is None or not _same_bond_stereo(bond.GetStereo(), written.GetStereo()):
            raise ValueError("plain dative SMILES cannot retain this double-bond stereo; use cxsmiles()")
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


def _slot_notes(vertices, haptic, keys, slots, at, bites):
    """Return canonical slot notes without separating donors that belong to one ligand.

    Identical chelates and haptic ligands may swap as units. Group them by constitution, not winding, so
    exchanging two identical faces with opposite windings cannot change the canonical string.
    """
    groups = [{v} for v, d in enumerate(vertices) if d != VACANT]
    for bite in bites:
        joined = [group for group in groups if group & bite]
        groups = [group for group in groups if group not in joined] + [set().union(*joined)]

    by_ligand = {}
    for group in groups:
        sites = []
        assigned = []
        for v in group:
            d = vertices[v]
            atoms = tuple(haptic[d]) if d in haptic else (d,)
            sites.append((keys[v], min(at[a] for a in atoms), atoms))
            assigned.append((keys[v], slots[v]))
        by_ligand.setdefault(tuple(sorted(keys[v][0] for v in group)), []).append((sites, assigned))

    notes = {}
    for ligands in by_ligand.values():
        sites = sorted((site for site, _assigned in ligands), key=lambda group: tuple(sorted(s[1] for s in group)))
        assigned = sorted((assigned for _site, assigned in ligands), key=lambda group: tuple(sorted(group)))
        for ligand_sites, ligand_slots in zip(sites, assigned, strict=True):
            for constitution in {site[0][0] for site in ligand_sites}:
                atoms = sorted((site[1], site[2]) for site in ligand_sites if site[0][0] == constitution)
                assignments = sorted((key, slot) for key, slot in ligand_slots if key[0] == constitution)
                for (_position, site_atoms), (key, slot) in zip(atoms, assignments, strict=True):
                    for atom in site_atoms:
                        notes[atom] = slot_note(slot, key[1])
    return notes


def _rebuild(iso):
    """Reconnect an `Isomer` before canonical valence-based dative writing.

    Sigma donors return as single bonds so `_donate_to_metal` derives the ionic form consistently. Haptic
    bonds remain dative because their ring atoms share one coordination site rather than acting as separate
    sigma donors.
    """
    faces = [site.atoms for state in iso.centres for site in state.vertices if isinstance(site, HapticSite)]
    ring = {atom for face in faces for atom in face}
    whole = connect_metal(iso.restore(Chem.Mol(iso._graph)), [b for b in iso.donor_bonds if b[0] in ring])
    return connect_metal(whole, [b for b in iso.donor_bonds if b[0] not in ring], order=Chem.BondType.SINGLE)


def _haptic_bond_stereo(core, records, at, bonds):
    """Return standard CX ``c:``/``t:`` fields needed to retain eta2 E/Z."""
    fields = {"c": set(), "t": set()}
    plain = parse_smiles(core)
    for iso, state in records:
        _vertices, haptic, _winding, _donors = materialized_state(iso, state)
        for face in haptic.values():
            if len(face) != _ETA2:
                continue
            source = iso._graph.GetBondBetweenAtoms(*face)
            if source is None or source.GetStereo() not in _E_BOND | _Z_BOND:
                continue
            key = frozenset(at[a] for a in face)
            written = plain.GetBondBetweenAtoms(*key)
            if written is not None and _same_bond_stereo(source.GetStereo(), written.GetStereo()):
                continue
            bond_index = bonds.get(key)
            if bond_index is None:
                raise ValueError("could not locate the eta2 double bond in the written CXSMILES")
            for marker in ("c", "t"):
                try:
                    probe = parse_smiles(f"{core} |{marker}:{bond_index}|")
                except ValueError:
                    continue
                candidate = probe.GetBondBetweenAtoms(*key)
                if candidate is not None and _same_bond_stereo(source.GetStereo(), candidate.GetStereo()):
                    fields[marker].add(bond_index)
                    break
            else:
                raise ValueError("could not retain eta2 double-bond stereo with standard CX c:/t: fields")
    return [f"{marker}:{','.join(map(str, sorted(indices)))}" for marker, indices in fields.items() if indices]


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
    keys = _site_keys(iso, vertices, haptic, winding)
    bites = _coord_stereo.chelate_edges(work, vertices, haptic)
    slots = canonical_slots(dirs, keys, bites)
    geom = record(state.geometry).code + (f"-{state.hand}" if state.hand else "")
    return {state.atom: geom} | _slot_notes(vertices, haptic, keys, slots, at, bites)


def cxsmiles(source):
    """Write canonical dative CXSMILES carrying the selected metal state.

    The plain core is a constitution key. Metal notes store geometry and hand; donor notes store canonical
    slots and haptic winding. Native bond directions retain eta2 E/Z, with standard CX ``c:``/``t:`` fields
    as fallback. Multiple centres use dative adjacency, and a bridge stores one slot per adjacent metal.

    `source` is an `Isomer` or a conformer-bearing `Mol`. Missing polyhedron templates, contradictory stereo,
    and symmetry-equivalent metals carrying different states fail rather than lose identity.
    """
    iso = None if isinstance(source, Chem.Mol) else source
    complexed = source if iso is None else _rebuild(iso)  # a Mol is already its own constitution
    metals = metal_indices(complexed)
    if not metals:
        raise ValueError("no transition metal found")
    if iso is not None:
        ligand_stereo = iso.stereo_label
    elif complexed.GetNumConformers():
        ligand_stereo = stereo_from_3d(complexed, metals)
    else:
        ligand_stereo = defined_stereo_label(complexed, metals)
    bound = {
        m: {n.GetIdx() for n in complexed.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() not in metals} for m in metals
    }
    records = (
        [(record, record.centres[0]) for record in (_isomer.from_geometry(source, center=m) for m in metals)]
        if iso is None
        else [(iso, state) for state in _isomer.centre_states(iso)]
    )
    if {state.atom for _record, state in records} != set(metals):
        raise ValueError("the isomer does not carry one state per metal")
    core, at, bond_positions = _write_dative(complexed, ligand_stereo)
    centre_notes = {}
    for record_iso, state in records:
        centre_notes[state.atom] = _arrangement_notes(record_iso, state, at)
    ranks = list(Chem.CanonicalRankAtoms(complexed, breakTies=False))
    by_rank = {}
    for m in metals:
        signature = (centre_notes[m][m], tuple(sorted((ranks[d], centre_notes[m][d]) for d in bound[m])))
        by_rank.setdefault(ranks[m], set()).add(signature)
    if any(len(signatures) > 1 for signatures in by_rank.values()):
        raise ValueError(
            "symmetry-equivalent metal centres carry different arrangements; global multi-metal "
            "canonicalization is not supported"
        )
    notes = {m: centre_notes[m][m] for m in metals}
    for donor in set().union(*bound.values()):
        centres = sorted((m for m in metals if donor in bound[m]), key=at.get)
        notes[donor] = ";".join(centre_notes[m][donor] for m in centres)
    # `atomNote` rather than a key of our own: RDKit reads, writes and DRAWS it, so the arrangement is visible
    # in a depiction and survives a round trip through `MolToCXSmiles` without special handling. Atom index
    # order is not a choice either; RDKit re-emits the block sorted by index whatever order it was built in.
    block = ":".join(f"{at[a]}.atomNote.{value}" for a, value in sorted(notes.items(), key=lambda x: at[x[0]]))
    fields = [
        f"atomProp:{block}",
        *_haptic_bond_stereo(core, records, at, bond_positions),
        *_atrop_bond_stereo(core, ligand_stereo, at),
    ]
    return f"{core} |{','.join(fields)}|"
