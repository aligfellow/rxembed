"""Label the ligand stereo a Mol specifies or a conformer shows, native atrop axes included, with its metals cut."""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

from rxembed.graph import haptic_sites, ligand_graph, metal_indices, torsion_path
from rxembed.utils import (
    DISCONNECTED,
    EZ_RELATION,
    POINT_TAGS,
    TETRAHEDRAL_TAGS,
    bond_removal_mirrors,
    flat_ranks,
    mirror_tag,
    remove_bond,
    tetrahedral_perception,
    without_zero_bonds,
)

_MIN_POINT_BRANCHES = 3
_CHELATE_LOCK_BONDS = 3  # axis atom, ortho atom, donor, metal
_MIN_BLOCKED_ORTHO_CONNECTIONS = 3  # the ring's two and a substituent or fusion
_MIN_BRIDGE_METALS = 2
_ISOTOPE_ELEMENT_STRIDE = 128  # exceeds the periodic table, so equal-element bridge caps stay distinct
ATROP_STEREO = (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
_ATROP_WEDGE = (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH)
_POINT_CIP = {"R", "S", "r", "s"}
_LONE_PAIR_GROUPS = {5, 6}  # outer electrons of groups 15 and 16: a three-coordinate atom keeps a lone pair
_RESONANCE_EZ_CAP = 32  # forms searched per fragment: independent conjugated groups multiply its form count
_RX_EZ_PROP = re.compile(r"_rxEZ(\d+)")
# Each ligand stereo kind: the separator between its atoms in a label, and the codes that name its configurations.
ELEMENT_KINDS = {"point": ("", ("R", "S", "r", "s", "CW", "CCW")), "ez": ("=", ("E", "Z")), "axis": ("-", ("M", "P"))}
# A `stereo=` selector shorthand: a bare code, or element symbols joined as `ELEMENT_KINDS` joins its atoms.
_SHORTHAND = re.compile(r"(?:([A-Z][a-z]?)(?:([=-])([A-Z][a-z]?))?:)?(\w+)")
logger = logging.getLogger("rxembed.stereo")  # under the "rxembed" tree `set_verbose` configures
_STEREO_SANITIZE = (
    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StereoElement:
    """Name one ligand stereo element: its `ELEMENT_KINDS` kind, its atoms as the label orders them, and its code."""

    kind: str
    atoms: tuple[int, ...]
    symbols: tuple[str, ...]  # each atom's element symbol, which the label writes before its index
    code: str

    def __str__(self):
        """Return the element's label, e.g. ``'C5=C6:E'``."""
        site = ELEMENT_KINDS[self.kind][0].join(f"{s}{a}" for s, a in zip(self.symbols, self.atoms, strict=True))
        return f"{site}:{self.code}"


def _element(mol, kind, atoms, code):
    """Return the `StereoElement` of `atoms` in `mol`."""
    symbols = tuple(mol.GetAtomWithIdx(atom).GetSymbol() for atom in atoms)
    return StereoElement(kind=kind, atoms=tuple(atoms), symbols=symbols, code=code)


@dataclass(frozen=True, slots=True, kw_only=True)
class LigandStereo:
    """Hold the ligand stereo a species specifies or a conformer shows, one `StereoElement` each, in label order.

    `str()` is the public label, e.g. ``'C3:R,C5=C6:E,C9-C10:M'``: points, double bonds and atropisomer axes.
    """

    elements: tuple[StereoElement, ...] = ()

    def __str__(self):
        """Return the label."""
        return ",".join(map(str, self.elements))

    def __bool__(self):
        """Return whether any element is specified."""
        return bool(self.elements)

    @property
    def points(self):
        """Return each point centre's code, keyed by its atom."""
        return {e.atoms[0]: e.code for e in self.elements if e.kind == "point"}

    @property
    def bonds(self):
        """Return each double bond's E/Z code, keyed by its atom pair."""
        return {frozenset(e.atoms): e.code for e in self.elements if e.kind == "ez"}

    @property
    def axes(self):
        """Return each atropisomer axis's M/P code, keyed by its sorted atom pair."""
        return {e.atoms: e.code for e in self.elements if e.kind == "axis"}

    def updated(self, other):
        """Return these elements with `other`'s replacing those on the same atoms, the rest of `other`'s appended."""
        theirs = {(e.kind, e.atoms): e for e in other.elements}
        kept = tuple(theirs.pop((e.kind, e.atoms), e) for e in self.elements)
        return LigandStereo(elements=kept + tuple(theirs.values()))


@dataclass(frozen=True, slots=True, kw_only=True)
class Miss:
    """Name one specified stereo element a conformer shows otherwise; `observed` is empty when it shows none.

    `kind` is 'point', 'ez' or 'axis' for ligand stereo; for a centre, 'hand' for the mirror image of its seating,
    'seating' for another arrangement and 'face' for a face winding.
    """

    kind: str
    atoms: tuple[int, ...]
    specified: str
    observed: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MetalCap:
    """Name the metal one cap of the ligand-stereo work graph stands for (`StereoGraph._built`).

    `metal` is its index and `atomic_num` its element; `donated` says the donor bond was a donor-to-metal dative, and
    `mirrors` that replacing the metal by the cap changed the donor's tetrahedral parity.
    """

    metal: int
    atomic_num: int
    donated: bool
    mirrors: bool


def ligand_elements(stereo):
    """Return a `LigandStereo` as ``{atoms: (kind, code)}`` in point, double-bond, axis order."""
    return {
        **{(e.atoms[0],): ("point", e.code) for e in stereo.elements if e.kind == "point"},
        **{tuple(sorted(frozenset(e.atoms))): ("ez", e.code) for e in stereo.elements if e.kind == "ez"},
        **{e.atoms: ("axis", e.code) for e in stereo.elements if e.kind == "axis"},
    }


def ligand_misses(specified, observed, specified_mol, observed_mol):
    """Return a `Miss` for each element of `specified` that `observed` reads otherwise, or not at all.

    This is the one comparison of specified with observed ligand stereo. `observed` is what the conformer shows, written
    as tags on `observed_mol`, which shares `specified_mol`'s atom order. A point both sides tag is compared by its
    local tag, since its CIP code depends on which other centres are assigned; one whose code reads the same either way
    (a pseudoasymmetric r or s) is named "the other hand". Anything else is compared by code.
    """
    found = ligand_elements(observed)
    misses = []
    for atoms, (kind, code) in ligand_elements(specified).items():
        read = found.get(atoms, (kind, ""))[1]
        tag = specified_mol.GetAtomWithIdx(atoms[0]).GetChiralTag() if kind == "point" else None
        if read and tag in TETRAHEDRAL_TAGS:
            if observed_mol.GetAtomWithIdx(atoms[0]).GetChiralTag() == tag:
                continue
            read = "the other hand" if read == code else read
        elif read == code:
            continue
        misses.append(Miss(kind=kind, atoms=atoms, specified=code, observed=read))
    return tuple(misses)


def _stereo_key(side, points, pairs):
    """Return a copy of `side` stating only the points and double bonds compared, its dative bonds single."""
    key = Chem.RWMol(side)
    key.RemoveAllConformers()
    key.ClearComputedProps()  # props left by earlier perception can make one species write two SMILES
    for atom in key.GetAtoms():
        if atom.GetIdx() not in points:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in key.GetBonds():
        if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) not in pairs:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)  # RDKit writes no hand on a dative donor
    Chem.SetDoubleBondNeighborDirections(key)  # with its props cleared, the SMILES writer reads E/Z from directions
    return key


def stereo_symmetry(specified_mol, observed_mol, specified):
    """Return the `Chem.RenumberAtoms` order that moves `specified` on `specified_mol` onto `observed_mol`, or None.

    `observed_mol` carries the tags its conformer shows. Each side is written as canonical SMILES of the stereo
    compared, so equal strings pair the atoms by output position: a graph symmetry with charges, isotopes, hydrogens and
    maps.
    """
    points = {
        idx
        for idx in specified.points
        if {specified_mol.GetAtomWithIdx(idx).GetChiralTag(), observed_mol.GetAtomWithIdx(idx).GetChiralTag()}
        <= TETRAHEDRAL_TAGS
    }
    keys = [_stereo_key(side, points, specified.bonds.keys()) for side in (specified_mol, observed_mol)]
    if len({Chem.MolToSmiles(key) for key in keys}) != 1:
        return None
    written, read = (key.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"] for key in keys)
    order = [0] * specified_mol.GetNumAtoms()
    for specified_atom, observed_atom in zip(written, read, strict=True):
        order[observed_atom] = specified_atom
    return order


def encoded_bond_stereo(mol):
    """Return the ``{pair: code}`` E/Z that `mol`'s CX ``_rxEZ`` records state; a malformed record raises."""
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


def _double_bond_elements(mol, **options):
    """Return RDKit's potential double-bond stereo elements of `mol`, keyed by atom pair."""
    return {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())): element
        for element in Chem.FindPotentialStereo(mol, **options)
        if element.type == Chem.StereoType.Bond_Double
        for bond in (mol.GetBondWithIdx(element.centeredOn),)
    }


def bond_references(mol, element):
    """Return the first real atom (not an implicit hydrogen) controlling each end of double-bond `element`."""
    controls = list(element.controllingAtoms)
    return tuple(next((idx for idx in side if idx < mol.GetNumAtoms()), None) for side in (controls[:2], controls[2:]))


def apply_encoded_bond_stereo(mol):
    """Apply CX-encoded E/Z against the references RDKit perceives, verifying each code by CIP.

    E is TRANS only relative to the CIP-leading substituents, so a code set against arbitrary neighbours can land
    as the other isomer.
    """
    encoded = encoded_bond_stereo(mol)
    if not encoded:
        return
    potential = _double_bond_elements(mol)
    for pair, wanted in encoded.items():
        element = potential.get(pair)
        if element is None:
            raise ValueError(f"could not restore CX E/Z on bond {tuple(sorted(pair))}")
        bond = mol.GetBondWithIdx(element.centeredOn)
        left, right = bond_references(mol, element)
        if left is None or right is None or left == right:
            raise ValueError(f"could not restore CX E/Z references on bond {tuple(sorted(pair))}")
        bond.SetStereoAtoms(left, right)
        bond.SetStereo(Chem.BondStereo.STEREOE)
        if bond_stereo_code(mol, bond.GetIdx()) != wanted:
            bond.SetStereo(Chem.BondStereo.STEREOZ)
        if bond_stereo_code(mol, bond.GetIdx()) != wanted:
            raise ValueError(f"could not restore CX {wanted} on bond {tuple(sorted(pair))}")


def apply_point_stereo(mol, stereo, centers):
    """Apply the absolute point codes of `stereo` at `centers` as local tags in ``mol``'s full-graph bond order."""
    centers = set(centers)
    for idx, wanted in stereo.points.items():
        if idx not in centers:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if wanted in POINT_TAGS:
            atom.SetChiralTag(POINT_TAGS[wanted])
            atom.ClearProp("_CIPCode")
            continue
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
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
    return {"trans": "E", "cis": "Z"}.get(EZ_RELATION.get(bond.GetStereo()))


def _caps_as_metals(mol, cap_to_metal):
    """Return a copy of `mol` with each cap given its metal's atomic number, for coordinated-complex CIP."""
    rw = Chem.RWMol(mol)
    charged = set()
    for d_idx, cap in cap_to_metal.items():
        rw.GetAtomWithIdx(d_idx).SetAtomicNum(cap.atomic_num)
        rw.GetAtomWithIdx(d_idx).SetIsotope(0)
        donor = next(iter(rw.GetAtomWithIdx(d_idx).GetNeighbors()))
        # a single donor-metal bond for CIP keeps the Lewis electron count: N -> N+, C- -> C
        if cap.donated and donor.GetIdx() not in charged:
            donor.SetFormalCharge(donor.GetFormalCharge() + 1)
            charged.add(donor.GetIdx())
        donor.SetNoImplicit(True)
    out = rw.GetMol()
    Chem.SanitizeMol(out, _STEREO_SANITIZE, catchErrors=True)
    return out


def _stereo_label(mol, atom_centers, bond_centers, atrop_centers, cap_to_metal, point_codes):
    """Return the `LigandStereo` of the given centres, e.g. ``'C1:R,C3=C4:E,C5-C6:M'``.

    A point reads RDKit's R/S or r/s, else its raw CW/CCW tag (a centre CIP cannot rank, e.g. some P). Full-graph
    ``point_codes`` win over capped-ligand codes: removing a metal can change priorities even at a remote centre.
    ``cap_to_metal`` gives each cap its metal's atomic number for centres RDKit cannot label on the full graph.
    """
    if cap_to_metal:
        mol = _caps_as_metals(mol, cap_to_metal)
    # perceive and label apart: legacy cleanup misses para-stereocentres and depends on a global RDKit mode
    Chem.FindPotentialStereo(mol, cleanIt=True, flagPossible=False)
    point_codes = {**_point_cip_codes(mol, [idx for idx in atom_centers if idx not in point_codes]), **point_codes}
    elements = []
    for idx in atom_centers:
        code = point_codes.get(idx) or {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW",
        }.get(mol.GetAtomWithIdx(idx).GetChiralTag())
        if code:  # an unresolved centre (an allene axis RDKit can't set) is dropped, never given a '?' tag
            elements.append(_element(mol, "point", (idx,), code))
    for bidx in bond_centers:
        b = mol.GetBondWithIdx(bidx)
        if tag := bond_stereo_code(mol, bidx):
            elements.append(_element(mol, "ez", (b.GetBeginAtomIdx(), b.GetEndAtomIdx()), tag))
    for i, j in atrop_centers:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None or bond.GetStereo() not in ATROP_STEREO:
            continue
        code = atrop_code(mol, bond)
        if code not in {"M", "P"}:
            raise ValueError(f"RDKit could not assign M/P to atropisomer bond {i}-{j}")
        elements.append(_element(mol, "axis", sorted((i, j)), code))
    return LigandStereo(elements=tuple(elements))


def cip_cache_key(mol, centers=()):
    """Return a hashable key of `mol`'s chemical content and the labelled `centers`, for the CIP memo.

    A wrong key hands back another molecule's answer, so it holds every field two graphs could differ on: each atom's
    element, isotope, charge, H count and whether that count is fixed, radicals, aromaticity and chiral tag; each
    bond's ends, type, aromaticity, stereo and stereo atoms; and each enhanced stereo group's type and members.
    """
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetIsotope(),
            a.GetFormalCharge(),
            a.GetTotalNumHs(),
            a.GetNumRadicalElectrons(),
            a.GetIsAromatic(),
            a.GetChiralTag(),
            a.GetNoImplicit(),
        )
        for a in mol.GetAtoms()
    )
    bonds = tuple(
        (
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            b.GetBondType(),
            b.GetIsAromatic(),
            b.GetStereo(),
            tuple(b.GetStereoAtoms()),
        )
        for b in mol.GetBonds()
    )
    groups = tuple(
        (int(g.GetGroupType()), tuple(a.GetIdx() for a in g.GetAtoms()), tuple(b.GetIdx() for b in g.GetBonds()))
        for g in mol.GetStereoGroups()
    )
    return atoms, bonds, groups, tuple(sorted(set(centers)))


_CIP_CACHE = {}
_CIP_CACHE_MAX = 256


def _point_cip_codes(mol, centers):
    """Return absolute R/S or pseudoasymmetric r/s labels RDKit assigns on the current full graph.

    Memoised on the graph's content (atoms, bonds and the centre list, in index order), because the same
    coordination graph is asked about many times and RDKit's CIP labeller goes superlinear on a metal-closed ring.
    Chiral tags are part of the key: a centre's label depends on every other centre's configuration.
    """
    centers = list(centers)
    if not centers:
        return {}
    try:
        key = cip_cache_key(mol, centers)
    except RuntimeError:
        # an unsanitised graph fails as AssignCIPLabels would; skip the cache and let the except path handle it
        key = None
    if key is not None:
        cached = _CIP_CACHE.get(key)
        if cached is not None:
            return dict(cached)
    probe = Chem.Mol(without_zero_bonds(mol))
    for idx in centers:
        atom = probe.GetAtomWithIdx(idx)
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


def _carriers(atom):
    """Return the atom's tetrahedral carriers in its graph: bonded neighbours and hydrogens."""
    return sum(bond.GetBondType() != Chem.BondType.ZERO for bond in atom.GetBonds()) + atom.GetTotalNumHs()


def _point_readable_from_3d(capped, atom):
    """Return whether `capped`, the work-graph copy of `atom`, has a tetrahedral tag 3D can read: four carriers.

    A lone pair is the fourth carrier of a group 15 or 16 `atom` with three, metals counted.
    """
    in_group = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum()) in _LONE_PAIR_GROUPS
    lone_pair = in_group and _carriers(atom) == 3  # noqa: PLR2004
    return capped.GetDegree() >= _MIN_POINT_BRANCHES and _carriers(capped) + lone_pair == 4  # noqa: PLR2004


def _add_cap(work, atom, element, isotope, points):
    """Bond a temporary fourth carrier to `atom`: a marked `element`, at one point per conformer; return its index.

    RDKit excludes a donor-originating dative edge from tetrahedral and alkene stereo, so the cap is an ordinary
    single-bonded neighbour on this disposable stereo graph. The isotope keeps it distinct from real substituents.
    """
    cap = Chem.Atom(element)
    cap.SetNoImplicit(True)
    cap.SetIsotope(isotope)
    index = work.AddAtom(cap)
    work.AddBond(atom, index, Chem.BondType.SINGLE)
    work.GetAtomWithIdx(atom).SetNoImplicit(True)
    for conf, point in zip(work.GetConformers(), points, strict=True):
        conf.SetAtomPosition(index, point)
    return index


def _cap_lone_pairs(work, centres):
    """Return `work` with each three-carrier centre's lone pair capped on its open side, so RDKit reads it from 3D.

    The cap is a dummy atom, which CIP ranks as it ranks a lone pair. RDKit reads a three-neighbour tag with its
    lone pair as the last neighbour, so the hand read with the cap last is the uncapped tag. A flat centre has no
    open side: its cap sits on the centre and reads as non-tetrahedral.
    """
    capped = Chem.RWMol(work)
    for idx in centres:
        atom = capped.GetAtomWithIdx(idx)
        if _carriers(atom) != 3:  # noqa: PLR2004  the lone pair is the fourth carrier
            continue
        neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
        points = []
        for conf in capped.GetConformers():
            pos = conf.GetPositions()
            bonds = pos[neighbours] - pos[idx]
            side = -(bonds / np.linalg.norm(bonds, axis=1, keepdims=True)).sum(axis=0)
            length = np.linalg.norm(side)
            points.append(Point3D(*(pos[idx] + (side / length if length > 1e-6 else 0.0))))  # noqa: PLR2004
        _add_cap(capped, idx, 0, 0, points)
    return capped.GetMol()


def potential_stereo(mol):
    """Return RDKit's potential stereo elements of `mol`, found on a copy with cleanIt.

    Without cleanIt, RDKit 2026.03 reads a pseudoasymmetric centre between two tagged ones from their raw tags, so on an
    explicit-H graph its answer can depend on atom order.
    """
    return list(Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True))


def without_stale_point_tags(mol):
    """Return `mol` without point tags that mark no stereocentre: `mol` itself or a copy.

    A reader that assigns tags from 3D without RDKit's cleanup, as `MolFromMolFile(sanitize=False)` does, tags every sp3
    atom, a CH2 included, and a symmetry rank would then tell equivalent donors apart. Each tag is judged on the work
    graph, metals cut and donors capped, so a hand only coordination fixes stays. With a conformer, a tag stays where
    the conformer shows a point centre, so a pseudoasymmetric carbon between two bound amines is judged beside their
    hands; where 3D cannot measure the atom, RDKit's cleanup decides. A tag on an atom whose hand 3D cannot hold, a
    carbanion's, stays for `StereoGraph` to refuse.
    """
    tagged = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetChiralTag() in TETRAHEDRAL_TAGS]
    if not tagged:
        return mol
    graph = StereoGraph(mol, metal_indices(mol))
    work = graph.work
    has_conformer = bool(mol.GetNumConformers())
    unreadable = _non_tetrahedral_points(_cap_lone_pairs(work, graph.points), graph.points) if has_conformer else set()
    Chem.FindPotentialStereo(work, cleanIt=True, flagPossible=False)
    held = {index for index in tagged if work.GetAtomWithIdx(index).GetChiralTag() in TETRAHEDRAL_TAGS}
    if has_conformer:
        held = stereo_from_3d(graph).points.keys() | (held & unreadable)
    out = Chem.Mol(mol)
    for index in set(tagged) - held:
        if _point_readable_from_3d(work.GetAtomWithIdx(index), mol.GetAtomWithIdx(index)):
            out.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out


def _tetrahedral_centres(mol):
    """Return the atoms RDKit's potential-stereo perception flags as tetrahedral centres."""
    return {element.centeredOn for element in potential_stereo(mol) if element.type == Chem.StereoType.Atom_Tetrahedral}


def _distinct_carriers(work, ranks, index, count=4):
    """Return whether an atom carries `count` substituents, hydrogens included, in distinct graph classes."""
    atom = work.GetAtomWithIdx(index)
    classes = [ranks[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()]
    classes.extend([-1] * atom.GetTotalNumHs())
    return len(classes) == len(set(classes)) == count


def _free_graph(mol, metals):
    """Return the metal-free ligand graph, its property cache updated."""
    free = ligand_graph(mol, metals)
    free.UpdatePropertyCache(strict=False)
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


def _graft_point_tags(full, work, centers, cap_to_metal):
    """Copy enumerated point tags from the capped graph onto the coordinated graph."""
    centers = set(centers)
    for atom in work.GetAtoms():
        if atom.GetIdx() >= full.GetNumAtoms() or atom.GetIdx() not in centers:
            continue
        tag = atom.GetChiralTag()
        for cap_idx, cap in cap_to_metal.items():
            if cap.mirrors and work.GetBondBetweenAtoms(atom.GetIdx(), cap_idx) is not None:
                tag = mirror_tag(tag)
        full.GetAtomWithIdx(atom.GetIdx()).SetChiralTag(tag)


def matches_stereo(stereo, selector):
    """Return whether `LigandStereo` `stereo` matches the public `stereo=` selector, the one text read back.

    A selector names elements by label (``'C5:R'``, ``'C5=C6:E'``), by index (``'5R'``, ``'5=6:E'``) or, for the
    one element of its kind, by symbols or code alone (``'C:R'``, ``'R'``); commas join selectors. A shorthand
    that fits several elements raises with their labels.
    """
    label = str(stereo)
    if selector == label:
        return True
    if "," in selector:
        return all(matches_stereo(stereo, part.strip()) for part in selector.split(","))
    for e in stereo.elements:
        indexed = ELEMENT_KINDS[e.kind][0].join(map(str, e.atoms))
        if selector in (str(e), f"{indexed}{e.code}" if e.kind == "point" else f"{indexed}:{e.code}"):
            return True
    match = _SHORTHAND.fullmatch(selector)
    first, separator, second, wanted = match.groups() if match else (None, None, None, "")
    kind = next((kind for kind, (_sep, codes) in ELEMENT_KINDS.items() if wanted in codes), None)
    if kind is None or (first and (separator or "") != ELEMENT_KINDS[kind][0]):
        return False
    symbols = tuple(sorted(filter(None, (first, second)))) if first else None
    candidates = [e for e in stereo.elements if e.kind == kind and symbols in (None, tuple(sorted(e.symbols)))]
    if len(candidates) > 1:
        raise ValueError(f"stereo={selector!r} is ambiguous for {label!r}; use one of {list(map(str, candidates))}")
    return len(candidates) == 1 and candidates[0].code == wanted


def clear_atrop(mol):
    """Clear native atropisomer tags and their signaling wedges; return the axis atom pairs."""
    axes = set()
    for bond in mol.GetBonds():
        if bond.GetStereo() not in ATROP_STEREO:
            continue
        axes.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.ClearProp("_CIPCode")
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for adjacent in atom.GetBonds():
                if adjacent.GetIdx() != bond.GetIdx() and adjacent.GetBondDir() in _ATROP_WEDGE:
                    adjacent.SetBondDir(Chem.BondDir.NONE)
    return axes


def renumber_atoms(mol, order):
    """Return `Chem.RenumberAtoms(mol, order)` with each atropisomer axis keeping its hand.

    RDKit reads an atrop tag against the lowest-index neighbour at each axis end and does not re-base it on
    renumbering, so the tag flips when exactly one end's lowest neighbour changes.
    """
    out = Chem.RenumberAtoms(mol, order)
    for bond in out.GetBonds():
        if bond.GetStereo() not in ATROP_STEREO:
            continue
        ends = ((bond.GetBeginAtom(), bond.GetEndAtomIdx()), (bond.GetEndAtom(), bond.GetBeginAtomIdx()))
        sides = [[n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != other] for atom, other in ends]
        if sum(min(side) != min(side, key=order.__getitem__) for side in sides) % 2:
            bond.SetStereo(ATROP_STEREO[ATROP_STEREO.index(bond.GetStereo()) - 1])
    return out


def atrop_code(mol, bond):
    """Return RDKit's sequence-rule descriptor for one assigned atropisomer bond."""
    Chem.AssignCIPLabels(mol, bondsToLabel=[bond.GetIdx()])
    return bond.GetPropsAsDict().get("_CIPCode")


def atrop_tagged(mol, pair, code):
    """Return a copy of `mol` whose axis `pair` carries the native atrop tag RDKit reads as `code`, else None."""
    for tag in ATROP_STEREO:
        candidate = Chem.Mol(mol)
        bond = candidate.GetBondBetweenAtoms(*pair)
        bond.SetStereo(tag)
        Chem.CleanupAtropisomers(candidate)
        if bond.GetStereo() in ATROP_STEREO and atrop_code(candidate, bond) == code:
            return candidate
    return None


def clear_ez(mol, pairs=None):
    """Clear selected double-bond stereo and adjacent slash bonds; return whether anything changed.

    A double bond named in `pairs` loses its CX-encoded E/Z record too.
    """
    if pairs is not None and not pairs:
        return False
    selected = [
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and (pairs is None or frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in pairs)
    ]
    changed = any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in selected)
    for bond in selected:
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for begin, end in ((bond.GetBeginAtom(), bond.GetEndAtom()) for bond in selected if pairs is not None):
        for name in begin.GetPropNames(includePrivate=True, includeComputed=False):
            if _RX_EZ_PROP.fullmatch(name) and end.HasProp(name):
                changed = True
                begin.ClearProp(name)
                end.ClearProp(name)
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
    for bond in probe.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)  # RDKit will not wedge an axis beside an E/Z slash bond
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


def _refuse_unclosable_ez(graph, closed, rings, pairs):
    """Refuse a specified E/Z on a locked double bond that its smallest ring through a metal cannot close.

    That ring is too small for a trans double bond, so it holds the bond's two ring neighbours cis. A CX-encoded E/Z is
    read where RDKit applies it, on the work graph, whose cap stands for its metal.
    """
    for pair in pairs & graph.specified_ez:
        bond = graph.mol.GetBondBetweenAtoms(*pair)
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            bond = graph.work.GetBondBetweenAtoms(*pair)
        relation = EZ_RELATION.get(bond.GetStereo())
        refs = [graph.real_atom(ref) for ref in bond.GetStereoAtoms()]
        atoms, ring_bonds = min((ring for ring in rings if pair <= ring[0]), key=lambda ring: len(ring[0]))
        ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        around = []  # each end's neighbour around that ring
        for end, partner in (ends, ends[::-1]):
            ring_bond = next(
                ring_bond
                for ring_bond in map(closed.GetBondWithIdx, ring_bonds)
                if end in (ring_bond.GetBeginAtomIdx(), ring_bond.GetEndAtomIdx())
                and partner not in (ring_bond.GetBeginAtomIdx(), ring_bond.GetEndAtomIdx())
            )
            around.append(ring_bond.GetOtherAtomIdx(end))
        forced = "cis" if (refs[0] == around[0]) == (refs[1] == around[1]) else "trans"
        if relation not in (None, forced):
            name = [f"{graph.mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in (*ends, *refs, min(atoms & graph.metals))]
            raise ValueError(
                f"the E/Z at {name[0]}={name[1]} puts {name[2]} and {name[3]} {relation}, but its ring through "
                f"{name[4]} holds them {forced}"
            )


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
    free_double_pairs = _double_bond_elements(_free_graph(mol, exclude))
    stereogenic_bridges = _stereogenic_bridges(mol, metal_neighbors)
    caps = {}
    for nb, metals in metal_neighbors.items():
        if not metals:
            continue
        donor = mol.GetAtomWithIdx(nb)
        ligand_bonds = [b for b in donor.GetBonds() if b.GetOtherAtomIdx(nb) not in exclude]
        double_bonds = [b for b in ligand_bonds if b.GetBondType() == Chem.BondType.DOUBLE]
        heavy_ligand_bonds = [b for b in ligand_bonds if b.GetOtherAtom(donor).GetAtomicNum() > 1]
        # C=[NH]->M needs M as its second E/Z reference; an N-substituted imine has its own and keeps it
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


class StereoGraph:
    """Hold one Mol's ligand-stereo work graph, built once for every labeller of that Mol and its tags.

    The labellers take a Mol or its graph. `work` and `caps` are `StereoGraph._built`'s; they, the supported
    points and the locked double bonds are read on first use. A graph holds only for the tags it was built
    on: after ``stereo_from_3d(..., apply=True)``, build a new one.
    """

    def __init__(self, mol, metals=()):
        self.mol = mol
        self.metals = set(metals)

    @cached_property
    def _built(self):
        """Return ``(work, cap_to_metal)``: `mol` without zero-order contacts, each metal of `exclude` disconnected.

        A donor whose ligand stereo needs the metal as a neighbour gets a single-bonded cap in its place, and
        `cap_to_metal` maps each cap's index to its `MetalCap`. With no `exclude` nothing is disconnected.
        """
        mol, exclude = self.mol, self.metals
        mol = without_zero_bonds(mol)
        if not exclude:
            return mol, {}
        # cut each metal first: a donor is a stereocentre only while bound, a hand the surrogate cannot hold
        cap_to_metal = {}
        metal_neighbors = {
            atom.GetIdx(): [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in exclude] for atom in mol.GetAtoms()
        }
        caps = _donor_caps(mol, exclude, metal_neighbors)
        work = Chem.RWMol(mol)
        for mi in sorted(exclude):
            z_metal = mol.GetAtomWithIdx(mi).GetAtomicNum()
            donors = [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]
            # a haptic atom is part of a pi face; capping it would make a phantom R/S centre on a dative Cp ring
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
                same_element = [m for m in metal_neighbors[nb] if mol.GetAtomWithIdx(m).GetAtomicNum() == z_metal]
                isotope = 2 + z_metal + _ISOTOPE_ELEMENT_STRIDE * same_element.index(mi)
                # The cap stands where the metal was; ``cap_to_metal`` retains the real bond kind for grafting and CIP.
                d = _add_cap(work, nb, caps[nb], isotope, [conf.GetAtomPosition(mi) for conf in work.GetConformers()])
                for original in mol.GetAtomWithIdx(nb).GetBonds():
                    if original.GetBondType() != Chem.BondType.DOUBLE or original.GetOtherAtomIdx(nb) in exclude:
                        continue
                    copied = work.GetBondBetweenAtoms(original.GetBeginAtomIdx(), original.GetEndAtomIdx())
                    refs = tuple(d if ref == mi else ref for ref in original.GetStereoAtoms())
                    if mi in original.GetStereoAtoms() and len(refs) == 2:  # noqa: PLR2004
                        copied.SetStereoAtoms(*refs)
                        copied.SetStereo(original.GetStereo())
                cap_to_metal[d] = MetalCap(metal=mi, atomic_num=z_metal, donated=donated, mirrors=replacement_mirrors)
        work = work.GetMol()
        Chem.SanitizeMol(work, _STEREO_SANITIZE, catchErrors=True)
        apply_encoded_bond_stereo(work)
        return work, cap_to_metal

    @property
    def caps(self):
        """Return the work graph's metal caps as ``{cap: MetalCap}`` (`StereoGraph._built`)."""
        return self._built[1]

    def real_atom(self, ref):
        """Return the atom of `mol` a work-graph atom stands for: a cap's metal, else the atom itself."""
        return self.caps[ref].metal if ref in self.caps else ref

    def apply_bond_stereo(self, full, work, bond_indices):
        """Apply selected double-bond tags from `work` to `full`, with cap references mapped back to metals.

        Leave unassigned bonds alone and set the directions RDKit's writer reads. The caller decides which
        bonds to clear or retain, including coordination-locked E/Z.
        """
        for source in map(work.GetBondWithIdx, bond_indices):
            if source.GetStereo() == Chem.BondStereo.STEREONONE:
                continue
            target = full.GetBondBetweenAtoms(source.GetBeginAtomIdx(), source.GetEndAtomIdx())
            refs = [self.real_atom(ref) for ref in source.GetStereoAtoms()]
            if target is not None and len(refs) == 2:  # noqa: PLR2004
                target.SetStereoAtoms(*refs)
                target.SetStereo(source.GetStereo())
        Chem.SetDoubleBondNeighborDirections(full)

    @property
    def work(self):
        """Return a fresh copy of the work graph, as a new build would, so no reader edits `mol` through it."""
        return Chem.Mol(self._built[0])

    @classmethod
    def of(cls, mol, metals=()):
        """Return the graph of `mol` with `metals` cut; a built graph, which carries its own, passes through."""
        return mol if isinstance(mol, cls) else cls(mol, metals)

    @cached_property
    def _capability(self):
        """Return supported and unmeasurable point centres on the authoritative graph, `metals` cut and capped."""
        mol, work, cap_to_metal, metals = self.mol, self.work, self.caps, self.metals
        tagged_points = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetIdx() not in metals and atom.GetChiralTag() in TETRAHEDRAL_TAGS
        }
        cap_donors = {cap: work.GetAtomWithIdx(cap).GetNeighbors()[0].GetIdx() for cap in cap_to_metal}
        ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False, includeAtomMaps=False))
        # a neutral N inverts unless RDKit holds it metal-free or a metal on its lone pair locks four distinct carriers
        free = _free_graph(mol, metals)
        free_ranks = list(Chem.CanonicalRankAtoms(free, breakTies=False, includeChirality=False, includeAtomMaps=False))
        held = {index for index in _tetrahedral_centres(free) if _distinct_carriers(free, free_ranks, index, 3)}
        held -= _cage_fixed_nitrogens(free, held)
        held.update(index for index in cap_donors.values() if _distinct_carriers(work, ranks, index))
        potential = _tetrahedral_centres(mol) | {
            index for index in _tetrahedral_centres(work) if index < mol.GetNumAtoms()
        }
        donors = list(cap_donors.values())
        bridged = {idx for idx in set(donors) if donors.count(idx) >= _MIN_BRIDGE_METALS}
        potential.update(bridged)
        potential.difference_update(metals)
        cap_created = set(donors)
        # A temporary metal cap is only a fourth carrier. It cannot distinguish symmetry-equivalent ligand arms.
        potential.difference_update(
            index
            for index in potential - tagged_points - bridged
            if index in cap_created and not _distinct_carriers(work, ranks, index)
        )
        potential.difference_update(
            index
            for index in potential - held - bridged
            if work.GetAtomWithIdx(index).GetAtomicNum() == 7  # noqa: PLR2004  nitrogen
            and work.GetAtomWithIdx(index).GetFormalCharge() == 0
        )
        # carriers only the drawn Lewis form tells apart are one group: bond orders and charges move no atom
        flat = flat_ranks(work)
        lewis_only = {
            index
            for index in tagged_points | potential
            if len({flat[n.GetIdx()] for n in work.GetAtomWithIdx(index).GetNeighbors()})
            < len({ranks[n.GetIdx()] for n in work.GetAtomWithIdx(index).GetNeighbors()})
        }
        tagged_points -= lewis_only
        potential -= lewis_only
        candidates = tagged_points | potential
        supported = {
            idx for idx in candidates if _point_readable_from_3d(work.GetAtomWithIdx(idx), mol.GetAtomWithIdx(idx))
        }
        impossible = sorted(tagged_points - supported)
        if impossible:
            raise ValueError(
                f"point stereo at atom(s) {impossible} cannot be read from 3D; remove the tag or use four neighbours"
            )
        return supported, potential - supported

    @property
    def points(self):
        """Return the point centres RDKit can measure from 3D (`StereoGraph._capability`)."""
        return self._capability[0]

    @property
    def unmeasurable(self):
        """Return the potential point centres RDKit cannot measure from 3D."""
        return self._capability[1]

    @cached_property
    def locked(self):
        """Return the double bonds whose E/Z the coordination fixes: endocyclic in a ring closed through a metal.

        Such a bond has one buildable geometry, which the coordination isomer decides; the other hand would force a bite
        the chelate cannot span. A specified E/Z the ring cannot close is refused; one it can close is no separate
        species.

        RDKit ignores dative bonds in ring perception, so the datives are made single to reveal the ring. A double bond
        still in a ring without the metal is an organic ring bond, which RDKit already handles.
        """
        mol, metals = self.mol, self.metals
        if not metals:
            return set()
        doubles = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
        if not doubles:
            encoded_bond_stereo(mol)  # a stale CX E/Z record still refuses when no bond can carry it
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
            for bond in doubles
            if bond.GetBeginAtomIdx() in site and bond.GetEndAtomIdx() in site
        }
        # A diene face's central bond, drawn double in its metallacyclopentene form, is the face's s-cis/s-trans class.
        central = {frozenset(path[1:3]) for site in faces if (path := torsion_path(mol, site)) is not None}
        closed, metal_rings = _metal_closed_rings(mol, metals)
        ring_stereo = _double_bond_elements(closed, cleanIt=False, flagPossible=True)
        free = ligand_graph(mol, metals)  # the metal-free graph: which double bonds are still cyclic without it?
        Chem.FastFindRings(free)
        locked = set()
        for b in doubles:
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
        _refuse_unclosable_ez(self, closed, metal_rings, locked - central)
        return locked

    @cached_property
    def locked_axes(self):
        """Return the locked biaryl axes of a metal complex: closed by a chelate, or tetra-ortho-substituted.

        A chelate locks an axis whose atoms both sit three bonds from one metal, through donors on the rings' ortho
        atoms (BINAP, BIPHEP): syn clashes the donors and anti puts them beyond one metal's reach. A heavy substituent
        or ring fusion on all four ortho atoms locks it by sterics (NAPHOS). A ring-atom donor (bipyridine) holds its
        axis planar, and fewer ortho substituents let it rotate, so such an axis is stereo only when specified. RDKit
        must accept the axis as a native atrop bond.
        """
        mol, metals, work = self.mol, self.metals, self.work
        if not metals:
            return []
        mol = Chem.Mol(mol)  # native distance and ring queries populate caches
        dist = Chem.GetDistanceMatrix(without_zero_bonds(mol), force=True)
        ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=True, includeAtomMaps=False))
        axes = []
        for bond in mol.GetBonds():
            pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if (
                bond.GetBondType() != Chem.BondType.SINGLE
                or bond.IsInRing()
                or not all(mol.GetAtomWithIdx(end).GetIsAromatic() for end in pair)
            ):
                continue
            ortho = [[n for n in work.GetAtomWithIdx(end).GetNeighbors() if n.GetIdx() not in pair] for end in pair]
            if any(len(side) != len({ranks[n.GetIdx()] for n in side}) for side in ortho):
                continue  # equal-ranked ortho atoms leave the axis no hand
            chelated = any(dist[pair[0], m] == dist[pair[1], m] == _CHELATE_LOCK_BONDS for m in metals)
            hindered = all(
                sum(adjacent.GetAtomicNum() > 1 for adjacent in n.GetNeighbors()) >= _MIN_BLOCKED_ORTHO_CONNECTIONS
                for side in ortho
                for n in side
            )
            if not (chelated or hindered):
                continue
            probe = Chem.Mol(work)
            probe.GetBondBetweenAtoms(*pair).SetStereo(Chem.BondStereo.STEREOATROPCW)
            Chem.CleanupAtropisomers(probe)
            if probe.GetBondBetweenAtoms(*pair).GetStereo() == Chem.BondStereo.STEREOATROPCW:
                axes.append(tuple(sorted(pair)))
        return axes

    @cached_property
    def locked_centres(self):
        """Return the donor stereocentres that exist only while bound: labile when free, fixed by coordination.

        RDKit's potential-stereo rule on the metal-free graph decides lability: a free amine N inverts, so it is no
        stereocentre there, while a phosphine P or a ring carbon is. A locked hand belongs to the coordination
        arrangement, yet no arrangement fixes it: a chelate ring can twist far enough to reverse it.
        """
        donors = {n.GetIdx() for m in self.metals for n in self.mol.GetAtomWithIdx(m).GetNeighbors()} - self.metals
        return (self.points & donors) - _tetrahedral_centres(_free_graph(self.mol, self.metals))

    @cached_property
    def specified_ez(self):
        """Return the double bonds whose E/Z the Mol specifies, natively or CX-encoded, as atom pairs."""
        return {
            frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in self.mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetStereo() != Chem.BondStereo.STEREONONE
            and len(bond.GetStereoAtoms()) == 2  # noqa: PLR2004
        } | set(encoded_bond_stereo(self.mol))

    @cached_property
    def missed_ez(self):
        """Return the bonds of `specified_ez` whose E/Z the Mol's conformer does not show: all of them without one.

        A CX-encoded E/Z, which no conformer carries, always counts.
        """
        if not self.specified_ez or not self.mol.GetNumConformers():
            return self.specified_ez
        observed_mol = Chem.Mol(self.mol)
        observed = stereo_from_3d(observed_mol, self.metals, apply=True)
        misses = ligand_misses(defined_stereo_label(self), observed, self.mol, observed_mol)
        return {frozenset(miss.atoms) for miss in misses if miss.kind == "ez"} | set(encoded_bond_stereo(self.mol))

    def enumerable_ez(self, work, potential_double, cleared=()):
        """Return the bonds of `work`, a copy of this graph's, whose E/Z enumeration may set.

        A bond qualifies when the input specifies it (`specified_ez`, or `cleared` pairs a request cleared) or its E/Z
        survives resonance, once per cumulene. `potential_double` holds RDKit's potential double-bond elements by bond
        index.
        """
        specified = self.specified_ez | {frozenset(pair) for pair in cleared}
        stated_bonds = {bond.GetIdx() for pair in specified if (bond := work.GetBondBetweenAtoms(*pair)) is not None}
        return {
            index
            for index in _resonance_stable_ez(work, potential_double, stated_bonds, structural=True)
            if index == min(component := cumulene_component(work, index))
            and _cumulene_terminal_controls(work, component, potential_double) is not None
        }

    def label(self, full, work, points, bonds, axes):
        """Return the `LigandStereo` of the given centres of `work`, first grafting its point tags onto `full`.

        A coordination-locked double bond (`locked`) is left out: its E/Z is the arrangement's, not ligand stereo.
        """
        _graft_point_tags(full, work, points, self.caps)
        ends = ((work.GetBondWithIdx(b).GetBeginAtomIdx(), work.GetBondWithIdx(b).GetEndAtomIdx()) for b in bonds)
        bonds = [b for b, pair in zip(bonds, ends, strict=True) if frozenset(pair) not in self.locked]
        return _stereo_label(work, points, bonds, axes, self.caps, _point_cip_codes(full, points))

    def centres(self, work):
        """Return the point, E/Z and native atrop centres `work`, a copy of this graph's work graph, defines."""
        points = [i for i in range(self.mol.GetNumAtoms()) if work.GetAtomWithIdx(i).GetChiralTag() in TETRAHEDRAL_TAGS]
        tagged_bonds = [b for b in work.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE]
        doubles = [b.GetIdx() for b in tagged_bonds if b.GetBondType() == Chem.BondType.DOUBLE]
        axes = [
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in tagged_bonds
            if b.GetStereo() in ATROP_STEREO
        ]
        return points, doubles, axes


def cumulene_component(work, bond_index):
    """Return the bond indices of the cumulated double bonds connected to bond `bond_index`, itself included."""
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
    component = cumulene_component(probe, bond_index)
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


def _ez_description(form, pair, structural):
    """Return a double bond's E/Z-defining graph signature, with its stereo code unless `structural`."""
    graph = _potential_ez_signature(form, pair)
    if structural or graph is None:
        return graph
    bond = form.GetBondBetweenAtoms(*pair)
    code = bond_stereo_code(form, bond.GetIdx())
    return (graph, code) if code else None


def _resonance_stable_ez(work, bond_centers, specified, *, structural=False):
    """Return the bonds of `bond_centers` that are `specified`, or whose inferred E/Z survives every resonance form.

    A delocalised charge can swap the CIP-leading path without moving an atom, turning E into Z, or move a double bond
    or extend it into a cumulene; such labels are Lewis-form artefacts, while explicit input is authoritative. The
    search over resonance forms is bounded: past the bound, or where RDKit cannot enumerate a fragment, a bond in a
    conjugated group abstains with a warning.
    """
    keep = set(specified)
    candidates = [idx for idx in bond_centers if idx not in keep]
    if not candidates:
        return list(bond_centers)
    unproven = set()
    mappings = []
    fragments = Chem.GetMolFrags(work, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mappings)
    bonds = [work.GetBondWithIdx(idx) for idx in candidates]
    for fragment, atoms in zip(fragments, mappings, strict=True):
        local = {original: idx for idx, original in enumerate(atoms)}
        selected = {
            bond.GetIdx(): (local[bond.GetBeginAtomIdx()], local[bond.GetEndAtomIdx()])
            for bond in bonds
            if bond.GetBeginAtomIdx() in local
        }
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
                    with tetrahedral_perception():
                        Chem.AssignStereochemistryFrom3D(
                            form,
                            confId=form.GetConformer().GetId(),
                            replaceExistingTags=True,
                        )
                stable -= {idx for idx in stable if _ez_description(form, selected[idx], structural) != expected[idx]}
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
    """Return the `LigandStereo` a coordinated molecule defines, its coordination-locked E/Z left out.

    `mol` is a Mol, with `exclude` its metals, or a `StereoGraph`, which carries its own.
    """
    graph = StereoGraph.of(mol, exclude)
    mol, work = graph.mol, graph.work
    stereo = graph.label(Chem.Mol(mol), work, *graph.centres(work))
    native = stereo.bonds
    encoded = []
    for pair, code in encoded_bond_stereo(mol).items():
        if pair in native and native[pair] != code:
            raise ValueError(f"native and CX E/Z disagree on bond {tuple(sorted(pair))}")
        if pair not in native and pair not in graph.locked:
            encoded.append(_element(mol, "ez", sorted(pair), code))
    return LigandStereo(elements=stereo.elements + tuple(encoded))


def _non_tetrahedral_points(mol, centres):
    """Return the four-carrier points of `centres` whose atom lies outside its carriers' open tetrahedron."""
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
    """Return the `LigandStereo` a conformer shows, leaving non-tetrahedral points unassigned.

    RDKit's 3D assignment uses three neighbours in bond order, and outside the neighbour tetrahedron those choices can
    disagree, so a point counts only with its centre inside that tetrahedron, as native DG requires. ``apply=True`` also
    writes what is read as tags, removing invalid point tags; a coordination-locked E/Z is written but left out of the
    record. `mol` is a Mol with a conformer and `exclude` its metals, or a `StereoGraph`, which carries its own.
    """
    graph = StereoGraph.of(mol, exclude)
    mol = graph.mol
    supported_points, work = graph.points, graph.work
    _, stated_bonds, stated_axes = graph.centres(work)
    inferred = set(graph.locked_axes) - set(stated_axes)
    work = _cap_lone_pairs(work, supported_points)
    with tetrahedral_perception():
        Chem.AssignStereochemistryFrom3D(work, confId=work.GetConformer().GetId(), replaceExistingTags=True)
    invalid = _non_tetrahedral_points(work, supported_points)
    full = Chem.Mol(mol)
    for idx in invalid:
        for view in (work, full):
            atom = view.GetAtomWithIdx(idx)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.ClearProp("_CIPCode")
    assign_atrop_from_3d(work, stated_axes)
    atrop_centers = sorted(set(stated_axes) | set(assign_atrop_from_3d(work, inferred, required=False)))
    atom_centers, bond_centers, _ = graph.centres(work)
    atom_centers = [idx for idx in atom_centers if idx in supported_points]
    bond_centers = _resonance_stable_ez(work, bond_centers, stated_bonds)
    stereo = graph.label(full, work, atom_centers, bond_centers, atrop_centers)
    if apply:
        for idx in set(atom_centers) | invalid:
            mol.GetAtomWithIdx(idx).SetChiralTag(full.GetAtomWithIdx(idx).GetChiralTag())
            if idx in invalid:
                mol.GetAtomWithIdx(idx).ClearProp("_CIPCode")
        clear_ez(mol)
        graph.apply_bond_stereo(mol, work, bond_centers)
        clear_atrop(mol)  # absolute M/P labels become RDKit native atrop bond tags
        for pair, target in stereo.axes.items():
            if (tagged := atrop_tagged(mol, pair, target)) is None:
                raise ValueError(f"could not apply {target} atropisomer stereo on bond {pair}")
            mol.GetBondBetweenAtoms(*pair).SetStereo(tagged.GetBondBetweenAtoms(*pair).GetStereo())
    return stereo
