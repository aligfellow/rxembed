"""Coordinate math, small RDKit facts, the QA result type and the package logger: the core's shared leaf.

Nothing here knows what a metal, a donor or a constraint is, so every layer reads one definition of each measure
without an import cycle.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from rdkit import Chem

logger = logging.getLogger("rxembed")  # every module logs under this name, which `set_verbose` configures
try:
    __version__ = version("rxembed")
except PackageNotFoundError:
    __version__ = "0+unknown"

CARBON_Z = 6
SP2_DEGREE = 3  # a planar sp2 centre has exactly three neighbours
DISCONNECTED = 1e6  # RDKit's topological distance for atoms in different fragments (it returns ~1e8)
_TETRAHEDRAL_DEGREE = 4  # the only degree at which a CW/CCW tag has a bond-order parity: see `bond_removal_mirrors`
# A point code's RDKit tag; no other ChiralType is a parity over the bond order.
POINT_TAGS = {"CW": Chem.ChiralType.CHI_TETRAHEDRAL_CW, "CCW": Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
TETRAHEDRAL_TAGS = frozenset(POINT_TAGS.values())
_MIRRORED = {POINT_TAGS["CW"]: POINT_TAGS["CCW"], POINT_TAGS["CCW"]: POINT_TAGS["CW"]}
STRAIGHT = 180.0  # degrees: a straight angle, the angle ceiling and one half-turn of a periodic dihedral
# What each E/Z tag says of its two stereo atoms; RDKit sets an E or Z tag's atoms to the CIP-leading substituents.
EZ_RELATION = {Chem.BondStereo.STEREOCIS: "cis", Chem.BondStereo.STEREOZ: "cis"}
EZ_RELATION |= {Chem.BondStereo.STEREOTRANS: "trans", Chem.BondStereo.STEREOE: "trans"}


def set_verbose(level: int | str = "INFO") -> None:
    """Turn on console logging at `level`."""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


def require(module: str, extra: str):
    """Import and return ``module``, or raise an ImportError naming the extra that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module} is not installed; pip install 'rxembed[{extra}]'") from exc


def require_mol(mol, verb, *, geometry=False, takes="an RDKit Mol"):
    """Return `mol` when public `verb` can read it: an RDKit Mol with atoms, and a conformer when it reads a geometry.

    The one input check of every public verb that takes a Mol; each refusal names what the verb `takes`.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError(
            f"{verb} takes {takes}, not {type(mol).__name__}; rx.parse_smiles parses a SMILES and rx.read_xyz reads "
            "an .xyz"
        )
    if not mol.GetNumAtoms():
        raise ValueError(f"{verb} got no atoms; pass a species, such as rx.parse_smiles returns")
    if geometry and not mol.GetNumConformers():
        raise ValueError(
            f"{verb} reads an existing geometry and this Mol has no conformer; use rx.embed or rx.read_xyz"
        )
    return mol


def flat_ranks(mol, *, break_ties=False, maps=False):
    """Return `mol`'s canonical atom ranks on its `flat_graph`.

    Atom maps are labels and rank nothing, unless `maps` is set by a caller that marks atoms with them.
    """
    return list(Chem.CanonicalRankAtoms(flat_graph(mol), breakTies=break_ties, includeAtomMaps=maps))


def flat_graph(mol):
    """Return a copy of `mol` with bond orders, charges, aromaticity and stereo removed and hydrogen counts kept."""
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)  # a backend graph may arrive without computed hydrogen counts
    hydrogens = [atom.GetTotalNumHs() for atom in rw.GetAtoms()]
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom, count in zip(rw.GetAtoms(), hydrogens, strict=True):
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(count)
        atom.SetNoImplicit(True)
    flat = rw.GetMol()
    flat.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(flat)
    return flat


def skeleton(mol):
    """Return `mol`'s connectivity in an RWMol: atoms labelled by element, isotope and atom map, single bonds."""
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        label = Chem.Atom(atom.GetAtomicNum())
        label.SetIsotope(atom.GetIsotope())
        label.SetAtomMapNum(atom.GetAtomMapNum())
        label.SetNoImplicit(True)
        rw.AddAtom(label)
    for bond in mol.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return rw


def without_atom_maps(mol):
    """Return `mol` without atom maps, a copy only when it has some; RDKit's ranks read maps as identity."""
    if not any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        return mol
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


@dataclass(frozen=True)
class Violation:
    """Hold one failed check: `value` breached `limit` for the atoms named."""

    kind: str
    atoms: tuple[int, ...]
    value: float = field(kw_only=True)
    limit: float = field(kw_only=True)
    detail: str = field(default="", kw_only=True)

    def __str__(self) -> str:
        """Format the violation as a readable one-liner."""
        a = "-".join(map(str, self.atoms))
        return f"[{self.kind}] atoms {a}: {self.value:.3f} vs {self.limit:.3f} {self.detail}".rstrip()


def conjugated_quartets(mol, exclude=frozenset()):
    """Yield each ``(a, c, x, s)`` quartet whose A=C-X-S dihedral measures a conjugation plane.

    That is a non-ring single bond X-C (X in N or O, C double-bonded to A) with a non-dative substituent S on X.
    `geom_check` and `ConjugationCap` both read it; `exclude` drops a quartet whose X or C it holds.
    """
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        for x_atom, c_atom in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
            if x_atom.GetAtomicNum() not in (7, 8) or c_atom.GetAtomicNum() != CARBON_Z:
                continue
            if x_atom.GetIdx() in exclude or c_atom.GetIdx() in exclude:
                continue
            dbl = [
                n
                for n in c_atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            subs = [
                n
                for n in x_atom.GetNeighbors()
                if n.GetIdx() != c_atom.GetIdx()
                and mol.GetBondBetweenAtoms(x_atom.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.DATIVE
            ]
            if not dbl or not subs:
                continue
            yield dbl[0].GetIdx(), c_atom.GetIdx(), x_atom.GetIdx(), subs[0].GetIdx()


def as_positions(mol_or_pos, conf_id: int = -1) -> np.ndarray:
    """Return (N, 3) coordinates from a Mol conformer, passing an array through unchanged."""
    if isinstance(mol_or_pos, np.ndarray):
        return mol_or_pos
    return mol_or_pos.GetConformer(conf_id).GetPositions()


def atom_label(mol, idx):
    """Return an atom's element symbol followed by its index, as messages print it: ``'C12'``."""
    return f"{mol.GetAtomWithIdx(int(idx)).GetSymbol()}{int(idx)}"


def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the a-b-c angle in degrees, or NaN when a leg has zero length."""
    u, w = a - b, c - b
    scale = np.linalg.norm(u) * np.linalg.norm(w)
    if scale == 0.0:
        return float("nan")
    cos = np.dot(u, w) / scale
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the signed dihedral p0-p1-p2-p3 in degrees, or NaN when it is undefined."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    axis = np.linalg.norm(b1)
    if axis == 0.0:
        return float("nan")
    b1 /= axis
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    eps = np.finfo(float).eps
    if np.linalg.norm(v) <= eps * np.linalg.norm(b0) or np.linalg.norm(w) <= eps * np.linalg.norm(b2):
        return float("nan")
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def mirror_tag(tag):
    """Return the tetrahedral tag naming the mirror image of `tag`; anything else passes through unchanged."""
    return _MIRRORED.get(tag, tag)


def bond_removal_mirrors(atom, partner, degrees=(_TETRAHEDRAL_DEGREE,)) -> bool:
    """Return whether removing the bond to `partner` flips `atom`'s tetrahedral tag.

    RDKit's CW/CCW is relative to bond order, and removing slot ``p`` of ``n`` costs ``n - 1 - p`` swaps. Call it while
    the bond exists; ``AddBond`` appends last, so it needs no correction. Only degrees in `degrees` answer True; a
    degree-three centre, whose lone pair is the fourth position, passes ``degrees=(3, 4)``.
    """
    partners = [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]
    if len(partners) not in degrees or partner not in partners:
        return False
    return (len(partners) - 1 - partners.index(partner)) % 2 == 1


def with_conformers(graph, mol, ids):
    """Return a copy of `graph` carrying the conformers `ids` of `mol`, which shares its atom order, ids kept."""
    out = Chem.Mol(graph)
    out.RemoveAllConformers()
    keep = {int(cid) for cid in ids}
    for conf in mol.GetConformers():
        if conf.GetId() in keep:
            out.AddConformer(Chem.Conformer(conf), assignId=False)
    return out


def remove_bond(rw, i, j) -> None:
    """Remove bond `i`-`j` from `rw`, keeping what degree-four tetrahedral and E/Z tags mean.

    RDKit's `RemoveBond` can invert a centre and empty the stereo atoms of a double bond; here such an E/Z is
    re-referenced to the end's other substituent, or dropped where there is none.
    """
    referenced = False
    for a, other in ((int(i), int(j)), (int(j), int(i))):
        atom = rw.GetAtomWithIdx(a)
        if bond_removal_mirrors(atom, other):
            atom.SetChiralTag(mirror_tag(atom.GetChiralTag()))
        for bond in atom.GetBonds():
            if bond.GetStereo() not in EZ_RELATION:
                continue
            refs = list(bond.GetStereoAtoms())
            side = 0 if bond.GetBeginAtomIdx() == a else 1
            if len(refs) != 2 or refs[side] != other:  # noqa: PLR2004
                continue
            spare = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in (other, bond.GetOtherAtomIdx(a))]
            refs[side] = spare[0] if spare else other
            bond.SetStereoAtoms(*refs)
            turned = (
                Chem.BondStereo.STEREOTRANS if EZ_RELATION[bond.GetStereo()] == "cis" else Chem.BondStereo.STEREOCIS
            )
            bond.SetStereo(turned if spare else Chem.BondStereo.STEREONONE)
            referenced = True
    rw.RemoveBond(int(i), int(j))
    if referenced:
        Chem.SetDoubleBondNeighborDirections(rw)  # the SMILES writer reads E/Z from bond directions


def without_zero_bonds(mol):
    """Return `mol` without its zero-order contacts: the graph an identity ranking reads.

    A zero-order contact is not constitution, yet RDKit's ranks, CIP labeller and distance matrix count it as an edge.
    Everything else is kept; `mol` itself is returned when it has none, so treat the result as read-only.
    """
    zero = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.ZERO]
    if not zero:
        return mol
    rw = Chem.RWMol(mol)
    for i, j in zero:
        remove_bond(rw, i, j)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


@contextmanager
def tetrahedral_perception():
    """Read only tetrahedral centres from 3D, as RDKit 2026.03 does; 2026.09 also tags distorted ones as TBP."""
    previous = Chem.GetAllowNontetrahedralChirality()
    Chem.SetAllowNontetrahedralChirality(False)
    try:
        yield
    finally:
        Chem.SetAllowNontetrahedralChirality(previous)
