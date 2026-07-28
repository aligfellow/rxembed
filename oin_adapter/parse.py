"""Decode an OIN cxSMILES into the ingredients rxembed's fixed-isomer builder needs.

Perception is OIN's: ``cxsmiles_to_mol`` decodes the ``|atomProp:...|`` block into an RDKit Mol
whose metal carries the geometry tag (``OCT``, ``TET``, ...) as its ``atomNote`` and whose donors
carry their vertex slot (``s0``, ``s1``, ...). This module only READS those notes and maps the OIN
geometry tag to rxembed's polyhedron name; it never hand-parses the atomProp string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from oinsmiles.core.constants import TRANSITION_METALS_NUM
from oinsmiles.utils.cxsmiles import cxsmiles_to_mol, geo_of, strip_to_dative_smiles
from rdkit import Chem

# OIN 3-letter geometry tag -> rxembed POLYHEDRA polyhedron name.
# TPY (CN4 trigonal-pyramidal) has NO rxembed equivalent -> None (caller must fall back).
OIN_TO_RX: dict[str, str | None] = {
    "LIN": "linear",
    "TPL": "trigonal_planar",
    "SPL": "square_planar",
    "TET": "tetrahedral",
    "TPY": None,  # trigonal_pyramidal — rxembed lacks it
    "TBP": "trigonal_bipyramidal",
    "SPY": "square_pyramidal",
    "OCT": "octahedral",
    "PBP": "pentagonal_bipyramidal",
    "SQA": "square_antiprism",
}

# OIN template-vertex slot n -> rxembed vertex_dirs vertex index.
# Identity for every geometry EXCEPT square_pyramidal: OIN basal trans-pairs are (1,2)/(3,4),
# rxembed's are (1,3)/(2,4), so a direct n==n would seat a trans donor at a cis vertex.
_SPY_REMAP = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}


def slot_to_vertex(rx_geometry: str, slot: int) -> int:
    """Map an OIN slot index to the rxembed vertex_dirs vertex (identity except square_pyramidal)."""
    if rx_geometry == "square_pyramidal":
        return _SPY_REMAP.get(slot, slot)
    return slot


class Fallback(Exception):  # noqa: N818 — a routing signal, not an error condition
    """Raised when the fixed-isomer path cannot honour the encoded isomer; caller enumerates instead."""


@dataclass
class Perceived:
    """One decoded single-metal complex, ready for the fixed-isomer build."""

    mol: Chem.Mol  # decoded dative Mol (real metal, DATIVE M-donor bonds, no 3D conformer)
    plain_smiles: str  # the dative SMILES (annotation stripped) — feeds rx.metal for the enumerate path
    metal_idx: int
    rx_geometry: str  # rxembed polyhedron name
    slot_of: dict[int, int]  # donor atom index -> OIN vertex slot


def _metal_indices(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS_NUM]


def _donor_indices(mol: Chem.Mol, metal: int) -> list[int]:
    return [
        b.GetOtherAtomIdx(metal)
        for b in mol.GetAtomWithIdx(metal).GetBonds()
        if b.GetBondType() == Chem.BondType.DATIVE
    ]


def parse_cxsmiles(cx: str) -> Perceived:
    """Decode a single-metal OIN cxSMILES into a `Perceived` fixed-isomer build spec.

    Raises `Fallback` (caller routes to rx.metal enumeration) when the fixed-isomer path cannot
    honour the encoded isomer: multi-metal, an unclassified/UNK or TPY geometry tag, or ANY face
    whose atoms share one slot — an eta>=3 ring AND an eta2 side-on unit alike (OIN writes ONE slot
    for both alkene carbons). That one-to-many slot -> vertex decode needs the centroid mapping this
    thin adapter does not do. A sigma donor and a chelate (each its own slot) take the fixed path.
    """
    plain = strip_to_dative_smiles(cx)
    mol = cxsmiles_to_mol(cx)
    metals = _metal_indices(mol)
    if len(metals) != 1:
        raise Fallback(f"{len(metals)} metals; fixed-isomer path is single-metal only")
    m = metals[0]
    note = mol.GetAtomWithIdx(m).GetProp("atomNote") if mol.GetAtomWithIdx(m).HasProp("atomNote") else "UNK"
    geo3 = geo_of(note)
    rx_geom = OIN_TO_RX.get(geo3)
    if rx_geom is None:
        raise Fallback(f"geometry tag {geo3!r} has no rxembed polyhedron")

    donors = _donor_indices(mol, m)
    slot_of: dict[int, int] = {}
    for d in donors:
        dn = mol.GetAtomWithIdx(d).GetProp("atomNote") if mol.GetAtomWithIdx(d).HasProp("atomNote") else ""
        mch = re.match(r"s(\d+)([RS])?$", dn)
        if not mch:
            raise Fallback(f"donor {d} carries no slot note ({dn!r})")
        slot_of[d] = int(mch.group(1))  # group(2) CIP hand already applied by cxsmiles_to_mol

    # A HAPTIC face writes the SAME slot on every atom of the face (one coordination SITE) — an eta>=3 ring
    # and an eta2 side-on unit both do. That one-to-many slot -> vertex map needs the centroid decode; defer.
    # NB the enumerate fallback is not a good answer for these: rxembed models an eta2 face as TWO independent
    # vertices, and an eta2-COD complex (RAFJER) has been seen to hang its embed. See plan.md (eta2 centroid).
    seen: dict[int, int] = {}
    for d, s in slot_of.items():
        if s in seen.values():
            raise Fallback(f"slot {s} shared by several donors (a haptic eta2 / eta>=3 face)")
        seen[d] = s

    return Perceived(mol=mol, plain_smiles=plain, metal_idx=m, rx_geometry=rx_geom, slot_of=slot_of)
