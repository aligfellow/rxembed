"""Read whether a geometry still shows its species: the bonds it makes, the stereo it keeps and its hands.

`changes` reports the bonds and stereo of one conformer: bonds as `read_xyz` perceives them, stereo as `reading.read`
judges it, with no tolerance of its own. Planar and helical hands, which no tag specifies, are read by xyzgraph
(`signature`) and checked against a `stereo=` request.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem

from rxembed.geometry.constraints import DIST_ATOMS, ML_WINDOW_TOL, within_window
from rxembed.graph import haptic_sites, metal_indices
from rxembed.identity.reading import read
from rxembed.identity.smiles_io import mol_from_species
from rxembed.metal.distances import OUTER, ionic_contacts, overbond_tier
from rxembed.perception.readers import read_geometry, xyz_atoms
from rxembed.stereo import without_stale_point_tags
from rxembed.utils import as_positions, atom_label, require_mol, with_conformers

__all__ = ["Changes", "bond_changes", "changes", "specified_like", "stereo_change"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Changes:
    """Hold how one conformer departs from its species: the bonds it forms and breaks, and the stereo it contradicts."""

    formed: tuple[tuple[int, int], ...] = ()
    broken: tuple[tuple[int, int], ...] = ()
    stereo: str | None = None  # the `problem` `reading.read` names; None when kept, or not read (`bond_changes`)

    @property
    def unchanged(self):
        """Return True when the conformer forms and breaks no bond and contradicts no specified stereo."""
        return not (self.formed or self.broken or self.stereo)

    def describe(self, mol):
        """Render the changes as chemistry on `mol` (``C12-N17 formed``), never as bare indices."""
        bits = [f"{atom_label(mol, i)}-{atom_label(mol, j)} formed" for i, j in self.formed]
        bits += [f"{atom_label(mol, i)}-{atom_label(mol, j)} broken" for i, j in self.broken]
        return ", ".join(bits + ([f"stereo {self.stereo}"] if self.stereo else []))


def changes(mol, conf_id=-1, reference=None):
    """Return how conformer `conf_id` of species Mol `mol` departs from the species `mol` is.

    Bonds are `bond_changes`; stereo is `stereo_change`, against what `reference` shows (`specified_like`) if given.
    """
    require_mol(mol, "rx.changes", geometry=True)
    one = without_stale_point_tags(Chem.Mol(mol, False, mol.GetConformer(conf_id).GetId()))
    specified = None if reference is None else specified_like(one, reference)
    return replace(bond_changes(mol, conf_id), stereo=stereo_change(one, -1, specified))


def bond_changes(mol, conf_id=-1, cons=None):
    """Return the bonds conformer `conf_id` of `mol` forms and breaks against `mol`'s graph; stereo is not read.

    Every bond counts, metal dative bonds included; a zero-order contact is none. A metal changes when a donor site
    leaves or a third-sphere atom arrives: a site the metal still binds through any atom is kept however far it slipped,
    and an atom bonded to a donor coming close is a geometry question, not a bond. A bond stays while inside its
    distance window. Pairs `cons` holds (frozen, fixed or under ``constrain=``) are not judged, so a TS keeps its
    forming and breaking bonds.
    """
    want, got = _pairs(mol), _pairs(read_geometry(mol, conf_id))
    formed, broken = got - want, want - got
    if cons is not None:
        held = {pair for pair in formed | broken if pair <= cons.frozen}
        held |= {frozenset(atoms) for atoms in (*cons.fixed, *cons.contacts.distances) if len(atoms) == DIST_ATOMS}
        formed, broken = formed - held, broken - held
    if broken:
        broken -= _in_window(mol, conf_id, broken, cons)
    metals = set(metal_indices(mol))
    for metal in metals:
        wanted, found = ({a for pair in pairs if metal in pair for a in pair} - metals for pairs in (want, got))
        formed -= {frozenset((metal, a)) for a in found - wanted if overbond_tier(mol, wanted, a) != OUTER}
        for site in haptic_sites(mol, wanted):
            if found.intersection(site):
                broken -= {frozenset((metal, atom)) for atom in site}
    return Changes(
        formed=tuple(sorted(tuple(sorted(pair)) for pair in formed)),
        broken=tuple(sorted(tuple(sorted(pair)) for pair in broken)),
    )


def stereo_change(mol, conf_id=-1, specified=None):
    """Return how conformer `conf_id` of `mol` contradicts the specified stereo, or None.

    The stereo is `mol`'s own, or that of `specified`, a Mol in `mol`'s atom order such as `specified_like` returns.
    """
    specified = mol if specified is None else specified
    return read(with_conformers(specified, mol, [mol.GetConformer(conf_id).GetId()])).problem


def specified_like(mol, reference):
    """Return a copy of `mol` specifying the species `reference` shows on its graph, with the reference's conformer.

    `reference` is a Mol, positions or an ``.xyz`` path (its first frame) in `mol`'s atom order. The species is read
    from the reference: its ligand stereo from 3D, every element `mol` specifies included, and each bound metal's
    arrangement.
    """
    if isinstance(reference, str):
        path, reference = reference, Chem.MolFromXYZFile(reference)  # coordinates only, the first frame
        if reference is None:
            raise ValueError(f"read no geometry from {path}; pass an existing .xyz in mol's atom order")
    positions = np.asarray(as_positions(reference), float)
    if positions.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"the reference has {len(positions)} atoms, mol {mol.GetNumAtoms()}; give it in mol's order")
    ref = Chem.Mol(mol)
    ref.RemoveAllConformers()
    conf = Chem.Conformer(ref.GetNumAtoms())
    conf.SetPositions(positions)
    ref.AddConformer(conf)
    for atom in ref.GetAtoms():
        atom.ClearProp("atomNote")  # a note would state an arrangement instead of reading it
    return mol_from_species(read(ref).species)


def _in_window(mol, conf_id, pairs, cons):
    """Return the `pairs` of conformer `conf_id` inside their distance window, as the window gate reads it.

    A pair's window is the one `cons` gives, else an alkali ion's contact, which the bond reader does not draw.
    """
    windows = {frozenset(key): window for key, window in ionic_contacts(mol).distances.items()}
    if cons is not None:
        windows |= {frozenset(key): window for key, window in cons.distances.items()}
    pos = mol.GetConformer(conf_id).GetPositions()
    return {
        pair
        for pair in pairs
        if pair in windows
        and within_window(np.linalg.norm(np.subtract(*pos[sorted(pair)])), windows[pair], ML_WINDOW_TOL)
    }


def _pairs(mol):
    """Return the bonded atom pairs of `mol`, a zero-order contact left out."""
    return {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
        if bond.GetBondType() != Chem.BondType.ZERO
    }


_XYZ_KEYS = (("planar", "ring"), ("helical", "atoms"))
_INVERT = {"Rₚ": "Sₚ", "Sₚ": "Rₚ", "M": "P", "P": "M"}


def _independent_summary(mol, summary):
    """Drop the planar labels xyzgraph puts on a metal or a haptic face: a face's hand is its specified winding."""
    metals = metal_indices(mol)
    faces = {
        frozenset(face)
        for metal in metals
        for face in haptic_sites(mol, [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()])
    }
    planar = [
        entry
        for entry in summary.get("planar") or ()
        if frozenset(entry.get("ring", ())) not in faces and not set(metals).intersection(entry.get("ring", ()))
    ]
    return {**summary, "planar": planar}


def signature(mol, conf_id=-1):
    """Return a conformer's planar and helical hands, each kind a multiset of labels, read at the Mol's charge.

    xyzgraph's representative atoms and multiplicity are not stable across perception, so only the hands count.
    """
    import xyzgraph
    from xyzgraph.stereo import annotate_stereo

    graph = xyzgraph.build_graph(xyz_atoms(mol, conf_id), charge=Chem.GetFormalCharge(mol), kekule=True)
    summary = _independent_summary(Chem.Mol(mol, confId=conf_id), annotate_stereo(graph))
    sig = {}
    for kind, _key in _XYZ_KEYS:
        labels = [e.get("label") for e in (summary.get(kind) or []) if e.get("label")]
        if labels:
            sig[kind] = Counter(labels)
    return sig


def mismatch(sig, ref, request):
    """Describe the first hand `sig` loses against `ref` under `request`, or None.

    xyzgraph may omit or repeat representatives, and a reference with both hands reads as its own mirror, so only a hand
    the request forbids and does not also expect fails.
    """
    for kind, refcount in ref.items():
        mode = request.kept(kind)
        if mode == "free":
            continue
        present = set(sig.get(kind, ()))
        ref_labels = set(refcount)
        flipped = {_INVERT.get(lbl, lbl) for lbl in ref_labels}
        expected = ref_labels if mode == "preserve" else flipped
        forbidden = flipped if mode == "preserve" else ref_labels
        if mode in {"preserve", "invert"}:
            if unexpected := present & (forbidden - expected):
                found = unexpected
                return f"{kind} expected {'/'.join(sorted(expected))}, found {'/'.join(sorted(found))}"
    return None


def stereo_filter(source, stereo):
    """Return the chirality filter ``(request, reference signature)`` an embedded ensemble keeps, or None.

    Only the source Mol's conformer is a reference: a SMILES, a stereo-free input or ``stereo='free'`` keeps none. The
    filter keeps the planar and helical hands no tag or note covers. A failure to read them raises.
    """
    if stereo.given == "free" or not source.GetNumConformers():
        return None
    ref = signature(source)
    return (stereo, ref) if ref else None
