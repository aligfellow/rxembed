"""Build stereochemistry fingerprints for chirality-aware conformer selection.

Core RDKit graph stereo owns point, E/Z, and native atrop identities. xyzgraph adds planar and helical
chirality by re-perceiving bonds from coordinates, so a metallocene remains recognizable after its metal bonds
were stripped for embedding. Comparing the combined fingerprint lets selection preserve or invert either kind
without letting xyzgraph's unstable representative atoms detach a hand from a native stereo element.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdMolHash

from rxembed.metal_core import canonical_metal_graph, haptic_sites, metal_indices
from rxembed.metal_stereo import donor_classes, face_has_orientation
from rxembed.stereo import atrop_code, axis_stereo, bond_stereo, clear_atrop, clear_ez, point_stereo, stereo_from_3d

_XYZ_KEYS = (("planar", "ring"), ("helical", "atoms"))
_NATIVE = ("point", "ez", "axial")
_NATIVE_KEY = "_native"
_ATROP_MARKER_DEGREE = 2
_ISOTOPE_LIMIT = 1 << 16
_POINT_TAGS = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
_EZ_TAGS = (
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)
_ATROP_TAGS = (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW)
_POINT_INVERT = {
    _POINT_TAGS[0]: _POINT_TAGS[1],
    _POINT_TAGS[1]: _POINT_TAGS[0],
}
_EZ_INVERT = {
    _EZ_TAGS[0]: _EZ_TAGS[1],
    _EZ_TAGS[1]: _EZ_TAGS[0],
    _EZ_TAGS[2]: _EZ_TAGS[3],
    _EZ_TAGS[3]: _EZ_TAGS[2],
}
_CX_KEEP = (
    int(Chem.CXSmilesFields.CX_BOND_CFG)
    | int(Chem.CXSmilesFields.CX_BOND_ATROPISOMER)
    | int(Chem.CXSmilesFields.CX_COORDINATE_BONDS)
)
_CX_SKIP = int(Chem.CXSmilesFields.CX_ALL) & ~_CX_KEEP


_INVERT = {
    "R": "S",
    "S": "R",
    "r": "s",
    "s": "r",
    "Rₐ": "Sₐ",
    "Sₐ": "Rₐ",
    "Rₚ": "Sₚ",
    "Sₚ": "Rₚ",
    "M": "P",
    "P": "M",
    "E": "Z",
    "Z": "E",
}
# the chirality RDKit's embed cannot keep, which is what this filter is for. Point R/S and E/Z are the embed's
# own job (defined SMILES stereocentres) or labile (a protic-amine centre we do not want to lock), so
# 'preserve' leaves them free by default; opt in with {'point': 'preserve'}.
_NONGRAPH = {"planar", "axial", "helical"}


def _mode(kind, spec):
    """Resolve the per-kind chirality mode from ``spec``.

    A global string: ``'free'`` -> free all; ``'preserve'`` -> preserve only the non-graph kinds
    (planar/axial/helical), free point/ez; ``'all'`` -> preserve every kind; ``'invert'`` -> invert the
    non-graph kinds. A dict overrides per kind, falling back to its ``'default'`` or the same 'preserve' rule.
    """
    if isinstance(spec, str):
        if spec == "free":
            return "free"
        if spec == "all":
            return "preserve"
        if spec == "invert":
            return "invert" if kind in _NONGRAPH else "free"
        return "preserve" if kind in _NONGRAPH else "free"  # 'preserve' = non-graph only
    if kind in spec:
        return spec[kind]
    if "default" in spec:
        return spec["default"]
    return "preserve" if kind in _NONGRAPH else "free"


def _independent_summary(mol, summary):
    """Drop planar labels for haptic faces whose ligand graph has no orientation."""
    metals = set(metal_indices(mol))
    if not metals or not summary.get("planar"):
        return summary
    out = dict(summary)
    oriented = {}
    for metal in metals:
        donors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(metal).GetNeighbors()]
        ranks = donor_classes(mol, donors)
        for face in haptic_sites(mol, donors):
            if len(face) > 1:
                oriented[frozenset(face)] = face_has_orientation(mol, face, ranks)
    out["planar"] = [entry for entry in summary["planar"] if oriented.get(frozenset(entry.get("ring", ())), True)]
    return out


def _one_conformer(mol, conf_id):
    """Return a graph copy carrying only the selected conformer."""
    probe = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetConformer(conf_id))
    probe.RemoveAllConformers()
    probe.AddConformer(conf, assignId=True)
    return probe


def _native_hash(mol):
    """Return one canonical graph key with every native stereo hand attached to its site."""
    source = Chem.Mol(mol)
    axes = []
    for bond in source.GetBonds():
        if bond.GetStereo() not in _ATROP_TAGS:
            continue
        hand = atrop_code(source, bond)
        if hand not in {"M", "P"}:
            raise ValueError(f"could not assign M/P to atropisomer bond {bond.GetIdx()}")
        axes.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), hand))
    used_isotopes = {atom.GetIsotope() for atom in source.GetAtoms()}
    available = (isotope for isotope in range(1, _ISOTOPE_LIMIT) if isotope not in used_isotopes)
    try:
        marker_isotopes = (next(available), next(available))
    except StopIteration as exc:
        raise ValueError("no two isotope values remain for internal atropisomer markers") from exc
    clear_atrop(source)
    graph = Chem.RWMol(source)
    for left, right, hand in axes:
        graph.RemoveBond(left, right)
        marker = Chem.Atom(0)
        marker.SetIsotope(marker_isotopes[hand == "P"])
        marker.SetNoImplicit(True)
        bridge = graph.AddAtom(marker)
        graph.AddBond(left, bridge, Chem.BondType.SINGLE)
        graph.AddBond(bridge, right, Chem.BondType.SINGLE)
    graph = graph.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    return marker_isotopes, rdMolHash.MolHash(graph, rdMolHash.HashFunction.CanonicalSmiles, True, _CX_SKIP)


def _native_signature(mol):
    """Return one exact native stereo graph key plus diagnostic hand counts."""
    graph = canonical_metal_graph(mol)
    label = stereo_from_3d(graph, exclude=metal_indices(graph), apply=True)
    points = point_stereo(label)
    for atom in graph.GetAtoms():
        if atom.GetIdx() not in points:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
    hands = {
        "point": tuple(points.values()),
        "ez": tuple(bond_stereo(label).values()),
        "axial": tuple(axis_stereo(label).values()),
    }
    return {_NATIVE_KEY: _native_hash(graph)} | {kind: Counter(values) for kind, values in hands.items() if values}


def _atrop_markers(mol, marker_isotopes):
    """Return internal bridge atoms carrying absolute M/P atrop labels."""
    isotopes = set(marker_isotopes)
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in isotopes and atom.GetDegree() == _ATROP_MARKER_DEGREE
    ]


def _clear_atrop_markers(mol, marker_isotopes):
    """Collapse internal atrop bridges back to their original single bonds."""
    markers = _atrop_markers(mol, marker_isotopes)
    if not markers:
        return mol
    graph = Chem.RWMol(mol)
    for marker in markers:
        neighbors = [atom.GetIdx() for atom in graph.GetAtomWithIdx(marker).GetNeighbors()]
        if len(neighbors) != _ATROP_MARKER_DEGREE:
            raise ValueError("invalid internal atropisomer marker")
        if graph.GetBondBetweenAtoms(*neighbors) is None:
            graph.AddBond(*neighbors, Chem.BondType.SINGLE)
    for marker in sorted(markers, reverse=True):
        graph.RemoveAtom(marker)
    out = graph.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def _clear_native(mol, kind, marker_isotopes):
    """Clear one native stereo kind and return the graph."""
    if kind == "point":
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
    elif kind == "ez":
        clear_ez(mol)
    else:
        mol = _clear_atrop_markers(mol, marker_isotopes)
    return mol


def _invert_native(mol, kind, marker_isotopes):
    """Invert every element of one native stereo kind and return the graph."""
    if kind == "point":
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() in _POINT_INVERT:
                atom.SetChiralTag(_POINT_INVERT[atom.GetChiralTag()])
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
    elif kind == "ez":
        for bond in mol.GetBonds():
            if bond.GetStereo() in _EZ_INVERT:
                bond.SetStereo(_EZ_INVERT[bond.GetStereo()])
            if bond.HasProp("_CIPCode"):
                bond.ClearProp("_CIPCode")
        for bond in mol.GetBonds():
            if bond.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT):
                bond.SetBondDir(Chem.BondDir.NONE)
        Chem.SetDoubleBondNeighborDirections(mol)
    else:
        inverse = dict(zip(marker_isotopes, reversed(marker_isotopes), strict=True))
        for marker in _atrop_markers(mol, marker_isotopes):
            atom = mol.GetAtomWithIdx(marker)
            atom.SetIsotope(inverse[atom.GetIsotope()])
    return mol


def _selected_native(mol, marker_isotopes, spec, *, expected=False):
    """Apply free and reference-invert modes to a copy of a parsed native stereo graph, then hash it."""
    mol = Chem.Mol(mol)
    for kind in _NATIVE:
        mode = _mode(kind, spec)
        if mode == "free":
            mol = _clear_native(mol, kind, marker_isotopes)
        elif expected and mode == "invert":
            mol = _invert_native(mol, kind, marker_isotopes)
    return marker_isotopes, rdMolHash.MolHash(mol, rdMolHash.HashFunction.CanonicalSmiles, True, _CX_SKIP)


def _native_mismatch(sig, ref, spec):
    """Describe an exact native stereo graph mismatch, or return ``None``."""
    if _NATIVE_KEY not in sig:
        raise ValueError("native stereo signature is missing its canonical graph key")
    # Parse each serialized key once here; every selection below reuses a cheap copy of these two mols
    # instead of re-parsing SMILES text.
    params = Chem.SmilesParserParams()
    params.removeHs = False
    sig_isotopes, sig_text = sig[_NATIVE_KEY]
    ref_isotopes, ref_text = ref[_NATIVE_KEY]
    sig_mol = Chem.MolFromSmiles(sig_text, params)
    ref_mol = Chem.MolFromSmiles(ref_text, params)
    if sig_mol is None or ref_mol is None:
        raise ValueError("invalid internal native stereo key")

    actual = _selected_native(sig_mol, sig_isotopes, spec)
    expected = _selected_native(ref_mol, ref_isotopes, spec, expected=True)
    if actual == expected:
        return None
    if _selected_native(sig_mol, sig_isotopes, "free") != _selected_native(ref_mol, ref_isotopes, "free"):
        return "molecular graphs differ after stereo is removed"
    for kind in _NATIVE:
        mode = _mode(kind, spec)
        if mode == "free":
            continue
        only = {"default": "free", kind: mode}
        found = _selected_native(sig_mol, sig_isotopes, only)
        wanted = _selected_native(ref_mol, ref_isotopes, only, expected=True)
        if found == wanted:
            continue
        present_hands = sig.get(kind, Counter())
        expected_hands = ref.get(kind, Counter())
        if mode == "invert":
            expected_hands = Counter({_INVERT.get(hand, hand): count for hand, count in expected_hands.items()})
        if present_hands == expected_hands:
            return f"{kind} stereo is attached to different graph sites"
        wanted_hands = "/".join(sorted(expected_hands.elements())) or "unassigned"
        found_hands = "/".join(sorted(present_hands.elements())) or "unassigned"
        return f"{kind} expected {wanted_hands}, found {found_hands}"
    return "native stereo elements have a different relative assignment"


def signature(mol, conf_id=-1, charge=0, native=True):
    """Compute a conformer's chirality fingerprint: a multiset of handedness labels per element kind.

    RDKit supplies graph-canonical identities for point, E/Z, and stated atrop stereo. xyzgraph supplies only
    planar and helical labels, whose representative atoms and multiplicity are not stable across perception.
    Pass ``native=False`` to skip the point/E-Z/axial half when a caller only ever reads planar/helical
    (e.g. a metal `Isomer`'s stereo_ref, which `dispatch._stereo_filter` reads only for planar/helical).
    """
    try:
        import xyzgraph
        from xyzgraph.stereo import annotate_stereo
    except ImportError as exc:
        raise ImportError("signature needs xyzgraph; pip install 'rxembed[workflow]'") from exc

    probe = _one_conformer(mol, conf_id)
    sig = _native_signature(probe) if native else {}
    fd, path = tempfile.mkstemp(suffix=".xyz")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(Chem.MolToXYZBlock(mol, conf_id))
        graph = xyzgraph.build_graph(path, charge=charge, kekule=True)
        summary = _independent_summary(probe, annotate_stereo(graph))
    finally:
        os.unlink(path)
    for kind, _key in _XYZ_KEYS:
        labels = [e.get("label") for e in (summary.get(kind) or []) if e.get("label")]
        if labels:
            sig[kind] = Counter(labels)
    return sig


def mismatch(sig, ref, spec="preserve"):
    """Describe the first chirality mismatch against ``ref``, or return ``None``.

    Native point, E/Z, and atrop elements compare exact canonical identities and hands. Planar and helical
    labels compare only hands because xyzgraph's representative atoms and multiplicity are unstable.
    """
    if _NATIVE_KEY in ref and (detail := _native_mismatch(sig, ref, spec)):
        return detail
    if _NATIVE_KEY not in ref and any(kind in ref for kind in _NATIVE):
        raise ValueError("native stereo signature is missing its canonical graph key")
    for kind, refcount in ref.items():
        if kind == _NATIVE_KEY or kind in _NATIVE:
            continue
        mode = _mode(kind, spec)
        if mode == "free":
            continue
        present = set(sig.get(kind, ()))  # distinct handedness labels in the conformer
        ref_labels = set(refcount)
        flipped = {_INVERT.get(lbl, lbl) for lbl in ref_labels}
        expected = ref_labels if mode == "preserve" else flipped
        forbidden = flipped if mode == "preserve" else ref_labels
        if mode in {"preserve", "invert"}:
            # xyzgraph may omit or duplicate representatives. Reject only an observed hand that is
            # unambiguously opposite to the requested set; when both hands are present, their inversion
            # is indistinguishable at this fingerprint level and must not make preservation impossible.
            if unexpected := present & (forbidden - expected):
                found = unexpected
                return f"{kind} expected {'/'.join(sorted(expected))}, found {'/'.join(sorted(found))}"
    return None
