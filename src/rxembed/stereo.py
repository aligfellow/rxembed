"""Stereochemistry fingerprints for chirality-aware conformer selection (xyzgraph ``--stereo``).

A conformer's chirality is a *fingerprint* of every stereo element xyzgraph detects — point (R/S), E/Z,
axial (Rₐ/Sₐ), **planar (a metallocene's planar chirality)**, and helical. Comparing fingerprints lets us
keep / drop / invert conformers by handedness on any one element (or all), which is how we sample a metal
centre's geometry while holding a spectator ferrocene's planar chirality fixed.

Bonds are re-perceived from the **coordinates** (not the Mol graph), so a metallocene is recognised even on
a surrogate Mol whose metal-donor bonds were stripped — the only requirement is real element symbols.
"""

from __future__ import annotations

import os
import tempfile

_KEYS = [("point", "atom"), ("ez", "bond"), ("axial", "atoms"), ("planar", "ring"), ("helical", "atoms")]


def _xyz_block(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    out = [str(mol.GetNumAtoms()), ""]
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        out.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(out) + "\n"


_INVERT = {"R": "S", "S": "R", "Rₐ": "Sₐ", "Sₐ": "Rₐ", "Rₚ": "Sₚ", "Sₚ": "Rₚ", "M": "P", "P": "M", "E": "Z", "Z": "E"}
# the chirality RDKit's embed cannot keep — what this filter is *for*. Point R/S and E/Z are the embed's
# own job (defined SMILES stereocentres) or labile (a protic-amine centre we do NOT want to lock), so
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


def signature(mol, conf_id=-1, charge=0):
    """Compute a conformer's chirality fingerprint: a multiset of handedness labels per element kind.

    ``{kind: Counter({label: count})}`` over ``point`` (R/S), ``ez``, ``axial`` (Rₐ/Sₐ), ``planar`` (a
    metallocene's Rₚ/Sₚ), ``helical`` (M/P) — via xyzgraph (bonds re-perceived from the geometry, so a
    metallocene is seen even on a bond-stripped surrogate). We key by *kind*, not atom indices, because
    xyzgraph picks different representative atoms per re-perception — the handedness label is the stable,
    physical quantity, the atom set is not.
    """
    from collections import Counter

    import xyzgraph
    from xyzgraph.stereo import annotate_stereo

    fd, path = tempfile.mkstemp(suffix=".xyz")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_xyz_block(mol, conf_id))
        summary = annotate_stereo(xyzgraph.build_graph(path, charge=charge, kekule=True))
    finally:
        os.unlink(path)
    sig = {}
    for kind, _key in _KEYS:
        labels = [e.get("label") for e in (summary.get(kind) or []) if e.get("label")]
        if labels:
            sig[kind] = Counter(labels)
    return sig


def passes(sig, ref, spec="preserve"):
    """Return whether fingerprint ``sig`` satisfies the chirality ``spec`` against reference ``ref``.

    Judged on **handedness, not count** — xyzgraph's per-conformer perception is count-unstable (a
    metallocene may surface as ``{}``, ``{Sₚ:1}`` or ``{Sₚ:2}`` for the *same* configuration), but the
    *label* is stable. So per kind: ``'preserve'`` rejects only if the conformer carries the **flipped**
    label of a reference element (an actual inversion); ``'invert'`` rejects if it carries any reference
    label **unchanged**; ``'free'`` is unconstrained. `spec` is a global string or a dict
    ``{kind: mode, 'default': mode}``. A conformer passes only if every reference kind is satisfied.
    """
    for kind, refcount in ref.items():
        mode = _mode(kind, spec)
        if mode == "free":
            continue
        present = set(sig.get(kind, ()))  # distinct handedness labels in the conformer
        ref_labels = set(refcount)
        if mode == "preserve" and present & {_INVERT.get(lbl, lbl) for lbl in ref_labels}:
            return False  # a flipped element is present -> rejected
        if mode == "invert" and present & ref_labels:
            return False  # an un-flipped (original) element is present
    return True


def kinds(sig):
    """Return the chirality element kinds present in ``sig`` (``'planar'``, ``'point'``, ...).

    Lets a caller decide if chirality control is even relevant (no planar/axial/helical and SMILES-defined
    point -> the embed handles it).
    """
    return set(sig)


def relevant(ref):
    """Return whether ``ref`` carries chirality the embed would drop (planar/axial/helical).

    That is whether the auto-default should engage; a molecule with only point/EZ stereo needs no filtering
    (RDKit keeps it).
    """
    return bool(_NONGRAPH & set(ref))
