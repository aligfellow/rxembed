"""Build a Constraints from simple, user-facing specs.

Atom references are an **index** or a **SMARTS** string (first matching atom). `freeze` locks the
given atoms to the *input* geometry (the racerts principle — only needs the atoms, the rest relaxes).
`distances`/`angles` take a single value or a `(lo, hi)` window.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from rxembed.log import logger

from .base import Constraints, add_distance, add_pairwise_shape

_MIN_SHAPE_ATOMS = 3  # below this a core has only a distance to pin, not a 3D shape


def resolve_atom(mol, ref):
    """Resolve an atom reference to an index (an int index, or a SMARTS matching exactly one atom)."""
    if isinstance(ref, (int, np.integer)):
        return int(ref)
    hit = mol.GetSubstructMatch(Chem.MolFromSmarts(ref))
    if not hit:
        raise ValueError(f"SMARTS {ref!r} matched nothing")
    return hit[0]


def match(mol, smarts):
    """All atom indices of the first SMARTS match (for picking several reactive atoms at once)."""
    hit = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    if not hit:
        raise ValueError(f"SMARTS {smarts!r} matched nothing")
    return hit


def from_template(new_mol, template_mol, smarts, anchor=None, pad=0.1):
    """Pin ``new_mol``'s reacting core onto a template TS geometry.

    `smarts` is matched in `new_mol`; the corresponding template atoms come from `anchor`:

    - **a dict ``{map_number: template_index}``** — pair each map-numbered SMARTS atom (``'[#6:1]~[Cl:2]'``)
      to a template atom. **The robust way**: the pairing is by *your* labels, so a symmetric/palindromic
      core can't silently flip (see below). Pins only the mapped atoms.
    - **a list** — template indices in the SMARTS' query-atom order (``[t0, t1, …]``), positional.
    - **``None``** — match the same `smarts` in the template too (k-th new atom ↔ k-th template atom).

    `GetSubstructMatch` returns atoms in query order, so positional correspondence needs no MCS — **but only
    up to automorphism**: for a symmetric ``match=`` (a ring, equal branches, ``[F,Cl]`` at both ends) RDKit
    picks an arbitrary automorph by *target* atom order (how the SMILES was written), so positional `new[k]`
    and `template[k]` can pair *different* atoms with no error. Use a **specific** SMARTS or the **map-number
    dict** for any symmetric core. A **TS template's reacting bonds often don't perceive** (a forming/breaking
    bond sits past the covalent cutoff), so SMARTS-on-the-template fails on the very cores this is for — hence
    `anchor` (you read the few core indices off your TS file).

    Returns ``(Constraints, ref)``: the Constraints carry the **all-pairwise template distances** among the
    matched core atoms (a frame-independent SHAPE bias for the embed) plus those atoms as `frozen`; `ref` is
    ``{new_idx: (x,y,z)}`` the template coordinate each matched atom is rigid-grafted onto after the embed.
    (Shape-bias + exact graft + a fixed-point relax — the same machinery as a frozen TS core — so it handles
    a multi-fragment substrate natively, where an RDKit coordMap silently fails.)
    """
    q = Chem.MolFromSmarts(smarts)
    if q is None:
        raise ValueError(f"template match={smarts!r} is not a valid SMARTS")
    nmatch = new_mol.GetSubstructMatches(q, uniquify=False)
    if not nmatch:
        raise ValueError(f"template match={smarts!r} matched nothing in the embedded molecule")
    if len(nmatch) > 1:
        logger.warning(
            "template: match=%s hit %d sites in the molecule — using the first %s; for a symmetric "
            "core pass map numbers ('[#6:1]') so the pairing can't flip",
            smarts,
            len(nmatch),
            nmatch[0],
        )
    nm = nmatch[0]
    nt = template_mol.GetNumAtoms()

    def _checked(idxs):
        for t in idxs:
            if not (0 <= t < nt):
                raise ValueError(f"anchor template index {t} is out of range (template has {nt} atoms, 0..{nt - 1})")
        return list(idxs)

    if isinstance(anchor, dict):  # {map number: template index} — automorphism-proof
        core_new, core_tmpl = [], []
        for k in range(q.GetNumAtoms()):
            mnum = q.GetAtomWithIdx(k).GetAtomMapNum()
            if mnum in anchor:
                core_new.append(nm[k])
                core_tmpl.append(int(anchor[mnum]))
        if not core_new:
            raise ValueError(
                f"anchor dict keyed by {sorted(anchor)} matched no map numbers in match={smarts!r} "
                f"— label the core atoms like '[#6:1]~[Cl:2]' and key anchor by those numbers"
            )
        _checked(core_tmpl)
    elif anchor is not None:
        core_tmpl = [int(a) for a in anchor]
        if len(core_tmpl) != len(nm):
            raise ValueError(
                f"anchor= has {len(core_tmpl)} atoms but match={smarts!r} matches {len(nm)} — they "
                f"must correspond one-to-one in the SMARTS' query-atom order (or use a "
                f"{{map_number: template_index}} dict to pin a subset)"
            )
        _checked(core_tmpl)
        core_new = list(nm)
    else:
        tmatch = template_mol.GetSubstructMatches(q, uniquify=False)
        if not tmatch:
            raise ValueError(
                f"template match={smarts!r} matched nothing in the template geometry — a TS's "
                f"forming/breaking bonds may not perceive; pass anchor=<the template's core atom "
                f"indices, in SMARTS query-atom order, or a {{map_number: template_index}} dict>"
            )
        core_tmpl, core_new = list(tmatch[0]), list(nm)
    tpos = template_mol.GetConformer().GetPositions()
    cons = Constraints()
    ref = {core_new[k]: tuple(tpos[core_tmpl[k]]) for k in range(len(core_new))}  # new atom -> template coord
    add_pairwise_shape(cons, core_new, ref, pad)  # all-pairwise template distances -> core SHAPE
    cons.frozen |= set(core_new)
    if len(core_new) < _MIN_SHAPE_ATOMS:
        logger.warning(
            "template: core has %d atom(s) — only its internal distance is fixed, not an absolute "
            "pose (a rigid graft needs >=3 atoms to orient)",
            len(core_new),
        )
    else:
        logger.info(
            "template: matched a %d-atom core; pinning it to the template geometry, conf-searching the rest",
            len(core_new),
        )
    return cons, ref


def _window(val, pad):
    return tuple(val) if isinstance(val, (tuple, list)) else (val - pad, val + pad)


def from_spec(mol, *, freeze=None, distances=None, angles=None, planes=None, has_geometry=False):
    """Constraints from `freeze` / `distances` / `angles` / `planes`. Atoms by index or SMARTS."""
    cons = Constraints()
    if freeze:
        if not has_geometry:
            raise ValueError("freeze= needs an input geometry (an .xyz, or a Mol with a conformer)")
        idx = [resolve_atom(mol, a) for a in freeze]
        add_pairwise_shape(cons, idx, mol.GetConformer().GetPositions(), 0.05)  # pin to the input geometry
        cons.frozen |= set(idx)
    for key, val in (distances or {}).items():
        a, b = (resolve_atom(mol, k) for k in key)
        lo, hi = _window(val, 0.05)
        add_distance(cons.distances, a, b, lo, hi)
    for key, val in (angles or {}).items():
        a, b, c = (resolve_atom(mol, k) for k in key)
        cons.angles[(a, b, c)] = _window(val, 3.0)
    cons.planes.extend(planes or [])
    if cons.is_constrained:
        logger.debug(
            "constraints: %d distance, %d angle, %d plane, %d frozen",
            len(cons.distances),
            len(cons.angles),
            len(cons.planes),
            len(cons.frozen),
        )
    return cons
