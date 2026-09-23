"""Geometry sanity check: is a generated conformer a real molecule rather than nonsense.

Two checks, kept apart because they answer differently for a metal:

* ``bonding_ok``: are the bond lengths sane? Covalent-radius cutoffs, heavy atoms, metals skipped (a
  dative distance is not a covalent one), and pairs the user gave a length skipped (that length is the
  request, not a broken bond). The cheap gate the FF stages use to drop garbage. It is the CORE's
  (``rxembed.relax``), which accepts its own relax on it; re-exported here so the two sanity checks
  are still read side by side.
* ``connectivity``: is it still the same molecule? Re-perceives the graph and diffs it against the
  intended one. Catches what a length check cannot: a proton transfer, a new bond at a normal 1.54 Å, a
  ligand leaving the metal. An optimiser can return a clean, low-energy geometry of a different species.
  Pipeline-only: it needs xyzgraph perception, which the core's numpy + rdkit floor excludes.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from rxembed.metal_core import metal_indices
from rxembed.relax import bonding_ok

__all__ = ["bonding_ok", "connectivity", "coordination_changed", "describe"]

_PT = GetPeriodicTable()
_BREAK_RATIO = 1.5  # a bond is broken only past this multiple of its covalent-radius sum: a dissociation
# test, not a length test, loose enough that a strained bond (a 1.71 Å C=P) is never called broken.
_FORM_RATIO = 1.2  # a new bond must be at a covalent distance, not merely close.
_MIN_TOPO = 3  # a new heavy-atom bond needs its atoms >= this many bonds apart. A terminal H is checked at
# topological distance 2 as well: collapse across an angle is a proton transfer, not a harmless short 1-3.
_ANGLE_PATH = 2  # graph distance between a 1-3 angle's endpoints


def _perceive(mol, conf_id, charge=0, elements=None):
    """Bonds perceived from the geometry alone, as ``{frozenset((i, j))}``, via xyzgraph.

    xyzgraph is the package's bond perceiver (it also backs the NCI and stereo paths); it reads a
    stretched partial bond in a bimetallic TS core that a flat covalent-radius cutoff calls broken.
    ``elements`` overrides the atomic numbers (the pipeline may still be carrying a metal surrogate).
    """
    try:
        import xyzgraph
    except ImportError as exc:
        raise ImportError("_perceive needs xyzgraph; pip install 'rxembed[workflow]'") from exc

    pos = mol.GetConformer(conf_id).GetPositions()
    z = elements or {}
    atoms = [
        (
            _PT.GetElementSymbol(z.get(a.GetIdx(), a.GetAtomicNum())),
            (float(pos[a.GetIdx()][0]), float(pos[a.GetIdx()][1]), float(pos[a.GetIdx()][2])),
        )
        for a in mol.GetAtoms()
    ]
    graph = xyzgraph.build_graph(atoms, charge=charge, quick=True)  # connectivity only, no bond orders
    return {frozenset(e) for e in graph.edges()}


def connectivity(mol, conf_id, *, exclude=frozenset(), metals=frozenset(), charge=0, elements=None):
    """Re-perceive the graph from the geometry and diff it against the intended one: ``(formed, broken)``.

    ``mol``'s own bond set is the reference and never changes across the pipeline. Two exclusions: metal
    pairs, since a dative bond is not covalent and no radius rule describes it (``coordination_changed``
    answers that question instead); and pairs wholly inside ``exclude``, a TS's partial bonds being held to
    the reference by design. A core atom's bond to a free atom is still checked.

    Unlike ``bonding_ok`` this sees hydrogen: a proton transfer is the most common silent change.
    """
    # A ZERO bond is an explicit non-covalent annotation (for example the H in an O-H~O bridge),
    # not a connectivity edge for the geometry perceiver to preserve.
    want = {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
        if b.GetBondType() != Chem.BondType.ZERO
    }
    got = _perceive(mol, conf_id, charge, elements)
    metals, exclude = set(metal_indices(mol)) | set(metals), set(exclude)
    pos = mol.GetConformer(conf_id).GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    z = elements or {}

    def ratio(i, j):  # the pair's separation as a multiple of its covalent-radius sum
        r = sum(_PT.GetRcovalent(z.get(k, mol.GetAtomWithIdx(k).GetAtomicNum())) for k in (i, j))
        return float(np.linalg.norm(pos[i] - pos[j])) / r if r else float("inf")

    def judged(pair):  # a pair this check has an opinion about at all
        return not (pair & metals) and not (pair <= exclude)

    # Require distance evidence before interpreting a heuristic perception mismatch as a reaction.
    def real_new(p):  # a new bond: far enough apart in the graph to be one, and actually at bonding distance
        i, j = sorted(p)
        hydrogens = [k for k in p if mol.GetAtomWithIdx(k).GetAtomicNum() == 1]
        one_terminal_h = len(hydrogens) == 1 and mol.GetAtomWithIdx(hydrogens[0]).GetDegree() == 1
        separated = topo[i][j] >= _MIN_TOPO or (topo[i][j] == _ANGLE_PATH and one_terminal_h)
        return judged(p) and separated and ratio(i, j) < _FORM_RATIO

    def real_lost(p):  # a lost bond: genuinely dissociated, not merely strained or oddly perceived
        i, j = sorted(p)
        return judged(p) and ratio(i, j) > _BREAK_RATIO

    formed = sorted(tuple(sorted(p)) for p in got - want if real_new(p))
    broken = sorted(tuple(sorted(p)) for p in want - got if real_lost(p))
    return formed, broken


def coordination_changed(mol, conf_id, metal, donors, factor=1.3, elements=None, exclude=frozenset(), constrained=()):
    """Donors that left the metal and non-donors that joined it: the metal's own connectivity check.

    ``connectivity`` cannot judge a dative bond, so the sphere is compared as a set: a donor within
    ``factor`` x the covalent-radius sum of the metal has stayed, and a non-donor within that reach has
    joined (judged more tightly, by ``overbond_tier``, so a beta-agostic contact still passes). A pair
    wholly inside ``exclude`` is a reacting core and is not judged; a donor's window in ``constrained``
    outranks the generic cutoff.
    """
    from rxembed.metal_distance import (
        APEX,
        NEAR,
        NEAR_REPORT_RATIO,
        OUTER_REPORT_MARGIN,
        overbond_tier,
    )

    pos = mol.GetConformer(conf_id).GetPositions()
    z = elements or {}
    zm = z.get(metal, mol.GetAtomWithIdx(metal).GetAtomicNum())
    rm = _PT.GetRcovalent(zm)
    donors = set(donors)
    near = set()
    for a in mol.GetAtoms():
        i = a.GetIdx()
        za = z.get(i, a.GetAtomicNum())
        if i == metal:
            continue
        r_sum = rm + _PT.GetRcovalent(za)
        if i in donors:
            limit = factor * r_sum  # a donor: did it leave?
            if window := getattr(constrained, "get", lambda _pair: None)((min(metal, i), max(metal, i))):
                limit = max(limit, window[1])  # a stated M-L window outranks the generic radius yardstick
        else:
            # An H nobody declared is not judged as a joiner: an agostic / eta2-H2 contact reaches the same
            # M-H distance as a hydride bond and is not an over-bond. The screen classifies only atoms nobody
            # declared; a declared hydride is judged as the donor it is, one branch up.
            if za == 1:
                continue
            tier = overbond_tier(mol, donors, i)
            if tier == APEX:
                continue
            limit = NEAR_REPORT_RATIO * r_sum if tier == NEAR else r_sum + OUTER_REPORT_MARGIN
        if float(np.linalg.norm(pos[i] - pos[metal])) <= limit:
            near.add(i)
    left, joined = donors - near, near - donors
    if metal in exclude:  # a pair wholly inside a reacting core is intentionally not ground-state coordination
        left -= set(exclude)
        joined -= set(exclude)
    return sorted(left), sorted(joined)


def describe(mol, formed, broken):
    """Render a connectivity diff as chemistry (``C12-N17 formed``), never as bare indices."""
    sym = lambda i: f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"  # noqa: E731
    bits = [f"{sym(i)}-{sym(j)} formed" for i, j in formed] + [f"{sym(i)}-{sym(j)} broken" for i, j in broken]
    return ", ".join(bits)
