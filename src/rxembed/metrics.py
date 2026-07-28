"""Geometry sanity check: is a generated conformer a real molecule rather than nonsense.

Two checks, kept apart because they answer differently for a metal:

* ``bonding_ok`` — are the bond lengths sane? Covalent-radius cutoffs, heavy atoms, metals skipped (a
  dative distance is not a covalent one), and pairs the user gave a length skipped (that length is the
  request, not a broken bond). The cheap gate the FF stages use to drop garbage.
* ``connectivity`` — is it still the same molecule? Re-perceives the graph and diffs it against the
  intended one. Catches what a length check cannot: a proton transfer, a new bond at a normal 1.54 Å, a
  ligand leaving the metal — an optimiser can return a clean, low-energy geometry of a different species.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

_PT = GetPeriodicTable()
# d- and f-block metals: their coordinate/dative bonds are not governed by covalent-radius cutoffs.
_METAL_Z = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
_BREAK_RATIO = 1.5  # a bond is broken only past this multiple of its covalent-radius sum: a dissociation
# test, not a length test — loose enough that a strained bond (a 1.71 Å C=P) is never called broken.
_FORM_RATIO = 1.2  # a new bond must be at a covalent distance, not merely close.
_MIN_TOPO = 3  # a new bond needs its atoms >= this many bonds apart: a 1-2 pair is already a bond and a
# 1-3 pair's separation is set by an angle, so neither can "form" one.


def bonding_ok(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset(), constrained=()):
    """Return True if geometry-perceived connectivity matches the graph (heavy atoms).

    Three things are skipped so *valid* geometries aren't rejected:

    * pairs *inside* a frozen/reacting core (``exclude``) — a partial forming/breaking bond is held to the
      reference, not a ground-state bond;
    * any pair involving a metal, whose dative/coordinate distances covalent radii don't describe;
    * pairs in ``constrained`` (pass ``Constraints.distances``) — a pair whose separation the constraint
      system *states* has no chemistry left for a radius rule to judge. ``rx.embed('CCCl', fix={(1, 2): 2.4})``
      asks for a dissociating C-Cl; calling the result broken rejects exactly the requested geometry. Whether
      the stated window was met is a different question, answered by ``check_constraints`` / ``.measure()``.

    The exemption is per **pair**, not per atom: a bond that tore elsewhere in a molecule that also carries
    constraints is still caught, and so is a genuinely broken free-periphery bond (the ensemble may
    legitimately empty).
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    metals = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in _METAL_Z}
    exclude = set(exclude)
    stated = {frozenset(p) for p in constrained}
    rcov = {i: _PT.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy}
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    for n, i in enumerate(heavy):
        for j in heavy[n + 1 :]:
            if (i in exclude and j in exclude) or i in metals or j in metals:
                continue
            if frozenset((i, j)) in stated:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cut = rcov[i] + rcov[j]
            if frozenset((i, j)) in bonded:
                if d > bond_tol * cut or d < clash_tol * cut:  # bonded pair stretched/broken OR crushed
                    return False
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                return False
    return True


def _metal_indices(mol, extra=frozenset()):
    """Metal atom indices — by element, plus any ``extra`` (a bond-stripped carbon surrogate isn't one)."""
    return {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_Z} | set(extra)


def _perceive(mol, conf_id, charge=0, elements=None):
    """Bonds perceived from the geometry alone, as ``{frozenset((i, j))}`` — via xyzgraph.

    xyzgraph is the package's bond perceiver (it also backs the NCI and stereo paths); it reads a
    stretched partial bond in a bimetallic TS core that a flat covalent-radius cutoff calls broken.
    ``elements`` overrides the atomic numbers (the pipeline may still be carrying a metal surrogate).
    """
    import xyzgraph

    pos = mol.GetConformer(conf_id).GetPositions()
    z = elements or {}
    atoms = [
        (_PT.GetElementSymbol(z.get(a.GetIdx(), a.GetAtomicNum())), tuple(float(v) for v in pos[a.GetIdx()]))
        for a in mol.GetAtoms()
    ]
    graph = xyzgraph.build_graph(atoms, charge=charge, quick=True)  # ~2 ms; connectivity only, no bond orders
    return {frozenset(e) for e in graph.edges()}


def connectivity(mol, conf_id, *, exclude=frozenset(), metals=frozenset(), charge=0, elements=None):
    """Re-perceive the graph from the geometry and diff it against the intended one: ``(formed, broken)``.

    ``mol``'s own bond set is the reference; it never changes across the pipeline (one Mol, many
    conformers; even ``optimize`` only copies it).

    Two exclusions:

    * metal pairs — a dative bond is not covalent and no radius rule describes it (the M-donor bonds are
      stripped from the graph for the surrogate anyway); dissociation is a coordination question that
      ``coordination_changed`` answers.
    * pairs wholly inside ``exclude`` (the frozen core) — a TS's partial forming/breaking bond is held to
      the reference by design and must not be judged as a ground-state bond. A core atom's bond to a free
      atom is still checked.

    Unlike ``bonding_ok`` this sees hydrogen: a proton transfer is the most common silent change and a
    heavy-atom-only rule cannot see one.
    """
    want = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    got = _perceive(mol, conf_id, charge, elements)
    metals, exclude = _metal_indices(mol, metals), set(exclude)
    pos = mol.GetConformer(conf_id).GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    z = elements or {}

    def ratio(i, j):  # the pair's separation as a multiple of its covalent-radius sum
        r = sum(_PT.GetRcovalent(z.get(k, mol.GetAtomWithIdx(k).GetAtomicNum())) for k in (i, j))
        return float(np.linalg.norm(pos[i] - pos[j])) / r if r else float("inf")

    def judged(pair):  # a pair this check has an opinion about at all
        return not (pair & metals) and not (pair <= exclude)

    # A bond perceiver is a heuristic, wrong at the edges: xyzgraph misses a textbook 1.71 Å C=P phosphaalkene
    # (0.93x the covalent sum, clearly intact) and reads a 1,3-geminal pair at 2.02 Å as a new bond (that
    # separation is set by an angle, not contact). So a perceived change only counts when the geometry agrees:
    # broken only past `_BREAK_RATIO` of the covalent sum, formed only between atoms `_MIN_TOPO` bonds apart
    # (1-2 and 1-3 pairs are fixed by bonds/angles) that are genuinely at a covalent distance.
    def real_new(p):  # a new bond: far enough apart in the graph to be one, and actually at bonding distance
        i, j = sorted(p)
        return judged(p) and topo[i][j] >= _MIN_TOPO and ratio(i, j) < _FORM_RATIO

    def real_lost(p):  # a lost bond: genuinely dissociated, not merely strained or oddly perceived
        i, j = sorted(p)
        return judged(p) and ratio(i, j) > _BREAK_RATIO

    formed = sorted(tuple(sorted(p)) for p in got - want if real_new(p))
    broken = sorted(tuple(sorted(p)) for p in want - got if real_lost(p))
    return formed, broken


def coordination_changed(mol, conf_id, metal, donors, factor=1.3, elements=None):
    """Donors that left the metal and non-donors that joined it — the metal's own connectivity check.

    ``connectivity`` cannot judge a dative bond, so the coordination sphere is compared as a set: which
    heavy atoms sit within ``factor`` x the covalent-radius sum of the metal. A donor that left has
    dissociated; an atom that arrived is an over-bond (a second-sphere atom collapsed onto the metal),
    invisible to every clash test since metals are excluded from them.

    ``factor`` is the donor yardstick. A non-donor is judged instead by ``constraints.distance.overbond_tier``
    (shared with ``coordination.metal_overbond`` and the FF floors), deliberately tighter than ``factor``: an
    atom counts as joined only at a genuinely bonded distance, not a close contact. A chelate's bite apex
    (bonded to >= 2 donors) is skipped — the bite legitimately drags it to ~2.5 Å of the metal. A
    second-sphere atom (bonded to one donor) is judged, but loose enough not to flag a beta-agostic / CMD
    contact.

    A **declared donor is judged as a donor whatever its element** — a hydride / eta2-H2 H is a donor, and the
    surrogate leaves a monatomic one (H-, Cl-) with no bonds at all, so the graph cannot be asked. Only an
    *undeclared* H is skipped, and only as a possible joiner (see below).
    """
    from rxembed.rdkit_embed.constraints.distance import (
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
        else:
            # An H nobody declared is not judged as a joiner: an agostic / eta2-H2 contact reaches the same
            # M-H distance as a hydride bond and is not an over-bond. The screen classifies UNDECLARED atoms
            # only — a declared hydride is judged as the donor it is, one branch up.
            if za == 1:
                continue
            tier = overbond_tier(mol, donors, i)
            if tier == APEX:
                continue
            limit = NEAR_REPORT_RATIO * r_sum if tier == NEAR else r_sum + OUTER_REPORT_MARGIN
        if float(np.linalg.norm(pos[i] - pos[metal])) <= limit:
            near.add(i)
    return sorted(donors - near), sorted(near - donors)  # (left, joined)


def describe(mol, formed, broken):
    """Render a connectivity diff as chemistry (``C12-N17 formed``), never as bare indices."""
    sym = lambda i: f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"  # noqa: E731
    bits = [f"{sym(i)}-{sym(j)} formed" for i, j in formed] + [f"{sym(i)}-{sym(j)} broken" for i, j in broken]
    return ", ".join(bits)
