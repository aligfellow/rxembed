"""Judge whether a conformer is physically sound in the free periphery that constraint windows miss.

A perfect frozen core and perfect constraint distances can coexist with a puckered aromatic ring, a twisted amide, a
stretched X-H or two atoms on top of each other. `check` reads one conformer into a `GeometryReport`, `ok` when it has
no defect:

    rep = geom_check.check(mol, conf_id, reference=ts_mol, constraints=rx.Constraints(frozen=core))
    print(rep.summary())     # a readable summary of any failed check

Stereo is read by `changes`, never here.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Numerics import rdAlignment

from rxembed.geometry.constraints import (
    DIHEDRAL_ATOMS,
    DIST_ATOMS,
    Constraints,
    canonical_key,
    constraint_value,
    donor_directions,
    within_window,
)
from rxembed.geometry.relax import bond_violations
from rxembed.graph import COORDINATION_METALS, haptic_sites, metal_indices, no_aromatic_axis
from rxembed.metal.orientation import stripped_hybridisation
from rxembed.metal.shape import declared_donors, donor_orientation, metal_overbond, spheres
from rxembed.perception.changes import specified_like, stereo_change
from rxembed.utils import (
    CARBON_Z,
    SP2_DEGREE,
    Violation,
    as_positions,
    atom_label,
    bond_angle,
    conjugated_quartets,
    dihedral_angle,
    require_mol,
)

_PT = Chem.GetPeriodicTable()
_FLEX_CONJ = 60.0  # deg: the conjugation-dihedral window of a metal-coordinated or side-on π atom
_FUSE_RATIO = 1.0  # x the covalent sum: a non-bonded 1-3 pair this close has fused (`over_compression`)
# steric floors: heavy pairs at a fraction of the vdW sum, H...H a distance, X-H a covalent multiple
_HEAVY_VDW, _HH_FLOOR, _XH_COV = 0.7, 1.5, 1.1  # fraction, A, multiple
_SP2_OOP, _RING_RMS = 0.15, 0.10  # A: an sp2 carbon's offset from its neighbours' plane, an aromatic ring's RMS
_CONJ_TWIST = 30.0  # deg: the A=C-X-S dihedral's deviation from planar (`conjugation`)
_CORE_RMSD = 0.05  # A: a frozen core's RMSD from its reference after superposition (`frozen_core`)
_WINDOW_SLACK = {"distance": 0.15, "angle": 5.0, "dihedral": 5.0}  # A, deg, deg past a window (`check_constraints`)
# perceived spheres only: a side-on pair's distance spread and longest bond, not fitted
_SIDEON_SYM = 0.5  # A
_SIDEON_MAX = 2.6  # A


@dataclass
class GeometryReport:
    """Result of ``check()``: every violation read, and the kinds among them that only inform.

    The conformer passes (`ok`) when it has no `defects`: violations of a kind outside `diagnostic`.
    """

    violations: list[Violation] = field(default_factory=list)
    shape: str | None = None  # each centre's reading of its polyhedron (`shape_report`); None without a species
    diagnostic: frozenset = frozenset()

    @property
    def defects(self) -> list[Violation]:
        """Return the violations that fail the conformer, in the order they were read."""
        return [v for v in self.violations if v.kind not in self.diagnostic]

    @property
    def ok(self) -> bool:
        """Return True when the conformer has no defect."""
        return not self.defects

    def summary(self) -> str:
        """Format the shape reading, then the violations one per line (or an all-clear message)."""
        lines = [self.shape] if self.shape is not None else []
        if not self.violations:
            lines.append("geometry OK (no violations)")
        else:
            lines.append(f"{len(self.violations)} geometry violation(s), {len(self.defects)} failing:")
            lines += [f"  - {v}{' (diagnostic)' if v.kind in self.diagnostic else ''}" for v in self.violations]
        return "\n".join(lines)


def clashes(mol, pos, *, exclude=frozenset(), coordination) -> list[Violation]:
    """Return the non-bonded pairs in steric overlap.

    Bonded and 1-3 pairs, donors of one metal and `exclude` atoms are skipped. The floors differ by kind (`_HEAVY_VDW`,
    `_HH_FLOOR`, `_XH_COV`), so H-bonds, halogen and pi contacts pass.
    """
    excluded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        excluded.add(canonical_key((i, j)))
    for atom in mol.GetAtoms():  # a 1-3 pair is its angle's question
        nb = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(nb)):
            for c in range(a + 1, len(nb)):
                if (
                    atom.GetAtomicNum() not in COORDINATION_METALS
                    and atom.GetIdx() not in exclude
                    and mol.GetAtomWithIdx(nb[a]).GetAtomicNum() == 1
                    and mol.GetAtomWithIdx(nb[c]).GetAtomicNum() == 1
                ):
                    continue
                excluded.add((min(nb[a], nb[c]), max(nb[a], nb[c])))
    for sphere in coordination.values():  # two donors of one metal are 1-3 through it
        excluded.update(itertools.combinations(sorted(sphere), 2))
    pos = np.asarray(pos, float)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    n = len(z)
    skip = np.zeros((n, n), dtype=bool)
    gone = np.isin(np.arange(n), list(exclude))
    skip[gone] = True
    skip[:, gone] = True
    for i, j in excluded:
        skip[i, j] = True
    h = z == 1
    rcov = np.array([_PT.GetRcovalent(int(x)) for x in z])
    rvdw = np.array([_PT.GetRvdw(int(x)) for x in z])
    limit = np.where(
        h[:, None] & h[None, :],
        _HH_FLOOR,
        np.where(
            h[:, None] | h[None, :],
            _XH_COV * (rcov[:, None] + rcov[None, :]),
            _HEAVY_VDW * (rvdw[:, None] + rvdw[None, :]),
        ),
    )
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    out = []
    # A matrix norm may differ from the scalar norm in the last ulp: screen a hair wide, re-check with the scalar.
    for i, j in np.argwhere(np.triu((dist < limit * (1 + 1e-9)) & ~skip, 1)):
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if d < limit[i, j]:
            why = ("heavy-atom steric overlap", "H buried in heavy atom", "H...H clash")[int(h[i]) + int(h[j])]
            out.append(Violation(kind="clash", atoms=(int(i), int(j)), value=d, limit=float(limit[i, j]), detail=why))
    return out


def over_compression(mol, pos, exclude=frozenset()) -> list[Violation]:
    """Return the non-bonded heavy 1-3 pairs closer than their covalent sum: a ring the relax closed.

    A real small ring (epoxide, cyclopropane) bonds the pair, so it never enters; a middle atom in `exclude`, or a
    hydrogen, is skipped.
    """
    out: list[Violation] = []
    for mid_atom in mol.GetAtoms():
        mid = mid_atom.GetIdx()
        if mid in exclude or mid_atom.GetAtomicNum() == 1:
            continue
        nbrs = [n.GetIdx() for n in mid_atom.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in exclude]
        for u in range(len(nbrs)):
            for w in range(u + 1, len(nbrs)):
                a, c = nbrs[u], nbrs[w]
                if mol.GetBondBetweenAtoms(a, c) is not None:
                    continue
                r_sum = _PT.GetRcovalent(mol.GetAtomWithIdx(a).GetAtomicNum()) + _PT.GetRcovalent(
                    mol.GetAtomWithIdx(c).GetAtomicNum()
                )
                floor = _FUSE_RATIO * r_sum
                d = float(np.linalg.norm(pos[a] - pos[c]))
                if d < floor:
                    ang = bond_angle(pos[a], pos[mid], pos[c])
                    a_s, c_s, m_s = atom_label(mol, a), atom_label(mol, c), atom_label(mol, mid)
                    out.append(
                        Violation(
                            kind="fusion",
                            atoms=(a, mid, c),
                            value=d,
                            limit=floor,
                            detail=f"{a_s}...{c_s} fused to {d:.2f} A across {m_s} (angle {ang:.0f}°): "
                            "a bond the graph does not have",
                        )
                    )
    return out


def planarity(mol, pos, exclude=frozenset()) -> list[Violation]:
    """Return each sp2 carbon off its neighbours' plane and each puckered aromatic ring.

    Nitrogen is left out, since an aniline N is pyramidal even where RDKit says sp2, and `conjugation` reads a twisted
    amide. Carbons in `exclude` are skipped; others are judged on their non-metal neighbours.
    """
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        if atom.GetIdx() in exclude:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() not in COORDINATION_METALS]
        if len(nbrs) != SP2_DEGREE:
            continue
        normal = np.cross(pos[nbrs[1]] - pos[nbrs[0]], pos[nbrs[2]] - pos[nbrs[0]])
        norm = np.linalg.norm(normal)
        height = 0.0 if norm < _EPS else float(abs(np.dot(pos[atom.GetIdx()] - pos[nbrs[0]], normal / norm)))
        if height > _SP2_OOP:
            out.append(
                Violation(
                    kind="planarity",
                    atoms=(atom.GetIdx(), *nbrs),
                    value=height,
                    limit=_SP2_OOP,
                    detail="sp2 out of plane",
                )
            )
    # SymmSSSR rewrites RingInfo, so a copy; a fused system's perimeter is left to the smaller rings sharing its edges
    graph = Chem.Mol(mol)
    Chem.GetSymmSSSR(graph)
    rings = graph.GetRingInfo()
    for ring, bonds in zip(rings.AtomRings(), rings.BondRings(), strict=True):
        if any(i in exclude for i in ring) or not all(graph.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        smaller_edges = sum(rings.MinBondRingSize(bond) < len(ring) for bond in bonds)
        if smaller_edges > 1:
            continue
        p = pos[list(ring)]
        rms = float(np.sqrt((np.linalg.svd(p - p.mean(0))[1][2] ** 2) / len(ring)))
        if rms > _RING_RMS:
            out.append(
                Violation(
                    kind="planarity",
                    atoms=tuple(ring),
                    value=rms,
                    limit=_RING_RMS,
                    detail="aromatic ring puckered",
                )
            )
    return out


def conjugation(mol, pos, exclude=frozenset(), flex=frozenset()) -> list[Violation]:
    """Return each twisted amide, ester, enamine or enol ether: a conjugated X-C bond out of plane.

    The A=C-X-S dihedral must lie within `_CONJ_TWIST` of 0 or 180 deg, or `_FLEX_CONJ` where a side-on eta2 atom is
    involved. A ring locks its own.
    """
    out = []
    for a, c, x, s in conjugated_quartets(mol, exclude):
        limit = _FLEX_CONJ if (c in flex or x in flex or a in flex) else _CONJ_TWIST
        dih = abs(dihedral_angle(pos[a], pos[c], pos[x], pos[s]))
        dev = min(dih, abs(180.0 - dih))
        if dev > limit:
            out.append(
                Violation(
                    kind="conjugation",
                    atoms=(a, c, x, s),
                    value=dev,
                    limit=limit,
                    detail="twisted out of plane",
                )
            )
    return out


def frozen_core(pos, frozen, reference) -> list[Violation]:
    """Return a violation when the `frozen` atoms sit over `_CORE_RMSD` from `reference` after superposition."""
    frozen = list(frozen)
    if not frozen:
        return []
    ref = as_positions(reference)
    ssd, _ = rdAlignment.GetAlignmentTransform(ref[frozen], pos[frozen])
    rmsd = (ssd / len(frozen)) ** 0.5
    if rmsd > _CORE_RMSD:
        return [Violation(kind="frozen_core", atoms=tuple(frozen), value=rmsd, limit=_CORE_RMSD, detail="core moved")]
    return []


def check_constraints(mol, pos, constraints) -> list[Violation]:
    """Return the distance, angle and dihedral windows of `constraints` a geometry misses by over `_WINDOW_SLACK`."""
    out = []
    for name, windows in (
        ("distance", constraints.distances),
        ("angle", constraints.angles),
        ("dihedral", constraints.dihedrals),
    ):
        slack = _WINDOW_SLACK[name]
        for atoms, given in windows.items():
            if not DIST_ATOMS <= len(atoms) <= DIHEDRAL_ATOMS:
                raise ValueError(f"constraint {atoms}: a geometric term needs 2, 3 or 4 atoms")
            if constraints.phantoms.intersection(atoms):
                continue  # private DG/UFF scaffold, not a geometric term on the stored molecule
            window = tuple(given) if isinstance(given, (tuple, list)) else (given, given)
            value = constraint_value(pos, atoms, constraints.haptic, window)
            if not within_window(value, window, slack):
                lo, hi = window
                out.append(
                    Violation(
                        kind="constraint",
                        atoms=atoms,
                        value=float("nan") if value is None else value,
                        limit=hi + slack,
                        detail=(
                            f"{name} could not be measured; check atom indices and haptic metadata"
                            if value is None or not np.isfinite(value)
                            else f"{name} window [{lo}, {hi}]"
                        ),
                    )
                )
    return out


def held(constraints):
    """Return the `Constraints` a geometry answers to: `constraints`, or an empty record for None.

    Any other kind is refused naming the record, the one form `check` and `Ensemble.check` read.
    """
    if constraints is None:
        return Constraints()
    if type(constraints) is not Constraints:
        raise TypeError(
            f"constraints= takes an rx.Constraints record, such as rx.Constraints(distances={{(i, j): (lo, hi)}}), "
            f"not a {type(constraints).__name__}"
        )
    return constraints


def check(mol, conf_id=-1, *, reference=None, constraints=None, donors=None, check_windows=True) -> GeometryReport:
    """Run the full physical gate on one conformer of `mol` and collect every violation.

    ``reference`` (a Mol, positions or an ``.xyz`` path in `mol`'s atom order) gives the stereo to keep and the geometry
    the frozen core is held to. ``constraints`` is an `rx.Constraints`; ``donors`` and ``check_windows`` are as `report`
    reads them.
    """
    require_mol(mol, "rx.geom_check.check", geometry=True)
    specified = None if reference is None else specified_like(mol, reference)
    return report(
        mol, conf_id, constraints=held(constraints), donors=donors, check_windows=check_windows, specified=specified
    )


def report(mol, conf_id, *, constraints, donors, check_windows, specified) -> GeometryReport:
    """Return the `GeometryReport` of one conformer under the `Constraints` it answers to.

    The frozen core (a TS core) skips the ground-state checks, where a forming bond is correct, and is held to the
    `specified` Mol's conformer by an RMSD check; a violation inside one fix, constrain or contact key is the request's
    own geometry. Donor-orientation floors and conjugation where the constraints leave cleanup out are diagnostics only.
    ``check_windows`` also reads the constraints' windows; leave it off where they are only biases. `specified` is a Mol
    from `changes.specified_like`, None for none; `donors` overrides the donors the graph names.
    """
    pos = as_positions(mol, conf_id)
    if not np.all(np.isfinite(pos)):
        return GeometryReport(
            [
                Violation(
                    kind="coordinates",
                    atoms=(),
                    value=float("nan"),
                    limit=0.0,
                    detail="non-finite coordinate",
                )
            ]
        )
    metals = metal_indices(mol)
    if donors is None:  # the graph names each metal's donors; a metal it leaves bond-less is perceived (`spheres`)
        donors = {m: [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()] for m in metals}
    core = constraints.frozen
    exclude = core | set(metals)  # dative, not vdW
    coordination = spheres(mol, pos, donors)
    bound = Counter(d for sphere in coordination.values() for d in sphere)
    hyb = stripped_hybridisation(mol, bound) if coordination else {}
    coord_c = _coordinating_carbons(mol, coordination, hyb, constraints)
    v: list[Violation] = []
    v += bond_violations(mol, pos, core, constraints.distances)
    v += clashes(mol, pos, exclude=exclude, coordination=coordination)
    v += over_compression(mol, pos, exclude=exclude)
    v += planarity(mol, pos, exclude=exclude | coord_c)
    v += conjugation(mol, pos, exclude=exclude, flex=_eta2_pi_atoms(mol, pos, donors, coordination) | coord_c)
    v += metal_overbond(mol, pos, donors)
    v += donor_orientation(mol, pos, donors, core, hyb=hyb)
    owned = [core, *map(set, (*constraints.fixed, *constraints.contacts.distances, *constraints.contacts.angular))]
    v = [x for x in v if not any(term.issuperset(x.atoms) for term in owned)]
    if specified is not None:
        v += frozen_core(pos, core, specified)
    if check_windows:
        v += check_constraints(mol, pos, constraints)
    if specified is not None and (problem := stereo_change(mol, conf_id, specified)):
        v.append(Violation(kind="stereo", atoms=(), value=0.0, limit=0.0, detail=problem))
    diagnostic = {"donor_orientation"} if constraints.conjugation else {"donor_orientation", "conjugation"}
    return GeometryReport(v, diagnostic=frozenset(diagnostic))


_EPS = 1e-9  # numerical floor for a degenerate cross-product / near-zero norm


def _eta2_pi_atoms(mol, pos, donors, coordination) -> set[int]:
    """Return the atoms of side-on eta2 units: two bonded donors of one metal.

    The metal pulls such a pair a little out of plane, so it gets the wider conjugation window. In a perceived sphere
    the pair must also be pi-bonded, bind symmetrically and sit within `_SIDEON_MAX`, since backbone atoms sit there
    too.
    """
    out: set[int] = set()
    for m, sphere in coordination.items():
        declared = bool(declared_donors(donors, m))  # `spheres` perceives only an undeclared metal's sphere
        for a, b in itertools.combinations(sorted(sphere), 2):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None:
                continue
            if not declared:
                da, db = float(np.linalg.norm(pos[m] - pos[a])), float(np.linalg.norm(pos[m] - pos[b]))
                if bond.GetBondTypeAsDouble() < 2 or abs(da - db) > _SIDEON_SYM or max(da, db) > _SIDEON_MAX:  # noqa: PLR2004
                    continue
            out.update((a, b))
    return out


def _coordinating_carbons(mol, coordination, hyb, constraints) -> set[int]:
    """Return the sphere carbons `report` spares in `planarity` and gives the wider `conjugation` window.

    A carbanion, carbene or eta2 carbon pyramidalises. A sphere carbon keeps the strict rule only when it is sp2 once
    stripped, aromatic with three ligand neighbours and no lone pair, in no haptic face, and the constraints hold
    neither its donor direction nor a dihedral over it.
    """
    metals = set(coordination)
    oriented = donor_directions(
        constraints.angles,
        constraints.coplanar,
        metals,
        releasable=constraints.contacts.angular,
        fixed=constraints.fixed,
    )
    held_dihedrals = {frozenset(atoms) for atoms in constraints.dihedrals}
    out = set()
    for sphere in coordination.values():
        haptic = {atom for face in haptic_sites(mol, sphere) if len(face) > 1 for atom in face}
        for i in sphere:
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() != CARBON_Z:
                continue
            framework = frozenset((i, *(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in metals)))
            if (
                i in haptic
                or i in oriented
                or framework in held_dihedrals
                or hyb.get(i) != Chem.HybridizationType.SP2
                or not no_aromatic_axis(atom)
            ):
                out.add(i)
    return out
