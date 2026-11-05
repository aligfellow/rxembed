"""Say which way a donor points, and write the soft holds that keep it there once its metal bond is stripped.

The holds (`orient_donor`, one wall per substituent; `coplanar_donor`, the sp2-plane cap) and the fold check
(`shape.donor_orientation`) read the same hybridisation and donation axis, so the two cannot drift. `ring_hinge` holds a
small conjugated chelate ring flat.
"""

from __future__ import annotations

import math

from rdkit import Chem

from rxembed.geometry.bounds import bounds_matrix
from rxembed.geometry.constraints import CoplanarRow
from rxembed.graph import (
    COORDINATION_METALS,
    haptic_sites,
    ligand_degree,
    ligand_graph,
    lone_pair_electrons,
    no_aromatic_axis,
)
from rxembed.metal.distances import APEX_DONORS
from rxembed.utils import CARBON_Z

_PT = Chem.GetPeriodicTable()

# the sp2-donor coplanarity cap: a donor with a free sigma direction holds the metal in its own plane
COPLANAR_CAP = 45.0  # deg half-window off the anchor, past the crystal-structure high percentile
# out of a donor's plane. Never a point, which would annihilate the real scatter out to that tail.
_COPLANAR_ANCHOR = 180.0  # deg: the external sector of a donor with two direct substituents

# --- the chelate-ring hinge (`ring_hinge`, `cons.coplanar`) ----------------------------------------------
_HINGE_CAP = {
    4: 23.0,  # deg: max fold of a conjugated 4-membered chelate ring in crystal structures
    5: 35.0,  # deg: p99 fold of a conjugated 5-membered chelate ring in crystal structures
}

# crystal-structure percentiles of where a ligand points, per (element, hyb); a class with n<6 abstains
_SP, _SP2, _SP3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
FOLD_WINDOW = {  # deg (floor, ceiling) per (element, hyb); floor gates, ceiling feeds `_ORIENT_WALL`
    ("N", _SP2): (96.0, 180.0),
    ("P", _SP3): (89.0, 138.5),
    ("C", _SP2): (85.0, 145.5),
    ("N", _SP3): (82.0, 158.4),
    ("C", _SP): (155.0, 180.0),
    ("O", _SP2): (90.0, 156.8),
    ("S", _SP3): (91.0, 110.0),  # ceiling: the wall knee; the crystal ceiling would let S flatten
    ("C", _SP3): (104.0, 133.8),
    ("N", _SP): (140.0, 180.0),
    ("As", _SP3): (95.0, 131.3),
}
FOLD_MEDIAN = {  # deg: crystal-structure median per class; feeds `_FOLD_WALL_FLOOR`
    ("N", _SP2): 120.7,
    ("P", _SP3): 115.8,
    ("C", _SP2): 124.4,
    ("N", _SP3): 110.3,
    ("C", _SP): 176.2,
    ("O", _SP2): 125.7,
    ("S", _SP3): 103.4,
    ("C", _SP3): 113.3,
    ("N", _SP): 179.3,
    ("As", _SP3): 119.4,
}
# deg: the embed's fold-wall floor, tighter than the gate's, which must never flag a real crystal
_FOLD_WALL_FLOOR = {cls: min(lo + 12.0, FOLD_MEDIAN[cls] - 5.0) for cls, (lo, _hi) in FOLD_WINDOW.items()}
# deg: half-width around a donor's graph-derived M-D-X: a free sp2 donor's bisector, an sp donor's 180
_CENTRED_PAD = 6.0
# deg (floor, ceiling) of `orient_donor`'s wall; an sp row is centred, as nothing else holds it straight
_ORIENT_WALL = {cls: (floor, FOLD_WINDOW[cls][1]) for cls, floor in _FOLD_WALL_FLOOR.items()}
_ORIENT_WALL[("N", _SP)] = _ORIENT_WALL[("C", _SP)] = (180.0 - _CENTRED_PAD, 180.0)
_MAX_SIGMA = {  # sigma bonds a class can carry: more is a hypervalent / mis-perceived centre -> unknown
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
}
_CONJUGATING_LP = frozenset({7, 8})  # period-2 only: N/O planarise into an adjacent π system, a period-3 lone
# pair does not (PPh3 is pyramidal). Letting P/S conjugate would type every triarylphosphine sp2.


# donor perception: walls and fold check must name one donor class, or a seed is flagged the other way


def _pi_hybridisation(atom) -> Chem.HybridizationType | None:
    """Return the hybridisation an atom's pi-bond count gives, independent of RDKit's typer.

    Two pi bonds give sp, one sp2 and none sp3, unless the atom is aromatic or a period-2 lone pair conjugates into a
    neighbouring pi system (amide N, carboxylate O). Where this and RDKit disagree (a carbanion, ylide, hypervalent S,
    an arbitrary Kekule form) the donor is left untyped.
    """
    n_pi = 0
    for b in atom.GetBonds():
        order = b.GetBondTypeAsDouble()
        if order >= 3:  # noqa: PLR2004  a triple bond is two π bonds
            n_pi += 2
        elif order >= 2:  # noqa: PLR2004
            n_pi += 1
    if n_pi >= 2:  # noqa: PLR2004
        return Chem.HybridizationType.SP
    if n_pi == 1:
        return Chem.HybridizationType.SP2
    if atom.GetIsAromatic():
        return Chem.HybridizationType.SP2
    if atom.GetAtomicNum() in _CONJUGATING_LP:
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic() or any(b.GetBondTypeAsDouble() >= 2 for b in nb.GetBonds()):  # noqa: PLR2004
                return Chem.HybridizationType.SP2
    return Chem.HybridizationType.SP3


def _stripped_graph(mol):
    """Return the sanitised metal-free ligand graph of `mol`, where hybridisation and conjugation are read.

    Stripping first makes covalent and dative input read alike: RDKit skips a dative bond but not a covalent metal bond.
    """
    stripped = ligand_graph(mol)
    # The isolated metal keeps indices stable: infer no radical on it, and skip properties on bare donors.
    flags = (
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    )
    Chem.SanitizeMol(stripped, flags, catchErrors=True)
    return stripped


def stripped_hybridisation(mol, bound=None) -> dict[int, Chem.HybridizationType]:
    """Return ``{atom: hybridisation}`` on the metal-stripped graph where RDKit's typer and the pi count agree.

    An atom is left untyped, so never judged, where they disagree or it is hypervalent for its class: abstaining beats a
    mis-typed fold. Three corrections come first:

    * a 3-coordinate centre with a non-conjugating lone pair is pyramidal even when one bond is drawn double (a neutral
    sulfoxide); * a period-3+ lone pair does not conjugate into a ring as a period-2 one does, so an aryl-thiolate S
    stays pyramidal; * a metal-bound sp heteroatom with two ligand sigma bonds is retyped sp2, since a truly cumulated
    centre has no lone pair left to donate; carbon is excluded, as a non-terminal sp carbon donates through its pi
    cloud.

    A metal-bound ``[CH-]`` stays untyped and its pyramid is judged through its chiral tag. `bound` gives each declared
    donor's metal count, so stripped and bonded graphs type a donor alike.
    """
    bound = {} if bound is None else bound
    stripped = _stripped_graph(mol)
    out: dict[int, Chem.HybridizationType] = {}
    for a in stripped.GetAtoms():
        rdkit_h, pi_h = a.GetHybridization(), _pi_hybridisation(a)
        nonbonding = lone_pair_electrons(a, ())  # no metal bond on this stripped graph
        if (
            pi_h == Chem.HybridizationType.SP2
            and a.GetTotalDegree() == 3  # noqa: PLR2004
            and a.GetAtomicNum() not in _CONJUGATING_LP
            and nonbonding >= 2  # noqa: PLR2004  a lone pair
        ):
            rdkit_h = pi_h = Chem.HybridizationType.SP3
        # the second correction; carbon is period 2, so the carbanion still abstains
        elif rdkit_h == _SP2 and pi_h == _SP3 and _PT.GetRow(a.GetAtomicNum()) > 2:  # noqa: PLR2004
            rdkit_h = _SP3
        # the third correction
        elif (
            rdkit_h == pi_h == _SP
            and a.GetTotalDegree() == 2  # noqa: PLR2004
            and a.GetAtomicNum() != CARBON_Z
            and bound.get(a.GetIdx(), 0)
        ):
            rdkit_h = pi_h = _SP2
        if rdkit_h != pi_h or rdkit_h not in _MAX_SIGMA:  # disagreement, or not sp/sp2/sp3
            continue
        if a.GetTotalDegree() > _MAX_SIGMA[rdkit_h]:  # more sigma bonds than the class can carry: hypervalent
            continue
        out[a.GetIdx()] = rdkit_h
    return out


def donation_axis(mol, d, all_donors, sphere=None, *, hyb=None, metals=1, stripped=None) -> list[int] | None:
    """Return donor `d`'s judgeable heavy substituents, or None when it donates along no axis.

    Hydrogen, bridging, haptic, nonterminal sp and filled aromatic sp2 donors have no axis of their own. Metals,
    protons, co-donors, bite bridgeheads and backbone arms another donor holds are excluded; two small, separately held
    donors each keep their arm. `sphere` supplies this metal's bite topology, and `metals` the declared binding counts.
    """
    a = mol.GetAtomWithIdx(d)
    if a.GetAtomicNum() == 1:  # hydride / η²-H₂ / agostic H: no lone pair, so no donation axis
        return None
    if metals > 1:  # bridging: the axis is set by the bridge
        return None
    sphere = all_donors if sphere is None else sphere
    if any(d in site and len(site) > 1 for site in haptic_sites(mol, sphere)):  # pi-face: metal is off-axis
        return None
    hyb = stripped_hybridisation(mol) if hyb is None else hyb
    if hyb.get(d) == _SP2 and no_aromatic_axis(a):
        return None
    if hyb.get(d) == _SP and ligand_degree(a) != 1:
        return None
    backbone = _backbone_targets(mol, d, sphere, stripped=stripped)
    d_class = (mol.GetAtomWithIdx(d).GetSymbol(), hyb.get(d))
    return [
        nb.GetIdx()
        for nb in a.GetNeighbors()
        if nb.GetAtomicNum() > 1  # protons have their own window (`orient_donor`)
        and nb.GetAtomicNum() not in COORDINATION_METALS
        and nb.GetIdx() not in all_donors  # co-donor
        and not (  # shared backbone arm
            (other := backbone.get(nb.GetIdx())) is not None
            and (mol.GetAtomWithIdx(other).GetSymbol(), hyb.get(other)) in _ORIENT_WALL
            and _heavy_substituent_count(mol, other) < 3  # noqa: PLR2004
            and not (d_class in _ORIENT_WALL and _heavy_substituent_count(mol, d) < 3)  # noqa: PLR2004
        )
        # A chelate bridgehead has no independent M-D-X angle even though it still needs anti-collapse repulsion.
        and sum(1 for x in sphere if mol.GetBondBetweenAtoms(nb.GetIdx(), int(x)) is not None) < APEX_DONORS
    ]


def _backbone_targets(mol, d, donor_set, *, stripped=None):
    """Map each substituent of donor ``d`` that reaches another donor through the ligand backbone to that donor.

    Coordination bonds are removed first, since a path through the metal is no backbone. In a macrocycle an arm can
    reach two nested donors; the nearer, its own chelate partner, is kept.
    """
    network = ligand_graph(mol) if stripped is None else stripped
    targets, lengths = {}, {}
    for other in donor_set:
        if other == d:
            continue
        path = Chem.GetShortestPath(network, int(d), int(other))
        if len(path) > 1 and len(path) < lengths.get(path[1], math.inf):
            targets[path[1]], lengths[path[1]] = other, len(path)
    return targets


def _heavy_substituent_count(mol, atom_idx):
    """Count ``atom_idx``'s non-metal heavy neighbours, for `donation_axis`'s shared-arm size test."""
    return sum(
        1
        for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nb.GetAtomicNum() > 1 and nb.GetAtomicNum() not in COORDINATION_METALS
    )


def _centred_sp2_window(mol, donor, neighbours):
    """Centre a free donor on the external bisector of RDKit's graph-derived ligand angle."""
    left, right = neighbours
    matrix = bounds_matrix(mol)
    a, b, c = (0.5 * (matrix[i, j] + matrix[j, i]) for i, j in ((left, donor), (right, donor), (left, right)))
    cosine = (a * a + b * b - c * c) / (2.0 * a * b)
    internal = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    centre = 180.0 - 0.5 * internal
    return centre - _CENTRED_PAD, min(180.0, centre + _CENTRED_PAD)


def orient_donor(mol, metal, d, donor_set, *, hyb=None, stripped=None, metals=1):
    """Return the walls holding every donor substituent, heavy or proton, off the metal: the M-D-X bend lost.

    One flat-bottomed angle wall per substituent, by the donor's (element, hyb) class, biasing both the bounds and the
    FF. The heavy substituents are exactly those the fold check judges, so walls and check abstain alike; a proton is
    walled too, which keeps an sp3 amine from folding its H onto the metal. An untyped donor or uncalibrated class gets
    none. A free C or N sp2 donor with two heavy neighbours and no backbone to another donor centres its window on the
    ligand's own bisector.
    """
    hyb = stripped_hybridisation(mol) if hyb is None else hyb
    a = mol.GetAtomWithIdx(d)
    cls = (a.GetSymbol(), hyb.get(d))
    window = _ORIENT_WALL.get(cls)
    heavy = donation_axis(mol, d, donor_set, hyb=hyb, metals=metals, stripped=stripped) if window else None
    if heavy is None:  # uncalibrated, or no donation axis (hydride, bridge, haptic face, nonterminal sp)
        return {}
    ligand = [
        nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1 and nb.GetAtomicNum() not in COORDINATION_METALS
    ]
    if (
        cls in {("C", _SP2), ("N", _SP2)}
        and len(ligand) == 2  # noqa: PLR2004  the centred window's math needs exactly two neighbours
        and not _backbone_targets(mol, d, donor_set, stripped=stripped)  # a chelate already pins the axis
    ):
        window = _centred_sp2_window(mol, d, ligand)
    heavy = set(heavy)
    return {  # neighbour order, so the walls reach the FF in a stable order
        (metal, d, nb.GetIdx()): window
        for nb in a.GetNeighbors()
        if nb.GetIdx() in heavy or (nb.GetAtomicNum() == 1 and nb.GetIdx() not in donor_set)
    }


def coplanar_donor(mol, metal, d, donor_set, *, hyb):
    """Return the cap holding an sp2 donor's metal in the donor's own plane: the improper the stripped bond lost.

    At most one soft `CoplanarRow` (+/- `COPLANAR_CAP` deg), applied in bounds and FF. The plane depends on the donor's
    heavy neighbours:

    * one (carboxylate O, thione S): ``donor_torsion``, the dihedral M-D-C-X against C's heaviest other substituent,
    with no syn/anti bias; * two (amidate or imine N, aryl carbanion C): ``donor_plane``, the metal's angle out of the
    X-D-Y plane, which belongs to neither donor bond, so an N-aryl amidate's phenyl stays free to twist.

    No M-D-C wall is written here: `orient_donor` already holds M-D-X for every calibrated class.
    """
    a = mol.GetAtomWithIdx(d)
    if hyb.get(d) != _SP2 or no_aromatic_axis(a):
        return []
    heavy = [
        nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1 and nb.GetAtomicNum() not in COORDINATION_METALS
    ]
    if len(heavy) == 1:  # one heavy neighbour: proper dihedral against C's heaviest substituent
        if hyb.get(heavy[0]) != Chem.HybridizationType.SP2:  # the neighbour must itself be sp2 (a real plane to hold)
            return []
        bridge = heavy[0]
        ref = max(
            (nb for nb in mol.GetAtomWithIdx(bridge).GetNeighbors() if nb.GetIdx() != d and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
            default=None,
        )
        # an aldehyde-O whose C carries only H gets none: no reference atom
        if ref is None:
            return []
        return [CoplanarRow(kind="donor_torsion", atoms=(metal, d, bridge, ref.GetIdx()), cap=COPLANAR_CAP)]
    if len(heavy) == 2:  # noqa: PLR2004  two heavy neighbours: improper M out of the donor's own X-D-Y plane
        atoms = (metal, d, *sorted(heavy))
        return [CoplanarRow(kind="donor_plane", atoms=atoms, anchor=_COPLANAR_ANCHOR, cap=COPLANAR_CAP)]
    return []


def ring_hinge(mol, metal, donor_set):
    """Return rows holding the metal in the plane of a small, fully conjugated chelate ring.

    A ring whose whole metal-free D...D' path conjugates is planar as a unit, whatever each donor's hybridisation (a
    dithiocarbamate and a benzenedithiolate hinge alike). Only 4- and 5-membered rings: a 6-ring (acac, nacnac, salen)
    really folds. One ``ring_hinge`` row per donor D, atoms ``(M, D, D', X)``: the dihedral about the D-D' diagonal,
    with M and X on opposite sides. D' is a co-donor, so no DG angle bound applies, only the FF torsion.
    """
    stripped = _stripped_graph(mol)
    donors = sorted(donor_set)
    rows = []
    for i, d in enumerate(donors):
        for other in donors[i + 1 :]:
            path = Chem.GetShortestPath(stripped, int(d), int(other))
            cap = _HINGE_CAP.get(len(path) + 1)  # ring size = the metal-free path atoms plus the metal itself
            if cap is None:
                continue
            if not all(
                stripped.GetBondBetweenAtoms(path[k], path[k + 1]).GetIsConjugated() for k in range(len(path) - 1)
            ):
                continue
            for atoms in ((metal, d, other, path[1]), (metal, other, d, path[-2])):
                rows.append(CoplanarRow(kind="ring_hinge", atoms=atoms, anchor=_COPLANAR_ANCHOR, cap=cap))
    return rows
