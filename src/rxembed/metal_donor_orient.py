"""Donor perception, and the orientation holds the stripped metal surrogate loses at the M-donor bond.

The metal-stripped hybridisation ruler (`stripped_hybridisation`) and fold-census predicates (`donation_axis`,
`inplane_sp2_donor`), plus the enforcement that puts those orientations back softly: `orient_donor` (one
census wall per substituent) and `coplanar_donor` (the sp2-plane cap). The gate side
(`metal_perceive.donor_orientation` / `donor_fold`) reads the same ruler, so they cannot drift.
"""

from __future__ import annotations

import math

from rdkit import Chem

from .bounds import bounds_matrix
from .metal_core import COORDINATION_METALS, haptic_sites, ligand_degree, ligand_graph
from .metal_distance import APEX_DONORS
from .utils import CARBON_Z, lone_pair_electrons

_PT = Chem.GetPeriodicTable()

# --- the sp2-donor coplanarity cap (`coplanar_donor`, `cons.coplanar`) -------------------------------
# An sp2 donor binds from an in-plane sigma lone pair, so the metal sits in its sp2 framework.
COPLANAR_CAP = 45.0  # deg half-window off the anchor: clears the tmQM/Kulik census p95 of 40° for the metal's
# angle out of a donor's plane. Never a point, which would annihilate the real scatter out to that tail.
_COPLANAR_ANCHOR = 180.0  # deg: the external sector of a donor with two direct substituents

# --- the chelate-ring hinge (`ring_hinge`, `cons.coplanar`) ----------------------------------------------
_HINGE_CAP = {
    4: 23.0,  # deg: crystal census max fold for a conjugated 4-membered chelate ring, 22.8 deg
    5: 35.0,  # deg: crystal census p99 fold for a conjugated 5-membered chelate ring, 34.2 deg
}

# --- the donor-fold census: what `orient_donor` enforces and `metal_perceive.donor_orientation` gates ------
# Crystal census percentiles of where a ligand points, keyed on (element, hyb): hybridisation alone admits a
# folded carbonyl, and thiolate donates 23° off carboxylate. A class with n<6 abstains.
_SP, _SP2, _SP3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
FOLD_WINDOW = {  # deg (floor, ceiling) per (element, hyb); floor gates, ceiling reports
    ("N", _SP2): (96.0, 180.0),
    ("P", _SP3): (89.0, 138.5),
    ("C", _SP2): (85.0, 145.5),
    ("N", _SP3): (82.0, 158.4),
    ("C", _SP): (155.0, 180.0),
    ("O", _SP2): (90.0, 156.8),
    ("S", _SP3): (91.0, 110.0),
    ("C", _SP3): (104.0, 133.8),
    ("N", _SP): (140.0, 180.0),
    ("As", _SP3): (95.0, 131.3),
}
FOLD_MEDIAN = {  # deg: census median per class; `fold` = max |M-D-X - median|, so it needs no reference structure
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
# deg: the seed-bias floor for the embed's fold wall, tighter than the gate floor above. The gate never
# false-flags a real crystal but sits 25-30 deg below typical donation, a dead band a fold hides in. Only
# the wall may tighten: a tighter gate would flag real crystals.
_FOLD_WALL_FLOOR = {cls: min(lo + 12.0, FOLD_MEDIAN[cls] - 5.0) for cls, (lo, _hi) in FOLD_WINDOW.items()}
# deg: half-width of a window centred on a donor's graph-derived M-D-X angle. A free sp2 donor centres on the
# external bisector of the ligand's native internal angle (a fixed 120-degree centre pushes a five-membered
# donor out of plane; a chelate already pins its donor axis). An sp donor is linear, so it centres on 180.
_CENTRED_PAD = 6.0
# deg (floor, ceiling): the wall `orient_donor` writes on every donor substituent, heavy or proton. One rule
# leans an acyl leg off the sphere, splays a slow-inverting P-H, and splays a folded amine's H. An sp row is
# centred instead: nothing else holds a linear donor straight, so the relax rides whatever floor it is given.
_ORIENT_WALL = {cls: (floor, FOLD_WINDOW[cls][1]) for cls, floor in _FOLD_WALL_FLOOR.items()}
_ORIENT_WALL[("N", _SP)] = _ORIENT_WALL[("C", _SP)] = (180.0 - _CENTRED_PAD, 180.0)
_MAX_SIGMA = {  # sigma bonds a class can carry: more is a hypervalent / mis-perceived centre -> unknown
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
}
_CONJUGATING_LP = frozenset({7, 8})  # period-2 only: N/O planarise into an adjacent π system, a period-3 lone
# pair does not (PPh3 is pyramidal). Letting P/S conjugate would type every triarylphosphine sp2.
_PYRAMIDAL_SIGMA = 3
_LONE_PAIR_ELECTRONS = 2


# --- donor perception: the metal-stripped hybridisation ruler + the fold-census predicates ------------
# Enforcement and gate must name the same donor class, or a wall-biased seed gets flagged the other way. Hence
# one definition here, which the gate imports.


def _pi_hybridisation(atom) -> Chem.HybridizationType | None:
    """Estimator B: hybridisation from a π-count, independent of RDKit's typer.

    Two π bonds = sp; one = sp2; none = sp3, unless a period-2 lone pair conjugates into an adjacent π system
    (amide N, carboxylate O), planarising it to sp2. Where the two estimators part company (a carbanion, ylide,
    hypervalent S, an arbitrary Kekulé form) the donor is unknown and gates nothing.
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
        # RDKit can label an anionic aryl/silyl N SP2 even when the graph has no explicit multiple-bond anchor.
        # Do not turn that ambiguous Lewis form into a hard anti donor plane; a nearby pi bond still establishes
        # the planar amidate-like case below.
        if atom.GetFormalCharge() < 0 and atom.GetDegree() >= 2:  # noqa: PLR2004
            pi_anchor = any(
                bond.GetBondTypeAsDouble() >= 2  # noqa: PLR2004
                for neighbour in atom.GetNeighbors()
                for bridge in (neighbour, *neighbour.GetNeighbors())
                if bridge.GetIdx() != atom.GetIdx()
                for bond in bridge.GetBonds()
            )
            if not pi_anchor:
                return Chem.HybridizationType.SP3
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic() or any(b.GetBondTypeAsDouble() >= 2 for b in nb.GetBonds()):  # noqa: PLR2004
                return Chem.HybridizationType.SP2
    return Chem.HybridizationType.SP3


def _stripped_graph(mol):
    """Return the metal-stripped, sanitised ligand graph: index-stable, so callers reuse `mol`'s atom indices.

    Strip coordination before perceiving hybridisation or conjugation so covalent and dative input conventions
    read the same ligand graph. RDKit excludes outgoing dative bonds from these perceptions but not covalent
    metal bonds, which can otherwise change the class of a donor or the conjugation of its ring path.
    """
    stripped = ligand_graph(mol)
    # The disconnected metal is retained only to keep atom indices stable. Do not infer radicals on its
    # deliberately isolated formal charge; properties are also skipped because stripped donors are bare.
    flags = (
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    )
    Chem.SanitizeMol(stripped, flags, catchErrors=True)
    return stripped


def stripped_hybridisation(mol) -> dict[int, Chem.HybridizationType]:
    """Return agreed hybridisation assignments on the metal-stripped graph.

    An atom is absent (unknown, never gated) when the typer and the pi-count estimator disagree, or when it
    is hypervalent for its class; abstaining beats a mis-typed fold. Three corrections to RDKit's own typer:

    * a 3-coordinate centre with a non-conjugating lone pair is pyramidal even when one bond is drawn double
      (a neutral sulfoxide `S(=O)R2`);
    * a period-3+ lone pair does not conjugate into a neighbour's ring the way a period-2 one does, so an
      aryl-thiolate-like S stays pyramidal even when RDKit reads it sp2 off the ring's aromaticity;
    * a metal-bound heteroatom sp centre with two ligand sigma bonds is retyped sp2: its only bond to the
      metal is a sigma lone pair, and a genuinely cumulated sp centre has none left to donate. Carbon is
      excluded, since a non-terminal sp carbon donates through its pi cloud, not a lone pair.

    A terminal donor also needs exactly one ligand-side neighbour; hybridisation alone does not fix a
    donation axis. Expected holdout: RDKit calls a metal-bound `[CH-]` carbanion sp2 and the pi-count sp3;
    rxembed judges it as a pyramidal stereocentre elsewhere, through its chiral tag, not here.
    """
    stripped = _stripped_graph(mol)
    out: dict[int, Chem.HybridizationType] = {}
    for a in stripped.GetAtoms():
        rdkit_h, pi_h = a.GetHybridization(), _pi_hybridisation(a)
        nonbonding = lone_pair_electrons(a, ())  # no metal bond on this stripped graph
        if (
            pi_h == Chem.HybridizationType.SP2
            and a.GetDegree() == _PYRAMIDAL_SIGMA
            and a.GetAtomicNum() not in _CONJUGATING_LP
            and nonbonding >= _LONE_PAIR_ELECTRONS
        ):
            rdkit_h = pi_h = Chem.HybridizationType.SP3
        # ring-conjugated sp2 by RDKit's aromaticity flag, not this atom's own lone pair (rule 2 above); carbon
        # stays period 2, so the carbanion holdout above still abstains rather than landing here.
        elif rdkit_h == _SP2 and pi_h == _SP3 and _PT.GetRow(a.GetAtomicNum()) > 2:  # noqa: PLR2004
            rdkit_h = _SP3
        # a metal-bound heteroatom sp centre with 2 ligand sigma bonds cannot really be sp (rule 3 above)
        elif (
            rdkit_h == pi_h == _SP
            and a.GetDegree() == 2  # noqa: PLR2004
            and a.GetAtomicNum() != CARBON_Z
            and any(nb.GetAtomicNum() in COORDINATION_METALS for nb in mol.GetAtomWithIdx(a.GetIdx()).GetNeighbors())
        ):
            rdkit_h = pi_h = _SP2
        if rdkit_h != pi_h or rdkit_h not in _MAX_SIGMA:  # the estimators disagree, or it is not sp/sp2/sp3
            continue
        if a.GetDegree() > _MAX_SIGMA[rdkit_h]:  # more sigma bonds than the class can carry: hypervalent
            continue
        out[a.GetIdx()] = rdkit_h
    return out


def inplane_sp2_donor(mol, d, hyb=None) -> bool:
    """Return whether donor `d` is sp2, and so holds the metal in its sigma plane.

    No conjugation test: an isolated C=O or C=N donor is sp2 but not conjugated.
    """
    if hyb is None:
        hyb = stripped_hybridisation(mol)
    return hyb.get(d) == Chem.HybridizationType.SP2


def donation_axis(mol, d, all_donors, sphere=None, *, hyb=None, network=True) -> list[int] | None:
    """Return donor `d`'s judgeable heavy substituents X, or `None` when it donates along no axis.

    Four donors have no M-D-X axis to judge: an H donor (hydride, sigma-complex, agostic) has no lone pair; a
    bridging donor (2+ metals) takes its axis from the bridge; a haptic donor paired or pi-bonded to a
    co-donor donates a face, sitting ~70 deg off any M-D-X axis regardless of that pair's bond order; a
    nonterminal sp atom has two opposing substituents, so neither can face away from the metal.

    Both the coordinate-space ruler (`_donor_walk`) and the enumeration screen share this abstention list. An
    empty list means every substituent is itself exempt (a co-donor, a kappa2 bite bridgehead, a proton, or a
    metal): "ask, but nothing to measure", distinct from `None` meaning "do not ask". Freeze ownership is
    applied later, over the complete M-D-X term, not here.

    `sphere` is this metal's own donors, for the bite-bridgehead test, and defaults to `all_donors`. `network`
    drops a backbone arm (R_pair) whose co-donor is calibrated and has fewer than 3 heavy substituents, so it
    holds the arm itself, unless `d` qualifies the same way, since a ring of two small donors must keep one
    walled arm. Enumeration disables it for its conservative outer-bound screen.
    """
    a = mol.GetAtomWithIdx(d)
    if a.GetAtomicNum() == 1:  # hydride / η²-H₂ / agostic H: no lone pair, so no donation axis
        return None
    if sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() in COORDINATION_METALS) > 1:  # bridging: set by the bridge
        return None
    sphere = all_donors if sphere is None else sphere
    if any(d in site and len(site) > 1 for site in haptic_sites(mol, sphere)):  # pi-face: metal is off-axis
        return None
    hyb = stripped_hybridisation(mol) if hyb is None else hyb
    if hyb.get(d) == _SP and ligand_degree(a) != 1:
        return None
    backbone = _backbone_targets(mol, d, sphere) if network else {}
    d_class = (mol.GetAtomWithIdx(d).GetSymbol(), hyb.get(d))
    return [
        nb.GetIdx()
        for nb in a.GetNeighbors()
        if nb.GetAtomicNum() > 1  # protons have their own window (`orient_donor`)
        and nb.GetAtomicNum() not in COORDINATION_METALS
        and nb.GetIdx() not in all_donors  # co-donor
        and not (  # R_pair, see the docstring
            (other := backbone.get(nb.GetIdx())) is not None
            and (mol.GetAtomWithIdx(other).GetSymbol(), hyb.get(other)) in _ORIENT_WALL
            and _heavy_substituent_count(mol, other) < 3  # noqa: PLR2004
            and not (d_class in _ORIENT_WALL and _heavy_substituent_count(mol, d) < 3)  # noqa: PLR2004
        )
        # A chelate bridgehead has no independent M-D-X angle even though it still needs anti-collapse repulsion.
        and sum(1 for x in sphere if mol.GetBondBetweenAtoms(nb.GetIdx(), int(x)) is not None) < APEX_DONORS
    ]


def _is_chelated(mol, d, donor_set, metal) -> bool:
    """Return True when another donor of ``metal`` is reachable from ``d`` through the ligand backbone.

    With the M-donor bonds stripped, a chelate's two donors relate only through their backbone (a bond path that
    never crosses the bond-less metal); a monodentate pair shares no such path. This is the structural split that
    gates the centred sp2 window: the chelate ring pins the donor's plane, a lone monodentate donor does not.
    """
    return any(
        (path := Chem.GetShortestPath(mol, int(d), int(dd))) and metal not in path for dd in donor_set if dd != d
    )


def _backbone_targets(mol, d, donor_set, *, stripped=None):
    """Map each of donor ``d``'s substituents that reaches another donor through the ligand backbone to it.

    Remove coordination edges before finding ligand paths, because public relaxed molecules restore those
    edges and a path through the metal is not a ligand backbone. A graph fact only; whether an arm's wall
    is exempted is the caller's policy (R_pair: is the co-donor at the far end itself independently held).
    In a macrocycle a single arm can reach two donors nested one past the other (a bridging donor between
    two chelate rings); keep the nearer one, the arm's own chelate partner, not one further down the ring.

    ``stripped`` reuses an already-built `ligand_graph`, shared across every donor of one candidate.
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
    """Count ``atom_idx``'s non-metal heavy neighbours, for the R_pair co-donor size check."""
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


def orient_donor(mol, metal, d, donor_set, cons, *, hyb=None, stripped=None):
    """Wall every donor substituent, heavy or proton, off the metal: the M-D-X bend the stripped bond lost.

    One flat-bottomed angle wall per non-metal, non-apex substituent, keyed on the donor's (element, hyb)
    census class (`_ORIENT_WALL`), written into `cons.angles` so it biases both the bounds matrix and the FF
    (a DG-only wall measures worse than none). One loop covers sp end-on, pnictogen proton splay and
    heavy-substituent fold together; a proton is a substituent too, which is what fixes an sp3 amine folding
    its H onto the metal over the lone pair.

    An uncalibrated class (estimators disagree, or n<6) abstains, exactly as the gate does. A heavy apex
    substituent (bonded to 2+ donors) is a geometrically forced bite apex, never walled. Freeze ownership is
    resolved later, over the complete term. `coplanar_donor` is the sibling out-of-plane cap; `stripped`
    reuses an already-built `ligand_graph`, shared across every donor of one candidate.
    """
    hyb = stripped_hybridisation(mol) if hyb is None else hyb
    a = mol.GetAtomWithIdx(d)
    sym, hybrid = a.GetSymbol(), hyb.get(d)
    if hybrid == _SP and ligand_degree(a) != 1:
        return  # The terminal-sp wall cannot orient two opposing ligand-side substituents.
    window = _ORIENT_WALL.get((sym, hybrid))
    if window is None:  # estimators disagree or the class is uncalibrated (n < 6): the gate abstains, so does this
        return
    backbone = _backbone_targets(mol, d, donor_set, stripped=stripped)
    heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
    if (
        sym in ("C", "N")
        and hybrid == _SP2
        and len(heavy) == 2  # noqa: PLR2004  the centred window's math needs exactly two neighbours
        and not _is_chelated(mol, d, donor_set, metal)
    ):
        window = _centred_sp2_window(mol, d, heavy)
    for nb in a.GetNeighbors():
        z = nb.GetAtomicNum()
        if z in COORDINATION_METALS:  # the M-D bond is stripped by now, but the surrogate keeps the fiction:
            continue  # never wall M
        if nb.GetIdx() in donor_set:
            continue  # the polyhedron and chelate bite own D-M-D; an M-D-D wall contradicts that angle
        other = backbone.get(nb.GetIdx())
        if (
            other is not None
            and (mol.GetAtomWithIdx(other).GetSymbol(), hyb.get(other)) in _ORIENT_WALL
            and (_heavy_substituent_count(mol, other) < 3)  # noqa: PLR2004
        ):
            # R_pair, as in `donation_axis`; `d` is already known calibrated (the window above).
            if _heavy_substituent_count(mol, d) >= 3:  # noqa: PLR2004
                continue
        if z > 1 and sum(1 for x in donor_set if mol.GetBondBetweenAtoms(nb.GetIdx(), x) is not None) >= APEX_DONORS:
            continue  # a heavy APEX substituent (bonded to >= 2 donors): a geometrically forced bite apex
        cons.angles.setdefault((metal, d, nb.GetIdx()), window)


def coplanar_donor(mol, metal, d, donor_set, cons, *, hyb=None):
    """Cap an sp2 donor's metal at its own sp2 plane: the improper the stripped M-donor bond removed.

    Records a soft flat-bottomed dihedral cap (+/- `COPLANAR_CAP` deg about an in-plane well) in
    `cons.coplanar`, applied in bounds and FF. `inplane_sp2_donor` decides who qualifies. The plane comes
    from the donor's heavy-neighbour count:

    * one heavy neighbour (carboxylate O, thione S): a proper dihedral M-D-C-X against C's heaviest other
      heavy substituent, graph-undirected so it carries no syn/anti bias. A donor bridged straight to
      another donor of the same ring is `ring_hinge`'s case instead;
    * two heavy neighbours (amidate/imine N, aryl carbanion C): the metal's angle out of the donor's own
      X-D-Y plane, the quantity the census measured. It belongs to neither donor bond, so `Coplanar` reads it
      through both, and an N-aryl amidate's phenyl stays free to twist.

    No M-D-C angle wall is written here: `orient_donor`'s fold wall already pins M-D-X for every calibrated
    class, and the FF torsion alone holds an uncalibrated one.
    """
    hyb = stripped_hybridisation(mol) if hyb is None else hyb
    if not inplane_sp2_donor(mol, d, hyb):
        return  # only an sp2 donor has an in-plane sigma lone pair to hold the metal to
    a = mol.GetAtomWithIdx(d)
    heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
    if len(heavy) == 1:  # one heavy neighbour: proper dihedral against C's heaviest substituent
        if hyb.get(heavy[0]) != Chem.HybridizationType.SP2:  # the neighbour must itself be sp2 (a real plane to hold)
            return
        bridge = heavy[0]
        ref = max(
            (nb for nb in mol.GetAtomWithIdx(bridge).GetNeighbors() if nb.GetIdx() != d and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
            default=None,
        )
        if ref is not None:  # an aldehyde-O whose C carries only H gets none: no reference atom
            cons.coplanar.append((metal, d, bridge, ref.GetIdx(), None, COPLANAR_CAP))
    elif len(heavy) == 2:  # noqa: PLR2004  two heavy neighbours: improper M out of the donor's own X-D-Y plane
        cons.coplanar.append((metal, d, *sorted(heavy), _COPLANAR_ANCHOR, COPLANAR_CAP))


def ring_hinge(mol, metal, donor_set, cons):
    """Hold the metal in a small, fully conjugated chelate ring's plane: a hinge no single donor cap reaches.

    `coplanar_donor` pins one sp2 donor's own plane; a ring whose whole metal-free D...D' path conjugates is
    planar as a unit regardless of either donor's own hybridisation (a dithiocarbamate's sp2 S and a
    benzenedithiolate's sp3-by-estimator S hinge the same way once the ring pi system ties them together).
    Only 4- and 5-membered rings: a 6-ring (acac, nacnac, salen) folds and boats for real.

    One row per donor D of the ring: `(M, D, D', X, anchor, cap)`, the dihedral about the D-D' diagonal, where
    X is D's own ring neighbour and D' the ring's other donor. In a planar ring that diagonal splits the ring
    into two triangles, so M and X sit on opposite sides of it (`_COPLANAR_ANCHOR`, anti). D' is a co-donor,
    never a substituent `orient_donor` walls, so `Coplanar.dg_post` finds no M-D-D' angle window and skips
    the DG bound, leaving only the FF torsion.
    """
    stripped = _stripped_graph(mol)
    donors = sorted(donor_set)
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
            cons.coplanar.append((metal, d, other, path[1], _COPLANAR_ANCHOR, cap))
            cons.coplanar.append((metal, other, d, path[-2], _COPLANAR_ANCHOR, cap))
