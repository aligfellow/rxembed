"""Donor perception and the orientation holds the metal surrogate loses when it strips the M-donor bond.

The metal-stripped hybridisation ruler (`_stripped_hybridisation`) and fold-census predicates (`donation_axis`,
`inplane_sp2_donor`), plus the enforcement that puts those orientations back softly: `_orient_donor` (one
census wall per substituent) and `_coplanar_donor` (the sp2-plane cap). The
gate side (`coordination.donor_orientation` / `donor_fold`) reads the same ruler, so they cannot drift.
"""

from __future__ import annotations

import math

from rdkit import Chem

from .metal_core import COORDINATION_METALS, _bounds_matrix, _haptic_sites, ligand_degree
from .metal_distance import _APEX_DONORS
from .utils import remove_bond

_PT = Chem.GetPeriodicTable()

# --- the sp2-donor coplanarity cap (`_coplanar_donor`, `cons.coplanar`) -------------------------------
# An sp2 donor binds from an in-plane sigma lone pair, so the metal sits in its sp2 framework. The surrogate
# strips the M-donor bond and UFF's improper with it; this puts it back as a flat-bottomed dihedral window.
# `inplane_sp2_donor` decides who qualifies: sp2 alone, no conjugation test and no element list.
_COPLANAR_CAP = 45.0  # deg half-window off the anchor: clears the tmQM/Kulik census p95 of 40°. Never a
# point, which would annihilate the real scatter out to that tail.
_COPLANAR_ANCHOR = 180.0  # deg: the external sector of a donor with two direct substituents
_ONE_HEAVY, _TWO_HEAVY = 1, 2  # heavy neighbours select the plane: 1 -> proper dihedral, 2 -> improper

# --- the chelate-ring hinge (`_ring_hinge`, `cons.coplanar`) ----------------------------------------------
# A ring closed through the metal holds it in the ring plane wherever the metal-free donor-to-donor path is
# fully conjugated: the ring pi system, not any one donor's own hybridisation, pins the metal (whatever the
# donor's own class). Only small rings are rigid enough to hold this: a 6-ring (acac, nacnac, salen) folds
# and boats for real, so it gets no hinge. One row per donor D of the ring: (M, D, D', X, 180, cap), the
# dihedral about the D-D' diagonal of the ring quadrilateral/pentagon; in a planar ring, M and D's own ring
# neighbour X sit on opposite sides of that diagonal (anchor 180, anti).
_HINGE_CAP = {
    4: 23.0,  # deg: crystal census of 82 conjugated 4-membered chelate rings, max fold 22.8 deg
    5: 35.0,  # deg: crystal census of 191 conjugated 5-membered chelate rings, p99 fold 34.2 deg
}

# --- the donor-fold census: what `_orient_donor` enforces and `coordination.donor_orientation` gates ------
# Where a ligand points, not where its donors are: a flat-folded carbonyl keeps a perfect M-C distance. Keyed on
# (element, hyb) because hybridisation alone admits a folded carbonyl, and thiolate donates 23° off carboxylate.
# Crystallographic census percentiles, not a fitted model; a class with n<6 abstains.
_SP, _SP2, _SP3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
_FOLD_WINDOW = {  # deg (floor, ceiling) per (element, hyb); floor gates, ceiling reports
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
_FOLD_MEDIAN = {  # deg: census median per class; `fold` = max |M-D-X - median|, so it needs no reference structure
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
# false-flags a crystal (0/144) but sits 25-30° below typical donation, a dead band a fold hides in. Only the
# wall may tighten: a tighter gate would flag real crystals.
_FOLD_WALL_FLOOR = {cls: min(lo + 12.0, _FOLD_MEDIAN[cls] - 5.0) for cls, (lo, _hi) in _FOLD_WINDOW.items()}
# deg (floor, ceiling): the wall `_orient_donor` writes on every donor substituent, heavy or proton. One rule
# leans an acyl leg off the sphere, splays a slow-inverting P-H, and splays a folded amine's H. The sp rows are
# overridden tight: an sp donor is linear, so its wall sits at its median (~165), not a nitrile-bending 152.
_ORIENT_WALL = {cls: (floor, _FOLD_WINDOW[cls][1]) for cls, floor in _FOLD_WALL_FLOOR.items()}
_ORIENT_WALL[("N", _SP)] = _ORIENT_WALL[("C", _SP)] = (165.0, 180.0)
# deg: retain the centring width, but derive the external bisector from the ligand's native internal angle.
# A fixed 120-degree centre pushes a five-membered donor out of plane. A chelate already pins its donor axis.
_CENTRED_SP2_PAD = 6.0
_MAX_SIGMA = {  # sigma bonds a class can carry: more is a hypervalent / mis-perceived centre -> unknown
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
}
_CONJUGATING_LP = frozenset({7, 8})  # period-2 only: N/O planarise into an adjacent π system, a period-3 lone
# pair does not (PPh3 is pyramidal). Letting P/S conjugate would type every triarylphosphine sp2.
_PYRAMIDAL_SIGMA = 3
_LONE_PAIR_ELECTRONS = 2
_DOUBLE_BOND = 2.0
_CARBON = 6


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
        if atom.GetFormalCharge() < 0 and atom.GetDegree() >= _TWO_HEAVY:
            pi_anchor = any(
                bond.GetBondTypeAsDouble() >= _DOUBLE_BOND
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
    rw = Chem.RWMol(mol)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in COORDINATION_METALS:
            for nb in [n.GetIdx() for n in a.GetNeighbors()]:
                remove_bond(rw, a.GetIdx(), nb)  # index-stable: removing a bond never renumbers atoms
    stripped = rw.GetMol()
    # The disconnected metal is retained only to keep atom indices stable. Do not infer radicals on its
    # deliberately isolated formal charge; properties are also skipped because stripped donors are bare.
    flags = (
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    )
    Chem.SanitizeMol(stripped, flags, catchErrors=True)
    return stripped


def _stripped_hybridisation(mol) -> dict[int, Chem.HybridizationType]:
    """Return agreed hybridisation assignments on the metal-stripped graph.

    An atom is absent (unknown, never gated) when the typer and the π-count disagree, or when it is hypervalent
    for its class: abstaining beats a mis-typed fold, and a rising unknown count is a free perception-bug
    detector (``FoldReport.unknown``). A three-coordinate centre with a non-conjugating lone pair is pyramidal
    even when one bond is drawn double (the neutral ``S(=O)R2`` form of a sulfoxide). Keep an agreed sp centre
    as sp when it is not a metal-bound heteroatom donor with two ligand sigma bonds: two ligand neighbours do
    not define an sp2 plane, even in a strained ring, and a non-terminal sp carbon (an internal alkyne's pi
    cloud) donates without ever needing a lone pair. A metal-bound HETEROATOM sp centre with two ligand sigma
    bonds, though, is retyped sp2: its only bond to the metal is a sigma lone pair, and a genuine sp donor has
    used its sigma framework and both pi orbitals on those two sigma bonds, leaving no lone pair to donate
    (MOCQIE's N12: xyz2mol's cumulated ``O=[N+]=C`` Kekule form types N sp by both estimators, but the
    dative M-N bond needs a lone pair that linear form does not have; a bent, sp2 nitro-like resonance form
    does). Terminal-donor models separately require one ligand-side neighbour; hybridisation alone does not
    establish a donation axis, so a genuine sp donor (a nitrile N, a carbyne C) still needs exactly one.

    Expected holdout: a metal-bound ``[CH-]`` carbanion, which RDKit calls sp2 and the π-count sp3, both defensible;
    rxembed treats it as a pyramidal stereocentre through its chiral tag and signed-volume gate, not here.
    """
    stripped = _stripped_graph(mol)
    out: dict[int, Chem.HybridizationType] = {}
    for a in stripped.GetAtoms():
        rdkit_h, pi_h = a.GetHybridization(), _pi_hybridisation(a)
        # Count sigma bonds here. Fractional aromatic bond orders describe the shared pi system and must not
        # consume the period-3 donor's remaining lone pair a second time.
        nonbonding = _PT.GetNOuterElecs(a.GetAtomicNum()) - a.GetFormalCharge() - a.GetDegree()
        if (
            pi_h == Chem.HybridizationType.SP2
            and a.GetDegree() == _PYRAMIDAL_SIGMA
            and a.GetAtomicNum() not in _CONJUGATING_LP
            and nonbonding >= _LONE_PAIR_ELECTRONS
        ):
            rdkit_h = pi_h = Chem.HybridizationType.SP3
        # RDKit types a ring-conjugated donor (an aryl thiolate S) sp2 from its neighbour's aromaticity,
        # not its own lone pair. A period-2 lone pair can planarise into that system (_CONJUGATING_LP);
        # a period-3+ one does not (FISCIT: an unwalled Tc-S-C folded 12 -> 40-76 deg). This keeps the
        # carbanion holdout above: carbon is period 2, so its sp2/sp3 disagreement still abstains.
        elif rdkit_h == _SP2 and pi_h == _SP3 and _PT.GetRow(a.GetAtomicNum()) > 2:  # noqa: PLR2004
            rdkit_h = _SP3
        # A metal-bound HETEROATOM sp centre with 2 ligand sigma bonds cannot really be sp: its only bond to
        # the metal is a sigma lone pair, and a genuinely cumulated (2 sigma + 2 pi) centre has spent every
        # valence electron on its ligand side, leaving none to donate. Type it sp2, the bent resonance form a
        # cumulated Kekule perception missed. Carbon is excluded: it has no lone pair at sp regardless of
        # charge (bond order 4 already exceeds even a carbanion's 5-electron budget by one), so a non-terminal
        # sp carbon donor (an internal alkyne's pi cloud, not a lone pair) is not this contradiction and stays
        # sp. A genuine terminal sp donor (nitrile N, carbyne C) has exactly one ligand sigma bond either way.
        elif (
            rdkit_h == pi_h == _SP
            and a.GetDegree() == _TWO_HEAVY
            and a.GetAtomicNum() != _CARBON
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
    """Return True when donor ``d`` binds from an in-plane sigma lone pair: the coplanarity-cap predicate.

    An sp2 donor's sigma framework lies in one plane with its π orbital perpendicular, so a sigma-bound metal sits
    in that plane whether or not the π system is *conjugated*. The predicate is therefore exactly ``sp2`` (the
    two-estimator ``_stripped_hybridisation`` class the fold gate reads), element-agnostic (an aryl carbanion C
    and a thione S qualify) and with NO conjugation test: an *isolated* ketone/imine C=X is sp2 in-plane, yet
    RDKit marks its bond non-conjugated, so a conjugation test wrongly dropped the cap it needs. The only sp2
    donor without an in-plane lone pair is a pure π-donor (a haptic carbon), which is one centroid vertex and
    never reaches the cap. The period-2 restriction that IS physics lives on ``_pi_hybridisation``.
    """
    if hyb is None:
        hyb = _stripped_hybridisation(mol)
    return hyb.get(d) == Chem.HybridizationType.SP2


def donation_axis(mol, d, all_donors, sphere=None, *, hyb=None, network=True) -> list[int] | None:
    """Return donor ``d``'s judgeable heavy substituents X, or ``None`` when it donates along no axis.

    The M-D-X question is only meaningful for a supported donation axis pointed at one metal:

    * an H donor (hydride, sigma-complex, agostic) has no lone pair;
    * a bridging donor (2 or more metals) has its axis set by the bridge;
    * a haptic donor, bonded to a co-donor (side-on η² alkene, η-n ring), donates a π face, so the metal
      sits ~70° off any M-D-X axis.
    * a nonterminal sp atom has no supported end-on axis; two opposing substituents cannot both face away
      from the metal under the terminal-sp model.

    The abstention is about the donor, so both the coordinate-space ruler (``_donor_walk``) and the enumerator's
    screen (``_donor_faces_metal``) read it here. Returns a possibly-empty list when every substituent is
    itself exempt (a co-donor, a κ2 bite bridgehead, a proton, or a metal): "ask, but nothing to measure",
    distinct from the ``None`` that means "do not ask". Freeze ownership is not chemistry and is applied only
    after the complete M-D-X term is known.

    ``sphere`` is this metal's own donors for the bite-bridgehead test; it defaults to ``all_donors``. ``network``
    removes a backbone arm only when the donor has two independent local substituents. Enumeration can disable
    that refinement for its conservative outer-bound screen.
    """
    a = mol.GetAtomWithIdx(d)
    if a.GetAtomicNum() == 1:  # hydride / η²-H₂ / agostic H: no lone pair, so no donation axis
        return None
    if sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() in COORDINATION_METALS) > 1:  # bridging: set by the bridge
        return None
    sphere = all_donors if sphere is None else sphere
    if any(d in site and len(site) > 1 for site in _haptic_sites(mol, sphere)):  # pi-face: metal is off-axis
        return None
    hyb = _stripped_hybridisation(mol) if hyb is None else hyb
    if hyb.get(d) == _SP and ligand_degree(a) != 1:
        return None
    backbone = _backbone_targets(mol, d, sphere) if network else {}
    return [
        nb.GetIdx()
        for nb in a.GetNeighbors()
        if nb.GetAtomicNum() > 1  # protons have their own window (`_orient_donor`)
        and nb.GetAtomicNum() not in COORDINATION_METALS
        and nb.GetIdx() not in all_donors  # co-donor
        # R_pair: a backbone arm is not an independent donation axis, but only when its co-donor is
        # itself calibrated and small enough to hold its own arm (else this arm is the co-donor's only hold).
        # A ring of two such small donors (a symmetric S,S chelate) must keep one walled arm: exempt
        # neither when this donor would equally qualify for exemption from the co-donor's own side.
        and not (
            (other := backbone.get(nb.GetIdx())) is not None
            and (mol.GetAtomWithIdx(other).GetSymbol(), hyb.get(other)) in _ORIENT_WALL
            and _heavy_substituent_count(mol, other) < _TWO_HEAVY + 1
            and not (
                (mol.GetAtomWithIdx(d).GetSymbol(), hyb.get(d)) in _ORIENT_WALL
                and _heavy_substituent_count(mol, d) < _TWO_HEAVY + 1
            )
        )
        # A chelate bridgehead has no independent M-D-X angle even though it still needs anti-collapse repulsion.
        and sum(1 for x in sphere if mol.GetBondBetweenAtoms(nb.GetIdx(), int(x)) is not None) < _APEX_DONORS
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


def _backbone_targets(mol, d, donor_set):
    """Map each of donor ``d``'s substituents that reaches another donor through the ligand backbone to it.

    Remove coordination edges before finding ligand paths, because public relaxed molecules restore those
    edges and a path through the metal is not a ligand backbone. A graph fact only; whether an arm's wall
    is exempted is the caller's policy (R_pair: is the co-donor at the far end itself independently held).
    In a macrocycle a single arm can reach two donors nested one past the other (a bridging donor between
    two chelate rings); keep the nearer one, the arm's own chelate partner, not one further down the ring.
    """
    network = Chem.RWMol(mol)
    for bond in list(mol.GetBonds()):
        if any(atom.GetAtomicNum() in COORDINATION_METALS for atom in (bond.GetBeginAtom(), bond.GetEndAtom())):
            network.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    network = network.GetMol()
    network.ClearComputedProps()
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
    matrix = _bounds_matrix(mol)
    a, b, c = (0.5 * (matrix[i, j] + matrix[j, i]) for i, j in ((left, donor), (right, donor), (left, right)))
    cosine = (a * a + b * b - c * c) / (2.0 * a * b)
    internal = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    centre = 180.0 - 0.5 * internal
    return centre - _CENTRED_SP2_PAD, min(180.0, centre + _CENTRED_SP2_PAD)


def _orient_donor(mol, metal, d, donor_set, cons, *, hyb=None):
    """Wall every donor substituent, heavy or proton, off the metal: the M-D-X bend the stripped bond lost.

    One flat-bottomed angle wall per non-metal, non-apex substituent, keyed on the donor's (element, hyb) census
    class (`_ORIENT_WALL`), written into ``cons.angles`` so it biases both the bounds matrix and the FF (they must
    agree or they fight, and a DG-only wall measures worse than none). This one loop subsumes the three
    holds the surrogate used to need separately (sp end-on, pnictogen proton splay, heavy-substituent fold); a
    PROTON is a substituent too, which is the fix for an sp3 amine folding an H onto the metal over its lone pair.

    An uncalibrated class (estimators disagree / n<6) abstains, exactly as the gate does. A heavy APEX
    substituent (bonded to >= 2 donors) is a geometrically forced bite apex, never walled. Freeze ownership is
    resolved after the complete term exists. The out-of-plane coplanarity is the sibling `_coplanar_donor`.
    """
    hyb = _stripped_hybridisation(mol) if hyb is None else hyb
    a = mol.GetAtomWithIdx(d)
    sym, hybrid = a.GetSymbol(), hyb.get(d)
    if hybrid == _SP and ligand_degree(a) != 1:
        return  # The terminal-sp wall cannot orient two opposing ligand-side substituents.
    window = _ORIENT_WALL.get((sym, hybrid))
    if window is None:  # estimators disagree or the class is uncalibrated (n < 6): the gate abstains, so does this
        return
    backbone = _backbone_targets(mol, d, donor_set)
    heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
    if sym in ("C", "N") and hybrid == _SP2 and len(heavy) == _TWO_HEAVY and not _is_chelated(mol, d, donor_set, metal):
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
            and (_heavy_substituent_count(mol, other) < _TWO_HEAVY + 1)
        ):
            # R_pair: the ligand path owns this arm, but only when its co-donor is itself calibrated and
            # small enough to hold its own arm; else that co-donor's own arm would go unwalled instead.
            # `d` is already known calibrated (window above); a ring of two equally small donors (a
            # symmetric S,S chelate) must keep one walled arm, so exempt neither in that case.
            if _heavy_substituent_count(mol, d) >= _TWO_HEAVY + 1:
                continue
        if z > 1 and sum(1 for x in donor_set if mol.GetBondBetweenAtoms(nb.GetIdx(), x) is not None) >= _APEX_DONORS:
            continue  # a heavy APEX substituent (bonded to >= 2 donors): a geometrically forced bite apex
        cons.angles.setdefault((metal, d, nb.GetIdx()), window)


def _coplanar_donor(mol, metal, d, donor_set, cons, *, hyb=None):
    """Cap an sp2 donor's metal at its own sp2 plane: the improper the stripped M-donor bond removed.

    Records a soft flat-bottomed dihedral cap (±`_COPLANAR_CAP`° about an in-plane well) in ``cons.coplanar``,
    applied in bounds and FF. The plane is picked by the donor's heavy-neighbour count:

    * one heavy neighbour (carboxylate O, thione S): proper dihedral M-D-C-X against C's heaviest other heavy
      substituent. A donor bridged straight to another donor of the same ring is `_ring_hinge`'s case, not
      this one: this row's well is graph-undirected (seed-selected), so it is not a substitute for that cap.
    * two heavy neighbours (amidate/imine N, aryl carbanion C): improper M-D-X-Y of the donor's own direct
      substituents, so an N-aryl amidate's phenyl stays free to twist.

    `inplane_sp2_donor` decides who qualifies. Two direct substituents define the external lone-pair sector as
    anti, independent of their order. A one-heavy proper row cannot distinguish syn from anti from the graph,
    so its well remains seed-selected and contributes no directional DG 1,4 bias.

    NB no M-D-C angle wall is written here: the DG half needs M-D-X pinned, but `_orient_donor`'s fold wall
    already pins it for every calibrated class, and an uncalibrated one is held by the FF torsion alone.
    """
    hyb = _stripped_hybridisation(mol) if hyb is None else hyb
    if not inplane_sp2_donor(mol, d, hyb):
        return  # only an sp2 donor has an in-plane sigma lone pair to hold the metal to
    a = mol.GetAtomWithIdx(d)
    heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
    if len(heavy) == _ONE_HEAVY:  # (perm 1) proper dihedral M-D-C-X against C's heaviest other heavy substituent.
        if hyb.get(heavy[0]) != Chem.HybridizationType.SP2:  # the neighbour must itself be sp2 (a real plane to hold)
            return
        bridge = heavy[0]
        ref = max(
            (nb for nb in mol.GetAtomWithIdx(bridge).GetNeighbors() if nb.GetIdx() != d and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
            default=None,
        )
        if ref is not None:  # an aldehyde-O whose C carries only H gets none: no reference atom
            cons.coplanar.append((metal, d, bridge, ref.GetIdx(), None, _COPLANAR_CAP))
    elif len(heavy) == _TWO_HEAVY:  # (perm 2) improper M out of the donor's own X-D-Y plane (direct substituents),
        cons.coplanar.append((metal, d, heavy[0], heavy[1], _COPLANAR_ANCHOR, _COPLANAR_CAP))  # a single dihedral


def _ring_hinge(mol, metal, donor_set, cons):
    """Hold the metal in a small, fully conjugated chelate ring's plane: a hinge no single donor cap reaches.

    `_coplanar_donor` only pins one sp2 donor's own plane; a ring whose WHOLE metal-free D...D' path
    conjugates is planar as a unit, whatever either donor's own hybridisation (a dithiocarbamate's sp2 S and
    a benzenedithiolate's sp3-by-estimator S both hinge the same way once the ring pi system ties them
    together). Only 4- and 5-membered rings: a 6-ring (acac, nacnac, salen) folds and boats for real.

    One row per donor D of the ring: ``(M, D, D', X, anchor, cap)``, the dihedral about the D-D' diagonal,
    where X is D's own ring neighbour and D' the ring's other donor. In a planar ring the diagonal D-D' splits
    the quadrilateral/pentagon M-D-...-D'-M into two triangles, so M and X sit on opposite sides of it
    (`_COPLANAR_ANCHOR`, anti). D' is a co-donor, never a substituent `_orient_donor` walls, so `Coplanar.
    _dg_post` finds no M-D-D' angle window and skips the DG bound on its own, leaving only the FF torsion.
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
