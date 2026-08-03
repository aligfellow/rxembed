"""Donor perception and the orientation holds the metal surrogate loses when it strips the M-donor bond.

The metal-stripped hybridisation ruler (`_stripped_hybridisation`) and the fold-census predicates
(`donation_axis`, `inplane_sp2_donor`, `codonor_in_plane`), plus the enforcement that puts those orientations
back softly: `_orient_donor` (one census wall per substituent) and `_coplanar_donor` (the sp2-plane cap). The
gate side (`coordination.donor_orientation` / `donor_fold`) reads the same ruler, so they cannot drift.
"""

from __future__ import annotations

from rdkit import Chem

from .metal_core import _METAL_Z
from .metal_distance import _APEX_DONORS, APEX, overbond_tier
from .utils import remove_bond

# --- the sp2-donor coplanarity cap (`_coplanar_donor`, `cons.coplanar`) -------------------------------
# An sp2 donor binds from an in-plane sigma lone pair, so the metal sits in its sp2 framework. The surrogate
# strips the M-donor bond and UFF's improper with it; this puts it back as a flat-bottomed dihedral window.
# `inplane_sp2_donor` decides who qualifies: sp2 alone, no conjugation test and no element list.
_COPLANAR_CAP = 45.0  # deg half-window off the anchor: clears the tmQM/Kulik census p95 of 40°. Never a
# point, which would annihilate the real scatter out to that tail.
_COPLANAR_ANCHOR = 180.0  # deg: the anti in-plane well a κ1 donor binds in; the FF re-detects it per conformer
_ONE_HEAVY, _TWO_HEAVY = 1, 2  # heavy neighbours select the plane: 1 -> proper dihedral, 2 -> improper

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
    ("S", _SP3): (91.0, 137.1),
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
# deg: the centred window for a monodentate two-heavy sp2 C/N donor, whose lone-pair axis has no co-donor to
# pin it and so needs centring on the substituents' external bisector (120°). A chelate's ring pins it already.
_CENTRED_SP2_WINDOW = (114.0, 126.0)
_MAX_SIGMA = {  # sigma bonds a class can carry: more is a hypervalent / mis-perceived centre -> unknown
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
}
_CONJUGATING_LP = frozenset({7, 8})  # period-2 only: N/O planarise into an adjacent π system, a period-3 lone
# pair does not (PPh3 is pyramidal). Letting P/S conjugate would type every triarylphosphine sp2.


# --- donor perception: the metal-stripped hybridisation ruler + the fold-census predicates ------------
# Enforcement and gate must name the same donor class, or a wall-biased seed gets flagged the other way. Hence
# one definition here, which the gate imports.


def _pi_hybridisation(atom) -> Chem.HybridizationType | None:
    """Estimator B: hybridisation from a π-count, independent of RDKit's typer.

    Two π bonds = sp; one = sp2; none = sp3, unless a period-2 lone pair conjugates into an adjacent π system
    (amide N, carboxylate O), planarising it to sp2. Where the two estimators part company (a carbanion, ylide,
    hypervalent S, an arbitrary Kekulé form) the donor is unknown and gates nothing.
    """
    if atom.GetIsAromatic():
        return Chem.HybridizationType.SP2
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
    if atom.GetAtomicNum() in _CONJUGATING_LP:
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic() or any(b.GetBondTypeAsDouble() >= 2 for b in nb.GetBonds()):  # noqa: PLR2004
                return Chem.HybridizationType.SP2
    return Chem.HybridizationType.SP3


def _stripped_hybridisation(mol) -> dict[int, Chem.HybridizationType]:
    """``{atom: hybridisation}`` for atoms two independent estimators agree on, on the metal-stripped graph.

    The metal must be stripped first: RDKit counts the dative bond, so a metal-bound donor is mis-typed by the
    coordination being judged (a κ1-alkoxide O goes degree-1 → degree-2, class flips). The surrogate already
    strips these; this rebuilds that graph.

    An atom is absent (unknown, never gated) when the typer and the π-count disagree, or when it is hypervalent
    for its class: abstaining beats a mis-typed fold, and a rising unknown count is a free perception-bug
    detector (``FoldReport.unknown``). The one class it *corrects* is an "sp" centre with two substituents: sp
    is linear, so a second substituent proves it bent and the sp a spurious-triple-bond artefact (a formyl
    H-C=O read C≡O), and is re-read sp2 so the acyl still earns its fold wall.

    Expected holdout: a metal-bound ``[CH-]`` carbanion, which RDKit calls sp2 and the π-count sp3, both defensible;
    rxembed treats it as a pyramidal stereocentre (``_hold_donor_chirality``), not gated.
    """
    rw = Chem.RWMol(mol)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in _METAL_Z:
            for nb in [n.GetIdx() for n in a.GetNeighbors()]:
                remove_bond(rw, a.GetIdx(), nb)  # index-stable: removing a bond never renumbers atoms
    stripped = rw.GetMol()
    Chem.SanitizeMol(  # properties (valence) are not enforced: a stripped donor is a bare anion/lone pair
        stripped, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
    )
    out: dict[int, Chem.HybridizationType] = {}
    for a in stripped.GetAtoms():
        rdkit_h, pi_h = a.GetHybridization(), _pi_hybridisation(a)
        if rdkit_h != pi_h or rdkit_h not in _MAX_SIGMA:  # the estimators disagree, or it is not sp/sp2/sp3
            continue
        if rdkit_h == Chem.HybridizationType.SP and a.GetDegree() != 1:  # a bent "sp": a spurious triple bond
            # (a formyl/acyl H-C=O read C≡O). Re-read sp2, not abstained, so the acyl still earns its fold wall.
            rdkit_h = Chem.HybridizationType.SP2
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


def codonor_in_plane(mol, d, donors, hyb=None) -> bool:
    """Return True when a co-donor of the same metal lies in donor ``d``'s own conjugated sp2 plane.

    A plane is pinned to the metal by TWO coplanar contacts, never one, so the coplanarity cap's improper is real
    information only where the metal meets the donor's plane at a SINGLE point. When a second donor of the same
    metal lies in that same conjugated sp2 plane, as in a conjugated bidentate (pyridylimine, acac), that pair
    pins the metal, so the improper is redundant and `_coplanar_donor`'s FF torsion is skipped for it.

    "Same conjugated sp2 plane" is a co-donor reachable through a backbone path whose every atom is sp2 (reading
    only ``_stripped_hybridisation``). A path crossing an sp3 hinge (the isomer's O-C-C(sp3)-N) is not all-sp2, so one
    contact, cap kept. Keys on the *second* donor's coplanarity, not the donor's own local sp2-ness.
    """
    if hyb is None:
        hyb = _stripped_hybridisation(mol)
    for dd in donors:
        if dd == d:
            continue
        path = Chem.GetShortestPath(mol, int(d), int(dd))  # empty for a co-donor on a separate ligand -> not locked
        if path and all(hyb.get(a) == Chem.HybridizationType.SP2 for a in path):
            return True
    return False


def donation_axis(mol, d, all_donors, sphere=None, frozen=frozenset()) -> list[int] | None:
    """Return donor ``d``'s judgeable heavy substituents X, or ``None`` when it donates along no axis.

    The M-D-X question is only meaningful for a donor with one lone-pair axis pointed at one metal. Three donors
    have none, so any M-D-X angle is a number about nothing:

    * an H donor (hydride, sigma-complex, agostic) has no lone pair;
    * a bridging donor (2 or more metals) has its axis set by the bridge;
    * a haptic donor, bonded to a co-donor (side-on η² alkene, η-n ring), donates a π face, so the metal
      sits ~70° off any M-D-X axis.

    The abstention is about the donor, so both the coordinate-space ruler (``_donor_walk``) and the enumerator's
    screen (``_donor_faces_metal``) read it here. Returns a possibly-empty list when every substituent is
    itself exempt (a co-donor, a κ2 bite ``APEX``, a proton, a metal, or a frozen-core D-X): "ask, but nothing
    to measure", distinct from the ``None`` that means "do not ask".

    ``sphere`` is this metal's own donors (the ``APEX`` bite test); it defaults to ``all_donors``.
    """
    a = mol.GetAtomWithIdx(d)
    if a.GetAtomicNum() == 1:  # hydride / η²-H₂ / agostic H: no lone pair, so no donation axis
        return None
    if sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() in _METAL_Z) > 1:  # bridging: set by the bridge
        return None
    if any(nb.GetIdx() in all_donors for nb in a.GetNeighbors()):  # haptic: side-on, the metal is off-axis
        return None
    sphere = all_donors if sphere is None else sphere
    return [
        nb.GetIdx()
        for nb in a.GetNeighbors()
        if nb.GetAtomicNum() > 1  # protons have their own window (`_orient_donor`)
        and nb.GetAtomicNum() not in _METAL_Z
        and nb.GetIdx() not in all_donors  # co-donor
        and overbond_tier(mol, sphere, nb.GetIdx()) != APEX  # a κ2 bite apex is forced by its ring
        and not (d in frozen and nb.GetIdx() in frozen)  # the frozen core's own orientation, grafted from the TS
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


def _orient_donor(mol, metal, d, donor_set, cons, core_frozen=()):
    """Wall every donor substituent, heavy or proton, off the metal: the M-D-X bend the stripped bond lost.

    One flat-bottomed angle wall per non-metal, non-apex substituent, keyed on the donor's (element, hyb) census
    class (`_ORIENT_WALL`), written into ``cons.angles`` so it biases both the bounds matrix and the FF (they must
    agree or they fight, and a DG-only wall measures worse than none). This one loop subsumes the three
    holds the surrogate used to need separately (sp end-on, pnictogen proton splay, heavy-substituent fold); a
    PROTON is a substituent too, which is the fix for an sp3 amine folding an H onto the metal over its lone pair.

    Skipped for a ``fix=`` TS-core donor (``core_frozen``, whose orientation is the reference's) and an uncalibrated
    class (estimators disagree / n<6), exactly as the gate abstains. A HEAVY APEX substituent (bonded to >= 2
    donors) is a geometrically forced bite apex, never walled. The OUT-of-plane coplanarity is the sibling
    `_coplanar_donor`; this covers the IN-plane / axial. The class reads `_stripped_hybridisation`, so a bent acyl
    read "sp" is re-read sp2 and earns the sp2 fold wall.

    The skip here is the DONOR's alone; whether a frozen METAL also silences it is the caller's call, since it
    turns on whether the input geometry is being trusted (`coordination.coordination`).
    """
    a = mol.GetAtomWithIdx(d)
    if d in core_frozen:  # a fix= TS core grafts this donor's orientation
        return
    sym, hyb = a.GetSymbol(), _stripped_hybridisation(mol).get(d)
    window = _ORIENT_WALL.get((sym, hyb))
    if window is None:  # estimators disagree or the class is uncalibrated (n < 6): the gate abstains, so does this
        return
    heavy = sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1)
    if sym in ("C", "N") and hyb == _SP2 and heavy == _TWO_HEAVY and not _is_chelated(mol, d, donor_set, metal):
        window = _CENTRED_SP2_WINDOW  # monodentate sp2 C/N: centre the in-plane axis (see `_CENTRED_SP2_WINDOW`)
    for nb in a.GetNeighbors():
        z = nb.GetAtomicNum()
        if z in _METAL_Z:  # the M-D bond is stripped by now, but the surrogate keeps the fiction: never wall M
            continue
        if z > 1 and sum(1 for x in donor_set if mol.GetBondBetweenAtoms(nb.GetIdx(), x) is not None) >= _APEX_DONORS:
            continue  # a heavy APEX substituent (bonded to >= 2 donors): a geometrically forced bite apex
        cons.angles.setdefault((metal, d, nb.GetIdx()), window)


def _coplanar_donor(mol, metal, d, cons):
    """Cap an sp2 donor's metal at its own sp2 plane: the improper the stripped M-donor bond removed.

    Records a soft flat-bottomed dihedral cap (±`_COPLANAR_CAP`° about the anchor) in ``cons.coplanar``,
    applied in bounds and FF. The plane is picked by the donor's heavy-neighbour count:

    * one heavy neighbour (carboxylate O, thione S): proper dihedral M-D-C-X against C's heaviest other
      substituent. One reference, not one per substituent: C's substituents sit ~180° apart in this dihedral,
      so a second entry adds nothing and only doubles the fc, overriding a co-donor's plane.
    * two heavy neighbours (amidate/imine N, aryl carbanion C): improper M-D-X-Y of the donor's own direct
      substituents, so an N-aryl amidate's phenyl stays free to twist.

    `inplane_sp2_donor` decides who qualifies. The anchor is the structural default (anti) and the FF
    re-detects syn vs anti per conformer.

    NB no M-D-C angle wall is written here: the DG half needs M-D-X pinned, but `_orient_donor`'s fold wall
    already pins it for every calibrated class, and an uncalibrated one is held by the FF torsion alone.
    """
    hyb = _stripped_hybridisation(mol)
    if not inplane_sp2_donor(mol, d, hyb):
        return  # only an sp2 donor has an in-plane sigma lone pair to hold the metal to
    a = mol.GetAtomWithIdx(d)
    heavy = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
    if len(heavy) == _ONE_HEAVY:  # (perm 1) proper dihedral M-D-C-X against C's heaviest other heavy substituent.
        if hyb.get(heavy[0]) != Chem.HybridizationType.SP2:  # the neighbour must itself be sp2 (a real plane to hold)
            return
        # one reference atom, the heaviest (see docstring): a κ1 carboxylate binds anti to its C=O partner, so the
        # 2nd O is the anchor the DG floor keys off; the FF re-detects the well per conformer.
        ref = max(
            (nb for nb in mol.GetAtomWithIdx(heavy[0]).GetNeighbors() if nb.GetIdx() != d and nb.GetAtomicNum() > 1),
            key=lambda nb: nb.GetAtomicNum(),
            default=None,
        )
        if ref is not None:  # an aldehyde-O whose C carries only H gets none: no reference atom
            cons.coplanar.append((metal, d, heavy[0], ref.GetIdx(), _COPLANAR_ANCHOR, _COPLANAR_CAP))
    elif len(heavy) == _TWO_HEAVY:  # (perm 2) improper M out of the donor's own X-D-Y plane (direct substituents),
        cons.coplanar.append((metal, d, heavy[0], heavy[1], _COPLANAR_ANCHOR, _COPLANAR_CAP))  # a single dihedral
