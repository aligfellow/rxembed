"""Read an ``.xyz`` into an RDKit Mol: the one input adaptation that needs perception.

Input adaptation only, ahead of the core `embed()` engine; a SMILES needs none of this (`rxembed.metal_smiles`
instead). Connectivity and bond order are separate arguments: RDKit, xyzgraph, and xyz2mol can supply
connectivity, but only the latter two rank bond orders.
"""

from __future__ import annotations

import contextlib
import itertools
import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from rxembed.metal_core import (
    COORDINATION_METALS,
    metal_indices,
    reject_boron_cages,
)
from rxembed.stereo import stereo_from_3d
from rxembed.utils import flat_ranks, hydrogen_bond, hydrogen_neighbor_order, lone_pair_electrons, remove_bond

from .xyz2mol_tmc import SEEDED_STRUCTURAL_CHARGES, TRANSITION_METALS_NUM, get_tmc_mol

_PT = Chem.GetPeriodicTable()

_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic
_BRIDGE_H_FACTOR = 1.2  # x (r_cov(H) + r_cov(X)); same margin as a new-bond distance elsewhere in the pipeline
_CONNECTIVITY = {"rdkit", "xyzgraph", "xyz2mol"}
_BOND_ORDERS = {"xyzgraph", "xyz2mol"}
_BRIDGEHEAD_SIGMA_MIN = 4  # a kappa2 chelate bridgehead (P, Si, B) bonds >=4 non-metal sigma neighbours
_BRIDGEHEAD_DONORS_MIN = 2  # fewer is a sigma-silane/borane bridgehead (one metal-bound neighbour), not this rule
_TRIGONAL_SIGMA = 3  # fewer, with no lone pair: an sp X keeps a pi orbital in the chelate plane, so the fold test fails
_OPEN_FACE_Z = 6  # an open D-X-C path can be an eta3 allyl-type face, so a carbon donor keeps an open X bound
# Fold of a ring's one unbound atom out of its bound ring-mates' plane, tmQMg census (2026-09-27, 47 rings bound
# through all atoms but one): a reader-dropped atom folds 0.7-7.3 deg and a bonded ring B 2.5-9.7; a slipped ring B
# folds 13.8, a genuinely unbound flap (cyclopentadienone C, phosphole P, thiophenium S, sp3 ring C) 19.2 or more.
_FACE_FLAP_MAX = 10.0  # deg
logger = logging.getLogger("rxembed")


def read_xyz(path, charge=0, connectivity="xyzgraph", bond_orders="xyzgraph", metal_charges=None, fallback=True):
    """Read an ``.xyz`` into a Mol with perceived bonds and a conformer; handles metals and TSs.

    `connectivity` picks RDKit's connect-the-dots, ``"xyzgraph"``, or ``"xyz2mol"``. `bond_orders` is xyzgraph's
    optimiser (needs ``connectivity="xyzgraph"``) or xyz2mol's ranked charge search (metal complexes only; a
    metal-free TS keeps xyzgraph's orders instead of failing); RDKit connectivity never adds or removes a metal-donor
    contact. ``metal_charges`` names a formal charge per metal index for a multi-metal XYZ whose total does not
    determine the split; every metal must be named. Sanitisation is lenient (rings yes, valence checks no); perceiver
    choices land on the returned Mol's ``_rxembed*`` properties; ``_rxembedChargeRescue`` says how a metal read
    past its valence electrons was read instead (`_assign_bond_orders`), empty when none was. A bridgehead M-X
    bond a reader mis-perceived (`_drop_bridgehead_bonds`) is dropped, and a face atom it left out of an
    otherwise bound ring (`_bond_flat_face_atoms`) is bonded, here, once, since only a coordinate-derived
    graph can carry either fault.
    """
    if connectivity not in _CONNECTIVITY:
        raise ValueError(f"connectivity must be 'rdkit', 'xyzgraph', or 'xyz2mol', got {connectivity!r}")
    if bond_orders not in _BOND_ORDERS:
        raise ValueError(f"bond_orders must be 'xyzgraph' or 'xyz2mol', got {bond_orders!r}")
    if bond_orders == "xyzgraph" and connectivity != "xyzgraph":
        raise ValueError(
            "bond_orders='xyzgraph' needs connectivity='xyzgraph': that optimiser runs inside "
            "build_graph and cannot be given another perceiver's bonds. Use bond_orders='xyz2mol'."
        )
    mol, perceived_by = _with_fallback(path, charge, connectivity, fallback)
    reject_boron_cages(mol)  # before xyz2mol forces multi-centre B-H/B-B into valence-two orders
    added = ()
    if fallback and perceived_by == "xyzgraph" and metal_indices(mol):
        try:
            native = _from_rdkit_connectivity(path, charge)
            _assert_same_atoms(mol, native)
        except (OSError, RuntimeError, ValueError):
            pass  # no independent native verdict means the requested graph remains authoritative
        else:
            mol, added = _restore_consensus_ligand_bonds(mol, native, path, charge)
            if added:
                logger.warning(
                    "read_xyz: added %s internal ligand bonds confirmed by RDKit and xyz2mol",
                    ", ".join(f"{i}-{j}" for i, j in added),
                )
    mol = _bond_flat_face_atoms(mol)
    used_fallback = perceived_by != connectivity
    supplied = _apply_metal_charges(mol, metal_charges)
    selected_connectivity = perceived_by
    mol, perceived_by, order_by, order_fallback = _assign_bond_orders(
        mol, path, charge, perceived_by, bond_orders, fallback
    )
    # After bond orders settle, not just after the requested connectivity: xyz2mol's own metal-free
    # fallback graph (not xyzgraph's) is what orphans a 3c-2e bridge into a free hydride for a structure
    # like B3H8-, and only the post-order mol shows it.
    _reject_unbonded_bridging_hydrogens(mol)
    mol = _drop_bridgehead_bonds(mol)
    used_fallback |= order_fallback
    if perceived_by != selected_connectivity:
        added = ()
    if supplied is not None:
        for index, value in supplied.items():
            mol.GetAtomWithIdx(index).SetFormalCharge(value)
    perceived_charge = Chem.GetFormalCharge(mol)
    if perceived_charge != charge:
        remedy = "use bond_orders='xyz2mol'" if bond_orders == "xyzgraph" else "check charge= and the input graph"
        raise ValueError(f"perceived total charge {perceived_charge} does not match charge={charge}; {remedy}")
    rescue = mol.GetProp("_rxembedChargeRescue") if mol.HasProp("_rxembedChargeRescue") else ""
    if rescue:
        logger.warning("read_xyz: %s", rescue)
    mol.SetProp("_rxembedConnectivity", perceived_by)
    mol.SetProp("_rxembedBondOrders", order_by)
    mol.SetProp("_rxembedConnectivityAdded", ",".join(f"{i}-{j}" for i, j in added))
    mol.SetProp("_rxembedChargeRescue", rescue)
    mol.SetBoolProp("_rxembedPerceptionFallback", used_fallback or order_by != bond_orders)
    _warn_if_charge_looks_missing(mol, charge)
    try:
        with rdBase.BlockLogs():
            stereo_from_3d(mol, metal_indices(mol), apply=True)
    except (RuntimeError, ValueError) as exc:
        logger.warning("read_xyz: could not assign input stereo (%s); keeping the perceived graph", exc)
    return mol


def _reject_unbonded_bridging_hydrogens(mol):
    """Reject a hydrogen with no perceived bond that geometrically bridges two or more heavy atoms.

    A connectivity backend's two-centre bond search can leave a 3c-2e bridging hydrogen (B-H-B, Al-H-Al,
    M-H-M) with zero neighbours instead of choosing one leg; it then survives as a free hydride and
    xyz2mol forces a nonsense two-centre graph around it, surfacing as a confusing failure five stages
    later. Geometry- and graph-derived only (no element list, no atom count), so this also catches a
    bridge `reject_boron_cages`'s cage-vertex rule cannot: a single B-H-B pair, or a non-boron bridge. A
    dropped terminal hydrogen sits near one heavy atom, not two; a free hydride counterion sits outside
    the covalent-radius reach of anything.
    """
    if mol.GetNumConformers() == 0:
        return
    pos = mol.GetConformer().GetPositions()
    r_h = _PT.GetRcovalent(1)
    bridges = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 0:
            continue
        h = atom.GetIdx()
        near = []
        for other in mol.GetAtoms():
            if other.GetIdx() == h or other.GetAtomicNum() == 1:
                continue
            distance = float(np.linalg.norm(pos[h] - pos[other.GetIdx()]))
            if distance <= _BRIDGE_H_FACTOR * (r_h + _PT.GetRcovalent(other.GetAtomicNum())):
                near.append((other.GetIdx(), distance))
        if len(near) >= 2:  # noqa: PLR2004  two or more heavy neighbours makes this a bridging H
            bridges.append((h, near))
    if bridges:
        details = ", ".join(
            f"H{h} to " + " and ".join(f"{i} ({distance:.3f} A)" for i, distance in near) for h, near in bridges
        )
        raise ValueError(
            f"3-centre-2-electron bridge(s) detected: {details}; multi-centre X-H-X bonding is outside "
            "rxembed's two-centre donor model; supply an explicit donor graph or use a cage-capable backend"
        )


def _metal_free_rings(rw, metals):
    """Return each ring of `rw` with every metal atom removed, as a list of atom-index frozensets.

    Built once per graph for the ring tests of `_bond_flat_face_atoms` and `_drop_bridgehead_bonds`.
    `RingInfo.AtomRings()` is a view into its owning Mol's C++ memory, so the ring atoms are read out here,
    before the stripped copy is dropped, rather than handing back the live RingInfo (that use-after-free
    crashed the measurement with a MemoryError).
    A fresh `Chem.Atom` is used per kept atom rather than a copy of the original: copying carries over
    cached valence/implicit-H state from the metal-bonded graph, which corrupted ring perception the
    same way once that atom sat in a differently-bonded graph.
    """
    em = Chem.RWMol()
    kept = {}
    for atom in rw.GetAtoms():
        if atom.GetIdx() in metals:
            continue
        fresh = Chem.Atom(atom.GetAtomicNum())
        fresh.SetFormalCharge(atom.GetFormalCharge())
        kept[atom.GetIdx()] = em.AddAtom(fresh)
    for bond in rw.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in metals or b in metals:
            continue
        em.AddBond(kept[a], kept[b], Chem.BondType.SINGLE)  # bond order is irrelevant to ring membership
    stripped = em.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(stripped)
    new_to_old = {new: old for old, new in kept.items()}
    return [frozenset(new_to_old[i] for i in ring) for ring in stripped.GetRingInfo().AtomRings()]


def _bond_flat_face_atoms(mol):
    """Return `mol` with M-Y bonded where a ring binds M through every atom but Y and Y lies in their plane.

    xyzgraph 1.6.14's ring checks cap an atom's non-metal bond orders at the largest entry of its valence
    list, which for a metal is its oxidation states (Mn up to 7): once a metal's count is full, the reader
    drops the weakest face atom it reaches, MACPUC's azaindolyl N1 on Mn(CO)3 or an azaborine N on
    Cr(CO)3. A genuinely unbound ring atom folds out of the face (`_FACE_FLAP_MAX`); one in its bound
    ring-mates' plane belongs to the face. Three bound atoms define that plane, so the ring needs four. This
    runs before bond orders, so xyz2mol ranks the completed face. A flat saturated Y (BATCUV's eta4
    cyclopentadiene CH2) is bonded here and dropped again by `_drop_bridgehead_bonds`. Logs one warning
    naming every added bond. Drop it once the installed xyzgraph exempts a metal from that valence cap
    (xyzgraph-2-valence-ringgate.patch).
    """
    metals = set(metal_indices(mol))
    if not metals or mol.GetNumConformers() == 0:
        return mol
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    rings = _metal_free_rings(rw, metals)
    pos = mol.GetConformer().GetPositions()
    added = []
    for metal in sorted(metals):
        bound = {n.GetIdx() for n in rw.GetAtomWithIdx(metal).GetNeighbors()}
        for ring in rings:
            free = [a for a in ring if a not in bound]
            if len(ring) < 4 or len(free) != 1:  # noqa: PLR2004  three bound atoms define the face plane
                continue
            y = free[0]
            face = pos[[a for a in ring if a != y]]
            normal = np.linalg.svd(face - face.mean(axis=0))[2][2]
            hinge = [n.GetIdx() for n in rw.GetAtomWithIdx(y).GetNeighbors() if n.GetIdx() in ring]
            flap = np.cross(pos[hinge[0]] - pos[y], pos[hinge[1]] - pos[y])
            if np.degrees(np.arccos(min(1.0, abs(normal @ flap) / np.linalg.norm(flap)))) <= _FACE_FLAP_MAX:
                added.append((y, rw.GetAtomWithIdx(y).GetSymbol(), metal))
    if not added:
        return mol
    for y, _sym, metal in added:
        rw.AddBond(y, metal, Chem.BondType.DATIVE)
    logger.warning(
        "read_xyz: bonded face atom(s) %s: in the plane of a ring bound through the rest",
        ", ".join(f"{sym}{y}-{metal}" for y, sym, metal in added),
    )
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def _drop_bridgehead_bonds(mol):
    """Return `mol` without the M-X bonds a coordinate reader draws across a chelate ring's diagonal.

    X is a candidate when two or more of its non-metal neighbours are bound to the same metal and not to each
    other: it is then the far corner of a ring M-D1-X-D2 that alone brings it near M (a kappa2 carboxylate or
    amidinate C, the P of S2PR2, the B of kappa2-BH4), and every reader bonds it. Without a lone pair, X's
    valence left after its sigma bonds is pi: none when saturated (four sigma bonds), so it always drops; two
    when linear, one of them in the chelate plane, so it stays. Any other X drops when its donors bind through
    lone pairs of their own, no ring holds X and all of them (a ring face binds through X's p orbital however
    slipped), no open path puts a carbon beside it, and X sits no closer to M than the longest M-D bond.

    That distance is the ring's own limit: folding the ring about D1...D2 brings X toward M, and once X is as
    close as the atoms bound through it, M-X has the evidence M-D has; a regular face sits exactly there.
    tmQMg census (5,079 read graphs, 2026-09-27): kappa2 rings, flat or folded, put X at 1.04-1.35x its longest
    leg (QAHFOV, SORGAK, AREPUK's re-read outputs); eta3-S,C,S carbons sit at 0.82-0.96x (XENLEI, TILDOH).
    Each graph test is forced by what geometry alone gets wrong, measured over that census and at five seeds:
    without them it keeps MACQUD's agostic C beside a short M-H leg, drops 26 ring-face atoms (MACPUC's ring
    fusion C, the cyclo-As6 crown at 1.00x) and seven fused-ring pi carbons whose flanks hold no lone pair, drops
    POBSUU's linear P, leaving a ring the embed cannot close (5/5 to 0/5), and drops LIBFOR's allyl-type C: a
    carbon donor's lone pair on an open path is one Lewis form, which the re-read of an output redraws without
    it (5/5 to 0/5). That carbon test also keeps LAPQIC's kappa2-C,C W...P diagonal (1.35x), a known miss.
    Geometry is read last, so only a candidate the graph cannot settle needs a conformer.

    `read_xyz` applies this to every read; call it directly only to re-apply the guard to a Mol built outside
    `read_xyz`. Logs one warning naming every removed bond. Drop it once the installed xyzgraph's
    `_prune_bridgeheads` (xyzgraph-1-bridgehead-ts.patch) prunes at this cut, M-X at or past the longest M-D
    leg, in place of its fixed -0.10 A planar-closure residual, which keeps QAHFOV (-0.30) and SORGAK (-0.27).
    """
    metals = set(metal_indices(mol))
    if not metals:
        return mol
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    rings = _metal_free_rings(rw, metals)

    def far(metal, x, donors):
        at = rw.GetConformer().GetAtomPosition
        return (at(metal) - at(x)).Length() >= max((at(metal) - at(d)).Length() for d in donors)

    bad = []
    for metal in metals:
        for x in rw.GetAtomWithIdx(metal).GetNeighbors():
            xi = x.GetIdx()
            if xi in metals:
                continue
            donors = [
                n.GetIdx()
                for n in x.GetNeighbors()
                if n.GetIdx() not in metals and rw.GetBondBetweenAtoms(n.GetIdx(), metal) is not None
            ]
            if len(donors) < _BRIDGEHEAD_DONORS_MIN or any(
                rw.GetBondBetweenAtoms(a, b) is not None for a, b in itertools.combinations(donors, 2)
            ):
                continue
            sigma_to_nonmetal = x.GetTotalDegree() - sum(1 for n in x.GetNeighbors() if n.GetIdx() in metals)
            lone_pair = lone_pair_electrons(x, metals) > 0
            saturated = not lone_pair and sigma_to_nonmetal >= _BRIDGEHEAD_SIGMA_MIN
            linear = not lone_pair and sigma_to_nonmetal < _TRIGONAL_SIGMA
            open_carbon = any(rw.GetAtomWithIdx(d).GetAtomicNum() == _OPEN_FACE_Z for d in donors) and not all(
                any(xi in ring and d in ring for ring in rings) for d in donors
            )
            if saturated or (
                not linear
                and all(lone_pair_electrons(rw.GetAtomWithIdx(d), metals) > 0 for d in donors)
                and not any(xi in ring and ring.issuperset(donors) for ring in rings)
                and not open_carbon
                and far(metal, xi, donors)
            ):
                bad.append((xi, x.GetSymbol(), metal))
    if not bad:
        return mol
    for xi, _sym, metal in bad:
        remove_bond(rw, xi, metal)
    logger.warning(
        "read_xyz: dropped bridgehead bond(s) %s: chelate-ring diagonals, not donors",
        ", ".join(f"{sym}{xi}-{metal}" for xi, sym, metal in bad),
    )
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def _apply_metal_charges(mol, charges):
    """Validate and apply an explicit oxidation-state allocation."""
    if charges is None:
        return None
    supplied = {int(index): int(value) for index, value in charges.items()}
    metals = set(metal_indices(mol))
    if set(supplied) != metals:
        raise ValueError(f"metal_charges must name every metal atom index {sorted(metals)}, got {sorted(supplied)}")
    for index, value in supplied.items():
        mol.GetAtomWithIdx(index).SetFormalCharge(value)
    return supplied


def _assign_bond_orders(mol, path, charge, connectivity, requested, allow_fallback):
    """Apply the requested bond-order backend without silently changing a strict choice.

    xyzgraph's charges on a metal complex that miss the stated total are replaced, with a warning, by xyz2mol's
    orders on the same graph, which conserve it. Ranking runs on the graph without its hydrogen bonds
    (`_hydrogen_bond_legs`), which xyz2mol cannot hold and which are not constitution. When xyz2mol finds no
    closed-shell reading on the selected graph or on its own connectivity, the last resort keeps the selected
    graph and takes the reading with fewer radical electrons: xyz2mol's, with a metal's charge past its valence
    electrons moved onto anionic donors as radicals, or xyzgraph's own charges when they reach the total and keep
    every metal within its valence electrons. On a tie xyz2mol's is taken.
    """
    total = Chem.GetFormalCharge(mol)
    if requested == "xyzgraph" and allow_fallback and total != charge and _has_xyz2mol_metal(mol):
        # ponytail: delete when xyzgraph conserves the stated total charge (upstream Q1).
        logger.warning(
            "read_xyz: xyzgraph's charges total %d, not charge=%d; using xyz2mol bond orders on its graph",
            total,
            charge,
        )
        requested = "xyz2mol"
    if requested != "xyz2mol" or connectivity == "xyz2mol":
        return mol, connectivity, connectivity, False
    if not _has_xyz2mol_metal(mol):
        if not allow_fallback:
            raise ValueError(
                "bond_orders='xyz2mol' does not support this selected graph; use fallback=True or "
                "connectivity='xyzgraph', bond_orders='xyzgraph'"
            )
        if connectivity == "xyzgraph":
            unmodelled = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS})
            reason = f"has no charge model for {', '.join(unmodelled)}" if unmodelled else "is metal-only"
            logger.warning("read_xyz: xyz2mol bond-order assignment %s; keeping xyzgraph bond orders", reason)
            return mol, connectivity, connectivity, True
    backend = mol
    if legs := _hydrogen_bond_legs(mol):
        # ponytail: the default xyzgraph read keeps a hydrogen bond's leg when xyzgraph's charges reach the stated
        # total; dropping it there as well removes a TS's proton-in-flight bond (mn-h2.xyz N5-H65).
        logger.warning(
            "read_xyz: dropped hydrogen bond(s) %s before ranking bond orders; a hydrogen bond is not a bond",
            ", ".join(f"H{h}...{mol.GetAtomWithIdx(x).GetSymbol()}{x}" for h, x in legs),
        )
        mol = _without(mol, legs)
    try:
        ranked = _rank_orders(mol, charge)
    except ValueError as first:
        if not allow_fallback:
            raise
        try:
            joint = _from_xyz2mol(path, charge)
        except (ImportError, OSError, RuntimeError, ValueError) as second:
            readings = []
            with contextlib.suppress(ValueError):
                readings.append((_rank_orders(mol, charge, radicals=True), "xyz2mol"))
            metals = _metal_charges(backend)
            if connectivity == "xyzgraph" and total == charge and all(q <= n for _s, q, n in metals):
                own = Chem.Mol(backend)
                kept = ", ".join(f"{symbol}{q:+d}" for symbol, q, _limit in metals)
                own.SetProp(
                    "_rxembedChargeRescue",
                    f"xyz2mol found no closed-shell reading at charge={charge}; kept xyzgraph's {kept}",
                )
                readings.append((own, "xyzgraph"))
            if not readings:
                raise ValueError(
                    f"xyz2mol found no bond orders on the {connectivity} graph ({first}); "
                    f"xyz2mol connectivity also failed ({second})"
                ) from second
            reading, order_by = min(
                readings, key=lambda item: sum(a.GetNumRadicalElectrons() for a in item[0].GetAtoms())
            )
            return reading, connectivity, order_by, order_by != requested
        logger.warning(
            "read_xyz: xyz2mol found no bond orders on the %s graph (%s); using xyz2mol connectivity",
            connectivity,
            first,
        )
        return joint, "xyz2mol", "xyz2mol", True
    actual = ranked.GetProp("_rxembedBondOrders") if ranked.HasProp("_rxembedBondOrders") else "xyz2mol"
    return ranked, connectivity, actual, actual != requested


def _warn_if_charge_looks_missing(mol, charge):
    """Warn when a metal read at charge=0 is over its valence electrons, negative, or odd-electron.

    An ion read without its charge gets the default charge=0, and the metal absorbs the missing total; these
    three readings are what that usually looks like. The warning never changes the reading or infers a total.
    """
    if charge != 0:
        return
    suspect = []
    for symbol, q, limit in _metal_charges(mol):
        if q > limit:
            suspect.append(f"{symbol}{q:+d} is over its {limit} valence electrons")
        elif q < 0:
            suspect.append(f"{symbol}{q:+d} is negative")
        elif (limit - q) % 2:
            suspect.append(f"{symbol}{q:+d} has an odd electron count")
    if suspect:
        logger.warning("read_xyz: at charge=0, %s; pass charge= for a charged complex", ", ".join(suspect))


def _metal_charges(mol):
    """Return ``(symbol, formal charge, valence electrons)`` for each metal atom."""
    return [
        (atom.GetSymbol(), atom.GetFormalCharge(), _PT.GetNOuterElecs(atom.GetAtomicNum()))
        for atom in (mol.GetAtomWithIdx(index) for index in metal_indices(mol))
    ]


def _has_xyz2mol_metal(mol):
    """Return whether xyz2mol_tmc supports a metal present in this graph."""
    return any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in mol.GetAtoms())


def _from_xyz2mol(path, charge):
    """Read connectivity and bond orders from the vendored perceiver."""
    before = _coordinates(path)
    with rdBase.BlockLogs():
        out = get_tmc_mol(path, charge)[0]
    _assert_same_atoms(before, out)
    return out


def _right_angle_ring_chords(graph):
    """Return xyzgraph's nonmetal, non-H bonds that sit opposite an angle of 90 degrees or more in a three-ring.

    Every side of a real three-ring is a bond, so each angle is acute: an angle at k of 90 degrees or more
    makes the i-j side at least the hypotenuse of the two bonds meeting at k, so i-j is a 1,3 contact across a
    larger ring, not a bond. xyzgraph 1.6.14's strict three-ring check misses this (its limit is 110 degrees
    plus 2 degrees per unit of mean Z above carbon). This is geometric, not chemical, so a genuine partial bond
    in a transition state (for example a cyclopropyl-cation ring-opening TS) could be dropped the same way;
    metal and H rings are excluded from this guard and stay untouched.
    """
    nodes = {
        n: np.asarray(data["position"], float)
        for n, data in graph.nodes(data=True)
        if data["atomic_number"] != 1 and data["atomic_number"] not in COORDINATION_METALS
    }
    chords = set()
    for i, j in graph.edges:
        if i not in nodes or j not in nodes:
            continue
        for k in set(graph[i]) & set(graph[j]) & nodes.keys():
            u, v = nodes[i] - nodes[k], nodes[j] - nodes[k]
            if u @ v <= 0.0:  # cos(angle at k) <= 0
                chords.add((min(i, j), max(i, j)))
    return sorted(chords)


def _from_xyzgraph(path, charge, bond=()):
    """Read connectivity and bond orders from xyzgraph, as a Mol with a conformer.

    ``bond`` names atom pairs xyzgraph adds before it assigns orders and charges, so its Lewis form is computed on
    the corrected graph; the three-ring guard below removes pairs the same way.
    """
    import xyzgraph

    # Never quick=True: it skips bond-order and charge perception and returns all-single bonds.
    graph = xyzgraph.build_graph(path, charge=charge, kekule=True, bond=[*bond] or None)
    if phantom := _right_angle_ring_chords(graph):
        # ponytail: drop this guard once the installed xyzgraph caps the strict three-ring angle at 90 deg.
        logger.warning(
            "read_xyz: dropped cross-ring contact(s) %s (a three-ring angle >= 90 deg)",
            ", ".join(f"{i}-{j}" for i, j in phantom),
        )
        graph = xyzgraph.build_graph(path, charge=charge, kekule=True, bond=[*bond] or None, unbond=phantom)
    orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    rw, idx = Chem.RWMol(), {}
    for node, data in sorted(graph.nodes(data=True)):
        atom = Chem.Atom(int(data["atomic_number"]))
        atom.SetFormalCharge(round(data.get("formal_charge", 0) or 0))
        atom.SetNoImplicit(True)  # an xyz is fully explicit, hydrogens included
        idx[node] = rw.AddAtom(atom)
    for u, v, data in graph.edges(data=True):
        u_metal = rw.GetAtomWithIdx(idx[u]).GetAtomicNum() in COORDINATION_METALS
        v_metal = rw.GetAtomWithIdx(idx[v]).GetAtomicNum() in COORDINATION_METALS
        if u_metal != v_metal:
            donor, metal = (v, u) if u_metal else (u, v)
            rw.AddBond(idx[donor], idx[metal], Chem.BondType.DATIVE)
            continue
        order = data.get("bond_order", 1.0)
        if abs(order - 1.5) < _AROMATIC_BO_TOL:  # aromatic, only if kekulization fell through
            bond = rw.GetBondWithIdx(rw.AddBond(idx[u], idx[v], Chem.BondType.AROMATIC) - 1)
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
        else:
            rw.AddBond(idx[u], idx[v], orders.get(round(order), Chem.BondType.SINGLE))

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for node, data in graph.nodes(data=True):
        x, y, z = data["position"]
        conf.SetAtomPosition(idx[node], Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    # A tolerant intermediate graph: bond-order ranking owns valence, so only this probe is silenced.
    with rdBase.BlockLogs():
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )
    return mol


def _from_rdkit_connectivity(path, charge):
    """Read connectivity with RDKit's native connect-the-dots algorithm."""
    mol = _coordinates(path)
    rdDetermineBonds.DetermineConnectivity(mol, charge=charge, useVdw=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _restore_consensus_ligand_bonds(mol, native, path, charge):
    """Re-read xyzgraph with the internal ligand bonds it missed but RDKit and xyz2mol both find.

    xyzgraph stays authoritative for metal contacts; a nonmetal edge is restored only when RDKit and
    xyz2mol independently agree it exists, so a stretched coordination edge cannot enter this way.
    """
    # ponytail: drop this guard once the installed xyzgraph keeps a ligand bond that closes a three-ring through a
    # metal (WICHIZ S-S) and its nonmetal cutoff reaches the covalent sum of heavy p-block pairs (ZUDWUQ As-As).
    metals = set(metal_indices(mol))
    selected_edges = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in mol.GetBonds()}
    native_edges = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in native.GetBonds()}
    candidates = sorted(pair for pair in native_edges - selected_edges if not pair.intersection(metals))
    if not candidates:
        return mol, ()
    try:
        joint = _from_xyz2mol(path, charge)
        _assert_same_atoms(mol, joint)
    except (ImportError, OSError, RuntimeError, ValueError):
        return mol, ()
    joint_edges = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in joint.GetBonds()}
    confirmed = tuple((min(pair), max(pair)) for pair in candidates if pair in joint_edges)
    if not confirmed:
        return mol, ()
    return _from_xyzgraph(path, charge, bond=confirmed), confirmed


def _coordinates(path):
    """Read the atoms and coordinates without inferring bonds."""
    mol = Chem.MolFromXYZFile(path)
    if mol is None:
        raise ValueError(f"could not read {path} as an .xyz")
    return mol


def _from_rdkit(path, charge):
    """Read an organic molecule with RDKit's bond perceiver."""
    mol = _coordinates(path)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except ValueError as exc:
        raise ValueError(
            f"could not perceive bonds in {path}: RDKit's perceiver is organic-only, and this is "
            f"neither a molecule it accepts nor a metal complex. For a stretched TS core, "
            f"pip install 'rxembed[workflow]' for xyzgraph ({exc})"
        ) from exc
    return mol


def _with_fallback(path, charge, backend, allow_fallback=True):
    """Try a perceiver, warning before the graph-changing fallback."""
    primary = {"rdkit": _from_rdkit_connectivity, "xyzgraph": _from_xyzgraph, "xyz2mol": _from_xyz2mol}[backend]
    if not allow_fallback:
        return primary(path, charge), backend
    try:
        return primary(path, charge), backend
    except (ImportError, OSError, RuntimeError, ValueError) as first:
        if backend == "xyzgraph":
            fallback, name = (
                (_from_xyz2mol, "xyz2mol") if _has_xyz2mol_metal(_coordinates(path)) else (_from_rdkit, "rdkit")
            )
        else:  # xyz2mol or RDKit connectivity falls back to the metal-aware xyzgraph path
            fallback, name = _from_xyzgraph, "xyzgraph"

        state = "unavailable" if isinstance(first, ImportError) else f"failed ({first})"
        display = "RDKit" if name == "rdkit" else name
        logger.warning("read_xyz: %s %s; using %s", backend, state, display)
        try:
            return fallback(path, charge), name
        except (ImportError, OSError, RuntimeError, ValueError) as second:
            # Name the remedy whenever xyzgraph itself, not just this input, is the reason nothing worked.
            hint = (
                "; pip install 'rxembed[workflow]' for the xyzgraph reader"
                if state == "unavailable" or isinstance(second, ImportError)
                else ""
            )
            raise ValueError(f"{backend} {state}; {display} fallback also failed ({second}){hint}") from second


def _assert_same_atoms(before, after):
    """Reject xyz2mol reassembly that drops or reorders the input atoms."""
    want = [a.GetAtomicNum() for a in before.GetAtoms()]
    got = [a.GetAtomicNum() for a in after.GetAtoms()]
    same_coordinates = (
        want == got
        and before.GetNumConformers()
        and after.GetNumConformers()
        and np.allclose(before.GetConformer().GetPositions(), after.GetConformer().GetPositions(), rtol=0.0, atol=1e-6)
    )
    if want != got or not same_coordinates:
        raise ValueError(f"xyz2mol changed the {len(want)} input atoms, their order, or their coordinates")


def _rdkit_bond_orders(mol, charge):
    """Assign bond orders with RDKit on a graph that has no metal xyz2mol's search supports."""
    out = Chem.Mol(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(out, charge=charge)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"RDKit could not assign bond orders on the selected organic connectivity: {exc}") from exc
    logger.warning("read_xyz: xyz2mol bond-order assignment is metal-only; using RDKit for this graph")
    out.SetProp("_rxembedBondOrders", "rdkit")
    return out


def _hydrogen_bond_legs(mol):
    """Return ``(hydrogen, acceptor)`` for each hydrogen bond `mol` holds as a bond.

    A hydrogen bonded to two or more nonmetals keeps its first leg by `hydrogen_neighbor_order`. Each other
    nonmetal leg is judged as a zero-order contact by `utils.hydrogen_bond`: a hydrogen bond when the acceptor
    keeps a lone pair, which leaves the graph; a three-centre bond (B-H-B) otherwise, which stays.
    """
    shared = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 1
        and sum(n.GetAtomicNum() not in COORDINATION_METALS for n in atom.GetNeighbors()) > 1
    ]
    if not shared:
        return []
    pos = mol.GetConformer().GetPositions()
    ranks = _flat_ranks(mol)
    metals = set(metal_indices(mol))
    contact = Chem.RWMol(mol)
    candidates = []
    for h in shared:
        ordered = hydrogen_neighbor_order(mol, h, metals=COORDINATION_METALS, positions=pos, ranks=ranks)
        for other in ordered[1:]:
            if other not in metals:  # an M-H leg is a bridge, which bond-order ranking restores
                contact.GetBondBetweenAtoms(h, other).SetBondType(Chem.BondType.ZERO)
                candidates.append((h, other))
    contact.UpdatePropertyCache(strict=False)  # lone-pair counts read valence without the contacts
    return [
        (h, other)
        for h, other in candidates
        if hydrogen_bond(contact.GetAtomWithIdx(h), contact.GetAtomWithIdx(other), metals)
    ]


def _flat_ranks(mol):
    """Return `flat_ranks` of a backend graph whose property cache may not be computed."""
    ranked = Chem.Mol(mol)
    ranked.UpdatePropertyCache(strict=False)
    return flat_ranks(ranked)


def _without(mol, pairs):
    """Return `mol` without the bonds between `pairs`."""
    rw = Chem.RWMol(mol)
    for i, j in pairs:
        remove_bond(rw, i, j)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def _bridging_hydrogen_contacts(mol):
    """Return ``(hydrogen, partner, bond type)`` for each hydrogen leg xyz2mol cannot hold.

    xyz2mol allows hydrogen exactly one valence, so a hydrogen bridging a metal, or a three-centre nonmetal bridge
    (`_hydrogen_bond_legs` has already dropped hydrogen bonds), keeps only its first leg by
    `hydrogen_neighbor_order`. Each other leg comes back after the search as a dative bond to a metal or a
    zero-order contact to a nonmetal.
    """
    pos = mol.GetConformer().GetPositions()
    ranks = _flat_ranks(mol)
    contacts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() <= 1:
            continue
        # A B-H-M or C-H-M bridge keeps its ligand-bond role even when the M-H distance is shorter; canonical rank
        # breaks a genuine X-H-X distance tie without depending on bond insertion order.
        ordered = hydrogen_neighbor_order(mol, atom.GetIdx(), metals=TRANSITION_METALS_NUM, positions=pos, ranks=ranks)
        for other in ordered[1:]:
            metal = mol.GetAtomWithIdx(other).GetAtomicNum() in TRANSITION_METALS_NUM
            contacts.append((atom.GetIdx(), other, Chem.BondType.DATIVE if metal else Chem.BondType.ZERO))
    return contacts


def _flatten_for_xyz2mol(mol, contacts):
    """Return the graph xyz2mol searches: `contacts` cut, M-L dative, other bonds single, charges cleared.

    `xyz2mol_tmc`'s three structural charges are re-applied for its calibrated search.
    """
    flat = Chem.RWMol(mol)
    for h, other, _contact in contacts:
        flat.RemoveBond(h, other)
    for bond in list(flat.GetBonds()):
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        begin_metal = flat.GetAtomWithIdx(begin).GetAtomicNum() in TRANSITION_METALS_NUM
        end_metal = flat.GetAtomWithIdx(end).GetAtomicNum() in TRANSITION_METALS_NUM
        if begin_metal == end_metal:
            continue
        donor, metal = (end, begin) if begin_metal else (begin, end)
        if bond.GetBondType() == Chem.BondType.DATIVE and begin == donor:
            continue
        flat.RemoveBond(begin, end)
        flat.AddBond(donor, metal, Chem.BondType.DATIVE)
    for bond in flat.GetBonds():
        if bond.GetBondType() != Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
    for atom in flat.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
            continue  # preserve a multi-metal backend's oxidation-state allocation
        heavy = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in TRANSITION_METALS_NUM)
        atom.SetFormalCharge(SEEDED_STRUCTURAL_CHARGES.get((atom.GetAtomicNum(), heavy), 0))
    graph = flat.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    return graph


def _rank_orders(mol, charge, radicals=False):
    """Re-assign bond orders and charges without changing connectivity.

    A structure with no `TRANSITION_METALS_NUM` metal delegates to RDKit; that set is narrower than
    `metal_core.COORDINATION_METALS` (no Ce to Yb, no actinides), which xyz2mol's search has no charge model
    for. ``radicals`` is passed to `get_tmc_mol`.
    """
    if not _has_xyz2mol_metal(mol):
        return _rdkit_bond_orders(mol, charge)
    contacts = _bridging_hydrogen_contacts(mol)
    graph = _flatten_for_xyz2mol(mol, contacts)
    coords = graph.GetConformer().GetPositions().tolist()
    try:
        with rdBase.BlockLogs():
            out = get_tmc_mol(None, charge, graph=(graph, coords), radicals=radicals)[0]
        # Atom-for-atom, in order: the contacts below are restored by input atom index.
        _assert_same_atoms(mol, out)
        rw = Chem.RWMol(out)
        for h, other, bond_type in contacts:
            if rw.GetBondBetweenAtoms(h, other) is None:
                rw.AddBond(h, other, bond_type)
        rw.UpdatePropertyCache(strict=False)
        result = rw.GetMol()
        before = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        after = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in result.GetBonds()}
        if before != after:
            raise ValueError(f"xyz2mol changed connectivity: lost {before - after}, added {after - before}")
        result.SetProp("_rxembedBondOrders", "xyz2mol")
        return result
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"xyz2mol could not assign bond orders on the selected connectivity: {exc}") from exc
