"""Read an ``.xyz`` into an RDKit Mol: the one input adaptation that needs perception.

Input adaptation only, ahead of the core `embed()` engine; a SMILES needs none of this (`rxembed.metal_smiles`
instead). Connectivity and bond order are separate arguments: RDKit, xyzgraph, and xyz2mol can supply
connectivity, but only the latter two rank bond orders.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, rdBase

from rxembed.metal_core import (
    COORDINATION_METALS,
    _reject_boron_cages,
    metal_indices,
)
from rxembed.stereo import stereo_from_3d
from rxembed.utils import _rcov, flat_ranks, hydrogen_neighbor_order

_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic
_SHARED_H_MAX_RATIO = 1.15  # comparable X-H legs denote a shared proton; the benchmark owns the calibration
_SHARED_H_NEIGHBOURS = 2
_BRIDGE_H_FACTOR = 1.2  # x (r_cov(H) + r_cov(X)); same margin as a new-bond distance elsewhere in the pipeline
_CONNECTIVITY = {"rdkit", "xyzgraph", "xyz2mol"}
_BOND_ORDERS = {"xyzgraph", "xyz2mol"}
logger = logging.getLogger("rxembed")


def read_xyz(path, charge=0, connectivity="xyzgraph", bond_orders="xyzgraph", metal_charges=None, fallback=True):
    """Read an ``.xyz`` into a Mol with perceived bonds and a conformer; robust for metals and TSs.

    `connectivity` picks RDKit's connect-the-dots, ``"xyzgraph"``, or ``"xyz2mol"``. `bond_orders` is xyzgraph's
    optimiser (needs ``connectivity="xyzgraph"``) or xyz2mol's ranked charge search (metal complexes only; a
    metal-free TS keeps xyzgraph's orders instead of failing); RDKit connectivity never adds or removes a metal-donor
    contact. ``metal_charges`` names a formal charge per metal index for a multi-metal XYZ whose total does not
    determine the split; every metal must be named. Sanitisation is lenient (rings yes, valence checks no); perceiver
    choices land on the returned Mol's ``_rxembed*`` properties.
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
    _reject_boron_cages(mol)  # before xyz2mol forces multi-centre B-H/B-B into valence-two orders
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
    mol.SetProp("_rxembedConnectivity", perceived_by)
    mol.SetProp("_rxembedBondOrders", order_by)
    mol.SetProp("_rxembedConnectivityAdded", ",".join(f"{i}-{j}" for i, j in added))
    mol.SetBoolProp("_rxembedPerceptionFallback", used_fallback or order_by != bond_orders)
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
    bridge `_reject_boron_cages`'s boron count cannot: a single B-H-B pair, or a non-boron bridge. A
    dropped terminal hydrogen sits near one heavy atom, not two; a free hydride counterion sits outside
    the covalent-radius reach of anything.
    """
    if mol.GetNumConformers() == 0:
        return
    pos = mol.GetConformer().GetPositions()
    r_h = _rcov(1)
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
            if distance <= _BRIDGE_H_FACTOR * (r_h + _rcov(other.GetAtomicNum())):
                near.append((other.GetIdx(), distance))
        if len(near) >= _SHARED_H_NEIGHBOURS:
            bridges.append((h, near))
    if bridges:
        details = ", ".join(
            f"H{h} to " + " and ".join(f"{i} ({distance:.3f} A)" for i, distance in near) for h, near in bridges
        )
        raise ValueError(
            f"3-centre-2-electron bridge(s) detected: {details}; multi-centre X-H-X bonding is outside "
            "rxembed's two-centre donor model; supply an explicit donor graph or use a cage-capable backend"
        )


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
    """Apply the requested bond-order backend without silently changing a strict choice."""
    if requested != "xyz2mol" or connectivity == "xyz2mol":
        return mol, connectivity, connectivity, False
    if not _has_xyz2mol_metal(mol):
        if not allow_fallback:
            raise ValueError(
                "bond_orders='xyz2mol' does not support this selected graph; use fallback=True or "
                "connectivity='xyzgraph', bond_orders='xyzgraph'"
            )
        if connectivity == "xyzgraph":
            logger.warning("read_xyz: xyz2mol bond-order assignment is metal-only; keeping xyzgraph bond orders")
            return mol, connectivity, connectivity, True
    try:
        ranked = _rank_orders(mol, charge)
    except ValueError as first:
        if not allow_fallback:
            raise
        try:
            joint = _from_xyz2mol(path, charge)
        except (ImportError, OSError, RuntimeError, ValueError) as second:
            raise ValueError(
                f"{connectivity} connectivity has no valid bond-order assignment ({first}); "
                f"xyz2mol connectivity also failed ({second})"
            ) from second
        logger.warning(
            "read_xyz: %s connectivity has no valid bond-order assignment (%s); using xyz2mol connectivity",
            connectivity,
            first,
        )
        return joint, "xyz2mol", "xyz2mol", True
    actual = ranked.GetProp("_rxembedBondOrders") if ranked.HasProp("_rxembedBondOrders") else "xyz2mol"
    return ranked, connectivity, actual, actual != requested


def _has_xyz2mol_metal(mol):
    """Return whether xyz2mol_tmc supports a metal present in this graph."""
    from .xyz2mol_tmc import TRANSITION_METALS_NUM

    return any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in mol.GetAtoms())


def _from_xyz2mol(path, charge):
    """Read connectivity and bond orders from the vendored perceiver."""
    from .xyz2mol_tmc import get_tmc_mol

    before = _coordinates(path)
    with rdBase.BlockLogs():
        out = get_tmc_mol(path, charge)[0]
    _assert_same_atoms(before, out)
    return out


def _from_xyzgraph(path, charge):
    """Read connectivity and bond orders from xyzgraph, as a Mol with a conformer."""
    import xyzgraph
    from rdkit.Chem import BondType, Conformer
    from rdkit.Geometry import Point3D

    # Never quick=True: it skips bond-order and charge perception and returns all-single bonds.
    graph = xyzgraph.build_graph(path, charge=charge, kekule=True)
    orders = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE}
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
            rw.AddBond(idx[donor], idx[metal], BondType.DATIVE)
            continue
        order = data.get("bond_order", 1.0)
        if abs(order - 1.5) < _AROMATIC_BO_TOL:  # aromatic, only if kekulization fell through
            bond = rw.GetBondWithIdx(rw.AddBond(idx[u], idx[v], BondType.AROMATIC) - 1)
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
        else:
            rw.AddBond(idx[u], idx[v], orders.get(round(order), BondType.SINGLE))

    mol = rw.GetMol()
    conf = Conformer(mol.GetNumAtoms())
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
    from rdkit.Chem import rdDetermineBonds

    rdDetermineBonds.DetermineConnectivity(mol, charge=charge, useVdw=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _restore_consensus_ligand_bonds(mol, native, path, charge):
    """Restore missing internal bonds present in both independent ligand perceivers.

    xyzgraph stays authoritative for metal contacts; a nonmetal edge is restored only when RDKit and
    xyz2mol independently agree it exists, so a stretched coordination edge cannot enter this way.
    """
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
    confirmed = [pair for pair in candidates if pair in joint_edges]
    if not confirmed:
        return mol, ()
    rw = Chem.RWMol(mol)
    for pair in confirmed:
        begin, end = sorted(pair)
        bond = joint.GetBondBetweenAtoms(begin, end)
        rw.AddBond(begin, end, bond.GetBondType())
        added = rw.GetBondBetweenAtoms(begin, end)
        added.SetIsAromatic(bond.GetIsAromatic())
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out, tuple((min(pair), max(pair)) for pair in confirmed)


def _coordinates(path):
    """Read the atoms and coordinates without inferring bonds."""
    mol = Chem.MolFromXYZFile(path)
    if mol is None:
        raise ValueError(f"could not read {path} as an .xyz")
    return mol


def _from_rdkit(path, charge):
    """Read an organic molecule with RDKit's bond perceiver."""
    mol = _coordinates(path)

    from rdkit.Chem import rdDetermineBonds

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
            from .xyz2mol_tmc import TRANSITION_METALS_NUM

            raw = _coordinates(path)
            has_metal = any(a.GetAtomicNum() in TRANSITION_METALS_NUM for a in raw.GetAtoms())
            fallback, name = (_from_xyz2mol, "xyz2mol") if has_metal else (_from_rdkit, "rdkit")
        else:  # xyz2mol or RDKit connectivity falls back to the metal-aware xyzgraph path
            fallback, name = _from_xyzgraph, "xyzgraph"

        state = "unavailable" if isinstance(first, ImportError) else f"failed ({first})"
        display = "RDKit" if name == "rdkit" else name
        logger.warning("read_xyz: %s %s; using %s", backend, state, display)
        try:
            return fallback(path, charge), name
        except (ImportError, OSError, RuntimeError, ValueError) as second:
            raise ValueError(f"{backend} {state}; {display} fallback also failed ({second})") from second


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


def _rank_orders(mol, charge):  # noqa: C901 - one linear graph-normalization pass
    """Re-assign bond orders and charges without changing connectivity.

    Bonds flatten to single and charges clear (`xyz2mol_tmc`'s three structural charges are re-applied for its
    calibrated search); a metal-free structure delegates to RDKit. xyz2mol allows hydrogen exactly one valence, so a
    hydrogen bridging a metal or shared between two nonmetal legs (proton transfer) has its longer contact removed
    before the search, restored as dative after.
    """
    from .xyz2mol_tmc import TRANSITION_METALS_NUM, get_tmc_mol

    if not _has_xyz2mol_metal(mol):
        from rdkit.Chem import rdDetermineBonds

        out = Chem.Mol(mol)
        try:
            rdDetermineBonds.DetermineBondOrders(out, charge=charge)
            logger.warning("read_xyz: xyz2mol bond-order assignment is metal-only; using RDKit for this graph")
            out.SetProp("_rxembedBondOrders", "rdkit")
            return out
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"RDKit could not assign bond orders on the selected organic connectivity: {exc}") from exc

    flat = Chem.RWMol(mol)
    pos = mol.GetConformer().GetPositions()
    donated = []  # (hydrogen, acceptor, type) contacts removed before the search, re-added after
    ranked = Chem.Mol(mol)
    ranked.UpdatePropertyCache(strict=False)
    ranks = flat_ranks(ranked)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() <= 1:
            continue
        h = atom.GetIdx()
        neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
        nonmetals = [n for n in neighbours if mol.GetAtomWithIdx(n).GetAtomicNum() not in TRANSITION_METALS_NUM]
        # A B-H-M or C-H-M bridge keeps its ligand-bond role even when the M-H distance is shorter; canonical rank
        # breaks a genuine X-H-X distance tie without depending on bond insertion order.
        ordered = hydrogen_neighbor_order(
            mol,
            h,
            metals=TRANSITION_METALS_NUM,
            positions=pos,
            ranks=ranks,
        )
        distances = [float(np.linalg.norm(pos[n] - pos[h])) for n in ordered]
        metal_bound = bool(set(neighbours) - set(nonmetals))
        shared = (
            len(ordered) == _SHARED_H_NEIGHBOURS
            and distances[0] > 0
            and distances[1] / distances[0] <= _SHARED_H_MAX_RATIO
        )
        if not metal_bound and not shared:
            continue
        for other in ordered[1:]:
            flat.RemoveBond(h, other)
            contact = (
                Chem.BondType.DATIVE
                if mol.GetAtomWithIdx(other).GetAtomicNum() in TRANSITION_METALS_NUM
                else Chem.BondType.ZERO
            )
            donated.append((h, other, contact))

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
        seeded = {(7, 4): 1, (8, 3): 1, (5, 4): -1}  # get_basic_mol's, off the metal-free valence
        atom.SetFormalCharge(seeded.get((atom.GetAtomicNum(), heavy), 0))

    graph = flat.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    coords = graph.GetConformer().GetPositions().tolist()
    try:
        with rdBase.BlockLogs():
            out = get_tmc_mol(None, charge, graph=(graph, coords))[0]
        # Atom-for-atom, in order: the contacts below are restored by input atom index.
        _assert_same_atoms(mol, out)
        rw = Chem.RWMol(out)
        for h, other, bond_type in donated:
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
