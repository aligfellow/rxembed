"""Read an ``.xyz``, in-memory atoms or a bonded graph into an RDKit Mol: the inputs that need perception.

xyzgraph, RDKit and xyz2mol supply connectivity, and xyzgraph or xyz2mol rank bond orders; `read_graph` ranks a caller's
connectivity the same way. A fallback warns and records the backend used, and no reader drops or reorders atoms or
changes the requested charge.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import os
from typing import NamedTuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from rxembed.graph import (
    COORDINATION_METALS,
    hydrogen_bond,
    hydrogen_legs,
    ligand_graph,
    lone_pair_electrons,
    metal_indices,
)
from rxembed.metal.centres import reject_boron_cages
from rxembed.perception.xyz2mol_tmc import SEEDED_STRUCTURAL_CHARGES, TRANSITION_METALS_NUM, get_tmc_mol
from rxembed.stereo import stereo_from_3d
from rxembed.utils import remove_bond, require_mol, skeleton

_PT = Chem.GetPeriodicTable()
_ELEMENTS = frozenset(_PT.GetElementSymbol(z) for z in range(1, 119))

_KEKULE_ORDERS = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
_BRIDGE_H_FACTOR = 1.2  # x (r_cov(H) + r_cov(X)); same margin as a new-bond distance elsewhere in the pipeline
_CONNECTIVITY = {"rdkit", "xyzgraph", "xyz2mol"}
_BOND_ORDERS = {"xyzgraph", "xyz2mol", None}
_BRIDGEHEAD_SIGMA_MIN = 4  # a kappa2 chelate bridgehead (P, Si, B) bonds >=4 non-metal sigma neighbours
_BRIDGEHEAD_DONORS_MIN = 2  # fewer is a sigma-silane/borane bridgehead (one metal-bound neighbour), not this rule
_TRIGONAL_SIGMA = 3  # fewer, with no lone pair: an sp X keeps a pi orbital in the chelate plane, so the fold test fails
_OPEN_FACE_Z = 6  # an open D-X-C path can be an eta3 allyl-type face, so a carbon donor keeps an open X bound
logger = logging.getLogger("rxembed")


def read_xyz(source, charge=None, connectivity="xyzgraph", bond_orders="xyzgraph", metal_charges=None, fallback=True):
    """Read an ``.xyz`` into a Mol with perceived bonds and its conformer, metals included.

    `source` is an ``.xyz`` path, or the atoms in memory as ``(symbol, (x, y, z))`` pairs; from ASE that is
    ``list(zip(atoms.get_chemical_symbols(), atoms.positions))``.

    Every XYZ is read as a minimum: a Lewis graph cannot encode a TS, so a TS geometry comes in through ``fix=`` or
    ``constrain=``. An XYZ stores no charge: `charge` is the total, xyzgraph's own reading when None; give it for a
    radical, a charged complex, another connectivity or a fallback reader. `connectivity` is "xyzgraph", "rdkit"
    (connect-the-dots) or "xyz2mol". `bond_orders` is xyzgraph's optimiser (needs xyzgraph connectivity) or "xyz2mol"'s
    ranked charge search for metal complexes; a graph without a metal xyz2mol models keeps xyzgraph's orders, with a
    warning. `metal_charges` gives every metal's formal charge, for a multi-metal XYZ whose total does not fix the
    split. ``fallback=False`` raises where a backend fails instead of warning and using another. The backends used are
    recorded in the Mol's ``_rxembed*`` properties.

    ``bond_orders=None`` skips bond orders, to check which bonds a geometry has: every bond single (metal bonds dative),
    no formal charges and no multi-centre refusals.
    """
    source = _xyz_source(source)
    if connectivity not in _CONNECTIVITY:
        raise ValueError(f"connectivity must be 'rdkit', 'xyzgraph', or 'xyz2mol', got {connectivity!r}")
    if bond_orders not in _BOND_ORDERS:
        raise ValueError(f"bond_orders must be 'xyzgraph', 'xyz2mol', or None, got {bond_orders!r}")
    if charge is None and connectivity != "xyzgraph":
        raise ValueError(f"connectivity={connectivity!r} needs charge=: only xyzgraph reads a total from a geometry")
    if bond_orders != "xyz2mol" and connectivity != "xyzgraph":
        raise ValueError(
            f"bond_orders={bond_orders!r} needs connectivity='xyzgraph'; use bond_orders='xyz2mol' with another "
            "perceiver"
        )
    mol = _read(source, charge, connectivity, bond_orders, metal_charges, fallback)
    try:
        with rdBase.BlockLogs():
            stereo_from_3d(mol, metal_indices(mol), apply=True)
    except (RuntimeError, ValueError) as exc:
        logger.warning("read_xyz: could not assign input stereo (%s); keeping the perceived graph", exc)
    return mol


def read_geometry(mol, conf_id=-1):
    """Return the bonds-only Mol `read_xyz` gives with ``bond_orders=None`` for conformer `conf_id` of `mol`."""
    return _read(xyz_atoms(mol, conf_id), Chem.GetFormalCharge(mol), "xyzgraph", None, None, fallback=False)


def read_graph(mol, charge=None, metal_charges=None):
    """Return the Lewis structure of `mol`'s bonded graph at total `charge`, as `read_xyz` returns one.

    `mol`'s bonds are the connectivity, in its atom order; its bond orders, charges, stereo and conformers are ignored,
    so every hydrogen must be an atom. Bond orders and charges are ranked as ``bond_orders="xyz2mol"`` ranks a
    geometry's, a metal-free graph by RDKit. With no coordinates to break ties, the reading has the fewest charges,
    which can give a metal a lower oxidation state than its geometry would. Raises without `charge`, or when no Lewis
    structure fits.
    """
    counted = Chem.Mol(require_mol(mol, "rx.read_graph"))
    counted.UpdatePropertyCache(strict=False)
    if held := [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in counted.GetAtoms() if atom.GetTotalNumHs()]:
        raise ValueError(
            f"rx.read_graph needs every hydrogen as an atom; add them with Chem.AddHs (counts on {', '.join(held)})"
        )
    if charge is None:
        raise ValueError("rx.read_graph needs charge=: a bonded graph has no geometry to read a total from")
    graph = skeleton(mol).GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    reject_boron_cages(graph)
    _apply_metal_charges(graph, metal_charges)
    ranked, ranker = _rank_orders(_without_hydrogen_bonds(graph), charge)
    return _settled(_BondOrders(ranked, "graph", ranker), charge, metal_charges, fallback=False)


def _read(source, charge, connectivity, bond_orders, metal_charges, fallback):
    """Return the checked graph `read_xyz` reads from `source`, a path or in-memory atoms."""
    if bond_orders is None:
        mol, perceived_by = _from_xyzgraph(source, charge, quick=True), connectivity
    else:
        mol, perceived_by = _with_fallback(source, charge, connectivity, fallback)
        reject_boron_cages(mol)  # before xyz2mol forces multi-centre B-H/B-B into valence-two orders
        if charge is None:  # xyzgraph read it as its formal charges' sum, the total every later step uses
            charge = Chem.GetFormalCharge(mol)
            logger.info("read_xyz: xyzgraph read a total charge of %+d", charge)
    _apply_metal_charges(mol, metal_charges)  # before bond orders, so xyz2mol ranks on the named split
    read = _assign_bond_orders(mol, source, charge, perceived_by, bond_orders, fallback)
    changed = read.connectivity != connectivity or read.ordered_by != bond_orders
    return _settled(read, charge, metal_charges, fallback=changed)


def _settled(read, charge, metal_charges, *, fallback):
    """Return the Mol of `read`, checked and labelled as `read_xyz` and `read_graph` return it.

    Bridgehead bonds are dropped and `metal_charges` re-applied, so no backend overrides the named split. A read with
    bond orders must reach `charge` and may not leave a bridging hydrogen unbonded; at charge 0 it warns about a metal
    over its valence electrons, negative or odd-electron.
    """
    mol = _drop_bridgehead_bonds(read.mol)
    _apply_metal_charges(mol, metal_charges)
    if read.ordered_by is not None:
        # after bond orders: xyz2mol's metal-free fallback can orphan a 3c-2e bridge, as in B3H8-
        _reject_unbonded_bridging_hydrogens(mol)
        perceived_charge = Chem.GetFormalCharge(mol)
        if perceived_charge != charge:
            named = metal_charges is not None and read.ordered_by == "xyzgraph"  # xyz2mol ranks ligands on a split
            remedy = "use bond_orders='xyz2mol' for the named metal_charges=" if named else "check charge="
            raise ValueError(f"perceived total charge {perceived_charge} does not match charge={charge}; {remedy}")
    if read.ordered_by is not None and charge == 0:  # how a complex read without its charge usually looks
        suspect = []
        for metal in (mol.GetAtomWithIdx(index) for index in metal_indices(mol)):
            q, limit = metal.GetFormalCharge(), _PT.GetNOuterElecs(metal.GetAtomicNum())
            if q > limit:
                suspect.append(f"{metal.GetSymbol()}{q:+d} is over its {limit} valence electrons")
            elif q < 0:
                suspect.append(f"{metal.GetSymbol()}{q:+d} is negative")
            elif (limit - q) % 2:
                suspect.append(f"{metal.GetSymbol()}{q:+d} has an odd electron count")
        if suspect:
            logger.warning("perceive: at charge=0, %s; pass charge= for a charged complex", ", ".join(suspect))
    rescue = mol.GetProp("_rxembedChargeRescue") if mol.HasProp("_rxembedChargeRescue") else ""
    if rescue:
        logger.warning("perceive: %s", rescue)
    mol.SetProp("_rxembedConnectivity", read.connectivity)
    mol.SetProp("_rxembedBondOrders", read.ordered_by or "")
    mol.SetProp("_rxembedChargeRescue", rescue)
    mol.SetBoolProp("_rxembedPerceptionFallback", fallback)
    return mol


def _with_fallback(path, charge, backend, allow_fallback=True):
    """Return the Mol `backend` perceives and the backend's name, or with `allow_fallback` another's.

    A fallback warns; without `charge` it raises instead, since no fallback reads a total.
    """
    primary = {"rdkit": _from_rdkit_connectivity, "xyzgraph": _from_xyzgraph, "xyz2mol": _from_xyz2mol}[backend]
    try:
        return primary(path, charge), backend
    except (OSError, RuntimeError, ValueError) as first:
        if not allow_fallback:
            raise
        if backend != "xyzgraph":  # xyz2mol or RDKit connectivity falls back to the metal-aware xyzgraph path
            fallback, name = _from_xyzgraph, "xyzgraph"
        elif any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in _coordinates(path).GetAtoms()):
            fallback, name = _from_xyz2mol, "xyz2mol"
        else:
            fallback, name = _from_rdkit, "rdkit"
        state = f"failed ({first})"
        display = "RDKit" if name == "rdkit" else name
        if charge is None:  # only xyzgraph reads a total from a geometry
            raise ValueError(f"{backend} failed ({first}); pass charge= to read with {display}") from first
        logger.warning("read_xyz: %s %s; using %s", backend, state, display)
        try:
            return fallback(path, charge), name
        except (OSError, RuntimeError, ValueError) as second:
            raise ValueError(f"{backend} {state}; {display} fallback also failed ({second})") from second


def xyz_atoms(mol, conf_id=-1):
    """Return conformer `conf_id` of `mol` as the ``(symbol, (x, y, z))`` atoms xyzgraph reads in place of a file."""
    pos = mol.GetConformer(conf_id).GetPositions()
    return [(atom.GetSymbol(), tuple(float(x) for x in pos[atom.GetIdx()])) for atom in mol.GetAtoms()]


def _from_xyzgraph(path, charge, quick=False):
    """Read connectivity and bond orders from xyzgraph, as a Mol with a conformer.

    An edge xyzgraph marks NCI is a hydrogen bond, not constitution, and is left out. ``quick`` reads connectivity
    alone, all bonds single, and finds hydrogen bonds itself.
    """
    import xyzgraph

    graph = xyzgraph.build_graph(path, charge=charge, kekule=True, quick=quick)
    rw, idx = Chem.RWMol(), {}
    for node, data in sorted(graph.nodes(data=True)):
        atom = Chem.Atom(int(data["atomic_number"]))
        atom.SetFormalCharge(0 if quick else round(data["formal_charge"]))
        atom.SetNoImplicit(True)  # an xyz is fully explicit, hydrogens included
        idx[node] = rw.AddAtom(atom)
    for u, v, data in graph.edges(data=True):
        if data.get("NCI"):
            continue
        # with kekule=True every other edge is order 1, 2 or 3; anything else is a changed xyzgraph
        if (order := data.get("bond_order")) not in _KEKULE_ORDERS:
            raise ValueError(
                f"xyzgraph {xyzgraph.__version__} gave edge {u}-{v} bond_order={order!r}; install the version "
                "rxembed requires"
            )
        u_metal = rw.GetAtomWithIdx(idx[u]).GetAtomicNum() in COORDINATION_METALS
        v_metal = rw.GetAtomWithIdx(idx[v]).GetAtomicNum() in COORDINATION_METALS
        if u_metal != v_metal:
            donor, metal = (v, u) if u_metal else (u, v)
            rw.AddBond(idx[donor], idx[metal], Chem.BondType.DATIVE)
        else:
            rw.AddBond(idx[u], idx[v], _KEKULE_ORDERS[order])

    mol = rw.GetMol()
    total = graph.graph["total_charge"]
    if not quick and Chem.GetFormalCharge(mol) != total:  # the reader takes the total as these charges' sum
        raise ValueError(
            f"xyzgraph {xyzgraph.__version__} gave total charge {total} but atom charges summing to "
            f"{Chem.GetFormalCharge(mol)}; install the version rxembed requires"
        )
    conf = Chem.Conformer(mol.GetNumAtoms())
    for node, data in graph.nodes(data=True):
        x, y, z = data["position"]
        conf.SetAtomPosition(idx[node], Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    if quick:
        rw = Chem.RWMol(mol)
        for h, x in _hydrogen_bond_legs(mol):
            remove_bond(rw, h, x)
        mol = rw.GetMol()
    # A tolerant intermediate graph: bond-order ranking owns valence, so only this probe is silenced.
    with rdBase.BlockLogs():
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )
    return mol


def _from_xyz2mol(path, charge):
    """Read connectivity and bond orders from the vendored perceiver."""
    before = _coordinates(path)
    if not isinstance(path, str):  # the vendored reader takes atomic numbers and coordinates in place of a file
        path = ([_PT.GetAtomicNumber(symbol) for symbol, _xyz in path], [list(xyz) for _symbol, xyz in path])
    with rdBase.BlockLogs():
        out = get_tmc_mol(path, charge)[0]
    _assert_same_atoms(before, out)
    return out


def _from_rdkit_connectivity(path, charge):
    """Read connectivity with RDKit's native connect-the-dots algorithm."""
    mol = _coordinates(path)
    rdDetermineBonds.DetermineConnectivity(mol, charge=charge, useVdw=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _from_rdkit(path, charge):
    """Read an organic molecule with RDKit's bond perceiver."""
    mol = _coordinates(path)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except ValueError as exc:
        raise ValueError(
            f"RDKit could not perceive bonds in {path if isinstance(path, str) else 'the atoms'} ({exc}); use "
            "connectivity='xyzgraph'"
        ) from exc
    return mol


def _xyz_source(source):
    """Return `source` as a path string, or as checked in-memory ``(symbol, (x, y, z))`` atoms."""
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    try:
        atoms = [(str(symbol), tuple(float(value) for value in xyz)) for symbol, xyz in source]
    except (TypeError, ValueError) as exc:
        raise TypeError("read_xyz takes an .xyz path or atoms as (symbol, (x, y, z)) pairs") from exc
    for symbol, xyz in atoms:
        if len(xyz) != 3 or not all(map(math.isfinite, xyz)):  # noqa: PLR2004 - three coordinates
            raise ValueError(f"atom {symbol} needs three finite coordinates, got {xyz}")
        if symbol not in _ELEMENTS:
            raise ValueError(f"{symbol!r} is not an element symbol")
    if not atoms:
        raise ValueError("read_xyz got no atoms")
    return atoms


def _coordinates(source):
    """Read the atoms and coordinates without inferring bonds, from a path or in-memory atoms."""
    if isinstance(source, str):
        mol = Chem.MolFromXYZFile(source)
    else:
        mol = Chem.MolFromXYZBlock(f"{len(source)}\n\n" + "".join(f"{s} {x} {y} {z}\n" for s, (x, y, z) in source))
    if mol is None:
        raise ValueError(f"could not read {source if isinstance(source, str) else 'the given atoms'} as an .xyz")
    return mol


def _assert_same_atoms(before, after):
    """Reject xyz2mol reassembly that drops or reorders the input atoms, or moves or drops their coordinates."""
    want = [a.GetAtomicNum() for a in before.GetAtoms()]
    same = want == [a.GetAtomicNum() for a in after.GetAtoms()]
    if same and before.GetNumConformers():  # a graph read has no coordinates to keep
        placed = before.GetConformer().GetPositions()
        same = after.GetNumConformers() > 0 and np.allclose(
            placed, after.GetConformer().GetPositions(), rtol=0.0, atol=1e-6
        )
    if not same:
        raise ValueError(f"xyz2mol changed the {len(want)} input atoms, their order, or their coordinates")


def _apply_metal_charges(mol, charges):
    """Validate and apply an explicit oxidation-state allocation."""
    if charges is None:
        return
    supplied = {int(index): int(value) for index, value in charges.items()}
    metals = set(metal_indices(mol))
    if set(supplied) != metals:
        raise ValueError(f"metal_charges must name every metal atom index {sorted(metals)}, got {sorted(supplied)}")
    for index, value in supplied.items():
        mol.GetAtomWithIdx(index).SetFormalCharge(value)


class _BondOrders(NamedTuple):
    """Hold a Mol whose bond orders are read, the connectivity it now has, and the backend that ordered it, or None."""

    mol: Chem.Mol
    connectivity: str
    ordered_by: str | None


def _assign_bond_orders(mol, path, charge, connectivity, requested, allow_fallback):
    """Apply the requested bond-order backend, falling back only where allowed; return `_BondOrders`."""
    if requested is None:
        return _BondOrders(mol, connectivity, None)
    if requested != "xyz2mol" or connectivity == "xyz2mol":
        return _BondOrders(mol, connectivity, connectivity)
    if not any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in mol.GetAtoms()):
        if not allow_fallback:
            raise ValueError(
                "bond_orders='xyz2mol' cannot rank this graph; use fallback=True or bond_orders='xyzgraph'"
            )
        if connectivity == "xyzgraph":
            unmodelled = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS})
            reason = f"has no charge model for {', '.join(unmodelled)}" if unmodelled else "is metal-only"
            logger.warning("read_xyz: xyz2mol bond-order assignment %s; keeping xyzgraph bond orders", reason)
            return _BondOrders(mol, connectivity, connectivity)
        logger.warning("read_xyz: xyz2mol bond-order assignment is metal-only; using RDKit for this graph")
    backend, mol = mol, _without_hydrogen_bonds(mol)
    try:
        ranked, ranker = _rank_orders(mol, charge)
    except ValueError as first:
        if not allow_fallback:
            raise
        return _fallback_reading(path, mol, backend, charge, connectivity, first)
    return _BondOrders(ranked, connectivity, ranker)


def _without_hydrogen_bonds(mol):
    """Return `mol` without its hydrogen bonds, warning when it drops any.

    A hydrogen bond is not constitution, and xyz2mol cannot hold one; RDKit and xyz2mol connectivity, and a caller's
    graph, draw them as bonds.
    """
    if not (legs := _hydrogen_bond_legs(mol)):
        return mol
    logger.warning(
        "perceive: dropped hydrogen bond(s) %s before ranking bond orders; a hydrogen bond is not a bond",
        ", ".join(f"H{h}...{mol.GetAtomWithIdx(x).GetSymbol()}{x}" for h, x in legs),
    )
    rw = Chem.RWMol(mol)
    for h, x in legs:
        remove_bond(rw, h, x)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def _fallback_reading(path, mol, backend, charge, connectivity, first):
    """Return the `_BondOrders` read when xyz2mol finds no bond orders on the selected graph.

    xyz2mol's own connectivity comes next. When that fails too, the selected graph is kept with the reading that has
    fewer radical electrons: xyz2mol's with a metal's excess charge moved onto donors as radicals, or xyzgraph's own
    charges when they reach the total within each metal's valence electrons. A tie takes xyz2mol's.
    """
    try:
        joint = _from_xyz2mol(path, charge)
    except (ImportError, OSError, RuntimeError, ValueError) as second:
        readings = []
        with contextlib.suppress(ValueError):
            readings.append(_rank_orders(mol, charge, radicals=True))
        metals = [backend.GetAtomWithIdx(index) for index in metal_indices(backend)]
        if (
            connectivity == "xyzgraph"
            and Chem.GetFormalCharge(backend) == charge
            and all(metal.GetFormalCharge() <= _PT.GetNOuterElecs(metal.GetAtomicNum()) for metal in metals)
        ):
            own = Chem.Mol(backend)
            kept = ", ".join(f"{metal.GetSymbol()}{metal.GetFormalCharge():+d}" for metal in metals)
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
        reading, order_by = min(readings, key=lambda item: sum(a.GetNumRadicalElectrons() for a in item[0].GetAtoms()))
        return _BondOrders(reading, connectivity, order_by)
    logger.warning(
        "read_xyz: xyz2mol found no bond orders on the %s graph (%s); using xyz2mol connectivity", connectivity, first
    )
    return _BondOrders(joint, "xyz2mol", "xyz2mol")


def _hydrogen_bond_legs(mol):
    """Return ``(hydrogen, acceptor)`` for each hydrogen bond `mol` holds as a bond.

    A hydrogen bonded to two or more nonmetals keeps its first leg; each other leg is a hydrogen bond when the acceptor
    keeps a lone pair, else a three-centre bond (B-H-B) that stays.
    """
    metals = set(metal_indices(mol))
    contact = Chem.RWMol(mol)
    candidates = []
    for h, ordered in hydrogen_legs(mol, COORDINATION_METALS).items():
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


def _rank_orders(mol, charge, radicals=False):
    """Return `mol` with bond orders and charges ranked on its connectivity, and the backend that ranked them.

    RDKit ranks a graph with no metal xyz2mol has a charge model for (no lanthanides or actinides), xyz2mol any other.
    `mol` needs no conformer; one only breaks ties.
    """
    if not any(atom.GetAtomicNum() in TRANSITION_METALS_NUM for atom in mol.GetAtoms()):
        out = Chem.Mol(mol)
        try:
            rdDetermineBonds.DetermineBondOrders(out, charge=charge, embedChiral=False)
            Chem.SanitizeMol(out)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"RDKit could not assign bond orders on this connectivity ({exc}); check charge= or give a SMILES"
            ) from exc
        return out, "rdkit"
    graph, contacts = _flatten_for_xyz2mol(mol)
    try:
        with rdBase.BlockLogs():
            out = get_tmc_mol(None, charge, graph=(graph, None), radicals=radicals)[0]
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
        return result, "xyz2mol"
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"xyz2mol could not assign bond orders on the selected connectivity: {exc}") from exc


def _flatten_for_xyz2mol(mol):
    """Return the graph xyz2mol searches and the ``(hydrogen, partner, bond type)`` legs it cannot hold.

    xyz2mol gives hydrogen one valence, so a bridging hydrogen keeps its first leg and each other leg is cut, to come
    back after the search as a dative bond to a metal or a zero-order contact to a nonmetal. In the searched graph M-L
    bonds are dative, every other bond single and charges cleared.
    """
    contacts = [
        (
            h,
            other,
            Chem.BondType.DATIVE
            if mol.GetAtomWithIdx(other).GetAtomicNum() in TRANSITION_METALS_NUM
            else Chem.BondType.ZERO,
        )
        for h, ordered in hydrogen_legs(mol, TRANSITION_METALS_NUM).items()
        for other in ordered[1:]
    ]
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
    return graph, contacts


def _reject_unbonded_bridging_hydrogens(mol):
    """Refuse a hydrogen left unbonded within bonding reach of two or more heavy atoms.

    That is a 3c-2e bridge (B-H-B, Al-H-Al, M-H-M) outside the two-centre donor model; a dropped terminal hydrogen sits
    near one heavy atom, and a free hydride near none.
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
            f"3-centre bridge(s) {details} lie outside the two-centre donor model; give an explicit donor graph"
        )


def _drop_bridgehead_bonds(mol):
    """Return `mol` without the M-X bonds a coordinate reader draws across a chelate ring's diagonal.

    X is a candidate when two of its unbonded neighbours bind the same metal: it is then the far corner of a ring
    M-D1-X-D2 (a kappa2 carboxylate or amidinate C, the P of S2PR2, the B of kappa2-BH4), and RDKit and xyz2mol bond it.
    Without a lone pair, a saturated X always drops and a linear X, with a pi orbital in the chelate plane, stays. Any
    other X drops when its donors bind through their own lone pairs, no ring holds X with all of them, no open path puts
    a carbon beside it, and X sits no closer to M than the longest M-D bond, the ring's own folding limit. Each graph
    test guards a case geometry alone gets wrong: an agostic C, a ring face, a linear sp donor, an allyl-type carbon. A
    graph without a conformer keeps a bond only geometry could judge. Logs one warning naming every dropped bond.
    """
    metals = set(metal_indices(mol))
    if not metals:
        return mol
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    rings = [frozenset(ring) for ring in Chem.GetSymmSSSR(ligand_graph(rw, metals))]  # the metal-free rings

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
                and rw.GetNumConformers()
                and far(metal, xi, donors)
            ):
                bad.append((xi, x.GetSymbol(), metal))
    if not bad:
        return mol
    for xi, _sym, metal in bad:
        remove_bond(rw, xi, metal)
    logger.warning(
        "perceive: dropped bridgehead bond(s) %s: chelate-ring diagonals, not donors",
        ", ".join(f"{sym}{xi}-{metal}" for xi, sym, metal in bad),
    )
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out
