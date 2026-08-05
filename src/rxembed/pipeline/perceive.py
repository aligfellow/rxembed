"""Read an ``.xyz`` into an RDKit Mol: the one input adaptation that needs perception.

Input adaptation only, no constraints and no embedding. This is the leaf the embed dispatch and the
metal isomer load-in hand a user source to before the core `embed()` engine, which already takes a
Mol, ever sees it. Reading a SMILES needs none of this and lives in `rxembed.metal_smiles`, next to
the writer whose atom order it has to agree with.

Perception is two decisions and there are two perceivers, each better at one of them, so they are
separate arguments rather than one mode. Which atoms are bonded is `connectivity`; what those bonds
are is `bond_orders`.

Reaching back into ``rxembed`` is limited to two leaves, because the cycle this module was split out
to break (metal -> dispatch -> metal) returns through any import with a sibling behind it.
``rxembed.utils`` holds the rule for writing a stereo tag, which has to live in one place or it
drifts. `xyz2mol_tmc` is the other perceiver, and imports nothing but its own vendored core.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from rdkit import Chem

from rxembed.utils import assign_stereo_from_3d

_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic
_PERCEIVERS = {"xyzgraph", "xyz2mol"}
logger = logging.getLogger("rxembed")


def _xyz_to_mol(path, charge=0, connectivity="xyzgraph", bond_orders="xyzgraph"):
    """Read an ``.xyz`` into a Mol with perceived bonds and a conformer; robust for metals and TSs.

    `connectivity` is ``"xyzgraph"`` or ``"xyz2mol"``, which uses a wider tolerance and so picks up
    a haptic contact xyzgraph can miss. `bond_orders` is ``"xyzgraph"`` for its optimiser or
    ``"xyz2mol"`` for `xyz2mol_tmc`'s search, which pools candidates across a charge ladder and ranks
    them instead of taking the first that fits.

    ``bond_orders="xyzgraph"`` needs ``connectivity="xyzgraph"``: that optimiser runs inside
    ``build_graph`` and cannot be given another perceiver's bonds. ``"xyz2mol"`` applies only to a
    transition-metal complex, since `xyz2mol_tmc` is the TMC door; a metal-free TS keeps xyzgraph's
    orders rather than failing.

    The result is only leniently sanitised -- rings yes, valence checks no -- so a metal or a
    stretched TS core survives. Index addressing always works, and SMARTS works on the organic part.
    """
    for argument, value in (("connectivity", connectivity), ("bond_orders", bond_orders)):
        if value not in _PERCEIVERS:
            raise ValueError(f"{argument} must be 'xyzgraph' or 'xyz2mol', got {value!r}")
    if bond_orders == "xyzgraph" and connectivity != "xyzgraph":
        raise ValueError(
            "bond_orders='xyzgraph' needs connectivity='xyzgraph': that optimiser runs inside "
            "build_graph and cannot be given another perceiver's bonds. Use bond_orders='xyz2mol'."
        )
    mol, perceived_by = _with_fallback(path, charge, connectivity)
    if perceived_by == "xyzgraph":
        if bond_orders == "xyz2mol":
            mol = _rank_orders(mol, charge)
    with contextlib.suppress(Exception):  # a tag we cannot read is not a reason to fail the read
        assign_stereo_from_3d(mol)  # the one door, so an M-L dative sits in the basis its readers use
    return mol


read_xyz = _xyz_to_mol  # the public name: five notebooks already import the private one


def _from_xyz2mol(path, charge):
    """Read connectivity and bond orders from the vendored perceiver."""
    from .xyz2mol_tmc import get_tmc_mol

    before = _coordinates(path)
    out = get_tmc_mol(path, charge)[0]
    _assert_same_atoms(before, out)
    return out


def _from_xyzgraph(path, charge):
    """Read connectivity and bond orders from xyzgraph, as a Mol with a conformer."""
    import xyzgraph
    from rdkit.Chem import BondType, Conformer
    from rdkit.Geometry import Point3D

    # Never quick=True: it skips bond-order and charge perception and returns all-single bonds.
    graph = xyzgraph.build_graph(path, charge=charge, kekule=True)  # integer orders, no 1.5
    orders = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE}
    rw, idx = Chem.RWMol(), {}
    for node, data in sorted(graph.nodes(data=True)):
        atom = Chem.Atom(int(data["atomic_number"]))
        atom.SetFormalCharge(round(data.get("formal_charge", 0) or 0))
        atom.SetNoImplicit(True)  # an xyz is fully explicit, hydrogens included
        idx[node] = rw.AddAtom(atom)
    for u, v, data in graph.edges(data=True):
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
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    return mol


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
            f"pip install 'rxembed[perceive]' for xyzgraph ({exc})"
        ) from exc
    return mol


def _with_fallback(path, charge, backend):
    """Try a perceiver, warning before the graph-changing fallback."""
    primary = _from_xyzgraph if backend == "xyzgraph" else _from_xyz2mol
    try:
        return primary(path, charge), backend
    except Exception as first:
        if backend == "xyzgraph":
            from .xyz2mol_tmc import TRANSITION_METALS_NUM

            raw = _coordinates(path)
            has_metal = any(a.GetAtomicNum() in TRANSITION_METALS_NUM for a in raw.GetAtoms())
            fallback, name = (_from_xyz2mol, "xyz2mol") if has_metal else (_from_rdkit, "RDKit")
        else:
            fallback, name = _from_xyzgraph, "xyzgraph"

        state = "unavailable" if isinstance(first, ImportError) else f"failed ({first})"
        logger.warning("read_xyz: %s %s; using %s", backend, state, name)
        try:
            return fallback(path, charge), name
        except Exception as second:
            raise ValueError(f"{backend} {state}; {name} fallback also failed ({second})") from second


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
        raise ValueError(
            f"xyz2mol returned {len(got)} atoms for {len(want)}, or in another order; its reassembly is single-metal"
        )


def _rank_orders(mol, charge):
    """Re-assign bond orders and charges on xyzgraph's bonds, keeping which atoms are bonded.

    The bonds are flattened to single and the charges cleared, so `xyz2mol_tmc` is given the
    connectivity and nothing else. ``get_basic_mol`` normally seeds three structural charges on the
    way in and they are re-applied here, since the search downstream is calibrated against them.
    A structure with no transition metal is returned untouched.

    A hydrogen with two bonds is held aside first. xyz2mol allows hydrogen exactly one valence
    (``atomic_valence[1] = [1]``), so a side-on H2, a bridging hydride or a shared proton has no
    assignment its search can make -- while xyzgraph draws both contacts, and they are real. The
    longer contact is removed before the search and written back as a dative afterwards, which is
    the same order `xyz2mol_tmc` uses for a hydride bridge and an agostic C-H, and the same
    convention `metal_smiles` writes: keep the shortest bond, donate through the rest.
    """
    from .xyz2mol_tmc import TRANSITION_METALS_NUM, get_tmc_mol

    if not any(a.GetAtomicNum() in TRANSITION_METALS_NUM for a in mol.GetAtoms()):
        return mol

    flat = Chem.RWMol(mol)
    pos = mol.GetConformer().GetPositions()
    donated = []  # (hydrogen, acceptor) contacts removed before the search, re-added after
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() <= 1:
            continue
        h = atom.GetIdx()
        far = sorted((n.GetIdx() for n in atom.GetNeighbors()), key=lambda n: sum((pos[n] - pos[h]) ** 2))
        for other in far[1:]:
            flat.RemoveBond(h, other)
            donated.append((h, other))

    for bond in flat.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in flat.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        heavy = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in TRANSITION_METALS_NUM)
        seeded = {(7, 4): 1, (8, 3): 1, (5, 4): -1}  # get_basic_mol's, off the metal-free valence
        atom.SetFormalCharge(seeded.get((atom.GetAtomicNum(), heavy), 0))

    graph = flat.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    coords = graph.GetConformer().GetPositions().tolist()
    try:
        out = get_tmc_mol(None, charge, graph=(graph, coords))[0]
        # Atom-for-atom, in order. The re-added contacts below address atoms by index, and
        # get_tmc_mol only restores the input order through a single metal centre -- on a
        # two-metal complex it drops one, which would otherwise land those bonds silently on
        # the wrong atoms rather than failing.
        _assert_same_atoms(mol, out)
        rw = Chem.RWMol(out)
        for h, other in donated:
            rw.AddBond(h, other, Chem.BondType.DATIVE)
        rw.UpdatePropertyCache(strict=False)
        result = rw.GetMol()
        before = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        after = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in result.GetBonds()}
        if before != after:
            raise ValueError(f"xyz2mol changed connectivity: lost {before - after}, added {after - before}")
        return result
    except Exception as exc:
        logger.warning(
            "read_xyz: xyz2mol bond-order assignment failed (%s); keeping input bond orders",
            exc,
        )
        return mol
