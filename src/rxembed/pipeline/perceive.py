"""Read a user source (SMILES string / ``.xyz`` path) into an RDKit Mol.

Input adaptation only: no constraints, no embedding. This is the leaf the embed dispatch and the metal
isomer load-in hand a user source to before the core `embed()` engine (which already takes a Mol)
ever sees it, with xyzgraph an optional guarded extra that ``_xyz_to_mol`` falls back from to RDKit's own
perception.

It reaches back into ``rxembed`` for exactly one thing, ``utils``, the numpy + rdkit leaf that imports no
sibling: writing a stereo tag is the one job here that has a correctness rule attached, and duplicating that
rule is how it drifts. Nothing else, because the cycle this module was split out to break
(metal -> dispatch -> metal) comes back through any import with a sibling behind it.
"""

from __future__ import annotations

from rdkit import Chem

from rxembed.utils import assign_stereo_from_3d

_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic


def _xyz_to_mol(path, charge=0):
    """Read an ``.xyz`` into an RDKit Mol with perceived bonds and a conformer; robust for metals and TSs.

    Bonds come from xyzgraph (transition-metal-aware perception plus a bond-order optimiser), not RDKit's
    organic-only ``rdDetermineBonds`` (which raises on a metal). The graph is converted to an RWMol and
    only *leniently* sanitised (ring perception, but no valence/property checks that choke on a metal),
    so a metal complex or a TS from xyz Just Works. Index addressing always works; SMARTS works too for the
    organic part. Pass `charge` for a charged species. Falls back to ``rdDetermineBonds`` only if xyzgraph
    is unavailable. Do not pass ``quick=True`` to ``build_graph`` here: it skips bond-order and charge
    perception and would return all-single bonds.)
    """
    try:
        import xyzgraph
    except ImportError:
        from rdkit.Chem import rdDetermineBonds

        mol = Chem.MolFromXYZFile(path)
        if mol is None:
            raise ValueError(f"could not read {path} as an .xyz") from None
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except ValueError as exc:  # RDKit's perceiver is organic-only: a metal or a TS core lands here
            raise ValueError(
                f"RDKit could not perceive bonds in {path}; pip install 'rxembed[perceive]' for xyzgraph, "
                f"which handles metals and stretched TS bonds ({exc})"
            ) from exc
        return mol
    from rdkit.Chem import BondType, Conformer
    from rdkit.Geometry import Point3D

    g = xyzgraph.build_graph(path, charge=charge, kekule=True)  # integer bond orders (no 1.5)
    order = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE}
    rw = Chem.RWMol()
    idx = {}
    for n, d in sorted(g.nodes(data=True)):
        a = Chem.Atom(int(d["atomic_number"]))
        a.SetFormalCharge(round(d.get("formal_charge", 0) or 0))
        a.SetNoImplicit(True)  # xyz is fully explicit (incl. H)
        idx[n] = rw.AddAtom(a)
    for u, v, d in g.edges(data=True):
        bo = d.get("bond_order", 1.0)
        if abs(bo - 1.5) < _AROMATIC_BO_TOL:  # aromatic (only if kekule fell through)
            b = rw.AddBond(idx[u], idx[v], BondType.AROMATIC) - 1
            rw.GetBondWithIdx(b).SetIsAromatic(True)
            rw.GetAtomWithIdx(idx[u]).SetIsAromatic(True)
            rw.GetAtomWithIdx(idx[v]).SetIsAromatic(True)
        else:
            rw.AddBond(idx[u], idx[v], order.get(round(bo), BondType.SINGLE))
    mol = rw.GetMol()
    conf = Conformer(mol.GetNumAtoms())
    for n, d in g.nodes(data=True):
        x, y, z = d["position"]
        conf.SetAtomPosition(idx[n], Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(
        mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
    )  # rings yes, valence checks no
    try:
        assign_stereo_from_3d(mol)  # point R/S + E/Z from the geometry -> graph tags, so the embed
    except Exception:  # preserves them exactly as it would for a SMILES @/@@ (an .xyz behaves like SMILES).
        pass  # Through `utils`, because the sanitize above has already rewritten M-L bonds as dative and
    return mol  # RDKit's 3D writer alone leaves those out of the tag's basis; see there.


def parse_smiles(smi):
    """Parse a SMILES to a Mol, raising a clear error instead of returning ``None`` (which crashes downstream)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smi!r}")
    return mol


read_xyz = _xyz_to_mol  # the public name: five notebooks already import the private one
