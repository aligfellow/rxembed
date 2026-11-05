"""Describe, cluster and deduplicate an ensemble's conformers; never move or reorder an atom.

Clustering and the landscape read one per-conformer latent (`features`). `apply` deduplicates the energy-sorted
conformers by one `DEDUP` row, each keeping the lowest-energy copy: 'rmsd' matches graph-equivalent atoms within an
energy window, while 'energy' merges near-degenerate energies alone.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import TorsionFingerprints, rdMolTransforms

from rxembed.graph import COORDINATION_METALS, frag_map, metal_index
from rxembed.identity.reading import observed_labels
from rxembed.metal.shape import coordinating_atoms
from rxembed.perception.contacts import observed_contacts
from rxembed.utils import bond_angle, require, without_atom_maps


def rotatable_quads(mol):
    """Return one heavy-atom dihedral (i, a, b, j) per rotatable bond, stable across conformers.

    RDKit's non-ring torsion list judges a terminal atom by its heavy neighbours, so a methyl or hydroxyl rotor stays
    out.
    """
    non_ring, _ring = TorsionFingerprints.CalculateTorsionLists(mol)
    return [quads[0] for quads, _max_dev in non_ring]


def dihedrals(mol, conf_id, quads=None):
    """Return (cos, sin) of each rotatable-bond dihedral: a frame-invariant feature vector."""
    quads = rotatable_quads(mol) if quads is None else quads
    conf = mol.GetConformer(conf_id)
    v = []
    for q in quads:
        t = np.radians(rdMolTransforms.GetDihedralDeg(conf, *q))
        v += [np.cos(t), np.sin(t)]
    return np.array(v) if v else np.zeros(1)


_EPS = 1e-9  # std floor for z-scoring
_MIN_CLUSTER = 3  # conformers HDBSCAN needs to call a set of them one mode


def metal_donors(mol, ids):
    """Return the donors of the first metal, or None when there is no metal.

    A bonded metal's donors are its non-metal neighbours; a bondless one's (a wrapped Mol) are read on the first of
    `ids`. The two are never mixed, so every conformer shares one descriptor schema.
    """
    m = metal_index(mol)
    if m is None:
        return None
    neighbors = list(mol.GetAtomWithIdx(m).GetNeighbors())
    if neighbors:
        return sorted(n.GetIdx() for n in neighbors if n.GetAtomicNum() not in COORDINATION_METALS)
    return sorted(coordinating_atoms(mol, mol.GetConformer(ids[0]).GetPositions(), m))


def _metal_features(mol, ids):
    """Return the L-M-L angles per conformer, which capture the coordination polyhedron, or None without a metal."""
    pairs = list(itertools.combinations(metal_donors(mol, ids) or (), 2))
    if not pairs:
        return None
    m, out = metal_index(mol), []
    for i in ids:
        pos = mol.GetConformer(i).GetPositions()
        out.append([bond_angle(pos[a], pos[m], pos[b]) for a, b in pairs])
    return np.array(out)


def _nci_features(contacts, fitted):
    """Return a binary inter-fragment NCI-contact fingerprint per conformer, or None when no contact is made.

    Its columns are the contacts the first `fitted` conformers make, None when they make none.
    """
    sigs = [{(c.kind, frozenset((*c.site_a, *c.site_b))) for c in found} for found in contacts]
    universe = sorted({k for s in sigs[:fitted] for k in s})
    if not universe:
        return None
    return np.array([[float(k in s) for k in universe] for s in sigs])


def _relpose_features(mol, ids, fitted):
    """Return where each fragment sits relative to the largest, z-scored on the first `fitted` conformers.

    This rigid-body freedom is invisible to dihedrals, and without it distinct encounter geometries collapse to one
    mode. Distances only, so it is frame-free, though it cannot tell the two faces of a planar anchor apart. None for a
    single fragment or a metal, whose relative pose is the L-M-L block.
    """
    if metal_index(mol) is not None:
        return None
    frags = Chem.GetMolFrags(mol)
    if len(frags) < 2:  # noqa: PLR2004  a single fragment has no relative pose to describe
        return None
    hf = [[a for a in f if mol.GetAtomWithIdx(a).GetAtomicNum() > 1] or list(f) for f in frags]
    anchor = max(range(len(frags)), key=lambda i: (len(hf[i]), -min(frags[i])))  # largest, ties->low idx
    a_atoms = hf[anchor]
    smi = [Chem.MolToSmiles(m) for m in Chem.GetMolFrags(without_atom_maps(mol), asMols=True, sanitizeFrags=False)]
    others = [i for i in range(len(frags)) if i != anchor]
    groups = [[o for o in others if smi[o] == s] for s in sorted({smi[o] for o in others})]  # fixed order
    rows = []
    for cid in ids:
        pos = mol.GetConformer(cid).GetPositions()
        ac = pos[a_atoms].mean(0)
        feat = []
        for grp in groups:
            poses = []
            for other in (hf[o] for o in grp):
                oc = pos[other].mean(0)
                poses.append(
                    [
                        float(np.linalg.norm(oc - ac)),
                        *sorted(float(np.linalg.norm(pos[a] - oc)) for a in a_atoms),
                        *sorted(float(np.linalg.norm(pos[h] - ac)) for h in other),
                    ]
                )
            feat += [x for pose in sorted(poses) for x in pose]
        rows.append(feat)
    mat = np.array(rows, float)
    sd = mat[:fitted].std(0)
    sd[sd < _EPS] = 1.0
    return (mat - mat[:fitted].mean(0)) / sd


def _blocks(mol, ids, fitted, contacts):
    """Return the latent of conformers `ids` as ``(name, array)`` blocks, fitted on the first `fitted`.

    'dihedral' always; 'metal' (L-M-L angles) for a metal with two or more donors; 'relpose' for several fragments
    without a metal; 'nci' (contact fingerprint) where, without a metal, a fitted conformer makes an inter-fragment
    contact.
    """
    quads = rotatable_quads(mol)
    blocks = [("dihedral", np.array([dihedrals(mol, i, quads) for i in ids]))]
    fm = _metal_features(mol, ids)
    if fm is not None:
        blocks.append(("metal", fm))
    rp = _relpose_features(mol, ids, fitted)
    if rp is not None:
        blocks.append(("relpose", rp))
    fn = _nci_features(contacts, fitted)
    if fn is not None:
        blocks.append(("nci", fn))
    return blocks


class Features(NamedTuple):
    """Hold one selection's matrix, block names and observed contacts in matrix row order.

    Keep this snapshot within one operation on unchanged coordinates; it is not an Ensemble cache.
    """

    matrix: np.ndarray
    blocks: tuple[str, ...]
    contacts: tuple

    @property
    def kind(self):
        """Return what a mode means: the most specific feature block's description."""
        return _MODE_KIND[self.blocks[-1]]


def features(mol, ids, placed=()):
    """Return one `Features` snapshot of conformers `ids`, with conformers `placed` projected into it.

    The blocks stand side by side, each scaled to unit variance when there are several. Every statistic is read on `ids`
    alone, so a row of `ids` is the same with or without `placed`.
    """
    fitted = len(ids)
    rows = [*ids, *placed]
    contacts = tuple(observed_contacts(mol, rows, interfragment=True)) if metal_index(mol) is None else ()
    blocks = _blocks(mol, rows, fitted, contacts)
    matrix = blocks[0][1]
    if len(blocks) > 1:
        normed = [b / np.sqrt(b[:fitted].var(0).sum()) if b.size and b[:fitted].var(0).sum() else b for _, b in blocks]
        matrix = np.hstack(normed)
    return Features(matrix=matrix, blocks=tuple(name for name, _ in blocks), contacts=contacts)


# what each latent block means when it is the most specific one present: the "mode" kind
_MODE_KIND = {
    "metal": "ligand arrangement",
    "nci": "contact pattern",
    "relpose": "relative arrangement",
    "dihedral": "conformer family",
}


def mode_signature(mol, ids, snapshot):
    """Return each conformer's hashable binding-mode signature, or None without an 'nci' or 'metal' block.

    A signature is the contact kinds and the fragments they bridge, plus the metal's arrangement label, so two
    conformers with one signature are one binding mode whatever their torsions.
    """
    blocks = snapshot.blocks
    if "nci" not in blocks and "metal" not in blocks:
        return None
    sigs = [[] for _ in ids]
    if "nci" in blocks:
        fmap = frag_map(mol)
        for sig, found in zip(sigs, snapshot.contacts, strict=True):
            bridged = {(c.kind, tuple(sorted({fmap[a] for a in (*c.site_a, *c.site_b)}))) for c in found}
            sig.append(("nci", frozenset(bridged)))
    if "metal" in blocks:
        for sig, label in zip(sigs, observed_labels(mol, metal_index(mol), ids), strict=True):
            sig.append(("metal", label))
    return [tuple(s) for s in sigs]


def cluster_on(feats):
    """Return HDBSCAN's cluster label per row of `feats`, -1 for noise; fewer than `_MIN_CLUSTER` rows are one mode."""
    if len(feats) < _MIN_CLUSTER:
        return np.zeros(len(feats), dtype=int)
    hdbscan = require("sklearn.cluster", "workflow").HDBSCAN
    return np.asarray(
        hdbscan(min_cluster_size=_MIN_CLUSTER, cluster_selection_method="leaf", copy=True).fit_predict(feats)
    )  # copy=True: future sklearn default, silences warning


class _Compared(NamedTuple):
    """Hold what the RMSD comparison reads of an ensemble: its Mol, its symmetry classes and its typed bonds."""

    mol: Chem.Mol
    classes: list[np.ndarray]  # atom index arrays sharing one canonical rank, two or more atoms each, smallest first
    adjacency: np.ndarray  # (atom, atom) bond type code, 0 where no bond
    bonds: np.ndarray  # (bond, 2) lower and higher atom index of every bond
    indices: np.ndarray  # compared atom to original atom


class _View(NamedTuple):
    """Hold what the comparison reads of one conformer: centred positions, principal axes and anchor frames."""

    positions: np.ndarray
    axes: np.ndarray
    frames: np.ndarray  # (pair, 3, 3) the frame of each anchor pair that spans a plane, the first a reference's target


class _Fits(NamedTuple):
    """Hold fitted RMSDs and largest deviations, in angstrom, and the reference each fit is onto."""

    rmsd: np.ndarray
    maximum: np.ndarray
    owners: np.ndarray
    orders: np.ndarray

    def passing(self, max_rmsd):
        """Select fits inside the RMSD and twice-RMSD displacement limits."""
        return (self.rmsd < max_rmsd) & (self.maximum < 2 * max_rmsd)


_ASSIGN_ROUNDS = 3  # matching rounds from one start: measured verdicts held from 1, the RMSD stopped improving at 3
_FRAME_FLOOR = 1e-2  # sine of the angle below which an anchor pair, seen from the centroid, defines no frame
_SIGNS = np.array([np.diag(signs) for signs in itertools.product((1.0, -1.0), repeat=3)])
_PERMUTATIONS = {size: np.array([*itertools.permutations(range(size))]) for size in (2, 3, 4)}  # matched by search


def _compared(mol, hydrogens):
    """Return what the RMSD comparison reads of `mol`: no ordinary hydrogens unless `hydrogens`, and its classes.

    RDKit's `RemoveHs` keeps the hydrogens it will not drop (isotope-labelled, molecular, bridging, stereo-defining and
    hydrides), so H2 and a hydride's site stay compared. A class is the atoms canonical ranking cannot tell apart.
    """
    indexed = Chem.Mol(mol)
    for atom in indexed.GetAtoms():
        atom.SetIntProp("comparison_index", atom.GetIdx())
    reduced = indexed if hydrogens else Chem.RemoveHs(indexed, sanitize=False)
    reduced.UpdatePropertyCache(strict=False)
    ranks = np.asarray(Chem.CanonicalRankAtoms(reduced, breakTies=False, includeAtomMaps=False, includeChirality=False))
    classes = sorted((atoms for rank in np.unique(ranks) if len(atoms := np.flatnonzero(ranks == rank)) > 1), key=len)
    adjacency = np.zeros((reduced.GetNumAtoms(),) * 2, int)
    for bond in reduced.GetBonds():
        adjacency[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = adjacency[
            bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()
        ] = int(bond.GetBondType()) + 1
    indices = np.array([atom.GetIntProp("comparison_index") for atom in reduced.GetAtoms()])
    return _Compared(
        mol=reduced, classes=classes, adjacency=adjacency, bonds=np.argwhere(np.triu(adjacency)), indices=indices
    )


def _rotations(p, q):
    """Return the proper rotation `r` that best takes centred rows `p` onto centred rows `q` as ``p @ r``.

    `p` may be a stack of row sets, one rotation each.
    """
    u, _, vt = np.linalg.svd(np.swapaxes(p, -1, -2) @ q)
    u[..., 2] *= np.sign(np.linalg.det(u @ vt))[..., None]
    return u @ vt


def _frame(u, v):
    """Return the right-handed axes along `u` and toward `v` as columns, or None when they define no plane.

    None when either vector is zero (an anchor at the centroid) or the two nearly align, so no start is ever NaN.
    """
    normal = np.cross(u, v)
    if not np.linalg.norm(normal) > _FRAME_FLOOR * np.linalg.norm(u) * np.linalg.norm(v):
        return None
    first, third = u / np.linalg.norm(u), normal / np.linalg.norm(normal)
    return np.column_stack([first, np.cross(third, first), third])


def _views(compared, ids):
    """Return the `_View` of each conformer in `ids`, read once for all its pair comparisons.

    The anchor pairs come from the smallest class, so a reference's first frame has its image among a probe's, and a
    ring gives one start per automorphism.
    """
    anchors = compared.classes[0] if compared.classes else np.arange(0)
    path = require("scipy.sparse.csgraph", "workflow").shortest_path  # from the anchors only: all pairs is cubic
    apart = path(compared.adjacency, unweighted=True, indices=anchors)  # (anchor, atom) bonds apart
    span = apart[0, anchors[1:]].min() if len(anchors) else 0
    pairs = [(a, b) for i, a in enumerate(anchors) for b in anchors if b != a and apart[i, b] == span]
    views = {}
    for cid in ids:
        p = (x := compared.mol.GetConformer(cid).GetPositions()) - x.mean(0)
        frames = [frame for a, b in pairs if (frame := _frame(p[a], p[b])) is not None]
        views[cid] = _View(positions=p, axes=np.linalg.eigh(p.T @ p)[1], frames=np.reshape(frames, (-1, 3, 3)))
    return views


def _assigned(compared, p, q, rotations):
    """Return the order each start in `rotations` reaches by matching atoms within classes.

    Each round matches every class's atoms by least squared distance, then refits each start's rotation to its order; a
    start stops once its order is unchanged.
    """
    match = require("scipy.optimize", "workflow").linear_sum_assignment
    orders = np.tile(np.arange(len(p)), (len(rotations), 1))
    moving = np.arange(len(rotations))
    for _ in range(_ASSIGN_ROUNDS):
        moved, found, onto = p @ rotations[moving], orders[moving], q[moving]
        for size, same in itertools.groupby(compared.classes, len):
            atoms = np.array([*same])  # every class of one size at once
            m = moved[:, atoms]  # squared distances less each reference atom's own term, which no matching changes
            costs = (m**2).sum(-1)[..., None, :] - 2 * onto[:, atoms] @ np.swapaxes(m, -1, -2)
            if (every := _PERMUTATIONS.get(size)) is not None:  # every order at once; a tie keeps the first
                nearest = every[costs[..., np.arange(size), every].sum(-1).argmin(-1)]
            else:
                nearest = costs.argmin(-1)
                rows = np.nonzero((np.sort(nearest) != np.arange(size)).any(-1))
                nearest[rows] = np.array([match(cost)[1] for cost in costs[rows]], int).reshape(-1, size)
            found[:, atoms] = atoms[np.arange(len(atoms))[:, None], nearest]
        changed = (found != orders[moving]).any(1)
        moving = moving[changed]
        if not len(moving):
            break
        orders[moving] = found[changed]
        rotations[moving] = _rotations(p[orders[moving]], q[moving])
    return orders


def _fits(compared, probe, references):
    """Return the `_Fits` of `probe` onto each of `references`, by proper rotation, in angstrom.

    Matching starts from each reference's stored order, the 4 proper maps between principal axes, and one map per probe
    frame. An order counts only when it sends every bond onto a bond of its type, so each value is true for an
    automorphism and a merge is never wrong; a symmetric duplicate no start reaches is kept apart.
    """
    p, q = probe.positions, np.array([reference.positions for reference in references])
    maps = probe.axes @ _SIGNS @ np.swapaxes([reference.axes for reference in references], 1, 2)[:, None]
    proper = np.linalg.det(maps) > 0
    framed = np.flatnonzero([len(reference.frames) for reference in references])
    anchored = probe.frames @ np.reshape([references[r].frames[0].T for r in framed], (-1, 1, 3, 3))
    rotations = np.concatenate([_rotations(p, q), maps[proper], anchored.reshape(-1, 3, 3)])
    starts = np.concatenate([np.arange(len(references)), np.nonzero(proper)[0], np.repeat(framed, len(probe.frames))])
    owners = np.concatenate([np.arange(len(references)), starts])
    matched = _assigned(compared, p, q[starts], rotations)
    orders = np.vstack([np.tile(np.arange(len(p)), (len(references), 1)), matched])
    first, second = compared.bonds.T
    certified = (compared.adjacency[orders[:, first], orders[:, second]] == compared.adjacency[first, second]).all(1)
    return _measured(p[orders[certified]], q[owners[certified]], owners[certified], orders[certified])


def _measured(p, q, owners, orders=None):
    """Return the `_Fits` of centred rows `p` fitted onto `q` by proper rotation, onto references `owners`.

    Either may be a stack of row sets, one fit each.
    """
    deviation = np.linalg.norm(p @ _rotations(p, q) - q, axis=-1)
    if orders is None:
        orders = np.broadcast_to(np.arange(p.shape[-2]), (len(owners), p.shape[-2]))
    return _Fits(rmsd=np.sqrt((deviation**2).mean(-1)), maximum=deviation.max(-1), owners=owners, orders=orders)


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """Describe one accepted fit; order[i] is the candidate atom for reference atom i."""

    rmsd: float
    maximum: float
    order: tuple[int, ...]


def _complete_order(mol, indices, order, reference, candidate):
    """Match omitted hydrogens within their bonded groups in the accepted fit's frame."""
    match = require("scipy.optimize", "workflow").linear_sum_assignment
    mapping = dict(zip(indices, indices[order], strict=True))
    omitted = {}  # the hydrogens `RemoveHs` left out, which the fit holds alike, by their bonded atom
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in mapping:
            omitted.setdefault(atom.GetNeighbors()[0].GetIdx(), []).append(atom.GetIdx())
    if omitted:
        p, q = mol.GetConformer(candidate).GetPositions(), mol.GetConformer(reference).GetPositions()
        pc, qc = p[indices].mean(0), q[indices].mean(0)
        moved = (p - pc) @ _rotations(p[indices[order]] - pc, q[indices] - qc) + qc
        for parent, atoms in omitted.items():
            candidates = omitted.get(mapping[parent], [])
            if len(atoms) != len(candidates):
                return None
            costs = ((q[atoms, None, :] - moved[candidates]) ** 2).sum(-1)
            rows, columns = match(costs)
            mapping.update((atoms[i], candidates[j]) for i, j in zip(rows, columns, strict=True))
    return tuple(int(mapping[i]) for i in range(mol.GetNumAtoms()))


def compare(mol, reference, candidate, *, max_rmsd, hydrogens=False):
    """Return the closest fit passing both geometric gates, or None; leave coordinates unchanged."""
    compared = _compared(mol, hydrogens)
    views = _views(compared, [reference, candidate])
    fits = _fits(compared, views[candidate], [views[reference]])
    accepted = np.flatnonzero(fits.passing(max_rmsd))
    if not len(accepted):
        return None
    for best in accepted[np.argsort(fits.rmsd[accepted], kind="stable")]:
        order = _complete_order(mol, compared.indices, fits.orders[best], reference, candidate)
        if order is not None:
            return Comparison(rmsd=float(fits.rmsd[best]), maximum=float(fits.maximum[best]), order=order)
    return None


def _radii(compared, views):
    """Return each conformer's atom-centroid distances, sorted within each class: one row per view.

    No rotation or relabelling within a class changes them, so the RMSD and largest difference of two rows bound the
    aligned RMSD and maximum deviation from below.
    """
    radii = np.array([np.linalg.norm(view.positions, axis=1) for view in views])
    alone = np.setdiff1d(np.arange(radii.shape[1]), np.concatenate([[], *compared.classes]))
    return np.hstack([radii[:, alone], *(np.sort(radii[:, atoms], axis=1) for atoms in compared.classes)])


def nearest_kept(mol, kept, dropped, *, hydrogens=False):
    """Map each dropped conformer to the nearest kept one, by the rmsd row's comparison, with their RMSD in Å.

    Nearness alone: the energy window and deviation gate are not applied, so it need not be the conformer that absorbed
    it.
    """
    compared = _compared(mol, hydrogens)
    views = _views(compared, [*kept, *dropped])
    out = {}
    for d in dropped:
        fits = _fits(compared, views[d], [views[k] for k in kept])
        nearest = np.full(len(kept), np.inf)
        np.minimum.at(nearest, fits.owners, fits.rmsd)
        out[d] = (kept[int(nearest.argmin())], round(float(nearest.min()), 3))
    return out


_ENERGY_WINDOW = 12.0  # kcal/mol
_ENERGY_TOL = 0.05  # kcal/mol
_SLACK = 1e-9  # angstrom: rounding must never let the radius bound skip a pair alignment would merge


def _rmsd(*, mol, ids, energies, max_rmsd, hydrogens, **_):
    """Keep each conformer unless one of its certified fits onto a kept one in its window passes `_Fits.passing`."""
    compared, kept = _compared(mol, hydrogens), []
    views = [*_views(compared, ids).values()]
    radii, stack = _radii(compared, views), np.array([view.positions for view in views])
    for k, view in enumerate(views):
        near = np.array([j for j in kept if abs(energies[k] - energies[j]) < _ENERGY_WINDOW], int)
        gap = np.abs(radii[near] - radii[k])  # pairs whose bound already fails a gate are not aligned
        bound = np.sqrt((gap**2).mean(1))
        passing = (bound < max_rmsd + _SLACK) & (gap.max(1) < 2 * max_rmsd + _SLACK)
        possible = near[passing][np.argsort(bound[passing])]
        batches = np.split(possible, [4**i for i in range(len(possible).bit_length()) if 4**i < len(possible)])
        if not len(possible) or not (  # every stored order at once, then starts in batches growing fourfold by bound
            _measured(view.positions, stack[possible], possible).passing(max_rmsd).any()
            or any(_fits(compared, view, [views[j] for j in batch]).passing(max_rmsd).any() for batch in batches)
        ):
            kept.append(k)
    return [k in kept for k in range(len(ids))]


def _energy(*, energies, **_):
    return energy_prune(energies, energy_tol=_ENERGY_TOL)


# Each row returns the keep-mask over the energy-sorted conformers `apply` gives it.
DEDUP = {"rmsd": _rmsd, "energy": _energy}


# a missing energy is inf, and inf - inf is NaN, which no window admits, so both are kept
@np.errstate(invalid="ignore")
def apply(mol, ids, energies, method="rmsd", *, max_rmsd=0.5, hydrogens=False):
    """Deduplicate ``ids`` by a `DEDUP` method; return the kept ids, energy-sorted."""
    try:
        keep = DEDUP[method]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown dedup method {method!r} (use {' | '.join(map(repr, DEDUP))}, "
            f"or representatives() for a binding-mode summary)"
        ) from None
    ids = list(ids)
    energies = np.asarray(energies, float)
    order = np.argsort(energies)
    ids_s = [ids[k] for k in order]
    mask = keep(mol=mol, ids=ids_s, energies=energies[order], max_rmsd=max_rmsd, hydrogens=hydrogens)
    return list(itertools.compress(ids_s, mask))


def energy_prune(energies, *, energy_tol=0.05):
    """Return a keep-mask dropping each conformer within `energy_tol` of an already-kept one; no coordinate is read.

    `energies` must be sorted ascending, in the units of `energy_tol`.
    """
    kept, mask = [], []
    for e in np.asarray(energies, float):
        dup = any(abs(e - k) <= energy_tol for k in kept)
        mask.append(not dup)
        if not dup:
            kept.append(e)
    return np.asarray(mask, bool)
