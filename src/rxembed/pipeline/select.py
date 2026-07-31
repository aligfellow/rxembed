"""Dedup and clustering over an existing ensemble: descriptors, the latent, and the prunes.

Three concerns, one module: per-conformer geometric descriptors (rotatable-bond dihedrals), the single
per-conformer latent used for both clustering and the landscape (dihedrals + NCI fingerprint + L-M-L
angles, block-balanced), and de-duplication behind ``apply()`` (moi | rmsd | descriptor | energy).

moi/rmsd/descriptor ride prism_pruner (energy-gated); ``energy`` is a prism-free near-degeneracy prune.
All pre-sort by energy so masks stay aligned. Binding-mode *summarising* (one conformer per pose) is a
separate concern; see ``cluster_labels`` / ``Ensemble.representatives``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rxembed.metal_core import _frag_map, metal_index
from rxembed.utils import _angle

_ROT = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")


def _ref(mol, a, b):
    """Pick a reference neighbour of ``a`` (not ``b``) for the dihedral, preferring a heavy atom."""
    nbrs = [x.GetIdx() for x in mol.GetAtomWithIdx(a).GetNeighbors() if x.GetIdx() != b]
    heavy = [i for i in nbrs if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    return (heavy or nbrs or [None])[0]


def rotatable_quads(mol):
    """Heavy-atom dihedral 4-tuples (i, a, b, j), one per rotatable bond, stable across conformers."""
    quads = []
    for a, b in mol.GetSubstructMatches(_ROT):
        i, j = _ref(mol, a, b), _ref(mol, b, a)
        if i is not None and j is not None:
            quads.append((i, a, b, j))
    return quads


def dihedrals(mol, conf_id, quads=None):
    """(cos, sin) of each rotatable-bond dihedral: a frame-invariant feature vector."""
    quads = rotatable_quads(mol) if quads is None else quads
    conf = mol.GetConformer(conf_id)
    v = []
    for q in quads:
        t = np.radians(rdMolTransforms.GetDihedralDeg(conf, *q))
        v += [np.cos(t), np.sin(t)]
    return np.array(v) if v else np.zeros(1)


# The single per-conformer latent, behind both clustering and the landscape: one representation, so a plot
# and a dedup never disagree about which conformers are alike.
_M_DONOR_CUT = 2.8  # Angstrom: a heavy atom within this of the metal counts as a donor
_MIN_FRAGS = 2  # a binding-mode block needs at least two fragments
_MIN_DONORS = 2  # two donors are needed to define an L-M-L angle
_EPS = 1e-9  # std floor for z-scoring


def _norm(m):
    return m / np.sqrt(m.var(0).sum()) if m.size and m.var(0).sum() else m


def _metal_donors(mol, ids):
    """``(metal, heavy atoms inside `_M_DONOR_CUT` of it)``, or ``(None, None)`` when there is no metal."""
    m = metal_index(mol)
    if m is None:
        return None, None
    pos0 = mol.GetConformer(ids[0]).GetPositions()
    donors = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIdx() != m and a.GetAtomicNum() > 1 and np.linalg.norm(pos0[a.GetIdx()] - pos0[m]) < _M_DONOR_CUT
    ]
    return m, donors


def _metal_features(mol, ids):
    """L-M-L angles per conformer, capturing the coordination polyhedron or isomer. None if no metal."""
    m, donors = _metal_donors(mol, ids)
    pairs = list(itertools.combinations(donors or (), 2))
    if not pairs:
        return None
    out = []
    for i in ids:
        pos = mol.GetConformer(i).GetPositions()
        out.append([_angle(pos[a], pos[m], pos[b]) for a, b in pairs])
    return np.array(out)


def _metal_present(mol):
    return metal_index(mol) is not None


def _interfragment_contacts(an, positions, fmap):
    """NCI contacts whose two sites span different fragments → list of (type, frozenset(atoms), pair).

    `pair` is the ``(lo, hi)`` fragment-index pair the contact bridges, so an H-bond to a *substrate*
    and an H-bond to *solvent* are distinguishable. Binding-mode semantics are *inter-molecular*:
    intramolecular NCIs (a molecule's own H-bond) are conformational detail already carried by the
    dihedral block and are excluded; otherwise a lone organic would be mislabelled a
    'contact pattern'.
    """
    out = []
    for n in an.detect(positions):
        atoms = set(n.site_a) | set(n.site_b)
        frags = {fmap[a] for a in atoms}
        if len(frags) > 1:  # spans ≥2 fragments
            out.append((n.type, frozenset(atoms), tuple(sorted(frags))))
    return out


def _nci_features(mol, ids):
    """Binary inter-fragment NCI-contact fingerprint per conformer, or None when there is none to describe.

    None for a single fragment, or a metal complex (the metal block defines the mode; the surrogate strips
    coordinate bonds, so without this guard every ligand looks like a separate fragment and
    coordination/inter-ligand contacts would pollute the latent).
    """
    if _metal_present(mol):
        return None
    from . import nci as nci_mod

    fmap = _frag_map(mol)
    if len(set(fmap.values())) < _MIN_FRAGS:  # single molecule -> no binding-mode block
        return None
    an = nci_mod.analyzer(mol)
    sigs = [{(t, a) for t, a, _ in _interfragment_contacts(an, mol.GetConformer(i).GetPositions(), fmap)} for i in ids]
    universe = sorted({k for s in sigs for k in s})
    if not universe:
        return None
    return np.array([[float(k in s) for k in universe] for s in sigs])


def _relpose_features(mol, ids):
    """Relative-pose block: where each fragment sits relative to the anchor (largest) fragment.

    The inter-fragment rigid-body DOF the dihedral block cannot see, and the reason a benzene...water
    ensemble's distinct encounter geometries otherwise collapse to one mode. Each non-anchor fragment
    contributes its centroid-anchor-centroid distance, the sorted distances from the anchor's heavy atoms to
    that centroid (where it sits *around* the anchor), and the sorted distances from its own heavy atoms to
    the anchor centroid (how it is *oriented*).

    Distance-only, so it is frame-free and respects the anchor's symmetry; identical non-anchor fragments are
    ordered canonically, so relabelling them cannot split one pose in two. Columns are z-scored so a mobile
    fragment does not swamp a quiet one. Limitation: pure distances cannot tell the two faces of an
    *asymmetric planar* anchor apart, which the NCI fingerprint resolves once a contact forms.

    None for a single fragment, or a metal, whose relative pose IS the L-M-L block.
    """
    if _metal_present(mol):
        return None
    frags = Chem.GetMolFrags(mol)
    if len(frags) < _MIN_FRAGS:
        return None

    def heavy(f):
        return [a for a in f if mol.GetAtomWithIdx(a).GetAtomicNum() > 1]

    hf = [heavy(f) or list(f) for f in frags]
    anchor = max(range(len(frags)), key=lambda i: (len(hf[i]), -min(frags[i])))  # largest, ties->low idx
    a_atoms = hf[anchor]
    smi = [Chem.MolToSmiles(m) for m in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)]
    others = [i for i in range(len(frags)) if i != anchor]
    groups = [[o for o in others if smi[o] == s] for s in sorted({smi[o] for o in others})]  # fixed order

    def sub(pos, ac, o):  # one non-anchor fragment's relative-pose descriptor
        oc = pos[hf[o]].mean(0)
        return [
            float(np.linalg.norm(oc - ac)),
            *sorted(float(np.linalg.norm(pos[a] - oc)) for a in a_atoms),  # around the anchor
            *sorted(float(np.linalg.norm(pos[h] - ac)) for h in hf[o]),  # its orientation
        ]

    rows = []
    for cid in ids:
        pos = mol.GetConformer(cid).GetPositions()
        ac = pos[a_atoms].mean(0)
        feat = []
        for grp in groups:
            feat += [x for s in sorted(sub(pos, ac, o) for o in grp) for x in s]  # canonical within group
        rows.append(feat)
    mat = np.array(rows, float)
    sd = mat.std(0)
    sd[sd < _EPS] = 1.0
    return (mat - mat.mean(0)) / sd  # z-score so no single column dominates


def _blocks(mol, ids, nci=True):
    """Build the latent as a list of (name, array) blocks, in concatenation order.

    Always ``'dihedral'`` (rotatable-bond sin/cos); ``'metal'`` (L-M-L angles) iff a transition metal
    with detectable donors is present; ``'relpose'`` (inter-fragment relative pose) iff a multi-fragment
    non-metal complex, so distinct encounter geometries don't collapse; ``'nci'`` (binary inter-fragment
    contact fingerprint) iff ``nci`` and any inter-fragment contact is detected. Used by both
    ``feature_matrix`` (the numbers) and ``active_feature_kinds`` (which kinds are live).
    """
    quads = rotatable_quads(mol)
    blocks = [("dihedral", np.array([dihedrals(mol, i, quads) for i in ids]))]
    fm = _metal_features(mol, ids)
    if fm is not None:
        blocks.append(("metal", fm))
    rp = _relpose_features(mol, ids)
    if rp is not None:
        blocks.append(("relpose", rp))
    if nci:
        fn = _nci_features(mol, ids)  # already None when a metal is present
        if fn is not None:
            blocks.append(("nci", fn))
    return blocks


def feature_matrix(mol, ids, nci=True):
    """Per-conformer latent: dihedrals (+ NCI fingerprint) (+ metal L-M-L angles), block-balanced."""
    blocks = _blocks(mol, ids, nci)
    return np.hstack([_norm(b) for _, b in blocks]) if len(blocks) > 1 else blocks[0][1]


# what each latent block means when it is the most specific one present: the "mode" kind
_MODE_KIND = {
    "metal": "ligand arrangement",
    "nci": "contact pattern",
    "relpose": "relative arrangement",
    "dihedral": "conformer family",
}


def active_feature_kinds(mol, ids, nci=True):
    """Names of the latent blocks live for this ensemble, e.g. ``['dihedral', 'relpose', 'nci']``.

    The honest answer to "what is clustering/representatives actually separating here?".

    Cheap: detects block *presence* (a metal with donors; multi-fragment → relative pose; any inter-
    fragment contact across the ensemble, short-circuiting on the first hit) without materialising the
    full per-conformer matrix, but agrees with what ``feature_matrix`` concatenates.
    """
    blocks = ["dihedral"]
    if _metal_present(mol):
        _, donors = _metal_donors(mol, ids)
        if donors and len(donors) >= _MIN_DONORS:
            blocks.append("metal")
        return blocks  # a metal suppresses relpose/nci (its mode = L-M-L)
    if len(set(_frag_map(mol).values())) >= _MIN_FRAGS:
        blocks.append("relpose")  # multi-fragment -> relative-pose block always present
        if nci:
            from . import nci as nci_mod

            an, fmap = nci_mod.analyzer(mol), _frag_map(mol)
            if any(_interfragment_contacts(an, mol.GetConformer(i).GetPositions(), fmap) for i in ids):
                blocks.append("nci")
    return blocks


def mode_kind(mol, ids, nci=True):
    """Human label for what a 'mode' means for THIS system (ligand arrangement / contact pattern / ...).

    'ligand arrangement' (metal), else 'contact pattern' (NCI), else 'relative arrangement' (multi-fragment,
    no detected contact), else 'conformer family' (a single flexible molecule).
    """
    blocks = active_feature_kinds(mol, ids, nci)
    for b in ("metal", "nci", "relpose"):  # most specific block wins
        if b in blocks:
            return _MODE_KIND[b]
    return _MODE_KIND["dihedral"]


def mode_signature(mol, ids, nci=True):
    """Per-conformer discrete binding-mode signature (hashable), or ``None`` when there is no such block.

    ``None`` for a plain organic (then 'modes' are purely torsional families and noise is just sampling
    scatter). Combines the NCI contact-type set and the metal coordination label, so two conformers with
    the same signature are the same *binding* mode even if their torsions differ. Used by
    ``Ensemble.representatives`` to recover a genuinely rare binding mode HDBSCAN flagged as noise,
    without letting torsional scatter inflate the representative set.
    """
    blocks = active_feature_kinds(mol, ids, nci)
    if "nci" not in blocks and "metal" not in blocks:
        return None
    sigs = [[] for _ in ids]
    if "nci" in blocks:
        from . import nci as nci_mod

        an = nci_mod.analyzer(mol)
        fmap = _frag_map(mol)
        for k, i in enumerate(ids):  # which (contact type, partner-fragment-pair)s
            contacts = _interfragment_contacts(an, mol.GetConformer(i).GetPositions(), fmap)
            sigs[k].append(("nci", frozenset((t, p) for t, _a, p in contacts)))
    if "metal" in blocks:
        from rxembed.metal_core import label as metal_label

        m, donors = _metal_donors(mol, ids)
        if donors:
            for k, i in enumerate(ids):
                sigs[k].append(("metal", metal_label(mol, m, donors, i)))
    return [tuple(s) for s in sigs]


def cluster_on(feats, *, min_cluster=3, reduce=None):
    """Cluster a feature latent with HDBSCAN (-1 = rare/noise).

    `reduce=k` PCA-compresses to k components first (denoise); by default clusters on the *full* latent
    (every column), not 2 PCA axes.
    """
    if len(feats) < max(_MIN_FRAGS, min_cluster):  # too few conformers to cluster: treat as one mode
        return np.zeros(len(feats), dtype=int)
    if reduce and reduce < feats.shape[1]:
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError("cluster_on needs scikit-learn; pip install 'rxembed[select]'") from exc

        feats = PCA(reduce).fit_transform(feats)
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError("cluster_on needs scikit-learn; pip install 'rxembed[select]'") from exc

    return np.asarray(
        HDBSCAN(min_cluster_size=min_cluster, cluster_selection_method="leaf", copy=True).fit_predict(feats)
    )  # copy=True: future sklearn default, silences warning


def cluster_labels(mol, ids, *, min_cluster=3, reduce=None, nci=True):
    """Binding-mode label per conformer on the unified latent (-1 = rare/noise)."""
    return cluster_on(feature_matrix(mol, ids, nci), min_cluster=min_cluster, reduce=reduce)


def nearest_kept(mol, kept, dropped):
    """Map each dropped conformer to the kept one it most resembles, with heavy-atom RMSD (the prune 'why').

    Kabsch-aligned RMSD in Angstrom: conf D was dropped as a near-duplicate of K.
    """
    try:
        from prism_pruner.rmsd import rmsd_and_max
    except ImportError as exc:
        raise ImportError("nearest_kept needs prism_pruner; pip install 'rxembed[select]'") from exc

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    pos = {i: mol.GetConformer(i).GetPositions()[heavy] for i in list(kept) + list(dropped)}
    out = {}
    for d in dropped:
        dists = {k: float(rmsd_and_max(pos[d], pos[k])[0]) for k in kept}
        best = min(dists, key=dists.__getitem__)
        out[d] = (best, round(dists[best], 3))
    return out


def apply(
    mol, ids, energies, method="rmsd", *, energy_window=12.0, max_dist=0.75, moi_dev=0.01, max_rmsd=0.5, energy_tol=0.05
):
    """Deduplicate ``ids`` by ``method`` (moi | rmsd | descriptor | energy | none) -> (kept_ids, {})."""
    ids = list(ids)
    energies = np.asarray(energies, float)
    order = np.argsort(energies)
    ids_s = [ids[k] for k in order]
    en = energies[order]
    coords = np.array([mol.GetConformer(i).GetPositions() for i in ids_s])
    atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])

    def keep_mask(mask):
        return [ids_s[k] for k in range(len(mask)) if mask[k]], {}

    if method == "none":
        return ids_s, {}
    if method == "moi":
        try:
            from prism_pruner.pruner import prune_by_moment_of_inertia as prune_by_moi
        except ImportError as exc:
            raise ImportError("apply needs prism_pruner; pip install 'rxembed[select]'") from exc

        return keep_mask(prune_by_moi(coords, atoms, moi_dev, en, energy_window)[1])
    if method == "rmsd":
        try:
            from prism_pruner.pruner import prune_by_rmsd
        except ImportError as exc:
            raise ImportError("apply needs prism_pruner; pip install 'rxembed[select]'") from exc

        return keep_mask(prune_by_rmsd(coords, atoms, max_rmsd, None, en, energy_window)[1])
    if method == "descriptor":
        quads = rotatable_quads(mol)
        feats = np.array([dihedrals(mol, i, quads) for i in ids_s])
        return keep_mask(descriptor_prune(coords, feats, en, max_dist=max_dist, energy_window=energy_window))
    if method == "energy":
        return keep_mask(energy_prune(en, energy_tol=energy_tol))
    raise ValueError(
        f"unknown dedup method {method!r} (use 'moi' | 'rmsd' | 'descriptor' | 'energy', "
        f"or representatives() for a binding-mode summary)"
    )


def energy_prune(energies, *, labels=None, energy_tol=0.05):
    """Energy-only de-dup: drop a frame within `energy_tol` (energy units) of an already-kept one.

    Prism-free by design: near-degenerate energies are treated as the same minimum, no coordinates
    consulted. `energies` must be ascending-sorted; optional `labels` gates the comparison to the same
    discrete binding mode. Returns a keep-mask aligned to input order.
    """
    energies = np.asarray(energies, float)
    kept: dict = {}
    mask = []
    for i, e in enumerate(energies):
        prev = kept.setdefault(None if labels is None else labels[i], [])
        dup = any(abs(e - k) <= energy_tol for k in prev)
        mask.append(not dup)
        if not dup:
            prev.append(e)
    return np.asarray(mask, bool)


@cache
def _descriptor_config():
    """Prism config with a custom similarity: a frame-invariant descriptor gated by a discrete label.

    Built on demand because its base class is prism's, which the `select` extra provides; a module-level
    subclass would make importing this module require the extra.
    """
    try:
        from prism_pruner.pruner import PrunerConfig
    except ImportError as exc:
        raise ImportError("_descriptor_config needs prism_pruner; pip install 'rxembed[select]'") from exc

    @dataclass
    class _DescriptorConfig(PrunerConfig):
        features: np.ndarray = field(kw_only=True)
        labels: list | None = field(kw_only=True, default=None)
        max_dist: float = field(kw_only=True, default=1.0)

        def evaluate_sim(self, i, j):
            if self.labels is not None and self.labels[i] != self.labels[j]:  # same binding mode only
                return False
            return float(np.linalg.norm(self.features[i] - self.features[j])) < self.max_dist

    return _DescriptorConfig


def descriptor_prune(coords, features, energies, *, labels=None, max_dist=1.0, energy_window=12.0):
    """Descriptor de-dup on prism's energy-sorted engine; keep-mask aligned to input order (energy-sorted).

    WART: ``_run`` is a *private* prism entry point (no public API for a custom ``evaluate_sim``; prism's
    own ``prune_by_rmsd`` also calls it). Accepted rather than reinventing prism's engine.
    """
    f = np.asarray(features, float)
    if len(f) > 1:
        s = f.std(0)
        s[s == 0] = 1.0
        f = (f - f.mean(0)) / s
    cfg = _descriptor_config()(
        structures=np.asarray(coords, float),
        energies=np.asarray(energies, float),
        max_dE=float(energy_window),
        features=f,
        labels=labels,
        max_dist=float(max_dist),
        logfunction=None,
    )
    try:
        from prism_pruner.pruner import _run
    except ImportError as exc:
        raise ImportError("descriptor_prune needs prism_pruner; pip install 'rxembed[select]'") from exc

    return np.asarray(_run(cfg)[1], bool)


prune = apply  # the public name for the dedup step (descriptor_prune / energy_prune keep their own names)
