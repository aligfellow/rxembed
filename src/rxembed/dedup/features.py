"""The single per-conformer latent, used for BOTH clustering and visualisation.

One representation everywhere: rotatable-bond dihedrals (sin/cos), plus a binary
NCI-contact fingerprint when contacts exist, plus the L-M-L coordination angles
when a transition metal is present. Blocks are variance-normalised so none
dominates. The binding-mode clustering (`dedup` clustering, `Ensemble.cluster`) and the
PCA/t-SNE landscape both project *this* matrix, so cluster colours and 2D
positions are always coherent. Optional blocks appear only when relevant — an
organic gets dihedrals only; an H-bonded complex adds NCI; a metal adds angles.
"""

from __future__ import annotations

import itertools

import numpy as np

from .descriptors import dihedrals, rotatable_quads

_M_DONOR_CUT = 2.8  # Angstrom: a heavy atom within this of the metal counts as a donor
_MIN_FRAGS = 2  # a binding-mode block needs at least two fragments
_MIN_DONORS = 2  # two donors are needed to define an L-M-L angle
_EPS = 1e-9  # std floor for z-scoring


def _norm(m):
    return m / np.sqrt(m.var(0).sum()) if m.size and m.var(0).sum() else m


def _first_metal(mol):
    """Index of the first transition-metal atom, or None."""
    from rxembed.rdkit_embed.constraints.metal import TRANSITION_METALS

    return next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS), None)


def _metal_features(mol, ids):
    """L-M-L angles per conformer — captures the coordination polyhedron/isomer. None if no metal."""
    m = _first_metal(mol)
    if m is None:
        return None
    pos0 = mol.GetConformer(ids[0]).GetPositions()
    donors = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIdx() != m and a.GetAtomicNum() > 1 and np.linalg.norm(pos0[a.GetIdx()] - pos0[m]) < _M_DONOR_CUT
    ]
    pairs = list(itertools.combinations(donors, 2))
    if not pairs:
        return None

    def lml(pos, a, b):  # the L-M-L angle in degrees
        va, vb = pos[a] - pos[m], pos[b] - pos[m]
        return np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)))

    out = []
    for i in ids:
        pos = mol.GetConformer(i).GetPositions()
        out.append([lml(pos, a, b) for a, b in pairs])
    return np.array(out)


def _frag_map(mol):
    from rdkit import Chem

    return {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}


def _metal_present(mol):
    return _first_metal(mol) is not None


def _interfragment_contacts(an, positions, fmap):
    """NCI contacts whose two sites span different fragments → list of (type, frozenset(atoms), pair).

    `pair` is the ``(lo, hi)`` fragment-index pair the contact bridges, so an H-bond to a *substrate*
    and an H-bond to *solvent* are distinguishable. Binding-mode semantics are *inter-molecular*:
    intramolecular NCIs (a molecule's own H-bond) are conformational detail already carried by the
    dihedral block and are excluded — otherwise a lone organic would be mislabelled a
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

    None for a single fragment, OR a metal complex (the metal block defines the mode — the surrogate strips
    coordinate bonds, so without this guard every ligand looks like a separate fragment and
    coordination/inter-ligand contacts would pollute the latent).
    """
    if _metal_present(mol):
        return None
    from rxembed.constraints import nci as nci_mod

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
    ensemble's distinct encounter geometries otherwise collapse to one 'mode'. Each non-anchor fragment
    contributes: its centroid-anchor-centroid distance, the sorted
    distances from the anchor's heavy atoms to that centroid (where it sits *around* the anchor), and
    the sorted distances from its own heavy atoms to the anchor centroid (how it is *oriented*).

    Distance-only, so it is **frame-free** (rotation/translation invariant) and respects the anchor's
    symmetry. Identical non-anchor fragments (two waters, two of one ligand) are ordered canonically, so
    swapping their labels gives the *same* vector — physically equal poses don't split. Columns are
    z-scored so one mobile fragment / the unbounded translational distance doesn't swamp a quiet one.
    Limitation: pure distances cannot tell the two faces of an *asymmetric planar* anchor apart (a
    reflection through its plane) — the NCI fingerprint resolves that when a contact forms.

    None for a single fragment, or a metal (a metal's relative pose IS the L-M-L block).
    """
    from rdkit import Chem

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
    non-metal complex — so distinct encounter geometries don't collapse; ``'nci'`` (binary inter-fragment
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


# what each latent block means when it is the most specific one present — the "mode" kind
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
            from rxembed.constraints import nci as nci_mod

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


def _metal_donors(mol, ids):
    m = _first_metal(mol)
    if m is None:
        return None, None
    pos0 = mol.GetConformer(ids[0]).GetPositions()

    donors = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIdx() != m and a.GetAtomicNum() > 1 and np.linalg.norm(pos0[a.GetIdx()] - pos0[m]) < _M_DONOR_CUT
    ]
    return m, donors


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
        from rxembed.constraints import nci as nci_mod

        an = nci_mod.analyzer(mol)
        fmap = _frag_map(mol)
        for k, i in enumerate(ids):  # which (contact type, partner-fragment-pair)s
            contacts = _interfragment_contacts(an, mol.GetConformer(i).GetPositions(), fmap)
            sigs[k].append(("nci", frozenset((t, p) for t, _a, p in contacts)))
    if "metal" in blocks:
        from rxembed.rdkit_embed.constraints.metal import label as metal_label

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
    from sklearn.cluster import HDBSCAN

    if len(feats) < max(_MIN_FRAGS, min_cluster):  # too few conformers to cluster: treat as one mode
        return np.zeros(len(feats), dtype=int)
    if reduce and reduce < feats.shape[1]:
        from sklearn.decomposition import PCA

        feats = PCA(reduce).fit_transform(feats)
    return np.asarray(
        HDBSCAN(min_cluster_size=min_cluster, cluster_selection_method="leaf", copy=True).fit_predict(feats)
    )  # copy=True: future sklearn default, silences warning


def cluster_labels(mol, ids, *, min_cluster=3, reduce=None, nci=True):
    """Binding-mode label per conformer on the unified latent (-1 = rare/noise)."""
    return cluster_on(feature_matrix(mol, ids, nci), min_cluster=min_cluster, reduce=reduce)
