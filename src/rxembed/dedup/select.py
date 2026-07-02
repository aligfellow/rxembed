"""De-duplication over an existing ensemble: moi | rmsd | descriptor | energy, behind ``apply()``.

moi/rmsd/descriptor ride prism_pruner (energy-gated); ``energy`` is a prism-free near-degeneracy prune.
All pre-sort by energy so masks stay aligned. Binding-mode *summarising* (one conformer per pose) is a
separate concern — see ``features.cluster_labels`` / ``Ensemble.representatives``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from prism_pruner.pruner import PrunerConfig, _run, prune_by_moment_of_inertia, prune_by_rmsd

from .descriptors import dihedrals, rotatable_quads


def nearest_kept(mol, kept, dropped):
    """Map each dropped conformer to the kept one it most resembles, with heavy-atom RMSD (the prune 'why').

    Kabsch-aligned RMSD in Angstrom: conf D was dropped as a near-duplicate of K.
    """
    from prism_pruner.rmsd import rmsd_and_max

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
        return keep_mask(prune_by_moment_of_inertia(coords, atoms, moi_dev, en, energy_window)[1])
    if method == "rmsd":
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

    Prism-free by design — near-degenerate energies are treated as the same minimum, no coordinates
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


@dataclass
class _DescriptorConfig(PrunerConfig):
    """prism config with a custom similarity: a frame-invariant descriptor gated by a discrete label."""

    features: np.ndarray = field(kw_only=True)
    labels: list | None = field(kw_only=True, default=None)
    max_dist: float = field(kw_only=True, default=1.0)

    def evaluate_sim(self, i, j):
        if self.labels is not None and self.labels[i] != self.labels[j]:  # same binding mode only
            return False
        return float(np.linalg.norm(self.features[i] - self.features[j])) < self.max_dist


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
    cfg = _DescriptorConfig(
        structures=np.asarray(coords, float),
        energies=np.asarray(energies, float),
        max_dE=float(energy_window),
        features=f,
        labels=labels,
        max_dist=float(max_dist),
        logfunction=None,
    )
    return np.asarray(_run(cfg)[1], bool)
