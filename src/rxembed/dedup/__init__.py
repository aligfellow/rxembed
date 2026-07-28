"""Dedup stage: prism (moi/rmsd/descriptor) or energy-only or binding-mode clustering, behind apply()."""

from .features import active_feature_kinds, cluster_labels, cluster_on, feature_matrix, mode_kind, mode_signature
from .select import apply, energy_prune, nearest_kept

prune = apply  # the public name for the dedup step (descriptor_prune / energy_prune keep their own names)

__all__ = [
    "active_feature_kinds",
    "apply",
    "cluster_labels",
    "cluster_on",
    "energy_prune",
    "feature_matrix",
    "mode_kind",
    "mode_signature",
    "nearest_kept",
    "prune",
]
