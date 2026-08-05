"""Ensemble landscape: a 2D map of the conformer set for diversity / kept-vs-dropped inspection.

The only plot that earns a place in the package: the dimensionality reduction of the per-conformer
latent (dihedral + NCI + metal, shared with the clustering in `select`) is real reusable work.
3D structure viewing is notebook-level: `align()`/`dump()` give the geometry, then py3Dmol or xyzrender
directly. `pip install rxembed[viz]`.
"""

from __future__ import annotations

from .select import cluster_on, feature_matrix

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def _setup():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("_setup needs matplotlib; pip install 'rxembed[viz]'") from exc
    plt.rcParams["figure.figsize"] = (6, 5)
    return plt


def project(feats, method="pca", seed=42):
    """Return 2D coordinates for the feature matrix ``feats`` (pca | tsne)."""
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError("project needs scikit-learn; pip install 'rxembed[viz]'") from exc

        return PCA(2).fit_transform(feats)
    if method in ("tsne", "t-sne"):
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError("project needs scikit-learn; pip install 'rxembed[viz]'") from exc

        perplexity = min(30, max(2, (len(feats) - 1) // 3))
        return TSNE(2, random_state=seed, perplexity=perplexity, init="random").fit_transform(feats)
    raise ValueError(method)


def landscape(ens, color="cluster", method="pca", nci=True, reduce=None, min_cluster=3, show_dropped=True):
    """2D conformer landscape, coloured by binding-mode cluster or energy.

    Colours (clusters) and coordinates both come from one latent, `feature_matrix` (rotatable dihedrals,
    + NCI fingerprint if any contacts, + L-M-L angles if a metal). `method` only sets the positions;
    clustering uses the full latent (or `reduce=k` PCA comps). With `show_dropped` (default), conformers
    a `prune()` merged away (`ens.discarded`) are drawn as small faded points *in the same projection*, so
    you see the whole ensemble and which regions were collapsed to a representative.
    """
    plt = _setup()
    kept = list(ens.ids)
    dropped = [i for i in dict.fromkeys(ens.discarded) if i not in set(kept)] if show_dropped else []
    feats = feature_matrix(ens.mol, kept + dropped, nci)  # project kept + dropped together (one space)
    xy = project(feats, method)
    xk, xd = xy[: len(kept)], xy[len(kept) :]
    fig, ax = plt.subplots()
    if dropped:  # the whole ensemble: pruned-away conformers as faint, small points behind the kept
        ax.scatter(*xd.T, s=50, color="0.75", alpha=0.45, lw=0, zorder=1, label=f"dropped ({len(dropped)})")
    if color == "energy":
        e = [ens.energies.get(i, float("nan")) for i in kept]
        sc = ax.scatter(*xk.T, c=e, cmap="viridis", s=70, edgecolor="k", lw=0.4, zorder=2)
        fig.colorbar(sc, ax=ax, label="energy (kcal/mol)")
        if dropped:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    elif color == "cluster":
        lab = cluster_on(feats[: len(kept)], min_cluster=min_cluster, reduce=reduce)  # cluster the kept only
        for k in sorted(set(lab)):
            ax.scatter(
                *xk[lab == k].T,
                s=70,
                edgecolor="k",
                lw=0.4,
                zorder=2,
                color="0.6" if k == -1 else PALETTE[k % len(PALETTE)],
                label="noise" if k == -1 else f"mode {k + 1}",  # 1-indexed, human-readable
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    else:
        ax.scatter(*xk.T, s=70, color=PALETTE[0], edgecolor="k", lw=0.4, zorder=2)
    ax.set(xlabel=f"{method.upper()} 1", ylabel=f"{method.upper()} 2")
    ax.set_box_aspect(1)
    return ax
