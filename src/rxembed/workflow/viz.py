"""Plot an ensemble's landscape: its conformers, kept and dropped, on a 2D projection of the clustering latent.

The projection reads `select.features` fit on the kept conformers, the latent `Ensemble.cluster` reads. 3D viewing
is the caller's (`Ensemble.align` and `Ensemble.dump`, then py3Dmol or xyzrender). It needs the ``workflow`` extra.
"""

from __future__ import annotations

from rxembed.utils import require
from rxembed.workflow.select import cluster_on, features

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def project(feats, method="pca"):
    """Return 2D coordinates of the rows of `feats` by ``'pca'`` or ``'tsne'`` (at one fixed seed)."""
    if method == "pca":
        return require("sklearn.decomposition", "workflow").PCA(2).fit_transform(feats)
    if method in ("tsne", "t-sne"):
        perplexity = min(30, max(2, (len(feats) - 1) // 3))
        return (
            require("sklearn.manifold", "workflow")
            .TSNE(2, random_state=42, perplexity=perplexity, init="random")
            .fit_transform(feats)
        )
    raise ValueError(f"a landscape projects by 'pca' or 'tsne', not {method!r}")


def landscape(ens, mol, color="cluster", method="pca"):
    """Return the axes of a 2D landscape of `ens`'s conformers, coloured by binding-mode cluster or energy.

    `mol` is the ensemble's working Mol, which still carries the conformers a stage dropped (``ens.discarded``); they
    are drawn faded in the same projection, so the plot shows which regions collapsed to a representative.
    Positions and clusters come from one latent (`select.features`), fit on the kept conformers with the dropped
    placed in it, so a kept conformer's cluster is its `Ensemble.cluster` label; `method` sets only the positions. An
    ensemble with no kept conformer has no latent to draw and raises.
    """
    if color not in ("cluster", "energy"):
        raise ValueError(f"a landscape is coloured by 'cluster' or 'energy', not {color!r}")
    kept = list(ens.ids)
    if not kept:
        raise ValueError(
            "this ensemble has no kept conformers to fit a landscape on; draw it before the stage that emptied it"
        )
    plt = require("matplotlib.pyplot", "workflow")
    plt.rcParams["figure.figsize"] = (6, 5)
    dropped = [i for i in dict.fromkeys(ens.discarded) if i not in set(kept)]
    feats = features(mol, kept, dropped).matrix
    xy = project(feats, method)
    xk, xd = xy[: len(kept)], xy[len(kept) :]
    fig, ax = plt.subplots()
    if dropped:
        ax.scatter(*xd.T, s=50, color="0.75", alpha=0.45, lw=0, zorder=1, label=f"dropped ({len(dropped)})")
    if color == "energy":
        e = [ens.energies.get(i, float("nan")) for i in kept]
        sc = ax.scatter(*xk.T, c=e, cmap="viridis", s=70, edgecolor="k", lw=0.4, zorder=2)
        fig.colorbar(sc, ax=ax, label="energy (kcal/mol)")
        if dropped:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    else:
        lab = cluster_on(feats[: len(kept)])
        for k in sorted(set(lab)):
            ax.scatter(
                *xk[lab == k].T,
                s=70,
                edgecolor="k",
                lw=0.4,
                zorder=2,
                color="0.6" if k == -1 else PALETTE[k % len(PALETTE)],
                label="noise" if k == -1 else f"mode {k + 1}",
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    ax.set(xlabel=f"{method.upper()} 1", ylabel=f"{method.upper()} 2")
    ax.set_box_aspect(1)
    return ax
