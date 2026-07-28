"""Generation stage: mass-embed (bounds) and/or MC search (openconf).

The bounds-matrix embedder lives in the `rxembed.rdkit_embed` kernel subpackage; the MC search (openconf)
stays here. The ``embed`` / ``n_confs`` re-exports keep ``from rxembed.embed import embed`` working for the
shell.
"""

from rxembed.rdkit_embed.embed.bounds import embed, n_confs

from . import mc

__all__ = ["embed", "mc", "n_confs"]
