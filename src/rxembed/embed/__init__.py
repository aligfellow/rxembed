"""Generation stage: mass-embed (bounds) and/or MC search (openconf)."""

from . import mc
from .bounds import embed, n_confs

__all__ = ["embed", "mc", "n_confs"]
