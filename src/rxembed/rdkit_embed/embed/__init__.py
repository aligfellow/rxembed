"""Generation stage kernel: bounds-matrix ETKDG embedding (rdkit + numpy).

Only the pure embedder lives in the kernel; the openconf Monte-Carlo search stays in the rxembed shell.
"""

from .bounds import embed, n_confs

__all__ = ["embed", "n_confs"]
