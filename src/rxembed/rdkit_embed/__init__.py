"""The rxembed embedding kernel — the pure engine, imported as ``from rxembed.rdkit_embed import ...``.

Depends on rdkit + numpy only (scipy the one soft extra, for the sphere solver; xyzgraph perception left with
the source readers, now the shell leaf ``rxembed.inputs``). It lives inside ``rxembed`` (the shell) as the
``rxembed.rdkit_embed`` subpackage for now; the placeholder ``pyproject.toml`` marks the intent to graduate
it to a standalone ``rdkit_embed`` package once stable. The whole constrained-embedding engine lives here:
the drop-in surface below (``embed`` the bounds-matrix embedder, ``restrained_uff`` the FF relax,
``Constraints`` / ``compose`` the one-struct-in/one-struct-out model, ``resolve_core`` the
fix/constrain/template builder).
"""

from . import io, log
from .constraints.base import Constraints, compose
from .constraints.builders import resolve_core
from .embed.bounds import embed, n_confs
from .log import set_verbose
from .refine.ff import ff_energies, restrained_uff

__all__ = [
    "Constraints",
    "compose",
    "embed",
    "ff_energies",
    "io",
    "log",
    "n_confs",
    "resolve_core",
    "restrained_uff",
    "set_verbose",
]
