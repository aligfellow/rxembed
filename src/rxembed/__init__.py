"""rxembed. Fast, flexible molecular embedding for reactive chemistry."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .constraints import Constraints, compose, match, resolve_core
from .embed import Conformers, embed, minimize
from .metal_core import Ligand, dative_smiles, ligands
from .metal_isomers import Isomer, IsomerSet, enumerate_isomers
from .relax import ff_energies, restrained_uff

logger = logging.getLogger("rxembed")


def set_verbose(level: int | str = "INFO") -> None:
    """Turn on console logging at `level` (e.g. 'INFO', 'DEBUG', logging.DEBUG)."""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(name)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)


__all__ = [
    "Conformers",
    "Constraints",
    "Isomer",
    "IsomerSet",
    "Ligand",
    "compose",
    "dative_smiles",
    "embed",
    "enumerate_isomers",
    "ff_energies",
    "ligands",
    "match",
    "minimize",
    "resolve_core",
    "restrained_uff",
    "set_verbose",
]

try:
    __version__ = version("rxembed")
except PackageNotFoundError:
    __version__ = "0+unknown"
