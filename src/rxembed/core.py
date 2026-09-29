"""Graph-owned constrained conformer embedding."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .bounds import EmbedParams
from .constraints import Constraints, compose, match, resolve_core
from .embed import Conformers, EmbeddingError, embed, minimize
from .metal_core import Ligand, ligands
from .metal_enumeration import enumerate_isomers
from .metal_isomer import Isomer, IsomerSet
from .metal_smiles import cxsmiles, dative_smiles, parse_smiles
from .relax import UFFRecord, ff_energies, restrained_uff

logger = logging.getLogger("rxembed")


def set_verbose(level: int | str = "INFO") -> None:
    """Turn on console logging at `level`."""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


try:
    __version__ = version("rxembed")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = [
    "Conformers",
    "Constraints",
    "EmbedParams",
    "EmbeddingError",
    "Isomer",
    "IsomerSet",
    "Ligand",
    "UFFRecord",
    "__version__",
    "compose",
    "cxsmiles",
    "dative_smiles",
    "embed",
    "enumerate_isomers",
    "ff_energies",
    "ligands",
    "match",
    "minimize",
    "parse_smiles",
    "resolve_core",
    "restrained_uff",
    "set_verbose",
]
