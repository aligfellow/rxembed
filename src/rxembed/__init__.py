"""Fast, flexible molecular embedding for reactive chemistry."""

from importlib.metadata import PackageNotFoundError, version

from . import geometry
from .constraints.metal import enumerate_isomers as metal
from .constraints.nci import Contact
from .constraints.nci import auto_binding_modes as nci_modes
from .constraints.nci import candidate_contacts as nci_candidates
from .log import set_verbose
from .pipeline import Ensemble, EnsembleSet, embed, minimize, wrap

__all__ = [
    "Contact",
    "Ensemble",
    "EnsembleSet",
    "embed",
    "geometry",
    "metal",
    "minimize",
    "nci_candidates",
    "nci_modes",
    "set_verbose",
    "wrap",
]

try:
    __version__ = version("rxembed")
except PackageNotFoundError:  # not installed (e.g. a source checkout without an install)
    __version__ = "0+unknown"
