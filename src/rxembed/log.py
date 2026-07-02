"""Package logger. Each pipeline stage logs INFO (counts) and DEBUG (detail).

import rxembed
rxembed.set_verbose()           # DEBUG: see every stage and constraint
rxembed.set_verbose("INFO")     # default-ish: one line per stage
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rxembed")
logger.addHandler(logging.NullHandler())


def set_verbose(level: int | str = "INFO") -> None:
    """Turn on console logging at `level` (e.g. 'INFO', 'DEBUG', logging.DEBUG)."""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(name)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
