"""Package logger. Each pipeline stage logs INFO (counts) and DEBUG (detail).

The logger is named ``"rxembed"`` (the shell configures it via ``rxembed.set_verbose()``); the kernel logs
under the same tree so a consumer sees one stream::

    import rxembed

    rxembed.set_verbose()  # DEBUG: see every stage and constraint
    rxembed.set_verbose("INFO")  # default-ish: one line per stage
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
