"""Path anchors shared by the test suite, resolved once so tests run from any cwd."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "structures"

# The tmQMg corpus is a local-only clone; override its location without editing tests.
TMQMG_DIR = Path(os.environ.get("RXEMBED_TMQMG_DIR", "/home/ali/Documents/Codes/tmQMg/data"))
