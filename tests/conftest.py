"""Path anchors shared by the test suite, resolved once so tests run from any cwd."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "structures"
