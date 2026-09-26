"""Test package exports and logging configuration."""

import subprocess
import sys
from pathlib import Path

import rxembed

# A real degradation on a base install: two fix distances over three atoms leave the angle free, and the
# library says so.
_DEGRADES = (
    "from rdkit import Chem\n"
    "from rxembed import core as rx\n"
    "{prelude}"
    "rx.embed(Chem.AddHs(Chem.MolFromSmiles('OCCCN')), fix={{(0, 1): 1.45, (1, 2): 1.55}}, n=2, seed=1)\n"
)


def _stderr_of(prelude=""):
    run = subprocess.run(
        [sys.executable, "-c", _DEGRADES.format(prelude=prelude)], capture_output=True, text=True, check=True
    )
    return run.stderr


def test_editable_install_loads_this_checkout():
    """An editable install pointing at a deleted scratch worktree runs stale code silently."""
    this_checkout_src = Path(__file__).resolve().parent.parent / "src" / "rxembed"
    loaded_from = Path(rxembed.__file__).resolve()
    assert loaded_from.is_relative_to(this_checkout_src), (
        f"rxembed imported from {loaded_from}, outside this checkout's {this_checkout_src}"
    )


def test_import_is_quiet_until_set_verbose():
    """A degradation warning reaches stderr unconfigured; INFO narration waits for set_verbose()."""
    assert rxembed.logger.level == 0, "the package logger must stay NOTSET and inherit the app's level"
    err = _stderr_of()
    assert "no angle is fixed" in err, f"the degradation warning never reached stderr: {err!r}"
    assert "embed[molecule]" not in err, "INFO must be quiet until set_verbose()"
    assert "embed[molecule]" in _stderr_of("rx.set_verbose('INFO')\n"), "set_verbose() must still narrate"
