"""The logging contract and the package surface: what `import rxembed` promises a user.

A degradation the user can act on has to be visible without them configuring anything first, and the
inverse -- importing a library must not configure logging on its behalf. Both are checked in a
SUBPROCESS, because pytest's log capture installs a root handler and so supplies the very handler whose
absence is the thing under test.
"""

import subprocess
import sys

import rxembed

# A real degradation on a base install: two fix distances over three atoms leave the angle free, and the
# library says so.
_DEGRADES = (
    "from rdkit import Chem\n"
    "import rxembed as rx\n"
    "{prelude}"
    "rx.embed(Chem.AddHs(Chem.MolFromSmiles('OCCCN')), fix={{(0, 1): 1.45, (1, 2): 1.55}}, n=2, seed=1)\n"
)


def _stderr_of(prelude=""):
    run = subprocess.run(
        [sys.executable, "-c", _DEGRADES.format(prelude=prelude)], capture_output=True, text=True, check=True
    )
    return run.stderr


def test_core_surface():
    for name in ("embed", "minimize", "enumerate_isomers", "Conformers", "Constraints", "Isomer", "IsomerSet"):
        assert hasattr(rxembed, name), name


def test_a_warning_reaches_stderr_with_no_logging_setup():
    err = _stderr_of()
    assert "no angle is fixed" in err, f"the degradation warning never reached stderr: {err!r}"


def test_import_alone_configures_nothing_so_info_waits_for_set_verbose():
    assert rxembed.logger.level == 0, "the package logger must stay NOTSET and inherit the app's level"
    assert "embed[molecule]" not in _stderr_of(), "INFO must be quiet until set_verbose()"
    assert "embed[molecule]" in _stderr_of("rx.set_verbose('INFO')\n"), "set_verbose() must still narrate"
