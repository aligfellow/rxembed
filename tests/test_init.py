"""`rxembed/__init__.py`: the one public tier, its input boundary and its logging."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

import rxembed as rx

# A real degradation on a base install: two fixed distances over three atoms leave the angle free, and rxembed says so.
_EMBED_TWICE = """
import sys
from rdkit import Chem
import rxembed as rx
assert rx.logger.level == 0 and not rx.logger.handlers, "importing rxembed must configure no logging"
for verbose in (False, True):
    if verbose:
        rx.set_verbose("INFO")
        rx.set_verbose("INFO")
    print("--", file=sys.stderr)
    rx.embed(Chem.AddHs(Chem.MolFromSmiles("OCCCN")), fix={(0, 1): 1.45, (1, 2): 1.55}, n=2, seed=1)
"""
# Each public verb that takes a Mol, its other arguments filled in: (name, call, takes a SMILES, reads a geometry)
PUBLIC_VERBS = (
    ("rx.identity", rx.identity, False, False),
    ("rx.cxsmiles", rx.cxsmiles, False, False),
    ("rx.smiles", rx.smiles, False, False),
    ("rx.isomers", rx.isomers, True, False),
    ("rx.embed", rx.embed, True, False),
    ("rx.minimize", rx.minimize, True, True),
    ("rx.coembed", lambda mol: rx.coembed((mol, mol)), True, False),
    ("rx.changes", rx.changes, False, True),
    ("rx.wrap", rx.wrap, False, True),
    ("rx.contacts", rx.contacts, False, False),
    ("rx.geom_check.check", rx.geom_check.check, False, True),
    ("rx.ligands", rx.ligands, False, False),
    ("rx.match", lambda mol: rx.match(mol, "[OX2H]"), False, False),
    ("rx.read_graph", rx.read_graph, False, False),
)


def test_import_is_quiet_until_set_verbose():
    """A degradation warning reaches stderr unconfigured; INFO narration waits for set_verbose, then shows once."""
    root = str(Path(rx.__file__).resolve().parents[1])  # the rxembed this test imported, not whatever is installed
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH"))))}
    run = subprocess.run([sys.executable, "-c", _EMBED_TWICE], capture_output=True, text=True, check=False, env=env)
    assert run.returncode == 0, run.stderr
    quiet, verbose = run.stderr.split("--\n")[1:]

    assert "no angle is fixed" in quiet, quiet
    assert "INFO" not in quiet, quiet
    assert verbose.count("embed: kept") == 1, verbose
    assert run.stdout == ""


def test_each_public_verb_refuses_unreadable_input_with_a_remedy():
    ethanol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    no_conformer = r" reads an existing geometry and this Mol has no conformer; use rx\.embed"
    for verb, call, takes_smiles, reads_geometry in PUBLIC_VERBS:
        for given, error, remedy, applies in (
            ("CCO", TypeError, r" takes an RDKit Mol, not str; rx\.parse_smiles parses a SMILES", not takes_smiles),
            (ethanol, ValueError, no_conformer, reads_geometry),
            (Chem.Mol(), ValueError, r" got no atoms; pass a species, such as rx\.parse_smiles", True),
        ):
            if applies:
                with pytest.raises(error, match=re.escape(verb) + remedy):
                    call(given)
