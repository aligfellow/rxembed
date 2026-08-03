"""The package surface, the tier rule, and the logging contract: the three things `import rxembed` promises.

The tier rule is enforced here by an AST walk rather than by importing anything, because the escape that has
actually happened in this project every time is a lazy `import networkx` inside a function body: it never
runs on a dev box, so no `sys.modules` probe sees it, and it reaches a base-install user as an ImportError
from the middle of a calculation. A walk sees it at any nesting depth.

The mirror half, every `pipeline/` module must still import on a base install, is a static per-import check
in `tests/pipeline/test_init.py`.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

import rxembed

_SRC = pathlib.Path(rxembed.__file__).parent
_BASE = {"numpy", "rdkit"}  # what a base install has, and so all a core module may reach for
# The one core exception: `metal_sphere.py`'s solver guards its own import and degrades with a message, which
# is why `sphere` is a core extra. Naming a module here buys a guarded import, not a free one.
_GUARDED = {"metal_sphere.py": {"scipy"}}

# A real degradation on a base install: two fix distances over three atoms leave the angle free, and the
# library says so. In a SUBPROCESS because pytest's log capture installs a root handler, which supplies the
# handler whose absence is the whole point.
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


def test_version_is_a_string():
    assert isinstance(rxembed.__version__, str)


def test_core_surface():
    for name in ("embed", "minimize", "enumerate_isomers", "Conformers", "Constraints", "Isomer", "IsomerSet"):
        assert hasattr(rxembed, name), name


def test_root_does_not_reach_into_the_pipeline():
    # the tier rule: a base install (numpy + rdkit) must be able to `import rxembed`, so no pipeline
    # name may be reachable from the root; `pipeline` itself is only there once someone imports it.
    for name in ("Ensemble", "EnsembleSet", "metal", "nci_modes", "geometry", "geom_check", "wrap"):
        assert not hasattr(rxembed, name), name


def test_pipeline_surface_is_a_superset():
    import rxembed.pipeline as rxp

    for name in ("embed", "minimize", "wrap", "metal", "nci_modes", "nci_candidates", "Contact", "geom_check"):
        assert hasattr(rxp, name), name
    for name in rxembed.__all__:  # every core name is re-exported, so `import rxembed.pipeline as rx` suffices
        assert hasattr(rxp, name), name


def test_the_two_embed_verbs_are_different_functions():
    import rxembed.pipeline as rxp

    assert rxembed.embed is not rxp.embed
    assert rxembed.minimize is not rxp.minimize


def test_a_warning_reaches_stderr_with_no_logging_setup():
    """Fail loud, not silent; on a bare install too: a degradation the user can act on must be visible.

    A NullHandler on the package logger defeats logging's last-resort stderr handler: the only channel a
    WARNING has when nothing configures logging, so every warning here was silent unless `set_verbose` was
    called first.
    """
    err = _stderr_of()
    assert "no angle is fixed" in err, f"the degradation warning never reached stderr: {err!r}"


def test_import_alone_configures_nothing_so_info_waits_for_set_verbose():
    """The complement: not suppressing warnings must not become configuring logging for the user.

    Warnings ride the last resort (WARNING and above); the stage narration stays off until asked for.
    """
    assert rxembed.logger.level == 0, "the package logger must stay NOTSET and inherit the app's level"
    assert "embed[molecule]" not in _stderr_of(), "INFO must be quiet until set_verbose()"
    assert "embed[molecule]" in _stderr_of("rx.set_verbose('INFO')\n"), "set_verbose() must still narrate"


# ---------------------------------------------------------------------------------------------------------
# The tier rule: core is numpy + rdkit + stdlib + its own siblings, and never reaches into `pipeline`
# ---------------------------------------------------------------------------------------------------------


def _core_modules():
    """Every core source file: the flat root and any subdirectory of it that is not `pipeline/`."""
    return sorted(p for p in _SRC.rglob("*.py") if "pipeline" not in p.relative_to(_SRC).parts)


def _tier_violations(rel, src):
    """Report every import in core module `rel` (source `src`) that the tier rule forbids.

    Takes the source as text rather than a path so the rule can be checked against a synthetic module; see
    `test_the_rule_bites`. `rel` is the path under `src/rxembed/`; it decides the package a relative import
    resolves against, and which guarded dependency (if any) this module is allowed.
    """
    pkg = ".".join(("rxembed", *rel.parts[:-1]))
    allowed = _BASE | set(sys.stdlib_module_names) | _GUARDED.get(rel.name, set())
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            pairs = [(a.name.split(".")[0], {a.name}) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
            mod = (f"{base}.{node.module}" if node.module else base) if node.level else (node.module or "")
            pairs = [("" if node.level else mod.split(".")[0], {mod, *(f"{mod}.{a.name}" for a in node.names)})]
        else:
            continue
        for top, reaches in pairs:
            where = f"{rel}:{node.lineno}"
            targets = {t for t in reaches if t == "rxembed" or t.startswith("rxembed.")}
            if pipeline := sorted(t for t in targets if t.startswith("rxembed.pipeline")):
                bad.append(f"{where}: core reaches into the pipeline -> {pipeline[0]}")
            elif top == "rxembed":
                bad.append(f"{where}: absolute self-reference -> {sorted(targets)[0]} (core imports with `.`)")
            elif top and top not in allowed:
                bad.append(f"{where}: not a base dependency -> {top} (core is numpy + rdkit + stdlib)")
    return bad


def test_core_reaches_no_further_than_numpy_rdkit_and_its_own_siblings():
    """A core module that grows a third-party or pipeline import breaks `pip install rxembed` for everyone."""
    bad = [v for p in _core_modules() for v in _tier_violations(p.relative_to(_SRC), p.read_text())]
    assert not bad, "the tier rule (AGENTS.md) is broken:\n  " + "\n  ".join(bad)


def test_the_perception_leaf_reaches_no_further_than_the_shared_leaf():
    """`pipeline/perceive.py` is the bottom of the stack: importing back up re-forms the cycle it broke.

    It was split out precisely to break metal -> dispatch -> metal, and an import in this direction is how
    that would come back; silently, since the dev environment imports both halves anyway. `rxembed.utils` is
    the one exemption and cannot re-form it, being the numpy + rdkit leaf that imports no sibling of its own.
    It is exempt because perception WRITES stereo tags, and the rule for writing one (a tag is a parity over
    a bond order, and RDKit's 3D writer uses a different one from every reader) has to live in a single
    place or it drifts; that is the same reason `utils.remove_bond` is the only bond removal in the core.
    """
    leaf = _SRC / "pipeline" / "perceive.py"
    reaches = []
    for node in ast.walk(ast.parse(leaf.read_text())):
        if isinstance(node, ast.Import):
            reaches += [a.name for a in node.names if a.name.split(".")[0] == "rxembed"]
        elif isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("rxembed")):
            reaches.append("." * node.level + (node.module or ""))  # a relative import here IS an rxembed one
    assert not set(reaches) - {"rxembed.utils"}, (
        f"pipeline/perceive.py may import rxembed.utils and nothing else -> {sorted(reaches)}"
    )


@pytest.mark.parametrize(
    ("escape", "why"),
    [
        ("import networkx", "a third-party import at module scope"),
        ("def f():\n    import networkx", "the LAZY form: never runs, so no sys.modules probe can see it"),
        ("import rxembed.pipeline", "core reaching into the optional tier"),
        ("from .pipeline.geom_check import check", "the same edge spelled relatively"),
        ("import rxembed.relax", "an absolute self-reference, which stops the core relocating as a unit"),
    ],
)
def test_the_rule_bites(escape, why):
    """A gate nobody has seen fail is enforcing nothing, so check it against each escape it exists to stop."""
    assert _tier_violations(pathlib.Path("relax.py"), escape), why


# ---------------------------------------------------------------------------------------------------------
# One door for bond removal, so the chiral-tag parity rule cannot be forgotten at a new surgery site
# ---------------------------------------------------------------------------------------------------------


def _direct_bond_removals(rel, src):
    """Report every raw ``RWMol.RemoveBond`` call in `src`, which `utils.remove_bond` is there to replace."""
    return [
        f"{rel}:{n.lineno}"
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "RemoveBond"
    ]


def test_utils_remove_bond_is_the_only_bond_removal_in_core():
    """A chiral tag is a parity over the atom's bond order, and `RWMol.RemoveBond` does not know that.

    The defect's real shape is "someone adds a bond removal and does not know a tag is a parity", and it is
    invisible to every test that does not measure a hand: no CIP complaint, no valence complaint, a
    plausible-looking geometry that is the mirror image. Making `utils.remove_bond` the only door turns that
    from vigilance into a failing build.
    """
    bad = [
        v
        for p in _core_modules()
        if p.name != "utils.py"  # the door itself is the one RemoveBond, and it re-bases both ends first
        for v in _direct_bond_removals(p.relative_to(_SRC), p.read_text())
    ]
    assert not bad, "these bond removals bypass the chiral-tag rule (use utils.remove_bond):\n  " + "\n  ".join(bad)


def test_the_bond_removal_rule_bites():
    """A gate nobody has seen fail is enforcing nothing."""
    assert _direct_bond_removals(pathlib.Path("relax.py"), "def f(rw):\n    rw.RemoveBond(1, 2)")


def _direct_stereo_writes(rel, src):
    """Report every raw ``AssignStereochemistryFrom3D`` call, which `utils.assign_stereo_from_3d` replaces."""
    return [
        f"{rel}:{n.lineno}"
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "AssignStereochemistryFrom3D"
    ]


def test_utils_assign_stereo_from_3d_is_the_only_stereo_writer():
    """The other half of the same rule: RDKit's 3D writer uses a bond order none of its readers use.

    It drops a dative bond leaving the centre; the embedder, both CIP labellers and the SMILES writer count
    it. So a raw call writes a tag that means the mirror to everything downstream, at every dative-bonded
    donor, with no complaint from anything. The door re-bases it at the writer, where the provenance is known
    and nothing has to guess it later. Covers `pipeline/` too, since two of the three writers live there.
    """
    bad = [
        v
        for p in sorted(_SRC.rglob("*.py"))
        if p.name != "utils.py"  # the door itself is the one raw call
        for v in _direct_stereo_writes(p.relative_to(_SRC), p.read_text())
    ]
    assert not bad, "these stereo writes bypass the re-base (use utils.assign_stereo_from_3d):\n  " + "\n  ".join(bad)


def test_the_stereo_writer_rule_bites():
    """A gate nobody has seen fail is enforcing nothing."""
    assert _direct_stereo_writes(pathlib.Path("relax.py"), "def f(m):\n    Chem.AssignStereochemistryFrom3D(m)")


def test_every_module_has_exactly_one_test_file_named_for_it():
    """`ls tests/` is the coverage map, so the mirror has to be exact: no orphans in either direction.

    A test file that mirrors nothing is a place for cross-cutting tests to accumulate until nobody can say
    what is covered; a module with no file is a gap that reads as covered. Both were true here (five extra
    files, `test_mol_state` / `test_embed_core` / `test_seed_windows` / `test_import` / `test_extras`) until
    each was folded into the module whose behaviour it actually pins.
    """
    tests = pathlib.Path(__file__).parent

    def mirrored(src, suite):
        modules = {"init" if p.stem == "__init__" else p.stem for p in src.glob("*.py")}
        files = {p.stem.removeprefix("test_") for p in suite.glob("test_*.py")}
        return modules, files

    for src, suite in ((_SRC, tests), (_SRC / "pipeline", tests / "pipeline")):
        modules, files = mirrored(src, suite)
        assert not modules - files, f"{src.name}: modules with no test file -> {sorted(modules - files)}"
        assert not files - modules, f"{src.name}: test files mirroring no module -> {sorted(files - modules)}"


def test_tests_holds_nothing_but_tests():
    """No `conftest.py`, no shared helper module: what a test needs, it says in the test."""
    tests = pathlib.Path(__file__).parent
    stray = sorted(
        str(p.relative_to(tests))
        for p in tests.rglob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("test_")
    )
    assert not stray, f"non-test modules under tests/ -> {stray}"
