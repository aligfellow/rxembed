"""Hold the package layers AGENTS.md states.

The shared foundations sit flat at the top, each folder imports only the foundations and the folders before it in
`LAYERS`, and `api` and `ensemble` compose them. `ROWS` orders every module so that each imports only modules listed
before it. The foundations and core folders use only the standard library, NumPy and RDKit.
"""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "rxembed"
FOUNDATIONS = ("utils", "graph", "stereo")
LAYERS = ("foundations", "geometry", "metal", "identity", "engine", "perception", "workflow", "top")
CORE = LAYERS[:5]
ROWS = """utils graph stereo
geometry.constraints geometry.mechanisms geometry.relax geometry.bounds
metal.polyhedra metal.centres metal.distances metal.orientation metal.shape metal.stereo metal.slots metal.sphere
identity.stereoisomers identity.species identity.smiles_io identity.arrangements identity.reading
engine.acceptance engine.seeding engine.conformers engine.atom_mapping engine.coembed
perception.xyz2mol_local perception.xyz2mol_tmc perception.readers perception.contacts perception.changes
workflow.calculators workflow.search workflow.select workflow.viz workflow.geom_check
ensemble api""".split()
# Each module's syntax tree by its dotted name under rxembed, package `__init__` files aside
TREES = {
    ".".join(path.relative_to(SRC).with_suffix("").parts): ast.parse(path.read_text())
    for path in SRC.rglob("*.py")
    if path.name != "__init__.py"
}


def _layer(name):
    """Return the layer of a dotted module name: its folder, "foundations" or "top"."""
    return name.rpartition(".")[0] or ("foundations" if name in FOUNDATIONS else "top")


def _imports(tree):
    """Yield ``(module, node)`` for each module an import names, each name a `from` import takes as a submodule too."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((alias.name, node) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node
            yield from ((f"{node.module}.{alias.name}", node) for alias in node.names)


def _core():
    """Yield the name and tree of each module in a core layer."""
    yield from ((name, tree) for name, tree in TREES.items() if _layer(name) in CORE)


def test_each_module_imports_only_rows_above_it_in_layer_order():
    order = [LAYERS.index(_layer(name)) for name in ROWS]
    assert sorted(ROWS) == sorted(TREES)
    assert order == sorted(order)

    rank: dict[str, int] = {name: row for row, name in enumerate(ROWS)}
    violations = [
        f"{name} imports {imported}, which is no module in a row above it"
        for name, tree in TREES.items()
        for module, _node in _imports(tree)
        if (imported := module.removeprefix("rxembed.")) in rank and rank[imported] >= rank[name]
    ]
    assert not violations, "\n".join(violations)


def test_core_imports_only_the_stdlib_numpy_rdkit_and_core():
    allowed = {"numpy", "rdkit", "__future__", *sys.stdlib_module_names}
    violations = {
        f"{name} imports {module}"
        for name, tree in _core()
        for module, _node in _imports(tree)
        if (parts := module.split("."))[0] not in {"rxembed", *allowed}
        or (parts[0] == "rxembed" and (len(parts) < 2 or parts[1] not in {*CORE, *FOUNDATIONS}))
    }

    assert not violations, "\n".join(sorted(violations))


def test_core_modules_import_each_other_at_module_top():
    local = set()
    for name, tree in _core():
        type_checking = (b for b in tree.body if isinstance(b, ast.If) and getattr(b.test, "id", "") == "TYPE_CHECKING")
        allowed = {id(node) for node in tree.body} | {id(node) for branch in type_checking for node in ast.walk(branch)}
        local |= {
            f"{name} imports {node.module} inside a function"
            for _module, node in _imports(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("rxembed")
            and id(node) not in allowed
        }

    assert not local, "\n".join(sorted(local))
