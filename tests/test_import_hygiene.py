"""Kernel import hygiene: the embed kernel must not reach up into the shell.

Nothing else enforces this — ruff's `PLC0415` (import-outside-top-level) is disabled, so a new *lazy*
upward import inside a function body passes lint silently, which is how every existing escape got there.

The check is STATIC, not a `sys.modules` probe: importing any `rxembed.X` first executes the package
`__init__`, which pulls the whole shell, so runtime import sets cannot express a per-module edge until
the kernel is a separate top-level package. Walking the AST sees function-level imports too, which is
precisely where the escapes live.

It is an ALLOWLIST, not a blocklist — that is the whole point of this file. A shell blocklist keyed on the
layer name (`parts[1]`) is blind to a shell module that shares a package with the kernel: `constraints/`
holds kernel `base.py` beside shell `nci.py` (networkx), `embed/` holds kernel `bounds.py` beside shell
`mc.py` (openconf), `refine/` holds kernel `ff.py` beside shell `xtb.py`/`calculator.py` (the xtb/ase
calculators). All four collapse to a kernel-looking layer (`constraints`, `embed`, `refine`) and slip
through. So instead we name every kernel MODULE below; anything else under `rxembed` is shell by
definition, and adding a new kernel module is a one-line addition to `_KERNEL`.
"""

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "rxembed"
# The whole kernel now lives in-place as the `rxembed.rdkit_embed` subpackage (src/rxembed/rdkit_embed) — one
# source root again. The excused facades an absolute import passes through are the `rxembed` shell root and the
# `rxembed.rdkit_embed` kernel package (see `_FACADES`); a kernel module may import only the allowlist plus them.
_ROOTS = {"rxembed": _SRC}
_KERNEL_PKG = "rxembed.rdkit_embed"
_FACADES = {"rxembed", _KERNEL_PKG}


def _module_name(path):
    for pkg, root in _ROOTS.items():
        if path.is_relative_to(root):
            rel = path.relative_to(root).with_suffix("")
            parts = [p for p in rel.parts if p != "__init__"]
            return ".".join([pkg, *parts])
    raise ValueError(f"{path} is under no known source root")


# Every module the embed kernel is built from — `bounds`, the constraint model, the shared perception, and the
# leaves they read. The WHOLE kernel lives in the `rxembed.rdkit_embed` subpackage (src/rxembed/rdkit_embed);
# no kernel module lives elsewhere under `rxembed`. A kernel module may import these and nothing else under
# `rxembed`; everything absent (pipeline, dedup, refine calculators, nci, viz, isomers, stereo, metrics,
# geometry, embed.dispatch, embed.mc) is shell. `geometry` is the QA gate: it imports FROM the kernel
# (`coordination`'s perception) but no kernel module imports it — the perception the FF caps once reached into
# it for (`conjugated_quartets`, `_SP2_DEGREE`) now lives kernel-side in `coordination`.
# `rxembed.rdkit_embed.constraints` is the sole kernel subpackage listed — its `__init__` re-exports base +
# builders only, so the metal layer's `from . import polyhedron` anchor resolves clean. The `rxembed` shell
# root and `rxembed.rdkit_embed` kernel package are the facades every absolute import passes through; they are
# excused in `_shell_reached` (via `_FACADES`) and judged only through the submodules.
_KERNEL = frozenset(
    f"{_KERNEL_PKG}.{m}"
    for m in (
        "io",
        "log",
        "report",
        "vecmath",
        "coordination",
        "constraints",
        "constraints.base",
        "constraints.builders",
        "constraints.coordination_builders",
        "constraints.polyhedron",
        "constraints.sphere",
        "constraints.mechanisms",
        "constraints.metal",
        "constraints.distance",
        "constraints.donor_orient",
        "constraints.solver",
        "embed.bounds",
        "refine.ff",
    )
)

_MODULE_PATHS = {_module_name(p): p for root in _ROOTS.values() for p in root.rglob("*.py")}
_ALL_MODULES = frozenset(_MODULE_PATHS)


def _imports(path):
    """Every `rxembed.*` module this file imports, at ANY nesting depth, absolute or relative."""
    tree = ast.parse(path.read_text())
    pkg = _module_name(path).rsplit(".", 1)[0] if path.name != "__init__.py" else _module_name(path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] in _ROOTS)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import — resolve against this file's package
                base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            if mod.split(".")[0] not in _ROOTS:
                continue
            found.add(mod)
            found.update(f"{mod}.{a.name}" for a in node.names)  # `from pkg import submodule`
    return found


def _module_of(dotted):
    """Resolve a dotted import target to the rxembed module it lives in (drop trailing symbol names).

    `rxembed.constraints.metal.TRANSITION_METALS` -> `rxembed.constraints.metal`; a `from pkg import sub`
    already yields the submodule directly. This is what lets the allowlist judge the leaf a shell package
    shares with the kernel — `constraints.nci` resolves to itself, not to the kernel `constraints` anchor.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in _ALL_MODULES:
            return candidate
    return dotted


def _shell_reached(imports):
    """The shell modules an import set reaches — the kernel allowlist and the package facades excused."""
    return sorted({_module_of(m) for m in imports} - _KERNEL - _FACADES)


@pytest.mark.parametrize("module", sorted(_KERNEL))
def test_every_kernel_module_imports_only_the_kernel(module):
    """The kernel is import-closed: no kernel module reaches a shell module, at any nesting depth."""
    offenders = _shell_reached(_imports(_MODULE_PATHS[module]))
    assert offenders == [], f"{module} reaches up into the shell: {offenders}"


def test_the_allowlist_names_only_real_modules():
    """A `_KERNEL` entry that maps to no file is a stale allowlist, not a passing kernel — fail loud on it."""
    assert sorted(_KERNEL - _ALL_MODULES) == []


def test_io_is_the_bottom_of_the_stack():
    """`rdkit_embed.io` must import no kernel module at all — that is what lets the metal layer read a source."""
    assert sorted(_imports(_SRC / "rdkit_embed" / "io.py")) == []


def test_inputs_is_the_bottom_of_the_stack():
    """`rxembed.inputs` (the shell IO leaf) must import NOTHING from `rxembed` — the same cycle guard as `io`.

    `_xyz_to_mol`/`parse_smiles` are reached by both `embed.dispatch` and `isomers`; if `inputs` reached back
    into `rxembed` it would re-form the metal->dispatch->metal cycle the readers were moved out here to break.
    """
    assert sorted(_imports(_SRC / "inputs.py")) == []


def test_metal_does_not_reach_the_embed_dispatch():
    """metal -> dispatch -> metal was a live cycle the handoff plan does not name.

    `enumerate_isomers` imported `_xyz_to_mol`/`parse_smiles` from `embed.dispatch` inside the function
    body, and `dispatch` imports `constraints.metal` at module level — so calling `rx.metal(...)` dragged
    the entire shell. The shell leaf `rxembed.inputs` now owns both readers (importing nothing from
    `rxembed`), which both `embed.dispatch` and `isomers` reach without re-forming the cycle.
    """
    reached = {_module_of(m) for m in _imports(_SRC / "rdkit_embed" / "constraints" / "metal.py")}
    assert not [m for m in reached if m.startswith("rxembed.embed")], f"metal -> embed: {sorted(reached)}"


@pytest.mark.parametrize(
    "edge",
    [
        "rxembed.constraints.nci",  # networkx / xyzgraph — the KINDS registry + binding modes
        "rxembed.embed.mc",  # openconf — the Monte-Carlo search
        "rxembed.refine.xtb",  # the xtb executable interface
        "rxembed.refine.calculator",  # ase — the calculator resolve / XTB / ASE surface
        "rxembed.pipeline",
        "rxembed.dedup",
        "rxembed.viz",
        "rxembed.isomers",
        "rxembed.stereo",
        "rxembed.metrics",
        "rxembed.geometry",  # the QA gate — shell now; the kernel reads coordination's perception, never this
    ],
)
def test_an_injected_kernel_to_shell_edge_is_caught(edge):
    """Both import shapes of a shell edge trip the allowlist, at the leaf.

    The widened `_SHELL`/`_layer` blocklist missed the first four: their layer name (`parts[1]`) is a kernel
    package — `_layer("rxembed.constraints.nci")` returned `"constraints"`, not in `_SHELL`, so the edge
    passed silently. The allowlist judges the module, so a shell leaf inside a kernel package is caught even
    though its `from <kernel-package> import <leaf>` anchor is itself allowed.
    """
    parent = edge.rsplit(".", 1)[0]
    assert _shell_reached({edge}), f"a direct `import {edge}` slipped the check"
    assert _shell_reached({parent, edge}), f"a `from {parent} import ...` shape slipped the check"
