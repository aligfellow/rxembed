"""Geometry-gate result types -- one ``Violation`` and the ``GeometryReport`` that collects them.

Split out so the two gate modules that both build ``Violation`` objects -- ``geometry`` (the general
checks + ``check()`` orchestration) and ``coordination`` (the metal-sphere gates ``check()`` calls) --
share the type without importing each other. This is the leaf that breaks the would-be report<->checks
cycle: both import from here, here imports nothing back.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Violation:
    """One failed check. ``value`` breached ``limit`` for the atoms named."""

    kind: str
    atoms: tuple[int, ...]
    value: float
    limit: float
    detail: str = ""

    def __str__(self) -> str:
        """Format the violation as a readable one-liner."""
        a = "-".join(map(str, self.atoms))
        return f"[{self.kind}] atoms {a}: {self.value:.3f} vs {self.limit:.3f} {self.detail}".rstrip()


@dataclass
class GeometryReport:
    """Result of ``check()``. Falsy/``ok()`` when there are no violations."""

    violations: list[Violation] = field(default_factory=list)

    def ok(self) -> bool:
        """Return True when the conformer passed every check."""
        return not self.violations

    def __bool__(self) -> bool:
        """Truthy when there are no violations."""
        return self.ok()

    def summary(self) -> str:
        """Format the violations one per line (or an all-clear message)."""
        if self.ok():
            return "geometry OK (no violations)"
        lines = [f"{len(self.violations)} geometry violation(s):"]
        lines += [f"  - {v}" for v in self.violations]
        return "\n".join(lines)

    def assert_ok(self) -> None:
        """Raise ``AssertionError`` with the summary if any check failed (for tests)."""
        if not self.ok():
            raise AssertionError(self.summary())
