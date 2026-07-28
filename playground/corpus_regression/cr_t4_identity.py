"""Is T4 EXACTLY inert on the corpus, or merely equal to the printed precision?

Compares the tree2 (T3 only) and tree3 (T3+T4) result JSONs field by field at full float precision.
An exact match on every conformer of every structure at every seed is the strongest available
statement that the change cannot reach this corpus — stronger than any aggregate agreeing to 4dp.

Usage: uv run python cr_t4_identity.py <tree2.json> <tree3.json>
"""

from __future__ import annotations

import json
import sys


def walk(a, b, path=""):
    """Yield (path, a, b) for every leaf that differs."""
    if type(a) is not type(b):
        yield path, a, b
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")
            else:
                yield from walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}[len]", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b


if __name__ == "__main__":
    A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    ia = {(r["name"], r["seed"]): r for r in A}
    ib = {(r["name"], r["seed"]): r for r in B}
    common = sorted(set(ia) & set(ib))
    print(f"records: A={len(ia)} B={len(ib)} common={len(common)}")
    diffs, structs = [], set()
    for k in common:
        a, b = dict(ia[k]), dict(ib[k])
        a.pop("secs", None)  # wall-clock is not a result
        b.pop("secs", None)
        for p, x, y in walk(a, b, f"{k[0]}@{hex(k[1])}"):
            diffs.append((p, x, y))
            structs.add(k[0])
    print(f"differing leaves: {len(diffs)}  across {len(structs)} structure(s)")
    for p, x, y in diffs[:40]:
        print(f"  {p}: {x!r} != {y!r}")
    if not diffs:
        print("\nEXACT IDENTITY: T4 changes NOTHING on this corpus, at full float precision.")
