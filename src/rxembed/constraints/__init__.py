"""Constraint sources and builders (re-exports from the rdkit_embed kernel).

The constraint model (`base`, `builders`) lives in the `rxembed.rdkit_embed` kernel subpackage; this shell
package retains only `nci` (networkx/xyzgraph). These re-exports keep `from rxembed.constraints import
Constraints / add_distance / resolve_core` working for the pipeline and any downstream caller.
"""

from rxembed.rdkit_embed.constraints.base import Constraints, add_distance
from rxembed.rdkit_embed.constraints.builders import match, resolve_atom, resolve_core

__all__ = ["Constraints", "add_distance", "match", "resolve_atom", "resolve_core"]
