"""Test polyhedron records, aliases and derived predicates."""

from __future__ import annotations

from rxembed.metal_polyhedron import (
    POLYHEDRA,
    describe,
    resolve_geometry,
)

# --- codes and aliases --------------------------------------------------------------------------------


def test_aliases_are_unique_and_case_insensitive():
    names, codes = [n.lower() for n in POLYHEDRA], [p.code.lower() for p in POLYHEDRA.values() if p.code]
    assert len(codes) == len(set(codes)), "two polyhedra share a 3-letter code"
    assert not set(names) & set(codes), "a polyhedron name is spelled like another's code"

    for spelling in ("OCT", "oct", " Oct ", "octahedral", "OCTAHEDRAL"):
        assert resolve_geometry(spelling) == "octahedral"

    assert describe("SPL") == "square_planar (SPL, CN 4)"
    assert describe("3-coordinate") == "3-coordinate"  # the from_geometry pseudo-name is not invented into a record


# --- the records themselves ---------------------------------------------------------------------------


# --- the fold group, the seating parity and the canonical slot labelling -------------------------------


# --- the convex-hull edge test, for the metal_slots chelate edge rule --------------------------------


# --- relaxed_shell ------------------------------------------------------------------------------------
