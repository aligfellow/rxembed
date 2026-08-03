"""`stereo.py`; expand a `Mol`'s undefined stereocentres into the distinct species to embed.

The embed-side half of stereochemistry: graph-only, RDKit's `EnumerateStereoisomers` over the *unspecified*
elements alone, so a defined centre is held. Metal-safe: the metal's own handedness is the coordination-isomer
path's job, and a double bond the coordination locks is not enumerated as a phantom pair.

The conformer-side half (the coordinate-derived chirality fingerprint that filters embedded conformers) is
perception-driven and lives in `pipeline/stereo_check.py`; the `stereo=` argument that drives this module from
the pipeline is `pipeline/dispatch.py`.
"""

from __future__ import annotations

from rdkit import Chem

from rxembed import stereo
from rxembed.metal_core import metal_indices

_CHIRAL_P_PD = "C[P](CC)(c1ccccc1)[Pd](Cl)(Cl)Cl"
# an alpha-diimine-style chelate: both imine C=N sit in the 5-membered ring the metal closes
_ALPHA_DIIMINE_NI = (
    "O=C1[O-]->[Ni+2]2(<-[N](=C3C(=[N]->2c2cccc4ccccc24)c2cccc4cccc3c24)c2cccc3ccccc23)<-[N-](c2ccccc2)C1c1ccccc1"
)


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"fixture SMILES did not parse: {smiles}"
    return mol


def _with_metals(smiles):
    """Return ``(mol, metal indices)``: what a coordination caller passes as ``exclude``."""
    mol = _mol(smiles)
    return mol, set(metal_indices(mol))


def _labels(mol, **kw):
    return sorted(label for _variant, label in stereo.enumerate_unassigned(mol, **kw)[0])


# ---------------------------------------------------------------------------------------------------------
# nothing to enumerate: the pass-through, which must be exact
# ---------------------------------------------------------------------------------------------------------


def test_a_molecule_with_no_stereocentre_passes_straight_through():
    """`([(mol, '')], 0, 1, 0)` and the same object: a caller branches on `n_unassigned == 0`."""
    mol = _mol("CCO")
    variants, n_unassigned, total, unresolved = stereo.enumerate_unassigned(mol)
    assert (n_unassigned, total, unresolved) == (0, 1, 0)
    assert variants == [(mol, "")]


def test_a_defined_centre_is_held_not_re_enumerated():
    """`onlyUnassigned`: a labelled centre is the user's stated species, never expanded into its enantiomer."""
    mol = _mol("C[C@H](N)C(=O)O")
    assert stereo.unassigned_centres(mol) == []
    assert stereo.enumerate_unassigned(mol)[1] == 0


# ---------------------------------------------------------------------------------------------------------
# the expansion
# ---------------------------------------------------------------------------------------------------------


def test_an_undefined_point_centre_expands_to_both_hands_with_index_keyed_cip_labels():
    """The label keys on the enumerated atom index, so distinct variants get distinct, stable names."""
    assert _labels(_mol("CC(N)C(=O)O")) == ["1R", "1S"]


def test_an_undefined_double_bond_expands_e_and_z_alongside_the_point_centres():
    """One undefined C plus one undefined C=C is 2 x 2, and each label names both axes."""
    labels = _labels(_mol("CC=CC(N)O"))
    assert len(labels) == 4
    assert sum(":E" in x for x in labels) == 2
    assert sum(":Z" in x for x in labels) == 2


def test_a_meso_duplicate_is_dropped():
    """Two centres give four configurations but only three species; `unique=True` collapses the meso pair."""
    variants, n_unassigned, total, _unresolved = stereo.enumerate_unassigned(_mol("CC(O)C(O)C"))
    assert (n_unassigned, total) == (2, 4)
    assert len(variants) == 3


def test_atom_order_is_preserved_so_index_based_specs_stay_valid():
    """`fix`/`constrain` key on the caller's atom indices, which a reordered variant would silently move."""
    mol = _mol("CC=CC(N)O")
    original = [a.GetAtomicNum() for a in mol.GetAtoms()]
    for variant, _label in stereo.enumerate_unassigned(mol)[0]:
        assert [a.GetAtomicNum() for a in variant.GetAtoms()] == original


def test_the_cap_truncates_the_variants_but_reports_the_true_total():
    """A caller warns off `total`, so it must be the real count and not the truncated one."""
    variants, n_unassigned, total, _unresolved = stereo.enumerate_unassigned(_mol("CC(N)C(=O)O"), cap=1)
    assert (n_unassigned, total) == (1, 2)
    assert len(variants) == 1


# ---------------------------------------------------------------------------------------------------------
# axial chirality: RDKit cannot encode it, so it is REPORTED rather than faked
# ---------------------------------------------------------------------------------------------------------


def test_an_allene_axis_is_counted_as_unresolved_and_never_gets_a_question_mark_label():
    """`EnumerateStereoisomers` cannot set an allene axis, so one arbitrary hand comes back and the caller warns."""
    variants, n_unassigned, _total, unresolved = stereo.enumerate_unassigned(_mol("CC(F)=C=C(F)C"))
    assert n_unassigned == 2
    assert unresolved == 2
    assert [label for _v, label in variants] == [""], "an unresolved centre must be dropped from the label"


# ---------------------------------------------------------------------------------------------------------
# metal safety
# ---------------------------------------------------------------------------------------------------------


def test_a_metal_bound_chiral_phosphorus_survives_the_sphere_strip():
    """The D-cap restores the valence the dative bond does not count, so both P-epimers are still enumerated."""
    mol, metals = _with_metals(_CHIRAL_P_PD)
    labels = _labels(mol, exclude=metals)
    assert len(labels) == 2
    assert len(set(labels)) == 2, "the two P-epimers must get distinct labels"


def test_a_double_bond_the_coordination_locks_is_not_enumerated():
    """A C=N endocyclic in a ring the metal closes has one buildable geometry: the other hand is a phantom.

    `FindPotentialStereo` runs on the metal-disconnected graph, which opens that ring, so the imine looks like
    a free acyclic double bond. An alpha-diimine enumerated 2x2, most of it unbuildable.
    """
    mol, metals = _with_metals(_ALPHA_DIIMINE_NI)
    assert stereo._coordination_locked_double_bonds(mol, metals), "the metal-closed imine was not detected"
    variants, _n, _total, _unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
    assert len(variants) == 2, "only the real point stereocentre should expand, not the locked imines"


def test_a_pendant_double_bond_is_not_locked_just_because_a_metal_is_present():
    """The lock is metal-RING-specific: over-suppressing would drop real organic E/Z from every complex."""
    mol, metals = _with_metals("CC=CC[NH2]->[Ni+2](<-[O-]C(=O)C)<-[NH2]CC=CC")
    assert stereo._coordination_locked_double_bonds(mol, metals) == set()


def test_the_metal_strip_and_the_graft_are_exact_inverses_at_a_defined_donor():
    """A DEFINED donor tag must survive the enumeration round trip verbatim, at an odd slot as at an even one.

    `_build_enumeration_graph` re-bases each donor's tag onto the stripped bond order and `graft` writes the
    enumerated tag back across a bond the full mol still has, so the two are one correction and its inverse.
    They have to move together: correcting only the strip inverts every enumerated hand at an odd slot, and
    no `.xyz` fixture reaches this route to notice (`_load_in_ligand_stereo` returns None whenever the input
    carries a conformer, and every corpus entry is an `.xyz`).
    """
    for smiles in ("Cl[Pd](Cl)(Cl)<-[P@](C)(CC)C(C)(N)O", "[P@](C)(CC)(C(C)(N)O)->[Pd](Cl)(Cl)Cl"):
        mol, metals = _with_metals(smiles)
        donor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P")
        declared = mol.GetAtomWithIdx(donor).GetChiralTag()
        variants, n_unassigned, _total, _unresolved = stereo.enumerate_unassigned(mol, exclude=metals)
        assert n_unassigned == 1, f"{smiles}: the carbon centre was not enumerated, so this asserts nothing"
        for vmol, _label in variants:
            assert vmol.GetAtomWithIdx(donor).GetChiralTag() == declared, smiles


def test_with_no_metal_to_exclude_the_enumeration_graph_is_the_molecule_itself():
    """Nothing to disconnect means nothing to cap, so the metal path never copies or re-sanitises an organic input."""
    mol = _mol("CC(N)C(=O)O")
    work, caps = stereo._build_enumeration_graph(mol, exclude=set())
    assert work is mol
    assert caps == {}
