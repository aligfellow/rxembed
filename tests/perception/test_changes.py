"""`perception/changes.py`: the bonds a conformer forms and breaks against its species, its stereo, and its hands."""

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.geometry.constraints import Constraints, SoftKeys
from rxembed.identity.stereoisomers import StereoRequest
from rxembed.perception.changes import Changes, bond_changes, mismatch, signature, specified_like, stereo_filter
from tests.geometry_fixtures import embedded, placed
from tests.metal_fixtures import PD2CL6, ferrocene

# Zeise's anion: Cl0 Pt1 Cl2 Cl3, the ethylene C4=C5 side-on, then its H
ZEISE = placed(
    Chem.AddHs(rx.parse_smiles("[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH2]->1")),
    [
        *((1.02, -2.16, 0.95), (1.13, 0.01, 0.29), (3.24, 0.08, 1.13), (1.47, 2.08, -0.58), (-1.04, 0.28, 0.38)),
        *((-0.76, -0.2, -0.84), (-1.4, -0.4, 1.16), (-1.22, 1.34, 0.53), (-0.88, -1.25, -1.06), (-0.7, 0.48, -1.69)),
    ],
)
# F0 Fe1 Cl2 Br3 I4 on tetrahedral directions
IRON = placed(
    rx.parse_smiles("[F-]->[Fe+2](<-[Cl-])(<-[Br-])<-[I-]"),
    np.array([(1.9, 1.9, 1.9), (0, 0, 0), (2.2, -2.2, -2.2), (-2.4, 2.4, -2.4), (-2.6, -2.6, 2.6)]) / np.sqrt(3),
)
_PD_XY = [(-3.35, 1.6), (-1.75, 0), (-3.35, -1.6), (0, 1.65), (1.75, 0), (3.35, 1.6), (3.35, -1.6), (0, -1.65)]


def _moved(mol, atoms, delta):
    """Return a copy of `mol` with `atoms` shifted by `delta`."""
    out = Chem.Mol(mol)
    pos = out.GetConformer().GetPositions()
    pos[list(atoms)] += delta
    out.GetConformer().SetPositions(pos)
    return out


def _with_mirror(mol):
    """Return a copy of `mol` with a second conformer, the first mirrored in x."""
    out = Chem.Mol(mol)
    conformer = Chem.Conformer(out.GetConformer())
    conformer.SetPositions(conformer.GetPositions() * (-1, 1, 1))
    out.AddConformer(conformer, assignId=True)
    return out


def _glycine_proton_on_oxygen():
    """Return zwitterionic glycine (N0 ... O4-, H5 on N0) with H5 moved 1 A out from O4."""
    mol = embedded("[NH3+]CC(=O)[O-]")
    pos = mol.GetConformer().GetPositions()
    pos[5] = pos[4] + (pos[4] - pos[2]) / np.linalg.norm(pos[4] - pos[2])
    mol.GetConformer().SetPositions(pos)
    return mol


def _titanium_alkyl(chain):
    """Return Cl3Ti-alkyl with no H: Ti0, Cl1-3, then the chain carbons from C4 on Ti, placed at `chain`."""
    rw = Chem.RWMol()
    for symbol in ["Ti", "Cl", "Cl", "Cl", *["C"] * len(chain)]:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for donor in (1, 2, 3, 4):
        rw.AddBond(donor, 0, Chem.BondType.DATIVE)
    for carbon in range(5, 4 + len(chain)):
        rw.AddBond(carbon - 1, carbon, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return placed(mol, [(0, 0, 0), (-2.2, 0, 0), (0, -2.2, 0), (0, 0, 2.2), *chain])


def test_bond_changes_name_each_bond_formed_or_broken():
    """A metal keeps a site it binds through any atom and the atoms bonded to a donor; an ion its ionic contact."""
    axis = np.subtract(*ZEISE.GetConformer().GetPositions()[[5, 4]])
    cases = {
        "proton onto carboxylate": (_glycine_proton_on_oxygen(), Changes(formed=((4, 5),), broken=((0, 5),))),
        "Pt slipped over one C": (_moved(ZEISE, [1], -axis / np.linalg.norm(axis)), Changes()),
        "ethylene-gone": (_moved(ZEISE, range(4, 10), (-4.0, 0, 0)), Changes(broken=((1, 4), (1, 5)))),
        "beta-agostic C at 2.554 A": (_titanium_alkyl([(2.1, 0, 0), (2.038, 1.539, 0)]), Changes()),
        "gamma C folded onto Ti": (
            _titanium_alkyl([(2.1, 0, 0), (2.038, 1.539, 0), (0.639, 2.157, 0)]),
            Changes(formed=((0, 6),)),
        ),
        "KCl contact unread": (placed("[Cl-]->[K+]", [(0, 0, 0), (3.1, 0, 0)]), Changes()),
    }
    for name, (mol, expected) in cases.items():
        assert bond_changes(mol) == expected, name


def test_bond_changes_leave_a_pair_the_request_holds_unjudged():
    """The proton moved from N0 onto O4: a frozen, fixed or contact pair is the request's, a windowed bond stays."""
    mol, n_h, o_h = _glycine_proton_on_oxygen(), Changes(broken=((0, 5),)), Changes(formed=((4, 5),))
    rows = (
        (Constraints(frozen={0, 5}), o_h),
        (Constraints(fixed={(4, 5): (1.0, 1.0)}), n_h),
        (Constraints(distances={(4, 5): (0.9, 1.1)}, contacts=SoftKeys(distances={(4, 5)})), n_h),
        (Constraints(distances={(0, 5): (1.0, 5.0)}), o_h),
    )
    assert [bond_changes(mol, -1, cons) for cons, expected in rows] == [expected for cons, expected in rows]


def test_changes_reads_a_conformers_hand_against_the_reference():
    """The mirror image of a tetrahedral iron contradicts it; a planar chloride-bridged palladium dimer is its own."""
    planar_pd2cl6 = placed(rx.parse_smiles(PD2CL6), [(x, y, 0.0) for x, y in _PD_XY])
    for name, mol, mirror_differs in (("iron", IRON, True), ("pd2cl6", planar_pd2cl6, False)):
        for order in (range(mol.GetNumAtoms()), reversed(range(mol.GetNumAtoms()))):
            both = Chem.RenumberAtoms(_with_mirror(mol), list(order))
            kept, mirrored = (rx.changes(both, cid, reference=both) for cid in (0, 1))
            assert kept.unchanged, name
            assert (mirrored.formed, mirrored.broken, mirrored.unchanged) == ((), (), not mirror_differs), name
    assert rx.changes(_with_mirror(IRON), 1).unchanged  # without a reference the iron specifies no arrangement


def test_changes_drops_a_point_the_conformer_makes_no_centre():
    """C3 of a pentane-2,3,4-triol specified R beside an open C5: like ends leave it no centre, unlike ends read r."""
    for geometry in ("C[C@@H](O)[C@H](O)[C@@H](C)O", "C[C@@H](O)[C@H](O)[C@H](C)O"):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)[C@H](O)C(C)O"))
        mol.AddConformer(embedded(geometry).GetConformer(), assignId=True)
        assert rx.changes(mol).unchanged, geometry


def test_specified_like_reads_a_mol_positions_or_xyz_reference(tmp_path):
    path = tmp_path / "iron.xyz"
    Chem.MolToXYZFile(IRON, str(path))
    mirrored = _with_mirror(IRON).GetConformer(1).GetPositions()

    read = {rx.cxsmiles(specified_like(IRON, form)) for form in (IRON, IRON.GetConformer().GetPositions(), str(path))}

    assert len(read) == 1
    assert read != {rx.cxsmiles(specified_like(IRON, mirrored))}


def test_specified_like_refuses_a_missized_or_missing_reference(tmp_path):
    for reference, message in (
        (placed("CCO", [(0, 0, 0), (1.5, 0, 0), (2, 1.4, 0)]), "give it in mol's order"),
        (str(tmp_path / "missing.xyz"), r"pass an existing \.xyz"),
    ):
        with pytest.raises(ValueError, match=message):
            specified_like(IRON, reference)


def test_changes_describe_names_each_change_by_its_atoms():
    found = Changes(formed=((0, 1),), broken=((1, 2),), stereo="specified C1 R but the conformer has S")

    assert found.describe(Chem.MolFromSmiles("CCO")) == (
        "C0-C1 formed, C1-O2 broken, stereo specified C1 R but the conformer has S"
    )


def test_signature_counts_xyzgraph_hands_but_none_at_a_metal(monkeypatch):
    """A face's hand is its specified winding, so planar labels on a ferrocenium Cp face or its iron are dropped."""
    from xyzgraph import stereo

    mol, seen = ferrocene(), []
    mol.GetAtomWithIdx(0).SetFormalCharge(3)
    summary = {
        "planar": [{"label": "Sₚ", "ring": [1, 2, 3, 4, 5]}, {"label": "Rₚ", "ring": [0, 1, 6]}],
        "helical": [{"label": "M", "atoms": [1, 2]}, {"label": "M", "atoms": [6, 7]}, {"atoms": [3, 4]}],
    }
    monkeypatch.setattr(stereo, "annotate_stereo", lambda graph: seen.append(graph.graph["total_charge"]) or summary)

    assert signature(mol) == {"helical": Counter({"M": 2})}
    assert seen == [1]


def test_mismatch_fails_only_a_forbidden_unexpected_hand():
    rp, sp, both = ({"planar": Counter(hands)} for hands in (["Rₚ"], ["Sₚ"], ["Rₚ", "Sₚ"]))
    for sig, ref, mode, lost in (
        (rp, rp, "preserve", False),
        (sp, rp, "preserve", True),
        (sp, rp, "invert", False),
        (rp, rp, "invert", True),
        (sp, rp, "free", False),
        (sp, rp, "racemic", False),
        (sp, both, "preserve", False),  # a reference showing both hands reads as its own mirror
        ({}, rp, "preserve", False),  # a hand xyzgraph omits is not lost
    ):
        assert (mismatch(sig, ref, StereoRequest.parse(mode)) is not None) == lost, (sig, ref, mode)


def test_stereo_filter_keeps_hands_only_a_source_shows():
    """A 2-fluoro[8]paracyclophane conformer shows a planar hand; ethanol none, and a free request keeps none."""
    cyclophane = embedded("FC1=CC2=CC=C1CCCCCCCC2", seed=1)
    graph_only = Chem.Mol(cyclophane)
    graph_only.RemoveAllConformers()
    for source, mode, kept in (
        (cyclophane, None, True),
        (cyclophane, "free", False),
        (graph_only, None, False),
        (embedded("CCO"), None, False),
    ):
        request = StereoRequest.parse(mode)
        assert stereo_filter(source, request) == ((request, signature(source)) if kept else None), (mode, kept)
