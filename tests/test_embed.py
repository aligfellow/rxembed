"""Test the core embed and Conformers API."""

from __future__ import annotations

import importlib
import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, SetBondLength

import rxembed as rx
from rxembed import bounds as bnd
from rxembed.constraints import Constraints, resolve_core
from rxembed.embed import BASE_STIFFNESS, Conformers, Failure, embed, fold_substrate, minimize
from rxembed.metal_core import COORDINATION_METALS, VACANT, donor_chirality_sign, state_with_winding
from rxembed.metal_enumeration import enumerate_isomers
from rxembed.metal_isomer import Isomer, from_geometry
from rxembed.metal_perceive import classify_geometry
from rxembed.metal_polyhedron import POLYHEDRA
from rxembed.metal_smiles import cxsmiles, parse_smiles
from rxembed.relax import bonding_failure
from rxembed.stereo import axis_stereo, point_stereo, stereo_from_3d

emb = importlib.import_module("rxembed.embed")  # the engine implementation module, not the public facade

_BIPY_PD = "Cl[Pd]1(Cl)<-n2ccccc2-c2ccccn->12"
# the N-bound Ni(II) isomer: the window relax tears 3 of 8 seeds at the base stiffness, the ladder's type case
# cis-[Co(en)2Cl2], the textbook Delta/Lambda pair, and the same complex with one sp3 centre on a backbone:
# the second is the case a reflection cannot repair, because it would invert that centre too
_CO_EN = "Cl[Co]12(Cl)(NCCN1)NCCN2"


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _with_geometry(smiles, seed=7):
    mol = _mol(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _isomer(smiles=_BIPY_PD, geometry="square_planar"):
    """The first enumerated coordination isomer of `smiles`: the metal spec `embed` accepts."""
    return next(iter(enumerate_isomers(_mol(smiles), geometry)))


# ---------------------------------------------------------------------------------------------------------
# the everyday spine
# ---------------------------------------------------------------------------------------------------------


def test_numeric_fix_rejects_when_cleanup_cannot_hold_it(monkeypatch, caplog):
    """A numeric fix= that fresh-seed replacement cannot hold is rejected, not silently returned off-target."""
    fix = {(1, 2): 2.026, (2, 0): 1.557}
    confs = embed(_mol("[O-].ClCCCCBr"), fix=fix, n=1, params=bnd.EmbedParams(seed=1, prune_rms=-1))
    assert confs._fixed_geometry_misses(confs.ids[0]), "the raw seed must miss for this test to exercise rejection"

    monkeypatch.setattr(
        emb,
        "restrained_uff",
        lambda mol, cons, conf_ids=None, **kw: np.zeros(len(conf_ids if conf_ids is not None else mol.GetConformers())),
    )
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        with pytest.raises(ValueError, match="fix="):
            confs.minimize()

    assert not confs.ids, "an off-target numeric fix was returned after cleanup"
    assert not confs.unrelaxed, "rejected conformer ids leaked into tracked state"
    assert "requested" in caplog.text
    assert "got" in caplog.text


def test_undefined_dihedral_fails_numeric_fix_gate():
    mol = _with_geometry("CCCC")
    conf = mol.GetConformer()
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0))):
        conf.SetAtomPosition(atom, xyz)
    cons = resolve_core(mol, fix={(0, 1, 2, 3): 60.0}, has_geometry=True)[0]
    misses = Conformers(mol, [0], cons)._fixed_geometry_misses(0)
    assert len(misses) == 1
    assert np.isinf(misses[0][0])


@pytest.mark.parametrize(
    ("smiles", "symbol", "request_geometry", "observed_polyhedron", "vacancy", "raw_conformer"),
    [
        ("[Pt](F)(Cl)(Br)I", "Pt", "tetrahedral", "seesaw", False, False),
    ],
    ids=["tet-see"],
)
def test_coordination_gate_rejects_a_different_named_shell(
    smiles, symbol, request_geometry, observed_polyhedron, vacancy, raw_conformer
):
    """A shell that reads as a different named polyhedron is rejected, with both residuals reported."""
    iso = _isomer(smiles, request_geometry)
    if raw_conformer:
        mol, cid = iso.mol, 0
        conf = Chem.Conformer(mol.GetNumAtoms())
    else:
        confs = embed(iso, n=1, seed=1)
        mol, cid = confs._mol, confs.ids[0]
        conf = mol.GetConformer(cid)
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    donors = [v for v in iso.vertices if v != VACANT] if vacancy else iso.vertices
    for donor, direction in zip(donors, POLYHEDRA[observed_polyhedron].vertex_dirs, strict=True):
        conf.SetAtomPosition(donor, tuple(map(float, 2 * np.asarray(direction))))
    if raw_conformer:
        mol.AddConformer(conf)
        assert classify_geometry(mol, iso.metal, list(iso.vertices), cid) == observed_polyhedron

    failure = emb._coordination_state_failure(mol, cid, iso)

    assert failure is not None
    requested_code, observed_code = POLYHEDRA[request_geometry].code, POLYHEDRA[observed_polyhedron].code
    assert re.fullmatch(
        rf"{symbol}{iso.metal} relaxes from {requested_code} \(\d\.\d{{3}}\) to {observed_code} \(\d\.\d{{3}}\)",
        str(failure),
    )


def test_coordination_gate_rejects_a_better_exact_search_slot_assignment():
    iso = _isomer("[Fe](N)(O)(F)(Cl)Br", "square_pyramidal")
    confs = embed(iso, n=1, seed=1)
    conf = confs._mol.GetConformer(confs.ids[0])
    conf.SetAtomPosition(iso.metal, (0.0, 0.0, 0.0))
    ideal = np.asarray(POLYHEDRA["square_pyramidal"].vertex_dirs, dtype=float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    alternate = ideal[[1, 2, 3, 0, 4]]
    observed = 0.49 * ideal + 0.51 * alternate
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    for donor, direction in zip(iso.vertices, observed, strict=True):
        conf.SetAtomPosition(donor, tuple(2.0 * direction))

    assert classify_geometry(confs._mol, iso.metal, list(iso.vertices), confs.ids[0]) == "square_pyramidal"
    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_coordination_gate_rejects_nonfinite_donor_coordinates():
    iso = _isomer()
    confs = embed(iso, n=1, seed=1)
    confs._mol.GetConformer(confs.ids[0]).SetAtomPosition(iso.donors[0], (float("nan"), 0.0, 0.0))

    assert emb._coordination_state_failure(confs._mol, confs.ids[0], iso) is not None


def test_structural_gate_rejects_undefined_coplanarity():
    mol = _with_geometry("CCCC")
    conf = mol.GetConformer()
    for atom, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))):
        conf.SetAtomPosition(atom, xyz)
    cons = Constraints(coplanar=[(0, 1, 2, 3, 180.0, 10.0)])

    assert emb._structural_failure(mol, 0, cons) == Failure(
        "structural_constraint", "coplanarity C0-C1-C2-C3 is undefined", atoms=(0, 1, 2, 3)
    )


@pytest.mark.parametrize(
    ("cons", "kind", "atoms", "detail"),
    [
        (Constraints(coplanar=[(0, 1, 2, 3, None, 10.0)]), "structural_constraint", (0, 1, 2, 3), None),
    ],
    ids=["coplanarity"],
)
def test_geometry_failure_names_the_structural_measurement(cons, kind, atoms, detail):
    """An M-L distance miss reports its accepted window verbatim; other structural misses name kind and atoms."""
    mol = Chem.MolFromSmiles("[He].[He].[He].[He]")
    conf = Chem.Conformer(4)
    conf.SetPositions(np.array(((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)), dtype=float))
    mol.AddConformer(conf)
    conformers = Conformers(mol, [0], cons)

    failure = conformers._geometry_failure(0)
    if detail is not None:
        assert failure == Failure(kind, detail, atoms=atoms)
    else:
        assert failure.kind == kind
        assert failure.atoms == atoms
        assert "outside" in failure.detail


# ---------------------------------------------------------------------------------------------------------
# Conformers: the one result type
# ---------------------------------------------------------------------------------------------------------


def test_indexing_reuses_mol_in_new_conformers():
    confs = embed(_mol("CCOCC"), n=6, params=bnd.EmbedParams(seed=7, prune_rms=-1))
    sub = confs[:2]
    assert sub.ids == confs.ids[:2]
    assert sub._mol is confs._mol
    assert confs[0].ids == [confs.ids[0]]
    assert {c.GetId() for c in sub.mol.GetConformers()} == set(sub.ids)
    with pytest.raises(ValueError, match="not one of this result's ids"):
        sub.xyz(confs.ids[-1])  # an id the slice no longer tracks


# ---------------------------------------------------------------------------------------------------------
# refusals: every one is a wrong answer the guard turned into an error
# ---------------------------------------------------------------------------------------------------------


def test_embed_refuses_implicit_hydrogens():
    with pytest.raises(ValueError, match="AddHs"):
        embed(Chem.MolFromSmiles("CCO"), n=2)


def test_embed_refuses_a_source_it_would_have_to_parse():
    with pytest.raises(TypeError, match="RDKit Mol or an Isomer"):
        embed("CCO", n=2)


def test_zero_conformer_embed_warns(caplog):
    mol = _mol("C1C2CC3CC1CC(C2)C3")  # adamantane: a 1.0 A fix across the cage is not embeddable
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs = embed(mol, fix={(0, 3): 1.0}, n=1, seed=0xF00D)
    assert len(confs) == 0
    assert "no conformer" in caplog.text


# ---------------------------------------------------------------------------------------------------------
# fold_substrate: a user spec must reach the embedder whole, and must not be demoted to soft
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ideal", [30.0])
def test_substrate_rotation_preserves_umbrella_support_ownership(ideal):
    base = Constraints(umbrellas={(0, 1, 2, 3): ideal})
    key = (4, 1, 2, 5)
    sub = Constraints(dihedrals={key: (40.0, 60.0)}, contacts=(frozenset(), frozenset({key})))
    folded = fold_substrate(base, sub, {})
    assert folded.dihedrals == sub.dihedrals
    assert folded.umbrellas == base.umbrellas
    same = (1, 0, 3, 2)
    sub = Constraints(dihedrals={same: (40.0, 60.0)}, contacts=(frozenset(), frozenset({same})))
    if ideal == 0.0:
        assert fold_substrate(base, sub, {}).dihedrals == sub.dihedrals
    else:
        with pytest.raises(ValueError, match="soft bias cannot replace"):
            fold_substrate(base, sub, {})


def test_ammonium_acetate_stays_within_contact_range():
    """A simple salt's two free ions embed together, not scattered apart.

    Every fragment repels every other one at its van der Waals floor (`bnd._cap_fragment_contacts`); nothing
    picks a contact atom, so the ions can settle anywhere within that shared range, never beyond it.
    """
    mol = _mol("CC(=O)[O-].[NH4+]")
    confs = embed(mol, n=6, seed=42)
    assert confs.ids, "the salt must actually embed for this test to say anything"
    pt = Chem.GetPeriodicTable()
    frags = Chem.GetMolFrags(confs.mol)
    assert len(frags) == 2, "acetate and ammonium stay their own fragments; nothing bonds them"
    for cid in confs.ids:
        pos = confs.mol.GetConformer(cid).GetPositions()
        for i in [a for a in frags[0] if confs.mol.GetAtomWithIdx(a).GetAtomicNum() > 1]:
            for j in [a for a in frags[1] if confs.mol.GetAtomWithIdx(a).GetAtomicNum() > 1]:
                d = float(np.linalg.norm(pos[i] - pos[j]))
                vdw = pt.GetRvdw(confs.mol.GetAtomWithIdx(i).GetAtomicNum()) + pt.GetRvdw(
                    confs.mol.GetAtomWithIdx(j).GetAtomicNum()
                )
                assert d >= vdw - 0.05, f"{i}-{j} sits inside its van der Waals floor"
                assert d < 15.0, f"{i}-{j} drifted out of contact range ({d:.1f} A)"


# ---------------------------------------------------------------------------------------------------------
# graft_frozen; distance geometry only approximates a rigid core; the graft restores it exactly
# ---------------------------------------------------------------------------------------------------------


def test_core_too_small_to_orient_slides_or_stands_still():
    mol = _with_geometry("CCCl")
    before = mol.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(mol, [0], [1, 2], np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]))

    pos = mol.GetConformer(0).GetPositions()
    assert np.allclose(pos[1], before[1]), "the anchor atom moved"
    assert np.linalg.norm(pos[2] - pos[1]) == pytest.approx(2.4, abs=1e-9)
    axis_before, axis_after = before[2] - before[1], pos[2] - pos[1]
    cos = axis_after @ axis_before / (np.linalg.norm(axis_after) * np.linalg.norm(axis_before))
    assert cos == pytest.approx(1.0, abs=1e-9), "the embedded axis was rotated, not just rescaled"

    one = _with_geometry("CCCl")
    before = one.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(one, [0], [1], np.array([[9.0, 9.0, 9.0]]))
    assert np.allclose(one.GetConformer(0).GetPositions(), before)


# ---------------------------------------------------------------------------------------------------------
# minimize: the search-free companion verb
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lengths", ["input"])
def test_minimize_uses_model_distances_unless_input_requested(lengths):
    mol = _with_geometry("Cl[Pd](Cl)(N)N")
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS)
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(pd).GetNeighbors()]
    before = [GetBondLength(mol.GetConformer(0), pd, d) for d in donors]

    target = from_geometry(mol, lengths=lengths)
    confs = minimize(mol if lengths == "model" else target)
    conf = confs.mol.GetConformer(confs.ids[0])
    after = [GetBondLength(conf, pd, d) for d in donors]
    if lengths == "input":
        assert np.allclose(before, after, atol=0.15)
    else:
        assert not np.allclose(before, after, atol=0.15)
    for donor, distance in zip(donors, after, strict=True):
        lo, hi = target.cons.distances[tuple(sorted((pd, donor)))]
        assert lo - 0.15 <= distance <= hi + 0.15


# ---------------------------------------------------------------------------------------------------------
# the stiffness ladder: the front door may not hand back a torn geometry wearing a plausible energy
# ---------------------------------------------------------------------------------------------------------


def test_high_force_candidate_and_stable_coordination_use_kind_not_message_wording():
    reworded_ml = Failure("ml_distance", "a completely reworded M-L overshoot message")
    assert emb._high_force_candidate(reworded_ml)
    assert not emb._high_force_candidate(Failure("structural_constraint", "M-L distance (0, 1): 2.2 A"))

    reworded_shape = Failure("coordination_shape", "a completely reworded shape mismatch message")
    assert emb._stable_coordination_failure(reworded_shape)
    assert not emb._stable_coordination_failure(Failure("metal_state", "coordination state at M0 is nonplanar"))


@pytest.mark.parametrize(
    ("requested", "recover", "planar"),
    [(True, False, False)],
)
def test_relax_ladder_checks_requested_haptic_face_before_accepting(monkeypatch, requested, recover, planar):
    smiles = "C[CH]1=[CH]2[CH]3=[CH2]->[Fe]<-3<-2<-1(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]"
    iso = enumerate_isomers(Chem.AddHs(parse_smiles(smiles)), "tetrahedral")[0]
    confs = embed(iso, n=1, seed=42, threads=1)
    cid = confs.ids[0]
    seed = confs._mol.GetConformer(cid).GetPositions().copy()
    assert confs._metal_states()[cid]
    if not requested:
        state = state_with_winding(iso.centres[0], iso.vertices, {})
        confs.iso = iso.with_stereo((state,))
        assert not emb._stereo_targets(confs.iso)
    attempts, faces = [], []

    def flip_then_recover(mol, _cons, *, stiffness, max_iters, conf_ids, record=None, **_kw):
        if max_iters:
            attempts.append(stiffness)
            assert np.array_equal(mol.GetConformer(cid).GetPositions(), seed)
            pos = seed.copy()
            if len(attempts) == 1 or not recover:
                if planar:
                    pos[:, 2] = 0.0  # metal in the face plane: no observed winding, not an opposite sign
                else:
                    pos[:, 0] *= -1
            else:
                pos[:, 0] += 0.1
            mol.GetConformer(cid).SetPositions(pos)
            assert record is not None
            assert record.snapshots is not None
            record.statuses[cid] = 0
            record.snapshots[cid] = [pos.copy()]
            faces.append(confs._metal_states()[cid])
        return np.zeros(len(conf_ids))

    monkeypatch.setattr(emb, "restrained_uff", flip_then_recover)
    monkeypatch.setattr(Conformers, "_geometry_failure", lambda *_args: None)
    frames = []
    confs._relax_constrained(BASE_STIFFNESS, max_iters=10, _frames=frames)

    if requested and recover:
        assert faces == [False, True]
        assert not confs.unrelaxed
        assert len(frames) == 2
        np.testing.assert_allclose(frames[-1], seed + np.array([0.1, 0.0, 0.0]))
    elif requested:
        assert len(attempts) > 1
        assert not any(faces)
        assert confs.unrelaxed == [cid]
        assert len(frames) == 1
        np.testing.assert_array_equal(confs._mol.GetConformer(cid).GetPositions(), seed)
    else:
        assert len(attempts) == 1
        assert not confs.unrelaxed
    assert not confs._acceptance_failures()


def test_unavailable_uff_clears_stale_energies_and_keeps_the_seed():
    """A genuinely untypeable network (a connected boron chain UFF rejects outright) clears stale energies."""
    mol = Chem.MolFromSmiles("C=[B]B")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (1.0, 3.0)}), energies={0: -99.0})
    before = mol.GetConformer().GetPositions().copy()

    assert confs._relax_constrained(BASE_STIFFNESS, max_iters=10) is False

    assert confs.energies == {}
    assert confs.unrelaxed == [0]
    assert np.array_equal(mol.GetConformer().GetPositions(), before)


def test_constrained_embed_returns_intact_or_seed():
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1)
    before = list(confs.ids)
    confs.minimize()
    assert confs.ids == before
    assert set(confs.energies) == {int(c) for c in confs.ids}
    assert all(np.isfinite(v) for v in confs.energies.values())
    for cid in confs.ids:
        assert bonding_failure(confs.mol, int(cid), constrained=confs.cons.distances) is None, (
            f"conformer {cid} came back torn"
        )


def test_relax_failure_keeps_the_first_physical_reason_when_donor_hand_also_changes(monkeypatch):
    iso = enumerate_isomers(Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    cid, donor = confs.ids[0], 4
    references = [iso.metal]
    hand = donor_chirality_sign(confs._mol, cid, donor, references)
    monkeypatch.setattr(
        Conformers,
        "_geometry_failure",
        lambda _self, _cid: Failure("bonding", "heavy-atom bonding/clash failure"),
    )
    monkeypatch.setattr(
        emb,
        "donor_chirality_sign",
        lambda *_args, **_kwargs: "S" if hand != "S" else "R",
    )

    failures = confs._relax_failures({cid: {donor: (hand, references)}})

    assert next(iter(failures.values())) == Failure("bonding", "heavy-atom bonding/clash failure")


# ---------------------------------------------------------------------------------------------------------
# the metal-centre handedness gate
# ---------------------------------------------------------------------------------------------------------


def _hands(confs):
    """The metal-centre hand each returned conformer actually realises, read back from its coordinates."""
    mol = confs.mol
    return [from_geometry(Chem.Mol(mol, False, int(c))).chirality for c in confs.ids]


def _mirrored_input(smiles):
    """An `Isomer` stating one hand over a conformer of the other: the delta seating on a reflected geometry."""
    delta = next(i for i in enumerate_isomers(_mol(smiles), "octahedral") if i.chirality == "delta")
    confs = embed(delta, n=1, seed=0xF00D).minimize()
    mol = confs.mol
    conf = mol.GetConformer(int(confs.ids[0]))
    pos = conf.GetPositions()
    pos[:, 0] *= -1.0  # the enantiomer, by hand: nothing in the constraint set can tell the two apart
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, xyz.tolist())
    got = from_geometry(Chem.Mol(mol, False, int(confs.ids[0]))).chirality
    assert got == "lambda", f"the premise: this geometry must be the mirror of the stated hand, got {got!r}"
    return Isomer(mol, "octahedral", dict(enumerate(delta.vertices)))


def test_minimize_preserves_isomer_hand():
    confs = minimize(_mirrored_input(_CO_EN))
    assert _hands(confs) == ["delta"]


def test_minimize_reflects_a_mirrored_pyramidal_zinc_back_to_its_hand():
    """[Zn(NH3)ClBr] on a tetrahedron with one empty vertex is chiral only through that vertex, so the relax
    must read the mirrored input's hand in the requested tetrahedron before it can reflect it back.
    """
    delta = next(i for i in enumerate_isomers(_mol("N->[Zn+2](<-[Cl-])<-[Br-]"), "TET") if i.chirality == "delta")
    confs = embed(delta, n=1, seed=0xF00D).minimize()
    mol = Chem.Mol(confs.mol, False, int(confs.ids[0]))
    pos = mol.GetConformer().GetPositions()
    pos[:, 0] *= -1.0
    mol.GetConformer().SetPositions(pos)
    stated = Isomer(mol, "tetrahedral", {slot: atom for slot, atom in enumerate(delta.vertices) if atom != VACANT})
    assert emb._realised_hand(mol, stated, mol.GetConformer().GetId()) == "lambda"

    assert cxsmiles(minimize(stated).mol) == cxsmiles(delta)


@pytest.mark.parametrize(
    ("request_kind", "kind", "remedy"),
    [
        ("fixed molecule", "bonding", "try a looser fix="),
    ],
)
def test_embedding_error_offers_only_remedies_the_request_can_use(monkeypatch, request_kind, kind, remedy):
    if request_kind.endswith("molecule"):
        cons = Constraints(frozen={0}) if request_kind == "fixed molecule" else Constraints()
        confs = Conformers(_with_geometry("CC"), [0], cons)
    else:
        iso = _isomer()
        if request_kind == "geometry isomer":
            iso = from_geometry(embed(iso, n=1, seed=1).minimize().mol)
        confs = embed(iso, n=1, seed=1)
        confs.params = None  # no seed parameters: no fresh-seed replacement
    monkeypatch.setattr(
        Conformers, "_acceptance_failures", lambda self, *_args, **_kw: {Failure(kind, "synthetic"): list(self.ids)}
    )

    with pytest.raises(emb.EmbeddingError) as caught:
        confs._accept_relaxed(BASE_STIFFNESS, 1, operation="embed", raise_if_empty=True)
    assert str(caught.value).endswith(f": synthetic in 1/1 rejected seeds; {remedy}")


@pytest.mark.parametrize(("shell", "accepted"), [("square_pyramidal", True), ("square_planar", False)])
def test_square_pyramid_with_an_empty_apex_is_judged_by_its_shape_not_a_flat_tolerance(shell, accepted):
    """[NiCl4]2- seated on a square pyramid's base: its ideal base lies 0.10 r from a plane, inside
    `COPLANAR_TOL` at Ni-Cl, so only the shape reading can tell it from a flat square.
    """
    iso = next(i for i in enumerate_isomers(_mol("Cl[Ni](Cl)(Cl)Cl"), "square_pyramidal") if i.vertices[0] == VACANT)
    conformers = embed(iso, n=1, seed=7)
    conf = conformers._mol.GetConformer(conformers.ids[0])
    pos = conf.GetPositions()
    base = [iso.vertices[slot] for slot in range(1, 5)]
    for donor, direction in zip(base, POLYHEDRA[shell].vertex_dirs[-4:], strict=True):
        radius = np.linalg.norm(pos[donor] - pos[iso.metal])
        conf.SetAtomPosition(donor, tuple(pos[iso.metal] + radius * np.asarray(direction, float)))

    failure = emb._coordination_state_failure(conformers._mol, conformers.ids[0], iso)

    assert (failure is None) == accepted, failure


def test_relax_gate_rejects_inverted_ligand_point_stereo(monkeypatch):
    """A relaxed ligand point stereocentre that reads inverted is rejected, with a remedy the caller can use."""
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    conformers = embed(isomer, n=1, seed=7)
    atom, expected = next(iter(point_stereo(isomer.stereo_label).items()))
    found = {"R": "S", "S": "R"}[expected]
    symbol = isomer.mol.GetAtomWithIdx(atom).GetSymbol()

    def inverted(published, **_kwargs):
        assert published.GetBondBetweenAtoms(*isomer.donor_bonds[0]).GetBondType() == Chem.BondType.DATIVE
        return f"{symbol}{atom}:{found}"

    monkeypatch.setattr(emb, "stereo_from_3d", inverted)
    monkeypatch.setattr(emb, "_coordination_state_failure", lambda *_args, **_kwargs: None)

    failure = conformers._geometry_failure(conformers.ids[0])

    assert failure.kind == "ligand_stereo"
    assert emb.remedy(failure.kind, isomer, conformers.cons) == "try another isomer"


def test_relax_gate_distinguishes_unassigned_ligand_stereo(monkeypatch):
    isomer = _isomer("C[C@H](F)C[NH2]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]")
    monkeypatch.setattr(emb, "stereo_from_3d", lambda *_args, **_kwargs: "")

    failure = emb._ligand_stereo_failure(isomer.mol, isomer)

    assert failure.kind == "ligand_stereo_unassigned"
    assert Conformers(_with_geometry("CC"), [0], Constraints())._required_failure({failure: [0]}) == {
        "ligand_stereo_unassigned"
    }


def test_relax_gate_rejects_reflected_ligand_axial_stereo():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|"))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
    wanted = stereo_from_3d(mol)
    request = SimpleNamespace(stereo_label=wanted)
    pair, expected = next(iter(axis_stereo(wanted).items()))
    positions = mol.GetConformer().GetPositions()
    positions[:, 0] *= -1
    mol.GetConformer().SetPositions(positions)
    found = axis_stereo(stereo_from_3d(mol))[pair]

    failure = emb._ligand_stereo_failure(mol, request)
    a, b = sorted(pair)
    names = f"{mol.GetAtomWithIdx(a).GetSymbol()}{a}-{mol.GetAtomWithIdx(b).GetSymbol()}{b}"
    assert str(failure) == f"axis {names} reads {found}, not {expected}"
    assert Conformers(mol, [0], Constraints())._required_failure({failure: [0]}) == {"ligand_stereo"}


@pytest.mark.parametrize("raise_if_empty", [True])
@pytest.mark.parametrize("survivor", [False])
def test_total_embed_rejection_reports_original_bond_failure(raise_if_empty, survivor):
    mol = _with_geometry("CC")
    if survivor:
        mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    SetBondLength(mol.GetConformer(0), 0, 1, 10.0)
    conformers = Conformers(mol, [conf.GetId() for conf in mol.GetConformers()])

    if raise_if_empty and not survivor:
        with pytest.raises(emb.EmbeddingError, match=r"embed: bond C0-C1 stretched to 10.00 A in 1/1 rejected seeds"):
            conformers._accept_relaxed(BASE_STIFFNESS, 1, operation="embed", raise_if_empty=True)
    else:
        failures = conformers._accept_relaxed(BASE_STIFFNESS, 1, operation="embed", raise_if_empty=raise_if_empty)
        assert {failure.kind for failure in failures} == {"bonding"}
    assert conformers.ids == ([1] if survivor else [])


@pytest.mark.parametrize("count", [7])
def test_replacements_rank_valid_settled_scores_without_touching_survivors(monkeypatch, count):
    mol = Chem.MolFromSmiles("[He]")
    for cid in range(8):
        conf = Chem.Conformer(1)
        conf.SetId(cid)
        conf.SetAtomPosition(0, (float(cid), 0.0, 0.0))
        mol.AddConformer(conf, assignId=False)
    target = Chem.Mol(mol)
    for conf in target.GetConformers():
        conf.SetId(conf.GetId() + 20)
    failed = list(range(20, 20 + count))
    owner = Conformers(Chem.Mol(target), [*failed, 27], params=bnd.EmbedParams(seed=42), energies={27: -77.0})
    calls = []
    marker = object()

    def seeds(*_args, **_kwargs):
        calls.append(True)
        return Chem.Mol(mol), list(range(8)), None

    def relax(batch, *_args):
        batch.energies = {0: -100.0, 1: 5.0, 2: 1.0, 3: 1.0, 4: -1000.0, 6: -np.inf, 7: np.nan}
        batch.unrelaxed = [4]

    def accept(_batch, operation, validator=None):
        assert operation == "embed"
        assert validator is marker
        return {Failure("physical_geometry", "invalid lowest energy"): [0]}

    monkeypatch.setattr(emb, "seed_conformers", seeds)
    monkeypatch.setattr(Conformers, "_relax_once", relax)
    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)
    assert owner._replace_failed(failed, BASE_STIFFNESS, 1, operation="embed", validator=marker) == []
    expected = [2, 3, 1, 4, 5, 6, 7][:count]
    assert [int(owner._mol.GetConformer(cid).GetAtomPosition(0).x) for cid in failed] == expected
    assert calls == [True]
    assert owner.ids == [*failed, 27]
    assert owner.energies[20] == 1.0
    assert owner.energies[27] == -77.0
    assert owner.unrelaxed == ([23] if count == 7 else [])
    if count == 7:
        assert 24 not in owner.energies
    np.testing.assert_array_equal(owner._mol.GetConformer(27).GetPositions(), target.GetConformer(27).GetPositions())


# ---------------------------------------------------------------------------------------------------------
# replacement's two stop rules: a single crossed seating, or the same (kind, atoms) twice in a row
# ---------------------------------------------------------------------------------------------------------


_REPLACEMENT_VICTIM = 5  # an arbitrary id distinct from the fresh batch's own conformer id (0)


def _replacement_owner(monkeypatch, *, accept, relax_failures=None, ladder=None):
    """Build a one-atom `Conformers` whose fresh-seed batches each seed exactly conformer 0.

    ``accept`` stands in for `_acceptance_failures`; every batch relaxes as a no-op that records ``ladder``,
    when given, as its seed's rejected UFF endpoint reason.
    """
    victim_mol = Chem.MolFromSmiles("[He]")
    victim_conf = Chem.Conformer(1)
    victim_conf.SetId(_REPLACEMENT_VICTIM)
    victim_mol.AddConformer(victim_conf, assignId=False)
    owner = Conformers(
        victim_mol,
        [_REPLACEMENT_VICTIM],
        params=bnd.EmbedParams(seed=1),
        relax_failures=relax_failures or {},
    )
    batch_mol = Chem.MolFromSmiles("[He]")
    batch_mol.AddConformer(Chem.Conformer(1), assignId=False)  # id 0: a fresh batch's own seed
    calls = []

    def seeds(*_args, **_kwargs):
        calls.append(True)
        return Chem.Mol(batch_mol), [0], None

    monkeypatch.setattr(emb, "seed_conformers", seeds)

    def relax(batch, *_args, **_kwargs):
        if ladder is not None:
            batch.relax_failures = dict.fromkeys(batch.ids, ladder)

    monkeypatch.setattr(Conformers, "_relax_once", relax)
    monkeypatch.setattr(Conformers, "_relax_constrained", relax)
    monkeypatch.setattr(Conformers, "_acceptance_failures", accept)
    return owner, _REPLACEMENT_VICTIM, calls


def test_replacement_stops_after_two_batches_tear_the_same_ligand_bond(monkeypatch):
    tear = Failure("bonding", "heavy-atom bonding/clash failure (bond 3-6 2.500 A above 1.690 A)", atoms=(3, 6))
    owner, victim, calls = _replacement_owner(monkeypatch, accept=lambda self, *_a, **_k: {tear: list(self.ids)})

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 2, "a third fresh batch cannot tear the same bond differently"
    assert result == [victim]


def test_replacement_stops_when_every_seed_crosses_to_another_seating(monkeypatch):
    """A seed that crosses its own donor-slot seating stops the replacement search after one batch."""
    wrong_shape = Failure("coordination_shape", "coordination state at M0 is nonplanar", atoms=(0,))
    crossed = Failure("seating_crossed", "coordination state at M0 crossed its donor-slot seating", atoms=(0,))
    owner, victim, calls = _replacement_owner(
        monkeypatch,
        accept=lambda self, *_a, **_k: {crossed: list(self.ids)},
        relax_failures={_REPLACEMENT_VICTIM: wrong_shape},  # this endpoint can reach a wrong labelled shape
        ladder=wrong_shape,  # the seed's UFF endpoint read a wrong shape; its restored seed crossed seating
    )

    result = owner._replace_failed([victim], BASE_STIFFNESS, 1)

    assert len(calls) == 1, "a crossed seating is a different arrangement, not worth a further batch"
    assert result == [victim]


@pytest.mark.parametrize("n", [8])
def test_stereo_seed_selection_prefers_a_seed_with_the_requested_geometry(monkeypatch, n):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality)
    mol, cons, prepared, graft = emb.prepare(iso)

    def seeds(candidate, _cons, n, _params, **_kwargs):
        candidate.RemoveAllConformers()
        for marker in range(n):
            conf = Chem.Conformer(candidate.GetNumAtoms())
            conf.SetAtomPosition(0, (float(marker), 0.0, 0.0))
            candidate.AddConformer(conf, assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(emb, "_seed_geometry_matches", lambda _mol, cid, _iso: cid == 2)

    seeded, ids, target = emb.seed_conformers(mol, cons, prepared, n, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert len(ids) == target == n
    assert len({seeded.GetConformer(cid).GetAtomPosition(0).x for cid in ids}) == n
    assert seeded.GetConformer(ids[0]).GetAtomPosition(0).x == 2.0


def test_terminal_carbonyls_seed_linear_and_relax_inside_the_linear_window():
    """ETKDG keeps M-C-O straight only if it sees the M-C bond; the relax then stays in the centred sp window."""
    iso = rx.metal("[O+]#[C-]->[Mo](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-P(C)(C)C", "octahedral")[0]
    carbonyls = [
        (c, o.GetIdx()) for c in iso.donors for o in iso.mol.GetAtomWithIdx(c).GetNeighbors() if o.GetAtomicNum() == 8
    ]
    assert len(carbonyls) == 5

    def bend(mol, cid):
        return min(GetAngleDeg(mol.GetConformer(cid), iso.metal, c, o) for c, o in carbonyls)

    for seed in range(5):
        raw = embed(iso, n=1, seed=seed)
        relaxed = rx.embed(iso, n=1, seed=seed, threads=1)
        assert bend(raw._mol, raw.ids[0]) >= 178.0, f"seed {seed}: ETKDG bent a carbonyl it should hold linear"
        assert bend(relaxed.mol, relaxed.ids[0]) >= 173.0, f"seed {seed}: the relax bent a carbonyl below 174 deg"


@pytest.mark.parametrize("kind", ["ez"])
def test_nonpoint_ligand_stereo_does_not_enable_the_point_fallback(monkeypatch, kind):
    """A non-point (E/Z or axial) stereo target must never trigger the relaxed-chirality point fallback."""
    iso = _isomer("CC=[NH]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Br-]", "square_planar")
    mol, cons, prepared, graft = emb.prepare(iso)
    calls = []
    if kind == "axial":
        monkeypatch.setattr(emb, "bond_stereo", lambda _label: {})
        monkeypatch.setattr(emb, "axis_stereo", lambda _label: {(0, 1): "Ra"})
    else:
        monkeypatch.setattr(emb, "axis_stereo", lambda _label: {})

    def seeds(candidate, _cons, n, _params, *, enforce_chirality=True, **kwargs):
        calls.append((enforce_chirality, kwargs["max_attempts"]))
        candidate.RemoveAllConformers()
        for _ in range(n):
            candidate.AddConformer(Chem.Conformer(candidate.GetNumAtoms()), assignId=True)
        return [conf.GetId() for conf in candidate.GetConformers()]

    monkeypatch.setattr(emb, "seed_coordinates", seeds)
    monkeypatch.setattr(emb, "_seed_stereo_matches", lambda *_args, **_kwargs: True)

    _mol_out, ids, target = emb.seed_conformers(mol, cons, prepared, 1, bnd.EmbedParams(seed=1), graft_ref=graft)

    assert len(calls) == 1, "no point-stereo fallback retry may run for a non-point stereo target"
    assert calls[0][0] is True, "the only call must use strict chirality"
    assert len(ids) == target == 1


def test_unconstrained_max_iteration_replaces_invalid_geometry_and_keeps_status(monkeypatch):
    confs = embed(_mol("CCC"), n=1, seed=1)
    owner = confs._mol

    def stall(mol, _cons, **kwargs):
        ids = kwargs.get("conf_ids") or [conf.GetId() for conf in mol.GetConformers()]
        if mol is owner:
            mol.GetConformer(ids[0]).SetAtomPosition(0, (99.0, 99.0, 99.0))
        record = kwargs.get("record")
        if record is not None:
            record.statuses.update(dict.fromkeys(ids, 1))
        return [0.0] * len(ids)

    monkeypatch.setattr(emb, "restrained_uff", stall)
    confs.minimize()

    assert confs.ids
    assert confs.unrelaxed == confs.ids
    assert all(confs._geometry_failure(cid) is None for cid in confs.ids)


def test_template_composes_with_rigid_core_forms():
    mol = _mol("CC(=O)Nc1ccccc1")
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    ref = Chem.Mol(mol)
    pos = ref.GetConformer().GetPositions()

    both = embed(mol, template=(ref, {0: 0, 1: 1}), fix=[2, 3], n=2, seed=1)
    assert sorted(both.cons.frozen) == [0, 1, 2, 3], "a list fix beside a template must not be dropped"

    mixed = embed(mol, template=(ref, {0: 0, 1: 1}), fix={2: tuple(pos[2])}, n=2, seed=1)
    assert sorted(mixed.cons.frozen) == [0, 1, 2]

    alone = embed(mol, template=(ref, {i: i for i in (0, 1, 2, 3)}), n=2, seed=1)
    assert sorted(alone.cons.frozen) == [0, 1, 2, 3]
    assert alone.cons.distances.keys() == embed(mol, fix=[0, 1, 2, 3], n=2, seed=1).cons.distances.keys(), (
        "a template and an own-coords fix over the same atoms must build the same rigid body"
    )
