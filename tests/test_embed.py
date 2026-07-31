"""`embed.py`, the front door: ``embed(spec, fix=, constrain=) -> Conformers``, and ``minimize`` beside it.

The seam that stacks `bounds.embed` and `relax.restrained_uff`: encounter bounds, the donor-chirality hold, the
embed, the Kabsch graft, then the stiffness ladder inside `Conformers.minimize`.

Every refusal here exists because the un-guarded call returned a plausible, wrong answer rather than raising;
a heavy-atom-only geometry, an un-surrogated metal UFF silently dropped every term for, a 0-byte dump that read
as a successful write, a relax that quietly dropped the arrangement it was handed. The *spec* refusals (a SMARTS
key, an out-of-range angle, one key under both verbs) belong to the resolver and are in `test_constraints.py`.
"""

from __future__ import annotations

import importlib
import itertools
import logging
import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetBondLength

from rxembed import bounds as bnd
from rxembed.constraints import Constraints, resolve_core
from rxembed.embed import FC_ESCALATION, Conformers, embed, fold_substrate, minimize
from rxembed.metal_core import TRANSITION_METALS, coplanar
from rxembed.metal_isomers import Isomer, enumerate_isomers
from rxembed.relax import bonding_ok

emb = importlib.import_module("rxembed.embed")  # `rxembed.embed` the ATTRIBUTE is the front-door FUNCTION

_BIPY_PD = "Cl[Pd]1(Cl)<-n2ccccc2-c2ccccn->12"
_EN_PD = "Cl[Pd](Cl)(<-N(C)(C)C)<-N(C)(C)C"
# the N-bound Ni(II) isomer: the window relax tears 3 of 8 seeds at the base stiffness, the ladder's type case
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _with_geometry(smiles, seed=7):
    mol = _mol(smiles)
    rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _isomer(smiles=_BIPY_PD, geometry="square_planar"):
    """The first enumerated coordination isomer of `smiles`: the metal spec `embed` accepts."""
    return next(iter(enumerate_isomers(_mol(smiles), geometry)))


def _fold(iso, **spec):
    """Fold a user `fix`/`constrain` onto the isomer's polyhedron, exactly as `embed(iso, **spec)` does."""
    sub, graft_ref = resolve_core(iso.mol, **spec, has_geometry=iso.mol.GetNumConformers() > 0)
    return fold_substrate(iso.coordination().copy(), sub, graft_ref)


def _sphere_key(iso, which=0):
    """The (i, j) distance key of one M-donor hold, in `add_distance`'s sorted order."""
    return (min(iso.metal, iso.donors[which]), max(iso.metal, iso.donors[which]))


def _distance(mol, cid, i, j):
    p = mol.GetConformer(cid).GetPositions()
    return float(np.linalg.norm(p[i] - p[j]))


# ---------------------------------------------------------------------------------------------------------
# the everyday spine
# ---------------------------------------------------------------------------------------------------------


def test_embed_holds_a_numbers_fix_and_chains_to_minimize_and_dump(tmp_path):
    """embed(mol, fix={(i, j): d}).minimize().dump(path): the fix lands and the file is real."""
    mol = _mol("OCCCN")
    confs = embed(mol, fix={(0, 4): 3.0}, n=8, seed=0xF00D)
    assert len(confs) == len(confs.ids) > 0
    confs.minimize()
    for cid in confs.ids:
        assert _distance(confs._mol, cid, 0, 4) == pytest.approx(3.0, abs=0.1)
    path = confs.dump(tmp_path / "out.xyz")
    assert path.read_text().count(f"{mol.GetNumAtoms()}\n") == len(confs), "one xyz frame per tracked conformer"


def test_the_input_molecule_is_never_touched():
    """`embed` works on its own copy, so a caller's conformers survive the call."""
    mol = _with_geometry("CCO")
    before = mol.GetConformer(0).GetPositions().copy()
    confs = embed(mol, n=3, seed=1)
    assert confs._mol is not mol
    assert mol.GetNumConformers() == 1
    assert np.allclose(mol.GetConformer(0).GetPositions(), before)


# ---------------------------------------------------------------------------------------------------------
# Conformers: the one result type
# ---------------------------------------------------------------------------------------------------------


def test_indexing_returns_a_new_conformers_over_the_same_mol():
    """`confs[0]` / `confs[:2]` select by POSITION and share the Mol; `xyz` still keys by conformer id."""
    confs = embed(_mol("CCOCC"), n=6, seed=7)
    sub = confs[:2]
    assert sub.ids == confs.ids[:2]
    assert sub._mol is confs._mol
    assert confs[0].ids == [confs.ids[0]]
    with pytest.raises(ValueError, match="not one of this result's ids"):
        sub.xyz(confs.ids[-1])  # an id the slice no longer tracks


def test_mol_is_a_usable_rdkit_molecule_and_always_a_copy():
    """`.mol` is a plain `Chem.Mol` an RDKit call can take directly, and never the working molecule."""
    confs = embed(_mol("CCO"), n=3, seed=1).minimize()
    out = confs.mol

    assert out is not confs._mol
    assert isinstance(out, Chem.Mol)
    assert out.GetNumConformers() == len(confs)
    assert {int(c.GetId()) for c in out.GetConformers()} == {int(i) for i in confs.ids}
    assert Chem.MolToXYZBlock(out, confId=int(confs.ids[0])).count("\n") == out.GetNumAtoms() + 2


def test_dump_refuses_a_result_with_no_conformers(tmp_path):
    """A 0-byte file that reads as a successful write is the worst possible outcome."""
    with pytest.raises(ValueError, match="nothing to dump"):
        Conformers(_mol("CCO"), []).dump(tmp_path / "empty.xyz")


def test_the_surrogate_stays_internal_while_mol_hands_back_the_real_metal():
    """`_mol` keeps the bond-less surrogate the engine needs; `.mol` gives the real element and dative bonds.

    The engine cannot work on the real element (UFF types Pd and not Ir, so the force field would depend on
    which metal you have) and a consumer cannot use the surrogate, so the restore happens on the way OUT.
    """
    src = _mol(_EN_PD)
    iso = Isomer(src, "SPL", {0: 0, 1: 2, 2: 3, 3: 7})
    confs = embed(iso, n=2, seed=0xF00D).minimize()

    internal = confs._mol.GetAtomWithIdx(iso.metal)
    assert internal.GetAtomicNum() not in TRANSITION_METALS, "the working mol must still hold the surrogate"
    assert internal.GetDegree() == 0, "the surrogate must stay bond-less: a bonded Li makes UFF singular"

    restored = confs.mol.GetAtomWithIdx(iso.metal)
    assert restored.GetAtomicNum() in TRANSITION_METALS
    assert restored.GetFormalCharge() == src.GetAtomWithIdx(iso.metal).GetFormalCharge()
    assert len(Chem.GetMolFrags(confs.mol)) == 1, "the M-donor bonds are re-added, so the complex is one fragment"


# ---------------------------------------------------------------------------------------------------------
# refusals: every one is a wrong answer the guard turned into an error
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "match"), [({"seed": -1}, "not reproducible"), ({"n": 0}, "positive conformer")])
def test_embed_refuses_an_argument_that_would_be_silently_wrong(kwargs, match):
    """seed=-1 draws from the global RNG (non-reproducible); n<=0 meant "auto" rather than "none"."""
    with pytest.raises(ValueError, match=match):
        embed(_mol("CCO"), **kwargs)


def test_embed_refuses_implicit_hydrogens():
    """The documented precondition is enforced: a heavy-atom-only Mol embedded happily and looked fine."""
    with pytest.raises(ValueError, match="AddHs"):
        embed(Chem.MolFromSmiles("CCO"), n=2)


def test_embed_refuses_a_bare_metal_mol():
    """UFF cannot type a metal: without the surrogate every metal FF term is dropped, silently."""
    with pytest.raises(ValueError, match="Isomer"):
        embed(_mol(_EN_PD), n=2)


def test_embed_refuses_a_source_it_would_have_to_parse():
    """Perception is the caller's job at this door: a SMILES string is a different verb (`pipeline.embed`)."""
    with pytest.raises(TypeError, match="RDKit Mol or an Isomer"):
        embed("CCO", n=2)


def test_a_zero_conformer_embed_warns_rather_than_reading_as_pruned(caplog):
    """An unrealisable spec must say so; silence here reads as "embedded, then pruned"."""
    mol = _mol("C1C2CC3CC1CC(C2)C3")  # adamantane: a 1.0 A fix across the cage is not embeddable
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs = embed(mol, fix={(0, 3): 1.0}, n=1, seed=0xF00D)
    assert len(confs) == 0
    assert "no conformer" in caplog.text


# ---------------------------------------------------------------------------------------------------------
# fold_substrate: a user spec must reach the embedder whole, and must not be demoted to soft
# ---------------------------------------------------------------------------------------------------------


def test_a_pi_stack_constrain_survives_the_fold():
    """A hand-listed merge dropped `sub.planes`; accepted, index-validated, logged, then discarded."""
    iso = _isomer()
    a, b = (tuple(r) for r in iso.mol.GetRingInfo().AtomRings() if len(r) == 6)
    cons = _fold(iso, constrain={(a, b): 3.6})
    assert any(set(pa) == set(a) and set(pb) == set(b) for pa, pb, _sep in cons.planes)


def test_a_fix_landing_on_a_sphere_hold_overrides_it():
    """Last-wins on distances: the user's own resolved window survives the fold, unclipped by the sphere's."""
    iso = _isomer()
    key = _sphere_key(iso)
    alone, _ref = resolve_core(iso.mol, fix={key: 2.42}, has_geometry=False)
    assert alone.distances[key] != iso.coordination().distances[key], "premise: the two windows must differ"
    assert _fold(iso, fix={key: 2.42}).distances[key] == alone.distances[key], "the sphere hold clipped the fix"


def test_a_constrain_landing_on_a_sphere_hold_is_not_demoted_to_releasable():
    """`constrain=` is normally releasable, but on a sphere hold `mc(explore=)` could dissociate the sphere.

    Provenance is subtractive, not the union `compose` would give: the key keeps the window and loses the
    release. `fix` cannot show this: it is never releasable to begin with.
    """
    iso = _isomer()
    key = _sphere_key(iso)
    cons = _fold(iso, constrain={key: (2.3, 2.5)})
    assert cons.distances[key] == pytest.approx((2.3, 2.5))
    assert key not in cons.contacts[0], "a sphere hold became releasable"
    assert key in cons.relaxed().distances, "the exploratory pass would drop the coordination sphere"


def test_a_graft_pinning_two_sphere_atoms_is_refused():
    """Two grafted sphere atoms fix their mutual placement, i.e. the arrangement, so the enumerated isomer
    would come back wearing the reference's geometry under its own label."""
    iso = _isomer(_EN_PD)
    fix = {iso.donors[0]: (0.0, 0.0, 0.0), iso.donors[1]: (2.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="pins coordination-sphere atoms"):
        _fold(iso, fix=fix)


def test_a_graft_pinning_one_sphere_atom_is_allowed():
    """One pinned vertex leaves the arrangement to the polyhedron, so it composes: the boundary of the refusal."""
    iso = _isomer(_EN_PD)
    assert iso.donors[0] in _fold(iso, fix={iso.donors[0]: (0.0, 0.0, 0.0)}).frozen


# ---------------------------------------------------------------------------------------------------------
# encounter bounds: a probe geometry decides a discrete question, so it must not read process history
# ---------------------------------------------------------------------------------------------------------


def _two_fragments():
    return _mol("OC(=O)CCCc1ccccc1.NCCCCN")


def _burn_global_rng(n=64):
    """Consume RDKit global randomness, standing in for whatever ran before us in a real session."""
    for _ in range(n):
        m = _mol("CCCCO")
        rdDistGeom.EmbedMolecule(m, rdDistGeom.ETKDGv3())  # deliberately unseeded


def test_encounter_bounds_do_not_move_when_unrelated_randomness_is_consumed():
    """The probe's own seed was unset, so a test passed alone and failed in a full suite, same wrong value."""
    mol = _two_fragments()
    assert len(Chem.GetMolFrags(mol)) >= 2, "fixture must be multi-fragment to exercise the encounter bounds"
    before = emb.encounter_bounds(mol)
    assert before, "no inter-fragment bound was produced: the fixture is not exercising the code"
    _burn_global_rng()
    assert emb.encounter_bounds(mol) == before


def test_the_probe_seed_is_threaded_through_rather_than_hard_coded(monkeypatch):
    """A constant baked past the signature would make the reproducibility guard above vacuous."""
    seen = []
    real = bnd.probe_conformer
    monkeypatch.setattr(emb, "probe_conformer", lambda m, s: (seen.append(s), real(m, s))[1])
    emb.encounter_bounds(_two_fragments(), seed=4321)
    assert seen == [4321]


def test_only_fragments_no_constraint_pins_get_a_floated_bound():
    """A pair already linked by a fix / constrain / contact is left to that constraint, not double-bounded."""
    mol = _two_fragments()
    assert emb.float_encounter_bounds(mol, Constraints()), "every pair of a free multi-fragment mol must be bounded"

    i, j = sorted(f[0] for f in Chem.GetMolFrags(mol))
    assert emb.float_encounter_bounds(mol, Constraints(distances={(i, j): (3.0, 3.5)})) == {}


def test_a_single_fragment_gets_no_encounter_bound():
    """There is no inter-fragment separation to enforce, so the whole path must be a no-op."""
    assert emb.float_encounter_bounds(_mol("CCO"), Constraints()) == {}


# ---------------------------------------------------------------------------------------------------------
# graft_frozen; distance geometry only approximates a rigid core; the graft restores it exactly
# ---------------------------------------------------------------------------------------------------------


def test_the_graft_restores_the_core_shape_exactly():
    """0.000 A internal RMSD is the guarantee a TS core needs; DG alone lands 0.2-0.4 A off."""
    mol = _with_geometry("CCCl")
    core = [0, 1, 2]
    ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.8, 0.0]])
    emb.graft_frozen(mol, [0], core, ref)

    pos = mol.GetConformer(0).GetPositions()
    for (a, b), want in (((0, 1), 1.5), ((1, 2), 1.8), ((0, 2), float(np.linalg.norm(ref[0] - ref[2])))):
        assert np.linalg.norm(pos[core[a]] - pos[core[b]]) == pytest.approx(want, abs=1e-9)


def test_a_two_atom_core_keeps_one_atom_put_and_slides_the_other():
    """A bond has no orientation to restore, so the first atom stays and the second slides along the embedded axis.

    A Kabsch fit over two points reproduces the length and the axis too, but moves both atoms about their
    centroid, which drags the frozen anchor away from the periphery that was embedded around it.
    """
    mol = _with_geometry("CCCl")
    before = mol.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(mol, [0], [1, 2], np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]))

    pos = mol.GetConformer(0).GetPositions()
    assert np.allclose(pos[1], before[1]), "the anchor atom moved"
    assert np.linalg.norm(pos[2] - pos[1]) == pytest.approx(2.4, abs=1e-9)
    axis_before, axis_after = before[2] - before[1], pos[2] - pos[1]
    cos = axis_after @ axis_before / (np.linalg.norm(axis_after) * np.linalg.norm(axis_before))
    assert cos == pytest.approx(1.0, abs=1e-9), "the embedded axis was rotated, not just rescaled"


def test_a_one_atom_core_is_left_alone():
    """A point has no shape, so the graft must not move it; nor raise on the degenerate case."""
    mol = _with_geometry("CCCl")
    before = mol.GetConformer(0).GetPositions().copy()
    emb.graft_frozen(mol, [0], [1], np.array([[9.0, 9.0, 9.0]]))
    assert np.allclose(mol.GetConformer(0).GetPositions(), before)


# ---------------------------------------------------------------------------------------------------------
# minimize: the search-free companion verb
# ---------------------------------------------------------------------------------------------------------


def test_minimize_pulls_the_input_geometry_toward_a_target_without_searching():
    """The input conformer is kept and pulled to the stated distance: no ETKDG, no extra conformers."""
    mol = _with_geometry("CCCl")
    before = GetBondLength(mol.GetConformer(0), 1, 2)
    confs = minimize(mol, fix={(1, 2): 2.4})

    assert len(confs) == mol.GetNumConformers(), "minimize must not add or drop conformers: it does not search"
    after = GetBondLength(confs.mol.GetConformer(confs.ids[0]), 1, 2)
    assert abs(after - 2.4) < 0.1, f"C-Cl was not pulled to the target: {before:.2f} -> {after:.2f}"
    assert GetBondLength(mol.GetConformer(0), 1, 2) == pytest.approx(before), "the caller's geometry was relaxed"


def test_minimize_refuses_a_graph_with_no_geometry():
    """A bare graph has nothing to relax; saying so beats embedding one silently and relaxing that."""
    with pytest.raises(ValueError, match="existing geometry"):
        minimize(_mol("CCO"))


def test_minimize_relaxes_an_isomer_under_its_own_coordination_polyhedron():
    """An `Isomer` states an arrangement, so the search-free verb must relax UNDER it, as `embed(iso)` does.

    It used to take `spec.mol` and drop the isomer: the relax then ran on a bond-less surrogate with no
    coordination constraint at all: the sphere drifted to ~4 Å, `.mol` handed back a carbon where the metal
    was, and the complex came apart into fragments, all with a plausible energy.
    """
    src = _with_geometry(_EN_PD)
    iso = Isomer(src, "SPL", {0: 0, 1: 2, 2: 3, 3: 7})
    before = iso.mol.GetConformer().GetPositions().copy()
    assert not coplanar(before, iso.metal, iso.donors), "the premise: this input does NOT realise the square plane"
    confs = minimize(iso)

    assert confs.iso is iso, "the isomer must be carried, or nothing downstream can restore the metal"
    assert set(confs.cons.distances) >= set(iso.cons.distances), "the M-donor windows never reached the relax"
    assert set(confs.cons.angles) == set(iso.cons.angles), "the polyhedron angles never reached the relax"

    pos = confs._mol.GetConformer(confs.ids[0]).GetPositions()
    for (i, j), (lo, hi) in iso.cons.distances.items():  # the isomer's OWN windows, not a tabulated length
        d = float(np.linalg.norm(pos[i] - pos[j]))
        assert lo - 0.15 <= d <= hi + 0.15, f"sphere not held: d({i},{j}) = {d:.2f}, window ({lo:.2f}, {hi:.2f})"
    assert coplanar(pos, iso.metal, iso.donors), "the relax was pulled toward the declared square plane"

    assert confs.mol.GetAtomWithIdx(iso.metal).GetAtomicNum() in TRANSITION_METALS
    assert len(Chem.GetMolFrags(confs.mol)) == 1, "the M-donor bonds are re-added, so the complex is one fragment"
    assert confs._mol is not iso.mol, "the relax works on a copy: the caller keeps its isomer"
    assert np.allclose(iso.mol.GetConformer().GetPositions(), before)


def test_minimize_holds_a_metal_sphere_at_the_input_geometry():
    """A relax aimed at an organic target must not rearrange a coordination sphere it was not asked about.

    The bond-less surrogate carries no M-L terms of its own, which is why `prepare_relax` re-states the sphere.
    """
    mol = _with_geometry("Cl[Pd](Cl)(N)N")
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS)
    donors = [n.GetIdx() for n in mol.GetAtomWithIdx(pd).GetNeighbors()]
    before = [GetBondLength(mol.GetConformer(0), pd, d) for d in donors]

    confs = minimize(mol)
    conf = confs.mol.GetConformer(confs.ids[0])
    after = [GetBondLength(conf, pd, d) for d in donors]
    assert np.allclose(before, after, atol=0.15), f"the sphere moved: {before} -> {after}"


# ---------------------------------------------------------------------------------------------------------
# the stiffness ladder: the front door may not hand back a torn geometry wearing a plausible energy
# ---------------------------------------------------------------------------------------------------------


def test_the_ladder_starts_at_the_callers_own_force_constant_and_ascends():
    """Rung 0 must be the caller's fc, or an unconstrained caller's `distance_fc` is silently multiplied."""
    assert list(FC_ESCALATION) == sorted(FC_ESCALATION), "the ladder must ascend"
    assert FC_ESCALATION[0] == 1.0, "rung 0 is the caller's own force constant"
    assert len(FC_ESCALATION) > 1, "a one-rung ladder cannot escalate at all"
    assert all(a < b for a, b in itertools.pairwise(FC_ESCALATION)), "a repeated rung retries the same relax"


def test_a_conformer_the_soft_relax_tears_is_rescued_at_its_own_stiffness(caplog):
    """`_relax_constrained` stops at the first rung leaving any conformer intact, so `_rescue_torn` retries the rest.

    `threads=1, prune_rms=-1` because the assertion is on which seeds tore: RDKit prunes across threads as
    conformers complete, so the default (all cores, 0.1 A) leaves the surviving set free to vary under load.
    """
    with caplog.at_level(logging.INFO, logger="rxembed"):
        confs = embed(_isomer(_NI_N), n=4, seed=1, threads=1, prune_rms=-1).minimize()
    line = next((r.getMessage() for r in caplog.records if r.getMessage().startswith("relax: tore")), "")
    assert line, "this fixture is supposed to tear at the base stiffness: the premise moved"
    rescued = re.search(r"(\d+) rescued", line)
    assert rescued is not None, line
    assert int(rescued.group(1)) >= 1, line
    assert len(confs) == 4, "the ladder must not spend the caller's n"


def test_a_constrained_embed_returns_intact_geometry_or_its_seed_never_a_torn_relax():
    """Every conformer that comes back is bonded: the ladder's whole job."""
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1).minimize()
    assert len(confs) >= 1
    for cid in confs.ids:
        assert bonding_ok(confs.mol, int(cid), constrained=confs.cons.distances), f"conformer {cid} came back torn"


def test_minimize_never_drops_a_conformer():
    """Deciding is the caller's; one that survives no rung falls back to its embedded seed, still counted."""
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1)
    before = list(confs.ids)
    assert confs.minimize().ids == before


def test_minimize_reports_an_energy_for_every_conformer_it_keeps():
    """An unenergised conformer would sort as if it were free: every kept id must carry a number."""
    confs = minimize(_with_geometry("CCCl"), fix={(1, 2): 2.4})
    assert set(confs.energies) == {int(c) for c in confs.ids}
    assert all(np.isfinite(v) for v in confs.energies.values())


# ---------------------------------------------------------------------------------------------------------
# HANDOVER; these two assert the `Isomer` CONSTRUCTOR (`metal_isomers.py`), not this door. They belong in
# `test_metal_isomers.py`; they live here, where they were written, until that file adopts them.
# ---------------------------------------------------------------------------------------------------------


def test_a_set_of_sites_is_refused():
    """`sites` must state which donor sits at which vertex: a set seats them in iteration order, silently.

    Every other guard passes for a set (each donor is seated exactly once), so the only symptom was a different
    isomer: the same donors as a `cis` seating came back labelled `trans`.
    """
    with pytest.raises(TypeError, match="vertex-ordered"):
        Isomer(_mol(_EN_PD), "SPL", {0, 2, 3, 7})


def test_an_undefined_ligand_stereocentre_warns(caplog):
    """One `Isomer` is one species: an unspecified centre would pool both hands into one result."""
    mol = _mol("[NH2](C(C)CC)->[Pd](<-[NH3])(Cl)Cl")
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        Isomer(mol, "SPL", {0: 0, 1: 6, 2: 7, 3: 8})
    assert "undefined" in caplog.text
    assert "enumerate_isomers" in caplog.text


def test_a_stated_number_can_be_read_back_on_the_core_tier():
    """The core promises "a pull, not a snap, so verify it", so the verification verb must be core too.

    `Conformers.measure` was named in this module's own `embed` docstring, in `constraints`, in `relax` and
    in the README while living only on `pipeline.Ensemble`. A base install could state an exact distance and
    had no supported way to check whether it held. Pinned here rather than in the pipeline tests because the
    tier is the point.
    """
    mol = _mol("CCCCO")
    confs = embed(mol, fix={(0, 4): 3.0}, n=6, seed=1).minimize()
    got = confs.measure((0, 4))
    assert set(got) == {"mean", "min", "max", "n"}
    assert got["n"] == len(confs.ids)
    assert abs(got["mean"] - 3.0) < 0.1, f"the stated 3.0 A was not realised: {got}"
    assert len(confs.measure((0, 1, 4))) == 4, "an angle (3 atoms) must work too"
    with pytest.raises(ValueError, match=r"2 \(distance\), 3 \(angle\) or 4"):
        confs.measure((0,))


def test_a_template_composes_with_every_other_way_of_stating_a_rigid_core():
    """All four rigid spellings are one question: where do these atoms' coordinates come from.

    They must compose, and none may vanish. When `template_to_fix` moved into the core it merged only a dict
    `fix` and returned the template's coordinates alone for a list one, so `fix=[2, 3]` beside a template was
    dropped without a word. The list is a coordinate source like the other two, so it merges.
    """
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
