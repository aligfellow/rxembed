"""Test the core embed and Conformers API."""

from __future__ import annotations

import importlib
import logging
import re
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength

from rxembed import bounds as bnd
from rxembed.constraints import FIX_ANGLE_TOL, FIX_DISTANCE_TOL, Constraints, resolve_core
from rxembed.embed import BASE_STIFFNESS, Conformers, embed, fold_substrate, minimize
from rxembed.metal_core import TRANSITION_METALS, coplanar
from rxembed.metal_isomers import Isomer, enumerate_isomers, from_geometry
from rxembed.metal_smiles import parse_smiles
from rxembed.relax import bonding_ok

emb = importlib.import_module("rxembed.embed")  # the engine implementation module, not the public facade

_BIPY_PD = "Cl[Pd]1(Cl)<-n2ccccc2-c2ccccn->12"
_EN_PD = "Cl[Pd](Cl)(<-N(C)(C)C)<-N(C)(C)C"
# the N-bound Ni(II) isomer: the window relax tears 3 of 8 seeds at the base stiffness, the ladder's type case
_NI_N = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
# cis-[Co(en)2Cl2], the textbook Delta/Lambda pair, and the same complex with one sp3 centre on a backbone:
# the second is the case a reflection cannot repair, because it would invert that centre too
_CO_EN = "Cl[Co]12(Cl)(NCCN1)NCCN2"
_CO_EN_ME = "Cl[Co]12(Cl)(N[C@@H](C)CN1)NCCN2"


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


def test_embed_fix_chains_minimize_and_dump(tmp_path):
    mol = _mol("OCCCN")
    confs = embed(mol, fix={(0, 4): 3.0}, n=8, seed=0xF00D)
    assert len(confs) == len(confs.ids) > 0
    confs.minimize()
    for cid in confs.ids:
        assert _distance(confs._mol, cid, 0, 4) == pytest.approx(3.0, abs=0.1)
    path = confs.dump(tmp_path / "out.xyz")
    assert path.read_text().count(f"{mol.GetNumAtoms()}\n") == len(confs), "one xyz frame per tracked conformer"


def test_numeric_fix_delivers_a_linear_three_centre_core():
    mol = _mol("[F-].CCl")  # F(0), C(1), Cl(2)
    fix = {(0, 1): 2.0, (1, 2): 2.2, (0, 1, 2): 178.0}
    confs = embed(mol, fix=fix, n=1, seed=0xF00D, prune_rms=-1).minimize()
    assert confs
    for cid in confs.ids:
        conf = confs.mol.GetConformer(int(cid))
        assert GetBondLength(conf, 0, 1) == pytest.approx(2.0, abs=FIX_DISTANCE_TOL)
        assert GetBondLength(conf, 1, 2) == pytest.approx(2.2, abs=FIX_DISTANCE_TOL)
        assert GetAngleDeg(conf, 0, 1, 2) == pytest.approx(178.0, abs=FIX_ANGLE_TOL)


def test_numeric_fix_rejects_when_cleanup_cannot_hold_it(monkeypatch, caplog):
    fix = {(1, 2): 2.026, (2, 0): 1.557}
    confs = embed(_mol("[O-].ClCCCCBr"), fix=fix, n=1, seed=1, prune_rms=-1)
    assert not confs._fixed_geometry_ok(confs.ids[0]), "the raw seed must miss for this test to exercise rejection"

    monkeypatch.setattr(
        emb,
        "restrained_uff",
        lambda mol, cons, conf_ids=None, **kw: np.zeros(len(conf_ids if conf_ids is not None else mol.GetConformers())),
    )
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs.minimize()

    assert not confs.ids, "an off-target numeric fix was returned after cleanup"
    assert not confs.unrelaxed, "rejected conformer ids leaked into tracked state"
    assert "requested" in caplog.text
    assert "+/- 0.001" in caplog.text
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


# ---------------------------------------------------------------------------------------------------------
# Conformers: the one result type
# ---------------------------------------------------------------------------------------------------------


def test_indexing_reuses_mol_in_new_conformers():
    confs = embed(_mol("CCOCC"), n=6, seed=7)
    sub = confs[:2]
    assert sub.ids == confs.ids[:2]
    assert sub._mol is confs._mol
    assert confs[0].ids == [confs.ids[0]]
    assert {c.GetId() for c in sub.mol.GetConformers()} == set(sub.ids)
    with pytest.raises(ValueError, match="not one of this result's ids"):
        sub.xyz(confs.ids[-1])  # an id the slice no longer tracks


def test_dump_refuses_a_result_with_no_conformers(tmp_path):
    with pytest.raises(ValueError, match="nothing to dump"):
        Conformers(_mol("CCO"), []).dump(tmp_path / "empty.xyz")


def test_surrogate_is_internal_and_output_restores_metal():
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"seed": -1}, "not reproducible"), ({"n": 0}, "positive conformer")],
    ids=["negative-seed", "zero-conformers"],
)
def test_embed_rejects_silent_misuse(kwargs, match):
    with pytest.raises(ValueError, match=match):
        embed(_mol("CCO"), **kwargs)


def test_embed_refuses_implicit_hydrogens():
    with pytest.raises(ValueError, match="AddHs"):
        embed(Chem.MolFromSmiles("CCO"), n=2)


def test_embed_refuses_a_bare_metal_mol():
    with pytest.raises(ValueError, match="Isomer"):
        embed(_mol(_EN_PD), n=2)


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


def test_pi_stack_constrain_survives_the_fold():
    iso = _isomer()
    a, b = (tuple(r) for r in iso.mol.GetRingInfo().AtomRings() if len(r) == 6)
    cons = _fold(iso, constrain={(a, b): 3.6})
    assert any(set(pa) == set(a) and set(pb) == set(b) for pa, pb, _sep in cons.planes)


def test_fix_landing_on_a_sphere_hold_overrides_it():
    iso = _isomer()
    key = _sphere_key(iso)
    alone, _ref = resolve_core(iso.mol, fix={key: 2.42}, has_geometry=False)
    assert alone.distances[key] != iso.coordination().distances[key], "premise: the two windows must differ"
    folded = _fold(iso, fix={key: 2.42})
    assert folded.distances[key] == alone.distances[key], "the sphere hold clipped the fix"
    assert key not in folded.pulls, "the sphere's approximate pull still competes with the numeric fix"


def test_sphere_hold_remains_nonreleasable():
    iso = _isomer()
    key = _sphere_key(iso)
    cons = _fold(iso, constrain={key: (2.3, 2.5)})
    assert cons.distances[key] == pytest.approx((2.3, 2.5))
    assert key not in cons.contacts[0], "a sphere hold became releasable"
    assert key in cons.relaxed().distances, "the exploratory pass would drop the coordination sphere"


def test_graft_allows_one_sphere_atom_not_two():
    iso = _isomer(_EN_PD)
    fix = {iso.donors[0]: (0.0, 0.0, 0.0), iso.donors[1]: (2.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="pins coordination-sphere atoms"):
        _fold(iso, fix=fix)
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


def test_encounter_bounds_ignore_global_rng():
    mol = _two_fragments()
    assert len(Chem.GetMolFrags(mol)) >= 2, "fixture must be multi-fragment to exercise the encounter bounds"
    before = emb.encounter_bounds(mol)
    assert before, "no inter-fragment bound was produced: the fixture is not exercising the code"
    _burn_global_rng()
    assert emb.encounter_bounds(mol) == before


def test_probe_seed_is_forwarded(monkeypatch):
    seen = []
    real = bnd.probe_conformer
    monkeypatch.setattr(emb, "probe_conformer", lambda m, s: (seen.append(s), real(m, s))[1])
    emb.encounter_bounds(_two_fragments(), seed=4321)
    assert seen == [4321]


def test_float_bounds_apply_only_without_pins():
    mol = _two_fragments()
    assert emb.float_encounter_bounds(mol, Constraints()), "every pair of a free multi-fragment mol must be bounded"

    i, j = sorted(f[0] for f in Chem.GetMolFrags(mol))
    assert emb.float_encounter_bounds(mol, Constraints(distances={(i, j): (3.0, 3.5)})) == {}
    assert emb.float_encounter_bounds(_mol("CCO"), Constraints()) == {}


# ---------------------------------------------------------------------------------------------------------
# graft_frozen; distance geometry only approximates a rigid core; the graft restores it exactly
# ---------------------------------------------------------------------------------------------------------


def test_graft_restores_the_core_shape_exactly():
    mol = _with_geometry("CCCl")
    core = [0, 1, 2]
    ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.8, 0.0]])
    emb.graft_frozen(mol, [0], core, ref)

    pos = mol.GetConformer(0).GetPositions()
    for (a, b), want in (((0, 1), 1.5), ((1, 2), 1.8), ((0, 2), float(np.linalg.norm(ref[0] - ref[2])))):
        assert np.linalg.norm(pos[core[a]] - pos[core[b]]) == pytest.approx(want, abs=1e-9)


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


def test_minimize_pulls_without_search():
    mol = _with_geometry("CCCl")
    before = GetBondLength(mol.GetConformer(0), 1, 2)
    confs = minimize(mol, fix={(1, 2): 2.4})

    assert len(confs) == mol.GetNumConformers(), "minimize must not add or drop conformers: it does not search"
    after = GetBondLength(confs.mol.GetConformer(confs.ids[0]), 1, 2)
    assert abs(after - 2.4) < 0.1, f"C-Cl was not pulled to the target: {before:.2f} -> {after:.2f}"
    assert GetBondLength(mol.GetConformer(0), 1, 2) == pytest.approx(before), "the caller's geometry was relaxed"


def test_minimize_refuses_a_graph_with_no_geometry():
    with pytest.raises(ValueError, match="existing geometry"):
        minimize(_mol("CCO"))


def test_minimize_holds_isomer_polyhedron():
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


def test_minimize_preserves_input_metal_sphere():
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


def test_stiffness_retry_rescues_torn_conformer(caplog, monkeypatch):
    confs = embed(_isomer(_NI_N), n=2, seed=1, threads=1, prune_rms=-1)
    victim = int(confs.ids[0])
    real_intact = confs._intact
    retries = []  # stiffnesses used when `_rescue_torn` retries the victim alone
    batch_marked = False

    embed_mod = sys.modules[Conformers.__module__]
    real_uff = embed_mod.restrained_uff

    def spy_uff(mol, cons, **kw):
        nonlocal batch_marked
        if kw.get("conf_ids") == [victim]:
            retries.append(kw.get("stiffness"))
        elif kw.get("conf_ids") == confs.ids and not batch_marked:
            batch_marked = True
            confs.unrelaxed = [victim]  # a failed batch may flag it before the individual retry succeeds
        return real_uff(mol, cons, **kw)

    def torn_until_retried(cid):  # torn until it has been re-relaxed on its own, honest from then on
        return False if cid == victim and not retries else real_intact(cid)

    monkeypatch.setattr(embed_mod, "restrained_uff", spy_uff)
    monkeypatch.setattr(confs, "_intact", torn_until_retried)
    with caplog.at_level(logging.INFO, logger="rxembed"):
        confs = confs.minimize()

    line = next(
        (r.getMessage() for r in caplog.records if r.getMessage().startswith("minimize:") and "torn" in r.getMessage()),
        "",
    )
    assert line, "the injected tear did not reach `_rescue_torn`"
    assert retries, "a torn conformer must be re-relaxed on its own, not left to the global ladder"
    assert retries[0] > BASE_STIFFNESS, f"the retry did not escalate above {BASE_STIFFNESS}: {retries[0]}"
    rescued = re.search(r"(\d+) rescued", line)
    assert rescued is not None, line
    assert int(rescued.group(1)) >= 1, f"a conformer that is intact on retry must be rescued, not seeded: {line}"
    assert victim not in confs.unrelaxed, "a rescued conformer must not be reported as unrelaxed"
    assert len(confs) == 2, "the ladder must not spend the caller's n"


def test_constrained_embed_returns_intact_or_seed():
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1).minimize()
    assert len(confs) >= 1
    for cid in confs.ids:
        assert bonding_ok(confs.mol, int(cid), constrained=confs.cons.distances), f"conformer {cid} came back torn"


def test_minimize_preserves_count_and_energies():
    confs = embed(_mol("CCCl"), fix={(1, 2): 2.4}, n=4, seed=1)
    before = list(confs.ids)
    assert confs.minimize().ids == before
    assert set(confs.energies) == {int(c) for c in confs.ids}
    assert all(np.isfinite(v) for v in confs.energies.values())


def test_constrained_energies_share_one_final_objective(monkeypatch):
    confs = embed(_mol("CCC"), fix={(0, 2): (2.0, 3.0)}, n=2, seed=1)
    calls = []
    real_uff = emb.restrained_uff

    def marked_uff(mol, cons, **kw):
        result = real_uff(mol, cons, **kw)
        calls.append((kw["stiffness"], kw.get("max_iters")))
        return np.full(len(result), 7.0 if kw.get("max_iters") == 0 else 99.0)

    def replace_after_relax(self, *args):
        self.energies = dict.fromkeys(self.ids, -50.0)  # stand in for a hand re-seed's separately scored batch

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(Conformers, "_hold_metal_hand", replace_after_relax)
    confs.minimize()

    assert calls[-1] == (BASE_STIFFNESS, 0)
    assert set(confs.energies.values()) == {7.0}


def test_restored_donor_hand_is_rescored(monkeypatch):
    mol = Chem.AddHs(parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)"))
    iso = enumerate_isomers(mol, "square_planar")[0]
    confs = embed(iso, n=1, seed=2)
    calls = []
    real_uff = emb.restrained_uff

    def spy_uff(mol, cons, **kw):
        result = real_uff(mol, cons, **kw)
        calls.append((kw.get("max_iters"), result.copy()))
        return result

    monkeypatch.setattr(emb, "restrained_uff", spy_uff)
    energies = confs._relax_constrained(BASE_STIFFNESS)

    assert confs.unrelaxed == [0]
    assert calls[-1][0] == 0
    assert np.array_equal(energies, calls[-1][1])


def test_trajectory_keeps_only_the_accepted_stiffness(monkeypatch):
    mol = _with_geometry("CCO")
    confs = Conformers(mol, [0], Constraints(distances={(0, 2): (2.0, 3.0)}))
    seed = mol.GetConformer(0).GetPositions().copy()
    attempts = []

    def marked_uff(mol, cons, *, stiffness, max_iters, conf_ids=None, _snapshots=None, **_kw):
        if max_iters == 0:
            return np.array([0.0])
        attempts.append(stiffness)
        cid = int(conf_ids[0]) if conf_ids else 0
        positions = mol.GetConformer(cid).GetPositions().copy()
        positions[0, 0] = len(attempts)
        for atom, xyz in enumerate(positions):
            mol.GetConformer(cid).SetAtomPosition(atom, xyz.tolist())
        if _snapshots is not None:
            _snapshots[cid] = [positions.copy()]
        return np.array([float(stiffness)])

    monkeypatch.setattr(emb, "restrained_uff", marked_uff)
    monkeypatch.setattr(Conformers, "_intact", lambda self, cid: len(attempts) >= 2)
    frames = []
    confs._relax_constrained(BASE_STIFFNESS, _frames=frames)
    confs._store_trajectory(frames)

    assert confs.trajectory is not None
    xs = [conf.GetPositions()[0, 0] for conf in confs.trajectory.GetConformers()]
    assert attempts[:2] == [BASE_STIFFNESS, 3 * BASE_STIFFNESS]
    assert xs == pytest.approx([seed[0, 0], 2.0]), "the rejected first-attempt frame leaked into the trajectory"
    assert confs[:0].trajectory is None


# ---------------------------------------------------------------------------------------------------------
# the metal-centre handedness gate
# ---------------------------------------------------------------------------------------------------------


def _hands(confs):
    """The metal-centre hand each returned conformer actually realises, read back from its coordinates."""
    mol = confs.mol
    return [from_geometry(Chem.Mol(mol, False, int(c))).chirality for c in confs.ids]


@pytest.mark.parametrize("want", ["delta", "lambda"])
def test_named_metal_hand_is_preserved(want):
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if i.chirality == want)
    confs = embed(iso, n=8, seed=0xF00D).minimize()
    assert len(confs) >= 4, "the fixture must return enough conformers to be a fair sample"
    assert _hands(confs) == [want] * len(confs)
    assert confs.wrong_hand == []


def test_unstated_metal_hand_skips_hand_check(monkeypatch):

    def boom(self):
        raise AssertionError("the handedness read ran for a caller who named no hand")

    monkeypatch.setattr(emb.Conformers, "_metal_hands", boom)
    assert len(embed(_mol("OCCCN"), n=4, seed=0xF00D).minimize()) == 4  # organic: there is no `iso` at all
    achiral = next(i for i in enumerate_isomers(_mol(_CO_EN), "octahedral") if not i.chirality)
    assert len(embed(achiral, n=4, seed=0xF00D).minimize()) >= 1  # a metal whose centre states no hand


def test_mirror_freedom_depends_on_metal_inversion():
    assert emb._mirror_is_free(_mol("OCCCN"))
    assert emb._mirror_is_free(_mol("C/C=C/CO")), "E/Z is reflection-invariant and must not block the mirror"
    assert not emb._mirror_is_free(_mol("C[C@H](N)CO"))
    assert not emb._mirror_is_free(_mol("CC(N)CO")), "an sp3 centre inverts whether or not it is assigned"


def test_stereocentre_preserves_metal_hand_and_energies():
    iso = next(i for i in enumerate_isomers(_mol(_CO_EN_ME), "octahedral") if i.chirality)
    assert not emb._mirror_is_free(iso.mol), "the premise: this fixture must be beyond the free fix"
    confs = embed(iso, n=6, seed=0xF00D).minimize()
    assert _hands(confs) == [iso.chirality] * len(confs)
    assert confs.wrong_hand == []
    assert set(confs.energies) == {int(c) for c in confs.ids}, "a re-seeded conformer must bring its own energy"


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
    assert confs.wrong_hand == []
    assert _hands(confs) == ["delta"]


def test_unfixable_metal_hand_is_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        confs = minimize(_mirrored_input(_CO_EN_ME))
    assert confs.seed is None, "nothing was embedded here, so there is no seed to re-roll"
    assert confs.wrong_hand == list(confs.ids)
    assert "wrong_hand" in caplog.text
    assert confs[0].wrong_hand == [confs.ids[0]], "a slice must not silently lose the wrong-hand flag"


# ---------------------------------------------------------------------------------------------------------
# HANDOVER; these two assert the `Isomer` CONSTRUCTOR (`metal_isomers.py`), not this door. They belong in
# `test_metal_isomers.py`; they live here, where they were written, until that file adopts them.
# ---------------------------------------------------------------------------------------------------------


def test_set_of_sites_is_refused():
    with pytest.raises(TypeError, match="vertex-ordered"):
        Isomer(_mol(_EN_PD), "SPL", {0, 2, 3, 7})


def test_undefined_ligand_stereocentre_warns(caplog):
    mol = _mol("[NH2](C(C)CC)->[Pd](<-[NH3])(Cl)Cl")
    with caplog.at_level(logging.WARNING, logger="rxembed"):
        Isomer(mol, "SPL", {0: 0, 1: 6, 2: 7, 3: 8})
    assert "undefined" in caplog.text
    assert "enumerate_isomers" in caplog.text


def test_measure_reports_distance_and_angle():
    mol = _mol("CCCCO")
    confs = embed(mol, fix={(0, 4): 3.0}, n=6, seed=1).minimize()
    got = confs.measure((0, 4))
    assert set(got) == {"mean", "min", "max", "n"}
    assert got["n"] == len(confs.ids)
    assert abs(got["mean"] - 3.0) < 0.1, f"the stated 3.0 A was not realised: {got}"
    assert len(confs.measure((0, 1, 4))) == 4, "an angle (3 atoms) must work too"
    with pytest.raises(ValueError, match=r"2 \(distance\), 3 \(angle\) or 4"):
        confs.measure((0,))


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
