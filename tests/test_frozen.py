"""Integration tests for the rigid paths — ``fix`` (graft), ``template``, stacking, and ``rx.minimize``.

The frozen-core Kabsch graft is the load-bearing claim (a TS's partial bonds must survive a random-frame
embed), so these assert the exact contract from CLAUDE.md: the fixed core's internal geometry is preserved
to < 0.01 Å on every conformer. A fast Mol-source case exercises the mechanics in milliseconds; one real TS
(``bimp.xyz``, ~1 s) proves it on a genuine hypervalent-adjacent reacting core plus the geometry gate.
``template`` and the ``fix`` + ``constrain`` composition ("hold a core, softly bias the periphery") round
out the redesign's headline behaviours. No xtb, no openconf.
"""

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rxembed import geometry as geom

_GRAFT_TOL = 0.01  # a fixed core is held EXACTLY — the frozen-core distance assertion the project guarantees


def _embedded(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def _max_core_drift(mol, ids, core, ref_pos):
    """Largest deviation of any core pair's distance from the reference, over all conformers (frame-free)."""
    drift = 0.0
    for cid in ids:
        pos = mol.GetConformer(cid).GetPositions()
        for i, j in itertools.combinations(core, 2):
            drift = max(drift, abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref_pos[i] - ref_pos[j])))
    return drift


# --- fix: own-coordinates graft ----------------------------------------------


def test_fix_own_coords_grafts_core_exactly():
    import rxembed as rx

    mol = _embedded("CC(=O)Nc1ccccc1", seed=1)  # planar amide + arene
    core = [0, 1, 2, 3]  # C-C(=O)-N reacting-core stand-in
    ref = mol.GetConformer().GetPositions()
    ens = rx.embed(mol, fix=core, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, core, ref) < _GRAFT_TOL


def test_fix_ts_core_from_xyz_holds_and_passes_gate():
    import rxembed as rx
    from rxembed.embed.dispatch import _xyz_to_mol

    path = "examples/structures/bimp.xyz"
    core = [10, 11, 12, 14]  # the reacting core (from the 04_organic_ts notebook)
    ref = _xyz_to_mol(path, 0)
    ens = rx.embed(path, fix=core, n=4)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, core, ref.GetConformer().GetPositions()) < _GRAFT_TOL
    for cid in ens.ids:
        geom.check(ens.mol, cid, frozen=core, reference=ref).assert_ok()


# --- fix: explicit coordinates (DESIGN W2 — the *primary* reference form) -----


def test_fix_explicit_coords_grafts_through_embed():
    import rxembed as rx

    ref_pos = _embedded("CC(=O)Nc1ccccc1", seed=7).GetConformer().GetPositions()
    core = [0, 1, 2, 3]
    # the coordinate-dict is DESIGN's primary reference mechanism; template= is only sugar over it
    ens = rx.embed("CC(=O)Nc1ccccc1", fix={i: tuple(ref_pos[i]) for i in core}, n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, core, ref_pos) < _GRAFT_TOL


# --- template: reference sugar (positions + explicit map) --------------------


def test_template_transfers_reference_core():
    import rxembed as rx

    ref_pos = _embedded("CC(=O)Nc1ccccc1", seed=7).GetConformer().GetPositions()
    core = [0, 1, 2, 3]
    ens = rx.embed("CC(=O)Nc1ccccc1", template=(ref_pos, {i: i for i in core}), n=6)
    assert ens.n >= 1
    assert _max_core_drift(ens.mol, ens.ids, core, ref_pos) < _GRAFT_TOL


# --- stacking: a rigid core held EXACTLY while a soft contact is biased -------


def test_fix_core_and_constrain_periphery_compose():
    import rxembed as rx

    # rigid carboxyl core (own coords) + a soft window that folds the distal ring toward it
    mol = _embedded("OC(=O)CCCCc1ccccc1", seed=3)
    core = [0, 1, 2]  # carboxyl O, C, =O
    ref = mol.GetConformer().GetPositions()
    soft, lo, hi = (1, 9), 3.5, 4.2  # carbonyl C to a ring carbon — free d ~ 7.5 A, so the window must bite
    ens = rx.embed(mol, fix=core, constrain={soft: (lo, hi)}, n=10).minimize()
    assert ens.n >= 1
    # 1) the fixed core stays EXACT even as the soft pull deforms the periphery
    assert _max_core_drift(ens.mol, ens.ids, core, ref) < _GRAFT_TOL
    # 2) the soft window is genuinely realised (not grazing the slack) and it actually bit vs a free embed
    stats = ens.measure(soft)
    assert stats["min"] >= lo - 0.15
    assert stats["max"] <= hi + 0.15
    assert rx.embed(mol, n=8).minimize().measure(soft)["mean"] > hi + 1.0
    # 3) a physically valid composed pose exists (the gate is applied, not skipped as the old test did)
    assert any(geom.check(ens.mol, cid, frozen=core).ok() for cid in ens.ids)


# --- fix on an explicit hydrogen: stable H-indexing on an AddHs Mol (invariant 7 / FLP) ---


def test_fix_number_on_explicit_hydrogen_is_delivered():
    from rdkit import Chem

    import rxembed as rx

    # index a specific N-H on the explicit-H Mol and stretch it (the FLP H-transfer idiom, minimal form)
    mol = Chem.AddHs(Chem.MolFromSmiles("CN"))
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    nbr = mol.GetAtomWithIdx(n).GetNeighbors()
    h = next(a.GetIdx() for a in nbr if a.GetAtomicNum() == 1)
    ens = rx.embed(mol, fix={(n, h): 1.20}, n=6).minimize()  # a stretched N-H, held past its ~1.01 equilibrium
    assert ens.n >= 1
    assert abs(ens.measure((n, h))["mean"] - 1.20) < 0.1  # the H index survived the embed and the fix bit


# --- rx.minimize: search-free relax toward targets ---------------------------


def test_minimize_pulls_toward_fix_numbers():
    import rxembed as rx

    mol = _embedded("CCCCCCC", seed=1)  # heptane — pull the two ends together
    ens = rx.minimize(mol, fix={(0, 6): 3.0})
    assert abs(ens.measure((0, 6))["mean"] - 3.0) < 0.15


def test_minimize_holds_constrain_window():
    import rxembed as rx

    mol = _embedded("CCCCCCC", seed=1)
    ens = rx.minimize(mol, constrain={(0, 6): (3.0, 3.4)})
    assert 3.0 - 0.15 <= ens.measure((0, 6))["mean"] <= 3.4 + 0.15


def test_minimize_from_xyz_path(tmp_path):
    import rxembed as rx

    xyz = tmp_path / "mol.xyz"  # DESIGN W5 literally shows minimize("mol.xyz", ...)
    xyz.write_text(Chem.MolToXYZBlock(_embedded("CCCCCCC", seed=1)))
    ens = rx.minimize(str(xyz), fix={(0, 6): 3.0})
    assert abs(ens.measure((0, 6))["mean"] - 3.0) < 0.15


def test_minimize_rejects_smiles():
    import rxembed as rx

    with pytest.raises(ValueError, match="existing geometry"):
        rx.minimize("CCO", fix={(0, 2): 2.0})
