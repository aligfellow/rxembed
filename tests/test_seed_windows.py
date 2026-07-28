"""`rx.embed()` output must satisfy the constraint windows it was embedded under.

The suite had no assertion of this at all, which is how a defect this size survived 285 green tests: the
raw ETKDG seed misses its own angle windows by 9.5 deg on average and up to 42.8, and every behavioural
test downstream calls a stage (`minimize`/`prune`/`score`) that relaxes first, so none of them could see
it. These tests read `ens.ids` straight off `rx.embed` and nothing else.

Evidence and per-axis measurements: `docs/findings/seed-vs-relax.md`.
"""

import numpy as np
import pytest
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed.embed.dispatch import _embed_dispatch

# Slack, not zero. The relax honours a window to within numerical noise, but `fix={(i, j): d}` writes a
# window narrower than UFF's own equilibrium, so a stiff pull settles a hair outside it. These bars are
# far below the RAW-SEED violations they exist to catch (17-63 deg, 0.55 A) — see the module docstring.
_ANGLE_SLACK = 2.0  # deg
_DIST_SLACK = 0.05  # A


def _per_conformer(ens):
    """[(max angle-window violation deg, max distance-window violation A)], one entry per conformer."""
    out = []
    for cid in ens.ids:
        pos = ens.mol.GetConformer(cid).GetPositions()
        ang = dist = 0.0
        for (i, j), (lo, hi) in ens.cons.distances.items():
            d = float(np.linalg.norm(pos[i] - pos[j]))
            dist = max(dist, lo - d, d - hi)
        for (i, j, k), (lo, hi) in ens.cons.angles.items():
            u, v = pos[i] - pos[j], pos[k] - pos[j]
            a = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
            ang = max(ang, lo - float(a), float(a) - hi)
        out.append((max(ang, 0.0), max(dist, 0.0)))
    return out


def _worst(ens):
    """(max angle-window violation deg, max distance-window violation A) over every conformer."""
    per = _per_conformer(ens)
    return max((a for a, _ in per), default=0.0), max((d for _, d in per), default=0.0)


def test_metal_embed_satisfies_its_coordination_windows():
    """A bis-en Co(III) octahedron: `rx.embed` alone, no `.minimize()`.

    The in-repo stand-in for XAWQUH (an external tmQM refcode, and `tests/` deliberately depends on no
    corpus outside the repo). Same phenomenon, larger: this arrangement's raw seed misses a coordination
    angle window by 63 deg, against XAWQUH's 41 -- both go to 0.00 once the relax runs. It is also the
    golden fixture for the angle INTERSECT branch, so the two nets cover one molecule.
    """
    ens = rx.embed(rx.metal("Cl[Co]12(Cl)(NCCN1)NCCN2", "octahedral")[0], n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    ang, dist = _worst(ens)
    assert ang < _ANGLE_SLACK, f"embed() left a coordination angle {ang:.1f} deg outside its window"
    assert dist < _DIST_SLACK, f"embed() left a distance {dist:.3f} A outside its window"


def test_a_seed_the_relax_tears_keeps_its_seed_geometry_not_a_wrecked_one():
    """`embed` must never hand back a conformer the relax wrecked — and must not silently spend the caller's `n`.

    henry Ni is the case that forces this: the window relax tears 3 of 8 seeds at the base stiffness, one with
    the kappa1 carboxylate wrenched to a 119 deg anti-offset (an sp2 carbon is rigidly 180). `_rescue_torn`
    re-relaxes each at its own minimal sufficient stiffness and falls back to the seed for any that survives no
    rung, so the count is preserved and no conformer is worse than the seed it came from. The residual is
    stated, not hidden: a fallback conformer keeps its seed's window violation, which is why this asserts a
    MAJORITY satisfy the windows rather than all of them.
    """
    henry = "CC[P]1(CC)CC[P](CC)(CC)->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1"
    iso = rx.metal(henry, "square_planar")[0]
    seeds = _embed_dispatch(iso, n=8, seed=1)
    ens = rx.embed(iso, n=8, seed=1)
    assert len(ens.ids) == len(seeds.ids), "embed dropped conformers — the caller's n must survive the relax"

    # an sp2 carboxyl carbon holds its two substituents rigidly anti; the seed is 180 on every conformer
    o = next(d for d in iso.donors if iso.mol.GetAtomWithIdx(d).GetSymbol() == "O")
    c = next(nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(o).GetNeighbors() if nb.GetSymbol() == "C")
    subs = [
        nb.GetIdx() for nb in iso.mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() != o and nb.GetAtomicNum() > 1
    ]
    for cid in ens.ids:
        conf = ens.mol.GetConformer(cid)
        phis = [rdMolTransforms.GetDihedralDeg(conf, iso.metal, o, c, x) for x in subs]
        anti = abs((phis[0] - phis[1] + 540) % 360 - 180)
        assert anti > 120.0, f"conformer {cid}: the relax wrecked the carboxylate (anti offset {anti:.1f} deg)"

    ok = sum(1 for a, d in _per_conformer(ens) if a < _ANGLE_SLACK and d < _DIST_SLACK)
    seed_ok = sum(1 for a, d in _per_conformer(seeds) if a < _ANGLE_SLACK and d < _DIST_SLACK)
    assert seed_ok == 0, "the raw seed is supposed to satisfy NOTHING here — the premise moved"
    assert ok >= len(ens.ids) - 2, f"only {ok}/{len(ens.ids)} conformers satisfy their windows"


@pytest.mark.parametrize(
    ("smiles", "constrain"),
    [
        ("OC(=O)CCCCc1ccccc1", {(1, 9): (2.6, 3.0)}),  # the golden `acid_arene` fixture: seed misses by 0.55 A
        ("NCCCCCCC(=O)O", {(0, 8): (2.5, 3.0)}),  # the class the finding reproduced at 0.208-0.432 A
    ],
)
def test_organic_constrain_window_is_satisfied_by_embed(smiles, constrain):
    """A soft `constrain=` window on a plain organic — no metal, no frozen core, no `.minimize()`."""
    ens = rx.embed(smiles, constrain=constrain, n=4, seed=1)
    assert ens.ids, "embed produced no conformers"
    _ang, dist = _worst(ens)
    assert dist < _DIST_SLACK, f"embed() left the constrain= window violated by {dist:.3f} A"
