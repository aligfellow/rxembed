"""The sp2-planarity preserve (``mechanisms.Sp2Planar``): the T3d fix, and its hold-at-seed safety guard.

Two tests, both driving the REAL relax path (no re-implemented force field):

1. ``test_thiourea_catalyst_stays_planar`` — ``rx.embed`` on the chb-tetramisole isothiourea catalyst +
   acetic-anhydride ChB complex. Without ``Sp2Planar`` the UFF relax puckers the conjugated sp2 carbon
   (C3 0.003 -> ~0.20 A), tripping the ``planarity`` gate on every conformer. With it, every conformer is
   planarity-clean (measured 0/8 -> 8/8, planarity violations 8 -> 0 per seed).

2. ``test_flat_seed_bowl_is_held_near_flat`` — the honest hold-at-seed behaviour on a genuinely-bowled PAH
   that ETKDG seeds FLAT (the R4 out-of-domain case). Starting from the FLAT ETKDG seed (not a pre-bowled
   geometry, which the old test used and which never exercised the real failure), the relax rides the sp2
   impropers out only to the ±``_SP2_HOLD_WIN`` window edge — it does NOT re-form corannulene's real ~19° bowl,
   and it does NOT pin them dead-flat either. R4 measured that no fc holding the chb thiourea clean frees this
   bowl (bowl needs fc≤0.01; thiourea needs fc≥3), so this is a documented limitation, not a defect. The test
   asserts the measured flat-seed reality, so it goes red both if the hold is over-stiffened to pin ~0° and if
   it is softened/widened enough to let the bowl (or the thiourea pucker) return.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import rxembed as rx
from rxembed import geometry as geom
from rxembed.pipeline import EnsembleSet
from rxembed.rdkit_embed.constraints import mechanisms as _mech
from rxembed.rdkit_embed.constraints.base import Constraints
from rxembed.rdkit_embed.refine.ff import restrained_uff

# tetramisole isothiourea catalyst + acetic anhydride — the ChB complex the T3d planarity mode surfaces on
_CHB = "C1CSC2=NC(CN12)c1ccccc1.CC(=O)OC(C)=O"
_CORANNULENE = "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51"  # a genuinely bowl-shaped PAH — its sp2 carbons SHOULD pucker


def _chb_contact():
    cands = rx.nci_candidates(Chem.AddHs(Chem.MolFromSmiles(_CHB)))
    return cands[next(k for k in cands if k.startswith("ChB"))]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_thiourea_catalyst_stays_planar(seed):
    res = rx.embed(_CHB, seed=seed, n=4, contacts=_chb_contact())
    ensembles = list(res) if isinstance(res, EnsembleSet) else [res]
    assert ensembles
    saw_conformer = False
    for ens in ensembles:
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            saw_conformer = True
            bad = [v for v in geom.check(ens.mol, int(cid), frozen=frozen).violations if v.kind == "planarity"]
            assert not bad, f"seed {seed} conf {cid}: {[str(v) for v in bad]}"
    assert saw_conformer, "no conformer was produced — the gate assertion never ran"


def _sp2_carbons(mol):
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) == 3:
            out.append((atom.GetIdx(), nbrs))
    return out


def _worst_improper(mol, conf):
    """Largest |sp2-carbon improper| (deg): the out-of-plane wag the hold is centred on."""
    vals = [abs(rdMolTransforms.GetDihedralDeg(conf, nb[0], nb[1], nb[2], c)) for c, nb in _sp2_carbons(mol)]
    return max(vals, default=0.0)


def test_flat_seed_bowl_is_held_near_flat():
    """Corannulene's real ~19° bowl is seeded FLAT by ETKDG; hold-at-seed then holds it ~flat (R4, out-of-domain).

    Starts from the FLAT ETKDG seed — the real pipeline input the old pre-bowled test never exercised. The hold
    rides each sp2 improper out to the ±_SP2_HOLD_WIN window edge (≈5°), NOT the ~19° bare-UFF bowl and NOT
    dead-flat 0°. Asserting that measured band catches BOTH regressions: an over-stiff hold (or zero window)
    would pin ~0°; a too-soft/too-wide one would re-form the bowl — the same lever that lets the thiourea
    pucker return (docs/findings/review-batch.md §R4).
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(_CORANNULENE))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    seed_imp = _worst_improper(mol, mol.GetConformer())
    assert seed_imp < 1.0, f"ETKDG did not seed corannulene flat ({seed_imp:.1f}°) — the R4 premise is void"
    # (bare UFF with no hold recovers corannulene's genuine ~19° bowl — the ceiling a too-soft hold would leak toward)
    # relax the FLAT seed through the real registry: Sp2Planar holds each sp2 C at its (flat) seed, ±window
    restrained_uff(mol, Constraints())
    held = _worst_improper(mol, mol.GetConformer())
    lo, hi = 0.5 * _mech._SP2_HOLD_WIN, _mech._SP2_HOLD_WIN + 3.0  # rides the ±5° edge: neither ~0° nor the ~19° bowl
    assert lo < held < hi, (
        f"corannulene worst sp2 improper {held:.1f}° — expected it to ride the ±{_mech._SP2_HOLD_WIN:.0f}° hold "
        f"window (over-stiff would pin it ~0°; too soft/wide would leak toward the ~19° bare-UFF bowl)"
    )
