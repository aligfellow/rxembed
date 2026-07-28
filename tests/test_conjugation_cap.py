"""The conjugation torsion cap (``mechanisms.ConjugationCap``): the T3c fix, and its no-harm guard on T3d.

Two tests, both driving the REAL relax path (``rx.embed`` -> ``geometry.check``), never a re-implemented FF:

1. ``test_thiourea_conjugation_stays_flat`` — the Schreiner thiourea + acetone NCI complex. Without the cap the
   UFF relax twists the conjugated C=S-N torsion far out of plane (36-40 deg, above the 30 deg ``conjugation``
   gate) because UFF has no term holding that plane; with the cap the twist is driven flat and the
   ``conjugation`` violation clears on every conformer (measured 0.0 deg, gate 12/12 in
   ``docs/findings/ff-handling.md`` §4). RED verified by commenting out the MECHANISM_ORDER entry.

2. ``test_conjugation_cap_does_not_harm_sp2_planarity`` — the no-harm property. The chb-tetramisole isothiourea
   ChB complex is the T3d case ``Sp2Planar`` fixes (sp2-carbon out-of-plane pucker). With BOTH caps active (both
   are in ``MECHANISM_ORDER``), its ``planarity`` gate must stay clean — the new torsion cap must not disturb the
   improper hold (the finding measured chb unharmed, 24/24).
"""

import pytest
from rdkit import Chem

import rxembed as rx
from rxembed import geometry as geom
from rxembed.pipeline import EnsembleSet

# Schreiner bis(3,5-CF3-phenyl)thiourea + acetone — the T3c conjugation mode surfaces on the C=S-N torsions
_SCHREINER = "FC(F)(F)c1cc(cc(c1)C(F)(F)F)NC(=S)Nc1cc(cc(c1)C(F)(F)F)C(F)(F)F.CC(C)=O"
# tetramisole isothiourea catalyst + acetic anhydride — the T3d planarity mode Sp2Planar fixes
_CHB = "C1CSC2=NC(CN12)c1ccccc1.CC(=O)OC(C)=O"


def _first_nci_mode(smi):
    modes = rx.nci_modes(Chem.AddHs(Chem.MolFromSmiles(smi)))
    return next(iter(modes.values()))


def _chb_contact():
    cands = rx.nci_candidates(Chem.AddHs(Chem.MolFromSmiles(_CHB)))
    return cands[next(k for k in cands if k.startswith("ChB"))]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_thiourea_conjugation_stays_flat(seed):
    """The conjugated C=S-N torsion is driven flat and the ``conjugation`` gate clears on every conformer."""
    res = rx.embed(_SCHREINER, seed=seed, n=4, contacts=_first_nci_mode(_SCHREINER))
    ensembles = list(res) if isinstance(res, EnsembleSet) else [res]
    assert ensembles
    saw_conformer = False
    for ens in ensembles:
        frozen = [int(f) for f in ens.cons.frozen] or None
        for cid in ens.ids:
            saw_conformer = True
            report = geom.check(ens.mol, int(cid), frozen=frozen)
            bad = [v for v in report.violations if v.kind == "conjugation"]
            assert not bad, f"seed {seed} conf {cid}: conjugation twisted out of plane {[str(v) for v in bad]}"
            # a strong positive signal the torsion is actually FLAT, not merely under the gate line
            worst = max(
                (v.value for v in geom.conjugation(ens.mol, ens.mol.GetConformer(int(cid)).GetPositions())), default=0.0
            )
            assert worst < 15.0, f"seed {seed} conf {cid}: conjugation twist {worst:.1f} deg not driven flat"
    assert saw_conformer, "no conformer was produced — the gate assertion never ran"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_conjugation_cap_does_not_harm_sp2_planarity(seed):
    """With both caps active, the T3d ChB complex's sp2 planarity stays clean (the cap must not harm Sp2Planar)."""
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
