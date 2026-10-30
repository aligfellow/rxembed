"""Test coembedding through its public stereo and paired-workflow contracts."""

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.pipeline import dispatch
from rxembed.stereo import defined_stereo_label, stereo_from_3d


def test_coembed_stereo():
    amines = tuple(
        rx.metal(Chem.AddHs(rx.parse_smiles(f"[Pd](Cl)(Cl)(Cl)([N{tag}H](C)O)")), "SPL")[0] for tag in ("@", "@@")
    )
    cobalt = rx.metal("N->[Co+3](<-N)(<-N)(<-[Cl-])(<-[Cl-])<-[Cl-]", "OCT")
    cases = [
        ("retention", "[Br-].CO[C@H](C)F", "CO[C@H](C)Br.[F-]"),
        ("inversion", "F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"),
        ("E/Z", "CC/C=C/CCC", r"CC/C=C\CCC"),
        ("axial", "Cc1cccc(O)c1-c1c(C)cccc1O |wU:7.6|", "Cc1cccc(O)c1-c1c(C)cccc1O |wD:7.6|"),
        ("coordinated donor", *amines),
        ("coordination", cobalt[0], cobalt[-1]),
    ]
    for name, reactant, product in cases:
        pair = rx.coembed(reactant, product, n=4, seed=1, threads=1)
        assert pair.ids, name
        for endpoint, source in zip((pair.reactant, pair.product), (reactant, product), strict=True):
            for cid in pair.ids:
                if isinstance(source, rx.Isomer):
                    assert rx.cxsmiles(endpoint.restored_mol(cid)) == rx.cxsmiles(source), name
                else:
                    assert stereo_from_3d(endpoint.restored_mol(cid)) == defined_stereo_label(
                        rx.parse_smiles(source)
                    ), name
        if name == "retention":
            hydrogens = {
                (i, j) for i, j in pair.correspondence if pair.reactant.mol.GetAtomWithIdx(i).GetAtomicNum() == 1
            }
            assert hydrogens <= set(pair.alignment_pairs)
        elif name == "inversion":
            assert max(pair.alignment_scores.values()) < 0.3


def test_coembed_workflow(monkeypatch, caplog, capsys):
    palladium = rx.metal("c1ccn(->[Pd+2](<-[Cl-])(<-[Cl-])<-n2ccccc2)cc1", "SPL")[0]
    cases = [
        ("CCCCBr.[Cl-]", "CCCCCl.[Br-]", {(0, 5): (3.0, 3.3)}, "ERROR"),
        ("CC(=O)NC", "CC(=O)NC", {(0, 4): (3.4, 3.4)}, "INFO"),
        (palladium, palladium, None, "DEBUG"),
    ]
    for reactant, product, constrain, level in cases:
        with caplog.at_level(level, logger="rxembed"):
            pair = rx.coembed(reactant, product, constrain=constrain, n=4, seed=42, threads=1)
        assert pair.ids == pair.reactant.ids == pair.product.ids
        assert pair.ids
        assert list(pair.scores.values()) == sorted(pair.scores.values())
        assert pair.scores == pytest.approx(
            {cid: pair.joint_results[cid]["score"] for cid in pair.ids}, rel=1e-12, abs=1e-10
        )
        onto = dict(pair.correspondence)
        for side, endpoint in enumerate((pair.reactant, pair.product)):
            for cid in pair.ids:
                positions = endpoint.mol.GetConformer(cid).GetPositions()
                assert np.isfinite(positions).all()
                for (i, j), (lo, hi) in (constrain or {}).items():
                    a, b = (onto[i], onto[j]) if side else (i, j)
                    assert lo - 0.05 < np.linalg.norm(positions[a] - positions[b]) < hi + 0.05
    assert capsys.readouterr().out == ""
    assert any(record.levelname == "INFO" and "coembed: retained" in record.message for record in caplog.records)
    assert any(record.levelname == "DEBUG" and "coembed: attempt" in record.message for record in caplog.records)

    # Reference perception is optional and must stay unused in free mode, including metal geometries.
    monkeypatch.setattr(dispatch, "signature", lambda *args, **kwargs: pytest.fail("free mode perceived a reference"))
    geometry = pair.reactant.mol
    assert rx.coembed(geometry, geometry, stereo="free", n=1, seed=42, threads=1).ids
