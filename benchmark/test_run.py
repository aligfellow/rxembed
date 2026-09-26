"""Contract tests for the benchmark runner: one real round trip, every check that guards a pass, and compare."""

import csv
import json
import math
import os
import time
from types import SimpleNamespace

import pytest
import run
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx

# cis-[PtCl2(NH3)2]: two isomers (cis is the reference), square planar, NH3 hydrogens pointing away from Pt.
_CISPLATIN = """11
cis-[PtCl2(NH3)2]
Pt   0.000000   0.000000   0.000000
Cl   2.300000   0.000000   0.000000
Cl   0.000000   2.300000   0.000000
N   -2.050000   0.000000   0.000000
N    0.000000  -2.050000   0.000000
H   -2.386700   0.952200   0.000000
H   -2.386700  -0.476100   0.824700
H   -2.386700  -0.476100  -0.824700
H    0.952200  -2.386700   0.000000
H   -0.476100  -2.386700   0.824700
H   -0.476100  -2.386700  -0.824700
"""


@pytest.fixture
def cisplatin_xyz(tmp_path):
    path = tmp_path / "cisplatin.xyz"
    path.write_text(_CISPLATIN)
    return path


@pytest.fixture
def ref(cisplatin_xyz):
    return rx.read_xyz(str(cisplatin_xyz), charge=0, connectivity="xyzgraph", bond_orders="xyz2mol")


def _placed(mol, coords):
    """Return a copy of `mol` whose only conformer is at `coords`."""
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


def _on_a_line(smiles, scale=1.0):
    """Return the SMILES graph with atom i at (i * scale, 0, 0); the redox checks never read the geometry."""
    mol = Chem.MolFromSmiles(smiles)
    return _placed(mol, [(i * scale, 0, 0) for i in range(mol.GetNumAtoms())])


def _scaled(mol, factor):
    """Copy `mol` scaled outward from atom 0: the same graph, a small RMSD away from the input."""
    positions = mol.GetConformer().GetPositions()
    return _placed(mol, positions[0] + (positions - positions[0]) * factor)


def _written(mol, tmp_path):
    path = tmp_path / "candidate.xyz"
    Chem.MolToXYZFile(mol, str(path))
    return path


def test_cisplatin_round_trips_through_main(tmp_path, monkeypatch):
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "cisplatin.xyz").write_text(_CISPLATIN)
    (tmp_path / "fixtures.csv").write_text("id,origin,charge,focus,smiles\ncisplatin,rxembed,0,test,\n")
    monkeypatch.setattr(run, "HERE", tmp_path)
    out = tmp_path / "out.csv"
    run.main(["fixtures", "--out", str(out), "--timeout", "60", "--keep-xyz"])

    with out.open() as fh:
        header = json.loads(fh.readline()[2:])
        (row,) = csv.DictReader(fh)
    assert header["cohort"] == "fixtures"
    assert (row["status"], row["isomers"], row["embedded"], row["valid"]) == ("pass", "2", "2", "2")
    assert len(row["valid_cx"].split()) == 2
    assert sorted(p.name for p in (tmp_path / "out-xyz").iterdir()) == ["cisplatin-s42-k1.xyz", "cisplatin-s42-k2.xyz"]


def _iso(k, cx, valid, fresh_cx="", core_rmsd="0.1"):
    return {"k": k, "cx": cx, "valid": valid, "stage": "", "error": "", "fresh_cx": fresh_cx, "core_rmsd": core_rmsd}


@pytest.mark.parametrize(
    ("candidates", "status", "stage"),
    [
        ([_iso(1, "REF", True, "REF"), _iso(2, "OTHER", False)], "pass", ""),
        ([_iso(1, "A", True, "A"), _iso(2, "B", True, "B")], "fail", "enumerate"),  # reference absent
        ([_iso(1, "REF", True, "REF"), _iso(2, "REF", True, "REF")], "fail", "enumerate"),  # matched twice
        ([_iso(1, "REF", True, "REF"), _iso(2, "B", True, "REF")], "fail", "validate:identity"),
        ([_iso(1, "REF", True, "REF", core_rmsd="1.21")], "fail", "validate:core-rmsd"),
        ([_iso(1, "REF", True, "REF", core_rmsd="")], "pass", ""),  # unmeasured never gates
    ],
)
def test_verdict(candidates, status, stage):
    verdict = run._verdict("REF", candidates)
    assert (verdict["status"], verdict["stage"]) == (status, stage)


def test_validate_fresh_rejects_the_input_geometry_played_back(ref, tmp_path):
    result = run._validate(ref, Chem.Mol(ref), None, 0, None, "", 0, _written(ref, tmp_path))
    assert result[1] == "validate:fresh"


def test_validate_connectivity_flags_a_chloride_moved_three_angstrom(ref, tmp_path):
    mol = _scaled(ref, 1.05)
    moved = mol.GetConformer().GetPositions()
    moved[1, 0] += 3.0  # Cl
    result = run._validate(ref, mol, None, 0, None, "", 0, _written(_placed(mol, moved), tmp_path))
    assert result[1] == "validate:connectivity"


def test_validate_cx_flags_the_wrong_expected_cx(ref, tmp_path):
    mol = _scaled(ref, 1.05)
    iso = rx.metal(mol, observed_only=True)[0]
    result = run._validate(ref, mol, None, 0, iso, "not-the-real-cx", 0, _written(mol, tmp_path))
    assert result[1] == "validate:cx"


def test_validate_geometry_flags_a_reported_violation(ref, tmp_path):
    mol = _scaled(ref, 1.05)
    iso = rx.metal(mol, observed_only=True)[0]
    violation = rx.geom_check.Violation(kind="clash", atoms=(0, 1), value=1.0, limit=2.0, detail="test")
    ens = SimpleNamespace(check=lambda: {0: rx.geom_check.GeometryReport(violations=[violation])})
    result = run._validate(ref, mol, ens, 0, iso, run._requested_cx(mol, iso), 0, _written(mol, tmp_path))
    assert result[1] == "validate:geometry"


def test_constitution_key_survives_an_atropisomeric_input():
    path = run.HERE / "fixtures" / "PdCl2-R-BINAP.xyz"
    atrop = rx.read_xyz(str(path), charge=0, connectivity="xyzgraph", bond_orders="xyz2mol")
    with pytest.raises(ValueError, match="atropisomer"):
        rx.dative_smiles(Chem.Mol(atrop))
    assert run._constitution(atrop) == run._constitution(Chem.Mol(atrop))


# A bis-dithiolene Mo complex read as Mo(VI) with two dithiolates, or Mo(IV) with one dithiolate and one dithione.
_DITHIOLATE = "[Mo+6]12(<-[Cl-])(<-[Br-])(<-[S-]C=C[S-]->1)<-[S-]C=C[S-]->2"
_DITHIONE = "[Mo+4]12(<-[Cl-])(<-[Br-])(<-[S-]C=C[S-]->1)<-S=CC=S->2"


@pytest.mark.parametrize(
    ("stored", "reread", "stage"),
    [
        (_DITHIOLATE, _DITHIONE, ""),  # the same complex in another Lewis form passes, with a note
        (_DITHIONE.replace("+4", "+3") + ".[Na+]", _DITHIONE + ".[Na]", "validate:constitution"),  # charge swap
    ],
)
def test_validate_judges_a_changed_lewis_form_by_its_redox_key(monkeypatch, stored, reread, stage):
    mol, fresh = _on_a_line(stored), _on_a_line(reread, scale=1.001)
    iso = SimpleNamespace(centres=())
    monkeypatch.setattr(run.rx, "read_xyz", lambda *a, **k: fresh)
    ens = SimpleNamespace(check=lambda: {0: True})
    _ok, got, detail, _cx = run._validate(_scaled(mol, 1.05), mol, ens, 0, iso, run._requested_cx(mol, iso), 0, "")
    assert got == stage
    assert ("Lewis re-read" in detail) == (stage == "")


def test_redox_key_flags_a_changed_radical_count():
    radical = Chem.RWMol(Chem.MolFromSmiles(_DITHIONE))
    radical.GetAtomWithIdx(7).SetNumRadicalElectrons(1)  # one dithione sulfur
    radical.GetAtomWithIdx(7).SetNoImplicit(True)
    radical.UpdatePropertyCache(strict=False)
    assert run._redox_key(Chem.MolFromSmiles(_DITHIONE)) != run._redox_key(radical)


def test_timeout_keeps_the_isomer_that_already_finished(cisplatin_xyz, tmp_path, monkeypatch):
    real_embed, calls = rx.embed, []

    def one_real_then_hang(*a, **k):
        calls.append(a)
        return real_embed(*a, **k) if len(calls) == 1 else time.sleep(15)

    monkeypatch.setattr(run.rx, "embed", one_real_then_hang)
    row = run._run_one(cisplatin_xyz, 0, 42, tmp_path / "half-slow", 10)
    assert (row["status"], row["stage"]) == ("timeout", "embed 2/2")
    assert (row["isomers"], row["embedded"], row["valid"]) == (1, 1, 1)


def test_worker_crash_is_a_failure_never_a_pass(cisplatin_xyz, tmp_path, monkeypatch):
    monkeypatch.setattr(run.rx, "embed", lambda *a, **k: os._exit(1))
    row = run._run_one(cisplatin_xyz, 0, 42, tmp_path / "crashy", 30)
    assert (row["status"], row["stage"]) == ("fail", "worker")


def test_ring_spun_about_the_metal_axis_scores_near_zero_core_rmsd():
    """A Cp face is scored at its centroid, which a spin about the Fe-centroid axis leaves in place."""
    fe = Chem.RWMol()
    fe.AddAtom(Chem.Atom(26))
    coords, rings = [(0, 0, 0)], []
    for z in (1.66, -1.66):
        ring = [fe.AddAtom(Chem.Atom(6)) for _ in range(5)]
        for k, carbon in enumerate(ring):
            fe.AddBond(carbon, ring[k - 1], Chem.BondType.SINGLE)
            fe.AddBond(carbon, 0, Chem.BondType.DATIVE)
            coords.append((1.21 * math.cos(2 * math.pi * k / 5), 1.21 * math.sin(2 * math.pi * k / 5), z))
        rings.append(ring)
    fe.UpdatePropertyCache(strict=False)
    ref = _placed(fe, coords)
    c, s = math.cos(math.radians(36)), math.sin(math.radians(36))
    spun = [(x * c - y * s, x * s + y * c, z) if i in rings[0] else (x, y, z) for i, (x, y, z) in enumerate(coords)]
    iso = SimpleNamespace(vertices=[-1, -2], haptic={-1: tuple(rings[0]), -2: tuple(rings[1])})

    assert run.core_rmsd(ref, _placed(fe, spun), iso) < 0.10
    assert run.core_rmsd(ref, _placed(fe, spun), None) > 0.10  # per-carbon scoring sees the spin


def test_swapped_dithiocarbamate_sulfurs_score_near_zero_core_rmsd():
    """The -1 sits on one of two equivalent sulfurs; ranking must ignore it so they may swap."""
    pd = Chem.MolFromSmiles("[Pd+2]1(<-[Cl-])(<-[Cl-])<-[S-]C(N(C)C)=S->1")
    coords = [(0, 0, 0), (0.6, -1.6, 0.3), (-1.9, -1.1, -0.2), (-1.15, 1.6, 0)]
    coords += [(0, 2.5, 0), (0, 3.7, 0), (-1.2, 4.4, 0), (1.2, 4.4, 0), (1.15, 1.6, 0)]
    swapped = [*coords[:3], coords[8], *coords[4:8], coords[3]]
    assert run.core_rmsd(_placed(pd, coords), _placed(pd, swapped), None) < 0.10


def test_core_rmsd_is_zero_when_cisplatin_chlorides_swap(ref):
    positions = ref.GetConformer().GetPositions()
    positions[[1, 2]] = positions[[2, 1]]
    assert run.core_rmsd(ref, _placed(ref, positions), None) == pytest.approx(0.0, abs=1e-9)


def _results(path, spec):
    """Write a results CSV from {id: one (status, valid_cx) pair per seed, seeds numbered from 1}."""
    with path.open("w", newline="") as fh:
        fh.write("# {}\n")
        writer = csv.DictWriter(fh, fieldnames=run.FIELDS, lineterminator="\n")
        writer.writeheader()
        for i, per_seed in spec.items():
            for seed, (status, valid_cx) in enumerate(per_seed, start=1):
                writer.writerow({"id": i, "seed": seed, "status": status, "seconds": "1", "valid_cx": valid_cx})
    return path


def test_compare_reports_majority_losses_only(tmp_path, capsys):
    p, f, ab, a = ("pass", ""), ("fail", ""), ("pass", "aaaa bbbb"), ("pass", "aaaa")
    base = {
        "LOSS": [p, p, p, f, f],  # 3/5 -> 2/5 crosses the majority of 3: lost
        "NOISE1": [p, p, p, p, p],  # 5/5 -> 4/5
        "NOISE2": [p, p, f, f, f],  # 2/5 -> 0/5
        "ISOLOSS": [ab, ab, ab, a, a],  # isomer bbbb 3/5 -> 1/5: lost
        "TIMEOUT": [("pass", "cccc")] * 5,
        "BASEONLY": [p],
    }
    new = {
        "LOSS": [p, p, f, f, f],
        "NOISE1": [p, p, p, p, f],
        "NOISE2": [f, f, f, f, f],
        "ISOLOSS": [ab, a, a, a, a],
        "TIMEOUT": [("timeout", "")] * 5,  # a lost reference, but its unreached isomer is no evidence
    }
    base_csv, new_csv = _results(tmp_path / "base.csv", base), _results(tmp_path / "new.csv", new)

    assert run.compare(base_csv, new_csv) == 1
    out = capsys.readouterr().out
    assert [line.split(" (")[0] for line in out.splitlines() if line.startswith("lost")] == [
        "lost isomer: ISOLOSS bbbb 3/5 -> 1/5",
        "lost reference: LOSS 3/5 -> 2/5",
        "lost reference: TIMEOUT 5/5 -> 0/5",
    ]
    assert "1 ids only in BASE, 0 only in NEW" in out
    assert "isomers: 1 lost, 0 gained, 0 noise, 1 skipped" in out
    assert run.compare(new_csv, base_csv) == 0
