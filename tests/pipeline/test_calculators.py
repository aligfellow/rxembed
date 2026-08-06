"""`pipeline/calculators.py`: the refine spec → Calculator map, and the xtb executable contract."""

import os
import shutil
import sys
from types import ModuleType
from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed.pipeline import calculators as calc

XTB = os.environ.get("XTB_EXE", "xtb")


def _reference_conformer(smiles, seed=1, optimize=True):
    """Return a Mol with one conformer from plain ETKDG (+ MMFF): a geometry rxembed had no hand in making."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, f"embed failed for {smiles}"
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
    return mol


class _Stub(calc.Calculator):
    def energy(self, mol, conf_id=-1):
        return -1.0


# --- resolve: the one place a refine spec becomes a calculator ---------------------------------------------


def test_resolve_maps_refine_specs_and_rejects_unknown():
    assert calc.resolve(None) is None
    assert calc.resolve("ff") is None

    assert {type(calc.resolve(m)) for m in ("gxtb", "gfn2", "gfnff")} == {calc.XTB}
    c = calc.resolve("gfn2", solvent="water", charge=-1)
    assert (c.method, c.solvent, c.charge) == ("gfn2", "water", -1)

    stub = _Stub()
    assert calc.resolve(stub) is stub

    with pytest.raises(ValueError, match="unknown refine"):
        calc.resolve("dft")


def test_base_calculator_names_missing_optimizer():
    with pytest.raises(NotImplementedError, match="_Stub"):
        _Stub().optimize(None)


# --- the xtb interface: what it refuses before ever running the binary --------------------------------------


def test_xtb_optimize_validates_before_execution():
    with pytest.raises(ValueError, match="unknown xtb opt level"):
        calc.xtb_optimize(None, level="thorough")
    with pytest.raises(ValueError, match="no implicit solvent"):
        calc.xtb_optimize(None, method="gxtb", solvent="water")


def test_solvated_gxtb_uses_thermodynamic_cycle(monkeypatch):
    energies = {("gxtb", None): -10.0, ("gfn2", "water"): -5.5, ("gfn2", None): -5.0}
    seen = []

    def fake(mol, conf_id=-1, method="gxtb", solvent=None, charge=0, grad=False):
        seen.append((method, solvent))
        return energies[method, solvent], None

    monkeypatch.setattr(calc, "singlepoint", fake)
    assert calc.xtb_energy(None, method="gxtb", solvent="water")[0] == pytest.approx(-10.5)
    assert seen == [("gxtb", None), ("gfn2", "water"), ("gfn2", None)]

    seen.clear()  # gfn2 carries its own ALPB, and gas-phase g-xTB has nothing to correct
    calc.xtb_energy(None, method="gfn2", solvent="water")
    calc.xtb_energy(None, method="gxtb")
    assert seen == [("gfn2", "water"), ("gxtb", None)]


# --- parsing xtb's output: be loud, never return a plausible None ------------------------------------------


def test_energy_parser_reads_total_or_raises():
    assert calc._energy("random preamble\n :: total energy   -11.3990 Eh ::\ntail") == pytest.approx(-11.3990)
    with pytest.raises(RuntimeError, match="no total energy"):
        calc._energy("normal termination\n")


def test_xtbopt_returns_coords_and_energy_or_raises(tmp_path):
    good = tmp_path / "xtbopt.xyz"
    good.write_text("2\n energy: -11.399 gnorm: 0.0001\nC 0.0 0.0 0.0\nO 0.0 0.0 1.43\n")
    coords, e = calc._read_xtbopt(str(good))
    assert e == pytest.approx(-11.399)
    assert coords.shape == (2, 3)
    assert coords[1][2] == pytest.approx(1.43)

    bad = tmp_path / "bad.xyz"
    bad.write_text("1\nno energy on this line\nC 0.0 0.0 0.0\n")
    with pytest.raises(RuntimeError, match="energy:"):
        calc._read_xtbopt(str(bad))


# --- the ASE wrapper: units are the whole job --------------------------------------------------------------


def test_ase_wrapper_converts_ev_to_hartree(monkeypatch):
    ase = ModuleType("ase")

    class _Atoms:
        def __init__(self, **_kw):
            self.calc: Any = None

        def get_potential_energy(self):
            return self.calc.get_potential_energy(atoms=self)

    ase.__dict__["Atoms"] = _Atoms
    monkeypatch.setitem(sys.modules, "ase", ase)

    class _EV:
        def get_potential_energy(self, atoms=None, **_kw):
            return 27.211386245988

    assert calc.ASE(_EV()).energy(_reference_conformer("CCO", optimize=False), 0) == pytest.approx(1.0)


# --- the executable, when it is actually there --------------------------------------------------------------


@pytest.mark.skipif(not (shutil.which(XTB) or os.path.exists(XTB)), reason="xtb not on PATH / $XTB_EXE")
def test_xtb_calculator_returns_a_real_energy():
    e = calc.XTB("gfnff").energy(_reference_conformer("CCO"), 0)
    assert e < 0.0, f"a bound molecule's GFN-FF energy must be negative, got {e}"
