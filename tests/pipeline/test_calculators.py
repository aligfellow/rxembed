"""`pipeline/calculators.py`: the refine spec → Calculator map, and the xtb executable contract."""

import os
import shutil
from importlib.util import find_spec

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


def test_the_ff_tier_resolves_to_no_calculator_under_either_spelling():
    """'ff' is the surrogate tier: it must return None, not a calculator that would shell out."""
    assert calc.resolve(None) is None
    assert calc.resolve("ff") is None


def test_every_xtb_method_resolves_to_an_xtb_calculator_carrying_its_settings():
    assert {type(calc.resolve(m)) for m in ("gxtb", "gfn2", "gfnff")} == {calc.XTB}
    c = calc.resolve("gfn2", solvent="water", charge=-1)
    assert (c.method, c.solvent, c.charge) == ("gfn2", "water", -1)


def test_a_calculator_instance_passes_through_untouched():
    """The plug-in point: any Calculator (MACE, AIMNet2, ORCA) must reach the ensemble as itself."""
    stub = _Stub()
    assert calc.resolve(stub) is stub


def test_an_unknown_refine_spec_is_refused():
    with pytest.raises(ValueError, match="unknown refine"):
        calc.resolve("dft")


def test_the_base_calculator_names_itself_when_it_has_no_optimiser():
    """`score(refine=<energy-only calc>)` is fine; `optimize()` on it must say which calculator cannot."""
    with pytest.raises(NotImplementedError, match="_Stub"):
        _Stub().optimize(None)


# --- the xtb interface: what it refuses before ever running the binary --------------------------------------


def test_an_unknown_optimisation_level_is_refused_with_the_list():
    with pytest.raises(ValueError, match="unknown xtb opt level"):
        calc.xtb_optimize(None, level="thorough")


def test_a_solvated_gxtb_optimisation_is_refused_rather_than_run_in_gas():
    """g-xTB has no ALPB; optimising gas-phase under a `solvent=` argument would be a silent wrong answer."""
    with pytest.raises(ValueError, match="no implicit solvent"):
        calc.xtb_optimize(None, method="gxtb", solvent="water")


def test_the_thermodynamic_cycle_fires_for_solvated_gxtb_and_for_nothing_else(monkeypatch):
    """E_gxtb(gas) + [E_gfn2(solv) - E_gfn2(gas)]: three calls and the arithmetic; every other case is one call."""
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


def test_the_energy_parser_reads_the_total_energy_line_or_says_it_could_not():
    assert calc._energy("random preamble\n :: total energy   -11.3990 Eh ::\ntail") == pytest.approx(-11.3990)
    with pytest.raises(RuntimeError, match="no total energy"):
        calc._energy("normal termination\n")


def test_an_xtbopt_is_read_back_as_coords_plus_energy_or_raises(tmp_path):
    """A silent None here would rank a conformer at the top of the ensemble on a parse failure."""
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


@pytest.mark.skipif(find_spec("ase") is None, reason="needs rxembed[score]")
def test_the_ase_wrapper_converts_ev_to_hartree():
    """ASE reports eV and every rxembed energy is Hartree: an unconverted number is 27x wrong, silently."""

    class _EV:
        def get_potential_energy(self, atoms=None, **_kw):
            return 27.211386245988

    assert calc.ASE(_EV()).energy(_reference_conformer("CCO", optimize=False), 0) == pytest.approx(1.0)


# --- the executable, when it is actually there --------------------------------------------------------------


@pytest.mark.skipif(not (shutil.which(XTB) or os.path.exists(XTB)), reason="xtb not on PATH / $XTB_EXE")
def test_the_xtb_calculator_returns_a_real_energy():
    e = calc.XTB("gfnff").energy(_reference_conformer("CCO"), 0)
    assert isinstance(e, float)
    assert e < 0.0, f"a bound molecule's GFN-FF energy must be negative, got {e}"
