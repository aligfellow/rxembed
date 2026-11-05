"""`workflow/calculators.py`: the refine specs, the xtb adapter and the ASE wrapper, against stub executables."""

import os
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from rxembed.workflow import calculators as calc
from tests.geometry_fixtures import embedded

ACETATE = embedded("CC(=O)[O-]")
METHANOL = embedded("CO")


class Stub(calc.Calculator):
    def energy(self, mol, conf_id=-1):
        return -1.0


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    """Return a function that puts an ``xtb`` on PATH running its Python lines, ``args`` being its command line.

    It stands in for the executable `XTB` runs when ``$XTB_EXE`` is unset, which finds ``xtb`` on PATH.
    """
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    def install(*lines):
        exe = tmp_path / "xtb"
        exe.write_text("\n".join((f"#!{sys.executable}", "import sys", "args = sys.argv[1:]", *lines, "")))
        exe.chmod(0o755)

    return install


def test_resolve_maps_each_refine_spec_to_its_calculator():
    """An xtb spec builds `XTB` with its solvent; 'ff' and None are the force field; a calculator is its own."""
    stub = Stub()
    xtb = [calc.resolve("gxtb"), calc.resolve("gfn2", "water"), calc.resolve("gfnff")]
    assert [(type(c), c.method, c.solvent) for c in xtb] == [
        (calc.XTB, m, s) for m, s in (("gxtb", None), ("gfn2", "water"), ("gfnff", None))
    ]
    assert [calc.resolve("ff"), calc.resolve(None), calc.resolve(stub)] == [None, None, stub]


def test_resolve_refuses_a_spec_it_cannot_compute():
    for spec, solvent, refusal in (
        ("b3lyp", None, "unknown refine"),
        ("ff", "water", "solvent="),
        (Stub(), "water", "solvent="),
    ):
        with pytest.raises(ValueError, match=refusal):
            calc.resolve(spec, solvent)


def test_each_calculator_names_the_model_its_energies_come_from():
    class MACECalculator:
        pass

    calculators = (calc.XTB("gxtb"), calc.XTB("gfn2", "water"), calc.ASE(MACECalculator()), Stub())
    assert [c.name for c in calculators] == ["gxtb", "gfn2 in water", "MACECalculator", "Stub"]


def test_a_calculator_without_a_model_names_what_it_lacks():
    class MACECalculator:
        pass

    for call, lacks in (
        (lambda: calc.Calculator().energy(METHANOL), "Calculator has no energy model"),
        (lambda: calc.ASE(MACECalculator()).optimize(METHANOL), "MACECalculator has no geometry optimiser"),
    ):
        with pytest.raises(NotImplementedError, match=lacks):
            call()


def test_xtb_energy_scores_the_charge_and_solvates_g_xtb(fake_xtb):
    """g-xTB has no ALPB: its energy in water is its gas energy plus GFN2's water-minus-gas correction."""
    fake_xtb(
        "assert args[args.index('-c') + 1] == '-1', 'acetate is an anion'",
        "energies = {('--gxtb', None): -10.0, ('--gfn2', 'water'): -5.5, ('--gfn2', None): -5.0}",
        "solvent = args[args.index('--alpb') + 1] if '--alpb' in args else None",
        "print('TOTAL ENERGY', energies[args[1], solvent], 'Eh')",
    )
    rows = {("gxtb", "water"): -10.5, ("gfn2", "water"): -5.5, ("gxtb", None): -10.0}
    assert {key: calc.XTB(*key).energy(ACETATE) for key in rows} == pytest.approx(rows)


def test_xtb_optimize_refuses_an_unknown_level_or_solvated_g_xtb():
    for xtb, level, refusal in (
        (calc.XTB("gfn2"), "thorough", "unknown xtb opt level"),
        (calc.XTB("gxtb", "water"), "normal", "no implicit solvent"),
    ):
        with pytest.raises(ValueError, match=refusal):
            xtb.optimize(METHANOL, level=level)


def test_xtb_optimize_reads_xtbopt_holding_the_fixed_atoms(fake_xtb):
    """Atoms 0 and 2 are held as xtb's 1-based $fix atoms 1 and 3, in the solvent and at the level asked for."""
    fake_xtb(
        "assert args[1:3] == ['--opt', 'loose'] and args[args.index('--alpb') + 1] == 'water', args",
        "assert open('xtb.inp').read().split()[1:3] == ['atoms:', '1,3']",
        "rows = open(args[0]).read().splitlines()",
        "rows[1] = ' energy: -11.399 gnorm: 0.0001'",
        "open('xtbopt.xyz', 'w').write(chr(10).join(rows) + chr(10))",
    )
    coords, energy = calc.XTB("gfn2", "water").optimize(METHANOL, level="loose", fix=(2, 0, 0))
    assert energy == pytest.approx(-11.399)
    np.testing.assert_allclose(coords, METHANOL.GetConformer().GetPositions(), atol=1e-5)


def test_xtb_raises_for_a_run_that_yields_no_energy(fake_xtb):
    """A failed run, a missing total energy or an unscored xtbopt.xyz is a RuntimeError, which drops one conformer."""
    rows = (
        (("sys.exit('SCF did not converge')",), "energy", "failed"),
        (("print('normal termination')",), "energy", "no total energy"),
        (("open('xtbopt.xyz', 'w').write(open(args[0]).read())",), "optimize", "energy:"),
    )
    for lines, step, refusal in rows:
        fake_xtb(*lines)
        with pytest.raises(RuntimeError, match=refusal):
            getattr(calc.XTB("gfn2"), step)(METHANOL)


def test_ase_converts_ev_to_hartree_and_passes_the_charge(monkeypatch):
    """A stub ASE whose calculator reads the per-atom and the system charge of acetate, and returns 1 Eh in eV."""
    ase = ModuleType("ase")

    class Atoms:
        def __init__(self, **_kw):
            self.charges, self.info = None, {}
            self.calc: Any = None

        def set_initial_charges(self, charges):
            self.charges = charges

        def get_potential_energy(self):
            return self.calc.get_potential_energy(atoms=self)

    ase.__dict__["Atoms"] = Atoms
    monkeypatch.setitem(sys.modules, "ase", ase)
    seen = {}

    class EVCalculator:
        def get_potential_energy(self, atoms):
            seen.update(initial=sum(atoms.charges), info=atoms.info["charge"])
            return 27.211386245988

    assert calc.ASE(EVCalculator()).energy(ACETATE, 0) == pytest.approx(1.0)
    assert seen == {"initial": -1, "info": -1}
