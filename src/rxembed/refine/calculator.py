"""Pluggable energy calculators for the refine tier (wired into ``Ensemble.score``/``optimize``).

`resolve(refine, ...)` maps a spec to a Calculator: "ff" (force field, no calc),
"gxtb"/"gfn2"/"gfnff" (the xtb executable — g-xTB has no ASE interface yet; GFN-FF
is the cheap NCI-aware tier, dispersion + H-bond), or any Calculator instance. Wrap
an ASE calculator (MACE, AIMNet2, xtb-python, ORCA, …) with ASE(...), or subclass
Calculator.energy for anything else.
"""

from __future__ import annotations

from . import xtb as _xtb

_EV_TO_HARTREE = 1 / 27.211386245988


class Calculator:
    """Single-conformer energy in Hartree (and, optionally, a constrained geometry optimisation).

    Subclass and implement ``energy()``; override ``optimize()`` to support ``Ensemble.optimize``.
    """

    def energy(self, mol, conf_id=-1) -> float:
        """Return the single-conformer energy in Hartree."""
        raise NotImplementedError

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        """(optimised coords [N,3] Å, energy_Eh), holding the 0-based atoms `fix` rigid."""
        raise NotImplementedError(f"{type(self).__name__} has no geometry optimiser")


class XTB(Calculator):
    """g-xTB / GFN2 / GFN-FF via the xtb executable (composite g-xTB+GFN2 solvent if solvent set).

    GFN-FF ('gfnff') is a force field with **dispersion + H-bond terms**, so it gives a real NCI-aware
    energy/relax at FF cost — the right tier for ranking an `mc(explore=)` pool before a g-xTB final opt.
    """

    def __init__(self, method="gxtb", solvent=None, charge=0):
        self.method, self.solvent, self.charge = method, solvent, charge

    def energy(self, mol, conf_id=-1):
        """Single-point energy in Hartree via the xtb executable."""
        return _xtb.energy(mol, conf_id, self.method, self.solvent, self.charge)[0]

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        """Constrained xtb optimisation -> (coords[N,3] Angstrom, energy_Eh)."""
        return _xtb.optimize(mol, conf_id, self.method, level, fix, self.charge, self.solvent)


class ASE(Calculator):
    """Wrap any ASE calculator."""

    def __init__(self, ase_calc):
        self.calc = ase_calc

    def energy(self, mol, conf_id=-1):
        """Single-point energy in Hartree via the wrapped ASE calculator."""
        from ase import Atoms

        conf = mol.GetConformer(conf_id)
        atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=conf.GetPositions())
        atoms.calc = self.calc
        return atoms.get_potential_energy() * _EV_TO_HARTREE


def resolve(refine, solvent=None, charge=0):
    """Map a refine spec → Calculator (or None for FF-only)."""
    if refine in (None, "ff"):
        return None
    if isinstance(refine, Calculator):
        return refine
    if refine in ("gxtb", "gfn2", "gfnff"):
        return XTB(refine, solvent, charge)
    raise ValueError(f"unknown refine {refine!r}")
