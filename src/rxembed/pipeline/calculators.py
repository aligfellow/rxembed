"""The refine tier: the xtb executable interface and the pluggable Calculator surface.

`resolve(refine, ...)` maps a spec to a Calculator: "ff" (no calc), "gxtb"/"gfn2"/"gfnff" (the xtb
executable), or any Calculator instance. Wrap an ASE calculator (MACE, AIMNet2, ORCA, …) with ASE(...),
or subclass Calculator.energy for anything else. The adapter imports ASE only when used; installing the
calculator supplies that dependency, rather than making every scoring install carry it.

The executable half talks to `xtb` on PATH. g-xTB has no ALPB parameters, so solvent comes from a GFN2
correction::

    E = E_gxtb(gas) + [E_gfn2(solvent) - E_gfn2(gas)]

and likewise for the gradient. Set $XTB_EXE to override the binary. `ff_energies` / `restrained_uff` are
re-exported from the core relax, so `Ensemble` reaches every tier through this one module.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from rdkit import Chem

from rxembed.relax import ff_energies, restrained_uff  # noqa: F401  the FF tier of the refine ladder

XTB_EXE = os.environ.get("XTB_EXE", os.path.expanduser("~/bin/xtb"))
XTB_TIMEOUT = int(os.environ.get("XTB_TIMEOUT", "600"))  # s; a hung/non-converging xtb must not freeze the caller


def _run(args, cwd):
    try:
        p = subprocess.run([XTB_EXE, *args], cwd=cwd, capture_output=True, text=True, timeout=XTB_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:  # caught per-conformer by score/optimize -> dropped,
        raise RuntimeError(f"xtb {' '.join(args)} timed out after {XTB_TIMEOUT}s") from e  # not an infinite hang
    if p.returncode:
        raise RuntimeError(f"xtb {' '.join(args)} failed:\n{p.stderr[-400:]}")
    return p.stdout


def _energy(out):
    for line in out.splitlines():
        if "TOTAL ENERGY" in line.upper():
            for tok in line.split():
                try:
                    return float(tok)
                except ValueError:
                    pass
    raise RuntimeError("no total energy in xtb output")


def _read_gradient(path, natoms):
    with open(path) as fh:
        rows = fh.read().splitlines()
    e = float(rows[1].split("SCF energy =")[1].split()[0])
    g = np.array([[float(x) for x in rows[2 + natoms + i].split()] for i in range(natoms)])
    return e, g


def singlepoint(mol, conf_id=-1, method="gxtb", solvent=None, charge=0, grad=False):
    """(energy_Eh, gradient[N,3] in Eh/bohr or None). `solvent` (ALPB) is GFN2-only."""
    # coord file first: --alpb otherwise eats the next token as its reference state
    args = ["m.xyz", f"--{method}", "-c", str(charge)]
    if grad:
        args.append("--grad")
    if solvent:
        args += ["--alpb", solvent]
    with tempfile.TemporaryDirectory() as td:
        Chem.MolToXYZFile(mol, os.path.join(td, "m.xyz"), confId=conf_id)
        out = _run(args, td)
        if grad:
            return _read_gradient(os.path.join(td, "gradient"), mol.GetNumAtoms())
        return _energy(out), None


_OPT_LEVELS = ("sloppy", "crude", "loose", "normal", "tight", "vtight", "extreme")


def _read_xtbopt(path):
    """(optimised coords [N,3] Å, energy_Eh) from an ``xtbopt.xyz`` (energy is on the comment line)."""
    with open(path) as fh:
        rows = fh.read().splitlines()
    n = int(rows[0])
    toks = rows[1].split()
    if "energy:" not in toks:  # be loud about a parse failure, don't return None
        raise RuntimeError(f"no 'energy:' on the xtbopt.xyz comment line: {rows[1]!r}")
    e = float(toks[toks.index("energy:") + 1])
    coords = np.array([[float(x) for x in rows[2 + i].split()[1:4]] for i in range(n)])
    return coords, e


def xtb_optimize(mol, conf_id=-1, method="gxtb", level="normal", fix=(), charge=0, solvent=None):
    """Constrained xtb geometry optimisation; return ``(coords[N,3] Angstrom, energy_Eh)``.

    `level` is loose/normal/tight/vtight; `fix` is 0-based atom indices held fixed (the frozen TS core),
    everything else relaxing. `solvent` (ALPB) is honoured for ``gfn2``; g-xTB has no ALPB, so a solvated
    optimisation must use ``gfn2`` (we refuse rather than silently optimise in gas).
    """
    if level not in _OPT_LEVELS:
        raise ValueError(f"unknown xtb opt level {level!r}; expected one of {_OPT_LEVELS}")
    if method == "gxtb" and solvent:
        raise ValueError(
            "g-xTB has no implicit solvent (ALPB); for a solvated optimisation use "
            "refine='gfn2' (or optimise in gas and score(solvent=...) for a single-point "
            "correction); not silently optimising gas-phase"
        )
    args = ["m.xyz", "--opt", level, f"--{method}", "-c", str(charge)]
    if solvent and method in ("gfn2", "gfnff"):
        args += ["--alpb", solvent]
    with tempfile.TemporaryDirectory() as td:
        Chem.MolToXYZFile(mol, os.path.join(td, "m.xyz"), confId=conf_id)
        if fix:
            ids = ",".join(str(i + 1) for i in sorted(set(fix)))  # xtb $fix is 1-indexed
            with open(os.path.join(td, "xtb.inp"), "w") as fh:
                fh.write(f"$fix\n   atoms: {ids}\n$end\n")
            args += ["--input", "xtb.inp"]
        _run(args, td)
        return _read_xtbopt(os.path.join(td, "xtbopt.xyz"))


def composite(mol, conf_id=-1, solvent=None, charge=0, grad=False):
    """g-xTB gas + GFN2 solvent correction. Returns (energy_Eh, gradient or None)."""
    e, g = singlepoint(mol, conf_id, "gxtb", None, charge, grad)
    if not solvent:
        return e, g
    es, gs = singlepoint(mol, conf_id, "gfn2", solvent, charge, grad)
    eg, gg = singlepoint(mol, conf_id, "gfn2", None, charge, grad)
    return e + (es - eg), (g + (gs - gg) if grad else None)


def xtb_energy(mol, conf_id=-1, method="gxtb", solvent=None, charge=0, grad=False):
    """Dispatch a ranking energy. method 'gxtb' / 'gfn2' / 'gfnff'; solvent (ALPB) via GFN2 or GFN-FF.

    gxtb + solvent → g-xTB gas + GFN2 solvent correction (g-xTB has no ALPB);
    gfn2 / gfnff + solvent → native ALPB; any method gas-phase otherwise.
    """
    if method == "gxtb" and solvent:
        return composite(mol, conf_id, solvent=solvent, charge=charge, grad=grad)
    return singlepoint(mol, conf_id, method, solvent if method in ("gfn2", "gfnff") else None, charge, grad)


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

    GFN-FF ('gfnff') is a force field with dispersion and H-bond terms, so it gives a real NCI-aware
    energy or relax at FF cost: the right tier for ranking an `mc(explore=)` pool before a g-xTB opt.
    """

    def __init__(self, method="gxtb", solvent=None, charge=0):
        self.method, self.solvent, self.charge = method, solvent, charge

    def energy(self, mol, conf_id=-1):
        """Single-point energy in Hartree via the xtb executable."""
        return xtb_energy(mol, conf_id, self.method, self.solvent, self.charge)[0]

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        """Constrained xtb optimisation -> (coords[N,3] Angstrom, energy_Eh)."""
        return xtb_optimize(mol, conf_id, self.method, level, fix, self.charge, self.solvent)


class ASE(Calculator):
    """Wrap an optional ASE calculator without making ASE a package dependency."""

    def __init__(self, ase_calc):
        self.calc = ase_calc

    def energy(self, mol, conf_id=-1):
        """Single-point energy in Hartree via the wrapped ASE calculator."""
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError("ASE.energy needs ase; pip install ase or the package providing your calculator") from exc

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
