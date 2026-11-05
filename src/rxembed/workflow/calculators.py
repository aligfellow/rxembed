"""Compute a conformer's real energy: the xtb executable and the `Calculator` interface.

`resolve` maps a spec to a Calculator: ``'ff'`` to None (the force field), ``'gxtb'``, ``'gfn2'`` or ``'gfnff'`` to
`XTB`, and a Calculator to itself. Wrap an ASE calculator (MACE, AIMNet2, ORCA, ...) with `ASE`, which imports ASE
only when used. `XTB` runs ``xtb`` on PATH, or ``$XTB_EXE``. g-xTB has no ALPB parameters, so its solvent comes from
a GFN2 correction::

    E = E_gxtb(gas) + [E_gfn2(solvent) - E_gfn2(gas)]
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from rdkit import Chem

XTB_TIMEOUT = int(os.environ.get("XTB_TIMEOUT", "600"))  # s: a hung or non-converging xtb must not freeze the caller


def _run(args, cwd):
    """Return xtb's stdout; a failed or timed-out run raises RuntimeError, a missing executable FileNotFoundError."""
    exe = os.environ.get("XTB_EXE", "xtb")  # read per run, so os.environ["XTB_EXE"] set in Python takes effect
    try:
        p = subprocess.run([exe, *args], cwd=cwd, capture_output=True, text=True, timeout=XTB_TIMEOUT, check=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'no xtb executable {exe!r}: put xtb on PATH or set os.environ["XTB_EXE"]') from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"xtb {' '.join(args)} timed out after {XTB_TIMEOUT}s") from e
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


def singlepoint(mol, conf_id=-1, method="gxtb", solvent=None):
    """Return the xtb energy in Eh at the Mol's formal charge, in `solvent` (ALPB) when given."""
    # coord file first: --alpb otherwise eats the next token as its reference state
    args = ["m.xyz", f"--{method}", "-c", str(Chem.GetFormalCharge(mol))]
    if solvent:
        args += ["--alpb", solvent]
    with tempfile.TemporaryDirectory() as td:
        Chem.MolToXYZFile(mol, os.path.join(td, "m.xyz"), confId=conf_id)
        return _energy(_run(args, td))


_OPT_LEVELS = ("sloppy", "crude", "loose", "normal", "tight", "vtight", "extreme")


def _read_xtbopt(path):
    """Return the coordinates [N, 3] in Å and the energy in Eh of an ``xtbopt.xyz``, the energy on its comment line."""
    with open(path) as fh:
        rows = fh.read().splitlines()
    n = int(rows[0])
    toks = rows[1].split()
    if "energy:" not in toks:
        raise RuntimeError(f"no 'energy:' on the xtbopt.xyz comment line: {rows[1]!r}")
    e = float(toks[toks.index("energy:") + 1])
    coords = np.array([[float(x) for x in rows[2 + i].split()[1:4]] for i in range(n)])
    return coords, e


_EV_TO_HARTREE = 1 / 27.211386245988


class Calculator:
    """Compute one conformer's energy in Hartree, and optionally optimise it with atoms held: `XTB` and `ASE`'s base.

    An implementation gives ``energy()``, ``optimize()`` where `Ensemble.optimize` runs it, and ``name`` where its class
    name does not name the model. The charge is the formal charge of the Mol given. A conformer the model cannot
    compute raises RuntimeError, which drops that conformer alone; any other error raises.
    """

    @property
    def name(self):
        """Return the name `Ensemble.energy_kind` gives this calculator's energies: its class name."""
        return type(self).__name__

    def energy(self, mol, conf_id=-1) -> float:
        """Return the single-conformer energy in Hartree."""
        raise NotImplementedError(f"{self.name} has no energy model")

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        """Return the coordinates [N, 3] in Å and energy in Eh of one conformer optimised with the atoms `fix` held."""
        raise NotImplementedError(f"{self.name} has no geometry optimiser; optimize with 'gxtb', 'gfn2' or 'gfnff'")


class XTB(Calculator):
    """Run g-xTB, GFN2 or GFN-FF (`method`) through the xtb executable, in an ALPB `solvent` when one is given.

    GFN-FF is a force field with dispersion and H-bond terms: an NCI-aware energy or relax at force-field cost, for
    ranking an ``mc(explore=True)`` pool before a g-xTB optimisation.
    """

    def __init__(self, method="gxtb", solvent=None):
        self.method, self.solvent = method, solvent

    @property
    def name(self):
        """Return the xtb method, with the solvent it scores in."""
        return f"{self.method} in {self.solvent}" if self.solvent else self.method

    def energy(self, mol, conf_id=-1):
        """Return the single-point energy in Hartree, in `solvent` by ALPB when one is set.

        GFN2 and GFN-FF take ALPB natively; g-xTB has none, so its solvated energy is its gas energy plus the GFN2
        solvent correction.
        """
        if self.method == "gxtb" and self.solvent:
            gas = singlepoint(mol, conf_id, "gxtb", None)
            return gas + (singlepoint(mol, conf_id, "gfn2", self.solvent) - singlepoint(mol, conf_id, "gfn2", None))
        return singlepoint(mol, conf_id, self.method, self.solvent)

    def optimize(self, mol, conf_id=-1, level="normal", fix=()):
        """Optimise one conformer at `level`, holding the 0-based atoms `fix`; return ``(coords[N,3] Å, energy_Eh)``.

        `level` is one of `_OPT_LEVELS`. GFN2 and GFN-FF optimise in `solvent` by ALPB; g-xTB has none, so a solvated
        g-xTB optimisation is refused rather than run in gas.
        """
        if level not in _OPT_LEVELS:
            raise ValueError(f"unknown xtb opt level {level!r}; expected one of {_OPT_LEVELS}")
        if self.method == "gxtb" and self.solvent:
            raise ValueError(
                "g-xTB has no implicit solvent; optimise in gas, then score('gxtb', solvent=...) adds the GFN2 "
                "correction"
            )
        args = ["m.xyz", "--opt", level, f"--{self.method}", "-c", str(Chem.GetFormalCharge(mol))]
        if self.solvent:
            args += ["--alpb", self.solvent]
        with tempfile.TemporaryDirectory() as td:
            Chem.MolToXYZFile(mol, os.path.join(td, "m.xyz"), confId=conf_id)
            if fix:
                ids = ",".join(str(i + 1) for i in sorted(set(fix)))  # xtb $fix is 1-indexed
                with open(os.path.join(td, "xtb.inp"), "w") as fh:
                    fh.write(f"$fix\n   atoms: {ids}\n$end\n")
                args += ["--input", "xtb.inp"]
            _run(args, td)
            return _read_xtbopt(os.path.join(td, "xtbopt.xyz"))


class ASE(Calculator):
    """Wrap an optional ASE calculator without making ASE a package dependency."""

    def __init__(self, ase_calc):
        self.calc = ase_calc

    @property
    def name(self):
        """Return the class name of the wrapped ASE calculator."""
        return type(self.calc).__name__

    def energy(self, mol, conf_id=-1):
        """Return the energy in Hartree from the wrapped ASE calculator, given the Mol's charge where ASE reads one."""
        try:
            from ase import Atoms  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError("ASE.energy needs ase; pip install ase or the package providing your calculator") from exc

        conf = mol.GetConformer(conf_id)
        atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=conf.GetPositions())
        atoms.set_initial_charges([a.GetFormalCharge() for a in mol.GetAtoms()])
        atoms.info["charge"] = Chem.GetFormalCharge(mol)  # the other ASE convention for a system charge
        atoms.calc = self.calc
        return atoms.get_potential_energy() * _EV_TO_HARTREE


def resolve(refine, solvent=None):
    """Return the Calculator a refine spec names, or None for the force field; only an xtb spec takes `solvent`."""
    if refine in ("gxtb", "gfn2", "gfnff"):
        return XTB(refine, solvent)
    if refine not in (None, "ff") and not isinstance(refine, Calculator):
        raise ValueError(f"unknown refine {refine!r}; use 'ff', 'gxtb', 'gfn2', 'gfnff' or rx.ASE(an ASE calculator)")
    if solvent is not None:
        raise ValueError("solvent= goes with an xtb spec; 'ff' has none and a calculator carries its own")
    return None if refine in (None, "ff") else refine
