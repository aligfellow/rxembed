"""g-xTB / GFN2 via the `xtb` executable on PATH (no tooltoad).

g-xTB has no ALPB parameters, so solvent comes from a GFN2 correction:
    E = E_gxtb(gas) + [E_gfn2(solvent) - E_gfn2(gas)]
and likewise for the gradient. Set $XTB_EXE to override the binary.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from rdkit import Chem

XTB = os.environ.get("XTB_EXE", os.path.expanduser("~/bin/xtb"))
XTB_TIMEOUT = int(os.environ.get("XTB_TIMEOUT", "600"))  # s; a hung/non-converging xtb must not freeze the kernel


def _run(args, cwd):
    try:
        p = subprocess.run([XTB, *args], cwd=cwd, capture_output=True, text=True, timeout=XTB_TIMEOUT, check=False)
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


def optimize(mol, conf_id=-1, method="gxtb", level="normal", fix=(), charge=0, solvent=None):
    """Constrained xtb geometry optimisation; return ``(coords[N,3] Angstrom, energy_Eh)``.

    `level` is loose/normal/tight/vtight; `fix` = **0-based** atom indices held FIXED (the frozen TS core),
    everything else relaxes. `solvent` (ALPB) is honoured for ``gfn2``; **g-xTB has no ALPB**, so a solvated
    optimisation must use ``gfn2`` (we refuse rather than silently optimise in gas).
    """
    if level not in _OPT_LEVELS:
        raise ValueError(f"unknown xtb opt level {level!r} — one of {_OPT_LEVELS}")
    if method == "gxtb" and solvent:
        raise ValueError(
            "g-xTB has no implicit solvent (ALPB) — for a solvated optimisation use "
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


def energy(mol, conf_id=-1, method="gxtb", solvent=None, charge=0, grad=False):
    """Dispatch a ranking energy. method 'gxtb' / 'gfn2' / 'gfnff'; solvent (ALPB) via GFN2 or GFN-FF.

    gxtb + solvent → g-xTB gas + GFN2 solvent correction (g-xTB has no ALPB);
    gfn2 / gfnff + solvent → native ALPB; any method gas-phase otherwise.
    """
    if method == "gxtb" and solvent:
        return composite(mol, conf_id, solvent=solvent, charge=charge, grad=grad)
    return singlepoint(mol, conf_id, method, solvent if method in ("gfn2", "gfnff") else None, charge, grad)
