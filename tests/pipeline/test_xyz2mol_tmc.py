"""The vendored TMC perceiver, at the two points rxembed departs from upstream. Everything else is Jensen's.

Every bond-order search in this module returns the FIRST valence-consistent assignment it reaches, so
without the canonical numbering boundary in `get_lig_mol` the answer moves when the .xyz line order
does; the first test shuffles the input and demands one string back. The second pins the solution
RANKING that replaced upstream's take-the-first, which is what stops a ligand being left undervalent.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from rdkit import Chem

from rxembed.pipeline.xyz2mol_tmc import get_tmc_mol

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmark" / "corpus"

# `benchmark/` is a local-only harness and is gitignored (AGENTS.md), so a clean clone does not have these
# structures. The suite has to stay fully runnable without it: skip rather than error.
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason="needs the local-only benchmark/corpus")


def _perceive(path, charge=0, **kw):
    mol, _coords = get_tmc_mol(path, charge, **kw)
    return mol


def _shuffled(path, tmp_dir, seed):
    """The same molecule, same coordinates, with the atom lines in a different order."""
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    body = lines[2 : 2 + n]
    random.Random(seed).shuffle(body)
    out = tmp_dir / f"{path.stem}_{seed}.xyz"
    out.write_text("\n".join([lines[0], lines[1], *body]) + "\n")
    return out


@pytest.mark.parametrize("name", ["Ferrocene", "CisPlatin", "FeCO5", "Cis-PtCl2(en)"])
@needs_corpus
def test_canonical_under_atom_reordering(name, tmp_path):
    path = CORPUS / f"{name}.xyz"
    seen = {Chem.MolToSmiles(_perceive(path))}
    for seed in (1, 2, 3):
        seen.add(Chem.MolToSmiles(_perceive(_shuffled(path, tmp_path, seed))))
    assert len(seen) == 1, f"{name} gave {len(seen)} strings for one molecule: {sorted(seen)}"


@needs_corpus
def test_ligands_close_their_own_valences():
    mol = _perceive(CORPUS / "Ferrocene.xyz")
    rw = Chem.RWMol(mol)
    metals = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "Fe"]
    for i in sorted(metals, reverse=True):
        rw.RemoveAtom(i)
    stripped = rw.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    for frag in Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False):
        work = Chem.RWMol(frag)
        for a in work.GetAtoms():
            if a.GetAtomicNum() != 1:
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
        m = work.GetMol()
        Chem.SanitizeMol(m)  # raises on an over-valent ligand
        invented = sum(a.GetNumImplicitHs() for a in m.GetAtoms() if a.GetAtomicNum() != 1)
        assert invented == 0, f"perception left a ligand undervalent by {invented} H"
