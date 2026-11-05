"""Run openconf's torsional Monte-Carlo search on an ensemble's Mol; never read a species.

Unconstrained, openconf's ``generate_conformers`` seeds and searches on its own. With held atoms,
``generate_conformers_from_pose`` searches around rxembed's seeds with those atoms pose-frozen, so no held contact
breaks, at the price of low-mode following and every ring or global move: only free rotors move. It needs the
``search`` extra, which `available` reports.
"""

from __future__ import annotations

import dataclasses
import importlib.util

from rdkit import Chem

from rxembed.utils import require


def available() -> bool:
    """Return True when the optional openconf backend is installed."""
    return importlib.util.find_spec("openconf") is not None


def config(preset, seed):
    """Return openconf's `preset` ConformerConfig, its random seed `seed` unless that is None."""
    cfg = require("openconf.config", "search").preset_config(preset)
    return cfg if seed is None else dataclasses.replace(cfg, random_seed=seed)


def search(mol, cons, cfg):
    """Return the ids of the conformers one openconf search under `cfg` adds to `mol`, its atom order kept.

    The search is pose-frozen on the atoms `cons` holds (`Constraints.constrained_atoms`), unconstrained when none.
    """
    openconf = require("openconf", "search")

    held = sorted(cons.constrained_atoms())
    if held:
        ens = openconf.generate_conformers_from_pose(mol, held, config=cfg)
    else:
        ens = openconf.generate_conformers(mol, config=cfg, add_hs=False)
    if ens.mol.GetNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError("openconf changed the atom set; cannot adopt conformers")
    return [mol.AddConformer(Chem.Conformer(ens.mol.GetConformer(cid)), assignId=True) for cid in ens.conf_ids]
