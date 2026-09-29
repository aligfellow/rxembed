"""Torsional Monte-Carlo (MCMM) search via openconf.

openconf is a complete generator (ETKDG seeding, a rich move set, a CrystalFF torsion library, and
adaptive moves), so it is strictly more exploratory than plain ETKDG. Used two ways:

- unconstrained: ``generate_conformers`` drives the whole generation, its own seeding and search;
- constrained: ``generate_conformers_from_pose`` searches around rxembed's bounds-biased seed with the
  held atoms pose-frozen, so an NCI / TS / metal contact is provably never broken. The price is that
  low-mode following and every ring or global move are disabled, leaving only free-rotor moves.

`preset` sets the effort (`transition_metal` being the metal-aware one); `seed`/`max_out`/`low_mode`
override single knobs and `config` is an escape hatch. Needs `openconf`; `available()` guards it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging

from rdkit import Chem

logger = logging.getLogger("rxembed")


def available() -> bool:
    """Return True when the optional openconf backend is installed."""
    return importlib.util.find_spec("openconf") is not None


def config(preset, seed, max_out, low_mode, config, constrained, **openconf_kw):
    """Resolve the openconf ConformerConfig from preset + single-knob overrides + passthrough field kwargs.

    ``openconf_kw`` are arbitrary ``ConformerConfig`` fields (from ``mc(**kwargs)``) applied verbatim; an
    unknown field raises the standard ``dataclasses.replace`` TypeError, naming it.
    """
    try:
        from openconf.config import preset_config
    except ImportError as exc:
        raise ImportError("config needs openconf; pip install 'rxembed[search]'") from exc

    cfg = config if config is not None else preset_config(preset)
    over = dict(openconf_kw)  # passthrough openconf ConformerConfig fields (kwargs win over the preset)
    if seed is not None:
        over["random_seed"] = seed
    if max_out is not None:
        over["max_out"] = max_out
    if low_mode is not None:
        if constrained and low_mode:
            logger.warning("mc: low-mode following is disabled in constrained search (openconf); ignored")
        else:
            over["use_low_mode_following"] = low_mode
    return dataclasses.replace(cfg, **over) if over else cfg


def search(mol, cons, cfg):
    """Run openconf with the resolved `cfg` and return the conformer ids it added to `mol` (atom order kept).

    Pose-constrained iff any atoms are held (`cons.constrained_atoms()`); resolve `cfg` with `config` under the
    same flag.
    """
    try:
        import openconf
    except ImportError as exc:
        raise ImportError("search needs openconf; pip install 'rxembed[search]'") from exc

    held = sorted(cons.constrained_atoms())
    if held:
        ens = openconf.generate_conformers_from_pose(mol, held, config=cfg)
    else:
        ens = openconf.generate_conformers(mol, config=cfg, add_hs=False)
    if ens.mol.GetNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError("openconf changed the atom set; cannot adopt conformers")
    return [mol.AddConformer(Chem.Conformer(ens.mol.GetConformer(cid)), assignId=True) for cid in ens.conf_ids]
