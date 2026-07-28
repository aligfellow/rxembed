"""Torsional Monte-Carlo (MCMM) search via openconf (rowansci).

openconf is a *complete* generator — ETKDG seeding + a rich move set (single/multi/correlated rotor,
global-shake, ring-flip, crankshaft, ring-KIC, amide-flip), a CrystalFF torsion library, topology-aware
seeding and adaptive moves — strictly more exploratory than plain ETKDG. We use it two ways:

- **unconstrained**: ``generate_conformers`` drives the whole generation (its own seeding + search);
- **constrained**: ``generate_conformers_from_pose`` searches *around* rxembed's bounds-biased seed
  with the held atoms pose-frozen (position-restrained + snapped), so an NCI / TS / metal contact is
  provably never broken. The price openconf pays for that safety: **low-mode following and all
  ring/global moves are disabled in constrained mode** — only free-rotor moves run. We surface that.

`preset` (rapid|ensemble|spectroscopic|docking|analogue|macrocycle|transition_metal) sets the effort —
`transition_metal` is the metal-aware preset (metal-ligand fragment rotations); `seed`/`max_out`/`low_mode`
override single knobs; `config` is an escape hatch (an openconf ConformerConfig). Optional: needs `openconf`
(pip install rxembed[mc]); `available()` guards it.
"""

from __future__ import annotations

from rdkit import Chem

from rxembed.rdkit_embed.log import logger


def available() -> bool:
    """Return True when the optional openconf backend is importable."""
    try:
        import openconf  # noqa: F401
    except ImportError:
        return False
    return True


def _config(preset, seed, max_out, low_mode, config, constrained, **openconf_kw):
    """Resolve the openconf ConformerConfig from preset + single-knob overrides + passthrough field kwargs.

    ``openconf_kw`` are arbitrary ``ConformerConfig`` fields (from ``mc(**kwargs)``) applied verbatim — an
    unknown field raises the standard ``dataclasses.replace`` TypeError, naming it.
    """
    import dataclasses

    from openconf.config import preset_config

    cfg = config if config is not None else preset_config(preset)
    over = dict(openconf_kw)  # passthrough openconf ConformerConfig fields (kwargs win over the preset)
    if seed is not None:
        over["random_seed"] = seed
    if max_out is not None:
        over["max_out"] = max_out
    if low_mode is not None:
        if constrained and low_mode:
            logger.warning("mc: low-mode following is disabled in constrained search (openconf) — ignored")
        else:
            over["use_low_mode_following"] = low_mode
    return dataclasses.replace(cfg, **over) if over else cfg


def search(mol, cons, *, preset="ensemble", seed=None, max_out=None, low_mode=None, config=None, **openconf_kw):
    """Run openconf → new conformer ids on `mol` (atom order preserved; energies recomputed by refine).

    Pose-constrained iff any atoms are held (`cons.constrained_atoms()`). ``openconf_kw`` are passthrough
    ``ConformerConfig`` field overrides. Returns the list of added conformer ids.
    """
    import openconf

    held = sorted(cons.constrained_atoms())
    cfg = _config(preset, seed, max_out, low_mode, config, constrained=bool(held), **openconf_kw)
    if held:
        ens = openconf.generate_conformers_from_pose(mol, held, config=cfg)
    else:
        ens = openconf.generate_conformers(mol, config=cfg, add_hs=False)
    if ens.mol.GetNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError("openconf changed the atom set; cannot adopt conformers")
    return [mol.AddConformer(Chem.Conformer(ens.mol.GetConformer(cid)), assignId=True) for cid in ens.conf_ids]
