"""The pipeline's public verbs: what a user calls, and what comes back.

`Ensemble` and `EnsembleSet` live in `ensemble.py`; the routing behind these verbs lives in `dispatch.py`.
Keeping the verbs here is what lets both of those import in one direction only.
"""

from __future__ import annotations

import logging

from rxembed.embed import BASE_STIFFNESS as _BASE_STIFFNESS
from rxembed.metal_isomers import Isomer

from . import calculators as _refine
from .dispatch import (
    _attach_stereo,
    _embed_dispatch,
    _normalize,
    _stereo_enumerated_embed,
    _stereo_expand,
    _template_to_fix,
    _validate_stereo,
)

# The workflow name accepts strings and paths; the engine name accepts a Mol.
from .dispatch import enumerate_isomers as metal
from .ensemble import Ensemble, EnsembleSet

logger = logging.getLogger("rxembed")

__all__ = ["Ensemble", "EnsembleSet", "embed", "metal", "minimize", "wrap"]


def embed(
    source,
    *,
    metal=None,
    fix=None,
    constrain=None,
    template=None,
    contacts=None,
    coordinate=None,
    charge=0,
    n=None,
    seed=0xF00D,
    threads=0,
    knowledge=True,
    stereo="racemic",
):
    """Embed conformers (optionally constrained), returning an `Ensemble`, an `EnsembleSet`, or a `list`.

    Return shape follows the input: an `Ensemble` for one molecule; an `EnsembleSet` (to `.select` from) when
    the input is inherently several poses (metal isomers, NCI modes, ambiguous ``coordinate=``, a racemate);
    a ``list[EnsembleSet]`` for ``stereo='separate'``, one per configuration, which does not chain.

    A constrained embed comes back relaxed into its windows; only the geometry moves.

    ``fix`` / ``constrain`` are the core verbs, documented in the `constraints` module docstring. ``template``
    is sugar for a coords-``fix``: ``(reference, SMARTS_or_map)``. A SMARTS matches the target and a Mol or
    Ensemble reference; an .xyz path or (N,3) array needs ``{target_i: ref_i}``.

    ``stereo=`` governs undefined chirality of a coordinate-free input (point R/S + double-bond E/Z; defined
    centres held, meso dropped, chiral-at-P included, metal never enumerated):

    - ``'racemic'`` (default): every stereoisomer embedded with equal effort, folded into one `EnsembleSet`
      and never energy-pruned against the others. A fully-defined input is untouched.
    - ``'separate'``: a ``list[EnsembleSet]``, one per configuration.
    - ``'free'``: one embed, stereocentres left to the ETKDG seed.

    For a geometry input stereo is already 3D-defined, so ``stereo=`` instead tunes the preservation filter
    for chirality the embed cannot keep. ``contacts=`` reaches `nci_modes`; ``metal=`` / ``coordinate=`` reach
    `rx.metal`.
    """
    _validate_stereo(stereo)

    dispatch_kw = {
        "metal": metal,
        "fix": fix,
        "constrain": constrain,
        "template": template,
        "contacts": contacts,
        "coordinate": coordinate,
        "charge": charge,
        "n": n,
        "seed": seed,
        "threads": threads,
        "knowledge": knowledge,
        "stereo": stereo,  # the metal load-in (enumerate_isomers) reads it; the organic path uses _stereo_expand
    }
    expanded = _stereo_expand(source, stereo)  # undefined-stereocentre racemate load-in (or None)
    if expanded is not None:
        result = _stereo_enumerated_embed(expanded, stereo, dispatch_kw)
    else:
        result = _embed_dispatch(source, **dispatch_kw)
        _attach_stereo(result, source, charge, stereo)
    return _relax_embedded(result)


def _relax_embedded(result):
    """Relax every embedded candidate into its windows: the one seam `embed` returns through.

    `EnsembleSet` subclasses `list`, so the plain-list branch must come last: an earlier
    ``isinstance(…, list)`` would downgrade an EnsembleSet to a bare list.
    """
    if isinstance(result, EnsembleSet):
        return result._map("_relax_into_windows")
    if isinstance(result, Ensemble):
        return result._relax_into_windows()
    return [r._map("_relax_into_windows") for r in result]  # stereo='separate' -> a plain list of EnsembleSet


def minimize(source, *, fix=None, constrain=None, template=None, charge=0, stiffness=_BASE_STIFFNESS):
    """Relax an existing structure toward ``fix``/``constrain`` targets: the search-free companion to `embed`.

    Same vocabulary as `embed` but no conformer search. Wraps the input geometry, grafts any coordinate-``fix``
    core, and runs the restrained UFF pull toward the targets. Needs an input geometry: an .xyz, a Mol with a
    conformer, or a metal `Isomer` whose Mol has one.

        rx.minimize('mol.xyz', fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    from rxembed.embed import prepare_relax

    if isinstance(source, Isomer):
        spec, mol = source, source.mol
        has_geom = mol.GetNumConformers() > 0
    else:
        mol, has_geom = _normalize(source, charge)
        spec = mol
    if not has_geom:
        raise ValueError(
            "minimize() relaxes an existing geometry; give an .xyz or a Mol with a conformer, not a SMILES"
        )
    if template is not None:  # the same sugar `embed` dissolves: a reference core IS a coordinate fix
        # after the normalise above, so a `fix=[atoms]` list resolves against the geometry just read
        fix = _template_to_fix(template, fix, mol.GetConformer().GetPositions(), mol)
    # The surrogate / sphere-hold / graft assembly is the core's (`rxembed.embed.minimize` is the same call
    # with a `Conformers` result); this only wraps it as an `Ensemble` so the pipeline verbs chain off it.
    mol, ids, cons, iso = prepare_relax(spec, fix=fix, constrain=constrain)
    return Ensemble(mol, ids, cons, iso).minimize(stiffness=stiffness)


def wrap(mol, ids=None, *, energies=None, minimized=False):
    """Wrap an existing RDKit Mol (with conformers) as an Ensemble to give it rxembed's methods.

    By default the wrapped geometries are treated as un-relaxed, so `prune()`/`representatives()`/`lowest()`
    run one FF `minimize()` first, which moves atoms. If your conformers are already optimised and must not be
    disturbed (a CREST / xtb / DFT output), pass ``minimized=True``: the relax is skipped and energies come
    from ``energies=`` (a list aligned to `ids`, or a dict), or as single points if omitted.
    """
    ids = ids if ids is not None else [c.GetId() for c in mol.GetConformers()]
    ens = Ensemble(mol, list(ids))
    if energies is not None:
        if not isinstance(energies, dict) and len(energies) != len(ids):
            raise ValueError(f"wrap(energies=) has {len(energies)} values but there are {len(ids)} conformers")
        if isinstance(energies, dict):
            ens.energies = dict(energies)
        else:
            ens.energies = {i: float(e) for i, e in zip(ids, energies, strict=False)}
    if minimized:
        if not ens.energies:
            e = _refine.ff_energies(mol, minimize=False)  # single-point, geometry untouched
            by_id = {c.GetId(): float(e[k]) for k, c in enumerate(mol.GetConformers())}
            ens.energies = {i: by_id[i] for i in ids if i in by_id}
        ens._minimized = True
    return ens
