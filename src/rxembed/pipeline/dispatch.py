"""Expand pipeline inputs into candidates and run each through one embed executor.

This module owns the public `embed`, `minimize` and `enumerate_isomers` (the facade's `metal`) verbs. It adapts
sources, templates, stereo, metal identities, vacant-site seating and NCI modes. Those axes only produce candidate
data. `_execute` then prepares constraints and calls the shared core seeding seam once per candidate.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass

from rdkit import Chem

from rxembed import metal_enumeration
from rxembed.bounds import resolve_params
from rxembed.constraints import Constraints, compose_soft, resolve_atom, resolve_core, template_to_fix
from rxembed.embed import (  # module functions, not the package facade
    BASE_STIFFNESS,
    EmbeddingError,
    prepare,
    prepare_relax,
    remedy,
    require_seed_count,
    seed_conformers,
    stereo_donor_bonds,
)
from rxembed.metal_core import VACANT, donor_chirality_sign, metal_index, metal_indices, reject_metal_bonds
from rxembed.metal_isomer import Isomer, from_geometry
from rxembed.metal_polyhedron import describe
from rxembed.metal_smiles import parse_smiles
from rxembed.relax import MAX_ITERS
from rxembed.stereo import enumerate_unassigned

from .ensemble import Ensemble, EnsembleSet
from .nci import Contact, auto_binding_modes
from .perceive import read_xyz
from .stereo_check import signature

logger = logging.getLogger("rxembed")


@dataclass
class _Candidate:
    """Carry one fully selected identity into the shared embed executor."""

    spec: object
    fix: object
    tag: dict
    coordinated: tuple = ()


def _default_stereo(source, stereo):
    """Retain stereo from coordinates by default; enumerate unspecified stereo from a graph."""
    if stereo is not None:
        return stereo
    if isinstance(source, Isomer):
        has_geometry = bool(source.mol.GetNumConformers())
    elif isinstance(source, Chem.Mol):
        has_geometry = bool(source.GetNumConformers())
    else:
        path = os.fspath(source) if isinstance(source, os.PathLike) else source
        has_geometry = isinstance(path, str) and path.lower().endswith(".xyz")
    return "all" if has_geometry else "unassigned"


def _validate_relax(max_iters, trajectory=False, n=None):
    """Require a positive restrained-UFF iteration cap and a trajectory request for exactly one conformer."""
    if not isinstance(trajectory, bool):
        raise TypeError("trajectory must be True or False")
    if trajectory and n != 1:
        raise ValueError("trajectory=True requires n=1")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")


def _normalize(source, charge=0):
    """Normalise SMILES / an .xyz path / an RDKit Mol to ``(mol with Hs, has_geometry)``.

    `charge` is used for xyz bond perception (a charged metal complex / ion).
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path, not just str
    if isinstance(source, Chem.Mol):
        mol = source
    elif isinstance(source, str) and source.lower().endswith(".xyz"):
        mol = read_xyz(source, charge)
    else:
        mol = parse_smiles(source)
    has_geom = mol.GetNumConformers() > 0
    return Chem.AddHs(mol, addCoords=has_geom), has_geom


def _contact_constraints(mol, contacts):
    """Resolve contact inputs through the validated soft-constraint path."""
    distances, angles = {}, {}
    if contacts is None:
        return Constraints()
    items: list[Contact] = []
    if isinstance(contacts, Contact):
        items = [contacts]
    elif isinstance(contacts, dict):
        vals = [v for v in contacts.values() if isinstance(v, Contact)]
        if vals and len(vals) != len(contacts):  # {label: Contact} must be all Contacts
            raise TypeError("contacts: dict mixes Contact and non-Contact values")
        if vals:
            items = vals
        else:
            distances.update(contacts)  # a raw {(i,j):(lo,hi)} distance dict
    else:
        values = list(contacts)
        items = [c for c in values if isinstance(c, Contact)]
        if len(items) != len(values):
            raise TypeError("contacts: list must contain Contact objects (from rx.nci_candidates)")
    for ct in items:
        distances.update(ct.distances)
        angles.update(ct.angles)
    if not distances and not angles:
        return Constraints()
    return resolve_core(mol, constrain={**distances, **angles}, has_geometry=bool(mol.GetNumConformers()))[0]


def _coordination_choices(iso, coordinate, nvac):
    """Resolve ``coordinate=`` to the donor set(s) to seat: one choice, or several for an ambiguous SMARTS."""
    if coordinate is None:
        return [None]
    if isinstance(coordinate, (list, tuple)):  # explicit: one spec per vacancy
        return [[resolve_atom(iso.mol, s) for s in coordinate]]
    if isinstance(coordinate, str):  # a SMARTS, which may match several atoms
        pattern = Chem.MolFromSmarts(coordinate)
        if pattern is None:
            raise ValueError(
                f"coordinate={coordinate!r} is not a valid SMARTS; pass a SMARTS pattern, an atom "
                "index, or a list with one spec per vacancy"
            )
        ms = [m[0] for m in iso.mol.GetSubstructMatches(pattern)]
        if not ms:
            raise ValueError(f"coordinate={coordinate!r} matched no atoms")
        if len(ms) == 1:
            return [[ms[0]]]
        if nvac == 1:  # ambiguous + 1 pocket -> one mode per donor
            logger.info(
                "coordinate=%r matched %d atoms; embedding %d modes, one per donor",
                coordinate,
                len(ms),
                len(ms),
            )
            return [[a] for a in ms]
        raise ValueError(  # ambiguous + several pockets -> can't map
            f"coordinate={coordinate!r} matched {len(ms)} atoms for {nvac} vacant site(s); "
            f"name them explicitly as a list, e.g. ['[OX1]', '[OX2]']"
        )
    return [[resolve_atom(iso.mol, coordinate)]]  # an int index


def _execute(spec, *, fix, constrain, contacts, n, params):
    """Compile and seed one candidate, returning its pipeline ensemble."""
    mol, cons, iso, graft_ref = prepare(spec, fix=fix, constrain=constrain, params=params)
    soft = _contact_constraints(mol, contacts)
    cons = compose_soft(cons, soft)
    if iso is not None and soft.is_constrained:
        accepted = set().union(*cons.contacts)
        active = Constraints(
            planes=soft.planes,
            distances={key: value for key, value in soft.distances.items() if key in accepted},
            angles={key: value for key, value in soft.angles.items() if key in accepted},
            dihedrals={key: value for key, value in soft.dihedrals.items() if key in accepted},
        )
        if active.is_constrained:
            # Resolve ownership first: an ignored contact must not change an unrelated model preference.
            mol, cons, iso, graft_ref = prepare(spec, fix=fix, constrain=constrain, external=active, params=params)
            cons = compose_soft(cons, soft)
    mol, ids, target = seed_conformers(mol, cons, iso, n, params, graft_ref=graft_ref)
    require_seed_count(ids, target, iso)
    ens = Ensemble(mol, list(ids), cons, iso, params=params)
    if iso is not None:
        for donor, metal_idx in iso.donor_bonds:
            ens.sphere.setdefault(metal_idx, []).append(donor)
        if ids:
            donors = sorted({donor for donor, _metal_idx in stereo_donor_bonds(mol, iso)})
            ens.donor_hand = {
                d: donor_chirality_sign(
                    mol,
                    ids[0],
                    d,
                    [metal for donor, metal in iso.donor_bonds if donor == d],
                )
                for d in donors
            }
    return ens


def _expand_isomers(isomers, coordinate, fix):
    """Expand selected metal identities by vacant-site seating without embedding them."""
    out = []
    for iso in isomers:
        choices = _coordination_choices(iso, coordinate, iso.vertices.count(VACANT))
        for atoms in choices:
            candidate = iso.seat_vacancies(atoms) if atoms is not None else iso
            tag = {
                "geometry": candidate.geometry,
                "arrangement": candidate.arrangement,
                "chirality": candidate.chirality,
                "label": candidate.label,
            }
            if candidate.stereo_label:
                tag["stereo"] = candidate.stereo_label
            if atoms is not None and len(choices) > 1:
                tag["coordinate"] = atoms[0]
            out.append(_Candidate(candidate, fix, tag, coordinated=tuple(atoms or ())))
    return out


def _contact_modes(source, contacts, params):
    """Return explicit contacts or discover independent automatic contact modes."""
    if contacts != "auto":
        return [(None, contacts)]
    disc = source.mol if isinstance(source, Isomer) else source
    try:
        modes = auto_binding_modes(disc, seed=params.seed)
    except ImportError:  # a missing extra names its own remedy
        raise
    except Exception as err:
        raise ValueError(
            f"contacts='auto' could not discover binding modes ({type(err).__name__}: {err}); "
            "pass a geometry to discover from (a multi-fragment SMILES or an .xyz), or give "
            "contacts= explicitly"
        ) from err
    if not modes:
        logger.info("contacts='auto': no inter-fragment NCI binding mode detected -> plain embed")
        return [(None, None)]
    logger.info("contacts='auto': %d binding modes", len(modes))
    return list(modes.items())


def _stereo_filter(source, charge, stereo):
    """Return the chirality filter ``(spec, reference signature)`` each embedded ensemble keeps, or ``None``.

    The default modes engage ``'preserve'`` only when the input geometry carries chirality
    the embed can't keep (planar/axial/helical); a SMILES or a stereo-less input is left untouched.
    """
    if stereo == "free":
        return None
    ref = source.stereo_ref if isinstance(source, Isomer) else None
    if ref:
        # stereo_ref was computed with native=False (point/ez/axial are the embed's own job); a haptic
        # source also owns its planar face, so it is the state metadata here too, not preserved.
        if source.haptic:
            ref = {kind: labels for kind, labels in ref.items() if kind != "planar"}
        if not isinstance(stereo, dict) and stereo != "invert":  # an enumeration default means keep the input
            stereo = "preserve"
    if ref is None and not isinstance(source, Isomer) and source.GetNumConformers():
        try:
            ref = signature(source, charge=charge)
        except Exception as err:
            logger.warning("stereo preservation unavailable for the input geometry: %s", err)
            ref = None
    if not ref:
        return None
    # An enumerating mode keeps only planar chirality, which the embed cannot re-enumerate: axial reads
    # differently every rotamer on a labile bond, and point R/S is the embed's own job. Any other mode is the
    # filter itself.
    if stereo in ("racemic", "separate"):
        spec = {"planar": "preserve", "default": "free"} if "planar" in set(ref) else None
    else:
        spec = stereo
    return None if spec is None else (spec, ref)


_STEREO_CAP = 32  # max stereoisomers embedded per source before truncating (a loud-logged safety valve)


def _stereo_variants(source, stereo):
    """Return source variants and whether organic stereo was expanded.

    Geometry inputs and metal complexes remain one source. Coordinate-free organic graphs expand undefined
    stereochemistry before every other candidate axis.
    """
    if (
        stereo not in ("unassigned", "racemic", "separate")
        or isinstance(source, Isomer)
        or source.GetNumConformers() > 0
        or metal_index(source) is not None
    ):
        return [(source, "")], False
    expanded = enumerate_unassigned(source, cap=_STEREO_CAP, include="all" if stereo == "racemic" else ())
    variants, n_unassigned, total, unresolved = expanded
    if n_unassigned == 0:
        return [(source, "")], False
    labels = ", ".join(lbl or "achiral" for _, lbl in variants)
    if total > _STEREO_CAP:
        logger.warning(
            "stereo=%r: embedding %d of %d isomers; assign the stereo in the SMILES or pass stereo='free': [%s]",
            stereo,
            len(variants),
            total,
            labels,
        )
    else:
        logger.info(
            "stereo=%r: %d undefined stereocentre(s) -> embedding %d stereoisomer(s) [%s]",
            stereo,
            n_unassigned,
            len(variants),
            labels,
        )
    if unresolved:
        logger.warning(
            "stereo=%r: %d stereo element(s) not enumerable and 3D-measurable by RDKit; embedded without a hand",
            stereo,
            unresolved,
        )
    return variants, True


def _resolve_template(template, fix, own=None, target=None):
    """Fold ``template=`` into a coordinate ``fix``, extending the core resolver with the pipeline's `Ensemble`.

    The core takes a Mol, an ``.xyz`` path or an (N, 3) array; this adds an `Ensemble`, handing over the
    positions of its first tracked conformer. An optional SMARTS is matched once on the target and reference;
    an xyz or coordinate array uses an explicit index map. The template is read for coordinates and never
    perceived, which lets a hypervalent reacting core serve as a reference.

    ``own`` is the source's own coordinates, which a ``fix=[atoms]`` list beside the template is resolved
    against; the caller must read them from the *normalised* source.
    """
    if isinstance(template, (tuple, list)) and len(template) == 2:  # noqa: PLR2004  template=(reference, mapping)
        reference, mapping = template
        if isinstance(reference, Ensemble):
            if not reference.ids:
                raise ValueError(
                    "a template reference Ensemble needs a tracked conformer; this one is empty or every "
                    "conformer was discarded"
                )
            cid = reference.ids[0]
            reference = Chem.Mol(reference.mol)
            chosen = Chem.Conformer(reference.GetConformer(cid))
            reference.RemoveAllConformers()
            reference.AddConformer(chosen, assignId=False)
            template = (reference, mapping)
    return template_to_fix(template, fix, own, target)


def enumerate_isomers(
    mol,
    geometry=None,
    center=None,
    fix=None,
    stereo=None,
    lengths="model",
    *,
    charge=0,
    screen=True,
    observed_only=False,
):
    """Enumerate all distinct coordination isomers as ready-to-embed `Isomer` objects (metal surrogated).

    The public `rx.metal` entry point: the two things the core enumerator does not do, namely reading
    a SMILES / ``.xyz`` path into a `Mol`, and the coordinate-derived chirality fingerprint (`stereo.signature`,
    xyzgraph) the ``stereo='preserve'`` gate compares against. See `rxembed.metal_enumeration.enumerate_isomers` for
    `geometry` / `center` / `fix` / `stereo` / `lengths` / `screen` / `observed_only`. A geometry is nameable in full
    (``'octahedral'``) or by its 3-letter code (``'OCT'``), case-insensitively. The `Isomer` stores the registry
    name and `.summary()` prints the compact code. ``charge`` is the total charge used to perceive an ``.xyz`` path.
    """
    stereo = _default_stereo(mol, stereo)
    metal_enumeration.validate_stereo(stereo)
    mol = _normalize(mol, charge)[0]
    reject_metal_bonds(mol)
    ref_sig = None  # chirality fingerprint of the input geometry (real metals), letting stereo='preserve'
    if mol.GetNumConformers() > 0:  # hold a spectator's planar/axial/helical handedness
        try:
            ref_sig = signature(mol, native=False)  # _stereo_filter only reads planar/helical
        except ImportError as err:
            logger.warning("stereo preservation unavailable for the input geometry: %s", err)
    return metal_enumeration.enumerate_isomers(
        mol,
        geometry,
        center,
        fix,
        stereo,
        stereo_ref=ref_sig,
        lengths=lengths,
        screen=screen,
        observed_only=observed_only,
    )


def _source_candidates(source, *, metal, fix, coordinate, stereo):
    """Expand one normalized Mol or selected metal Isomer into molecular identities."""
    stated = False
    if metal is None and not isinstance(source, Isomer):
        metals = metal_indices(source)
        arrangements = [metal_enumeration.stated_arrangement(source, center=center) for center in metals]
        if len(metals) > 1 and any(arrangements):
            if not all(arrangements):
                raise ValueError("a coordinate-free multi-metal embed needs one geometry note per centre")
            isomers = metal_enumeration.enumerate_isomers(source, center="all", stereo=stereo)
            if len(isomers) != 1:
                raise ValueError("stereo expansion produced several states; select one with rx.metal before embedding")
            source, stated = isomers[0], True
        elif len(metals) == 1 and arrangements[0] is not None:
            metal, stated = arrangements[0][0], True

    if coordinate is not None and metal is None and not isinstance(source, Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__}. Use contacts=/constrain= otherwise"
        )

    if metal is not None or isinstance(source, Isomer):
        if metal is None:
            isomers, pending_fix = [source], fix
        else:
            if isinstance(source, Isomer):
                raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
            stated = stated or metal_enumeration.stated_arrangement(source) is not None
            isomers = enumerate_isomers(source, metal, fix=fix, stereo=stereo)
            pending_fix = None  # identity enumeration compiled the fix once and stored it on every Isomer
        candidates = _expand_isomers(isomers, coordinate, pending_fix)
        if not candidates:
            raise ValueError(
                f"metal={metal!r} produced no feasible coordination identity; choose a compatible geometry, "
                "relax fix=, or call rx.metal(...) to inspect the enumeration"
            )
        if metal is not None and not stated:
            logger.info(
                "metal: %s -> %d distinct isomer(s); .summary() / .select() to pick one",
                ", ".join(describe(g) for g in (metal if isinstance(metal, (list, tuple)) else [metal])),
                len(isomers),
            )
        return source, candidates, metal is not None and not stated

    has_geometry = bool(source.GetNumConformers())
    if metal_index(source) is not None:
        if not has_geometry:
            raise ValueError("plain RDKit embedding does not model metals; pass metal=<geometry>")
        centre = "all" if len(metal_indices(source)) > 1 else None
        iso = from_geometry(source, center=centre)
        logger.info(
            "metal: geometry input and no metal= -> retaining the input arrangement (%s: %s)",
            describe(iso.geometry),
            iso.arrangement,
        )
        return source, _expand_isomers([iso], None, fix), False
    return source, [_Candidate(source, fix, {})], False


def _embed_mode(candidates, contact, nci_label, stereo_label, execute_kw, errors):
    """Seed every molecular identity for one contact mode, recording each `EmbeddingError` in `errors`."""
    live = EnsembleSet()
    for candidate in candidates:
        tag = dict(candidate.tag)
        if stereo_label:
            tag["stereo"] = stereo_label
        if nci_label is not None:
            tag["nci"] = nci_label
        identity = tag.get("arrangement", "molecule")
        coordinated = f" with atom(s) {list(candidate.coordinated)} coordinated" if candidate.coordinated else ""
        try:
            ens = _execute(candidate.spec, fix=candidate.fix, contacts=contact, **execute_kw)
        except EmbeddingError as err:
            errors.append(err)
            continue
        except RuntimeError as err:
            raise ValueError(f"could not embed {identity}{coordinated}: {err}") from err
        if not ens.ids:
            who = candidate.spec if isinstance(candidate.spec, Isomer) else "embed"
            reason = f"no conformer satisfied the constraints; {remedy('seeding', ens.iso, ens.cons)}"
            errors.append(EmbeddingError(f"{who}: {reason}", isomer=ens.iso))
            continue
        ens.tag = tag
        live.append(ens)
    return live


def _relax_seeded(groups, stereo_filter, trajectory, max_iters):
    """Relax every seeded candidate into its windows, recording each `EmbeddingError` in its group's `errors`.

    Return the groups that kept a candidate and every recorded error, each logged as one WARNING line. Raise
    a lone candidate's own error, or one `EmbeddingError` that counts the candidates when none embeds.
    """
    for group in groups:
        seeded, failed = list(group), []
        group.clear()
        for ens in seeded:
            ens.stereo_filter = stereo_filter
            try:
                relaxed = ens.relax_into_windows(trajectory=trajectory, max_iters=max_iters)
            except EmbeddingError as err:
                failed.append(err)
                continue
            logger.info(
                "embed[%s]: kept %d conformer(s)", ens.iso if ens.iso is not None else "molecule", len(relaxed.ids)
            )
            group.append(relaxed)
        group.errors += tuple(failed)
    kept = [group for group in groups if group]
    errors = [err for group in groups for err in group.errors]
    if not kept and len(errors) == 1:
        raise errors[0]
    for err in errors:
        logger.warning("%s", err)
    if not kept:
        failures = sum((err.failures for err in errors), Counter())
        raise EmbeddingError(f"none of {len(errors)} candidates could be built; first: {errors[0]}", failures=failures)
    return kept, errors


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
    seed=None,
    threads=None,
    params=None,
    stereo=None,
    trajectory=False,
    max_iters=MAX_ITERS,
):
    """Embed conformers (optionally constrained), returning an `Ensemble`, an `EnsembleSet`, or a `list`.

    Return shape follows the input: an `Ensemble` for one molecule; an `EnsembleSet` (to `.select` from) when
    the input is inherently several poses (metal isomers, NCI modes, ambiguous ``coordinate=``, a racemate);
    a ``list[EnsembleSet]`` for ``stereo='separate'``, one per configuration, which does not chain.

    A constrained embed comes back relaxed into its windows; only the geometry moves. ``trajectory=True``
    requires one conformer and stores the accepted restrained-UFF cleanup in ``result.trajectory``.

    Several candidates return every one that embedded, as an `EnsembleSet` whenever one failed. Each failure
    logs one warning line and stays in ``result.errors`` as an `EmbeddingError`; for ``stereo='separate'`` a
    configuration's set keeps its own, and a configuration that failed stays as an empty set. The call raises
    only when none embeds, and a lone candidate, such as one selected `Isomer`, raises its own error. An
    expanded stereoisomer or automatic contact mode is recorded the same way for any `ValueError` or
    `RuntimeError`; elsewhere those, and a native reach `TimeoutError`, abort the call.

    ``fix`` / ``constrain`` are the core verbs, documented in the `constraints` module docstring. ``template``
    is sugar for a coords-``fix``: ``(reference, SMARTS_or_map)``, matched by SMARTS or by an explicit
    ``{target_i: ref_i}`` index map. ``contacts=`` reaches `nci_modes`; ``metal=`` / ``coordinate=`` reach
    `rx.metal`.

    ``stereo=`` governs point R/S, double-bond E/Z, and native atropisomer M/P (meso dropped, chiral-at-P
    included, metal never enumerated):

    - omitted: enumerate only stereo left undefined in a graph; preserve the measured state of a geometry.
    - ``'racemic'``: enumerate every configurable graph element, including defined ones.
    - ``'separate'``: enumerate undefined elements as a ``list[EnsembleSet]``, one per configuration.
    - ``'free'``: one embed, stereocentres left to the ETKDG seed.

    For a geometry input, stereo is already 3D-defined, so ``stereo=`` instead tunes the preservation filter
    for chirality the embed cannot keep.

    ``seed`` / ``threads`` are convenience for `EmbedParams.seed` / `.threads` when every other field can stay
    at its default; give at most one of ``seed``/``threads`` or ``params``, not both. ``params`` carries the
    full sampling and model choice (including a native RDKit ``EmbedParameters``, see `rx.EmbedParams`) and
    rxembed's ``coplanar_14``/``metal_floor_relief``/``donor_orientation``/``conjugation`` ablation switches;
    explicit stereo and ``fix`` stay authoritative over the switches. ``max_iters`` caps the restrained-UFF
    relax that publishes the geometry; it does not change the DG seed or retry ladder.
    """
    _validate_relax(max_iters, trajectory, n)
    params = resolve_params(params, seed, threads)
    stereo = _default_stereo(source, stereo)
    metal_enumeration.validate_stereo(stereo)
    if not isinstance(source, Isomer):
        source = _normalize(source, charge)[0]
    own_mol = source.mol if isinstance(source, Isomer) else source
    if template is not None:
        own = own_mol.GetConformer().GetPositions() if own_mol.GetNumConformers() else None
        fix = _resolve_template(template, fix, own, own_mol)
    variants, expanded = _stereo_variants(source, stereo)
    groups, force_set = [], False
    execute_kw = {"constrain": constrain, "n": n, "params": params}
    for variant, stereo_label in variants:
        group, failed = EnsembleSet(), []
        try:
            discovery, candidates, source_set = _source_candidates(
                variant,
                metal=metal,
                fix=fix,
                coordinate=coordinate,
                stereo=stereo,
            )
            modes = _contact_modes(discovery, contacts, params)
            force_set = force_set or source_set
            for nci_label, contact in modes:
                try:
                    group.extend(_embed_mode(candidates, contact, nci_label, stereo_label, execute_kw, failed))
                except ValueError as err:
                    if nci_label is None:
                        raise
                    failed.append(EmbeddingError(f"contacts='auto' mode {nci_label!r}: {err}"))
        except (ValueError, RuntimeError) as err:
            if not expanded:
                raise
            group.clear()
            failed.append(EmbeddingError(f"stereoisomer [{stereo_label or 'achiral'}]: {err}"))
        else:
            if expanded:
                logger.info(
                    "stereo=%r: stereoisomer [%s] -> %d candidate(s)", stereo, stereo_label or "achiral", len(group)
                )
        group.errors = tuple(failed)
        groups.append(group)

    kept, errors = _relax_seeded(groups, _stereo_filter(source, charge, stereo), trajectory, max_iters)
    if stereo == "separate" and expanded:
        return groups
    if expanded and len(kept) > 1:
        logger.info(
            "stereo=%r: %d stereoisomers in one EnsembleSet; do not energy-prune across them", stereo, len(kept)
        )
    flattened = EnsembleSet(ens for group in kept for ens in group)
    flattened.errors = tuple(errors)
    return flattened if force_set or errors or len(flattened) != 1 else flattened[0]


def minimize(
    source,
    *,
    fix=None,
    constrain=None,
    template=None,
    charge=0,
    stiffness=BASE_STIFFNESS,
    max_iters=MAX_ITERS,
):
    """Relax an existing structure toward ``fix``/``constrain`` targets: the search-free companion to `embed`.

    Same vocabulary as `embed` but no conformer search. Wraps the input geometry, grafts any coordinate-``fix``
    core, and runs the restrained UFF pull toward the targets. Needs an input geometry: an .xyz, a Mol with a
    conformer, or a metal `Isomer` whose Mol has one.
    ``max_iters`` controls the restrained-UFF iteration cap.

        rx.minimize('mol.xyz', fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    _validate_relax(max_iters)

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
    if template is not None:  # `template` resolves exactly as it does in `embed`: a reference core becomes a fix
        # after the normalise above, so a `fix=[atoms]` list resolves against the geometry just read
        fix = _resolve_template(template, fix, mol.GetConformer().GetPositions(), mol)
    # The surrogate / sphere-hold / graft assembly is the core's (`rxembed.embed.minimize` is the same call
    # with a `Conformers` result); this only wraps it as an `Ensemble` so the pipeline verbs chain off it.
    mol, ids, cons, iso = prepare_relax(spec, fix=fix, constrain=constrain)
    return Ensemble(mol, ids, cons, iso).minimize(stiffness=stiffness, max_iters=max_iters)
