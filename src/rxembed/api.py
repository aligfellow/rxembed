"""Compose the public verbs `isomers`, `embed`, `minimize` and `coembed`.

Each verb parses its arguments once, reads each source as a species, runs the engine once per species and contact mode,
and wraps the result: one species gives an `Ensemble` (a `Coembedded` pair for `coembed`), a sequence or several
contact modes (``contacts='auto'`` or `rx.contacts`' modes) an `EnsembleSet` (a `CoembeddedSet`) whose `.errors` keep
what failed.
"""

from __future__ import annotations

import itertools
import logging
import os
from collections import Counter

import numpy as np
from rdkit import Chem

from rxembed.engine.acceptance import EmbeddingError
from rxembed.engine.atom_mapping import in_reactant_order, reaction_atoms
from rxembed.engine.coembed import coembed_pair, coupling, least_motion
from rxembed.engine.seeding import prepare, relaxed, seeded
from rxembed.ensemble import FINAL, Coembedded, CoembeddedSet, Ensemble, EnsembleSet, SpeciesSet, species_name
from rxembed.geometry.bounds import resolve_sampling
from rxembed.geometry.constraints import is_index, resolve_core
from rxembed.geometry.relax import MAX_ITERS, CoEmbedParams
from rxembed.graph import metal_indices
from rxembed.identity.reading import (
    IsomerRequest,
    coordinate_request,
    cxsmiles,
    named_species,
    possible_species,
    read_species,
    requested_centre,
    requested_geometry,
    vacancy_seatings,
)
from rxembed.identity.smiles_io import mol_from_species, parse_smiles, read_native_tags
from rxembed.identity.species import graph_with_stereo
from rxembed.identity.stereoisomers import StereoRequest
from rxembed.metal.centres import canonical_metal_graph
from rxembed.perception.changes import stereo_filter
from rxembed.perception.contacts import contact_modes, parse_contacts
from rxembed.stereo import without_stale_point_tags
from rxembed.utils import require_mol

logger = logging.getLogger("rxembed")


def _validate_relax(max_iters, trajectory=False, n=None):
    """Require positive integer counts and a trajectory request for exactly one conformer."""
    if n is not None and (not is_index(n) or n <= 0):
        raise ValueError(f"n={n!r}: give a positive conformer count as an integer, or None for the default")
    if not isinstance(trajectory, bool):
        raise TypeError("trajectory must be True or False")
    if trajectory and n != 1:
        raise ValueError("trajectory=True requires n=1")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")


def _mol_from_source(source, verb, *, geometry=False):
    """Return `source` as the checked, explicit-H, canonical metal graph every stage trusts; refuse an .xyz.

    Stale point tags are dropped and native metal tags read into notes.
    """
    if isinstance(source, os.PathLike) or (isinstance(source, str) and source.lower().endswith(".xyz")):
        raise ValueError("read an .xyz with rx.read_xyz(path, charge) and pass its Mol, so its charge is given")
    mol = parse_smiles(source) if isinstance(source, str) else source
    mol = without_stale_point_tags(require_mol(mol, verb, geometry=geometry, takes="a SMILES or an RDKit Mol"))
    mol = Chem.AddHs(mol, addCoords=bool(mol.GetNumConformers()))
    Chem.SetDoubleBondNeighborDirections(mol)  # RDKit's writer re-reads E/Z from directions, absent or stale here
    return read_native_tags(canonical_metal_graph(mol))


def isomers(source, *, geometry=None, stereo=None, center=None, coordinate=None, observed=None, screen=True, fix=None):
    """Return every species `source` can be: its stereoisomers and, for a metal, its coordination arrangements.

    `source` is a SMILES or a Mol (read a charged .xyz with `read_xyz`). A Mol with a conformer is one species carrying
    that conformer: what it specifies, else what the conformer shows, written in the conformer's frame (a meso
    compound's tags may read as the mirrored labels). ``observed=False``, `geometry` or `stereo` enumerate around the
    conformer instead, without it (``stereo='invert'`` alone gives the mirror image of what it shows); ``observed=True``
    reads the conformer in the given `geometry`, and each species keeps only the conformers that show it. Conformers
    that read as different species raise.

    `coordinate` seats atoms in vacant sites: a SMARTS, an atom index, or one per vacancy; an ambiguous SMARTS with one
    vacancy gives one species per matching atom. `center` selects one metal of several. ``screen=False`` keeps
    arrangements beyond the model's reach screens, which may not embed. `fix` takes what `rx.embed` takes and keeps only
    the species its pinned atoms agree with, so `rx.embed` accepts each under that fix; it raises when none is left, and
    pinning a sphere atom lifts the screens.
    """
    mol = _mol_from_source(source, "rx.isomers")
    if observed and not mol.GetNumConformers():
        raise ValueError("observed=True reads a conformer, and this Mol has none")
    if not metal_indices(mol) and any(choice is not None for choice in (geometry, center, coordinate, fix)):
        raise ValueError("geometry=, center=, coordinate= and fix= choose a metal arrangement; this species has none")
    if not isinstance(screen, bool):
        raise TypeError("screen must be a bool")
    request = IsomerRequest(
        stereo=StereoRequest.parse(stereo).on(mol),
        pins=resolve_core(mol, fix=fix),
        center=requested_centre(mol, center),
        coordinate=coordinate_request(mol, coordinate),
        geometry=requested_geometry(geometry),
        observed=observed,
        screen=screen,
    )
    found = possible_species(mol, request)
    if request.coordinate is None:
        return SpeciesSet.of_isomers(found)
    # Each donor choice seats each species once; CX order, not slot or atom order, picks the first.
    return SpeciesSet(
        mol
        for species in found
        for seatings in vacancy_seatings(species, request.coordinate, request.center)
        for mol in sorted(map(mol_from_species, seatings), key=cxsmiles)
    )


def _collect(errors, err, *, lone=False):
    """Keep `err`, the `EmbeddingError` of one candidate, in `errors`, logged as one WARNING line unless `lone`."""
    errors.append(err)
    if not lone:
        logger.warning("%s", err)


def _raise_unless_built(built, errors, counted, *, lone=False):
    """Raise when `built` is empty: a `lone` candidate's own error, else one `EmbeddingError` for all `errors`.

    That error counts the failures and names the first.
    """
    if built:
        return
    if lone:
        raise errors[0]
    failures = sum((err.failures for err in errors), Counter())
    raise EmbeddingError(f"none of {len(errors)} {counted}; first: {errors[0]}", failures=failures)


def embed(
    source,
    *,
    fix=None,
    constrain=None,
    contacts=None,
    stereo=None,
    n=None,
    seed=None,
    threads=None,
    params=None,
    trajectory=False,
    max_iters=MAX_ITERS,
):
    """Embed conformers of one species, or of each species in a sequence.

    A species is a Mol (from `rx.isomers`, `rx.parse_smiles` or `rx.read_xyz`) or a SMILES. One species gives an
    `Ensemble`. A sequence, or several contact modes (one member each), gives an `EnsembleSet` of every member
    that embedded, each failure in ``.errors``; it raises only when none embeds. A source that is several species, such
    as unassigned stereo or a metal with no specified or observed arrangement, raises naming `rx.isomers`;
    ``stereo="free"`` leaves unassigned ligand stereo to the seed.

    ``fix`` and ``constrain`` take the forms the `constraints` module lists; ``fix=rx.Template(reference,
    map_or_smarts)`` copies a reference's coordinates. ``contacts='auto'`` embeds each mode `rx.contacts` lists by
    default, ``contacts=rx.contacts(mol, kinds=...)`` each mode of those kinds, one of its values that mode alone, and
    ``'input'`` holds the contacts the input's first conformer makes. ``stereo=`` applies to the species read
    (``'invert'`` gives the mirror image: every hand inverted, a metal's included, every E/Z kept); a request that makes
    several species raises. A constrained embed comes back relaxed into its windows. ``trajectory=True`` needs ``n=1``
    and stores the accepted restrained-UFF cleanup in ``result.trajectory``; ``max_iters`` caps that relax. ``seed`` and
    ``threads`` set RDKit's sampling, even when `params` wraps a native RDKit object.
    """
    _validate_relax(max_iters, trajectory, n)
    request, sampling, parsed = (
        StereoRequest.parse(stereo, one_species=True),
        resolve_sampling(params, seed, threads),
        parse_contacts(contacts),
    )
    sequence = isinstance(source, (list, tuple))
    sources = source if sequence else [source]
    if not sources:
        raise ValueError("rx.embed got an empty sequence; pass one species or a non-empty sequence of them")
    out, errors = [], []
    for item in sources:
        mol = _mol_from_source(item, "rx.embed")
        species = None
        try:
            pins = resolve_core(mol, fix=fix, constrain=constrain)
            species = read_species(mol, request.on(mol), pins=pins)
            modes = contact_modes(graph_with_stereo(species), parsed)
            chirality = stereo_filter(mol, request.on(mol))
            lone = len(modes) == 1
            kept, failed = [], []
            for label, terms in modes.items():
                try:
                    prepared = prepare(species, pins, contacts=terms, params=sampling.params)
                    ens = seeded(species, prepared, n, sampling, Ensemble)
                    ens.tag = {} if label is None else {"nci": label}
                    ens._stereo_filter = chirality
                    kept.append(ens.relax_into_windows(trajectory=trajectory, max_iters=max_iters))
                    logger.info("embed: kept %r", kept[-1])  # named as its set names it, only when the line is written
                except EmbeddingError as err:
                    _collect(failed, err, lone=lone)
                except ValueError as err:
                    if label is None:
                        raise
                    _collect(failed, EmbeddingError(f"contact mode {label!r}: {err}"), lone=lone)
            _raise_unless_built(kept, failed, "candidates could be built", lone=lone)
        except (ValueError, RuntimeError) as err:  # in a sequence, one species fails alone, named as its set names it
            if not sequence:
                raise
            if not isinstance(err, EmbeddingError):
                if species is None:  # it could not be read, so it is named as given
                    named = item if isinstance(item, str) else Chem.MolToSmiles(item)
                else:
                    named = species_name(named_species(species))
                err = EmbeddingError(f"species [{named}]: {err}")
            _collect(errors, err)
            continue
        out.extend(kept)
        errors.extend(failed)  # logged where they failed
    _raise_unless_built(out, errors, "species could be built")
    return EnsembleSet(out, errors) if sequence or parsed == "auto" or isinstance(parsed, dict) else out[0]


def minimize(source, *, fix=None, constrain=None, max_iters=MAX_ITERS):
    """Relax an existing geometry toward ``fix`` / ``constrain`` targets without a conformer search.

    `source` is a Mol with a conformer (from `rx.read_xyz` or an earlier embed). The species is what it specifies, and a
    result that does not agree with it raises. A coordinate ``fix`` is grafted exactly; ``fix=rx.Template(...)`` copies
    a reference's coordinates. ``max_iters`` caps the restrained-UFF relax.

        rx.minimize(mol, fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    _validate_relax(max_iters)
    mol = _mol_from_source(source, "rx.minimize", geometry=True)
    pins = resolve_core(mol, fix=fix, constrain=constrain)
    return relaxed(read_species(mol, pins=pins), pins, max_iters, Ensemble)


def coembed(
    reactant,
    product=None,
    *,
    fixed=None,
    spectator_atoms=None,
    react_atoms=None,
    constrain=None,
    contacts=None,
    stereo=None,
    n=None,
    seed=None,
    threads=None,
    params=None,
    coembed_params=None,
):
    """Co-embed a reaction's reactant and product as paired conformers in one frame; return a `Coembedded`.

    The endpoints are two sources (Mols or SMILES), a ``(reactant, product)`` pair or a ``'reactant>>product'`` SMILES,
    each one species up to the stereo its graph leaves unspecified. The product is put on the reactant's atom order
    (atom maps first, else the graph overlay that keeps most bonds), so atom i is one atom in both. The reacting atoms
    are told from the spectators the endpoints align by; ``spectator_atoms`` and ``react_atoms`` select either, as
    reactant atoms. `constrain` and `contacts` take reactant atom indices and hold both endpoints, a contact only where
    both bond its atoms alike. ``contacts='auto'``, or `rx.contacts`' modes, give a `CoembeddedSet`, one `Coembedded`
    per binding mode, each failed mode in its ``.errors``. `stereo` applies to each endpoint as in `rx.embed`; the
    stereo each endpoint specifies is held and checked, and stereo a graph leaves open takes the shared seed's hand.

    A shared ZERO-bond union seeds both endpoints, which get a short guarded preparation and one coupled relaxation of
    their restrained UFF plus spectator and reacting penalties. ``coembed_params`` takes `rx.CoEmbedParams`: the coupled
    relaxation's weights, stopping tolerances and step budgets, which few runs need to change.

    ``fixed="reactant"`` or ``fixed="product"`` keeps that source's single conformer exactly and builds one partner
    around it in the same frame (``n`` None or 1). The fixed geometry must pass its gates first, and a failure raises
    without moving it. Both sources are left unchanged.

        pairs = rx.coembed("CBr.[Cl-]>>CCl.[Br-]", n=4)
        best = pairs.lowest(1)
        reactant_mol, product_mol = best.reactant.mol, best.product.mol
    """
    sampling, contacts = resolve_sampling(params, seed, threads), parse_contacts(contacts)
    request = StereoRequest.parse(stereo, one_species=True)
    tuning = CoEmbedParams() if coembed_params is None else coembed_params
    if n is not None and (not is_index(n) or n <= 0):
        raise ValueError(f"n={n!r}: give a positive conformer count as an integer, or None for the default")
    mols = [_mol_from_source(side, "rx.coembed") for side in _endpoints(reactant, product)]
    fixed_side = _fixed_side(mols, fixed, n)
    mols[1] = in_reactant_order(mols[0], mols[1])
    reading = request if request.given is not None else StereoRequest.parse("free")
    sides = least_motion(*(read_species(mol, reading.on(mol)) for mol in mols))
    atoms = reaction_atoms(sides[0], sides[1], spectator_atoms, react_atoms)
    modes = contact_modes(mols[0], contacts)
    for terms in (constrain, *modes.values()):  # both endpoints share the reactant's atom order
        if terms and not all(is_index(atom) and 0 <= atom < mols[0].GetNumAtoms() for key in terms for atom in key):
            raise ValueError("coembed geometry terms need atom-index keys of the reactant's atoms")
    pins = resolve_core(mols[0], constrain=constrain)
    filters = [stereo_filter(mol, request.on(mol)) for mol in mols]
    lone = len(modes) == 1
    results, errors = [], []
    for label, terms in modes.items():
        # a contact the reaction makes a bond holds only in the reactant
        alike = terms and {
            key: value
            for key, value in terms.items()
            if all(
                (mols[0].GetBondBetweenAtoms(*map(int, pair)) is None)
                == (mols[1].GetBondBetweenAtoms(*map(int, pair)) is None)
                for pair in itertools.combinations(key, 2)
            )
        }
        try:
            endpoints = []
            for species, held, chirality in zip(sides, (terms, alike), filters, strict=True):
                prepared = prepare(species, pins, contacts=held, params=sampling.params)
                endpoints.append(
                    Ensemble(
                        prepared.mol, [], prepared.cons, species=species, sampling=sampling, _stereo_filter=chirality
                    )
                )
            if fixed_side is not None:  # the supplied conformer must pass its gates before it is frozen
                given = endpoints[fixed_side]
                given.ids = [given._mol.GetConformer().GetId()]
                if reason := given._first_failure(given.ids[0], given.gates):
                    raise EmbeddingError(
                        f"coembed: fixed {fixed} fails ({reason}); supply a valid conformer",
                        failures=Counter({reason: 1}),
                    )
            coupled = coupling(atoms.spectators, atoms.reacting, tuning)
            paired = coembed_pair(endpoints, coupled, n, tuning, fixed=fixed_side)
        except EmbeddingError as err:
            _collect(errors, err, lone=lone)
            continue
        except ValueError as err:
            if label is None:
                raise
            _collect(errors, EmbeddingError(f"contact mode {label!r}: {err}"), lone=lone)
            continue
        for ens in (paired.reactant, paired.product):
            ens.tag = {} if label is None else {"nci": label}
            ens._record(FINAL, ens.energies, ens._uff_name)
        result = Coembedded(
            reactant=paired.reactant,
            product=paired.product,
            coupling=coupled,
            joint_results=paired.joint_results,
            optimizer_attempts=paired.optimizer_attempts,
        )
        logger.info("coembed: retained %d pairs, %d joint attempts", len(result), len(result.optimizer_attempts))
        for index, record in enumerate(result.optimizer_attempts, 1):
            logger.debug(
                "coembed: attempt %d: seed %s, %s iterations, stop %s, max force %s",
                index,
                record.get("seed"),
                record.get("iterations"),
                record.get("stopping_reason"),
                record.get("max_force"),
            )
        results.append(result)
    _raise_unless_built(results, errors, "contact modes could be co-embedded", lone=lone)
    return CoembeddedSet(results, errors) if contacts == "auto" or isinstance(contacts, dict) else results[0]


def _endpoints(reactant, product):
    """Return the two endpoint sources of a reaction given as two sources, a pair or a ``'reactant>>product'``."""
    if product is not None:
        return reactant, product
    if isinstance(reactant, str):  # a dative arrow '->' is not a reaction arrow, so split on '>>' alone
        reactant = tuple(reactant.split(">>"))
        if len(reactant) != 2 or any("|" in side for side in reactant):  # noqa: PLR2004 - two sides
            raise ValueError("coembed() takes 'reactant>>product' without agents or CX extensions, or a pair")
    if not isinstance(reactant, (tuple, list)) or len(reactant) != 2:  # noqa: PLR2004 - two endpoints
        raise ValueError(
            "coembed() takes two sources, a (reactant, product) pair or 'reactant>>product'; got "
            f"{type(reactant).__name__} {reactant!r:.60}"
        )
    return tuple(reactant)


def _fixed_side(mols, fixed, n):
    """Return the selected fixed endpoint's index after checking its count and coordinates, else None."""
    if fixed is None:
        return None
    if fixed not in ("reactant", "product"):
        raise ValueError('fixed must be "reactant" or "product"; omit it to generate both endpoints')
    if n is not None and n != 1:
        raise ValueError("fixed constructs one partner of one supplied conformer; use n=1 or omit n")
    side = ("reactant", "product").index(fixed)
    if mols[side].GetNumConformers() != 1:
        raise ValueError(f"fixed {fixed} needs exactly one conformer; select one with ensemble[k].mol")
    conf = mols[side].GetConformer()
    if not conf.Is3D() or not np.isfinite(conf.GetPositions()).all():
        raise ValueError(f"fixed {fixed} needs finite 3D coordinates; supply a valid conformer")
    return side
