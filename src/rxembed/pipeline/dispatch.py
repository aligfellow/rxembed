"""Expand pipeline inputs into candidates and run each through one embed executor.

This module owns the public `embed`, `coembed`, `minimize` and `enumerate_isomers` (the facade's `metal`) verbs.
It adapts sources, templates, stereo, metal identities, vacant-site seating and NCI modes. Those axes only produce
candidate data. `_execute` then prepares constraints and calls the shared core seeding seam once per candidate.
"""

from __future__ import annotations

import itertools
import logging
import os
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem

from rxembed import metal_enumeration
from rxembed.bounds import resolve_params, seed_count
from rxembed.coembedding import (
    ALIGNMENT_WEIGHT,
    TS_WEIGHT,
    PairedConformers,
    correspondence_budgets,
    seed_paired_conformers,
)
from rxembed.constraints import (
    Constraints,
    is_index,
    resolve_atom,
    substruct_matches,
    template_to_fix,
)
from rxembed.embed import (  # module functions, not the package facade
    BASE_STIFFNESS,
    EmbeddingError,
    prepare,
    prepare_relax,
    remedy,
    require_seed_count,
    seed_conformers,
)
from rxembed.metal_core import (
    VACANT,
    connect_metal,
    materialise_phantoms,
    metal_index,
    metal_indices,
    reject_metal_bonds,
)
from rxembed.metal_isomer import Isomer, from_geometry, least_motion_pair
from rxembed.metal_polyhedron import describe
from rxembed.metal_smiles import parse_smiles
from rxembed.relax import MAX_ITERS
from rxembed.stereo import (
    axis_stereo,
    bond_stereo,
    defined_stereo_label,
    enumerate_unassigned,
    point_stereo,
    unassigned_centres,
)
from rxembed.utils import reaction_correspondence

from .ensemble import Coembedded, Ensemble, EnsembleSet
from .nci import Contact, auto_binding_modes, input_contacts
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


def _is_xyz(source):
    """Return whether `source` names an .xyz file, as a str or a path."""
    path = os.fspath(source) if isinstance(source, os.PathLike) else source
    return isinstance(path, str) and path.lower().endswith(".xyz")


def _default_stereo(source, stereo):
    """Retain stereo from coordinates by default; enumerate unspecified stereo from a graph."""
    if stereo is not None:
        return stereo
    if isinstance(source, Isomer):
        has_geometry = bool(source.mol.GetNumConformers())
    elif isinstance(source, Chem.Mol):
        has_geometry = bool(source.GetNumConformers())
    else:
        has_geometry = _is_xyz(source)
    return "all" if has_geometry else "unassigned"


def _validate_relax(max_iters, trajectory=False, n=None):
    """Require a positive restrained-UFF iteration cap and a trajectory request for exactly one conformer."""
    if not isinstance(trajectory, bool):
        raise TypeError("trajectory must be True or False")
    if trajectory and n != 1:
        raise ValueError("trajectory=True requires n=1")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")


def _normalize(source, charge=0, *, remove_hs=True):
    """Normalise SMILES / an .xyz path / an RDKit Mol to ``(mol with Hs, has_geometry)``.

    `charge` is used for xyz bond perception (a charged metal complex / ion).
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path, not just str
    if isinstance(source, Chem.Mol):
        mol = source
    elif _is_xyz(source):
        mol = read_xyz(source, charge)
    else:
        mol = parse_smiles(source, remove_hs=remove_hs)
    has_geom = mol.GetNumConformers() > 0
    return Chem.AddHs(mol, addCoords=has_geom), has_geom


def _contact_terms(contacts):
    """Return `contacts` as one ``{key: window}`` dict of distance and angle terms, or None for no contacts."""
    if contacts is None:
        return None
    if isinstance(contacts, Contact):
        items = [contacts]
    elif isinstance(contacts, dict):
        items = [v for v in contacts.values() if isinstance(v, Contact)]
        if items and len(items) != len(contacts):  # {label: Contact} must be all Contacts
            raise TypeError("contacts: dict mixes Contact and non-Contact values")
        if not items:
            return dict(contacts)  # a raw {(i,j):(lo,hi)} distance dict
    else:
        values = list(contacts)
        items = [c for c in values if isinstance(c, Contact)]
        if len(items) != len(values):
            raise TypeError("contacts: list must contain Contact objects (from rx.nci_candidates)")
    return {key: value for ct in items for key, value in (*ct.distances.items(), *ct.angles.items())} or None


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
        ms = [m[0] for m in substruct_matches(iso.mol, pattern)]
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
    mol, cons, iso, graft_ref = prepare(
        spec, fix=fix, constrain=constrain, contacts=_contact_terms(contacts), params=params
    )
    mol, ids, target = seed_conformers(mol, cons, iso, n, params, graft_ref=graft_ref)
    require_seed_count(ids, target, iso)
    return Ensemble(mol, list(ids), cons, iso, params=params).record_metal_context()


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
    """Return explicit contacts, the contacts an input geometry makes, or automatic contact modes."""
    disc = (
        connect_metal(source.restore(Chem.Mol(source.mol)), source.donor_bonds)
        if isinstance(source, Isomer)
        else source
    )
    if contacts == "input":
        if not disc.GetNumConformers():
            raise ValueError("contacts='input' holds the contacts an input geometry makes; pass an .xyz or use 'auto'")
        mode = input_contacts(disc)
        if mode is None:
            logger.info("contacts='input': the input geometry makes no NCI contact -> plain embed")
        return [(mode.label, mode) if mode else (None, None)]
    if contacts != "auto":
        return [(None, contacts)]
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


def _resolve_template(template, fix, target):
    """Fold ``template=`` into a coordinate ``fix``, extending the core resolver with the pipeline's `Ensemble`.

    The core takes a Mol, an ``.xyz`` path or an (N, 3) array; this adds an `Ensemble`, handing over the
    positions of its first tracked conformer. An optional SMARTS is matched once on the target and reference;
    an xyz or coordinate array uses an explicit index map. The template is read for coordinates and never
    perceived, which lets a hypervalent reacting core serve as a reference.

    ``target`` is the *normalised* source, whose own conformer resolves a ``fix=[atoms]`` list beside the
    template.
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
    return template_to_fix(template, fix, target)


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


def _mapped_neighborhood(atom):
    """Read mapped bonds, charges and explicit hydrogen neighbors from an endpoint graph."""
    neighbors = []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        neighbors.append(
            (
                other.GetAtomMapNum(),
                other.GetAtomicNum(),
                other.GetIsotope(),
                bond.GetBondTypeAsDouble(),
                bond.GetBeginAtomIdx() == atom.GetIdx() if bond.GetBondType() == Chem.BondType.DATIVE else False,
            )
        )
    return atom.GetFormalCharge(), tuple(sorted(neighbors))


def _mapped_atoms(molecules):
    """Resolve unique map numbers to real endpoint atoms without assuming the same atom order."""
    mapped = []
    for mol in molecules:
        atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum()]
        numbers = [atom.GetAtomMapNum() for atom in atoms]
        if len(set(numbers)) != len(numbers):
            raise ValueError("coembed found duplicate atom maps; assign a unique map number to each atom")
        mapped.append(dict(zip(numbers, atoms, strict=True)))
    if mapped[0].keys() != mapped[1].keys():
        raise ValueError("mapped endpoints must use the same atom-map numbers; correct the atom mapping")
    for number in mapped[0].keys() & mapped[1].keys():
        a, b = mapped[0][number], mapped[1][number]
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            raise ValueError(f"atom map {number} changes element or isotope; correct the atom mapping")
    return mapped


def _mapped_stereo(mol):
    """Associate declared point, E/Z and axial identities with their mapped participating atoms."""
    label = defined_stereo_label(mol, exclude=metal_indices(mol))
    out = {}
    for index, code in point_stereo(label).items():
        number = mol.GetAtomWithIdx(index).GetAtomMapNum()
        if number:
            out.setdefault(number, []).append(("point", code))
    for kind, labels in (("ez", bond_stereo(label)), ("axis", axis_stereo(label))):
        for pair, code in labels.items():
            numbers = tuple(sorted(mol.GetAtomWithIdx(index).GetAtomMapNum() for index in pair))
            if all(numbers):
                for number in numbers:
                    out.setdefault(number, []).append((kind, numbers, code))
    return {number: tuple(sorted(records)) for number, records in out.items()}


def _coembed_pairs(molecules, selection, name):
    """Adapt an explicit one-to-one atom-index correspondence for alignment or TS measurement."""
    if not isinstance(selection, dict):
        raise TypeError(f"{name} must be a {{reactant_index: product_index}} mapping")
    pairs = tuple((resolve_atom(molecules[0], i), resolve_atom(molecules[1], j)) for i, j in selection.items())
    if len({i for i, _j in pairs}) != len(pairs) or len({j for _i, j in pairs}) != len(pairs):
        raise ValueError(f"{name} must be one-to-one; select each endpoint atom once")
    for i, j in pairs:
        a, b = molecules[0].GetAtomWithIdx(i), molecules[1].GetAtomWithIdx(j)
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            raise ValueError(f"{name} pair ({i}, {j}) changes element or isotope; correct the correspondence")
    return pairs


def _atrop_rotating_atoms(molecules, *, alignment):
    """Return mapped moving groups for changes between declared axial hands."""
    axes = [
        {
            tuple(sorted(mol.GetAtomWithIdx(index).GetAtomMapNum() for index in axis)): code
            for axis, code in axis_stereo(defined_stereo_label(mol)).items()
        }
        for mol in molecules
    ]
    heavy_labels = {atom.GetAtomMapNum() for atom in molecules[0].GetAtoms() if atom.GetAtomicNum() != 1}
    moving = set()
    for axis in axes[0].keys() & axes[1].keys():
        if axes[0][axis] == axes[1][axis]:
            continue
        sides = [set(), set()]
        for mol in molecules:
            indices = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
            cut = Chem.RWMol(mol)
            cut.RemoveBond(*(indices[label] for label in axis))
            fragments = Chem.GetMolFrags(cut)
            for side, label in enumerate(axis):
                fragment = next(part for part in fragments if indices[label] in part)
                sides[side].update(mol.GetAtomWithIdx(index).GetAtomMapNum() for index in fragment)
        if set(axis) <= sides[0] or set(axis) <= sides[1]:
            if alignment is None:
                raise ValueError(
                    "an atropisomeric axis lies in a cycle; select stationary atoms with spectator_atoms= "
                    "and moving atoms with ts_atoms="
                )
            continue
        # Rotating an axis moves its whole attached group. Keep the larger scaffold as the reference frame;
        # use both endpoints' fragments so a substituent cannot remain a strongly coupled spectator.
        rotating = min(
            sides,
            key=lambda labels: (
                len(labels & (alignment or set())),
                len(labels & heavy_labels),
                len(labels),
                tuple(sorted(labels)),
            ),
        )
        moving.update(rotating)
    return moving


def _coembed_correspondence(molecules, spectator_atoms, ts_atoms, *, frames=False):
    """Resolve full reaction identity and keep spectator alignment separate from all identified TS atoms."""
    mapped = _mapped_atoms(molecules)
    selected = None if spectator_atoms is None else _coembed_pairs(molecules, spectator_atoms, "spectator_atoms")
    requested = None if ts_atoms is None else _coembed_pairs(molecules, ts_atoms, "ts_atoms")
    if selected == ():
        raise ValueError("spectator_atoms must contain at least one atom pair")
    if (
        selected is not None
        and requested is not None
        and (
            {i for i, _j in selected} & {i for i, _j in requested}
            or {j for _i, j in selected} & {j for _i, j in requested}
        )
    ):
        raise ValueError("ts_atoms must be outside spectator_atoms; remove TS atoms from the spectator selection")
    graphs = [Chem.Mol(mol) for mol in molecules]
    number = max((atom.GetAtomMapNum() for mol in graphs for atom in mol.GetAtoms()), default=0)
    for i, j in (*(selected or ()), *(requested or ())):
        a, b = graphs[0].GetAtomWithIdx(i), graphs[1].GetAtomWithIdx(j)
        partners = mapped[1].get(a.GetAtomMapNum()), mapped[0].get(b.GetAtomMapNum())
        if (partners[0] is not None and partners[0].GetIdx() != j) or (
            partners[1] is not None and partners[1].GetIdx() != i
        ):
            raise ValueError("explicit correspondence disagrees with atom maps; correct spectator_atoms= or ts_atoms=")
        if a.GetAtomMapNum() and b.GetAtomMapNum() and a.GetAtomMapNum() != b.GetAtomMapNum():
            raise ValueError("explicit correspondence disagrees with atom maps; correct spectator_atoms= or ts_atoms=")
        number += 1
        label = a.GetAtomMapNum() or b.GetAtomMapNum() or number
        a.SetAtomMapNum(label)
        b.SetAtomMapNum(label)
    inventories = [sorted((atom.GetAtomicNum(), atom.GetIsotope()) for atom in mol.GetAtoms()) for mol in graphs]
    if inventories[0] != inventories[1]:
        raise ValueError("coembed needs matching atom inventories; embed endpoints separately for partial systems")
    if frames and all(
        (a.GetAtomicNum(), a.GetIsotope(), a.GetAtomMapNum()) == (b.GetAtomicNum(), b.GetIsotope(), b.GetAtomMapNum())
        for a, b in zip(*[mol.GetAtoms() for mol in graphs], strict=True)
    ):
        correspondence = tuple((i, i) for i in range(graphs[0].GetNumAtoms()))
    else:
        correspondence = reaction_correspondence(*graphs)
    # Temporary complete maps let neighborhood and stereo readers compare identities, including implicit Hs.
    for label, (i, j) in enumerate(correspondence, 1):
        graphs[0].GetAtomWithIdx(i).SetAtomMapNum(label)
        graphs[1].GetAtomWithIdx(j).SetAtomMapNum(label)
    stereo = [_mapped_stereo(mol) for mol in graphs]
    changed = {
        (i, j)
        for label, (i, j) in enumerate(correspondence, 1)
        if _mapped_neighborhood(graphs[0].GetAtomWithIdx(i)) != _mapped_neighborhood(graphs[1].GetAtomWithIdx(j))
        or stereo[0].get(label) != stereo[1].get(label)
    }
    alignment = None if selected is None else {graphs[0].GetAtomWithIdx(i).GetAtomMapNum() for i, _j in selected}
    rotating = _atrop_rotating_atoms(graphs, alignment=alignment)
    changed.update(pair for label, pair in enumerate(correspondence, 1) if label in rotating)
    if selected is None:
        unchanged = tuple(pair for pair in correspondence if pair not in changed)
        heavy = tuple(pair for pair in unchanged if graphs[0].GetAtomWithIdx(pair[0]).GetAtomicNum() != 1)
        mapped_hydrogens = tuple(
            (left.GetIdx(), mapped[1][number].GetIdx())
            for number, left in mapped[0].items()
            if left.GetAtomicNum() == 1 and (left.GetIdx(), mapped[1][number].GetIdx()) in unchanged
        )
        pairs = tuple(dict.fromkeys((*heavy, *mapped_hydrogens))) or unchanged
    else:
        pairs = selected
    if not pairs:
        raise ValueError("coembed cannot align the endpoints: the mapped correspondence has no unchanged atoms")
    if requested and (
        {i for i, _j in pairs} & {i for i, _j in requested} or {j for _i, j in pairs} & {j for _i, j in requested}
    ):
        raise ValueError("ts_atoms must be outside spectator_atoms; remove TS atoms from the spectator selection")
    reactive = tuple(
        dict.fromkeys((*(pair for pair in correspondence if pair in changed and pair not in pairs), *(requested or ())))
    )
    return pairs, reactive, correspondence


def _coembed_endpoint(source, stereo, params, *, metal=None, charge=0, constrain=None, contacts=None):
    """Adapt one selected endpoint to its native workflow owner without changing its atom order."""
    if not isinstance(source, (str, os.PathLike, Chem.Mol, Isomer)):
        raise TypeError("coembed takes one SMILES, Mol or selected Isomer per endpoint; select a single identity")
    selected_stereo = _default_stereo(source, stereo)
    metal_enumeration.validate_stereo(selected_stereo)
    source = source if isinstance(source, Isomer) else _normalize(source, charge, remove_hs=False)[0]
    if metal is not None or (not isinstance(source, Isomer) and metal_indices(source)):
        _discovery, candidates, _expanded = _source_candidates(
            source, metal=metal, fix=None, coordinate=None, stereo=selected_stereo
        )
        if len(candidates) != 1:
            raise ValueError("coembed needs one coordination identity; select one endpoint with rx.metal(...)[index]")
        source = candidates[0].spec
    variants, _expanded = _stereo_variants(source, selected_stereo)
    if len(variants) != 1:
        raise ValueError("coembed needs one stereo identity per endpoint; specify stereo or pass stereo='free'")
    source = variants[0][0]
    work, terms, iso, _ref = prepare(source, constrain=constrain, contacts=contacts, params=params)
    if terms.frozen:
        raise ValueError("coembed needs free endpoint geometries; select unfrozen Isomers or embed separately")
    owner = Ensemble(work, [conf.GetId() for conf in work.GetConformers()], terms, iso, params=params)
    owner.stereo_filter = _stereo_filter(source, 0, selected_stereo)
    if iso is not None and selected_stereo != "free":
        known = {frozenset((index,)) for index in point_stereo(iso.stereo_label)}
        known.update(bond_stereo(iso.stereo_label))
        known.update(axis_stereo(iso.stereo_label))
        public = owner.restored_mol()
        undefined = {frozenset(site) for site in unassigned_centres(public, exclude=metal_indices(public))} - known
        if undefined:
            raise ValueError(
                "coembed needs a defined ligand stereoisomer; select rx.metal(..., stereo='unassigned')[index] "
                "or explicitly allow mixed hands with stereo='free'"
            )
    if iso is not None:
        for donor, metal_atom in iso.donor_bonds:
            owner.sphere.setdefault(metal_atom, []).append(donor)
    return owner


def _coembed_inputs(reactant, product):
    """Adapt two endpoint arguments, a pair, or a reaction SMILES without changing their atom orders."""
    if product is None:
        if isinstance(reactant, str):
            sources = reactant.split(">>")
            if len(sources) != 2 or any("|" in side for side in sources):  # noqa: PLR2004 - two endpoints
                raise ValueError("coembed takes 'reactant>>product' without agents or CX extensions, or two sources")
        elif isinstance(reactant, (tuple, list)) and len(reactant) == 2:  # noqa: PLR2004 - two endpoints
            sources = reactant
        else:
            raise TypeError("coembed needs two endpoint sources or a 'reactant>>product' SMILES")
        reactant, product = sources
    return reactant, product


def _coembed_geometry_terms(terms, onto):
    """Map reactant-index geometry terms into the product's own atom order."""
    try:
        if any(not is_index(atom) for key in terms or {} for atom in key):
            raise ValueError("non-index geometry key")
        return {tuple(onto[int(atom)] for atom in key): value for key, value in (terms or {}).items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("coembed geometry terms need matched atom-index keys; complete the correspondence") from error


def coembed(
    reactant,
    product=None,
    *,
    spectator_atoms=None,
    ts_atoms=None,
    n=None,
    alignment_weight=ALIGNMENT_WEIGHT,
    ts_weight=TS_WEIGHT,
    metal=None,
    charge=0,
    constrain=None,
    contacts=None,
    joint_gtol=0.1,
    joint_energy_tol=1e-3,
    max_iters=5000,
    seed=None,
    threads=None,
    params=None,
    stereo=None,
):
    """Coembed endpoints from a shared mapped geometry using native UFF and whole-pair recovery.

    Accept two SMILES, XYZ paths, RDKit molecules or selected Isomers, a pair, or a 'reactant>>product'
    SMILES. ``metal`` selects a unique geometry; ``charge`` supplies XYZ charge. ``constrain`` and
    ``contacts`` use reactant atom-index keys mapped to the product, with the normal ``embed`` discovery.
    Contacts that become bonds hold only the reactant. Multiple surviving contact modes return a list of
    paired results with NCI tags; failures remain in their ``errors``.

    Match complete explicit-H inventories by the graph overlay that keeps the most bonds, honoring maps
    and explicit selections. Ambiguous overlays require reacting-atom maps. Preserve input atom orders and
    connectivity. Different inventories require separate embedding. ``spectator_atoms`` selects spectator index pairs;
    otherwise use unchanged heavy atoms and unchanged explicitly mapped hydrogens. Changed neighborhoods,
    charges and declared stereo identify TS atoms, including transferring hydrogens. ``ts_atoms`` adds
    one-to-one pairs outside the alignment; inferred reacting pairs are always included.

    Use shared seeding. Seed and relax
    the more bonded endpoint, transfer its mapped coordinates, repair the copied endpoint and rebuild its
    own haptic helpers. Drop wrong copied coordination shapes or donor seatings; bounded shared-seed retries
    refill only missing pairs. Initial and replacement batches use this same native endpoint workflow. With
    ``n=None``, use the larger endpoint's normal sampling target, counting each endpoint once.
    Omit ``seed`` and ``params`` to try fresh seeds after failures. An explicit ``seed`` or supplied
    ``EmbedParams`` keeps its RNG seed fixed, including stereo selection and replacement batches;
    same-seed resampling remains available to repair copied endpoint geometry.

    Derive each coupled atom's budget from its largest incompatible endpoint distance-window gap, with no
    added floor. These are soft penalty thresholds and do not establish 3D or stereo feasibility.
    Inspect effective budgets through ``Coembedded.budgets``.

    Alternate native RDKit minimizations on the sum of endpoint energies and separately normalized
    spectator and TS squared-excess penalties. ``alignment_weight=25`` and ``ts_weight=1.25`` control the
    two groups independently. Run native endpoint cleanup between initial coupling and a final coupled
    polish. Retry missed metal-distance walls through the native stiffness ladder before reseeding; retain
    the last validated coordinates if polishing fails. Before each joint pass, rigidly register spectators
    only when the complete coupling penalty decreases. Final displacements can exceed their budgets;
    ``budget_excesses`` reports them.
    ``alignment_scores`` and ``ts_rmsds`` measure returned coordinates without an additional fit.

    ``joint_gtol`` controls the maximum joint-gradient criterion; native minimizations keep RDKit tolerances.
    ``joint_energy_tol`` also permits a non-increasing cycle-score plateau without a force guarantee;
    zero disables plateau stopping. ``max_iters`` caps completed native steps across both endpoints in each
    joint pass, and each endpoint cleanup attempt.
    ``joint_results`` describes final polishing; ``optimizer_attempts`` also retains earlier stage records.
    Validation coordinate corrections mark the final result ``validation_adjusted`` and invalidate convergence.

    Preserve each endpoint's declared point, E/Z, axial and selected coordination identity through normal
    validation and bounded recovery. Undefined ligand stereo in a selected Isomer requires selecting a
    defined identity or explicitly permitting unspecified hands with ``stereo='free'``. Both endpoints
    must pass publication; whole-pair replacements keep their ids. Fewer conformers may survive or required
    identity failures may raise. Frozen geometries and stereo expansion require separate embedding.
    """
    _validate_relax(max_iters)
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n <= 0):
        raise ValueError("n must be a positive integer or None")
    for name, value in (
        ("alignment_weight", alignment_weight),
        ("ts_weight", ts_weight),
        ("joint_gtol", joint_gtol),
        ("joint_energy_tol", joint_energy_tol),
    ):
        nonnegative = name == "joint_energy_tol"
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 0
            or (not nonnegative and value == 0)
        ):
            raise ValueError(f"{name} must be finite and {'nonnegative' if nonnegative else 'positive'}")
    reactant, product = _coembed_inputs(reactant, product)
    reseed = seed is None and params is None
    params = resolve_params(params, seed, threads)
    originals = [
        _coembed_endpoint(source, stereo, params, metal=metal, charge=charge) for source in (reactant, product)
    ]
    public = [owner.restored_mol() for owner in originals]
    pairs, reactive, correspondence = _coembed_correspondence(
        public, spectator_atoms, ts_atoms, frames=_is_xyz(reactant) and _is_xyz(product)
    )
    specs = [owner.iso if owner.iso is not None else mol for owner, mol in zip(originals, public, strict=True)]
    if all(isinstance(spec, Isomer) for spec in specs):
        specs = least_motion_pair(specs[0], specs[1], correspondence)
    onto = dict(correspondence)

    product_constrain = _coembed_geometry_terms(constrain, onto)
    results, failed = [], []
    for label, contact in _contact_modes(public[0], contacts, params):
        try:
            terms = _contact_terms(contact)
            product_terms = _coembed_geometry_terms(terms, onto)
            for key in terms or {}:
                if any(
                    bool(public[0].GetBondBetweenAtoms(int(i), int(j)))
                    != bool(public[1].GetBondBetweenAtoms(onto[int(i)], onto[int(j)]))
                    for i, j in itertools.combinations(key, 2)
                ):
                    del product_terms[tuple(onto[int(atom)] for atom in key)]
            owners = [
                _coembed_endpoint(spec, stereo, params, constrain=constraints, contacts=soft)
                for spec, constraints, soft in zip(
                    specs, (constrain, product_constrain), (terms, product_terms), strict=True
                )
            ]
            result = _coembed_pair(
                owners,
                pairs,
                reactive,
                correspondence,
                params=params,
                n=n,
                alignment_weight=alignment_weight,
                ts_weight=ts_weight,
                joint_gtol=joint_gtol,
                joint_energy_tol=joint_energy_tol,
                max_iters=max_iters,
                reseed=reseed,
            )
        except (ValueError, RuntimeError) as error:
            if label is None:
                raise
            failed.append(EmbeddingError(f"contacts={contacts!r} mode {label!r}: {error}"))
            logger.warning("%s", failed[-1])
            continue
        if label is not None:
            for owner in (result.reactant, result.product):
                owner.tag["nci"] = label
        results.append(result)
    if not results:
        raise EmbeddingError(f"no contact mode co-embedded; first failure: {failed[0]}")
    results = [replace(result, errors=tuple(failed)) for result in results]
    return results[0] if len(results) == 1 else results


def _coembed_pair(
    owners,
    pairs,
    reactive,
    correspondence,
    *,
    params,
    n,
    alignment_weight,
    ts_weight,
    joint_gtol,
    joint_energy_tol,
    max_iters,
    reseed,
):
    """Seed, optimize and publish one endpoint pair through the shared bounded recovery engine."""
    allowances = correspondence_budgets(
        [owner._mol for owner in owners], [owner.cons for owner in owners], correspondence, params
    )
    coupled = (*pairs, *reactive)
    effective = {pair: allowances[pair] for pair in coupled}
    endpoint_graphs = [materialise_phantoms(owner._mol, owner.cons.haptic) for owner in owners]
    joint = Chem.CombineMols(*endpoint_graphs)
    Chem.GetSymmSSSR(joint)
    split = endpoint_graphs[0].GetNumAtoms()
    cons = Constraints(
        distances={(i, split + j): (0.0, effective[i, j]) for i, j in coupled},
        coupling_weights={
            **{(i, split + j): alignment_weight / len(pairs) for i, j in pairs},
            **{(i, split + j): ts_weight / len(reactive) for i, j in reactive},
        },
        alignment_pairs=tuple((i, split + j) for i, j in pairs),
        seed_correspondence=tuple(correspondence),
    )
    if n is None:
        n = max(seed_count(owner._mol, constrained=owner.cons.is_constrained) for owner in owners)
    for owner in owners:
        owner._reseed = reseed
    joint, ids = seed_paired_conformers(joint, cons, owners, n, params, max_iters=max_iters, reseed=reseed)
    if not ids:
        raise EmbeddingError("coembed: no shared seeds; inspect endpoint bounds/stereo or try another seed=")
    engine = PairedConformers(
        joint,
        list(ids),
        cons,
        params=params,
        _reseed=reseed,
        endpoints=tuple(owners),
        _seed_max_iters=max_iters,
        validators=tuple(owner._workflow_failure for owner in owners),
        alignment_weight=alignment_weight,
        joint_gtol=joint_gtol,
        joint_energy_tol=joint_energy_tol,
    )
    engine._sync_endpoints()
    for owner in owners:
        if owner.iso is not None:
            hands = owner._seed_donor_hands(owner.ids[:1])
            owner.donor_hand = {donor: hand for donor, (hand, _) in hands.get(owner.ids[0], {}).items()}
    typed = engine._relax_once(BASE_STIFFNESS, max_iters, operation="coembed")
    engine._accept_relaxed(BASE_STIFFNESS, max_iters, operation="coembed", typed=typed, raise_if_empty=True)
    engine._sync_endpoints()
    engine._refresh_scores(BASE_STIFFNESS)
    for owner in owners:
        owner._finalize_minimize(list(ids))
    common = set(owners[0].ids) & set(owners[1].ids)
    for owner in owners:
        owner._remove(set(owner.ids) - common)
    if not common:
        raise EmbeddingError("coembed: no pair survived native endpoint filtering; try another seed or alignment")
    return Coembedded(
        owners[0],
        owners[1],
        cons,
        alignment_weight,
        {cid: engine.joint_results[cid] for cid in owners[0].ids},
        tuple(record for batch in (engine, *engine.recovery_batches) for record in batch.optimizer_attempts),
        ts_weight=ts_weight,
    )


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
    ``{target_i: ref_i}`` index map. ``contacts=`` reaches `nci_modes` (``'auto'``) or `input_contacts`
    (``'input'``); ``metal=`` / ``coordinate=`` reach `rx.metal`.

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
    if template is not None:
        fix = _resolve_template(template, fix, source.mol if isinstance(source, Isomer) else source)
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
                    failed.append(EmbeddingError(f"contacts={contacts!r} mode {nci_label!r}: {err}"))
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
        fix = _resolve_template(template, fix, mol)
    # The surrogate / sphere-hold / graft assembly is the core's (`rxembed.embed.minimize` is the same call
    # with a `Conformers` result); this only wraps it as an `Ensemble` so the pipeline verbs chain off it.
    mol, ids, cons, iso = prepare_relax(spec, fix=fix, constrain=constrain)
    return Ensemble(mol, ids, cons, iso).minimize(stiffness=stiffness, max_iters=max_iters)
