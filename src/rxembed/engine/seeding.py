"""Build one species: compile its constraints (`prepare`), draw its seeds (`seeded`) or relax a given geometry.

Seeds come from edited RDKit bounds with the requested metal hand, face winding and ligand stereo, and an exact fixed
core grafted back; `Conformers.minimize` then relaxes them with restrained UFF. A reaction's endpoints are built in
`coembed`.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem
from rdkit.Numerics import rdAlignment

from rxembed.engine.acceptance import EPS, EmbeddingError, conformer_misses, remedy
from rxembed.geometry.bounds import MAX_SEED_COUNT, EmbedParams, seed_coordinates, seed_count
from rxembed.geometry.constraints import DIST_ATOMS, Constraints, compose, resolve_core, retained_soft
from rxembed.graph import ligand_degree, surrogate_all_metals
from rxembed.identity.reading import Specification
from rxembed.identity.species import graph_with_stereo, misplaced_pins, shape_readings
from rxembed.metal.centres import donor_bonds, engine_graph, seed_graph
from rxembed.metal.distances import ionic_contacts
from rxembed.metal.orientation import stripped_hybridisation
from rxembed.metal.sphere import LigandFacts, compile_constraints
from rxembed.metal.stereo import face_orientations
from rxembed.utils import TETRAHEDRAL_TAGS, mirror_tag, tetrahedral_perception, with_conformers

logger = logging.getLogger(__name__)  # under the "rxembed" tree `set_verbose` configures


REPLACEMENT_FACTOR = 4  # seeds drawn per seed still wanted, against selection and post-relax losses


def graft_frozen(mol, conf_ids, graft):
    """Put the atoms of each conformer at their `graft` coordinates, rotated, never mirrored, onto the embedded core.

    `graft` maps atom to xyz. A pair keeps its first atom and embedded axis and takes the grafted distance; a single
    atom has nothing to restore.
    """
    frozen = sorted(graft)
    core = np.array([graft[i] for i in frozen], float)
    if len(frozen) <= 1:
        return
    if len(frozen) == DIST_ATOMS:
        dt = float(np.linalg.norm(core[0] - core[1]))
        for c in conf_ids:
            conf = mol.GetConformer(c)
            pa = np.array(conf.GetAtomPosition(frozen[0]))
            pb = np.array(conf.GetAtomPosition(frozen[1]))
            v = pb - pa
            n = np.linalg.norm(v)
            v = v / n if n > EPS else np.array([1.0, 0.0, 0.0])
            conf.SetAtomPosition(frozen[1], (pa + dt * v).tolist())
        return
    for c in conf_ids:
        conf = mol.GetConformer(c)
        emb = np.array([list(conf.GetAtomPosition(i)) for i in frozen])  # embedded core
        # reflect=False (the default): a proper rotation only, never a mirror image of the input core.
        _ssd, transform = rdAlignment.GetAlignmentTransform(emb, core)
        transform = np.asarray(transform)
        fitted = core @ transform[:3, :3].T + transform[:3, 3]  # rotate input core onto the embedded one
        for i, p in zip(frozen, fitted, strict=True):
            conf.SetAtomPosition(i, p.tolist())


def seed_takes_shape(mol, cid, species):
    """Return whether a raw seed already takes each centre's polyhedron, before cleanup."""
    return all(reading.accepted for reading in shape_readings(mol, species, cid).values())


def stereo_donor_bonds(mol, species):
    """Return the donor-metal bonds of donors with a specified or tagged hand, which ETKDG must see to hold it."""
    points = species.stereo.points
    return [
        (donor, metal)
        for donor, metal in donor_bonds(species.mol)
        if (donor in points or mol.GetAtomWithIdx(donor).GetChiralTag() in TETRAHEDRAL_TAGS)
    ]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Seeds:
    """Hold seeded conformers on their working Mol, and whether stereo selection chose them."""

    mol: Chem.Mol
    ids: list
    selected: bool  # chosen for a specified hand, face or ligand stereo, so none means no seed has it

    def require(self, species, cons):
        """Raise `EmbeddingError` when no seed of `species` was kept: none had its stereo, or none met `cons`."""
        if self.ids:
            return
        if self.selected:
            raise EmbeddingError(
                f"{str(species) or 'embed'}: no DG seed has the requested metal and ligand stereo; try another isomer",
            )
        reason = f"no conformer satisfied the constraints; {remedy('seeding', species, cons)}"
        raise EmbeddingError(f"{str(species) or 'embed'}: {reason}")


def _terminal_sp_bonds(mol, species):
    """Return the metal bonds of terminal sp donors, which ETKDG holds linear only across bonds it sees."""
    sites = Counter(donor for donor, _metal in donor_bonds(species.mol))
    terminal = [
        (donor, metal)
        for donor, metal in donor_bonds(species.mol)
        if sites[donor] == 1
        and ligand_degree(atom := mol.GetAtomWithIdx(donor)) == 1
        and not any(nb.GetIdx() in sites for nb in atom.GetNeighbors())
    ]
    hyb = stripped_hybridisation(mol) if terminal else {}
    return [(donor, metal) for donor, metal in terminal if hyb.get(donor) == Chem.HybridizationType.SP]


def seed_conformers(mol, cons, species, n, sampling, graft_ref=None):
    """Return `Seeds` of `n` conformers with metal and donor hands selected and a fixed core grafted.

    DG draws may replace `mol`'s conformers; callers that keep them pass a copy. A species with metal centres or held
    stereo draws batches until `n` seeds agree with it or their mirror images do, those that already take each
    polyhedron first, within a draw budget that grows with the hands and faces a seed can land on unless a reflection
    can fix the one hand. When too few agree, seeds whose only miss is their raw best-fit seating fill in: relaxation
    can still seat them, and the coordination_state gate judges the result.
    """
    specification = Specification(species=species)
    keep = n or seed_count(mol, constrained=cons.is_constrained)
    # Capture the draw budget and frozen core before the surrogate carrier changes the graph or coordinates.
    hands = sum(bool(state.hand) for state, _sites in specification.centres)
    faces = math.prod(
        len(face_orientations(mol, sites.haptic[dummy], specification.face_classes[state.atom]))
        for state, sites in specification.centres
        for dummy in sites.winding
    )
    trials = (
        keep if reflection_is_free(specification, cons) else 2**hands * faces * (keep + 2 * (math.isqrt(keep - 1) + 1))
    )
    # a seed's mirror image is as good a draw, unless a frozen core or a signed dihedral tells the two apart
    mirror = not (cons.frozen or cons.dihedrals)
    draw = max(keep, min(MAX_SEED_COUNT, max(2 * trials, seed_count(mol, constrained=True))))
    graft = dict(graft_ref or {})  # a frozen atom grafts at its `graft_ref` coordinate, else at its own
    if cons.frozen and mol.GetNumConformers():
        own = mol.GetConformer().GetPositions()
        graft = {i: graft.get(i, own[i]) for i in cons.frozen}
    stereo_bonds = stereo_donor_bonds(mol, species)
    if carried := [*stereo_bonds, *_terminal_sp_bonds(mol, species)]:
        mol = seed_graph(species.mol, mol, carried)
        for donor, _metal in stereo_bonds:
            mol.GetAtomWithIdx(donor).SetChiralTag(specification.mol.GetAtomWithIdx(donor).GetChiralTag())
    if not (specification.centres or species.held_stereo):
        ids = seed_coordinates(mol, cons, keep, sampling)
        graft_frozen(mol, ids, graft)  # restore the exact frozen core
        return Seeds(mol=surrogate_all_metals(mol) if carried else mol, ids=ids, selected=False)

    kept = Chem.Mol(mol)
    kept.RemoveAllConformers()
    unseated = Chem.Mol(kept)
    has_point_stereo = bool(species.held_stereo.points)
    # RDKit's chirality check also tests sphere windows only the relax meets, so ligand point stereo skips it
    trial_params = replace(sampling.params, prune_rms=-1) if len(specification.centres) > 1 else sampling.params
    used = attempt = generated = 0
    while kept.GetNumConformers() < keep and used < draw:
        need = keep - kept.GetNumConformers()
        batch_n = min(REPLACEMENT_FACTOR * need, REPLACEMENT_FACTOR * keep, draw - used)
        trial = replace(sampling.retry("seed batch", attempt), params=trial_params)
        ids = seed_coordinates(mol, cons, batch_n, trial, enforce_chirality=not has_point_stereo)
        used += batch_n
        attempt += 1
        generated += len(ids)
        graft_frozen(mol, ids, graft)
        if mirror:
            reflect_to_specification(specification, mol, ids)
        matches = []
        for cid in ids:
            kinds = {miss.kind for miss in conformer_misses(specification, mol, cid)}
            if not kinds:
                matches.append(cid)
            elif kinds == {"seating"} and unseated.GetNumConformers() < keep:
                unseated.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
        # geometry-matching seeds first, testing only as many as `need`
        good, rest = [], []
        for cid in matches:
            if len(good) < need and seed_takes_shape(mol, cid, species):
                good.append(cid)
            else:
                rest.append(cid)
        for cid in good + rest:
            kept.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True)
            if kept.GetNumConformers() == keep:
                break
    logger.debug(
        "DG search returned %d candidates; %d/%d satisfied requested stereo selection",
        generated,
        kept.GetNumConformers(),
        keep,
    )
    for conf in list(unseated.GetConformers())[: keep - kept.GetNumConformers()]:
        kept.AddConformer(Chem.Conformer(conf), assignId=True)
    ids = [conf.GetId() for conf in kept.GetConformers()]
    # UFF still receives the bondless surrogate graph
    return Seeds(mol=surrogate_all_metals(kept) if carried else kept, ids=ids, selected=True)


def reflection_is_free(specification, cons):
    """Return whether reflecting a conformer would invert one metal centre's hand and nothing else.

    That needs one handed centre without a haptic face; no other point or axial stereo and no chiral tag, since
    stripping metal bonds can hide a donor stereocentre; no stereo-carrier donor bond; and no frozen core or dihedral
    hold.
    """
    if len(specification.centres) != 1 or cons.frozen or cons.dihedrals:
        return False
    [(state, sites)], mol = specification.centres, engine_graph(specification.species.mol)
    return bool(
        state.hand
        and not sites.haptic
        and not stereo_donor_bonds(mol, specification.species)
        and all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
        and all(element.type == Chem.StereoType.Bond_Double for element in Chem.FindPotentialStereo(mol))
    )


def reflect(mol, cid):
    """Reflect conformer `cid` in place without changing its mirror-symmetric restraint energy."""
    conf = mol.GetConformer(int(cid))
    pos = conf.GetPositions()
    pos[:, 0] *= -1.0
    conf.SetPositions(pos)


def reflect_to_specification(specification, mol, ids):
    """Reflect each conformer in `ids` whose mirror image shows all `specification` gives; return the reflected ids.

    The mirror inverts every hand, face, point and axis and keeps each seating and E/Z bond, so only a conformer that
    misses each specified centre (one miss apiece) and ligand point and axis, and nothing else, can show the
    specification reflected. Every gate judges these through `conformer_misses`, which judges the mirror again; a
    mirror that misses too is put back. Use only where no frozen core or signed dihedral tells a mirror image apart.
    """
    invertible = len(specification.centres) + sum(e.kind != "ez" for e in specification.species.stereo.elements)
    reflected = []
    for cid in ids:
        misses = conformer_misses(specification, mol, cid)
        if not misses or len(misses) != invertible or any(miss.kind in ("seating", "ez") for miss in misses):
            continue
        reflect(mol, cid)
        if conformer_misses(specification, mol, cid):
            reflect(mol, cid)
        else:
            reflected.append(cid)
    return reflected


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Prepared:
    """Hold a species' working Mol, its compiled constraints and the coordinates `fix` grafts (atom to xyz)."""

    mol: Chem.Mol
    cons: Constraints
    graft: dict


def seeded(species, prepared, n, sampling, kind):
    """Return `n` raw DG seeds of `species` under its `Prepared` constraints as a `kind`; keeping none raises."""
    seeds = seed_conformers(prepared.mol, prepared.cons, species, n, sampling, graft_ref=prepared.graft)
    seeds.require(species, prepared.cons)
    return kind(seeds.mol, list(seeds.ids), prepared.cons, species=species, sampling=sampling)


def prepare(species, pins, *, contacts=None, params=None):
    """Return the working graph and constraints of `species` for embedding or relaxation.

    `species` has a polyhedron and full vertex seating for every bound metal, as `read_species` or enumeration give it.
    `pins` is the verb's ``fix`` and ``constrain``, resolved once. Each soft term in `contacts` is kept unless a
    structural term owns its degree of freedom, and the metal model is compiled again around the kept ones, so a dropped
    contact changes no model preference.
    """
    if pinned := misplaced_pins(species, pins.graft):  # a graft off the specified vertices would replace it
        raise ValueError(
            f"the graft pins sphere atoms {pinned} off this species' vertices; embed the species they show "
            "(observed=True)"
        )
    if flipped := _pinned_otherwise(species, pins.graft):
        raise ValueError(
            f"the coordinate fix pins stereocentre(s) {flipped} in the other hand; unpin it or two neighbours, "
            "or ask for that hand"
        )
    params = EmbedParams() if params is None else params
    facts = LigandFacts(species.mol)
    mol = facts.mol
    ions = ionic_contacts(mol)
    cons = compose(ions, compile_constraints(species, pins.cons, None, params=params, facts=facts))
    if contacts:
        soft = resolve_core(mol, constrain=contacts).cons
        active = retained_soft(cons, soft)
        if active.is_constrained:  # `active` exposes the kept contacts to the structural terms
            cons = compose(ions, compile_constraints(species, pins.cons, active, params=params, facts=facts))
            active = retained_soft(cons, soft)
        cons = compose(cons, active)
    return Prepared(mol=mol, cons=cons, graft=pins.graft)


def _pinned_otherwise(species, graft):
    """Return the specified point centres `graft` pins, with three neighbours, in the opposite hand to the species'.

    A free fourth neighbour is put opposite the pinned three, where an embedding puts it, and each hand is read from its
    own positions with dative bonds read as single, so a pseudoasymmetric centre is judged without its partners.
    """
    if not graft.keys() & species.stereo.points.keys():
        return []
    graph = graph_with_stereo(species)
    pos = np.zeros((graph.GetNumAtoms(), 3))
    for atom, xyz in graft.items():
        pos[int(atom)] = xyz
    pinned = []
    for centre in sorted(graft.keys() & species.stereo.points.keys()):
        neighbours = {n.GetIdx() for n in graph.GetAtomWithIdx(centre).GetNeighbors()}
        if len(held := sorted(neighbours & graft.keys())) >= 3:  # noqa: PLR2004 - three neighbours fix a hand
            pos[sorted(neighbours - graft.keys())] = 2 * pos[centre] - pos[held].mean(axis=0)
            pinned.append(centre)
    pinned_mol = Chem.RWMol(with_conformers(graph, graph, ()))
    for bond in pinned_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
    pinned_mol.AddConformer(Chem.Conformer(pinned_mol.GetNumAtoms()), assignId=True)
    pinned_mol.GetConformer().SetPositions(pos)
    with tetrahedral_perception():
        Chem.AssignAtomChiralTagsFromStructure(pinned_mol, replaceExistingTags=True)
    specified_tags = {centre: graph.GetAtomWithIdx(centre).GetChiralTag() for centre in pinned}
    return [
        c for c, tag in specified_tags.items() if tag != mirror_tag(tag) == pinned_mol.GetAtomWithIdx(c).GetChiralTag()
    ]


def relaxed(species, pins, max_iters, kind):
    """Return the geometry of `species` relaxed under `pins` as a `kind`, its fixed core grafted first.

    `max_iters` caps the restrained UFF; a result that leaves the species fails the acceptance gates, which raise.
    """
    prepared = prepare(species, pins)
    mol, ref = prepared.mol, prepared.graft
    ids = [c.GetId() for c in mol.GetConformers()]
    graft_frozen(mol, ids, ref)
    return kind(mol, ids, prepared.cons, species=species).minimize(max_iters=max_iters)
